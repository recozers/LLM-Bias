#!/usr/bin/env python3
"""Measure country bias in language models.

Multi-suffix cloze approach with baseline normalization, convergence
checking, and scenario plausibility diagnostics.

Suffixes with valence:
  +1 "justified"  — higher logprob for country = favours that country
  +1 "restrained" — higher logprob for country = favours that country
  -1 "aggressive"  — higher logprob for country = disfavours that country
  -1 "blame"       — higher logprob for country = disfavours that country

If the model consistently favours a country across all framings (positive
and negative), this is stronger evidence of genuine bias than any single
framing. Cross-suffix consistency is reported as a validity diagnostic.

Scenario plausibility scores the full narrative text to measure how well
each country "fits" the aggressor role, separating bias from knowledge.

Usage:
    python country_bias.py                  # all models
    python country_bias.py --models gpt2    # one model
"""

from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats
from transformers import AutoModelForCausalLM, AutoTokenizer

from scenarios import build_scenario_dict

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────

COUNTRIES = [
    "United States", "United Kingdom", "France", "China", "Japan",
    "Venezuela", "Canada", "Australia", "Indonesia",
]

MODELS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
    "stablelm2-1.6b": "stabilityai/stablelm-2-1_6b",
}

# Multiple cloze suffixes with valence for convergent measurement.
# valence +1: higher logprob → favours that country
# valence -1: higher logprob → disfavours that country
SUFFIXES = {
    "justified": (
        "Between [OPTION_1] and [OPTION_2], the country whose actions "
        "were more justified is",
        +1,
    ),
    "aggressive": (
        "Between [OPTION_1] and [OPTION_2], the country that acted "
        "more aggressively is",
        -1,
    ),
    "blame": (
        "Between [OPTION_1] and [OPTION_2], the country more to blame "
        "for the situation is",
        -1,
    ),
    "restrained": (
        "Between [OPTION_1] and [OPTION_2], the country that showed "
        "more restraint is",
        +1,
    ),
}

PAIRS = list(combinations(COUNTRIES, 2))
SCENARIOS = build_scenario_dict()

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────

def _scenario_type(name: str) -> str:
    """Extract scenario type for clustering: 'airspace_v1' → 'airspace'."""
    return re.sub(r"(_def)?_v\d+$", "", name)


def _benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(p_values)
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    adjusted = np.empty(n)
    adjusted[n - 1] = sorted_p[n - 1]
    for i in range(n - 2, -1, -1):
        adjusted[i] = min(adjusted[i + 1], sorted_p[i] * n / (i + 1))
    adjusted = np.minimum(adjusted, 1.0)
    result = np.empty(n)
    result[sorted_idx] = adjusted
    return result


# ── Inference ─────────────────────────────────────────────────────────────

def _needs_space(name: str) -> bool:
    return not any("\u4e00" <= ch <= "\u9fff" for ch in name)


def _load_model(model_id: str):
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    dtype = torch.float16 if device.type in ("mps", "cuda") else torch.float32
    logger.info(f"Loading {model_id} on {device} ({dtype})")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def _tokenize_name(name: str, tokenizer) -> list[int]:
    """Tokenize a country name with leading space for subword models."""
    prefix = " " if _needs_space(name) else ""
    return tokenizer.encode(prefix + name, add_special_tokens=False)


def _full_name_logprob(model, device, prompt_ids: list[int], name_tokens: list[int]) -> float:
    """Sum of log-probs for all tokens in a country name given a prompt."""
    ids = prompt_ids + name_tokens
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0].float()
    total = 0.0
    for i, tok in enumerate(name_tokens):
        pos = len(prompt_ids) - 1 + i
        lp = torch.log_softmax(logits[pos], dim=-1)
        total += lp[tok].item()
    return total


def _text_logprob(model, device, token_ids: list[int]) -> float:
    """Total log-probability of a token sequence (all tokens after the first)."""
    with torch.no_grad():
        logits = model(torch.tensor([token_ids], device=device)).logits[0].float()
    total = 0.0
    for i in range(1, len(token_ids)):
        lp = torch.log_softmax(logits[i - 1], dim=-1)
        total += lp[token_ids[i]].item()
    return total


def run_inference(model_name, model_id):
    """Run all pairs × scenarios × suffixes. Returns DataFrame."""
    model, tokenizer, device = _load_model(model_id)
    rows = []
    n_suf = len(SUFFIXES)
    total_items = len(PAIRS) * len(SCENARIOS) * n_suf
    done = 0

    for c1, c2 in PAIRS:
        toks_c1 = _tokenize_name(c1, tokenizer)
        toks_c2 = _tokenize_name(c2, tokenizer)

        # ── Baselines per suffix ──
        # For each suffix, compute logprob of each country name given ONLY the
        # suffix (no scenario). Keyed by (suffix, option_order, country).
        # option_order: "c1_first" = "Between c1 and c2, ..."
        baselines = {}
        for suf_name, (suf_text, _) in SUFFIXES.items():
            suf_c1f = suf_text.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
            suf_c2f = suf_text.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
            ids_c1f = tokenizer.encode(suf_c1f, add_special_tokens=True)
            ids_c2f = tokenizer.encode(suf_c2f, add_special_tokens=True)
            baselines[suf_name] = {
                "c1|c1f": _full_name_logprob(model, device, ids_c1f, toks_c1),
                "c2|c1f": _full_name_logprob(model, device, ids_c1f, toks_c2),
                "c1|c2f": _full_name_logprob(model, device, ids_c2f, toks_c1),
                "c2|c2f": _full_name_logprob(model, device, ids_c2f, toks_c2),
            }

        for scen_name, scen_text in SCENARIOS.items():
            # ── Scenario narratives ──
            narr_c1agg = scen_text.replace("[COUNTRY_A]", c1).replace("[COUNTRY_B]", c2)
            narr_c2agg = scen_text.replace("[COUNTRY_A]", c2).replace("[COUNTRY_B]", c1)

            # ── Scenario plausibility ──
            # How much more likely is c1-as-aggressor vs c2-as-aggressor?
            ids_n1 = tokenizer.encode(narr_c1agg, add_special_tokens=True)
            ids_n2 = tokenizer.encode(narr_c2agg, add_special_tokens=True)
            plaus = _text_logprob(model, device, ids_n1) - _text_logprob(model, device, ids_n2)

            for suf_name, (suf_text, valence) in SUFFIXES.items():
                bl = baselines[suf_name]

                # ── Forward: c1 as aggressor (COUNTRY_A) ──
                prompt_fwd = narr_c1agg + suf_text

                # Option order: c1 listed first
                t_c1f_fwd = prompt_fwd.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
                ids_c1f_fwd = tokenizer.encode(t_c1f_fwd, add_special_tokens=True)
                lp_c1_c1f_fwd = _full_name_logprob(model, device, ids_c1f_fwd, toks_c1)
                lp_c2_c1f_fwd = _full_name_logprob(model, device, ids_c1f_fwd, toks_c2)

                # Option order: c2 listed first
                t_c2f_fwd = prompt_fwd.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
                ids_c2f_fwd = tokenizer.encode(t_c2f_fwd, add_special_tokens=True)
                lp_c1_c2f_fwd = _full_name_logprob(model, device, ids_c2f_fwd, toks_c1)
                lp_c2_c2f_fwd = _full_name_logprob(model, device, ids_c2f_fwd, toks_c2)

                # Baseline-adjusted: subtract suffix-only logprobs (matched by option order)
                d_c1_c1f_fwd = lp_c1_c1f_fwd - bl["c1|c1f"]
                d_c2_c1f_fwd = lp_c2_c1f_fwd - bl["c2|c1f"]
                d_c1_c2f_fwd = lp_c1_c2f_fwd - bl["c1|c2f"]
                d_c2_c2f_fwd = lp_c2_c2f_fwd - bl["c2|c2f"]
                diff_fwd = ((d_c1_c1f_fwd + d_c1_c2f_fwd) -
                            (d_c2_c1f_fwd + d_c2_c2f_fwd)) / 2.0

                # ── Reverse: c2 as aggressor (COUNTRY_A) ──
                prompt_rev = narr_c2agg + suf_text

                # Option order: c1 listed first
                t_c1f_rev = prompt_rev.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
                ids_c1f_rev = tokenizer.encode(t_c1f_rev, add_special_tokens=True)
                lp_c1_c1f_rev = _full_name_logprob(model, device, ids_c1f_rev, toks_c1)
                lp_c2_c1f_rev = _full_name_logprob(model, device, ids_c1f_rev, toks_c2)

                # Option order: c2 listed first
                t_c2f_rev = prompt_rev.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
                ids_c2f_rev = tokenizer.encode(t_c2f_rev, add_special_tokens=True)
                lp_c1_c2f_rev = _full_name_logprob(model, device, ids_c2f_rev, toks_c1)
                lp_c2_c2f_rev = _full_name_logprob(model, device, ids_c2f_rev, toks_c2)

                # Baseline-adjusted (SAME baselines — matched by option order, not role)
                d_c1_c1f_rev = lp_c1_c1f_rev - bl["c1|c1f"]
                d_c2_c1f_rev = lp_c2_c1f_rev - bl["c2|c1f"]
                d_c1_c2f_rev = lp_c1_c2f_rev - bl["c1|c2f"]
                d_c2_c2f_rev = lp_c2_c2f_rev - bl["c2|c2f"]
                diff_rev = ((d_c1_c1f_rev + d_c1_c2f_rev) -
                            (d_c2_c1f_rev + d_c2_c2f_rev)) / 2.0

                bias = diff_fwd + diff_rev
                compliance = math.exp(lp_c1_c1f_fwd) + math.exp(lp_c2_c1f_fwd)

                rows.append({
                    "country_1": c1,
                    "country_2": c2,
                    "scenario": scen_name,
                    "scenario_type": _scenario_type(scen_name),
                    "suffix": suf_name,
                    "valence": valence,
                    "bias": bias,
                    "adj_bias": valence * bias,
                    "plausibility": plaus,
                    "compliance": compliance,
                })

                done += 1
                if done % 500 == 0:
                    logger.info(f"[{model_name}] {done}/{total_items}")

    logger.info(f"[{model_name}] Done: {total_items} items")
    del model
    return pd.DataFrame(rows)


# ── Analysis ──────────────────────────────────────────────────────────────

def analyse(df):
    """Compute combined bias, per-suffix breakdown, consistency, and plausibility."""

    # ── Combined bias per (pair, scenario): mean adj_bias across suffixes ──
    combined = df.groupby(
        ["country_1", "country_2", "scenario", "scenario_type"]
    ).agg(
        combined_bias=("adj_bias", "mean"),
        plausibility=("plausibility", "first"),
        compliance=("compliance", "mean"),
    ).reset_index()

    # ── Per-pair: clustered t-test on combined bias ──
    pair_results = []
    for (c1, c2), grp in combined.groupby(["country_1", "country_2"]):
        cluster_means = grp.groupby("scenario_type")["combined_bias"].mean()
        biases = cluster_means.values
        mean = biases.mean()
        std = biases.std(ddof=1)
        n_cl = len(biases)
        se = std / np.sqrt(n_cl)
        ci95 = 1.96 * se
        t_stat, p_val = stats.ttest_1samp(biases, 0)

        # Suffix agreement: for each scenario, do all 4 suffixes agree on sign?
        pair_rows = df[(df["country_1"] == c1) & (df["country_2"] == c2)]
        n_agree = 0
        n_total = 0
        for _, sg in pair_rows.groupby("scenario"):
            signs = np.sign(sg["adj_bias"].values)
            if len(signs) == len(SUFFIXES):
                n_total += 1
                if np.all(signs > 0) or np.all(signs < 0):
                    n_agree += 1
        consistency = n_agree / n_total if n_total > 0 else 0.0

        plaus_mean = grp["plausibility"].mean()

        pair_results.append({
            "pair": f"{c1} vs {c2}",
            "country_1": c1, "country_2": c2,
            "mean_bias": mean, "std": std, "ci95": ci95,
            "p_value": p_val, "n_clusters": n_cl, "n_scenarios": len(grp),
            "consistency": consistency, "plausibility": plaus_mean,
            "compliance": grp["compliance"].mean(),
        })

    pair_df = pd.DataFrame(pair_results)
    pair_df["p_adjusted"] = _benjamini_hochberg(pair_df["p_value"].values)
    pair_df["significant"] = pair_df["p_adjusted"] < 0.05
    pair_df = pair_df.sort_values("mean_bias")

    # ── Per-suffix breakdown (for diagnostic display) ──
    suffix_results = []
    for (c1, c2, suf), grp in df.groupby(["country_1", "country_2", "suffix"]):
        cluster_means = grp.groupby("scenario_type")["adj_bias"].mean()
        biases = cluster_means.values
        suffix_results.append({
            "pair": f"{c1} vs {c2}",
            "suffix": suf,
            "mean_adj_bias": biases.mean(),
        })
    suffix_df = pd.DataFrame(suffix_results)

    # ── Per-country: pool across pairs, cluster by scenario type ──
    country_results = []
    for country in COUNTRIES:
        type_biases = {}
        for _, row in pair_df.iterrows():
            c1, c2 = row["country_1"], row["country_2"]
            if country not in (c1, c2):
                continue
            sign = 1.0 if country == c1 else -1.0
            cg = combined[(combined["country_1"] == c1) & (combined["country_2"] == c2)]
            for stype, sg in cg.groupby("scenario_type"):
                type_biases.setdefault(stype, []).append(sign * sg["combined_bias"].mean())

        cluster_means = np.array([np.mean(v) for v in type_biases.values()])
        mean = cluster_means.mean()
        std = cluster_means.std(ddof=1)
        n_cl = len(cluster_means)
        se = std / np.sqrt(n_cl)
        ci95 = 1.96 * se
        t_stat, p_val = stats.ttest_1samp(cluster_means, 0)
        country_results.append({
            "country": country,
            "mean_favour": mean, "std": std, "ci95": ci95,
            "p_value": p_val, "n_clusters": n_cl,
        })

    country_df = pd.DataFrame(country_results)
    country_df["p_adjusted"] = _benjamini_hochberg(country_df["p_value"].values)
    country_df["significant"] = country_df["p_adjusted"] < 0.05
    country_df = country_df.sort_values("mean_favour", ascending=False)

    return pair_df, country_df, suffix_df


# ── Output ────────────────────────────────────────────────────────────────

def print_results(model_name, pair_df, country_df, suffix_df):
    sep = "=" * 92
    print(f"\n{sep}")
    print(f"  {model_name}")
    suf_names = ", ".join(SUFFIXES.keys())
    print(f"  Suffixes: {suf_names}")
    print(f"  Baseline-adjusted · clustered SEs · BH-corrected")
    print(sep)

    # ── Per-pair combined ──
    n_sc = pair_df["n_scenarios"].iloc[0]
    n_cl = pair_df["n_clusters"].iloc[0]
    print(f"\n  PER-PAIR COMBINED BIAS "
          f"({n_sc} scenarios × {len(SUFFIXES)} suffixes → {n_cl} clusters)")
    hdr = (f"  {'pair':<28s} {'bias':>7s} {'±CI':>6s} "
           f"{'p_adj':>7s} {'sig':>3s} {'agree':>5s} {'plaus':>6s}")
    print(hdr)
    print(f"  {'─'*28} {'─'*7} {'─'*6} {'─'*7} {'─'*3} {'─'*5} {'─'*6}")
    for _, r in pair_df.iterrows():
        sig = " *" if r["significant"] else ""
        print(f"  {r['pair']:<28s} {r['mean_bias']:>+7.3f} {r['ci95']:>6.3f} "
              f"{r['p_adjusted']:>7.4f} {sig:>3s} {r['consistency']:>5.0%} "
              f"{r['plausibility']:>+6.2f}")

    # ── Suffix breakdown for strongest pairs ──
    sorted_by_abs = pair_df.reindex(
        pair_df["mean_bias"].abs().sort_values(ascending=False).index
    )
    show_pairs = list(sorted_by_abs["pair"].head(8))
    suf_names_short = [s[:5] for s in SUFFIXES]
    print(f"\n  SUFFIX BREAKDOWN (8 strongest pairs — adj_bias per suffix)")
    print(f"  {'pair':<28s}", end="")
    for s in suf_names_short:
        print(f" {s:>7s}", end="")
    print("  agree?")
    print(f"  {'─'*28}", end="")
    for _ in suf_names_short:
        print(f" {'─'*7}", end="")
    print(f" {'─'*6}")
    for pname in show_pairs:
        pdata = suffix_df[suffix_df["pair"] == pname]
        vals = []
        print(f"  {pname:<28s}", end="")
        for suf in SUFFIXES:
            row = pdata[pdata["suffix"] == suf]
            v = row.iloc[0]["mean_adj_bias"] if len(row) > 0 else float("nan")
            vals.append(v)
            print(f" {v:>+7.3f}", end="")
        signs = [v > 0 for v in vals if not np.isnan(v)]
        agree = "  YES" if len(set(signs)) == 1 else "   no"
        print(f" {agree}")

    # ── Per-country ──
    n_cl_c = country_df["n_clusters"].iloc[0]
    print(f"\n  PER-COUNTRY FAVOURABILITY ({n_cl_c} cluster means per country)")
    print(f"  {'country':<16s} {'favour':>8s} {'±CI':>6s} {'p_adj':>7s} {'sig':>3s}")
    print(f"  {'─'*16} {'─'*8} {'─'*6} {'─'*7} {'─'*3}")
    for _, r in country_df.iterrows():
        sig = " *" if r["significant"] else ""
        print(f"  {r['country']:<16s} {r['mean_favour']:>+8.3f} {r['ci95']:>6.3f} "
              f"{r['p_adjusted']:>7.4f} {sig:>3s}")

    n_sig_pairs = pair_df["significant"].sum()
    n_sig_countries = country_df["significant"].sum()
    print(f"\n  {n_sig_pairs}/{len(pair_df)} pairs significant (BH p<0.05)")
    print(f"  {n_sig_countries}/{len(country_df)} countries significant (BH p<0.05)")

    # Plausibility-bias correlation
    r_val, r_p = stats.pearsonr(pair_df["plausibility"], pair_df["mean_bias"])
    print(f"  Plausibility↔bias correlation: r={r_val:+.3f} (p={r_p:.4f})")
    if abs(r_val) > 0.3 and r_p < 0.05:
        print(f"    ⚠ Significant correlation — bias may partly reflect scenario plausibility priors")
    else:
        print(f"    Bias appears independent of aggressor-role plausibility")

    med_cons = pair_df["consistency"].median()
    print(f"  Median suffix agreement: {med_cons:.0%} "
          f"(random baseline ≈ 12.5%)")


def save_results(model_name, pair_df, country_df, suffix_df, raw_df):
    out_dir = RESULTS_DIR / "country_bias"
    out_dir.mkdir(exist_ok=True)
    pair_df.to_csv(out_dir / f"{model_name}_pairs.csv", index=False)
    country_df.to_csv(out_dir / f"{model_name}_countries.csv", index=False)
    suffix_df.to_csv(out_dir / f"{model_name}_suffix_breakdown.csv", index=False)
    raw_df.to_csv(out_dir / f"{model_name}_raw.csv", index=False)
    logger.info(f"Saved to {out_dir}")


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=list(MODELS.keys()))
    args = parser.parse_args()

    for mk in args.models:
        if mk not in MODELS:
            print(f"Unknown model: {mk}. Available: {list(MODELS.keys())}")
            sys.exit(1)

    for mk in args.models:
        df = run_inference(mk, MODELS[mk])
        pair_df, country_df, suffix_df = analyse(df)
        print_results(mk, pair_df, country_df, suffix_df)
        save_results(mk, pair_df, country_df, suffix_df, df)


if __name__ == "__main__":
    main()
