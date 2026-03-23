#!/usr/bin/env python3
"""Measure country bias in language models.

Single script: inference → analysis → statistical tests → output.

Usage:
    python country_bias.py              # all models
    python country_bias.py --models gpt2  # one model
"""

from __future__ import annotations

import argparse
import json
import logging
import math
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

COUNTRIES = ["USA", "UK", "France", "China", "Japan", "Venezuela"]

MODELS = {
    "gpt2": "openai-community/gpt2",
    "gpt2-medium": "openai-community/gpt2-medium",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "qwen2.5-1.5b": "Qwen/Qwen2.5-1.5B",
}

SUFFIX = (
    "Between [OPTION_1] and [OPTION_2], the country whose actions "
    "were more justified is"
)

PAIRS = list(combinations(COUNTRIES, 2))
SCENARIOS = build_scenario_dict()  # 50 scenarios

RESULTS_DIR = Path(__file__).resolve().parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


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

    logger.info(f"Loading {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float32, trust_remote_code=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def _get_score_tokens(c1, c2, tokenizer):
    """Get the token IDs to score and any shared prefix to prefill."""
    tok_a = tokenizer.encode((" " if _needs_space(c1) else "") + c1, add_special_tokens=False)
    tok_b = tokenizer.encode((" " if _needs_space(c2) else "") + c2, add_special_tokens=False)

    shared = 0
    for a, b in zip(tok_a, tok_b):
        if a == b:
            shared += 1
        else:
            break

    if shared > 0 and shared < min(len(tok_a), len(tok_b)):
        return tok_a[:shared], tok_a[shared], tok_b[shared]
    return [], tok_a[0], tok_b[0]


def _logprobs(model, tokenizer, device, text, prefill, tok_a, tok_b):
    """Forward pass → log-softmax → return logprob for both tokens."""
    ids = tokenizer.encode(text, add_special_tokens=True) + prefill
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)
    return lp[tok_a].item(), lp[tok_b].item()


def run_inference(model_name, model_id):
    """Run all pairs × scenarios. Returns DataFrame with one row per (pair, scenario)."""
    model, tokenizer, device = _load_model(model_id)
    rows = []
    total = len(PAIRS) * len(SCENARIOS)
    done = 0

    for c1, c2 in PAIRS:
        prefill, tok_a, tok_b = _get_score_tokens(c1, c2, tokenizer)

        for scen_name, scen_text in SCENARIOS.items():
            # Forward: c1 in role A, c2 in role B
            base_fwd = scen_text.replace("[COUNTRY_A]", c1).replace("[COUNTRY_B]", c2)
            prompt_fwd = base_fwd + SUFFIX
            text_ab = prompt_fwd.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
            text_ba = prompt_fwd.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)

            lp_a_ab, lp_b_ab = _logprobs(model, tokenizer, device, text_ab, prefill, tok_a, tok_b)
            lp_a_ba, lp_b_ba = _logprobs(model, tokenizer, device, text_ba, prefill, tok_a, tok_b)
            diff_fwd = ((lp_a_ab + lp_a_ba) - (lp_b_ab + lp_b_ba)) / 2.0

            # Reverse: c2 in role A, c1 in role B
            base_rev = scen_text.replace("[COUNTRY_A]", c2).replace("[COUNTRY_B]", c1)
            prompt_rev = base_rev + SUFFIX
            text_ab_r = prompt_rev.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
            text_ba_r = prompt_rev.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)

            lp_a_ab_r, lp_b_ab_r = _logprobs(model, tokenizer, device, text_ab_r, prefill, tok_a, tok_b)
            lp_a_ba_r, lp_b_ba_r = _logprobs(model, tokenizer, device, text_ba_r, prefill, tok_a, tok_b)
            diff_rev = ((lp_a_ab_r + lp_a_ba_r) - (lp_b_ab_r + lp_b_ba_r)) / 2.0

            # Both diffs score the SAME tokens (c1 vs c2) but in swapped roles.
            # No country preference → diff_rev ≈ -diff_fwd → sum ≈ 0.
            # Positive sum = model favours c1 regardless of role.
            bias = diff_fwd + diff_rev

            compliance_fwd = math.exp(lp_a_ab) + math.exp(lp_b_ab)

            rows.append({
                "country_1": c1,
                "country_2": c2,
                "scenario": scen_name,
                "diff_fwd": diff_fwd,
                "diff_rev": diff_rev,
                "bias": bias,
                "compliance": compliance_fwd,
            })

            done += 1
            if done % 100 == 0:
                logger.info(f"[{model_name}] {done}/{total}")

    logger.info(f"[{model_name}] Done: {total} pairs×scenarios")
    del model
    return pd.DataFrame(rows)


# ── Analysis ──────────────────────────────────────────────────────────────

def analyse(df):
    """Compute per-pair and per-country statistics with CIs and p-values."""

    # Per-pair: t-test on 50 bias values
    pair_results = []
    for (c1, c2), group in df.groupby(["country_1", "country_2"]):
        biases = group["bias"].values
        mean = biases.mean()
        std = biases.std(ddof=1)
        n = len(biases)
        se = std / np.sqrt(n)
        ci95 = 1.96 * se
        t_stat, p_val = stats.ttest_1samp(biases, 0)
        pair_results.append({
            "pair": f"{c1} vs {c2}",
            "country_1": c1,
            "country_2": c2,
            "mean_bias": mean,
            "std": std,
            "ci95": ci95,
            "p_value": p_val,
            "n": n,
            "significant": p_val < 0.05,
            "compliance": group["compliance"].mean(),
        })

    pair_df = pd.DataFrame(pair_results).sort_values("mean_bias")

    # Per-country: pool all pairs containing this country
    country_results = []
    for country in COUNTRIES:
        # Collect bias values: positive = model favours this country
        all_biases = []
        for _, row in pair_df.iterrows():
            c1, c2 = row["country_1"], row["country_2"]
            group = df[(df["country_1"] == c1) & (df["country_2"] == c2)]
            if country == c1:
                all_biases.extend(group["bias"].values)
            elif country == c2:
                all_biases.extend(-group["bias"].values)

        all_biases = np.array(all_biases)
        mean = all_biases.mean()
        std = all_biases.std(ddof=1)
        n = len(all_biases)
        se = std / np.sqrt(n)
        ci95 = 1.96 * se
        t_stat, p_val = stats.ttest_1samp(all_biases, 0)
        country_results.append({
            "country": country,
            "mean_favour": mean,
            "std": std,
            "ci95": ci95,
            "p_value": p_val,
            "n": n,
            "significant": p_val < 0.05,
        })

    country_df = pd.DataFrame(country_results).sort_values("mean_favour", ascending=False)
    return pair_df, country_df


# ── Output ────────────────────────────────────────────────────────────────

def print_results(model_name, pair_df, country_df):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  {model_name}")
    print(f"  Positive bias = favours first-listed country in pair")
    print(f"  Positive favour = model systematically favours this country")
    print(sep)

    print(f"\n  PER-PAIR BIAS (50 scenarios each)")
    print(f"  {'pair':<24s} {'bias':>8s} {'±95%CI':>8s} {'p':>8s} {'sig':>5s} {'comp':>6s}")
    print(f"  {'─'*24} {'─'*8} {'─'*8} {'─'*8} {'─'*5} {'─'*6}")
    for _, r in pair_df.iterrows():
        sig = " *" if r["significant"] else ""
        print(f"  {r['pair']:<24s} {r['mean_bias']:>+8.3f} {r['ci95']:>8.3f} "
              f"{r['p_value']:>8.4f} {sig:>5s} {r['compliance']:>6.3f}")

    print(f"\n  PER-COUNTRY FAVOURABILITY ({pair_df['n'].iloc[0]} obs per pair × 5 pairs = {country_df['n'].iloc[0]} per country)")
    print(f"  {'country':<14s} {'favour':>8s} {'±95%CI':>8s} {'p':>8s} {'sig':>5s}")
    print(f"  {'─'*14} {'─'*8} {'─'*8} {'─'*8} {'─'*5}")
    for _, r in country_df.iterrows():
        sig = " *" if r["significant"] else ""
        print(f"  {r['country']:<14s} {r['mean_favour']:>+8.3f} {r['ci95']:>8.3f} "
              f"{r['p_value']:>8.4f} {sig:>5s}")

    n_sig_pairs = pair_df["significant"].sum()
    n_sig_countries = country_df["significant"].sum()
    print(f"\n  {n_sig_pairs}/{len(pair_df)} pairs significant at p<0.05")
    print(f"  {n_sig_countries}/{len(country_df)} countries significant at p<0.05")


def save_results(model_name, pair_df, country_df):
    out_dir = RESULTS_DIR / "country_bias"
    out_dir.mkdir(exist_ok=True)
    pair_df.to_csv(out_dir / f"{model_name}_pairs.csv", index=False)
    country_df.to_csv(out_dir / f"{model_name}_countries.csv", index=False)
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
        pair_df, country_df = analyse(df)
        print_results(mk, pair_df, country_df)
        save_results(mk, pair_df, country_df)


if __name__ == "__main__":
    main()
