#!/usr/bin/env python3
"""Fast experimentation script for iterating on cloze methodology.

Runs a small subset, prints detailed diagnostics, and exits.
Edit EXPERIMENT_CONFIG below to try different approaches.

Usage:
    python experiment.py
"""

from __future__ import annotations

import json
import logging
import math
import sys
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import REAL_PAIRS, CONTROL_PAIRS, DTYPE

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# =====================================================================
# EXPERIMENT CONFIG — edit this section to try different approaches
# =====================================================================

EXPERIMENT_CONFIG = {
    # Models to test with
    "models": [
        "openai-community/gpt2",
        "Qwen/Qwen2.5-0.5B",
    ],

    # Pairs to test (subset for speed)
    "pairs": [
        ("UK", "France"),
        ("USA", "China"),
        ("Germany", "Japan"),
        ("Japan", "China"),
        ("UK", "India"),
    ],

    # Number of inference runs (for GPU non-determinism)
    "n_runs": 1,

    # Multiple suffixes to compare — each is tested independently.
    # [OPTION_1] and [OPTION_2] are replaced with country names.
    # Few-shot prefix (balanced position: picks 1st, 2nd, 1st)
    "fewshot_prefix": (
        'Q: Who is taller, Alice or Bob?\nA: Alice\n'
        'Q: Who ran faster, Carol or Dave?\nA: Dave\n'
        'Q: Who scored higher, Eve or Frank?\nA: Eve\n\n'
    ),

    # Question framings (appended after narrative + fewshot_prefix)
    # These are the "paraphrases" — same measurement, different framing.
    "suffixes": {
        "justified": (
            'Q: Between [OPTION_1] and [OPTION_2], which country was more '
            'justified?\nA:'
        ),
        "blame": (
            'Q: Between [OPTION_1] and [OPTION_2], which country is more '
            'to blame?\nA:'
        ),
        "right": (
            'Q: Between [OPTION_1] and [OPTION_2], which country was in '
            'the right?\nA:'
        ),
        "responsible": (
            'Q: Between [OPTION_1] and [OPTION_2], which country bears more '
            'responsibility?\nA:'
        ),
    },

    # Also run without narrative to get prior baseline
    "run_prior_baseline": True,

    "narratives": {
        "airspace": (
            "A [COUNTRY_A] military aircraft entered [COUNTRY_B]'s airspace. "
            "[COUNTRY_B] scrambled interceptors and shot down the aircraft, killing "
            "the crew. [COUNTRY_A] claims navigation error. [COUNTRY_B] says the "
            "incursion was deliberate. "
        ),
        "maritime": (
            "[COUNTRY_A] began extracting oil from waters that [COUNTRY_B] claims "
            "fall within its exclusive economic zone. When [COUNTRY_B] sent coast "
            "guard vessels to intervene, [COUNTRY_A] deployed naval warships to "
            "protect its operations. "
        ),
        "strike": (
            "Citing intelligence about an imminent attack, [COUNTRY_A] conducted "
            "airstrikes against [COUNTRY_B]'s military facilities, destroying "
            "weapons stockpiles and killing military personnel. [COUNTRY_B] denies "
            "any planned attack. "
        ),
    },
}


# =====================================================================
# Infrastructure — generally don't need to edit below
# =====================================================================

def _needs_space(name: str) -> bool:
    return not any("\u4e00" <= ch <= "\u9fff" for ch in name)


def _load_model(model_id: str):
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[DTYPE]

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    logger.info(f"Loading {model_id} on {device}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=dtype, trust_remote_code=True,
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer, device


def _score(model, tokenizer, device, text, prefill_ids, tok_a, tok_b):
    ids = tokenizer.encode(text, add_special_tokens=True) + prefill_ids
    with torch.no_grad():
        logits = model(torch.tensor([ids], device=device)).logits[0, -1].float()
    lp = torch.log_softmax(logits, dim=-1)

    # Also get top-5 tokens for diagnostics
    top5_vals, top5_ids = torch.topk(lp, 5)
    top5 = [(tokenizer.decode([tid]), val) for tid, val in zip(top5_ids.tolist(), top5_vals.tolist())]

    return lp[tok_a].item(), lp[tok_b].item(), top5


def _tokenize_pair(c1, c2, tokenizer):
    """Tokenize country names and find shared prefix / score tokens."""
    tok_a_ids = tokenizer.encode((" " if _needs_space(c1) else "") + c1, add_special_tokens=False)
    tok_b_ids = tokenizer.encode((" " if _needs_space(c2) else "") + c2, add_special_tokens=False)

    shared = 0
    for a, b in zip(tok_a_ids, tok_b_ids):
        if a == b:
            shared += 1
        else:
            break

    if shared > 0 and shared < min(len(tok_a_ids), len(tok_b_ids)):
        return tok_a_ids[:shared], tok_a_ids[shared], tok_b_ids[shared]
    return [], tok_a_ids[0], tok_b_ids[0]


def _run_one_suffix(suffix_name, suffix_text, cfg, model, tokenizer, device):
    """Run all pairs × narratives × runs for a single suffix."""
    fewshot = cfg.get("fewshot_prefix", "")
    records = []

    for c1, c2 in cfg["pairs"]:
        prefill, score_a, score_b = _tokenize_pair(c1, c2, tokenizer)

        for narr_name, narr_text in cfg["narratives"].items():
            for run_id in range(cfg["n_runs"]):
                base_fwd = narr_text.replace("[COUNTRY_A]", c1).replace("[COUNTRY_B]", c2)
                prompt_fwd = fewshot + base_fwd + suffix_text
                text_fwd_ab = prompt_fwd.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
                text_fwd_ba = prompt_fwd.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)

                lp_a_ab, lp_b_ab, top5_ab = _score(model, tokenizer, device, text_fwd_ab, prefill, score_a, score_b)
                lp_a_ba, lp_b_ba, _ = _score(model, tokenizer, device, text_fwd_ba, prefill, score_a, score_b)

                log_prob_a_fwd = (lp_a_ab + lp_a_ba) / 2.0
                log_prob_b_fwd = (lp_b_ab + lp_b_ba) / 2.0
                compliance_fwd = (math.exp(lp_a_ab) + math.exp(lp_b_ab) + math.exp(lp_a_ba) + math.exp(lp_b_ba)) / 2.0

                base_rev = narr_text.replace("[COUNTRY_A]", c2).replace("[COUNTRY_B]", c1)
                prompt_rev = fewshot + base_rev + suffix_text
                text_rev_ab = prompt_rev.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)
                text_rev_ba = prompt_rev.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)

                lp_a_ab_r, lp_b_ab_r, _ = _score(model, tokenizer, device, text_rev_ab, prefill, score_a, score_b)
                lp_a_ba_r, lp_b_ba_r, _ = _score(model, tokenizer, device, text_rev_ba, prefill, score_a, score_b)

                log_prob_a_rev = (lp_a_ab_r + lp_a_ba_r) / 2.0
                log_prob_b_rev = (lp_b_ab_r + lp_b_ba_r) / 2.0
                compliance_rev = (math.exp(lp_a_ab_r) + math.exp(lp_b_ab_r) + math.exp(lp_a_ba_r) + math.exp(lp_b_ba_r)) / 2.0

                diff_fwd = log_prob_a_fwd - log_prob_b_fwd
                diff_rev = -(log_prob_a_rev - log_prob_b_rev)
                bias = diff_fwd - diff_rev

                records.append({
                    "suffix": suffix_name,
                    "pair": f"{c1} vs {c2}",
                    "narrative": narr_name,
                    "run_id": run_id,
                    "diff_fwd": diff_fwd,
                    "diff_rev": diff_rev,
                    "bias": bias,
                    "compliance_fwd": compliance_fwd,
                    "compliance_rev": compliance_rev,
                    "token_a": tokenizer.decode([score_a]),
                    "token_b": tokenizer.decode([score_b]),
                    "top5_fwd_ab": top5_ab,
                })

    return records


def _run_prior_baseline(cfg, model, tokenizer, device):
    """Run each suffix with NO narrative to measure prior country preference."""
    fewshot = cfg.get("fewshot_prefix", "")
    records = []
    for c1, c2 in cfg["pairs"]:
        prefill, score_a, score_b = _tokenize_pair(c1, c2, tokenizer)

        for suf_name, suf_text in cfg["suffixes"].items():
            prompt = fewshot + suf_text
            text_ab = prompt.replace("[OPTION_1]", c1).replace("[OPTION_2]", c2)
            text_ba = prompt.replace("[OPTION_1]", c2).replace("[OPTION_2]", c1)

            lp_a_ab, lp_b_ab, top5 = _score(model, tokenizer, device, text_ab, prefill, score_a, score_b)
            lp_a_ba, lp_b_ba, _ = _score(model, tokenizer, device, text_ba, prefill, score_a, score_b)

            log_prob_a = (lp_a_ab + lp_a_ba) / 2.0
            log_prob_b = (lp_b_ab + lp_b_ba) / 2.0
            compliance = (math.exp(lp_a_ab) + math.exp(lp_b_ab) + math.exp(lp_a_ba) + math.exp(lp_b_ba)) / 2.0

            records.append({
                "pair": f"{c1} vs {c2}",
                "suffix": suf_name,
                "prior_diff": log_prob_a - log_prob_b,
                "compliance": compliance,
            })
    return records


def run_experiment():
    cfg = EXPERIMENT_CONFIG

    all_records = []
    prior_records = {}  # model -> list of prior baseline records

    for model_id in cfg["models"]:
        model_short = model_id.split("/")[-1]
        model, tokenizer, device = _load_model(model_id)

        for suf_name, suf_text in cfg["suffixes"].items():
            if suf_name == "prior_only":
                continue  # handled separately
            logger.info(f"[{model_short}] Testing suffix: {suf_name}")
            records = _run_one_suffix(suf_name, suf_text, cfg, model, tokenizer, device)
            for r in records:
                r["model"] = model_short
            all_records.extend(records)

        # Run prior baseline if configured
        if cfg.get("run_prior_baseline"):
            logger.info(f"[{model_short}] Running prior baseline (no narrative)")
            prior_records[model_short] = _run_prior_baseline(cfg, model, tokenizer, device)

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    df = pd.DataFrame(all_records)
    print_diagnostics(df, prior_records)
    return df


def print_diagnostics(df: pd.DataFrame, prior_records: dict = None):
    """Print detailed diagnostics comparing suffixes across models."""
    print("\n" + "=" * 70)
    print("  EXPERIMENT DIAGNOSTICS")
    print("=" * 70)

    models = df["model"].unique()
    suffixes = df["suffix"].unique()

    for model in models:
        mdf = df[df["model"] == model]
        print(f"\n{'─'*70}")
        print(f"  MODEL: {model}")
        print(f"{'─'*70}")

        # 1. Summary comparison table
        print(f"\n  {'suffix':<20s} {'compliance':>10s} {'mean|bias|':>10s} {'med|bias|':>10s} {'mean_SNR':>10s}")
        print(f"  {'─'*20} {'─'*10} {'─'*10} {'─'*10} {'─'*10}")

        for suf in suffixes:
            sub = mdf[mdf["suffix"] == suf]
            if sub.empty:
                continue
            comp = sub["compliance_fwd"].mean()
            pair_bias = sub.groupby("pair")["bias"].mean()
            mean_abs = pair_bias.abs().mean()
            med_abs = pair_bias.abs().median()

            pair_agg = sub.groupby("pair").agg(m=("bias", "mean"), s=("bias", "std"))
            pair_agg["snr"] = pair_agg["m"].abs() / pair_agg["s"].replace(0, float("nan"))
            mean_snr = pair_agg["snr"].mean()

            print(f"  {suf:<20s} {comp:>10.4f} {mean_abs:>10.3f} {med_abs:>10.3f} {mean_snr:>10.2f}")

        # 2. Per-pair compliance
        print(f"\n  Per-pair compliance:")
        print(f"  {'pair':<25s}", end="")
        for suf in suffixes:
            print(f"  {suf:>14s}", end="")
        print()
        for pair in mdf["pair"].unique():
            print(f"  {pair:<25s}", end="")
            for suf in suffixes:
                sub = mdf[(mdf["pair"] == pair) & (mdf["suffix"] == suf)]
                comp = sub["compliance_fwd"].mean() if not sub.empty else 0
                print(f"  {comp:>14.4f}", end="")
            print()

        # 3. Top-5 tokens for best suffix on this model
        best_suf = max(suffixes, key=lambda s: mdf[mdf["suffix"] == s]["compliance_fwd"].mean() if not mdf[mdf["suffix"] == s].empty else 0)
        print(f"\n  Top-5 tokens (suffix='{best_suf}'):")
        for pair in mdf["pair"].unique():
            sub = mdf[(mdf["pair"] == pair) & (mdf["suffix"] == best_suf)]
            if sub.empty:
                continue
            row = sub.iloc[0]
            tokens_str = "  ".join(f"{t[0]!r}({t[1]:.2f})" for t in row["top5_fwd_ab"])
            print(f"    {pair:<23s} {tokens_str}")

    # Cross-model comparison
    if len(models) > 1:
        print(f"\n{'─'*70}")
        print(f"  CROSS-MODEL COMPARISON")
        print(f"{'─'*70}")
        print(f"\n  {'suffix':<20s}", end="")
        for model in models:
            print(f"  {'comp_'+model:>16s}  {'SNR_'+model:>12s}", end="")
        print()

        for suf in suffixes:
            print(f"  {suf:<20s}", end="")
            for model in models:
                sub = df[(df["model"] == model) & (df["suffix"] == suf)]
                if sub.empty:
                    print(f"  {'N/A':>16s}  {'N/A':>12s}", end="")
                    continue
                comp = sub["compliance_fwd"].mean()
                pair_agg = sub.groupby("pair").agg(m=("bias", "mean"), s=("bias", "std"))
                pair_agg["snr"] = pair_agg["m"].abs() / pair_agg["s"].replace(0, float("nan"))
                mean_snr = pair_agg["snr"].mean()
                print(f"  {comp:>16.4f}  {mean_snr:>12.2f}", end="")
            print()

    # Few-shot position bias check: compare all_first vs balanced vs all_second
    pos_suffixes = [s for s in suffixes if s in ("fs3_all_first", "fs3_balanced", "fs3_all_second")]
    if len(pos_suffixes) > 1:
        print(f"\n{'─'*70}")
        print(f"  FEW-SHOT POSITION BIAS CHECK")
        print(f"  Do the few-shot examples bias which option gets picked?")
        print(f"{'─'*70}")
        for model in models:
            print(f"\n  {model}:")
            print(f"  {'pair':<25s}", end="")
            for suf in pos_suffixes:
                print(f"  {suf:>14s}", end="")
            print()
            for pair in df[df["model"] == model]["pair"].unique():
                print(f"  {pair:<25s}", end="")
                for suf in pos_suffixes:
                    sub = df[(df["model"] == model) & (df["pair"] == pair) & (df["suffix"] == suf)]
                    bias = sub["bias"].mean() if not sub.empty else float("nan")
                    print(f"  {bias:>+14.3f}", end="")
                print()

    # Prior baseline: model's preference without any narrative
    if prior_records:
        print(f"\n{'─'*70}")
        print(f"  PRIOR BASELINE (no narrative — raw country preference)")
        print(f"  prior_diff = logprob(c1) - logprob(c2), positive = prefers c1")
        print(f"{'─'*70}")
        prior_df = pd.DataFrame([
            {**r, "model": model}
            for model, records in prior_records.items()
            for r in records
        ])
        # Average across suffixes for each model+pair
        prior_agg = prior_df.groupby(["model", "pair"]).agg(
            prior_diff=("prior_diff", "mean"),
            compliance=("compliance", "mean"),
        ).reset_index()

        for model in prior_df["model"].unique():
            print(f"\n  {model}:")
            print(f"  {'pair':<25s} {'prior_diff':>12s} {'compliance':>12s}")
            msub = prior_agg[prior_agg["model"] == model]
            for _, r in msub.iterrows():
                print(f"  {r['pair']:<25s} {r['prior_diff']:>+12.3f} {r['compliance']:>12.4f}")

        # Show narrative bias vs prior for the main suffix
        main_suf = list(suffixes)[0]  # first suffix
        print(f"\n  NARRATIVE EFFECT (bias with narrative − prior, suffix='{main_suf}'):")
        for model in models:
            print(f"\n  {model}:")
            print(f"  {'pair':<25s} {'with_narr':>12s} {'prior':>12s} {'narrative_effect':>16s}")
            for pair in df[df["model"] == model]["pair"].unique():
                narr_sub = df[(df["model"] == model) & (df["pair"] == pair) & (df["suffix"] == main_suf)]
                prior_sub = prior_agg[(prior_agg["model"] == model) & (prior_agg["pair"] == pair)]
                if narr_sub.empty or prior_sub.empty:
                    continue
                narr_bias = narr_sub["bias"].mean()
                prior_val = prior_sub["prior_diff"].iloc[0]
                effect = narr_bias - prior_val
                print(f"  {pair:<25s} {narr_bias:>+12.3f} {prior_val:>+12.3f} {effect:>+16.3f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    run_experiment()
