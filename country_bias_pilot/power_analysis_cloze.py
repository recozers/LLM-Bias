#!/usr/bin/env python3
"""Power analysis for cloze preference scores."""

from __future__ import annotations

import argparse
import math
from statistics import NormalDist

import pandas as pd

from analysis import load_cloze_results, compute_asymmetry_cloze
from config import MODELS, CONTROL_PAIRS, PHONETIC_PAIRS, REAL_PAIRS


def _pair_filter(name: str) -> set[tuple[str, str]]:
    if name == "real":
        return {tuple(p) for p in REAL_PAIRS}
    if name == "control":
        return {tuple(p) for p in CONTROL_PAIRS}
    if name == "phonetic":
        return {tuple(p) for p in PHONETIC_PAIRS}
    if name == "all":
        return {tuple(p) for p in (CONTROL_PAIRS + PHONETIC_PAIRS + REAL_PAIRS)}
    raise ValueError(f"Unknown pair set: {name}")


def _required_prompts(effect: float, sigma_prompt: float, alpha: float, power: float) -> int:
    if effect <= 0:
        raise ValueError("Effect size must be > 0")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    n = ((z_alpha + z_power) * sigma_prompt / effect) ** 2
    return math.ceil(n)


def _required_prompts_sampled(
    effect: float,
    sigma_prompt: float,
    sigma_run: float,
    runs_per_prompt: int,
    alpha: float,
    power: float,
) -> int:
    if effect <= 0:
        raise ValueError("Effect size must be > 0")
    if runs_per_prompt < 1:
        raise ValueError("runs_per_prompt must be >= 1")
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_power = NormalDist().inv_cdf(power)
    var_term = sigma_prompt**2 + (sigma_run**2) / runs_per_prompt
    n = ((z_alpha + z_power) ** 2) * var_term / (effect**2)
    return math.ceil(n)


def _power_at_n(effect: float, sigma_prompt: float, n_prompts: int, alpha: float) -> float:
    if n_prompts < 1:
        return 0.0
    z_alpha = NormalDist().inv_cdf(1 - alpha / 2)
    z_effect = abs(effect) / (sigma_prompt / math.sqrt(n_prompts))
    nd = NormalDist()
    return (1.0 - nd.cdf(z_alpha - z_effect)) + nd.cdf(-z_alpha - z_effect)


def parse_args():
    p = argparse.ArgumentParser(description="Cloze power analysis")
    p.add_argument("--models", nargs="*", default=list(MODELS.keys()), help="Model keys to include")
    p.add_argument(
        "--pair-set",
        choices=["real", "control", "phonetic", "all"],
        default="real",
        help="Which pair category to analyze",
    )
    p.add_argument("--alpha", type=float, default=0.05, help="Two-sided significance level")
    p.add_argument("--power", type=float, default=0.80, help="Target statistical power")
    p.add_argument(
        "--effect-sizes",
        type=float,
        nargs="+",
        default=[0.05, 0.10, 0.15],
        help="Target absolute asymmetry effects to detect",
    )
    p.add_argument(
        "--sigma-quantile",
        type=float,
        default=0.75,
        help="Quantile of within-pair SD used as sigma_prompt (conservative if high)",
    )
    p.add_argument(
        "--sampled-runs",
        type=int,
        nargs="*",
        default=[],
        help="Optional: evaluate prompts-vs-runs tradeoff for stochastic A/B sampling",
    )
    return p.parse_args()


def main():
    args = parse_args()
    pair_set = _pair_filter(args.pair_set)

    per_model_pair = []
    all_rows = []

    for model_key in args.models:
        try:
            df = load_cloze_results(model_key)
        except FileNotFoundError:
            continue

        asym_df = compute_asymmetry_cloze(df)
        asym_df["pair"] = list(zip(asym_df["country_1"], asym_df["country_2"]))
        asym_df = asym_df[asym_df["pair"].isin(pair_set)]
        if asym_df.empty:
            continue

        all_rows.append(asym_df)
        grouped = asym_df.groupby(["model", "country_1", "country_2"])["asymmetry"].agg(["mean", "std", "count"])
        per_model_pair.append(grouped.reset_index())

    if not per_model_pair:
        print("No cloze data found for selected models/pair set.")
        return

    summary = pd.concat(per_model_pair, ignore_index=True)
    all_asym = pd.concat(all_rows, ignore_index=True)

    sigma_prompt = float(summary["std"].quantile(args.sigma_quantile))
    n_current = int(summary["count"].median())
    effect_abs = summary["mean"].abs()

    print("CLOZE POWER ANALYSIS")
    print(f"Models used: {sorted(summary['model'].unique())}")
    print(f"Pair set: {args.pair_set}")
    print(f"Model-pair units: {len(summary)}")
    print(f"Current prompts per pair (median): {n_current}")
    print(f"alpha={args.alpha:.3f}, target power={args.power:.2f}")
    print(f"Sigma prompt (q={args.sigma_quantile:.2f}): {sigma_prompt:.4f}")
    print(f"Observed |effect| quantiles: q25={effect_abs.quantile(0.25):.4f}, "
          f"q50={effect_abs.quantile(0.50):.4f}, q75={effect_abs.quantile(0.75):.4f}")

    print("\nRequired distinct prompts per pair (deterministic logit scoring):")
    for effect in args.effect_sizes:
        needed = _required_prompts(effect, sigma_prompt, args.alpha, args.power)
        print(f"  effect={effect:.3f} -> prompts={needed}")

    print("\nPower at current prompt count:")
    for q in [0.25, 0.50, 0.75]:
        effect_q = float(effect_abs.quantile(q))
        power_q = _power_at_n(effect_q, sigma_prompt, n_current, args.alpha)
        print(f"  observed effect q{int(q*100)} ({effect_q:.4f}) -> power={power_q:.3f}")

    print("\nRuns guidance:")
    print("  For this cloze pipeline, runs are deterministic (log-prob extraction),")
    print("  so repeating the exact same prompt does not add statistical power.")
    print("  Use runs=1 and increase distinct prompts instead.")

    if args.sampled_runs:
        # Hypothetical variance if one sampled A/B answer is collected per direction.
        run_var = (all_asym["p_forward_raw"] * (1 - all_asym["p_forward_raw"])
                   + all_asym["p_reverse_raw"] * (1 - all_asym["p_reverse_raw"]))
        sigma_run = float(math.sqrt(run_var.mean()))
        print("\nHypothetical prompts-vs-runs (if using stochastic sampled A/B choices):")
        print(f"  Estimated sigma_run={sigma_run:.4f} per prompt (one sample per direction)")
        for effect in args.effect_sizes:
            for r in args.sampled_runs:
                needed = _required_prompts_sampled(
                    effect=effect,
                    sigma_prompt=sigma_prompt,
                    sigma_run=sigma_run,
                    runs_per_prompt=r,
                    alpha=args.alpha,
                    power=args.power,
                )
                print(f"  effect={effect:.3f}, runs={r} -> prompts={needed}")


if __name__ == "__main__":
    main()
