"""Simple bias analysis: raw logprob differences, no corrections."""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    RAW_DIR, ASYMMETRY_DIR, SUMMARY_DIR,
    CONTROL_PAIRS, PHONETIC_PAIRS, REAL_PAIRS,
)

logger = logging.getLogger(__name__)


def load_cloze_results(model_name: str, lang: str = "en") -> pd.DataFrame:
    """Load cloze JSONL raw results into a DataFrame."""
    suffix = f"_{lang}" if lang != "en" else ""
    path = RAW_DIR / "cloze" / f"{model_name}_cloze{suffix}.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["pair"] = df["pair"].apply(tuple)
    return df


def compute_bias(df: pd.DataFrame) -> pd.DataFrame:
    """Compute country preference bias for each (pair, scenario).

    For a pair (c1, c2):
      forward:  diff_fwd = logprob(c1) - logprob(c2)   [c1 in role A, c2 in role B]
      reverse:  diff_rev = logprob(c1) - logprob(c2)   [c2 in role A, c1 in role B]

    Note: in the reverse direction, country_a=c2 and country_b=c1 in the
    raw data, so diff_rev = logprob(c2) - logprob(c1) = -(logprob(c1) - logprob(c2)).
    We flip the sign so both diffs are in the same c1-vs-c2 frame.

    bias = diff_fwd - diff_rev (difference of differences)

    If the model has no country preference, swapping roles should flip the
    logprob difference equally, so bias ≈ 0. A positive bias means the model
    favours c1 over c2 beyond what role assignment explains.
    """
    rows = []
    for pair in df["pair"].unique():
        c1, c2 = pair
        pair_df = df[df["pair"] == pair]

        for scenario in pair_df["scenario"].unique():
            scen_df = pair_df[pair_df["scenario"] == scenario]

            fwd = scen_df[scen_df["direction"] == "forward"]
            rev = scen_df[scen_df["direction"] == "reverse"]

            if fwd.empty or rev.empty:
                logger.warning(f"Missing direction for {pair} / {scenario}")
                continue

            # diff = logprob(c1) - logprob(c2), averaged across runs
            # Forward: country_a=c1, country_b=c2, so use as-is
            diff_fwd = (fwd["log_prob_a"] - fwd["log_prob_b"]).mean()
            diff_fwd_std = (fwd["log_prob_a"] - fwd["log_prob_b"]).std()
            # Reverse: country_a=c2, country_b=c1, so flip sign
            diff_rev = -(rev["log_prob_a"] - rev["log_prob_b"]).mean()
            diff_rev_std = (rev["log_prob_a"] - rev["log_prob_b"]).std()

            bias = diff_fwd - diff_rev

            compliance_fwd = fwd["compliance"].mean()
            compliance_rev = rev["compliance"].mean()
            n_runs = len(fwd)

            rows.append({
                "country_1": c1,
                "country_2": c2,
                "scenario": scenario,
                "diff_fwd": diff_fwd,
                "diff_fwd_std": diff_fwd_std,
                "diff_rev": diff_rev,
                "diff_rev_std": diff_rev_std,
                "bias": bias,
                "compliance_fwd": compliance_fwd,
                "compliance_rev": compliance_rev,
                "n_runs": n_runs,
                "model": fwd.iloc[0]["model"],
            })

    return pd.DataFrame(rows)


def aggregate_bias(bias_df: pd.DataFrame) -> pd.DataFrame:
    """Mean bias per pair across narrative scenarios (excluding baseline).

    Also computes narrative_effect = mean_narrative_bias - baseline_bias.
    """
    baseline = bias_df[bias_df["scenario"] == "baseline"].copy()
    narratives = bias_df[bias_df["scenario"] != "baseline"].copy()

    agg = narratives.groupby(["model", "country_1", "country_2"]).agg(
        mean_bias=("bias", "mean"),
        std_bias=("bias", "std"),
        mean_diff_fwd=("diff_fwd", "mean"),
        mean_diff_rev=("diff_rev", "mean"),
        mean_compliance=("compliance_fwd", "mean"),
        n_scenarios=("scenario", "count"),
    ).reset_index()

    # Merge baseline bias
    if not baseline.empty:
        bl = baseline[["country_1", "country_2", "model", "bias"]].rename(
            columns={"bias": "baseline_bias"}
        )
        agg = agg.merge(bl, on=["model", "country_1", "country_2"], how="left")
        agg["narrative_effect"] = agg["mean_bias"] - agg["baseline_bias"]
    else:
        agg["baseline_bias"] = float("nan")
        agg["narrative_effect"] = float("nan")

    return agg


def run_analysis(model_name: str, lang: str = "en") -> dict:
    """Full analysis pipeline for one model."""
    suffix = f"_{lang}" if lang != "en" else ""
    df = load_cloze_results(model_name, lang=lang)
    bias_df = compute_bias(df)

    # Save per-scenario bias matrix
    pivot = bias_df.pivot_table(
        index=["country_1", "country_2"],
        columns="scenario",
        values="bias",
    )
    path = ASYMMETRY_DIR / f"{model_name}_cloze_bias{suffix}.csv"
    pivot.to_csv(path)
    logger.info(f"Saved bias matrix to {path}")

    # Aggregate
    agg = aggregate_bias(bias_df)
    agg_path = SUMMARY_DIR / f"{model_name}_summary{suffix}.csv"
    agg.to_csv(agg_path, index=False)
    logger.info(f"Saved summary to {agg_path}")

    return {"bias": bias_df, "aggregate": agg}


def build_cross_model_summary(model_names: list[str]) -> pd.DataFrame:
    """Build pairs × models summary table."""
    frames = []
    for mn in model_names:
        df = load_cloze_results(mn)
        bias = compute_bias(df)
        agg = aggregate_bias(bias)
        agg = agg.rename(columns={"mean_bias": f"bias_{mn}"})
        frames.append(agg[["country_1", "country_2", f"bias_{mn}"]])

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["country_1", "country_2"], how="outer")

    path = SUMMARY_DIR / "cross_model_comparison.csv"
    merged.to_csv(path, index=False)
    logger.info(f"Saved cross-model summary to {path}")
    return merged
