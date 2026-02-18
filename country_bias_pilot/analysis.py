"""Asymmetry calculation and aggregation."""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    RAW_DIR, ASYMMETRY_DIR, SUMMARY_DIR,
    FICTIONAL_PAIRS, REAL_PAIRS, ALL_PAIRS,
    CONTROL_ASYMMETRY_FLAG, COMPLIANCE_WARN,
)

logger = logging.getLogger(__name__)


def load_raw_results(model_name: str) -> pd.DataFrame:
    """Load JSONL raw results into a DataFrame."""
    path = RAW_DIR / f"{model_name}_logits.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    # Convert pair list back to tuple for grouping
    df["pair"] = df["pair"].apply(tuple)
    return df


def compute_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Compute asymmetry for each (pair, scenario).

    For a canonical pair (c1, c2):
      forward row: country_a=c1, country_b=c2  → p_forward = prob_a_normalized
      reverse row: country_a=c2, country_b=c1  → p_reverse = prob_a_normalized

      asymmetry = p_forward + p_reverse - 1
        > 0 → model favors c1
        < 0 → model favors c2
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

            p_forward = fwd.iloc[0]["prob_a_normalized"]
            p_reverse = rev.iloc[0]["prob_a_normalized"]
            compliance_fwd = fwd.iloc[0]["compliance"]
            compliance_rev = rev.iloc[0]["compliance"]

            asymmetry = p_forward + p_reverse - 1.0

            rows.append({
                "country_1": c1,
                "country_2": c2,
                "scenario": scenario,
                "p_forward": p_forward,
                "p_reverse": p_reverse,
                "asymmetry": asymmetry,
                "compliance_fwd": compliance_fwd,
                "compliance_rev": compliance_rev,
                "model": fwd.iloc[0]["model"],
            })

    return pd.DataFrame(rows)


def aggregate_asymmetry(asym_df: pd.DataFrame) -> pd.DataFrame:
    """Mean asymmetry per pair (across scenarios)."""
    agg = asym_df.groupby(["model", "country_1", "country_2"]).agg(
        mean_asymmetry=("asymmetry", "mean"),
        std_asymmetry=("asymmetry", "std"),
        mean_compliance=("compliance_fwd", "mean"),
        n_scenarios=("scenario", "count"),
    ).reset_index()
    return agg


def save_asymmetry(model_name: str, asym_df: pd.DataFrame):
    """Save pair × scenario asymmetry matrix."""
    pivot = asym_df.pivot_table(
        index=["country_1", "country_2"],
        columns="scenario",
        values="asymmetry",
    )
    path = ASYMMETRY_DIR / f"{model_name}_asymmetry.csv"
    pivot.to_csv(path)
    logger.info(f"Saved asymmetry matrix to {path}")
    return pivot


def build_cross_model_summary(model_names: list[str]) -> pd.DataFrame:
    """Build pairs × models summary table."""
    frames = []
    for mn in model_names:
        df = load_raw_results(mn)
        asym = compute_asymmetry(df)
        agg = aggregate_asymmetry(asym)
        agg = agg.rename(columns={"mean_asymmetry": f"asym_{mn}"})
        frames.append(agg[["country_1", "country_2", f"asym_{mn}"]])

    merged = frames[0]
    for f in frames[1:]:
        merged = merged.merge(f, on=["country_1", "country_2"], how="outer")

    path = SUMMARY_DIR / "cross_model_comparison.csv"
    merged.to_csv(path, index=False)
    logger.info(f"Saved cross-model summary to {path}")
    return merged


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def validate_results(asym_df: pd.DataFrame) -> dict:
    """Run all validation checks and return a report dict."""
    report = {}

    # 1. Fictional control asymmetries
    fictional_set = {tuple(p) for p in FICTIONAL_PAIRS}
    fict = asym_df[asym_df.apply(lambda r: (r["country_1"], r["country_2"]) in fictional_set, axis=1)]
    fict_agg = fict.groupby(["country_1", "country_2"])["asymmetry"].mean()
    flagged_controls = fict_agg[fict_agg.abs() > CONTROL_ASYMMETRY_FLAG]
    report["fictional_controls"] = {
        "mean_abs_asymmetry": fict["asymmetry"].abs().mean() if not fict.empty else None,
        "flagged_pairs": {f"{k[0]} vs {k[1]}": v for k, v in flagged_controls.to_dict().items()} if not flagged_controls.empty else {},
    }

    # 2. Within-phonetic vs cross-phonetic fictional pairs
    within_phonetic = [("Aeloria", "Brevnia"), ("Zembala", "Kotundi"), ("Junwei", "Khemara")]
    cross_phonetic = [("Aeloria", "Zembala"), ("Aeloria", "Junwei"), ("Zembala", "Junwei")]
    wp_set, cp_set = {tuple(p) for p in within_phonetic}, {tuple(p) for p in cross_phonetic}

    wp_asym = fict[fict.apply(lambda r: (r["country_1"], r["country_2"]) in wp_set, axis=1)]["asymmetry"].abs().mean()
    cp_asym = fict[fict.apply(lambda r: (r["country_1"], r["country_2"]) in cp_set, axis=1)]["asymmetry"].abs().mean()
    report["phonetic_comparison"] = {
        "within_phonetic_mean_abs": wp_asym if not np.isnan(wp_asym) else None,
        "cross_phonetic_mean_abs": cp_asym if not np.isnan(cp_asym) else None,
    }

    # 3. Compliance scores
    low_compliance = asym_df[
        (asym_df["compliance_fwd"] < COMPLIANCE_WARN) | (asym_df["compliance_rev"] < COMPLIANCE_WARN)
    ]
    report["low_compliance_count"] = len(low_compliance)
    if not low_compliance.empty:
        report["low_compliance_examples"] = low_compliance[
            ["model", "country_1", "country_2", "scenario", "compliance_fwd", "compliance_rev"]
        ].head(10).to_dict("records")

    return report


def run_analysis(model_name: str) -> dict:
    """Full analysis pipeline for one model. Returns validation report."""
    df = load_raw_results(model_name)
    asym_df = compute_asymmetry(df)
    save_asymmetry(model_name, asym_df)
    report = validate_results(asym_df)

    logger.info(f"Validation report for {model_name}:")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    return {"asymmetry": asym_df, "report": report}
