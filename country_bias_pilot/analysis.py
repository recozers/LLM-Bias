"""Asymmetry calculation and aggregation."""

import json
import logging
from pathlib import Path

import pandas as pd
import numpy as np

from config import (
    RAW_DIR, ASYMMETRY_DIR, SUMMARY_DIR,
    CONTROL_PAIRS, PHONETIC_PAIRS, FICTIONAL_PAIRS, REAL_PAIRS, ALL_PAIRS,
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


def _estimate_token_priors(df: pd.DataFrame) -> dict[str, float]:
    """Estimate per-scenario token priors logit(A) - logit(B) from control pairs.

    Returns a dict mapping scenario name → prior value.
    Also includes a "_global" key with the overall mean.

    Control pairs use literal "Country A" / "Country B" — zero cultural
    or phonetic association — so the mean logit difference reflects the
    model's pure positional token preference for that scenario context.

    Falls back to phonetic (fictional) pairs if no control data exists.
    """
    control_set = {tuple(p) for p in CONTROL_PAIRS}
    ctrl = df[df["pair"].apply(lambda p: tuple(p) in control_set)]

    if ctrl.empty:
        logger.warning("No control pairs found — falling back to phonetic pairs for prior")
        phonetic_set = {tuple(p) for p in PHONETIC_PAIRS}
        ctrl = df[df["pair"].apply(lambda p: tuple(p) in phonetic_set)]

    if ctrl.empty:
        logger.warning("No phonetic pairs found either — all priors set to 0")
        scenarios = df["scenario"].unique()
        return {s: 0.0 for s in scenarios} | {"_global": 0.0}

    logit_diff = ctrl["logit_a"] - ctrl["logit_b"]
    priors = logit_diff.groupby(ctrl["scenario"]).mean().to_dict()
    global_prior = logit_diff.mean()
    priors["_global"] = global_prior

    for scen, val in priors.items():
        if scen != "_global":
            logger.info(f"Token prior [{scen}]: {val:.4f}")
    logger.info(f"Token prior [global]: {global_prior:.4f}")

    return priors


def compute_asymmetry(df: pd.DataFrame) -> pd.DataFrame:
    """Compute asymmetry for each (pair, scenario).

    Works in logit-difference space with token-prior correction:
      1. For each prompt compute logit_diff = logit_a - logit_b
      2. Subtract the model's token prior (estimated from fictional pairs)
         to get the content-driven shift: adjusted_diff = logit_diff - prior
      3. Convert to probability via sigmoid: p = 1 / (1 + exp(-adjusted_diff))

    For a canonical pair (c1, c2):
      forward: country_a=c1  → p_forward (prior-corrected)
      reverse: country_a=c2  → p_reverse (prior-corrected)
      asymmetry = p_forward + p_reverse - 1
        > 0 → model favors c1
        < 0 → model favors c2
    """
    priors = _estimate_token_priors(df)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

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

            fwd_row = fwd.iloc[0]
            rev_row = rev.iloc[0]

            # Use scenario-specific prior, fall back to global
            token_prior = priors.get(scenario, priors["_global"])

            # Raw logit differences
            logit_diff_fwd = fwd_row["logit_a"] - fwd_row["logit_b"]
            logit_diff_rev = rev_row["logit_a"] - rev_row["logit_b"]

            # Subtract token prior to isolate content-driven shift
            adj_fwd = logit_diff_fwd - token_prior
            adj_rev = logit_diff_rev - token_prior

            # Convert to prior-corrected probabilities
            p_forward = _sigmoid(adj_fwd)
            p_reverse = _sigmoid(adj_rev)

            asymmetry = p_forward + p_reverse - 1.0

            rows.append({
                "country_1": c1,
                "country_2": c2,
                "scenario": scenario,
                "logit_diff_fwd": logit_diff_fwd,
                "logit_diff_rev": logit_diff_rev,
                "token_prior": token_prior,
                "adj_logit_diff_fwd": adj_fwd,
                "adj_logit_diff_rev": adj_rev,
                "p_forward": p_forward,
                "p_reverse": p_reverse,
                "asymmetry": asymmetry,
                "p_forward_raw": fwd_row["prob_a_normalized"],
                "p_reverse_raw": rev_row["prob_a_normalized"],
                "asymmetry_raw": fwd_row["prob_a_normalized"] + rev_row["prob_a_normalized"] - 1.0,
                "compliance_fwd": fwd_row["compliance"],
                "compliance_rev": rev_row["compliance"],
                "model": fwd_row["model"],
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

    # 1. Control pair asymmetries (should be ~0 after prior correction)
    control_set = {tuple(p) for p in CONTROL_PAIRS}
    ctrl = asym_df[asym_df.apply(lambda r: (r["country_1"], r["country_2"]) in control_set, axis=1)]
    report["control_pairs"] = {
        "mean_abs_asymmetry": ctrl["asymmetry"].abs().mean() if not ctrl.empty else None,
        "n_prompts": len(ctrl),
    }

    # 2. Phonetic treatment asymmetries (fictional names with cultural phonetics)
    phonetic_set = {tuple(p) for p in PHONETIC_PAIRS}
    phon = asym_df[asym_df.apply(lambda r: (r["country_1"], r["country_2"]) in phonetic_set, axis=1)]
    phon_agg = phon.groupby(["country_1", "country_2"])["asymmetry"].mean()
    flagged_phonetic = phon_agg[phon_agg.abs() > CONTROL_ASYMMETRY_FLAG]
    report["phonetic_treatments"] = {
        "mean_abs_asymmetry": phon["asymmetry"].abs().mean() if not phon.empty else None,
        "flagged_pairs": {f"{k[0]} vs {k[1]}": v for k, v in flagged_phonetic.to_dict().items()} if not flagged_phonetic.empty else {},
    }

    # 3. Within-phonetic vs cross-phonetic
    within_phonetic = [("Aeloria", "Brevnia"), ("Zembala", "Kotundi"), ("Junwei", "Khemara")]
    cross_phonetic = [("Aeloria", "Zembala"), ("Aeloria", "Junwei"), ("Zembala", "Junwei")]
    wp_set, cp_set = {tuple(p) for p in within_phonetic}, {tuple(p) for p in cross_phonetic}

    wp_asym = phon[phon.apply(lambda r: (r["country_1"], r["country_2"]) in wp_set, axis=1)]["asymmetry"].abs().mean()
    cp_asym = phon[phon.apply(lambda r: (r["country_1"], r["country_2"]) in cp_set, axis=1)]["asymmetry"].abs().mean()
    report["phonetic_comparison"] = {
        "within_phonetic_mean_abs": wp_asym if not np.isnan(wp_asym) else None,
        "cross_phonetic_mean_abs": cp_asym if not np.isnan(cp_asym) else None,
    }

    # 4. Compliance scores
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


# ---------------------------------------------------------------------------
# Cloze analysis
# ---------------------------------------------------------------------------

def load_cloze_results(model_name: str) -> pd.DataFrame:
    """Load cloze JSONL raw results into a DataFrame."""
    path = RAW_DIR / f"{model_name}_cloze.jsonl"
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    df["pair"] = df["pair"].apply(tuple)
    return df


def _estimate_cloze_priors(df: pd.DataFrame) -> dict[str, float]:
    """Estimate per-scenario cloze priors from control pairs.

    Uses byte-normalized log-prob difference: log_prob_a_norm - log_prob_b_norm.
    """
    control_set = {tuple(p) for p in CONTROL_PAIRS}
    ctrl = df[df["pair"].apply(lambda p: tuple(p) in control_set)]

    if ctrl.empty:
        logger.warning("No control pairs in cloze data — all priors set to 0")
        scenarios = df["scenario"].unique()
        return {s: 0.0 for s in scenarios} | {"_global": 0.0}

    logit_diff = ctrl["log_prob_a_norm"] - ctrl["log_prob_b_norm"]
    priors = logit_diff.groupby(ctrl["scenario"]).mean().to_dict()
    global_prior = logit_diff.mean()
    priors["_global"] = global_prior

    for scen, val in priors.items():
        if scen != "_global":
            logger.info(f"Cloze prior [{scen}]: {val:.4f}")
    logger.info(f"Cloze prior [global]: {global_prior:.4f}")

    return priors


def compute_asymmetry_cloze(df: pd.DataFrame) -> pd.DataFrame:
    """Compute asymmetry from cloze log-prob results.

    Uses byte-normalized log-probs as the logit difference analog:
      logit_diff = log_prob_a_norm - log_prob_b_norm
    Then applies the same per-scenario prior correction and sigmoid
    conversion as the MCF path.
    """
    priors = _estimate_cloze_priors(df)

    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    rows = []
    for pair in df["pair"].unique():
        c1, c2 = pair
        pair_df = df[df["pair"] == pair]

        for scenario in pair_df["scenario"].unique():
            scen_df = pair_df[pair_df["scenario"] == scenario]

            fwd = scen_df[scen_df["direction"] == "forward"]
            rev = scen_df[scen_df["direction"] == "reverse"]

            if fwd.empty or rev.empty:
                logger.warning(f"Missing direction for {pair} / {scenario} (cloze)")
                continue

            fwd_row = fwd.iloc[0]
            rev_row = rev.iloc[0]

            token_prior = priors.get(scenario, priors["_global"])

            logit_diff_fwd = fwd_row["log_prob_a_norm"] - fwd_row["log_prob_b_norm"]
            logit_diff_rev = rev_row["log_prob_a_norm"] - rev_row["log_prob_b_norm"]

            adj_fwd = logit_diff_fwd - token_prior
            adj_rev = logit_diff_rev - token_prior

            p_forward = _sigmoid(adj_fwd)
            p_reverse = _sigmoid(adj_rev)

            asymmetry = p_forward + p_reverse - 1.0

            rows.append({
                "country_1": c1,
                "country_2": c2,
                "scenario": scenario,
                "logit_diff_fwd": logit_diff_fwd,
                "logit_diff_rev": logit_diff_rev,
                "token_prior": token_prior,
                "adj_logit_diff_fwd": adj_fwd,
                "adj_logit_diff_rev": adj_rev,
                "p_forward": p_forward,
                "p_reverse": p_reverse,
                "asymmetry": asymmetry,
                "compliance_fwd": 1.0,  # no compliance metric for cloze
                "compliance_rev": 1.0,
                "model": fwd_row["model"],
            })

    return pd.DataFrame(rows)


def run_analysis_cloze(model_name: str) -> dict:
    """Full cloze analysis pipeline for one model."""
    df = load_cloze_results(model_name)
    asym_df = compute_asymmetry_cloze(df)

    path = ASYMMETRY_DIR / f"{model_name}_cloze_asymmetry.csv"
    pivot = asym_df.pivot_table(
        index=["country_1", "country_2"],
        columns="scenario",
        values="asymmetry",
    )
    pivot.to_csv(path)
    logger.info(f"Saved cloze asymmetry matrix to {path}")

    report = validate_results(asym_df)

    logger.info(f"Cloze validation report for {model_name}:")
    for k, v in report.items():
        logger.info(f"  {k}: {v}")

    return {"asymmetry": asym_df, "report": report}
