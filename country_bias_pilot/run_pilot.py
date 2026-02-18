#!/usr/bin/env python3
"""Main script — end-to-end pilot pipeline.

Usage:
    # Full run (all models, all pairs, all scenarios)
    python run_pilot.py

    # Quick smoke test (1 model, 2 pairs, 1 scenario)
    python run_pilot.py --test

    # Single model
    python run_pilot.py --models llama3-8b

    # Analysis + plots only (skip inference, use existing raw results)
    python run_pilot.py --analysis-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from config import MODELS, FICTIONAL_PAIRS, REAL_PAIRS, SCENARIOS, SUMMARY_DIR, RESULTS_DIR
from prompts import generate_all_prompts
from inference import run_inference
from analysis import run_analysis, build_cross_model_summary
from visualize import generate_all_plots

logger = logging.getLogger(__name__)


def _setup_logging():
    """Log to both stdout and a timestamped file in results/."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = RESULTS_DIR / f"run_{ts}.log"

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(name)s  %(message)s")

    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(fmt)
    root.addHandler(stdout_handler)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(fmt)
    root.addHandler(file_handler)

    logger.info(f"Logging to {log_path}")
    return ts


def _check_hf_token():
    """Warn early if no HuggingFace token is available (needed for gated models)."""
    try:
        from huggingface_hub import HfFolder
        token = HfFolder.get_token()
        if token:
            logger.info("HuggingFace token found")
            return
    except ImportError:
        pass

    import os
    if os.environ.get("HF_TOKEN"):
        logger.info("HF_TOKEN environment variable found")
        return

    logger.warning(
        "No HuggingFace token detected. Gated models (Llama 3, Mistral) will fail to download. "
        "Run 'huggingface-cli login' or set HF_TOKEN env var."
    )


def parse_args():
    p = argparse.ArgumentParser(description="LLM Country Bias Pilot")
    p.add_argument("--test", action="store_true", help="Smoke test: 1 model, 2 pairs, 1 scenario")
    p.add_argument("--models", nargs="+", default=None, help="Model keys to run (default: all)")
    p.add_argument("--analysis-only", action="store_true", help="Skip inference, run analysis + plots on existing results")
    return p.parse_args()


def main():
    ts = _setup_logging()
    args = parse_args()
    _check_hf_token()

    # --- Resolve which models / pairs / scenarios to use ---
    if args.test:
        model_keys = [list(MODELS.keys())[0]]  # first model only
        pairs = [FICTIONAL_PAIRS[0], REAL_PAIRS[0]]
        scenarios = {k: v for k, v in list(SCENARIOS.items())[:1]}
        logger.info("=== SMOKE TEST MODE ===")
    else:
        model_keys = args.models or list(MODELS.keys())
        pairs = None       # all
        scenarios = None    # all

    # Validate model keys
    for mk in model_keys:
        if mk not in MODELS:
            logger.error(f"Unknown model key '{mk}'. Available: {list(MODELS.keys())}")
            sys.exit(1)

    # --- Generate prompts ---
    prompts = generate_all_prompts(pairs=pairs, scenarios=scenarios)
    logger.info(f"Generated {len(prompts)} prompts")

    # --- Inference ---
    all_warnings = []
    if not args.analysis_only:
        for mk in model_keys:
            logger.info(f"\n{'='*60}\n  Running inference: {mk}  ({MODELS[mk]})\n{'='*60}")
            results = run_inference(mk, MODELS[mk], prompts)
            for r in results:
                all_warnings.extend(r.get("warnings", []))

    # --- Analysis ---
    asym_dfs = {}
    reports = {}
    for mk in model_keys:
        logger.info(f"\n{'='*60}\n  Analyzing: {mk}\n{'='*60}")
        result = run_analysis(mk)
        asym_dfs[mk] = result["asymmetry"]
        reports[mk] = result["report"]

    # --- Cross-model summary ---
    if len(model_keys) > 1:
        build_cross_model_summary(model_keys)

    # --- Visualizations ---
    logger.info(f"\n{'='*60}\n  Generating plots\n{'='*60}")
    generate_all_plots(asym_dfs)

    # --- Print summary ---
    print("\n" + "=" * 60)
    print("  PILOT RESULTS SUMMARY")
    print("=" * 60)
    for mk in model_keys:
        print(f"\n--- {mk} ---")
        report = reports[mk]

        fc = report["fictional_controls"]
        if fc["mean_abs_asymmetry"] is not None:
            print(f"  Fictional controls mean |asymmetry|: {fc['mean_abs_asymmetry']:.4f}")
        else:
            print("  Fictional controls: N/A")
        if fc["flagged_pairs"]:
            print(f"  WARNING Flagged fictional pairs: {fc['flagged_pairs']}")

        pc = report["phonetic_comparison"]
        if pc["within_phonetic_mean_abs"] is not None:
            print(f"  Within-phonetic |asym|: {pc['within_phonetic_mean_abs']:.4f}")
            print(f"  Cross-phonetic  |asym|: {pc['cross_phonetic_mean_abs']:.4f}")

        print(f"  Low-compliance prompts: {report['low_compliance_count']}")

    # --- Save run metadata ---
    metadata = {
        "timestamp": ts,
        "models": model_keys,
        "model_ids": {mk: MODELS[mk] for mk in model_keys},
        "prompt_count": len(prompts),
        "test_mode": args.test,
        "analysis_only": args.analysis_only,
        "inference_warnings": all_warnings[:100],  # cap to avoid huge files
        "inference_warning_count": len(all_warnings),
    }

    # Save reports as JSON
    report_path = SUMMARY_DIR / "validation_reports.json"
    serializable = {}
    for mk, r in reports.items():
        serializable[mk] = json.loads(json.dumps(r, default=str))
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved validation reports to {report_path}")

    meta_path = SUMMARY_DIR / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved run metadata to {meta_path}")

    print(f"\nResults saved to results/")
    print("Done.")


if __name__ == "__main__":
    main()
