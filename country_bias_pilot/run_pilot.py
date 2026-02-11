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

    # Specify device
    python run_pilot.py --device cuda:1
"""

import argparse
import json
import logging
import sys

from config import MODELS, FICTIONAL_PAIRS, REAL_PAIRS, SCENARIOS, SUMMARY_DIR
from prompts import generate_all_prompts
from inference import run_inference
from analysis import run_analysis, build_cross_model_summary, compute_asymmetry, load_raw_results
from visualize import generate_all_plots

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="LLM Country Bias Pilot")
    p.add_argument("--test", action="store_true", help="Smoke test: 1 model, 2 pairs, 1 scenario")
    p.add_argument("--models", nargs="+", default=None, help="Model keys to run (default: all)")
    p.add_argument("--device", default="cuda", help="Torch device (default: cuda)")
    p.add_argument("--analysis-only", action="store_true", help="Skip inference, run analysis + plots on existing results")
    return p.parse_args()


def main():
    args = parse_args()

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
    if not args.analysis_only:
        for mk in model_keys:
            logger.info(f"\n{'='*60}\n  Running inference: {mk}  ({MODELS[mk]})\n{'='*60}")
            run_inference(mk, MODELS[mk], prompts, device=args.device)

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
        print(f"  Fictional controls mean |asymmetry|: {fc['mean_abs_asymmetry']:.4f}" if fc['mean_abs_asymmetry'] is not None else "  Fictional controls: N/A")
        if fc["flagged_pairs"]:
            print(f"  ⚠ Flagged fictional pairs: {fc['flagged_pairs']}")

        pc = report["phonetic_comparison"]
        if pc["within_phonetic_mean_abs"] is not None:
            print(f"  Within-phonetic |asym|: {pc['within_phonetic_mean_abs']:.4f}")
            print(f"  Cross-phonetic  |asym|: {pc['cross_phonetic_mean_abs']:.4f}")

        print(f"  Low-compliance prompts: {report['low_compliance_count']}")

    # Save reports as JSON
    report_path = SUMMARY_DIR / "validation_reports.json"
    # Convert any non-serializable values
    serializable = {}
    for mk, r in reports.items():
        serializable[mk] = json.loads(json.dumps(r, default=str))
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    logger.info(f"Saved validation reports to {report_path}")

    print(f"\nResults saved to results/")
    print("Done.")


if __name__ == "__main__":
    main()
