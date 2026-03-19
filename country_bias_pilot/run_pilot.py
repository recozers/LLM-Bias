#!/usr/bin/env python3
"""Main script — end-to-end cloze bias pipeline.

Usage:
    # Full run (all models, all pairs, all scenarios, 5 runs)
    python run_pilot.py

    # Quick smoke test (1 model, 2 pairs, 1 scenario, 1 run)
    python run_pilot.py --test

    # Single model, custom run count
    python run_pilot.py --models gpt2 --n-runs 3

    # Analysis + plots only (skip inference, use existing raw results)
    python run_pilot.py --analysis-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime

from config import (
    MODELS, N_RUNS, PHONETIC_PAIRS, REAL_PAIRS, CONTROL_PAIRS,
    SCENARIOS_CLOZE, ALL_PAIRS_CLOZE,
    SUMMARY_DIR, RESULTS_DIR,
)

logger = logging.getLogger(__name__)


def _setup_logging():
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


def parse_args():
    p = argparse.ArgumentParser(description="LLM Country Bias — Cloze Pipeline")
    p.add_argument("--test", action="store_true", help="Smoke test: 1 model, 2 pairs, 1 scenario, 1 run")
    p.add_argument("--models", nargs="+", default=None, help="Model keys to run (default: all)")
    p.add_argument("--n-runs", type=int, default=N_RUNS, help=f"Number of inference runs per prompt (default: {N_RUNS})")
    p.add_argument("--analysis-only", action="store_true", help="Skip inference, run analysis + plots on existing results")
    return p.parse_args()


def main():
    ts = _setup_logging()
    args = parse_args()

    # Import here to avoid loading torch at parse time
    from local_inference import generate_all_prompts_cloze, run_cloze_inference
    from analysis import run_analysis, build_cross_model_summary
    from visualize import generate_all_plots

    # --- Resolve what to run ---
    if args.test:
        model_keys = [list(MODELS.keys())[0]]
        pairs = [CONTROL_PAIRS[0], PHONETIC_PAIRS[0], REAL_PAIRS[0]]
        scenarios = {k: v for k, v in list(SCENARIOS_CLOZE.items())[:1]}
        n_runs = 1
        logger.info("=== SMOKE TEST MODE ===")
    else:
        model_keys = args.models or list(MODELS.keys())
        pairs = None
        scenarios = None
        n_runs = args.n_runs

    for mk in model_keys:
        if mk not in MODELS:
            logger.error(f"Unknown model key '{mk}'. Available: {list(MODELS.keys())}")
            sys.exit(1)

    # --- Inference ---
    failed_models = []
    if not args.analysis_only:
        prompts = generate_all_prompts_cloze(pairs=pairs, scenarios=scenarios)
        logger.info(f"Generated {len(prompts)} cloze prompts × {n_runs} runs = {len(prompts) * n_runs} total passes")

        for mk in model_keys:
            logger.info(f"\n{'='*60}\n  Running: {mk}  ({MODELS[mk]})\n{'='*60}")
            try:
                run_cloze_inference(mk, MODELS[mk], prompts, n_runs=n_runs)
            except Exception:
                logger.exception(f"Inference FAILED for {mk}")
                failed_models.append(mk)

    # --- Analysis ---
    completed_models = [mk for mk in model_keys if mk not in failed_models]
    bias_dfs = {}
    for mk in completed_models:
        logger.info(f"\n{'='*60}\n  Analyzing: {mk}\n{'='*60}")
        try:
            result = run_analysis(mk)
            bias_dfs[mk] = result["bias"]
        except Exception:
            logger.exception(f"Analysis FAILED for {mk}")

    # --- Cross-model summary ---
    if len(bias_dfs) > 1:
        build_cross_model_summary(list(bias_dfs.keys()))

    # --- Visualizations ---
    if bias_dfs:
        logger.info(f"\n{'='*60}\n  Generating plots\n{'='*60}")
        generate_all_plots(bias_dfs)

    # --- Print summary ---
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    if failed_models:
        print(f"\n  FAILED models: {', '.join(failed_models)}")

    for mk, df in bias_dfs.items():
        print(f"\n--- {mk} ---")
        real = df[~df.apply(
            lambda r: (r["country_1"], r["country_2"]) in
            {tuple(p) for p in CONTROL_PAIRS} | {tuple(p) for p in PHONETIC_PAIRS},
            axis=1,
        )]
        if not real.empty:
            print(f"  Mean |bias| (real pairs): {real['bias'].abs().mean():.4f}")
            print(f"  Mean diff_fwd: {real['diff_fwd'].mean():.4f}")
            print(f"  Mean diff_rev: {real['diff_rev'].mean():.4f}")
            print(f"  N scenarios: {real['scenario'].nunique()}")

    # --- Save metadata ---
    metadata = {
        "timestamp": ts,
        "models": model_keys,
        "model_ids": {mk: MODELS[mk] for mk in model_keys},
        "n_runs": n_runs,
        "test_mode": args.test,
        "analysis_only": args.analysis_only,
    }
    meta_path = SUMMARY_DIR / "run_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nResults saved to results/")
    print("Done.")


if __name__ == "__main__":
    main()
