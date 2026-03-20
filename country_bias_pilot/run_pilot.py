#!/usr/bin/env python3
"""Main script — end-to-end cloze bias pipeline.

Usage:
    # Full run, English only (default)
    python run_pilot.py

    # Both languages
    python run_pilot.py --lang en zh

    # Chinese only
    python run_pilot.py --lang zh

    # Quick smoke test
    python run_pilot.py --test

    # Analysis + plots only
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
    PHONETIC_PAIRS_ZH, REAL_PAIRS_ZH, CONTROL_PAIRS_ZH,
    SCENARIOS_CLOZE, ALL_PAIRS_CLOZE,
    SCENARIOS_CLOZE_ZH, ALL_PAIRS_CLOZE_ZH,
    SUMMARY_DIR, RESULTS_DIR,
)

logger = logging.getLogger(__name__)

# Per-language control/phonetic sets for summary filtering
_FILTER_SETS = {
    "en": (
        {tuple(p) for p in CONTROL_PAIRS} | {tuple(p) for p in PHONETIC_PAIRS}
    ),
    "zh": (
        {tuple(p) for p in CONTROL_PAIRS_ZH} | {tuple(p) for p in PHONETIC_PAIRS_ZH}
    ),
}


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
    p.add_argument("--lang", nargs="+", default=["en"], choices=["en", "zh"], help="Languages to run (default: en)")
    p.add_argument("--analysis-only", action="store_true", help="Skip inference, run analysis + plots on existing results")
    return p.parse_args()


def _run_lang(lang, model_keys, n_runs, analysis_only, generate_all_prompts_cloze, run_cloze_inference, run_analysis):
    """Run inference + analysis for one language. Returns {model: bias_df}."""
    logger.info(f"\n{'#'*60}\n  LANGUAGE: {lang.upper()}\n{'#'*60}")

    # Test mode subsets
    if lang == "zh":
        test_pairs = [CONTROL_PAIRS_ZH[0], PHONETIC_PAIRS_ZH[0], REAL_PAIRS_ZH[0]]
        test_scenarios = {k: v for k, v in list(SCENARIOS_CLOZE_ZH.items())[:1]}
    else:
        test_pairs = [CONTROL_PAIRS[0], PHONETIC_PAIRS[0], REAL_PAIRS[0]]
        test_scenarios = {k: v for k, v in list(SCENARIOS_CLOZE.items())[:1]}

    failed_models = []
    if not analysis_only:
        prompts = generate_all_prompts_cloze(lang=lang)
        logger.info(f"[{lang}] Generated {len(prompts)} cloze prompts × {n_runs} runs = {len(prompts) * n_runs} total passes")

        for mk in model_keys:
            logger.info(f"\n{'='*60}\n  [{lang}] Running: {mk}  ({MODELS[mk]})\n{'='*60}")
            try:
                run_cloze_inference(mk, MODELS[mk], prompts, n_runs=n_runs, lang=lang)
            except Exception:
                logger.exception(f"[{lang}] Inference FAILED for {mk}")
                failed_models.append(mk)

    completed_models = [mk for mk in model_keys if mk not in failed_models]
    bias_dfs = {}
    for mk in completed_models:
        logger.info(f"\n{'='*60}\n  [{lang}] Analyzing: {mk}\n{'='*60}")
        try:
            result = run_analysis(mk, lang=lang)
            bias_dfs[mk] = result["bias"]
        except Exception:
            logger.exception(f"[{lang}] Analysis FAILED for {mk}")

    return bias_dfs, failed_models


def main():
    ts = _setup_logging()
    args = parse_args()

    from local_inference import generate_all_prompts_cloze, run_cloze_inference
    from analysis import run_analysis, build_cross_model_summary
    from visualize import generate_all_plots

    model_keys = args.models or list(MODELS.keys())
    n_runs = args.n_runs

    if args.test:
        model_keys = [list(MODELS.keys())[0]]
        n_runs = 1
        logger.info("=== SMOKE TEST MODE ===")

    for mk in model_keys:
        if mk not in MODELS:
            logger.error(f"Unknown model key '{mk}'. Available: {list(MODELS.keys())}")
            sys.exit(1)

    # Run each language
    all_bias_dfs = {}  # {lang: {model: bias_df}}
    all_failed = {}
    for lang in args.lang:
        bias_dfs, failed = _run_lang(
            lang, model_keys, n_runs, args.analysis_only,
            generate_all_prompts_cloze, run_cloze_inference, run_analysis,
        )
        all_bias_dfs[lang] = bias_dfs
        all_failed[lang] = failed

    # Plots per language
    for lang, bias_dfs in all_bias_dfs.items():
        if bias_dfs:
            suffix = f"_{lang}" if len(args.lang) > 1 else ""
            logger.info(f"\n{'='*60}\n  Generating plots ({lang})\n{'='*60}")
            generate_all_plots(bias_dfs, suffix=suffix)

    # Cross-language comparison plot if both languages ran
    if len(args.lang) > 1 and all(all_bias_dfs.get(l) for l in args.lang):
        from visualize import plot_lang_comparison
        logger.info(f"\n{'='*60}\n  Generating cross-language comparison\n{'='*60}")
        plot_lang_comparison(all_bias_dfs)

    # Summary
    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")

    for lang in args.lang:
        bias_dfs = all_bias_dfs.get(lang, {})
        failed = all_failed.get(lang, [])
        filter_set = _FILTER_SETS[lang]

        if failed:
            print(f"\n  [{lang}] FAILED models: {', '.join(failed)}")

        for mk, df in bias_dfs.items():
            print(f"\n--- {mk} ({lang}) ---")
            real = df[~df.apply(
                lambda r: (r["country_1"], r["country_2"]) in filter_set,
                axis=1,
            )]
            if not real.empty:
                print(f"  Mean |bias| (real pairs): {real['bias'].abs().mean():.4f}")
                print(f"  Mean diff_fwd: {real['diff_fwd'].mean():.4f}")
                print(f"  Mean diff_rev: {real['diff_rev'].mean():.4f}")
                print(f"  N scenarios: {real['scenario'].nunique()}")

    # Save metadata
    metadata = {
        "timestamp": ts,
        "models": model_keys,
        "model_ids": {mk: MODELS[mk] for mk in model_keys},
        "n_runs": n_runs,
        "languages": args.lang,
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
