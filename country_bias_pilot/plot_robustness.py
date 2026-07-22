#!/usr/bin/env python3
"""Figure 6 — Robustness / ablation panel.

Three sub-panels:
  A  Hedge ablation          — does the hedge prefix drive the bias?
  B  Phrasing robustness     — does the result depend on the exact MCQ wording?
  C  Cross-prompting 2x2     — is the language effect from the scenario or the question?
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_main_figure import (
    RESULTS, COHERENT_SCENARIOS, AAAI_MODE, PLOT_OUTPUT_DIR, PLOT_DPI,
    format_aaai_figure,
)

MODELS = [
    ("Qwen 2.5 7B", "qwen2.5-7b-inst", "CN", "#DD8452"),
    ("Mistral 7B",  "mistral-7b-inst", "FR", "#4C72B0"),
]


def china_fav(dir_name, model_key):
    f = RESULTS / dir_name / f"{model_key}_raw.csv"
    if not f.exists():
        return np.nan
    df = pd.read_csv(f)
    df = df[df.question == "justified"]
    df = df[df.scenario.isin(COHERENT_SCENARIOS)] if "scenario" in df.columns else df
    vals = []
    for _, r in df.iterrows():
        if r.country_1 == "China":
            vals.append(r.bias)
        elif r.country_2 == "China":
            vals.append(-r.bias)
    return float(np.mean(vals)) if vals else np.nan


def panel_hedge(ax):
    x = np.arange(len(MODELS))
    w = 0.32
    baselines = [china_fav("gpu_bias", m) for _, m, _, _ in MODELS]
    nohedge = [china_fav("gpu_bias_nohedge", m) for _, m, _, _ in MODELS]
    colors = [c for _, _, _, c in MODELS]

    b1 = ax.bar(x - w/2, baselines, w, color=colors, alpha=0.65,
                edgecolor="black", linewidth=0.6, label="with hedge")
    b2 = ax.bar(x + w/2, nohedge, w, color=colors, alpha=1.0,
                edgecolor="black", linewidth=0.6, hatch="//",
                label="hedge removed")

    for rect, v in zip(list(b1) + list(b2), baselines + nohedge):
        if np.isnan(v):
            continue
        sign = "+" if v >= 0 else ""
        pad = 0.15 if v >= 0 else -0.15
        ax.text(rect.get_x() + rect.get_width() / 2, v + pad,
                f"{sign}{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top",
                fontsize=8, fontweight="bold", color="#222")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{fam}  [{m}]" for fam, _, m, _ in MODELS], fontsize=9)
    ax.set_ylabel("China favourability (log-odds)", fontsize=9)
    ax.set_title("A  ·  Hedge ablation",
                 fontsize=11, fontweight="bold", loc="left", pad=6)
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def panel_phrasing(ax):
    phrasings = [
        ("baseline", "gpu_bias",              "Which country acted\nmore justifiably?"),
        ("alt1",     "gpu_bias_phrasing_alt1", "Whose action was\nmore justified?"),
        ("alt2",     "gpu_bias_phrasing_alt2", "Which country behaved\nmore reasonably?"),
        ("alt3",     "gpu_bias_phrasing_alt3", "Who was in\nthe right?"),
    ]
    x = np.arange(len(phrasings))
    w = 0.38
    for i, (fam, model_key, bloc, color) in enumerate(MODELS):
        vals = [china_fav(d, model_key) for _, d, _ in phrasings]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=color, alpha=0.88,
                       edgecolor="black", linewidth=0.5,
                       hatch="//" if i else "",
                       label=f"{fam} [{bloc}]")
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                continue
            sign = "+" if v >= 0 else ""
            pad = 0.12 if v >= 0 else -0.12
            ax.text(rect.get_x() + rect.get_width() / 2, v + pad,
                    f"{sign}{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color="#222")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    labels = (["Baseline", "Whose action?", "More reasonable?", "Who was right?"]
              if AAAI_MODE else [p[2] for p in phrasings])
    ax.set_xticklabels(labels, fontsize=7.5,
                        rotation=0 if AAAI_MODE else 15, ha="center" if AAAI_MODE else "right")
    ax.set_ylabel("China favourability (log-odds)", fontsize=9)
    ax.set_title("B  ·  Phrasing robustness",
                 fontsize=11, fontweight="bold", loc="left", pad=6)
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def panel_cross(ax):
    conditions = [
        ("EN scen / EN Q\n(baseline)", "gpu_bias"),
        ("EN scen / ZH Q",              "gpu_bias_cross_en_qzh"),
        ("ZH scen / EN Q",              "gpu_bias_cross_zh_qen"),
        ("ZH scen / ZH Q\n(main ZH)",   "gpu_bias_zh"),
    ]
    x = np.arange(len(conditions))
    w = 0.38
    for i, (fam, model_key, bloc, color) in enumerate(MODELS):
        vals = [china_fav(d, model_key) for _, d in conditions]
        offset = (i - 0.5) * w
        bars = ax.bar(x + offset, vals, w, color=color, alpha=0.88,
                       edgecolor="black", linewidth=0.5,
                       hatch="//" if i else "",
                       label=f"{fam} [{bloc}]")
        for rect, v in zip(bars, vals):
            if np.isnan(v):
                continue
            sign = "+" if v >= 0 else ""
            pad = 0.13 if v >= 0 else -0.13
            ax.text(rect.get_x() + rect.get_width() / 2, v + pad,
                    f"{sign}{v:.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7.5, color="#222")

    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in conditions], fontsize=8)
    ax.set_ylabel("China favourability (log-odds)", fontsize=9)
    ax.set_title("C  ·  Cross-prompting factorial",
                 fontsize=11, fontweight="bold", loc="left", pad=6)
    ax.legend(fontsize=8, loc="best", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def main():
    if AAAI_MODE:
        fig, axes = plt.subplots(3, 1, figsize=(7.0, 7.6),
                                 gridspec_kw={"hspace": 0.58})
    else:
        fig, axes = plt.subplots(1, 3, figsize=(17, 5.8),
                                 gridspec_kw={"wspace": 0.28})
    panel_hedge(axes[0])
    panel_phrasing(axes[1])
    panel_cross(axes[2])
    if AAAI_MODE:
        axes[1].get_legend().remove()
        axes[2].get_legend().remove()

    title = (
        "Robustness checks for hedge, phrasing, and scenario language"
        if AAAI_MODE else
        "Robustness and mechanism: the bias is not driven by the hedge, "
        "is stable across phrasings, and is primarily triggered by the scenario language"
    )
    fig.suptitle(
        title,
        fontsize=12, fontweight="bold", y=0.98 if AAAI_MODE else 1.02,
    )
    footer = (
        "Qwen and Mistral; 31-scenario exploratory subset."
        if AAAI_MODE else
        "Qwen 2.5 7B-inst (Alibaba) and Mistral 7B-inst (Mistral AI). "
        "31-scenario coherence subset; variant-sum scoring."
    )
    fig.text(0.5, 0.015 if AAAI_MODE else -0.04, footer,
             ha="center", fontsize=8, color="#555", style="italic")

    format_aaai_figure(fig)
    if AAAI_MODE:
        fig.subplots_adjust(left=0.13, right=0.98, top=0.92, bottom=0.08,
                            hspace=0.58)
    out = PLOT_OUTPUT_DIR / "figure6_robustness.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=PLOT_DPI,
                bbox_inches=None if AAAI_MODE else "tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
