"""Simple plotting for cloze bias results."""

import logging

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, CONTROL_PAIRS, PHONETIC_PAIRS

logger = logging.getLogger(__name__)

sns.set_theme(style="ticks", font_scale=1.05)

_CTRL_SET = {tuple(p) for p in CONTROL_PAIRS}
_PHON_SET = {tuple(p) for p in PHONETIC_PAIRS}


def _pair_label(row) -> str:
    return f"{row['country_1']} vs {row['country_2']}"


def _is_real(row) -> bool:
    return (row["country_1"], row["country_2"]) not in _CTRL_SET | _PHON_SET


# ── Plot 1: Heatmap of bias (diff of diffs) ──────────────────────────────

def plot_bias_heatmap(bias_dfs: dict[str, pd.DataFrame], suffix=""):
    """Heatmap of mean bias (diff_fwd - diff_rev): real country pairs × models."""
    rows = []
    for model, df in bias_dfs.items():
        real = df[df.apply(_is_real, axis=1)].copy()
        agg = real.groupby(["country_1", "country_2"])["bias"].mean().reset_index()
        agg["model"] = model
        agg["pair"] = agg.apply(_pair_label, axis=1)
        rows.append(agg)

    combined = pd.concat(rows, ignore_index=True)
    pivot = combined.pivot_table(index="pair", columns="model", values="bias")

    # Order rows by mean |bias| descending
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(6, len(pivot.columns) * 2), max(5, len(pivot) * 0.5)))
    vmax = max(0.5, pivot.abs().max().max())
    sns.heatmap(
        pivot, annot=True, fmt=".2f", center=0, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
        annot_kws={"fontsize": 9},
    )
    ax.set_ylabel("")
    ax.set_xlabel("")
    ax.set_title(
        "Country Bias (difference of differences)\n"
        "bias = (logprob_c1 − logprob_c2)_fwd − (logprob_c1 − logprob_c2)_rev\n"
        "Positive (red) = favours first-listed  |  "
        "Negative (blue) = favours second-listed",
        fontsize=10, pad=12,
    )

    plt.tight_layout()
    path = PLOTS_DIR / f"bias_heatmap{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Plot 2: Bar chart with error bars ─────────────────────────────────────

def plot_bias_bars(bias_dfs: dict[str, pd.DataFrame], suffix=""):
    """Per-pair bias bars with error bars (std across scenarios)."""
    models = list(bias_dfs.keys())
    n_models = len(models)

    rows = []
    for model, df in bias_dfs.items():
        real = df[df.apply(_is_real, axis=1)].copy()
        real["pair"] = real.apply(_pair_label, axis=1)
        agg = real.groupby("pair").agg(
            mean=("bias", "mean"),
            std=("bias", "std"),
        ).reset_index()
        agg["model"] = model
        rows.append(agg)

    combined = pd.concat(rows, ignore_index=True)

    pair_order = (
        combined.groupby("pair")["mean"].apply(lambda x: x.abs().mean())
        .sort_values(ascending=False).index.tolist()
    )

    fig, ax = plt.subplots(figsize=(max(8, len(pair_order) * 0.8), 6))
    x = np.arange(len(pair_order))
    width = 0.8 / n_models

    for i, model in enumerate(models):
        model_data = combined[combined["model"] == model].set_index("pair").reindex(pair_order)
        offset = (i - n_models / 2 + 0.5) * width
        ax.bar(
            x + offset, model_data["mean"], width,
            yerr=model_data["std"], label=model,
            capsize=2, alpha=0.8,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(pair_order, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Bias (diff of diffs)")
    ax.set_title("Country Bias by Pair", fontsize=11)
    ax.legend(fontsize=8)
    sns.despine(ax=ax)
    plt.tight_layout()

    path = PLOTS_DIR / f"bias_bars{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Plot 3: Forward vs Reverse scatter ────────────────────────────────────

def plot_fwd_vs_rev(bias_dfs: dict[str, pd.DataFrame], suffix=""):
    """Scatter of forward diff vs reverse diff per model.

    If no country preference, points land on x = y (role effect is symmetric).
    Deviation from diagonal = country bias.
    """
    models = list(bias_dfs.keys())
    n_models = len(models)
    cols = min(n_models, 3)
    rows_n = (n_models + cols - 1) // cols

    fig, axes = plt.subplots(rows_n, cols, figsize=(5 * cols, 5 * rows_n), squeeze=False)

    for idx, model in enumerate(models):
        ax = axes[idx // cols][idx % cols]
        df = bias_dfs[model]
        real = df[df.apply(_is_real, axis=1)].copy()
        real["pair"] = real.apply(_pair_label, axis=1)

        agg = real.groupby("pair").agg(
            fwd=("diff_fwd", "mean"),
            rev=("diff_rev", "mean"),
        ).reset_index()

        ax.scatter(agg["fwd"], agg["rev"], s=40, alpha=0.7)
        for _, row in agg.iterrows():
            ax.annotate(row["pair"], (row["fwd"], row["rev"]),
                        fontsize=6, alpha=0.7, textcoords="offset points", xytext=(3, 3))

        lim = max(0.5, agg[["fwd", "rev"]].abs().max().max()) * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.2, linewidth=1)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.set_xlabel("diff_fwd: logprob(c1)−logprob(c2), c1 in role A")
        ax.set_ylabel("diff_rev: logprob(c1)−logprob(c2), c2 in role A")
        ax.set_title(model, fontsize=10)
        ax.set_aspect("equal")

    for idx in range(n_models, rows_n * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(
        "Forward vs Reverse logprob difference\n"
        "Points on diagonal → no country preference (pure role effect)\n"
        "Off diagonal → genuine country bias",
        fontsize=11, y=1.02,
    )
    plt.tight_layout()
    path = PLOTS_DIR / f"fwd_vs_rev{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────

def generate_all_plots(bias_dfs: dict[str, pd.DataFrame], suffix=""):
    """Generate the core set of plots."""
    plot_bias_heatmap(bias_dfs, suffix=suffix)
    plot_bias_bars(bias_dfs, suffix=suffix)
    plot_fwd_vs_rev(bias_dfs, suffix=suffix)
