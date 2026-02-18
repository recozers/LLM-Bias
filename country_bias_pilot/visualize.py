"""Plotting functions for bias analysis results."""

import logging

import matplotlib
matplotlib.use("Agg")  # headless-safe backend — must precede pyplot import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, FICTIONAL_PAIRS

logger = logging.getLogger(__name__)

sns.set_theme(style="whitegrid", font_scale=1.1)


def _pair_label(row) -> str:
    return f"{row['country_1']} vs {row['country_2']}"


def plot_heatmap(asym_dfs: dict[str, pd.DataFrame]):
    """Heatmap: asymmetry by pair × model.

    asym_dfs: {model_name: asymmetry DataFrame from compute_asymmetry()}
    """
    records = []
    for model, df in asym_dfs.items():
        agg = df.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["model"] = model
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    combined["pair"] = combined.apply(_pair_label, axis=1)

    pivot = combined.pivot_table(index="pair", columns="model", values="asymmetry")

    # Separate fictional and real for ordering
    fict_set = {f"{a} vs {b}" for a, b in FICTIONAL_PAIRS}
    fict_labels = [p for p in pivot.index if p in fict_set]
    real_labels = [p for p in pivot.index if p not in fict_set]
    pivot = pivot.reindex(fict_labels + real_labels)

    fig, ax = plt.subplots(figsize=(max(8, len(asym_dfs) * 2.5), max(10, len(pivot) * 0.5)))
    vmax = max(0.3, pivot.abs().max().max())
    sns.heatmap(
        pivot, annot=True, fmt=".3f", center=0, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
    )
    ax.set_title("Country Preference Asymmetry by Pair × Model")
    ax.set_ylabel("")
    ax.set_xlabel("")

    # Draw line between fictional and real pairs
    if fict_labels and real_labels:
        ax.axhline(y=len(fict_labels), color="black", linewidth=2)

    plt.tight_layout()
    path = PLOTS_DIR / "heatmap_pair_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved heatmap to {path}")


def plot_bar_chart(asym_dfs: dict[str, pd.DataFrame]):
    """Bar chart: mean asymmetry per pair, grouped by model."""
    records = []
    for model, df in asym_dfs.items():
        agg = df.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["model"] = model
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    combined["pair"] = combined.apply(_pair_label, axis=1)

    # Only real pairs for clarity
    fict_set = {f"{a} vs {b}" for a, b in FICTIONAL_PAIRS}
    real = combined[~combined["pair"].isin(fict_set)]

    fig, ax = plt.subplots(figsize=(14, 6))
    pairs_order = sorted(real["pair"].unique())
    x = np.arange(len(pairs_order))
    models = sorted(real["model"].unique())
    width = 0.8 / len(models)

    for i, model in enumerate(models):
        model_data = real[real["model"] == model].set_index("pair").reindex(pairs_order)
        ax.bar(x + i * width, model_data["asymmetry"], width, label=model, alpha=0.85)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(pairs_order, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Mean Asymmetry (positive = favors Country 1)")
    ax.set_title("Mean Asymmetry per Country Pair by Model")
    ax.legend()
    plt.tight_layout()

    path = PLOTS_DIR / "bar_asymmetry_by_pair.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved bar chart to {path}")


def plot_model_scatter(asym_dfs: dict[str, pd.DataFrame], model_x: str, model_y: str):
    """Scatter: model_x asymmetry vs model_y asymmetry per pair."""
    if model_x not in asym_dfs or model_y not in asym_dfs:
        logger.warning(f"Need both {model_x} and {model_y} in results for scatter plot")
        return

    agg_x = asym_dfs[model_x].groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
    agg_y = asym_dfs[model_y].groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()

    merged = agg_x.merge(agg_y, on=["country_1", "country_2"], suffixes=(f"_{model_x}", f"_{model_y}"))
    merged["pair"] = merged.apply(_pair_label, axis=1)

    fict_set = {f"{a} vs {b}" for a, b in FICTIONAL_PAIRS}
    merged["is_fictional"] = merged["pair"].isin(fict_set)

    fig, ax = plt.subplots(figsize=(8, 8))

    for is_fict, marker, label in [(False, "o", "Real"), (True, "s", "Fictional")]:
        subset = merged[merged["is_fictional"] == is_fict]
        ax.scatter(
            subset[f"asymmetry_{model_x}"], subset[f"asymmetry_{model_y}"],
            marker=marker, s=60, label=label, alpha=0.8,
        )
        for _, row in subset.iterrows():
            ax.annotate(
                row["pair"], (row[f"asymmetry_{model_x}"], row[f"asymmetry_{model_y}"]),
                fontsize=7, alpha=0.7, textcoords="offset points", xytext=(5, 5),
            )

    lim = max(0.3, merged[[f"asymmetry_{model_x}", f"asymmetry_{model_y}"]].abs().max().max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel(f"{model_x} asymmetry")
    ax.set_ylabel(f"{model_y} asymmetry")
    ax.set_title(f"Cross-Model Asymmetry Correlation: {model_x} vs {model_y}")
    ax.legend()
    plt.tight_layout()

    path = PLOTS_DIR / f"scatter_{model_x}_vs_{model_y}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved scatter to {path}")


def generate_all_plots(asym_dfs: dict[str, pd.DataFrame]):
    """Generate all standard visualizations."""
    plot_heatmap(asym_dfs)
    plot_bar_chart(asym_dfs)

    # Scatter plots for each pair of models
    models = list(asym_dfs.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            plot_model_scatter(asym_dfs, models[i], models[j])
