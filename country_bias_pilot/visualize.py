"""Plotting functions for bias analysis results."""

import logging

import matplotlib
matplotlib.use("Agg")  # headless-safe backend — must precede pyplot import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from config import PLOTS_DIR, CONTROL_PAIRS, PHONETIC_PAIRS, FICTIONAL_PAIRS

logger = logging.getLogger(__name__)

sns.set_theme(style="ticks", font_scale=1.0)


def _pair_label(row) -> str:
    return f"{row['country_1']} vs {row['country_2']}"


_WESTERN = ["llama3-8b", "mistral-7b", "falcon3-7b", "gemma2-9b", "gpt-oss-20b"]
_CHINESE = ["qwen2.5-7b", "deepseek-v2-lite", "deepseek-moe-16b"]
_CLR_W, _CLR_C = "#4878CF", "#D65F5F"


def plot_heatmap(asym_dfs: dict[str, pd.DataFrame], suffix: str = "", title_extra: str = ""):
    """Dot strip plot: asymmetry by pair, one dot per model.

    asym_dfs: {model_name: asymmetry DataFrame from compute_asymmetry()}
    """
    records = []
    for model, df in asym_dfs.items():
        agg = df.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["model"] = model
        agg["group"] = "Western" if model in _WESTERN else "Chinese"
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    combined["pair"] = combined.apply(_pair_label, axis=1)

    # Order: control, phonetic, real — with gaps between sections
    ctrl_set = {f"{a} vs {b}" for a, b in CONTROL_PAIRS}
    phon_set = {f"{a} vs {b}" for a, b in PHONETIC_PAIRS}
    all_pairs = sorted(combined["pair"].unique())
    ctrl_labels = [p for p in all_pairs if p in ctrl_set]
    phon_labels = [p for p in all_pairs if p in phon_set]
    real_labels = [p for p in all_pairs if p not in ctrl_set and p not in phon_set]

    # Build y-positions with gaps between sections
    y_positions = {}
    y = 0
    for label in ctrl_labels:
        y_positions[label] = y
        y += 1
    if ctrl_labels and (phon_labels or real_labels):
        y += 0.8  # gap
    for label in phon_labels:
        y_positions[label] = y
        y += 1
    if phon_labels and real_labels:
        y += 0.8  # gap
    for label in real_labels:
        y_positions[label] = y
        y += 1

    combined["y"] = combined["pair"].map(y_positions)

    fig, ax = plt.subplots(figsize=(10, max(8, y * 0.45)))

    for group, color, marker in [("Western", _CLR_W, "o"), ("Chinese", _CLR_C, "^")]:
        subset = combined[combined["group"] == group]
        ax.scatter(
            subset["asymmetry"], subset["y"],
            c=color, marker=marker, s=35, alpha=0.65,
            label=f"{group} models", zorder=3,
        )

    ax.axvline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_yticks(list(y_positions.values()))
    ax.set_yticklabels(list(y_positions.keys()), fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Asymmetry (positive = favors Country 1)")
    ax.set_title(f"Country Preference Asymmetry{title_extra}", fontsize=12, pad=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(axis="x", alpha=0.3, linewidth=0.5)
    ax.tick_params(axis="y", length=0)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    path = PLOTS_DIR / f"heatmap_pair_model{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved dot strip plot to {path}")


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
    non_real = {f"{a} vs {b}" for a, b in CONTROL_PAIRS + PHONETIC_PAIRS}
    real = combined[~combined["pair"].isin(non_real)]

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

    ctrl_set = {f"{a} vs {b}" for a, b in CONTROL_PAIRS}
    phon_set = {f"{a} vs {b}" for a, b in PHONETIC_PAIRS}

    def _pair_category(pair):
        if pair in ctrl_set:
            return "Control"
        if pair in phon_set:
            return "Phonetic"
        return "Real"

    merged["category"] = merged["pair"].apply(_pair_category)

    fig, ax = plt.subplots(figsize=(8, 8))

    for cat, marker, color in [("Real", "o", None), ("Phonetic", "s", None), ("Control", "D", "red")]:
        subset = merged[merged["category"] == cat]
        kwargs = {"marker": marker, "s": 60, "label": cat, "alpha": 0.8}
        if color:
            kwargs["color"] = color
        ax.scatter(
            subset[f"asymmetry_{model_x}"], subset[f"asymmetry_{model_y}"],
            **kwargs,
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


def plot_scenario_heatmap(asym_dfs: dict[str, pd.DataFrame]):
    """Heatmap: mean asymmetry by scenario × model (real pairs only)."""
    records = []
    for model, df in asym_dfs.items():
        non_real = {tuple(p) for p in CONTROL_PAIRS + PHONETIC_PAIRS}
        real = df[~df.apply(lambda r: (r["country_1"], r["country_2"]) in non_real, axis=1)]
        agg = real.groupby("scenario")["asymmetry"].agg(["mean", "std"]).reset_index()
        agg["model"] = model
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    pivot = combined.pivot_table(index="scenario", columns="model", values="mean")

    fig, ax = plt.subplots(figsize=(max(8, len(asym_dfs) * 2.5), 4))
    vmax = max(0.15, pivot.abs().max().max())
    sns.heatmap(
        pivot, annot=True, fmt=".3f", center=0, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
    )
    ax.set_title("Mean Asymmetry by Scenario × Model (Real Pairs Only)")
    ax.set_ylabel("")
    plt.tight_layout()
    path = PLOTS_DIR / "heatmap_scenario_model.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved scenario heatmap to {path}")


def plot_phonetic_bias(asym_dfs: dict[str, pd.DataFrame]):
    """Bar chart showing phonetic pair asymmetries — do European-sounding names get favored?"""
    records = []
    phon_set = {tuple(p) for p in PHONETIC_PAIRS}
    for model, df in asym_dfs.items():
        phon = df[df.apply(lambda r: (r["country_1"], r["country_2"]) in phon_set, axis=1)]
        agg = phon.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["model"] = model
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    combined["pair"] = combined.apply(_pair_label, axis=1)

    # Separate within-region vs cross-region
    cross = {"Aeloria vs Zembala", "Aeloria vs Junwei", "Zembala vs Junwei"}
    combined["type"] = combined["pair"].apply(lambda p: "Cross-region" if p in cross else "Within-region")

    fig, ax = plt.subplots(figsize=(12, 5))
    pairs_order = ["Aeloria vs Brevnia", "Zembala vs Kotundi", "Junwei vs Khemara",
                    "Aeloria vs Zembala", "Aeloria vs Junwei", "Zembala vs Junwei"]
    pairs_order = [p for p in pairs_order if p in combined["pair"].values]

    x = np.arange(len(pairs_order))
    models = sorted(combined["model"].unique())
    width = 0.8 / len(models)
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))

    for i, model in enumerate(models):
        mdata = combined[combined["model"] == model].set_index("pair").reindex(pairs_order)
        ax.bar(x + i * width, mdata["asymmetry"], width, label=model, color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(pairs_order, rotation=30, ha="right")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(2.5, color="gray", linewidth=1, linestyle="--", alpha=0.5)
    ax.text(1.0, ax.get_ylim()[1] * 0.9, "Within-region", ha="center", fontsize=9, style="italic")
    ax.text(4.0, ax.get_ylim()[1] * 0.9, "Cross-region", ha="center", fontsize=9, style="italic")
    ax.set_ylabel("Asymmetry (positive = favors first name)")
    ax.set_title("Phonetic Bias: Do European-Sounding Names Get Favored?")
    ax.legend(fontsize=7, ncol=4)
    plt.tight_layout()

    path = PLOTS_DIR / "phonetic_bias.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved phonetic bias chart to {path}")


def plot_western_vs_chinese(asym_dfs: dict[str, pd.DataFrame], suffix: str = "", title_extra: str = ""):
    """Forest plot: Western vs Chinese model group means for real pairs."""
    records = []
    non_real = {tuple(p) for p in CONTROL_PAIRS + PHONETIC_PAIRS}
    for model, df in asym_dfs.items():
        real = df[~df.apply(lambda r: (r["country_1"], r["country_2"]) in non_real, axis=1)]
        agg = real.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["model"] = model
        agg["group"] = "Western" if model in _WESTERN else "Chinese" if model in _CHINESE else "Other"
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    combined["pair"] = combined.apply(_pair_label, axis=1)

    # Aggregate by group
    group_agg = combined.groupby(["pair", "group"])["asymmetry"].agg(["mean", "std"]).reset_index()

    # Order pairs by absolute difference between Western and Chinese means
    pivot_means = group_agg.pivot(index="pair", columns="group", values="mean").fillna(0)
    if "Western" in pivot_means.columns and "Chinese" in pivot_means.columns:
        order_key = (pivot_means["Western"] - pivot_means["Chinese"]).abs()
    else:
        order_key = pivot_means.iloc[:, 0].abs()
    pairs_order = order_key.sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(5, len(pairs_order) * 0.5)))
    y_offset = 0.12  # vertical offset so dots don't overlap

    for group, color, marker, dy in [
        ("Western", _CLR_W, "o", -y_offset),
        ("Chinese", _CLR_C, "^", y_offset),
    ]:
        gdata = group_agg[group_agg["group"] == group].set_index("pair").reindex(pairs_order)
        y = np.arange(len(pairs_order)) + dy
        ax.errorbar(
            gdata["mean"], y,
            xerr=gdata["std"], fmt="none",
            ecolor=color, elinewidth=1.2, capsize=3, capthick=1.2, alpha=0.6, zorder=2,
        )
        ax.scatter(
            gdata["mean"], y,
            c=color, marker=marker, s=50, label=f"{group} models",
            zorder=3, edgecolors="white", linewidths=0.5,
        )
        # Connect Western and Chinese dots for each pair
        for i, pair in enumerate(pairs_order):
            w_val = group_agg[(group_agg["pair"] == pair) & (group_agg["group"] == "Western")]["mean"]
            c_val = group_agg[(group_agg["pair"] == pair) & (group_agg["group"] == "Chinese")]["mean"]
            if not w_val.empty and not c_val.empty:
                ax.plot(
                    [w_val.values[0], c_val.values[0]],
                    [i - y_offset, i + y_offset],
                    color="gray", linewidth=0.6, alpha=0.4, zorder=1,
                )

    ax.axvline(0, color="black", linewidth=0.8, zorder=1)
    ax.set_yticks(np.arange(len(pairs_order)))
    ax.set_yticklabels(pairs_order, fontsize=9)
    ax.set_xlabel("Mean Asymmetry (positive = favors Country 1)")
    ax.set_title(f"Western vs Chinese Model Bias{title_extra}", fontsize=12, pad=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.tick_params(axis="y", length=0)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    path = PLOTS_DIR / f"western_vs_chinese{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved forest plot to {path}")


def plot_control_residuals(asym_dfs: dict[str, pd.DataFrame], suffix: str = "", title_extra: str = ""):
    """Show control pair residual asymmetries as a validation check."""
    ctrl_set = {tuple(p) for p in CONTROL_PAIRS}
    records = []
    for model, df in asym_dfs.items():
        ctrl = df[df.apply(lambda r: (r["country_1"], r["country_2"]) in ctrl_set, axis=1)]
        for _, row in ctrl.iterrows():
            records.append({
                "model": model,
                "pair": f"{row['country_1']} vs {row['country_2']}",
                "scenario": row["scenario"],
                "asymmetry": row["asymmetry"],
            })
    combined = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 5))
    sns.stripplot(data=combined, x="model", y="asymmetry", hue="pair",
                  dodge=True, alpha=0.7, jitter=True, ax=ax, size=5)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axhspan(-0.1, 0.1, alpha=0.1, color="green", label="±0.1 target zone")
    ax.set_ylabel("Residual Asymmetry (should be ~0)")
    ax.set_title(f"Control Pair Residual Asymmetries (Diagnostic){title_extra}")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    path = PLOTS_DIR / f"control_residuals{suffix}.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved control residuals to {path}")


def plot_pair_consistency(asym_dfs: dict[str, pd.DataFrame]):
    """Dot plot: each model's asymmetry for each real pair, showing agreement/disagreement."""
    non_real = {tuple(p) for p in CONTROL_PAIRS + PHONETIC_PAIRS}
    records = []
    for model, df in asym_dfs.items():
        real = df[~df.apply(lambda r: (r["country_1"], r["country_2"]) in non_real, axis=1)]
        agg = real.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["model"] = model
        records.append(agg)
    combined = pd.concat(records, ignore_index=True)
    combined["pair"] = combined.apply(_pair_label, axis=1)

    # Order pairs by absolute mean asymmetry
    pair_means = combined.groupby("pair")["asymmetry"].mean().sort_values(key=abs, ascending=True)
    pairs_order = pair_means.index.tolist()

    fig, ax = plt.subplots(figsize=(10, 7))
    y = np.arange(len(pairs_order))

    western = ["llama3-8b", "mistral-7b", "falcon3-7b", "gemma2-9b", "gpt-oss-20b"]
    chinese = ["qwen2.5-7b", "deepseek-v2-lite", "deepseek-moe-16b"]

    for model in sorted(combined["model"].unique()):
        mdata = combined[combined["model"] == model].set_index("pair").reindex(pairs_order)
        marker = "o" if model in western else "^"
        color = "#2166ac" if model in western else "#b2182b"
        ax.scatter(mdata["asymmetry"], y, marker=marker, s=40, alpha=0.6,
                   color=color, label=model)

    # Cross-model mean
    for i, pair in enumerate(pairs_order):
        mean_val = pair_means[pair]
        ax.plot(mean_val, i, "k|", markersize=15, markeredgewidth=2)

    ax.set_yticks(y)
    ax.set_yticklabels(pairs_order)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Asymmetry (positive = favors Country 1)")
    ax.set_title("Per-Model Asymmetry for Each Real Pair\n(○ Western, △ Chinese, | cross-model mean)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    path = PLOTS_DIR / "pair_consistency.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved pair consistency to {path}")


def plot_mcf_vs_cloze(
    mcf_asym_dfs: dict[str, pd.DataFrame],
    cloze_asym_dfs: dict[str, pd.DataFrame],
):
    """Scatter: MCF asymmetry vs cloze asymmetry per pair, per model.

    Each point is one (pair, model) combination. Points are colored by model.
    """
    # Find models present in both
    shared_models = sorted(set(mcf_asym_dfs) & set(cloze_asym_dfs))
    if not shared_models:
        logger.warning("No shared models between MCF and cloze results — skipping comparison plot")
        return

    records = []
    for model in shared_models:
        mcf_agg = mcf_asym_dfs[model].groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        cloze_agg = cloze_asym_dfs[model].groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        merged = mcf_agg.merge(cloze_agg, on=["country_1", "country_2"], suffixes=("_mcf", "_cloze"))
        merged["model"] = model
        merged["pair"] = merged.apply(_pair_label, axis=1)

        ctrl_set = {f"{a} vs {b}" for a, b in CONTROL_PAIRS}
        phon_set = {f"{a} vs {b}" for a, b in PHONETIC_PAIRS}
        merged["category"] = merged["pair"].apply(
            lambda p: "Control" if p in ctrl_set else ("Phonetic" if p in phon_set else "Real")
        )
        records.append(merged)

    combined = pd.concat(records, ignore_index=True)

    fig, ax = plt.subplots(figsize=(9, 9))
    colors = plt.cm.tab10(np.linspace(0, 1, len(shared_models)))

    for idx, model in enumerate(shared_models):
        subset = combined[combined["model"] == model]
        for cat, marker in [("Real", "o"), ("Phonetic", "s"), ("Control", "D")]:
            cat_sub = subset[subset["category"] == cat]
            if cat_sub.empty:
                continue
            label = f"{model} ({cat})" if cat != "Real" else model
            ax.scatter(
                cat_sub["asymmetry_mcf"], cat_sub["asymmetry_cloze"],
                marker=marker, s=40, alpha=0.7, color=colors[idx],
                label=label if cat == "Real" else None,
            )

    lim = max(0.3, combined[["asymmetry_mcf", "asymmetry_cloze"]].abs().max().max()) * 1.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.3, label="y=x")
    ax.set_xlabel("MCF Asymmetry")
    ax.set_ylabel("Cloze Asymmetry")
    ax.set_title("MCF vs Cloze Formulation Asymmetry Comparison")
    ax.legend(fontsize=7, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()

    path = PLOTS_DIR / "mcf_vs_cloze.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    logger.info(f"Saved MCF vs cloze comparison to {path}")


def generate_all_plots(asym_dfs: dict[str, pd.DataFrame]):
    """Generate all standard visualizations."""
    plot_heatmap(asym_dfs)
    plot_bar_chart(asym_dfs)
    plot_scenario_heatmap(asym_dfs)
    plot_phonetic_bias(asym_dfs)
    plot_western_vs_chinese(asym_dfs)
    plot_control_residuals(asym_dfs)
    plot_pair_consistency(asym_dfs)

    # Scatter plots for each pair of models
    models = list(asym_dfs.keys())
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            plot_model_scatter(asym_dfs, models[i], models[j])
