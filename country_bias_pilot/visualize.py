"""Plotting functions for cloze bias analysis results."""

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

_WESTERN = ["llama3-8b", "mistral-7b", "gemma2-9b", "gpt-oss-20b"]
_CHINESE = ["qwen2.5-7b", "deepseek-v2-lite", "deepseek-moe-16b"]
_OTHER = ["falcon3-7b"]  # TII, UAE
_CLR_W, _CLR_C, _CLR_O = "#2166ac", "#b2182b", "#1b7837"

_CTRL_SET = {tuple(p) for p in CONTROL_PAIRS}
_PHON_SET = {tuple(p) for p in PHONETIC_PAIRS}


def _pair_label(row) -> str:
    return f"{row['country_1']} vs {row['country_2']}"


def _is_real(row) -> bool:
    return (row["country_1"], row["country_2"]) not in _CTRL_SET | _PHON_SET


def _build_combined(asym_dfs, real_only=False):
    """Stack all models into one DataFrame with pair labels and group tags."""
    rows = []
    for model, df in asym_dfs.items():
        sub = df.copy()
        if real_only:
            sub = sub[sub.apply(_is_real, axis=1)]
        agg = sub.groupby(["country_1", "country_2"])["asymmetry"].agg(
            ["mean", "std", "count"]
        ).reset_index()
        agg.columns = ["country_1", "country_2", "asymmetry", "std", "n"]
        agg["model"] = model
        if model in _WESTERN:
            agg["group"] = "Western"
        elif model in _CHINESE:
            agg["group"] = "Chinese"
        else:
            agg["group"] = "Other"
        agg["pair"] = agg.apply(_pair_label, axis=1)
        rows.append(agg)
    return pd.concat(rows, ignore_index=True)


# ── Plot 1: Main heatmap ─────────────────────────────────────────────────

def plot_main_heatmap(asym_dfs, suffix=""):
    """Heatmap of mean asymmetry: real country pairs × models.

    Rows ordered by absolute cross-model mean so the strongest
    biases appear at the top.
    """
    combined = _build_combined(asym_dfs, real_only=True)
    pivot = combined.pivot_table(index="pair", columns="model", values="asymmetry")

    # Order rows by mean |asymmetry| descending
    pivot = pivot.loc[pivot.abs().mean(axis=1).sort_values(ascending=False).index]

    # Order columns: western, other, chinese
    col_order = [m for m in _WESTERN + _OTHER + _CHINESE if m in pivot.columns]
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(10, 7))
    vmax = max(0.3, pivot.abs().max().max())
    sns.heatmap(
        pivot, annot=True, fmt=".2f", center=0, cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, linewidths=0.5, ax=ax,
        annot_kws={"fontsize": 9},
    )
    # Move x-tick labels to bottom and add group annotations there
    ax.xaxis.tick_bottom()
    ax.xaxis.set_label_position("bottom")
    ax.set_ylabel("")
    ax.set_xlabel("")

    ax.set_title(
        "Country Bias in Cloze Completions\n"
        "Each cell averages 16 scenario paraphrases × 2 role orderings\n"
        "Positive (red) = favours first-listed country  |  "
        "Negative (blue) = favours second-listed country",
        fontsize=11, pad=12,
    )

    # Annotate column groups below the x-tick labels
    n_rows = len(pivot)
    label_y = n_rows + 1.8
    for group_list, label, color in [
        (_WESTERN, "Western-origin", _CLR_W),
        (_OTHER, "Gulf-origin", _CLR_O),
        (_CHINESE, "Chinese-origin", _CLR_C),
    ]:
        cols = [i for i, m in enumerate(col_order) if m in group_list]
        if cols:
            ax.text(
                np.mean(cols) + 0.5, label_y, label,
                ha="center", fontsize=9, color=color, fontweight="bold",
            )

    plt.tight_layout()
    path = PLOTS_DIR / f"cloze_heatmap{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Plot 2: Western vs Chinese forest plot ────────────────────────────────

def plot_western_vs_chinese(asym_dfs, suffix=""):
    """Forest plot comparing Western and Chinese model group means.

    Each real pair gets one row.  Blue dot = Western mean, red triangle =
    Chinese mean, with ±1 SD error bars across models in each group.
    Pairs ordered by the gap between the two groups.
    """
    combined = _build_combined(asym_dfs, real_only=True)

    group_agg = (
        combined.groupby(["pair", "group"])["asymmetry"]
        .agg(["mean", "std"])
        .reset_index()
    )

    # Order by |Western mean − Chinese mean|
    pivot_m = group_agg.pivot(index="pair", columns="group", values="mean").fillna(0)
    if "Western" in pivot_m.columns and "Chinese" in pivot_m.columns:
        gap = (pivot_m["Western"] - pivot_m["Chinese"]).abs()
    else:
        gap = pivot_m.iloc[:, 0].abs()
    pairs_order = gap.sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(10, max(5, len(pairs_order) * 0.5)))
    dy = 0.13

    for group, color, marker, offset in [
        ("Western", _CLR_W, "o", -dy),
        ("Other", _CLR_O, "s", 0),
        ("Chinese", _CLR_C, "^", dy),
    ]:
        gdata = group_agg[group_agg["group"] == group].set_index("pair").reindex(pairs_order)
        if gdata["mean"].isna().all():
            continue
        y = np.arange(len(pairs_order)) + offset
        label = {"Western": "Western-origin", "Chinese": "Chinese-origin",
                 "Other": "Gulf-origin (Falcon)"}[group]
        ax.errorbar(
            gdata["mean"], y,
            xerr=gdata["std"], fmt="none",
            ecolor=color, elinewidth=1.2, capsize=3, capthick=1.2, alpha=0.55, zorder=2,
        )
        ax.scatter(
            gdata["mean"], y,
            c=color, marker=marker, s=55, label=label,
            zorder=3, edgecolors="white", linewidths=0.5,
        )

    # Connect Western and Chinese dots
    for i, pair in enumerate(pairs_order):
        w = group_agg[(group_agg["pair"] == pair) & (group_agg["group"] == "Western")]["mean"]
        c = group_agg[(group_agg["pair"] == pair) & (group_agg["group"] == "Chinese")]["mean"]
        if not w.empty and not c.empty:
            ax.plot(
                [w.values[0], c.values[0]], [i - dy, i + dy],
                color="gray", linewidth=0.5, alpha=0.4, zorder=1,
            )

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(np.arange(len(pairs_order)))
    ax.set_yticklabels(pairs_order, fontsize=9)
    ax.set_xlabel(
        "Mean asymmetry\n"
        "← favours second-listed country          favours first-listed country →",
        fontsize=10,
    )
    ax.set_title(
        "Do Western and Chinese LLMs Disagree on Who Is Justified?\n"
        "Dots = group mean across models  |  Bars = ±1 SD across models in group\n"
        "Ordered by gap between Western and Chinese groups",
        fontsize=11, pad=12,
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.tick_params(axis="y", length=0)
    sns.despine(ax=ax, left=True)
    plt.tight_layout()

    path = PLOTS_DIR / f"western_vs_chinese{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Plot 3: Per-pair boxplots showing paraphrase robustness ───────────────

def plot_paraphrase_robustness(asym_dfs, suffix=""):
    """Box plots for each model showing asymmetry spread across 16 paraphrases.

    Demonstrates that the bias signal is stable across different wordings,
    not an artefact of one particular prompt.
    """
    # Pick 3 representative models: one Western, one Chinese, one neutral
    candidates = ["llama3-8b", "qwen2.5-7b", "gemma2-9b"]
    models = [m for m in candidates if m in asym_dfs]
    if not models:
        models = list(asym_dfs.keys())[:3]

    fig, axes = plt.subplots(1, len(models), figsize=(6 * len(models), 7), sharey=False)
    if len(models) == 1:
        axes = [axes]

    for ax, model in zip(axes, models):
        df = asym_dfs[model]
        real = df[df.apply(_is_real, axis=1)].copy()
        real["pair"] = real.apply(_pair_label, axis=1)

        pair_order = real.groupby("pair")["asymmetry"].mean().sort_values().index

        if model in _CHINESE:
            color, group_tag = _CLR_C, "Chinese"
        elif model in _OTHER:
            color, group_tag = _CLR_O, "Gulf"
        else:
            color, group_tag = _CLR_W, "Western"
        sns.boxplot(
            data=real, y="pair", x="asymmetry", order=pair_order,
            ax=ax, color=color, fliersize=2, linewidth=0.8,
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{model}  ({group_tag})", fontsize=10, color=color)
        ax.set_ylabel("")
        ax.set_xlabel(
            "← favours 2nd country    Asymmetry    favours 1st country →",
            fontsize=8,
        )
        ax.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "Bias Is Robust Across 16 Scenario Paraphrases\n"
        "Each box = distribution of asymmetry across paraphrases for one country pair",
        fontsize=12, y=1.02,
    )
    plt.tight_layout()
    path = PLOTS_DIR / f"paraphrase_robustness{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Plot 4: Llama vs Qwen scatter ────────────────────────────────────────

def plot_origin_scatter(asym_dfs, model_w="llama3-8b", model_c="qwen2.5-7b", suffix=""):
    """Scatter of one Western vs one Chinese model's asymmetries.

    Points off the diagonal reveal where training-data origin
    drives disagreement.  Annotated with pair labels.
    """
    if model_w not in asym_dfs or model_c not in asym_dfs:
        logger.warning(f"Need both {model_w} and {model_c} for scatter")
        return

    def _agg(model):
        df = asym_dfs[model]
        agg = df.groupby(["country_1", "country_2"])["asymmetry"].mean().reset_index()
        agg["pair"] = agg.apply(_pair_label, axis=1)
        return agg.set_index("pair")["asymmetry"]

    w = _agg(model_w)
    c = _agg(model_c)
    merged = pd.DataFrame({"western": w, "chinese": c}).dropna()

    ctrl_labels = {f"{a} vs {b}" for a, b in CONTROL_PAIRS}
    phon_labels = {f"{a} vs {b}" for a, b in PHONETIC_PAIRS}

    fig, ax = plt.subplots(figsize=(8, 8))

    for pair in merged.index:
        x, y = merged.loc[pair, "western"], merged.loc[pair, "chinese"]
        if pair in ctrl_labels:
            marker, color, alpha = "D", "gray", 0.5
        elif pair in phon_labels:
            marker, color, alpha = "s", "#e08214", 0.7
        else:
            marker, color, alpha = "o", "#542788", 0.8
        ax.scatter(x, y, marker=marker, s=55, color=color, alpha=alpha, zorder=3)
        ax.annotate(
            pair, (x, y), fontsize=7, alpha=0.75,
            textcoords="offset points", xytext=(5, 5),
        )

    lim = max(0.4, merged.abs().max().max()) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.25, linewidth=1)

    # Shade quadrants of disagreement
    ax.fill_between(
        [0, lim], 0, -lim, alpha=0.04, color=_CLR_W,
    )
    ax.fill_between(
        [-lim, 0], 0, lim, alpha=0.04, color=_CLR_C,
    )
    ax.text(lim * 0.55, -lim * 0.85,
            f"{model_w} favours\n1st country, {model_c}\nfavours 2nd",
            fontsize=7.5, ha="center", color=_CLR_W, alpha=0.6)
    ax.text(-lim * 0.55, lim * 0.85,
            f"{model_c} favours\n1st country, {model_w}\nfavours 2nd",
            fontsize=7.5, ha="center", color=_CLR_C, alpha=0.6)

    ax.set_xlabel(
        f"{model_w} asymmetry  (Western-developed)\n"
        "← favours 2nd country          favours 1st country →",
        fontsize=10, color=_CLR_W,
    )
    ax.set_ylabel(
        f"{model_c} asymmetry  (Chinese-developed)\n"
        "← favours 2nd country          favours 1st country →",
        fontsize=10, color=_CLR_C,
    )
    ax.set_title(
        "Training-Origin Bias: Where Do Models Disagree?\n"
        "Points on the diagonal = models agree  |  Off diagonal = divergent preferences\n"
        "o = Real pairs   s = Phonetic (fictional)   d = Controls",
        fontsize=11, pad=12,
    )
    sns.despine(ax=ax)
    plt.tight_layout()

    path = PLOTS_DIR / f"origin_scatter_{model_w}_vs_{model_c}{suffix}.png"
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {path}")


# ── Entry point ───────────────────────────────────────────────────────────

def generate_all_plots(asym_dfs, suffix=""):
    """Generate the core set of publication-quality plots."""
    plot_main_heatmap(asym_dfs, suffix=suffix)
    plot_western_vs_chinese(asym_dfs, suffix=suffix)
    plot_paraphrase_robustness(asym_dfs, suffix=suffix)

    # Scatter for most interesting Western vs Chinese pair
    w_models = [m for m in _WESTERN if m in asym_dfs]
    c_models = [m for m in _CHINESE if m in asym_dfs]
    if w_models and c_models:
        # Prefer llama vs qwen
        mw = "llama3-8b" if "llama3-8b" in asym_dfs else w_models[0]
        mc = "qwen2.5-7b" if "qwen2.5-7b" in asym_dfs else c_models[0]
        plot_origin_scatter(asym_dfs, mw, mc, suffix=suffix)
