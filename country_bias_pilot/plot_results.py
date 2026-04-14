#!/usr/bin/env python3
"""Plot country bias results, excluding pairs that don't flip between justified/unjustified."""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

RESULTS_DIR = Path(__file__).resolve().parent / "results" / "gpu_bias"

MODELS = {
    "mistral-7b":      ("Mistral 7B", "base"),
    "qwen2.5-7b":      ("Qwen 2.5 7B", "base"),
    "gemma4-8b":        ("Gemma4 8B", "base"),
    "llama3-8b":        ("LLaMA3 8B", "base"),
    "mistral-7b-inst":  ("Mistral 7B Inst", "instruct"),
    "llama3-8b-inst":   ("LLaMA3 8B Inst", "instruct"),
    "qwen2.5-7b-inst":  ("Qwen 2.5 7B Inst", "instruct"),
    "gemma4-8b-it":     ("Gemma4 8B IT", "instruct"),
}

COUNTRIES = [
    "USA", "France", "China", "Japan",
    "Venezuela", "Canada", "Australia", "Indonesia",
]

WESTERN = {"USA", "France", "Canada", "Australia"}
NON_WESTERN = {"China", "Japan", "Venezuela", "Indonesia"}

FAMILY_PAIRS = [
    ("mistral-7b", "mistral-7b-inst"),
    ("llama3-8b", "llama3-8b-inst"),
    ("qwen2.5-7b", "qwen2.5-7b-inst"),
    ("gemma4-8b", "gemma4-8b-it"),
]


def _scenario_type(name):
    return re.sub(r"(_def)?_v\d+$", "", name)


def load_and_filter(model_key):
    """Load raw CSV, keep only pairs where justified and unjustified bias flip sign."""
    df = pd.read_csv(RESULTS_DIR / f"{model_key}_raw.csv")
    df["scenario_type"] = df["scenario"].apply(_scenario_type)

    # Get mean bias per pair per question (averaging over scenario types)
    pair_q = (
        df.groupby(["country_1", "country_2", "question"])
        .apply(lambda g: g.groupby("scenario_type")["bias"].mean().mean())
        .reset_index(name="mean_bias")
    )

    # Pivot to get justified and unjustified side by side
    pivot = pair_q.pivot_table(
        index=["country_1", "country_2"], columns="question", values="mean_bias"
    )

    # Keep only pairs that flip sign
    if "justified" in pivot.columns and "unjustified" in pivot.columns:
        flipped = pivot[pivot["justified"] * pivot["unjustified"] < 0]
        keep_pairs = set(flipped.index)
    else:
        keep_pairs = set(pivot.index)

    # Filter raw data to only flipped pairs
    df_filtered = df[
        df.apply(lambda r: (r["country_1"], r["country_2"]) in keep_pairs, axis=1)
    ]

    return df_filtered, len(keep_pairs), len(pivot)


def compute_country_favour(df):
    """Compute per-country mean favour from justified question only."""
    jdf = df[df["question"] == "justified"]

    # Per scenario-type cluster means for each pair
    country_biases = {}
    for country in COUNTRIES:
        type_biases = {}
        for (c1, c2), grp in jdf.groupby(["country_1", "country_2"]):
            if country not in (c1, c2):
                continue
            sign = 1.0 if country == c1 else -1.0
            for stype, sg in grp.groupby("scenario_type"):
                type_biases.setdefault(stype, []).append(sign * sg["bias"].mean())

        if type_biases:
            cluster_means = np.array([np.mean(v) for v in type_biases.values()])
            country_biases[country] = cluster_means.mean()
        else:
            country_biases[country] = 0.0

    return country_biases


def plot_heatmap(all_favour, all_meta):
    """Heatmap: countries x models, color = logprob favour."""
    base_keys = [k for k, (_, t) in MODELS.items() if t == "base"]
    inst_keys = [k for k, (_, t) in MODELS.items() if t == "instruct"]
    order = base_keys + inst_keys
    labels = [MODELS[k][0] for k in order]

    matrix = np.array([[all_favour[k].get(c, 0) for k in order] for c in COUNTRIES])

    vmax = max(abs(matrix.min()), abs(matrix.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, cmap="RdBu_r", norm=norm, aspect="auto")

    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_yticks(range(len(COUNTRIES)))
    ax.set_yticklabels(COUNTRIES, fontsize=11)

    # Add values
    for i in range(len(COUNTRIES)):
        for j in range(len(order)):
            val = matrix[i, j]
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:+.2f}", ha="center", va="center",
                    fontsize=8, color=color)

    # Divider between base and instruct
    ax.axvline(len(base_keys) - 0.5, color="black", linewidth=2)
    ax.text(len(base_keys) / 2 - 0.5, -1.2, "BASE", ha="center", fontsize=11, fontweight="bold")
    ax.text(len(base_keys) + len(inst_keys) / 2 - 0.5, -1.2, "INSTRUCT",
            ha="center", fontsize=11, fontweight="bold")

    # Pair counts
    for j, k in enumerate(order):
        kept, total = all_meta[k]
        ax.text(j, len(COUNTRIES) + 0.3, f"{kept}/{total}", ha="center",
                fontsize=7, color="gray")

    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.set_label("Mean logprob favour (justified)", fontsize=10)

    ax.set_title("Country Bias Across Models\n(pairs filtered: justified↔unjustified must flip sign)",
                 fontsize=13, pad=30)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved heatmap.png")


def plot_base_vs_instruct(all_favour, all_meta):
    """Paired bar chart: base vs instruct per model family."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, (base_key, inst_key) in enumerate(FAMILY_PAIRS):
        ax = axes[idx // 2][idx % 2]
        base_name = MODELS[base_key][0]
        inst_name = MODELS[inst_key][0]

        base_vals = [all_favour[base_key].get(c, 0) for c in COUNTRIES]
        inst_vals = [all_favour[inst_key].get(c, 0) for c in COUNTRIES]

        # Sort by base values
        sorted_idx = np.argsort(base_vals)[::-1]
        sorted_countries = [COUNTRIES[i] for i in sorted_idx]
        base_sorted = [base_vals[i] for i in sorted_idx]
        inst_sorted = [inst_vals[i] for i in sorted_idx]

        x = np.arange(len(COUNTRIES))
        w = 0.35
        bars1 = ax.bar(x - w/2, base_sorted, w, label=base_name, color="#4C72B0", alpha=0.8)
        bars2 = ax.bar(x + w/2, inst_sorted, w, label=inst_name, color="#DD8452", alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_countries, rotation=35, ha="right", fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_ylabel("Logprob favour")
        ax.legend(fontsize=8)

        bk, bt = all_meta[base_key]
        ik, it = all_meta[inst_key]
        ax.set_title(f"{base_name.split()[0]} family  (base: {bk}/{bt}, inst: {ik}/{it} pairs)",
                     fontsize=11)

    fig.suptitle("Base vs Instruct: RLHF Amplification of Country Bias",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "base_vs_instruct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved base_vs_instruct.png")


def plot_coherence(all_data):
    """Scatter: justified vs unjustified bias per pair, one panel per model."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    model_keys = list(MODELS.keys())

    for idx, mk in enumerate(model_keys):
        ax = axes[idx // 4][idx % 4]
        df = all_data[mk]

        pair_q = (
            df.groupby(["country_1", "country_2", "question"])
            .apply(lambda g: g.groupby("scenario_type")["bias"].mean().mean())
            .reset_index(name="mean_bias")
        )
        pivot = pair_q.pivot_table(
            index=["country_1", "country_2"], columns="question", values="mean_bias"
        )

        if "justified" in pivot.columns and "unjustified" in pivot.columns:
            j = pivot["justified"].values
            u = pivot["unjustified"].values
            flipped = (j * u) < 0

            ax.scatter(j[flipped], u[flipped], alpha=0.6, s=20, c="#4C72B0", label="flipped")
            ax.scatter(j[~flipped], u[~flipped], alpha=0.4, s=20, c="#DD3333", marker="x", label="not flipped")

            r = np.corrcoef(j, u)[0, 1]
            n_flip = flipped.sum()
            ax.set_title(f"{MODELS[mk][0]}\nr={r:.2f}, {n_flip}/{len(j)} flip", fontsize=9)
        else:
            ax.set_title(f"{MODELS[mk][0]}\nno data", fontsize=9)

        lim = max(abs(ax.get_xlim()[0]), abs(ax.get_xlim()[1]),
                  abs(ax.get_ylim()[0]), abs(ax.get_ylim()[1])) * 1.1
        if lim > 0:
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
        ax.axhline(0, color="gray", linewidth=0.5)
        ax.axvline(0, color="gray", linewidth=0.5)
        ax.plot([-lim, lim], [lim, -lim], "k--", linewidth=0.5, alpha=0.3)
        ax.set_xlabel("justified", fontsize=8)
        ax.set_ylabel("unjustified", fontsize=8)

    fig.suptitle("Coherence: Justified vs Unjustified Bias Per Pair", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "coherence.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved coherence.png")


def main():
    all_favour = {}
    all_meta = {}
    all_data = {}

    for mk in MODELS:
        df, kept, total = load_and_filter(mk)
        favour = compute_country_favour(df)
        all_favour[mk] = favour
        all_meta[mk] = (kept, total)
        all_data[mk] = df
        print(f"{mk}: {kept}/{total} pairs kept after coherence filter")

    plot_heatmap(all_favour, all_meta)
    plot_base_vs_instruct(all_favour, all_meta)
    plot_coherence(all_data)
    plot_western_vs_nonwestern(all_favour)
    plot_instruct_bias(all_favour)


def plot_western_vs_nonwestern(all_favour):
    """Grouped bar: mean Western vs non-Western favour per model, base vs instruct."""
    base_keys = [k for k, (_, t) in MODELS.items() if t == "base"]
    inst_keys = [k for k, (_, t) in MODELS.items() if t == "instruct"]
    all_keys = base_keys + inst_keys
    labels = [MODELS[k][0] for k in all_keys]

    west_means = []
    nonwest_means = []
    west_countries_per_model = []
    nonwest_countries_per_model = []

    for mk in all_keys:
        fav = all_favour[mk]
        w_vals = [fav[c] for c in WESTERN if c in fav]
        nw_vals = [fav[c] for c in NON_WESTERN if c in fav]
        west_means.append(np.mean(w_vals) if w_vals else 0)
        nonwest_means.append(np.mean(nw_vals) if nw_vals else 0)
        west_countries_per_model.append(w_vals)
        nonwest_countries_per_model.append(nw_vals)

    x = np.arange(len(all_keys))
    w = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(12, 9), gridspec_kw={"height_ratios": [2, 1]})

    # Top: grouped bars
    ax = axes[0]
    bars1 = ax.bar(x - w/2, west_means, w, label="Western (USA, France, Canada, Australia)",
                   color="#4C72B0", alpha=0.85)
    bars2 = ax.bar(x + w/2, nonwest_means, w, label="Non-Western (China, Japan, Venezuela, Indonesia)",
                   color="#DD8452", alpha=0.85)

    # Individual country dots
    for i, mk in enumerate(all_keys):
        fav = all_favour[mk]
        for c in WESTERN:
            if c in fav:
                ax.scatter(i - w/2, fav[c], color="#2C5282", s=20, zorder=5, alpha=0.7)
        for c in NON_WESTERN:
            if c in fav:
                ax.scatter(i + w/2, fav[c], color="#9C4221", s=20, zorder=5, alpha=0.7)

    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(len(base_keys) - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax.set_ylabel("Mean logprob favour (justified)", fontsize=11)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_title("Western vs Non-Western Country Bias", fontsize=14, fontweight="bold")

    # Annotations
    ax.text(len(base_keys) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "BASE",
            ha="center", fontsize=10, fontstyle="italic", alpha=0.5)
    ax.text(len(base_keys) + len(inst_keys) / 2 - 0.5, ax.get_ylim()[1] * 0.95, "INSTRUCT",
            ha="center", fontsize=10, fontstyle="italic", alpha=0.5)

    # Bottom: gap (Western - NonWestern)
    ax2 = axes[1]
    gaps = [wm - nwm for wm, nwm in zip(west_means, nonwest_means)]
    colors = ["#4C72B0" if g > 0 else "#DD8452" for g in gaps]
    ax2.bar(x, gaps, 0.6, color=colors, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.axvline(len(base_keys) - 0.5, color="black", linewidth=1.5, linestyle="--", alpha=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=35, ha="right", fontsize=10)
    ax2.set_ylabel("Gap (W - NW)", fontsize=11)
    ax2.set_title("Pro-Western gap: positive = model favours Western countries", fontsize=10)

    # Label the Qwen models
    for i, mk in enumerate(all_keys):
        if "qwen" in mk:
            ax2.annotate("Chinese-made", xy=(i, gaps[i]),
                         xytext=(i, gaps[i] + (0.3 if gaps[i] < 0 else -0.3)),
                         ha="center", fontsize=8, color="red", fontstyle="italic",
                         arrowprops=dict(arrowstyle="->", color="red", lw=0.8))

    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "western_vs_nonwestern.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved western_vs_nonwestern.png")


def plot_instruct_bias(all_favour):
    """Horizontal bar chart: instruct models, country preference % (human-readable)."""
    inst_keys = [k for k, (_, t) in MODELS.items() if t == "instruct"]

    fig, axes = plt.subplots(1, len(inst_keys), figsize=(16, 5), sharey=True)

    for idx, mk in enumerate(inst_keys):
        ax = axes[idx]
        fav = all_favour[mk]

        # Convert logprob to preference %: sigmoid(bias) * 100
        # Fixed country order across all panels
        countries_sorted = COUNTRIES
        pref_pcts = [100.0 / (1.0 + np.exp(-fav[c])) for c in countries_sorted]

        colors = ["#4C72B0" if p >= 50 else "#C44E52" for p in pref_pcts]

        bars = ax.barh(range(len(countries_sorted)), [p - 50 for p in pref_pcts],
                       left=50, color=colors, alpha=0.85, height=0.7)

        # Value labels
        for i, (c, p) in enumerate(zip(countries_sorted, pref_pcts)):
            offset = 1.5 if p >= 50 else -1.5
            ha = "left" if p >= 50 else "right"
            ax.text(p + offset, i, f"{p:.0f}%", va="center", ha=ha, fontsize=9, fontweight="bold")

        ax.set_yticks(range(len(countries_sorted)))
        ax.set_yticklabels(countries_sorted, fontsize=10)
        ax.axvline(50, color="black", linewidth=1)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Preference %", fontsize=10)
        ax.set_title(MODELS[mk][0], fontsize=12, fontweight="bold")

        # Shade the "disfavoured" zone
        ax.axvspan(0, 50, color="#FFEEEE", alpha=0.3, zorder=0)
        ax.axvspan(50, 100, color="#EEEEFF", alpha=0.3, zorder=0)

        if idx == 0:
            ax.text(25, len(countries_sorted) - 0.2, "Disfavoured", ha="center", fontsize=8, color="#C44E52", fontstyle="italic")
            ax.text(75, len(countries_sorted) - 0.2, "Favoured", ha="center", fontsize=8, color="#4C72B0", fontstyle="italic")

    fig.suptitle(
        "Country Bias in Instruct Models\n"
        "\"If the model had to judge which country acted more justifiably, how often would it side with each?\"",
        fontsize=13, fontweight="bold", y=1.02,
    )
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "instruct_bias_pct.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved instruct_bias_pct.png")


if __name__ == "__main__":
    main()
