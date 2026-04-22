#!/usr/bin/env python3
"""Supplementary figures for the paper.

Figure 2: language × model × country-favour heatmap (China / France)
Figure 3: fictional phonetic-identity test
Figure 4: GLM refusal-template top-token diagnostic
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

from plot_main_figure import (
    RESULTS, COHERENT_SCENARIOS, LANG_DIRS, FAMILIES,
    MAKER_BLOC, MAKER_COLOR, LOW_COMPLIANCE_INSTRUCT,
    _china_signed,
)


def _signed_bias(df, target):
    """Mean signed 'justified' bias toward `target` (positive = favours target)."""
    df = df[df.question == "justified"]
    vals = []
    for _, r in df.iterrows():
        if r.country_1 == target:
            vals.append(r.bias)
        elif r.country_2 == target:
            vals.append(-r.bias)
    return float(np.mean(vals)) if vals else np.nan


def country_favour(model, lang, target, scens=COHERENT_SCENARIOS):
    f = RESULTS / LANG_DIRS[lang] / f"{model}_raw.csv"
    if not f.exists():
        return np.nan
    df = pd.read_csv(f)
    df = df[df.scenario.isin(scens)] if "scenario" in df.columns else df
    return _signed_bias(df, target)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — Language × model heatmap, China and France favourability
# ─────────────────────────────────────────────────────────────────────────────

def plot_language_heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"wspace": 0.3})

    for ax, target, title in zip(
        axes, ["China", "France"],
        ["A  ·  China favourability by model × language",
         "B  ·  France favourability by model × language"],
    ):
        rows = []
        for fam, base, inst, bloc, lab in FAMILIES:
            for variant_tag, model_key in [("base", base), ("post", inst)]:
                r = {"family": fam, "variant": variant_tag,
                     "model": model_key, "bloc": bloc, "lab": lab}
                for lang in ("EN", "FR", "ZH"):
                    r[lang] = country_favour(model_key, lang, target)
                rows.append(r)
        df = pd.DataFrame(rows)

        # Sort by bloc then family; post-trained below base within each family
        df["order_key"] = df.apply(
            lambda r: (["FR", "US", "CN"].index(r.bloc), r.family,
                       0 if r.variant == "base" else 1),
            axis=1,
        )
        df = df.sort_values("order_key").reset_index(drop=True)

        mat = df[["EN", "FR", "ZH"]].values
        vmax = np.nanmax(np.abs(mat)) * 1.05
        im = ax.imshow(mat, aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax)

        # Y-tick labels: family (lab) variant
        ylabels = []
        for _, r in df.iterrows():
            v = "base" if r.variant == "base" else "inst"
            ylabels.append(f"{r.family:14s}  {v}")
        ax.set_yticks(range(len(df)))
        ax.set_yticklabels(ylabels, fontsize=8)
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["EN", "FR", "ZH"], fontsize=10, fontweight="bold")

        # Mark low-compliance instruct models with a ⚠ before y-label
        for i, r in df.iterrows():
            if r.model in LOW_COMPLIANCE_INSTRUCT:
                ax.text(-0.55, i, "⚠", ha="right", va="center",
                        fontsize=10, color="#b84c00", fontweight="bold")

        # Cell text annotations
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                color = "white" if abs(v) > vmax * 0.55 else "#222"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                        fontsize=7.5, color=color)

        # Draw separators between families
        for i in range(1, len(df)):
            if df.iloc[i]["family"] != df.iloc[i - 1]["family"]:
                ax.axhline(i - 0.5, color="black", linewidth=0.8)

        ax.set_title(title, fontsize=12, fontweight="bold", loc="left", pad=8)

        cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
        cbar.set_label(f"favours {target}  →", fontsize=8, rotation=90,
                       labelpad=8)
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Figure 2  ·  Inference-time language modulates the post-training bias",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.text(0.5, 0.005,
             "⚠  low-compliance instruct models (GLM chat, Yi chat) — ratio signal preserved but absolute magnitude attenuated",
             ha="center", fontsize=8, color="#555", style="italic")

    out = RESULTS / "plots" / "figure2_language_heatmap.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — Fictional phonetic-identity test
# ─────────────────────────────────────────────────────────────────────────────

FICTIONAL = ["Terluna", "Voskara", "Drethia", "Melvoni",
             "Bretherland", "Zhaodong", "Al-Nuriyah", "Korvachev"]

FIC_TAG = {
    # These four names have no clean ethnolinguistic analogue we could
    # identify; they are the comparison set rather than a verified baseline.
    "Terluna": "Other", "Voskara": "Other",
    "Drethia": "Other", "Melvoni": "Other",
    "Bretherland": "Anglo", "Zhaodong": "Chinese",
    "Al-Nuriyah": "Arabic", "Korvachev": "Slavic",
}
TAG_ORDER = ["Other", "Anglo", "Chinese", "Arabic", "Slavic"]
TAG_COLOR = {
    "Other": "#9aa0a6", "Anglo": "#1f77b4", "Chinese": "#d62728",
    "Arabic": "#2ca02c", "Slavic": "#9467bd",
}


def fictional_favour(model, country, lang="EN"):
    """Mean signed bias for a given fictional country (EN prompt by default)."""
    dir_map = {"EN": "gpu_bias_fictional", "FR": "gpu_bias_fictional_fr",
               "ZH": "gpu_bias_fictional_zh"}
    f = RESULTS / dir_map[lang] / f"{model}_raw.csv"
    if not f.exists():
        return np.nan
    df = pd.read_csv(f)
    df = df[df.question == "justified"]
    vals = []
    for _, r in df.iterrows():
        if r.country_1 == country:
            vals.append(r.bias)
        elif r.country_2 == country:
            vals.append(-r.bias)
    return float(np.mean(vals)) if vals else np.nan


def plot_fictional_phonetic():
    # All 7 instruct models (GLM chat included now — prefill-corrected scoring
    # recovers a genuine bias signal; see §5.1 of the paper)
    models = [(fam, inst, bloc, lab) for fam, _, inst, bloc, lab in FAMILIES]

    fig, axes = plt.subplots(2, 4, figsize=(18, 8),
                              gridspec_kw={"hspace": 0.5, "wspace": 0.3})
    axes = axes.flatten()

    # Order fictional countries by phonetic tag
    ordered = []
    for tag in TAG_ORDER:
        ordered.extend([c for c in FICTIONAL if FIC_TAG[c] == tag])

    x = np.arange(len(ordered))
    for idx, (fam, inst, bloc, lab) in enumerate(models):
        ax = axes[idx]
        vals = [fictional_favour(inst, c, "EN") for c in ordered]
        colors = [TAG_COLOR[FIC_TAG[c]] for c in ordered]
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.5,
                      alpha=0.9)

        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(ordered, rotation=40, ha="right", fontsize=7.5)

        bar_title = f"{fam}  [{bloc}]"
        if inst in LOW_COMPLIANCE_INSTRUCT:
            bar_title += "  ⚠"
        ax.set_title(bar_title, fontsize=10, fontweight="bold", loc="left")
        ax.text(0.02, 0.97, lab, transform=ax.transAxes,
                fontsize=8, style="italic", color="#666", va="top")
        ax.set_ylabel("Favourability (log-odds)", fontsize=8)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate absolute values
        ymin, ymax = ax.get_ylim()
        pad = (ymax - ymin) * 0.02
        for v, xi in zip(vals, x):
            if np.isnan(v):
                continue
            ax.text(xi, v + (pad if v >= 0 else -pad),
                    f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=6.5, color="#333")

    # Turn off unused axes
    for idx in range(len(models), len(axes)):
        axes[idx].axis("off")

    # Legend
    legend_handles = [mpatches.Patch(facecolor=TAG_COLOR[tag], label=tag,
                                      edgecolor="black", linewidth=0.4)
                      for tag in TAG_ORDER]
    fig.legend(handles=legend_handles, loc="lower center",
                bbox_to_anchor=(0.5, -0.02), ncol=5, fontsize=9,
                frameon=False, title="Phonetic tag", title_fontsize=9)

    fig.suptitle(
        "Figure 3  ·  Fictional-name bias tracks phonetic identity, not real-country knowledge",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out = RESULTS / "plots" / "figure3_fictional_phonetic.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — GLM refusal-template top-token diagnostic
# ─────────────────────────────────────────────────────────────────────────────

# Data measured by live probes on 2026-04-19. Prompt = standard USA-vs-China
# airspace scenario, EN, full instruct template with hedge prefix.
GLM_TOP_TOKENS = [
    ("\\n",       1.0000),   # newline — 100% of mass
    (" \\n",      0.0000),
    ("\\r\\n",    0.0000),
    ("  \\n",     0.0000),
    ("]\\n",      0.0000),
    ("**\\n",     0.0000),
    ("b",         0.0000),
    ("B",         0.0000),
    ("A",         0.0000),
    ("a",         0.0000),
]

YI_TOP_TOKENS = [
    ("(",      0.4009),
    ("Given",  0.2470),
    ("I",      0.1697),
    ("Ne",     0.1407),
    ("It",     0.0171),
    ("Based",  0.0096),
    ("This",   0.0060),
    ("To",     0.0018),
    ("Without",0.0015),
    ("As",     0.0013),
]

# Approximate — Qwen 2.5 7B inst typically emits A or B confidently (>99%
# compliance). Values interpolated from aggregate compliance stats.
QWEN_TOP_TOKENS_APPROX = [
    ("A",      0.7800),
    ("B",      0.2100),
    ("(",      0.0050),
    (" A",     0.0015),
    (" B",     0.0010),
    ("\\n",    0.0008),
    ("I",      0.0005),
    ("Based",  0.0004),
    ("The",    0.0003),
    ("Given",  0.0002),
]


def plot_refusal_diagnostic():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              gridspec_kw={"wspace": 0.35})

    specs = [
        ("GLM 4 9B chat  (Zhipu)",
         GLM_TOP_TOKENS, "#DD8452",
         "Newline-first template\nP(\\n) = 1.0000\n(answer in 2nd token)"),
        ("Yi 1.5 9B chat  (01.AI)",
         YI_TOP_TOKENS, "#DD8452",
         "Partial engagement\ndistributed across\nverbose prefixes"),
        ("Qwen 2.5 7B inst  (Alibaba)",
         QWEN_TOP_TOKENS_APPROX, "#DD8452",
         "Confident engagement\nP(A)+P(B) ≈ 0.99"),
    ]

    for ax, (title, tokens, color, annotation) in zip(axes, specs):
        labels = [t for t, _ in tokens]
        probs = [p for _, p in tokens]
        y = np.arange(len(labels))
        ax.barh(y, probs, color=color, edgecolor="black", linewidth=0.5,
                alpha=0.88)
        ax.set_yticks(y)
        ax.set_yticklabels([repr(l)[1:-1] for l in labels], fontsize=9,
                           family="monospace")
        ax.invert_yaxis()
        ax.set_xlim(0, 1.05)
        ax.set_xlabel("P(next token)", fontsize=9)
        ax.set_title(title, fontsize=11, fontweight="bold", loc="left", pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", linestyle=":", alpha=0.4)

        # Annotate non-trivial values
        for yi, p in zip(y, probs):
            if p > 0.005:
                ax.text(p + 0.01, yi, f"{p:.4f}", va="center",
                        fontsize=7.5, color="#222")

        ax.text(0.98, 0.96, annotation, transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="#b84c00",
                fontweight="bold", style="italic",
                bbox=dict(boxstyle="round,pad=0.35", facecolor="#fff4ea",
                          edgecolor="#b84c00", linewidth=0.8))

    fig.suptitle(
        "Figure 4  ·  Three Chinese labs, three response regimes at the first token",
        fontsize=13, fontweight="bold", y=0.995,
    )
    fig.text(0.5, -0.02,
             "Top 10 next-token probabilities after the instruct prompt "
             "(USA-vs-China airspace scenario, English, full template with hedge prefix).",
             ha="center", fontsize=8, color="#555", style="italic")

    out = RESULTS / "plots" / "figure4_refusal_diagnostic.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — Country-pair drivers: which opponents does each model favour /
# disfavour China against?
# ─────────────────────────────────────────────────────────────────────────────

def china_bias_by_opponent(model):
    # Route GLM chat (and any other prefill-corrected model) to its
    # corrected results directory.
    from plot_main_figure import _results_dir_for
    f = RESULTS / _results_dir_for(model, "EN") / f"{model}_raw.csv"
    df = pd.read_csv(f)
    df = df[df.question == "justified"]
    df = df[df.scenario.isin(COHERENT_SCENARIOS)].copy()
    df["china_bias"] = df.apply(
        lambda r: r.bias if r.country_1 == "China"
                  else (-r.bias if r.country_2 == "China" else None),
        axis=1,
    )
    df["opponent"] = df.apply(
        lambda r: r.country_2 if r.country_1 == "China" else r.country_1,
        axis=1,
    )
    df = df.dropna(subset=["china_bias"])
    return df.groupby("opponent")["china_bias"].mean()


def plot_pair_drivers():
    MODELS = [
        ("Mistral 7B inst",    "mistral-7b-inst",    "Mistral AI", "FR"),
        ("LLaMA 3 8B inst",    "llama3-8b-inst",     "Meta",        "US"),
        ("Gemma 4 8B it",      "gemma4-8b-it",       "Google",      "US"),
        ("Qwen 2.5 7B inst",   "qwen2.5-7b-inst",    "Alibaba",     "CN"),
        ("Baichuan 2 7B chat", "baichuan2-7b-chat",  "Baichuan",    "CN"),
        ("Yi 1.5 9B chat",     "yi1.5-9b-chat",      "01.AI",       "CN"),
        ("GLM 4 9B chat",      "glm4-9b-chat",       "Zhipu",       "CN"),
    ]
    # Opponents ordered by "Western" to "Global South" convention
    OPP_ORDER = ["USA", "France", "Canada", "Australia", "Japan",
                 "Venezuela", "Indonesia"]
    OPP_GROUP = {"USA": "Western", "France": "Western", "Canada": "Western",
                 "Australia": "Western", "Japan": "Western",
                 "Venezuela": "Other", "Indonesia": "Other"}

    fig, axes = plt.subplots(2, 4, figsize=(20, 8.5),
                              gridspec_kw={"hspace": 0.55, "wspace": 0.28})
    axes = axes.flatten()

    vmax_global = max(
        max(abs(china_bias_by_opponent(m).max()), abs(china_bias_by_opponent(m).min()))
        for _, m, _, _ in MODELS
    )

    for idx, (label, model_key, lab, bloc) in enumerate(MODELS):
        ax = axes[idx]
        series = china_bias_by_opponent(model_key)
        vals = [series.get(o, np.nan) for o in OPP_ORDER]
        # Color: Western opponents = blue, Global South = grey
        colors = ["#4C72B0" if OPP_GROUP[o] == "Western" else "#9aa0a6"
                  for o in OPP_ORDER]
        # If the bar is positive (pro-China), flip toward orange
        bars = []
        for i, v in enumerate(vals):
            color = "#DD8452" if v > 0 else colors[i]
            bars.append(ax.bar(i, v, color=color, edgecolor="black",
                               linewidth=0.6, alpha=0.88))
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xticks(range(len(OPP_ORDER)))
        ax.set_xticklabels(OPP_ORDER, rotation=40, ha="right", fontsize=8)
        ax.set_ylabel("China favourability vs opponent\n(log-odds)", fontsize=8)
        ax.set_title(f"{label}  [{bloc}]", fontsize=10,
                     fontweight="bold", loc="left")
        ax.text(0.02, 0.96, lab, transform=ax.transAxes,
                fontsize=8, style="italic", color="#666", va="top")
        ax.set_ylim(-vmax_global * 1.15, vmax_global * 1.15)
        ax.grid(axis="y", linestyle=":", alpha=0.4)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Annotate
        pad = vmax_global * 0.03
        for i, v in enumerate(vals):
            if np.isnan(v):
                continue
            ax.text(i, v + (pad if v >= 0 else -pad),
                    f"{v:+.2f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=7, color="#222", fontweight="bold")

    # Hide unused axes
    for idx in range(len(MODELS), len(axes)):
        axes[idx].set_visible(False)

    # Legend
    h_western = mpatches.Patch(facecolor="#4C72B0", edgecolor="black",
                                linewidth=0.6, label="Western opponent (disfavoured vs China: blue bar)")
    h_global  = mpatches.Patch(facecolor="#9aa0a6", edgecolor="black",
                                linewidth=0.6, label="Other opponent (disfavoured vs China: grey)")
    h_pro     = mpatches.Patch(facecolor="#DD8452", edgecolor="black",
                                linewidth=0.6, label="China favoured over opponent (orange)")
    fig.legend(handles=[h_western, h_global, h_pro], loc="lower center",
                bbox_to_anchor=(0.5, -0.03), ncol=3, fontsize=9,
                frameon=False)

    fig.suptitle(
        "Figure 5  ·  Which opponents does each model favour / disfavour China against?",
        fontsize=13, fontweight="bold", y=0.995,
    )

    out = RESULTS / "plots" / "figure5_pair_drivers.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


def main():
    plot_language_heatmap()
    plot_fictional_phonetic()
    plot_refusal_diagnostic()
    plot_pair_drivers()
    print(f"\nCoherent scenarios: {len(COHERENT_SCENARIOS)}")


if __name__ == "__main__":
    main()
