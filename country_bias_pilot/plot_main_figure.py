#!/usr/bin/env python3
"""Main paper figure: post-training amplifies geopolitical bias."""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

COUNTRIES = ["USA", "France", "China", "Japan", "Venezuela", "Canada", "Australia", "Indonesia"]

FICTIONAL = ["Terluna", "Voskara", "Drethia", "Melvoni",
             "Bretherland", "Zhaodong", "Al-Nuriyah", "Korvachev"]
FIC_TAG = {
    "Terluna": "Neutral", "Voskara": "Neutral", "Drethia": "Neutral", "Melvoni": "Neutral",
    "Bretherland": "Anglo", "Zhaodong": "Chinese", "Al-Nuriyah": "Arabic", "Korvachev": "Slavic",
}
TAG_ORDER = ["Neutral", "Anglo", "Chinese", "Arabic", "Slavic"]
TAG_COLOR = {
    "Neutral": "#9aa0a6", "Anglo": "#1f77b4", "Chinese": "#d62728",
    "Arabic": "#2ca02c", "Slavic": "#9467bd",
}

FAMILIES = [
    ("Mistral 7B",  "mistral-7b",  "mistral-7b-inst",  "Western"),
    ("LLaMA 3 8B",  "llama3-8b",   "llama3-8b-inst",   "Western"),
    ("Gemma 4 8B",  "gemma4-8b",   "gemma4-8b-it",     "Western"),
    ("Qwen 2.5 7B", "qwen2.5-7b",  "qwen2.5-7b-inst",  "Chinese"),
]

MAKER_COLOR = {"Western": "#4C72B0", "Chinese": "#DD8452"}

COUNTRY_COLORS = {
    "USA": "#1f77b4", "France": "#aec7e8", "Canada": "#98df8a", "Australia": "#2ca02c",
    "China": "#d62728", "Japan": "#ff9896", "Venezuela": "#ff7f0e", "Indonesia": "#ffbb78",
}


def favour(csv_path, names):
    """Mean signed bias per name from the 'justified' suffix."""
    df = pd.read_csv(csv_path)
    jdf = df[df.question == "justified"]
    out = {}
    for n in names:
        vals = []
        for (c1, c2), grp in jdf.groupby(["country_1", "country_2"]):
            if n not in (c1, c2):
                continue
            sign = 1.0 if n == c1 else -1.0
            vals.append(sign * grp.bias.mean())
        out[n] = float(np.mean(vals)) if vals else 0.0
    return out


def to_pref(logprob):
    return 100.0 / (1.0 + np.exp(-logprob))


def panel_a(ax):
    """Base → post-trained connected dot plot."""
    for fam_idx, (fam_name, base_key, inst_key, maker) in enumerate(FAMILIES):
        base_fav = favour(RESULTS / "gpu_bias" / f"{base_key}_raw.csv", COUNTRIES)
        inst_fav = favour(RESULTS / "gpu_bias" / f"{inst_key}_raw.csv", COUNTRIES)

        x_base = fam_idx * 3
        x_inst = fam_idx * 3 + 1.2

        base_prefs = [to_pref(base_fav[c]) for c in COUNTRIES]
        inst_prefs = [to_pref(inst_fav[c]) for c in COUNTRIES]
        spread_base = np.std(base_prefs)
        spread_inst = np.std(inst_prefs)

        to_label = []
        for country in COUNTRIES:
            bp = to_pref(base_fav[country])
            ip = to_pref(inst_fav[country])
            color = COUNTRY_COLORS[country]
            ax.plot([x_base, x_inst], [bp, ip], color=color, alpha=0.45, linewidth=1.5)
            ax.scatter(x_base, bp, color=color, s=45, zorder=5,
                       edgecolors="white", linewidth=0.6)
            ax.scatter(x_inst, ip, color=color, s=90, zorder=6,
                       edgecolors="white", linewidth=0.8)
            if ip > 70 or ip < 30:
                to_label.append((ip, country, color))

        min_gap = 4.5
        for group, sign in (([t for t in to_label if t[0] > 50], +1),
                             ([t for t in to_label if t[0] <= 50], -1)):
            group.sort(key=lambda t: t[0], reverse=(sign > 0))
            last_y = None
            for ip, country, color in group:
                y = ip
                if last_y is not None and abs(y - last_y) < min_gap:
                    y = last_y - sign * min_gap
                ax.annotate(country, xy=(x_inst, ip),
                            xytext=(x_inst + 0.25, y),
                            fontsize=8, color=color, fontweight="bold",
                            va="center", ha="left")
                last_y = y

        ax.text(fam_idx * 3 + 0.6, 104, fam_name, ha="center",
                fontsize=11, fontweight="bold", clip_on=False)
        ax.text(fam_idx * 3 + 0.6, 99.5,
                f"spread  {spread_base:.1f}  →  {spread_inst:.1f}",
                ha="center", fontsize=8, color="#555", style="italic")
        if fam_idx < len(FAMILIES) - 1:
            ax.axvline(fam_idx * 3 + 2.1, color="gray", linewidth=0.5, alpha=0.25)

    ax.axhline(50, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(-0.3, 50, "neutral", fontsize=8, color="gray",
            va="center", ha="right", style="italic")

    xticks, xlabels = [], []
    for i in range(len(FAMILIES)):
        xticks.extend([i * 3, i * 3 + 1.2])
        xlabels.extend(["Base", "Post-trained"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.set_ylabel("Preference  (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.8, (len(FAMILIES) - 1) * 3 + 2.2)
    ax.set_title("A  ·  Post-training amplifies geopolitical bias",
                 fontsize=13, fontweight="bold", loc="left", pad=30)

    for country in COUNTRIES:
        ax.scatter([], [], color=COUNTRY_COLORS[country], s=55, label=country)
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              fontsize=9, frameon=False, handletextpad=0.3, columnspacing=1.2)


def panel_b(ax):
    """Per-country preference across 4 post-trained models, sorted by Western average."""
    prefs = {}
    for fam_name, _, inst_key, maker in FAMILIES:
        fav = favour(RESULTS / "gpu_bias" / f"{inst_key}_raw.csv", COUNTRIES)
        prefs[fam_name] = {c: to_pref(fav[c]) for c in COUNTRIES}

    western = [f for f, *_ in FAMILIES if f != "Qwen 2.5 7B"]
    western_mean = {c: np.mean([prefs[f][c] for f in western]) for c in COUNTRIES}
    order = sorted(COUNTRIES, key=lambda c: western_mean[c], reverse=True)

    y = np.arange(len(order))
    h = 0.19
    offsets = {fam: (i - 1.5) * h for i, (fam, *_) in enumerate(FAMILIES)}

    for fam_name, _, _, maker in FAMILIES:
        color = MAKER_COLOR[maker]
        alpha = 1.0 if maker == "Chinese" else 0.55
        lw = 1.2 if maker == "Chinese" else 0.4
        vals = [prefs[fam_name][c] - 50 for c in order]
        ax.barh(y + offsets[fam_name], vals, h, left=50,
                color=color, alpha=alpha,
                edgecolor="black" if maker == "Chinese" else "white",
                linewidth=lw, label=fam_name)

    ax.axvline(50, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(order, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(5, 95)
    ax.set_xlabel("Preference  (%)", fontsize=10)
    ax.set_title("B  ·  Direction of amplification follows the model's maker",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

    western_patch = plt.Rectangle((0, 0), 1, 1, facecolor=MAKER_COLOR["Western"],
                                   alpha=0.55, edgecolor="white")
    qwen_patch = plt.Rectangle((0, 0), 1, 1, facecolor=MAKER_COLOR["Chinese"],
                               alpha=1.0, edgecolor="black", linewidth=1.2)
    ax.legend([western_patch, qwen_patch],
              ["Western-made  (Mistral, LLaMA, Gemma)", "Chinese-made  (Qwen)"],
              fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              frameon=False, ncol=2)


def panel_c(ax):
    """Fictional-name bias across 4 post-trained models, grouped by phonetic tag."""
    model_labels = [("Mistral",  "mistral-7b-inst",  "Western"),
                    ("LLaMA",    "llama3-8b-inst",   "Western"),
                    ("Gemma 4",  "gemma4-8b-it",     "Western"),
                    ("Qwen",     "qwen2.5-7b-inst",  "Chinese")]

    prefs = {}
    for label, key, _ in model_labels:
        fav = favour(RESULTS / "gpu_bias_fictional" / f"{key}_raw.csv", FICTIONAL)
        prefs[label] = {n: to_pref(fav[n]) for n in FICTIONAL}

    x = np.arange(len(model_labels))
    group_width = 0.82
    slots = []
    for tag in TAG_ORDER:
        slots.extend([n for n in FICTIONAL if FIC_TAG[n] == tag])
    n_slots = len(slots)
    slot_w = group_width / n_slots

    qwen_idx = next(i for i, (_, _, m) in enumerate(model_labels) if m == "Chinese")
    ax.axvspan(qwen_idx - 0.5, qwen_idx + 0.5, color="#fff4ea", zorder=0)

    for slot_i, name in enumerate(slots):
        tag = FIC_TAG[name]
        color = TAG_COLOR[tag]
        vals = [prefs[lab][name] for lab, *_ in model_labels]
        ax.bar(x + (slot_i - (n_slots - 1) / 2) * slot_w,
               [v - 50 for v in vals], slot_w * 0.9, bottom=50,
               color=color, alpha=0.9,
               edgecolor="white", linewidth=0.4, zorder=2)

    ax.axhline(50, color="black", linewidth=0.8, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([lab for lab, *_ in model_labels], fontsize=10)
    ax.set_ylabel("Preference  (%)", fontsize=10)
    ax.set_ylim(15, 88)
    ax.set_xlim(-0.5, len(model_labels) - 0.5)
    ax.set_title("C  ·  Bias is triggered by linguistic identity, not real-country knowledge",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

    for tag in TAG_ORDER:
        members = [n for n in FICTIONAL if FIC_TAG[n] == tag]
        lbl = f"{tag}  (n={len(members)})" if tag == "Neutral" \
              else f"{tag}  ({members[0]})"
        ax.bar([], [], color=TAG_COLOR[tag], label=lbl)
    ax.legend(fontsize=8, loc="upper left", ncol=1, frameon=False,
              handlelength=1.2, handletextpad=0.5, title="Phonetic tag",
              title_fontsize=8)

    ax.text(qwen_idx, 86, "Anglo ↓   Chinese ↑   (maker flip)",
            ha="center", fontsize=9, fontweight="bold",
            color="#b9570f", style="italic")


def main():
    fig = plt.figure(figsize=(17, 13))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.28,
                            height_ratios=[1.05, 1.0])

    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a)
    panel_b(ax_b)
    panel_c(ax_c)

    for ax in (ax_a, ax_b, ax_c):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle("Aligned to whom?  Post-training amplifies geopolitical bias in LLMs",
                 fontsize=16, fontweight="bold", y=0.995)

    out = RESULTS / "plots" / "main_figure.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
