#!/usr/bin/env python3
"""Main paper figure: post-training amplifies geopolitical bias, maker-aligned and
language-amplified.

Uses the 34-scenario strict-coherence subset (justified/unjustified flip sign in
≥70% of model × language combinations) as the primary analysis base.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent / "results"

COUNTRIES = ["USA", "France", "China", "Japan", "Venezuela", "Canada", "Australia", "Indonesia"]

FAMILIES = [
    ("Mistral 7B",    "mistral-7b",     "mistral-7b-inst",   "FR", "Mistral AI"),
    ("LLaMA 3 8B",    "llama3-8b",      "llama3-8b-inst",    "US", "Meta"),
    ("Gemma 4 8B",    "gemma4-8b",      "gemma4-8b-it",      "US", "Google"),
    ("Qwen 2.5 7B",   "qwen2.5-7b",     "qwen2.5-7b-inst",   "CN", "Alibaba"),
    ("Baichuan 2 7B", "baichuan2-7b",   "baichuan2-7b-chat", "CN", "Baichuan"),
    ("Yi 1.5 9B",     "yi1.5-9b",       "yi1.5-9b-chat",     "CN", "01.AI"),
    ("GLM 4 9B",      "glm4-9b",        "glm4-9b-chat",      "CN", "Zhipu"),
]

# Models whose post-trained variant has compliance < 0.1 under the standard
# scoring — their bias numbers are still directionally informative (variant
# ratios are preserved) but the model is largely not engaging with the A/B
# forced-choice. GLM 4 chat and Yi 1.5 chat are both prefill-corrected via
# PREFILL_DIR_MAP below (prefill characters: \n for GLM, ( for Yi), achieving
# $\geq$0.99 compliance in all three language conditions. Set kept empty.
LOW_COMPLIANCE_INSTRUCT = set()

# Models whose bias data for the post-trained variant should come from a
# prefill-corrected results directory instead of the default gpu_bias/ tree.
# For GLM 4 chat the standard template places P(\n)=1.0000 on the first token;
# prefilling \n and scoring the next token recovers the actual forced-choice
# answer distribution. Key: model name. Value: dict mapping condition suffix
# to the directory override.
PREFILL_DIR_MAP = {
    "glm4-9b-chat": {
        "EN":  "gpu_bias_glm_prefill",
        "FR":  "gpu_bias_glm_prefill_fr",
        "ZH":  "gpu_bias_glm_prefill_zh",
    },
    "yi1.5-9b-chat": {
        "EN":  "gpu_bias_yi_prefill",
        "FR":  "gpu_bias_yi_prefill_fr",
        "ZH":  "gpu_bias_yi_prefill_zh",
    },
}

MAKER_BLOC = {"US": "Western", "FR": "Western", "CN": "Chinese"}
MAKER_COLOR = {"Western": "#4C72B0", "Chinese": "#DD8452"}

COUNTRY_COLORS = {
    "USA": "#1f77b4", "France": "#aec7e8", "Canada": "#98df8a", "Australia": "#2ca02c",
    "China": "#d62728", "Japan": "#ff9896", "Venezuela": "#ff7f0e", "Indonesia": "#ffbb78",
}

LANG_DIRS = {"EN": "gpu_bias", "FR": "gpu_bias_fr", "ZH": "gpu_bias_zh"}


# ─────────────────────────────────────────────────────────────────────────────
# Coherence filter: scenarios where justified and unjustified flip sign in
# ≥70% of (model × language) combinations. Used as the primary analysis subset.
# ─────────────────────────────────────────────────────────────────────────────

def _china_signed(df):
    df = df.copy()
    df["china_bias"] = df.apply(
        lambda r: r["bias"] if r["country_1"] == "China"
                  else (-r["bias"] if r["country_2"] == "China" else None),
        axis=1,
    )
    return df.dropna(subset=["china_bias"])


def compute_coherent_scenarios(threshold=0.7):
    records = []
    all_models = [m for _, base, inst, _, _ in FAMILIES for m in (base, inst)]
    for m in all_models:
        for lang in LANG_DIRS:
            # Route to prefill-corrected dir where applicable (GLM chat)
            d = (PREFILL_DIR_MAP.get(m, {}).get(lang)
                 or LANG_DIRS[lang])
            f = RESULTS / d / f"{m}_raw.csv"
            if not f.exists():
                continue
            raw = _china_signed(pd.read_csv(f))
            for scen, g in raw.groupby("scenario"):
                j = g[g["question"] == "justified"]["china_bias"].mean()
                u = g[g["question"] == "unjustified"]["china_bias"].mean()
                if pd.notna(j) and pd.notna(u):
                    records.append({"scenario": scen, "coherent": (j * u < 0)})
    coh = pd.DataFrame(records).groupby("scenario")["coherent"].mean()
    return set(coh[coh >= threshold].index)


COHERENT_SCENARIOS = compute_coherent_scenarios(0.7)


def favour(csv_path, names, scens=None):
    """Mean signed 'justified' bias per name, optionally restricted to `scens`."""
    df = pd.read_csv(csv_path)
    df = df[df.question == "justified"]
    if scens is not None:
        df = df[df.scenario.isin(scens)] if "scenario" in df.columns else df
    out = {}
    for n in names:
        vals = []
        for (c1, c2), grp in df.groupby(["country_1", "country_2"]):
            if n not in (c1, c2):
                continue
            sign = 1.0 if n == c1 else -1.0
            vals.append(sign * grp.bias.mean())
        out[n] = float(np.mean(vals)) if vals else 0.0
    return out


def _results_dir_for(model, lang):
    """Return the directory that holds the raw CSV for (model, lang). For
    models in PREFILL_DIR_MAP we use the prefill-corrected directory."""
    if model in PREFILL_DIR_MAP and lang in PREFILL_DIR_MAP[model]:
        return PREFILL_DIR_MAP[model][lang]
    return LANG_DIRS[lang]


def china_favour(model, lang, scens=COHERENT_SCENARIOS):
    """Mean China-signed 'justified' bias across coherent scenarios."""
    f = RESULTS / _results_dir_for(model, lang) / f"{model}_raw.csv"
    if not f.exists():
        return np.nan
    raw = _china_signed(pd.read_csv(f))
    raw = raw[(raw["question"] == "justified") & (raw["scenario"].isin(scens))]
    return raw["china_bias"].mean() if len(raw) else np.nan


def to_pref(logprob):
    return 100.0 / (1.0 + np.exp(-logprob))


# ─────────────────────────────────────────────────────────────────────────────
# Panel A — base → post-trained connected dot plot, EN real countries
# ─────────────────────────────────────────────────────────────────────────────

def panel_a(ax):
    # Tighter spacing for 7 families
    col_w = 2.4   # total width per family
    gap = 1.1     # separation between base and post-trained within a family
    for fam_idx, (fam_name, base_key, inst_key, maker, lab) in enumerate(FAMILIES):
        base_fav = favour(RESULTS / _results_dir_for(base_key, "EN") / f"{base_key}_raw.csv",
                          COUNTRIES, COHERENT_SCENARIOS)
        inst_fav = favour(RESULTS / _results_dir_for(inst_key, "EN") / f"{inst_key}_raw.csv",
                          COUNTRIES, COHERENT_SCENARIOS)

        x_base = fam_idx * col_w
        x_inst = fam_idx * col_w + gap
        x_center = fam_idx * col_w + gap / 2

        base_prefs = [to_pref(base_fav[c]) for c in COUNTRIES]
        inst_prefs = [to_pref(inst_fav[c]) for c in COUNTRIES]
        spread_base = np.std(base_prefs)
        spread_inst = np.std(inst_prefs)

        to_label = []
        for country in COUNTRIES:
            bp = to_pref(base_fav[country])
            ip = to_pref(inst_fav[country])
            color = COUNTRY_COLORS[country]
            ax.plot([x_base, x_inst], [bp, ip], color=color, alpha=0.45, linewidth=1.3)
            ax.scatter(x_base, bp, color=color, s=30, zorder=5,
                       edgecolors="white", linewidth=0.5)
            ax.scatter(x_inst, ip, color=color, s=60, zorder=6,
                       edgecolors="white", linewidth=0.7)
            if ip > 70 or ip < 30:
                to_label.append((ip, country, color))

        min_gap = 5.0
        for group, sign in (([t for t in to_label if t[0] > 50], +1),
                             ([t for t in to_label if t[0] <= 50], -1)):
            group.sort(key=lambda t: t[0], reverse=(sign > 0))
            last_y = None
            for ip, country, color in group:
                y = ip
                if last_y is not None and abs(y - last_y) < min_gap:
                    y = last_y - sign * min_gap
                ax.annotate(country, xy=(x_inst, ip),
                            xytext=(x_inst + 0.18, y),
                            fontsize=6.5, color=color, fontweight="bold",
                            va="center", ha="left")
                last_y = y

        # Title line: family name + [maker_bloc] + (lab)
        ax.text(x_center, 105, fam_name,
                ha="center", fontsize=9, fontweight="bold", clip_on=False)
        ax.text(x_center, 100,
                f"{lab}  [{maker}]",
                ha="center", fontsize=7.5, color="#333",
                fontweight="bold", clip_on=False)
        ax.text(x_center, 95.5,
                f"σ  {spread_base:.1f}  →  {spread_inst:.1f}",
                ha="center", fontsize=7, color="#555", style="italic",
                clip_on=False)
        if fam_idx < len(FAMILIES) - 1:
            ax.axvline(x_center + col_w / 2, color="gray",
                       linewidth=0.5, alpha=0.25)

    ax.axhline(50, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(-0.4, 50, "neutral", fontsize=7, color="gray",
            va="center", ha="right", style="italic")

    xticks, xlabels = [], []
    for i in range(len(FAMILIES)):
        xticks.extend([i * col_w, i * col_w + gap])
        xlabels.extend(["Base", "Inst"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=7)
    ax.set_ylabel("Preference  (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.9, (len(FAMILIES) - 1) * col_w + gap + 0.9)
    ax.set_title("A  ·  Post-training amplifies geopolitical bias",
                 fontsize=13, fontweight="bold", loc="left", pad=38)

    for country in COUNTRIES:
        ax.scatter([], [], color=COUNTRY_COLORS[country], s=55, label=country)
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — China favorability by maker bloc, post-trained models
# ─────────────────────────────────────────────────────────────────────────────

def panel_b(ax):
    data = []
    for fam_name, base, inst, maker, lab in FAMILIES:
        cb_base = china_favour(base, "EN")
        cb_inst = china_favour(inst, "EN")
        data.append({"family": fam_name, "maker": maker, "lab": lab,
                     "bloc": MAKER_BLOC[maker], "inst_key": inst,
                     "base": cb_base, "inst": cb_inst, "delta": cb_inst - cb_base})

    df = pd.DataFrame(data).sort_values("delta").reset_index(drop=True)
    y = np.arange(len(df))

    for i, row in df.iterrows():
        color = MAKER_COLOR[row["bloc"]]
        low_comp = row["inst_key"] in LOW_COMPLIANCE_INSTRUCT
        hatch = "//" if low_comp else None
        edge = "#666" if low_comp else "black"
        lw = 1.1 if low_comp else 0.6
        ax.barh(i, row["delta"], 0.65, color=color, alpha=0.9,
                edgecolor=edge, linewidth=lw, hatch=hatch)
        sign = "+" if row["delta"] >= 0 else ""
        ax.text(row["delta"] + (0.12 if row["delta"] >= 0 else -0.12),
                i, f"{sign}{row['delta']:.2f}",
                va="center", ha="left" if row["delta"] >= 0 else "right",
                fontsize=9, color="#222", fontweight="bold")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['family']}  ({r['lab']})"
                        for _, r in df.iterrows()], fontsize=9)
    ax.invert_yaxis()

    xmax = max(abs(df["delta"].min()), abs(df["delta"].max())) * 1.55
    ax.set_xlim(-xmax, xmax)
    ax.set_ylim(len(df) - 0.5, -0.7)
    ax.set_xlabel("Post-training Δ in China favorability  (log-odds)", fontsize=10)
    ax.set_title("B  ·  Direction of amplification follows the maker",
                 fontsize=12, fontweight="bold", loc="left", pad=12)

    western = plt.Rectangle((0, 0), 1, 1, facecolor=MAKER_COLOR["Western"],
                            alpha=0.9, edgecolor="black", linewidth=0.6)
    chinese = plt.Rectangle((0, 0), 1, 1, facecolor=MAKER_COLOR["Chinese"],
                            alpha=0.9, edgecolor="black", linewidth=0.6)
    lowc = plt.Rectangle((0, 0), 1, 1, facecolor="white",
                          alpha=0.9, edgecolor="#666", linewidth=1.1, hatch="//")
    ax.legend([western, chinese, lowc],
              ["Western-made", "Chinese-made",
               "Low-compliance (< 0.1)"],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, ncol=3)


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — language trigger: EN → ZH shift on post-trained models
# ─────────────────────────────────────────────────────────────────────────────

def panel_c(ax):
    rows = []
    for fam_name, _, inst, maker, lab in FAMILIES:
        rows.append({
            "family": fam_name, "maker": maker, "lab": lab,
            "bloc": MAKER_BLOC[maker], "inst_key": inst,
            "EN": china_favour(inst, "EN"),
            "ZH": china_favour(inst, "ZH"),
        })
    df = pd.DataFrame(rows)
    df["shift"] = df["ZH"] - df["EN"]
    df = df.sort_values("shift").reset_index(drop=True)
    y = np.arange(len(df))

    # Two-axis layout: leave right margin for Δ column
    data_min = df[["EN", "ZH"]].values.min()
    data_max = df[["EN", "ZH"]].values.max()
    pad = (data_max - data_min) * 0.10
    x_left = data_min - pad
    delta_col = data_max + (data_max - data_min) * 0.35

    for i, row in df.iterrows():
        color = MAKER_COLOR[row["bloc"]]
        low_comp = row["inst_key"] in LOW_COMPLIANCE_INSTRUCT
        # arrow from EN to ZH (dashed for low-compliance)
        linestyle = "--" if low_comp else "-"
        alpha = 0.55 if low_comp else 0.85
        ax.annotate("",
                    xy=(row["ZH"], i), xytext=(row["EN"], i),
                    arrowprops=dict(arrowstyle="->,head_width=0.35,head_length=0.6",
                                    color=color, lw=2.2, alpha=alpha,
                                    linestyle=linestyle),
                    zorder=2)
        # EN: hollow circle
        ax.scatter(row["EN"], i, color="white", s=70, zorder=4,
                   edgecolors=color, linewidths=1.4,
                   alpha=alpha)
        # ZH: filled circle (hollow with hatch if low compliance)
        if low_comp:
            ax.scatter(row["ZH"], i, color="white", s=90, zorder=4,
                       edgecolors=color, linewidths=2.0, hatch="///")
        else:
            ax.scatter(row["ZH"], i, color=color, s=90, zorder=4,
                       edgecolors="black", linewidths=0.8)
        # Δ annotation in right margin column
        sign = "+" if row["shift"] >= 0 else ""
        ax.text(delta_col, i, f"{sign}{row['shift']:.2f}",
                va="center", ha="right", fontsize=9, color="#222",
                fontweight="bold")

    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['family']}  ({r['lab']})"
                        for _, r in df.iterrows()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(x_left, delta_col + pad * 0.5)
    ax.set_ylim(len(df) - 0.5, -0.9)
    ax.set_xlabel("China favorability  (log-odds)", fontsize=10)
    ax.set_title("C  ·  Chinese-language prompt shifts bias pro-China",
                 fontsize=12, fontweight="bold", loc="left", pad=12)
    ax.text(delta_col, -0.75, "Δ (ZH−EN)", ha="right", fontsize=9,
            color="#555", style="italic", fontweight="bold")

    # Legend for EN/ZH markers
    en_proxy = plt.Line2D([0], [0], marker="o", linestyle="", markersize=8,
                          markerfacecolor="white", markeredgecolor="#444",
                          markeredgewidth=1.4, label="English prompt")
    zh_proxy = plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                          markerfacecolor="#444", markeredgecolor="black",
                          markeredgewidth=0.8, label="Chinese prompt")
    lowc_proxy = plt.Line2D([0], [0], marker="o", linestyle="", markersize=9,
                             markerfacecolor="white", markeredgecolor="#999",
                             markeredgewidth=1.8, label="Low compliance")
    ax.legend(handles=[en_proxy, zh_proxy, lowc_proxy],
              fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18),
              frameon=False, ncol=3)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(20, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.5, wspace=0.3,
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

    fig.suptitle(
        "It's the humans, not the data  ·  Post-training shifts geopolitical bias in the direction of the maker — unevenly",
        fontsize=15, fontweight="bold", y=0.995,
    )

    fig.text(0.5, 0.002,
             f"n = {len(COHERENT_SCENARIOS)} scenarios (justified/unjustified sign-flip ≥70% across 14 models × 3 languages)  ·  "
             f"7 model families from 7 labs: Western-made [FR/US] and Chinese-made [CN]",
             ha="center", fontsize=9, color="#555", style="italic")

    out = RESULTS / "plots" / "main_figure.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print(f"Coherent scenarios: {len(COHERENT_SCENARIOS)}")


if __name__ == "__main__":
    main()
