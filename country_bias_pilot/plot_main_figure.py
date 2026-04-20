#!/usr/bin/env python3
"""Main paper figure: post-training amplifies geopolitical bias, maker-aligned and
language-triggered.

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
    ("Mistral 7B",  "mistral-7b",  "mistral-7b-inst",  "FR"),
    ("LLaMA 3 8B",  "llama3-8b",   "llama3-8b-inst",   "US"),
    ("Gemma 4 8B",  "gemma4-8b",   "gemma4-8b-it",     "US"),
    ("Qwen 2.5 7B", "qwen2.5-7b",  "qwen2.5-7b-inst",  "CN"),
    ("Yi 1.5 9B",   "yi1.5-9b",    "yi1.5-9b-chat",    "CN"),
    ("GLM 4 9B",    "glm4-9b",     "glm4-9b-chat",     "CN"),
]

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
    all_models = [m for _, base, inst, _ in FAMILIES for m in (base, inst)]
    for m in all_models:
        for lang, d in LANG_DIRS.items():
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


def china_favour(model, lang, scens=COHERENT_SCENARIOS):
    """Mean China-signed 'justified' bias across coherent scenarios."""
    f = RESULTS / LANG_DIRS[lang] / f"{model}_raw.csv"
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
    for fam_idx, (fam_name, base_key, inst_key, maker) in enumerate(FAMILIES):
        base_fav = favour(RESULTS / "gpu_bias" / f"{base_key}_raw.csv",
                          COUNTRIES, COHERENT_SCENARIOS)
        inst_fav = favour(RESULTS / "gpu_bias" / f"{inst_key}_raw.csv",
                          COUNTRIES, COHERENT_SCENARIOS)

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
            ax.scatter(x_base, bp, color=color, s=38, zorder=5,
                       edgecolors="white", linewidth=0.6)
            ax.scatter(x_inst, ip, color=color, s=75, zorder=6,
                       edgecolors="white", linewidth=0.8)
            if ip > 68 or ip < 32:
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
                            xytext=(x_inst + 0.22, y),
                            fontsize=7, color=color, fontweight="bold",
                            va="center", ha="left")
                last_y = y

        maker_tag = f"[{maker}]"
        ax.text(fam_idx * 3 + 0.6, 104, f"{fam_name}  {maker_tag}",
                ha="center", fontsize=10, fontweight="bold", clip_on=False)
        ax.text(fam_idx * 3 + 0.6, 99.5,
                f"spread  {spread_base:.1f}  →  {spread_inst:.1f}",
                ha="center", fontsize=7.5, color="#555", style="italic")
        if fam_idx < len(FAMILIES) - 1:
            ax.axvline(fam_idx * 3 + 2.1, color="gray", linewidth=0.5, alpha=0.25)

    ax.axhline(50, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(-0.3, 50, "neutral", fontsize=7.5, color="gray",
            va="center", ha="right", style="italic")

    xticks, xlabels = [], []
    for i in range(len(FAMILIES)):
        xticks.extend([i * 3, i * 3 + 1.2])
        xlabels.extend(["Base", "Post-trained"])
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=8)
    ax.set_ylabel("Preference  (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.set_xlim(-0.8, (len(FAMILIES) - 1) * 3 + 2.2)
    ax.set_title("A  ·  Post-training amplifies geopolitical bias",
                 fontsize=13, fontweight="bold", loc="left", pad=30)

    for country in COUNTRIES:
        ax.scatter([], [], color=COUNTRY_COLORS[country], s=55, label=country)
    ax.legend(ncol=8, loc="upper center", bbox_to_anchor=(0.5, -0.12),
              fontsize=8, frameon=False, handletextpad=0.3, columnspacing=1.2)


# ─────────────────────────────────────────────────────────────────────────────
# Panel B — China favorability by maker bloc, post-trained models
# ─────────────────────────────────────────────────────────────────────────────

def panel_b(ax):
    data = []
    for fam_name, base, inst, maker in FAMILIES:
        cb_base = china_favour(base, "EN")
        cb_inst = china_favour(inst, "EN")
        data.append({"family": fam_name, "maker": maker, "bloc": MAKER_BLOC[maker],
                     "base": cb_base, "inst": cb_inst, "delta": cb_inst - cb_base})

    df = pd.DataFrame(data).sort_values("delta")
    y = np.arange(len(df))

    for i, row in df.reset_index(drop=True).iterrows():
        color = MAKER_COLOR[row["bloc"]]
        ax.barh(i, row["delta"], 0.65, color=color, alpha=0.9,
                edgecolor="black", linewidth=0.6)
        # annotate maker country
        ax.text(row["delta"] + (0.15 if row["delta"] >= 0 else -0.15),
                i, f"[{row['maker']}]",
                va="center", ha="left" if row["delta"] >= 0 else "right",
                fontsize=8, color="#444")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(df["family"].values, fontsize=10)
    ax.invert_yaxis()

    xmax = max(abs(df["delta"].min()), abs(df["delta"].max())) * 1.35
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("Post-training Δ in China favorability  (log-odds)", fontsize=10)
    ax.set_title("B  ·  Direction of amplification follows the maker",
                 fontsize=12, fontweight="bold", loc="left", pad=10)

    western = plt.Rectangle((0, 0), 1, 1, facecolor=MAKER_COLOR["Western"],
                            alpha=0.9, edgecolor="black", linewidth=0.6)
    chinese = plt.Rectangle((0, 0), 1, 1, facecolor=MAKER_COLOR["Chinese"],
                            alpha=0.9, edgecolor="black", linewidth=0.6)
    ax.legend([western, chinese],
              ["Western-made (3 models)", "Chinese-made (3 models)"],
              fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.15),
              frameon=False, ncol=2)

    ax.text(-xmax * 0.96, -0.8, "← anti-China after post-training",
            fontsize=8, color="#444", style="italic")
    ax.text(xmax * 0.96, -0.8, "pro-China after post-training →",
            fontsize=8, color="#444", style="italic", ha="right")


# ─────────────────────────────────────────────────────────────────────────────
# Panel C — language trigger: EN → ZH shift on post-trained models
# ─────────────────────────────────────────────────────────────────────────────

def panel_c(ax):
    rows = []
    for fam_name, _, inst, maker in FAMILIES:
        rows.append({
            "family": fam_name, "maker": maker, "bloc": MAKER_BLOC[maker],
            "EN": china_favour(inst, "EN"),
            "ZH": china_favour(inst, "ZH"),
        })
    df = pd.DataFrame(rows)
    df["shift"] = df["ZH"] - df["EN"]
    df = df.sort_values("shift")
    y = np.arange(len(df))

    for i, row in df.reset_index(drop=True).iterrows():
        color = MAKER_COLOR[row["bloc"]]
        ax.plot([row["EN"], row["ZH"]], [i, i],
                color=color, linewidth=2.5, alpha=0.65, zorder=2)
        ax.scatter(row["EN"], i, color="white", s=95, zorder=4,
                   edgecolors=color, linewidths=1.5)
        ax.text(row["EN"], i, "EN", ha="center", va="center",
                fontsize=6.5, color=color, fontweight="bold", zorder=5)
        ax.scatter(row["ZH"], i, color=color, s=130, zorder=4,
                   edgecolors="black", linewidths=0.8)
        ax.text(row["ZH"], i, "ZH", ha="center", va="center",
                fontsize=6.5, color="white", fontweight="bold", zorder=5)
        sign = "+" if row["shift"] >= 0 else ""
        ax.text(max(row["EN"], row["ZH"]) + 0.15, i,
                f"Δ = {sign}{row['shift']:.2f}",
                va="center", fontsize=8, color="#333")

    ax.axvline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
    ax.text(0, len(df) - 0.3, "neutral", fontsize=7.5, color="gray",
            ha="center", va="top", style="italic")

    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['family']}  [{r['maker']}]" for _, r in df.iterrows()],
                       fontsize=9)
    ax.invert_yaxis()
    xmax = max(abs(df[["EN", "ZH"]].values.min()),
               abs(df[["EN", "ZH"]].values.max())) * 1.25
    ax.set_xlim(-xmax, xmax)
    ax.set_xlabel("China favorability  (log-odds)", fontsize=10)
    ax.set_title("C  ·  Prompting in Chinese shifts post-trained models pro-China",
                 fontsize=12, fontweight="bold", loc="left", pad=10)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    fig = plt.figure(figsize=(18, 14))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.32,
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
        "Aligned to whom?  Post-training implants a maker-aligned bias, triggered by linguistic identity",
        fontsize=15, fontweight="bold", y=0.995,
    )

    fig.text(0.5, 0.002,
             f"n = {len(COHERENT_SCENARIOS)} scenarios (justified/unjustified sign-flip ≥70% across 12 models × 3 languages)  ·  "
             f"6 model families: Western-made [FR/US] and Chinese-made [CN]",
             ha="center", fontsize=9, color="#555", style="italic")

    out = RESULTS / "plots" / "main_figure.png"
    out.parent.mkdir(exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")
    print(f"Coherent scenarios: {len(COHERENT_SCENARIOS)}")


if __name__ == "__main__":
    main()
