#!/usr/bin/env python3
"""Generate all 9 publication figures for the TISMIR paper.

Reads regenerated CSVs in `results/` (written by 01-05) plus feature parquets
from `artifacts/` (for fig6 pattern counts). Writes PDF+PNG to
`results/figures/`.

Adapted from the original `scripts/generate_paper_figures.py`. Style, panel
geometry, and colourblind palette are identical.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS = REPO_ROOT / "results"
ARTIFACTS = REPO_ROOT / "artifacts"
FIGURES = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)

CORPUS_COLORS = {
    "symbtr": "#d95f02",
    "the_session": "#7570b3",
    "essen": "#1b9e77",
}
CORPUS_LABELS = {
    "symbtr": "SymbTr (makam)",
    "the_session": "The Session (tune type)",
    "essen": "Essen (collection)",
}

VP_ORDER = ["M", "R", "B", "MR", "MB", "RB", "MRB"]
VP_LABELS = {
    "M": "M\n(melodic)",
    "R": "R\n(rhythmic)",
    "B": "B\n(metric)",
    "MR": "MR",
    "MB": "MB",
    "RB": "RB",
    "MRB": "MRB\n(all)",
}

SINGLE_COL = 3.5
DOUBLE_COL = 7.0

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


def save(fig, name: str, close: bool = True) -> None:
    for ext in (".pdf", ".png"):
        fig.savefig(FIGURES / f"{name}{ext}")
    print(f"  Saved: {name}")
    if close:
        plt.close(fig)


def _load_viewpoint_comparison() -> pd.DataFrame | None:
    """Build a viewpoint_comparison-like DataFrame from per-VP classification summaries."""
    rows = []
    for corpus in ["symbtr", "the_session", "essen"]:
        for vp in VP_ORDER:
            summary_path = RESULTS / corpus / vp / "classification" / "summary.csv"
            if not summary_path.exists():
                continue
            df = pd.read_csv(summary_path)
            real = df[~df["classifier"].isin(["majority", "stratified"])]
            if real.empty:
                continue
            best = real.loc[real["f1_macro_mean"].idxmax()]
            rows.append({
                "corpus": corpus,
                "viewpoint_config": vp,
                "best_feature_config": best["feature_config"],
                "best_classifier": best["classifier"],
                "best_f1_macro": best["f1_macro_mean"],
            })
    if not rows:
        return None
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------
# Figure 1
# ----------------------------------------------------------------------------
def _load_ablation_table() -> pd.DataFrame | None:
    """Viewpoint x corpus `combined` F1 under each corpus's fixed classifier.

    Written by 02_run_ablation_statistics.py. This is the quantity the paper's viewpoint x corpus
    heatmap and rank-slope chart report; an earlier version of these two figures used
    `best_f1_macro` (best over feature configs *and* classifiers), which is a different quantity
    and does not match their captions.
    """
    path = RESULTS / "viewpoint_comparison" / "ablation_significance.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df.rename(columns={"vp_config": "viewpoint_config", "combined_mean": "best_f1_macro"})


def fig1_heatmap() -> None:
    df = _load_ablation_table()
    if df is None:
        df = _load_viewpoint_comparison()
    if df is None:
        print("  SKIP fig1: no classification summaries found")
        return
    pivot = df.pivot_table(values="best_f1_macro", index="viewpoint_config", columns="corpus")
    pivot = pivot.reindex(VP_ORDER).rename(columns=CORPUS_LABELS)

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.65, 4.0))
    sns.heatmap(
        pivot, annot=True, fmt=".3f", annot_kws={"size": 11},
        cmap="YlOrRd", ax=ax, linewidths=0.5,
        vmin=pivot.values.min() - 0.02, vmax=pivot.values.max() + 0.02,
        cbar_kws={"label": "F1-macro", "shrink": 0.8},
    )
    ax.set_yticklabels([VP_LABELS.get(vp, vp) for vp in VP_ORDER], rotation=0, fontsize=11)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("Viewpoint configuration")
    save(fig, "fig1_vp_corpus_heatmap")


# ----------------------------------------------------------------------------
# Figure 2
# ----------------------------------------------------------------------------
def fig2_slope_chart() -> None:
    df = _load_ablation_table()
    if df is None:
        df = _load_viewpoint_comparison()
    if df is None:
        print("  SKIP fig2: no classification summaries found")
        return
    aligned_dim = {"symbtr": "M", "the_session": "B", "essen": None}
    row_labels = {
        "symbtr": "SymbTr\n(makam)",
        "the_session": "The Session\n(tune type)",
        "essen": "Essen\n(collection)",
    }

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.75, 2.4))
    corpora = ["symbtr", "the_session", "essen"]
    for row, corpus in enumerate(corpora):
        subset = (
            df[df["corpus"] == corpus]
            .sort_values("best_f1_macro", ascending=False)
            .reset_index(drop=True)
        )
        color = CORPUS_COLORS[corpus]
        dim = aligned_dim[corpus]
        ink = "0.1" if dim else "0.45"
        for rank, rec in subset.iterrows():
            vp = rec["viewpoint_config"]
            aligned = dim is not None and dim in vp
            ax.add_patch(matplotlib.patches.FancyBboxPatch(
                (rank + 0.07, row + 0.1), 0.86, 0.8,
                boxstyle="round,pad=0,rounding_size=0.08",
                facecolor=matplotlib.colors.to_rgba(color, 0.25) if aligned else "white",
                edgecolor=color if aligned else "0.75", linewidth=1.0,
            ))
            ax.text(rank + 0.5, row + 0.42, vp, ha="center", va="center",
                    fontsize=11, fontweight="bold", color=ink)
            ax.text(rank + 0.5, row + 0.72, f"{rec['best_f1_macro']:.3f}", ha="center",
                    va="center", fontsize=8, color="0.4")
        if dim is None and not subset.empty:
            span = subset["best_f1_macro"].max() - subset["best_f1_macro"].min()
            row_labels[corpus] += f"\nspan {span:.2f}"

    ax.set_xlim(0, 7)
    ax.set_ylim(len(corpora), 0)
    ax.set_xticks(np.arange(7) + 0.5)
    ax.set_xticklabels([str(k) for k in range(1, 8)], fontsize=10)
    ax.set_yticks(np.arange(len(corpora)) + 0.5)
    ax.set_yticklabels([row_labels[c] for c in corpora], fontsize=9)
    ax.set_xlabel("Within-corpus rank (1 = best combined F1-macro)", fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()
    fig.tight_layout()

    save(fig, "fig2_vp_rank_slope")


# ----------------------------------------------------------------------------
# Figure 3
# ----------------------------------------------------------------------------
def fig3_delta_bars() -> None:
    sig_path = RESULTS / "viewpoint_comparison" / "ablation_significance.csv"
    if not sig_path.exists():
        print("  SKIP fig3: ablation_significance.csv not found")
        return
    df = pd.read_csv(sig_path)
    corpora = ["symbtr", "the_session", "essen"]

    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.9), sharey=True)
    for ax, corpus in zip(axes, corpora):
        subset = df[df["corpus"] == corpus].set_index("vp_config").reindex(VP_ORDER).reset_index()
        colors = [CORPUS_COLORS[corpus] if d > 0 else "#999999" for d in subset["delta"]]
        ax.bar(range(len(VP_ORDER)), subset["delta"],
               color=colors, edgecolor="black", linewidth=0.3, alpha=0.8)
        for i, (_, row) in enumerate(subset.iterrows()):
            p = row.get("bonferroni_p", row.get("wilcoxon_p", 1.0))
            star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
            if star:
                y_pos = row["delta"] + 0.004 if row["delta"] >= 0 else row["delta"] - 0.014
                ax.text(i, y_pos, star, ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(range(len(VP_ORDER)))
        ax.set_xticklabels(VP_ORDER, fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(CORPUS_LABELS[corpus], fontsize=9)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel(r"$\Delta$ F1-macro" + "\n(combined \u2212 global)", fontsize=9)
    fig.tight_layout()
    save(fig, "fig3_delta_bars_significance")


# ----------------------------------------------------------------------------
# Figure 4
# ----------------------------------------------------------------------------
def fig4_vp_sensitivity() -> None:
    sig_path = RESULTS / "viewpoint_comparison" / "ablation_significance.csv"
    if not sig_path.exists():
        print("  SKIP fig4: ablation_significance.csv not found")
        return
    df = pd.read_csv(sig_path)
    corpora = ["symbtr", "the_session", "essen"]
    sigmas = {c: float(np.std(df[df["corpus"] == c]["combined_mean"].values, ddof=1))
              for c in corpora}

    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.6))
    x = range(len(corpora))
    ax.bar(x, [sigmas[c] for c in corpora],
           color=[CORPUS_COLORS[c] for c in corpora],
           edgecolor="black", linewidth=0.5, alpha=0.85)
    for i, c in enumerate(corpora):
        ax.text(i, sigmas[c] + 0.001, f"{sigmas[c]:.4f}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([CORPUS_LABELS[c].split(" (")[0] for c in corpora], fontsize=9)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylabel(r"$\sigma$ of combined F1" + "\nacross VP configs", fontsize=9)
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    save(fig, "fig4_vp_sensitivity")


# ----------------------------------------------------------------------------
# Figure 5
# ----------------------------------------------------------------------------
def fig5_importance_shift() -> None:
    panels = [
        ("symbtr", "M", "SymbTr: MRB vs M (melodic)"),
        ("the_session", "B", "Session: MRB vs B (metric)"),
    ]
    found = []
    for corpus, best_vp, title in panels:
        mrb_path = RESULTS / corpus / "interpretability" / "importance_combined.csv"
        vp_path = RESULTS / corpus / best_vp / "interpretability" / "importance_combined.csv"
        if mrb_path.exists() and vp_path.exists():
            found.append((corpus, best_vp, title, mrb_path, vp_path))

    if not found:
        print("  SKIP fig5: need MRB + best VP importance files")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(DOUBLE_COL, 4.2))
    if len(found) == 1:
        axes = [axes]

    top_k = 12
    for ax, (corpus, best_vp, title, mrb_path, vp_path) in zip(axes, found):
        mrb = pd.read_csv(mrb_path).drop_duplicates(subset="feature").head(top_k)
        vp = pd.read_csv(vp_path).drop_duplicates(subset="feature")

        all_features = mrb["feature"].tolist()
        vp_top = vp.head(top_k)["feature"].tolist()
        for f in vp_top:
            if f not in all_features:
                all_features.append(f)
        all_features = all_features[:top_k]

        def _short(f: str) -> str:
            f = f.replace("pattern_", "pat.")
            f = f.replace("_patterns", " (pat)")
            f = f.replace("rhythmic_", "rhythmic ")
            f = f.replace("melodic_", "melodic ")
            f = f.replace("sectional_", "sectional ")
            f = f.replace("_", " ")
            return f
        short = {f: _short(f) for f in all_features}
        mrb_dict = dict(zip(mrb["feature"], mrb["importance_mean"]))
        vp_dict = dict(zip(vp["feature"], vp["importance_mean"]))

        features = all_features[::-1]
        y = np.arange(len(features))
        width = 0.42
        mrb_vals = [mrb_dict.get(f, 0) for f in features]
        vp_vals = [vp_dict.get(f, 0) for f in features]

        ax.barh(y - width / 2, mrb_vals, width, label="MRB",
                color="#8da0cb", alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.barh(y + width / 2, vp_vals, width, label=best_vp,
                color=CORPUS_COLORS[corpus], alpha=0.8, edgecolor="white", linewidth=0.3)

        data_max = max(max(mrb_vals), max(vp_vals)) if (mrb_vals + vp_vals) else 1.0
        ax.set_xlim(0, data_max * 1.12)
        ax.set_yticks(y)
        ax.set_yticklabels([short[f] for f in features], fontsize=8)
        ax.tick_params(axis="x", labelsize=8)
        ax.set_xlabel("Permutation importance", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=9, loc="lower right", framealpha=0.95, edgecolor="0.8")
        ax.set_ylim(-0.5, len(features) - 0.5)
        ax.set_axisbelow(True)
        ax.xaxis.grid(True, alpha=0.45, linewidth=0.6, linestyle="--", color="#888888")
        ax.yaxis.grid(False)

    fig.tight_layout()
    save(fig, "fig5_importance_shift")


# ----------------------------------------------------------------------------
# Figure 6
# ----------------------------------------------------------------------------
def fig6_pattern_count_vs_f1() -> None:
    """Pattern count vs F1 scatter — reads structural parquets from artifacts/."""
    vc = _load_viewpoint_comparison()
    if vc is None:
        print("  SKIP fig6: no classification summaries found")
        return

    rows = []
    for corpus in ["symbtr", "the_session", "essen"]:
        for vp in VP_ORDER:
            struct_path = ARTIFACTS / "features" / f"{corpus}_{vp}_structural.parquet"
            if not struct_path.exists():
                continue
            struct = pd.read_parquet(struct_path)
            if "n_patterns" not in struct.columns:
                continue
            mean_pats = float(struct["n_patterns"].mean())
            f1_row = vc[(vc["corpus"] == corpus) & (vc["viewpoint_config"] == vp)]
            if f1_row.empty:
                continue
            rows.append({
                "corpus": corpus, "vp": vp,
                "mean_patterns": mean_pats,
                "f1": float(f1_row.iloc[0]["best_f1_macro"]),
            })
    if not rows:
        print("  SKIP fig6: no pattern count data")
        return

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, 3.3))
    for corpus in df["corpus"].unique():
        subset = df[df["corpus"] == corpus]
        ax.scatter(subset["mean_patterns"], subset["f1"],
                   color=CORPUS_COLORS[corpus],
                   label=CORPUS_LABELS.get(corpus, corpus),
                   s=50, edgecolors="black", linewidths=0.4, alpha=0.85,
                   zorder=3)

    texts = []
    for _, row in df.iterrows():
        t = ax.text(row["mean_patterns"], row["f1"], row["vp"],
                    fontsize=8, ha="left", va="bottom", zorder=4)
        texts.append(t)
    adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="0.5", lw=0.5),
                expand=(1.3, 1.5), force_text=(0.5, 0.8))

    ymin, ymax = ax.get_ylim()
    ax.set_ylim(ymin, ymax + 0.06)
    ax.set_xlabel("Mean patterns per piece", fontsize=9)
    ax.set_ylabel("F1-macro", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95, edgecolor="0.8")
    ax.grid(alpha=0.2, linewidth=0.5)
    save(fig, "fig6_pattern_count_vs_f1")


# ----------------------------------------------------------------------------
# Figure 7
# ----------------------------------------------------------------------------
def fig7_per_class_improvement() -> None:
    panels = [("symbtr", "M"), ("the_session", "B")]
    found = []
    for corpus, vp in panels:
        path = RESULTS / "viewpoint_comparison" / f"per_class_improvement_{corpus}.csv"
        if path.exists():
            found.append((corpus, vp, pd.read_csv(path)))

    if not found:
        print("  SKIP fig7: no per_class_improvement files")
        return

    fig, axes = plt.subplots(1, len(found), figsize=(DOUBLE_COL * 0.75, 3.2))
    if len(found) == 1:
        axes = [axes]
    short_corpus = {"symbtr": "SymbTr", "the_session": "The Session", "essen": "Essen"}

    for ax, (corpus, vp, df) in zip(axes, found):
        configs = df["feature_config"].tolist()
        f1s = df["f1_macro_mean"].tolist()
        stds = df["f1_macro_std"].tolist()
        colors = ["#999999" if c == "global" else CORPUS_COLORS[corpus] for c in configs]
        ax.bar(range(len(configs)), f1s, yerr=stds,
               color=colors, edgecolor="black", linewidth=0.3,
               capsize=3, alpha=0.85, width=0.6)
        for i, (f1, s) in enumerate(zip(f1s, stds)):
            ax.text(i, f1 + s + 0.01, f"{f1:.3f}", ha="center", fontsize=8)
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, fontsize=9, rotation=20, ha="right")
        ax.tick_params(axis="y", labelsize=8)
        ax.set_title(f"{short_corpus[corpus]} (VP={vp})", fontsize=9)
        ax.grid(axis="y", alpha=0.2, linewidth=0.5)

    axes[0].set_ylabel("F1-macro", fontsize=9)
    fig.tight_layout()
    save(fig, "fig7_per_class_improvement")


# ----------------------------------------------------------------------------
# Figure 8
# ----------------------------------------------------------------------------
def fig8_classifier_robustness() -> None:
    path = RESULTS / "viewpoint_comparison" / "classifier_robustness.csv"
    if not path.exists():
        print("  SKIP fig8: classifier_robustness.csv not found")
        return
    df = pd.read_csv(path)
    corpora = [c for c in ["symbtr", "the_session", "essen"] if c in df["corpus"].values]
    all_classifiers = df["classifier"].unique()

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.75, 3.5))
    n_corpora = len(corpora)
    n_clf = len(all_classifiers)
    bar_width = 0.8 / n_corpora
    x = np.arange(n_clf)

    for i, corpus in enumerate(corpora):
        subset = df[df["corpus"] == corpus]
        deltas = []
        for clf in all_classifiers:
            clf_row = subset[subset["classifier"] == clf]
            deltas.append(clf_row.iloc[0]["delta"] if not clf_row.empty else 0)
        offset = (i - (n_corpora - 1) / 2) * bar_width
        ax.bar(x + offset, deltas, bar_width,
               label=CORPUS_LABELS.get(corpus, corpus),
               color=CORPUS_COLORS[corpus],
               edgecolor="black", linewidth=0.3, alpha=0.85)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() if len(c) <= 3 else c.capitalize()
                        for c in all_classifiers], fontsize=10)
    ax.set_xlabel("Classifier")
    ax.set_ylabel(r"$\Delta$ F1-macro" + "\n(combined \u2212 global)")
    ax.legend(fontsize=9, loc="best", framealpha=0.95, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    save(fig, "fig8_classifier_robustness")


# ----------------------------------------------------------------------------
# Figure 9
# ----------------------------------------------------------------------------
def fig9_essen_diagnostic() -> None:
    path = RESULTS / "essen" / "essen_diagnostic" / "comparison.csv"
    if not path.exists():
        print("  SKIP fig9: essen_diagnostic/comparison.csv not found")
        return
    df = pd.read_csv(path)
    classifiers = df["classifier"].tolist()
    n_clf = len(classifiers)
    configs = ["global_f1", "combined_no_bop_f1", "combined_f1"]
    config_labels = ["Global (D)", "No BoP (A+B+D)", "Combined (A+B+C+D)"]
    config_colors = ["#636363", "#41b6c4", "#d6604d"]

    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.75, 3.5))
    bar_width = 0.25
    x = np.arange(n_clf)
    for i, (col, label, color) in enumerate(zip(configs, config_labels, config_colors)):
        vals = df[col].tolist()
        offset = (i - 1) * bar_width
        ax.bar(x + offset, vals, bar_width,
               label=label, color=color,
               edgecolor="black", linewidth=0.3, alpha=0.85)

    global_vals = df["global_f1"].tolist()
    combined_vals = df["combined_f1"].tolist()
    for j in range(n_clf):
        delta = combined_vals[j] - global_vals[j]
        sign = "+" if delta >= 0 else ""
        color = "#2a7f2a" if delta >= 0 else "#c0392b"
        y_pos = max(global_vals[j], combined_vals[j]) + 0.008
        ax.text(x[j], y_pos, f"{sign}{delta:.3f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color=color)

    ax.set_xticks(x)
    ax.set_xticklabels([c.upper() if len(c) <= 3 else c.capitalize()
                        for c in classifiers], fontsize=10)
    ax.set_xlabel("Classifier")
    ax.set_ylabel("F1-macro")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.95, edgecolor="0.8")
    ax.grid(axis="y", alpha=0.2, linewidth=0.5)
    ymax = float(max(df[configs].max())) + 0.04
    ax.set_ylim(0, ymax)
    save(fig, "fig9_essen_diagnostic")


def main() -> None:
    print("=" * 60)
    print(f"Generating paper figures into: {FIGURES}")
    print("=" * 60)
    fig1_heatmap()
    fig2_slope_chart()
    fig3_delta_bars()
    fig4_vp_sensitivity()
    fig5_importance_shift()
    fig6_pattern_count_vs_f1()
    fig7_per_class_improvement()
    fig8_classifier_robustness()
    fig9_essen_diagnostic()
    print(f"\nAll figures saved to: {FIGURES}")


if __name__ == "__main__":
    main()
