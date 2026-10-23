"""Viewpoint ablation analysis: compare results across viewpoint configs."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, spearmanr

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALL_VIEWPOINT_CONFIGS = ["M", "R", "B", "MR", "MB", "RB", "MRB"]

VIEWPOINT_LABELS = {
    "M": "Melodic\n(SEQ_INT)",
    "R": "Rhythmic\n(CONTOUR_DUR)",
    "B": "Metric\n(BEAT_STR)",
    "MR": "Mel+Rhy",
    "MB": "Mel+Met",
    "RB": "Rhy+Met",
    "MRB": "All three\n(baseline)",
}


def collect_viewpoint_results(corpus_name: str) -> pd.DataFrame | None:
    """Collect classification summaries across viewpoint configs for a corpus.

    Returns a DataFrame with columns:
        viewpoint_config, feature_config, classifier, f1_macro_mean, f1_macro_std, ...
    or None if no results found.
    """
    rows = []
    for vp in ALL_VIEWPOINT_CONFIGS:
        summary_path = (
            PROJECT_ROOT / "results" / corpus_name / vp / "classification" / "summary.csv"
        )
        if summary_path.exists():
            df = pd.read_csv(summary_path)
            df["viewpoint_config"] = vp
            rows.append(df)

    if not rows:
        logger.warning("No viewpoint ablation results found for %s", corpus_name)
        return None

    result = pd.concat(rows, ignore_index=True)
    logger.info(
        "Collected viewpoint results for %s: %d rows across %d configs",
        corpus_name,
        len(result),
        result["viewpoint_config"].nunique(),
    )
    return result


def compare_viewpoint_configs(
    all_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Compare best F1 scores across viewpoint configs and corpora.

    Args:
        all_results: {corpus_name: viewpoint_results_df}

    Returns:
        DataFrame with columns: corpus, viewpoint_config, best_feature_config,
        best_classifier, best_f1_macro
    """
    rows = []
    for corpus, df in all_results.items():
        # Exclude baseline classifiers
        real = df[~df["classifier"].isin(["majority", "stratified"])]

        for vp in real["viewpoint_config"].unique():
            vp_data = real[real["viewpoint_config"] == vp]
            if vp_data.empty:
                continue
            best_row = vp_data.loc[vp_data["f1_macro_mean"].idxmax()]
            rows.append({
                "corpus": corpus,
                "viewpoint_config": vp,
                "best_feature_config": best_row["feature_config"],
                "best_classifier": best_row["classifier"],
                "best_f1_macro": best_row["f1_macro_mean"],
            })

    return pd.DataFrame(rows)


def get_best_config_per_corpus(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """For each corpus, find the best viewpoint config."""
    idx = comparison_df.groupby("corpus")["best_f1_macro"].idxmax()
    return comparison_df.loc[idx].reset_index(drop=True)


# ──────────────────────────────────────────────────────
# Visualization
# ──────────────────────────────────────────────────────

def plot_viewpoint_heatmap(
    comparison_df: pd.DataFrame,
    out_path: Path | None = None,
) -> None:
    """Heatmap: viewpoint config (rows) × corpus (columns), colored by F1."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    pivot = comparison_df.pivot_table(
        values="best_f1_macro",
        index="viewpoint_config",
        columns="corpus",
        aggfunc="max",
    )

    # Reorder rows by standard viewpoint order
    ordered = [vp for vp in ALL_VIEWPOINT_CONFIGS if vp in pivot.index]
    pivot = pivot.reindex(ordered)

    fig, ax = plt.subplots(figsize=(max(4, len(pivot.columns) * 1.5 + 1), len(ordered) * 0.6 + 1.5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        linewidths=0.5,
    )
    ax.set_title("Best F1 (macro) by Viewpoint Config × Corpus")
    ax.set_ylabel("Viewpoint Config")
    ax.set_xlabel("Corpus")

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        logger.info("Saved viewpoint heatmap: %s", out_path)
    plt.close(fig)


def plot_viewpoint_bar_chart(
    results_df: pd.DataFrame,
    out_path: Path | None = None,
    title: str = "Viewpoint Ablation",
    feature_config: str = "pattern_all",
) -> None:
    """Bar chart: F1 per viewpoint config for a single corpus.

    Shows the best classifier result for each viewpoint config,
    using the specified feature config (default: pattern_all).
    Falls back to 'combined' if pattern_all not available.
    """
    import matplotlib.pyplot as plt

    real = results_df[~results_df["classifier"].isin(["majority", "stratified"])]

    # Try requested feature config, fall back to combined
    if feature_config in real["feature_config"].unique():
        subset = real[real["feature_config"] == feature_config]
    elif "combined" in real["feature_config"].unique():
        subset = real[real["feature_config"] == "combined"]
        feature_config = "combined"
    else:
        # Use best across all feature configs
        subset = real
        feature_config = "best"

    # Best classifier per viewpoint config
    best_per_vp = subset.groupby("viewpoint_config").apply(
        lambda g: g.loc[g["f1_macro_mean"].idxmax()], include_groups=False
    ).reset_index()

    # Order by standard viewpoint order
    ordered = [vp for vp in ALL_VIEWPOINT_CONFIGS if vp in best_per_vp["viewpoint_config"].values]
    best_per_vp = best_per_vp.set_index("viewpoint_config").reindex(ordered).reset_index()

    fig, ax = plt.subplots(figsize=(max(6, len(ordered) * 0.8 + 1), 4))

    # Color by number of viewpoints
    colors = []
    for vp in ordered:
        n = len(vp)
        if n == 1:
            colors.append("#66c2a5")  # single
        elif n == 2:
            colors.append("#fc8d62")  # pair
        else:
            colors.append("#8da0cb")  # triple

    bars = ax.bar(
        range(len(ordered)),
        best_per_vp["f1_macro_mean"],
        color=colors,
        edgecolor="black",
        linewidth=0.5,
    )

    # Add error bars if std available
    if "f1_macro_std" in best_per_vp.columns:
        ax.errorbar(
            range(len(ordered)),
            best_per_vp["f1_macro_mean"],
            yerr=best_per_vp["f1_macro_std"],
            fmt="none",
            ecolor="black",
            capsize=3,
        )

    # Add value labels on bars
    for i, (idx, row) in enumerate(best_per_vp.iterrows()):
        ax.text(
            i, row["f1_macro_mean"] + 0.005,
            f"{row['f1_macro_mean']:.3f}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels([VIEWPOINT_LABELS.get(vp, vp) for vp in ordered], fontsize=8)
    ax.set_ylabel("F1 (macro)")
    ax.set_title(f"{title}\nfeature_config={feature_config}")
    ax.set_ylim(0, min(1.0, best_per_vp["f1_macro_mean"].max() * 1.3))

    if out_path:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
        fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
        logger.info("Saved viewpoint bar chart: %s", out_path)
    plt.close(fig)


def viewpoint_feature_interaction(
    all_results: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Analyze which feature configs work best with which viewpoint configs.

    For each corpus × viewpoint_config, find the best feature_config.
    Returns a DataFrame showing the interaction.
    """
    rows = []
    for corpus, df in all_results.items():
        real = df[~df["classifier"].isin(["majority", "stratified"])]
        for vp in real["viewpoint_config"].unique():
            vp_data = real[real["viewpoint_config"] == vp]
            for fc in vp_data["feature_config"].unique():
                fc_data = vp_data[vp_data["feature_config"] == fc]
                best = fc_data.loc[fc_data["f1_macro_mean"].idxmax()]
                rows.append({
                    "corpus": corpus,
                    "viewpoint_config": vp,
                    "feature_config": fc,
                    "best_classifier": best["classifier"],
                    "f1_macro": best["f1_macro_mean"],
                })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────
# WP2b: VP Importance Comparison
# ──────────────────────────────────────────────────────

def compare_importance_across_viewpoints(
    corpus_name: str,
    vp_configs: list[str],
    top_k: int = 15,
) -> pd.DataFrame:
    """Compare feature importance across viewpoint configs for a corpus.

    For each VP config, loads importance_combined.csv and compares
    feature rank shifts.

    Returns DataFrame with columns:
        feature, {vp}_rank, {vp}_importance, ...
    """
    importance_dfs = {}
    for vp in vp_configs:
        path = (
            PROJECT_ROOT / "results" / corpus_name / vp
            / "interpretability" / "importance_combined.csv"
        )
        if path.exists():
            importance_dfs[vp] = pd.read_csv(path)

    if len(importance_dfs) < 2:
        logger.warning(
            "Need ≥2 VP importance files for %s, found %d",
            corpus_name, len(importance_dfs),
        )
        return pd.DataFrame()

    # Collect all features
    all_features = set()
    for df in importance_dfs.values():
        all_features.update(df["feature"].tolist())
    all_features = sorted(all_features)

    # Build comparison table
    result_rows = []
    for feat in all_features:
        row = {"feature": feat}
        for vp, df in importance_dfs.items():
            match = df[df["feature"] == feat]
            if not match.empty:
                row[f"{vp}_rank"] = int(match.iloc[0]["rank"])
                row[f"{vp}_importance"] = float(match.iloc[0]["importance_mean"])
            else:
                row[f"{vp}_rank"] = len(df) + 1
                row[f"{vp}_importance"] = 0.0
        result_rows.append(row)

    result_df = pd.DataFrame(result_rows)

    # Add rank shift between first and second VP config
    if len(vp_configs) >= 2:
        vp_a, vp_b = vp_configs[0], vp_configs[1]
        if f"{vp_a}_rank" in result_df.columns and f"{vp_b}_rank" in result_df.columns:
            result_df["rank_shift"] = (
                result_df[f"{vp_a}_rank"] - result_df[f"{vp_b}_rank"]
            )

    # Spearman correlation between VP rank vectors
    vp_list = [vp for vp in vp_configs if f"{vp}_rank" in result_df.columns]
    for i, vp_a in enumerate(vp_list):
        for vp_b in vp_list[i + 1:]:
            rho, p = spearmanr(
                result_df[f"{vp_a}_rank"], result_df[f"{vp_b}_rank"]
            )
            logger.info(
                "%s importance rank correlation %s↔%s: ρ=%.3f (p=%.4f)",
                corpus_name, vp_a, vp_b, rho, p,
            )

    return result_df


# ──────────────────────────────────────────────────────
# WP5a: Feature Value Statistics
# ──────────────────────────────────────────────────────

def compute_feature_statistics(
    feature_df: pd.DataFrame,
    labels: list[str],
) -> pd.DataFrame:
    """Compute per-feature statistics across classes.

    For each feature, computes:
    - Per-class mean/std/median
    - Kruskal-Wallis test (non-parametric ANOVA)
    - Eta-squared effect size
    """
    # Deduplicate columns (keep first occurrence)
    feature_df = feature_df.loc[:, ~feature_df.columns.duplicated()]

    y = np.array(labels)
    classes = sorted(set(labels))
    n_total = len(y)

    rows = []
    for col in feature_df.columns:
        values = feature_df[col].values
        valid_mask = ~np.isnan(values)
        if valid_mask.sum() < 10:
            continue

        # Per-class groups
        groups = []
        class_stats = {}
        for cls in classes:
            cls_vals = values[(y == cls) & valid_mask]
            if len(cls_vals) > 0:
                groups.append(cls_vals)
                class_stats[cls] = {
                    "mean": float(np.mean(cls_vals)),
                    "std": float(np.std(cls_vals, ddof=1)) if len(cls_vals) > 1 else 0.0,
                    "median": float(np.median(cls_vals)),
                    "n": len(cls_vals),
                }

        if len(groups) < 2:
            continue

        # Kruskal-Wallis test
        try:
            h_stat, kw_p = kruskal(*groups)
        except ValueError:
            h_stat, kw_p = np.nan, np.nan

        # Eta-squared (effect size for Kruskal-Wallis)
        # η² = (H - k + 1) / (n - k), where k = number of groups
        k = len(groups)
        if not np.isnan(h_stat) and n_total > k:
            eta_sq = (h_stat - k + 1) / (n_total - k)
            eta_sq = max(0.0, eta_sq)  # clip negative
        else:
            eta_sq = np.nan

        row = {
            "feature": col,
            "kruskal_wallis_h": round(h_stat, 3) if not np.isnan(h_stat) else np.nan,
            "kruskal_wallis_p": kw_p,
            "eta_squared": round(eta_sq, 4) if not np.isnan(eta_sq) else np.nan,
            "significant_005": kw_p < 0.05 if not np.isnan(kw_p) else False,
            "overall_mean": round(float(np.nanmean(values)), 4),
            "overall_std": round(float(np.nanstd(values, ddof=1)), 4),
        }

        # Add per-class means for top classes
        for cls in classes[:10]:  # limit to first 10 classes
            if cls in class_stats:
                row[f"mean_{cls}"] = round(class_stats[cls]["mean"], 4)

        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        # Bonferroni correction
        n_tests = len(df)
        df["bonferroni_p"] = (df["kruskal_wallis_p"] * n_tests).clip(upper=1.0)
        df["significant_bonferroni"] = df["bonferroni_p"] < 0.05
        df = df.sort_values("kruskal_wallis_p").reset_index(drop=True)

    return df


# ──────────────────────────────────────────────────────
# WP5b: Per-Class F1 Improvement
# ──────────────────────────────────────────────────────

def compute_per_class_improvement(
    corpus_name: str,
    vp_config: str,
) -> pd.DataFrame | None:
    """Compare per-class F1 between global and combined for a specific VP.

    Loads classification results and extracts per-class F1 from the
    best non-baseline classifiers for global and combined configs.
    """
    summary_path = (
        PROJECT_ROOT / "results" / corpus_name / vp_config
        / "classification" / "summary.csv"
    )
    if not summary_path.exists():
        logger.warning("No summary for %s/%s", corpus_name, vp_config)
        return None

    summary = pd.read_csv(summary_path)
    real = summary[~summary["classifier"].isin(["majority", "stratified"])]

    rows = []
    for config in ["global", "combined", "pattern_all"]:
        subset = real[real["feature_config"] == config]
        if subset.empty:
            continue
        best = subset.loc[subset["f1_macro_mean"].idxmax()]
        rows.append({
            "feature_config": config,
            "classifier": best["classifier"],
            "f1_macro_mean": best["f1_macro_mean"],
            "f1_macro_std": best["f1_macro_std"],
        })

    if len(rows) < 2:
        return None

    return pd.DataFrame(rows)
