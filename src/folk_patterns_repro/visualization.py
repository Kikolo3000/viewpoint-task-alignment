"""Visualization functions for paper figures."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Paper-quality defaults
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})


def save_fig(fig: plt.Figure, path: Path, close: bool = True):
    """Save figure as PNG and PDF."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"))
    fig.savefig(path.with_suffix(".pdf"))
    logger.info("Saved figure: %s", path.with_suffix(".png"))
    if close:
        plt.close(fig)


# ──────────────────────────────────────────────────────
# RQ1: Classification Results
# ──────────────────────────────────────────────────────

def plot_classification_heatmap(
    summary_df: pd.DataFrame,
    metric: str = "f1_macro_mean",
    out_path: Path | None = None,
    title: str = "Classification Results",
) -> plt.Figure:
    """
    Heatmap of classifier × feature_config performance.
    """
    pivot = summary_df.pivot_table(
        index="classifier", columns="feature_config", values=metric
    )

    # Reorder
    config_order = ["global", "structural", "musical", "bop", "pattern_all", "combined"]
    clf_order = ["majority", "stratified", "logistic", "rf", "svc", "gbm", "mlp"]
    pivot = pivot.reindex(index=[c for c in clf_order if c in pivot.index])
    pivot = pivot.reindex(columns=[c for c in config_order if c in pivot.columns])

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title(title)
    ax.set_xlabel("Feature Configuration")
    ax.set_ylabel("Classifier")

    if out_path:
        save_fig(fig, out_path)
    return fig


def plot_statistical_comparison(
    stat_df: pd.DataFrame,
    out_path: Path | None = None,
) -> plt.Figure:
    """Bar chart of pairwise statistical comparisons."""
    if stat_df.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No comparisons available", ha="center")
        return fig

    fig, ax = plt.subplots(figsize=(8, 4))
    labels_list = [f"{r['config_a']} vs {r['config_b']}" for _, r in stat_df.iterrows()]
    x = range(len(labels_list))
    diffs = stat_df["diff"].values
    colors = ["green" if d > 0 else "red" for d in diffs]
    significant = stat_df["significant_wilcoxon"].values

    bars = ax.barh(x, diffs, color=colors, alpha=0.7)
    for i, (bar, sig) in enumerate(zip(bars, significant)):
        if sig:
            ax.text(
                bar.get_width() + 0.002,
                bar.get_y() + bar.get_height() / 2,
                "*",
                va="center",
                fontsize=14,
                fontweight="bold",
            )

    ax.set_yticks(x)
    ax.set_yticklabels(labels_list)
    ax.set_xlabel("Difference in F1-macro")
    ax.set_title("Pairwise Statistical Comparisons (* = p < 0.05)")
    ax.axvline(0, color="black", linewidth=0.5)

    if out_path:
        save_fig(fig, out_path)
    return fig


def plot_confusion_matrix(
    conf_matrix: np.ndarray,
    classes: list[str],
    out_path: Path | None = None,
    title: str = "Confusion Matrix",
) -> plt.Figure:
    """Plot a confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(max(6, len(classes) * 0.8), max(5, len(classes) * 0.7)))
    sns.heatmap(
        conf_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    if out_path:
        save_fig(fig, out_path)
    return fig


# ──────────────────────────────────────────────────────
# RQ2: Interpretability
# ──────────────────────────────────────────────────────

def plot_feature_importance(
    importance_df: pd.DataFrame,
    top_k: int = 15,
    out_path: Path | None = None,
    title: str = "Feature Importance",
) -> plt.Figure:
    """Horizontal bar chart of top-k features by permutation importance."""
    df = importance_df.head(top_k).copy()
    df = df.iloc[::-1]  # Reverse for bottom-to-top

    fig, ax = plt.subplots(figsize=(7, max(4, top_k * 0.35)))
    ax.barh(
        df["feature"],
        df["importance_mean"],
        xerr=df["importance_std"],
        color="steelblue",
        alpha=0.8,
    )
    ax.set_xlabel("Permutation Importance (F1-macro decrease)")
    ax.set_title(title)

    if out_path:
        save_fig(fig, out_path)
    return fig


def plot_shap_beeswarm(
    shap_values,
    X_shap: np.ndarray,
    feature_names: list[str],
    class_names: list[str],
    out_path: Path | None = None,
    max_display: int = 15,
) -> plt.Figure:
    """SHAP beeswarm plot for multi-class classification."""
    import shap

    # For multi-class, shap_values is a list of arrays (one per class)
    # Use mean absolute SHAP for summary
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
    else:
        mean_abs_shap = np.abs(shap_values)

    fig, ax = plt.subplots(figsize=(8, max(4, max_display * 0.35)))
    shap.summary_plot(
        shap_values if isinstance(shap_values, np.ndarray) else np.mean(shap_values, axis=0),
        X_shap,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
    )

    if out_path:
        save_fig(plt.gcf(), out_path)
    return plt.gcf()


def plot_cross_corpus_importance(
    importance_dfs: dict[str, pd.DataFrame],
    top_k: int = 15,
    out_path: Path | None = None,
) -> plt.Figure:
    """3-column heatmap of feature importance across corpora."""
    # Get the union of top_k features from each corpus
    top_features = set()
    for df in importance_dfs.values():
        top_features.update(df.head(top_k)["feature"].tolist())
    top_features = sorted(top_features)

    # Build importance matrix
    data = {}
    for corpus, df in importance_dfs.items():
        feat_to_imp = dict(zip(df["feature"], df["importance_mean"]))
        data[corpus] = [feat_to_imp.get(f, 0.0) for f in top_features]

    heatmap_df = pd.DataFrame(data, index=top_features)

    # Sort by mean importance across corpora
    heatmap_df["mean"] = heatmap_df.mean(axis=1)
    heatmap_df = heatmap_df.sort_values("mean", ascending=False).drop(columns=["mean"])

    fig, ax = plt.subplots(figsize=(6, max(5, len(top_features) * 0.35)))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Feature Importance Across Corpora")

    if out_path:
        save_fig(fig, out_path)
    return fig


# ──────────────────────────────────────────────────────
# RQ3: Unsupervised
# ──────────────────────────────────────────────────────

def plot_umap(
    embedding_df: pd.DataFrame,
    out_path: Path | None = None,
    title: str = "UMAP Embedding",
) -> plt.Figure:
    """Scatter plot of UMAP 2D embedding colored by label."""
    fig, ax = plt.subplots(figsize=(8, 6))
    classes = sorted(embedding_df["label"].unique())
    palette = sns.color_palette("husl", len(classes))

    for cls, color in zip(classes, palette):
        mask = embedding_df["label"] == cls
        ax.scatter(
            embedding_df.loc[mask, "umap_1"],
            embedding_df.loc[mask, "umap_2"],
            label=cls,
            alpha=0.5,
            s=10,
            color=color,
        )

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", markerscale=3)

    if out_path:
        save_fig(fig, out_path)
    return fig


def plot_umap_grid(
    embeddings: dict[str, pd.DataFrame],
    out_path: Path | None = None,
    suptitle: str = "UMAP Embeddings by Feature Group",
) -> plt.Figure:
    """Grid of UMAP embeddings for different feature configurations."""
    n = len(embeddings)
    ncols = min(n, 2)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    if n == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for idx, (config_name, emb_df) in enumerate(embeddings.items()):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]
        classes = sorted(emb_df["label"].unique())
        palette = sns.color_palette("husl", len(classes))

        for cls, color in zip(classes, palette):
            mask = emb_df["label"] == cls
            ax.scatter(
                emb_df.loc[mask, "umap_1"],
                emb_df.loc[mask, "umap_2"],
                label=cls,
                alpha=0.4,
                s=8,
                color=color,
            )
        ax.set_title(config_name)
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")

    # Add legend to last axis
    handles, labels_list = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="lower center", ncol=min(6, len(labels_list)), markerscale=3)

    # Hide unused axes
    for idx in range(n, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    if out_path:
        save_fig(fig, out_path)
    return fig


def plot_pattern_sharing_heatmap(
    jaccard_df: pd.DataFrame,
    out_path: Path | None = None,
    title: str = "Pattern Sharing (Jaccard Similarity)",
) -> plt.Figure:
    """Heatmap with dendrogram of class × class pattern sharing."""
    g = sns.clustermap(
        jaccard_df,
        cmap="YlOrRd",
        annot=True,
        fmt=".2f",
        figsize=(max(6, len(jaccard_df) * 0.9), max(5, len(jaccard_df) * 0.8)),
        linewidths=0.5,
        method="ward",
    )
    g.fig.suptitle(title, y=1.02)

    if out_path:
        save_fig(g.fig, out_path)
    return g.fig


def plot_clustering_evaluation(
    clustering_df: pd.DataFrame,
    out_path: Path | None = None,
    title: str = "Clustering Evaluation",
) -> plt.Figure:
    """Line plots of ARI, NMI, and silhouette across k values."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    for ax, metric in zip(axes, ["ari", "nmi", "silhouette"]):
        ax.plot(clustering_df["k"], clustering_df[metric], "o-", markersize=3)
        ax.set_xlabel("Number of clusters (k)")
        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()

    if out_path:
        save_fig(fig, out_path)
    return fig
