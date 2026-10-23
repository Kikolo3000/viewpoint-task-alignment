"""Feature importance, SHAP analysis, and cross-corpus comparison."""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Permutation Importance
# ──────────────────────────────────────────────────────

def compute_permutation_importance(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_splits: int = 10,
    n_repeats: int = 3,
    n_perm_repeats: int = 5,
) -> pd.DataFrame:
    """
    Compute permutation importance averaged across CV folds.

    Uses RandomForest as the model. Importance is measured on the test set
    of each fold, then averaged across all folds.

    All-NaN columns are dropped before fitting to avoid imputer issues.
    """
    # Drop all-NaN columns upfront
    nan_mask = np.all(np.isnan(X), axis=0)
    valid_cols = ~nan_mask
    X_clean = X[:, valid_cols]
    clean_names = [f for f, v in zip(feature_names, valid_cols) if v]
    if nan_mask.any():
        dropped = [f for f, v in zip(feature_names, valid_cols) if not v]
        logger.warning("Dropped %d all-NaN features: %s", len(dropped), dropped)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)
    all_importances = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_clean, y_enc)):
        X_train, X_test = X_clean[train_idx], X_clean[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
        ])
        pipe.fit(X_train, y_train)

        # Permutation importance on the preprocessed test data
        imputer = pipe.named_steps["imputer"]
        scaler = pipe.named_steps["scaler"]
        clf = pipe.named_steps["clf"]
        X_test_processed = scaler.transform(imputer.transform(X_test))

        perm_result = permutation_importance(
            clf,
            X_test_processed,
            y_test,
            n_repeats=n_perm_repeats,
            scoring="f1_macro",
            random_state=42,
            n_jobs=-1,
        )
        all_importances.append(perm_result.importances_mean)

    # Average across folds
    avg_importance = np.mean(all_importances, axis=0)
    std_importance = np.std(all_importances, axis=0)

    df = pd.DataFrame({
        "feature": clean_names,
        "importance_mean": avg_importance,
        "importance_std": std_importance,
    }).sort_values("importance_mean", ascending=False).reset_index(drop=True)

    df["rank"] = range(1, len(df) + 1)
    return df


# ──────────────────────────────────────────────────────
# SHAP Analysis
# ──────────────────────────────────────────────────────

def compute_shap_values(
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    max_samples: int = 500,
) -> tuple:
    """
    Train a RF on full data and compute SHAP values.

    Returns (shap_values, explainer, model, X_processed, le, clean_names).
    """
    import shap

    # Drop all-NaN columns
    nan_mask = np.all(np.isnan(X), axis=0)
    valid_cols = ~nan_mask
    X_clean = X[:, valid_cols]
    clean_names = [f for f, v in zip(feature_names, valid_cols) if v]

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
    ])
    pipe.fit(X_clean, y_enc)

    imputer = pipe.named_steps["imputer"]
    scaler = pipe.named_steps["scaler"]
    clf = pipe.named_steps["clf"]
    X_processed = scaler.transform(imputer.transform(X_clean))

    # Subsample for SHAP if dataset is large
    if X_processed.shape[0] > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(X_processed.shape[0], max_samples, replace=False)
        X_shap = X_processed[idx]
    else:
        X_shap = X_processed

    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_shap)

    return shap_values, explainer, clf, X_shap, le, clean_names


# ──────────────────────────────────────────────────────
# Cross-Corpus Comparison
# ──────────────────────────────────────────────────────

def compare_importance_across_corpora(
    importance_dfs: dict[str, pd.DataFrame],
    top_k: int = 10,
) -> dict[str, pd.DataFrame]:
    """
    Compare feature importance rankings across corpora.

    Returns dict with:
    - 'rank_comparison': DataFrame with rank of each feature per corpus
    - 'rank_correlations': Spearman correlations between corpus pairs
    - 'universal': features in top_k across ALL corpora
    - 'tradition_specific': features in top_k in exactly 1 corpus
    """
    # Build rank comparison matrix
    all_features = set()
    for df in importance_dfs.values():
        all_features.update(df["feature"].tolist())
    all_features = sorted(all_features)

    rank_data = {}
    for corpus, df in importance_dfs.items():
        feat_to_rank = dict(zip(df["feature"], df["rank"]))
        rank_data[corpus] = [feat_to_rank.get(f, len(df) + 1) for f in all_features]

    rank_df = pd.DataFrame(rank_data, index=all_features)
    rank_df.index.name = "feature"

    # Spearman rank correlations between corpus pairs
    corpus_names = list(importance_dfs.keys())
    corr_rows = []
    for c1, c2 in combinations(corpus_names, 2):
        # Only compare features that exist in both corpora
        common_features = set(importance_dfs[c1]["feature"]) & set(importance_dfs[c2]["feature"])
        if len(common_features) < 3:
            continue
        ranks_1 = []
        ranks_2 = []
        for f in common_features:
            r1 = importance_dfs[c1].loc[importance_dfs[c1]["feature"] == f, "rank"].iloc[0]
            r2 = importance_dfs[c2].loc[importance_dfs[c2]["feature"] == f, "rank"].iloc[0]
            ranks_1.append(r1)
            ranks_2.append(r2)
        rho, p = spearmanr(ranks_1, ranks_2)
        corr_rows.append({
            "corpus_a": c1,
            "corpus_b": c2,
            "spearman_rho": rho,
            "p_value": p,
            "n_common_features": len(common_features),
        })
    corr_df = pd.DataFrame(corr_rows)

    # Universal features (top_k in all corpora)
    top_per_corpus = {}
    for corpus, df in importance_dfs.items():
        top_per_corpus[corpus] = set(df.nsmallest(top_k, "rank")["feature"])

    universal = set.intersection(*top_per_corpus.values()) if top_per_corpus else set()

    # Tradition-specific features (top_k in exactly 1 corpus)
    tradition_specific = {}
    for corpus in corpus_names:
        others = [c for c in corpus_names if c != corpus]
        other_tops = set.union(*(top_per_corpus[c] for c in others)) if others else set()
        specific = top_per_corpus[corpus] - other_tops
        if specific:
            tradition_specific[corpus] = specific

    return {
        "rank_comparison": rank_df,
        "rank_correlations": corr_df,
        "universal": sorted(universal),
        "tradition_specific": {k: sorted(v) for k, v in tradition_specific.items()},
    }
