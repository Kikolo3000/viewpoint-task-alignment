"""Classification pipeline with proper CV, statistical tests, and reporting."""

import logging
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────
# Classifier Definitions
# ──────────────────────────────────────────────────────

def get_classifiers() -> dict[str, object]:
    """Return the 7 classifiers (2 baselines + 5 models)."""
    return {
        "majority": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=42),
        "logistic": LogisticRegression(
            solver="lbfgs",
            max_iter=2000,
            random_state=42,
        ),
        "rf": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1,
        ),
        "svc": SVC(kernel="rbf", random_state=42),
        "gbm": GradientBoostingClassifier(
            n_estimators=200,
            random_state=42,
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=500,
            random_state=42,
            early_stopping=True,
        ),
    }


# ──────────────────────────────────────────────────────
# Pipeline Builder
# ──────────────────────────────────────────────────────

def build_pipeline(
    classifier: object,
    needs_svd: bool = False,
    n_svd_components: int = 50,
) -> Pipeline:
    """
    Build an sklearn Pipeline with preprocessing INSIDE (critical for CV).

    Steps: impute → scale → (optional SVD) → classify
    """
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if needs_svd:
        from sklearn.decomposition import TruncatedSVD

        steps.append(("svd", TruncatedSVD(n_components=n_svd_components, random_state=42)))
    steps.append(("clf", classifier))
    return Pipeline(steps)


# ──────────────────────────────────────────────────────
# Cross-Validation Experiment
# ──────────────────────────────────────────────────────

def run_experiment(
    X: np.ndarray,
    y: np.ndarray,
    classifier_name: str,
    classifier: object,
    feature_config: str,
    n_splits: int = 10,
    n_repeats: int = 3,
    needs_svd: bool = False,
    n_svd_components: int = 50,
) -> dict:
    """
    Run a single classification experiment with RepeatedStratifiedKFold.

    Returns dict with fold-wise metrics and aggregated confusion matrix.
    """
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    n_classes = len(le.classes_)

    pipeline = build_pipeline(classifier, needs_svd=needs_svd, n_svd_components=n_svd_components)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=42)

    fold_accuracies = []
    fold_f1_macro = []
    fold_f1_weighted = []
    agg_conf_matrix = np.zeros((n_classes, n_classes), dtype=int)
    per_class_f1_folds = []

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y_enc)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_enc[train_idx], y_enc[test_idx]

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        fold_accuracies.append(accuracy_score(y_test, y_pred))
        fold_f1_macro.append(f1_score(y_test, y_pred, average="macro", zero_division=0))
        fold_f1_weighted.append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        agg_conf_matrix += confusion_matrix(y_test, y_pred, labels=range(n_classes))

        per_class = f1_score(y_test, y_pred, average=None, labels=range(n_classes), zero_division=0)
        per_class_f1_folds.append(per_class)

    per_class_f1_mean = np.mean(per_class_f1_folds, axis=0)

    result = {
        "classifier": classifier_name,
        "feature_config": feature_config,
        "accuracy_mean": np.mean(fold_accuracies),
        "accuracy_std": np.std(fold_accuracies),
        "f1_macro_mean": np.mean(fold_f1_macro),
        "f1_macro_std": np.std(fold_f1_macro),
        "f1_weighted_mean": np.mean(fold_f1_weighted),
        "f1_weighted_std": np.std(fold_f1_weighted),
        "fold_accuracies": fold_accuracies,
        "fold_f1_macro": fold_f1_macro,
        "fold_f1_weighted": fold_f1_weighted,
        "confusion_matrix": agg_conf_matrix,
        "classes": le.classes_.tolist(),
        "per_class_f1": dict(zip(le.classes_.tolist(), per_class_f1_mean.tolist())),
    }

    logger.info(
        "%s / %s: acc=%.3f±%.3f, F1_macro=%.3f±%.3f",
        feature_config,
        classifier_name,
        result["accuracy_mean"],
        result["accuracy_std"],
        result["f1_macro_mean"],
        result["f1_macro_std"],
    )
    return result


# ──────────────────────────────────────────────────────
# Full Experiment Suite
# ──────────────────────────────────────────────────────

def run_all_experiments(
    feature_groups: dict[str, pd.DataFrame],
    labels: list[str],
    corpus_name: str,
    n_splits: int = 10,
    n_repeats: int = 3,
    skip_classifiers: list[str] | None = None,
) -> list[dict]:
    """
    Run all 6 feature configs × 7 classifiers experiments.

    Uses pre-computed (and cached) feature DataFrames for all groups.
    BoP features use pre-computed SVD components from the parquet cache.

    skip_classifiers: list of classifier names to skip (e.g. ["svc"] for large datasets).
    """
    from folk_patterns_repro.data_loading import FEATURE_CONFIGS, assemble_features

    y = np.array(labels)
    classifiers = get_classifiers()
    skip = set(skip_classifiers or [])
    results = []

    total_experiments = sum(
        1 for _ in FEATURE_CONFIGS for c in classifiers if c not in skip
    )
    done = 0

    for config_name, group_letters in FEATURE_CONFIGS.items():
        logger.info("=== Config: %s (groups: %s) ===", config_name, group_letters)

        try:
            assembled = assemble_features(feature_groups, config_name)
        except ValueError as e:
            logger.warning("Skipping config '%s': %s", config_name, e)
            continue

        X = assembled.values
        logger.info("Feature matrix: %d × %d", X.shape[0], X.shape[1])

        for clf_name, clf in classifiers.items():
            if clf_name in skip:
                logger.info("Skipping %s (in skip list)", clf_name)
                continue
            done += 1
            logger.info("[%d/%d] Running %s / %s", done, total_experiments, config_name, clf_name)
            result = run_experiment(
                X=X,
                y=y,
                classifier_name=clf_name,
                classifier=clf,
                feature_config=config_name,
                n_splits=n_splits,
                n_repeats=n_repeats,
            )
            results.append(result)

    return results


# ──────────────────────────────────────────────────────
# Statistical Tests
# ──────────────────────────────────────────────────────

def corrected_resampled_ttest(
    scores_a: list[float],
    scores_b: list[float],
    n_train: int,
    n_test: int,
) -> tuple[float, float]:
    """
    Corrected resampled t-test (Nadeau & Bengio 2003).

    Adjusts for the variance inflation caused by overlapping training sets
    in repeated cross-validation.
    """
    diffs = np.array(scores_a) - np.array(scores_b)
    n = len(diffs)
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)

    # Correction factor
    correction = (1 / n) + (n_test / n_train)
    corrected_var = correction * var_diff

    if corrected_var <= 0:
        return 0.0, 1.0

    from scipy.stats import t as t_dist

    t_stat = mean_diff / np.sqrt(corrected_var)
    df = n - 1
    p_value = 2 * t_dist.sf(abs(t_stat), df)
    return t_stat, p_value


def cohens_d(scores_a: list[float], scores_b: list[float]) -> float:
    """Compute Cohen's d effect size."""
    diffs = np.array(scores_a) - np.array(scores_b)
    if np.std(diffs, ddof=1) == 0:
        return 0.0
    return float(np.mean(diffs) / np.std(diffs, ddof=1))


def compare_statistically(
    results: list[dict],
    pairs: list[tuple[str, str]] | None = None,
    metric_key: str = "fold_f1_macro",
    n_train: int | None = None,
    n_test: int | None = None,
) -> pd.DataFrame:
    """
    Run pairwise statistical tests between feature configurations.

    Uses the best non-baseline classifier (RF by default) for each config.
    """
    # Group results by config, pick best non-baseline classifier per config
    config_best: dict[str, dict] = {}
    for r in results:
        if r["classifier"] in ("majority", "stratified"):
            continue
        cfg = r["feature_config"]
        if cfg not in config_best or r["f1_macro_mean"] > config_best[cfg]["f1_macro_mean"]:
            config_best[cfg] = r

    if pairs is None:
        pairs = [
            ("pattern_all", "global"),
            ("combined", "global"),
            ("bop", "structural"),
            ("musical", "structural"),
            ("combined", "pattern_all"),
        ]

    rows = []
    for cfg_a, cfg_b in pairs:
        if cfg_a not in config_best or cfg_b not in config_best:
            continue

        scores_a = config_best[cfg_a][metric_key]
        scores_b = config_best[cfg_b][metric_key]

        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = wilcoxon(scores_a, scores_b, alternative="two-sided")
        except ValueError:
            w_stat, w_p = np.nan, np.nan

        # Corrected t-test
        if n_train is not None and n_test is not None:
            t_stat, t_p = corrected_resampled_ttest(scores_a, scores_b, n_train, n_test)
        else:
            t_stat, t_p = np.nan, np.nan

        d = cohens_d(scores_a, scores_b)

        rows.append({
            "config_a": cfg_a,
            "config_b": cfg_b,
            "clf_a": config_best[cfg_a]["classifier"],
            "clf_b": config_best[cfg_b]["classifier"],
            "mean_a": np.mean(scores_a),
            "mean_b": np.mean(scores_b),
            "diff": np.mean(scores_a) - np.mean(scores_b),
            "wilcoxon_stat": w_stat,
            "wilcoxon_p": w_p,
            "corrected_t_stat": t_stat,
            "corrected_t_p": t_p,
            "cohens_d": d,
            "significant_wilcoxon": w_p < 0.05 if not np.isnan(w_p) else False,
            "significant_corrected_t": t_p < 0.05 if not np.isnan(t_p) else False,
        })

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────
# Result Formatting
# ──────────────────────────────────────────────────────

def results_to_summary_df(results: list[dict]) -> pd.DataFrame:
    """Convert experiment results to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "feature_config": r["feature_config"],
            "classifier": r["classifier"],
            "accuracy_mean": r["accuracy_mean"],
            "accuracy_std": r["accuracy_std"],
            "f1_macro_mean": r["f1_macro_mean"],
            "f1_macro_std": r["f1_macro_std"],
            "f1_weighted_mean": r["f1_weighted_mean"],
            "f1_weighted_std": r["f1_weighted_std"],
        })
    return pd.DataFrame(rows)


def results_to_fold_details(results: list[dict]) -> pd.DataFrame:
    """Convert experiment results to per-fold detail DataFrame."""
    rows = []
    for r in results:
        for fold_idx in range(len(r["fold_accuracies"])):
            rows.append({
                "feature_config": r["feature_config"],
                "classifier": r["classifier"],
                "fold": fold_idx,
                "accuracy": r["fold_accuracies"][fold_idx],
                "f1_macro": r["fold_f1_macro"][fold_idx],
                "f1_weighted": r["fold_f1_weighted"][fold_idx],
            })
    return pd.DataFrame(rows)
