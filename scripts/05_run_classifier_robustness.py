#!/usr/bin/env python3
"""Classifier-agnostic robustness: global vs combined across classifier families.

Runs every non-baseline classifier on `global` and `combined` features for
each corpus's best VP config. SVC is skipped above 5k pieces (O(n²) cost);
GBM is skipped above 10k pieces.

Outputs:
    results/{corpus}/classifier_robustness/summary.csv
    results/{corpus}/classifier_robustness/fold_details.csv
    results/viewpoint_comparison/classifier_robustness.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from folk_patterns_repro.classification import (
    get_classifiers,
    results_to_fold_details,
    results_to_summary_df,
    run_experiment,
)
from folk_patterns_repro.data_loading import assemble_features, load_corpus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

CORPUS_BEST_VP = {
    "symbtr": "M",
    "the_session": "B",
    "essen": "MR",
}
CORPORA = list(CORPUS_BEST_VP.keys())

SVC_PIECE_LIMIT = 5000
GBM_PIECE_LIMIT = 10_000


def run_robustness_for_corpus(corpus: str, vp_config: str) -> list[dict]:
    logger.info("=== Classifier robustness: %s (vp=%s) ===", corpus, vp_config)
    out_dir = RESULTS_DIR / corpus / "classifier_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        feature_groups, labels = load_corpus(
            corpus, vp_config, artifacts_dir=ARTIFACTS_DIR, min_class_count=10
        )
    except FileNotFoundError as e:
        logger.warning("SKIP: %s", e)
        return []

    n_pieces = len(labels)
    n_classes = len(set(labels))
    n_splits, n_repeats = 10, 3
    logger.info(
        "  %d pieces, %d classes, %d-fold × %d repeats",
        n_pieces, n_classes, n_splits, n_repeats,
    )

    classifiers = {
        name: clf for name, clf in get_classifiers().items()
        if name not in ("majority", "stratified")
    }
    if n_pieces > SVC_PIECE_LIMIT and "svc" in classifiers:
        logger.info("  Skipping SVC: %d > %d", n_pieces, SVC_PIECE_LIMIT)
        del classifiers["svc"]
    if n_pieces > GBM_PIECE_LIMIT and "gbm" in classifiers:
        logger.info("  Skipping GBM: %d > %d", n_pieces, GBM_PIECE_LIMIT)
        del classifiers["gbm"]

    results: list[dict] = []
    for config_name in ("global", "combined"):
        try:
            assembled = assemble_features(feature_groups, config_name)
        except ValueError as e:
            logger.warning("Cannot assemble '%s': %s", config_name, e)
            continue
        X = assembled.values
        y = np.array(labels)
        logger.info("  Config '%s': %d × %d", config_name, X.shape[0], X.shape[1])

        for clf_name, clf in classifiers.items():
            logger.info("  Running %s / %s", config_name, clf_name)
            results.append(run_experiment(
                X=X, y=y,
                classifier_name=clf_name, classifier=clf,
                feature_config=config_name,
                n_splits=n_splits, n_repeats=n_repeats,
            ))

    if results:
        results_to_summary_df(results).to_csv(out_dir / "summary.csv", index=False)
        results_to_fold_details(results).to_csv(out_dir / "fold_details.csv", index=False)
        logger.info("Saved to %s", out_dir)
    return results


def compile_robustness() -> pd.DataFrame | None:
    rows = []
    for corpus in CORPORA:
        vp = CORPUS_BEST_VP[corpus]
        summary_path = RESULTS_DIR / corpus / "classifier_robustness" / "summary.csv"
        if not summary_path.exists():
            logger.warning("No robustness results for %s", corpus)
            continue
        df = pd.read_csv(summary_path)
        for clf in df["classifier"].unique():
            clf_df = df[df["classifier"] == clf]
            global_row = clf_df[clf_df["feature_config"] == "global"]
            combined_row = clf_df[clf_df["feature_config"] == "combined"]
            if global_row.empty or combined_row.empty:
                continue
            g = float(global_row.iloc[0]["f1_macro_mean"])
            c = float(combined_row.iloc[0]["f1_macro_mean"])
            rows.append({
                "corpus": corpus,
                "vp_config": vp,
                "classifier": clf,
                "global_f1": round(g, 4),
                "combined_f1": round(c, 4),
                "delta": round(c - g, 4),
                "combined_wins": c > g,
            })

    if not rows:
        return None

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("CLASSIFIER-AGNOSTIC ROBUSTNESS: global vs combined")
    print("=" * 70)
    for corpus in CORPORA:
        subset = df[df["corpus"] == corpus]
        if subset.empty:
            continue
        n_wins = int(subset["combined_wins"].sum())
        n_total = len(subset)
        print(f"\n{corpus} (VP={CORPUS_BEST_VP[corpus]}):")
        print(f"  Combined > Global: {n_wins}/{n_total} classifiers")
        for _, row in subset.iterrows():
            marker = "+" if row["combined_wins"] else "-"
            print(
                f"  [{marker}] {row['classifier']:10s}: "
                f"global={row['global_f1']:.4f} -> combined={row['combined_f1']:.4f} "
                f"(delta={row['delta']:+.4f})"
            )

    out_path = RESULTS_DIR / "viewpoint_comparison" / "classifier_robustness.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %s", out_path)
    return df


def main() -> None:
    print("=" * 70)
    print("Classifier-Agnostic Robustness")
    print("=" * 70)
    for corpus in CORPORA:
        run_robustness_for_corpus(corpus, CORPUS_BEST_VP[corpus])
    compile_robustness()


if __name__ == "__main__":
    main()
