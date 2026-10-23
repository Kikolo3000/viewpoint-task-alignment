#!/usr/bin/env python3
"""Reproduce classification results from Zenodo artifacts.

For every corpus × VP config, load the pre-extracted feature parquets and
label CSV, run the 7 classifiers × 6 feature configs CV experiment, and
write `summary.csv`, `fold_details.csv`, and `statistical_tests.csv`.

Output layout mirrors the original experiments repo:
    results/{corpus}/{vp_config}/classification/
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from folk_patterns_repro.classification import (
    compare_statistically,
    results_to_fold_details,
    results_to_summary_df,
    run_all_experiments,
)
from folk_patterns_repro.data_loading import (
    CORPORA,
    VP_CONFIGS,
    load_corpus,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger("01_run_classification")


def run_corpus_vp(
    corpus: str,
    vp_config: str,
    artifacts_dir: Path,
    results_dir: Path,
    min_class_count: int = 10,
) -> None:
    out_dir = results_dir / corpus / vp_config / "classification"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        feature_groups, labels = load_corpus(
            corpus, vp_config, artifacts_dir=artifacts_dir, min_class_count=min_class_count
        )
    except FileNotFoundError as e:
        logger.warning("SKIP %s/%s: %s", corpus, vp_config, e)
        return

    n_pieces = len(labels)
    n_classes = len(set(labels))
    n_splits = 10
    n_repeats = 3

    logger.info(
        "=== %s / %s: %d pieces, %d classes, %d-fold × %d repeats ===",
        corpus, vp_config, n_pieces, n_classes, n_splits, n_repeats,
    )

    skip_classifiers: list[str] = []
    if n_pieces > 5000 or n_classes > 20:
        skip_classifiers = ["svc", "gbm", "mlp"]
        logger.info(
            "Skipping SVC+GBM+MLP for %s (%d pieces, %d classes)",
            corpus, n_pieces, n_classes,
        )

    results = run_all_experiments(
        feature_groups=feature_groups,
        labels=labels,
        corpus_name=corpus,
        n_splits=n_splits,
        n_repeats=n_repeats,
        skip_classifiers=skip_classifiers,
    )

    summary = results_to_summary_df(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    fold_details = results_to_fold_details(results)
    fold_details.to_csv(out_dir / "fold_details.csv", index=False)

    n_test = n_pieces // n_splits
    n_train = n_pieces - n_test
    stat_df = compare_statistically(results, n_train=n_train, n_test=n_test)
    stat_df.to_csv(out_dir / "statistical_tests.csv", index=False)

    logger.info(
        "Wrote %d summary rows, %d fold-detail rows → %s",
        len(summary), len(fold_details), out_dir,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=REPO_ROOT / "artifacts",
        help="Directory with features/ and labels/ (populated by download_artifacts.py)",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "results",
        help="Output directory for regenerated classification results",
    )
    parser.add_argument(
        "--corpora",
        nargs="*",
        default=CORPORA,
        help="Subset of corpora to run (default: all three)",
    )
    parser.add_argument(
        "--vp-configs",
        nargs="*",
        default=VP_CONFIGS,
        help="Subset of VP configs to run (default: all seven)",
    )
    args = parser.parse_args()

    np.random.seed(42)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    for corpus in args.corpora:
        for vp in args.vp_configs:
            run_corpus_vp(corpus, vp, args.artifacts_dir, args.results_dir)

    logger.info("All classification runs complete.")


if __name__ == "__main__":
    main()
