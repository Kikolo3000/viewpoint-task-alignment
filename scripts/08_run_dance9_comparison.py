#!/usr/bin/env python3
"""Reproduce the Dance-9 benchmark comparison (Table 4 of the paper).

Loads the pre-extracted Dance-9 features (VP=B, combined A+B+C+D) and
runs five classifiers via 10×3 RepeatedStratifiedKFold, then positions
our best result against seven published baselines from the same corpus.

Reference benchmarks (10-fold CV accuracy on the same 9-class corpus design):
  Hillewaere, Manderick & Conklin (2012), ISMIR
    - Global features + SVM RBF (duration): 67.7%
    - Pentagram n-gram (ioi):                76.1%
  Hillewaere, Manderick & Conklin (2014), Springer
    - Alignment (melodic int):               50.0%
    - Pentagram (melodic int):               66.1%
    - Alignment (duration ratio):            63.2%
    - Pentagram (duration ratio):            74.4%
  Conklin (2013), JNMR — MVS(28) ensemble:  88.7%

Output:
    results/dance9/dance9_comparison/summary.csv
    results/dance9/dance9_comparison/fold_details.csv
    results/dance9/dance9_comparison/positioning.csv
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
logger = logging.getLogger("08_run_dance9_comparison")

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

CORPUS = "dance9"
VP_CONFIG = "B"  # metric viewpoint — aligned for dance-type classification

PUBLISHED = {
    "Hillewaere 2014 — alignment (melodic int)": 0.500,
    "Hillewaere 2014 — pentagram (melodic int)": 0.661,
    "Hillewaere 2014 — alignment (dur ratio)": 0.632,
    "Hillewaere 2014 — pentagram (dur ratio)": 0.744,
    "Hillewaere 2012 — global + SVM (duration)": 0.677,
    "Hillewaere 2012 — pentagram (ioi)": 0.761,
    "Conklin 2013    — MVS(28) ensemble": 0.887,
}


def main() -> None:
    logger.info("=" * 70)
    logger.info("Dance-9 benchmark comparison (VP=%s, combined features)", VP_CONFIG)
    logger.info("=" * 70)

    out_dir = RESULTS_DIR / CORPUS / "dance9_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        feature_groups, labels = load_corpus(
            CORPUS, VP_CONFIG, artifacts_dir=ARTIFACTS_DIR, min_class_count=10
        )
    except FileNotFoundError as e:
        logger.error("Artifacts not found — run download_artifacts.py first.\n%s", e)
        sys.exit(1)

    n_pieces = len(labels)
    n_classes = len(set(labels))
    logger.info("Dataset: %d pieces, %d classes", n_pieces, n_classes)

    assembled = assemble_features(feature_groups, "combined")
    X = assembled.values
    y = np.array(labels)
    logger.info("Feature matrix: %d × %d", *X.shape)

    all_clf = get_classifiers()
    to_run = {k: v for k, v in all_clf.items() if k not in ("majority", "stratified")}

    results = []
    for name, clf in to_run.items():
        logger.info("Running %s ...", name)
        results.append(
            run_experiment(
                X=X,
                y=y,
                classifier_name=name,
                classifier=clf,
                feature_config="combined",
                n_splits=10,
                n_repeats=3,
            )
        )

    summary_df = results_to_summary_df(results)
    summary_df.to_csv(out_dir / "summary.csv", index=False)
    fold_details = results_to_fold_details(results)
    fold_details.to_csv(out_dir / "fold_details.csv", index=False)

    summary_df_sorted = summary_df.sort_values("accuracy_mean", ascending=False)
    logger.info("-" * 70)
    logger.info("Results (sorted by accuracy):")
    for _, row in summary_df_sorted.iterrows():
        logger.info(
            "  %-10s  accuracy=%.3f±%.3f  F1_macro=%.3f±%.3f",
            row["classifier"],
            row["accuracy_mean"],
            row["accuracy_std"],
            row["f1_macro_mean"],
            row["f1_macro_std"],
        )

    best = summary_df_sorted.iloc[0]
    our_acc = best["accuracy_mean"]
    our_f1 = best["f1_macro_mean"]

    logger.info("=" * 70)
    logger.info("Positioning against published benchmarks:")
    rows = []
    for label, ref_acc in PUBLISHED.items():
        delta = our_acc - ref_acc
        logger.info("  %-45s acc=%.3f  Δ=%+.3f", label, ref_acc, delta)
        rows.append({"method": label, "accuracy": ref_acc, "delta_vs_ours": -delta})

    rows.append(
        {
            "method": f"Ours ({best['classifier']}, VP=B, combined)",
            "accuracy": our_acc,
            "f1_macro": our_f1,
            "delta_vs_ours": 0.0,
        }
    )
    positioning_df = pd.DataFrame(rows)
    positioning_df.to_csv(out_dir / "positioning.csv", index=False)

    n_beaten = sum(1 for v in PUBLISHED.values() if our_acc > v)
    logger.info(
        "Our best (%s): accuracy=%.3f, F1_macro=%.3f — beats %d/%d published baselines",
        best["classifier"],
        our_acc,
        our_f1,
        n_beaten,
        len(PUBLISHED),
    )
    logger.info("Saved results to %s", out_dir)


if __name__ == "__main__":
    main()
