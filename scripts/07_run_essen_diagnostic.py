#!/usr/bin/env python3
"""Essen BoP diagnostic: isolating BoP noise on the MR config.

Runs all non-baseline classifiers on Essen/MR for three feature configs
(global, combined_no_bop, combined) to test whether the 50-dim BoP SVD
component degrades Essen classification.

Outputs:
    results/essen/essen_diagnostic/summary.csv
    results/essen/essen_diagnostic/fold_details.csv
    results/essen/essen_diagnostic/comparison.csv
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

VP_CONFIG = "MR"
CONFIGS_TO_TEST = ("global", "combined_no_bop", "combined")


def main() -> None:
    print("=" * 70)
    print("Essen Boundary Condition Diagnostic (VP=MR)")
    print("=" * 70)

    artifacts_dir = REPO_ROOT / "artifacts"
    results_dir = REPO_ROOT / "results"
    out_dir = results_dir / "essen" / "essen_diagnostic"
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        feature_groups, labels = load_corpus(
            "essen", VP_CONFIG, artifacts_dir=artifacts_dir, min_class_count=10
        )
    except FileNotFoundError as e:
        logger.error("SKIP: %s", e)
        return

    n_pieces = len(labels)
    n_classes = len(set(labels))
    logger.info("  %d pieces, %d classes", n_pieces, n_classes)

    classifiers = {
        name: clf for name, clf in get_classifiers().items()
        if name not in ("majority", "stratified")
    }

    results: list[dict] = []
    for config_name in CONFIGS_TO_TEST:
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
                n_splits=10, n_repeats=3,
            ))

    if not results:
        logger.error("No results produced")
        return

    summary = results_to_summary_df(results)
    summary.to_csv(out_dir / "summary.csv", index=False)
    results_to_fold_details(results).to_csv(out_dir / "fold_details.csv", index=False)

    rows = []
    for clf_name in classifiers:
        clf_summary = summary[summary["classifier"] == clf_name]

        def pick(cfg: str) -> float | None:
            row = clf_summary[clf_summary["feature_config"] == cfg]
            return float(row.iloc[0]["f1_macro_mean"]) if not row.empty else None

        g = pick("global")
        nb = pick("combined_no_bop")
        c = pick("combined")
        rows.append({
            "classifier": clf_name,
            "global_f1": round(g, 4) if g is not None else None,
            "combined_no_bop_f1": round(nb, 4) if nb is not None else None,
            "combined_f1": round(c, 4) if c is not None else None,
            "bop_hurts": nb > c if (nb is not None and c is not None) else None,
            "pattern_helps": nb > g if (nb is not None and g is not None) else None,
        })

    comparison = pd.DataFrame(rows)
    comparison.to_csv(out_dir / "comparison.csv", index=False)

    print("\n" + "=" * 70)
    print("ESSEN BOP DIAGNOSTIC: global vs combined_no_bop vs combined")
    print("=" * 70)
    for _, row in comparison.iterrows():
        print(
            f"  {row['classifier']:10s}: global={row['global_f1']}  "
            f"no_bop={row['combined_no_bop_f1']}  combined={row['combined_f1']}  "
            f"| bop_hurts={row['bop_hurts']}  pattern_helps={row['pattern_helps']}"
        )
    print(f"\n  Saved to: {out_dir}")


if __name__ == "__main__":
    main()
