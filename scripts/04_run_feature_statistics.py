#!/usr/bin/env python3
"""WP5: Feature value statistics and per-class F1 improvement.

WP5a: Kruskal-Wallis per-feature tests across classes (MRB baseline).
WP5b: Per-class F1 improvement between global and combined for each
      corpus's best VP config.

Outputs:
    results/{corpus}/feature_statistics.csv
    results/viewpoint_comparison/per_class_improvement_{corpus}.csv
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from folk_patterns_repro.data_loading import assemble_features, load_corpus
from folk_patterns_repro.viewpoint_analysis import (
    compute_feature_statistics,
    compute_per_class_improvement,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

CORPORA = ["symbtr", "the_session", "essen"]
CORPUS_BEST_VP = {
    "symbtr": "M",
    "the_session": "B",
    "essen": "MR",
}


def run_feature_statistics(corpus: str, vp_config: str = "MRB") -> None:
    print(f"\n{'=' * 60}")
    print(f"WP5a: Feature statistics for {corpus} (vp={vp_config})")
    print(f"{'=' * 60}")

    try:
        feature_groups, labels = load_corpus(
            corpus, vp_config, artifacts_dir=ARTIFACTS_DIR, min_class_count=10
        )
    except FileNotFoundError as e:
        print(f"  SKIP: {e}")
        return

    try:
        assembled = assemble_features(feature_groups, "combined")
    except ValueError as e:
        print(f"  Cannot assemble combined features: {e}")
        return

    stats_df = compute_feature_statistics(assembled, labels)

    out_dir = RESULTS_DIR / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "feature_statistics.csv"
    stats_df.to_csv(out_path, index=False)

    n_sig = int(stats_df["significant_bonferroni"].sum()) if not stats_df.empty else 0
    print(f"  {len(stats_df)} features tested")
    print(f"  {n_sig}/{len(stats_df)} significant after Bonferroni correction")

    if not stats_df.empty:
        print("\n  Top 5 discriminative features:")
        for _, row in stats_df.head(5).iterrows():
            print(
                f"    {row['feature']}: H={row['kruskal_wallis_h']:.1f}, "
                f"η²={row['eta_squared']:.3f}, p={row['kruskal_wallis_p']:.2e}"
            )


def run_per_class_improvement() -> None:
    print(f"\n{'=' * 60}")
    print("WP5b: Per-class F1 improvement for best VP per corpus")
    print(f"{'=' * 60}")

    out_dir = RESULTS_DIR / "viewpoint_comparison"
    out_dir.mkdir(parents=True, exist_ok=True)

    for corpus in CORPORA:
        vp = CORPUS_BEST_VP[corpus]
        result = compute_per_class_improvement(corpus, vp)
        if result is not None:
            result.to_csv(out_dir / f"per_class_improvement_{corpus}.csv", index=False)
            print(f"\n  {corpus} (VP={vp}):")
            for _, row in result.iterrows():
                print(
                    f"    {row['feature_config']}/{row['classifier']}: "
                    f"F1={row['f1_macro_mean']:.4f}±{row['f1_macro_std']:.4f}"
                )


def main() -> None:
    print("WP5: Feature Statistics & Per-Class Analysis")
    print("=" * 60)

    for corpus in CORPORA:
        run_feature_statistics(corpus, vp_config="MRB")

    run_per_class_improvement()
    print("\nDone!")


if __name__ == "__main__":
    main()
