#!/usr/bin/env python3
"""WP2b: Compare feature importance across viewpoint configs.

Reads the pre-computed `importance_combined.csv` files that
`download_artifacts.py` seeded into `results/{corpus}/{vp}/interpretability/`,
compares rank shifts, and writes:

    results/viewpoint_comparison/importance_shift_{corpus}.csv
"""
from __future__ import annotations

import logging
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from folk_patterns_repro.viewpoint_analysis import compare_importance_across_viewpoints

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS = REPO_ROOT / "results"
OUT_DIR = RESULTS / "viewpoint_comparison"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Corpus → VPs to compare: [best, worst, baseline] ordering from the paper.
CORPUS_VP_CONFIGS = {
    "symbtr": ["M", "R", "MRB"],
    "the_session": ["B", "M", "MRB"],
    "essen": ["MR", "R", "MRB"],
}


def _ensure_mrb_in_vp_dir(corpus: str) -> None:
    """Copy the baseline MRB importance CSV into the MRB/ subdir if needed.

    Upstream convention: `results/{corpus}/interpretability/importance_combined.csv`
    is the default (MRB) baseline. To reuse the uniform per-VP access path, we mirror
    it into `results/{corpus}/MRB/interpretability/` when missing.
    """
    baseline = RESULTS / corpus / "interpretability" / "importance_combined.csv"
    mrb_vp = RESULTS / corpus / "MRB" / "interpretability" / "importance_combined.csv"
    if baseline.exists() and not mrb_vp.exists():
        mrb_vp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(baseline, mrb_vp)
        logger.info("Copied MRB baseline importance to %s", mrb_vp)


def main() -> None:
    print("=" * 60)
    print("WP2b: VP Importance Comparison")
    print("=" * 60)

    for corpus, vp_configs in CORPUS_VP_CONFIGS.items():
        print(f"\n--- {corpus} ({vp_configs}) ---")
        _ensure_mrb_in_vp_dir(corpus)

        available = []
        for vp in vp_configs:
            path = RESULTS / corpus / vp / "interpretability" / "importance_combined.csv"
            if path.exists():
                available.append(vp)
                print(f"  Found: {vp} → {path}")
            else:
                print(f"  MISSING: {vp} → {path}")

        if len(available) < 2:
            print(f"  SKIP: need ≥2 VP importance files, found {len(available)}")
            continue

        result = compare_importance_across_viewpoints(corpus, available)
        if not result.empty:
            out_path = OUT_DIR / f"importance_shift_{corpus}.csv"
            result.to_csv(out_path, index=False)
            print(f"  Saved: {out_path} ({len(result)} features)")

            if "rank_shift" in result.columns:
                risers = result.nlargest(5, "rank_shift")
                fallers = result.nsmallest(5, "rank_shift")
                print(f"\n  Top 5 features that RISE under {available[0]} vs {available[1]}:")
                for _, row in risers.iterrows():
                    print(f"    {row['feature']}: shift={row['rank_shift']:+.0f}")
                print(f"\n  Top 5 features that FALL under {available[0]} vs {available[1]}:")
                for _, row in fallers.iterrows():
                    print(f"    {row['feature']}: shift={row['rank_shift']:+.0f}")


if __name__ == "__main__":
    main()
