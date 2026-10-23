#!/usr/bin/env python3
"""Build the Zenodo archive from the original experiments repo.

This is a maintainer script, not part of the reproduction path.

Reads from the parent repo's `results/` and `data/` directories, assembles
the artifact payload, writes `MANIFEST.sha256`, and produces
`paper-folk-patterns-artifacts-v1.zip` in the `to_share/` root.

NOTE (v2, 2026-07-02): this script reflects the ORIGINAL 84-file / 3-corpus build.
The shipped v2 archive was rebuilt to the corrected 12-class Essen task and the
88-file / 4-label / 22-breakdown layout documented in REPRODUCE.md, sourcing Essen
features/interpretability from the essen-rejoin-fix worktree and adding Dance-9. Key
deltas a future rebuild must apply: (1) source Essen from the corrected results, not the
pre-fix parent tree; (2) build labels as (piece,label) covering the UNION of all
feature-parquet indices (essen keyed on `{collection}__{esacid}`, not the metadata
filename); (3) include dance9_B features/breakdown/labels. See
2nd_submission/reports/zenodo-deposit-refresh.md.

Usage:
    uv run python scripts/_build_zenodo_archive.py
"""
from __future__ import annotations

import hashlib
import logging
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TO_SHARE = Path(__file__).resolve().parent.parent
PARENT_REPO = TO_SHARE.parent
ORIG_RESULTS = PARENT_REPO / "results"
ORIG_DATA = PARENT_REPO / "data"

STAGING = TO_SHARE / "_zenodo_staging"
ZIP_OUT = TO_SHARE / "paper-folk-patterns-artifacts-v1.zip"

CORPORA = ["essen", "the_session", "symbtr"]
VP_CONFIGS = ["M", "R", "B", "MR", "MB", "RB", "MRB"]
FEATURE_GROUPS = ["structural", "musical", "bop", "global"]

LABEL_COLUMNS = {
    "essen": ("filename", "collection"),
    "the_session": ("filename", "type"),
    "symbtr": ("filename", "makam"),
}
METADATA_FILES = {
    "essen": "data/essen/metadata.csv",
    "the_session": "data/the_session/manifest.csv",
    "symbtr": "data/symbtr/metadata.csv",
}


def _features_dir_for(corpus: str, vp: str) -> Path:
    """Features layout: MRB baseline at results/{corpus}/features; others at results/{corpus}/{vp}/features."""
    if vp == "MRB":
        return ORIG_RESULTS / corpus / "features"
    return ORIG_RESULTS / corpus / vp / "features"


def _latest_breakdown(corpus: str, vp: str) -> Path | None:
    """MRB baseline at results/{corpus}/{timestamp}/breakdown_optimized.pkl; others at results/{corpus}/{vp}/{ts}/breakdown_optimized.pkl."""
    if vp == "MRB":
        candidates = sorted((ORIG_RESULTS / corpus).glob("*/breakdown_optimized.pkl"))
    else:
        candidates = sorted((ORIG_RESULTS / corpus / vp).glob("*/breakdown_optimized.pkl"))
    return candidates[-1] if candidates else None


def _copy_features(staging: Path) -> None:
    target = staging / "features"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for corpus in CORPORA:
        for vp in VP_CONFIGS:
            src_dir = _features_dir_for(corpus, vp)
            for group in FEATURE_GROUPS:
                src = src_dir / f"{group}.parquet"
                if not src.exists():
                    logger.warning("  missing %s", src)
                    continue
                dest = target / f"{corpus}_{vp}_{group}.parquet"
                shutil.copy2(src, dest)
                count += 1
    logger.info("Copied %d feature parquets", count)


def _copy_breakdowns(staging: Path) -> None:
    target = staging / "breakdowns"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for corpus in CORPORA:
        for vp in VP_CONFIGS:
            src = _latest_breakdown(corpus, vp)
            if src is None:
                logger.warning("  no breakdown for %s/%s", corpus, vp)
                continue
            dest = target / f"{corpus}_{vp}.pkl"
            shutil.copy2(src, dest)
            count += 1
    logger.info("Copied %d breakdown pickles", count)


def _build_labels(staging: Path) -> None:
    """Emit `labels/{corpus}_labels.csv` with columns (piece, label) indexed by
    piece title (== feature-parquet index)."""
    target = staging / "labels"
    target.mkdir(parents=True, exist_ok=True)
    for corpus in CORPORA:
        meta_path = PARENT_REPO / METADATA_FILES[corpus]
        if not meta_path.exists():
            logger.error("  missing metadata: %s", meta_path)
            continue
        key_col, label_col = LABEL_COLUMNS[corpus]
        df = pd.read_csv(meta_path)
        if key_col not in df.columns or label_col not in df.columns:
            logger.error("  %s missing expected columns (%s, %s)", meta_path, key_col, label_col)
            continue
        labels = df[[key_col, label_col]].rename(columns={key_col: "piece", label_col: "label"})
        labels = labels.dropna(subset=["label"]).drop_duplicates(subset="piece")
        dest = target / f"{corpus}_labels.csv"
        labels.to_csv(dest, index=False)
        logger.info("  wrote %s (%d rows)", dest, len(labels))


def _copy_interpretability(staging: Path) -> None:
    """Copy importance_combined.csv for each (corpus, vp) where available."""
    target = staging / "interpretability"
    target.mkdir(parents=True, exist_ok=True)
    count = 0
    for corpus in CORPORA:
        for vp in VP_CONFIGS:
            if vp == "MRB":
                src = ORIG_RESULTS / corpus / "interpretability" / "importance_combined.csv"
            else:
                src = ORIG_RESULTS / corpus / vp / "interpretability" / "importance_combined.csv"
            if src.exists():
                dest = target / f"{corpus}_{vp}_importance_combined.csv"
                shutil.copy2(src, dest)
                count += 1
    logger.info("Copied %d interpretability CSVs", count)


def _copy_extras(staging: Path) -> None:
    robustness = ORIG_RESULTS / "viewpoint_comparison" / "classifier_robustness.csv"
    if robustness.exists():
        shutil.copy2(robustness, staging / "classifier_robustness.csv")
        logger.info("  Copied classifier_robustness.csv")
    else:
        logger.warning("  classifier_robustness.csv not found in original results")

    essen_diag = ORIG_RESULTS / "essen" / "essen_diagnostic" / "comparison.csv"
    if essen_diag.exists():
        shutil.copy2(essen_diag, staging / "essen_diagnostic_comparison.csv")
        logger.info("  Copied essen_diagnostic_comparison.csv")
    else:
        logger.warning("  essen_diagnostic comparison.csv not found")


def _write_manifest(staging: Path) -> None:
    manifest = staging / "MANIFEST.sha256"
    lines: list[str] = ["# SHA-256  path"]
    for path in sorted(staging.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            h = hashlib.sha256()
            with path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            rel = path.relative_to(staging).as_posix()
            lines.append(f"{h.hexdigest()}  {rel}")
    manifest.write_text("\n".join(lines) + "\n")
    logger.info("Wrote manifest with %d entries", len(lines) - 1)


def _write_readme(staging: Path) -> None:
    readme = staging / "README.md"
    readme.write_text(
        "# TISMIR Paper — Artifact Archive (v1)\n\n"
        "Derived artifacts accompanying the TISMIR paper on pattern-derived features\n"
        "for folk music classification. Paired with the code repository at\n"
        "<https://github.com/Kikolo3000/viewpoint-task-alignment>.\n\n"
        "## Layout\n\n"
        "```\n"
        "features/    # 84 parquet files (3 corpora × 7 VP configs × 4 groups)\n"
        "labels/      # 3 CSVs: {essen,the_session,symbtr}_labels.csv\n"
        "breakdowns/  # 21 pickle files (pattern graphs — require simpg_music to load)\n"
        "interpretability/  # Per-VP permutation-importance CSVs\n"
        "classifier_robustness.csv         # Pre-computed fig 8 input\n"
        "essen_diagnostic_comparison.csv   # Pre-computed fig 9 input\n"
        "MANIFEST.sha256                   # Integrity check\n"
        "```\n\n"
        "## Licenses\n\n"
        "- **Code** (GitHub repo): MIT.\n"
        "- **Derived artifacts** (this archive): CC-BY 4.0.\n"
        "- **Source corpora** retain their original licenses:\n"
        "  - The Session (songs/tunes): CC0 1.0.\n"
        "  - SymbTr: CC BY-NC-SA 4.0.\n"
        "  - Essen Folksong Collection: distributed via EsAC, public-domain terms.\n\n"
        "See `DATA_LICENSES.md` in the GitHub repo for full attribution.\n\n"
        "## Usage\n\n"
        "Download this zip, then run from the GitHub repo:\n\n"
        "```bash\n"
        "uv run python scripts/download_artifacts.py \\\n"
        "    --source local --path paper-folk-patterns-artifacts-v1.zip\n"
        "bash reproduce.sh\n"
        "```\n"
    )


def main() -> None:
    if STAGING.exists():
        logger.info("Removing previous staging dir")
        shutil.rmtree(STAGING)
    STAGING.mkdir(parents=True)

    logger.info("=== Building Zenodo archive ===")
    _copy_features(STAGING)
    _copy_breakdowns(STAGING)
    _build_labels(STAGING)
    _copy_interpretability(STAGING)
    _copy_extras(STAGING)
    _write_readme(STAGING)
    _write_manifest(STAGING)

    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
    logger.info("Zipping → %s", ZIP_OUT)
    with zipfile.ZipFile(ZIP_OUT, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for path in sorted(STAGING.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(STAGING))

    size_mb = ZIP_OUT.stat().st_size / (1 << 20)
    logger.info("Done: %s (%.1f MB)", ZIP_OUT, size_mb)
    logger.info("Staging dir left at %s for inspection", STAGING)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
