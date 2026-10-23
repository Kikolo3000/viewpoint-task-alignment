#!/usr/bin/env python3
"""Download and unpack the Zenodo artifact archive.

Two source modes:
    --source zenodo  (default)  Downloads the zip from the Zenodo DOI below.
    --source local --path PATH  Uses a local zip file (useful for dev builds).

After extraction the top-level layout is:
    artifacts/features/...       # input parquets
    artifacts/labels/...          # input label CSVs
    artifacts/breakdowns/...      # pattern breakdown pickles (optional)
    artifacts/MANIFEST.sha256     # integrity manifest
    artifacts/README.md

Pre-computed outputs that the analysis scripts expect inside the results tree
(interpretability CSVs, essen_diagnostic comparison, classifier robustness
summary) are mirrored into `results/` automatically.
"""
from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = REPO_ROOT / "artifacts"
RESULTS_DIR = REPO_ROOT / "results"

# Zenodo v1.1 (record 21709412; concept DOI 10.5281/zenodo.21129773).
ZENODO_ZIP_URL = "https://zenodo.org/records/21709412/files/patterns-artifacts-v1.zip?download=1"


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            buf = f.read(chunk)
            if not buf:
                break
            h.update(buf)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    import urllib.request

    logger.info("Downloading %s", url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as resp, dest.open("wb") as f:
        shutil.copyfileobj(resp, f)
    logger.info("Downloaded %s (%s bytes)", dest, dest.stat().st_size)


def _verify_manifest(extracted_root: Path) -> None:
    manifest = extracted_root / "MANIFEST.sha256"
    if not manifest.exists():
        logger.warning("No MANIFEST.sha256 found — skipping integrity check")
        return
    logger.info("Verifying SHA256 manifest...")
    errors = 0
    with manifest.open() as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            expected, rel_path = line.split("  ", 1)
            target = extracted_root / rel_path
            if not target.exists():
                logger.error("Missing file listed in manifest: %s", rel_path)
                errors += 1
                continue
            actual = _sha256(target)
            if actual != expected:
                logger.error("Hash mismatch for %s", rel_path)
                errors += 1
    if errors:
        raise RuntimeError(f"Manifest verification failed ({errors} errors)")
    logger.info("Manifest OK.")


def _seed_results_tree(artifacts_dir: Path, results_dir: Path) -> None:
    """Copy interpretability/ and pre-computed output CSVs into `results/` so the
    downstream figure scripts find them at their expected paths."""

    interp_src = artifacts_dir / "interpretability"
    if interp_src.exists():
        for csv in interp_src.glob("*.csv"):
            # filename: {corpus}_{vp}_importance_{config}.csv
            name = csv.stem  # no .csv
            parts = name.split("_")
            # Support both:
            #   essen_MRB_importance_combined  (vp=MRB)
            #   the_session_MRB_importance_combined  (vp=MRB, 2-word corpus)
            # We detect by scanning from the right: the last 2 tokens are 'importance' + config.
            # Everything before the last 3 tokens is the corpus name; the 3rd-from-last is the VP.
            if len(parts) < 4 or parts[-2] != "importance":
                logger.warning("Unexpected interpretability filename: %s", csv.name)
                continue
            config = parts[-1]
            vp = parts[-3]
            corpus = "_".join(parts[:-3])

            if vp == "MRB":
                # Place MRB baseline at both locations for uniformity.
                for dest in (
                    results_dir / corpus / "interpretability" / f"importance_{config}.csv",
                    results_dir / corpus / "MRB" / "interpretability" / f"importance_{config}.csv",
                ):
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(csv, dest)
                    logger.info("Seeded %s", dest)
            else:
                dest = results_dir / corpus / vp / "interpretability" / f"importance_{config}.csv"
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(csv, dest)
                logger.info("Seeded %s", dest)

    # Optional pre-computed outputs (speeds up fig8/fig9 if user skips 05/07).
    robustness_src = artifacts_dir / "classifier_robustness.csv"
    if robustness_src.exists():
        dest = results_dir / "viewpoint_comparison" / "classifier_robustness.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(robustness_src, dest)
        logger.info("Seeded %s", dest)

    essen_diag_src = artifacts_dir / "essen_diagnostic_comparison.csv"
    if essen_diag_src.exists():
        dest = results_dir / "essen" / "essen_diagnostic" / "comparison.csv"
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(essen_diag_src, dest)
        logger.info("Seeded %s", dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=("zenodo", "local"), default="zenodo")
    parser.add_argument(
        "--path",
        type=Path,
        default=None,
        help="Local path to the zip file (required when --source local).",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=ARTIFACTS_DIR,
        help="Destination directory (default: to_share/artifacts/).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Results tree to seed (default: to_share/results/).",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Do not verify the SHA256 manifest after extraction.",
    )
    args = parser.parse_args()

    if args.source == "local":
        if args.path is None or not args.path.exists():
            parser.error("--source local requires --path pointing to an existing zip")
        zip_path = args.path
    else:
        zip_path = REPO_ROOT / "paper-folk-patterns-artifacts.zip"
        if not zip_path.exists():
            _download(ZENODO_ZIP_URL, zip_path)
        else:
            logger.info("Using cached download %s", zip_path)

    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s → %s", zip_path, args.artifacts_dir)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.artifacts_dir)

    if not args.skip_verify:
        _verify_manifest(args.artifacts_dir)

    _seed_results_tree(args.artifacts_dir, args.results_dir)

    logger.info("Done. Artifacts at %s.", args.artifacts_dir)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)
