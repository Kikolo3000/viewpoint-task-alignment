#!/usr/bin/env python3
"""Stage 1d — Convert the 2,198 downsampled ABC tunes to MusicXML and write the
pipeline-compatible metadata.csv.

Input:  data/dance9/build/03_downsampled.csv
Output: data/dance9/scores/*.musicxml
        data/dance9/metadata.csv   (columns: filename, dance_type, raw_label, source_path)

Naming scheme: `{dance_type}__{short_hash}.musicxml`. The short hash is an 8-hex-digit
digest of the source path, guaranteeing stable filenames and no collisions.

This stage does NOT apply monophonic reduction or repeat handling — those
transformations happen inside the pipeline's `src/cleaning.py::clean_tune`
which is already invoked during mining, matching how SymbTr/Session/Essen
are processed. We write the raw parsed content (all pitches, original repeats)
as MusicXML so the downstream pipeline sees identical input to other corpora.
"""

from __future__ import annotations

import csv
import hashlib
import logging
import multiprocessing as mp
import os
import sys
import warnings
from pathlib import Path

import music21
from music21 import converter

warnings.filterwarnings("ignore", category=UserWarning, module="music21")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "dance9"
BUILD_DIR = DATA_DIR / "build"
IN_PATH = BUILD_DIR / "03_downsampled.csv"
SCORES_DIR = DATA_DIR / "scores"
META_PATH = DATA_DIR / "metadata.csv"


def _short_hash(text: str, n: int = 8) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def _materialize_one(row: dict) -> tuple[str, dict | None, str | None]:
    """Convert one ABC tune to MusicXML.

    Returns (outcome, metadata_row, error_message).
    outcome: "ok" | "parse_error" | "write_error"
    """
    src = Path(row["source_path"])
    dance_type = row["dance_type"]
    out_name = f"{dance_type}__{_short_hash(row['source_path'])}.musicxml"
    out_path = SCORES_DIR / out_name

    # Idempotency: skip already-materialised files
    if out_path.exists():
        meta = {
            "filename": out_name,
            "dance_type": dance_type,
            "raw_label": row.get("raw_label", ""),
            "source_path": row["source_path"],
            "n_notes": row.get("n_notes", ""),
        }
        return "ok", meta, None

    try:
        score = converter.parse(str(src), format="abc")
    except Exception as e:
        return "parse_error", None, f"{type(e).__name__}: {e}"
    if score is None or not list(score.recurse().notes):
        return "parse_error", None, "empty score"

    # music21's MusicXML writer splits multi-part scores into multiple files
    # (`X.musicxml`, `X-1.musicxml`, …). Folk tunes are monophonic — take the
    # first part only and wrap in a single-part Score so the writer emits one
    # file per tune.
    #
    # ABC files containing multiple tunes parse as an ``Opus``. We take the
    # first contained ``Score`` as the tune of interest.
    try:
        if isinstance(score, music21.stream.Opus):
            inner = score.scores
            if inner:
                score = inner[0]
            else:
                return "write_error", None, "empty Opus"
        if hasattr(score, "parts") and score.parts:
            first_part = score.parts[0]
            new_score = music21.stream.Score()
            if score.metadata is not None:
                new_score.insert(0, score.metadata)
            new_score.append(first_part)
            score = new_score
    except Exception as e:
        return "write_error", None, f"part-isolation: {type(e).__name__}: {e}"

    try:
        score.write("musicxml", fp=str(out_path))
    except Exception as e:
        return "write_error", None, f"{type(e).__name__}: {e}"

    meta = {
        "filename": out_name,
        "dance_type": dance_type,
        "raw_label": row.get("raw_label", ""),
        "source_path": row["source_path"],
        "n_notes": row.get("n_notes", ""),
    }
    return "ok", meta, None


def main() -> None:
    if not IN_PATH.exists():
        logger.error("Stage 1c output not found: %s", IN_PATH)
        sys.exit(1)

    SCORES_DIR.mkdir(parents=True, exist_ok=True)

    with IN_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info("Materializing %d tunes → %s", len(rows), SCORES_DIR)

    n_workers = max(1, (os.cpu_count() or 4) - 1)
    logger.info("Using %d workers", n_workers)

    ok: list[dict] = []
    parse_errors: list[tuple[str, str]] = []
    write_errors: list[tuple[str, str]] = []

    with mp.Pool(n_workers) as pool:
        for i, (outcome, meta, err) in enumerate(
            pool.imap_unordered(_materialize_one, rows, chunksize=20),
            start=1,
        ):
            if i % 200 == 0:
                logger.info("  materialized %d / %d (ok=%d)", i, len(rows), len(ok))
            if outcome == "ok":
                ok.append(meta)  # type: ignore[arg-type]
            elif outcome == "parse_error":
                parse_errors.append((err or "", ""))
            else:
                write_errors.append((err or "", ""))

    # Sort metadata by filename for stable output
    ok.sort(key=lambda m: m["filename"])

    fieldnames = ["filename", "dance_type", "raw_label", "source_path", "n_notes"]
    with META_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(ok)

    logger.info("=" * 60)
    logger.info("Stage 1d — materialization summary")
    logger.info("=" * 60)
    logger.info("  Input rows:         %d", len(rows))
    logger.info("  MusicXML written:   %d", len(ok))
    logger.info("  Parse errors:       %d", len(parse_errors))
    logger.info("  Write errors:       %d", len(write_errors))
    logger.info("  Wrote %s", META_PATH)

    if parse_errors[:5]:
        logger.info("  (first 5 parse errors: %s)", [e[0][:80] for e in parse_errors[:5]])
    if write_errors[:5]:
        logger.info("  (first 5 write errors: %s)", [e[0][:80] for e in write_errors[:5]])


if __name__ == "__main__":
    main()
