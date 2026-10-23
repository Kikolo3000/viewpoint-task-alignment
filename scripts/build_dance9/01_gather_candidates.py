#!/usr/bin/env python3
"""Stage 1a — Gather Dance-9 candidate tunes from the chambers ABC aggregator.

Scans every .abc file in $CHAMBERS_DIR (Chambers archive), extracts the R:
(rhythm / dance-type) header, normalises it to one of the 9 Dance-9 classes
and keeps only tunes with an unambiguous single-class label (matching
Hillewaere et al.'s criterion).

Input:  $CHAMBERS_DIR/*.abc
Output: data/dance9/build/01_candidates.csv
        columns: source_path, raw_label, dance_type

The normalisation map merges spelling variants (e.g. "shottish" → "schottische",
"Double jig" → "jig") but rejects ambiguous labels like "jig, march" or
"reel, hornpipe", following Hillewaere's rule.
"""

from __future__ import annotations

import csv
import logging
import os
import re
import sys
from pathlib import Path

# Local copy of John Chambers' Trillian ABC archive; override with the CHAMBERS_DIR env var.
CHAMBERS_DIR = Path(
    os.environ.get("CHAMBERS_DIR", Path(__file__).resolve().parents[2] / "data" / "chambers")
)
OUT_PATH = Path(__file__).resolve().parents[2] / "data" / "dance9" / "build" / "01_candidates.csv"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Dance-9 classes (Hillewaere 2012 Table 1)
DANCE9_CLASSES = (
    "bourree", "hornpipe", "jig", "march", "polska",
    "reel", "schottische", "strathspey", "waltz",
)

# R: header value → canonical Dance-9 class.
# Everything not listed here is either ambiguous or out-of-class and gets dropped.
# Normalisation rules:
#   - Match after lowercasing and stripping whitespace.
#   - Accept minor spelling variants (schottis/schottische/shottish, hornpipe/hornpipes).
#   - Accept jig-family labels (double jig, single jig, slip jig → all "jig")
#     because Hillewaere's Dance-9 treats these as jigs (confirmed by 793 count).
#   - Accept polska variants (slängpolska, jämna polska, hambo-polska, bingsjö-polska,
#     polska J/L1/K1) → all "polska"; these are Swedish polska sub-types.
#   - Reject multi-class labels like "jig, march" or "reel, hornpipe" (ambiguous).
LABEL_MAP: dict[str, str] = {
    # bourrée variants (French; u007f is literal apostrophe escape seen in TeX-encoded ABC)
    "bourree": "bourree",
    "bourrée": "bourree",
    "bourr\\'ee": "bourree",
    "bourr\\`ee": "bourree",
    # hornpipe
    "hornpipe": "hornpipe",
    "hornpipes": "hornpipe",
    # jig family
    "jig": "jig",
    "jigs": "jig",
    "double jig": "jig",
    "single jig": "jig",
    "slip jig": "jig",
    "slipjig": "jig",
    "hop jig": "jig",
    "hop, slip jig": "jig",
    # march
    "march": "march",
    "marches": "march",
    # polska family (Swedish)
    "polska": "polska",
    "slängpolska": "polska",
    "sl\\\"angpolska": "polska",
    "sl\\ängpolska": "polska",
    "jämna polska": "polska",
    "j\\\"amna polska": "polska",
    "polska j": "polska",
    "polska k1": "polska",
    "polska l1": "polska",
    "hambo-polska": "polska",
    "hambopolska": "polska",
    "bingsjö-polska": "polska",
    "bingsj\\\"o-polska": "polska",
    # reel
    "reel": "reel",
    "reels": "reel",
    # schottische variants (European round dance)
    "schottische": "schottische",
    "schottis": "schottische",
    "shottish": "schottische",
    "schottisch": "schottische",
    # strathspey (Scottish)
    "strathspey": "strathspey",
    "strathspeys": "strathspey",
    # waltz
    "waltz": "waltz",
    "waltzes": "waltz",
    "vals": "waltz",
    "valse": "waltz",
}

# Regex to find the first R: header (ABC rhythm/genre field) in a tune.
_R_HEADER = re.compile(r"^R:\s*(.+?)\s*$", re.MULTILINE)


def normalise_label(raw: str) -> str | None:
    """Map raw R: value → canonical Dance-9 class, or None if ambiguous / out of set."""
    key = raw.strip().lower()
    # Reject multi-class labels (Hillewaere's "unambiguous annotation" rule)
    if "," in key or ";" in key or "/" in key:
        return None
    # Strip common prefixes/suffixes
    key = key.strip().strip(".")
    return LABEL_MAP.get(key)


def extract_r_header(abc_text: str) -> str | None:
    """Return first R: value, or None if no R: header found."""
    m = _R_HEADER.search(abc_text)
    if not m:
        return None
    return m.group(1)


def main() -> None:
    if not CHAMBERS_DIR.exists():
        logger.error("Chambers directory not found: %s", CHAMBERS_DIR)
        sys.exit(1)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    abc_files = sorted(CHAMBERS_DIR.rglob("*.abc"))
    logger.info("Found %d ABC files in chambers", len(abc_files))

    class_counts: dict[str, int] = {cls: 0 for cls in DANCE9_CLASSES}
    rows: list[tuple[str, str, str]] = []
    stats = {
        "scanned": 0,
        "no_r_header": 0,
        "ambiguous": 0,
        "out_of_class": 0,
        "kept": 0,
        "read_errors": 0,
    }

    for abc_path in abc_files:
        stats["scanned"] += 1
        if stats["scanned"] % 10000 == 0:
            logger.info("  scanned %d / %d", stats["scanned"], len(abc_files))

        try:
            text = abc_path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            stats["read_errors"] += 1
            continue

        raw_label = extract_r_header(text)
        if raw_label is None:
            stats["no_r_header"] += 1
            continue

        # Ambiguous multi-class tags
        if "," in raw_label or ";" in raw_label or "/" in raw_label:
            stats["ambiguous"] += 1
            continue

        dance_type = normalise_label(raw_label)
        if dance_type is None:
            stats["out_of_class"] += 1
            continue

        # Sanitize for CSV (drop surrogate code points that can leak from
        # errors="replace" decoding of non-UTF-8 bytes in some ABC files).
        safe_path = str(abc_path).encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        safe_raw = raw_label.strip().encode("utf-8", errors="replace").decode("utf-8", errors="replace")
        rows.append((safe_path, safe_raw, dance_type))
        class_counts[dance_type] += 1
        stats["kept"] += 1

    # Write output CSV
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source_path", "raw_label", "dance_type"])
        writer.writerows(rows)

    logger.info("=" * 60)
    logger.info("Stage 1a — candidate gather summary")
    logger.info("=" * 60)
    for k, v in stats.items():
        logger.info("  %-14s %d", k, v)
    logger.info("-" * 60)
    logger.info("Per-class counts (kept):")
    for cls in DANCE9_CLASSES:
        logger.info("  %-12s %5d", cls, class_counts[cls])
    logger.info("-" * 60)
    logger.info("Wrote %s (%d rows)", OUT_PATH, stats["kept"])


if __name__ == "__main__":
    main()
