#!/usr/bin/env python3
"""Stage 1c — Seeded random downsample of the filtered pool to Hillewaere's
exact Dance-9 class counts.

Input:  data/dance9/build/02_filtered.csv
Output: data/dance9/build/03_downsampled.csv  (2,198 rows)

We use a fixed random seed (42) so the selection is reproducible.

If any class has fewer post-filter tunes than Hillewaere's target, we keep
all of them and emit a warning — the downstream pipeline still works; the
paper caption reports actual counts.
"""

from __future__ import annotations

import csv
import logging
import random
import sys
from collections import defaultdict
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUILD_DIR = Path(__file__).resolve().parents[2] / "data" / "dance9" / "build"
IN_PATH = BUILD_DIR / "02_filtered.csv"
OUT_PATH = BUILD_DIR / "03_downsampled.csv"

SEED = 42

# Hillewaere 2014 Table 1 per-class counts. Total = 2,198.
HILLEWAERE_COUNTS = {
    "bourree":     59,
    "hornpipe":   108,
    "jig":        793,
    "march":       76,
    "polska":     339,
    "reel":       453,
    "schottische": 119,
    "strathspey": 123,
    "waltz":      128,
}


def main() -> None:
    if not IN_PATH.exists():
        logger.error("Stage 1b output not found: %s", IN_PATH)
        sys.exit(1)

    with IN_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    logger.info("Loaded %d filtered pieces from %s", len(rows), IN_PATH)

    by_class: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_class[r["dance_type"]].append(r)

    rng = random.Random(SEED)
    selected: list[dict] = []
    actual: dict[str, int] = {}

    for cls, target in HILLEWAERE_COUNTS.items():
        pool = by_class.get(cls, [])
        available = len(pool)
        if available < target:
            logger.warning("  %s: only %d available, target %d — keeping all", cls, available, target)
            chosen = list(pool)
        else:
            chosen = rng.sample(pool, target)
        selected.extend(chosen)
        actual[cls] = len(chosen)

    # Sort for stable output (dance_type, then source_path)
    selected.sort(key=lambda r: (r["dance_type"], r["source_path"]))

    fieldnames = list(rows[0].keys())
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)

    logger.info("=" * 60)
    logger.info("Stage 1c — downsample summary (seed=%d)", SEED)
    logger.info("=" * 60)
    logger.info("  %-12s %5s / %s", "class", "kept", "target")
    logger.info("  " + "-" * 35)
    total = 0
    for cls, target in HILLEWAERE_COUNTS.items():
        logger.info("  %-12s %5d / %s", cls, actual[cls], target)
        total += actual[cls]
    logger.info("  " + "-" * 35)
    logger.info("  %-12s %5d / %d", "TOTAL", total, sum(HILLEWAERE_COUNTS.values()))
    logger.info("Wrote %s", OUT_PATH)


if __name__ == "__main__":
    main()
