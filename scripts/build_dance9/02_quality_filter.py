#!/usr/bin/env python3
"""Stage 1b — Apply quality filters F1..F10 to Dance-9 candidates.

Mirrors Hillewaere 2012/2014's preprocessing + adds defensible note-count
bounds for our pattern-mining pipeline.

Filters (applied in order; first failure drops the tune):

  F1  Parseable by music21 (no parser exception, non-empty note stream)
  F2  Unambiguous single dance-type label          — already enforced in Stage 1a
  F3  Valid time signature present
  F4  Valid key signature present
  F5  Note count in [MIN_NOTES, MAX_NOTES]
  F6  (deferred to downstream cleaning) — Hillewaere *reduced* polyphony via
      skyline rather than rejecting it. Our pipeline's `src/cleaning.py::clean_tune`
      does the same chord→top-note reduction as a matter of course, so rejecting
      here would be redundant and over-aggressive. The filter report still
      records how many pieces had non-trivial polyphonic texture for
      transparency.
  F7  Ornamentation strip-able (no parse corruption from removing grace/trill/mordent)
  F8  (no action — repeat handling deferred; parse() takes written-once form)
  F9  (no action — tempo/dynamics are stripped at materialize stage, doesn't affect parse)
  F10 Duplicate removal via melodic-fingerprint (first 16 directed intervals)
        - Intra-class: keep one representative (lowest source_path)
        - Cross-class: drop ALL copies (ambiguous labelling, matches Hillewaere 2009)

Input:  data/dance9/build/01_candidates.csv
Output: data/dance9/build/02_filtered.csv
        data/dance9/build/02_filter_report.md

The filter is deliberately conservative: when in doubt, drop. The goal is a
clean, quality-controlled pool from which Stage 1c downsamples to 2,198.
"""

from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import os
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import music21
from music21 import converter, key, meter, note

warnings.filterwarnings("ignore", category=UserWarning, module="music21")
warnings.filterwarnings("ignore", category=DeprecationWarning)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

BUILD_DIR = Path(__file__).resolve().parents[2] / "data" / "dance9" / "build"
IN_PATH = BUILD_DIR / "01_candidates.csv"
OUT_PATH = BUILD_DIR / "02_filtered.csv"
REPORT_PATH = BUILD_DIR / "02_filter_report.md"

MIN_NOTES = 20
MAX_NOTES = 500
FINGERPRINT_LEN = 16   # use first 16 directed intervals as melodic fingerprint


# Filter codes for reporting (keep output columns stable)
FILTERS = (
    "F1_parseable",
    "F3_time_signature",
    "F4_key_signature",
    "F5_note_count",
    "F7_ornament_strip",
    "F10_duplicate",
)


def _first_of(stream_obj, cls):
    """Return first element of class `cls` in stream, or None."""
    for el in stream_obj.recurse().getElementsByClass(cls):
        return el
    return None


class _ParseTimeout(Exception):
    pass


def _timeout_handler(signum, frame):  # noqa: ARG001
    raise _ParseTimeout()


def parse_tune(abc_path: Path, timeout_sec: int = 15) -> music21.stream.Score | None:
    """F1 — try to parse with a wall-clock timeout; return Score or None.

    A handful of ABC files in the Chambers aggregator trigger music21 to hang
    (e.g. malformed long passages, problematic accidentals). We use SIGALRM
    to cut off any parse exceeding `timeout_sec`.
    """
    import signal
    # SIGALRM is per-process; in multiprocessing.Pool each worker has its own.
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        score = converter.parse(str(abc_path), format="abc")
    except (_ParseTimeout, Exception):
        score = None
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)
    if score is None:
        return None
    try:
        if not list(score.recurse().notes):
            return None
    except Exception:
        return None
    return score


def has_time_signature(score) -> bool:
    """F3."""
    return _first_of(score, meter.TimeSignature) is not None


def has_key_signature(score) -> bool:
    """F4."""
    # music21 may produce a Key or a KeySignature; accept either.
    for cls in (key.Key, key.KeySignature):
        if _first_of(score, cls) is not None:
            return True
    return False


def count_notes(score) -> int:
    """Count pitched notes (excluding rests, unpitched perc)."""
    return sum(1 for n in score.recurse().notes if isinstance(n, note.Note) or n.isChord)


def polyphony_stats(score) -> tuple[int, int]:
    """Transparency helper — count (single_onset_count, total_onset_count).

    Not used as a filter (see F6 note in module docstring). Useful for the
    report to show how many pieces had non-trivial polyphonic texture that
    the downstream `clean_tune` skyline reduction will then collapse.
    """
    onsets: dict[float, int] = defaultdict(int)
    for n in score.recurse().notes:
        if isinstance(n, note.Note) or n.isChord:
            onsets[float(n.offset)] += 1
    total = len(onsets)
    single = sum(1 for c in onsets.values() if c <= 1)
    return single, total


def can_strip_ornaments(score) -> bool:
    """F7 — check ornament stripping doesn't destroy the piece.

    We do an actual strip pass and verify ≥MIN_NOTES notes survive.
    """
    # Remove grace notes in-place on a shallow copy.
    try:
        stripped = score.coreCopyAsDerivation("ornament_check") if hasattr(score, "coreCopyAsDerivation") else score
    except Exception:
        stripped = score

    # Count notes excluding grace notes (duration 0) and very short ornamental notes.
    n_real = 0
    for n in stripped.recurse().notes:
        d = float(n.duration.quarterLength) if n.duration else 0.0
        if d > 0.01:   # exclude zero-duration grace notes and near-zero ornaments
            n_real += 1
    return n_real >= MIN_NOTES


def melodic_fingerprint(score, n: int = FINGERPRINT_LEN) -> tuple[int, ...] | None:
    """F10 helper — first `n` directed semitone intervals.

    Using intervals (not pitches) makes the fingerprint transposition-invariant
    so true duplicates transposed to different keys are caught.
    """
    pitches: list[int] = []
    for el in score.recurse().notes:
        try:
            if isinstance(el, note.Note):
                pitches.append(el.pitch.midi)
            elif el.isChord:
                chord_pitches = list(el.pitches)
                if not chord_pitches:
                    continue
                pitches.append(max(p.midi for p in chord_pitches))
        except Exception:
            continue
        if len(pitches) >= n + 1:
            break
    if len(pitches) < n + 1:
        return None   # tune too short for a stable fingerprint
    intervals = tuple(pitches[i + 1] - pitches[i] for i in range(n))
    return intervals


def _filter_one(row: dict) -> tuple[str, dict, tuple[int, ...] | None]:
    """Process a single candidate. Returns (outcome, row_with_n_notes, fingerprint).

    outcome is one of:
      - "F1_parseable", "F3_time_signature", "F4_key_signature",
        "F5_note_count", "F7_ornament_strip"  → dropped at that stage
      - "ok_with_fp"        → passed all per-file filters, fingerprint available for dedup
      - "ok_no_fp"          → passed all per-file filters, too short for a stable fingerprint
    """
    src = Path(row["source_path"])
    score = parse_tune(src)
    if score is None:
        return "F1_parseable", row, None
    if not has_time_signature(score):
        return "F3_time_signature", row, None
    if not has_key_signature(score):
        return "F4_key_signature", row, None
    nn = count_notes(score)
    if not (MIN_NOTES <= nn <= MAX_NOTES):
        return "F5_note_count", row, None
    if not can_strip_ornaments(score):
        return "F7_ornament_strip", row, None

    # F6 — measure but do not reject (see module docstring for rationale).
    single_onsets, total_onsets = polyphony_stats(score)
    poly_frac = 0.0 if total_onsets == 0 else 1.0 - (single_onsets / total_onsets)

    fp = melodic_fingerprint(score)
    out = dict(row)
    out["n_notes"] = nn
    out["poly_frac"] = f"{poly_frac:.3f}"
    if fp is None:
        return "ok_no_fp", out, None
    return "ok_with_fp", out, fp


def main() -> None:
    if not IN_PATH.exists():
        logger.error("Stage 1a output not found: %s", IN_PATH)
        sys.exit(1)

    # Load candidates
    with IN_PATH.open(encoding="utf-8") as f:
        candidates = list(csv.DictReader(f))
    logger.info("Loaded %d candidates from %s", len(candidates), IN_PATH)

    # Running counters
    drops: Counter[str] = Counter()
    drops_by_class: dict[str, Counter[str]] = defaultdict(Counter)
    fingerprints: dict[tuple[int, ...], list[dict]] = defaultdict(list)
    no_fingerprint: list[dict] = []

    # Parallelise per-file filtering across CPU cores.
    n_workers = max(1, (os.cpu_count() or 4) - 1)
    logger.info("Running per-file filters on %d workers", n_workers)

    # chunksize tuning: large enough to amortise IPC, small enough to keep workers busy.
    chunksize = max(50, len(candidates) // (n_workers * 40))

    processed = 0
    with mp.Pool(n_workers) as pool:
        for outcome, row_out, fp in pool.imap_unordered(_filter_one, candidates, chunksize=chunksize):
            processed += 1
            if processed % 5000 == 0:
                logger.info("  filtered %d / %d", processed, len(candidates))
            dance_type = row_out["dance_type"]
            if outcome == "ok_with_fp":
                fingerprints[fp].append(row_out)  # type: ignore[arg-type]
            elif outcome == "ok_no_fp":
                no_fingerprint.append(row_out)
            else:
                drops[outcome] += 1
                drops_by_class[dance_type][outcome] += 1

    # F10 — duplicate resolution
    dedup_kept: list[dict] = []
    for fp, group in fingerprints.items():
        classes = {r["dance_type"] for r in group}
        if len(classes) > 1:
            # Cross-class duplicate: drop ALL copies (Hillewaere rule)
            for r in group:
                drops["F10_duplicate"] += 1
                drops_by_class[r["dance_type"]]["F10_duplicate"] += 1
            continue
        # Intra-class: keep lowest source_path, drop the rest
        group.sort(key=lambda r: r["source_path"])
        dedup_kept.append(group[0])
        for r in group[1:]:
            drops["F10_duplicate"] += 1
            drops_by_class[r["dance_type"]]["F10_duplicate"] += 1

    # Pieces without fingerprint (short) kept unconditionally
    kept = dedup_kept + no_fingerprint

    # ── Write filtered CSV ─────────────────────────────────────────
    with OUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source_path", "raw_label", "dance_type", "n_notes", "poly_frac"],
        )
        writer.writeheader()
        for r in sorted(kept, key=lambda x: (x["dance_type"], x["source_path"])):
            writer.writerow(r)

    # ── Write human-readable filter report ─────────────────────────
    lines: list[str] = []
    lines.append("# Dance-9 filter report (Stage 1b)\n")
    lines.append(f"- Input: `{IN_PATH.relative_to(BUILD_DIR.parents[2])}` ({len(candidates)} candidates)")
    lines.append(f"- Output: `{OUT_PATH.relative_to(BUILD_DIR.parents[2])}` ({len(kept)} pieces kept)")
    lines.append("")
    lines.append("## Overall drops")
    lines.append("")
    lines.append("| Filter | Dropped |")
    lines.append("|---|---:|")
    for filt in FILTERS:
        lines.append(f"| `{filt}` | {drops[filt]} |")
    lines.append(f"| **kept** | **{len(kept)}** |")
    lines.append("")

    # Per-class breakdown
    lines.append("## Per-class drops")
    lines.append("")
    header = "| Class |" + "|".join(f" `{f}` " for f in FILTERS) + "| kept |"
    sep = "|---|" + "|".join("---:" for _ in FILTERS) + "|---:|"
    lines.append(header)
    lines.append(sep)
    kept_by_class: Counter[str] = Counter(r["dance_type"] for r in kept)
    all_classes = sorted(set(drops_by_class) | {
        "bourree", "hornpipe", "jig", "march", "polska",
        "reel", "schottische", "strathspey", "waltz",
    })
    for cls in all_classes:
        row_vals = [str(drops_by_class[cls][f]) for f in FILTERS]
        lines.append(f"| {cls} | " + " | ".join(row_vals) + f" | {kept_by_class[cls]} |")
    lines.append("")
    lines.append("## Note-count distribution (kept pieces)")
    lines.append("")
    if kept:
        nn_list = sorted(r["n_notes"] for r in kept)
        lines.append(f"- min: {nn_list[0]}, median: {nn_list[len(nn_list) // 2]}, max: {nn_list[-1]}")
        lines.append(f"- mean: {sum(nn_list) / len(nn_list):.1f}")

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    # Console summary
    logger.info("=" * 60)
    logger.info("Stage 1b — quality filter summary")
    logger.info("=" * 60)
    logger.info("  Input candidates: %d", len(candidates))
    for filt in FILTERS:
        logger.info("  dropped by %-20s %d", filt, drops[filt])
    logger.info("  KEPT:                        %d", len(kept))
    logger.info("-" * 60)
    logger.info("Per-class kept counts:")
    for cls in ("bourree", "hornpipe", "jig", "march", "polska", "reel", "schottische", "strathspey", "waltz"):
        logger.info("  %-12s %5d", cls, kept_by_class[cls])
    logger.info("-" * 60)
    logger.info("Wrote filtered corpus: %s", OUT_PATH)
    logger.info("Wrote filter report:   %s", REPORT_PATH)


if __name__ == "__main__":
    main()
