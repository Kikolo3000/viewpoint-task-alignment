# Dance-9 Corpus Reconstruction

Reconstructs a corpus matching Hillewaere, Manderick & Conklin's "Dance-9"
benchmark (Hillewaere 2012, 2014; Conklin 2013) from the John Chambers
Trillian ABC aggregator (~194k ABC files),
which is the same source used by the original authors.

The result is a 2,198-piece symbolic corpus in MusicXML, labelled with one of
nine dance types, usable as a drop-in corpus in our mining/classification
pipeline for a head-to-head benchmark comparison in the TISMIR paper (§5.10).

## Pipeline stages

Four numbered scripts, run in order. Each reads the previous stage's output
and writes to `data/dance9/build/`.

| # | Script | Input | Output | Purpose |
|---|---|---|---|---|
| 1 | `01_gather_candidates.py` | `$CHAMBERS_DIR/*.abc` | `build/01_candidates.csv` | Scan every ABC file, extract `R:` header, normalise to one of the 9 Dance-9 classes, keep only unambiguous single-label tunes. |
| 2 | `02_quality_filter.py` | `build/01_candidates.csv` | `build/02_filtered.csv`, `build/02_filter_report.md` | Apply F1–F10 quality filters (parseable, valid meter+key, length bounds, monophonic reduction, strip ornaments, dedup). Log per-filter drop counts. |
| 3 | `03_downsample.py` | `build/02_filtered.csv` | `build/03_downsampled.csv` | Seeded random sample per class to match Hillewaere's exact counts (Bourrée 59, Hornpipe 108, Jig 793, March 76, Polska 339, Reel 453, Schottische 119, Strathspey 123, Waltz 128 = 2,198). |
| 4 | `04_materialize.py` | `build/03_downsampled.csv` | `data/dance9/scores/*.musicxml`, `data/dance9/metadata.csv` | Apply preprocessing (monophonic reduce, strip ornaments, don't expand repeats), write MusicXML files + pipeline-compatible metadata CSV. |

## Hillewaere's original selection criteria

From Hillewaere 2012 (ISMIR) and 2014 (Springer):

- Source: ~14,000 ABC tunes from John Chambers' Trillian aggregator.
- Selection: tunes with an **unambiguous dance-type annotation** (the ABC `R:` header).
- Class threshold: ≥50 tunes per dance type.
- Preprocessing: monophonic reduction (highest note of double stops), strip grace notes / trills / staccato, ignore repeat signs, keep key + time signature, strip tempo and dynamics.
- Deduplication (Hillewaere 2009, same cleaning pipeline): cluster by Jesser feature vector; cross-class duplicates dropped; intra-class duplicates collapsed to one.

Our reconstruction applies the same source, the same label-unambiguity rule,
the same ≥50 threshold, and the same preprocessing. The only additional
filter we apply is F5 (note count in [20, 500]) to exclude stub fragments
and multi-section medleys. Full filter list in `02_quality_filter.py`.

## Running

Stages 2 and 4 additionally need `music21` (`uv pip install music21`). Point stage 1 at a local
copy of the Chambers archive (default: `data/chambers/`):

```bash
export CHAMBERS_DIR=/path/to/chambers   # ~194k .abc files
uv run python scripts/build_dance9/01_gather_candidates.py
uv run python scripts/build_dance9/02_quality_filter.py
uv run python scripts/build_dance9/03_downsample.py
uv run python scripts/build_dance9/04_materialize.py
```

All scripts are idempotent and read-only w.r.t. the chambers source.

## Reference labels

Dance-9 classes and Hillewaere 2014's piece counts (Table 1):

| Dance | Pieces | Typical meter |
|---|---|---|
| Bourrée | 59 | 3/8 or 2/4 (Auvergnat, French traditional) |
| Hornpipe | 108 | 4/4 with dotted rhythm (British/Irish) |
| Jig | 793 | 6/8 (Irish) |
| March | 76 | 4/4 or 2/4 |
| Polska | 339 | 3/4 (Swedish, asymmetric beat) |
| Reel | 453 | 4/4 (Irish/Scottish) |
| Schottische | 119 | 2/4 (Continental European round dance) |
| Strathspey | 123 | 4/4 with Scottish snap (Scottish) |
| Waltz | 128 | 3/4 |
