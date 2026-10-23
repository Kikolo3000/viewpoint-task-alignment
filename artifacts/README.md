# TISMIR Paper — Artifact Archive (v2)

Derived artifacts accompanying the TISMIR paper on viewpoint–task alignment for
symbolic folk-music classification. Paired with the code repository
<https://github.com/Kikolo3000/viewpoint-task-alignment>. Version 2 reflects the corrected 12-class /
4,580-piece Essen task (metadata-join fix) and adds the Dance-9 benchmark artifacts.

## Layout

```
features/          # 88 parquet files: {corpus}_{vp}_{group}.parquet
                   #   3 corpora × 7 viewpoint configs × 4 groups (structural,
                   #   musical, bop, global) + dance9_B × 4 groups
labels/            # 4 CSVs: {essen,the_session,symbtr,dance9}_labels.csv
                   #   (piece, label); covers every feature-parquet row
breakdowns/        # 22 pickle files: {corpus}_{vp}.pkl + dance9_B.pkl
                   #   (pattern graphs — require simpg_music to load)
interpretability/  # 7 permutation-importance CSVs (SymbTr M/R/MRB;
                   #   The Session B/M/MRB; Essen MR)
classifier_robustness.csv         # Pre-computed fig 8 input (12-class Essen)
essen_diagnostic_comparison.csv   # Pre-computed fig 9 input (12-class Essen)
MANIFEST.sha256                   # Integrity check
```

Corpus tasks in the deposited features (post ≥10-piece class filter):
Essen 12 collections / 4,580 pieces (MRB base; per-viewpoint configs mine more,
all covered by the label file), The Session 12 tune types, SymbTr 28 makams,
Dance-9 9 dance types.

The analysis reproduces every table and figure from the `features/` and `labels/`
in this archive with no dependency on the `SimPG-Music` miner. The `breakdowns/`
are the raw mining output and require `simpg_music` to deserialise; that miner is
not yet public (its methods paper is under review), the exact version used is
recorded by commit hash `82857bd`, and its source will be released with that paper.

## Licenses

- **Code** (GitHub repo): MIT.
- **Derived artifacts** (this archive): CC-BY 4.0.
- **Source corpora** retain their original licenses:
  - The Session (songs/tunes): CC0 1.0.
  - SymbTr: CC BY-NC-SA 4.0.
  - Essen Folksong Collection: distributed via EsAC, public-domain terms.

See `DATA_LICENSES.md` in the GitHub repo for full attribution.

## Usage

```bash
uv run python scripts/download_artifacts.py \
    --source local --path paper-folk-patterns-artifacts-v1.zip
bash reproduce.sh
```
