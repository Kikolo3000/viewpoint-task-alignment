# Viewpoint–task alignment — reproducibility package

This repository reproduces every table and figure in the paper **"Viewpoint–Task Alignment: When and Why Pattern-Derived Features Improve Symbolic Folk-Music Classification"** by Enrique Gutiérrez (*Transactions of the International Society for Music Information Retrieval*, accepted 2026), using pre-extracted features deposited on Zenodo ([10.5281/zenodo.21129773](https://doi.org/10.5281/zenodo.21129773)).

**Status**: analysis-only. Pattern mining and feature extraction (which require the `simpg_music` library) are out of scope. We ship the pre-mined breakdowns and extracted feature matrices so that classification, interpretability, statistics, and figures can be regenerated end-to-end without external editable installs.

---

## Quickstart

```bash
# 1. install dependencies (uv sync or pip install -e .)
uv sync

# 2. download + unpack artifacts from Zenodo
uv run python scripts/download_artifacts.py

# 3. regenerate everything
bash reproduce.sh
```

`reproduce.sh` runs all analysis scripts in order, then `verify_reproduction.py` compares the fresh outputs against `results_reference/` within a 1e-3 tolerance.

---

## What gets reproduced

| Script | Paper content |
|---|---|
| `01_run_classification.py` | All classification F1 scores per corpus × VP config (Table 4; Fig. 3; Supp. Fig. S2) |
| `02_run_ablation_statistics.py` | Corrected resampled *t*-test, Wilcoxon and Cohen's d significance, rank correlations, Levene's test (Fig. 4; Supp. Table S2, Fig. S3) |
| `03_run_vp_importance_comparison.py` | Permutation-importance rank shifts across VP configs (Fig. 6) |
| `04_run_feature_statistics.py` | Kruskal–Wallis per feature; per-class F1 improvements (Fig. 8) |
| `05_run_classifier_robustness.py` | Global → combined delta across classifier families (Fig. 9) |
| `07_run_essen_diagnostic.py` | Essen BoP isolation diagnostic (Supp. Fig. S5) |
| `08_run_dance9_comparison.py` | Dance-9 benchmark comparison vs published baselines (Table 5, §5.10) |
| `06_generate_paper_figures.py` | The 9 data figures as PDF+PNG (Figs. 3, 4, 6–9; Supp. Figs. S2, S3, S5) |

Approximate wall-clock on a 16-core laptop: 01 ≈ 25 min (dominated by Session/MRB), 02–04 ≈ 2 min each, 05 ≈ 10 min, 07 ≈ 10 min, 08 ≈ 5 min, 06 ≈ 30 s. Full `reproduce.sh` budget: about 1 hour.

---

## Layout

```
├── src/folk_patterns_repro/   # Importable analysis package (no simpg)
├── scripts/                   # Entry points (download + 01-08 + verify)
│   └── build_dance9/          # Dance-9 corpus reconstruction (4 stages)
├── configs/                   # Original mining configs (read-only reference)
├── artifacts/                 # Populated by download_artifacts.py
├── results/                   # Regenerated outputs (classification CSVs, figures)
├── results_reference/         # Known-good outputs for verification
├── reproduce.sh               # One-shot driver
└── pyproject.toml             # PyPI dependencies only
```

---

## Re-mining from raw scores (advanced)

Pattern mining and feature extraction depend on the `SimPG-Music` miner, which is not yet public; it is specified in full in the paper's Supplementary Material (Sec. S1), and its source will be released with the companion methods paper. The deposited `breakdowns/` are the raw mining output. The four source corpora (Essen, The Session, SymbTr, Dance-9) are available under their original licenses (see `DATA_LICENSES.md`). The Dance-9 corpus was reconstructed from John Chambers' Trillian ABC archive following Hillewaere et al. (2012, 2014); the full four-stage reconstruction pipeline is in `scripts/build_dance9/`.

---

## Citation

If you use these artifacts, please cite the paper and the Zenodo record. See `CITATION.cff`.
