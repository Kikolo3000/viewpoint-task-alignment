# Reproduction guide

This document maps every table and figure in the paper to the script that regenerates it and the CSV/PDF it writes to.

## 0. Prerequisites

```bash
uv sync                                   # Python ≥ 3.11, all PyPI deps
uv run python scripts/download_artifacts.py   # fetches + seeds artifacts
```

Result tree after seeding:
- `artifacts/features/{corpus}_{vp}_{group}.parquet` — 88 files (includes dance9_B_*)
- `artifacts/labels/{corpus}_labels.csv` — 4 files (includes dance9_labels.csv)
- `artifacts/breakdowns/{corpus}_{vp}.pkl` — 22 files (optional, require `simpg_music` to load)
- `results/{corpus}/(vp)/interpretability/importance_combined.csv` — seeded from archive

The standard reproduction (steps 1–9 below) reads the pre-computed feature parquets and
needs **no** `simpg_music` install. Only re-mining patterns from scratch — or deserialising
the optional breakdown pickles — requires the `SimPG-Music` miner, which is not yet public
(its methods paper is under review); the exact version used for the paper is recorded by
commit hash `82857bd` and its source will be released with that paper.

## 1. Classification (Tables 3, 4; Fig 1, 2)

```bash
uv run python scripts/01_run_classification.py
```

Writes, per `(corpus, vp)`:
- `results/{corpus}/{vp}/classification/summary.csv` — F1 per (feature_config, classifier)
- `results/{corpus}/{vp}/classification/fold_details.csv` — per-fold F1
- `results/{corpus}/{vp}/classification/statistical_tests.csv` — Wilcoxon + corrected t

Runtime: ~25 min total (dominated by Session).

## 2. Viewpoint ablation statistics (Table 5; Fig 3, 4)

```bash
uv run python scripts/02_run_ablation_statistics.py
```

Writes, under `results/viewpoint_comparison/`:
- `ablation_significance.csv` — global→combined Δ, Wilcoxon p, Cohen's d, Bonferroni
- `rank_correlations_ablation.csv` — Spearman ρ across corpus pairs + bootstrap CIs
- `variance_comparison.csv` — Levene's test on per-VP F1 variance

## 3. VP importance comparison (Fig 5)

```bash
uv run python scripts/03_run_vp_importance_comparison.py
```

Writes `results/viewpoint_comparison/importance_shift_{corpus}.csv` for each of the three corpora, derived from the pre-computed permutation-importance CSVs seeded in step 0.

## 4. Feature statistics + per-class F1 (Table 6; Fig 7)

```bash
uv run python scripts/04_run_feature_statistics.py
```

Writes:
- `results/{corpus}/feature_statistics.csv` — Kruskal–Wallis per feature
- `results/viewpoint_comparison/per_class_improvement_{corpus}.csv`

## 5. Classifier-agnostic robustness (Fig 8)

```bash
uv run python scripts/05_run_classifier_robustness.py
```

Writes:
- `results/{corpus}/classifier_robustness/summary.csv`
- `results/viewpoint_comparison/classifier_robustness.csv`

SVC is skipped above 5k pieces; GBM above 10k.

## 6. Essen BoP diagnostic (Fig 9)

```bash
uv run python scripts/07_run_essen_diagnostic.py
```

Writes `results/essen/essen_diagnostic/{summary,fold_details,comparison}.csv`.

## 7. Dance-9 benchmark comparison (Table 4)

```bash
uv run python scripts/08_run_dance9_comparison.py
```

Reproduces Table 4 (§5.7): five classifiers on the Dance-9 corpus (VP=B, combined features),
positioned against seven published baselines (Hillewaere 2012, 2014; Conklin 2013).

Writes:
- `results/dance9/dance9_comparison/summary.csv` — per-classifier accuracy + F1_macro
- `results/dance9/dance9_comparison/fold_details.csv` — per-fold breakdown
- `results/dance9/dance9_comparison/positioning.csv` — ours vs published baselines

Runtime: ~5 min.

## 8. Figures (1–9)

```bash
uv run python scripts/06_generate_paper_figures.py
```

Writes `results/figures/fig{1..9}_*.{pdf,png}`.

## 9. Verification

```bash
uv run python scripts/verify_reproduction.py
```

Compares every tracked CSV in `results/` against `results_reference/` within a 1e-3 numerical tolerance. Exits non-zero on mismatch.

> **Note on significance tests.** `02_run_ablation_statistics.py` reports the Nadeau–Bengio
> *corrected resampled t-test* (with Bonferroni) as the primary statistic in
> `ablation_significance.csv` (columns `corrected_t_p`, `corrected_t_bonferroni_p`,
> `significant_bonferroni`); the Wilcoxon test is retained as a secondary descriptor. These
> match the manuscript and `results/stats_corrected/ablation_corrected.csv` exactly (Essen 0/7
> significant; SymbTr 5/7; The Session 6/7).

---

## One-shot equivalent

```bash
bash reproduce.sh
```

Equivalent to steps 1–9 run in sequence.
