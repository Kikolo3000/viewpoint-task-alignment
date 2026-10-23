#!/usr/bin/env bash
# End-to-end reproduction driver. Expects artifacts already downloaded.
set -euo pipefail
cd "$(dirname "$0")"

echo "[1/8] Classification"; uv run python scripts/01_run_classification.py
echo "[2/8] Ablation statistics"; uv run python scripts/02_run_ablation_statistics.py
echo "[3/8] VP importance comparison"; uv run python scripts/03_run_vp_importance_comparison.py
echo "[4/8] Feature statistics"; uv run python scripts/04_run_feature_statistics.py
echo "[5/8] Classifier robustness"; uv run python scripts/05_run_classifier_robustness.py
echo "[6/8] Essen diagnostic"; uv run python scripts/07_run_essen_diagnostic.py
echo "[7/8] Dance-9 comparison"; uv run python scripts/08_run_dance9_comparison.py
echo "[8/8] Figures"; uv run python scripts/06_generate_paper_figures.py

echo "--- Verifying reproduction ---"
uv run python scripts/verify_reproduction.py
