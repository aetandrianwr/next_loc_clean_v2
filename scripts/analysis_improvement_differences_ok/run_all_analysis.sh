#!/bin/bash
"""
Master script to run all analysis scripts in order.

Usage:
    cd /data/next_loc_clean_v2
    source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
    bash scripts/analysis_improvement_differences_ok/run_all_analysis.sh
"""

set -e  # Exit on error

echo "========================================"
echo "IMPROVEMENT DIFFERENCE ANALYSIS"
echo "========================================"
echo ""

cd /data/next_loc_clean_v2

# Create results directory
mkdir -p scripts/analysis_improvement_differences_ok/results

echo "[1/5] Running Dataset Characteristics Analysis..."
python scripts/analysis_improvement_differences_ok/01_dataset_characteristics.py
echo ""

echo "[2/5] Running MHSA Baseline Analysis..."
python scripts/analysis_improvement_differences_ok/02_mhsa_baseline_analysis.py
echo ""

echo "[3/5] Running Pointer Effectiveness Analysis..."
python scripts/analysis_improvement_differences_ok/03_pointer_effectiveness.py
echo ""

echo "[4/5] Running Root Cause Analysis..."
python scripts/analysis_improvement_differences_ok/04_root_cause_analysis.py
echo ""

echo "[5/5] Generating Visualizations..."
python scripts/analysis_improvement_differences_ok/05_visualizations.py
echo ""

echo "========================================"
echo "ANALYSIS COMPLETE"
echo "========================================"
echo "Results saved to: scripts/analysis_improvement_differences_ok/results/"
echo "Documentation: docs/analysis_improvement_differences.md"
