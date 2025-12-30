#!/bin/bash
"""
Runner script for all performance gap analysis scripts.
Run with: bash run_all_analysis.sh
"""

# Set up conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Change to project directory
cd /data/next_loc_clean_v2

# Create results directory
mkdir -p scripts/analysis_performance_gap_differences/results

echo "========================================"
echo "Running Performance Gap Analysis"
echo "========================================"
echo ""

echo "[1/6] Running Dataset Statistics Analysis..."
python scripts/analysis_performance_gap_differences/01_dataset_statistics.py
echo ""

echo "[2/6] Running Sequence Patterns Analysis..."
python scripts/analysis_performance_gap_differences/02_sequence_patterns.py
echo ""

echo "[3/6] Running Location Frequency Analysis..."
python scripts/analysis_performance_gap_differences/03_location_frequency.py
echo ""

echo "[4/6] Running User Behavior Analysis..."
python scripts/analysis_performance_gap_differences/04_user_behavior.py
echo ""

echo "[5/6] Running Model Mechanism Analysis..."
python scripts/analysis_performance_gap_differences/05_model_mechanism.py
echo ""

echo "[6/6] Running Comprehensive Analysis..."
python scripts/analysis_performance_gap_differences/06_comprehensive_analysis.py
echo ""

echo "========================================"
echo "All Analysis Complete!"
echo "Results saved to: scripts/analysis_performance_gap_differences/results/"
echo "========================================"
