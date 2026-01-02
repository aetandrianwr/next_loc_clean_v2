#!/bin/bash
#
# Sequence Length Days Experiment - Master Run Script
#
# This script runs the complete sequence length experiment including:
# 1. Evaluation on DIY and GeoLife datasets
# 2. Generation of visualizations and analysis
#
# Usage:
#   ./run_experiment.sh
#
# Requirements:
#   - Conda environment 'mlenv' with PyTorch, matplotlib, seaborn, pandas, numpy
#   - Pre-trained checkpoints in experiments/ directory
#   - Test data in data/ directory
#
# Author: PhD Research Team
# Date: January 2026

set -e

echo "=========================================="
echo "SEQUENCE LENGTH EXPERIMENT"
echo "=========================================="
echo "Starting: $(date)"
echo ""

# Activate conda environment
echo "[1/4] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Navigate to project root
cd /data/next_loc_clean_v2

# Create results directory if not exists
mkdir -p scripts/experiment_sequence_len_days/results

# Run evaluation experiments
echo ""
echo "[2/4] Running evaluation experiments..."
echo "----------------------------------------"
python scripts/experiment_sequence_len_days/evaluate_sequence_length.py --dataset all

# Generate visualizations
echo ""
echo "[3/4] Generating visualizations and analysis..."
echo "----------------------------------------"
python scripts/experiment_sequence_len_days/visualize_results.py

# Display summary
echo ""
echo "[4/4] Experiment Complete!"
echo "=========================================="
echo "Results saved to: scripts/experiment_sequence_len_days/results/"
echo ""
echo "Output Files:"
ls -1 scripts/experiment_sequence_len_days/results/
echo ""
echo "Documentation: docs/experiment_sequence_length_days.md"
echo "=========================================="
echo "Finished: $(date)"
