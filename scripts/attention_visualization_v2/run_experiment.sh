#!/bin/bash
# =============================================================================
# Attention Visualization Experiment - Full Pipeline
# =============================================================================
# This script runs the complete attention visualization experiment for both
# DIY and GeoLife datasets with seed=42.
#
# Usage:
#   chmod +x run_experiment.sh
#   ./run_experiment.sh
#
# Output:
#   - results/diy/          : DIY dataset attention analysis
#   - results/geolife/      : GeoLife dataset attention analysis
#   - results/              : Cross-dataset comparison
#
# Author: PhD Thesis Experiment
# Date: 2026
# =============================================================================

set -e  # Exit on error

# Configuration
SEED=42
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "============================================================"
echo "ATTENTION VISUALIZATION EXPERIMENT - FULL PIPELINE"
echo "============================================================"
echo "Script Directory: $SCRIPT_DIR"
echo "Project Root: $PROJECT_ROOT"
echo "Seed: $SEED"
echo "============================================================"

# Activate conda environment
echo ""
echo "[Step 1/4] Activating conda environment..."
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Run DIY dataset experiment
echo ""
echo "[Step 2/4] Running DIY dataset attention analysis..."
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/run_attention_experiment.py" --dataset diy --seed $SEED

# Run GeoLife dataset experiment
echo ""
echo "[Step 3/4] Running GeoLife dataset attention analysis..."
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/run_attention_experiment.py" --dataset geolife --seed $SEED

# Run cross-dataset comparison
echo ""
echo "[Step 4/4] Running cross-dataset comparison..."
echo "------------------------------------------------------------"
python "$SCRIPT_DIR/cross_dataset_comparison.py"

# Summary
echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results saved to:"
echo "  - $SCRIPT_DIR/results/diy/"
echo "  - $SCRIPT_DIR/results/geolife/"
echo "  - $SCRIPT_DIR/results/ (cross-dataset comparison)"
echo ""
echo "Generated outputs:"
echo "  - Attention visualizations (PNG, PDF, SVG)"
echo "  - Statistical tables (CSV, LaTeX)"
echo "  - Experiment metadata (JSON)"
echo "============================================================"
