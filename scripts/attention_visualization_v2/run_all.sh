#!/bin/bash
# =============================================================================
# Attention Visualization Experiment - Quick Run Script
# =============================================================================
# Simplified script to run the complete experiment pipeline.
#
# Usage:
#   bash run_all.sh
#
# Author: PhD Thesis Experiment
# Date: 2026
# =============================================================================

# Activate conda and run
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

cd /data/next_loc_clean_v2

echo "Running DIY dataset..."
python scripts/attention_visualization_v2/run_attention_experiment.py --dataset diy --seed 42

echo ""
echo "Running GeoLife dataset..."
python scripts/attention_visualization_v2/run_attention_experiment.py --dataset geolife --seed 42

echo ""
echo "Running cross-dataset comparison..."
python scripts/attention_visualization_v2/cross_dataset_comparison.py

echo ""
echo "Done! Results in scripts/attention_visualization_v2/results/"
