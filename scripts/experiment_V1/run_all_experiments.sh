#!/bin/bash
# Master script to run all experiments for Pointer V45 model analysis
# Usage: bash run_all_experiments.sh

set -e

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "Running All Pointer V45 Experiments"
echo "============================================================"
echo "Start time: $(date)"
echo ""

# Experiment 1: Sequence Length Analysis
echo "============================================================"
echo "Experiment 1: Sequence Length Analysis"
echo "============================================================"
python exp1_sequence_length/run_experiment.py
echo ""

# Experiment 2: Time-of-Day Analysis
echo "============================================================"
echo "Experiment 2: Time-of-Day Analysis"
echo "============================================================"
python exp2_time_of_day/run_experiment.py
echo ""

# Experiment 3: Weekday vs Weekend Analysis
echo "============================================================"
echo "Experiment 3: Weekday vs Weekend Analysis"
echo "============================================================"
python exp3_weekday_weekend/run_experiment.py
echo ""

# Experiment 4: User Activity Level Analysis
echo "============================================================"
echo "Experiment 4: User Activity Level Analysis"
echo "============================================================"
python exp4_user_activity/run_experiment.py
echo ""

# Experiment 5: Location Frequency Analysis
echo "============================================================"
echo "Experiment 5: Location Frequency Analysis"
echo "============================================================"
python exp5_location_frequency/run_experiment.py
echo ""

# Experiment 6: Pointer-Generator Gate Analysis
echo "============================================================"
echo "Experiment 6: Pointer-Generator Gate Analysis"
echo "============================================================"
python exp6_pointer_gate/run_experiment.py
echo ""

# Experiment 7: Recency Analysis
echo "============================================================"
echo "Experiment 7: Recency Analysis"
echo "============================================================"
python exp7_recency/run_experiment.py
echo ""

# Experiment 8: Cross-Dataset Comparison
echo "============================================================"
echo "Experiment 8: Cross-Dataset Comparison"
echo "============================================================"
python exp8_cross_dataset/run_experiment.py
echo ""

echo "============================================================"
echo "All Experiments Complete!"
echo "End time: $(date)"
echo "============================================================"
