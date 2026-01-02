#!/bin/bash
# Run all LSTM and RNN baseline experiments
# 
# This script trains all baseline models and compares them with Pointer V45.
# 
# Usage:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
#   bash scripts/baseline_lstm_rnn/run_all_baselines.sh
#
# Expected Results:
#   Geolife:
#     - Pointer V45: ~53.94% Acc@1
#     - LSTM: ~33.01% Acc@1
#     - RNN: ~32.95% Acc@1
#
#   DIY:
#     - Pointer V45: ~56.88% Acc@1
#     - LSTM: ~52-53% Acc@1
#     - RNN: ~52-53% Acc@1

set -e

echo "=============================================="
echo "LSTM & RNN Baseline Training Script"
echo "=============================================="
echo ""

# Change to project root
cd "$(dirname "$0")/../.."

echo "[1/6] Training LSTM on Geolife..."
python scripts/baseline_lstm_rnn/train_baseline.py \
    --config scripts/baseline_lstm_rnn/config_lstm_geolife.yaml

echo ""
echo "[2/6] Training RNN on Geolife..."
python scripts/baseline_lstm_rnn/train_baseline.py \
    --config scripts/baseline_lstm_rnn/config_rnn_geolife.yaml

echo ""
echo "[3/6] Training LSTM on DIY..."
python scripts/baseline_lstm_rnn/train_baseline.py \
    --config scripts/baseline_lstm_rnn/config_lstm_diy.yaml

echo ""
echo "[4/6] Training RNN on DIY..."
python scripts/baseline_lstm_rnn/train_baseline.py \
    --config scripts/baseline_lstm_rnn/config_rnn_diy.yaml

echo ""
echo "[5/6] Training Pointer V45 on Geolife..."
python src/training/train_pointer_v45.py \
    --config config/models/config_pointer_v45_geolife.yaml

echo ""
echo "[6/6] Training Pointer V45 on DIY..."
python src/training/train_pointer_v45.py \
    --config config/models/config_pointer_v45_diy.yaml

echo ""
echo "=============================================="
echo "All experiments completed!"
echo "Results saved in experiments/ directory"
echo "=============================================="
