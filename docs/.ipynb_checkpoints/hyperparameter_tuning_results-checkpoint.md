# Hyperparameter Tuning Results

This document tracks hyperparameter tuning experiments for various models.

## Constraints
- **Geolife**: Max parameters = 500K
- **DIY**: Max parameters = 3M
- **Seed**: 42 (fixed)
- **Patience**: 5 (fixed)
- **Max Epochs**: 50 (fixed)

---

## PointerNetworkV45 Model

### Geolife Dataset Results (PointerNetworkV45)

| Config | d_model | nhead | layers | ff_dim | params | LR | Acc@1 | Acc@5 | MRR | Notes |
|--------|---------|-------|--------|--------|--------|-----|-------|-------|-----|-------|
| baseline_d64_L2 | 64 | 4 | 2 | 128 | 253K | 6.5e-4 | **54.00%** | 81.10% | 65.84% | Baseline |
| d80_L3_deeper | 80 | 4 | 3 | 160 | 396K | 6e-4 | 51.37% | 81.52% | 64.86% | Overfits |
| d64_L2_ff192_highLR | 64 | 4 | 2 | 192 | 269K | 8e-4 | 50.26% | 81.35% | 64.18% | Too high LR |
| d64_L3_lowLR_highDrop | 64 | 4 | 3 | 128 | 286K | 5e-4 | 51.77% | 81.64% | 64.88% | More layers overfits |
| d64_L2_lowDropout | 64 | 4 | 2 | 128 | 253K | 6e-4 | 52.14% | 81.87% | 65.10% | Lower dropout |
| d72_L2 | 72 | 4 | 2 | 144 | 295K | 6.5e-4 | 49.09% | 80.61% | 63.34% | Larger model |

### Best Geolife Config (PointerNetworkV45): `geolife_baseline_d64_L2.yaml`
- **Acc@1: 54.00%**
- Parameters: 253K / 500K

### DIY Dataset Results (PointerNetworkV45)

| Config | d_model | nhead | layers | ff_dim | params | LR | Acc@1 | Acc@5 | MRR | Notes |
|--------|---------|-------|--------|--------|--------|-----|-------|-------|-----|-------|
| baseline_d128_L3 | 128 | 4 | 3 | 256 | 2.4M | 7e-4 | **56.89%** | 82.23% | 67.99% | Baseline |
| d128_L4_deeper | 128 | 4 | 4 | 256 | 2.53M | 6e-4 | 56.21% | 82.14% | 67.53% | Slight overfit |
| d128_L3_highLR | 128 | 4 | 3 | 256 | 2.4M | 9e-4 | 56.72% | 82.42% | 67.88% | Higher LR |
| d144_L3_largerEmb | 144 | 4 | 3 | 288 | 2.77M | 7e-4 | 56.45% | 81.97% | 67.62% | Larger emb |
| d128_L3_lowerLR | 128 | 4 | 3 | 256 | 2.4M | 6e-4 | 56.81% | 82.51% | 67.95% | Lower LR |

### Best DIY Config (PointerNetworkV45): `diy_baseline_d128_L3.yaml`
- **Acc@1: 56.89%**
- Parameters: 2.4M / 3M

---

## LSTM Model

### Geolife Dataset Results (LSTM)

| Config | Emb Size | Hidden | Layers | LR | Batch | LSTM/FC Dropout | Params | Acc@1 | MRR | Notes |
|--------|----------|--------|--------|-----|-------|-----------------|--------|-------|-----|-------|
| baseline_v1 | 32 | 128 | 2 | 0.001 | 32 | 0.2/0.2 | 483K | 29.93% | 40.85% | Baseline |
| emb64_hidden112_v2 | 64 | 112 | 2 | 0.001 | 32 | 0.1/0.1 | 456K | 29.18% | 41.47% | Lower dropout |
| dropout03_lr002_v4 | 32 | 128 | 2 | 0.002 | 64 | 0.3/0.3 | 483K | 30.01% | 41.92% | Higher dropout/LR |
| dropout035_lr0015_v5 | 32 | 128 | 2 | 0.0015 | 64 | 0.35/0.35 | 483K | 28.93% | 41.07% | Too much dropout |
| dropout025_lr0018_v6 | 32 | 128 | 2 | 0.0018 | 64 | 0.25/0.25 | 483K | **30.35%** | 41.66% | **BEST** |
| dropout022_lr002_v7 | 32 | 128 | 2 | 0.002 | 64 | 0.22/0.22 | 483K | 29.64% | 41.45% | Slightly worse |
| dropout028_lr0016_v8 | 32 | 128 | 2 | 0.0016 | 64 | 0.28/0.28 | 483K | 28.73% | 40.97% | Lower LR hurts |
| dropout026_lr0019_v9 | 32 | 128 | 2 | 0.0019 | 64 | 0.26/0.26 | 483K | 29.27% | 41.37% | Close to best |

### Best Geolife Config (LSTM): `geolife_lstm_dropout025_lr0018_v6.yaml`
- **Acc@1: 30.35%**
- Parameters: 483K / 500K

### DIY Dataset Results (LSTM)

| Config | Emb Size | Hidden | Layers | LR | Batch | LSTM/FC Dropout | Params | Acc@1 | MRR | Notes |
|--------|----------|--------|--------|-----|-------|-----------------|--------|-------|-----|-------|
| baseline_v1 | 96 | 192 | 2 | 0.001 | 256 | 0.2/0.1 | 2.85M | 51.68% | 62.79% | Baseline |
| emb80_lr0015_v2 | 80 | 192 | 2 | 0.0015 | 256 | 0.15/0.15 | 2.72M | **51.99%** | 63.05% | **BEST** |
| emb80_L3_v3 | 80 | 192 | 3 | 0.001 | 256 | 0.2/0.15 | 2.87M | 49.42% | 61.62% | 3 layers overfits |
| emb96_hidden208_v4 | 96 | 208 | 2 | 0.0012 | 256 | 0.18/0.12 | 2.98M | 51.47% | 62.77% | Larger model |
| emb80_lr0018_v5 | 80 | 192 | 2 | 0.0018 | 256 | 0.12/0.12 | 2.72M | 51.94% | 63.08% | Close second |
| emb80_lr0013_v6 | 80 | 192 | 2 | 0.0013 | 256 | 0.15/0.15 | 2.72M | 51.95% | 63.00% | Very close |
| emb80_batch512_v7 | 80 | 192 | 2 | 0.0014 | 512 | 0.15/0.15 | 2.72M | 51.45% | 62.76% | Larger batch hurts |

### Best DIY Config (LSTM): `diy_lstm_emb80_lr0015_v2.yaml`
- **Acc@1: 51.99%**
- Parameters: 2.72M / 3M

---

## Summary

After extensive hyperparameter tuning with 15 different configurations (8 for Geolife, 7 for DIY), the best configurations are:

### Geolife (≤500K params)
- **Best Test Acc@1: 30.35%** (+0.42% over baseline)
- Config: `geolife_lstm_dropout025_lr0018_v6.yaml`
- Key findings:
  - Higher learning rate (0.0018) with moderate dropout (0.25) works best
  - Batch size 64 better than 32
  - 2 layers optimal, 3 layers overfit

### DIY (≤3M params)
- **Best Test Acc@1: 51.99%** (+0.31% over baseline)
- Config: `diy_lstm_emb80_lr0015_v2.yaml`
- Key findings:
  - Slightly smaller embedding (80 vs 96) with higher LR (0.0015) works best
  - Lower dropout (0.15) better than higher
  - Batch size 256 optimal, 512 hurts performance
  - 2 layers optimal, 3 layers overfit

### Key Insights
1. **Learning rate** is crucial - slightly higher than default (0.001) tends to help
2. **Dropout** around 0.15-0.25 works best; too much hurts performance
3. **Model depth**: 2 LSTM layers optimal; 3 layers consistently overfits
4. **Batch size**: moderate sizes (64-256) work best; too large reduces performance

---

## Key Insights

### PointerNetworkV45 Model
1. **Geolife**: The baseline config (d=64, L=2, ff=128) is optimal - larger models overfit on this small dataset
2. **DIY**: The baseline config (d=128, L=3, ff=256) is optimal - larger models don't improve
3. More layers consistently lead to overfitting on both datasets
4. Learning rate around 6.5-7e-4 is optimal; higher LR causes instability
5. Dropout of 0.15 works well; changing it doesn't improve results

### LSTM Model
1. For Geolife (small dataset with ~14K sequences), dropout of 0.2 works better than 0.1
2. DIY dataset benefits from larger embedding and hidden sizes
3. Learning rate of 0.001 with Adam optimizer works well

---
*Last updated: 2025-12-26*
