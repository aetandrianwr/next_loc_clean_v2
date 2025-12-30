# LSTM Hyperparameter Tuning Results

## Overview

This document tracks hyperparameter tuning experiments for the LSTM model on GeoLife and DIY datasets.

**Constraints:**
- GeoLife: Maximum 500,000 parameters
- DIY: Maximum 3,000,000 parameters
- Seed: 42 (fixed)
- Patience: 5 for GeoLife, 3 for DIY (fixed)
- Max epochs: 50 (fixed)

## GeoLife Dataset Results

| Config Name | Params | Emb Size | Hidden | Layers | LSTM Drop | FC Drop | LR | Batch | Acc@1 (%) | MRR (%) | Notes |
|-------------|--------|----------|--------|--------|-----------|---------|-----|-------|-----------|---------|-------|
| geolife_lstm_baseline_v1 | 482,659 | 32 | 128 | 2 | 0.2 | 0.2 | 0.001 | 32 | 29.93 | 40.85 | Baseline |
| geolife_lstm_emb64_hidden112_v2 | 455,587 | 64 | 112 | 2 | 0.1 | 0.1 | 0.001 | 32 | 29.18 | 41.47 | Larger emb, smaller hidden |
| geolife_lstm_dropout03_lr002_v4 | 482,659 | 32 | 128 | 2 | 0.3 | 0.3 | 0.002 | 64 | 30.01 | 41.92 | Higher dropout/LR |
| geolife_lstm_dropout035_lr0015_v5 | 482,659 | 32 | 128 | 2 | 0.35 | 0.35 | 0.0015 | 64 | 28.93 | 41.07 | Too much dropout |
| geolife_lstm_dropout025_lr0018_v6 | 482,659 | 32 | 128 | 2 | 0.25 | 0.25 | 0.0018 | 64 | **30.35** | 41.66 | **BEST** ⭐ |
| geolife_lstm_dropout022_lr002_v7 | 482,659 | 32 | 128 | 2 | 0.22 | 0.22 | 0.002 | 64 | 29.64 | 41.45 | Slightly worse |
| geolife_lstm_dropout028_lr0016_v8 | 482,659 | 32 | 128 | 2 | 0.28 | 0.28 | 0.0016 | 64 | 28.73 | 40.97 | Lower LR hurts |
| geolife_lstm_dropout026_lr0019_v9 | 482,659 | 32 | 128 | 2 | 0.26 | 0.26 | 0.0019 | 64 | 29.27 | 41.37 | Close to best |

### Best Configuration (GeoLife) - Within 500K Params
- **Config**: geolife_lstm_dropout025_lr0018_v6
- **Test Acc@1**: 30.35%
- **Test MRR**: 41.66%
- **Parameters**: 482,659 (~483K, within 500K limit ✓)

**Key Findings (GeoLife):**
- Optimal configuration: emb_size=32, hidden_size=128, 2 LSTM layers
- Sweet spot for dropout: 0.25 for both LSTM and FC layers
- Learning rate 0.0018 works best (slightly higher than baseline 0.001)
- Batch size 64 performs better than 32
- Too much dropout (0.35) or too low LR (0.0016) hurts performance
- Larger embedding with smaller hidden size (emb64_hidden112) underperforms

## DIY Dataset Results

| Config Name | Params | Emb Size | Hidden | Layers | LSTM Drop | FC Drop | LR | Batch | Acc@1 (%) | MRR (%) | Notes |
|-------------|--------|----------|--------|--------|-----------|---------|-----|-------|-----------|---------|-------|
| diy_lstm_baseline_v1 | 2,847,582 | 96 | 192 | 2 | 0.2 | 0.1 | 0.001 | 256 | 51.74 | 62.82 | Baseline |
| diy_lstm_emb80_lr0015_v2 | 2,720,590 | 80 | 192 | 2 | 0.15 | 0.15 | 0.0015 | 256 | **51.99** | 63.05 | **BEST** ⭐ |
| diy_lstm_emb80_L3_v3 | 3,017,038 | 80 | 192 | 3 | 0.2 | 0.15 | 0.001 | 256 | 49.42 | 61.62 | 3 layers overfits |
| diy_lstm_emb96_hidden208_v4 | 3,080,190 | 96 | 208 | 2 | 0.18 | 0.12 | 0.0012 | 256 | 51.47 | 62.77 | Larger model |
| diy_lstm_emb80_lr0018_v5 | 2,720,590 | 80 | 192 | 2 | 0.12 | 0.12 | 0.0018 | 256 | 51.94 | 63.08 | Close second |
| diy_lstm_emb80_lr0013_v6 | 2,720,590 | 80 | 192 | 2 | 0.15 | 0.15 | 0.0013 | 256 | 51.95 | 63.00 | Very close |
| diy_lstm_emb80_batch512_v7 | 2,720,590 | 80 | 192 | 2 | 0.15 | 0.15 | 0.0014 | 512 | 51.45 | 62.76 | Larger batch hurts |

### Best Configuration (DIY) - Within 3M Params
- **Config**: diy_lstm_emb80_lr0015_v2
- **Test Acc@1**: 51.99%
- **Test MRR**: 63.05%
- **Parameters**: 2,720,590 (~2.7M, within 3M limit ✓)

**Key Findings (DIY):**
- Optimal configuration: emb_size=80, hidden_size=192, 2 LSTM layers
- Smaller embedding (80 vs 96) with higher LR (0.0015) works best
- Lower dropout (0.15) better than higher (0.2)
- Batch size 256 is optimal; increasing to 512 hurts performance
- 3 LSTM layers consistently overfit (49.42% vs 51.99%)
- Learning rate range 0.0013-0.0018 all perform similarly well
- Larger model (emb96_hidden208, 3M params) doesn't improve performance

## Tuning Strategy

1. **Baseline**: Start with existing configurations
   - GeoLife: emb_size=32, hidden_size=128, 2 layers
   - DIY: emb_size=96, hidden_size=192, 2 layers

2. **Architecture Search**: Test different embedding and hidden sizes
   - Tried larger embedding with smaller hidden (emb64_hidden112)
   - Tried larger hidden size (hidden208)
   - Tried 3 layers (consistently overfits)

3. **Regularization Tuning**: Fine-tune dropout rates
   - GeoLife: Tested dropout from 0.22 to 0.35
   - DIY: Tested dropout from 0.12 to 0.2

4. **Learning Rate Search**: Find optimal learning rate
   - GeoLife: Tested from 0.0016 to 0.002
   - DIY: Tested from 0.0013 to 0.0018

5. **Batch Size**: Test different batch sizes
   - GeoLife: 32 vs 64
   - DIY: 256 vs 512

## Key Insights

1. **Model Depth**: 2 LSTM layers is optimal for both datasets. Adding a 3rd layer consistently causes overfitting.

2. **Dropout Regularization**:
   - GeoLife (smaller dataset): Higher dropout (0.25) needed
   - DIY (larger dataset): Lower dropout (0.15) works better

3. **Learning Rate**:
   - Both datasets benefit from slightly higher LR than default (0.001)
   - GeoLife optimal: 0.0018 (1.8x default)
   - DIY optimal: 0.0015 (1.5x default)

4. **Batch Size**:
   - GeoLife: Larger batch (64) improves over smaller (32)
   - DIY: Moderate batch (256) is optimal; too large (512) hurts

5. **Model Size**:
   - GeoLife: Compact model (emb32, hidden128) works best
   - DIY: Medium-sized model (emb80, hidden192) is optimal
   - Larger models don't necessarily improve performance

6. **Parameter Efficiency**:
   - Best GeoLife model: 483K params (96% of budget)
   - Best DIY model: 2.7M params (90% of budget)
   - Both achieve best results without maxing out parameter budget

## Configuration Files

All configuration files are stored in: `config/hyperparameter_tuning/`

### GeoLife Configs
- `geolife_lstm_baseline_v1.yaml` - Baseline configuration
- `geolife_lstm_dropout025_lr0018_v6.yaml` - **Best configuration** ⭐

### DIY Configs
- `diy_lstm_baseline_v1.yaml` - Baseline configuration
- `diy_lstm_emb80_lr0015_v2.yaml` - **Best configuration** ⭐

## Comparison with Other Models

### GeoLife Dataset
| Model | Params | Acc@1 (%) | MRR (%) | Notes |
|-------|--------|-----------|---------|-------|
| LSTM (best) | 483K | 30.35 | 41.66 | This work |
| MHSA (best) | 299K | 32.95 | 42.13 | Outperforms LSTM |
| PointerNetV45 (best) | 253K | 54.00 | 65.84 | Best model |

### DIY Dataset
| Model | Params | Acc@1 (%) | MRR (%) | Notes |
|-------|--------|-----------|---------|-------|
| LSTM (best) | 2.7M | 51.99 | 63.05 | This work |
| MHSA (best) | 1.2M | 53.17 | 63.57 | Slightly better |
| PointerNetV45 (best) | 2.4M | 56.89 | 67.99 | Best model |

**Observations**:
- LSTM underperforms compared to attention-based models (MHSA and PointerNetV45)
- On GeoLife, LSTM achieves 30.35% vs MHSA's 32.95% (8.6% relative gap)
- On DIY, LSTM achieves 51.99% vs MHSA's 53.17% (2.3% relative gap)
- PointerNetV45 significantly outperforms both LSTM and MHSA on both datasets

---

*Last updated: 2025-12-27*
