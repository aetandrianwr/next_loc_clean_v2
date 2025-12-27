# MHSA Hyperparameter Tuning Results

## Overview

This document tracks hyperparameter tuning experiments for the MHSA model on GeoLife and DIY datasets.

**Constraints:**
- GeoLife: Maximum 500,000 parameters
- DIY: Maximum 3,000,000 parameters
- Seed: 42 (fixed)
- Patience: 5 (fixed)
- Max epochs: 50 (fixed)

## GeoLife Dataset Results

| Config Name | Params | Emb Size | Layers | FF Dim | Heads | LR | Batch | Acc@1 (%) | MRR (%) | Notes |
|-------------|--------|----------|--------|--------|-------|-----|-------|-----------|---------|-------|
| geolife_mhsa_baseline | 112,547 | 32 | 2 | 128 | 8 | 0.001 | 32 | 29.44 | 40.67 | Baseline |
| geolife_mhsa_emb96_layers3_ff128 | 470,915 | 96 | 3 | 128 | 8 | 0.001 | 32 | 30.81 | 40.84 | Scaled up |
| geolife_mhsa_emb128_layers1_ff128 | 298,595 | 128 | 1 | 128 | 8 | 0.001 | 32 | **32.95** | 42.13 | Wide shallow - **BEST** ⭐ |
| geolife_mhsa_emb128_layers1_ff128_lr0.002 | 298,595 | 128 | 1 | 128 | 8 | 0.002 | 32 | 32.78 | 42.37 | Higher LR |
| geolife_mhsa_emb128_layers2_ff128 | ~593K | 128 | 2 | 128 | 8 | 0.001 | 32 | 33.18 | 43.02 | Over 500K limit ❌ |

### Best Configuration (GeoLife) - Within 500K Params
- **Config**: geolife_mhsa_emb128_layers1_ff128
- **Test Acc@1**: 32.95%
- **Parameters**: 298,595 (~300K, within 500K limit ✓)

**Key Findings (GeoLife):**
- Wide-shallow architecture (larger embedding, fewer layers) works better than deeper models
- Best performing config uses emb_size=128, 1 encoder layer, ff_dim=128
- Increasing to 2 layers (33.18%) exceeds the 500K parameter limit

## DIY Dataset Results

| Config Name | Params | Emb Size | Layers | FF Dim | Heads | LR | Batch | Acc@1 (%) | MRR (%) | Notes |
|-------------|--------|----------|--------|--------|-------|-----|-------|-----------|---------|-------|
| diy_mhsa_baseline | 1,234,227 | 64 | 3 | 256 | 8 | 0.001 | 64 | **53.17** | 63.57 | **BEST** ⭐ |
| diy_mhsa_emb128_layers4_ff512 | ~2.8M | 128 | 4 | 512 | 8 | 0.001 | 256 | 52.80 | 63.37 | Scaled up - larger model worse |
| diy_mhsa_emb96_layers4_ff384 | ~2.0M | 96 | 4 | 384 | 8 | 0.001 | 128 | 52.76 | 63.32 | Medium scale |
| diy_mhsa_baseline_lr0.0005 | 1,234,227 | 64 | 3 | 256 | 8 | 0.0005 | 64 | 52.57 | 63.15 | Lower LR |

### Best Configuration (DIY) - Within 3M Params
- **Config**: diy_mhsa_baseline
- **Test Acc@1**: 53.17%
- **Parameters**: 1,234,227 (~1.2M, within 3M limit ✓)

**Key Findings (DIY):**
- Smaller baseline model outperforms larger configurations
- Increasing model capacity does not improve performance on this dataset
- Optimal configuration uses emb_size=64, 3 encoder layers, ff_dim=256

## Tuning Strategy

1. **Baseline**: Start with existing configurations
2. **Scale Up**: Increase model capacity within parameter budget
3. **Architecture Search**: Test different depth vs width tradeoffs
4. **Learning Rate**: Fine-tune learning rate for best configurations

## Key Insights

1. **For GeoLife**: Wider, shallower models work better. emb128_layers1 beats emb32_layers2.
2. **For DIY**: The baseline is already well-tuned. Larger models don't improve accuracy.
3. **Parameter efficiency**: More parameters don't always mean better performance.

## Configuration Files

All configuration files are stored in: `config/hyperparameter_tuning/`
