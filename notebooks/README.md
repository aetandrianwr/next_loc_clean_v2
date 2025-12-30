# Hyperparameter Tuning Evidence Notebook

## Overview

This directory contains the comprehensive Jupyter notebook documenting all hyperparameter tuning experiments for next location prediction models.

## File

**`hyperparameter_tuning_evidence.ipynb`** (71KB, 99 cells)

A complete, self-contained notebook providing evidence and verification of all hyperparameter tuning experiments conducted for baseline and proposed models.

## Contents

### 1. Executive Summary
- Overview of all models tested
- Best results summary table
- Key findings from hyperparameter tuning

### 2. Setup & Configuration
- Environment setup
- Utility functions for running experiments and displaying results

### 3. Pointer Network V45 (Proposed Model)
**GeoLife Dataset** (6 experiments, ordered by Acc@1):
1. baseline_d64_L2 - **54.00%** ✅ BEST
2. d64_L2_lowDropout - 52.14%
3. d64_L3_lowLR_highDrop - 51.77%
4. d80_L3_deeper - 51.37%
5. d64_L2_ff192_highLR - 50.26%
6. d72_L2 - 49.09%

**DIY Dataset** (5 experiments, ordered by Acc@1):
1. baseline_d128_L3 - **56.89%** ✅ BEST
2. d128_L3_lowerLR - 56.81%
3. d128_L3_highLR - 56.72%
4. d144_L3_largerEmb - 56.45%
5. d128_L4_deeper - 56.21%

### 4. MHSA Model (Transformer Baseline)
**GeoLife Dataset** (5 experiments, ordered by Acc@1):
1. emb128_layers2_ff128 - 33.18% (exceeds 500K limit)
2. emb128_layers1_ff128 - **32.95%** ✅ BEST (within limit)
3. emb128_layers1_ff128_lr0.002 - 32.78%
4. emb96_layers3_ff128 - 30.81%
5. baseline - 29.44%

**DIY Dataset** (4 experiments, ordered by Acc@1):
1. baseline - **53.17%** ✅ BEST
2. emb128_layers4_ff512 - 52.80%
3. emb96_layers4_ff384 - 52.76%
4. baseline_lr0.0005 - 52.57%

### 5. LSTM Model (Recurrent Baseline)
**GeoLife Dataset** (8 experiments, ordered by Acc@1):
1. dropout025_lr0018_v6 - **30.35%** ✅ BEST
2. dropout03_lr002_v4 - 30.01%
3. baseline_v1 - 29.93%
4. dropout022_lr002_v7 - 29.64%
5. dropout026_lr0019_v9 - 29.27%
6. emb64_hidden112_v2 - 29.18%
7. dropout035_lr0015_v5 - 28.93%
8. dropout028_lr0016_v8 - 28.73%

**DIY Dataset** (7 experiments, ordered by Acc@1):
1. emb80_lr0015_v2 - **51.99%** ✅ BEST
2. emb80_lr0013_v6 - 51.95%
3. emb80_lr0018_v5 - 51.94%
4. baseline_v1 - 51.68%
5. emb96_hidden208_v4 - 51.47%
6. emb80_batch512_v7 - 51.45%
7. emb80_L3_v3 - 49.42%

### 6. Markov Baseline (Statistical Model)
**No hyperparameter tuning** (deterministic model)

- markov1st (GeoLife): 27.64%
- markov1st (DIY): 50.60%
- markov_ori (GeoLife): 24.18%
- markov_ori (DIY): 44.13%

### 7. Comparative Analysis
- Summary tables for both datasets
- Performance visualizations
- Cross-model comparison
- Key insights and recommendations

### 8. Conclusion
- Performance hierarchy
- Best configurations identified
- Verification status
- Recommendations

## Usage

### Opening the Notebook

```bash
cd /data/next_loc_clean_v2/notebooks
jupyter notebook hyperparameter_tuning_evidence.ipynb
```

Or with JupyterLab:
```bash
jupyter lab hyperparameter_tuning_evidence.ipynb
```

### Using the Notebook

1. **View Results**: All cells display results from pre-run experiments
2. **Re-run Experiments**: Uncomment the `run_experiment()` calls in code cells
3. **Compare Models**: Navigate to Section 6 for comparative analysis
4. **Extract Insights**: See Section 6.4 for key findings

## Features

✅ **Comprehensive Coverage**: All 39 experiments documented
✅ **Organized Structure**: Experiments ordered by accuracy (highest first)
✅ **Evidence-Based**: References actual experiment directories
✅ **Reproducible**: Contains code to re-run any experiment
✅ **Interactive**: Display results, visualizations, and comparisons
✅ **Documentation Aligned**: Matches results in `/data/next_loc_clean_v2/docs/`

## Experiment Directory Structure

Each experiment referenced in the notebook follows this structure:
```
experiments/{dataset}_{model}_{timestamp}/
├── checkpoints/
│   └── best.pt (or checkpoint.pt)
├── config.yaml
├── config_original.yaml
├── training.log
├── val_results.json
└── test_results.json
```

## Results Summary

| Model | GeoLife Acc@1 | DIY Acc@1 | Notes |
|-------|---------------|-----------|-------|
| **Pointer V45** | **54.00%** | **56.89%** | Best overall |
| MHSA | 32.95% | 53.17% | Good baseline |
| LSTM | 30.35% | 51.99% | Competitive |
| Markov1st | 27.64% | 50.60% | Simple baseline |
| Markov_ori | 24.18% | 44.13% | Original implementation |

## Parameter Budgets

- **GeoLife**: ≤ 500K parameters
- **DIY**: ≤ 3M parameters

All best configurations respect these constraints.

## Key Insights

1. **Pointer V45** significantly outperforms all baselines
2. **Attention mechanisms** (Pointer V45, MHSA) beat recurrent models (LSTM)
3. **Hyperparameter tuning** provides 0.3-3% improvements
4. **Model depth**: Overfitting is a major concern; shallower often better
5. **Regularization**: Crucial for small datasets (GeoLife)

## Documentation Alignment

This notebook provides evidence for claims made in:
- `/data/next_loc_clean_v2/docs/hyperparameter_tuning_results.md`
- `/data/next_loc_clean_v2/docs/hyperparameter_tuning_MHSA.md`
- `/data/next_loc_clean_v2/docs/hyperparameter_tuning_LSTM.md`
- `/data/next_loc_clean_v2/docs/pointer_v45_model.md`
- `/data/next_loc_clean_v2/docs/MHSA_model.md`
- `/data/next_loc_clean_v2/docs/LSTM_model.md`
- `/data/next_loc_clean_v2/docs/markov1st_baseline.md`

All results in the notebook match the documentation.

## Version Info

- **Created**: December 28, 2025
- **Notebook Format**: 4.4
- **Total Cells**: 99 (55 markdown, 44 code)
- **File Size**: 71KB

---

For questions or issues, refer to the main project documentation or experiment logs.
