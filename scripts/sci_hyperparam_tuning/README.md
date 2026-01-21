# Scientific Hyperparameter Tuning

This directory contains all scripts, configurations, and results for the scientific hyperparameter tuning experiments conducted on three models (Pointer Generator Transformer, MHSA, LSTM) across two datasets (Geolife and DIY).

## Overview

The hyperparameter tuning follows a rigorous scientific methodology based on best practices in machine learning research (Bergstra & Bengio, 2012):

1. **Random Search**: 20 trials per model-dataset combination
2. **Fixed Seed**: Reproducibility with seed=42
3. **Early Stopping**: Patience=5 for efficient training
4. **Objective**: Val Acc@1 (primary metric for model selection)
5. **Fair Comparison**: Same search budget and methodology for all models

## Directory Structure

```
sci_hyperparam_tuning/
├── README.md                           # This file
├── hyperparam_search_space.py          # Search space definitions
├── generate_configs.py                 # Config generation script
├── run_hyperparam_tuning.py            # Parallel tuning manager
├── run_final_evaluation.py             # Final evaluation script
├── configs/                            # Generated configurations (120 files)
│   ├── pointer_v45_geolife_trial*.yaml
│   ├── pointer_v45_diy_trial*.yaml
│   ├── mhsa_geolife_trial*.yaml
│   ├── mhsa_diy_trial*.yaml
│   ├── lstm_geolife_trial*.yaml
│   ├── lstm_diy_trial*.yaml
│   └── all_configs_summary.yaml
└── results/                            # Experimental results
    ├── pointer_v45_geolife_val_results.csv
    ├── pointer_v45_geolife_test_results.csv
    ├── pointer_v45_diy_val_results.csv
    ├── pointer_v45_diy_test_results.csv
    ├── mhsa_geolife_val_results.csv
    ├── mhsa_geolife_test_results.csv
    ├── mhsa_diy_val_results.csv
    ├── mhsa_diy_test_results.csv
    ├── lstm_geolife_val_results.csv
    ├── lstm_geolife_test_results.csv
    ├── lstm_diy_val_results.csv
    └── lstm_diy_test_results.csv
```

## Methodology

### 1. Search Space Design

Each model has a carefully designed search space that includes both architectural and optimization hyperparameters:

**Pointer Generator Transformer (Proposed Model)**:
- Architecture: d_model, nhead, num_layers, dim_feedforward, dropout
- Optimization: learning_rate, weight_decay, batch_size, label_smoothing, warmup_epochs

**MHSA (Baseline)**:
- Architecture: base_emb_size, num_encoder_layers, nhead, dim_feedforward, fc_dropout
- Optimization: lr, weight_decay, batch_size, num_warmup_epochs

**LSTM (Baseline)**:
- Architecture: base_emb_size, lstm_hidden_size, lstm_num_layers, lstm_dropout, fc_dropout
- Optimization: lr, weight_decay, batch_size, num_warmup_epochs

### 2. Experimental Protocol

- **Number of Trials**: 20 per model-dataset (120 total)
- **Parallel Execution**: 5 concurrent training jobs
- **Random Seed**: 42 (for reproducibility)
- **Patience**: 5 epochs (early stopping)
- **Validation Metric**: Acc@1 (primary objective)
- **Datasets**: 
  - Geolife: data/geolife_eps20/processed/geolife_eps20_prev7_*.pk
  - DIY: data/diy_eps50/processed/diy_eps50_prev7_*.pk

### 3. Result Tracking

All results are logged to CSV files with comprehensive information:
- Config name and path
- Hyperparameter values
- Model parameters count
- All evaluation metrics (Acc@1, Acc@5, Acc@10, MRR, NDCG, F1, loss)
- Experiment directory path
- Execution timestamp

## Usage

### Generate Configurations

```bash
cd /data/next_loc_clean_v2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
python scripts/sci_hyperparam_tuning/generate_configs.py
```

This generates 120 YAML configuration files (20 for each model-dataset pair).

### Run Hyperparameter Tuning

```bash
cd /data/next_loc_clean_v2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py
```

This runs all hyperparameter trials with 5 parallel jobs, automatically resuming from where it left off if interrupted.

### Find Best Hyperparameters

```python
import pandas as pd

# Load validation results
df = pd.read_csv('scripts/sci_hyperparam_tuning/results/pointer_v45_geolife_val_results.csv')
df = df[df['status'] == 'SUCCESS']

# Find best config by Val Acc@1
best_idx = df['acc_at_1'].idxmax()
best = df.loc[best_idx]
print(f"Best Val Acc@1: {best['acc_at_1']:.2f}%")
print(f"Config: {best['config_name']}")
```

### Run Final Evaluation (5 runs with best hyperparameters)

```bash
cd /data/next_loc_clean_v2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
python scripts/sci_hyperparam_tuning/run_final_evaluation.py
```

## Results Summary

### Best Hyperparameters (by Val Acc@1)

**Geolife Dataset**:
- **Pointer Generator Transformer**: 49.25% (config: pointer_v45_geolife_trial01, params: 443,404)
- **MHSA**: 42.38% (config: mhsa_geolife_trial17, params: 281,251)
- **LSTM**: 40.58% (config: lstm_geolife_trial00, params: 467,683)

**DIY Dataset**:
- **Pointer Generator Transformer**: 54.92% (config: pointer_v45_diy_trial09, params: 1,081,554)
- **LSTM**: 53.90% (config: lstm_diy_trial02, params: 3,564,990)
- **MHSA**: 53.69% (config: mhsa_diy_trial04, params: 797,982)

### Key Findings

1. **Pointer Generator Transformer outperforms baselines** on both datasets, validating the proposed architecture
2. **Performance ranking varies by dataset**:
   - Geolife: Pointer Generator Transformer > MHSA > LSTM
   - DIY: Pointer Generator Transformer > LSTM > MHSA
3. **Model complexity**: Pointer Generator Transformer achieves best results with moderate parameter counts

## Reproducibility

To reproduce the hyperparameter tuning:

1. Ensure the environment is set up with `mlenv` conda environment
2. Run the config generation script
3. Run the tuning script
4. Results will be saved to `results/` directory

All random seeds are fixed (seed=42) for full reproducibility.

## References

- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. Journal of machine learning research, 13(2).

## Notes

- Total experiments run: ~120 trials
- Average training time: ~5-15 minutes per trial (Geolife), ~10-30 minutes per trial (DIY)
- Total compute time: ~30-40 hours
- GPU: Tesla V100 32GB
