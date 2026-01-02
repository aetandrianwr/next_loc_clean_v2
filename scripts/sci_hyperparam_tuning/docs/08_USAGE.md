# Usage Guide

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Quick Start](#quick-start)
3. [Step-by-Step Guide](#step-by-step-guide)
4. [Configuration Customization](#configuration-customization)
5. [Analyzing Results](#analyzing-results)
6. [Troubleshooting](#troubleshooting)
7. [Best Practices](#best-practices)

---

## Prerequisites

### Environment Setup

1. **Conda Environment**:
   ```bash
   # The project uses mlenv conda environment
   source ~/miniconda3/etc/profile.d/conda.sh
   conda activate mlenv
   ```

2. **Required Packages**:
   - PyTorch (with CUDA support recommended)
   - transformers
   - pandas
   - numpy
   - pyyaml
   - scikit-learn
   - tqdm

3. **GPU Recommendation**:
   - Minimum: 8GB VRAM
   - Recommended: 16GB+ VRAM (Tesla V100 used in experiments)
   - CPU training is possible but significantly slower

### Data Requirements

The preprocessed data must exist in:
```
data/
├── geolife_eps20/processed/
│   ├── geolife_eps20_prev7_train.pk
│   ├── geolife_eps20_prev7_val.pk
│   └── geolife_eps20_prev7_test.pk
└── diy_eps50/processed/
    ├── diy_eps50_prev7_train.pk
    ├── diy_eps50_prev7_val.pk
    └── diy_eps50_prev7_test.pk
```

---

## Quick Start

### Reproduce All Experiments

```bash
# Navigate to repository root
cd /data/next_loc_clean_v2

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Step 1: Generate all configurations (120 files)
python scripts/sci_hyperparam_tuning/generate_configs.py

# Step 2: Run hyperparameter tuning (this takes ~30-40 hours)
python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py

# Step 3: (Optional) Run final evaluation with multiple seeds
python scripts/sci_hyperparam_tuning/run_final_evaluation.py
```

### Run Single Experiment

```bash
# Train a single model with specific config
python src/training/train_pointer_v45.py \
    --config scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml
```

---

## Step-by-Step Guide

### Step 1: Generate Configurations

```bash
python scripts/sci_hyperparam_tuning/generate_configs.py
```

**What it does**:
- Creates `configs/` directory
- Generates 120 YAML configuration files:
  - 20 for Pointer V45 on Geolife
  - 20 for Pointer V45 on DIY
  - 20 for MHSA on Geolife
  - 20 for MHSA on DIY
  - 20 for LSTM on Geolife
  - 20 for LSTM on DIY
- Creates `all_configs_summary.yaml` with all configurations

**Output**:
```
Generated 120 configuration files in scripts/sci_hyperparam_tuning/configs
Saved configs summary to scripts/sci_hyperparam_tuning/configs/all_configs_summary.yaml
```

### Step 2: Run Hyperparameter Tuning

```bash
python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py
```

**What it does**:
- Loads all configurations from `all_configs_summary.yaml`
- Checks for already completed experiments (resume capability)
- Runs experiments with 5 parallel jobs
- Logs results to CSV files in `results/`
- Creates experiment directories in `experiments/`

**Monitoring Progress**:
```bash
# Watch the tuning log
tail -f scripts/sci_hyperparam_tuning/tuning_log.txt

# Count completed experiments
grep "SUCCESS" scripts/sci_hyperparam_tuning/results/*_val_results.csv | wc -l
```

**Expected Output**:
```
============================================================
SCIENTIFIC HYPERPARAMETER TUNING
============================================================
Base directory: /data/next_loc_clean_v2
Max parallel jobs: 5
Delay between jobs: 1.5s
============================================================
Total configs: 120
Already completed: 0
Remaining: 120
============================================================

[08:10:34] Starting: pointer_v45_geolife_trial00
[08:10:36] Starting: pointer_v45_geolife_trial01
...
[08:11:32] DONE: pointer_v45_geolife_trial01 (Val Acc@1: 49.25%, Time: 0.9min)
```

### Step 3: Analyze Results

```python
import pandas as pd

# Load validation results
df = pd.read_csv('scripts/sci_hyperparam_tuning/results/pointer_v45_geolife_val_results.csv')
df_success = df[df['status'] == 'SUCCESS']

# Find best configuration
best_idx = df_success['acc_at_1'].idxmax()
best = df_success.loc[best_idx]

print(f"Best Config: {best['config_name']}")
print(f"Val Acc@1: {best['acc_at_1']:.2f}%")
print(f"Parameters: {best['num_params']:,}")
print(f"Hyperparameters: {best['hyperparameters']}")
```

### Step 4: (Optional) Final Evaluation

```bash
python scripts/sci_hyperparam_tuning/run_final_evaluation.py
```

**What it does**:
- Identifies best configuration for each model-dataset pair
- Runs 5 training runs with different seeds (42, 123, 456, 789, 1011)
- Computes mean ± std for all metrics
- Saves results to `final_results.json` and `final_results_summary.csv`

---

## Configuration Customization

### Modifying Search Space

Edit `hyperparam_search_space.py`:

```python
# Add new learning rate values
POINTER_V45_SEARCH_SPACE = {
    'learning_rate': [1e-4, 2e-4, 3e-4, 5e-4, 7e-4, 1e-3],  # Added 2e-4
    # ...
}

# Increase number of trials
NUM_TRIALS = 30  # Default is 20
```

### Creating Custom Configuration

```python
# custom_config.py
import yaml

config = {
    'seed': 42,
    'data': {
        'data_dir': 'data/geolife_eps20/processed',
        'dataset_prefix': 'geolife_eps20_prev7',
        'dataset': 'geolife',
        'experiment_root': 'experiments',
    },
    'model': {
        'd_model': 96,
        'nhead': 4,
        'num_layers': 3,
        'dim_feedforward': 256,
        'dropout': 0.2,
    },
    'training': {
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.0005,
        'weight_decay': 0.0001,
        'label_smoothing': 0.01,
        'patience': 5,
        'warmup_epochs': 5,
    },
}

with open('my_custom_config.yaml', 'w') as f:
    yaml.dump(config, f)
```

### Running with Custom Config

```bash
python src/training/train_pointer_v45.py --config my_custom_config.yaml
```

---

## Analyzing Results

### Loading and Exploring Results

```python
import pandas as pd
import numpy as np

# Load all validation results
models = ['pointer_v45', 'mhsa', 'lstm']
datasets = ['geolife', 'diy']

all_results = []
for model in models:
    for dataset in datasets:
        df = pd.read_csv(f'scripts/sci_hyperparam_tuning/results/{model}_{dataset}_val_results.csv')
        all_results.append(df)

combined = pd.concat(all_results)
print(combined.groupby(['model_name', 'dataset'])['acc_at_1'].describe())
```

### Finding Best Hyperparameters

```python
def get_best_config(model, dataset):
    """Get best configuration for a model-dataset pair."""
    df = pd.read_csv(f'scripts/sci_hyperparam_tuning/results/{model}_{dataset}_val_results.csv')
    df = df[df['status'] == 'SUCCESS']
    
    if len(df) == 0:
        return None
    
    best_idx = df['acc_at_1'].idxmax()
    return df.loc[best_idx]

# Example usage
best = get_best_config('pointer_v45', 'geolife')
print(f"Best trial: {best['config_name']}")
print(f"Val Acc@1: {best['acc_at_1']:.2f}%")
```

### Hyperparameter Importance Analysis

```python
import ast

def analyze_hyperparameter_importance(model, dataset, param_name):
    """Analyze how a hyperparameter affects performance."""
    df = pd.read_csv(f'scripts/sci_hyperparam_tuning/results/{model}_{dataset}_val_results.csv')
    df = df[df['status'] == 'SUCCESS']
    
    # Parse hyperparameters
    df['hp'] = df['hyperparameters'].apply(ast.literal_eval)
    df[param_name] = df['hp'].apply(lambda x: x.get(param_name))
    
    # Group by parameter value
    grouped = df.groupby(param_name)['acc_at_1'].agg(['mean', 'std', 'count'])
    return grouped.sort_values('mean', ascending=False)

# Example: Learning rate importance
print(analyze_hyperparameter_importance('pointer_v45', 'geolife', 'learning_rate'))
```

---

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory

```bash
RuntimeError: CUDA out of memory
```

**Solution**: Reduce batch size in config
```yaml
training:
  batch_size: 32  # Reduce from 64/128
```

#### 2. Training Script Not Found

```bash
ModuleNotFoundError: No module named 'src'
```

**Solution**: Ensure you're in the correct directory
```bash
cd /data/next_loc_clean_v2
python src/training/train_pointer_v45.py --config ...
```

#### 3. Data File Not Found

```bash
FileNotFoundError: data/geolife_eps20/processed/...
```

**Solution**: Run preprocessing first or check data paths in config

#### 4. Experiment Already Exists

The tuning manager automatically skips completed experiments. To re-run:
```bash
# Remove specific result from CSV
# Or delete the experiment directory
rm -rf experiments/geolife_pointer_v45_YYYYMMDD_HHMMSS
```

### Debugging Tips

```python
# Check if configs were generated correctly
import yaml
with open('scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml') as f:
    config = yaml.safe_load(f)
print(config)

# Check results file format
df = pd.read_csv('scripts/sci_hyperparam_tuning/results/pointer_v45_geolife_val_results.csv')
print(df.columns)
print(df.head())
```

---

## Best Practices

### 1. Start Small

Before running full hyperparameter tuning:
```bash
# Test with a single configuration
python src/training/train_pointer_v45.py \
    --config scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial00.yaml
```

### 2. Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# Check if parallel jobs are running
ps aux | grep python
```

### 3. Use Screen/tmux for Long Runs

```bash
# Start a screen session
screen -S tuning

# Run tuning
python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py

# Detach: Ctrl+A, D
# Reattach: screen -r tuning
```

### 4. Regular Backups

```bash
# Backup results periodically
cp -r scripts/sci_hyperparam_tuning/results results_backup_$(date +%Y%m%d)
```

### 5. Document Custom Experiments

When modifying search spaces or adding experiments:
- Update `hyperparam_search_space.py` with comments
- Log changes in a separate file
- Keep original configurations as reference

---

## File Reference

| File | Purpose |
|------|---------|
| `hyperparam_search_space.py` | Define search spaces |
| `generate_configs.py` | Create configuration files |
| `run_hyperparam_tuning.py` | Execute parallel tuning |
| `run_final_evaluation.py` | Run final evaluation |
| `configs/` | Generated YAML configs |
| `results/` | CSV result files |
| `tuning_log.txt` | Execution log |
| `docs/` | This documentation |

---

## Summary Commands

```bash
# Full workflow
cd /data/next_loc_clean_v2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Generate configs
python scripts/sci_hyperparam_tuning/generate_configs.py

# Run tuning (background)
nohup python scripts/sci_hyperparam_tuning/run_hyperparam_tuning.py > tuning_log.txt 2>&1 &

# Check progress
tail -f scripts/sci_hyperparam_tuning/tuning_log.txt

# Analyze results
python -c "
import pandas as pd
for m in ['pointer_v45', 'mhsa', 'lstm']:
    for d in ['geolife', 'diy']:
        df = pd.read_csv(f'scripts/sci_hyperparam_tuning/results/{m}_{d}_val_results.csv')
        df = df[df['status']=='SUCCESS']
        if len(df) > 0:
            best = df.loc[df['acc_at_1'].idxmax()]
            print(f'{m} on {d}: {best[\"acc_at_1\"]:.2f}% ({best[\"config_name\"]})')
"
```

---

## Contact and Support

For issues or questions about this hyperparameter tuning system:
1. Check this documentation
2. Review the code comments
3. Examine the tuning log for error messages
4. Check the experiment directories for training logs

---

**End of Documentation**

Return to: [01_OVERVIEW.md](01_OVERVIEW.md)
