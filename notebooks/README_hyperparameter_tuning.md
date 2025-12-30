# Hyperparameter Tuning Results Notebook

## Overview

The `hyperparameter_tuning_results.ipynb` notebook provides a comprehensive analysis of all hyperparameter tuning experiments conducted for the next location prediction models.

## Contents

The notebook includes results for:

### Models
1. **LSTM** - Long Short-Term Memory network
2. **MHSA** - Multi-Head Self-Attention (Transformer-based)
3. **Pointer Network V45** - Proposed model with pointer mechanism
4. **Markov Baseline** - Traditional Markov model (no hyperparameter tuning)

### Datasets
- **GeoLife**: Urban mobility dataset (parameter constraint: max 500K)
- **DIY**: Larger trajectory dataset (parameter constraint: max 3M)

### Data Presented

For each model and dataset combination, the notebook shows:
- **4 tables per model** (2 datasets × 2 splits):
  - GeoLife Validation Results
  - GeoLife Test Results
  - DIY Validation Results
  - DIY Test Results

Each table includes:
- **Config Name**: Experiment identifier
- **Config**: Key hyperparameters used
- **Params**: Number of trainable parameters
- **DateTime**: When the experiment was run
- **Metrics**: All evaluation metrics
  - correct@1, correct@3, correct@5, correct@10
  - total (number of samples)
  - rr (reciprocal rank)
  - ndcg (Normalized Discounted Cumulative Gain)
  - f1 (F1 score)
  - acc@1, acc@5, acc@10 (Top-k accuracy)
  - mrr (Mean Reciprocal Rank)
  - loss
- **Notes**: Tuning rationale (derived from experiment progression)

## Key Features

✓ **No hardcoded results** - All data is read directly from experiment files  
✓ **Automatic data collection** - Scans experiments directory  
✓ **Chronological tracking** - Shows progression of tuning attempts  
✓ **Metric rounding** - All metrics rounded to 2 decimal places  
✓ **Comprehensive coverage** - All experiments up to 2025-12-27  

## Usage

### Running the Notebook

```bash
# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Start Jupyter
jupyter notebook hyperparameter_tuning_results.ipynb
```

### Viewing Results

Simply execute all cells in order. The notebook will:
1. Import required libraries
2. Define helper functions
3. Collect experiment data automatically
4. Display tables for each model/dataset combination
5. Show summary statistics

## Experiment Coverage

The notebook includes experiments with dates **≤ 2025-12-27**:

- LSTM: ~9-10 experiments per dataset
- MHSA: ~7-9 experiments per dataset  
- Pointer V45: ~6-8 experiments per dataset
- Markov: ~2-3 experiments per dataset (baseline only)

**Total: ~50+ experiments**

## Data Sources

All data is read from:
- **Experiment directory**: `/data/next_loc_clean_v2/experiments/`
- **Config files**: `config.yaml` in each experiment folder
- **Results files**: `val_results.json` and `test_results.json`
- **Log files**: `training.log` (for parameter counts)
- **Documentation**: `/data/next_loc_clean_v2/docs/` (for tuning context)

## Summary Statistics

The notebook includes a summary table showing the **best configuration** for each model on each dataset, ranked by Test Acc@1.

## Notes Derivation

The "Notes" column is automatically derived by:
1. First experiment: marked as "Baseline configuration"
2. Subsequent experiments: analyzed against previous config to identify:
   - Dropout tuning
   - Learning rate adjustments
   - Architecture changes (embedding size, hidden dimensions, layers)
   - Batch size tuning
   - Other hyperparameter modifications

## Requirements

- Python 3.8+
- Libraries: pandas, numpy, yaml, json, glob, datetime
- Conda environment: `mlenv`
- Access to experiment files in `/data/next_loc_clean_v2/experiments/`

## Output Format

All metrics are presented in pandas DataFrames with:
- Clean formatting
- Rounded values (2 decimal places)
- Organized by model → dataset → split (val/test)
- Chronological ordering within each section

---

*Last updated: 2025-12-28*  
*Notebook version: 1.0*
