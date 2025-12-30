# Hyperparameter Tuning Notebook - Update Summary

## Changes Made (2025-12-28)

### 1. Validation Tables - Dual Sorting ✓
Each validation table is now shown **TWO ways**:

1. **Sorted by Acc@1 (Descending)** - Shows best performing configurations first
2. **Sorted by Config Name (Ascending)** - Shows chronological/alphabetical order

Example for LSTM GeoLife:
- Section: "Validation Results (Sorted by Acc@1 Descending)"
- Section: "Validation Results (Sorted by Config Name Ascending)"

### 2. Test Tables - Single Sorting ✓
Test tables are shown **ONCE**:
- **Sorted by Config Name (Ascending)** - For easy reference and comparison

### 3. Config Parameters Exploded ✓
The "Config" column has been **exploded** into separate columns:

**Before:**
```
Config: "emb=32, bs=32, lr=0.001, hidden=128, layers=2, lstm_drop=0.2, fc_drop=0.2"
```

**After:**
```
lr | batch_size | base_emb_size | lstm_hidden_size | lstm_num_layers | lstm_dropout | fc_dropout | ...
0.001 | 32 | 32 | 128 | 2 | 0.2 | 0.2 | ...
```

### 4. Column Ordering ✓
Columns are now ordered as specified:

1. **Config Name** - Experiment identifier
2. **Params** - Number of trainable parameters
3. **All Metrics** - In exact order:
   - correct@1, correct@3, correct@5, correct@10
   - total
   - rr, ndcg, f1
   - acc@1, acc@5, acc@10
   - mrr, loss
4. **Config Parameters** - Ordered by dynamic → static:
   - **Dynamic params** (tuned): lr, batch_size, base_emb_size, hidden_size, num_layers, dropout, etc.
   - **Static params** (fixed): optimizer, patience, seed, dataset, etc.
5. **Notes** - Tuning rationale (last column)

### 5. Complete Parameter Coverage ✓
**ALL** config parameters from the YAML files are included, not just a summary. This includes:

- Model architecture params (embedding sizes, hidden dimensions, layers)
- Training params (lr, batch_size, optimizer, momentum, betas)
- Regularization params (dropout, weight_decay)
- Schedule params (lr_gamma, lr_step_size)
- Embedding flags (if_embed_time, if_embed_duration, if_embed_user, if_embed_poi)
- Dataset info (total_loc_num, total_user_num, previous_day)
- Experiment metadata (seed, patience, max_epoch)

## Tables Summary

### Per Model
Each model (LSTM, MHSA, Pointer V45, Markov) has:

**GeoLife Dataset:**
- Validation (sorted by Acc@1 ↓)
- Validation (sorted by Config Name ↑)
- Test (sorted by Config Name ↑)

**DIY Dataset:**
- Validation (sorted by Acc@1 ↓)
- Validation (sorted by Config Name ↑)
- Test (sorted by Config Name ↑)

**Total per model:** 6 tables (3 for each dataset)

**Total in notebook:** 24 main tables + 1 summary table = **25 tables**

## Usage

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Open notebook
cd /data/next_loc_clean_v2/notebooks
jupyter notebook hyperparameter_tuning_results.ipynb
```

## File Locations

- **Notebook**: `/data/next_loc_clean_v2/notebooks/hyperparameter_tuning_results.ipynb`
- **README**: `/data/next_loc_clean_v2/notebooks/README_hyperparameter_tuning.md`
- **Backup (v1)**: `/data/next_loc_clean_v2/notebooks/hyperparameter_tuning_results_v1.ipynb.bak`

## Key Improvements

1. ✓ **Better comparison**: Validation sorted by Acc@1 makes it easy to see best configs
2. ✓ **Better tracking**: Validation sorted by Config Name shows tuning progression
3. ✓ **Full transparency**: All config params visible, not hidden in summary string
4. ✓ **Easy analysis**: Can filter/sort on any specific parameter value
5. ✓ **Consistent ordering**: Same column order across all tables

## Technical Details

- All data still dynamically loaded (no hardcoded values)
- Metrics rounded to 2 decimal places
- Notes automatically derived from config changes
- Config parameter order: dynamic params first (frequently tuned), static params last
- Handles missing parameters gracefully (shows None/null)

---

*Updated: 2025-12-28*  
*Previous version backed up as: hyperparameter_tuning_results_v1.ipynb.bak*
