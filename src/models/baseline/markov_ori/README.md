# Original 1st-Order Markov Chain Baseline (markov_ori)

This folder contains a **faithful reproduction** of the original `markov.py` from `location-prediction-ori-freeze/baselines/markov.py`.

## Key Features

- Exact reproduction of the original Markov chain implementation
- Uses the original data splitting method (60% train, 20% val, 20% test by user's tracked days)
- Per-user Markov transition matrices
- Original metric calculation functions

## Differences from `markov1st.py`

| Aspect | markov_ori (this) | markov1st.py |
|--------|-------------------|--------------|
| Data format | Raw CSV data | Preprocessed pickle files |
| Evaluation | Consecutive pairs in sequences | Pre-extracted (X, Y) samples |
| Data split | Original 60/20/20 time-based | From preprocessed data |
| Metric calculation | Original functions | Uses metrics.py |
| GeoLife Test Acc@1 | **24.18%** | 27.64% |
| DIY Test Acc@1 | **44.13%** | 50.60% |

## File Structure

```
next_loc_clean_v2/
├── src/models/baseline/markov_ori/
│   └── run_markov_ori.py         # Main script
├── config/models/
│   ├── config_markov_ori_geolife.yaml
│   └── config_markov_ori_diy.yaml
├── data/
│   └── geolife_eps20/markov_ori_data/   # Original GeoLife data
│       ├── dataset_geolife.csv
│       └── valid_ids_geolife.pk
└── experiments/
    └── {dataset}_markov_ori_{timestamp}/
```

## Usage

From the `next_loc_clean_v2` root directory:

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Run on GeoLife
python src/models/baseline/markov_ori/run_markov_ori.py --config config/models/config_markov_ori_geolife.yaml

# Run on DIY
python src/models/baseline/markov_ori/run_markov_ori.py --config config/models/config_markov_ori_diy.yaml
```

## Configuration

### GeoLife (config_markov_ori_geolife.yaml)

```yaml
data:
  dataset: geolife
  data_csv: data/geolife_eps20/markov_ori_data/dataset_geolife.csv
  valid_ids_file: data/geolife_eps20/markov_ori_data/valid_ids_geolife.pk
  experiment_root: experiments

model:
  model_name: markov_ori
  markov_order: 1
```

### DIY (config_markov_ori_diy.yaml)

```yaml
data:
  dataset: diy
  data_csv: data/diy_eps50/interim/intermediate_eps50.csv
  processed_dir: data/diy_eps50/processed
  dataset_prefix: diy_eps50_prev7
  experiment_root: experiments

model:
  model_name: markov_ori
  markov_order: 1
```

## Benchmark Results

### GeoLife Dataset

| Metric | Validation | Test |
|--------|------------|------|
| **Acc@1**  | 33.57%     | **24.18%** |
| Acc@5  | 47.43%     | 37.87% |
| Acc@10 | 48.59%     | 38.76% |
| MRR    | 39.91%     | 30.34% |
| F1     | 32.87%     | 23.38% |
| NDCG   | 42.01%     | 32.38% |

- Total samples (val): 3,289
- Total samples (test): 3,457
- Total parameters: 166,309
- Number of users: 45
- Training time: ~5s

### DIY Dataset

| Metric | Validation | Test |
|--------|------------|------|
| **Acc@1**  | 48.10%     | **44.13%** |
| Acc@5  | 67.24%     | 62.56% |
| Acc@10 | 69.48%     | 64.80% |
| MRR    | 56.30%     | 52.13% |
| F1     | 46.01%     | 42.68% |
| NDCG   | 59.51%     | 55.22% |

- Total samples (val): 26,499
- Total samples (test): 26,872
- Total parameters: 366,338
- Number of users: 692
- Training time: ~43s

## Output Structure

Each run creates an experiment directory:

```
experiments/{dataset}_markov_ori_{yyyyMMdd_hhmmss}/
├── checkpoints/           # (empty for Markov model)
├── training.log           # Training log
├── config.yaml            # Flattened config
├── config_original.yaml   # Copy of original config
├── val_results.json       # Validation metrics
└── test_results.json      # Test metrics
```

## Algorithm

The original 1st-order Markov implementation:

1. **Data Loading**: Read CSV with columns (id, user_id, location_id, duration, start_day, end_day, start_min, end_min, weekday)

2. **Data Splitting**: For each user, split by tracked days:
   - Train: days 0 to 60% of max_day
   - Validation: days 60% to 80% of max_day
   - Test: days 80% to 100% of max_day

3. **Filtering**: Apply valid_ids to filter records (ensures consistency with neural models)

4. **Training**: For each user:
   - Build transition count table from consecutive location pairs
   - Store counts in DataFrame: (loc_1, toLoc, size)

5. **Prediction**: For each test sequence:
   - Look up current location in transition table
   - Return destinations sorted by transition count
   - If no match found, return zeros

6. **Metrics**: Calculate acc@1, acc@5, acc@10, MRR, F1, NDCG using original functions

## Notes

- Random seed is set to 0 (as in original)
- The original GeoLife data (dataset_geolife.csv) is copied to `data/geolife_eps20/markov_ori_data/`
- DIY uses intermediate CSV and generates valid_ids from preprocessed data
- The timestamp in experiment folder names is in GMT+7 timezone
