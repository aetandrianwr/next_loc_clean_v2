# 1st-Order Markov Chain Model for Next Location Prediction

## Overview

This document describes two implementations of the 1st-Order Markov Chain baseline model for next location prediction:

1. **markov1st** (`src/models/baseline/markov1st.py`) - Adapted implementation using preprocessed data and `metrics.py`
2. **markov_ori** (`src/models/baseline/markov_ori/`) - Faithful reproduction of the original `location-prediction-ori-freeze/baselines/markov.py`

## Model Description

The 1st-Order Markov Chain model is based on the assumption that the next location depends only on the current location (and optionally the user). The model learns transition probabilities:

```
P(next_location | current_location, user)
```

For each user, the model builds a transition matrix where each entry represents the count of transitions from one location to another. During prediction, the model returns locations ranked by their transition counts from the current location.

## Implementation Comparison

| Aspect | markov1st | markov_ori |
|--------|-----------|------------|
| Data format | Preprocessed pickle files | Raw CSV data |
| Evaluation method | Pre-extracted (X, Y) samples | Consecutive pairs in sequences |
| Data split | From preprocessed data | Original 60/20/20 time-based |
| Metric calculation | Uses `metrics.py` | Original functions |
| **GeoLife Test Acc@1** | 27.64% | **24.18%** |
| **DIY Test Acc@1** | 50.60% | **44.13%** |

The difference in performance is due to the evaluation methodology:
- `markov_ori` evaluates on consecutive location pairs within user trajectories
- `markov1st` evaluates on pre-extracted samples from preprocessed data

## File Structure

```
next_loc_clean_v2/
├── src/
│   ├── models/
│   │   └── baseline/
│   │       ├── markov1st.py              # Adapted implementation
│   │       └── markov_ori/               # Original implementation
│   │           ├── run_markov_ori.py
│   │           └── README.md
│   ├── training/
│   │   └── calc_prob_markov1st.py        # Training script for markov1st
│   └── evaluation/
│       └── metrics.py                     # Shared evaluation metrics
├── config/
│   └── models/
│       ├── config_markov1st_geolife.yaml
│       ├── config_markov1st_diy.yaml
│       ├── config_markov_ori_geolife.yaml
│       └── config_markov_ori_diy.yaml
└── experiments/
    └── {dataset}_{model}_{timestamp}/
```

## Usage

### Prerequisites

Activate the conda environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
```

### Running markov1st (Adapted Implementation)

```bash
# GeoLife
python src/training/calc_prob_markov1st.py --config config/models/config_markov1st_geolife.yaml

# DIY
python src/training/calc_prob_markov1st.py --config config/models/config_markov1st_diy.yaml
```

### Running markov_ori (Original Implementation)

```bash
# GeoLife
python src/models/baseline/markov_ori/run_markov_ori.py --config config/models/config_markov_ori_geolife.yaml

# DIY
python src/models/baseline/markov_ori/run_markov_ori.py --config config/models/config_markov_ori_diy.yaml
```

### Configuration File Format

#### markov1st Configuration

```yaml
seed: 42

data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments

model:
  model_name: markov1st
  order: 1

dataset_info:
  total_loc_num: 1187
  total_user_num: 46
```

#### markov_ori Configuration

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

## Output Format

### Experiment Directory

Each training run creates a timestamped directory in `experiments/` with the format:
```
{dataset_name}_{model_name}_{yyyyMMdd_hhmmss}
```

For example: `geolife_markov1st_20251226_170200`

The timestamp is in GMT+7 timezone.

### Output Files

1. **checkpoints/markov1st_model.pkl**: Serialized model containing transition matrices
2. **training.log**: Complete log of the training process
3. **config.yaml**: Flattened configuration used for training
4. **config_original.yaml**: Copy of the original configuration file
5. **val_results.json**: Evaluation metrics on validation set
6. **test_results.json**: Evaluation metrics on test set

### Metrics Output Format

Both `val_results.json` and `test_results.json` contain:

```json
{
  "correct@1": 968.0,
  "correct@3": 1612.0,
  "correct@5": 1733.0,
  "correct@10": 1861.0,
  "rr": 1288.74,
  "ndcg": 39.68,
  "f1": 22.26,
  "total": 3502.0,
  "acc@1": 27.64,
  "acc@5": 49.49,
  "acc@10": 53.14,
  "mrr": 36.80
}
```

## Evaluation Metrics

All metrics are calculated using `src/evaluation/metrics.py`:

- **Acc@k**: Top-k accuracy (percentage of samples where correct answer is in top-k predictions)
- **MRR**: Mean Reciprocal Rank (average of 1/rank for correct predictions)
- **NDCG@10**: Normalized Discounted Cumulative Gain at rank 10
- **F1**: Weighted F1 score for top-1 predictions

## Performance Results

### markov1st (Adapted Implementation)

#### GeoLife Dataset

| Metric | Validation | Test |
|--------|------------|------|
| Acc@1  | 38.06%     | **27.64%** |
| Acc@5  | 59.03%     | 49.49% |
| Acc@10 | 63.11%     | 53.14% |
| MRR    | 46.51%     | 36.80% |
| NDCG   | 49.48%     | 39.68% |
| F1     | 32.30%     | 22.26% |

- Training samples: 7,424
- Test samples: 3,502
- Training time: ~0.2 seconds

#### DIY Dataset

| Metric | Validation | Test |
|--------|------------|------|
| Acc@1  | 50.73%     | **50.60%** |
| Acc@5  | 73.94%     | 72.99% |
| Acc@10 | 77.61%     | 76.61% |
| MRR    | 60.64%     | 60.31% |
| NDCG   | 64.57%     | 64.02% |
| F1     | 44.86%     | 45.72% |

- Training samples: 151,421
- Test samples: 12,368
- Training time: ~6.4 seconds

### markov_ori (Original Implementation)

#### GeoLife Dataset

| Metric | Validation | Test |
|--------|------------|------|
| Acc@1  | 33.57%     | **24.18%** |
| Acc@5  | 47.43%     | 37.87% |
| Acc@10 | 48.59%     | 38.76% |
| MRR    | 39.91%     | 30.34% |
| NDCG   | 42.01%     | 32.38% |
| F1     | 32.87%     | 23.38% |

- Total samples (val): 3,289
- Total samples (test): 3,457
- Total parameters: 166,309
- Number of users: 45

#### DIY Dataset

| Metric | Validation | Test |
|--------|------------|------|
| Acc@1  | 48.10%     | **44.13%** |
| Acc@5  | 67.24%     | 62.56% |
| Acc@10 | 69.48%     | 64.80% |
| MRR    | 56.30%     | 52.13% |
| NDCG   | 59.51%     | 55.22% |
| F1     | 46.01%     | 42.68% |

- Total samples (val): 26,499
- Total samples (test): 26,872
- Total parameters: 366,338
- Number of users: 692

## Model API

### Markov1stModel Class

```python
from src.models.baseline.markov1st import Markov1stModel

# Initialize
model = Markov1stModel(num_locations=1000, random_seed=42)

# Fit on training data
model.fit(train_data)

# Predict (returns list of predictions and targets)
predictions, targets = model.predict(test_data, top_k=10)

# Get logits for metrics calculation
logits, targets = model.predict_as_logits(test_data)

# Save/Load model
model.save('model.pkl')
loaded_model = Markov1stModel.load('model.pkl')
```

### Data Format

The model expects data as a list of dictionaries, where each dictionary contains:

```python
{
    'X': np.array([loc1, loc2, loc3]),  # Location sequence
    'user_X': np.array([user_id, ...]),  # User ID array (uses first element)
    'Y': target_location_id,             # Target location
    # ... other fields are ignored
}
```

## References

- Original implementation: `location-prediction-ori-freeze/baselines/markov.py`
- Evaluation metrics: `src/evaluation/metrics.py`

## Notes

1. The random seed is set to 42 by default for reproducibility
2. The model uses the last location in the input sequence (X) to predict the next location (Y)
3. Transition counts are stored per-user for personalized predictions
4. When a transition is not found, the model falls back to global statistics or frequency-based predictions
