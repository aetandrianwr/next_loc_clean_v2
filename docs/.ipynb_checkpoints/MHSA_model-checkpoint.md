# MHSA (Multi-Head Self-Attention) Model for Next Location Prediction

## Overview

The MHSA model is a Transformer Encoder-based baseline model for next location prediction. This model predicts a user's next location based on their historical visit sequence and contextual features.

## Model Architecture

```
Input Sequence → AllEmbedding → TransformerEncoder → FullyConnected → Output Logits
```

### Components

1. **AllEmbedding**: Combines multiple embedding layers
   - Location embedding
   - Temporal embedding (hour, minute, weekday)
   - Duration embedding
   - Optional POI embedding
   - Positional encoding

2. **TransformerEncoder**: Multi-layer transformer encoder with self-attention
   - Configurable number of layers
   - Multi-head attention mechanism
   - GELU activation
   - Layer normalization

3. **FullyConnected**: Output layer
   - Optional user embedding
   - Residual connections with batch normalization
   - Final classification layer

## Installation

The model uses the `mlenv` conda environment. Make sure it's activated:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
```

Required packages:
- torch
- numpy
- pandas
- scikit-learn
- transformers
- PyYAML
- tqdm

## File Structure

```
next_loc_clean_v2/
├── config/
│   └── models/
│       ├── config_MHSA_geolife.yaml  # GeoLife dataset config
│       └── config_MHSA_diy.yaml      # DIY dataset config
├── data/
│   ├── geolife_eps20/processed/      # GeoLife preprocessed data
│   └── diy_eps50/processed/          # DIY preprocessed data
├── experiments/                       # Training outputs
│   └── {dataset}_MHSA_{timestamp}/
│       ├── checkpoints/
│       │   └── checkpoint.pt         # Best model checkpoint
│       ├── config.yaml               # Saved configuration
│       ├── config_original.yaml      # Original config file
│       ├── training.log              # Training log
│       ├── val_results.json          # Validation metrics
│       └── test_results.json         # Test metrics
├── src/
│   ├── evaluation/
│   │   └── metrics.py                # Evaluation metrics
│   ├── models/
│   │   └── baseline/
│   │       └── MHSA.py               # Model implementation
│   └── training/
│       └── train_MHSA.py             # Training script
└── docs/
    └── MHSA_model.md                 # This documentation
```

## Usage

### Training

Run training with a configuration file:

```bash
cd /data/next_loc_clean_v2

# Train on GeoLife dataset
python src/training/train_MHSA.py --config config/models/config_MHSA_geolife.yaml

# Train on DIY dataset
python src/training/train_MHSA.py --config config/models/config_MHSA_diy.yaml
```

### Configuration

Example configuration (GeoLife):

```yaml
seed: 42

data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments

training:
  if_embed_user: true
  if_embed_poi: false
  if_embed_time: true
  if_embed_duration: true
  previous_day: 7
  verbose: true
  debug: false
  batch_size: 32
  print_step: 20
  num_workers: 0
  day_selection: default

dataset_info:
  total_loc_num: 1187
  total_user_num: 46

embedding:
  base_emb_size: 32
  poi_original_size: 16

model:
  networkName: transformer
  num_encoder_layers: 2
  nhead: 8
  dim_feedforward: 128
  fc_dropout: 0.2

optimiser:
  optimizer: Adam
  max_epoch: 100
  lr: 0.001
  weight_decay: 0.000001
  beta1: 0.9
  beta2: 0.999
  num_warmup_epochs: 2
  num_training_epochs: 50
  patience: 3
  lr_step_size: 1
  lr_gamma: 0.1
```

### Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `seed` | Random seed for reproducibility |
| `data_dir` | Path to preprocessed data |
| `dataset_prefix` | Prefix for data files |
| `batch_size` | Training batch size |
| `base_emb_size` | Base embedding dimension |
| `num_encoder_layers` | Number of transformer encoder layers |
| `nhead` | Number of attention heads |
| `dim_feedforward` | FFN hidden dimension |
| `fc_dropout` | Dropout rate in FC layer |
| `max_epoch` | Maximum training epochs |
| `lr` | Learning rate |
| `patience` | Early stopping patience |

## Evaluation Metrics

The model uses the following metrics (from `src/evaluation/metrics.py`):

- **Acc@1, Acc@5, Acc@10**: Top-k accuracy percentages
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **F1**: Weighted F1 score

## Expected Performance

| Dataset | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|---------|-------|-------|--------|-----|------|
| GeoLife | ~29-31% | ~54% | ~59% | ~41% | ~45% |
| DIY | ~53% | ~77% | ~81% | ~63% | ~68% |

Note: Results may vary slightly due to random initialization.

## Output Format

After training, the experiment folder contains:

### `test_results.json`
```json
{
  "correct@1": 1037.0,
  "correct@3": 1627.0,
  "correct@5": 1908.0,
  "correct@10": 2065.0,
  "rr": 1430.5,
  "ndcg": 44.96,
  "f1": 0.212,
  "total": 3502.0,
  "acc@1": 29.61,
  "acc@5": 54.48,
  "acc@10": 58.94,
  "mrr": 40.84
}
```

### `training.log`
Contains detailed training progress including:
- Epoch-by-epoch metrics
- Learning rate changes
- Early stopping information
- Final test results

## Data Format

The preprocessed data files (`*.pk`) contain lists of dictionaries with:

| Key | Description |
|-----|-------------|
| `X` | Location sequence (numpy array) |
| `Y` | Target next location (int) |
| `user_X` | User ID sequence |
| `weekday_X` | Weekday sequence (0-6) |
| `start_min_X` | Start time in minutes |
| `dur_X` | Duration in minutes |
| `diff` | Days difference from current |

## Model API

### MHSA Class

```python
from src.models.baseline.MHSA import MHSA

# Create model
model = MHSA(config=config, total_loc_num=1187)

# Forward pass
# src: [seq_len, batch_size] - location sequence
# context_dict: dictionary with 'len', 'user', 'time', 'weekday', 'duration'
# device: torch.device
logits = model(src, context_dict, device)
# logits: [batch_size, total_loc_num]

# Get attention maps (for interpretation)
attention_maps = model.get_attention_maps(src, context_dict, device)
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in config
2. **KeyError 'poi'**: Set `if_embed_poi: false` if dataset doesn't have POI data
3. **Import errors**: Ensure you're in the correct directory and mlenv is activated

### Debug Mode

Enable debug mode for quick testing:
```yaml
training:
  debug: true
```

This limits training to ~20 batches per epoch.

## References

- Original implementation: `/data/location-prediction-ori-freeze`
- Transformer architecture: "Attention Is All You Need" (Vaswani et al., 2017)

## License

See LICENSE file in the repository root.
