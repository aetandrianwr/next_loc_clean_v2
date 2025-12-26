# LSTM Model for Next Location Prediction

## Overview

The LSTM model is a baseline model for next location prediction that replaces the Transformer Encoder from the MHSA model with LSTM layers. This model predicts a user's next location based on their historical visit sequence and contextual features.

The LSTM naturally handles sequential dependencies without requiring explicit causal masking or positional encoding, making it a simpler alternative to the transformer-based approach.

## Model Architecture

```
Input Sequence → AllEmbeddingLSTM → LSTM → LayerNorm → FullyConnected → Output Logits
```

### Components

1. **AllEmbeddingLSTM**: Combines multiple embedding layers (without positional encoding)
   - Location embedding
   - Temporal embedding (hour, minute, weekday)
   - Duration embedding
   - Optional POI embedding
   - Dropout (no positional encoding needed for LSTM)

2. **LSTM**: Multi-layer LSTM encoder
   - Configurable hidden size
   - Configurable number of layers
   - Dropout between layers
   - Packed sequences for efficient variable-length processing
   - Extracts final hidden state for each sequence

3. **LayerNorm**: Normalization after LSTM output

4. **FullyConnected**: Output layer
   - Optional user embedding
   - Residual connections with batch normalization
   - Final classification layer

## Key Differences from MHSA

| Feature | MHSA (Transformer) | LSTM |
|---------|-------------------|------|
| Positional Encoding | Sinusoidal | Not needed (inherent in RNN) |
| Causal Masking | Required | Not needed (unidirectional) |
| Attention Mechanism | Multi-head self-attention | Not used |
| Sequence Processing | Parallel | Sequential |
| Memory | Attention-based | Hidden states |

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

## File Structure

```
next_loc_clean_v2/
├── config/
│   └── models/
│       ├── config_LSTM_geolife.yaml  # GeoLife dataset config
│       └── config_LSTM_diy.yaml      # DIY dataset config
├── data/
│   ├── geolife_eps20/processed/      # GeoLife preprocessed data
│   └── diy_eps50/processed/          # DIY preprocessed data
├── experiments/                       # Training outputs
│   └── {dataset}_LSTM_{timestamp}/
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
│   │       └── LSTM.py               # Model implementation
│   └── training/
│       └── train_LSTM.py             # Training script
└── docs/
    └── LSTM_model.md                 # This documentation
```

## Usage

### Training

Run training with a configuration file:

```bash
cd /data/next_loc_clean_v2

# Train on GeoLife dataset
python src/training/train_LSTM.py --config config/models/config_LSTM_geolife.yaml

# Train on DIY dataset
python src/training/train_LSTM.py --config config/models/config_LSTM_diy.yaml
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
  networkName: lstm
  lstm_hidden_size: 128
  lstm_num_layers: 2
  lstm_dropout: 0.2
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
| `base_emb_size` | Base embedding dimension (input to LSTM) |
| `lstm_hidden_size` | LSTM hidden dimension |
| `lstm_num_layers` | Number of LSTM layers |
| `lstm_dropout` | Dropout between LSTM layers |
| `fc_dropout` | Dropout rate in FC layer |
| `max_epoch` | Maximum training epochs |
| `lr` | Learning rate |
| `patience` | Early stopping patience |

### Parameter Budget

The model is designed to meet specific parameter budgets:

| Dataset | Budget | Actual Parameters | Configuration |
|---------|--------|-------------------|---------------|
| GeoLife | < 500K | ~483K | hidden=128, layers=2 |
| DIY | < 3M | ~2.85M | hidden=192, layers=2 |

## Evaluation Metrics

The model uses the following metrics (from `src/evaluation/metrics.py`):

- **Acc@1, Acc@5, Acc@10**: Top-k accuracy percentages
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **F1**: Weighted F1 score

## Benchmark Results

Results from training with random seed 42:

### GeoLife Dataset
| Metric | Value |
|--------|-------|
| Acc@1 | 29.73% |
| Acc@5 | 54.31% |
| Acc@10 | 58.85% |
| MRR | 40.81% |
| NDCG | 44.93% |
| F1 | 18.60% |
| Parameters | 482,659 |
| Training Time | ~65s |

### DIY Dataset
| Metric | Value |
|--------|-------|
| Acc@1 | 51.74% |
| Acc@5 | 76.66% |
| Acc@10 | 80.76% |
| MRR | 62.82% |
| NDCG | 66.97% |
| F1 | 44.82% |
| Parameters | 2,847,582 |
| Training Time | ~556s |

## Output Format

After training, the experiment folder contains:

### `test_results.json`
```json
{
  "correct@1": 1041.0,
  "correct@3": 1695.0,
  "correct@5": 1902.0,
  "correct@10": 2061.0,
  "rr": 1429.09,
  "ndcg": 44.93,
  "f1": 0.186,
  "total": 3502.0,
  "acc@1": 29.73,
  "acc@5": 54.31,
  "acc@10": 58.85,
  "mrr": 40.81
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

### LSTMModel Class

```python
from src.models.baseline.LSTM import LSTMModel

# Create config object
class Config:
    pass

config = Config()
config.base_emb_size = 32
config.lstm_hidden_size = 128
config.lstm_num_layers = 2
config.lstm_dropout = 0.2
config.fc_dropout = 0.2
config.if_embed_user = True
config.if_embed_time = True
config.if_embed_duration = True
config.if_embed_poi = False
config.total_user_num = 46

# Create model
model = LSTMModel(config=config, total_loc_num=1187)

# Forward pass
# src: [seq_len, batch_size] - location sequence
# context_dict: dictionary with 'len', 'user', 'time', 'weekday', 'duration'
# device: torch.device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

logits = model(src, context_dict, device)
# logits: [batch_size, total_loc_num]
```

### Parameter Count Calculation

```python
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
```

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `batch_size` in config
2. **KeyError 'poi'**: Set `if_embed_poi: false` if dataset doesn't have POI data
3. **Import errors**: Ensure you're in the correct directory and mlenv is activated
4. **Parameter count exceeded**: Reduce `lstm_hidden_size` or `lstm_num_layers`

### Debug Mode

Enable debug mode for quick testing:
```yaml
training:
  debug: true
```

This limits training to ~20 batches per epoch.

## Comparison with MHSA

| Aspect | LSTM | MHSA |
|--------|------|------|
| Sequence Modeling | Recurrent (sequential) | Self-attention (parallel) |
| Long-range Dependencies | May struggle | Handles well |
| Training Speed | Generally faster | May be slower |
| Memory Usage | Lower | Higher |
| Interpretability | Hidden states | Attention maps available |

## References

- Original MHSA implementation: `src/models/baseline/MHSA.py`
- LSTM: "Long Short-Term Memory" (Hochreiter & Schmidhuber, 1997)
- Evaluation metrics: `src/evaluation/metrics.py`

## License

See LICENSE file in the repository root.
