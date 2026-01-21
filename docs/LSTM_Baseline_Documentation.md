# LSTM Baseline Model Documentation

## Overview

The LSTM Baseline model is a recurrent neural network implementation for next location prediction. This implementation follows the architecture described in:

> Hong et al., 2023 - "Context-aware multi-head self-attentional neural network model for next location prediction" (Transportation Research Part C)

The LSTM baseline serves as a fair comparison benchmark against more advanced architectures like MHSA (Multi-Head Self-Attention) and Pointer Networks.

## Architecture

### Model Components

1. **AllEmbedding Layer**
   - Location embedding: Maps location IDs to dense vectors
   - Temporal embedding: Encodes time features (hour, minute, weekday)
   - Duration embedding: Encodes stay duration at locations

2. **LSTM Network**
   - Multi-layer LSTM for sequential pattern mining
   - Processes variable-length sequences using packed sequences
   - Hidden state captures sequential dependencies

3. **Fully Connected Output Layer**
   - User embedding for personalization
   - Residual block for enhanced representation
   - Softmax output over all locations

### Configuration Parameters

| Parameter | Geolife | DIY | Description |
|-----------|---------|-----|-------------|
| base_emb_size | 32 | 96 | Location/temporal embedding dimension |
| hidden_size | 64 | 128 | LSTM hidden state dimension |
| num_layers | 2 | 2 | Number of LSTM layers |
| lstm_dropout | 0.2 | 0.1 | Dropout between LSTM layers |
| fc_dropout | 0.2 | 0.1 | Dropout in FC layer |
| batch_size | 32 | 256 | Training batch size |

## Performance Results

### Geolife Dataset

| Metric | Paper (Hong et al.) | Our Implementation |
|--------|---------------------|-------------------|
| Acc@1 | 28.4 ± 0.8% | **29.98%** |
| Acc@5 | 55.8 ± 1.3% | **55.63%** |
| Acc@10 | 59.1 ± 0.7% | **58.57%** |
| MRR | 40.2 ± 1.1% | **41.77%** |
| NDCG@10 | 44.7 ± 0.6% | **45.67%** |

### DIY Dataset

| Metric | Our Implementation |
|--------|-------------------|
| Acc@1 | **52.73%** |
| Acc@5 | **77.01%** |
| Acc@10 | **81.17%** |
| MRR | **63.44%** |
| NDCG@10 | **67.53%** |

### Model Comparison (Geolife)

The performance hierarchy matches expectations from the paper:

| Model | Acc@1 | Acc@5 | MRR |
|-------|-------|-------|-----|
| Pointer Generator Transformer | ~54% | ~76% | ~65% |
| MHSA | ~31% | ~56% | ~43% |
| **LSTM Baseline** | ~30% | ~56% | ~42% |

## Usage

### Training

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Train on Geolife dataset
python src/training/train_LSTM_Baseline.py --config config/models/config_LSTM_Baseline_geolife.yaml

# Train on DIY dataset
python src/training/train_LSTM_Baseline.py --config config/models/config_LSTM_Baseline_diy.yaml
```

### Model Instantiation

```python
from src.models.baseline.LSTM_Baseline import LSTMBaseline

# Create model with configuration
model = LSTMBaseline(config=config, total_loc_num=1187)

# Forward pass
# x: [seq_len, batch_size] - location sequence
# x_dict: dictionary with 'user', 'time', 'weekday', 'duration', 'len'
# device: torch.device
logits = model(x, x_dict, device)  # Output: [batch_size, total_loc_num]
```

## File Structure

```
next_loc_clean_v2/
├── src/
│   ├── models/
│   │   └── baseline/
│   │       └── LSTM_Baseline.py      # Model implementation
│   └── training/
│       └── train_LSTM_Baseline.py    # Training script
├── config/
│   └── models/
│       ├── config_LSTM_Baseline_geolife.yaml
│       └── config_LSTM_Baseline_diy.yaml
└── experiments/
    └── {dataset}_{model}_{timestamp}/
        ├── checkpoints/
        │   └── checkpoint.pt         # Best model weights
        ├── training.log              # Training logs
        ├── config.yaml               # Used configuration
        ├── config_original.yaml      # Original config file
        ├── val_results.json          # Validation metrics
        └── test_results.json         # Test metrics
```

## Configuration Files

### Geolife Configuration

```yaml
# config/models/config_LSTM_Baseline_geolife.yaml
seed: 42

data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments

training:
  if_embed_user: true
  if_embed_time: true
  if_embed_duration: true
  batch_size: 32
  verbose: true

dataset_info:
  total_loc_num: 1187
  total_user_num: 46

embedding:
  base_emb_size: 32

model:
  hidden_size: 64
  num_layers: 2
  lstm_dropout: 0.2
  fc_dropout: 0.2

optimiser:
  optimizer: Adam
  lr: 0.001
  weight_decay: 0.000001
  patience: 5
```

## Implementation Notes

### Fair Comparison Design

To ensure fair comparison with MHSA and other baselines:

1. **Same Input Features**: Location, time, weekday, duration, and user embeddings
2. **Same Embedding Strategy**: Additive combination of embeddings
3. **Same Output Layer**: FC residual block with user embedding
4. **Same Training Setup**: Adam optimizer, early stopping, learning rate schedule
5. **Same Evaluation Metrics**: Using `src/evaluation/metrics.py`

### Key Design Decisions

1. **Packed Sequences**: Uses `pack_padded_sequence` for efficient handling of variable-length inputs
2. **Last Hidden State**: Uses the LSTM output at the last valid timestep for prediction
3. **Orthogonal Initialization**: LSTM hidden-to-hidden weights use orthogonal initialization
4. **Forget Gate Bias**: Initialized to 1.0 for better gradient flow

## Evaluation Metrics

The model is evaluated using the standard metrics from `src/evaluation/metrics.py`:

- **Acc@k**: Top-k accuracy (k = 1, 5, 10)
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **F1**: Weighted F1 score

## Training Details

### Early Stopping Strategy

Following the paper's setup:
1. Monitor validation loss
2. Stop if no improvement for `patience=5` epochs
3. Reduce learning rate by 0.1x and continue
4. Repeat early stopping process 3 times total

### Learning Rate Schedule

- Initial learning rate: 1e-3
- Warmup: 2 epochs
- Linear schedule with decay after warmup
- Learning rate reduction on early stopping

## References

1. Hong, Y., et al. (2023). "Context-aware multi-head self-attentional neural network model for next location prediction." Transportation Research Part C, 156, 104315.

2. Hochreiter, S., & Schmidhuber, J. (1997). "Long short-term memory." Neural computation, 9(8), 1735-1780.

## Version Information

- Created: 2026-01-01
- Random Seed: 42
- PyTorch Version: Compatible with 1.x and 2.x
- Training Environment: CUDA GPU (RTX series) or CPU
