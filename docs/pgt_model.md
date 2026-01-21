# Pointer Generator Transformer - Proposed Model Documentation

## Overview

The Pointer Generator Transformer (PointerGeneratorTransformer) is a proposed model for next location prediction that combines a Transformer encoder with a pointer mechanism and a generation head. The model uses a learned gate to adaptively blend copy-based predictions (from user history) with generation-based predictions (from the full location vocabulary).

## Model Architecture

### Key Components

1. **Embedding Layers**
   - **Location Embedding**: Maps location IDs to dense vectors (d_model dimensions)
   - **User Embedding**: Maps user IDs to dense vectors (d_model dimensions)
   - **Temporal Embeddings**:
     - Time of day (96 intervals, 15-min each)
     - Day of week (7 days)
     - Recency (days ago, 8 levels)
     - Duration (30-min buckets, 100 levels)
   - **Position-from-end Embedding**: Encodes how far each position is from the sequence end

2. **Input Projection**
   - Combines all embeddings into a single representation
   - Projects to model dimension (d_model)
   - LayerNorm for stability

3. **Transformer Encoder**
   - Pre-norm architecture with GELU activation
   - Configurable number of layers, heads, and feedforward dimension
   - Dropout for regularization

4. **Pointer Mechanism**
   - Query-Key attention over encoded sequence
   - Position bias for recency preference
   - Scatters attention weights to location vocabulary

5. **Generation Head**
   - Linear projection to full location vocabulary
   - Softmax for probability distribution

6. **Pointer-Generation Gate**
   - MLP with sigmoid output
   - Learns to blend pointer and generation distributions

### Architecture Diagram

```
Input Sequence → [Location + User + Temporal Embeddings]
                            ↓
                    Input Projection
                            ↓
                 Sinusoidal Position Encoding
                            ↓
                   Transformer Encoder
                            ↓
              [Last Position Context Vector]
                     ↓            ↓
              Pointer Query   Generation Head
                     ↓            ↓
              Pointer Attention  Gen Distribution
                     ↓            ↓
              Pointer Distribution  ←  Gate  →  Gen Distribution
                            ↓
                   Final Probability
                            ↓
                      Log Softmax
```

## Configuration

### GeoLife Dataset Configuration

```yaml
model:
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 50
  learning_rate: 0.00065
  weight_decay: 0.015
  label_smoothing: 0.03
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 5
  use_amp: true
  min_lr: 0.000001
```

### DIY Dataset Configuration

```yaml
model:
  d_model: 128
  nhead: 4
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 50
  learning_rate: 0.0007
  weight_decay: 0.015
  label_smoothing: 0.03
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 5
  use_amp: true
  min_lr: 0.000001
```

## Usage

### Training

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Train on GeoLife dataset
python src/training/train_pgt.py --config config/models/config_pgt_geolife.yaml

# Train on DIY dataset
python src/training/train_pgt.py --config config/models/config_pgt_diy.yaml
```

### Using the Model

```python
from src.models.proposed.pgt import PointerGeneratorTransformer

# Create model
model = PointerGeneratorTransformer(
    num_locations=1000,
    num_users=100,
    d_model=128,
    nhead=4,
    num_layers=3,
    dim_feedforward=256,
    dropout=0.15,
)

# Forward pass
# x: [seq_len, batch_size] - location sequence
# x_dict: dictionary with temporal features
log_probs = model(x, x_dict)  # [batch_size, num_locations]

# Get prediction
prediction = log_probs.argmax(dim=-1)
```

## Expected Performance

| Dataset | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|---------|-------|-------|--------|-----|------|
| GeoLife | ~54.00% | ~81.10% | ~84.38% | ~65.81% | ~70.21% |
| DIY | ~56.87% | ~82.24% | ~86.13% | ~67.97% | ~72.29% |

## Output Structure

Training creates an experiment directory with the following structure:

```
experiments/{dataset}_{model}_{timestamp}/
├── checkpoints/
│   └── best.pt           # Best model checkpoint
├── config.yaml           # Configuration used
├── config_original.yaml  # Original config file
├── training.log          # Training log
├── val_results.json      # Validation metrics
└── test_results.json     # Test metrics
```

### Example test_results.json

```json
{
  "correct@1": 1889.0,
  "correct@3": 2671.0,
  "correct@5": 2840.0,
  "correct@10": 2955.0,
  "rr": 2304.576904296875,
  "ndcg": 70.21452188491821,
  "f1": 0.49764558422132177,
  "total": 3502.0,
  "acc@1": 53.94,
  "acc@5": 81.10,
  "acc@10": 84.38,
  "mrr": 65.81,
  "loss": 2.70
}
```

## Training Features

1. **Mixed Precision Training (AMP)**: Enabled by default for faster training
2. **Warmup + Cosine LR Schedule**: Gradual warmup followed by cosine decay
3. **Early Stopping**: Based on validation loss with configurable patience
4. **Gradient Clipping**: Prevents gradient explosion
5. **Label Smoothing**: Regularization for better generalization

## File Structure

```
src/
├── models/
│   └── proposed/
│       ├── __init__.py
│       └── pgt.py      # Model implementation
├── training/
│   └── train_pgt.py    # Training script
└── evaluation/
    └── metrics.py              # Evaluation metrics

config/
└── models/
    ├── config_pgt_geolife.yaml
    └── config_pgt_diy.yaml
```

## Evaluation Metrics

The model is evaluated using the following metrics (from `src/evaluation/metrics.py`):

- **Acc@k**: Top-k accuracy (k=1, 5, 10)
- **MRR**: Mean Reciprocal Rank
- **NDCG@10**: Normalized Discounted Cumulative Gain
- **F1**: Weighted F1 score

## Dependencies

- Python 3.8+
- PyTorch 1.9+
- NumPy
- scikit-learn
- PyYAML
- tqdm

## References

- Original implementation based on experiment_new_lose_2/poi_h3_2/train_pgt.py
- Pointer Networks paper: "Pointer Networks" (Vinyals et al., 2015)
- Transformer architecture: "Attention is All You Need" (Vaswani et al., 2017)
