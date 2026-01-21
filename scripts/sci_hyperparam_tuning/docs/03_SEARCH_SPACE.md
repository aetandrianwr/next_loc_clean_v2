# Hyperparameter Search Space Design

## Table of Contents

1. [Search Space Philosophy](#search-space-philosophy)
2. [Pointer Generator Transformer Search Space](#pointer-v45-search-space)
3. [MHSA Search Space](#mhsa-search-space)
4. [LSTM Search Space](#lstm-search-space)
5. [Parameter Interactions](#parameter-interactions)
6. [Design Rationale](#design-rationale)

---

## Search Space Philosophy

### Goals of Search Space Design

1. **Cover Reasonable Range**: Include values that are likely to work well
2. **Avoid Extreme Values**: Exclude values known to fail (e.g., LR=1.0)
3. **Balance Breadth vs. Depth**: Not too narrow (miss optimal), not too wide (waste trials)
4. **Model-Appropriate**: Different architectures need different ranges

### Discrete vs. Continuous Search

We use **discrete search spaces** where each hyperparameter has a finite set of values:

```python
# Discrete (our approach)
learning_rate: [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]  # 5 values

# vs. Continuous (not used)
learning_rate: LogUniform(1e-4, 1e-3)  # infinite values
```

**Rationale**:
- Simpler implementation
- Easier to reproduce
- Values chosen from established best practices
- Sufficient granularity for our search budget

---

## Pointer Generator Transformer Search Space

### Complete Definition

```python
PGT_SEARCH_SPACE = {
    # ========== Architecture Hyperparameters ==========
    'd_model': [64, 96, 128],           # Model dimension
    'nhead': [2, 4, 8],                  # Number of attention heads
    'num_layers': [2, 3, 4],             # Transformer encoder layers
    'dim_feedforward': [128, 192, 256],  # FFN hidden dimension
    'dropout': [0.1, 0.15, 0.2, 0.25],   # Dropout probability
    
    # ========== Optimization Hyperparameters ==========
    'learning_rate': [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 0.01, 0.015],
    'batch_size': [64, 128, 256],
    'label_smoothing': [0.0, 0.01, 0.03, 0.05],
    'warmup_epochs': [3, 5, 7],
}
```

### Parameter Descriptions

#### Architecture Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `d_model` | 64, 96, 128 | Dimension of embeddings and hidden states |
| `nhead` | 2, 4, 8 | Number of parallel attention heads |
| `num_layers` | 2, 3, 4 | Depth of Transformer encoder stack |
| `dim_feedforward` | 128, 192, 256 | Hidden dimension in FFN layers |
| `dropout` | 0.1-0.25 | Dropout rate for regularization |

**Constraint**: `d_model` must be divisible by `nhead`
- ✅ 64 / 8 = 8 (valid)
- ✅ 96 / 2 = 48 (valid)
- ✅ 128 / 4 = 32 (valid)

#### Optimization Parameters

| Parameter | Values | Description |
|-----------|--------|-------------|
| `learning_rate` | 1e-4 to 1e-3 | Initial learning rate |
| `weight_decay` | 1e-5 to 0.015 | L2 regularization strength |
| `batch_size` | 64, 128, 256 | Number of samples per batch |
| `label_smoothing` | 0.0-0.05 | Label smoothing for cross-entropy |
| `warmup_epochs` | 3, 5, 7 | Linear warmup duration |

### Sample Configuration

```yaml
# pointer_v45_geolife_trial01.yaml - BEST configuration for Geolife
seed: 42
data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
model:
  d_model: 96
  nhead: 2
  num_layers: 2
  dim_feedforward: 192
  dropout: 0.25
training:
  batch_size: 64
  learning_rate: 0.001
  weight_decay: 1.0e-05
  label_smoothing: 0.0
  warmup_epochs: 5
  patience: 5
  num_epochs: 50
```

**Result**: Val Acc@1 = **49.25%** with 443,404 parameters

---

## MHSA Search Space

### Complete Definition

```python
MHSA_SEARCH_SPACE = {
    # ========== Architecture Hyperparameters ==========
    'base_emb_size': [32, 48, 64, 96],       # Base embedding dimension
    'num_encoder_layers': [2, 3, 4],          # Transformer encoder layers
    'nhead': [4, 8],                          # Number of attention heads
    'dim_feedforward': [128, 192, 256],       # FFN hidden dimension
    'fc_dropout': [0.1, 0.15, 0.2, 0.25],     # Fully-connected dropout
    
    # ========== Optimization Hyperparameters ==========
    'lr': [5e-4, 1e-3, 2e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'batch_size': [32, 64, 128, 256],
    'num_warmup_epochs': [1, 2, 3],
}
```

### Key Differences from Pointer Generator Transformer

1. **Simpler Learning Rate Range**: Only 3 values (MHSA is known to prefer higher LR)
2. **Lower Weight Decay Range**: MHSA tends to overfit less
3. **More Batch Size Options**: Includes 32 for fine-grained updates
4. **No Label Smoothing**: MHSA doesn't use this technique

### Sample Configuration

```yaml
# mhsa_geolife_trial17.yaml - BEST configuration for Geolife
seed: 42
embedding:
  base_emb_size: 64
model:
  networkName: transformer
  num_encoder_layers: 2
  nhead: 4
  dim_feedforward: 256
  fc_dropout: 0.15
optimiser:
  lr: 0.001
  weight_decay: 1.0e-05
  batch_size: 32
  num_warmup_epochs: 2
  patience: 5
```

**Result**: Val Acc@1 = **42.38%** with 281,251 parameters

---

## LSTM Search Space

### Complete Definition

```python
LSTM_SEARCH_SPACE = {
    # ========== Architecture Hyperparameters ==========
    'base_emb_size': [32, 48, 64, 96],       # Base embedding dimension
    'lstm_hidden_size': [128, 192, 256],      # LSTM hidden state size
    'lstm_num_layers': [1, 2, 3],             # Number of LSTM layers
    'lstm_dropout': [0.1, 0.2, 0.3],          # LSTM inter-layer dropout
    'fc_dropout': [0.1, 0.15, 0.2, 0.25],     # Fully-connected dropout
    
    # ========== Optimization Hyperparameters ==========
    'lr': [5e-4, 1e-3, 2e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'batch_size': [32, 64, 128, 256],
    'num_warmup_epochs': [1, 2, 3],
}
```

### Key Characteristics

1. **LSTM-Specific Parameters**: `lstm_hidden_size`, `lstm_num_layers`, `lstm_dropout`
2. **Larger Hidden Sizes**: LSTMs often need larger hidden states than Transformers
3. **Two Dropout Types**: 
   - `lstm_dropout`: Between LSTM layers (only applies when `num_layers > 1`)
   - `fc_dropout`: In final classification layer

### Sample Configuration

```yaml
# lstm_geolife_trial00.yaml - BEST configuration for Geolife
seed: 42
embedding:
  base_emb_size: 96
model:
  networkName: lstm
  lstm_hidden_size: 128
  lstm_num_layers: 1
  lstm_dropout: 0.2
  fc_dropout: 0.15
optimiser:
  lr: 0.002
  weight_decay: 0.0001
  batch_size: 64
  num_warmup_epochs: 1
  patience: 5
```

**Result**: Val Acc@1 = **40.58%** with 467,683 parameters

---

## Parameter Interactions

### Important Interactions to Consider

#### 1. `d_model` / `nhead` Constraint

For Transformer-based models, `d_model` must be divisible by `nhead`:

$$\text{head\_dim} = \frac{d\_model}{nhead}$$

Valid combinations in our search space:
- 64 / 2 = 32 ✓
- 64 / 4 = 16 ✓
- 64 / 8 = 8 ✓
- 96 / 2 = 48 ✓
- 96 / 4 = 24 ✓
- 96 / 8 = 12 ✓
- 128 / 2 = 64 ✓
- 128 / 4 = 32 ✓
- 128 / 8 = 16 ✓

All combinations are valid by design.

#### 2. Learning Rate × Batch Size Scaling

Larger batch sizes often need higher learning rates:

```
batch_size=64  → lr ∈ [1e-4, 5e-4] typical
batch_size=256 → lr ∈ [5e-4, 1e-3] typical
```

The search space covers this implicitly through random sampling.

#### 3. Model Size × Regularization

Larger models need more regularization:
- Large `d_model` + large `num_layers` → higher `dropout` helps
- Small models → lower `dropout` to prevent underfitting

#### 4. LSTM Depth × Dropout

For LSTMs, `lstm_dropout` only applies between layers:
- `lstm_num_layers=1` → `lstm_dropout` has no effect
- `lstm_num_layers=2,3` → `lstm_dropout` actively regularizes

---

## Design Rationale

### Why These Specific Values?

#### Learning Rates

```python
learning_rate: [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]
```

- **1e-4**: Conservative, good for fine-tuning or unstable training
- **3e-4**: Common default for Adam optimizer
- **5e-4**: Mid-range, good starting point
- **7e-4**: Slightly aggressive
- **1e-3**: Maximum reasonable for most architectures

Values below 1e-4 are too slow; values above 1e-3 often cause divergence.

#### Dropout Rates

```python
dropout: [0.1, 0.15, 0.2, 0.25]
```

- **0.1**: Minimal regularization
- **0.15**: Light regularization
- **0.2**: Standard for many NLP tasks
- **0.25**: Aggressive regularization for overfitting prevention

Values below 0.1 provide little benefit; values above 0.3 often hurt learning.

#### Weight Decay

```python
weight_decay: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 0.01, 0.015]
```

Extended range because:
- Some models benefit from very light regularization (1e-5)
- Others need strong regularization (0.01-0.015)
- Pointer Generator Transformer's pointer mechanism may need specific regularization

### Search Space Size Calculation

| Model | Formula | Total Combinations |
|-------|---------|-------------------|
| Pointer Generator Transformer | 3×3×3×3×4×5×7×3×4×3 | **816,480** |
| MHSA | 4×3×2×3×4×3×3×4×3 | **20,736** |
| LSTM | 4×3×3×3×4×3×3×4×3 | **41,472** |

With 20 trials per model-dataset, we explore:
- Pointer Generator Transformer: 0.002% of search space
- MHSA: 0.096% of search space
- LSTM: 0.048% of search space

This is sufficient because:
1. Most hyperparameters have low importance
2. The important hyperparameters are well-covered
3. Random search efficiently explores the important dimensions

---

## Summary Tables

### Pointer Generator Transformer Summary

| Category | Parameters | Total Values |
|----------|------------|--------------|
| Architecture | d_model, nhead, num_layers, dim_feedforward, dropout | 324 |
| Optimization | lr, weight_decay, batch_size, label_smoothing, warmup | 2,520 |
| **Total Combinations** | | **816,480** |

### MHSA Summary

| Category | Parameters | Total Values |
|----------|------------|--------------|
| Architecture | base_emb_size, num_encoder_layers, nhead, dim_feedforward, fc_dropout | 288 |
| Optimization | lr, weight_decay, batch_size, num_warmup_epochs | 72 |
| **Total Combinations** | | **20,736** |

### LSTM Summary

| Category | Parameters | Total Values |
|----------|------------|--------------|
| Architecture | base_emb_size, lstm_hidden_size, lstm_num_layers, lstm_dropout, fc_dropout | 432 |
| Optimization | lr, weight_decay, batch_size, num_warmup_epochs | 72 |
| **Total Combinations** | | **31,104** |

---

## Next: [04_IMPLEMENTATION.md](04_IMPLEMENTATION.md) - Code Implementation Details
