# Configuration Guide

This document provides a comprehensive guide to configuring the Pointer Generator Transformer model, including all parameters, their effects, and recommendations for different scenarios.

---

## 1. Configuration File Structure

### 1.1 Full Configuration Template

```yaml
# Pointer Generator Transformer Model Configuration
# Template with all available options

seed: 42  # Random seed for reproducibility

# Data settings
data:
  data_dir: data/geolife_eps20/processed    # Path to processed data
  dataset_prefix: geolife_eps20_prev7       # Prefix for data files
  dataset: geolife                          # Dataset name
  experiment_root: experiments              # Output directory for experiments
  num_workers: 0                            # DataLoader workers

# Model architecture
model:
  d_model: 64          # Model embedding dimension
  nhead: 4             # Number of attention heads
  num_layers: 2        # Number of transformer layers
  dim_feedforward: 128 # FFN hidden dimension
  dropout: 0.15        # Dropout probability

# Training settings
training:
  batch_size: 128           # Training batch size
  num_epochs: 50            # Maximum training epochs
  learning_rate: 0.00065    # Initial learning rate
  weight_decay: 0.015       # L2 regularization
  label_smoothing: 0.03     # Label smoothing factor
  grad_clip: 0.8            # Gradient clipping threshold
  patience: 5               # Early stopping patience
  min_epochs: 8             # Minimum epochs before early stopping
  warmup_epochs: 5          # Learning rate warmup epochs
  use_amp: true             # Enable mixed precision training
  min_lr: 0.000001          # Minimum learning rate
```

---

## 2. Data Configuration

### 2.1 data_dir

```yaml
data_dir: data/geolife_eps20/processed
```

| Setting | Description |
|---------|-------------|
| **Type** | String (path) |
| **Default** | None (required) |
| **Purpose** | Directory containing preprocessed pickle files |

**Expected files in directory:**
- `{prefix}_train.pk`
- `{prefix}_validation.pk`
- `{prefix}_test.pk`
- `{prefix}_metadata.json`

### 2.2 dataset_prefix

```yaml
dataset_prefix: geolife_eps20_prev7
```

| Setting | Description |
|---------|-------------|
| **Type** | String |
| **Default** | None (required) |
| **Purpose** | Prefix for data files |

The prefix encodes:
- Dataset name: `geolife` or `diy`
- Epsilon value: Clustering parameter
- Previous days: History window

### 2.3 dataset

```yaml
dataset: geolife
```

| Setting | Description |
|---------|-------------|
| **Type** | String |
| **Values** | `geolife`, `diy` |
| **Purpose** | Dataset identifier for experiment naming |

### 2.4 experiment_root

```yaml
experiment_root: experiments
```

| Setting | Description |
|---------|-------------|
| **Type** | String (path) |
| **Default** | `experiments` |
| **Purpose** | Root directory for experiment outputs |

### 2.5 num_workers

```yaml
num_workers: 0
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 0 |
| **Range** | 0-8 |
| **Purpose** | Number of DataLoader worker processes |

**Recommendations:**
- `0`: Single process (safest, works everywhere)
- `2-4`: Multi-process (faster for large datasets)
- Higher values: Diminishing returns

---

## 3. Model Architecture

### 3.1 d_model

```yaml
d_model: 64
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 128 |
| **Range** | 32-256 |
| **Purpose** | Main embedding/hidden dimension |

**Effects:**
- Higher: More expressive, more parameters
- Lower: Faster, less memory, may underfit

**Recommendations:**
| Dataset Size | d_model |
|--------------|---------|
| Small (<10K samples) | 32-64 |
| Medium (10K-100K) | 64-128 |
| Large (>100K) | 128-256 |

### 3.2 nhead

```yaml
nhead: 4
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 4 |
| **Values** | Must divide d_model |
| **Purpose** | Number of attention heads |

**Constraint:** `d_model % nhead == 0`

Valid combinations:
- d_model=64: nhead ∈ {1, 2, 4, 8, 16, 32, 64}
- d_model=128: nhead ∈ {1, 2, 4, 8, 16, 32, 64, 128}

**Recommendations:**
| d_model | Recommended nhead |
|---------|-------------------|
| 32-64 | 2-4 |
| 128 | 4-8 |
| 256 | 8-16 |

### 3.3 num_layers

```yaml
num_layers: 2
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 3 |
| **Range** | 1-6 |
| **Purpose** | Number of transformer encoder layers |

**Effects:**
- Higher: More capacity, can learn complex patterns
- Lower: Faster, less prone to overfitting

**Recommendations:**
| Dataset Complexity | num_layers |
|--------------------|------------|
| Simple patterns | 1-2 |
| Moderate | 2-3 |
| Complex | 3-4 |

**Ablation Result:** Single layer retains ~97% of performance.

### 3.4 dim_feedforward

```yaml
dim_feedforward: 128
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 256 |
| **Typical** | 2-4 × d_model |
| **Purpose** | FFN hidden layer dimension |

**Effects:**
- Higher: More non-linear capacity
- Lower: Faster, fewer parameters

**Recommendations:**
| d_model | dim_feedforward |
|---------|-----------------|
| 64 | 128-256 |
| 128 | 256-512 |

### 3.5 dropout

```yaml
dropout: 0.15
```

| Setting | Description |
|---------|-------------|
| **Type** | Float |
| **Default** | 0.15 |
| **Range** | 0.0-0.5 |
| **Purpose** | Dropout probability for regularization |

**Effects:**
- Higher: More regularization, prevents overfitting
- Lower: Full model capacity, may overfit

**Recommendations:**
| Scenario | dropout |
|----------|---------|
| Small dataset | 0.2-0.3 |
| Medium dataset | 0.1-0.2 |
| Large dataset | 0.05-0.15 |

---

## 4. Training Configuration

### 4.1 batch_size

```yaml
batch_size: 128
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 128 |
| **Range** | 16-512 |
| **Purpose** | Number of samples per batch |

**Effects:**
- Higher: Smoother gradients, faster (GPU utilization)
- Lower: More gradient noise (can help escape local minima)

**Recommendations:**
| GPU Memory | batch_size |
|------------|------------|
| 4GB | 32-64 |
| 8GB | 64-128 |
| 16GB+ | 128-256 |

### 4.2 num_epochs

```yaml
num_epochs: 50
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 50 |
| **Range** | 10-200 |
| **Purpose** | Maximum training epochs |

**Note:** Early stopping usually triggers before reaching num_epochs.

### 4.3 learning_rate

```yaml
learning_rate: 0.00065
```

| Setting | Description |
|---------|-------------|
| **Type** | Float |
| **Default** | 3e-4 |
| **Range** | 1e-5 to 1e-2 |
| **Purpose** | Initial learning rate |

**Effects:**
- Higher: Faster learning, may overshoot
- Lower: More stable, may be slow

**Recommendations:**
| Model Size | learning_rate |
|------------|---------------|
| Small (d=64) | 5e-4 to 1e-3 |
| Medium (d=128) | 3e-4 to 7e-4 |
| Large (d=256) | 1e-4 to 3e-4 |

**Tuned values:**
- GeoLife: `0.00065` (6.5e-4)
- DIY: `0.0007` (7e-4)

### 4.4 weight_decay

```yaml
weight_decay: 0.015
```

| Setting | Description |
|---------|-------------|
| **Type** | Float |
| **Default** | 0.015 |
| **Range** | 0.0-0.1 |
| **Purpose** | L2 regularization strength |

**Effects:**
- Higher: Stronger regularization
- Lower: More fitting capacity

**Recommendations:**
- Start with 0.01-0.02
- Increase if overfitting
- Decrease if underfitting

### 4.5 label_smoothing

```yaml
label_smoothing: 0.03
```

| Setting | Description |
|---------|-------------|
| **Type** | Float |
| **Default** | 0.03 |
| **Range** | 0.0-0.2 |
| **Purpose** | Label smoothing for cross-entropy |

**Effects:**
- Higher: More smoothing, prevents overconfidence
- Lower: Harder targets, more peaked predictions

**Recommendations:**
- 0.0: No smoothing (for interpretable probabilities)
- 0.03-0.1: Standard regularization
- 0.1-0.2: Heavy smoothing (rarely needed)

### 4.6 grad_clip

```yaml
grad_clip: 0.8
```

| Setting | Description |
|---------|-------------|
| **Type** | Float |
| **Default** | 0.8 |
| **Range** | 0.1-5.0 |
| **Purpose** | Gradient norm clipping threshold |

**Effects:**
- Higher: Less aggressive clipping
- Lower: More aggressive clipping (stable but slow)

**Recommendations:**
- 0.5-1.0: Standard for Transformers
- Increase if loss is unstable
- Decrease if seeing NaN

### 4.7 patience

```yaml
patience: 5
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 25 |
| **Range** | 3-50 |
| **Purpose** | Early stopping patience |

**Effects:**
- Higher: More chance for recovery
- Lower: Faster experiments

**Recommendations:**
- 3-5: Quick experiments
- 10-15: Standard training
- 25+: Maximum performance

### 4.8 min_epochs

```yaml
min_epochs: 8
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 8 |
| **Range** | 0-20 |
| **Purpose** | Minimum epochs before early stopping |

Prevents stopping too early during warmup phase.

### 4.9 warmup_epochs

```yaml
warmup_epochs: 5
```

| Setting | Description |
|---------|-------------|
| **Type** | Integer |
| **Default** | 5 |
| **Range** | 0-10 |
| **Purpose** | Learning rate warmup epochs |

**Effects:**
- Higher: More gradual warmup
- Lower: Faster to full learning rate

**Recommendations:**
- ~10% of num_epochs
- 3-5 epochs for most cases

### 4.10 use_amp

```yaml
use_amp: true
```

| Setting | Description |
|---------|-------------|
| **Type** | Boolean |
| **Default** | True |
| **Purpose** | Enable mixed precision training |

**Effects:**
- `true`: ~2x faster, lower memory
- `false`: Full precision (more stable)

**Recommendations:**
- `true` for modern GPUs (Volta+)
- `false` if seeing NaN or instability

### 4.11 min_lr

```yaml
min_lr: 0.000001
```

| Setting | Description |
|---------|-------------|
| **Type** | Float |
| **Default** | 1e-6 |
| **Range** | 1e-7 to 1e-5 |
| **Purpose** | Minimum learning rate floor |

Learning rate won't drop below this during cosine decay.

---

## 5. Dataset-Specific Configurations

### 5.1 GeoLife Configuration

```yaml
# config/models/config_pgt_geolife.yaml
seed: 42

data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments
  num_workers: 0

model:
  d_model: 64          # Smaller model for smaller dataset
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 50
  learning_rate: 0.00065  # Tuned value
  weight_decay: 0.015
  label_smoothing: 0.03
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 5
  use_amp: true
  min_lr: 0.000001
```

**Key choices:**
- Smaller d_model (64) for smaller dataset
- Fewer layers (2) to prevent overfitting
- Higher learning rate (6.5e-4) for faster convergence

### 5.2 DIY Configuration

```yaml
# config/models/config_pgt_diy.yaml
seed: 42

data:
  data_dir: data/diy_eps50/processed
  dataset_prefix: diy_eps50_prev7
  dataset: diy
  experiment_root: experiments
  num_workers: 0

model:
  d_model: 128         # Larger model for larger dataset
  nhead: 4
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 50
  learning_rate: 0.0007   # Tuned value
  weight_decay: 0.015
  label_smoothing: 0.03
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 5
  use_amp: true
  min_lr: 0.000001
```

**Key choices:**
- Larger d_model (128) for more locations
- More layers (3) for complex patterns
- Higher learning rate (7e-4)

---

## 6. Configuration Recommendations by Scenario

### 6.1 Quick Experimentation

```yaml
model:
  d_model: 64
  nhead: 4
  num_layers: 1
  dim_feedforward: 128
  dropout: 0.1

training:
  batch_size: 128
  num_epochs: 20
  patience: 3
  min_epochs: 5
```

**Purpose:** Fast iteration, baseline establishment

### 6.2 Maximum Performance

```yaml
model:
  d_model: 128
  nhead: 8
  num_layers: 3
  dim_feedforward: 256
  dropout: 0.15

training:
  batch_size: 128
  num_epochs: 100
  patience: 15
  min_epochs: 20
  warmup_epochs: 10
```

**Purpose:** Best possible results

### 6.3 Memory Constrained

```yaml
model:
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.15

training:
  batch_size: 64
  use_amp: true
```

**Purpose:** Fit in limited GPU memory

### 6.4 Overfitting Prevention

```yaml
model:
  d_model: 64
  dropout: 0.25

training:
  weight_decay: 0.03
  label_smoothing: 0.1
```

**Purpose:** When validation loss increases

---

## 7. Hyperparameter Tuning

### 7.1 Most Important Parameters

In order of impact:
1. **learning_rate**: Biggest effect on convergence
2. **d_model**: Model capacity
3. **num_layers**: Depth
4. **batch_size**: Training dynamics

### 7.2 Tuning Strategy

```python
# 1. Start with defaults
config = {
    'd_model': 64,
    'learning_rate': 3e-4,
    'num_layers': 2,
    'batch_size': 128,
}

# 2. Tune learning rate first
# Grid: [1e-4, 3e-4, 5e-4, 7e-4, 1e-3]

# 3. Scale d_model based on dataset size
# Small dataset: 64, Large: 128

# 4. Adjust layers
# More data → more layers

# 5. Fine-tune regularization
# weight_decay, dropout, label_smoothing
```

### 7.3 Signs of Misconfiguration

| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| Loss not decreasing | LR too low | Increase learning_rate |
| Loss oscillating | LR too high | Decrease learning_rate |
| Loss NaN | Numerical issues | Enable grad_clip, disable AMP |
| Val loss increasing | Overfitting | Increase dropout, weight_decay |
| Very slow training | Too many params | Reduce d_model, num_layers |

---

## 8. Configuration Validation

### 8.1 Automatic Checks

The training script validates:
- `d_model % nhead == 0`
- File paths exist
- Positive hyperparameters

### 8.2 Manual Verification

Before training, check:
```bash
# Verify data files exist
ls data/geolife_eps20/processed/
# Should see: ..._train.pk, ..._validation.pk, ..._test.pk, ..._metadata.json

# Verify config syntax
python -c "import yaml; yaml.safe_load(open('config/models/config_pgt_geolife.yaml'))"
```

---

## 9. Environment Variables

### 9.1 CUDA Settings

```bash
# Select GPU
export CUDA_VISIBLE_DEVICES=0

# For reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
```

### 9.2 PyTorch Settings

```python
# Set in code (already done in set_seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

*Next: [08_RESULTS_ANALYSIS.md](08_RESULTS_ANALYSIS.md) - Results and Analysis*
