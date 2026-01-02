# 6. Experimental Setup

## Datasets, Hyperparameters, and Training Protocol

---

## 6.1 Datasets

### 6.1.1 GeoLife Dataset

**Source**: Microsoft Research Asia GeoLife Project

**Description**: GPS trajectory data collected from 182 users over a period of 3+ years in Beijing, China.

**Characteristics**:
- Urban mobility patterns
- Mix of commute, leisure, and other trips
- High-quality GPS data

**Preprocessing**:
```
Raw GPS trajectories
        ↓
Stay point detection (ε=20 meters)
        ↓
Location clustering
        ↓
Sequence generation (prev_7 history window)
        ↓
Train/Validation/Test split
```

**Statistics**:
| Split | Samples | Purpose |
|-------|---------|---------|
| Training | 7,672 | Model learning |
| Validation | 3,485 | Early stopping |
| Test | 3,686 | Final evaluation |
| **Total** | **14,843** | |

**Data Location**:
```
data/geolife_eps20/processed/
├── geolife_eps20_prev7_train.pk
├── geolife_eps20_prev7_validation.pk
├── geolife_eps20_prev7_test.pk
└── geolife_eps20_prev7_metadata.json
```

---

### 6.1.2 DIY Dataset

**Source**: Custom mobility dataset

**Description**: Larger-scale mobility data with diverse user patterns.

**Preprocessing**:
```
Raw data
        ↓
Stay point detection (ε=50 meters)
        ↓
Location clustering
        ↓
Sequence generation (prev_7 history window)
        ↓
Train/Validation/Test split
```

**Statistics**:
| Split | Samples | Purpose |
|-------|---------|---------|
| Training | 193,510 | Model learning |
| Validation | 13,147 | Early stopping |
| Test | 16,348 | Final evaluation |
| **Total** | **223,005** | |

**Data Location**:
```
data/diy_eps50/processed/
├── diy_eps50_prev7_train.pk
├── diy_eps50_prev7_validation.pk
├── diy_eps50_prev7_test.pk
└── diy_eps50_prev7_metadata.json
```

---

### 6.1.3 Dataset Comparison

| Characteristic | GeoLife | DIY |
|----------------|---------|-----|
| Size | Small (14.8K) | Large (223K) |
| ε clustering | 20 meters | 50 meters |
| History window | 7 visits | 7 visits |
| Geographic scope | Beijing, China | Varies |
| Data quality | High (research) | Production |

---

### 6.1.4 Data Format

Each sample in the pickle files contains:

```python
sample = {
    'X': np.array([loc_1, loc_2, ..., loc_n]),  # Historical locations
    'Y': int,                                    # Target location
    'user_X': np.array([user_id, ...]),         # User ID (repeated)
    'weekday_X': np.array([day_1, day_2, ...]), # Day of week (0-6)
    'start_min_X': np.array([min_1, ...]),      # Start minute (0-1439)
    'dur_X': np.array([dur_1, ...]),            # Duration in minutes
    'diff': np.array([diff_1, ...]),            # Days since visit
}
```

**Preprocessing in DataLoader**:
```python
# Time discretization (15-min buckets)
time = sample['start_min_X'] // 15  # 0-95

# Duration discretization (30-min buckets)
duration = sample['dur_X'] // 30  # 0-99 (clamped)

# Recency clamping
recency = np.clip(sample['diff'], 0, 8)  # 0-8 days
```

---

## 6.2 Hyperparameters

### 6.2.1 GeoLife Configuration

```yaml
# File: pointer_v45_geolife_trial01.yaml

seed: 42

data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments
  num_workers: 0

model:
  d_model: 96          # Model dimension
  nhead: 2             # Attention heads
  num_layers: 2        # Transformer layers
  dim_feedforward: 192 # FFN dimension (2× d_model)
  dropout: 0.25        # Dropout rate

training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.001  # 1e-3
  weight_decay: 1.0e-05
  label_smoothing: 0.0
  grad_clip: 0.8
  patience: 5           # Early stopping patience
  min_epochs: 8
  warmup_epochs: 5
  use_amp: true
  min_lr: 1.0e-06
```

---

### 6.2.2 DIY Configuration

```yaml
# File: pointer_v45_diy_trial09.yaml

seed: 42

data:
  data_dir: data/diy_eps50/processed
  dataset_prefix: diy_eps50_prev7
  dataset: diy
  experiment_root: experiments
  num_workers: 0

model:
  d_model: 64           # Smaller model dimension
  nhead: 4              # More attention heads
  num_layers: 2
  dim_feedforward: 256  # 4× d_model
  dropout: 0.2

training:
  batch_size: 64
  num_epochs: 50
  learning_rate: 0.0005  # 5e-4 (lower than GeoLife)
  weight_decay: 1.0e-05
  label_smoothing: 0.05  # Non-zero smoothing
  grad_clip: 0.8
  patience: 5
  min_epochs: 8
  warmup_epochs: 7       # Longer warmup
  use_amp: true
  min_lr: 1.0e-06
```

---

### 6.2.3 Hyperparameter Comparison

| Parameter | GeoLife | DIY | Rationale |
|-----------|---------|-----|-----------|
| d_model | 96 | 64 | GeoLife needs more capacity |
| nhead | 2 | 4 | DIY benefits from more heads |
| num_layers | 2 | 2 | Same depth |
| dim_feedforward | 192 | 256 | Standard ratios (2×, 4×) |
| dropout | 0.25 | 0.2 | GeoLife needs more regularization |
| learning_rate | 1e-3 | 5e-4 | DIY converges slower |
| label_smoothing | 0.0 | 0.05 | DIY benefits from smoothing |
| warmup_epochs | 5 | 7 | DIY needs longer warmup |

---

### 6.2.4 Fixed Ablation Parameters

For all ablation experiments, we fix:

| Parameter | Value | Reason |
|-----------|-------|--------|
| **seed** | 42 | Reproducibility |
| **patience** | 5 | Consistent early stopping |
| **min_epochs** | 8 | Ensure minimum training |
| **grad_clip** | 0.8 | Stable training |
| **use_amp** | True | Efficient training |

---

## 6.3 Training Protocol

### 6.3.1 Optimizer

**AdamW** (Adam with Decoupled Weight Decay):

```python
optimizer = optim.AdamW(
    model.parameters(),
    lr=learning_rate,       # From config
    weight_decay=weight_decay,
    betas=(0.9, 0.98),      # Standard for transformers
    eps=1e-9,               # Numerical stability
)
```

**Why AdamW?**
- Better generalization than Adam
- Decoupled weight decay is theoretically sound
- Standard for transformer training

---

### 6.3.2 Learning Rate Schedule

**Warmup + Cosine Decay**:

```python
def get_lr(epoch):
    if epoch < warmup_epochs:
        # Linear warmup
        return base_lr * (epoch + 1) / warmup_epochs
    else:
        # Cosine decay
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(π * progress))
```

**Visual**:
```
Learning Rate
    │
base_lr ─────────────╮
    │       /         ╲
    │      /           ╲
    │     /             ╲
    │    /               ╲
min_lr /                 ───────
    └────┬───────────────┬────── Epoch
        warmup      end of training
```

**Why This Schedule?**
- Warmup prevents early instability
- Cosine decay is smoother than step decay
- Standard for transformer training

---

### 6.3.3 Loss Function

**Cross-Entropy with Label Smoothing**:

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=0,              # Ignore padding
    label_smoothing=label_smoothing,
)
```

**Label Smoothing**:
```
Without smoothing:  Target = [0, 0, 1, 0, 0]  (one-hot)
With smoothing=0.1: Target = [0.02, 0.02, 0.92, 0.02, 0.02]
```

**Why Label Smoothing?**
- Prevents overconfident predictions
- Acts as regularization
- Improves generalization

---

### 6.3.4 Mixed Precision Training

```python
# Enable AMP (Automatic Mixed Precision)
scaler = torch.cuda.amp.GradScaler()

# Training step
with torch.cuda.amp.autocast():
    logits = model(x, x_dict)
    loss = criterion(logits, y)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
scaler.step(optimizer)
scaler.update()
```

**Benefits**:
- 2× memory efficiency
- Faster training
- No accuracy loss (when done correctly)

---

### 6.3.5 Early Stopping

```python
patience = 5
min_epochs = 8
best_val_loss = float('inf')
patience_counter = 0

for epoch in range(num_epochs):
    # ... training ...
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        save_checkpoint()
    else:
        patience_counter += 1
        
    if patience_counter >= patience and epoch >= min_epochs:
        print("Early stopping!")
        break
```

**Rationale**:
- `patience=5`: Wait 5 epochs without improvement
- `min_epochs=8`: Train at least 8 epochs regardless
- Uses validation loss (not accuracy) for stability

---

### 6.3.6 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```

**Purpose**:
- Prevents exploding gradients
- Stabilizes training
- `grad_clip=0.8` is relatively aggressive

---

## 6.4 Hardware and Environment

### 6.4.1 Hardware Specifications

| Component | Specification |
|-----------|---------------|
| GPU | Tesla V100-SXM2-32GB |
| GPU Memory | 32 GB HBM2 |
| Compute Capability | 7.0 |
| CUDA Cores | 5,120 |

### 6.4.2 Software Environment

```bash
# Conda environment
conda activate mlenv

# Versions
Python: 3.8+
PyTorch: 1.12.1
CUDA: 11.x
NumPy: 1.x
Pandas: 1.x
scikit-learn: 1.x
```

### 6.4.3 Activation Command

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
```

---

## 6.5 Training Time Estimates

### 6.5.1 Per-Experiment Time

| Dataset | Epochs | Time/Epoch | Total Time |
|---------|--------|------------|------------|
| GeoLife | ~10 | ~5 sec | ~1 min |
| DIY | ~12 | ~70 sec | ~15 min |

### 6.5.2 Full Study Time

```
GeoLife: 9 ablations × ~1 min = ~9 min
DIY: 9 ablations × ~15 min = ~135 min

Parallel execution (3 jobs): Reduced by ~3×

Estimated total: ~1-2 hours with parallel execution
```

---

## 6.6 Parallel Execution Strategy

### 6.6.1 Batch Configuration

```python
MAX_PARALLEL_JOBS = 3    # Run 3 experiments simultaneously
JOB_DELAY_SECONDS = 5    # 5 second delay between starts
```

### 6.6.2 Batching Strategy

```
Batch 1: [full, no_pointer, no_generation]      - Start: 0s, 5s, 10s
         Wait for all to complete...
         
Batch 2: [no_position_bias, no_temporal, no_user]
         Wait for all to complete...
         
Batch 3: [no_pos_from_end, single_layer, no_gate]
         Wait for all to complete...
```

### 6.6.3 Why 5-Second Delay?

- Prevents file system conflicts
- Avoids timestamp collision in directory names
- Allows GPU memory to stabilize

---

## 6.7 Output Structure

### 6.7.1 Per-Experiment Output

```
ablation_{dataset}_{variant}_{timestamp}/
├── checkpoints/
│   └── best.pt                    # Best model weights
├── training.log                   # Detailed training log
├── results.json                   # Final metrics (JSON)
└── config.yaml                    # Experiment config
```

### 6.7.2 Aggregated Output

```
results/{dataset}/
├── ablation_results.csv           # All results in CSV
├── ablation_table.tex             # LaTeX table
└── ablation_{dataset}_{variant}_{timestamp}/
    └── ...                        # Individual experiments
```

---

## 6.8 Verification Checklist

Before running the full study:

- [ ] **Environment**: `conda activate mlenv`
- [ ] **GPU**: `nvidia-smi` shows available GPU
- [ ] **Data**: Files exist in expected locations
- [ ] **Config**: YAML files are correct
- [ ] **Space**: Sufficient disk space for checkpoints
- [ ] **Baseline**: Run full model first to verify

---

*Next: [07_results.md](07_results.md) - Complete experimental results*
