# MHSA Training and Configuration Guide

## Complete Guide to Training the MHSA Model

---

## Table of Contents

1. [Quick Start](#1-quick-start)
2. [Environment Setup](#2-environment-setup)
3. [Data Preparation](#3-data-preparation)
4. [Configuration Deep Dive](#4-configuration-deep-dive)
5. [Training Workflow](#5-training-workflow)
6. [Monitoring and Debugging](#6-monitoring-and-debugging)
7. [Hyperparameter Tuning](#7-hyperparameter-tuning)
8. [Best Practices](#8-best-practices)

---

## 1. Quick Start

### 1.1 Minimal Commands

```bash
# Navigate to project
cd /data/next_loc_clean_v2

# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Train on GeoLife
python src/training/train_MHSA.py --config config/models/config_MHSA_geolife.yaml

# Train on DIY
python src/training/train_MHSA.py --config config/models/config_MHSA_diy.yaml
```

### 1.2 Expected Output

```
Using device: cuda
Experiment directory: experiments/geolife_MHSA_20260101_120000
Data loaders: train=232, val=105, test=110
Total trainable parameters: 112547
Epoch 1, 100.0%  loss: 4.234 acc@1: 8.12 mrr: 15.23, ndcg: 18.45, took: 12.34s
Validation loss = 3.89 acc@1 = 15.23 f1 = 12.45 mrr = 22.34, ndcg = 25.67
...
Test Results:
  acc@1: 29.61%
  acc@5: 54.48%
  acc@10: 58.94%
  mrr: 40.84%
  ndcg: 44.96%
  f1: 21.20%
```

---

## 2. Environment Setup

### 2.1 Required Packages

```
torch>=1.9.0
numpy>=1.19.0
pandas>=1.2.0
scikit-learn>=0.24.0
transformers>=4.5.0
PyYAML>=5.4.0
```

### 2.2 Hardware Requirements

| Dataset | Minimum GPU Memory | Recommended |
|---------|-------------------|-------------|
| GeoLife | 2 GB | 4 GB |
| DIY | 4 GB | 8 GB |

CPU training is supported but significantly slower.

### 2.3 Verify Installation

```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 3. Data Preparation

### 3.1 Required Files

Before training, ensure these files exist:

```
data/{dataset}_eps{epsilon}/processed/
├── {dataset}_eps{epsilon}_prev{days}_train.pk
├── {dataset}_eps{epsilon}_prev{days}_validation.pk
├── {dataset}_eps{epsilon}_prev{days}_test.pk
└── {dataset}_eps{epsilon}_prev{days}_metadata.json
```

### 3.2 Metadata File Contents

```json
{
  "dataset_name": "geolife",
  "epsilon": 20,
  "previous_day": 7,
  "total_user_num": 46,
  "total_loc_num": 1187,
  "train_sequences": 7424,
  "val_sequences": 3334,
  "test_sequences": 3502
}
```

### 3.3 Data Verification

```python
import pickle
import json

# Check training data
with open('data/geolife_eps20/processed/geolife_eps20_prev7_train.pk', 'rb') as f:
    train_data = pickle.load(f)
    
print(f"Number of training samples: {len(train_data)}")
print(f"Sample keys: {train_data[0].keys()}")
print(f"Location sequence length: {len(train_data[0]['X'])}")
```

---

## 4. Configuration Deep Dive

### 4.1 Complete Configuration Template

```yaml
# ============================================
# MHSA Model Configuration Template
# ============================================

# Random seed for reproducibility
seed: 42

# ----- DATA SETTINGS -----
data:
  # Path to processed data directory
  data_dir: data/geolife_eps20/processed
  
  # Prefix for data files (without _train.pk, _test.pk, etc.)
  dataset_prefix: geolife_eps20_prev7
  
  # Dataset name (geolife or diy)
  dataset: geolife
  
  # Directory for saving experiments
  experiment_root: experiments

# ----- TRAINING SETTINGS -----
training:
  # Feature toggles
  if_embed_user: true      # User personalization
  if_embed_poi: false      # POI features (if available)
  if_embed_time: true      # Time of day embedding
  if_embed_duration: true  # Stay duration embedding
  
  # History window (must match data preprocessing)
  previous_day: 7
  
  # Verbosity
  verbose: true            # Print training progress
  debug: false             # Run only few batches (for testing)
  
  # Batch configuration
  batch_size: 32           # Samples per batch
  print_step: 20           # Print every N batches
  num_workers: 0           # DataLoader workers (0 for safety)

# ----- DATASET INFO (from metadata) -----
dataset_info:
  # Must match metadata.json
  total_loc_num: 1187      # Number of unique locations + special tokens
  total_user_num: 46       # Number of unique users + buffer

# ----- EMBEDDING SETTINGS -----
embedding:
  # Main embedding dimension
  base_emb_size: 32        # D: dimension of all embeddings
  
  # POI vector size (if using POI)
  poi_original_size: 16
  
  # For concat mode (not default)
  time_emb_size: 32

# ----- MODEL ARCHITECTURE -----
model:
  networkName: transformer
  
  # Transformer encoder configuration
  num_encoder_layers: 2    # Number of encoder layers
  nhead: 8                 # Number of attention heads
  dim_feedforward: 128     # FFN hidden dimension
  
  # Dropout in output layer
  fc_dropout: 0.2

# ----- OPTIMIZER SETTINGS -----
optimiser:
  # Optimizer type
  optimizer: Adam          # Adam or SGD
  
  # Learning rate
  lr: 0.001               # Initial learning rate
  weight_decay: 0.000001  # L2 regularization
  
  # Adam-specific
  beta1: 0.9              # First moment decay
  beta2: 0.999            # Second moment decay
  
  # SGD-specific (if using SGD)
  momentum: 0.98
  
  # Training schedule
  max_epoch: 100          # Maximum epochs
  num_warmup_epochs: 2    # LR warmup period
  num_training_epochs: 50 # Scheduled training length
  
  # Early stopping
  patience: 5             # Stop after N epochs without improvement
  
  # LR reduction (after early stop trigger)
  lr_step_size: 1         # Step size for StepLR
  lr_gamma: 0.1           # LR multiplier
```

### 4.2 Configuration Parameter Reference

#### Data Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `data_dir` | str | required | Path to processed data |
| `dataset_prefix` | str | required | File name prefix |
| `dataset` | str | required | "geolife" or "diy" |
| `experiment_root` | str | "experiments" | Output directory |

#### Training Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `if_embed_user` | bool | true | - | Enable user embeddings |
| `if_embed_time` | bool | true | - | Enable temporal embeddings |
| `if_embed_duration` | bool | true | - | Enable duration embeddings |
| `batch_size` | int | 32 | 8-512 | Batch size |
| `verbose` | bool | true | - | Print progress |
| `debug` | bool | false | - | Debug mode (~20 batches) |

#### Model Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `base_emb_size` | int | 32 | 16-256 | Embedding dimension |
| `num_encoder_layers` | int | 2 | 1-8 | Encoder depth |
| `nhead` | int | 8 | 1-16 | Attention heads |
| `dim_feedforward` | int | 128 | 64-1024 | FFN hidden size |
| `fc_dropout` | float | 0.2 | 0.0-0.5 | Output dropout |

**Constraint:** `base_emb_size` must be divisible by `nhead`

#### Optimizer Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `optimizer` | str | "Adam" | Adam/SGD | Optimizer type |
| `lr` | float | 0.001 | 1e-5 to 0.1 | Learning rate |
| `weight_decay` | float | 1e-6 | 0 to 1e-3 | L2 penalty |
| `patience` | int | 5 | 3-10 | Early stop patience |
| `max_epoch` | int | 100 | 50-500 | Max training epochs |

---

## 5. Training Workflow

### 5.1 Training Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING WORKFLOW                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. INITIALIZATION                                              │
│  ─────────────────                                              │
│  • Load configuration                                           │
│  • Set random seed                                              │
│  • Create experiment directory                                  │
│  • Initialize log file                                          │
│                                                                 │
│  2. DATA LOADING                                                │
│  ────────────────                                               │
│  • Load train/val/test .pk files                               │
│  • Create DataLoaders with collate function                     │
│  • Move to device (GPU if available)                           │
│                                                                 │
│  3. MODEL CREATION                                              │
│  ────────────────                                               │
│  • Create MHSA model with config                               │
│  • Initialize weights (Xavier uniform)                         │
│  • Move model to device                                        │
│  • Create optimizer and schedulers                             │
│                                                                 │
│  4. TRAINING LOOP                                               │
│  ───────────────                                                │
│  For each epoch:                                                │
│    ┌───────────────────────────────────────────────────────┐   │
│    │ TRAIN PHASE                                            │   │
│    │ • Iterate through train_loader                         │   │
│    │ • Forward pass, compute loss                           │   │
│    │ • Backward pass, clip gradients                        │   │
│    │ • Update parameters                                    │   │
│    │ • Update learning rate (warmup scheduler)              │   │
│    └───────────────────────────────────────────────────────┘   │
│                                                                 │
│    ┌───────────────────────────────────────────────────────┐   │
│    │ VALIDATION PHASE                                       │   │
│    │ • Evaluate on val_loader                              │   │
│    │ • Compute metrics (loss, acc, mrr, ndcg, f1)          │   │
│    │ • Log results                                          │   │
│    └───────────────────────────────────────────────────────┘   │
│                                                                 │
│    ┌───────────────────────────────────────────────────────┐   │
│    │ EARLY STOPPING CHECK                                   │   │
│    │ • If val_loss improved: save checkpoint, reset counter │   │
│    │ • If no improvement: increment counter                 │   │
│    │ • If counter >= patience:                             │   │
│    │     - If not max retries: reduce LR, reload best      │   │
│    │     - Else: stop training                             │   │
│    └───────────────────────────────────────────────────────┘   │
│                                                                 │
│  5. FINAL EVALUATION                                            │
│  ──────────────────                                             │
│  • Load best checkpoint                                         │
│  • Evaluate on test_loader                                     │
│  • Compute per-user metrics                                    │
│  • Save final results                                          │
│                                                                 │
│  6. OUTPUT                                                      │
│  ────────                                                       │
│  • checkpoint.pt (best model weights)                          │
│  • config.yaml (used configuration)                            │
│  • training.log (full training log)                            │
│  • test_results.json (final metrics)                           │
│  • val_results.json (best validation metrics)                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 5.2 Understanding Training Output

```
Epoch 1, 50.0%   loss: 5.234 acc@1: 5.12 mrr: 10.23, ndcg: 12.45, took: 6.12s
│       │        │          │         │         │            │
│       │        │          │         │         │            └─ Time for batches
│       │        │          │         │         └─ NDCG@10 so far
│       │        │          │         └─ MRR so far
│       │        │          └─ Top-1 accuracy so far
│       │        └─ Average loss over print_step batches
│       └─ Progress through epoch
└─ Current epoch number
```

### 5.3 Learning Rate Schedule

```
Learning Rate over Training:

    LR
    │
0.001├────┐
    │    │    ┌──────────────────────┐
    │    │    │ Linear decay         │
    │    └────┘                      │
    │                                │
0.0001├                              └───┐
    │                                    │ After ES
    │                                    │ trigger
0.00001├                                 └───
    │
    └────┬─────────────────────────────────────▶ Epochs
         │←Warmup→│←───Main Training───→│←Fine-tune→│
         0        2                     ~30        ~40
```

---

## 6. Monitoring and Debugging

### 6.1 Training Log Structure

```
training.log contents:
─────────────────────
Training MHSA model
Dataset: geolife
Config: config/models/config_MHSA_geolife.yaml
Device: cuda
Seed: 42
==================================================
Data loaders: train=232, val=105, test=110
Total trainable parameters: 112547

=== Epoch 1 ===
Epoch 1, 100.0%  loss: 4.234 acc@1: 8.12 mrr: 15.23, ndcg: 18.45, took: 12.34s
Validation - loss: 3.890, acc@1: 15.23%, f1: 12.45%, mrr: 22.34%, ndcg: 25.67%
Validation loss decreased (inf --> 3.8900). Saving model...
Current learning rate: 0.000900
==================================================

...

=== Test Results ===
Test Results:
  acc@1: 29.61%
  acc@5: 54.48%
  acc@10: 58.94%
  mrr: 40.84%
  ndcg: 44.96%
  f1: 21.20%

=== Training Complete ===
```

### 6.2 Common Issues and Solutions

#### Issue: CUDA Out of Memory

```
RuntimeError: CUDA out of memory. Tried to allocate X MiB
```

**Solutions:**
```yaml
# Option 1: Reduce batch size
training:
  batch_size: 16  # Reduce from 32

# Option 2: Reduce model size
embedding:
  base_emb_size: 24  # Reduce from 32
model:
  dim_feedforward: 96  # Reduce from 128
```

#### Issue: Loss Not Decreasing

**Possible causes:**
1. Learning rate too high or too low
2. Data preprocessing issue
3. Model too small for task

**Solutions:**
```yaml
# Try different learning rates
optimiser:
  lr: 0.0005  # or 0.002

# Increase model capacity
embedding:
  base_emb_size: 64
model:
  num_encoder_layers: 3
```

#### Issue: Validation Accuracy Decreasing (Overfitting)

**Solutions:**
```yaml
# Increase regularization
optimiser:
  weight_decay: 0.0001  # Increase from 1e-6
model:
  fc_dropout: 0.3  # Increase from 0.2

# Reduce patience
optimiser:
  patience: 3  # Stop earlier
```

### 6.3 Debug Mode

For quick testing of configuration:

```yaml
training:
  debug: true
```

This limits training to ~20 batches per epoch, useful for:
- Verifying configuration is valid
- Testing code changes
- Checking GPU memory usage

---

## 7. Hyperparameter Tuning

### 7.1 Tuning Priority Order

1. **Learning Rate** (most impact)
2. **Embedding Dimension**
3. **Number of Layers**
4. **Batch Size**
5. **Feedforward Dimension**
6. **Dropout**

### 7.2 Recommended Search Ranges

| Parameter | Search Range | Search Type |
|-----------|-------------|-------------|
| lr | [0.0001, 0.01] | Log scale |
| base_emb_size | [32, 64, 96, 128] | Categorical |
| num_encoder_layers | [1, 2, 3, 4] | Categorical |
| batch_size | [16, 32, 64, 128, 256] | Categorical |
| dim_feedforward | [64, 128, 256, 512] | Categorical |
| fc_dropout | [0.1, 0.2, 0.3] | Categorical |

### 7.3 Dataset-Specific Recommendations

**GeoLife (Small Dataset):**
```yaml
# Smaller models work better
embedding:
  base_emb_size: 32-64
model:
  num_encoder_layers: 1-2
  dim_feedforward: 64-128
training:
  batch_size: 16-32
```

**DIY (Large Dataset):**
```yaml
# Can support larger models
embedding:
  base_emb_size: 64-128
model:
  num_encoder_layers: 3-4
  dim_feedforward: 256-512
training:
  batch_size: 128-256
```

### 7.4 Parameter Budget Constraints

Keep within parameter limits:
- GeoLife: ~500K parameters max
- DIY: ~3M parameters max

**Estimate parameters:**
```
Params ≈ (num_locations × D) +           # Location embedding
          (24 + 4 + 7 + 96) × D +        # Temporal + duration
          N × (12 × D² + 8 × D × FF) +   # Encoder layers
          (num_users × D) +               # User embedding
          (D × num_locations)             # Output layer
```

---

## 8. Best Practices

### 8.1 Reproducibility

Always set seed at beginning:
```yaml
seed: 42
```

For full reproducibility:
```python
import torch
import random
import numpy as np

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
```

### 8.2 Experiment Naming

Use descriptive experiment names:
```
experiments/
├── geolife_MHSA_baseline_20260101/
├── geolife_MHSA_emb64_layers3_20260102/
├── geolife_MHSA_lr0.0005_20260103/
└── diy_MHSA_batch256_20260104/
```

### 8.3 Checkpointing Strategy

The training script saves:
- Best model (lowest validation loss)
- Configuration for reproducibility
- Full training log

### 8.4 Evaluation Protocol

1. **Always evaluate on test set** only after all tuning is complete
2. **Use validation set** for hyperparameter selection
3. **Report mean and std** if running multiple seeds
4. **Compare fairly**: same data splits, preprocessing

### 8.5 Resource Management

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Kill process if needed
pkill -f train_MHSA.py
```

---

## Appendix: Configuration Examples

### A.1 GeoLife Baseline

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
  momentum: 0.98
  num_warmup_epochs: 2
  num_training_epochs: 50
  patience: 5
  lr_step_size: 1
  lr_gamma: 0.1
```

### A.2 DIY Baseline

```yaml
seed: 42
data:
  data_dir: data/diy_eps50/processed
  dataset_prefix: diy_eps50_prev7
  dataset: diy
  experiment_root: experiments
training:
  if_embed_user: true
  if_embed_poi: false
  if_embed_time: true
  if_embed_duration: true
  previous_day: 7
  verbose: true
  debug: false
  batch_size: 256
  print_step: 10
  num_workers: 0
dataset_info:
  total_loc_num: 7038
  total_user_num: 693
embedding:
  base_emb_size: 96
  poi_original_size: 16
model:
  networkName: transformer
  num_encoder_layers: 4
  nhead: 8
  dim_feedforward: 256
  fc_dropout: 0.1
optimiser:
  optimizer: Adam
  max_epoch: 100
  lr: 0.001
  weight_decay: 0.000001
  beta1: 0.9
  beta2: 0.999
  momentum: 0.98
  num_warmup_epochs: 2
  num_training_epochs: 50
  patience: 5
  lr_step_size: 1
  lr_gamma: 0.1
```

---

*Training guide for MHSA model - next_loc_clean_v2 project*
