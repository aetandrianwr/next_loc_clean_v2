# Default Configuration Comparison

## Table of Contents
1. [Overview](#overview)
2. [Model Architecture Defaults](#model-architecture-defaults)
3. [Training Hyperparameters](#training-hyperparameters)
4. [Data Configuration](#data-configuration)
5. [Regularization Settings](#regularization-settings)
6. [Complete Configuration Files](#complete-configuration-files)
7. [Configuration Impact Analysis](#configuration-impact-analysis)

---

## Overview

This document provides a comprehensive side-by-side comparison of all default configuration values between the Original Pointer-Generator (TensorFlow) and the Proposed PointerNetworkV45 (PyTorch).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  CONFIGURATION PHILOSOPHY COMPARISON                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL (Text Summarization):                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Large vocabulary (50K words)                                      │   │
│  │  - Large hidden dimensions (256)                                     │   │
│  │  - Long sequences (400 encoder, 100 decoder)                        │   │
│  │  - Simple optimizer (Adagrad)                                        │   │
│  │  - High learning rate (0.15)                                        │   │
│  │  - Small batch size (16)                                            │   │
│  │  - Manual stopping (no early stopping)                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED (Next Location Prediction):                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Small vocabulary (~500 locations)                                │   │
│  │  - Smaller dimensions (64)                                          │   │
│  │  - Shorter sequences (50 window)                                    │   │
│  │  - Modern optimizer (AdamW)                                         │   │
│  │  - Low learning rate (6.5e-4)                                       │   │
│  │  - Larger batch size (128)                                          │   │
│  │  - Automatic early stopping (patience=5)                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Model Architecture Defaults

### Complete Architecture Comparison Table

| Parameter | Original | Proposed | Ratio | Notes |
|-----------|----------|----------|-------|-------|
| **Vocabulary/Location Size** | 50,000 | ~500 | 100x | Task-dependent |
| **Embedding Dimension** | 128 | 64 | 2x | Original has word embeddings only |
| **Hidden Dimension** | 256 | 64 | 4x | Proposed uses d_model |
| **Encoder Type** | BiLSTM | Transformer | - | Fundamental change |
| **Encoder Layers** | 1 | 2 | - | Transformer layers |
| **Encoder Heads** | N/A | 4 | - | Multi-head attention |
| **Feed-Forward Dim** | N/A | 128 | - | Transformer FFN |
| **Decoder Type** | LSTM | None | - | No sequential decoding |
| **Decoder Layers** | 1 | N/A | - | No decoder |
| **Attention Type** | Bahdanau | Scaled Dot-Product | - | Different mechanism |
| **Attention Dim** | 256 | 64 | 4x | Smaller in proposed |
| **Pointer Mechanism** | p_gen | gate | - | Inverted semantics |
| **Coverage** | Optional | No | - | Not applicable |

### Detailed Dimension Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIMENSION COMPARISON                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL DIMENSIONS:                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Input:                                                               │   │
│  │    Word IDs → Embedding(50000, 128) → [batch, seq, 128]             │   │
│  │                                                                       │   │
│  │  Encoder (BiLSTM):                                                   │   │
│  │    Hidden: 256 × 2 directions = 512 → [batch, seq, 512]             │   │
│  │    State: (256, 256) → reduced to (256, 256)                        │   │
│  │                                                                       │   │
│  │  Decoder (LSTM):                                                     │   │
│  │    Hidden: 256 → [batch, 256]                                       │   │
│  │    Output: 256 + 512 = 768 → vocab projection                       │   │
│  │                                                                       │   │
│  │  Attention:                                                          │   │
│  │    Query (s_t): [batch, 256]                                        │   │
│  │    Keys (h_i): [batch, seq, 512]                                    │   │
│  │    Weights: W_h (512→256), W_s (256→256), v (256→1)                │   │
│  │                                                                       │   │
│  │  Output:                                                             │   │
│  │    Vocab dist: [batch, 50000]                                       │   │
│  │    Attention: [batch, seq]                                          │   │
│  │    Final: [batch, 50000 + max_oov]                                  │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED DIMENSIONS:                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Input:                                                               │   │
│  │    Location IDs → Embedding(500, 64) → [batch, seq, 64]             │   │
│  │    + 6 other embeddings → combined [batch, seq, 64]                 │   │
│  │                                                                       │   │
│  │  Encoder (Transformer):                                              │   │
│  │    d_model: 64                                                      │   │
│  │    nhead: 4 (head_dim = 64/4 = 16)                                  │   │
│  │    dim_feedforward: 128                                              │   │
│  │    Output: [batch, seq, 64]                                         │   │
│  │                                                                       │   │
│  │  No Decoder (single-step prediction):                               │   │
│  │    Query: learned [1, 64] or last position                          │   │
│  │                                                                       │   │
│  │  Attention:                                                          │   │
│  │    Query: [batch, 64]                                               │   │
│  │    Keys: [batch, seq, 64]                                           │   │
│  │    Output: context [batch, 64] + attention [batch, seq]             │   │
│  │                                                                       │   │
│  │  Output:                                                             │   │
│  │    Generate dist: [batch, 500]                                      │   │
│  │    Pointer dist: [batch, seq] → scattered to [batch, 500]          │   │
│  │    Final: [batch, 500]                                              │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Hyperparameters

### Complete Training Configuration Table

| Parameter | Original | Proposed | Notes |
|-----------|----------|----------|-------|
| **Optimizer** | Adagrad | AdamW | Modern optimizer |
| **Base Learning Rate** | 0.15 | 6.5e-4 | 230x lower |
| **Min Learning Rate** | N/A | 1e-6 | For cosine decay |
| **LR Schedule** | None (Adagrad adaptive) | Warmup + Cosine | Explicit scheduling |
| **Warmup Epochs** | N/A | 5 | Linear warmup |
| **Weight Decay** | None | 0.015 | AdamW regularization |
| **Momentum (β₁)** | N/A | 0.9 | Adam first moment |
| **β₂** | N/A | 0.98 | Adam second moment |
| **Epsilon** | N/A | 1e-9 | Adam stability |
| **Adagrad Init Acc** | 0.1 | N/A | Initial accumulator |
| **Gradient Clip (max norm)** | 2.0 | 0.8 | 2.5x lower |
| **Batch Size** | 16 | 128 | 8x larger |
| **Max Epochs** | ∞ (manual stop) | 50 | Explicit limit |
| **Early Stopping** | No | Yes | Automatic |
| **Patience** | N/A | 5 | Epochs without improvement |
| **Min Epochs** | N/A | 8 | Before early stop allowed |
| **Mixed Precision (AMP)** | No | Yes | Faster training |

### Learning Rate Schedule Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LEARNING RATE COMPARISON                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL: Adagrad with Fixed Base LR                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LR                                                                   │   │
│  │  ↑                                                                    │   │
│  │  0.15 ●────────────────────────────────────────────────              │   │
│  │       │   (Base LR stays constant, but effective LR                  │   │
│  │       │    decreases as Adagrad accumulates gradients)               │   │
│  │       │                                                               │   │
│  │       │         Effective LR ≈ 0.15 / √(accumulated_grad²)          │   │
│  │       │         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~          │   │
│  │       │                                                       ▼       │   │
│  │  0    └───────────────────────────────────────────────────────→      │   │
│  │       0                                                  Steps        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED: Warmup + Cosine Decay                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  LR                                                                   │   │
│  │  ↑                                                                    │   │
│  │  6.5e-4    ╭───────╮ Peak                                            │   │
│  │           ╱         ╲                                                 │   │
│  │          ╱           ╲    Cosine Decay                               │   │
│  │         ╱             ╲                                               │   │
│  │        ╱               ╲                                              │   │
│  │       ╱  Warmup         ╲                                             │   │
│  │      ╱   (5 epochs)       ╲                                           │   │
│  │     ╱                       ╲                                         │   │
│  │    ╱                         ╲______ min (1e-6)                      │   │
│  │  0└───────────────────────────────────────────────────────→          │   │
│  │    0      5                                            50 Epochs     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Configuration

### Data Processing Defaults

| Parameter | Original | Proposed | Notes |
|-----------|----------|----------|-------|
| **Max Encoder Steps** | 400 | N/A | No encoder step limit |
| **Max Decoder Steps** | 100 | N/A | No decoder |
| **Window Size** | N/A | 50 | Fixed input length |
| **Vocabulary Size** | 50,000 | ~500 | Task-dependent |
| **Min Input Length** | N/A | N/A | Handled by padding |
| **Truncation** | Yes (to max steps) | Yes (to window) | Different approach |
| **Padding** | Dynamic per-batch | Fixed (window_size) | Different approach |
| **OOV Handling** | Extended vocab | Not needed | Closed vocabulary |
| **Batch Queue Size** | 100 | N/A | PyTorch handles |
| **Example Queue Size** | 100 × batch | N/A | PyTorch handles |
| **Num Reader Threads** | 16 | 0 | PyTorch DataLoader |
| **Bucketing Cache** | 100 | N/A | Not needed |

### Sequence Length Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE LENGTH HANDLING                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL:                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Encoder: Up to 400 tokens                                           │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │ w1  w2  w3  ...  w398  w399  w400 │ truncate if longer        │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                                                                       │   │
│  │  Decoder: Up to 100 tokens                                           │   │
│  │  ┌─────────────────────────────────┐                                 │   │
│  │  │ <START>  s1  s2  ...  s99 <STOP>│                                │   │
│  │  └─────────────────────────────────┘                                 │   │
│  │                                                                       │   │
│  │  Padding: Within batch, pad to max length in batch                  │   │
│  │  Example batch with lengths [50, 120, 89]:                          │   │
│  │    All padded to 120                                                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED:                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Input: Fixed window_size = 50                                      │   │
│  │                                                                       │   │
│  │  Short sequence (3 locations):                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ 0  0  0  0  ... 0  0  0  loc1  loc2  loc3 │ (47 zeros + 3 locs)│ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  │  Long sequence (75 locations):                                       │   │
│  │  ┌────────────────────────────────────────────────────────────────┐ │   │
│  │  │ loc26  loc27  loc28  ...  loc74  loc75 │ (last 50 kept)        │ │   │
│  │  └────────────────────────────────────────────────────────────────┘ │   │
│  │                                                                       │   │
│  │  Output: Single prediction (no sequence)                            │   │
│  │  ┌───────┐                                                           │   │
│  │  │next_loc│                                                          │   │
│  │  └───────┘                                                           │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Regularization Settings

### Regularization Comparison Table

| Technique | Original | Proposed | Notes |
|-----------|----------|----------|-------|
| **Dropout** | 0.0 | 0.15 | Applied in Transformer |
| **Weight Decay** | None | 0.015 | In AdamW |
| **Label Smoothing** | None | 0.03 | In CrossEntropyLoss |
| **Gradient Clipping** | 2.0 | 0.8 | More aggressive |
| **Coverage Loss** | λ=1.0 (optional) | N/A | Not applicable |
| **Embedding Dropout** | None | Inherits 0.15 | Via Transformer |
| **Attention Dropout** | None | 0.15 | In TransformerEncoder |

### Regularization Effects

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    REGULARIZATION TECHNIQUES                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL: Minimal Regularization                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Primary regularization: Coverage loss (optional)                    │   │
│  │    - Penalizes repeated attention                                   │   │
│  │    - Prevents repetitive outputs                                    │   │
│  │                                                                       │   │
│  │  No dropout, no weight decay, no label smoothing                    │   │
│  │                                                                       │   │
│  │  Why?                                                                │   │
│  │    - Large dataset (millions of examples)                           │   │
│  │    - Long training (models train for days)                          │   │
│  │    - Adagrad provides implicit regularization via LR decay          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED: Multiple Regularization Techniques                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  1. Dropout (0.15):                                                 │   │
│  │     - Applied in Transformer layers                                 │   │
│  │     - Prevents co-adaptation of features                            │   │
│  │                                                                       │   │
│  │  2. Weight Decay (0.015):                                           │   │
│  │     - L2 regularization in AdamW                                    │   │
│  │     - Prevents large weights                                        │   │
│  │                                                                       │   │
│  │  3. Label Smoothing (0.03):                                         │   │
│  │     - Softens hard labels                                           │   │
│  │     - Prevents overconfident predictions                            │   │
│  │                                                                       │   │
│  │  4. Early Stopping (patience=5):                                    │   │
│  │     - Stops before overfitting                                      │   │
│  │     - Based on validation loss                                      │   │
│  │                                                                       │   │
│  │  Why more regularization?                                           │   │
│  │    - Smaller dataset (mobility data limited)                        │   │
│  │    - Shorter training (early stopping)                              │   │
│  │    - Transformers prone to overfitting on small data               │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Configuration Files

### Original: Command-Line Flags (run_summarization.py)

```python
# File: run_summarization.py, lines 26-68

# Model hyperparameters
tf.app.flags.DEFINE_string('mode', 'train', 'train/eval/decode')
tf.app.flags.DEFINE_string('data_path', '', 'Path to tf.Example datafiles')
tf.app.flags.DEFINE_string('vocab_path', '', 'Path to vocabulary file')
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for logs')
tf.app.flags.DEFINE_string('exp_name', '', 'Name of experiment')

# Hyperparameters
tf.app.flags.DEFINE_integer('hidden_dim', 256, 'dimension of RNN hidden states')
tf.app.flags.DEFINE_integer('emb_dim', 128, 'dimension of word embeddings')
tf.app.flags.DEFINE_integer('batch_size', 16, 'minibatch size')
tf.app.flags.DEFINE_integer('max_enc_steps', 400, 'max timesteps of encoder')
tf.app.flags.DEFINE_integer('max_dec_steps', 100, 'max timesteps of decoder')
tf.app.flags.DEFINE_integer('beam_size', 4, 'beam size for decoding')
tf.app.flags.DEFINE_integer('min_dec_steps', 35, 'min timesteps of decoder')
tf.app.flags.DEFINE_integer('vocab_size', 50000, 'vocabulary size')
tf.app.flags.DEFINE_float('lr', 0.15, 'learning rate')
tf.app.flags.DEFINE_float('adagrad_init_acc', 0.1, 'initial accumulator value')
tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 'uniform init magnitude')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 'std of trunc norm init')
tf.app.flags.DEFINE_float('max_grad_norm', 2.0, 'for gradient clipping')

# Pointer-generator parameters
tf.app.flags.DEFINE_boolean('pointer_gen', True, 'use pointer-generator model')
tf.app.flags.DEFINE_boolean('coverage', False, 'use coverage mechanism')
tf.app.flags.DEFINE_float('cov_loss_wt', 1.0, 'coverage loss weight')

# Utility flags
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', False, 'convert to coverage')
tf.app.flags.DEFINE_boolean('restore_best_model', False, 'restore best checkpoint')
tf.app.flags.DEFINE_boolean('debug', False, 'run in TensorFlow debug mode')
tf.app.flags.DEFINE_boolean('single_pass', False, 'single pass through data')

# Original defaults summary:
ORIGINAL_DEFAULTS = {
    'hidden_dim': 256,
    'emb_dim': 128,
    'batch_size': 16,
    'max_enc_steps': 400,
    'max_dec_steps': 100,
    'beam_size': 4,
    'min_dec_steps': 35,
    'vocab_size': 50000,
    'lr': 0.15,
    'adagrad_init_acc': 0.1,
    'rand_unif_init_mag': 0.02,
    'trunc_norm_init_std': 1e-4,
    'max_grad_norm': 2.0,
    'pointer_gen': True,
    'coverage': False,
    'cov_loss_wt': 1.0,
}
```

### Proposed: YAML Configuration File

```yaml
# File: config/models/config_pointer_v45_geolife.yaml

# Model Architecture
model:
  name: "PointerNetworkV45"
  
  # Dimensions
  d_model: 64
  nhead: 4
  num_layers: 2
  dim_feedforward: 128
  dropout: 0.15
  
  # Vocabulary sizes
  num_locations: 500  # Dataset-dependent
  num_users: 200      # Dataset-dependent
  num_time_slots: 24  # Hours in a day
  num_weekdays: 7     # Days in a week
  num_duration_bins: 20
  num_recency_bins: 20
  
  # Feature flags
  use_time_embedding: true
  use_user_embedding: true
  use_weekday_embedding: true
  use_duration_embedding: true
  use_recency_embedding: true
  use_position_from_end: true

# Training Configuration
training:
  batch_size: 128
  num_epochs: 50
  
  # Optimizer
  optimizer: "adamw"
  lr: 0.00065  # 6.5e-4
  min_lr: 0.000001  # 1e-6
  weight_decay: 0.015
  betas: [0.9, 0.98]
  eps: 0.000000001  # 1e-9
  
  # Learning rate schedule
  warmup_epochs: 5
  lr_schedule: "cosine"
  
  # Gradient clipping
  grad_clip: 0.8
  
  # Early stopping
  patience: 5
  min_epochs: 8
  
  # Mixed precision
  use_amp: true
  
  # Loss
  label_smoothing: 0.03

# Data Configuration
data:
  path: "data/geolife/processed"
  window_size: 50
  train_split: "train.pkl"
  val_split: "val.pkl"
  test_split: "test.pkl"

# Experiment
experiment:
  seed: 42
  device: "cuda"
  output_dir: "experiments"
  save_best: true
  save_last: true
  log_interval: 10

# Proposed defaults summary:
PROPOSED_DEFAULTS = {
    'd_model': 64,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 128,
    'dropout': 0.15,
    'batch_size': 128,
    'num_epochs': 50,
    'lr': 0.00065,
    'min_lr': 0.000001,
    'weight_decay': 0.015,
    'warmup_epochs': 5,
    'grad_clip': 0.8,
    'patience': 5,
    'min_epochs': 8,
    'label_smoothing': 0.03,
    'window_size': 50,
    'seed': 42,
    'use_amp': True,
}
```

---

## Configuration Impact Analysis

### Parameter Impact on Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER IMPACT ANALYSIS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  HIDDEN DIMENSION: 256 → 64                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Impact:                                                              │   │
│  │    - 16x fewer parameters per layer                                  │   │
│  │    - Faster training and inference                                   │   │
│  │    - Lower memory usage                                              │   │
│  │                                                                       │   │
│  │  Justification:                                                      │   │
│  │    - Location prediction simpler than text generation               │   │
│  │    - Fewer unique patterns to learn                                 │   │
│  │    - Smaller dataset requires smaller model                         │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  BATCH SIZE: 16 → 128                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Impact:                                                              │   │
│  │    - 8x more samples per gradient update                            │   │
│  │    - More stable gradients                                          │   │
│  │    - Better GPU utilization                                         │   │
│  │                                                                       │   │
│  │  Justification:                                                      │   │
│  │    - Smaller model = smaller memory per sample                      │   │
│  │    - Fixed sequence length = predictable memory                     │   │
│  │    - Modern GPUs have more memory                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  LEARNING RATE: 0.15 → 6.5e-4                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Impact:                                                              │   │
│  │    - 230x lower base learning rate                                  │   │
│  │    - Slower but more stable convergence                             │   │
│  │    - Less risk of overshooting optima                               │   │
│  │                                                                       │   │
│  │  Justification:                                                      │   │
│  │    - Adam/AdamW works best with lower LR than Adagrad              │   │
│  │    - Transformers are sensitive to high LR                         │   │
│  │    - Larger batch size allows lower LR                              │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  GRADIENT CLIPPING: 2.0 → 0.8                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Impact:                                                              │   │
│  │    - More aggressive clipping                                       │   │
│  │    - Prevents gradient explosions                                   │   │
│  │    - Smoother training                                              │   │
│  │                                                                       │   │
│  │  Justification:                                                      │   │
│  │    - Transformers can have large gradients                          │   │
│  │    - Lower max norm = more conservative updates                     │   │
│  │    - Combined with warmup for stability                             │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Parameter Count Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER COUNT BREAKDOWN                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL (~29M parameters):                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Embeddings:     50,000 × 128 = 6,400,000                           │   │
│  │                                                                       │   │
│  │  Encoder BiLSTM: 4 × (128 × 256 + 256 × 256 + 256) × 2              │   │
│  │                  ≈ 1,050,000                                         │   │
│  │                                                                       │   │
│  │  State Reduce:   512 × 256 + 256 + 512 × 256 + 256 ≈ 262,000       │   │
│  │                                                                       │   │
│  │  Decoder LSTM:   4 × (768 × 256 + 256 × 256 + 256)                  │   │
│  │                  ≈ 1,050,000                                         │   │
│  │                                                                       │   │
│  │  Attention:      512 × 256 + 256 × 256 + 256 × 1 ≈ 196,000         │   │
│  │                                                                       │   │
│  │  Output Project: (256 + 512) × 50,000 ≈ 38,400,000                  │   │
│  │                                                                       │   │
│  │  p_gen MLP:      768 + 256 + 128 → 1 ≈ 1,200                        │   │
│  │                                                                       │   │
│  │  TOTAL: ~47M parameters (dominated by output projection)            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED (~154K parameters):                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Embeddings (7 types):                                               │   │
│  │    Location:     500 × 64 = 32,000                                  │   │
│  │    User:         200 × 64 = 12,800                                  │   │
│  │    Time:         24 × 64 = 1,536                                    │   │
│  │    Weekday:      7 × 64 = 448                                       │   │
│  │    Duration:     20 × 64 = 1,280                                    │   │
│  │    Recency:      20 × 64 = 1,280                                    │   │
│  │    PosFromEnd:   100 × 64 = 6,400                                   │   │
│  │                  Subtotal: ≈ 56,000                                  │   │
│  │                                                                       │   │
│  │  Transformer (2 layers):                                             │   │
│  │    Self-Attention: 4 × 64 × 64 × 3 = 49,152 per layer              │   │
│  │    FFN: 64 × 128 + 128 × 64 = 16,384 per layer                     │   │
│  │    LayerNorms: 64 × 4 = 256 per layer                               │   │
│  │                  Subtotal: 2 × 65,792 ≈ 132,000                     │   │
│  │                                                                       │   │
│  │  Pointer Attention: 64 × 64 × 3 ≈ 12,288                            │   │
│  │                                                                       │   │
│  │  Generation Head: 64 × 500 ≈ 32,000                                 │   │
│  │                                                                       │   │
│  │  Gate MLP: 64 → 64 → 1 ≈ 4,200                                      │   │
│  │                                                                       │   │
│  │  TOTAL: ~237K parameters                                            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  RATIO: Original/Proposed ≈ 47M/237K ≈ 200x more parameters               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary Table

| Category | Parameter | Original | Proposed |
|----------|-----------|----------|----------|
| **Architecture** | Hidden dim | 256 | 64 |
| | Embedding dim | 128 | 64 |
| | Encoder type | BiLSTM | Transformer |
| | Encoder layers | 1 | 2 |
| | Attention heads | 1 | 4 |
| | Decoder | LSTM | None |
| | Total params | ~47M | ~237K |
| **Training** | Optimizer | Adagrad | AdamW |
| | Learning rate | 0.15 | 6.5e-4 |
| | Batch size | 16 | 128 |
| | Gradient clip | 2.0 | 0.8 |
| | Weight decay | 0 | 0.015 |
| | LR schedule | None | Warmup+Cosine |
| | Early stopping | Manual | Automatic |
| **Regularization** | Dropout | 0 | 0.15 |
| | Label smoothing | 0 | 0.03 |
| | Coverage loss | Optional | None |
| **Data** | Vocab size | 50,000 | ~500 |
| | Max enc steps | 400 | 50 (window) |
| | Max dec steps | 100 | 1 |
| | Padding | Dynamic | Fixed |

---

*Next: [11_CODE_WALKTHROUGH_PROPOSED.md](11_CODE_WALKTHROUGH_PROPOSED.md) - Line-by-line analysis of the proposed model*
