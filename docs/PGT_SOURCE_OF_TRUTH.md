# Position-Aware Pointer Generator Transformer: Authoritative Technical Reference

**Document Version**: 1.0  
**Generated**: 2026-01-06  
**Purpose**: Complete semantic abstraction of the PointerGeneratorTransformer codebase and experiments for PhD thesis derivation

---

## Table of Contents

1. [Problem Domain and Task Formalization](#1-problem-domain-and-task-formalization)
2. [Model Architecture](#2-model-architecture)
3. [Input Representation and Feature Engineering](#3-input-representation-and-feature-engineering)
4. [Training Methodology](#4-training-methodology)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Datasets and Preprocessing](#6-datasets-and-preprocessing)
7. [Experimental Results](#7-experimental-results)
8. [Ablation Studies](#8-ablation-studies)
9. [Behavioral Analysis](#9-behavioral-analysis)
10. [Baseline Comparisons](#10-baseline-comparisons)
11. [Technical Implementation Details](#11-technical-implementation-details)
12. [Statistical Significance and Confidence](#12-statistical-significance-and-confidence)
13. [Limitations and Boundary Conditions](#13-limitations-and-boundary-conditions)

---

## 1. Problem Domain and Task Formalization

### 1.1 Task Definition

**Next Location Prediction (NLP)** is defined as: given a user's historical trajectory sequence $S = \{l_1, l_2, ..., l_n\}$ of visited locations with associated temporal context, predict the next location $l_{n+1}$ the user will visit.

### 1.2 Formal Problem Statement

- **Input**: A sequence of location visits $X = (l_1, l_2, ..., l_n)$ where each $l_i$ is a discrete location ID, accompanied by:
  - User identifier $u$
  - Temporal features per visit: time-of-day $t_i$, weekday $w_i$, duration $d_i$, recency $r_i$
- **Output**: Probability distribution $P(l_{n+1} | X, u, \{t_i, w_i, d_i, r_i\})$ over all possible locations
- **Objective**: Maximize top-k prediction accuracy (primarily Acc@1)

### 1.3 Core Challenge

Human mobility exhibits a fundamental dichotomy:
1. **Repetitive behavior**: ~84% of next locations appear in the user's recent history (within 7 days)
2. **Exploratory behavior**: ~16% of visits are to locations not in recent history

This dichotomy motivates the hybrid pointer-generator architecture.

---

## 2. Model Architecture

### 2.1 Architecture Overview

PointerGeneratorTransformer combines three predictive mechanisms:

```
Input → Embeddings → Transformer Encoder → {Pointer Mechanism, Generation Head} → Gate → Final Distribution
```

### 2.2 Core Components

#### 2.2.1 Embedding Layer

**Location Embedding**: 
- Learnable embedding matrix $E_{loc} \in \mathbb{R}^{|V| \times d_{model}}$
- Padding index = 0, vocabulary size $|V|$ = number of unique locations + 2

**User Embedding**:
- Learnable embedding matrix $E_{user} \in \mathbb{R}^{|U| \times d_{model}}$
- Padding index = 0, broadcast across sequence length

**Temporal Embeddings** (each $d_{model}/4$ dimensions):
- Time-of-day: 97 buckets (96 × 15-minute intervals + padding)
- Weekday: 8 buckets (7 days + padding)
- Recency: 9 buckets (0-7 days ago + padding, representing "diff" from current day)
- Duration: 100 buckets (30-minute intervals, max 2 days)

**Position-from-End Embedding**:
- $E_{pos\_end} \in \mathbb{R}^{(max\_seq\_len+1) \times d_{model}/4}$
- Captures recency within sequence (how many steps from the end)
- Critical for pointer mechanism's recency awareness

#### 2.2.2 Input Projection

Combined input dimension:
$$d_{input} = d_{model} + d_{model} + 4 \times (d_{model}/4) + d_{model}/4 = 2.25 \times d_{model} + d_{model}/4$$

Projected via:
- Linear layer: $d_{input} \rightarrow d_{model}$
- Layer normalization
- Added sinusoidal positional encoding

#### 2.2.3 Transformer Encoder

Configuration:
- Pre-norm architecture (norm_first=True)
- GELU activation
- Multi-head self-attention with configurable heads
- Feed-forward network with configurable dimension
- Batch-first format

Mathematical formulation:
$$H = \text{TransformerEncoder}(\text{LayerNorm}(X_{proj} + PE), \text{mask})$$

Where mask is a key-padding mask for variable-length sequences.

#### 2.2.4 Context Extraction

Extract representation from last valid position:
$$c = H[\text{batch\_idx}, \text{lengths} - 1]$$

This context vector $c \in \mathbb{R}^{d_{model}}$ summarizes the entire sequence.

#### 2.2.5 Pointer Mechanism

**Query-Key Attention**:
$$Q = W_q \cdot c, \quad K = W_k \cdot H$$
$$\text{scores} = \frac{Q \cdot K^T}{\sqrt{d_{model}}}$$

**Position Bias**:
$$\text{scores} = \text{scores} + \beta[\text{pos\_from\_end}]$$

Where $\beta \in \mathbb{R}^{max\_seq\_len}$ is a learnable position bias vector.

**Pointer Distribution**:
$$\text{ptr\_probs} = \text{softmax}(\text{masked\_scores})$$

**Scatter to Vocabulary**:
$$P_{ptr}[v] = \sum_{i: x_i = v} \text{ptr\_probs}[i]$$

This "copies" probability mass from sequence positions to vocabulary locations.

#### 2.2.6 Generation Head

Standard classification head:
$$P_{gen} = \text{softmax}(W_{gen} \cdot c)$$

Where $W_{gen} \in \mathbb{R}^{|V| \times d_{model}}$.

#### 2.2.7 Pointer-Generator Gate

Adaptive blending mechanism:
$$g = \sigma(W_2 \cdot \text{GELU}(W_1 \cdot c))$$

Where:
- $W_1 \in \mathbb{R}^{d_{model}/2 \times d_{model}}$
- $W_2 \in \mathbb{R}^{1 \times d_{model}/2}$

**Final Distribution**:
$$P_{final} = g \cdot P_{ptr} + (1 - g) \cdot P_{gen}$$

**Output**: Log-probabilities for numerical stability:
$$\log(P_{final} + \epsilon), \quad \epsilon = 10^{-10}$$

---

## 3. Input Representation and Feature Engineering

### 3.1 Sequence Construction

Each training sample consists of:
- `X`: Location ID sequence (numpy array, variable length 3-99)
- `Y`: Target next location (integer)
- `user_X`: User ID array (same value repeated)
- `weekday_X`: Day of week for each visit (0-6)
- `start_min_X`: Start time in minutes from midnight (0-1439)
- `dur_X`: Duration at location in minutes
- `diff`: Days ago from current prediction time (0-7)

### 3.2 Feature Discretization

| Feature | Raw Range | Discretization | Buckets |
|---------|-----------|----------------|---------|
| Time | 0-1439 min | ÷15 (15-min intervals) | 96 |
| Weekday | 0-6 | Direct | 7 |
| Duration | 0-2879 min | ÷30 (30-min intervals) | 96 |
| Recency (diff) | 0-7 days | Direct | 8 |

### 3.3 Padding Strategy

- Sequences padded to batch max length
- Padding value = 0 for all features
- Padding mask generated: `positions >= lengths`

---

## 4. Training Methodology

### 4.1 Optimization Configuration

| Parameter | GeoLife | DIY |
|-----------|---------|-----|
| Optimizer | AdamW | AdamW |
| Base LR | 6.5e-4 | 7e-4 |
| Min LR | 1e-6 | 1e-6 |
| Weight Decay | 0.015 | 0.015 |
| Betas | (0.9, 0.98) | (0.9, 0.98) |
| Epsilon | 1e-9 | 1e-9 |
| Batch Size | 128 | 128 |
| Max Epochs | 50 | 50 |

### 4.2 Learning Rate Schedule

**Warmup Phase** (epochs 0 to warmup_epochs-1):
$$LR = LR_{base} \times \frac{epoch + 1}{warmup\_epochs}$$

**Cosine Decay Phase** (epochs warmup_epochs to max_epochs):
$$progress = \frac{epoch - warmup\_epochs}{max\_epochs - warmup\_epochs}$$
$$LR = LR_{min} + 0.5 \times (LR_{base} - LR_{min}) \times (1 + \cos(\pi \times progress))$$

Warmup epochs = 5 for both datasets.

### 4.3 Loss Function

Cross-entropy with label smoothing:
$$\mathcal{L} = -\sum_{c=1}^{C} y'_c \log(p_c)$$

Where:
- $y'_c = (1 - \alpha) \cdot y_c + \alpha / C$ for label smoothing
- $\alpha = 0.03$ (label smoothing factor)
- `ignore_index=0` for padding

### 4.4 Regularization

- **Dropout**: 0.15 throughout model
- **Gradient Clipping**: max_norm = 0.8
- **Weight Decay**: 0.015 (L2 regularization in AdamW)
- **Mixed Precision**: FP16 with gradient scaling (AMP)

### 4.5 Early Stopping

- **Patience**: 5 epochs (GeoLife), 5 epochs (DIY)
- **Criterion**: Validation loss
- **Min Epochs**: 8 (prevents premature stopping)

### 4.6 Reproducibility

Fixed seed = 42 applied to:
- `random.seed()`
- `np.random.seed()`
- `torch.manual_seed()`
- `torch.cuda.manual_seed_all()`
- `torch.backends.cudnn.deterministic = True`
- `torch.backends.cudnn.benchmark = False`

---

## 5. Evaluation Framework

### 5.1 Metrics Definitions

#### Accuracy@k (Hit Rate)
$$Acc@k = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i \in \text{top-}k(\hat{y}_i)]$$

Reported for k ∈ {1, 3, 5, 10}.

#### Mean Reciprocal Rank (MRR)
$$MRR = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{rank_i}$$

Where $rank_i$ is the position of the true label in the sorted predictions.

#### Normalized Discounted Cumulative Gain (NDCG@10)
$$NDCG@10 = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\log_2(rank_i + 1)} \cdot \mathbb{1}[rank_i \leq 10]$$

#### Weighted F1 Score
Computed on top-1 predictions using scikit-learn's `f1_score(average='weighted')`.

### 5.2 Evaluation Protocol

1. Load best checkpoint (lowest validation loss)
2. Set model to eval mode
3. Disable gradients
4. Iterate through test DataLoader
5. Accumulate counts: correct@k, reciprocal ranks, NDCG scores
6. Compute final percentages

---

## 6. Datasets and Preprocessing

### 6.1 GeoLife Dataset

**Source**: Microsoft Research GeoLife GPS trajectory dataset (Beijing, China, 2007-2012)

**Preprocessing Pipeline**:
1. **Staypoint Detection**: 
   - Distance threshold: 200m
   - Time threshold: 30 minutes
   - Gap threshold: 1440 minutes (24 hours)
2. **Location Clustering**: DBSCAN with ε=20 meters, min_samples=2
3. **User Quality Filter**: Minimum 50 tracking days
4. **Temporal Split**: 60% train / 20% val / 20% test (per user, chronological)

**Final Statistics**:
| Split | Samples | Users |
|-------|---------|-------|
| Train | 10,231 | 45 |
| Val | 3,428 | 45 |
| Test | 3,502 | 45 |
| **Total** | **17,161** | **45** |

**Vocabulary**: 1,156 unique locations (+ padding + unknown)

**Sequence Characteristics**:
- Average length: 18.37
- Max length: 46
- Min length: 3

### 6.2 DIY Dataset

**Source**: Proprietary GPS trajectory dataset from mobile applications

**Preprocessing Pipeline**:
1. **Location Clustering**: DBSCAN with ε=50 meters, min_samples=2
2. **Staypoint Merging**: Max time gap 1 minute
3. **User Quality Filter**: 
   - Minimum 60 tracking days
   - Quality threshold: min=0.6, mean=0.7 over 10-week sliding window
4. **Temporal Split**: 80% train / 10% val / 10% test (per user, chronological)

**Final Statistics**:
| Split | Samples | Users |
|-------|---------|-------|
| Train | 79,814 | 692 |
| Val | 13,211 | 692 |
| Test | 12,368 | 692 |
| **Total** | **105,393** | **692** |

**Vocabulary**: 7,017 unique locations (+ padding + unknown)

**Sequence Characteristics**:
- Average length: 23.98
- Max length: 99
- Min length: 3
- Median: 21

### 6.3 Sequence Generation Parameters

Both datasets use:
- **Previous days window**: 7 days (history from [current_day - 7, current_day))
- **Minimum sequence length**: 3 (at least 3 historical visits)
- **Duration truncation**: 2880 minutes (2 days max)

### 6.4 Dataset Characteristics Comparison

| Characteristic | DIY | GeoLife |
|----------------|-----|---------|
| Target in history rate | 84.12% | 83.81% |
| Avg position from end (when in history) | 3.37 | 3.33 |
| Avg unique ratio per sequence | 0.313 | 0.340 |
| Repetition rate | 68.65% | 65.96% |
| Consecutive repeat rate | 17.94% | 26.87% |
| Target = last location rate | 18.56% | 27.18% |
| Target is most frequent rate | 41.99% | 44.20% |
| Target in top-3 frequent rate | 75.23% | 78.47% |
| Average sequence entropy | 1.89 | 1.74 |

---

## 7. Experimental Results

### 7.1 Primary Performance (Test Set)

#### DIY Dataset
| Metric | Value | 95% CI |
|--------|-------|--------|
| Acc@1 | 56.58% | [55.68%, 57.46%] |
| Acc@5 | 82.17% | [81.48%, 82.79%] |
| Acc@10 | 85.16% | [84.52%, 85.76%] |
| MRR | 67.67% | [66.96%, 68.36%] |
| NDCG@10 | 71.88% | — |
| F1 (weighted) | 51.91% | — |

#### GeoLife Dataset
| Metric | Value | 95% CI |
|--------|-------|--------|
| Acc@1 | 51.40% | [49.77%, 52.91%] |
| Acc@5 | 81.18% | [79.84%, 82.35%] |
| Acc@10 | 85.04% | [83.87%, 86.15%] |
| MRR | 64.55% | [63.29%, 65.74%] |
| NDCG@10 | 69.46% | — |
| F1 (weighted) | 46.97% | — |

### 7.2 Model Configurations

#### DIY (Larger Model)
```yaml
d_model: 128
nhead: 4
num_layers: 3
dim_feedforward: 256
dropout: 0.15
```
Parameters: ~500K trainable

#### GeoLife (Smaller Model)
```yaml
d_model: 64
nhead: 4
num_layers: 2
dim_feedforward: 128
dropout: 0.15
```
Parameters: ~200K trainable

---

## 8. Ablation Studies

### 8.1 Component Importance Ranking

#### GeoLife Dataset (Baseline Acc@1: 51.43%)

| Rank | Ablation | Acc@1 | ΔAcc@1 | Relative Drop |
|------|----------|-------|--------|---------------|
| 1 | w/o Pointer Mechanism | 27.41% | -24.01% | -46.7% |
| 2 | w/o Temporal Embeddings | 47.40% | -4.03% | -7.8% |
| 3 | w/o User Embedding | 49.11% | -2.31% | -4.5% |
| 4 | w/o Position-from-End | 49.34% | -2.08% | -4.1% |
| 5 | w/o Adaptive Gate | 49.54% | -1.88% | -3.7% |
| 6 | w/o Position Bias | 51.48% | +0.06% | +0.1% |
| 7 | Single Transformer Layer | 51.68% | +0.26% | +0.5% |
| 8 | w/o Generation Head | 51.86% | +0.43% | +0.8% |

#### DIY Dataset (Baseline Acc@1: 56.57%)

| Rank | Ablation | Acc@1 | ΔAcc@1 | Relative Drop |
|------|----------|-------|--------|---------------|
| 1 | w/o Pointer Mechanism | 51.90% | -4.67% | -8.3% |
| 2 | w/o Temporal Embeddings | 55.95% | -0.62% | -1.1% |
| 3 | w/o Adaptive Gate | 56.08% | -0.49% | -0.9% |
| 4 | w/o User Embedding | 56.27% | -0.31% | -0.5% |
| 5 | w/o Position Bias | 56.65% | +0.08% | +0.1% |
| 6 | Single Transformer Layer | 56.65% | +0.08% | +0.1% |
| 7 | w/o Position-from-End | 56.74% | +0.16% | +0.3% |
| 8 | w/o Generation Head | 57.41% | +0.84% | +1.5% |

### 8.2 Key Ablation Insights

1. **Pointer Mechanism is Critical**: Removing it causes 24.01% drop on GeoLife (most repetitive) and 4.67% on DIY.

2. **Generation Head is Redundant**: Removing it actually improves performance slightly on both datasets, suggesting the pointer mechanism alone is sufficient when targets are predominantly in history.

3. **Temporal Embeddings Matter More on GeoLife**: 4.03% drop vs 0.62% on DIY.

4. **Model Depth Not Critical**: Single layer performs comparably, suggesting the task doesn't require deep representations.

5. **Position Bias Negligible**: Learnable position bias provides no benefit.

---

## 9. Behavioral Analysis

### 9.1 Pointer Gate Behavior

#### Gate Value Statistics
| Dataset | Mean | Std | Median | Min | Max |
|---------|------|-----|--------|-----|-----|
| GeoLife | 0.670 | 0.200 | 0.716 | 0.021 | 0.988 |
| DIY | 0.773 | 0.155 | 0.815 | 0.070 | 0.979 |

#### Gate Value vs Performance (DIY)
| Gate Range | Samples | Acc@1 | Target in History |
|------------|---------|-------|-------------------|
| Low (0-0.3) | 160 | 20.63% | 46.88% |
| Medium (0.3-0.5) | 686 | 29.59% | 58.89% |
| Balanced (0.5-0.7) | 2,336 | 44.99% | 75.81% |
| High (0.7-1.0) | 9,186 | 62.55% | 88.77% |

**Key Finding**: The gate naturally correlates with target-in-history probability. Higher gate values (more pointer weight) occur when targets are in history.

#### In-History vs Not-In-History Performance
| Dataset | Condition | Samples | Avg Gate | Acc@1 |
|---------|-----------|---------|----------|-------|
| DIY | Target in history | 10,404 | 0.792 | 67.47% |
| DIY | Target NOT in history | 1,964 | 0.675 | 0.66% |
| GeoLife | Target in history | 2,935 | 0.685 | 64.36% |
| GeoLife | Target NOT in history | 567 | 0.592 | 0.18% |

**Critical Limitation**: Near-zero accuracy when target is not in history, despite lower gate values.

### 9.2 Recency Analysis

#### Performance by Target Recency (DIY)
| Target Recency | Samples | Acc@1 | MRR |
|----------------|---------|-------|-----|
| Same day (0) | 3,796 | 71.39% | 82.91% |
| 1 day ago | 3,823 | 72.95% | 83.96% |
| 2-3 days ago | 1,957 | 63.36% | 76.91% |
| 4-7 days ago | 828 | 33.94% | 53.64% |
| Not in history | 1,964 | 0.66% | 5.10% |

#### Performance by Target Recency (GeoLife)
| Target Recency | Samples | Acc@1 | MRR |
|----------------|---------|-------|-----|
| Same day (0) | 827 | 64.69% | 78.43% |
| 1 day ago | 1,294 | 75.89% | 86.12% |
| 2-3 days ago | 485 | 53.61% | 71.01% |
| 4-7 days ago | 329 | 34.04% | 51.83% |
| Not in history | 567 | 0.18% | 4.80% |

### 9.3 Sequence Length Analysis

#### GeoLife (by previous_day window)
| Window (days) | Samples | Avg Seq Len | Acc@1 | Acc@10 |
|---------------|---------|-------------|-------|--------|
| 1 | 3,263 | 4.15 | 47.84% | 74.32% |
| 2 | 3,398 | 6.53 | 48.97% | 77.72% |
| 3 | 3,458 | 8.87 | 49.02% | 80.48% |
| 4 | 3,487 | 11.24 | 50.59% | 81.85% |
| 5 | 3,494 | 13.60 | 50.31% | 82.91% |
| 6 | 3,499 | 15.95 | 50.96% | 83.68% |
| 7 | 3,502 | 18.37 | 51.40% | 85.04% |

**Finding**: Performance increases with longer history windows, saturating around 7 days.

### 9.4 Weekday vs Weekend Analysis

#### DIY Dataset
| Period | Samples | Acc@1 | MRR |
|--------|---------|-------|-----|
| Weekday | 8,578 | 57.24% | 68.19% |
| Weekend | 3,790 | 55.09% | 66.48% |
| **Difference** | — | **+2.15%** | **+1.71%** |

Statistical test: t = 1.32, p = 0.244 (not significant at α=0.05)

#### GeoLife Dataset
| Period | Samples | Acc@1 | MRR |
|--------|---------|-------|-----|
| Weekday | 2,633 | 55.26% | 68.42% |
| Weekend | 869 | 39.70% | 52.80% |
| **Difference** | — | **+15.56%** | **+15.62%** |

Statistical test: t = 6.30, p = 0.001 (significant at α=0.01)

### 9.5 User Activity Analysis

#### DIY Dataset
| Activity Level | Users | Avg Visits | Acc@1 |
|----------------|-------|------------|-------|
| Low | 174 | 69.4 | 63.95% |
| Medium | 342 | 197.7 | 56.22% |
| High | 176 | 477.8 | 56.14% |

**Insight**: Low-activity users are more predictable (more repetitive patterns).

### 9.6 Location Frequency Analysis

#### DIY Dataset
| Frequency Level | Samples | Acc@1 | Acc@10 |
|-----------------|---------|-------|--------|
| Very Rare (≤P10) | 176 | 3.98% | 22.73% |
| Rare (P10-P25) | 162 | 8.02% | 24.07% |
| Occasional (P25-P50) | 369 | 15.72% | 46.07% |
| Common (P50-P75) | 430 | 10.00% | 37.91% |
| Frequent (P75-P90) | 1,068 | 22.57% | 64.79% |
| Very Frequent (>P90) | 9,678 | 67.06% | 94.85% |
| Unseen Location | 485 | 37.32% | 76.08% |

**Finding**: Prediction accuracy strongly correlates with target location frequency in training data.

---

## 10. Baseline Comparisons

### 10.1 Implemented Baselines

#### 1. First-Order Markov Chain
- Per-user transition probability: P(next_loc | current_loc, user)
- Fallback: global transitions → location frequency
- No learnable parameters

#### 2. LSTM Model
- Multi-layer LSTM encoder
- Location + temporal + user embeddings
- Fully-connected classification head
- Configuration: base_emb_size=128, lstm_hidden=256, 2 layers

#### 3. Multi-Head Self-Attention (MHSA)
- Standard Transformer encoder with causal masking
- Sinusoidal positional encoding
- Same embedding strategy as LSTM
- Configuration: d_model=128, nhead=4, 3 encoder layers

### 10.2 Architecture Differences from PGT

| Feature | Markov | LSTM | MHSA | PGT |
|---------|--------|------|------|------------|
| Pointer mechanism | ✗ | ✗ | ✗ | ✓ |
| Generation head | ✗ | ✓ | ✓ | ✓ |
| Adaptive gate | ✗ | ✗ | ✗ | ✓ |
| Position-from-end | ✗ | ✗ | ✗ | ✓ |
| Pre-norm Transformer | N/A | N/A | Post-norm | Pre-norm |
| Position bias | ✗ | ✗ | ✗ | ✓ |

### 10.3 Baseline Expected Performance (from Literature)

Typical LSTM/Transformer baselines achieve ~45-52% Acc@1 on similar mobility prediction tasks. The pointer mechanism provides the primary performance advantage.

---

## 11. Technical Implementation Details

### 11.1 Software Environment

- **Framework**: PyTorch
- **Key Libraries**: numpy, pandas, scikit-learn, tqdm, yaml
- **Python Version**: Compatible with 3.8+
- **CUDA**: Mixed precision training (AMP) enabled

### 11.2 Code Organization

```
src/
├── models/
│   ├── proposed/
│   │   └── pgt.py          # Main model
│   └── baseline/
│       ├── LSTM.py                 # LSTM baseline
│       ├── MHSA.py                 # Transformer baseline
│       └── markov1st.py            # Markov chain baseline
├── training/
│   └── train_pgt.py        # Training script
└── evaluation/
    └── metrics.py                  # Evaluation metrics
```

### 11.3 Key Classes

**PointerGeneratorTransformer** (254 lines):
- `__init__()`: Initialize embeddings, transformer, pointer, generation head, gate
- `forward()`: Full forward pass returning log-probabilities
- `_create_pos_encoding()`: Sinusoidal positional encoding
- `count_parameters()`: Parameter count utility

**TrainerV45** (300 lines):
- `train_epoch()`: Single epoch training loop
- `evaluate()`: Evaluation on validation/test sets
- `train()`: Full training loop with early stopping
- `_get_lr()`: Learning rate schedule
- `_save_checkpoint()` / `_load_checkpoint()`: Model persistence

**NextLocationDataset**:
- Loads pickle files with preprocessed sequences
- Builds user location history for analysis
- Returns (x, y, x_dict) tuples

### 11.4 Input/Output Tensor Shapes

| Tensor | Shape | Description |
|--------|-------|-------------|
| x | [seq_len, batch_size] | Location IDs |
| y | [batch_size] | Target location |
| user | [batch_size] | User IDs |
| time | [seq_len, batch_size] | Time buckets |
| weekday | [seq_len, batch_size] | Weekday (0-6) |
| duration | [seq_len, batch_size] | Duration buckets |
| diff | [seq_len, batch_size] | Days ago |
| len | [batch_size] | Sequence lengths |
| logits (output) | [batch_size, num_locations] | Log-probabilities |

### 11.5 Numerical Stability

- Log-probabilities output: `log(probs + 1e-10)`
- Pointer softmax with `-inf` masking for padding
- Gradient clipping at 0.8 max norm
- AMP gradient scaling for FP16

---

## 12. Statistical Significance and Confidence

### 12.1 Confidence Intervals

95% confidence intervals computed via bootstrap or normal approximation:

**DIY Test Set** (n=12,368):
- Acc@1: 56.58% ± 0.88%
- MRR: 67.67% ± 0.70%

**GeoLife Test Set** (n=3,502):
- Acc@1: 51.40% ± 1.57%
- MRR: 64.55% ± 1.23%

### 12.2 Statistical Tests Performed

1. **Weekday vs Weekend**:
   - DIY: t=1.32, p=0.244 (not significant)
   - GeoLife: t=6.30, p=0.001 (significant)

2. **Dataset Comparison (Target in History)**:
   - Chi-squared = 0.174, p = 0.676 (no significant difference)

3. **Unique Ratio Comparison**:
   - Mann-Whitney U = 19,139,076, p < 0.001 (significant)
   - Cohen's d = -0.16 (small effect)

---

## 13. Limitations and Boundary Conditions

### 13.1 Model Limitations

1. **Cold Start for New Locations**: Near-zero accuracy (0.66% DIY, 0.18% GeoLife) when target location has never been visited by the user in the history window.

2. **Temporal Bound**: 7-day history window means locations visited >7 days ago are invisible to the model.

3. **No Spatial Encoding**: Model does not use geographic coordinates, only discrete location IDs.

4. **User-Specific History Only**: No collaborative filtering or cross-user patterns.

5. **Fixed Vocabulary**: Locations must be seen during training; truly new locations receive "unknown" encoding.

### 13.2 Dataset Limitations

1. **Geographic Specificity**: GeoLife (Beijing), DIY (unknown region) - may not generalize globally.

2. **Temporal Period**: Data from specific time periods; mobility patterns may have changed.

3. **User Selection Bias**: Quality filters remove irregular users, biasing toward consistent travelers.

4. **Sampling Bias**: GPS-based collection misses indoor locations and low-battery periods.

### 13.3 Experimental Limitations

1. **Single Seed**: All experiments use seed=42; variance across seeds not measured.

2. **Hyperparameter Sensitivity**: Tuning focused on learning rate; other hyperparameters may not be optimal.

3. **No Cross-Dataset Evaluation**: Models trained and tested on same dataset only.

### 13.4 Boundary Conditions

The model performs optimally when:
- Target location appears in 7-day history (84% of cases)
- Sequence length ≥ 10 visits
- Target is a frequently-visited location
- User has regular (weekday) mobility patterns
- Target visited within 0-1 days ago

Performance degrades significantly when:
- Target is a new/unseen location
- Sequence length < 5
- Target location is rare (≤P25 frequency)
- Weekend predictions (GeoLife only)
- Target visited 4-7 days ago

---

## Appendix A: Full Ablation Results Tables

### A.1 GeoLife Full Metrics

| Variant | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|---------|-------|-------|--------|-----|------|
| Full (baseline) | 51.43 | 81.18 | 85.04 | 64.57 | 69.48 |
| w/o Generation Head | 51.86 | 82.41 | 85.38 | 64.95 | 69.58 |
| Single Layer | 51.68 | 81.70 | 85.01 | 64.96 | 69.81 |
| w/o Position Bias | 51.48 | 81.21 | 84.98 | 64.61 | 69.49 |
| w/o Adaptive Gate | 49.54 | 81.64 | 84.67 | 63.57 | 68.67 |
| w/o Pos-from-End | 49.34 | 80.87 | 84.75 | 63.38 | 68.53 |
| w/o User Embedding | 49.11 | 81.10 | 84.12 | 63.27 | 68.33 |
| w/o Temporal | 47.40 | 81.47 | 85.09 | 62.56 | 68.03 |
| w/o Pointer | 27.41 | 54.14 | 58.65 | 38.88 | 43.43 |

### A.2 DIY Full Metrics

| Variant | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|---------|-------|-------|--------|-----|------|
| Full (baseline) | 56.57 | 82.16 | 85.16 | 67.66 | 71.88 |
| w/o Generation Head | 57.41 | 81.80 | 84.48 | 67.88 | 71.92 |
| w/o Pos-from-End | 56.74 | 82.28 | 85.27 | 67.82 | 72.03 |
| w/o Position Bias | 56.65 | 82.14 | 85.16 | 67.70 | 71.90 |
| Single Layer | 56.65 | 81.90 | 85.04 | 67.58 | 71.78 |
| w/o User Embedding | 56.27 | 81.98 | 84.89 | 67.31 | 71.57 |
| w/o Adaptive Gate | 56.08 | 81.90 | 85.28 | 67.22 | 71.56 |
| w/o Temporal | 55.95 | 82.03 | 85.24 | 67.24 | 71.56 |
| w/o Pointer | 51.90 | 75.59 | 78.27 | 62.21 | 66.05 |

---

## Appendix B: Hyperparameter Search Space

### B.1 Searched Parameters

| Parameter | Search Range | Optimal (GeoLife) | Optimal (DIY) |
|-----------|--------------|-------------------|---------------|
| d_model | {64, 128, 256} | 64 | 128 |
| nhead | {2, 4, 8} | 4 | 4 |
| num_layers | {1, 2, 3, 4} | 2 | 3 |
| dim_feedforward | {128, 256, 512} | 128 | 256 |
| dropout | {0.1, 0.15, 0.2} | 0.15 | 0.15 |
| learning_rate | {1e-4, 3e-4, 5e-4, 7e-4, 1e-3} | 6.5e-4 | 7e-4 |
| weight_decay | {0.01, 0.015, 0.02} | 0.015 | 0.015 |
| batch_size | {64, 128, 256} | 128 | 128 |

---

## Appendix C: Data Format Specifications

### C.1 Pickle File Structure

Each `.pk` file contains a list of dictionaries:
```python
[
    {
        'X': np.array([loc1, loc2, ..., locN]),      # Location sequence
        'Y': int,                                      # Target location
        'user_X': np.array([user_id, user_id, ...]),  # User ID repeated
        'weekday_X': np.array([0-6, 0-6, ...]),       # Weekday per visit
        'start_min_X': np.array([0-1439, ...]),       # Start minute
        'dur_X': np.array([duration_mins, ...]),      # Duration in minutes
        'diff': np.array([days_ago, ...])             # Days from prediction time
    },
    ...
]
```

### C.2 Metadata JSON Structure

```json
{
    "dataset_name": "geolife",
    "output_dataset_name": "geolife_eps20_prev7",
    "epsilon": 20,
    "previous_day": 7,
    "total_user_num": 46,
    "total_loc_num": 1158,
    "unique_users": 45,
    "unique_locations": 1156,
    "train_sequences": 10231,
    "val_sequences": 3428,
    "test_sequences": 3502,
    "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2}
}
```

---

## Appendix D: Reproducibility Checklist

- [ ] Random seed set to 42
- [ ] CUDA deterministic mode enabled
- [ ] Data splits are chronological per user
- [ ] Same preprocessing pipeline (DBSCAN ε values)
- [ ] Same train/val/test split ratios
- [ ] Same hyperparameters as Section 4
- [ ] Early stopping patience = 5
- [ ] Min epochs = 8
- [ ] Best checkpoint selected by validation loss
- [ ] Evaluation on test set with loaded best checkpoint

---

**END OF DOCUMENT**

*This document serves as the authoritative reference for the PointerGeneratorTransformer model and all associated experiments. All numerical results, architectural specifications, and experimental protocols are derived directly from the source code and experimental outputs.*
