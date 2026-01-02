# Multi-Head Self-Attention (MHSA) Model for Next Location Prediction

## Complete Technical Documentation

**Version:** 1.0  
**Last Updated:** January 2026  
**Repository:** `next_loc_clean_v2`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [Theoretical Background](#3-theoretical-background)
4. [Model Architecture](#4-model-architecture)
5. [Component Deep Dive](#5-component-deep-dive)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Process](#7-training-process)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Experimental Results](#9-experimental-results)
10. [Configuration Reference](#10-configuration-reference)
11. [Usage Guide](#11-usage-guide)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

---

## 1. Executive Summary

### 1.1 What is the MHSA Model?

The **Multi-Head Self-Attention (MHSA) Model** is a Transformer Encoder-based neural network designed for **next location prediction**. Given a user's historical sequence of visited locations along with temporal context (time of day, weekday, duration), the model predicts the most likely next location the user will visit.

### 1.2 Key Features

- **Transformer Architecture**: Leverages self-attention mechanisms to capture complex temporal dependencies in location sequences
- **Multi-Modal Input**: Combines location embeddings with temporal features (time, weekday, duration)
- **User Personalization**: Optional user embeddings for personalized predictions
- **Scalable Design**: Configurable depth (encoder layers) and width (embedding dimensions)
- **State-of-the-Art Performance**: Achieves competitive accuracy on benchmark datasets

### 1.3 Performance Summary

| Dataset | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG@10 |
|---------|-------|-------|--------|-----|---------|
| GeoLife | ~30% | ~51% | ~58% | ~40% | ~44% |
| DIY | ~53% | ~77% | ~81% | ~63% | ~68% |

---

## 2. Introduction and Problem Statement

### 2.1 The Next Location Prediction Problem

**Next location prediction** is a fundamental task in human mobility modeling. Given a sequence of locations a user has visited, the goal is to predict where they will go next.

**Formal Definition:**
```
Given: L = [l₁, l₂, ..., lₜ] - sequence of t historical locations
       C = [c₁, c₂, ..., cₜ] - associated context (time, weekday, duration)
       u - user identifier
Predict: l_{t+1} - the next location
```

### 2.2 Why is This Problem Important?

1. **Location-Based Services**: Personalized recommendations, targeted advertising
2. **Urban Planning**: Understanding movement patterns for infrastructure design
3. **Transportation**: Predicting traffic flow and transit demand
4. **Public Health**: Contact tracing, epidemic modeling
5. **Security**: Anomaly detection in movement patterns

### 2.3 Challenges

1. **Spatial Sparsity**: Large number of possible locations, each user visits only a small subset
2. **Temporal Dynamics**: Movement patterns change by time of day, day of week, season
3. **Individual Variability**: Different users have different mobility patterns
4. **Sequence Dependencies**: Current location depends on past trajectory, not just previous location
5. **Cold Start**: Predicting for users/locations with limited history

### 2.4 Why Transformer/MHSA Architecture?

Traditional approaches (Markov chains, RNNs/LSTMs) have limitations:

| Approach | Limitation |
|----------|------------|
| Markov Models | Only capture limited-order dependencies |
| RNNs | Sequential processing limits parallelization |
| LSTMs | Gradient issues with very long sequences |
| CNNs | Fixed receptive field size |

**Transformers address these limitations:**
- Self-attention allows direct modeling of dependencies at any distance
- Parallel processing enables faster training
- Multi-head attention captures different types of relationships
- Proven success in sequence modeling tasks (NLP, time series)

---

## 3. Theoretical Background

### 3.1 Self-Attention Mechanism

Self-attention computes a weighted sum of values, where weights are determined by the similarity between queries and keys.

**Mathematical Formulation:**

Given an input sequence X ∈ ℝ^(n×d) where n is sequence length and d is embedding dimension:

1. **Compute Q, K, V matrices:**
```
Q = XW_Q    (Query)
K = XW_K    (Key)  
V = XW_V    (Value)
```
Where W_Q, W_K, W_V ∈ ℝ^(d×d_k) are learnable projection matrices.

2. **Scaled Dot-Product Attention:**
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

The scaling factor √d_k prevents the dot products from becoming too large (which would push softmax into regions with extremely small gradients).

**Intuition:**
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What do I provide?"

For each position, attention scores indicate how much to "attend" to each other position.

### 3.2 Multi-Head Attention

Instead of performing a single attention function, multi-head attention runs h parallel attention heads:

```
MultiHead(Q, K, V) = Concat(head₁, ..., head_h)W_O

where head_i = Attention(QW_Q^i, KW_K^i, VW_V^i)
```

**Why Multiple Heads?**
- Different heads can learn different types of relationships
- Some heads might focus on recent locations, others on periodic patterns
- Provides model with more expressive power

### 3.3 Positional Encoding

Transformers have no inherent notion of sequence order. Positional encoding adds position information:

**Sinusoidal Positional Encoding:**
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
```

Where:
- `pos` is the position in sequence
- `i` is the dimension index
- `d` is the embedding dimension

**Properties:**
- Unique encoding for each position
- Can generalize to unseen sequence lengths
- Relative positions can be computed from absolute encodings

### 3.4 Transformer Encoder Layer

Each encoder layer consists of:

```
┌─────────────────────────────────────┐
│           Input Embeddings          │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│     Multi-Head Self-Attention       │
│         + Residual Connection       │
│         + Layer Normalization       │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│     Feed-Forward Network (FFN)      │
│         + Residual Connection       │
│         + Layer Normalization       │
└────────────────┬────────────────────┘
                 ▼
┌─────────────────────────────────────┐
│            Output                   │
└─────────────────────────────────────┘
```

**Feed-Forward Network:**
```
FFN(x) = GELU(xW₁ + b₁)W₂ + b₂
```

### 3.5 Causal Masking

For autoregressive prediction, we use a causal mask to prevent attending to future positions:

```
Mask = upper_triangular(−∞)

      pos₁  pos₂  pos₃  pos₄
pos₁ [  0    -∞    -∞    -∞  ]
pos₂ [  0     0    -∞    -∞  ]
pos₃ [  0     0     0    -∞  ]
pos₄ [  0     0     0     0  ]
```

This ensures that prediction at position t only uses information from positions 1 to t.

---

## 4. Model Architecture

### 4.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MHSA MODEL                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌─────────────┐    ┌──────────────────┐    ┌──────────────────┐   │
│  │   Input     │    │   AllEmbedding   │    │ TransformerEncoder│   │
│  │  Sequence   │───▶│     Layer        │───▶│     Layers       │   │
│  │  + Context  │    │                  │    │                  │   │
│  └─────────────┘    └──────────────────┘    └────────┬─────────┘   │
│                                                       │             │
│                                                       ▼             │
│                     ┌──────────────────┐    ┌──────────────────┐   │
│                     │   Location       │◀───│  FullyConnected  │   │
│                     │   Probabilities  │    │     Layer        │   │
│                     └──────────────────┘    └──────────────────┘   │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Simplified Architecture Diagram

```
INPUT                    EMBEDDING                 ENCODER              OUTPUT
─────                    ─────────                 ───────              ──────

Location IDs ───┐
                │
Time ──────────┐│        ┌─────────────┐
               ││   ┌───▶│  Location   │
Weekday ──────┐││   │    │  Embedding  │
              │││   │    └──────┬──────┘
Duration ────┐│││   │           │         ┌─────────────────┐
             ││││   │           ▼         │                 │
             ▼▼▼▼   │    ┌─────────────┐  │  Transformer    │    ┌────────────┐
        ┌─────────┐ │    │   Combined  │  │  Encoder        │    │  Logits    │
        │  Input  │─┘    │   Embedding │──▶│  (N layers)    │───▶│  [B, L]    │
        │  Dict   │      │ + Position  │  │                 │    │            │
        └─────────┘      └─────────────┘  └─────────────────┘    └────────────┘
```

### 4.3 Detailed Architecture Diagram

```
╔══════════════════════════════════════════════════════════════════════════════════╗
║                           MHSA MODEL - DETAILED VIEW                              ║
╠══════════════════════════════════════════════════════════════════════════════════╣
║                                                                                  ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │                           INPUT LAYER                                       │ ║
║  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │ ║
║  │  │ Location │  │   Time   │  │ Weekday  │  │ Duration │  │  Length  │       │ ║
║  │  │ [S, B]   │  │ [S, B]   │  │ [S, B]   │  │ [S, B]   │  │   [B]    │       │ ║
║  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │ ║
║  └───────┼─────────────┼─────────────┼─────────────┼─────────────┼─────────────┘ ║
║          │             │             │             │             │               ║
║          ▼             ▼             ▼             ▼             │               ║
║  ┌───────────────────────────────────────────────────────────────┼─────────────┐ ║
║  │                      AllEmbedding LAYER                       │             │ ║
║  │  ┌──────────────┐  ┌──────────────────────┐  ┌──────────────┐ │             │ ║
║  │  │   Location   │  │  TemporalEmbedding   │  │   Duration   │ │             │ ║
║  │  │  Embedding   │  │ ┌─────┐ ┌─────┐ ┌───┐│  │  Embedding   │ │             │ ║
║  │  │  [V, D]      │  │ │Hour │+│Min  │+│Day││  │  [96, D]     │ │             │ ║
║  │  │              │  │ │[24,D]│ │[4,D]│ │[7,D│  │              │ │             │ ║
║  │  └──────┬───────┘  └─┴─────┴─┴─────┴─┴───┴┘  └──────┬───────┘ │             │ ║
║  │         │                    │                       │         │             │ ║
║  │         └────────────────────┼───────────────────────┘         │             │ ║
║  │                              ▼                                 │             │ ║
║  │                    ┌──────────────────┐                        │             │ ║
║  │                    │  Element-wise    │                        │             │ ║
║  │                    │     Addition     │                        │             │ ║
║  │                    └────────┬─────────┘                        │             │ ║
║  │                             ▼                                  │             │ ║
║  │                    ┌──────────────────┐                        │             │ ║
║  │                    │   Positional     │                        │             │ ║
║  │                    │   Encoding       │                        │             │ ║
║  │                    │   + Dropout      │                        │             │ ║
║  │                    └────────┬─────────┘                        │             │ ║
║  └─────────────────────────────┼──────────────────────────────────┼─────────────┘ ║
║                                │                                  │               ║
║                                ▼                                  │               ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │                    TRANSFORMER ENCODER (N layers)                           │ ║
║  │  ┌─────────────────────────────────────────────────────────────────────┐    │ ║
║  │  │                    Encoder Layer 1                                  │    │ ║
║  │  │  ┌─────────────────────────────────────────────────────────────┐    │    │ ║
║  │  │  │              Multi-Head Self-Attention                      │    │    │ ║
║  │  │  │   ┌─────────────────────────────────────────────────────┐   │    │    │ ║
║  │  │  │   │  Causal Mask        Padding Mask                    │   │    │    │ ║
║  │  │  │   │  ┌───────────┐     ┌───────────┐                    │   │    │    │ ║
║  │  │  │   │  │ 0  -∞ -∞  │     │ 0  0  1   │                    │   │    │    │ ║
║  │  │  │   │  │ 0   0 -∞  │  +  │ 0  0  1   │                    │   │    │    │ ║
║  │  │  │   │  │ 0   0  0  │     │ 0  0  1   │                    │   │    │    │ ║
║  │  │  │   │  └───────────┘     └───────────┘                    │   │    │    │ ║
║  │  │  │   │                                                      │   │    │    │ ║
║  │  │  │   │  Q, K, V Projections → Scaled Dot-Product Attention  │   │    │    │ ║
║  │  │  │   │  8 Heads × d_k dimensions                            │   │    │    │ ║
║  │  │  │   └─────────────────────────────────────────────────────┘   │    │    │ ║
║  │  │  │                         │                                    │    │    │ ║
║  │  │  │              Residual + LayerNorm                           │    │    │ ║
║  │  │  └─────────────────────────┬───────────────────────────────────┘    │    │ ║
║  │  │                            │                                        │    │ ║
║  │  │  ┌─────────────────────────▼───────────────────────────────────┐    │    │ ║
║  │  │  │              Feed-Forward Network                           │    │    │ ║
║  │  │  │   Linear(D, 4D) → GELU → Linear(4D, D)                     │    │    │ ║
║  │  │  │              Residual + LayerNorm                           │    │    │ ║
║  │  │  └─────────────────────────┬───────────────────────────────────┘    │    │ ║
║  │  └────────────────────────────┼────────────────────────────────────────┘    │ ║
║  │                               │                                             │ ║
║  │                               ▼                                             │ ║
║  │  ┌────────────────────────────────────────────────────────────────────┐     │ ║
║  │  │                    Encoder Layer 2...N                             │     │ ║
║  │  │                         (same structure)                           │     │ ║
║  │  └────────────────────────────┬───────────────────────────────────────┘     │ ║
║  │                               │                                             │ ║
║  │                    Final LayerNorm                                          │ ║
║  └───────────────────────────────┼─────────────────────────────────────────────┘ ║
║                                  │                                               ║
║                                  ▼                                  Length Info  ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │                         SEQUENCE SELECTION                                  │ ║
║  │                                                                             │ ║
║  │   Select last valid timestep for each sequence using length info            │ ║
║  │   out[i] = encoder_output[length[i]-1, i, :]                                │ ║
║  │                                                                             │ ║
║  └─────────────────────────────────┬───────────────────────────────────────────┘ ║
║                                    │                                             ║
║                                    ▼                                             ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │                      FullyConnected LAYER                                   │ ║
║  │  ┌──────────────┐  ┌────────────────────────────────────────┐               │ ║
║  │  │    User      │  │           Residual Block               │               │ ║
║  │  │  Embedding   │  │  Linear(D, 2D) → ReLU → Linear(2D, D)  │               │ ║
║  │  │  [U, D]      │  │       + BatchNorm + Dropout            │               │ ║
║  │  └──────┬───────┘  └────────────────────┬───────────────────┘               │ ║
║  │         │                               │                                   │ ║
║  │         └───────────────────────────────┘                                   │ ║
║  │                          │                                                  │ ║
║  │                          ▼                                                  │ ║
║  │                  ┌───────────────┐                                          │ ║
║  │                  │  Final Linear │                                          │ ║
║  │                  │  [D, V]       │                                          │ ║
║  │                  └───────┬───────┘                                          │ ║
║  └──────────────────────────┼──────────────────────────────────────────────────┘ ║
║                             │                                                    ║
║                             ▼                                                    ║
║  ┌─────────────────────────────────────────────────────────────────────────────┐ ║
║  │                            OUTPUT                                           │ ║
║  │                                                                             │ ║
║  │                    Logits: [Batch, num_locations]                           │ ║
║  │                    → CrossEntropyLoss with ground truth                     │ ║
║  │                    → Top-k predictions for evaluation                       │ ║
║  │                                                                             │ ║
║  └─────────────────────────────────────────────────────────────────────────────┘ ║
║                                                                                  ║
║  LEGEND:                                                                         ║
║  S = Sequence Length, B = Batch Size, D = Embedding Dimension                   ║
║  V = Vocabulary Size (num locations), U = Number of Users                       ║
║                                                                                  ║
╚══════════════════════════════════════════════════════════════════════════════════╝
```

### 4.4 Data Flow Summary

| Stage | Input Shape | Output Shape | Description |
|-------|-------------|--------------|-------------|
| Input | [S, B] | - | Location IDs |
| Location Embedding | [S, B] | [S, B, D] | Lookup location vectors |
| Temporal Embedding | [S, B] | [S, B, D] | Combine hour, minute, weekday |
| Duration Embedding | [S, B] | [S, B, D] | Duration lookup |
| Combined Embedding | [S, B, D] × 3 | [S, B, D] | Element-wise addition |
| Positional Encoding | [S, B, D] | [S, B, D] | Add position info |
| Encoder | [S, B, D] | [S, B, D] | N transformer layers |
| Sequence Selection | [S, B, D] + [B] | [B, D] | Extract last valid |
| User Embedding | [B] | [B, D] | User vectors |
| Residual FC | [B, D] | [B, D] | Non-linear transform |
| Output Linear | [B, D] | [B, V] | Classification logits |

---

## 5. Component Deep Dive

### 5.1 PositionalEncoding

**Purpose:** Add position information to embeddings since Transformers have no inherent notion of sequence order.

**Location:** `src/models/baseline/MHSA.py`, lines 24-46

```python
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
```

**Mathematical Details:**

For position `pos` and dimension `i`:
- Even dimensions: `PE(pos, 2i) = sin(pos / 10000^(2i/d))`
- Odd dimensions: `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`

**Why Sinusoidal?**
1. **Bounded values**: Output always between -1 and 1
2. **Relative positions**: `PE(pos+k)` can be represented as a linear function of `PE(pos)`
3. **Unique encodings**: Each position has a distinct pattern
4. **Extrapolation**: Can handle sequences longer than training data

**Input/Output:**
- Input: `[seq_len, batch_size, emb_dim]` - token embeddings
- Output: `[seq_len, batch_size, emb_dim]` - embeddings with position info

### 5.2 TemporalEmbedding

**Purpose:** Encode time-related features (hour, minute, weekday) into dense vectors.

**Location:** `src/models/baseline/MHSA.py`, lines 49-93

```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()
        self.emb_info = emb_info
        self.minute_size = 4  # quarter-hour
        hour_size = 24
        weekday = 7

        if self.emb_info == "all":
            self.minute_embed = nn.Embedding(self.minute_size, d_input)  # 4 quarter-hours
            self.hour_embed = nn.Embedding(hour_size, d_input)            # 24 hours
            self.weekday_embed = nn.Embedding(weekday, d_input)           # 7 days
```

**Time Representation:**
- Time is encoded as a single integer: `time_slot = hour * 4 + quarter`
- Quarter-hour: 0-3 (0-14 min, 15-29 min, 30-44 min, 45-59 min)
- This gives 96 possible time slots per day (24 hours × 4 quarters)

**Embedding Modes:**
| Mode | Embeddings Used | Formula |
|------|-----------------|---------|
| `"all"` | hour + minute + weekday | Three separate embeddings summed |
| `"time"` | single time slot | 96-slot embedding (0-95) |
| `"weekday"` | weekday only | 7-slot embedding (0-6) |

**Justification for "all" mode:**
- Separating hour and minute allows learning different patterns
- Morning rush (8-9 AM) shares features across different minutes
- Weekday patterns (Mon-Fri vs Sat-Sun) are distinct
- Additive combination allows flexible representation

**Input/Output:**
- Input: `time [S, B]` (0-95), `weekday [S, B]` (0-6)
- Output: `[S, B, D]` - temporal embedding vector

### 5.3 POINet (Optional)

**Purpose:** Process Point of Interest (POI) feature vectors when available.

**Location:** `src/models/baseline/MHSA.py`, lines 96-151

**Architecture:**
```
POI Features [S, B, 16, 11]
        │
        ▼
  Linear(11 → 32) → ReLU
  Linear(32 → 11)
  + Residual + LayerNorm
        │
        ▼
  Flatten to [S, B, 176]
  Dense(176 → 16) → ReLU
        │
        ▼
  Linear(16 → 64) → ReLU
  Linear(64 → 16)
  + Residual + LayerNorm
        │
        ▼
  Linear(16 → D)
```

**Why This Structure?**
- First block: Process relationships between POI categories
- Flatten + Dense: Combine all POI information
- Second block: Non-linear transformation with residual
- Final linear: Project to embedding dimension

**Note:** POI features are not used in the default GeoLife and DIY configurations (`if_embed_poi: false`).

### 5.4 AllEmbedding

**Purpose:** Combine all input embeddings into a single representation.

**Location:** `src/models/baseline/MHSA.py`, lines 154-226

**Components:**
1. **Location Embedding**: `nn.Embedding(total_loc_num, d_input)`
2. **Temporal Embedding**: Hour + Minute + Weekday
3. **Duration Embedding**: `nn.Embedding(96, d_input)` (max 48 hours in 30-min bins)
4. **POI Embedding**: Optional POINet
5. **Positional Encoding**: Sinusoidal

**Combination Strategy:**
```python
def forward(self, src, context_dict):
    emb = self.emb_loc(src)                    # Location embedding
    
    if self.if_include_time:
        emb = emb + self.temporal_embedding(   # Add temporal
            context_dict["time"], 
            context_dict["weekday"]
        )
    
    if self.if_include_duration:
        emb = emb + self.emb_duration(         # Add duration
            context_dict["duration"]
        )
    
    if self.if_include_poi:
        emb = emb + self.poi_net(              # Add POI
            context_dict["poi"]
        )
    
    return self.pos_encoder(emb * math.sqrt(self.d_input))  # Scale + position
```

**Why Additive Combination?**
- Each modality contributes independently
- Scaling by √d prevents embeddings from becoming too large
- Maintains dimensionality (no increase like concatenation)
- Allows gradient flow to all embedding types

### 5.5 TransformerEncoder (Core MHSA)

**Purpose:** The main attention-based sequence encoder.

**Location:** PyTorch's `nn.TransformerEncoder` + `nn.TransformerEncoderLayer`

**Configuration:**
```python
encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=self.d_input,           # Embedding dimension
    nhead=config.nhead,              # Number of attention heads (8)
    activation="gelu",               # Activation function
    dim_feedforward=config.dim_feedforward  # FFN hidden dim (128-256)
)
encoder_norm = torch.nn.LayerNorm(self.d_input)
self.encoder = torch.nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=config.num_encoder_layers,  # Number of layers (2-4)
    norm=encoder_norm
)
```

**GELU Activation:**
```
GELU(x) = x * Φ(x)
```
Where Φ is the cumulative distribution function of the standard normal.

**Why GELU over ReLU?**
- Smoother gradient flow
- Better performance in Transformer architectures
- Non-zero gradient for negative inputs (unlike ReLU)

**Masking:**
```python
# Causal mask - prevent attending to future positions
src_mask = self._generate_square_subsequent_mask(src.shape[0])

# Padding mask - ignore padded positions
src_padding_mask = (src == 0).transpose(0, 1)
```

### 5.6 FullyConnected (Output Layer)

**Purpose:** Transform encoder output to location predictions.

**Location:** `src/models/baseline/MHSA.py`, lines 229-272

**Architecture:**
```
Encoder Output [B, D]
        │
        ▼ (if if_embed_user)
  + User Embedding [U, D]
        │
        ▼
    Dropout(0.1)
        │
        ▼ (if if_residual_layer)
  ┌─────────────────────┐
  │   Residual Block    │
  │ Linear(D → 2D)      │
  │ ReLU                │
  │ Dropout             │
  │ Linear(2D → D)      │
  │ Dropout             │
  │ + Residual          │
  │ BatchNorm           │
  └──────────┬──────────┘
             ▼
      Linear(D → V)
             │
             ▼
    Logits [B, V]
```

**User Embedding Justification:**
- Different users have different mobility patterns
- User embedding captures individual preferences
- Added to encoder output before final prediction

**Residual Block Justification:**
- Additional non-linear transformation capacity
- Residual connection prevents gradient degradation
- BatchNorm stabilizes training

---

## 6. Data Pipeline

### 6.1 Preprocessing Overview

The data pipeline converts raw GPS trajectories into training samples:

```
Raw GPS Data → Staypoints → Locations (DBSCAN) → Sequences → Training Samples
```

**Key Parameters:**
| Parameter | GeoLife | DIY | Description |
|-----------|---------|-----|-------------|
| epsilon | 20m | 50m | DBSCAN clustering radius |
| previous_day | 7 | 7 | Days of history to include |
| min_sequence | 3 | 3 | Minimum locations in sequence |
| split | 60/20/20 | 80/10/10 | Train/Val/Test ratios |

### 6.2 Data Format

**Training Sample (from `.pk` file):**
```python
{
    'X': np.array([loc_1, loc_2, ..., loc_t]),      # History locations
    'Y': int,                                        # Target location
    'user_X': np.array([user, user, ..., user]),    # User ID (repeated)
    'weekday_X': np.array([day_1, day_2, ..., day_t]),  # Weekdays (0-6)
    'start_min_X': np.array([min_1, min_2, ..., min_t]),  # Start minute (0-1439)
    'dur_X': np.array([dur_1, dur_2, ..., dur_t]),  # Duration (minutes)
    'diff': np.array([diff_1, diff_2, ..., diff_t])  # Days from target
}
```

**Location ID Encoding:**
- `0`: Padding token
- `1`: Unknown location (not in training vocabulary)
- `2+`: Known locations

### 6.3 LocationDataset Class

**Location:** `src/training/train_MHSA.py`, lines 141-177

```python
class LocationDataset(Dataset):
    def __init__(self, data_path, dataset_name="geolife"):
        self.data = pickle.load(open(data_path, "rb"))
        
    def __getitem__(self, idx):
        selected = self.data[idx]
        
        return_dict = {}
        x = torch.tensor(selected["X"])
        y = torch.tensor(selected["Y"])
        
        return_dict["user"] = torch.tensor(selected["user_X"][0])
        return_dict["time"] = torch.tensor(selected["start_min_X"] // 15)  # Convert to 15-min slots
        return_dict["diff"] = torch.tensor(selected["diff"])
        return_dict["duration"] = torch.tensor(selected["dur_X"] // 30)    # Convert to 30-min bins
        return_dict["weekday"] = torch.tensor(selected["weekday_X"])
        
        return x, y, return_dict
```

**Time Conversion:**
- Raw: minutes from midnight (0-1439)
- Converted: 15-minute slots (0-95) = `minutes // 15`

**Duration Conversion:**
- Raw: duration in minutes
- Converted: 30-minute bins = `duration // 30`
- Max: 96 bins (48 hours)

### 6.4 Collate Function

**Purpose:** Handle variable-length sequences in batches.

**Location:** `src/training/train_MHSA.py`, lines 180-206

```python
def collate_fn(batch):
    x_batch, y_batch = [], []
    x_dict_batch = {"len": []}
    
    for src_sample, tgt_sample, return_dict in batch:
        x_batch.append(src_sample)
        y_batch.append(tgt_sample)
        x_dict_batch["len"].append(len(src_sample))
        
    # Pad sequences to same length
    x_batch = pad_sequence(x_batch)  # [max_len, batch_size]
    y_batch = torch.tensor(y_batch, dtype=torch.int64)
    
    # Pad context features similarly
    for key in x_dict_batch:
        if key not in ["user", "len"]:
            x_dict_batch[key] = pad_sequence(x_dict_batch[key])
    
    return x_batch, y_batch, x_dict_batch
```

**Padding Strategy:**
- PyTorch's `pad_sequence` pads with 0
- Location 0 is the padding token
- Padding mask is created from `src == 0`

---

## 7. Training Process

### 7.1 Training Loop Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TRAINING LOOP                                │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  for epoch in range(max_epoch):                                │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   TRAIN EPOCH                             │  │
│  │  for batch in train_loader:                              │  │
│  │      logits = model(x, context_dict)                     │  │
│  │      loss = CrossEntropyLoss(logits, y)                  │  │
│  │      loss.backward()                                      │  │
│  │      clip_grad_norm_(model.parameters(), 1.0)            │  │
│  │      optimizer.step()                                     │  │
│  │      scheduler.step()  # warmup scheduler                │  │
│  └──────────────────────────────────────────────────────────┘  │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   VALIDATE                                │  │
│  │  val_loss, val_metrics = validate(model, val_loader)     │  │
│  └──────────────────────────────────────────────────────────┘  │
│      │                                                          │
│      ▼                                                          │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                EARLY STOPPING CHECK                       │  │
│  │  if val_loss < best_loss:                                │  │
│  │      save_checkpoint(model)                              │  │
│  │      reset_counter()                                      │  │
│  │  else:                                                    │  │
│  │      counter += 1                                         │  │
│  │      if counter >= patience:                             │  │
│  │          if scheduler_count < 2:                         │  │
│  │              load_best_model()                           │  │
│  │              reduce_lr()                                  │  │
│  │          else:                                            │  │
│  │              STOP                                         │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 7.2 Loss Function

**CrossEntropyLoss:**
```python
CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
```

- `ignore_index=0`: Ignores padding tokens in loss calculation
- `reduction="mean"`: Average loss across all valid predictions

**Mathematical Form:**
```
Loss = -Σ y_true * log(softmax(logits))
```

### 7.3 Optimizer Configuration

**Adam Optimizer:**
```python
optim = torch.optim.Adam(
    model.parameters(),
    lr=0.001,              # Learning rate
    betas=(0.9, 0.999),    # Momentum parameters
    weight_decay=0.000001  # L2 regularization
)
```

**Learning Rate Schedule:**

1. **Warmup Phase** (first 2 epochs × batches):
   - Linear increase from 0 to target LR
   - Prevents large gradients at initialization

2. **Linear Decay**:
   - Gradual decrease over training epochs

3. **Step LR** (after early stopping trigger):
   - Multiply LR by 0.1
   - Allows fine-tuning at lower LR

```python
scheduler = get_linear_schedule_with_warmup(
    optim,
    num_warmup_steps=len(train_loader) * 2,     # 2 epochs warmup
    num_training_steps=len(train_loader) * 50   # 50 epochs total
)

scheduler_ES = StepLR(optim, step_size=1, gamma=0.1)  # After early stop
```

### 7.4 Early Stopping

**Algorithm:**
```
best_loss = infinity
counter = 0
patience = 5
scheduler_count = 0

for each epoch:
    val_loss = validate()
    
    if val_loss < best_loss - delta:
        best_loss = val_loss
        save_model()
        counter = 0
    else:
        counter += 1
        if counter >= patience:
            if scheduler_count < 2:
                load_best_model()
                reduce_lr()
                scheduler_count += 1
                counter = 0
            else:
                STOP TRAINING
```

**Justification:**
- Prevents overfitting by stopping when validation loss plateaus
- LR reduction gives model chance to escape local minima
- Multiple reduction attempts (2) before final stop

### 7.5 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

**Why?**
- Prevents exploding gradients
- Stabilizes training with attention mechanisms
- Max gradient norm of 1.0 is standard for Transformers

---

## 8. Evaluation Metrics

### 8.1 Accuracy@k

**Definition:** Proportion of test samples where the correct location is in the top-k predictions.

```
Acc@k = (# samples with correct in top-k) / (total samples) × 100%
```

**Interpretation:**
- Acc@1: Exact prediction accuracy
- Acc@5: Correct location in top 5 suggestions
- Acc@10: Useful for recommendation systems

### 8.2 Mean Reciprocal Rank (MRR)

**Definition:** Average of reciprocal ranks of the correct prediction.

```
MRR = (1/N) Σ (1 / rank_i)
```

Where `rank_i` is the position of the correct answer in the sorted predictions.

**Properties:**
- Ranges from 0 to 1
- Gives more weight to higher rankings
- MRR = 1.0 means all predictions are exactly correct

**Example:**
| Sample | Correct Rank | Reciprocal Rank |
|--------|--------------|-----------------|
| 1 | 1 | 1.0 |
| 2 | 3 | 0.333 |
| 3 | 2 | 0.5 |
| MRR | - | 0.611 |

### 8.3 NDCG@10 (Normalized Discounted Cumulative Gain)

**Definition:** Measures ranking quality with position-weighted scoring.

```
DCG@k = Σ (2^rel_i - 1) / log2(i + 1)
NDCG@k = DCG@k / IDCG@k
```

For binary relevance (correct/incorrect):
```
NDCG = 1 / log2(rank + 1)  if rank <= k
     = 0                    otherwise
```

**Properties:**
- Ranges from 0 to 1
- Penalizes relevant items appearing lower in ranking
- NDCG = 1.0 means correct answer is always rank 1

### 8.4 F1 Score (Weighted)

**Definition:** Harmonic mean of precision and recall, weighted by class frequency.

```
F1_weighted = Σ (support_c / total) × F1_c
```

Where `F1_c = 2 × (precision_c × recall_c) / (precision_c + recall_c)`

**Implementation:**
```python
f1 = f1_score(true_labels, top1_predictions, average="weighted")
```

### 8.5 Metric Implementation

**Location:** `src/evaluation/metrics.py`

```python
def calculate_correct_total_prediction(logits, true_y):
    result_ls = []
    
    # Top-k accuracy
    for k in [1, 3, 5, 10]:
        prediction = torch.topk(logits, k=k, dim=-1).indices
        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum()
        result_ls.append(top_k)
    
    # MRR
    result_ls.append(get_mrr(logits, true_y))
    
    # NDCG
    result_ls.append(get_ndcg(logits, true_y))
    
    # Total count
    result_ls.append(true_y.shape[0])
    
    return np.array(result_ls), true_y.cpu(), top1
```

---

## 9. Experimental Results

### 9.1 Dataset Statistics

| Metric | GeoLife | DIY |
|--------|---------|-----|
| Total Users | 45 | 692 |
| Total Locations | 1,185 | 7,036 |
| Total Staypoints | 15,978 | ~200,000 |
| Train Sequences | 7,424 | ~160,000 |
| Val Sequences | 3,334 | ~20,000 |
| Test Sequences | 3,502 | ~20,000 |
| Epsilon (DBSCAN) | 20m | 50m |
| Previous Days | 7 | 7 |

### 9.2 Model Configurations

**GeoLife (Best):**
```yaml
base_emb_size: 32
num_encoder_layers: 2
nhead: 8
dim_feedforward: 128
fc_dropout: 0.2
batch_size: 32
lr: 0.001
```
Parameters: ~112K

**DIY (Best):**
```yaml
base_emb_size: 96
num_encoder_layers: 4
nhead: 8
dim_feedforward: 256
fc_dropout: 0.1
batch_size: 256
lr: 0.001
```
Parameters: ~1.2M

### 9.3 Performance Results

**GeoLife Dataset:**
| Configuration | Params | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|--------------|--------|-------|-------|--------|-----|------|
| Baseline (emb32, L2) | 112K | 29.44% | 49.5% | 56.3% | 40.67% | 43.8% |
| emb128, L1 | 298K | **32.95%** | 51.1% | 57.6% | 42.1% | 45.2% |
| emb96, L3 | 470K | 30.81% | 50.8% | 57.2% | 40.84% | 44.1% |

**DIY Dataset:**
| Configuration | Params | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|--------------|--------|-------|-------|--------|-----|------|
| Baseline (emb64, L3) | 1.2M | **53.17%** | 77.3% | 81.4% | 63.57% | 68.2% |
| emb128, L4 | 2.8M | 52.80% | 76.9% | 81.1% | 63.37% | 67.9% |
| emb96, L4 | 2.0M | 52.76% | 76.8% | 81.0% | 63.32% | 67.8% |

### 9.4 Key Findings

1. **Model Size vs Performance:**
   - Larger models don't always perform better
   - GeoLife benefits from wider (emb128) over deeper models
   - DIY baseline is already well-optimized

2. **Architecture Insights:**
   - For GeoLife: 1-2 encoder layers are sufficient
   - For DIY: 3-4 layers capture more complex patterns
   - 8 attention heads is optimal for both

3. **Training Dynamics:**
   - GeoLife converges faster (smaller dataset)
   - DIY requires more epochs but larger batches help
   - Early stopping is crucial to prevent overfitting

---

## 10. Configuration Reference

### 10.1 Complete Configuration Options

```yaml
# Random seed for reproducibility
seed: 42

# Data settings
data:
  data_dir: data/geolife_eps20/processed    # Path to processed data
  dataset_prefix: geolife_eps20_prev7        # File prefix
  dataset: geolife                           # Dataset name
  experiment_root: experiments               # Output directory

# Training settings
training:
  if_embed_user: true        # Use user embeddings
  if_embed_poi: false        # Use POI embeddings (if available)
  if_embed_time: true        # Use temporal embeddings
  if_embed_duration: true    # Use duration embeddings
  
  previous_day: 7            # Days of history
  verbose: true              # Print progress
  debug: false               # Debug mode (few batches)
  batch_size: 32             # Training batch size
  print_step: 20             # Print every N batches
  num_workers: 0             # DataLoader workers

# Dataset info (from metadata)
dataset_info:
  total_loc_num: 1187        # Number of locations
  total_user_num: 46         # Number of users

# Embedding settings
embedding:
  base_emb_size: 32          # Main embedding dimension
  poi_original_size: 16      # POI vector size
  time_emb_size: 32          # Time embedding size (for concat mode)

# Model architecture
model:
  networkName: transformer
  num_encoder_layers: 2      # Number of encoder layers
  nhead: 8                   # Number of attention heads
  dim_feedforward: 128       # FFN hidden dimension
  fc_dropout: 0.2            # Output layer dropout

# Optimizer settings
optimiser:
  optimizer: Adam            # Optimizer type
  max_epoch: 100             # Maximum epochs
  lr: 0.001                  # Learning rate
  weight_decay: 0.000001     # L2 regularization
  beta1: 0.9                 # Adam beta1
  beta2: 0.999               # Adam beta2
  num_warmup_epochs: 2       # LR warmup epochs
  num_training_epochs: 50    # Scheduled epochs
  patience: 5                # Early stopping patience
  lr_step_size: 1            # Step LR size
  lr_gamma: 0.1              # LR decay factor
```

### 10.2 Parameter Guidelines

| Parameter | Small Dataset | Large Dataset | Notes |
|-----------|--------------|---------------|-------|
| base_emb_size | 32-64 | 64-128 | Higher for more locations |
| num_encoder_layers | 1-2 | 2-4 | Deeper for complex patterns |
| dim_feedforward | 64-128 | 128-512 | 2-4x embedding size |
| batch_size | 16-32 | 128-512 | Larger for stability |
| lr | 0.001 | 0.0005-0.001 | Lower for larger models |
| patience | 3-5 | 5-7 | More for slower convergence |

---

## 11. Usage Guide

### 11.1 Environment Setup

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Navigate to project
cd /data/next_loc_clean_v2
```

### 11.2 Training

```bash
# Train on GeoLife
python src/training/train_MHSA.py --config config/models/config_MHSA_geolife.yaml

# Train on DIY
python src/training/train_MHSA.py --config config/models/config_MHSA_diy.yaml
```

### 11.3 Output Structure

```
experiments/geolife_MHSA_20260101_120000/
├── checkpoints/
│   └── checkpoint.pt        # Best model weights
├── config.yaml              # Saved configuration
├── config_original.yaml     # Original config file
├── training.log             # Training progress log
├── val_results.json         # Validation metrics
└── test_results.json        # Test metrics
```

### 11.4 Loading a Trained Model

```python
import torch
from src.models.baseline.MHSA import MHSA

# Load config
config = EasyDict(load_config("experiments/geolife_MHSA_xxx/config.yaml"))

# Create model
model = MHSA(config=config, total_loc_num=config.total_loc_num)

# Load weights
checkpoint = torch.load("experiments/geolife_MHSA_xxx/checkpoints/checkpoint.pt")
model.load_state_dict(checkpoint)
model.eval()

# Make predictions
with torch.no_grad():
    logits = model(src, context_dict, device)
    predictions = torch.topk(logits, k=5, dim=-1).indices
```

### 11.5 Extracting Attention Maps

```python
# Get attention maps for visualization
attention_maps = model.get_attention_maps(src, context_dict, device)

# attention_maps is a list of [batch, seq_len] tensors
# One per encoder layer
for layer_idx, attn in enumerate(attention_maps):
    print(f"Layer {layer_idx}: {attn.shape}")
```

---

## 12. Troubleshooting

### 12.1 Common Errors

**CUDA Out of Memory:**
```
RuntimeError: CUDA out of memory
```
Solution: Reduce `batch_size` in config

**KeyError 'poi':**
```
KeyError: 'poi'
```
Solution: Set `if_embed_poi: false` in config

**Import Errors:**
```
ModuleNotFoundError: No module named 'src'
```
Solution: Run from project root directory

### 12.2 Performance Issues

**Low Accuracy:**
- Check data preprocessing
- Verify vocabulary size matches metadata
- Try different hyperparameters

**Slow Training:**
- Use GPU if available
- Increase batch size
- Reduce num_workers if CPU bottleneck

**Overfitting:**
- Reduce model size
- Increase dropout
- Decrease patience

### 12.3 Debug Mode

Enable debug mode for quick testing:
```yaml
training:
  debug: true
```
This limits training to ~20 batches per epoch.

---

## 13. References

### 13.1 Academic References

1. **Attention Is All You Need**
   - Vaswani et al., 2017
   - Original Transformer architecture

2. **Context-aware Multi-Head Self-Attentional Neural Network Model for Next Location Prediction**
   - Hong et al., 2023
   - Reference paper for this implementation

3. **BERT: Pre-training of Deep Bidirectional Transformers**
   - Devlin et al., 2018
   - Transformer encoder design principles

### 13.2 Code References

- **Model Implementation:** `src/models/baseline/MHSA.py`
- **Training Script:** `src/training/train_MHSA.py`
- **Evaluation Metrics:** `src/evaluation/metrics.py`
- **Configurations:** `config/models/config_MHSA_*.yaml`

### 13.3 Dataset References

- **GeoLife Dataset:** Microsoft Research Asia GPS Trajectory Dataset
- **DIY Dataset:** Internal mobility dataset

---

## Appendix A: Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| S | Sequence length |
| D | Embedding dimension |
| V | Vocabulary size (locations) |
| U | Number of users |
| H | Number of attention heads |
| d_k | Per-head dimension (D/H) |
| L | Number of encoder layers |

---

## Appendix B: Hyperparameter Tuning Results

See `docs/hyperparameter_tuning_MHSA.md` for complete tuning experiments.

---

*Document generated for next_loc_clean_v2 project*
