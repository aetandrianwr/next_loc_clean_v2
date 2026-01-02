# LSTM Model for Next Location Prediction: Complete Technical Documentation

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Introduction and Problem Statement](#2-introduction-and-problem-statement)
3. [Theoretical Background](#3-theoretical-background)
4. [Architecture Overview](#4-architecture-overview)
5. [Detailed Component Analysis](#5-detailed-component-analysis)
6. [Data Pipeline](#6-data-pipeline)
7. [Training Process](#7-training-process)
8. [Evaluation Metrics](#8-evaluation-metrics)
9. [Experimental Results](#9-experimental-results)
10. [Configuration Reference](#10-configuration-reference)
11. [File Structure and Organization](#11-file-structure-and-organization)
12. [Usage Guide](#12-usage-guide)

---

## 1. Executive Summary

### 1.1 What is This Model?

The **LSTM (Long Short-Term Memory) Model** is a deep learning architecture designed for **next location prediction** - the task of predicting where a person will visit next based on their historical movement patterns.

### 1.2 Key Characteristics

| Aspect | Description |
|--------|-------------|
| **Model Type** | Recurrent Neural Network (LSTM-based) |
| **Task** | Multi-class Classification |
| **Input** | Sequence of visited locations with temporal features |
| **Output** | Probability distribution over all possible locations |
| **Training Paradigm** | Supervised Learning |

### 1.3 Performance Summary

| Dataset | Acc@1 | Acc@5 | Acc@10 | MRR | Parameters |
|---------|-------|-------|--------|-----|------------|
| GeoLife | ~32% | ~56% | ~60% | ~43% | ~500K-900K |
| DIY | ~50%+ | ~77% | ~81% | ~63% | ~2.8M |

---

## 2. Introduction and Problem Statement

### 2.1 The Next Location Prediction Problem

**Definition**: Given a user's historical trajectory (sequence of visited locations with timestamps), predict the most likely next location the user will visit.

```
Input:  [Home → Work → Restaurant → Gym → ...]
Output: Most likely next location (e.g., Home)
```

### 2.2 Why is This Important?

1. **Urban Planning**: Understanding mobility patterns helps city planners design better transportation systems
2. **Personalized Services**: Location-based recommendations, navigation assistance
3. **Traffic Management**: Predicting movement patterns for congestion avoidance
4. **Emergency Response**: Anticipating crowd movements during emergencies

### 2.3 Challenges in Next Location Prediction

| Challenge | Description | How LSTM Addresses It |
|-----------|-------------|----------------------|
| **Sequential Dependencies** | Location visits follow temporal patterns | LSTM's recurrent structure captures sequential dependencies |
| **Variable-Length History** | Different users have different history lengths | Packed sequences handle variable lengths efficiently |
| **Temporal Patterns** | Time of day/week affects behavior | Temporal embeddings encode time features |
| **User Personalization** | Different users have different preferences | User embeddings capture individual behavior |
| **Long-Range Dependencies** | Patterns span multiple time steps | LSTM's gating mechanism preserves information |

### 2.4 Model Role in Research Context

This LSTM model serves as a **baseline** for comparison against more advanced architectures:

```
Performance Hierarchy (Expected):
Pointer Networks > MHSA (Transformer) ≥ LSTM > Markov Models
```

---

## 3. Theoretical Background

### 3.1 Recurrent Neural Networks (RNN) - Foundation

#### 3.1.1 Basic RNN Concept

A Recurrent Neural Network processes sequences by maintaining a **hidden state** that gets updated at each time step.

**Mathematical Formulation:**

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

Where:
- `h_t`: Hidden state at time t
- `x_t`: Input at time t
- `W_hh`, `W_xh`, `W_hy`: Weight matrices
- `b_h`, `b_y`: Bias vectors

#### 3.1.2 The Vanishing Gradient Problem

Standard RNNs suffer from **vanishing gradients** - gradients become extremely small when backpropagating through many time steps, making it difficult to learn long-range dependencies.

```
Gradient at time step t:
∂L/∂h_1 = ∂L/∂h_T · ∏(k=2 to T) ∂h_k/∂h_{k-1}

Problem: If |∂h_k/∂h_{k-1}| < 1 consistently, gradient → 0
```

### 3.2 LSTM - The Solution

#### 3.2.1 LSTM Architecture Overview

LSTM (Long Short-Term Memory) was introduced by Hochreiter & Schmidhuber (1997) to address the vanishing gradient problem through a **gating mechanism**.

**Key Innovation**: LSTM maintains TWO types of state:
1. **Cell State (c_t)**: Long-term memory
2. **Hidden State (h_t)**: Short-term/working memory

```
                    ┌─────────────────────────────────────────┐
                    │              LSTM Cell                   │
                    │                                          │
    c_{t-1} ─────►──┼──► [×] ──────► [+] ─────────►─ c_t       │
                    │     ↑           ↑                        │
                    │     │           │                        │
                    │   f_t         i_t × c̃_t                  │
                    │  (forget)    (input × candidate)         │
                    │     │           │                        │
    h_{t-1} ─────►──┼──►[Gates]◄────[x_t]                      │
                    │     │                                    │
                    │     └──► o_t ──► [×] ──► tanh(c_t) ─► h_t│
                    │         (output)                         │
                    └──────────────────────────────────────────┘
```

#### 3.2.2 LSTM Gates - Detailed Explanation

**1. Forget Gate (f_t)**
- **Purpose**: Decide what information to DISCARD from the cell state
- **Intuition**: "Should I forget where I went yesterday?"
- **Formula**: `f_t = σ(W_f · [h_{t-1}, x_t] + b_f)`
- **Output**: Values between 0 (forget completely) and 1 (keep completely)

**2. Input Gate (i_t)**
- **Purpose**: Decide what NEW information to STORE in the cell state
- **Intuition**: "Is this new location visit important to remember?"
- **Formula**: `i_t = σ(W_i · [h_{t-1}, x_t] + b_i)`

**3. Candidate Cell State (c̃_t)**
- **Purpose**: Create NEW candidate values that COULD be added to the state
- **Formula**: `c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)`

**4. Cell State Update**
- **Purpose**: Update the long-term memory
- **Formula**: `c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t`
- **Intuition**: Forget some old stuff + remember some new stuff

**5. Output Gate (o_t)**
- **Purpose**: Decide what to OUTPUT based on current cell state
- **Intuition**: "What's relevant to predict the next location?"
- **Formula**: `o_t = σ(W_o · [h_{t-1}, x_t] + b_o)`

**6. Hidden State Output**
- **Formula**: `h_t = o_t ⊙ tanh(c_t)`

#### 3.2.3 Why LSTM Works for Location Prediction

| LSTM Feature | Benefit for Location Prediction |
|--------------|--------------------------------|
| **Forget Gate** | Can ignore irrelevant past visits (e.g., one-time trip) |
| **Input Gate** | Can emphasize important patterns (e.g., daily commute) |
| **Cell State** | Maintains long-term routines (e.g., weekly patterns) |
| **Output Gate** | Focuses on contextually relevant information |

### 3.3 Embedding Theory

#### 3.3.1 What are Embeddings?

Embeddings are **dense vector representations** that map discrete entities (locations, users, time slots) to continuous vector spaces where semantic similarity is preserved.

```
Location ID: 42        →  Embedding: [0.23, -0.15, 0.87, ...]
Location ID: 43        →  Embedding: [0.25, -0.12, 0.85, ...]  (similar locations → similar vectors)
Location ID: 1000      →  Embedding: [-0.71, 0.45, -0.22, ...] (different location → different vector)
```

#### 3.3.2 Types of Embeddings in This Model

| Embedding Type | Dimension | Purpose | Cardinality |
|----------------|-----------|---------|-------------|
| Location | base_emb_size (32-96) | Represent visited places | total_loc_num |
| User | hidden_size (128-192) | Capture user preferences | total_user_num |
| Hour | base_emb_size | Time of day pattern | 24 |
| Minute | base_emb_size | Quarter-hour precision | 4 |
| Weekday | base_emb_size | Day of week pattern | 7 |
| Duration | base_emb_size | Stay duration | 96 (2 days / 30 min) |

---

## 4. Architecture Overview

### 4.1 High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        LSTM MODEL ARCHITECTURE                                │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                           INPUT LAYER                                    │ │
│  │                                                                          │ │
│  │   Location IDs     Time Features      Duration      User ID             │ │
│  │   [seq_len, B]     [seq_len, B]      [seq_len, B]   [B]                 │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       EMBEDDING LAYER                                    │ │
│  │                     (AllEmbeddingLSTM)                                   │ │
│  │                                                                          │ │
│  │   ┌───────────────┐ ┌───────────────┐ ┌───────────────┐                 │ │
│  │   │   Location    │ │   Temporal    │ │   Duration    │                 │ │
│  │   │   Embedding   │+│   Embedding   │+│   Embedding   │                 │ │
│  │   │  (d_input)    │ │  (d_input)    │ │  (d_input)    │                 │ │
│  │   └───────────────┘ └───────────────┘ └───────────────┘                 │ │
│  │                           │                                              │ │
│  │                           ▼                                              │ │
│  │                    Combined Embedding                                    │ │
│  │                    [seq_len, B, d_input]                                 │ │
│  │                           │                                              │ │
│  │                      Dropout (0.1)                                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        LSTM ENCODER                                      │ │
│  │                                                                          │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │  pack_padded_sequence (handles variable length)                  │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                           │                                              │ │
│  │                           ▼                                              │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │                    LSTM Layer 1                                  │   │ │
│  │   │        input_size: d_input, hidden_size: hidden_size            │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                           │                                              │ │
│  │                     Dropout (0.2)                                        │ │
│  │                           │                                              │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │                    LSTM Layer 2                                  │   │ │
│  │   │      input_size: hidden_size, hidden_size: hidden_size          │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                           │                                              │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │  pad_packed_sequence (restore padded format)                     │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                           │                                              │ │
│  │                           ▼                                              │ │
│  │              Extract last valid output per sequence                      │ │
│  │                    [B, hidden_size]                                      │ │
│  │                           │                                              │ │
│  │                     Layer Norm                                           │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                     OUTPUT LAYER (FullyConnected)                        │ │
│  │                                                                          │ │
│  │   LSTM Output ────┬──► + ◄── User Embedding                             │ │
│  │                   │         [B, hidden_size]                             │ │
│  │                   ▼                                                       │ │
│  │            Dropout (0.1)                                                 │ │
│  │                   │                                                       │ │
│  │                   ▼                                                       │ │
│  │   ┌─────────────────────────────────────────────────────────────────┐   │ │
│  │   │                  RESIDUAL BLOCK                                  │   │ │
│  │   │   Input ──┬──────────────────────────────┐                       │   │ │
│  │   │           │                              │                       │   │ │
│  │   │           ▼                              │                       │   │ │
│  │   │    Linear (d→2d) → ReLU → Dropout       │                       │   │ │
│  │   │           │                              │                       │   │ │
│  │   │           ▼                              │                       │   │ │
│  │   │    Linear (2d→d) → Dropout              │                       │   │ │
│  │   │           │                              │                       │   │ │
│  │   │           ▼                              ▼                       │   │ │
│  │   │         [+]◄─────────────────────────────┘                       │   │ │
│  │   │           │                                                      │   │ │
│  │   │    BatchNorm1d                                                   │   │ │
│  │   └─────────────────────────────────────────────────────────────────┘   │ │
│  │                   │                                                       │ │
│  │                   ▼                                                       │ │
│  │         Linear (d → total_loc_num)                                       │ │
│  │                   │                                                       │ │
│  │                   ▼                                                       │ │
│  │              Logits [B, total_loc_num]                                   │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                         │
│                                    ▼                                         │
│                          CrossEntropyLoss                                    │
│                            (training)                                        │
│                                                                              │
│                                or                                            │
│                                                                              │
│                             Softmax                                          │
│                           (inference)                                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 4.2 Simplified Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    SIMPLIFIED VIEW                               │
│                                                                 │
│   Location Sequence  ────►  Embeddings  ────►  LSTM  ────►  FC  ────►  Prediction
│   + Time Features                              (2 layers)          (with user emb)
│                                                                 │
│   Example:                                                      │
│   [Home, Work, Gym]   →   [emb1, emb2, emb3]  →  [h1, h2, h3]  →  logits  →  Home
│   + timestamps                                      ↑                        (next)
│                                                   h3 only                    
│                                                (last state)                  
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 Data Flow Dimensions

```
Input Processing:
─────────────────
Location IDs:        [seq_len, batch_size]           e.g., [15, 32]
Time slots:          [seq_len, batch_size]           e.g., [15, 32]
Weekdays:            [seq_len, batch_size]           e.g., [15, 32]
Durations:           [seq_len, batch_size]           e.g., [15, 32]
Sequence lengths:    [batch_size]                    e.g., [32]
User IDs:            [batch_size]                    e.g., [32]

After Embedding:
───────────────
Combined embedding:  [seq_len, batch_size, d_input]  e.g., [15, 32, 32]

After LSTM:
──────────
LSTM output:         [seq_len, batch_size, hidden]   e.g., [15, 32, 128]
Last valid output:   [batch_size, hidden]            e.g., [32, 128]

After FC:
────────
Logits:              [batch_size, total_loc_num]     e.g., [32, 1187]
```

---

## 5. Detailed Component Analysis

### 5.1 AllEmbeddingLSTM - Combined Embedding Layer

#### 5.1.1 Purpose and Responsibility

The `AllEmbeddingLSTM` class combines multiple feature embeddings into a single representation for each location in the sequence.

**Key Difference from Transformer Version**: No positional encoding (LSTM inherently captures sequence order through recurrence).

#### 5.1.2 Component Breakdown

```python
class AllEmbeddingLSTM(nn.Module):
    def __init__(self, d_input, config, total_loc_num, emb_info="all", emb_type="add"):
```

| Component | Type | Purpose |
|-----------|------|---------|
| `emb_loc` | nn.Embedding | Map location IDs to dense vectors |
| `temporal_embedding` | TemporalEmbedding | Encode time features |
| `emb_duration` | nn.Embedding | Encode stay duration |
| `poi_net` | POINet (optional) | Process POI features |
| `dropout` | nn.Dropout | Regularization (p=0.1) |

#### 5.1.3 Embedding Combination Strategy

The model uses **additive combination** of embeddings:

```
final_embedding = loc_emb + time_emb + duration_emb (+ poi_emb if enabled)
```

**Why Additive?**
- Preserves dimensionality (no increase in model size)
- Allows features to "blend" naturally
- Each feature can independently influence any dimension
- Matches the approach in the MHSA baseline paper

#### 5.1.4 Forward Pass Visualization

```
Input: src [seq_len, batch], context_dict {time, weekday, duration, ...}
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    emb_loc(src)                               │
│              [seq_len, batch, d_input]                       │
└──────────────────────────────────────────────────────────────┘
       │
       + ◄───────────────────────────────────────────┐
       │                                             │
       ▼                                             │
┌──────────────────────────────────────────────────────────────┐
│            temporal_embedding(time, weekday)                 │
│         (hour_emb + minute_emb + weekday_emb)               │
│              [seq_len, batch, d_input]                       │
└──────────────────────────────────────────────────────────────┘
       │
       + ◄───────────────────────────────────────────┐
       │                                             │
       ▼                                             │
┌──────────────────────────────────────────────────────────────┐
│               emb_duration(duration)                         │
│              [seq_len, batch, d_input]                       │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   Dropout(0.1)
       │
       ▼
Output: [seq_len, batch, d_input]
```

### 5.2 TemporalEmbedding - Time Feature Encoder

#### 5.2.1 Purpose

Encode temporal information (when the visit occurred) into a learnable representation.

#### 5.2.2 Time Decomposition

Time is decomposed into three components:

```
Original: 14:45 on Wednesday

Decomposition:
├── Hour: 14 (0-23)
├── Quarter: 3 (0-3, representing 45 minutes = 3rd quarter)
└── Weekday: 3 (0-6, Wednesday)

Final embedding = hour_emb[14] + minute_emb[3] + weekday_emb[3]
```

#### 5.2.3 Supported Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| `"all"` | Separate hour, minute, weekday embeddings | Full temporal resolution |
| `"time"` | Combined time slot embedding (96 slots/day) | Simpler time encoding |
| `"weekday"` | Only weekday embedding | Focus on weekly patterns |

### 5.3 LSTM Layer - Sequence Processor

#### 5.3.1 Configuration

```python
self.lstm = nn.LSTM(
    input_size=self.d_input,      # Embedding dimension
    hidden_size=self.hidden_size,  # Output dimension
    num_layers=self.num_layers,    # Depth (default: 2)
    dropout=lstm_dropout,          # Between layers
    batch_first=False,             # [seq_len, batch, features]
    bidirectional=False            # Unidirectional (causal)
)
```

#### 5.3.2 Why Unidirectional?

For next location prediction, we can only use **past information** to predict the future. Bidirectional LSTM would "cheat" by using future locations.

```
Correct (Unidirectional):
Home → Work → Gym → [PREDICT: ?]
 ↓      ↓      ↓
 h1  → h2  → h3 → prediction

Incorrect (Bidirectional - would leak future info):
Home ← Work ← Gym → [PREDICT: ?]
```

#### 5.3.3 Handling Variable-Length Sequences

**Problem**: Different users have different history lengths.

**Solution**: PyTorch's packed sequences.

```python
# Pack sequences (removes padding from computation)
packed_emb = pack_padded_sequence(emb, seq_len.cpu(), batch_first=False, enforce_sorted=False)

# Process through LSTM (efficient - ignores padding)
packed_output, (h_n, c_n) = self.lstm(packed_emb)

# Unpack back to padded format
output, _ = pad_packed_sequence(packed_output, batch_first=False)
```

#### 5.3.4 Extracting Last Valid Output

```python
# output shape: [seq_len, batch, hidden_size]
# seq_len contains the actual length of each sequence

# For each sequence, get the output at the last VALID position
out = output.gather(
    0,  # dimension to gather from
    seq_len.view([1, -1, 1]).expand([1, output.shape[1], output.shape[-1]]) - 1
).squeeze(0)
# Result: [batch, hidden_size]
```

**Visual Example:**

```
Sequence 1 (length=3): [h1, h2, h3*, padding, padding]
Sequence 2 (length=5): [h1, h2, h3, h4, h5*]
                            (* = selected for output)
```

### 5.4 FullyConnected - Output Layer

#### 5.4.1 Purpose

Transform the LSTM hidden state into a probability distribution over all locations.

#### 5.4.2 Architecture

```
Input: LSTM output [batch, hidden_size]
       │
       + ◄── User Embedding [batch, hidden_size]
       │
       ▼
   Dropout(0.1)
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│                    RESIDUAL BLOCK                            │
│                                                              │
│   x ───┬───────────────────────────────────┐                │
│        │                                   │                │
│        ▼                                   │                │
│   Linear(d, 2d) → ReLU → Dropout(0.2)     │                │
│        │                                   │                │
│        ▼                                   │                │
│   Linear(2d, d) → Dropout(0.2)            │                │
│        │                                   │                │
│        ▼                                   ▼                │
│      [+]◄──────────────────────────────────┘                │
│        │                                                    │
│        ▼                                                    │
│   BatchNorm1d(d)                                            │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
   Linear(hidden_size, total_loc_num)
       │
       ▼
Output: Logits [batch, total_loc_num]
```

#### 5.4.3 User Embedding Integration

```python
if self.if_embed_user:
    out = out + self.emb_user(user)  # Additive personalization
```

**Why Add User Embedding?**
- Different users have different preferences
- User embedding shifts the prediction distribution toward user-specific locations
- Allows model to learn "user A prefers coffee shops, user B prefers gyms"

#### 5.4.4 Why Residual Connection?

Residual connections help with:
1. **Gradient Flow**: Gradients can skip the transformation if needed
2. **Identity Mapping**: Network can learn to keep important information unchanged
3. **Training Stability**: Easier to optimize deep networks

### 5.5 Weight Initialization

#### 5.5.1 Strategy

```python
def _init_weights(self):
    for name, param in self.named_parameters():
        if 'weight_ih' in name:
            # Input-hidden: Xavier uniform
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-hidden: Orthogonal (preserves gradient magnitude)
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # Forget gate bias = 1 (encourage remembering initially)
            n = param.size(0)
            param.data[n // 4:n // 2].fill_(1.0)
```

#### 5.5.2 Forget Gate Bias = 1

**Why?**
- Initial forget gate should be "open" (value close to 1)
- This allows gradients to flow through the cell state initially
- The model can learn to forget later if needed
- Without this, the network might initially forget everything

---

## 6. Data Pipeline

### 6.1 Data Flow Overview

```
Raw GPS Data
     │
     ▼
┌─────────────────────────────────────┐
│  Script 1: Raw to Interim           │
│  - GPS clustering (DBSCAN)          │
│  - Staypoint detection              │
│  - Location ID assignment           │
└─────────────────────────────────────┘
     │
     ▼
Intermediate CSV
     │
     ▼
┌─────────────────────────────────────┐
│  Script 2: Interim to Processed     │
│  - Train/Val/Test split per user    │
│  - Location ID encoding             │
│  - Sequence generation              │
│  - Feature extraction               │
└─────────────────────────────────────┘
     │
     ▼
Pickle Files (.pk)
     │
     ▼
┌─────────────────────────────────────┐
│  Training Script                     │
│  - LocationDataset class            │
│  - DataLoader with collate_fn       │
│  - Batch processing                 │
└─────────────────────────────────────┘
```

### 6.2 Sequence Generation Logic

#### 6.2.1 Previous Days Parameter

The `previous_day` parameter controls how much history to consider:

```
Example: previous_day = 7

Day 1  Day 2  Day 3  Day 4  Day 5  Day 6  Day 7  Day 8
[Home] [Work] [Gym]  [Home] [Work] [Cafe] [Work] [???]
                                                  ↑
                     └────────────────────────────┘
                           History Window (7 days)
```

#### 6.2.2 Sequence Construction

For each valid target staypoint:
1. Find all staypoints in the previous N days
2. Require at least 3 historical staypoints
3. Extract features for each historical staypoint

```python
# Example sequence structure
{
    "X": [102, 45, 103, 45, 102],      # Location sequence
    "Y": 89,                            # Target location
    "user_X": [1, 1, 1, 1, 1],         # User ID (same for all)
    "weekday_X": [0, 0, 1, 2, 2],      # Day of week
    "start_min_X": [480, 720, 420, 480, 720],  # Start time (minutes from midnight)
    "dur_X": [240, 60, 90, 240, 60],   # Duration in minutes
    "diff": [6, 5, 4, 2, 1]            # Days until target
}
```

### 6.3 Dataset Class

```python
class LocationDataset(Dataset):
    def __init__(self, data_path, dataset_name="geolife"):
        self.data = pickle.load(open(data_path, "rb"))
    
    def __getitem__(self, idx):
        selected = self.data[idx]
        
        return_dict = {}
        x = torch.tensor(selected["X"])               # Location sequence
        y = torch.tensor(selected["Y"])               # Target
        
        return_dict["user"] = torch.tensor(selected["user_X"][0])
        return_dict["time"] = torch.tensor(selected["start_min_X"] // 15)  # 15-min slots
        return_dict["weekday"] = torch.tensor(selected["weekday_X"])
        return_dict["duration"] = torch.tensor(selected["dur_X"] // 30, dtype=torch.long)
        return_dict["diff"] = torch.tensor(selected["diff"])
        
        return x, y, return_dict
```

### 6.4 Collate Function

Handles batching of variable-length sequences:

```python
def collate_fn(batch):
    x_batch, y_batch = [], []
    x_dict_batch = {"len": []}
    
    for src_sample, tgt_sample, return_dict in batch:
        x_batch.append(src_sample)
        y_batch.append(tgt_sample)
        x_dict_batch["len"].append(len(src_sample))
        # ... collect other features
    
    # Pad sequences to same length
    x_batch = pad_sequence(x_batch)  # [max_len, batch]
    y_batch = torch.tensor(y_batch)
    
    return x_batch, y_batch, x_dict_batch
```

---

## 7. Training Process

### 7.1 Training Loop Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    TRAINING PIPELINE                           │
│                                                                │
│  for epoch in range(max_epoch):                               │
│      │                                                         │
│      ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              TRAINING EPOCH                               │ │
│  │                                                           │ │
│  │  for batch in train_loader:                              │ │
│  │      1. Move data to device (GPU)                        │ │
│  │      2. Forward pass: logits = model(x, x_dict, device)  │ │
│  │      3. Compute loss: CrossEntropyLoss(logits, y)        │ │
│  │      4. Backward pass: loss.backward()                   │ │
│  │      5. Gradient clipping: clip_grad_norm_(params, 1)    │ │
│  │      6. Optimizer step: optimizer.step()                 │ │
│  │      7. Learning rate schedule step                      │ │
│  └──────────────────────────────────────────────────────────┘ │
│      │                                                         │
│      ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              VALIDATION                                   │ │
│  │                                                           │ │
│  │  model.eval()                                            │ │
│  │  with torch.no_grad():                                   │ │
│  │      Compute validation loss and metrics                 │ │
│  └──────────────────────────────────────────────────────────┘ │
│      │                                                         │
│      ▼                                                         │
│  ┌──────────────────────────────────────────────────────────┐ │
│  │              EARLY STOPPING CHECK                         │ │
│  │                                                           │ │
│  │  if val_loss improved:                                   │ │
│  │      Save checkpoint                                     │ │
│  │      Reset patience counter                              │ │
│  │  else:                                                   │ │
│  │      Increment patience counter                          │ │
│  │      if counter >= patience:                             │ │
│  │          Reduce learning rate OR stop training           │ │
│  └──────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────┘
```

### 7.2 Loss Function

**Cross-Entropy Loss** for multi-class classification:

```python
CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
loss = CEL(logits.view(-1, logits.shape[-1]), y.reshape(-1))
```

**Mathematical Definition:**

```
L = -∑(y_true * log(softmax(logits)))

Where:
- y_true: One-hot encoded true class
- logits: Raw model output
- softmax converts logits to probabilities
```

**Why ignore_index=0?**
- Index 0 is reserved for padding
- Should not contribute to the loss

### 7.3 Optimizer Configuration

```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.001,           # Initial learning rate
    betas=(0.9, 0.999), # Momentum parameters
    weight_decay=1e-6   # L2 regularization
)
```

### 7.4 Learning Rate Schedule

Two-phase learning rate schedule:

**Phase 1: Warmup + Linear Decay**
```python
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=len(train_loader) * config.num_warmup_epochs,  # 2 epochs
    num_training_steps=len(train_loader) * config.num_training_epochs  # 50 epochs
)
```

```
LR
 │
 │   ╱──────────────────────────╲
 │  ╱                            ╲
 │ ╱                              ╲
 │╱                                ╲
 └─────────────────────────────────────▶ epochs
   │ warmup │       decay           │
```

**Phase 2: Step Decay on Early Stopping**
```python
scheduler_ES = StepLR(optimizer, step_size=1, gamma=0.1)
# Reduces LR by 10x when early stopping triggers
```

### 7.5 Early Stopping Strategy

```python
class EarlyStopping:
    def __init__(self, patience=3, delta=0.001):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
    
    def __call__(self, val_loss, model):
        if val_loss < self.best_score - self.delta:
            # Improvement - save model
            self.save_checkpoint(model)
            self.best_score = val_loss
            self.counter = 0
        else:
            # No improvement
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
```

**Three-Stage Training:**
1. Train until early stopping (patience=3)
2. Load best checkpoint, reduce LR by 10x, continue
3. Repeat early stopping
4. If triggered 3 times total, stop training

### 7.6 Gradient Clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Purpose:**
- Prevents exploding gradients
- Stabilizes training
- Especially important for RNNs which can have large gradient magnitudes

---

## 8. Evaluation Metrics

### 8.1 Metrics Overview

| Metric | Full Name | Range | Interpretation |
|--------|-----------|-------|----------------|
| Acc@1 | Top-1 Accuracy | 0-100% | % of correct first predictions |
| Acc@5 | Top-5 Accuracy | 0-100% | % where true label in top 5 |
| Acc@10 | Top-10 Accuracy | 0-100% | % where true label in top 10 |
| MRR | Mean Reciprocal Rank | 0-100% | Average of 1/rank |
| NDCG@10 | Normalized DCG | 0-100% | Ranking quality |
| F1 | Weighted F1 Score | 0-1 | Balance of precision/recall |

### 8.2 Top-K Accuracy

**Definition**: Percentage of predictions where the true label appears in the top K predicted classes.

```python
# For k in [1, 3, 5, 10]:
prediction = torch.topk(logits, k=k, dim=-1).indices
correct = torch.eq(true_y[:, None], prediction).any(dim=1).sum()
acc_k = correct / total * 100
```

**Example:**
```
Prediction (sorted by probability): [Coffee Shop, Home, Work, Gym, ...]
True label: Work

Acc@1: ❌ (Work is not #1)
Acc@3: ✓ (Work is in top 3)
Acc@5: ✓ (Work is in top 3, so also in top 5)
```

### 8.3 Mean Reciprocal Rank (MRR)

**Definition**: Average of the reciprocal of the rank of the true label.

```
MRR = (1/N) * Σ(1/rank_i)

Where rank_i is the position of the true label in the sorted predictions
```

**Example:**
```
Sample 1: True label at rank 1 → 1/1 = 1.0
Sample 2: True label at rank 3 → 1/3 = 0.33
Sample 3: True label at rank 2 → 1/2 = 0.5

MRR = (1.0 + 0.33 + 0.5) / 3 = 0.61
```

**Why MRR?**
- Rewards getting the correct answer ranked higher
- More nuanced than binary accuracy
- Important when users see ranked lists

### 8.4 NDCG@10 (Normalized Discounted Cumulative Gain)

**Definition**: Measures ranking quality with position-based discounting.

```
DCG@10 = Σ(rel_i / log2(i + 1))  for i = 1 to 10

NDCG@10 = DCG@10 / Ideal_DCG@10
```

**For binary relevance (one correct answer):**
```
NDCG = 1 / log2(rank + 1)

If true label at rank 1: 1/log2(2) = 1.0
If true label at rank 2: 1/log2(3) ≈ 0.63
If true label at rank 3: 1/log2(4) = 0.5
If true label at rank 10: 1/log2(11) ≈ 0.29
If true label at rank >10: 0
```

### 8.5 F1 Score (Weighted)

**Definition**: Harmonic mean of precision and recall, weighted by class frequency.

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 * (Precision * Recall) / (Precision + Recall)

Weighted F1 = Σ(F1_class * support_class) / total_support
```

---

## 9. Experimental Results

### 9.1 GeoLife Dataset Results

**Dataset Statistics:**
| Attribute | Value |
|-----------|-------|
| Users | 45 |
| Locations | 1,185 |
| Train Sequences | 7,424 |
| Val Sequences | 3,334 |
| Test Sequences | 3,502 |
| Previous Days | 7 |

**Performance (Test Set):**
| Metric | Value |
|--------|-------|
| Acc@1 | 32.12% |
| Acc@5 | 56.25% |
| Acc@10 | 60.11% |
| MRR | 43.15% |
| NDCG | 47.03% |
| F1 | 0.201 |

**Model Configuration:**
| Parameter | Value |
|-----------|-------|
| base_emb_size | 32 |
| lstm_hidden_size | 128 |
| lstm_num_layers | 2 |
| Total Parameters | ~932K |

### 9.2 DIY Dataset Results

**Dataset Statistics:**
| Attribute | Value |
|-----------|-------|
| Users | 692 |
| Locations | 7,036 |
| Previous Days | 7 |

**Expected Performance:**
| Metric | Value |
|--------|-------|
| Acc@1 | ~50%+ |
| Acc@5 | ~77% |
| Acc@10 | ~81% |
| MRR | ~63% |

**Model Configuration:**
| Parameter | Value |
|-----------|-------|
| base_emb_size | 96 |
| lstm_hidden_size | 192 |
| lstm_num_layers | 2 |
| Total Parameters | ~2.85M |

### 9.3 Training Dynamics

**Typical Training Curve (GeoLife):**

```
Acc@1
  │
35│                    ●●●●●●●●●●●●●●●●●●●●●●●
  │                 ●●●
  │              ●●●
30│           ●●●
  │        ●●●
  │     ●●●
25│   ●●
  │  ●
  │ ●
20│●
  │
  └────────────────────────────────────────────▶ Epoch
    1  2  3  4  5  6  7  8  9  10 ... 19
```

**Observations:**
- Rapid improvement in first 5-8 epochs
- Convergence around epoch 8-10
- Early stopping typically triggers around epoch 8
- LR reduction extends training for few more epochs

---

## 10. Configuration Reference

### 10.1 GeoLife Configuration

```yaml
# config/models/config_LSTM_geolife.yaml

seed: 42

# Data settings
data:
  data_dir: data/geolife_eps20/processed
  dataset_prefix: geolife_eps20_prev7
  dataset: geolife
  experiment_root: experiments

# Training settings
training:
  if_embed_user: true       # Include user embedding
  if_embed_poi: false       # No POI features
  if_embed_time: true       # Include temporal embedding
  if_embed_duration: true   # Include duration embedding
  
  previous_day: 7           # History window
  verbose: true
  debug: false
  batch_size: 32
  print_step: 20
  num_workers: 0

# Dataset info
dataset_info:
  total_loc_num: 1187
  total_user_num: 46

# Embedding settings
embedding:
  base_emb_size: 32
  poi_original_size: 16

# Model architecture
model:
  networkName: lstm
  lstm_hidden_size: 128
  lstm_num_layers: 2
  lstm_dropout: 0.2
  fc_dropout: 0.2

# Optimizer settings
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
  patience: 3
  lr_step_size: 1
  lr_gamma: 0.1
```

### 10.2 DIY Configuration

```yaml
# config/models/config_LSTM_diy.yaml

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
  batch_size: 256          # Larger batch for larger dataset
  print_step: 10
  num_workers: 0

dataset_info:
  total_loc_num: 7038
  total_user_num: 693

embedding:
  base_emb_size: 96        # Larger embedding for more locations
  poi_original_size: 16

model:
  networkName: lstm
  lstm_hidden_size: 192    # Larger hidden size
  lstm_num_layers: 2
  lstm_dropout: 0.2
  fc_dropout: 0.1          # Less dropout for larger dataset

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
  patience: 3
  lr_step_size: 1
  lr_gamma: 0.1
```

### 10.3 Parameter Scaling Guidelines

| Dataset Size | base_emb_size | lstm_hidden_size | batch_size | Target Params |
|--------------|---------------|------------------|------------|---------------|
| Small (~1K locs) | 32 | 64-128 | 32 | <500K |
| Medium (~5K locs) | 64-96 | 128-192 | 128-256 | 1-3M |
| Large (~10K+ locs) | 128 | 256 | 256-512 | 3-5M |

---

## 11. File Structure and Organization

### 11.1 Project Structure

```
next_loc_clean_v2/
├── src/
│   ├── models/
│   │   └── baseline/
│   │       └── LSTM.py              # LSTM model definition
│   ├── training/
│   │   └── train_LSTM.py            # Training script
│   └── evaluation/
│       └── metrics.py               # Evaluation metrics
│
├── config/
│   └── models/
│       ├── config_LSTM_geolife.yaml
│       └── config_LSTM_diy.yaml
│
├── data/
│   ├── geolife_eps20/
│   │   ├── interim/                 # Intermediate data
│   │   └── processed/               # Final .pk files
│   │       ├── geolife_eps20_prev7_train.pk
│   │       ├── geolife_eps20_prev7_validation.pk
│   │       ├── geolife_eps20_prev7_test.pk
│   │       └── geolife_eps20_prev7_metadata.json
│   │
│   └── diy_eps50/
│       └── processed/
│
├── preprocessing/
│   ├── geolife_1_raw_to_interim.py
│   └── geolife_2_interim_to_processed.py
│
├── experiments/
│   └── {dataset}_{model}_{timestamp}/
│       ├── checkpoints/
│       │   └── checkpoint.pt        # Best model weights
│       ├── training.log             # Training logs
│       ├── config.yaml              # Used configuration
│       ├── config_original.yaml
│       ├── val_results.json
│       └── test_results.json
│
└── docs/
    └── docs_LSTM/                   # This documentation
```

### 11.2 Key Files Description

| File | Purpose |
|------|---------|
| `LSTM.py` | Model architecture definition |
| `train_LSTM.py` | Training loop and experiment management |
| `metrics.py` | Evaluation metric calculations |
| `config_LSTM_*.yaml` | Hyperparameter configurations |
| `*_train.pk` | Training data (preprocessed sequences) |
| `*_metadata.json` | Dataset statistics and configuration |

---

## 12. Usage Guide

### 12.1 Quick Start

```bash
# 1. Activate environment
conda activate mlenv

# 2. Train on GeoLife
python src/training/train_LSTM.py --config config/models/config_LSTM_geolife.yaml

# 3. Train on DIY
python src/training/train_LSTM.py --config config/models/config_LSTM_diy.yaml
```

### 12.2 Expected Output

```
Using device: cuda
Experiment directory: experiments/geolife_LSTM_20260102_120000
Data loaders: train=232, val=105, test=110
Total trainable parameters: 483291

=== Epoch 1 ===
Epoch 1, 8.6%	 loss: 7.465 acc@1: 0.00 mrr: 0.49, ndcg: 0.24, took: 1.51s
...
Validation - loss: 4.7039, acc@1: 29.09%, f1: 20.20%, mrr: 40.67%, ndcg: 44.93%

=== Epoch 2 ===
...

Training finished.	 Time: 220.46s.	 acc@1: 34.97%

=== Test Results ===
Test Results:
  acc@1: 32.12%
  acc@5: 56.25%
  acc@10: 60.11%
  mrr: 43.15%
  ndcg: 47.03%
  f1: 20.14%

Results saved to: experiments/geolife_LSTM_20260102_120000
```

### 12.3 Loading a Trained Model

```python
import torch
from src.models.baseline.LSTM import LSTMModel

# Load configuration
config = load_config("config/models/config_LSTM_geolife.yaml")

# Create model
model = LSTMModel(config=config, total_loc_num=1187)

# Load weights
checkpoint = torch.load("experiments/geolife_LSTM_xxx/checkpoints/checkpoint.pt")
model.load_state_dict(checkpoint)

# Inference
model.eval()
with torch.no_grad():
    logits = model(x, x_dict, device)
    predictions = torch.argmax(logits, dim=-1)
```

### 12.4 Custom Dataset Preparation

To use this model on a new dataset:

1. **Preprocess data** to create sequences with required features:
   - `X`: Location ID sequence
   - `Y`: Target location ID
   - `user_X`: User ID
   - `weekday_X`: Day of week (0-6)
   - `start_min_X`: Start time in minutes
   - `dur_X`: Duration in minutes
   - `diff`: Days until target

2. **Create metadata.json** with:
   - `total_loc_num`: Maximum location ID + 1
   - `total_user_num`: Maximum user ID + 1

3. **Create configuration** file with appropriate parameters

4. **Run training** with your config file

---

## Appendix A: Mathematical Notation Reference

| Symbol | Meaning |
|--------|---------|
| B | Batch size |
| T | Sequence length |
| d | Embedding dimension (base_emb_size) |
| h | Hidden size (lstm_hidden_size) |
| L | Number of locations (total_loc_num) |
| U | Number of users (total_user_num) |
| σ | Sigmoid function |
| ⊙ | Element-wise multiplication (Hadamard product) |

## Appendix B: Glossary

| Term | Definition |
|------|------------|
| **Staypoint** | A location where a user stayed for a significant duration |
| **Sequence** | Ordered list of staypoints within a time window |
| **Epoch** | One complete pass through the training data |
| **Batch** | Subset of training samples processed together |
| **Embedding** | Dense vector representation of a discrete entity |
| **Hidden State** | Internal representation maintained by the LSTM |
| **Cell State** | Long-term memory in LSTM |
| **Logits** | Raw model output before softmax |

---

**Document Version**: 1.0
**Last Updated**: January 2026
**Author**: Generated from source code analysis
