# Model Architectures

## Table of Contents

1. [Overview](#overview)
2. [Pointer V45 (Proposed Model)](#pointer-v45-proposed-model)
3. [MHSA (Transformer Baseline)](#mhsa-transformer-baseline)
4. [LSTM (Recurrent Baseline)](#lstm-recurrent-baseline)
5. [Architecture Comparison](#architecture-comparison)
6. [Parameter Count Analysis](#parameter-count-analysis)

---

## Overview

This project compares three deep learning architectures for next location prediction:

| Model | Type | Key Feature | Paper Reference |
|-------|------|-------------|-----------------|
| **Pointer V45** | Hybrid | Pointer mechanism + Generation | Vinyals et al., 2015 |
| **MHSA** | Transformer | Multi-head self-attention | Vaswani et al., 2017 |
| **LSTM** | Recurrent | Sequential hidden state | Hochreiter & Schmidhuber, 1997 |

All models share the same input format:
- Location sequence: Previous 7 days of location visits
- Temporal features: Time of day, day of week, duration
- User embedding: User-specific patterns

---

## Pointer V45 (Proposed Model)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    POINTER V45 ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ┌────────────────────────────────────────┐                    │
│  │ Location Sequence [seq_len, batch]     │                    │
│  │ User IDs [batch]                       │                    │
│  │ Time, Weekday, Duration, Recency       │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │           EMBEDDING LAYER              │                    │
│  │  loc_emb + user_emb + temporal_emb     │                    │
│  │  + pos_from_end_emb                    │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │     INPUT PROJECTION + LAYER NORM      │                    │
│  │     Linear(input_dim → d_model)        │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │     SINUSOIDAL POSITIONAL ENCODING     │                    │
│  │     PE(pos, 2i) = sin(pos/10000^(2i/d))│                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │      TRANSFORMER ENCODER               │                    │
│  │   ┌──────────────────────────────┐    │                    │
│  │   │  LayerNorm (Pre-Norm)        │    │                    │
│  │   │  Multi-Head Self-Attention   │ ×N │                    │
│  │   │  LayerNorm                   │    │                    │
│  │   │  FFN (GELU activation)       │    │                    │
│  │   └──────────────────────────────┘    │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│         ┌───────────┴───────────┐                              │
│         │                       │                               │
│         ▼                       ▼                               │
│  ┌──────────────┐      ┌──────────────┐                        │
│  │   POINTER    │      │  GENERATION  │                        │
│  │   MECHANISM  │      │     HEAD     │                        │
│  │              │      │              │                        │
│  │ Attend to    │      │ Linear →     │                        │
│  │ input locs   │      │ num_locs     │                        │
│  │ with pos     │      │              │                        │
│  │ bias         │      │              │                        │
│  └──────┬───────┘      └──────┬───────┘                        │
│         │                     │                                 │
│         └─────────┬───────────┘                                │
│                   ▼                                             │
│  ┌────────────────────────────────────────┐                    │
│  │        POINTER-GENERATION GATE         │                    │
│  │   g = σ(Linear(Linear(h) + GELU))     │                    │
│  │                                        │                    │
│  │   final = g × pointer + (1-g) × gen   │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │       LOG SOFTMAX OUTPUT               │                    │
│  │   [batch, num_locations]               │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Embedding Layer

```python
# Core embeddings
self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)

# Temporal embeddings (each d_model // 4)
self.time_emb = nn.Embedding(97, d_model // 4)      # 96 time slots + padding
self.weekday_emb = nn.Embedding(8, d_model // 4)    # 7 days + padding
self.recency_emb = nn.Embedding(9, d_model // 4)    # 8 recency levels + padding
self.duration_emb = nn.Embedding(100, d_model // 4) # Duration buckets

# Position from end (important for pointer mechanism)
self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
```

**Total embedding dimension**: `d_model × 2 + d_model // 4 × 5 = d_model × 3.25`

#### 2. Pointer Mechanism with Position Bias

```python
def compute_pointer_scores(self, encoder_output, lengths):
    """Compute pointer attention scores with position bias."""
    batch_size, seq_len, d_model = encoder_output.shape
    
    # Get last token representation as query
    last_hidden = encoder_output[torch.arange(batch_size), lengths - 1]  # [B, d]
    
    # Project query and keys
    query = self.pointer_query(last_hidden)  # [B, d]
    keys = self.pointer_key(encoder_output)   # [B, S, d]
    
    # Compute attention scores
    scores = torch.einsum('bd,bsd->bs', query, keys) / math.sqrt(d_model)
    
    # Add position bias (learnable parameter)
    position_bias = self.position_bias[:seq_len]  # [S]
    scores = scores + position_bias
    
    # Mask padding positions
    mask = torch.arange(seq_len).expand(batch_size, -1) >= lengths.unsqueeze(1)
    scores = scores.masked_fill(mask, float('-inf'))
    
    return F.softmax(scores, dim=-1)  # [B, S]
```

**Position bias** encourages the model to attend to recent locations, which is crucial for mobility prediction.

#### 3. Pointer-Generation Gate

```python
self.ptr_gen_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Sigmoid()
)

# Usage:
gate = self.ptr_gen_gate(last_hidden)  # [B, 1]
final_dist = gate * pointer_dist + (1 - gate) * gen_dist
```

The gate learns when to **copy** from input (pointer) vs. **generate** from full vocabulary.

### Mathematical Formulation

**Pointer Distribution**:
$$P_{ptr}(l) = \sum_{j: x_j = l} \alpha_j$$

where $\alpha_j$ is the attention weight for position $j$, and $x_j$ is the location at position $j$.

**Generation Distribution**:
$$P_{gen}(l) = \text{softmax}(W_o h_{last})_l$$

**Final Distribution**:
$$P(l) = g \cdot P_{ptr}(l) + (1 - g) \cdot P_{gen}(l)$$

where $g \in [0, 1]$ is the pointer-generation gate.

---

## MHSA (Transformer Baseline)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    MHSA ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ┌────────────────────────────────────────┐                    │
│  │ Location, User, Time, Duration         │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │           ALL EMBEDDING                │                    │
│  │  loc_emb + user_emb + time_emb         │                    │
│  │  + duration_emb (optional POI)         │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │     SINUSOIDAL POSITIONAL ENCODING     │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │      TRANSFORMER ENCODER               │                    │
│  │   ┌──────────────────────────────┐    │                    │
│  │   │  Multi-Head Self-Attention   │    │                    │
│  │   │  Add & Norm                  │ ×N │                    │
│  │   │  Feed-Forward Network        │    │                    │
│  │   │  Add & Norm                  │    │                    │
│  │   └──────────────────────────────┘    │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │     FULLY CONNECTED OUTPUT             │                    │
│  │  Linear(d_model → num_locations)       │                    │
│  │  Dropout + Residual (optional)         │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │       SOFTMAX OUTPUT                   │                    │
│  │   [batch, num_locations]               │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. Temporal Embedding

```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        self.minute_embed = nn.Embedding(4, d_input)   # Quarter-hour
        self.hour_embed = nn.Embedding(24, d_input)    # Hour
        self.weekday_embed = nn.Embedding(7, d_input)  # Day of week
    
    def forward(self, time, weekday):
        hour = time // 4
        minutes = time % 4
        return self.minute_embed(minutes) + self.hour_embed(hour) + self.weekday_embed(weekday)
```

#### 2. Standard Transformer Encoder

```python
class MHSA(nn.Module):
    def __init__(self, config, total_loc_num):
        # Embedding layer
        self.all_embed = AllEmbedding(d_input, config, total_loc_num)
        
        # Standard Transformer encoder (Post-Norm)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_input,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=0.1,
            activation='relu',  # Standard ReLU
            batch_first=False,  # [seq, batch, d]
            norm_first=False    # Post-norm (standard)
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.fc = nn.Linear(d_input, total_loc_num)
```

### Differences from Pointer V45

| Aspect | MHSA | Pointer V45 |
|--------|------|-------------|
| Normalization | Post-norm | Pre-norm |
| Activation | ReLU | GELU |
| Output | Direct classification | Pointer + Generation |
| Position | Standard sinusoidal | + Position-from-end |

---

## LSTM (Recurrent Baseline)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM ARCHITECTURE                             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  INPUT                                                          │
│  ┌────────────────────────────────────────┐                    │
│  │ Location, User, Time, Duration         │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │           ALL EMBEDDING                │                    │
│  │  loc_emb + user_emb + time_emb         │                    │
│  │  + duration_emb (NO positional enc)    │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │            LSTM LAYERS                 │                    │
│  │   ┌──────────────────────────────┐    │                    │
│  │   │  Input Gate: i_t             │    │                    │
│  │   │  Forget Gate: f_t            │    │                    │
│  │   │  Cell State: c_t             │ ×N │                    │
│  │   │  Output Gate: o_t            │    │                    │
│  │   │  Hidden State: h_t           │    │                    │
│  │   │  Dropout (between layers)    │    │                    │
│  │   └──────────────────────────────┘    │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│           Take last hidden state h_T                           │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │     FULLY CONNECTED OUTPUT             │                    │
│  │  Linear(hidden_size → num_locations)   │                    │
│  │  Dropout                               │                    │
│  └───────────────────┬────────────────────┘                    │
│                      │                                          │
│                      ▼                                          │
│  ┌────────────────────────────────────────┐                    │
│  │       SOFTMAX OUTPUT                   │                    │
│  │   [batch, num_locations]               │                    │
│  └────────────────────────────────────────┘                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### LSTM Equations

**Input Gate**: $i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$

**Forget Gate**: $f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$

**Cell Candidate**: $\tilde{c}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$

**Cell State**: $c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$

**Output Gate**: $o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$

**Hidden State**: $h_t = o_t \odot \tanh(c_t)$

### Key Implementation Details

```python
class LSTMModel(nn.Module):
    def __init__(self, config, total_loc_num):
        # Embedding (WITHOUT positional encoding)
        self.all_embed = AllEmbeddingLSTM(d_input, config, total_loc_num)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.lstm_dropout if config.lstm_num_layers > 1 else 0,
            batch_first=False
        )
        
        # Output projection
        self.fc = nn.Linear(config.lstm_hidden_size, total_loc_num)
        self.fc_dropout = nn.Dropout(config.fc_dropout)
    
    def forward(self, src, user, time, weekday, duration, lengths):
        # Get embeddings
        embedded = self.all_embed(src, user, time, weekday, duration)
        
        # Pack for variable length sequences
        packed = pack_padded_sequence(embedded, lengths, enforce_sorted=False)
        
        # LSTM forward pass
        output, (hidden, cell) = self.lstm(packed)
        
        # Take last layer's hidden state
        last_hidden = hidden[-1]  # [batch, hidden_size]
        
        # Output projection
        logits = self.fc(self.fc_dropout(last_hidden))
        return logits
```

---

## Architecture Comparison

### Feature Comparison Table

| Feature | Pointer V45 | MHSA | LSTM |
|---------|-------------|------|------|
| **Core Mechanism** | Transformer + Pointer | Transformer | LSTM |
| **Attention** | Self + Pointer | Self only | None |
| **Sequence Modeling** | Parallel | Parallel | Sequential |
| **Positional Encoding** | Sinusoidal + learned | Sinusoidal | Implicit |
| **Output Mechanism** | Pointer-Gen hybrid | Direct FC | Direct FC |
| **Normalization** | Pre-norm | Post-norm | LayerNorm |
| **Activation** | GELU | ReLU | tanh/sigmoid |
| **Copy Mechanism** | Yes | No | No |

### Theoretical Advantages

**Pointer V45**:
- Can copy from input (good for repeat locations)
- Position bias helps with recency patterns
- GELU activation for smoother gradients
- Pre-norm for training stability

**MHSA**:
- Standard well-tested architecture
- Good for capturing long-range dependencies
- Simpler than Pointer V45

**LSTM**:
- Strong sequential inductive bias
- Well-suited for temporal sequences
- No position encoding needed

---

## Parameter Count Analysis

### Geolife Dataset (1,187 locations, 46 users)

| Model | Parameters | Best Val Acc@1 |
|-------|------------|----------------|
| Pointer V45 | 251,476 - 813,252 | 49.25% |
| MHSA | 112,547 - 545,027 | 42.38% |
| LSTM | 467,683 - 599,779 | 40.58% |

### DIY Dataset (7,038 locations, 693 users)

| Model | Parameters | Best Val Acc@1 |
|-------|------------|----------------|
| Pointer V45 | 1,081,554 - 1,747,874 | 54.92% |
| MHSA | 797,982 - 835,230 | 53.69% |
| LSTM | 1,762,478 - 3,564,990 | 53.90% |

### Parameter Efficiency

$$\text{Efficiency} = \frac{\text{Accuracy}}{\text{Parameters (millions)}}$$

**Geolife**:
- Pointer V45: 49.25 / 0.443 = **111.2** acc/M params
- MHSA: 42.38 / 0.281 = **150.8** acc/M params
- LSTM: 40.58 / 0.468 = **86.7** acc/M params

**DIY**:
- Pointer V45: 54.92 / 1.082 = **50.8** acc/M params
- MHSA: 53.69 / 0.798 = **67.3** acc/M params
- LSTM: 53.90 / 3.565 = **15.1** acc/M params

While MHSA is more parameter-efficient on Geolife, Pointer V45 achieves the highest absolute accuracy on both datasets.

---

## Next: [06_RESULTS.md](06_RESULTS.md) - Comprehensive Results Analysis
