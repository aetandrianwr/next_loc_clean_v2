# Model Architecture Documentation

## 1. Architecture Overview

The Pointer Network V45 is a **Position-Aware Pointer Network** that combines:
1. **Rich Embedding Layer**: Location + User + Temporal features
2. **Transformer Encoder**: For sequence understanding
3. **Dual Output Heads**: Pointer mechanism + Generation head
4. **Adaptive Gate**: To blend pointer and generation distributions

### High-Level Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           POINTER NETWORK V45                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                        INPUT LAYER                                   │   │
│  │  • Location Sequence: [l₁, l₂, ..., lₙ]                             │   │
│  │  • User ID: u                                                        │   │
│  │  • Temporal: time, weekday, recency, duration                        │   │
│  │  • Sequence lengths                                                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                      EMBEDDING LAYER                                 │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────────────────────────────────┐ │   │
│  │  │  Loc     │ │  User    │ │          Temporal Embeddings         │ │   │
│  │  │  Emb     │ │  Emb     │ │  ┌────┐┌────┐┌────┐┌────┐┌────────┐ │ │   │
│  │  │ d_model  │ │ d_model  │ │  │Time││Week││Rec ││Dur ││PosEnd  │ │ │   │
│  │  │          │ │          │ │  │d/4 ││d/4 ││d/4 ││d/4 ││d/4     │ │ │   │
│  │  └──────────┘ └──────────┘ │  └────┘└────┘└────┘└────┘└────────┘ │ │   │
│  │       │            │       └──────────────────────────────────────┘ │   │
│  │       └────────────┴───────────────────┬──────────────────────────┘    │
│  │                                        │                                │
│  │                                        ▼                                │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  │  Input Projection: Linear(2*d + 5*d/4 → d) + LayerNorm          │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              POSITIONAL ENCODING (Sinusoidal)                        │   │
│  │                  hidden = hidden + PE[:seq_len]                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     TRANSFORMER ENCODER                              │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  Layer 1: PreNorm → MultiHeadAttn → PreNorm → FFN (GELU)   │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                              │                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  Layer 2: PreNorm → MultiHeadAttn → PreNorm → FFN (GELU)   │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  │                              │                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │  Layer N: PreNorm → MultiHeadAttn → PreNorm → FFN (GELU)   │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │           CONTEXT EXTRACTION (Last Valid Position)                   │   │
│  │                  context = encoded[batch_idx, last_idx]              │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│              ┌────────────────────┬┴───────────────────┐                   │
│              │                    │                     │                   │
│              ▼                    ▼                     ▼                   │
│  ┌───────────────────┐  ┌───────────────────┐  ┌───────────────────┐      │
│  │  POINTER HEAD     │  │  GENERATION HEAD  │  │  GATE             │      │
│  │                   │  │                   │  │                   │      │
│  │  Q = Linear(ctx)  │  │  Linear(d→V)      │  │  MLP(d→d/2→1)    │      │
│  │  K = Linear(enc)  │  │  Softmax          │  │  Sigmoid          │      │
│  │  scores = QK^T/√d │  │                   │  │                   │      │
│  │  + position_bias  │  │                   │  │                   │      │
│  │  Softmax+Scatter  │  │                   │  │                   │      │
│  └───────────────────┘  └───────────────────┘  └───────────────────┘      │
│              │                    │                     │                   │
│              └────────────────────┴─────────────────────┘                   │
│                                    │                                        │
│                                    ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    FINAL PROBABILITY                                 │   │
│  │        P_final = gate × P_pointer + (1-gate) × P_generation          │   │
│  │                    output = log(P_final + ε)                         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│                         [batch_size, num_locations]                         │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Input/Output Specification

### 2.1 Inputs

| Input | Shape | Type | Description |
|-------|-------|------|-------------|
| `x` | `[seq_len, batch_size]` | LongTensor | Location IDs (0-padded) |
| `x_dict['user']` | `[batch_size]` | LongTensor | User IDs |
| `x_dict['time']` | `[seq_len, batch_size]` | LongTensor | Time of day (0-96) |
| `x_dict['weekday']` | `[seq_len, batch_size]` | LongTensor | Day of week (0-7) |
| `x_dict['diff']` | `[seq_len, batch_size]` | LongTensor | Days ago (0-8) |
| `x_dict['duration']` | `[seq_len, batch_size]` | LongTensor | Duration bucket (0-99) |
| `x_dict['len']` | `[batch_size]` | LongTensor | Sequence lengths |

### 2.2 Outputs

| Output | Shape | Type | Description |
|--------|-------|------|-------------|
| `log_probs` | `[batch_size, num_locations]` | FloatTensor | Log probabilities over locations |

### 2.3 Example Shapes

For a batch of 32 sequences with max length 50 and 1000 locations:

```python
x.shape = [50, 32]           # 50 time steps, 32 samples
x_dict['user'].shape = [32]  # One user per sample
x_dict['time'].shape = [50, 32]
x_dict['weekday'].shape = [50, 32]
x_dict['diff'].shape = [50, 32]
x_dict['duration'].shape = [50, 32]
x_dict['len'].shape = [32]

output.shape = [32, 1000]    # 32 samples, 1000 locations
```

---

## 3. Layer-by-Layer Breakdown

### 3.1 Embedding Layers

```python
# Core embeddings
self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)

# Temporal embeddings (each d_model // 4)
self.time_emb = nn.Embedding(97, d_model // 4)      # 96 intervals + 1 padding
self.weekday_emb = nn.Embedding(8, d_model // 4)    # 7 days + 1 padding
self.recency_emb = nn.Embedding(9, d_model // 4)    # 8 recency levels + 1 padding
self.duration_emb = nn.Embedding(100, d_model // 4) # 100 duration buckets

# Position from end embedding
self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
```

**Embedding Dimension Breakdown**:
- Location: `d_model` = 128 (for DIY)
- User: `d_model` = 128
- Time: `d_model // 4` = 32
- Weekday: `d_model // 4` = 32
- Recency: `d_model // 4` = 32
- Duration: `d_model // 4` = 32
- Position-from-end: `d_model // 4` = 32

**Total Input**: `2 * d_model + 5 * (d_model // 4)` = `2 * 128 + 5 * 32` = 416

### 3.2 Input Projection

```python
input_dim = d_model * 2 + d_model // 4 * 5  # 416 for d_model=128
self.input_proj = nn.Linear(input_dim, d_model)  # 416 → 128
self.input_norm = nn.LayerNorm(d_model)
```

This layer:
1. Projects concatenated embeddings to model dimension
2. Normalizes for stable training

### 3.3 Positional Encoding

```python
def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]
```

**Sinusoidal Formula**:
- Even dimensions: `sin(pos / 10000^(2i/d_model))`
- Odd dimensions: `cos(pos / 10000^(2i/d_model))`

### 3.4 Transformer Encoder

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-norm architecture
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

**Key Settings**:
- **Pre-norm**: LayerNorm applied before attention/FFN (more stable)
- **GELU activation**: Smooth activation function
- **Batch-first**: Input shape is `[batch, seq, feature]`

### 3.5 Pointer Mechanism

```python
# Query and Key projections
self.pointer_query = nn.Linear(d_model, d_model)
self.pointer_key = nn.Linear(d_model, d_model)

# Learnable position bias
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
```

**Pointer Computation**:
```python
# Query from context (last position)
query = self.pointer_query(context).unsqueeze(1)  # [B, 1, d]

# Keys from all encoded positions
keys = self.pointer_key(encoded)  # [B, seq, d]

# Attention scores
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
# [B, seq]

# Add position bias
ptr_scores = ptr_scores + self.position_bias[pos_from_end]

# Mask padding positions
ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))

# Softmax to get pointer probabilities
ptr_probs = F.softmax(ptr_scores, dim=-1)  # [B, seq]

# Scatter to location vocabulary
ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
ptr_dist.scatter_add_(1, x, ptr_probs)  # [B, V]
```

### 3.6 Generation Head

```python
self.gen_head = nn.Linear(d_model, num_locations)
```

**Generation Computation**:
```python
gen_probs = F.softmax(self.gen_head(context), dim=-1)  # [B, V]
```

### 3.7 Pointer-Generation Gate

```python
self.ptr_gen_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Sigmoid()
)
```

**Gate Computation**:
```python
gate = self.ptr_gen_gate(context)  # [B, 1]
final_probs = gate * ptr_dist + (1 - gate) * gen_probs  # [B, V]
```

---

## 4. Dimension Flow

Here's how dimensions change through the network:

```
Input: x [seq_len=50, batch=32]

After Embeddings:
  loc_emb:     [32, 50, 128]  (batch first after transpose)
  user_emb:    [32, 50, 128]  (expanded)
  time_emb:    [32, 50, 32]
  weekday_emb: [32, 50, 32]
  recency_emb: [32, 50, 32]
  duration_emb:[32, 50, 32]
  pos_emb:     [32, 50, 32]

After Concat:    [32, 50, 416]

After Proj+Norm: [32, 50, 128]

After Pos Enc:   [32, 50, 128]

After Transformer: [32, 50, 128]

Context (last pos): [32, 128]

Pointer Probs:   [32, 50] → scatter → [32, V]
Gen Probs:       [32, V]
Gate:            [32, 1]

Final Output:    [32, V]  (log probabilities)
```

---

## 5. Parameter Count

### 5.1 GeoLife Configuration (d_model=64)

| Component | Parameters |
|-----------|------------|
| Location Embedding | 1,187 × 64 = 75,968 |
| User Embedding | 46 × 64 = 2,944 |
| Time Embedding | 97 × 16 = 1,552 |
| Weekday Embedding | 8 × 16 = 128 |
| Recency Embedding | 9 × 16 = 144 |
| Duration Embedding | 100 × 16 = 1,600 |
| Pos-from-End Embedding | 160 × 16 = 2,560 |
| Input Projection | 208 × 64 + 64 = 13,376 |
| Input LayerNorm | 64 × 2 = 128 |
| Transformer (2 layers) | ~100K |
| Pointer Query/Key | 64 × 64 × 2 = 8,192 |
| Position Bias | 150 |
| Generation Head | 64 × 1,187 + 1,187 = 77,155 |
| Gate MLP | 64 × 32 + 32 + 32 × 1 + 1 = 2,113 |
| **Total** | **~180K** |

### 5.2 DIY Configuration (d_model=128)

| Component | Parameters |
|-----------|------------|
| Location Embedding | 6,866 × 128 = 878,848 |
| User Embedding | 121 × 128 = 15,488 |
| Time Embedding | 97 × 32 = 3,104 |
| Weekday Embedding | 8 × 32 = 256 |
| Recency Embedding | 9 × 32 = 288 |
| Duration Embedding | 100 × 32 = 3,200 |
| Pos-from-End Embedding | 160 × 32 = 5,120 |
| Input Projection | 416 × 128 + 128 = 53,376 |
| Input LayerNorm | 128 × 2 = 256 |
| Transformer (3 layers) | ~400K |
| Pointer Query/Key | 128 × 128 × 2 = 32,768 |
| Position Bias | 150 |
| Generation Head | 128 × 6,866 + 6,866 = 885,714 |
| Gate MLP | 128 × 64 + 64 + 64 × 1 + 1 = 8,321 |
| **Total** | **~2.3M** |

---

## 6. Configuration Options

### 6.1 Model Configuration (YAML)

```yaml
model:
  d_model: 128        # Model dimension
  nhead: 4            # Number of attention heads
  num_layers: 3       # Transformer encoder layers
  dim_feedforward: 256 # FFN hidden dimension
  dropout: 0.15       # Dropout probability
```

### 6.2 Configuration Effects

| Parameter | Effect of Increasing |
|-----------|---------------------|
| `d_model` | More expressive embeddings, more parameters |
| `nhead` | More attention patterns, must divide d_model |
| `num_layers` | Deeper model, more parameters, longer training |
| `dim_feedforward` | More expressive FFN, more parameters |
| `dropout` | More regularization, may hurt if too high |

---

## 7. Key Design Decisions

### 7.1 Why Pre-Norm?

```python
norm_first=True  # Pre-norm architecture
```

**Pre-norm** (normalize before sub-layer) vs **Post-norm** (normalize after):
- More stable gradients during training
- Can use larger learning rates
- Better performance in practice

### 7.2 Why GELU?

```python
activation='gelu'
```

**GELU** (Gaussian Error Linear Unit) vs ReLU:
- Smoother activation function
- Better gradient flow
- Standard in modern Transformers (BERT, GPT)

### 7.3 Why Position-from-End?

```python
pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
```

Instead of position-from-start:
- Recent positions are most predictive
- Consistent meaning regardless of sequence length
- Position 0 always means "most recent"

### 7.4 Why Scatter for Pointer?

```python
ptr_dist.scatter_add_(1, x, ptr_probs)
```

The pointer attention gives probabilities over **positions**, but we need probabilities over **locations**:
- Multiple positions might have the same location
- `scatter_add_` accumulates probabilities for repeated locations

---

## 8. Comparison with Related Architectures

### 8.1 vs. Standard Transformer

| Feature | Standard Transformer | Pointer Network V45 |
|---------|---------------------|---------------------|
| Output | Vocabulary softmax | Pointer + Generation |
| Position encoding | Sinusoidal only | Sinusoidal + Pos-from-end |
| Temporal features | Not included | Rich temporal embeddings |
| User modeling | Not included | User embedding |

### 8.2 vs. Standard Pointer Network

| Feature | Standard Pointer Net | Pointer Network V45 |
|---------|---------------------|---------------------|
| Encoder | LSTM | Transformer |
| Output | Pointer only | Pointer + Generation |
| Gate | None | Adaptive gate |
| Position modeling | Implicit | Explicit bias + embedding |

### 8.3 vs. DeepMove

| Feature | DeepMove | Pointer Network V45 |
|---------|----------|---------------------|
| Architecture | Attention RNN | Transformer |
| Copy mechanism | Not explicit | Explicit pointer |
| Temporal modeling | Embedded | Rich embeddings |
| Gate | None | Adaptive gate |

---

*Next: [04_COMPONENTS_DEEP_DIVE.md](04_COMPONENTS_DEEP_DIVE.md) - Detailed Component Analysis*
