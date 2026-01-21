# 3. Model Architecture

## Deep Dive into PointerGeneratorTransformer

---

## 3.1 Overview

PointerGeneratorTransformer is a hybrid neural network architecture designed for next location prediction. It combines:

1. **Transformer Encoder**: For sequence modeling
2. **Pointer Mechanism**: For copying from historical locations
3. **Generation Head**: For predicting any location in the vocabulary
4. **Adaptive Gate**: For dynamically blending the two strategies

### High-Level Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                           PointerGeneratorTransformer                                   │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                         INPUT LAYER                                      │ │
│  │  Location Sequence: [loc₁, loc₂, loc₃, ..., locₙ]                        │ │
│  │  User ID: user_id                                                        │ │
│  │  Temporal: time_of_day, weekday, duration, recency                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                       EMBEDDING LAYER                                    │ │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │ │
│  │  │ Location │ │   User   │ │   Time   │ │ Weekday  │ │ Duration │       │ │
│  │  │   Emb    │ │   Emb    │ │   Emb    │ │   Emb    │ │   Emb    │ ...   │ │
│  │  │ (d_model)│ │(d_model) │ │(d_model/4)│(d_model/4)│ │(d_model/4)│      │ │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘ └──────────┘       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                      PROJECTION LAYER                                    │ │
│  │  Concatenate all embeddings → Linear(input_dim, d_model) → LayerNorm    │ │
│  │  Add sinusoidal positional encoding                                      │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                    TRANSFORMER ENCODER                                   │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐    │ │
│  │  │  TransformerEncoderLayer × num_layers                           │    │ │
│  │  │  • Multi-Head Self-Attention (nhead)                            │    │ │
│  │  │  • Feed-Forward Network (dim_feedforward)                       │    │ │
│  │  │  • Pre-Norm + GELU + Dropout                                    │    │ │
│  │  └─────────────────────────────────────────────────────────────────┘    │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                     ┌──────────────┴──────────────┐                          │
│                     │                             │                          │
│                     ▼                             ▼                          │
│  ┌────────────────────────────┐  ┌────────────────────────────┐              │
│  │    POINTER MECHANISM       │  │    GENERATION HEAD         │              │
│  │  • Query = Linear(context) │  │  • Linear(context, vocab)  │              │
│  │  • Key = Linear(encoded)   │  │  • Softmax                 │              │
│  │  • Attention + Pos Bias    │  │                            │              │
│  │  • Scatter to vocabulary   │  │                            │              │
│  └────────────────────────────┘  └────────────────────────────┘              │
│                     │                             │                          │
│                     └──────────────┬──────────────┘                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                        ADAPTIVE GATE                                     │ │
│  │  gate = σ(Linear(Linear(context)))                                       │ │
│  │  P(y) = gate × P_pointer + (1 - gate) × P_generation                     │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                    │                                          │
│                                    ▼                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │                          OUTPUT                                          │ │
│  │  Log probabilities: [batch_size, num_locations]                          │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 3.2 Component Details

### 3.2.1 Location Embedding

**Purpose**: Convert discrete location IDs into dense vector representations.

```python
self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
```

**Details**:
- Input: Location ID (integer)
- Output: Dense vector of dimension `d_model`
- `padding_idx=0` means location 0 is reserved for padding and always outputs zeros

**Example**:
```
Location ID: 42
                  ↓
            ┌─────┴─────┐
            │ Embedding │
            │  Matrix   │
            │ (L × d)   │
            └─────┬─────┘
                  ↓
         [0.23, -0.15, 0.87, ..., 0.12]  (d_model dimensions)
```

---

### 3.2.2 User Embedding

**Purpose**: Capture user-specific mobility patterns through learned representations.

```python
self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
```

**Why It Matters**:
- Different users have different mobility habits
- Some prefer public transit, others drive
- Work/home locations vary by user

**Usage in Model**:
```python
# User embedding is expanded to match sequence length
user_emb = self.user_emb(x_dict['user'])  # [batch, d_model]
user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, d_model]
```

---

### 3.2.3 Temporal Embeddings

**Purpose**: Encode time-related features that influence mobility patterns.

```python
# Time of day (96 15-minute intervals)
self.time_emb = nn.Embedding(97, d_model // 4)

# Day of week (7 days)
self.weekday_emb = nn.Embedding(8, d_model // 4)

# Recency (days since visit)
self.recency_emb = nn.Embedding(9, d_model // 4)

# Visit duration (in 30-min buckets)
self.duration_emb = nn.Embedding(100, d_model // 4)
```

**Time Discretization**:
```
24 hours × 4 (15-min intervals) = 96 time slots

00:00-00:15 → 0
00:15-00:30 → 1
...
23:45-00:00 → 95
```

**Why Each Temporal Feature Matters**:

| Feature | Captures | Example |
|---------|----------|---------|
| Time | Daily patterns | Work at 9am, lunch at 12pm |
| Weekday | Weekly patterns | Gym on Tuesday, church on Sunday |
| Recency | Visit freshness | Recently visited → more likely |
| Duration | Activity type | Quick stop vs. long stay |

---

### 3.2.4 Position-from-End Embedding

**Purpose**: Encode how far each location is from the current position in the sequence.

```python
self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
```

**Intuition**:
- More recent visits are more relevant
- Position-from-end captures "how many visits ago"

**Calculation**:
```python
# If sequence length is 5 and we're predicting position 6:
# Position:     [1, 2, 3, 4, 5]
# Pos-from-end: [5, 4, 3, 2, 1]  # Last visit is 1 step away

positions = torch.arange(seq_len, device=device)
pos_from_end = lengths.unsqueeze(1) - positions
pos_emb = self.pos_from_end_emb(pos_from_end)
```

**Visual**:
```
Sequence: Home → Work → Lunch → Work → Home → ?
Pos-from-end:  5     4      3      2     1    (predicting here)

More recent = smaller pos_from_end = more relevant
```

---

### 3.2.5 Input Projection

**Purpose**: Combine all embeddings and project to model dimension.

```python
# Calculate input dimension
input_dim = d_model  # Location
input_dim += d_model  # User (if used)
input_dim += d_model // 4 * 4  # 4 temporal features
input_dim += d_model // 4  # Position-from-end

# Projection layers
self.input_proj = nn.Linear(input_dim, d_model)
self.input_norm = nn.LayerNorm(d_model)
```

**Process**:
```
[loc_emb | user_emb | time_emb | weekday_emb | recency_emb | duration_emb | pos_from_end_emb]
                                            ↓
                                    Concatenation
                                            ↓
                                   Linear(input_dim → d_model)
                                            ↓
                                       LayerNorm
                                            ↓
                              Add Sinusoidal Position Encoding
```

---

### 3.2.6 Sinusoidal Positional Encoding

**Purpose**: Inject absolute position information into the sequence.

```python
def _create_pos_encoding(self, max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)  # Even indices
    pe[:, 1::2] = torch.cos(position * div_term)  # Odd indices
    return pe.unsqueeze(0)
```

**Formula**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why Sinusoidal**:
- Fixed (not learned) → generalizes to unseen sequence lengths
- Unique encoding for each position
- Allows model to learn relative positions

---

### 3.2.7 Transformer Encoder

**Purpose**: Model interactions between all positions in the sequence.

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='gelu',      # GELU activation (smoother than ReLU)
    batch_first=True,
    norm_first=True         # Pre-norm (more stable training)
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

**Pre-Norm Architecture**:
```
┌─────────────────────────────────────────────┐
│         TransformerEncoderLayer             │
│                                             │
│    Input                                    │
│      ↓                                      │
│   LayerNorm  ←─────────────┐                │
│      ↓                     │                │
│   Multi-Head Attention     │ (Residual)     │
│      ↓                     │                │
│   Add ──────────────────────┘               │
│      ↓                                      │
│   LayerNorm  ←─────────────┐                │
│      ↓                     │                │
│   Feed-Forward Network     │ (Residual)     │
│      ↓                     │                │
│   Add ──────────────────────┘               │
│      ↓                                      │
│    Output                                   │
└─────────────────────────────────────────────┘
```

**Attention Masking**:
```python
# Create padding mask
mask = positions >= lengths.unsqueeze(1)  # True for padded positions

# Apply in transformer (padded positions don't attend or get attended to)
encoded = self.transformer(hidden, src_key_padding_mask=mask)
```

---

### 3.2.8 Pointer Mechanism

**Purpose**: Select from locations in the input sequence (copy mechanism).

```python
# Learned projections
self.pointer_query = nn.Linear(d_model, d_model)
self.pointer_key = nn.Linear(d_model, d_model)

# Learnable position bias
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
```

**How It Works**:

```python
def _compute_pointer_dist(self, x, encoded, context, mask, pos_from_end):
    # 1. Project context to query
    query = self.pointer_query(context).unsqueeze(1)  # [batch, 1, d_model]
    
    # 2. Project encoded sequence to keys
    keys = self.pointer_key(encoded)  # [batch, seq_len, d_model]
    
    # 3. Compute attention scores
    ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # [batch, seq_len]
    ptr_scores = ptr_scores / math.sqrt(self.d_model)  # Scale
    
    # 4. Add position bias (learned preference for recent positions)
    ptr_scores = ptr_scores + self.position_bias[pos_from_end]
    
    # 5. Mask padded positions
    ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
    
    # 6. Softmax to get attention weights
    ptr_probs = F.softmax(ptr_scores, dim=-1)  # [batch, seq_len]
    
    # 7. Scatter probabilities to vocabulary
    ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
    ptr_dist.scatter_add_(1, x, ptr_probs)
    
    return ptr_dist
```

**Visual Explanation**:
```
Input sequence:  [Home, Work, Cafe, Work, Home]
                   ↓     ↓     ↓     ↓     ↓
Attention weights: 0.1   0.15  0.05  0.2   0.5
                   ↓     ↓     ↓     ↓     ↓
                   └──────┴──────┴──────┴──────┘
                              ↓
                   Scatter to vocabulary
                              ↓
P(Home) = 0.1 + 0.5 = 0.6  (appears twice, probabilities add)
P(Work) = 0.15 + 0.2 = 0.35
P(Cafe) = 0.05
P(other) = 0
```

**Position Bias**:
- Learnable parameter that adds preference for certain positions
- Typically learns to prefer more recent positions (lower pos_from_end)

---

### 3.2.9 Generation Head

**Purpose**: Predict any location from the full vocabulary.

```python
self.gen_head = nn.Linear(d_model, num_locations)
```

**How It Works**:
```python
# Simple linear projection + softmax
gen_logits = self.gen_head(context)  # [batch, num_locations]
gen_probs = F.softmax(gen_logits, dim=-1)
```

**When Generation Helps**:
- Predicting locations never visited before
- Handling rare locations not in recent history
- Providing a "fallback" distribution

---

### 3.2.10 Adaptive Gate (Pointer-Generator)

**Purpose**: Learn when to copy vs. generate.

```python
self.ptr_gen_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Sigmoid()
)
```

**How It Works**:
```python
# Compute gate value (0 to 1)
gate = self.ptr_gen_gate(context)  # [batch, 1]

# Blend distributions
final_probs = gate * ptr_dist + (1 - gate) * gen_probs
```

**Interpretation**:
- `gate ≈ 1`: Trust the pointer (copy from history)
- `gate ≈ 0`: Trust the generator (predict from vocabulary)
- Model learns when each strategy is appropriate

**Visual**:
```
                    context
                       ↓
              ┌───────┴───────┐
              ↓               ↓
        Pointer Dist    Generation Dist
         [0, 0.6, 0,      [0.1, 0.2, 0.1,
          0.35, 0, 0.05]   0.15, 0.2, 0.25]
              ↓               ↓
              └───────┬───────┘
                      ↓
               Adaptive Gate
                 g = 0.8
                      ↓
    Final = 0.8 × Pointer + 0.2 × Generation
         = [0.02, 0.52, 0.02, 0.31, 0.04, 0.09]
```

---

## 3.3 Complete Forward Pass

```python
def forward(self, x: torch.Tensor, x_dict: dict) -> torch.Tensor:
    # 1. Reshape input
    x = x.T  # [seq_len, batch] → [batch, seq_len]
    batch_size, seq_len = x.shape
    
    # 2. Get all embeddings
    loc_emb = self.loc_emb(x)
    user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
    temporal = concat([time_emb, weekday_emb, recency_emb, duration_emb])
    pos_emb = self.pos_from_end_emb(pos_from_end)
    
    # 3. Combine and project
    combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
    hidden = self.input_norm(self.input_proj(combined))
    hidden = hidden + self.pos_encoding[:, :seq_len, :]
    
    # 4. Transformer encoding
    mask = positions >= lengths.unsqueeze(1)
    encoded = self.transformer(hidden, src_key_padding_mask=mask)
    
    # 5. Extract context (last valid position)
    context = encoded[batch_idx, last_idx]
    
    # 6. Compute pointer distribution
    ptr_dist = self._compute_pointer_dist(x, encoded, context, mask, pos_from_end)
    
    # 7. Compute generation distribution
    gen_probs = F.softmax(self.gen_head(context), dim=-1)
    
    # 8. Blend with adaptive gate
    gate = self.ptr_gen_gate(context)
    final_probs = gate * ptr_dist + (1 - gate) * gen_probs
    
    # 9. Return log probabilities
    return torch.log(final_probs + 1e-10)
```

---

## 3.4 Parameter Count

### GeoLife Configuration (d_model=96, num_layers=2)

| Component | Parameters | Percentage |
|-----------|------------|------------|
| Location Embedding | num_locations × 96 | ~30% |
| User Embedding | num_users × 96 | ~5% |
| Temporal Embeddings | ~2,000 | ~1% |
| Input Projection | ~20,000 | ~10% |
| Transformer Encoder | ~75,000 | ~40% |
| Pointer Mechanism | ~20,000 | ~10% |
| Generation Head | 96 × num_locations | ~5% |
| Adaptive Gate | ~2,400 | ~1% |

### Model Size Comparison

```
Full Model:              ~200,000 parameters
Single Layer Ablation:   ~150,000 parameters
No Pointer Ablation:     ~180,000 parameters
No Generation Ablation:  ~190,000 parameters
```

---

## 3.5 Why This Architecture Works

### Key Insight: Human Mobility is Repetitive

Studies show that:
- 80% of visits are to previously visited locations
- People have ~25-30 regularly visited places
- Weekly patterns are very predictable

### Architecture Matches the Problem

| Problem Characteristic | Architecture Solution |
|------------------------|----------------------|
| Repetitive visits | Pointer mechanism |
| Novel locations | Generation head |
| Time patterns | Temporal embeddings |
| Personal habits | User embeddings |
| Sequence dependencies | Transformer encoder |
| Recency matters | Position-from-end + bias |

---

*Next: [04_methodology.md](04_methodology.md) - Scientific methodology explanation*
