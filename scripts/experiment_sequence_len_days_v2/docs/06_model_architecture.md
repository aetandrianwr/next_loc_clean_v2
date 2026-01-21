# 06. Model Architecture

## PointerGeneratorTransformer - Deep Dive

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Technical Architecture Documentation |
| **Audience** | ML Engineers, Researchers |
| **Reading Time** | 18-20 minutes |
| **Prerequisites** | Transformer architecture, attention mechanisms |

---

## 1. Architecture Overview

### 1.1 High-Level Design

PointerGeneratorTransformer is a **Transformer-based Pointer-Generator Network** designed for next location prediction. It combines three key mechanisms:

1. **Pointer Mechanism**: Attends to input sequence locations and "points" to likely candidates
2. **Generation Head**: Predicts over the full location vocabulary
3. **Adaptive Gate**: Learns to blend pointer and generation distributions

```
┌─────────────────────────────────────────────────────────────────────┐
│                       PointerGeneratorTransformer                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Sequence                                                      │
│  [l₁, l₂, ..., lₙ]                                                  │
│        │                                                             │
│        ▼                                                             │
│  ┌─────────────────┐                                                │
│  │   Embeddings    │  Location + User + Temporal + Position         │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │   Input Proj    │  Projection + LayerNorm + Positional Encoding  │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ▼                                                          │
│  ┌─────────────────┐                                                │
│  │  Transformer    │  Pre-norm Encoder (N layers)                   │
│  │    Encoder      │                                                │
│  └────────┬────────┘                                                │
│           │                                                          │
│           ├──────────────────────┬──────────────────────┐           │
│           │                      │                      │           │
│           ▼                      ▼                      ▼           │
│  ┌───────────────┐      ┌───────────────┐      ┌───────────────┐   │
│  │    Pointer    │      │  Generation   │      │   Ptr-Gen     │   │
│  │   Mechanism   │      │     Head      │      │     Gate      │   │
│  └───────┬───────┘      └───────┬───────┘      └───────┬───────┘   │
│          │                      │                      │           │
│          │      P_ptr           │      P_gen           │    g      │
│          │                      │                      │           │
│          └──────────────────────┼──────────────────────┘           │
│                                 │                                    │
│                                 ▼                                    │
│                    P_final = g · P_ptr + (1-g) · P_gen              │
│                                 │                                    │
│                                 ▼                                    │
│                          log(P_final)                               │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Encoder architecture | Transformer | Parallel processing, long-range dependencies |
| Normalization | Pre-norm | More stable training |
| Activation | GELU | Smoother gradients than ReLU |
| Prediction strategy | Pointer + Generator | Copy from history OR generate new |
| Position encoding | Sinusoidal + from-end | Absolute and relative position awareness |

### 1.3 Model Configurations Used

**DIY Dataset Configuration**:
```yaml
d_model: 64
nhead: 4
num_layers: 2
dim_feedforward: 256
dropout: 0.2
```

**GeoLife Dataset Configuration**:
```yaml
d_model: 96
nhead: 2
num_layers: 2
dim_feedforward: 192
dropout: 0.25
```

---

## 2. Embedding Layer

### 2.1 Location Embedding

```python
self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
```

**Purpose**: Convert discrete location IDs to dense vectors.

**Details**:
- Input: Location ID ∈ {0, 1, ..., num_locations-1}
- Output: Vector ∈ ℝ^d_model
- Padding: Index 0 reserved for padding (outputs zeros)

**Example** (d_model=64):
```
Location ID 42 → [0.12, -0.45, 0.78, ..., 0.33]  (64 dimensions)
```

### 2.2 User Embedding

```python
self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
```

**Purpose**: Capture user-specific patterns.

**Details**:
- Each user has a unique embedding vector
- Broadcast across sequence length (same user for all positions)

### 2.3 Temporal Embeddings

Four separate embeddings capture different temporal aspects:

```python
self.time_emb = nn.Embedding(97, d_model // 4)      # Time of day
self.weekday_emb = nn.Embedding(8, d_model // 4)    # Day of week
self.recency_emb = nn.Embedding(9, d_model // 4)    # Days ago
self.duration_emb = nn.Embedding(100, d_model // 4) # Visit duration
```

#### 2.3.1 Time of Day Embedding

| Index | Time Range | Interval |
|-------|------------|----------|
| 0 | Padding | - |
| 1 | 00:00 - 00:15 | 15 min |
| 2 | 00:15 - 00:30 | 15 min |
| ... | ... | ... |
| 96 | 23:45 - 24:00 | 15 min |

**Dimension**: d_model // 4 = 16 (for d_model=64)

#### 2.3.2 Day of Week Embedding

| Index | Day |
|-------|-----|
| 0 | Padding |
| 1 | Monday |
| 2 | Tuesday |
| ... | ... |
| 7 | Sunday |

#### 2.3.3 Recency Embedding

| Index | Meaning |
|-------|---------|
| 0 | Padding |
| 1 | Today (diff=0) |
| 2 | Yesterday (diff=1) |
| ... | ... |
| 8 | 7+ days ago |

**Why Recency Matters**: Recent visits are more predictive than old ones.

#### 2.3.4 Duration Embedding

| Index | Duration |
|-------|----------|
| 0 | 0-30 minutes |
| 1 | 30-60 minutes |
| ... | ... |
| 99 | 49.5+ hours |

**Why Duration Matters**: Distinguishes quick stops from long stays.

### 2.4 Position-from-End Embedding

```python
self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
```

**Purpose**: Encode how far each position is from the end of the sequence.

**Calculation**:
```python
positions = torch.arange(seq_len)  # [0, 1, 2, ..., seq_len-1]
pos_from_end = lengths.unsqueeze(1) - positions  # Distance from end
pos_from_end = pos_from_end.clamp(0, max_seq_len - 1)
```

**Example** (sequence length = 5):
```
Position:      [0,  1,  2,  3,  4]
Pos-from-end:  [5,  4,  3,  2,  1]  (most recent = 1)
```

**Why This Matters**: 
- The pointer mechanism needs to know which locations are recent
- Last position (pos-from-end = 1) is most important for prediction

### 2.5 Embedding Combination

All embeddings are concatenated and projected:

```python
# Dimensions (for d_model=64):
# loc_emb:      [B, L, 64]
# user_emb:     [B, L, 64]  (broadcast)
# time_emb:     [B, L, 16]
# weekday_emb:  [B, L, 16]
# recency_emb:  [B, L, 16]
# duration_emb: [B, L, 16]
# pos_emb:      [B, L, 16]

# Total input dimension: 64 + 64 + 16*5 = 208

combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
# Shape: [B, L, 208]

hidden = self.input_proj(combined)  # Linear(208, 64)
# Shape: [B, L, 64]

hidden = self.input_norm(hidden)    # LayerNorm
```

---

## 3. Positional Encoding

### 3.1 Sinusoidal Positional Encoding

The model uses standard sinusoidal positional encoding from "Attention is All You Need":

```python
def _create_pos_encoding(self, max_len, d_model):
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]
```

### 3.2 Mathematical Formula

For position $pos$ and dimension $i$:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

### 3.3 Visualization

```
Position 0:  [sin(0/1), cos(0/1), sin(0/10), cos(0/10), ...]
Position 1:  [sin(1/1), cos(1/1), sin(1/10), cos(1/10), ...]
Position 2:  [sin(2/1), cos(2/1), sin(2/10), cos(2/10), ...]
...

Low-frequency dimensions (large i) → capture long-range positions
High-frequency dimensions (small i) → capture local positions
```

### 3.4 Addition to Hidden States

```python
hidden = hidden + self.pos_encoding[:, :seq_len, :]
```

This adds positional information to each token's representation.

---

## 4. Transformer Encoder

### 4.1 Architecture Configuration

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-normalization
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

### 4.2 Pre-Normalization vs Post-Normalization

**Pre-norm (used here)**:
```
x → LayerNorm → Attention → + → LayerNorm → FFN → +
        └──────────────────┘         └─────────┘
```

**Post-norm (original Transformer)**:
```
x → Attention → + → LayerNorm → FFN → + → LayerNorm
        └──────────┘        └────────┘
```

**Why Pre-norm**: More stable training, especially for deeper models.

### 4.3 Single Encoder Layer Structure

```
Input: x [B, L, d_model]
    │
    ▼
LayerNorm
    │
    ▼
Multi-Head Self-Attention
    │
    + ← Residual Connection
    │
    ▼
LayerNorm
    │
    ▼
Feed-Forward Network
    │
    + ← Residual Connection
    │
    ▼
Output: [B, L, d_model]
```

### 4.4 Multi-Head Self-Attention

For DIY (d_model=64, nhead=4):
- Head dimension: 64 / 4 = 16
- Each head learns different attention patterns

```python
# Conceptually:
Q = x @ W_Q  # [B, L, 64]
K = x @ W_K  # [B, L, 64]
V = x @ W_V  # [B, L, 64]

# Split into heads
Q = Q.view(B, L, 4, 16).transpose(1, 2)  # [B, 4, L, 16]
K = K.view(B, L, 4, 16).transpose(1, 2)  # [B, 4, L, 16]
V = V.view(B, L, 4, 16).transpose(1, 2)  # [B, 4, L, 16]

# Attention per head
attn_scores = Q @ K.transpose(-2, -1) / sqrt(16)  # [B, 4, L, L]
attn_probs = softmax(attn_scores, dim=-1)
head_output = attn_probs @ V  # [B, 4, L, 16]

# Concatenate heads
output = head_output.transpose(1, 2).contiguous().view(B, L, 64)
output = output @ W_O  # Final projection
```

### 4.5 Feed-Forward Network

```python
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
```

Where:
- W1: [d_model, dim_feedforward] = [64, 256]
- W2: [dim_feedforward, d_model] = [256, 64]

**GELU Activation**:
$$\text{GELU}(x) = x \cdot \Phi(x)$$

where $\Phi(x)$ is the standard normal CDF. Approximation:
$$\text{GELU}(x) \approx 0.5x(1 + \tanh[\sqrt{2/\pi}(x + 0.044715x^3)])$$

### 4.6 Padding Mask

```python
mask = positions >= lengths.unsqueeze(1)  # [B, L]
encoded = self.transformer(hidden, src_key_padding_mask=mask)
```

**Purpose**: Prevent attention to padding positions.

**Example** (lengths = [3, 5], max_len = 5):
```
Sample 1 (length=3): [False, False, False, True, True]   # Attend to first 3
Sample 2 (length=5): [False, False, False, False, False] # Attend to all 5
```

---

## 5. Pointer Mechanism

### 5.1 Concept

The pointer mechanism computes attention over input locations and creates a distribution for "copying" from the input.

**Intuition**: "Which of the locations I've visited am I most likely to go to next?"

### 5.2 Implementation

```python
# Extract context from last valid position
batch_idx = torch.arange(batch_size, device=device)
last_idx = (lengths - 1).clamp(min=0)
context = encoded[batch_idx, last_idx]  # [B, d_model]

# Compute pointer attention
query = self.pointer_query(context).unsqueeze(1)  # [B, 1, d_model]
keys = self.pointer_key(encoded)                   # [B, L, d_model]

# Attention scores
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # [B, L]
ptr_scores = ptr_scores / math.sqrt(self.d_model)  # Scale

# Add position bias
ptr_scores = ptr_scores + self.position_bias[pos_from_end]

# Mask padding
ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))

# Softmax over input positions
ptr_probs = F.softmax(ptr_scores, dim=-1)  # [B, L]
```

### 5.3 Position Bias

```python
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
```

**Purpose**: Learnable bias that adjusts pointer probability based on position from end.

**Expected Pattern** (after training):
- Higher bias for recent positions (pos_from_end = 1, 2, 3)
- Lower bias for older positions

### 5.4 Scatter to Vocabulary

```python
# Input locations: x [B, L] contains location IDs
# Pointer probs: ptr_probs [B, L]

# Create distribution over full vocabulary
ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
ptr_dist.scatter_add_(1, x, ptr_probs)
```

**Example**:
```
x = [42, 17, 42, 8]     # Location sequence
ptr_probs = [0.1, 0.3, 0.4, 0.2]  # Attention weights

ptr_dist[42] = 0.1 + 0.4 = 0.5  # Accumulate for location 42
ptr_dist[17] = 0.3
ptr_dist[8] = 0.2
# Other locations = 0
```

This creates a sparse distribution focused on visited locations.

---

## 6. Generation Head

### 6.1 Purpose

The generation head predicts over the **full location vocabulary**, enabling prediction of:
1. Locations not in the current sequence
2. Novel exploration patterns

### 6.2 Implementation

```python
self.gen_head = nn.Linear(d_model, num_locations)

# In forward:
gen_logits = self.gen_head(context)  # [B, num_locations]
gen_probs = F.softmax(gen_logits, dim=-1)
```

### 6.3 Comparison with Pointer

| Aspect | Pointer | Generation |
|--------|---------|------------|
| Vocabulary | Only input locations | All locations |
| Sparse/Dense | Sparse | Dense |
| Good for | Returning to known places | Exploring new places |
| Parameters | O(d_model²) | O(d_model × num_locations) |

---

## 7. Pointer-Generation Gate

### 7.1 Concept

The gate learns when to trust the pointer (copy) vs generation (generate):

- **g → 1**: Trust pointer (return to visited locations)
- **g → 0**: Trust generation (go somewhere new)

### 7.2 Implementation

```python
self.ptr_gen_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Sigmoid()
)

# In forward:
gate = self.ptr_gen_gate(context)  # [B, 1], range (0, 1)
final_probs = gate * ptr_dist + (1 - gate) * gen_probs
```

### 7.3 Mathematical Formulation

$$P_{final}(l) = g \cdot P_{ptr}(l) + (1-g) \cdot P_{gen}(l)$$

where:
- $g \in (0, 1)$ is the gate value
- $P_{ptr}(l)$ is pointer distribution over locations
- $P_{gen}(l)$ is generation distribution over locations

### 7.4 Gate Behavior

**High gate (g ≈ 1)**:
- User likely returning to visited location
- Pointer distribution dominates
- Typical for routine movements (home, work)

**Low gate (g ≈ 0)**:
- User likely exploring
- Generation distribution dominates
- Typical for weekends, new activities

---

## 8. Output and Loss

### 8.1 Final Output

```python
return torch.log(final_probs + 1e-10)  # [B, num_locations]
```

**Why log-probabilities**: 
- Numerical stability (avoid underflow)
- Compatible with NLLLoss / CrossEntropyLoss
- Log-space addition for probability multiplication

### 8.2 Loss Function

```python
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
loss = criterion(log_probs, targets)
```

**Label Smoothing**:
- Prevents overconfident predictions
- Smoothed target: $(1-\epsilon) \cdot \text{one-hot} + \epsilon / K$
- $\epsilon = 0.05$ means 5% of probability mass spread across all classes

---

## 9. Parameter Count

### 9.1 Component-wise Breakdown (DIY, d_model=64)

| Component | Parameters | Calculation |
|-----------|------------|-------------|
| Location embedding | ~1.3M | num_locations × 64 |
| User embedding | ~128K | num_users × 64 |
| Time embedding | 1,552 | 97 × 16 |
| Weekday embedding | 128 | 8 × 16 |
| Recency embedding | 144 | 9 × 16 |
| Duration embedding | 1,600 | 100 × 16 |
| Position-from-end | 2,416 | 151 × 16 |
| Input projection | 13,376 | 208 × 64 + 64 |
| Input LayerNorm | 128 | 64 × 2 |
| Transformer (2 layers) | ~100K | Complex |
| Pointer query/key | 8,192 | 64 × 64 × 2 |
| Position bias | 150 | 150 |
| Generation head | ~1.3M | 64 × num_locations |
| Gate network | 2,081 | 64×32 + 32×1 + biases |
| **Total** | **~2.8M** | - |

### 9.2 Counting Function

```python
def count_parameters(self):
    return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

## 10. Forward Pass Walkthrough

### 10.1 Input

```python
x: [seq_len, batch_size] = [24, 64]  # 24 timesteps, batch of 64
x_dict: {
    'user': [64],           # User IDs
    'time': [24, 64],       # Time of day
    'weekday': [24, 64],    # Day of week
    'diff': [24, 64],       # Days ago
    'duration': [24, 64],   # Duration
    'len': [64],            # Sequence lengths
}
```

### 10.2 Step-by-Step Shapes

```python
# 1. Transpose input
x = x.T  # [64, 24]

# 2. Embeddings
loc_emb = self.loc_emb(x)  # [64, 24, 64]
user_emb = ...             # [64, 24, 64]
temporal = ...             # [64, 24, 64]
pos_emb = ...              # [64, 24, 16]

# 3. Combine and project
combined = cat(...)        # [64, 24, 208]
hidden = input_proj(combined)  # [64, 24, 64]
hidden = input_norm(hidden)    # [64, 24, 64]
hidden = hidden + pos_encoding # [64, 24, 64]

# 4. Transformer
encoded = transformer(hidden)  # [64, 24, 64]

# 5. Extract context
context = encoded[batch_idx, last_idx]  # [64, 64]

# 6. Pointer mechanism
ptr_scores = ...           # [64, 24]
ptr_probs = softmax(...)   # [64, 24]
ptr_dist = scatter(...)    # [64, num_locations]

# 7. Generation
gen_probs = softmax(gen_head(context))  # [64, num_locations]

# 8. Gate and combine
gate = ptr_gen_gate(context)  # [64, 1]
final_probs = gate * ptr_dist + (1-gate) * gen_probs  # [64, num_locations]

# 9. Output
return log(final_probs + 1e-10)  # [64, num_locations]
```

---

## 11. Why This Architecture Works for Mobility

### 11.1 Pointer Mechanism Matches Mobility Patterns

Human mobility is dominated by **returns to known locations**:
- ~80% of movements are to previously visited places
- Pointer mechanism naturally captures this

### 11.2 Generation Handles Exploration

Occasional visits to new places:
- Generation head can predict any location
- Gate learns when exploration is likely

### 11.3 Temporal Features Capture Routines

Rich temporal encoding captures:
- Time-of-day dependencies (lunch at noon)
- Day-of-week patterns (gym on Mondays)
- Recency effects (recent locations more likely)

### 11.4 Transformer Handles Variable History

Self-attention naturally handles:
- Variable sequence lengths
- Long-range dependencies
- Permutation-aware processing

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,600 |
| **Status** | Final |

---

**Navigation**: [← Technical Implementation](./05_technical_implementation.md) | [Index](./INDEX.md) | [Next: Datasets →](./07_datasets.md)
