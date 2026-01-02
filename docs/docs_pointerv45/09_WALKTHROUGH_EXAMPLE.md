# Line-by-Line Walkthrough with Examples

This document provides a detailed walkthrough of the Pointer Network V45 model, tracing data flow through each component with concrete numerical examples.

---

## 1. Example Setup

### 1.1 Sample Input Data

Let's trace a single sample through the model:

```python
# Configuration
num_locations = 10  # Small vocabulary for illustration
num_users = 3
d_model = 8         # Small for illustration
nhead = 2
seq_len = 4
batch_size = 1

# Input: User 1 visited locations [2, 5, 3, 5]
x = torch.tensor([[2], [5], [3], [5]])  # [seq_len=4, batch_size=1]

# Target: Will visit location 2
y = torch.tensor([2])  # [batch_size=1]

# Temporal features
x_dict = {
    'user': torch.tensor([1]),                      # User ID
    'len': torch.tensor([4]),                       # Sequence length
    'time': torch.tensor([[32], [48], [36], [52]]), # ~8am, noon, 9am, 1pm
    'weekday': torch.tensor([[1], [1], [2], [2]]),  # Mon, Mon, Tue, Tue
    'diff': torch.tensor([[3], [2], [1], [0]]),     # 3, 2, 1, 0 days ago
    'duration': torch.tensor([[4], [2], [8], [1]]), # 2hr, 1hr, 4hr, 30min
}
```

### 1.2 Interpretation

This represents:
- **User 1** with a 4-visit history
- **Day 1 (Mon)**: Office (loc 2) at 8am for 2hrs, then Lunch (loc 5) at noon for 1hr
- **Day 2 (Tue)**: Office (loc 3) at 9am for 4hrs, then Lunch (loc 5) at 1pm for 30min
- **Target**: Where will they go next? (Answer: loc 2 - back to office)

---

## 2. Step 1: Input Transposition

```python
# In forward():
x = x.T  # [batch_size, seq_len]
# x: [[2, 5, 3, 5]]  shape: [1, 4]

batch_size, seq_len = x.shape  # batch_size=1, seq_len=4
device = x.device
lengths = x_dict['len']  # tensor([4])
```

**Purpose**: PyTorch Transformer expects `[batch, seq, feature]`, so we transpose.

---

## 3. Step 2: Location Embedding

```python
# self.loc_emb = nn.Embedding(10, 8)
loc_emb = self.loc_emb(x)  # [1, 4, 8]
```

**Concrete Example**:
```python
# Each location ID maps to an 8-dimensional vector
# (values are learned; these are illustrative)

loc_emb_lookup = {
    0: [0, 0, 0, 0, 0, 0, 0, 0],           # Padding
    2: [0.5, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.2],  # Office
    3: [0.4, -0.2, 0.3, 0.7, -0.2, 0.5, -0.5, 0.3],  # Different Office
    5: [0.1, 0.5, -0.4, 0.2, 0.6, -0.3, 0.4, -0.1],  # Lunch spot
}

# loc_emb result:
# Position 0 (loc 2): [0.5, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.2]
# Position 1 (loc 5): [0.1, 0.5, -0.4, 0.2, 0.6, -0.3, 0.4, -0.1]
# Position 2 (loc 3): [0.4, -0.2, 0.3, 0.7, -0.2, 0.5, -0.5, 0.3]
# Position 3 (loc 5): [0.1, 0.5, -0.4, 0.2, 0.6, -0.3, 0.4, -0.1]
```

---

## 4. Step 3: User Embedding

```python
# self.user_emb = nn.Embedding(3, 8)
user_emb = self.user_emb(x_dict['user'])  # [1, 8]
user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [1, 4, 8]
```

**Concrete Example**:
```python
# User 1's embedding
user_1_emb = [0.3, 0.1, -0.2, 0.5, 0.4, -0.1, 0.2, -0.3]

# Expanded to all positions (same user context everywhere):
# user_emb[0, 0] = user_emb[0, 1] = user_emb[0, 2] = user_emb[0, 3]
#                = [0.3, 0.1, -0.2, 0.5, 0.4, -0.1, 0.2, -0.3]
```

---

## 5. Step 4: Temporal Embeddings

```python
# Each temporal embedding has dimension d_model // 4 = 2

# Time embedding (d_model // 4 = 2)
time = torch.clamp(x_dict['time'].T, 0, 96)  # [1, 4] = [[32, 48, 36, 52]]
time_emb = self.time_emb(time)  # [1, 4, 2]

# Weekday embedding
weekday = torch.clamp(x_dict['weekday'].T, 0, 7)  # [[1, 1, 2, 2]]
weekday_emb = self.weekday_emb(weekday)  # [1, 4, 2]

# Recency embedding
recency = torch.clamp(x_dict['diff'].T, 0, 8)  # [[3, 2, 1, 0]]
recency_emb = self.recency_emb(recency)  # [1, 4, 2]

# Duration embedding  
duration = torch.clamp(x_dict['duration'].T, 0, 99)  # [[4, 2, 8, 1]]
duration_emb = self.duration_emb(duration)  # [1, 4, 2]
```

**Concrete Example**:
```python
# Time embeddings (8am=32, noon=48, 9am=36, 1pm=52)
time_emb = [
    [[0.2, -0.1],   # time=32 (8am)
     [0.5, 0.3],    # time=48 (noon)
     [0.3, 0.0],    # time=36 (9am)
     [0.6, 0.4]]    # time=52 (1pm)
]

# Weekday embeddings (Mon=1, Tue=2)
weekday_emb = [
    [[0.4, -0.2],   # Monday
     [0.4, -0.2],   # Monday
     [-0.3, 0.5],   # Tuesday
     [-0.3, 0.5]]   # Tuesday
]

# Recency embeddings (3, 2, 1, 0 days ago)
recency_emb = [
    [[0.1, 0.1],    # 3 days ago
     [0.2, 0.2],    # 2 days ago
     [0.4, 0.3],    # 1 day ago
     [0.8, 0.6]]    # 0 days ago (most recent)
]

# Duration embeddings
duration_emb = [
    [[0.3, -0.1],   # 2 hours
     [0.1, 0.0],    # 1 hour
     [0.5, -0.2],   # 4 hours
     [0.0, 0.1]]    # 30 min
]
```

---

## 6. Step 5: Position-from-End Embedding

```python
# Calculate position from end of sequence
positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
# positions = [[0, 1, 2, 3]]

pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, max_seq_len - 1)
# lengths = [4], positions = [[0, 1, 2, 3]]
# pos_from_end = [[4-0, 4-1, 4-2, 4-3]] = [[4, 3, 2, 1]]

pos_emb = self.pos_from_end_emb(pos_from_end)  # [1, 4, 2]
```

**Concrete Example**:
```python
# Position from end embeddings
# Higher values = older, lower values = more recent
pos_from_end_emb = [
    [[0.1, 0.1],    # pos_from_end=4 (oldest)
     [0.2, 0.2],    # pos_from_end=3
     [0.4, 0.3],    # pos_from_end=2
     [0.8, 0.6]]    # pos_from_end=1 (most recent)
]
```

---

## 7. Step 6: Combine Temporal Features

```python
temporal = torch.cat([
    time_emb,       # [1, 4, 2]
    weekday_emb,    # [1, 4, 2]
    recency_emb,    # [1, 4, 2]
    duration_emb    # [1, 4, 2]
], dim=-1)          # [1, 4, 8]
```

**Concrete Example**:
```python
# Concatenate all temporal features for position 3 (most recent):
temporal[0, 3] = [0.6, 0.4,    # time (1pm)
                 -0.3, 0.5,    # weekday (Tuesday)
                  0.8, 0.6,    # recency (0 days ago)
                  0.0, 0.1]    # duration (30 min)
# Shape: [8]
```

---

## 8. Step 7: Combine All Embeddings

```python
combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
# loc_emb:  [1, 4, 8]
# user_emb: [1, 4, 8]
# temporal: [1, 4, 8]
# pos_emb:  [1, 4, 2]
# combined: [1, 4, 26]

# Note: With d_model=8, temporal features are each d/4=2 dimension
# So total: 8 + 8 + 4*2 + 2 = 8 + 8 + 8 + 2 = 26
```

---

## 9. Step 8: Input Projection

```python
# self.input_proj = nn.Linear(26, 8)
hidden = self.input_proj(combined)  # [1, 4, 8]
hidden = self.input_norm(hidden)    # LayerNorm: [1, 4, 8]
```

**Purpose**: Project concatenated 26-dim features to 8-dim model space.

**Concrete Example**:
```python
# After projection and normalization for position 3:
hidden[0, 3] = [0.42, -0.31, 0.18, 0.55, -0.23, 0.38, -0.47, 0.21]
```

---

## 10. Step 9: Add Positional Encoding

```python
hidden = hidden + self.pos_encoding[:, :seq_len, :]  # [1, 4, 8]
```

**Concrete Example**:
```python
# Sinusoidal positional encoding for position 3:
pos_enc[3] = [sin(3/1), cos(3/1), sin(3/100), cos(3/100), ...]
           ≈ [0.14, -0.99, 0.03, 1.00, ...]

# Adding to hidden:
hidden[0, 3] = hidden[0, 3] + pos_enc[3]
             = [0.42+0.14, -0.31-0.99, 0.18+0.03, 0.55+1.00, ...]
             = [0.56, -1.30, 0.21, 1.55, ...]
```

---

## 11. Step 10: Create Attention Mask

```python
mask = positions >= lengths.unsqueeze(1)
# positions = [[0, 1, 2, 3]]
# lengths = [[4]]
# mask = [[False, False, False, False]]  (no padding in this example)
```

**Purpose**: Prevent attention to padding positions.

---

## 12. Step 11: Transformer Encoding

```python
encoded = self.transformer(hidden, src_key_padding_mask=mask)  # [1, 4, 8]
```

**Inside the Transformer** (simplified for 1 layer):

```python
# 1. Pre-norm
normed = LayerNorm(hidden)

# 2. Multi-head self-attention
# Each position attends to all positions
attn_output = MultiHeadAttention(normed, normed, normed)

# 3. Residual connection
hidden1 = hidden + attn_output

# 4. Pre-norm
normed1 = LayerNorm(hidden1)

# 5. Feedforward
ff_output = FFN(normed1)  # Linear → GELU → Linear

# 6. Residual connection
encoded = hidden1 + ff_output
```

**Attention Visualization** (hypothetical weights):

```
Query Pos 3 attending to all positions:
           Pos 0   Pos 1   Pos 2   Pos 3
           (loc2)  (loc5)  (loc3)  (loc5)
Attention: [0.15,   0.30,   0.20,   0.35]

Pos 3 attends most to itself (0.35) and Pos 1 (0.30) which is the same location.
```

---

## 13. Step 12: Extract Context Vector

```python
batch_idx = torch.arange(batch_size, device=device)  # [0]
last_idx = (lengths - 1).clamp(min=0)                # [3]
context = encoded[batch_idx, last_idx]               # [1, 8]

# context = encoded[0, 3] - the encoding of the most recent position
```

**Concrete Example**:
```python
context = encoded[0, 3]  # The last valid position's encoding
        = [0.38, -0.22, 0.45, 0.67, -0.18, 0.29, -0.41, 0.15]
```

---

## 14. Step 13: Pointer Mechanism

### 14.1 Query and Key Projection

```python
query = self.pointer_query(context).unsqueeze(1)  # [1, 1, 8]
keys = self.pointer_key(encoded)                   # [1, 4, 8]
```

**Concrete Example**:
```python
query = [[[ 0.25, -0.18, 0.32, 0.54, -0.12, 0.21, -0.35, 0.10]]]
keys = [[[ 0.30, -0.15, 0.28, 0.50, -0.10, 0.25, -0.30, 0.12],  # pos 0
         [ 0.12, 0.35, -0.25, 0.18, 0.40, -0.22, 0.30, -0.08],  # pos 1
         [ 0.28, -0.12, 0.30, 0.48, -0.15, 0.28, -0.28, 0.15],  # pos 2
         [ 0.15, 0.38, -0.28, 0.20, 0.42, -0.20, 0.32, -0.10]]] # pos 3
```

### 14.2 Compute Attention Scores

```python
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(d_model)
# query: [1, 1, 8], keys.T: [1, 8, 4]
# scores: [1, 4]
```

**Concrete Example**:
```python
# Dot product of query with each key, divided by sqrt(8)=2.83
raw_scores = [
    dot(query, keys[0]),  # High - similar location context
    dot(query, keys[1]),  # Lower - different location
    dot(query, keys[2]),  # Medium
    dot(query, keys[3])   # Lower - different location
]
ptr_scores = raw_scores / 2.83
           = [0.85, 0.32, 0.65, 0.40]
```

### 14.3 Add Position Bias

```python
ptr_scores = ptr_scores + self.position_bias[pos_from_end]
# pos_from_end = [[4, 3, 2, 1]]
# position_bias might be learned as: [..., -0.2, 0.0, 0.3, 0.8, ...]
#                                         pos4  pos3 pos2 pos1
```

**Concrete Example**:
```python
position_bias = [-0.2, 0.0, 0.3, 0.8]  # Favors recent positions
ptr_scores_biased = [0.85-0.2, 0.32+0.0, 0.65+0.3, 0.40+0.8]
                  = [0.65, 0.32, 0.95, 1.20]
```

### 14.4 Softmax and Scatter

```python
# Mask padding (none in this example)
ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))

# Softmax over sequence positions
ptr_probs = F.softmax(ptr_scores, dim=-1)  # [1, 4]
```

**Concrete Example**:
```python
# softmax([0.65, 0.32, 0.95, 1.20])
exp_scores = [1.92, 1.38, 2.59, 3.32]
ptr_probs = [0.19, 0.14, 0.26, 0.34]  # Attention weights
# Highest attention on position 3 (most recent)
```

### 14.5 Scatter to Location Vocabulary

```python
ptr_dist = torch.zeros(batch_size, num_locations, device=device)  # [1, 10]
ptr_dist.scatter_add_(1, x, ptr_probs)  # x = [[2, 5, 3, 5]]
```

**Concrete Example**:
```python
# Input locations: [2, 5, 3, 5]
# Attention: [0.19, 0.14, 0.26, 0.34]

# Scatter attention to location vocabulary:
ptr_dist = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]  # 10 locations
#              0  1  2  3  4  5  6  7  8  9

# Location 2 gets attention from position 0: ptr_dist[2] += 0.19
# Location 5 gets attention from positions 1 and 3: ptr_dist[5] += 0.14 + 0.34 = 0.48
# Location 3 gets attention from position 2: ptr_dist[3] += 0.26

ptr_dist = [0, 0, 0.19, 0.26, 0, 0.48, 0, 0, 0, 0]
#              0  1   2     3  4    5  6  7  8  9
```

---

## 15. Step 14: Generation Head

```python
gen_probs = F.softmax(self.gen_head(context), dim=-1)  # [1, 10]
```

**Concrete Example**:
```python
# Linear projection context → 10 logits, then softmax
gen_logits = [0.2, 0.1, 1.5, 0.8, 0.3, 1.2, 0.1, 0.2, 0.3, 0.1]
gen_probs = softmax(gen_logits)
          = [0.04, 0.04, 0.18, 0.09, 0.05, 0.13, 0.04, 0.05, 0.05, 0.04]
# Generation prefers location 2 (0.18) and location 5 (0.13)
```

---

## 16. Step 15: Gate Computation

```python
gate = self.ptr_gen_gate(context)  # [1, 1]
# MLP: Linear(8→4) → GELU → Linear(4→1) → Sigmoid
```

**Concrete Example**:
```python
# Gate computation
hidden = Linear_1(context)  # [8] → [4]
hidden = GELU(hidden)
gate_logit = Linear_2(hidden)  # [4] → [1]
gate = sigmoid(gate_logit)  # scalar in [0, 1]

gate = 0.72  # Model thinks 72% should come from pointer
```

---

## 17. Step 16: Final Probability

```python
final_probs = gate * ptr_dist + (1 - gate) * gen_probs  # [1, 10]
```

**Concrete Example**:
```python
gate = 0.72

# Weighted combination:
# ptr_dist = [0, 0, 0.19, 0.26, 0, 0.48, 0, 0, 0, 0]
# gen_probs = [0.04, 0.04, 0.18, 0.09, 0.05, 0.13, 0.04, 0.05, 0.05, 0.04]

final_probs = 0.72 * ptr_dist + 0.28 * gen_probs
            = 0.72 * [0, 0, 0.19, 0.26, 0, 0.48, 0, 0, 0, 0]
            + 0.28 * [0.04, 0.04, 0.18, 0.09, 0.05, 0.13, 0.04, 0.05, 0.05, 0.04]

# Location 2: 0.72 * 0.19 + 0.28 * 0.18 = 0.137 + 0.050 = 0.187
# Location 3: 0.72 * 0.26 + 0.28 * 0.09 = 0.187 + 0.025 = 0.212
# Location 5: 0.72 * 0.48 + 0.28 * 0.13 = 0.346 + 0.036 = 0.382
# Others: Small contributions from generation

final_probs ≈ [0.01, 0.01, 0.19, 0.21, 0.01, 0.38, 0.01, 0.01, 0.01, 0.01]
```

---

## 18. Step 17: Log Probabilities

```python
return torch.log(final_probs + 1e-10)  # [1, 10]
```

**Concrete Example**:
```python
log_probs = log(final_probs + 1e-10)
          = log([0.01, 0.01, 0.19, 0.21, 0.01, 0.38, ...])
          = [-4.6, -4.6, -1.66, -1.56, -4.6, -0.97, ...]
```

---

## 19. Prediction

```python
prediction = log_probs.argmax(dim=-1)  # [1]
# prediction = [5]  (location 5 has highest probability)
```

**Analysis**:
- Model predicts **location 5** (lunch spot)
- True answer was **location 2** (office)
- Model was influenced by the repetition of location 5 in history
- The pointer mechanism accumulated probability for location 5 (appeared twice)

---

## 20. Loss Computation (Training)

```python
loss = criterion(log_probs, y)  # y = [2]

# CrossEntropyLoss with log_probs
# Since true label is 2, loss = -log_probs[2] = -(-1.66) = 1.66
```

---

## 21. Summary Flow Diagram

```
Input: x=[2,5,3,5], user=1, temporal features
                    ↓
┌─────────────────────────────────────────────────────┐
│                   EMBEDDING                          │
│  loc_emb + user_emb + temporal_emb + pos_from_end   │
│                    [1, 4, 26]                        │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│              INPUT PROJECTION + NORM                 │
│                   [1, 4, 26] → [1, 4, 8]            │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│              ADD POSITIONAL ENCODING                 │
│                    [1, 4, 8]                         │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│               TRANSFORMER ENCODER                    │
│                    [1, 4, 8]                         │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│          EXTRACT CONTEXT (last position)             │
│                    [1, 8]                            │
└─────────────────────────────────────────────────────┘
            ↓           ↓           ↓
┌────────────────┐ ┌────────────────┐ ┌────────────────┐
│ POINTER HEAD   │ │ GENERATION     │ │ GATE           │
│ Q·K attention  │ │ Linear+Softmax │ │ MLP+Sigmoid    │
│ +position bias │ │                │ │                │
│ Scatter to V   │ │                │ │                │
│   [1, 10]      │ │   [1, 10]      │ │   [1, 1]       │
└────────────────┘ └────────────────┘ └────────────────┘
            ↓           ↓           ↓
┌─────────────────────────────────────────────────────┐
│           COMBINE: gate*ptr + (1-gate)*gen          │
│                    [1, 10]                           │
└─────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────┐
│                 LOG PROBABILITIES                    │
│                    [1, 10]                           │
└─────────────────────────────────────────────────────┘
                    ↓
            Output: log P(location)
```

---

## 22. Key Takeaways from Walkthrough

1. **Pointer mechanism accumulates probability for repeated locations**
   - Location 5 appeared twice → got higher pointer probability

2. **Position bias favors recent positions**
   - Most recent position (3) had highest raw attention

3. **Gate balances pointer and generation**
   - High gate (0.72) → trusts history more

4. **Generation provides a fallback**
   - Even unseen locations get some probability

5. **Temporal features influence encoding**
   - Same location at different times has different encodings

---

*Next: [10_DIAGRAMS.md](10_DIAGRAMS.md) - Visual Diagrams*
