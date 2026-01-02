# Complete Data Flow: From Raw Input to Final Prediction

## An End-to-End Journey Through the Model

This document traces every single step of data transformation from raw input to final prediction, with exact tensor shapes, numerical examples, and explanations of what happens at each stage.

---

## 1. Raw Data Format

### 1.1 What the Training Data Looks Like

Each sample in the preprocessed pickle file is a dictionary:

```python
sample = {
    'X': np.array([2, 5, 3, 7, 5, 2, 8, 3]),  # Location IDs visited
    'Y': 5,                                     # Target: next location to predict
    'user_X': np.array([1, 1, 1, 1, 1, 1, 1, 1]),  # User ID (repeated)
    'weekday_X': np.array([0, 0, 1, 1, 2, 2, 3, 3]),  # Day of week for each visit
    'start_min_X': np.array([480, 720, 510, 750, 495, 735, 525, 765]),  # Start time in minutes
    'dur_X': np.array([120, 45, 480, 60, 30, 90, 420, 45]),  # Duration in minutes
    'diff': np.array([7, 7, 5, 5, 3, 3, 1, 1]),  # Days ago from target
}
```

### 1.2 Interpretation of Sample

This represents a user's 8-visit history over 7 days:

| Position | Location | Day | Time | Duration | Days Ago |
|----------|----------|-----|------|----------|----------|
| 0 | 2 (Home) | Mon | 8:00 AM | 2 hrs | 7 |
| 1 | 5 (Work) | Mon | 12:00 PM | 45 min | 7 |
| 2 | 3 (Gym) | Wed | 8:30 AM | 8 hrs | 5 |
| 3 | 7 (Restaurant) | Wed | 12:30 PM | 1 hr | 5 |
| 4 | 5 (Work) | Fri | 8:15 AM | 30 min | 3 |
| 5 | 2 (Home) | Fri | 12:15 PM | 1.5 hrs | 3 |
| 6 | 8 (Cafe) | Sun | 8:45 AM | 7 hrs | 1 |
| 7 | 3 (Gym) | Sun | 12:45 PM | 45 min | 1 |

**Target**: Predict next location (5 = Work)

---

## 2. Dataset and DataLoader

### 2.1 Dataset `__getitem__` Processing

```python
def __getitem__(self, idx):
    sample = self.data[idx]
    
    # Time: Convert minutes to 15-minute buckets
    # 480 minutes = 8:00 AM = bucket 32 (480 // 15 = 32)
    time_buckets = sample['start_min_X'] // 15
    # [480, 720, 510, 750, 495, 735, 525, 765] // 15
    # = [32, 48, 34, 50, 33, 49, 35, 51]
    
    # Duration: Convert minutes to 30-minute buckets
    duration_buckets = sample['dur_X'] // 30
    # [120, 45, 480, 60, 30, 90, 420, 45] // 30
    # = [4, 1, 16, 2, 1, 3, 14, 1]
    
    return_dict = {
        'user': torch.tensor(1, dtype=torch.long),
        'weekday': torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.long),
        'time': torch.tensor([32, 48, 34, 50, 33, 49, 35, 51], dtype=torch.long),
        'duration': torch.tensor([4, 1, 16, 2, 1, 3, 14, 1], dtype=torch.long),
        'diff': torch.tensor([7, 7, 5, 5, 3, 3, 1, 1], dtype=torch.long),
    }
    
    x = torch.tensor([2, 5, 3, 7, 5, 2, 8, 3], dtype=torch.long)
    y = torch.tensor(5, dtype=torch.long)
    
    return x, y, return_dict
```

### 2.2 Collate Function (Batching)

When multiple samples are batched together:

```python
# Assume batch_size = 2, with sequences of length 8 and 6

# Padding with 0s
x_batch = [
    [2, 5, 3, 7, 5, 2, 8, 3],  # Sample 1: length 8
    [4, 2, 6, 4, 2, 0, 0, 0],  # Sample 2: length 5, padded to 8
]

# After pad_sequence (batch_first=False):
x_batch.shape = [8, 2]  # [seq_len, batch_size]

# Lengths stored for masking
x_dict['len'] = [8, 5]
```

---

## 3. Model Forward Pass - Step by Step

### 3.1 Step 0: Input Transposition

```python
# Input: x.shape = [seq_len=8, batch_size=2]
x = x.T  # Transpose
# Output: x.shape = [batch_size=2, seq_len=8]

x = [[2, 5, 3, 7, 5, 2, 8, 3],   # Sample 1
     [4, 2, 6, 4, 2, 0, 0, 0]]   # Sample 2 (padded)

batch_size, seq_len = 2, 8
```

### 3.2 Step 1: Location Embedding

```python
# self.loc_emb = nn.Embedding(num_locations=10, d_model=8)
loc_emb = self.loc_emb(x)  # [2, 8, 8]
```

**What happens:**
- Each location ID is mapped to a learned 8-dimensional vector
- Padding (0) maps to zero vector due to `padding_idx=0`

```python
# Conceptually (with made-up embedding values):
loc_emb[0, 0] = embed(2) = [0.5, -0.3, 0.2, 0.8, -0.1, 0.4, -0.6, 0.2]  # Home
loc_emb[0, 1] = embed(5) = [0.1, 0.5, -0.4, 0.2, 0.6, -0.3, 0.4, -0.1]  # Work
loc_emb[0, 2] = embed(3) = [0.4, -0.2, 0.3, 0.7, -0.2, 0.5, -0.5, 0.3]  # Gym
# ... etc

# For padded positions:
loc_emb[1, 5] = embed(0) = [0, 0, 0, 0, 0, 0, 0, 0]  # Zero vector
```

### 3.3 Step 2: User Embedding

```python
# x_dict['user'] = [1, 2]  (user IDs for each sample)
# self.user_emb = nn.Embedding(num_users=10, d_model=8)

user_emb = self.user_emb(x_dict['user'])  # [2, 8]
# Shape: [batch_size, d_model]

user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [2, 8, 8]
# Shape: [batch_size, seq_len, d_model]
# Same user embedding repeated for all positions
```

**What happens:**
- Each user ID maps to an 8-dimensional vector capturing their preferences
- This is broadcasted to all positions (same user context everywhere)

### 3.4 Step 3: Temporal Embeddings

```python
# Time of day embedding
time = torch.clamp(x_dict['time'].T, 0, 96)  # [2, 8]
# Input: [[32, 48, 34, 50, 33, 49, 35, 51], [28, 44, 30, 46, 29, 0, 0, 0]]
time_emb = self.time_emb(time)  # [2, 8, 2] (d_model // 4 = 2)

# Weekday embedding
weekday = torch.clamp(x_dict['weekday'].T, 0, 7)  # [2, 8]
# Input: [[0, 0, 1, 1, 2, 2, 3, 3], [1, 1, 2, 2, 3, 0, 0, 0]]
weekday_emb = self.weekday_emb(weekday)  # [2, 8, 2]

# Recency embedding
recency = torch.clamp(x_dict['diff'].T, 0, 8)  # [2, 8]
# Input: [[7, 7, 5, 5, 3, 3, 1, 1], [6, 6, 4, 4, 2, 0, 0, 0]]
recency_emb = self.recency_emb(recency)  # [2, 8, 2]

# Duration embedding
duration = torch.clamp(x_dict['duration'].T, 0, 99)  # [2, 8]
# Input: [[4, 1, 16, 2, 1, 3, 14, 1], [3, 2, 10, 1, 2, 0, 0, 0]]
duration_emb = self.duration_emb(duration)  # [2, 8, 2]

# Concatenate all temporal embeddings
temporal = torch.cat([time_emb, weekday_emb, recency_emb, duration_emb], dim=-1)
# Shape: [2, 8, 8] (4 embeddings × 2 dimensions each)
```

### 3.5 Step 4: Position-from-End Embedding

```python
# Compute positions
positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
# positions = [[0, 1, 2, 3, 4, 5, 6, 7],
#              [0, 1, 2, 3, 4, 5, 6, 7]]

# Compute position from end
lengths = x_dict['len']  # [8, 5]
pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, max_seq_len - 1)
# For sample 1 (length 8): [8-0, 8-1, 8-2, ..., 8-7] = [8, 7, 6, 5, 4, 3, 2, 1]
# For sample 2 (length 5): [5-0, 5-1, 5-2, ..., 5-7] = [5, 4, 3, 2, 1, 0, 0, 0]
#   (clamped to ≥0 for padded positions)

pos_from_end = [[8, 7, 6, 5, 4, 3, 2, 1],
                [5, 4, 3, 2, 1, 0, 0, 0]]  # Note: padding gets 0

pos_emb = self.pos_from_end_emb(pos_from_end)  # [2, 8, 2]
```

### 3.6 Step 5: Combine All Embeddings

```python
combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
# Shapes:
# - loc_emb:  [2, 8, 8]
# - user_emb: [2, 8, 8]
# - temporal: [2, 8, 8]
# - pos_emb:  [2, 8, 2]
# Total: [2, 8, 26]

# For d_model=8, the actual dimensions are:
# - loc_emb:  d_model = 8
# - user_emb: d_model = 8
# - temporal: 4 × (d_model // 4) = 4 × 2 = 8
# - pos_emb:  d_model // 4 = 2
# Total: 8 + 8 + 8 + 2 = 26
```

### 3.7 Step 6: Input Projection

```python
# self.input_proj = nn.Linear(26, 8)
# self.input_norm = nn.LayerNorm(8)

hidden = self.input_proj(combined)  # [2, 8, 26] → [2, 8, 8]
hidden = self.input_norm(hidden)    # [2, 8, 8]
```

**What the linear projection does:**
- Learns to weight and combine all 26 input features
- Maps to uniform d_model dimension for Transformer

**What LayerNorm does:**
```
For each position (independently):
x_norm = (x - mean(x)) / sqrt(var(x) + ε) * γ + β

Where γ, β are learnable scale and shift parameters
```

### 3.8 Step 7: Add Positional Encoding

```python
# self.pos_encoding: precomputed sinusoidal [1, max_len, d_model]
hidden = hidden + self.pos_encoding[:, :seq_len, :]  # [2, 8, 8]
```

**Sinusoidal values for position 0, d_model=8:**
```python
PE[0] = [sin(0), cos(0), sin(0), cos(0), sin(0), cos(0), sin(0), cos(0)]
      = [0, 1, 0, 1, 0, 1, 0, 1]

PE[1] = [sin(1/1), cos(1/1), sin(1/10), cos(1/10), sin(1/100), cos(1/100), ...]
      ≈ [0.84, 0.54, 0.10, 0.99, 0.01, 1.00, ...]
```

### 3.9 Step 8: Create Padding Mask

```python
# lengths = [8, 5]
# positions = [[0, 1, 2, 3, 4, 5, 6, 7],
#              [0, 1, 2, 3, 4, 5, 6, 7]]

mask = positions >= lengths.unsqueeze(1)
# mask = [[0>=8, 1>=8, 2>=8, 3>=8, 4>=8, 5>=8, 6>=8, 7>=8],
#         [0>=5, 1>=5, 2>=5, 3>=5, 4>=5, 5>=5, 6>=5, 7>=5]]
#      = [[False, False, False, False, False, False, False, False],
#         [False, False, False, False, False, True,  True,  True]]
```

**Purpose:** Prevent attention to padding positions.

### 3.10 Step 9: Transformer Encoding

```python
encoded = self.transformer(hidden, src_key_padding_mask=mask)  # [2, 8, 8]
```

**Inside the Transformer (for each layer):**

```python
# Pre-norm self-attention
normed = LayerNorm(hidden)

# Multi-head attention
Q = normed @ W_Q  # [2, 8, 8]
K = normed @ W_K  # [2, 8, 8]
V = normed @ W_V  # [2, 8, 8]

# Split into heads (nhead=2, head_dim=4)
Q = Q.reshape(2, 8, 2, 4).transpose(1, 2)  # [2, 2, 8, 4]
K = K.reshape(2, 8, 2, 4).transpose(1, 2)  # [2, 2, 8, 4]
V = V.reshape(2, 8, 2, 4).transpose(1, 2)  # [2, 2, 8, 4]

# Attention scores
scores = Q @ K.transpose(-2, -1) / sqrt(4)  # [2, 2, 8, 8]
scores = scores.masked_fill(mask, -inf)     # Mask padding
attn = softmax(scores, dim=-1)              # [2, 2, 8, 8]

# Weighted sum
attn_out = attn @ V  # [2, 2, 8, 4]
attn_out = attn_out.transpose(1, 2).reshape(2, 8, 8)  # [2, 8, 8]

# Residual
hidden1 = hidden + attn_out

# Pre-norm feedforward
normed1 = LayerNorm(hidden1)
ff_out = Linear2(GELU(Linear1(normed1)))  # [2, 8, 8]

# Residual
encoded = hidden1 + ff_out  # [2, 8, 8]
```

### 3.11 Step 10: Extract Context Vector

```python
# Get the encoding at the last valid position for each sample
batch_idx = torch.arange(batch_size)  # [0, 1]
last_idx = (lengths - 1).clamp(min=0)  # [7, 4]

context = encoded[batch_idx, last_idx]  # [2, 8]
# context[0] = encoded[0, 7]  (last position of sample 1)
# context[1] = encoded[1, 4]  (last valid position of sample 2)
```

### 3.12 Step 11: Pointer Mechanism

```python
# Query from context
query = self.pointer_query(context)  # [2, 8] → [2, 8]
query = query.unsqueeze(1)           # [2, 1, 8]

# Keys from encoded sequence
keys = self.pointer_key(encoded)     # [2, 8, 8]

# Attention scores
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)  # [2, 8]
ptr_scores = ptr_scores / math.sqrt(self.d_model)  # Scale by √8

# Add position bias
# self.position_bias: learnable [max_seq_len]
# pos_from_end: [[8, 7, 6, 5, 4, 3, 2, 1], [5, 4, 3, 2, 1, 0, 0, 0]]
ptr_scores = ptr_scores + self.position_bias[pos_from_end]

# Mask padding
ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))

# Softmax to get attention weights
ptr_probs = F.softmax(ptr_scores, dim=-1)  # [2, 8]
# Sample 1: probabilities over all 8 positions
# Sample 2: probabilities over positions 0-4 (5-7 are -inf → 0 after softmax)
```

**Example attention weights:**
```python
ptr_probs[0] = [0.05, 0.12, 0.03, 0.08, 0.15, 0.07, 0.20, 0.30]
# Most attention on positions 7 (0.30) and 6 (0.20) - most recent

ptr_probs[1] = [0.10, 0.15, 0.20, 0.25, 0.30, 0.00, 0.00, 0.00]
# Position 4 gets highest attention (most recent valid)
```

### 3.13 Step 12: Scatter to Location Vocabulary

```python
# x = [[2, 5, 3, 7, 5, 2, 8, 3],
#      [4, 2, 6, 4, 2, 0, 0, 0]]

ptr_dist = torch.zeros(batch_size, num_locations)  # [2, 10]
ptr_dist.scatter_add_(1, x, ptr_probs)
```

**What scatter_add_ does:**

For sample 1 with x=[2, 5, 3, 7, 5, 2, 8, 3] and ptr_probs=[0.05, 0.12, 0.03, 0.08, 0.15, 0.07, 0.20, 0.30]:

```python
ptr_dist[0, 2] += 0.05  # Position 0: location 2
ptr_dist[0, 5] += 0.12  # Position 1: location 5
ptr_dist[0, 3] += 0.03  # Position 2: location 3
ptr_dist[0, 7] += 0.08  # Position 3: location 7
ptr_dist[0, 5] += 0.15  # Position 4: location 5 (ACCUMULATES!)
ptr_dist[0, 2] += 0.07  # Position 5: location 2 (ACCUMULATES!)
ptr_dist[0, 8] += 0.20  # Position 6: location 8
ptr_dist[0, 3] += 0.30  # Position 7: location 3 (ACCUMULATES!)

# Result:
ptr_dist[0] = [0, 0, 0.12, 0.33, 0, 0.27, 0, 0.08, 0.20, 0]
#              0  1    2     3   4    5   6    7     8   9
#                    Home  Gym     Work    Rest  Cafe

# Location 3 (Gym) gets 0.03 + 0.30 = 0.33 (highest!)
# Location 5 (Work) gets 0.12 + 0.15 = 0.27
# Location 2 (Home) gets 0.05 + 0.07 = 0.12
```

### 3.14 Step 13: Generation Distribution

```python
# self.gen_head = nn.Linear(8, 10)  # d_model → num_locations
gen_logits = self.gen_head(context)  # [2, 8] → [2, 10]
gen_probs = F.softmax(gen_logits, dim=-1)  # [2, 10]
```

**Example:**
```python
gen_probs[0] = [0.02, 0.03, 0.10, 0.12, 0.05, 0.25, 0.08, 0.15, 0.12, 0.08]
#                 0     1     2     3     4     5     6     7     8     9
# Generation head predicts location 5 (Work) with highest probability
```

### 3.15 Step 14: Compute Gate

```python
# self.ptr_gen_gate = Sequential(Linear(8, 4), GELU, Linear(4, 1), Sigmoid)
gate = self.ptr_gen_gate(context)  # [2, 8] → [2, 1]
```

**Example:**
```python
gate[0] = 0.72  # 72% pointer, 28% generation
gate[1] = 0.65  # 65% pointer, 35% generation
```

### 3.16 Step 15: Combine Distributions

```python
final_probs = gate * ptr_dist + (1 - gate) * gen_probs  # [2, 10]
```

**For sample 1:**
```python
ptr_dist[0] = [0, 0, 0.12, 0.33, 0, 0.27, 0, 0.08, 0.20, 0]
gen_probs[0] = [0.02, 0.03, 0.10, 0.12, 0.05, 0.25, 0.08, 0.15, 0.12, 0.08]
gate[0] = 0.72

final_probs[0] = 0.72 * ptr_dist[0] + 0.28 * gen_probs[0]

# Location 5 (Work) - the target:
final_probs[0, 5] = 0.72 * 0.27 + 0.28 * 0.25
                  = 0.194 + 0.070
                  = 0.264

# Full distribution:
final_probs[0] = [0.006, 0.008, 0.114, 0.271, 0.014, 0.264, 0.022, 0.100, 0.178, 0.022]
#                   0      1      2      3      4      5      6      7      8      9

# Top predictions:
# 1. Location 3 (Gym): 0.271
# 2. Location 5 (Work): 0.264  ← Target! (rank 2)
# 3. Location 8 (Cafe): 0.178
```

### 3.17 Step 16: Log Probabilities

```python
output = torch.log(final_probs + 1e-10)  # [2, 10]
```

**Example:**
```python
output[0, 5] = log(0.264 + 1e-10) = -1.33
```

---

## 4. Loss Computation

### 4.1 Cross-Entropy Loss

```python
# y = [5, 3]  (targets for each sample)
# output = log(final_probs + ε)

loss = F.cross_entropy(output, y, ignore_index=0, label_smoothing=0.03)
```

**Without label smoothing:**
```python
loss_sample1 = -output[0, 5] = -(-1.33) = 1.33
```

**With label smoothing (ε=0.03):**
```python
# Target distribution for sample 1:
y_smooth[5] = (1 - 0.03) = 0.97
y_smooth[other] = 0.03 / 9 ≈ 0.0033

loss_sample1 = -Σᵢ y_smooth[i] * output[0, i]
             = -0.97 * (-1.33) - 0.0033 * (sum of other log probs)
             ≈ 1.29 + 0.02
             ≈ 1.31
```

---

## 5. Backward Pass (Gradient Flow)

### 5.1 Gradient to Final Probs

```python
∂L/∂final_probs[0, 5] = -y_smooth[5] / (final_probs[0, 5] + ε)
                      = -0.97 / 0.264
                      ≈ -3.67
```

### 5.2 Gradient to Gate

```python
∂L/∂gate[0] = ∂L/∂final_probs · ∂final_probs/∂gate
            = Σᵢ (∂L/∂final_probs[0,i]) * (ptr_dist[0,i] - gen_probs[0,i])
            
# For location 5:
∂final_probs[0,5]/∂gate = ptr_dist[0,5] - gen_probs[0,5]
                        = 0.27 - 0.25 = 0.02
```

### 5.3 Gradient to Pointer Distribution

```python
∂L/∂ptr_dist[0, 5] = ∂L/∂final_probs[0, 5] * gate[0]
                   = -3.67 * 0.72
                   = -2.64
```

### 5.4 Gradient to Pointer Attention

Through scatter_add_, gradient flows back to attention weights that contributed to location 5:
- Position 1 (location 5): Gets gradient
- Position 4 (location 5): Gets gradient

### 5.5 Gradient to Transformer

From pointer attention, gradients flow to:
- Query projection (from context)
- Key projection (from encoded sequence)
- Position bias parameters

From generation head:
- Linear layer weights
- Bias terms

All gradients eventually flow to:
- All embedding tables
- Transformer parameters
- Gate MLP parameters

---

## 6. Summary of Data Transformations

| Stage | Input Shape | Output Shape | Key Operation |
|-------|-------------|--------------|---------------|
| Raw Input | Dictionary | - | Load from pickle |
| Dataset | Dictionary | Tensors | Convert to tensors, bucket time/duration |
| Collate | List of samples | Batched tensors | Pad sequences |
| Transpose | [S, B] | [B, S] | x.T |
| Location Embed | [B, S] int | [B, S, d] float | Lookup |
| User Embed | [B] int | [B, S, d] float | Lookup + expand |
| Temporal Embed | [B, S] int each | [B, S, d] float | Lookups + concat |
| Pos-from-End | [B, S] int | [B, S, d/4] float | Compute + lookup |
| Combine | Multiple [B, S, *] | [B, S, 26] float | Concatenate |
| Project | [B, S, 26] | [B, S, d] | Linear + LayerNorm |
| Add PE | [B, S, d] | [B, S, d] | Add sinusoidal |
| Transformer | [B, S, d] | [B, S, d] | Self-attention + FFN |
| Extract Context | [B, S, d] | [B, d] | Index last valid |
| Pointer | [B, d] + [B, S, d] | [B, V] | Attention + scatter |
| Generation | [B, d] | [B, V] | Linear + softmax |
| Gate | [B, d] | [B, 1] | MLP + sigmoid |
| Combine | [B, V] × 2 + [B, 1] | [B, V] | Weighted sum |
| Output | [B, V] | [B, V] | Log |

---

*This document is part of the comprehensive Pointer V45 documentation series.*
