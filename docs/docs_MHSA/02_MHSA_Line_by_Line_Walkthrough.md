# MHSA Model: Line-by-Line Walkthrough with Examples

## Complete Code Walkthrough with Concrete Examples

**Purpose:** This document walks through the MHSA model code line by line with concrete numerical examples to help understand exactly what happens at each step.

---

## Table of Contents

1. [Sample Input Data](#1-sample-input-data)
2. [PositionalEncoding Walkthrough](#2-positionalencoding-walkthrough)
3. [TemporalEmbedding Walkthrough](#3-temporalembedding-walkthrough)
4. [AllEmbedding Walkthrough](#4-allembedding-walkthrough)
5. [TransformerEncoder Walkthrough](#5-transformerencoder-walkthrough)
6. [FullyConnected Walkthrough](#6-fullyconnected-walkthrough)
7. [Complete Forward Pass](#7-complete-forward-pass)
8. [Training Step Walkthrough](#8-training-step-walkthrough)

---

## 1. Sample Input Data

Let's use a concrete example throughout this walkthrough.

### 1.1 Raw Sample from Dataset

```python
# One sample from the training data (pickle file)
sample = {
    'X': np.array([45, 12, 45, 78, 23]),  # 5 historical locations
    'Y': 67,                               # Target: next location is 67
    'user_X': np.array([3, 3, 3, 3, 3]),  # User ID = 3
    'weekday_X': np.array([1, 1, 2, 2, 3]),  # Tue, Tue, Wed, Wed, Thu
    'start_min_X': np.array([510, 720, 480, 1020, 540]),  # Start times in minutes
    'dur_X': np.array([120, 45, 180, 90, 60]),  # Durations in minutes
    'diff': np.array([5, 4, 3, 2, 1])    # Days from target
}
```

### 1.2 After Dataset Processing

```python
# In LocationDataset.__getitem__():
x = torch.tensor([45, 12, 45, 78, 23])  # Shape: [5]
y = torch.tensor(67)                     # Shape: scalar

return_dict = {
    'user': torch.tensor(3),                          # Single user ID
    'time': torch.tensor([34, 48, 32, 68, 36]),      # start_min // 15 (15-min slots)
    'weekday': torch.tensor([1, 1, 2, 2, 3]),         # 0=Mon, 1=Tue, etc.
    'duration': torch.tensor([4, 1, 6, 3, 2]),        # dur // 30 (30-min bins)
    'diff': torch.tensor([5, 4, 3, 2, 1])
}
```

**Time Conversion Example:**
- 510 minutes = 8:30 AM → slot 34 (510 // 15 = 34)
- 720 minutes = 12:00 PM → slot 48 (720 // 15 = 48)

### 1.3 After Collate (Batch of 2)

```python
# Batch with 2 samples of lengths 5 and 3
# After pad_sequence():

x_batch = torch.tensor([
    [45, 102],   # Position 0
    [12,  55],   # Position 1
    [45,  89],   # Position 2
    [78,   0],   # Position 3 (sample 2 padded)
    [23,   0]    # Position 4 (sample 2 padded)
])
# Shape: [5, 2] (seq_len=5, batch_size=2)

y_batch = torch.tensor([67, 42])  # Shape: [2]

context_dict = {
    'len': torch.tensor([5, 3]),  # Actual lengths
    'user': torch.tensor([3, 7]),
    'time': torch.tensor([
        [34, 52],
        [48, 61],
        [32, 45],
        [68,  0],   # Padded
        [36,  0]    # Padded
    ]),
    'weekday': torch.tensor([
        [1, 4],
        [1, 4],
        [2, 5],
        [2, 0],
        [3, 0]
    ]),
    'duration': torch.tensor([
        [4, 2],
        [1, 5],
        [6, 3],
        [3, 0],
        [2, 0]
    ])
}
```

---

## 2. PositionalEncoding Walkthrough

### 2.1 Code

```python
class PositionalEncoding(nn.Module):
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        
        # Line 1: Calculate frequency denominators
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        
        # Line 2: Create position indices
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        
        # Line 3: Initialize position embedding matrix
        pos_embedding = torch.zeros((maxlen, emb_size))
        
        # Line 4-5: Fill with sinusoidal values
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        
        # Line 6: Add middle dimension for broadcasting
        pos_embedding = pos_embedding.unsqueeze(-2)
        
        # Line 7: Register as buffer (not a parameter)
        self.register_buffer("pos_embedding", pos_embedding)
        self.dropout = nn.Dropout(dropout)
```

### 2.2 Step-by-Step Example (emb_size=8)

```python
emb_size = 8

# Step 1: Calculate denominators
# For dimensions 0, 2, 4, 6:
den = torch.exp(-torch.arange(0, 8, 2) * math.log(10000) / 8)
# = torch.exp(-[0, 2, 4, 6] * 9.21 / 8)
# = torch.exp([-0, -2.30, -4.61, -6.91])
# = [1.0, 0.1, 0.01, 0.001]

# Step 2: Position indices
pos = torch.arange(0, 5).reshape(5, 1)
# = [[0], [1], [2], [3], [4]]

# Step 3: Compute sin/cos values
# For position 0:
#   sin([0] * [1.0, 0.1, 0.01, 0.001]) = [0, 0, 0, 0]
#   cos([0] * [1.0, 0.1, 0.01, 0.001]) = [1, 1, 1, 1]
#   PE[0] = [0, 1, 0, 1, 0, 1, 0, 1]  (interleaved)

# For position 1:
#   sin([1] * [1.0, 0.1, 0.01, 0.001]) = [0.841, 0.0998, 0.01, 0.001]
#   cos([1] * [1.0, 0.1, 0.01, 0.001]) = [0.540, 0.995, 0.9999, 1.0]
#   PE[1] = [0.841, 0.540, 0.0998, 0.995, 0.01, 0.9999, 0.001, 1.0]

# Position Embedding Matrix (first 5 positions, 8 dims):
pos_embedding = [
    [0.000, 1.000, 0.000, 1.000, 0.000, 1.000, 0.000, 1.000],  # pos 0
    [0.841, 0.540, 0.100, 0.995, 0.010, 1.000, 0.001, 1.000],  # pos 1
    [0.909, -0.416, 0.199, 0.980, 0.020, 1.000, 0.002, 1.000], # pos 2
    [0.141, -0.990, 0.296, 0.955, 0.030, 1.000, 0.003, 1.000], # pos 3
    [-0.757, -0.654, 0.389, 0.921, 0.040, 0.999, 0.004, 1.000] # pos 4
]
```

### 2.3 Forward Pass

```python
def forward(self, token_embedding: Tensor):
    # token_embedding: [seq_len=5, batch=2, emb_dim=8]
    
    # Add positional encoding
    # pos_embedding[:5, :] is [5, 1, 8]
    # Broadcasting: [5, 2, 8] + [5, 1, 8] = [5, 2, 8]
    
    result = token_embedding + self.pos_embedding[:token_embedding.size(0), :]
    
    # Apply dropout
    return self.dropout(result)
    # Output: [5, 2, 8]
```

---

## 3. TemporalEmbedding Walkthrough

### 3.1 Code

```python
class TemporalEmbedding(nn.Module):
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()
        
        self.emb_info = emb_info
        self.minute_size = 4  # 4 quarter-hours per hour
        hour_size = 24
        weekday = 7

        if self.emb_info == "all":
            self.minute_embed = nn.Embedding(4, d_input)     # 4 embeddings
            self.hour_embed = nn.Embedding(24, d_input)      # 24 embeddings
            self.weekday_embed = nn.Embedding(7, d_input)    # 7 embeddings
```

### 3.2 Forward Example

```python
def forward(self, time, weekday):
    # time: [5, 2] - values 0-95 (24*4 slots per day)
    # weekday: [5, 2] - values 0-6
    
    # Example: time[0, 0] = 34
    # Hour: 34 // 4 = 8 (8 AM)
    # Quarter: 34 % 4 = 2 (30-44 minutes)
    
    hour = torch.div(time, self.minute_size, rounding_mode="floor")
    minutes = time % 4
    
    # Lookup embeddings
    minute_x = self.minute_embed(minutes)    # [5, 2, D]
    hour_x = self.hour_embed(hour)           # [5, 2, D]
    weekday_x = self.weekday_embed(weekday)  # [5, 2, D]
    
    # Combine by addition
    return hour_x + minute_x + weekday_x     # [5, 2, D]
```

### 3.3 Detailed Example (D=4)

```python
# Sample time values for first sequence position:
time = torch.tensor([[34, 52]])  # [1, 2]
weekday = torch.tensor([[1, 4]])  # [1, 2]

# Decomposition:
# time=34: hour=8, minute=2 (8:30-8:44)
# time=52: hour=13, minute=0 (13:00-13:14)

# Suppose embeddings are:
hour_embed[8] = [0.1, 0.2, 0.3, 0.4]
minute_embed[2] = [0.01, 0.02, 0.03, 0.04]
weekday_embed[1] = [0.5, 0.5, 0.5, 0.5]  # Tuesday

# For sample 0, position 0:
temporal = [0.1+0.01+0.5, 0.2+0.02+0.5, 0.3+0.03+0.5, 0.4+0.04+0.5]
        = [0.61, 0.72, 0.83, 0.94]
```

---

## 4. AllEmbedding Walkthrough

### 4.1 Initialization

```python
class AllEmbedding(nn.Module):
    def __init__(self, d_input, config, total_loc_num, if_pos_encoder=True, 
                 emb_info="all", emb_type="add"):
        super(AllEmbedding, self).__init__()
        
        self.d_input = d_input  # e.g., 32
        
        # Location embedding: maps location IDs to vectors
        self.emb_loc = nn.Embedding(total_loc_num, d_input)
        # total_loc_num = 1187 for GeoLife
        # Creates 1187 × 32 parameter matrix
        
        # Temporal embedding
        self.temporal_embedding = TemporalEmbedding(d_input, emb_info)
        
        # Duration embedding: max 96 bins (48 hours / 30 min)
        self.emb_duration = nn.Embedding(96, d_input)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_input, dropout=0.1)
```

### 4.2 Forward Pass Detailed

```python
def forward(self, src, context_dict):
    # src: [5, 2] - location IDs
    # src = [[45, 102], [12, 55], [45, 89], [78, 0], [23, 0]]
    
    # Step 1: Location embedding lookup
    emb = self.emb_loc(src)
    # Shape: [5, 2, 32]
    # Each location ID is replaced by its 32-dim embedding vector
    
    # Step 2: Add temporal embedding
    if self.if_include_time:
        temporal = self.temporal_embedding(
            context_dict["time"],     # [5, 2]
            context_dict["weekday"]   # [5, 2]
        )
        # temporal: [5, 2, 32]
        emb = emb + temporal
        # Element-wise addition: [5, 2, 32] + [5, 2, 32] = [5, 2, 32]
    
    # Step 3: Add duration embedding
    if self.if_include_duration:
        duration_emb = self.emb_duration(context_dict["duration"])
        # duration: [5, 2]
        # duration_emb: [5, 2, 32]
        emb = emb + duration_emb
        # [5, 2, 32]
    
    # Step 4: Scale and add positional encoding
    scaled_emb = emb * math.sqrt(self.d_input)
    # Scaling by sqrt(32) ≈ 5.66
    # This prevents positional encoding from dominating
    
    return self.pos_encoder(scaled_emb)
    # Output: [5, 2, 32] with dropout applied
```

### 4.3 Numerical Example

```python
# For a single element (position 0, batch 0):
# src[0,0] = 45 (location ID)

# Step 1: Location embedding
loc_emb = emb_loc(45)
# = [0.23, -0.15, 0.42, ...]  # 32 values (learned)

# Step 2: Temporal embedding (time=34, weekday=1)
temp_emb = temporal_embedding(34, 1)
# = hour_emb[8] + minute_emb[2] + weekday_emb[1]
# = [0.1, 0.2, ...] + [0.01, 0.02, ...] + [0.5, 0.5, ...]
# = [0.61, 0.72, ...]  # 32 values

# Step 3: Duration embedding (duration=4)
dur_emb = emb_duration(4)
# = [0.05, -0.08, 0.12, ...]  # 32 values (learned)

# Combined:
combined = loc_emb + temp_emb + dur_emb
# = [0.23+0.61+0.05, -0.15+0.72-0.08, 0.42+...+0.12, ...]
# = [0.89, 0.49, ...]  # 32 values

# Step 4: Scale
scaled = combined * 5.66  # sqrt(32)
# = [5.03, 2.77, ...]

# Step 5: Add positional encoding for position 0
final = scaled + pos_encoding[0]
# = [5.03+0, 2.77+1, ...]  # Adding PE values
# = [5.03, 3.77, ...]

# Step 6: Dropout (10% probability of zeroing)
# During training, some values might become 0
```

---

## 5. TransformerEncoder Walkthrough

### 5.1 Setup

```python
# In MHSA.__init__():
encoder_layer = torch.nn.TransformerEncoderLayer(
    d_model=32,           # Same as embedding dim
    nhead=8,              # 8 attention heads
    activation="gelu",    # Activation function
    dim_feedforward=128   # FFN hidden size
)
encoder_norm = torch.nn.LayerNorm(32)
self.encoder = torch.nn.TransformerEncoder(
    encoder_layer=encoder_layer,
    num_layers=2,         # 2 encoder layers
    norm=encoder_norm
)
```

### 5.2 Attention Head Calculation

```python
# With d_model=32, nhead=8:
# Each head has d_k = 32 / 8 = 4 dimensions

# For one attention head:
# Input: [5, 2, 4] (4 dims per head, per position, per sample)
# Q, K, V projections: Linear(4 → 4)

Q = input @ W_Q  # [5, 2, 4]
K = input @ W_K  # [5, 2, 4]
V = input @ W_V  # [5, 2, 4]

# Attention scores
scores = Q @ K.transpose(-2, -1)  # [5, 2, 5]
scores = scores / sqrt(4)         # Scale by sqrt(d_k)

# Apply causal mask
# scores + mask where mask[i,j] = -inf if i < j

# Softmax
attn_weights = softmax(scores, dim=-1)  # [5, 2, 5]

# Apply attention to values
output = attn_weights @ V  # [5, 2, 4]
```

### 5.3 Causal Mask Example

```python
def _generate_square_subsequent_mask(self, sz):
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

# For sequence length 5:
mask = [
    [0,    -inf, -inf, -inf, -inf],  # pos 0 can only see pos 0
    [0,    0,    -inf, -inf, -inf],  # pos 1 sees pos 0,1
    [0,    0,    0,    -inf, -inf],  # pos 2 sees pos 0,1,2
    [0,    0,    0,    0,    -inf],  # pos 3 sees pos 0,1,2,3
    [0,    0,    0,    0,    0   ]   # pos 4 sees all
]

# After adding to attention scores and softmax:
# Position 0: [1.0, 0, 0, 0, 0]
# Position 4: [0.1, 0.2, 0.3, 0.2, 0.2]  # can attend to all
```

### 5.4 Padding Mask

```python
# src = [[45, 102], [12, 55], [45, 89], [78, 0], [23, 0]]
# Sample 1 has length 3 (positions 3,4 are padding)

src_padding_mask = (src == 0).transpose(0, 1)
# = [[False, False, False, False, False],  # Sample 0: no padding
#    [False, False, False, True, True]]     # Sample 1: pos 3,4 padded

# This mask is used to ignore padded positions
# True = ignore this position
```

### 5.5 Complete Encoder Layer

```python
def encoder_layer_forward(src, src_mask, src_key_padding_mask):
    # src: [5, 2, 32]
    
    # 1. Multi-head self-attention
    attn_output, _ = self.self_attn(
        src, src, src,
        attn_mask=src_mask,           # Causal mask
        key_padding_mask=src_key_padding_mask  # Padding mask
    )
    # attn_output: [5, 2, 32]
    
    # 2. Residual connection + LayerNorm
    src = src + self.dropout1(attn_output)
    src = self.norm1(src)  # LayerNorm along last dim
    
    # 3. Feed-forward network
    ff_output = self.linear2(self.dropout(self.activation(self.linear1(src))))
    # linear1: [32 → 128]
    # activation: GELU
    # linear2: [128 → 32]
    
    # 4. Residual connection + LayerNorm
    src = src + self.dropout2(ff_output)
    src = self.norm2(src)
    
    return src
    # Output: [5, 2, 32]
```

---

## 6. FullyConnected Walkthrough

### 6.1 Sequence Selection

```python
# In MHSA.forward():

# After encoder: out = [5, 2, 32]
seq_len = context_dict["len"]  # [5, 3]

# We want the last valid position for each sample
# Sample 0: position 4 (index 4)
# Sample 1: position 2 (index 2)

# Using gather operation:
indices = seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1
# indices: [1, 2, 32] with values [[4, 2]] repeated

out = out.gather(0, indices).squeeze(0)
# This selects:
# - From sample 0: out[4, 0, :] (last position)
# - From sample 1: out[2, 1, :] (position 2)
# Result: [2, 32]
```

### 6.2 FullyConnected Forward

```python
class FullyConnected(nn.Module):
    def forward(self, out, user):
        # out: [2, 32]
        # user: [2] = [3, 7]
        
        # Step 1: Add user embedding
        if self.if_embed_user:
            user_emb = self.emb_user(user)  # [2, 32]
            out = out + user_emb            # [2, 32]
        
        # Step 2: Dropout
        out = self.emb_dropout(out)  # [2, 32], 10% dropout
        
        # Step 3: Residual block
        if self.if_residual_layer:
            # Linear: [32 → 64]
            x = self.linear1(out)     # [2, 64]
            x = F.relu(x)             # [2, 64]
            x = self.fc_dropout1(x)   # [2, 64], dropout
            
            # Linear: [64 → 32]
            x = self.linear2(x)       # [2, 32]
            x = self.fc_dropout2(x)   # [2, 32], dropout
            
            # Residual + BatchNorm
            out = out + x             # [2, 32]
            out = self.norm1(out)     # [2, 32], BatchNorm
        
        # Step 4: Final classification
        logits = self.fc_loc(out)     # Linear: [32 → 1187]
        # Output: [2, 1187]
        
        return logits
```

### 6.3 Numerical Example

```python
# For sample 0 (batch index 0):

# After encoder selection:
out = [0.5, -0.3, 0.8, ...]  # 32 values

# User embedding for user 3:
user_emb = [0.1, 0.2, -0.1, ...]  # 32 values

# Combined:
out = [0.6, -0.1, 0.7, ...]  # 32 values

# After residual block (simplified):
out = [0.55, -0.08, 0.65, ...]  # 32 values

# Final linear projection to 1187 locations:
logits = out @ W + b  # [1187] values

# Example logits:
logits = [-2.1, 0.5, 1.2, ..., 3.5, ...]
#         loc0  loc1  loc2      loc67

# After softmax for probabilities:
probs = softmax(logits)
# Location 67 (target) might have prob 0.15
# Top prediction might be location 45 with prob 0.18
```

---

## 7. Complete Forward Pass

### 7.1 Full Example

```python
# Input batch:
src = torch.tensor([[45, 102], [12, 55], [45, 89], [78, 0], [23, 0]])
# Shape: [5, 2]

context_dict = {
    'len': torch.tensor([5, 3]),
    'user': torch.tensor([3, 7]),
    'time': torch.tensor([[34, 52], [48, 61], [32, 45], [68, 0], [36, 0]]),
    'weekday': torch.tensor([[1, 4], [1, 4], [2, 5], [2, 0], [3, 0]]),
    'duration': torch.tensor([[4, 2], [1, 5], [6, 3], [3, 0], [2, 0]])
}

# Forward pass:
def forward(self, src, context_dict, device):
    # Step 1: Embedding layer
    emb = self.Embedding(src, context_dict)
    # Shape: [5, 2, 32]
    # Combines location + time + weekday + duration + positional
    
    seq_len = context_dict["len"]  # [5, 3]
    
    # Step 2: Create masks
    src_mask = self._generate_square_subsequent_mask(5)
    # [5, 5] causal mask
    
    src_padding_mask = (src == 0).transpose(0, 1)
    # [[F,F,F,F,F], [F,F,F,T,T]]
    
    # Step 3: Transformer encoder
    out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)
    # Shape: [5, 2, 32]
    
    # Step 4: Select last valid position
    out = out.gather(0, seq_len.view([1,-1,1]).expand([1,2,32])-1).squeeze(0)
    # Shape: [2, 32]
    
    # Step 5: Output layer
    return self.FC(out, context_dict["user"])
    # Shape: [2, 1187]
```

### 7.2 Data Shape Summary

| Step | Shape | Description |
|------|-------|-------------|
| Input locations | [5, 2] | seq_len × batch |
| After loc_embed | [5, 2, 32] | Add embedding dim |
| After temporal | [5, 2, 32] | Same (addition) |
| After duration | [5, 2, 32] | Same (addition) |
| After pos_enc | [5, 2, 32] | Same (addition) |
| After encoder | [5, 2, 32] | Transformed |
| After selection | [2, 32] | One vector per sample |
| After user_emb | [2, 32] | Same (addition) |
| After residual | [2, 32] | Same |
| Final logits | [2, 1187] | Per-location scores |

---

## 8. Training Step Walkthrough

### 8.1 One Training Step

```python
def train_step(model, batch, optimizer, device):
    # Get batch
    x, y, x_dict = batch
    # x: [5, 2], y: [2], x_dict: context
    
    # Move to device
    x = x.to(device)
    y = y.to(device)
    for key in x_dict:
        x_dict[key] = x_dict[key].to(device)
    
    # Forward pass
    logits = model(x, x_dict, device)
    # logits: [2, 1187]
    
    # Calculate loss
    CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    
    # Reshape for loss calculation
    logits_flat = logits.view(-1, 1187)  # [2, 1187]
    y_flat = y.reshape(-1)               # [2]
    
    loss = CEL(logits_flat, y_flat)
    # Cross entropy between predictions and targets
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()  # Compute gradients
    
    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    
    # Update parameters
    optimizer.step()
    
    return loss.item()
```

### 8.2 Loss Calculation Example

```python
# logits for sample 0: [2, 1187]
# y = [67, 42] (targets)

# For sample 0:
logits_0 = [-2.1, 0.5, 1.2, ..., 3.5, ...]  # 1187 values
#           loc0  loc1  loc2     loc67

# Softmax:
probs_0 = softmax(logits_0)
# probs_0[67] = 0.15 (probability of correct location)

# Cross entropy:
loss_0 = -log(probs_0[67])
       = -log(0.15)
       = 1.90

# For sample 1:
loss_1 = -log(probs_1[42])

# Average loss:
loss = (loss_0 + loss_1) / 2
```

### 8.3 Metric Calculation

```python
# After forward pass:
# logits: [2, 1187]
# y: [2] = [67, 42]

# Top-k accuracy
k = 1
top1_preds = torch.topk(logits, k=1, dim=-1).indices  # [2, 1]
# top1_preds = [[45], [42]]  # Example predictions

# Check if correct
correct = torch.eq(y[:, None], top1_preds).any(dim=1)
# correct = [False, True]  # Only sample 1 correct

acc_at_1 = correct.sum() / len(correct)  # 0.5 = 50%

# MRR calculation
sorted_indices = torch.argsort(logits, dim=-1, descending=True)
# For sample 0: [45, 23, 67, 12, ...]  (67 is at rank 3)
# For sample 1: [42, 55, 89, ...]       (42 is at rank 1)

ranks = [3, 1]
reciprocal_ranks = [1/3, 1/1] = [0.333, 1.0]
MRR = mean([0.333, 1.0]) = 0.667
```

---

## Appendix: Code Reference

### Key Files

1. **Model Definition:** `src/models/baseline/MHSA.py`
   - PositionalEncoding: lines 24-46
   - TemporalEmbedding: lines 49-93
   - AllEmbedding: lines 154-226
   - FullyConnected: lines 229-272
   - MHSA: lines 275-405

2. **Training Script:** `src/training/train_MHSA.py`
   - LocationDataset: lines 141-177
   - collate_fn: lines 180-206
   - train_epoch: lines 273-326
   - validate: lines 329-377

3. **Metrics:** `src/evaluation/metrics.py`
   - get_mrr: lines 33-68
   - get_ndcg: lines 71-113
   - calculate_correct_total_prediction: lines 116-173

---

*This walkthrough uses emb_size=32 and batch_size=2 for clarity. Production configs may use different values.*
