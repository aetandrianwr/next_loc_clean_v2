# LSTM Model: Line-by-Line Code Walkthrough with Examples

## Table of Contents

1. [Introduction](#1-introduction)
2. [Model Architecture Walkthrough](#2-model-architecture-walkthrough)
3. [Training Script Walkthrough](#3-training-script-walkthrough)
4. [Evaluation Metrics Walkthrough](#4-evaluation-metrics-walkthrough)
5. [Complete Forward Pass Example](#5-complete-forward-pass-example)

---

## 1. Introduction

This document provides a **line-by-line walkthrough** of the LSTM model code with concrete numerical examples. Each code section is explained with actual tensor values to help understand exactly what happens at each step.

### 1.1 Running Example Setup

Throughout this document, we'll use this example:

```
Dataset: GeoLife
Batch size: 4
Sequence lengths: [3, 5, 4, 2]  (variable length sequences)
Number of locations: 1187
Number of users: 46
Embedding dimension (d_input): 32
LSTM hidden size: 128
LSTM layers: 2
```

---

## 2. Model Architecture Walkthrough

### 2.1 LSTM.py - File Header and Imports

```python
"""
LSTM Model for Next Location Prediction.
...
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math
```

**Explanation:**
- `torch.nn`: Neural network building blocks
- `torch.nn.functional as F`: Functional operations (ReLU, etc.)
- `pack_padded_sequence, pad_packed_sequence`: Handle variable-length sequences efficiently

---

### 2.2 TemporalEmbedding Class

```python
class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time-related features.
    
    Supports three modes:
    - "all": Separate embeddings for minute (quarter-hour), hour, and weekday
    - "time": Single embedding for time slots
    - "weekday": Only weekday embedding
    """
```

#### 2.2.1 Constructor

```python
def __init__(self, d_input, emb_info="all"):
    super(TemporalEmbedding, self).__init__()

    self.emb_info = emb_info
    # quarter of an hour
    self.minute_size = 4      # 0, 15, 30, 45 minutes -> 4 options
    hour_size = 24            # 0-23 hours
    weekday = 7               # Mon-Sun

    if self.emb_info == "all":
        self.minute_embed = nn.Embedding(self.minute_size, d_input)  # 4 x 32
        self.hour_embed = nn.Embedding(hour_size, d_input)           # 24 x 32
        self.weekday_embed = nn.Embedding(weekday, d_input)          # 7 x 32
```

**Example:**
```
d_input = 32

Created embeddings:
- minute_embed: Lookup table with 4 rows, each row is 32-dimensional vector
- hour_embed: Lookup table with 24 rows, each row is 32-dimensional vector
- weekday_embed: Lookup table with 7 rows, each row is 32-dimensional vector

Total parameters: (4 + 24 + 7) × 32 = 1,120 parameters
```

#### 2.2.2 Forward Method

```python
def forward(self, time, weekday):
    if self.emb_info == "all":
        # time is in 15-minute slots: 0-95 (96 slots per day)
        # Convert to hour and minute components
        hour = torch.div(time, self.minute_size, rounding_mode="floor")  # time // 4
        minutes = time % 4

        minute_x = self.minute_embed(minutes)
        hour_x = self.hour_embed(hour)
        weekday_x = self.weekday_embed(weekday)

        return hour_x + minute_x + weekday_x
```

**Step-by-Step Example:**

```
Input:
  time = tensor([[36, 48, 52],      # Sequence 1: 3 timesteps
                 [32, 36, 40, 44, 48]])  # Sequence 2: 5 timesteps (simplified to 2D for example)
  
  weekday = tensor([[2, 2, 2],      # All Wednesday
                    [3, 3, 3, 3, 3]])  # All Thursday

Step 1: Extract hour and minute
  hour = time // 4
       = tensor([[9, 12, 13],       # 9am, 12pm, 1pm
                 [8, 9, 10, 11, 12]])  # 8am, 9am, 10am, 11am, 12pm
  
  minutes = time % 4
          = tensor([[0, 0, 0],       # All on the hour (:00)
                    [0, 0, 0, 0, 0]])

Step 2: Look up embeddings
  hour_x = hour_embed([9, 12, 13]) 
         = tensor([[0.12, -0.34, ...],  # 32-dim vector for hour 9
                   [0.23, -0.11, ...],  # 32-dim vector for hour 12
                   [0.18, -0.22, ...]])  # 32-dim vector for hour 13
  
  minute_x = minute_embed([0, 0, 0])
           = tensor([[0.05, 0.11, ...],  # Same vector repeated (all :00)
                     [0.05, 0.11, ...],
                     [0.05, 0.11, ...]])
  
  weekday_x = weekday_embed([2, 2, 2])
            = tensor([[0.08, -0.15, ...],  # Wednesday embedding repeated
                      [0.08, -0.15, ...],
                      [0.08, -0.15, ...]])

Step 3: Combine (element-wise addition)
  output = hour_x + minute_x + weekday_x
         = tensor([[0.25, -0.38, ...],  # Combined time representation
                   [0.36, -0.15, ...],
                   [0.31, -0.26, ...]])
  
  Shape: [seq_len, batch, d_input] = [3, batch, 32] (after proper batching)
```

---

### 2.3 AllEmbeddingLSTM Class

```python
class AllEmbeddingLSTM(nn.Module):
    """
    Combined embedding layer for all input features (LSTM version without positional encoding).
    """
```

#### 2.3.1 Constructor

```python
def __init__(self, d_input, config, total_loc_num, emb_info="all", emb_type="add"):
    super(AllEmbeddingLSTM, self).__init__()
    self.d_input = d_input
    self.emb_type = emb_type

    # location embedding
    if self.emb_type == "add":
        self.emb_loc = nn.Embedding(total_loc_num, d_input)
```

**Example:**
```
total_loc_num = 1187
d_input = 32

emb_loc: Embedding(1187, 32)
  - 1187 locations, each represented as 32-dimensional vector
  - Parameters: 1187 × 32 = 37,984
  
Lookup example:
  Location ID 42 → [0.12, -0.45, 0.23, ..., 0.08]  (32 values)
  Location ID 43 → [0.14, -0.41, 0.25, ..., 0.06]  (similar if nearby)
  Location ID 500 → [-0.33, 0.67, -0.12, ..., 0.44]  (different area)
```

```python
    # time embedding
    self.if_include_time = config.if_embed_time  # True
    if self.if_include_time:
        if self.emb_type == "add":
            self.temporal_embedding = TemporalEmbedding(d_input, emb_info)
```

```python
    # duration embedding (in minutes, max 2 days)
    self.if_include_duration = config.if_embed_duration  # True
    if self.if_include_duration:
        self.emb_duration = nn.Embedding(60 * 24 * 2 // 30, d_input)
        # 60*24*2 = 2880 minutes in 2 days
        # 2880 // 30 = 96 duration buckets (30-minute granularity)
```

**Duration Buckets Example:**
```
Duration encoding:
  0-29 min  → bucket 0
  30-59 min → bucket 1
  60-89 min → bucket 2
  ...
  2850-2879 min → bucket 95

emb_duration: Embedding(96, 32)
  Parameters: 96 × 32 = 3,072
```

```python
    # Dropout (no positional encoding for LSTM)
    self.dropout = nn.Dropout(0.1)
```

#### 2.3.2 Forward Method

```python
def forward(self, src, context_dict) -> Tensor:
    emb = self.emb_loc(src)
```

**Step 1: Location Embedding**
```
Input:
  src = tensor([[102, 45, 103],     # Sequence 1: 3 locations (padded)
                [200, 45, 89, 102, 45],  # Sequence 2: 5 locations
                [33, 77, 102, 45],       # Sequence 3: 4 locations
                [500, 102, 0, 0, 0]])    # Sequence 4: 2 locations + padding
  
  Shape: [5, 4] (max_len=5, batch=4) - actually transposed: [seq_len, batch]

After emb_loc(src):
  emb = tensor([
      [[0.12, -0.34, ..., 0.21],   # Embedding for location 102
       [0.45, -0.12, ..., 0.33],   # Embedding for location 200
       [0.08, -0.55, ..., 0.11],   # Embedding for location 33
       [-0.21, 0.44, ..., -0.08]], # Embedding for location 500
      
      [[0.23, -0.11, ..., 0.18],   # Embedding for location 45
       [0.23, -0.11, ..., 0.18],   # Embedding for location 45
       [0.32, -0.22, ..., 0.27],   # Embedding for location 77
       [0.12, -0.34, ..., 0.21]],  # Embedding for location 102
      
      # ... more rows
  ])
  
  Shape: [5, 4, 32] (seq_len, batch, d_input)
```

```python
    if self.if_include_time:
        if self.emb_type == "add":
            emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday"])
```

**Step 2: Add Temporal Embedding**
```
Inputs from context_dict:
  time = tensor([[36, 48, 52, 0, 0],    # Time slots (0 is padding)
                 [32, 36, 40, 44, 48],
                 [28, 32, 36, 40, 0],
                 [60, 64, 0, 0, 0]])
  
  weekday = tensor([[2, 2, 2, 0, 0],
                    [3, 3, 3, 3, 3],
                    [1, 1, 1, 1, 0],
                    [4, 4, 0, 0, 0]])

temporal_emb = temporal_embedding(time, weekday)
  Shape: [5, 4, 32]

emb = emb + temporal_emb  (element-wise addition)
  Shape: still [5, 4, 32]
  
  Each location embedding is now "enriched" with time information
```

```python
    if self.if_include_duration:
        emb = emb + self.emb_duration(context_dict["duration"])
```

**Step 3: Add Duration Embedding**
```
duration = tensor([[8, 2, 3, 0, 0],    # Duration buckets
                   [4, 8, 2, 1, 2],
                   [6, 4, 8, 2, 0],
                   [10, 4, 0, 0, 0]])

duration_emb = emb_duration(duration)
  Shape: [5, 4, 32]

emb = emb + duration_emb
  Shape: [5, 4, 32]
```

```python
    return self.dropout(emb)
```

**Step 4: Apply Dropout**
```
During training (dropout rate 0.1):
  - Randomly zero out 10% of values
  - Scale remaining values by 1/(1-0.1) = 1.111
  
Example:
  Before: [0.35, -0.47, 0.28, 0.15, ...]
  After:  [0.39, 0.00, 0.31, 0.17, ...]  (second value dropped, others scaled)
  
During inference:
  - Dropout is disabled
  - Values pass through unchanged
```

---

### 2.4 FullyConnected Class (Output Layer)

#### 2.4.1 Constructor

```python
class FullyConnected(nn.Module):
    def __init__(self, d_input, config, total_loc_num, if_residual_layer=True):
        super(FullyConnected, self).__init__()

        fc_dim = d_input  # 128 (hidden_size)
        self.if_embed_user = config.if_embed_user  # True
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, fc_dim)
            # Embedding(46, 128) - 46 users, 128-dim vectors
```

**User Embedding:**
```
46 users, each with a 128-dimensional "preference vector"

Example:
  User 1 (commuter) → [0.5, -0.2, ..., 0.1]  (might have high values for work-related dims)
  User 2 (student)  → [-0.3, 0.4, ..., 0.2]  (different pattern)
  
Parameters: 46 × 128 = 5,888
```

```python
        self.fc_loc = nn.Linear(fc_dim, total_loc_num)
        # Linear(128, 1187) - maps hidden state to location scores
```

**Output Linear Layer:**
```
Input: 128-dimensional hidden state
Output: 1187 scores (one per location)

Parameters: 128 × 1187 + 1187 (bias) = 152,123
```

```python
        self.if_residual_layer = if_residual_layer  # True
        if self.if_residual_layer:
            self.linear1 = nn.Linear(fc_dim, fc_dim * 2)  # 128 → 256
            self.linear2 = nn.Linear(fc_dim * 2, fc_dim)  # 256 → 128

            self.norm1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout1 = nn.Dropout(p=config.fc_dropout)  # 0.2
            self.fc_dropout2 = nn.Dropout(p=config.fc_dropout)
```

**Residual Block Parameters:**
```
linear1: 128 × 256 + 256 = 33,024
linear2: 256 × 128 + 128 = 32,896
norm1: 128 × 2 = 256 (scale and shift)

Total residual block: ~66,000 parameters
```

#### 2.4.2 Forward Method

```python
def forward(self, out, user) -> Tensor:
    if self.if_embed_user:
        out = out + self.emb_user(user)
```

**Step 1: Add User Embedding**
```
Input:
  out = tensor([[0.23, -0.45, ..., 0.12],   # LSTM output for sample 1
                [0.34, -0.22, ..., 0.08],   # LSTM output for sample 2
                [0.11, -0.67, ..., 0.33],   # LSTM output for sample 3
                [0.45, -0.11, ..., 0.21]])  # LSTM output for sample 4
  Shape: [4, 128]
  
  user = tensor([5, 12, 5, 23])  # User IDs for each sample

user_emb = emb_user(user)
  = tensor([[0.08, -0.12, ..., 0.05],   # User 5's preference vector
            [0.15, -0.33, ..., 0.11],   # User 12's preference vector
            [0.08, -0.12, ..., 0.05],   # User 5's preference vector (same as sample 1)
            [0.22, -0.08, ..., 0.18]])  # User 23's preference vector
  Shape: [4, 128]

out = out + user_emb
  = tensor([[0.31, -0.57, ..., 0.17],   # Combined representation
            [0.49, -0.55, ..., 0.19],
            [0.19, -0.79, ..., 0.38],
            [0.67, -0.19, ..., 0.39]])
  Shape: [4, 128]
```

```python
    out = self.emb_dropout(out)

    if self.if_residual_layer:
        out = self.norm1(out + self._res_block(out))
```

**Step 2: Residual Block**
```python
def _res_block(self, x: Tensor) -> Tensor:
    x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
    return self.fc_dropout2(x)
```

```
Input x: [4, 128]

Step 2a: linear1(x)
  x = Wx + b  (128 → 256)
  Shape: [4, 256]
  
Step 2b: ReLU
  x = max(0, x)  (element-wise)
  Shape: [4, 256]
  
Step 2c: Dropout
  Randomly zero 20% of values
  Shape: [4, 256]
  
Step 2d: linear2(x)
  x = Wx + b  (256 → 128)
  Shape: [4, 128]
  
Step 2e: Dropout
  Shape: [4, 128]

Residual connection:
  out = original_out + _res_block_output
  Shape: [4, 128]

BatchNorm:
  out = (out - mean) / std * gamma + beta
  Shape: [4, 128]
```

```python
    return self.fc_loc(out)
```

**Step 3: Final Projection to Location Scores**
```
out = fc_loc(out)  # Linear(128, 1187)
  
Shape: [4, 1187]

Example output (logits):
  tensor([[2.3, -1.2, 0.5, ..., -0.8],   # Scores for all 1187 locations (sample 1)
          [1.1, 0.3, -0.2, ..., 1.5],   # Sample 2
          [-0.5, 2.1, 0.8, ..., 0.2],   # Sample 3
          [0.8, -0.3, 1.9, ..., -1.1]])  # Sample 4

Interpretation:
  - Higher score = model thinks this location is more likely
  - Sample 1: Location 0 has highest score (2.3)
  - Sample 3: Location 1 has highest score (2.1)
```

---

### 2.5 LSTMModel Class (Main Model)

#### 2.5.1 Constructor

```python
class LSTMModel(nn.Module):
    def __init__(self, config, total_loc_num) -> None:
        super(LSTMModel, self).__init__()

        self.d_input = config.base_emb_size    # 32
        self.hidden_size = config.lstm_hidden_size  # 128
        self.num_layers = config.lstm_num_layers    # 2
        
        # Embedding layer
        self.Embedding = AllEmbeddingLSTM(self.d_input, config, total_loc_num)
```

```python
        # LSTM encoder
        lstm_dropout = config.lstm_dropout if self.num_layers > 1 else 0  # 0.2
        self.lstm = nn.LSTM(
            input_size=self.d_input,      # 32
            hidden_size=self.hidden_size,  # 128
            num_layers=self.num_layers,    # 2
            dropout=lstm_dropout,          # 0.2
            batch_first=False,             # [seq_len, batch, features]
            bidirectional=False            # Unidirectional
        )
```

**LSTM Parameter Count:**
```
For single LSTM layer:
  Input gates: 4 × (input_size × hidden_size + hidden_size × hidden_size + hidden_size)
  
Layer 1 (input_size=32, hidden_size=128):
  Weight_ih: 4 × 128 × 32 = 16,384
  Weight_hh: 4 × 128 × 128 = 65,536
  Bias_ih: 4 × 128 = 512
  Bias_hh: 4 × 128 = 512
  Total Layer 1: 82,944

Layer 2 (input_size=128, hidden_size=128):
  Weight_ih: 4 × 128 × 128 = 65,536
  Weight_hh: 4 × 128 × 128 = 65,536
  Bias_ih: 4 × 128 = 512
  Bias_hh: 4 × 128 = 512
  Total Layer 2: 132,096

Total LSTM: 215,040 parameters
```

```python
        # Layer norm after LSTM
        self.layer_norm = nn.LayerNorm(self.hidden_size)  # LayerNorm(128)
        
        # Fully connected output layer
        self.FC = FullyConnected(self.hidden_size, config, if_residual_layer=True, total_loc_num=total_loc_num)

        # init parameters
        self._init_weights()
```

#### 2.5.2 Forward Method - The Core

```python
def forward(self, src, context_dict, device) -> Tensor:
    # Get embeddings
    emb = self.Embedding(src, context_dict)  # [seq_len, batch, d_input]
    seq_len = context_dict["len"]
```

**Step 1: Get Embeddings**
```
Input:
  src: [5, 4] (max_seq_len=5, batch=4)
  context_dict["len"]: tensor([3, 5, 4, 2])  # Actual lengths

After Embedding:
  emb: [5, 4, 32]
```

```python
    # Pack padded sequence for efficient LSTM processing
    packed_emb = pack_padded_sequence(
        emb, 
        seq_len.cpu(), 
        batch_first=False, 
        enforce_sorted=False
    )
```

**Step 2: Pack Sequences**
```
Before packing (with padding):
  Sequence 1 (len=3): [e1, e2, e3, PAD, PAD]
  Sequence 2 (len=5): [e1, e2, e3, e4, e5]
  Sequence 3 (len=4): [e1, e2, e3, e4, PAD]
  Sequence 4 (len=2): [e1, e2, PAD, PAD, PAD]

After packing:
  PackedSequence containing:
    - data: tensor of shape [14, 32] (3+5+4+2=14 actual embeddings)
    - batch_sizes: [4, 4, 3, 2, 1]  # How many sequences at each timestep
    
  Timestep 0: All 4 sequences have data → batch_size=4
  Timestep 1: All 4 sequences have data → batch_size=4
  Timestep 2: Sequences 1,2,3 have data → batch_size=3
  Timestep 3: Sequences 2,3 have data → batch_size=2
  Timestep 4: Only sequence 2 has data → batch_size=1
```

```python
    # Pass through LSTM
    packed_output, (h_n, c_n) = self.lstm(packed_emb)
```

**Step 3: LSTM Processing**
```
LSTM processes the packed sequence efficiently:
  - Doesn't waste computation on padding
  - Outputs are also packed

packed_output: PackedSequence
  - data: [14, 128] (14 outputs, each 128-dim)
  
h_n: [2, 4, 128] (num_layers=2, batch=4, hidden=128)
  - Final hidden states for each layer and each sequence
  
c_n: [2, 4, 128] 
  - Final cell states
```

```python
    # Unpack sequence
    output, _ = pad_packed_sequence(packed_output, batch_first=False)
```

**Step 4: Unpack to Padded Format**
```
output: [5, 4, 128]  # Back to padded format
  
  Sequence 1: [h1, h2, h3, 0, 0]      # Zeros where padding was
  Sequence 2: [h1, h2, h3, h4, h5]    # Full sequence
  Sequence 3: [h1, h2, h3, h4, 0]
  Sequence 4: [h1, h2, 0, 0, 0]
```

```python
    # Get the last valid output for each sequence
    out = output.gather(
        0,
        seq_len.view([1, -1, 1]).expand([1, output.shape[1], output.shape[-1]]).to(device) - 1,
    ).squeeze(0)
```

**Step 5: Extract Last Valid Output**
```
This is the key step - we need the LSTM output at the LAST valid position for each sequence.

seq_len = [3, 5, 4, 2]
seq_len - 1 = [2, 4, 3, 1]  # 0-indexed positions

Visualization:
  output[2, 0, :] → Last output for sequence 1 (position 2, 0-indexed)
  output[4, 1, :] → Last output for sequence 2 (position 4)
  output[3, 2, :] → Last output for sequence 3 (position 3)
  output[1, 3, :] → Last output for sequence 4 (position 1)

Mechanics of gather:
  indices = seq_len.view([1, -1, 1]).expand([1, 4, 128]) - 1
          = tensor([[[2, 2, ..., 2],    # 128 times
                     [4, 4, ..., 4],
                     [3, 3, ..., 3],
                     [1, 1, ..., 1]]])
  Shape: [1, 4, 128]
  
  out = output.gather(0, indices).squeeze(0)
      = tensor([[h3_seq1],     # 128-dim
                [h5_seq2],     # 128-dim
                [h4_seq3],     # 128-dim
                [h2_seq4]])    # 128-dim
  Shape: [4, 128]
```

```python
    # Apply layer normalization
    out = self.layer_norm(out)
```

**Step 6: Layer Normalization**
```
For each sample, normalize across the 128 dimensions:
  out = (out - mean) / sqrt(var + eps) * gamma + beta
  
Example for sample 1:
  Before: [0.5, -1.2, 0.8, ..., 0.3]
  mean = 0.1, var = 0.25
  After:  [0.8, -2.6, 1.4, ..., 0.4] (normalized, then scaled and shifted)
```

```python
    return self.FC(out, context_dict["user"])
```

**Step 7: Output Layer**
```
FC receives:
  out: [4, 128]
  user: tensor([5, 12, 5, 23])

Returns:
  logits: [4, 1187]  # Scores for each location
```

---

### 2.6 Weight Initialization

```python
def _init_weights(self):
    """Initialize parameters."""
    for name, param in self.named_parameters():
        if 'weight_ih' in name:
            # Input-hidden weights: Xavier uniform
            nn.init.xavier_uniform_(param)
        elif 'weight_hh' in name:
            # Hidden-hidden weights: Orthogonal initialization
            nn.init.orthogonal_(param)
        elif 'bias' in name:
            nn.init.zeros_(param)
            # Set forget gate bias to 1
            n = param.size(0)
            param.data[n // 4:n // 2].fill_(1.0)
```

**Initialization Visualization:**
```
LSTM has 4 gates stacked in its weight matrices:
  [input_gate | forget_gate | cell_gate | output_gate]
  
For a hidden_size=128:
  bias shape: [512] = [128, 128, 128, 128]
                       i    f    c    o
  
Forget gate bias = 1:
  bias[128:256] = 1.0
  
Why? Initial forget gate output:
  f = sigmoid(Wf·x + bf)
  If bf = 0: f ≈ 0.5 (forget half)
  If bf = 1: f ≈ 0.73 (keep more) - better for gradient flow initially
```

---

## 3. Training Script Walkthrough

### 3.1 Main Training Function

```python
def train_model(config, model, train_loader, val_loader, device, log_dir, log_file):
    optim = get_optimizer(config, model)

    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,  # 232 * 2 = 464
        num_training_steps=len(train_loader) * config.num_training_epochs,  # 232 * 50 = 11600
    )
```

**Learning Rate Schedule Example (GeoLife):**
```
train_loader length: 232 batches
warmup_epochs: 2
training_epochs: 50

warmup_steps: 232 * 2 = 464
total_steps: 232 * 50 = 11,600

LR progression:
  Step 0:     lr = 0.000000 (start)
  Step 232:   lr = 0.000500 (half warmup)
  Step 464:   lr = 0.001000 (full warmup, peak)
  Step 5000:  lr = 0.000600 (decaying)
  Step 11600: lr = 0.000000 (end)
```

### 3.2 Training Epoch

```python
def train_epoch(config, model, train_loader, optim, device, epoch, scheduler, ...):
    model.train()
    running_loss = 0.0
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    
    CEL = nn.CrossEntropyLoss(reduction="mean", ignore_index=0)

    for i, inputs in enumerate(train_loader):
        x, y, x_dict = send_to_device(inputs, device, config)
```

**Batch Processing Example:**
```
Batch i=0:
  x: [max_len, 32] = [12, 32] (padded sequences)
  y: [32] (target locations)
  x_dict: {
    "len": tensor([5, 8, 3, 12, ...]),  # 32 lengths
    "user": tensor([5, 12, 5, 23, ...]),  # 32 user IDs
    "time": [12, 32],
    "weekday": [12, 32],
    "duration": [12, 32],
    ...
  }
```

```python
        logits = model(x, x_dict, device)
        # logits: [32, 1187]

        loss_size = CEL(logits.view(-1, logits.shape[-1]), y.reshape(-1))
```

**Loss Calculation:**
```
logits: [32, 1187]
y: [32]

Example:
  logits[0] = [2.3, -1.2, 0.5, ..., -0.8]  # 1187 scores
  y[0] = 42  # True location is 42
  
  Softmax(logits[0]) = [0.15, 0.01, 0.02, ..., 0.001]  # Probabilities
  
  CrossEntropy = -log(probability of true class)
               = -log(Softmax(logits[0])[42])
               = -log(0.05)  # If true class had 5% probability
               = 3.0

  Average over batch: mean of 32 CE values
```

```python
        optim.zero_grad(set_to_none=True)
        loss_size.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
```

**Gradient Clipping Example:**
```
Before clipping:
  param1.grad: norm = 2.5
  param2.grad: norm = 0.3
  Total norm: sqrt(2.5² + 0.3²) = 2.52

After clip_grad_norm_(params, max_norm=1.0):
  Scale factor: 1.0 / 2.52 = 0.397
  param1.grad: norm = 2.5 * 0.397 = 0.99
  param2.grad: norm = 0.3 * 0.397 = 0.12
  Total norm: 1.0
```

---

## 4. Evaluation Metrics Walkthrough

### 4.1 calculate_correct_total_prediction

```python
def calculate_correct_total_prediction(logits, true_y):
    top1 = []
    result_ls = []
    
    for k in [1, 3, 5, 10]:
        prediction = torch.topk(logits, k=k, dim=-1).indices
        
        if k == 1:
            top1 = torch.squeeze(prediction).cpu()
        
        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        result_ls.append(top_k)
```

**Example with 4 Samples:**
```
logits: [4, 1187]
true_y: tensor([42, 100, 42, 500])

For k=1:
  prediction = topk(logits, k=1).indices
             = tensor([[150],   # Sample 1: highest score at location 150
                       [100],   # Sample 2: highest score at location 100
                       [42],    # Sample 3: highest score at location 42
                       [300]])  # Sample 4: highest score at location 300
  
  Compare with true_y:
    Sample 1: 42 in [150]? No
    Sample 2: 100 in [100]? Yes
    Sample 3: 42 in [42]? Yes
    Sample 4: 500 in [300]? No
  
  correct@1 = 2

For k=5:
  prediction = topk(logits, k=5).indices
             = tensor([[150, 42, 88, 200, 55],
                       [100, 99, 101, 98, 102],
                       [42, 100, 88, 45, 50],
                       [300, 450, 500, 200, 100]])
  
  Compare:
    Sample 1: 42 in [150, 42, 88, 200, 55]? Yes (position 2)
    Sample 2: 100 in [100, 99, 101, 98, 102]? Yes (position 1)
    Sample 3: 42 in [42, 100, 88, 45, 50]? Yes (position 1)
    Sample 4: 500 in [300, 450, 500, 200, 100]? Yes (position 3)
  
  correct@5 = 4
```

### 4.2 get_mrr (Mean Reciprocal Rank)

```python
def get_mrr(prediction, targets):
    # Sort predictions in descending order
    index = torch.argsort(prediction, dim=-1, descending=True)
    
    # Find where target appears in sorted predictions
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    
    # Get ranks (1-indexed)
    ranks = (hits[:, -1] + 1).float()
    
    # Reciprocal ranks
    rranks = torch.reciprocal(ranks)
    
    return torch.sum(rranks).cpu().numpy()
```

**Example:**
```
prediction (logits): [4, 1187]
targets: tensor([42, 100, 42, 500])

After argsort (descending), for each sample:
  Sample 1: [150, 42, 88, 200, 55, ...]  → target 42 at index 1 → rank = 2
  Sample 2: [100, 99, 101, ...]          → target 100 at index 0 → rank = 1
  Sample 3: [42, 100, 88, ...]           → target 42 at index 0 → rank = 1
  Sample 4: [300, 450, 500, 200, ...]    → target 500 at index 2 → rank = 3

Reciprocal ranks:
  Sample 1: 1/2 = 0.5
  Sample 2: 1/1 = 1.0
  Sample 3: 1/1 = 1.0
  Sample 4: 1/3 = 0.333

Sum = 0.5 + 1.0 + 1.0 + 0.333 = 2.833
MRR = 2.833 / 4 = 0.708 = 70.8%
```

---

## 5. Complete Forward Pass Example

Let's trace through a complete forward pass with concrete values.

### 5.1 Input Data

```
Batch of 2 sequences:

Sequence 1 (user 5, length 3):
  Locations: [Home(102), Work(45), Gym(103)]
  Times: [9:00am, 12:00pm, 6:00pm]
  Weekdays: [Monday, Monday, Monday]
  Durations: [2hr, 4hr, 1hr]
  Target: Restaurant(89)

Sequence 2 (user 12, length 4):
  Locations: [Home(102), Coffee(200), Work(45), Home(102)]
  Times: [8:00am, 8:30am, 9:00am, 6:00pm]
  Weekdays: [Tuesday, Tuesday, Tuesday, Tuesday]
  Durations: [30min, 15min, 8hr, 2hr]
  Target: Restaurant(89)
```

### 5.2 Tensor Representations

```python
# After padding (max_len = 4)
src = tensor([[102, 102],    # Position 0
              [45, 200],     # Position 1
              [103, 45],     # Position 2
              [0, 102]])     # Position 3 (0 is padding for seq 1)
# Shape: [4, 2]

context_dict = {
    "len": tensor([3, 4]),
    "user": tensor([5, 12]),
    "time": tensor([[36, 32],      # 9am, 8am
                    [48, 34],      # 12pm, 8:30am
                    [72, 36],      # 6pm, 9am
                    [0, 72]]),     # pad, 6pm
    "weekday": tensor([[0, 1],
                       [0, 1],
                       [0, 1],
                       [0, 1]]),
    "duration": tensor([[4, 1],    # 2hr/30min=4, 30min/30min=1
                        [8, 0],    # 4hr/30min=8, 15min/30min=0
                        [2, 16],   # 1hr/30min=2, 8hr/30min=16
                        [0, 4]])
}

y = tensor([89, 89])  # Both predict Restaurant
```

### 5.3 Step-by-Step Forward Pass

```
Step 1: Embedding Layer
─────────────────────────────────────────────────────────

Location Embedding:
  emb_loc(src) → [4, 2, 32]
  Example values (position 0):
    [[-0.12, 0.34, ..., 0.08],   # Embedding for location 102
     [-0.12, 0.34, ..., 0.08]]   # Same location 102

Temporal Embedding:
  hour = time // 4 = [[9, 8], [12, 8], [18, 9], [0, 18]]
  minute = time % 4 = [[0, 0], [0, 2], [0, 0], [0, 0]]
  
  temporal_emb → [4, 2, 32]

Duration Embedding:
  duration_emb → [4, 2, 32]

Combined (with dropout):
  emb = loc_emb + temporal_emb + duration_emb
  emb → [4, 2, 32]


Step 2: Pack Sequences
─────────────────────────────────────────────────────────

Before packing:
  Seq 1: [e0, e1, e2, PAD]  (len=3)
  Seq 2: [e0, e1, e2, e3]   (len=4)

packed_emb.data → [7, 32] (3+4=7 real embeddings)
packed_emb.batch_sizes = [2, 2, 2, 1]


Step 3: LSTM Processing
─────────────────────────────────────────────────────────

Layer 1:
  Process timestep 0: batch_size=2
    Input: [e0_seq1, e0_seq2] → [2, 32]
    Hidden: initialize to zeros → [2, 128]
    Cell: initialize to zeros → [2, 128]
    
    f_t = sigmoid(W_f @ [h, x] + b_f)  # Forget gates
    i_t = sigmoid(W_i @ [h, x] + b_i)  # Input gates
    c̃_t = tanh(W_c @ [h, x] + b_c)    # Candidate
    o_t = sigmoid(W_o @ [h, x] + b_o)  # Output gates
    
    c_1 = f_t * c_0 + i_t * c̃_t       # Update cell
    h_1 = o_t * tanh(c_1)              # Update hidden
    
    Output: h_1 → [2, 128]
  
  Process timestep 1: batch_size=2
    ... similar ...
    Output: h_2 → [2, 128]
  
  Process timestep 2: batch_size=2
    Output: h_3 → [2, 128]
  
  Process timestep 3: batch_size=1 (only seq 2)
    Output: h_4 → [1, 128]

Layer 1 final outputs (packed): [7, 128]

Layer 2:
  Takes Layer 1 outputs as input
  Same process, but input_size=128
  
Layer 2 final outputs (packed): [7, 128]


Step 4: Unpack and Gather Last Output
─────────────────────────────────────────────────────────

Unpack:
  output → [4, 2, 128]
  
  Seq 1: [h1, h2, h3, 0]      (zeros at position 3)
  Seq 2: [h1, h2, h3, h4]

Gather last valid:
  seq_len = [3, 4]
  indices = [2, 3] (0-indexed: 3-1=2, 4-1=3)
  
  out = [output[2, 0, :],     # h3 from seq 1
         output[3, 1, :]]     # h4 from seq 2
  
  out → [2, 128]


Step 5: Layer Normalization
─────────────────────────────────────────────────────────

out = LayerNorm(out)
  For each sample, normalize across 128 dims
  out → [2, 128]


Step 6: Fully Connected Layer
─────────────────────────────────────────────────────────

Add user embedding:
  user_emb = emb_user([5, 12]) → [2, 128]
  out = out + user_emb → [2, 128]

Dropout:
  out = dropout(out) → [2, 128]

Residual block:
  residual = linear2(relu(linear1(out)))  # [2, 128] → [2, 256] → [2, 128]
  out = norm(out + residual) → [2, 128]

Final projection:
  logits = fc_loc(out)  # [2, 128] → [2, 1187]
  
  logits[0]: scores for all 1187 locations (seq 1)
  logits[1]: scores for all 1187 locations (seq 2)


Step 7: Loss Calculation (Training)
─────────────────────────────────────────────────────────

y = [89, 89]  # True targets (Restaurant)

softmax_probs = softmax(logits, dim=-1)  # [2, 1187]

CrossEntropyLoss:
  loss_1 = -log(softmax_probs[0, 89])
  loss_2 = -log(softmax_probs[1, 89])
  loss = (loss_1 + loss_2) / 2

Example values:
  If softmax_probs[0, 89] = 0.05 (5% probability for Restaurant)
  loss_1 = -log(0.05) = 2.996
  
  If softmax_probs[1, 89] = 0.10 (10% probability)
  loss_2 = -log(0.10) = 2.303
  
  loss = (2.996 + 2.303) / 2 = 2.65


Step 8: Prediction (Inference)
─────────────────────────────────────────────────────────

predictions = argmax(logits, dim=-1)  # [2]

Example:
  logits[0] = [..., 2.1, ..., 1.8, ...]  # Location 45 has highest score
  predictions[0] = 45  # Predict Work
  
  logits[1] = [..., 3.2, ..., 2.5, ...]  # Location 89 has highest score
  predictions[1] = 89  # Predict Restaurant (correct!)

Accuracy@1: 1/2 = 50%
```

---

## Summary

This walkthrough has covered:

1. **TemporalEmbedding**: How time is decomposed and embedded
2. **AllEmbeddingLSTM**: Combining all input features
3. **LSTM**: Processing sequences with packed sequences
4. **FullyConnected**: User personalization and output projection
5. **Weight Initialization**: Why forget gate bias = 1
6. **Training Loop**: Gradient clipping and learning rate schedule
7. **Metrics**: How Acc@K and MRR are calculated
8. **Complete Example**: End-to-end forward pass with numbers

The key insights are:
- Variable-length sequences are handled efficiently via packing/unpacking
- Multiple embeddings are combined additively
- LSTM naturally handles sequential dependencies without positional encoding
- User embedding allows personalization
- The residual block helps with gradient flow

---

**Document Version**: 1.0
**Last Updated**: January 2026
