# Complete Code Reference: Annotated Source Code

## Line-by-Line Explanation of pointer_v45.py

This document provides a complete annotated version of the model implementation, explaining every line of code.

---

## 1. File Header and Imports

```python
"""
Position-Aware Pointer Network V45 - Clean and Lean Version.

This module implements the PointerNetworkV45 model for next location prediction.
The model combines a Transformer encoder with a pointer mechanism and a generation
head, using a learned gate to blend the two prediction strategies.

Architecture:
- Location + User + Temporal embeddings
- Transformer encoder with pre-norm and GELU activation
- Pointer mechanism with position bias
- Generation head with full vocabulary prediction
- Pointer-Generation gate for adaptive blending

Key Features:
- Sinusoidal positional encoding
- Position-from-end embedding for recency awareness
- Temporal features: time of day, weekday, recency, duration
- Mixed precision training support

Usage:
    from src.models.proposed.pointer_v45 import PointerNetworkV45
    
    model = PointerNetworkV45(
        num_locations=1000,
        num_users=100,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.15,
    )
    
    # Forward pass
    # x: [seq_len, batch_size] - location sequence
    # x_dict: dictionary with 'user', 'time', 'weekday', 'diff', 'duration', 'len'
    log_probs = model(x, x_dict)  # [batch_size, num_locations]
"""

import math                    # For sqrt in attention scaling
import torch                   # Core PyTorch
import torch.nn as nn          # Neural network modules
import torch.nn.functional as F # Functional operations (softmax, etc.)
```

**Explanation:**
- The docstring provides a complete overview of the model
- We import minimal dependencies: just PyTorch core modules
- `math` is only used for `sqrt` in attention scaling

---

## 2. Class Definition and Documentation

```python
class PointerNetworkV45(nn.Module):
    """
    Clean Pointer Network for Next Location Prediction.
    
    This model predicts the next location a user will visit based on their
    location history and temporal context. It uses a hybrid approach combining:
    
    1. Pointer Mechanism: Attends to input sequence and copies from history
    2. Generation Head: Generates prediction over full vocabulary
    3. Adaptive Gate: Learns to blend pointer and generation distributions
    
    Args:
        num_locations (int): Total number of unique locations in vocabulary
        num_users (int): Total number of unique users
        d_model (int): Dimension of model embeddings (default: 128)
        nhead (int): Number of attention heads (default: 4)
        num_layers (int): Number of transformer encoder layers (default: 3)
        dim_feedforward (int): Feedforward network dimension (default: 256)
        dropout (float): Dropout probability (default: 0.15)
        max_seq_len (int): Maximum sequence length (default: 150)
    
    Input:
        x (torch.Tensor): Location sequence tensor of shape [seq_len, batch_size]
        x_dict (dict): Dictionary containing:
            - 'user': User IDs [batch_size]
            - 'time': Time of day in 15-min intervals [seq_len, batch_size]
            - 'weekday': Day of week [seq_len, batch_size]
            - 'diff': Days ago for each visit [seq_len, batch_size]
            - 'duration': Duration in 30-min buckets [seq_len, batch_size]
            - 'len': Sequence lengths [batch_size]
    
    Output:
        torch.Tensor: Log probabilities over locations [batch_size, num_locations]
    """
```

**Explanation:**
- Inherits from `nn.Module` (standard PyTorch pattern)
- Comprehensive docstring documents all arguments, inputs, and outputs
- Input shape: `[seq_len, batch_size]` (PyTorch convention for sequences)
- Output: log probabilities (for numerical stability with cross-entropy)

---

## 3. Constructor (__init__)

```python
    def __init__(
        self,
        num_locations: int,      # Number of unique locations (vocabulary size)
        num_users: int,          # Number of unique users
        d_model: int = 128,      # Model dimension
        nhead: int = 4,          # Number of attention heads
        num_layers: int = 3,     # Number of Transformer layers
        dim_feedforward: int = 256,  # FFN hidden dimension
        dropout: float = 0.15,   # Dropout probability
        max_seq_len: int = 150,  # Maximum sequence length
    ):
        super().__init__()  # Initialize parent nn.Module
        
        # Store hyperparameters for use in forward pass
        self.num_locations = num_locations
        self.d_model = d_model
        self.max_seq_len = max_seq_len
```

**Explanation:**
- Type hints for all parameters
- Default values represent good starting points
- `super().__init__()` is required for PyTorch modules
- Store key values as instance attributes for later use

### 3.1 Core Embeddings

```python
        # Core embeddings
        # Location embedding: maps location ID → d_model dimensional vector
        # padding_idx=0 means location 0 (padding) always maps to zero vector
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # User embedding: maps user ID → d_model dimensional vector
        # Same dimension as location for easy combination
        self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
```

**Explanation:**
- `nn.Embedding(vocab_size, embedding_dim)` creates a lookup table
- `padding_idx=0` ensures padding tokens map to zero vectors
- Both embeddings have same dimension for architectural simplicity

### 3.2 Temporal Embeddings

```python
        # Temporal embeddings (time, weekday, recency, duration)
        # Each has dimension d_model // 4 to balance parameter count
        
        # Time of day: 96 intervals (15-minute slots) + 1 for padding
        # Example: 8:00 AM = minute 480 / 15 = bucket 32
        self.time_emb = nn.Embedding(97, d_model // 4)
        
        # Weekday: 7 days + 1 for padding
        # 0 = padding, 1 = Monday, ..., 7 = Sunday
        self.weekday_emb = nn.Embedding(8, d_model // 4)
        
        # Recency: how many days ago (0-7, plus padding)
        # 0 = today, 1 = yesterday, ..., 7+ = 7 days or more ago
        self.recency_emb = nn.Embedding(9, d_model // 4)
        
        # Duration: 100 buckets of 30 minutes each
        # 0 = 0-29 min, 1 = 30-59 min, ..., 99 = 49.5+ hours
        self.duration_emb = nn.Embedding(100, d_model // 4)
```

**Explanation:**
- Each temporal embedding has `d_model // 4` dimensions
- This keeps total parameters reasonable
- +1 for padding in each (index 0 reserved)
- Bucket sizes chosen to balance granularity and generalization

### 3.3 Position-from-End Embedding

```python
        # Position from end embedding (important for pointer mechanism)
        # Instead of "this is position 5", captures "this is 3 positions from end"
        # Position 0 from end = most recent, higher = older
        self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
```

**Explanation:**
- Captures relative recency instead of absolute position
- `max_seq_len + 1` to handle edge cases
- Same dimension as other temporal features

### 3.4 Input Projection

```python
        # Input projection: combines all embeddings into single d_model vector
        # Input: loc (d_model) + user (d_model) + 5 temporal (d_model//4 each)
        # Total: d_model * 2 + d_model // 4 * 5
        input_dim = d_model * 2 + d_model // 4 * 5
        
        # Linear layer to project concatenated features to d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        
        # Layer normalization for stable training
        self.input_norm = nn.LayerNorm(d_model)
```

**Explanation:**
- Concatenated embeddings have different total dimension than `d_model`
- Linear projection learns optimal combination
- LayerNorm stabilizes input to Transformer

### 3.5 Positional Encoding

```python
        # Sinusoidal positional encoding (not learned, just computed once)
        # register_buffer: saves with model but not as parameter
        self.register_buffer('pos_encoding', self._create_pos_encoding(max_seq_len, d_model))
```

**Explanation:**
- `register_buffer` stores tensor with model state but excludes from gradient computation
- Sinusoidal encoding is fixed (not learned)
- Created once at initialization

### 3.6 Transformer Encoder

```python
        # Transformer encoder with pre-norm and GELU activation
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,           # Input/output dimension
            nhead=nhead,               # Number of attention heads
            dim_feedforward=dim_feedforward,  # FFN hidden dimension
            dropout=dropout,           # Dropout probability
            activation='gelu',         # GELU activation (smoother than ReLU)
            batch_first=True,          # Input shape: [batch, seq, feature]
            norm_first=True            # Pre-norm: LayerNorm before attention/FFN
        )
        # Stack multiple layers
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

**Explanation:**
- Single layer defined first, then stacked
- `batch_first=True` makes shapes more intuitive
- `norm_first=True` enables pre-norm architecture (more stable)
- `activation='gelu'` is smoother than ReLU

### 3.7 Pointer Mechanism Components

```python
        # Pointer mechanism
        # Query projection: context → query vector for attention
        self.pointer_query = nn.Linear(d_model, d_model)
        
        # Key projection: encoded positions → key vectors
        self.pointer_key = nn.Linear(d_model, d_model)
        
        # Learnable position bias: adds bias based on position-from-end
        # Initialized to zeros, learns to prefer recent positions
        self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
```

**Explanation:**
- Query computed from context (last position)
- Keys computed from all encoded positions
- Position bias is `nn.Parameter` so it's learned during training
- Initialized to zeros (no initial preference)

### 3.8 Generation Head and Gate

```python
        # Generation head: predicts over full vocabulary
        # Maps d_model → num_locations (vocabulary size)
        self.gen_head = nn.Linear(d_model, num_locations)
        
        # Pointer-Generation gate: decides how much to trust pointer vs generator
        # MLP: d_model → d_model//2 → 1 → sigmoid
        self.ptr_gen_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # Reduce dimension
            nn.GELU(),                          # Non-linearity
            nn.Linear(d_model // 2, 1),         # Single output
            nn.Sigmoid()                        # Squash to [0, 1]
        )
```

**Explanation:**
- Generation head is simple linear layer with softmax applied later
- Gate is MLP with dimension reduction
- Sigmoid ensures gate value is between 0 and 1

### 3.9 Weight Initialization

```python
        # Initialize weights
        self._init_weights()
```

---

## 4. Positional Encoding Method

```python
    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        
        Uses the standard positional encoding formula from "Attention is All You Need":
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        
        Args:
            max_len: Maximum sequence length
            d_model: Model dimension
            
        Returns:
            Positional encoding tensor [1, max_len, d_model]
        """
        # Create empty tensor
        pe = torch.zeros(max_len, d_model)
        
        # Position indices: 0, 1, 2, ..., max_len-1
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Compute the divisor term: 10000^(2i/d_model) for i = 0, 2, 4, ...
        # Using exp(log(...)) for numerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sin to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cos to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add batch dimension: [max_len, d_model] → [1, max_len, d_model]
        return pe.unsqueeze(0)
```

**Explanation:**
- Standard sinusoidal encoding from the original Transformer paper
- Even dimensions use sine, odd dimensions use cosine
- `div_term` creates different frequencies for different dimensions
- Lower dimensions have faster oscillations

---

## 5. Weight Initialization Method

```python
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:  # Only for matrices (not biases)
                nn.init.xavier_uniform_(p)
```

**Explanation:**
- Xavier initialization scales weights based on fan-in/fan-out
- Only applied to matrices (dim > 1), not biases
- Helps with gradient flow at initialization

---

## 6. Forward Method

### 6.1 Input Processing

```python
    def forward(self, x: torch.Tensor, x_dict: dict) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Location sequence [seq_len, batch_size]
            x_dict: Dictionary with 'user', 'time', 'weekday', 'diff', 'duration', 'len'
            
        Returns:
            Log probabilities [batch_size, num_locations]
        """
        # Transpose to batch-first: [seq_len, batch] → [batch, seq_len]
        x = x.T
        batch_size, seq_len = x.shape
        device = x.device
        lengths = x_dict['len']  # Actual sequence lengths (before padding)
```

**Explanation:**
- Input comes as `[seq_len, batch]` (PyTorch convention)
- Transpose to `[batch, seq_len]` for batch_first Transformer
- Store device for creating new tensors on same device

### 6.2 Embedding Computation

```python
        # Embeddings
        # Location embedding: [batch, seq_len] → [batch, seq_len, d_model]
        loc_emb = self.loc_emb(x)
        
        # User embedding: [batch] → [batch, d_model] → expand → [batch, seq_len, d_model]
        user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
```

**Explanation:**
- Location embedding is straightforward lookup
- User embedding needs expansion to match sequence length
- `unsqueeze(1)` adds seq dimension, `expand` broadcasts

### 6.3 Temporal Feature Processing

```python
        # Temporal features (clamped to valid ranges)
        # Clamp prevents index-out-of-bounds errors
        
        # Time: 0-96 (96 = 24 hours * 4 intervals/hour)
        time = torch.clamp(x_dict['time'].T, 0, 96)
        
        # Weekday: 0-7 (0 = padding, 1-7 = Mon-Sun)
        weekday = torch.clamp(x_dict['weekday'].T, 0, 7)
        
        # Recency: 0-8 (0 = today, 1-7 = days ago, 8 = older)
        recency = torch.clamp(x_dict['diff'].T, 0, 8)
        
        # Duration: 0-99 (30-minute buckets)
        duration = torch.clamp(x_dict['duration'].T, 0, 99)
        
        # Get embeddings and concatenate
        temporal = torch.cat([
            self.time_emb(time),
            self.weekday_emb(weekday),
            self.recency_emb(recency),
            self.duration_emb(duration)
        ], dim=-1)  # Concatenate along feature dimension
```

**Explanation:**
- `.T` transposes from `[seq, batch]` to `[batch, seq]`
- `clamp` ensures values are in valid range for embedding lookup
- All temporal embeddings concatenated: `[batch, seq, 4 * d_model//4]`

### 6.4 Position-from-End Computation

```python
        # Position from end
        # positions: [0, 1, 2, ..., seq_len-1] for each batch
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        
        # pos_from_end = length - position
        # Example: length=5, positions=[0,1,2,3,4] → pos_from_end=[5,4,3,2,1]
        pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
        
        # Get position-from-end embeddings
        pos_emb = self.pos_from_end_emb(pos_from_end)
```

**Explanation:**
- `positions` is [0, 1, 2, ...] repeated for each batch item
- `lengths.unsqueeze(1) - positions` gives distance from end
- Clamp handles edge cases and padding

### 6.5 Feature Combination

```python
        # Combine features
        # Concatenate: loc + user + temporal + pos_from_end
        combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
        
        # Project to d_model and normalize
        hidden = self.input_norm(self.input_proj(combined))
        
        # Add sinusoidal positional encoding
        hidden = hidden + self.pos_encoding[:, :seq_len, :]
```

**Explanation:**
- All features concatenated into single tensor
- Linear projection reduces to `d_model`
- LayerNorm stabilizes
- Add sinusoidal PE for absolute position info

### 6.6 Transformer Encoding

```python
        # Transformer encoding
        # Create padding mask: True for padding positions
        mask = positions >= lengths.unsqueeze(1)
        
        # Encode with Transformer
        encoded = self.transformer(hidden, src_key_padding_mask=mask)
```

**Explanation:**
- Mask is `True` where `position >= length` (padding)
- Transformer ignores masked positions in attention
- Output has same shape as input: `[batch, seq, d_model]`

### 6.7 Context Extraction

```python
        # Extract context from last valid position
        # batch_idx: [0, 1, 2, ..., batch_size-1]
        batch_idx = torch.arange(batch_size, device=device)
        
        # last_idx: index of last valid position for each batch item
        last_idx = (lengths - 1).clamp(min=0)
        
        # Index to get context: encoded[batch_i, last_idx_i]
        context = encoded[batch_idx, last_idx]
```

**Explanation:**
- We predict from the last valid position
- `lengths - 1` gives last valid index (0-indexed)
- Advanced indexing selects one position per batch item

### 6.8 Pointer Mechanism

```python
        # Pointer attention
        # Query from context: [batch, d_model] → [batch, 1, d_model]
        query = self.pointer_query(context).unsqueeze(1)
        
        # Keys from encoded sequence: [batch, seq, d_model]
        keys = self.pointer_key(encoded)
        
        # Compute attention scores: [batch, 1, d_model] × [batch, d_model, seq] → [batch, 1, seq]
        # Then squeeze to [batch, seq]
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
        
        # Add position bias: bias recent positions higher
        ptr_scores = ptr_scores + self.position_bias[pos_from_end]
        
        # Mask padding positions: set to -inf so softmax gives 0
        ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
        
        # Softmax to get attention weights
        ptr_probs = F.softmax(ptr_scores, dim=-1)
```

**Explanation:**
- Query computed from context (what we're looking for)
- Keys computed from all positions (what we have)
- Dot product gives similarity scores
- Scale by sqrt(d_model) for stable gradients
- Position bias favors recent positions
- Mask ensures padding gets zero probability

### 6.9 Scatter to Vocabulary

```python
        # Scatter pointer probabilities to location vocabulary
        # Initialize with zeros
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
        
        # Accumulate: for each position i, add ptr_probs[i] to ptr_dist[x[i]]
        ptr_dist.scatter_add_(1, x, ptr_probs)
```

**Explanation:**
- `scatter_add_` accumulates probabilities to location indices
- If location appears multiple times, probabilities sum
- Result: distribution over vocabulary based on what's in sequence

### 6.10 Generation Distribution

```python
        # Generation distribution
        # Linear layer + softmax
        gen_probs = F.softmax(self.gen_head(context), dim=-1)
```

**Explanation:**
- Simple linear projection from context to vocabulary
- Softmax converts to probability distribution
- Can predict any location, not just those in sequence

### 6.11 Gate and Final Combination

```python
        # Gate and combine
        # Compute gate value: [batch, 1]
        gate = self.ptr_gen_gate(context)
        
        # Weighted combination
        # gate * pointer_dist + (1 - gate) * generation_dist
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs
        
        # Return log probabilities (for numerical stability with cross-entropy)
        # Add small epsilon to prevent log(0)
        return torch.log(final_probs + 1e-10)
```

**Explanation:**
- Gate computed from context: scalar per batch item
- Final distribution is weighted average
- Log transform for numerical stability
- `1e-10` prevents log(0) = -inf

---

## 7. Parameter Counting Method

```python
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

**Explanation:**
- Iterates over all parameters
- `numel()` gives number of elements
- Only counts trainable parameters (`requires_grad=True`)

---

## Summary: Code Statistics

| Component | Lines | Parameters (d=128, V=7000, U=100) |
|-----------|-------|----------------------------------|
| Imports | 4 | 0 |
| Docstrings | ~50 | 0 |
| `__init__` | ~40 | ~2.3M total |
| `_create_pos_encoding` | 15 | 0 (buffer) |
| `_init_weights` | 4 | 0 (no new params) |
| `forward` | ~50 | 0 (uses existing) |
| `count_parameters` | 2 | 0 |
| **Total** | ~250 | ~2.3M |

---

*This document is part of the comprehensive Pointer V45 documentation series.*
