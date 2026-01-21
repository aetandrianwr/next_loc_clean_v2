# Attention Mechanism Comparison

## Table of Contents
1. [Overview](#overview)
2. [Original Bahdanau Attention](#original-bahdanau-attention)
3. [Proposed Scaled Dot-Product Attention](#proposed-scaled-dot-product-attention)
4. [Mathematical Comparison](#mathematical-comparison)
5. [Code Implementation](#code-implementation)
6. [Attention Visualization Example](#attention-visualization-example)
7. [Position Bias Mechanism](#position-bias-mechanism)

---

## Overview

Both models use attention mechanisms, but with different formulations:

| Aspect | Original (Bahdanau) | Proposed (Scaled Dot-Product) |
|--------|---------------------|------------------------------|
| **Type** | Additive attention | Multiplicative attention |
| **Computation** | v^T · tanh(W_h·h + W_s·s + b) | QK^T / √d_k |
| **Parameters** | W_h, W_s, v, b (learned) | W_q, W_k (learned) |
| **Scaling** | None | √d_k |
| **Position Handling** | Coverage mechanism | Position bias |

---

## Original Bahdanau Attention

### Concept

Bahdanau attention (also called "additive attention") computes attention scores using a feed-forward neural network:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         BAHDANAU ATTENTION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given:                                                                      │
│    - Encoder states: H = [h₁, h₂, ..., hₙ] ∈ ℝ^(n × 512)                   │
│    - Decoder state:  s ∈ ℝ^256                                              │
│                                                                              │
│  Step 1: Project encoder states (once per sequence)                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  encoder_features = conv2d(H, W_h)                                    │   │
│  │  W_h: [1, 1, 512, 512] → output: [batch, seq, 1, 512]                │   │
│  │                                                                       │   │
│  │  This is equivalent to:                                               │   │
│  │  encoder_featuresᵢ = hᵢ · W_h   for each position i                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 2: Project decoder state (once per decoder step)                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  decoder_features = Linear(s)                                         │   │
│  │  Shape: [batch, 512]                                                  │   │
│  │                                                                       │   │
│  │  Expand to: [batch, 1, 1, 512] for broadcasting                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 3: Compute attention scores                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  e = v^T · tanh(encoder_features + decoder_features)                  │   │
│  │                                                                       │   │
│  │  For each encoder position i:                                         │   │
│  │    eᵢ = v^T · tanh(W_h · hᵢ + W_s · s + b)                           │   │
│  │                                                                       │   │
│  │  Shape: [batch, seq]                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 4: Apply softmax with masking                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  α = softmax(e)           # Shape: [batch, seq]                       │   │
│  │  α = α * padding_mask     # Zero out padded positions                 │   │
│  │  α = α / sum(α)           # Re-normalize                              │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 5: Compute context vector                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  c = Σᵢ αᵢ · hᵢ          # Weighted sum of encoder states            │   │
│  │                                                                       │   │
│  │  Shape: [batch, 512]                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BAHDANAU ATTENTION ARCHITECTURE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                     Encoder States (H)                                       │
│                     [h₁, h₂, h₃, h₄, h₅]                                    │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                    ┌───────────────────────┐                                │
│                    │    W_h Projection     │                                │
│                    │    (1×1 convolution)  │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│                    [W_h·h₁, W_h·h₂, W_h·h₃, W_h·h₄, W_h·h₅]               │
│                        │   │   │   │   │                                    │
│                        │   │   │   │   │                                    │
│     Decoder State (s) ─┼───┼───┼───┼───┼────────────────────────────────   │
│           │            │   │   │   │   │                                    │
│           ▼            │   │   │   │   │                                    │
│     ┌─────────────┐    │   │   │   │   │                                    │
│     │ W_s·s       │    │   │   │   │   │                                    │
│     └──────┬──────┘    │   │   │   │   │                                    │
│            │           │   │   │   │   │                                    │
│            └───────────┼───┼───┼───┼───┼────────────────────────────────   │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                       [+] [+] [+] [+] [+]    (Element-wise addition)        │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                     [tanh, tanh, tanh, tanh, tanh]                          │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                     ┌───────────────────────┐                                │
│                     │   v^T · (dot product)  │                               │
│                     └───────────┬───────────┘                                │
│                                 │                                            │
│                        [e₁, e₂, e₃, e₄, e₅]  (Attention scores)            │
│                                 │                                            │
│                                 ▼                                            │
│                           ┌──────────┐                                       │
│                           │ Softmax  │                                       │
│                           └────┬─────┘                                       │
│                                │                                             │
│                        [α₁, α₂, α₃, α₄, α₅]  (Attention weights)           │
│                                │                                             │
│                                ▼                                             │
│                     ┌───────────────────────┐                                │
│                     │ c = Σ αᵢ·hᵢ           │                               │
│                     │ (Weighted sum)        │                                │
│                     └───────────┬───────────┘                                │
│                                 │                                            │
│                                 ▼                                            │
│                          Context Vector (c)                                  │
│                             [batch, 512]                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original Code

```python
# File: attention_decoder.py, lines 51-129

def attention(decoder_state, coverage=None):
    """Calculate context vector and attention distribution."""
    
    with variable_scope.variable_scope("Attention"):
        # Project decoder state: W_s · s + b
        decoder_features = linear(decoder_state, attention_vec_size, True)
        # Shape: [batch, 512]
        
        # Reshape for broadcasting
        decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
        # Shape: [batch, 1, 1, 512]
        
        def masked_attention(e):
            """Apply softmax and masking."""
            attn_dist = nn_ops.softmax(e)
            attn_dist *= enc_padding_mask  # Zero out padding
            masked_sums = tf.reduce_sum(attn_dist, axis=1)
            return attn_dist / tf.reshape(masked_sums, [-1, 1])  # Re-normalize
        
        if use_coverage and coverage is not None:
            # With coverage: add coverage features
            coverage_features = nn_ops.conv2d(coverage, w_c, [1,1,1,1], "SAME")
            e = math_ops.reduce_sum(
                v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                [2, 3]
            )
            attn_dist = masked_attention(e)
            coverage += array_ops.reshape(attn_dist, [batch_size, -1, 1, 1])
        else:
            # Without coverage
            e = math_ops.reduce_sum(
                v * math_ops.tanh(encoder_features + decoder_features),
                [2, 3]
            )
            attn_dist = masked_attention(e)
        
        # Compute context vector
        context_vector = math_ops.reduce_sum(
            array_ops.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
            [1, 2]
        )
        context_vector = array_ops.reshape(context_vector, [-1, attn_size])
        
    return context_vector, attn_dist, coverage
```

---

## Proposed Scaled Dot-Product Attention

### Concept

The proposed model uses scaled dot-product attention (from "Attention Is All You Need"):

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     SCALED DOT-PRODUCT ATTENTION                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given:                                                                      │
│    - Encoded sequence: E = [e₁, e₂, ..., eₙ] ∈ ℝ^(n × d_model)             │
│    - Context vector:   c ∈ ℝ^d_model (last valid position)                  │
│                                                                              │
│  Step 1: Compute Query and Keys                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Q = W_q · c          # Query from context                            │   │
│  │  Shape: [batch, 1, d_model]                                          │   │
│  │                                                                       │   │
│  │  K = W_k · E          # Keys from all positions                       │   │
│  │  Shape: [batch, seq, d_model]                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 2: Compute scaled dot-product scores                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  scores = Q · K^T / √d_model                                          │   │
│  │                                                                       │   │
│  │  For each encoder position i:                                         │   │
│  │    scoreᵢ = (W_q · c) · (W_k · eᵢ)^T / √d_model                      │   │
│  │                                                                       │   │
│  │  Shape: [batch, seq]                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 3: Add position bias                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  scores = scores + position_bias[pos_from_end]                        │   │
│  │                                                                       │   │
│  │  This adds a learnable bias based on recency                          │   │
│  │  Recent positions (pos_from_end=1) get higher bias                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 4: Mask and softmax                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  scores[padding_mask] = -inf                                          │   │
│  │  ptr_probs = softmax(scores)                                          │   │
│  │                                                                       │   │
│  │  Shape: [batch, seq]                                                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Step 5: Scatter to location vocabulary                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  ptr_dist = zeros(batch, num_locations)                               │   │
│  │  ptr_dist.scatter_add_(locations, ptr_probs)                          │   │
│  │                                                                       │   │
│  │  This aggregates probabilities for repeated locations:                │   │
│  │  If locations = [101, 205, 150, 312, 150]                            │   │
│  │  ptr_probs =    [0.1, 0.15, 0.2, 0.25, 0.3]                          │   │
│  │                                                                       │   │
│  │  ptr_dist[101] = 0.1                                                  │   │
│  │  ptr_dist[150] = 0.2 + 0.3 = 0.5  (aggregated!)                      │   │
│  │  ptr_dist[205] = 0.15                                                 │   │
│  │  ptr_dist[312] = 0.25                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                SCALED DOT-PRODUCT ATTENTION ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                     Encoded Sequence (E)                                     │
│                     [e₁, e₂, e₃, e₄, e₅]                                    │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                    ┌───────────────────────┐                                │
│                    │    W_k Projection     │                                │
│                    │      (Linear)         │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│                    [k₁, k₂, k₃, k₄, k₅]  (Keys)                            │
│                        │   │   │   │   │                                    │
│                        │   │   │   │   │                                    │
│     Context (c) ───────┼───┼───┼───┼───┼────────────────────────────────   │
│         │              │   │   │   │   │                                    │
│         ▼              │   │   │   │   │                                    │
│   ┌─────────────┐      │   │   │   │   │                                    │
│   │ W_q·c = q   │      │   │   │   │   │                                    │
│   │  (Query)    │      │   │   │   │   │                                    │
│   └──────┬──────┘      │   │   │   │   │                                    │
│          │             │   │   │   │   │                                    │
│          └─────────────┼───┼───┼───┼───┼────────────────────────────────   │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                       [·] [·] [·] [·] [·]    (Dot products: q·kᵢ)           │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                       [÷] [÷] [÷] [÷] [÷]    (Scale by √d_model)           │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│     Position Bias ────[+] [+] [+] [+] [+]    (Add position bias)            │
│     [b₅, b₄, b₃, b₂, b₁]                    (Recent → higher bias)         │
│                        │   │   │   │   │                                    │
│                        ▼   ▼   ▼   ▼   ▼                                    │
│                    ┌───────────────────────┐                                │
│                    │   Mask + Softmax      │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│                        [α₁, α₂, α₃, α₄, α₅]  (Pointer probabilities)       │
│                                │                                            │
│                                ▼                                            │
│                    ┌───────────────────────┐                                │
│                    │    Scatter to         │                                │
│                    │  Location Vocabulary  │                                │
│                    └───────────┬───────────┘                                │
│                                │                                            │
│                                ▼                                            │
│                      Pointer Distribution                                    │
│                      [batch, num_locations]                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Code

```python
# File: pgt.py, lines 132-146

def __init__(self, ...):
    # Pointer mechanism layers
    self.pointer_query = nn.Linear(d_model, d_model)
    self.pointer_key = nn.Linear(d_model, d_model)
    
    # Learnable position bias
    self.position_bias = nn.Parameter(torch.zeros(max_seq_len))

# File: pgt.py, lines 230-240

def forward(self, x, x_dict):
    # ... encoding ...
    
    # Extract context from last valid position
    context = encoded[batch_idx, last_idx]  # [batch, d_model]
    
    # Pointer attention
    query = self.pointer_query(context).unsqueeze(1)  # [batch, 1, d_model]
    keys = self.pointer_key(encoded)                   # [batch, seq, d_model]
    
    # Scaled dot-product
    ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
    ptr_scores = ptr_scores / math.sqrt(self.d_model)  # Scale
    
    # Add position bias
    ptr_scores = ptr_scores + self.position_bias[pos_from_end]
    
    # Mask padding and softmax
    ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
    ptr_probs = F.softmax(ptr_scores, dim=-1)
    
    # Scatter to location vocabulary
    ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
    ptr_dist.scatter_add_(1, x, ptr_probs)
```

---

## Mathematical Comparison

### Bahdanau (Additive) Attention

```
Score computation:
  eᵢ = v^T · tanh(W_h · hᵢ + W_s · s + b)

Parameters:
  - W_h ∈ ℝ^(d_attn × d_enc)    e.g., [512 × 512]
  - W_s ∈ ℝ^(d_attn × d_dec)    e.g., [512 × 256]
  - v ∈ ℝ^d_attn                 e.g., [512]
  - b ∈ ℝ^d_attn                 e.g., [512]

Total parameters: 512×512 + 512×256 + 512 + 512 = 393,728

Complexity per score: O(d_attn × (d_enc + d_dec))
```

### Scaled Dot-Product Attention

```
Score computation:
  scoreᵢ = (q · kᵢ) / √d_model
         = (W_q · c) · (W_k · eᵢ)^T / √d_model

Parameters:
  - W_q ∈ ℝ^(d_model × d_model)  e.g., [64 × 64]
  - W_k ∈ ℝ^(d_model × d_model)  e.g., [64 × 64]
  - position_bias ∈ ℝ^max_seq    e.g., [150]

Total parameters: 64×64 + 64×64 + 150 = 8,342

Complexity per score: O(d_model)
```

### Comparison

| Aspect | Bahdanau | Scaled Dot-Product |
|--------|----------|-------------------|
| **Score Formula** | v^T·tanh(W_h·h + W_s·s) | (W_q·c)·(W_k·e)^T/√d |
| **Non-linearity** | tanh | None (linear) |
| **Parameters** | ~400K | ~8K |
| **Scaling** | None | √d_model |
| **Computation** | Matrix-vector + tanh + vector | Matrix multiply |

---

## Code Implementation

### Side-by-Side Comparison

```python
# ==============================================================================
# ORIGINAL: Bahdanau Attention (TensorFlow)
# ==============================================================================

# Setup (once per sequence)
W_h = variable_scope.get_variable("W_h", [1, 1, attn_size, attention_vec_size])
encoder_features = nn_ops.conv2d(encoder_states, W_h, [1, 1, 1, 1], "SAME")
v = variable_scope.get_variable("v", [attention_vec_size])

# Per decoder step
def attention(decoder_state, coverage=None):
    # Project decoder state
    decoder_features = linear(decoder_state, attention_vec_size, True)
    decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
    
    # Compute scores with tanh non-linearity
    e = math_ops.reduce_sum(
        v * math_ops.tanh(encoder_features + decoder_features), 
        [2, 3]
    )
    
    # Apply masking (multiply then renormalize)
    attn_dist = nn_ops.softmax(e)
    attn_dist *= enc_padding_mask
    attn_dist /= tf.reduce_sum(attn_dist, axis=1, keepdims=True)
    
    # Context vector
    context_vector = tf.reduce_sum(
        tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
        [1, 2]
    )
    
    return context_vector, attn_dist

# ==============================================================================
# PROPOSED: Scaled Dot-Product Attention (PyTorch)
# ==============================================================================

# Setup (in __init__)
self.pointer_query = nn.Linear(d_model, d_model)
self.pointer_key = nn.Linear(d_model, d_model)
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))

# Per forward pass (single step)
def compute_pointer_attention(self, context, encoded, x, mask, pos_from_end):
    # Query from context
    query = self.pointer_query(context).unsqueeze(1)  # [batch, 1, d]
    
    # Keys from encoded sequence
    keys = self.pointer_key(encoded)  # [batch, seq, d]
    
    # Scaled dot-product scores
    ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
    ptr_scores = ptr_scores / math.sqrt(self.d_model)  # Scale
    
    # Add learnable position bias
    ptr_scores = ptr_scores + self.position_bias[pos_from_end]
    
    # Apply masking (set to -inf before softmax)
    ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
    
    # Softmax
    ptr_probs = F.softmax(ptr_scores, dim=-1)
    
    # Scatter to location vocabulary
    ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
    ptr_dist.scatter_add_(1, x, ptr_probs)
    
    return ptr_dist
```

### Key Differences in Implementation

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Projection** | Conv2D + Linear | Two Linear layers |
| **Score computation** | `v * tanh(sum)` → reduce | `bmm` (batch matrix multiply) |
| **Masking strategy** | Multiply + renormalize | Set -inf before softmax |
| **Output** | Context vector | Pointer distribution over vocabulary |
| **Position handling** | Coverage mechanism | Position bias parameter |

---

## Attention Visualization Example

### Example: Alice's Visit History

```
Locations: [Home(101), Coffee(205), Office(150), Restaurant(312), Office(150)]
Positions: [1, 2, 3, 4, 5]
Pos_from_end: [5, 4, 3, 2, 1]
```

### Original Bahdanau Attention Scores

```
Decoder at step t, trying to predict next location:

Decoder state s_t encodes: "Looking for evening location after work"

Attention scores (before masking):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Position:      1         2         3          4           5                 │
│  Location:    Home     Coffee    Office   Restaurant   Office               │
│                                                                              │
│  e_i:        -0.5       -0.3       0.8        0.2         1.2               │
│              ─────     ─────     ─────      ─────       ─────               │
│                │         │         │          │           │                 │
│                ▼         ▼         ▼          ▼           ▼                 │
│             Low         Low      High       Medium      Highest             │
│             (old)      (old)    (work)     (lunch)    (current)             │
│                                                                              │
│  After softmax:                                                              │
│  α:          0.08       0.10       0.30        0.15         0.37            │
│                                                                              │
│  Context vector is weighted sum of encoder states                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Scaled Dot-Product Attention Scores

```
Query (from last position encoding "after work, evening"):
q = W_q · context = [0.5, -0.2, 0.8, ...]  (d_model dimensions)

Keys (from each position):
k_1 (Home):       [0.2, 0.1, -0.3, ...]
k_2 (Coffee):     [0.1, 0.4, -0.1, ...]
k_3 (Office):     [0.6, -0.1, 0.7, ...]   ← Similar to query
k_4 (Restaurant): [0.3, 0.2, 0.4, ...]
k_5 (Office):     [0.6, -0.1, 0.7, ...]   ← Similar to query

Raw scores (q · k_i):
┌─────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Position:      1         2         3          4           5                 │
│  Location:    Home     Coffee    Office   Restaurant   Office               │
│                                                                              │
│  q·k:         0.15       0.22       0.65        0.35         0.65           │
│                                                                              │
│  Scaled (÷√64=8):                                                           │
│              0.019      0.028      0.081        0.044        0.081          │
│                                                                              │
│  + Position bias (recent → higher):                                          │
│  bias:       -0.3       -0.2       -0.1         0.0          0.2            │
│  ─────       ─────     ─────     ─────        ─────        ─────            │
│                                                                              │
│  Final:      -0.28      -0.17      -0.02        0.04         0.28           │
│                                                                              │
│  After softmax:                                                              │
│  ptr_probs:   0.12       0.14       0.16        0.17         0.41           │
│                                                                              │
│  After scatter_add to vocabulary:                                            │
│  ptr_dist[101] = 0.12           (Home)                                      │
│  ptr_dist[150] = 0.16 + 0.41 = 0.57  (Office - aggregated!)                │
│  ptr_dist[205] = 0.14           (Coffee)                                    │
│  ptr_dist[312] = 0.17           (Restaurant)                                │
│                                                                              │
│  Key: Office gets 0.57 probability because it appears twice!                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Visualization Comparison

```
ORIGINAL (Context Vector):
═══════════════════════════════════════════════════════════════════════════════

The attention produces a context vector (weighted sum of encoder states):

  c = 0.08·h_Home + 0.10·h_Coffee + 0.30·h_Office + 0.15·h_Rest + 0.37·h_Office
    = weighted combination of hidden states

This context vector is then used by the decoder to make predictions.
The context is a "summary" of what the model should attend to.

═══════════════════════════════════════════════════════════════════════════════

PROPOSED (Pointer Distribution):
═══════════════════════════════════════════════════════════════════════════════

The attention produces a distribution directly over locations:

  Pointer Attention Weights:
  ┌────────────────────────────────────────────────────────────────────────┐
  │  Home(101)     Coffee(205)    Office(150)    Restaurant(312)          │
  │     0.12          0.14           0.57            0.17                  │
  │      │             │              │               │                    │
  │      ▼             ▼              ▼               ▼                    │
  │   ████          █████         ████████████     ██████                 │
  └────────────────────────────────────────────────────────────────────────┘

  Office (150) has the highest probability because:
  1. It appears twice in the input (positions 3 and 5)
  2. Position 5 is most recent (high position bias)
  3. Both Office positions have high similarity to query

═══════════════════════════════════════════════════════════════════════════════
```

---

## Position Bias Mechanism

### Purpose

The position bias in the proposed model serves a similar purpose to the coverage mechanism in the original, but with a key difference:

| Aspect | Coverage (Original) | Position Bias (Proposed) |
|--------|---------------------|-------------------------|
| **Purpose** | Prevent repetition | Favor recent positions |
| **Computation** | Accumulates attention over decoder steps | Fixed bias per position |
| **Learning** | Learned weight for coverage | Learned bias per position |
| **Updates** | Updated at each decoder step | Static after training |

### Position Bias Implementation

```python
# File: pgt.py, line 135
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))

# File: pgt.py, lines 212-214, 234
# Position from end calculation
positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)

# Apply bias
ptr_scores = ptr_scores + self.position_bias[pos_from_end]
```

### Example Position Bias Values (After Training)

```
Position from end:  1      2      3      4      5      6      7      8      ...
Typical bias:      0.5    0.3    0.1    0.0   -0.1   -0.2   -0.3   -0.4    ...

Interpretation:
- Most recent (pos_from_end=1): +0.5 bonus
- Second most recent: +0.3 bonus
- Middle positions: ~0 (neutral)
- Oldest positions: negative (penalty)

This encourages the model to attend more to recent locations,
which is empirically important for mobility prediction.
```

### Why Position Bias Instead of Coverage?

1. **Task Difference**: 
   - Original: Multi-step generation (need to avoid repeating same word)
   - Proposed: Single-step prediction (no repetition problem)

2. **Semantic Difference**:
   - Coverage tracks what has been "said" in the output
   - Position bias reflects that recent visits are more predictive

3. **Computational Efficiency**:
   - Coverage requires updating at each decoder step
   - Position bias is computed once per forward pass

---

## Summary

| Feature | Original (Bahdanau) | Proposed (Scaled Dot-Product) |
|---------|---------------------|------------------------------|
| **Formula** | v^T·tanh(W_h·h + W_s·s) | QK^T/√d_k |
| **Type** | Additive | Multiplicative |
| **Non-linearity** | tanh | None |
| **Parameters** | ~400K | ~8K |
| **Output** | Context vector | Pointer distribution |
| **Position handling** | Coverage (dynamic) | Position bias (static) |
| **Masking** | Multiply + renormalize | -inf before softmax |

The attention mechanism change represents a shift from:
- **Original**: Complex additive attention producing a context summary
- **Proposed**: Efficient multiplicative attention producing a direct pointer distribution

---

*Next: [05_POINTER_GENERATION_GATE.md](05_POINTER_GENERATION_GATE.md) - The pointer-generation gate mechanism*
