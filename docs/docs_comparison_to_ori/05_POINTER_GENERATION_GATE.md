# Pointer-Generation Gate Mechanism

## Table of Contents
1. [Overview](#overview)
2. [Concept of Pointer-Generator Networks](#concept-of-pointer-generator-networks)
3. [Original Gate Implementation](#original-gate-implementation)
4. [Proposed Gate Implementation](#proposed-gate-implementation)
5. [Mathematical Formulation](#mathematical-formulation)
6. [Code Comparison](#code-comparison)
7. [Example Walkthrough](#example-walkthrough)
8. [Justification for Changes](#justification-for-changes)

---

## Overview

The pointer-generation gate is the core innovation of the Pointer-Generator Network. It learns to decide whether to:
- **Copy** from the input sequence (pointer mechanism)
- **Generate** from the vocabulary (generation head)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    POINTER-GENERATION CONCEPT                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: [Home, Coffee, Office, Restaurant, Office]                          │
│  Output: ?                                                                   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                                                                      │    │
│  │    POINTER MODE                        GENERATION MODE               │    │
│  │    ────────────                        ───────────────               │    │
│  │                                                                      │    │
│  │    "Copy from input"                   "Generate from vocabulary"   │    │
│  │                                                                      │    │
│  │    Best for:                           Best for:                     │    │
│  │    - Returning to known places         - New places                  │    │
│  │    - Repeated visits                   - Never-visited locations     │    │
│  │    - Routine behavior                  - Novel predictions           │    │
│  │                                                                      │    │
│  │    Example:                            Example:                      │    │
│  │    "Go back to Office"                 "Go to new Restaurant"        │    │
│  │    (already in input)                  (not in input history)        │    │
│  │                                                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│                          ┌─────────────┐                                    │
│                          │    GATE     │                                    │
│                          │  (Learned)  │                                    │
│                          │             │                                    │
│                          │  p_gen or   │                                    │
│                          │    gate     │                                    │
│                          │  ∈ [0, 1]   │                                    │
│                          └──────┬──────┘                                    │
│                                 │                                           │
│                                 ▼                                           │
│                                                                              │
│  Final = gate × Pointer_dist + (1-gate) × Generation_dist                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Concept of Pointer-Generator Networks

### The Problem It Solves

```
Problem in Text Summarization (Original):
═══════════════════════════════════════════════════════════════════════════════

Article: "The quick brown fox named Xerxes jumped over the lazy dog named Zeus"

Standard Seq2Seq Output: "The quick brown fox named [UNK] jumped over the [UNK]"
                         ← Can't handle rare names!

Pointer-Generator Output: "The fox Xerxes jumped over Zeus"
                          ← Copies rare names from source!

═══════════════════════════════════════════════════════════════════════════════

Problem in Location Prediction (Proposed):
═══════════════════════════════════════════════════════════════════════════════

History: [Home, Coffee, Office, Restaurant, Office]

Generation-only Output: Might predict "Park" (never visited but common)
                        ← Ignores user's actual history!

Pointer-Generator Output: Predicts "Home" or "Office" (from history)
                          ← Captures return-visit patterns!

═══════════════════════════════════════════════════════════════════════════════
```

### Why Combine Pointer and Generator?

```
Scenario 1: User returns to a known place
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
History: [Home → Work → Gym → Work → ...]
Next: Work (return to familiar place)

→ Pointer should dominate (gate ≈ 1)
→ Copy "Work" from input sequence


Scenario 2: User explores a new place
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
History: [Home → Work → Gym → Work → ...]
Next: New Restaurant (never visited)

→ Generator should dominate (gate ≈ 0)
→ Generate from full vocabulary


The gate learns when to use each strategy!
```

---

## Original Gate Implementation

### p_gen Calculation

The original model computes `p_gen` using four inputs:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORIGINAL p_gen CALCULATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Inputs to p_gen:                                                           │
│    1. context_vector (c): Weighted sum of encoder states [512]              │
│    2. cell_state (s.c): Decoder LSTM cell state [256]                      │
│    3. hidden_state (s.h): Decoder LSTM hidden state [256]                  │
│    4. decoder_input (x): Current decoder input embedding [128]              │
│                                                                              │
│  Total input dimension: 512 + 256 + 256 + 128 = 1152                       │
│                                                                              │
│  Computation:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  p_gen = σ( w_c · c + w_s · s.c + w_h · s.h + w_x · x + b )          │   │
│  │                                                                       │   │
│  │  where:                                                               │   │
│  │    w_c ∈ ℝ^512    (weight for context)                               │   │
│  │    w_s ∈ ℝ^256    (weight for cell state)                            │   │
│  │    w_h ∈ ℝ^256    (weight for hidden state)                          │   │
│  │    w_x ∈ ℝ^128    (weight for input)                                 │   │
│  │    b ∈ ℝ          (bias)                                             │   │
│  │                                                                       │   │
│  │  Parameters: 512 + 256 + 256 + 128 + 1 = 1153                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: p_gen ∈ [0, 1]                                                     │
│    p_gen ≈ 1: Favor generating from vocabulary                              │
│    p_gen ≈ 0: Favor copying from source                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original Code

```python
# File: attention_decoder.py, lines 163-168

# Calculate p_gen
if pointer_gen:
    with tf.variable_scope('calculate_pgen'):
        # Linear combination of context, cell state, hidden state, and input
        p_gen = linear([context_vector, state.c, state.h, x], 1, True)
        p_gen = tf.sigmoid(p_gen)
        p_gens.append(p_gen)
```

### Linear Function Used

```python
# File: attention_decoder.py, lines 184-228

def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """Linear map: sum_i(args[i] * W[i])
    
    Args:
        args: List of 2D tensors [batch, n]
        output_size: Output dimension (1 for p_gen)
        bias: Whether to add bias
    """
    if not isinstance(args, (list, tuple)):
        args = [args]
    
    # Calculate total input size
    total_arg_size = sum(a.get_shape().as_list()[1] for a in args)
    # For p_gen: 512 + 256 + 256 + 128 = 1152
    
    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [total_arg_size, output_size])
        # Shape: [1152, 1]
        
        # Concatenate inputs and multiply
        res = tf.matmul(tf.concat(axis=1, values=args), matrix)
        
        if bias:
            bias_term = tf.get_variable("Bias", [output_size])
            res = res + bias_term
    
    return res  # Shape: [batch, 1]
```

### Final Distribution Calculation (Original)

```python
# File: model.py, lines 146-183

def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate final distribution by combining vocabulary and attention."""
    
    with tf.variable_scope('final_distribution'):
        # Weight distributions by p_gen
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
        attn_dists = [(1 - p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]
        
        # Extend vocabulary for OOVs
        extended_vsize = self._vocab.size() + self._max_art_oovs
        extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
        vocab_dists_extended = [tf.concat([dist, extra_zeros], axis=1) for dist in vocab_dists]
        
        # Project attention to vocabulary indices
        batch_nums = tf.range(0, limit=self._hps.batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        attn_len = tf.shape(self._enc_batch_extend_vocab)[1]
        batch_nums = tf.tile(batch_nums, [1, attn_len])
        indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)
        
        shape = [self._hps.batch_size, extended_vsize]
        attn_dists_projected = [tf.scatter_nd(indices, dist, shape) for dist in attn_dists]
        
        # Combine: p_gen * vocab + (1-p_gen) * attn
        final_dists = [vocab_dist + attn_dist 
                       for (vocab_dist, attn_dist) in zip(vocab_dists_extended, attn_dists_projected)]
        
        return final_dists
```

---

## Proposed Gate Implementation

### Gate Calculation

The proposed model uses a simpler but more expressive MLP:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED GATE CALCULATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input to gate:                                                             │
│    - context: Encoded representation of last position [d_model=64]          │
│                                                                              │
│  Computation:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  gate = MLP(context)                                                  │   │
│  │                                                                       │   │
│  │  MLP Architecture:                                                    │   │
│  │    Layer 1: Linear(d_model → d_model/2) = Linear(64 → 32)            │   │
│  │    Activation: GELU                                                   │   │
│  │    Layer 2: Linear(d_model/2 → 1) = Linear(32 → 1)                   │   │
│  │    Activation: Sigmoid                                                │   │
│  │                                                                       │   │
│  │  Parameters:                                                          │   │
│  │    Layer 1: 64 × 32 + 32 = 2080                                      │   │
│  │    Layer 2: 32 × 1 + 1 = 33                                          │   │
│  │    Total: 2113                                                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Output: gate ∈ [0, 1]                                                      │
│    gate ≈ 1: Favor pointer (copying from input)                             │
│    gate ≈ 0: Favor generation (from vocabulary)                             │
│                                                                              │
│  Note: The semantics are INVERTED from original!                            │
│    Original: p_gen ≈ 1 means generate                                       │
│    Proposed: gate ≈ 1 means pointer (copy)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Code

```python
# File: pointer_v45.py, lines 140-146

def __init__(self, ...):
    # ...
    
    # Pointer-Generation gate (MLP)
    self.ptr_gen_gate = nn.Sequential(
        nn.Linear(d_model, d_model // 2),  # 64 → 32
        nn.GELU(),                          # Non-linearity
        nn.Linear(d_model // 2, 1),         # 32 → 1
        nn.Sigmoid()                        # Output in [0, 1]
    )

# File: pointer_v45.py, lines 245-248

def forward(self, x, x_dict):
    # ... pointer and generation distributions ...
    
    # Gate and combine
    gate = self.ptr_gen_gate(context)  # [batch, 1]
    final_probs = gate * ptr_dist + (1 - gate) * gen_probs
    
    return torch.log(final_probs + 1e-10)
```

---

## Mathematical Formulation

### Original p_gen

```
Input concatenation:
  z = [context; cell_state; hidden_state; input]
  z ∈ ℝ^(512 + 256 + 256 + 128) = ℝ^1152

Linear transformation:
  p_gen = σ(W · z + b)
  
  where:
    W ∈ ℝ^(1 × 1152)
    b ∈ ℝ

Final distribution:
  P_final(w) = p_gen · P_vocab(w) + (1 - p_gen) · Σᵢ α_i · 𝟙[w_i = w]

  - P_vocab: Softmax over vocabulary from decoder output
  - α: Attention weights
  - The sum aggregates attention over all positions with word w
```

### Proposed Gate

```
MLP transformation:
  h = GELU(W₁ · context + b₁)
  gate = σ(W₂ · h + b₂)
  
  where:
    W₁ ∈ ℝ^(d_model/2 × d_model) = ℝ^(32 × 64)
    b₁ ∈ ℝ^(d_model/2) = ℝ^32
    W₂ ∈ ℝ^(1 × d_model/2) = ℝ^(1 × 32)
    b₂ ∈ ℝ

Final distribution:
  P_final(l) = gate · P_ptr(l) + (1 - gate) · P_gen(l)

  - P_ptr: Pointer distribution scattered to location vocabulary
  - P_gen: Softmax over locations from generation head
  - l: Location index
```

### Key Differences

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Input** | concat([context, cell, hidden, input]) | context only |
| **Input dim** | 1152 | 64 |
| **Architecture** | Single linear layer | 2-layer MLP |
| **Non-linearity** | None (before sigmoid) | GELU |
| **Parameters** | 1153 | 2113 |
| **Semantics** | p_gen=1 → generate | gate=1 → pointer |

---

## Code Comparison

### Side-by-Side Implementation

```python
# ==============================================================================
# ORIGINAL: p_gen Calculation (TensorFlow)
# ==============================================================================

# File: attention_decoder.py

# In the attention_decoder function, for each decoder step:
for i, inp in enumerate(decoder_inputs):
    # ... attention computation ...
    
    # Calculate p_gen
    if pointer_gen:
        with tf.variable_scope('calculate_pgen'):
            # Concatenate all relevant vectors
            # context_vector: [batch, 512]
            # state.c: [batch, 256]  (cell state)
            # state.h: [batch, 256]  (hidden state)
            # x: [batch, 128]        (fused input)
            
            # Linear: [batch, 1152] → [batch, 1]
            p_gen = linear([context_vector, state.c, state.h, x], 1, True)
            p_gen = tf.sigmoid(p_gen)
            p_gens.append(p_gen)

# File: model.py - Final distribution

def _calc_final_dist(self, vocab_dists, attn_dists):
    vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
    attn_dists = [(1-p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]
    # ... extend vocab and scatter ...
    final_dists = [vocab_dist + attn_dist for ...]
    return final_dists

# ==============================================================================
# PROPOSED: Gate Calculation (PyTorch)
# ==============================================================================

# File: pointer_v45.py

class PointerNetworkV45(nn.Module):
    def __init__(self, ...):
        # ...
        
        # Define gate as an MLP
        self.ptr_gen_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 64 → 32
            nn.GELU(),                          # Non-linear activation
            nn.Linear(d_model // 2, 1),         # 32 → 1
            nn.Sigmoid()                        # Squash to [0, 1]
        )
    
    def forward(self, x, x_dict):
        # ... encoding and attention ...
        
        # context: [batch, d_model] from last position
        context = encoded[batch_idx, last_idx]
        
        # Pointer distribution (scattered to vocabulary)
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
        ptr_dist.scatter_add_(1, x, ptr_probs)
        
        # Generation distribution
        gen_probs = F.softmax(self.gen_head(context), dim=-1)
        
        # Gate calculation (single forward pass)
        gate = self.ptr_gen_gate(context)  # [batch, 1]
        
        # Combine: gate * ptr + (1-gate) * gen
        # Note: gate=1 means POINTER (opposite of original p_gen!)
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs
        
        return torch.log(final_probs + 1e-10)
```

### Architecture Diagram Comparison

```
ORIGINAL GATE:
═══════════════════════════════════════════════════════════════════════════════

                    context [512]
                        │
                        │
            ┌───────────┼───────────┐
            │           │           │
     cell_state [256]   │    hidden_state [256]
            │           │           │
            └───────────┼───────────┘
                        │
                   input [128]
                        │
            ┌───────────┴───────────┐
            │     Concatenate       │
            │       [1152]          │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │   Linear(1152 → 1)    │
            │   + Bias              │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │       Sigmoid         │
            └───────────┬───────────┘
                        │
                        ▼
                   p_gen [1]
            (1 = generate, 0 = copy)

═══════════════════════════════════════════════════════════════════════════════

PROPOSED GATE:
═══════════════════════════════════════════════════════════════════════════════

                   context [64]
                        │
            ┌───────────┴───────────┐
            │   Linear(64 → 32)     │
            │   + Bias              │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │        GELU           │
            │   (Non-linearity)     │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │   Linear(32 → 1)      │
            │   + Bias              │
            └───────────┬───────────┘
                        │
            ┌───────────┴───────────┐
            │       Sigmoid         │
            └───────────┬───────────┘
                        │
                        ▼
                    gate [1]
            (1 = pointer, 0 = generate)

═══════════════════════════════════════════════════════════════════════════════
```

---

## Example Walkthrough

### Scenario: Alice Predicting Next Location

```
Input: [Home(101), Coffee(205), Office(150), Restaurant(312), Office(150)]
User: Alice (user_id=42)
Time: 18:00 (end of work day)

Context encodes: "Alice, after work, typically goes to..."
```

### Pointer Distribution

```
From pointer attention:
  ptr_probs = [0.12, 0.14, 0.16, 0.17, 0.41]  (over 5 positions)

After scatter_add to vocabulary:
  ptr_dist[101] = 0.12         (Home)
  ptr_dist[150] = 0.57         (Office: 0.16 + 0.41)
  ptr_dist[205] = 0.14         (Coffee)
  ptr_dist[312] = 0.17         (Restaurant)
  ptr_dist[others] = 0         (Not in history)

Pointer strongly suggests: Office (0.57) because it appears twice
```

### Generation Distribution

```
From generation head:
  gen_probs = softmax(Linear(context))

Typical output:
  gen_probs[101] = 0.15        (Home - common evening destination)
  gen_probs[150] = 0.10        (Office - less likely evening)
  gen_probs[89] = 0.25         (Gym - common after-work activity!)
  gen_probs[205] = 0.05        (Coffee - unlikely evening)
  gen_probs[312] = 0.08        (Restaurant - possible)
  gen_probs[xxx] = ...         (Other locations)

Generation suggests: Gym (0.25) as a new location not in history
```

### Gate Decision

```
ORIGINAL p_gen:
═══════════════════════════════════════════════════════════════════════════════

Input to p_gen:
  concat([context, cell_state, hidden_state, input])
  = [512 + 256 + 256 + 128] = [1152] dimensions

Let's say p_gen = σ(W · z + b) = 0.35

Meaning: 35% generate from vocab, 65% copy from source

Final for location 101 (Home):
  P(101) = 0.35 × 0.15 + 0.65 × 0.12 = 0.0525 + 0.078 = 0.1305

Final for location 89 (Gym):
  P(89) = 0.35 × 0.25 + 0.65 × 0 = 0.0875 + 0 = 0.0875
  (Gym not in history, so pointer gives 0)

Final for location 150 (Office):
  P(150) = 0.35 × 0.10 + 0.65 × 0.57 = 0.035 + 0.3705 = 0.4055

Prediction: Office (0.4055) - most likely due to pointer

═══════════════════════════════════════════════════════════════════════════════

PROPOSED gate:
═══════════════════════════════════════════════════════════════════════════════

Input to gate: context [64 dimensions]

Processing:
  h = GELU(W₁ · context + b₁)    # [64] → [32]
  gate = σ(W₂ · h + b₂)          # [32] → [1]

Let's say gate = 0.7

Meaning: 70% pointer (copy), 30% generator

Final for location 101 (Home):
  P(101) = 0.7 × 0.12 + 0.3 × 0.15 = 0.084 + 0.045 = 0.129

Final for location 89 (Gym):
  P(89) = 0.7 × 0 + 0.3 × 0.25 = 0 + 0.075 = 0.075

Final for location 150 (Office):
  P(150) = 0.7 × 0.57 + 0.3 × 0.10 = 0.399 + 0.03 = 0.429

Prediction: Office (0.429) - most likely due to pointer

═══════════════════════════════════════════════════════════════════════════════
```

### Visualization of Gate Effect

```
                    Gate Value Effect on Final Distribution
═══════════════════════════════════════════════════════════════════════════════

gate = 0.0 (Pure Generation)                gate = 1.0 (Pure Pointer)
─────────────────────────────               ─────────────────────────
     Gym: 0.25 ████████████                      Office: 0.57 ███████████████
    Home: 0.15 ███████                           Rest.: 0.17 █████
  Office: 0.10 █████                            Coffee: 0.14 ████
   Rest.: 0.08 ████                               Home: 0.12 ███
  Coffee: 0.05 ██                                  Gym: 0.00

Model favors NEW locations              Model favors KNOWN locations
(Gym is predicted)                      (Office is predicted)

═══════════════════════════════════════════════════════════════════════════════

gate = 0.5 (Balanced)                       gate = 0.7 (Pointer-heavy)
─────────────────────────────               ─────────────────────────────
  Office: 0.335 ███████████                   Office: 0.429 ██████████████
     Gym: 0.125 ████                            Home: 0.129 ████
    Home: 0.135 ████                            Rest.: 0.143 █████
   Rest.: 0.125 ████                          Coffee: 0.113 ████
  Coffee: 0.095 ███                              Gym: 0.075 ██

Blends both strategies                  Strongly favors pointer
                                        but keeps generation influence

═══════════════════════════════════════════════════════════════════════════════
```

---

## Justification for Changes

### 1. Simpler Input to Gate

| Original | Proposed | Justification |
|----------|----------|---------------|
| 4 inputs (context, cell, hidden, input) | 1 input (context) | No decoder in proposed model; context already encodes all information |

**Reasoning**: The original needs multiple inputs because:
- Context: What the encoder says
- Cell/Hidden: What the decoder has generated so far
- Input: Current step's input

The proposed model has no decoder, and the Transformer's context already captures all relevant information through self-attention.

### 2. MLP Instead of Linear Layer

| Original | Proposed | Justification |
|----------|----------|---------------|
| Single linear layer | 2-layer MLP with GELU | More expressive decision boundary |

**Reasoning**: The gate decision is non-trivial:
- Need to decide based on complex patterns
- Single linear layer has limited expressivity
- GELU provides smooth non-linearity for better gradient flow

```
Example Decision Boundary:

Linear Gate (Original):
  ┌───────────────────────────────┐
  │                               │
  │   Generate  │   Pointer       │  ← Straight line separates regions
  │    Region   │    Region       │
  │             │                 │
  │             │                 │
  └───────────────────────────────┘

MLP Gate (Proposed):
  ┌───────────────────────────────┐
  │                               │
  │   Generate  ╲   Pointer       │  ← Curved boundary
  │    Region    ╲   Region       │     can capture more complex
  │           ╱   ╲               │     decision rules
  │          ╱     ╲              │
  └───────────────────────────────┘
```

### 3. Inverted Semantics

| Original | Proposed | Justification |
|----------|----------|---------------|
| p_gen=1 → generate | gate=1 → pointer | More intuitive for location prediction |

**Reasoning**: In location prediction:
- Most predictions are return visits (pointer)
- Novel locations are less common
- gate=1 meaning "favor the main strategy (pointer)" is clearer

### 4. No Extended Vocabulary

| Original | Proposed | Justification |
|----------|----------|---------------|
| Extended vocab for OOVs | Fixed vocabulary | All locations are known |

**Reasoning**: In text summarization, new words (names, rare terms) appear in articles. In location prediction, all locations are pre-defined in the vocabulary. There's no need for dynamic vocabulary extension.

---

## Summary Table

| Feature | Original | Proposed |
|---------|----------|----------|
| **Input** | [context, cell, hidden, input] | context |
| **Input Dim** | 1152 | 64 |
| **Architecture** | Linear + Sigmoid | Linear + GELU + Linear + Sigmoid |
| **Parameters** | 1153 | 2113 |
| **Output Meaning** | p_gen=1 → generate | gate=1 → pointer |
| **Vocabulary** | Extended (dynamic) | Fixed |
| **Computation** | Per decoder step | Once per input |

The gate mechanism change represents a simplification and adaptation:
- **Simpler input**: Only context needed (no decoder states)
- **More expressive**: MLP allows complex decision boundaries
- **Task-appropriate**: Inverted semantics match location prediction patterns
- **Efficient**: Single computation per input (no iterative decoding)

---

*Next: [06_EMBEDDING_COMPARISON.md](06_EMBEDDING_COMPARISON.md) - Feature embeddings and representation*
