# Attention Mechanism Deep Dive

## Table of Contents
1. [Introduction to Attention](#introduction-to-attention)
2. [Bahdanau Attention](#bahdanau-attention)
3. [Implementation Details](#implementation-details)
4. [Code Walkthrough](#code-walkthrough)
5. [Masking and Normalization](#masking-and-normalization)
6. [Context Vector Computation](#context-vector-computation)
7. [Attention Feeding](#attention-feeding)
8. [Visualization Examples](#visualization-examples)

---

## Introduction to Attention

### Why Attention?

The attention mechanism solves the fundamental problem of information bottleneck in sequence-to-sequence models.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    THE INFORMATION BOTTLENECK PROBLEM                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WITHOUT ATTENTION:                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   Input: "Germany emerged as the winners of the 2014 FIFA World Cup after        │
│           defeating Argentina 1-0 in the final at the Maracanã Stadium in        │
│           Rio de Janeiro. Mario Götze scored the winning goal in extra time."   │
│                                                                                   │
│           ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓            │
│                                                                                   │
│                    ┌──────────────────────────────┐                               │
│                    │     Fixed-Size Vector        │                               │
│                    │        (256 dims)            │                               │
│                    │                              │                               │
│                    │  How can 256 numbers         │                               │
│                    │  encode 40+ words?           │                               │
│                    └──────────────────────────────┘                               │
│                                                                                   │
│           ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓            │
│                                                                                   │
│   Output: "Germany won the World Cup."  (Lost: Argentina, 1-0, Götze, etc.)      │
│                                                                                   │
│                                                                                   │
│   WITH ATTENTION:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Input: [h₁, h₂, h₃, ..., h₄₀]  (All encoder states preserved!)                │
│                                                                                   │
│   When generating "Germany": Look at h₁ (Germany)                                │
│   When generating "1-0":     Look at h₁₂ (1-0)                                   │
│   When generating "Götze":   Look at h₂₈ (Götze)                                 │
│                                                                                   │
│   Output: "Germany beat Argentina 1-0. Götze scored."  (Details preserved!)      │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Attention as Soft Search

Attention can be thought of as a differentiable search operation:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      ATTENTION AS SOFT SEARCH                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   QUERY: "What information do I need to generate the next word?"                 │
│   ─────  (Represented by decoder state s_t)                                      │
│                                                                                   │
│   KEYS: "What information does each input word contain?"                         │
│   ────  (Represented by encoder states h_1, h_2, ..., h_n)                       │
│                                                                                   │
│   VALUES: "What is the actual content at each position?"                         │
│   ──────  (Also encoder states h_1, h_2, ..., h_n in basic attention)            │
│                                                                                   │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                         │    │
│   │   Query (s_t)                                                          │    │
│   │      │                                                                  │    │
│   │      ▼                                                                  │    │
│   │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                     │    │
│   │   │ h₁  │   │ h₂  │   │ h₃  │   │ h₄  │   │ h₅  │    Keys             │    │
│   │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘                     │    │
│   │      │         │         │         │         │                         │    │
│   │      ▼         ▼         ▼         ▼         ▼                         │    │
│   │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                     │    │
│   │   │ 0.4 │   │ 0.1 │   │ 0.3 │   │ 0.1 │   │ 0.1 │    Weights (α)      │    │
│   │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘    (sum = 1.0)      │    │
│   │      │         │         │         │         │                         │    │
│   │      ▼         ▼         ▼         ▼         ▼                         │    │
│   │   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                     │    │
│   │   │0.4h₁│ + │0.1h₂│ + │0.3h₃│ + │0.1h₄│ + │0.1h₅│    Weighted Sum     │    │
│   │   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘                     │    │
│   │      │         │         │         │         │                         │    │
│   │      └─────────┴─────────┴────┬────┴─────────┘                         │    │
│   │                               │                                        │    │
│   │                               ▼                                        │    │
│   │                        ┌───────────┐                                   │    │
│   │                        │ Context   │                                   │    │
│   │                        │ Vector c  │  = 0.4h₁ + 0.1h₂ + 0.3h₃ + ...  │    │
│   │                        └───────────┘                                   │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Bahdanau Attention

### Mathematical Formulation

The pointer-generator uses Bahdanau attention (additive attention):

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         BAHDANAU ATTENTION EQUATIONS                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Given:                                                                          │
│   - Encoder states: h₁, h₂, ..., hₙ  ∈ ℝ^(2×hidden_dim)                          │
│   - Decoder state at step t: sₜ  ∈ ℝ^hidden_dim                                  │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: Compute Energy (Alignment Scores)                                       │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   For each encoder position i:                                                    │
│                                                                                   │
│   eₜᵢ = v^T · tanh(Wₕ · hᵢ + Wₛ · sₜ + bₐₜₜₙ)                                   │
│                                                                                   │
│   Where:                                                                          │
│   - Wₕ ∈ ℝ^(attn_size × 2×hidden_dim)  : Encoder state projection               │
│   - Wₛ ∈ ℝ^(attn_size × hidden_dim)    : Decoder state projection               │
│   - v  ∈ ℝ^attn_size                   : Attention weight vector                │
│   - bₐₜₜₙ ∈ ℝ^attn_size               : Bias term                               │
│   - attn_size = 2 × hidden_dim = 512                                             │
│                                                                                   │
│                                                                                   │
│   Intuition:                                                                      │
│   ──────────                                                                      │
│   Wₕ · hᵢ  →  "Encode what this source position contains"                       │
│   Wₛ · sₜ  →  "Encode what the decoder is looking for"                          │
│   tanh(sum) → "Non-linear combination of both"                                   │
│   v^T · ... → "Score how well they match" (scalar)                              │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: Normalize to Probabilities                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   αₜᵢ = softmax(eₜᵢ) = exp(eₜᵢ) / Σⱼ exp(eₜⱼ)                                   │
│                                                                                   │
│   Properties:                                                                      │
│   - αₜᵢ ≥ 0 for all i                                                            │
│   - Σᵢ αₜᵢ = 1                                                                   │
│   - αₜᵢ represents "probability of attending to position i"                      │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: Compute Context Vector                                                  │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   cₜ = Σᵢ αₜᵢ · hᵢ                                                               │
│                                                                                   │
│   The context vector is a weighted average of encoder states.                    │
│   Shape: ℝ^(2×hidden_dim) = ℝ^512                                               │
│                                                                                   │
│   Intuition: "Summarize the relevant parts of the input for this step"          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Comparison with Other Attention Types

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       ATTENTION MECHANISM COMPARISON                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   BAHDANAU (Additive) Attention - Used in this model                             │
│   ──────────────────────────────                                                  │
│                                                                                   │
│   eₜᵢ = v^T · tanh(Wₕ · hᵢ + Wₛ · sₜ)                                           │
│                                                                                   │
│   Pros:                                                                           │
│   + More expressive (learns separate projections)                                │
│   + Works well when query and key have different dimensions                      │
│   + More parameters → more capacity                                              │
│                                                                                   │
│   Cons:                                                                           │
│   - Slower (two matrix multiplications + tanh)                                   │
│   - More memory                                                                   │
│                                                                                   │
│   ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                   │
│   LUONG (Multiplicative) Attention                                               │
│   ────────────────────────────────                                               │
│                                                                                   │
│   eₜᵢ = sₜ^T · Wₐ · hᵢ      (general)                                           │
│   eₜᵢ = sₜ^T · hᵢ           (dot product)                                       │
│                                                                                   │
│   Pros:                                                                           │
│   + Faster computation                                                           │
│   + Fewer parameters                                                             │
│                                                                                   │
│   Cons:                                                                           │
│   - Less expressive                                                              │
│   - Requires query and key to have same dimension                                │
│                                                                                   │
│   ─────────────────────────────────────────────────────────────────────────────  │
│                                                                                   │
│   SCALED DOT-PRODUCT Attention (Transformer)                                     │
│   ──────────────────────────────────────────                                      │
│                                                                                   │
│   eₜᵢ = (sₜ · hᵢ) / √dₖ                                                          │
│                                                                                   │
│   Pros:                                                                           │
│   + Very fast (can be parallelized with matrix ops)                             │
│   + Works great with self-attention                                             │
│                                                                                   │
│   Cons:                                                                           │
│   - Scaling needed to prevent softmax saturation                                 │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Pre-computed Encoder Features

The implementation uses a clever optimization: encoder features are computed **once** and reused:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    PRE-COMPUTED ENCODER FEATURES                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   attention_decoder.py: Lines 51-67                                              │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   NAIVE IMPLEMENTATION (slow):                                                    │
│   ─────────────────────────────                                                   │
│                                                                                   │
│   For each decoder step t:                                                        │
│       For each encoder position i:                                                │
│           encoder_feature = Wₕ · hᵢ    ← Recomputed every step!                  │
│           e_ti = v^T · tanh(encoder_feature + decoder_feature)                   │
│                                                                                   │
│   Time complexity: O(T × N × d²)   where T=dec_len, N=enc_len, d=dim            │
│                                                                                   │
│                                                                                   │
│   OPTIMIZED IMPLEMENTATION (actual):                                              │
│   ───────────────────────────────────                                             │
│                                                                                   │
│   # Compute ONCE before decoder loop                                             │
│   encoder_states = expand_dims(encoder_states, axis=2)                           │
│   # Shape: [batch, enc_len, 1, attn_size]                                        │
│                                                                                   │
│   W_h = get_variable("W_h", [1, 1, attn_size, attn_size])                        │
│   encoder_features = conv2d(encoder_states, W_h)                                 │
│   # Shape: [batch, enc_len, 1, attn_size]                                        │
│                                                                                   │
│   For each decoder step t:                                                        │
│       decoder_features = Linear(sₜ)   ← Only THIS is computed each step         │
│       e_t = v^T · tanh(encoder_features + decoder_features)                      │
│                                                                                   │
│   Time complexity: O(N × d²) + O(T × d²)  ← Much better!                         │
│                                                                                   │
│                                                                                   │
│   WHY CONV2D?                                                                     │
│   ──────────                                                                      │
│   Using conv2d with kernel [1,1] is equivalent to matrix multiplication          │
│   but allows batch processing of all encoder positions at once.                  │
│                                                                                   │
│   encoder_states: [batch, enc_len, 1, 512]                                       │
│   W_h:            [1, 1, 512, 512]                                               │
│   Result:         [batch, enc_len, 1, 512]                                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Walkthrough

### attention_decoder.py: The `attention()` Function

```python
# attention_decoder.py: Lines 79-129 (annotated)

def attention(decoder_state, coverage=None):
    """Calculate the context vector and attention distribution.
    
    Args:
        decoder_state: state of the decoder (sₜ)
        coverage: Previous timestep's coverage vector (optional)
    
    Returns:
        context_vector: weighted sum of encoder_states
        attn_dist: attention distribution
        coverage: new coverage vector (if using coverage)
    """
```

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION FUNCTION WALKTHROUGH                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   STEP 1: Project decoder state                                                   │
│   ─────────────────────────────                                                   │
│                                                                                   │
│   decoder_features = linear(decoder_state, attention_vec_size, True)             │
│   # Input:  decoder_state  [batch, hidden_dim]     = [16, 256]                   │
│   # Output: decoder_features [batch, attn_size]    = [16, 512]                   │
│                                                                                   │
│   decoder_features = expand_dims(expand_dims(decoder_features, 1), 1)            │
│   # Shape: [batch, 1, 1, attn_size] = [16, 1, 1, 512]                           │
│   # This allows broadcasting with encoder_features                               │
│                                                                                   │
│                                                                                   │
│   STEP 2: Calculate energy scores                                                 │
│   ───────────────────────────────                                                 │
│                                                                                   │
│   WITHOUT COVERAGE:                                                               │
│   ─────────────────                                                               │
│   e = v * tanh(encoder_features + decoder_features)                              │
│   # encoder_features: [batch, enc_len, 1, 512]                                   │
│   # decoder_features: [batch, 1, 1, 512]    (broadcasts)                         │
│   # tanh(...):        [batch, enc_len, 1, 512]                                   │
│   # v:                [512]                                                       │
│   # v * tanh:         [batch, enc_len, 1, 512]  (element-wise)                   │
│                                                                                   │
│   e = reduce_sum(e, [2, 3])                                                      │
│   # Shape: [batch, enc_len] = [16, 350]                                          │
│   # Each value = score for that encoder position                                 │
│                                                                                   │
│                                                                                   │
│   WITH COVERAGE:                                                                  │
│   ──────────────                                                                  │
│   coverage_features = conv2d(coverage, w_c)                                      │
│   # coverage:          [batch, enc_len, 1, 1]                                    │
│   # w_c:               [1, 1, 1, 512]                                            │
│   # coverage_features: [batch, enc_len, 1, 512]                                  │
│                                                                                   │
│   e = v * tanh(encoder_features + decoder_features + coverage_features)          │
│   # The coverage is added to influence attention scores                          │
│                                                                                   │
│                                                                                   │
│   STEP 3: Masked softmax                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   attn_dist = masked_attention(e)                                                │
│   # See next section for details                                                 │
│   # Output: [batch, enc_len] = [16, 350]                                         │
│   # Sum of each row = 1.0 (probability distribution)                             │
│                                                                                   │
│                                                                                   │
│   STEP 4: Update coverage (if using)                                              │
│   ──────────────────────────────────                                              │
│                                                                                   │
│   coverage += reshape(attn_dist, [batch, enc_len, 1, 1])                         │
│   # Accumulate attention for coverage loss                                       │
│                                                                                   │
│                                                                                   │
│   STEP 5: Compute context vector                                                  │
│   ──────────────────────────────                                                  │
│                                                                                   │
│   context_vector = reduce_sum(                                                    │
│       reshape(attn_dist, [batch, enc_len, 1, 1]) * encoder_states,              │
│       [1, 2]                                                                      │
│   )                                                                               │
│   # attn_dist:       [batch, enc_len, 1, 1]                                      │
│   # encoder_states:  [batch, enc_len, 1, 512]                                    │
│   # weighted:        [batch, enc_len, 1, 512]                                    │
│   # sum over enc_len: [batch, 512]                                               │
│                                                                                   │
│   context_vector = reshape(context_vector, [batch, attn_size])                   │
│   # Final shape: [16, 512]                                                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Masking and Normalization

### The Masked Attention Function

Padding tokens should not receive attention. The model handles this with masking:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        MASKED ATTENTION                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   attention_decoder.py: Lines 96-101                                             │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   def masked_attention(e):                                                        │
│       """Take softmax of e then apply enc_padding_mask and re-normalize"""       │
│       attn_dist = softmax(e)                                                      │
│       attn_dist *= enc_padding_mask                                              │
│       masked_sums = reduce_sum(attn_dist, axis=1)                                │
│       return attn_dist / reshape(masked_sums, [-1, 1])                           │
│                                                                                   │
│                                                                                   │
│   EXAMPLE:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   Source:    ["Germany", "won", ".", "[PAD]", "[PAD]"]                           │
│   enc_mask:  [1.0,       1.0,  1.0,  0.0,     0.0]                               │
│                                                                                   │
│   Step 1: Raw softmax                                                             │
│   e = [2.5, 1.0, 0.5, -0.3, -0.5]                                                │
│   softmax(e) = [0.55, 0.12, 0.07, 0.03, 0.02] + 0.21 = 1.0 ✓                    │
│                                       ↑     ↑                                     │
│                              These should be ZERO!                               │
│                                                                                   │
│   Step 2: Apply mask                                                              │
│   attn_dist = [0.55, 0.12, 0.07, 0.03, 0.02]                                     │
│   enc_mask  = [1.0,  1.0,  1.0,  0.0,  0.0]                                      │
│   masked    = [0.55, 0.12, 0.07, 0.0,  0.0]  (sum = 0.74)                        │
│                                                                                   │
│   Step 3: Re-normalize                                                            │
│   masked_sum = 0.74                                                               │
│   final = [0.55/0.74, 0.12/0.74, 0.07/0.74, 0, 0]                                │
│         = [0.74,      0.16,      0.10,      0, 0]  (sum = 1.0) ✓                 │
│                                                                                   │
│                                                                                   │
│   WHY THIS APPROACH?                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   Alternative: Set e to -inf for padded positions before softmax                 │
│   - This works but can cause numerical issues                                    │
│   - Current approach is more stable                                              │
│                                                                                   │
│   Key insight: We want attention ONLY on real tokens, never on [PAD]            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Context Vector Computation

### Detailed Breakdown

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    CONTEXT VECTOR COMPUTATION                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   MATHEMATICAL DEFINITION:                                                        │
│   cₜ = Σᵢ αₜᵢ · hᵢ                                                               │
│                                                                                   │
│                                                                                   │
│   EXAMPLE WITH REAL NUMBERS:                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   Encoder states (simplified to 4-dim for visualization):                        │
│                                                                                   │
│   h₁ ("Germany") = [0.8, -0.2, 0.5, 0.1]                                        │
│   h₂ ("won")     = [0.1, 0.7, -0.3, 0.4]                                        │
│   h₃ (".")       = [-0.1, 0.2, 0.1, -0.3]                                       │
│                                                                                   │
│   Attention weights (from softmax):                                              │
│   α = [0.74, 0.16, 0.10]                                                         │
│                                                                                   │
│                                                                                   │
│   Context vector calculation:                                                     │
│   ───────────────────────────                                                     │
│                                                                                   │
│   c = 0.74 × h₁ + 0.16 × h₂ + 0.10 × h₃                                         │
│                                                                                   │
│   c = 0.74 × [0.8, -0.2, 0.5, 0.1]                                              │
│     + 0.16 × [0.1, 0.7, -0.3, 0.4]                                              │
│     + 0.10 × [-0.1, 0.2, 0.1, -0.3]                                             │
│                                                                                   │
│   c = [0.592, -0.148, 0.370, 0.074]    # From h₁                                │
│     + [0.016, 0.112, -0.048, 0.064]    # From h₂                                │
│     + [-0.01, 0.02, 0.01, -0.03]       # From h₃                                │
│                                                                                   │
│   c = [0.598, -0.016, 0.332, 0.108]                                             │
│                                                                                   │
│                                                                                   │
│   INTERPRETATION:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   The context vector is dominated by "Germany" (74% weight)                      │
│   because that's what the decoder is focusing on for this step.                  │
│                                                                                   │
│   The context vector "looks like" h₁ but with some influence from h₂ and h₃.   │
│                                                                                   │
│                                                                                   │
│   IN PRACTICE (512 dimensions):                                                   │
│   ─────────────────────────────                                                   │
│                                                                                   │
│   # Attention weights: [batch, enc_len] = [16, 350]                              │
│   # Encoder states:    [batch, enc_len, 512]                                     │
│                                                                                   │
│   # Reshape attention for broadcasting                                            │
│   attn_reshaped = attn_dist[:, :, None]  # [16, 350, 1]                         │
│                                                                                   │
│   # Element-wise multiplication + sum                                             │
│   weighted = attn_reshaped * encoder_states  # [16, 350, 512]                   │
│   context = sum(weighted, axis=1)            # [16, 512]                         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Feeding

### Why Feed Context to Decoder Input?

The model uses a technique called "attention feeding" or "input feeding":

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         ATTENTION FEEDING                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   attention_decoder.py: Lines 146-150                                            │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   # Merge input and previous attentions into one vector                          │
│   input_size = inp.get_shape().with_rank(2)[1]                                   │
│   x = linear([inp] + [context_vector], input_size, True)                         │
│                                                                                   │
│                                                                                   │
│   WHAT THIS DOES:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Previous word embedding:  yₜ₋₁     ∈ ℝ^128                                     │
│   Previous context:         cₜ₋₁     ∈ ℝ^512                                     │
│   Concatenated:             [yₜ₋₁; cₜ₋₁] ∈ ℝ^640                                │
│   After linear:             x̃ₜ       ∈ ℝ^128  (same as embedding dim)           │
│                                                                                   │
│   x̃ₜ = W · [yₜ₋₁; cₜ₋₁] + b                                                     │
│                                                                                   │
│                                                                                   │
│   VISUALIZATION:                                                                  │
│   ──────────────                                                                  │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                      DECODER STEP t                                    │     │
│   │                                                                        │     │
│   │   Previous word: "Germany"                                            │     │
│   │   Previous context: [summary of what was attended at step t-1]        │     │
│   │                                                                        │     │
│   │   ┌─────────────────┐     ┌─────────────────┐                         │     │
│   │   │ y_{t-1}         │     │ c_{t-1}         │                         │     │
│   │   │ "Germany" emb   │     │ context vector  │                         │     │
│   │   │ [128 dims]      │     │ [512 dims]      │                         │     │
│   │   └────────┬────────┘     └────────┬────────┘                         │     │
│   │            │                       │                                   │     │
│   │            └───────────┬───────────┘                                   │     │
│   │                        │                                               │     │
│   │                        ▼                                               │     │
│   │            ┌───────────────────────┐                                   │     │
│   │            │    CONCATENATE        │                                   │     │
│   │            │    [128 + 512 = 640]  │                                   │     │
│   │            └───────────┬───────────┘                                   │     │
│   │                        │                                               │     │
│   │                        ▼                                               │     │
│   │            ┌───────────────────────┐                                   │     │
│   │            │    LINEAR LAYER       │                                   │     │
│   │            │    W: [640, 128]      │                                   │     │
│   │            └───────────┬───────────┘                                   │     │
│   │                        │                                               │     │
│   │                        ▼                                               │     │
│   │            ┌───────────────────────┐                                   │     │
│   │            │    x̃_t               │                                   │     │
│   │            │    [128 dims]         │                                   │     │
│   │            │                       │                                   │     │
│   │            │    "Enhanced input    │                                   │     │
│   │            │     with attention    │                                   │     │
│   │            │     history"          │                                   │     │
│   │            └───────────┬───────────┘                                   │     │
│   │                        │                                               │     │
│   │                        ▼                                               │     │
│   │            ┌───────────────────────┐                                   │     │
│   │            │    DECODER LSTM       │                                   │     │
│   │            │                       │                                   │     │
│   │            └───────────────────────┘                                   │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│                                                                                   │
│   WHY IS THIS IMPORTANT?                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   1. Alignment History:                                                           │
│      - Decoder knows what it attended to before                                  │
│      - Helps avoid repeating the same content                                    │
│                                                                                   │
│   2. Coverage Complement:                                                         │
│      - Works together with coverage mechanism                                    │
│      - Provides "soft" attention history even without coverage                   │
│                                                                                   │
│   3. Coherence:                                                                   │
│      - Previous focus informs current generation                                 │
│      - "I was talking about Germany, so I should continue that thought"          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Visualization Examples

### Example: Generating a Summary

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION VISUALIZATION EXAMPLE                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Source: "Germany emerged as the winners of the 2014 FIFA World Cup after       │
│            defeating Argentina 1-0 in the final at the Maracanã Stadium."        │
│                                                                                   │
│   Target: "Germany beat Argentina 1-0 to win the World Cup."                     │
│                                                                                   │
│                                                                                   │
│   STEP 1: Generate "Germany"                                                      │
│   ══════════════════════════                                                      │
│                                                                                   │
│   Source Words:                                                                   │
│   Germany emerged as the winners of the 2014 FIFA World Cup after defeating ...  │
│   ████▌   ▏      ▏  ▏   ▏       ▏  ▏   ▏    ▏    ▏     ▏   ▏     ▏          │
│   0.82   0.02  0.01 0.01 0.02  0.01 0.01 0.01 0.01 0.02 0.01 0.01 0.01      │
│                                                                                   │
│   Attention heavily focused on "Germany" → Will likely COPY                      │
│   p_gen ≈ 0.15 (low → copy mode)                                                 │
│                                                                                   │
│                                                                                   │
│   STEP 2: Generate "beat"                                                         │
│   ════════════════════════                                                        │
│                                                                                   │
│   Source Words:                                                                   │
│   Germany emerged as the winners of the 2014 FIFA World Cup after defeating ...  │
│   ▏       ▏      ▏  ▏   ██▌     ▏  ▏   ▏    ▏    ▏     ▏   ▏     █████     │
│   0.05   0.03  0.02 0.02 0.25  0.02 0.02 0.02 0.02 0.05 0.02 0.02 0.40      │
│                                                                                   │
│   Attention split between "winners" and "defeating"                              │
│   p_gen ≈ 0.72 (high → generate mode)                                            │
│   Model GENERATES "beat" (not in source but semantically similar)                │
│                                                                                   │
│                                                                                   │
│   STEP 3: Generate "Argentina"                                                    │
│   ════════════════════════════                                                    │
│                                                                                   │
│   Source Words:                                                                   │
│   Germany emerged as the winners of the 2014 FIFA World Cup after defeating ARG. │
│   ▏       ▏      ▏  ▏   ▏       ▏  ▏   ▏    ▏    ▏     ▏   ▏     ▏     ████│
│   0.02   0.01  0.01 0.01 0.02  0.01 0.01 0.01 0.01 0.02 0.01 0.01 0.02 0.80│
│                                                                                   │
│   Attention heavily focused on "Argentina" → Will COPY                           │
│   p_gen ≈ 0.08 (very low → copy mode)                                            │
│                                                                                   │
│                                                                                   │
│   STEP 4: Generate "1-0"                                                          │
│   ══════════════════════                                                          │
│                                                                                   │
│   Source Words:                                                                   │
│   ... defeating Argentina 1-0 in the final at the Maracanã Stadium.              │
│   ... ▏         ▏         ████ ▏  ▏   ▏     ▏  ▏   ▏        ▏       │
│   ... 0.02     0.05       0.75 0.02 0.02 0.02 0.02 0.02 0.02 0.02   │
│                                                                                   │
│   Attention focused on "1-0" → Will COPY (can't generate this!)                  │
│   p_gen ≈ 0.05 (very low → must copy this OOV number)                            │
│                                                                                   │
│                                                                                   │
│   FINAL SUMMARY                                                                   │
│   ═════════════                                                                   │
│                                                                                   │
│   "Germany beat Argentina 1-0 to win the World Cup."                             │
│    ─────── ──── ───────── ─── ── ─── ─── ───── ────                              │
│    COPIED  GEN  COPIED    CPY GN GEN GEN COPD  COPD                              │
│                                                                                   │
│   Legend:                                                                         │
│   COPIED = Directly copied from source via pointer                               │
│   GEN = Generated from vocabulary                                                │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Attention Heatmap

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        ATTENTION HEATMAP                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│                          SOURCE WORDS (truncated)                                 │
│              Germany emerged winners defeating Argentina 1-0 World Cup           │
│   ┌──────────────────────────────────────────────────────────────────┐          │
│   │                                                                   │          │
│ O │ Germany    █████  ░░░░   ░░░░    ░░░░     ░░░░   ░░░░  ░░░░ ░░░░│          │
│ U │ beat       ░░░░   ░░░░   ████    █████    ░░░░   ░░░░  ░░░░ ░░░░│          │
│ T │ Argentina  ░░░░   ░░░░   ░░░░    ░░░░     █████  ░░░░  ░░░░ ░░░░│          │
│ P │ 1-0        ░░░░   ░░░░   ░░░░    ░░░░     ░░░░   █████ ░░░░ ░░░░│          │
│ U │ to         ░░░░   ░░░░   ░░░░    ░░░░     ░░░░   ░░░░  ░░░░ ░░░░│          │
│ T │ win        ░░░░   ░░░░   ████    ░░░░     ░░░░   ░░░░  ░░░░ ░░░░│          │
│   │ the        ░░░░   ░░░░   ░░░░    ░░░░     ░░░░   ░░░░  ████ ████│          │
│ W │ World      ░░░░   ░░░░   ░░░░    ░░░░     ░░░░   ░░░░  █████ ░░░│          │
│ O │ Cup        ░░░░   ░░░░   ░░░░    ░░░░     ░░░░   ░░░░  ░░░░ █████│          │
│ R │ .          ░░░░   ░░░░   ░░░░    ░░░░     ░░░░   ░░░░  ░░░░ ░░░░│          │
│ D │                                                                   │          │
│ S │                                                                   │          │
│   └──────────────────────────────────────────────────────────────────┘          │
│                                                                                   │
│   ████ = High attention (> 0.5)                                                  │
│   ░░░░ = Low attention (< 0.1)                                                   │
│                                                                                   │
│   Key observations:                                                               │
│   1. Diagonal pattern: Output word often attends to aligned source word          │
│   2. "beat" attends to "winners" AND "defeating" (needs both concepts)           │
│   3. "World" and "Cup" attend to corresponding source words                      │
│   4. Proper nouns (Germany, Argentina, 1-0) have very focused attention          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

The attention mechanism in the Pointer-Generator Network:

1. **Uses Bahdanau (additive) attention** for computing relevance scores
2. **Pre-computes encoder features** for efficiency
3. **Applies masking** to prevent attending to padding tokens
4. **Computes context vector** as a weighted sum of encoder states
5. **Feeds previous context** to the decoder input for coherence
6. **Enables copying** by using attention weights as copy probabilities

The attention mechanism is the foundation that makes both the pointer mechanism and the coverage mechanism possible.

---

*Next: [05_pointer_mechanism.md](05_pointer_mechanism.md) - Pointer Mechanism and Copy Distribution*
