# attention_decoder.py Line-by-Line Walkthrough

## Table of Contents
1. [File Overview](#file-overview)
2. [Function Signature](#function-signature)
3. [Pre-computed Encoder Features](#pre-computed-encoder-features)
4. [Coverage Variables](#coverage-variables)
5. [Attention Function](#attention-function)
6. [Decoder Loop](#decoder-loop)
7. [P_gen Calculation](#p_gen-calculation)
8. [Return Values](#return-values)
9. [Complete Code Reference](#complete-code-reference)

---

## File Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION_DECODER.PY OVERVIEW                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FILE: pointer-generator/attention_decoder.py                                   │
│   LINES: ~229                                                                    │
│   PURPOSE: Implements the attention mechanism and decoder loop                   │
│                                                                                   │
│   STRUCTURE:                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   Lines 1-20:     Imports                                                        │
│   Lines 21-45:    Function signature and docstring                              │
│   Lines 46-70:    Variable setup and pre-computation                            │
│   Lines 71-85:    Coverage variable initialization                              │
│   Lines 86-130:   Attention function definition                                 │
│   Lines 131-180:  Decoder loop                                                  │
│   Lines 181-215:  P_gen calculation                                             │
│   Lines 216-229:  Return values                                                 │
│                                                                                   │
│                                                                                   │
│   KEY CONCEPTS:                                                                   │
│   ──────────────                                                                  │
│                                                                                   │
│   1. Pre-computed encoder features (for efficiency)                              │
│   2. Bahdanau attention mechanism                                                │
│   3. Input feeding (previous context to decoder)                                 │
│   4. Coverage mechanism                                                           │
│   5. Pointer-generator (p_gen) calculation                                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Function Signature

```python
# attention_decoder.py: Lines 21-45

def attention_decoder(decoder_inputs, initial_state, encoder_states,
                      enc_padding_mask, cell, initial_state_attention=False,
                      pointer_gen=True, use_coverage=False, prev_coverage=None):
    """
    Attention decoder with pointer-generator option.
    
    This function implements a decoder with Bahdanau attention and optional
    pointer-generator mechanism.
    
    Args:
        decoder_inputs: 
            Decoder input embeddings
            Shape: [batch, max_dec_steps, emb_dim]
        
        initial_state:
            Initial decoder state from encoder
            LSTMStateTuple(c, h) each with shape [batch, hidden_dim]
        
        encoder_states:
            Encoder outputs for attention
            Shape: [batch, enc_len, 2*hidden_dim]
        
        enc_padding_mask:
            Mask for encoder padding (1=real, 0=padding)
            Shape: [batch, enc_len]
        
        cell:
            LSTM cell for decoder
        
        initial_state_attention:
            If True, compute attention at step 0 (for decode mode)
            If False, use zeros for step 0 context
        
        pointer_gen:
            If True, compute p_gen for pointer mechanism
        
        use_coverage:
            If True, use coverage mechanism
        
        prev_coverage:
            Previous coverage vector (for decode mode)
            Shape: [batch, enc_len]
    
    Returns:
        outputs: List of decoder outputs [batch, hidden_dim] per step
        state: Final decoder state
        attn_dists: List of attention distributions [batch, enc_len] per step
        p_gens: List of p_gen values [batch, 1] per step (if pointer_gen)
        coverage: Final coverage vector [batch, enc_len] (if use_coverage)
    """
```

**Parameter Summary:**

| Parameter | Shape | Description |
|-----------|-------|-------------|
| `decoder_inputs` | [B, T, E] | Embedded decoder inputs |
| `initial_state` | (c, h) each [B, H] | Initial decoder state |
| `encoder_states` | [B, L, 2H] | Encoder hidden states |
| `enc_padding_mask` | [B, L] | Mask for attention |
| `cell` | LSTM cell | Decoder RNN cell |
| `prev_coverage` | [B, L] | Previous coverage (decode) |

Where B=batch, T=dec_steps, E=emb_dim, L=enc_len, H=hidden_dim

---

## Pre-computed Encoder Features

```python
# attention_decoder.py: Lines 46-70

with variable_scope("attention_decoder"):
    batch_size = encoder_states.get_shape()[0].value
    attn_size = encoder_states.get_shape()[2].value  # 2*hidden_dim
    
    # ═══════════════════════════════════════════════════════════════════════
    # PRE-COMPUTE ENCODER FEATURES
    # ═══════════════════════════════════════════════════════════════════════
    
    # Reshape encoder states for conv2d
    # [batch, enc_len, 1, attn_size]
    encoder_states_reshaped = tf.expand_dims(encoder_states, axis=2)
    
    # Create attention weights
    W_h = tf.get_variable(
        "W_h",
        [1, 1, attn_size, attn_size],  # Conv2d kernel
        dtype=tf.float32
    )
    
    # Pre-compute W_h × encoder_states using conv2d
    # This is computed ONCE and reused at every decoder step
    encoder_features = tf.nn.conv2d(
        encoder_states_reshaped,
        W_h,
        [1, 1, 1, 1],  # Strides
        "SAME"
    )  # [batch, enc_len, 1, attn_size]
```

**Why Pre-compute?**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  PRE-COMPUTATION OPTIMIZATION                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Attention formula:                                                              │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + b)                                  │
│                 ↑                                                                 │
│            This part depends ONLY on encoder states (h_i)                       │
│            It's the SAME for all decoder steps!                                 │
│                                                                                   │
│   Without pre-computation:                                                        │
│   ─────────────────────────                                                       │
│   Step 1: compute W_h · h for all h_i  (O(L × D²))                              │
│   Step 2: compute W_h · h for all h_i  (O(L × D²)) ← REPEATED!                 │
│   Step 3: compute W_h · h for all h_i  (O(L × D²)) ← REPEATED!                 │
│   ...                                                                             │
│   Total: O(T × L × D²) where T = decoder steps                                  │
│                                                                                   │
│   With pre-computation:                                                           │
│   ──────────────────────                                                          │
│   Setup: compute W_h · h for all h_i ONCE  (O(L × D²))                         │
│   Step 1: use pre-computed                   (O(L × D))                         │
│   Step 2: use pre-computed                   (O(L × D))                         │
│   Step 3: use pre-computed                   (O(L × D))                         │
│   ...                                                                             │
│   Total: O(L × D²) + O(T × L × D)                                               │
│                                                                                   │
│   This is a significant speedup, especially for long sequences!                 │
│                                                                                   │
│                                                                                   │
│   Why conv2d instead of matmul?                                                  │
│   ─────────────────────────────                                                   │
│                                                                                   │
│   Conv2d with 1×1 kernel is equivalent to matmul but:                           │
│   • Works on 4D tensors directly (no reshape needed)                            │
│   • May be more optimized on GPU                                                │
│   • Cleaner code when dealing with batch + sequence dimensions                  │
│                                                                                   │
│   Shape transformation:                                                           │
│   encoder_states:     [batch, enc_len, 2*hidden_dim]                            │
│   → expand_dims:      [batch, enc_len, 1, 2*hidden_dim]                         │
│   → conv2d(W_h):      [batch, enc_len, 1, attn_size]                            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coverage Variables

```python
# attention_decoder.py: Lines 71-85

# ═══════════════════════════════════════════════════════════════════════
# COVERAGE VARIABLES
# ═══════════════════════════════════════════════════════════════════════

if use_coverage:
    with variable_scope("coverage"):
        # Coverage weight: projects coverage to attention size
        w_c = tf.get_variable(
            "w_c",
            [1, 1, 1, attn_size],  # Conv2d kernel shape
            dtype=tf.float32
        )

# ═══════════════════════════════════════════════════════════════════════
# ATTENTION WEIGHTS
# ═══════════════════════════════════════════════════════════════════════

# v vector for computing attention scores
v = tf.get_variable(
    "v",
    [attn_size],
    dtype=tf.float32
)

# W_s for decoder state projection
W_s = tf.get_variable(
    "W_s",
    [attn_size, attn_size],  # [hidden_dim, attn_size] when decoder is hidden_dim
    dtype=tf.float32
)

# Bias
b_attn = tf.get_variable(
    "b_attn",
    [attn_size],
    dtype=tf.float32
)
```

**Variable Summary:**

| Variable | Shape | Purpose |
|----------|-------|---------|
| `W_h` | [1,1,2H,A] | Project encoder states |
| `W_s` | [H,A] | Project decoder state |
| `w_c` | [1,1,1,A] | Project coverage |
| `v` | [A] | Score vector |
| `b_attn` | [A] | Attention bias |

Where A = attn_size = 2*hidden_dim

---

## Attention Function

```python
# attention_decoder.py: Lines 86-130

def attention(decoder_state, coverage=None):
    """
    Compute attention distribution over encoder states.
    
    Implements Bahdanau attention:
    e_ti = v^T · tanh(W_h·h_i + W_s·s_t + w_c·c_ti + b)
    α_ti = softmax(e_ti)
    
    Args:
        decoder_state: Current decoder hidden state [batch, hidden_dim]
        coverage: Coverage vector [batch, enc_len] or None
    
    Returns:
        attn_dist: Attention distribution [batch, enc_len]
        context: Context vector [batch, 2*hidden_dim]
        coverage: Updated coverage [batch, enc_len]
    """
    with variable_scope("Attention"):
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE DECODER STATE FEATURES
        # ═══════════════════════════════════════════════════════════════════
        
        # Project decoder state: W_s · s_t
        # [batch, hidden_dim] @ [hidden_dim, attn_size] = [batch, attn_size]
        decoder_features = tf.nn.xw_plus_b(decoder_state, W_s, b_attn)
        
        # Expand for broadcasting with encoder features
        # [batch, attn_size] → [batch, 1, 1, attn_size]
        decoder_features = tf.expand_dims(
            tf.expand_dims(decoder_features, 1), 1
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # ADD COVERAGE FEATURES (if enabled)
        # ═══════════════════════════════════════════════════════════════════
        
        if use_coverage and coverage is not None:
            # Reshape coverage: [batch, enc_len] → [batch, enc_len, 1, 1]
            coverage_reshaped = tf.expand_dims(
                tf.expand_dims(coverage, 2), 3
            )
            
            # Project coverage: w_c · c_ti
            coverage_features = tf.nn.conv2d(
                coverage_reshaped,
                w_c,
                [1, 1, 1, 1],
                "SAME"
            )  # [batch, enc_len, 1, attn_size]
            
            # Combine all features
            e = tf.reduce_sum(
                v * tf.tanh(encoder_features + decoder_features + coverage_features),
                [2, 3]
            )  # [batch, enc_len]
        else:
            # Without coverage
            e = tf.reduce_sum(
                v * tf.tanh(encoder_features + decoder_features),
                [2, 3]
            )  # [batch, enc_len]
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE ATTENTION DISTRIBUTION
        # ═══════════════════════════════════════════════════════════════════
        
        # Softmax over encoder positions
        attn_dist = tf.nn.softmax(e)  # [batch, enc_len]
        
        # Apply padding mask (zero out padding positions)
        attn_dist *= enc_padding_mask
        
        # Renormalize (since masked positions now have zero probability)
        masked_sums = tf.reduce_sum(attn_dist, axis=1, keepdims=True)
        attn_dist = attn_dist / masked_sums
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE CONTEXT VECTOR
        # ═══════════════════════════════════════════════════════════════════
        
        # Weighted sum of encoder states
        # attn_dist: [batch, enc_len]
        # encoder_states: [batch, enc_len, 2*hidden_dim]
        # context: [batch, 2*hidden_dim]
        
        context = tf.reduce_sum(
            tf.expand_dims(attn_dist, 2) * encoder_states,
            axis=1
        )  # [batch, 2*hidden_dim]
        
        # ═══════════════════════════════════════════════════════════════════
        # UPDATE COVERAGE
        # ═══════════════════════════════════════════════════════════════════
        
        if use_coverage:
            coverage = coverage + attn_dist  # Accumulate attention
        
        return attn_dist, context, coverage
```

**Attention Computation Visual:**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION COMPUTATION FLOW                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   encoder_features (pre-computed):    [batch, enc_len, 1, attn_size]            │
│   decoder_state:                      [batch, hidden_dim]                        │
│   coverage:                           [batch, enc_len]                           │
│                                                                                   │
│   Step 1: Project decoder state                                                  │
│   ─────────────────────────────────                                               │
│   decoder_features = W_s × decoder_state + b                                    │
│   → [batch, attn_size]                                                           │
│   → expand to [batch, 1, 1, attn_size] for broadcasting                         │
│                                                                                   │
│   Step 2: Project coverage (if enabled)                                          │
│   ──────────────────────────────────────                                          │
│   coverage_features = conv2d(coverage, w_c)                                      │
│   → [batch, enc_len, 1, attn_size]                                              │
│                                                                                   │
│   Step 3: Combine and score                                                      │
│   ─────────────────────────                                                       │
│   combined = encoder_features + decoder_features + coverage_features            │
│   → [batch, enc_len, 1, attn_size] (broadcasting!)                              │
│                                                                                   │
│   scores = v × tanh(combined)                                                   │
│   → [batch, enc_len, 1, attn_size]                                              │
│                                                                                   │
│   e = sum over last 2 dims                                                       │
│   → [batch, enc_len]                                                             │
│                                                                                   │
│   Step 4: Softmax with masking                                                   │
│   ─────────────────────────────                                                   │
│   attn_dist = softmax(e) × mask                                                 │
│   attn_dist = attn_dist / sum(attn_dist)  # renormalize                         │
│   → [batch, enc_len]                                                             │
│                                                                                   │
│   Step 5: Compute context                                                        │
│   ───────────────────────                                                         │
│   context = sum(attn_dist × encoder_states, axis=1)                             │
│   → [batch, 2*hidden_dim]                                                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Decoder Loop

```python
# attention_decoder.py: Lines 131-180

# ═══════════════════════════════════════════════════════════════════════════════
# INITIALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

outputs = []          # Decoder outputs at each step
attn_dists = []       # Attention distributions at each step
p_gens = []           # p_gen values at each step
state = initial_state # Current decoder state

# Initialize coverage
if use_coverage:
    if prev_coverage is not None:
        coverage = prev_coverage  # Use provided (for decode mode)
    else:
        coverage = tf.zeros([batch_size, tf.shape(encoder_states)[1]])
else:
    coverage = None

# Initialize context vector
if initial_state_attention:
    # Compute attention with initial state (for decode mode)
    attn_dist, context_vector, coverage = attention(initial_state.h, coverage)
else:
    # Use zero context for first step (training)
    context_vector = tf.zeros([batch_size, attn_size])
    attn_dist = tf.zeros([batch_size, tf.shape(encoder_states)[1]])

# ═══════════════════════════════════════════════════════════════════════════════
# DECODER LOOP
# ═══════════════════════════════════════════════════════════════════════════════

for i, inp in enumerate(tf.unstack(decoder_inputs, axis=1)):
    # inp: [batch, emb_dim] - current input embedding
    
    # ═══════════════════════════════════════════════════════════════════════
    # INPUT FEEDING: Concatenate input with previous context
    # ═══════════════════════════════════════════════════════════════════════
    
    # inp: [batch, emb_dim]
    # context_vector: [batch, 2*hidden_dim]
    # x: [batch, emb_dim + 2*hidden_dim]
    x = tf.concat([inp, context_vector], axis=1)
    
    # ═══════════════════════════════════════════════════════════════════════
    # RUN LSTM CELL
    # ═══════════════════════════════════════════════════════════════════════
    
    # cell_output: [batch, hidden_dim]
    # state: LSTMStateTuple(c, h) each [batch, hidden_dim]
    cell_output, state = cell(x, state)
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE ATTENTION
    # ═══════════════════════════════════════════════════════════════════════
    
    # Reuse variables from first iteration
    if i > 0:
        variable_scope.get_variable_scope().reuse_variables()
    
    # Compute attention over encoder states
    attn_dist, context_vector, coverage = attention(state.h, coverage)
    
    # ═══════════════════════════════════════════════════════════════════════
    # STORE OUTPUTS
    # ═══════════════════════════════════════════════════════════════════════
    
    attn_dists.append(attn_dist)
    
    # Combine cell output with context for final output
    # This is passed to the output projection layer
    with variable_scope("AttnOutputProjection"):
        output = linear([cell_output, context_vector], cell.output_size, True)
    
    outputs.append(output)
```

**Decoder Step Visualization:**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      DECODER STEP FLOW                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   For each time step i:                                                           │
│                                                                                   │
│   ┌─────────────────┐   ┌──────────────────┐                                    │
│   │  inp (embed)    │   │ prev_context     │                                    │
│   │  [B, emb_dim]   │   │ [B, 2*hidden]    │                                    │
│   └────────┬────────┘   └────────┬─────────┘                                    │
│            │                     │                                               │
│            └──────────┬──────────┘                                               │
│                       │ concat                                                    │
│                       ▼                                                           │
│            ┌─────────────────────┐                                               │
│            │    x = [inp, ctx]   │                                               │
│            │  [B, emb+2*hidden]  │                                               │
│            └──────────┬──────────┘                                               │
│                       │                                                           │
│                       ▼                                                           │
│            ┌─────────────────────┐      ┌──────────────────┐                    │
│            │     LSTM Cell       │◀─────│   prev_state     │                    │
│            │                     │      │   (c, h)         │                    │
│            └──────────┬──────────┘      └──────────────────┘                    │
│                       │                                                           │
│            ┌──────────┴──────────┐                                               │
│            │                     │                                               │
│            ▼                     ▼                                               │
│   ┌─────────────────┐   ┌─────────────────┐                                     │
│   │  cell_output    │   │   new_state     │                                     │
│   │  [B, hidden]    │   │   (c, h)        │                                     │
│   └────────┬────────┘   └────────┬────────┘                                     │
│            │                     │                                               │
│            │                     ▼                                               │
│            │            ┌─────────────────┐                                      │
│            │            │   Attention     │                                      │
│            │            │                 │                                      │
│            │            │ encoder_states  │                                      │
│            │            │ coverage        │                                      │
│            │            └────────┬────────┘                                      │
│            │                     │                                               │
│            │            ┌────────┴────────┐                                      │
│            │            │                 │                                      │
│            │            ▼                 ▼                                      │
│            │   ┌─────────────────┐ ┌───────────┐                                │
│            │   │  new_context    │ │ attn_dist │                                │
│            │   │  [B, 2*hidden]  │ │ [B, L]    │                                │
│            │   └────────┬────────┘ └─────┬─────┘                                │
│            │            │                │                                       │
│            └────────────┼────────────────┘                                       │
│                         │ concat + linear                                        │
│                         ▼                                                        │
│              ┌─────────────────────┐                                             │
│              │      output         │                                             │
│              │    [B, hidden]      │                                             │
│              └─────────────────────┘                                             │
│                         │                                                        │
│                         ▼                                                        │
│              (to output projection layer)                                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## P_gen Calculation

```python
# attention_decoder.py: Lines 181-215

if pointer_gen:
    # ═══════════════════════════════════════════════════════════════════════
    # P_GEN CALCULATION
    # ═══════════════════════════════════════════════════════════════════════
    #
    # p_gen determines how much to generate vs copy:
    # p_gen = sigmoid(w_c·context + w_s·state + w_x·input + b)
    #
    # p_gen close to 1: generate from vocabulary
    # p_gen close to 0: copy from source
    
    with variable_scope("calculate_pgen"):
        # Weight for context vector
        w_c_pgen = tf.get_variable(
            "w_c",
            [attn_size, 1],
            dtype=tf.float32
        )
        
        # Weight for decoder state
        w_s_pgen = tf.get_variable(
            "w_s",
            [cell.state_size.h, 1],  # hidden_dim
            dtype=tf.float32
        )
        
        # Weight for input embedding
        w_x_pgen = tf.get_variable(
            "w_x",
            [cell.input_size - attn_size, 1],  # emb_dim (input minus context)
            dtype=tf.float32
        )
        
        # Bias
        b_pgen = tf.get_variable(
            "b_pgen",
            [1],
            dtype=tf.float32
        )
    
    # Calculate p_gen for each decoder step
    for i, (context, state_h, inp) in enumerate(zip(
        context_vectors,  # List of context vectors
        states_h,         # List of decoder hidden states
        tf.unstack(decoder_inputs, axis=1)  # Input embeddings
    )):
        # context: [batch, 2*hidden_dim]
        # state_h: [batch, hidden_dim]
        # inp: [batch, emb_dim]
        
        # Compute p_gen
        p_gen = tf.nn.sigmoid(
            tf.matmul(context, w_c_pgen) +    # Context contribution
            tf.matmul(state_h, w_s_pgen) +    # State contribution
            tf.matmul(inp, w_x_pgen) +        # Input contribution
            b_pgen                             # Bias
        )  # [batch, 1]
        
        p_gens.append(p_gen)
```

**P_gen Intuition:**

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         P_GEN INTUITION                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   p_gen = σ(w_c·context + w_s·state + w_x·input + b)                            │
│                                                                                   │
│   WHAT EACH TERM CAPTURES:                                                        │
│   ────────────────────────                                                        │
│                                                                                   │
│   w_c·context (context contribution):                                            │
│   • Context vector summarizes what the model is attending to                     │
│   • If attending to a rare/OOV word → might want to copy (low p_gen)           │
│   • If attending to common content → might want to generate (high p_gen)        │
│                                                                                   │
│   w_s·state (state contribution):                                                │
│   • Decoder state captures what's been generated so far                         │
│   • Certain states might favor copying vs generating                            │
│   • Example: after generating "said", might copy the name that follows         │
│                                                                                   │
│   w_x·input (input contribution):                                               │
│   • Current input token influences the decision                                  │
│   • If previous token was a common word → might continue generating            │
│   • If previous token was copied → might continue copying                       │
│                                                                                   │
│                                                                                   │
│   EXAMPLE SCENARIOS:                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   Source: "Elon Musk announced a new product"                                    │
│                                                                                   │
│   Generating "said":                                                              │
│   • context points to "announced" (common paraphrase)                           │
│   • state: just started the sentence                                            │
│   • p_gen ≈ 0.8 → GENERATE from vocabulary                                     │
│                                                                                   │
│   Generating "Musk":                                                              │
│   • context points to "Musk" (rare name)                                        │
│   • state: just generated a name-like context                                   │
│   • p_gen ≈ 0.2 → COPY from source                                             │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Return Values

```python
# attention_decoder.py: Lines 216-229

# ═══════════════════════════════════════════════════════════════════════════════
# RETURN VALUES
# ═══════════════════════════════════════════════════════════════════════════════

return outputs, state, attn_dists, p_gens, coverage

# outputs:
#   List of decoder outputs, one per time step
#   Each element: [batch, hidden_dim]
#   Length: max_dec_steps
#   Used for: output projection to vocabulary
#
# state:
#   Final decoder state
#   LSTMStateTuple(c, h)
#   Used for: beam search (continue from this state)
#
# attn_dists:
#   List of attention distributions, one per time step
#   Each element: [batch, enc_len]
#   Length: max_dec_steps
#   Used for: copy mechanism, visualization, coverage
#
# p_gens:
#   List of generation probabilities, one per time step
#   Each element: [batch, 1]
#   Length: max_dec_steps
#   Used for: combining vocab and copy distributions
#
# coverage:
#   Final coverage vector
#   Shape: [batch, enc_len]
#   Used for: coverage loss, beam search continuation
```

**Return Shape Summary:**

| Return | Type | Shape per element | Description |
|--------|------|-------------------|-------------|
| `outputs` | List[Tensor] | [B, H] | Decoder outputs |
| `state` | LSTMStateTuple | (c,h) [B, H] | Final state |
| `attn_dists` | List[Tensor] | [B, L] | Attention weights |
| `p_gens` | List[Tensor] | [B, 1] | Gen/copy switch |
| `coverage` | Tensor | [B, L] | Cumulative attention |

---

## Complete Code Reference

```python
# attention_decoder.py - Complete annotated version

"""Attention decoder for pointer-generator network."""

import tensorflow as tf
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import math_ops


def attention_decoder(decoder_inputs,        # [B, T, E] decoder embeddings
                      initial_state,          # (c, h) initial decoder state
                      encoder_states,         # [B, L, 2H] encoder outputs
                      enc_padding_mask,       # [B, L] padding mask
                      cell,                   # LSTM cell
                      initial_state_attention=False,  # Attend at step 0?
                      pointer_gen=True,       # Use pointer mechanism?
                      use_coverage=False,     # Use coverage?
                      prev_coverage=None):    # Previous coverage (decode)
    """
    Attention decoder with pointer-generator.
    
    Returns:
        outputs: List of [B, H] decoder outputs
        state: Final decoder state
        attn_dists: List of [B, L] attention distributions
        p_gens: List of [B, 1] generation probabilities
        coverage: [B, L] final coverage vector
    """
    
    with variable_scope.variable_scope("attention_decoder"):
        # Get dimensions
        batch_size = encoder_states.get_shape()[0].value
        attn_size = encoder_states.get_shape()[2].value  # 2*hidden_dim
        
        # === PRE-COMPUTE ENCODER FEATURES (for efficiency) ===
        # Shape: [B, L, 1, A]
        encoder_states_4d = tf.expand_dims(encoder_states, axis=2)
        
        W_h = tf.get_variable("W_h", [1, 1, attn_size, attn_size])
        encoder_features = nn_ops.conv2d(encoder_states_4d, W_h, [1,1,1,1], "SAME")
        
        # === COVERAGE VARIABLE ===
        if use_coverage:
            with variable_scope.variable_scope("coverage"):
                w_c = tf.get_variable("w_c", [1, 1, 1, attn_size])
        
        # === ATTENTION VARIABLES ===
        v = tf.get_variable("v", [attn_size])
        
        # === ATTENTION FUNCTION ===
        def attention(decoder_state, coverage=None):
            """
            Compute Bahdanau attention.
            
            e_ti = v^T tanh(W_h·h_i + W_s·s_t + w_c·c_ti + b)
            α = softmax(e)
            context = Σ α_i · h_i
            """
            with variable_scope.variable_scope("Attention"):
                # Decoder state features: [B, 1, 1, A]
                decoder_features = linear(decoder_state, attn_size, True)
                decoder_features = tf.expand_dims(tf.expand_dims(decoder_features, 1), 1)
                
                # Compute attention scores
                if use_coverage and coverage is not None:
                    # With coverage
                    coverage_4d = tf.expand_dims(tf.expand_dims(coverage, 2), 3)
                    coverage_features = nn_ops.conv2d(coverage_4d, w_c, [1,1,1,1], "SAME")
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features + coverage_features),
                        [2, 3]
                    )
                else:
                    # Without coverage
                    e = math_ops.reduce_sum(
                        v * math_ops.tanh(encoder_features + decoder_features),
                        [2, 3]
                    )
                
                # Softmax with masking
                attn_dist = nn_ops.softmax(e)
                attn_dist *= enc_padding_mask
                masked_sums = math_ops.reduce_sum(attn_dist, axis=1, keep_dims=True)
                attn_dist = attn_dist / masked_sums
                
                # Context vector
                context = math_ops.reduce_sum(
                    array_ops.expand_dims(attn_dist, 2) * encoder_states,
                    axis=1
                )
                
                # Update coverage
                if use_coverage:
                    coverage = coverage + attn_dist
                
                return attn_dist, context, coverage
        
        # === INITIALIZE ===
        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        context_vector = tf.zeros([batch_size, attn_size])
        
        if use_coverage and prev_coverage is not None:
            coverage = prev_coverage
        elif use_coverage:
            coverage = tf.zeros([batch_size, tf.shape(encoder_states)[1]])
        else:
            coverage = None
        
        if initial_state_attention:
            _, context_vector, coverage = attention(initial_state.h, coverage)
        
        # === DECODER LOOP ===
        for i, inp in enumerate(tf.unstack(decoder_inputs, axis=1)):
            if i > 0:
                variable_scope.get_variable_scope().reuse_variables()
            
            # Input feeding: concat input with context
            x = array_ops.concat([inp, context_vector], 1)
            
            # LSTM step
            cell_output, state = cell(x, state)
            
            # Attention
            attn_dist, context_vector, coverage = attention(state.h, coverage)
            attn_dists.append(attn_dist)
            
            # Output projection
            with variable_scope.variable_scope("AttnOutputProjection"):
                output = linear([cell_output, context_vector], cell.output_size, True)
            outputs.append(output)
            
            # P_gen calculation
            if pointer_gen:
                with variable_scope.variable_scope("calculate_pgen"):
                    p_gen = nn_ops.sigmoid(
                        linear([context_vector, state.h, inp], 1, True)
                    )
                p_gens.append(p_gen)
        
        return outputs, state, attn_dists, p_gens, coverage
```

---

## Summary

**attention_decoder.py** implements:

| Component | Lines | Description |
|-----------|-------|-------------|
| Pre-computation | 46-70 | W_h × encoder_states once |
| Coverage vars | 71-85 | w_c for coverage features |
| Attention func | 86-130 | Bahdanau attention |
| Decoder loop | 131-180 | LSTM + attention per step |
| P_gen calc | 181-215 | Generate vs copy decision |
| Returns | 216-229 | outputs, state, attn, p_gen, cov |

Key design decisions:
1. **Pre-compute encoder features** for O(L) vs O(T×L) speedup
2. **Input feeding**: concat context with input
3. **Conv2d for projections**: cleaner 4D tensor handling
4. **Masked softmax**: handle variable-length sequences
5. **Coverage accumulation**: track attention history

---

*Next: [14_running_example.md](14_running_example.md) - End-to-End Worked Example*
