# Mathematical Formulations

## Table of Contents
1. [Notation Reference](#notation-reference)
2. [Embedding Formulations](#embedding-formulations)
3. [Encoder Formulations](#encoder-formulations)
4. [Attention Formulations](#attention-formulations)
5. [Gate Mechanism](#gate-mechanism)
6. [Final Distribution](#final-distribution)
7. [Loss Functions](#loss-functions)
8. [Complete Forward Pass](#complete-forward-pass)

---

## Notation Reference

### Common Symbols

| Symbol | Description | Original | Proposed |
|--------|-------------|----------|----------|
| $B$ | Batch size | 16 | 128 |
| $T$ | Sequence length (encoder) | ≤400 | ≤50 |
| $T_{dec}$ | Sequence length (decoder) | ≤100 | 1 |
| $d$ | Hidden dimension | 256 | 64 |
| $d_{emb}$ | Embedding dimension | 128 | 64 |
| $V$ | Vocabulary size | 50,000 | ~500 |
| $h$ | Number of attention heads | 1 | 4 |
| $L$ | Number of encoder layers | 1 | 2 |

### Tensor Shapes

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TENSOR SHAPE NOTATION                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  [B, T, d]  = Batch × Sequence × Hidden dimension                           │
│  [B, T]     = Batch × Sequence (e.g., attention weights)                    │
│  [B, V]     = Batch × Vocabulary (output distribution)                      │
│  [d, d]     = Square weight matrix                                          │
│  [d]        = Bias vector                                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Embedding Formulations

### Original: Word Embedding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL EMBEDDING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given:                                                                      │
│    - Input word IDs: x = [x₁, x₂, ..., xₜ] where xᵢ ∈ {0, 1, ..., V-1}    │
│    - Embedding matrix: E ∈ ℝ^{V × d_emb}                                    │
│                                                                              │
│  Embedding lookup:                                                           │
│                                                                              │
│    e_i = E[x_i, :]    ∈ ℝ^{d_emb}                                          │
│                                                                              │
│  Result:                                                                     │
│    E_input = [e₁, e₂, ..., eₜ]    ∈ ℝ^{T × d_emb}                         │
│                                                                              │
│  Parameters:                                                                 │
│    E ∈ ℝ^{50000 × 128} = 6,400,000 parameters                              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Multi-Modal Embedding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED EMBEDDING                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given:                                                                      │
│    - Location IDs: loc = [loc₁, loc₂, ..., locₜ]                           │
│    - User IDs: user = [u₁, u₂, ..., uₜ]                                    │
│    - Time indices: time = [t₁, t₂, ..., tₜ]                                │
│    - Weekdays: week = [w₁, w₂, ..., wₜ]                                    │
│    - Durations: dur = [dur₁, dur₂, ..., durₜ]                              │
│    - Recency: rec = [rec₁, rec₂, ..., recₜ]                                │
│    - Position from end: pos = [T-1, T-2, ..., 0]                           │
│                                                                              │
│  Embedding matrices:                                                         │
│    E_loc ∈ ℝ^{N_loc × d}     (location)                                    │
│    E_user ∈ ℝ^{N_user × d}   (user)                                        │
│    E_time ∈ ℝ^{24 × d}       (hour of day)                                 │
│    E_week ∈ ℝ^{7 × d}        (day of week)                                 │
│    E_dur ∈ ℝ^{N_dur × d}     (duration bins)                               │
│    E_rec ∈ ℝ^{N_rec × d}     (recency bins)                                │
│    E_pos ∈ ℝ^{max_seq × d}   (position from end)                           │
│                                                                              │
│  Combined embedding (element-wise sum):                                      │
│                                                                              │
│    e_i = E_loc[loc_i] + E_user[u_i] + E_time[t_i] + E_week[w_i]           │
│          + E_dur[dur_i] + E_rec[rec_i] + E_pos[T-i]                        │
│                                                                              │
│  Normalized:                                                                 │
│                                                                              │
│    ê_i = LayerNorm(e_i)                                                     │
│                                                                              │
│  Result:                                                                     │
│    E_input = [ê₁, ê₂, ..., êₜ]    ∈ ℝ^{T × d}                             │
│                                                                              │
│  Parameters:                                                                 │
│    ~56,000 total (500×64 + 200×64 + 24×64 + 7×64 + 20×64 + 20×64 + 100×64)│
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Encoder Formulations

### Original: Bidirectional LSTM

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: BiLSTM ENCODER                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LSTM Cell Equations:                                                        │
│                                                                              │
│    Forget gate:   f_t = σ(W_f · [h_{t-1}, x_t] + b_f)                      │
│    Input gate:    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)                      │
│    Candidate:     c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)                   │
│    Cell state:    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t                         │
│    Output gate:   o_t = σ(W_o · [h_{t-1}, x_t] + b_o)                      │
│    Hidden state:  h_t = o_t ⊙ tanh(c_t)                                     │
│                                                                              │
│  Where:                                                                      │
│    σ = sigmoid function                                                     │
│    ⊙ = element-wise multiplication                                         │
│    W_f, W_i, W_c, W_o ∈ ℝ^{d × (d + d_emb)}                                │
│    b_f, b_i, b_c, b_o ∈ ℝ^d                                                 │
│                                                                              │
│  Bidirectional Processing:                                                  │
│                                                                              │
│    Forward:  →h_t = LSTM_fw(→h_{t-1}, e_t)    for t = 1, 2, ..., T        │
│    Backward: ←h_t = LSTM_bw(←h_{t+1}, e_t)    for t = T, T-1, ..., 1      │
│                                                                              │
│  Concatenate:                                                                │
│                                                                              │
│    h_t = [→h_t; ←h_t]    ∈ ℝ^{2d}                                          │
│                                                                              │
│  Encoder output:                                                             │
│    H = [h₁, h₂, ..., hₜ]    ∈ ℝ^{T × 2d}                                  │
│                                                                              │
│  Final states (for decoder initialization):                                 │
│    →h_T, →c_T (forward final)                                              │
│    ←h_1, ←c_1 (backward final)                                             │
│                                                                              │
│  State reduction:                                                            │
│    h_dec_init = ReLU(W_h · [→h_T; ←h_1] + b_h)    ∈ ℝ^d                   │
│    c_dec_init = ReLU(W_c · [→c_T; ←c_1] + b_c)    ∈ ℝ^d                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Transformer Encoder

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: TRANSFORMER ENCODER                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Single Transformer Encoder Layer:                                          │
│                                                                              │
│  1. Multi-Head Self-Attention:                                              │
│                                                                              │
│     For each head i = 1, ..., h:                                           │
│                                                                              │
│       Q_i = X · W_Q^i    ∈ ℝ^{T × d_k}       (d_k = d/h)                   │
│       K_i = X · W_K^i    ∈ ℝ^{T × d_k}                                     │
│       V_i = X · W_V^i    ∈ ℝ^{T × d_k}                                     │
│                                                                              │
│       Attention_i(Q, K, V) = softmax(Q_i · K_i^T / √d_k) · V_i             │
│                                                                              │
│     Concatenate and project:                                                │
│       MultiHead(X) = Concat(head_1, ..., head_h) · W_O                     │
│                                                                              │
│  2. Pre-LayerNorm + Residual:                                              │
│                                                                              │
│     X' = X + MultiHead(LayerNorm(X))                                       │
│                                                                              │
│  3. Feed-Forward Network:                                                   │
│                                                                              │
│     FFN(x) = GELU(x · W_1 + b_1) · W_2 + b_2                              │
│                                                                              │
│     Where:                                                                   │
│       W_1 ∈ ℝ^{d × d_ff}    (d_ff = 128)                                   │
│       W_2 ∈ ℝ^{d_ff × d}                                                   │
│                                                                              │
│  4. Pre-LayerNorm + Residual:                                              │
│                                                                              │
│     X'' = X' + FFN(LayerNorm(X'))                                          │
│                                                                              │
│  Stacked Layers:                                                            │
│                                                                              │
│     H = TransformerEncoder(E_input) = Layer_L(Layer_{L-1}(...Layer_1(X)))  │
│                                                                              │
│  Final output:                                                               │
│     H ∈ ℝ^{T × d}                                                          │
│                                                                              │
│  Parameters per layer:                                                       │
│     Self-attention: 4 × d × d = 4 × 64 × 64 = 16,384                       │
│     FFN: 2 × d × d_ff = 2 × 64 × 128 = 16,384                              │
│     LayerNorm: 4 × d = 256                                                  │
│     Total per layer: ~33,024                                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison Visual

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ENCODER COMPARISON                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL BiLSTM:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input: x₁ ──→ x₂ ──→ x₃ ──→ x₄ ──→ x₅                            │   │
│  │          │       │       │       │       │                          │   │
│  │          ↓       ↓       ↓       ↓       ↓                          │   │
│  │  Forward: →h₁ → →h₂ → →h₃ → →h₄ → →h₅    (sequential)             │   │
│  │  Backward: ←h₁ ← ←h₂ ← ←h₃ ← ←h₄ ← ←h₅   (sequential)             │   │
│  │          │       │       │       │       │                          │   │
│  │          ↓       ↓       ↓       ↓       ↓                          │   │
│  │  Output: [→h₁;←h₁] ... [→h₅;←h₅]                                   │   │
│  │          [512]     ...  [512]                                       │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROPOSED Transformer:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │  Input: x₁    x₂    x₃    x₄    x₅                                 │   │
│  │          │     │     │     │     │                                  │   │
│  │          ↓     ↓     ↓     ↓     ↓                                  │   │
│  │         ┌─────────────────────────┐                                 │   │
│  │         │   Self-Attention        │  (parallel, all-to-all)        │   │
│  │         │   h_i = Σ α_ij · v_j    │                                │   │
│  │         └─────────────────────────┘                                 │   │
│  │          │     │     │     │     │                                  │   │
│  │          ↓     ↓     ↓     ↓     ↓                                  │   │
│  │  Output: h₁   h₂    h₃    h₄    h₅                                 │   │
│  │         [64]  [64]  [64]  [64]  [64]                               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Formulations

### Original: Bahdanau Additive Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: BAHDANAU ATTENTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given:                                                                      │
│    - Encoder outputs: H = [h₁, h₂, ..., hₜ]    ∈ ℝ^{T × 2d}               │
│    - Decoder state: s_t    ∈ ℝ^d                                           │
│                                                                              │
│  Attention weights:                                                          │
│    W_h ∈ ℝ^{2d × d_attn}   (encoder projection)                            │
│    W_s ∈ ℝ^{d × d_attn}    (decoder projection)                            │
│    v ∈ ℝ^{d_attn}          (attention vector)                              │
│                                                                              │
│  Score computation:                                                          │
│                                                                              │
│    e_{t,i} = v^T · tanh(W_h · h_i + W_s · s_t)                             │
│                                                                              │
│  Alternatively written:                                                      │
│                                                                              │
│    e_t = v^T · tanh(H · W_h + s_t · W_s)    ∈ ℝ^T                          │
│                                                                              │
│  Attention distribution:                                                     │
│                                                                              │
│    α_t = softmax(e_t)    ∈ ℝ^T                                             │
│                                                                              │
│  Context vector:                                                             │
│                                                                              │
│    c_t = Σ_{i=1}^{T} α_{t,i} · h_i    ∈ ℝ^{2d}                            │
│                                                                              │
│  With masking (for padded positions):                                       │
│                                                                              │
│    α_t = softmax(e_t ⊙ mask) / Σ(softmax(e_t ⊙ mask))                     │
│                                                                              │
│  Coverage (optional):                                                        │
│                                                                              │
│    coverage_t = Σ_{t'=1}^{t-1} α_{t'}    ∈ ℝ^T                             │
│                                                                              │
│    e_{t,i} = v^T · tanh(W_h · h_i + W_s · s_t + W_c · coverage_{t,i})     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Scaled Dot-Product Attention

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: SCALED DOT-PRODUCT ATTENTION                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Given:                                                                      │
│    - Encoder outputs: H = [h₁, h₂, ..., hₜ]    ∈ ℝ^{T × d}                │
│    - Query vector: q    ∈ ℝ^d  (from last encoder position)                │
│                                                                              │
│  Projection weights:                                                         │
│    W_Q ∈ ℝ^{d × d}   (query projection)                                    │
│    W_K ∈ ℝ^{d × d}   (key projection)                                      │
│    W_V ∈ ℝ^{d × d}   (value projection)                                    │
│                                                                              │
│  Projections:                                                                │
│                                                                              │
│    Q = q · W_Q       ∈ ℝ^d                                                 │
│    K = H · W_K       ∈ ℝ^{T × d}                                           │
│    V = H · W_V       ∈ ℝ^{T × d}                                           │
│                                                                              │
│  Score computation:                                                          │
│                                                                              │
│    e = Q · K^T / √d    ∈ ℝ^T                                               │
│                                                                              │
│  Expanded:                                                                   │
│                                                                              │
│    e_i = (q · W_Q) · (h_i · W_K)^T / √d                                    │
│        = Q · K_i^T / √d                                                     │
│                                                                              │
│  With masking (for padded positions):                                       │
│                                                                              │
│    e_i = -∞  if position i is padded                                       │
│                                                                              │
│  Attention distribution:                                                     │
│                                                                              │
│    α = softmax(e)    ∈ ℝ^T                                                 │
│                                                                              │
│  Context vector:                                                             │
│                                                                              │
│    c = α · V = Σ_{i=1}^{T} α_i · V_i    ∈ ℝ^d                             │
│                                                                              │
│  No coverage needed (single-step prediction)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison Formula

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION FORMULA COMPARISON                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ORIGINAL (Bahdanau Additive):                                              │
│                                                                              │
│    e_{t,i} = v^T · tanh(W_h · h_i + W_s · s_t)                             │
│                                                                              │
│    α_t = softmax(e_t)                                                       │
│                                                                              │
│    c_t = Σ α_{t,i} · h_i                                                   │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  PROPOSED (Scaled Dot-Product):                                             │
│                                                                              │
│    e_i = (W_Q · q) · (W_K · h_i)^T / √d                                    │
│                                                                              │
│    α = softmax(e)                                                           │
│                                                                              │
│    c = Σ α_i · (W_V · h_i)                                                 │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  KEY DIFFERENCES:                                                           │
│                                                                              │
│    1. Score function:                                                       │
│       - Original: tanh(linear combination)  [additive]                     │
│       - Proposed: dot product / √d          [multiplicative]               │
│                                                                              │
│    2. Complexity:                                                           │
│       - Original: O(T × d × d_attn)                                        │
│       - Proposed: O(T × d)  [more efficient]                               │
│                                                                              │
│    3. Value transform:                                                      │
│       - Original: None (use h_i directly)                                  │
│       - Proposed: W_V projection                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Gate Mechanism

### Original: Generation Probability (p_gen)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: P_GEN CALCULATION                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Inputs at decoder step t:                                                   │
│    - Context vector: c_t ∈ ℝ^{2d}                                          │
│    - LSTM cell state: cell_t ∈ ℝ^d                                         │
│    - LSTM hidden state: hidden_t ∈ ℝ^d                                     │
│    - Input embedding: x_t ∈ ℝ^{d_emb}                                      │
│                                                                              │
│  Concatenated input:                                                         │
│                                                                              │
│    input = [c_t; cell_t; hidden_t; x_t]    ∈ ℝ^{2d + d + d + d_emb}       │
│                                            ∈ ℝ^{512 + 256 + 256 + 128}     │
│                                            ∈ ℝ^{1152}                       │
│                                                                              │
│  Linear projection:                                                          │
│                                                                              │
│    W_pgen ∈ ℝ^{1152 × 1}                                                   │
│    b_pgen ∈ ℝ^1                                                            │
│                                                                              │
│  Computation:                                                                │
│                                                                              │
│    p_gen = σ(W_pgen · input + b_pgen)    ∈ (0, 1)                          │
│                                                                              │
│  Interpretation:                                                             │
│    p_gen → 1: Generate from vocabulary                                      │
│    p_gen → 0: Copy from input (pointer)                                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Gate Calculation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: GATE CALCULATION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Inputs:                                                                     │
│    - Context vector: c ∈ ℝ^d                                               │
│    - Query vector: q ∈ ℝ^d                                                 │
│                                                                              │
│  Concatenated input:                                                         │
│                                                                              │
│    input = [c; q]    ∈ ℝ^{2d}                                              │
│                      ∈ ℝ^{128}                                              │
│                                                                              │
│  MLP computation:                                                            │
│                                                                              │
│    Layer 1:                                                                  │
│      W₁ ∈ ℝ^{2d × d}                                                       │
│      b₁ ∈ ℝ^d                                                              │
│      h = GELU(W₁ · input + b₁)    ∈ ℝ^d                                   │
│                                                                              │
│    Dropout:                                                                  │
│      h' = Dropout(h, p=0.15)                                               │
│                                                                              │
│    Layer 2:                                                                  │
│      W₂ ∈ ℝ^{d × 1}                                                        │
│      b₂ ∈ ℝ^1                                                              │
│      gate = σ(W₂ · h' + b₂)    ∈ (0, 1)                                   │
│                                                                              │
│  Interpretation (INVERTED from original):                                   │
│    gate → 1: Copy from input (pointer)                                      │
│    gate → 0: Generate from vocabulary                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Final Distribution

### Original: Vocab + Pointer Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: FINAL DISTRIBUTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  At each decoder step t:                                                    │
│                                                                              │
│  1. Vocabulary distribution:                                                │
│                                                                              │
│     P_vocab(w) = softmax(W_out · [output_t; c_t] + b_out)    ∈ ℝ^V        │
│                                                                              │
│     Scale by p_gen:                                                         │
│     P_vocab'(w) = p_gen × P_vocab(w)                                       │
│                                                                              │
│  2. Pointer distribution (scatter attention):                               │
│                                                                              │
│     P_ptr(w) = Σ_{i: x_i = w} α_{t,i}                                      │
│                                                                              │
│     Scale by (1 - p_gen):                                                   │
│     P_ptr'(w) = (1 - p_gen) × P_ptr(w)                                     │
│                                                                              │
│  3. Final distribution:                                                     │
│                                                                              │
│     P(w) = P_vocab'(w) + P_ptr'(w)                                         │
│          = p_gen × P_vocab(w) + (1 - p_gen) × P_ptr(w)                     │
│                                                                              │
│  Extended vocabulary (for OOV):                                             │
│     P(w) ∈ ℝ^{V + max_oov}                                                 │
│                                                                              │
│     - For in-vocab words: sum of vocab and pointer                         │
│     - For OOV words: only pointer contribution                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Generation + Pointer Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: FINAL DISTRIBUTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Single prediction (no decoder loop):                                       │
│                                                                              │
│  1. Generation distribution:                                                │
│                                                                              │
│     P_gen(loc) = softmax(W_gen · c)    ∈ ℝ^{N_loc}                        │
│                                                                              │
│  2. Pointer distribution (scatter attention):                               │
│                                                                              │
│     Initialize: P_ptr = zeros(N_loc)                                       │
│                                                                              │
│     For each position i:                                                    │
│       P_ptr[x_i] += α_i                                                    │
│                                                                              │
│     Result: P_ptr ∈ ℝ^{N_loc}                                              │
│                                                                              │
│  3. Final distribution (INVERTED gate semantics):                          │
│                                                                              │
│     P(loc) = gate × P_ptr(loc) + (1 - gate) × P_gen(loc)                  │
│            ↑                      ↑                                         │
│         copy weight           generate weight                               │
│                                                                              │
│  No extended vocabulary needed (all locations known)                        │
│                                                                              │
│  Output:                                                                     │
│     log P(loc) = log(P(loc) + ε)    ∈ ℝ^{N_loc}                           │
│                                                                              │
│     (Log probabilities for numerical stability)                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Loss Functions

### Original: Negative Log Likelihood

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: NLL LOSS                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Per-step loss:                                                             │
│                                                                              │
│    L_t = -log P(y_t^* | y_{<t}, x)                                         │
│                                                                              │
│  Where y_t^* is the target word at step t                                  │
│                                                                              │
│  Masked average:                                                             │
│                                                                              │
│            Σ_{t=1}^{T_{dec}} L_t × mask_t                                   │
│    L_NLL = ──────────────────────────────────                               │
│               Σ_{t=1}^{T_{dec}} mask_t                                      │
│                                                                              │
│  Coverage loss (optional):                                                  │
│                                                                              │
│    L_cov_t = Σ_{i=1}^{T} min(α_{t,i}, coverage_{t,i})                      │
│                                                                              │
│            Σ_{t=1}^{T_{dec}} L_cov_t × mask_t                               │
│    L_cov = ─────────────────────────────────────                            │
│              Σ_{t=1}^{T_{dec}} mask_t                                       │
│                                                                              │
│  Total loss:                                                                 │
│                                                                              │
│    L_total = L_NLL + λ × L_cov    (λ = 1.0)                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Cross-Entropy with Label Smoothing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: CROSS-ENTROPY LOSS                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Standard cross-entropy:                                                     │
│                                                                              │
│    L_CE = -log P(y^*)                                                       │
│                                                                              │
│  Where y^* is the target location                                          │
│                                                                              │
│  With label smoothing (ε = 0.03):                                          │
│                                                                              │
│    Target distribution q:                                                    │
│                                                                              │
│           ┌ 1 - ε           if k = y^*                                     │
│    q_k = ─┤                                                                 │
│           └ ε / (K - 1)     if k ≠ y^*                                     │
│                                                                              │
│    Smoothed loss:                                                            │
│                                                                              │
│    L_smooth = -Σ_{k=1}^{K} q_k × log P(k)                                  │
│                                                                              │
│             = -(1-ε) × log P(y^*) - ε/(K-1) × Σ_{k≠y^*} log P(k)          │
│                                                                              │
│  Batch average:                                                              │
│                                                                              │
│            1   B                                                            │
│    L = ─────── Σ L_smooth^{(i)}                                            │
│          B   i=1                                                            │
│                                                                              │
│  No coverage loss (single-step prediction)                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Forward Pass

### Original Forward Pass

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: COMPLETE FORWARD PASS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT:                                                                      │
│    x_enc = [w₁, w₂, ..., w_T]           (encoder word IDs)                 │
│    x_dec = [<START>, y₁, ..., y_{T-1}]  (decoder input)                    │
│    y = [y₁, y₂, ..., y_T]               (target words)                     │
│                                                                              │
│  STEP 1: Embedding                                                          │
│    E_enc = Embedding(x_enc)    ∈ ℝ^{T × 128}                               │
│    E_dec = [Embedding(x) for x in x_dec]                                   │
│                                                                              │
│  STEP 2: Encode                                                             │
│    H, (→h_T, ←h_1), (→c_T, ←c_1) = BiLSTM(E_enc)                          │
│    H ∈ ℝ^{T × 512}                                                         │
│                                                                              │
│  STEP 3: Reduce states                                                      │
│    h_init = ReLU(W_h · [→h_T; ←h_1])    ∈ ℝ^{256}                         │
│    c_init = ReLU(W_c · [→c_T; ←c_1])    ∈ ℝ^{256}                         │
│                                                                              │
│  STEP 4: Decode (for t = 1 to T_dec)                                       │
│    For each step t:                                                         │
│      a) Attention:                                                          │
│         c_t, α_t = Attention(H, s_{t-1})                                   │
│                                                                              │
│      b) LSTM:                                                               │
│         output_t, s_t = LSTM([E_dec[t]; c_t], s_{t-1})                    │
│                                                                              │
│      c) p_gen:                                                              │
│         p_gen_t = σ(W · [c_t; cell_t; hidden_t; E_dec[t]])                │
│                                                                              │
│      d) Vocab distribution:                                                 │
│         P_vocab_t = softmax(W_out · [output_t; c_t])                       │
│                                                                              │
│      e) Final distribution:                                                 │
│         P_t = p_gen_t × P_vocab_t + (1-p_gen_t) × scatter(α_t, x_enc)     │
│                                                                              │
│  STEP 5: Loss                                                               │
│    L = -1/T × Σ_t log P_t(y_t) × mask_t                                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Forward Pass

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: COMPLETE FORWARD PASS                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  INPUT:                                                                      │
│    x = [loc₁, loc₂, ..., loc_T]      (location IDs)                        │
│    x_dict = {user, time, weekday, duration, recency}                       │
│    y = next_loc                       (target location)                     │
│                                                                              │
│  STEP 1: Embedding                                                          │
│    E = E_loc(x) + E_user(x_dict['user']) + E_time(x_dict['time'])         │
│        + E_week(x_dict['weekday']) + E_dur(x_dict['duration'])            │
│        + E_rec(x_dict['recency']) + E_pos([T-1, ..., 0])                   │
│    E = LayerNorm(E)    ∈ ℝ^{T × 64}                                        │
│                                                                              │
│  STEP 2: Encode                                                             │
│    H = TransformerEncoder(E, padding_mask)    ∈ ℝ^{T × 64}                │
│                                                                              │
│  STEP 3: Get query                                                          │
│    q = H[last_valid_position]    ∈ ℝ^{64}                                  │
│                                                                              │
│  STEP 4: Attention                                                          │
│    Q = W_Q · q                                                              │
│    K = W_K · H                                                              │
│    V = W_V · H                                                              │
│    α = softmax(Q · K^T / √64)    ∈ ℝ^T                                     │
│    c = α · V    ∈ ℝ^{64}                                                   │
│                                                                              │
│  STEP 5: Gate                                                               │
│    gate = σ(MLP([c; q]))    ∈ (0, 1)                                       │
│                                                                              │
│  STEP 6: Distributions                                                      │
│    P_gen = softmax(W_gen · c)    ∈ ℝ^{N_loc}                              │
│    P_ptr = scatter_add(zeros(N_loc), x, α)    ∈ ℝ^{N_loc}                 │
│                                                                              │
│  STEP 7: Final output                                                       │
│    P = gate × P_ptr + (1-gate) × P_gen                                     │
│    output = log(P + ε)    ∈ ℝ^{N_loc}                                      │
│                                                                              │
│  STEP 8: Loss                                                               │
│    L = CrossEntropy(output, y)  with label_smoothing=0.03                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary of Formulas

| Component | Original | Proposed |
|-----------|----------|----------|
| **Embedding** | $E[x]$ | $\sum_i E_i[x_i]$ (multi-modal) |
| **Encoder** | $BiLSTM(E)$ | $Transformer(E)$ |
| **Attention Score** | $v^T \cdot tanh(W_h h + W_s s)$ | $\frac{QK^T}{\sqrt{d}}$ |
| **Gate** | $\sigma(W \cdot [c; c_t; h_t; x_t])$ | $\sigma(MLP([c; q]))$ |
| **Final Dist** | $p_{gen} \cdot P_{vocab} + (1-p_{gen}) \cdot P_{ptr}$ | $gate \cdot P_{ptr} + (1-gate) \cdot P_{gen}$ |
| **Loss** | $-\frac{1}{T}\sum_t \log P(y_t) \cdot mask_t$ | $-\sum_k q_k \log P(k)$ |

---

*Next: [14_EXAMPLE_WALKTHROUGH.md](14_EXAMPLE_WALKTHROUGH.md) - Complete numerical example*
