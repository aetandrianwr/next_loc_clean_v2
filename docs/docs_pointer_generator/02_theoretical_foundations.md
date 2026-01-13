# Theoretical Foundations

## Table of Contents
1. [Sequence-to-Sequence Learning](#sequence-to-sequence-learning)
2. [Attention Mechanism](#attention-mechanism)
3. [Pointer Networks](#pointer-networks)
4. [Copy Mechanism](#copy-mechanism)
5. [Coverage Mechanism](#coverage-mechanism)
6. [Mathematical Formulation](#mathematical-formulation)
7. [Loss Functions](#loss-functions)

---

## Sequence-to-Sequence Learning

### The Basic Framework

Sequence-to-sequence (Seq2Seq) learning transforms an input sequence into an output sequence of potentially different length.

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                      SEQUENCE-TO-SEQUENCE FRAMEWORK                             │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Input Sequence (x):    x₁, x₂, x₃, ..., xₙ                                   │
│                          ↓   ↓   ↓        ↓                                    │
│                      ┌───────────────────────────┐                              │
│                      │        ENCODER            │                              │
│                      │    (Bidirectional LSTM)   │                              │
│                      └───────────┬───────────────┘                              │
│                                  │                                              │
│                                  ▼                                              │
│                      ┌───────────────────────────┐                              │
│                      │    Context/Hidden State   │                              │
│                      └───────────┬───────────────┘                              │
│                                  │                                              │
│                                  ▼                                              │
│                      ┌───────────────────────────┐                              │
│                      │        DECODER            │                              │
│                      │    (Unidirectional LSTM)  │                              │
│                      └───────────────────────────┘                              │
│                          ↓   ↓   ↓        ↓                                    │
│   Output Sequence (y):   y₁, y₂, y₃, ..., yₘ                                   │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### LSTM (Long Short-Term Memory)

The model uses LSTM cells, which solve the vanishing gradient problem of standard RNNs:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           LSTM CELL STRUCTURE                                   │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│                         Cell State (c_{t-1}) ────────────────▶ Cell State (c_t)│
│                              │                                      ▲          │
│                              │      ┌──────┐      ┌──────┐         │          │
│                              │      │  ×   │      │  +   │─────────┘          │
│                              │      └──┬───┘      └──┬───┘                     │
│                              │         │             │                         │
│                              ▼         │             │                         │
│                         ┌────────┐     │       ┌─────┴─────┐                   │
│                         │ Forget │     │       │   Input   │                   │
│                         │  Gate  │     │       │   Gate    │                   │
│                         │  (f_t) │     │       │   (i_t)   │                   │
│                         └────┬───┘     │       └─────┬─────┘                   │
│                              │         │             │                         │
│                              │         │             │                         │
│   Hidden State ──────────────┼─────────┼─────────────┼─────────▶ Hidden State  │
│     (h_{t-1})                │         │             │              (h_t)      │
│                              │         │             │                         │
│                              │    ┌────┴────┐  ┌─────┴─────┐                   │
│                              │    │  tanh   │  │   Output  │                   │
│                              │    │  (c̃_t)  │  │   Gate    │                   │
│                              │    └─────────┘  │   (o_t)   │                   │
│                              │                 └───────────┘                   │
│                              │                                                 │
│   Input (x_t) ───────────────┴─────────────────────────────────────────────── │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

#### LSTM Equations

For each time step t:

```
Forget Gate:     f_t = σ(W_f · [h_{t-1}, x_t] + b_f)
Input Gate:      i_t = σ(W_i · [h_{t-1}, x_t] + b_i)
Candidate:       c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)
Cell State:      c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t
Output Gate:     o_t = σ(W_o · [h_{t-1}, x_t] + b_o)
Hidden State:    h_t = o_t ⊙ tanh(c_t)

Where:
  σ     = sigmoid function
  tanh  = hyperbolic tangent
  ⊙     = element-wise multiplication
  [·,·] = concatenation
```

### Bidirectional LSTM

The encoder uses a **bidirectional** LSTM to capture context from both directions:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                       BIDIRECTIONAL LSTM ENCODER                                │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Input:     "The"  "cat"  "sat"  "on"  "the"  "mat"                           │
│               x₁     x₂     x₃     x₄    x₅     x₆                             │
│               │      │      │      │     │      │                              │
│               ▼      ▼      ▼      ▼     ▼      ▼                              │
│            ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐                          │
│   Forward: │ →h₁ ││ →h₂ ││ →h₃ ││ →h₄ ││ →h₅ ││ →h₆ │                          │
│            └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘                          │
│               ──────────────▶   Direction   ──────────────▶                    │
│                                                                                 │
│            ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐                          │
│   Backward:│ ←h₁ ││ ←h₂ ││ ←h₃ ││ ←h₄ ││ ←h₅ ││ ←h₆ │                          │
│            └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘                          │
│               ◀────────────────   Direction   ◀────────────────                │
│                                                                                 │
│               │      │      │      │     │      │                              │
│               ▼      ▼      ▼      ▼     ▼      ▼                              │
│            ┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐┌─────┐                          │
│   Concat:  │ h₁  ││ h₂  ││ h₃  ││ h₄  ││ h₅  ││ h₆  │                          │
│            │[→,←]││[→,←]││[→,←]││[→,←]││[→,←]││[→,←]│                          │
│            └─────┘└─────┘└─────┘└─────┘└─────┘└─────┘                          │
│                                                                                 │
│   h_i = [→h_i ; ←h_i]  ∈ ℝ^(2 × hidden_dim)                                    │
│                                                                                 │
│   For hidden_dim = 256:                                                        │
│   Each encoder state h_i ∈ ℝ^512                                               │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Mechanism

### The Problem with Basic Seq2Seq

In basic Seq2Seq, the entire input sequence is compressed into a single fixed-size vector. This creates an **information bottleneck**:

```
                    Information Bottleneck
                            │
   Long Article ───────▶ [Fixed Vector] ───────▶ Summary
   (400 words)              (256 dim)           (100 words)
                                │
                                ▼
                    "How can 256 numbers
                     encode all information
                     in 400 words?"
```

### The Attention Solution

Attention allows the decoder to **look back** at all encoder states at each decoding step:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         BAHDANAU ATTENTION MECHANISM                            │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Encoder Hidden States: h₁, h₂, h₃, ..., hₙ                                   │
│                                                                                 │
│   At decoder step t with decoder state s_t:                                    │
│                                                                                 │
│   Step 1: Calculate attention scores (energy)                                  │
│   ────────────────────────────────────────────                                 │
│                                                                                 │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + b_attn)                            │
│                                                                                 │
│   Where:                                                                        │
│   - W_h: Weight matrix for encoder states                                      │
│   - W_s: Weight matrix for decoder state                                       │
│   - v: Attention weight vector                                                 │
│   - b_attn: Bias term                                                          │
│                                                                                 │
│                                                                                 │
│   Step 2: Normalize scores to probabilities                                    │
│   ────────────────────────────────────────────                                 │
│                                                                                 │
│   α_ti = softmax(e_ti) = exp(e_ti) / Σⱼ exp(e_tj)                              │
│                                                                                 │
│                                                                                 │
│   Step 3: Calculate context vector (weighted sum)                              │
│   ────────────────────────────────────────────                                 │
│                                                                                 │
│   c_t = Σᵢ α_ti · h_i                                                          │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Visual Example of Attention

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        ATTENTION VISUALIZATION                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Source: "Germany beat Argentina in the World Cup final"                       │
│                                                                                 │
│   Generating word "Germany":                                                    │
│   ─────────────────────────                                                     │
│                                                                                 │
│   Germany  beat  Argentina  in  the  World  Cup  final                         │
│    0.85   0.03    0.02    0.01 0.01  0.03  0.03  0.02                          │
│   ████    █       █        █    █     █     █     █                            │
│                                                                                 │
│                                                                                 │
│   Generating word "defeated":                                                   │
│   ─────────────────────────                                                     │
│                                                                                 │
│   Germany  beat  Argentina  in  the  World  Cup  final                         │
│    0.10   0.72    0.05    0.03 0.02  0.03  0.03  0.02                          │
│    █      █████    █        █    █     █     █     █                            │
│                                                                                 │
│                                                                                 │
│   Generating word "Argentina":                                                  │
│   ─────────────────────────                                                     │
│                                                                                 │
│   Germany  beat  Argentina  in  the  World  Cup  final                         │
│    0.05   0.05    0.80    0.02 0.02  0.02  0.02  0.02                          │
│    █       █      █████     █    █     █     █     █                            │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Pointer Networks

### The Copying Problem

Standard Seq2Seq models can only generate words from a **fixed vocabulary**:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                     THE OUT-OF-VOCABULARY PROBLEM                               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Vocabulary: {"the", "a", "is", "was", "said", "it", "[UNK]", ...}            │
│               (50,000 most common words)                                        │
│                                                                                 │
│   Input:  "John Smith visited Paris on January 15th"                           │
│                │        │               │                                       │
│                ▼        ▼               ▼                                       │
│           Not in    Not in          Not in                                      │
│           Vocab     Vocab           Vocab                                       │
│                                                                                 │
│   Standard Model Output: "[UNK] visited [UNK] on [UNK]"                         │
│                                                                                 │
│   Problem: Proper nouns, dates, rare words become [UNK]!                       │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### The Pointer Solution

Pointer Networks (Vinyals et al., 2015) can **point** to positions in the input:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           POINTER NETWORK CONCEPT                               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Instead of generating over vocabulary:                                        │
│                                                                                 │
│   P(word) = softmax(W · h_decoder)  →  Distribution over 50K words             │
│                                                                                 │
│                                                                                 │
│   Generate over INPUT POSITIONS:                                                │
│                                                                                 │
│   P(position i) = attention_distribution[i]  →  Distribution over N positions  │
│                                                                                 │
│                                                                                 │
│   Example:                                                                      │
│   ─────────                                                                     │
│                                                                                 │
│   Input:    "John   Smith   visited   Paris"                                   │
│   Position:   1       2        3        4                                      │
│                                                                                 │
│   Attention:  0.8    0.1      0.05     0.05                                    │
│                │                                                                │
│                └─────▶ Point to position 1 → Output "John"                     │
│                                                                                 │
│   Key Insight: Can copy ANY word from input, including OOV words!              │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Copy Mechanism

### The Pointer-Generator Hybrid

Pure pointer networks can only copy—they can't generate new words. The Pointer-Generator combines both:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                      POINTER-GENERATOR HYBRID MECHANISM                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   TWO DISTRIBUTIONS:                                                            │
│   ──────────────────                                                            │
│                                                                                 │
│   1. Vocabulary Distribution (P_vocab):                                         │
│      - Generated by decoder                                                     │
│      - P_vocab(w) = softmax(V · [s_t, c_t])                                    │
│      - Can produce any word in vocabulary                                       │
│                                                                                 │
│   2. Attention Distribution (P_attn):                                           │
│      - From attention mechanism                                                 │
│      - P_attn(w) = Σ_{i:w_i=w} α_ti                                            │
│      - Can copy any word from source                                           │
│                                                                                 │
│                                                                                 │
│   GENERATION PROBABILITY (p_gen):                                               │
│   ────────────────────────────────                                              │
│                                                                                 │
│   p_gen = σ(w_c^T · c_t + w_s^T · s_t + w_x^T · x_t + b_ptr)                   │
│                                                                                 │
│   Where:                                                                        │
│   - c_t: context vector                                                        │
│   - s_t: decoder state                                                         │
│   - x_t: decoder input                                                         │
│                                                                                 │
│                                                                                 │
│   FINAL DISTRIBUTION:                                                           │
│   ───────────────────                                                           │
│                                                                                 │
│   P(w) = p_gen · P_vocab(w) + (1 - p_gen) · P_attn(w)                          │
│                                                                                 │
│          ▲ ▲                         ▲ ▲                                        │
│          │ │                         │ │                                        │
│          │ └─ Generate probability   │ └─ Copy probability                     │
│          └─── Vocab distribution     └─── Attention distribution               │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Extended Vocabulary

The model extends the vocabulary with **in-article OOV words**:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          EXTENDED VOCABULARY                                    │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Base Vocabulary: [PAD, UNK, START, STOP, the, a, is, ...]                    │
│                    Index: 0    1     2     3    4   5   6                       │
│                                                                                 │
│   Article: "John Smith visited Paris yesterday"                                 │
│   OOVs:    "John", "Smith", "Paris"                                            │
│                                                                                 │
│   Extended Vocabulary for this article:                                         │
│   ─────────────────────────────────────                                         │
│                                                                                 │
│   Index 0-49999:     Base vocabulary                                           │
│   Index 50000:       "John"    (1st OOV)                                       │
│   Index 50001:       "Smith"   (2nd OOV)                                       │
│   Index 50002:       "Paris"   (3rd OOV)                                       │
│                                                                                 │
│                                                                                 │
│   Encoder Input (enc_batch):                                                    │
│   ──────────────────────────                                                    │
│   "John"     →  1 (UNK)     - Can't look up OOV embeddings                     │
│   "Smith"    →  1 (UNK)                                                        │
│   "visited"  →  4523        - Normal vocab word                                │
│   "Paris"    →  1 (UNK)                                                        │
│                                                                                 │
│   Encoder Input Extended (enc_batch_extend_vocab):                              │
│   ───────────────────────────────────────────────                               │
│   "John"     →  50000       - 1st article OOV                                  │
│   "Smith"    →  50001       - 2nd article OOV                                  │
│   "visited"  →  4523        - Normal vocab word                                │
│   "Paris"    →  50002       - 3rd article OOV                                  │
│                                                                                 │
│   This extended encoding is used when scattering attention to final dist!      │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coverage Mechanism

### The Repetition Problem

Without coverage, the decoder often attends to the same positions repeatedly:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                         THE REPETITION PROBLEM                                  │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Source: "The cat sat on the mat. The cat was happy."                         │
│                                                                                 │
│   WITHOUT COVERAGE:                                                             │
│   ────────────────                                                              │
│                                                                                 │
│   Step 1: Attention focuses on "The cat sat"                                   │
│   Step 2: Attention focuses on "The cat sat"                                   │
│   Step 3: Attention focuses on "The cat sat"                                   │
│   Step 4: Attention focuses on "The cat sat"                                   │
│                                                                                 │
│   Output: "The cat sat. The cat sat. The cat sat."                             │
│                                                                                 │
│   Problem: Model gets "stuck" attending to the same region!                    │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Coverage Solution

Coverage tracks what has been attended to and discourages repeated attention:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                          COVERAGE MECHANISM                                     │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   COVERAGE VECTOR:                                                              │
│   ────────────────                                                              │
│                                                                                 │
│   c_t = Σ_{t'=0}^{t-1} α_{t'}    (sum of all previous attention distributions) │
│                                                                                 │
│   c_t^i represents "how much attention has been paid to source position i"     │
│                                                                                 │
│                                                                                 │
│   MODIFIED ATTENTION:                                                           │
│   ──────────────────                                                            │
│                                                                                 │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + w_c · c_t^i + b_attn)              │
│                                       │                                         │
│                                       └─ Coverage feature!                     │
│                                                                                 │
│   The coverage vector c_t^i is included in the attention calculation.          │
│   This allows the model to "know" what it has already attended to.             │
│                                                                                 │
│                                                                                 │
│   COVERAGE LOSS:                                                                │
│   ──────────────                                                                │
│                                                                                 │
│   covloss_t = Σ_i min(α_ti, c_t^i)                                             │
│                                                                                 │
│   This penalizes attending to the same location repeatedly:                    │
│                                                                                 │
│   - If c_t^i is high (already attended) AND α_ti is high (attending again)     │
│     → Loss is high → Model is penalized                                        │
│                                                                                 │
│   - If c_t^i is low (not attended) OR α_ti is low (not attending now)          │
│     → Loss is low → No penalty                                                 │
│                                                                                 │
│                                                                                 │
│   TOTAL LOSS:                                                                   │
│   ───────────                                                                   │
│                                                                                 │
│   L_total = L_NLL + λ · L_coverage                                             │
│                                                                                 │
│   Where λ is typically 1.0                                                     │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Coverage Example

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                        COVERAGE EXAMPLE (Step by Step)                          │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Source: "Germany beat Argentina in the final"                                 │
│            [0]     [1]    [2]     [3] [4]  [5]                                  │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   STEP 1: Generate "Germany"                                                    │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Coverage c₁:      [0.0,  0.0,  0.0,  0.0,  0.0,  0.0]  (initialized)         │
│   Attention α₁:     [0.8,  0.05, 0.05, 0.03, 0.02, 0.05]                       │
│   Coverage Loss:    Σ min(α₁, c₁) = 0.0  (no prior attention)                  │
│                                                                                 │
│   Output: "Germany" ✓                                                          │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   STEP 2: Generate "defeated"                                                   │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Coverage c₂:      [0.8,  0.05, 0.05, 0.03, 0.02, 0.05]  (= α₁)              │
│   Attention α₂:     [0.1,  0.7,  0.1,  0.03, 0.02, 0.05]                       │
│   Coverage Loss:    min(0.1, 0.8) + min(0.7, 0.05) + ... = 0.10 + 0.05 = 0.15 │
│                                                                                 │
│   Model attends to "beat" [1], coverage is low there → OK!                     │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   STEP 3: Generate "Argentina"                                                  │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Coverage c₃:      [0.9,  0.75, 0.15, 0.06, 0.04, 0.10]  (= c₂ + α₂)        │
│   Attention α₃:     [0.05, 0.05, 0.8,  0.03, 0.02, 0.05]                       │
│   Coverage Loss:    min(0.05, 0.9) + min(0.05, 0.75) + min(0.8, 0.15) = 0.25  │
│                                                                                 │
│   Model attends to "Argentina" [2], coverage is low there → OK!                │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   STEP 4: What if model tries to attend to "Germany" again?                     │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Coverage c₄:      [0.95, 0.80, 0.95, 0.09, 0.06, 0.15]                       │
│                                                                                 │
│   BAD Attention:    [0.6,  0.1,  0.1,  0.1,  0.05, 0.05]  (focusing on [0])    │
│   Coverage Loss:    min(0.6, 0.95) + ... = 0.6 + ...  VERY HIGH!               │
│                                                                                 │
│   GOOD Attention:   [0.05, 0.1,  0.1,  0.3,  0.2,  0.25]  (focusing on [3-5])  │
│   Coverage Loss:    min(0.05, 0.95) + ... = 0.05 + ... MUCH LOWER!             │
│                                                                                 │
│   Result: Model learns to attend to NEW positions!                              │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Mathematical Formulation

### Complete Model Equations

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE MATHEMATICAL FORMULATION                            │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   1. ENCODER                                                                    │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Word Embedding:                                                               │
│   e_i = E · x_i        where E ∈ ℝ^(V×d), x_i is one-hot                       │
│                                                                                 │
│   Forward LSTM:                                                                 │
│   →h_i = LSTM_fw(e_i, →h_{i-1})                                                │
│                                                                                 │
│   Backward LSTM:                                                                │
│   ←h_i = LSTM_bw(e_i, ←h_{i+1})                                                │
│                                                                                 │
│   Encoder Output:                                                               │
│   h_i = [→h_i ; ←h_i]  ∈ ℝ^(2·hidden_dim)                                      │
│                                                                                 │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   2. REDUCE STATES                                                              │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   s_0^c = ReLU(W_c · [→c_n ; ←c_0] + b_c)                                      │
│   s_0^h = ReLU(W_h · [→h_n ; ←h_0] + b_h)                                      │
│                                                                                 │
│   Initial decoder state: s_0 = (s_0^c, s_0^h)                                  │
│                                                                                 │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   3. ATTENTION                                                                  │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Attention score (without coverage):                                           │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + b_attn)                            │
│                                                                                 │
│   Attention score (with coverage):                                              │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + w_c · c_t^i + b_attn)              │
│                                                                                 │
│   Masked softmax:                                                               │
│   α̃_ti = exp(e_ti) · mask_i                                                   │
│   α_ti = α̃_ti / Σ_j α̃_tj                                                      │
│                                                                                 │
│   Context vector:                                                               │
│   c_t^* = Σ_i α_ti · h_i                                                       │
│                                                                                 │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   4. DECODER                                                                    │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Input transformation (attention-feeding):                                     │
│   x̃_t = W_x · [y_{t-1} ; c_{t-1}^*]                                           │
│                                                                                 │
│   Decoder LSTM:                                                                 │
│   (s_t^c, s_t^h) = LSTM_dec(x̃_t, s_{t-1})                                      │
│                                                                                 │
│   Output projection:                                                            │
│   ŝ_t = W_o · [s_t^h ; c_t^*] + b_o                                            │
│                                                                                 │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   5. POINTER-GENERATOR                                                          │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Generation probability:                                                       │
│   p_gen = σ(w_c^T · c_t^* + w_s^T · s_t^h + w_x^T · x̃_t + b_ptr)              │
│                                                                                 │
│   Vocabulary distribution:                                                      │
│   P_vocab(w) = softmax(W_v · ŝ_t + b_v)                                        │
│                                                                                 │
│   Copy distribution:                                                            │
│   P_copy(w) = Σ_{i:x_i=w} α_ti                                                 │
│                                                                                 │
│   Final distribution:                                                           │
│   P(w) = p_gen · P_vocab(w) + (1 - p_gen) · P_copy(w)                          │
│                                                                                 │
│                                                                                 │
│   ═══════════════════════════════════════════════════════════════════════════  │
│   6. COVERAGE                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════  │
│                                                                                 │
│   Coverage vector:                                                              │
│   c_t = Σ_{t'=0}^{t-1} α_{t'}                                                  │
│                                                                                 │
│   Coverage loss:                                                                │
│   L_cov = Σ_t Σ_i min(α_ti, c_t^i)                                             │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Loss Functions

### Negative Log-Likelihood Loss

The primary training objective is to maximize the probability of the correct output sequence:

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                      NEGATIVE LOG-LIKELIHOOD LOSS                               │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   For target sequence y* = (y*_1, y*_2, ..., y*_T):                            │
│                                                                                 │
│   L_NLL = - Σ_t log P(y*_t | y*_1, ..., y*_{t-1}, x)                           │
│                                                                                 │
│                                                                                 │
│   In code (pointer-generator mode):                                             │
│   ─────────────────────────────────                                             │
│                                                                                 │
│   For each decoder step t:                                                      │
│     1. Get target word index: target_t = y*_t                                  │
│     2. Get probability of target: p_target = P(target_t)                       │
│     3. Calculate loss: loss_t = -log(p_target)                                 │
│                                                                                 │
│   Final loss = average(loss_1, loss_2, ..., loss_T)                            │
│                                                                                 │
│                                                                                 │
│   Example:                                                                      │
│   ────────                                                                      │
│   Target word: "Germany" (index 50000 in extended vocab)                       │
│   Model probability: P(50000) = 0.85                                           │
│   Loss: -log(0.85) = 0.163                                                     │
│                                                                                 │
│   Target word: "defeated" (index 4523 in vocab)                                │
│   Model probability: P(4523) = 0.30                                            │
│   Loss: -log(0.30) = 1.204                                                     │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Coverage Loss

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           COVERAGE LOSS                                         │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   L_cov = (1/T) Σ_t Σ_i min(α_ti, c_t^i)                                       │
│                                                                                 │
│                                                                                 │
│   Interpretation:                                                               │
│   ───────────────                                                               │
│                                                                                 │
│   min(α_ti, c_t^i) is high when:                                               │
│   - c_t^i is high: Position i was heavily attended before                      │
│   - α_ti is high: Position i is being attended now                             │
│                                                                                 │
│   In other words: PENALTY for attending to same position twice!                │
│                                                                                 │
│                                                                                 │
│   Why min() and not product or sum?                                            │
│   ─────────────────────────────────                                             │
│                                                                                 │
│   - min(0.9, 0.1) = 0.1 → Small penalty (only one is high)                     │
│   - min(0.9, 0.9) = 0.9 → Large penalty (both are high)                        │
│   - min(0.1, 0.1) = 0.1 → Small penalty (neither is high)                      │
│                                                                                 │
│   Product (0.9 × 0.1 = 0.09) would be too harsh for first overlap              │
│   Sum (0.9 + 0.1 = 1.0) wouldn't distinguish the cases properly                │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

### Total Loss

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                           TOTAL LOSS                                            │
├────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│   Without coverage:                                                             │
│   L_total = L_NLL                                                              │
│                                                                                 │
│   With coverage:                                                                │
│   L_total = L_NLL + λ · L_cov                                                  │
│                                                                                 │
│   Where λ (cov_loss_wt) is typically 1.0                                       │
│                                                                                 │
│                                                                                 │
│   Training Strategy:                                                            │
│   ──────────────────                                                            │
│                                                                                 │
│   1. Train WITHOUT coverage until convergence                                  │
│      - Model learns basic generation and copying                               │
│      - May develop repetition issues                                           │
│                                                                                 │
│   2. Train WITH coverage for a short phase                                     │
│      - Add coverage mechanism to existing checkpoint                           │
│      - Fine-tune to reduce repetition                                          │
│                                                                                 │
│   This two-phase approach is more stable than training with coverage           │
│   from the start.                                                              │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Dimensions Summary

| Component | Dimension | Default Value |
|-----------|-----------|---------------|
| Word embedding | `emb_dim` | 128 |
| Encoder hidden state (each direction) | `hidden_dim` | 256 |
| Encoder output (both directions) | `2 × hidden_dim` | 512 |
| Decoder hidden state | `hidden_dim` | 256 |
| Attention vector | `2 × hidden_dim` | 512 |
| Context vector | `2 × hidden_dim` | 512 |
| Vocabulary size | `vocab_size` | 50,000 |
| Max encoder steps | `max_enc_steps` | 400 |
| Max decoder steps | `max_dec_steps` | 100 |

---

*Next: [03_architecture_deep_dive.md](03_architecture_deep_dive.md) - Complete Architecture Deep Dive*
