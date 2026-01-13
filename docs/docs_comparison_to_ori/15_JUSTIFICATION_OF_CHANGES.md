# Justification of Design Changes

## Table of Contents
1. [Overview](#overview)
2. [Framework Change: TensorFlow → PyTorch](#framework-change-tensorflow--pytorch)
3. [Task Adaptation: Summarization → Location Prediction](#task-adaptation-summarization--location-prediction)
4. [Encoder Change: BiLSTM → Transformer](#encoder-change-bilstm--transformer)
5. [Removing the Decoder](#removing-the-decoder)
6. [Attention Change: Bahdanau → Scaled Dot-Product](#attention-change-bahdanau--scaled-dot-product)
7. [Gate Semantics Inversion](#gate-semantics-inversion)
8. [Multi-Modal Embeddings](#multi-modal-embeddings)
9. [Training Configuration Changes](#training-configuration-changes)
10. [Summary of Justifications](#summary-of-justifications)

---

## Overview

This document provides detailed justification for each significant design change from the original Pointer-Generator Network to the proposed PointerNetworkV45.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DESIGN CHANGE PHILOSOPHY                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Guiding Principles:                                                        │
│                                                                              │
│  1. TASK APPROPRIATENESS                                                    │
│     └─► Adapt architecture to next location prediction                     │
│                                                                              │
│  2. MODERN BEST PRACTICES                                                   │
│     └─► Use proven advances in deep learning (Transformers, AdamW)         │
│                                                                              │
│  3. EFFICIENCY                                                              │
│     └─► Reduce parameters while maintaining expressiveness                 │
│                                                                              │
│  4. DOMAIN-SPECIFIC FEATURES                                                │
│     └─► Leverage mobility data characteristics (time, user, etc.)          │
│                                                                              │
│  5. SIMPLICITY                                                              │
│     └─► Remove unnecessary complexity (decoder, OOV handling)              │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Framework Change: TensorFlow → PyTorch

### The Change

| Aspect | Original | Proposed |
|--------|----------|----------|
| Framework | TensorFlow 1.x | PyTorch |
| Execution | Static graph | Dynamic graph (eager) |
| API Style | Define-then-run | Define-by-run |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY PYTORCH?                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. RESEARCH DOMINANCE                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - 75%+ of ML research papers use PyTorch (as of 2023)              │   │
│  │  - Easier to implement novel architectures                          │   │
│  │  - Better debugging with dynamic graphs                             │   │
│  │  - Standard in academic settings (PhD thesis target)                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. TensorFlow 1.x DEPRECATION                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - TF 1.x is no longer maintained                                   │   │
│  │  - Original code uses deprecated APIs:                              │   │
│  │    • tf.contrib.rnn (removed in TF 2.x)                            │   │
│  │    • tf.train.Supervisor (replaced)                                │   │
│  │    • tf.placeholder (deprecated)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. DEVELOPMENT EFFICIENCY                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Pythonic API (more intuitive)                                   │   │
│  │  - Better error messages                                           │   │
│  │  - Easier to debug with standard Python tools                      │   │
│  │  - Native support for latest architectures (Transformers)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4. ECOSYSTEM COMPATIBILITY                                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Hugging Face Transformers (PyTorch-first)                       │   │
│  │  - PyTorch Lightning for training infrastructure                   │   │
│  │  - Better mixed-precision support (torch.cuda.amp)                 │   │
│  │  - Integration with modern tools (Weights & Biases, etc.)          │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code Impact

```python
# ORIGINAL (TensorFlow 1.x static graph)
with tf.variable_scope('embedding'):
    embedding = tf.get_variable('embedding', [vsize, emb_dim])
    emb_enc = tf.nn.embedding_lookup(embedding, self._enc_batch)

# PROPOSED (PyTorch dynamic)
self.embedding = nn.Embedding(num_locations, d_model)
emb = self.embedding(x)  # Direct, Pythonic call
```

---

## Task Adaptation: Summarization → Location Prediction

### The Change

| Aspect | Original | Proposed |
|--------|----------|----------|
| Task | Text summarization | Next location prediction |
| Input | Document (words) | Trajectory (locations + features) |
| Output | Summary (word sequence) | Single location |
| Vocabulary | 50,000 words | ~500 locations |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY ADAPT TO LOCATION PREDICTION?                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. CORE INSIGHT: Pointer mechanism is applicable to both tasks            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  SUMMARIZATION:                                                      │   │
│  │    Input:  "The president announced new economic policies..."        │   │
│  │    Output: "President unveils economic plan"                        │   │
│  │    Pointer: Copy rare words (names, numbers) from input             │   │
│  │                                                                       │   │
│  │  LOCATION PREDICTION:                                                │   │
│  │    Input:  [Home, Coffee, Office, Restaurant, Office]               │   │
│  │    Output: Gym                                                       │   │
│  │    Pointer: Revisit recent locations (return to Office, Home)       │   │
│  │                                                                       │   │
│  │  COMMON PATTERN: Copy-or-generate decision                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. MOBILITY DATA CHARACTERISTICS                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  • High repetition: People revisit the same locations               │   │
│  │    - Home, Office, Gym account for 50%+ of visits                   │   │
│  │    - Pointer mechanism naturally handles this                       │   │
│  │                                                                       │   │
│  │  • Temporal patterns: Time of day affects destination               │   │
│  │    - Morning → Work, Evening → Home                                 │   │
│  │    - Multi-modal embeddings capture this                            │   │
│  │                                                                       │   │
│  │  • User-specific: Different users have different routines           │   │
│  │    - User embedding captures personal patterns                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. CLOSED VOCABULARY                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Text summarization: Open vocabulary (50K+ words, OOV problem)      │   │
│  │  Location prediction: Closed vocabulary (~500 locations, no OOV)    │   │
│  │                                                                       │   │
│  │  Implication:                                                       │   │
│  │    - No need for extended vocabulary                                │   │
│  │    - Simpler pointer scatter operation                             │   │
│  │    - Smaller output layer (500 vs 50,000)                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Encoder Change: BiLSTM → Transformer

### The Change

| Aspect | BiLSTM (Original) | Transformer (Proposed) |
|--------|-------------------|------------------------|
| Processing | Sequential | Parallel |
| Context | Through hidden state | Through self-attention |
| Long-range | Decays with distance | Direct attention |
| Parameters | ~2.1M | ~132K |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY TRANSFORMER?                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. BETTER LONG-RANGE DEPENDENCIES                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  BiLSTM:                                                             │   │
│  │    Position 1 ──hidden──► Position 10 ──hidden──► Position 50       │   │
│  │    └────────── Information must flow through 49 steps ──────────────┘│   │
│  │    Long sequences: gradient vanishing, information loss             │   │
│  │                                                                       │   │
│  │  Transformer:                                                        │   │
│  │    Position 1 ─────────────────────────────► Position 50            │   │
│  │    └────────── Direct attention connection (1 step) ────────────────┘│   │
│  │    Any position can directly attend to any other position           │   │
│  │                                                                       │   │
│  │  For mobility: First location of day can influence last decision    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. PARALLELIZATION                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  BiLSTM: O(T) sequential steps                                      │   │
│  │    h_1 → h_2 → h_3 → ... → h_T (must compute in order)             │   │
│  │                                                                       │   │
│  │  Transformer: O(1) parallel (with O(T²) attention)                  │   │
│  │    All positions computed simultaneously                            │   │
│  │    Much faster on GPU                                               │   │
│  │                                                                       │   │
│  │  Training time: ~2-3x faster for same model capacity               │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. PROVEN EFFECTIVENESS                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Transformers have become the dominant architecture:                │   │
│  │    - NLP: BERT, GPT, T5                                            │   │
│  │    - Vision: ViT, DINO                                             │   │
│  │    - Multimodal: CLIP, Flamingo                                    │   │
│  │    - Time series: Temporal Fusion Transformer                      │   │
│  │                                                                       │   │
│  │  Research shows Transformers often outperform RNNs/LSTMs           │   │
│  │  for sequence modeling tasks                                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4. POSITION AWARENESS                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  BiLSTM: Position is implicit in hidden state                       │   │
│  │  Transformer: Explicit position encoding                            │   │
│  │                                                                       │   │
│  │  Proposed: Position-from-end embedding                              │   │
│  │    - Position 0 = most recent (highest weight)                     │   │
│  │    - Position 49 = oldest in window                                │   │
│  │    - Explicitly captures recency importance                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Removing the Decoder

### The Change

| Aspect | Original | Proposed |
|--------|----------|----------|
| Decoder | LSTM (100 steps) | None |
| Output | Sequence of words | Single location |
| Beam search | Yes (size 4) | Not needed |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY NO DECODER?                                           │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. TASK DIFFERENCE                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Summarization (sequence-to-sequence):                               │   │
│  │    Input: [w₁, w₂, ..., w₄₀₀]  (document)                          │   │
│  │    Output: [s₁, s₂, ..., s₁₀₀] (summary words one by one)          │   │
│  │    Need decoder for autoregressive generation                       │   │
│  │                                                                       │   │
│  │  Location prediction (sequence-to-one):                             │   │
│  │    Input: [loc₁, loc₂, ..., loc₅₀] (trajectory)                    │   │
│  │    Output: [next_loc]              (single prediction)              │   │
│  │    No need for sequential generation                                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. COMPUTATIONAL SAVINGS                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Original decoder overhead:                                          │   │
│  │    - LSTM cell: ~1M parameters                                      │   │
│  │    - 100 decoder steps per example                                  │   │
│  │    - Attention computed 100 times                                   │   │
│  │    - State reduction layer: ~262K parameters                        │   │
│  │                                                                       │   │
│  │  Proposed:                                                           │   │
│  │    - No decoder parameters                                          │   │
│  │    - 1 attention computation                                        │   │
│  │    - No state reduction needed                                      │   │
│  │                                                                       │   │
│  │  Savings: ~1.3M parameters, 100x fewer attention computations      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. SIMPLER INFERENCE                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Original inference:                                                 │   │
│  │    - Run encoder                                                    │   │
│  │    - Run beam search (4 beams × 100 steps = 400 decoder calls)     │   │
│  │    - Track multiple hypotheses                                      │   │
│  │    - Complex stopping criteria                                      │   │
│  │                                                                       │   │
│  │  Proposed inference:                                                 │   │
│  │    - Run encoder                                                    │   │
│  │    - Single forward pass                                            │   │
│  │    - argmax(output) = prediction                                   │   │
│  │                                                                       │   │
│  │  Inference time: ~100x faster                                       │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Change: Bahdanau → Scaled Dot-Product

### The Change

| Aspect | Bahdanau (Original) | Scaled Dot-Product (Proposed) |
|--------|---------------------|-------------------------------|
| Formula | $v^T \cdot tanh(W_h h + W_s s)$ | $\frac{QK^T}{\sqrt{d}}$ |
| Non-linearity | tanh | None (linear) |
| Parameters | W_h, W_s, v (~780K) | W_Q, W_K, W_V (~12K) |
| Computation | Additive | Multiplicative |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY SCALED DOT-PRODUCT?                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. COMPUTATIONAL EFFICIENCY                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Bahdanau (for T positions, d dimensions):                          │   │
│  │    1. Project h: T × d × d = O(Td²) per decoder step               │   │
│  │    2. Project s: d × d = O(d²) per decoder step                    │   │
│  │    3. Add + tanh: T × d = O(Td)                                    │   │
│  │    4. Dot with v: T × d = O(Td)                                    │   │
│  │    Total: O(Td² + d²) per decoder step                              │   │
│  │                                                                       │   │
│  │  Scaled Dot-Product:                                                 │   │
│  │    1. Q×K^T: d × d×T = O(Td)                                       │   │
│  │    2. Scale: T = O(T)                                               │   │
│  │    3. Softmax: T = O(T)                                             │   │
│  │    Total: O(Td)                                                     │   │
│  │                                                                       │   │
│  │  Speedup: ~d times faster (64x in our case)                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. TRANSFORMER COMPATIBILITY                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Scaled dot-product is the standard attention in Transformers       │   │
│  │    - Used in encoder self-attention                                 │   │
│  │    - Used in decoder cross-attention                                │   │
│  │    - Highly optimized implementations available                     │   │
│  │                                                                       │   │
│  │  Using same attention type throughout:                              │   │
│  │    - Consistent behavior                                            │   │
│  │    - Shared understanding of attention patterns                    │   │
│  │    - Simpler codebase                                              │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. VALUE PROJECTION                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Bahdanau uses encoder outputs directly as values                   │   │
│  │  Scaled dot-product projects to value space (W_V)                   │   │
│  │                                                                       │   │
│  │  Benefit of W_V:                                                    │   │
│  │    - Separates "what to attend to" (K) from "what to retrieve" (V) │   │
│  │    - More expressive: can learn different key vs value representations│   │
│  │    - Standard in modern attention mechanisms                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Gate Semantics Inversion

### The Change

| Aspect | Original (p_gen) | Proposed (gate) |
|--------|------------------|-----------------|
| Value → 1 | Generate from vocab | Copy from input |
| Value → 0 | Copy from input | Generate from vocab |
| Name | p_gen (prob of generate) | gate (pointer gate) |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY INVERT GATE SEMANTICS?                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. DOMAIN-SPECIFIC INTERPRETATION                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Text Summarization:                                                 │   │
│  │    - Most words come from vocabulary (common words)                 │   │
│  │    - Rare/specific words copied from input                         │   │
│  │    - Default behavior: generate (p_gen high)                        │   │
│  │                                                                       │   │
│  │  Location Prediction:                                                │   │
│  │    - Most next locations are revisits (return home, return office) │   │
│  │    - New destinations are less common                               │   │
│  │    - Default behavior: copy (gate high)                            │   │
│  │                                                                       │   │
│  │  The inversion reflects the different nature of the tasks          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. NAMING CLARITY                                                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Original: "p_gen" (probability of generation)                      │   │
│  │    - p_gen=1 → generate                                             │   │
│  │    - p_gen=0 → copy                                                 │   │
│  │    - Confusing: "copy" is NOT 1-p_gen conceptually                 │   │
│  │                                                                       │   │
│  │  Proposed: "gate" (pointer gate)                                    │   │
│  │    - gate=1 → activate pointer (copy)                              │   │
│  │    - gate=0 → deactivate pointer (generate)                        │   │
│  │    - Intuitive: gate controls the pointer mechanism                │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. PRACTICAL CONSIDERATION                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  In mobility data, people tend to:                                  │   │
│  │    - Revisit the same 5-10 places 80% of the time                  │   │
│  │    - Visit new places rarely (exploration)                         │   │
│  │                                                                       │   │
│  │  Having gate → 1 mean "copy" aligns with this:                     │   │
│  │    - High gate (common case): revisit recent location              │   │
│  │    - Low gate (rare case): explore new destination                 │   │
│  │                                                                       │   │
│  │  Trained gate values typically: 0.6-0.8 (favoring pointer)        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Multi-Modal Embeddings

### The Change

| Aspect | Original | Proposed |
|--------|----------|----------|
| Number | 1 (word) | 7 (multi-modal) |
| Type | Word embedding | Location, user, time, weekday, duration, recency, position |
| Combination | N/A | Element-wise sum |
| Parameters | 6.4M | ~56K |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY MULTI-MODAL EMBEDDINGS?                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. MOBILITY DATA IS INHERENTLY MULTI-MODAL                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Each visit has multiple attributes:                                 │   │
│  │    - WHERE: Location (Home, Office, Gym)                            │   │
│  │    - WHO: User (Alice, Bob)                                         │   │
│  │    - WHEN: Time of day (8am, 5pm)                                   │   │
│  │    - WHICH DAY: Weekday (Monday, Sunday)                            │   │
│  │    - HOW LONG: Duration (30 min, 8 hours)                           │   │
│  │    - HOW RECENT: Position in sequence                               │   │
│  │                                                                       │   │
│  │  All these factors influence next destination:                      │   │
│  │    - Monday 9am → likely Office                                     │   │
│  │    - Sunday 10am → likely Cafe or Park                              │   │
│  │    - Alice after work → Gym                                         │   │
│  │    - Bob after work → Home                                          │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. RICH CONTEXT WITHOUT EXPLICIT FEATURES                                 │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Traditional approach: Hand-crafted features                        │   │
│  │    - is_morning, is_weekend, time_since_last_visit, ...            │   │
│  │    - Requires domain expertise                                      │   │
│  │    - May miss important patterns                                    │   │
│  │                                                                       │   │
│  │  Embedding approach: Learned representations                        │   │
│  │    - Model learns what's important                                  │   │
│  │    - Time embedding learns "morning vs evening" automatically       │   │
│  │    - User embedding captures individual habits                      │   │
│  │    - More flexible and adaptable                                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. SUMMATION VS CONCATENATION                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Concatenation: [loc_emb; user_emb; time_emb; ...]                 │   │
│  │    Dimension: 64 × 7 = 448                                          │   │
│  │    Parameters scale with number of modalities                       │   │
│  │    May overfit on small datasets                                    │   │
│  │                                                                       │   │
│  │  Summation: loc_emb + user_emb + time_emb + ...                    │   │
│  │    Dimension: 64 (constant)                                         │   │
│  │    Similar to positional encoding in Transformers                   │   │
│  │    Forces embeddings to share semantic space                        │   │
│  │    More parameter-efficient                                         │   │
│  │                                                                       │   │
│  │  Why summation works:                                               │   │
│  │    - Each embedding adds a "bias" to the representation            │   │
│  │    - Time embedding shifts representation toward "morning" space   │   │
│  │    - User embedding shifts toward "Alice's pattern" space          │   │
│  │    - Similar to how positional encoding works                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4. EACH EMBEDDING'S PURPOSE                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Location: Core semantic (what place)                               │   │
│  │  User: Personal patterns (who)                                      │   │
│  │  Time: Temporal patterns (when in day)                              │   │
│  │  Weekday: Weekly patterns (which day)                               │   │
│  │  Duration: Stay patterns (how long)                                 │   │
│  │  Recency: Recent history importance                                 │   │
│  │  Position-from-end: Explicit position encoding                      │   │
│  │                                                                       │   │
│  │  Example: "Office at 17:00 on Monday for 8 hours by Alice"         │   │
│  │    → Strong signal for "end of workday, likely going home/gym"     │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Configuration Changes

### The Changes

| Aspect | Original | Proposed | Ratio |
|--------|----------|----------|-------|
| Optimizer | Adagrad | AdamW | - |
| Learning Rate | 0.15 | 6.5e-4 | 230x lower |
| Batch Size | 16 | 128 | 8x larger |
| Gradient Clip | 2.0 | 0.8 | 2.5x lower |
| Dropout | 0.0 | 0.15 | - |
| Weight Decay | 0.0 | 0.015 | - |

### Justification

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WHY THESE TRAINING CHANGES?                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ADAGRAD → ADAMW                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Adagrad:                                                            │   │
│  │    - Good for sparse gradients (NLP, word embeddings)               │   │
│  │    - Learning rate only decreases (never increases)                 │   │
│  │    - Can get stuck in suboptimal solutions                          │   │
│  │    - Popular in 2016-2017 (when original was written)              │   │
│  │                                                                       │   │
│  │  AdamW:                                                              │   │
│  │    - Modern standard for Transformers                               │   │
│  │    - Adaptive learning rate (can increase or decrease)              │   │
│  │    - Decoupled weight decay (better regularization)                │   │
│  │    - Used in BERT, GPT, all modern Transformers                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. LEARNING RATE: 0.15 → 6.5e-4                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Why 230x lower?                                                     │   │
│  │    - Adagrad needs high LR (internally scales down)                 │   │
│  │    - Adam/AdamW needs low LR (already adaptive)                     │   │
│  │    - Transformers are sensitive to learning rate                    │   │
│  │    - Standard Transformer LR: 1e-4 to 1e-3                         │   │
│  │                                                                       │   │
│  │  Warmup schedule:                                                    │   │
│  │    - Transformers benefit from warmup                               │   │
│  │    - 5 epochs linear warmup → peak → cosine decay                  │   │
│  │    - Prevents early divergence                                      │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. BATCH SIZE: 16 → 128                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Why 8x larger?                                                      │   │
│  │    - Smaller model = smaller memory footprint                       │   │
│  │    - Fixed sequence length = predictable memory                     │   │
│  │    - Larger batches = more stable gradients                        │   │
│  │    - Better GPU utilization                                         │   │
│  │                                                                       │   │
│  │  Original constraint:                                                │   │
│  │    - Large model (47M params)                                       │   │
│  │    - Variable sequence lengths (up to 400)                         │   │
│  │    - Limited GPU memory in 2017                                    │   │
│  │                                                                       │   │
│  │  Proposed advantage:                                                 │   │
│  │    - Small model (240K params)                                      │   │
│  │    - Fixed sequence length (50)                                    │   │
│  │    - Modern GPUs with more memory                                  │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4. REGULARIZATION (Dropout, Weight Decay, Label Smoothing)               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Original (minimal regularization):                                 │   │
│  │    - Large dataset (millions of examples)                          │   │
│  │    - Long training (days)                                          │   │
│  │    - Adagrad provides implicit regularization                      │   │
│  │                                                                       │   │
│  │  Proposed (more regularization):                                    │   │
│  │    - Smaller dataset (mobility data limited)                       │   │
│  │    - Shorter training (early stopping)                             │   │
│  │    - Transformers can overfit on small data                        │   │
│  │                                                                       │   │
│  │  Dropout (0.15):                                                    │   │
│  │    - Prevents co-adaptation of neurons                             │   │
│  │    - Standard for Transformers                                      │   │
│  │                                                                       │   │
│  │  Weight Decay (0.015):                                              │   │
│  │    - L2 regularization in AdamW                                    │   │
│  │    - Prevents large weights                                        │   │
│  │                                                                       │   │
│  │  Label Smoothing (0.03):                                           │   │
│  │    - Prevents overconfident predictions                            │   │
│  │    - Better calibration                                            │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary of Justifications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SUMMARY: DESIGN DECISIONS                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Change                  │ Primary Justification                            │
│  ────────────────────────┼───────────────────────────────────────────────── │
│  TensorFlow → PyTorch    │ Modern standard, better research ecosystem       │
│  BiLSTM → Transformer    │ Better long-range dependencies, parallelization │
│  Remove decoder          │ Single-output task, not sequence generation     │
│  Bahdanau → Dot-Product  │ Efficiency, Transformer compatibility           │
│  Invert gate semantics   │ Matches domain (revisits more common)           │
│  Multi-modal embeddings  │ Mobility data is inherently multi-modal         │
│  Adagrad → AdamW         │ Modern optimizer standard for Transformers      │
│  Lower learning rate     │ Required for Adam, better stability            │
│  Larger batch size       │ Smaller model, more stable gradients           │
│  Add regularization      │ Prevent overfitting on smaller datasets        │
│                                                                              │
│  RESULT:                                                                     │
│    - 196x fewer parameters                                                 │
│    - ~100x faster inference                                                │
│    - Better suited for location prediction                                 │
│    - Modern, maintainable codebase                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

*Next: [16_SUMMARY_AND_CONCLUSIONS.md](16_SUMMARY_AND_CONCLUSIONS.md) - Final summary and key takeaways*
