# Architecture Comparison: Original vs Proposed

## Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Component-by-Component Comparison](#component-by-component-comparison)
3. [Data Flow Comparison](#data-flow-comparison)
4. [Architectural Diagrams](#architectural-diagrams)
5. [Key Architectural Decisions](#key-architectural-decisions)

---

## High-Level Architecture

### Original Pointer-Generator (Text Summarization)

The original architecture follows a classic **encoder-decoder** paradigm with attention:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL POINTER-GENERATOR ARCHITECTURE                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│   │   INPUT     │         │   ENCODER   │         │  ATTENTION  │          │
│   │  (Article)  │  ───▶   │  (BiLSTM)   │  ───▶   │   (Bahdanau)│          │
│   └─────────────┘         └─────────────┘         └──────┬──────┘          │
│                                                          │                  │
│                                                          ▼                  │
│                           ┌─────────────┐         ┌─────────────┐          │
│                           │   OUTPUT    │  ◀───   │   DECODER   │          │
│                           │  (Summary)  │         │   (LSTM)    │          │
│                           └─────────────┘         └─────────────┘          │
│                                                                              │
│   Key Components:                                                            │
│   • Encoder: Bidirectional LSTM (256 × 2 = 512 dimensional output)          │
│   • Decoder: Unidirectional LSTM (256 dimensional)                          │
│   • Attention: Bahdanau-style additive attention                            │
│   • Pointer: Copy mechanism from source to output                           │
│   • Generator: Vocabulary distribution from decoder state                    │
│   • Coverage: Optional mechanism to prevent repetition                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed PointerNetworkV45 (Next Location Prediction)

The proposed architecture is an **encoder-only** model with pointer-generation output:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  PROPOSED POINTERNETWORKV45 ARCHITECTURE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌─────────────┐         ┌─────────────┐         ┌─────────────┐          │
│   │   INPUT     │         │   FEATURE   │         │ TRANSFORMER │          │
│   │ (Location + │  ───▶   │   FUSION    │  ───▶   │   ENCODER   │          │
│   │  Temporal)  │         │             │         │             │          │
│   └─────────────┘         └─────────────┘         └──────┬──────┘          │
│                                                          │                  │
│                                  ┌───────────────────────┴───────────┐     │
│                                  │                                   │     │
│                                  ▼                                   ▼     │
│                           ┌─────────────┐                   ┌────────────┐ │
│                           │   POINTER   │                   │ GENERATION │ │
│                           │  MECHANISM  │                   │    HEAD    │ │
│                           └──────┬──────┘                   └─────┬──────┘ │
│                                  │                                │        │
│                                  └───────────┬────────────────────┘        │
│                                              ▼                              │
│                                       ┌─────────────┐                      │
│                                       │    GATE     │                      │
│                                       │  (Combine)  │                      │
│                                       └──────┬──────┘                      │
│                                              ▼                              │
│                                       ┌─────────────┐                      │
│                                       │   OUTPUT    │                      │
│                                       │(Next Loc)   │                      │
│                                       └─────────────┘                      │
│                                                                              │
│   Key Components:                                                            │
│   • Feature Fusion: Multi-modal embedding combination                        │
│   • Encoder: Transformer encoder (configurable layers and heads)            │
│   • Pointer: Scaled dot-product attention over input sequence               │
│   • Generator: Linear projection to location vocabulary                      │
│   • Gate: Learned blending of pointer and generator distributions           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component-by-Component Comparison

### 1. Input Processing

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Input Type** | Word tokens | Location IDs + Temporal features |
| **Embedding** | Single word embedding | Multiple embeddings (location, user, temporal) |
| **Embedding Dim** | 128 (emb_dim) | 64-128 (d_model) |
| **Vocabulary** | 50,000 words | ~1,000-2,000 locations |
| **Padding** | [PAD] token (ID=1) | 0 (padding_idx=0) |
| **OOV Handling** | [UNK] token + extended vocab | N/A (all locations known) |

```python
# ORIGINAL: Single embedding lookup
# File: model.py, lines 209-214
embedding = tf.get_variable('embedding', [vsize, hps.emb_dim])
emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)

# PROPOSED: Multiple embeddings combined
# File: pointer_v45.py, lines 194-218
loc_emb = self.loc_emb(x)                    # [batch, seq, d_model]
user_emb = self.user_emb(x_dict['user'])     # [batch, d_model]
temporal = torch.cat([
    self.time_emb(time),                      # [batch, seq, d_model/4]
    self.weekday_emb(weekday),                # [batch, seq, d_model/4]
    self.recency_emb(recency),                # [batch, seq, d_model/4]
    self.duration_emb(duration)               # [batch, seq, d_model/4]
], dim=-1)
```

### 2. Encoder Architecture

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Type** | Bidirectional LSTM | Transformer Encoder |
| **Layers** | 1 layer | 2-3 layers (configurable) |
| **Hidden Size** | 256 per direction | 64-128 (d_model) |
| **Output Size** | 512 (256×2) | 64-128 (d_model) |
| **Attention Heads** | N/A | 4 (configurable) |
| **Position Encoding** | Implicit (LSTM order) | Sinusoidal |
| **Normalization** | None | Pre-LayerNorm |
| **Activation** | tanh (LSTM default) | GELU |

```python
# ORIGINAL: BiLSTM Encoder
# File: model.py, lines 76-94
cell_fw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
cell_bw = tf.contrib.rnn.LSTMCell(self._hps.hidden_dim)
(encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
    cell_fw, cell_bw, encoder_inputs, sequence_length=seq_len
)
encoder_outputs = tf.concat(axis=2, values=encoder_outputs)  # [batch, seq, 512]

# PROPOSED: Transformer Encoder
# File: pointer_v45.py, lines 120-130
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-norm
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

### 3. Decoder Architecture

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Type** | Unidirectional LSTM | None (encoder-only) |
| **Steps** | max_dec_steps (100) | 1 (single output) |
| **Initial State** | Reduced from BiLSTM | N/A |
| **Teacher Forcing** | Yes (during training) | N/A |
| **Beam Search** | Yes (during inference) | No (argmax) |

```python
# ORIGINAL: LSTM Decoder with attention
# File: model.py, lines 124-144
cell = tf.contrib.rnn.LSTMCell(hps.hidden_dim)
outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(
    inputs, self._dec_in_state, self._enc_states, ...
)

# PROPOSED: No decoder - single output from encoder
# File: pointer_v45.py, lines 226-228
batch_idx = torch.arange(batch_size, device=device)
last_idx = (lengths - 1).clamp(min=0)
context = encoded[batch_idx, last_idx]  # Extract last valid position
```

### 4. Attention Mechanism

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Type** | Bahdanau (additive) | Scaled dot-product |
| **Computation** | v^T·tanh(W_h·h + W_s·s + b) | QK^T/√d_model |
| **Masking** | Softmax with re-normalization | -inf masking before softmax |
| **Coverage** | Optional coverage vector | Position bias |

```python
# ORIGINAL: Bahdanau Attention
# File: attention_decoder.py, lines 91-127
decoder_features = linear(decoder_state, attention_vec_size, True)
e = math_ops.reduce_sum(v * math_ops.tanh(encoder_features + decoder_features), [2, 3])
attn_dist = nn_ops.softmax(e)
attn_dist *= enc_padding_mask  # Apply mask
attn_dist /= tf.reduce_sum(attn_dist)  # Re-normalize

# PROPOSED: Scaled Dot-Product Attention
# File: pointer_v45.py, lines 230-236
query = self.pointer_query(context).unsqueeze(1)  # [batch, 1, d_model]
keys = self.pointer_key(encoded)                   # [batch, seq, d_model]
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
ptr_scores = ptr_scores + self.position_bias[pos_from_end]  # Add position bias
ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))     # Mask padding
ptr_probs = F.softmax(ptr_scores, dim=-1)
```

### 5. Pointer-Generation Gate

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Gate Input** | context, cell_state, hidden, input | context only |
| **Gate Computation** | Linear + Sigmoid | MLP (2 layers) + Sigmoid |
| **Distribution Blend** | p_gen × vocab + (1-p_gen) × attn | gate × ptr + (1-gate) × gen |

```python
# ORIGINAL: p_gen calculation
# File: attention_decoder.py, lines 163-168
with tf.variable_scope('calculate_pgen'):
    p_gen = linear([context_vector, state.c, state.h, x], 1, True)
    p_gen = tf.sigmoid(p_gen)

# PROPOSED: gate calculation
# File: pointer_v45.py, lines 140-146
self.ptr_gen_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Sigmoid()
)
```

### 6. Output Layer

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Vocab Size** | 50,000 + OOVs | num_locations (~1,000-2,000) |
| **Extended Vocab** | Yes (for OOVs) | No |
| **Final Operation** | Beam search / sample | Log-softmax + argmax |
| **Output Length** | Variable (up to 100) | Fixed (1 location) |

---

## Data Flow Comparison

### Original Data Flow (Per Decoder Step)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORIGINAL DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. ENCODING (once per input):                                              │
│     Article Words → Word Embeddings → BiLSTM → Encoder States               │
│     [batch, max_enc] → [batch, max_enc, emb_dim] → [batch, max_enc, 512]   │
│                                                                              │
│  2. STATE REDUCTION (once per input):                                        │
│     (fw_state, bw_state) → Linear Layer → Initial Decoder State             │
│     ([batch, 256], [batch, 256]) → [batch, 256]                            │
│                                                                              │
│  3. DECODING (per output step):                                             │
│     For step t = 1 to max_dec_steps:                                        │
│       a. Decoder Input: Previous output embedding → [batch, emb_dim]        │
│       b. Context: Attention over encoder states → [batch, 512]              │
│       c. Decoder Step: LSTM(input + context) → [batch, 256]                │
│       d. Vocab Dist: Linear(decoder_state) → [batch, vocab_size]           │
│       e. Attn Dist: Attention weights → [batch, max_enc]                   │
│       f. p_gen: Sigmoid(concat features) → [batch, 1]                       │
│       g. Final Dist: p_gen × Vocab + (1-p_gen) × Attn → [batch, ext_vocab] │
│                                                                              │
│  4. OUTPUT:                                                                  │
│     Beam Search over final distributions → Generated summary tokens         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Data Flow (Single Pass)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         PROPOSED DATA FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. FEATURE EXTRACTION:                                                      │
│     Locations → Location Embeddings  [batch, seq, d_model]                  │
│     User ID → User Embedding (expanded) [batch, seq, d_model]               │
│     Time → Time Embedding [batch, seq, d_model/4]                           │
│     Weekday → Weekday Embedding [batch, seq, d_model/4]                     │
│     Duration → Duration Embedding [batch, seq, d_model/4]                   │
│     Recency → Recency Embedding [batch, seq, d_model/4]                     │
│     Position → Position-from-End Embedding [batch, seq, d_model/4]          │
│                                                                              │
│  2. FEATURE FUSION:                                                          │
│     Concatenate all features → [batch, seq, 2*d_model + 5*(d_model/4)]     │
│     Linear projection + LayerNorm → [batch, seq, d_model]                   │
│     Add sinusoidal positional encoding → [batch, seq, d_model]              │
│                                                                              │
│  3. ENCODING:                                                                │
│     Transformer Encoder (with padding mask) → [batch, seq, d_model]         │
│                                                                              │
│  4. CONTEXT EXTRACTION:                                                      │
│     Extract last valid position → [batch, d_model]                          │
│                                                                              │
│  5. POINTER ATTENTION:                                                       │
│     Query = Linear(context) → [batch, 1, d_model]                           │
│     Keys = Linear(encoded) → [batch, seq, d_model]                          │
│     Scores = QK^T/√d + position_bias → [batch, seq]                         │
│     Ptr_probs = Softmax(masked_scores) → [batch, seq]                       │
│     Scatter to locations → [batch, num_locations]                           │
│                                                                              │
│  6. GENERATION:                                                              │
│     Gen_probs = Softmax(Linear(context)) → [batch, num_locations]           │
│                                                                              │
│  7. COMBINATION:                                                             │
│     Gate = MLP(context) → [batch, 1]                                        │
│     Final = Gate × Ptr + (1-Gate) × Gen → [batch, num_locations]            │
│     Log_probs = Log(Final + ε) → [batch, num_locations]                     │
│                                                                              │
│  8. OUTPUT:                                                                  │
│     Prediction = Argmax(Log_probs) → [batch]                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Architectural Diagrams

### Detailed Architecture: Original Pointer-Generator

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        ORIGINAL POINTER-GENERATOR (DETAILED)                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUT: "The quick brown fox jumps over the lazy dog"                               │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                         EMBEDDING LAYER                                   │       │
│  │  embedding: [vocab_size=50000, emb_dim=128]                              │       │
│  │  Output: [batch=16, max_enc=400, 128]                                    │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                         ENCODER (BiLSTM)                                  │       │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │       │
│  │  │  Forward LSTM: hidden_dim=256                                      │  │       │
│  │  │  Input: [batch, seq, 128] → Output: [batch, seq, 256]             │  │       │
│  │  └────────────────────────────────────────────────────────────────────┘  │       │
│  │  ┌────────────────────────────────────────────────────────────────────┐  │       │
│  │  │  Backward LSTM: hidden_dim=256                                     │  │       │
│  │  │  Input: [batch, seq, 128] → Output: [batch, seq, 256]             │  │       │
│  │  └────────────────────────────────────────────────────────────────────┘  │       │
│  │  Concat: [batch, seq, 512]                                               │       │
│  │  Final States: fw_st = (c, h), bw_st = (c, h)                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                       STATE REDUCTION                                     │       │
│  │  Concat(fw.c, bw.c) → Linear → ReLU → new_c [batch, 256]                │       │
│  │  Concat(fw.h, bw.h) → Linear → ReLU → new_h [batch, 256]                │       │
│  │  Initial decoder state = LSTMStateTuple(new_c, new_h)                    │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                     DECODER (Attention LSTM)                              │       │
│  │  For each timestep t:                                                     │       │
│  │                                                                           │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Input Fusion                                                   │    │       │
│  │    │  x = Linear([prev_embedding, prev_context], emb_dim)           │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  LSTM Cell                                                      │    │       │
│  │    │  cell_output, state = LSTM(x, state)                           │    │       │
│  │    │  cell_output: [batch, 256]                                     │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Bahdanau Attention                                            │    │       │
│  │    │  e = v^T tanh(W_h·encoder_states + W_s·state.h + b)           │    │       │
│  │    │  α = softmax(e) * mask, then renormalize                       │    │       │
│  │    │  context = Σ(α_i × encoder_state_i)                           │    │       │
│  │    │  Output: context [batch, 512], α [batch, enc_len]             │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  p_gen Calculation                                             │    │       │
│  │    │  p_gen = σ(w_c·context + w_s·state.c + w_h·state.h + w_x·x)   │    │       │
│  │    │  Output: p_gen [batch, 1]                                      │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Output Projection                                             │    │       │
│  │    │  output = Linear([cell_output, context], hidden_dim)          │    │       │
│  │    │  vocab_scores = Linear(output, vocab_size)                    │    │       │
│  │    │  vocab_dist = softmax(vocab_scores)                           │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Final Distribution                                            │    │       │
│  │    │  final_dist = p_gen × vocab_dist + (1-p_gen) × attn_dist      │    │       │
│  │    │  (with extended vocab for OOVs)                                │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │                                                                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  OUTPUT: Generated summary tokens via beam search                                    │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Detailed Architecture: Proposed PointerNetworkV45

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        PROPOSED POINTERNETWORKV45 (DETAILED)                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  INPUTS:                                                                             │
│  x = [101, 205, 150, 312, 150]     # Location IDs (Home, Coffee, Office, Rest, Off) │
│  user = 42                          # User ID (Alice)                                │
│  time = [30, 36, 38, 56, 60]       # Time slots (07:30, 09:00, 09:30, 14:00, 15:00) │
│  weekday = [1, 1, 1, 1, 1]         # Monday                                         │
│  duration = [3, 1, 8, 2, 6]        # Duration buckets                               │
│  diff = [0, 0, 0, 0, 0]            # Days ago (same day)                            │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                       EMBEDDING LAYERS                                    │       │
│  │                                                                           │       │
│  │  loc_emb: [num_locs, d_model=64]     → [batch, seq, 64]                  │       │
│  │  user_emb: [num_users, d_model=64]   → [batch, 64] → expand → [batch, seq, 64]   │
│  │  time_emb: [97, d_model/4=16]        → [batch, seq, 16]                  │       │
│  │  weekday_emb: [8, d_model/4=16]      → [batch, seq, 16]                  │       │
│  │  recency_emb: [9, d_model/4=16]      → [batch, seq, 16]                  │       │
│  │  duration_emb: [100, d_model/4=16]   → [batch, seq, 16]                  │       │
│  │  pos_from_end_emb: [max_len, d_model/4=16] → [batch, seq, 16]            │       │
│  │                                                                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                       FEATURE FUSION                                      │       │
│  │                                                                           │       │
│  │  Concatenate:                                                             │       │
│  │    loc_emb [64] + user_emb [64] + time [16] + weekday [16] +            │       │
│  │    recency [16] + duration [16] + pos_from_end [16]                      │       │
│  │    = [batch, seq, 208]                                                   │       │
│  │                                                                           │       │
│  │  input_proj: Linear(208, 64)                                             │       │
│  │  input_norm: LayerNorm(64)                                               │       │
│  │  + Sinusoidal Positional Encoding [1, seq, 64]                           │       │
│  │                                                                           │       │
│  │  Output: [batch, seq, 64]                                                │       │
│  │                                                                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                     TRANSFORMER ENCODER                                   │       │
│  │                                                                           │       │
│  │  For each layer l = 1 to num_layers (2):                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Pre-LayerNorm                                                  │    │       │
│  │    │  x = LayerNorm(x)                                              │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Multi-Head Self-Attention (nhead=4)                           │    │       │
│  │    │  Q, K, V = Linear(x)                                           │    │       │
│  │    │  Attention = Softmax(QK^T/√d_k) × V                           │    │       │
│  │    │  + Residual connection                                         │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Pre-LayerNorm                                                  │    │       │
│  │    │  x = LayerNorm(x)                                              │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │         ↓                                                                 │       │
│  │    ┌────────────────────────────────────────────────────────────────┐    │       │
│  │    │  Feed-Forward Network                                          │    │       │
│  │    │  FFN(x) = GELU(Linear(x, dim_ff)) × Linear                    │    │       │
│  │    │  + Residual connection                                         │    │       │
│  │    └────────────────────────────────────────────────────────────────┘    │       │
│  │                                                                           │       │
│  │  Output: encoded [batch, seq, 64]                                        │       │
│  │                                                                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                    CONTEXT EXTRACTION                                     │       │
│  │                                                                           │       │
│  │  last_idx = lengths - 1  (e.g., 4 for sequence length 5)                 │       │
│  │  context = encoded[batch_idx, last_idx]                                  │       │
│  │  Output: context [batch, 64]                                             │       │
│  │                                                                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  ┌───────────────────────────────┐    ┌───────────────────────────────┐            │
│  │     POINTER MECHANISM         │    │     GENERATION HEAD           │            │
│  │                               │    │                               │            │
│  │  query = Linear(context, 64)  │    │  gen_head = Linear(64, num_locs)│          │
│  │  keys = Linear(encoded, 64)   │    │  gen_probs = Softmax(gen_head(ctx))│       │
│  │                               │    │                               │            │
│  │  scores = Q·K^T / √64         │    │  Output: gen_probs [batch, num_locs]│      │
│  │  scores += position_bias      │    │                               │            │
│  │  scores.mask(padding, -inf)   │    │                               │            │
│  │  ptr_probs = Softmax(scores)  │    │                               │            │
│  │                               │    │                               │            │
│  │  # Scatter to location vocab  │    │                               │            │
│  │  ptr_dist = zeros(num_locs)   │    │                               │            │
│  │  ptr_dist.scatter_add_(x, ptr_probs)│   │                               │       │
│  │                               │    │                               │            │
│  │  Output: ptr_dist [batch, num_locs]│   │                               │       │
│  │                               │    │                               │            │
│  └───────────────────────────────┘    └───────────────────────────────┘            │
│         ↓                                         ↓                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐       │
│  │                          GATED COMBINATION                                │       │
│  │                                                                           │       │
│  │  ptr_gen_gate = Sequential(                                              │       │
│  │      Linear(64, 32) → GELU → Linear(32, 1) → Sigmoid                    │       │
│  │  )                                                                        │       │
│  │  gate = ptr_gen_gate(context)  # [batch, 1]                              │       │
│  │                                                                           │       │
│  │  final_probs = gate × ptr_dist + (1 - gate) × gen_probs                  │       │
│  │  log_probs = log(final_probs + 1e-10)                                    │       │
│  │                                                                           │       │
│  │  Output: log_probs [batch, num_locations]                                │       │
│  │                                                                           │       │
│  └──────────────────────────────────────────────────────────────────────────┘       │
│         ↓                                                                            │
│  OUTPUT: Predicted next location = argmax(log_probs)                                │
│          e.g., location_id = 89 (Gym)                                               │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Architectural Decisions

### Why Transformer Instead of BiLSTM?

| Factor | BiLSTM (Original) | Transformer (Proposed) |
|--------|-------------------|------------------------|
| **Parallelization** | Sequential O(n) | Parallel O(1) |
| **Long-range Dependencies** | Gradient issues | Direct attention |
| **Training Speed** | Slower | Faster |
| **Interpretability** | Hidden state | Attention weights |

**Justification**: Location sequences are typically short (5-50 visits), but require understanding global patterns. Transformer's self-attention can directly model relationships between any two visits (e.g., Home→Office patterns across different days).

### Why No Decoder?

| Original (seq2seq) | Proposed (encoder-only) |
|--------------------|------------------------|
| Generate multiple tokens | Predict single next location |
| Variable output length | Fixed output size |
| Teacher forcing needed | Direct prediction |
| Beam search for inference | Simple argmax |

**Justification**: Next location prediction is a classification task (pick one from num_locations), not a generation task. A decoder adds unnecessary complexity.

### Why Multiple Embeddings?

| Original | Proposed |
|----------|----------|
| Only word meaning | Location + Context |
| Words have semantics | Locations need temporal context |
| Vocabulary captures meaning | Time, day, user matter |

**Justification**: In mobility prediction, **when** you visit is as important as **what** you visit. Going to a restaurant at noon vs midnight has different implications. User habits are highly personal.

### Why Position-from-End Embedding?

```
Sequence: [Home, Coffee, Office, Restaurant, Office]
Position: [  1,    2,      3,       4,        5    ]  # Original position
Pos-End:  [  5,    4,      3,       2,        1    ]  # Position from end

The last visit (Office at position 5) has pos_from_end = 1
The first visit (Home at position 1) has pos_from_end = 5
```

**Justification**: Recent visits are more predictive than older ones. By encoding position-from-end, the model learns that pos_from_end=1 means "most recent" regardless of sequence length.

---

## Summary Table: Architecture Components

| Component | Original Location | Proposed Location | Key Difference |
|-----------|-------------------|-------------------|----------------|
| Embedding | `model.py:209-214` | `pointer_v45.py:99-109` | Single vs Multi-modal |
| Encoder | `model.py:76-94` | `pointer_v45.py:120-130` | BiLSTM vs Transformer |
| Attention | `attention_decoder.py:79-129` | `pointer_v45.py:230-236` | Bahdanau vs Scaled Dot |
| Decoder | `model.py:124-144` | N/A | Present vs Removed |
| Gate | `attention_decoder.py:163-168` | `pointer_v45.py:140-146` | Linear vs MLP |
| Output | `model.py:146-183` | `pointer_v45.py:239-249` | Extended vocab vs Fixed |

---

*Next: [03_ENCODER_COMPARISON.md](03_ENCODER_COMPARISON.md) - Deep dive into encoder differences*
