# Embedding Comparison: Original vs Proposed

## Table of Contents
1. [Overview](#overview)
2. [Original Word Embeddings](#original-word-embeddings)
3. [Proposed Multi-Modal Embeddings](#proposed-multi-modal-embeddings)
4. [Embedding Architecture Diagrams](#embedding-architecture-diagrams)
5. [Code Comparison](#code-comparison)
6. [Example Walkthrough](#example-walkthrough)
7. [Justification for Multi-Modal Approach](#justification-for-multi-modal-approach)

---

## Overview

The embedding layer is the first processing step that converts discrete inputs into continuous representations. The two models have fundamentally different input types and therefore different embedding strategies:

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Input Type** | Words (text) | Locations + Context |
| **Embedding Count** | 1 (word only) | 7 (location, user, time, weekday, duration, recency, position) |
| **Total Dimension** | 128 | 208 (before projection) → 64-128 (after projection) |
| **Shared Embedding** | Yes (encoder/decoder share) | N/A (no decoder) |

---

## Original Word Embeddings

### Architecture

The original model uses a single embedding matrix shared between encoder and decoder:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ORIGINAL WORD EMBEDDING                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Vocabulary:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │ ID 0: [UNK]      ID 1: [PAD]      ID 2: [START]    ID 3: [STOP]       │ │
│  │ ID 4: "the"      ID 5: "a"        ID 6: "is"       ID 7: "to"         │ │
│  │ ...                                                                    │ │
│  │ ID 49999: "rarely_used_word"                                          │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Embedding Matrix:                                                           │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  embedding: [vocab_size × emb_dim] = [50000 × 128]                    │ │
│  │                                                                        │ │
│  │  Total parameters: 50000 × 128 = 6,400,000                            │ │
│  │                                                                        │ │
│  │     Word ID →  ┌─────────────────────────────────────────────────┐    │ │
│  │                │ embedding[word_id] = [e₁, e₂, ..., e₁₂₈]        │    │ │
│  │                └─────────────────────────────────────────────────┘    │ │
│  │                                 ↓                                      │ │
│  │                    Word vector [128 dims]                              │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Usage:                                                                      │
│  - Encoder: emb_enc_inputs = embedding_lookup(embedding, enc_batch)         │
│  - Decoder: emb_dec_inputs = embedding_lookup(embedding, dec_batch)         │
│                                                                              │
│  Both encoder and decoder share the same embedding matrix!                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original Code

```python
# File: model.py, lines 199-214

def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    hps = self._hps
    vsize = self._vocab.size()  # 50000
    
    with tf.variable_scope('seq2seq'):
        # Initializers
        self.rand_unif_init = tf.random_uniform_initializer(
            -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123
        )
        self.trunc_norm_init = tf.truncated_normal_initializer(
            stddev=hps.trunc_norm_init_std
        )
        
        # Add embedding matrix (shared by encoder and decoder)
        with tf.variable_scope('embedding'):
            embedding = tf.get_variable(
                'embedding', 
                [vsize, hps.emb_dim],  # [50000, 128]
                dtype=tf.float32, 
                initializer=self.trunc_norm_init
            )
            
            # Add TensorBoard visualization
            if hps.mode == "train":
                self._add_emb_vis(embedding)
            
            # Encoder input embeddings
            emb_enc_inputs = tf.nn.embedding_lookup(
                embedding, 
                self._enc_batch  # [batch, max_enc_steps]
            )
            # Shape: [batch, max_enc_steps, emb_dim] = [16, ≤400, 128]
            
            # Decoder input embeddings (list of tensors for each step)
            emb_dec_inputs = [
                tf.nn.embedding_lookup(embedding, x) 
                for x in tf.unstack(self._dec_batch, axis=1)
            ]
            # List of max_dec_steps tensors, each [batch, emb_dim]
```

### Embedding Initialization

```python
# File: run_summarization.py, lines 57-58

tf.app.flags.DEFINE_float('rand_unif_init_mag', 0.02, 
    'magnitude for lstm cells random uniform initialization')
tf.app.flags.DEFINE_float('trunc_norm_init_std', 1e-4, 
    'std of trunc norm init, used for initializing everything else')

# Embedding uses truncated normal: N(0, 0.0001)
# This produces small initial values close to zero
```

---

## Proposed Multi-Modal Embeddings

### Architecture

The proposed model uses multiple embeddings for different feature types:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                   PROPOSED MULTI-MODAL EMBEDDINGS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input Features for one visit:                                               │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  Location ID: 150 (Office)                                             │ │
│  │  User ID: 42 (Alice)                                                   │ │
│  │  Time: 38 (09:30 → slot 38 out of 96)                                 │ │
│  │  Weekday: 1 (Monday)                                                   │ │
│  │  Duration: 8 (4 hours → bucket 8)                                      │ │
│  │  Recency: 0 (same day)                                                 │ │
│  │  Position from end: 3 (third from last)                                │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Embedding Matrices:                                                         │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │                                                                        │ │
│  │  1. Location Embedding: [num_locations × d_model]                      │ │
│  │     e.g., [2000 × 64] = 128,000 params                                │ │
│  │     Output: [batch, seq, 64]                                          │ │
│  │                                                                        │ │
│  │  2. User Embedding: [num_users × d_model]                             │ │
│  │     e.g., [100 × 64] = 6,400 params                                   │ │
│  │     Output: [batch, 64] → expand to [batch, seq, 64]                  │ │
│  │                                                                        │ │
│  │  3. Time Embedding: [97 × d_model/4]                                  │ │
│  │     97 = 96 time slots + 1 padding                                    │ │
│  │     e.g., [97 × 16] = 1,552 params                                    │ │
│  │     Output: [batch, seq, 16]                                          │ │
│  │                                                                        │ │
│  │  4. Weekday Embedding: [8 × d_model/4]                                │ │
│  │     8 = 7 days + 1 padding                                            │ │
│  │     e.g., [8 × 16] = 128 params                                       │ │
│  │     Output: [batch, seq, 16]                                          │ │
│  │                                                                        │ │
│  │  5. Duration Embedding: [100 × d_model/4]                             │ │
│  │     100 buckets for different durations                                │ │
│  │     e.g., [100 × 16] = 1,600 params                                   │ │
│  │     Output: [batch, seq, 16]                                          │ │
│  │                                                                        │ │
│  │  6. Recency Embedding: [9 × d_model/4]                                │ │
│  │     9 = 8 recency levels + 1 padding                                  │ │
│  │     e.g., [9 × 16] = 144 params                                       │ │
│  │     Output: [batch, seq, 16]                                          │ │
│  │                                                                        │ │
│  │  7. Position-from-End Embedding: [max_seq_len × d_model/4]            │ │
│  │     e.g., [150 × 16] = 2,400 params                                   │ │
│  │     Output: [batch, seq, 16]                                          │ │
│  │                                                                        │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Concatenation:                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  concat([loc, user, time, weekday, duration, recency, pos_from_end])  │ │
│  │  = [64 + 64 + 16 + 16 + 16 + 16 + 16] = 208 dimensions               │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
│  Projection:                                                                 │
│  ┌────────────────────────────────────────────────────────────────────────┐ │
│  │  input_proj: Linear(208 → d_model) = Linear(208 → 64)                 │ │
│  │  input_norm: LayerNorm(d_model) = LayerNorm(64)                       │ │
│  │                                                                        │ │
│  │  Output: [batch, seq, 64]                                             │ │
│  └────────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Code

```python
# File: pgt.py, lines 98-114

def __init__(self, ...):
    super().__init__()
    
    self.num_locations = num_locations
    self.d_model = d_model
    self.max_seq_len = max_seq_len
    
    # Core embeddings (full d_model dimension)
    self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
    self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
    
    # Temporal embeddings (d_model/4 dimension each)
    self.time_emb = nn.Embedding(97, d_model // 4)      # 96 intervals + padding
    self.weekday_emb = nn.Embedding(8, d_model // 4)    # 7 days + padding
    self.recency_emb = nn.Embedding(9, d_model // 4)    # 8 levels + padding
    self.duration_emb = nn.Embedding(100, d_model // 4) # 100 buckets
    
    # Position from end embedding
    self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
    
    # Input projection: combine all features
    # Location (d_model) + User (d_model) + 5 temporal features (d_model/4 each)
    input_dim = d_model * 2 + d_model // 4 * 5  # = 208 when d_model=64
    self.input_proj = nn.Linear(input_dim, d_model)
    self.input_norm = nn.LayerNorm(d_model)

# File: pgt.py, lines 194-219

def forward(self, x, x_dict):
    x = x.T  # Convert from [seq, batch] to [batch, seq]
    batch_size, seq_len = x.shape
    device = x.device
    lengths = x_dict['len']
    
    # Location embedding
    loc_emb = self.loc_emb(x)  # [batch, seq, d_model]
    
    # User embedding (expand to sequence length)
    user_emb = self.user_emb(x_dict['user'])  # [batch, d_model]
    user_emb = user_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [batch, seq, d_model]
    
    # Temporal features (with clamping to valid ranges)
    time = torch.clamp(x_dict['time'].T, 0, 96)
    weekday = torch.clamp(x_dict['weekday'].T, 0, 7)
    recency = torch.clamp(x_dict['diff'].T, 0, 8)
    duration = torch.clamp(x_dict['duration'].T, 0, 99)
    
    temporal = torch.cat([
        self.time_emb(time),        # [batch, seq, d_model/4]
        self.weekday_emb(weekday),  # [batch, seq, d_model/4]
        self.recency_emb(recency),  # [batch, seq, d_model/4]
        self.duration_emb(duration) # [batch, seq, d_model/4]
    ], dim=-1)  # [batch, seq, d_model]
    
    # Position from end
    positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
    pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
    pos_emb = self.pos_from_end_emb(pos_from_end)  # [batch, seq, d_model/4]
    
    # Combine all features
    combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
    # [batch, seq, 208]
    
    # Project to d_model and normalize
    hidden = self.input_norm(self.input_proj(combined))  # [batch, seq, d_model]
```

---

## Embedding Architecture Diagrams

### Original: Single Embedding Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL EMBEDDING FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: "The quick brown fox jumps"                                         │
│                                                                              │
│         ┌─────┬─────┬─────┬─────┬─────┐                                    │
│  Tokens:│ The │quick│brown│ fox │jumps│                                    │
│         └──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘                                    │
│            │     │     │     │     │                                        │
│            ▼     ▼     ▼     ▼     ▼                                        │
│         ┌─────────────────────────────┐                                    │
│         │    Vocabulary Lookup        │                                    │
│         │    (word → ID)              │                                    │
│         └─────────────────────────────┘                                    │
│            │     │     │     │     │                                        │
│  IDs:      4    102   543   1205  892                                       │
│            │     │     │     │     │                                        │
│            ▼     ▼     ▼     ▼     ▼                                        │
│         ┌─────────────────────────────┐                                    │
│         │    Embedding Matrix         │                                    │
│         │    [50000 × 128]            │                                    │
│         │                             │                                    │
│         │    embedding[4]    → e₁     │                                    │
│         │    embedding[102]  → e₂     │                                    │
│         │    embedding[543]  → e₃     │                                    │
│         │    embedding[1205] → e₄     │                                    │
│         │    embedding[892]  → e₅     │                                    │
│         └─────────────────────────────┘                                    │
│            │     │     │     │     │                                        │
│            ▼     ▼     ▼     ▼     ▼                                        │
│         ┌─────┬─────┬─────┬─────┬─────┐                                    │
│  Output:│ e₁  │ e₂  │ e₃  │ e₄  │ e₅  │  Each eᵢ ∈ ℝ¹²⁸                   │
│         └─────┴─────┴─────┴─────┴─────┘                                    │
│                                                                              │
│  Shape: [batch=16, seq=5, dim=128]                                          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Multi-Modal Embedding Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED EMBEDDING FLOW                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input for one position:                                                     │
│  ┌──────────────────────────────────────────────────────────────────┐       │
│  │ location=150, user=42, time=38, weekday=1, duration=8, diff=0   │       │
│  │ position_from_end=3                                               │       │
│  └──────────────────────────────────────────────────────────────────┘       │
│                                                                              │
│         ┌─────────────────────────────────────────────────────────────┐     │
│         │                   EMBEDDING LOOKUPS                          │     │
│         │                                                              │     │
│   loc=150 │ ──▶ loc_emb[150]     ─▶ [64 dims] ─────────────────────┐  │     │
│         │                                                           │  │     │
│  user=42 │ ──▶ user_emb[42]      ─▶ [64 dims] ───────────────────┐ │  │     │
│         │                                                         │ │  │     │
│  time=38 │ ──▶ time_emb[38]      ─▶ [16 dims] ─────────────────┐ │ │  │     │
│         │                                                       │ │ │  │     │
│ weekday=1│ ──▶ weekday_emb[1]    ─▶ [16 dims] ───────────────┐ │ │ │  │     │
│         │                                                     │ │ │ │  │     │
│duration=8│ ──▶ duration_emb[8]   ─▶ [16 dims] ─────────────┐ │ │ │ │  │     │
│         │                                                   │ │ │ │ │  │     │
│  diff=0  │ ──▶ recency_emb[0]    ─▶ [16 dims] ───────────┐ │ │ │ │ │  │     │
│         │                                                 │ │ │ │ │ │  │     │
│pos_end=3 │ ──▶ pos_from_end_emb[3]─▶ [16 dims] ─────────┐│ │ │ │ │ │  │     │
│         │                                               ││ │ │ │ │ │  │     │
│         └───────────────────────────────────────────────┼┼─┼─┼─┼─┼─┼──┘     │
│                                                         ││ │ │ │ │ │        │
│         ┌───────────────────────────────────────────────┼┼─┼─┼─┼─┼─┼──┐     │
│         │                  CONCATENATION                ▼▼ ▼ ▼ ▼ ▼ ▼  │     │
│         │                                                              │     │
│         │  [loc(64), user(64), time(16), day(16), dur(16), rec(16), pos(16)]│
│         │  ────────────────────────────────────────────────────────────│     │
│         │                    Total: 208 dimensions                     │     │
│         │                                                              │     │
│         └───────────────────────────────────────────────┬──────────────┘     │
│                                                         │                    │
│         ┌───────────────────────────────────────────────▼──────────────┐     │
│         │                   PROJECTION                                  │     │
│         │                                                              │     │
│         │  Linear(208 → 64) + LayerNorm(64)                           │     │
│         │                                                              │     │
│         │  Output: [64 dimensions]                                    │     │
│         │                                                              │     │
│         └───────────────────────────────────────────────┬──────────────┘     │
│                                                         │                    │
│                                                         ▼                    │
│                                  Final embedding for position: [64 dims]     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Full Sequence Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              MULTI-MODAL EMBEDDING FOR FULL SEQUENCE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Sequence: [Home, Coffee, Office, Restaurant, Office]                        │
│  User: Alice (42)                                                            │
│                                                                              │
│  Position:     1          2          3           4          5                │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  loc_emb:   [64]       [64]       [64]        [64]       [64]               │
│              │          │          │           │          │                  │
│  user_emb:  [64]       [64]       [64]        [64]       [64]  (same Alice)  │
│              │          │          │           │          │                  │
│  time_emb:  [16]       [16]       [16]        [16]       [16]               │
│              │          │          │           │          │                  │
│  week_emb:  [16]       [16]       [16]        [16]       [16]  (all Monday)  │
│              │          │          │           │          │                  │
│  dur_emb:   [16]       [16]       [16]        [16]       [16]               │
│              │          │          │           │          │                  │
│  rec_emb:   [16]       [16]       [16]        [16]       [16]  (all day 0)   │
│              │          │          │           │          │                  │
│  pos_emb:   [16]       [16]       [16]        [16]       [16]               │
│              │          │          │           │          │                  │
│  ─────────────────────────────────────────────────────────────────────────   │
│                                                                              │
│  concat:   [208]      [208]      [208]       [208]      [208]               │
│              │          │          │           │          │                  │
│              ▼          ▼          ▼           ▼          ▼                  │
│           proj+norm  proj+norm  proj+norm  proj+norm  proj+norm             │
│              │          │          │           │          │                  │
│              ▼          ▼          ▼           ▼          ▼                  │
│  output:   [64]       [64]       [64]        [64]       [64]                │
│                                                                              │
│  Final shape: [batch, seq=5, d_model=64]                                    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Comparison

### Side-by-Side

```python
# ==============================================================================
# ORIGINAL: Single Word Embedding (TensorFlow)
# ==============================================================================

# Configuration
vocab_size = 50000
emb_dim = 128

# Definition
with tf.variable_scope('embedding'):
    embedding = tf.get_variable(
        'embedding', 
        [vocab_size, emb_dim],
        dtype=tf.float32,
        initializer=tf.truncated_normal_initializer(stddev=1e-4)
    )

# Usage
emb_enc_inputs = tf.nn.embedding_lookup(embedding, enc_batch)
# Shape: [batch, seq, 128]

# That's it! Single embedding, single lookup.

# ==============================================================================
# PROPOSED: Multi-Modal Embeddings (PyTorch)
# ==============================================================================

# Configuration
num_locations = 2000
num_users = 100
d_model = 64

# Definition
class PointerGeneratorTransformer(nn.Module):
    def __init__(self, ...):
        # Core embeddings (full dimension)
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
        
        # Temporal embeddings (quarter dimension each)
        self.time_emb = nn.Embedding(97, d_model // 4)       # 96 slots + pad
        self.weekday_emb = nn.Embedding(8, d_model // 4)     # 7 days + pad
        self.recency_emb = nn.Embedding(9, d_model // 4)     # 8 levels + pad
        self.duration_emb = nn.Embedding(100, d_model // 4)  # 100 buckets
        
        # Position embedding
        self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
        
        # Projection layer
        input_dim = d_model * 2 + d_model // 4 * 5  # 208 when d_model=64
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

# Usage
def forward(self, x, x_dict):
    # Multiple lookups
    loc_emb = self.loc_emb(x)
    user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
    
    temporal = torch.cat([
        self.time_emb(x_dict['time'].T),
        self.weekday_emb(x_dict['weekday'].T),
        self.recency_emb(x_dict['diff'].T),
        self.duration_emb(x_dict['duration'].T)
    ], dim=-1)
    
    pos_emb = self.pos_from_end_emb(pos_from_end)
    
    # Combine all
    combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
    hidden = self.input_norm(self.input_proj(combined))
    
    # Shape: [batch, seq, 64]
```

### Parameter Count Comparison

```
ORIGINAL:
═══════════════════════════════════════════════════════════════════════════════
Embedding Parameters:
  embedding: 50000 × 128 = 6,400,000 parameters

Total: 6.4M parameters just for embeddings!
═══════════════════════════════════════════════════════════════════════════════

PROPOSED (with d_model=64, num_locations=2000, num_users=100):
═══════════════════════════════════════════════════════════════════════════════
Embedding Parameters:
  loc_emb:      2000 × 64 = 128,000
  user_emb:      100 × 64 = 6,400
  time_emb:       97 × 16 = 1,552
  weekday_emb:     8 × 16 = 128
  recency_emb:     9 × 16 = 144
  duration_emb:  100 × 16 = 1,600
  pos_from_end: 151 × 16 = 2,416
  ──────────────────────────────
  Subtotal:              140,240

Projection Parameters:
  input_proj: 208 × 64 + 64 = 13,376
  input_norm: 64 + 64 = 128
  ──────────────────────────────
  Subtotal:               13,504

Total: ~154K parameters (vs 6.4M in original)
═══════════════════════════════════════════════════════════════════════════════
```

---

## Example Walkthrough

### Alice's Visit at Position 3 (Office)

```
Input Data:
  location_id = 150 (Office)
  user_id = 42 (Alice)
  time = 38 (09:30 → slot 38)
  weekday = 1 (Monday)
  duration = 8 (4 hours → bucket 8)
  recency = 0 (same day)
  position = 3, length = 5 → pos_from_end = 5 - 3 = 2

Step 1: Embedding Lookups
═══════════════════════════════════════════════════════════════════════════════

loc_emb[150] = [0.12, -0.34, 0.56, ..., 0.23]   # 64 dimensions
               ↓
               Represents: "This is an Office location"

user_emb[42] = [-0.11, 0.45, -0.22, ..., 0.67]  # 64 dimensions
               ↓
               Represents: "This is Alice's behavior pattern"

time_emb[38] = [0.33, -0.12, 0.78, ..., 0.11]   # 16 dimensions
               ↓
               Represents: "This is mid-morning time"

weekday_emb[1] = [0.21, 0.54, ..., -0.33]       # 16 dimensions
                 ↓
                 Represents: "This is Monday (workday)"

duration_emb[8] = [-0.44, 0.23, ..., 0.56]      # 16 dimensions
                  ↓
                  Represents: "This is a long stay (4 hours)"

recency_emb[0] = [0.88, -0.11, ..., 0.33]       # 16 dimensions
                 ↓
                 Represents: "This happened today (very recent)"

pos_from_end_emb[2] = [0.55, 0.22, ..., -0.44]  # 16 dimensions
                      ↓
                      Represents: "This is second from last"

═══════════════════════════════════════════════════════════════════════════════

Step 2: Concatenation
═══════════════════════════════════════════════════════════════════════════════

combined = [loc_emb, user_emb, time_emb, weekday_emb, 
            duration_emb, recency_emb, pos_from_end_emb]

         = [64 dims | 64 dims | 16 dims | 16 dims | 16 dims | 16 dims | 16 dims]
         
         = [208 dimensions total]

Visualization:
┌────────────────────────────────────────────────────────────────────────────┐
│ loc_emb [████████████████████████████████████████████████████████████████] │
│ usr_emb [████████████████████████████████████████████████████████████████] │
│ time    [████████████████]                                                  │
│ weekday [████████████████]                                                  │
│ duration[████████████████]                                                  │
│ recency [████████████████]                                                  │
│ pos_end [████████████████]                                                  │
└────────────────────────────────────────────────────────────────────────────┘
                              ↓
                     [208 dimensions]

═══════════════════════════════════════════════════════════════════════════════

Step 3: Projection and Normalization
═══════════════════════════════════════════════════════════════════════════════

projected = Linear(combined)    # [208] → [64]
            ↓
            W @ combined + b where W ∈ ℝ^(64×208), b ∈ ℝ^64

normalized = LayerNorm(projected)
             ↓
             (projected - mean) / std * γ + β

Final embedding for this position: [64 dimensions]

═══════════════════════════════════════════════════════════════════════════════
```

### What Each Embedding Captures

```
┌─────────────────────────────────────────────────────────────────────────────┐
│              SEMANTIC MEANING OF EACH EMBEDDING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LOCATION EMBEDDING (64 dims):                                              │
│  ─────────────────────────────                                              │
│  Captures: Location semantics and relationships                              │
│    - Office locations cluster together                                       │
│    - Restaurants cluster together                                            │
│    - Similar locations have similar embeddings                               │
│                                                                              │
│  Example learned relationships:                                              │
│    Office_A ≈ Office_B (both are workplaces)                                │
│    Restaurant_1 ≈ Restaurant_2 (both are food places)                       │
│    Home ≠ Office (different semantics)                                      │
│                                                                              │
│  USER EMBEDDING (64 dims):                                                  │
│  ────────────────────────                                                   │
│  Captures: User-specific behavior patterns                                   │
│    - Alice always goes to gym after work                                    │
│    - Bob prefers restaurants over cooking at home                           │
│    - Similar users have similar embeddings                                   │
│                                                                              │
│  TIME EMBEDDING (16 dims):                                                  │
│  ─────────────────────────                                                  │
│  Captures: Time-of-day patterns                                              │
│    - Morning times cluster (commute to work)                                │
│    - Lunch times cluster (food-related)                                     │
│    - Evening times cluster (leisure activities)                              │
│                                                                              │
│  WEEKDAY EMBEDDING (16 dims):                                               │
│  ───────────────────────────                                                │
│  Captures: Day-of-week patterns                                              │
│    - Mon-Fri cluster (workdays)                                             │
│    - Sat-Sun cluster (weekends)                                             │
│                                                                              │
│  DURATION EMBEDDING (16 dims):                                              │
│  ────────────────────────────                                               │
│  Captures: Stay duration patterns                                            │
│    - Short stays: Coffee shops, gas stations                                │
│    - Medium stays: Restaurants, gyms                                         │
│    - Long stays: Home, office                                               │
│                                                                              │
│  RECENCY EMBEDDING (16 dims):                                               │
│  ───────────────────────────                                                │
│  Captures: How recent the visit was                                          │
│    - Today's visits are more relevant                                        │
│    - Yesterday's visits are somewhat relevant                               │
│    - Week-old visits are less relevant                                      │
│                                                                              │
│  POSITION-FROM-END EMBEDDING (16 dims):                                     │
│  ─────────────────────────────────────                                      │
│  Captures: Sequence position importance                                      │
│    - Most recent (pos=1): Very important                                    │
│    - Second recent (pos=2): Important                                       │
│    - Oldest (pos=N): Less important                                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Justification for Multi-Modal Approach

### Why Multiple Embeddings?

| Feature | Why It Matters | Example |
|---------|----------------|---------|
| **Location** | Core entity being predicted | Office ≠ Restaurant |
| **User** | Personalization | Alice goes to gym, Bob doesn't |
| **Time** | Temporal patterns | 9am → work, 8pm → dinner |
| **Weekday** | Weekly patterns | Monday → office, Saturday → park |
| **Duration** | Stay type | 4 hours = work, 30 min = coffee |
| **Recency** | Relevance decay | Today's data > last week's |
| **Position** | Sequence structure | Recent visits matter more |

### Why Dimension Allocation?

```
Dimension Allocation Rationale:
═══════════════════════════════════════════════════════════════════════════════

Location (d_model = 64):
  - Core feature, needs most capacity
  - Many unique locations (2000+)
  - Complex relationships to learn

User (d_model = 64):
  - Critical for personalization
  - Each user has unique patterns
  - Affects all other features

Temporal Features (d_model/4 = 16 each):
  - Auxiliary features
  - Limited vocabulary (96 times, 7 days, etc.)
  - Provide context but not primary signal

Position (d_model/4 = 16):
  - Structural feature
  - Relative importance indicator
  - Works with attention mechanism

═══════════════════════════════════════════════════════════════════════════════
```

### Comparison with Original

```
ORIGINAL (Text Summarization):
═══════════════════════════════════════════════════════════════════════════════
- Words carry all semantic information
- Context comes from sequence structure
- No temporal features needed
- No user personalization needed
- Single embedding is sufficient

Example: "The quick brown fox"
  "The" → embedding captures article meaning
  "quick" → embedding captures adjective meaning
  "brown" → embedding captures color meaning
  "fox" → embedding captures animal meaning

═══════════════════════════════════════════════════════════════════════════════

PROPOSED (Location Prediction):
═══════════════════════════════════════════════════════════════════════════════
- Location ID alone is insufficient
- Same location at different times → different meaning
- Same location for different users → different patterns
- Temporal context is crucial

Example: Location "Office" (ID=150)
  Office at 9am on Monday → Start of work
  Office at 9pm on Saturday → Working overtime (unusual!)
  Office for Alice → Regular (she works there)
  Office for Bob → Meeting (he doesn't work there)

Same location ID, completely different interpretations!
Multi-modal embeddings capture this richness.

═══════════════════════════════════════════════════════════════════════════════
```

---

## Summary

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Number of Embeddings** | 1 | 7 |
| **Input Features** | Word ID only | Location + User + 5 temporal |
| **Embedding Dimension** | 128 | Varies (64, 16) |
| **Parameter Count** | 6.4M | ~154K |
| **Personalization** | None | User embedding |
| **Temporal Awareness** | None | Time, weekday, duration, recency |
| **Position Handling** | Implicit (RNN) | Explicit embedding |
| **Projection** | None | Linear + LayerNorm |

The multi-modal embedding approach is a key innovation that enables the proposed model to capture the rich contextual information necessary for accurate location prediction.

---

*Next: [07_TRAINING_PIPELINE.md](07_TRAINING_PIPELINE.md) - Training configuration and optimization*
