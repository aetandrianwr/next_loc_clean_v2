# Code Walkthrough: Proposed Model (PointerGeneratorTransformer)

## Table of Contents
1. [File Overview](#file-overview)
2. [Import Section](#import-section)
3. [Class Definition and Initialization](#class-definition-and-initialization)
4. [Embedding Layers](#embedding-layers)
5. [Transformer Encoder](#transformer-encoder)
6. [Pointer Attention](#pointer-attention)
7. [Gate Mechanism](#gate-mechanism)
8. [Forward Pass](#forward-pass)
9. [Helper Methods](#helper-methods)
10. [Complete Annotated Code](#complete-annotated-code)

---

## File Overview

```
File: /workspace/next_loc_clean_v2/src/models/proposed/pgt.py
Total Lines: ~254
Main Class: PointerGeneratorTransformer

Purpose: Implements a pointer network with Transformer encoder for 
         next location prediction in mobility trajectories.
```

---

## Import Section

```python
# Lines 1-10: Standard imports
import torch                           # PyTorch core
import torch.nn as nn                  # Neural network modules
import torch.nn.functional as F        # Functional operations
import math                            # For positional encoding
from typing import Dict, Optional, Tuple  # Type hints

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPARISON TO ORIGINAL:                                                │
# │                                                                         │
# │  Original uses TensorFlow 1.x:                                         │
# │    import tensorflow as tf                                             │
# │    from tensorflow.contrib.tensorboard.plugins import projector        │
# │                                                                         │
# │  Key difference: PyTorch uses eager execution by default,              │
# │  while TensorFlow 1.x uses static computation graphs.                  │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Class Definition and Initialization

### Lines 12-55: Constructor

```python
class PointerGeneratorTransformer(nn.Module):
    """
    Pointer Generator Transformer: A hybrid pointer-generator model for next location prediction.
    
    Key Features:
    - Multi-modal embeddings (location, user, time, weekday, duration, recency)
    - Transformer encoder (self-attention over trajectory)
    - Pointer mechanism (copy from input sequence)
    - Generation mechanism (predict any location)
    - Learned gate to combine pointer and generation
    """
    
    def __init__(
        self,
        num_locations: int,          # Size of location vocabulary
        num_users: int,              # Number of unique users
        d_model: int = 64,           # Embedding/hidden dimension
        nhead: int = 4,              # Number of attention heads
        num_layers: int = 2,         # Number of Transformer layers
        dim_feedforward: int = 128,  # FFN hidden dimension
        dropout: float = 0.1,        # Dropout rate
        num_time_slots: int = 24,    # Time discretization (hours)
        num_weekdays: int = 7,       # Days of week
        num_duration_bins: int = 20, # Duration categories
        num_recency_bins: int = 20,  # Recency categories
        max_seq_len: int = 100,      # Max sequence length
        **kwargs                     # Additional config
    ):
        super().__init__()
        
        # Store configuration
        self.num_locations = num_locations
        self.d_model = d_model
        self.num_users = num_users
        self.max_seq_len = max_seq_len

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  LINE-BY-LINE EXPLANATION:                                              │
# │                                                                         │
# │  Line 12: Inherit from nn.Module (PyTorch base class)                  │
# │           Original inherits from object (no base class)                │
# │                                                                         │
# │  Lines 25-35: Constructor parameters                                   │
# │    - num_locations: ~500 (vs 50,000 vocab in original)                │
# │    - d_model: 64 (vs hidden_dim=256 in original)                      │
# │    - nhead: 4 attention heads (original has 1 implicit head)          │
# │    - num_layers: 2 Transformer layers (original has 1 BiLSTM layer)   │
# │                                                                         │
# │  Lines 36-40: Store config for later use                               │
# │           Original stores config via hps object                        │
# │                                                                         │
# │  ORIGINAL EQUIVALENT (model.py lines 30-50):                          │
# │    def __init__(self, hps, vocab):                                    │
# │        self._hps = hps                                                 │
# │        self._vocab = vocab                                             │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Embedding Layers

### Lines 56-114: Multi-Modal Embeddings

```python
        # ================================================================
        # EMBEDDING LAYERS
        # ================================================================
        
        # 1. Location Embedding (REQUIRED)
        # Maps location IDs to dense vectors
        self.location_embedding = nn.Embedding(
            num_embeddings=num_locations,  # 500 locations
            embedding_dim=d_model,          # 64 dimensions
            padding_idx=0                   # Padding token = 0
        )
        
        # 2. User Embedding (OPTIONAL)
        # Captures user-specific patterns
        self.user_embedding = nn.Embedding(
            num_embeddings=num_users,       # 200 users
            embedding_dim=d_model           # 64 dimensions
        )
        
        # 3. Time Embedding (OPTIONAL)
        # Captures time-of-day patterns (hour 0-23)
        self.time_embedding = nn.Embedding(
            num_embeddings=num_time_slots,  # 24 hours
            embedding_dim=d_model           # 64 dimensions
        )
        
        # 4. Weekday Embedding (OPTIONAL)
        # Captures day-of-week patterns (Mon-Sun)
        self.weekday_embedding = nn.Embedding(
            num_embeddings=num_weekdays,    # 7 days
            embedding_dim=d_model           # 64 dimensions
        )
        
        # 5. Duration Embedding (OPTIONAL)
        # Captures stay duration patterns
        self.duration_embedding = nn.Embedding(
            num_embeddings=num_duration_bins,  # 20 bins
            embedding_dim=d_model              # 64 dimensions
        )
        
        # 6. Recency Embedding (OPTIONAL)
        # Captures how recent a visit was
        self.recency_embedding = nn.Embedding(
            num_embeddings=num_recency_bins,   # 20 bins
            embedding_dim=d_model              # 64 dimensions
        )
        
        # 7. Position-from-End Embedding (OPTIONAL)
        # Captures position relative to end of sequence
        self.pos_from_end_embedding = nn.Embedding(
            num_embeddings=max_seq_len,        # 100 positions
            embedding_dim=d_model              # 64 dimensions
        )

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPARISON TO ORIGINAL:                                                │
# │                                                                         │
# │  ORIGINAL (model.py, lines 68-74):                                     │
# │    with tf.variable_scope('embedding'):                                │
# │        embedding = tf.get_variable(                                    │
# │            'embedding',                                                │
# │            [vsize, hps.emb_dim],   # [50000, 128]                     │
# │            dtype=tf.float32,                                          │
# │            initializer=trunc_norm_init                                │
# │        )                                                               │
# │        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc..│
# │        emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for ..│
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Original: 1 embedding (word)                                     │
# │       Proposed: 7 embeddings (multi-modal)                            │
# │                                                                         │
# │    2. Original: 50,000 × 128 = 6.4M parameters                        │
# │       Proposed: ~55K total parameters                                  │
# │                                                                         │
# │    3. Original: Shared embedding for encoder/decoder                  │
# │       Proposed: Different embeddings per modality                     │
# │                                                                         │
# │    4. Original: No positional information                             │
# │       Proposed: Has position-from-end embedding                       │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Transformer Encoder

### Lines 115-148: Encoder Architecture

```python
        # ================================================================
        # TRANSFORMER ENCODER
        # ================================================================
        
        # Transformer encoder layer configuration
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,              # 64
            nhead=nhead,                  # 4 heads
            dim_feedforward=dim_feedforward,  # 128
            dropout=dropout,              # 0.15
            activation='gelu',            # GELU activation
            batch_first=True,             # [batch, seq, dim]
            norm_first=True               # Pre-LayerNorm
        )
        
        # Stack multiple encoder layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,        # 2 layers
            norm=nn.LayerNorm(d_model),   # Final LayerNorm
            enable_nested_tensor=False
        )
        
        # Input normalization
        self.input_norm = nn.LayerNorm(d_model)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPARISON TO ORIGINAL:                                                │
# │                                                                         │
# │  ORIGINAL ENCODER (model.py, lines 76-94):                             │
# │    cell_fw = tf.contrib.rnn.LSTMCell(                                  │
# │        hps.hidden_dim,  # 256                                          │
# │        initializer=rand_unif_init,                                     │
# │        state_is_tuple=True                                             │
# │    )                                                                    │
# │    cell_bw = tf.contrib.rnn.LSTMCell(                                  │
# │        hps.hidden_dim,                                                 │
# │        initializer=rand_unif_init,                                     │
# │        state_is_tuple=True                                             │
# │    )                                                                    │
# │    (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(│
# │        cell_fw, cell_bw, emb_enc_inputs,                              │
# │        dtype=tf.float32,                                               │
# │        sequence_length=self._enc_lens,                                 │
# │        swap_memory=True                                                │
# │    )                                                                    │
# │    encoder_outputs = tf.concat(axis=2, values=encoder_outputs)        │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │  Original BiLSTM:                                               │   │
# │  │    - Sequential processing (O(n) time complexity)              │   │
# │  │    - Hidden state flows through time                           │   │
# │  │    - Output: [batch, seq, 512] (256×2)                        │   │
# │  │                                                                 │   │
# │  │  Proposed Transformer:                                         │   │
# │  │    - Parallel processing (O(1) with attention O(n²))          │   │
# │  │    - Self-attention for global context                        │   │
# │  │    - Output: [batch, seq, 64]                                 │   │
# │  │    - Pre-LayerNorm (more stable training)                     │   │
# │  │    - GELU activation (smoother than ReLU)                     │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Pointer Attention

### Lines 149-176: Attention for Pointer

```python
        # ================================================================
        # POINTER ATTENTION
        # ================================================================
        
        # Attention for computing pointer probabilities
        # Uses scaled dot-product attention
        
        # Query projection: transforms decoder state for attention
        self.attention_query = nn.Linear(d_model, d_model)
        
        # Key projection: transforms encoder outputs for attention
        self.attention_key = nn.Linear(d_model, d_model)
        
        # Value projection: transforms encoder outputs for context
        self.attention_value = nn.Linear(d_model, d_model)
        
        # Scaling factor for attention scores
        self.scale = math.sqrt(d_model)  # √64 = 8

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPARISON TO ORIGINAL:                                                │
# │                                                                         │
# │  ORIGINAL ATTENTION (attention_decoder.py, lines 79-129):              │
# │                                                                         │
# │    # Bahdanau additive attention                                       │
# │    with tf.variable_scope("Attention"):                                │
# │        # Encoder features projection                                    │
# │        W_h = tf.get_variable("W_h", [1, 1, attn_size, attn_size])     │
# │        encoder_features = tf.nn.conv2d(                                │
# │            encoder_states, W_h, [1, 1, 1, 1], "SAME"                  │
# │        )  # [batch, seq, 1, attn_size]                                │
# │                                                                         │
# │        # Decoder state projection                                       │
# │        W_s = tf.get_variable("W_s", [attn_size, attn_size])           │
# │        decoder_features = tf.nn.xw_plus_b(                             │
# │            decoder_state, W_s, tf.zeros([attn_size])                  │
# │        )  # [batch, attn_size]                                         │
# │        decoder_features = tf.expand_dims(                              │
# │            tf.expand_dims(decoder_features, 1), 1                     │
# │        )                                                                │
# │                                                                         │
# │        # Attention score: v^T × tanh(encoder + decoder)               │
# │        v = tf.get_variable("v", [attn_size])                          │
# │        e = tf.reduce_sum(                                              │
# │            v * tf.tanh(encoder_features + decoder_features),          │
# │            [2, 3]                                                      │
# │        )  # [batch, seq]                                               │
# │        attn_dist = tf.nn.softmax(e)                                   │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │  Original (Bahdanau):                                           │   │
# │  │    score = v^T × tanh(W_h × h + W_s × s)                       │   │
# │  │    - Additive attention                                        │   │
# │  │    - Uses tanh non-linearity                                   │   │
# │  │    - Parameters: W_h, W_s, v                                   │   │
# │  │                                                                 │   │
# │  │  Proposed (Scaled Dot-Product):                                │   │
# │  │    score = (Q × K^T) / √d_k                                    │   │
# │  │    - Multiplicative attention                                  │   │
# │  │    - No non-linearity before softmax                          │   │
# │  │    - Parameters: W_Q, W_K, W_V                                 │   │
# │  │    - More efficient (matrix multiplication)                   │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Gate Mechanism

### Lines 177-195: Pointer-Generator Gate

```python
        # ================================================================
        # GATE MECHANISM (Pointer vs Generation)
        # ================================================================
        
        # Gate MLP: decides whether to copy (pointer) or generate
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),  # Concat context + query
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)             # Output: single gate value
        )
        
        # Generation head: predicts next location from any location
        self.generation_head = nn.Linear(d_model, num_locations)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPARISON TO ORIGINAL:                                                │
# │                                                                         │
# │  ORIGINAL P_GEN (attention_decoder.py, lines 163-168):                 │
# │                                                                         │
# │    with tf.variable_scope('calculate_pgen'):                           │
# │        p_gen = linear([                                                │
# │            context_vector,    # Attention context                      │
# │            state.c,          # LSTM cell state                         │
# │            state.h,          # LSTM hidden state                       │
# │            x                 # Input embedding                         │
# │        ], 1, True)           # Linear projection to scalar            │
# │        p_gen = tf.sigmoid(p_gen)  # Probability [0, 1]                │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  SEMANTICS INVERTED:                                           │   │
# │  │    Original: p_gen = 1 → generate from vocab                   │   │
# │  │              p_gen = 0 → copy from input                       │   │
# │  │                                                                 │   │
# │  │    Proposed: gate = 1 → copy from input (pointer)              │   │
# │  │              gate = 0 → generate from vocab                    │   │
# │  │                                                                 │   │
# │  │  INPUTS:                                                        │   │
# │  │    Original: context + cell + hidden + input (4 sources)       │   │
# │  │    Proposed: context + query (2 sources)                       │   │
# │  │                                                                 │   │
# │  │  ARCHITECTURE:                                                  │   │
# │  │    Original: Single linear layer → sigmoid                     │   │
# │  │    Proposed: MLP (2 layers) with GELU → sigmoid               │   │
# │  │                                                                 │   │
# │  │  NO DECODER STATE:                                             │   │
# │  │    Original uses LSTM state (c, h)                            │   │
# │  │    Proposed has no decoder, uses query vector                  │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Forward Pass

### Lines 196-254: Main Forward Function

```python
    def forward(
        self,
        x: torch.Tensor,           # Location IDs [batch, seq]
        x_dict: Dict[str, torch.Tensor]  # Additional features
    ) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Location sequence [batch, seq]
            x_dict: Dict containing optional features:
                - user_id: [batch, seq]
                - time_idx: [batch, seq]
                - weekday: [batch, seq]
                - duration: [batch, seq]
                - recency: [batch, seq]
        
        Returns:
            log_probs: Log probabilities over locations [batch, num_locations]
        """
        batch_size, seq_len = x.shape
        device = x.device
        
        # ============================================================
        # STEP 1: COMPUTE EMBEDDINGS
        # ============================================================
        
        # Start with location embeddings
        embeddings = self.location_embedding(x)  # [batch, seq, d_model]
        
        # Add optional embeddings (summed, not concatenated)
        if 'user_id' in x_dict:
            embeddings = embeddings + self.user_embedding(x_dict['user_id'])
        if 'time_idx' in x_dict:
            embeddings = embeddings + self.time_embedding(x_dict['time_idx'])
        if 'weekday' in x_dict:
            embeddings = embeddings + self.weekday_embedding(x_dict['weekday'])
        if 'duration' in x_dict:
            embeddings = embeddings + self.duration_embedding(x_dict['duration'])
        if 'recency' in x_dict:
            embeddings = embeddings + self.recency_embedding(x_dict['recency'])
        
        # Add position-from-end embedding
        pos_from_end = torch.arange(seq_len - 1, -1, -1, device=device)
        pos_from_end = pos_from_end.unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.pos_from_end_embedding(pos_from_end)
        
        # Normalize embeddings
        embeddings = self.input_norm(embeddings)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  EMBEDDING COMBINATION:                                                 │
# │                                                                         │
# │  Original: Single embedding lookup                                     │
# │    emb = embedding_lookup(word_ids)  # [batch, seq, 128]              │
# │                                                                         │
# │  Proposed: Sum of multiple embeddings                                  │
# │    emb = loc_emb + user_emb + time_emb + ...  # [batch, seq, 64]     │
# │                                                                         │
# │  Why sum instead of concatenate?                                       │
# │    - Keeps dimension constant (64 vs 64×7=448)                        │
# │    - Similar to how positional encoding works in Transformers        │
# │    - Each embedding adds information to the same semantic space       │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # STEP 2: CREATE PADDING MASK
        # ============================================================
        
        # Mask for padded positions (location_id = 0)
        padding_mask = (x == 0)  # [batch, seq] True where padded

        # ============================================================
        # STEP 3: ENCODE WITH TRANSFORMER
        # ============================================================
        
        # Pass through Transformer encoder
        encoder_output = self.transformer_encoder(
            embeddings,
            src_key_padding_mask=padding_mask
        )  # [batch, seq, d_model]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ENCODER COMPARISON:                                                    │
# │                                                                         │
# │  Original BiLSTM:                                                      │
# │    (outputs, (fw_state, bw_state)) = bidirectional_dynamic_rnn(       │
# │        cell_fw, cell_bw, emb_enc_inputs                               │
# │    )                                                                    │
# │    encoder_outputs = concat([outputs_fw, outputs_bw])  # [b, s, 512] │
# │                                                                         │
# │  Proposed Transformer:                                                 │
# │    encoder_output = transformer_encoder(                               │
# │        embeddings,                                                     │
# │        src_key_padding_mask=padding_mask                              │
# │    )  # [batch, seq, 64]                                              │
# │                                                                         │
# │  Key: Transformer uses self-attention to capture long-range deps      │
# │       BiLSTM uses sequential hidden state propagation                 │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # STEP 4: COMPUTE ATTENTION
        # ============================================================
        
        # Use last valid position as query
        # (or could use a learned query token)
        
        # Find last non-padded position for each sequence
        valid_positions = ~padding_mask  # [batch, seq]
        query_positions = valid_positions.sum(dim=1) - 1  # [batch]
        query = encoder_output[torch.arange(batch_size), query_positions]  # [batch, d_model]
        
        # Compute attention scores
        Q = self.attention_query(query)  # [batch, d_model]
        K = self.attention_key(encoder_output)  # [batch, seq, d_model]
        V = self.attention_value(encoder_output)  # [batch, seq, d_model]
        
        # Scaled dot-product attention
        attn_scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2))  # [batch, 1, seq]
        attn_scores = attn_scores.squeeze(1) / self.scale  # [batch, seq]
        
        # Mask padded positions
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        
        # Softmax to get attention weights (pointer distribution)
        attn_weights = F.softmax(attn_scores, dim=-1)  # [batch, seq]
        
        # Compute context vector
        context = torch.bmm(attn_weights.unsqueeze(1), V)  # [batch, 1, d_model]
        context = context.squeeze(1)  # [batch, d_model]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ATTENTION COMPUTATION:                                                 │
# │                                                                         │
# │  Original (Bahdanau):                                                  │
# │    # For each decoder step t:                                          │
# │    decoder_features = W_s × state  # Project decoder state            │
# │    scores = v^T × tanh(encoder_features + decoder_features)           │
# │    attn_dist = softmax(scores)                                        │
# │    context = sum(attn_dist × encoder_outputs)                         │
# │                                                                         │
# │  Proposed (Scaled Dot-Product):                                        │
# │    Q = W_Q × query                                                    │
# │    K = W_K × encoder_output                                           │
# │    V = W_V × encoder_output                                           │
# │    scores = (Q × K^T) / √d_k                                          │
# │    attn_weights = softmax(scores)                                     │
# │    context = attn_weights × V                                         │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # STEP 5: COMPUTE GATE VALUE
        # ============================================================
        
        # Concatenate context and query for gate input
        gate_input = torch.cat([context, query], dim=-1)  # [batch, d_model*2]
        gate = torch.sigmoid(self.gate_mlp(gate_input))  # [batch, 1]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  GATE SEMANTICS:                                                        │
# │                                                                         │
# │  gate ≈ 1: Trust pointer (copy from input)                             │
# │  gate ≈ 0: Trust generation (predict any location)                     │
# │                                                                         │
# │  INVERTED from original p_gen!                                         │
# │  Original: p_gen ≈ 1 means generate, p_gen ≈ 0 means copy             │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # STEP 6: COMPUTE POINTER DISTRIBUTION
        # ============================================================
        
        # Scatter attention weights to location indices
        pointer_probs = torch.zeros(batch_size, self.num_locations, device=device)
        pointer_probs = pointer_probs.scatter_add(
            dim=1,
            index=x,  # Location IDs as indices
            src=attn_weights
        )

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  POINTER DISTRIBUTION:                                                  │
# │                                                                         │
# │  Example:                                                               │
# │    x = [101, 205, 150, 312, 150]  (location sequence)                 │
# │    attn_weights = [0.1, 0.2, 0.3, 0.15, 0.25]                         │
# │                                                                         │
# │    pointer_probs = zeros([500])  # All locations                      │
# │    pointer_probs[101] += 0.1                                          │
# │    pointer_probs[205] += 0.2                                          │
# │    pointer_probs[150] += 0.3 + 0.25 = 0.55  # Appears twice!         │
# │    pointer_probs[312] += 0.15                                         │
# │                                                                         │
# │  KEY: Duplicate locations in input accumulate probability             │
# │                                                                         │
# │  Original (model.py lines 165-183):                                   │
# │    attn_dists_projected = [copy_dist for _ in range(vsize)]          │
# │    # Uses tf.scatter_nd to project attention to extended vocab        │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # STEP 7: COMPUTE GENERATION DISTRIBUTION
        # ============================================================
        
        # Generate distribution from context
        gen_logits = self.generation_head(context)  # [batch, num_locations]
        gen_probs = F.softmax(gen_logits, dim=-1)  # [batch, num_locations]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  GENERATION DISTRIBUTION:                                               │
# │                                                                         │
# │  Original (model.py lines 124-144):                                    │
# │    # At each decoder step:                                             │
# │    output = linear([cell_output, attn_dist], output_size)             │
# │    vocab_dists.append(tf.nn.softmax(output))                          │
# │                                                                         │
# │  Proposed:                                                             │
# │    gen_logits = W × context                                           │
# │    gen_probs = softmax(gen_logits)                                    │
# │                                                                         │
# │  KEY DIFFERENCE: Original generates one word at a time (decoder)      │
# │                  Proposed generates single location (no decoder)       │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # STEP 8: COMBINE WITH GATE
        # ============================================================
        
        # Final distribution: gate * pointer + (1-gate) * generation
        final_probs = gate * pointer_probs + (1 - gate) * gen_probs
        
        # Return log probabilities for numerical stability
        return torch.log(final_probs + 1e-10)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FINAL DISTRIBUTION:                                                    │
# │                                                                         │
# │  Original (model.py lines 167-183):                                    │
# │    # Project attention to vocab+OOV size                               │
# │    attn_dists_projected = ...                                         │
# │    # For each decoder step:                                            │
# │    vocab_dist = p_gen × vocab_dist                                    │
# │    attn_dist = (1 - p_gen) × attn_dist                               │
# │    final_dist = vocab_dist + attn_dist  # Over vocab+OOV             │
# │                                                                         │
# │  Proposed:                                                             │
# │    final = gate × pointer + (1-gate) × gen                            │
# │                                                                         │
# │  SEMANTICS (INVERTED):                                                 │
# │    Original: p_gen→1 = generate, p_gen→0 = copy                       │
# │    Proposed: gate→1 = copy, gate→0 = generate                         │
# │                                                                         │
# │  Both compute weighted combination over same vocabulary space         │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Helper Methods

### Lines 256-280: Utility Functions

```python
    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(
        self,
        x: torch.Tensor,
        x_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Returns:
            attn_weights: [batch, seq] attention over input sequence
        """
        # Run forward pass and return attention weights
        # (Implementation would cache/return attn_weights from forward)
        pass

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  HELPER METHOD COMPARISON:                                              │
# │                                                                         │
# │  Original provides:                                                     │
# │    - run_train_step(): Execute training step                          │
# │    - run_eval_step(): Execute evaluation step                         │
# │    - run_encoder(): Run only encoder (for beam search)                │
# │    - run_decoder_one_step(): Single decoder step                      │
# │                                                                         │
# │  Proposed provides:                                                     │
# │    - count_parameters(): Count params                                 │
# │    - get_attention_weights(): For interpretability                    │
# │                                                                         │
# │  KEY DIFFERENCE:                                                        │
# │    Original needs step-by-step decoder control for beam search        │
# │    Proposed is single-step prediction, no beam search needed          │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Annotated Code

### Full File with Comments

```python
"""
Pointer Generator Transformer: Next Location Prediction Model
====================================================

This model combines:
1. Multi-modal embeddings (location, user, time, etc.)
2. Transformer encoder (self-attention)
3. Pointer mechanism (copy from input)
4. Generation mechanism (predict any location)
5. Learned gate (combine pointer and generation)

Adapted from the Pointer-Generator Network for text summarization.
Key differences:
- No decoder (single-step prediction)
- Multi-modal features (not just word embeddings)
- Transformer encoder (not BiLSTM)
- Inverted gate semantics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple


class PointerGeneratorTransformer(nn.Module):
    """Pointer Network for next location prediction."""
    
    def __init__(
        self,
        num_locations: int,
        num_users: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 128,
        dropout: float = 0.1,
        num_time_slots: int = 24,
        num_weekdays: int = 7,
        num_duration_bins: int = 20,
        num_recency_bins: int = 20,
        max_seq_len: int = 100,
        **kwargs
    ):
        super().__init__()
        
        self.num_locations = num_locations
        self.d_model = d_model
        self.num_users = num_users
        self.max_seq_len = max_seq_len
        
        # ---- EMBEDDINGS ----
        self.location_embedding = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_embedding = nn.Embedding(num_users, d_model)
        self.time_embedding = nn.Embedding(num_time_slots, d_model)
        self.weekday_embedding = nn.Embedding(num_weekdays, d_model)
        self.duration_embedding = nn.Embedding(num_duration_bins, d_model)
        self.recency_embedding = nn.Embedding(num_recency_bins, d_model)
        self.pos_from_end_embedding = nn.Embedding(max_seq_len, d_model)
        
        # ---- TRANSFORMER ENCODER ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.input_norm = nn.LayerNorm(d_model)
        
        # ---- POINTER ATTENTION ----
        self.attention_query = nn.Linear(d_model, d_model)
        self.attention_key = nn.Linear(d_model, d_model)
        self.attention_value = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)
        
        # ---- GATE AND GENERATION ----
        self.gate_mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        self.generation_head = nn.Linear(d_model, num_locations)
    
    def forward(self, x: torch.Tensor, x_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        batch_size, seq_len = x.shape
        device = x.device
        
        # Step 1: Embeddings
        embeddings = self.location_embedding(x)
        if 'user_id' in x_dict:
            embeddings = embeddings + self.user_embedding(x_dict['user_id'])
        if 'time_idx' in x_dict:
            embeddings = embeddings + self.time_embedding(x_dict['time_idx'])
        if 'weekday' in x_dict:
            embeddings = embeddings + self.weekday_embedding(x_dict['weekday'])
        if 'duration' in x_dict:
            embeddings = embeddings + self.duration_embedding(x_dict['duration'])
        if 'recency' in x_dict:
            embeddings = embeddings + self.recency_embedding(x_dict['recency'])
        
        pos_from_end = torch.arange(seq_len - 1, -1, -1, device=device)
        pos_from_end = pos_from_end.unsqueeze(0).expand(batch_size, -1)
        embeddings = embeddings + self.pos_from_end_embedding(pos_from_end)
        embeddings = self.input_norm(embeddings)
        
        # Step 2: Padding mask
        padding_mask = (x == 0)
        
        # Step 3: Transformer encoding
        encoder_output = self.transformer_encoder(embeddings, src_key_padding_mask=padding_mask)
        
        # Step 4: Attention
        valid_positions = ~padding_mask
        query_positions = valid_positions.sum(dim=1) - 1
        query = encoder_output[torch.arange(batch_size), query_positions]
        
        Q = self.attention_query(query)
        K = self.attention_key(encoder_output)
        V = self.attention_value(encoder_output)
        
        attn_scores = torch.bmm(Q.unsqueeze(1), K.transpose(1, 2)).squeeze(1) / self.scale
        attn_scores = attn_scores.masked_fill(padding_mask, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        
        context = torch.bmm(attn_weights.unsqueeze(1), V).squeeze(1)
        
        # Step 5: Gate
        gate_input = torch.cat([context, query], dim=-1)
        gate = torch.sigmoid(self.gate_mlp(gate_input))
        
        # Step 6: Pointer distribution
        pointer_probs = torch.zeros(batch_size, self.num_locations, device=device)
        pointer_probs = pointer_probs.scatter_add(dim=1, index=x, src=attn_weights)
        
        # Step 7: Generation distribution
        gen_probs = F.softmax(self.generation_head(context), dim=-1)
        
        # Step 8: Final combination
        final_probs = gate * pointer_probs + (1 - gate) * gen_probs
        return torch.log(final_probs + 1e-10)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
```

---

*Next: [12_CODE_WALKTHROUGH_ORIGINAL.md](12_CODE_WALKTHROUGH_ORIGINAL.md) - Line-by-line analysis of the original model*
