# Code Walkthrough: Original Model (Pointer-Generator)

## Table of Contents
1. [File Overview](#file-overview)
2. [Model Class Structure](#model-class-structure)
3. [Embedding and Encoder](#embedding-and-encoder)
4. [State Reduction](#state-reduction)
5. [Attention Decoder](#attention-decoder)
6. [Pointer-Generator Mechanism](#pointer-generator-mechanism)
7. [Loss Computation](#loss-computation)
8. [Training Operations](#training-operations)
9. [Complete Annotated Code](#complete-annotated-code)

---

## File Overview

```
Main Files:
├── model.py             (~481 lines) - Main model definition
├── attention_decoder.py (~229 lines) - Attention mechanism
├── batcher.py           (~220 lines) - Data loading
├── data.py              (~120 lines) - Data utilities
├── run_summarization.py (~230 lines) - Training/evaluation
├── decode.py            (~200 lines) - Beam search decoding
└── beam_search.py       (~200 lines) - Beam search algorithm

Primary Focus: model.py and attention_decoder.py
```

---

## Model Class Structure

### model.py Lines 1-45: Imports and Class Definition

```python
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
# Licensed under the Apache License, Version 2.0

"""This file contains code to build and run the tensorflow graph 
for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from attention_decoder import attention_decoder
from tensorflow.contrib.tensorboard.plugins import projector

FLAGS = tf.app.flags.FLAGS


class SummarizationModel(object):
    """A class to represent a sequence-to-sequence model for text summarization."""
    
    def __init__(self, hps, vocab):
        """Create the model.
        
        Args:
            hps: Hyperparameters object (NamedTuple)
            vocab: Vocabulary object
        """
        self._hps = hps
        self._vocab = vocab

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  COMPARISON TO PROPOSED:                                                │
# │                                                                         │
# │  Proposed (pgt.py):                                            │
# │    class PointerGeneratorTransformer(nn.Module):                                 │
# │        def __init__(self, num_locations, num_users, d_model=64, ...): │
# │            super().__init__()                                          │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Original: Inherits from object (plain Python class)             │
# │       Proposed: Inherits from nn.Module (PyTorch)                     │
# │                                                                         │
# │    2. Original: Uses hps NamedTuple for config                        │
# │       Proposed: Individual parameters with defaults                    │
# │                                                                         │
# │    3. Original: Needs vocab object (word ↔ id mapping)                │
# │       Proposed: Just num_locations (simple integer IDs)               │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

### model.py Lines 47-65: Placeholder Setup

```python
    def _add_placeholders(self):
        """Add placeholders to the graph. These are entry points for any input data."""
        hps = self._hps
        
        # Encoder inputs
        self._enc_batch = tf.placeholder(
            tf.int32, [hps.batch_size, None], name='enc_batch'
        )
        self._enc_lens = tf.placeholder(
            tf.int32, [hps.batch_size], name='enc_lens'
        )
        self._enc_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, None], name='enc_padding_mask'
        )
        
        if FLAGS.pointer_gen:
            self._enc_batch_extend_vocab = tf.placeholder(
                tf.int32, [hps.batch_size, None], name='enc_batch_extend_vocab'
            )
            self._max_art_oovs = tf.placeholder(
                tf.int32, [], name='max_art_oovs'
            )
        
        # Decoder inputs
        self._dec_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='dec_batch'
        )
        self._target_batch = tf.placeholder(
            tf.int32, [hps.batch_size, hps.max_dec_steps], name='target_batch'
        )
        self._dec_padding_mask = tf.placeholder(
            tf.float32, [hps.batch_size, hps.max_dec_steps], name='dec_padding_mask'
        )

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  PLACEHOLDER COMPARISON:                                                │
# │                                                                         │
# │  TensorFlow 1.x uses placeholders for input data:                      │
# │    - Define graph structure first                                      │
# │    - Feed data through feed_dict at runtime                           │
# │                                                                         │
# │  PyTorch (Proposed) passes data directly:                             │
# │    def forward(self, x, x_dict):  # Data passed as arguments          │
# │                                                                         │
# │  ORIGINAL PLACEHOLDERS:                                                 │
# │    - enc_batch: [batch, seq] encoder word IDs                         │
# │    - enc_lens: [batch] actual sequence lengths                        │
# │    - enc_padding_mask: [batch, seq] 1.0 for real, 0.0 for padding    │
# │    - enc_batch_extend_vocab: [batch, seq] with OOV IDs                │
# │    - max_art_oovs: scalar, max OOV words in batch                     │
# │    - dec_batch: [batch, max_dec] decoder input words                  │
# │    - target_batch: [batch, max_dec] target output words               │
# │    - dec_padding_mask: [batch, max_dec] decoder padding mask          │
# │                                                                         │
# │  PROPOSED INPUTS (forward args):                                        │
# │    - x: [batch, seq] location IDs                                     │
# │    - x_dict: additional features (user, time, etc.)                   │
# │    - No decoder inputs (single-step prediction)                       │
# │    - No OOV handling (closed vocabulary)                              │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Embedding and Encoder

### model.py Lines 67-94: Embedding and BiLSTM Encoder

```python
    def _add_seq2seq(self):
        """Add the whole sequence-to-sequence model to the graph."""
        hps = self._hps
        vsize = self._vocab.size()
        
        with tf.variable_scope('seq2seq'):
            # Some initializers
            self.rand_unif_init = tf.random_uniform_initializer(
                -hps.rand_unif_init_mag, hps.rand_unif_init_mag, seed=123
            )
            self.trunc_norm_init = tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
            
            # ============================================================
            # EMBEDDING LAYER
            # ============================================================
            with tf.variable_scope('embedding'):
                embedding = tf.get_variable(
                    'embedding', 
                    [vsize, hps.emb_dim],  # [50000, 128]
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )
                # Look up embeddings for encoder inputs
                emb_enc_inputs = tf.nn.embedding_lookup(
                    embedding, self._enc_batch
                )  # [batch, seq, 128]
                
                # Look up embeddings for decoder inputs (one per step)
                emb_dec_inputs = [
                    tf.nn.embedding_lookup(embedding, x) 
                    for x in tf.unstack(self._dec_batch, axis=1)
                ]  # list of [batch, 128]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  EMBEDDING COMPARISON:                                                  │
# │                                                                         │
# │  ORIGINAL:                                                              │
# │    embedding = tf.get_variable('embedding', [50000, 128])             │
# │    emb_enc = tf.nn.embedding_lookup(embedding, enc_batch)             │
# │    emb_dec = [tf.nn.embedding_lookup(embedding, x) for x in dec...]   │
# │                                                                         │
# │  PROPOSED:                                                              │
# │    self.location_embedding = nn.Embedding(500, 64, padding_idx=0)     │
# │    embeddings = self.location_embedding(x)                            │
# │    + self.user_embedding(x_dict['user_id'])                           │
# │    + self.time_embedding(x_dict['time_idx'])                          │
# │    + ... (5 more embeddings)                                          │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Original: Single embedding, shared encoder/decoder              │
# │       Proposed: 7 separate embeddings (multi-modal)                   │
# │                                                                         │
# │    2. Original: [50000, 128] = 6.4M parameters                        │
# │       Proposed: ~55K total parameters                                  │
# │                                                                         │
# │    3. Original: No padding_idx handling                               │
# │       Proposed: padding_idx=0 for proper masking                      │
# │                                                                         │
# │    4. Original: Needs decoder embeddings (autoregressive)             │
# │       Proposed: No decoder embeddings needed                          │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

            # ============================================================
            # BIDIRECTIONAL LSTM ENCODER
            # ============================================================
            with tf.variable_scope('encoder'):
                # Forward LSTM cell
                cell_fw = tf.contrib.rnn.LSTMCell(
                    hps.hidden_dim,  # 256
                    initializer=self.rand_unif_init,
                    state_is_tuple=True
                )
                # Backward LSTM cell
                cell_bw = tf.contrib.rnn.LSTMCell(
                    hps.hidden_dim,  # 256
                    initializer=self.rand_unif_init,
                    state_is_tuple=True
                )
                
                # Run bidirectional LSTM
                (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw, 
                    cell_bw, 
                    emb_enc_inputs,  # [batch, seq, 128]
                    dtype=tf.float32,
                    sequence_length=self._enc_lens,  # Actual lengths
                    swap_memory=True
                )
                
                # Concatenate forward and backward outputs
                encoder_outputs = tf.concat(
                    axis=2, 
                    values=encoder_outputs
                )  # [batch, seq, 512]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ENCODER COMPARISON:                                                    │
# │                                                                         │
# │  ORIGINAL (BiLSTM):                                                     │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  Input: [batch, seq, 128] (word embeddings)                    │   │
# │  │                                                                 │   │
# │  │  Forward LSTM:                                                  │   │
# │  │    h_fw_1 → h_fw_2 → h_fw_3 → ... → h_fw_T                    │   │
# │  │    [batch, 256] at each step                                   │   │
# │  │                                                                 │   │
# │  │  Backward LSTM:                                                 │   │
# │  │    h_bw_T ← h_bw_{T-1} ← ... ← h_bw_1                          │   │
# │  │    [batch, 256] at each step                                   │   │
# │  │                                                                 │   │
# │  │  Output: concat([h_fw, h_bw]) = [batch, seq, 512]             │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  PROPOSED (Transformer):                                                │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  Input: [batch, seq, 64] (summed embeddings)                   │   │
# │  │                                                                 │   │
# │  │  TransformerEncoderLayer × 2:                                  │   │
# │  │    - Multi-head self-attention (4 heads)                       │   │
# │  │    - Feed-forward network (64 → 128 → 64)                     │   │
# │  │    - LayerNorm + Dropout                                       │   │
# │  │                                                                 │   │
# │  │  Output: [batch, seq, 64]                                      │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    - Sequential (BiLSTM) vs Parallel (Transformer)                     │
# │    - 512-dim output vs 64-dim output                                   │
# │    - State propagation vs Self-attention                              │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## State Reduction

### model.py Lines 96-121: Reduce States

```python
            # ============================================================
            # REDUCE STATES (BiLSTM → Decoder initial state)
            # ============================================================
            
            # The encoder is bidirectional but decoder is unidirectional
            # Need to reduce 2 LSTM states to 1
            
            with tf.variable_scope('reduce_final_st'):
                # Weights for reducing forward/backward hidden states
                w_reduce_h = tf.get_variable(
                    'w_reduce_h', 
                    [hps.hidden_dim * 2, hps.hidden_dim],  # [512, 256]
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )
                # Bias for hidden state
                b_reduce_h = tf.get_variable(
                    'bias_reduce_h', 
                    [hps.hidden_dim],  # [256]
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )
                
                # Weights for reducing forward/backward cell states
                w_reduce_c = tf.get_variable(
                    'w_reduce_c', 
                    [hps.hidden_dim * 2, hps.hidden_dim],  # [512, 256]
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )
                b_reduce_c = tf.get_variable(
                    'bias_reduce_c', 
                    [hps.hidden_dim],
                    dtype=tf.float32,
                    initializer=self.trunc_norm_init
                )
                
                # Concatenate forward and backward states
                # fw_st.h, bw_st.h are [batch, 256]
                old_h = tf.concat(
                    axis=1, 
                    values=[fw_st.h, bw_st.h]
                )  # [batch, 512]
                old_c = tf.concat(
                    axis=1, 
                    values=[fw_st.c, bw_st.c]
                )  # [batch, 512]
                
                # Apply reduction with ReLU
                new_h = tf.nn.relu(
                    tf.matmul(old_h, w_reduce_h) + b_reduce_h
                )  # [batch, 256]
                new_c = tf.nn.relu(
                    tf.matmul(old_c, w_reduce_c) + b_reduce_c
                )  # [batch, 256]
                
                # Create new LSTM state tuple
                self._dec_in_state = tf.contrib.rnn.LSTMStateTuple(new_c, new_h)

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  STATE REDUCTION COMPARISON:                                            │
# │                                                                         │
# │  ORIGINAL: Needs state reduction                                        │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  BiLSTM produces 2 states:                                     │   │
# │  │    Forward: (c_fw, h_fw) each [batch, 256]                     │   │
# │  │    Backward: (c_bw, h_bw) each [batch, 256]                    │   │
# │  │                                                                 │   │
# │  │  Decoder needs 1 state:                                        │   │
# │  │    (c, h) each [batch, 256]                                    │   │
# │  │                                                                 │   │
# │  │  Reduction:                                                     │   │
# │  │    h = ReLU(W × [h_fw; h_bw] + b)                              │   │
# │  │    c = ReLU(W × [c_fw; c_bw] + b)                              │   │
# │  │                                                                 │   │
# │  │  Parameters: 512×256 + 256 = ~131K (×2 for c and h)           │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  PROPOSED: No state reduction needed                                    │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  Transformer has no decoder state to initialize                │   │
# │  │  Single-step prediction: just use encoder output              │   │
# │  │                                                                 │   │
# │  │  Query is derived from last encoder position:                 │   │
# │  │    query = encoder_output[batch_idx, last_valid_pos]          │   │
# │  │                                                                 │   │
# │  │  No state reduction parameters                                 │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Decoder

### attention_decoder.py Lines 79-168: Attention and P_gen

```python
def attention_decoder(
    decoder_inputs,      # List of [batch, emb_dim] tensors
    initial_state,       # LSTM state tuple (c, h)
    encoder_states,      # [batch, seq, 2*hidden_dim]
    enc_padding_mask,    # [batch, seq]
    cell,                # LSTM cell
    initial_state_attention=False,
    pointer_gen=True,
    use_coverage=False,
    prev_coverage=None
):
    """
    Attention decoder for sequence-to-sequence model.
    
    Returns:
        outputs: List of [batch, hidden_dim] tensors (decoder outputs)
        out_state: Final LSTM state
        attn_dists: List of [batch, seq] attention distributions
        p_gens: List of [batch, 1] generation probabilities
        coverage: Final coverage vector (if use_coverage)
    """
    
    with tf.variable_scope("attention_decoder"):
        batch_size = encoder_states.get_shape()[0].value
        attn_size = encoder_states.get_shape()[2].value  # 512
        
        # ============================================================
        # ATTENTION WEIGHTS (computed once, reused each step)
        # ============================================================
        
        # Reshape encoder states for convolution
        encoder_states = tf.expand_dims(encoder_states, axis=2)
        # [batch, seq, 1, attn_size]
        
        # Attention weight for encoder features
        W_h = tf.get_variable(
            "W_h", 
            [1, 1, attn_size, attn_size]  # [1, 1, 512, 512]
        )
        
        # Project encoder states (done once for efficiency)
        encoder_features = tf.nn.conv2d(
            encoder_states, W_h, [1, 1, 1, 1], "SAME"
        )  # [batch, seq, 1, attn_size]
        
        # Attention weights for decoder state
        W_s = tf.get_variable("W_s", [attn_size, attn_size])  # [512, 512]
        
        # Attention vector
        v = tf.get_variable(
            "v", [attn_size], 
            initializer=tf.contrib.layers.xavier_initializer()
        )  # [512]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ATTENTION PARAMETERS COMPARISON:                                       │
# │                                                                         │
# │  ORIGINAL (Bahdanau):                                                   │
# │    W_h: [1, 1, 512, 512] - encoder projection (conv2d)                │
# │    W_s: [512, 512] - decoder state projection                          │
# │    v: [512] - attention vector                                         │
# │    Total: ~780K parameters                                             │
# │                                                                         │
# │  PROPOSED (Scaled Dot-Product):                                         │
# │    W_Q: [64, 64] - query projection                                    │
# │    W_K: [64, 64] - key projection                                      │
# │    W_V: [64, 64] - value projection                                    │
# │    Total: ~12K parameters                                              │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    - Original uses additive attention (tanh)                          │
# │    - Proposed uses multiplicative attention (dot product)             │
# │    - Original projects to same dim, proposed can vary                 │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

        # ============================================================
        # ATTENTION FUNCTION (called at each decoder step)
        # ============================================================
        
        def attention(decoder_state, coverage=None):
            """
            Calculate attention distribution and context vector.
            
            Args:
                decoder_state: LSTM state tuple (c, h)
                coverage: Previous attention coverage (if using coverage)
            
            Returns:
                context_vector: [batch, attn_size]
                attn_dist: [batch, seq] attention weights
                coverage: Updated coverage (if using coverage)
            """
            with tf.variable_scope("Attention"):
                # Project decoder state
                decoder_features = linear(
                    decoder_state, attn_size, True
                )  # [batch, attn_size]
                decoder_features = tf.expand_dims(
                    tf.expand_dims(decoder_features, 1), 1
                )  # [batch, 1, 1, attn_size]
                
                # Compute attention scores
                # e_i = v^T * tanh(W_h * h_i + W_s * s_t + b)
                def masked_attention(e):
                    """Apply mask and softmax."""
                    attn_dist = tf.nn.softmax(e)  # [batch, seq]
                    attn_dist *= enc_padding_mask  # Zero out padding
                    # Renormalize
                    masked_sums = tf.reduce_sum(attn_dist, axis=1, keepdims=True)
                    return attn_dist / masked_sums
                
                if use_coverage and coverage is not None:
                    # Add coverage features
                    coverage_features = tf.nn.conv2d(
                        coverage, w_c, [1, 1, 1, 1], "SAME"
                    )
                    e = tf.reduce_sum(
                        v * tf.tanh(encoder_features + decoder_features + coverage_features),
                        [2, 3]
                    )  # [batch, seq]
                    attn_dist = masked_attention(e)
                    coverage += tf.reshape(attn_dist, [batch_size, -1, 1, 1])
                else:
                    # Standard attention (no coverage)
                    e = tf.reduce_sum(
                        v * tf.tanh(encoder_features + decoder_features),
                        [2, 3]
                    )  # [batch, seq]
                    attn_dist = masked_attention(e)
                    if use_coverage:
                        coverage = tf.expand_dims(
                            tf.expand_dims(attn_dist, 2), 2
                        )
                
                # Compute context vector
                context_vector = tf.reduce_sum(
                    tf.reshape(attn_dist, [batch_size, -1, 1, 1]) * encoder_states,
                    [1, 2]
                )  # [batch, attn_size]
                context_vector = tf.reshape(context_vector, [-1, attn_size])
                
            return context_vector, attn_dist, coverage

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  ATTENTION MECHANISM COMPARISON:                                        │
# │                                                                         │
# │  ORIGINAL (Bahdanau Additive):                                          │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  For each decoder step t:                                      │   │
# │  │                                                                 │   │
# │  │  1. Project decoder state:                                     │   │
# │  │     s_proj = W_s × s_t      # [batch, 512]                    │   │
# │  │                                                                 │   │
# │  │  2. Combine with encoder features (already projected):         │   │
# │  │     combined = tanh(enc_features + s_proj)                    │   │
# │  │                                                                 │   │
# │  │  3. Compute scores:                                            │   │
# │  │     e_i = v^T × combined_i  # For each encoder position       │   │
# │  │                                                                 │   │
# │  │  4. Apply softmax:                                             │   │
# │  │     α = softmax(e)          # [batch, seq]                    │   │
# │  │                                                                 │   │
# │  │  5. Compute context:                                           │   │
# │  │     c = Σ α_i × h_i         # [batch, 512]                    │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  PROPOSED (Scaled Dot-Product):                                         │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  Single computation (no decoder loop):                         │   │
# │  │                                                                 │   │
# │  │  1. Project query and keys:                                    │   │
# │  │     Q = W_Q × query         # [batch, 64]                     │   │
# │  │     K = W_K × encoder_out   # [batch, seq, 64]               │   │
# │  │     V = W_V × encoder_out   # [batch, seq, 64]               │   │
# │  │                                                                 │   │
# │  │  2. Compute scores (dot product):                             │   │
# │  │     scores = Q × K^T / √64  # [batch, seq]                   │   │
# │  │                                                                 │   │
# │  │  3. Apply softmax:                                             │   │
# │  │     α = softmax(scores)     # [batch, seq]                   │   │
# │  │                                                                 │   │
# │  │  4. Compute context:                                           │   │
# │  │     c = α × V               # [batch, 64]                     │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Pointer-Generator Mechanism

### attention_decoder.py Lines 163-180: P_gen Calculation

```python
        # ============================================================
        # DECODER LOOP
        # ============================================================
        
        outputs = []
        attn_dists = []
        p_gens = []
        state = initial_state
        coverage = prev_coverage
        context_vector = tf.zeros([batch_size, attn_size])
        
        for i, inp in enumerate(decoder_inputs):
            # inp is [batch, emb_dim]
            
            if i > 0:
                tf.get_variable_scope().reuse_variables()
            
            # Run attention
            context_vector, attn_dist, _ = attention(state, coverage)
            attn_dists.append(attn_dist)
            
            # Concatenate input embedding with context
            input_with_context = tf.concat([inp, context_vector], 1)
            # [batch, emb_dim + attn_size]
            
            # Run LSTM cell
            cell_output, state = cell(input_with_context, state)
            # cell_output: [batch, hidden_dim]
            # state: (c, h) each [batch, hidden_dim]
            
            # ============================================================
            # P_GEN CALCULATION
            # ============================================================
            
            if pointer_gen:
                with tf.variable_scope('calculate_pgen'):
                    p_gen = linear(
                        [context_vector, state.c, state.h, inp], 
                        1, 
                        True
                    )  # [batch, 1]
                    p_gen = tf.sigmoid(p_gen)
                    p_gens.append(p_gen)
            
            # Project output
            with tf.variable_scope("AttnOutputProjection"):
                output = linear(
                    [cell_output, context_vector], 
                    cell.output_size, 
                    True
                )  # [batch, hidden_dim]
            
            outputs.append(output)
        
        return outputs, state, attn_dists, p_gens, coverage

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  P_GEN vs GATE COMPARISON:                                              │
# │                                                                         │
# │  ORIGINAL P_GEN:                                                        │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  Inputs to p_gen MLP:                                          │   │
# │  │    - context_vector: [batch, 512] (attention context)         │   │
# │  │    - state.c: [batch, 256] (LSTM cell state)                  │   │
# │  │    - state.h: [batch, 256] (LSTM hidden state)                │   │
# │  │    - inp: [batch, 128] (current input embedding)              │   │
# │  │    Total: 1152-dim input                                       │   │
# │  │                                                                 │   │
# │  │  Computation:                                                   │   │
# │  │    p_gen = σ(W × [context; c; h; inp] + b)                    │   │
# │  │                                                                 │   │
# │  │  Semantics:                                                     │   │
# │  │    p_gen → 1: generate from vocabulary                         │   │
# │  │    p_gen → 0: copy from input                                  │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  PROPOSED GATE:                                                         │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  Inputs to gate MLP:                                           │   │
# │  │    - context: [batch, 64] (attention context)                 │   │
# │  │    - query: [batch, 64] (query vector)                        │   │
# │  │    Total: 128-dim input                                        │   │
# │  │                                                                 │   │
# │  │  Computation:                                                   │   │
# │  │    x = GELU(W1 × [context; query] + b1)                       │   │
# │  │    gate = σ(W2 × x + b2)                                       │   │
# │  │                                                                 │   │
# │  │  Semantics (INVERTED):                                         │   │
# │  │    gate → 1: copy from input (pointer)                         │   │
# │  │    gate → 0: generate from vocabulary                          │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Input dimensionality: 1152 vs 128                               │
# │    2. Architecture: Single linear vs 2-layer MLP                      │
# │    3. Semantics: INVERTED (p_gen=1→gen vs gate=1→copy)               │
# │    4. No LSTM state in proposed (no decoder)                          │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Loss Computation

### model.py Lines 249-285: Loss Calculation

```python
    def _add_seq2seq(self):
        # ... (previous code) ...
        
        # ============================================================
        # OUTPUT PROJECTION AND LOSS
        # ============================================================
        
        with tf.variable_scope('output_projection'):
            # Project decoder outputs to vocabulary
            w = tf.get_variable(
                'w', 
                [hps.hidden_dim, vsize],  # [256, 50000]
                dtype=tf.float32,
                initializer=self.trunc_norm_init
            )
            w_t = tf.transpose(w)  # For sharing with embedding
            v = tf.get_variable(
                'v', 
                [vsize],  # [50000]
                dtype=tf.float32,
                initializer=self.trunc_norm_init
            )
            
            # Compute vocabulary scores for each decoder step
            vocab_scores = []
            for i, output in enumerate(outputs):
                if i > 0:
                    tf.get_variable_scope().reuse_variables()
                vocab_scores.append(tf.nn.xw_plus_b(output, w, v))
            # vocab_scores: list of [batch, vocab_size]
            
            # Convert to distributions
            vocab_dists = [tf.nn.softmax(s) for s in vocab_scores]
            # vocab_dists: list of [batch, vocab_size]

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  OUTPUT PROJECTION COMPARISON:                                          │
# │                                                                         │
# │  ORIGINAL:                                                              │
# │    # For each decoder step:                                            │
# │    vocab_score = W × output + b  # [batch, 50000]                     │
# │    vocab_dist = softmax(vocab_score)                                  │
# │                                                                         │
# │    Parameters: 256 × 50000 + 50000 ≈ 12.8M                            │
# │                                                                         │
# │  PROPOSED:                                                              │
# │    # Single prediction:                                                │
# │    gen_logits = W × context  # [batch, 500]                           │
# │    gen_probs = softmax(gen_logits)                                    │
# │                                                                         │
# │    Parameters: 64 × 500 = 32K                                         │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘

            # ========================================================
            # FINAL DISTRIBUTION (Pointer-Generator)
            # ========================================================
            
            if FLAGS.pointer_gen:
                final_dists = self._calc_final_dist(
                    vocab_dists, self.attn_dists
                )
            else:
                final_dists = vocab_dists
            
            # ========================================================
            # LOSS COMPUTATION
            # ========================================================
            
            if hps.mode in ['train', 'eval']:
                with tf.variable_scope('loss'):
                    if FLAGS.pointer_gen:
                        # For pointer-generator: negative log likelihood
                        loss_per_step = []
                        batch_nums = tf.range(0, limit=hps.batch_size)
                        
                        for dec_step, dist in enumerate(final_dists):
                            # Get target word for this step
                            targets = self._target_batch[:, dec_step]
                            # Get indices for gathering
                            indices = tf.stack((batch_nums, targets), axis=1)
                            # Get probability of correct word
                            gold_probs = tf.gather_nd(dist, indices)
                            # Compute loss
                            losses = -tf.log(gold_probs + 1e-10)
                            loss_per_step.append(losses)
                        
                        # Average over non-padding positions
                        self._loss = _mask_and_avg(
                            loss_per_step, 
                            self._dec_padding_mask
                        )
                    else:
                        # For baseline: sequence loss
                        self._loss = tf.contrib.seq2seq.sequence_loss(
                            tf.stack(vocab_scores, axis=1),
                            self._target_batch,
                            self._dec_padding_mask
                        )

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  LOSS COMPUTATION COMPARISON:                                           │
# │                                                                         │
# │  ORIGINAL:                                                              │
# │    # For each decoder step t:                                          │
# │    prob_t = final_dist[t][target_t]  # Prob of correct word           │
# │    loss_t = -log(prob_t)                                               │
# │                                                                         │
# │    # Average over all steps (masked):                                  │
# │    loss = Σ loss_t × mask_t / Σ mask_t                                │
# │                                                                         │
# │  PROPOSED:                                                              │
# │    # Single prediction:                                                │
# │    loss = CrossEntropyLoss(logits, target)                            │
# │    # With label smoothing (ε=0.03)                                    │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Original: Per-step loss, then average                           │
# │       Proposed: Single loss (one prediction)                          │
# │                                                                         │
# │    2. Original: Manual NLL with masking                               │
# │       Proposed: PyTorch CrossEntropyLoss                              │
# │                                                                         │
# │    3. Original: No label smoothing                                    │
# │       Proposed: Label smoothing (0.03)                                │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

### model.py Lines 146-183: Final Distribution Calculation

```python
    def _calc_final_dist(self, vocab_dists, attn_dists):
        """
        Calculate final distribution by combining vocab and attention.
        
        Args:
            vocab_dists: List of [batch, vocab_size] vocab distributions
            attn_dists: List of [batch, seq_len] attention distributions
        
        Returns:
            final_dists: List of [batch, vocab_size + max_oovs] distributions
        """
        with tf.variable_scope('final_distribution'):
            # Get vocab_size and extended vocab size
            vocab_dists = [p_gen * dist for p_gen, dist in zip(self.p_gens, vocab_dists)]
            # Scale vocab dist by p_gen
            
            attn_dists = [(1 - p_gen) * dist for p_gen, dist in zip(self.p_gens, attn_dists)]
            # Scale attention dist by (1 - p_gen)
            
            # Extend vocab size for OOV words
            extended_vsize = self._vocab.size() + self._max_art_oovs
            
            # Project attention distribution to extended vocab
            batch_nums = tf.range(0, limit=hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)
            batch_nums = tf.tile(batch_nums, [1, tf.shape(attn_dists[0])[1]])
            
            indices = tf.stack([batch_nums, self._enc_batch_extend_vocab], axis=2)
            # indices shape: [batch, seq_len, 2]
            
            # For each decoder step, scatter attention to extended vocab
            final_dists = []
            for vocab_dist, attn_dist in zip(vocab_dists, attn_dists):
                # Create tensor of zeros for extended vocab
                extra_zeros = tf.zeros((hps.batch_size, self._max_art_oovs))
                # Concatenate vocab_dist with extra zeros
                vocab_dist_extended = tf.concat([vocab_dist, extra_zeros], axis=1)
                
                # Scatter attention to appropriate positions
                shape = [hps.batch_size, extended_vsize]
                attn_dist_projected = tf.scatter_nd(
                    indices, attn_dist, shape
                )  # [batch, extended_vsize]
                
                # Combine vocab and attention
                final_dist = vocab_dist_extended + attn_dist_projected
                final_dists.append(final_dist)
            
            return final_dists

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  FINAL DISTRIBUTION COMPARISON:                                         │
# │                                                                         │
# │  ORIGINAL:                                                              │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  final = p_gen × vocab_dist + (1-p_gen) × attn_dist_projected │   │
# │  │                                                                 │   │
# │  │  Where:                                                         │   │
# │  │    vocab_dist: [batch, 50000] over vocabulary                  │   │
# │  │    attn_dist_projected: [batch, 50000+max_oov] scattered      │   │
# │  │                                                                 │   │
# │  │  OOV Handling:                                                  │   │
# │  │    - Words not in vocab get special OOV IDs                   │   │
# │  │    - Attention can copy these OOV words                       │   │
# │  │    - Extended vocab size = 50000 + max_oov_in_batch           │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  PROPOSED:                                                              │
# │  ┌─────────────────────────────────────────────────────────────────┐   │
# │  │                                                                 │   │
# │  │  final = gate × pointer_probs + (1-gate) × gen_probs          │   │
# │  │                                                                 │   │
# │  │  Where:                                                         │   │
# │  │    pointer_probs: [batch, 500] scattered from attention       │   │
# │  │    gen_probs: [batch, 500] from generation head               │   │
# │  │                                                                 │   │
# │  │  No OOV Handling:                                              │   │
# │  │    - All locations are known (closed vocabulary)              │   │
# │  │    - No extended vocabulary needed                            │   │
# │  │    - Simpler scatter operation                                │   │
# │  │                                                                 │   │
# │  └─────────────────────────────────────────────────────────────────┘   │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Original handles OOV, proposed doesn't need to                  │
# │    2. Original: p_gen=1→vocab, proposed: gate=1→pointer              │
# │    3. Original: 50000+ output dim, proposed: 500 output dim          │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Training Operations

### model.py Lines 288-320: Training Setup

```python
    def _add_train_op(self):
        """Sets self._train_op, the op to run for training."""
        
        # Get loss to minimize
        loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
        
        # Get all trainable variables
        tvars = tf.trainable_variables()
        
        # Compute gradients
        gradients = tf.gradients(
            loss_to_minimize, 
            tvars,
            aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
        )
        
        # Clip gradients by global norm
        with tf.device("/gpu:0"):
            grads, global_norm = tf.clip_by_global_norm(
                gradients, 
                self._hps.max_grad_norm  # 2.0
            )
        
        # Log gradient norm
        tf.summary.scalar('global_norm', global_norm)
        
        # Create optimizer
        optimizer = tf.train.AdagradOptimizer(
            self._hps.lr,  # 0.15
            initial_accumulator_value=self._hps.adagrad_init_acc  # 0.1
        )
        
        # Apply gradients
        with tf.device("/gpu:0"):
            self._train_op = optimizer.apply_gradients(
                zip(grads, tvars),
                global_step=self.global_step,
                name='train_step'
            )

# ┌─────────────────────────────────────────────────────────────────────────┐
# │  TRAINING OPERATION COMPARISON:                                         │
# │                                                                         │
# │  ORIGINAL:                                                              │
# │    optimizer = Adagrad(lr=0.15, init_acc=0.1)                          │
# │    grads = clip_by_global_norm(gradients, 2.0)                         │
# │    train_op = optimizer.apply_gradients(grads, tvars)                  │
# │                                                                         │
# │  PROPOSED:                                                              │
# │    optimizer = AdamW(lr=6.5e-4, weight_decay=0.015)                    │
# │    clip_grad_norm_(model.parameters(), 0.8)                            │
# │    optimizer.step()                                                    │
# │                                                                         │
# │  KEY DIFFERENCES:                                                       │
# │    1. Optimizer: Adagrad vs AdamW                                      │
# │    2. Learning rate: 0.15 vs 6.5e-4                                   │
# │    3. Gradient clipping: 2.0 vs 0.8                                   │
# │    4. Weight decay: None vs 0.015                                     │
# │    5. LR scheduling: None (Adagrad adaptive) vs Warmup+Cosine        │
# │                                                                         │
# └─────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Annotated Code

### Summary of Key Files

```python
"""
ORIGINAL POINTER-GENERATOR NETWORK
==================================

File Structure:
├── model.py             - Main model (SummarizationModel class)
│   ├── __init__         - Initialize with hps and vocab
│   ├── _add_placeholders - Define TF placeholders for input
│   ├── _add_seq2seq     - Build encoder, decoder, attention
│   ├── _calc_final_dist - Combine vocab and pointer distributions
│   ├── _add_train_op    - Setup Adagrad optimizer
│   └── build_graph      - Call all setup methods
│
├── attention_decoder.py - Attention mechanism
│   ├── attention_decoder - Main function (decoder loop)
│   │   ├── attention     - Compute attention distribution
│   │   ├── p_gen        - Compute generation probability
│   │   └── output       - Compute decoder output
│   └── linear           - Helper for linear layers
│
├── batcher.py          - Data loading with queues
│   ├── Batcher         - Multi-threaded batch generator
│   ├── Example         - Single training example
│   └── Batch           - Batch of examples
│
└── run_summarization.py - Training/evaluation entry point
    ├── run_training    - Training loop
    ├── run_eval        - Evaluation loop
    └── main            - Command-line interface

Key Hyperparameters:
  - hidden_dim: 256
  - emb_dim: 128
  - vocab_size: 50,000
  - batch_size: 16
  - max_enc_steps: 400
  - max_dec_steps: 100
  - lr: 0.15 (Adagrad)
  - max_grad_norm: 2.0

Total Parameters: ~47M
"""
```

---

*Next: [13_MATHEMATICAL_FORMULATION.md](13_MATHEMATICAL_FORMULATION.md) - Complete mathematical formulations*
