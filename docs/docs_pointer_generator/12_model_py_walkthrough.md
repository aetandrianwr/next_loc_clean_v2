# model.py Line-by-Line Walkthrough

## Table of Contents
1. [File Overview](#file-overview)
2. [Imports and Dependencies](#imports-and-dependencies)
3. [Linear Helper Function](#linear-helper-function)
4. [SummarizationModel Class](#summarizationmodel-class)
5. [Placeholders](#placeholders)
6. [Encoder](#encoder)
7. [Reduce States](#reduce-states)
8. [Decoder](#decoder)
9. [Output Projection](#output-projection)
10. [Loss Calculation](#loss-calculation)
11. [Training Operations](#training-operations)
12. [Runtime Methods](#runtime-methods)

---

## File Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         MODEL.PY OVERVIEW                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FILE: pointer-generator/model.py                                               │
│   LINES: ~480                                                                    │
│   PURPOSE: Main model definition for pointer-generator network                   │
│                                                                                   │
│   STRUCTURE:                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   Lines 1-30:     Imports and dependencies                                       │
│   Lines 31-50:    linear() helper function                                       │
│   Lines 51-480:   SummarizationModel class                                       │
│     Lines 51-80:    __init__()                                                   │
│     Lines 81-130:   _add_placeholders()                                          │
│     Lines 131-180:  _add_encoder()                                               │
│     Lines 181-220:  _reduce_states()                                             │
│     Lines 221-280:  _add_decoder()                                               │
│     Lines 281-350:  _add_output_projection() and final distribution              │
│     Lines 351-400:  Loss calculation                                             │
│     Lines 401-440:  _add_train_op()                                              │
│     Lines 441-480:  Runtime methods (run_encoder, decode_onestep, etc.)         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Imports and Dependencies

```python
# model.py: Lines 1-30

"""Seq2seq attention model for text summarization."""

import time
import tensorflow as tf

# Custom attention decoder
from attention_decoder import attention_decoder

# TensorFlow operations
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple

# Flags for hyperparameters
FLAGS = tf.app.flags.FLAGS
```

**Explanation:**

| Import | Purpose |
|--------|---------|
| `time` | For timing operations |
| `tensorflow` | Main deep learning framework |
| `attention_decoder` | Custom decoder with attention (from attention_decoder.py) |
| `LSTMCell` | LSTM cell for encoder/decoder |
| `LSTMStateTuple` | Named tuple for LSTM state (c, h) |
| `FLAGS` | Access to command-line hyperparameters |

---

## Linear Helper Function

```python
# model.py: Lines 31-50

def linear(args, output_size, bias, bias_start=0.0, scope=None):
    """
    Linear transformation: output = W × args + b
    
    Args:
        args: Tensor or list of tensors to transform
        output_size: Output dimension
        bias: Whether to add bias
        bias_start: Initial value for bias
        scope: Variable scope name
    
    Returns:
        Tensor of shape [batch, output_size]
    """
    if args is None or (isinstance(args, (list, tuple)) and not args):
        raise ValueError("args must be specified")
    
    if not isinstance(args, (list, tuple)):
        args = [args]
    
    # Get total input size by summing dimensions
    total_arg_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        if len(shape) != 2:
            raise ValueError("Linear expects 2D arguments: %s" % shapes)
        total_arg_size += shape[1]
    
    with tf.variable_scope(scope or "Linear"):
        # Create weight matrix
        matrix = tf.get_variable(
            "Matrix",
            [total_arg_size, output_size]
        )
        
        # Concatenate inputs if multiple
        if len(args) == 1:
            result = tf.matmul(args[0], matrix)
        else:
            result = tf.matmul(tf.concat(args, 1), matrix)
        
        # Add bias if requested
        if bias:
            bias_var = tf.get_variable(
                "Bias",
                [output_size],
                initializer=tf.constant_initializer(bias_start)
            )
            result = result + bias_var
        
        return result
```

**Usage:** This function is used throughout the model to project tensors to different dimensions.

---

## SummarizationModel Class

### Initialization

```python
# model.py: Lines 51-80

class SummarizationModel(object):
    """
    Pointer-generator network for text summarization.
    """
    
    def __init__(self, hps, vocab):
        """
        Initialize the model.
        
        Args:
            hps: Hyperparameters namedtuple
            vocab: Vocabulary object
        """
        self._hps = hps
        self._vocab = vocab
    
    def build_graph(self):
        """
        Build the TensorFlow computation graph.
        
        Called once to construct all operations.
        """
        tf.logging.info("Building graph...")
        t0 = time.time()
        
        # Step 1: Add placeholders (input nodes)
        self._add_placeholders()
        
        # Step 2: Build encoder
        with tf.variable_scope('seq2seq'):
            self._add_encoder()
            self._reduce_states()
            self._add_decoder()
        
        # Step 3: Add loss and training op
        if self._hps.mode in ['train', 'eval']:
            self._add_seq2seq_loss()
            self._add_train_op()
        
        t1 = time.time()
        tf.logging.info("Time to build graph: %.2f seconds" % (t1 - t0))
```

---

## Placeholders

```python
# model.py: Lines 81-130

def _add_placeholders(self):
    """
    Add TensorFlow placeholder nodes for inputs.
    
    Placeholders are "empty" nodes that receive data at runtime
    via feed_dict.
    """
    hps = self._hps
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENCODER INPUTS
    # ═══════════════════════════════════════════════════════════════════════
    
    # Encoder input token IDs (OOV → UNK)
    self._enc_batch = tf.placeholder(
        tf.int32, 
        [hps.batch_size, None],  # [batch, enc_len]
        name='enc_batch'
    )
    
    # Actual encoder sequence lengths (before padding)
    self._enc_lens = tf.placeholder(
        tf.int32,
        [hps.batch_size],  # [batch]
        name='enc_lens'
    )
    
    # Encoder padding mask (1 for real tokens, 0 for padding)
    self._enc_padding_mask = tf.placeholder(
        tf.float32,
        [hps.batch_size, None],  # [batch, enc_len]
        name='enc_padding_mask'
    )
    
    # For pointer-generator: extended vocab encoding
    if FLAGS.pointer_gen:
        # Encoder input with extended vocab IDs for OOV words
        self._enc_batch_extend_vocab = tf.placeholder(
            tf.int32,
            [hps.batch_size, None],  # [batch, enc_len]
            name='enc_batch_extend_vocab'
        )
        
        # Maximum number of OOVs in any article in the batch
        self._max_art_oovs = tf.placeholder(
            tf.int32, 
            [],  # scalar
            name='max_art_oovs'
        )
    
    # ═══════════════════════════════════════════════════════════════════════
    # DECODER INPUTS (only for train/eval modes)
    # ═══════════════════════════════════════════════════════════════════════
    
    if hps.mode in ['train', 'eval']:
        # Decoder input token IDs [START, word1, word2, ...]
        self._dec_batch = tf.placeholder(
            tf.int32,
            [hps.batch_size, hps.max_dec_steps],  # [batch, dec_len]
            name='dec_batch'
        )
        
        # Target token IDs [word1, word2, ..., STOP]
        self._target_batch = tf.placeholder(
            tf.int32,
            [hps.batch_size, hps.max_dec_steps],  # [batch, dec_len]
            name='target_batch'
        )
        
        # Decoder padding mask
        self._dec_padding_mask = tf.placeholder(
            tf.float32,
            [hps.batch_size, hps.max_dec_steps],  # [batch, dec_len]
            name='dec_padding_mask'
        )
```

**Placeholder Summary:**

| Placeholder | Shape | Description |
|-------------|-------|-------------|
| `enc_batch` | [B, enc_len] | Encoder input IDs (UNK for OOV) |
| `enc_lens` | [B] | Actual encoder lengths |
| `enc_padding_mask` | [B, enc_len] | 1=real, 0=padding |
| `enc_batch_extend_vocab` | [B, enc_len] | IDs with extended vocab |
| `max_art_oovs` | scalar | Max OOVs in batch |
| `dec_batch` | [B, dec_len] | Decoder input IDs |
| `target_batch` | [B, dec_len] | Target IDs |
| `dec_padding_mask` | [B, dec_len] | 1=real, 0=padding |

---

## Encoder

```python
# model.py: Lines 131-180

def _add_encoder(self):
    """
    Build the bidirectional LSTM encoder.
    
    Processes the source sequence and produces:
    - encoder_outputs: Hidden states at each position
    - fw_st, bw_st: Final forward/backward states
    """
    hps = self._hps
    
    # ═══════════════════════════════════════════════════════════════════════
    # WORD EMBEDDINGS
    # ═══════════════════════════════════════════════════════════════════════
    
    with tf.variable_scope('embedding'):
        # Create embedding matrix
        embedding = tf.get_variable(
            'embedding',
            [self._vocab.size(), hps.emb_dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        # Look up embeddings for encoder input
        # Shape: [batch, enc_len, emb_dim]
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch)
        
        # Also look up decoder embeddings (for training)
        if hps.mode in ['train', 'eval']:
            emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch)
    
    # ═══════════════════════════════════════════════════════════════════════
    # BIDIRECTIONAL LSTM
    # ═══════════════════════════════════════════════════════════════════════
    
    with tf.variable_scope('encoder'):
        # Forward LSTM cell
        cell_fw = LSTMCell(
            hps.hidden_dim,
            state_is_tuple=True,
            initializer=tf.random_uniform_initializer(
                -hps.rand_unif_init_mag,
                hps.rand_unif_init_mag
            )
        )
        
        # Backward LSTM cell
        cell_bw = LSTMCell(
            hps.hidden_dim,
            state_is_tuple=True,
            initializer=tf.random_uniform_initializer(
                -hps.rand_unif_init_mag,
                hps.rand_unif_init_mag
            )
        )
        
        # Run bidirectional LSTM
        # outputs: (fw_outputs, bw_outputs)
        # states: (fw_state, bw_state)
        (encoder_outputs, (fw_st, bw_st)) = tf.nn.bidirectional_dynamic_rnn(
            cell_fw,
            cell_bw,
            emb_enc_inputs,
            dtype=tf.float32,
            sequence_length=self._enc_lens,
            swap_memory=True
        )
        
        # Concatenate forward and backward outputs
        # Shape: [batch, enc_len, 2*hidden_dim]
        encoder_outputs = tf.concat(encoder_outputs, 2)
    
    # Store for later use
    self._encoder_outputs = encoder_outputs
    self._fw_st = fw_st
    self._bw_st = bw_st
    self._emb_dec_inputs = emb_dec_inputs
```

**Encoder Output Shapes:**

```
Input:
  enc_batch:        [16, 400]           # Token IDs
  
After embedding:
  emb_enc_inputs:   [16, 400, 128]      # [batch, enc_len, emb_dim]
  
After bidirectional LSTM:
  encoder_outputs:  [16, 400, 512]      # [batch, enc_len, 2*hidden_dim]
  fw_st.h:          [16, 256]           # Forward final hidden
  bw_st.h:          [16, 256]           # Backward final hidden
```

---

## Reduce States

```python
# model.py: Lines 181-220

def _reduce_states(self):
    """
    Reduce bidirectional encoder states to single decoder initial state.
    
    The encoder produces 2 states (forward and backward), each with
    (cell, hidden). The decoder needs a single state.
    
    We concatenate and project:
    [fw_c, bw_c] → dec_c  (via linear layer)
    [fw_h, bw_h] → dec_h  (via linear layer)
    """
    hps = self._hps
    
    with tf.variable_scope('reduce_final_st'):
        # ═══════════════════════════════════════════════════════════════════
        # PROJECT CELL STATE
        # ═══════════════════════════════════════════════════════════════════
        
        # Weight matrix for cell state: [2*hidden_dim, hidden_dim]
        w_reduce_c = tf.get_variable(
            'w_reduce_c',
            [hps.hidden_dim * 2, hps.hidden_dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        # Bias for cell state
        bias_reduce_c = tf.get_variable(
            'bias_reduce_c',
            [hps.hidden_dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        # Concatenate forward and backward cell states
        # [batch, hidden_dim] concat [batch, hidden_dim] = [batch, 2*hidden_dim]
        old_c = tf.concat([self._fw_st.c, self._bw_st.c], axis=1)
        
        # Project to decoder dimension
        # [batch, 2*hidden_dim] @ [2*hidden_dim, hidden_dim] = [batch, hidden_dim]
        new_c = tf.nn.relu(tf.matmul(old_c, w_reduce_c) + bias_reduce_c)
        
        # ═══════════════════════════════════════════════════════════════════
        # PROJECT HIDDEN STATE
        # ═══════════════════════════════════════════════════════════════════
        
        # Same process for hidden state
        w_reduce_h = tf.get_variable(
            'w_reduce_h',
            [hps.hidden_dim * 2, hps.hidden_dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        bias_reduce_h = tf.get_variable(
            'bias_reduce_h',
            [hps.hidden_dim],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        old_h = tf.concat([self._fw_st.h, self._bw_st.h], axis=1)
        new_h = tf.nn.relu(tf.matmul(old_h, w_reduce_h) + bias_reduce_h)
        
        # Create LSTM state tuple for decoder
        self._dec_in_state = LSTMStateTuple(new_c, new_h)
```

**Reduce States Visualization:**

```
Forward:  fw_st.c [16, 256]  fw_st.h [16, 256]
                    ↓                   ↓
                   concat              concat
                    ↓                   ↓
         [fw_c, bw_c] [16, 512]  [fw_h, bw_h] [16, 512]
                    ↓                   ↓
              W_reduce_c @ x      W_reduce_h @ x
                    ↓                   ↓
             ReLU(... + bias)    ReLU(... + bias)
                    ↓                   ↓
           new_c [16, 256]       new_h [16, 256]
                    ↓                   ↓
                    └─────────┬─────────┘
                              ↓
                 LSTMStateTuple(new_c, new_h)
```

---

## Decoder

```python
# model.py: Lines 221-280

def _add_decoder(self):
    """
    Build the attention decoder.
    
    Uses the custom attention_decoder function from attention_decoder.py.
    """
    hps = self._hps
    
    with tf.variable_scope('decoder'):
        # Create decoder LSTM cell
        cell = LSTMCell(
            hps.hidden_dim,
            state_is_tuple=True,
            initializer=tf.random_uniform_initializer(
                -hps.rand_unif_init_mag,
                hps.rand_unif_init_mag
            )
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # PREPARE COVERAGE (if enabled)
        # ═══════════════════════════════════════════════════════════════════
        
        if hps.mode in ['train', 'eval']:
            # For training: no previous coverage
            prev_coverage = None
            if hps.coverage:
                # Initialize coverage to zeros
                prev_coverage = tf.zeros([hps.batch_size, tf.shape(self._enc_batch)[1]])
        else:
            # For decoding: coverage passed as placeholder
            prev_coverage = self._prev_coverage  # Set in decode mode
        
        # ═══════════════════════════════════════════════════════════════════
        # RUN ATTENTION DECODER
        # ═══════════════════════════════════════════════════════════════════
        
        # Call attention_decoder (from attention_decoder.py)
        outputs, out_state, attn_dists, p_gens, coverage = attention_decoder(
            self._emb_dec_inputs,       # Decoder input embeddings
            self._dec_in_state,          # Initial decoder state
            self._encoder_outputs,       # Encoder outputs (for attention)
            self._enc_padding_mask,      # Mask for encoder padding
            cell,                        # Decoder LSTM cell
            initial_state_attention=     # Whether to attend at step 0
                (hps.mode == 'decode'),
            pointer_gen=hps.pointer_gen, # Use pointer-generator?
            use_coverage=hps.coverage,   # Use coverage?
            prev_coverage=prev_coverage  # Previous coverage (for decode)
        )
        
        # Store outputs
        self._decoder_outputs = outputs      # [max_dec_steps, batch, hidden_dim]
        self._dec_out_state = out_state      # Final decoder state
        self._attn_dists = attn_dists        # List of attention distributions
        self._p_gens = p_gens                # List of p_gen values
        self._coverage = coverage            # Final coverage vector
```

---

## Output Projection

```python
# model.py: Lines 281-350

def _add_output_projection(self):
    """
    Project decoder outputs to vocabulary distribution.
    
    decoder_outputs → vocab_scores → vocab_dist → final_dist
    """
    hps = self._hps
    
    with tf.variable_scope('output_projection'):
        # ═══════════════════════════════════════════════════════════════════
        # PROJECT TO VOCABULARY SIZE
        # ═══════════════════════════════════════════════════════════════════
        
        # Weight matrix: [hidden_dim, vocab_size]
        w = tf.get_variable(
            'w',
            [hps.hidden_dim, self._vocab.size()],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        # Bias: [vocab_size]
        b = tf.get_variable(
            'b',
            [self._vocab.size()],
            dtype=tf.float32,
            initializer=tf.truncated_normal_initializer(
                stddev=hps.trunc_norm_init_std
            )
        )
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE VOCAB DISTRIBUTION FOR EACH TIME STEP
        # ═══════════════════════════════════════════════════════════════════
        
        vocab_dists = []
        for i, output in enumerate(self._decoder_outputs):
            # output shape: [batch, hidden_dim]
            
            # Project to vocabulary
            vocab_scores = tf.nn.xw_plus_b(output, w, b)  # [batch, vocab_size]
            
            # Softmax to get distribution
            vocab_dist = tf.nn.softmax(vocab_scores)  # [batch, vocab_size]
            
            vocab_dists.append(vocab_dist)
        
        # ═══════════════════════════════════════════════════════════════════
        # COMPUTE FINAL DISTRIBUTION (with pointer-generator)
        # ═══════════════════════════════════════════════════════════════════
        
        if FLAGS.pointer_gen:
            final_dists = self._calc_final_dist(vocab_dists, self._attn_dists)
        else:
            final_dists = vocab_dists
        
        self._vocab_dists = vocab_dists
        self._final_dists = final_dists


def _calc_final_dist(self, vocab_dists, attn_dists):
    """
    Calculate final distribution by combining vocab and copy distributions.
    """
    with tf.variable_scope('final_distribution'):
        # Extended vocabulary size
        vocab_size = self._vocab.size()
        extended_vsize = vocab_size + self._max_art_oovs
        
        final_dists = []
        
        for p_gen, vocab_dist, attn_dist in zip(
            self._p_gens, vocab_dists, attn_dists
        ):
            # ═══════════════════════════════════════════════════════════════
            # WEIGHTED VOCAB DISTRIBUTION
            # ═══════════════════════════════════════════════════════════════
            
            # Scale vocab dist by p_gen
            vocab_dist = p_gen * vocab_dist  # [batch, vocab_size]
            
            # Extend with zeros for OOV slots
            extra_zeros = tf.zeros([self._hps.batch_size, self._max_art_oovs])
            vocab_dist_extended = tf.concat([vocab_dist, extra_zeros], axis=1)
            # Shape: [batch, extended_vsize]
            
            # ═══════════════════════════════════════════════════════════════
            # WEIGHTED COPY DISTRIBUTION
            # ═══════════════════════════════════════════════════════════════
            
            # Scale attention by (1 - p_gen)
            attn_dist = (1 - p_gen) * attn_dist  # [batch, enc_len]
            
            # Project attention to extended vocabulary using scatter_nd
            batch_size = self._hps.batch_size
            attn_len = tf.shape(self._enc_batch_extend_vocab)[1]
            
            # Create indices for scatter: [batch, enc_len, 2]
            batch_nums = tf.expand_dims(tf.range(batch_size), 1)  # [batch, 1]
            batch_nums = tf.tile(batch_nums, [1, attn_len])        # [batch, enc_len]
            indices = tf.stack(
                [batch_nums, self._enc_batch_extend_vocab],
                axis=2
            )  # [batch, enc_len, 2]
            
            # Scatter attention weights to vocabulary positions
            shape = [batch_size, extended_vsize]
            attn_dist_projected = tf.scatter_nd(indices, attn_dist, shape)
            # Shape: [batch, extended_vsize]
            
            # ═══════════════════════════════════════════════════════════════
            # COMBINE
            # ═══════════════════════════════════════════════════════════════
            
            final_dist = vocab_dist_extended + attn_dist_projected
            final_dists.append(final_dist)
        
        return final_dists
```

---

## Loss Calculation

```python
# model.py: Lines 351-400

def _add_seq2seq_loss(self):
    """
    Add sequence-to-sequence loss.
    """
    with tf.variable_scope('loss'):
        
        if FLAGS.pointer_gen:
            # ═══════════════════════════════════════════════════════════════
            # LOSS WITH POINTER-GENERATOR
            # ═══════════════════════════════════════════════════════════════
            
            loss_per_step = []
            
            for dec_step, dist in enumerate(self._final_dists):
                # Get target for this step
                targets = self._target_batch[:, dec_step]  # [batch]
                
                # Create indices for gathering
                indices = tf.stack(
                    [tf.range(0, self._hps.batch_size), targets],
                    axis=1
                )  # [batch, 2]
                
                # Get probability of correct target
                gold_probs = tf.gather_nd(dist, indices)  # [batch]
                
                # Avoid log(0)
                losses = -tf.log(tf.maximum(gold_probs, 1e-10))  # [batch]
                
                loss_per_step.append(losses)
            
            # Mask and average
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
            
        else:
            # ═══════════════════════════════════════════════════════════════
            # STANDARD CROSS-ENTROPY LOSS
            # ═══════════════════════════════════════════════════════════════
            
            # Use built-in sparse softmax cross entropy
            loss_per_step = []
            
            for dec_step, output in enumerate(self._decoder_outputs):
                targets = self._target_batch[:, dec_step]
                
                # Compute vocab scores
                vocab_scores = tf.nn.xw_plus_b(output, self._w, self._b)
                
                # Cross entropy loss
                losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=vocab_scores,
                    labels=targets
                )
                
                loss_per_step.append(losses)
            
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
        
        # ═══════════════════════════════════════════════════════════════════
        # ADD COVERAGE LOSS
        # ═══════════════════════════════════════════════════════════════════
        
        if self._hps.coverage:
            self._coverage_loss = _coverage_loss(
                self._attn_dists,
                self._dec_padding_mask
            )
            self._loss += self._hps.cov_loss_wt * self._coverage_loss


def _mask_and_avg(values, padding_mask):
    """
    Apply padding mask and compute average loss.
    """
    # Stack values: [batch, dec_steps]
    dec_lens = tf.reduce_sum(padding_mask, axis=1)  # [batch]
    
    values_per_step = tf.stack(values, axis=1)  # [batch, dec_steps]
    
    # Apply mask
    values_per_ex = tf.reduce_sum(values_per_step * padding_mask, axis=1)
    
    # Average per example
    values_per_ex /= dec_lens
    
    # Average over batch
    return tf.reduce_mean(values_per_ex)


def _coverage_loss(attn_dists, padding_mask):
    """
    Calculate coverage loss.
    """
    coverage = tf.zeros_like(attn_dists[0])  # [batch, enc_len]
    
    covlosses = []
    for a in attn_dists:
        covloss = tf.reduce_sum(tf.minimum(a, coverage), axis=1)  # [batch]
        covlosses.append(covloss)
        coverage += a  # Accumulate coverage
    
    return _mask_and_avg(covlosses, padding_mask)
```

---

## Training Operations

```python
# model.py: Lines 401-440

def _add_train_op(self):
    """
    Add training operation (gradient computation and application).
    """
    # ═══════════════════════════════════════════════════════════════════════
    # GET TRAINABLE VARIABLES
    # ═══════════════════════════════════════════════════════════════════════
    
    tvars = tf.trainable_variables()
    
    # ═══════════════════════════════════════════════════════════════════════
    # COMPUTE GRADIENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    gradients = tf.gradients(
        self._loss,
        tvars,
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CLIP GRADIENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    grads, global_norm = tf.clip_by_global_norm(
        gradients,
        self._hps.max_grad_norm
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # CREATE OPTIMIZER
    # ═══════════════════════════════════════════════════════════════════════
    
    optimizer = tf.train.AdagradOptimizer(
        self._hps.lr,
        initial_accumulator_value=self._hps.adagrad_init_acc
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # APPLY GRADIENTS
    # ═══════════════════════════════════════════════════════════════════════
    
    self._train_op = optimizer.apply_gradients(
        zip(grads, tvars),
        global_step=self.global_step,
        name='train_step'
    )
    
    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARIES FOR TENSORBOARD
    # ═══════════════════════════════════════════════════════════════════════
    
    tf.summary.scalar('loss', self._loss)
    tf.summary.scalar('global_norm', global_norm)
    
    if self._hps.coverage:
        tf.summary.scalar('coverage_loss', self._coverage_loss)
    
    self._summaries = tf.summary.merge_all()
```

---

## Runtime Methods

```python
# model.py: Lines 441-480

def run_encoder(self, sess, batch):
    """
    Run encoder only (for decoding).
    
    Returns:
        enc_states: Encoder outputs [batch, enc_len, 2*hidden_dim]
        dec_in_state: Initial decoder state
    """
    feed_dict = {
        self._enc_batch: batch.enc_batch,
        self._enc_lens: batch.enc_lens,
        self._enc_padding_mask: batch.enc_padding_mask,
    }
    
    if FLAGS.pointer_gen:
        feed_dict[self._enc_batch_extend_vocab] = batch.enc_batch_extend_vocab
        feed_dict[self._max_art_oovs] = batch.max_art_oovs
    
    (enc_states, dec_in_state) = sess.run(
        [self._encoder_outputs, self._dec_in_state],
        feed_dict
    )
    
    # Convert state to numpy for easier manipulation
    dec_in_state = LSTMStateTuple(dec_in_state.c, dec_in_state.h)
    
    return enc_states, dec_in_state


def decode_onestep(self, sess, batch, latest_tokens, enc_states,
                   dec_init_states, prev_coverage):
    """
    Run one step of decoding (for beam search).
    
    Args:
        sess: TensorFlow session
        batch: Batch object
        latest_tokens: Last generated token for each beam [beam_size]
        enc_states: Encoder outputs [beam_size, enc_len, 2*hidden]
        dec_init_states: Decoder states for each beam
        prev_coverage: Previous coverage vectors
    
    Returns:
        topk_ids: Top-k token IDs for each beam [beam_size, 2*beam_size]
        topk_log_probs: Log probs for top-k tokens
        new_states: Updated decoder states
        attn_dists: Attention distributions
        p_gens: Generation probabilities
        new_coverage: Updated coverage vectors
    """
    beam_size = len(dec_init_states)
    
    # Build feed dict
    feed_dict = {
        self._enc_states: enc_states,
        self._enc_padding_mask: batch.enc_padding_mask,
        self._dec_batch: np.array([[t] for t in latest_tokens]),
        # ... more fields
    }
    
    # Run decoder one step
    results = sess.run({
        'ids': self._topk_ids,
        'probs': self._topk_log_probs,
        'states': self._dec_out_state,
        'attn_dists': self._attn_dists,
        'p_gens': self._p_gens,
        'coverage': self._coverage,
    }, feed_dict)
    
    return (results['ids'], results['probs'], results['states'],
            results['attn_dists'], results['p_gens'], results['coverage'])


def run_train_step(self, sess, batch):
    """
    Run one training step.
    
    Returns dictionary with loss and summaries.
    """
    feed_dict = self._make_feed_dict(batch)
    
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    
    if self._hps.coverage:
        to_return['coverage_loss'] = self._coverage_loss
    
    return sess.run(to_return, feed_dict)


def run_eval_step(self, sess, batch):
    """
    Run one evaluation step (no training).
    """
    feed_dict = self._make_feed_dict(batch)
    
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'global_step': self.global_step,
    }
    
    return sess.run(to_return, feed_dict)
```

---

## Summary

**model.py** is the core of the pointer-generator implementation:

| Component | Lines | Purpose |
|-----------|-------|---------|
| Placeholders | 81-130 | Define input tensors |
| Encoder | 131-180 | Bidirectional LSTM |
| Reduce States | 181-220 | Compress encoder states |
| Decoder | 221-280 | Attention decoder |
| Output Projection | 281-350 | Vocab + copy distribution |
| Loss | 351-400 | NLL + coverage loss |
| Train Op | 401-440 | Gradient computation |
| Runtime | 441-480 | Session run methods |

---

*Next: [13_attention_decoder_walkthrough.md](13_attention_decoder_walkthrough.md) - Line-by-Line attention_decoder.py Analysis*
