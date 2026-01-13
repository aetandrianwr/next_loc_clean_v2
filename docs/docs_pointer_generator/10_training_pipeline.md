# Training Pipeline Deep Dive

## Table of Contents
1. [Training Overview](#training-overview)
2. [Hyperparameters](#hyperparameters)
3. [Model Construction](#model-construction)
4. [Training Loop](#training-loop)
5. [Optimization Details](#optimization-details)
6. [Gradient Clipping](#gradient-clipping)
7. [Checkpointing](#checkpointing)
8. [Tensorboard Logging](#tensorboard-logging)
9. [Evaluation Mode](#evaluation-mode)
10. [Training Tips](#training-tips)

---

## Training Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        TRAINING PIPELINE OVERVIEW                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│                          run_summarization.py                                     │
│                                │                                                  │
│                                ▼                                                  │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │  1. SETUP PHASE                                                         │    │
│   │     ─────────────                                                       │    │
│   │     • Parse command-line flags (hyperparameters)                       │    │
│   │     • Load vocabulary                                                   │    │
│   │     • Create Batcher (data pipeline)                                   │    │
│   │     • Build TensorFlow graph                                           │    │
│   │     • Initialize variables or restore from checkpoint                  │    │
│   └───────────────────────────────────┬────────────────────────────────────┘    │
│                                       │                                          │
│                                       ▼                                          │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │  2. TRAINING LOOP                                                       │    │
│   │     ─────────────                                                       │    │
│   │                                                                         │    │
│   │     while True:                                                         │    │
│   │         ┌─────────────────────────────────────────────────────────┐    │    │
│   │         │  a. Get batch from Batcher                               │    │    │
│   │         │     batch = batcher.next_batch()                         │    │    │
│   │         └─────────────────────────┬───────────────────────────────┘    │    │
│   │                                   │                                     │    │
│   │                                   ▼                                     │    │
│   │         ┌─────────────────────────────────────────────────────────┐    │    │
│   │         │  b. Run training step                                    │    │    │
│   │         │     - Forward pass (compute loss)                        │    │    │
│   │         │     - Backward pass (compute gradients)                  │    │    │
│   │         │     - Apply gradients (update weights)                   │    │    │
│   │         └─────────────────────────┬───────────────────────────────┘    │    │
│   │                                   │                                     │    │
│   │                                   ▼                                     │    │
│   │         ┌─────────────────────────────────────────────────────────┐    │    │
│   │         │  c. Log progress                                         │    │    │
│   │         │     - Print loss every N steps                           │    │    │
│   │         │     - Write Tensorboard summaries                        │    │    │
│   │         │     - Save checkpoint periodically                       │    │    │
│   │         └─────────────────────────────────────────────────────────┘    │    │
│   │                                                                         │    │
│   └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│                                                                                   │
│   FILES INVOLVED:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   run_summarization.py  │  Entry point, training loop                            │
│   model.py              │  Model definition, loss computation                    │
│   attention_decoder.py  │  Decoder with attention                                │
│   batcher.py            │  Data loading                                          │
│   data.py               │  Vocabulary and encoding                               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hyperparameters

### Command-Line Flags (run_summarization.py)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          HYPERPARAMETERS                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   MODEL ARCHITECTURE:                                                             │
│   ───────────────────                                                             │
│                                                                                   │
│   Flag              Default    Description                                        │
│   ────              ───────    ───────────                                        │
│   hidden_dim        256        LSTM hidden state dimension                        │
│   emb_dim           128        Word embedding dimension                           │
│   vocab_size        50000      Vocabulary size                                    │
│   max_enc_steps     400        Max encoder sequence length                        │
│   max_dec_steps     100        Max decoder sequence length                        │
│                                                                                   │
│   Note: Encoder is bidirectional, so encoder output is 2×hidden_dim = 512       │
│                                                                                   │
│                                                                                   │
│   POINTER-GENERATOR OPTIONS:                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   Flag              Default    Description                                        │
│   ────              ───────    ───────────                                        │
│   pointer_gen       True       Use pointer-generator mechanism                    │
│   coverage          False      Use coverage mechanism                             │
│   cov_loss_wt       1.0        Weight for coverage loss                          │
│                                                                                   │
│                                                                                   │
│   TRAINING:                                                                       │
│   ─────────                                                                       │
│                                                                                   │
│   Flag              Default    Description                                        │
│   ────              ───────    ───────────                                        │
│   lr                0.15       Learning rate (for Adagrad)                        │
│   batch_size        16         Batch size                                         │
│   max_grad_norm     2.0        Gradient clipping threshold                        │
│   rand_unif_init    0.02       Random uniform initialization magnitude            │
│   trunc_norm_init   1e-4       Truncated normal initialization std               │
│                                                                                   │
│                                                                                   │
│   DECODING:                                                                       │
│   ─────────                                                                       │
│                                                                                   │
│   Flag              Default    Description                                        │
│   ────              ───────    ───────────                                        │
│   beam_size         4          Beam search beam size                              │
│   min_dec_steps     35         Minimum decoded sequence length                    │
│                                                                                   │
│                                                                                   │
│   PATHS:                                                                          │
│   ──────                                                                          │
│                                                                                   │
│   Flag              Description                                                   │
│   ────              ───────────                                                   │
│   data_path         Path to tf.Example data files                                │
│   vocab_path        Path to vocabulary file                                       │
│   log_root          Root directory for logs and checkpoints                       │
│   exp_name          Name of experiment (subdirectory)                             │
│                                                                                   │
│                                                                                   │
│   MODES:                                                                          │
│   ──────                                                                          │
│                                                                                   │
│   --mode=train      Train the model                                              │
│   --mode=eval       Evaluate on validation set                                   │
│   --mode=decode     Run beam search decoding                                     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Hyperparameters Object

```python
# run_summarization.py: Create hyperparameters object (Lines 50-75)

hps = {
    'mode': FLAGS.mode,
    'lr': FLAGS.lr,
    'adagrad_init_acc': 0.1,
    'rand_unif_init_mag': FLAGS.rand_unif_init_mag,
    'trunc_norm_init_std': FLAGS.trunc_norm_init_std,
    'max_grad_norm': FLAGS.max_grad_norm,
    'hidden_dim': FLAGS.hidden_dim,
    'emb_dim': FLAGS.emb_dim,
    'batch_size': FLAGS.batch_size,
    'max_dec_steps': FLAGS.max_dec_steps,
    'max_enc_steps': FLAGS.max_enc_steps,
    'coverage': FLAGS.coverage,
    'cov_loss_wt': FLAGS.cov_loss_wt,
    'pointer_gen': FLAGS.pointer_gen,
}

# Convert to namedtuple for attribute access
hps = namedtuple("HParams", hps.keys())(**hps)
```

---

## Model Construction

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        MODEL CONSTRUCTION                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   GRAPH BUILDING (run_summarization.py):                                          │
│   ─────────────────────────────────────                                           │
│                                                                                   │
│   # Build the TensorFlow graph                                                    │
│   with tf.Graph().as_default():                                                  │
│                                                                                   │
│       # 1. Load vocabulary                                                        │
│       vocab = Vocab(vocab_path, vocab_size)                                      │
│                                                                                   │
│       # 2. Create data pipeline                                                   │
│       batcher = Batcher(data_path, vocab, hps, single_pass=False)               │
│                                                                                   │
│       # 3. Build model                                                            │
│       model = SummarizationModel(hps, vocab)                                     │
│       model.build_graph()                                                        │
│                                                                                   │
│       # 4. Create session and initialize                                          │
│       sv = tf.train.Supervisor(...)                                              │
│       sess = sv.prepare_or_wait_for_session(config)                             │
│                                                                                   │
│                                                                                   │
│   MODEL BUILD_GRAPH (model.py):                                                   │
│   ─────────────────────────────                                                   │
│                                                                                   │
│   def build_graph(self):                                                         │
│       """Builds the TensorFlow computation graph."""                             │
│                                                                                   │
│       # 1. Add placeholders (inputs)                                             │
│       self._add_placeholders()                                                   │
│                                                                                   │
│       # 2. Add encoder                                                            │
│       with tf.variable_scope('encoder'):                                         │
│           encoder_outputs, (fw_st, bw_st) = self._add_encoder()                 │
│                                                                                   │
│       # 3. Reduce encoder states for decoder init                                │
│       self._reduce_states(fw_st, bw_st)                                         │
│                                                                                   │
│       # 4. Add decoder with attention                                            │
│       with tf.variable_scope('decoder'):                                         │
│           decoder_outputs, ... = self._add_decoder(encoder_outputs)             │
│                                                                                   │
│       # 5. Add output projection and loss                                        │
│       with tf.variable_scope('output_projection'):                               │
│           self._add_output_projection(decoder_outputs)                          │
│                                                                                   │
│       # 6. Add training operations                                                │
│       if self._hps.mode == 'train':                                             │
│           self._add_train_op()                                                  │
│                                                                                   │
│                                                                                   │
│   PLACEHOLDERS:                                                                   │
│   ─────────────                                                                   │
│                                                                                   │
│   enc_batch          [batch, max_enc_steps]     Encoder input IDs               │
│   enc_lens           [batch]                    Actual encoder lengths          │
│   enc_padding_mask   [batch, max_enc_steps]     1=real, 0=padding              │
│   enc_batch_extend_vocab [batch, max_enc_steps] Extended vocab IDs             │
│   max_art_oovs       scalar                     Max OOVs in batch              │
│   dec_batch          [batch, max_dec_steps]     Decoder input IDs              │
│   target_batch       [batch, max_dec_steps]     Target IDs                     │
│   dec_padding_mask   [batch, max_dec_steps]     1=real, 0=padding              │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Loop

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING LOOP                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   MAIN TRAINING FUNCTION (run_summarization.py):                                  │
│   ─────────────────────────────────────────────                                   │
│                                                                                   │
│   def run_training(model, batcher, sess, sv):                                    │
│       """Runs training loop."""                                                  │
│                                                                                   │
│       # Training loop - runs forever until killed                                │
│       while True:                                                                │
│           # 1. Get next batch                                                    │
│           batch = batcher.next_batch()                                          │
│                                                                                   │
│           # 2. Time the step                                                     │
│           t0 = time.time()                                                       │
│                                                                                   │
│           # 3. Run training step                                                 │
│           results = model.run_train_step(sess, batch)                           │
│                                                                                   │
│           # 4. Log progress                                                      │
│           t1 = time.time()                                                       │
│           step = results['global_step']                                         │
│           loss = results['loss']                                                │
│                                                                                   │
│           print("step %d: loss=%.3f (%.3f sec)" % (step, loss, t1-t0))         │
│                                                                                   │
│           # 5. Write summaries                                                   │
│           if step % 100 == 0:                                                   │
│               sv.summary_computed(sess, results['summaries'])                   │
│                                                                                   │
│           # 6. Flush summaries periodically                                      │
│           if step % 1000 == 0:                                                  │
│               sv.summary_writer.flush()                                         │
│                                                                                   │
│                                                                                   │
│   RUN_TRAIN_STEP (model.py):                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   def run_train_step(self, sess, batch):                                        │
│       """Runs one training step."""                                             │
│                                                                                   │
│       # Build feed dictionary                                                    │
│       feed_dict = self._make_feed_dict(batch)                                   │
│                                                                                   │
│       # Define what to compute                                                   │
│       to_return = {                                                             │
│           'train_op': self._train_op,        # Apply gradients                 │
│           'summaries': self._summaries,      # For Tensorboard                 │
│           'loss': self._loss,                # Total loss                      │
│           'global_step': self.global_step,   # Current step                    │
│       }                                                                          │
│                                                                                   │
│       # Optional: return coverage loss                                           │
│       if self._hps.coverage:                                                    │
│           to_return['coverage_loss'] = self._coverage_loss                      │
│                                                                                   │
│       # Run the graph                                                            │
│       return sess.run(to_return, feed_dict)                                     │
│                                                                                   │
│                                                                                   │
│   FEED DICTIONARY:                                                                │
│   ─────────────────                                                               │
│                                                                                   │
│   def _make_feed_dict(self, batch):                                             │
│       """Creates feed dictionary from batch."""                                  │
│       feed_dict = {                                                             │
│           self._enc_batch:        batch.enc_batch,                              │
│           self._enc_lens:         batch.enc_lens,                               │
│           self._enc_padding_mask: batch.enc_padding_mask,                       │
│           self._dec_batch:        batch.dec_batch,                              │
│           self._target_batch:     batch.target_batch,                           │
│           self._dec_padding_mask: batch.dec_padding_mask,                       │
│       }                                                                          │
│                                                                                   │
│       if self._hps.pointer_gen:                                                 │
│           feed_dict[self._enc_batch_extend_vocab] = \                           │
│               batch.enc_batch_extend_vocab                                      │
│           feed_dict[self._max_art_oovs] = batch.max_art_oovs                   │
│                                                                                   │
│       return feed_dict                                                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Optimization Details

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       OPTIMIZATION DETAILS                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   OPTIMIZER: Adagrad                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   Why Adagrad?                                                                    │
│   • Adapts learning rate per parameter                                           │
│   • Good for sparse gradients (common in NLP)                                    │
│   • Simple and stable                                                             │
│                                                                                   │
│   How Adagrad works:                                                              │
│                                                                                   │
│   accumulator_t = accumulator_{t-1} + gradient_t^2                               │
│   param_t = param_{t-1} - lr × gradient_t / sqrt(accumulator_t + ε)             │
│                                                                                   │
│   • Frequent parameters get smaller updates (large accumulator)                  │
│   • Rare parameters get larger updates (small accumulator)                       │
│                                                                                   │
│                                                                                   │
│   IMPLEMENTATION (model.py):                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   def _add_train_op(self):                                                       │
│       """Adds training operation to graph."""                                    │
│                                                                                   │
│       # 1. Compute loss (already done in _add_output_projection)                │
│       loss_to_minimize = self._loss                                             │
│       if self._hps.coverage:                                                    │
│           loss_to_minimize += self._hps.cov_loss_wt * self._coverage_loss       │
│                                                                                   │
│       # 2. Get all trainable variables                                          │
│       tvars = tf.trainable_variables()                                          │
│                                                                                   │
│       # 3. Compute gradients                                                     │
│       gradients = tf.gradients(loss_to_minimize, tvars,                         │
│                               aggregation_method=2)                             │
│                                                                                   │
│       # 4. Clip gradients                                                        │
│       grads, global_norm = tf.clip_by_global_norm(                              │
│           gradients, self._hps.max_grad_norm                                    │
│       )                                                                          │
│                                                                                   │
│       # 5. Create optimizer                                                      │
│       optimizer = tf.train.AdagradOptimizer(                                    │
│           self._hps.lr,                                                         │
│           initial_accumulator_value=self._hps.adagrad_init_acc                  │
│       )                                                                          │
│                                                                                   │
│       # 6. Apply gradients                                                       │
│       self._train_op = optimizer.apply_gradients(                               │
│           zip(grads, tvars),                                                    │
│           global_step=self.global_step,                                         │
│           name='train_step'                                                     │
│       )                                                                          │
│                                                                                   │
│                                                                                   │
│   ADAGRAD HYPERPARAMETERS:                                                        │
│   ────────────────────────                                                        │
│                                                                                   │
│   Parameter               Default    Description                                  │
│   ─────────               ───────    ───────────                                  │
│   lr (learning rate)      0.15       Step size                                   │
│   adagrad_init_acc        0.1        Initial accumulator value                   │
│                                                                                   │
│   Note: Learning rate 0.15 is quite high compared to Adam defaults (0.001).     │
│   This works with Adagrad because it rapidly decreases effective LR.            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Gradient Clipping

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        GRADIENT CLIPPING                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHY GRADIENT CLIPPING?                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   RNNs (including LSTMs) can suffer from EXPLODING GRADIENTS:                    │
│                                                                                   │
│   • Gradients are computed via backpropagation through time (BPTT)              │
│   • Long sequences mean many matrix multiplications                              │
│   • Gradients can grow exponentially: O(|λ|^T) where T is seq length            │
│   • Leads to NaN losses and training collapse                                    │
│                                                                                   │
│   Example:                                                                        │
│   ─────────                                                                       │
│   Without clipping:                                                               │
│   Step 1000: loss = 3.5                                                          │
│   Step 1001: loss = 45.2   (gradient spike)                                      │
│   Step 1002: loss = NaN    (exploded!)                                           │
│                                                                                   │
│                                                                                   │
│   CLIPPING METHOD: Global Norm Clipping                                           │
│   ─────────────────────────────────────                                           │
│                                                                                   │
│   Clip gradients by their GLOBAL NORM (L2 norm of all gradients combined).      │
│                                                                                   │
│   global_norm = sqrt(Σ_i ||g_i||^2)                                             │
│                                                                                   │
│   if global_norm > max_grad_norm:                                                │
│       g_i = g_i × (max_grad_norm / global_norm)  for all i                      │
│                                                                                   │
│   This SCALES all gradients proportionally, preserving their relative            │
│   directions while limiting the overall magnitude.                               │
│                                                                                   │
│                                                                                   │
│   VISUAL EXAMPLE:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Before clipping:                                                                │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  g1 = [1.0, 2.0]                                                        │   │
│   │  g2 = [3.0, 4.0]                                                        │   │
│   │  g3 = [100.0, 200.0]  ← HUGE!                                          │   │
│   │                                                                          │   │
│   │  global_norm = sqrt(1² + 2² + 3² + 4² + 100² + 200²)                   │   │
│   │              = sqrt(1 + 4 + 9 + 16 + 10000 + 40000)                     │   │
│   │              = sqrt(50030) ≈ 223.7                                      │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│   max_grad_norm = 2.0                                                            │
│   scale = 2.0 / 223.7 ≈ 0.0089                                                  │
│                                                                                   │
│   After clipping:                                                                 │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  g1 = [0.0089, 0.018]                                                   │   │
│   │  g2 = [0.027, 0.036]                                                    │   │
│   │  g3 = [0.89, 1.78]    ← Now reasonable!                                │   │
│   │                                                                          │   │
│   │  new_global_norm ≈ 2.0  ✓                                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│                                                                                   │
│   IMPLEMENTATION:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   # In model.py _add_train_op()                                                  │
│                                                                                   │
│   # Compute gradients                                                            │
│   gradients = tf.gradients(loss, tvars)                                         │
│                                                                                   │
│   # Clip by global norm                                                          │
│   grads, global_norm = tf.clip_by_global_norm(                                  │
│       gradients,                                                                 │
│       self._hps.max_grad_norm  # Default: 2.0                                   │
│   )                                                                              │
│                                                                                   │
│   # Log global norm for debugging                                                │
│   tf.summary.scalar('global_norm', global_norm)                                 │
│                                                                                   │
│                                                                                   │
│   MONITORING:                                                                     │
│   ───────────                                                                     │
│                                                                                   │
│   Watch the global_norm in Tensorboard:                                          │
│   • Normal training: global_norm around 1-10                                    │
│   • Exploding: global_norm > 100 (clipping will activate)                       │
│   • If frequently clipping, consider lowering learning rate                     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Checkpointing

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          CHECKPOINTING                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE:                                                                        │
│   ────────                                                                        │
│   • Save model weights periodically                                              │
│   • Resume training after interruption                                           │
│   • Evaluate different training stages                                           │
│   • Deploy trained model                                                         │
│                                                                                   │
│                                                                                   │
│   CHECKPOINT DIRECTORY STRUCTURE:                                                 │
│   ────────────────────────────────                                                │
│                                                                                   │
│   log_root/                                                                       │
│   └── exp_name/                                                                   │
│       └── train/                                                                  │
│           ├── checkpoint            # List of saved checkpoints                  │
│           ├── model.ckpt-1000.data  # Weights at step 1000                      │
│           ├── model.ckpt-1000.index # Index file                                │
│           ├── model.ckpt-1000.meta  # Graph metadata                            │
│           ├── model.ckpt-2000.data  # Weights at step 2000                      │
│           ├── model.ckpt-2000.index                                             │
│           ├── model.ckpt-2000.meta                                              │
│           └── events.out.tfevents.* # Tensorboard logs                          │
│                                                                                   │
│                                                                                   │
│   SAVING CHECKPOINTS (Supervisor handles this):                                   │
│   ─────────────────────────────────────────────                                   │
│                                                                                   │
│   # Create supervisor                                                             │
│   sv = tf.train.Supervisor(                                                      │
│       logdir=train_dir,                                                          │
│       is_chief=True,                                                             │
│       saver=saver,                                                               │
│       summary_op=None,          # We write summaries manually                    │
│       save_summaries_secs=60,   # Save summaries every 60 seconds               │
│       save_model_secs=60,       # Save checkpoint every 60 seconds              │
│       global_step=model.global_step                                             │
│   )                                                                              │
│                                                                                   │
│                                                                                   │
│   RESTORING FROM CHECKPOINT:                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   # Supervisor automatically restores latest checkpoint if exists                │
│   sess = sv.prepare_or_wait_for_session(config=config)                          │
│                                                                                   │
│   # Or manually:                                                                  │
│   ckpt = tf.train.get_checkpoint_state(train_dir)                               │
│   if ckpt and ckpt.model_checkpoint_path:                                       │
│       saver.restore(sess, ckpt.model_checkpoint_path)                           │
│       print("Restored from %s" % ckpt.model_checkpoint_path)                    │
│                                                                                   │
│                                                                                   │
│   BEST CHECKPOINT TRACKING:                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   # During evaluation, track best checkpoint                                      │
│   best_loss = float('inf')                                                       │
│   best_ckpt = None                                                               │
│                                                                                   │
│   for ckpt_path in all_checkpoints:                                             │
│       loss = evaluate(ckpt_path)                                                │
│       if loss < best_loss:                                                       │
│           best_loss = loss                                                       │
│           best_ckpt = ckpt_path                                                 │
│                                                                                   │
│   # Use best_ckpt for decoding                                                   │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tensorboard Logging

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       TENSORBOARD LOGGING                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   METRICS LOGGED:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   1. Loss metrics:                                                                │
│      • loss: Total loss (NLL + coverage)                                        │
│      • coverage_loss: Coverage loss only (if enabled)                           │
│                                                                                   │
│   2. Gradient metrics:                                                            │
│      • global_norm: L2 norm of all gradients                                    │
│      (Useful for debugging exploding gradients)                                  │
│                                                                                   │
│                                                                                   │
│   ADDING SUMMARIES (model.py):                                                    │
│   ────────────────────────────                                                    │
│                                                                                   │
│   def _add_train_op(self):                                                       │
│       ...                                                                         │
│       # Create summary operations                                                │
│       tf.summary.scalar('loss', self._loss)                                     │
│       tf.summary.scalar('global_norm', global_norm)                             │
│                                                                                   │
│       if self._hps.coverage:                                                    │
│           tf.summary.scalar('coverage_loss', self._coverage_loss)               │
│                                                                                   │
│       # Merge all summaries                                                      │
│       self._summaries = tf.summary.merge_all()                                  │
│                                                                                   │
│                                                                                   │
│   WRITING SUMMARIES (run_summarization.py):                                       │
│   ─────────────────────────────────────────                                       │
│                                                                                   │
│   # In training loop                                                              │
│   results = model.run_train_step(sess, batch)                                   │
│                                                                                   │
│   if step % 100 == 0:                                                           │
│       # Write summaries to Tensorboard                                          │
│       sv.summary_computed(sess, results['summaries'])                           │
│                                                                                   │
│                                                                                   │
│   VIEWING TENSORBOARD:                                                            │
│   ────────────────────                                                            │
│                                                                                   │
│   # Start Tensorboard server                                                     │
│   tensorboard --logdir=/path/to/log_root/exp_name/train                         │
│                                                                                   │
│   # Open browser                                                                  │
│   http://localhost:6006                                                          │
│                                                                                   │
│                                                                                   │
│   WHAT TO LOOK FOR:                                                               │
│   ─────────────────                                                               │
│                                                                                   │
│   Healthy training:                                                               │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │  Loss                                                                  │     │
│   │   ↑                                                                    │     │
│   │  5│▄▄                                                                  │     │
│   │   │ ▀▀▄▄                                                               │     │
│   │  3│    ▀▀▀▄▄▄▄                                                         │     │
│   │   │         ▀▀▀▀▀▀▄▄▄▄▄▄▄▄▄▄▄▄▄▄                                       │     │
│   │  1│                              ▀▀▀▀▀▀▀▀▀───────────                  │     │
│   │   └───────────────────────────────────────────────────▶ Step          │     │
│   │        0      50K     100K    150K    200K                             │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│   Problematic training:                                                           │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │  Loss                                                                  │     │
│   │   ↑                                                                    │     │
│   │  5│▄▄            ▄                                                     │     │
│   │   │ ▀▀▄▄        ▄█▄    ← Spikes = instability                         │     │
│   │  3│    ▀▀▀▄    ▄▀ ▀▄                                                   │     │
│   │   │        ▀▀▄▀   ▀▀▀▄                                                 │     │
│   │  1│                   ▀ ← Not converging                              │     │
│   │   └───────────────────────────────────────────────────▶ Step          │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Evaluation Mode

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         EVALUATION MODE                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE:                                                                        │
│   ────────                                                                        │
│   • Validate model during training (on held-out data)                            │
│   • Monitor overfitting                                                           │
│   • Select best checkpoint                                                        │
│                                                                                   │
│                                                                                   │
│   COMMAND:                                                                        │
│   ────────                                                                        │
│   python run_summarization.py --mode=eval --data_path=val_*.bin                 │
│                                                                                   │
│                                                                                   │
│   DIFFERENCES FROM TRAINING:                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   Training:                                                                       │
│   • single_pass=False (loop forever through data)                               │
│   • Runs forward AND backward pass                                              │
│   • Updates weights                                                              │
│   • Uses dropout (if any)                                                        │
│                                                                                   │
│   Evaluation:                                                                     │
│   • single_pass=True (go through data once)                                     │
│   • Runs forward pass ONLY                                                      │
│   • Does NOT update weights                                                     │
│   • No dropout                                                                   │
│                                                                                   │
│                                                                                   │
│   EVALUATION LOOP (run_summarization.py):                                         │
│   ────────────────────────────────────────                                        │
│                                                                                   │
│   def run_eval(model, batcher, sess):                                            │
│       """Runs evaluation loop."""                                                │
│                                                                                   │
│       running_loss = 0.0                                                         │
│       num_batches = 0                                                            │
│                                                                                   │
│       while True:                                                                │
│           batch = batcher.next_batch()                                          │
│                                                                                   │
│           if batch is None:  # End of data                                      │
│               break                                                             │
│                                                                                   │
│           # Run forward pass only                                               │
│           results = model.run_eval_step(sess, batch)                            │
│           loss = results['loss']                                                │
│                                                                                   │
│           running_loss += loss                                                  │
│           num_batches += 1                                                      │
│                                                                                   │
│       # Compute average loss                                                     │
│       avg_loss = running_loss / num_batches                                     │
│       print("Eval loss: %.3f" % avg_loss)                                       │
│       return avg_loss                                                           │
│                                                                                   │
│                                                                                   │
│   RUN_EVAL_STEP (model.py):                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   def run_eval_step(self, sess, batch):                                         │
│       """Runs one eval step (no gradient computation)."""                       │
│                                                                                   │
│       feed_dict = self._make_feed_dict(batch)                                   │
│                                                                                   │
│       to_return = {                                                             │
│           'summaries': self._summaries,                                         │
│           'loss': self._loss,                                                   │
│           'global_step': self.global_step,                                      │
│       }                                                                          │
│                                                                                   │
│       # NOTE: No train_op! Just compute loss.                                   │
│       return sess.run(to_return, feed_dict)                                     │
│                                                                                   │
│                                                                                   │
│   CONTINUOUS EVALUATION:                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   # Run eval in a separate process while training                                │
│   # It will automatically pick up new checkpoints                               │
│                                                                                   │
│   Terminal 1: python run_summarization.py --mode=train ...                      │
│   Terminal 2: python run_summarization.py --mode=eval ...                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Tips

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          TRAINING TIPS                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   1. TWO-PHASE TRAINING (for coverage)                                            │
│   ────────────────────────────────────                                            │
│                                                                                   │
│   Phase 1: Train WITHOUT coverage                                                 │
│   python run_summarization.py --coverage=False --pointer_gen=True               │
│   # Train until convergence (~100K-200K steps)                                   │
│                                                                                   │
│   Phase 2: Fine-tune WITH coverage                                                │
│   python run_summarization.py --coverage=True --convert_to_coverage_model       │
│   # Then continue training with coverage                                         │
│                                                                                   │
│                                                                                   │
│   2. MONITOR KEY METRICS                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   In Tensorboard, watch:                                                          │
│   • loss: Should decrease and stabilize                                          │
│   • global_norm: Should be <10 normally, spikes indicate instability            │
│   • coverage_loss: Should decrease as model learns to not repeat                │
│                                                                                   │
│                                                                                   │
│   3. LEARNING RATE TUNING                                                         │
│   ───────────────────────                                                         │
│                                                                                   │
│   Default LR=0.15 works well with Adagrad.                                       │
│                                                                                   │
│   If training is unstable:                                                        │
│   • Try LR=0.1 or LR=0.05                                                        │
│   • Increase max_grad_norm to 5.0                                               │
│                                                                                   │
│   If training is too slow:                                                        │
│   • Try LR=0.2 (but watch for instability)                                      │
│                                                                                   │
│                                                                                   │
│   4. BATCH SIZE                                                                   │
│   ─────────────                                                                   │
│                                                                                   │
│   Default: batch_size=16                                                          │
│                                                                                   │
│   • Larger batch (32, 64): More stable gradients, faster per-step               │
│     But needs more GPU memory                                                    │
│                                                                                   │
│   • Smaller batch (8): Less memory, but noisier gradients                       │
│     May need to reduce LR                                                        │
│                                                                                   │
│                                                                                   │
│   5. MEMORY OPTIMIZATION                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   If running out of GPU memory:                                                   │
│   • Reduce batch_size                                                            │
│   • Reduce max_enc_steps (but may hurt quality)                                 │
│   • Reduce hidden_dim (but may hurt quality)                                    │
│                                                                                   │
│                                                                                   │
│   6. COMMON ISSUES AND SOLUTIONS                                                  │
│   ──────────────────────────────                                                  │
│                                                                                   │
│   Loss doesn't decrease:                                                          │
│   • Check data pipeline (is data being loaded?)                                 │
│   • Try higher learning rate                                                    │
│   • Check for bugs in preprocessing                                             │
│                                                                                   │
│   Loss explodes (NaN):                                                            │
│   • Reduce learning rate                                                        │
│   • Reduce max_grad_norm                                                        │
│   • Check for inf/nan in data                                                   │
│                                                                                   │
│   Model produces repetitive output:                                               │
│   • Enable coverage mechanism                                                    │
│   • Make sure coverage loss is being computed                                   │
│                                                                                   │
│   Model produces too-short summaries:                                             │
│   • Increase min_dec_steps                                                       │
│   • Check beam search implementation                                            │
│                                                                                   │
│                                                                                   │
│   7. EXPECTED TRAINING TIME                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   On CNN/DailyMail dataset:                                                       │
│   • ~200K steps for baseline (without coverage)                                 │
│   • ~250K steps total (with coverage fine-tuning)                               │
│   • About 3-5 days on a single GPU                                              │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

Training pipeline key points:

1. **Setup**: Load vocab, create batcher, build graph
2. **Loop**: Get batch → Run train step → Log → Save checkpoint
3. **Optimizer**: Adagrad with LR=0.15
4. **Gradient clipping**: Global norm, max=2.0
5. **Checkpointing**: Auto-save every 60 seconds
6. **Tensorboard**: Monitor loss and gradient norm
7. **Two-phase training**: Base model first, then coverage

---

*Next: [11_loss_functions.md](11_loss_functions.md) - Loss Calculation Detailed*
