# Training Pipeline Comparison

## Table of Contents
1. [Overview](#overview)
2. [Original Training Pipeline](#original-training-pipeline)
3. [Proposed Training Pipeline](#proposed-training-pipeline)
4. [Optimizer Comparison](#optimizer-comparison)
5. [Learning Rate Scheduling](#learning-rate-scheduling)
6. [Loss Functions](#loss-functions)
7. [Early Stopping and Checkpointing](#early-stopping-and-checkpointing)
8. [Code Comparison](#code-comparison)

---

## Overview

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Framework** | TensorFlow 1.x | PyTorch |
| **Optimizer** | Adagrad | AdamW |
| **Learning Rate** | Fixed (0.15) | Warmup + Cosine decay |
| **Gradient Clipping** | Global norm (2.0) | Global norm (0.8) |
| **Early Stopping** | Based on running avg loss | Based on validation loss |
| **Mixed Precision** | No | Yes (AMP) |
| **Checkpointing** | TensorFlow Supervisor | Manual PyTorch saving |

---

## Original Training Pipeline

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      ORIGINAL TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INITIALIZATION                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Load vocabulary                                                    │   │
│  │  - Create Batcher (multi-threaded data loading)                      │   │
│  │  - Build TensorFlow graph                                            │   │
│  │  - Initialize variables                                               │   │
│  │  - Create TensorFlow Supervisor                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. TRAINING LOOP (infinite)                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  while True:                                                          │   │
│  │      batch = batcher.next_batch()                                    │   │
│  │      feed_dict = make_feed_dict(batch)                               │   │
│  │                                                                       │   │
│  │      results = sess.run({                                            │   │
│  │          'train_op': self._train_op,                                 │   │
│  │          'summaries': self._summaries,                               │   │
│  │          'loss': self._loss,                                         │   │
│  │          'global_step': self.global_step                             │   │
│  │      }, feed_dict)                                                   │   │
│  │                                                                       │   │
│  │      # Log to TensorBoard                                            │   │
│  │      summary_writer.add_summary(results['summaries'])                │   │
│  │                                                                       │   │
│  │      # Supervisor auto-saves every 60 seconds                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. EVALUATION (separate process)                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  while True:                                                          │   │
│  │      load_latest_checkpoint()                                        │   │
│  │      batch = batcher.next_batch()                                    │   │
│  │      loss = sess.run(self._loss, feed_dict)                         │   │
│  │      running_avg_loss = decay * running_avg + (1-decay) * loss      │   │
│  │                                                                       │   │
│  │      if running_avg_loss < best_loss:                                │   │
│  │          save_best_model()                                           │   │
│  │          best_loss = running_avg_loss                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original Code

```python
# File: run_summarization.py, lines 183-216

def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    """Repeatedly runs training iterations, logging loss to screen."""
    tf.logging.info("starting run_training")
    
    with sess_context_manager as sess:
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess)
            sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
        
        while True:  # Infinite training loop
            batch = batcher.next_batch()
            
            tf.logging.info('running training step...')
            t0 = time.time()
            results = model.run_train_step(sess, batch)
            t1 = time.time()
            tf.logging.info('seconds for training step: %.3f', t1-t0)
            
            loss = results['loss']
            tf.logging.info('loss: %f', loss)
            
            if not np.isfinite(loss):
                raise Exception("Loss is not finite. Stopping.")
            
            if FLAGS.coverage:
                coverage_loss = results['coverage_loss']
                tf.logging.info("coverage_loss: %f", coverage_loss)
            
            # Write summaries to TensorBoard
            summaries = results['summaries']
            train_step = results['global_step']
            summary_writer.add_summary(summaries, train_step)
            
            if train_step % 100 == 0:
                summary_writer.flush()

# File: model.py, lines 288-305

def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    loss_to_minimize = self._total_loss if self._hps.coverage else self._loss
    tvars = tf.trainable_variables()
    
    # Compute gradients
    gradients = tf.gradients(
        loss_to_minimize, tvars, 
        aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE
    )
    
    # Clip gradients by global norm
    with tf.device("/gpu:0"):
        grads, global_norm = tf.clip_by_global_norm(gradients, self._hps.max_grad_norm)
    
    tf.summary.scalar('global_norm', global_norm)
    
    # Apply Adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(
        self._hps.lr,  # 0.15
        initial_accumulator_value=self._hps.adagrad_init_acc  # 0.1
    )
    
    with tf.device("/gpu:0"):
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars), 
            global_step=self.global_step, 
            name='train_step'
        )
```

---

## Proposed Training Pipeline

### Training Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      PROPOSED TRAINING PIPELINE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  1. INITIALIZATION                                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Load config from YAML                                             │   │
│  │  - Set random seeds (reproducibility)                                │   │
│  │  - Load datasets from pickle files                                   │   │
│  │  - Create DataLoaders                                                │   │
│  │  - Initialize model                                                  │   │
│  │  - Initialize optimizer (AdamW)                                      │   │
│  │  - Initialize AMP scaler (mixed precision)                          │   │
│  │  - Create experiment directory                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2. TRAINING LOOP (epoch-based)                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  for epoch in range(num_epochs):                                     │   │
│  │      # Set learning rate (warmup + cosine)                          │   │
│  │      lr = get_lr(epoch)                                             │   │
│  │      set_lr(optimizer, lr)                                          │   │
│  │                                                                       │   │
│  │      # Training                                                       │   │
│  │      model.train()                                                   │   │
│  │      for x, y, x_dict in train_loader:                              │   │
│  │          optimizer.zero_grad()                                       │   │
│  │                                                                       │   │
│  │          with autocast():  # Mixed precision                        │   │
│  │              logits = model(x, x_dict)                              │   │
│  │              loss = criterion(logits, y)                            │   │
│  │                                                                       │   │
│  │          scaler.scale(loss).backward()                              │   │
│  │          scaler.unscale_(optimizer)                                 │   │
│  │          clip_grad_norm_(model.parameters(), max_norm)              │   │
│  │          scaler.step(optimizer)                                     │   │
│  │          scaler.update()                                            │   │
│  │                                                                       │   │
│  │      # Validation                                                     │   │
│  │      val_metrics = evaluate(val_loader)                             │   │
│  │                                                                       │   │
│  │      # Early stopping check                                          │   │
│  │      if val_loss < best_val_loss:                                   │   │
│  │          best_val_loss = val_loss                                   │   │
│  │          save_checkpoint('best.pt')                                 │   │
│  │          patience_counter = 0                                        │   │
│  │      else:                                                           │   │
│  │          patience_counter += 1                                       │   │
│  │          if patience_counter >= patience:                           │   │
│  │              break  # Early stop                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3. FINAL EVALUATION                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  load_checkpoint('best.pt')                                          │   │
│  │  val_metrics = evaluate(val_loader)                                  │   │
│  │  test_metrics = evaluate(test_loader)                                │   │
│  │  save_results()                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Code

```python
# File: train_pointer_v45.py, lines 410-450

def train_epoch(self) -> float:
    """Train for one epoch."""
    self.model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
    
    for x, y, x_dict in pbar:
        # Move to device
        x = x.to(self.device)
        y = y.to(self.device)
        x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
        
        self.optimizer.zero_grad()
        
        if self.scaler:  # Mixed precision
            with torch.cuda.amp.autocast():
                logits = self.model(x, x_dict)
                loss = self.criterion(logits, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            logits = self.model(x, x_dict)
            loss = self.criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg': f"{total_loss/num_batches:.4f}",
        })
    
    return total_loss / num_batches

# File: train_pointer_v45.py, lines 511-547

def train(self) -> Dict:
    """Full training loop."""
    self.logger.info(f"Training for {self.num_epochs} epochs")
    self.logger.info(f"Model parameters: {self.model.count_parameters():,}")
    
    for epoch in range(self.num_epochs):
        self.current_epoch = epoch + 1
        
        # Set learning rate (warmup + cosine)
        lr = self._get_lr(epoch)
        self._set_lr(lr)
        
        # Train
        train_loss = self.train_epoch()
        
        # Validate
        val_metrics = self.evaluate(self.val_loader, "val")
        
        self.logger.info(
            f"Epoch {self.current_epoch}/{self.num_epochs} | "
            f"LR: {lr:.2e} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_metrics['loss']:.4f} | "
            f"Acc@1: {val_metrics['acc@1']:.2f}%"
        )
        
        # Early stopping check
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.patience_counter = 0
            self._save_checkpoint("best.pt")
            self.logger.info(f"  ✓ New best (Acc@1: {val_metrics['acc@1']:.2f}%)")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience and self.current_epoch >= self.min_epochs:
                self.logger.info(f"Early stopping at epoch {self.current_epoch}")
                break
```

---

## Optimizer Comparison

### Adagrad (Original)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ADAGRAD OPTIMIZER                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Update Rule:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  G_t = G_{t-1} + g_t²                  (Accumulate squared gradients) │   │
│  │  θ_t = θ_{t-1} - lr / √(G_t + ε) × g_t (Parameter update)           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Characteristics:                                                            │
│  - Learning rate decreases over time (monotonically)                        │
│  - Good for sparse gradients (NLP, text)                                    │
│  - No momentum                                                               │
│  - Simple, well-understood                                                   │
│                                                                              │
│  Configuration:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  lr = 0.15                                                           │   │
│  │  initial_accumulator_value = 0.1                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Code:                                                                       │
│  optimizer = tf.train.AdagradOptimizer(                                     │
│      learning_rate=0.15,                                                    │
│      initial_accumulator_value=0.1                                          │
│  )                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### AdamW (Proposed)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            ADAMW OPTIMIZER                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Update Rule:                                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  m_t = β₁ × m_{t-1} + (1-β₁) × g_t     (First moment estimate)       │   │
│  │  v_t = β₂ × v_{t-1} + (1-β₂) × g_t²    (Second moment estimate)      │   │
│  │  m̂_t = m_t / (1 - β₁^t)                (Bias correction)            │   │
│  │  v̂_t = v_t / (1 - β₂^t)                (Bias correction)            │   │
│  │  θ_t = θ_{t-1} - lr × m̂_t / (√v̂_t + ε) - lr × λ × θ_{t-1}         │   │
│  │                                         (Update with weight decay)   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Characteristics:                                                            │
│  - Adaptive learning rates per parameter                                     │
│  - Momentum for faster convergence                                           │
│  - Decoupled weight decay (better regularization)                           │
│  - Standard for Transformers                                                 │
│                                                                              │
│  Configuration:                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  lr = 3e-4 to 6.5e-4                                                 │   │
│  │  betas = (0.9, 0.98)                                                 │   │
│  │  eps = 1e-9                                                          │   │
│  │  weight_decay = 0.015                                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Code:                                                                       │
│  optimizer = optim.AdamW(                                                   │
│      model.parameters(),                                                    │
│      lr=3e-4,                                                               │
│      weight_decay=0.015,                                                    │
│      betas=(0.9, 0.98),                                                     │
│      eps=1e-9                                                               │
│  )                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Comparison Table

| Aspect | Adagrad | AdamW |
|--------|---------|-------|
| **Learning Rate** | 0.15 (high, fixed effective) | 3e-4 to 6.5e-4 (low, scheduled) |
| **Momentum** | None | β₁ = 0.9 |
| **Adaptive LR** | Yes (decreasing only) | Yes (bidirectional) |
| **Weight Decay** | No | 0.015 (decoupled) |
| **Typical Use** | NLP/Sparse data | Transformers/Dense data |

---

## Learning Rate Scheduling

### Original: Fixed Learning Rate

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     ORIGINAL: FIXED LEARNING RATE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LR                                                                          │
│  │                                                                           │
│  │  0.15 ┼────────────────────────────────────────────────                  │
│  │       │                                                                   │
│  │       │    (Adagrad internally reduces effective LR over time            │
│  │       │     but the base LR stays constant)                              │
│  │       │                                                                   │
│  └───────┼───────────────────────────────────────────────────→ Steps        │
│          0                                                                   │
│                                                                              │
│  Note: No explicit scheduling; Adagrad's accumulator naturally              │
│        decreases the effective learning rate over training.                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Warmup + Cosine Decay

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                PROPOSED: WARMUP + COSINE DECAY                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  LR                                                                          │
│  │                                                                           │
│  │           ╭─── Peak LR (6.5e-4)                                          │
│  │          ╱ ╲                                                             │
│  │         ╱   ╲                                                            │
│  │        ╱     ╲                                                           │
│  │       ╱       ╲  ← Cosine Decay                                         │
│  │      ╱         ╲                                                         │
│  │     ╱           ╲                                                        │
│  │    ╱             ╲                                                       │
│  │   ╱               ╲                                                      │
│  │  ╱  ← Warmup       ╲                                                     │
│  │ ╱                   ╲________ Min LR (1e-6)                              │
│  └───────┼─────────────┼─────────────────────────────────────→ Epoch       │
│          0      5 (warmup)                               50                  │
│                                                                              │
│  Formula:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  if epoch < warmup_epochs:                                           │   │
│  │      lr = base_lr × (epoch + 1) / warmup_epochs                     │   │
│  │  else:                                                               │   │
│  │      progress = (epoch - warmup) / (total - warmup)                 │   │
│  │      lr = min_lr + 0.5 × (base_lr - min_lr) × (1 + cos(π × progress))│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Code Comparison

```python
# ORIGINAL: No explicit scheduling
# Adagrad handles this internally

# PROPOSED: Manual warmup + cosine scheduling
# File: train_pointer_v45.py, lines 372-378

def _get_lr(self, epoch: int) -> float:
    """Get learning rate with warmup and cosine decay."""
    if epoch < self.warmup_epochs:
        # Linear warmup
        return self.base_lr * (epoch + 1) / self.warmup_epochs
    
    # Cosine decay
    progress = (epoch - self.warmup_epochs) / max(1, self.num_epochs - self.warmup_epochs)
    return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(math.pi * progress))

def _set_lr(self, lr: float):
    """Set learning rate for all parameter groups."""
    for param_group in self.optimizer.param_groups:
        param_group['lr'] = lr
```

---

## Loss Functions

### Original Loss

```python
# File: model.py, lines 249-278

# Standard cross-entropy for baseline
self._loss = tf.contrib.seq2seq.sequence_loss(
    tf.stack(vocab_scores, axis=1),  # [batch, max_dec_steps, vocab_size]
    self._target_batch,               # [batch, max_dec_steps]
    self._dec_padding_mask            # [batch, max_dec_steps]
)

# For pointer-generator: Negative log likelihood
if FLAGS.pointer_gen:
    loss_per_step = []
    for dec_step, dist in enumerate(final_dists):
        targets = self._target_batch[:, dec_step]
        indices = tf.stack((batch_nums, targets), axis=1)
        gold_probs = tf.gather_nd(dist, indices)  # Prob of correct word
        losses = -tf.log(gold_probs)
        loss_per_step.append(losses)
    
    self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)

# Optional coverage loss
if hps.coverage:
    self._coverage_loss = _coverage_loss(self.attn_dists, self._dec_padding_mask)
    self._total_loss = self._loss + hps.cov_loss_wt * self._coverage_loss
```

### Proposed Loss

```python
# File: train_pointer_v45.py, lines 334-337

# Cross-entropy with label smoothing
self.criterion = nn.CrossEntropyLoss(
    ignore_index=0,              # Ignore padding
    label_smoothing=0.03,        # Smooth labels for regularization
)

# During forward pass (pointer_v45.py, line 249)
# Model outputs log probabilities
return torch.log(final_probs + 1e-10)

# CrossEntropyLoss expects raw logits, but we use log_probs
# So effectively: loss = -log(softmax(logits))[target]
#                      = -log_probs[target]
```

### Loss Comparison

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Base Loss** | Negative log likelihood | Cross-entropy |
| **Sequence Handling** | Per-step, then average | Single step (no decoder) |
| **Padding Handling** | Manual masking | ignore_index=0 |
| **Label Smoothing** | No | 0.03 |
| **Coverage Loss** | Optional (λ × coverage) | No |

---

## Early Stopping and Checkpointing

### Original Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL EARLY STOPPING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Separate training and evaluation processes:                                 │
│                                                                              │
│  TRAINING PROCESS (infinite loop):                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Trains indefinitely                                               │   │
│  │  - Supervisor saves checkpoints every 60 seconds                     │   │
│  │  - No explicit early stopping in training                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  EVAL PROCESS (monitors training):                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - Loads latest checkpoint periodically                              │   │
│  │  - Computes running average loss: decay × old + (1-decay) × new     │   │
│  │  - If running_avg_loss < best_loss: save as "bestmodel"             │   │
│  │  - User manually stops training when satisfied                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  No automatic early stopping - user decides when to stop!                   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Approach

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED EARLY STOPPING                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Single process with epoch-based early stopping:                            │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  patience = 5  (stop after 5 epochs without improvement)             │   │
│  │  min_epochs = 8 (train at least 8 epochs)                           │   │
│  │                                                                       │   │
│  │  for epoch in range(max_epochs):                                     │   │
│  │      train_epoch()                                                   │   │
│  │      val_loss = evaluate()                                           │   │
│  │                                                                       │   │
│  │      if val_loss < best_val_loss:                                    │   │
│  │          best_val_loss = val_loss                                    │   │
│  │          patience_counter = 0                                        │   │
│  │          save_checkpoint('best.pt')                                  │   │
│  │      else:                                                           │   │
│  │          patience_counter += 1                                       │   │
│  │                                                                       │   │
│  │      if patience_counter >= patience and epoch >= min_epochs:        │   │
│  │          break  # Early stop                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Automatic early stopping based on validation loss!                         │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Checkpointing Code

```python
# ORIGINAL (TensorFlow Supervisor)
# File: run_summarization.py, lines 153-172

sv = tf.train.Supervisor(
    logdir=train_dir,
    is_chief=True,
    saver=saver,
    save_summaries_secs=60,  # Save summaries every 60 seconds
    save_model_secs=60,      # Save checkpoint every 60 seconds
    global_step=model.global_step
)

# PROPOSED (Manual PyTorch)
# File: train_pointer_v45.py, lines 572-598

def _save_checkpoint(self, filename: str):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': self.current_epoch,
        'model_state_dict': self.model.state_dict(),
        'optimizer_state_dict': self.optimizer.state_dict(),
        'best_val_loss': self.best_val_loss,
        'patience_counter': self.patience_counter,
        'config': self.config,
    }
    if self.scaler:
        checkpoint['scaler_state_dict'] = self.scaler.state_dict()
    
    torch.save(checkpoint, self.checkpoint_dir / filename)

def _load_checkpoint(self, filename: str):
    """Load model checkpoint."""
    checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
    self.model.load_state_dict(checkpoint['model_state_dict'])
    # ... restore other state ...
```

---

## Code Comparison

### Full Training Loop Comparison

```python
# ==============================================================================
# ORIGINAL: TensorFlow Training Loop
# ==============================================================================

def run_training(model, batcher, sess_context_manager, sv, summary_writer):
    with sess_context_manager as sess:
        while True:  # INFINITE LOOP
            batch = batcher.next_batch()
            
            results = model.run_train_step(sess, batch)
            # run_train_step does:
            #   feed_dict = make_feed_dict(batch)
            #   return sess.run({
            #       'train_op': self._train_op,
            #       'summaries': self._summaries,
            #       'loss': self._loss,
            #       'global_step': self.global_step,
            #   }, feed_dict)
            
            loss = results['loss']
            if not np.isfinite(loss):
                raise Exception("Loss is not finite")
            
            # Log to TensorBoard
            summary_writer.add_summary(results['summaries'], results['global_step'])

# ==============================================================================
# PROPOSED: PyTorch Training Loop
# ==============================================================================

def train(self) -> Dict:
    for epoch in range(self.num_epochs):  # FINITE LOOP
        self.current_epoch = epoch + 1
        
        # Learning rate scheduling
        lr = self._get_lr(epoch)
        self._set_lr(lr)
        
        # Training epoch
        self.model.train()
        total_loss = 0.0
        
        for x, y, x_dict in self.train_loader:
            x, y = x.to(self.device), y.to(self.device)
            x_dict = {k: v.to(self.device) for k, v in x_dict.items()}
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            with torch.cuda.amp.autocast():
                logits = self.model(x, x_dict)
                loss = self.criterion(logits, y)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            total_loss += loss.item()
        
        # Validation
        val_metrics = self.evaluate(self.val_loader, "val")
        
        # Early stopping
        if val_metrics['loss'] < self.best_val_loss:
            self.best_val_loss = val_metrics['loss']
            self.patience_counter = 0
            self._save_checkpoint("best.pt")
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                break
```

---

## Summary

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Framework** | TensorFlow 1.x | PyTorch |
| **Optimizer** | Adagrad (lr=0.15) | AdamW (lr=3e-4 to 6.5e-4) |
| **LR Schedule** | Fixed (Adagrad adaptive) | Warmup + Cosine decay |
| **Gradient Clipping** | max_norm=2.0 | max_norm=0.8 |
| **Loss** | NLL + optional coverage | CrossEntropy + label smoothing |
| **Early Stopping** | Manual (eval process) | Automatic (patience) |
| **Mixed Precision** | No | Yes (AMP) |
| **Checkpointing** | TF Supervisor (auto) | Manual PyTorch |
| **Training Loop** | Infinite | Epoch-based |

The training pipeline changes reflect:
1. **Modern framework**: TensorFlow 1.x → PyTorch
2. **Modern optimizer**: Adagrad → AdamW (standard for Transformers)
3. **Better scheduling**: Fixed → Warmup + Cosine
4. **Efficiency**: No AMP → AMP for faster training
5. **Reproducibility**: Explicit early stopping and checkpointing

---

*Next: [08_DATA_PROCESSING.md](08_DATA_PROCESSING.md) - Data loading and batching strategies*
