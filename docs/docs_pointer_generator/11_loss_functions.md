# Loss Functions Deep Dive

## Table of Contents
1. [Loss Overview](#loss-overview)
2. [Negative Log-Likelihood Loss](#negative-log-likelihood-loss)
3. [Loss with Pointer-Generator](#loss-with-pointer-generator)
4. [Coverage Loss](#coverage-loss)
5. [Total Loss Calculation](#total-loss-calculation)
6. [Masking for Padding](#masking-for-padding)
7. [Implementation Details](#implementation-details)
8. [Worked Example](#worked-example)

---

## Loss Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           LOSS OVERVIEW                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHAT IS LOSS?                                                                   │
│   ─────────────                                                                   │
│                                                                                   │
│   Loss is a measure of how wrong the model's predictions are.                    │
│   Training minimizes the loss to make predictions closer to targets.             │
│                                                                                   │
│   Low loss = Good predictions                                                    │
│   High loss = Bad predictions                                                    │
│                                                                                   │
│                                                                                   │
│   LOSS COMPONENTS IN POINTER-GENERATOR:                                           │
│   ──────────────────────────────────────                                          │
│                                                                                   │
│   L_total = L_NLL + λ × L_coverage                                               │
│                                                                                   │
│   Where:                                                                          │
│   • L_NLL: Negative log-likelihood (main prediction loss)                        │
│   • L_coverage: Coverage loss (penalizes repeated attention)                     │
│   • λ (cov_loss_wt): Weight for coverage loss (default: 1.0)                    │
│                                                                                   │
│                                                                                   │
│   LOSS FLOW:                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   Target sequence: [germany, won, the, cup, STOP]                               │
│                         ↓                                                         │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  For each time step t:                                                   │   │
│   │                                                                          │   │
│   │  1. Model produces P(word) distribution                                 │   │
│   │  2. Get probability of correct target word                              │   │
│   │  3. Compute -log(P(target))                                             │   │
│   │  4. Sum across all time steps                                           │   │
│   │  5. Average over batch                                                  │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                         ↓                                                         │
│   L_NLL = (1/T) × Σ_t -log(P(y_t))                                              │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Negative Log-Likelihood Loss

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   NEGATIVE LOG-LIKELIHOOD LOSS                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FORMULA:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   L_NLL = -log P(y|x)                                                            │
│         = -log Π_t P(y_t | y_<t, x)                                             │
│         = -Σ_t log P(y_t | y_<t, x)                                             │
│         = Σ_t -log P(y_t | y_<t, x)                                             │
│                                                                                   │
│   In words: Sum of negative log probabilities of each target word.              │
│                                                                                   │
│                                                                                   │
│   WHY NEGATIVE LOG?                                                               │
│   ──────────────────                                                              │
│                                                                                   │
│   1. Probabilities are in [0, 1]                                                │
│   2. We want to MAXIMIZE probability of correct target                          │
│   3. But optimization MINIMIZES loss                                            │
│   4. So we use NEGATIVE log probability                                         │
│                                                                                   │
│   Also, log is numerically stable:                                               │
│   • Product of small numbers → can underflow to 0                               │
│   • Log of product = sum of logs → much more stable                             │
│                                                                                   │
│                                                                                   │
│   EXAMPLE:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   Target: "germany won"                                                          │
│                                                                                   │
│   Step 1: P(germany) = 0.2                                                       │
│           -log(0.2) = 1.61                                                       │
│                                                                                   │
│   Step 2: P(won | germany) = 0.3                                                │
│           -log(0.3) = 1.20                                                       │
│                                                                                   │
│   L_NLL = 1.61 + 1.20 = 2.81                                                    │
│                                                                                   │
│   If model improves:                                                              │
│   P(germany) = 0.8 → -log(0.8) = 0.22                                           │
│   P(won | germany) = 0.7 → -log(0.7) = 0.36                                     │
│   L_NLL = 0.22 + 0.36 = 0.58  ← LOWER! Model improved.                         │
│                                                                                   │
│                                                                                   │
│   CROSS-ENTROPY EQUIVALENT:                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   NLL is equivalent to cross-entropy with one-hot targets:                       │
│                                                                                   │
│   H(p, q) = -Σ_i p_i × log(q_i)                                                 │
│                                                                                   │
│   Where:                                                                          │
│   • p = one-hot target (1 at correct word, 0 elsewhere)                         │
│   • q = model's probability distribution                                         │
│                                                                                   │
│   Since p is one-hot, only the correct word term survives:                       │
│   H(p, q) = -log(q_target) = NLL                                                │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Loss with Pointer-Generator

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   LOSS WITH POINTER-GENERATOR                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   KEY DIFFERENCE:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Without pointer: P(word) comes from vocab distribution only                    │
│   With pointer: P(word) comes from COMBINED final distribution                   │
│                                                                                   │
│   P_final(w) = p_gen × P_vocab(w) + (1 - p_gen) × Σ_{i:w_i=w} α_ti              │
│                                                                                   │
│   The loss is computed on this FINAL distribution.                               │
│                                                                                   │
│                                                                                   │
│   HANDLING OOV WORDS:                                                             │
│   ───────────────────                                                             │
│                                                                                   │
│   For in-vocab words:                                                             │
│   • Look up probability in vocab distribution                                    │
│   • Combine with copy probability                                                │
│                                                                                   │
│   For OOV words (extended vocab):                                                │
│   • Vocab distribution contribution is 0 (word not in vocab)                    │
│   • All probability comes from copy mechanism                                    │
│   • P(OOV_word) = (1 - p_gen) × Σ_{i:w_i=OOV_word} α_ti                         │
│                                                                                   │
│                                                                                   │
│   EXAMPLE WITH OOV:                                                               │
│   ─────────────────                                                               │
│                                                                                   │
│   Article: "Elon announced the product"                                          │
│   Target:  "Elon made announcement"                                              │
│   "Elon" is OOV (not in vocab)                                                   │
│                                                                                   │
│   Step 1: Target = "Elon"                                                        │
│                                                                                   │
│   p_gen = 0.3 (model wants to copy)                                             │
│   P_vocab("Elon") = 0 (not in vocab!)                                           │
│   α_1 = 0.8 (attention on "Elon" in source)                                     │
│                                                                                   │
│   P_final("Elon") = 0.3 × 0 + 0.7 × 0.8 = 0.56                                  │
│                                                                                   │
│   Loss for this step: -log(0.56) = 0.58                                         │
│                                                                                   │
│                                                                                   │
│   IMPORTANT IMPLEMENTATION DETAIL:                                                │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   Target IDs must use EXTENDED vocabulary:                                        │
│   • In-vocab words: use regular vocab ID                                        │
│   • OOV words: use extended vocab ID (vocab_size + oov_index)                   │
│                                                                                   │
│   This allows the loss function to look up the correct probability              │
│   from the extended distribution.                                                │
│                                                                                   │
│   # Final distribution shape: [batch, vocab_size + max_art_oovs]                │
│   # Target shape: [batch, max_dec_steps]                                        │
│   # Target IDs can be 0 to vocab_size + max_art_oovs - 1                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coverage Loss

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         COVERAGE LOSS                                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE:                                                                        │
│   ────────                                                                        │
│   Penalize the model for attending to same positions repeatedly.                 │
│   This reduces repetition in generated summaries.                                │
│                                                                                   │
│                                                                                   │
│   FORMULA:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   L_cov = Σ_t Σ_i min(α_ti, c_ti)                                               │
│                                                                                   │
│   Where:                                                                          │
│   • α_ti: Attention weight on position i at step t                              │
│   • c_ti: Coverage at position i at step t (sum of previous attentions)         │
│   • min: Element-wise minimum                                                    │
│                                                                                   │
│                                                                                   │
│   INTUITION:                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   Coverage c_ti = Σ_{t'<t} α_{t'i}                                              │
│   (Total attention received by position i before step t)                         │
│                                                                                   │
│   min(α_ti, c_ti) is high when BOTH:                                            │
│   • Current attention α_ti is high (attending now)                              │
│   • Previous coverage c_ti is high (already attended before)                    │
│                                                                                   │
│   This creates a penalty for re-attending to already-covered positions.          │
│                                                                                   │
│                                                                                   │
│   EXAMPLE:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   Source: [germany, won, the, cup]                                              │
│                                                                                   │
│   Step 1: α₁ = [0.8, 0.1, 0.05, 0.05]                                          │
│           c₁ = [0.0, 0.0, 0.0, 0.0]  (no previous attention)                    │
│           min(α₁, c₁) = [0.0, 0.0, 0.0, 0.0]                                    │
│           covloss₁ = 0.0                                                         │
│                                                                                   │
│   Step 2: α₂ = [0.1, 0.7, 0.1, 0.1]                                            │
│           c₂ = [0.8, 0.1, 0.05, 0.05]  (previous attention)                     │
│           min(α₂, c₂) = [0.1, 0.1, 0.05, 0.05]                                  │
│           covloss₂ = 0.1 + 0.1 + 0.05 + 0.05 = 0.30                             │
│                                                                                   │
│   Step 3 (BAD - re-attending to "germany"):                                     │
│           α₃ = [0.7, 0.1, 0.1, 0.1]  ← High on "germany" again!               │
│           c₃ = [0.9, 0.8, 0.15, 0.15]                                           │
│           min(α₃, c₃) = [0.7, 0.1, 0.1, 0.1]                                    │
│           covloss₃ = 0.7 + 0.1 + 0.1 + 0.1 = 1.00  ← HIGH PENALTY!             │
│                                                                                   │
│   Step 3 (GOOD - attending to new position):                                     │
│           α₃ = [0.1, 0.1, 0.7, 0.1]  ← High on "the" (new!)                    │
│           c₃ = [0.9, 0.8, 0.15, 0.15]                                           │
│           min(α₃, c₃) = [0.1, 0.1, 0.15, 0.1]                                   │
│           covloss₃ = 0.1 + 0.1 + 0.15 + 0.1 = 0.45  ← LOWER!                   │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Total Loss Calculation

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      TOTAL LOSS CALCULATION                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FORMULA:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   L_total = L_NLL + λ × L_coverage                                               │
│                                                                                   │
│   Where λ = cov_loss_wt (default 1.0)                                           │
│                                                                                   │
│                                                                                   │
│   WITHOUT COVERAGE (--coverage=False):                                            │
│   ─────────────────────────────────────                                           │
│                                                                                   │
│   L_total = L_NLL                                                                │
│                                                                                   │
│   Only the negative log-likelihood is used.                                      │
│                                                                                   │
│                                                                                   │
│   WITH COVERAGE (--coverage=True):                                                │
│   ────────────────────────────────                                                │
│                                                                                   │
│   L_total = L_NLL + 1.0 × L_coverage                                            │
│                                                                                   │
│   Both terms contribute equally (when λ=1).                                      │
│                                                                                   │
│                                                                                   │
│   TYPICAL VALUES:                                                                 │
│   ────────────────                                                                │
│                                                                                   │
│   Early training (random model):                                                  │
│   • L_NLL ≈ 5-7 (low probability for correct words)                             │
│   • L_coverage ≈ 0.5-1.0                                                         │
│   • L_total ≈ 6-8                                                                │
│                                                                                   │
│   Converged model:                                                                │
│   • L_NLL ≈ 2-3 (higher probability for correct words)                          │
│   • L_coverage ≈ 0.1-0.3 (less repetitive attention)                            │
│   • L_total ≈ 2-3.5                                                              │
│                                                                                   │
│                                                                                   │
│   EFFECT OF λ (cov_loss_wt):                                                     │
│   ──────────────────────────                                                      │
│                                                                                   │
│   λ = 0: Coverage loss ignored (same as --coverage=False)                       │
│                                                                                   │
│   λ = 0.5: Mild coverage penalty                                                │
│   • Model prioritizes correct words over avoiding repetition                    │
│                                                                                   │
│   λ = 1.0 (default): Balanced                                                   │
│   • Equal weight on prediction and repetition avoidance                         │
│                                                                                   │
│   λ = 2.0: Strong coverage penalty                                              │
│   • May hurt fluency to avoid ANY repetition                                    │
│   • Can lead to unnatural outputs                                               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Masking for Padding

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       MASKING FOR PADDING                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   THE PROBLEM:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Sequences in a batch have different lengths.                                   │
│   Shorter sequences are padded to match the longest.                             │
│                                                                                   │
│   Example batch:                                                                  │
│   Seq 1: [germany, won, cup, STOP, PAD, PAD]                                    │
│   Seq 2: [the, team, played, well, today, STOP]                                 │
│                                                                                   │
│   We should NOT compute loss on PAD tokens!                                      │
│   • PAD is artificial, not part of the target                                   │
│   • Would bias model toward predicting PAD                                      │
│                                                                                   │
│                                                                                   │
│   SOLUTION: Masking                                                               │
│   ───────────────────                                                             │
│                                                                                   │
│   Multiply per-token loss by a mask (1 for real, 0 for padding):                │
│                                                                                   │
│   Seq 1 losses: [0.5, 0.3, 0.4, 0.2, 0.1, 0.1]                                 │
│   Seq 1 mask:   [1,   1,   1,   1,   0,   0  ]                                  │
│   Masked:       [0.5, 0.3, 0.4, 0.2, 0.0, 0.0]                                  │
│                                                                                   │
│   Then average over NON-PAD tokens only:                                         │
│   L = sum(masked) / sum(mask) = 1.4 / 4 = 0.35                                  │
│                                                                                   │
│                                                                                   │
│   IMPLEMENTATION (model.py):                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   def _mask_and_avg(values, padding_mask):                                       │
│       """                                                                         │
│       Apply mask to values and compute average.                                  │
│                                                                                   │
│       Args:                                                                       │
│           values: List of tensors [batch_size] per time step                    │
│           padding_mask: Tensor [batch_size, max_dec_steps]                      │
│                                                                                   │
│       Returns:                                                                    │
│           Scalar: Average of masked values                                       │
│       """                                                                         │
│       # Stack values: [max_dec_steps, batch_size]                               │
│       values_per_step = tf.stack(values, axis=1)                                │
│       # Shape: [batch_size, max_dec_steps]                                      │
│                                                                                   │
│       # Apply mask (element-wise multiply)                                       │
│       dec_lens = tf.reduce_sum(padding_mask, axis=1)  # [batch_size]           │
│       values_per_ex = tf.reduce_sum(                                            │
│           values_per_step * padding_mask, axis=1                                │
│       )  # [batch_size]                                                         │
│                                                                                   │
│       # Average per example                                                      │
│       values_per_ex /= dec_lens  # [batch_size]                                 │
│                                                                                   │
│       # Average over batch                                                       │
│       return tf.reduce_mean(values_per_ex)                                      │
│                                                                                   │
│                                                                                   │
│   VISUAL EXAMPLE:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Batch of 2 sequences:                                                           │
│                                                                                   │
│   losses = [[0.5, 0.3, 0.4, 0.2, 0.1, 0.1],   # Seq 1                          │
│             [0.6, 0.4, 0.3, 0.5, 0.2, 0.3]]   # Seq 2                          │
│                                                                                   │
│   mask   = [[1,   1,   1,   1,   0,   0  ],   # Seq 1 (4 real tokens)          │
│             [1,   1,   1,   1,   1,   1  ]]   # Seq 2 (6 real tokens)          │
│                                                                                   │
│   masked = [[0.5, 0.3, 0.4, 0.2, 0.0, 0.0],                                     │
│             [0.6, 0.4, 0.3, 0.5, 0.2, 0.3]]                                     │
│                                                                                   │
│   Per-example loss:                                                               │
│   Seq 1: (0.5+0.3+0.4+0.2) / 4 = 0.35                                          │
│   Seq 2: (0.6+0.4+0.3+0.5+0.2+0.3) / 6 = 0.383                                 │
│                                                                                   │
│   Batch loss: (0.35 + 0.383) / 2 = 0.367                                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Loss Calculation in model.py

```python
# model.py: _calc_final_dist and loss calculation (Lines 350-420)

def _calc_final_dist(self, vocab_dists, attn_dists):
    """
    Calculate final distributions for loss computation.
    
    Args:
        vocab_dists: List of vocab distributions [batch, vocab_size]
        attn_dists: List of attention distributions [batch, enc_len]
    
    Returns:
        final_dists: List of final distributions [batch, extended_vocab_size]
    """
    with tf.variable_scope('final_distribution'):
        # Extended vocab size = vocab + max article OOVs
        extended_vsize = self._vocab.size() + self._max_art_oovs
        
        final_dists = []
        
        for p_gen, vocab_dist, attn_dist in zip(
            self._p_gens, vocab_dists, attn_dists
        ):
            # Weighted vocab distribution
            vocab_dist_weighted = p_gen * vocab_dist  # [batch, vocab_size]
            
            # Extend vocab dist with zeros for OOVs
            extra_zeros = tf.zeros([self._hps.batch_size, self._max_art_oovs])
            vocab_dist_extended = tf.concat([vocab_dist_weighted, extra_zeros], 1)
            # Shape: [batch, extended_vsize]
            
            # Weighted copy distribution
            attn_dist_weighted = (1 - p_gen) * attn_dist  # [batch, enc_len]
            
            # Project attention to extended vocabulary using enc_batch_extend_vocab
            # This maps each attention weight to its corresponding word ID
            batch_nums = tf.range(0, self._hps.batch_size)
            batch_nums = tf.expand_dims(batch_nums, 1)  # [batch, 1]
            attn_len = tf.shape(attn_dist)[1]
            batch_nums = tf.tile(batch_nums, [1, attn_len])  # [batch, enc_len]
            
            indices = tf.stack([batch_nums, self._enc_batch_extend_vocab], axis=2)
            # Shape: [batch, enc_len, 2]
            
            # Scatter attention weights to extended vocabulary
            shape = [self._hps.batch_size, extended_vsize]
            attn_dist_projected = tf.scatter_nd(indices, attn_dist_weighted, shape)
            # Shape: [batch, extended_vsize]
            
            # Combine vocab and copy distributions
            final_dist = vocab_dist_extended + attn_dist_projected
            
            final_dists.append(final_dist)
        
        return final_dists


def _add_seq2seq_loss(self):
    """Add sequence-to-sequence loss."""
    
    with tf.variable_scope('loss'):
        if self._hps.pointer_gen:
            # Use final distribution (vocab + copy)
            loss_per_step = []
            for dec_step, dist in enumerate(self._final_dists):
                # Get target for this step
                targets = self._target_batch[:, dec_step]  # [batch]
                
                # Get indices for gathering
                indices = tf.stack([
                    tf.range(self._hps.batch_size),
                    targets
                ], axis=1)  # [batch, 2]
                
                # Get probability of correct target
                gold_probs = tf.gather_nd(dist, indices)  # [batch]
                
                # Clip for numerical stability
                gold_probs = tf.clip_by_value(gold_probs, 1e-10, 1.0)
                
                # Negative log probability
                losses = -tf.log(gold_probs)  # [batch]
                loss_per_step.append(losses)
            
            # Apply mask and average
            self._loss = _mask_and_avg(loss_per_step, self._dec_padding_mask)
        else:
            # Standard cross-entropy loss (no pointer)
            # ... similar but on vocab distribution only
            pass
```

### Coverage Loss Implementation

```python
# model.py: _coverage_loss function (Lines 463-480)

def _coverage_loss(self, attn_dists, padding_mask):
    """
    Calculate coverage loss.
    
    Args:
        attn_dists: List of attention distributions [batch, enc_len]
        padding_mask: Decoder padding mask [batch, dec_len]
    
    Returns:
        Scalar coverage loss
    """
    coverage = tf.zeros_like(attn_dists[0])  # [batch, enc_len]
    
    covlosses = []
    for a in attn_dists:
        # Coverage loss: sum of min(attention, coverage)
        covloss = tf.reduce_sum(tf.minimum(a, coverage), axis=1)  # [batch]
        covlosses.append(covloss)
        
        # Update coverage
        coverage += a
    
    # Apply mask and average
    coverage_loss = _mask_and_avg(covlosses, padding_mask)
    
    return coverage_loss
```

---

## Worked Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE LOSS EXAMPLE                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   SETUP:                                                                          │
│   ──────                                                                          │
│   • Source: "Elon Musk announced Tesla"                                          │
│   • Target: "Musk announced Tesla" (with STOP)                                   │
│   • vocab_size = 1000, "Elon", "Musk", "Tesla" are OOV                          │
│   • Extended vocab: Elon=1000, Musk=1001, Tesla=1002                            │
│   • announced=500 (in vocab)                                                     │
│                                                                                   │
│   Target IDs: [1001, 500, 1002, STOP_ID]                                        │
│                Musk  announced Tesla STOP                                        │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: Predict "Musk" (target ID=1001)                                        │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Model outputs:                                                                  │
│   • p_gen = 0.2 (wants to copy)                                                 │
│   • P_vocab: [0.01, 0.02, ...] (1000 probs, none high for Musk)                │
│   • α = [0.1, 0.8, 0.05, 0.05] (attention on "Elon Musk announced Tesla")       │
│           ↑    ↑                                                                 │
│          Elon Musk (high attention!)                                            │
│                                                                                   │
│   Final distribution:                                                             │
│   P(Musk) = 0.2 × P_vocab(Musk) + 0.8 × α(position of Musk)                    │
│           = 0.2 × 0 + 0.8 × 0.8                                                 │
│           = 0.64                                                                 │
│                                                                                   │
│   Loss step 1: -log(0.64) = 0.446                                               │
│                                                                                   │
│   Coverage:                                                                       │
│   c₁ = [0.0, 0.0, 0.0, 0.0] (initial)                                          │
│   covloss₁ = Σ min(α₁, c₁) = 0                                                 │
│   c₂ = c₁ + α₁ = [0.1, 0.8, 0.05, 0.05]                                        │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: Predict "announced" (target ID=500)                                    │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Model outputs:                                                                  │
│   • p_gen = 0.6 (wants to generate)                                             │
│   • P_vocab: [..., P_vocab[500]=0.7, ...] (high for "announced")               │
│   • α = [0.05, 0.1, 0.75, 0.1]                                                  │
│                     ↑                                                            │
│                 "announced"                                                      │
│                                                                                   │
│   Final distribution:                                                             │
│   P(announced) = 0.6 × P_vocab(announced) + 0.4 × α(announced_pos)             │
│                = 0.6 × 0.7 + 0.4 × 0.75                                         │
│                = 0.42 + 0.30 = 0.72                                             │
│                                                                                   │
│   Loss step 2: -log(0.72) = 0.329                                               │
│                                                                                   │
│   Coverage:                                                                       │
│   c₂ = [0.1, 0.8, 0.05, 0.05]                                                  │
│   covloss₂ = min(0.05, 0.1) + min(0.1, 0.8) + min(0.75, 0.05) + min(0.1, 0.05)│
│            = 0.05 + 0.1 + 0.05 + 0.05 = 0.25                                   │
│   c₃ = [0.15, 0.9, 0.8, 0.15]                                                  │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: Predict "Tesla" (target ID=1002)                                       │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Model outputs:                                                                  │
│   • p_gen = 0.3 (wants to copy)                                                 │
│   • P_vocab: (0 for Tesla, not in vocab)                                        │
│   • α = [0.05, 0.05, 0.1, 0.8]                                                  │
│                           ↑                                                       │
│                       "Tesla"                                                    │
│                                                                                   │
│   Final distribution:                                                             │
│   P(Tesla) = 0.3 × 0 + 0.7 × 0.8 = 0.56                                        │
│                                                                                   │
│   Loss step 3: -log(0.56) = 0.580                                               │
│                                                                                   │
│   Coverage:                                                                       │
│   c₃ = [0.15, 0.9, 0.8, 0.15]                                                  │
│   covloss₃ = 0.05 + 0.05 + 0.1 + 0.15 = 0.35                                   │
│   c₄ = [0.2, 0.95, 0.9, 0.95]                                                  │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   TOTAL LOSS:                                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   L_NLL = (0.446 + 0.329 + 0.580) / 3 = 0.452                                   │
│                                                                                   │
│   L_coverage = (0 + 0.25 + 0.35) / 3 = 0.20                                     │
│                                                                                   │
│   L_total = 0.452 + 1.0 × 0.20 = 0.652                                          │
│                                                                                   │
│   This is the loss for ONE example. For a batch, average over all examples.     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

Loss functions key points:

1. **NLL Loss**: -log P(correct word) summed over sequence
2. **Pointer-Generator**: Loss computed on combined final distribution
3. **Coverage Loss**: Penalizes repeated attention (min of attention and coverage)
4. **Total Loss**: L_NLL + λ × L_coverage
5. **Masking**: Zero out loss on PAD tokens, average over real tokens
6. **Extended Vocab**: OOV targets use IDs beyond vocab_size

---

*Next: [12_model_py_walkthrough.md](12_model_py_walkthrough.md) - Line-by-Line model.py Analysis*
