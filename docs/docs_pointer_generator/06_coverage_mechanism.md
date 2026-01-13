# Coverage Mechanism Deep Dive

## Table of Contents
1. [The Repetition Problem](#the-repetition-problem)
2. [Coverage Vector Concept](#coverage-vector-concept)
3. [Coverage in Attention Calculation](#coverage-in-attention-calculation)
4. [Coverage Loss](#coverage-loss)
5. [Implementation Details](#implementation-details)
6. [Training Strategy](#training-strategy)
7. [Worked Examples](#worked-examples)

---

## The Repetition Problem

### Why Models Repeat

Without coverage, sequence-to-sequence models often generate repetitive text:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      THE REPETITION PROBLEM                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Example without coverage:                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   Source Article:                                                                 │
│   "Germany emerged as the winners of the 2014 FIFA World Cup after               │
│    defeating Argentina 1-0 in the final at the Maracanã Stadium.                 │
│    Mario Götze scored the winning goal in extra time."                           │
│                                                                                   │
│   Generated Summary (BAD):                                                        │
│   "Germany won the World Cup. Germany won the World Cup. Germany won             │
│    the World Cup after defeating Argentina."                                      │
│                                          ↑ ↑ ↑                                    │
│                                    REPEATED 3 TIMES!                              │
│                                                                                   │
│                                                                                   │
│   WHY DOES THIS HAPPEN?                                                           │
│   ─────────────────────                                                           │
│                                                                                   │
│   1. Attention tends to focus on the same "important" regions                    │
│                                                                                   │
│   2. Without memory, the model doesn't know what it already covered             │
│                                                                                   │
│   3. High-probability outputs create feedback loops:                             │
│                                                                                   │
│      Generated "World Cup" → Strong decoder state for sports                     │
│            ↓                                                                      │
│      Attention focuses on "World Cup" in source again                            │
│            ↓                                                                      │
│      High probability for "World Cup" again                                      │
│            ↓                                                                      │
│      Loop continues!                                                              │
│                                                                                   │
│                                                                                   │
│   VISUAL EXAMPLE OF ATTENTION WITHOUT COVERAGE:                                   │
│   ─────────────────────────────────────────────                                   │
│                                                                                   │
│   Source:  Germany  won  the  World  Cup  defeating  Argentina                   │
│                                                                                   │
│   Step 1:  ████▌   ██   █    ████   ███    █         █                          │
│            Focus on "Germany won World Cup"                                      │
│                                                                                   │
│   Step 2:  ████    ███  █    █████  ████   █         █                          │
│            Still focusing on "Germany World Cup"!                                │
│                                                                                   │
│   Step 3:  ███▌    ██   █    █████  ████   █         █                          │
│            STILL the same region!                                                │
│                                                                                   │
│   Step 4:  ███     ██   █    ████   ████   █         █                          │
│            Model is "stuck" on the same words                                    │
│                                                                                   │
│   Output:  "Germany World Cup. Germany World Cup. Germany..."                    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coverage Vector Concept

### What is Coverage?

The coverage vector tracks how much attention each source position has received so far:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        COVERAGE VECTOR CONCEPT                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   DEFINITION:                                                                     │
│   ───────────                                                                     │
│                                                                                   │
│   c_t = Σ_{t'=0}^{t-1} α_{t'}                                                   │
│                                                                                   │
│   Where α_{t'} is the attention distribution at decoder step t'.                 │
│                                                                                   │
│   c_t[i] = Total attention received by source position i up to step t           │
│                                                                                   │
│                                                                                   │
│   INTUITION:                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   Coverage is like a "visited" counter for each source word:                     │
│                                                                                   │
│   • c_t[i] ≈ 0: Position i has NOT been attended to                             │
│   • c_t[i] ≈ 1: Position i has been attended to roughly once                    │
│   • c_t[i] > 1: Position i has been attended to MULTIPLE times                  │
│                                                                                   │
│                                                                                   │
│   VISUAL EXAMPLE:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Source: "Germany beat Argentina in the final"                                  │
│           [0]     [1]    [2]      [3] [4]  [5]                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Step 0: Initialize                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Coverage c₀ = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                                  │
│   (No attention has been paid yet)                                               │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Step 1: Generate "Germany"                                                      │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Attention α₁ = [0.85, 0.05, 0.03, 0.02, 0.02, 0.03]                           │
│                   ↑                                                               │
│                Heavy focus on "Germany"                                          │
│                                                                                   │
│   Coverage c₁ = c₀ + α₁ = [0.85, 0.05, 0.03, 0.02, 0.02, 0.03]                 │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Step 2: Generate "defeated"                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Attention α₂ = [0.10, 0.70, 0.10, 0.03, 0.02, 0.05]                           │
│                         ↑                                                         │
│                  Focus on "beat"                                                 │
│                                                                                   │
│   Coverage c₂ = c₁ + α₂ = [0.95, 0.75, 0.13, 0.05, 0.04, 0.08]                 │
│                                                                                   │
│   Now "Germany" and "beat" have high coverage.                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Step 3: Generate "Argentina"                                                    │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Attention α₃ = [0.05, 0.05, 0.80, 0.03, 0.02, 0.05]                           │
│                               ↑                                                   │
│                    Focus on "Argentina"                                          │
│                                                                                   │
│   Coverage c₃ = c₂ + α₃ = [1.00, 0.80, 0.93, 0.08, 0.06, 0.13]                 │
│                                                                                   │
│   Now "Germany", "beat", "Argentina" all have high coverage.                     │
│   The model "knows" these have been addressed.                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Step 4: What's next?                                                           │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Coverage c₃ = [1.00, 0.80, 0.93, 0.08, 0.06, 0.13]                            │
│                  ↑     ↑     ↑     ↓     ↓     ↓                                 │
│                HIGH  HIGH  HIGH   LOW   LOW   LOW                                │
│                                                                                   │
│   Words "in", "the", "final" have LOW coverage → should attend there!           │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coverage in Attention Calculation

### Modified Attention Score

The coverage vector is incorporated into the attention calculation:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   COVERAGE IN ATTENTION CALCULATION                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WITHOUT COVERAGE:                                                               │
│   ─────────────────                                                               │
│                                                                                   │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + b_attn)                             │
│                                                                                   │
│                                                                                   │
│   WITH COVERAGE:                                                                  │
│   ──────────────                                                                  │
│                                                                                   │
│   e_ti = v^T · tanh(W_h · h_i + W_s · s_t + w_c · c_t[i] + b_attn)              │
│                                               ↑                                   │
│                                    Coverage feature!                             │
│                                                                                   │
│   Where:                                                                          │
│   • w_c ∈ ℝ^attn_size : Learnable coverage weight vector                        │
│   • c_t[i] : Coverage for position i at step t                                  │
│                                                                                   │
│                                                                                   │
│   HOW DOES THIS HELP?                                                             │
│   ───────────────────                                                             │
│                                                                                   │
│   The model can learn:                                                            │
│                                                                                   │
│   If w_c is negative (learned to be negative typically):                         │
│   • High c_t[i] (already attended) → Lower e_ti → Lower attention               │
│   • Low c_t[i] (not yet attended) → Higher e_ti → Higher attention              │
│                                                                                   │
│                                                                                   │
│   EXAMPLE:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   Source: "Germany beat Argentina"                                               │
│   Coverage at step 3: c₃ = [1.0, 0.8, 0.2]                                      │
│                             ↑    ↑    ↓                                          │
│                           High High  Low                                         │
│                                                                                   │
│   Attention scores (before softmax):                                             │
│                                                                                   │
│   Without coverage:                                                               │
│   e_Germany   = tanh(encoder_feature + decoder_feature) = 2.5                   │
│   e_beat      = tanh(encoder_feature + decoder_feature) = 1.8                   │
│   e_Argentina = tanh(encoder_feature + decoder_feature) = 2.8                   │
│                                                                                   │
│   With coverage (w_c ≈ -2.0):                                                    │
│   e_Germany   = tanh(2.5 + (-2.0) × 1.0) = tanh(0.5)  ≈ 0.46  ↓ REDUCED       │
│   e_beat      = tanh(1.8 + (-2.0) × 0.8) = tanh(0.2)  ≈ 0.20  ↓ REDUCED       │
│   e_Argentina = tanh(2.8 + (-2.0) × 0.2) = tanh(2.4)  ≈ 0.98  ↑ PRESERVED     │
│                                                                                   │
│   Softmax attention:                                                              │
│   Without coverage: [0.35, 0.18, 0.47] → Model might attend to Germany again    │
│   With coverage:    [0.25, 0.15, 0.60] → Strong preference for Argentina!       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Coverage Loss

### Penalizing Repeated Attention

The coverage loss explicitly penalizes the model for attending to already-attended positions:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          COVERAGE LOSS                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FORMULA:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   covloss_t = Σ_i min(α_ti, c_t[i])                                             │
│                                                                                   │
│   Total coverage loss = (1/T) Σ_t covloss_t                                     │
│                                                                                   │
│                                                                                   │
│   WHY min()?                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   The min function captures "overlap" between current attention and coverage:   │
│                                                                                   │
│   Case 1: c_t[i] = 0.9 (already attended), α_ti = 0.8 (attending again)         │
│           min(0.8, 0.9) = 0.8 → HIGH PENALTY                                    │
│           "You're re-attending to something you've already covered!"             │
│                                                                                   │
│   Case 2: c_t[i] = 0.1 (barely attended), α_ti = 0.8 (attending now)            │
│           min(0.8, 0.1) = 0.1 → LOW PENALTY                                     │
│           "This is new territory, no penalty."                                   │
│                                                                                   │
│   Case 3: c_t[i] = 0.9 (already attended), α_ti = 0.1 (not attending now)       │
│           min(0.1, 0.9) = 0.1 → LOW PENALTY                                     │
│           "You attended before but not now, that's fine."                        │
│                                                                                   │
│   Case 4: c_t[i] = 0.1 (barely attended), α_ti = 0.1 (not attending now)        │
│           min(0.1, 0.1) = 0.1 → LOW PENALTY                                     │
│           "Low coverage, low attention, no problem."                             │
│                                                                                   │
│                                                                                   │
│   VISUAL REPRESENTATION:                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│                         min(α_ti, c_t[i])                                        │
│                                                                                   │
│   α_ti                                                                            │
│    ↑                                                                              │
│   1│           ████████████                                                      │
│    │          █            █                                                      │
│    │         █              █        Loss = overlap area                         │
│    │        █                █       (when both are high)                        │
│    │       █                  █                                                   │
│    │      █     ░░░░░░░░      █                                                  │
│    │     █    ░░░░░░░░░░░      █                                                 │
│    │    █   ░░░░░░░░░░░░░░      █                                                │
│    │   █  ░░░░░░░░░░░░░░░░░      █                                               │
│    └───█░░░░░░░░░░░░░░░░░░░░───────▶ c_t[i]                                     │
│       0                            1                                              │
│                                                                                   │
│   ░░░ = Loss region (min of both values)                                         │
│   Maximum loss occurs when both α and c are high (repeated attention)            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Total Loss with Coverage

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    TOTAL LOSS WITH COVERAGE                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   L_total = L_NLL + λ × L_coverage                                               │
│                                                                                   │
│   Where:                                                                          │
│   • L_NLL = Negative log-likelihood (main generation loss)                       │
│   • L_coverage = Coverage loss (repetition penalty)                              │
│   • λ = cov_loss_wt (hyperparameter, typically 1.0)                             │
│                                                                                   │
│                                                                                   │
│   TRAINING DYNAMICS:                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                                                                         │    │
│   │   Early training:                                                       │    │
│   │   • L_NLL dominates                                                    │    │
│   │   • Model learns basic word prediction                                 │    │
│   │   • May develop repetition patterns                                    │    │
│   │                                                                         │    │
│   │   After adding coverage:                                                │    │
│   │   • L_coverage starts contributing                                     │    │
│   │   • Model gets penalized for repetition                                │    │
│   │   • Learns to diversify attention                                      │    │
│   │                                                                         │    │
│   │   Converged model:                                                      │    │
│   │   • Balance between accurate prediction and diversity                  │    │
│   │   • L_coverage typically small (model learned to not repeat)          │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│                                                                                   │
│   EXAMPLE LOSS VALUES:                                                            │
│   ─────────────────────                                                           │
│                                                                                   │
│   Without coverage (bad summary with repetition):                                │
│   L_NLL = 2.5                                                                    │
│   L_total = 2.5                                                                  │
│                                                                                   │
│   With coverage (same bad summary):                                               │
│   L_NLL = 2.5                                                                    │
│   L_coverage = 1.8  (high due to repetition!)                                   │
│   L_total = 2.5 + 1.0 × 1.8 = 4.3  ← Much higher! Model is penalized.          │
│                                                                                   │
│   With coverage (good diverse summary):                                           │
│   L_NLL = 2.6  (slightly higher, but diverse)                                   │
│   L_coverage = 0.3  (low, good coverage pattern)                                │
│   L_total = 2.6 + 1.0 × 0.3 = 2.9  ← Lower total! Model prefers this.          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Details

### Coverage in attention_decoder.py

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    COVERAGE IMPLEMENTATION                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   attention_decoder.py: Key code sections                                        │
│   ───────────────────────────────────────                                         │
│                                                                                   │
│   1. COVERAGE WEIGHT INITIALIZATION (Lines 71-73):                               │
│   ─────────────────────────────────────────────────                               │
│                                                                                   │
│   if use_coverage:                                                               │
│       with variable_scope("coverage"):                                           │
│           w_c = get_variable("w_c", [1, 1, 1, attention_vec_size])              │
│                                                                                   │
│   Shape explained:                                                                │
│   [1, 1, 1, attention_vec_size] = [1, 1, 1, 512]                                │
│   This allows using conv2d for efficient computation.                            │
│                                                                                   │
│                                                                                   │
│   2. COVERAGE FEATURE COMPUTATION (Lines 103-108):                               │
│   ─────────────────────────────────────────────────                               │
│                                                                                   │
│   if use_coverage and coverage is not None:                                      │
│       # Multiply coverage vector by w_c                                          │
│       coverage_features = conv2d(coverage, w_c, [1,1,1,1], "SAME")              │
│       # Shape: [batch, enc_len, 1, attn_size]                                   │
│                                                                                   │
│       # Add coverage to attention score                                          │
│       e = reduce_sum(                                                            │
│           v * tanh(encoder_features + decoder_features + coverage_features),    │
│           [2, 3]                                                                 │
│       )                                                                          │
│                                                                                   │
│                                                                                   │
│   3. COVERAGE UPDATE (Lines 113-114):                                            │
│   ─────────────────────────────────────                                           │
│                                                                                   │
│   # Update coverage vector after attention                                        │
│   coverage += reshape(attn_dist, [batch_size, -1, 1, 1])                        │
│                                                                                   │
│   This accumulates attention: c_t = c_{t-1} + α_t                               │
│                                                                                   │
│                                                                                   │
│   4. COVERAGE INITIALIZATION (Lines 122-123):                                    │
│   ────────────────────────────────────────────                                    │
│                                                                                   │
│   if use_coverage:  # first step                                                 │
│       coverage = expand_dims(expand_dims(attn_dist, 2), 3)                      │
│                                                                                   │
│   For the first step, coverage starts with the first attention distribution.    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Coverage Loss in model.py

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   COVERAGE LOSS IMPLEMENTATION                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   model.py: _coverage_loss() function (Lines 463-480)                            │
│   ─────────────────────────────────────────────────────                           │
│                                                                                   │
│   def _coverage_loss(attn_dists, padding_mask):                                  │
│       """Calculates the coverage loss from the attention distributions."""       │
│                                                                                   │
│       # Initialize coverage to zeros                                             │
│       coverage = zeros_like(attn_dists[0])                                       │
│       # Shape: [batch_size, attn_length]                                         │
│                                                                                   │
│       covlosses = []  # Coverage loss per decoder timestep                       │
│                                                                                   │
│       for a in attn_dists:  # For each decoder step                             │
│           # Calculate coverage loss for this step                                │
│           # covloss = Σ_i min(a[i], coverage[i])                                │
│           covloss = reduce_sum(minimum(a, coverage), [1])                       │
│           # Shape: [batch_size]                                                  │
│                                                                                   │
│           covlosses.append(covloss)                                              │
│                                                                                   │
│           # Update coverage: c_t = c_{t-1} + a_t                                │
│           coverage += a                                                          │
│                                                                                   │
│       # Average over steps and batch (with masking)                              │
│       coverage_loss = _mask_and_avg(covlosses, padding_mask)                    │
│                                                                                   │
│       return coverage_loss                                                       │
│                                                                                   │
│                                                                                   │
│   STEP-BY-STEP EXECUTION:                                                         │
│   ───────────────────────                                                         │
│                                                                                   │
│   attn_dists = [α₁, α₂, α₃, α₄]  # 4 decoder steps                             │
│                                                                                   │
│   Step 1: coverage = [0, 0, 0, 0, 0]                                            │
│           α₁ = [0.8, 0.1, 0.05, 0.03, 0.02]                                     │
│           covloss₁ = Σ min(α₁, coverage) = 0  (all zeros)                       │
│           coverage = [0.8, 0.1, 0.05, 0.03, 0.02]                               │
│                                                                                   │
│   Step 2: coverage = [0.8, 0.1, 0.05, 0.03, 0.02]                               │
│           α₂ = [0.1, 0.7, 0.1, 0.05, 0.05]                                      │
│           min(α₂, cov) = [0.1, 0.1, 0.05, 0.03, 0.02]                           │
│           covloss₂ = 0.1 + 0.1 + 0.05 + 0.03 + 0.02 = 0.30                     │
│           coverage = [0.9, 0.8, 0.15, 0.08, 0.07]                               │
│                                                                                   │
│   Step 3: coverage = [0.9, 0.8, 0.15, 0.08, 0.07]                               │
│           α₃ = [0.05, 0.05, 0.8, 0.05, 0.05]                                    │
│           min(α₃, cov) = [0.05, 0.05, 0.15, 0.05, 0.05]                         │
│           covloss₃ = 0.35                                                        │
│           coverage = [0.95, 0.85, 0.95, 0.13, 0.12]                             │
│                                                                                   │
│   Step 4: coverage = [0.95, 0.85, 0.95, 0.13, 0.12]                             │
│           α₄ = [0.03, 0.02, 0.05, 0.45, 0.45]                                   │
│           min(α₄, cov) = [0.03, 0.02, 0.05, 0.13, 0.12]                         │
│           covloss₄ = 0.35                                                        │
│                                                                                   │
│   Total coverage loss = mean([0, 0.30, 0.35, 0.35]) = 0.25                      │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training Strategy

### Two-Phase Training

The recommended approach is to train in two phases:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    TWO-PHASE TRAINING STRATEGY                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PHASE 1: Train WITHOUT Coverage                                                 │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   Command:                                                                        │
│   python run_summarization.py --coverage=False --pointer_gen=True                │
│                                                                                   │
│   What happens:                                                                   │
│   • Model learns basic seq2seq with pointer mechanism                            │
│   • Learns word embeddings, attention patterns                                   │
│   • Learns when to copy vs. generate (p_gen)                                     │
│   • May develop some repetition (that's okay for now)                            │
│                                                                                   │
│   Duration: Until convergence (~100K-200K steps)                                 │
│                                                                                   │
│                                                                                   │
│   PHASE 2: Fine-tune WITH Coverage                                                │
│   ────────────────────────────────                                                │
│                                                                                   │
│   Step 1: Convert checkpoint to coverage-ready model                             │
│   python run_summarization.py --mode=train --convert_to_coverage_model=True \   │
│                               --coverage=True                                    │
│                                                                                   │
│   This adds coverage variables (w_c) initialized to zeros.                       │
│                                                                                   │
│   Step 2: Continue training with coverage                                         │
│   python run_summarization.py --coverage=True --cov_loss_wt=1.0                 │
│                                                                                   │
│   What happens:                                                                   │
│   • Coverage mechanism starts influencing attention                              │
│   • Model learns to diversify attention                                          │
│   • Repetition decreases                                                         │
│                                                                                   │
│   Duration: Short phase (~20K-50K steps)                                         │
│                                                                                   │
│                                                                                   │
│   WHY TWO PHASES?                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   1. Training with coverage from scratch is harder:                              │
│      • Model must learn attention AND coverage simultaneously                    │
│      • Coverage loss can interfere with learning good attention                  │
│                                                                                   │
│   2. Pre-trained model has better foundation:                                    │
│      • Already knows how to attend to relevant words                            │
│      • Just needs to learn NOT to repeat                                        │
│      • Faster convergence with coverage                                          │
│                                                                                   │
│   3. Empirically better results:                                                 │
│      • Paper reports this strategy works better                                  │
│      • More stable training                                                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Coverage Hyperparameters

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    COVERAGE HYPERPARAMETERS                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FLAG: --coverage                                                                │
│   ────────────────                                                                │
│   Type: Boolean                                                                   │
│   Default: False                                                                  │
│   Description: Enable/disable coverage mechanism                                  │
│                                                                                   │
│                                                                                   │
│   FLAG: --cov_loss_wt                                                            │
│   ──────────────────                                                              │
│   Type: Float                                                                     │
│   Default: 1.0                                                                    │
│   Description: Weight of coverage loss (λ in L_total = L_NLL + λ × L_cov)       │
│                                                                                   │
│   Tuning guidelines:                                                              │
│   • λ = 0: No coverage penalty (equivalent to --coverage=False)                  │
│   • λ = 0.5: Mild coverage penalty                                               │
│   • λ = 1.0: Standard coverage penalty (recommended)                             │
│   • λ = 2.0: Strong coverage penalty (might hurt fluency)                        │
│                                                                                   │
│                                                                                   │
│   FLAG: --convert_to_coverage_model                                              │
│   ──────────────────────────────────                                              │
│   Type: Boolean                                                                   │
│   Default: False                                                                  │
│   Description: Convert a non-coverage checkpoint to coverage-ready               │
│                                                                                   │
│   What it does:                                                                   │
│   1. Loads existing checkpoint (without w_c variable)                            │
│   2. Initializes w_c to zeros                                                    │
│   3. Saves new checkpoint with "_cov_init" suffix                               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Worked Examples

### Example: Coverage Preventing Repetition

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│             EXAMPLE: COVERAGE PREVENTING REPETITION                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Source: "Germany beat Argentina. Germany is the champion."                     │
│   Target: "Germany defeated Argentina to become champion."                       │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: Generate "Germany"                                                      │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Source:     Germany  beat  Argentina  .  Germany  is  the  champion  .        │
│   Position:      0       1       2      3     4      5   6      7      8        │
│                                                                                   │
│   Coverage:   [0.0,   0.0,    0.0,    0.0,  0.0,   0.0, 0.0,   0.0,   0.0]      │
│                                                                                   │
│   Attention:  [0.50,  0.05,   0.05,   0.02, 0.30,  0.02, 0.02, 0.02,  0.02]     │
│                ↑                             ↑                                    │
│           "Germany" at pos 0 and 4 both get attention                           │
│                                                                                   │
│   New coverage: [0.50, 0.05, 0.05, 0.02, 0.30, 0.02, 0.02, 0.02, 0.02]          │
│                                                                                   │
│   Coverage loss: min(α, c) summed = 0 (c was all zeros)                         │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: Generate "defeated"                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Coverage:   [0.50,  0.05,   0.05,   0.02, 0.30,  0.02, 0.02,  0.02,  0.02]    │
│                 ↑                             ↑                                   │
│              Already attended!                                                    │
│                                                                                   │
│   Without coverage, attention might focus on "Germany" again!                    │
│                                                                                   │
│   WITH coverage:                                                                  │
│   • Position 0 (Germany): High coverage → attention discouraged                 │
│   • Position 4 (Germany): High coverage → attention discouraged                 │
│   • Position 1 (beat): Low coverage → attention ENCOURAGED                      │
│                                                                                   │
│   Attention:  [0.10,  0.65,   0.10,   0.05, 0.05,  0.02, 0.01,  0.01,  0.01]    │
│                       ↑                                                          │
│                Focus shifts to "beat"!                                           │
│                                                                                   │
│   New coverage: [0.60, 0.70, 0.15, 0.07, 0.35, 0.04, 0.03, 0.03, 0.03]          │
│                                                                                   │
│   Coverage loss: min([0.10, 0.65, ...], [0.50, 0.05, ...])                      │
│                = min(0.10, 0.50) + min(0.65, 0.05) + ...                        │
│                = 0.10 + 0.05 + 0.05 + ... ≈ 0.25 (acceptable)                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: Generate "Argentina"                                                    │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Coverage:   [0.60,  0.70,   0.15,   0.07, 0.35,  0.04, 0.03,  0.03,  0.03]    │
│                 ↑      ↑                    ↑                                    │
│              Covered  Covered             Partially                              │
│                                                                                   │
│   Model should now attend to "Argentina" (position 2)                            │
│                                                                                   │
│   Attention:  [0.05,  0.05,   0.75,   0.05, 0.05,  0.02, 0.01,  0.01,  0.01]    │
│                               ↑                                                  │
│                      Focus on Argentina!                                         │
│                                                                                   │
│   New coverage: [0.65, 0.75, 0.90, 0.12, 0.40, 0.06, 0.04, 0.04, 0.04]          │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 4: Generate "champion"                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Coverage:   [0.65,  0.75,   0.90,   0.12, 0.40,  0.06, 0.04,  0.04,  0.04]    │
│                 ↑      ↑       ↑                                ↓                │
│              High coverage already         Low coverage - attend here!           │
│                                                                                   │
│   WITHOUT coverage, model might attend to "Germany" again!                       │
│   ("Germany is the champion" is in source)                                       │
│                                                                                   │
│   WITH coverage:                                                                  │
│   • Positions 0,1,2 have high coverage → discouraged                            │
│   • Position 7 "champion" has low coverage → ENCOURAGED                         │
│                                                                                   │
│   Attention:  [0.05,  0.03,   0.05,   0.02, 0.08,  0.02, 0.05,  0.68,  0.02]    │
│                                                              ↑                   │
│                                                   Focus on "champion"!           │
│                                                                                   │
│   OUTPUT: "Germany defeated Argentina champion" → Close to target!              │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   CONTRAST: WITHOUT COVERAGE                                                      │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Step 1: Attend to "Germany" → Output "Germany"                                │
│   Step 2: Attend to "Germany" again! → Output "Germany"                         │
│   Step 3: Attend to "Germany" again! → Output "Germany"                         │
│   ...                                                                             │
│                                                                                   │
│   OUTPUT: "Germany Germany Germany..."  ← BAD!                                  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

The Coverage Mechanism:

1. **Tracks attention history** via the coverage vector c_t
2. **Influences future attention** by adding coverage features to attention scores
3. **Penalizes repetition** through coverage loss
4. **Requires two-phase training** for best results

Key formulas:
- **Coverage update**: c_t = c_{t-1} + α_{t-1}
- **Modified attention**: e_ti = v^T · tanh(W_h·h_i + W_s·s_t + w_c·c_t[i])
- **Coverage loss**: L_cov = Σ_t Σ_i min(α_ti, c_t[i])
- **Total loss**: L_total = L_NLL + λ × L_cov

---

*Next: [07_data_pipeline.md](07_data_pipeline.md) - Data Loading and Batch Processing*
