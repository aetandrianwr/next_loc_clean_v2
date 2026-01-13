# Complete Example Walkthrough

## Table of Contents
1. [Example Setup: Alice's Day Trip](#example-setup-alices-day-trip)
2. [Original Model Walkthrough](#original-model-walkthrough)
3. [Proposed Model Walkthrough](#proposed-model-walkthrough)
4. [Step-by-Step Numerical Comparison](#step-by-step-numerical-comparison)
5. [Attention Visualization](#attention-visualization)
6. [Final Prediction Comparison](#final-prediction-comparison)

---

## Example Setup: Alice's Day Trip

We'll use a consistent example throughout this document to demonstrate both models.

### The Scenario

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ALICE'S DAY TRIP                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  User: Alice (user_id = 42)                                                 │
│  Day: Monday                                                                 │
│                                                                              │
│  Timeline:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  08:00  ┌─────────┐   30 min     "Leaving home"                      │   │
│  │   ───►  │  HOME   │   ─────►                                         │   │
│  │         │  (101)  │                                                   │   │
│  │         └─────────┘                                                   │   │
│  │                                                                       │   │
│  │  08:30  ┌─────────┐   15 min     "Quick coffee stop"                 │   │
│  │   ───►  │ COFFEE  │   ─────►                                         │   │
│  │         │  (205)  │                                                   │   │
│  │         └─────────┘                                                   │   │
│  │                                                                       │   │
│  │  09:00  ┌─────────┐   480 min    "Working at office"                 │   │
│  │   ───►  │ OFFICE  │   ─────►                                         │   │
│  │         │  (150)  │                                                   │   │
│  │         └─────────┘                                                   │   │
│  │                                                                       │   │
│  │  17:00  ┌──────────┐  60 min     "Lunch break"                       │   │
│  │   ───►  │RESTAURANT│  ─────►                                         │   │
│  │         │  (312)   │                                                  │   │
│  │         └──────────┘                                                  │   │
│  │                                                                       │   │
│  │  18:00  ┌─────────┐   120 min    "Back to office"                    │   │
│  │   ───►  │ OFFICE  │   ─────►                                         │   │
│  │         │  (150)  │                                                   │   │
│  │         └─────────┘                                                   │   │
│  │                                                                       │   │
│  │  20:00  ┌─────────┐              "What's next?"                       │   │
│  │   ───►  │   ???   │                                                   │   │
│  │         └─────────┘                                                   │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Target: Gym (location_id = 89)                                             │
│                                                                              │
│  Alice usually goes to the gym on Monday evenings after work.               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Input Data

```python
# Input sequence (for proposed model)
x = [101, 205, 150, 312, 150]  # Location IDs
x_dict = {
    'user_id': [42, 42, 42, 42, 42],
    'time_idx': [8, 8, 9, 17, 18],      # Hour of day
    'weekday': [0, 0, 0, 0, 0],          # Monday = 0
    'duration': [2, 1, 15, 4, 8],        # Duration bins
    'recency': [5, 4, 3, 2, 1],          # Position from current
}
target = 89  # Gym
```

---

## Original Model Walkthrough

Since the original model is designed for **text summarization**, let's first show how it would conceptually work if adapted for location prediction, then explain why the proposed model is more appropriate.

### Hypothetical Original Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│          ORIGINAL MODEL (Conceptual for Location Task)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  If we were to use the original model:                                      │
│                                                                              │
│  STEP 1: Input (as "document")                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Encode locations as "words": [101, 205, 150, 312, 150]             │   │
│  │  (Would need a mapping to make this work)                           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  STEP 2: Embedding                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E = Embedding(x)  → [5, 128]  (5 positions × 128 dims)             │   │
│  │                                                                       │   │
│  │  Only location information - no user, time, weekday, etc.           │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  STEP 3: BiLSTM Encoding                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Forward LSTM:  h_fw = [h1_fw, h2_fw, h3_fw, h4_fw, h5_fw]         │   │
│  │  Backward LSTM: h_bw = [h1_bw, h2_bw, h3_bw, h4_bw, h5_bw]         │   │
│  │  Concatenate:   H = [5, 512]                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  STEP 4: Decoder (for sequence generation)                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  The original generates a sequence of words:                        │   │
│  │    Step 1: <START> → "Gym"                                          │   │
│  │    Step 2: "Gym" → "is"                                             │   │
│  │    Step 3: "is" → "next"                                            │   │
│  │    ...                                                               │   │
│  │                                                                       │   │
│  │  NOT SUITABLE for single location prediction!                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  PROBLEM: Original model is designed for sequence-to-sequence,             │
│           not single-output classification.                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Proposed Model Walkthrough

Now let's walk through the proposed model step by step with actual numbers.

### Step 1: Embedding Computation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 1: EMBEDDING                                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: x = [101, 205, 150, 312, 150]                                       │
│                                                                              │
│  1.1 Location Embedding (lookup from E_loc ∈ ℝ^{500 × 64}):                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_loc[101] = [0.12, -0.34, 0.56, ..., 0.08]   (64 dims)            │   │
│  │  E_loc[205] = [0.45, 0.23, -0.67, ..., -0.12]  (64 dims)            │   │
│  │  E_loc[150] = [-0.23, 0.78, 0.12, ..., 0.34]   (64 dims)            │   │
│  │  E_loc[312] = [0.67, -0.45, 0.89, ..., -0.56]  (64 dims)            │   │
│  │  E_loc[150] = [-0.23, 0.78, 0.12, ..., 0.34]   (64 dims) (same!)    │   │
│  │                                                                       │   │
│  │  Shape: [5, 64]                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.2 User Embedding (E_user[42] repeated 5 times):                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_user[42] = [0.15, -0.28, 0.42, ..., 0.19]                        │   │
│  │  Shape: [5, 64] (same vector for all positions)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.3 Time Embedding (E_time for hours [8, 8, 9, 17, 18]):                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_time[8]  = [0.08, 0.16, -0.24, ..., 0.32]   (morning)            │   │
│  │  E_time[8]  = [0.08, 0.16, -0.24, ..., 0.32]                        │   │
│  │  E_time[9]  = [0.10, 0.18, -0.22, ..., 0.30]   (morning)            │   │
│  │  E_time[17] = [-0.17, 0.34, 0.51, ..., -0.68]  (evening)            │   │
│  │  E_time[18] = [-0.18, 0.36, 0.54, ..., -0.72]  (evening)            │   │
│  │  Shape: [5, 64]                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.4 Weekday Embedding (E_week[0] = Monday, repeated 5 times):             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_week[0] = [0.25, -0.50, 0.75, ..., -0.25]  (Monday pattern)      │   │
│  │  Shape: [5, 64]                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.5 Duration Embedding (bins [2, 1, 15, 4, 8]):                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_dur[2]  = [0.05, 0.10, 0.15, ..., 0.20]   (30 min → short)       │   │
│  │  E_dur[1]  = [0.02, 0.04, 0.06, ..., 0.08]   (15 min → very short)  │   │
│  │  E_dur[15] = [-0.30, 0.60, 0.90, ..., -1.20] (8 hrs → long)         │   │
│  │  E_dur[4]  = [0.10, 0.20, 0.30, ..., 0.40]   (1 hr → medium)        │   │
│  │  E_dur[8]  = [-0.16, 0.32, 0.48, ..., -0.64] (2 hrs → medium-long)  │   │
│  │  Shape: [5, 64]                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.6 Recency Embedding (positions from end [5, 4, 3, 2, 1]):               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_rec[5] = [0.50, -1.00, 1.50, ..., -2.00]  (oldest)               │   │
│  │  E_rec[4] = [0.40, -0.80, 1.20, ..., -1.60]                         │   │
│  │  E_rec[3] = [0.30, -0.60, 0.90, ..., -1.20]                         │   │
│  │  E_rec[2] = [0.20, -0.40, 0.60, ..., -0.80]                         │   │
│  │  E_rec[1] = [0.10, -0.20, 0.30, ..., -0.40]  (most recent)          │   │
│  │  Shape: [5, 64]                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.7 Position-from-End Embedding (reverse positions [4, 3, 2, 1, 0]):      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_pos[4] = [-0.40, 0.80, -1.20, ..., 1.60]  (position 0 from start)│   │
│  │  E_pos[3] = [-0.30, 0.60, -0.90, ..., 1.20]  (position 1 from start)│   │
│  │  E_pos[2] = [-0.20, 0.40, -0.60, ..., 0.80]  (position 2 from start)│   │
│  │  E_pos[1] = [-0.10, 0.20, -0.30, ..., 0.40]  (position 3 from start)│   │
│  │  E_pos[0] = [0.00, 0.00, 0.00, ..., 0.00]    (last position)        │   │
│  │  Shape: [5, 64]                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.8 Sum All Embeddings:                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_combined = E_loc + E_user + E_time + E_week + E_dur + E_rec + E_pos│   │
│  │  Shape: [5, 64]                                                      │   │
│  │                                                                       │   │
│  │  Position 0 (Home):                                                  │   │
│  │    e_0 = [0.12 + 0.15 + 0.08 + 0.25 + 0.05 + 0.50 - 0.40, ...]     │   │
│  │        = [0.75, ...]                                                 │   │
│  │                                                                       │   │
│  │  Position 4 (Office, 2nd visit):                                    │   │
│  │    e_4 = [-0.23 + 0.15 - 0.18 + 0.25 - 0.16 + 0.10 + 0.00, ...]   │   │
│  │        = [-0.07, ...]                                                │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  1.9 Layer Normalization:                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  E_final = LayerNorm(E_combined)                                    │   │
│  │  Shape: [5, 64]                                                      │   │
│  │                                                                       │   │
│  │  Mean and variance computed per position, then normalize            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Transformer Encoding

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 2: TRANSFORMER ENCODER                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: E_final ∈ [5, 64]                                                   │
│  Padding mask: [False, False, False, False, False] (no padding)            │
│                                                                              │
│  2.1 Self-Attention (Layer 1):                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Q = E × W_Q    [5, 64] × [64, 64] = [5, 64]                        │   │
│  │  K = E × W_K    [5, 64] × [64, 64] = [5, 64]                        │   │
│  │  V = E × W_V    [5, 64] × [64, 64] = [5, 64]                        │   │
│  │                                                                       │   │
│  │  For 4 heads (head_dim = 16):                                       │   │
│  │    Q_h1 = [5, 16], Q_h2 = [5, 16], Q_h3 = [5, 16], Q_h4 = [5, 16]  │   │
│  │                                                                       │   │
│  │  Attention scores (head 1):                                          │   │
│  │    scores = Q_h1 × K_h1^T / √16 = [5, 5]                            │   │
│  │                                                                       │   │
│  │    ┌─────────────────────────────────────────────────┐               │   │
│  │    │       Home  Coffee Office  Rest  Office         │               │   │
│  │    │ Home   1.2   0.3    0.8    0.1    0.5           │               │   │
│  │    │ Coffee 0.3   1.1    0.4    0.2    0.3           │               │   │
│  │    │ Office 0.8   0.4    1.5    0.6    1.4           │               │   │
│  │    │ Rest   0.1   0.2    0.6    1.0    0.5           │               │   │
│  │    │ Office 0.5   0.3    1.4    0.5    1.5           │               │   │
│  │    └─────────────────────────────────────────────────┘               │   │
│  │                                                                       │   │
│  │  After softmax (attention weights):                                  │   │
│  │    ┌─────────────────────────────────────────────────┐               │   │
│  │    │       Home  Coffee Office  Rest  Office         │               │   │
│  │    │ Home   0.35  0.14   0.23   0.11   0.17          │               │   │
│  │    │ Coffee 0.17  0.32   0.19   0.15   0.17          │               │   │
│  │    │ Office 0.20  0.14   0.28   0.17   0.21          │ ← Office     │   │
│  │    │ Rest   0.14  0.15   0.23   0.25   0.23          │    attends   │   │
│  │    │ Office 0.16  0.13   0.27   0.17   0.27          │ ← both       │   │
│  │    └─────────────────────────────────────────────────┘    Offices   │   │
│  │                                                                       │   │
│  │  Context: Attn × V = [5, 64]                                        │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2.2 Feed-Forward Network (Layer 1):                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  FFN(x) = GELU(x × W_1) × W_2                                       │   │
│  │  W_1 ∈ [64, 128], W_2 ∈ [128, 64]                                   │   │
│  │                                                                       │   │
│  │  Output: [5, 64]                                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  2.3 Repeat for Layer 2                                                    │
│                                                                              │
│  Final encoder output: H ∈ [5, 64]                                         │
│                                                                              │
│  The Office positions (index 2 and 4) have learned to attend to each      │
│  other, recognizing the pattern of "return to Office".                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 3: Attention and Context

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 3: POINTER ATTENTION                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  3.1 Extract Query (last valid position):                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  query = H[4]  (position 4 = last Office visit)                     │   │
│  │  Shape: [64]                                                         │   │
│  │                                                                       │   │
│  │  This represents: "Just finished 2 hours at office on Monday evening"│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3.2 Compute Attention Scores:                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Q = query × W_Q    [64] × [64, 64] = [64]                          │   │
│  │  K = H × W_K        [5, 64] × [64, 64] = [5, 64]                    │   │
│  │                                                                       │   │
│  │  scores = Q × K^T / √64 = [5]                                       │   │
│  │                                                                       │   │
│  │  Raw scores:                                                         │   │
│  │    ┌───────────────────────────────────────────┐                    │   │
│  │    │ Home  Coffee  Office  Rest  Office        │                    │   │
│  │    │  0.8   0.4     1.2    0.6    1.4          │                    │   │
│  │    └───────────────────────────────────────────┘                    │   │
│  │                                                                       │   │
│  │  Why highest on last Office?                                        │   │
│  │    - Query is from last Office position                             │   │
│  │    - Model learns that recent positions are most relevant           │   │
│  │    - Office-to-Office pattern is strong                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3.3 Apply Softmax:                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  α = softmax(scores) = [0.15, 0.10, 0.23, 0.12, 0.40]              │   │
│  │                         Home Coffee Office Rest  Office             │   │
│  │                                                                       │   │
│  │  Interpretation:                                                     │   │
│  │    - 40% attention on last Office (most recent, same location)      │   │
│  │    - 23% attention on first Office (same type of place)            │   │
│  │    - 15% attention on Home (starting point of day)                 │   │
│  │    - Lower attention on Coffee and Restaurant (less relevant)       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  3.4 Compute Context:                                                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  V = H × W_V    [5, 64]                                             │   │
│  │  context = α × V = Σ α_i × V_i                                      │   │
│  │          = 0.15×V[0] + 0.10×V[1] + 0.23×V[2] + 0.12×V[3] + 0.40×V[4]│   │
│  │          = [0.23, -0.45, 0.67, ...]  (64 dims)                      │   │
│  │                                                                       │   │
│  │  Context captures: "Weighted summary of trajectory, emphasizing     │   │
│  │                     office visits and most recent locations"        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 4: Gate Computation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 4: GATE COMPUTATION                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  4.1 Concatenate Context and Query:                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  gate_input = [context; query]                                      │   │
│  │             = [[0.23, -0.45, ...], [0.18, -0.32, ...]]              │   │
│  │             = [0.23, -0.45, ..., 0.18, -0.32, ...]                  │   │
│  │  Shape: [128]                                                        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4.2 MLP Layer 1:                                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  h = GELU(W_1 × gate_input + b_1)                                   │   │
│  │    = GELU([128] × [128, 64] + [64])                                 │   │
│  │    = GELU([0.45, -0.23, 0.78, ...])                                 │   │
│  │    = [0.32, -0.14, 0.62, ...]  (64 dims)                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4.3 Dropout (p=0.15):                                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  h' = Dropout(h)                                                    │   │
│  │  (During training: randomly zero ~15% of elements)                  │   │
│  │  (During inference: no dropout)                                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  4.4 MLP Layer 2 + Sigmoid:                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  gate = σ(W_2 × h' + b_2)                                           │   │
│  │       = σ([64] × [64, 1] + [1])                                     │   │
│  │       = σ(1.2)                                                       │   │
│  │       = 0.77                                                         │   │
│  │                                                                       │   │
│  │  Interpretation:                                                     │   │
│  │    gate = 0.77 means:                                               │   │
│  │      - 77% weight on pointer (copy from input sequence)             │   │
│  │      - 23% weight on generation (predict any location)              │   │
│  │                                                                       │   │
│  │    The model believes that for Alice's evening routine,             │   │
│  │    the next location is likely related to places she visited       │   │
│  │    today, but there's also a chance it's a new destination.        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Step 5: Final Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STEP 5: FINAL DISTRIBUTION                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  5.1 Generation Distribution:                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  gen_logits = W_gen × context + b_gen                               │   │
│  │             = [64] × [64, 500] + [500]                               │   │
│  │             = [500]                                                  │   │
│  │                                                                       │   │
│  │  gen_probs = softmax(gen_logits)                                    │   │
│  │                                                                       │   │
│  │  Top probabilities:                                                  │   │
│  │    ┌─────────────────────────────────────────────────────┐          │   │
│  │    │  Location     ID    Probability                     │          │   │
│  │    │  Gym          89    0.25        ← Model learned!    │          │   │
│  │    │  Home         101   0.18                            │          │   │
│  │    │  Office       150   0.15                            │          │   │
│  │    │  Park         178   0.08                            │          │   │
│  │    │  Restaurant   312   0.05                            │          │   │
│  │    │  ...          ...   ...                             │          │   │
│  │    └─────────────────────────────────────────────────────┘          │   │
│  │                                                                       │   │
│  │  The generation head learned Alice's Monday evening pattern:        │   │
│  │  After work → Gym (25% confident)                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  5.2 Pointer Distribution (scatter attention):                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Input sequence: [101, 205, 150, 312, 150]                          │   │
│  │  Attention:      [0.15, 0.10, 0.23, 0.12, 0.40]                    │   │
│  │                                                                       │   │
│  │  Scatter to location indices:                                       │   │
│  │    ptr_probs = zeros(500)                                           │   │
│  │    ptr_probs[101] += 0.15  (Home)                                   │   │
│  │    ptr_probs[205] += 0.10  (Coffee)                                 │   │
│  │    ptr_probs[150] += 0.23 + 0.40 = 0.63  (Office - appears twice!) │   │
│  │    ptr_probs[312] += 0.12  (Restaurant)                             │   │
│  │                                                                       │   │
│  │  Result:                                                             │   │
│  │    ┌─────────────────────────────────────────────────────┐          │   │
│  │    │  Location     ID    Probability                     │          │   │
│  │    │  Office       150   0.63        ← Accumulated!      │          │   │
│  │    │  Home         101   0.15                            │          │   │
│  │    │  Restaurant   312   0.12                            │          │   │
│  │    │  Coffee       205   0.10                            │          │   │
│  │    │  All others   ...   0.00                            │          │   │
│  │    └─────────────────────────────────────────────────────┘          │   │
│  │                                                                       │   │
│  │  Note: Gym (89) is NOT in pointer distribution!                     │   │
│  │        (Alice hasn't visited the gym today yet)                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  5.3 Combine with Gate:                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  final_probs = gate × ptr_probs + (1-gate) × gen_probs              │   │
│  │              = 0.77 × ptr_probs + 0.23 × gen_probs                  │   │
│  │                                                                       │   │
│  │  For key locations:                                                  │   │
│  │    ┌───────────────────────────────────────────────────────────────┐│   │
│  │    │ Location   Pointer   Gen      Final                          ││   │
│  │    │ Office 150 0.77×0.63 0.23×0.15 = 0.485 + 0.035 = 0.520      ││   │
│  │    │ Home 101   0.77×0.15 0.23×0.18 = 0.116 + 0.041 = 0.157      ││   │
│  │    │ Restaurant 0.77×0.12 0.23×0.05 = 0.092 + 0.012 = 0.104      ││   │
│  │    │ Coffee 205 0.77×0.10 0.23×0.03 = 0.077 + 0.007 = 0.084      ││   │
│  │    │ Gym 89     0.77×0.00 0.23×0.25 = 0.000 + 0.058 = 0.058      ││   │
│  │    │ ...        ...       ...       ...                           ││   │
│  │    └───────────────────────────────────────────────────────────────┘│   │
│  │                                                                       │   │
│  │  Top predictions:                                                    │   │
│  │    1. Office (150): 52.0%    ← Most likely (pointer dominates)     │   │
│  │    2. Home (101):   15.7%                                           │   │
│  │    3. Restaurant:   10.4%                                           │   │
│  │    4. Coffee:        8.4%                                           │   │
│  │    5. Gym (89):      5.8%    ← Target location!                    │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Numerical Comparison

### Parameter Count Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER COUNT FOR THIS EXAMPLE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  PROPOSED MODEL (for location prediction):                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Embeddings:                                                          │   │
│  │    Location: 500 × 64 = 32,000                                       │   │
│  │    User: 200 × 64 = 12,800                                           │   │
│  │    Time: 24 × 64 = 1,536                                             │   │
│  │    Weekday: 7 × 64 = 448                                             │   │
│  │    Duration: 20 × 64 = 1,280                                         │   │
│  │    Recency: 20 × 64 = 1,280                                          │   │
│  │    Position: 100 × 64 = 6,400                                        │   │
│  │    Subtotal: 55,744                                                  │   │
│  │                                                                       │   │
│  │  Transformer (2 layers):                                             │   │
│  │    Per layer: 4×64×64×3 + 64×128×2 + 64×4 = 49,152 + 16,384 + 256  │   │
│  │             = 65,792                                                  │   │
│  │    Total: 2 × 65,792 = 131,584                                      │   │
│  │                                                                       │   │
│  │  Attention: 64 × 64 × 3 = 12,288                                    │   │
│  │  Gate MLP: 128 × 64 + 64 × 1 = 8,256                                │   │
│  │  Generation: 64 × 500 = 32,000                                      │   │
│  │                                                                       │   │
│  │  TOTAL: ~240,000 parameters                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ORIGINAL MODEL (for text summarization):                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Embeddings: 50,000 × 128 = 6,400,000                               │   │
│  │  BiLSTM encoder: ~2,100,000                                         │   │
│  │  State reduction: ~262,000                                          │   │
│  │  LSTM decoder: ~1,050,000                                           │   │
│  │  Attention: ~780,000                                                │   │
│  │  Output projection: ~38,400,000                                     │   │
│  │  p_gen: ~1,200                                                      │   │
│  │                                                                       │   │
│  │  TOTAL: ~47,000,000 parameters                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Ratio: Original / Proposed ≈ 196× more parameters                        │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Attention Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ATTENTION VISUALIZATION                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Input: [Home, Coffee, Office, Restaurant, Office]                          │
│  Query: Office (position 4)                                                  │
│                                                                              │
│  Attention weights (α):                                                      │
│                                                                              │
│  Home     │████████████████                              │ 0.15             │
│  Coffee   │██████████                                    │ 0.10             │
│  Office   │███████████████████████████████████           │ 0.23             │
│  Rest     │████████████████                              │ 0.12             │
│  Office*  │████████████████████████████████████████████████████ 0.40       │
│           └──────────────────────────────────────────────┘                  │
│           0.00            0.25            0.50                               │
│                                                                              │
│  * = query position                                                          │
│                                                                              │
│  Interpretation:                                                             │
│  - Highest attention on most recent Office (0.40) - recency bias           │
│  - Second highest on first Office (0.23) - same location type              │
│  - Moderate attention on Home (0.15) - start of journey                    │
│  - Lower attention on Coffee (0.10) and Restaurant (0.12) - less relevant  │
│                                                                              │
│  After scatter_add for Office (150):                                        │
│  - Total pointer probability for Office = 0.23 + 0.40 = 0.63               │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Final Prediction Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREDICTION ANALYSIS                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Target: Gym (location 89)                                                   │
│                                                                              │
│  Model's top-5 predictions:                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Rank   Location        ID     Probability   Source                  │   │
│  │  ────────────────────────────────────────────────────────────────── │   │
│  │  1      Office          150    52.0%         Pointer (63% × 0.77)   │   │
│  │  2      Home            101    15.7%         Mixed                   │   │
│  │  3      Restaurant      312    10.4%         Pointer                 │   │
│  │  4      Coffee Shop     205     8.4%         Pointer                 │   │
│  │  5      Gym             89      5.8%         Generation (25% × 0.23)│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Metrics:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Acc@1: 0 (Office ≠ Gym)                                            │   │
│  │  Acc@5: 1 (Gym is in top 5)                                         │   │
│  │  MRR:   1/5 = 0.20                                                  │   │
│  │  NDCG@5: 1/log₂(6) = 0.387                                         │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Analysis:                                                                   │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                                                                       │   │
│  │  Why did the model predict Office first?                            │   │
│  │  - Alice has visited Office twice today (strong pointer signal)     │   │
│  │  - Gate value 0.77 heavily weights the pointer mechanism           │   │
│  │  - Office accumulates 63% of pointer probability                    │   │
│  │                                                                       │   │
│  │  Why is Gym in top 5 despite not being visited today?               │   │
│  │  - Generation mechanism captures Alice's regular patterns           │   │
│  │  - Model learned "Monday evening after work → Gym" pattern         │   │
│  │  - 23% generation weight allows this to surface                     │   │
│  │                                                                       │   │
│  │  How could the model improve?                                       │   │
│  │  - More training data with gym visits after office                 │   │
│  │  - Lower gate value when context suggests new destination          │   │
│  │  - Better user-specific pattern learning                           │   │
│  │                                                                       │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This example demonstrates:

1. **Multi-modal embeddings** capture rich context (user, time, location, etc.)
2. **Transformer encoder** enables positions to attend to each other (Office-to-Office)
3. **Pointer mechanism** provides strong signal for repeated locations
4. **Generation mechanism** enables prediction of unvisited locations (Gym)
5. **Gate** balances between copying (pointer) and generating (new predictions)

The prediction of Office first is reasonable given the trajectory, and the presence of Gym in top-5 shows the model learned Alice's patterns.

---

*Next: [15_JUSTIFICATION_OF_CHANGES.md](15_JUSTIFICATION_OF_CHANGES.md) - Why these design changes were made*
