# MHSA Architecture Diagrams

## Visual Representations at Different Detail Levels

---

## 1. Simplified Overview (High-Level)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MHSA MODEL - BIRD'S EYE VIEW                     │
└─────────────────────────────────────────────────────────────────────┘

    Historical Location Sequence              Next Location Prediction
    [Home → Work → Cafe → Gym → ...]   →→→   [Restaurant (67%)]
                                              [Cafe (15%)]
                                              [Home (10%)]
                                              [...]

    ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
    │   Input     │────▶│ Transformer │────▶│   Output    │
    │  Features   │     │   Encoder   │     │ Prediction  │
    └─────────────┘     └─────────────┘     └─────────────┘
         │                    │                    │
    Location IDs         Self-Attention      Probability
    Time of Day          captures patterns   over all
    Day of Week          in sequence         locations
    Duration
```

---

## 2. Moderate Detail View

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         MHSA MODEL ARCHITECTURE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT FEATURES                                                          │
│  ══════════════                                                          │
│  • Location IDs: [45, 12, 45, 78, 23]      Sequence of visited places   │
│  • Time:         [34, 48, 32, 68, 36]      15-min time slots (0-95)     │
│  • Weekday:      [1,  1,  2,  2,  3]       Day of week (0-6)            │
│  • Duration:     [4,  1,  6,  3,  2]       30-min bins (stay duration)  │
│  • User ID:      3                          User identifier              │
│                                                                          │
│                         ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      EMBEDDING LAYER                              │  │
│  │                                                                   │  │
│  │   Location      Temporal        Duration       Positional         │  │
│  │   Embedding  +  Embedding   +   Embedding  +   Encoding           │  │
│  │   [V → D]       [96,24,7→D]     [96 → D]       [sin/cos]          │  │
│  │                                                                   │  │
│  │   Result: [seq_len, batch, D] combined embedding                  │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                         ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    TRANSFORMER ENCODER                            │  │
│  │                                                                   │  │
│  │   ┌─────────────────────────────────────────────────────────┐     │  │
│  │   │  Multi-Head Self-Attention (8 heads)                    │     │  │
│  │   │  + Residual Connection + Layer Normalization            │     │  │
│  │   └─────────────────────────────────────────────────────────┘     │  │
│  │                         ▼                                         │  │
│  │   ┌─────────────────────────────────────────────────────────┐     │  │
│  │   │  Feed-Forward Network (D → 4D → D)                      │     │  │
│  │   │  + Residual Connection + Layer Normalization            │     │  │
│  │   └─────────────────────────────────────────────────────────┘     │  │
│  │                                                                   │  │
│  │   × N layers (typically 2-4)                                      │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                         ▼                                                │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      OUTPUT LAYER                                 │  │
│  │                                                                   │  │
│  │   Select Last Position → Add User Embedding → Residual FC        │  │
│  │                        → Final Linear [D → V]                     │  │
│  │                                                                   │  │
│  │   Result: [batch, num_locations] logits                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                         ▼                                                │
│  OUTPUT                                                                  │
│  ══════                                                                  │
│  • Logits for each possible location                                    │
│  • Softmax → Probability distribution                                   │
│  • Top-k predictions for evaluation                                     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component View

### 3.1 AllEmbedding Layer

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        AllEmbedding LAYER                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUTS:                                                                 │
│  ───────                                                                 │
│  src:      [S, B]     Location IDs (0 to V-1)                           │
│  time:     [S, B]     Time slots (0-95)                                 │
│  weekday:  [S, B]     Day of week (0-6)                                 │
│  duration: [S, B]     Duration bins (0-95)                              │
│                                                                          │
│  EMBEDDING LOOKUPS:                                                      │
│  ═════════════════                                                       │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Location Embedding                                                  │ │
│  │ ────────────────────                                                │ │
│  │ nn.Embedding(num_locations, D)                                      │ │
│  │ Example: 1187 locations × 32 dimensions                             │ │
│  │                                                                     │ │
│  │ Input: [45, 12, 45, 78, 23]                                         │ │
│  │ Output: [[v₄₅], [v₁₂], [v₄₅], [v₇₈], [v₂₃]]  each vᵢ ∈ ℝ³²        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    │ + (element-wise addition)                          │
│                    ▼                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Temporal Embedding                                                  │ │
│  │ ───────────────────                                                 │ │
│  │                                                                     │ │
│  │ Time Slot → Decompose:                                              │ │
│  │   slot 34 → hour=8, quarter=2 (8:30-8:44)                          │ │
│  │                                                                     │ │
│  │ ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │ │
│  │ │ Hour Embed   │  │ Minute Embed │  │ Weekday Embed│               │ │
│  │ │ [24 × D]     │  │ [4 × D]      │  │ [7 × D]      │               │ │
│  │ │ hour_emb[8]  │+ │ min_emb[2]   │+ │ day_emb[1]   │               │ │
│  │ └──────────────┘  └──────────────┘  └──────────────┘               │ │
│  │                                                                     │ │
│  │ Result: temporal_embedding ∈ ℝᴰ                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    │ + (element-wise addition)                          │
│                    ▼                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Duration Embedding                                                  │ │
│  │ ───────────────────                                                 │ │
│  │ nn.Embedding(96, D)  # 96 bins × 30 min = 48 hours max             │ │
│  │                                                                     │ │
│  │ Duration 4 (2 hours) → dur_emb[4] ∈ ℝᴰ                             │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    │ × √D (scaling)                                     │
│                    ▼                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Positional Encoding                                                 │ │
│  │ ────────────────────                                                │ │
│  │                                                                     │ │
│  │ For position p and dimension i:                                     │ │
│  │   PE(p,2i)   = sin(p / 10000^(2i/D))                               │ │
│  │   PE(p,2i+1) = cos(p / 10000^(2i/D))                               │ │
│  │                                                                     │ │
│  │ Position 0: [0, 1, 0, 1, 0, 1, ...]                                 │ │
│  │ Position 1: [0.84, 0.54, 0.1, 0.99, ...]                           │ │
│  │                                                                     │ │
│  │ Combined + PE + Dropout(0.1)                                        │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    ▼                                                     │
│  OUTPUT:                                                                 │
│  ───────                                                                 │
│  [S, B, D] - Embedded sequence ready for Transformer                    │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Transformer Encoder Layer

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     TRANSFORMER ENCODER LAYER                            │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: X ∈ ℝˢˣᴮˣᴰ                                                       │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════════ │
│  MULTI-HEAD SELF-ATTENTION                                               │
│  ═══════════════════════════════════════════════════════════════════════ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  Step 1: Linear Projections                                         │ │
│  │  ─────────────────────────────                                      │ │
│  │  Q = X @ W_Q    [S,B,D] @ [D,D] = [S,B,D]                          │ │
│  │  K = X @ W_K    [S,B,D] @ [D,D] = [S,B,D]                          │ │
│  │  V = X @ W_V    [S,B,D] @ [D,D] = [S,B,D]                          │ │
│  │                                                                     │ │
│  │  Step 2: Reshape for Multi-Head                                     │ │
│  │  ────────────────────────────────                                   │ │
│  │  Q,K,V → [S, B, H, D/H] → [B, H, S, D/H]                           │ │
│  │  With H=8 heads, D=32: [B, 8, S, 4]                                 │ │
│  │                                                                     │ │
│  │  Step 3: Compute Attention Scores                                   │ │
│  │  ─────────────────────────────────                                  │ │
│  │                                                                     │ │
│  │     Q · Kᵀ                                                          │ │
│  │  ──────────                                                         │ │
│  │     √d_k                                                            │ │
│  │                                                                     │ │
│  │  [B,H,S,d_k] @ [B,H,d_k,S] → [B,H,S,S]                             │ │
│  │                                                                     │ │
│  │  Example attention matrix (before mask):                            │ │
│  │  ┌─────────────────────────────────┐                               │ │
│  │  │  0.5  0.3  0.1  0.05  0.05     │  Row: query position          │ │
│  │  │  0.4  0.4  0.1  0.05  0.05     │  Col: key position            │ │
│  │  │  0.2  0.3  0.4  0.05  0.05     │                               │ │
│  │  │  0.1  0.2  0.3  0.3   0.1      │                               │ │
│  │  │  0.1  0.1  0.2  0.3   0.3      │                               │ │
│  │  └─────────────────────────────────┘                               │ │
│  │                                                                     │ │
│  │  Step 4: Apply Causal Mask                                          │ │
│  │  ─────────────────────────────                                      │ │
│  │                                                                     │ │
│  │  Causal Mask (prevent seeing future):                               │ │
│  │  ┌─────────────────────────────────┐                               │ │
│  │  │   0    -∞   -∞   -∞   -∞       │                               │ │
│  │  │   0     0   -∞   -∞   -∞       │                               │ │
│  │  │   0     0    0   -∞   -∞       │                               │ │
│  │  │   0     0    0    0   -∞       │                               │ │
│  │  │   0     0    0    0    0       │                               │ │
│  │  └─────────────────────────────────┘                               │ │
│  │                                                                     │ │
│  │  After mask + softmax:                                              │ │
│  │  ┌─────────────────────────────────┐                               │ │
│  │  │  1.0   0    0    0    0        │  Only attend to past          │ │
│  │  │  0.55 0.45  0    0    0        │                               │ │
│  │  │  0.27 0.33 0.40  0    0        │                               │ │
│  │  │  0.11 0.22 0.33 0.34  0        │                               │ │
│  │  │  0.09 0.09 0.18 0.28 0.36      │                               │ │
│  │  └─────────────────────────────────┘                               │ │
│  │                                                                     │ │
│  │  Step 5: Apply Attention to Values                                  │ │
│  │  ────────────────────────────────────                               │ │
│  │  Output = Attention_Weights @ V                                     │ │
│  │  [B,H,S,S] @ [B,H,S,d_k] → [B,H,S,d_k]                            │ │
│  │                                                                     │ │
│  │  Step 6: Concatenate Heads                                          │ │
│  │  ─────────────────────────                                          │ │
│  │  [B,H,S,d_k] → [B,S,H*d_k] → [B,S,D]                              │ │
│  │  Final projection: @ W_O                                            │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    │ + X (residual)                                     │
│                    ▼                                                     │
│              LayerNorm                                                   │
│                    │                                                     │
│                    ▼                                                     │
│  ═══════════════════════════════════════════════════════════════════════ │
│  FEED-FORWARD NETWORK                                                    │
│  ═══════════════════════════════════════════════════════════════════════ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  FFN(x) = Linear₂(GELU(Linear₁(x)))                                │ │
│  │                                                                     │ │
│  │  Linear₁: [D → 4D]    e.g., [32 → 128]                             │ │
│  │  GELU activation                                                    │ │
│  │  Linear₂: [4D → D]    e.g., [128 → 32]                             │ │
│  │                                                                     │ │
│  │  GELU(x) = x · Φ(x) where Φ is standard normal CDF                 │ │
│  │  Smoother than ReLU, works well in Transformers                     │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    │ + (residual from after attention)                  │
│                    ▼                                                     │
│              LayerNorm                                                   │
│                    │                                                     │
│                    ▼                                                     │
│  Output: [S, B, D]                                                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.3 FullyConnected Output Layer

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     FullyConnected OUTPUT LAYER                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Input: Encoder output [S, B, D]                                         │
│         Sequence lengths [B]                                             │
│         User IDs [B]                                                     │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════════ │
│  STEP 1: SELECT LAST VALID POSITION                                      │
│  ═══════════════════════════════════════════════════════════════════════ │
│                                                                          │
│  Encoder output (example with batch=2):                                  │
│  ┌─────────────────────────────────────┐                                │
│  │ Position 0: [enc₀₀, enc₀₁]          │                                │
│  │ Position 1: [enc₁₀, enc₁₁]          │                                │
│  │ Position 2: [enc₂₀, enc₂₁]          │                                │
│  │ Position 3: [enc₃₀, PAD  ]          │  ← Sample 1 ends               │
│  │ Position 4: [enc₄₀, PAD  ]          │  ← Sample 0 ends               │
│  └─────────────────────────────────────┘                                │
│                                                                          │
│  lengths = [5, 3]                                                        │
│  Select: [enc₄₀, enc₂₁] → [B, D]                                        │
│                                                                          │
│  ═══════════════════════════════════════════════════════════════════════ │
│  STEP 2: ADD USER EMBEDDING                                              │
│  ═══════════════════════════════════════════════════════════════════════ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ User Embedding Table: [num_users × D]                               │ │
│  │                                                                     │ │
│  │ User 3: [0.1, 0.2, -0.1, ...]  ∈ ℝᴰ                                │ │
│  │ User 7: [-0.05, 0.3, 0.15, ...] ∈ ℝᴰ                               │ │
│  │                                                                     │ │
│  │ out = encoder_out + user_embedding                                  │ │
│  │                                                                     │ │
│  │ Intuition: Different users have different location preferences      │ │
│  │            User embedding captures individual mobility patterns     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│              Dropout(0.1)                                                │
│                    │                                                     │
│                    ▼                                                     │
│  ═══════════════════════════════════════════════════════════════════════ │
│  STEP 3: RESIDUAL BLOCK                                                  │
│  ═══════════════════════════════════════════════════════════════════════ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  Input: x [B, D]                                                    │ │
│  │                                                                     │ │
│  │  ┌─────────────────────────────────────────────────────────────┐   │ │
│  │  │  Linear₁: [D → 2D]        [32 → 64]                         │   │ │
│  │  │  ReLU activation                                             │   │ │
│  │  │  Dropout                                                     │   │ │
│  │  │  Linear₂: [2D → D]        [64 → 32]                         │   │ │
│  │  │  Dropout                                                     │   │ │
│  │  └─────────────────────────────────────────────────────────────┘   │ │
│  │                        │                                            │ │
│  │                        │ + x (residual)                             │ │
│  │                        ▼                                            │ │
│  │               BatchNorm1d(D)                                        │ │
│  │                                                                     │ │
│  │  Purpose: Additional non-linear transformation capacity             │ │
│  │           Residual prevents gradient degradation                    │ │
│  │           BatchNorm stabilizes training                             │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    ▼                                                     │
│  ═══════════════════════════════════════════════════════════════════════ │
│  STEP 4: FINAL CLASSIFICATION                                            │
│  ═══════════════════════════════════════════════════════════════════════ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │                                                                     │ │
│  │  Linear: [D → num_locations]                                        │ │
│  │                                                                     │ │
│  │  [32] → [1187]  (for GeoLife)                                       │ │
│  │  [96] → [7038]  (for DIY)                                           │ │
│  │                                                                     │ │
│  │  Output: logits [B, V]                                              │ │
│  │                                                                     │ │
│  │  Each logit represents the "score" for a location                   │ │
│  │  Higher score = more likely prediction                              │ │
│  │                                                                     │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                    │                                                     │
│                    ▼                                                     │
│  OUTPUT: [B, num_locations] logits                                       │
│                                                                          │
│  During training: CrossEntropyLoss(logits, targets)                     │
│  During inference: softmax(logits) → probabilities                      │
│                    argmax(logits) → prediction                          │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE DATA FLOW                               │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Raw Data (.pk file)                                                     │
│  ══════════════════                                                      │
│  │                                                                       │
│  │  {                                                                    │
│  │    'X': [45, 12, 45, 78, 23],     # Locations                        │
│  │    'Y': 67,                        # Target                          │
│  │    'user_X': [3, 3, 3, 3, 3],     # User                             │
│  │    'weekday_X': [1, 1, 2, 2, 3],  # Weekdays                         │
│  │    'start_min_X': [510, 720, ...], # Times                           │
│  │    'dur_X': [120, 45, 180, ...]   # Durations                        │
│  │  }                                                                    │
│  │                                                                       │
│  ▼                                                                       │
│  LocationDataset.__getitem__()                                           │
│  ═════════════════════════════                                           │
│  │                                                                       │
│  │  x = tensor([45, 12, 45, 78, 23])                                    │
│  │  y = tensor(67)                                                       │
│  │  context = {                                                          │
│  │    'user': tensor(3),                                                 │
│  │    'time': tensor([34, 48, 32, 68, 36]),  # //15                     │
│  │    'weekday': tensor([1, 1, 2, 2, 3]),                               │
│  │    'duration': tensor([4, 1, 6, 3, 2])    # //30                     │
│  │  }                                                                    │
│  │                                                                       │
│  ▼                                                                       │
│  collate_fn() - Batch multiple samples                                   │
│  ═════════════════════════════════════                                   │
│  │                                                                       │
│  │  x_batch = [S, B]        (padded)                                    │
│  │  y_batch = [B]                                                        │
│  │  context_batch = {                                                    │
│  │    'len': [B],           (original lengths)                          │
│  │    'user': [B],                                                       │
│  │    'time': [S, B],       (padded)                                    │
│  │    ...                                                                │
│  │  }                                                                    │
│  │                                                                       │
│  ▼                                                                       │
│  MHSA.forward()                                                          │
│  ═════════════════                                                       │
│  │                                                                       │
│  │  Step 1: AllEmbedding                                                │
│  │  ┌──────────────────────────────────────────┐                        │
│  │  │ [S, B] → [S, B, D]                       │                        │
│  │  │ loc + time + dur + pos                    │                        │
│  │  └──────────────────────────────────────────┘                        │
│  │                   │                                                   │
│  │  Step 2: Generate Masks                                               │
│  │  ┌──────────────────────────────────────────┐                        │
│  │  │ Causal: [S, S]                           │                        │
│  │  │ Padding: [B, S]                          │                        │
│  │  └──────────────────────────────────────────┘                        │
│  │                   │                                                   │
│  │  Step 3: TransformerEncoder                                          │
│  │  ┌──────────────────────────────────────────┐                        │
│  │  │ [S, B, D] → [S, B, D]                    │                        │
│  │  │ N layers of self-attention + FFN          │                        │
│  │  └──────────────────────────────────────────┘                        │
│  │                   │                                                   │
│  │  Step 4: Select Last Position                                        │
│  │  ┌──────────────────────────────────────────┐                        │
│  │  │ [S, B, D] + [B] → [B, D]                 │                        │
│  │  │ Use lengths to index last valid           │                        │
│  │  └──────────────────────────────────────────┘                        │
│  │                   │                                                   │
│  │  Step 5: FullyConnected                                              │
│  │  ┌──────────────────────────────────────────┐                        │
│  │  │ [B, D] + [B] → [B, V]                    │                        │
│  │  │ + user_emb + residual + linear            │                        │
│  │  └──────────────────────────────────────────┘                        │
│  │                                                                       │
│  ▼                                                                       │
│  logits = [B, V]                                                         │
│  │                                                                       │
│  │  Training:                                                            │
│  │  ──────────                                                           │
│  │  loss = CrossEntropyLoss(logits, y_batch)                            │
│  │  loss.backward()                                                      │
│  │  optimizer.step()                                                     │
│  │                                                                       │
│  │  Evaluation:                                                          │
│  │  ────────────                                                         │
│  │  probs = softmax(logits)                                             │
│  │  top_k = topk(logits, k=10)                                          │
│  │  metrics = calculate_metrics(logits, y_batch)                        │
│  │                                                                       │
│  ▼                                                                       │
│  Results                                                                 │
│  ═══════                                                                 │
│  • Predictions: top-k locations                                         │
│  • Metrics: Acc@1, MRR, NDCG, F1                                        │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Parameter Count Diagram

```
┌──────────────────────────────────────────────────────────────────────────┐
│                       PARAMETER COUNT BREAKDOWN                          │
│                        (GeoLife Configuration)                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Total: ~112,547 parameters                                              │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ AllEmbedding Layer                                    ~40,000 params │ │
│  │ ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                     │ │
│  │  Location Embedding:    1187 × 32 = 37,984                          │ │
│  │  Hour Embedding:          24 × 32 =    768                          │ │
│  │  Minute Embedding:         4 × 32 =    128                          │ │
│  │  Weekday Embedding:        7 × 32 =    224                          │ │
│  │  Duration Embedding:      96 × 32 =  3,072                          │ │
│  │                                       ──────                         │ │
│  │                               Total: 42,176                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ Transformer Encoder (2 layers)                        ~50,000 params │ │
│  │ ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                     │ │
│  │  Per layer:                                                         │ │
│  │    Q, K, V projections: 3 × (32 × 32 + 32) = 3,168                  │ │
│  │    Output projection:   32 × 32 + 32 = 1,056                        │ │
│  │    FFN Layer 1:         32 × 128 + 128 = 4,224                      │ │
│  │    FFN Layer 2:         128 × 32 + 32 = 4,128                       │ │
│  │    LayerNorm (×2):      2 × (32 + 32) = 128                         │ │
│  │                                         ──────                       │ │
│  │                            Per layer: 12,704                         │ │
│  │                                                                     │ │
│  │  Total (2 layers):        12,704 × 2 = 25,408                       │ │
│  │  Final LayerNorm:             32 + 32 = 64                          │ │
│  │                                       ──────                         │ │
│  │                               Total: 25,472                          │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  ┌────────────────────────────────────────────────────────────────────┐ │
│  │ FullyConnected Layer                                  ~45,000 params │ │
│  │ ─────────────────────────────────────────────────────────────────── │ │
│  │                                                                     │ │
│  │  User Embedding:         46 × 32 = 1,472                            │ │
│  │  Residual Linear 1:  32 × 64 + 64 = 2,112                           │ │
│  │  Residual Linear 2:  64 × 32 + 32 = 2,080                           │ │
│  │  BatchNorm:               32 + 32 = 64                              │ │
│  │  Final Linear:    32 × 1187 + 1187 = 39,171                         │ │
│  │                                      ──────                          │ │
│  │                              Total: 44,899                           │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                          │
│  GRAND TOTAL: 42,176 + 25,472 + 44,899 = 112,547 parameters             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Attention Visualization

```
┌──────────────────────────────────────────────────────────────────────────┐
│                     ATTENTION PATTERN EXAMPLES                           │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  Example Sequence: Home → Work → Cafe → Work → ?                        │
│                                                                          │
│  CAUSAL ATTENTION PATTERN                                                │
│  ═══════════════════════                                                 │
│                                                                          │
│  Attention matrix for predicting next location:                          │
│                                                                          │
│           ┌────────────────────────────────────────┐                    │
│           │      Home   Work   Cafe   Work        │                    │
│           │                                        │                    │
│  Query    │ Home  █░░░░  ░░░░░  ░░░░░  ░░░░░      │                    │
│  Position │ Work  █░░░░  █████  ░░░░░  ░░░░░      │                    │
│           │ Cafe  ░░░░░  █████  ██░░░  ░░░░░      │                    │
│           │ Work  ░░░░░  █████  ░░░░░  █████      │   ← Uses this     │
│           └────────────────────────────────────────┘                    │
│                                                                          │
│  Legend: █ = high attention, ░ = low/no attention                       │
│                                                                          │
│  Interpretation:                                                         │
│  • Last Work position attends strongly to previous Work                 │
│  • This suggests pattern recognition (Work → Work → Home?)               │
│                                                                          │
│  MULTI-HEAD ATTENTION                                                    │
│  ═══════════════════                                                     │
│                                                                          │
│  Different heads learn different patterns:                               │
│                                                                          │
│  Head 1: Recency Pattern        Head 2: Location Repeat Pattern         │
│  ┌────────────────────┐         ┌────────────────────┐                  │
│  │ H  W  C  W         │         │ H  W  C  W         │                  │
│  │ ░  ░  ░  █  ← W    │         │ ░  █  ░  █  ← W    │                  │
│  └────────────────────┘         └────────────────────┘                  │
│  Focuses on most recent         Attends to same location                │
│                                                                          │
│  Head 3: First Position         Head 4: Temporal Pattern                │
│  ┌────────────────────┐         ┌────────────────────┐                  │
│  │ H  W  C  W         │         │ H  W  C  W         │                  │
│  │ █  ░  ░  ░  ← W    │         │ ░  █  █  ░  ← W    │                  │
│  └────────────────────┘         └────────────────────┘                  │
│  Origin/home reference          Mid-day activities                       │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

*Diagrams created for MHSA model documentation - next_loc_clean_v2 project*
