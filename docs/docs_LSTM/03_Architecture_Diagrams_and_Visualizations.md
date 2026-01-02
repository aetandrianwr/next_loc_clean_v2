# LSTM Model: Architecture Diagrams and Visual Explanations

## Table of Contents

1. [Simplified Overview](#1-simplified-overview)
2. [Moderate Detail Diagrams](#2-moderate-detail-diagrams)
3. [Detailed Component Diagrams](#3-detailed-component-diagrams)
4. [Data Flow Visualizations](#4-data-flow-visualizations)
5. [LSTM Cell Internals](#5-lstm-cell-internals)
6. [Training Pipeline Visualization](#6-training-pipeline-visualization)

---

## 1. Simplified Overview

### 1.1 The Big Picture (One-Line Summary)

```
Historical Locations → Embeddings → LSTM → Location Prediction
```

### 1.2 Simple Block Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    LSTM MODEL (Simplified)                       │
│                                                                 │
│   ┌─────────┐     ┌─────────┐     ┌─────────┐     ┌─────────┐  │
│   │  Input  │ ──► │ Embed   │ ──► │  LSTM   │ ──► │ Output  │  │
│   │ Sequence│     │ Layer   │     │ Layers  │     │ Layer   │  │
│   └─────────┘     └─────────┘     └─────────┘     └─────────┘  │
│                                                       │         │
│                                                       ▼         │
│                                               [1187 Locations]  │
│                                                 Probabilities   │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Real-World Analogy

```
┌─────────────────────────────────────────────────────────────────┐
│                    HOW THE MODEL THINKS                          │
│                                                                 │
│   Imagine reading a person's diary of places they visited:      │
│                                                                 │
│   Day 1: "I went to HOME, then WORK, then GYM"                  │
│   Day 2: "I went to HOME, then COFFEE SHOP, then WORK"          │
│   Day 3: "I went to HOME, then WORK, then ?"                    │
│                                                                 │
│   The model reads the sequence, remembers patterns,             │
│   and predicts: "GYM" (similar to Day 1 pattern)                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Moderate Detail Diagrams

### 2.1 Model Architecture with Dimensions

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LSTM MODEL ARCHITECTURE                           │
│                                                                       │
│  INPUT                                                                │
│  ─────                                                                │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ │
│  │ Location IDs │ │  Time Slots  │ │   Weekdays   │ │  Durations   │ │
│  │  [seq, B]    │ │  [seq, B]    │ │  [seq, B]    │ │  [seq, B]    │ │
│  │  e.g.[5,32]  │ │  e.g.[5,32]  │ │  e.g.[5,32]  │ │  e.g.[5,32]  │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ │
│         │                │                │                │         │
│         ▼                ▼                ▼                ▼         │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │              EMBEDDING LAYER (AllEmbeddingLSTM)              │   │
│  │                                                              │   │
│  │  loc_emb(1187,32) + hour_emb(24,32) + minute_emb(4,32) +    │   │
│  │  weekday_emb(7,32) + duration_emb(96,32)                    │   │
│  │                                                              │   │
│  │                    Output: [seq, B, 32]                      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                    LSTM ENCODER                               │   │
│  │                                                              │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  pack_padded_sequence (handle variable lengths)     │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                         │                                    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  LSTM Layer 1: input=32, hidden=128                 │    │   │
│  │  │  LSTM Layer 2: input=128, hidden=128                │    │   │
│  │  │  (with dropout=0.2 between layers)                  │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                         │                                    │   │
│  │  ┌─────────────────────────────────────────────────────┐    │   │
│  │  │  pad_packed_sequence + gather last valid output     │    │   │
│  │  └─────────────────────────────────────────────────────┘    │   │
│  │                                                              │   │
│  │                    Output: [B, 128]                          │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   FULLY CONNECTED OUTPUT                      │   │
│  │                                                              │   │
│  │  Input + User_Emb(46,128) ──► Residual Block ──► Linear     │   │
│  │                                                              │   │
│  │                    Output: [B, 1187]                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                              │                                       │
│                              ▼                                       │
│                        PREDICTION                                    │
│                     argmax → Location ID                             │
└──────────────────────────────────────────────────────────────────────┘
```

### 2.2 Information Flow Through Time

```
┌─────────────────────────────────────────────────────────────────────┐
│                 SEQUENCE PROCESSING VISUALIZATION                    │
│                                                                     │
│  Input Sequence: [Home, Work, Gym, Coffee, Work]                    │
│  Target: Home (next location)                                       │
│                                                                     │
│  Time Step:   t=1      t=2      t=3       t=4      t=5             │
│               │        │        │         │        │               │
│  Location:   Home → Work  →  Gym  → Coffee → Work                  │
│               │        │        │         │        │               │
│               ▼        ▼        ▼         ▼        ▼               │
│  ┌─────┐   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                     │
│  │Embed│   │Embed│  │Embed│  │Embed│  │Embed│                     │
│  │32-d │   │32-d │  │32-d │  │32-d │  │32-d │                     │
│  └──┬──┘   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘                     │
│     │         │        │        │        │                         │
│     ▼         ▼        ▼        ▼        ▼                         │
│  ┌─────┐   ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐                     │
│  │LSTM │──►│LSTM │─►│LSTM │─►│LSTM │─►│LSTM │                     │
│  │Cell │   │Cell │  │Cell │  │Cell │  │Cell │                     │
│  │     │   │     │  │     │  │     │  │     │──► h5 (128-d)       │
│  └──┬──┘   └──┬──┘  └──┬──┘  └──┬──┘  └──┬──┘      │              │
│     │         │        │        │        │          │              │
│     h1        h2       h3       h4       h5         │              │
│   (128-d)  (128-d)  (128-d)  (128-d)  (128-d)      │              │
│                                                     ▼              │
│                                            ┌──────────────┐        │
│                                            │ FC + User Emb│        │
│                                            │              │        │
│                                            │ Output:      │        │
│                                            │ [1187 probs] │        │
│                                            └──────────────┘        │
│                                                     │              │
│                                                     ▼              │
│                                            Prediction: Home        │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.3 Embedding Combination

```
┌─────────────────────────────────────────────────────────────────────┐
│                   EMBEDDING COMBINATION (Additive)                   │
│                                                                     │
│  For a single location visit: Work at 9:00 AM Monday, stayed 4 hrs │
│                                                                     │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                                                             │   │
│  │   Location "Work" (ID: 45)                                  │   │
│  │   ─────────────────────────                                 │   │
│  │   emb_loc[45] = [0.23, -0.11, 0.45, ..., 0.18]  (32 dims)  │   │
│  │                                                             │   │
│  │                          +                                  │   │
│  │                                                             │   │
│  │   Time "9:00 AM"                                            │   │
│  │   ───────────────                                           │   │
│  │   hour_emb[9] + minute_emb[0] + weekday_emb[0]             │   │
│  │   = [0.12, -0.34, 0.22, ..., 0.08]  (32 dims)              │   │
│  │                                                             │   │
│  │                          +                                  │   │
│  │                                                             │   │
│  │   Duration "4 hours" (bucket 8: 4*60/30=8)                 │   │
│  │   ─────────────────────────────────────────                 │   │
│  │   duration_emb[8] = [0.08, 0.15, -0.11, ..., 0.22] (32d)   │   │
│  │                                                             │   │
│  │                          =                                  │   │
│  │                                                             │   │
│  │   Combined Embedding                                        │   │
│  │   ──────────────────                                        │   │
│  │   [0.43, -0.30, 0.56, ..., 0.48]  (32 dims)                │   │
│  │                                                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Benefits of Additive Combination:                                  │
│  • Same dimensionality (no size increase)                          │
│  • Features can reinforce or cancel each other                     │
│  • Model can learn which dimensions represent which features       │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Detailed Component Diagrams

### 3.1 AllEmbeddingLSTM Component

```
┌──────────────────────────────────────────────────────────────────────┐
│                    AllEmbeddingLSTM (Detailed)                        │
│                                                                       │
│  INPUTS:                                                              │
│  ┌──────────────┐                                                     │
│  │ src          │ Location IDs [seq_len, batch_size]                 │
│  │ context_dict │ Dictionary containing:                              │
│  │   - time     │   Time slots (0-95) [seq_len, batch]               │
│  │   - weekday  │   Day of week (0-6) [seq_len, batch]               │
│  │   - duration │   Duration buckets [seq_len, batch]                │
│  └──────────────┘                                                     │
│                                                                       │
│  EMBEDDING TABLES:                                                    │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  emb_loc:      Embedding(1187, 32)    37,984 params           │  │
│  │                                                                │  │
│  │  TemporalEmbedding:                                           │  │
│  │    ├─ hour_embed:    Embedding(24, 32)      768 params        │  │
│  │    ├─ minute_embed:  Embedding(4, 32)       128 params        │  │
│  │    └─ weekday_embed: Embedding(7, 32)       224 params        │  │
│  │                                                                │  │
│  │  emb_duration: Embedding(96, 32)     3,072 params             │  │
│  │                                                                │  │
│  │  TOTAL: 42,176 parameters                                     │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  FORWARD PASS:                                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │   src ──────────────► emb_loc(src) ───────────┐               │  │
│  │                            │                   │               │  │
│  │                            │                   │               │  │
│  │   time ─────► hour_embed ─┬─► [+] ────────────│──► [+] ──┐    │  │
│  │              minute_embed ─┘                   │          │    │  │
│  │   weekday ──► weekday_embed ──────────────────┘          │    │  │
│  │                                                           │    │  │
│  │   duration ──► emb_duration ─────────────────────────────│──► [+]│
│  │                                                           │    │  │
│  │                                                           │    │  │
│  │                                                     Dropout(0.1)│ │
│  │                                                           │    │  │
│  │                                                           ▼    │  │
│  │                                               Output [seq,B,32]│  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.2 LSTM Layer Stack

```
┌──────────────────────────────────────────────────────────────────────┐
│                        LSTM ENCODER (2 Layers)                        │
│                                                                       │
│  INPUT: Packed Sequence [total_valid_elements, 32]                   │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                       LSTM LAYER 1                             │  │
│  │                                                                │  │
│  │  Configuration:                                                │  │
│  │  ├─ input_size:  32                                           │  │
│  │  ├─ hidden_size: 128                                          │  │
│  │  └─ bidirectional: False                                      │  │
│  │                                                                │  │
│  │  Parameters:                                                   │  │
│  │  ├─ weight_ih_l0: [512, 32]   (4 * 128 * 32)                  │  │
│  │  ├─ weight_hh_l0: [512, 128]  (4 * 128 * 128)                 │  │
│  │  ├─ bias_ih_l0:   [512]       (4 * 128)                       │  │
│  │  └─ bias_hh_l0:   [512]       (4 * 128)                       │  │
│  │                                                                │  │
│  │  Total Layer 1: 82,944 params                                 │  │
│  │                                                                │  │
│  │  Output: [seq, batch, 128]                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                       Dropout(0.2)                                    │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                       LSTM LAYER 2                             │  │
│  │                                                                │  │
│  │  Configuration:                                                │  │
│  │  ├─ input_size:  128                                          │  │
│  │  ├─ hidden_size: 128                                          │  │
│  │  └─ bidirectional: False                                      │  │
│  │                                                                │  │
│  │  Parameters:                                                   │  │
│  │  ├─ weight_ih_l1: [512, 128]  (4 * 128 * 128)                 │  │
│  │  ├─ weight_hh_l1: [512, 128]  (4 * 128 * 128)                 │  │
│  │  ├─ bias_ih_l1:   [512]       (4 * 128)                       │  │
│  │  └─ bias_hh_l1:   [512]       (4 * 128)                       │  │
│  │                                                                │  │
│  │  Total Layer 2: 132,096 params                                │  │
│  │                                                                │  │
│  │  Output: [seq, batch, 128]                                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  OUTPUT: [seq, batch, 128] + hidden states (h_n, c_n)               │
│                                                                       │
│  TOTAL LSTM PARAMETERS: 215,040                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 FullyConnected Output Layer

```
┌──────────────────────────────────────────────────────────────────────┐
│                    FULLY CONNECTED OUTPUT LAYER                       │
│                                                                       │
│  INPUT: LSTM output [batch, 128]                                     │
│         User IDs    [batch]                                          │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    USER EMBEDDING                              │  │
│  │                                                                │  │
│  │  emb_user: Embedding(46, 128)                                 │  │
│  │  Parameters: 5,888                                            │  │
│  │                                                                │  │
│  │  user_emb = emb_user(user_ids)  → [batch, 128]                │  │
│  │                                                                │  │
│  │  combined = lstm_output + user_emb  → [batch, 128]            │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                       Dropout(0.1)                                    │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    RESIDUAL BLOCK                              │  │
│  │                                                                │  │
│  │  x ──────────────────────────────────────────────────┐        │  │
│  │  │                                                   │        │  │
│  │  │     ┌──────────┐     ┌──────────┐     ┌────────┐  │        │  │
│  │  └────►│ Linear   │────►│   ReLU   │────►│Dropout │──│        │  │
│  │        │ 128→256  │     │          │     │  0.2   │  │        │  │
│  │        └──────────┘     └──────────┘     └────────┘  │        │  │
│  │                                              │       │        │  │
│  │        ┌──────────┐     ┌──────────┐        │       │        │  │
│  │        │ Linear   │◄────│ Dropout  │◄───────┘       │        │  │
│  │        │ 256→128  │     │   0.2    │                │        │  │
│  │        └────┬─────┘     └──────────┘                │        │  │
│  │             │                                       │        │  │
│  │             └────────────────► [+] ◄────────────────┘        │  │
│  │                                 │                            │  │
│  │                          BatchNorm1d(128)                    │  │
│  │                                 │                            │  │
│  │                         Output: [batch, 128]                 │  │
│  │                                                              │  │
│  │  Parameters:                                                 │  │
│  │  ├─ linear1: 128*256+256 = 33,024                           │  │
│  │  ├─ linear2: 256*128+128 = 32,896                           │  │
│  │  └─ norm:    256                                            │  │
│  │  Total: 66,176                                              │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                    OUTPUT PROJECTION                           │  │
│  │                                                                │  │
│  │  fc_loc: Linear(128, 1187)                                    │  │
│  │  Parameters: 128*1187+1187 = 152,123                          │  │
│  │                                                                │  │
│  │  Output: [batch, 1187] (logits for each location)             │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  TOTAL FC PARAMETERS: 224,187                                        │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 4. Data Flow Visualizations

### 4.1 Variable-Length Sequence Handling

```
┌──────────────────────────────────────────────────────────────────────┐
│              HANDLING VARIABLE-LENGTH SEQUENCES                       │
│                                                                       │
│  ORIGINAL SEQUENCES (different lengths):                              │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Seq 1: [A, B, C]              (length 3)                       │ │
│  │  Seq 2: [D, E, F, G, H]        (length 5)                       │ │
│  │  Seq 3: [I, J, K, L]           (length 4)                       │ │
│  │  Seq 4: [M, N]                 (length 2)                       │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  AFTER PADDING (pad to max_len=5):                                   │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Position:    0     1     2     3     4                         │ │
│  │  ─────────────────────────────────────                          │ │
│  │  Seq 1:    [ A  ,  B  ,  C  , PAD , PAD ]                       │ │
│  │  Seq 2:    [ D  ,  E  ,  F  ,  G  ,  H  ]                       │ │
│  │  Seq 3:    [ I  ,  J  ,  K  ,  L  , PAD ]                       │ │
│  │  Seq 4:    [ M  ,  N  , PAD , PAD , PAD ]                       │ │
│  │                                                                  │ │
│  │  Tensor shape: [5, 4] (seq_len, batch)                          │ │
│  │  lengths: [3, 5, 4, 2]                                          │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  AFTER PACKING (removes padding for computation):                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  PackedSequence.data:                                           │ │
│  │  ┌───────────────────────────────────────────────────────────┐  │ │
│  │  │ [A, D, I, M] │ [B, E, J, N] │ [C, F, K] │ [G, L] │ [H] │  │ │
│  │  └───────────────────────────────────────────────────────────┘  │ │
│  │    timestep 0     timestep 1    timestep 2   t=3     t=4        │ │
│  │    batch_size=4   batch_size=4  batch_size=3 bs=2    bs=1       │ │
│  │                                                                  │ │
│  │  batch_sizes: [4, 4, 3, 2, 1]                                   │ │
│  │  Total elements: 3+5+4+2 = 14                                   │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                       LSTM Processing                                 │
│                     (efficient: no pad computation)                   │
│                              │                                        │
│                              ▼                                        │
│  AFTER UNPACKING + GATHER LAST:                                      │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  lengths: [3, 5, 4, 2]                                          │ │
│  │  indices: [2, 4, 3, 1]  (0-indexed: length - 1)                 │ │
│  │                                                                  │ │
│  │  Gather from unpacked output:                                   │ │
│  │    output[2, 0, :] ──► last hidden for seq 1 (position C)       │ │
│  │    output[4, 1, :] ──► last hidden for seq 2 (position H)       │ │
│  │    output[3, 2, :] ──► last hidden for seq 3 (position L)       │ │
│  │    output[1, 3, :] ──► last hidden for seq 4 (position N)       │ │
│  │                                                                  │ │
│  │  Result: [4, 128] (one 128-dim vector per sequence)             │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

### 4.2 Batch Processing Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                    BATCH PROCESSING PIPELINE                          │
│                                                                       │
│  RAW DATA (.pk file)                                                 │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │  Sample 0: {"X": [102,45,103], "Y": 89, "user_X": [5,5,5], ...} │ │
│  │  Sample 1: {"X": [200,45,89,102,45], "Y": 102, ...}             │ │
│  │  Sample 2: {"X": [33,77,102,45], "Y": 200, ...}                 │ │
│  │  ...                                                            │ │
│  │  (7424 training samples)                                        │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                        DataLoader                                     │
│                      (batch_size=32)                                  │
│                              │                                        │
│                              ▼                                        │
│  COLLATE FUNCTION                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  1. Collect 32 samples into lists                               │ │
│  │  2. Record sequence lengths: [3, 5, 4, 2, ...]                  │ │
│  │  3. Pad sequences to max length in batch                        │ │
│  │  4. Stack into tensors                                          │ │
│  │                                                                  │ │
│  │  Output:                                                        │ │
│  │    x:      [max_len, 32]      (padded location sequences)       │ │
│  │    y:      [32]               (target locations)                │ │
│  │    x_dict: {                                                    │ │
│  │      "len":      [32],        (actual lengths)                  │ │
│  │      "user":     [32],        (user IDs)                        │ │
│  │      "time":     [max_len, 32],                                 │ │
│  │      "weekday":  [max_len, 32],                                 │ │
│  │      "duration": [max_len, 32],                                 │ │
│  │    }                                                            │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                        send_to_device                                 │
│                      (move to GPU if available)                       │
│                              │                                        │
│                              ▼                                        │
│  MODEL FORWARD PASS                                                  │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  logits = model(x, x_dict, device)                              │ │
│  │                                                                  │ │
│  │  logits shape: [32, 1187]                                       │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                              │                                        │
│                              ▼                                        │
│  LOSS COMPUTATION                                                    │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │                                                                  │ │
│  │  loss = CrossEntropyLoss(logits, y)                             │ │
│  │                                                                  │ │
│  │  Example: batch average loss = 3.5                              │ │
│  │                                                                  │ │
│  └─────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 5. LSTM Cell Internals

### 5.1 Single LSTM Cell Detailed

```
┌──────────────────────────────────────────────────────────────────────┐
│                        LSTM CELL INTERNALS                            │
│                                                                       │
│  INPUTS at time t:                                                   │
│    x_t:     current input        [batch, 32]                         │
│    h_{t-1}: previous hidden      [batch, 128]                        │
│    c_{t-1}: previous cell state  [batch, 128]                        │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │                     ┌──────────────────┐                       │  │
│  │    c_{t-1} ────────►│    Element-wise  │                       │  │
│  │                     │   Multiplication │                       │  │
│  │                     │    (forget)      │                       │  │
│  │         f_t ───────►│                  │                       │  │
│  │                     └────────┬─────────┘                       │  │
│  │                              │                                 │  │
│  │                              ▼                                 │  │
│  │                     ┌───────────────────┐                      │  │
│  │                     │    Element-wise   │                      │  │
│  │                     │     Addition      │────────► c_t         │  │
│  │                     │                   │                      │  │
│  │    i_t ◄───┐        └────────┬──────────┘                      │  │
│  │            │                 ▲                                 │  │
│  │   ┌────────┴───────┐        │                                 │  │
│  │   │  Element-wise  │        │                                 │  │
│  │   │ Multiplication │◄───────┤                                 │  │
│  │   └────────────────┘        │                                 │  │
│  │            ▲                │                                 │  │
│  │            │                │                                 │  │
│  │          c̃_t               │                                 │  │
│  │                             │                                 │  │
│  │                    ┌────────┴──────────┐                      │  │
│  │                    │      tanh         │                      │  │
│  │                    └────────┬──────────┘                      │  │
│  │                             │                                 │  │
│  │                    ┌────────┴──────────┐                      │  │
│  │                    │   Element-wise    │                      │  │
│  │    o_t ───────────►│  Multiplication   │────────► h_t         │  │
│  │                    └───────────────────┘                      │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  GATE COMPUTATIONS:                                                  │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  Concatenate inputs: [h_{t-1}, x_t] → [batch, 160]            │  │
│  │                                                                │  │
│  │  Forget gate:                                                 │  │
│  │    f_t = σ(W_f · [h_{t-1}, x_t] + b_f)                        │  │
│  │    "What to forget from old cell state"                       │  │
│  │    Output: [batch, 128], values in (0, 1)                     │  │
│  │                                                                │  │
│  │  Input gate:                                                  │  │
│  │    i_t = σ(W_i · [h_{t-1}, x_t] + b_i)                        │  │
│  │    "How much new info to add"                                 │  │
│  │    Output: [batch, 128], values in (0, 1)                     │  │
│  │                                                                │  │
│  │  Candidate cell state:                                        │  │
│  │    c̃_t = tanh(W_c · [h_{t-1}, x_t] + b_c)                    │  │
│  │    "What new info is available"                               │  │
│  │    Output: [batch, 128], values in (-1, 1)                    │  │
│  │                                                                │  │
│  │  Output gate:                                                 │  │
│  │    o_t = σ(W_o · [h_{t-1}, x_t] + b_o)                        │  │
│  │    "What to output"                                           │  │
│  │    Output: [batch, 128], values in (0, 1)                     │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  STATE UPDATES:                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │  Cell state update:                                           │  │
│  │    c_t = f_t ⊙ c_{t-1} + i_t ⊙ c̃_t                           │  │
│  │    "Forget old + add new"                                     │  │
│  │                                                                │  │
│  │  Hidden state output:                                         │  │
│  │    h_t = o_t ⊙ tanh(c_t)                                      │  │
│  │    "Filter cell state for output"                             │  │
│  │                                                                │  │
│  │  ⊙ = element-wise multiplication (Hadamard product)           │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 5.2 LSTM Through Time

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LSTM UNROLLED THROUGH TIME                        │
│                                                                       │
│  Sequence: [Home, Work, Gym] → Predict: Restaurant                   │
│                                                                       │
│       t=1 (Home)         t=2 (Work)          t=3 (Gym)               │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐       │
│  │                 │  │                 │  │                 │       │
│  │  c_0 = 0 ─────────► c_1 ───────────────► c_2 ────────────► c_3   │
│  │       ↑         │  │      ↑         │  │      ↑         │       │
│  │       │         │  │      │         │  │      │         │       │
│  │  ┌────┴────┐    │  │ ┌────┴────┐    │  │ ┌────┴────┐    │       │
│  │  │  LSTM   │    │  │ │  LSTM   │    │  │ │  LSTM   │    │       │
│  │  │  Cell   │    │  │ │  Cell   │    │  │ │  Cell   │    │       │
│  │  └────┬────┘    │  │ └────┬────┘    │  │ └────┬────┘    │       │
│  │       │         │  │      │         │  │      │         │       │
│  │       ↓         │  │      ↓         │  │      ↓         │       │
│  │  h_0 = 0 ─────────► h_1 ───────────────► h_2 ────────────► h_3   │
│  │                 │  │                 │  │                 │  │    │
│  │       ↑         │  │      ↑         │  │      ↑         │  │    │
│  └───────┼─────────┘  └──────┼─────────┘  └──────┼─────────┘  │    │
│          │                   │                   │             │    │
│     x_1 (Home)          x_2 (Work)          x_3 (Gym)         │    │
│     [32-dim]            [32-dim]            [32-dim]          ▼    │
│                                                         FC Layer   │
│                                                              │     │
│                                                              ▼     │
│                                                      [1187 logits] │
│                                                              │     │
│                                                              ▼     │
│                                                    Prediction: Restaurant
│                                                                       │
│  What h_3 "remembers":                                               │
│  - Recent visit: Gym (strong signal)                                 │
│  - Earlier: Work, Home (weaker but preserved)                        │
│  - Time patterns: morning→afternoon→evening                          │
│  - User behavior encoded in activations                              │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 6. Training Pipeline Visualization

### 6.1 Complete Training Loop

```
┌──────────────────────────────────────────────────────────────────────┐
│                      COMPLETE TRAINING PIPELINE                       │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                         INITIALIZATION                         │  │
│  │                                                                │  │
│  │  1. Load config (YAML)                                        │  │
│  │  2. Set random seeds (42)                                     │  │
│  │  3. Initialize model                                          │  │
│  │  4. Create data loaders                                       │  │
│  │  5. Setup optimizer (Adam)                                    │  │
│  │  6. Setup schedulers (warmup + step)                          │  │
│  │  7. Initialize early stopping                                 │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                         EPOCH LOOP                             │  │
│  │  for epoch in range(max_epochs):                              │  │
│  │                                                                │  │
│  │    ┌──────────────────────────────────────────────────────┐   │  │
│  │    │                  TRAINING PHASE                      │   │  │
│  │    │  model.train()                                       │   │  │
│  │    │                                                      │   │  │
│  │    │  for batch in train_loader:  (232 batches)          │   │  │
│  │    │    │                                                 │   │  │
│  │    │    ▼                                                 │   │  │
│  │    │  ┌──────────────────────────────────────────┐       │   │  │
│  │    │  │ 1. Forward: logits = model(x, x_dict)   │       │   │  │
│  │    │  │ 2. Loss: CEL(logits, y)                 │       │   │  │
│  │    │  │ 3. Backward: loss.backward()            │       │   │  │
│  │    │  │ 4. Clip: clip_grad_norm_(params, 1.0)   │       │   │  │
│  │    │  │ 5. Step: optimizer.step()               │       │   │  │
│  │    │  │ 6. Schedule: scheduler.step()           │       │   │  │
│  │    │  │ 7. Log metrics every print_step         │       │   │  │
│  │    │  └──────────────────────────────────────────┘       │   │  │
│  │    │                                                      │   │  │
│  │    └──────────────────────────────────────────────────────┘   │  │
│  │                              │                                 │  │
│  │                              ▼                                 │  │
│  │    ┌──────────────────────────────────────────────────────┐   │  │
│  │    │                 VALIDATION PHASE                     │   │  │
│  │    │  model.eval()                                        │   │  │
│  │    │  with torch.no_grad():                               │   │  │
│  │    │                                                      │   │  │
│  │    │  for batch in val_loader:  (105 batches)            │   │  │
│  │    │    Compute: loss, acc@1, acc@5, acc@10, MRR, NDCG   │   │  │
│  │    │                                                      │   │  │
│  │    │  val_loss = mean(batch_losses)                      │   │  │
│  │    │                                                      │   │  │
│  │    └──────────────────────────────────────────────────────┘   │  │
│  │                              │                                 │  │
│  │                              ▼                                 │  │
│  │    ┌──────────────────────────────────────────────────────┐   │  │
│  │    │                 EARLY STOPPING CHECK                 │   │  │
│  │    │                                                      │   │  │
│  │    │  if val_loss < best_val_loss - delta:               │   │  │
│  │    │    ✓ Save checkpoint                                │   │  │
│  │    │    ✓ Reset patience counter                         │   │  │
│  │    │    ✓ Update best_val_loss                           │   │  │
│  │    │  else:                                               │   │  │
│  │    │    ✗ Increment patience counter                     │   │  │
│  │    │    if counter >= patience (3):                      │   │  │
│  │    │      if num_stops < 3:                              │   │  │
│  │    │        → Reduce LR by 0.1x                          │   │  │
│  │    │        → Load best checkpoint                       │   │  │
│  │    │        → Reset counter                              │   │  │
│  │    │      else:                                          │   │  │
│  │    │        → STOP TRAINING                              │   │  │
│  │    │                                                      │   │  │
│  │    └──────────────────────────────────────────────────────┘   │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                              │                                        │
│                              ▼                                        │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                         TEST PHASE                             │  │
│  │                                                                │  │
│  │  1. Load best checkpoint                                      │  │
│  │  2. model.eval()                                              │  │
│  │  3. for batch in test_loader: (110 batches)                   │  │
│  │       Compute all metrics                                     │  │
│  │  4. Save results to test_results.json                         │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.2 Learning Rate Schedule

```
┌──────────────────────────────────────────────────────────────────────┐
│                     LEARNING RATE SCHEDULE                            │
│                                                                       │
│  Learning                                                            │
│  Rate                                                                │
│    │                                                                 │
│    │                                                                 │
│  0.001 ├──────────────────────●●●●●●●●●●                             │
│        │                    ●●            ●●●                        │
│        │                  ●●                 ●●●                     │
│        │                ●●                      ●●●                  │
│        │              ●●                           ●●●               │
│        │            ●●                                ●●●            │
│  0.0005├          ●●                                    ●●●          │
│        │        ●●                                         ●         │
│        │      ●●                                            │        │
│        │    ●●                                              │ LR     │
│        │  ●●                                                │ drop   │
│  0.0001├●●                                                  ▼        │
│        │●                                              ●●●●●●●●      │
│        │                                                     ▼       │
│  0.00001├──────────────────────────────────────────────────●●●●●●    │
│        │                                                         ▼   │
│        └─────────┬─────────┬────────────────────────────────┬────►   │
│                  │         │                                │        │
│                Warmup    Peak                         Early Stop     │
│               (2 epochs)  LR                          triggers       │
│                                                                       │
│  Phase 1: Linear Warmup (0 → 0.001)                                  │
│           Steps: 0 to 464 (2 epochs × 232 batches)                   │
│                                                                       │
│  Phase 2: Linear Decay (0.001 → ~0)                                  │
│           Steps: 464 to 11600                                        │
│                                                                       │
│  Phase 3: Step Decay (on early stopping)                             │
│           LR × 0.1 each time early stopping triggers                 │
│                                                                       │
└──────────────────────────────────────────────────────────────────────┘
```

### 6.3 Parameter Count Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                    PARAMETER COUNT BREAKDOWN                          │
│                       (GeoLife Configuration)                         │
│                                                                       │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │                                                                │  │
│  │   EMBEDDING LAYER (AllEmbeddingLSTM)                          │  │
│  │   ─────────────────────────────────                           │  │
│  │   emb_loc:        1187 × 32   =  37,984                       │  │
│  │   hour_embed:       24 × 32   =     768                       │  │
│  │   minute_embed:      4 × 32   =     128                       │  │
│  │   weekday_embed:     7 × 32   =     224                       │  │
│  │   emb_duration:     96 × 32   =   3,072                       │  │
│  │   ───────────────────────────────────────                     │  │
│  │   Subtotal:                      42,176                       │  │
│  │                                                                │  │
│  │   LSTM ENCODER                                                │  │
│  │   ────────────                                                │  │
│  │   Layer 1:    4×(32×128 + 128×128 + 128 + 128) = 82,944      │  │
│  │   Layer 2:    4×(128×128 + 128×128 + 128 + 128) = 132,096    │  │
│  │   ───────────────────────────────────────                     │  │
│  │   Subtotal:                     215,040                       │  │
│  │                                                                │  │
│  │   LAYER NORM                                                  │  │
│  │   ──────────                                                  │  │
│  │   layer_norm:     128 × 2     =     256                       │  │
│  │                                                                │  │
│  │   FULLY CONNECTED (FC)                                        │  │
│  │   ────────────────────                                        │  │
│  │   emb_user:        46 × 128   =   5,888                       │  │
│  │   emb_dropout:              (no params)                       │  │
│  │   linear1:        128 × 256 + 256 = 33,024                    │  │
│  │   linear2:        256 × 128 + 128 = 32,896                    │  │
│  │   norm1:          128 × 2     =     256                       │  │
│  │   fc_loc:         128 × 1187 + 1187 = 152,123                 │  │
│  │   ───────────────────────────────────────                     │  │
│  │   Subtotal:                     224,187                       │  │
│  │                                                                │  │
│  │   ═══════════════════════════════════════                     │  │
│  │   TOTAL TRAINABLE PARAMETERS:   481,659                       │  │
│  │   ═══════════════════════════════════════                     │  │
│  │                                                                │  │
│  │   Memory estimate (FP32): ~1.84 MB                            │  │
│  │                                                                │  │
│  └────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  For DIY dataset (larger):                                           │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │   emb_loc:        7038 × 96   = 675,648                       │  │
│  │   LSTM (larger):              ~ 850,000                       │  │
│  │   FC (larger):                ~ 1,400,000                     │  │
│  │   ───────────────────────────────────────                     │  │
│  │   TOTAL:                      ~2,850,000                      │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This document provides visual representations of:

1. **Simplified diagrams** for quick understanding
2. **Moderate detail diagrams** showing dimensions and flow
3. **Detailed component diagrams** with parameter counts
4. **Data flow visualizations** including sequence handling
5. **LSTM internals** showing gate operations
6. **Training pipeline** with learning rate schedules

These diagrams complement the technical documentation and code walkthrough, providing visual anchors for understanding the model architecture.

---

**Document Version**: 1.0
**Last Updated**: January 2026
