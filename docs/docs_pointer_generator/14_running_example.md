# End-to-End Worked Example

## Table of Contents
1. [Example Setup](#example-setup)
2. [Data Preparation](#data-preparation)
3. [Encoding Phase](#encoding-phase)
4. [Decoding Phase](#decoding-phase)
5. [Complete Numerical Example](#complete-numerical-example)
6. [Training vs Inference](#training-vs-inference)

---

## Example Setup

Throughout this document, we use a **consistent example** to trace every step of the pointer-generator network.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          EXAMPLE SETUP                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   SOURCE ARTICLE:                                                                 │
│   ───────────────                                                                 │
│   "Germany beat Argentina in the World Cup final . Mario Götze scored ."        │
│                                                                                   │
│                                                                                   │
│   TARGET SUMMARY:                                                                 │
│   ───────────────                                                                 │
│   "Germany won the Cup . Götze scored the winner ."                             │
│                                                                                   │
│                                                                                   │
│   VOCABULARY (sample - showing relevant words):                                   │
│   ────────────────────────────────────────────────                                │
│                                                                                   │
│   ID      Word                                                                    │
│   ──      ────                                                                    │
│   0       [PAD]                                                                   │
│   1       [UNK]                                                                   │
│   2       [START]                                                                 │
│   3       [STOP]                                                                  │
│   4       the                                                                     │
│   5       .                                                                       │
│   6       in                                                                      │
│   100     germany                                                                 │
│   101     beat                                                                    │
│   102     argentina                                                               │
│   103     world                                                                   │
│   104     cup                                                                     │
│   105     final                                                                   │
│   106     won                                                                     │
│   107     scored                                                                  │
│   108     winner                                                                  │
│                                                                                   │
│   NOT IN VOCABULARY (OOV):                                                        │
│   • mario                                                                         │
│   • götze                                                                         │
│                                                                                   │
│                                                                                   │
│   MODEL HYPERPARAMETERS:                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   vocab_size = 50000                                                             │
│   emb_dim = 128                                                                  │
│   hidden_dim = 256                                                               │
│   batch_size = 1 (for this example)                                              │
│   pointer_gen = True                                                             │
│   coverage = True                                                                │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Preparation

### Step 1: Tokenization

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                          TOKENIZATION                                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ARTICLE TOKENIZATION:                                                           │
│   ─────────────────────                                                           │
│                                                                                   │
│   Raw: "Germany beat Argentina in the World Cup final . Mario Götze scored ."   │
│                                                                                   │
│   Tokens:                                                                         │
│   Position:  0        1      2          3    4    5      6     7      8          │
│   Word:      germany  beat   argentina  in   the  world  cup   final  .          │
│                                                                                   │
│   Position:  9      10     11                                                     │
│   Word:      mario  götze  scored  .                                             │
│                                                                                   │
│   article_words = ["germany", "beat", "argentina", "in", "the",                 │
│                    "world", "cup", "final", ".", "mario",                        │
│                    "götze", "scored", "."]                                       │
│                                                                                   │
│                                                                                   │
│   ABSTRACT TOKENIZATION:                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   Raw: "Germany won the Cup . Götze scored the winner ."                        │
│                                                                                   │
│   abstract_words = ["germany", "won", "the", "cup", ".",                        │
│                     "götze", "scored", "the", "winner", "."]                    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Convert to IDs

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       CONVERT TO IDS                                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ENCODER INPUT (enc_input) - OOV → [UNK]:                                       │
│   ─────────────────────────────────────────                                       │
│                                                                                   │
│   Word:     germany  beat  argentina  in  the  world  cup  final  .             │
│   ID:         100    101     102      6    4    103   104   105   5             │
│                                                                                   │
│   Word:     mario  götze  scored  .                                              │
│   ID:         1      1     107    5                                              │
│              ↑      ↑                                                             │
│            [UNK]  [UNK]                                                          │
│                                                                                   │
│   enc_input = [100, 101, 102, 6, 4, 103, 104, 105, 5, 1, 1, 107, 5]             │
│               Length: 13                                                          │
│                                                                                   │
│                                                                                   │
│   ENCODER INPUT EXTENDED (enc_input_extend_vocab) - OOV → temp IDs:              │
│   ─────────────────────────────────────────────────────────────────               │
│                                                                                   │
│   article_oovs = ["mario", "götze"]   # OOV words in order                      │
│   mario → 50000 (vocab_size + 0)                                                │
│   götze → 50001 (vocab_size + 1)                                                │
│                                                                                   │
│   enc_input_extend_vocab = [100, 101, 102, 6, 4, 103, 104, 105, 5,              │
│                             50000, 50001, 107, 5]                                │
│                              ↑      ↑                                             │
│                         mario   götze (temporary IDs!)                           │
│                                                                                   │
│                                                                                   │
│   DECODER INPUT (dec_input):                                                      │
│   ──────────────────────────                                                      │
│                                                                                   │
│   Format: [START, word1, word2, ...]                                             │
│                                                                                   │
│   dec_input = [2, 100, 106, 4, 104, 5, 50001, 107, 4, 108, 5]                   │
│                ↑   ↑    ↑   ↑   ↑   ↑    ↑     ↑   ↑   ↑   ↑                    │
│             START ger- won the cup  . götze scor the win- .                     │
│                   many                      -ed     ner                          │
│                                                                                   │
│                                                                                   │
│   TARGET (target):                                                                │
│   ────────────────                                                                │
│                                                                                   │
│   Format: [word1, word2, ..., STOP]                                              │
│                                                                                   │
│   target = [100, 106, 4, 104, 5, 50001, 107, 4, 108, 5, 3]                      │
│             ↑    ↑   ↑   ↑   ↑    ↑     ↑   ↑   ↑   ↑  ↑                        │
│           ger- won the cup  . götze scor the win- . STOP                        │
│           many                      -ed     ner                                  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 3: Create Batch

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        CREATE BATCH                                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   For batch_size=1:                                                               │
│                                                                                   │
│   enc_batch:              [1, 13]                                                │
│   [[100, 101, 102, 6, 4, 103, 104, 105, 5, 1, 1, 107, 5]]                       │
│                                                                                   │
│   enc_lens:               [1]                                                     │
│   [13]                                                                            │
│                                                                                   │
│   enc_padding_mask:       [1, 13]                                                │
│   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]                                     │
│                                                                                   │
│   enc_batch_extend_vocab: [1, 13]                                                │
│   [[100, 101, 102, 6, 4, 103, 104, 105, 5, 50000, 50001, 107, 5]]               │
│                                                                                   │
│   max_art_oovs:           2  (mario, götze)                                     │
│                                                                                   │
│   dec_batch:              [1, 11]                                                │
│   [[2, 100, 106, 4, 104, 5, 50001, 107, 4, 108, 5]]                             │
│                                                                                   │
│   target_batch:           [1, 11]                                                │
│   [[100, 106, 4, 104, 5, 50001, 107, 4, 108, 5, 3]]                             │
│                                                                                   │
│   dec_padding_mask:       [1, 11]                                                │
│   [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]                                           │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Encoding Phase

### Step 4: Embedding Lookup

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       EMBEDDING LOOKUP                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   enc_batch: [100, 101, 102, 6, 4, 103, 104, 105, 5, 1, 1, 107, 5]              │
│                                                                                   │
│   embedding_matrix: [50000, 128]  (vocab_size × emb_dim)                        │
│                                                                                   │
│   emb_enc_inputs = embedding_matrix[enc_batch]                                   │
│   Shape: [1, 13, 128]  (batch × enc_len × emb_dim)                              │
│                                                                                   │
│   Visualized:                                                                     │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │  Position 0 (germany): [0.12, -0.34, 0.56, ..., 0.78]  ← 128 dims      │   │
│   │  Position 1 (beat):    [0.23, 0.45, -0.12, ..., -0.34]                  │   │
│   │  Position 2 (argentina):[0.34, -0.56, 0.78, ..., 0.12]                  │   │
│   │  ...                                                                     │   │
│   │  Position 9 (mario→UNK): [0.01, 0.02, 0.01, ..., 0.02]  ← UNK embed    │   │
│   │  Position 10 (götze→UNK):[0.01, 0.02, 0.01, ..., 0.02]  ← Same UNK!   │   │
│   │  ...                                                                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│   Note: OOV words (mario, götze) both get the same [UNK] embedding!             │
│   This is why we need the pointer mechanism to distinguish them.                 │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 5: Bidirectional LSTM Encoding

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    BIDIRECTIONAL LSTM ENCODING                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   INPUT: emb_enc_inputs [1, 13, 128]                                            │
│                                                                                   │
│   FORWARD LSTM:                                                                   │
│   ─────────────                                                                   │
│   Processes: position 0 → 1 → 2 → ... → 12                                      │
│                                                                                   │
│   h_fw_0 = LSTM(emb[0])                    # [1, 256]                           │
│   h_fw_1 = LSTM(emb[1], h_fw_0)            # [1, 256]                           │
│   h_fw_2 = LSTM(emb[2], h_fw_1)            # [1, 256]                           │
│   ...                                                                             │
│   h_fw_12 = LSTM(emb[12], h_fw_11)         # [1, 256] ← Final forward state    │
│                                                                                   │
│                                                                                   │
│   BACKWARD LSTM:                                                                  │
│   ──────────────                                                                  │
│   Processes: position 12 → 11 → 10 → ... → 0                                    │
│                                                                                   │
│   h_bw_12 = LSTM(emb[12])                  # [1, 256]                           │
│   h_bw_11 = LSTM(emb[11], h_bw_12)         # [1, 256]                           │
│   h_bw_10 = LSTM(emb[10], h_bw_11)         # [1, 256]                           │
│   ...                                                                             │
│   h_bw_0 = LSTM(emb[0], h_bw_1)            # [1, 256] ← Final backward state   │
│                                                                                   │
│                                                                                   │
│   CONCATENATE:                                                                    │
│   ────────────                                                                    │
│   For each position i:                                                            │
│   encoder_outputs[i] = [h_fw_i ; h_bw_i]   # [1, 512]                           │
│                                                                                   │
│   encoder_outputs shape: [1, 13, 512]  (batch × enc_len × 2*hidden_dim)        │
│                                                                                   │
│                                                                                   │
│   VISUALIZED:                                                                     │
│   ───────────                                                                     │
│                                                                                   │
│   Position:    0        1        2       ...      12                             │
│   Word:       germany   beat  argentina  ...      .                              │
│                                                                                   │
│   Forward:    →h_0     →h_1     →h_2     ...    →h_12                           │
│   Backward:   h_0←     h_1←     h_2←     ...    h_12←                           │
│                  ↓        ↓        ↓               ↓                             │
│   Concat:     [→;←]    [→;←]    [→;←]    ...    [→;←]                          │
│   Shape:     [1,512]  [1,512]  [1,512]   ...   [1,512]                          │
│                                                                                   │
│   Final encoder_outputs: [1, 13, 512]                                           │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 6: Reduce States

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        REDUCE STATES                                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   ENCODER FINAL STATES:                                                           │
│   ─────────────────────                                                           │
│                                                                                   │
│   fw_st.c: [1, 256]  (forward cell state at position 12)                        │
│   fw_st.h: [1, 256]  (forward hidden state at position 12)                      │
│   bw_st.c: [1, 256]  (backward cell state at position 0)                        │
│   bw_st.h: [1, 256]  (backward hidden state at position 0)                      │
│                                                                                   │
│                                                                                   │
│   REDUCE TO DECODER INITIAL STATE:                                                │
│   ────────────────────────────────                                                │
│                                                                                   │
│   Step 1: Concatenate                                                             │
│   old_c = [fw_st.c ; bw_st.c]  # [1, 512]                                       │
│   old_h = [fw_st.h ; bw_st.h]  # [1, 512]                                       │
│                                                                                   │
│   Step 2: Project with ReLU                                                       │
│   new_c = ReLU(old_c @ W_reduce_c + b_c)  # [1, 512] @ [512, 256] → [1, 256]   │
│   new_h = ReLU(old_h @ W_reduce_h + b_h)  # [1, 512] @ [512, 256] → [1, 256]   │
│                                                                                   │
│   dec_in_state = LSTMStateTuple(new_c, new_h)                                   │
│   Each component: [1, 256]                                                       │
│                                                                                   │
│                                                                                   │
│   NUMERICAL EXAMPLE:                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   Suppose:                                                                        │
│   fw_st.c = [0.5, -0.3, 0.8, ...]  (256 values)                                │
│   bw_st.c = [0.2, 0.4, -0.1, ...]  (256 values)                                │
│                                                                                   │
│   old_c = [0.5, -0.3, 0.8, ..., 0.2, 0.4, -0.1, ...]  (512 values)             │
│                                                                                   │
│   new_c = ReLU(old_c @ W_c + b_c)                                               │
│         = ReLU([0.34, -0.12, 0.56, ...])  (256 values)                          │
│         = [0.34, 0.0, 0.56, ...]  (negatives → 0)                               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Decoding Phase

### Step 7: Pre-compute Encoder Features

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   PRE-COMPUTE ENCODER FEATURES                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   encoder_outputs: [1, 13, 512]                                                  │
│                                                                                   │
│   Step 1: Expand dims for conv2d                                                 │
│   encoder_outputs_4d: [1, 13, 1, 512]                                           │
│                                                                                   │
│   Step 2: Apply W_h via conv2d                                                   │
│   W_h: [1, 1, 512, 512]                                                         │
│                                                                                   │
│   encoder_features = conv2d(encoder_outputs_4d, W_h)                            │
│   Shape: [1, 13, 1, 512]                                                        │
│                                                                                   │
│   This computes W_h × h_i for all encoder positions ONCE.                       │
│   Will be reused at every decoder step.                                         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 8: Decoder Step 1 (Generate "germany")

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│               DECODER STEP 1: Generate "germany"                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   INPUT: [START] token (ID=2)                                                    │
│   TARGET: "germany" (ID=100)                                                     │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.1: Get input embedding                                                        │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   inp = embedding_matrix[2]  # [START] embedding                                │
│   inp shape: [1, 128]                                                            │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.2: Input feeding (concat with context)                                       │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   context_vector (initial): zeros [1, 512]                                      │
│   x = [inp ; context_vector]                                                     │
│   x shape: [1, 128 + 512] = [1, 640]                                           │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.3: LSTM cell                                                                  │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   cell_output, state = LSTM(x, dec_in_state)                                    │
│   cell_output: [1, 256]                                                          │
│   state: (c: [1, 256], h: [1, 256])                                             │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.4: Compute attention                                                          │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   decoder_features = W_s × state.h + b                                          │
│   Shape: [1, 512]                                                                │
│                                                                                   │
│   For each encoder position i:                                                   │
│   e_i = v × tanh(encoder_features[i] + decoder_features)                        │
│                                                                                   │
│   e = [e_0, e_1, ..., e_12]  # [1, 13]                                         │
│                                                                                   │
│   Suppose e = [2.1, 0.5, 0.8, 0.2, 0.3, 0.4, 0.6, 0.3, 0.1,                    │
│                0.2, 0.2, 0.3, 0.1]                                               │
│                 ↑                                                                 │
│              "germany" gets highest score                                        │
│                                                                                   │
│   attn_dist = softmax(e)                                                        │
│             = [0.52, 0.05, 0.07, 0.02, 0.03, 0.04, 0.06, 0.03, 0.02,           │
│                0.03, 0.03, 0.04, 0.02]                                          │
│                 ↑                                                                 │
│              High attention on "germany"                                         │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.5: Compute context vector                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   context = Σ_i attn_dist[i] × encoder_outputs[i]                               │
│   context shape: [1, 512]                                                        │
│   (Weighted sum heavily weighted toward "germany"'s encoding)                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.6: Update coverage                                                            │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   coverage (before): [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]                   │
│   coverage (after):  [0.52, 0.05, 0.07, 0.02, 0.03, 0.04, 0.06, 0.03,          │
│                       0.02, 0.03, 0.03, 0.04, 0.02]                             │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.7: Compute p_gen                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   p_gen = sigmoid(w_c × context + w_s × state.h + w_x × inp + b)               │
│                                                                                   │
│   Suppose p_gen = 0.6 (model favors generating from vocab)                      │
│   This makes sense: "germany" is in vocabulary!                                 │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.8: Compute output distribution                                                │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   decoder_output = linear([cell_output, context], hidden_dim)                   │
│   Shape: [1, 256]                                                                │
│                                                                                   │
│   vocab_scores = decoder_output @ W_out + b_out                                 │
│   Shape: [1, 50000]                                                              │
│                                                                                   │
│   vocab_dist = softmax(vocab_scores)                                            │
│   Suppose vocab_dist[100] = 0.45 (high prob for "germany")                      │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.9: Compute final distribution                                                 │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   extended_vocab_size = 50000 + 2 = 50002                                       │
│                                                                                   │
│   vocab_dist_weighted = p_gen × vocab_dist                                      │
│                       = 0.6 × vocab_dist                                        │
│   vocab_dist_weighted[100] = 0.6 × 0.45 = 0.27                                 │
│                                                                                   │
│   copy_dist = (1 - p_gen) × attn_dist projected to vocab                       │
│             = 0.4 × attn_dist                                                   │
│                                                                                   │
│   Position 0 (germany, ID=100): copy_dist[100] += 0.4 × 0.52 = 0.208          │
│   Position 9 (mario, ID=50000): copy_dist[50000] = 0.4 × 0.03 = 0.012         │
│   Position 10 (götze, ID=50001): copy_dist[50001] = 0.4 × 0.03 = 0.012        │
│                                                                                   │
│   final_dist = vocab_dist_weighted + copy_dist                                  │
│   final_dist[100] = 0.27 + 0.208 = 0.478                                       │
│                                                                                   │
│   "germany" has highest probability! ✓                                          │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   8.10: Compute loss for this step                                                │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Target: "germany" (ID=100)                                                     │
│   P(target) = final_dist[100] = 0.478                                           │
│   Loss = -log(0.478) = 0.738                                                    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Step 9: Decoder Step 6 (Generate "götze")

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│               DECODER STEP 6: Generate "götze"                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   This is an interesting step because "götze" is OOV!                           │
│                                                                                   │
│   INPUT: "." token (ID=5) from previous step                                    │
│   TARGET: "götze" (ID=50001 in extended vocab)                                  │
│                                                                                   │
│   Previous coverage (accumulated):                                               │
│   Position:    0     1     2     3    4     5     6     7    8                  │
│   Word:      ger-  beat  arg-  in  the  wor-  cup  fin-  .                      │
│   Coverage:  [0.65, 0.15, 0.20, 0.10, 0.45, 0.30, 0.55, 0.15, 0.40,             │
│                                                                                   │
│   Position:    9      10     11    12                                            │
│   Word:      mario  götze  scor-  .                                              │
│   Coverage:   0.08,  0.05,  0.25, 0.10]                                          │
│                      ↑                                                            │
│               Low coverage on "götze" - not yet attended!                       │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Attention with coverage                                                         │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   With coverage mechanism:                                                        │
│   e_i = v × tanh(encoder_features[i] + decoder_features + w_c × coverage[i])   │
│                                                                                   │
│   High coverage positions get LOWER scores (coverage discourages re-attending)  │
│                                                                                   │
│   Attention scores (before softmax):                                             │
│   Position 0 (germany, cov=0.65): e_0 = 0.8 (reduced due to coverage)          │
│   Position 6 (cup, cov=0.55):     e_6 = 0.6 (reduced due to coverage)          │
│   Position 10 (götze, cov=0.05): e_10 = 2.1 (HIGH - low coverage!)            │
│                                                                                   │
│   attn_dist = softmax(e)                                                        │
│             = [0.05, 0.03, 0.04, 0.02, 0.06, 0.05, 0.04, 0.03, 0.05,           │
│                0.08, 0.48, 0.04, 0.03]                                          │
│                      ↑                                                            │
│               High attention on "götze"!                                        │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   p_gen calculation                                                               │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Context vector is dominated by "götze"'s encoding                             │
│   Since "götze" is OOV, the model should COPY                                   │
│                                                                                   │
│   p_gen = 0.2 (model wants to COPY, not generate!)                              │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Final distribution                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   vocab_dist[götze] = 0 (not in vocabulary!)                                   │
│   vocab_dist_weighted[götze] = 0.2 × 0 = 0                                     │
│                                                                                   │
│   copy_dist:                                                                      │
│   Position 10 (götze, ID=50001):                                                │
│   copy_dist[50001] = (1 - 0.2) × 0.48 = 0.384                                  │
│                                                                                   │
│   final_dist[50001] = 0 + 0.384 = 0.384                                        │
│                                                                                   │
│   "götze" (ID 50001) has reasonable probability                                │
│   entirely from the COPY mechanism!                                             │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Loss                                                                            │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Target: "götze" (ID=50001)                                                    │
│   P(target) = final_dist[50001] = 0.384                                        │
│   Loss = -log(0.384) = 0.957                                                    │
│                                                                                   │
│   Coverage loss for this step:                                                    │
│   covloss = Σ min(attn_dist[i], coverage[i])                                    │
│           = min(0.05, 0.65) + min(0.03, 0.15) + ... + min(0.48, 0.05) + ...    │
│           = 0.05 + 0.03 + 0.04 + 0.02 + 0.06 + 0.05 + 0.04 + 0.03 + 0.05       │
│             + 0.08 + 0.05 + 0.04 + 0.03                                         │
│           ≈ 0.57                                                                │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Numerical Example

### Full Training Step

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE TRAINING STEP                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Target: "germany won the cup . götze scored the winner ."                     │
│                                                                                   │
│   Loss per step:                                                                  │
│   ──────────────                                                                  │
│   Step 1 (germany):  0.738                                                       │
│   Step 2 (won):      0.892                                                       │
│   Step 3 (the):      0.453                                                       │
│   Step 4 (cup):      0.621                                                       │
│   Step 5 (.):        0.312                                                       │
│   Step 6 (götze):   0.957                                                       │
│   Step 7 (scored):   0.534                                                       │
│   Step 8 (the):      0.423                                                       │
│   Step 9 (winner):   0.845                                                       │
│   Step 10 (.):       0.298                                                       │
│   Step 11 (STOP):    0.187                                                       │
│                                                                                   │
│   L_NLL = (0.738 + 0.892 + 0.453 + 0.621 + 0.312 +                              │
│            0.957 + 0.534 + 0.423 + 0.845 + 0.298 + 0.187) / 11                  │
│         = 6.26 / 11                                                              │
│         = 0.569                                                                  │
│                                                                                   │
│   Coverage loss per step:                                                         │
│   ───────────────────────                                                         │
│   Step 1:  0.00                                                                  │
│   Step 2:  0.15                                                                  │
│   Step 3:  0.28                                                                  │
│   Step 4:  0.35                                                                  │
│   Step 5:  0.42                                                                  │
│   Step 6:  0.57                                                                  │
│   Step 7:  0.48                                                                  │
│   Step 8:  0.52                                                                  │
│   Step 9:  0.45                                                                  │
│   Step 10: 0.38                                                                  │
│   Step 11: 0.32                                                                  │
│                                                                                   │
│   L_coverage = (0.00 + 0.15 + ... + 0.32) / 11                                  │
│              = 3.92 / 11                                                         │
│              = 0.356                                                             │
│                                                                                   │
│   Total loss:                                                                     │
│   ───────────                                                                     │
│   L_total = L_NLL + λ × L_coverage                                               │
│           = 0.569 + 1.0 × 0.356                                                 │
│           = 0.925                                                                │
│                                                                                   │
│   This loss is backpropagated to update all model parameters.                    │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Training vs Inference

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    TRAINING VS INFERENCE                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   TRAINING (Teacher Forcing):                                                     │
│   ───────────────────────────                                                     │
│                                                                                   │
│   Decoder input at step t = GROUND TRUTH word from step t-1                     │
│                                                                                   │
│   Step 1: Input = [START],  Target = "germany"                                  │
│   Step 2: Input = "germany", Target = "won"     ← Use ground truth!            │
│   Step 3: Input = "won",     Target = "the"     ← Use ground truth!            │
│   ...                                                                             │
│                                                                                   │
│   Even if model predicts wrong word, we feed correct word.                       │
│   This provides stable training signal.                                          │
│                                                                                   │
│                                                                                   │
│   INFERENCE (Beam Search):                                                        │
│   ────────────────────────                                                        │
│                                                                                   │
│   Decoder input at step t = MODEL'S PREDICTION from step t-1                    │
│                                                                                   │
│   Step 1: Input = [START]                                                        │
│           Model predicts: "germany" (0.48), "the" (0.15), ...                   │
│           Keep top-k hypotheses                                                  │
│                                                                                   │
│   Step 2: For hypothesis "germany":                                              │
│           Input = "germany"                                                      │
│           Model predicts: "won" (0.35), "beat" (0.28), ...                      │
│           Expand beam                                                            │
│                                                                                   │
│   ...continue until [STOP] or max_length...                                     │
│                                                                                   │
│   Final output = highest probability complete hypothesis                         │
│                                                                                   │
│                                                                                   │
│   KEY DIFFERENCES:                                                                │
│   ─────────────────                                                               │
│                                                                                   │
│   Aspect          Training              Inference                                │
│   ──────          ────────              ─────────                                │
│   Decoder input   Ground truth          Model's prediction                      │
│   Loss computed   Yes                   No                                       │
│   Gradients       Yes                   No                                       │
│   Beam search     No                    Yes                                      │
│   Coverage init   Zeros                 May carry over                          │
│   batch_size      16+                   1 (per article)                         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

This end-to-end example traced:

1. **Data preparation**: Tokenization, ID conversion, OOV handling
2. **Encoding**: Embedding lookup, bidirectional LSTM, state reduction
3. **Decoding**: Attention, p_gen, final distribution, loss
4. **Special case**: OOV word "götze" copied via pointer mechanism
5. **Full loss**: NLL + coverage loss calculation

Key insights:
- OOV words get the same UNK embedding but different extended vocab IDs
- Coverage mechanism helps attend to new positions
- p_gen adapts: high for vocab words, low for OOV copying
- Final distribution combines vocab and copy probabilities

---

*Next: [15_diagrams.md](15_diagrams.md) - Visual Diagrams and Flowcharts*
