# Architecture Deep Dive

## Table of Contents
1. [High-Level Data Flow](#high-level-data-flow)
2. [Encoder Architecture](#encoder-architecture)
3. [State Reduction Layer](#state-reduction-layer)
4. [Decoder Architecture](#decoder-architecture)
5. [Output Projection](#output-projection)
6. [Final Distribution Calculation](#final-distribution-calculation)
7. [Component Interaction Diagram](#component-interaction-diagram)

---

## High-Level Data Flow

### Complete Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       COMPLETE DATA FLOW PIPELINE                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   INPUT ARTICLE                                                                   │
│   ─────────────                                                                   │
│   "Germany emerged as the winners of the 2014 FIFA World Cup after defeating     │
│    Argentina 1-0 in the final at the Maracanã Stadium..."                        │
│                                                                                   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         1. TOKENIZATION                                  │   │
│   │   ["Germany", "emerged", "as", "the", "winners", "of", "the", "2014",   │   │
│   │    "FIFA", "World", "Cup", "after", "defeating", "Argentina", ...]      │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    2. VOCABULARY LOOKUP + OOV HANDLING                   │   │
│   │   enc_batch:          [1, 234, 15, 8, 567, 23, 8, 1, 1, 89, 156, ...]   │   │
│   │   enc_batch_extend:   [50000, 234, 15, 8, 567, 23, 8, 50001, 50002,     │   │
│   │                        89, 156, ...]                                     │   │
│   │   article_oovs:       ["Germany", "2014", "FIFA", "Argentina", ...]     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         3. EMBEDDING LOOKUP                              │   │
│   │   Shape: [batch_size, max_enc_steps, emb_dim]                           │   │
│   │   Example: [16, 400, 128]                                               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                    4. BIDIRECTIONAL LSTM ENCODER                         │   │
│   │   Output: encoder_states                                                 │   │
│   │   Shape: [batch_size, max_enc_steps, 2*hidden_dim]                      │   │
│   │   Example: [16, 400, 512]                                               │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      5. REDUCE STATES LAYER                              │   │
│   │   Combines bidirectional final states into single decoder initial state │   │
│   │   Output: dec_in_state                                                  │   │
│   │   Shape: LSTMStateTuple(c=[batch, hidden], h=[batch, hidden])          │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                   6. ATTENTION DECODER (per time step)                   │   │
│   │   For each decoder step t = 1, 2, ..., T:                               │   │
│   │     a) Attention over encoder states → context vector                   │   │
│   │     b) LSTM step → decoder state                                        │   │
│   │     c) Calculate p_gen (if pointer-generator)                           │   │
│   │     d) Update coverage (if enabled)                                     │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      7. OUTPUT PROJECTION                                │   │
│   │   Project decoder output to vocabulary distribution                      │   │
│   │   vocab_scores: [batch_size, vocab_size]                                │   │
│   │   vocab_dists: softmax(vocab_scores)                                    │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                   8. FINAL DISTRIBUTION CALCULATION                      │   │
│   │   P_final = p_gen × P_vocab + (1-p_gen) × P_attn                        │   │
│   │   Shape: [batch_size, extended_vocab_size]                              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      9. LOSS CALCULATION                                 │   │
│   │   NLL Loss + Coverage Loss (if enabled)                                 │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                     │                                             │
│                                     ▼                                             │
│   OUTPUT SUMMARY                                                                  │
│   ──────────────                                                                  │
│   "Germany beat Argentina 1-0 to win the 2014 World Cup."                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Encoder Architecture

### Bidirectional LSTM Encoder

The encoder processes the input sequence in both directions to capture full context:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        ENCODER ARCHITECTURE DETAIL                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   model.py: _add_encoder() method                                                │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   Input: emb_enc_inputs                                                          │
│   Shape: [batch_size, <=max_enc_steps, emb_dim]                                  │
│   Example: [16, 350, 128]                                                        │
│                                                                                   │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                        FORWARD LSTM                                     │    │
│   │                                                                         │    │
│   │   cell_fw = LSTMCell(hidden_dim=256)                                   │    │
│   │                                                                         │    │
│   │   x₁ ──▶ LSTM ──▶ →h₁ ──▶ LSTM ──▶ →h₂ ──▶ ... ──▶ →hₙ                │    │
│   │                                                                         │    │
│   │   Final state: (→c_n, →h_n)                                            │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                        BACKWARD LSTM                                    │    │
│   │                                                                         │    │
│   │   cell_bw = LSTMCell(hidden_dim=256)                                   │    │
│   │                                                                         │    │
│   │   xₙ ──▶ LSTM ──▶ ←hₙ ──▶ LSTM ──▶ ←h_{n-1} ──▶ ... ──▶ ←h₁           │    │
│   │                                                                         │    │
│   │   Final state: (←c_1, ←h_1)                                            │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                      CONCATENATION                                      │    │
│   │                                                                         │    │
│   │   For each position i:                                                  │    │
│   │   h_i = [→h_i ; ←h_i]                                                  │    │
│   │                                                                         │    │
│   │   Shape: [batch_size, seq_len, 2*hidden_dim]                           │    │
│   │   Example: [16, 350, 512]                                              │    │
│   │                                                                         │    │
│   │   h₁ = [→h₁(256); ←h₁(256)] = h₁(512)                                 │    │
│   │   h₂ = [→h₂(256); ←h₂(256)] = h₂(512)                                 │    │
│   │   ...                                                                   │    │
│   │   hₙ = [→hₙ(256); ←hₙ(256)] = hₙ(512)                                 │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│   Output:                                                                         │
│   - encoder_outputs: [batch_size, seq_len, 512]                                 │
│   - fw_st: LSTMStateTuple(c=[batch, 256], h=[batch, 256])                       │
│   - bw_st: LSTMStateTuple(c=[batch, 256], h=[batch, 256])                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Why Bidirectional?

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   WHY BIDIRECTIONAL ENCODING MATTERS                              │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Example sentence: "The bank by the river was steep."                           │
│                      │    │                                                       │
│                      │    └─ Ambiguous: financial bank or river bank?            │
│                      └────── Needs context from BOTH sides                       │
│                                                                                   │
│                                                                                   │
│   Forward only (at "bank"):                                                      │
│   ─────────────────────────                                                       │
│   Context: "The" → "bank"                                                        │
│   Missing: "by the river"                                                        │
│   Result: Ambiguous representation of "bank"                                     │
│                                                                                   │
│                                                                                   │
│   Bidirectional (at "bank"):                                                     │
│   ─────────────────────────                                                       │
│   Forward context: "The" → "bank"                                                │
│   Backward context: "river" ← "the" ← "by" ← "bank"                             │
│   Result: "bank" representation includes "river" context                         │
│           → Correctly understood as river bank!                                  │
│                                                                                   │
│                                                                                   │
│   For summarization:                                                              │
│   ──────────────────                                                              │
│   - Early words need late-article context                                        │
│   - Late words need early-article context                                        │
│   - Proper nouns defined later help understand earlier mentions                  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## State Reduction Layer

### Purpose

The encoder is bidirectional (produces 2 states) but the decoder is unidirectional (needs 1 state):

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        STATE REDUCTION LAYER                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   model.py: _reduce_states() method                                              │
│   ─────────────────────────────────                                               │
│                                                                                   │
│   Problem:                                                                        │
│   ─────────                                                                       │
│   Encoder produces: fw_st (256-dim) + bw_st (256-dim)                           │
│   Decoder needs: single initial state (256-dim)                                 │
│                                                                                   │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                       REDUCTION PROCESS                                │     │
│   │                                                                        │     │
│   │                                                                        │     │
│   │   Forward Final State (fw_st):     Backward Final State (bw_st):      │     │
│   │   ┌─────────────────────────┐     ┌─────────────────────────┐         │     │
│   │   │ c_fw: [batch, 256]      │     │ c_bw: [batch, 256]      │         │     │
│   │   │ h_fw: [batch, 256]      │     │ h_bw: [batch, 256]      │         │     │
│   │   └───────────┬─────────────┘     └───────────┬─────────────┘         │     │
│   │               │                               │                        │     │
│   │               └───────────┬───────────────────┘                        │     │
│   │                           │                                            │     │
│   │                           ▼                                            │     │
│   │               ┌───────────────────────┐                                │     │
│   │               │     CONCATENATE       │                                │     │
│   │               └───────────┬───────────┘                                │     │
│   │                           │                                            │     │
│   │           ┌───────────────┴───────────────┐                            │     │
│   │           │                               │                            │     │
│   │           ▼                               ▼                            │     │
│   │   old_c = [c_fw; c_bw]           old_h = [h_fw; h_bw]                 │     │
│   │   Shape: [batch, 512]            Shape: [batch, 512]                  │     │
│   │                                                                        │     │
│   │           │                               │                            │     │
│   │           ▼                               ▼                            │     │
│   │   ┌─────────────────┐           ┌─────────────────┐                   │     │
│   │   │ Linear + ReLU   │           │ Linear + ReLU   │                   │     │
│   │   │ W_reduce_c      │           │ W_reduce_h      │                   │     │
│   │   │ [512, 256]      │           │ [512, 256]      │                   │     │
│   │   └────────┬────────┘           └────────┬────────┘                   │     │
│   │            │                              │                            │     │
│   │            ▼                              ▼                            │     │
│   │   new_c: [batch, 256]           new_h: [batch, 256]                   │     │
│   │                                                                        │     │
│   │            └──────────────┬───────────────┘                            │     │
│   │                           │                                            │     │
│   │                           ▼                                            │     │
│   │               ┌───────────────────────┐                                │     │
│   │               │  LSTMStateTuple       │                                │     │
│   │               │  (new_c, new_h)       │                                │     │
│   │               │  = dec_in_state       │                                │     │
│   │               └───────────────────────┘                                │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│   Equations:                                                                      │
│   ──────────                                                                      │
│   old_c = concat([fw_st.c, bw_st.c])       # [batch, 512]                        │
│   old_h = concat([fw_st.h, bw_st.h])       # [batch, 512]                        │
│   new_c = ReLU(W_reduce_c × old_c + bias_reduce_c)  # [batch, 256]              │
│   new_h = ReLU(W_reduce_h × old_h + bias_reduce_h)  # [batch, 256]              │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Decoder Architecture

### Attention Decoder Overview

The decoder generates one word at a time, using attention to focus on relevant parts of the input:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        DECODER ARCHITECTURE DETAIL                                │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   attention_decoder.py: attention_decoder() function                             │
│   ────────────────────────────────────────────────                                │
│                                                                                   │
│   For each decoder time step t:                                                  │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                      STEP 1: INPUT PREPARATION                         │     │
│   │                                                                        │     │
│   │   Previous word embedding: y_{t-1}  [batch, emb_dim]                  │     │
│   │   Previous context: c_{t-1}         [batch, 2*hidden_dim]             │     │
│   │                                                                        │     │
│   │   Combined input:                                                      │     │
│   │   x_t = Linear([y_{t-1}; c_{t-1}], emb_dim)                          │     │
│   │       = Linear([batch, 128+512], 128)                                 │     │
│   │       = [batch, 128]                                                  │     │
│   │                                                                        │     │
│   │   This "attention feeding" mechanism helps the decoder know           │     │
│   │   what it attended to previously.                                      │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                     │                                             │
│                                     ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                      STEP 2: LSTM STEP                                 │     │
│   │                                                                        │     │
│   │   (cell_output, state) = LSTM_cell(x_t, state_{t-1})                  │     │
│   │                                                                        │     │
│   │   cell_output: [batch, hidden_dim] = [batch, 256]                     │     │
│   │   state: LSTMStateTuple(c, h) each [batch, 256]                       │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                     │                                             │
│                                     ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                   STEP 3: ATTENTION CALCULATION                        │     │
│   │                                                                        │     │
│   │   For each encoder position i:                                         │     │
│   │                                                                        │     │
│   │   ┌────────────────────────────────────────────────────────────┐      │     │
│   │   │  WITHOUT COVERAGE:                                          │      │     │
│   │   │  e_ti = v^T · tanh(W_h·h_i + W_s·s_t + b_attn)             │      │     │
│   │   │                                                             │      │     │
│   │   │  WITH COVERAGE:                                             │      │     │
│   │   │  e_ti = v^T · tanh(W_h·h_i + W_s·s_t + w_c·c_t^i + b_attn) │      │     │
│   │   └────────────────────────────────────────────────────────────┘      │     │
│   │                                                                        │     │
│   │   Attention weights:                                                   │     │
│   │   α_t = masked_softmax(e_t)                                           │     │
│   │   Shape: [batch, enc_len]                                             │     │
│   │                                                                        │     │
│   │   Context vector:                                                      │     │
│   │   c_t = Σ_i α_ti · h_i                                                │     │
│   │   Shape: [batch, 2*hidden_dim] = [batch, 512]                         │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                     │                                             │
│                                     ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │              STEP 4: GENERATION PROBABILITY (if pointer_gen)           │     │
│   │                                                                        │     │
│   │   p_gen = σ(Linear([c_t, s_t.c, s_t.h, x_t], 1))                     │     │
│   │         = σ(Linear([512, 256, 256, 128], 1))                         │     │
│   │         = σ(Linear([batch, 1152], 1))                                │     │
│   │         ∈ [0, 1]                                                      │     │
│   │                                                                        │     │
│   │   Shape: [batch, 1]                                                   │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                     │                                             │
│                                     ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │              STEP 5: OUTPUT PROJECTION                                 │     │
│   │                                                                        │     │
│   │   output = Linear([cell_output; c_t], hidden_dim)                     │     │
│   │          = Linear([batch, 256+512], 256)                              │     │
│   │          = [batch, 256]                                               │     │
│   │                                                                        │     │
│   │   This is the decoder output for this time step.                      │     │
│   │   Will be projected to vocabulary in the next stage.                  │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                     │                                             │
│                                     ▼                                             │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │              STEP 6: COVERAGE UPDATE (if coverage)                     │     │
│   │                                                                        │     │
│   │   coverage_{t+1} = coverage_t + α_t                                   │     │
│   │                                                                        │     │
│   │   Shape: [batch, enc_len]                                             │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│   Outputs at each step:                                                          │
│   ─────────────────────                                                           │
│   - output: [batch, hidden_dim]                                                  │
│   - state: LSTMStateTuple                                                        │
│   - attn_dist: [batch, enc_len]                                                  │
│   - p_gen: [batch, 1] (if pointer_gen)                                          │
│   - coverage: [batch, enc_len] (if coverage)                                     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Decoder Loop Visualization

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       DECODER LOOP VISUALIZATION                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Target: "[START] Germany beat Argentina [STOP]"                                │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   TIME STEP 0                                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Input: [START] token                                                           │
│   Previous context: zeros (initialized)                                          │
│   Target output: "Germany"                                                       │
│                                                                                   │
│   ┌─────────┐     ┌───────────┐     ┌─────────────┐                             │
│   │ [START] │────▶│ Decoder   │────▶│ "Germany"   │                             │
│   │ + zeros │     │ LSTM      │     │ (predict)   │                             │
│   └─────────┘     └───────────┘     └─────────────┘                             │
│                                                                                   │
│   Attention: Focuses heavily on "Germany" in source                             │
│   p_gen: ~0.2 (low, likely to copy)                                             │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   TIME STEP 1                                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Input: "Germany" token + context from step 0                                   │
│   Target output: "beat"                                                          │
│                                                                                   │
│   ┌─────────────┐     ┌───────────┐     ┌─────────┐                             │
│   │ "Germany"   │────▶│ Decoder   │────▶│ "beat"  │                             │
│   │ + context   │     │ LSTM      │     │(predict)│                             │
│   └─────────────┘     └───────────┘     └─────────┘                             │
│                                                                                   │
│   Attention: Focuses on "emerged", "winners", action words                       │
│   p_gen: ~0.7 (higher, might generate "beat" or similar)                        │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   TIME STEP 2                                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Input: "beat" token + context from step 1                                      │
│   Target output: "Argentina"                                                     │
│                                                                                   │
│   ┌─────────┐     ┌───────────┐     ┌─────────────┐                             │
│   │ "beat"  │────▶│ Decoder   │────▶│ "Argentina" │                             │
│   │+ context│     │ LSTM      │     │ (predict)   │                             │
│   └─────────┘     └───────────┘     └─────────────┘                             │
│                                                                                   │
│   Attention: Focuses heavily on "Argentina" in source                           │
│   p_gen: ~0.1 (very low, will copy)                                             │
│                                                                                   │
│   ...continues until [STOP] is generated or max_dec_steps reached               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Projection

### Vocabulary Distribution

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        OUTPUT PROJECTION LAYER                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   model.py: _add_seq2seq() - output_projection section                           │
│   ──────────────────────────────────────────────────                              │
│                                                                                   │
│   For each decoder output at step t:                                             │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │   decoder_output[t]                                                   │     │
│   │   Shape: [batch_size, hidden_dim] = [16, 256]                         │     │
│   │                                                                        │     │
│   │                          │                                             │     │
│   │                          ▼                                             │     │
│   │   ┌────────────────────────────────────────────────────┐              │     │
│   │   │            LINEAR PROJECTION                        │              │     │
│   │   │                                                     │              │     │
│   │   │   W: [hidden_dim, vocab_size] = [256, 50000]       │              │     │
│   │   │   v: [vocab_size] = [50000]   (bias)               │              │     │
│   │   │                                                     │              │     │
│   │   │   vocab_scores = decoder_output × W + v            │              │     │
│   │   │   Shape: [batch_size, vocab_size] = [16, 50000]    │              │     │
│   │   │                                                     │              │     │
│   │   └───────────────────────────┬────────────────────────┘              │     │
│   │                               │                                        │     │
│   │                               ▼                                        │     │
│   │   ┌────────────────────────────────────────────────────┐              │     │
│   │   │              SOFTMAX                                │              │     │
│   │   │                                                     │              │     │
│   │   │   vocab_dists = softmax(vocab_scores)              │              │     │
│   │   │   Shape: [batch_size, vocab_size] = [16, 50000]    │              │     │
│   │   │                                                     │              │     │
│   │   │   Sum of each row = 1.0                            │              │     │
│   │   │   Each value = P(word | decoder_state)             │              │     │
│   │   │                                                     │              │     │
│   │   └────────────────────────────────────────────────────┘              │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│   Example output:                                                                 │
│   ───────────────                                                                 │
│   vocab_dists[0] = [0.001, 0.002, 0.0001, ..., 0.35, ..., 0.003]                 │
│                     │       │      │           │                                  │
│                    P(PAD)  P(UNK) P(START)  P("the") = 0.35                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Final Distribution Calculation

### Pointer-Generator Combination

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    FINAL DISTRIBUTION CALCULATION                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   model.py: _calc_final_dist() method                                            │
│   ────────────────────────────────────                                            │
│                                                                                   │
│   TWO INPUT DISTRIBUTIONS:                                                        │
│   ────────────────────────                                                        │
│                                                                                   │
│   1. vocab_dists: [batch, vocab_size] = [16, 50000]                             │
│      - From output projection                                                    │
│      - P(word) from generation                                                   │
│                                                                                   │
│   2. attn_dists: [batch, enc_len] = [16, 350]                                   │
│      - From attention mechanism                                                  │
│      - P(position) for copying                                                   │
│                                                                                   │
│   3. p_gens: [batch, 1] = [16, 1]                                               │
│      - From generation probability calculation                                   │
│      - Scalar per example: how much to generate vs copy                         │
│                                                                                   │
│                                                                                   │
│   STEP 1: Scale distributions by p_gen                                           │
│   ─────────────────────────────────────                                           │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │   vocab_dists_scaled = p_gen × vocab_dists                            │     │
│   │   Shape: [batch, vocab_size]                                          │     │
│   │                                                                        │     │
│   │   If p_gen = 0.3, then:                                               │     │
│   │   P_gen("the") = 0.3 × 0.35 = 0.105                                  │     │
│   │                                                                        │     │
│   │   attn_dists_scaled = (1 - p_gen) × attn_dists                        │     │
│   │   Shape: [batch, enc_len]                                             │     │
│   │                                                                        │     │
│   │   If p_gen = 0.3, then:                                               │     │
│   │   P_copy(position_5) = 0.7 × 0.8 = 0.56                              │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│                                                                                   │
│   STEP 2: Extend vocabulary with zeros for OOVs                                  │
│   ──────────────────────────────────────────────                                  │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │   extended_vsize = vocab_size + max_art_oovs                          │     │
│   │                  = 50000 + 3 = 50003  (example with 3 OOVs)           │     │
│   │                                                                        │     │
│   │   extra_zeros = zeros([batch, max_art_oovs])                          │     │
│   │               = zeros([16, 3])                                        │     │
│   │                                                                        │     │
│   │   vocab_dists_extended = concat(vocab_dists_scaled, extra_zeros)      │     │
│   │   Shape: [batch, extended_vsize] = [16, 50003]                        │     │
│   │                                                                        │     │
│   │   Before:  [P₀, P₁, ..., P₄₉₉₉₉]  (50000 values)                     │     │
│   │   After:   [P₀, P₁, ..., P₄₉₉₉₉, 0, 0, 0]  (50003 values)            │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│                                                                                   │
│   STEP 3: Project attention to extended vocabulary (SCATTER)                     │
│   ───────────────────────────────────────────────────────────                     │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │   This is the KEY operation that enables copying!                      │     │
│   │                                                                        │     │
│   │   enc_batch_extend_vocab: [batch, enc_len]                            │     │
│   │   Contains word indices (including OOV indices 50000+)                │     │
│   │                                                                        │     │
│   │   Example:                                                             │     │
│   │   enc_batch_extend_vocab[0] = [50000, 234, 15, 8, 567, 50001, ...]   │     │
│   │                                 │                    │                 │     │
│   │                            "Germany"            "Argentina"            │     │
│   │                            (OOV #0)             (OOV #1)               │     │
│   │                                                                        │     │
│   │   attn_dists_scaled[0] = [0.56, 0.02, 0.01, 0.005, 0.01, 0.35, ...]  │     │
│   │                           │                         │                  │     │
│   │                      attention               attention                 │     │
│   │                      on "Germany"           on "Argentina"             │     │
│   │                                                                        │     │
│   │   SCATTER operation:                                                   │     │
│   │   attn_dists_projected = zeros([batch, extended_vsize])               │     │
│   │                                                                        │     │
│   │   For each position i in encoder:                                      │     │
│   │     word_idx = enc_batch_extend_vocab[batch, i]                       │     │
│   │     attn_dists_projected[batch, word_idx] += attn_dists_scaled[i]    │     │
│   │                                                                        │     │
│   │   Result (for batch 0):                                                │     │
│   │   attn_dists_projected[0, 50000] = 0.56  (copied to "Germany" slot)  │     │
│   │   attn_dists_projected[0, 234] = 0.02    (copied to word 234)        │     │
│   │   attn_dists_projected[0, 50001] = 0.35  (copied to "Argentina" slot)│     │
│   │   ...                                                                  │     │
│   │                                                                        │     │
│   │   NOTE: Same words at different positions ADD their probabilities!    │     │
│   │   If "the" appears 5 times, its total copy prob = sum of 5 attentions│     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│                                                                                   │
│   STEP 4: Add distributions together                                             │
│   ──────────────────────────────────                                              │
│                                                                                   │
│   ┌───────────────────────────────────────────────────────────────────────┐     │
│   │                                                                        │     │
│   │   final_dists = vocab_dists_extended + attn_dists_projected           │     │
│   │                                                                        │     │
│   │   Shape: [batch, extended_vsize] = [16, 50003]                        │     │
│   │                                                                        │     │
│   │   Example (for "Germany" at index 50000):                              │     │
│   │                                                                        │     │
│   │   vocab_dists_extended[0, 50000] = 0.0  (OOV, can't generate)         │     │
│   │   attn_dists_projected[0, 50000] = 0.56 (copied from source)          │     │
│   │   ─────────────────────────────────────────────────────────           │     │
│   │   final_dists[0, 50000] = 0.0 + 0.56 = 0.56                           │     │
│   │                                                                        │     │
│   │   Example (for "the" at index 8):                                      │     │
│   │                                                                        │     │
│   │   vocab_dists_extended[0, 8] = 0.105 (p_gen × P_vocab)                │     │
│   │   attn_dists_projected[0, 8] = 0.12  (from multiple "the"s in source) │     │
│   │   ─────────────────────────────────────────────────────────           │     │
│   │   final_dists[0, 8] = 0.105 + 0.12 = 0.225                            │     │
│   │                                                                        │     │
│   └───────────────────────────────────────────────────────────────────────┘     │
│                                                                                   │
│                                                                                   │
│   FINAL OUTPUT                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   final_dists: probability distribution over extended vocabulary                 │
│                                                                                   │
│   Index 0-49999:     Standard vocabulary words (generated + copied)              │
│   Index 50000+:      Article-specific OOV words (copied only)                    │
│                                                                                   │
│   The model selects: argmax(final_dists) or samples from it                     │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Interaction Diagram

### Complete System Interaction

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE COMPONENT INTERACTION DIAGRAM                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│                                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                           INPUT PROCESSING                               │   │
│   │                                                                          │   │
│   │   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────────┐  │   │
│   │   │   Article    │───▶│  Vocabulary  │───▶│   enc_batch (UNK ids)    │  │   │
│   │   │   Tokens     │    │   Lookup     │    │   enc_batch_extend_vocab │  │   │
│   │   │              │    │              │    │   article_oovs           │  │   │
│   │   └──────────────┘    └──────────────┘    └──────────────────────────┘  │   │
│   │                                                                          │   │
│   └───────────────────────────────────────┬─────────────────────────────────┘   │
│                                           │                                       │
│                                           ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                            ENCODER                                       │   │
│   │                                                                          │   │
│   │   ┌──────────┐        ┌───────────────────────┐        ┌──────────┐    │   │
│   │   │ Embedding │──────▶│  Bidirectional LSTM   │──────▶│ enc_states│    │   │
│   │   │  Lookup   │       │  forward + backward   │       │ [b,t,512] │    │   │
│   │   └──────────┘        └───────────┬───────────┘       └────┬──────┘    │   │
│   │                                    │                        │           │   │
│   │                        ┌───────────┴───────────┐            │           │   │
│   │                        │                       │            │           │   │
│   │                    ┌───▼───┐              ┌────▼────┐       │           │   │
│   │                    │ fw_st │              │  bw_st  │       │           │   │
│   │                    └───┬───┘              └────┬────┘       │           │   │
│   │                        │                       │            │           │   │
│   └────────────────────────┼───────────────────────┼────────────┼───────────┘   │
│                            │                       │            │               │
│                            └───────────┬───────────┘            │               │
│                                        │                        │               │
│                                        ▼                        │               │
│   ┌─────────────────────────────────────────────────────────────┼───────────┐   │
│   │                        REDUCE STATES                        │           │   │
│   │                                                             │           │   │
│   │   ┌──────────────────────────────────────────────────┐     │           │   │
│   │   │  new_c = ReLU(W_c · [fw_st.c; bw_st.c] + b_c)   │     │           │   │
│   │   │  new_h = ReLU(W_h · [fw_st.h; bw_st.h] + b_h)   │     │           │   │
│   │   │  dec_in_state = (new_c, new_h)                   │     │           │   │
│   │   └──────────────────────────┬───────────────────────┘     │           │   │
│   │                              │                              │           │   │
│   └──────────────────────────────┼──────────────────────────────┼───────────┘   │
│                                  │                              │               │
│                                  ▼                              │               │
│   ┌──────────────────────────────────────────────────────────────────────────┐  │
│   │                      ATTENTION DECODER                      │            │  │
│   │                                                             │            │  │
│   │   FOR EACH TIME STEP t:                                    │            │  │
│   │                                                             │            │  │
│   │   ┌───────────────────────────────────────────────────────────────────┐ │  │
│   │   │                                                         │         │ │  │
│   │   │   ┌──────────────┐    ┌────────────────────────────────▼────────┐│ │  │
│   │   │   │ y_{t-1}      │───▶│              ATTENTION                  ││ │  │
│   │   │   │ (prev word)  │    │                                         ││ │  │
│   │   │   └──────────────┘    │   e_ti = v^T·tanh(W_h·h_i + W_s·s_t)   ││ │  │
│   │   │                       │   α_t = softmax(e_t)                    ││ │  │
│   │   │   ┌──────────────┐    │   c_t = Σ α_ti · h_i                   ││ │  │
│   │   │   │ state_{t-1}  │───▶│                                         ││ │  │
│   │   │   │              │    └─────────────────┬───────────────────────┘│ │  │
│   │   │   └──────────────┘                      │                        │ │  │
│   │   │                                         │                        │ │  │
│   │   │   ┌─────────────────────────────────────┼────────────────────────┤ │  │
│   │   │   │                                     │                        │ │  │
│   │   │   │   ┌──────────────────┐             ▼                        │ │  │
│   │   │   │   │   DECODER LSTM   │◀───── [y_{t-1}; c_{t-1}]             │ │  │
│   │   │   │   │                  │                                      │ │  │
│   │   │   │   │  state_t, output │                                      │ │  │
│   │   │   │   └────────┬─────────┘                                      │ │  │
│   │   │   │            │                                                │ │  │
│   │   │   │            ▼                                                │ │  │
│   │   │   │   ┌──────────────────────────────────────────────────────┐ │ │  │
│   │   │   │   │           p_gen CALCULATION                          │ │ │  │
│   │   │   │   │                                                      │ │ │  │
│   │   │   │   │  p_gen = σ(W·[c_t; state_t.c; state_t.h; x_t] + b)  │ │ │  │
│   │   │   │   │                                                      │ │ │  │
│   │   │   │   └────────────────────────┬─────────────────────────────┘ │ │  │
│   │   │   │                            │                               │ │  │
│   │   │   └────────────────────────────┼───────────────────────────────┘ │  │
│   │   │                                │                                 │  │
│   │   │   OUTPUTS:                     │                                 │  │
│   │   │   - output_t                   │                                 │  │
│   │   │   - state_t                    │                                 │  │
│   │   │   - attn_dist_t ──────────────────────────────────────────────────┼──│
│   │   │   - p_gen_t ──────────────────────────────────────────────────────┼──│
│   │   │                                │                                 │  │
│   │   └────────────────────────────────┼─────────────────────────────────┘  │
│   │                                    │                                    │
│   └────────────────────────────────────┼────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                      OUTPUT PROJECTION                                   │  │
│   │                                                                          │  │
│   │   vocab_scores = W_v · output_t + b_v                                   │  │
│   │   vocab_dists = softmax(vocab_scores)                                   │  │
│   │                                                                          │  │
│   │   Shape: [batch, vocab_size] = [16, 50000]                              │  │
│   │                                                                          │  │
│   └────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                    FINAL DISTRIBUTION                                    │  │
│   │                                                                          │  │
│   │   vocab_dists_scaled = p_gen × vocab_dists                              │  │
│   │   attn_dists_scaled = (1-p_gen) × attn_dists                            │  │
│   │                                                                          │  │
│   │   vocab_dists_extended = [vocab_dists_scaled, zeros]                    │  │
│   │   attn_dists_projected = scatter(attn_dists_scaled, enc_batch_extend)   │  │
│   │                                                                          │  │
│   │   final_dists = vocab_dists_extended + attn_dists_projected             │  │
│   │                                                                          │  │
│   │   Shape: [batch, extended_vocab] = [16, 50003]                          │  │
│   │                                                                          │  │
│   └────────────────────────────────────┬────────────────────────────────────┘  │
│                                        │                                       │
│                                        ▼                                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐  │
│   │                        LOSS CALCULATION                                  │  │
│   │                                                                          │  │
│   │   NLL Loss:                                                              │  │
│   │   loss_t = -log(final_dists[target_t])                                  │  │
│   │   loss = mean(mask(loss_1, ..., loss_T))                                │  │
│   │                                                                          │  │
│   │   Coverage Loss (if enabled):                                            │  │
│   │   cov_loss = Σ min(attn_t, coverage_t)                                  │  │
│   │                                                                          │  │
│   │   Total Loss:                                                            │  │
│   │   total_loss = loss + λ × cov_loss                                      │  │
│   │                                                                          │  │
│   └─────────────────────────────────────────────────────────────────────────┘  │
│                                                                                 │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Tensor Shapes Summary

| Component | Tensor | Shape | Example |
|-----------|--------|-------|---------|
| Input | enc_batch | [batch, enc_len] | [16, 350] |
| Input | enc_batch_extend_vocab | [batch, enc_len] | [16, 350] |
| Input | dec_batch | [batch, dec_len] | [16, 100] |
| Input | target_batch | [batch, dec_len] | [16, 100] |
| Embedding | emb_enc_inputs | [batch, enc_len, emb_dim] | [16, 350, 128] |
| Embedding | emb_dec_inputs | list of [batch, emb_dim] | 100 × [16, 128] |
| Encoder | encoder_outputs | [batch, enc_len, 2*hidden] | [16, 350, 512] |
| Encoder | fw_st, bw_st | (c, h): [batch, hidden] | [16, 256] each |
| Reduce | dec_in_state | (c, h): [batch, hidden] | [16, 256] each |
| Decoder | output | [batch, hidden] | [16, 256] |
| Decoder | attn_dist | [batch, enc_len] | [16, 350] |
| Decoder | p_gen | [batch, 1] | [16, 1] |
| Decoder | coverage | [batch, enc_len] | [16, 350] |
| Output | vocab_scores | [batch, vocab_size] | [16, 50000] |
| Output | vocab_dists | [batch, vocab_size] | [16, 50000] |
| Final | final_dists | [batch, ext_vocab] | [16, 50003] |

---

*Next: [04_attention_mechanism.md](04_attention_mechanism.md) - Attention Mechanism Deep Dive*
