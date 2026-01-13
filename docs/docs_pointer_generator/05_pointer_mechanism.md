# Pointer Mechanism and Copy Distribution

## Table of Contents
1. [The Pointer Concept](#the-pointer-concept)
2. [Generation Probability (p_gen)](#generation-probability-p_gen)
3. [Copy Distribution](#copy-distribution)
4. [Extended Vocabulary](#extended-vocabulary)
5. [Final Distribution Calculation](#final-distribution-calculation)
6. [Code Implementation](#code-implementation)
7. [Worked Examples](#worked-examples)

---

## The Pointer Concept

### Core Idea

The pointer mechanism allows the model to **copy** words directly from the source text, rather than generating from a fixed vocabulary.

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     THE POINTER MECHANISM CORE IDEA                               │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   TRADITIONAL SEQ2SEQ:                                                            │
│   ────────────────────                                                            │
│                                                                                   │
│   Decoder state sₜ                                                                │
│         │                                                                         │
│         ▼                                                                         │
│   ┌─────────────────┐                                                            │
│   │  Linear Layer   │                                                            │
│   │  W: [256, 50000]│                                                            │
│   └────────┬────────┘                                                            │
│            │                                                                      │
│            ▼                                                                      │
│   ┌─────────────────┐                                                            │
│   │    Softmax      │                                                            │
│   │  (50000 words)  │                                                            │
│   └────────┬────────┘                                                            │
│            │                                                                      │
│            ▼                                                                      │
│   Only vocabulary words can be generated!                                        │
│   OOV words → [UNK] token                                                        │
│                                                                                   │
│                                                                                   │
│   POINTER MECHANISM:                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   Decoder state sₜ           Encoder states [h₁, h₂, ..., hₙ]                   │
│         │                              │                                          │
│         └──────────┬───────────────────┘                                          │
│                    │                                                              │
│                    ▼                                                              │
│   ┌─────────────────────────────────────────┐                                    │
│   │         ATTENTION MECHANISM              │                                    │
│   │  e_i = v^T · tanh(W_h·h_i + W_s·s_t)    │                                    │
│   │  α_i = softmax(e_i)                      │                                    │
│   └────────────────┬────────────────────────┘                                    │
│                    │                                                              │
│                    ▼                                                              │
│   ┌─────────────────────────────────────────┐                                    │
│   │    COPY DISTRIBUTION                     │                                    │
│   │    P(copy word at pos i) = α_i          │                                    │
│   └────────────────┬────────────────────────┘                                    │
│                    │                                                              │
│                    ▼                                                              │
│   ANY word from source can be copied!                                            │
│   Including OOV words, names, numbers!                                           │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Why Pointing Works

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       WHY POINTING IS POWERFUL                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Consider this source text:                                                      │
│                                                                                   │
│   "The Maracanã Stadium in Rio de Janeiro hosted the 2014 World Cup final."     │
│                                                                                   │
│   With vocabulary size 50,000:                                                    │
│                                                                                   │
│   OOV words: "Maracanã", "Rio", "Janeiro", "2014"                               │
│                                                                                   │
│                                                                                   │
│   WITHOUT POINTER:                                                                │
│   ────────────────                                                                │
│   Input:  "The [UNK] Stadium in [UNK] de [UNK] hosted the [UNK] World Cup..."   │
│   Output: "The World Cup was held at the [UNK] stadium."                         │
│                                                                                   │
│   Problems:                                                                        │
│   • Lost the stadium name                                                        │
│   • Lost the city name                                                           │
│   • Lost the year                                                                │
│   • Summary is vague and uninformative                                           │
│                                                                                   │
│                                                                                   │
│   WITH POINTER:                                                                   │
│   ─────────────                                                                   │
│   Input:  "The Maracanã Stadium in Rio de Janeiro hosted the 2014 World Cup..."│
│           [positions: 0, 1, 2, 3, 4, 5, 6, 7, ...]                               │
│                                                                                   │
│   When generating:                                                                │
│   • For "Maracanã": Point to position 1 → Copy "Maracanã"                       │
│   • For "Rio": Point to position 4 → Copy "Rio"                                  │
│   • For "2014": Point to position 9 → Copy "2014"                               │
│                                                                                   │
│   Output: "The 2014 World Cup final was at the Maracanã in Rio."                │
│                                                                                   │
│   Benefits:                                                                        │
│   • Preserves proper nouns exactly                                               │
│   • Handles numbers correctly                                                    │
│   • Maintains factual accuracy                                                   │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Generation Probability (p_gen)

### The Soft Switch

The model doesn't hard-code when to copy vs. generate. Instead, it learns a **soft switch** called p_gen:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    GENERATION PROBABILITY (p_gen)                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FORMULA:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   p_gen = σ(w_c^T · c_t + w_s^T · s_t + w_x^T · x_t + b_ptr)                    │
│                                                                                   │
│   Where:                                                                          │
│   • c_t    = Context vector at step t        [512 dims]                          │
│   • s_t    = Decoder state at step t         [256 dims each for c and h]         │
│   • x_t    = Decoder input at step t         [128 dims]                          │
│   • w_c, w_s, w_x = Learnable weight vectors                                     │
│   • b_ptr  = Learnable bias scalar                                               │
│   • σ      = Sigmoid function (output ∈ [0, 1])                                  │
│                                                                                   │
│                                                                                   │
│   INTERPRETATION:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   p_gen ≈ 1.0: "I should GENERATE from vocabulary"                              │
│   p_gen ≈ 0.0: "I should COPY from source"                                       │
│   p_gen ≈ 0.5: "Blend both equally"                                              │
│                                                                                   │
│                                                                                   │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │                   p_gen CALCULATION FLOW                                │    │
│   │                                                                         │    │
│   │   ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │    │
│   │   │ Context c_t │  │ State s_t.c │  │ State s_t.h │  │  Input x_t  │  │    │
│   │   │  [512]      │  │   [256]     │  │   [256]     │  │   [128]     │  │    │
│   │   └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  │    │
│   │          │                │                │                │          │    │
│   │          ▼                ▼                ▼                ▼          │    │
│   │   ┌──────────────────────────────────────────────────────────────┐   │    │
│   │   │                    CONCATENATE                                │   │    │
│   │   │              [512 + 256 + 256 + 128 = 1152]                  │   │    │
│   │   └────────────────────────┬─────────────────────────────────────┘   │    │
│   │                            │                                         │    │
│   │                            ▼                                         │    │
│   │   ┌──────────────────────────────────────────────────────────────┐   │    │
│   │   │                 LINEAR LAYER                                  │   │    │
│   │   │            W: [1152, 1] + bias                               │   │    │
│   │   └────────────────────────┬─────────────────────────────────────┘   │    │
│   │                            │                                         │    │
│   │                            ▼                                         │    │
│   │   ┌──────────────────────────────────────────────────────────────┐   │    │
│   │   │                  SIGMOID                                      │   │    │
│   │   │              p_gen = σ(score)                                │   │    │
│   │   │              p_gen ∈ [0, 1]                                   │   │    │
│   │   └──────────────────────────────────────────────────────────────┘   │    │
│   │                                                                         │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│                                                                                   │
│   WHAT INFLUENCES p_gen?                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   The model learns when to copy based on:                                        │
│                                                                                   │
│   1. Context vector (c_t):                                                        │
│      • If attention is focused → probably should copy                            │
│      • If attention is diffuse → might need to generate                          │
│                                                                                   │
│   2. Decoder state (s_t):                                                         │
│      • Encodes what has been generated so far                                    │
│      • "Am I in a copying mode or generating mode?"                              │
│                                                                                   │
│   3. Decoder input (x_t):                                                         │
│      • The previous word generated                                               │
│      • If previous word was copied → might continue copying                      │
│      • If previous word was common → might switch to generating                  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### p_gen in Practice

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      p_gen VALUES IN PRACTICE                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Example source: "John Smith visited Paris last Tuesday."                       │
│   Target summary: "Smith went to Paris on Tuesday."                              │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Generating "Smith":                                                             │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   p_gen ≈ 0.1  ← LOW (should copy)                                              │
│                                                                                   │
│   Why? "Smith" is:                                                               │
│   • A proper noun (likely OOV)                                                   │
│   • Present in source                                                            │
│   • Attention focused on position 1                                              │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Generating "went":                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   p_gen ≈ 0.8  ← HIGH (should generate)                                         │
│                                                                                   │
│   Why? "went" is:                                                                 │
│   • A common verb (in vocabulary)                                                │
│   • NOT in source (paraphrase of "visited")                                      │
│   • Attention diffuse across action-related words                                │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Generating "to":                                                                │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   p_gen ≈ 0.6  ← MEDIUM (either could work)                                     │
│                                                                                   │
│   Why? "to" is:                                                                   │
│   • Very common word                                                             │
│   • Likely in source somewhere                                                   │
│   • Not critical to copy exactly                                                 │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Generating "Paris":                                                             │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   p_gen ≈ 0.05  ← VERY LOW (must copy)                                          │
│                                                                                   │
│   Why? "Paris" is:                                                               │
│   • A proper noun (possibly OOV)                                                 │
│   • Exactly in source                                                            │
│   • Attention sharply focused on position 3                                      │
│   • Copying ensures exact match                                                  │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   Generating "Tuesday":                                                           │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   p_gen ≈ 0.15  ← LOW (prefer copy)                                             │
│                                                                                   │
│   Why? "Tuesday" is:                                                             │
│   • Exact word in source                                                         │
│   • Could be in vocab, but copying is safer                                      │
│   • Attention focused on position 5                                              │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Copy Distribution

### From Attention to Copy Probability

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         COPY DISTRIBUTION                                         │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   The copy distribution is simply the ATTENTION DISTRIBUTION!                    │
│                                                                                   │
│   P_copy(word w) = Σ_{i: source[i] = w} α_ti                                    │
│                                                                                   │
│   In other words: The probability of copying word w is the sum of attention      │
│   weights on all positions where w appears in the source.                        │
│                                                                                   │
│                                                                                   │
│   EXAMPLE:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   Source: "The cat sat on the mat"                                               │
│   Positions: [0]  [1]  [2]  [3] [4]  [5]                                        │
│                                                                                   │
│   Attention: [0.05, 0.3, 0.15, 0.05, 0.1, 0.35]                                  │
│                                                                                   │
│   Copy probabilities:                                                             │
│   • P_copy("The") = 0.05 + 0.10 = 0.15  (appears at positions 0 and 4)          │
│   • P_copy("cat") = 0.30                 (appears at position 1)                 │
│   • P_copy("sat") = 0.15                 (appears at position 2)                 │
│   • P_copy("on")  = 0.05                 (appears at position 3)                 │
│   • P_copy("mat") = 0.35                 (appears at position 5)                 │
│                                                                                   │
│   Sum = 0.15 + 0.30 + 0.15 + 0.05 + 0.35 = 1.00 ✓                              │
│                                                                                   │
│                                                                                   │
│   KEY INSIGHT:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Words that appear MULTIPLE times in the source get HIGHER copy probability    │
│   because their attention weights are summed!                                    │
│                                                                                   │
│   This makes sense: If a word appears many times, it's probably important.      │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Extended Vocabulary

### Handling Out-of-Vocabulary Words

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         EXTENDED VOCABULARY                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   The vocabulary is EXTENDED for each article with that article's OOV words:    │
│                                                                                   │
│   Base Vocabulary (50,000 words):                                                │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │ Index 0:     [PAD]                                                      │    │
│   │ Index 1:     [UNK]                                                      │    │
│   │ Index 2:     [START]                                                    │    │
│   │ Index 3:     [STOP]                                                     │    │
│   │ Index 4:     the                                                        │    │
│   │ Index 5:     a                                                          │    │
│   │ Index 6:     is                                                         │    │
│   │ ...                                                                      │    │
│   │ Index 49999: (50000th most common word)                                 │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│                                                                                   │
│   For article: "John Smith visited Paris"                                        │
│   OOV words:   ["John", "Smith", "Paris"]                                        │
│                                                                                   │
│   Extended Vocabulary (50,003 words):                                            │
│   ┌────────────────────────────────────────────────────────────────────────┐    │
│   │ Index 0-49999:  Base vocabulary                                         │    │
│   │ Index 50000:    "John"   (Article OOV #0)                               │    │
│   │ Index 50001:    "Smith"  (Article OOV #1)                               │    │
│   │ Index 50002:    "Paris"  (Article OOV #2)                               │    │
│   └────────────────────────────────────────────────────────────────────────┘    │
│                                                                                   │
│                                                                                   │
│   TWO ENCODINGS OF THE SOURCE:                                                    │
│   ────────────────────────────                                                    │
│                                                                                   │
│   enc_batch (for embedding lookup):                                              │
│   "John"    → 1 (UNK)      ← Can't look up OOV embeddings                       │
│   "Smith"   → 1 (UNK)                                                            │
│   "visited" → 4523         ← Normal vocab word                                   │
│   "Paris"   → 1 (UNK)                                                            │
│                                                                                   │
│   enc_batch_extend_vocab (for copy distribution):                                │
│   "John"    → 50000        ← First article OOV                                  │
│   "Smith"   → 50001        ← Second article OOV                                 │
│   "visited" → 4523         ← Same as enc_batch                                  │
│   "Paris"   → 50002        ← Third article OOV                                  │
│                                                                                   │
│                                                                                   │
│   WHY TWO ENCODINGS?                                                              │
│   ──────────────────                                                              │
│                                                                                   │
│   1. enc_batch: Used for embedding lookup                                        │
│      • OOVs get [UNK] embedding                                                  │
│      • Still learn something useful from context                                 │
│                                                                                   │
│   2. enc_batch_extend_vocab: Used for copy mechanism                             │
│      • OOVs get unique temporary IDs                                             │
│      • Allows copying exact OOV words                                            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### OOV Handling Functions

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                       OOV HANDLING FUNCTIONS                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   data.py: article2ids()                                                         │
│   ──────────────────────                                                          │
│                                                                                   │
│   def article2ids(article_words, vocab):                                         │
│       """Map article words to IDs, tracking OOVs."""                             │
│       ids = []                                                                    │
│       oovs = []                                                                   │
│       unk_id = vocab.word2id(UNKNOWN_TOKEN)                                      │
│                                                                                   │
│       for w in article_words:                                                    │
│           i = vocab.word2id(w)                                                   │
│           if i == unk_id:  # OOV word                                            │
│               if w not in oovs:                                                  │
│                   oovs.append(w)                                                 │
│               oov_num = oovs.index(w)                                            │
│               ids.append(vocab.size() + oov_num)  # 50000+                       │
│           else:                                                                   │
│               ids.append(i)                                                      │
│       return ids, oovs                                                           │
│                                                                                   │
│                                                                                   │
│   Example:                                                                        │
│   article_words = ["John", "visited", "Paris", "John"]                           │
│                                                                                   │
│   Processing:                                                                     │
│   • "John":    OOV → oovs=["John"], id=50000                                    │
│   • "visited": in vocab → id=4523                                                │
│   • "Paris":   OOV → oovs=["John", "Paris"], id=50001                           │
│   • "John":    OOV (already in oovs) → id=50000  (same as first!)               │
│                                                                                   │
│   Result:                                                                         │
│   ids = [50000, 4523, 50001, 50000]                                              │
│   oovs = ["John", "Paris"]                                                       │
│                                                                                   │
│                                                                                   │
│   data.py: abstract2ids()                                                        │
│   ────────────────────────                                                        │
│                                                                                   │
│   def abstract2ids(abstract_words, vocab, article_oovs):                         │
│       """Map abstract words to IDs, using article OOVs."""                       │
│       ids = []                                                                    │
│       unk_id = vocab.word2id(UNKNOWN_TOKEN)                                      │
│                                                                                   │
│       for w in abstract_words:                                                   │
│           i = vocab.word2id(w)                                                   │
│           if i == unk_id:  # OOV word                                            │
│               if w in article_oovs:  # In-article OOV                            │
│                   ids.append(vocab.size() + article_oovs.index(w))               │
│               else:  # Out-of-article OOV                                        │
│                   ids.append(unk_id)  # Can't copy, use UNK                     │
│           else:                                                                   │
│               ids.append(i)                                                      │
│       return ids                                                                  │
│                                                                                   │
│   Key distinction:                                                                │
│   • In-article OOV: Can be copied → Use temporary ID                            │
│   • Out-of-article OOV: Can't be copied → Use UNK                               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Final Distribution Calculation

### Combining Generation and Copy

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   FINAL DISTRIBUTION CALCULATION                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   model.py: _calc_final_dist() method                                            │
│   ────────────────────────────────────                                            │
│                                                                                   │
│   The final probability of generating word w is:                                 │
│                                                                                   │
│   P(w) = p_gen × P_vocab(w) + (1 - p_gen) × P_copy(w)                           │
│              ↑                         ↑                                          │
│        Generation                  Copy from                                      │
│        probability                 source                                         │
│                                                                                   │
│                                                                                   │
│   STEP-BY-STEP CALCULATION:                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: Scale distributions                                                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   vocab_dists_scaled = p_gen × vocab_dists                                       │
│   attn_dists_scaled  = (1 - p_gen) × attn_dists                                 │
│                                                                                   │
│   Example with p_gen = 0.3:                                                       │
│   vocab_dists = [0.0, 0.1, 0.05, ..., 0.15, ...]  (50000 values)               │
│   vocab_dists_scaled = 0.3 × [0.0, 0.1, ...] = [0.0, 0.03, 0.015, ..., 0.045]  │
│                                                                                   │
│   attn_dists = [0.6, 0.1, 0.05, 0.15, 0.1]  (over 5 source positions)          │
│   attn_dists_scaled = 0.7 × [0.6, ...] = [0.42, 0.07, 0.035, 0.105, 0.07]      │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: Extend vocab distribution with zeros for OOVs                          │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   extended_vsize = vocab_size + max_art_oovs                                     │
│                  = 50000 + 2 = 50002  (if article has 2 OOVs)                    │
│                                                                                   │
│   vocab_dists_extended = concat(vocab_dists_scaled, [0, 0])                      │
│   Shape: [batch, 50002]                                                          │
│                                                                                   │
│   [P(PAD), P(UNK), ..., P(word_49999), 0, 0]                                    │
│                                        ↑  ↑                                      │
│                                   Slots for OOVs                                 │
│                                   (generation prob = 0)                          │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: Project attention to vocabulary indices (SCATTER)                      │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   This is the KEY operation!                                                     │
│                                                                                   │
│   Source:          ["John",  "visited", "Paris", ".", "John"]                   │
│   Extended IDs:    [50000,   4523,      50001,  789,  50000]                     │
│   Scaled attention:[0.42,    0.07,      0.105,  0.07, 0.35]                      │
│                                                                                   │
│   Initialize: attn_dists_projected = zeros([50002])                              │
│                                                                                   │
│   Scatter operation:                                                              │
│   • Position 0: word_id=50000, attn=0.42                                         │
│     attn_dists_projected[50000] += 0.42  → [50000] = 0.42                       │
│                                                                                   │
│   • Position 1: word_id=4523, attn=0.07                                          │
│     attn_dists_projected[4523] += 0.07   → [4523] = 0.07                        │
│                                                                                   │
│   • Position 2: word_id=50001, attn=0.105                                        │
│     attn_dists_projected[50001] += 0.105 → [50001] = 0.105                      │
│                                                                                   │
│   • Position 3: word_id=789, attn=0.07                                           │
│     attn_dists_projected[789] += 0.07    → [789] = 0.07                         │
│                                                                                   │
│   • Position 4: word_id=50000, attn=0.35                                         │
│     attn_dists_projected[50000] += 0.35  → [50000] = 0.42 + 0.35 = 0.77        │
│                                                                                   │
│   Result:                                                                         │
│   attn_dists_projected[789] = 0.07     (".")                                    │
│   attn_dists_projected[4523] = 0.07    ("visited")                              │
│   attn_dists_projected[50000] = 0.77   ("John" - appears twice!)                │
│   attn_dists_projected[50001] = 0.105  ("Paris")                                │
│                                                                                   │
│   Note: "John" gets 0.77 because it appears at TWO positions!                   │
│         0.42 + 0.35 = 0.77                                                       │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 4: Add the two distributions                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   final_dists = vocab_dists_extended + attn_dists_projected                      │
│                                                                                   │
│   For each word/OOV index i:                                                     │
│   final_dists[i] = p_gen × P_vocab[i] + (1-p_gen) × P_copy[i]                   │
│                                                                                   │
│   Examples:                                                                       │
│   • "John" (index 50000):                                                        │
│     vocab_dists_extended[50000] = 0.0  (OOV, can't generate)                    │
│     attn_dists_projected[50000] = 0.77 (from copy)                              │
│     final_dists[50000] = 0.0 + 0.77 = 0.77                                      │
│                                                                                   │
│   • "visited" (index 4523):                                                      │
│     vocab_dists_extended[4523] = 0.045 (some generation prob)                   │
│     attn_dists_projected[4523] = 0.07  (from copy)                              │
│     final_dists[4523] = 0.045 + 0.07 = 0.115                                    │
│                                                                                   │
│   • "the" (index 8, not in source):                                              │
│     vocab_dists_extended[8] = 0.06                                              │
│     attn_dists_projected[8] = 0.0 (not in source)                               │
│     final_dists[8] = 0.06 + 0.0 = 0.06                                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Implementation

### _calc_final_dist() Method

```python
# model.py: Lines 146-183 (annotated)

def _calc_final_dist(self, vocab_dists, attn_dists):
    """Calculate the final distribution, for the pointer-generator model

    Args:
      vocab_dists: The vocabulary distributions. 
                   List length max_dec_steps of (batch_size, vsize) arrays.
      attn_dists: The attention distributions. 
                  List length max_dec_steps of (batch_size, attn_len) arrays

    Returns:
      final_dists: The final distributions. 
                   List length max_dec_steps of (batch_size, extended_vsize) arrays.
    """
    with tf.variable_scope('final_distribution'):
        # STEP 1: Multiply vocab dists by p_gen and attention dists by (1-p_gen)
        vocab_dists = [p_gen * dist for (p_gen, dist) in zip(self.p_gens, vocab_dists)]
        attn_dists = [(1-p_gen) * dist for (p_gen, dist) in zip(self.p_gens, attn_dists)]

        # STEP 2: Concatenate zeros for OOV slots
        extended_vsize = self._vocab.size() + self._max_art_oovs
        extra_zeros = tf.zeros((self._hps.batch_size, self._max_art_oovs))
        vocab_dists_extended = [tf.concat(axis=1, values=[dist, extra_zeros]) 
                                for dist in vocab_dists]

        # STEP 3: Project attention onto vocabulary using scatter_nd
        batch_nums = tf.range(0, limit=self._hps.batch_size)
        batch_nums = tf.expand_dims(batch_nums, 1)
        attn_len = tf.shape(self._enc_batch_extend_vocab)[1]
        batch_nums = tf.tile(batch_nums, [1, attn_len])
        
        # indices[batch, position] = (batch_index, word_index)
        indices = tf.stack((batch_nums, self._enc_batch_extend_vocab), axis=2)
        shape = [self._hps.batch_size, extended_vsize]
        
        # scatter_nd: Add attention probs to their word indices
        attn_dists_projected = [tf.scatter_nd(indices, copy_dist, shape) 
                                for copy_dist in attn_dists]

        # STEP 4: Add vocab and copy distributions
        final_dists = [vocab_dist + copy_dist 
                       for (vocab_dist, copy_dist) 
                       in zip(vocab_dists_extended, attn_dists_projected)]

        return final_dists
```

### Understanding scatter_nd

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      tf.scatter_nd OPERATION                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   scatter_nd(indices, updates, shape)                                            │
│                                                                                   │
│   Creates a tensor of given shape, then adds updates at specified indices.       │
│                                                                                   │
│                                                                                   │
│   EXAMPLE:                                                                        │
│   ────────                                                                        │
│                                                                                   │
│   Batch size = 2, Extended vocab size = 50002                                    │
│                                                                                   │
│   Batch 0:                                                                        │
│   Source words: ["John", "went", "home"]                                         │
│   Extended IDs: [50000,  234,    567]                                            │
│   Attention:    [0.6,    0.25,   0.15]                                           │
│                                                                                   │
│   Batch 1:                                                                        │
│   Source words: ["Mary", "went", "away"]                                         │
│   Extended IDs: [50001,  234,    890]                                            │
│   Attention:    [0.5,    0.3,    0.2]                                            │
│                                                                                   │
│                                                                                   │
│   indices tensor:                                                                 │
│   [[[0, 50000], [0, 234], [0, 567]],     # Batch 0                              │
│    [[1, 50001], [1, 234], [1, 890]]]     # Batch 1                              │
│                                                                                   │
│   updates tensor (attention weights):                                             │
│   [[0.6, 0.25, 0.15],                     # Batch 0                             │
│    [0.5, 0.3,  0.2]]                      # Batch 1                             │
│                                                                                   │
│   shape = [2, 50002]                                                             │
│                                                                                   │
│                                                                                   │
│   Result of scatter_nd:                                                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │         0      ...   234    ...   567    ...  50000  50001              │   │
│   │ Batch 0: [0.0,  ...,  0.25, ...,  0.15, ...,  0.6,   0.0]              │   │
│   │ Batch 1: [0.0,  ...,  0.3,  ...,  0.0,  ...,  0.0,   0.5]              │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│   Shape: [2, 50002]                                                              │
│                                                                                   │
│   Note: Index 234 ("went") gets value in both batches                           │
│   Note: Index 50000 ("John") only has value in batch 0                          │
│   Note: Index 50001 ("Mary") only has value in batch 1                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Worked Examples

### Example 1: Copying a Proper Noun

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE: COPYING "ARGENTINA"                                   │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Source: "Germany beat Argentina 1-0"                                           │
│   Current output: "Germany defeated ___"                                         │
│   Target: "Argentina"                                                            │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   ATTENTION DISTRIBUTION:                                                         │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Source:     Germany  beat  Argentina  1-0                                      │
│   Position:      0       1       2       3                                       │
│   Attention:   0.05    0.08    0.82    0.05                                     │
│                                                                                   │
│   The model strongly attends to position 2 ("Argentina")                         │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   p_gen CALCULATION:                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Context vector: c = 0.05×h₀ + 0.08×h₁ + 0.82×h₂ + 0.05×h₃                    │
│                     ≈ h₂ (dominated by Argentina's hidden state)                │
│                                                                                   │
│   Decoder state: s = (processing "Germany defeated ...")                         │
│   Decoder input: x = embedding("defeated")                                       │
│                                                                                   │
│   p_gen = σ(W·[c; s.c; s.h; x] + b)                                             │
│         = σ(-3.5)  ← Model learned Argentina should be copied                   │
│         ≈ 0.03     ← Very low! Strongly favors copying                          │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   VOCABULARY DISTRIBUTION (scaled by p_gen = 0.03):                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Before scaling:                                                                 │
│   P_vocab("argentina") = 0.15  (lowercase version in vocab)                      │
│   P_vocab("the")       = 0.08                                                    │
│   P_vocab("country")   = 0.05                                                    │
│   ...                                                                             │
│                                                                                   │
│   After scaling (× 0.03):                                                         │
│   P_vocab_scaled("argentina") = 0.0045                                           │
│   P_vocab_scaled("the")       = 0.0024                                           │
│   P_vocab_scaled("country")   = 0.0015                                           │
│                                                                                   │
│   Note: "Argentina" (capitalized, OOV) has P_vocab = 0                           │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   COPY DISTRIBUTION (scaled by 1-p_gen = 0.97):                                  │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Source IDs:     [50000,   234,   50001,  50002]                               │
│                   Germany  beat  Argentina  1-0                                  │
│                   (OOV)   (vocab)  (OOV)   (OOV)                                 │
│                                                                                   │
│   Attention:      [0.05,   0.08,   0.82,   0.05]                                │
│   Scaled (×0.97): [0.049,  0.078,  0.795,  0.049]                               │
│                                                                                   │
│   After scatter:                                                                  │
│   attn_projected[234]   = 0.078   (index for "beat")                            │
│   attn_projected[50000] = 0.049   (index for "Germany")                         │
│   attn_projected[50001] = 0.795   (index for "Argentina")  ← HIGHEST!          │
│   attn_projected[50002] = 0.049   (index for "1-0")                             │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   FINAL DISTRIBUTION:                                                             │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   final_dist = vocab_dist_scaled + attn_projected                                │
│                                                                                   │
│   final_dist[50001]      = 0.0 + 0.795 = 0.795   "Argentina" (OOV) ← WINNER!   │
│   final_dist[234]        = 0.001 + 0.078 = 0.079 "beat"                         │
│   final_dist[50000]      = 0.0 + 0.049 = 0.049   "Germany" (OOV)               │
│   final_dist[vocab_idx]  = 0.0045 + 0 = 0.0045   "argentina" (lowercase)       │
│   ...                                                                             │
│                                                                                   │
│   argmax(final_dist) = 50001 = "Argentina" ✓                                    │
│                                                                                   │
│   The model correctly COPIES "Argentina" from the source!                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Example 2: Generating a Paraphrase

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE: GENERATING "DEFEATED"                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Source: "Germany beat Argentina 1-0"                                           │
│   Current output: "Germany ___"                                                  │
│   Target: "defeated" (paraphrase of "beat")                                      │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   ATTENTION DISTRIBUTION:                                                         │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Source:     Germany  beat  Argentina  1-0                                      │
│   Attention:   0.10    0.65    0.15    0.10                                     │
│                                                                                   │
│   Attention focuses on "beat" but not as sharply as for copying                 │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   p_gen CALCULATION:                                                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   The model needs to GENERATE a paraphrase, not copy "beat"                     │
│                                                                                   │
│   p_gen = σ(W·[c; s.c; s.h; x] + b)                                             │
│         = σ(1.8)                                                                 │
│         ≈ 0.86    ← HIGH! Strongly favors generation                            │
│                                                                                   │
│   Why? The model learned:                                                        │
│   • "beat" is a common word (can generate alternatives)                          │
│   • Context suggests an action verb is needed                                    │
│   • Better style to paraphrase sometimes                                         │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   VOCABULARY DISTRIBUTION (scaled by p_gen = 0.86):                              │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Before scaling:                                                                 │
│   P_vocab("defeated") = 0.35                                                     │
│   P_vocab("beat")     = 0.20                                                     │
│   P_vocab("won")      = 0.15                                                     │
│   P_vocab("against")  = 0.08                                                     │
│   ...                                                                             │
│                                                                                   │
│   After scaling (× 0.86):                                                         │
│   P_vocab_scaled("defeated") = 0.301                                             │
│   P_vocab_scaled("beat")     = 0.172                                             │
│   P_vocab_scaled("won")      = 0.129                                             │
│   P_vocab_scaled("against")  = 0.069                                             │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   COPY DISTRIBUTION (scaled by 1-p_gen = 0.14):                                  │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Attention scaled:  [0.014,  0.091,  0.021,  0.014]                            │
│                      Germany  beat   Argentina  1-0                              │
│                                                                                   │
│   attn_projected[beat_idx] = 0.091                                               │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   FINAL DISTRIBUTION:                                                             │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   final_dist["defeated"] = 0.301 + 0.0   = 0.301  ← WINNER!                     │
│   final_dist["beat"]     = 0.172 + 0.091 = 0.263                                │
│   final_dist["won"]      = 0.129 + 0.0   = 0.129                                │
│                                                                                   │
│   argmax(final_dist) = "defeated" ✓                                             │
│                                                                                   │
│   The model correctly GENERATES "defeated" instead of copying "beat"!           │
│                                                                                   │
│   Note: "beat" still has decent probability because:                            │
│   - Some vocab probability (0.172)                                               │
│   - Some copy probability (0.091)                                                │
│   Both contribute, but "defeated" wins on vocab alone!                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

The Pointer Mechanism enables:

1. **Copying OOV words** - Proper nouns, numbers, rare words
2. **Soft switching** - p_gen learns when to copy vs. generate
3. **Extended vocabulary** - Temporary IDs for article-specific OOVs
4. **Hybrid distribution** - Best of both generation and copying

Key formulas:
- **p_gen** = σ(W · [context, state, input] + b)
- **P_copy(w)** = Σ attention on positions containing w
- **P_final(w)** = p_gen × P_vocab(w) + (1-p_gen) × P_copy(w)

---

*Next: [06_coverage_mechanism.md](06_coverage_mechanism.md) - Coverage Mechanism and Repetition Prevention*
