# Vocabulary and OOV Handling Deep Dive

## Table of Contents
1. [Vocabulary Fundamentals](#vocabulary-fundamentals)
2. [Special Tokens](#special-tokens)
3. [OOV Problem](#oov-problem)
4. [Dual Encoding Strategy](#dual-encoding-strategy)
5. [Article to IDs Conversion](#article-to-ids-conversion)
6. [Abstract to IDs Conversion](#abstract-to-ids-conversion)
7. [Output to Words Conversion](#output-to-words-conversion)
8. [Extended Vocabulary in Practice](#extended-vocabulary-in-practice)
9. [Complete Worked Example](#complete-worked-example)

---

## Vocabulary Fundamentals

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     VOCABULARY FUNDAMENTALS                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHAT IS A VOCABULARY?                                                           │
│   ─────────────────────                                                           │
│                                                                                   │
│   A vocabulary is a mapping between words and integer IDs:                        │
│                                                                                   │
│   Word          ID              Word          ID                                 │
│   ────          ──              ────          ──                                 │
│   [PAD]         0               cup           104                                │
│   [UNK]         1               world         103                                │
│   [START]       2               germany       100                                │
│   [STOP]        3               argentina     102                                │
│   the           4               ...           ...                                │
│   .             5               (word n)      49999                              │
│   a             6                                                                 │
│   ...           ...                                                               │
│                                                                                   │
│                                                                                   │
│   WHY DO WE NEED IDs?                                                             │
│   ───────────────────                                                             │
│                                                                                   │
│   Neural networks work with numbers, not words:                                  │
│                                                                                   │
│   1. Embedding lookup requires integer indices                                   │
│      embedding = embedding_matrix[word_id]  # shape [emb_dim]                   │
│                                                                                   │
│   2. Output layer produces probability over vocabulary                           │
│      P(word) = softmax(logits)[word_id]  # shape [vocab_size]                   │
│                                                                                   │
│   3. Efficient computation with matrix operations                                │
│      batch_embeddings = embedding_matrix[batch_word_ids]                        │
│                                                                                   │
│                                                                                   │
│   VOCABULARY SIZE TRADEOFF:                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   Small vocab (e.g., 10K):                                                       │
│   ✓ Faster training (smaller embedding matrix)                                  │
│   ✓ More frequent words have better representations                             │
│   ✗ More OOV words                                                              │
│                                                                                   │
│   Large vocab (e.g., 200K):                                                      │
│   ✓ Fewer OOV words                                                             │
│   ✗ Slower training                                                             │
│   ✗ Rare words have poor representations (few examples)                         │
│                                                                                   │
│   Default in pointer-generator: 50,000 words                                     │
│   (A good balance for news summarization)                                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Special Tokens

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         SPECIAL TOKENS                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   TOKEN          ID      PURPOSE                                                 │
│   ─────          ──      ───────                                                 │
│                                                                                   │
│   [PAD]          0       PADDING TOKEN                                           │
│                          ─────────────                                            │
│                          Used to pad sequences to uniform length.                │
│                          Should be masked out in attention and loss.             │
│                                                                                   │
│                          Example:                                                 │
│                          Sequence 1: [45, 67, 89, 0, 0, 0]  ← 3 PAD tokens      │
│                          Sequence 2: [12, 34, 56, 78, 90, 0] ← 1 PAD token      │
│                                                                                   │
│   ──────────────────────────────────────────────────────────────────────────     │
│                                                                                   │
│   [UNK]          1       UNKNOWN TOKEN                                           │
│                          ─────────────                                            │
│                          Represents any word NOT in vocabulary.                  │
│                          Used in encoder input for embedding lookup.             │
│                                                                                   │
│                          Example:                                                 │
│                          Word "Ronaldo" not in vocab → ID 1                      │
│                          Word "cryptocurrency" not in vocab → ID 1               │
│                                                                                   │
│                          Problem: All OOV words map to same embedding!           │
│                          Solution: Pointer mechanism can copy original word.     │
│                                                                                   │
│   ──────────────────────────────────────────────────────────────────────────     │
│                                                                                   │
│   [START]        2       START DECODING TOKEN                                    │
│                          ───────────────────                                      │
│                          First input to decoder at each generation step.         │
│                          Signals the beginning of output sequence.               │
│                                                                                   │
│                          Decoder input sequence:                                  │
│                          [START, word1, word2, ..., word_n]                      │
│                           ↓      ↓      ↓            ↓                           │
│                          word1  word2  word3  ...  STOP                          │
│                          (target sequence)                                        │
│                                                                                   │
│   ──────────────────────────────────────────────────────────────────────────     │
│                                                                                   │
│   [STOP]         3       STOP DECODING TOKEN                                     │
│                          ──────────────────                                       │
│                          Signals end of output sequence.                          │
│                          Generation stops when this is predicted.                │
│                                                                                   │
│                          Target sequence:                                        │
│                          [word1, word2, ..., word_n, STOP]                       │
│                                                                                   │
│                          During beam search:                                      │
│                          If model outputs STOP → hypothesis is complete          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## OOV Problem

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         THE OOV PROBLEM                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHAT IS OOV?                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   OOV = Out-Of-Vocabulary                                                        │
│   Words that are NOT in the fixed vocabulary.                                    │
│                                                                                   │
│   Common OOV words:                                                               │
│   • Proper nouns: "Ronaldo", "Tesla", "COVID-19"                                │
│   • Technical terms: "cryptocurrency", "blockchain"                              │
│   • Rare words: "defenestration", "pulchritudinous"                             │
│   • Misspellings: "teh", "definately"                                           │
│   • New words: "selfie", "hashtag" (before they became common)                  │
│                                                                                   │
│                                                                                   │
│   THE PROBLEM WITH TRADITIONAL SEQ2SEQ:                                           │
│   ──────────────────────────────────────                                          │
│                                                                                   │
│   Source: "Elon Musk announced Tesla's new battery technology."                  │
│                ↑          ↑                         ↑                            │
│               OOV        OOV                       OOV                           │
│                                                                                   │
│   After conversion to IDs (vocab doesn't contain these names):                   │
│   [UNK] [UNK] announced [UNK]'s new battery technology .                        │
│                                                                                   │
│   Traditional model can ONLY output vocabulary words.                            │
│                                                                                   │
│   Generated summary: "[UNK] announced new battery technology."                  │
│                        ↑                                                          │
│                       USELESS! We need "Elon Musk"!                              │
│                                                                                   │
│                                                                                   │
│   WHY CAN'T WE JUST USE A HUGE VOCABULARY?                                        │
│   ─────────────────────────────────────────                                       │
│                                                                                   │
│   1. Memory: Each word needs an embedding vector                                 │
│      200K words × 128 dim × 4 bytes = 100 MB just for embeddings               │
│                                                                                   │
│   2. Computation: Softmax over huge vocabulary is slow                           │
│      P(word) = exp(score_word) / Σ exp(score_i)  ← Sum over ALL words          │
│                                                                                   │
│   3. Rare words: Even with huge vocab, some words appear rarely                 │
│      Model can't learn good representations from few examples                   │
│                                                                                   │
│   4. Dynamic words: New proper nouns appear constantly                          │
│      "GPT-4", "iPhone 15" - can't be in pre-built vocab                        │
│                                                                                   │
│                                                                                   │
│   THE POINTER-GENERATOR SOLUTION:                                                 │
│   ───────────────────────────────                                                 │
│                                                                                   │
│   Instead of forcing all outputs through vocabulary:                             │
│   • Allow COPYING words directly from source                                     │
│   • OOV words can be copied even without embeddings                             │
│   • Model learns WHEN to copy vs. generate                                       │
│                                                                                   │
│   Source: "Elon Musk announced Tesla's new battery technology."                  │
│            ↑    ↑              ↑                                                 │
│           COPY COPY           COPY (via pointer mechanism)                       │
│                                                                                   │
│   Generated: "Elon Musk announced Tesla's battery technology."                  │
│               ↑    ↑              ↑                                              │
│              COPIED from source!                                                 │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Dual Encoding Strategy

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                      DUAL ENCODING STRATEGY                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   The pointer-generator uses TWO encodings for each article:                     │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   ENCODING 1: enc_input (for embedding lookup)                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   OOV words are replaced with [UNK] (ID 1).                                      │
│                                                                                   │
│   Why? The embedding matrix only has vectors for vocabulary words.              │
│   We can't look up an embedding for a word not in the matrix!                   │
│                                                                                   │
│   Example:                                                                        │
│   Article: "Elon Musk announced Tesla's technology"                              │
│   Vocab:   {announced=100, technology=200, 's=300}                              │
│   NOT in vocab: Elon, Musk, Tesla                                               │
│                                                                                   │
│   enc_input = [1, 1, 100, 1, 300, 200]                                          │
│                ↑  ↑       ↑                                                       │
│               UNK UNK    UNK                                                     │
│                                                                                   │
│   This encoding is used to get embeddings:                                       │
│   embeddings = embedding_matrix[enc_input]                                       │
│   → All OOVs get the same [UNK] embedding                                       │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   ENCODING 2: enc_input_extend_vocab (for pointer mechanism)                     │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   OOV words get TEMPORARY IDs starting from vocab_size.                          │
│                                                                                   │
│   Why? The pointer mechanism needs to distinguish different OOV words.           │
│   When we copy, we need to know WHICH word to copy!                             │
│                                                                                   │
│   Example (continuing from above):                                               │
│   vocab_size = 50000                                                             │
│                                                                                   │
│   OOV list (article_oovs): ["Elon", "Musk", "Tesla"]                            │
│   Temporary IDs:           [50000,  50001, 50002]                               │
│                                                                                   │
│   enc_input_extend_vocab = [50000, 50001, 100, 50002, 300, 200]                 │
│                             ↑      ↑           ↑                                 │
│                            Elon   Musk        Tesla (unique IDs!)               │
│                                                                                   │
│   This encoding is used for copy distribution:                                   │
│   When model attends to position 0, it can copy "Elon" (ID 50000)              │
│   When model attends to position 1, it can copy "Musk" (ID 50001)              │
│                                                                                   │
│                                                                                   │
│   VISUALIZATION:                                                                  │
│   ──────────────                                                                  │
│                                                                                   │
│   Article:                   "Elon  Musk  announced  Tesla's  technology"       │
│                                ↓     ↓       ↓         ↓         ↓              │
│   enc_input:                 [  1,    1,    100,       1,       200  ]          │
│   (for embeddings)             ↑     ↑                 ↑                         │
│                              All OOV → [UNK]                                     │
│                                                                                   │
│   enc_input_extend_vocab:    [50000, 50001, 100,   50002,      200  ]          │
│   (for copying)                ↑      ↑            ↑                             │
│                              Unique temporary IDs                                │
│                                                                                   │
│   article_oovs:              ["Elon", "Musk", "Tesla"]                          │
│   (for decoding)               idx 0   idx 1   idx 2                            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Article to IDs Conversion

### article2ids Function (data.py)

```python
# data.py: article2ids function (Lines 93-130)

def article2ids(article_words, vocab):
    """
    Convert article words to IDs, building OOV list.
    
    Args:
        article_words: List of words in the article
        vocab: Vocab object
    
    Returns:
        ids: List of word IDs (extended vocab for OOV)
        oovs: List of OOV words
    """
    ids = []
    oovs = []
    unk_id = vocab.word2id(vocab.UNKNOWN_TOKEN)
    
    for w in article_words:
        i = vocab.word2id(w)
        
        if i == unk_id:  # Word is OOV
            if w not in oovs:
                oovs.append(w)  # Add to OOV list
            
            # Get temporary ID: vocab_size + index in oovs
            oov_num = oovs.index(w)
            ids.append(vocab.size() + oov_num)
        else:
            ids.append(i)  # In-vocab word
    
    return ids, oovs
```

### Visual Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    ARTICLE TO IDS CONVERSION                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   INPUT:                                                                          │
│   ──────                                                                          │
│   article_words = ["The", "quick", "Zephyr", "jumped", "over", "Xander"]        │
│   vocab_size = 50000                                                             │
│   Vocab contains: The=4, quick=500, jumped=600, over=50                         │
│   NOT in vocab: Zephyr, Xander                                                   │
│                                                                                   │
│                                                                                   │
│   STEP-BY-STEP EXECUTION:                                                         │
│   ────────────────────────                                                        │
│                                                                                   │
│   ids = []                                                                        │
│   oovs = []                                                                       │
│                                                                                   │
│   Word 1: "The"                                                                   │
│     vocab.word2id("The") = 4 (in vocab)                                          │
│     ids = [4]                                                                     │
│     oovs = []                                                                     │
│                                                                                   │
│   Word 2: "quick"                                                                 │
│     vocab.word2id("quick") = 500 (in vocab)                                      │
│     ids = [4, 500]                                                               │
│     oovs = []                                                                     │
│                                                                                   │
│   Word 3: "Zephyr"                                                                │
│     vocab.word2id("Zephyr") = 1 (UNK - not in vocab!)                           │
│     "Zephyr" not in oovs, so add it                                             │
│     oovs = ["Zephyr"]                                                            │
│     oov_num = oovs.index("Zephyr") = 0                                          │
│     temp_id = vocab_size + oov_num = 50000 + 0 = 50000                          │
│     ids = [4, 500, 50000]                                                        │
│                                                                                   │
│   Word 4: "jumped"                                                                │
│     vocab.word2id("jumped") = 600 (in vocab)                                     │
│     ids = [4, 500, 50000, 600]                                                   │
│     oovs = ["Zephyr"]                                                            │
│                                                                                   │
│   Word 5: "over"                                                                  │
│     vocab.word2id("over") = 50 (in vocab)                                        │
│     ids = [4, 500, 50000, 600, 50]                                              │
│     oovs = ["Zephyr"]                                                            │
│                                                                                   │
│   Word 6: "Xander"                                                                │
│     vocab.word2id("Xander") = 1 (UNK - not in vocab!)                           │
│     "Xander" not in oovs, so add it                                             │
│     oovs = ["Zephyr", "Xander"]                                                  │
│     oov_num = oovs.index("Xander") = 1                                          │
│     temp_id = vocab_size + oov_num = 50000 + 1 = 50001                          │
│     ids = [4, 500, 50000, 600, 50, 50001]                                       │
│                                                                                   │
│                                                                                   │
│   OUTPUT:                                                                         │
│   ───────                                                                         │
│   ids = [4, 500, 50000, 600, 50, 50001]                                         │
│   oovs = ["Zephyr", "Xander"]                                                    │
│                                                                                   │
│   MAPPING:                                                                        │
│   ────────                                                                        │
│   Word      ID      Type                                                         │
│   ────      ──      ────                                                         │
│   The       4       In vocab                                                     │
│   quick     500     In vocab                                                     │
│   Zephyr    50000   OOV (temp ID)                                               │
│   jumped    600     In vocab                                                     │
│   over      50      In vocab                                                     │
│   Xander    50001   OOV (temp ID)                                               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Abstract to IDs Conversion

### abstract2ids Function (data.py)

```python
# data.py: abstract2ids function (Lines 132-170)

def abstract2ids(abstract_words, vocab, article_oovs):
    """
    Convert abstract words to IDs, using article's OOV list.
    
    Args:
        abstract_words: List of words in the abstract
        vocab: Vocab object
        article_oovs: OOV list from article (from article2ids)
    
    Returns:
        ids: List of word IDs
    """
    ids = []
    unk_id = vocab.word2id(vocab.UNKNOWN_TOKEN)
    
    for w in abstract_words:
        i = vocab.word2id(w)
        
        if i == unk_id:  # Word is OOV
            if w in article_oovs:
                # OOV word is in article - use extended vocab ID
                vocab_idx = vocab.size() + article_oovs.index(w)
                ids.append(vocab_idx)
            else:
                # OOV word NOT in article - use UNK
                # (Model can't copy a word that's not in source!)
                ids.append(unk_id)
        else:
            ids.append(i)
    
    return ids
```

### Visual Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   ABSTRACT TO IDS CONVERSION                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   INPUT:                                                                          │
│   ──────                                                                          │
│   abstract_words = ["Zephyr", "jumped", "and", "Xander", "watched", "Yolanda"]  │
│   article_oovs = ["Zephyr", "Xander"]  (from article2ids)                       │
│   vocab_size = 50000                                                             │
│                                                                                   │
│   Vocab contains: jumped=600, and=10, watched=700                               │
│   NOT in vocab: Zephyr, Xander, Yolanda                                         │
│                                                                                   │
│                                                                                   │
│   STEP-BY-STEP EXECUTION:                                                         │
│   ────────────────────────                                                        │
│                                                                                   │
│   ids = []                                                                        │
│                                                                                   │
│   Word 1: "Zephyr"                                                                │
│     vocab.word2id("Zephyr") = 1 (UNK)                                           │
│     "Zephyr" in article_oovs? YES!                                              │
│     article_oovs.index("Zephyr") = 0                                            │
│     id = vocab_size + 0 = 50000                                                 │
│     ids = [50000]                                                                │
│                                                                                   │
│   Word 2: "jumped"                                                                │
│     vocab.word2id("jumped") = 600 (in vocab)                                    │
│     ids = [50000, 600]                                                           │
│                                                                                   │
│   Word 3: "and"                                                                   │
│     vocab.word2id("and") = 10 (in vocab)                                        │
│     ids = [50000, 600, 10]                                                       │
│                                                                                   │
│   Word 4: "Xander"                                                                │
│     vocab.word2id("Xander") = 1 (UNK)                                           │
│     "Xander" in article_oovs? YES!                                              │
│     article_oovs.index("Xander") = 1                                            │
│     id = vocab_size + 1 = 50001                                                 │
│     ids = [50000, 600, 10, 50001]                                               │
│                                                                                   │
│   Word 5: "watched"                                                               │
│     vocab.word2id("watched") = 700 (in vocab)                                   │
│     ids = [50000, 600, 10, 50001, 700]                                          │
│                                                                                   │
│   Word 6: "Yolanda"                                                               │
│     vocab.word2id("Yolanda") = 1 (UNK)                                          │
│     "Yolanda" in article_oovs? NO!                                              │
│     Can't copy - use UNK                                                         │
│     ids = [50000, 600, 10, 50001, 700, 1]                                       │
│                        ↑                  ↑                                       │
│               Yolanda → UNK (can't copy, not in article)                        │
│                                                                                   │
│                                                                                   │
│   OUTPUT:                                                                         │
│   ───────                                                                         │
│   ids = [50000, 600, 10, 50001, 700, 1]                                         │
│                                                                                   │
│   IMPORTANT OBSERVATION:                                                          │
│   ──────────────────────                                                          │
│                                                                                   │
│   • "Zephyr" and "Xander" get extended vocab IDs (can be copied)               │
│   • "Yolanda" gets UNK (cannot be copied - not in source article)              │
│                                                                                   │
│   This makes sense! The model can only copy words that exist in the input.      │
│   If the target contains a word not in input, model must use UNK.               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output to Words Conversion

### outputids2words Function (data.py)

```python
# data.py: outputids2words function (Lines 172-210)

def outputids2words(id_list, vocab, article_oovs):
    """
    Convert output IDs back to words.
    
    Args:
        id_list: List of word IDs (may include extended vocab IDs)
        vocab: Vocab object
        article_oovs: OOV list from article
    
    Returns:
        words: List of word strings
    """
    words = []
    
    for i in id_list:
        try:
            w = vocab.id2word(i)  # Try regular vocab
        except ValueError:
            # Not in vocab - must be extended vocab (OOV)
            assert article_oovs is not None, \
                "Error: model produced OOV but no article_oovs provided"
            
            # Calculate index into article_oovs
            article_oov_idx = i - vocab.size()
            
            if article_oov_idx < len(article_oovs):
                w = article_oovs[article_oov_idx]
            else:
                # This shouldn't happen in normal operation
                w = '[UNK]'
        
        words.append(w)
    
    return words
```

### Visual Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   OUTPUT TO WORDS CONVERSION                                      │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   INPUT:                                                                          │
│   ──────                                                                          │
│   id_list = [50000, 600, 10, 50001, 700]                                        │
│   article_oovs = ["Zephyr", "Xander"]                                           │
│   vocab_size = 50000                                                             │
│                                                                                   │
│                                                                                   │
│   STEP-BY-STEP EXECUTION:                                                         │
│   ────────────────────────                                                        │
│                                                                                   │
│   words = []                                                                      │
│                                                                                   │
│   ID 1: 50000                                                                     │
│     vocab.id2word(50000) → ValueError (not in vocab!)                           │
│     article_oov_idx = 50000 - 50000 = 0                                         │
│     w = article_oovs[0] = "Zephyr"                                              │
│     words = ["Zephyr"]                                                           │
│                                                                                   │
│   ID 2: 600                                                                       │
│     vocab.id2word(600) = "jumped"                                               │
│     words = ["Zephyr", "jumped"]                                                 │
│                                                                                   │
│   ID 3: 10                                                                        │
│     vocab.id2word(10) = "and"                                                   │
│     words = ["Zephyr", "jumped", "and"]                                         │
│                                                                                   │
│   ID 4: 50001                                                                     │
│     vocab.id2word(50001) → ValueError (not in vocab!)                           │
│     article_oov_idx = 50001 - 50000 = 1                                         │
│     w = article_oovs[1] = "Xander"                                              │
│     words = ["Zephyr", "jumped", "and", "Xander"]                               │
│                                                                                   │
│   ID 5: 700                                                                       │
│     vocab.id2word(700) = "watched"                                              │
│     words = ["Zephyr", "jumped", "and", "Xander", "watched"]                    │
│                                                                                   │
│                                                                                   │
│   OUTPUT:                                                                         │
│   ───────                                                                         │
│   words = ["Zephyr", "jumped", "and", "Xander", "watched"]                      │
│                                                                                   │
│                                                                                   │
│   THE MAGIC:                                                                      │
│   ──────────                                                                      │
│                                                                                   │
│   IDs 50000 and 50001 were OOV words during encoding.                           │
│   During decoding, we use article_oovs to recover the original words!           │
│                                                                                   │
│   This is how the pointer-generator can output words not in vocabulary!         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Extended Vocabulary in Practice

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                  EXTENDED VOCABULARY IN PRACTICE                                  │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   DURING TRAINING:                                                                │
│   ─────────────────                                                               │
│                                                                                   │
│   1. For each batch, find max_art_oovs                                           │
│      (Maximum number of OOV words in any article in the batch)                   │
│                                                                                   │
│   2. Extended vocab size = vocab_size + max_art_oovs                             │
│      Example: 50000 + 5 = 50005                                                  │
│                                                                                   │
│   3. Model's output distribution has shape [batch, 50005]                        │
│      • First 50000 entries: Regular vocabulary probabilities                     │
│      • Last 5 entries: Article-specific OOV probabilities                        │
│                                                                                   │
│                                                                                   │
│   VOCABULARY EXTENSION DIAGRAM:                                                   │
│   ─────────────────────────────                                                   │
│                                                                                   │
│   Regular Vocabulary          Extended Part (per-batch)                          │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │ ID:     0    1     2     ...  49999  │  50000   50001   50002   50003  │   │
│   │        ───  ───   ───        ─────   │  ─────   ─────   ─────   ─────  │   │
│   │       PAD  UNK  START  ...   (last   │  OOV_0   OOV_1   OOV_2   OOV_3  │   │
│   │                             vocab    │                                  │   │
│   │                             word)    │  (Article-specific OOV words)   │   │
│   └─────────────────────────────────────┴──────────────────────────────────┘   │
│   │<────────── Fixed (50000) ─────────>│<──── Variable (max_art_oovs) ────>│   │
│                                                                                   │
│                                                                                   │
│   DURING INFERENCE (Beam Search):                                                 │
│   ───────────────────────────────                                                 │
│                                                                                   │
│   1. Single article at a time, so OOV list is fixed                             │
│                                                                                   │
│   2. Model outputs extended vocab probabilities                                  │
│                                                                                   │
│   3. When decoding:                                                               │
│      • If ID < vocab_size: Look up word in vocabulary                           │
│      • If ID ≥ vocab_size: Look up word in article_oovs                         │
│                                                                                   │
│   4. Final output is the actual words (including copied OOVs)                   │
│                                                                                   │
│                                                                                   │
│   IMPORTANT CONSTRAINT:                                                           │
│   ─────────────────────                                                           │
│                                                                                   │
│   OOV IDs are PER-EXAMPLE, not global!                                           │
│                                                                                   │
│   Example 1: "Elon announced"  → article_oovs = ["Elon"]                        │
│              50000 means "Elon"                                                  │
│                                                                                   │
│   Example 2: "Tesla launched"  → article_oovs = ["Tesla"]                       │
│              50000 means "Tesla"                                                 │
│                                                                                   │
│   Same ID (50000), different words!                                              │
│   That's why we need article_oovs for each example.                             │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Worked Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE OOV HANDLING EXAMPLE                                 │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   SCENARIO:                                                                       │
│   ─────────                                                                       │
│                                                                                   │
│   Article: "Elon Musk announced Tesla's new Cybertruck model today."            │
│   Target:  "Musk unveiled Tesla's Cybertruck."                                   │
│                                                                                   │
│   vocab_size = 50000                                                             │
│   Vocab: {announced=100, new=200, model=300, today=400, unveiled=500, 's=600}   │
│   NOT in vocab: Elon, Musk, Tesla, Cybertruck                                   │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: ARTICLE ENCODING (batcher.py)                                          │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   article2ids called:                                                             │
│                                                                                   │
│   Word         In Vocab?    ID                                                   │
│   ────         ─────────    ──                                                   │
│   Elon         No           50000 (OOV #0)                                       │
│   Musk         No           50001 (OOV #1)                                       │
│   announced    Yes          100                                                   │
│   Tesla        No           50002 (OOV #2)                                       │
│   's           Yes          600                                                   │
│   new          Yes          200                                                   │
│   Cybertruck   No           50003 (OOV #3)                                       │
│   model        Yes          300                                                   │
│   today        Yes          400                                                   │
│   .            Yes          5                                                     │
│                                                                                   │
│   enc_input (for embedding):                                                      │
│   [1, 1, 100, 1, 600, 200, 1, 300, 400, 5]                                      │
│    ↑  ↑       ↑           ↑                                                      │
│   All OOVs → [UNK]                                                               │
│                                                                                   │
│   enc_input_extend_vocab (for copying):                                          │
│   [50000, 50001, 100, 50002, 600, 200, 50003, 300, 400, 5]                      │
│    ↑      ↑           ↑                ↑                                         │
│   Elon   Musk        Tesla            Cybertruck (unique IDs)                   │
│                                                                                   │
│   article_oovs: ["Elon", "Musk", "Tesla", "Cybertruck"]                         │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: TARGET ENCODING (batcher.py)                                           │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   abstract2ids called with article_oovs:                                         │
│                                                                                   │
│   Word         In Vocab?    In article_oovs?    ID                              │
│   ────         ─────────    ────────────────    ──                              │
│   Musk         No           Yes (index 1)       50001                           │
│   unveiled     Yes          N/A                 500                              │
│   Tesla        No           Yes (index 2)       50002                           │
│   's           Yes          N/A                 600                              │
│   Cybertruck   No           Yes (index 3)       50003                           │
│   .            Yes          N/A                 5                                 │
│                                                                                   │
│   target (with STOP):                                                            │
│   [50001, 500, 50002, 600, 50003, 5, 3]                                         │
│    ↑           ↑           ↑       ↑                                             │
│   Musk        Tesla      Cybertruck STOP                                         │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: MODEL TRAINING                                                          │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Extended vocab size for this batch: 50000 + 4 = 50004                         │
│                                                                                   │
│   Model output distribution shape: [batch_size, 50004]                          │
│                                                                                   │
│   For target word "Musk" (ID 50001):                                            │
│   • Loss encourages high probability at index 50001                             │
│   • This means: high p_copy AND high attention on position 1 (Musk)            │
│                                                                                   │
│   For target word "unveiled" (ID 500):                                          │
│   • Loss encourages high probability at index 500                               │
│   • This means: high p_gen AND high probability from vocabulary                 │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 4: MODEL INFERENCE (Beam Search)                                          │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Model generates IDs:                                                            │
│   output_ids = [50001, 500, 50002, 600, 50003, 5]                               │
│                                                                                   │
│   outputids2words conversion:                                                     │
│                                                                                   │
│   ID         ID < vocab_size?    Result                                         │
│   ──         ────────────────    ──────                                         │
│   50001      No                  article_oovs[1] = "Musk"                        │
│   500        Yes                 vocab.id2word(500) = "unveiled"                │
│   50002      No                  article_oovs[2] = "Tesla"                       │
│   600        Yes                 vocab.id2word(600) = "'s"                       │
│   50003      No                  article_oovs[3] = "Cybertruck"                  │
│   5          Yes                 vocab.id2word(5) = "."                          │
│                                                                                   │
│   Final output: "Musk unveiled Tesla's Cybertruck."                             │
│                  ↑              ↑        ↑                                       │
│                 COPIED from source article!                                      │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

Key concepts in vocabulary and OOV handling:

1. **Fixed vocabulary** maps common words to integer IDs
2. **Special tokens** (PAD, UNK, START, STOP) serve specific purposes
3. **Dual encoding** allows embedding lookup AND copying
4. **Extended vocabulary** gives unique IDs to each OOV word
5. **article_oovs** list enables decoding of copied OOV words
6. **Per-example OOV IDs** are temporary and batch-specific

The pointer-generator's OOV handling allows it to copy rare and unseen words from the source, overcoming a fundamental limitation of traditional seq2seq models.

---

*Next: [09_beam_search.md](09_beam_search.md) - Beam Search Decoding*
