# Data Pipeline Deep Dive

## Table of Contents
1. [Pipeline Overview](#pipeline-overview)
2. [Data Format](#data-format)
3. [Vocabulary Loading](#vocabulary-loading)
4. [Example Class](#example-class)
5. [Batch Class](#batch-class)
6. [Batcher Class](#batcher-class)
7. [Multi-threading Architecture](#multi-threading-architecture)
8. [Padding and Masking](#padding-and-masking)
9. [Complete Worked Example](#complete-worked-example)

---

## Pipeline Overview

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE OVERVIEW                                     │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Raw Data                                                                        │
│   ────────                                                                        │
│       │                                                                           │
│       │  TFRecord Files (.bin)                                                    │
│       │  containing article-abstract pairs                                       │
│       ▼                                                                           │
│   ┌─────────────────────┐                                                         │
│   │  example_generator  │  Reads raw data from files                             │
│   │  (data.py)          │  Yields raw text strings                               │
│   └──────────┬──────────┘                                                         │
│              │                                                                    │
│              │ article_text, abstract_text                                       │
│              ▼                                                                    │
│   ┌─────────────────────┐                                                         │
│   │   Example Class     │  Converts text to integer IDs                          │
│   │   (batcher.py)      │  Handles OOV words                                     │
│   │                     │  Truncates/pads sequences                              │
│   └──────────┬──────────┘                                                         │
│              │                                                                    │
│              │ Example object with tokenized data                                │
│              ▼                                                                    │
│   ┌─────────────────────┐                                                         │
│   │    Example Queue    │  Thread-safe queue                                     │
│   │   (multi-threaded)  │  Pre-fills examples                                    │
│   └──────────┬──────────┘                                                         │
│              │                                                                    │
│              │ Multiple Examples                                                  │
│              ▼                                                                    │
│   ┌─────────────────────┐                                                         │
│   │    Batch Class      │  Combines Examples into batches                        │
│   │    (batcher.py)     │  Creates numpy arrays                                  │
│   │                     │  Adds padding masks                                    │
│   └──────────┬──────────┘                                                         │
│              │                                                                    │
│              │ Batch object (numpy arrays)                                       │
│              ▼                                                                    │
│   ┌─────────────────────┐                                                         │
│   │    Batch Queue      │  Pre-filled batch queue                                │
│   │   (multi-threaded)  │  Ready for training                                    │
│   └──────────┬──────────┘                                                         │
│              │                                                                    │
│              │ batch.next_batch()                                                │
│              ▼                                                                    │
│   ┌─────────────────────┐                                                         │
│   │   Model Training    │  Receives batched data                                 │
│   │   (model.py)        │  via feed_dict                                         │
│   └─────────────────────┘                                                         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Format

### TFRecord Structure

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           DATA FORMAT                                             │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   FILE FORMAT: TFRecord (TensorFlow's binary format)                              │
│   ─────────────────────────────────────────────────────                           │
│                                                                                   │
│   Each record contains:                                                           │
│   • article: The source text (input)                                              │
│   • abstract: The summary (target)                                                │
│                                                                                   │
│                                                                                   │
│   EXAMPLE RECORD:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   {                                                                               │
│       "article": "germany beat argentina in the world cup final . "              │
│                  "mario götze scored the winning goal in extra time .",          │
│       "abstract": "&lt;s&gt; germany won the world cup . &lt;/s&gt; "            │
│                   "&lt;s&gt; götze scored the winner . &lt;/s&gt;"               │
│   }                                                                               │
│                                                                                   │
│                                                                                   │
│   ABSTRACT FORMAT:                                                                │
│   ─────────────────                                                               │
│                                                                                   │
│   Abstracts have special sentence markers:                                        │
│   • &lt;s&gt;   : Sentence start marker (becomes [START] token)                  │
│   • &lt;/s&gt;  : Sentence end marker (becomes [STOP] token)                     │
│                                                                                   │
│   These markers help the model learn sentence boundaries.                         │
│                                                                                   │
│                                                                                   │
│   DATA DIRECTORY STRUCTURE:                                                       │
│   ─────────────────────────                                                       │
│                                                                                   │
│   /path/to/data/                                                                  │
│   ├── train_*.bin       # Training data files                                    │
│   ├── val_*.bin         # Validation data files                                  │
│   ├── test_*.bin        # Test data files                                        │
│   └── vocab             # Vocabulary file                                        │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Reading Data (data.py)

```python
# data.py: example_generator function (Lines 51-91)

def example_generator(data_path, single_pass):
    """
    Generates tf.Examples from data files.
    
    Args:
        data_path: Path pattern to data files (e.g., "data/train_*")
        single_pass: If True, go through data once (for eval)
                     If False, loop forever (for training)
    
    Yields:
        tf.Example objects containing article and abstract
    """
    while True:
        # Get list of data files matching pattern
        filelist = glob.glob(data_path)
        assert filelist, "No data files found: %s" % data_path
        
        if single_pass:
            filelist = sorted(filelist)  # Deterministic order
        else:
            random.shuffle(filelist)  # Random order for training
        
        for f in filelist:
            # Read TFRecord file
            reader = open(f, 'rb')
            while True:
                # Read length-prefixed record
                len_bytes = reader.read(8)
                if not len_bytes:
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = reader.read(str_len)
                
                # Parse as tf.Example
                yield tf.train.Example.FromString(example_str)
            
            reader.close()
        
        if single_pass:
            break  # Exit after one pass
```

---

## Vocabulary Loading

### Vocab Class (data.py)

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                         VOCABULARY CLASS                                          │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   VOCABULARY FILE FORMAT:                                                         │
│   ────────────────────────                                                        │
│                                                                                   │
│   One word per line, with word count:                                             │
│                                                                                   │
│   the 45678901                                                                    │
│   . 23456789                                                                      │
│   , 12345678                                                                      │
│   a 11234567                                                                      │
│   to 10234567                                                                     │
│   of 9876543                                                                      │
│   and 8765432                                                                     │
│   ...                                                                             │
│   (up to max_size words)                                                         │
│                                                                                   │
│                                                                                   │
│   SPECIAL TOKENS:                                                                 │
│   ───────────────                                                                 │
│                                                                                   │
│   Token           ID      Purpose                                                │
│   ─────────       ──      ───────                                                │
│   [PAD]           0       Padding for sequence alignment                         │
│   [UNK]           1       Unknown word (OOV replacement)                         │
│   [START]         2       Start-of-sequence (decoder input)                      │
│   [STOP]          3       End-of-sequence (stop generation)                      │
│                                                                                   │
│                                                                                   │
│   ID ASSIGNMENT:                                                                  │
│   ──────────────                                                                  │
│                                                                                   │
│   Word            ID                                                              │
│   ────            ──                                                              │
│   [PAD]           0                                                               │
│   [UNK]           1                                                               │
│   [START]         2                                                               │
│   [STOP]          3                                                               │
│   the             4                                                               │
│   .               5                                                               │
│   ,               6                                                               │
│   a               7                                                               │
│   ...             ...                                                             │
│   (word 49996)    49999   ← vocab_size = 50000                                   │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Vocab Class Implementation

```python
# data.py: Vocab class (Lines 12-49)

class Vocab(object):
    """Vocabulary class for mapping words to IDs."""
    
    # Special tokens
    UNKNOWN_TOKEN = '[UNK]'
    PAD_TOKEN = '[PAD]'
    START_DECODING = '[START]'
    STOP_DECODING = '[STOP]'
    
    def __init__(self, vocab_file, max_size):
        """
        Creates a vocab of up to max_size words.
        
        Args:
            vocab_file: Path to vocabulary file
            max_size: Maximum vocabulary size
        """
        # Initialize word <-> ID mappings
        self._word_to_id = {}
        self._id_to_word = {}
        
        # Add special tokens (IDs 0, 1, 2, 3)
        self._word_to_id[self.PAD_TOKEN] = 0
        self._word_to_id[self.UNKNOWN_TOKEN] = 1
        self._word_to_id[self.START_DECODING] = 2
        self._word_to_id[self.STOP_DECODING] = 3
        
        for w in [self.PAD_TOKEN, self.UNKNOWN_TOKEN, 
                  self.START_DECODING, self.STOP_DECODING]:
            self._id_to_word[self._word_to_id[w]] = w
        
        self._count = 4  # Next available ID
        
        # Read vocab file
        with open(vocab_file, 'r') as vocab_f:
            for line in vocab_f:
                pieces = line.split()
                if len(pieces) != 2:
                    continue
                    
                w = pieces[0]
                if w in [self.UNKNOWN_TOKEN, self.PAD_TOKEN,
                         self.START_DECODING, self.STOP_DECODING]:
                    continue  # Skip if special token
                
                self._word_to_id[w] = self._count
                self._id_to_word[self._count] = w
                self._count += 1
                
                if self._count >= max_size:
                    break  # Reached max vocab size
        
        print("Finished loading vocab of %i words." % self._count)
    
    def word2id(self, word):
        """Returns the id of a word, or UNK if not in vocab."""
        if word not in self._word_to_id:
            return self._word_to_id[self.UNKNOWN_TOKEN]
        return self._word_to_id[word]
    
    def id2word(self, word_id):
        """Returns the word corresponding to an id."""
        if word_id not in self._id_to_word:
            raise ValueError('Id not found in vocab: %d' % word_id)
        return self._id_to_word[word_id]
    
    def size(self):
        """Returns the vocabulary size."""
        return self._count
```

---

## Example Class

The Example class processes a single article-abstract pair:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           EXAMPLE CLASS                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE: Process a single article-abstract pair into model-ready format        │
│                                                                                   │
│   INPUT:                                                                          │
│   ──────                                                                          │
│   • article: "germany beat argentina in the final . götze scored ."             │
│   • abstract: "<s> germany won . </s> <s> götze scored . </s>"                  │
│   • vocab: Vocabulary object                                                      │
│   • hps: Hyperparameters                                                          │
│                                                                                   │
│                                                                                   │
│   PROCESSING STEPS:                                                               │
│   ─────────────────                                                               │
│                                                                                   │
│   1. TOKENIZE                                                                     │
│   ───────────                                                                     │
│                                                                                   │
│   article_words = ["germany", "beat", "argentina", "in", "the",                 │
│                    "final", ".", "götze", "scored", "."]                         │
│                                                                                   │
│   abstract_words = ["<s>", "germany", "won", ".", "</s>",                       │
│                     "<s>", "götze", "scored", ".", "</s>"]                       │
│                                                                                   │
│   2. TRUNCATE IF NEEDED                                                           │
│   ─────────────────────                                                           │
│                                                                                   │
│   if len(article_words) > max_enc_steps:                                         │
│       article_words = article_words[:max_enc_steps]  # Keep first 400           │
│                                                                                   │
│   3. CONVERT TO IDS (with OOV handling)                                          │
│   ──────────────────────────────────────                                          │
│                                                                                   │
│   TWO ENCODINGS ARE CREATED:                                                      │
│                                                                                   │
│   a) enc_input: Standard encoding (OOV → [UNK])                                  │
│      For embedding lookup (only vocab words can be embedded)                     │
│                                                                                   │
│      "germany" → 100                                                              │
│      "beat"    → 234                                                              │
│      "götze"   → 1 (UNK - not in vocab!)                                         │
│                                                                                   │
│      enc_input = [100, 234, 567, 89, 4, 234, 5, 1, 345, 5]                       │
│                                            ↑                                      │
│                                        UNK for "götze"                           │
│                                                                                   │
│   b) enc_input_extend_vocab: Extended encoding (OOV → temporary ID)             │
│      For pointer mechanism (to copy OOV words)                                   │
│                                                                                   │
│      "germany" → 100                                                              │
│      "beat"    → 234                                                              │
│      "götze"   → 50000 (vocab_size + OOV_index)                                  │
│                                                                                   │
│      enc_input_extend_vocab = [100, 234, 567, 89, 4, 234, 5, 50000, 345, 5]      │
│                                                         ↑                         │
│                                              Temporary ID for "götze"            │
│                                                                                   │
│   4. BUILD OOV LIST                                                               │
│   ─────────────────                                                               │
│                                                                                   │
│   article_oovs = ["götze"]  # List of OOV words in article                       │
│   (maps "götze" to temporary ID 50000)                                           │
│                                                                                   │
│   5. PROCESS ABSTRACT (decoder side)                                             │
│   ───────────────────────────────────                                             │
│                                                                                   │
│   a) dec_input: Decoder input (with START token)                                 │
│      [START, word1, word2, ..., word_n]                                          │
│                                                                                   │
│   b) target: Target output (with STOP token)                                     │
│      [word1, word2, ..., word_n, STOP]                                           │
│                                                                                   │
│   Example:                                                                        │
│   abstract_words = ["<s>", "germany", "won", "."]                               │
│   dec_input = [START_ID, germany_id, won_id, period_id]                         │
│   target    = [germany_id, won_id, period_id, STOP_ID]                          │
│                                                                                   │
│                                                                                   │
│   OUTPUT:                                                                         │
│   ───────                                                                         │
│                                                                                   │
│   Example object with:                                                            │
│   • enc_input           [batch, enc_len]: Article IDs (UNK for OOV)             │
│   • enc_input_extend    [batch, enc_len]: Article IDs (extended vocab)          │
│   • enc_len             Integer: Article length                                  │
│   • dec_input           [batch, dec_len]: Decoder input IDs                     │
│   • dec_target          [batch, dec_len]: Target IDs (extended vocab)           │
│   • dec_len             Integer: Abstract length                                 │
│   • article_oovs        List: OOV words from article                            │
│   • original_article    String: Original article text                           │
│   • original_abstract   String: Original abstract text                          │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Example Class Implementation

```python
# batcher.py: Example class (Lines 35-120)

class Example(object):
    """
    Class representing a single example (article-abstract pair).
    """
    
    def __init__(self, article, abstract, vocab, hps):
        """
        Initializes the Example.
        
        Args:
            article: Source article text (string)
            abstract: Target abstract text (string)
            vocab: Vocab object
            hps: Hyperparameters
        """
        # Store original text
        self.original_article = article
        self.original_abstract = abstract
        
        # --- PROCESS ARTICLE (ENCODER INPUT) ---
        
        # Tokenize
        article_words = article.split()
        
        # Truncate if too long
        if len(article_words) > hps.max_enc_steps:
            article_words = article_words[:hps.max_enc_steps]
        
        self.enc_len = len(article_words)
        
        # Convert to IDs (OOV → UNK)
        self.enc_input = [vocab.word2id(w) for w in article_words]
        
        # For pointer-generator: extended vocabulary encoding
        if hps.pointer_gen:
            # Build OOV list and extended vocab encoding
            self.enc_input_extend_vocab, self.article_oovs = \
                article2ids(article_words, vocab)
        
        # --- PROCESS ABSTRACT (DECODER INPUT/TARGET) ---
        
        # Get start and stop IDs
        start_decoding = vocab.word2id(vocab.START_DECODING)
        stop_decoding = vocab.word2id(vocab.STOP_DECODING)
        
        # Tokenize abstract (sentence by sentence)
        abstract_sentences = abstract.split(' ')
        abstract_words = []
        for sent in abstract_sentences:
            if sent.startswith('<s>'):
                sent = sent[3:]  # Remove <s>
            if sent.endswith('</s>'):
                sent = sent[:-4]  # Remove </s>
            abstract_words.extend(sent.split())
        
        # Truncate if too long
        if len(abstract_words) > hps.max_dec_steps:
            abstract_words = abstract_words[:hps.max_dec_steps]
        
        # Convert to IDs
        abs_ids = [vocab.word2id(w) for w in abstract_words]
        
        # Create decoder input: [START, word1, word2, ...]
        self.dec_input = [start_decoding] + abs_ids
        
        # Create target: [word1, word2, ..., STOP]
        self.target = abs_ids + [stop_decoding]
        
        # For pointer-generator: target with extended vocab
        if hps.pointer_gen:
            abs_ids_extend = abstract2ids(
                abstract_words, vocab, self.article_oovs
            )
            self.target = abs_ids_extend + [stop_decoding]
        
        self.dec_len = len(self.dec_input)
    
    def pad_encoder_input(self, max_len, pad_id):
        """Pad encoder input to max_len."""
        while len(self.enc_input) < max_len:
            self.enc_input.append(pad_id)
        if hasattr(self, 'enc_input_extend_vocab'):
            while len(self.enc_input_extend_vocab) < max_len:
                self.enc_input_extend_vocab.append(pad_id)
    
    def pad_decoder_input(self, max_len, pad_id):
        """Pad decoder input and target to max_len."""
        while len(self.dec_input) < max_len:
            self.dec_input.append(pad_id)
        while len(self.target) < max_len:
            self.target.append(pad_id)
```

---

## Batch Class

The Batch class combines multiple Examples into batched tensors:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                            BATCH CLASS                                            │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE: Combine multiple Example objects into batched numpy arrays            │
│                                                                                   │
│   INPUT: List of Example objects [ex1, ex2, ..., ex_batch_size]                  │
│                                                                                   │
│                                                                                   │
│   BATCH CONSTRUCTION:                                                             │
│   ───────────────────                                                             │
│                                                                                   │
│   Given batch_size=4 examples:                                                    │
│                                                                                   │
│   Example 1: enc_len=8,  dec_len=5                                               │
│   Example 2: enc_len=10, dec_len=7                                               │
│   Example 3: enc_len=6,  dec_len=4                                               │
│   Example 4: enc_len=9,  dec_len=6                                               │
│                                                                                   │
│   Step 1: Find max lengths in batch                                              │
│   ─────────────────────────────────                                               │
│   max_enc_len = max(8, 10, 6, 9) = 10                                           │
│   max_dec_len = max(5, 7, 4, 6) = 7                                             │
│                                                                                   │
│   Step 2: Pad all examples to max length                                         │
│   ──────────────────────────────────────                                          │
│                                                                                   │
│   Example 1: enc_input padded from 8 → 10 (add 2 PAD)                           │
│   Example 2: enc_input already 10 (no padding needed)                           │
│   Example 3: enc_input padded from 6 → 10 (add 4 PAD)                           │
│   Example 4: enc_input padded from 9 → 10 (add 1 PAD)                           │
│                                                                                   │
│   Step 3: Stack into batch arrays                                                │
│   ───────────────────────────────                                                 │
│                                                                                   │
│   enc_batch shape: [batch_size, max_enc_len] = [4, 10]                          │
│   dec_batch shape: [batch_size, max_dec_len] = [4, 7]                           │
│                                                                                   │
│                                                                                   │
│   BATCH ATTRIBUTES:                                                               │
│   ─────────────────                                                               │
│                                                                                   │
│   Attribute                  Shape                   Description                  │
│   ─────────                  ─────                   ───────────                  │
│   enc_batch                  [B, enc_len]           Encoder input IDs            │
│   enc_lens                   [B]                    Actual encoder lengths       │
│   enc_padding_mask           [B, enc_len]           1=real, 0=padding            │
│   enc_batch_extend_vocab     [B, enc_len]           Extended vocab IDs           │
│   max_art_oovs               scalar                 Max OOVs in batch            │
│   art_oovs                   [B, ?]                 OOV words per example        │
│   dec_batch                  [B, dec_len]           Decoder input IDs            │
│   target_batch               [B, dec_len]           Target IDs                   │
│   dec_lens                   [B]                    Actual decoder lengths       │
│   dec_padding_mask           [B, dec_len]           1=real, 0=padding            │
│                                                                                   │
│   Where B = batch_size                                                            │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Batch Class Implementation

```python
# batcher.py: Batch class (Lines 122-220)

class Batch(object):
    """
    Class representing a batch of examples.
    """
    
    def __init__(self, example_list, hps, vocab):
        """
        Creates a Batch from a list of Examples.
        
        Args:
            example_list: List of Example objects
            hps: Hyperparameters
            vocab: Vocab object
        """
        self.pad_id = vocab.word2id(vocab.PAD_TOKEN)
        
        # --- INITIALIZE BATCH ---
        self._init_encoder_seq(example_list, hps)
        self._init_decoder_seq(example_list, hps)
        
        # Store original strings
        self.original_articles = [ex.original_article for ex in example_list]
        self.original_abstracts = [ex.original_abstract for ex in example_list]
    
    def _init_encoder_seq(self, example_list, hps):
        """Initialize encoder batch data."""
        
        # Find max encoder length in batch
        max_enc_seq_len = max([ex.enc_len for ex in example_list])
        
        # Pad examples to max length
        for ex in example_list:
            ex.pad_encoder_input(max_enc_seq_len, self.pad_id)
        
        # Stack into batch arrays
        self.enc_batch = np.array([ex.enc_input for ex in example_list])
        self.enc_lens = np.array([ex.enc_len for ex in example_list])
        
        # Create padding mask (1 for real tokens, 0 for padding)
        self.enc_padding_mask = np.zeros_like(self.enc_batch, dtype=np.float32)
        for i, ex in enumerate(example_list):
            self.enc_padding_mask[i, :ex.enc_len] = 1.0
        
        # For pointer-generator
        if hps.pointer_gen:
            # Extended vocabulary batch
            self.enc_batch_extend_vocab = np.array(
                [ex.enc_input_extend_vocab for ex in example_list]
            )
            
            # OOV handling
            self.art_oovs = [ex.article_oovs for ex in example_list]
            self.max_art_oovs = max([len(ex.article_oovs) for ex in example_list])
            
            # Ensure at least 1 to avoid issues
            if self.max_art_oovs == 0:
                self.max_art_oovs = 1
    
    def _init_decoder_seq(self, example_list, hps):
        """Initialize decoder batch data."""
        
        # Find max decoder length in batch
        max_dec_seq_len = max([ex.dec_len for ex in example_list])
        
        # Pad examples to max length
        for ex in example_list:
            ex.pad_decoder_input(max_dec_seq_len, self.pad_id)
        
        # Stack into batch arrays
        self.dec_batch = np.array([ex.dec_input for ex in example_list])
        self.target_batch = np.array([ex.target for ex in example_list])
        self.dec_lens = np.array([ex.dec_len for ex in example_list])
        
        # Create padding mask
        self.dec_padding_mask = np.zeros_like(self.dec_batch, dtype=np.float32)
        for i, ex in enumerate(example_list):
            self.dec_padding_mask[i, :ex.dec_len] = 1.0
```

---

## Batcher Class

The Batcher manages the data pipeline with multi-threading:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                           BATCHER CLASS                                           │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   PURPOSE: Manage data loading with pre-filled queues                             │
│                                                                                   │
│   ARCHITECTURE:                                                                    │
│   ─────────────                                                                   │
│                                                                                   │
│   ┌────────────────┐                                                              │
│   │   Data Files   │                                                              │
│   │  (*.bin)       │                                                              │
│   └───────┬────────┘                                                              │
│           │                                                                        │
│           │  example_generator()                                                   │
│           ▼                                                                        │
│   ┌────────────────────────────────────────────────────┐                         │
│   │              EXAMPLE QUEUE                          │                         │
│   │  ┌──────┬──────┬──────┬──────┬──────┬─────────┐   │                         │
│   │  │ Ex 1 │ Ex 2 │ Ex 3 │ Ex 4 │ Ex 5 │   ...   │   │                         │
│   │  └──────┴──────┴──────┴──────┴──────┴─────────┘   │                         │
│   │  (Queue.Queue with QUEUE_NUM_BATCH × batch_size)   │                         │
│   │                                                     │                         │
│   │  Filled by: 4 × fill_example_queue threads         │                         │
│   └────────────────────────┬───────────────────────────┘                         │
│                            │                                                       │
│                            │  batch_size examples                                  │
│                            ▼                                                       │
│   ┌────────────────────────────────────────────────────┐                         │
│   │               BATCH QUEUE                           │                         │
│   │  ┌─────────┬─────────┬─────────┬─────────────┐    │                         │
│   │  │ Batch 1 │ Batch 2 │ Batch 3 │    ...      │    │                         │
│   │  └─────────┴─────────┴─────────┴─────────────┘    │                         │
│   │  (Queue.Queue with QUEUE_NUM_BATCH batches)        │                         │
│   │                                                     │                         │
│   │  Filled by: 4 × fill_batch_queue threads           │                         │
│   └────────────────────────┬───────────────────────────┘                         │
│                            │                                                       │
│                            │  next_batch()                                         │
│                            ▼                                                       │
│   ┌────────────────────────────────────────────────────┐                         │
│   │              TRAINING LOOP                          │                         │
│   │                                                     │                         │
│   │  batch = batcher.next_batch()                      │                         │
│   │  model.run_train_step(batch)                       │                         │
│   │                                                     │                         │
│   └────────────────────────────────────────────────────┘                         │
│                                                                                   │
│                                                                                   │
│   CONSTANTS:                                                                       │
│   ──────────                                                                       │
│                                                                                   │
│   QUEUE_NUM_BATCH = 100  # Pre-fill 100 batches                                  │
│   NUM_EXAMPLE_Q_THREADS = 4  # Threads filling example queue                     │
│   NUM_BATCH_Q_THREADS = 4    # Threads filling batch queue                       │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

### Batcher Class Implementation

```python
# batcher.py: Batcher class (Lines 222-375)

QUEUE_NUM_BATCH = 100  # Number of batches to pre-fill

class Batcher(object):
    """
    A class to generate minibatches of data.
    """
    
    def __init__(self, data_path, vocab, hps, single_pass):
        """
        Initialize the batcher.
        
        Args:
            data_path: Path pattern to data files
            vocab: Vocab object
            hps: Hyperparameters
            single_pass: If True, process data once (for eval)
        """
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass
        
        # Create queues
        self._example_queue = Queue.Queue(
            maxsize=QUEUE_NUM_BATCH * hps.batch_size
        )
        self._batch_queue = Queue.Queue(
            maxsize=QUEUE_NUM_BATCH
        )
        
        # For single pass, use simpler single-threaded approach
        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
        else:
            self._num_example_q_threads = 4
            self._num_batch_q_threads = 4
        
        # Start threads to fill queues
        self._example_q_threads = []
        for _ in range(self._num_example_q_threads):
            t = Thread(target=self.fill_example_queue)
            t.daemon = True
            t.start()
            self._example_q_threads.append(t)
        
        self._batch_q_threads = []
        for _ in range(self._num_batch_q_threads):
            t = Thread(target=self.fill_batch_queue)
            t.daemon = True
            t.start()
            self._batch_q_threads.append(t)
    
    def fill_example_queue(self):
        """Thread function: reads data and fills example queue."""
        example_gen = example_generator(
            self._data_path, self._single_pass
        )
        
        while True:
            try:
                # Get next example from generator
                ex = next(example_gen)
            except StopIteration:
                break  # No more data
            
            # Parse article and abstract
            article = ex.features.feature['article'].bytes_list.value[0].decode()
            abstract = ex.features.feature['abstract'].bytes_list.value[0].decode()
            
            # Create Example object
            example = Example(article, abstract, self._vocab, self._hps)
            
            # Put in queue (blocks if queue is full)
            self._example_queue.put(example)
    
    def fill_batch_queue(self):
        """Thread function: creates batches from examples."""
        while True:
            # Collect batch_size examples
            examples = []
            for _ in range(self._hps.batch_size):
                try:
                    ex = self._example_queue.get(timeout=30)
                    examples.append(ex)
                except Queue.Empty:
                    if self._single_pass:
                        break
                    continue
            
            if len(examples) == 0:
                break  # No more examples
            
            # For decode mode, sort by encoder length for efficiency
            if self._hps.mode == 'decode':
                examples = sorted(examples, key=lambda x: x.enc_len, reverse=True)
            
            # Create Batch object
            batch = Batch(examples, self._hps, self._vocab)
            
            # Put in batch queue
            self._batch_queue.put(batch)
    
    def next_batch(self):
        """
        Returns the next batch of data.
        
        Returns:
            Batch object, or None if single_pass and exhausted
        """
        if self._batch_queue.empty():
            if self._single_pass:
                return None
        
        batch = self._batch_queue.get()
        return batch
```

---

## Multi-threading Architecture

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   MULTI-THREADING ARCHITECTURE                                    │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHY MULTI-THREADING?                                                            │
│   ────────────────────                                                            │
│                                                                                   │
│   1. Prevent GPU starvation                                                       │
│      • GPU processes batches faster than CPU prepares them                       │
│      • Pre-filling queues ensures batches are ready                              │
│                                                                                   │
│   2. Overlap I/O and computation                                                 │
│      • While GPU trains, CPU prepares next batches                               │
│      • No waiting for data loading                                               │
│                                                                                   │
│   3. Parallel data processing                                                     │
│      • Multiple threads tokenize/encode in parallel                              │
│      • Faster throughput for large datasets                                      │
│                                                                                   │
│                                                                                   │
│   THREAD ORGANIZATION:                                                            │
│   ────────────────────                                                            │
│                                                                                   │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      EXAMPLE QUEUE THREADS (4)                           │   │
│   │                                                                          │   │
│   │   Thread 1 ─────┐                                                        │   │
│   │                 │                                                        │   │
│   │   Thread 2 ─────┼─────▶  EXAMPLE QUEUE  ────────────┐                   │   │
│   │                 │      (1600 examples)               │                   │   │
│   │   Thread 3 ─────┤                                    │                   │   │
│   │                 │                                    │                   │   │
│   │   Thread 4 ─────┘                                    │                   │   │
│   │                                                      │                   │   │
│   │   Each thread:                                       │                   │   │
│   │   • Reads from different data files                  │                   │   │
│   │   • Creates Example objects                          │                   │   │
│   │   • Puts into shared queue                           │                   │   │
│   │                                                      │                   │   │
│   └──────────────────────────────────────────────────────┼───────────────────┘   │
│                                                          │                       │
│                                                          ▼                       │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                      BATCH QUEUE THREADS (4)                             │   │
│   │                                                                          │   │
│   │   Thread 1 ─────┐                                                        │   │
│   │                 │                                                        │   │
│   │   Thread 2 ─────┼─────▶  BATCH QUEUE  ──────────────────────────────▶   │   │
│   │                 │      (100 batches)     Training Loop                   │   │
│   │   Thread 3 ─────┤                                                        │   │
│   │                 │                                                        │   │
│   │   Thread 4 ─────┘                                                        │   │
│   │                                                                          │   │
│   │   Each thread:                                                           │   │
│   │   • Takes batch_size examples from example queue                        │   │
│   │   • Creates Batch object (padding, stacking)                            │   │
│   │   • Puts into batch queue                                               │   │
│   │                                                                          │   │
│   └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                   │
│                                                                                   │
│   QUEUE SIZES:                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Example Queue Size = QUEUE_NUM_BATCH × batch_size                              │
│                      = 100 × 16 = 1,600 examples                                 │
│                                                                                   │
│   Batch Queue Size = QUEUE_NUM_BATCH = 100 batches                               │
│                                                                                   │
│   This ensures ~100 batches worth of data is always ready!                       │
│                                                                                   │
│                                                                                   │
│   THREAD SAFETY:                                                                  │
│   ──────────────                                                                  │
│                                                                                   │
│   • Queue.Queue is thread-safe                                                   │
│   • Blocking operations (put/get) prevent race conditions                        │
│   • Daemon threads exit when main program exits                                  │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Padding and Masking

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                        PADDING AND MASKING                                        │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   WHY PADDING?                                                                    │
│   ────────────                                                                    │
│                                                                                   │
│   Different examples have different lengths, but tensors need fixed shapes:      │
│                                                                                   │
│   Example 1: "Germany won the cup"         → 4 words                            │
│   Example 2: "Argentina lost in final"     → 4 words                            │
│   Example 3: "It was a great match today"  → 6 words                            │
│   Example 4: "Goals scored"                → 2 words                            │
│                                                                                   │
│   To batch together, pad to max length (6):                                      │
│                                                                                   │
│   Example 1: [Germany, won, the, cup, PAD, PAD]                                 │
│   Example 2: [Argentina, lost, in, final, PAD, PAD]                             │
│   Example 3: [It, was, a, great, match, today]                                  │
│   Example 4: [Goals, scored, PAD, PAD, PAD, PAD]                                │
│                                                                                   │
│   Now all have shape [6], can stack to [4, 6] batch tensor.                     │
│                                                                                   │
│                                                                                   │
│   PADDING MASK:                                                                   │
│   ─────────────                                                                   │
│                                                                                   │
│   Padding mask indicates which positions are real tokens (1) vs padding (0):    │
│                                                                                   │
│   Example 1: [1, 1, 1, 1, 0, 0]   ← 4 real tokens                               │
│   Example 2: [1, 1, 1, 1, 0, 0]   ← 4 real tokens                               │
│   Example 3: [1, 1, 1, 1, 1, 1]   ← 6 real tokens                               │
│   Example 4: [1, 1, 0, 0, 0, 0]   ← 2 real tokens                               │
│                                                                                   │
│   Uses of padding mask:                                                           │
│                                                                                   │
│   1. Attention masking: Don't attend to padding positions                        │
│      attention = softmax(scores × enc_padding_mask)                              │
│                                                                                   │
│   2. Loss masking: Don't count loss on padding positions                         │
│      loss = sum(per_token_loss × dec_padding_mask) / sum(dec_padding_mask)      │
│                                                                                   │
│                                                                                   │
│   ENCODER PADDING MASK USAGE:                                                     │
│   ───────────────────────────                                                     │
│                                                                                   │
│   In attention calculation (model.py):                                            │
│                                                                                   │
│   # Multiply attention scores by mask                                            │
│   attn_dist = softmax(attn_scores)                                               │
│   attn_dist *= enc_padding_mask  # Zero out attention to padding                │
│   attn_dist /= sum(attn_dist)    # Re-normalize                                 │
│                                                                                   │
│                                                                                   │
│   DECODER PADDING MASK USAGE:                                                     │
│   ───────────────────────────                                                     │
│                                                                                   │
│   In loss calculation (model.py):                                                 │
│                                                                                   │
│   # Mask and average losses                                                       │
│   per_step_loss = cross_entropy(logits, targets)  # [batch, dec_len]            │
│   masked_loss = per_step_loss * dec_padding_mask  # Zero out padding            │
│   avg_loss = sum(masked_loss) / sum(dec_padding_mask)  # Proper average         │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Worked Example

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                     COMPLETE WORKED EXAMPLE                                       │
├──────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   RAW DATA:                                                                       │
│   ─────────                                                                       │
│                                                                                   │
│   TFRecord entry:                                                                 │
│   {                                                                               │
│       "article": "germany beat argentina in the world cup final . "              │
│                  "mario götze scored the winning goal .",                        │
│       "abstract": "<s> germany won the world cup . </s>"                        │
│   }                                                                               │
│                                                                                   │
│   VOCABULARY (sample):                                                            │
│   ────────────────────                                                            │
│                                                                                   │
│   [PAD]=0, [UNK]=1, [START]=2, [STOP]=3, the=4, .=5, in=6, a=7,                 │
│   germany=100, beat=101, argentina=102, world=103, cup=104,                      │
│   final=105, won=106, goal=107, scored=108                                       │
│   (götze and mario NOT in vocab - they're OOV)                                  │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 1: EXAMPLE CREATION                                                        │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Article tokenization:                                                           │
│   article_words = ["germany", "beat", "argentina", "in", "the", "world",        │
│                    "cup", "final", ".", "mario", "götze", "scored",             │
│                    "the", "winning", "goal", "."]                               │
│                                                                                   │
│   Encoder input (OOV → UNK):                                                     │
│   enc_input = [100, 101, 102, 6, 4, 103, 104, 105, 5, 1, 1, 108, 4, 1, 107, 5]  │
│                                                        ↑  ↑        ↑            │
│                                              mario→UNK götze→UNK winning→UNK    │
│                                                                                   │
│   Extended vocab encoding (for pointer):                                         │
│   article_oovs = ["mario", "götze", "winning"]                                  │
│   OOV IDs: mario=50000, götze=50001, winning=50002                              │
│                                                                                   │
│   enc_input_extend_vocab = [100, 101, 102, 6, 4, 103, 104, 105, 5,              │
│                             50000, 50001, 108, 4, 50002, 107, 5]                 │
│                              ↑      ↑               ↑                            │
│                         Extended OOV IDs!                                        │
│                                                                                   │
│   Abstract tokenization:                                                          │
│   abstract_words = ["germany", "won", "the", "world", "cup", "."]               │
│                                                                                   │
│   Decoder input (with START):                                                     │
│   dec_input = [2, 100, 106, 4, 103, 104, 5]                                     │
│                ↑                                                                  │
│              START                                                               │
│                                                                                   │
│   Target (with STOP):                                                            │
│   target = [100, 106, 4, 103, 104, 5, 3]                                        │
│                                       ↑                                          │
│                                     STOP                                         │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 2: BATCH CREATION (with 4 examples)                                        │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   Example 1: enc_len=16, dec_len=7  (from above)                                │
│   Example 2: enc_len=12, dec_len=5                                               │
│   Example 3: enc_len=20, dec_len=9                                               │
│   Example 4: enc_len=8,  dec_len=4                                               │
│                                                                                   │
│   max_enc_len = 20                                                                │
│   max_dec_len = 9                                                                 │
│                                                                                   │
│   After padding:                                                                  │
│                                                                                   │
│   enc_batch shape: [4, 20]                                                       │
│   ┌────────────────────────────────────────────────────────────────────┐        │
│   │ Ex1: [100,101,102,6,4,103,104,105,5,50000,50001,108,4,50002,107,5,│        │
│   │       0, 0, 0, 0]                                                  │        │
│   │                ↑↑↑↑ = 4 PAD tokens                                 │        │
│   │ Ex2: [... 12 real tokens ..., 0, 0, 0, 0, 0, 0, 0, 0]            │        │
│   │                                ↑↑↑↑↑↑↑↑ = 8 PAD tokens            │        │
│   │ Ex3: [... 20 real tokens ...]  (no padding)                       │        │
│   │ Ex4: [... 8 real tokens ..., 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] │        │
│   └────────────────────────────────────────────────────────────────────┘        │
│                                                                                   │
│   enc_padding_mask shape: [4, 20]                                                │
│   ┌────────────────────────────────────────────────────────────────────┐        │
│   │ Ex1: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0]                    │        │
│   │ Ex2: [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0]                    │        │
│   │ Ex3: [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]                    │        │
│   │ Ex4: [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0]                    │        │
│   └────────────────────────────────────────────────────────────────────┘        │
│                                                                                   │
│   enc_lens: [16, 12, 20, 8]                                                      │
│                                                                                   │
│   max_art_oovs: max(3, 1, 5, 0) = 5                                             │
│   (Example 3 has 5 OOV words, the most in this batch)                           │
│                                                                                   │
│                                                                                   │
│   ═══════════════════════════════════════════════════════════════════════════   │
│   STEP 3: FEED TO MODEL                                                           │
│   ═══════════════════════════════════════════════════════════════════════════   │
│                                                                                   │
│   batch = batcher.next_batch()                                                   │
│                                                                                   │
│   feed_dict = {                                                                   │
│       enc_batch:              batch.enc_batch,              # [4, 20]            │
│       enc_lens:               batch.enc_lens,               # [4]                │
│       enc_padding_mask:       batch.enc_padding_mask,       # [4, 20]            │
│       enc_batch_extend_vocab: batch.enc_batch_extend_vocab, # [4, 20]            │
│       max_art_oovs:           batch.max_art_oovs,           # 5                  │
│       dec_batch:              batch.dec_batch,              # [4, 9]             │
│       target_batch:           batch.target_batch,           # [4, 9]             │
│       dec_padding_mask:       batch.dec_padding_mask,       # [4, 9]             │
│   }                                                                               │
│                                                                                   │
│   _, loss = session.run([train_op, loss_op], feed_dict=feed_dict)               │
│                                                                                   │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Summary

The data pipeline:

1. **Reads TFRecord files** containing article-abstract pairs
2. **Tokenizes and converts to IDs** using vocabulary
3. **Handles OOV words** with extended vocabulary for pointer mechanism
4. **Creates Example objects** with encoder/decoder sequences
5. **Batches examples** with padding to uniform length
6. **Uses multi-threading** for efficient data loading
7. **Provides masking** for attention and loss calculation

Key classes:
- **Vocab**: Word-to-ID mappings with special tokens
- **Example**: Single article-abstract pair processing
- **Batch**: Multiple examples combined into tensors
- **Batcher**: Multi-threaded data loading manager

---

*Next: [08_vocabulary.md](08_vocabulary.md) - Vocabulary and OOV Handling Deep Dive*
