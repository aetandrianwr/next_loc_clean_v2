# Data Processing Comparison

## Table of Contents
1. [Overview](#overview)
2. [Data Format Comparison](#data-format-comparison)
3. [Original Data Pipeline](#original-data-pipeline)
4. [Proposed Data Pipeline](#proposed-data-pipeline)
5. [Batching Strategies](#batching-strategies)
6. [Vocabulary Handling](#vocabulary-handling)
7. [Example Walkthrough](#example-walkthrough)
8. [Code Comparison](#code-comparison)

---

## Overview

| Aspect | Original | Proposed |
|--------|----------|----------|
| **Task** | Text Summarization | Next Location Prediction |
| **Data Format** | Binary tf.Example files | Pickle files |
| **Data Loading** | Multi-threaded queue | PyTorch DataLoader |
| **Vocabulary** | 50K word vocab + OOV handling | Location IDs (numeric) |
| **Sequence Length** | Variable (up to 400 enc, 100 dec) | Fixed (window size, e.g., 50) |
| **Features** | Word IDs only | Multi-modal (user, time, etc.) |

---

## Data Format Comparison

### Original: Text Summarization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: TEXT SUMMARIZATION DATA                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Source Document:                                                            │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  "The president announced new economic policies today at the         │   │
│  │   White House. The measures include tax cuts for middle-class        │   │
│  │   families and increased funding for infrastructure projects..."     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Target Summary:                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  "President unveils economic plan with tax cuts."                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Binary File Format (tf.Example):                                           │
│  {                                                                           │
│      "article": "the president announced new economic...",                  │
│      "abstract": "<s> president unveils economic plan... </s>"              │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Next Location Prediction

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                  PROPOSED: NEXT LOCATION PREDICTION DATA                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Trajectory Sequence:                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  User: Alice (ID: 42)                                                │   │
│  │                                                                       │   │
│  │  Visit 1: Home (101)      @ 08:00  Monday,  30 min                   │   │
│  │  Visit 2: Coffee (205)    @ 08:30  Monday,  15 min                   │   │
│  │  Visit 3: Office (150)    @ 09:00  Monday,  480 min                  │   │
│  │  Visit 4: Restaurant (312)@ 17:00  Monday,  60 min                   │   │
│  │  Visit 5: Office (150)    @ 18:00  Monday,  120 min                  │   │
│  │                                                                       │   │
│  │  Target: Gym (89) @ 20:00                                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Pickle File Format:                                                        │
│  {                                                                           │
│      'locations': [101, 205, 150, 312, 150],  # Input sequence             │
│      'user_ids': [42, 42, 42, 42, 42],                                     │
│      'time_indices': [8, 8, 9, 17, 18],       # Hour of day                │
│      'weekday': [0, 0, 0, 0, 0],              # Monday = 0                 │
│      'durations': [30, 15, 480, 60, 120],     # Minutes                    │
│      'target': 89                              # Next location              │
│  }                                                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Original Data Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL DATA PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         BINARY FILES                                     ││
│  │  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐                ││
│  │  │chunk_00│ │chunk_01│ │chunk_02│ │chunk_03│ │...     │                ││
│  │  │.bin    │ │.bin    │ │.bin    │ │.bin    │ │        │                ││
│  │  └────┬───┘ └────┬───┘ └────┬───┘ └────┬───┘ └────────┘                ││
│  │       │          │          │          │                                 ││
│  │       └──────────┴──────────┴──────────┘                                ││
│  │                        │                                                 ││
│  │                        ▼                                                 ││
│  │              ┌────────────────────┐                                      ││
│  │              │  FILE QUEUE        │                                      ││
│  │              │  (shuffled files)  │                                      ││
│  │              └────────┬───────────┘                                      ││
│  │                       │                                                  ││
│  │     ┌─────────────────┼─────────────────┐                               ││
│  │     │                 │                 │                                ││
│  │     ▼                 ▼                 ▼                                ││
│  │  ┌──────┐         ┌──────┐         ┌──────┐                             ││
│  │  │Reader│         │Reader│         │Reader│  (4 Reader threads)        ││
│  │  │Thread│         │Thread│         │Thread│                             ││
│  │  │  1   │         │  2   │         │  3   │                             ││
│  │  └───┬──┘         └───┬──┘         └───┬──┘                             ││
│  │      │                │                │                                 ││
│  │      └────────────────┼────────────────┘                                ││
│  │                       │                                                  ││
│  │                       ▼                                                  ││
│  │              ┌────────────────────┐                                      ││
│  │              │  EXAMPLE QUEUE     │                                      ││
│  │              │  (single examples) │                                      ││
│  │              │  bucket_queue_size │                                      ││
│  │              │  = 1000            │                                      ││
│  │              └────────┬───────────┘                                      ││
│  │                       │                                                  ││
│  │                       ▼                                                  ││
│  │              ┌────────────────────┐                                      ││
│  │              │  BUCKETING         │                                      ││
│  │              │  (by seq length)   │                                      ││
│  │              └────────┬───────────┘                                      ││
│  │                       │                                                  ││
│  │                       ▼                                                  ││
│  │              ┌────────────────────┐                                      ││
│  │              │  BATCH QUEUE       │                                      ││
│  │              │  (padded batches)  │                                      ││
│  │              │  max_batches = 100 │                                      ││
│  │              └────────┬───────────┘                                      ││
│  │                       │                                                  ││
│  └───────────────────────┼─────────────────────────────────────────────────┘│
│                          ▼                                                   │
│                  ┌───────────────┐                                           │
│                  │    BATCH      │                                           │
│                  │  (ready for   │                                           │
│                  │   training)   │                                           │
│                  └───────────────┘                                           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original Code

```python
# File: batcher.py, lines 37-78

class Batcher(object):
    """A class to generate minibatches of data."""
    
    BATCH_QUEUE_MAX = 100  # max number of batches the batch_queue can hold
    
    def __init__(self, data_path, vocab, hps, single_pass):
        self._data_path = data_path
        self._vocab = vocab
        self._hps = hps
        self._single_pass = single_pass
        
        # Initialize queue for filenames
        self._batch_queue = Queue.Queue(self.BATCH_QUEUE_MAX)
        self._example_queue = Queue.Queue(self.BATCH_QUEUE_MAX * self._hps.batch_size)
        
        # Different behavior for single_pass (decode) vs. training
        if single_pass:
            self._num_example_q_threads = 1
            self._num_batch_q_threads = 1
            self._bucketing_cache_size = 1
        else:
            self._num_example_q_threads = 16  # 16 threads for examples
            self._num_batch_q_threads = 4    # 4 threads for batching
            self._bucketing_cache_size = 100  # Cache 100 examples for bucketing
        
        # Start threads
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
```

---

## Proposed Data Pipeline

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED DATA PIPELINE                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                         PICKLE FILES                                     ││
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐                         ││
│  │  │train.pkl   │  │val.pkl     │  │test.pkl    │                         ││
│  │  │            │  │            │  │            │                         ││
│  │  │- locations │  │- locations │  │- locations │                         ││
│  │  │- user_ids  │  │- user_ids  │  │- user_ids  │                         ││
│  │  │- times     │  │- times     │  │- times     │                         ││
│  │  │- weekdays  │  │- weekdays  │  │- weekdays  │                         ││
│  │  │- durations │  │- durations │  │- durations │                         ││
│  │  │- targets   │  │- targets   │  │- targets   │                         ││
│  │  └──────┬─────┘  └──────┬─────┘  └──────┬─────┘                         ││
│  │         │               │               │                                ││
│  └─────────┼───────────────┼───────────────┼────────────────────────────────┘│
│            │               │               │                                 │
│            ▼               ▼               ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                      PYTORCH DATASETS                                    ││
│  │                                                                          ││
│  │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    ││
│  │  │NextLocationDataset│  │NextLocationDataset│  │NextLocationDataset│   ││
│  │  │     (train)       │  │     (val)         │  │     (test)        │   ││
│  │  │                   │  │                   │  │                   │    ││
│  │  │ __len__()         │  │ __len__()         │  │ __len__()         │    ││
│  │  │ __getitem__(idx)  │  │ __getitem__(idx)  │  │ __getitem__(idx)  │    ││
│  │  └────────┬──────────┘  └────────┬──────────┘  └────────┬──────────┘   ││
│  │           │                      │                      │               ││
│  └───────────┼──────────────────────┼──────────────────────┼───────────────┘│
│              │                      │                      │                │
│              ▼                      ▼                      ▼                │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                       PYTORCH DATALOADERS                               ││
│  │                                                                          ││
│  │  ┌──────────────────┐   ┌──────────────────┐   ┌──────────────────┐    ││
│  │  │  DataLoader      │   │  DataLoader      │   │  DataLoader      │    ││
│  │  │  (train)         │   │  (val)           │   │  (test)          │    ││
│  │  │                  │   │                  │   │                  │     ││
│  │  │  batch_size=128  │   │  batch_size=256  │   │  batch_size=256  │    ││
│  │  │  shuffle=True    │   │  shuffle=False   │   │  shuffle=False   │    ││
│  │  │  num_workers=0   │   │  num_workers=0   │   │  num_workers=0   │    ││
│  │  │  pin_memory=True │   │  pin_memory=True │   │  pin_memory=True │    ││
│  │  └────────┬─────────┘   └────────┬─────────┘   └────────┬─────────┘    ││
│  │           │                      │                      │               ││
│  └───────────┼──────────────────────┼──────────────────────┼───────────────┘│
│              │                      │                      │                │
│              ▼                      ▼                      ▼                │
│          ┌───────┐              ┌───────┐              ┌───────┐           │
│          │ BATCH │              │ BATCH │              │ BATCH │           │
│          │(x,y,d)│              │(x,y,d)│              │(x,y,d)│           │
│          └───────┘              └───────┘              └───────┘           │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed Code

```python
# File: train_pgt.py, lines 119-195

class NextLocationDataset(Dataset):
    """Dataset for next location prediction."""
    
    def __init__(self, data_dict: Dict, window_size: int):
        self.window_size = window_size
        
        # Core data: location sequences and targets
        self.locations = data_dict['locations']    # List of location ID lists
        self.targets = data_dict['targets']        # List of target location IDs
        
        # Optional features (may not be present in all datasets)
        self.user_ids = data_dict.get('user_ids', None)
        self.time_indices = data_dict.get('time_indices', None)
        self.weekday = data_dict.get('weekday', None)
        self.durations = data_dict.get('durations', None)
        self.recency = data_dict.get('recency', None)
    
    def __len__(self) -> int:
        return len(self.targets)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        # Get location sequence
        locations = self.locations[idx]
        target = self.targets[idx]
        
        # Pad/truncate to window_size
        if len(locations) > self.window_size:
            locations = locations[-self.window_size:]  # Take last N
        elif len(locations) < self.window_size:
            # Pad with zeros at the beginning
            padding = [0] * (self.window_size - len(locations))
            locations = padding + locations
        
        x = torch.tensor(locations, dtype=torch.long)
        y = torch.tensor(target, dtype=torch.long)
        
        # Additional features dictionary
        x_dict = {}
        
        if self.user_ids is not None:
            user_id = self.user_ids[idx]
            x_dict['user_id'] = torch.tensor([user_id] * self.window_size, dtype=torch.long)
        
        if self.time_indices is not None:
            times = self.time_indices[idx]
            times = self._pad_or_truncate(times)
            x_dict['time_idx'] = torch.tensor(times, dtype=torch.long)
        
        if self.weekday is not None:
            weekdays = self.weekday[idx]
            weekdays = self._pad_or_truncate(weekdays)
            x_dict['weekday'] = torch.tensor(weekdays, dtype=torch.long)
        
        if self.durations is not None:
            durs = self.durations[idx]
            durs = self._pad_or_truncate(durs)
            x_dict['duration'] = torch.tensor(durs, dtype=torch.long)
        
        if self.recency is not None:
            rec = self.recency[idx]
            rec = self._pad_or_truncate(rec)
            x_dict['recency'] = torch.tensor(rec, dtype=torch.long)
        
        return x, y, x_dict
    
    def _pad_or_truncate(self, seq):
        """Pad or truncate sequence to window_size."""
        if len(seq) > self.window_size:
            return seq[-self.window_size:]
        elif len(seq) < self.window_size:
            return [0] * (self.window_size - len(seq)) + seq
        return seq
```

---

## Batching Strategies

### Original: Bucketing by Sequence Length

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: BUCKETING BY SEQUENCE LENGTH                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Problem: Variable-length sequences require padding                          │
│           Padding short sequences to match long ones wastes computation     │
│                                                                              │
│  Solution: Group similar-length sequences together                          │
│                                                                              │
│  Example:                                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Raw examples:                                                        │   │
│  │    Ex1: 50 words    Ex2: 200 words   Ex3: 55 words                   │   │
│  │    Ex4: 210 words   Ex5: 300 words   Ex6: 45 words                   │   │
│  │                                                                       │   │
│  │  After bucketing:                                                     │   │
│  │    Bucket 1 (short): Ex1(50), Ex3(55), Ex6(45)  → pad to 55         │   │
│  │    Bucket 2 (medium): Ex2(200), Ex4(210)        → pad to 210        │   │
│  │    Bucket 3 (long): Ex5(300)                    → pad to 300        │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Code:                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # File: batcher.py, lines 101-130                                   │   │
│  │                                                                       │   │
│  │  def fill_batch_queue(self):                                         │   │
│  │      while True:                                                     │   │
│  │          # Fill bucketing cache                                      │   │
│  │          inputs = []                                                 │   │
│  │          for _ in range(self._hps.batch_size * self._bucketing...): │   │
│  │              inputs.append(self._example_queue.get())               │   │
│  │                                                                       │   │
│  │          # Sort by encoder sequence length (for bucketing)          │   │
│  │          inputs = sorted(inputs, key=lambda x: x.enc_len)           │   │
│  │                                                                       │   │
│  │          # Group into batches of size batch_size                    │   │
│  │          batches = []                                                │   │
│  │          for i in range(0, len(inputs), self._hps.batch_size):      │   │
│  │              batches.append(inputs[i:i + self._hps.batch_size])     │   │
│  │                                                                       │   │
│  │          # Shuffle batches (not examples within batch)              │   │
│  │          shuffle(batches)                                           │   │
│  │                                                                       │   │
│  │          # Add to batch queue                                        │   │
│  │          for b in batches:                                          │   │
│  │              self._batch_queue.put(Batch(b, self._hps, self._vocab))│   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Fixed Window Size

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: FIXED WINDOW SIZE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  All sequences are pre-processed to same length:                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  Config: window_size = 50                                            │   │
│  │                                                                       │   │
│  │  Raw trajectories:                                                    │   │
│  │    User1: [loc1, loc2, loc3]                     (length 3)          │   │
│  │    User2: [loc1, loc2, ..., loc75]               (length 75)         │   │
│  │    User3: [loc1, loc2, ..., loc50]               (length 50)         │   │
│  │                                                                       │   │
│  │  After padding/truncation:                                            │   │
│  │    User1: [0,0,0,...,loc1,loc2,loc3]             (50 tokens)         │   │
│  │    User2: [loc26,loc27,...,loc75]                (50 tokens, recent) │   │
│  │    User3: [loc1,loc2,...,loc50]                  (50 tokens, exact)  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Benefits:                                                                   │
│  - No bucketing needed                                                       │
│  - Simpler batching logic                                                    │
│  - Consistent memory usage                                                   │
│  - Easier to parallelize                                                     │
│                                                                              │
│  Code:                                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # In Dataset.__getitem__:                                           │   │
│  │  if len(locations) > self.window_size:                              │   │
│  │      locations = locations[-self.window_size:]  # Keep most recent  │   │
│  │  elif len(locations) < self.window_size:                            │   │
│  │      padding = [0] * (self.window_size - len(locations))            │   │
│  │      locations = padding + locations  # Pad at beginning            │   │
│  │                                                                       │   │
│  │  # In DataLoader:                                                    │   │
│  │  train_loader = DataLoader(                                         │   │
│  │      dataset=train_dataset,                                         │   │
│  │      batch_size=128,     # All sequences same length, easy batch   │   │
│  │      shuffle=True,                                                  │   │
│  │      pin_memory=True,                                               │   │
│  │  )                                                                   │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Vocabulary Handling

### Original: Large Text Vocabulary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ORIGINAL: TEXT VOCABULARY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Vocabulary Structure:                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  vocab_size = 50,000                                                 │   │
│  │                                                                       │   │
│  │  Special tokens:                                                     │   │
│  │    [PAD] = 0                                                         │   │
│  │    [UNK] = 1                                                         │   │
│  │    [START] = 2                                                       │   │
│  │    [STOP] = 3                                                        │   │
│  │                                                                       │   │
│  │  Regular words:                                                      │   │
│  │    "the" = 4                                                         │   │
│  │    "a" = 5                                                           │   │
│  │    ... = ...                                                         │   │
│  │    "president" = 156                                                 │   │
│  │    ... = ...                                                         │   │
│  │    "zygote" = 49,999                                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OOV (Out-of-Vocabulary) Handling:                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  For pointer mechanism, OOV words get temporary IDs:                │   │
│  │                                                                       │   │
│  │  Input: "The CEO announced new policies"                            │   │
│  │                                                                       │   │
│  │  If "CEO" is OOV:                                                   │   │
│  │    - Regular encoding: [4, 1, 156, 89, 234]  (1 = UNK for CEO)     │   │
│  │    - Extended encoding: [4, 50000, 156, 89, 234]                    │   │
│  │                          (50000 = first extended vocab ID)          │   │
│  │                                                                       │   │
│  │  The pointer can then copy "CEO" (ID 50000) from input              │   │
│  │  even though it's not in the generation vocabulary                  │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Code: (batcher.py, lines 166-192)                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  def article2ids(article_words, vocab):                             │   │
│  │      ids = []                                                       │   │
│  │      oovs = []                                                      │   │
│  │      unk_id = vocab.word2id(UNKNOWN_TOKEN)                         │   │
│  │                                                                       │   │
│  │      for w in article_words:                                        │   │
│  │          i = vocab.word2id(w)                                       │   │
│  │          if i == unk_id:  # OOV word                               │   │
│  │              if w not in oovs:                                      │   │
│  │                  oovs.append(w)                                     │   │
│  │              oov_num = oovs.index(w)                                │   │
│  │              ids.append(vocab.size() + oov_num)  # Extended ID     │   │
│  │          else:                                                      │   │
│  │              ids.append(i)                                          │   │
│  │      return ids, oovs                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Proposed: Location ID Vocabulary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PROPOSED: LOCATION VOCABULARY                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Vocabulary Structure:                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  num_locations = 500 (dataset-dependent)                            │   │
│  │                                                                       │   │
│  │  Location IDs are pre-assigned integers:                            │   │
│  │    [PAD] = 0                                                         │   │
│  │    Home = 101                                                        │   │
│  │    Office = 150                                                      │   │
│  │    Coffee Shop = 205                                                 │   │
│  │    Restaurant = 312                                                  │   │
│  │    Gym = 89                                                          │   │
│  │    ...                                                               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  No OOV Handling Needed:                                                    │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  - All locations are known in advance                               │   │
│  │  - Locations are discrete POIs (points of interest)                 │   │
│  │  - Dataset preprocessing ensures all locations have IDs             │   │
│  │  - No "unknown location" concept                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  The "vocabulary" is simply the location embedding:                         │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # pgt.py, line 98                                          │   │
│  │  self.location_embedding = nn.Embedding(num_locations, d_model)     │   │
│  │                                                                       │   │
│  │  # Example: 500 locations × 64 dims = 32,000 parameters             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Extended Vocabulary for Pointer:                                           │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  NOT NEEDED! Unlike text where rare words may not be in vocab,      │   │
│  │  all locations are known. The pointer simply points to locations    │   │
│  │  that exist in the input sequence.                                  │   │
│  │                                                                       │   │
│  │  Input: [101, 205, 150, 312, 150]                                   │   │
│  │  Pointer can select any of: {101, 205, 150, 312}                    │   │
│  │  Generation can output any of: {1, 2, ..., 500}                     │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Example Walkthrough

### Alice's Day Trip: Data Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               EXAMPLE: ALICE'S DATA THROUGH THE PIPELINE                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RAW DATA (in pickle file):                                                  │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  {                                                                    │   │
│  │      'locations': [101, 205, 150, 312, 150],                        │   │
│  │      'user_ids': [42, 42, 42, 42, 42],                              │   │
│  │      'time_indices': [8, 8, 9, 17, 18],                             │   │
│  │      'weekday': [0, 0, 0, 0, 0],                                    │   │
│  │      'durations': [30, 15, 480, 60, 120],                           │   │
│  │      'recency': [5, 4, 3, 2, 1],                                    │   │
│  │      'target': 89                                                    │   │
│  │  }                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  AFTER DATASET.__getitem__(idx=0):                                          │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # Assuming window_size=50                                           │   │
│  │                                                                       │   │
│  │  x = tensor([0,0,0,...,0,0,0,101,205,150,312,150])  # shape: [50]   │   │
│  │             └───────────────────┘ └──────────────┘                   │   │
│  │               45 zeros (padding)   5 locations                       │   │
│  │                                                                       │   │
│  │  y = tensor(89)  # Target: Gym                                       │   │
│  │                                                                       │   │
│  │  x_dict = {                                                          │   │
│  │      'user_id': tensor([42,42,42,...,42,42])      # shape: [50]     │   │
│  │      'time_idx': tensor([0,0,0,...,8,8,9,17,18])  # shape: [50]     │   │
│  │      'weekday': tensor([0,0,0,...,0,0,0,0,0])     # shape: [50]     │   │
│  │      'duration': tensor([0,0,0,...,30,15,480,60,120])               │   │
│  │      'recency': tensor([0,0,0,...,5,4,3,2,1])                       │   │
│  │  }                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  AFTER DATALOADER BATCHING (batch_size=128):                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  x_batch = tensor(shape [128, 50])       # 128 sequences            │   │
│  │  y_batch = tensor(shape [128])           # 128 targets              │   │
│  │  x_dict_batch = {                                                    │   │
│  │      'user_id': tensor(shape [128, 50])                             │   │
│  │      'time_idx': tensor(shape [128, 50])                            │   │
│  │      'weekday': tensor(shape [128, 50])                             │   │
│  │      'duration': tensor(shape [128, 50])                            │   │
│  │      'recency': tensor(shape [128, 50])                             │   │
│  │  }                                                                    │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Original: News Article Processing

```
┌─────────────────────────────────────────────────────────────────────────────┐
│               EXAMPLE: NEWS ARTICLE THROUGH ORIGINAL PIPELINE                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  RAW DATA (from tf.Example):                                                │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  article = "the president announced new economic policies today..."  │   │
│  │  abstract = "<s> president unveils economic plan </s>"               │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  AFTER TOKENIZATION:                                                        │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  article_words = ['the', 'president', 'announced', 'new',           │   │
│  │                   'economic', 'policies', 'today', ...]             │   │
│  │                                                                       │   │
│  │  abstract_words = ['president', 'unveils', 'economic', 'plan']      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  AFTER article2ids (with OOV handling):                                     │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  # Assume 'unveils' is OOV                                          │   │
│  │                                                                       │   │
│  │  enc_input = [4, 156, 2341, 89, 1502, 3421, 5012, ...]             │   │
│  │              (word IDs for article)                                  │   │
│  │                                                                       │   │
│  │  enc_input_extend = [4, 156, 2341, 89, 1502, 3421, 5012, ...]      │   │
│  │              (extended IDs, may have OOV IDs > vocab_size)          │   │
│  │                                                                       │   │
│  │  article_oovs = ['unveils']  # OOV words                            │   │
│  │                                                                       │   │
│  │  dec_input = [2, 156, 50000, 1502, 234]  # START + words            │   │
│  │                       └─────── OOV 'unveils' gets ID 50000         │   │
│  │                                                                       │   │
│  │  target = [156, 50000, 1502, 234, 3]     # words + STOP             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  AFTER Batch PADDING:                                                       │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │  enc_batch = [batch_size, max_enc_seq_len]  # Padded encoder input │   │
│  │  dec_batch = [batch_size, max_dec_seq_len]  # Padded decoder input │   │
│  │  target_batch = [batch_size, max_dec_seq_len]                       │   │
│  │                                                                       │   │
│  │  enc_padding_mask = [batch_size, max_enc_seq_len]  # 1 for padding │   │
│  │  dec_padding_mask = [batch_size, max_dec_seq_len]                   │   │
│  │                                                                       │   │
│  │  max_oovs = max number of OOVs in batch                             │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Comparison

### Loading Data

```python
# ==============================================================================
# ORIGINAL: Reading tf.Example files
# ==============================================================================

# File: batcher.py, lines 79-99

def fill_example_queue(self):
    """Reads data from file and processes into examples."""
    input_gen = self.text_generator(
        data.example_generator(self._data_path, self._single_pass)
    )
    
    while True:
        try:
            (article, abstract) = input_gen.next()
        except StopIteration:
            tf.logging.info("example generator exhausted")
            break
        
        # Process article and abstract
        article_words = article.split()
        abstract_sentences = data.abstract2sents(abstract)
        
        # Create Example object
        example = Example(article_words, abstract_sentences, self._vocab, self._hps)
        self._example_queue.put(example)

# File: data.py, lines 53-75

def example_generator(data_path, single_pass):
    """Generates tf.Examples from data files."""
    while True:
        filelist = glob.glob(data_path)
        if not single_pass:
            random.shuffle(filelist)
        
        for f in filelist:
            reader = open(f, 'rb')
            while True:
                len_bytes = reader.read(8)
                if not len_bytes:
                    break
                str_len = struct.unpack('q', len_bytes)[0]
                example_str = struct.unpack('%ds' % str_len, reader.read(str_len))[0]
                yield tf.train.Example.FromString(example_str)

# ==============================================================================
# PROPOSED: Loading pickle files
# ==============================================================================

# File: train_pgt.py, lines 260-285

def load_data(config: Dict) -> Tuple[Dataset, Dataset, Dataset]:
    """Load train, validation, and test datasets."""
    data_path = Path(config['data']['path'])
    window_size = config['data']['window_size']
    
    # Load pickle files
    with open(data_path / 'train.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    with open(data_path / 'val.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    with open(data_path / 'test.pkl', 'rb') as f:
        test_data = pickle.load(f)
    
    # Create datasets
    train_dataset = NextLocationDataset(train_data, window_size)
    val_dataset = NextLocationDataset(val_data, window_size)
    test_dataset = NextLocationDataset(test_data, window_size)
    
    return train_dataset, val_dataset, test_dataset

# File: train_pgt.py, lines 290-310

def create_dataloaders(train_dataset, val_dataset, test_dataset, config):
    """Create DataLoaders."""
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=0,  # No multiprocessing
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        pin_memory=True,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'] * 2,
        shuffle=False,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader
```

---

## Summary Comparison Table

| Aspect | Original | Proposed |
|--------|----------|----------|
| **File Format** | Binary tf.Example | Pickle dict |
| **Data Source** | Streaming from disk | Loaded into memory |
| **Threading** | 16 example + 4 batch threads | PyTorch DataLoader workers |
| **Batching** | Dynamic bucketing by length | Fixed window size |
| **Padding** | Dynamic per-batch | Fixed (window_size) |
| **Vocabulary** | 50K words + OOV handling | N locations (e.g., 500) |
| **OOV Handling** | Extended vocab IDs | Not needed |
| **Features** | Word IDs only | Multi-modal features |
| **Memory Usage** | Low (streaming) | Higher (all in memory) |
| **Flexibility** | Variable seq lengths | Fixed seq lengths |

---

*Next: [09_LOSS_AND_METRICS.md](09_LOSS_AND_METRICS.md) - Loss functions and evaluation metrics*
