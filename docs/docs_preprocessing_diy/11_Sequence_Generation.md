# Sequence Generation Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [What is a Sequence?](#what-is-a-sequence)
3. [Sequence Generation Algorithm](#sequence-generation-algorithm)
4. [Sliding Window Mechanism](#sliding-window-mechanism)
5. [Parallel Processing](#parallel-processing)
6. [Edge Cases and Handling](#edge-cases-and-handling)
7. [Complete Code Walkthrough](#complete-code-walkthrough)
8. [Output Format](#output-format)

---

## Overview

Sequence generation is the final step in preprocessing that transforms staypoint data into training samples for the next location prediction model.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE GENERATION OVERVIEW                              │
└─────────────────────────────────────────────────────────────────────────────┘

Input: Processed staypoints with temporal features
─────────────────────────────────────────────────

    user_id │ location_id │ start_day │ weekday │ start_min │ duration
    ────────┼─────────────┼───────────┼─────────┼───────────┼──────────
    user_01 │     42      │     0     │    0    │    420    │   540
    user_01 │     15      │     0     │    0    │   1080    │   720
    user_01 │     42      │     1     │    1    │    450    │   510
    user_01 │      8      │     1     │    1    │   1200    │    60
    user_01 │     15      │     2     │    2    │    480    │   540
    ...

Output: Training sequences
──────────────────────────

    {
        "X": [42, 15, 42, 8],           # History locations
        "user_X": [1, 1, 1, 1],         # User ID (encoded)
        "weekday_X": [0, 0, 1, 1],      # Day of week
        "start_min_X": [420, 1080, 450, 1200],
        "dur_X": [540, 720, 510, 60],   # Duration
        "diff": [2, 2, 1, 1],           # Days ago
        "Y": 15                          # Target location
    }

Each staypoint becomes ONE target prediction!
History = previous staypoints within window
```

---

## What is a Sequence?

### Sequence Structure

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE STRUCTURE                                        │
└─────────────────────────────────────────────────────────────────────────────┘

A sequence answers: "Given historical visits, predict the next location"

Structure:
──────────

    ┌───────────────────────────────────────────┐    ┌─────────┐
    │              HISTORY (X)                   │    │ TARGET  │
    │                                           │    │  (Y)    │
    │  Day 0      Day 1      Day 2      Day 3  │    │  Day 4  │
    │ ┌──┐┌──┐   ┌──┐┌──┐   ┌──┐       ┌──┐   │    │  ┌──┐   │
    │ │42││15│   │42││ 8│   │15│       │42│   │───▶│  │17│   │
    │ └──┘└──┘   └──┘└──┘   └──┘       └──┘   │    │  └──┘   │
    │                                           │    │         │
    │ Location IDs from past N days             │    │ Predict │
    └───────────────────────────────────────────┘    └─────────┘

The model learns: P(Y | X, temporal_features, user)
```

### Sequence Components

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE DICTIONARY KEYS                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Key          │ Type        │ Shape       │ Description
─────────────┼─────────────┼─────────────┼─────────────────────────────────────
X            │ List[int]   │ (seq_len,)  │ Historical location IDs
user_X       │ List[int]   │ (seq_len,)  │ User ID (same for all positions)
weekday_X    │ List[int]   │ (seq_len,)  │ Day of week (0-6) for each visit
start_min_X  │ List[int]   │ (seq_len,)  │ Start minute (0-1439) for each
dur_X        │ List[float] │ (seq_len,)  │ Duration in minutes for each
diff         │ List[int]   │ (seq_len,)  │ Days ago for each historical visit
Y            │ int         │ scalar      │ Target location ID to predict

Note: seq_len varies per sequence (depends on user's history)
```

---

## Sequence Generation Algorithm

### High-Level Algorithm

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE GENERATION ALGORITHM                             │
└─────────────────────────────────────────────────────────────────────────────┘

For each user:
    For each staypoint (as potential target):
        1. Get target staypoint info (location, day, etc.)
        2. Define history window: [target_day - previous_day, target_day)
        3. Filter historical staypoints within window
        4. Check if valid (enough history, meets minimum length)
        5. If valid, create sequence dictionary
        6. Add to output list

Visual:
───────

    User's staypoints timeline:
    
    Day: 0    1    2    3    4    5    6    7    8    9    10
         │    │    │    │    │    │    │    │    │    │    │
         *    **   *    **   *    *    **   *    *    **   *
         │                        │                        │
         └──────── history ───────┴────── target ──────────┘
         
    For target on Day 7 with previous_day=7:
    • History window: Days 0-6
    • Valid if enough staypoints in window
```

### Pseudocode

```python
def generate_sequences(user_df, previous_day, min_sequence_length, max_duration):
    """
    Generate sequences for one user.
    
    Args:
        user_df: DataFrame of user's staypoints (sorted by time)
        previous_day: Number of days to look back for history
        min_sequence_length: Minimum historical staypoints required
        max_duration: Maximum duration value (for capping)
    
    Returns:
        List of sequence dictionaries
    """
    sequences = []
    
    # Get encoded user ID (same for all sequences from this user)
    user_id = user_df['user_id'].iloc[0]
    
    # Iterate through each staypoint as potential target
    for target_idx in range(len(user_df)):
        target = user_df.iloc[target_idx]
        target_day = target['start_day']
        
        # Define history window
        window_start = target_day - previous_day
        
        # Get historical staypoints (strictly before target)
        history_mask = (
            (user_df['start_day'] >= window_start) & 
            (user_df['start_day'] < target_day) &
            (user_df.index < target_idx)  # Must be before target in timeline
        )
        history = user_df[history_mask]
        
        # Check validity
        if len(history) < min_sequence_length:
            continue  # Skip if not enough history
        
        # Create sequence
        sequence = {
            'X': history['location_id'].tolist(),
            'user_X': [user_id] * len(history),
            'weekday_X': history['weekday'].tolist(),
            'start_min_X': history['start_min'].tolist(),
            'dur_X': history['duration'].clip(upper=max_duration).tolist(),
            'diff': (target_day - history['start_day']).tolist(),
            'Y': target['location_id']
        }
        
        sequences.append(sequence)
    
    return sequences
```

---

## Sliding Window Mechanism

### Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW FOR SEQUENCE GENERATION                    │
└─────────────────────────────────────────────────────────────────────────────┘

Configuration: previous_day = 7

User's staypoints over 10 days:
───────────────────────────────

    Day:    0    1    2    3    4    5    6    7    8    9
            │    │    │    │    │    │    │    │    │    │
    SP:     A    BC   D    E    F    GH   I    J    KL   M


TARGET 1: Staypoint J (Day 7)
─────────────────────────────

    Window: Days 0-6 (7 days before target)
    
    Day:    0    1    2    3    4    5    6  │  7
            │    │    │    │    │    │    │  │  │
    SP:     A    BC   D    E    F    GH   I  │  J
            └────────────────────────────────┘  │
                    HISTORY (X)                 │
                                             TARGET (Y)
    
    X = [A, B, C, D, E, F, G, H, I]
    Y = J


TARGET 2: Staypoint K (Day 8)
─────────────────────────────

    Window: Days 1-7 (window slides forward)
    
    Day:         1    2    3    4    5    6    7  │  8
                 │    │    │    │    │    │    │  │  │
    SP:          BC   D    E    F    GH   I    J  │  KL
                 └────────────────────────────────┘  │
                         HISTORY (X)                 │
                                                  TARGET (Y)
    
    X = [B, C, D, E, F, G, H, I, J]
    Y = K
    
    Note: Staypoint A (Day 0) is now outside the window!


TARGET 3: Staypoint M (Day 9)
─────────────────────────────

    Window: Days 2-8
    
    Day:              2    3    4    5    6    7    8  │  9
                      │    │    │    │    │    │    │  │  │
    SP:               D    E    F    GH   I    J    KL │  M
                      └────────────────────────────────┘  │
                               HISTORY (X)                │
                                                       TARGET (Y)
    
    X = [D, E, F, G, H, I, J, K, L]
    Y = M
```

### Window Parameters

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    WINDOW CONFIGURATION                                      │
└─────────────────────────────────────────────────────────────────────────────┘

Parameter: previous_day (default: 7)
────────────────────────────────────

    previous_day │ Window Size │ Captures                    │ Trade-off
    ─────────────┼─────────────┼─────────────────────────────┼─────────────
         3       │   3 days    │ Recent patterns only        │ Less context
         7       │   7 days    │ Full week (weekly patterns) │ Balanced
        14       │  14 days    │ Two weeks history           │ More context
        30       │  30 days    │ Monthly patterns            │ Longer sequences

Why 7 days (default)?
─────────────────────

    1. Captures weekly patterns
       - Same weekday last week provides strong signal
       - Mon-Fri work pattern, Sat-Sun leisure pattern
    
    2. Manageable sequence length
       - Typical user: 3-10 staypoints per day
       - 7 days ≈ 20-70 staypoints in history
    
    3. Balances recency and context
       - Recent visits are most predictive
       - Weekly patterns add valuable context


Parameter: min_sequence_length (default: 3)
───────────────────────────────────────────

    Minimum historical staypoints required to create a sequence.
    
    If user has fewer staypoints in window → sequence is SKIPPED
    
    Why 3 (default)?
    • At least some context for prediction
    • Filters out users with very sparse data
    • Balances sequence quantity vs quality
```

---

## Parallel Processing

### Why Parallel Processing?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL SEQUENCE GENERATION                              │
└─────────────────────────────────────────────────────────────────────────────┘

Problem: Sequential processing is slow
──────────────────────────────────────

    Users: 150
    Avg staypoints per user: 2000
    Total staypoints: 300,000
    
    Sequential: Process one user at a time
    Time: ~5-10 minutes
    
    Parallel: Process multiple users simultaneously
    Time: ~30-60 seconds (10x speedup)


Implementation using joblib:
────────────────────────────

    from joblib import Parallel, delayed
    
    # Process users in parallel
    results = Parallel(n_jobs=-1)(
        delayed(generate_user_sequences)(user_df, previous_day, ...)
        for user_id, user_df in grouped_df
    )
    
    # Flatten results
    all_sequences = [seq for user_seqs in results for seq in user_seqs]
```

### Parallel Processing Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PARALLEL PROCESSING FLOW                                  │
└─────────────────────────────────────────────────────────────────────────────┘

Input DataFrame (grouped by user):
──────────────────────────────────

    ┌─────────────────────────────────────────────────────────────────────┐
    │                          All Staypoints                              │
    │  User 1 data │ User 2 data │ User 3 data │ ... │ User N data       │
    └──────┬───────┴──────┬──────┴──────┬──────┴─────┴──────┬───────────┘
           │              │              │                    │
           ▼              ▼              ▼                    ▼
    ┌──────────────┬──────────────┬──────────────┬────┬──────────────┐
    │   Worker 1   │   Worker 2   │   Worker 3   │... │   Worker K   │
    │              │              │              │    │              │
    │ Process      │ Process      │ Process      │    │ Process      │
    │ User 1       │ User 2       │ User 3       │    │ Users N-K+1  │
    │ sequences    │ sequences    │ sequences    │    │ to N         │
    └──────┬───────┴──────┬───────┴──────┬───────┴────┴──────┬───────┘
           │              │              │                    │
           ▼              ▼              ▼                    ▼
    ┌──────────────┬──────────────┬──────────────┬────┬──────────────┐
    │ User 1 seqs  │ User 2 seqs  │ User 3 seqs  │... │ User N seqs  │
    │   (500)      │   (720)      │   (450)      │    │   (680)      │
    └──────┬───────┴──────┬───────┴──────┬───────┴────┴──────┬───────┘
           │              │              │                    │
           └──────────────┴──────────────┴────────────────────┘
                                    │
                                    ▼
                         ┌────────────────────┐
                         │  Combine Results   │
                         │   All Sequences    │
                         │     (65,000+)      │
                         └────────────────────┘
```

### Code Implementation

```python
from joblib import Parallel, delayed
from tqdm import tqdm

def generate_all_sequences(df, previous_day, min_length, max_duration, n_jobs=-1):
    """
    Generate sequences for all users in parallel.
    
    Args:
        df: DataFrame with all staypoints
        previous_day: History window size
        min_length: Minimum sequence length
        max_duration: Maximum duration cap
        n_jobs: Number of parallel workers (-1 = all CPUs)
    
    Returns:
        List of all sequences
    """
    # Group by user
    grouped = df.groupby('user_id')
    
    # Process each user in parallel
    results = Parallel(n_jobs=n_jobs)(
        delayed(generate_user_sequences)(
            user_df, 
            previous_day, 
            min_length, 
            max_duration
        )
        for user_id, user_df in tqdm(grouped, desc="Generating sequences")
    )
    
    # Flatten results
    all_sequences = []
    for user_sequences in results:
        all_sequences.extend(user_sequences)
    
    return all_sequences


def generate_user_sequences(user_df, previous_day, min_length, max_duration):
    """
    Generate sequences for a single user.
    """
    sequences = []
    user_id = user_df['user_id'].iloc[0]
    
    # Sort by time to ensure correct order
    user_df = user_df.sort_values('started_at').reset_index(drop=True)
    
    for target_idx in range(len(user_df)):
        target = user_df.iloc[target_idx]
        target_day = target['start_day']
        window_start = target_day - previous_day
        
        # Get history (staypoints before target, within window)
        history = user_df[
            (user_df['start_day'] >= window_start) & 
            (user_df.index < target_idx)  # Before target in sequence
        ]
        
        if len(history) < min_length:
            continue
        
        sequence = {
            'X': history['location_id'].tolist(),
            'user_X': [user_id] * len(history),
            'weekday_X': history['weekday'].tolist(),
            'start_min_X': history['start_min'].tolist(),
            'dur_X': history['duration'].clip(upper=max_duration).tolist(),
            'diff': (target_day - history['start_day']).tolist(),
            'Y': target['location_id']
        }
        
        sequences.append(sequence)
    
    return sequences
```

---

## Edge Cases and Handling

### Case 1: Insufficient History

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: NOT ENOUGH HISTORY                             │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: User's first few staypoints have no prior history
───────────────────────────────────────────────────────────

    Day:    0    1    2    3    4    5    6    7
            │    │    │    │    │    │    │    │
    SP:     A    B    C    D    E    F    G    H
            │                                  │
            ▼                                  ▼
        Target A                           Target H
        History: []                        History: [A,B,C,D,E,F,G]
        Length: 0                          Length: 7
        SKIP! < min_length(3)              VALID!

Handling:
─────────
    Target A: 0 history points → SKIPPED
    Target B: 1 history point  → SKIPPED (< 3)
    Target C: 2 history points → SKIPPED (< 3)
    Target D: 3 history points → VALID (first sequence created)
    
    Early staypoints never become targets, but ARE used as history!
```

### Case 2: Gap in Data

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: DATA GAP                                       │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: User has gap in data (vacation, phone issues, etc.)
─────────────────────────────────────────────────────────────

    Day:    0    1    2    3    4    5    6    7    ...   20   21   22
            │    │    │    │    │    │    │    │          │    │    │
    SP:     A    BC   D         E                         F    G    H
                           └────────────────────────────┘
                                   DATA GAP (15 days)

Target H (Day 22) with previous_day=7:
──────────────────────────────────────

    Window: Days 15-21
    History in window: [F, G] (only 2 staypoints)
    
    If min_length=3: SKIPPED (insufficient history)
    If min_length=2: VALID sequence created

Handling strategy:
──────────────────
    
    1. Default min_length=3 filters out gap-affected targets
    2. Historical staypoints before gap are NOT used (outside window)
    3. Fresh history begins after gap
```

### Case 3: Multiple Staypoints Same Day

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: SAME DAY STAYPOINTS                            │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: User has many staypoints on same day
──────────────────────────────────────────────

    Day 5: User visits Home → Work → Restaurant → Gym → Home
    
    Staypoints:
        SP1: Day 5, 07:00, Home
        SP2: Day 5, 08:30, Work  
        SP3: Day 5, 12:30, Restaurant
        SP4: Day 5, 14:00, Gym
        SP5: Day 5, 18:00, Home

For Target SP5 (Day 5 evening):
───────────────────────────────

    Window: Days 0-4 (previous days) + Day 5 (same day, before target)
    
    History includes:
    • All staypoints from Days 0-4
    • SP1, SP2, SP3, SP4 from Day 5 (same day, earlier time)
    
    Key point: Same-day earlier visits ARE included!
    
    diff calculation:
    • SP1, SP2, SP3, SP4: diff = 5 - 5 = 0 (same day)
    • Earlier days: diff = 5 - N > 0

This allows model to learn:
    "User was at Gym (SP4), where will they go next?" → Home (SP5)
```

### Case 4: Unknown Locations in Test Set

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: UNKNOWN LOCATIONS                              │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: Test set contains location not seen in training
─────────────────────────────────────────────────────────

    Train locations: {0, 1, 2, 3, 4, 5}
    Test target location: 7 (new location!)

Handling (during encoding):
───────────────────────────

    1. OrdinalEncoder fit on TRAIN data only
    2. Test locations not in encoder → mapped to UNKNOWN (1)
    
    Before encoding:
        Train X: [2, 3, 4, 5, 2]
        Test X:  [2, 3, 7, 5, 2]  # Location 7 is new!
        
    After encoding (+2 offset):
        Train X: [4, 5, 6, 7, 4]   # Normal encoding
        Test X:  [4, 5, 1, 7, 4]   # Location 7 → Unknown (1)

Location ID scheme reminder:
    0 = Padding (for variable-length sequences)
    1 = Unknown location
    2+ = Known locations from training
```

---

## Complete Code Walkthrough

### Full Implementation

```python
"""
Sequence Generation - Complete Implementation
From diy_2_interim_to_processed.py (simplified for clarity)
"""

import pickle
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

def generate_sequences_for_split(df, split_name, previous_day, min_length, max_duration):
    """
    Generate sequences for a specific data split.
    
    Args:
        df: DataFrame containing staypoints for this split
        split_name: 'train', 'validation', or 'test'
        previous_day: Days of history to include
        min_length: Minimum number of historical staypoints
        max_duration: Maximum duration value
    
    Returns:
        List of sequence dictionaries
    """
    print(f"  Processing {split_name} sequences...")
    
    # Group by user
    grouped = df.groupby('user_id')
    
    # Process each user in parallel
    results = Parallel(n_jobs=-1)(
        delayed(process_single_user)(
            user_df=user_df,
            previous_day=previous_day,
            min_length=min_length,
            max_duration=max_duration
        )
        for user_id, user_df in tqdm(grouped, desc=f"    {split_name}")
    )
    
    # Flatten results
    sequences = []
    for user_seqs in results:
        sequences.extend(user_seqs)
    
    print(f"  Generated {len(sequences)} {split_name} sequences")
    return sequences


def process_single_user(user_df, previous_day, min_length, max_duration):
    """
    Generate sequences for one user.
    
    This function is called in parallel for each user.
    """
    sequences = []
    
    # Sort by time (critical for correct sequence order)
    user_df = user_df.sort_values('started_at').reset_index(drop=True)
    
    # Get user ID (will be same for all sequences from this user)
    user_id = user_df['user_id'].iloc[0]
    
    # Iterate through each staypoint as potential target
    for target_idx in range(len(user_df)):
        
        # Get target information
        target = user_df.iloc[target_idx]
        target_day = target['start_day']
        target_location = target['location_id']
        
        # Define history window
        window_start = target_day - previous_day
        
        # Get historical staypoints
        # 1. Within time window (start_day >= window_start)
        # 2. Before target day OR same day but earlier in sequence
        history_mask = (
            (user_df['start_day'] >= window_start) &
            (user_df.index < target_idx)  # Must be before in sequence
        )
        history = user_df[history_mask]
        
        # Skip if insufficient history
        if len(history) < min_length:
            continue
        
        # Calculate diff (days ago) for each historical staypoint
        diff_values = (target_day - history['start_day']).tolist()
        
        # Create sequence dictionary
        sequence = {
            'X': history['location_id'].tolist(),
            'user_X': [user_id] * len(history),
            'weekday_X': history['weekday'].tolist(),
            'start_min_X': history['start_min'].tolist(),
            'dur_X': history['duration'].clip(upper=max_duration).tolist(),
            'diff': diff_values,
            'Y': target_location
        }
        
        sequences.append(sequence)
    
    return sequences


def save_sequences(sequences, output_path, metadata=None):
    """
    Save sequences to pickle file.
    
    Args:
        sequences: List of sequence dictionaries
        output_path: Path to save pickle file
        metadata: Optional metadata to include
    """
    # Optionally include metadata in saved file
    if metadata:
        output_data = {
            'sequences': sequences,
            'metadata': metadata
        }
    else:
        output_data = sequences
    
    with open(output_path, 'wb') as f:
        pickle.dump(output_data, f)
    
    print(f"✓ Saved {len(sequences)} sequences to: {output_path}")


# Example usage
if __name__ == "__main__":
    # Load data
    df = pd.read_csv("intermediate.csv")
    
    # Split into train/val/test
    train_df = df[df['split'] == 'train']
    val_df = df[df['split'] == 'validation']
    test_df = df[df['split'] == 'test']
    
    # Configuration
    config = {
        'previous_day': 7,
        'min_sequence_length': 3,
        'max_duration': 2880
    }
    
    # Generate sequences for each split
    train_seqs = generate_sequences_for_split(
        train_df, 'train', 
        config['previous_day'],
        config['min_sequence_length'],
        config['max_duration']
    )
    
    val_seqs = generate_sequences_for_split(
        val_df, 'validation',
        config['previous_day'],
        config['min_sequence_length'],
        config['max_duration']
    )
    
    test_seqs = generate_sequences_for_split(
        test_df, 'test',
        config['previous_day'],
        config['min_sequence_length'],
        config['max_duration']
    )
    
    # Save
    save_sequences(train_seqs, "train.pk")
    save_sequences(val_seqs, "validation.pk")
    save_sequences(test_seqs, "test.pk")
```

---

## Output Format

### Pickle File Structure

```python
# Loading and inspecting sequences
import pickle

with open("diy_eps50_prev7_train.pk", "rb") as f:
    train_data = pickle.load(f)

print(f"Number of sequences: {len(train_data)}")
# Output: Number of sequences: 65234

# Inspect first sequence
seq = train_data[0]
print("Sequence keys:", seq.keys())
# Output: dict_keys(['X', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff', 'Y'])

print("\nExample sequence:")
for key, value in seq.items():
    if isinstance(value, list):
        print(f"  {key}: {value[:5]}... (length: {len(value)})")
    else:
        print(f"  {key}: {value}")

# Output:
# Example sequence:
#   X: [44, 17, 44, 10, 44]... (length: 23)
#   user_X: [1, 1, 1, 1, 1]... (length: 23)
#   weekday_X: [6, 0, 1, 2, 3]... (length: 23)
#   start_min_X: [420, 510, 450, 540, 480]... (length: 23)
#   dur_X: [720.0, 540.0, 660.0, 480.0, 720.0]... (length: 23)
#   diff: [7, 7, 6, 5, 4]... (length: 23)
#   Y: 17
```

### Statistics Example

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TYPICAL SEQUENCE STATISTICS                               │
└─────────────────────────────────────────────────────────────────────────────┘

Dataset: DIY with previous_day=7, min_length=3

Split      │ Sequences │ Avg Length │ Min Length │ Max Length
───────────┼───────────┼────────────┼────────────┼────────────
Train      │   65,234  │    18.5    │     3      │    87
Validation │    8,123  │    17.2    │     3      │    72
Test       │    8,234  │    16.8    │     3      │    68

Sequence Length Distribution (Train):
────────────────────────────────────

    Length 3-5:   ████████████ 15%
    Length 6-10:  ████████████████████████ 28%
    Length 11-20: ████████████████████████████████████ 35%
    Length 21-30: ████████████████ 15%
    Length 31+:   ████████ 7%

Most sequences have 10-20 historical staypoints
```

---

## Summary

### Key Points

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE GENERATION SUMMARY                               │
└─────────────────────────────────────────────────────────────────────────────┘

1. ONE SEQUENCE PER TARGET STAYPOINT
   Every staypoint (with enough history) becomes one training sample
   
2. SLIDING WINDOW FOR HISTORY
   History = staypoints within [target_day - previous_day, target_day)
   
3. MINIMUM LENGTH FILTER
   Sequences with < min_length history are skipped
   
4. PARALLEL PROCESSING
   Users processed independently → significant speedup
   
5. TEMPORAL FEATURES INCLUDED
   weekday, start_min, duration, diff for each historical visit
   
6. OUTPUT: PICKLE FILES
   Binary format, fast to load, preserves exact data types

Configuration defaults:
    previous_day: 7 (one week of history)
    min_sequence_length: 3 (at least 3 historical visits)
    max_duration: 2880 (cap at 48 hours)
```

### Quick Reference Code

```python
# Minimum viable sequence generation
def create_sequence(user_df, target_idx, prev_day, min_len, max_dur):
    target = user_df.iloc[target_idx]
    target_day = target['start_day']
    
    history = user_df[
        (user_df['start_day'] >= target_day - prev_day) &
        (user_df.index < target_idx)
    ]
    
    if len(history) < min_len:
        return None
    
    return {
        'X': history['location_id'].tolist(),
        'user_X': [user_df['user_id'].iloc[0]] * len(history),
        'weekday_X': history['weekday'].tolist(),
        'start_min_X': history['start_min'].tolist(),
        'dur_X': history['duration'].clip(upper=max_dur).tolist(),
        'diff': (target_day - history['start_day']).tolist(),
        'Y': target['location_id']
    }
```
