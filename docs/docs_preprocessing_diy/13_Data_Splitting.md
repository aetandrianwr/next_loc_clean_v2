# Data Splitting Strategy Deep Dive

## 📋 Table of Contents
1. [Overview](#overview)
2. [Why Temporal Splitting?](#why-temporal-splitting)
3. [Per-User Temporal Split](#per-user-temporal-split)
4. [Split Ratios](#split-ratios)
5. [Implementation Details](#implementation-details)
6. [Edge Cases and Handling](#edge-cases-and-handling)
7. [Common Pitfalls to Avoid](#common-pitfalls-to-avoid)
8. [Validation and Verification](#validation-and-verification)

---

## Overview

Data splitting is a critical step in machine learning that separates data into training, validation, and test sets. For mobility prediction, **temporal splitting** is essential to prevent data leakage.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING OVERVIEW                                   │
└─────────────────────────────────────────────────────────────────────────────┘

Input: All staypoints for a user
─────────────────────────────────

    Day: 0    10    20    30    40    50    60    70    80    90    100
         │     │     │     │     │     │     │     │     │     │     │
         ████████████████████████████████████████████████████████████
         │                                                           │
    First staypoint                                          Last staypoint

Temporal Split (80/10/10):
──────────────────────────

    Day: 0                                           80       90       100
         │                                           │        │        │
         ├───────────────────────────────────────────┼────────┼────────┤
         │           TRAIN (80%)                     │ VAL    │ TEST   │
         │           Days 0-80                       │ 10%    │ 10%    │
         │                                           │ 80-90  │ 90-100 │
         └───────────────────────────────────────────┴────────┴────────┘

Key Principle: NEVER use future data to predict the past!
```

---

## Why Temporal Splitting?

### The Problem with Random Splitting

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RANDOM SPLITTING: DATA LEAKAGE                            │
└─────────────────────────────────────────────────────────────────────────────┘

WRONG: Random Split
───────────────────

    User's staypoints (chronological order):
    Day: 0    1    2    3    4    5    6    7    8    9    10
         │    │    │    │    │    │    │    │    │    │    │
         A    B    C    D    E    F    G    H    I    J    K

    Random shuffle for splitting:
    Train: [A, D, E, H, J, K]     (random selection)
    Val:   [B, G]
    Test:  [C, F, I]

    Problem: 
    ─────────────────────────────────────────────────────────
    
    Test sequence for predicting C (Day 2):
    History could include [D, E, H, J, K] from FUTURE days!
    
    The model learns: "After seeing D (Day 3), user goes to C (Day 2)"
    This is IMPOSSIBLE in real-world deployment!
    
    Result: Artificially inflated accuracy, model fails in production.


CORRECT: Temporal Split
───────────────────────

    Same staypoints, temporal split:
    Train: [A, B, C, D, E, F, G, H]   (Days 0-8)
    Val:   [I]                        (Day 9)
    Test:  [J, K]                     (Days 10-11)

    Test prediction for K (Day 10):
    History: [A, B, C, D, E, F, G, H, I, J] - all from PAST
    
    This matches real-world deployment!
```

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RANDOM vs TEMPORAL SPLITTING                              │
└─────────────────────────────────────────────────────────────────────────────┘

Timeline:
─────────

    Day: 0    5    10   15   20   25   30   35   40   45   50
         │────│────│────│────│────│────│────│────│────│────│
         ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼    ▼
         *    *    *    *    *    *    *    *    *    *    *
         │    │    │    │    │    │    │    │    │    │    │
         SP1  SP2  SP3  SP4  SP5  SP6  SP7  SP8  SP9  SP10 SP11


RANDOM SPLIT (WRONG):
─────────────────────

    Train: SP1, SP3, SP5, SP7, SP9, SP11
    Val:   SP2, SP6, SP10
    Test:  SP4, SP8
    
    Day: 0    5    10   15   20   25   30   35   40   45   50
         │    │    │    │    │    │    │    │    │    │    │
         T    V    T    X    T    V    T    X    T    V    T
         │              │              │
         └──────────────┴──────────────┘
         Training data from AFTER test data = DATA LEAKAGE


TEMPORAL SPLIT (CORRECT):
─────────────────────────

    Train: SP1, SP2, SP3, SP4, SP5, SP6, SP7, SP8
    Val:   SP9
    Test:  SP10, SP11
    
    Day: 0    5    10   15   20   25   30   35   40   45   50
         │    │    │    │    │    │    │    │    │    │    │
         T    T    T    T    T    T    T    T    V    X    X
         │────────────────────────────────│────│─────────│
                   TRAIN (80%)             VAL   TEST
                                          (10%) (10%)
         
         All training data is BEFORE validation and test!
```

---

## Per-User Temporal Split

### Why Per-User?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PER-USER vs GLOBAL SPLITTING                              │
└─────────────────────────────────────────────────────────────────────────────┘

Challenge: Users have different tracking periods
─────────────────────────────────────────────────

    User A: Tracked from Jan 1 to Mar 31 (90 days)
    User B: Tracked from Feb 15 to May 15 (90 days)
    User C: Tracked from Mar 1 to May 30 (91 days)

    Global calendar-based split (PROBLEMATIC):
    ──────────────────────────────────────────
    
    Train: Jan 1 - Apr 15
    Val:   Apr 16 - Apr 30
    Test:  May 1 - May 31
    
    Results:
    • User A: 100% train, 0% val, 0% test (missing from val/test!)
    • User B: 60% train, 20% val, 20% test (reasonable)
    • User C: 50% train, 30% val, 20% test (imbalanced)
    
    Problem: User A never appears in validation or test!


Solution: Per-User Temporal Split
─────────────────────────────────

    Each user is split based on THEIR OWN timeline:
    
    User A (Days 0-89):
    • Train: Days 0-71 (80%)
    • Val:   Days 72-80 (10%)
    • Test:  Days 81-89 (10%)
    
    User B (Days 0-89):
    • Train: Days 0-71 (80%)
    • Val:   Days 72-80 (10%)
    • Test:  Days 81-89 (10%)
    
    User C (Days 0-90):
    • Train: Days 0-72 (80%)
    • Val:   Days 73-81 (10%)
    • Test:  Days 82-90 (10%)
    
    Every user has data in all three splits!
```

### Visualization of Per-User Split

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PER-USER TEMPORAL SPLIT VISUALIZATION                     │
└─────────────────────────────────────────────────────────────────────────────┘

Calendar Timeline:
──────────────────

    Jan 1        Feb 1        Mar 1        Apr 1        May 1        Jun 1
    │            │            │            │            │            │
    ▼            ▼            ▼            ▼            ▼            ▼


User A (starts Jan 1, ends Mar 31):
───────────────────────────────────

    Jan 1                                         Mar 31
    │──────────────────────────────────────────────│
    │           User A's 90 days                   │
    │                                              │
    │        TRAIN (80%)      │ VAL │    TEST     │
    │         72 days         │ 9d  │    9d       │
    └──────────────────────────┴─────┴────────────┘


User B (starts Feb 15, ends May 15):
────────────────────────────────────

              Feb 15                                          May 15
              │──────────────────────────────────────────────────│
              │           User B's 90 days                       │
              │                                                  │
              │        TRAIN (80%)      │ VAL │    TEST         │
              │         72 days         │ 9d  │    9d           │
              └──────────────────────────┴─────┴────────────────┘


User C (starts Mar 1, ends May 30):
───────────────────────────────────

                      Mar 1                                          May 30
                      │──────────────────────────────────────────────────│
                      │           User C's 91 days                       │
                      │                                                  │
                      │        TRAIN (80%)      │ VAL │    TEST         │
                      │         73 days         │ 9d  │    9d           │
                      └──────────────────────────┴─────┴────────────────┘


Combined Dataset:
─────────────────

    Each split contains data from ALL users:
    
    TRAIN set: User A (Jan-Mar) + User B (Feb-Apr) + User C (Mar-May)
    VAL set:   User A (Mar) + User B (Apr) + User C (May)
    TEST set:  User A (Mar) + User B (May) + User C (May)
```

---

## Split Ratios

### Default Configuration

```yaml
# Default split ratios
train_ratio: 0.8   # 80% for training
val_ratio: 0.1     # 10% for validation
test_ratio: 0.1    # 10% for testing
```

### Why 80/10/10?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPLIT RATIO JUSTIFICATION                                 │
└─────────────────────────────────────────────────────────────────────────────┘

Training (80%):
───────────────
    • Needs most data for learning patterns
    • Mobility patterns are complex and varied
    • Location vocabulary can be large (1000s)
    • Temporal patterns need enough examples

    With 100 days of data:
    • Train: 80 days
    • Approximately 8-10 weeks of patterns
    • Covers multiple weekly cycles


Validation (10%):
─────────────────
    • Used for hyperparameter tuning
    • Early stopping to prevent overfitting
    • Model selection (compare architectures)
    
    With 100 days of data:
    • Val: 10 days
    • Approximately 1-1.5 weeks
    • Enough for reliable validation metrics


Test (10%):
───────────
    • Final evaluation only
    • NEVER used for model decisions
    • Represents "future" predictions
    
    With 100 days of data:
    • Test: 10 days
    • Simulates real-world deployment
    • Evaluates generalization


Alternative Ratios:
───────────────────

    Ratio    │ Train │ Val │ Test │ Use Case
    ─────────┼───────┼─────┼──────┼──────────────────────────────────
    70/15/15 │  70%  │ 15% │ 15%  │ More robust evaluation
    80/10/10 │  80%  │ 10% │ 10%  │ Standard (default)
    85/5/10  │  85%  │  5% │ 10%  │ More training, less tuning
    90/5/5   │  90%  │  5% │  5%  │ Maximum training (less evaluation)
```

---

## Implementation Details

### Split Calculation

```python
def calculate_split_boundaries(user_df, train_ratio=0.8, val_ratio=0.1):
    """
    Calculate temporal split boundaries for a user.
    
    Args:
        user_df: DataFrame with user's staypoints
        train_ratio: Proportion for training (default 0.8)
        val_ratio: Proportion for validation (default 0.1)
    
    Returns:
        train_cutoff: Day number marking end of training
        val_cutoff: Day number marking end of validation
    """
    # Get user's day range
    min_day = user_df['start_day'].min()  # Should be 0
    max_day = user_df['start_day'].max()
    
    # Calculate cutoff days
    total_days = max_day - min_day + 1
    
    train_cutoff = min_day + int(total_days * train_ratio)
    val_cutoff = min_day + int(total_days * (train_ratio + val_ratio))
    
    return train_cutoff, val_cutoff


def assign_split(user_df, train_cutoff, val_cutoff):
    """
    Assign each staypoint to a split based on start_day.
    
    Args:
        user_df: DataFrame with user's staypoints
        train_cutoff: End day for training
        val_cutoff: End day for validation
    
    Returns:
        DataFrame with 'split' column
    """
    def get_split(start_day):
        if start_day <= train_cutoff:
            return 'train'
        elif start_day <= val_cutoff:
            return 'validation'
        else:
            return 'test'
    
    user_df['split'] = user_df['start_day'].apply(get_split)
    return user_df
```

### Complete Implementation

```python
"""
Temporal Data Splitting Implementation
From diy_2_interim_to_processed.py
"""

import pandas as pd
import numpy as np

def temporal_split_per_user(df, train_ratio=0.8, val_ratio=0.1):
    """
    Apply temporal split to each user's data.
    
    Args:
        df: DataFrame with all users' staypoints
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
    
    Returns:
        train_df, val_df, test_df: Split DataFrames
    """
    train_dfs = []
    val_dfs = []
    test_dfs = []
    
    # Process each user separately
    for user_id, user_df in df.groupby('user_id'):
        # Sort by time
        user_df = user_df.sort_values('started_at').reset_index(drop=True)
        
        # Get day range for this user
        max_day = user_df['start_day'].max()
        
        # Calculate cutoffs
        train_cutoff = int(max_day * train_ratio)
        val_cutoff = int(max_day * (train_ratio + val_ratio))
        
        # Split
        train = user_df[user_df['start_day'] <= train_cutoff]
        val = user_df[(user_df['start_day'] > train_cutoff) & 
                      (user_df['start_day'] <= val_cutoff)]
        test = user_df[user_df['start_day'] > val_cutoff]
        
        train_dfs.append(train)
        val_dfs.append(val)
        test_dfs.append(test)
    
    # Combine all users
    train_df = pd.concat(train_dfs, ignore_index=True)
    val_df = pd.concat(val_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    return train_df, val_df, test_df


def verify_split(train_df, val_df, test_df):
    """
    Verify that temporal ordering is preserved.
    """
    # Check each user
    all_users = set(train_df['user_id'].unique())
    
    for user_id in all_users:
        train_max = train_df[train_df['user_id'] == user_id]['start_day'].max()
        
        # Val should be after train
        if user_id in val_df['user_id'].values:
            val_min = val_df[val_df['user_id'] == user_id]['start_day'].min()
            assert val_min > train_max, f"User {user_id}: Val overlaps with train!"
        
        # Test should be after val (or train if no val)
        if user_id in test_df['user_id'].values:
            test_min = test_df[test_df['user_id'] == user_id]['start_day'].min()
            if user_id in val_df['user_id'].values:
                val_max = val_df[val_df['user_id'] == user_id]['start_day'].max()
                assert test_min > val_max, f"User {user_id}: Test overlaps with val!"
    
    print("✓ Temporal ordering verified for all users")
```

### Example Walkthrough

```
User: user_001 (Example Split Calculation)
═══════════════════════════════════════════════════════════════════════════════

User's data:
    Total days: 100 (start_day: 0 to 99)
    Total staypoints: 250

Step 1: Calculate cutoffs
─────────────────────────

    train_ratio = 0.8
    val_ratio = 0.1
    test_ratio = 0.1 (implied)
    
    max_day = 99
    
    train_cutoff = int(99 * 0.8) = 79
    val_cutoff = int(99 * 0.9) = 89

Step 2: Assign splits
─────────────────────

    Train: start_day <= 79  → Days 0-79  (80 days)
    Val:   79 < start_day <= 89  → Days 80-89 (10 days)
    Test:  start_day > 89  → Days 90-99 (10 days)

Step 3: Count staypoints per split
──────────────────────────────────

    Train: 200 staypoints (80%)
    Val:   25 staypoints (10%)
    Test:  25 staypoints (10%)

Visualization:
──────────────

    Day:  0    10   20   30   40   50   60   70   80   90   100
          │────│────│────│────│────│────│────│────│────│────│
          
    Split:│◄───────────── TRAIN ─────────────▶│◄VAL ▶│◄TEST▶│
          │         Days 0-79 (200 SP)        │ 25 SP│ 25 SP│
          └───────────────────────────────────┴──────┴──────┘
                                              │      │
                                         train_cutoff│
                                              79     val_cutoff
                                                     89
```

---

## Edge Cases and Handling

### Case 1: User Missing from Split

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: MISSING FROM SPLIT                             │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: User has very few days, split creates empty partitions
───────────────────────────────────────────────────────────────

    User with 5 days total:
    
    train_cutoff = int(4 * 0.8) = 3
    val_cutoff = int(4 * 0.9) = 3  # Same as train!
    
    Result:
    • Train: Days 0-3 (4 days)
    • Val: Days 4-3 (EMPTY - val_cutoff == train_cutoff)
    • Test: Days 4-4 (1 day only)

Handling:
─────────

    Option 1: Skip users with insufficient days (Quality filtering)
    Option 2: Adjust cutoffs to ensure non-empty splits
    Option 3: Require minimum days per split
    
    In this pipeline:
    • Quality filtering removes short-period users
    • day_filter >= 60 ensures minimum 60 days
    • With 60 days: train=48, val=6, test=6 days minimum
```

### Case 2: All Staypoints on Same Day

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: SAME-DAY STAYPOINTS                            │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: User has multiple staypoints on boundary day
───────────────────────────────────────────────────────

    Day 79 (train_cutoff): 5 staypoints
    Day 80 (first val day): 3 staypoints
    
    With <= operator:
    • All 5 staypoints on Day 79 go to TRAIN
    • All 3 staypoints on Day 80 go to VAL
    
    Correct behavior: Same-day staypoints stay together in same split

Important Note:
───────────────
    
    start_day is INTEGER (day number), not timestamp
    All staypoints on same day have same start_day
    No staypoint spans split boundaries
```

### Case 3: Uneven Distribution

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EDGE CASE: UNEVEN STAYPOINT DISTRIBUTION                  │
└─────────────────────────────────────────────────────────────────────────────┘

Scenario: User very active at start, sparse at end
──────────────────────────────────────────────────

    Day: 0-20: 150 staypoints (busy period)
    Day: 21-80: 40 staypoints (normal activity)
    Day: 81-100: 10 staypoints (vacation/sick)
    
    Total: 200 staypoints over 100 days
    
    Split by DAYS (temporal):
    • Train (Days 0-79): 190 staypoints (95%)
    • Val (Days 80-89): 5 staypoints (2.5%)
    • Test (Days 90-99): 5 staypoints (2.5%)
    
    NOT 80/10/10 by staypoint count, but correct by TIME!

Why this is correct:
────────────────────

    The goal is to predict FUTURE locations.
    
    In deployment:
    • Model trained on past data (80% of time)
    • Deployed to predict future (remaining time)
    
    If user becomes less active, predictions are still needed.
    
    Alternative (split by staypoint count) would:
    • Use Day 95 data for training
    • Try to predict Day 80 in test
    • DATA LEAKAGE!
```

---

## Common Pitfalls to Avoid

### Pitfall 1: Shuffling Before Split

```python
# WRONG: Shuffling destroys temporal order
df = df.sample(frac=1).reset_index(drop=True)  # NEVER DO THIS!
train_df = df.iloc[:int(len(df)*0.8)]
val_df = df.iloc[int(len(df)*0.8):int(len(df)*0.9)]
test_df = df.iloc[int(len(df)*0.9):]

# CORRECT: Sort by time, then split
df = df.sort_values('started_at')
# ... then apply temporal split
```

### Pitfall 2: Global Calendar Split

```python
# WRONG: Using global dates
global_train_end = '2023-03-31'
global_val_end = '2023-04-15'

train_df = df[df['started_at'] <= global_train_end]
val_df = df[(df['started_at'] > global_train_end) & 
            (df['started_at'] <= global_val_end)]
test_df = df[df['started_at'] > global_val_end]

# PROBLEM: Users who stopped tracking before April are missing from test!

# CORRECT: Per-user temporal split
for user_id, user_df in df.groupby('user_id'):
    user_train, user_val, user_test = split_user_temporally(user_df)
```

### Pitfall 3: Using Future Context in Sequences

```python
# WRONG: Including future staypoints in history
def create_sequence_wrong(user_df, target_idx, window_days):
    target_day = user_df.iloc[target_idx]['start_day']
    
    # This includes ALL staypoints in window, even after target!
    history = user_df[(user_df['start_day'] >= target_day - window_days) &
                      (user_df['start_day'] <= target_day)]  # WRONG: includes target day
    
# CORRECT: Only include staypoints BEFORE target
def create_sequence_correct(user_df, target_idx, window_days):
    target_day = user_df.iloc[target_idx]['start_day']
    
    # Only staypoints before target in sequence
    history = user_df[(user_df['start_day'] >= target_day - window_days) &
                      (user_df.index < target_idx)]  # Correct: before target
```

---

## Validation and Verification

### Verification Checklist

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SPLIT VERIFICATION CHECKLIST                              │
└─────────────────────────────────────────────────────────────────────────────┘

☐ Temporal Ordering
    For each user:
    max(train_day) < min(val_day) < min(test_day)

☐ No Data Leakage
    No sequence in train uses history from val/test
    No sequence in val uses history from test

☐ All Users in All Splits
    Each valid user appears in train, val, AND test

☐ Reasonable Split Sizes
    Train: 70-90% of data
    Val: 5-15% of data
    Test: 5-15% of data

☐ Per-User Split (not global)
    Each user's data is split based on THEIR timeline
```

### Verification Code

```python
def verify_temporal_split(df, train_df, val_df, test_df):
    """
    Comprehensive verification of temporal split.
    """
    errors = []
    
    # 1. Check temporal ordering per user
    print("Checking temporal ordering...")
    for user_id in df['user_id'].unique():
        train_user = train_df[train_df['user_id'] == user_id]
        val_user = val_df[val_df['user_id'] == user_id]
        test_user = test_df[test_df['user_id'] == user_id]
        
        if len(train_user) > 0 and len(val_user) > 0:
            if train_user['start_day'].max() >= val_user['start_day'].min():
                errors.append(f"User {user_id}: Train overlaps with val")
        
        if len(val_user) > 0 and len(test_user) > 0:
            if val_user['start_day'].max() >= test_user['start_day'].min():
                errors.append(f"User {user_id}: Val overlaps with test")
    
    # 2. Check all users present in all splits
    print("Checking user coverage...")
    train_users = set(train_df['user_id'].unique())
    val_users = set(val_df['user_id'].unique())
    test_users = set(test_df['user_id'].unique())
    
    missing_val = train_users - val_users
    missing_test = train_users - test_users
    
    if missing_val:
        errors.append(f"{len(missing_val)} users missing from validation")
    if missing_test:
        errors.append(f"{len(missing_test)} users missing from test")
    
    # 3. Check split sizes
    print("Checking split sizes...")
    total = len(train_df) + len(val_df) + len(test_df)
    train_pct = len(train_df) / total * 100
    val_pct = len(val_df) / total * 100
    test_pct = len(test_df) / total * 100
    
    print(f"  Train: {train_pct:.1f}%")
    print(f"  Val: {val_pct:.1f}%")
    print(f"  Test: {test_pct:.1f}%")
    
    # Report results
    if errors:
        print("\n⚠️ ERRORS FOUND:")
        for error in errors:
            print(f"  - {error}")
        return False
    else:
        print("\n✓ All verification checks passed!")
        return True


# Usage example
if verify_temporal_split(df, train_df, val_df, test_df):
    print("Safe to proceed with model training")
else:
    print("Fix splitting issues before training!")
```

---

## Summary

### Key Principles

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DATA SPLITTING SUMMARY                                    │
└─────────────────────────────────────────────────────────────────────────────┘

1. TEMPORAL SPLIT, NOT RANDOM
   ─────────────────────────
   • Past data for training, future data for testing
   • Matches real-world deployment scenario
   • Prevents data leakage

2. PER-USER SPLITTING
   ──────────────────
   • Each user's data split based on their timeline
   • Ensures all users in all splits
   • Accounts for different tracking periods

3. DEFAULT RATIOS: 80/10/10
   ─────────────────────────
   • Train: 80% (majority for learning)
   • Validation: 10% (hyperparameter tuning)
   • Test: 10% (final evaluation)

4. SPLIT BY DAYS, NOT STAYPOINTS
   ──────────────────────────────
   • Temporal ordering preserved
   • May result in uneven staypoint counts
   • This is correct and expected

5. VERIFY BEFORE TRAINING
   ───────────────────────
   • Check temporal ordering
   • Check user coverage
   • Check split sizes
```

### Quick Reference Code

```python
# Minimal temporal split implementation
def temporal_split(user_df, train_ratio=0.8, val_ratio=0.1):
    max_day = user_df['start_day'].max()
    train_cutoff = int(max_day * train_ratio)
    val_cutoff = int(max_day * (train_ratio + val_ratio))
    
    train = user_df[user_df['start_day'] <= train_cutoff]
    val = user_df[(user_df['start_day'] > train_cutoff) & 
                  (user_df['start_day'] <= val_cutoff)]
    test = user_df[user_df['start_day'] > val_cutoff]
    
    return train, val, test
```
