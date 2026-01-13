# Script 2: Interim to Processed - Complete Line-by-Line Guide

## Table of Contents
1. [Overview](#overview)
2. [File Header and Imports](#file-header-and-imports)
3. [Function: split_dataset](#function-split_dataset)
4. [Function: _get_split_days_user](#function-_get_split_days_user)
5. [Function: get_valid_sequence](#function-get_valid_sequence)
6. [Function: get_valid_sequence_per_user](#function-get_valid_sequence_per_user)
7. [Function: generate_sequences](#function-generate_sequences)
8. [Function: process_for_previous_day](#function-process_for_previous_day)
9. [Main Function: process_intermediate_to_final](#main-function-process_intermediate_to_final)
10. [Complete Data Flow Example](#complete-data-flow-example)
11. [Output File Formats](#output-file-formats)

---

## Overview

**Script**: `preprocessing/geolife_2_interim_to_processed.py`

**Purpose**: Transform intermediate staypoint data into machine learning-ready sequence files (.pk format).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SCRIPT 2: INTERIM TO PROCESSED                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT                          PROCESS                          OUTPUT          │
│  ─────                          ───────                          ──────          │
│                                                                                  │
│  data/geolife_eps20/            5 Processing Steps               data/geolife_   │
│  interim/                                                        eps20/processed/│
│  └── intermediate               [1] Split train/val/test                         │
│       _eps20.csv                [2] Encode location IDs          ├── *_train.pk  │
│                                 [3] Filter valid sequences       ├── *_val.pk    │
│                                 [4] Filter valid users           ├── *_test.pk   │
│                                 [5] Generate sequences           └── *_metadata  │
│                                                                       .json      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Header and Imports

### Docstring (Lines 1-17)

```python
"""
Geolife Dataset Preprocessing - Script 2: Interim to Processed
Processes intermediate staypoint data to final sequence .pk files.

This script:
1. Loads intermediate staypoint dataset from Script 1
2. Splits data into train/val/test per user
3. Encodes location IDs
4. For each previous_day value in config:
   - Filters valid sequences based on previous_day parameter
   - Generates sequences with features (X, user_X, weekday_X, etc.)
   - Saves train/validation/test .pk files
   - Saves metadata.json

Input: data/geolife_eps{epsilon}/interim/
Output: data/geolife_eps{epsilon}/processed/
"""
```

### Import Statements (Lines 19-34)

```python
import os
import sys
import json
import pickle            # For saving .pk files
import argparse
import json
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder  # For encoding location IDs

# Set random seed
RANDOM_SEED = 42
```

**Key Library: pickle**
- Used to serialize Python objects (lists of dictionaries) to binary files
- `.pk` files are the final output format
- Can be loaded directly into PyTorch/TensorFlow dataloaders

---

## Function: split_dataset

**Lines 37-52**

Splits the dataset into train, validation, and test sets **per user** based on chronological order.

### Function Signature

```python
def split_dataset(totalData, split_ratios):
    """Split dataset into train, val and test per user."""
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `totalData` | DataFrame | All staypoints with user_id, start_day, etc. |
| `split_ratios` | dict | Contains `train`, `val`, `test` ratios |

**Returns**: Three DataFrames (train_data, vali_data, test_data)

### Line-by-Line Explanation

```python
def split_dataset(totalData, split_ratios):
    """Split dataset into train, val and test per user."""
    # Apply split logic to each user
    totalData = totalData.groupby("user_id", group_keys=False).apply(
        _get_split_days_user, split_ratios=split_ratios
    )

    # Separate by Dataset column
    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "val"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # Final cleaning - remove the Dataset column
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data
```

**Visualization**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         SPLIT DATASET FUNCTION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input: All staypoints DataFrame                                                 │
│                                                                                  │
│  Processing: For each user, split by time (chronological)                        │
│                                                                                  │
│  User 1 timeline:                                                                │
│  Day:     0        20        40        60        80        100                  │
│           │─────────│─────────│─────────│─────────│─────────│                   │
│           │         │                   │                   │                   │
│           │←───── 60% train ──────►│← 20% val ─►│← 20% test ►│                  │
│           │   Day 0-60 (train)     │ Day 60-80  │ Day 80-100 │                  │
│                                    │   (val)    │   (test)   │                  │
│                                                                                  │
│  User 2 timeline (different period):                                             │
│  Day:     0        30        60        90                                        │
│           │─────────│─────────│─────────│                                        │
│           │←───── 60% ─────►│← 20% ►│← 20% ►│                                    │
│                                                                                  │
│  Each user's data is split independently based on THEIR tracking period         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Function: _get_split_days_user

**Lines 55-65**

Helper function that assigns each staypoint to train/val/test based on the day.

```python
def _get_split_days_user(df, split_ratios):
    """Split the dataset according to the tracked day of each user."""
    maxDay = df["start_day"].max()
    train_split = maxDay * split_ratios["train"]
    validation_split = maxDay * (split_ratios["train"] + split_ratios["val"])

    df["Dataset"] = "test"  # Default to test
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & (df["start_day"] < validation_split), "Dataset"] = "val"

    return df
```

### Line-by-Line Breakdown

| Line | Code | Explanation |
|------|------|-------------|
| 57 | `maxDay = df["start_day"].max()` | Get last day of tracking for this user |
| 58 | `train_split = maxDay * 0.6` | Day threshold for train (60%) |
| 59 | `validation_split = maxDay * 0.8` | Day threshold for val (80%) |
| 61 | `df["Dataset"] = "test"` | Default assignment |
| 62 | `df.loc[... < train_split] = "train"` | Days 0 to 60% → train |
| 63 | `df.loc[... >= train_split & < val_split] = "val"` | Days 60% to 80% → val |

**Concrete Example**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SPLIT CALCULATION EXAMPLE                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User has staypoints from Day 0 to Day 100                                       │
│  maxDay = 100                                                                    │
│                                                                                  │
│  Calculations:                                                                   │
│  • train_split = 100 * 0.6 = 60                                                  │
│  • validation_split = 100 * 0.8 = 80                                             │
│                                                                                  │
│  Assignment rules:                                                               │
│  • start_day < 60           → "train"                                            │
│  • 60 ≤ start_day < 80      → "val"                                              │
│  • start_day ≥ 80           → "test" (default)                                   │
│                                                                                  │
│  Example staypoints:                                                             │
│  ┌────────────────────────────────────────────────────────────────┐             │
│  │ id │ user_id │ start_day │ location_id │ ... │ Dataset        │             │
│  │  0 │    1    │     5     │      3      │     │ train          │             │
│  │  1 │    1    │    25     │      1      │     │ train          │             │
│  │  2 │    1    │    55     │      3      │     │ train          │             │
│  │  3 │    1    │    65     │      2      │     │ val            │             │
│  │  4 │    1    │    78     │      1      │     │ val            │             │
│  │  5 │    1    │    85     │      3      │     │ test           │             │
│  │  6 │    1    │    95     │      2      │     │ test           │             │
│  └────────────────────────────────────────────────────────────────┘             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Function: get_valid_sequence

**Lines 68-89**

Identifies which staypoints can serve as valid prediction targets based on the `previous_day` requirement.

### Function Signature

```python
def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """Get valid sequence IDs based on previous_day requirement."""
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `input_df` | DataFrame | Staypoints for one split (train/val/test) |
| `previous_day` | int | Days of history required |
| `min_length` | int | Minimum staypoints in history (default: 3) |

**Returns**: List of valid staypoint IDs

### Line-by-Line Explanation

```python
def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """Get valid sequence IDs based on previous_day requirement."""
    valid_id = []
    
    for user in input_df["user_id"].unique():
        df = input_df.loc[input_df["user_id"] == user].copy().reset_index(drop=True)

        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days  # Days since first record

        for index, row in df.iterrows():
            # Exclude the first records (not enough history)
            if row["diff_day"] < previous_day:
                continue

            # Get history within previous_day window
            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
            
            # Need at least min_length staypoints in history
            if len(hist) < min_length:
                continue

            valid_id.append(row["id"])

    return valid_id
```

### Validation Logic Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VALID SEQUENCE FILTERING                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Parameters: previous_day = 7, min_length = 3                                    │
│                                                                                  │
│  User timeline (days since first tracking):                                      │
│  Day:   0    1    2    3    4    5    6    7    8    9    10   11   12          │
│         │    │    │    │    │    │    │    │    │    │    │    │    │           │
│  Visits: ●        ●    ●              ●    ●         ●    ●              ●       │
│         sp0      sp1  sp2            sp3  sp4       sp5  sp6            sp7      │
│                                                                                  │
│  Checking each staypoint as potential target:                                    │
│  ─────────────────────────────────────────────                                   │
│                                                                                  │
│  sp0 (day 0): diff_day = 0 < 7 → SKIP (not enough days passed)                  │
│  sp1 (day 2): diff_day = 2 < 7 → SKIP                                           │
│  sp2 (day 3): diff_day = 3 < 7 → SKIP                                           │
│  sp3 (day 6): diff_day = 6 < 7 → SKIP                                           │
│                                                                                  │
│  sp4 (day 7): diff_day = 7 ≥ 7 → CHECK HISTORY                                  │
│              History window: [day 0, day 7)                                      │
│              Staypoints in window: sp0, sp1, sp2, sp3 = 4 staypoints            │
│              4 ≥ 3 (min_length) → VALID ✓                                       │
│                                                                                  │
│  sp5 (day 9): diff_day = 9 ≥ 7 → CHECK HISTORY                                  │
│              History window: [day 2, day 9)                                      │
│              Staypoints in window: sp1, sp2, sp3, sp4 = 4 staypoints            │
│              4 ≥ 3 → VALID ✓                                                    │
│                                                                                  │
│  sp6 (day 10): History window [day 3, day 10)                                   │
│               Staypoints: sp2, sp3, sp4, sp5 = 4 → VALID ✓                      │
│                                                                                  │
│  sp7 (day 12): History window [day 5, day 12)                                   │
│               Staypoints: sp3, sp4, sp5, sp6 = 4 → VALID ✓                      │
│                                                                                  │
│  Result: valid_id = [sp4.id, sp5.id, sp6.id, sp7.id]                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Function: get_valid_sequence_per_user

**Lines 92-133**

Generates actual sequence dictionaries for a single user.

### Function Signature

```python
def get_valid_sequence_per_user(df, previous_day, valid_ids):
    """Get valid sequences for a single user."""
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `df` | DataFrame | Staypoints for one user |
| `previous_day` | int | Days of history to include |
| `valid_ids` | set | Set of valid staypoint IDs |

**Returns**: List of sequence dictionaries

### Line-by-Line Explanation

```python
def get_valid_sequence_per_user(df, previous_day, valid_ids):
    """Get valid sequences for a single user."""
    df = df.reset_index(drop=True)
    data_single_user = []
    
    # Get the day of tracking (relative)
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days
    
    for index, row in df.iterrows():
        # Exclude the first records that do not include enough previous_day
        if row["diff_day"] < previous_day:
            continue
        
        # Get the history records [curr-previous_day, curr)
        hist = df.iloc[:index]  # All records before current
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
        
        # Should be in the valid user ids
        if not (row["id"] in valid_ids):
            continue
        
        # Require at least 3 staypoints in sequence (history + target)
        if len(hist) < 3:
            continue
        
        # Build the sequence dictionary
        data_dict = {}
        # Get the features: location, user, weekday, start time, duration, diff to curr day
        data_dict["X"] = hist["location_id"].values              # History locations
        data_dict["user_X"] = hist["user_id"].values             # User IDs (same)
        data_dict["weekday_X"] = hist["weekday"].values          # Day of week
        data_dict["start_min_X"] = hist["start_min"].values      # Start time (minutes)
        data_dict["dur_X"] = hist["duration"].values             # Duration (minutes)
        data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values  # Days ago
        
        # The next location is the target
        data_dict["Y"] = int(row["location_id"])
        
        # Append the single sample to list
        data_single_user.append(data_dict)
    
    return data_single_user
```

### Sequence Structure Visualization

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE DICTIONARY STRUCTURE                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  For target staypoint at Day 10:                                                 │
│  History window: Day 3 to Day 9 (previous_day = 7)                               │
│                                                                                  │
│  History staypoints:                                                             │
│  ┌───────────────────────────────────────────────────────────────────┐          │
│  │ idx │ loc_id │ user_id │ weekday │ start_min │ duration │ day   │          │
│  │  0  │   3    │    1    │    2    │    540    │   480    │   3   │          │
│  │  1  │   1    │    1    │    4    │    510    │   540    │   5   │          │
│  │  2  │   3    │    1    │    5    │    600    │   300    │   6   │          │
│  │  3  │   2    │    1    │    0    │    480    │   600    │   9   │          │
│  └───────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
│  Target: Day 10, location_id = 3                                                 │
│                                                                                  │
│  Generated sequence dictionary:                                                  │
│  ─────────────────────────────────                                               │
│  {                                                                               │
│      "X": [3, 1, 3, 2],           # Location IDs (input sequence)               │
│      "user_X": [1, 1, 1, 1],      # User IDs (all same)                         │
│      "weekday_X": [2, 4, 5, 0],   # Days of week (Tue, Thu, Fri, Mon)           │
│      "start_min_X": [540, 510, 600, 480],  # Start times (9am, 8:30am, 10am, 8am)│
│      "dur_X": [480, 540, 300, 600],  # Durations (8h, 9h, 5h, 10h)              │
│      "diff": [7, 5, 4, 1],        # Days ago from target (10-3, 10-5, 10-6, 10-9)│
│      "Y": 3                       # Target location to predict                  │
│  }                                                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Feature Explanation Table

| Feature | Key | Type | Description | Example |
|---------|-----|------|-------------|---------|
| Location history | `X` | array[int] | Sequence of visited location IDs | [3, 1, 3, 2] |
| User IDs | `user_X` | array[int] | User ID for each visit (same) | [1, 1, 1, 1] |
| Weekday | `weekday_X` | array[int] | Day of week (0=Mon, 6=Sun) | [2, 4, 5, 0] |
| Start time | `start_min_X` | array[int] | Minutes from midnight | [540, 510, 600, 480] |
| Duration | `dur_X` | array[int] | Visit duration in minutes | [480, 540, 300, 600] |
| Days ago | `diff` | array[int] | Days before target | [7, 5, 4, 1] |
| Target | `Y` | int | Next location to predict | 3 |

---

## Function: generate_sequences

**Lines 136-146**

Orchestrates sequence generation for all users.

```python
def generate_sequences(data_df, valid_ids, previous_day):
    """Generate sequences for all users."""
    all_sequences = []
    valid_ids_set = set(valid_ids)  # Convert to set for O(1) lookup
    
    for user_id in tqdm(data_df["user_id"].unique(), desc="Generating sequences"):
        user_df = data_df[data_df["user_id"] == user_id].copy()
        user_sequences = get_valid_sequence_per_user(user_df, previous_day, valid_ids_set)
        all_sequences.extend(user_sequences)
    
    return all_sequences
```

**Process Flow**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GENERATE SEQUENCES FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input: train_data DataFrame, valid_ids list                                     │
│                                                                                  │
│  For each user:                                                                  │
│  ┌─────────────────────────────────────────────────────────────┐                │
│  │ User 1 ──► get_valid_sequence_per_user() ──► [seq1, seq2, seq3, ...]         │
│  │ User 2 ──► get_valid_sequence_per_user() ──► [seq7, seq8, ...]               │
│  │ User 3 ──► get_valid_sequence_per_user() ──► [seq10, seq11, seq12, ...]      │
│  │ ...                                                                          │
│  └─────────────────────────────────────────────────────────────┘                │
│                                                                                  │
│  Output: [seq1, seq2, seq3, ..., seq7, seq8, ..., seq10, seq11, seq12, ...]     │
│          All sequences combined into one list                                    │
│                                                                                  │
│  Progress bar shows: "Generating sequences: 100%|████████| 30/30 users"         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Function: process_for_previous_day

**Lines 149-295**

Main processing function for a specific `previous_day` value.

### Function Signature

```python
def process_for_previous_day(sp, split_ratios, max_duration, previous_day, epsilon, dataset_name, output_base_path):
    """Process data for a specific previous_day value."""
```

### Step-by-Step Breakdown

#### Step 0: Setup (Lines 151-162)

```python
output_name = f"{dataset_name}_eps{epsilon}_prev{previous_day}"
processed_path = os.path.join(output_base_path, "processed")
os.makedirs(processed_path, exist_ok=True)

print("\n" + "-" * 60)
print(f"Processing for previous_day = {previous_day}")
print("-" * 60)

# Truncate too long duration
sp_copy = sp.copy()
sp_copy.loc[sp_copy["duration"] > max_duration - 1, "duration"] = max_duration - 1
```

**Duration Capping**:
```
max_duration = 2880 (48 hours in minutes)

Before: duration = 5000 minutes (83 hours)
After:  duration = 2879 minutes (capped)

Why cap duration?
• Extreme outliers can skew model learning
• 48 hours is a reasonable maximum for a single visit
• Keeps feature values in a reasonable range
```

#### Step 1: Split Dataset (Lines 164-167)

```python
# 1. Split dataset
print("\n[1/5] Splitting dataset into train/val/test...")
train_data, vali_data, test_data = split_dataset(sp_copy, split_ratios)
print(f"Train: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}")
```

#### Step 2: Encode Locations (Lines 169-181)

```python
# 2. Encode locations
print("\n[2/5] Encoding location IDs...")
enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
    train_data["location_id"].values.reshape(-1, 1)
)
# Add 2 to account for unseen locations (1) and padding (0)
train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

print(f"Max location ID: {train_data['location_id'].max()}")
print(f"Unique locations in train: {train_data['location_id'].nunique()}")
```

**Encoding Visualization**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LOCATION ID ENCODING                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Why encode location IDs?                                                        │
│  • Original IDs may be sparse (e.g., 0, 5, 12, 47, 89)                          │
│  • Neural networks work better with continuous IDs (0, 1, 2, 3, 4)              │
│  • Embedding layers need to know vocabulary size                                 │
│                                                                                  │
│  OrdinalEncoder process:                                                         │
│  ──────────────────────                                                          │
│  1. Fit on TRAINING data only (to simulate real-world scenario)                  │
│  2. Transform all splits (train, val, test)                                      │
│  3. Unknown locations in val/test get value -1 (then +2 = 1)                     │
│                                                                                  │
│  Encoding scheme:                                                                │
│  ┌────────────────────────────────────────────────────┐                         │
│  │ Final ID │ Meaning                                 │                         │
│  │    0     │ Padding token (for variable length)    │                         │
│  │    1     │ Unknown location (not in training)     │                         │
│  │    2     │ First location in training set         │                         │
│  │    3     │ Second location                        │                         │
│  │   ...    │ ...                                    │                         │
│  │   N+1    │ Last location                          │                         │
│  └────────────────────────────────────────────────────┘                         │
│                                                                                  │
│  Example:                                                                        │
│  Original location_id: [47, 12, 5, 89, 0]                                       │
│  After OrdinalEncoder: [ 0,  1, 2,  3, 4]  (mapped by frequency in train)       │
│  After +2:             [ 2,  3, 4,  5, 6]  (final IDs)                          │
│                                                                                  │
│  If val/test has location 99 (not in train):                                     │
│  OrdinalEncoder returns: -1 (unknown_value)                                      │
│  After +2: 1 (maps to "unknown" token)                                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Step 3: Get Valid Sequences (Lines 183-198)

```python
# 3. Get valid sequences
print(f"\n[3/5] Filtering valid sequences (previous_day={previous_day})...")

all_ids = sp_copy[["id"]].copy()

valid_ids = get_valid_sequence(train_data, previous_day=previous_day)
valid_ids.extend(get_valid_sequence(vali_data, previous_day=previous_day))
valid_ids.extend(get_valid_sequence(test_data, previous_day=previous_day))

all_ids[f"{previous_day}"] = 0
all_ids.loc[all_ids["id"].isin(valid_ids), f"{previous_day}"] = 1

# Get final valid staypoint IDs
all_ids.set_index("id", inplace=True)
final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values

print(f"Valid staypoints: {len(final_valid_id)}")
```

#### Step 4: Filter Valid Users (Lines 200-232)

```python
# 4. Filter users based on final_valid_id
print("\n[4/5] Filtering users with valid sequences in all splits...")
valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()

valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))
print(f"Valid users (in all splits): {len(valid_users)}")

filtered_sp = sp_copy.loc[sp_copy["user_id"].isin(valid_users)].copy()

# Re-split with filtered users
train_data, vali_data, test_data = split_dataset(filtered_sp, split_ratios)

# Re-encode locations (with filtered data)
enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
    train_data["location_id"].values.reshape(-1, 1)
)
train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

# Re-encode user IDs to be continuous
user_enc = OrdinalEncoder(dtype=np.int64)
filtered_sp["user_id"] = user_enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1

train_data["user_id"] = user_enc.transform(train_data["user_id"].values.reshape(-1, 1)) + 1
vali_data["user_id"] = user_enc.transform(vali_data["user_id"].values.reshape(-1, 1)) + 1
test_data["user_id"] = user_enc.transform(test_data["user_id"].values.reshape(-1, 1)) + 1
```

**User Filtering Logic**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    USER FILTERING LOGIC                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Why filter users?                                                               │
│  • Users must have valid sequences in ALL splits (train, val, test)              │
│  • This ensures we can evaluate model on same users we trained on                │
│  • Prevents data leakage from user distribution differences                      │
│                                                                                  │
│  Process:                                                                        │
│  ─────────                                                                       │
│  1. Find users with valid sequences in train: {1, 2, 5, 7, 9}                   │
│  2. Find users with valid sequences in val:   {1, 2, 3, 5, 7}                   │
│  3. Find users with valid sequences in test:  {1, 5, 7, 8, 9}                   │
│                                                                                  │
│  4. Intersection: {1, 5, 7}                                                      │
│     Only users 1, 5, 7 have valid sequences in ALL splits                        │
│                                                                                  │
│  5. Re-filter data to keep only these users                                      │
│  6. Re-split (same logic, but fewer users/staypoints)                            │
│  7. Re-encode locations and users (vocabulary may change)                        │
│                                                                                  │
│  User ID encoding:                                                               │
│  Original: [1, 5, 7] → Encoded: [1, 2, 3] (0 reserved for padding)              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Step 5: Generate and Save Sequences (Lines 234-293)

```python
# 5. Generate sequences and save .pk files
print("\n[5/5] Generating sequences and saving .pk files...")

print("Processing train split...")
train_sequences = generate_sequences(train_data, final_valid_id, previous_day)
print(f"  Generated {len(train_sequences)} train sequences")

print("Processing validation split...")
val_sequences = generate_sequences(vali_data, final_valid_id, previous_day)
print(f"  Generated {len(val_sequences)} validation sequences")

print("Processing test split...")
test_sequences = generate_sequences(test_data, final_valid_id, previous_day)
print(f"  Generated {len(test_sequences)} test sequences")

# Save pickle files
train_pk_file = os.path.join(processed_path, f"{output_name}_train.pk")
val_pk_file = os.path.join(processed_path, f"{output_name}_validation.pk")
test_pk_file = os.path.join(processed_path, f"{output_name}_test.pk")

with open(train_pk_file, "wb") as f:
    pickle.dump(train_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✓ Saved train sequences to: {train_pk_file}")

with open(val_pk_file, "wb") as f:
    pickle.dump(val_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✓ Saved validation sequences to: {val_pk_file}")

with open(test_pk_file, "wb") as f:
    pickle.dump(test_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
print(f"✓ Saved test sequences to: {test_pk_file}")
```

#### Generate and Save Metadata (Lines 267-293)

```python
# Generate and save metadata
metadata = {
    "dataset_name": dataset_name,
    "output_dataset_name": output_name,
    "epsilon": epsilon,
    "previous_day": previous_day,
    "total_user_num": int(train_data["user_id"].max() + 1),
    "total_loc_num": int(train_data["location_id"].max() + 1),
    "unique_users": int(train_data["user_id"].nunique()),
    "unique_locations": int(train_data["location_id"].nunique()),
    "total_staypoints": int(len(filtered_sp)),
    "valid_staypoints": int(len(final_valid_id)),
    "train_staypoints": int(len(train_data)),
    "val_staypoints": int(len(vali_data)),
    "test_staypoints": int(len(test_data)),
    "train_sequences": len(train_sequences),
    "val_sequences": len(val_sequences),
    "test_sequences": len(test_sequences),
    "total_sequences": len(train_sequences) + len(val_sequences) + len(test_sequences),
    "split_ratios": split_ratios,
    "max_duration_minutes": max_duration,
}

metadata_file = os.path.join(processed_path, f"{output_name}_metadata.json")
with open(metadata_file, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✓ Saved metadata to: {metadata_file}")
```

**Metadata Contents**:

```json
{
  "dataset_name": "geolife",
  "output_dataset_name": "geolife_eps20_prev7",
  "epsilon": 20,
  "previous_day": 7,
  "total_user_num": 31,
  "total_loc_num": 245,
  "unique_users": 30,
  "unique_locations": 243,
  "total_staypoints": 15420,
  "valid_staypoints": 12350,
  "train_staypoints": 9252,
  "val_staypoints": 3084,
  "test_staypoints": 3084,
  "train_sequences": 8500,
  "val_sequences": 2100,
  "test_sequences": 2100,
  "total_sequences": 12700,
  "split_ratios": {"train": 0.6, "val": 0.2, "test": 0.2},
  "max_duration_minutes": 2880
}
```

---

## Main Function: process_intermediate_to_final

**Lines 298-356**

Orchestrates the entire Script 2 process.

```python
def process_intermediate_to_final(config):
    """Main processing function: intermediate to processed datasets."""
    
    dataset_config = config["dataset"]
    preproc_config = config["preprocessing"]
    
    dataset_name = dataset_config["name"]
    epsilon = dataset_config["epsilon"]
    previous_day_list = dataset_config["previous_day"]  # Now a list
    
    # Ensure previous_day is a list
    if not isinstance(previous_day_list, list):
        previous_day_list = [previous_day_list]
    
    # Paths
    output_folder = f"{dataset_name}_eps{epsilon}"
    interim_path = os.path.join("data", output_folder, "interim")
    output_base_path = os.path.join("data", output_folder)
    
    split_ratios = preproc_config["split"]
    max_duration = preproc_config.get("max_duration", 2880)
    
    # Load intermediate data
    print("\n[LOAD] Loading intermediate dataset...")
    interim_file = os.path.join(interim_path, f"intermediate_eps{epsilon}.csv")
    sp = pd.read_csv(interim_file)
    print(f"Loaded {len(sp)} staypoints from {sp['user_id'].nunique()} users")
    
    # Process for each previous_day value
    all_metadata = {}
    for previous_day in previous_day_list:
        metadata = process_for_previous_day(
            sp, split_ratios, max_duration, previous_day, 
            epsilon, dataset_name, output_base_path
        )
        all_metadata[previous_day] = metadata
    
    # Print summary
    print("\n" + "=" * 80)
    print("SCRIPT 2 COMPLETE: Interim to Processed")
    print("=" * 80)
```

**Multiple previous_day Values**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MULTIPLE PREVIOUS_DAY PROCESSING                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Config: previous_day: [3, 7, 14]                                                │
│                                                                                  │
│  Processing loop:                                                                │
│  ─────────────────                                                               │
│                                                                                  │
│  Iteration 1: previous_day = 3                                                   │
│  └─► geolife_eps20_prev3_train.pk                                               │
│  └─► geolife_eps20_prev3_validation.pk                                          │
│  └─► geolife_eps20_prev3_test.pk                                                │
│  └─► geolife_eps20_prev3_metadata.json                                          │
│                                                                                  │
│  Iteration 2: previous_day = 7                                                   │
│  └─► geolife_eps20_prev7_train.pk                                               │
│  └─► geolife_eps20_prev7_validation.pk                                          │
│  └─► geolife_eps20_prev7_test.pk                                                │
│  └─► geolife_eps20_prev7_metadata.json                                          │
│                                                                                  │
│  Iteration 3: previous_day = 14                                                  │
│  └─► geolife_eps20_prev14_train.pk                                              │
│  └─► geolife_eps20_prev14_validation.pk                                         │
│  └─► geolife_eps20_prev14_test.pk                                               │
│  └─► geolife_eps20_prev14_metadata.json                                         │
│                                                                                  │
│  Use case: Compare model performance with different history window sizes         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Example

Let's trace a complete example through Script 2:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              COMPLETE DATA FLOW: Script 2 Example                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT: intermediate_eps20.csv                                                   │
│  ─────────────────────────────                                                   │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ id │user_id│loc_id│start_day│end_day│start_min│end_min│weekday│duration │   │
│  │  0 │   1   │  15  │    0    │   0   │   540   │ 1020  │   0   │  480    │   │
│  │  1 │   1   │  12  │    0    │   0   │  1080   │ 1380  │   0   │  300    │   │
│  │  2 │   1   │  15  │    1    │   1   │   510   │ 1050  │   1   │  540    │   │
│  │ ...│  ...  │ ... │   ...   │  ... │   ...   │  ...  │  ...  │  ...    │   │
│  │500 │   1   │  12  │   99    │  99   │   480   │  960  │   0   │  480    │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  ↓ STEP 1: Split Dataset (60/20/20)                                             │
│                                                                                  │
│  User 1 has maxDay = 100                                                         │
│  train_split = 60, val_split = 80                                                │
│                                                                                  │
│  Train (days 0-59):    300 staypoints                                            │
│  Val (days 60-79):     100 staypoints                                            │
│  Test (days 80-99):    100 staypoints                                            │
│                                                                                  │
│  ↓ STEP 2: Encode Locations                                                      │
│                                                                                  │
│  Original loc_id: [15, 12, 15, 8, 3, 12, ...]                                   │
│  Encoded loc_id:  [ 4,  3,  4, 2, 5,  3, ...]  (+2 offset)                      │
│                                                                                  │
│  Location vocabulary: {0: pad, 1: unknown, 2: loc_A, 3: loc_B, 4: loc_C, ...}   │
│                                                                                  │
│  ↓ STEP 3: Filter Valid Sequences (previous_day=7)                              │
│                                                                                  │
│  For each staypoint, check:                                                      │
│  - diff_day >= 7?                                                                │
│  - len(history in 7-day window) >= 3?                                            │
│                                                                                  │
│  Valid staypoints: 400 out of 500 (80%)                                          │
│                                                                                  │
│  ↓ STEP 4: Filter Users (must have valid in all splits)                         │
│                                                                                  │
│  User 1: Has valid sequences in train ✓, val ✓, test ✓ → KEEP                   │
│  User 2: Has valid sequences in train ✓, val ✓, test ✗ → REMOVE                 │
│                                                                                  │
│  Final users: 25 out of 30                                                       │
│                                                                                  │
│  ↓ STEP 5: Generate Sequences                                                    │
│                                                                                  │
│  For each valid staypoint (target), create sequence dictionary:                  │
│                                                                                  │
│  Example sequence (target at day 50):                                            │
│  ┌────────────────────────────────────────────────────────────────────────┐     │
│  │ {                                                                      │     │
│  │   "X": [4, 3, 4, 2, 4],        # History locations (days 43-49)      │     │
│  │   "user_X": [1, 1, 1, 1, 1],   # User IDs                            │     │
│  │   "weekday_X": [2, 3, 4, 5, 6], # Wed, Thu, Fri, Sat, Sun            │     │
│  │   "start_min_X": [540, 510, 480, 600, 720],  # Start times           │     │
│  │   "dur_X": [480, 540, 300, 120, 180],  # Durations                   │     │
│  │   "diff": [7, 6, 5, 3, 1],     # Days before target                  │     │
│  │   "Y": 3                       # Target: predict location 3          │     │
│  │ }                                                                      │     │
│  └────────────────────────────────────────────────────────────────────────┘     │
│                                                                                  │
│  OUTPUT FILES:                                                                   │
│  ─────────────                                                                   │
│  geolife_eps20_prev7_train.pk       → [seq1, seq2, ..., seq8500]                │
│  geolife_eps20_prev7_validation.pk  → [seq1, seq2, ..., seq2100]                │
│  geolife_eps20_prev7_test.pk        → [seq1, seq2, ..., seq2100]                │
│  geolife_eps20_prev7_metadata.json  → Dataset statistics                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output File Formats

### Pickle (.pk) File Format

```python
# Loading a .pk file
import pickle

with open("geolife_eps20_prev7_train.pk", "rb") as f:
    train_sequences = pickle.load(f)

# train_sequences is a list of dictionaries
print(type(train_sequences))  # <class 'list'>
print(len(train_sequences))   # 8500 (number of sequences)

# Each element is a dictionary
sample = train_sequences[0]
print(sample.keys())  # dict_keys(['X', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff', 'Y'])

# Access features
print(sample["X"])           # array([4, 3, 4, 2, 4])
print(sample["Y"])           # 3
print(sample["weekday_X"])   # array([2, 3, 4, 5, 6])
```

### Using in PyTorch DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pickle

class GeoLifeDataset(Dataset):
    def __init__(self, pk_file):
        with open(pk_file, "rb") as f:
            self.sequences = pickle.load(f)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return {
            "X": torch.LongTensor(seq["X"]),
            "user_X": torch.LongTensor(seq["user_X"]),
            "weekday_X": torch.LongTensor(seq["weekday_X"]),
            "start_min_X": torch.LongTensor(seq["start_min_X"]),
            "dur_X": torch.LongTensor(seq["dur_X"]),
            "diff": torch.LongTensor(seq["diff"]),
            "Y": torch.LongTensor([seq["Y"]])
        }

# Usage
train_dataset = GeoLifeDataset("geolife_eps20_prev7_train.pk")
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    X = batch["X"]         # Shape: [32, variable_seq_len]
    Y = batch["Y"]         # Shape: [32, 1]
    # ... model training
```

---

## Summary

Script 2 transforms intermediate staypoint data into training-ready sequences:

1. **Split Dataset**: Chronological split per user (60/20/20)
2. **Encode Locations**: Create vocabulary with padding/unknown tokens
3. **Filter Valid Sequences**: Ensure sufficient history for each target
4. **Filter Valid Users**: Keep users with data in all splits
5. **Generate Sequences**: Create feature dictionaries for each valid target

The output `.pk` files are ready for direct use in deep learning frameworks.

---

## Next Steps

- [06-H3-SCRIPT1-RAW-TO-INTERIM.md](06-H3-SCRIPT1-RAW-TO-INTERIM.md) - H3 version of Script 1
- [08-DATA-STRUCTURES.md](08-DATA-STRUCTURES.md) - Detailed data format reference
- [10-EXAMPLES.md](10-EXAMPLES.md) - More concrete examples

---

*Documentation Version: 1.0*
*For PhD Research Reference*
