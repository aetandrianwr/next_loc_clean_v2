# Interim to Processed Script Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Script Architecture](#script-architecture)
3. [Input/Output Specification](#inputoutput-specification)
4. [Line-by-Line Code Walkthrough](#line-by-line-code-walkthrough)
5. [Step 1: Dataset Splitting](#step-1-dataset-splitting)
6. [Step 2: Location Encoding](#step-2-location-encoding)
7. [Step 3: Valid Sequence Filtering](#step-3-valid-sequence-filtering)
8. [Step 4: User Filtering](#step-4-user-filtering)
9. [Step 5: Sequence Generation](#step-5-sequence-generation)
10. [Output Format](#output-format)
11. [Complete Example](#complete-example)

---

## Overview

**Script**: `preprocessing/diy_2_interim_to_processed.py`  
**Purpose**: Transform interim staypoint data into sequence files (.pk) for model training  
**Input**: Intermediate CSV from Script 1  
**Output**: Train/Validation/Test pickle files with sequence dictionaries

### What This Script Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    diy_2_interim_to_processed.py OVERVIEW                    │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT FILE:
└── data/diy_eps50/interim/intermediate_eps50.csv

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PROCESSING STEPS (for each previous_day value in config):                   │
│                                                                              │
│ STEP 1: Split Dataset ──────────────────────────────────────────────────────│
│         • Temporal split per user                                            │
│         • train=80%, val=10%, test=10%                                       │
│                                                                              │
│ STEP 2: Encode Locations ───────────────────────────────────────────────────│
│         • OrdinalEncoder fit on train data                                   │
│         • Add +2 offset (0=padding, 1=unknown)                               │
│                                                                              │
│ STEP 3: Filter Valid Sequences ─────────────────────────────────────────────│
│         • previous_day requirement (e.g., 7 days history)                    │
│         • min_sequence_length requirement (e.g., 3 staypoints)              │
│                                                                              │
│ STEP 4: Filter Users ───────────────────────────────────────────────────────│
│         • Keep users with valid sequences in ALL splits                      │
│         • Re-encode location and user IDs                                    │
│                                                                              │
│ STEP 5: Generate Sequences ─────────────────────────────────────────────────│
│         • Create X/Y pairs with features                                     │
│         • Parallel processing per user                                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
OUTPUT FILES:
├── data/diy_eps50/processed/diy_eps50_prev7_train.pk
├── data/diy_eps50/processed/diy_eps50_prev7_validation.pk
├── data/diy_eps50/processed/diy_eps50_prev7_test.pk
└── data/diy_eps50/processed/diy_eps50_prev7_metadata.json
```

---

## Script Architecture

### High-Level Structure

```python
"""
diy_2_interim_to_processed.py - Script Structure
"""

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ IMPORTS (Lines 19-31)                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘
import os, sys, json, pickle, argparse
from pathlib import Path
import yaml, pandas as pd, numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from joblib import Parallel, delayed, parallel_backend

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SPLITTING FUNCTIONS (Lines 37-65)                                           │
# └─────────────────────────────────────────────────────────────────────────────┘
def split_dataset(totalData, split_ratios):    # Main split function
def _get_split_days_user(df, split_ratios):    # Per-user split helper

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SEQUENCE VALIDATION (Lines 68-89)                                           │
# └─────────────────────────────────────────────────────────────────────────────┘
def get_valid_sequence(input_df, previous_day, min_length):  # Find valid IDs

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ SEQUENCE GENERATION (Lines 92-152)                                          │
# └─────────────────────────────────────────────────────────────────────────────┘
def _get_valid_sequence_user(args):    # Per-user sequence generation
def generate_sequences(data, ...):      # Parallel orchestration

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MAIN PROCESSING (Lines 155-298)                                             │
# └─────────────────────────────────────────────────────────────────────────────┘
def process_for_previous_day(...):     # Process for one previous_day value

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MAIN ENTRY POINT (Lines 301-374)                                            │
# └─────────────────────────────────────────────────────────────────────────────┘
def main():                            # CLI entry point
```

---

## Input/Output Specification

### Input File

```
INPUT FILE: intermediate_eps{epsilon}.csv
═══════════════════════════════════════════════════════════════════════════════

Location: data/diy_eps{epsilon}/interim/intermediate_eps{epsilon}.csv
Format: CSV

Schema:
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ id             │ int64        │ Sequential staypoint ID                        │
│ user_id        │ int64        │ Integer user ID                                │
│ location_id    │ int64        │ DBSCAN cluster ID                              │
│ start_day      │ int64        │ Days since user's first record                 │
│ end_day        │ int64        │ End day number                                 │
│ start_min      │ int64        │ Start minute of day (0-1439)                   │
│ end_min        │ int64        │ End minute of day (1-1440)                     │
│ weekday        │ int64        │ Day of week (0-6)                              │
│ duration       │ float64      │ Duration in minutes                            │
└────────────────┴──────────────┴────────────────────────────────────────────────┘

Example:
┌────┬─────────┬─────────────┬───────────┬─────────┬───────────┬─────────┬─────────┬──────────┐
│ id │ user_id │ location_id │ start_day │ end_day │ start_min │ end_min │ weekday │ duration │
├────┼─────────┼─────────────┼───────────┼─────────┼───────────┼─────────┼─────────┼──────────┤
│ 0  │ 0       │ 42          │ 0         │ 0       │ 127       │ 510     │ 6       │ 383      │
│ 1  │ 0       │ 15          │ 0         │ 0       │ 555       │ 765     │ 6       │ 210      │
│ 2  │ 0       │ 42          │ 0         │ 0       │ 810       │ 1020    │ 6       │ 210      │
│ 3  │ 0       │ 8           │ 1         │ 1       │ 65        │ 170     │ 0       │ 105      │
└────┴─────────┴─────────────┴───────────┴─────────┴───────────┴─────────┴─────────┴──────────┘
```

### Output Files

```
OUTPUT FILES: Pickle (.pk) files with sequence lists
═══════════════════════════════════════════════════════════════════════════════

Location: data/diy_eps{epsilon}/processed/

Files Generated (for each previous_day):
├── diy_eps{epsilon}_prev{day}_train.pk
├── diy_eps{epsilon}_prev{day}_validation.pk
├── diy_eps{epsilon}_prev{day}_test.pk
└── diy_eps{epsilon}_prev{day}_metadata.json

Pickle File Structure:
─────────────────────────────────────────────────────────────────────────────────
List[Dict] - Each dictionary is one training sample

Sample Dictionary:
{
    "X":           numpy.ndarray  # History location IDs
    "user_X":      numpy.ndarray  # User ID for each history step
    "weekday_X":   numpy.ndarray  # Weekday for each history step
    "start_min_X": numpy.ndarray  # Start minute for each history step
    "dur_X":       numpy.ndarray  # Duration for each history step
    "diff":        numpy.ndarray  # Days before target for each step
    "Y":           int            # Target location ID to predict
}
```

---

## Line-by-Line Code Walkthrough

### Imports (Lines 19-35)

```python
"""
DIY Dataset Preprocessing - Script 2: Interim to Processed
Processes intermediate staypoint data to final sequence .pk files.
"""

import os
import sys
import json
import pickle                          # For saving sequences
import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder  # Location ID encoding
from joblib import Parallel, delayed, parallel_backend  # Parallel processing

# Set random seed
RANDOM_SEED = 42
```

---

## Step 1: Dataset Splitting

### Function: `_get_split_days_user(df, split_ratios)` (Lines 55-65)

```python
def _get_split_days_user(df, split_ratios):
    """Split the dataset according to the tracked day of each user.
    
    This function is called per user to assign each staypoint to
    train, validation, or test based on temporal position.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1a.1: Get maximum day for this user
    # ─────────────────────────────────────────────────────────────────────────
    maxDay = df["start_day"].max()
    # Example: User tracked for 100 days, maxDay = 100
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1a.2: Calculate split boundaries
    # ─────────────────────────────────────────────────────────────────────────
    train_split = maxDay * split_ratios["train"]
    # train_split = 100 * 0.8 = 80
    
    validation_split = maxDay * (split_ratios["train"] + split_ratios["val"])
    # validation_split = 100 * (0.8 + 0.1) = 90
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1a.3: Assign dataset labels based on start_day
    # ─────────────────────────────────────────────────────────────────────────
    df["Dataset"] = "test"  # Default to test
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & (df["start_day"] < validation_split), "Dataset"] = "vali"
    
    # Result:
    # Day 0-79:  "train"
    # Day 80-89: "vali"
    # Day 90-100: "test"
    
    return df
```

### Function: `split_dataset(totalData, split_ratios)` (Lines 37-52)

```python
def split_dataset(totalData, split_ratios):
    """Split dataset into train, val and test per user."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.1: Apply split to each user
    # ─────────────────────────────────────────────────────────────────────────
    totalData = totalData.groupby("user_id", group_keys=False).apply(
        _get_split_days_user, split_ratios=split_ratios
    )
    # Each user's timeline is split independently
    # User A's train might be Day 0-64 (if tracked 80 days)
    # User B's train might be Day 0-120 (if tracked 150 days)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.2: Separate into three DataFrames
    # ─────────────────────────────────────────────────────────────────────────
    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.3: Clean up - remove Dataset column
    # ─────────────────────────────────────────────────────────────────────────
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data
```

**Temporal Splitting Visualization:**

```
TEMPORAL SPLITTING PER USER
═══════════════════════════════════════════════════════════════════════════════

Split Ratios: train=0.8, val=0.1, test=0.1

User A (tracked 100 days):
Day: 0          20          40          60          80    90        100
     ├───────────┼───────────┼───────────┼───────────┼─────┼──────────┤
     │◄──────────────────── TRAIN (80%) ───────────────▶│VAL │◄TEST(10%)▶│
     │                      80 days                     │10% │  10 days  │

User B (tracked 80 days):
Day: 0          16          32          48          64  72         80
     ├───────────┼───────────┼───────────┼───────────┼────┼──────────┤
     │◄──────────────────── TRAIN (80%) ───────────────▶│VAL│◄TEST(10%)▶│
     │                      64 days                     │8d │  8 days   │


WHY TEMPORAL SPLIT (NOT RANDOM)?
─────────────────────────────────────────────────────────────────────────────────

Random Split (BAD - Data Leakage):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Day:  0   10   20   30   40   50   60   70   80   90   100                 │
│       T   V    T    T    V    T    T    V    T    T    Test                │
│       ↑   ↑                                                                 │
│       │   └─ Validation sample at Day 10                                    │
│       └─ Train sample at Day 0                                              │
│                                                                             │
│ PROBLEM: Model learns from Day 40 data to predict Day 10 location!         │
│          This is "future leakage" - unrealistic in production              │
└─────────────────────────────────────────────────────────────────────────────┘

Temporal Split (GOOD - No Leakage):
┌─────────────────────────────────────────────────────────────────────────────┐
│ Day:  0   10   20   30   40   50   60   70   80   90   100                 │
│       │◄────────────── TRAIN ──────────────────▶│VAL│◄──TEST──▶│           │
│                                                                             │
│ CORRECT: Train only on past data, test on future data                       │
│          Mirrors real-world prediction scenario                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Step 2: Location Encoding

### Code Block (Lines 176-186 in `process_for_previous_day`)

```python
# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Encode Location IDs
# ─────────────────────────────────────────────────────────────────────────────
print("\n[2/5] Encoding location IDs...")

# Create encoder fitted on training data only
enc = OrdinalEncoder(
    dtype=np.int64,                    # Output integer type
    handle_unknown="use_encoded_value", # Handle unseen locations
    unknown_value=-1                    # Map unseen to -1 (then +2 = 1)
).fit(train_data["location_id"].values.reshape(-1, 1))

# Apply encoding with +2 offset
# 0 = padding (for sequence padding in models)
# 1 = unknown location (locations not seen in training)
# 2+ = actual encoded location IDs
train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

print(f"Max location ID: {train_data['location_id'].max()}")
print(f"Unique locations in train: {train_data['location_id'].nunique()}")
```

**Location Encoding Visualization:**

```
LOCATION ID ENCODING PROCESS
═══════════════════════════════════════════════════════════════════════════════

Why Encode Location IDs?
─────────────────────────────────────────────────────────────────────────────────
Original DBSCAN location IDs may be sparse (e.g., 0, 5, 42, 103, 500)
Neural networks need dense, consecutive IDs starting from 0


OrdinalEncoder Transformation:
─────────────────────────────────────────────────────────────────────────────────

Training locations: [5, 42, 103, 500, 1024]

Encoder mapping (fit on train):
┌─────────────────┬─────────────────┐
│ Original ID     │ Encoded ID      │
├─────────────────┼─────────────────┤
│ 5               │ 0               │
│ 42              │ 1               │
│ 103             │ 2               │
│ 500             │ 3               │
│ 1024            │ 4               │
└─────────────────┴─────────────────┘


Adding +2 Offset:
─────────────────────────────────────────────────────────────────────────────────

Final encoding (after +2):
┌─────────────────┬─────────────────┬─────────────────────────────────────────┐
│ Original ID     │ Final ID        │ Purpose                                 │
├─────────────────┼─────────────────┼─────────────────────────────────────────┤
│ (reserved)      │ 0               │ Padding token (for batch sequences)     │
│ (reserved)      │ 1               │ Unknown location (not in train)         │
│ 5               │ 2               │ First actual location                   │
│ 42              │ 3               │ Second actual location                  │
│ 103             │ 4               │ Third actual location                   │
│ 500             │ 5               │ Fourth actual location                  │
│ 1024            │ 6               │ Fifth actual location                   │
└─────────────────┴─────────────────┴─────────────────────────────────────────┘


Handling Unknown Locations in Val/Test:
─────────────────────────────────────────────────────────────────────────────────

If validation has location ID 200 (not in train):
• OrdinalEncoder returns -1 (unknown_value)
• After +2: -1 + 2 = 1 (unknown token)

Example:
┌────────────────┬───────────────────┬────────────────┐
│ Data Split     │ Original Loc ID   │ Final Loc ID   │
├────────────────┼───────────────────┼────────────────┤
│ Train          │ 42                │ 3              │
│ Train          │ 103               │ 4              │
│ Validation     │ 42                │ 3              │
│ Validation     │ 200 (new!)        │ 1 (unknown)    │
│ Test           │ 500               │ 5              │
│ Test           │ 999 (new!)        │ 1 (unknown)    │
└────────────────┴───────────────────┴────────────────┘


Model Embedding Layer Size:
─────────────────────────────────────────────────────────────────────────────────
total_loc_num = max_location_id + 1
             = 6 + 1 = 7

Embedding layer: nn.Embedding(num_embeddings=7, embedding_dim=64)
• Index 0: Padding embedding
• Index 1: Unknown location embedding
• Index 2-6: Actual location embeddings
```

---

## Step 3: Valid Sequence Filtering

### Function: `get_valid_sequence(input_df, previous_day, min_length)` (Lines 68-89)

```python
def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """Get valid sequence IDs based on previous_day requirement.
    
    A staypoint is valid if:
    1. It has at least `previous_day` days of history
    2. The history contains at least `min_length` staypoints
    """
    valid_id = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.1: Process each user separately
    # ─────────────────────────────────────────────────────────────────────────
    for user in input_df["user_id"].unique():
        df = input_df.loc[input_df["user_id"] == user].copy().reset_index(drop=True)
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3.2: Calculate days since user's first record
        # ─────────────────────────────────────────────────────────────────────
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days
        # diff_day = relative day number starting from 0
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 3.3: Check each staypoint for validity
        # ─────────────────────────────────────────────────────────────────────
        for index, row in df.iterrows():
            # Requirement 1: Must have enough history days
            if row["diff_day"] < previous_day:
                continue
            # If diff_day < 7, we don't have 7 days of history yet
            
            # ─────────────────────────────────────────────────────────────────
            # Step 3.4: Get history within previous_day window
            # ─────────────────────────────────────────────────────────────────
            hist = df.iloc[:index]  # All staypoints before current
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
            # Keep only staypoints from the last 7 days
            
            # Requirement 2: Must have enough history staypoints
            if len(hist) < min_length:
                continue
            # If fewer than 3 staypoints in history, not enough context

            # ─────────────────────────────────────────────────────────────────
            # Step 3.5: Mark as valid
            # ─────────────────────────────────────────────────────────────────
            valid_id.append(row["id"])

    return valid_id
```

**Valid Sequence Filtering Visualization:**

```
VALID SEQUENCE FILTERING PROCESS
═══════════════════════════════════════════════════════════════════════════════

Parameters:
• previous_day = 7 (look back 7 days for history)
• min_length = 3 (need at least 3 history staypoints)


User Timeline with Staypoints:
─────────────────────────────────────────────────────────────────────────────────

Day:  0    1    2    3    4    5    6    7    8    9    10   11   12
      │    │    │    │    │    │    │    │    │    │    │    │    │
      SP0  SP1  SP2       SP3  SP4  SP5  SP6  SP7       SP8  SP9  SP10


Checking Each Staypoint:
─────────────────────────────────────────────────────────────────────────────────

SP0 (Day 0): diff_day=0 < 7     → SKIP (not enough history days)
SP1 (Day 1): diff_day=1 < 7     → SKIP
SP2 (Day 2): diff_day=2 < 7     → SKIP
SP3 (Day 4): diff_day=4 < 7     → SKIP
SP4 (Day 5): diff_day=5 < 7     → SKIP
SP5 (Day 6): diff_day=6 < 7     → SKIP
SP6 (Day 7): diff_day=7 >= 7    → CHECK HISTORY
             History window: Day 0-7
             History staypoints: [SP0, SP1, SP2, SP3, SP4, SP5]
             Count: 6 >= 3       → VALID ✓

SP7 (Day 8): diff_day=8 >= 7    → CHECK HISTORY
             History window: Day 1-8
             History staypoints: [SP1, SP2, SP3, SP4, SP5, SP6]
             Count: 6 >= 3       → VALID ✓

SP8 (Day 10): diff_day=10 >= 7  → CHECK HISTORY
              History window: Day 3-10
              History staypoints: [SP3, SP4, SP5, SP6, SP7]
              Count: 5 >= 3      → VALID ✓

SP9 (Day 11): diff_day=11 >= 7  → CHECK HISTORY
              History window: Day 4-11
              History staypoints: [SP4, SP5, SP6, SP7, SP8]
              Count: 5 >= 3      → VALID ✓

SP10 (Day 12): diff_day=12 >= 7 → CHECK HISTORY
               History window: Day 5-12
               History staypoints: [SP5, SP6, SP7, SP8, SP9]
               Count: 5 >= 3     → VALID ✓


Valid Staypoint IDs: [SP6, SP7, SP8, SP9, SP10]
─────────────────────────────────────────────────────────────────────────────────

EDGE CASE: Sparse History
─────────────────────────────────────────────────────────────────────────────────

Day:  0    1    2    3    4    5    6    7    8    9    10
      │    │    │    │    │    │    │    │    │    │
      SP0                 SP1                      SP2

SP2 (Day 10): diff_day=10 >= 7  → CHECK HISTORY
              History window: Day 3-10
              History staypoints: [SP1] (only 1!)
              Count: 1 < 3       → INVALID ✗ (not enough history)
```

---

## Step 4: User Filtering

### Code Block (Lines 206-238 in `process_for_previous_day`)

```python
# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Filter Users with Valid Sequences in ALL Splits
# ─────────────────────────────────────────────────────────────────────────────
print("\n[4/5] Filtering users with valid sequences in all splits...")

# Find users with valid sequences in each split
valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()

# Keep only users who have valid sequences in ALL THREE splits
valid_users = set.intersection(
    set(valid_users_train), 
    set(valid_users_vali), 
    set(valid_users_test)
)
print(f"Valid users (in all splits): {len(valid_users)}")

# Filter to valid users only
filtered_sp = sp_copy.loc[sp_copy["user_id"].isin(valid_users)].copy()

# Re-split with filtered users
train_data, vali_data, test_data = split_dataset(filtered_sp, split_ratios)

# Re-encode locations (vocabulary may have changed)
enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
    train_data["location_id"].values.reshape(-1, 1)
)
train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

# Re-encode user IDs to be continuous (0, 1, 2, ...)
user_enc = OrdinalEncoder(dtype=np.int64)
filtered_sp["user_id"] = user_enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1

train_data["user_id"] = user_enc.transform(train_data["user_id"].values.reshape(-1, 1)) + 1
vali_data["user_id"] = user_enc.transform(vali_data["user_id"].values.reshape(-1, 1)) + 1
test_data["user_id"] = user_enc.transform(test_data["user_id"].values.reshape(-1, 1)) + 1
```

**User Filtering Visualization:**

```
USER FILTERING PROCESS
═══════════════════════════════════════════════════════════════════════════════

Why Filter Users?
─────────────────────────────────────────────────────────────────────────────────
We need users who have valid sequences in ALL splits for fair evaluation.
A user with valid train sequences but no valid test sequences can't be evaluated.


Example User Scenarios:
─────────────────────────────────────────────────────────────────────────────────

User A (KEEP ✓):
Train: [SP0✓, SP1✓, SP2✓, SP3✓, SP4✓]  → Has valid sequences
Val:   [SP5✓, SP6✓]                     → Has valid sequences  
Test:  [SP7✓, SP8✓, SP9✓]               → Has valid sequences

User B (REMOVE ✗):
Train: [SP0✓, SP1✓, SP2✓, SP3✓]         → Has valid sequences
Val:   [SP4✗]                           → Only 1 staypoint, no valid sequence
Test:  [SP5✓, SP6✓]                     → Has valid sequences

User C (REMOVE ✗):
Train: [SP0✓, SP1✓, SP2✓]               → Has valid sequences
Val:   [SP3✓]                           → Has valid sequence
Test:  []                               → No staypoints at all!


Set Intersection:
─────────────────────────────────────────────────────────────────────────────────

valid_users_train = {A, B, C, D, E}
valid_users_vali  = {A, C, D, F}
valid_users_test  = {A, D, E, G}

valid_users = {A, B, C, D, E} ∩ {A, C, D, F} ∩ {A, D, E, G}
            = {A, D}

Only users A and D have valid sequences in ALL THREE splits!


User ID Re-encoding:
─────────────────────────────────────────────────────────────────────────────────

Before filtering:
user_ids = [0, 1, 2, 3, 4]  (5 users)

After filtering (users 0, 2 removed):
user_ids = [1, 3, 4]  (3 users, but IDs have gaps)

After re-encoding (+1 offset):
user_ids = [1, 2, 3]  (3 users, consecutive starting from 1)

Why +1?
• User ID 0 is reserved for "padding user" in batch processing
```

---

## Step 5: Sequence Generation

### Function: `_get_valid_sequence_user(args)` (Lines 92-130)

```python
def _get_valid_sequence_user(args):
    """Get valid sequences per user - for parallel processing.
    
    This function generates all valid sequences for ONE user.
    Called in parallel for efficiency.
    """
    df, previous_day, valid_ids = args
    df = df.reset_index(drop=True)
    data_single_user = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5.1: Calculate relative day for each staypoint
    # ─────────────────────────────────────────────────────────────────────────
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5.2: Iterate through each staypoint as potential target
    # ─────────────────────────────────────────────────────────────────────────
    for index, row in df.iterrows():
        # Skip if not enough history days
        if row["diff_day"] < previous_day:
            continue
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5.3: Get history window
        # ─────────────────────────────────────────────────────────────────────
        hist = df.iloc[:index]  # All staypoints before current
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
        # Filter to only last `previous_day` days
        
        # Skip if this staypoint ID is not in valid set
        if not (row["id"] in valid_ids):
            continue
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5.4: Build sequence dictionary
        # ─────────────────────────────────────────────────────────────────────
        data_dict = {}
        
        # Feature: Location sequence (history)
        data_dict["X"] = hist["location_id"].values
        
        # Feature: User ID (repeated for each history step)
        data_dict["user_X"] = hist["user_id"].values
        
        # Feature: Weekday (0-6)
        data_dict["weekday_X"] = hist["weekday"].values
        
        # Feature: Start minute of day (0-1439)
        data_dict["start_min_X"] = hist["start_min"].values
        
        # Feature: Duration in minutes
        data_dict["dur_X"] = hist["duration"].values
        
        # Feature: Days difference to current target
        data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values
        # diff tells the model "how many days ago was this history point"
        
        # Target: Next location to predict
        data_dict["Y"] = int(row["location_id"])
        
        # ─────────────────────────────────────────────────────────────────────
        # Step 5.5: Add to results
        # ─────────────────────────────────────────────────────────────────────
        data_single_user.append(data_dict)
    
    return data_single_user
```

### Function: `generate_sequences(data, valid_ids, previous_day, split_name)` (Lines 133-152)

```python
def generate_sequences(data, valid_ids, previous_day, split_name):
    """Generate sequences from data using parallel processing."""
    print(f"  Processing {split_name} sequences...")
    
    valid_ids_set = set(valid_ids)  # For O(1) lookup
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5a.1: Prepare arguments for parallel workers
    # ─────────────────────────────────────────────────────────────────────────
    user_groups = [
        (group.copy(), previous_day, valid_ids_set) 
        for _, group in data.groupby("user_id")
    ]
    # Each tuple contains: (user's DataFrame, previous_day, valid IDs set)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5a.2: Run parallel processing
    # ─────────────────────────────────────────────────────────────────────────
    with parallel_backend("threading", n_jobs=-1):  # Use all CPU cores
        valid_user_ls = Parallel()(
            delayed(_get_valid_sequence_user)(args) 
            for args in tqdm(user_groups, desc=f"    {split_name}")
        )
    # valid_user_ls is a list of lists: [[user1_sequences], [user2_sequences], ...]
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 5a.3: Flatten results
    # ─────────────────────────────────────────────────────────────────────────
    valid_records = [item for sublist in valid_user_ls for item in sublist]
    # Combine all sequences into one flat list
    
    return valid_records
```

**Sequence Generation Visualization:**

```
SEQUENCE GENERATION DETAILED EXAMPLE
═══════════════════════════════════════════════════════════════════════════════

User: user_1 (user_id=1)
previous_day: 7

User's Staypoint Timeline:
┌────┬───────────┬─────────────┬─────────┬───────────┬──────────┬──────────┐
│ id │ start_day │ location_id │ weekday │ start_min │ duration │ diff_day │
├────┼───────────┼─────────────┼─────────┼───────────┼──────────┼──────────┤
│ 0  │ 0         │ 44          │ 6       │ 420       │ 380      │ 0        │
│ 1  │ 1         │ 17          │ 0       │ 540       │ 210      │ 1        │
│ 2  │ 2         │ 44          │ 1       │ 480       │ 240      │ 2        │
│ 3  │ 3         │ 10          │ 2       │ 600       │ 105      │ 3        │
│ 4  │ 4         │ 44          │ 3       │ 450       │ 660      │ 4        │
│ 5  │ 5         │ 17          │ 4       │ 540       │ 210      │ 5        │
│ 6  │ 6         │ 44          │ 5       │ 420       │ 380      │ 6        │
│ 7  │ 7         │ 17          │ 6       │ 540       │ 210      │ 7        │ ← First valid target
│ 8  │ 8         │ 44          │ 0       │ 480       │ 300      │ 8        │
│ 9  │ 9         │ 10          │ 1       │ 600       │ 150      │ 9        │
└────┴───────────┴─────────────┴─────────┴───────────┴──────────┴──────────┘


Processing id=7 (First Valid Target):
─────────────────────────────────────────────────────────────────────────────────

Target: id=7, diff_day=7, location_id=17

History window: diff_day >= (7-7) = 0 → All staypoints from diff_day 0-6
History staypoints: [0, 1, 2, 3, 4, 5, 6]

Generated Dictionary:
┌─────────────────┬──────────────────────────────────────────────────────────────┐
│ Key             │ Value                                                        │
├─────────────────┼──────────────────────────────────────────────────────────────┤
│ X               │ [44, 17, 44, 10, 44, 17, 44]    # 7 locations               │
│ user_X          │ [1, 1, 1, 1, 1, 1, 1]           # User ID repeated          │
│ weekday_X       │ [6, 0, 1, 2, 3, 4, 5]           # Sat, Mon, Tue, Wed...     │
│ start_min_X     │ [420, 540, 480, 600, 450, 540, 420]  # Start minutes        │
│ dur_X           │ [380, 210, 240, 105, 660, 210, 380]  # Durations            │
│ diff            │ [7, 6, 5, 4, 3, 2, 1]           # Days before target        │
│ Y               │ 17                              # Target location            │
└─────────────────┴──────────────────────────────────────────────────────────────┘


Diff Calculation Explained:
─────────────────────────────────────────────────────────────────────────────────

Target diff_day = 7

diff = target_diff_day - history_diff_day

For each history staypoint:
• SP0 (diff_day=0): diff = 7 - 0 = 7 (7 days before target)
• SP1 (diff_day=1): diff = 7 - 1 = 6 (6 days before target)
• SP2 (diff_day=2): diff = 7 - 2 = 5 (5 days before target)
• ...
• SP6 (diff_day=6): diff = 7 - 6 = 1 (1 day before target)

This tells the model the temporal distance of each history point to the target.


Processing id=8 (Second Valid Target):
─────────────────────────────────────────────────────────────────────────────────

Target: id=8, diff_day=8, location_id=44

History window: diff_day >= (8-7) = 1 → Staypoints from diff_day 1-7
History staypoints: [1, 2, 3, 4, 5, 6, 7]  (SP0 is excluded! diff_day=0 < 1)

Generated Dictionary:
┌─────────────────┬──────────────────────────────────────────────────────────────┐
│ Key             │ Value                                                        │
├─────────────────┼──────────────────────────────────────────────────────────────┤
│ X               │ [17, 44, 10, 44, 17, 44, 17]    # 7 locations               │
│ user_X          │ [1, 1, 1, 1, 1, 1, 1]                                        │
│ weekday_X       │ [0, 1, 2, 3, 4, 5, 6]                                        │
│ start_min_X     │ [540, 480, 600, 450, 540, 420, 540]                          │
│ dur_X           │ [210, 240, 105, 660, 210, 380, 210]                          │
│ diff            │ [7, 6, 5, 4, 3, 2, 1]                                        │
│ Y               │ 44                                                           │
└─────────────────┴──────────────────────────────────────────────────────────────┘
```

---

## Output Format

### Pickle File Structure

```python
# Loading and using the output files

import pickle

# Load training sequences
with open("data/diy_eps50/processed/diy_eps50_prev7_train.pk", "rb") as f:
    train_sequences = pickle.load(f)

# train_sequences is a list of dictionaries
print(f"Number of training sequences: {len(train_sequences)}")
# Output: Number of training sequences: 65234

# Inspect one sequence
seq = train_sequences[0]
print(f"History length: {len(seq['X'])}")
print(f"Location sequence: {seq['X']}")
print(f"Target location: {seq['Y']}")

# Output:
# History length: 7
# Location sequence: [44 17 44 10 44 17 44]
# Target location: 17
```

### Metadata JSON Structure

```json
{
    "dataset_name": "diy",
    "output_dataset_name": "diy_eps50_prev7",
    "epsilon": 50,
    "previous_day": 7,
    "total_user_num": 156,
    "total_loc_num": 4523,
    "unique_users": 155,
    "unique_locations": 4521,
    "total_staypoints": 125432,
    "valid_staypoints": 98765,
    "train_staypoints": 78432,
    "val_staypoints": 9876,
    "test_staypoints": 9889,
    "train_sequences": 65234,
    "val_sequences": 8123,
    "test_sequences": 8234,
    "total_sequences": 81591,
    "split_ratios": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    },
    "max_duration_minutes": 2880
}
```

---

## Complete Example

### Running the Script

```bash
# Standard run
python preprocessing/diy_2_interim_to_processed.py --config config/preprocessing/diy.yaml

# Custom configuration
python preprocessing/diy_2_interim_to_processed.py --config config/preprocessing/diy_custom.yaml
```

### Example Console Output

```
================================================================================
DIY PREPROCESSING - Script 2: Interim to Processed
================================================================================
[INPUT]  Interim folder: data/diy_eps50/interim
[OUTPUT] Processed folder: data/diy_eps50/processed/
[CONFIG] Dataset: diy, Epsilon: 50
[CONFIG] Previous days: [7]
================================================================================

[LOAD] Loading intermediate dataset...
Loaded 285678 staypoints from 155 users
Input file: data/diy_eps50/interim/intermediate_eps50.csv

------------------------------------------------------------
Processing for previous_day = 7
------------------------------------------------------------

[1/5] Splitting dataset into train/val/test...
Train: 228542, Val: 28568, Test: 28568

[2/5] Encoding location IDs...
Max location ID: 4523
Unique locations in train: 4521

[3/5] Filtering valid sequences (previous_day=7)...
Valid staypoints: 198765

[4/5] Filtering users with valid sequences in all splits...
Valid users (in all splits): 152

Final max location ID: 4456
Final unique locations: 4454
Final user count: 152

[5/5] Generating sequences and saving .pk files...
  Processing train sequences...
    train: 100%|████████████████████████| 152/152 [00:05<00:00, 28.4it/s]
  Generated 65234 train sequences
  Processing validation sequences...
    validation: 100%|████████████████████| 152/152 [00:01<00:00, 95.2it/s]
  Generated 8123 validation sequences
  Processing test sequences...
    test: 100%|██████████████████████████| 152/152 [00:01<00:00, 92.8it/s]
  Generated 8234 test sequences

✓ Saved train sequences to: data/diy_eps50/processed/diy_eps50_prev7_train.pk
✓ Saved validation sequences to: data/diy_eps50/processed/diy_eps50_prev7_validation.pk
✓ Saved test sequences to: data/diy_eps50/processed/diy_eps50_prev7_test.pk
✓ Saved metadata to: data/diy_eps50/processed/diy_eps50_prev7_metadata.json

================================================================================
SCRIPT 2 COMPLETE: Interim to Processed
================================================================================
Output folder: data/diy_eps50/processed/

previous_day=7:
  Train: 65234, Val: 8123, Test: 8234
  Total users: 156, Total locations: 4523
================================================================================
```

---

## Summary

The `diy_2_interim_to_processed.py` script:

1. **Splits** data temporally per user (80/10/10)
2. **Encodes** location IDs with padding/unknown tokens
3. **Filters** valid sequences based on history requirements
4. **Filters** users with valid data in all splits
5. **Generates** sequence dictionaries with features
6. **Saves** pickle files ready for model training

Key parameters:
- `previous_day`: History window size (default: 7)
- `min_sequence_length`: Minimum history length (default: 3)
- `split`: Train/Val/Test ratios (default: 0.8/0.1/0.1)
- `max_duration`: Duration truncation (default: 2880 min)

Output: `{dataset}_train.pk`, `{dataset}_validation.pk`, `{dataset}_test.pk`, `{dataset}_metadata.json`
