# Interim to Processed Script Documentation (H3 Version)

## 📋 Table of Contents
1. [Overview](#overview)
2. [Differences from DBSCAN Version](#differences-from-dbscan-version)
3. [Complete Script Walkthrough](#complete-script-walkthrough)
4. [Input/Output Specification](#inputoutput-specification)
5. [Running the Script](#running-the-script)
6. [Integration with Model Training](#integration-with-model-training)

---

## Overview

**Script**: `preprocessing/diy_h3_2_interim_to_processed.py`  
**Purpose**: Transform H3-based interim data into sequence files (.pk) for model training  
**Input**: Intermediate CSV from H3 Script 1  
**Output**: Train/Validation/Test pickle files

### Script Summary

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            diy_h3_2_interim_to_processed.py OVERVIEW                         │
└─────────────────────────────────────────────────────────────────────────────┘

This script is FUNCTIONALLY IDENTICAL to diy_2_interim_to_processed.py
The ONLY differences are:

1. Input file naming: intermediate_h3r{resolution}.csv (instead of eps{epsilon})
2. Output file naming: diy_h3r{resolution}_prev{day}*.pk (instead of eps{epsilon})
3. Metadata field: h3_resolution (instead of epsilon)

All processing logic is EXACTLY THE SAME:
• Temporal splitting per user
• Location ID encoding with +2 offset
• Valid sequence filtering
• User filtering
• Parallel sequence generation
```

---

## Differences from DBSCAN Version

### Side-by-Side Comparison

```
DBSCAN VERSION                          H3 VERSION
═══════════════                         ═══════════════

File: diy_2_interim_to_processed.py    File: diy_h3_2_interim_to_processed.py

Config:                                 Config:
  config/preprocessing/diy.yaml           config/preprocessing/diy_h3.yaml

Input:                                  Input:
  intermediate_eps{epsilon}.csv           intermediate_h3r{resolution}.csv

Output:                                 Output:
  diy_eps{epsilon}_prev{day}_*.pk         diy_h3r{resolution}_prev{day}_*.pk

Metadata field:                         Metadata field:
  "epsilon": 50                           "h3_resolution": 8

ALL OTHER LOGIC: IDENTICAL              ALL OTHER LOGIC: IDENTICAL
```

### Code Differences (Only Naming)

```python
# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENCE 1: Configuration Parameter
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
epsilon = config['dataset']['epsilon']

# H3 version:
h3_resolution = config['dataset']['h3_resolution']


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENCE 2: Input File Path
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
interim_file = os.path.join(interim_path, f"intermediate_eps{epsilon}.csv")

# H3 version:
interim_file = os.path.join(interim_path, f"intermediate_h3r{h3_resolution}.csv")


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENCE 3: Output File Names
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
output_name = f"{dataset_name}_eps{epsilon}_prev{previous_day}"
# Example: "diy_eps50_prev7"

# H3 version:
output_name = f"{dataset_name}_h3r{h3_resolution}_prev{previous_day}"
# Example: "diy_h3r8_prev7"


# ═══════════════════════════════════════════════════════════════════════════════
# DIFFERENCE 4: Metadata Field
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
metadata = {
    "epsilon": epsilon,
    ...
}

# H3 version:
metadata = {
    "h3_resolution": h3_resolution,
    ...
}
```

---

## Complete Script Walkthrough

Since the logic is identical to the DBSCAN version, this section highlights the key processing stages with H3-specific context.

### Processing Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    H3 VERSION PROCESSING FLOW                                │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT: data/diy_h3r8/interim/intermediate_h3r8.csv
─────────────────────────────────────────────────────────────────────────────────

Note: The intermediate file contains location_ids that were assigned by H3 cells.
      The location_id values are already integers (0, 1, 2, ...).
      The H3 cell information is stored separately in locations_h3r8.csv.


STEP 1: Split Dataset (Temporal, Per-User)
─────────────────────────────────────────────────────────────────────────────────

        User Timeline:
        Day 0                                    Day 80    Day 90        Day 100
        ├──────────────────────────────────────────┼─────────┼────────────────┤
        │◄───────────── TRAIN (80%) ──────────────▶│◄─VAL ──▶│◄──── TEST ───▶│

        Same logic as DBSCAN version - split based on start_day


STEP 2: Encode Location IDs
─────────────────────────────────────────────────────────────────────────────────

        Original H3-based location_ids:  [0, 1, 2, 5, 10, 42, 100]
                                                 │
                                                 ▼
        OrdinalEncoder (fit on train):   [0, 1, 2, 3,  4,  5,   6]
                                                 │
                                                 ▼
        Final (+2 offset):               [2, 3, 4, 5,  6,  7,   8]

        0 = Padding
        1 = Unknown location
        2+ = Actual H3-based locations


STEP 3: Filter Valid Sequences
─────────────────────────────────────────────────────────────────────────────────

        Same previous_day and min_length requirements as DBSCAN version.
        
        The only difference is what "location" means:
        • DBSCAN: location_id = DBSCAN cluster ID
        • H3: location_id = H3 cell-based ID
        
        Both are just integer IDs - the filtering logic is identical.


STEP 4: Generate Sequences
─────────────────────────────────────────────────────────────────────────────────

        Same sequence dictionary structure:
        {
            "X":           [44, 17, 44, 10, 44, 17, 44],  # H3-based location IDs
            "user_X":      [1, 1, 1, 1, 1, 1, 1],
            "weekday_X":   [0, 1, 2, 3, 4, 5, 6],
            "start_min_X": [420, 540, 480, 600, 450, 540, 420],
            "dur_X":       [383, 210, 240, 105, 660, 210, 383],
            "diff":        [7, 6, 5, 4, 3, 2, 1],
            "Y":           17  # Target H3-based location
        }


OUTPUT: data/diy_h3r8/processed/
─────────────────────────────────────────────────────────────────────────────────

        ├── diy_h3r8_prev7_train.pk
        ├── diy_h3r8_prev7_validation.pk
        ├── diy_h3r8_prev7_test.pk
        └── diy_h3r8_prev7_metadata.json
```

---

## Input/Output Specification

### Input File

```
INPUT: intermediate_h3r{resolution}.csv
═══════════════════════════════════════════════════════════════════════════════

Location: data/diy_h3r{resolution}/interim/intermediate_h3r{resolution}.csv

Schema (IDENTICAL to DBSCAN version):
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ id             │ int64        │ Sequential staypoint ID                        │
│ user_id        │ int64        │ Integer user ID                                │
│ location_id    │ int64        │ H3 cell-based location ID                      │
│ start_day      │ int64        │ Days since user's first record                 │
│ end_day        │ int64        │ End day number                                 │
│ start_min      │ int64        │ Start minute of day (0-1439)                   │
│ end_min        │ int64        │ End minute of day (1-1440)                     │
│ weekday        │ int64        │ Day of week (0-6)                              │
│ duration       │ float64      │ Duration in minutes                            │
└────────────────┴──────────────┴────────────────────────────────────────────────┘

The ONLY difference from DBSCAN version:
• location_id values reference H3-based locations instead of DBSCAN clusters
• The actual values are just integers - processing is identical
```

### Output Files

```
OUTPUT FILES: Same structure as DBSCAN version
═══════════════════════════════════════════════════════════════════════════════

Location: data/diy_h3r{resolution}/processed/

Files:
├── diy_h3r{resolution}_prev{day}_train.pk        # Training sequences
├── diy_h3r{resolution}_prev{day}_validation.pk   # Validation sequences
├── diy_h3r{resolution}_prev{day}_test.pk         # Test sequences
└── diy_h3r{resolution}_prev{day}_metadata.json   # Dataset metadata


Metadata JSON (H3 version):
─────────────────────────────────────────────────────────────────────────────────

{
    "dataset_name": "diy",
    "output_dataset_name": "diy_h3r8_prev7",
    "h3_resolution": 8,                          // ← H3-specific field
    "previous_day": 7,
    "total_user_num": 156,
    "total_loc_num": 5678,                       // ← May differ from DBSCAN
    "unique_users": 152,
    "unique_locations": 5456,                    // ← May differ from DBSCAN
    "total_staypoints": 290123,
    "valid_staypoints": 245678,
    "train_staypoints": 196543,
    "val_staypoints": 24568,
    "test_staypoints": 24567,
    "train_sequences": 175432,
    "val_sequences": 21234,
    "test_sequences": 21456,
    "total_sequences": 218122,
    "split_ratios": {
        "train": 0.8,
        "val": 0.1,
        "test": 0.1
    },
    "max_duration_minutes": 2880
}
```

---

## Running the Script

### Command Line Usage

```bash
# Default H3 configuration (resolution=8, previous_day=7)
python preprocessing/diy_h3_2_interim_to_processed.py \
    --config config/preprocessing/diy_h3.yaml

# Example with custom config for different resolution
python preprocessing/diy_h3_2_interim_to_processed.py \
    --config config/preprocessing/diy_h3_r9.yaml
```

### Example Console Output

```
================================================================================
DIY PREPROCESSING (H3) - Script 2: Interim to Processed
================================================================================
[INPUT]  Interim folder: data/diy_h3r8/interim
[OUTPUT] Processed folder: data/diy_h3r8/processed/
[CONFIG] Dataset: diy, H3 Resolution: 8
[CONFIG] Previous days: [7]
================================================================================

[LOAD] Loading intermediate dataset...
Loaded 290123 staypoints from 155 users
Input file: data/diy_h3r8/interim/intermediate_h3r8.csv

------------------------------------------------------------
Processing for previous_day = 7
------------------------------------------------------------

[1/5] Splitting dataset into train/val/test...
Train: 232098, Val: 29012, Test: 29013

[2/5] Encoding location IDs...
Max location ID: 5678
Unique locations in train: 5456

[3/5] Filtering valid sequences (previous_day=7)...
Valid staypoints: 245678

[4/5] Filtering users with valid sequences in all splits...
Valid users (in all splits): 152

Final max location ID: 5432
Final unique locations: 5234
Final user count: 152

[5/5] Generating sequences and saving .pk files...
  Processing train sequences...
    train: 100%|████████████████████████| 152/152 [00:06<00:00, 24.5it/s]
  Generated 175432 train sequences
  Processing validation sequences...
    validation: 100%|████████████████████| 152/152 [00:01<00:00, 89.2it/s]
  Generated 21234 validation sequences
  Processing test sequences...
    test: 100%|██████████████████████████| 152/152 [00:01<00:00, 87.8it/s]
  Generated 21456 test sequences

✓ Saved train sequences to: data/diy_h3r8/processed/diy_h3r8_prev7_train.pk
✓ Saved validation sequences to: data/diy_h3r8/processed/diy_h3r8_prev7_validation.pk
✓ Saved test sequences to: data/diy_h3r8/processed/diy_h3r8_prev7_test.pk
✓ Saved metadata to: data/diy_h3r8/processed/diy_h3r8_prev7_metadata.json

================================================================================
SCRIPT 2 COMPLETE: Interim to Processed
================================================================================
Output folder: data/diy_h3r8/processed/

previous_day=7:
  Train: 175432, Val: 21234, Test: 21456
  Total users: 156, Total locations: 5678
================================================================================
```

---

## Integration with Model Training

### Loading H3-Based Data

```python
import pickle
import json

# Load H3-processed training data
with open("data/diy_h3r8/processed/diy_h3r8_prev7_train.pk", "rb") as f:
    train_data = pickle.load(f)

# Load metadata for model configuration
with open("data/diy_h3r8/processed/diy_h3r8_prev7_metadata.json", "r") as f:
    metadata = json.load(f)

# Configure model
num_locations = metadata["total_loc_num"]  # For embedding layer size
num_users = metadata["total_user_num"]      # For user embedding size

print(f"Training samples: {len(train_data)}")
print(f"Location vocabulary size: {num_locations}")
print(f"User count: {num_users}")
```

### Comparing DBSCAN and H3 Outputs

```python
import pickle
import json

# Load both versions
with open("data/diy_eps50/processed/diy_eps50_prev7_metadata.json", "r") as f:
    dbscan_meta = json.load(f)

with open("data/diy_h3r8/processed/diy_h3r8_prev7_metadata.json", "r") as f:
    h3_meta = json.load(f)

# Compare statistics
print("Comparison: DBSCAN vs H3")
print("-" * 50)
print(f"Locations:  {dbscan_meta['total_loc_num']} vs {h3_meta['total_loc_num']}")
print(f"Users:      {dbscan_meta['unique_users']} vs {h3_meta['unique_users']}")
print(f"Train seq:  {dbscan_meta['train_sequences']} vs {h3_meta['train_sequences']}")
print(f"Val seq:    {dbscan_meta['val_sequences']} vs {h3_meta['val_sequences']}")
print(f"Test seq:   {dbscan_meta['test_sequences']} vs {h3_meta['test_sequences']}")

# Typical output:
# Comparison: DBSCAN vs H3
# --------------------------------------------------
# Locations:  4523 vs 5678      (H3 typically has more)
# Users:      152 vs 152        (Same user filtering logic)
# Train seq:  65234 vs 175432   (May vary based on locations)
# Val seq:    8123 vs 21234
# Test seq:   8234 vs 21456
```

### Using with PyTorch DataLoader

```python
import pickle
import torch
from torch.utils.data import Dataset, DataLoader

class LocationSequenceDataset(Dataset):
    """Dataset class works with both DBSCAN and H3 versions."""
    
    def __init__(self, pickle_path):
        with open(pickle_path, "rb") as f:
            self.data = pickle.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        return {
            "X": torch.LongTensor(sample["X"]),
            "user_X": torch.LongTensor(sample["user_X"]),
            "weekday_X": torch.LongTensor(sample["weekday_X"]),
            "start_min_X": torch.LongTensor(sample["start_min_X"]),
            "dur_X": torch.FloatTensor(sample["dur_X"]),
            "diff": torch.LongTensor(sample["diff"]),
            "Y": torch.LongTensor([sample["Y"]])
        }

# Works with either version!
# H3 version:
train_dataset = LocationSequenceDataset("data/diy_h3r8/processed/diy_h3r8_prev7_train.pk")

# DBSCAN version:
# train_dataset = LocationSequenceDataset("data/diy_eps50/processed/diy_eps50_prev7_train.pk")

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Iterate through batches
for batch in train_loader:
    X = batch["X"]           # (batch_size, seq_len)
    Y = batch["Y"]           # (batch_size, 1)
    # ... training code
```

---

## Summary

The `diy_h3_2_interim_to_processed.py` script:

1. **Is functionally identical** to the DBSCAN version
2. **Only differs in file naming** (h3r{resolution} instead of eps{epsilon})
3. **Produces same output format** - sequences usable by same models
4. **Uses same processing logic** - splitting, encoding, filtering, generation

Key points:
- H3 locations are just integer IDs (like DBSCAN)
- All downstream processing treats them identically
- Models trained on DBSCAN data work with H3 data (and vice versa)
- Only metadata field differs: `h3_resolution` vs `epsilon`

When to use H3 version:
- When you used H3 Script 1 for location generation
- When you need reproducible location assignments
- When comparing across multiple datasets with same H3 grid
