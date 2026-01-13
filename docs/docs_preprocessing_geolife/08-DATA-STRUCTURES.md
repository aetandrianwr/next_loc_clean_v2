# Data Structures Reference - Complete Format Specifications

## Table of Contents
1. [Overview](#overview)
2. [Input Data Formats](#input-data-formats)
3. [Intermediate Data Formats](#intermediate-data-formats)
4. [Output Data Formats](#output-data-formats)
5. [Data Type Reference](#data-type-reference)
6. [Sample Data with Realistic Values](#sample-data-with-realistic-values)

---

## Overview

This document provides detailed specifications for all data formats used throughout the preprocessing pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DATA FORMAT OVERVIEW                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  PIPELINE STAGE           FORMAT                     PRIMARY FILES               │
│  ──────────────           ──────                     ─────────────               │
│                                                                                  │
│  1. Raw Input             .plt (text)                *.plt trajectory files      │
│                                                                                  │
│  2. Script 1 Internal     GeoDataFrame               In-memory processing        │
│     (position fixes)      (trackintel format)                                    │
│                                                                                  │
│  3. Script 1 Output       CSV (pandas)               intermediate_eps{X}.csv     │
│     (intermediate)                                   locations_eps{X}.csv        │
│                                                                                  │
│  4. Script 2 Internal     DataFrame                  In-memory processing        │
│     (encoded data)        (pandas)                                               │
│                                                                                  │
│  5. Script 2 Output       Pickle (.pk)               *_train.pk, *_test.pk       │
│     (sequences)           JSON (.json)               *_metadata.json             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Input Data Formats

### 1. PLT File Format (Raw GPS Trajectories)

**Location**: `data/raw_geolife/Data/{user_id}/Trajectory/*.plt`

**Structure**:
```
Line 1-6: Header (skip)
Line 7+: Comma-separated GPS data
```

**Field Specification**:

| Position | Field | Type | Description | Example |
|----------|-------|------|-------------|---------|
| 0 | Latitude | float | Decimal degrees (WGS84) | 39.984702 |
| 1 | Longitude | float | Decimal degrees (WGS84) | 116.318417 |
| 2 | Reserved | int | Always 0 | 0 |
| 3 | Altitude | float | Feet (-777 = invalid) | 492.0 |
| 4 | DateDays | float | Days since Dec 30, 1899 | 39744.1201 |
| 5 | Date | string | YYYY-MM-DD | 2008-10-23 |
| 6 | Time | string | HH:MM:SS | 02:53:04 |

**Complete Example**:
```
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.984702,116.318417,0,492,39744.1201157407,2008-10-23,02:53:04
39.984683,116.318450,0,492,39744.1201388889,2008-10-23,02:53:06
39.984686,116.318417,0,492,39744.1201504630,2008-10-23,02:53:07
39.984688,116.318385,0,492,39744.1201620370,2008-10-23,02:53:08
```

### 2. Labels File Format (Optional)

**Location**: `data/raw_geolife/Data/{user_id}/labels.txt`

**Structure**:
```
Start Time,End Time,Transportation Mode
2008/04/02 11:24:21,2008/04/02 11:50:45,bus
```

**Note**: Labels are not used in the preprocessing pipeline.

---

## Intermediate Data Formats

### 1. Position Fixes (GeoDataFrame)

**Created by**: `read_geolife()` in Script 1

**Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| user_id | int64 | User identifier | 0 |
| tracked_at | datetime64 | Timestamp of GPS fix | 2008-10-23 02:53:04 |
| geom | Point | Geometry (lat, lon) | POINT(116.318 39.985) |
| elevation | float64 | Altitude in meters | 149.96 |

**Example Data**:
```python
   user_id          tracked_at                      geom  elevation
0        0 2008-10-23 02:53:04  POINT (116.318 39.985)     149.96
1        0 2008-10-23 02:53:06  POINT (116.318 39.985)     149.96
2        0 2008-10-23 02:53:07  POINT (116.318 39.985)     149.96
```

### 2. Staypoints (GeoDataFrame)

**Created by**: `generate_staypoints()` in Script 1

**Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| user_id | int64 | User identifier | 0 |
| started_at | datetime64 | Start of staypoint | 2008-10-23 09:00:00 |
| finished_at | datetime64 | End of staypoint | 2008-10-23 17:00:00 |
| geom | Point | Centroid geometry | POINT(116.318 39.985) |
| is_activity | bool | Activity flag | True |
| location_id | int64/NaN | Assigned location | 15 or NaN |

**Example Data**:
```python
   user_id          started_at         finished_at                      geom  is_activity  location_id
0        0 2008-10-23 09:00:00 2008-10-23 17:00:00  POINT (116.318 39.985)         True         15.0
1        0 2008-10-23 18:30:00 2008-10-23 23:00:00  POINT (116.320 39.990)         True         12.0
2        0 2008-10-24 08:00:00 2008-10-24 12:00:00  POINT (116.318 39.984)         True         15.0
```

### 3. Locations (GeoDataFrame/DataFrame)

**Created by**: `generate_locations()` (DBSCAN) or `generate_h3_locations()` (H3)

**DBSCAN Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| location_id | int64 (index) | Location identifier | 0 |
| center | Point | Cluster centroid | POINT(116.318 39.985) |
| extent | Polygon | Cluster boundary | POLYGON(...) |

**H3 Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| location_id | int64 (index) | Location identifier | 0 |
| h3_cell | string | H3 cell index | 88283082b9fffff |
| center_lat | float64 | Cell center latitude | 39.9847 |
| center_lng | float64 | Cell center longitude | 116.3184 |
| h3_resolution | int64 | H3 resolution used | 8 |

### 4. Intermediate CSV (Final Output of Script 1)

**File**: `data/geolife_eps{X}/interim/intermediate_eps{X}.csv`

**Schema**:

| Column | Type | Description | Range | Example |
|--------|------|-------------|-------|---------|
| id | int64 | Unique staypoint ID | 0 to N-1 | 0 |
| user_id | int64 | User identifier | 0 to U-1 | 1 |
| location_id | int64 | Location identifier | 0 to L-1 | 15 |
| start_day | int64 | Days since user's first record | 0 to D | 5 |
| end_day | int64 | Day when staypoint ended | 0 to D | 5 |
| start_min | int64 | Start time in minutes from midnight | 0 to 1439 | 540 |
| end_min | int64 | End time in minutes from midnight | 1 to 1440 | 1020 |
| weekday | int64 | Day of week | 0 (Mon) to 6 (Sun) | 3 |
| duration | int64 | Duration in minutes | 1 to max_duration | 480 |

**Example CSV**:
```csv
id,user_id,location_id,start_day,end_day,start_min,end_min,weekday,duration
0,1,15,0,0,540,1020,0,480
1,1,12,0,0,1080,1380,0,300
2,1,15,1,1,510,1050,1,540
3,1,8,1,1,1110,1320,1,210
4,1,15,2,2,480,1020,2,540
```

---

## Output Data Formats

### 1. Sequence Pickle File (.pk)

**Files**: 
- `data/geolife_eps{X}/processed/geolife_eps{X}_prev{Y}_train.pk`
- `data/geolife_eps{X}/processed/geolife_eps{X}_prev{Y}_validation.pk`
- `data/geolife_eps{X}/processed/geolife_eps{X}_prev{Y}_test.pk`

**Format**: Python list of dictionaries

**Structure**:
```python
[
    {
        "X": numpy.ndarray,           # Location history
        "user_X": numpy.ndarray,      # User IDs for each history point
        "weekday_X": numpy.ndarray,   # Weekday for each history point
        "start_min_X": numpy.ndarray, # Start time for each history point
        "dur_X": numpy.ndarray,       # Duration for each history point
        "diff": numpy.ndarray,        # Days ago for each history point
        "Y": int                      # Target location ID
    },
    ...
]
```

**Detailed Field Specification**:

| Key | Type | Shape | Range | Description |
|-----|------|-------|-------|-------------|
| X | ndarray[int64] | (seq_len,) | 2 to total_loc_num | Location IDs (0=pad, 1=unknown) |
| user_X | ndarray[int64] | (seq_len,) | 1 to total_user_num | User IDs (0=pad) |
| weekday_X | ndarray[int64] | (seq_len,) | 0 to 6 | Monday=0, Sunday=6 |
| start_min_X | ndarray[int64] | (seq_len,) | 0 to 1439 | Minutes from midnight |
| dur_X | ndarray[int64] | (seq_len,) | 1 to max_duration | Duration in minutes |
| diff | ndarray[int64] | (seq_len,) | 1 to previous_day | Days before target |
| Y | int | scalar | 2 to total_loc_num | Target location ID |

**Complete Example Sequence**:
```python
{
    # History of 5 location visits in the past 7 days
    "X": array([4, 3, 4, 2, 4]),              # Visited locations 4, 3, 4, 2, 4
    "user_X": array([1, 1, 1, 1, 1]),          # All from user 1
    "weekday_X": array([2, 3, 4, 5, 6]),       # Wed, Thu, Fri, Sat, Sun
    "start_min_X": array([540, 510, 480, 600, 720]),  # 9:00, 8:30, 8:00, 10:00, 12:00
    "dur_X": array([480, 540, 300, 120, 180]), # 8h, 9h, 5h, 2h, 3h
    "diff": array([7, 6, 5, 3, 1]),            # 7, 6, 5, 3, 1 days ago
    "Y": 3                                     # Predict location 3
}
```

### 2. Metadata JSON File

**File**: `data/geolife_eps{X}/processed/geolife_eps{X}_prev{Y}_metadata.json`

**Schema**:

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
    "split_ratios": {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2
    },
    "max_duration_minutes": 2880
}
```

**Field Descriptions**:

| Field | Type | Description |
|-------|------|-------------|
| dataset_name | string | Source dataset name |
| output_dataset_name | string | Full output name with parameters |
| epsilon/h3_resolution | int | Location clustering parameter |
| previous_day | int | History window in days |
| total_user_num | int | Max user ID + 1 (for embedding size) |
| total_loc_num | int | Max location ID + 1 (for embedding size) |
| unique_users | int | Actual number of unique users |
| unique_locations | int | Actual number of unique locations |
| total_staypoints | int | Total staypoints in filtered data |
| valid_staypoints | int | Staypoints that can be prediction targets |
| train/val/test_staypoints | int | Staypoints in each split |
| train/val/test_sequences | int | Generated sequences in each split |
| total_sequences | int | Sum of all sequences |
| split_ratios | dict | Train/val/test split ratios |
| max_duration_minutes | int | Maximum duration cap value |

---

## Data Type Reference

### ID Encoding Schemes

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ID ENCODING REFERENCE                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  LOCATION IDs (in output sequences):                                             │
│  ─────────────────────────────────────                                           │
│  ID │ Meaning                                                                    │
│   0 │ Padding token (for variable-length sequences)                              │
│   1 │ Unknown location (location not seen in training)                           │
│   2 │ First encoded location                                                     │
│   3 │ Second encoded location                                                    │
│  ...│ ...                                                                        │
│ N+1 │ Last encoded location (where N = number of unique locations)               │
│                                                                                  │
│  USER IDs (in output sequences):                                                 │
│  ──────────────────────────────────                                              │
│  ID │ Meaning                                                                    │
│   0 │ Padding token                                                              │
│   1 │ First encoded user                                                         │
│   2 │ Second encoded user                                                        │
│  ...│ ...                                                                        │
│   N │ Last encoded user                                                          │
│                                                                                  │
│  WEEKDAY (unchanged):                                                            │
│  ────────────────────                                                            │
│  ID │ Day                                                                        │
│   0 │ Monday                                                                     │
│   1 │ Tuesday                                                                    │
│   2 │ Wednesday                                                                  │
│   3 │ Thursday                                                                   │
│   4 │ Friday                                                                     │
│   5 │ Saturday                                                                   │
│   6 │ Sunday                                                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Time Encoding

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    TIME ENCODING REFERENCE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  start_min / end_min: Minutes from midnight                                      │
│  ─────────────────────────────────────────                                       │
│                                                                                  │
│  Time      │ Minutes │ Calculation                                               │
│  00:00     │    0    │ 0 * 60 + 0                                                │
│  01:00     │   60    │ 1 * 60 + 0                                                │
│  08:00     │  480    │ 8 * 60 + 0                                                │
│  08:30     │  510    │ 8 * 60 + 30                                               │
│  09:00     │  540    │ 9 * 60 + 0                                                │
│  12:00     │  720    │ 12 * 60 + 0                                               │
│  17:00     │ 1020    │ 17 * 60 + 0                                               │
│  18:30     │ 1110    │ 18 * 60 + 30                                              │
│  23:59     │ 1439    │ 23 * 60 + 59                                              │
│  24:00*    │ 1440    │ Used only for end_min at midnight                         │
│                                                                                  │
│  *Special case: end_min = 0 (midnight) is converted to 1440                      │
│                                                                                  │
│  start_day / end_day: Days since user's first record                             │
│  ────────────────────────────────────────────────────                            │
│                                                                                  │
│  User first tracked: 2008-10-20                                                  │
│  Current staypoint:  2008-10-25                                                  │
│  start_day = (2008-10-25) - (2008-10-20) = 5                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Sample Data with Realistic Values

### Complete Processing Example

#### Input: Raw PLT Data

```
# User 001, File: 20081023090000.plt
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.984702,116.318417,0,492,39744.375,2008-10-23,09:00:00
39.984703,116.318418,0,492,39744.375,2008-10-23,09:00:05
... (continuous GPS points for 8 hours)
39.984701,116.318420,0,492,39744.708,2008-10-23,17:00:00
```

#### After Script 1: Intermediate CSV

```csv
id,user_id,location_id,start_day,end_day,start_min,end_min,weekday,duration
0,1,15,0,0,540,1020,3,480
1,1,12,0,0,1080,1380,3,300
2,1,15,1,1,510,1050,4,540
3,1,8,2,2,600,840,5,240
4,1,15,3,3,480,1020,6,540
5,1,12,3,3,1050,1350,6,300
6,1,3,4,4,540,720,0,180
7,1,15,5,5,480,1050,1,570
8,1,12,5,5,1110,1380,1,270
9,1,15,6,6,510,1020,2,510
10,1,8,6,6,1080,1260,2,180
11,1,15,7,7,480,1020,3,540
12,1,12,7,7,1080,1380,3,300
```

**Interpretation**:
- User 1 has 13 staypoints over 8 days (days 0-7)
- Location 15 appears frequently (likely home or work)
- Location 12 appears in evenings (likely home)
- Location 8 appears occasionally (maybe gym or restaurant)
- Location 3 appears once on day 4 (one-off visit)

#### After Script 2: Training Sequence

For predicting staypoint at day 7 (id=12), with previous_day=7:

```python
# History: All staypoints from day 0-6 before target at day 7
# (staypoints with id 0-10)

sequence = {
    # Location visits before target (encoded IDs after +2 offset)
    "X": array([17, 14, 17, 10, 17, 14, 5, 17, 14, 17, 10]),  # 11 visits
    
    # User (all same user, encoded ID = 2 after +1 offset)
    "user_X": array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]),
    
    # Weekdays (Thu=3, Fri=4, Sat=5, Sun=6, Mon=0, Tue=1, Wed=2)
    "weekday_X": array([3, 3, 4, 5, 6, 6, 0, 1, 1, 2, 2]),
    
    # Start times (in minutes from midnight)
    "start_min_X": array([540, 1080, 510, 600, 480, 1050, 540, 480, 1110, 510, 1080]),
    
    # Durations (in minutes)
    "dur_X": array([480, 300, 540, 240, 540, 300, 180, 570, 270, 510, 180]),
    
    # Days before target (day 7)
    "diff": array([7, 7, 6, 5, 4, 4, 3, 2, 2, 1, 1]),
    
    # Target: Location 12 (encoded as 14)
    "Y": 14
}
```

---

## Loading and Using Data

### Python Code Examples

```python
import pickle
import json
import pandas as pd

# 1. Load intermediate CSV
intermediate = pd.read_csv("data/geolife_eps20/interim/intermediate_eps20.csv")
print(intermediate.head())
print(f"Users: {intermediate['user_id'].nunique()}")
print(f"Locations: {intermediate['location_id'].nunique()}")

# 2. Load training sequences
with open("data/geolife_eps20/processed/geolife_eps20_prev7_train.pk", "rb") as f:
    train_sequences = pickle.load(f)

print(f"Number of training sequences: {len(train_sequences)}")
print(f"Sample sequence: {train_sequences[0]}")

# 3. Load metadata
with open("data/geolife_eps20/processed/geolife_eps20_prev7_metadata.json", "r") as f:
    metadata = json.load(f)

print(f"Total locations (for embedding): {metadata['total_loc_num']}")
print(f"Total users (for embedding): {metadata['total_user_num']}")

# 4. Create PyTorch embeddings
import torch.nn as nn

loc_embedding = nn.Embedding(
    num_embeddings=metadata['total_loc_num'],  # Includes padding and unknown
    embedding_dim=64
)
user_embedding = nn.Embedding(
    num_embeddings=metadata['total_user_num'],  # Includes padding
    embedding_dim=32
)
```

---

## Next Steps

- [09-FUNCTIONS-REFERENCE.md](09-FUNCTIONS-REFERENCE.md) - Complete function reference
- [10-EXAMPLES.md](10-EXAMPLES.md) - More detailed examples

---

*Documentation Version: 1.0*
*For PhD Research Reference*
