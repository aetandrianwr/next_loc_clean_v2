# Concrete Examples - End-to-End Walkthrough

## Table of Contents
1. [Introduction](#introduction)
2. [Example User: Alice (User 001)](#example-user-alice-user-001)
3. [Script 1 Step-by-Step](#script-1-step-by-step)
4. [Script 2 Step-by-Step](#script-2-step-by-step)
5. [Final Output Verification](#final-output-verification)
6. [PyTorch Usage Example](#pytorch-usage-example)

---

## Introduction

This document traces a **complete, consistent example** through the entire preprocessing pipeline using a fictional but realistic user "Alice" (User 001).

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    EXAMPLE OVERVIEW: ALICE'S DATA                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User: Alice (user_id = 001)                                                    │
│  Location: Beijing, China                                                        │
│  Tracking Period: 120 days (Oct 1, 2008 - Jan 28, 2009)                         │
│                                                                                  │
│  Alice's Regular Places:                                                         │
│  ─────────────────────────                                                       │
│  🏠 Home:       39.9847°N, 116.3184°E  (Haidian District)                       │
│  🏢 Office:     39.9892°N, 116.3210°E  (Microsoft Research Asia)                │
│  🍜 Restaurant: 39.9756°N, 116.3098°E  (Local noodle shop)                      │
│  🏋️ Gym:        39.9801°N, 116.3150°E  (Fitness center)                          │
│                                                                                  │
│  Alice's Weekly Routine:                                                         │
│  ───────────────────────                                                         │
│  Mon-Fri: Home → Office (8am-6pm) → Home                                        │
│  Tue/Thu: Home → Office → Gym (7pm-8pm) → Home                                  │
│  Sat: Home → Restaurant (12pm) → Shopping → Home                                 │
│  Sun: Home (all day)                                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Example User: Alice (User 001)

### Raw GPS Data Sample

**File**: `Data/001/Trajectory/20081001080000.plt`

This file contains GPS recordings from Alice's commute to work on October 1, 2008.

```
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.984702,116.318417,0,492,39722.333333,2008-10-01,08:00:00
39.984710,116.318420,0,492,39722.333403,2008-10-01,08:00:06
39.984715,116.318425,0,492,39722.333472,2008-10-01,08:00:12
... (GPS points every ~6 seconds during travel)
39.989150,116.320980,0,498,39722.354167,2008-10-01,08:30:00
39.989155,116.320985,0,498,39722.354236,2008-10-01,08:30:06
... (GPS points while stationary at office)
39.989152,116.320982,0,498,39722.750000,2008-10-01,18:00:00
```

### Alice's Complete Day (October 1, 2008)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ALICE'S DAY: OCTOBER 1, 2008 (WEDNESDAY)                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Time         Activity                          Location                         │
│  ────         ────────                          ────────                         │
│  00:00-07:30  Sleeping                          🏠 Home                          │
│  07:30-08:00  Getting ready, GPS not active     (no data)                        │
│  08:00-08:30  Commuting to work                 Movement (GPS active)            │
│  08:30-18:00  Working                           🏢 Office                        │
│  18:00-18:30  Commuting home                    Movement (GPS active)            │
│  18:30-23:59  Evening at home                   🏠 Home                          │
│                                                                                  │
│  Timeline visualization:                                                         │
│                                                                                  │
│  00:00        08:00   08:30                    18:00  18:30        24:00         │
│    │────────────│═════════│────────────────────│═════════│────────────│         │
│    │    Home    │ Travel  │      Office        │ Travel  │    Home    │         │
│    │  (no GPS)  │  (GPS)  │       (GPS)        │  (GPS)  │   (GPS)    │         │
│                                                                                  │
│  Staypoints generated:                                                           │
│  SP1: Home     00:00-07:30 (if GPS was active, but likely not)                  │
│  SP2: Office   08:30-18:00 (9.5 hours) ✓                                        │
│  SP3: Home     18:30-23:59 (5.5 hours) ✓                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Script 1 Step-by-Step

### Step 1: Read Raw GPS Data

```python
pfs, _ = read_geolife("data/raw_geolife", print_progress=True)
```

**Alice's portion of position fixes (pfs)**:

```
   user_id          tracked_at                        geom  elevation
0        1 2008-10-01 08:00:00  POINT(116.318417 39.984702)     149.96
1        1 2008-10-01 08:00:06  POINT(116.318420 39.984710)     149.96
2        1 2008-10-01 08:00:12  POINT(116.318425 39.984715)     149.96
...
500      1 2008-10-01 08:30:00  POINT(116.320980 39.989150)     151.79
501      1 2008-10-01 08:30:06  POINT(116.320985 39.989155)     151.79
...
(~50,000 GPS points for Alice over 120 days)
```

### Step 2: Generate Staypoints

```python
pfs, sp = pfs.as_positionfixes.generate_staypoints(
    gap_threshold=1440,    # 24 hours
    dist_threshold=200,    # 200 meters
    time_threshold=30,     # 30 minutes minimum
    include_last=True,
    n_jobs=-1
)
```

**Algorithm in action for October 1**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAYPOINT DETECTION: OCTOBER 1                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  08:00 GPS at Home (39.9847, 116.3184)                                          │
│  08:06 GPS at (39.9848, 116.3185) - 15m from 08:00 point                        │
│  08:12 GPS at (39.9850, 116.3188) - 45m from 08:00 point                        │
│  ...                                                                             │
│  08:25 GPS at (39.9880, 116.3200) - 400m from 08:00 point                       │
│        └─► Distance > 200m, user is MOVING (not a staypoint)                     │
│                                                                                  │
│  08:30 GPS at Office (39.9892, 116.3210)                                        │
│  08:36 GPS at (39.9892, 116.3210) - 5m from 08:30 point                         │
│  08:42 GPS at (39.9891, 116.3211) - 12m from 08:30 point                        │
│  ...                                                                             │
│  18:00 GPS at (39.9892, 116.3210) - still within 200m                           │
│        └─► Duration = 9.5 hours > 30 min threshold                               │
│        └─► CREATE STAYPOINT: Office, 08:30-18:00                                 │
│                                                                                  │
│  18:30 GPS at Home (39.9847, 116.3184)                                          │
│  ...                                                                             │
│  23:59 GPS at (39.9847, 116.3185)                                               │
│        └─► Duration = 5.5 hours > 30 min threshold                               │
│        └─► CREATE STAYPOINT: Home, 18:30-23:59                                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Alice's staypoints (first 10 days)**:

```
   user_id          started_at          finished_at                        geom
0        1 2008-10-01 08:30:00 2008-10-01 18:00:00  POINT(116.3210 39.9892)  # Office
1        1 2008-10-01 18:30:00 2008-10-01 23:59:00  POINT(116.3184 39.9847)  # Home
2        1 2008-10-02 08:30:00 2008-10-02 18:30:00  POINT(116.3210 39.9892)  # Office
3        1 2008-10-02 19:00:00 2008-10-02 20:00:00  POINT(116.3150 39.9801)  # Gym
4        1 2008-10-02 20:30:00 2008-10-02 23:59:00  POINT(116.3184 39.9847)  # Home
5        1 2008-10-03 08:30:00 2008-10-03 18:00:00  POINT(116.3210 39.9892)  # Office
6        1 2008-10-03 18:30:00 2008-10-03 23:59:00  POINT(116.3184 39.9847)  # Home
...
```

### Step 3: Create Activity Flag

```python
sp = sp.as_staypoints.create_activity_flag(
    method="time_threshold",
    time_threshold=25  # 25 minutes
)
```

**Result**: All staypoints above have duration > 25 min, so `is_activity = True` for all.

### Step 4: Quality Filter

```python
# Alice's quality assessment
# 120 days of tracking, average coverage ~70%
# Passes day_filter (120 > 50) ✓
# Passes quality threshold ✓
```

**Alice passes the quality filter** and is included in valid_users.

### Step 5: Filter Activity Staypoints

All of Alice's staypoints have `is_activity = True`, so none are filtered.

### Step 6: Generate Locations (DBSCAN)

```python
sp, locs = sp.as_staypoints.generate_locations(
    epsilon=20,          # 20 meters
    num_samples=2,       # minimum 2 visits
    distance_metric="haversine",
    agg_level="dataset"
)
```

**Location clustering for Alice's staypoints**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DBSCAN CLUSTERING                                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Staypoint coordinates:                                                          │
│                                                                                  │
│  🏢 Office cluster:                                                              │
│     (39.9892, 116.3210) - 60 visits (Mon-Fri, 12 weeks)                         │
│     (39.9891, 116.3211) - GPS variation                                          │
│     (39.9892, 116.3209) - GPS variation                                          │
│     All within 20m → CLUSTER → location_id = 0                                   │
│                                                                                  │
│  🏠 Home cluster:                                                                │
│     (39.9847, 116.3184) - 100 visits                                             │
│     (39.9848, 116.3185) - GPS variation                                          │
│     All within 20m → CLUSTER → location_id = 1                                   │
│                                                                                  │
│  🏋️ Gym cluster:                                                                  │
│     (39.9801, 116.3150) - 24 visits (Tue/Thu)                                   │
│     All within 20m → CLUSTER → location_id = 2                                   │
│                                                                                  │
│  🍜 Restaurant cluster:                                                          │
│     (39.9756, 116.3098) - 12 visits (Saturdays)                                 │
│     All within 20m → CLUSTER → location_id = 3                                   │
│                                                                                  │
│  ✗ Random one-off visits → No cluster → location_id = NaN (filtered)            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Alice's staypoints with location_id**:

```
   user_id          started_at          finished_at  location_id
0        1 2008-10-01 08:30:00 2008-10-01 18:00:00            0  # Office
1        1 2008-10-01 18:30:00 2008-10-01 23:59:00            1  # Home
2        1 2008-10-02 08:30:00 2008-10-02 18:30:00            0  # Office
3        1 2008-10-02 19:00:00 2008-10-02 20:00:00            2  # Gym
4        1 2008-10-02 20:30:00 2008-10-02 23:59:00            1  # Home
...
```

### Step 7: Enrich Time Info

**Final intermediate data for Alice**:

```
intermediate_eps20.csv (Alice's portion):

id,user_id,location_id,start_day,end_day,start_min,end_min,weekday,duration
0,1,0,0,0,510,1080,2,570       # Oct 1: Office (Wed)
1,1,1,0,0,1110,1439,2,329      # Oct 1: Home evening (Wed)
2,1,0,1,1,510,1110,3,600       # Oct 2: Office (Thu)
3,1,2,1,1,1140,1200,3,60       # Oct 2: Gym (Thu)
4,1,1,1,1,1230,1439,3,209      # Oct 2: Home evening (Thu)
5,1,0,2,2,510,1080,4,570       # Oct 3: Office (Fri)
6,1,1,2,2,1110,1439,4,329      # Oct 3: Home evening (Fri)
7,1,3,3,3,720,840,5,120        # Oct 4: Restaurant (Sat)
8,1,1,3,3,900,1439,5,539       # Oct 4: Home (Sat)
9,1,1,4,4,0,1439,6,1439        # Oct 5: Home all day (Sun)
...
```

**Interpretation**:
- `start_day=0` is October 1, 2008 (Alice's first tracked day)
- `start_min=510` = 8:30 AM (8×60 + 30)
- `weekday=2` = Wednesday (0=Mon)
- `duration=570` = 9.5 hours in minutes

---

## Script 2 Step-by-Step

### Step 1: Split Dataset

Alice has 120 days of data (days 0-119):
- Train: Days 0-71 (60%)
- Val: Days 72-95 (20%)
- Test: Days 96-119 (20%)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    ALICE'S DATA SPLIT                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Day:    0        24        48        72        96        119                   │
│          │─────────│─────────│─────────│─────────│─────────│                    │
│          │◄─────────── Train (60%) ─────────────►│                              │
│          │         Oct 1 - Dec 11                │                              │
│                                                  │◄─ Val (20%) ─►│              │
│                                                  │  Dec 12 - Jan 4│              │
│                                                                   │◄─ Test ─►│  │
│                                                                   │ Jan 5-28 │  │
│                                                                                  │
│  Alice's staypoints by split:                                                    │
│  Train: 180 staypoints (72 days × ~2.5 per day)                                 │
│  Val:   60 staypoints (24 days × ~2.5 per day)                                  │
│  Test:  60 staypoints (24 days × ~2.5 per day)                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Step 2: Encode Locations

Original location_id → Encoded location_id (fit on train only):

```
Original → Encoded (+2)
   0     →    2    (Office)
   1     →    3    (Home)
   2     →    4    (Gym)
   3     →    5    (Restaurant)

Reserved IDs:
   0 = Padding
   1 = Unknown (location not in training)
```

### Step 3: Filter Valid Sequences

For `previous_day=7`, a staypoint is valid if:
1. It's at least 7 days after user's first record
2. It has at least 3 staypoints in the past 7 days

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VALID SEQUENCE CHECK                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Day 0-6: Cannot be targets (not enough history)                                │
│                                                                                  │
│  Day 7 (Oct 8), first valid check:                                              │
│  ─────────────────────────────────                                               │
│  Target: Staypoint at day 7, Office visit at 08:30                              │
│  History window: Days 0-6 (past 7 days)                                         │
│  Staypoints in window: 14 staypoints (from days 0-6)                            │
│  14 >= 3 minimum ✓                                                              │
│  → VALID TARGET                                                                  │
│                                                                                  │
│  Day 8 (Oct 9), second check:                                                   │
│  ─────────────────────────────                                                   │
│  Target: Staypoint at day 8, Office visit                                       │
│  History window: Days 1-7 (past 7 days)                                         │
│  Staypoints in window: 14 staypoints                                            │
│  → VALID TARGET                                                                  │
│                                                                                  │
│  ... (continues for all staypoints after day 7)                                 │
│                                                                                  │
│  Alice's valid staypoints: ~250 (out of 300 total)                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Step 4: Generate Sequences

Let's generate a specific sequence for Alice:

**Target**: Day 10 (October 11), Morning office visit

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SEQUENCE GENERATION: Day 10 Target                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Target staypoint:                                                               │
│  • Day 10, location_id = 2 (Office, encoded)                                    │
│  • start_min = 510 (08:30)                                                       │
│  • weekday = 5 (Saturday? No, recalculate: Oct 11 is Saturday)                  │
│    Actually Oct 11, 2008 is Saturday, but office visit? Let's say Oct 13 (Mon)  │
│                                                                                  │
│  Let's use Day 12 (Oct 13, Monday) as target instead:                           │
│  ─────────────────────────────────────────────────────                           │
│  Target: Day 12, Office visit                                                    │
│  History window: Days 5-11 (past 7 days)                                         │
│                                                                                  │
│  History staypoints (days 5-11):                                                 │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ day │ loc_id │ weekday │ start_min │ duration │                          │   │
│  │  5  │   2    │    4    │    510    │   570    │ Office (Fri)             │   │
│  │  5  │   3    │    4    │   1110    │   329    │ Home (Fri evening)       │   │
│  │  6  │   5    │    5    │    720    │   120    │ Restaurant (Sat)         │   │
│  │  6  │   3    │    5    │    900    │   539    │ Home (Sat)               │   │
│  │  7  │   3    │    6    │      0    │  1439    │ Home (Sun, all day)      │   │
│  │  8  │   2    │    0    │    510    │   570    │ Office (Mon)             │   │
│  │  8  │   3    │    0    │   1110    │   329    │ Home (Mon evening)       │   │
│  │  9  │   2    │    1    │    510    │   600    │ Office (Tue)             │   │
│  │  9  │   4    │    1    │   1140    │    60    │ Gym (Tue)                │   │
│  │  9  │   3    │    1    │   1230    │   209    │ Home (Tue evening)       │   │
│  │ 10  │   2    │    2    │    510    │   570    │ Office (Wed)             │   │
│  │ 10  │   3    │    2    │   1110    │   329    │ Home (Wed evening)       │   │
│  │ 11  │   2    │    3    │    510    │   600    │ Office (Thu)             │   │
│  │ 11  │   4    │    3    │   1140    │    60    │ Gym (Thu)                │   │
│  │ 11  │   3    │    3    │   1230    │   209    │ Home (Thu evening)       │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  Target: Day 12, Office visit (loc_id = 2)                                       │
│                                                                                  │
│  Generated sequence:                                                             │
│  ───────────────────                                                             │
│  {                                                                               │
│    "X": [2, 3, 5, 3, 3, 2, 3, 2, 4, 3, 2, 3, 2, 4, 3],  # 15 visits in 7 days  │
│    "user_X": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],  # All user 1      │
│    "weekday_X": [4, 4, 5, 5, 6, 0, 0, 1, 1, 1, 2, 2, 3, 3, 3],                  │
│    "start_min_X": [510, 1110, 720, 900, 0, 510, 1110, 510, 1140, 1230,          │
│                    510, 1110, 510, 1140, 1230],                                  │
│    "dur_X": [570, 329, 120, 539, 1439, 570, 329, 600, 60, 209,                  │
│              570, 329, 600, 60, 209],                                            │
│    "diff": [7, 7, 6, 6, 5, 4, 4, 3, 3, 3, 2, 2, 1, 1, 1],  # Days before target │
│    "Y": 2  # Target: Office                                                      │
│  }                                                                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Final Output Verification

### Pickle File Contents

```python
import pickle

with open("data/geolife_eps20/processed/geolife_eps20_prev7_train.pk", "rb") as f:
    train_sequences = pickle.load(f)

# Alice's sequence (one of many)
alice_seq = train_sequences[42]  # Example index

print("Location history (X):", alice_seq["X"])
# [2, 3, 5, 3, 3, 2, 3, 2, 4, 3, 2, 3, 2, 4, 3]

print("Target location (Y):", alice_seq["Y"])
# 2 (Office)

print("Sequence length:", len(alice_seq["X"]))
# 15

print("Days before target (diff):", alice_seq["diff"])
# [7, 7, 6, 6, 5, 4, 4, 3, 3, 3, 2, 2, 1, 1, 1]
```

### Metadata Verification

```json
{
  "dataset_name": "geolife",
  "output_dataset_name": "geolife_eps20_prev7",
  "epsilon": 20,
  "previous_day": 7,
  "total_user_num": 31,
  "total_loc_num": 248,
  "unique_users": 30,
  "unique_locations": 246,
  "total_staypoints": 15000,
  "train_sequences": 8500,
  "val_sequences": 2100,
  "test_sequences": 2100,
  "total_sequences": 12700
}
```

**Interpreting metadata**:
- Use `total_loc_num = 248` for embedding size (includes pad=0, unknown=1)
- Use `total_user_num = 31` for user embedding size (includes pad=0)
- Alice contributes ~280 sequences to training (based on her 120 days)

---

## PyTorch Usage Example

### Complete Dataset Class

```python
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json

class GeoLifeDataset(Dataset):
    """PyTorch Dataset for GeoLife preprocessed data."""
    
    def __init__(self, pk_file, max_seq_len=50):
        """
        Args:
            pk_file: Path to .pk file
            max_seq_len: Maximum sequence length (for padding)
        """
        with open(pk_file, "rb") as f:
            self.sequences = pickle.load(f)
        self.max_seq_len = max_seq_len
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        seq_len = len(seq["X"])
        
        # Pad sequences to max_seq_len
        def pad_sequence(arr, pad_value=0):
            padded = torch.zeros(self.max_seq_len, dtype=torch.long)
            actual_len = min(seq_len, self.max_seq_len)
            padded[:actual_len] = torch.LongTensor(arr[-actual_len:])
            return padded
        
        return {
            "X": pad_sequence(seq["X"], pad_value=0),  # 0 = padding token
            "user_X": pad_sequence(seq["user_X"], pad_value=0),
            "weekday_X": pad_sequence(seq["weekday_X"], pad_value=0),
            "start_min_X": pad_sequence(seq["start_min_X"], pad_value=0),
            "dur_X": pad_sequence(seq["dur_X"], pad_value=0),
            "diff": pad_sequence(seq["diff"], pad_value=0),
            "Y": torch.LongTensor([seq["Y"]]),
            "seq_len": torch.LongTensor([min(seq_len, self.max_seq_len)])
        }


# Usage
train_dataset = GeoLifeDataset(
    "data/geolife_eps20/processed/geolife_eps20_prev7_train.pk",
    max_seq_len=50
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Load metadata for model initialization
with open("data/geolife_eps20/processed/geolife_eps20_prev7_metadata.json") as f:
    metadata = json.load(f)

# Model initialization
import torch.nn as nn

loc_embedding = nn.Embedding(
    num_embeddings=metadata["total_loc_num"],  # 248
    embedding_dim=64,
    padding_idx=0
)

user_embedding = nn.Embedding(
    num_embeddings=metadata["total_user_num"],  # 31
    embedding_dim=32,
    padding_idx=0
)

# Training loop
for batch in train_loader:
    X = batch["X"]           # Shape: [32, 50]
    Y = batch["Y"]           # Shape: [32, 1]
    seq_lens = batch["seq_len"]  # Shape: [32, 1]
    
    # Embed locations
    X_emb = loc_embedding(X)  # Shape: [32, 50, 64]
    
    # ... rest of model forward pass
    break
```

### Training a Simple Model

```python
import torch
import torch.nn as nn
import torch.optim as optim

class SimpleLocationPredictor(nn.Module):
    def __init__(self, num_locations, embedding_dim=64, hidden_dim=128):
        super().__init__()
        self.loc_embedding = nn.Embedding(num_locations, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_locations)
    
    def forward(self, X, seq_lens):
        # X: [batch, seq_len]
        embedded = self.loc_embedding(X)  # [batch, seq_len, emb_dim]
        
        # Pack padded sequence for efficient LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, seq_lens.squeeze().cpu(), batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        
        # Use last hidden state for prediction
        output = self.fc(hidden.squeeze(0))  # [batch, num_locations]
        return output


# Initialize model
model = SimpleLocationPredictor(
    num_locations=metadata["total_loc_num"],
    embedding_dim=64,
    hidden_dim=128
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
model.train()
for epoch in range(10):
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in train_loader:
        optimizer.zero_grad()
        
        X = batch["X"]
        Y = batch["Y"].squeeze()
        seq_lens = batch["seq_len"]
        
        output = model(X, seq_lens)
        loss = criterion(output, Y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(output, 1)
        correct += (predicted == Y).sum().item()
        total += Y.size(0)
    
    print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
          f"Acc={100*correct/total:.2f}%")
```

---

## Summary

This example traced Alice's data through the complete pipeline:

1. **Raw GPS** → 50,000+ position fixes over 120 days
2. **Staypoints** → ~300 staypoints (2-3 per day)
3. **Locations** → 4 semantic locations (Home, Office, Gym, Restaurant)
4. **Sequences** → ~280 training sequences with 7-day history windows
5. **Model Input** → Padded tensors ready for PyTorch

The consistent example demonstrates how real mobility patterns are preserved through preprocessing and made available for machine learning.

---

*Documentation Version: 1.0*
*For PhD Research Reference*
