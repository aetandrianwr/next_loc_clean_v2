# Script 1: Raw to Interim - Complete Line-by-Line Guide

## Table of Contents
1. [Overview](#overview)
2. [File Header and Imports](#file-header-and-imports)
3. [Function: calculate_user_quality](#function-calculate_user_quality)
4. [Function: _get_tracking_quality](#function-_get_tracking_quality)
5. [Function: _get_time](#function-_get_time)
6. [Function: enrich_time_info](#function-enrich_time_info)
7. [Main Function: process_raw_to_intermediate](#main-function-process_raw_to_intermediate)
8. [Complete Data Flow Example](#complete-data-flow-example)

---

## Overview

**Script**: `preprocessing/geolife_1_raw_to_interim.py`

**Purpose**: Transform raw GeoLife GPS trajectories into an intermediate staypoint dataset with temporal features.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                        SCRIPT 1: RAW TO INTERIM                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT                          PROCESS                          OUTPUT          │
│  ─────                          ───────                          ──────          │
│                                                                                  │
│  data/raw_geolife/              7 Processing Steps               data/geolife_   │
│  └── Data/                                                       eps20/interim/  │
│      └── 000-181/               [1] Read GPS points                              │
│          └── *.plt              [2] Generate staypoints          ├── intermediate│
│                                 [3] Create activity flag              _eps20.csv │
│                                 [4] Filter quality users         ├── locations   │
│                                 [5] Filter activities                 _eps20.csv │
│                                 [6] Cluster → locations          ├── staypoints  │
│                                 [7] Enrich time features              _*.csv     │
│                                                                  └── quality/    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## File Header and Imports

### Docstring (Lines 1-16)

```python
"""
Geolife Dataset Preprocessing - Script 1: Raw to Interim
Processes raw Geolife trajectory data to intermediate staypoint dataset.

This script:
1. Reads raw Geolife GPS trajectories
2. Generates staypoints from position fixes
3. Creates activity flags
4. Filters users based on quality metrics
5. Generates locations using DBSCAN clustering
6. Enriches with temporal information
7. Saves intermediate dataset for sequence generation

Input: data/raw_geolife/
Output: data/geolife_eps{epsilon}/interim/
"""
```

**Explanation**: 
- The docstring documents the script's purpose and the 7 processing steps
- Clearly states input and output paths
- `{epsilon}` is replaced with the actual epsilon value from config

### Import Statements (Lines 18-37)

```python
import os                    # Operating system interface (paths, directories)
import sys                   # System-specific parameters
import json                  # JSON file handling
import pickle                # Python object serialization
import argparse              # Command-line argument parsing
import datetime              # Date and time operations
from pathlib import Path     # Object-oriented filesystem paths

import yaml                  # YAML configuration file parsing
import pandas as pd          # Data manipulation and analysis
import numpy as np           # Numerical computing
import geopandas as gpd      # Geospatial data handling
from tqdm import tqdm        # Progress bars
from sklearn.preprocessing import OrdinalEncoder  # Label encoding

# Trackintel for mobility data processing
from trackintel.io.dataset_reader import read_geolife  # Read GeoLife format
from trackintel.preprocessing.triplegs import generate_trips  # Trip generation
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
import trackintel as ti      # Mobility analytics library

# Set random seed
RANDOM_SEED = 42
```

**Import Categories**:

| Category | Libraries | Purpose |
|----------|-----------|---------|
| Standard | os, sys, json, pickle, argparse, datetime, pathlib | Basic Python operations |
| Data | yaml, pandas, numpy, geopandas | Data loading and manipulation |
| Progress | tqdm | User feedback during processing |
| ML | sklearn.preprocessing | Encoding categorical variables |
| Mobility | trackintel | Specialized trajectory processing |

**Trackintel Library**: This is a specialized library for mobility data analysis that provides:
- `read_geolife()`: Native support for GeoLife format
- `generate_staypoints()`: Convert GPS points to staypoints
- `generate_locations()`: Cluster staypoints into locations
- `temporal_tracking_quality()`: Assess data quality

---

## Function: calculate_user_quality

**Lines 43-89**

This function calculates and filters users based on their tracking data quality.

### Function Signature

```python
def calculate_user_quality(sp, trips, quality_file, quality_filter):
    """Calculate user quality based on temporal tracking coverage."""
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `sp` | GeoDataFrame | Staypoints with user_id, started_at, finished_at |
| `trips` | DataFrame | Trip segments between staypoints |
| `quality_file` | str | Path to save quality CSV |
| `quality_filter` | dict | Contains `day_filter` and `window_size` |

**Returns**: Array of valid user IDs

### Line-by-Line Explanation

#### Lines 45-48: Timestamp Normalization

```python
trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)
```

**Purpose**: Remove timezone information for consistent datetime operations

**Before**:
```
started_at: 2008-10-23 02:53:04+08:00  (Beijing timezone)
```

**After**:
```
started_at: 2008-10-23 02:53:04  (timezone-naive)
```

**Why?**: Simplifies datetime arithmetic and avoids timezone conversion issues

#### Lines 50-56: Merge Staypoints and Trips

```python
print("Starting merge", sp.shape, trips.shape)
sp["type"] = "sp"
trips["type"] = "tpl"
df_all = pd.concat([sp, trips])
df_all = _split_overlaps(df_all, granularity="day")
df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
print("Finished merge", df_all.shape)
```

**Visualization**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MERGE STAYPOINTS AND TRIPS                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input:                                                                          │
│  ───────                                                                         │
│  Staypoints (sp):                    Trips (trips):                             │
│  ┌─────────────────────┐             ┌─────────────────────┐                    │
│  │ user_id │ start│end │             │ user_id │ start│end │                    │
│  │    1    │ 08:00│10:00│             │    1    │ 10:00│10:30│                   │
│  │    1    │ 10:30│12:00│             │    1    │ 12:00│12:15│                   │
│  └─────────────────────┘             └─────────────────────┘                    │
│                                                                                  │
│  Add type column:                                                                │
│  sp["type"] = "sp"    →  type: sp                                               │
│  trips["type"] = "tpl" →  type: tpl (tripleg)                                   │
│                                                                                  │
│  After concat (df_all):                                                          │
│  ┌────────────────────────────────┐                                             │
│  │ user_id │ start│ end  │ type  │                                              │
│  │    1    │ 08:00│ 10:00│  sp   │                                              │
│  │    1    │ 10:00│ 10:30│  tpl  │                                              │
│  │    1    │ 10:30│ 12:00│  sp   │                                              │
│  │    1    │ 12:00│ 12:15│  tpl  │                                              │
│  └────────────────────────────────┘                                             │
│                                                                                  │
│  _split_overlaps: Handles records spanning multiple days                         │
│  Duration: Calculate seconds between start and end                               │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Lines 62-72: Quality Calculation and Day Filter

```python
# Get quality
total_quality = temporal_tracking_quality(df_all, granularity="all")

# Get tracking days
total_quality["days"] = (
    df_all.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
)

# Filter based on days
user_filter_day = (
    total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
    .reset_index(drop=True)["user_id"]
    .unique()
)
```

**Explanation**:

1. `temporal_tracking_quality()`: Calculates overall tracking coverage for each user
2. Calculate tracking days: `max(finished_at) - min(started_at)` = total span
3. Filter users with > 50 days of tracking (from config)

**Example**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         DAY FILTER EXAMPLE                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User │ First Visit  │ Last Visit   │ Days │ Filter (>50) │ Result             │
│  ─────│──────────────│──────────────│──────│──────────────│────────            │
│    1  │ 2008-01-01   │ 2008-06-30   │ 181  │   181 > 50   │ ✓ Keep             │
│    2  │ 2008-03-15   │ 2008-04-10   │  26  │    26 > 50   │ ✗ Remove           │
│    3  │ 2008-02-01   │ 2008-05-15   │ 104  │   104 > 50   │ ✓ Keep             │
│    4  │ 2008-01-01   │ 2008-01-20   │  19  │    19 > 50   │ ✗ Remove           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Lines 74-81: Sliding Window Quality

```python
sliding_quality = (
    df_all.groupby("user_id")
    .apply(_get_tracking_quality, window_size=quality_filter["window_size"])
    .reset_index(drop=True)
)

filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]
filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()
```

**Explanation**:
1. For each user, calculate quality scores using sliding windows
2. Filter to only include users who passed the day filter
3. Calculate mean quality across all windows for each user

#### Lines 86-89: Save and Return

```python
# Save quality file
os.makedirs(os.path.dirname(quality_file), exist_ok=True)
filter_after_user_quality.to_csv(quality_file, index=False)

return filter_after_user_quality["user_id"].values
```

**Output**: CSV file with user_id and quality scores, returns array of valid user IDs

---

## Function: _get_tracking_quality

**Lines 92-111**

Helper function to calculate tracking quality using a sliding window approach.

```python
def _get_tracking_quality(df, window_size):
    """Calculate tracking quality using sliding window."""
    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        total_sec = (curr_end - curr_start).total_seconds()

        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret
```

### Line-by-Line Breakdown

| Line | Code | Explanation |
|------|------|-------------|
| 94 | `weeks = ... // 7` | Total weeks of tracking data |
| 95 | `start_date = ...` | First day of tracking |
| 97-107 | `for i in range(...)` | Slide window across weeks |
| 99-100 | `curr_start, curr_end` | Current window boundaries |
| 102 | `cAll_gdf = df.loc[...]` | Filter data in window |
| 107 | `quality = duration / total_sec` | Calculate coverage ratio |

**Visualization**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SLIDING WINDOW QUALITY CALCULATION                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  User tracking period: 20 weeks                                                  │
│  Window size: 10 weeks                                                           │
│                                                                                  │
│  Week:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19              │
│         ─────────────────────────────────────────────────────────               │
│  Data:  █  █  █  ░  ░  █  █  █  ░  █  █  ░  ░  █  █  █  █  █  █  █              │
│         (█ = data present, ░ = gap)                                              │
│                                                                                  │
│  Window 0 (weeks 0-9):                                                           │
│  |←───────────────────────►|                                                    │
│  Tracked: 6 weeks / 10 weeks = 60% quality                                       │
│                                                                                  │
│  Window 1 (weeks 1-10):                                                          │
│     |←───────────────────────►|                                                 │
│  Tracked: 6 weeks / 10 weeks = 60% quality                                       │
│                                                                                  │
│  Window 10 (weeks 10-19):                                                        │
│                          |←───────────────────────►|                            │
│  Tracked: 8 weeks / 10 weeks = 80% quality                                       │
│                                                                                  │
│  Final user quality = mean(all windows) = (60 + 60 + ... + 80) / N              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Function: _get_time

**Lines 114-128**

Helper function to calculate temporal features for each user's staypoints.

```python
def _get_time(df):
    """Calculate time features for a user."""
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)

    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["end_day"] = (df["finished_at"] - min_day).dt.days

    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["end_min"] = df["finished_at"].dt.hour * 60 + df["finished_at"].dt.minute
    df.loc[df["end_min"] == 0, "end_min"] = 24 * 60

    df["weekday"] = df["started_at"].dt.weekday
    return df
```

### Feature Engineering Explained

| Feature | Formula | Range | Example |
|---------|---------|-------|---------|
| `start_day` | (started_at - min_day).days | 0 to N | Day 0, Day 1, ... |
| `end_day` | (finished_at - min_day).days | 0 to N | Day 0, Day 1, ... |
| `start_min` | hour * 60 + minute | 0-1439 | 08:30 → 510 |
| `end_min` | hour * 60 + minute | 1-1440 | 17:45 → 1065 |
| `weekday` | datetime.weekday() | 0-6 | Monday=0, Sunday=6 |

**Visualization**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       TIME FEATURE CALCULATION                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input Staypoint:                                                                │
│  ─────────────────                                                               │
│  started_at:  2008-10-23 08:30:00                                               │
│  finished_at: 2008-10-23 17:45:00                                               │
│                                                                                  │
│  User's first staypoint: 2008-10-20 (min_day)                                   │
│                                                                                  │
│  Calculated Features:                                                            │
│  ─────────────────────                                                           │
│  start_day:  (2008-10-23) - (2008-10-20) = 3 days                               │
│  end_day:    (2008-10-23) - (2008-10-20) = 3 days                               │
│  start_min:  8 * 60 + 30 = 510 minutes from midnight                            │
│  end_min:    17 * 60 + 45 = 1065 minutes from midnight                          │
│  weekday:    2008-10-23 is Thursday = 3 (Mon=0, Thu=3)                          │
│                                                                                  │
│  Timeline visualization:                                                         │
│  ───────────────────────                                                         │
│  0       510                    1065                    1440                    │
│  │────────│═══════════════════════│──────────────────────│                      │
│  midnight  start                  end                    midnight               │
│           08:30                  17:45                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Special Case: Midnight

```python
df.loc[df["end_min"] == 0, "end_min"] = 24 * 60
```

**Why?**: If a staypoint ends exactly at midnight (00:00), `end_min` would be 0. But we want it to represent the end of the day (1440 minutes), not the start.

```
Example:
  finished_at: 2008-10-24 00:00:00 (midnight)
  
  Without fix: end_min = 0 * 60 + 0 = 0 (wrong - looks like start of day)
  With fix:    end_min = 24 * 60 = 1440 (correct - end of day)
```

---

## Function: enrich_time_info

**Lines 131-144**

Applies time feature calculation to all users and prepares the final intermediate dataset.

```python
def enrich_time_info(sp):
    """Add temporal features to staypoints."""
    sp = sp.groupby("user_id", group_keys=False).apply(_get_time)
    sp.drop(columns={"finished_at", "started_at"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    # Reassign ids
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp
```

### Step-by-Step

| Line | Operation | Effect |
|------|-----------|--------|
| 133 | `groupby().apply(_get_time)` | Calculate time features per user |
| 134 | `drop(columns=...)` | Remove original timestamps (replaced by features) |
| 135 | `sort_values()` | Order by user, then day, then time |
| 136 | `reset_index()` | Clean up index |
| 138-139 | `astype(int)` | Convert IDs to integers |
| 142-143 | Reassign ids | Create new sequential `id` column |

**Before and After**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    BEFORE AND AFTER ENRICH_TIME_INFO                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  BEFORE:                                                                         │
│  ┌────────────────────────────────────────────────────────────────────┐         │
│  │ user_id │ started_at          │ finished_at         │ location_id │         │
│  │    1    │ 2008-10-23 08:30:00 │ 2008-10-23 17:45:00 │    15.0     │         │
│  │    1    │ 2008-10-23 18:00:00 │ 2008-10-23 22:30:00 │    12.0     │         │
│  └────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
│  AFTER:                                                                          │
│  ┌─────────────────────────────────────────────────────────────────────────┐    │
│  │ id │user_id│location_id│start_day│end_day│start_min│end_min│weekday│dur │    │
│  │  0 │   1   │    15     │    3    │   3   │   510   │ 1065  │   3   │555 │    │
│  │  1 │   1   │    12     │    3    │   3   │  1080   │ 1350  │   3   │270 │    │
│  └─────────────────────────────────────────────────────────────────────────┘    │
│                                                                                  │
│  Changes:                                                                        │
│  • Timestamps → Relative day and minute values                                   │
│  • Floating point IDs → Integers                                                 │
│  • New sequential id column for unique identification                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Main Function: process_raw_to_intermediate

**Lines 147-320**

This is the main orchestration function that executes all 7 processing steps.

### Function Signature (Lines 147-149)

```python
def process_raw_to_intermediate(config):
    """Main processing function: raw trajectories to intermediate staypoint dataset."""
```

### Step 0: Configuration Extraction (Lines 150-169)

```python
dataset_config = config["dataset"]
preproc_config = config["preprocessing"]

dataset_name = dataset_config["name"]
epsilon = dataset_config["epsilon"]

# Paths
raw_path = os.path.join("data", f"raw_{dataset_name}")
output_folder = f"{dataset_name}_eps{epsilon}"
interim_path = os.path.join("data", output_folder, "interim")

os.makedirs(interim_path, exist_ok=True)

print("=" * 80)
print(f"GEOLIFE PREPROCESSING - Script 1: Raw to Interim")
print("=" * 80)
print(f"[INPUT]  Raw data: {raw_path}")
print(f"[OUTPUT] Interim folder: {interim_path}")
print(f"[CONFIG] Dataset: {dataset_name}, Epsilon: {epsilon}")
print("=" * 80)
```

**Path Resolution Example**:
```
config["dataset"]["name"] = "geolife"
config["dataset"]["epsilon"] = 20

raw_path = "data/raw_geolife"
output_folder = "geolife_eps20"
interim_path = "data/geolife_eps20/interim"
```

---

### Step 1: Read Raw Data (Lines 171-184)

```python
# 1. Read raw Geolife data
print("\n[1/7] Reading raw Geolife trajectories...")
pfs, _ = read_geolife(raw_path, print_progress=True)
print(f"Loaded {len(pfs)} position fixes from {len(pfs['user_id'].unique())} users")

# Save raw statistics
raw_stats = {
    "total_position_fixes": len(pfs),
    "total_users": len(pfs['user_id'].unique()),
}
stats_file = os.path.join(interim_path, f"raw_stats_eps{epsilon}.json")
with open(stats_file, 'w') as f:
    json.dump(raw_stats, f, indent=2)
print(f"Saved raw statistics to: {stats_file}")
```

**What happens**:
1. `read_geolife()` from trackintel reads all .plt files
2. Returns `pfs` (position fixes) - a GeoDataFrame with all GPS points
3. Saves statistics for later reference

**Output Schema (pfs)**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    POSITION FIXES (pfs) DATAFRAME                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Column       │ Type          │ Description                 │ Example           │
│  ─────────────│───────────────│─────────────────────────────│───────────────    │
│  user_id      │ int           │ User identifier             │ 0, 1, 2, ...      │
│  tracked_at   │ datetime      │ When GPS recorded point     │ 2008-10-23 02:53  │
│  geom         │ Point         │ Lat/lon geometry            │ POINT(116.3 39.9) │
│  elevation    │ float         │ Altitude in meters          │ 150.0             │
│                                                                                  │
│  Sample rows:                                                                    │
│  ┌───────────────────────────────────────────────────────────────────┐          │
│  │ user_id │     tracked_at      │         geom          │elevation │          │
│  │    0    │ 2008-10-23 02:53:04 │ POINT(116.318 39.985) │  149.96  │          │
│  │    0    │ 2008-10-23 02:53:06 │ POINT(116.318 39.985) │  149.96  │          │
│  │    0    │ 2008-10-23 02:53:07 │ POINT(116.318 39.985) │  149.96  │          │
│  └───────────────────────────────────────────────────────────────────┘          │
│                                                                                  │
│  Total: ~24.9 million position fixes                                             │
│  Users: 182                                                                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 2: Generate Staypoints (Lines 186-202)

```python
# 2. Generate staypoints
print("\n[2/7] Generating staypoints...")
sp_config = preproc_config["staypoint"]
pfs, sp = pfs.as_positionfixes.generate_staypoints(
    gap_threshold=sp_config["gap_threshold"],    # 1440 minutes
    include_last=True,
    print_progress=True,
    dist_threshold=sp_config["dist_threshold"],  # 200 meters
    time_threshold=sp_config["time_threshold"],  # 30 minutes
    n_jobs=-1
)
print(f"Generated {len(sp)} staypoints")

# Save staypoints before filtering
sp_before_file = os.path.join(interim_path, f"staypoints_all_eps{epsilon}.csv")
sp.to_csv(sp_before_file)
print(f"Saved all staypoints to: {sp_before_file}")
```

**Staypoint Detection Algorithm**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAYPOINT DETECTION ALGORITHM                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Algorithm: For each position fix, look ahead and find all points within        │
│             dist_threshold. If duration exceeds time_threshold, create staypoint│
│                                                                                  │
│  Parameters used:                                                                │
│  • gap_threshold = 1440 min (24h): Max gap before new trajectory                │
│  • dist_threshold = 200m: Max movement within staypoint                         │
│  • time_threshold = 30min: Min duration to be staypoint                         │
│                                                                                  │
│  Example:                                                                        │
│  ─────────                                                                       │
│  Position fixes at office:                                                       │
│                                                                                  │
│  Time:    09:00  09:05  09:30  10:00  ...  17:00  17:05                         │
│  Points:    ●      ●      ●      ●    ...    ●      ●                           │
│           │←─────────── All within 200m ──────────────►│                        │
│           │←─────────── Duration: 8 hours ─────────────►│                       │
│                                                                                  │
│  Result: One staypoint at office                                                 │
│  • started_at: 09:00                                                             │
│  • finished_at: 17:05                                                            │
│  • geom: centroid of all points                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Output Schema (sp)**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       STAYPOINTS (sp) DATAFRAME                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Column       │ Type          │ Description                 │ Example           │
│  ─────────────│───────────────│─────────────────────────────│───────────────    │
│  user_id      │ int           │ User identifier             │ 0                 │
│  started_at   │ datetime      │ When staypoint began        │ 2008-10-23 09:00  │
│  finished_at  │ datetime      │ When staypoint ended        │ 2008-10-23 17:00  │
│  geom         │ Point         │ Centroid of GPS points      │ POINT(116.3 39.9) │
│                                                                                  │
│  Sample row:                                                                     │
│  ┌────────────────────────────────────────────────────────────────────┐         │
│  │ user_id │     started_at      │    finished_at      │    geom     │         │
│  │    0    │ 2008-10-23 09:00:00 │ 2008-10-23 17:05:00 │ POINT(...)  │         │
│  └────────────────────────────────────────────────────────────────────┘         │
│                                                                                  │
│  Typical reduction: ~24.9M GPS points → ~100K staypoints (99.6% reduction)      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 3: Create Activity Flag (Lines 204-210)

```python
# 3. Create activity flag
print("\n[3/7] Creating activity flags...")
sp = sp.as_staypoints.create_activity_flag(
    method="time_threshold",
    time_threshold=sp_config["activity_time_threshold"]  # 25 minutes
)
```

**Purpose**: Mark staypoints as "activities" if they exceed the activity threshold.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       ACTIVITY FLAG ASSIGNMENT                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  threshold = 25 minutes                                                          │
│                                                                                  │
│  Staypoint │ Duration │ >= 25 min? │ is_activity                               │
│  ──────────│──────────│────────────│─────────────                               │
│  Office    │ 480 min  │    Yes     │    True                                    │
│  Coffee    │  45 min  │    Yes     │    True                                    │
│  Bus stop  │  10 min  │    No      │    False                                   │
│  Home      │ 600 min  │    Yes     │    True                                    │
│  Traffic   │   5 min  │    No      │    False                                   │
│                                                                                  │
│  Note: Bus stop and traffic staypoints will be filtered out in Step 5           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 4: Filter Valid Users (Lines 212-235)

```python
# 4. Filter valid users based on quality
print("\n[4/7] Filtering valid users based on quality...")
quality_path = os.path.join(interim_path, "quality")
quality_file = os.path.join(quality_path, f"user_quality_eps{epsilon}.csv")

if Path(quality_file).is_file():
    print(f"Loading existing quality file: {quality_file}")
    valid_user = pd.read_csv(quality_file)["user_id"].values
else:
    print("Calculating user quality (this may take a while)...")
    # Generate triplegs for quality calculation
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    # Generate trips
    sp_temp, tpls_temp, trips = generate_trips(sp.copy(), tpls, add_geometry=False)
    
    quality_filter = preproc_config["quality_filter"]
    valid_user = calculate_user_quality(sp_temp.copy(), trips.copy(), quality_file, quality_filter)

print(f"Valid users after quality filter: {len(valid_user)}")
sp = sp.loc[sp["user_id"].isin(valid_user)]

# Save user quality info
valid_users_file = os.path.join(interim_path, f"valid_users_eps{epsilon}.csv")
pd.DataFrame({"user_id": valid_user}).to_csv(valid_users_file, index=False)
print(f"Saved valid users to: {valid_users_file}")
```

**Logic Flow**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                       USER QUALITY FILTERING                                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│                         ┌──────────────────┐                                    │
│                         │ Quality file     │                                    │
│                         │ exists?          │                                    │
│                         └────────┬─────────┘                                    │
│                                  │                                               │
│                    ┌─────────────┼─────────────┐                                │
│                    │ Yes         │             │ No                             │
│                    ▼                           ▼                                 │
│           ┌───────────────┐           ┌───────────────────────┐                │
│           │ Load from CSV │           │ Calculate quality     │                │
│           └───────────────┘           │ • Generate triplegs   │                │
│                    │                  │ • Generate trips      │                │
│                    │                  │ • Run calculate_user  │                │
│                    │                  │   _quality()          │                │
│                    │                  │ • Save to CSV         │                │
│                    │                  └───────────────────────┘                │
│                    │                           │                                 │
│                    └───────────┬───────────────┘                                │
│                                │                                                 │
│                                ▼                                                 │
│                    ┌───────────────────────┐                                    │
│                    │ Filter staypoints to  │                                    │
│                    │ keep only valid users │                                    │
│                    └───────────────────────┘                                    │
│                                                                                  │
│  Typical result: 182 users → ~30-50 valid users                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 5: Filter Activity Staypoints (Lines 237-240)

```python
# 5. Filter activity staypoints
print("\n[5/7] Filtering activity staypoints...")
sp = sp.loc[sp["is_activity"] == True]
print(f"Activity staypoints: {len(sp)}")
```

**Simple but Important**: Only keep staypoints that were marked as activities (duration >= 25 min).

---

### Step 6: Generate Locations (Lines 242-264)

```python
# 6. Generate locations using DBSCAN
print("\n[6/7] Generating locations using DBSCAN clustering...")
loc_config = preproc_config["location"]
sp, locs = sp.as_staypoints.generate_locations(
    epsilon=epsilon,                              # 20 meters
    num_samples=loc_config["num_samples"],       # 2
    distance_metric=loc_config["distance_metric"], # "haversine"
    agg_level=loc_config["agg_level"],           # "dataset"
    n_jobs=-1
)

# Filter noise staypoints
sp = sp.loc[~sp["location_id"].isna()].copy()
print(f"Staypoints after filtering noise: {len(sp)}")

# Save locations
locs = locs[~locs.index.duplicated(keep="first")]
filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
locations_file = os.path.join(interim_path, f"locations_eps{epsilon}.csv")
filtered_locs.as_locations.to_csv(locations_file)
print(f"Unique locations: {sp['location_id'].nunique()}")
print(f"Saved locations to: {locations_file}")
```

**DBSCAN Location Clustering**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DBSCAN LOCATION CLUSTERING                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Parameters:                                                                     │
│  • epsilon = 20 meters (max distance in cluster)                                 │
│  • num_samples = 2 (min points to form cluster)                                  │
│  • distance_metric = "haversine" (earth surface distance)                        │
│  • agg_level = "dataset" (cluster across all users)                              │
│                                                                                  │
│  Input: Staypoints (each with geom point)                                        │
│  Output: location_id assigned to each staypoint                                  │
│                                                                                  │
│  Visualization:                                                                  │
│  ─────────────                                                                   │
│                                                                                  │
│       Home area (3 visits)              Office area (50 visits)                  │
│       ┌─────────────┐                   ┌─────────────┐                         │
│       │    ●  ●     │                   │ ●●●●●●●●●●● │                         │
│       │      ●      │                   │ ●●●●●●●●●●● │                         │
│       └─────────────┘                   │ ●●●●●●●●●●● │                         │
│        Location 1                       └─────────────┘                         │
│                                          Location 2                              │
│                                                                                  │
│         ●   (single visit, isolated)                                             │
│         └── location_id = NaN (noise, will be filtered)                          │
│                                                                                  │
│  After filtering: Only staypoints with valid location_id kept                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

### Step 7: Merge and Enrich (Lines 266-319)

```python
# 7. Merge consecutive staypoints and enrich temporal info
print("\n[7/7] Merging staypoints and enriching temporal information...")
sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]

# Merge staypoints
sp_merged = sp.as_staypoints.merge_staypoints(
    triplegs=pd.DataFrame([]),
    max_time_gap="1min",
    agg={"location_id": "first"}
)
print(f"Staypoints after merging: {len(sp_merged)}")

# Save merged staypoints before time enrichment
sp_merged_file = os.path.join(interim_path, f"staypoints_merged_eps{epsilon}.csv")
sp_merged.to_csv(sp_merged_file)
print(f"Saved merged staypoints to: {sp_merged_file}")

# Recalculate duration
sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60

# Add time features
sp_time = enrich_time_info(sp_merged)

print(f"Final users in intermediate data: {sp_time['user_id'].nunique()}")
print(f"Final staypoints: {len(sp_time)}")

# Save intermediate results - this is the main output for Script 2
interim_file = os.path.join(interim_path, f"intermediate_eps{epsilon}.csv")
sp_time.to_csv(interim_file, index=False)
print(f"\n✓ Saved intermediate dataset to: {interim_file}")
```

**Merge Consecutive Staypoints**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    MERGE CONSECUTIVE STAYPOINTS                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Why merge?                                                                      │
│  • GPS may create multiple staypoints at same location with tiny gaps            │
│  • max_time_gap = 1 minute: merge if gap < 1 min                                │
│                                                                                  │
│  Before merging:                                                                 │
│  ───────────────                                                                 │
│  │ started_at │ finished_at │ location_id │                                     │
│  │   08:00    │    08:45    │      1      │                                     │
│  │   08:46    │    09:30    │      1      │  ← Same location, 1 min gap        │
│  │   09:31    │    10:00    │      1      │  ← Same location, 1 min gap        │
│                                                                                  │
│  After merging:                                                                  │
│  ──────────────                                                                  │
│  │ started_at │ finished_at │ location_id │                                     │
│  │   08:00    │    10:00    │      1      │  ← Single merged staypoint         │
│                                                                                  │
│  Duration recalculation:                                                         │
│  duration = (finished_at - started_at) in minutes                                │
│  duration = (10:00 - 08:00) = 120 minutes                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Data Flow Example

Let's trace a concrete example through the entire pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              COMPLETE DATA FLOW: User 001 Example                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STEP 1: Raw GPS Points                                                          │
│  ───────────────────────                                                         │
│  User 001 has .plt files with GPS recordings:                                    │
│                                                                                  │
│  File: 20081023093000.plt                                                        │
│  39.984702,116.318417,0,492,39744.12,2008-10-23,09:30:00                        │
│  39.984703,116.318418,0,492,39744.12,2008-10-23,09:30:05                        │
│  ... (1000s of points throughout the day)                                        │
│  39.984700,116.318420,0,492,39744.75,2008-10-23,18:00:00                        │
│                                                                                  │
│  Total for user: ~50,000 GPS points over 120 days                                │
│                                                                                  │
│  ↓ Step 2: generate_staypoints()                                                 │
│                                                                                  │
│  STEP 2: Staypoints                                                              │
│  ──────────────────                                                              │
│  user_id │     started_at      │    finished_at      │        geom              │
│     1    │ 2008-10-23 09:30:00 │ 2008-10-23 18:00:00 │ POINT(116.318, 39.984)   │
│     1    │ 2008-10-23 18:30:00 │ 2008-10-23 23:00:00 │ POINT(116.320, 39.990)   │
│     1    │ 2008-10-24 08:00:00 │ 2008-10-24 12:00:00 │ POINT(116.318, 39.984)   │
│  ... (500 staypoints total)                                                      │
│                                                                                  │
│  ↓ Step 3: create_activity_flag()                                                │
│                                                                                  │
│  STEP 3: Activity Flags                                                          │
│  ──────────────────────                                                          │
│  All staypoints > 25 min duration get is_activity = True                         │
│  (In this case, all 500 staypoints qualify)                                      │
│                                                                                  │
│  ↓ Step 4: calculate_user_quality()                                              │
│                                                                                  │
│  STEP 4: Quality Filter                                                          │
│  ────────────────────                                                            │
│  User 001: 120 tracking days, 75% average coverage                               │
│  → Passes quality filter (days > 50) ✓                                           │
│                                                                                  │
│  ↓ Step 5: Filter activities                                                     │
│                                                                                  │
│  STEP 5: Activity Filter                                                         │
│  ──────────────────────                                                          │
│  500 staypoints → 500 activity staypoints (all qualify)                          │
│                                                                                  │
│  ↓ Step 6: generate_locations()                                                  │
│                                                                                  │
│  STEP 6: Location Clustering                                                     │
│  ────────────────────────                                                        │
│  Staypoints clustered into locations:                                            │
│  • Home: 200 visits → Location 0                                                 │
│  • Office: 180 visits → Location 1                                               │
│  • Gym: 50 visits → Location 2                                                   │
│  • Restaurant: 30 visits → Location 3                                            │
│  • Random places: 40 visits → NaN (noise, filtered out)                          │
│                                                                                  │
│  After filtering: 460 staypoints with valid location_id                          │
│                                                                                  │
│  ↓ Step 7: merge_staypoints() + enrich_time_info()                              │
│                                                                                  │
│  STEP 7: Final Intermediate Data                                                 │
│  ───────────────────────────────                                                 │
│  id │user_id│loc_id│start_day│end_day│start_min│end_min│weekday│duration       │
│   0 │   1   │  1   │    0    │   0   │   570   │ 1080  │   3   │  510          │
│   1 │   1   │  0   │    0    │   0   │  1110   │ 1380  │   3   │  270          │
│   2 │   1   │  1   │    1    │   1   │   480   │  720  │   4   │  240          │
│  ... (450 staypoints after merging)                                              │
│                                                                                  │
│  OUTPUT: intermediate_eps20.csv                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files Summary

| File | Contents | Purpose |
|------|----------|---------|
| `intermediate_eps{X}.csv` | Main output with all features | Input for Script 2 |
| `staypoints_all_eps{X}.csv` | All staypoints before filtering | Debugging/analysis |
| `staypoints_merged_eps{X}.csv` | Merged staypoints | Debugging/analysis |
| `locations_eps{X}.csv` | Location definitions with geometry | Visualization |
| `valid_users_eps{X}.csv` | List of valid user IDs | Reference |
| `raw_stats_eps{X}.json` | Raw data statistics | Documentation |
| `interim_stats_eps{X}.json` | Interim statistics | Documentation |
| `quality/user_quality_eps{X}.csv` | User quality scores | Analysis |

---

## Next Steps

Continue to [05-SCRIPT2-INTERIM-TO-PROCESSED.md](05-SCRIPT2-INTERIM-TO-PROCESSED.md) to learn how the intermediate data is converted into training sequences.

---

*Documentation Version: 1.0*
*For PhD Research Reference*
