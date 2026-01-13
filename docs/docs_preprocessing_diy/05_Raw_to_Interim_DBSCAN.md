# Raw to Interim Script Documentation (DBSCAN Version)

## 📋 Table of Contents
1. [Overview](#overview)
2. [Script Architecture](#script-architecture)
3. [Input/Output Specification](#inputoutput-specification)
4. [Line-by-Line Code Walkthrough](#line-by-line-code-walkthrough)
5. [Stage 1: Loading Raw Data](#stage-1-loading-raw-data)
6. [Stage 2: Generating Locations](#stage-2-generating-locations)
7. [Stage 3: Merging Staypoints](#stage-3-merging-staypoints)
8. [Stage 4: Enriching Temporal Features](#stage-4-enriching-temporal-features)
9. [Complete Example](#complete-example)
10. [Troubleshooting](#troubleshooting)

---

## Overview

**Script**: `preprocessing/diy_1_raw_to_interim.py`  
**Purpose**: Transform raw staypoint data into interim dataset with location clusters and temporal features  
**Clustering Method**: DBSCAN (Density-Based Spatial Clustering)

### What This Script Does

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    diy_1_raw_to_interim.py OVERVIEW                          │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT FILES (from notebook):
├── data/raw_diy/3_staypoints_fun_generate_trips.csv
└── data/raw_diy/10_filter_after_user_quality_DIY_slide_filteres.csv

                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ PROCESSING STAGES:                                                           │
│                                                                              │
│ STAGE 1: Load Raw Data ─────────────────────────────────────────────────────│
│          • Read staypoints CSV                                               │
│          • Read valid user list                                              │
│          • Filter to valid users only                                        │
│          • Filter to activity staypoints only (is_activity=True)            │
│                                                                              │
│ STAGE 2: Generate Locations ────────────────────────────────────────────────│
│          • DBSCAN clustering (epsilon=50m, num_samples=2)                   │
│          • Assign location_id to each staypoint                             │
│          • Filter noise points (no cluster)                                  │
│                                                                              │
│ STAGE 3: Merge Staypoints ──────────────────────────────────────────────────│
│          • Merge consecutive staypoints at same location                    │
│          • max_time_gap: 1 minute                                            │
│          • Recalculate duration                                              │
│                                                                              │
│ STAGE 4: Enrich Temporal Features ──────────────────────────────────────────│
│          • Extract start_day, end_day                                        │
│          • Extract start_min, end_min                                        │
│          • Extract weekday                                                   │
│          • Re-encode user_id to integers                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
OUTPUT FILES:
├── data/diy_eps50/interim/intermediate_eps50.csv (main output)
├── data/diy_eps50/interim/locations_eps50.csv
├── data/diy_eps50/interim/staypoints_merged_eps50.csv
├── data/diy_eps50/interim/valid_users_eps50.csv
└── data/diy_eps50/interim/interim_stats_eps50.json
```

---

## Script Architecture

### High-Level Structure

```python
"""
diy_1_raw_to_interim.py - Script Structure
"""

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ IMPORTS (Lines 18-30)                                                       │
# └─────────────────────────────────────────────────────────────────────────────┘
import os, sys, json, pickle, argparse
from pathlib import Path
import yaml, pandas as pd, numpy as np, geopandas as gpd
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
import trackintel as ti

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ HELPER FUNCTIONS (Lines 38-75)                                              │
# └─────────────────────────────────────────────────────────────────────────────┘
def _get_time(df):           # Extract temporal features per user
def enrich_time_info(sp):    # Apply _get_time to all users

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ STAGE FUNCTIONS (Lines 78-216)                                              │
# └─────────────────────────────────────────────────────────────────────────────┘
def load_raw_data(config):              # STAGE 1: Load and filter
def generate_locations(sp, ...):        # STAGE 2: DBSCAN clustering
def merge_staypoints(sp, ...):          # STAGE 3: Merge consecutive
def process_temporal_features(sp, ...): # STAGE 4: Add time features

# ┌─────────────────────────────────────────────────────────────────────────────┐
# │ MAIN FUNCTION (Lines 219-278)                                               │
# └─────────────────────────────────────────────────────────────────────────────┘
def main():
    # Parse arguments
    # Load configuration
    # Create directories
    # Execute pipeline stages
    # Print summary
```

---

## Input/Output Specification

### Input Files

```
INPUT FILE 1: 3_staypoints_fun_generate_trips.csv
═══════════════════════════════════════════════════════════════════════════════

Location: data/raw_diy/3_staypoints_fun_generate_trips.csv

Schema:
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ id             │ int64        │ Staypoint identifier                           │
│ user_id        │ string       │ User UUID                                      │
│ started_at     │ datetime     │ Start timestamp                                │
│ finished_at    │ datetime     │ End timestamp                                  │
│ geom           │ WKT          │ Point geometry                                 │
│ is_activity    │ bool         │ Activity flag                                  │
│ trip_id        │ int64/NaN    │ Trip association                               │
│ prev_trip_id   │ int64/NaN    │ Previous trip                                  │
│ next_trip_id   │ int64/NaN    │ Next trip                                      │
└────────────────┴──────────────┴────────────────────────────────────────────────┘


INPUT FILE 2: 10_filter_after_user_quality_DIY_slide_filteres.csv
═══════════════════════════════════════════════════════════════════════════════

Location: data/raw_diy/10_filter_after_user_quality_DIY_slide_filteres.csv

Schema:
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ user_id        │ string       │ User UUID (quality-filtered)                   │
│ quality        │ float64      │ Mean tracking quality                          │
└────────────────┴──────────────┴────────────────────────────────────────────────┘
```

### Output Files

```
OUTPUT DIRECTORY: data/diy_eps{epsilon}/interim/
═══════════════════════════════════════════════════════════════════════════════

OUTPUT 1: intermediate_eps{epsilon}.csv (MAIN OUTPUT - used by Script 2)
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ id             │ int64        │ Sequential staypoint ID                        │
│ user_id        │ int64        │ Integer user ID (re-encoded)                   │
│ location_id    │ int64        │ DBSCAN cluster ID                              │
│ start_day      │ int64        │ Days since user's first record                 │
│ end_day        │ int64        │ End day number                                 │
│ start_min      │ int64        │ Start minute of day (0-1439)                   │
│ end_min        │ int64        │ End minute of day (1-1440)                     │
│ weekday        │ int64        │ Day of week (0-6)                              │
│ duration       │ float64      │ Duration in minutes                            │
└────────────────┴──────────────┴────────────────────────────────────────────────┘


OUTPUT 2: locations_eps{epsilon}.csv
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ location_id    │ int64        │ Location identifier (index)                    │
│ center         │ POINT        │ Cluster centroid                               │
│ extent         │ POLYGON      │ Cluster boundary                               │
└────────────────┴──────────────┴────────────────────────────────────────────────┘


OUTPUT 3: staypoints_merged_eps{epsilon}.csv
- Intermediate file showing merged staypoints before temporal enrichment


OUTPUT 4: valid_users_eps{epsilon}.csv
- List of user_ids that were processed


OUTPUT 5: interim_stats_eps{epsilon}.json
{
    "epsilon": 50,
    "total_staypoints": 125432,
    "total_users": 155,
    "total_locations": 4521,
    "staypoints_per_user_mean": 809.2,
    "duration_mean_min": 142.5,
    "duration_median_min": 85.0,
    "duration_max_min": 2879.0,
    "days_tracked_mean": 78.5
}
```

---

## Line-by-Line Code Walkthrough

### Imports and Constants (Lines 1-35)

```python
"""
DIY Dataset Preprocessing - Script 1: Raw to Interim
Processes raw DIY staypoint data to interim dataset with locations.

This script:
1. Reads preprocessed staypoints from raw CSV files
2. Filters to valid users based on quality criteria
3. Filters to activity staypoints only
4. Generates locations using DBSCAN clustering
5. Merges consecutive staypoints at same location
6. Enriches with temporal information
7. Saves interim dataset for further processing

Input: data/raw_diy/
Output: data/diy_eps{epsilon}/interim/
"""

import os                           # File path operations
import sys                          # System exit on errors
import json                         # Save statistics as JSON
import pickle                       # Unused (kept for consistency)
import argparse                     # Command-line argument parsing
from pathlib import Path            # Path manipulation

import yaml                         # Load YAML configuration
import pandas as pd                 # DataFrame operations
import numpy as np                  # Numerical operations
import geopandas as gpd             # Geospatial operations
from tqdm import tqdm               # Progress bars
from sklearn.preprocessing import OrdinalEncoder  # Encode user IDs

import trackintel as ti             # Mobility data processing

# Set random seed for reproducibility
RANDOM_SEED = 42
```

---

## Stage 1: Loading Raw Data

### Function: `load_raw_data(config)` (Lines 78-114)

```python
def load_raw_data(config):
    """Load raw DIY staypoints and valid users."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.1: Print stage header
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 1: Loading Raw Data")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.2: Construct input path from config
    # ─────────────────────────────────────────────────────────────────────────
    dataset_name = config['dataset']['name']  # "diy"
    raw_path = f"data/raw_{dataset_name}"     # "data/raw_diy"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.3: Read staypoints using Trackintel
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/2] Reading preprocessed staypoints from {raw_path}...")
    
    sp = ti.read_staypoints_csv(
        f'{raw_path}/3_staypoints_fun_generate_trips.csv',
        columns={'geometry': 'geom'},  # Map 'geom' column to geometry
        index_col='id'                  # Use 'id' as index
    )
    # sp is now a Trackintel GeoDataFrame with geometry
    
    print(f"  Loaded {len(sp):,} staypoints")
    # Example output: "  Loaded 523,456 staypoints"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.4: Read valid users list
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[2/2] Reading valid users...")
    
    valid_user_df = pd.read_csv(
        f'{raw_path}/10_filter_after_user_quality_DIY_slide_filteres.csv'
    )
    valid_user = valid_user_df["user_id"].values
    # valid_user is a numpy array of user UUIDs
    
    print(f"  Loaded {len(valid_user):,} valid users")
    # Example output: "  Loaded 155 valid users"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.5: Filter staypoints to valid users only
    # ─────────────────────────────────────────────────────────────────────────
    sp = sp.loc[sp["user_id"].isin(valid_user)]
    print(f"  Valid users after quality filter: {len(valid_user):,}")
    # This removes all staypoints from users who didn't pass quality filtering

    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.6: Filter to activity staypoints only
    # ─────────────────────────────────────────────────────────────────────────
    sp = sp.loc[sp["is_activity"] == True]
    print(f"  Activity staypoints: {len(sp):,}")
    # is_activity=True means staypoint duration > 25 minutes
    # This removes brief transit stops
    
    # Example output: "  Activity staypoints: 312,789"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 1.7: Validate we have data to process
    # ─────────────────────────────────────────────────────────────────────────
    if len(sp) == 0:
        print("\n❌ Error: No valid staypoints found after quality filtering.")
        sys.exit(1)
    
    return sp, valid_user
```

**Data Transformation Visualization:**

```
STAGE 1: DATA FILTERING
═══════════════════════════════════════════════════════════════════════════════

Initial Staypoints: 523,456
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Filter 1: Valid Users Only                                                   │
│                                                                              │
│ All staypoints from ~50,000 users                                           │
│                    ↓                                                         │
│ Only staypoints from 155 quality-filtered users                             │
│                                                                              │
│ Removal: ~98% of users, ~90% of staypoints                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
Filtered by User: ~52,000
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│ Filter 2: Activity Staypoints Only                                           │
│                                                                              │
│ is_activity = True  (duration > 25 min)  → KEEP                             │
│ is_activity = False (duration ≤ 25 min)  → REMOVE                           │
│                                                                              │
│ Removal: ~40% of remaining staypoints                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
Activity Staypoints: ~31,200
```

---

## Stage 2: Generating Locations

### Function: `generate_locations(sp, config, interim_dir, epsilon)` (Lines 117-146)

```python
def generate_locations(sp, config, interim_dir, epsilon):
    """Generate locations from staypoints using DBSCAN clustering."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2.1: Print stage header
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 2: Generating Locations")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2.2: Get clustering parameters from config
    # ─────────────────────────────────────────────────────────────────────────
    loc_params = config['preprocessing']['location']
    # loc_params = {
    #     'num_samples': 2,
    #     'distance_metric': 'haversine',
    #     'agg_level': 'dataset'
    # }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2.3: Run DBSCAN clustering via Trackintel
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/1] Clustering staypoints with epsilon={epsilon}m...")
    
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=epsilon,                          # 50 meters radius
        num_samples=loc_params['num_samples'],    # Minimum 2 points per cluster
        distance_metric=loc_params['distance_metric'],  # 'haversine' for lat/lon
        agg_level=loc_params['agg_level'],        # 'dataset' = all users together
        n_jobs=-1                                  # Use all CPU cores
    )
    
    # After this call:
    # - sp has a new 'location_id' column
    # - locs is a GeoDataFrame with cluster centroids and extents
    # - Noise points (no cluster) have location_id = NaN
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 2.4: Filter out noise staypoints
    # ─────────────────────────────────────────────────────────────────────────
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print(f"  After filtering non-location staypoints: {len(sp):,}")
    # Noise points are staypoints that couldn't be assigned to any cluster
    # This happens when they're too isolated (no neighbors within epsilon)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 2.5: Save locations to file
    # ─────────────────────────────────────────────────────────────────────────
    locs = locs[~locs.index.duplicated(keep="first")]  # Remove duplicates
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    # Only keep locations that have at least one staypoint
    
    locations_file = os.path.join(interim_dir, f"locations_eps{epsilon}.csv")
    filtered_locs.as_locations.to_csv(locations_file)
    print(f"  Saved {len(filtered_locs):,} unique locations to: {locations_file}")
    
    return sp, filtered_locs
```

**DBSCAN Clustering Visualization:**

```
DBSCAN CLUSTERING PROCESS
═══════════════════════════════════════════════════════════════════════════════

Parameters:
• epsilon = 50 meters (neighborhood radius)
• num_samples = 2 (minimum points to form cluster)
• distance_metric = haversine (great-circle distance)
• agg_level = dataset (cluster all users together)

Geographic Visualization:
                    
     User A's staypoints: ●
     User B's staypoints: ○
     Cluster centers:     ★
     
                    ▲ Latitude
                    │
        ┌───────────┼───────────┐
        │  ●  ●     │           │
        │    ●      │   ○  ○    │
        │  ★        │     ★     │
        │           │   ○       │
        │           │           │
        └───────────┼───────────┘
                    │
    ────────────────┼────────────────▶ Longitude
    
    Location 1: 3 staypoints     Location 2: 3 staypoints
    (2 from User A,              (3 from User B)
     1 from User B)
     
     
DBSCAN Steps:
─────────────────────────────────────────────────────────────────────────────────

1. For each point, find all points within epsilon (50m) radius
   
   Point P1 ──────────┐
          ε=50m       │
       ╭─────────╮    ▼
       │    P1   │   [P1, P2, P3]  (3 neighbors including self)
       │  P2  P3 │   
       ╰─────────╯   

2. If point has >= num_samples (2) neighbors → it's a core point

3. Core points in same neighborhood → same cluster

4. Points with < 2 neighbors → NOISE (filtered out)


Example Cluster Assignment:
┌──────────┬──────────────────────────────────┬─────────────────────┬─────────────┐
│ SP ID    │ Geometry                         │ Neighbors in 50m    │ location_id │
├──────────┼──────────────────────────────────┼─────────────────────┼─────────────┤
│ 1        │ POINT(110.4315 -7.7478)          │ [1, 2, 5]           │ 42          │
│ 2        │ POINT(110.4316 -7.7477)          │ [1, 2, 5]           │ 42          │
│ 3        │ POINT(110.3857 -7.7117)          │ [3, 4]              │ 15          │
│ 4        │ POINT(110.3858 -7.7118)          │ [3, 4]              │ 15          │
│ 5        │ POINT(110.4315 -7.7479)          │ [1, 2, 5]           │ 42          │
│ 6        │ POINT(110.5000 -7.8000)          │ [6]                 │ NaN (noise) │
└──────────┴──────────────────────────────────┴─────────────────────┴─────────────┘

After filtering: SP 6 is removed (no cluster assignment)
```

---

## Stage 3: Merging Staypoints

### Function: `merge_staypoints(sp, config, interim_dir, epsilon)` (Lines 149-181)

```python
def merge_staypoints(sp, config, interim_dir, epsilon):
    """Merge consecutive staypoints at the same location."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.1: Print stage header
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 3: Merging Staypoints")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.2: Get merging parameters
    # ─────────────────────────────────────────────────────────────────────────
    merge_params = config['preprocessing']['staypoint_merging']
    # merge_params = {'max_time_gap': '1min'}
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.3: Keep only necessary columns
    # ─────────────────────────────────────────────────────────────────────────
    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    # Drop other columns (trip_id, prev_trip_id, next_trip_id, is_activity)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.4: Ensure index is named 'id' for Trackintel
    # ─────────────────────────────────────────────────────────────────────────
    if sp.index.name != 'id':
        sp.index.name = 'id'
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.5: Merge consecutive staypoints at same location
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/1] Merging consecutive staypoints (max gap: {merge_params['max_time_gap']})...")
    
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),      # No triplegs to consider
        max_time_gap=merge_params['max_time_gap'],  # '1min'
        agg={"location_id": "first"}     # Keep first location_id when merging
    )
    
    print(f"  After merging: {len(sp_merged):,} staypoints")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.6: Save merged staypoints
    # ─────────────────────────────────────────────────────────────────────────
    sp_merged_file = os.path.join(interim_dir, f"staypoints_merged_eps{epsilon}.csv")
    sp_merged.to_csv(sp_merged_file)
    print(f"  Saved merged staypoints to: {sp_merged_file}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 3.7: Recalculate duration after merging
    # ─────────────────────────────────────────────────────────────────────────
    sp_merged["duration"] = (
        sp_merged["finished_at"] - sp_merged["started_at"]
    ).dt.total_seconds() // 60  # Convert to minutes (integer division)
    
    return sp_merged
```

**Merging Process Visualization:**

```
STAYPOINT MERGING PROCESS
═══════════════════════════════════════════════════════════════════════════════

Why Merge?
─────────────────────────────────────────────────────────────────────────────────
Sometimes the staypoint detection creates multiple consecutive staypoints
at the same location due to brief GPS gaps or movement within the location.
Merging combines these into a single continuous staypoint.


Before Merging:
┌────┬─────────┬─────────────────────┬─────────────────────┬─────────────┐
│ id │ user_id │ started_at          │ finished_at         │ location_id │
├────┼─────────┼─────────────────────┼─────────────────────┼─────────────┤
│ 0  │ user_1  │ 08:00:00            │ 08:30:00            │ 42          │
│ 1  │ user_1  │ 08:30:30            │ 09:00:00            │ 42          │ ← 30s gap
│ 2  │ user_1  │ 09:15:00            │ 10:00:00            │ 15          │
│ 3  │ user_1  │ 10:05:00            │ 11:00:00            │ 15          │ ← 5min gap
│ 4  │ user_1  │ 11:30:00            │ 12:00:00            │ 42          │
└────┴─────────┴─────────────────────┴─────────────────────┴─────────────┘


Merging Logic (max_time_gap = 1 minute):
─────────────────────────────────────────────────────────────────────────────────

Check pairs of consecutive staypoints at same location:

SP0 → SP1:  Same location (42), gap = 30s < 1min  → MERGE ✓
SP1 → SP2:  Different location (42→15)            → NO MERGE
SP2 → SP3:  Same location (15), gap = 5min > 1min → NO MERGE ✗
SP3 → SP4:  Different location (15→42)            → NO MERGE


After Merging:
┌────┬─────────┬─────────────────────┬─────────────────────┬─────────────┬──────────┐
│ id │ user_id │ started_at          │ finished_at         │ location_id │ duration │
├────┼─────────┼─────────────────────┼─────────────────────┼─────────────┼──────────┤
│ 0  │ user_1  │ 08:00:00            │ 09:00:00            │ 42          │ 60 min   │ ← MERGED
│ 1  │ user_1  │ 09:15:00            │ 10:00:00            │ 15          │ 45 min   │
│ 2  │ user_1  │ 10:05:00            │ 11:00:00            │ 15          │ 55 min   │
│ 3  │ user_1  │ 11:30:00            │ 12:00:00            │ 42          │ 30 min   │
└────┴─────────┴─────────────────────┴─────────────────────┴─────────────┴──────────┘


Timeline Visualization:
─────────────────────────────────────────────────────────────────────────────────

Before:
08:00   08:30 08:30:30  09:00   09:15        10:00 10:05        11:00    11:30   12:00
  │───────┤    │────────┤         │───────────┤    │────────────┤          │──────┤
    SP0           SP1                  SP2             SP3                    SP4
    L=42          L=42                 L=15            L=15                   L=42

After:
08:00                   09:00   09:15        10:00 10:05        11:00    11:30   12:00
  │─────────────────────┤         │───────────┤    │────────────┤          │──────┤
         SP0 (merged)                 SP1             SP2                    SP3
         L=42, dur=60min              L=15            L=15                   L=42
```

---

## Stage 4: Enriching Temporal Features

### Helper Function: `_get_time(df)` (Lines 38-52)

```python
def _get_time(df):
    """Extract temporal features from timestamps.
    
    Called per user to calculate relative day numbers.
    """
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4a.1: Get minimum day (first tracking day for this user)
    # ─────────────────────────────────────────────────────────────────────────
    min_day = pd.to_datetime(df["started_at"].min().date())
    # Example: User started tracking on 2021-10-24
    # min_day = 2021-10-24 00:00:00
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4a.2: Remove timezone info (if present)
    # ─────────────────────────────────────────────────────────────────────────
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)
    # Timezones can cause issues in calculations, so we remove them
    # The timezone was already applied during initial loading

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4a.3: Calculate relative day numbers
    # ─────────────────────────────────────────────────────────────────────────
    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["end_day"] = (df["finished_at"] - min_day).dt.days
    # start_day = number of days since user's first record
    #
    # Example:
    #   min_day = 2021-10-24
    #   started_at = 2021-10-26 08:00:00
    #   start_day = (2021-10-26 - 2021-10-24).days = 2

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4a.4: Calculate minute of day
    # ─────────────────────────────────────────────────────────────────────────
    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["end_min"] = df["finished_at"].dt.hour * 60 + df["finished_at"].dt.minute
    # start_min = minute of the day (0-1439)
    #
    # Example:
    #   started_at = 2021-10-26 08:15:00
    #   start_min = 8 * 60 + 15 = 495
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4a.5: Handle midnight edge case
    # ─────────────────────────────────────────────────────────────────────────
    df.loc[df["end_min"] == 0, "end_min"] = 24 * 60  # 1440
    # If finished_at is exactly midnight (00:00), set end_min to 1440
    # This represents "end of the previous day"

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4a.6: Extract weekday
    # ─────────────────────────────────────────────────────────────────────────
    df["weekday"] = df["started_at"].dt.weekday
    # 0 = Monday, 1 = Tuesday, ..., 6 = Sunday
    
    return df
```

### Main Function: `enrich_time_info(sp)` (Lines 55-75)

```python
def enrich_time_info(sp):
    """Add temporal features to staypoints."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b.1: Apply _get_time to each user group
    # ─────────────────────────────────────────────────────────────────────────
    sp = sp.groupby("user_id", group_keys=False).apply(_get_time)
    # This ensures each user's start_day is relative to THEIR first record
    # User A's day 0 might be 2021-10-24
    # User B's day 0 might be 2021-11-15
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b.2: Drop timestamp columns (no longer needed)
    # ─────────────────────────────────────────────────────────────────────────
    sp.drop(columns={"finished_at", "started_at"}, inplace=True)
    # We now have start_day, end_day, start_min, end_min instead
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b.3: Sort by user and time
    # ─────────────────────────────────────────────────────────────────────────
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)
    # Ensures chronological order within each user

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b.4: Convert user_id to integer
    # ─────────────────────────────────────────────────────────────────────────
    if sp["user_id"].dtype == 'object' or sp["user_id"].dtype == 'string':
        # User IDs are UUIDs (strings), convert to integers
        unique_users = sp["user_id"].unique()
        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        sp["user_id"] = sp["user_id"].map(user_mapping)
        # Example: "9358664f-ad4b-46ff-..." → 0
    else:
        sp["user_id"] = sp["user_id"].astype(int)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b.5: Ensure location_id is integer
    # ─────────────────────────────────────────────────────────────────────────
    sp["location_id"] = sp["location_id"].astype(int)

    # ─────────────────────────────────────────────────────────────────────────
    # Step 4b.6: Final cleaning - assign sequential IDs
    # ─────────────────────────────────────────────────────────────────────────
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    # Now 'id' is a column with sequential integers 0, 1, 2, ...
    
    return sp
```

### Stage Function: `process_temporal_features(sp, config, interim_dir, epsilon)` (Lines 184-216)

```python
def process_temporal_features(sp, config, interim_dir, epsilon):
    """Add temporal features to staypoints."""
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4.1: Print stage header
    # ─────────────────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("STAGE 4: Enriching Temporal Features")
    print("="*70)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4.2: Apply temporal enrichment
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[1/1] Extracting temporal features (day, time, weekday)...")
    sp_time = enrich_time_info(sp)
    print(f"  Users with temporal features: {sp_time['user_id'].nunique():,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4.3: Save main interim output
    # ─────────────────────────────────────────────────────────────────────────
    interim_file = os.path.join(interim_dir, f"intermediate_eps{epsilon}.csv")
    sp_time.to_csv(interim_file, index=False)
    print(f"  Saved interim data to: {interim_file}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Step 4.4: Save statistics for analysis
    # ─────────────────────────────────────────────────────────────────────────
    interim_stats = {
        "epsilon": epsilon,
        "total_staypoints": len(sp_time),
        "total_users": sp_time['user_id'].nunique(),
        "total_locations": sp_time['location_id'].nunique(),
        "staypoints_per_user_mean": len(sp_time) / sp_time['user_id'].nunique(),
        "duration_mean_min": float(sp_time['duration'].mean()),
        "duration_median_min": float(sp_time['duration'].median()),
        "duration_max_min": float(sp_time['duration'].max()),
        "days_tracked_mean": float(sp_time.groupby('user_id')['start_day'].max().mean()),
    }
    interim_stats_file = os.path.join(interim_dir, f"interim_stats_eps{epsilon}.json")
    with open(interim_stats_file, 'w') as f:
        json.dump(interim_stats, f, indent=2)
    print(f"  Saved interim statistics to: {interim_stats_file}")
    
    return sp_time
```

**Temporal Feature Calculation Examples:**

```
TEMPORAL FEATURE CALCULATION EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

User: user_001
First tracking day: 2021-10-24 (Sunday)

Input Staypoint:
┌───────────────────────────────────────────────────────────────────────────────┐
│ started_at: 2021-10-26 08:15:00                                               │
│ finished_at: 2021-10-26 12:45:00                                              │
│ location_id: 42                                                               │
│ duration: 270 minutes (calculated earlier)                                    │
└───────────────────────────────────────────────────────────────────────────────┘

Calculations:
─────────────────────────────────────────────────────────────────────────────────

min_day = 2021-10-24 00:00:00 (user's first day)

start_day = (2021-10-26 - 2021-10-24).days = 2
end_day   = (2021-10-26 - 2021-10-24).days = 2

start_min = 8 * 60 + 15 = 495    (8:15 AM = minute 495 of the day)
end_min   = 12 * 60 + 45 = 765   (12:45 PM = minute 765 of the day)

weekday = 2021-10-26.weekday() = 1 (Tuesday)

Output Staypoint:
┌────┬─────────┬─────────────┬───────────┬─────────┬───────────┬─────────┬─────────┬──────────┐
│ id │ user_id │ location_id │ start_day │ end_day │ start_min │ end_min │ weekday │ duration │
├────┼─────────┼─────────────┼───────────┼─────────┼───────────┼─────────┼─────────┼──────────┤
│ 5  │ 0       │ 42          │ 2         │ 2       │ 495       │ 765     │ 1       │ 270      │
└────┴─────────┴─────────────┴───────────┴─────────┴───────────┴─────────┴─────────┴──────────┘


Edge Case: Overnight Staypoint
─────────────────────────────────────────────────────────────────────────────────

Input:
│ started_at: 2021-10-26 22:00:00                                               │
│ finished_at: 2021-10-27 07:00:00                                              │

Calculations:
start_day = 2  (Oct 26)
end_day   = 3  (Oct 27)   ← Different from start_day!
start_min = 22 * 60 + 0 = 1320
end_min   = 7 * 60 + 0 = 420
weekday   = 1  (Tuesday, based on start)

Output:
│ start_day=2, end_day=3, start_min=1320, end_min=420, weekday=1               │


Edge Case: Midnight End Time
─────────────────────────────────────────────────────────────────────────────────

Input:
│ finished_at: 2021-10-27 00:00:00                                              │

Calculation:
end_min = 0 * 60 + 0 = 0
Since end_min == 0, set to 1440 (24 * 60)

This represents "end of day" rather than "start of day"
```

---

## Complete Example

### Running the Script

```bash
# Default configuration (epsilon=50m)
python preprocessing/diy_1_raw_to_interim.py --config config/preprocessing/diy.yaml

# Custom configuration
python preprocessing/diy_1_raw_to_interim.py --config config/preprocessing/diy_custom.yaml
```

### Example Output

```
================================================================================
DIY PREPROCESSING - Script 1: Raw to Interim
================================================================================
[INPUT]  Raw data: data/raw_diy
[OUTPUT] Interim folder: data/diy_eps50/interim
[CONFIG] Dataset: diy, Epsilon: 50
[CONFIG] Random seed: 42
================================================================================

======================================================================
STAGE 1: Loading Raw Data
======================================================================

[1/2] Reading preprocessed staypoints from data/raw_diy...
  Loaded 523,456 staypoints

[2/2] Reading valid users...
  Loaded 155 valid users
  Valid users after quality filter: 155
  Activity staypoints: 312,789
  Saved valid users to: data/diy_eps50/interim/valid_users_eps50.csv

======================================================================
STAGE 2: Generating Locations
======================================================================

[1/1] Clustering staypoints with epsilon=50m...
  After filtering non-location staypoints: 298,432
  Saved 4,521 unique locations to: data/diy_eps50/interim/locations_eps50.csv

======================================================================
STAGE 3: Merging Staypoints
======================================================================

[1/1] Merging consecutive staypoints (max gap: 1min)...
  After merging: 285,678 staypoints
  Saved merged staypoints to: data/diy_eps50/interim/staypoints_merged_eps50.csv

======================================================================
STAGE 4: Enriching Temporal Features
======================================================================

[1/1] Extracting temporal features (day, time, weekday)...
  Users with temporal features: 155
  Saved interim data to: data/diy_eps50/interim/intermediate_eps50.csv
  Saved interim statistics to: data/diy_eps50/interim/interim_stats_eps50.json

================================================================================
SCRIPT 1 COMPLETE: Raw to Interim
================================================================================
Output folder: data/diy_eps50/interim
Main output: data/diy_eps50/interim/intermediate_eps50.csv
================================================================================
```

---

## Troubleshooting

### Common Errors

```
ERROR: No valid staypoints found after quality filtering
─────────────────────────────────────────────────────────────────────────────────
Cause: The valid users file doesn't match the staypoints file
Fix: Ensure both files come from the same preprocessing run

ERROR: generate_locations() got an unexpected keyword argument
─────────────────────────────────────────────────────────────────────────────────
Cause: Trackintel version mismatch
Fix: pip install trackintel==1.4.1

ERROR: FileNotFoundError: [Errno 2] No such file or directory
─────────────────────────────────────────────────────────────────────────────────
Cause: Input files not in expected location
Fix: Create data/raw_diy/ directory with required CSV files

WARNING: Geometry column not found
─────────────────────────────────────────────────────────────────────────────────
Cause: The CSV uses different column name for geometry
Fix: Ensure 'geom' column exists, or modify columns={'geometry': 'your_column'}
```

---

## Summary

The `diy_1_raw_to_interim.py` script:

1. **Loads** raw staypoints and filters to valid users
2. **Clusters** staypoints into locations using DBSCAN
3. **Merges** consecutive same-location staypoints
4. **Enriches** with temporal features (day, minute, weekday)
5. **Outputs** intermediate CSV ready for sequence generation

Key parameters:
- `epsilon`: DBSCAN clustering radius (default: 50m)
- `num_samples`: Minimum points per cluster (default: 2)
- `max_time_gap`: Maximum gap for merging (default: 1min)

Output: `intermediate_eps{epsilon}.csv` used by Script 2.
