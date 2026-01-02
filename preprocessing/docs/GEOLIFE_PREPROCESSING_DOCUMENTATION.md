# Geolife Dataset Preprocessing: Comprehensive Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Background](#2-theoretical-background)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Script 1: Raw to Interim](#4-script-1-raw-to-interim)
5. [Script 2: Interim to Processed](#5-script-2-interim-to-processed)
6. [Configuration Reference](#6-configuration-reference)
7. [Data Formats](#7-data-formats)
8. [Statistical Analysis](#8-statistical-analysis)
9. [Troubleshooting](#9-troubleshooting)
10. [References](#10-references)

---

## 1. Introduction

### 1.1 Purpose

This documentation provides a comprehensive guide to preprocessing the **GeoLife GPS Trajectory Dataset** for next location prediction tasks. The preprocessing pipeline transforms raw GPS trajectories into structured sequences suitable for machine learning models.

### 1.2 What is the GeoLife Dataset?

The **GeoLife GPS Trajectory Dataset** was collected by Microsoft Research Asia from April 2007 to August 2012. It contains GPS trajectories from 182 users, primarily located in Beijing, China. Key characteristics:

- **Total Trajectories**: ~17,621 trajectories
- **Total Points**: ~24 million GPS points
- **Coverage Area**: Primarily Beijing (93% of data), with global travel data
- **Total Distance**: ~1.2 million kilometers
- **Total Duration**: ~48,000+ hours
- **Collection Devices**: Various GPS loggers and GPS-enabled smartphones

### 1.3 Research Context

**Next Location Prediction** is the task of predicting where a user will go next based on their historical movement patterns. This requires:

1. Converting raw GPS points into meaningful locations (staypoints)
2. Clustering similar staypoints into discrete locations
3. Creating temporal sequences with features for model training

### 1.4 Key Concepts

| Concept | Definition |
|---------|------------|
| **Position Fix (GPS Point)** | A single GPS coordinate with timestamp |
| **Staypoint** | A geographic region where a user stayed for a significant duration |
| **Location** | A cluster of staypoints representing a meaningful place |
| **Sequence** | A time-ordered series of locations visited by a user |
| **Previous Day** | The historical window (in days) used to build input sequences |

---

## 2. Theoretical Background

### 2.1 Staypoint Detection

#### 2.1.1 What is a Staypoint?

A **staypoint** represents a location where a user remained for a significant period. It's detected when:
- The user stays within a small geographic area (distance threshold)
- For a minimum duration (time threshold)

#### 2.1.2 Mathematical Definition

Given a trajectory `T = {p1, p2, ..., pn}` where each point `pi = (lati, loni, ti)`:

A staypoint `S` is detected at points `{pi, pi+1, ..., pj}` if:

1. **Spatial Constraint**: For all points in the set:
   ```
   Distance(pi, pk) ≤ dist_threshold, for all k ∈ [i, j]
   ```

2. **Temporal Constraint**: 
   ```
   tj - ti ≥ time_threshold
   ```

#### 2.1.3 The Sliding Window Algorithm

The staypoint detection uses a **sliding window approach**:

```
Algorithm: Staypoint Detection
Input: Trajectory T, dist_threshold, time_threshold, gap_threshold
Output: Set of staypoints S

1. Initialize i = 0
2. While i < len(T):
   a. j = i + 1
   b. While j < len(T) AND Distance(T[i], T[j]) ≤ dist_threshold:
      c. If T[j].time - T[i].time ≥ time_threshold:
         - Create staypoint with centroid of T[i:j+1]
         - Record started_at = T[i].time, finished_at = T[j].time
         - i = j + 1
         - break
      d. If T[j].time - T[j-1].time > gap_threshold:
         - Break (gap too large, start new search)
      e. j = j + 1
   f. If no staypoint found: i = i + 1
3. Return S
```

#### 2.1.4 Geolife Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `dist_threshold` | 200 meters | Captures stops at buildings/POIs |
| `time_threshold` | 30 minutes | Filters brief stops (traffic, etc.) |
| `gap_threshold` | 1440 minutes (24h) | Handles overnight gaps |
| `activity_time_threshold` | 25 minutes | Defines "meaningful" activities |

### 2.2 Distance Calculation: Haversine Formula

GPS coordinates are on a sphere, so we use the **Haversine formula** for great-circle distance:

```
a = sin²(Δlat/2) + cos(lat1) × cos(lat2) × sin²(Δlon/2)
c = 2 × atan2(√a, √(1-a))
d = R × c
```

Where:
- `R` = Earth's radius (6,371 km)
- `lat1, lat2` = Latitudes in radians
- `Δlat, Δlon` = Differences in latitude and longitude

### 2.3 Location Clustering: DBSCAN

#### 2.3.1 Why DBSCAN?

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** is ideal for location clustering because:
- No need to specify number of clusters
- Handles arbitrary cluster shapes
- Identifies noise points (outliers)
- Works well with geographic data

#### 2.3.2 DBSCAN Algorithm

```
Algorithm: DBSCAN
Input: Points P, epsilon (ε), min_samples
Output: Cluster labels

1. Mark all points as unvisited
2. For each unvisited point p:
   a. Mark p as visited
   b. Find neighbors N = {q ∈ P : distance(p, q) ≤ ε}
   c. If |N| < min_samples:
      - Mark p as noise
   d. Else:
      - Create new cluster C
      - Add p to C
      - For each q in N:
        - If q is unvisited:
          - Mark q as visited
          - Find neighbors N' of q
          - If |N'| ≥ min_samples: N = N ∪ N'
        - If q is not in any cluster: Add q to C
3. Return cluster labels
```

#### 2.3.3 Epsilon Selection

The **epsilon (ε)** parameter determines the neighborhood radius:

| Epsilon | Effect | Use Case |
|---------|--------|----------|
| 20m | Fine-grained | Urban areas with dense POIs |
| 50m | Medium | Mixed environments |
| 100m | Coarse | Rural or sparse areas |

For Geolife (Beijing urban environment), **ε = 20 meters** provides good granularity.

### 2.4 User Quality Assessment

#### 2.4.1 Temporal Tracking Quality

User quality measures how completely a user's movements are tracked over time:

```
Quality = (Total tracked duration) / (Total time period)
```

#### 2.4.2 Sliding Window Quality

To ensure consistent tracking, we use a sliding window approach:

```
Algorithm: Sliding Window Quality
Input: User data, window_size (weeks)
Output: Quality scores per window

1. total_weeks = (end_date - start_date) / 7 days
2. For i = 0 to (total_weeks - window_size):
   a. window_start = start_date + i weeks
   b. window_end = window_start + window_size weeks
   c. tracked_duration = sum of staypoint + tripleg durations in window
   d. total_duration = window_end - window_start
   e. quality[i] = tracked_duration / total_duration
3. Return quality scores
```

#### 2.4.3 Quality Filter Criteria (Geolife)

Users are retained if:
1. **Tracking duration ≥ 50 days**: Sufficient data for patterns
2. **Mean sliding window quality ≥ threshold**: Consistent tracking

### 2.5 Sequence Generation

#### 2.5.1 Temporal Split Strategy

Data is split **chronologically per user** to prevent data leakage:

```
User Timeline: |-------- Train (60%) --------|-- Val (20%) --|-- Test (20%) --|
```

This ensures:
- Model doesn't see future data during training
- Realistic evaluation scenario
- Each user contributes to all splits

#### 2.5.2 Valid Sequence Criteria

A sequence is valid if:
1. Target staypoint is ≥ `previous_day` days from user's first record
2. History window contains ≥ 3 staypoints
3. User has valid sequences in train, val, AND test splits

#### 2.5.3 Sequence Features

Each sequence sample contains:

| Feature | Description | Shape |
|---------|-------------|-------|
| `X` | Historical location IDs | `(seq_len,)` |
| `user_X` | User ID (repeated) | `(seq_len,)` |
| `weekday_X` | Day of week (0-6) | `(seq_len,)` |
| `start_min_X` | Start time in minutes (0-1439) | `(seq_len,)` |
| `dur_X` | Duration in minutes | `(seq_len,)` |
| `diff` | Days until target | `(seq_len,)` |
| `Y` | Target location ID | scalar |

---

## 3. Pipeline Overview

### 3.1 Two-Stage Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE                          │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  RAW DATA              SCRIPT 1                INTERIM DATA              │
│  ┌─────────┐          ┌─────────┐             ┌─────────────┐            │
│  │ GPS     │          │ Raw to  │             │ Intermediate │            │
│  │ Points  │ ──────►  │ Interim │ ──────────► │ Staypoints   │            │
│  │ (.plt)  │          │         │             │ (.csv)       │            │
│  └─────────┘          └─────────┘             └─────────────┘            │
│       │                    │                        │                    │
│       │              Parameters:                    │                    │
│       │              • epsilon                      │                    │
│       │              • dist_threshold               │                    │
│       │              • time_threshold               │                    │
│       │                                             │                    │
│       │                                             ▼                    │
│       │                                      ┌─────────────┐            │
│       │                                      │ Locations   │            │
│       │                                      │ (.csv)      │            │
│       │                                      └─────────────┘            │
│       │                                                                  │
│  ─────┼──────────────────────────────────────────────────────────────── │
│       │                                                                  │
│       │               SCRIPT 2                PROCESSED DATA             │
│       │              ┌─────────┐             ┌─────────────┐            │
│       │              │ Interim │             │ Train/Val/  │            │
│       └──────────────│   to    │ ──────────► │ Test        │            │
│                      │Processed│             │ (.pk)       │            │
│                      └─────────┘             └─────────────┘            │
│                           │                        │                    │
│                     Parameters:                    │                    │
│                     • previous_day                 │                    │
│                     • split ratios                 ▼                    │
│                                              ┌─────────────┐            │
│                                              │ Metadata    │            │
│                                              │ (.json)     │            │
│                                              └─────────────┘            │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Why Two Scripts?

| Script | Parameterized By | Runtime | When to Re-run |
|--------|------------------|---------|----------------|
| Script 1 | `epsilon` | ~15-30 min | Change clustering radius |
| Script 2 | `previous_day` | ~2-5 min | Change history window |

This separation enables faster iteration when only changing sequence parameters.

### 3.3 File Organization

```
data/geolife_eps20/
├── interim/                          # Script 1 outputs
│   ├── intermediate_eps20.csv        # Main intermediate data
│   ├── locations_eps20.csv           # Location coordinates
│   ├── staypoints_all_eps20.csv      # All staypoints (before filtering)
│   ├── staypoints_merged_eps20.csv   # After merging consecutive
│   ├── valid_users_eps20.csv         # Filtered users
│   ├── raw_stats_eps20.json          # Raw data statistics
│   ├── interim_stats_eps20.json      # Interim statistics
│   └── quality/
│       └── user_quality_eps20.csv    # Per-user quality scores
│
└── processed/                        # Script 2 outputs
    ├── geolife_eps20_prev7_train.pk
    ├── geolife_eps20_prev7_validation.pk
    ├── geolife_eps20_prev7_test.pk
    └── geolife_eps20_prev7_metadata.json
```

---

## 4. Script 1: Raw to Interim

### 4.1 Overview

**File**: `preprocessing/geolife_1_raw_to_interim.py`

**Purpose**: Transform raw GPS trajectories into intermediate staypoint data with locations and temporal features.

### 4.2 Step-by-Step Processing

#### Step 1: Read Raw Geolife Trajectories

```python
from trackintel.io.dataset_reader import read_geolife

pfs, _ = read_geolife(raw_path, print_progress=True)
```

**Input Format** (`.plt` files):
```
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04
39.984683,116.31845,0,492,39744.1202546296,2008-10-23,02:53:10
```

**Output**: GeoDataFrame with columns:
- `user_id`: Integer user identifier (0-181)
- `tracked_at`: Timestamp
- `geometry`: POINT geometry (lon, lat)
- `elevation`: Altitude in meters

**Code Explanation**:
```python
# The trackintel library provides a specialized reader for Geolife format
# It automatically:
# 1. Scans all user folders (000, 001, ..., 181)
# 2. Reads all .plt files in each user's Trajectory folder
# 3. Parses the header and GPS points
# 4. Creates standardized positionfixes GeoDataFrame
```

#### Step 2: Generate Staypoints

```python
pfs, sp = pfs.as_positionfixes.generate_staypoints(
    gap_threshold=1440,      # 24 hours - split trajectories at gaps
    include_last=True,       # Include last points as staypoints
    print_progress=True,
    dist_threshold=200,      # 200 meters - spatial threshold
    time_threshold=30,       # 30 minutes - temporal threshold
    n_jobs=-1                # Use all CPU cores
)
```

**Algorithm Details**:

1. **Sort position fixes** by user and time
2. **For each user**, iterate through GPS points
3. **Start candidate staypoint** at current point
4. **Expand window** while:
   - Next point is within `dist_threshold` of centroid
   - No time gap > `gap_threshold` exists
5. **Create staypoint** if duration ≥ `time_threshold`
6. **Calculate centroid** as mean of all points in staypoint

**Code Deep Dive**:
```python
# The generate_staypoints method uses the sliding window algorithm
# Key implementation details:

# 1. Distance calculation uses Haversine formula for geographic accuracy
# 2. Centroid is the geometric mean of all points
# 3. Parallelization splits users across CPU cores

# Output columns:
# - user_id: Same as input
# - started_at: First point timestamp
# - finished_at: Last point timestamp
# - geom: POINT geometry of centroid
# - elevation: Mean elevation of points
```

**Statistics After This Step**:
- Input: ~24 million GPS points
- Output: ~20,000-30,000 staypoints
- Reduction ratio: ~99%

#### Step 3: Create Activity Flag

```python
sp = sp.as_staypoints.create_activity_flag(
    method="time_threshold",
    time_threshold=25  # 25 minutes
)
```

**Logic**:
```python
# A staypoint is marked as "activity" if:
# duration = finished_at - started_at
# is_activity = (duration >= activity_time_threshold)

# This filters out brief stops like:
# - Traffic lights
# - Brief parking
# - Pickup/dropoff
```

**Purpose**: Focus on meaningful locations (home, work, stores) rather than transient stops.

#### Step 4: Filter Valid Users (Quality Calculation)

```python
def calculate_user_quality(sp, trips, quality_file, quality_filter):
    """
    Calculate user tracking quality using temporal coverage analysis.
    
    The quality metric measures what fraction of time is accounted for
    by tracked movements (staypoints + triplegs/trips).
    """
    
    # Step 4a: Combine staypoints and trips
    sp["type"] = "sp"
    trips["type"] = "tpl"
    df_all = pd.concat([sp, trips])
    
    # Step 4b: Split overlapping periods (handle data issues)
    df_all = _split_overlaps(df_all, granularity="day")
    
    # Step 4c: Calculate duration of each record
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
    
    # Step 4d: Calculate overall tracking quality
    total_quality = temporal_tracking_quality(df_all, granularity="all")
    
    # Step 4e: Calculate tracking days per user
    total_quality["days"] = df_all.groupby("user_id").apply(
        lambda x: (x["finished_at"].max() - x["started_at"].min()).days
    ).values
    
    # Step 4f: Filter users with minimum tracking days
    user_filter_day = total_quality.loc[
        total_quality["days"] > quality_filter["day_filter"]  # > 50 days
    ]["user_id"].unique()
    
    # Step 4g: Calculate sliding window quality
    sliding_quality = df_all.groupby("user_id").apply(
        _get_tracking_quality, 
        window_size=quality_filter["window_size"]  # 10 weeks
    ).reset_index(drop=True)
    
    # Step 4h: Filter and return valid users
    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]
    filter_after_user_quality = filter_after_day.groupby("user_id")["quality"].mean()
    
    return filter_after_user_quality["user_id"].values
```

**Quality Calculation Helper**:
```python
def _get_tracking_quality(df, window_size):
    """
    Calculate tracking quality using sliding windows.
    
    For each window of 'window_size' weeks:
    quality = (sum of tracked durations) / (total window duration)
    
    This ensures users have consistent tracking, not just long duration.
    """
    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()
    
    quality_list = []
    for i in range(0, weeks - window_size + 1):
        # Define window boundaries
        curr_start = datetime.combine(
            start_date + timedelta(weeks=i), 
            datetime.time()
        )
        curr_end = datetime.combine(
            curr_start + timedelta(weeks=window_size),
            datetime.time()
        )
        
        # Get data in window
        window_data = df.loc[
            (df["started_at"] >= curr_start) & 
            (df["finished_at"] < curr_end)
        ]
        
        if window_data.empty:
            continue
            
        # Calculate quality for this window
        total_seconds = (curr_end - curr_start).total_seconds()
        quality = window_data["duration"].sum() / total_seconds
        quality_list.append([i, quality])
    
    return pd.DataFrame(quality_list, columns=["timestep", "quality"])
```

**Filter Result**:
- Input: 182 users
- Output: ~91 users (50% retained)
- Criteria: ≥50 days tracking, consistent quality

#### Step 5: Filter Activity Staypoints

```python
sp = sp.loc[sp["is_activity"] == True]
```

**Simple but critical**: Removes short-duration stops to focus on meaningful locations.

#### Step 6: Generate Locations (DBSCAN Clustering)

```python
sp, locs = sp.as_staypoints.generate_locations(
    epsilon=20,                    # 20 meters radius
    num_samples=2,                 # Minimum 2 staypoints per cluster
    distance_metric="haversine",   # Geographic distance
    agg_level="dataset",           # Cluster across all users
    n_jobs=-1                      # Parallel processing
)
```

**Implementation Details**:
```python
# The generate_locations method:
# 1. Extracts coordinates from staypoint geometries
# 2. Converts to radians for Haversine distance
# 3. Runs DBSCAN with specified parameters
# 4. Assigns location_id to each staypoint
# 5. Creates location centroids

# Why agg_level="dataset"?
# - Clusters staypoints from ALL users together
# - Creates shared location vocabulary
# - Enables cross-user pattern learning
# - More robust location definitions
```

**Noise Handling**:
```python
# DBSCAN assigns -1 to noise points (outliers)
# These are staypoints that don't cluster with others

# Filter out noise staypoints
sp = sp.loc[~sp["location_id"].isna()].copy()

# Noise points typically represent:
# - One-time visits
# - GPS errors
# - Uncommon locations
```

**Output Statistics**:
- Staypoints clustered: ~15,000-20,000
- Unique locations: ~2,000-3,000
- Noise filtered: ~5-10%

#### Step 7: Merge Consecutive Staypoints & Enrich Temporal Info

```python
# Step 7a: Merge consecutive staypoints at same location
sp_merged = sp.as_staypoints.merge_staypoints(
    triplegs=pd.DataFrame([]),  # No triplegs to consider
    max_time_gap="1min",        # Merge if gap < 1 minute
    agg={"location_id": "first"}  # Keep first location_id
)
```

**Why Merge?**
- GPS noise can split single visits into multiple staypoints
- Merging creates cleaner sequences
- Reduces artificial complexity

```python
# Step 7b: Recalculate duration after merging
sp_merged["duration"] = (
    sp_merged["finished_at"] - sp_merged["started_at"]
).dt.total_seconds() // 60  # Convert to minutes

# Step 7c: Add temporal features
def _get_time(df):
    """Extract temporal features for each user."""
    # Get user's first day as reference
    min_day = pd.to_datetime(df["started_at"].min().date())
    
    # Remove timezone for consistent calculation
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)
    
    # Calculate day indices
    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["end_day"] = (df["finished_at"] - min_day).dt.days
    
    # Calculate minute of day (0-1439)
    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["end_min"] = df["finished_at"].dt.hour * 60 + df["finished_at"].dt.minute
    
    # Handle midnight edge case
    df.loc[df["end_min"] == 0, "end_min"] = 24 * 60
    
    # Day of week (0=Monday, 6=Sunday)
    df["weekday"] = df["started_at"].dt.weekday
    
    return df

sp_time = sp.groupby("user_id", group_keys=False).apply(_get_time)
```

**Final Intermediate Output Schema**:
```
id,user_id,location_id,duration,start_day,end_day,start_min,end_min,weekday
0,0,0,64.0,0,0,183,248,3
1,0,1,309.0,0,0,272,582,3
2,0,2,899.0,0,1,670,130,3
```

---

## 5. Script 2: Interim to Processed

### 5.1 Overview

**File**: `preprocessing/geolife_2_interim_to_processed.py`

**Purpose**: Transform intermediate staypoint data into model-ready sequence files.

### 5.2 Step-by-Step Processing

#### Step 1: Load Intermediate Data

```python
interim_file = os.path.join(interim_path, f"intermediate_eps{epsilon}.csv")
sp = pd.read_csv(interim_file)
```

**Data at this point**:
```
Columns: id, user_id, location_id, duration, start_day, end_day, 
         start_min, end_min, weekday
Rows: ~19,000 staypoints
Users: ~91
Locations: ~2,000
```

#### Step 2: Truncate Duration

```python
max_duration = 2880  # 2 days in minutes

sp_copy = sp.copy()
sp_copy.loc[sp_copy["duration"] > max_duration - 1, "duration"] = max_duration - 1
```

**Rationale**:
- Extremely long durations (e.g., week-long trips) can skew the model
- Capping at 2 days preserves most information
- Duration becomes a bounded feature (0 to 2879 minutes)

#### Step 3: Split Dataset

```python
def split_dataset(totalData, split_ratios):
    """
    Split dataset chronologically per user.
    
    Each user's timeline is divided:
    - First 60%: Training
    - Next 20%: Validation
    - Last 20%: Testing
    """
    totalData = totalData.groupby("user_id", group_keys=False).apply(
        _get_split_days_user, 
        split_ratios=split_ratios
    )
    
    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "val"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()
    
    return train_data, vali_data, test_data


def _get_split_days_user(df, split_ratios):
    """Assign split labels based on day index."""
    maxDay = df["start_day"].max()
    
    # Calculate split boundaries
    train_split = maxDay * split_ratios["train"]      # Day 0 to 60%
    validation_split = maxDay * (split_ratios["train"] + split_ratios["val"])  # 60% to 80%
    
    # Assign labels
    df["Dataset"] = "test"  # Default to test (80% to 100%)
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[
        (df["start_day"] >= train_split) & 
        (df["start_day"] < validation_split), 
        "Dataset"
    ] = "val"
    
    return df
```

**Visualization**:
```
User 0 (tracked 307 days):
Day 0 -------- Day 184 -------- Day 245 -------- Day 307
|---- Train ----|---- Val ----|---- Test ----|
      60%            20%            20%
```

#### Step 4: Encode Location IDs

```python
from sklearn.preprocessing import OrdinalEncoder

# Fit encoder on training data only
enc = OrdinalEncoder(
    dtype=np.int64, 
    handle_unknown="use_encoded_value", 
    unknown_value=-1
).fit(train_data["location_id"].values.reshape(-1, 1))

# Transform all splits
# Add 2 to reserve: 0 for padding, 1 for unknown
train_data["location_id"] = enc.transform(
    train_data["location_id"].values.reshape(-1, 1)
) + 2

vali_data["location_id"] = enc.transform(
    vali_data["location_id"].values.reshape(-1, 1)
) + 2

test_data["location_id"] = enc.transform(
    test_data["location_id"].values.reshape(-1, 1)
) + 2
```

**Encoding Scheme**:
| ID | Meaning |
|----|---------|
| 0 | Padding (for batch processing) |
| 1 | Unknown location (not in training) |
| 2+ | Known locations |

**Why This Encoding?**
- Padding (0): Required for variable-length sequences in batches
- Unknown (1): Handles locations appearing only in val/test
- Continuous IDs: Efficient embedding lookup

#### Step 5: Get Valid Sequences

```python
def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """
    Identify staypoints that can be valid prediction targets.
    
    A staypoint is valid if:
    1. It's at least 'previous_day' days from the user's start
    2. Its history window contains at least 'min_length' staypoints
    
    Returns: List of valid staypoint IDs
    """
    valid_id = []
    
    for user in input_df["user_id"].unique():
        df = input_df.loc[input_df["user_id"] == user].copy()
        df = df.reset_index(drop=True)
        
        # Calculate days from user's start
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days
        
        for index, row in df.iterrows():
            # Rule 1: Must have enough history days
            if row["diff_day"] < previous_day:
                continue
            
            # Get history window [current_day - previous_day, current_day)
            hist = df.iloc[:index]  # All staypoints before current
            hist = hist.loc[hist["start_day"] >= (row["start_day"] - previous_day)]
            
            # Rule 2: Must have minimum history length
            if len(hist) < min_length:
                continue
            
            valid_id.append(row["id"])
    
    return valid_id
```

**Logic Visualization**:
```
User Timeline:
Day 0    Day 3    Day 5    Day 7    Day 10   Day 14
  │        │        │        │         │        │
  ●────────●────────●────────●─────────●────────●
  sp1      sp2      sp3      sp4       sp5      sp6

For previous_day=7:
- sp4 (Day 7): History = [sp1,sp2,sp3] (3 items) ✓ Valid
- sp5 (Day 10): History = [sp2,sp3,sp4] (3 items) ✓ Valid
- sp6 (Day 14): History = [sp4,sp5] (2 items) ✗ Invalid (< 3)
```

#### Step 6: Filter Users with Valid Sequences in All Splits

```python
# Get valid sequences per split
valid_ids = get_valid_sequence(train_data, previous_day=previous_day)
valid_ids.extend(get_valid_sequence(vali_data, previous_day=previous_day))
valid_ids.extend(get_valid_sequence(test_data, previous_day=previous_day))

# Find users with valid sequences in ALL splits
valid_users_train = train_data.loc[
    train_data["id"].isin(final_valid_id), "user_id"
].unique()
valid_users_vali = vali_data.loc[
    vali_data["id"].isin(final_valid_id), "user_id"
].unique()
valid_users_test = test_data.loc[
    test_data["id"].isin(final_valid_id), "user_id"
].unique()

valid_users = set.intersection(
    set(valid_users_train), 
    set(valid_users_vali), 
    set(valid_users_test)
)
```

**Why This Filter?**
- Ensures every user contributes to all splits
- Prevents users with sparse data from skewing results
- Guarantees consistent user set across train/val/test

#### Step 7: Re-encode User IDs

```python
# Make user IDs continuous (0, 1, 2, ..., N-1)
user_enc = OrdinalEncoder(dtype=np.int64)

filtered_sp["user_id"] = user_enc.fit_transform(
    filtered_sp["user_id"].values.reshape(-1, 1)
) + 1  # Start from 1, reserve 0 for padding
```

**Result**: User IDs become 1, 2, 3, ..., 45 (for 45 users)

#### Step 8: Generate Sequence Dictionaries

```python
def get_valid_sequence_per_user(df, previous_day, valid_ids):
    """
    Generate sequence samples for a single user.
    
    For each valid target staypoint, create a dictionary containing:
    - X: History location IDs
    - user_X: User ID (repeated)
    - weekday_X: Day of week for each history item
    - start_min_X: Start minute for each history item
    - dur_X: Duration for each history item
    - diff: Days until target for each history item
    - Y: Target location ID
    """
    df = df.reset_index(drop=True)
    data_single_user = []
    
    # Calculate days from start
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days
    
    for index, row in df.iterrows():
        # Skip if not enough history
        if row["diff_day"] < previous_day:
            continue
        
        # Get history window
        hist = df.iloc[:index]
        hist = hist.loc[hist["start_day"] >= (row["start_day"] - previous_day)]
        
        # Skip if not a valid target
        if row["id"] not in valid_ids:
            continue
        
        # Skip if insufficient history
        if len(hist) < 3:
            continue
        
        # Build sequence dictionary
        data_dict = {
            # Input features (history)
            "X": hist["location_id"].values,
            "user_X": hist["user_id"].values,
            "weekday_X": hist["weekday"].values,
            "start_min_X": hist["start_min"].values,
            "dur_X": hist["duration"].values,
            "diff": (row["diff_day"] - hist["diff_day"]).astype(int).values,
            
            # Target
            "Y": int(row["location_id"])
        }
        
        data_single_user.append(data_dict)
    
    return data_single_user


def generate_sequences(data_df, valid_ids, previous_day):
    """Generate sequences for all users."""
    all_sequences = []
    valid_ids_set = set(valid_ids)
    
    for user_id in tqdm(data_df["user_id"].unique()):
        user_df = data_df[data_df["user_id"] == user_id].copy()
        user_sequences = get_valid_sequence_per_user(
            user_df, previous_day, valid_ids_set
        )
        all_sequences.extend(user_sequences)
    
    return all_sequences
```

#### Step 9: Save Output Files

```python
import pickle

# Save pickle files
with open(train_pk_file, "wb") as f:
    pickle.dump(train_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(val_pk_file, "wb") as f:
    pickle.dump(val_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

with open(test_pk_file, "wb") as f:
    pickle.dump(test_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save metadata
metadata = {
    "dataset_name": dataset_name,
    "epsilon": epsilon,
    "previous_day": previous_day,
    "total_user_num": int(train_data["user_id"].max() + 1),
    "total_loc_num": int(train_data["location_id"].max() + 1),
    "train_sequences": len(train_sequences),
    "val_sequences": len(val_sequences),
    "test_sequences": len(test_sequences),
    # ... more fields
}

with open(metadata_file, "w") as f:
    json.dump(metadata, f, indent=2)
```

---

## 6. Configuration Reference

### 6.1 Full Configuration File

```yaml
# config/preprocessing/geolife.yaml

dataset:
  name: "geolife"
  epsilon: 20              # DBSCAN epsilon in meters
  previous_day: [7]        # List: [7, 14] generates both

preprocessing:
  # Staypoint detection parameters
  staypoint:
    gap_threshold: 1440    # Max gap between points (minutes)
    dist_threshold: 200    # Spatial threshold (meters)
    time_threshold: 30     # Minimum duration (minutes)
    activity_time_threshold: 25  # Activity flag threshold
  
  # DBSCAN clustering parameters
  location:
    num_samples: 2         # Minimum points per cluster
    distance_metric: "haversine"
    agg_level: "dataset"   # Cluster all users together
  
  # User filtering parameters
  quality_filter:
    day_filter: 50         # Minimum tracking days
    window_size: 10        # Sliding window weeks
  
  # Duration handling
  max_duration: 2880       # Cap duration at 2 days
  
  # Data splits (chronological per user)
  split:
    train: 0.6
    val: 0.2
    test: 0.2

random_seed: 42
```

### 6.2 Parameter Impact Analysis

| Parameter | Low Value | High Value | Trade-off |
|-----------|-----------|------------|-----------|
| `epsilon` | 20m | 100m | Granularity vs. sparsity |
| `previous_day` | 3 | 14 | Context vs. data availability |
| `dist_threshold` | 100m | 500m | Sensitivity vs. robustness |
| `time_threshold` | 15min | 60min | Coverage vs. noise |
| `day_filter` | 30 | 100 | Data quantity vs. quality |

---

## 7. Data Formats

### 7.1 Raw Input Format

**File**: `data/raw_geolife/{user_id}/Trajectory/{datetime}.plt`

```
Geolife trajectory
WGS 84
Altitude is in Feet
Reserved 3
0,2,255,My Track,0,0,2,8421376
0
39.984702,116.318417,0,492,39744.1201851852,2008-10-23,02:53:04
39.984683,116.31845,0,492,39744.1202546296,2008-10-23,02:53:10
```

**Columns** (after header):
1. Latitude (decimal degrees)
2. Longitude (decimal degrees)
3. Zero (unused)
4. Altitude (feet)
5. Days since 1899-12-30 (Excel format)
6. Date (YYYY-MM-DD)
7. Time (HH:MM:SS)

### 7.2 Intermediate Format

**File**: `data/geolife_eps20/interim/intermediate_eps20.csv`

```csv
id,user_id,location_id,duration,start_day,end_day,start_min,end_min,weekday
0,0,0,64.0,0,0,183,248,3
1,0,1,309.0,0,0,272,582,3
2,0,2,899.0,0,1,670,130,3
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Unique staypoint identifier |
| `user_id` | int | User identifier (encoded) |
| `location_id` | int | Location cluster ID |
| `duration` | float | Duration in minutes |
| `start_day` | int | Day index from user's first record |
| `end_day` | int | End day index |
| `start_min` | int | Start minute of day (0-1439) |
| `end_min` | int | End minute of day (1-1440) |
| `weekday` | int | Day of week (0=Monday, 6=Sunday) |

### 7.3 Processed Format

**File**: `data/geolife_eps20/processed/geolife_eps20_prev7_train.pk`

```python
# List of sequence dictionaries
[
    {
        'X': np.array([7, 6, 8]),           # History location IDs
        'user_X': np.array([1, 1, 1]),      # User ID (repeated)
        'weekday_X': np.array([0, 1, 2]),   # Weekdays
        'start_min_X': np.array([111, 17, 178]),  # Start minutes
        'dur_X': np.array([96., 36., 34.]), # Durations
        'diff': np.array([2, 1, 0]),        # Days to target
        'Y': 4                               # Target location
    },
    # ... more sequences
]
```

### 7.4 Metadata Format

**File**: `data/geolife_eps20/processed/geolife_eps20_prev7_metadata.json`

```json
{
  "dataset_name": "geolife",
  "output_dataset_name": "geolife_eps20_prev7",
  "epsilon": 20,
  "previous_day": 7,
  "total_user_num": 46,
  "total_loc_num": 1187,
  "unique_users": 45,
  "unique_locations": 1185,
  "total_staypoints": 16600,
  "valid_staypoints": 15978,
  "train_staypoints": 8380,
  "val_staypoints": 4018,
  "test_staypoints": 4202,
  "train_sequences": 7424,
  "val_sequences": 3334,
  "test_sequences": 3502,
  "total_sequences": 14260,
  "split_ratios": {
    "train": 0.6,
    "val": 0.2,
    "test": 0.2
  },
  "max_duration_minutes": 2880
}
```

---

## 8. Statistical Analysis

### 8.1 Raw Data Statistics

| Metric | Value |
|--------|-------|
| Total Position Fixes | ~24 million |
| Total Users | 182 |
| Date Range | Apr 2007 - Aug 2012 |
| Primary Location | Beijing, China |

### 8.2 Interim Statistics (ε=20m)

```json
{
  "epsilon": 20,
  "total_staypoints": 19191,
  "total_users": 91,
  "total_locations": 2049,
  "staypoints_per_user_mean": 210.89,
  "duration_mean_min": 421.46,
  "duration_median_min": 308.0,
  "duration_max_min": 4265.0,
  "days_tracked_mean": 307.67
}
```

### 8.3 Processed Statistics (prev7)

| Metric | Train | Validation | Test |
|--------|-------|------------|------|
| Staypoints | 8,380 | 4,018 | 4,202 |
| Sequences | 7,424 | 3,334 | 3,502 |
| Users | 45 | 45 | 45 |
| Locations | 1,185 | varies | varies |

### 8.4 Data Reduction Summary

```
Stage                    Records    Reduction
─────────────────────────────────────────────
Raw GPS Points          24M         -
After Staypoints        30K         99.87%
After Quality Filter    19K         36.67%
After Activity Filter   19K         ~0%
After Location Cluster  19K         ~5% (noise)
After Sequence Gen      14K seq     varies
```

---

## 9. Troubleshooting

### 9.1 Common Issues

#### Issue: "No valid staypoints found"

**Cause**: Quality filter too strict or missing data

**Solution**:
```python
# Check raw data exists
import os
print(os.listdir("data/raw_geolife"))

# Lower quality threshold
quality_filter:
  day_filter: 30  # Reduced from 50
```

#### Issue: "Empty processed files"

**Cause**: `previous_day` too large for available data

**Solution**:
```python
# Check user tracking duration
import pandas as pd
sp = pd.read_csv("data/geolife_eps20/interim/intermediate_eps20.csv")
print(sp.groupby("user_id")["start_day"].max().describe())

# Use smaller previous_day
previous_day: [3]  # Start small
```

#### Issue: "Memory error during DBSCAN"

**Cause**: Too many staypoints for clustering

**Solution**:
```python
# Increase epsilon to reduce points
epsilon: 50  # Instead of 20

# Or reduce n_jobs
n_jobs: 4  # Instead of -1
```

### 9.2 Validation Checks

```python
# Verify output integrity
import pickle
import json

# Load and check sequences
with open("data/geolife_eps20/processed/geolife_eps20_prev7_train.pk", "rb") as f:
    train = pickle.load(f)

print(f"Sequences: {len(train)}")
print(f"Sample keys: {train[0].keys()}")
print(f"X shape: {train[0]['X'].shape}")
print(f"Y range: {min(t['Y'] for t in train)} - {max(t['Y'] for t in train)}")

# Verify metadata consistency
with open("data/geolife_eps20/processed/geolife_eps20_prev7_metadata.json") as f:
    meta = json.load(f)

assert len(train) == meta["train_sequences"]
print("✓ Metadata consistent")
```

---

## 10. References

### 10.1 Dataset Citation

```bibtex
@inproceedings{zheng2008understanding,
  title={Understanding mobility based on GPS data},
  author={Zheng, Yu and Li, Quannan and Chen, Yukun and Xie, Xing and Ma, Wei-Ying},
  booktitle={UbiComp},
  year={2008}
}

@article{zheng2010geolife,
  title={GeoLife: A collaborative social networking service among user, location and trajectory},
  author={Zheng, Yu and Xie, Xing and Ma, Wei-Ying},
  journal={IEEE Data Engineering Bulletin},
  year={2010}
}
```

### 10.2 Algorithm References

- **DBSCAN**: Ester et al., "A density-based algorithm for discovering clusters" (KDD 1996)
- **Staypoint Detection**: Zheng et al., "Learning transportation mode from raw GPS data" (UbiComp 2008)
- **Trackintel Library**: https://github.com/mie-lab/trackintel

### 10.3 Related Documentation

- [DIY Preprocessing Documentation](./DIY_PREPROCESSING_DOCUMENTATION.md)
- [Geolife Walkthrough with Examples](./GEOLIFE_PREPROCESSING_WALKTHROUGH.md)
- [Configuration Reference](../../config/preprocessing/geolife.yaml)

---

*Last updated: January 2026*
*Version: 1.0*
