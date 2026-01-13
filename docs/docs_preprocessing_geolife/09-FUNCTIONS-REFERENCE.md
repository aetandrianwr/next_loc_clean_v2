# Functions Reference - Complete API Documentation

## Table of Contents
1. [Overview](#overview)
2. [Script 1 Functions](#script-1-functions)
3. [Script 2 Functions](#script-2-functions)
4. [H3-Specific Functions](#h3-specific-functions)
5. [External Library Functions](#external-library-functions)

---

## Overview

This document provides a complete API reference for all functions in the preprocessing pipeline.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FUNCTION ORGANIZATION                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Script 1 Functions:                     Script 2 Functions:                     │
│  ────────────────────                    ────────────────────                    │
│  • calculate_user_quality()              • split_dataset()                       │
│  • _get_tracking_quality()               • _get_split_days_user()                │
│  • _get_time()                           • get_valid_sequence()                  │
│  • enrich_time_info()                    • get_valid_sequence_per_user()         │
│  • process_raw_to_intermediate()         • generate_sequences()                  │
│  • main()                                • process_for_previous_day()            │
│                                          • process_intermediate_to_final()       │
│  H3-Specific:                            • main()                                │
│  ────────────                                                                    │
│  • generate_h3_locations()               External (trackintel):                  │
│                                          ────────────────────────                │
│                                          • read_geolife()                        │
│                                          • generate_staypoints()                 │
│                                          • generate_locations()                  │
│                                          • create_activity_flag()                │
│                                          • merge_staypoints()                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Script 1 Functions

### calculate_user_quality

```python
def calculate_user_quality(sp, trips, quality_file, quality_filter):
    """
    Calculate user quality based on temporal tracking coverage.
    
    This function assesses the quality of each user's tracking data by:
    1. Merging staypoints and trips for complete timeline coverage
    2. Filtering users based on minimum tracking days
    3. Calculating sliding window quality scores
    4. Saving quality metrics to CSV
    
    Parameters
    ----------
    sp : GeoDataFrame
        Staypoints DataFrame with columns:
        - user_id: int, user identifier
        - started_at: datetime, staypoint start time
        - finished_at: datetime, staypoint end time
        
    trips : DataFrame
        Trip segments DataFrame with columns:
        - user_id: int, user identifier
        - started_at: datetime, trip start time
        - finished_at: datetime, trip end time
        
    quality_file : str
        Path to save the quality CSV file
        Example: "data/geolife_eps20/interim/quality/user_quality_eps20.csv"
        
    quality_filter : dict
        Dictionary with quality filter parameters:
        - day_filter: int, minimum tracking days required (e.g., 50)
        - window_size: int, sliding window size in weeks (e.g., 10)
    
    Returns
    -------
    numpy.ndarray
        Array of valid user IDs that pass the quality filter
    
    Side Effects
    ------------
    - Creates quality directory if not exists
    - Saves user_quality CSV with columns: user_id, quality
    
    Example
    -------
    >>> quality_filter = {"day_filter": 50, "window_size": 10}
    >>> valid_users = calculate_user_quality(sp, trips, "quality.csv", quality_filter)
    >>> print(f"Valid users: {len(valid_users)}")
    Valid users: 30
    """
```

### _get_tracking_quality

```python
def _get_tracking_quality(df, window_size):
    """
    Calculate tracking quality using sliding window for a single user.
    
    This is a helper function called by calculate_user_quality().
    It slides a window across the user's tracking period and calculates
    coverage ratio (tracked time / total time) for each window position.
    
    Parameters
    ----------
    df : DataFrame
        User's combined staypoints and trips with columns:
        - user_id: int
        - started_at: datetime
        - finished_at: datetime
        - duration: float, seconds
        
    window_size : int
        Size of sliding window in weeks
    
    Returns
    -------
    DataFrame
        Quality scores with columns:
        - timestep: int, window position index
        - quality: float, coverage ratio (0.0 to 1.0)
        - user_id: int
    
    Algorithm
    ---------
    1. Calculate total weeks of tracking
    2. For each window position (i = 0 to weeks - window_size):
       a. Define window boundaries: [start_date + i weeks, start_date + i + window_size weeks)
       b. Filter records in window
       c. Calculate quality = sum(durations) / total_window_seconds
    3. Return DataFrame with all quality scores
    
    Example
    -------
    >>> df = user_data  # 20 weeks of tracking
    >>> quality = _get_tracking_quality(df, window_size=10)
    >>> print(quality)
       timestep  quality  user_id
    0         0     0.65        1
    1         1     0.70        1
    2         2     0.68        1
    ...
    """
```

### _get_time

```python
def _get_time(df):
    """
    Calculate temporal features for a user's staypoints.
    
    Adds relative time features based on user's first tracking day.
    Called via groupby().apply() on user_id groups.
    
    Parameters
    ----------
    df : DataFrame
        User's staypoints with columns:
        - started_at: datetime
        - finished_at: datetime
    
    Returns
    -------
    DataFrame
        Input DataFrame with added columns:
        - start_day: int, days since first record (0, 1, 2, ...)
        - end_day: int, days since first record
        - start_min: int, minutes from midnight (0-1439)
        - end_min: int, minutes from midnight (1-1440)
        - weekday: int, day of week (0=Monday, 6=Sunday)
    
    Special Cases
    -------------
    - end_min = 0 (midnight) is converted to 1440
    - Timezone info is removed for consistency
    
    Example
    -------
    Input:
        started_at: 2008-10-23 09:30:00
        finished_at: 2008-10-23 17:45:00
        (User's first record: 2008-10-20)
    
    Output:
        start_day: 3
        end_day: 3
        start_min: 570 (9*60 + 30)
        end_min: 1065 (17*60 + 45)
        weekday: 3 (Thursday)
    """
```

### enrich_time_info

```python
def enrich_time_info(sp):
    """
    Add temporal features to all staypoints and prepare final format.
    
    Orchestrates time feature calculation across all users and
    prepares the intermediate dataset format.
    
    Parameters
    ----------
    sp : GeoDataFrame
        Merged staypoints with columns:
        - user_id: int
        - started_at: datetime
        - finished_at: datetime
        - location_id: int
        - duration: float (calculated)
    
    Returns
    -------
    DataFrame
        Final intermediate format with columns:
        - id: int, unique sequential identifier
        - user_id: int
        - location_id: int
        - start_day: int
        - end_day: int
        - start_min: int
        - end_min: int
        - weekday: int
        - duration: float
    
    Processing Steps
    ----------------
    1. Apply _get_time() to each user group
    2. Drop original timestamp columns
    3. Sort by user_id, start_day, start_min
    4. Convert IDs to int type
    5. Create new sequential id column
    
    Example
    -------
    >>> sp_time = enrich_time_info(sp_merged)
    >>> print(sp_time.columns)
    Index(['id', 'user_id', 'location_id', 'start_day', 'end_day',
           'start_min', 'end_min', 'weekday', 'duration'])
    """
```

### process_raw_to_intermediate

```python
def process_raw_to_intermediate(config):
    """
    Main processing function: raw trajectories to intermediate staypoint dataset.
    
    Executes the complete Script 1 pipeline with 7 processing steps.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary loaded from YAML with structure:
        {
            "dataset": {
                "name": str,           # "geolife"
                "epsilon": int,        # DBSCAN epsilon in meters
                "previous_day": list   # [7] or [3, 7, 14]
            },
            "preprocessing": {
                "staypoint": {
                    "gap_threshold": int,      # 1440 (24 hours)
                    "dist_threshold": int,     # 200 meters
                    "time_threshold": int,     # 30 minutes
                    "activity_time_threshold": int  # 25 minutes
                },
                "location": {
                    "num_samples": int,        # 2
                    "distance_metric": str,    # "haversine"
                    "agg_level": str           # "dataset"
                },
                "quality_filter": {
                    "day_filter": int,         # 50 days
                    "window_size": int         # 10 weeks
                },
                "max_duration": int,           # 2880 minutes
                "split": {
                    "train": float,            # 0.6
                    "val": float,              # 0.2
                    "test": float              # 0.2
                }
            },
            "random_seed": int                 # 42
        }
    
    Returns
    -------
    DataFrame
        Final intermediate dataset (sp_time)
    
    Side Effects
    ------------
    Creates output files in data/geolife_eps{X}/interim/:
    - intermediate_eps{X}.csv (main output)
    - staypoints_all_eps{X}.csv
    - staypoints_merged_eps{X}.csv
    - locations_eps{X}.csv
    - valid_users_eps{X}.csv
    - raw_stats_eps{X}.json
    - interim_stats_eps{X}.json
    - quality/user_quality_eps{X}.csv
    
    Processing Steps
    ----------------
    1. Read raw Geolife GPS trajectories
    2. Generate staypoints from position fixes
    3. Create activity flags
    4. Filter users based on quality metrics
    5. Filter activity staypoints
    6. Generate locations using DBSCAN clustering
    7. Merge staypoints and enrich temporal information
    
    Example
    -------
    >>> with open("config/geolife.yaml") as f:
    ...     config = yaml.safe_load(f)
    >>> sp_time = process_raw_to_intermediate(config)
    >>> print(f"Generated {len(sp_time)} staypoints")
    """
```

---

## Script 2 Functions

### split_dataset

```python
def split_dataset(totalData, split_ratios):
    """
    Split dataset into train, validation, and test sets per user.
    
    Performs chronological split for each user based on their tracking days.
    
    Parameters
    ----------
    totalData : DataFrame
        All staypoints with columns:
        - user_id: int
        - start_day: int
        - (other columns preserved)
        
    split_ratios : dict
        Split ratios dictionary:
        - train: float (e.g., 0.6)
        - val: float (e.g., 0.2)
        - test: float (e.g., 0.2)
        Note: Must sum to 1.0
    
    Returns
    -------
    tuple[DataFrame, DataFrame, DataFrame]
        (train_data, vali_data, test_data)
        Each DataFrame has the same columns as input minus "Dataset"
    
    Algorithm
    ---------
    1. For each user, call _get_split_days_user() to assign "Dataset" label
    2. Filter rows by Dataset == "train", "val", "test"
    3. Drop the temporary "Dataset" column
    4. Return three separate DataFrames
    
    Example
    -------
    >>> split_ratios = {"train": 0.6, "val": 0.2, "test": 0.2}
    >>> train, val, test = split_dataset(data, split_ratios)
    >>> print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    Train: 9000, Val: 3000, Test: 3000
    """
```

### _get_split_days_user

```python
def _get_split_days_user(df, split_ratios):
    """
    Assign train/val/test labels based on tracking day for single user.
    
    Helper function called via groupby().apply().
    
    Parameters
    ----------
    df : DataFrame
        Single user's staypoints with start_day column
        
    split_ratios : dict
        Split ratios (train, val, test)
    
    Returns
    -------
    DataFrame
        Input with added "Dataset" column containing "train", "val", or "test"
    
    Algorithm
    ---------
    Given maxDay = max(start_day) for user:
    - Days [0, maxDay * train)           → "train"
    - Days [maxDay * train, maxDay * 0.8) → "val"
    - Days [maxDay * 0.8, maxDay]         → "test"
    
    Example
    -------
    User with 100 days (maxDay=99), split 60/20/20:
    - train_split = 99 * 0.6 = 59.4
    - val_split = 99 * 0.8 = 79.2
    
    Day 50:  50 < 59.4  → "train"
    Day 70:  59.4 ≤ 70 < 79.2 → "val"
    Day 90:  90 ≥ 79.2 → "test"
    """
```

### get_valid_sequence

```python
def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """
    Identify valid staypoint IDs that can serve as prediction targets.
    
    A staypoint is valid if it has sufficient history in the previous_day window.
    
    Parameters
    ----------
    input_df : DataFrame
        Staypoints for one split with columns:
        - id: int, unique staypoint identifier
        - user_id: int
        - start_day: int
        
    previous_day : int, default=7
        Number of days to look back for history
        
    min_length : int, default=3
        Minimum number of staypoints required in history window
    
    Returns
    -------
    list[int]
        List of valid staypoint IDs
    
    Validation Rules
    ----------------
    For each staypoint to be valid:
    1. diff_day >= previous_day (enough days have passed)
    2. History window [current_day - previous_day, current_day) has >= min_length staypoints
    
    Example
    -------
    >>> valid_ids = get_valid_sequence(train_data, previous_day=7)
    >>> print(f"Valid targets: {len(valid_ids)}")
    Valid targets: 8500
    """
```

### get_valid_sequence_per_user

```python
def get_valid_sequence_per_user(df, previous_day, valid_ids):
    """
    Generate sequence dictionaries for a single user.
    
    Creates the actual training samples with features and targets.
    
    Parameters
    ----------
    df : DataFrame
        Single user's staypoints with columns:
        - id, user_id, location_id, start_day, weekday, start_min, duration
        
    previous_day : int
        History window in days
        
    valid_ids : set
        Set of valid staypoint IDs (from get_valid_sequence)
    
    Returns
    -------
    list[dict]
        List of sequence dictionaries, each with keys:
        - X: ndarray, location history
        - user_X: ndarray, user IDs
        - weekday_X: ndarray, weekdays
        - start_min_X: ndarray, start times
        - dur_X: ndarray, durations
        - diff: ndarray, days before target
        - Y: int, target location
    
    Algorithm
    ---------
    For each valid staypoint (target):
    1. Get all staypoints before target (chronologically)
    2. Filter to history window [target_day - previous_day, target_day)
    3. Extract features from history
    4. Create dictionary with history features and target
    
    Example
    -------
    >>> user_seqs = get_valid_sequence_per_user(user_df, 7, valid_ids)
    >>> print(f"User sequences: {len(user_seqs)}")
    >>> print(user_seqs[0].keys())
    dict_keys(['X', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff', 'Y'])
    """
```

### generate_sequences

```python
def generate_sequences(data_df, valid_ids, previous_day):
    """
    Generate sequences for all users in a split.
    
    Orchestrates sequence generation across all users with progress bar.
    
    Parameters
    ----------
    data_df : DataFrame
        All staypoints for one split (train, val, or test)
        
    valid_ids : list
        Valid staypoint IDs from get_valid_sequence()
        
    previous_day : int
        History window in days
    
    Returns
    -------
    list[dict]
        All sequences combined from all users
    
    Example
    -------
    >>> train_seqs = generate_sequences(train_data, valid_ids, 7)
    Generating sequences: 100%|████████| 30/30 [00:05<00:00, 5.43it/s]
    >>> print(f"Total sequences: {len(train_seqs)}")
    Total sequences: 8500
    """
```

### process_for_previous_day

```python
def process_for_previous_day(sp, split_ratios, max_duration, previous_day, 
                              epsilon, dataset_name, output_base_path):
    """
    Complete processing pipeline for a specific previous_day value.
    
    Executes all 5 steps of Script 2 and saves output files.
    
    Parameters
    ----------
    sp : DataFrame
        Intermediate staypoint data from Script 1
        
    split_ratios : dict
        Train/val/test split ratios
        
    max_duration : int
        Maximum duration cap in minutes
        
    previous_day : int
        History window in days
        
    epsilon : int
        DBSCAN epsilon (for naming only, not used in processing)
        
    dataset_name : str
        Dataset name (e.g., "geolife")
        
    output_base_path : str
        Base output path (e.g., "data/geolife_eps20")
    
    Returns
    -------
    dict
        Metadata dictionary with dataset statistics
    
    Side Effects
    ------------
    Creates files in {output_base_path}/processed/:
    - {name}_train.pk
    - {name}_validation.pk
    - {name}_test.pk
    - {name}_metadata.json
    
    Processing Steps
    ----------------
    1. Split dataset into train/val/test
    2. Encode location IDs (OrdinalEncoder + 2)
    3. Filter valid sequences
    4. Filter users with valid sequences in all splits
    5. Generate and save sequences
    """
```

### process_intermediate_to_final

```python
def process_intermediate_to_final(config):
    """
    Main processing function: intermediate to processed datasets.
    
    Orchestrates Script 2 processing for all previous_day values.
    
    Parameters
    ----------
    config : dict
        Configuration dictionary (same as Script 1)
    
    Returns
    -------
    dict
        Dictionary mapping previous_day values to their metadata
        {7: metadata_dict, 14: metadata_dict, ...}
    
    Side Effects
    ------------
    Creates processed files for each previous_day value in config
    
    Example
    -------
    >>> config = yaml.safe_load(open("config/geolife.yaml"))
    >>> all_metadata = process_intermediate_to_final(config)
    >>> print(all_metadata[7]["total_sequences"])
    12700
    """
```

---

## H3-Specific Functions

### generate_h3_locations

```python
def generate_h3_locations(sp, h3_resolution, num_samples):
    """
    Generate locations from staypoints using H3 hexagonal grid.
    
    Replaces DBSCAN-based generate_locations() with H3-based assignment.
    
    Parameters
    ----------
    sp : GeoDataFrame
        Staypoints with geometry column (geom or geometry)
        
    h3_resolution : int
        H3 resolution level (0-15)
        - 7: ~1.2 km edge
        - 8: ~461 m edge (recommended)
        - 9: ~174 m edge
        
    num_samples : int
        Minimum staypoints in H3 cell to be valid location
    
    Returns
    -------
    tuple[GeoDataFrame, DataFrame]
        sp: Input with location_id column added
        locs: Locations DataFrame with columns:
            - location_id (index)
            - h3_cell: str, H3 cell index
            - center_lat: float
            - center_lng: float
            - h3_resolution: int
    
    Algorithm
    ---------
    1. Extract lat/lon from geometry
    2. Assign H3 cell to each staypoint: h3.latlng_to_cell(lat, lon, resolution)
    3. Count staypoints per cell
    4. Filter cells with count >= num_samples
    5. Create integer location_id mapping
    6. Build locations DataFrame with cell centers
    
    Example
    -------
    >>> sp, locs = generate_h3_locations(sp, h3_resolution=8, num_samples=2)
    >>> print(f"Locations: {len(locs)}")
    >>> print(locs.head())
                       h3_cell  center_lat  center_lng  h3_resolution
    location_id
    0            88283082b9fffff    39.9847   116.3184              8
    1            88283082b1fffff    39.9892   116.3210              8
    """
```

---

## External Library Functions

### trackintel.io.dataset_reader.read_geolife

```python
def read_geolife(geolife_path, print_progress=False):
    """
    Read GeoLife dataset from raw .plt files.
    
    Parameters
    ----------
    geolife_path : str
        Path to GeoLife data folder containing Data/ subdirectory
        
    print_progress : bool
        Whether to show progress bar
    
    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame]
        (pfs, labels)
        - pfs: Position fixes with user_id, tracked_at, geom, elevation
        - labels: Transportation mode labels (may be empty)
    
    Example
    -------
    >>> pfs, labels = read_geolife("data/raw_geolife", print_progress=True)
    >>> print(f"Position fixes: {len(pfs)}")
    """
```

### trackintel.generate_staypoints

```python
# Called as: pfs.as_positionfixes.generate_staypoints(...)
def generate_staypoints(gap_threshold, dist_threshold, time_threshold,
                        include_last=True, print_progress=False, n_jobs=1):
    """
    Detect staypoints from position fixes.
    
    Parameters
    ----------
    gap_threshold : int
        Maximum gap in minutes before new trajectory
        
    dist_threshold : int
        Maximum distance in meters to be considered staying
        
    time_threshold : int
        Minimum duration in minutes to qualify as staypoint
        
    include_last : bool
        Include last staypoint even if incomplete
        
    n_jobs : int
        Number of parallel jobs (-1 for all cores)
    
    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame]
        (pfs, sp)
        - pfs: Updated position fixes with staypoint_id
        - sp: Generated staypoints
    """
```

### trackintel.generate_locations

```python
# Called as: sp.as_staypoints.generate_locations(...)
def generate_locations(epsilon, num_samples, distance_metric="haversine",
                       agg_level="dataset", n_jobs=1):
    """
    Cluster staypoints into locations using DBSCAN.
    
    Parameters
    ----------
    epsilon : int
        DBSCAN epsilon in meters
        
    num_samples : int
        Minimum samples for DBSCAN cluster
        
    distance_metric : str
        Distance metric ("haversine" for earth surface)
        
    agg_level : str
        Aggregation level ("dataset" for all users together)
    
    Returns
    -------
    tuple[GeoDataFrame, GeoDataFrame]
        (sp, locs)
        - sp: Staypoints with location_id (NaN for noise)
        - locs: Location definitions
    """
```

---

## Function Call Graph

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    FUNCTION CALL GRAPH                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Script 1:                                                                       │
│  ─────────                                                                       │
│  main()                                                                          │
│    └─► process_raw_to_intermediate(config)                                       │
│          ├─► read_geolife()           [trackintel]                              │
│          ├─► generate_staypoints()     [trackintel]                              │
│          ├─► create_activity_flag()    [trackintel]                              │
│          ├─► calculate_user_quality()                                            │
│          │     └─► _get_tracking_quality()                                       │
│          ├─► generate_locations()      [trackintel, DBSCAN version]              │
│          │   OR                                                                  │
│          ├─► generate_h3_locations()   [H3 version]                              │
│          ├─► merge_staypoints()        [trackintel]                              │
│          └─► enrich_time_info()                                                  │
│                └─► _get_time()                                                   │
│                                                                                  │
│  Script 2:                                                                       │
│  ─────────                                                                       │
│  main()                                                                          │
│    └─► process_intermediate_to_final(config)                                     │
│          └─► [for each previous_day]                                             │
│                └─► process_for_previous_day()                                    │
│                      ├─► split_dataset()                                         │
│                      │     └─► _get_split_days_user()                            │
│                      ├─► OrdinalEncoder.fit_transform()  [sklearn]               │
│                      ├─► get_valid_sequence()                                    │
│                      └─► generate_sequences()                                    │
│                            └─► get_valid_sequence_per_user()                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

*Documentation Version: 1.0*
*For PhD Research Reference*
