# DIY Dataset Preprocessing: Comprehensive Documentation

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Background](#2-theoretical-background)
3. [Pipeline Overview](#3-pipeline-overview)
4. [Pre-Processing: Raw Data Generation](#4-pre-processing-raw-data-generation)
5. [Script 1: Raw to Interim](#5-script-1-raw-to-interim)
6. [Script 2: Interim to Processed](#6-script-2-interim-to-processed)
7. [Configuration Reference](#7-configuration-reference)
8. [Data Formats](#8-data-formats)
9. [Statistical Analysis](#9-statistical-analysis)
10. [Troubleshooting](#10-troubleshooting)
11. [Comparison with Geolife](#11-comparison-with-geolife)
12. [References](#12-references)

---

## 1. Introduction

### 1.1 Purpose

This documentation provides a comprehensive guide to preprocessing the **DIY (Do-It-Yourself) GPS Dataset** for next location prediction tasks. The DIY dataset is a proprietary mobility dataset collected from smartphone applications in Indonesia.

### 1.2 What is the DIY Dataset?

The **DIY Dataset** contains GPS trajectory data collected from smartphone users in the **Yogyakarta Special Region (Daerah Istimewa Yogyakarta)** of Indonesia. Key characteristics:

- **Collection Period**: October 2021 - June 2022
- **Geographic Region**: Yogyakarta, Indonesia (approximately -7.8° to -7.6° latitude, 110.2° to 110.5° longitude)
- **Total Records**: ~165 million GPS points
- **Total Users**: ~50,000+ unique users (before filtering)
- **Final Users**: ~1,300 users (after quality filtering)
- **Data Source**: Mobile application tracking

### 1.3 Key Differences from Geolife

| Aspect | Geolife | DIY |
|--------|---------|-----|
| Location | Beijing, China | Yogyakarta, Indonesia |
| Period | 2007-2012 | 2021-2022 |
| Raw Data | GPS trajectories (.plt) | Pre-processed staypoints (.csv) |
| Staypoint Detection | Done in preprocessing | Done externally (Trackintel) |
| Quality Filter | 50 days, mean quality | 60 days, min/mean thresholds |
| Default Epsilon | 20 meters | 50 meters |
| Default Split | 60/20/20 | 80/10/10 |

### 1.4 Data Pre-Processing Chain

The DIY data goes through an **external pre-processing step** before entering this pipeline:

```
Raw GPS Points → [External: 02_psl_detection_all.ipynb] → Staypoints CSV
                                                              │
                                                              ▼
                                                    This Pipeline
```

---

## 2. Theoretical Background

### 2.1 Position-Staypoint-Location (PSL) Framework

The DIY dataset follows the **PSL (Position-Staypoint-Location)** framework:

```
Positions (P)      →    Staypoints (S)    →    Locations (L)
Raw GPS points          Meaningful stops        Clustered places
(millions)              (thousands)             (hundreds)
```

### 2.2 Pre-Processing with Trackintel

The raw GPS data is pre-processed using the **Trackintel** library before entering this pipeline. The external notebook (`02_psl_detection_all.ipynb`) performs:

1. **GPS Point Loading**: Read raw position fixes from CSV
2. **GeoDataFrame Creation**: Convert to spatial format
3. **Staypoint Detection**: Using sliding window algorithm
4. **Trip Generation**: Generate trips between staypoints
5. **Quality Filtering**: Calculate user tracking quality

#### 2.2.1 Trackintel Staypoint Detection Parameters

The external preprocessing uses these parameters:

```python
pfs.generate_staypoints(
    method='sliding',
    distance_metric='haversine',
    dist_threshold=100,       # 100 meters
    time_threshold=30,        # 30 minutes
    gap_threshold=24*60,      # 24 hours
    print_progress=True,
    n_jobs=32
)
```

| Parameter | Value | Description |
|-----------|-------|-------------|
| `dist_threshold` | 100m | Spatial threshold for staypoint |
| `time_threshold` | 30min | Minimum duration for staypoint |
| `gap_threshold` | 1440min | Maximum gap before splitting |
| `method` | sliding | Sliding window algorithm |

### 2.3 External Quality Filtering

The DIY dataset includes pre-computed quality filtering with stricter criteria:

```python
# Quality filter applied externally:
quality_filter = {
    "day_filter": 60,      # Minimum 60 days (vs 50 for Geolife)
    "window_size": 10,     # 10-week sliding window
    "min_thres": 0.6,      # Minimum window quality ≥ 60%
    "mean_thres": 0.7      # Mean window quality ≥ 70%
}
```

### 2.4 Why Different Parameters?

| Parameter | Geolife | DIY | Reason |
|-----------|---------|-----|--------|
| `epsilon` | 20m | 50m | Sparser GPS in mobile data |
| `day_filter` | 50 | 60 | Ensure sufficient data |
| `split_ratio` | 60/20/20 | 80/10/10 | Larger dataset, need more training data |

---

## 3. Pipeline Overview

### 3.1 Complete Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DIY PREPROCESSING PIPELINE                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  EXTERNAL (02_psl_detection_all.ipynb)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ Raw GPS CSV → Trackintel → Staypoints + Quality Filter              │   │
│  │                                                                      │   │
│  │ Output Files:                                                        │   │
│  │ • 3_staypoints_fun_generate_trips.csv (970MB)                       │   │
│  │ • 10_filter_after_user_quality_DIY_slide_filteres.csv               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                    │                                        │
│                                    ▼                                        │
│  RAW INPUT              SCRIPT 1               INTERIM OUTPUT               │
│  ┌─────────────┐       ┌─────────────┐        ┌─────────────────┐          │
│  │ Staypoints  │       │ diy_1_raw   │        │ intermediate    │          │
│  │ CSV + Valid │  ───► │ _to_interim │  ────► │ _eps50.csv      │          │
│  │ Users CSV   │       │ .py         │        │                 │          │
│  └─────────────┘       └─────────────┘        └─────────────────┘          │
│                              │                       │                      │
│                        Parameters:              Also outputs:               │
│                        • epsilon=50             • locations_eps50.csv       │
│                                                 • staypoints_merged.csv     │
│                                                                             │
│  ────────────────────────────────────────────────────────────────────────  │
│                                                                             │
│                        SCRIPT 2                PROCESSED OUTPUT             │
│                       ┌─────────────┐         ┌─────────────────┐          │
│                       │ diy_2_      │         │ train/val/test  │          │
│                       │ interim_to_ │  ────►  │ .pk files       │          │
│                       │ processed.py│         │                 │          │
│                       └─────────────┘         └─────────────────┘          │
│                              │                       │                      │
│                        Parameters:              Also outputs:               │
│                        • previous_day          • metadata.json              │
│                        • split ratios                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Script Responsibilities

| Script | Input | Output | Key Operations |
|--------|-------|--------|----------------|
| External | Raw GPS | Staypoints + Quality | Detection, Quality |
| Script 1 | Staypoints | Intermediate | Clustering, Merging, Temporal |
| Script 2 | Intermediate | Sequences | Splitting, Encoding, Generation |

### 3.3 File Organization

```
data/
├── raw_diy/
│   ├── 3_staypoints_fun_generate_trips.csv      # 970MB, ~5M staypoints
│   └── 10_filter_after_user_quality_DIY_slide_filteres.csv  # Valid users
│
└── diy_eps50/
    ├── interim/
    │   ├── intermediate_eps50.csv               # Main intermediate data
    │   ├── locations_eps50.csv                  # Location centroids
    │   ├── staypoints_merged_eps50.csv          # After merging
    │   ├── valid_users_eps50.csv                # Filtered users
    │   └── interim_stats_eps50.json             # Statistics
    │
    └── processed/
        ├── diy_eps50_prev7_train.pk             # Training sequences
        ├── diy_eps50_prev7_validation.pk        # Validation sequences
        ├── diy_eps50_prev7_test.pk              # Test sequences
        └── diy_eps50_prev7_metadata.json        # Metadata
```

---

## 4. Pre-Processing: Raw Data Generation

### 4.1 The External Notebook

**File**: `/data/data_diy_viz/02_psl_detection_all.ipynb`

This Jupyter notebook runs on Google Colab and processes the raw DIY GPS data.

### 4.2 Raw GPS Data Format

**Input**: CSV file with columns:
```csv
user_id,latitude,longitude,tracked_at
9358664f-ad4b-46ff-9a65-e2efbf646e6e,-7.74776,110.4315414428711,2021-10-24T02:07:56.000Z
19b36aee-5acc-402f-a19b-c7f391d9361c,-7.81982,110.46869659423828,2021-10-24T03:05:21.000Z
```

**Key Characteristics**:
- **user_id**: UUID format
- **latitude**: Negative (Southern hemisphere)
- **longitude**: Around 110° (Central Java, Indonesia)
- **tracked_at**: ISO 8601 format with UTC timezone

### 4.3 External Processing Steps

#### Step 1: Load Raw GPS Data

```python
df = pd.read_csv('clean_gps_data_v11.csv')
df['tracked_at'] = pd.to_datetime(df['tracked_at'])

# Convert to GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"
)
gdf = gdf[['user_id', 'tracked_at', 'geometry']]
```

**Statistics**:
- Total rows: 165,429,633
- Total users: ~50,000+
- Memory: ~4.9 GB

#### Step 2: Convert to Trackintel Format

```python
import trackintel as ti

pfs = ti.io.read_positionfixes_gpd(
    gdf,
    tracked_at='tracked_at',
    user_id='user_id',
    geom_col='geometry',
    tz='Asia/Jakarta'  # UTC+7
)
```

#### Step 3: Generate Staypoints

```python
pfs, staypoints = pfs.generate_staypoints(
    method='sliding',
    distance_metric='haversine',
    dist_threshold=100,        # 100 meters
    time_threshold=30,         # 30 minutes
    gap_threshold=24*60,       # 24 hours (1440 minutes)
    print_progress=True,
    n_jobs=32                  # Parallel processing
)
```

**Output Columns**:
- `user_id`: Original UUID
- `started_at`: Staypoint start time
- `finished_at`: Staypoint end time
- `geometry`: POINT geometry (centroid)
- `is_activity`: Activity flag (duration ≥ 25 min)

#### Step 4: Generate Trips (for Quality Calculation)

```python
# Generate triplegs (movement segments)
pfs, triplegs = pfs.generate_triplegs(staypoints)

# Generate trips
sp_with_trips, tpls_with_trips, trips = staypoints.generate_trips(triplegs)
```

#### Step 5: Quality Filtering

```python
# Calculate temporal tracking quality
quality = temporal_tracking_quality(df_all, granularity="all")

# Sliding window quality
sliding_quality = df_all.groupby("user_id").apply(
    _get_tracking_quality,
    window_size=10  # 10-week windows
)

# Apply filters:
# 1. day_filter: ≥ 60 days of tracking
# 2. min_thres: Minimum window quality ≥ 0.6
# 3. mean_thres: Mean window quality ≥ 0.7
```

### 4.4 Output Files

**File 1**: `3_staypoints_fun_generate_trips.csv` (~970MB)

```csv
id,finished_at,geometry,is_activity,started_at,user_id,prev_trip_id,next_trip_id,trip_id
0,2021-12-16 06:54:08+00:00,POINT (110.4619674682617188 -7.7604059999999997),True,2021-12-15 06:30:26+00:00,00001a8b-69eb-4f44-809c-843b584f9797,,,
1,2021-12-19 08:45:35+00:00,POINT (110.4333900000000028 -7.7631170000000003),True,2021-12-18 13:22:50+00:00,00001a8b-69eb-4f44-809c-843b584f9797,,,
```

**File 2**: `10_filter_after_user_quality_DIY_slide_filteres.csv`

```csv
,user_id,quality
0,000316d6-8800-4964-92ea-b87c519cd2e6,0.8437084656084656
1,00169ae3-9851-4707-9582-c373668000c0,0.9243006944444444
```

---

## 5. Script 1: Raw to Interim

### 5.1 Overview

**File**: `preprocessing/diy_1_raw_to_interim.py`

**Purpose**: Transform pre-processed staypoints into intermediate format with locations and temporal features.

### 5.2 Key Difference from Geolife

| Step | Geolife | DIY |
|------|---------|-----|
| Read Data | Raw GPS trajectories | Pre-processed staypoints |
| Staypoint Detection | Yes | No (already done) |
| Quality Calculation | Yes | No (use pre-computed) |
| Activity Filter | Yes | Yes |
| Location Clustering | Yes | Yes |
| Staypoint Merging | Yes | Yes |
| Temporal Features | Yes | Yes |

### 5.3 Step-by-Step Processing

#### Stage 1: Load Raw Data

```python
def load_raw_data(config):
    """Load pre-processed staypoints and valid users."""
    
    # Read staypoints
    sp = ti.read_staypoints_csv(
        f'{raw_path}/3_staypoints_fun_generate_trips.csv',
        columns={'geometry': 'geom'},
        index_col='id'
    )
    
    # Read valid users (pre-computed)
    valid_user_df = pd.read_csv(
        f'{raw_path}/10_filter_after_user_quality_DIY_slide_filteres.csv'
    )
    valid_user = valid_user_df["user_id"].values
    
    # Filter to valid users
    sp = sp.loc[sp["user_id"].isin(valid_user)]
    
    # Filter to activity staypoints only
    sp = sp.loc[sp["is_activity"] == True]
    
    return sp, valid_user
```

**Input Statistics**:
- Total staypoints in file: ~5 million
- After user filter: ~400,000
- After activity filter: ~350,000

#### Stage 2: Generate Locations (DBSCAN)

```python
def generate_locations(sp, config, interim_dir, epsilon):
    """Cluster staypoints into locations using DBSCAN."""
    
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=50,                # 50 meters (larger than Geolife's 20m)
        num_samples=2,             # Minimum 2 staypoints per cluster
        distance_metric="haversine",
        agg_level="dataset",       # Cluster all users together
        n_jobs=-1
    )
    
    # Filter noise staypoints (location_id = NaN)
    sp = sp.loc[~sp["location_id"].isna()].copy()
    
    # Save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    filtered_locs.as_locations.to_csv(f"{interim_dir}/locations_eps{epsilon}.csv")
    
    return sp, filtered_locs
```

**Why ε=50m for DIY?**
- Mobile GPS is less accurate than dedicated GPS devices
- Urban density in Yogyakarta is different from Beijing
- Empirically chosen for good location granularity

#### Stage 3: Merge Consecutive Staypoints

```python
def merge_staypoints(sp, config, interim_dir, epsilon):
    """Merge consecutive staypoints at the same location."""
    
    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap="1min",           # Merge if gap < 1 minute
        agg={"location_id": "first"}   # Keep first location_id
    )
    
    # Recalculate duration after merging
    sp_merged["duration"] = (
        sp_merged["finished_at"] - sp_merged["started_at"]
    ).dt.total_seconds() // 60  # Convert to minutes
    
    return sp_merged
```

#### Stage 4: Extract Temporal Features

```python
def _get_time(df):
    """Extract temporal features for DIY data."""
    # Reference: User's first staypoint date
    min_day = pd.to_datetime(df["started_at"].min().date())
    
    # Remove timezone for calculation
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)
    
    # Calculate day indices (relative to user's start)
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


def enrich_time_info(sp):
    """Add temporal features to all staypoints."""
    sp = sp.groupby("user_id", group_keys=False).apply(_get_time)
    
    # Handle UUID user_ids (convert to integer)
    if sp["user_id"].dtype == 'object':
        unique_users = sp["user_id"].unique()
        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        sp["user_id"] = sp["user_id"].map(user_mapping)
    
    sp["location_id"] = sp["location_id"].astype(int)
    
    # Reassign sequential IDs
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    
    return sp
```

### 5.4 Output Files

| File | Description | Size |
|------|-------------|------|
| `intermediate_eps50.csv` | Main intermediate data | ~15MB |
| `locations_eps50.csv` | Location centroids | ~2MB |
| `staypoints_merged_eps50.csv` | After merging | ~20MB |
| `valid_users_eps50.csv` | Filtered user list | ~100KB |
| `interim_stats_eps50.json` | Statistics | <1KB |

---

## 6. Script 2: Interim to Processed

### 6.1 Overview

**File**: `preprocessing/diy_2_interim_to_processed.py`

**Purpose**: Convert intermediate data to model-ready sequences.

### 6.2 Key Differences from Geolife

| Aspect | Geolife | DIY |
|--------|---------|-----|
| Split Ratio | 60/20/20 | 80/10/10 |
| Val Label | "val" | "vali" |
| Parallelization | Sequential | joblib parallel |

### 6.3 Step-by-Step Processing

#### Step 1: Load Intermediate Data

```python
interim_file = os.path.join(interim_path, f"intermediate_eps{epsilon}.csv")
sp = pd.read_csv(interim_file)
```

**Statistics**:
```
Loaded 265,621 staypoints from 1,306 users
```

#### Step 2: Truncate Duration

```python
max_duration = 2880  # 2 days in minutes (same as Geolife)

sp_copy = sp.copy()
sp_copy.loc[sp_copy["duration"] > max_duration - 1, "duration"] = max_duration - 1
```

**Note**: DIY has some very long durations (up to 95,230 minutes = 66 days), so truncation is important.

#### Step 3: Split Dataset

```python
def _get_split_days_user(df, split_ratios):
    """Split chronologically per user (80/10/10 for DIY)."""
    maxDay = df["start_day"].max()
    
    # DIY uses 80/10/10 split
    train_split = maxDay * 0.8       # 0 to 80%
    validation_split = maxDay * 0.9  # 80% to 90%
    
    df["Dataset"] = "test"  # Default: 90% to 100%
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & 
           (df["start_day"] < validation_split), "Dataset"] = "vali"
    
    return df
```

**Why 80/10/10 for DIY?**
- Larger dataset → more training data beneficial
- Shorter tracking period → smaller val/test still meaningful
- More users → diversity maintained in smaller splits

#### Step 4: Encode Location IDs

```python
# Same encoding scheme as Geolife
# 0 = padding, 1 = unknown, 2+ = known locations

enc = OrdinalEncoder(
    dtype=np.int64,
    handle_unknown="use_encoded_value",
    unknown_value=-1
).fit(train_data["location_id"].values.reshape(-1, 1))

train_data["location_id"] = enc.transform(...) + 2
vali_data["location_id"] = enc.transform(...) + 2
test_data["location_id"] = enc.transform(...) + 2
```

#### Step 5: Valid Sequence Identification

```python
def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """
    Same logic as Geolife:
    1. Target ≥ previous_day days from user's start
    2. History window contains ≥ min_length staypoints
    """
    valid_id = []
    
    for user in input_df["user_id"].unique():
        df = input_df.loc[input_df["user_id"] == user].copy()
        df = df.reset_index(drop=True)
        
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days
        
        for index, row in df.iterrows():
            if row["diff_day"] < previous_day:
                continue
            
            hist = df.iloc[:index]
            hist = hist.loc[hist["start_day"] >= (row["start_day"] - previous_day)]
            
            if len(hist) < min_length:
                continue
            
            valid_id.append(row["id"])
    
    return valid_id
```

#### Step 6: Generate Sequences (Parallel)

```python
from joblib import Parallel, delayed, parallel_backend

def generate_sequences(data, valid_ids, previous_day, split_name):
    """Generate sequences using parallel processing."""
    
    valid_ids_set = set(valid_ids)
    
    # Prepare arguments for parallel processing
    user_groups = [
        (group.copy(), previous_day, valid_ids_set)
        for _, group in data.groupby("user_id")
    ]
    
    # Parallel execution
    with parallel_backend("threading", n_jobs=-1):
        valid_user_ls = Parallel()(
            delayed(_get_valid_sequence_user)(args)
            for args in tqdm(user_groups, desc=f"{split_name}")
        )
    
    # Flatten results
    valid_records = [item for sublist in valid_user_ls for item in sublist]
    
    return valid_records
```

**Why Parallel for DIY?**
- 1,306 users vs 91 for Geolife
- Parallelization provides significant speedup
- Threading backend for I/O-bound operations

#### Step 7: Save Output Files

```python
# Same format as Geolife
with open(f"{output_name}_train.pk", "wb") as f:
    pickle.dump(train_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)

# Metadata includes DIY-specific info
metadata = {
    "dataset_name": "diy",
    "epsilon": 50,
    "previous_day": 7,
    "total_user_num": 693,
    "total_loc_num": 7038,
    # ... same structure as Geolife
}
```

---

## 7. Configuration Reference

### 7.1 Full Configuration File

```yaml
# config/preprocessing/diy.yaml

dataset:
  name: "diy"
  epsilon: 50              # DBSCAN epsilon (larger than Geolife)
  previous_day: [7]        # History window(s)

preprocessing:
  # Location clustering
  location:
    num_samples: 2
    distance_metric: "haversine"
    agg_level: "dataset"
  
  # Staypoint merging
  staypoint_merging:
    max_time_gap: "1min"   # Merge if gap < 1 minute
  
  # Quality filter (pre-computed, used for validation)
  quality_filter:
    day_filter: 60         # Minimum tracking days
    window_size: 10        # Sliding window weeks
    min_thres: 0.6         # Minimum window quality
    mean_thres: 0.7        # Mean window quality
  
  # Duration handling
  max_duration: 2880       # Cap at 2 days
  min_sequence_length: 3   # Minimum history length
  
  # Data splits (different from Geolife)
  split:
    train: 0.8             # 80% for training
    val: 0.1               # 10% for validation
    test: 0.1              # 10% for testing

random_seed: 42
```

### 7.2 Parameter Comparison

| Parameter | Geolife | DIY | Reason |
|-----------|---------|-----|--------|
| `epsilon` | 20m | 50m | Mobile GPS less accurate |
| `day_filter` | 50 | 60 | Stricter quality |
| `min_thres` | N/A | 0.6 | Additional quality check |
| `mean_thres` | N/A | 0.7 | Additional quality check |
| `train` | 0.6 | 0.8 | More training data |
| `val` | 0.2 | 0.1 | Smaller but sufficient |
| `test` | 0.2 | 0.1 | Smaller but sufficient |

---

## 8. Data Formats

### 8.1 Raw Input Format

**File**: `data/raw_diy/3_staypoints_fun_generate_trips.csv`

```csv
id,finished_at,geometry,is_activity,started_at,user_id,prev_trip_id,next_trip_id,trip_id
0,2021-12-16 06:54:08+00:00,POINT (110.4619674682617188 -7.7604059999999997),True,2021-12-15 06:30:26+00:00,00001a8b-69eb-4f44-809c-843b584f9797,,,
```

| Column | Type | Description |
|--------|------|-------------|
| `id` | int | Staypoint identifier |
| `finished_at` | datetime | End timestamp (UTC) |
| `geometry` | WKT | POINT geometry |
| `is_activity` | bool | Activity flag (≥25 min) |
| `started_at` | datetime | Start timestamp (UTC) |
| `user_id` | UUID | User identifier |
| `prev_trip_id` | int/null | Previous trip ID |
| `next_trip_id` | int/null | Next trip ID |
| `trip_id` | int/null | Associated trip ID |

### 8.2 Valid Users Format

**File**: `data/raw_diy/10_filter_after_user_quality_DIY_slide_filteres.csv`

```csv
,user_id,quality
0,000316d6-8800-4964-92ea-b87c519cd2e6,0.8437084656084656
1,00169ae3-9851-4707-9582-c373668000c0,0.9243006944444444
```

| Column | Type | Description |
|--------|------|-------------|
| (index) | int | Row number |
| `user_id` | UUID | User identifier |
| `quality` | float | Mean tracking quality (0-1) |

### 8.3 Intermediate Format

**File**: `data/diy_eps50/interim/intermediate_eps50.csv`

```csv
id,user_id,location_id,duration,start_day,end_day,start_min,end_min,weekday
0,0,0,5347.0,0,3,42,1069,1
1,0,0,87.0,4,4,1129,1216,5
2,0,1,131.0,4,4,1216,1348,5
```

**Note**: User IDs are converted from UUIDs to integers.

### 8.4 Processed Format

**Same structure as Geolife**:

```python
{
    'X': np.array([...]),           # Location IDs (history)
    'user_X': np.array([...]),      # User IDs
    'weekday_X': np.array([...]),   # Weekdays (0-6)
    'start_min_X': np.array([...]), # Start minutes (0-1439)
    'dur_X': np.array([...]),       # Durations (minutes)
    'diff': np.array([...]),        # Days to target
    'Y': int                         # Target location
}
```

---

## 9. Statistical Analysis

### 9.1 Raw Data Statistics

| Metric | Value |
|--------|-------|
| Total GPS Points | 165,429,633 |
| Total Unique Users | ~50,000+ |
| Date Range | Oct 2021 - Jun 2022 |
| Primary Location | Yogyakarta, Indonesia |

### 9.2 After External Quality Filter

| Metric | Value |
|--------|-------|
| Valid Users | 1,306 |
| Total Staypoints | ~350,000 |
| Mean Quality | 0.85 |
| Min Tracking Days | 60 |

### 9.3 Interim Statistics (ε=50m)

```json
{
  "epsilon": 50,
  "total_staypoints": 265621,
  "total_users": 1306,
  "total_locations": 8439,
  "staypoints_per_user_mean": 203.38,
  "duration_mean_min": 774.60,
  "duration_median_min": 303.0,
  "duration_max_min": 95230.0,
  "days_tracked_mean": 128.57
}
```

### 9.4 Processed Statistics (prev7)

```json
{
  "total_user_num": 693,
  "total_loc_num": 7038,
  "unique_users": 692,
  "unique_locations": 7036,
  "train_sequences": 151421,
  "val_sequences": 10160,
  "test_sequences": 12368,
  "total_sequences": 173949
}
```

### 9.5 Comparison Summary

| Metric | Geolife | DIY | Ratio |
|--------|---------|-----|-------|
| Users | 45 | 692 | 15x |
| Locations | 1,185 | 7,036 | 6x |
| Train Sequences | 7,424 | 151,421 | 20x |
| Total Sequences | 14,260 | 173,949 | 12x |

---

## 10. Troubleshooting

### 10.1 Common Issues

#### Issue: "Memory error loading staypoints"

**Cause**: 970MB staypoints file is too large

**Solution**:
```python
# Read in chunks
chunks = pd.read_csv(file, chunksize=100000)
sp_list = []
for chunk in chunks:
    sp_list.append(chunk[chunk['user_id'].isin(valid_users)])
sp = pd.concat(sp_list)
```

#### Issue: "UUID user_ids causing errors"

**Cause**: User IDs are UUIDs instead of integers

**Solution**: Already handled in `enrich_time_info()`:
```python
if sp["user_id"].dtype == 'object':
    unique_users = sp["user_id"].unique()
    user_mapping = {user: idx for idx, user in enumerate(unique_users)}
    sp["user_id"] = sp["user_id"].map(user_mapping)
```

#### Issue: "Very long durations skewing results"

**Cause**: Some staypoints have duration up to 95,230 minutes

**Solution**: Duration truncation in Script 2:
```python
sp_copy.loc[sp_copy["duration"] > 2879, "duration"] = 2879
```

### 10.2 Validation Checks

```python
"""
DIY preprocessing validation script.
"""
import pickle
import json

# Load processed data
with open("data/diy_eps50/processed/diy_eps50_prev7_train.pk", "rb") as f:
    train = pickle.load(f)

with open("data/diy_eps50/processed/diy_eps50_prev7_metadata.json") as f:
    meta = json.load(f)

# Verify counts
print(f"Train: {len(train)} (expected: {meta['train_sequences']})")
assert len(train) == meta['train_sequences']

# Verify location range
all_Y = [s['Y'] for s in train]
print(f"Y range: {min(all_Y)} to {max(all_Y)}")
print(f"Max location: {meta['total_loc_num'] - 1}")
assert max(all_Y) < meta['total_loc_num']

# Check sequence lengths
lengths = [len(s['X']) for s in train]
print(f"Min length: {min(lengths)}, Max length: {max(lengths)}")
assert min(lengths) >= 3

print("✓ All validations passed")
```

---

## 11. Comparison with Geolife

### 11.1 Pipeline Differences

```
GEOLIFE PIPELINE:
Raw GPS → Staypoint Detection → Quality Filter → Location Cluster → ...

DIY PIPELINE:
[External: Staypoint Detection + Quality Filter] → Location Cluster → ...
```

### 11.2 Parameter Differences

| Parameter | Geolife | DIY | Reason |
|-----------|---------|-----|--------|
| Staypoint Detection | In pipeline | External | Pre-processed data |
| Quality Calculation | In pipeline | External | Pre-computed |
| Epsilon | 20m | 50m | GPS accuracy |
| Split Ratio | 60/20/20 | 80/10/10 | Dataset size |

### 11.3 Output Scale

| Metric | Geolife | DIY | DIY/Geolife |
|--------|---------|-----|-------------|
| Final Users | 45 | 692 | 15.4x |
| Final Locations | 1,185 | 7,036 | 5.9x |
| Train Sequences | 7,424 | 151,421 | 20.4x |
| Total Sequences | 14,260 | 173,949 | 12.2x |

### 11.4 Use Cases

| Dataset | Best For |
|---------|----------|
| Geolife | Prototype development, algorithm testing |
| DIY | Production models, scale testing |

---

## 12. References

### 12.1 Libraries Used

- **Trackintel**: https://github.com/mie-lab/trackintel
- **GeoPandas**: https://geopandas.org/
- **Scikit-learn**: https://scikit-learn.org/ (DBSCAN, OrdinalEncoder)
- **Joblib**: https://joblib.readthedocs.io/ (Parallel processing)

### 12.2 Algorithm References

- **DBSCAN**: Ester et al., "A density-based algorithm for discovering clusters" (KDD 1996)
- **Staypoint Detection**: Zheng et al., "Learning transportation mode from raw GPS data" (UbiComp 2008)

### 12.3 Related Documentation

- [DIY Walkthrough with Examples](./DIY_PREPROCESSING_WALKTHROUGH.md)
- [Geolife Preprocessing Documentation](./GEOLIFE_PREPROCESSING_DOCUMENTATION.md)
- [Configuration Reference](../../config/preprocessing/diy.yaml)
- [External Notebook](../../notebooks/02_psl_detection_all.ipynb)

---

*Last updated: January 2026*
*Version: 1.0*
