# PSL Detection Notebook Documentation

## 📋 Table of Contents
1. [Overview](#overview)
2. [Notebook Purpose](#notebook-purpose)
3. [Complete Workflow](#complete-workflow)
4. [Section-by-Section Walkthrough](#section-by-section-walkthrough)
5. [Output Files](#output-files)
6. [Quality Filtering Deep Dive](#quality-filtering-deep-dive)
7. [Code Examples with Explanations](#code-examples-with-explanations)

---

## Overview

**File**: `preprocessing/02_psl_detection_all.ipynb`  
**Purpose**: Transform raw GPS trajectory data into staypoints with quality filtering  
**Framework**: Trackintel (mobility analytics library)

### What is PSL?
**P**ositions → **S**taypoints → **L**ocations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PSL PIPELINE                                    │
└─────────────────────────────────────────────────────────────────────────────┘

   Raw GPS Points          Staypoints              Locations
   ═══════════════         ═══════════             ═══════════
   
   • • •                   ┌─────────┐             ┌───────────┐
   •   •                   │ Stay 1  │             │ Location  │
   • • •                   │ (Home)  │─────────────│     1     │
                           └─────────┘             │  (Home)   │
       ↓                                           └───────────┘
   • •                     ┌─────────┐                  │
     •                     │ Stay 2  │                  │
   • •                     │(Office) │─────────────────┬┘
                           └─────────┘             ┌───────────┐
                                                   │ Location  │
   • • • • •               ┌─────────┐             │     2     │
   • • • • •               │ Stay 3  │─────────────│ (Office)  │
                           │(Office) │             └───────────┘
                           └─────────┘

   165M points    →        ~500K staypoints   →    ~50K locations
```

---

## Notebook Purpose

This notebook performs the foundational preprocessing that transforms raw GPS data into the two critical input files for the downstream processing scripts:

### Primary Outputs

| Output File | Description | Used By |
|-------------|-------------|---------|
| `3_staypoints_fun_generate_trips.csv` | All detected staypoints with activity flags and trip associations | `diy_1_raw_to_interim.py`, `diy_h3_1_raw_to_interim.py` |
| `10_filter_after_user_quality_DIY_slide_filteres.csv` | User IDs that pass quality filtering | `diy_1_raw_to_interim.py`, `diy_h3_1_raw_to_interim.py` |

---

## Complete Workflow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         NOTEBOOK WORKFLOW DIAGRAM                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌──────────────────────────┐
                    │   Raw GPS CSV File       │
                    │   (165M+ points)         │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Load & Parse Data      │
                    │   • Read CSV             │
                    │   • Parse timestamps     │
                    │   • Create geometry      │
                    └────────────┬─────────────┘
                                 │
                    ┌────────────▼─────────────┐
                    │   Create Positionfixes   │
                    │   (Trackintel format)    │
                    └────────────┬─────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
          ▼                      ▼                      ▼
┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
│ Generate        │   │ Generate        │   │ Generate        │
│ Staypoints      │   │ Triplegs        │   │ Trips           │
│ (sliding window)│   │ (movement)      │   │ (journeys)      │
└────────┬────────┘   └────────┬────────┘   └────────┬────────┘
         │                     │                     │
         └──────────────┬──────┴─────────────────────┘
                        │
                        ▼
          ┌─────────────────────────────┐
          │   Create Activity Flag      │
          │   (is_activity = True if    │
          │    duration > 25 min)       │
          └─────────────┬───────────────┘
                        │
          ╔═════════════▼═══════════════╗
          ║   OUTPUT FILE 1:            ║
          ║   3_staypoints_fun_         ║
          ║   generate_trips.csv        ║
          ╚═════════════╤═══════════════╝
                        │
          ┌─────────────▼───────────────┐
          │   Quality Filtering         │
          │   ┌───────────────────────┐ │
          │   │ 1. Day Filter (>60)   │ │
          │   │ 2. Sliding Window     │ │
          │   │ 3. Min Quality (0.6)  │ │
          │   │ 4. Mean Quality (0.7) │ │
          │   └───────────────────────┘ │
          └─────────────┬───────────────┘
                        │
          ╔═════════════▼═══════════════╗
          ║   OUTPUT FILE 2:            ║
          ║   10_filter_after_user_     ║
          ║   quality_DIY_slide_        ║
          ║   filteres.csv              ║
          ╚═════════════════════════════╝
```

---

## Section-by-Section Walkthrough

### Section 1: Setup and Imports (Cells 1-6)

```python
# Cell 1: Mount Google Drive (for Colab)
from google.colab import drive
drive.mount('/content/drive')

# Cell 2: Install dependencies
!pip install geopandas==1.1.1 trackintel

# Cell 3: Import libraries
import pandas as pd
import trackintel as ti
import geopandas as gpd
import tqdm.auto
import time
import warnings
from datetime import datetime
import glob

# Configure display and warnings
tqdm.auto.tqdm = tqdm.auto.tqdm
pd.set_option("display.precision", 15)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

# Cell 4: Verify versions
print("trackintel version:", ti.__version__)  # 1.4.1
print("geopandas version:", gpd.__version__)  # 1.1.1
```

**Why these versions?**
- Trackintel 1.4.1: Latest stable version with all required functions
- GeoPandas 1.1.1: Compatible with Trackintel's geometry operations

### Section 2: Load Raw GPS Data (Cells 8-16)

```python
# Cell 8: Preview data
!head /path/to/clean_gps_data.csv

# Output:
# user_id,latitude,longitude,tracked_at
# 9358664f-ad4b-46ff-9a65-e2efbf646e6e,-7.74776,110.4315414428711,2021-10-24T02:07:56.000Z

# Cell 9: Count records
!wc -l /path/to/clean_gps_data.csv
# Output: 165429634 (165+ million lines)

# Cell 10: Load into DataFrame
df = pd.read_csv('/path/to/clean_gps_data.csv')

# Cell 13: Parse timestamps
df['tracked_at'] = pd.to_datetime(df['tracked_at'])

# Cell 16: Create GeoDataFrame
gdf = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df.longitude, df.latitude),
    crs="EPSG:4326"  # WGS84 coordinate system
)
gdf = gdf[['user_id', 'tracked_at', 'geometry']]
```

**What the data looks like:**

```
Before (CSV):
┌──────────────────────────────────────────┬───────────┬─────────────┬─────────────────────────┐
│ user_id                                  │ latitude  │ longitude   │ tracked_at              │
├──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────────┤
│ 9358664f-ad4b-46ff-9a65-e2efbf646e6e    │ -7.74776  │ 110.431541  │ 2021-10-24T02:07:56.000Z│
└──────────────────────────────────────────┴───────────┴─────────────┴─────────────────────────┘

After (GeoDataFrame):
┌──────────────────────────────────────────┬─────────────────────────────────┬───────────────────────────┐
│ user_id                                  │ tracked_at                      │ geometry                  │
├──────────────────────────────────────────┼─────────────────────────────────┼───────────────────────────┤
│ 9358664f-ad4b-46ff-9a65-e2efbf646e6e    │ 2021-10-24 09:07:56+07:00       │ POINT(110.4315 -7.7478)   │
└──────────────────────────────────────────┴─────────────────────────────────┴───────────────────────────┘
```

### Section 3: Create Positionfixes (Cell 18)

```python
# Cell 18: Convert to Trackintel positionfixes format
pfs = ti.io.read_positionfixes_gpd(
    gdf,
    tracked_at='tracked_at',
    user_id='user_id',
    geom_col='geometry',
    tz='Asia/Jakarta'  # Local timezone for DIY dataset (Indonesia)
)
```

**Why timezone matters:**
- GPS timestamps are often in UTC
- Local timezone ensures correct day/time calculations
- Asia/Jakarta = UTC+7 (Indonesia Western Time)

### Section 4: Generate Staypoints (Cell 20)

```python
# Cell 20: Detect staypoints using sliding window algorithm
pfs, staypoints = pfs.generate_staypoints(
    method='sliding',           # Sliding window detection algorithm
    distance_metric='haversine', # Great-circle distance (for lat/lon)
    dist_threshold=100,         # Maximum distance within staypoint (meters)
    time_threshold=30,          # Minimum duration for staypoint (minutes)
    gap_threshold=24*60,        # Maximum gap in tracking (minutes)
    print_progress=True,
    n_jobs=-1                   # Use all CPU cores
)
```

**Staypoint Detection Algorithm Explained:**

```
SLIDING WINDOW STAYPOINT DETECTION
═══════════════════════════════════════════════════════════════════════════════

Parameters:
• dist_threshold = 100m (radius of stationary area)
• time_threshold = 30min (minimum stay duration)
• gap_threshold = 24h (maximum gap before reset)

Algorithm Visualization:

Timeline:  08:00  08:05  08:10  08:15  08:20  08:25  08:30  08:35  08:40
Position:    •      •      •      •      •      •      •      •      •
             └──────────────────────────────────────────┘      │
                          All within 100m radius                │
                          Duration = 30 min                     │
                          → STAYPOINT DETECTED!                 │
                                                                │
                                                         ← Movement starts
                                                           (exceeds 100m)


Example Sequence:
┌───────┬───────────┬────────────┬────────────────┬────────────────────────────────┐
│ Time  │ Lat       │ Lon        │ Dist from Prev │ Status                         │
├───────┼───────────┼────────────┼────────────────┼────────────────────────────────┤
│ 08:00 │ -7.7478   │ 110.4315   │ 0              │ Start window                   │
│ 08:05 │ -7.7478   │ 110.4316   │ 10m            │ Within 100m, continue          │
│ 08:10 │ -7.7477   │ 110.4315   │ 15m            │ Within 100m, continue          │
│ 08:15 │ -7.7479   │ 110.4314   │ 20m            │ Within 100m, continue          │
│ 08:20 │ -7.7478   │ 110.4315   │ 12m            │ Within 100m, continue          │
│ 08:25 │ -7.7477   │ 110.4316   │ 18m            │ Within 100m, continue          │
│ 08:30 │ -7.7478   │ 110.4315   │ 15m            │ Within 100m, 30min elapsed     │
│       │           │            │                │ → STAYPOINT: 08:00-08:30       │
│ 08:35 │ -7.7600   │ 110.4400   │ 1500m          │ Outside 100m, movement start   │
└───────┴───────────┴────────────┴────────────────┴────────────────────────────────┘
```

### Section 5: Create Activity Flag (Cell 23)

```python
# Cell 23: Mark staypoints as activities based on duration
staypoints = staypoints.as_staypoints.create_activity_flag(
    method="time_threshold",
    time_threshold=25  # minutes
)
```

**What is an Activity?**

```
ACTIVITY FLAG DETERMINATION
═══════════════════════════════════════════════════════════════════════════════

Threshold: 25 minutes

Duration >= 25 min  →  is_activity = True   (Meaningful visit)
Duration <  25 min  →  is_activity = False  (Transit stop)

Examples:
┌──────────────────────────────────┬──────────────┬─────────────┬──────────────┐
│ Location                         │ Duration     │ is_activity │ Interpretation│
├──────────────────────────────────┼──────────────┼─────────────┼──────────────┤
│ Home                             │ 8 hours      │ True        │ Living       │
│ Office                           │ 4 hours      │ True        │ Working      │
│ Restaurant                       │ 45 minutes   │ True        │ Dining       │
│ Bus stop                         │ 10 minutes   │ False       │ Waiting      │
│ ATM                              │ 5 minutes    │ False       │ Transaction  │
│ Traffic light                    │ 3 minutes    │ False       │ Transit      │
└──────────────────────────────────┴──────────────┴─────────────┴──────────────┘

Why filter non-activities?
• Reduces noise from brief stops
• Focuses on semantically meaningful locations
• Improves prediction relevance
```

### Section 6: Generate Triplegs and Trips (Cells 24-33)

```python
# Cell 24: Generate triplegs (movement segments between staypoints)
pfs, tpls = pfs.generate_triplegs(staypoints)

# Cell 30: Generate trips (complete journeys from origin to destination)
from trackintel.preprocessing.triplegs import generate_trips
staypoints, tpls, trips = generate_trips(staypoints, tpls, add_geometry=True)

# Cell 32: Save staypoints with trip associations
staypoints.to_csv('.../3_staypoints_fun_generate_trips.csv')
```

**Understanding Triplegs vs Trips:**

```
TRIPLEGS AND TRIPS VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

                    Staypoint 1          Staypoint 2          Staypoint 3
                       (Home)              (Coffee)             (Office)
                         ●─────────────────────●─────────────────────●
                              Tripleg 1              Tripleg 2
                         │◄─────────────────────────────────────────▶│
                                         Trip 1
                                   (Home to Office)


Definitions:
• STAYPOINT: Where user stays (stationary period)
• TRIPLEG: Single movement segment between consecutive staypoints
• TRIP: Complete journey from origin to final destination

Trip associations in staypoints:
┌────┬──────────┬──────────┬──────────┬───────────────┬───────────────┬───────────────┐
│ id │ user_id  │ geom     │ is_act   │ trip_id       │ prev_trip_id  │ next_trip_id  │
├────┼──────────┼──────────┼──────────┼───────────────┼───────────────┼───────────────┤
│ 0  │ user_1   │ POINT... │ True     │ NaN           │ NaN           │ 1             │
│ 1  │ user_1   │ POINT... │ False    │ 1             │ NaN           │ NaN           │
│ 2  │ user_1   │ POINT... │ True     │ NaN           │ 1             │ 2             │
└────┴──────────┴──────────┴──────────┴───────────────┴───────────────┴───────────────┘
```

---

## Section 7: Quality Filtering (Cells 44-107)

This is the **most critical section** for ensuring data quality.

### Step 7.1: Calculate Tracking Days (Cells 78-89)

```python
# Cell 89: Filter users with at least 60 days of tracking
quality_filter = {"day_filter": 60}

user_filter_day = (
    total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
    .reset_index(drop=True)["user_id"]
    .unique()
)
```

**Why 60 days minimum?**

```
DAY FILTER JUSTIFICATION
═══════════════════════════════════════════════════════════════════════════════

Minimum Tracking Days: 60

Reasoning:
┌────────────────────────────────────────────────────────────────────────────────┐
│ 1. SUFFICIENT TRAINING DATA                                                    │
│    • With train=80%, 60 days gives 48 days of training                        │
│    • Enough staypoints for meaningful patterns                                 │
│                                                                                │
│ 2. VALID SPLITS                                                                │
│    • Val (10%) = 6 days, Test (10%) = 6 days                                  │
│    • Each split has reasonable coverage                                        │
│                                                                                │
│ 3. WEEKLY PATTERNS                                                             │
│    • 60 days ≈ 8.5 weeks                                                       │
│    • Captures multiple cycles of weekly routines                               │
│                                                                                │
│ 4. PREVIOUS_DAY REQUIREMENT                                                    │
│    • With previous_day=7, need at least 7 days before first valid target      │
│    • 60 days ensures plenty of valid sequences                                 │
└────────────────────────────────────────────────────────────────────────────────┘

User Distribution:
┌───────────────┬──────────────┐
│ Tracking Days │ User Count   │
├───────────────┼──────────────┤
│ < 30 days     │ Many         │ → Excluded (insufficient data)
│ 30-60 days    │ Some         │ → Excluded (borderline)
│ > 60 days     │ Selected     │ → KEPT for analysis
└───────────────┴──────────────┘
```

### Step 7.2: Sliding Window Quality (Cells 91-96)

```python
# Cell 91: Define sliding window quality function
from datetime import datetime, timedelta, time

def _get_tracking_quality(df, window_size):
    """Calculate tracking quality in sliding windows."""
    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    # Iterate through sliding windows
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.combine(start_date + timedelta(weeks=i), time())
        curr_end = datetime.combine(curr_start + timedelta(weeks=window_size), time())

        # Get records in this window
        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & 
                          (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        
        # Calculate quality = tracked_time / total_time
        total_sec = (curr_end - curr_start).total_seconds()
        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret

# Cell 93: Apply sliding window quality
quality_filter = {"window_size": 10}  # 10 weeks

sliding_quality = (
    df_all_gt0.groupby("user_id")
    .apply(_get_tracking_quality, window_size=quality_filter["window_size"])
    .reset_index(drop=True)
)
```

**Sliding Window Quality Visualization:**

```
SLIDING WINDOW QUALITY CALCULATION
═══════════════════════════════════════════════════════════════════════════════

Window Size: 10 weeks

User Timeline (20 weeks of tracking):
Week: 0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20
      ├──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┼──┤
      
Window 0:  [════════════════════]                              Quality=0.75
Window 1:     [════════════════════]                           Quality=0.82
Window 2:        [════════════════════]                        Quality=0.78
...
Window 10:                            [════════════════════]   Quality=0.85


Quality Calculation for One Window:

Window: Week 0-10 (10 weeks = 70 days = 1,680 hours)

Tracked Time = Sum of all (staypoint + tripleg) durations
             = 1,260 hours

Quality = 1,260 / 1,680 = 0.75 (75%)

Interpretation:
• Quality of 0.75 means user was tracked 75% of the time
• 25% is untracked (phone off, indoors, etc.)
```

### Step 7.3: Apply Quality Thresholds (Cells 98-107)

```python
# Cell 100: Define user filter function
def _filter_user(df, min_thres, mean_thres):
    """Filter user based on quality thresholds."""
    consider = df.loc[df["quality"] != 0]
    if (consider["quality"].min() > min_thres) and \
       (consider["quality"].mean() > mean_thres):
        return df
    # Return None (filtered out) if thresholds not met

# Cell 101: Apply quality filter
quality_filter = {"min_thres": 0.6, "mean_thres": 0.7}

filter_after_day = (
    filter_after_day.groupby("user_id")
    .apply(_filter_user, 
           min_thres=quality_filter["min_thres"], 
           mean_thres=quality_filter["mean_thres"])
    .reset_index(drop=True)
    .dropna()
)

# Cell 107: Save final valid users
filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()
filter_after_user_quality.to_csv('.../10_filter_after_user_quality_DIY_slide_filteres.csv')
```

**Quality Threshold Decision Tree:**

```
QUALITY FILTERING DECISION TREE
═══════════════════════════════════════════════════════════════════════════════

For each user:

                         Tracking Days > 60?
                                │
                    ┌───────────┴───────────┐
                   NO                       YES
                    │                        │
                  REJECT              Calculate Sliding
                                     Window Quality
                                            │
                                            ▼
                            Min Quality in ANY window > 0.6?
                                            │
                                ┌───────────┴───────────┐
                               NO                       YES
                                │                        │
                              REJECT              Mean Quality > 0.7?
                                                        │
                                            ┌───────────┴───────────┐
                                           NO                       YES
                                            │                        │
                                          REJECT                  ACCEPT ✓


Example Users:
┌─────────┬───────────┬─────────────┬──────────────┬────────────┐
│ User    │ Days      │ Min Quality │ Mean Quality │ Result     │
├─────────┼───────────┼─────────────┼──────────────┼────────────┤
│ User A  │ 45        │ -           │ -            │ REJECT     │
│ User B  │ 80        │ 0.45        │ 0.75         │ REJECT     │
│ User C  │ 90        │ 0.65        │ 0.68         │ REJECT     │
│ User D  │ 75        │ 0.62        │ 0.78         │ ACCEPT ✓   │
│ User E  │ 120       │ 0.71        │ 0.85         │ ACCEPT ✓   │
└─────────┴───────────┴─────────────┴──────────────┴────────────┘
```

---

## Output Files

### File 1: Staypoints (`3_staypoints_fun_generate_trips.csv`)

```
STAYPOINTS OUTPUT FILE
═══════════════════════════════════════════════════════════════════════════════

File: 3_staypoints_fun_generate_trips.csv
Format: CSV (Trackintel staypoints format)

Columns:
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ id             │ int64        │ Unique staypoint identifier (index)            │
│ user_id        │ string       │ UUID of user                                   │
│ started_at     │ datetime     │ Start timestamp                                │
│ finished_at    │ datetime     │ End timestamp                                  │
│ geom           │ WKT          │ Point geometry (POINT(lon lat))                │
│ is_activity    │ bool         │ True if duration > 25 minutes                  │
│ trip_id        │ int64/NaN    │ Trip ID (if middle of trip)                    │
│ prev_trip_id   │ int64/NaN    │ Previous trip ID (origin staypoint)            │
│ next_trip_id   │ int64/NaN    │ Next trip ID (destination staypoint)           │
└────────────────┴──────────────┴────────────────────────────────────────────────┘

Sample Data:
┌────┬────────────────────────┬─────────────────────────────┬─────────────────────────────┬───────────────────────────┬─────────────┐
│ id │ user_id                │ started_at                  │ finished_at                 │ geom                      │ is_activity │
├────┼────────────────────────┼─────────────────────────────┼─────────────────────────────┼───────────────────────────┼─────────────┤
│ 0  │ 9358664f-ad4b-46ff...  │ 2021-10-24 08:07:56+07:00   │ 2021-10-24 14:30:00+07:00   │ POINT(110.4315 -7.7478)   │ True        │
│ 1  │ 9358664f-ad4b-46ff...  │ 2021-10-24 14:45:00+07:00   │ 2021-10-24 15:00:00+07:00   │ POINT(110.4280 -7.7500)   │ False       │
│ 2  │ 9358664f-ad4b-46ff...  │ 2021-10-24 15:30:00+07:00   │ 2021-10-24 18:00:00+07:00   │ POINT(110.3857 -7.7117)   │ True        │
└────┴────────────────────────┴─────────────────────────────┴─────────────────────────────┴───────────────────────────┴─────────────┘
```

### File 2: Valid Users (`10_filter_after_user_quality_DIY_slide_filteres.csv`)

```
VALID USERS OUTPUT FILE
═══════════════════════════════════════════════════════════════════════════════

File: 10_filter_after_user_quality_DIY_slide_filteres.csv
Format: CSV

Columns:
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ user_id        │ string       │ UUID of user passing quality filter            │
│ quality        │ float64      │ Mean tracking quality (0.0-1.0)                │
└────────────────┴──────────────┴────────────────────────────────────────────────┘

Sample Data:
┌──────────────────────────────────────────┬──────────────┐
│ user_id                                  │ quality      │
├──────────────────────────────────────────┼──────────────┤
│ 0a1b2c3d-4e5f-6789-abcd-ef0123456789    │ 0.823        │
│ 1b2c3d4e-5f6a-7890-bcde-f01234567890    │ 0.756        │
│ 2c3d4e5f-6a7b-8901-cdef-012345678901    │ 0.891        │
│ ...                                      │ ...          │
└──────────────────────────────────────────┴──────────────┘

Statistics:
• Input users: ~50,000 (all users with GPS data)
• Output users: ~150-300 (users passing all quality filters)
• Reduction: ~99% of users filtered out

This aggressive filtering ensures:
✓ Only high-quality tracking data
✓ Reliable patterns for prediction
✓ Consistent coverage across tracking period
```

---

## Quality Filtering Deep Dive

### Why Quality Filtering is Critical

```
QUALITY FILTERING IMPORTANCE
═══════════════════════════════════════════════════════════════════════════════

Problem: Raw GPS data has varying quality

         User A (Good)              User B (Bad - Gaps)          User C (Bad - Short)
         ════════════               ════════════════════          ═══════════════════
         
Week 1:  ████████████              ████████████                  ████████████
Week 2:  ████████████              ░░░░░░░░░░░░ (no data)        ████████████
Week 3:  ████████████              ████████████                  ████████████
Week 4:  ████████████              ░░░░░░░░░░░░ (no data)        (tracking ends)
Week 5:  ████████████              ████████████                  
Week 6:  ████████████              ████████████                  
Week 7:  ████████████              ░░░░░░░░░░░░ (no data)        
Week 8:  ████████████              ████████████                  
Week 9:  ████████████              ████████████                  
Week 10: ████████████              ████████████                  

Quality: HIGH (consistent)        LOW (gaps = min_thres fail)   LOW (short = day fail)
Result:  ACCEPTED ✓               REJECTED ✗                    REJECTED ✗


Impact on Predictions:

Good Quality User:
• Weekly patterns are clear
• Home/work locations stable
• Model can learn routines

Bad Quality User (Gaps):
• Missing patterns during gaps
• Incomplete weekly cycles
• Model gets confused

Bad Quality User (Short):
• Not enough history
• Can't validate patterns
• Insufficient test data
```

### Quality Metrics Calculation

```python
# Complete quality calculation example

# User data spans 100 days
user_start = datetime(2021, 10, 24)
user_end = datetime(2022, 2, 1)  # 100 days later

# Calculate tracking quality for one 10-week window
window_start = user_start
window_end = user_start + timedelta(weeks=10)  # 70 days

# Total possible time in window
total_seconds = 70 * 24 * 60 * 60  # 6,048,000 seconds

# Sum of tracked durations (staypoints + triplegs)
tracked_durations = [
    # (staypoint/tripleg start, end)
    (datetime(2021, 10, 24, 8, 0), datetime(2021, 10, 24, 18, 0)),  # 10 hours
    (datetime(2021, 10, 24, 18, 30), datetime(2021, 10, 25, 8, 0)), # 13.5 hours
    # ... many more entries
]

tracked_seconds = sum(
    (end - start).total_seconds() 
    for start, end in tracked_durations
)

# Quality for this window
quality = tracked_seconds / total_seconds
# Example: 4,536,000 / 6,048,000 = 0.75 (75%)
```

---

## Summary

The `02_psl_detection_all.ipynb` notebook:

1. **Loads** 165M+ raw GPS points
2. **Detects** staypoints using sliding window algorithm
3. **Generates** triplegs and trips for context
4. **Filters** users based on quality criteria
5. **Outputs** two files for downstream processing

The quality filtering ensures only reliable users are included:
- Day filter: > 60 days tracking
- Min quality: > 0.6 in all windows
- Mean quality: > 0.7 overall

These two output files (`3_staypoints_fun_generate_trips.csv` and `10_filter_after_user_quality_DIY_slide_filteres.csv`) are the foundation for all subsequent preprocessing steps.
