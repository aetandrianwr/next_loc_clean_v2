# Data Pipeline: Input/Output Formats and Data Flow

## 1. Overview

This document describes the complete data pipeline from raw GPS trajectories to the final analysis results. Understanding this pipeline is essential for:
- Reproducing the analysis
- Adapting to new datasets
- Debugging data issues

---

## 2. Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DATA PIPELINE ARCHITECTURE                             │
└─────────────────────────────────────────────────────────────────────────────┘

┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│   RAW GPS        │     │  INTERMEDIATE    │     │   ANALYSIS       │
│   TRAJECTORIES   │────►│  CSV DATA        │────►│   RESULTS        │
│                  │     │  (Preprocessed)  │     │                  │
└──────────────────┘     └──────────────────┘     └──────────────────┘
        │                         │                        │
        ▼                         ▼                        ▼
   GPS points              Cleaned events           Return times,
   (lat, lon, time)        (user, loc, day, min)   PDF, Plots

   PREPROCESSING            ANALYSIS                 OUTPUT
   (not covered here)       (this analysis)         (results)
```

---

## 3. Input Data Format

### 3.1 Intermediate CSV Structure

The analysis reads from **intermediate CSV files** that have already been preprocessed from raw GPS data.

**File Locations**:
```
data/geolife_eps20/interim/intermediate_eps20.csv
data/diy_eps50/interim/intermediate_eps50.csv
```

**Schema**:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` | int/str | Unique user identifier | 1, 2, 3, ... |
| `location_id` | int | Encoded location (staypoint cluster) | 100, 200, 300, ... |
| `start_day` | int | Day number since tracking started | 0, 1, 2, ... |
| `start_min` | int | Minute of day (0-1439) | 480 (08:00), 720 (12:00) |

**Example Data**:
```csv
user_id,location_id,start_day,start_min
1,100,0,480
1,200,0,555
1,100,0,1110
1,100,1,465
2,300,0,540
2,300,1,1080
```

### 3.2 Understanding the Columns

#### user_id
- Unique identifier for each person in the dataset
- Can be integer or string (converted internally)
- Used to group trajectories

```
User 1: Alice's trajectory
User 2: Bob's trajectory
```

#### location_id
- Encoded location from DBSCAN clustering of GPS coordinates
- Each ID represents a **staypoint cluster** (place where user spent time)
- Location 0 is typically reserved for padding

```
location_id = 100  →  Cluster representing HOME
location_id = 200  →  Cluster representing WORK
location_id = 300  →  Cluster representing CAFE
```

#### start_day
- Integer day count from start of tracking
- Day 0 = first day of data collection
- Enables handling multi-year datasets

```
start_day = 0   →  Day 1 of tracking (e.g., Jan 1)
start_day = 1   →  Day 2 of tracking (e.g., Jan 2)
start_day = 365 →  One year later
```

#### start_min
- Minute of the day when staypoint started
- Range: 0 to 1439 (0:00 to 23:59)
- 1440 minutes in a day

```
start_min = 0    →  00:00 (midnight)
start_min = 480  →  08:00 (8 AM)
start_min = 720  →  12:00 (noon)
start_min = 1110 →  18:30 (6:30 PM)
```

---

## 4. Timestamp Computation

### 4.1 From Day+Minute to Absolute Hours

The analysis converts the day+minute format to absolute hours:

```
timestamp_minutes = start_day × 1440 + start_min
timestamp_hours = timestamp_minutes / 60.0
```

### 4.2 Worked Example

**Input Record**:
```
user_id=1, location_id=100, start_day=2, start_min=720
```

**Computation**:
```
timestamp_minutes = 2 × 1440 + 720
                  = 2880 + 720
                  = 3600 minutes

timestamp_hours = 3600 / 60.0
                = 60.0 hours
```

**Interpretation**: This event occurred 60 hours (2.5 days) after tracking started, at noon on day 2.

### 4.3 Visual Timeline

```
                    timestamp_hours
                         ↓
     ┌───────────────────┼───────────────────────────────────────►
     │                   │                                       time
     │                   │
Day 0│  ├────────────────┤
     │  0h              24h
     │
Day 1│  ├────────────────┤
     │ 24h              48h
     │                   │
Day 2│  ├───────┼────────┤
     │ 48h     60h      72h
               ↑
           noon on day 2
           (start_day=2, start_min=720)
```

---

## 5. Data Processing Pipeline

### 5.1 Stage 1: Load and Convert

```python
# Input: Raw CSV
df = pd.read_csv('intermediate_eps20.csv')

# Transform: Convert to absolute timestamp
df['timestamp_hours'] = (df['start_day'] * 1440 + df['start_min']) / 60.0

# Output: Clean DataFrame
df = df[['user_id', 'location_id', 'timestamp_hours']]
```

**Input**:
```
user_id,location_id,start_day,start_min
1,100,0,480
1,200,0,555
```

**Output**:
```
   user_id  location_id  timestamp_hours
0        1          100             8.00
1        1          200             9.25
```

### 5.2 Stage 2: Sort by User and Time

```python
df_sorted = df.sort_values(['user_id', 'timestamp_hours']).reset_index(drop=True)
```

**Purpose**: Ensure chronological order for each user.

**Before**:
```
user_id  location_id  timestamp_hours
   1         100          18.50      # Out of order!
   1         200           9.25
   1         100           8.00
```

**After**:
```
user_id  location_id  timestamp_hours
   1         100           8.00      # First
   1         200           9.25      # Second
   1         100          18.50      # Third
```

### 5.3 Stage 3: Extract First Events

```python
first_events = df_sorted.groupby('user_id').first().reset_index()
first_events = first_events.rename(columns={
    'location_id': 'first_location',
    'timestamp_hours': 'first_time'
})
```

**Input** (sorted DataFrame):
```
user_id  location_id  timestamp_hours
   1         100           8.00
   1         200           9.25
   1         100          18.50
   2         300           5.00
   2         300          29.00
```

**Output** (first events per user):
```
user_id  first_location  first_time
   1          100            8.00
   2          300            5.00
```

### 5.4 Stage 4: Find Returns

```python
# Merge first event info
df_with_first = df_sorted.merge(first_events, on='user_id')

# Filter to later events
df_later = df_with_first[df_with_first['timestamp_hours'] > df_with_first['first_time']]

# Filter to returns
df_returns = df_later[df_later['location_id'] == df_later['first_location']]

# Get first return per user
first_returns = df_returns.groupby('user_id').first().reset_index()

# Compute delta_t
first_returns['delta_t_hours'] = first_returns['timestamp_hours'] - first_returns['first_time']
```

**Step-by-step for User 1**:

```
Initial trajectory:
  time     location
   8.00      100  ← First location (L₀ = 100, t₀ = 8.00)
   9.25      200
  18.50      100  ← Return to L₀!

After merge (add first_location, first_time):
  time     location  first_location  first_time
   8.00      100          100           8.00
   9.25      200          100           8.00
  18.50      100          100           8.00

After filter (timestamp > first_time):
  time     location  first_location  first_time
   9.25      200          100           8.00  ← Later event
  18.50      100          100           8.00  ← Later event

After filter (location == first_location):
  time     location  first_location  first_time
  18.50      100          100           8.00  ← Return!

delta_t = 18.50 - 8.00 = 10.50 hours
```

---

## 6. Output Data Formats

### 6.1 Return Times CSV

**File**: `{dataset}_return_probability_data_returns.csv`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `user_id` | int/str | User identifier |
| `delta_t_hours` | float | First return time in hours |

**Example**:
```csv
user_id,delta_t_hours
1,10.50
2,24.00
3,47.75
4,23.25
```

### 6.2 Probability Density CSV

**File**: `{dataset}_return_probability_data.csv`

**Schema**:
| Column | Type | Description |
|--------|------|-------------|
| `t_hours` | float | Bin center (time in hours) |
| `F_pt` | float | Probability density |

**Example**:
```csv
t_hours,F_pt
1.0,0.012245
3.0,0.008163
5.0,0.006122
...
23.0,0.024490
25.0,0.020408
...
```

### 6.3 Plot Files

**Files**:
- `{dataset}_return_probability.png` - V1 plot
- `{dataset}_return_probability_v2.png` - V2 plot with RW baseline
- `comparison_return_probability.png` - Cross-dataset comparison

**Specifications**:
| Property | Value |
|----------|-------|
| Format | PNG |
| DPI | 300 |
| Dimensions | ~2700 × 2100 pixels |
| Color space | RGB |

---

## 7. Data Validation

### 7.1 Input Validation Checks

The analysis performs these validations:

```python
# Check required columns exist
required_columns = ['user_id', 'location_id', 'start_day', 'start_min']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Missing required column: {col}")

# Check for valid ranges
assert df['start_min'].between(0, 1439).all(), "Invalid start_min values"
assert df['start_day'].ge(0).all(), "Negative start_day values"
```

### 7.2 Output Validation

```python
# Probability density should sum to ~1.0
total_prob = (pdf * bin_width).sum()
assert abs(total_prob - 1.0) < 0.01, f"Probability mass = {total_prob}, expected ~1.0"

# All probabilities should be non-negative
assert (pdf >= 0).all(), "Negative probability densities found"
```

---

## 8. Dataset Statistics

### 8.1 Geolife Dataset

```
┌─────────────────────────────────────────────────────────────────────┐
│ GEOLIFE DATASET STATISTICS                                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Raw Data:                                                           │
│   • Events: 19,191                                                  │
│   • Users: 91                                                       │
│   • Unique locations: 2,049                                         │
│   • Time range: 0.02h to 46,376.43h (~5.3 years)                   │
│   • DBSCAN epsilon: 20 meters                                       │
│                                                                      │
│ Return Analysis:                                                     │
│   • Users with returns: 49 (53.85%)                                 │
│   • Mean return time: 58.96 hours                                   │
│   • Median return time: 35.28 hours                                 │
│   • Std dev: 65.62 hours                                            │
│   • Range: [1.25, 239.98] hours                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 8.2 DIY Dataset

```
┌─────────────────────────────────────────────────────────────────────┐
│ DIY DATASET STATISTICS                                              │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Raw Data:                                                           │
│   • Events: 265,621                                                 │
│   • Users: 1,306                                                    │
│   • Unique locations: 8,439                                         │
│   • Time range: 0.02h to 5,464.22h (~7.6 months)                   │
│   • DBSCAN epsilon: 50 meters                                       │
│                                                                      │
│ Return Analysis:                                                     │
│   • Users with returns: 1,091 (83.54%)                              │
│   • Mean return time: 60.02 hours                                   │
│   • Median return time: 42.77 hours                                 │
│   • Std dev: 54.48 hours                                            │
│   • Range: [0.65, 238.63] hours                                     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 9. File System Structure

### 9.1 Complete Directory Structure

```
next_loc_clean_v2/
├── data/
│   ├── geolife_eps20/
│   │   └── interim/
│   │       └── intermediate_eps20.csv    ← INPUT
│   └── diy_eps50/
│       └── interim/
│           └── intermediate_eps50.csv    ← INPUT
│
├── scripts/
│   └── analysis_returner/
│       ├── return_probability_analysis.py      ← SCRIPT V1
│       ├── return_probability_analysis_v2.py   ← SCRIPT V2
│       ├── compare_datasets.py                 ← COMPARISON SCRIPT
│       ├── run_analysis.sh                     ← RUNNER
│       │
│       ├── geolife_return_probability.png      ← OUTPUT (V1)
│       ├── geolife_return_probability_v2.png   ← OUTPUT (V2)
│       ├── geolife_return_probability_data.csv
│       ├── geolife_return_probability_data_returns.csv
│       │
│       ├── diy_return_probability.png
│       ├── diy_return_probability_v2.png
│       ├── diy_return_probability_data.csv
│       ├── diy_return_probability_data_returns.csv
│       │
│       ├── comparison_return_probability.png
│       │
│       └── docs/                               ← DOCUMENTATION
│           ├── 00_INDEX.md
│           ├── 01_OVERVIEW.md
│           ├── ...
│           └── 10_APPENDIX.md
│
└── src/
    └── models/
        └── proposed/
            └── pointer_v45.py                  ← MODEL
```

### 9.2 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          DATA FLOW DIAGRAM                                   │
└─────────────────────────────────────────────────────────────────────────────┘

data/geolife_eps20/interim/                scripts/analysis_returner/
┌───────────────────────┐                  ┌───────────────────────┐
│ intermediate_eps20.csv │───────────────► │ return_probability_   │
│                       │       READ       │ analysis_v2.py        │
│ • user_id             │                  │                       │
│ • location_id         │                  │ • Load data           │
│ • start_day           │                  │ • Compute returns     │
│ • start_min           │                  │ • Build histogram     │
└───────────────────────┘                  │ • Generate plots      │
                                           └───────────┬───────────┘
                                                       │ WRITE
                                                       ▼
                                           ┌───────────────────────┐
                                           │ geolife_return_       │
                                           │ probability_v2.png    │
                                           │                       │
                                           │ geolife_return_       │
                                           │ probability_data.csv  │
                                           │                       │
                                           │ geolife_return_prob_  │
                                           │ data_returns.csv      │
                                           └───────────────────────┘
```

---

## 10. Adapting to New Datasets

### 10.1 Requirements for New Data

To use this analysis with a new dataset, your CSV must have:

| Column | Required | Format |
|--------|----------|--------|
| `user_id` | Yes | Integer or string |
| `location_id` | Yes | Integer (from clustering) |
| `start_day` | Yes | Integer (day count from start) |
| `start_min` | Yes | Integer (0-1439) |

### 10.2 Conversion Example

If your data has timestamps in datetime format:

```python
import pandas as pd

# Load your data
df = pd.read_csv('my_data.csv')

# Convert datetime to day + minute
df['datetime'] = pd.to_datetime(df['timestamp'])
start_date = df['datetime'].min().date()
df['start_day'] = (df['datetime'].dt.date - start_date).dt.days
df['start_min'] = df['datetime'].dt.hour * 60 + df['datetime'].dt.minute

# Save in required format
df[['user_id', 'location_id', 'start_day', 'start_min']].to_csv(
    'my_intermediate_data.csv', 
    index=False
)
```

### 10.3 Running Analysis on New Data

```bash
# Modify path in script or pass via argument
python return_probability_analysis_v2.py \
    --output-dir my_results/

# Or edit the script to add your dataset path
```

---

*← Back to [Code Walkthrough](03_CODE_WALKTHROUGH.md) | Continue to [Algorithm Details](05_ALGORITHM_DETAILS.md) →*
