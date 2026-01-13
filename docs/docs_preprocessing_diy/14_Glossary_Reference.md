# Glossary and Reference Guide

## 📋 Table of Contents
1. [Terminology Glossary](#terminology-glossary)
2. [File Reference](#file-reference)
3. [Configuration Reference](#configuration-reference)
4. [Data Schema Reference](#data-schema-reference)
5. [Formula Reference](#formula-reference)
6. [Troubleshooting Guide](#troubleshooting-guide)

---

## Terminology Glossary

### Core Concepts

| Term | Definition | Example |
|------|------------|---------|
| **GPS Point** | Raw position record with latitude, longitude, and timestamp | `(-7.762, 110.378, 2023-01-15 07:30:00)` |
| **Staypoint** | A location where user stayed for significant time (>30 min) | User stayed at home from 7:00 to 18:00 |
| **Location** | A clustered group of staypoints representing a place | "Home" cluster containing all home visits |
| **PSL Pipeline** | Positions → Staypoints → Locations processing chain | Converting GPS tracks to meaningful places |
| **Sequence** | Historical location visits used to predict next location | `[Home, Work, Restaurant] → predict: ?` |

### Location Clustering

| Term | Definition | Example |
|------|------------|---------|
| **DBSCAN** | Density-Based Spatial Clustering algorithm | Groups points within 50m radius |
| **H3** | Uber's Hexagonal hierarchical spatial index | Resolution 8 = ~461m hexagon edge |
| **Epsilon (ε)** | DBSCAN radius parameter in meters | `epsilon=50` → 50 meter radius |
| **Resolution** | H3 grid level (0-15, higher = smaller cells) | `resolution=8` → ~0.74 km² cells |
| **Location ID** | Integer identifier for a location | Location 42 = User's home |

### Temporal Features

| Term | Definition | Example |
|------|------------|---------|
| **start_day** | Days since user's first record | Day 0 = first day, Day 30 = one month later |
| **start_min** | Minute of day (0-1439) | 7:30 AM = 450 (7×60 + 30) |
| **weekday** | Day of week (0=Monday, 6=Sunday) | Friday = 4 |
| **duration** | Time spent at location in minutes | 540 = 9 hours |
| **diff** | Days ago relative to target | diff=3 means "3 days before target" |

### Data Splitting

| Term | Definition | Example |
|------|------------|---------|
| **Temporal Split** | Dividing data by time, not randomly | Days 0-80 train, 81-90 val, 91-100 test |
| **Per-User Split** | Each user split based on their timeline | User A: split at Day 72, User B: split at Day 80 |
| **Data Leakage** | Using future information to predict past | Training on Day 50 data to predict Day 30 |
| **Train/Val/Test** | Training, Validation, Testing sets | 80%/10%/10% split |

### Quality Filtering

| Term | Definition | Example |
|------|------------|---------|
| **day_filter** | Minimum active days required | User must have ≥60 active days |
| **Sliding Window** | Moving time window for quality assessment | 10-week window, sliding by 1 week |
| **Activity Rate** | Proportion of days with staypoints | 70% = 49/70 active days in window |
| **min_thres** | Minimum activity rate in worst window | 0.6 = 60% minimum |
| **mean_thres** | Minimum average activity rate | 0.7 = 70% average |

---

## File Reference

### Input Files

| File | Description | Source |
|------|-------------|--------|
| `3_staypoints_fun_generate_trips.csv` | Staypoints with temporal features | PSL notebook output |
| `10_filter_after_user_quality_DIY_slide_filteres.csv` | Valid user IDs | Quality filtering |

### Intermediate Files

| File | Description | Location |
|------|-------------|----------|
| `intermediate_eps{N}.csv` | DBSCAN interim data | `data/diy_eps{N}/interim/` |
| `intermediate_h3r{N}.csv` | H3 interim data | `data/diy_h3r{N}/interim/` |
| `locations_eps{N}.csv` | DBSCAN location centers | `data/diy_eps{N}/interim/` |
| `locations_h3r{N}.csv` | H3 cell information | `data/diy_h3r{N}/interim/` |

### Output Files

| File | Description | Location |
|------|-------------|----------|
| `*_train.pk` | Training sequences | `data/diy_*/processed/` |
| `*_validation.pk` | Validation sequences | `data/diy_*/processed/` |
| `*_test.pk` | Test sequences | `data/diy_*/processed/` |
| `*_metadata.json` | Dataset statistics | `data/diy_*/processed/` |

### Configuration Files

| File | Description |
|------|-------------|
| `config/preprocessing/diy.yaml` | DBSCAN configuration |
| `config/preprocessing/diy_h3.yaml` | H3 configuration |

### Script Files

| Script | Purpose |
|--------|---------|
| `preprocessing/02_psl_detection_all.ipynb` | PSL detection and quality filtering |
| `preprocessing/diy_1_raw_to_interim.py` | DBSCAN raw to interim |
| `preprocessing/diy_2_interim_to_processed.py` | DBSCAN interim to sequences |
| `preprocessing/diy_h3_1_raw_to_interim.py` | H3 raw to interim |
| `preprocessing/diy_h3_2_interim_to_processed.py` | H3 interim to sequences |

---

## Configuration Reference

### DBSCAN Configuration (diy.yaml)

```yaml
dataset:
  name: "diy"                    # Dataset identifier
  epsilon: 50                    # DBSCAN radius in meters
  num_samples: 2                 # Min points per cluster
  
preprocessing:
  previous_day: [7]              # History window(s) in days
  min_sequence_length: 3         # Min historical staypoints
  max_time_gap: 1                # Staypoint merge gap (minutes)
  max_duration: 2880             # Duration cap (48 hours)

paths:
  raw_folder: "data/raw_diy"
  interim_folder: "data/diy_eps{epsilon}/interim"
  processed_folder: "data/diy_eps{epsilon}/processed"
```

### H3 Configuration (diy_h3.yaml)

```yaml
dataset:
  name: "diy"
  h3_resolution: 8               # H3 grid resolution
  
preprocessing:
  previous_day: [7]
  min_sequence_length: 3
  max_time_gap: 1
  max_duration: 2880

paths:
  raw_folder: "data/raw_diy"
  interim_folder: "data/diy_h3r{h3_resolution}/interim"
  processed_folder: "data/diy_h3r{h3_resolution}/processed"
```

### Parameter Ranges

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| epsilon | 50 | 30-200 | Smaller = more locations |
| h3_resolution | 8 | 5-11 | Higher = smaller cells |
| previous_day | 7 | 3-30 | Larger = more history |
| min_sequence_length | 3 | 1-10 | Higher = fewer sequences |
| max_duration | 2880 | 1440-4320 | Cap for outliers |

---

## Data Schema Reference

### Raw Staypoints Schema

```
┌───────────────────┬──────────────┬───────────────────────────────────────────┐
│ Column            │ Type         │ Description                               │
├───────────────────┼──────────────┼───────────────────────────────────────────┤
│ id                │ int64        │ Unique staypoint identifier               │
│ user_id           │ object       │ User identifier (string)                  │
│ started_at        │ datetime64   │ Start timestamp                           │
│ finished_at       │ datetime64   │ End timestamp                             │
│ center_lat        │ float64      │ Latitude of staypoint center              │
│ center_lon        │ float64      │ Longitude of staypoint center             │
│ is_activity       │ bool         │ True if duration > 25 minutes             │
└───────────────────┴──────────────┴───────────────────────────────────────────┘
```

### Intermediate Schema

```
┌────────────────┬──────────────┬────────────────────────────────────────────────┐
│ Column         │ Type         │ Description                                    │
├────────────────┼──────────────┼────────────────────────────────────────────────┤
│ id             │ int64        │ Sequential staypoint ID                        │
│ user_id        │ int64        │ Encoded user ID (integer)                      │
│ location_id    │ int64        │ Clustered location ID                          │
│ start_day      │ int64        │ Days since user's first record                 │
│ end_day        │ int64        │ End day number                                 │
│ start_min      │ int64        │ Start minute of day (0-1439)                   │
│ end_min        │ int64        │ End minute of day (1-1440)                     │
│ weekday        │ int64        │ Day of week (0-6)                              │
│ duration       │ float64      │ Duration in minutes                            │
└────────────────┴──────────────┴────────────────────────────────────────────────┘
```

### Sequence Schema

```
┌─────────────┬─────────────┬─────────────────────────────────────────────────────┐
│ Key         │ Type        │ Description                                         │
├─────────────┼─────────────┼─────────────────────────────────────────────────────┤
│ X           │ List[int]   │ Historical location IDs (encoded)                   │
│ user_X      │ List[int]   │ User ID repeated for sequence length                │
│ weekday_X   │ List[int]   │ Day of week for each historical visit               │
│ start_min_X │ List[int]   │ Start minute for each historical visit              │
│ dur_X       │ List[float] │ Duration for each historical visit                  │
│ diff        │ List[int]   │ Days ago for each historical visit                  │
│ Y           │ int         │ Target location ID to predict                       │
└─────────────┴─────────────┴─────────────────────────────────────────────────────┘
```

### Location ID Encoding

```
ID │ Meaning
───┼─────────────────────────
 0 │ Padding (for batching)
 1 │ Unknown location
2+ │ Actual locations
```

---

## Formula Reference

### Time Calculations

```
start_day = (staypoint_date - user_first_date).days

start_min = hour × 60 + minute
           Example: 7:30 AM = 7 × 60 + 30 = 450

weekday = started_at.dayofweek
          (0 = Monday, 6 = Sunday)

duration = (finished_at - started_at).total_seconds() / 60
           Capped at max_duration
```

### Sequence Generation

```
History window: [target_day - previous_day, target_day)

diff = target_day - historical_start_day
       Example: Target Day 7, History Day 3 → diff = 4
```

### Quality Filtering

```
activity_rate = active_days_in_window / total_days_in_window

User passes if:
    mean(all_window_rates) >= mean_thres  AND
    min(all_window_rates) >= min_thres
```

### Temporal Splitting

```
train_cutoff = int(max_day × 0.8)
val_cutoff = int(max_day × 0.9)

Train: start_day <= train_cutoff
Val:   train_cutoff < start_day <= val_cutoff
Test:  start_day > val_cutoff
```

---

## Troubleshooting Guide

### Common Errors

#### Error: "No valid users after filtering"

```
Cause: Quality filtering is too strict
Solution:
    1. Lower day_filter (try 30 instead of 60)
    2. Lower min_thres (try 0.4 instead of 0.6)
    3. Lower mean_thres (try 0.5 instead of 0.7)
```

#### Error: "KeyError: 'location_id'"

```
Cause: Location clustering failed or file not found
Solution:
    1. Verify Script 1 completed successfully
    2. Check intermediate file exists
    3. Verify column names in input file
```

#### Error: "Empty sequences generated"

```
Cause: min_sequence_length too high or previous_day too small
Solution:
    1. Lower min_sequence_length (try 1 or 2)
    2. Increase previous_day (try 14 instead of 7)
    3. Check that users have enough historical data
```

#### Error: "Memory error during sequence generation"

```
Cause: Dataset too large for available memory
Solution:
    1. Process users in batches
    2. Reduce n_jobs in parallel processing
    3. Use a machine with more RAM
```

### Verification Checks

```python
# Check temporal ordering
for user_id in df['user_id'].unique():
    user_data = df[df['user_id'] == user_id]
    train_max = user_data[user_data['split'] == 'train']['start_day'].max()
    test_min = user_data[user_data['split'] == 'test']['start_day'].min()
    assert train_max < test_min, f"User {user_id} has data leakage!"

# Check location ID encoding
assert 0 not in sequences['X'].values, "Location 0 (padding) in data!"
assert sequences['Y'].min() >= 2, "Target locations should be >= 2"

# Check sequence lengths
assert all(len(s['X']) >= min_sequence_length for s in sequences)
```

### Performance Tips

```
1. Use parallel processing (n_jobs=-1)
2. Process by user, not by staypoint
3. Pre-filter before expensive operations
4. Save intermediate results to avoid recomputation
5. Use pickle for large datasets (faster than CSV)
```

---

## Quick Reference Card

### Pipeline Commands

```bash
# Run complete DBSCAN pipeline
python preprocessing/diy_1_raw_to_interim.py --config config/preprocessing/diy.yaml
python preprocessing/diy_2_interim_to_processed.py --config config/preprocessing/diy.yaml

# Run complete H3 pipeline
python preprocessing/diy_h3_1_raw_to_interim.py --config config/preprocessing/diy_h3.yaml
python preprocessing/diy_h3_2_interim_to_processed.py --config config/preprocessing/diy_h3.yaml
```

### Loading Data

```python
import pickle
import json

# Load sequences
with open("diy_eps50_prev7_train.pk", "rb") as f:
    train_data = pickle.load(f)

# Load metadata
with open("diy_eps50_prev7_metadata.json", "r") as f:
    metadata = json.load(f)

# Access
num_sequences = len(train_data)
num_locations = metadata["total_loc_num"]
num_users = metadata["total_user_num"]
```

### Key Statistics

```
Typical DIY Dataset Statistics:
─────────────────────────────────
Raw users:        ~50,000
After filtering:  ~150-300
Staypoints:       ~200,000-400,000
Locations:        ~4,000-6,000
Train sequences:  ~60,000-80,000
Val sequences:    ~8,000-10,000
Test sequences:   ~8,000-10,000
```
