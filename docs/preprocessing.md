# Preprocessing Documentation

This document describes how to preprocess the Geolife and DIY datasets for next location prediction research.

## Table of Contents

1. [Overview](#overview)
2. [Folder Structure](#folder-structure)
3. [Configuration Files](#configuration-files)
4. [Running the Preprocessing Scripts](#running-the-preprocessing-scripts)
5. [Script Details](#script-details)
6. [Raw Input Format](#raw-input-format)
7. [Final Output Format](#final-output-format)
8. [Troubleshooting](#troubleshooting)

---

## Overview

The preprocessing pipeline is organized into two scripts per dataset:

1. **Script 1: Raw to Interim** - Processes raw GPS/staypoint data into intermediate results with locations, temporal features, and user filtering. This script is parameterized by `epsilon` (DBSCAN clustering parameter).

2. **Script 2: Interim to Processed** - Converts intermediate data into final `.pk` files for model training. This script is parameterized by `previous_day` (number of days of history to consider).

This separation allows you to:
- Change `epsilon` → Re-run both scripts
- Change `previous_day` → Only re-run Script 2 (faster iteration)

---

## Folder Structure

```
next_loc_clean_v2/
├── config/                          # Configuration files
│   ├── geolife.yaml
│   └── diy.yaml
├── data/
│   ├── raw_geolife/                 # Raw Geolife trajectory data (182 user folders)
│   ├── raw_diy/                     # Raw DIY staypoint data (CSV files)
│   ├── geolife_eps{epsilon}/        # Output for Geolife with specific epsilon
│   │   ├── interim/                 # Intermediate files from Script 1
│   │   │   ├── intermediate_eps{epsilon}.csv
│   │   │   ├── locations_eps{epsilon}.csv
│   │   │   ├── valid_users_eps{epsilon}.csv
│   │   │   ├── staypoints_all_eps{epsilon}.csv
│   │   │   ├── staypoints_merged_eps{epsilon}.csv
│   │   │   ├── raw_stats_eps{epsilon}.json
│   │   │   ├── interim_stats_eps{epsilon}.json
│   │   │   └── quality/
│   │   │       └── user_quality_eps{epsilon}.csv
│   │   └── processed/               # Final .pk files from Script 2
│   │       ├── geolife_eps{epsilon}_prev{previous_day}_train.pk
│   │       ├── geolife_eps{epsilon}_prev{previous_day}_validation.pk
│   │       ├── geolife_eps{epsilon}_prev{previous_day}_test.pk
│   │       └── geolife_eps{epsilon}_prev{previous_day}_metadata.json
│   └── diy_eps{epsilon}/            # Output for DIY with specific epsilon
│       ├── interim/
│       └── processed/
├── preprocessing/                   # Preprocessing scripts
│   ├── geolife_1_raw_to_interim.py
│   ├── geolife_2_interim_to_processed.py
│   ├── diy_1_raw_to_interim.py
│   └── diy_2_interim_to_processed.py
└── docs/
    └── preprocessing.md             # This file
```

---

## Configuration Files

### Geolife Configuration (`config/preprocessing/geolife.yaml`)

```yaml
dataset:
  name: "geolife"
  epsilon: 20              # DBSCAN epsilon for location clustering (meters)
  previous_day: [7]        # List of previous days for sequence generation

preprocessing:
  staypoint:
    gap_threshold: 1440    # 24 hours in minutes
    dist_threshold: 200    # Distance threshold in meters
    time_threshold: 30     # Time threshold in minutes
    activity_time_threshold: 25
  
  location:
    num_samples: 2         # Minimum samples for DBSCAN
    distance_metric: "haversine"
    agg_level: "dataset"
  
  quality_filter:
    day_filter: 50         # Minimum tracking days
    window_size: 10        # Sliding window size
  
  max_duration: 2880       # Maximum duration in minutes (2 days)
  # min_sequence_length: 3 # Hardcoded to 3
  
  split:
    train: 0.6
    val: 0.2
    test: 0.2

random_seed: 42
```

### DIY Configuration (`config/preprocessing/diy.yaml`)

```yaml
dataset:
  name: "diy"
  epsilon: 50              # DBSCAN epsilon (meters)
  previous_day: [7]        # List of previous days

preprocessing:
  location:
    num_samples: 2
    distance_metric: "haversine"
    agg_level: "dataset"
  
  staypoint_merging:
    max_time_gap: "1min"
  
  quality_filter:
    day_filter: 60
    window_size: 10
    min_thres: 0.6
    mean_thres: 0.7
  
  max_duration: 2880       # Maximum duration in minutes
  min_sequence_length: 3
  
  split:
    train: 0.8
    val: 0.1
    test: 0.1

random_seed: 42
```

### Key Parameters

| Parameter | Description | Geolife Default | DIY Default |
|-----------|-------------|-----------------|-------------|
| `epsilon` | DBSCAN clustering radius (meters) | 20 | 50 |
| `previous_day` | List of history window sizes (days) | [7] | [7] |
| `split.train/val/test` | Data split ratios | 0.6/0.2/0.2 | 0.8/0.1/0.1 |
| `max_duration` | Max staypoint duration (minutes) | 2880 | 2879 |
| `min_sequence_length` | Minimum history length | 3 | 3 |

---

## Running the Preprocessing Scripts

### Prerequisites

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
```

### Geolife Dataset

```bash
cd /data/next_loc_clean_v2

# Script 1: Raw to Interim (run when epsilon changes)
python preprocessing/geolife_1_raw_to_interim.py --config config/preprocessing/geolife.yaml

# Script 2: Interim to Processed (run when previous_day changes)
python preprocessing/geolife_2_interim_to_processed.py --config config/preprocessing/geolife.yaml
```

### DIY Dataset

```bash
cd /data/next_loc_clean_v2

# Script 1: Raw to Interim
python preprocessing/diy_1_raw_to_interim.py --config config/preprocessing/diy.yaml

# Script 2: Interim to Processed
python preprocessing/diy_2_interim_to_processed.py --config config/preprocessing/diy.yaml
```

### Multiple previous_day Values

To generate outputs for multiple `previous_day` values, edit the config:

```yaml
previous_day: [7, 14]  # Will generate files for both 7 and 14 days
```

Then only run Script 2:

```bash
python preprocessing/geolife_2_interim_to_processed.py --config config/preprocessing/geolife.yaml
```

This will create:
- `geolife_eps20_prev7_train.pk`, `geolife_eps20_prev7_validation.pk`, etc.
- `geolife_eps20_prev14_train.pk`, `geolife_eps20_prev14_validation.pk`, etc.

---

## Script Details

### Script 1: Raw to Interim

**Geolife (`geolife_1_raw_to_interim.py`)**

| Step | Description |
|------|-------------|
| 1 | Read raw GPS trajectories (position fixes) |
| 2 | Generate staypoints from position fixes |
| 3 | Create activity flags based on time threshold |
| 4 | Filter valid users based on tracking quality |
| 5 | Filter to activity staypoints only |
| 6 | Generate locations using DBSCAN clustering |
| 7 | Merge consecutive staypoints and add temporal features |

**DIY (`diy_1_raw_to_interim.py`)**

| Step | Description |
|------|-------------|
| 1 | Read preprocessed staypoints and valid users |
| 2 | Filter to activity staypoints |
| 3 | Generate locations using DBSCAN clustering |
| 4 | Merge consecutive staypoints |
| 5 | Add temporal features |

**Output Files (in `interim/`):**

| File | Description |
|------|-------------|
| `intermediate_eps{epsilon}.csv` | Main intermediate dataset with temporal features |
| `locations_eps{epsilon}.csv` | Location centers (lat, lon, geometry) |
| `valid_users_eps{epsilon}.csv` | List of valid user IDs |
| `staypoints_all_eps{epsilon}.csv` | All staypoints before filtering |
| `staypoints_merged_eps{epsilon}.csv` | Staypoints after merging |
| `interim_stats_eps{epsilon}.json` | Statistics for EDA |
| `quality/user_quality_eps{epsilon}.csv` | User tracking quality scores |

### Script 2: Interim to Processed

**Steps (same for both datasets):**

| Step | Description |
|------|-------------|
| 1 | Split dataset into train/val/test per user (chronological) |
| 2 | Encode location IDs (0=padding, 1=unknown, 2+=known) |
| 3 | Filter valid sequences (minimum `previous_day` history, min 3 staypoints) |
| 4 | Filter users with valid sequences in all splits |
| 5 | Re-encode user IDs to be continuous |
| 6 | Generate sequence dictionaries |
| 7 | Save .pk files |

**Output Files (in `processed/`):**

| File | Description |
|------|-------------|
| `{name}_eps{eps}_prev{prev}_train.pk` | Training sequences |
| `{name}_eps{eps}_prev{prev}_validation.pk` | Validation sequences |
| `{name}_eps{eps}_prev{prev}_test.pk` | Test sequences |
| `{name}_eps{eps}_prev{prev}_metadata.json` | Dataset metadata |

---

## Raw Input Format

### Geolife Raw Data (`data/raw_geolife/`)

Standard Geolife dataset format:
```
raw_geolife/
├── 000/
│   └── Trajectory/
│       ├── 20081023025304.plt
│       ├── 20081024020959.plt
│       └── ...
├── 001/
│   └── Trajectory/
│       └── ...
└── ...
```

Each `.plt` file contains GPS points with: latitude, longitude, altitude, date, time.

### DIY Raw Data (`data/raw_diy/`)

```
raw_diy/
├── 3_staypoints_fun_generate_trips.csv    # Staypoints with activity flags
└── 10_filter_after_user_quality_DIY_slide_filteres.csv  # Valid users
```

The staypoints CSV contains columns:
- `id`, `user_id`, `started_at`, `finished_at`, `geom`, `is_activity`

---

## Final Output Format

Each `.pk` file contains a list of dictionaries, where each dictionary represents one sequence sample:

```python
{
    'X': np.array([...]),           # Location IDs (history), shape (seq_len,)
    'user_X': np.array([...]),      # User IDs (same user), shape (seq_len,)
    'weekday_X': np.array([...]),   # Weekday (0-6), shape (seq_len,)
    'start_min_X': np.array([...]), # Start minute of day (0-1439), shape (seq_len,)
    'dur_X': np.array([...]),       # Duration in minutes, shape (seq_len,)
    'diff': np.array([...]),        # Days difference to target, shape (seq_len,)
    'Y': int                         # Target location ID (scalar)
}
```

**Location ID Encoding:**
- `0`: Padding (not used in sequences)
- `1`: Unknown location (not in training set)
- `2+`: Known locations from training set

**Metadata (.pk):**

```python
{
    'dataset_name': 'geolife',
    'epsilon': 20,
    'previous_day': 7,
    'total_user_num': 46,          # For embedding layer size
    'total_loc_num': 1187,         # For embedding layer size
    'unique_locations': 1185,
    'train_sequences': 7424,
    'val_sequences': 3334,
    'test_sequences': 3502,
    'split_ratios': {'train': 0.6, 'val': 0.2, 'test': 0.2},
    ...
}
```

---

## Troubleshooting

### Common Issues

1. **"No valid staypoints found after quality filtering"**
   - Check that raw data exists in `data/raw_{dataset}/`
   - Verify quality filter parameters aren't too strict

2. **Long processing time for Script 1 (Geolife)**
   - User quality calculation takes ~10 minutes
   - Once computed, it's cached in `interim/quality/`

3. **Empty processed files**
   - Check that `previous_day` isn't larger than users' tracking duration
   - Verify `min_sequence_length` isn't too high

### Verifying Output

```python
import pickle

# Load and check
with open('data/geolife_eps20/processed/geolife_eps20_prev7_train.pk', 'rb') as f:
    data = pickle.load(f)

print(f"Number of sequences: {len(data)}")
print(f"Sample keys: {data[0].keys()}")
print(f"Sample X shape: {data[0]['X'].shape}")
```

---

## Summary

| Task | Script | Parameters Used | Output Location |
|------|--------|-----------------|-----------------|
| Change epsilon | Script 1 + Script 2 | `epsilon` from config | `data/{name}_eps{eps}/` |
| Change previous_day | Script 2 only | `previous_day` from config | `data/{name}_eps{eps}/processed/` |
| Add new previous_day | Script 2 only | Add to `previous_day` list | New .pk files in processed/ |

Random seed is set to **42** for reproducibility.
