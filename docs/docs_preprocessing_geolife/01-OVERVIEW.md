# GeoLife Dataset Preprocessing Pipeline - Complete Overview

## Table of Contents
1. [Introduction](#introduction)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Two Processing Approaches](#two-processing-approaches)
4. [Pipeline Flow Diagram](#pipeline-flow-diagram)
5. [Key Concepts](#key-concepts)
6. [Output Structure](#output-structure)
7. [Quick Start](#quick-start)

---

## Introduction

This documentation provides a comprehensive guide to the **GeoLife Dataset Preprocessing Pipeline**, which transforms raw GPS trajectory data into machine learning-ready sequences for **Next Location Prediction** tasks. This pipeline is designed for research purposes and can serve as a reference for PhD thesis work in mobility prediction, location-based services, and human mobility analysis.

### Purpose of the Pipeline

The preprocessing pipeline serves to:

1. **Transform Raw GPS Data** → Convert GPS position fixes into meaningful location visits (staypoints)
2. **Cluster Locations** → Group nearby staypoints into semantic locations
3. **Filter Quality Users** → Select users with sufficient tracking quality
4. **Generate Sequences** → Create input sequences for deep learning models
5. **Prepare for Training** → Split data and encode features for model consumption

### What is Next Location Prediction?

Next Location Prediction is a fundamental task in mobility analysis where given a user's historical location visits, we predict their next location. This has applications in:

- **Urban Planning**: Understanding city movement patterns
- **Transportation**: Optimizing routes and services
- **Location-Based Services**: Personalized recommendations
- **Emergency Response**: Predicting crowd movements

---

## Pipeline Architecture

The preprocessing pipeline follows a **two-stage architecture**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         PREPROCESSING PIPELINE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   ┌───────────────┐                                                             │
│   │  RAW DATA     │  GPS trajectories (.plt files)                              │
│   │  GeoLife      │  182 users, 17,621 trajectories                             │
│   └───────┬───────┘                                                             │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐             │
│   │                    SCRIPT 1: Raw to Interim                    │             │
│   │  ┌─────────────────────────────────────────────────────────┐  │             │
│   │  │  1. Read GPS trajectories (position fixes)              │  │             │
│   │  │  2. Generate staypoints (where users stopped)           │  │             │
│   │  │  3. Create activity flags                               │  │             │
│   │  │  4. Filter users by tracking quality                    │  │             │
│   │  │  5. Cluster staypoints → Locations (DBSCAN or H3)       │  │             │
│   │  │  6. Merge consecutive staypoints                        │  │             │
│   │  │  7. Enrich with temporal features                       │  │             │
│   │  └─────────────────────────────────────────────────────────┘  │             │
│   └───────────────────────────────────────────────────────────────┘             │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────┐                                                             │
│   │ INTERIM DATA  │  intermediate_eps{X}.csv                                    │
│   │               │  Cleaned staypoints with temporal features                  │
│   └───────┬───────┘                                                             │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────────────────────────────────────────────────────┐             │
│   │                SCRIPT 2: Interim to Processed                  │             │
│   │  ┌─────────────────────────────────────────────────────────┐  │             │
│   │  │  1. Load intermediate data                              │  │             │
│   │  │  2. Split into train/val/test (per user, chronological) │  │             │
│   │  │  3. Encode location IDs (ordinal encoding)              │  │             │
│   │  │  4. Filter valid sequences (previous_day window)        │  │             │
│   │  │  5. Generate sequence dictionaries                      │  │             │
│   │  │  6. Save .pk files and metadata                         │  │             │
│   │  └─────────────────────────────────────────────────────────┘  │             │
│   └───────────────────────────────────────────────────────────────┘             │
│           │                                                                      │
│           ▼                                                                      │
│   ┌───────────────┐                                                             │
│   │PROCESSED DATA │  train.pk, validation.pk, test.pk, metadata.json            │
│   │               │  Ready for deep learning model training                     │
│   └───────────────┘                                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Two Processing Approaches

The pipeline provides **two different approaches** for location clustering:

### Approach 1: DBSCAN-Based (Standard)

**Scripts**: `geolife_1_raw_to_interim.py` + `geolife_2_interim_to_processed.py`

- Uses **DBSCAN clustering** algorithm to group staypoints into locations
- Parameter: `epsilon` (in meters) - defines the maximum distance between staypoints in a cluster
- **Advantage**: Adapts to natural clustering of data points
- **Output folder naming**: `geolife_eps{epsilon}/` (e.g., `geolife_eps20/`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                    DBSCAN Clustering Visualization                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│     Staypoints                         Clustered Locations           │
│                                                                      │
│        •  •                                 ○──○                     │
│       • • •              ───────►        ○───○──○  Location 1        │
│        •  •                                 ○──○                     │
│                                                                      │
│              •                                    ○                  │
│            • • •         ───────►              ○───○  Location 2     │
│              •                                    ○                  │
│                                                                      │
│    •   (isolated)        ───────►           ✕ (noise, filtered)     │
│                                                                      │
│    epsilon = 20m (max distance within cluster)                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Approach 2: H3-Based (Hexagonal Grid)

**Scripts**: `geolife_h3_1_raw_to_interim.py` + `geolife_h3_2_interim_to_processed.py`

- Uses **Uber H3** hierarchical hexagonal grid system
- Parameter: `h3_resolution` (0-15) - defines hexagon size
- **Advantage**: Consistent cell sizes, scalable, computationally efficient
- **Output folder naming**: `geolife_h3r{resolution}/` (e.g., `geolife_h3r8/`)

```
┌─────────────────────────────────────────────────────────────────────┐
│                      H3 Grid Visualization                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│     Staypoints                           H3 Hexagonal Grid           │
│                                                                      │
│       •  •                            ╱╲    ╱╲    ╱╲                 │
│      • • •                           │• •│    │  │                   │
│       •  •                           │•••│    │  │                   │
│                                       ╲╱    ╲╱    ╲╱                 │
│             •                         ╱╲    ╱╲    ╱╲                 │
│           • • •                      │  │ • │  │                     │
│             •                        │  │•••│  │                     │
│                                       ╲╱    ╲╱    ╲╱                 │
│                                                                      │
│     H3 Resolution 8: ~461m edge length                               │
│     Each hexagon = one location (if min samples met)                 │
└─────────────────────────────────────────────────────────────────────┘
```

### H3 Resolution Reference Table

| Resolution | Edge Length | Area (km²) | Use Case |
|------------|-------------|------------|----------|
| 7 | ~1.2 km | ~5.16 | City district level |
| 8 | ~461 m | ~0.74 | Neighborhood level |
| 9 | ~174 m | ~0.11 | Block level |
| 10 | ~66 m | ~0.015 | Building level |

---

## Pipeline Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          COMPLETE PIPELINE FLOW                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  INPUT                    SCRIPT 1                           OUTPUT              │
│  ─────                    ────────                           ──────              │
│                                                                                  │
│  data/raw_geolife/   ┌─────────────────────────────┐    data/geolife_eps20/     │
│       │              │                             │         │                   │
│       │              │  [1] read_geolife()         │         │                   │
│       └──────────────►      Position Fixes         │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│  config/geolife.yaml │  [2] generate_staypoints() │         │                   │
│       │              │      Staypoints             │         │                   │
│       │              │           │                 │         │                   │
│       └──────────────►           ▼                 │         │                   │
│                      │  [3] create_activity_flag() │         │                   │
│                      │      Activity Staypoints    │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [4] calculate_user_quality()        │                   │
│                      │      Valid Users            │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [5] generate_locations()   │         │                   │
│                      │      (DBSCAN/H3)            │         │                   │
│                      │      Locations              │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [6] merge_staypoints()     │         │                   │
│                      │      Merged Staypoints      │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [7] enrich_time_info()     ├─────────► interim/          │
│                      │      Temporal Features      │         │  intermediate.csv │
│                      └─────────────────────────────┘         │                   │
│                                                              │                   │
│                      ┌─────────────────────────────┐         │                   │
│                      │                             │         │                   │
│  interim/            │  [1] Load interim data      │◄────────┘                   │
│  intermediate.csv    │           │                 │                             │
│       │              │           ▼                 │                             │
│       └──────────────►  [2] split_dataset()        │    data/geolife_eps20/     │
│                      │      train/val/test         │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [3] OrdinalEncoder()       │         │                   │
│                      │      Encode locations       │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [4] get_valid_sequence()   │         │                   │
│                      │      Filter sequences       │         │                   │
│                      │           │                 │         │                   │
│                      │           ▼                 │         │                   │
│                      │  [5] generate_sequences()   ├─────────► processed/        │
│                      │      Create .pk files       │         │  train.pk         │
│                      │                             │         │  validation.pk    │
│                      │  SCRIPT 2                   │         │  test.pk          │
│                      └─────────────────────────────┘         │  metadata.json    │
│                                                              │                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Concepts

### 1. Position Fixes (GPS Points)
Raw GPS recordings with latitude, longitude, altitude, and timestamp. Each point represents where a GPS device recorded the user's location at a specific moment.

### 2. Staypoints
Locations where a user stayed for a significant period (defined by `time_threshold` and `dist_threshold`). Staypoints represent meaningful stops like home, work, or shops.

### 3. Triplegs
Movement segments between staypoints. Used for calculating tracking quality but not directly in the final output.

### 4. Locations
Clustered staypoints that represent the same semantic place. Multiple visits to "home" result in multiple staypoints but one location.

### 5. Activity Flag
Boolean indicating if a staypoint represents an activity (longer than `activity_time_threshold`).

### 6. Tracking Quality
Metric measuring how complete a user's tracking data is, calculated using a sliding window approach.

### 7. Previous Day Window
The number of days of history used to create input sequences. For example, `previous_day=7` means we look at the past 7 days of visits to predict the next location.

### 8. Sequence
A training sample consisting of:
- **X**: History of location visits (input)
- **Y**: Next location to predict (target)
- **Auxiliary features**: User ID, weekday, start time, duration, day difference

---

## Output Structure

```
data/
├── raw_geolife/                          # Input: Raw GeoLife data
│   └── Data/
│       ├── 000/
│       │   └── Trajectory/
│       │       └── *.plt files
│       ├── 001/
│       └── ...
│
├── geolife_eps20/                        # Output: DBSCAN version
│   ├── interim/                          # Script 1 outputs
│   │   ├── intermediate_eps20.csv        # Main interim file
│   │   ├── staypoints_all_eps20.csv      # All staypoints
│   │   ├── staypoints_merged_eps20.csv   # Merged staypoints
│   │   ├── locations_eps20.csv           # Location definitions
│   │   ├── valid_users_eps20.csv         # Valid user list
│   │   ├── raw_stats_eps20.json          # Raw data statistics
│   │   ├── interim_stats_eps20.json      # Interim statistics
│   │   └── quality/
│   │       └── user_quality_eps20.csv    # User quality scores
│   │
│   └── processed/                        # Script 2 outputs
│       ├── geolife_eps20_prev7_train.pk      # Training sequences
│       ├── geolife_eps20_prev7_validation.pk # Validation sequences
│       ├── geolife_eps20_prev7_test.pk       # Test sequences
│       └── geolife_eps20_prev7_metadata.json # Dataset metadata
│
└── geolife_h3r8/                         # Output: H3 version
    ├── interim/
    │   └── ... (similar structure)
    └── processed/
        └── ... (similar structure)
```

---

## Quick Start

### Running DBSCAN-Based Pipeline

```bash
# Step 1: Raw to Interim
python preprocessing/geolife_1_raw_to_interim.py --config config/preprocessing/geolife.yaml

# Step 2: Interim to Processed
python preprocessing/geolife_2_interim_to_processed.py --config config/preprocessing/geolife.yaml
```

### Running H3-Based Pipeline

```bash
# Step 1: Raw to Interim (H3)
python preprocessing/geolife_h3_1_raw_to_interim.py --config config/preprocessing/geolife_h3.yaml

# Step 2: Interim to Processed (H3)
python preprocessing/geolife_h3_2_interim_to_processed.py --config config/preprocessing/geolife_h3.yaml
```

---

## Next Steps

Continue reading the documentation in this order:
1. [02-GEOLIFE-DATASET.md](02-GEOLIFE-DATASET.md) - Understanding the GeoLife dataset
2. [03-CONFIGURATION.md](03-CONFIGURATION.md) - Configuration parameters explained
3. [04-SCRIPT1-RAW-TO-INTERIM.md](04-SCRIPT1-RAW-TO-INTERIM.md) - Detailed Script 1 walkthrough
4. [05-SCRIPT2-INTERIM-TO-PROCESSED.md](05-SCRIPT2-INTERIM-TO-PROCESSED.md) - Detailed Script 2 walkthrough
5. [06-H3-SCRIPT1-RAW-TO-INTERIM.md](06-H3-SCRIPT1-RAW-TO-INTERIM.md) - H3 version Script 1
6. [07-H3-SCRIPT2-INTERIM-TO-PROCESSED.md](07-H3-SCRIPT2-INTERIM-TO-PROCESSED.md) - H3 version Script 2

---

*Documentation Version: 1.0*
*Last Updated: 2024*
*For PhD Research Reference*
