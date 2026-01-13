# Pipeline Overview: DIY Dataset Preprocessing

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Pipeline Architecture](#pipeline-architecture)
3. [Data Flow Visualization](#data-flow-visualization)
4. [Processing Phases](#processing-phases)
5. [Two Location Clustering Approaches](#two-location-clustering-approaches)
6. [Directory Structure](#directory-structure)

---

## Introduction

The DIY (Do-It-Yourself) dataset preprocessing pipeline transforms raw GPS trajectory data into structured sequences suitable for next location prediction machine learning models. This pipeline follows a modular design with clear separation of concerns.

### Purpose
- Transform raw GPS points into meaningful location visits
- Filter high-quality users for reliable predictions
- Generate train/validation/test sequences for model training

### Key Outputs
- **Sequences**: Historical location visits with temporal features
- **Target**: Next location to predict
- **Metadata**: Dataset statistics and configuration

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         COMPLETE PIPELINE ARCHITECTURE                       │
└─────────────────────────────────────────────────────────────────────────────┘

                              ┌─────────────────┐
                              │   Raw GPS Data   │
                              │  (165M+ points)  │
                              └────────┬────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     NOTEBOOK: 02_psl_detection_all.ipynb                    │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                              │
│  ┌───────────────┐   ┌───────────────┐   ┌───────────────┐                  │
│  │ Position      │──▶│  Staypoint    │──▶│    Trip       │                  │
│  │ Fixes (pfs)   │   │  Detection    │   │  Generation   │                  │
│  └───────────────┘   └───────────────┘   └───────────────┘                  │
│                              │                                               │
│                              ▼                                               │
│                      ┌───────────────┐                                       │
│                      │    Quality    │                                       │
│                      │   Filtering   │                                       │
│                      └───────────────┘                                       │
│                              │                                               │
│                              ▼                                               │
│  OUTPUT: 3_staypoints_fun_generate_trips.csv                                │
│          10_filter_after_user_quality_DIY_slide_filteres.csv                │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCRIPT 1: Raw to Interim                                │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                              │
│           ┌─────────────────────┐     ┌─────────────────────┐              │
│           │  diy_1_raw_to_     │     │  diy_h3_1_raw_to_   │              │
│           │  interim.py        │ OR  │  interim.py          │              │
│           │  (DBSCAN)          │     │  (H3 Grid)           │              │
│           └─────────────────────┘     └─────────────────────┘              │
│                              │                                               │
│  STAGES:  1. Load Raw Data                                                   │
│           2. Generate Locations (DBSCAN or H3)                               │
│           3. Merge Consecutive Staypoints                                    │
│           4. Enrich Temporal Features                                        │
│                              │                                               │
│  OUTPUT: intermediate_eps{X}.csv or intermediate_h3r{X}.csv                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      SCRIPT 2: Interim to Processed                          │
│  ═══════════════════════════════════════════════════════════════════════    │
│                                                                              │
│           ┌─────────────────────┐     ┌─────────────────────┐              │
│           │  diy_2_interim_to_ │     │  diy_h3_2_interim_  │              │
│           │  processed.py      │ OR  │  to_processed.py     │              │
│           └─────────────────────┘     └─────────────────────┘              │
│                              │                                               │
│  STAGES:  1. Split Dataset (Train/Val/Test)                                  │
│           2. Encode Location IDs                                             │
│           3. Filter Valid Sequences                                          │
│           4. Generate Sequences                                              │
│           5. Save Pickle Files                                               │
│                              │                                               │
│  OUTPUT: *_train.pk, *_validation.pk, *_test.pk, *_metadata.json            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Data Flow Visualization

### Raw GPS to Staypoints

```
RAW GPS POINTS (165,429,633 records)
══════════════════════════════════════════════════════════════════════════════

Example Raw GPS Data:
┌──────────────────────────────────────────┬───────────┬─────────────┬─────────────────────────┐
│ user_id                                  │ latitude  │ longitude   │ tracked_at              │
├──────────────────────────────────────────┼───────────┼─────────────┼─────────────────────────┤
│ 9358664f-ad4b-46ff-9a65-e2efbf646e6e    │ -7.74776  │ 110.431541  │ 2021-10-24T02:07:56.000Z│
│ 9358664f-ad4b-46ff-9a65-e2efbf646e6e    │ -7.74778  │ 110.431542  │ 2021-10-24T02:08:01.000Z│
│ 9358664f-ad4b-46ff-9a65-e2efbf646e6e    │ -7.74775  │ 110.431540  │ 2021-10-24T02:08:06.000Z│
│ ...                                      │ ...       │ ...         │ ...                     │
└──────────────────────────────────────────┴───────────┴─────────────┴─────────────────────────┘
                                          │
                                          │ Sliding window detection:
                                          │ • dist_threshold: 100m
                                          │ • time_threshold: 30 min
                                          │ • gap_threshold: 24h
                                          ▼
STAYPOINTS (Detected stationary periods)
══════════════════════════════════════════════════════════════════════════════

Example Staypoint Data:
┌────┬────────────┬─────────────────────┬─────────────────────┬───────────────────────────┬─────────────┐
│ id │ user_id    │ started_at          │ finished_at         │ geom (POINT)              │ is_activity │
├────┼────────────┼─────────────────────┼─────────────────────┼───────────────────────────┼─────────────┤
│ 0  │ user_001   │ 2021-10-24 02:07:56 │ 2021-10-24 08:30:00 │ POINT(110.431541 -7.74776)│ True        │
│ 1  │ user_001   │ 2021-10-24 09:15:00 │ 2021-10-24 12:45:00 │ POINT(110.385742 -7.71172)│ True        │
│ 2  │ user_001   │ 2021-10-24 13:30:00 │ 2021-10-24 17:00:00 │ POINT(110.390480 -7.76398)│ True        │
└────┴────────────┴─────────────────────┴─────────────────────┴───────────────────────────┴─────────────┘

Key Insight: A staypoint is created when a user remains within 100m radius for at least 30 minutes.
```

### Staypoints to Locations

```
STAYPOINTS → LOCATION CLUSTERING
══════════════════════════════════════════════════════════════════════════════

Geographic View:
                    ▲ Latitude
                    │
        ★────────★  │  • = Individual Staypoint
       /  LOC_1  \  │  ★ = Cluster Center (Location)
      •    •    •   │
       \   •   /    │     DBSCAN: epsilon=50m, min_samples=2
        ★────────★  │           → Adaptive irregular shapes
                    │
    ★──────★        │     H3: resolution=8
   /  LOC_2 \       │           → Fixed hexagonal cells (~461m edge)
  •    •     •      │
   \  •  •  /       │
    ★──────★        │
                    │
          •         │     Noise points (not enough nearby points)
                    │     → Filtered out (location_id = NaN)
                    │
    ────────────────┼────────────────────▶ Longitude


LOCATION ASSIGNMENT RESULT:
┌────┬────────────┬─────────────────────┬─────────────────────┬─────────────┐
│ id │ user_id    │ started_at          │ finished_at         │ location_id │
├────┼────────────┼─────────────────────┼─────────────────────┼─────────────┤
│ 0  │ user_001   │ 2021-10-24 02:07:56 │ 2021-10-24 08:30:00 │ 42          │
│ 1  │ user_001   │ 2021-10-24 09:15:00 │ 2021-10-24 12:45:00 │ 15          │
│ 2  │ user_001   │ 2021-10-24 13:30:00 │ 2021-10-24 17:00:00 │ 42          │
│ 3  │ user_001   │ 2021-10-24 17:45:00 │ 2021-10-24 19:30:00 │ 8           │
│ 4  │ user_001   │ 2021-10-24 20:00:00 │ 2021-10-25 07:00:00 │ 42          │
└────┴────────────┴─────────────────────┴─────────────────────┴─────────────┘

Notice: Rows 0, 2, and 4 all have location_id=42 (same physical location, e.g., home)
```

### Locations to Sequences

```
TEMPORAL SPLITTING & SEQUENCE GENERATION
══════════════════════════════════════════════════════════════════════════════

USER TIMELINE (example: user tracked for 100 days)
┌──────────────────────────────────────────────────────────────────────────────┐
│ Day 0                        Day 80    Day 90                        Day 100│
│ ├──────────────────────────────┼─────────┼────────────────────────────────┤ │
│ │◄────────── TRAIN (80%) ────────▶│◄VAL(10%)▶│◄──── TEST (10%) ───▶│         │
└──────────────────────────────────────────────────────────────────────────────┘


SEQUENCE GENERATION (previous_day=7)
══════════════════════════════════════════════════════════════════════════════

For target staypoint at Day 10:

    Day 3   Day 4   Day 5   Day 6   Day 7   Day 8   Day 9   Day 10
    ┌─────┬───────┬───────┬───────┬───────┬───────┬───────┬────────┐
    │ SP1 │  SP2  │  SP3  │  SP4  │  SP5  │  SP6  │  SP7  │  SP8   │
    │L=42 │ L=15  │ L=42  │ L=8   │ L=42  │ L=15  │ L=42  │ L=15   │
    └─────┴───────┴───────┴───────┴───────┴───────┴───────┴────────┘
          │◄──────── History (X) ──────────────▶│  │◄Target(Y)▶│
          
    Valid because:
    • SP8 is at Day 10, which is >= previous_day (7)
    • History has 7 staypoints (>= min_length of 3)
    • All within the 7-day lookback window


GENERATED SEQUENCE DICTIONARY:
┌─────────────────────────────────────────────────────────────────────────────┐
│ {                                                                           │
│   "X":           [42, 15, 42, 8, 42, 15, 42]     # Location sequence       │
│   "user_X":      [1, 1, 1, 1, 1, 1, 1]           # User ID (repeated)      │
│   "weekday_X":   [0, 1, 2, 3, 4, 5, 6]           # Day of week (Mon=0)     │
│   "start_min_X": [420, 540, 480, 600, 450, 540, 420]  # Start minute       │
│   "dur_X":       [383, 210, 240, 105, 660, 210, 383]  # Duration (min)     │
│   "diff":        [7, 6, 5, 4, 3, 2, 1]           # Days before target      │
│   "Y":           15                              # Target location          │
│ }                                                                           │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Processing Phases

### Phase 0: PSL Detection (Notebook)

| Step | Operation | Input | Output | Description |
|------|-----------|-------|--------|-------------|
| 1 | Load GPS Data | CSV file | DataFrame | 165M+ GPS points |
| 2 | Create Positionfixes | DataFrame | GeoDataFrame | Add geometry column |
| 3 | Generate Staypoints | Positionfixes | Staypoints | Sliding window detection |
| 4 | Create Activity Flag | Staypoints | Staypoints | Mark activity staypoints |
| 5 | Generate Triplegs | Positionfixes + Staypoints | Triplegs | Movement segments |
| 6 | Generate Trips | Staypoints + Triplegs | Trips | Complete journeys |
| 7 | Quality Filtering | Staypoints + Trips | Valid Users | Filter reliable users |

### Phase 1: Raw to Interim (Python Script)

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Load Raw Data | Read staypoints and valid users |
| 2 | Filter Users | Keep only quality-filtered users |
| 3 | Filter Activities | Keep only activity staypoints |
| 4 | Generate Locations | DBSCAN or H3 clustering |
| 5 | Merge Staypoints | Combine consecutive same-location visits |
| 6 | Enrich Temporal | Add day, time, weekday features |
| 7 | Save Interim | Write intermediate CSV files |

### Phase 2: Interim to Processed (Python Script)

| Step | Operation | Description |
|------|-----------|-------------|
| 1 | Split Dataset | Temporal train/val/test split per user |
| 2 | Encode Locations | OrdinalEncoder with padding offset |
| 3 | Filter Sequences | Remove invalid sequences |
| 4 | Generate Sequences | Create X/Y pairs with features |
| 5 | Save Pickle | Write .pk files for each split |

---

## Two Location Clustering Approaches

### DBSCAN Clustering (Default)

```
DBSCAN ALGORITHM VISUALIZATION
══════════════════════════════════════════════════════════════════════════════

Parameters:
• epsilon = 50 meters (neighborhood radius)
• num_samples = 2 (minimum points to form cluster)

      ε=50m
    ┌───────┐
    │ •   • │  ← 3 points within 50m = CLUSTER (Location)
    │   •   │
    └───────┘

    •         ← 1 isolated point = NOISE (filtered out)


Advantages:
✓ Adaptive cluster shapes (follows natural boundaries)
✓ Works well with irregular spatial distributions
✓ No need to predefine number of clusters

Disadvantages:
✗ Sensitive to epsilon parameter
✗ May create very small or very large clusters
✗ Computationally expensive for large datasets
```

### H3 Grid Clustering (Alternative)

```
H3 HEXAGONAL GRID VISUALIZATION
══════════════════════════════════════════════════════════════════════════════

Resolution 8 (~461m edge length):

      _____
     /     \
    /   •   \    ← Hexagonal cell with 2+ staypoints = LOCATION
   /  •   •  \
   \         /
    \_______/
                  
       •          ← Cell with < 2 staypoints = NOISE (filtered out)
                  

Advantages:
✓ Consistent cell sizes globally
✓ Efficient spatial indexing
✓ Reproducible (same location always maps to same cell)
✓ Computationally efficient

Disadvantages:
✗ Fixed grid doesn't adapt to spatial density
✗ Boundaries are arbitrary
✗ May split natural clusters

H3 Resolution Comparison:
┌────────────┬──────────────────┬────────────────────┐
│ Resolution │ Edge Length (km) │ Area (km²)         │
├────────────┼──────────────────┼────────────────────┤
│ 7          │ 1.220            │ 5.161              │
│ 8          │ 0.461            │ 0.737 (default)    │
│ 9          │ 0.174            │ 0.105              │
│ 10         │ 0.066            │ 0.015              │
└────────────┴──────────────────┴────────────────────┘
```

---

## Directory Structure

```
next_loc_clean_v2/
├── config/
│   └── preprocessing/
│       ├── diy.yaml              # DBSCAN configuration
│       └── diy_h3.yaml           # H3 configuration
│
├── preprocessing/
│   ├── 02_psl_detection_all.ipynb  # PSL detection notebook
│   ├── diy_1_raw_to_interim.py     # DBSCAN: raw → interim
│   ├── diy_2_interim_to_processed.py # DBSCAN: interim → processed
│   ├── diy_h3_1_raw_to_interim.py    # H3: raw → interim
│   └── diy_h3_2_interim_to_processed.py # H3: interim → processed
│
├── data/
│   ├── raw_diy/                  # Input data from notebook
│   │   ├── 3_staypoints_fun_generate_trips.csv
│   │   └── 10_filter_after_user_quality_DIY_slide_filteres.csv
│   │
│   ├── diy_eps50/                # DBSCAN output (epsilon=50)
│   │   ├── interim/
│   │   └── processed/
│   │
│   └── diy_h3r8/                 # H3 output (resolution=8)
│       ├── interim/
│       └── processed/
│
└── docs/
    └── docs_preprocessing_diy/   # This documentation
```

---

## Summary

The DIY preprocessing pipeline transforms 165+ million raw GPS points into structured sequences for next location prediction:

1. **PSL Detection**: Raw GPS → Staypoints (stationary periods)
2. **Quality Filtering**: Remove unreliable users
3. **Location Clustering**: Staypoints → Locations (DBSCAN or H3)
4. **Temporal Enrichment**: Add time-based features
5. **Sequence Generation**: Create train/val/test sequences

The output sequences contain historical location visits with temporal features, ready for machine learning model training.
