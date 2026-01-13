# Configuration Files - Complete Reference

## Table of Contents
1. [Overview](#overview)
2. [DBSCAN Configuration (geolife.yaml)](#dbscan-configuration-geolifeyaml)
3. [H3 Configuration (geolife_h3.yaml)](#h3-configuration-geolife_h3yaml)
4. [Parameter Deep Dive](#parameter-deep-dive)
5. [How Parameters Affect Output](#how-parameters-affect-output)
6. [Parameter Tuning Guidelines](#parameter-tuning-guidelines)

---

## Overview

The preprocessing pipeline is controlled by **YAML configuration files** that define all parameters for data processing. There are two main configuration files:

| Config File | Location | Purpose |
|-------------|----------|---------|
| `geolife.yaml` | `config/preprocessing/geolife.yaml` | DBSCAN-based location clustering |
| `geolife_h3.yaml` | `config/preprocessing/geolife_h3.yaml` | H3-based location assignment |

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    CONFIGURATION FILE ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│   config/                                                                        │
│   └── preprocessing/                                                             │
│       ├── geolife.yaml        ────► DBSCAN Pipeline                             │
│       │   • epsilon: 20                                                          │
│       │   • Clusters nearby staypoints                                           │
│       │                                                                          │
│       └── geolife_h3.yaml     ────► H3 Pipeline                                 │
│           • h3_resolution: 8                                                     │
│           • Uses hexagonal grid                                                  │
│                                                                                  │
│   Both configs share:                                                            │
│   • staypoint parameters                                                         │
│   • quality_filter settings                                                      │
│   • split ratios                                                                 │
│   • previous_day list                                                            │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## DBSCAN Configuration (geolife.yaml)

### Complete Configuration File

```yaml
# Geolife Dataset Configuration
# Parameters: epsilon (for DBSCAN), previous_day (list, for sequence generation)

dataset:
  name: "geolife"
  epsilon: 20  # DBSCAN epsilon for location clustering (in meters)
  previous_day: [7]  # List of previous days for sequence generation

preprocessing:
  # Staypoint detection parameters
  staypoint:
    gap_threshold: 1440  # 24 hours in minutes
    dist_threshold: 200  # Distance threshold in meters
    time_threshold: 30   # Time threshold in minutes
    activity_time_threshold: 25  # Activity flag threshold in minutes
  
  # Location clustering parameters
  location:
    num_samples: 2  # Minimum samples for DBSCAN
    distance_metric: "haversine"
    agg_level: "dataset"
  
  # User quality filter
  quality_filter:
    day_filter: 50  # Minimum tracking days
    window_size: 10  # Sliding window size for quality calculation
  
  # Duration truncation (in minutes)
  max_duration: 2880  # 2 days (60 * 24 * 2)
  
  # Train/validation/test split ratios (per user)
  split:
    train: 0.6
    val: 0.2
    test: 0.2

random_seed: 42
```

### Parameter Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                     DBSCAN CONFIG PARAMETER MAP                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  dataset:                                                                        │
│  ├── name: "geolife"          ─► Output folder: geolife_eps{epsilon}/           │
│  ├── epsilon: 20              ─► DBSCAN radius parameter (meters)               │
│  └── previous_day: [7]        ─► History window sizes to generate               │
│                                                                                  │
│  preprocessing:                                                                  │
│  ├── staypoint:               ─► Staypoint detection settings                   │
│  │   ├── gap_threshold        ─► Max gap before new trajectory (min)            │
│  │   ├── dist_threshold       ─► Max movement distance to stay (m)              │
│  │   ├── time_threshold       ─► Min duration to be staypoint (min)             │
│  │   └── activity_time_threshold ─► Min duration for activity (min)             │
│  │                                                                               │
│  ├── location:                ─► DBSCAN clustering settings                     │
│  │   ├── num_samples          ─► Min points to form cluster                     │
│  │   ├── distance_metric      ─► Distance calculation method                    │
│  │   └── agg_level            ─► Aggregation level                              │
│  │                                                                               │
│  ├── quality_filter:          ─► User selection criteria                        │
│  │   ├── day_filter           ─► Min tracking days required                     │
│  │   └── window_size          ─► Sliding window for quality                     │
│  │                                                                               │
│  ├── max_duration             ─► Cap duration at this value (min)               │
│  │                                                                               │
│  └── split:                   ─► Dataset split ratios                           │
│      ├── train: 0.6           ─► 60% for training                               │
│      ├── val: 0.2             ─► 20% for validation                             │
│      └── test: 0.2            ─► 20% for testing                                │
│                                                                                  │
│  random_seed: 42              ─► Reproducibility seed                           │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## H3 Configuration (geolife_h3.yaml)

### Complete Configuration File

```yaml
# Geolife Dataset Configuration - H3 Version
# Uses Uber H3 hexagonal grid for location clustering instead of DBSCAN

dataset:
  name: "geolife"
  h3_resolution: 8  # H3 resolution (0-15, higher = finer grid). Resolution 8 ~= 461m edge length
  previous_day: [7]  # List of previous days for sequence generation

preprocessing:
  # Staypoint detection parameters
  staypoint:
    gap_threshold: 1440  # 24 hours in minutes
    dist_threshold: 200  # Distance threshold in meters
    time_threshold: 30   # Time threshold in minutes
    activity_time_threshold: 25  # Activity flag threshold in minutes
  
  # H3 location clustering parameters
  location:
    num_samples: 2  # Minimum samples (staypoints) required in H3 cell to be valid location
  
  # User quality filter
  quality_filter:
    day_filter: 50  # Minimum tracking days
    window_size: 10  # Sliding window size for quality calculation
  
  # Duration truncation (in minutes)
  max_duration: 2880  # 2 days (60 * 24 * 2)
  
  # Train/validation/test split ratios (per user)
  split:
    train: 0.6
    val: 0.2
    test: 0.2

random_seed: 42
```

### Key Difference: epsilon vs h3_resolution

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      DBSCAN vs H3 - KEY PARAMETER DIFFERENCE                     │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DBSCAN (geolife.yaml):                                                         │
│  ─────────────────────                                                          │
│  dataset:                                                                        │
│    epsilon: 20  ◄──────── Distance parameter in METERS                          │
│                           Points within 20m can be in same cluster              │
│                                                                                  │
│  Output folder: geolife_eps20/                                                   │
│                                                                                  │
│  ─────────────────────────────────────────────────────────────────────────      │
│                                                                                  │
│  H3 (geolife_h3.yaml):                                                          │
│  ─────────────────────                                                          │
│  dataset:                                                                        │
│    h3_resolution: 8  ◄───── Resolution level (0-15)                             │
│                              Resolution 8 = ~461m hexagon edge                   │
│                                                                                  │
│  Output folder: geolife_h3r8/                                                    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Deep Dive

### 1. Staypoint Parameters

These parameters control how GPS points are converted into staypoints.

#### gap_threshold (1440 minutes = 24 hours)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          gap_threshold = 1440 (24 hours)                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Split trajectories when GPS is off for too long                        │
│                                                                                  │
│  Scenario 1: Gap < 24 hours (same trajectory)                                    │
│  ─────────────────────────────────────────────                                   │
│  Timeline: ●●●●●●●●●      (8h gap)      ●●●●●●●●●                               │
│            Day 1                        Day 1 later                              │
│            └──────────── Same Trajectory ───────────┘                           │
│                                                                                  │
│  Scenario 2: Gap > 24 hours (new trajectory)                                     │
│  ─────────────────────────────────────────────                                   │
│  Timeline: ●●●●●●●●●     (36h gap)     ●●●●●●●●●                                │
│            Day 1                       Day 3                                     │
│            └─ Traj 1 ─┘               └─ Traj 2 ─┘                               │
│                                                                                  │
│  Why 24 hours? Assumes user typically has GPS on during daily activities.        │
│  Longer gaps suggest device was intentionally turned off.                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### dist_threshold (200 meters)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          dist_threshold = 200 meters                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Define maximum movement to still be considered "staying"               │
│                                                                                  │
│  Visualization:                                                                  │
│  ─────────────                                                                   │
│                    200m radius                                                   │
│                   ┌──────────┐                                                   │
│                  ╱            ╲                                                  │
│                 │   ●  ●  ●    │                                                 │
│                 │  ●   ●   ●   │  All points within 200m circle                 │
│                 │   ●  ●  ●    │  = User is "staying"                           │
│                  ╲            ╱                                                  │
│                   └──────────┘                                                   │
│                                                                                  │
│  Why 200m?                                                                       │
│  • GPS has inherent error (~5-15m outdoors, more indoors)                        │
│  • Users move within a location (e.g., walking around a mall)                    │
│  • 200m captures typical "place" boundaries                                      │
│                                                                                  │
│  Effect of different values:                                                     │
│  • 50m: Very strict, only captures true stationary points                        │
│  • 200m: Balanced, captures activity at locations                                │
│  • 500m: Loose, may merge nearby distinct locations                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### time_threshold (30 minutes)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          time_threshold = 30 minutes                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Minimum duration to qualify as a staypoint                             │
│                                                                                  │
│  Example:                                                                        │
│  ─────────                                                                       │
│                                                                                  │
│  Visit 1: At coffee shop for 45 minutes                                          │
│           Duration (45 min) ≥ Threshold (30 min) ✓                               │
│           → Creates STAYPOINT                                                    │
│                                                                                  │
│  Visit 2: Stopped at traffic light for 5 minutes                                 │
│           Duration (5 min) < Threshold (30 min) ✗                                │
│           → NOT a staypoint (filtered out)                                       │
│                                                                                  │
│  Why 30 minutes?                                                                 │
│  • Filters out brief stops (traffic, quick pickups)                              │
│  • Captures meaningful visits (meals, meetings, shopping)                        │
│  • Based on mobility research: 15-30 min typical threshold                       │
│                                                                                  │
│  Timeline visualization:                                                         │
│  ─────────────────────────────────────────────────────────────────────────      │
│  |←──────── 30 min threshold ────────►|                                         │
│  |   ✗ Brief stop (5 min)            |                                          │
│  |───────────────────────────────────|── ✓ Valid staypoint (45 min) ──────|    │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### activity_time_threshold (25 minutes)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                      activity_time_threshold = 25 minutes                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Flag staypoints as "activity" if duration exceeds threshold            │
│                                                                                  │
│  Why separate from time_threshold?                                               │
│  • time_threshold: Decides IF a staypoint is created                             │
│  • activity_time_threshold: Marks staypoints as meaningful activities            │
│                                                                                  │
│  Example:                                                                        │
│  ─────────                                                                       │
│  Staypoint at Mall: Duration = 120 minutes                                       │
│  120 min ≥ 25 min → is_activity = True ✓                                        │
│                                                                                  │
│  In this pipeline:                                                               │
│  • Only activity staypoints are kept for location prediction                     │
│  • Non-activity staypoints are filtered out in step 5 of Script 1                │
│                                                                                  │
│  Filtering logic:                                                                │
│  sp = sp.loc[sp["is_activity"] == True]                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 2. Location Parameters (DBSCAN)

#### epsilon (20 meters)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           DBSCAN epsilon = 20 meters                             │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Maximum distance between staypoints in the same location cluster       │
│                                                                                  │
│  DBSCAN Algorithm:                                                               │
│  ─────────────────                                                               │
│  For each staypoint:                                                             │
│    1. Find all staypoints within epsilon distance                                │
│    2. If count ≥ num_samples, form/join cluster                                  │
│    3. Expand cluster recursively                                                 │
│    4. Points not in any cluster = noise (filtered out)                           │
│                                                                                  │
│  Visualization:                                                                  │
│  ─────────────                                                                   │
│                                                                                  │
│     epsilon = 20m                                                                │
│     ┌───────┐                                                                    │
│    │●  ●  ● │ ← These staypoints within 20m of each other                       │
│    │ ●    ● │   = Same location (Cluster 1)                                      │
│     └───────┘                                                                    │
│                                                                                  │
│            ●  ← Isolated staypoint > 20m from others                             │
│               = Noise (filtered out)                                             │
│                                                                                  │
│     ┌────┐                                                                       │
│    │ ●● │ ← Different cluster (Location 2)                                       │
│     └────┘   because > 20m from Location 1                                       │
│                                                                                  │
│  Why 20 meters?                                                                  │
│  • GPS error is typically 5-15 meters                                            │
│  • 20m accounts for GPS error + small movement within place                      │
│  • Common values in literature: 10-50 meters                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### num_samples (2)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            num_samples = 2                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Minimum staypoints required to form a valid location                   │
│                                                                                  │
│  Effect:                                                                         │
│  ────────                                                                        │
│  num_samples = 2 means:                                                          │
│  • Need at least 2 visits to the same place to be a "location"                   │
│  • Single visits are treated as noise (filtered out)                             │
│                                                                                  │
│  Justification:                                                                  │
│  ─────────────                                                                   │
│  For location prediction, we need:                                               │
│  • Repeated visits to establish patterns                                         │
│  • Single visits don't help prediction                                           │
│  • num_samples = 2 is minimum for meaningful location                            │
│                                                                                  │
│  Example:                                                                        │
│  ─────────                                                                       │
│  User visits:                                                                    │
│  - Home: 50 times → Location ✓ (50 ≥ 2)                                         │
│  - Work: 30 times → Location ✓ (30 ≥ 2)                                         │
│  - Random cafe: 1 time → Noise ✗ (1 < 2)                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 3. H3 Resolution Parameter

#### h3_resolution (8)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           H3 Resolution = 8                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  H3 Resolution Reference:                                                        │
│  ─────────────────────────                                                       │
│                                                                                  │
│  Resolution │ Edge Length │   Area   │ Example Use Case                         │
│  ───────────│─────────────│──────────│────────────────────────────              │
│      0      │   ~1,100 km │  Large   │ Continent level                          │
│      1      │    ~420 km  │    ↓     │ Large region                             │
│      2      │    ~158 km  │    ↓     │ Country/State                            │
│      3      │     ~60 km  │    ↓     │ Metropolitan area                        │
│      4      │     ~22 km  │    ↓     │ Large city                               │
│      5      │    ~8.5 km  │    ↓     │ City district                            │
│      6      │    ~3.2 km  │    ↓     │ Large neighborhood                       │
│      7      │    ~1.2 km  │    ↓     │ Neighborhood                             │
│  →   8      │   ~461 m    │    ↓     │ Block/Several buildings ← USED           │
│      9      │   ~174 m    │    ↓     │ Single building                          │
│     10      │    ~66 m    │    ↓     │ Small building                           │
│     11      │    ~25 m    │    ↓     │ Single room                              │
│     12      │    ~9.4 m   │    ↓     │ Precise location                         │
│     13      │    ~3.6 m   │    ↓     │ Very precise                             │
│     14      │    ~1.4 m   │    ↓     │ GPS precision                            │
│     15      │   ~0.5 m    │  Small   │ Sub-meter                                │
│                                                                                  │
│  Why Resolution 8?                                                               │
│  • ~461m edge length (~730m diameter)                                            │
│  • Similar granularity to DBSCAN epsilon=20 with multiple visits                 │
│  • Captures "place" level locations (mall, office building)                      │
│  • Not too fine (would create too many locations)                                │
│  • Not too coarse (would merge distinct places)                                  │
│                                                                                  │
│  Hexagon visualization:                                                          │
│  ──────────────────────                                                          │
│                                                                                  │
│       ╱╲    ╱╲    ╱╲                                                             │
│      │  │  │  │  │  │   Each hexagon is a location                              │
│       ╲╱    ╲╱    ╲╱                                                             │
│       ╱╲    ╱╲    ╱╲    Resolution 8: ~461m edge                                │
│      │●●│  │  │  │●│    Staypoints assigned to hexagons                         │
│       ╲╱    ╲╱    ╲╱                                                             │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 4. Quality Filter Parameters

#### day_filter (50 days)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          day_filter = 50 days                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Minimum tracking days required for user to be included                 │
│                                                                                  │
│  Example:                                                                        │
│  ─────────                                                                       │
│  User A: Tracked for 120 days → Included ✓ (120 ≥ 50)                           │
│  User B: Tracked for 30 days  → Excluded ✗ (30 < 50)                            │
│  User C: Tracked for 50 days  → Included ✓ (50 ≥ 50)                            │
│                                                                                  │
│  Why 50 days?                                                                    │
│  • Need enough data to capture weekly patterns (at least 7 weeks)                │
│  • 50 days allows for train/val/test split with meaningful data                  │
│  • Filters out users with sporadic tracking                                      │
│                                                                                  │
│  Timeline:                                                                       │
│  |←──────────────── 50 days minimum ─────────────────►|                         │
│  |      Week 1-7 (training)     |  Week 8+ (val/test) |                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### window_size (10 weeks)

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          window_size = 10 weeks                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: Sliding window size for calculating tracking quality                   │
│                                                                                  │
│  How it works:                                                                   │
│  ─────────────                                                                   │
│  1. Slide 10-week window across user's tracking period                           │
│  2. Calculate coverage (tracked time / total time) for each window               │
│  3. Average all window qualities for final user quality score                    │
│                                                                                  │
│  Visualization:                                                                  │
│  ─────────────                                                                   │
│  User tracking period: 20 weeks                                                  │
│                                                                                  │
│  |←── Window 1 (10 weeks) ──►|                                                  │
│     |←── Window 2 (10 weeks) ──►|                                               │
│        |←── Window 3 (10 weeks) ──►|                                            │
│           ...                                                                    │
│                    |←── Window N (10 weeks) ──►|                                │
│                                                                                  │
│  Each window: Calculate % of time with GPS data                                  │
│  Final quality: Average of all window qualities                                  │
│                                                                                  │
│  Why 10 weeks?                                                                   │
│  • Captures long-term tracking consistency                                       │
│  • 10 weeks = ~2.5 months, enough to see patterns                                │
│  • Balances between too short (noisy) and too long (misses gaps)                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 5. Split Ratios

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    SPLIT RATIOS: train=0.6, val=0.2, test=0.2                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Split is CHRONOLOGICAL and PER USER (not random!)                               │
│                                                                                  │
│  For each user:                                                                  │
│  ───────────────                                                                 │
│  User's tracking days: Day 0 ────────────────────────────────────► Day N        │
│                                                                                  │
│  |←────── 60% Train ──────►|←─ 20% Val ─►|←─ 20% Test ─►|                       │
│  |    Early visits         |  Middle     |   Recent     |                       │
│                                                                                  │
│  Example: User with 100 days of data                                             │
│  • Days 0-59:   Training data   (60 days)                                        │
│  • Days 60-79:  Validation data (20 days)                                        │
│  • Days 80-99:  Test data       (20 days)                                        │
│                                                                                  │
│  Why chronological split?                                                        │
│  • Simulates real-world prediction scenario                                      │
│  • Train on past, predict future (no data leakage)                               │
│  • Captures temporal patterns correctly                                          │
│                                                                                  │
│  Why per-user split?                                                             │
│  • Each user has different tracking period                                       │
│  • Ensures each user appears in all splits                                       │
│  • Maintains user-specific patterns                                              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### 6. previous_day Parameter

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          previous_day = [7]                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Purpose: History window for sequence generation (in days)                       │
│                                                                                  │
│  How it works:                                                                   │
│  ─────────────                                                                   │
│  For each target staypoint, collect history from past N days                     │
│                                                                                  │
│  Example: previous_day = 7                                                       │
│  ─────────────────────────                                                       │
│                                                                                  │
│  Day:    0   1   2   3   4   5   6   7   8   9   10                             │
│          │   │   │   │   │   │   │   │   │   │   │                              │
│  Visits: ●───●───────●───────●───────●───●   ●   TARGET                         │
│          │←────── History Window (7 days) ──────►│                              │
│                                                                                  │
│  Input sequence X: [loc_day0, loc_day1, loc_day3, loc_day5, loc_day6]           │
│  Target Y: loc_day10                                                             │
│                                                                                  │
│  Why a list? [7]                                                                 │
│  • Can specify multiple values: [3, 7, 14, 21]                                   │
│  • Script 2 processes each value, generating separate output files               │
│  • Enables comparing different history window sizes                              │
│                                                                                  │
│  Example with previous_day = [3, 7, 14]:                                        │
│  • geolife_eps20_prev3_train.pk                                                  │
│  • geolife_eps20_prev7_train.pk                                                  │
│  • geolife_eps20_prev14_train.pk                                                 │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## How Parameters Affect Output

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    PARAMETER EFFECTS ON OUTPUT                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Parameter               │ Effect on Output                                      │
│  ────────────────────────│───────────────────────────────────────────────────    │
│                          │                                                       │
│  epsilon ↑ (larger)      │ Fewer, larger locations (more staypoints merged)     │
│  epsilon ↓ (smaller)     │ More, smaller locations (finer granularity)          │
│                          │                                                       │
│  h3_resolution ↑         │ More, smaller hexagons (finer granularity)           │
│  h3_resolution ↓         │ Fewer, larger hexagons (coarser)                     │
│                          │                                                       │
│  num_samples ↑           │ Fewer locations (need more visits to form loc)       │
│  num_samples ↓           │ More locations (single visits can form locs)         │
│                          │                                                       │
│  time_threshold ↑        │ Fewer staypoints (need longer visits)                │
│  time_threshold ↓        │ More staypoints (shorter visits included)            │
│                          │                                                       │
│  day_filter ↑            │ Fewer valid users (stricter quality)                 │
│  day_filter ↓            │ More valid users (lenient quality)                   │
│                          │                                                       │
│  previous_day ↑          │ Longer history, fewer valid sequences                │
│  previous_day ↓          │ Shorter history, more valid sequences                │
│                          │                                                       │
│  max_duration            │ Caps staypoint duration (outlier handling)           │
│                          │                                                       │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Parameter Tuning Guidelines

### For More Locations (Fine-Grained)

```yaml
# Fine-grained location detection
dataset:
  epsilon: 10          # Smaller clusters
  # OR
  h3_resolution: 9     # Smaller hexagons (~174m)

preprocessing:
  location:
    num_samples: 1     # Allow single-visit locations
  staypoint:
    time_threshold: 15 # Capture shorter visits
```

### For Fewer Locations (Coarse-Grained)

```yaml
# Coarse-grained location detection
dataset:
  epsilon: 50          # Larger clusters
  # OR
  h3_resolution: 7     # Larger hexagons (~1.2km)

preprocessing:
  location:
    num_samples: 5     # Require multiple visits
  staypoint:
    time_threshold: 60 # Only capture long visits
```

### For Strict Quality Control

```yaml
preprocessing:
  quality_filter:
    day_filter: 100    # Require 100+ days of tracking
    window_size: 15    # Longer quality assessment window
```

### For More Training Sequences

```yaml
dataset:
  previous_day: [3]    # Shorter history window

preprocessing:
  split:
    train: 0.7         # More training data
    val: 0.15
    test: 0.15
```

---

## Next Steps

Now that you understand the configuration parameters, proceed to:
- [04-SCRIPT1-RAW-TO-INTERIM.md](04-SCRIPT1-RAW-TO-INTERIM.md) - See how these parameters are used in Script 1

---

*Documentation Version: 1.0*
*For PhD Research Reference*
