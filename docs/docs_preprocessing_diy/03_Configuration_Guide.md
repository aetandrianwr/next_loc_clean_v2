# Configuration Guide

## 📋 Table of Contents
1. [Overview](#overview)
2. [Configuration Files](#configuration-files)
3. [DBSCAN Configuration (diy.yaml)](#dbscan-configuration-diyyaml)
4. [H3 Configuration (diy_h3.yaml)](#h3-configuration-diy_h3yaml)
5. [Parameter Deep Dive](#parameter-deep-dive)
6. [Configuration Examples](#configuration-examples)
7. [Parameter Tuning Guide](#parameter-tuning-guide)

---

## Overview

The DIY preprocessing pipeline uses YAML configuration files to control all aspects of data processing. This allows for reproducible experiments and easy parameter tuning without modifying code.

### Configuration File Locations
```
config/
└── preprocessing/
    ├── diy.yaml           # DBSCAN-based location clustering
    ├── diy_h3.yaml        # H3 grid-based location clustering
    ├── diy_multidays.yaml # Multiple previous_day values
    └── diy_multidays_p24_p28.yaml  # Extended history windows
```

---

## Configuration Files

### File Structure Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONFIGURATION FILE STRUCTURE                          │
└─────────────────────────────────────────────────────────────────────────────┘

diy.yaml / diy_h3.yaml
├── dataset:
│   ├── name              → Dataset identifier
│   ├── epsilon/h3_resolution → Location clustering parameter
│   └── previous_day      → History window size(s)
│
├── preprocessing:
│   ├── location:         → Clustering parameters
│   │   ├── num_samples
│   │   ├── distance_metric
│   │   └── agg_level
│   │
│   ├── staypoint_merging:
│   │   └── max_time_gap
│   │
│   ├── quality_filter:   → User filtering criteria
│   │   ├── day_filter
│   │   ├── window_size
│   │   ├── min_thres
│   │   └── mean_thres
│   │
│   ├── max_duration      → Duration truncation
│   ├── min_sequence_length → Minimum history length
│   │
│   └── split:            → Train/val/test ratios
│       ├── train
│       ├── val
│       └── test
│
└── random_seed           → Reproducibility seed
```

---

## DBSCAN Configuration (diy.yaml)

### Complete Configuration with Annotations

```yaml
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                    DIY DATASET CONFIGURATION (DBSCAN)                      ║
# ║ File: config/preprocessing/diy.yaml                                        ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ DATASET SECTION                                                           │
# │ Core parameters that define the dataset and output naming                 │
# └───────────────────────────────────────────────────────────────────────────┘
dataset:
  name: "diy"                    # Dataset identifier
                                 # Used in: output folder names, file prefixes
                                 # Example output: data/diy_eps50/
                                 
  epsilon: 50                    # DBSCAN epsilon parameter (in meters)
                                 # Defines the maximum distance between two points
                                 # to be considered in the same neighborhood
                                 # 
                                 # Visual explanation:
                                 #     ε=50m radius
                                 #    ┌─────────┐
                                 #    │  • • •  │  Points within 50m = same cluster
                                 #    │    •    │
                                 #    └─────────┘
                                 #
                                 # Smaller ε → More, smaller locations
                                 # Larger ε  → Fewer, larger locations
                                 
  previous_day: [7]              # History window size in days (LIST)
                                 # For sequence generation, look back N days
                                 # 
                                 # Can be list for multiple outputs:
                                 # previous_day: [7, 14, 28]
                                 # 
                                 # Example: previous_day=7
                                 #   Day 0  Day 1  Day 2  Day 3  Day 4  Day 5  Day 6  Day 7
                                 #   ├──────────────────────────────────────────────┤   ↑
                                 #              History (X)                         Target

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ PREPROCESSING SECTION                                                      │
# │ All preprocessing parameters for data transformation                       │
# └───────────────────────────────────────────────────────────────────────────┘
preprocessing:
  
  # ─────────────────────────────────────────────────────────────────────────
  # LOCATION CLUSTERING (DBSCAN Parameters)
  # Used by: diy_1_raw_to_interim.py → generate_locations()
  # ─────────────────────────────────────────────────────────────────────────
  location:
    num_samples: 2               # Minimum number of points to form a cluster
                                 # DBSCAN min_samples parameter
                                 #
                                 # num_samples=2: At least 2 staypoints needed
                                 # to form a location (most permissive)
                                 #
                                 #   • •  → Location (2 points)
                                 #   •    → Noise (only 1 point, filtered out)
                                 
    distance_metric: "haversine" # Distance calculation method
                                 # Options: "haversine", "euclidean"
                                 # 
                                 # haversine: Great-circle distance on sphere
                                 #   Accounts for Earth's curvature
                                 #   Required for geographic coordinates (lat/lon)
                                 #   
                                 # euclidean: Straight-line distance
                                 #   Only for projected coordinates (x/y meters)
                                 
    agg_level: "dataset"         # Aggregation level for clustering
                                 # Options: "user", "dataset"
                                 #
                                 # "dataset": Cluster ALL staypoints together
                                 #   → Same location_id for same physical place
                                 #   → Enables cross-user location patterns
                                 #
                                 # "user": Cluster per user separately
                                 #   → Each user has own location IDs
                                 #   → Location 1 for user A ≠ Location 1 for user B
  
  # ─────────────────────────────────────────────────────────────────────────
  # STAYPOINT MERGING
  # Used by: diy_1_raw_to_interim.py → merge_staypoints()
  # ─────────────────────────────────────────────────────────────────────────
  staypoint_merging:
    max_time_gap: "1min"         # Maximum gap to merge consecutive staypoints
                                 # at the SAME location
                                 #
                                 # Format: "{number}{unit}" 
                                 # Units: "min", "hour", "day"
                                 #
                                 # Example with max_time_gap="1min":
                                 # 
                                 # Before:
                                 #   SP1 @ Loc42: 08:00-08:30
                                 #   SP2 @ Loc42: 08:30:30-09:00  (30s gap)
                                 #   SP3 @ Loc15: 09:15-10:00
                                 #
                                 # After:
                                 #   SP1 @ Loc42: 08:00-09:00  (merged!)
                                 #   SP2 @ Loc15: 09:15-10:00
  
  # ─────────────────────────────────────────────────────────────────────────
  # USER QUALITY FILTER
  # Applied in notebook: 02_psl_detection_all.ipynb
  # These parameters document the filtering done upstream
  # ─────────────────────────────────────────────────────────────────────────
  quality_filter:
    day_filter: 60               # Minimum tracking days required
                                 # Users with < 60 days are excluded
                                 #
                                 # Justification: Need sufficient history for:
                                 #   - Training data volume
                                 #   - Meaningful patterns
                                 #   - Valid train/val/test splits
                                 
    window_size: 10              # Sliding window size (in weeks) for quality
                                 # Calculates tracking quality in 10-week windows
                                 #
                                 # Quality = tracked_time / total_time in window
                                 
    min_thres: 0.6               # Minimum quality threshold
                                 # ANY sliding window must have quality > 0.6
                                 # 
                                 # Ensures no "dead" periods in tracking
                                 
    mean_thres: 0.7              # Mean quality threshold
                                 # Average quality across all windows > 0.7
                                 #
                                 # Ensures overall good tracking coverage
  
  # ─────────────────────────────────────────────────────────────────────────
  # DURATION TRUNCATION
  # Used by: diy_2_interim_to_processed.py → process_for_previous_day()
  # ─────────────────────────────────────────────────────────────────────────
  max_duration: 2880             # Maximum duration in minutes (2 days)
                                 # 2880 = 60 min × 24 hours × 2 days
                                 #
                                 # Durations exceeding this are truncated to 2879
                                 # 
                                 # Justification:
                                 #   - Extremely long stays are outliers
                                 #   - Prevents one feature from dominating
                                 #   - Common in embedding-based models
  
  # ─────────────────────────────────────────────────────────────────────────
  # SEQUENCE GENERATION
  # Used by: diy_2_interim_to_processed.py → get_valid_sequence()
  # ─────────────────────────────────────────────────────────────────────────
  min_sequence_length: 3         # Minimum number of staypoints in history
                                 # 
                                 # A valid sequence needs at least 3 history points
                                 # to provide meaningful patterns for prediction
                                 #
                                 # Too short: [SP1, SP2] → Y (not enough context)
                                 # Valid:     [SP1, SP2, SP3] → Y (minimum context)
  
  # ─────────────────────────────────────────────────────────────────────────
  # TRAIN/VALIDATION/TEST SPLIT
  # Used by: diy_2_interim_to_processed.py → split_dataset()
  # ─────────────────────────────────────────────────────────────────────────
  split:
    train: 0.8                   # 80% of each user's timeline for training
    val: 0.1                     # 10% for validation
    test: 0.1                    # 10% for testing
                                 #
                                 # TEMPORAL SPLIT (not random!):
                                 # 
                                 # User Timeline:
                                 # Day 0                                    Day 100
                                 # ├───────────────────┼─────┼─────────────────┤
                                 # │◄─── Train (80%) ──▶│Val │◄── Test (10%) ─▶│
                                 # │     Day 0-80      │10% │    Day 90-100    │
                                 #
                                 # Why temporal split?
                                 #   - Prevents data leakage (future → past)
                                 #   - Simulates real prediction scenario
                                 #   - Each user contributes to all splits

# ┌───────────────────────────────────────────────────────────────────────────┐
# │ RANDOM SEED                                                                │
# │ For reproducibility                                                        │
# └───────────────────────────────────────────────────────────────────────────┘
random_seed: 42                  # Seed for reproducible results
                                 # Used in numpy random operations
                                 # Standard ML seed (42 from Hitchhiker's Guide)
```

---

## H3 Configuration (diy_h3.yaml)

### Complete Configuration with Annotations

```yaml
# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║                     DIY DATASET CONFIGURATION (H3)                         ║
# ║ File: config/preprocessing/diy_h3.yaml                                     ║
# ║ Uses Uber H3 hexagonal grid instead of DBSCAN clustering                   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

dataset:
  name: "diy"                    # Same dataset name (different clustering)
  
  h3_resolution: 8               # H3 hexagonal grid resolution (0-15)
                                 # 
                                 # INSTEAD of epsilon, we use h3_resolution
                                 # 
                                 # H3 Resolution Reference:
                                 # ┌────────────┬──────────────┬──────────────┐
                                 # │ Resolution │ Edge (m)     │ Area (km²)   │
                                 # ├────────────┼──────────────┼──────────────┤
                                 # │ 6          │ 3,229        │ 36.129       │
                                 # │ 7          │ 1,220        │ 5.161        │
                                 # │ 8          │ 461 ◄────────│ 0.737        │ DEFAULT
                                 # │ 9          │ 174          │ 0.105        │
                                 # │ 10         │ 66           │ 0.015        │
                                 # └────────────┴──────────────┴──────────────┘
                                 #
                                 # Resolution 8 (~461m edge) is comparable to
                                 # DBSCAN epsilon=50m in terms of location granularity
                                 
  previous_day: [7]              # Same as DBSCAN config

preprocessing:
  # H3-specific location parameters
  location:
    num_samples: 2               # Minimum staypoints per H3 cell
                                 # Cells with < 2 staypoints are filtered
                                 # (similar to DBSCAN noise filtering)
                                 #
                                 # Note: distance_metric and agg_level not needed
                                 # H3 uses fixed hexagonal grid
  
  # Same merging, quality, split parameters as DBSCAN
  staypoint_merging:
    max_time_gap: "1min"
  
  quality_filter:
    day_filter: 60
    window_size: 10
    min_thres: 0.6
    mean_thres: 0.7
  
  max_duration: 2880
  min_sequence_length: 3
  
  split:
    train: 0.8
    val: 0.1
    test: 0.1

random_seed: 42
```

### H3 vs DBSCAN Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      DBSCAN vs H3 LOCATION CLUSTERING                        │
└─────────────────────────────────────────────────────────────────────────────┘

                    DBSCAN (epsilon=50m)          H3 (resolution=8)
                    ════════════════════          ═════════════════
                    
Shape:              Irregular (adaptive)          Hexagonal (fixed)
                    
                         •  •                          ____
                      •   •   •                       /    \
                       •  •  •                       /  •   \
                          •                        /  •  •   \
                                                   \   •    /
                                                    \______/

Boundary:           Based on point density        Fixed grid cells
                    
Reproducibility:    May vary slightly             100% reproducible
                    
Computation:        O(n²) worst case              O(n) linear
                    
Best for:           Natural clusters              Uniform coverage
                    Variable density              Large-scale analysis
```

---

## Parameter Deep Dive

### 1. Epsilon (DBSCAN)

```
EPSILON PARAMETER VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

epsilon = 30m (Small)                    epsilon = 100m (Large)
─────────────────────                    ──────────────────────

     •  •                                      •  •  •  •
    •    •   → 2 clusters                     •  •  •  •  •  → 1 cluster
   •      •                                    •  •  •  •

More locations, finer granularity        Fewer locations, coarser granularity
Home, Office, Kitchen at home            Home = entire house
                                         Office = entire floor


Recommended values by use case:
┌─────────────────────────────────┬──────────┬────────────────────────────────┐
│ Use Case                        │ Epsilon  │ Reasoning                      │
├─────────────────────────────────┼──────────┼────────────────────────────────┤
│ Indoor tracking                 │ 10-30m   │ Room-level precision           │
│ Urban mobility (default)        │ 50m      │ Building-level precision       │
│ Regional analysis               │ 100-200m │ Block-level precision          │
│ City-wide patterns              │ 500m+    │ Neighborhood-level             │
└─────────────────────────────────┴──────────┴────────────────────────────────┘
```

### 2. H3 Resolution

```
H3 RESOLUTION VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Resolution 6 (Coarse)           Resolution 8 (Default)         Resolution 10 (Fine)
─────────────────────           ──────────────────────         ────────────────────

    __________                       _____                          ___
   /          \                     /     \                        /   \
  /            \                   /       \                      /     \
 /              \                 /         \                    /       \
 \              /                 \         /                    \       /
  \            /                   \       /                      \     /
   \__________/                     \_____/                        \___/

Edge: ~3.2 km                    Edge: ~461 m                   Edge: ~66 m
City district                    Building/block                 Room-level


Coverage example (Jakarta area):
┌────────────┬───────────────┬────────────────────────────────────────────────┐
│ Resolution │ Cells needed  │ Description                                    │
├────────────┼───────────────┼────────────────────────────────────────────────┤
│ 6          │ ~50           │ Entire city covered by few large hexagons      │
│ 8          │ ~5,000        │ Each neighborhood has multiple cells           │
│ 10         │ ~500,000      │ Individual buildings distinguishable           │
└────────────┴───────────────┴────────────────────────────────────────────────┘
```

### 3. Previous Day

```
PREVIOUS_DAY PARAMETER VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

previous_day = 7 (Default)
─────────────────────────────────────────────────────────────────────────────────

User's staypoint timeline:
Day: 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼
    │   │   │   │◄─── 7 days history ──▶│ T │
    │   │   │   │                        │   │
                 Valid sequence: Day 3-9 predicts Day 10


previous_day = 14 (Longer)
─────────────────────────────────────────────────────────────────────────────────

Day: 0   1   2   3   4   5   6   7   8   9   10  11  12  13  14
    ├───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼───┼
    │◄────────────── 14 days history ─────────────────▶│ T │
    
    More context, but:
    - Fewer valid sequences (need 14+ days of history)
    - More memory required
    - May include less relevant old patterns


Trade-offs:
┌─────────────┬─────────────────────┬────────────────────────────────────────┐
│ previous_day│ Pros                │ Cons                                   │
├─────────────┼─────────────────────┼────────────────────────────────────────┤
│ 3           │ More sequences      │ Limited context                        │
│ 7 (default) │ Balanced            │ Standard weekly patterns               │
│ 14          │ Bi-weekly patterns  │ Fewer sequences, more memory           │
│ 28          │ Monthly patterns    │ Much fewer sequences                   │
└─────────────┴─────────────────────┴────────────────────────────────────────┘
```

### 4. Split Ratios

```
SPLIT RATIOS VISUALIZATION
═══════════════════════════════════════════════════════════════════════════════

Default: train=0.8, val=0.1, test=0.1

User with 100 days of tracking:

Day 0                                Day 80    Day 90              Day 100
├──────────────────────────────────────┼─────────┼────────────────────┤
│◄────────── TRAIN (80%) ─────────────▶│◄─VAL ──▶│◄──── TEST ────────▶│
│            80 days                   │ 10 days │      10 days       │


Why temporal split (not random)?
─────────────────────────────────────────────────────────────────────────────────

Random Split (BAD):              Temporal Split (GOOD):
═══════════════════              ═════════════════════

Day 0  Day 50  Day 100           Day 0  Day 80  Day 100
├──────┼───────┤                 ├───────┼───────┤
│T│V│T│T│V│T│T│T│                │ TRAIN │V│TEST │
 ↑   ↑                                   │
 │   └─ Test sample                      └─ Clear boundary
 └─ Train sample                 
                                 
PROBLEM: Training on Day 60      No future information leaks
to predict Day 50 (future        into training
leaks into past!)                
```

---

## Configuration Examples

### Example 1: Fine-Grained Indoor Analysis

```yaml
# config/preprocessing/diy_indoor.yaml
dataset:
  name: "diy"
  epsilon: 20  # Smaller epsilon for room-level locations
  previous_day: [3]  # Shorter history for frequent check-ins

preprocessing:
  location:
    num_samples: 3  # Require more points to reduce noise
    distance_metric: "haversine"
    agg_level: "user"  # Per-user locations (personal spaces)
  
  staypoint_merging:
    max_time_gap: "30s"  # Shorter gap for indoor movement
  
  max_duration: 1440  # 1 day max
  min_sequence_length: 5  # Need more context for fine-grained
  
  split:
    train: 0.7
    val: 0.15
    test: 0.15

random_seed: 42
```

### Example 2: City-Wide Mobility Study

```yaml
# config/preprocessing/diy_citywide.yaml
dataset:
  name: "diy"
  h3_resolution: 7  # Coarser grid for city-level
  previous_day: [14, 28]  # Longer history for urban patterns

preprocessing:
  location:
    num_samples: 5  # More samples for significant locations
  
  staypoint_merging:
    max_time_gap: "5min"  # Allow larger gaps
  
  max_duration: 4320  # 3 days max (covers weekends)
  min_sequence_length: 10  # Longer sequences
  
  split:
    train: 0.75
    val: 0.1
    test: 0.15  # Larger test set for evaluation

random_seed: 123
```

### Example 3: Multi-Resolution Comparison

```yaml
# config/preprocessing/diy_multiresolution.yaml
# Run pipeline multiple times with different configs

# Config A: Fine resolution
dataset:
  name: "diy"
  epsilon: 30
  previous_day: [7]

# Config B: Medium resolution (default)
dataset:
  name: "diy"
  epsilon: 50
  previous_day: [7]

# Config C: Coarse resolution
dataset:
  name: "diy"
  epsilon: 100
  previous_day: [7]
```

---

## Parameter Tuning Guide

### Decision Tree for Parameter Selection

```
START: What is your analysis goal?
│
├─▶ Indoor/Building-level analysis
│   ├── epsilon: 10-30m OR h3_resolution: 10-11
│   ├── num_samples: 2-3
│   └── previous_day: 1-3 days
│
├─▶ Urban mobility (DEFAULT)
│   ├── epsilon: 50m OR h3_resolution: 8
│   ├── num_samples: 2
│   └── previous_day: 7 days
│
├─▶ Regional patterns
│   ├── epsilon: 100-200m OR h3_resolution: 7
│   ├── num_samples: 3-5
│   └── previous_day: 14-28 days
│
└─▶ City-wide analysis
    ├── epsilon: 500m+ OR h3_resolution: 6
    ├── num_samples: 5-10
    └── previous_day: 28+ days


TRADE-OFF MATRIX:
═══════════════════════════════════════════════════════════════════════════════

                    Small ε / High H3 Res    Large ε / Low H3 Res
                    ─────────────────────    ────────────────────
Locations           More (finer)             Fewer (coarser)
Sequences           More (per location)      Fewer (per location)
Patterns            Local/specific           Global/general
Noise               Higher risk              Lower risk
Memory              Higher                   Lower
Training time       Longer                   Shorter
```

### Recommended Starting Configuration

For most next location prediction tasks, start with the default configuration:

```yaml
# Recommended default configuration
dataset:
  name: "diy"
  epsilon: 50  # OR h3_resolution: 8
  previous_day: [7]

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
  
  max_duration: 2880
  min_sequence_length: 3
  
  split:
    train: 0.8
    val: 0.1
    test: 0.1

random_seed: 42
```

---

## Summary

This guide covered:

1. **Configuration structure**: YAML files with dataset, preprocessing, and seed sections
2. **DBSCAN parameters**: epsilon, num_samples, distance_metric
3. **H3 parameters**: h3_resolution, num_samples
4. **Common parameters**: previous_day, split ratios, max_duration
5. **Parameter tuning**: Decision tree and trade-off matrix

Key takeaways:
- Use `epsilon=50m` or `h3_resolution=8` for urban mobility
- Use `previous_day=7` for weekly patterns
- Always use temporal split to prevent data leakage
- Document your configuration for reproducibility
