# Raw to Interim Script Documentation (H3 Version)

## 📋 Table of Contents
1. [Overview](#overview)
2. [H3 vs DBSCAN Comparison](#h3-vs-dbscan-comparison)
3. [Script Architecture](#script-architecture)
4. [H3 Location Generation Deep Dive](#h3-location-generation-deep-dive)
5. [Line-by-Line Code Walkthrough](#line-by-line-code-walkthrough)
6. [H3 Resolution Selection Guide](#h3-resolution-selection-guide)
7. [Complete Example](#complete-example)

---

## Overview

**Script**: `preprocessing/diy_h3_1_raw_to_interim.py`  
**Purpose**: Transform raw staypoint data into interim dataset using H3 hexagonal grid for location assignment  
**Location Method**: Uber H3 Hexagonal Hierarchical Spatial Index

### Key Difference from DBSCAN Version

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    H3 vs DBSCAN LOCATION ASSIGNMENT                          │
└─────────────────────────────────────────────────────────────────────────────┘

DBSCAN (diy_1_raw_to_interim.py):
─────────────────────────────────────────────────────────────────────────────────
• Clusters points based on density
• Variable, irregular cluster shapes
• Epsilon parameter controls cluster size
• Adaptive to data distribution
• Computationally expensive O(n²)

         •  •  •
        •  ★  •  •    ← Irregular cluster shape
         •  •  •       follows point distribution
            •


H3 (diy_h3_1_raw_to_interim.py):
─────────────────────────────────────────────────────────────────────────────────
• Assigns points to fixed hexagonal grid cells
• Uniform, regular hexagonal shapes
• Resolution parameter controls cell size
• Fixed spatial grid (consistent globally)
• Computationally efficient O(n)

         _____
        /     \
       /   •   \      ← Fixed hexagonal cell
      /  • ★ •  \      regardless of point distribution
      \   •    /
       \_____/
```

---

## H3 vs DBSCAN Comparison

### When to Use Each Method

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         METHOD SELECTION GUIDE                               │
└─────────────────────────────────────────────────────────────────────────────┘

USE DBSCAN WHEN:
═══════════════════════════════════════════════════════════════════════════════
✓ Location boundaries should follow natural data patterns
✓ Cluster density varies significantly across space
✓ You need adaptive cluster sizes
✓ Data is relatively small (DBSCAN is O(n²))
✓ You're studying natural human mobility patterns

Example use cases:
• Identifying meaningful places (home, work, favorites)
• Discovering activity hotspots
• Small-scale urban mobility analysis


USE H3 WHEN:
═══════════════════════════════════════════════════════════════════════════════
✓ You need reproducible, consistent location IDs
✓ Same physical location must always map to same cell
✓ Data is large (H3 is O(n))
✓ You need hierarchical spatial analysis
✓ You want to compare across different datasets
✓ Integration with other H3-based systems

Example use cases:
• Large-scale mobility studies
• Cross-dataset comparisons
• Production systems requiring consistency
• Integration with mapping/logistics systems


COMPARISON TABLE:
═══════════════════════════════════════════════════════════════════════════════

┌───────────────────────┬─────────────────────────┬─────────────────────────────┐
│ Aspect                │ DBSCAN                  │ H3                          │
├───────────────────────┼─────────────────────────┼─────────────────────────────┤
│ Cluster Shape         │ Irregular (data-driven)│ Regular hexagons            │
│ Reproducibility       │ May vary slightly      │ 100% reproducible           │
│ Computation           │ O(n²) or O(n log n)    │ O(n) linear                 │
│ Parameter             │ epsilon (meters)       │ resolution (0-15)           │
│ Boundary Handling     │ Natural boundaries     │ Arbitrary grid boundaries   │
│ Empty Regions         │ No clusters            │ Empty cells (filtered out)  │
│ Cross-dataset Compare │ Difficult              │ Easy (same grid globally)   │
│ Hierarchical          │ No                     │ Yes (zoom in/out)           │
└───────────────────────┴─────────────────────────┴─────────────────────────────┘
```

---

## Script Architecture

### Structural Differences from DBSCAN Version

```
┌─────────────────────────────────────────────────────────────────────────────┐
│         STRUCTURAL COMPARISON: DBSCAN vs H3 VERSION                          │
└─────────────────────────────────────────────────────────────────────────────┘

SHARED COMPONENTS (identical logic):
─────────────────────────────────────────────────────────────────────────────────
• load_raw_data()           - Same loading and filtering
• merge_staypoints()        - Same merging logic
• process_temporal_features() - Same temporal enrichment
• enrich_time_info()        - Same time extraction
• _get_time()               - Same per-user time calculation
• main()                    - Same orchestration structure


KEY DIFFERENCE - Location Generation:
─────────────────────────────────────────────────────────────────────────────────

DBSCAN Version:                          H3 Version:
─────────────────────                    ─────────────────────
def generate_locations(...):             def generate_h3_locations(...):
    sp.as_staypoints.generate_locations(     # Uses H3 library directly
        epsilon=50,                          h3.latlng_to_cell(lat, lon, res)
        num_samples=2,                   
        distance_metric="haversine"      
    )                                    


Configuration Difference:
─────────────────────────────────────────────────────────────────────────────────

diy.yaml:                                diy_h3.yaml:
─────────────────                        ─────────────────
dataset:                                 dataset:
  epsilon: 50                              h3_resolution: 8
                                         
preprocessing:                           preprocessing:
  location:                                location:
    num_samples: 2                           num_samples: 2  # (same)
    distance_metric: "haversine"           # (not needed for H3)
    agg_level: "dataset"                   # (not needed for H3)
```

---

## H3 Location Generation Deep Dive

### Function: `generate_h3_locations()` (Lines 121-209)

```python
def generate_h3_locations(sp, config, interim_dir, h3_resolution):
    """
    Generate locations from staypoints using H3 hexagonal grid.
    
    This function replaces DBSCAN-based generate_locations with H3-based 
    location assignment. Each staypoint is assigned to an H3 cell based 
    on its coordinates.
    
    Parameters:
    -----------
    sp : GeoDataFrame
        Staypoints with geometry column
    config : dict
        Configuration dictionary
    interim_dir : str
        Output directory path
    h3_resolution : int
        H3 resolution level (0-15)
    
    Returns:
    --------
    sp : GeoDataFrame
        Staypoints with location_id column added
    locs : DataFrame
        Location information with H3 cell details
    """
    print("\n" + "="*70)
    print("STAGE 2: Generating Locations using H3")
    print("="*70)
    
    # Get minimum samples parameter (same concept as DBSCAN)
    loc_params = config['preprocessing']['location']
    num_samples = loc_params['num_samples']
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1: Extract coordinates from geometry
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[1/3] Assigning staypoints to H3 cells (resolution={h3_resolution})...")
    
    sp = sp.copy()
    
    # Get latitude/longitude from geometry column
    if hasattr(sp, 'geom'):
        sp['lat'] = sp['geom'].y    # Latitude (Y coordinate)
        sp['lon'] = sp['geom'].x    # Longitude (X coordinate)
    elif hasattr(sp, 'geometry'):
        sp['lat'] = sp['geometry'].y
        sp['lon'] = sp['geometry'].x
    else:
        raise ValueError("No geometry column found in staypoints")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2: Assign H3 cell to each staypoint
    # ─────────────────────────────────────────────────────────────────────────
    # H3 cell assignment is O(1) per point!
    sp['h3_cell'] = sp.apply(
        lambda row: h3.latlng_to_cell(row['lat'], row['lon'], h3_resolution), 
        axis=1
    )
    # h3_cell is a hexadecimal string like "88680c4801fffff"
    
    print(f"  Assigned {len(sp):,} staypoints to {sp['h3_cell'].nunique():,} unique H3 cells")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3: Filter cells with minimum samples (like DBSCAN noise filtering)
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[2/3] Filtering locations with min {num_samples} staypoints...")
    
    # Count staypoints per H3 cell
    cell_counts = sp['h3_cell'].value_counts()
    
    # Keep only cells with at least num_samples staypoints
    valid_cells = cell_counts[cell_counts >= num_samples].index
    
    # Mark invalid cells as NaN (similar to DBSCAN noise points)
    sp.loc[~sp['h3_cell'].isin(valid_cells), 'h3_cell'] = None
    
    # Filter out noise staypoints
    sp = sp.loc[~sp["h3_cell"].isna()].copy()
    print(f"  After filtering non-location staypoints: {len(sp):,}")
    print(f"  Valid H3 cells (locations): {sp['h3_cell'].nunique():,}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 4: Create integer location_id from H3 cells
    # ─────────────────────────────────────────────────────────────────────────
    print(f"\n[3/3] Creating location IDs...")
    
    unique_cells = sp['h3_cell'].unique()
    cell_to_id = {cell: idx for idx, cell in enumerate(unique_cells)}
    sp['location_id'] = sp['h3_cell'].map(cell_to_id)
    # Convert H3 cell string to integer ID for model compatibility
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 5: Create locations DataFrame with H3 metadata
    # ─────────────────────────────────────────────────────────────────────────
    print("  Creating locations dataframe...")
    
    locs_data = []
    for cell, loc_id in cell_to_id.items():
        # Get cell center coordinates
        lat, lng = h3.cell_to_latlng(cell)
        
        locs_data.append({
            'location_id': loc_id,
            'h3_cell': cell,           # Original H3 cell index
            'center_lat': lat,          # Cell center latitude
            'center_lng': lng,          # Cell center longitude
            'h3_resolution': h3_resolution
        })
    
    locs = pd.DataFrame(locs_data)
    locs = locs.set_index('location_id')
    
    # ─────────────────────────────────────────────────────────────────────────
    # STEP 6: Save locations file
    # ─────────────────────────────────────────────────────────────────────────
    locations_file = os.path.join(interim_dir, f"locations_h3r{h3_resolution}.csv")
    locs.to_csv(locations_file)
    print(f"  Saved {len(locs):,} unique locations to: {locations_file}")
    
    # Clean up temporary columns
    sp = sp.drop(columns=['lat', 'lon', 'h3_cell'], errors='ignore')
    
    return sp, locs
```

### H3 Cell Assignment Visualization

```
H3 CELL ASSIGNMENT PROCESS
═══════════════════════════════════════════════════════════════════════════════

Step 1: Extract Coordinates
─────────────────────────────────────────────────────────────────────────────────

Staypoint geometry: POINT(110.4315 -7.7478)
                           │       │
                           │       └── Latitude: -7.7478 (7.7478°S)
                           └── Longitude: 110.4315 (110.4315°E)


Step 2: H3 Cell Assignment
─────────────────────────────────────────────────────────────────────────────────

h3.latlng_to_cell(-7.7478, 110.4315, 8)
                    │        │      │
                    │        │      └── Resolution 8 (~461m edge)
                    │        └── Longitude
                    └── Latitude

Returns: "88680c4801fffff"  (H3 cell index as hexadecimal string)


H3 Cell Visualization (Resolution 8):
─────────────────────────────────────────────────────────────────────────────────

                    N
                    ↑
              _____/  \_____
             /               \
            /                 \
      ←W   |    •  Staypoint   |   E→
            \      (center)   /
             \_____    _____/
                   \  /
                    S

Cell "88680c4801fffff":
• Edge length: ~461 meters
• Area: ~0.737 km²
• Center: (-7.7475, 110.4318)


Step 3: Filter by num_samples
─────────────────────────────────────────────────────────────────────────────────

Cell counts after assignment:
┌───────────────────┬─────────────┐
│ H3 Cell           │ Staypoints  │
├───────────────────┼─────────────┤
│ 88680c4801fffff   │ 45          │ ← Keep (45 >= 2)
│ 88680c4803fffff   │ 12          │ ← Keep (12 >= 2)
│ 88680c4805fffff   │ 1           │ ← Remove (1 < 2) - NOISE
│ 88680c4807fffff   │ 8           │ ← Keep (8 >= 2)
│ 88680c4809fffff   │ 1           │ ← Remove (1 < 2) - NOISE
└───────────────────┴─────────────┘

Just like DBSCAN's num_samples parameter, cells with too few staypoints
are considered "noise" and filtered out.


Step 4: Create Integer location_id
─────────────────────────────────────────────────────────────────────────────────

H3 cell strings → Integer IDs (for model embedding layers):

┌───────────────────┬─────────────┐
│ H3 Cell           │ location_id │
├───────────────────┼─────────────┤
│ 88680c4801fffff   │ 0           │
│ 88680c4803fffff   │ 1           │
│ 88680c4807fffff   │ 2           │
└───────────────────┴─────────────┘


Step 5: Create Locations DataFrame
─────────────────────────────────────────────────────────────────────────────────

locations_h3r8.csv:
┌─────────────┬───────────────────┬────────────┬────────────┬───────────────┐
│ location_id │ h3_cell           │ center_lat │ center_lng │ h3_resolution │
├─────────────┼───────────────────┼────────────┼────────────┼───────────────┤
│ 0           │ 88680c4801fffff   │ -7.7475    │ 110.4318   │ 8             │
│ 1           │ 88680c4803fffff   │ -7.7117    │ 110.3860   │ 8             │
│ 2           │ 88680c4807fffff   │ -7.7640    │ 110.3905   │ 8             │
└─────────────┴───────────────────┴────────────┴────────────┴───────────────┘
```

---

## H3 Resolution Selection Guide

### Resolution Reference Table

```
H3 RESOLUTION REFERENCE
═══════════════════════════════════════════════════════════════════════════════

┌────────────┬─────────────────┬────────────────┬───────────────────────────────┐
│ Resolution │ Edge Length     │ Area           │ Use Case                      │
├────────────┼─────────────────┼────────────────┼───────────────────────────────┤
│ 0          │ 1,107 km        │ 4,250,547 km²  │ Global/continental            │
│ 1          │ 418 km          │ 607,221 km²    │ Large country                 │
│ 2          │ 158 km          │ 86,746 km²     │ Region/state                  │
│ 3          │ 60 km           │ 12,393 km²     │ Metropolitan area             │
│ 4          │ 22.6 km         │ 1,770 km²      │ Large city                    │
│ 5          │ 8.5 km          │ 252.9 km²      │ City district                 │
│ 6          │ 3.2 km          │ 36.13 km²      │ Neighborhood                  │
│ 7          │ 1.2 km          │ 5.16 km²       │ City block                    │
│ 8          │ 461 m           │ 0.737 km²      │ Building/POI ◄─── DEFAULT     │
│ 9          │ 174 m           │ 0.105 km²      │ Individual building           │
│ 10         │ 66 m            │ 0.015 km²      │ Building entrance             │
│ 11         │ 25 m            │ 0.002 km²      │ Room-level                    │
│ 12         │ 9.4 m           │ 0.0003 km²     │ Precise indoor                │
│ 13         │ 3.6 m           │ 0.00004 km²    │ Sub-room                      │
│ 14         │ 1.3 m           │ 0.000006 km²   │ Device-level                  │
│ 15         │ 0.5 m           │ 0.0000009 km²  │ Centimeter precision          │
└────────────┴─────────────────┴────────────────┴───────────────────────────────┘


VISUALIZATION BY RESOLUTION:
═══════════════════════════════════════════════════════════════════════════════

Resolution 6 (Neighborhood):         Resolution 8 (Building):
    __________                            _____
   /          \                          /     \
  /            \  ~3.2 km               /       \  ~461 m
 /              \                      /         \
 \              /                      \         /
  \            /                        \       /
   \__________/                          \_____/


Resolution 10 (Entrance):            Resolution 12 (Room):
      ___                                 _
     /   \  ~66 m                        / \  ~9 m
    /     \                             /   \
    \     /                             \   /
     \___/                               \_/


WHY RESOLUTION 8 IS DEFAULT:
═══════════════════════════════════════════════════════════════════════════════

Resolution 8 (~461m edge, ~0.737 km² area):

✓ Similar scale to DBSCAN epsilon=50m results
  (A cluster with 50m radius has ~314m diameter, 
   H3 cell with 461m edge covers similar area)

✓ Captures building/venue-level locations
  • A home with front/back yard fits in one cell
  • A small mall fits in one cell
  • Office buildings have 1-4 cells typically

✓ Balances location granularity and data sparsity
  • Fine enough to distinguish meaningful places
  • Coarse enough to have sufficient data per cell

✓ Common choice in mobility research literature
```

### Resolution Selection Decision Tree

```
RESOLUTION SELECTION DECISION TREE
═══════════════════════════════════════════════════════════════════════════════

Q1: What is your analysis scale?
│
├─▶ City-wide / Regional analysis
│   │
│   └─▶ Q2: Need neighborhood-level or district-level?
│       ├─▶ Neighborhood → Resolution 6-7
│       └─▶ District → Resolution 5-6
│
├─▶ Urban mobility (most common)
│   │
│   └─▶ Q2: Indoor or outdoor focus?
│       ├─▶ Outdoor (street-level) → Resolution 8 (DEFAULT)
│       └─▶ Indoor (room-level) → Resolution 10-11
│
└─▶ Fine-grained / Indoor
    │
    └─▶ Q2: Building or room level?
        ├─▶ Building level → Resolution 9
        └─▶ Room level → Resolution 11-12


COMPARING DBSCAN EPSILON TO H3 RESOLUTION:
═══════════════════════════════════════════════════════════════════════════════

The relationship is approximate since DBSCAN creates irregular clusters:

┌─────────────────┬───────────────────────────────────────────────────────────┐
│ DBSCAN epsilon  │ Approximate H3 Resolution                                 │
├─────────────────┼───────────────────────────────────────────────────────────┤
│ 30-50m          │ Resolution 9 (174m edge)                                  │
│ 50-100m         │ Resolution 8 (461m edge) ◄─── BOTH DEFAULTS              │
│ 100-200m        │ Resolution 7 (1.2km edge)                                 │
│ 200-500m        │ Resolution 6 (3.2km edge)                                 │
└─────────────────┴───────────────────────────────────────────────────────────┘

Note: This is a rough equivalence. The actual number of resulting locations
may vary significantly between the two methods.
```

---

## Line-by-Line Code Walkthrough

### Key Differences from DBSCAN Version

```python
# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS - H3 library added
# ═══════════════════════════════════════════════════════════════════════════════

import trackintel as ti
import h3                    # ← NEW: Uber H3 library for hexagonal grids

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION - h3_resolution instead of epsilon
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
# epsilon = config['dataset']['epsilon']  # 50 meters

# H3 version:
h3_resolution = config['dataset']['h3_resolution']  # 8

# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT NAMING - h3r{resolution} instead of eps{epsilon}
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
# output_folder = f"{dataset_name}_eps{epsilon}"  # "diy_eps50"

# H3 version:
output_folder = f"{dataset_name}_h3r{h3_resolution}"  # "diy_h3r8"

# ═══════════════════════════════════════════════════════════════════════════════
# LOCATION GENERATION - H3 instead of DBSCAN
# ═══════════════════════════════════════════════════════════════════════════════

# DBSCAN version:
# sp, locs = sp.as_staypoints.generate_locations(
#     epsilon=epsilon,
#     num_samples=loc_params['num_samples'],
#     distance_metric=loc_params['distance_metric'],
#     agg_level=loc_params['agg_level'],
#     n_jobs=-1
# )

# H3 version:
sp, locs = generate_h3_locations(sp, config, interim_dir, h3_resolution)
```

### Main Function Differences

```python
def main():
    # ─────────────────────────────────────────────────────────────────────────
    # Parse arguments (same structure, different default config)
    # ─────────────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="DIY Dataset Preprocessing - H3 Version - Script 1: Raw to Interim"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocessing/diy_h3.yaml",  # ← Different default
        help="Path to dataset configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    np.random.seed(config.get('random_seed', RANDOM_SEED))
    
    # ─────────────────────────────────────────────────────────────────────────
    # Get H3 resolution instead of epsilon
    # ─────────────────────────────────────────────────────────────────────────
    dataset_name = config['dataset']['name']
    h3_resolution = config['dataset']['h3_resolution']  # ← Key difference
    
    # ─────────────────────────────────────────────────────────────────────────
    # Create output directories with H3 naming
    # ─────────────────────────────────────────────────────────────────────────
    output_folder = f"{dataset_name}_h3r{h3_resolution}"  # "diy_h3r8"
    interim_dir = os.path.join("data", output_folder, "interim")
    os.makedirs(interim_dir, exist_ok=True)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Execute pipeline (same stages, different location generation)
    # ─────────────────────────────────────────────────────────────────────────
    sp, valid_users = load_raw_data(config)  # Same
    
    # Save valid users
    valid_users_file = os.path.join(interim_dir, f"valid_users_h3r{h3_resolution}.csv")
    pd.DataFrame({"user_id": valid_users}).to_csv(valid_users_file, index=False)
    
    # H3 location generation instead of DBSCAN
    sp, locs = generate_h3_locations(sp, config, interim_dir, h3_resolution)
    
    sp_merged = merge_staypoints(sp, config, interim_dir, h3_resolution)  # Same
    sp_time = process_temporal_features(sp_merged, config, interim_dir, h3_resolution)  # Same
```

---

## Complete Example

### Running the H3 Script

```bash
# Default H3 configuration (resolution=8)
python preprocessing/diy_h3_1_raw_to_interim.py --config config/preprocessing/diy_h3.yaml

# Custom resolution
# First, create config/preprocessing/diy_h3_r9.yaml with h3_resolution: 9
python preprocessing/diy_h3_1_raw_to_interim.py --config config/preprocessing/diy_h3_r9.yaml
```

### Example Console Output

```
================================================================================
DIY PREPROCESSING (H3) - Script 1: Raw to Interim
================================================================================
[INPUT]  Raw data: data/raw_diy
[OUTPUT] Interim folder: data/diy_h3r8/interim
[CONFIG] Dataset: diy, H3 Resolution: 8
[CONFIG] Random seed: 42
================================================================================

======================================================================
STAGE 1: Loading Raw Data
======================================================================

[1/2] Reading preprocessed staypoints from data/raw_diy...
  Loaded 523,456 staypoints

[2/2] Reading valid users...
  Loaded 155 valid users
  Valid users after quality filter: 155
  Activity staypoints: 312,789
  Saved valid users to: data/diy_h3r8/interim/valid_users_h3r8.csv

======================================================================
STAGE 2: Generating Locations using H3
======================================================================

[1/3] Assigning staypoints to H3 cells (resolution=8)...
  Assigned 312,789 staypoints to 8,234 unique H3 cells

[2/3] Filtering locations with min 2 staypoints...
  After filtering non-location staypoints: 305,432
  Valid H3 cells (locations): 5,678

[3/3] Creating location IDs...
  Creating locations dataframe...
  Saved 5,678 unique locations to: data/diy_h3r8/interim/locations_h3r8.csv

======================================================================
STAGE 3: Merging Staypoints
======================================================================

[1/1] Merging consecutive staypoints (max gap: 1min)...
  After merging: 290,123 staypoints
  Saved merged staypoints to: data/diy_h3r8/interim/staypoints_merged_h3r8.csv

======================================================================
STAGE 4: Enriching Temporal Features
======================================================================

[1/1] Extracting temporal features (day, time, weekday)...
  Users with temporal features: 155
  Saved interim data to: data/diy_h3r8/interim/intermediate_h3r8.csv
  Saved interim statistics to: data/diy_h3r8/interim/interim_stats_h3r8.json

================================================================================
SCRIPT 1 COMPLETE: Raw to Interim (H3)
================================================================================
Output folder: data/diy_h3r8/interim
Main output: data/diy_h3r8/interim/intermediate_h3r8.csv
================================================================================
```

### Output Directory Structure

```
data/
├── diy_eps50/             # DBSCAN version output
│   └── interim/
│       ├── intermediate_eps50.csv
│       ├── locations_eps50.csv
│       └── ...
│
└── diy_h3r8/              # H3 version output
    └── interim/
        ├── intermediate_h3r8.csv
        ├── locations_h3r8.csv        # Contains H3 cell info
        ├── staypoints_merged_h3r8.csv
        ├── valid_users_h3r8.csv
        └── interim_stats_h3r8.json
```

### Comparing Outputs

```
DBSCAN vs H3 OUTPUT COMPARISON
═══════════════════════════════════════════════════════════════════════════════

DBSCAN (epsilon=50):
• Locations: ~4,521
• Location shapes: Irregular (data-driven)
• locations_eps50.csv contains: center, extent (polygon)

H3 (resolution=8):
• Locations: ~5,678 (typically more)
• Location shapes: Regular hexagons
• locations_h3r8.csv contains: h3_cell, center_lat, center_lng

Why H3 typically produces more locations?
• H3 assigns ALL points to grid cells
• DBSCAN may merge nearby clusters
• H3 boundaries don't adapt to data distribution


Intermediate CSV Comparison:
─────────────────────────────────────────────────────────────────────────────────

Both produce same columns:
• id, user_id, location_id, start_day, end_day, start_min, end_min, weekday, duration

The only difference:
• location_id values reference different location systems
• But the meaning is the same: "which location was visited"
```

---

## Summary

The `diy_h3_1_raw_to_interim.py` script:

1. **Uses H3** hexagonal grid instead of DBSCAN for location assignment
2. **Produces consistent** location IDs (same point → same cell always)
3. **Runs faster** than DBSCAN (O(n) vs O(n²))
4. **Creates same output format** as DBSCAN version for downstream compatibility

Key parameters:
- `h3_resolution`: Grid resolution (default: 8, ~461m edge)
- `num_samples`: Minimum staypoints per cell (default: 2)

Output: `intermediate_h3r{resolution}.csv` for Script 2 (H3 version).

When to use H3 over DBSCAN:
- Large datasets (faster processing)
- Need reproducible location IDs
- Cross-dataset comparisons
- Integration with H3-based systems
