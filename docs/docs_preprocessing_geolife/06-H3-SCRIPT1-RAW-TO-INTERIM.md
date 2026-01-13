# H3 Script 1: Raw to Interim - Complete Line-by-Line Guide

## Table of Contents
1. [Overview](#overview)
2. [Key Difference: DBSCAN vs H3](#key-difference-dbscan-vs-h3)
3. [New Function: generate_h3_locations](#new-function-generate_h3_locations)
4. [H3 Library Deep Dive](#h3-library-deep-dive)
5. [Complete Code Walkthrough](#complete-code-walkthrough)
6. [H3 Cell Assignment Visualization](#h3-cell-assignment-visualization)

---

## Overview

**Script**: `preprocessing/geolife_h3_1_raw_to_interim.py`

**Purpose**: Same as Script 1, but uses **Uber H3 hexagonal grid** for location assignment instead of DBSCAN clustering.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              H3 SCRIPT 1: RAW TO INTERIM (HEXAGONAL GRID)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  The ONLY difference from standard Script 1:                                     │
│                                                                                  │
│  STANDARD (geolife_1_raw_to_interim.py):                                        │
│  └─► Step 6: generate_locations() using DBSCAN clustering                       │
│                                                                                  │
│  H3 VERSION (geolife_h3_1_raw_to_interim.py):                                   │
│  └─► Step 6: generate_h3_locations() using H3 hexagonal grid                    │
│                                                                                  │
│  All other steps (1-5, 7) are IDENTICAL                                         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Difference: DBSCAN vs H3

### Conceptual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DBSCAN vs H3 LOCATION ASSIGNMENT                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  DBSCAN CLUSTERING                       H3 HEXAGONAL GRID                       │
│  ─────────────────                       ──────────────────                      │
│                                                                                  │
│  Method: Density-based clustering        Method: Grid-based assignment           │
│                                                                                  │
│  Parameter: epsilon (meters)             Parameter: resolution (0-15)            │
│                                                                                  │
│       Staypoints                              Staypoints                         │
│        ●  ●                                    ●  ●                              │
│       ● • ●                                   ● • ●                              │
│        ●  ●                                    ●  ●                              │
│           ↓                                       ↓                              │
│    ┌───────────┐                          ╱╲    ╱╲    ╱╲                        │
│    │ ●  ●      │                         │●● │    │  │                          │
│    │● • ●      │ Location 1             │●●●│    │  │ Location at              │
│    │ ●  ●      │ (cluster)               ╲╱    ╲╱    ╲╱  H3 cell               │
│    └───────────┘                                                                 │
│                                                                                  │
│  Pros:                                   Pros:                                   │
│  • Adapts to data density                • Consistent cell sizes                 │
│  • Natural cluster shapes                • Computationally efficient             │
│  • Semantic grouping                     • Hierarchical (zoom in/out)            │
│                                          • Globally consistent grid              │
│                                                                                  │
│  Cons:                                   Cons:                                   │
│  • Slow for large datasets               • May split natural clusters            │
│  • Parameter sensitive                   • Fixed grid orientation                │
│  • Non-deterministic (order)             • Requires H3 library                   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Output Naming Convention

| Version | Output Folder | Example |
|---------|--------------|---------|
| DBSCAN | `geolife_eps{epsilon}/` | `geolife_eps20/` |
| H3 | `geolife_h3r{resolution}/` | `geolife_h3r8/` |

---

## New Function: generate_h3_locations

**Lines 151-226**

This is the key function that replaces DBSCAN-based `generate_locations()`.

### Function Signature

```python
def generate_h3_locations(sp, h3_resolution, num_samples):
    """
    Generate locations from staypoints using H3 hexagonal grid.
    
    This function replaces the DBSCAN-based generate_locations with H3-based location assignment.
    Each staypoint is assigned to an H3 cell based on its coordinates.
    Locations with fewer than num_samples staypoints are filtered out (similar to DBSCAN noise filtering).
    
    Args:
        sp: GeoDataFrame with staypoints
        h3_resolution: H3 resolution level (0-15)
        num_samples: Minimum samples required in H3 cell to be valid location
    
    Returns:
        sp: GeoDataFrame with location_id column added
        locs: DataFrame with location information
    """
```

**Parameters**:
| Parameter | Type | Description |
|-----------|------|-------------|
| `sp` | GeoDataFrame | Staypoints with geometry (geom column) |
| `h3_resolution` | int | H3 resolution (0-15), typically 8 |
| `num_samples` | int | Minimum staypoints in cell to be valid |

**Returns**: 
- `sp`: GeoDataFrame with `location_id` column added
- `locs`: DataFrame with H3 cell information

### Complete Line-by-Line Walkthrough

#### Lines 168-180: Extract Coordinates

```python
print(f"\n  Assigning staypoints to H3 cells (resolution={h3_resolution})...")

# Extract coordinates from geometry
sp = sp.copy()

# Get lat/lon from geometry
if hasattr(sp, 'geom'):
    sp['lat'] = sp['geom'].y
    sp['lon'] = sp['geom'].x
elif hasattr(sp, 'geometry'):
    sp['lat'] = sp['geometry'].y
    sp['lon'] = sp['geometry'].x
else:
    raise ValueError("No geometry column found in staypoints")
```

**Explanation**:
- GeoDataFrame stores geometry in either `geom` or `geometry` column
- Extract latitude (y) and longitude (x) for H3 conversion
- `.y` and `.x` are Shapely Point accessors

**Example**:
```
Geometry: POINT(116.318417 39.984702)
         └─────────┘ └────────┘
            lon (x)    lat (y)

Extracted:
  lat = 39.984702
  lon = 116.318417
```

#### Lines 182-186: Assign H3 Cells

```python
# Assign H3 cell to each staypoint
sp['h3_cell'] = sp.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lon'], h3_resolution), axis=1)

print(f"  Assigned {len(sp):,} staypoints to {sp['h3_cell'].nunique():,} unique H3 cells")
```

**H3 Cell Assignment**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    H3 CELL ASSIGNMENT                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Function: h3.latlng_to_cell(lat, lon, resolution)                              │
│                                                                                  │
│  Input:                                                                          │
│    lat = 39.984702                                                               │
│    lon = 116.318417                                                              │
│    resolution = 8                                                                │
│                                                                                  │
│  Output:                                                                         │
│    h3_cell = "88283082b9fffff"  (H3 index as hexadecimal string)                │
│                                                                                  │
│  Visualization:                                                                  │
│  ─────────────                                                                   │
│       ╱╲    ╱╲    ╱╲    ╱╲                                                       │
│      │  │  │  │  │●●│  │  │   ← Point falls in this hexagon                     │
│       ╲╱    ╲╱    ╲╱    ╲╱      h3_cell = "88283082b9fffff"                      │
│       ╱╲    ╱╲    ╱╲    ╱╲                                                       │
│      │  │  │●●│  │  │  │  │   ← Another point in different cell                 │
│       ╲╱    ╲╱    ╲╱    ╲╱      h3_cell = "88283082b1fffff"                      │
│                                                                                  │
│  All staypoints get assigned to their containing H3 hexagon                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Lines 188-199: Filter by num_samples

```python
# Filter cells with at least num_samples staypoints (similar to DBSCAN num_samples)
print(f"  Filtering locations with min {num_samples} staypoints...")
cell_counts = sp['h3_cell'].value_counts()
valid_cells = cell_counts[cell_counts >= num_samples].index

# Mark invalid cells as NaN (similar to DBSCAN noise points)
sp.loc[~sp['h3_cell'].isin(valid_cells), 'h3_cell'] = None

# Filter out noise staypoints
sp = sp.loc[~sp["h3_cell"].isna()].copy()
print(f"  After filtering non-location staypoints: {len(sp):,}")
print(f"  Valid H3 cells (locations): {sp['h3_cell'].nunique():,}")
```

**num_samples Filtering**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    NUM_SAMPLES FILTERING                                         │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  num_samples = 2 (from config)                                                   │
│                                                                                  │
│  Cell counts (value_counts):                                                     │
│  ┌───────────────────────────────────────┐                                      │
│  │ H3 Cell           │ Count │ Valid?   │                                      │
│  │ 88283082b9fffff   │   45  │  ✓ (≥2)  │                                      │
│  │ 88283082b1fffff   │   30  │  ✓ (≥2)  │                                      │
│  │ 88283082adfffff   │   12  │  ✓ (≥2)  │                                      │
│  │ 88283082a5fffff   │    1  │  ✗ (<2)  │ ← Filtered out                       │
│  │ 88283082c3fffff   │    1  │  ✗ (<2)  │ ← Filtered out                       │
│  └───────────────────────────────────────┘                                      │
│                                                                                  │
│  Why filter?                                                                     │
│  • Single-visit locations don't help prediction                                  │
│  • Similar to DBSCAN's noise point removal                                       │
│  • Keeps only meaningful, repeated locations                                     │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Lines 201-206: Create Integer Location IDs

```python
# Create integer location_id from H3 cells
print("  Creating location IDs...")
unique_cells = sp['h3_cell'].unique()
cell_to_id = {cell: idx for idx, cell in enumerate(unique_cells)}
sp['location_id'] = sp['h3_cell'].map(cell_to_id)
```

**ID Mapping**:

```
unique_cells = ['88283082b9fffff', '88283082b1fffff', '88283082adfffff', ...]

cell_to_id = {
    '88283082b9fffff': 0,
    '88283082b1fffff': 1,
    '88283082adfffff': 2,
    ...
}

Staypoint with h3_cell = '88283082b9fffff' → location_id = 0
Staypoint with h3_cell = '88283082b1fffff' → location_id = 1
```

#### Lines 208-221: Create Locations DataFrame

```python
# Create locations DataFrame with H3 cell center coordinates
print("  Creating locations dataframe...")
locs_data = []
for cell, loc_id in cell_to_id.items():
    lat, lng = h3.cell_to_latlng(cell)  # Get hexagon center
    locs_data.append({
        'location_id': loc_id,
        'h3_cell': cell,
        'center_lat': lat,
        'center_lng': lng,
        'h3_resolution': h3_resolution
    })

locs = pd.DataFrame(locs_data)
locs = locs.set_index('location_id')
```

**Locations DataFrame**:

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    LOCATIONS DATAFRAME                                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌──────────────────────────────────────────────────────────────────────────┐   │
│  │ location_id │      h3_cell        │ center_lat │ center_lng │ h3_res    │   │
│  │      0      │ 88283082b9fffff     │  39.9847   │  116.3184  │    8      │   │
│  │      1      │ 88283082b1fffff     │  39.9892   │  116.3210  │    8      │   │
│  │      2      │ 88283082adfffff     │  39.9756   │  116.3098  │    8      │   │
│  │     ...     │        ...          │    ...     │    ...     │   ...     │   │
│  └──────────────────────────────────────────────────────────────────────────┘   │
│                                                                                  │
│  h3.cell_to_latlng(cell) returns the center coordinates of the hexagon         │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

#### Lines 223-226: Cleanup and Return

```python
# Clean up temporary columns
sp = sp.drop(columns=['lat', 'lon', 'h3_cell'], errors='ignore')

return sp, locs
```

---

## H3 Library Deep Dive

### What is H3?

H3 is Uber's open-source hierarchical hexagonal grid system. It divides the Earth's surface into hexagonal cells at multiple resolutions.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    H3 HIERARCHICAL GRID SYSTEM                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  H3 divides the Earth into hexagons (and some pentagons at poles)               │
│                                                                                  │
│  Resolution hierarchy:                                                           │
│  ─────────────────────                                                           │
│                                                                                  │
│  Resolution 0:                     Resolution 4:                                 │
│  (Very large hexagons)             (Medium hexagons)                             │
│       ╱╲                                ╱╲ ╱╲ ╱╲                                 │
│      │  │                              │  │  │  │                                │
│       ╲╱                                ╲╱ ╲╱ ╲╱                                 │
│  ~1100 km edge                          ╱╲ ╱╲ ╱╲                                 │
│                                        │  │  │  │                                │
│                                         ╲╱ ╲╱ ╲╱                                 │
│                                        ~22 km edge                               │
│                                                                                  │
│  Resolution 8 (commonly used):     Resolution 12:                                │
│       ╱╲ ╱╲ ╱╲ ╱╲ ╱╲               (Very small hexagons)                        │
│      │  │  │  │  │  │                    ╱╲╱╲╱╲╱╲╱╲╱╲╱╲                         │
│       ╲╱ ╲╱ ╲╱ ╲╱ ╲╱                    ╲╱╲╱╲╱╲╱╲╱╲╱╲╱                         │
│       ╱╲ ╱╲ ╱╲ ╱╲ ╱╲                     ╱╲╱╲╱╲╱╲╱╲╱╲╱╲                         │
│      │  │  │  │  │  │                    ~9.4m edge                              │
│       ╲╱ ╲╱ ╲╱ ╲╱ ╲╱                                                             │
│      ~461m edge                                                                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### H3 Resolution Reference Table

| Resolution | Edge Length | Area | Hexagons on Earth | Use Case |
|------------|-------------|------|-------------------|----------|
| 0 | 1,107.71 km | 4,250,546.85 km² | 122 | Continental |
| 1 | 418.68 km | 607,220.98 km² | 842 | Large region |
| 2 | 158.24 km | 86,745.85 km² | 5,882 | Country |
| 3 | 59.81 km | 12,392.26 km² | 41,162 | State |
| 4 | 22.61 km | 1,770.32 km² | 288,122 | Large city |
| 5 | 8.54 km | 252.90 km² | 2,016,842 | City district |
| 6 | 3.23 km | 36.13 km² | 14,117,882 | Neighborhood |
| 7 | 1.22 km | 5.16 km² | 98,825,162 | Blocks |
| **8** | **461.35 m** | **0.74 km²** | **691,776,122** | **Place (Used)** |
| 9 | 174.38 m | 0.11 km² | 4,842,432,842 | Building |
| 10 | 65.91 m | 0.015 km² | 33,897,029,882 | Small building |
| 11 | 24.91 m | 0.002 km² | 237,279,209,162 | Room |
| 12 | 9.42 m | 0.0003 km² | 1,660,954,464,122 | Precise |
| 13 | 3.56 m | 0.00004 km² | 11,626,681,248,842 | Very precise |
| 14 | 1.35 m | 0.000006 km² | 81,386,768,741,882 | Sub-meter |
| 15 | 0.51 m | 0.0000009 km² | 569,707,381,193,162 | GPS precision |

### Key H3 Functions Used

```python
import h3

# 1. Convert lat/lon to H3 cell
cell = h3.latlng_to_cell(39.984702, 116.318417, 8)
# Returns: '88283082b9fffff'

# 2. Get center of H3 cell
lat, lng = h3.cell_to_latlng('88283082b9fffff')
# Returns: (39.98486..., 116.31829...)

# 3. Get resolution of cell
res = h3.get_resolution('88283082b9fffff')
# Returns: 8

# 4. Check if two cells are neighbors
is_neighbor = h3.are_neighbor_cells('88283082b9fffff', '88283082b1fffff')
# Returns: True or False

# 5. Get boundary vertices of cell
boundary = h3.cell_to_boundary('88283082b9fffff')
# Returns: tuple of (lat, lng) pairs
```

### H3 Cell ID Structure

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    H3 CELL ID STRUCTURE                                          │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  H3 cell ID: "88283082b9fffff"                                                  │
│                                                                                  │
│  Structure (64-bit integer as hex):                                              │
│  ┌────┬────┬────────────────────────────────────────────────┐                   │
│  │Mode│Res │           Cell Index (hierarchical)            │                   │
│  └────┴────┴────────────────────────────────────────────────┘                   │
│    8    8                     48 bits                                            │
│   ↓     ↓                                                                        │
│  "8"  res=8              Base cell + child indices                               │
│                                                                                  │
│  The ID encodes:                                                                 │
│  • Resolution (8)                                                                │
│  • Base cell (one of 122 at resolution 0)                                        │
│  • Path from base cell to current cell                                           │
│                                                                                  │
│  Hierarchical property:                                                          │
│  Parent cell at res 7: "87283082bffffff"                                        │
│  Child cells at res 9: ['89283082b83ffff', '89283082b87ffff', ...]              │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Complete Code Walkthrough

### Changes from Standard Script 1

Only Step 6 changes. Here's the comparison:

#### Standard Script 1 (DBSCAN)

```python
# 6. Generate locations using DBSCAN
print("\n[6/7] Generating locations using DBSCAN clustering...")
loc_config = preproc_config["location"]
sp, locs = sp.as_staypoints.generate_locations(
    epsilon=epsilon,                              # 20 meters
    num_samples=loc_config["num_samples"],       # 2
    distance_metric=loc_config["distance_metric"], # "haversine"
    agg_level=loc_config["agg_level"],           # "dataset"
    n_jobs=-1
)
```

#### H3 Script 1 (Hexagonal Grid)

```python
# 6. Generate locations using H3 (instead of DBSCAN)
print("\n[6/7] Generating locations using H3 hexagonal grid...")
loc_config = preproc_config["location"]
sp, locs = generate_h3_locations(
    sp, 
    h3_resolution=h3_resolution,  # 8 (from config)
    num_samples=loc_config["num_samples"]  # 2
)
```

### Additional Import

```python
import h3  # Uber H3 library for hexagonal grid
```

### Configuration Parameter Change

```yaml
# Standard config
dataset:
  epsilon: 20  # DBSCAN parameter

# H3 config
dataset:
  h3_resolution: 8  # H3 resolution parameter
```

---

## H3 Cell Assignment Visualization

### Beijing Example

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    H3 GRID OVER BEIJING (Resolution 8)                           │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Latitude 39.9° to 40.1° N, Longitude 116.2° to 116.5° E                        │
│                                                                                  │
│      116.2°   116.25°   116.3°   116.35°   116.4°   116.45°   116.5°            │
│        │        │        │        │        │        │        │                  │
│  40.1° ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ─              │
│       │  │ │  │ │  │ │  │ │  │ │  │ │  │ │  │ │  │ │  │ │  │ │  │               │
│        ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱                  │
│  40.0° ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ─              │
│       │  │ │●●│ │  │ │  │ │●●│ │  │ │  │ │  │ │  │ │  │ │  │ │  │               │
│        ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱                  │
│       ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲  ╱╲                   │
│  39.9°│  │ │  │ │██│ │  │ │  │ │  │ │●●│ │  │ │  │ │  │ │  │ │  │ ─             │
│        ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱  ╲╱                  │
│                │                                                                 │
│                └── User's home location                                          │
│                    H3 cell: 88283082b9fffff                                     │
│                    Staypoints: 150                                               │
│                    → Location ID: 0                                              │
│                                                                                  │
│  ●● = Staypoints (multiple in same cell)                                        │
│  ██ = High-density cell (many visits)                                           │
│                                                                                  │
│  Each hexagon: ~461m edge, ~730m diameter                                       │
│  Covers area: ~0.74 km²                                                          │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Staypoint to Location Mapping Example

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    STAYPOINT TO H3 LOCATION MAPPING                              │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  Input Staypoints:                                                               │
│  ─────────────────                                                               │
│  │ sp_id │    lat    │    lon     │                                             │
│  │   0   │ 39.984702 │ 116.318417 │                                             │
│  │   1   │ 39.984750 │ 116.318500 │  ← Same H3 cell as sp_0                     │
│  │   2   │ 39.989200 │ 116.321000 │  ← Different cell                           │
│  │   3   │ 39.984720 │ 116.318420 │  ← Same cell as sp_0, sp_1                  │
│  │   4   │ 39.950000 │ 116.300000 │  ← Single visit (will be filtered)         │
│                                                                                  │
│  After H3 assignment:                                                            │
│  ─────────────────────                                                           │
│  │ sp_id │       h3_cell        │ cell_count │                                  │
│  │   0   │ 88283082b9fffff      │     3      │ ✓ Keep (≥2)                     │
│  │   1   │ 88283082b9fffff      │     3      │ ✓ Keep                          │
│  │   2   │ 88283082b1fffff      │     1      │ ✗ Filter (<2)                   │
│  │   3   │ 88283082b9fffff      │     3      │ ✓ Keep                          │
│  │   4   │ 88283082adfffff      │     1      │ ✗ Filter (<2)                   │
│                                                                                  │
│  After filtering and ID assignment:                                              │
│  ──────────────────────────────────                                              │
│  │ sp_id │ location_id │                                                        │
│  │   0   │      0      │  All same location (same H3 cell)                     │
│  │   1   │      0      │                                                        │
│  │   3   │      0      │                                                        │
│                                                                                  │
│  sp_2 and sp_4 are removed (insufficient visits to their cells)                  │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files

The H3 version produces files with different naming:

| Standard Version | H3 Version |
|-----------------|------------|
| `intermediate_eps20.csv` | `intermediate_h3r8.csv` |
| `locations_eps20.csv` | `locations_h3r8.csv` |
| `staypoints_all_eps20.csv` | `staypoints_all_h3r8.csv` |
| `valid_users_eps20.csv` | `valid_users_h3r8.csv` |

The `locations_h3r8.csv` includes H3-specific columns:
- `h3_cell`: The H3 cell index
- `center_lat`, `center_lng`: Hexagon center coordinates
- `h3_resolution`: The resolution used

---

## Summary

The H3 version of Script 1:
1. **Same 7-step process** as standard Script 1
2. **Only Step 6 changes**: DBSCAN → H3 hexagonal grid
3. **New function**: `generate_h3_locations()` replaces `generate_locations()`
4. **Different parameter**: `h3_resolution` instead of `epsilon`
5. **Same output format**: Compatible with Script 2

The choice between DBSCAN and H3 depends on your use case:
- **DBSCAN**: Better for natural clustering, semantic grouping
- **H3**: Better for consistent cell sizes, scalability, computational efficiency

---

## Next Steps

- [07-H3-SCRIPT2-INTERIM-TO-PROCESSED.md](07-H3-SCRIPT2-INTERIM-TO-PROCESSED.md) - H3 version of Script 2
- [11-COMPARISON-DBSCAN-VS-H3.md](11-COMPARISON-DBSCAN-VS-H3.md) - Detailed comparison

---

*Documentation Version: 1.0*
*For PhD Research Reference*
