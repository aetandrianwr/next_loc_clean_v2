# Location Clustering Deep Dive: DBSCAN vs H3

## 📋 Table of Contents
1. [Introduction](#introduction)
2. [Fundamental Concepts](#fundamental-concepts)
3. [DBSCAN Clustering](#dbscan-clustering)
4. [H3 Hexagonal Indexing](#h3-hexagonal-indexing)
5. [Side-by-Side Comparison](#side-by-side-comparison)
6. [Impact on Next Location Prediction](#impact-on-next-location-prediction)
7. [Choosing the Right Method](#choosing-the-right-method)
8. [Practical Examples](#practical-examples)

---

## Introduction

### Why Location Clustering?

Raw GPS coordinates cannot be used directly for next location prediction:

```
Problem: GPS Points are Too Precise
═══════════════════════════════════════════════════════════════════════════════

Example: Person visits "Home" multiple times

Visit 1: (-7.76245123, 110.37891456)
Visit 2: (-7.76244987, 110.37892134)
Visit 3: (-7.76245567, 110.37890789)
Visit 4: (-7.76244234, 110.37891678)

These are all the SAME location (their home)!
But GPS coordinates differ by small amounts due to:
• GPS measurement noise
• Moving within the location
• Device accuracy variations

Without clustering, the model sees 4 DIFFERENT locations
With clustering, the model sees 1 location (HOME)
```

### Two Approaches

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    LOCATION CLUSTERING APPROACHES                            │
└─────────────────────────────────────────────────────────────────────────────┘

                    ┌─────────────────────┐
                    │    Staypoints       │
                    │   (lat, lon, time)  │
                    └─────────┬───────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
              ▼               │               ▼
    ┌─────────────────┐       │     ┌─────────────────┐
    │     DBSCAN      │       │     │       H3        │
    │   Clustering    │       │     │    Indexing     │
    └────────┬────────┘       │     └────────┬────────┘
             │                │              │
             ▼                │              ▼
    ┌─────────────────┐       │     ┌─────────────────┐
    │  Density-Based  │       │     │   Grid-Based    │
    │   Locations     │       │     │   Locations     │
    │                 │       │     │                 │
    │ • Adaptive size │       │     │ • Fixed size    │
    │ • Data-driven   │       │     │ • Deterministic │
    │ • Variable IDs  │       │     │ • Stable IDs    │
    └─────────────────┘       │     └─────────────────┘
```

---

## Fundamental Concepts

### What is a "Location"?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    CONCEPT: LOCATION vs COORDINATES                          │
└─────────────────────────────────────────────────────────────────────────────┘

                         REAL WORLD                    
                    ┌────────────────────┐             
                    │                    │             
                    │    🏠 Home         │    ← Semantic meaning
                    │                    │             
                    │  * * *  *   *      │    ← Multiple GPS points
                    │    * * *  *        │       within this area
                    │     *   * *        │             
                    └────────────────────┘             
                              │                        
                              │                        
              ┌───────────────┼───────────────┐        
              │               │               │        
              ▼               ▼               ▼        
         GPS Points      DBSCAN Cluster    H3 Cell    
                                                       
    (-7.762, 110.378)      Location ID       H3 Index  
    (-7.761, 110.379)         = 42        = "872d...c" 
    (-7.763, 110.377)                      Location ID  
          ...                                 = 127    


GOAL: Map multiple GPS points → Single Location ID
```

### Coordinates to Location ID Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GPS → LOCATION ID TRANSFORMATION                          │
└─────────────────────────────────────────────────────────────────────────────┘

Step 1: Raw GPS Coordinates (Staypoints)
─────────────────────────────────────────

    staypoint_id │ user_id │     lat      │      lon      │ started_at          
    ─────────────┼─────────┼──────────────┼───────────────┼─────────────────────
    1001         │ user_01 │ -7.76245123  │ 110.37891456  │ 2023-01-15 07:00:00 
    1002         │ user_01 │ -7.76244987  │ 110.37892134  │ 2023-01-15 19:00:00 
    1003         │ user_01 │ -7.28567234  │ 110.40123567  │ 2023-01-16 08:30:00 
    1004         │ user_01 │ -7.76245567  │ 110.37890789  │ 2023-01-16 18:00:00 


Step 2: Clustering/Indexing
─────────────────────────────

    DBSCAN Method:                          H3 Method:
    ────────────────                        ────────────────
    Group nearby points                     Assign to hex cell
    into clusters                           based on coordinates

    1001 ─┬─► Cluster 0 = Home              1001 ─► Cell A = Home
    1002 ─┘                                 1002 ─► Cell A = Home
    1003 ───► Cluster 1 = Office            1003 ─► Cell B = Office
    1004 ───► Cluster 0 = Home              1004 ─► Cell A = Home


Step 3: Location ID Assignment
──────────────────────────────

    staypoint_id │ DBSCAN location_id │ H3 location_id
    ─────────────┼────────────────────┼─────────────────
    1001         │ 0                  │ 127
    1002         │ 0                  │ 127
    1003         │ 1                  │ 243
    1004         │ 0                  │ 127


Step 4: Final Encoding (+2 offset)
──────────────────────────────────

    staypoint_id │ Final location_id (DBSCAN) │ Final location_id (H3)
    ─────────────┼────────────────────────────┼────────────────────────
    1001         │ 2                          │ 129
    1002         │ 2                          │ 129
    1003         │ 3                          │ 245
    1004         │ 2                          │ 129

    ID 0 = Padding
    ID 1 = Unknown location (for test set)
    ID 2+ = Actual locations
```

---

## DBSCAN Clustering

### Algorithm Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DBSCAN: Density-Based Spatial Clustering                  │
└─────────────────────────────────────────────────────────────────────────────┘

Parameters:
─────────────
• epsilon (ε): Maximum distance between points (default: 50 meters)
• num_samples: Minimum points per cluster (default: 2)

Algorithm Steps:
────────────────

1. Pick an unvisited point
2. Find all points within ε distance
3. If >= num_samples neighbors found:
   - Create new cluster
   - Recursively add all reachable points
4. Repeat for all unvisited points
5. Points not in any cluster → "noise" (separate locations)


Visual Example (ε = 50m, num_samples = 2):
──────────────────────────────────────────

        ┌─────────────────────────────────────────────────┐
        │                                                 │
        │    *    * *                                     │
        │     * * *        <- Cluster A (8 points)        │
        │      * *                                        │
        │                                                 │
        │                        *                        │
        │                       * * *  <- Cluster B       │
        │                        * *     (6 points)       │
        │                                                 │
        │      ●                                 ●        │
        │  (isolated)                       (isolated)    │
        │                                                 │
        │              Each becomes its own "location"    │
        │              with only 1 visit                  │
        └─────────────────────────────────────────────────┘

Result: 
• Cluster A → location_id = 0
• Cluster B → location_id = 1
• Isolated point 1 → location_id = 2 (separate location)
• Isolated point 2 → location_id = 3 (separate location)
```

### DBSCAN in Trackintel

```python
# From trackintel library (used in preprocessing)
locations = staypoints.generate_locations(
    method='dbscan',
    epsilon=50,          # 50 meters radius
    num_samples=2,       # Minimum 2 staypoints
    distance_metric='haversine',  # Spherical distance
    agg_level='dataset'  # Cluster across all users
)

# Result: Each unique geographic area becomes a location
```

### DBSCAN Characteristics

```
ADVANTAGES:
───────────
✓ Adapts to data density
  - Dense areas (downtown) → smaller locations
  - Sparse areas (suburbs) → larger locations

✓ No fixed grid alignment issues
  - Locations follow natural point distributions

✓ Handles arbitrary shapes
  - Can identify L-shaped or irregular locations

DISADVANTAGES:
──────────────
✗ Not deterministic
  - Different point orders can give different results
  - Adding new data may change existing clusters

✗ Sensitive to parameters
  - epsilon too small → too many tiny clusters
  - epsilon too large → distinct places merged

✗ Computationally expensive
  - O(n²) worst case for distance calculations

✗ Not comparable across datasets
  - Location ID 42 in Dataset A ≠ Location ID 42 in Dataset B
```

---

## H3 Hexagonal Indexing

### Algorithm Explanation

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    H3: Hierarchical Hexagonal Index                          │
└─────────────────────────────────────────────────────────────────────────────┘

Developed by Uber for geospatial indexing

Concept: Divide Earth's surface into hexagonal cells
─────────────────────────────────────────────────────

        ┌─────┬─────┬─────┬─────┬─────┐
       / \   / \   / \   / \   / \   /
      /   \ /   \ /   \ /   \ /   \ /
     ├─────┼─────┼─────┼─────┼─────┤
    / \   / \   / \   / \   / \   /
   /   \ / * * \ /   \ /   \ /   \ /    ← GPS point falls into
  ├─────┼─────┼─────┼─────┼─────┤        one specific cell
 / \   / \ * * / \   / \   / \   /
/   \ /   \ * /   \ /   \ /   \ /
├─────┼─────┼─────┼─────┼─────┤
 \   / \   / \   / \   / \   / \
  \ /   \ /   \ /   \ /   \ /   \

Resolution: Controls cell size
──────────────────────────────

Res │ Edge Length │ Cell Area    │ Use Case
────┼─────────────┼──────────────┼─────────────────────────
 5  │ ~8.5 km     │ ~252.9 km²   │ Regional analysis
 6  │ ~3.2 km     │ ~36.1 km²    │ City districts
 7  │ ~1.2 km     │ ~5.2 km²     │ Neighborhoods
 8  │ ~461 m      │ ~0.74 km²    │ Points of interest ← Default
 9  │ ~174 m      │ ~0.11 km²    │ Individual buildings
10  │ ~66 m       │ ~0.015 km²   │ Precise locations
```

### H3 Conversion

```python
import h3

# Convert GPS to H3 cell (resolution 8)
lat, lon = -7.76245123, 110.37891456
h3_index = h3.latlng_to_cell(lat, lon, 8)
# Result: '872d9a534ffffff'

# All points in same hex cell get same index
h3.latlng_to_cell(-7.76244987, 110.37892134, 8)  # Same cell!
# Result: '872d9a534ffffff'

# Get cell center
h3.cell_to_latlng('872d9a534ffffff')
# Result: (-7.762423, 110.378912)

# Get cell boundary
h3.cell_to_boundary('872d9a534ffffff')
# Result: [(-7.761, 110.378), (-7.762, 110.379), ...]
```

### H3 Characteristics

```
ADVANTAGES:
───────────
✓ Deterministic
  - Same coordinates always → same cell ID
  - Results are reproducible

✓ Comparable across datasets
  - Cell ID '872d9a534ffffff' means the same location globally
  - Can merge/compare datasets directly

✓ Computationally efficient
  - O(1) conversion for each point
  - No clustering computation needed

✓ Hierarchical
  - Can zoom in/out by changing resolution
  - Parent-child relationships between cells

DISADVANTAGES:
──────────────
✗ Fixed grid may not align with actual places
  - A building might span 2-3 cells
  - Edge effects at cell boundaries

✗ Hexagon size is uniform
  - Urban areas might need finer resolution
  - Rural areas might need coarser resolution

✗ Not adaptive to data density
  - Same cell size everywhere regardless of visits
```

---

## Side-by-Side Comparison

### Visual Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DBSCAN vs H3: Same GPS Points                             │
└─────────────────────────────────────────────────────────────────────────────┘

                   Raw GPS Points
                   ──────────────
        ┌─────────────────────────────────────────┐
        │                                         │
        │    * *                     *            │
        │   * * * *           * *  *   *          │
        │    * * *             * * *              │
        │     *                  *                │
        │                                         │
        │                                         │
        │        * * *                            │
        │       *   * *                           │
        │        * *                              │
        │                                         │
        └─────────────────────────────────────────┘


           DBSCAN (ε=50m)                    H3 (Resolution 8)
           ───────────────                   ─────────────────
        ┌─────────────────────┐           ┌─────────────────────┐
        │                     │           │  ⬡   ⬡   ⬡   ⬡   ⬡  │
        │   ┌───────┐   ┌────┐│           │ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ │
        │   │ * * * │   │* * ││           │  ⬡ █ █ ⬡ ⬡ █ █ ⬡  │
        │   │* * *  │   │* * ││           │ ⬡ █ █ ⬡ ⬡ ⬡ █ ⬡ ⬡ │
        │   │ * *   │   │ *  ││           │  ⬡ █ ⬡ ⬡ ⬡ █ ⬡   │
        │   └───────┘   └────┘│           │ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ │
        │     Loc 0     Loc 1 │           │  ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡   │
        │                     │           │ ⬡ ⬡ █ █ ⬡ ⬡ ⬡ ⬡ ⬡ │
        │       ┌─────────┐   │           │  ⬡ █ █ ⬡ ⬡ ⬡ ⬡   │
        │       │* * *    │   │           │ ⬡ ⬡ █ ⬡ ⬡ ⬡ ⬡ ⬡ ⬡ │
        │       │  * * *  │   │           │  ⬡   ⬡   ⬡   ⬡   │
        │       └─────────┘   │           │ █ = Cell with points │
        │         Loc 2       │           │ ⬡ = Empty cell       │
        └─────────────────────┘           └─────────────────────┘
        
        Result:                            Result:
        • 3 locations                      • 7 occupied cells
        • Arbitrary shapes                 • Fixed hexagonal shapes
        • IDs: 0, 1, 2                     • IDs: Cell indices
```

### Feature Comparison Table

```
┌────────────────────────┬──────────────────────┬──────────────────────┐
│ Feature                │ DBSCAN               │ H3                   │
├────────────────────────┼──────────────────────┼──────────────────────┤
│ Algorithm Type         │ Density-based        │ Grid-based           │
│ Location Shape         │ Arbitrary            │ Hexagonal            │
│ Size                   │ Adaptive to data     │ Fixed by resolution  │
│ Determinism           │ Pseudo-random        │ Fully deterministic  │
│ Cross-dataset compare │ Not possible         │ Yes, same cell IDs   │
│ Computation           │ O(n log n) to O(n²)  │ O(n)                 │
│ Parameter             │ epsilon (meters)     │ resolution (0-15)    │
│ Default setting       │ 50 meters            │ Resolution 8 (~461m) │
│ Handles noise         │ Yes (outliers)       │ No (all assigned)    │
│ Edge effects          │ Minimal              │ At cell boundaries   │
│ Hierarchical          │ No                   │ Yes (parent cells)   │
│ Memory usage          │ Higher (clustering)  │ Lower (indexing)     │
└────────────────────────┴──────────────────────┴──────────────────────┘
```

### Statistical Comparison (Typical DIY Dataset)

```
┌────────────────────────┬──────────────────────┬──────────────────────┐
│ Metric                 │ DBSCAN (ε=50)        │ H3 (res=8)           │
├────────────────────────┼──────────────────────┼──────────────────────┤
│ Total locations        │ ~4,500               │ ~5,700               │
│ Avg visits per loc     │ ~60                  │ ~50                  │
│ Single-visit locs      │ ~15%                 │ ~20%                 │
│ Processing time        │ ~5 minutes           │ ~30 seconds          │
│ Unique users after QF  │ ~152                 │ ~152                 │
│ Train sequences        │ ~65,000              │ ~75,000              │
│ Location granularity   │ Fine (50m)           │ Coarser (~461m)      │
└────────────────────────┴──────────────────────┴──────────────────────┘

Note: H3 resolution 8 is less precise than DBSCAN ε=50m
      For comparable precision, use H3 resolution 9 (~174m) or 10 (~66m)
```

---

## Impact on Next Location Prediction

### Model Perspective

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    HOW CLUSTERING AFFECTS PREDICTION                         │
└─────────────────────────────────────────────────────────────────────────────┘

The model learns: "Given history [L1, L2, L3, ...], predict next location"

DBSCAN Example:
───────────────
    L1 = DBSCAN Cluster 42 (exact area around user's home)
    L2 = DBSCAN Cluster 15 (exact area around user's office)
    
    Model learns: User often goes from Cluster 42 → Cluster 15
                  (Home → Office transition pattern)

H3 Example:
───────────
    L1 = H3 Cell 127 (hexagon containing user's home)
    L2 = H3 Cell 243 (hexagon containing user's office)
    
    Model learns: User often goes from Cell 127 → Cell 243
                  (Same semantic pattern, different location IDs)

Both approaches capture the same mobility patterns!
The difference is in the granularity and consistency of location definitions.
```

### Accuracy Implications

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PREDICTION ACCURACY FACTORS                               │
└─────────────────────────────────────────────────────────────────────────────┘

Factor 1: Location Granularity
──────────────────────────────

    Fine-grained (small locations):
    • More precise predictions
    • But harder to learn (more vocabulary)
    • More location_ids to predict from
    
    Example: 
    DBSCAN ε=50m → 4,500 locations → harder prediction task
    H3 res=8     → 5,700 locations → similar difficulty
    H3 res=7     → 1,200 locations → easier task, less precise


Factor 2: Location Consistency
─────────────────────────────

    DBSCAN:
    • Location boundaries adapt to visit patterns
    • Places with many visits → well-defined clusters
    • Rarely visited places → might be noise/separate
    
    H3:
    • Location boundaries are fixed
    • Same cell for everyone visiting that area
    • Might split a single building into multiple cells


Factor 3: Vocabulary Size
─────────────────────────

    Embedding Layer: num_locations → embedding_dim
    
    More locations = Larger embedding matrix
                   = More parameters to learn
                   = Need more training data
    
    DBSCAN with small ε or H3 with high resolution:
    → More locations → Larger model → Need more data
```

### Which is Better for Prediction?

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    EMPIRICAL OBSERVATIONS                                    │
└─────────────────────────────────────────────────────────────────────────────┘

Research findings (general trends):
──────────────────────────────────

1. Similar Accuracy
   Both methods achieve comparable prediction accuracy when properly tuned
   The model can learn patterns from either representation

2. DBSCAN May Be Better When:
   • You have dense, clustered data
   • Location boundaries should follow human behavior
   • You want to capture POI-level granularity
   
3. H3 May Be Better When:
   • You need reproducible results
   • You're comparing across datasets
   • Computational efficiency matters
   • You need hierarchical analysis

4. Hybrid Approach:
   Some research uses both:
   • H3 for regional features
   • DBSCAN for local features
```

---

## Choosing the Right Method

### Decision Flowchart

```
                              START
                                │
                                ▼
                    ┌───────────────────────┐
                    │  Need reproducible    │
                    │  results across runs? │
                    └───────────┬───────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
                   YES                      NO
                    │                       │
                    ▼                       ▼
              ┌─────────┐         ┌─────────────────────┐
              │ Use H3  │         │  Need to compare    │
              └─────────┘         │  across datasets?   │
                                  └──────────┬──────────┘
                                             │
                                  ┌──────────┴──────────┐
                                  │                     │
                                 YES                    NO
                                  │                     │
                                  ▼                     ▼
                            ┌─────────┐      ┌─────────────────────┐
                            │ Use H3  │      │  Have dense,        │
                            └─────────┘      │  clustered visits?  │
                                             └──────────┬──────────┘
                                                        │
                                             ┌──────────┴──────────┐
                                             │                     │
                                            YES                    NO
                                             │                     │
                                             ▼                     ▼
                                       ┌──────────┐          ┌──────────┐
                                       │ DBSCAN   │          │  Either  │
                                       └──────────┘          │   works  │
                                                             └──────────┘
```

### Configuration Recommendations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    RECOMMENDED CONFIGURATIONS                                │
└─────────────────────────────────────────────────────────────────────────────┘

For PhD Research (reproducibility is key):
──────────────────────────────────────────
    Use H3 Resolution 8 or 9
    • Deterministic results
    • Comparable across experiments
    • Document which resolution used

For Industry Applications:
──────────────────────────
    Use H3 Resolution 7-9
    • Consistent across deployments
    • Easy to update with new data
    • Efficient computation

For Fine-grained Analysis:
──────────────────────────
    Use DBSCAN with ε=30-50m
    • Captures POI-level details
    • Adapts to local density
    • Good for urban environments

For Large-scale Analysis:
─────────────────────────
    Use H3 Resolution 6-7
    • Neighborhood/district level
    • Manageable vocabulary size
    • Good for regional patterns


Parameter Mapping (approximate equivalence):
────────────────────────────────────────────

    DBSCAN ε │ H3 Resolution │ Typical Scale
    ─────────┼───────────────┼──────────────
    30m      │ 10            │ Building
    50m      │ 9-10          │ Block
    100m     │ 9             │ POI area
    200m     │ 8-9           │ Small area
    500m     │ 8             │ Neighborhood
```

---

## Practical Examples

### Example 1: Same User, Different Methods

```
User: user_001 (Sample Data)
═══════════════════════════════════════════════════════════════════════════════

Raw Staypoints:
───────────────
    staypoint │     lat        │      lon       │ started_at
    ──────────┼────────────────┼────────────────┼─────────────────────
    SP_001    │ -7.76245123    │ 110.37891456   │ 2023-01-15 07:00:00
    SP_002    │ -7.28567234    │ 110.40123567   │ 2023-01-15 09:00:00
    SP_003    │ -7.76244987    │ 110.37892134   │ 2023-01-15 18:00:00
    SP_004    │ -7.28568123    │ 110.40124890   │ 2023-01-16 08:30:00
    SP_005    │ -7.76245567    │ 110.37890789   │ 2023-01-16 17:30:00
    SP_006    │ -7.32145678    │ 110.41234567   │ 2023-01-16 20:00:00


DBSCAN Clustering (ε=50m):
──────────────────────────
    
    Cluster 0 (Home): SP_001, SP_003, SP_005
        Center: (-7.762, 110.379)
        Points within 50m of each other
    
    Cluster 1 (Work): SP_002, SP_004
        Center: (-7.286, 110.401)
        Points within 50m of each other
    
    Cluster 2 (Restaurant): SP_006
        Single point → own cluster (or noise)
        
    Result: 3 unique locations


H3 Indexing (Resolution 8):
───────────────────────────
    
    Cell A: SP_001, SP_003, SP_005 → H3 index '872d9a534ffffff'
    Cell B: SP_002, SP_004         → H3 index '872d9b123ffffff'
    Cell C: SP_006                 → H3 index '872d9c789ffffff'
    
    Result: 3 unique locations (same semantic result!)


Sequence Representation:
────────────────────────
    
    DBSCAN sequence: [Home→Work→Home→Work→Home→Restaurant]
                     [  0 → 1  → 0  → 1  → 0  → 2        ]
    
    H3 sequence:     [CellA→CellB→CellA→CellB→CellA→CellC]
                     [ 127 → 243 → 127 → 243 → 127 → 456 ]
    
    Same pattern! Just different ID numbers.
```

### Example 2: Edge Case - Boundary Issues

```
Scenario: User visits places near a hexagon boundary
═══════════════════════════════════════════════════════════════════════════════

        H3 Cell Boundary
             │
    ┌────────┼────────┐
    │        │        │
    │  Cell  │  Cell  │
    │   A    │   B    │
    │        │        │
    │    *   │  *     │  ← Two visits to SAME building
    │        │        │     but split across cells!
    └────────┼────────┘
             │

DBSCAN Result:
──────────────
    Both points within ε=50m → Same cluster
    ✓ Correctly identifies as same location

H3 Result:
──────────
    Point 1 → Cell A
    Point 2 → Cell B
    ✗ Incorrectly identifies as different locations

Solution: Use higher resolution (smaller cells) or accept some edge effects
```

### Example 3: Impact on Prediction

```
Prediction Task: Where will user go next?
═══════════════════════════════════════════════════════════════════════════════

History: User visited [Home, Work, Gym, Home, Work] 

DBSCAN (ε=50m):
───────────────
    History IDs: [42, 15, 8, 42, 15]
    Target prediction: 8 (Gym) or 42 (Home)?
    
    Model sees precise location patterns:
    • Work (cluster 15) → Home (cluster 42): common transition
    • Work (cluster 15) → Gym (cluster 8): occasional

H3 (Resolution 8):
──────────────────
    History IDs: [127, 243, 456, 127, 243]
    Target prediction: 456 (Gym cell) or 127 (Home cell)?
    
    Model sees similar patterns, just different IDs:
    • Cell 243 → Cell 127: common transition
    • Cell 243 → Cell 456: occasional

Both models can learn the same transition patterns!
The key is consistency within the dataset.
```

---

## Summary

### Key Takeaways

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    SUMMARY: DBSCAN vs H3                                     │
└─────────────────────────────────────────────────────────────────────────────┘

1. BOTH METHODS ARE VALID
   They serve the same purpose: converting GPS coordinates to location IDs
   Both can achieve good prediction accuracy with proper tuning

2. CHOOSE DBSCAN WHEN:
   • You want adaptive location boundaries
   • You have dense, clustered data
   • You need POI-level precision
   • You're doing single-dataset analysis

3. CHOOSE H3 WHEN:
   • You need reproducible results (PhD research!)
   • You're comparing across multiple datasets
   • Computational efficiency matters
   • You want hierarchical analysis capability

4. DEFAULT RECOMMENDATIONS:
   • DBSCAN: ε=50 meters, num_samples=2
   • H3: Resolution 8 (general) or 9 (fine-grained)

5. FOR PHD THESIS:
   Document your choice clearly:
   "We used [DBSCAN/H3] with [parameters] because [justification]"
   Consider running experiments with both for comparison
```

### Quick Reference

```
DBSCAN (diy.yaml):               H3 (diy_h3.yaml):
────────────────────             ────────────────────
epsilon: 50                      h3_resolution: 8
num_samples: 2                   
                                 
Script 1: diy_1_raw_to_interim   Script 1: diy_h3_1_raw_to_interim
Script 2: diy_2_interim_to_proc  Script 2: diy_h3_2_interim_to_proc

Output: diy_eps50_*              Output: diy_h3r8_*
```
