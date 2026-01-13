# Code Walkthrough: Line-by-Line Explanation

## 1. Script Overview

The main analysis script `zipf_location_frequency_analysis.py` implements the complete pipeline for analyzing location visit frequency distribution.

### Script Structure

```
zipf_location_frequency_analysis.py
│
├── Imports and Setup (Lines 1-21)
├── Data Loading (Lines 24-48)
├── Frequency Computation (Lines 50-102)
├── User Grouping (Lines 104-158)
├── Group Statistics (Lines 160-223)
├── Reference Line Fitting (Lines 225-283)
├── Plotting (Lines 285-429)
├── Data Saving (Lines 431-493)
├── Main Analysis Pipeline (Lines 495-547)
└── Entry Point (Lines 549-627)
```

---

## 2. Detailed Code Explanation

### 2.1 Imports and Setup (Lines 1-21)

```python
"""
Zipf Plot of Location Visit Frequency
Reproduces Figure 2d from González et al. (2008)
...
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
```

**Purpose:** Import necessary libraries for:
- `numpy`: Numerical operations
- `pandas`: Data manipulation
- `matplotlib`: Plotting
- `argparse`: Command-line argument parsing

---

### 2.2 Data Loading Function (Lines 24-48)

```python
def load_intermediate_data(dataset_path):
    """
    Load intermediate CSV data from preprocessing.
    """
    print(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # We only need user_id and location_id for visit counts
    df_visits = df[['user_id', 'location_id']].copy()
    
    print(f"Loaded {len(df_visits):,} visits from {df_visits['user_id'].nunique():,} users")
    print(f"Unique locations: {df_visits['location_id'].nunique():,}")
    
    return df_visits
```

**What it does:**
1. Reads CSV file containing mobility data
2. Extracts only the columns needed (`user_id`, `location_id`)
3. Each row = one visit to a location

**Example Input CSV:**
```
user_id,location_id,timestamp,lat,lon,...
1,42,2025-01-01 08:00:00,39.9,116.3,...
1,42,2025-01-01 12:00:00,39.9,116.3,...
1,17,2025-01-01 18:00:00,40.0,116.4,...
```

**Example Output DataFrame:**
```
   user_id  location_id
0        1           42
1        1           42
2        1           17
3        2           55
...
```

---

### 2.3 Computing Location Frequencies (Lines 50-102)

This is the **core computation**. Let's break it down step by step.

#### Step 1: Count Visits Per Location Per User (Lines 73-74)

```python
location_counts = df_visits.groupby(['user_id', 'location_id']).size().reset_index(name='visit_count')
```

**What it does:** For each user, count how many times they visited each location.

**Example:**
```
Input:
   user_id  location_id
0        1           42    ← User 1 visits loc 42
1        1           42    ← User 1 visits loc 42 again
2        1           17    ← User 1 visits loc 17
3        1           42    ← User 1 visits loc 42 again
4        2           55    ← User 2 visits loc 55

Output:
   user_id  location_id  visit_count
0        1           17            1     ← User 1: loc 17 visited 1 time
1        1           42            3     ← User 1: loc 42 visited 3 times
2        2           55            1     ← User 2: loc 55 visited 1 time
```

#### Step 2: Sort by Visit Count (Lines 76-78)

```python
location_counts = location_counts.sort_values(['user_id', 'visit_count'], 
                                               ascending=[True, False])
```

**What it does:** Within each user, sort locations from most to least visited.

**Example:**
```
Before sorting:                    After sorting:
user  loc  count                   user  loc  count
1     17   1                       1     42   3     ← Most visited first
1     42   3                       1     17   1     ← Less visited second
```

#### Step 3: Assign Ranks (Lines 80-82)

```python
location_counts['rank'] = location_counts.groupby('user_id').cumcount() + 1
```

**What it does:** Assign rank L = 1, 2, 3, ... to each location.

**Example:**
```
   user_id  location_id  visit_count  rank
0        1           42            3     1   ← Rank 1: most visited
1        1           17            1     2   ← Rank 2: second most
```

**Visual Representation:**
```
User 1's Locations (after ranking):

    Rank    Location    Visits
    ─────────────────────────────
    L=1     loc_42      50 visits   ← Most visited (e.g., Home)
    L=2     loc_17      30 visits   ← Second most (e.g., Work)
    L=3     loc_89      10 visits   ← Third (e.g., Gym)
    L=4     loc_05       5 visits   ← Fourth (e.g., Store)
    L=5     loc_23       3 visits   ← Fifth (e.g., Restaurant)
```

#### Step 4: Compute Total Visits Per User (Lines 84-86)

```python
user_totals = location_counts.groupby('user_id')['visit_count'].sum().reset_index()
user_totals = user_totals.rename(columns={'visit_count': 'total_visits'})
```

**Example:**
```
   user_id  total_visits
0        1            98    ← User 1 has 98 total visits
1        2            55    ← User 2 has 55 total visits
```

#### Step 5: Compute Probabilities (Lines 88-90)

```python
location_counts = location_counts.merge(user_totals, on='user_id')
location_counts['probability'] = location_counts['visit_count'] / location_counts['total_visits']
```

**What it does:** Convert visit counts to probabilities.

**Formula:**
```
                visits to location at rank L
p_u(L) = ───────────────────────────────────────
                total visits for user u
```

**Example:**
```
   user_id  location_id  visit_count  rank  total_visits  probability
0        1           42           50     1            98        0.510
1        1           17           30     2            98        0.306
2        1           89           10     3            98        0.102
3        1           05            5     4            98        0.051
4        1           23            3     5            98        0.031
                                                              ─────────
                                                        Sum:    1.000 ✓
```

#### Step 6: Count Unique Locations Per User (Lines 92-96)

```python
n_locations = location_counts.groupby('user_id')['location_id'].nunique().reset_index()
n_locations = n_locations.rename(columns={'location_id': 'n_unique_locations'})
location_counts = location_counts.merge(n_locations, on='user_id')
```

**What it does:** Add column for number of unique locations per user.

**Example:**
```
   user_id  n_unique_locations
0        1                   5    ← User 1 visits 5 different locations
1        2                  12    ← User 2 visits 12 different locations
```

---

### 2.4 User Grouping (Lines 104-158)

```python
def assign_user_groups(location_counts, target_n_locations=[5, 10, 30, 50], 
                       bin_widths=[1, 2, 5, 5]):
    """
    Group users by number of unique locations visited.
    Uses binning: for target n_L, include users with 
    n_unique_locations in [n_L - bin_width, n_L + bin_width].
    """
```

**Purpose:** Group users by how many unique locations they visit.

**Why binning?** Exact counts (e.g., exactly 5 locations) may be rare. Binning ensures sufficient users per group.

**Example:**
```
Target n_L = 10, bin_width = 2
→ Include users with 8, 9, 10, 11, or 12 unique locations

Groups:
┌────────────────────────────────────────────────────┐
│ Target n_L │ Range    │ Users included            │
├────────────────────────────────────────────────────┤
│ 5          │ [4, 6]   │ Users with 4-6 locations  │
│ 10         │ [8, 12]  │ Users with 8-12 locations │
│ 30         │ [25, 35] │ Users with 25-35 locations│
│ 50         │ [45, 55] │ Users with 45-55 locations│
└────────────────────────────────────────────────────┘
```

**Key Code:**
```python
for target_n, bin_width in zip(target_n_locations, bin_widths):
    min_n = target_n - bin_width
    max_n = target_n + bin_width
    
    users_in_group = users_info[
        (users_info['n_unique_locations'] >= min_n) & 
        (users_info['n_unique_locations'] <= max_n)
    ]['user_id'].values
```

---

### 2.5 Computing Group Statistics (Lines 160-223)

```python
def compute_group_statistics(user_groups):
    """
    For each group and each rank L, compute mean P(L) and standard error.
    
    P_G(L) = mean_{u in G}[p_u(L)]
    SE_G(L) = std_{u in G}[p_u(L)] / sqrt(|G|)
    """
```

**The pivot table approach:**
```python
pivot = group_data.pivot_table(
    index='user_id', 
    columns='rank', 
    values='probability',
    aggfunc='first'
)
```

**Pivot Table Visualization:**
```
                    Rank L
           │  1      2      3      4      5   ...
User ID    │
───────────┼─────────────────────────────────────
user_1     │ 0.510  0.306  0.102  0.051  0.031
user_2     │ 0.450  0.280  0.150  0.070  0.050
user_3     │ 0.600  0.200  0.120  0.050  0.030
user_4     │ 0.480  0.320  0.110  0.060  0.030
           │  ↓      ↓      ↓      ↓      ↓
Mean       │ 0.510  0.277  0.121  0.058  0.035  ← P_G(L)
Std        │ 0.063  0.055  0.021  0.009  0.010  ← σ_G(L)
SE         │ 0.032  0.028  0.011  0.005  0.005  ← SE_G(L)
```

**Standard Error Calculation:**
```python
mean_prob = pivot.mean(axis=0)      # Mean across users (rows)
std_prob = pivot.std(axis=0, ddof=1) # Sample std dev
n_users = pivot.count(axis=0)        # Number of users per rank
std_error = std_prob / np.sqrt(n_users)  # SE = σ/√n
```

---

### 2.6 Reference Line Fitting (Lines 225-283)

```python
def fit_reference_line(group_stats, fit_rank_range=(3, 10)):
    """
    Fit a reference line: P(L) = c × L^(-1)
    
    In log space: log(P(L)) = log(c) - log(L)
    """
```

**Fitting Process:**

1. **Collect data** from mid-ranks (L=3 to L=10):
```python
fit_data = stats_df[
    (stats_df['rank'] >= fit_rank_range[0]) & 
    (stats_df['rank'] <= fit_rank_range[1])
]
```

2. **Transform to log space:**
```python
log_L = np.log(all_ranks)
log_P = np.log(all_probs)
```

3. **Solve for c:**
```python
# log(P) = log(c) - log(L)
# → log(c) = log(P) + log(L)
# → log(c) = mean(log(P) + log(L))
log_c = np.mean(log_P + log_L)
c = np.exp(log_c)
```

**Why ranks 3-10?**
- **Avoid L=1:** Often higher than predicted (home bias)
- **Avoid large L:** Too few users, noisy data
- **Mid-ranks:** Best representation of Zipf behavior

---

### 2.7 Plotting Function (Lines 285-429)

```python
def plot_zipf(group_stats, c_ref, dataset_name, output_path, max_rank=None):
    """
    Create the Zipf plot with main panel (log-log) and inset (linear).
    """
```

**Plot Structure:**
```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  MAIN PANEL (log-log scale)                                 │
│                                            ┌──────────────┐ │
│  P(L)                                      │ INSET        │ │
│  10⁰ ┤ ●                                   │ (linear)     │ │
│      │  ○                                  │              │ │
│ 10⁻¹ ┤    ■                                │ Shows L=1-6  │ │
│      │      △                              │ with error   │ │
│ 10⁻² ┤        ◇                            │ bars         │ │
│      │          ○                          └──────────────┘ │
│ 10⁻³ ┤            ■                                         │
│      └──────────────────────────────────                    │
│      10⁰     10¹     10²                                    │
│                L (rank)                                      │
│                                                              │
│  Legend: ○ 5 loc.  □ 10 loc.  △ 30 loc.  ◇ 50 loc.         │
│          ── L^(-1) reference                                │
└─────────────────────────────────────────────────────────────┘
```

**Key Plotting Code:**

1. **Main panel (log-log):**
```python
ax_main.loglog(
    stats_df['rank'], 
    stats_df['mean_prob'],
    marker=style['marker'],
    color=style['color'],
    ...
)
```

2. **Reference line:**
```python
L_ref = np.logspace(0, np.log10(max_rank * 1.5), 100)
P_ref = c_ref / L_ref
ax_main.loglog(L_ref, P_ref, 'k-', linewidth=2.5, ...)
```

3. **Inset (linear scale):**
```python
ax_inset = fig.add_axes([0.56, 0.58, 0.32, 0.28])
ax_inset.errorbar(
    inset_data['rank'],
    inset_data['mean_prob'],
    yerr=inset_data['std_error'],
    ...
)
```

---

### 2.8 Main Analysis Pipeline (Lines 495-547)

```python
def analyze_dataset(dataset_path, dataset_name, output_dir, ...):
    """Complete analysis pipeline for one dataset."""
    
    # Step 1: Load data
    df_visits = load_intermediate_data(dataset_path)
    
    # Step 2: Compute frequencies
    location_counts = compute_user_location_frequencies(df_visits)
    
    # Step 3: Group users
    user_groups = assign_user_groups(location_counts, target_n_locations, bin_widths)
    
    # Step 4: Compute statistics
    group_stats = compute_group_statistics(user_groups)
    
    # Step 5: Fit reference line
    c_ref = fit_reference_line(group_stats, fit_rank_range)
    
    # Step 6: Create plot
    plot_zipf(group_stats, c_ref, dataset_name, plot_file)
    
    # Step 7: Save data
    save_results_data(group_stats, user_groups, data_file)
```

**Pipeline Visualization:**
```
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Raw CSV      │ → → │  Count Visits  │ → → │  Assign Ranks  │
│   user,loc     │     │  per user,loc  │     │  L=1,2,3,...   │
└────────────────┘     └────────────────┘     └────────────────┘
                                                      │
                                                      ▼
┌────────────────┐     ┌────────────────┐     ┌────────────────┐
│   Generate     │ ← ← │  Fit L^(-1)    │ ← ← │  Compute P(L)  │
│   Plots        │     │  Reference     │     │  & Group Stats │
└────────────────┘     └────────────────┘     └────────────────┘
```

---

## 3. Example: Complete Walkthrough

Let's trace through the code with a **concrete example**.

### Input Data
```
Suppose we have 3 users with these visits:

User 1: Home(×50), Work(×30), Gym(×10), Store(×5), Cafe(×3), Park(×2)
        → 6 unique locations, 100 total visits

User 2: Home(×40), Work(×25), School(×20), Shop(×10), Clinic(×5)
        → 5 unique locations, 100 total visits

User 3: Home(×35), Work(×30), Gym(×15), Mall(×10), Park(×10)
        → 5 unique locations, 100 total visits
```

### Step-by-Step Processing

**1. After counting and ranking:**
```
User 1:
  Rank L=1: Home,  50 visits, p(1) = 0.50
  Rank L=2: Work,  30 visits, p(2) = 0.30
  Rank L=3: Gym,   10 visits, p(3) = 0.10
  Rank L=4: Store,  5 visits, p(4) = 0.05
  Rank L=5: Cafe,   3 visits, p(5) = 0.03
  Rank L=6: Park,   2 visits, p(6) = 0.02

User 2:
  Rank L=1: Home,   40 visits, p(1) = 0.40
  Rank L=2: Work,   25 visits, p(2) = 0.25
  Rank L=3: School, 20 visits, p(3) = 0.20
  Rank L=4: Shop,   10 visits, p(4) = 0.10
  Rank L=5: Clinic,  5 visits, p(5) = 0.05

User 3:
  Rank L=1: Home,  35 visits, p(1) = 0.35
  Rank L=2: Work,  30 visits, p(2) = 0.30
  Rank L=3: Gym,   15 visits, p(3) = 0.15
  Rank L=4: Mall,  10 visits, p(4) = 0.10
  Rank L=5: Park,  10 visits, p(5) = 0.10
```

**2. User grouping (n_L=5, bin_width=1 → range [4,6]):**
```
User 1: 6 locations → IN GROUP (4≤6≤6) ✓
User 2: 5 locations → IN GROUP (4≤5≤6) ✓
User 3: 5 locations → IN GROUP (4≤5≤6) ✓
```

**3. Compute group statistics:**
```
For rank L=1:
  p_user1(1) = 0.50
  p_user2(1) = 0.40
  p_user3(1) = 0.35
  ──────────────────
  Mean P(1) = (0.50 + 0.40 + 0.35) / 3 = 0.417
  Std       = 0.076
  SE        = 0.076 / √3 = 0.044

For rank L=2:
  p_user1(2) = 0.30
  p_user2(2) = 0.25
  p_user3(2) = 0.30
  ──────────────────
  Mean P(2) = (0.30 + 0.25 + 0.30) / 3 = 0.283
  Std       = 0.029
  SE        = 0.029 / √3 = 0.017

... and so on for L=3, 4, 5
```

**4. Final output for plotting:**
```
Rank  Mean P(L)  Std Error
─────────────────────────────
1     0.417      0.044
2     0.283      0.017
3     0.150      0.029
4     0.083      0.017
5     0.060      0.021
```

---

## 4. Summary

The code follows a clear pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│  1. LOAD: CSV with (user_id, location_id) pairs            │
│                          ↓                                   │
│  2. COUNT: Visits per location per user                     │
│                          ↓                                   │
│  3. RANK: Sort locations by visit count                     │
│                          ↓                                   │
│  4. PROBABILITY: P(L) = visits_to_L / total_visits          │
│                          ↓                                   │
│  5. GROUP: Users by n_unique_locations                      │
│                          ↓                                   │
│  6. AGGREGATE: Mean P(L) and SE per group                   │
│                          ↓                                   │
│  7. FIT: Reference line c × L^(-1)                          │
│                          ↓                                   │
│  8. PLOT: Log-log main panel + linear inset                 │
└─────────────────────────────────────────────────────────────┘
```

---

*Next: [04_DATA_PIPELINE.md](./04_DATA_PIPELINE.md) - Input/output data formats*
