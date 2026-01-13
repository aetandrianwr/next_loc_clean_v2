# Data Pipeline: Input, Processing, and Output

## 1. Overview

This document describes the complete data flow from raw input to final outputs.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        DATA PIPELINE OVERVIEW                            │
│                                                                          │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────┐    ┌────────────┐ │
│  │  INPUT   │ →→ │  TRANSFORM  │ →→ │  AGGREGATE   │ →→ │   OUTPUT   │ │
│  │  CSV     │    │  (Per User) │    │  (Per Group) │    │  (Plots &  │ │
│  │          │    │             │    │              │    │   CSVs)    │ │
│  └──────────┘    └─────────────┘    └──────────────┘    └────────────┘ │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Input Data

### 2.1 Source Files

| Dataset | Path | Description |
|---------|------|-------------|
| Geolife | `data/geolife_eps20/interim/intermediate_eps20.csv` | GPS trajectory data |
| DIY | `data/diy_eps50/interim/intermediate_eps50.csv` | DIY trajectory data |

### 2.2 Input File Format

```
intermediate_eps{epsilon}.csv
```

**Required Columns:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `user_id` | int | Unique user identifier | 1, 2, 3, ... |
| `location_id` | int | Cluster/location identifier | 42, 17, 89, ... |

**Optional Columns (not used in this analysis):**
| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | datetime | Visit timestamp |
| `lat` | float | Latitude |
| `lon` | float | Longitude |
| `duration` | int | Stay duration |

### 2.3 Input Data Structure

Each row represents **one visit** to a location:

```
┌─────────────────────────────────────────────────────────────┐
│  INPUT CSV STRUCTURE                                         │
│                                                              │
│  user_id │ location_id │ timestamp           │ ...          │
│  ────────┼─────────────┼─────────────────────┼─────         │
│  1       │ 42          │ 2025-01-01 08:00:00 │              │
│  1       │ 42          │ 2025-01-01 12:00:00 │              │ ← Same user, same loc
│  1       │ 17          │ 2025-01-01 18:00:00 │              │
│  1       │ 42          │ 2025-01-02 08:00:00 │              │ ← User 1 visits loc 42 again
│  2       │ 55          │ 2025-01-01 09:00:00 │              │ ← Different user
│  2       │ 55          │ 2025-01-01 14:00:00 │              │
│  2       │ 23          │ 2025-01-01 20:00:00 │              │
│  ...     │ ...         │ ...                 │              │
│                                                              │
│  Interpretation:                                             │
│  - Row 1: User 1 visited location 42                        │
│  - Row 2: User 1 visited location 42 again                  │
│  - Each row = 1 visit                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2.4 Data Statistics

| Statistic | Geolife | DIY |
|-----------|---------|-----|
| Total rows (visits) | 19,191 | 265,621 |
| Unique users | 91 | 1,306 |
| Unique locations | 2,049 | 8,439 |
| Avg visits per user | 211 | 203 |
| Avg locations per user | 23 | 29 |

---

## 3. Intermediate Data Structures

### 3.1 After Visit Counting

**Structure:** One row per (user, location) pair

```
┌─────────────────────────────────────────────────────────────┐
│  VISIT COUNT DATA                                            │
│                                                              │
│  user_id │ location_id │ visit_count                        │
│  ────────┼─────────────┼─────────────                       │
│  1       │ 42          │ 50           ← User 1: loc 42      │
│  1       │ 17          │ 30           ← User 1: loc 17      │
│  1       │ 89          │ 10           ← User 1: loc 89      │
│  2       │ 55          │ 40           ← User 2: loc 55      │
│  2       │ 23          │ 25           ← User 2: loc 23      │
│  ...     │ ...         │ ...                                 │
│                                                              │
│  One row per unique (user, location) combination            │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 After Ranking

**Structure:** Adds rank and probability columns

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  RANKED DATA WITH PROBABILITIES                                              │
│                                                                              │
│  user_id │ location_id │ visit_count │ rank │ total_visits │ probability   │
│  ────────┼─────────────┼─────────────┼──────┼──────────────┼─────────────  │
│  1       │ 42          │ 50          │ 1    │ 100          │ 0.50          │
│  1       │ 17          │ 30          │ 2    │ 100          │ 0.30          │
│  1       │ 89          │ 10          │ 3    │ 100          │ 0.10          │
│  1       │ 05          │ 5           │ 4    │ 100          │ 0.05          │
│  1       │ 23          │ 3           │ 5    │ 100          │ 0.03          │
│  1       │ 77          │ 2           │ 6    │ 100          │ 0.02          │
│  ────────┼─────────────┼─────────────┼──────┼──────────────┼─────────────  │
│  2       │ 55          │ 40          │ 1    │ 80           │ 0.50          │
│  2       │ 23          │ 25          │ 2    │ 80           │ 0.31          │
│  2       │ 11          │ 15          │ 3    │ 80           │ 0.19          │
│  ...     │ ...         │ ...         │ ...  │ ...          │ ...           │
│                                                                              │
│  Sorted by (user_id, visit_count DESC)                                      │
│  rank = 1, 2, 3, ... within each user                                       │
│  probability = visit_count / total_visits                                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3.3 User Groups

**Structure:** Users assigned to groups based on n_unique_locations

```
┌─────────────────────────────────────────────────────────────┐
│  USER GROUP ASSIGNMENT                                       │
│                                                              │
│  Group n_L=5 (range [4,6]):                                 │
│  ┌────────┬───────────────────┬──────────────┐              │
│  │user_id │ n_unique_locations│ total_visits │              │
│  ├────────┼───────────────────┼──────────────┤              │
│  │ 3      │ 5                 │ 75           │              │
│  │ 7      │ 6                 │ 92           │              │
│  │ 12     │ 4                 │ 48           │              │
│  │ ...    │ ...               │ ...          │              │
│  └────────┴───────────────────┴──────────────┘              │
│                                                              │
│  Group n_L=10 (range [8,12]):                               │
│  ┌────────┬───────────────────┬──────────────┐              │
│  │user_id │ n_unique_locations│ total_visits │              │
│  ├────────┼───────────────────┼──────────────┤              │
│  │ 1      │ 10                │ 156          │              │
│  │ 5      │ 8                 │ 103          │              │
│  │ 9      │ 12                │ 211          │              │
│  │ ...    │ ...               │ ...          │              │
│  └────────┴───────────────────┴──────────────┘              │
│                                                              │
│  ... similar for n_L=30 and n_L=50                          │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 Pivot Table (for statistics)

**Structure:** Users × Ranks matrix of probabilities

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  PIVOT TABLE: Group n_L=5                                                    │
│                                                                              │
│              │ rank=1 │ rank=2 │ rank=3 │ rank=4 │ rank=5 │ rank=6         │
│  ────────────┼────────┼────────┼────────┼────────┼────────┼─────────       │
│  user_3      │ 0.45   │ 0.28   │ 0.15   │ 0.08   │ 0.04   │   -            │
│  user_7      │ 0.52   │ 0.22   │ 0.12   │ 0.08   │ 0.04   │ 0.02          │
│  user_12     │ 0.60   │ 0.25   │ 0.10   │ 0.05   │   -    │   -            │
│  user_15     │ 0.48   │ 0.30   │ 0.13   │ 0.06   │ 0.03   │   -            │
│  ────────────┼────────┼────────┼────────┼────────┼────────┼─────────       │
│  Mean        │ 0.513  │ 0.263  │ 0.125  │ 0.068  │ 0.037  │ 0.020         │
│  Std         │ 0.063  │ 0.035  │ 0.022  │ 0.015  │ 0.007  │   -            │
│  n_users     │ 4      │ 4      │ 4      │ 4      │ 3      │ 1              │
│  Std Error   │ 0.032  │ 0.018  │ 0.011  │ 0.008  │ 0.004  │   -            │
│                                                                              │
│  Note: '-' indicates missing value (user has fewer locations)               │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Output Files

### 4.1 Output File Structure

```
scripts/analysis_returner_2/
├── geolife_zipf_location_frequency.png    ← Plot
├── geolife_zipf_data_stats.csv            ← Statistics
├── geolife_zipf_data_user_groups.csv      ← User assignments
├── geolife_zipf_data.csv                  ← Detailed data
├── diy_zipf_location_frequency.png        ← Plot
├── diy_zipf_data_stats.csv                ← Statistics
├── diy_zipf_data_user_groups.csv          ← User assignments
├── diy_zipf_data.csv                      ← Detailed data
└── comparison_zipf_location_frequency.png ← Comparison plot
```

### 4.2 Statistics CSV (`*_stats.csv`)

**Purpose:** Group-level statistics for plotting and analysis

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `n_locations_group` | int | Target group (5, 10, 30, 50) |
| `rank` | int | Location rank (1, 2, 3, ...) |
| `mean_prob` | float | Mean P(L) across users |
| `std_error` | float | Standard error of P(L) |
| `n_users` | int | Number of users contributing |

**Example:**
```csv
n_locations_group,rank,mean_prob,std_error,n_users
5,1,0.6426,0.0201,95
5,2,0.2315,0.0125,95
5,3,0.0823,0.0068,95
5,4,0.0311,0.0035,88
5,5,0.0125,0.0018,72
10,1,0.5462,0.0128,230
10,2,0.2156,0.0072,230
...
```

**Visual Structure:**
```
┌────────────────────────────────────────────────────────────────────────┐
│  *_stats.csv                                                           │
│                                                                        │
│  n_locations_group │ rank │ mean_prob │ std_error │ n_users           │
│  ──────────────────┼──────┼───────────┼───────────┼─────────          │
│  5                 │ 1    │ 0.6426    │ 0.0201    │ 95                │
│  5                 │ 2    │ 0.2315    │ 0.0125    │ 95                │
│  5                 │ 3    │ 0.0823    │ 0.0068    │ 95                │
│  ...               │ ...  │ ...       │ ...       │ ...               │
│  10                │ 1    │ 0.5462    │ 0.0128    │ 230               │
│  10                │ 2    │ 0.2156    │ 0.0072    │ 230               │
│  ...               │ ...  │ ...       │ ...       │ ...               │
│                                                                        │
│  One row per (group, rank) combination                                │
│  This file is used to generate the plots                              │
└────────────────────────────────────────────────────────────────────────┘
```

### 4.3 User Groups CSV (`*_user_groups.csv`)

**Purpose:** Track which users are in which groups

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `n_locations_group` | int | Assigned group |
| `user_id` | int | User identifier |
| `n_unique_locations` | int | Actual number of unique locations |
| `visit_count` | int | Total visits by user |

**Example:**
```csv
n_locations_group,user_id,n_unique_locations,visit_count
5,23,5,75
5,45,6,92
5,67,4,48
10,12,10,156
10,34,8,103
...
```

### 4.4 Detailed Data CSV (`*_data.csv`)

**Purpose:** Full per-user, per-location probabilities

**Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `n_locations_group` | int | Group assignment |
| `user_id` | int | User identifier |
| `location_id` | int | Location identifier |
| `rank` | int | Rank of this location for this user |
| `probability` | float | p_u(L) for this user and rank |
| `n_unique_locations` | int | Total unique locations for user |

**Example:**
```csv
n_locations_group,user_id,location_id,rank,probability,n_unique_locations
5,23,42,1,0.45,5
5,23,17,2,0.28,5
5,23,89,3,0.15,5
5,23,05,4,0.08,5
5,23,23,5,0.04,5
5,45,55,1,0.52,6
...
```

---

## 5. Data Flow Diagram

```
┌────────────────────────────────────────────────────────────────────────────────┐
│                            COMPLETE DATA FLOW                                   │
│                                                                                 │
│  INPUT                                                                          │
│  ────────────────────────────────────────────────────────                      │
│  intermediate_eps{X}.csv                                                        │
│      │                                                                          │
│      │  Columns: user_id, location_id, ...                                     │
│      │  Each row = 1 visit                                                     │
│      ▼                                                                          │
│  ┌──────────────────────────────┐                                              │
│  │  load_intermediate_data()    │                                              │
│  │  - Read CSV                  │                                              │
│  │  - Extract user_id, loc_id   │                                              │
│  └──────────────────────────────┘                                              │
│      │                                                                          │
│      ▼                                                                          │
│  ┌──────────────────────────────┐                                              │
│  │compute_user_location_freq()  │                                              │
│  │  - groupby → count visits    │                                              │
│  │  - sort → assign ranks       │                                              │
│  │  - compute probabilities     │                                              │
│  └──────────────────────────────┘                                              │
│      │                                                                          │
│      ▼                                                                          │
│  ┌──────────────────────────────┐                                              │
│  │  assign_user_groups()        │                                              │
│  │  - Filter by n_unique_loc    │                                              │
│  │  - Create groups: 5,10,30,50 │                                              │
│  └──────────────────────────────┘                                              │
│      │                                                                          │
│      ▼                                                                          │
│  ┌──────────────────────────────┐                                              │
│  │  compute_group_statistics()  │                                              │
│  │  - Pivot: users × ranks      │                                              │
│  │  - Mean, Std, SE per rank    │                                              │
│  └──────────────────────────────┘                                              │
│      │                                                                          │
│      ├───────────────────────────────────────────┐                             │
│      │                                           │                             │
│      ▼                                           ▼                             │
│  ┌──────────────────────────────┐   ┌──────────────────────────────┐          │
│  │  fit_reference_line()        │   │  save_results_data()         │          │
│  │  - Log-space fitting         │   │  - *_stats.csv               │          │
│  │  - Compute c in c×L^(-1)     │   │  - *_user_groups.csv         │          │
│  └──────────────────────────────┘   │  - *_data.csv                │          │
│      │                              └──────────────────────────────┘          │
│      │                                                                          │
│      ▼                                                                          │
│  ┌──────────────────────────────┐                                              │
│  │  plot_zipf()                 │                                              │
│  │  - Log-log main panel        │                                              │
│  │  - Linear inset              │                                              │
│  │  - Reference line            │                                              │
│  │  - Save PNG                  │                                              │
│  └──────────────────────────────┘                                              │
│      │                                                                          │
│      ▼                                                                          │
│  OUTPUT                                                                         │
│  ────────────────────────────────────────────────────────                      │
│  *_zipf_location_frequency.png                                                 │
│  *_stats.csv                                                                   │
│  *_user_groups.csv                                                             │
│  *_data.csv                                                                    │
│                                                                                 │
└────────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Data Validation

### 6.1 Probability Conservation

For each user, probabilities must sum to 1:

```
Σ_L p_u(L) = 1.0

Validation:
  ✓ Geolife: All 91 users sum to 1.0000
  ✓ DIY: All 1,306 users sum to 1.0000
```

### 6.2 Rank Ordering

Probabilities must be non-increasing:

```
p_u(1) ≥ p_u(2) ≥ p_u(3) ≥ ...

Validation:
  ✓ Enforced by sorting before rank assignment
```

### 6.3 Group Membership

Users assigned to correct groups:

```
For group n_L=10 (range [8,12]):
  ✓ All users have 8 ≤ n_unique_locations ≤ 12
```

---

## 7. Using the Output Data

### 7.1 Reproducing the Plot

```python
import pandas as pd
import matplotlib.pyplot as plt

# Load statistics
stats = pd.read_csv('diy_zipf_data_stats.csv')

# Plot each group
for n in [5, 10, 30, 50]:
    group_data = stats[stats['n_locations_group'] == n]
    plt.loglog(group_data['rank'], group_data['mean_prob'], 
               marker='o', label=f'{n} loc.')

plt.xlabel('L (rank)')
plt.ylabel('P(L)')
plt.legend()
plt.savefig('reproduced_plot.png')
```

### 7.2 Custom Analysis

```python
import pandas as pd

# Load detailed data
data = pd.read_csv('diy_zipf_data.csv')

# Find users who spend >50% at top location
high_concentration = data[(data['rank'] == 1) & (data['probability'] > 0.5)]
print(f"Users with P(1) > 50%: {high_concentration['user_id'].nunique()}")

# Average probability at rank 1 by group
rank1 = data[data['rank'] == 1].groupby('n_locations_group')['probability'].mean()
print(rank1)
```

---

*Next: [05_RESULTS_ANALYSIS.md](./05_RESULTS_ANALYSIS.md) - Comprehensive analysis of results*
