# Code Walkthrough: Line-by-Line Explanation

## 1. Introduction

This document provides a comprehensive, line-by-line explanation of the analysis scripts. We use a **consistent example** throughout to make the concepts concrete and easy to follow.

### Consistent Example: Alice's Mobility Data

Throughout this document, we will follow **Alice** (user_id=1):

```
┌─────────────────────────────────────────────────────────────────────┐
│ ALICE'S TRAJECTORY                                                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│ Day 1, 08:00: HOME (loc_id=100)     ← First observation (L₀)        │
│ Day 1, 09:15: WORK (loc_id=200)                                     │
│ Day 1, 18:30: HOME (loc_id=100)     ← First return! Δt = 10.5h     │
│ Day 2, 07:45: HOME (loc_id=100)                                     │
│ Day 2, 09:00: WORK (loc_id=200)                                     │
│ ...                                                                  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Script Structure Overview

### 2.1 File: `return_probability_analysis_v2.py`

```
return_probability_analysis_v2.py
├── Imports and Dependencies (Lines 1-21)
├── load_intermediate_data() (Lines 24-52)
├── compute_first_return_times() (Lines 55-124)
├── compute_probability_density() (Lines 127-160)
├── compute_random_walk_baseline() (Lines 163-182)
├── plot_return_probability() (Lines 185-262)
├── save_results_data() (Lines 265-291)
├── analyze_dataset() (Lines 294-340)
└── main() (Lines 343-410)
```

---

## 3. Detailed Code Walkthrough

### 3.1 Imports and Header (Lines 1-21)

```python
"""
Return Probability Distribution Analysis
Reproduces Figure 2c from González et al. (2008)

Computes the first-return time distribution F_pt(t) for users.
For each user, finds their first observed location L0 at time t0,
then finds the first later event where the user returns to L0 (time t1 > t0).
The first-return time is Δt = (t1 - t0) in hours.

Author: Data Scientist
Date: 2025-12-31
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

**Explanation**:
- **Docstring**: Explains the purpose (reproducing González et al. 2008)
- **Standard imports**: `os`, `sys`, `argparse` for file handling and CLI
- **Scientific imports**: `numpy`, `pandas` for data processing; `matplotlib` for plotting

---

### 3.2 Function: `load_intermediate_data()` (Lines 24-52)

```python
def load_intermediate_data(dataset_path):
    """
    Load intermediate CSV data from preprocessing.
    
    Parameters
    ----------
    dataset_path : str
        Path to the intermediate CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, location_id, timestamp (in minutes)
    """
    print(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Compute timestamp in minutes from start_day and start_min
    # timestamp = start_day * 1440 (minutes per day) + start_min
    df['timestamp_minutes'] = df['start_day'] * 1440 + df['start_min']
    
    # Convert to hours for easier interpretation
    df['timestamp_hours'] = df['timestamp_minutes'] / 60.0
    
    print(f"Loaded {len(df):,} events from {df['user_id'].nunique():,} users")
    print(f"Time range: {df['timestamp_hours'].min():.2f}h to {df['timestamp_hours'].max():.2f}h")
    print(f"Unique locations: {df['location_id'].nunique():,}")
    
    return df[['user_id', 'location_id', 'timestamp_hours']].copy()
```

#### Line-by-Line Explanation:

| Line | Code | Explanation |
|------|------|-------------|
| 38 | `df = pd.read_csv(dataset_path)` | Load CSV file into pandas DataFrame |
| 42 | `df['timestamp_minutes'] = df['start_day'] * 1440 + df['start_min']` | Convert day+minute to absolute minutes |
| 45 | `df['timestamp_hours'] = df['timestamp_minutes'] / 60.0` | Convert minutes to hours |
| 52 | `return df[['user_id', 'location_id', 'timestamp_hours']].copy()` | Return only needed columns |

#### Example with Alice's Data:

**Input CSV** (intermediate_eps20.csv):
```
user_id,location_id,start_day,start_min
1,100,0,480        # Alice, HOME, Day 0, 08:00 (480 min)
1,200,0,555        # Alice, WORK, Day 0, 09:15 (555 min)
1,100,0,1110       # Alice, HOME, Day 0, 18:30 (1110 min)
```

**Processing**:
```
timestamp_minutes = start_day * 1440 + start_min
                  = 0 * 1440 + 480
                  = 480 minutes

timestamp_hours = 480 / 60.0 = 8.0 hours
```

**Output DataFrame**:
```
   user_id  location_id  timestamp_hours
0        1          100              8.0    # Alice at HOME
1        1          200              9.25   # Alice at WORK
2        1          100             18.5    # Alice at HOME
```

---

### 3.3 Function: `compute_first_return_times()` (Lines 55-124)

This is the **core algorithm**. Let's break it down:

```python
def compute_first_return_times(df, bin_width_hours=2.0, max_hours=240):
    """
    Compute first-return time distribution for all users.
    
    For each user:
    - Identify first location L0 at time t0
    - Find first later event where location == L0 (time t1 > t0)
    - Record Δt = t1 - t0 in hours
    """
    print("\n" + "="*60)
    print("Computing first-return times...")
    print("="*60)
    
    # Step 1: Sort by user_id and timestamp
    df_sorted = df.sort_values(['user_id', 'timestamp_hours']).reset_index(drop=True)
```

#### Step 1: Sort Data

**Purpose**: Ensure chronological order within each user's trajectory.

```
BEFORE SORTING:                    AFTER SORTING:
user_id  location  time            user_id  location  time
   1        100      18.5            1         100       8.0  ← First
   1        200       9.25           1         200       9.25
   1        100       8.0            1         100      18.5
   2        300       5.0            2         300       5.0  ← First
   2        300      29.0            2         300      29.0
```

```python
    # Step 2: Group by user and get first location for each user
    first_events = df_sorted.groupby('user_id').first().reset_index()
    first_events = first_events.rename(columns={
        'location_id': 'first_location',
        'timestamp_hours': 'first_time'
    })
    
    print(f"Total users: {len(first_events):,}")
```

#### Step 2: Get First Location Per User

**Purpose**: For each user, find their first observed location (L₀) and time (t₀).

```
GROUPED FIRST EVENTS:
user_id  first_location  first_time
   1          100            8.0      # Alice: HOME at 08:00
   2          300            5.0      # Bob: OFFICE at 05:00
```

```python
    # Step 3: Merge to get first location info for all events
    df_with_first = df_sorted.merge(
        first_events[['user_id', 'first_location', 'first_time']],
        on='user_id',
        how='left'
    )
```

#### Step 3: Merge First Location Info

**Purpose**: Add first_location and first_time columns to all events.

```
MERGED DATAFRAME:
user_id  location_id  timestamp_hours  first_location  first_time
   1         100            8.0             100            8.0
   1         200            9.25            100            8.0
   1         100           18.5             100            8.0
   2         300            5.0             300            5.0
   2         300           29.0             300            5.0
```

```python
    # Step 4: Filter to events after first event (timestamp > first_time)
    df_later = df_with_first[df_with_first['timestamp_hours'] > df_with_first['first_time']].copy()
```

#### Step 4: Filter Later Events

**Purpose**: Keep only events that occurred AFTER the first observation.

```
LATER EVENTS (timestamp > first_time):
user_id  location_id  timestamp_hours  first_location  first_time
   1         200            9.25            100            8.0     # Keep (9.25 > 8.0)
   1         100           18.5             100            8.0     # Keep (18.5 > 8.0)
   2         300           29.0             300            5.0     # Keep (29.0 > 5.0)

REMOVED: First events (timestamp == first_time)
```

```python
    # Step 5: Filter to returns (location == first_location)
    df_returns = df_later[df_later['location_id'] == df_later['first_location']].copy()
```

#### Step 5: Filter to Returns

**Purpose**: Keep only events where user returned to their first location.

```
RETURNS (location == first_location):
user_id  location_id  timestamp_hours  first_location  first_time
   1         100           18.5             100            8.0     # Return! (100==100)
   2         300           29.0             300            5.0     # Return! (300==300)

REMOVED: Non-returns (location ≠ first_location)
   1         200            9.25            100            8.0     # Not return (200≠100)
```

```python
    # Step 6: Get first return for each user (earliest timestamp after first event)
    first_returns = df_returns.groupby('user_id').first().reset_index()
    
    # Step 7: Compute delta_t
    first_returns['delta_t_hours'] = first_returns['timestamp_hours'] - first_returns['first_time']
    
    # Step 8: Filter to max_hours
    first_returns = first_returns[first_returns['delta_t_hours'] <= max_hours].copy()
```

#### Step 6-8: Compute Return Times

**Purpose**: Get the FIRST return for each user and compute time difference.

```
FIRST RETURNS WITH DELTA_T:
user_id  timestamp_hours  first_time  delta_t_hours
   1          18.5           8.0          10.5       # Alice: 18.5 - 8.0 = 10.5 hours
   2          29.0           5.0          24.0       # Bob: 29.0 - 5.0 = 24.0 hours
```

```python
    print(f"Users with returns: {len(first_returns):,}")
    print(f"Return rate: {len(first_returns) / len(first_events) * 100:.2f}%")
    print(f"Mean return time: {first_returns['delta_t_hours'].mean():.2f}h")
    print(f"Median return time: {first_returns['delta_t_hours'].median():.2f}h")
    print(f"Min return time: {first_returns['delta_t_hours'].min():.2f}h")
    print(f"Max return time: {first_returns['delta_t_hours'].max():.2f}h")
    
    return first_returns
```

#### Statistics Output Example:
```
Users with returns: 2
Return rate: 100.00%
Mean return time: 17.25h
Median return time: 17.25h
Min return time: 10.5h
Max return time: 24.0h
```

---

### 3.4 Function: `compute_probability_density()` (Lines 127-160)

```python
def compute_probability_density(delta_t_values, bin_width_hours=2.0, max_hours=240):
    """
    Convert return times to probability density F_pt(t).
    
    Parameters
    ----------
    delta_t_values : np.array
        Array of first-return times in hours
    bin_width_hours : float
        Width of histogram bins in hours
    max_hours : int
        Maximum time in hours
        
    Returns
    -------
    bin_centers : np.array
        Centers of histogram bins (x-axis)
    pdf : np.array
        Probability density values (y-axis)
    """
    # Create bins
    bins = np.arange(0, max_hours + bin_width_hours, bin_width_hours)
    
    # Compute histogram
    counts, bin_edges = np.histogram(delta_t_values, bins=bins)
    
    # Convert to probability density: count / (N_returns * bin_width)
    n_returns = len(delta_t_values)
    pdf = counts / (n_returns * bin_width_hours)
    
    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return bin_centers, pdf
```

#### Line-by-Line Explanation:

| Line | Code | Explanation |
|------|------|-------------|
| 148 | `bins = np.arange(0, max_hours + bin_width_hours, bin_width_hours)` | Create bin edges: [0, 2, 4, ..., 240, 242] |
| 151 | `counts, bin_edges = np.histogram(delta_t_values, bins=bins)` | Count values in each bin |
| 155 | `pdf = counts / (n_returns * bin_width_hours)` | Normalize to probability density |
| 158 | `bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0` | Compute center of each bin |

#### Visual Example:

```
INPUT: delta_t_values = [10.5, 24.0, 23.5, 25.0, 48.5, 72.0]

HISTOGRAM BINS (bin_width=2h):
Bin        [0,2)  [2,4)  ...  [10,12)  ...  [22,24)  [24,26)  ...
Counts       0      0    ...     1     ...     1        2     ...

                              ↑ Alice         ↑              ↑
                           (10.5h)       (23.5h)      (24.0h, 25.0h)

PROBABILITY DENSITY:
pdf = counts / (n_returns * bin_width)
    = [0, 0, ..., 1, ..., 1, 2, ...] / (6 * 2)
    = [0, 0, ..., 0.083, ..., 0.083, 0.167, ...]

BIN CENTERS:
[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, ...]
```

---

### 3.5 Function: `compute_random_walk_baseline()` (Lines 163-182)

```python
def compute_random_walk_baseline(bin_centers, initial_prob=0.01):
    """
    Compute a simple exponential decay baseline (RW - Random Walk model).
    
    Parameters
    ----------
    bin_centers : np.array
        Time bins in hours
    initial_prob : float
        Initial probability value
        
    Returns
    -------
    np.array
        Random walk baseline probability values
    """
    # Simple exponential decay: P(t) = P0 * exp(-t/tau)
    tau = 30.0  # decay constant in hours
    rw = initial_prob * np.exp(-bin_centers / tau)
    return rw
```

#### Explanation:

The Random Walk baseline models what would happen if users moved randomly:

```
F_RW(t) = P₀ × exp(-t/τ)

Where:
• P₀ = 0.01 (initial probability)
• τ = 30 hours (decay constant)
```

**Visual Comparison**:
```
F_pt(t)
   ▲
   │     ∿∿∿ Users (real data - with peaks)
   │    ∿   ∿
   │   ∿     ∿∿
   │  ∿        ∿∿∿
   │ ╲            ∿∿∿∿∿∿ (remains elevated)
   │  ╲╲ RW (exponential decay)
   │    ╲╲╲
   │       ╲╲╲╲________ (decays to near zero)
   └────────────────────────────► t (hours)
      0   24   48   72   96  120

KEY INSIGHT: Users stay ABOVE RW baseline → Non-random behavior
```

---

### 3.6 Function: `plot_return_probability()` (Lines 185-262)

```python
def plot_return_probability(bin_centers, pdf, dataset_name, output_path, 
                            max_hours=240, bin_width_hours=2.0):
    """
    Create the return probability distribution plot matching reference style.
    """
    # Create figure with exact reference styling
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Compute RW baseline
    rw_baseline = compute_random_walk_baseline(bin_centers)
    
    # Plot Users data (blue dashed line)
    ax.plot(bin_centers, pdf, color='#3366CC', linestyle='--', 
            linewidth=2.5, label='Users', dash_capstyle='round')
    
    # Plot RW baseline (black solid line)
    ax.plot(bin_centers, rw_baseline, 'k-', linewidth=2, label='RW')
    
    # Axis labels with larger font
    ax.set_xlabel('$t$ (h)', fontsize=20, labelpad=8)
    ax.set_ylabel('$F_{pt}(t)$', fontsize=20, labelpad=8)
    
    # Set x-axis ticks at 24-hour intervals
    x_ticks = np.arange(0, max_hours + 1, 24)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=16, 
                   width=1.5, length=6)
    
    # Set axis limits
    ax.set_xlim(0, max_hours)
    ax.set_ylim(0, 0.026)
    
    # Legend with custom styling
    legend = ax.legend(loc='upper right', fontsize=18, frameon=True, 
                      framealpha=1.0, edgecolor='black', fancybox=False)
    legend.get_frame().set_linewidth(1.5)
    
    # Thicker spines (all visible)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_edgecolor('black')
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', pad_inches=0.1)
    print(f"\n✓ Saved plot to: {output_path}")
    
    plt.close()
```

#### Plot Elements Explained:

```
┌─────────────────────────────────────────────────────────────────────┐
│                                           ┌─────────────┐           │
│                                           │  Legend     │           │
│                                           │  -----      │           │
│   ↑                                       │  --- Users  │           │
│ F_pt(t)                                   │  ── RW      │           │
│   │                                       └─────────────┘           │
│   │     ∿∿∿  Blue dashed line = Users (actual data)                │
│   │    ∿   ∿                                                        │
│   │   ∿     ∿                                                       │
│   │  ∿       ∿∿                                                     │
│   │ ╲          ∿∿∿                                                  │
│   │  ╲  Black solid line = RW baseline                              │
│   │   ╲╲                                                            │
│   │     ╲╲╲______                                                   │
│   └────────────────────────────────────────────────────────► t (h) │
│       0    24    48    72    96   120   144   168   ...  240       │
│                                                                      │
│   X-axis: Time in hours, ticks every 24 hours                       │
│   Y-axis: Probability density F_pt(t)                               │
└─────────────────────────────────────────────────────────────────────┘
```

---

### 3.7 Function: `analyze_dataset()` (Lines 294-340)

```python
def analyze_dataset(dataset_path, dataset_name, output_dir, 
                    bin_width_hours=2.0, max_hours=240):
    """
    Complete analysis pipeline for one dataset.
    """
    print("\n" + "="*80)
    print(f"ANALYZING: {dataset_name}")
    print("="*80)
    
    # Step 1: Load data
    df = load_intermediate_data(dataset_path)
    
    # Step 2: Compute first-return times
    first_returns = compute_first_return_times(df, bin_width_hours, max_hours)
    
    # Step 3: Compute probability density
    bin_centers, pdf = compute_probability_density(
        first_returns['delta_t_hours'].values,
        bin_width_hours,
        max_hours
    )
    
    # Step 4: Plot
    plot_file = os.path.join(output_dir, f'{dataset_name.lower()}_return_probability_v2.png')
    plot_return_probability(bin_centers, pdf, dataset_name, plot_file, 
                           max_hours, bin_width_hours)
    
    # Step 5: Save data
    data_file = os.path.join(output_dir, f'{dataset_name.lower()}_return_probability_data_v2.csv')
    save_results_data(first_returns, bin_centers, pdf, data_file)
    
    print(f"\n✓ Analysis complete for {dataset_name}")
    
    return first_returns, bin_centers, pdf
```

#### Pipeline Flow:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                         ANALYSIS PIPELINE                                 │
└──────────────────────────────────────────────────────────────────────────┘

 ┌─────────────────┐
 │ Raw CSV Data    │
 │ intermediate_   │
 │ eps20.csv       │
 └────────┬────────┘
          │
          ▼ Step 1: load_intermediate_data()
 ┌─────────────────┐
 │ DataFrame       │
 │ user_id,        │
 │ location_id,    │
 │ timestamp_hours │
 └────────┬────────┘
          │
          ▼ Step 2: compute_first_return_times()
 ┌─────────────────┐
 │ Return Times    │
 │ user_id,        │
 │ delta_t_hours   │
 └────────┬────────┘
          │
          ▼ Step 3: compute_probability_density()
 ┌─────────────────┐     ┌─────────────────┐
 │ bin_centers     │     │ pdf             │
 │ [1, 3, 5, ...]  │     │ [0.01, 0.02,...]│
 └────────┬────────┘     └────────┬────────┘
          │                       │
          └───────────┬───────────┘
                      │
                      ▼ Step 4 & 5: plot & save
          ┌──────────────────────────┐
          │ geolife_return_prob_v2.png │
          │ geolife_return_prob_data.csv │
          └──────────────────────────┘
```

---

### 3.8 Function: `main()` (Lines 343-410)

```python
def main():
    parser = argparse.ArgumentParser(
        description='Compute return probability distribution (González et al. 2008, Figure 2c)'
    )
    parser.add_argument(
        '--bin-width',
        type=float,
        default=2.0,
        help='Histogram bin width in hours (default: 2.0)'
    )
    parser.add_argument(
        '--max-hours',
        type=int,
        default=240,
        help='Maximum return time in hours (default: 240)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts/analysis_returner',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset paths
    geolife_path = 'data/geolife_eps20/interim/intermediate_eps20.csv'
    diy_path = 'data/diy_eps50/interim/intermediate_eps50.csv'
    
    print("="*80)
    print("RETURN PROBABILITY DISTRIBUTION ANALYSIS")
    print("Reproducing González et al. (2008) Figure 2c")
    print("="*80)
    print(f"Bin width: {args.bin_width} hours")
    print(f"Max time: {args.max_hours} hours")
    print(f"Output directory: {output_dir}")
    
    # Analyze Geolife
    if os.path.exists(geolife_path):
        geolife_results = analyze_dataset(
            geolife_path, 'Geolife', output_dir,
            args.bin_width, args.max_hours
        )
    else:
        print(f"\n⚠ Geolife data not found at: {geolife_path}")
    
    # Analyze DIY
    if os.path.exists(diy_path):
        diy_results = analyze_dataset(
            diy_path, 'DIY', output_dir,
            args.bin_width, args.max_hours
        )
    else:
        print(f"\n⚠ DIY data not found at: {diy_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
```

---

## 4. Compare Datasets Script (`compare_datasets.py`)

### 4.1 Complete Code Walkthrough

```python
"""
Compare Return Probability Distributions
Plot both Geolife and DIY datasets on the same figure for comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison():
    """Create comparison plot with both datasets."""
    
    # Load pre-computed probability density data
    geolife_pdf = pd.read_csv('geolife_return_probability_data.csv')
    diy_pdf = pd.read_csv('diy_return_probability_data.csv')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot both curves
    plt.plot(geolife_pdf['t_hours'], geolife_pdf['F_pt'], 
             'b--', linewidth=2, label='Geolife', alpha=0.8, marker='o', 
             markersize=3, markevery=5)
    plt.plot(diy_pdf['t_hours'], diy_pdf['F_pt'], 
             'r-', linewidth=2, label='DIY', alpha=0.8, marker='s', 
             markersize=3, markevery=5)
    
    # Styling
    plt.xlabel('t (h)', fontsize=12)
    plt.ylabel('F$_{pt}$(t)', fontsize=12)
    plt.title('Return Probability Distribution - Dataset Comparison', 
              fontsize=14, pad=15)
    
    # Set x-axis ticks at 24-hour intervals
    x_ticks = np.arange(0, 241, 24)
    plt.xticks(x_ticks)
    
    # Set axis limits
    plt.xlim(0, 240)
    plt.ylim(bottom=0)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Save
    plt.savefig('comparison_return_probability.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to: comparison_return_probability.png")
    
    plt.close()
    
    # Print statistics comparison
    geolife_returns = pd.read_csv('geolife_return_probability_data_returns.csv')
    diy_returns = pd.read_csv('diy_return_probability_data_returns.csv')
    
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    print(f"\n{'Metric':<30} {'Geolife':>15} {'DIY':>15}")
    print("-"*70)
    print(f"{'Users with returns':<30} {len(geolife_returns):>15,} {len(diy_returns):>15,}")
    print(f"{'Mean return time (h)':<30} {geolife_returns['delta_t_hours'].mean():>15.2f} {diy_returns['delta_t_hours'].mean():>15.2f}")
    print(f"{'Median return time (h)':<30} {geolife_returns['delta_t_hours'].median():>15.2f} {diy_returns['delta_t_hours'].median():>15.2f}")
```

---

## 5. Summary: Data Flow from Input to Output

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        COMPLETE DATA FLOW                                    │
└─────────────────────────────────────────────────────────────────────────────┘

INPUT (CSV):
┌────────────────────────────────────────┐
│ user_id, location_id, start_day, start_min │
│    1         100          0         480    │
│    1         200          0         555    │
│    1         100          0        1110    │
│   ...        ...         ...        ...    │
└────────────────────────────────────────┘
              │
              ▼ load_intermediate_data()
┌────────────────────────────────────────┐
│ user_id, location_id, timestamp_hours  │
│    1         100           8.0         │
│    1         200           9.25        │
│    1         100          18.5         │
└────────────────────────────────────────┘
              │
              ▼ compute_first_return_times()
┌────────────────────────────────────────┐
│ user_id, delta_t_hours                 │
│    1         10.5                      │  ← Alice returns after 10.5h
│    2         24.0                      │  ← Bob returns after 24.0h
│   ...        ...                       │
└────────────────────────────────────────┘
              │
              ▼ compute_probability_density()
┌────────────────────────────────────────┐
│ bin_centers       pdf                  │
│     1            0.000                 │
│     3            0.001                 │
│    ...           ...                   │
│    23            0.024   ← Peak!       │
│    25            0.020                 │
│    ...           ...                   │
└────────────────────────────────────────┘
              │
              ▼ plot_return_probability()

OUTPUT (PNG):
┌──────────────────────────────────────────┐
│                              ┌──────┐    │
│                              │Legend│    │
│   F_pt(t)                    └──────┘    │
│      ▲                                   │
│      │    ∿∿∿                            │
│      │   ∿   ∿                           │
│      │  ∿     ∿∿∿                        │
│      │ ╲        ∿∿∿∿                     │
│      │  ╲╲         ∿∿∿∿                  │
│      │    ╲╲╲_________                   │
│      └────────────────────────────► t(h) │
│         0   24   48   72  ...  240       │
│                                          │
│    geolife_return_probability_v2.png     │
└──────────────────────────────────────────┘
```

---

*← Back to [Theoretical Background](02_THEORETICAL_BACKGROUND.md) | Continue to [Data Pipeline](04_DATA_PIPELINE.md) →*
