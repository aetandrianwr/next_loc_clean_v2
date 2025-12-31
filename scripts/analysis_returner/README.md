# Return Probability Distribution Analysis

This directory contains the analysis script and results for reproducing the **return probability distribution** (F_pt(t)) plot from **González et al. (2008), Figure 2c**.

## Overview

The analysis computes the **first-return time distribution** for users:

- For each user, identify their **first observed location** (L₀) at time (t₀)
- Find the **first later event** where the user returns to L₀ (time t₁ > t₀)
- Compute the first-return time: Δt = (t₁ - t₀) in hours
- Create a histogram of Δt values across all users
- Convert to probability density: F_pt(t) = count / (N_returns × bin_width)

## Files

### Scripts
- `return_probability_analysis.py` - Main analysis script

### Outputs

#### Geolife Dataset
- `geolife_return_probability.png` - Return probability distribution plot
- `geolife_return_probability_data.csv` - Probability density data (t_hours, F_pt)
- `geolife_return_probability_data_returns.csv` - Individual user return times

#### DIY Dataset
- `diy_return_probability.png` - Return probability distribution plot
- `diy_return_probability_data.csv` - Probability density data (t_hours, F_pt)
- `diy_return_probability_data_returns.csv` - Individual user return times

## Usage

### Basic Usage

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Run analysis with default parameters
cd /data/next_loc_clean_v2
python scripts/analysis_returner/return_probability_analysis.py
```

### Custom Parameters

```bash
# Run with custom bin width and max time
python scripts/analysis_returner/return_probability_analysis.py \
    --bin-width 1.0 \
    --max-hours 240 \
    --output-dir scripts/analysis_returner
```

### Arguments

- `--bin-width`: Histogram bin width in hours (default: 2.0)
- `--max-hours`: Maximum return time to analyze in hours (default: 240)
- `--output-dir`: Output directory for results (default: scripts/analysis_returner)

## Data Sources

The analysis uses intermediate data from the preprocessing pipeline:

- **Geolife**: `data/geolife_eps20/interim/intermediate_eps20.csv`
- **DIY**: `data/diy_eps50/interim/intermediate_eps50.csv`

This data is taken right after the **"Encode locations"** step (Step 2/5) in the preprocessing scripts, combining all cleaned events from train, validation, and test splits.

### Required Columns

- `user_id`: User identifier (string/int)
- `location_id`: Location identifier (string/int)
- `start_day`: Day number since tracking started
- `start_min`: Minute of day when staypoint started (0-1439)

Timestamp is computed as: `timestamp_hours = (start_day × 1440 + start_min) / 60`

## Results Summary

### Geolife Dataset (eps=20)
- Total events: 19,191
- Total users: 91
- Users with returns: 49 (53.85%)
- Mean return time: 58.96 hours
- Median return time: 35.28 hours

### DIY Dataset (eps=50)
- Total events: 265,621
- Total users: 1,306
- Users with returns: 1,091 (83.54%)
- Mean return time: 60.02 hours
- Median return time: 42.77 hours

## Plot Styling

The plots match the style of González et al. (2008) Figure 2c:

- **Figure size**: 8×6 inches (roughly square/portrait)
- **Background**: Plain white
- **Curve**: Blue dashed line (`b--`)
- **X-axis**: Time in hours (t), ticks at 0, 24, 48, ..., 240
- **Y-axis**: Probability density F_pt(t)
- **Legend**: "Users" at top-right
- **Spines**: Top and right spines hidden for clean appearance

## Implementation Details

### Vectorized Computation

The script uses **pandas vectorized operations** to avoid per-row Python loops:

1. Sort all events by `user_id` and `timestamp_hours`
2. Use `groupby().first()` to get each user's first location
3. Merge to add first location info to all events
4. Filter to events after first event: `timestamp > first_time`
5. Filter to returns: `location == first_location`
6. Use `groupby().first()` again to get earliest return for each user
7. Compute Δt vectorially: `timestamp - first_time`

### Probability Density Computation

```python
# Create histogram
counts, bin_edges = np.histogram(delta_t_values, bins=bins)

# Convert to probability density
pdf = counts / (n_returns * bin_width_hours)
```

## Dependencies

- Python 3.x
- pandas
- numpy
- matplotlib

All dependencies are available in the `mlenv` conda environment.

## References

González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008). Understanding individual human mobility patterns. *Nature*, 453(7196), 779-782.

## Author

Data Scientist  
Date: December 31, 2025
