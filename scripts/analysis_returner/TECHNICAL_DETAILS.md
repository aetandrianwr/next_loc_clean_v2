# Technical Implementation Details

## Return Probability Distribution Analysis

This document provides technical details about the implementation of the return probability distribution analysis, reproducing Figure 2c from González et al. (2008).

## Mathematical Definition

### First-Return Time

For each user:
1. Identify the **first observed location** L₀ at time t₀
2. Find the **first later event** where `location_id == L₀` at time t₁ > t₀
3. Compute the **first-return time**: Δt = t₁ - t₀ (in hours)

### Probability Density Function

The return probability distribution F_pt(t) is computed as:

```
F_pt(t) = N(t) / (N_total × Δt)
```

Where:
- N(t) = number of returns in bin [t, t+Δt)
- N_total = total number of users with returns
- Δt = bin width (default: 2 hours)

The probability density integrates to 1:
```
∫₀^∞ F_pt(t) dt ≈ Σ F_pt(tᵢ) × Δt = 1
```

## Data Processing Pipeline

### Step 1: Load Intermediate Data

Input: `intermediate_eps{epsilon}.csv` files from preprocessing pipeline

Columns used:
- `user_id`: User identifier
- `location_id`: Location identifier (staypoint cluster)
- `start_day`: Day number since tracking started
- `start_min`: Minute of day (0-1439)

Timestamp reconstruction:
```python
timestamp_minutes = start_day × 1440 + start_min
timestamp_hours = timestamp_minutes / 60.0
```

### Step 2: Sort Events

Sort all events by:
1. `user_id` (primary key)
2. `timestamp_hours` (secondary key)

This ensures chronological order within each user's trajectory.

### Step 3: Identify First Locations

For each user, extract the first event:
```python
first_events = df_sorted.groupby('user_id').first().reset_index()
```

Creates a lookup table:
- `user_id` → `first_location`, `first_time`

### Step 4: Find First Returns

Vectorized approach (no per-row loops):

```python
# Add first location info to all events
df_with_first = df_sorted.merge(first_events, on='user_id')

# Filter to events after first observation
df_later = df_with_first[df_with_first['timestamp_hours'] > df_with_first['first_time']]

# Filter to returns (same location as first)
df_returns = df_later[df_later['location_id'] == df_later['first_location']]

# Get earliest return for each user
first_returns = df_returns.groupby('user_id').first()
```

### Step 5: Compute Return Times

```python
first_returns['delta_t_hours'] = first_returns['timestamp_hours'] - first_returns['first_time']
```

### Step 6: Create Histogram

```python
bins = np.arange(0, max_hours + bin_width, bin_width)
counts, bin_edges = np.histogram(delta_t_values, bins=bins)
```

### Step 7: Normalize to Probability Density

```python
n_returns = len(delta_t_values)
pdf = counts / (n_returns × bin_width)
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
```

## Performance Optimization

### Vectorization

All operations use **pandas/numpy vectorized methods**:
- `groupby()` for aggregation
- `merge()` for joins
- Boolean indexing for filtering
- No explicit Python loops over rows

### Memory Efficiency

- Load only required columns: `user_id`, `location_id`, `start_day`, `start_min`
- Drop intermediate DataFrames after use
- Use appropriate data types (int64 for IDs, float64 for timestamps)

### Time Complexity

- Sorting: O(N log N) where N = total events
- Groupby operations: O(N)
- Merge: O(N)
- Overall: O(N log N)

For Geolife (19K events): < 1 second
For DIY (265K events): < 2 seconds

## Plot Styling

Matches González et al. (2008) Figure 2c:

```python
plt.figure(figsize=(8, 6))
plt.plot(bin_centers, pdf, 'b--', linewidth=2, label='Users', alpha=0.8)
plt.xlabel('t (h)', fontsize=12)
plt.ylabel('F$_{pt}$(t)', fontsize=12)
plt.xticks(np.arange(0, 241, 24))
plt.xlim(0, 240)
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
plt.legend(loc='upper right', fontsize=11)

# Hide top and right spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
```

## Validation

### Probability Mass Conservation

The probability density should integrate to 1:

```python
total_probability = (pdf × bin_width).sum()
# Should be ≈ 1.0
```

Results:
- Geolife: 1.0000 ✓
- DIY: 1.0000 ✓

### Edge Cases Handled

1. **Users with no returns**: Excluded from analysis
   - Geolife: 42 users (46.15%) have no returns
   - DIY: 215 users (16.46%) have no returns

2. **Return time > max_hours**: Filtered out
   - Only returns within [0, 240] hours included

3. **Zero bin width**: Would cause division by zero
   - Input validation ensures bin_width > 0

4. **Empty dataset**: Would cause errors
   - Check performed before processing

## Output Files

### Individual Return Times
- `{dataset}_return_probability_data_returns.csv`
- Columns: `user_id`, `delta_t_hours`
- One row per user with a return

### Probability Density
- `{dataset}_return_probability_data.csv`
- Columns: `t_hours`, `F_pt`
- One row per histogram bin

### Plots
- `{dataset}_return_probability.png`
- High-resolution PNG (300 DPI)
- Dimensions: 2358 × 1771 pixels

## Comparison with González et al. (2008)

### Similarities
- Same mathematical definition of first-return time
- Same histogram binning approach
- Same probability density normalization
- Similar plot styling

### Differences
- **Data source**: We use GPS trajectory data (Geolife, DIY) vs. mobile phone data
- **Spatial resolution**: Staypoint clusters vs. cell tower locations
- **Time range**: 0-240 hours vs. full range in paper
- **Bin width**: 2 hours (configurable) vs. unspecified in paper

## Parameters

### Configurable via Command Line

```bash
--bin-width FLOAT     Histogram bin width in hours (default: 2.0)
--max-hours INT       Maximum return time in hours (default: 240)
--output-dir PATH     Output directory (default: scripts/analysis_returner)
```

### Recommended Values

- **High resolution**: `--bin-width 1.0` (more detail, noisier)
- **Standard**: `--bin-width 2.0` (balanced)
- **Smooth**: `--bin-width 4.0` (smoother, less detail)

- **Short term**: `--max-hours 120` (5 days)
- **Standard**: `--max-hours 240` (10 days)
- **Long term**: `--max-hours 480` (20 days)

## Code Quality

### Style
- PEP 8 compliant
- Docstrings for all functions
- Type hints in function signatures
- Clear variable names

### Robustness
- Input validation
- Error handling for missing files
- Graceful handling of edge cases
- Informative error messages

### Testing
- Validated on two datasets (Geolife, DIY)
- Probability mass conservation check
- Visual inspection of plots
- Statistical summary verification

## Future Enhancements

Potential improvements:
1. Add confidence intervals (bootstrap)
2. Fit theoretical distributions (exponential, power-law)
3. Compare with null models (random mobility)
4. Stratify by user characteristics
5. Time-of-day analysis (circadian patterns)
6. Location-specific return times

## References

González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008). Understanding individual human mobility patterns. *Nature*, 453(7196), 779-782. doi:10.1038/nature06958

## Contact

For questions or issues, contact the data science team.

Last updated: December 31, 2025
