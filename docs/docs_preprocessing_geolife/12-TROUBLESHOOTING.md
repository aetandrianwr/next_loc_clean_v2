# Troubleshooting Guide

## Table of Contents
1. [Installation Issues](#installation-issues)
2. [Data Loading Errors](#data-loading-errors)
3. [Memory Issues](#memory-issues)
4. [Processing Errors](#processing-errors)
5. [Output Issues](#output-issues)
6. [Configuration Errors](#configuration-errors)
7. [Common Warnings](#common-warnings)
8. [Performance Issues](#performance-issues)
9. [Debugging Guide](#debugging-guide)
10. [FAQ](#faq)

---

## Installation Issues

### Issue 1: `ModuleNotFoundError: No module named 'trackintel'`

**Error Message**:
```
ModuleNotFoundError: No module named 'trackintel'
```

**Cause**: The trackintel library is not installed.

**Solution**:
```bash
# Install trackintel
pip install trackintel

# Or with conda
conda install -c conda-forge trackintel
```

**Verification**:
```python
import trackintel
print(trackintel.__version__)  # Should print version number
```

---

### Issue 2: `ModuleNotFoundError: No module named 'h3'`

**Error Message**:
```
ModuleNotFoundError: No module named 'h3'
```

**Cause**: The H3 library (for hexagonal grid processing) is not installed.

**Solution**:
```bash
# Install h3-py
pip install h3

# For better performance, install with pre-compiled binaries
pip install h3 --prefer-binary
```

**Note**: H3 only required for the H3-based scripts (`geolife_h3_*.py`).

---

### Issue 3: `ImportError: cannot import name 'OmegaConf' from 'omegaconf'`

**Error Message**:
```
ImportError: cannot import name 'OmegaConf' from 'omegaconf'
```

**Cause**: OmegaConf package not installed or version mismatch.

**Solution**:
```bash
pip install omegaconf>=2.0.0
```

---

### Issue 4: GEOS Library Missing

**Error Message**:
```
OSError: Could not find lib geos_c.so or libgeos_c.so.1
```

**Cause**: GEOS library required by Shapely/GeoPandas is not installed.

**Solution**:
```bash
# Ubuntu/Debian
sudo apt-get install libgeos-dev

# macOS
brew install geos

# Then reinstall shapely
pip uninstall shapely
pip install shapely --no-binary shapely
```

---

## Data Loading Errors

### Issue 5: `FileNotFoundError: GeoLife data not found`

**Error Message**:
```
FileNotFoundError: [Errno 2] No such file or directory: 'data/raw/Geolife Trajectories 1.3/Data'
```

**Cause**: Raw GeoLife data not in expected location.

**Solution**:
1. Download GeoLife dataset from Microsoft Research
2. Extract to `data/raw/Geolife Trajectories 1.3/`
3. Verify structure:

```
data/raw/Geolife Trajectories 1.3/
├── Data/
│   ├── 000/
│   │   └── Trajectory/
│   │       ├── 20081023025304.plt
│   │       └── ...
│   ├── 001/
│   └── ...
└── README.txt
```

**Verification**:
```bash
# Check data directory exists
ls -la "data/raw/Geolife Trajectories 1.3/Data"

# Count users
ls "data/raw/Geolife Trajectories 1.3/Data" | wc -l
# Should be around 182
```

---

### Issue 6: `ValueError: No .plt files found`

**Error Message**:
```
ValueError: No trajectory data found for user 000
```

**Cause**: PLT files missing or incorrect directory structure.

**Solution**:
1. Check that each user folder contains a `Trajectory` subfolder
2. Verify PLT files exist:

```bash
# Check for PLT files
find "data/raw/Geolife Trajectories 1.3/Data" -name "*.plt" | head -5
```

**Common Mistake**: Extracting to wrong level:
```
# Wrong (nested too deep):
data/raw/Geolife Trajectories 1.3/Geolife Trajectories 1.3/Data/

# Correct:
data/raw/Geolife Trajectories 1.3/Data/
```

---

### Issue 7: `UnicodeDecodeError` when reading PLT files

**Error Message**:
```
UnicodeDecodeError: 'utf-8' codec can't decode byte 0xfe in position 0
```

**Cause**: Corrupted PLT file or non-UTF-8 encoding.

**Solution**:
```python
# The scripts use latin-1 encoding by default
# If you need to modify, change in read_geolife:
pf = ti.io.read_geolife(
    geolife_path=path,
    # encoding='latin-1'  # Uncomment if needed
)
```

---

## Memory Issues

### Issue 8: `MemoryError` during staypoint generation

**Error Message**:
```
MemoryError: Unable to allocate 8.00 GiB for an array
```

**Cause**: Processing too many GPS points at once.

**Solution 1**: Process users in batches
```python
# Modify script to process in chunks
users = list(pf.user_id.unique())
chunk_size = 50  # Process 50 users at a time

all_staypoints = []
for i in range(0, len(users), chunk_size):
    user_chunk = users[i:i+chunk_size]
    pf_chunk = pf[pf.user_id.isin(user_chunk)]
    sp_chunk, _ = pf_chunk.as_positionfixes.generate_staypoints(...)
    all_staypoints.append(sp_chunk)
    
sp = pd.concat(all_staypoints)
```

**Solution 2**: Increase system memory or use swap
```bash
# Increase swap (Linux)
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

**Solution 3**: Use H3 instead of DBSCAN (more memory efficient)

---

### Issue 9: Memory leak during location generation

**Symptom**: Memory usage grows continuously during processing.

**Cause**: DBSCAN with large epsilon creates many distance computations.

**Solution**:
```python
# Reduce epsilon to limit cluster size
epsilon = 20  # Use smaller values

# Or use H3 for large datasets
h3_resolution = 8  # H3 is O(n) memory
```

---

## Processing Errors

### Issue 10: `ValueError: No staypoints generated`

**Error Message**:
```
ValueError: No staypoints generated. Check distance_threshold and time_threshold.
```

**Cause**: Parameters too restrictive or GPS data too sparse.

**Solution**:
```yaml
# Relax staypoint parameters in config
preprocessing:
  staypoint:
    dist_threshold: 300  # Increase from 200
    time_threshold: 1200  # Decrease from 1800 (30min → 20min)
    gap_threshold: 86400  # Keep at 1 day
```

**Diagnostic**:
```python
# Check GPS point distribution
import pandas as pd

# Load position fixes
pf = ...  # Your position fixes

# Check temporal coverage
print(f"Date range: {pf['tracked_at'].min()} to {pf['tracked_at'].max()}")
print(f"Total points: {len(pf)}")
print(f"Points per day: {len(pf) / (pf['tracked_at'].max() - pf['tracked_at'].min()).days}")
```

---

### Issue 11: `All users filtered out after quality check`

**Error Message**:
```
Warning: No users passed quality filter
```

**Cause**: Quality thresholds too strict.

**Solution**:
```yaml
# Relax quality parameters
preprocessing:
  quality:
    min_days: 30       # Reduce from 50
    min_points: 200    # Reduce from 500
```

**Diagnostic**:
```python
# Check user statistics before filtering
for user_id in sp['user_id'].unique():
    user_sp = sp[sp['user_id'] == user_id]
    days = (user_sp['finished_at'].max() - user_sp['started_at'].min()).days
    points = len(user_sp)
    print(f"User {user_id}: {days} days, {points} staypoints")
```

---

### Issue 12: `KeyError: 'location_id'`

**Error Message**:
```
KeyError: 'location_id'
```

**Cause**: Location generation failed or returned empty results.

**Solution**:
1. Check that staypoints exist before location generation
2. Verify DBSCAN parameters:

```python
# Debug location generation
print(f"Staypoints before location gen: {len(sp)}")

sp, locs = sp.as_staypoints.generate_locations(
    epsilon=20,
    num_samples=2,
    distance_metric="haversine",
    agg_level="dataset"
)

print(f"Staypoints after location gen: {len(sp)}")
print(f"Unique locations: {sp['location_id'].nunique()}")

# Check for NaN (noise)
print(f"Noise points: {sp['location_id'].isna().sum()}")
```

---

### Issue 13: `No valid sequences found`

**Error Message**:
```
Warning: User 000 has no valid sequences in train split
```

**Cause**: User doesn't have enough history before each target.

**Solution**:
```yaml
# Adjust previous_day parameter
dataset:
  previous_day: [3]  # Reduce from 7 (require 3 days instead of 7)
```

**Explanation**:
```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    VALID SEQUENCE REQUIREMENT                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  previous_day = 7 means: Need at least 1 staypoint in past 7 days               │
│                                                                                  │
│  Timeline:        Day 1    Day 2    Day 3    Day 4    Day 5    Day 6    Day 7   │
│  User visits:       ●        -        -        -        -        -        -     │
│  Target on Day 8:   ← No visits in 7-day window → Invalid!                      │
│                                                                                  │
│  Timeline:        Day 1    Day 2    Day 3    Day 4    Day 5    Day 6    Day 7   │
│  User visits:       ●        ●        -        -        ●        -        -     │
│  Target on Day 8:   ← Has visits in 7-day window → Valid ✓                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Issues

### Issue 14: Empty output files

**Symptom**: Output `.pk` files exist but contain empty data.

**Cause**: All sequences were filtered out.

**Diagnostic**:
```python
import pickle

with open('data/processed/geolife_eps20/train.pk', 'rb') as f:
    data = pickle.load(f)
    
print(f"Number of sequences: {len(data)}")
print(f"Keys: {data[0].keys() if data else 'Empty!'}")
```

**Solution**: Check intermediate files for data:
```python
# Check interim staypoints
sp = pd.read_csv('data/interim/geolife_eps20/sp_merged.csv')
print(f"Total staypoints: {len(sp)}")
print(f"Unique users: {sp['user_id'].nunique()}")
print(f"Unique locations: {sp['location_id'].nunique()}")
```

---

### Issue 15: `PermissionError` when writing output

**Error Message**:
```
PermissionError: [Errno 13] Permission denied: 'data/interim/geolife_eps20/sp.csv'
```

**Cause**: Output directory doesn't exist or no write permission.

**Solution**:
```bash
# Create output directories
mkdir -p data/interim/geolife_eps20
mkdir -p data/processed/geolife_eps20

# Check permissions
ls -la data/
chmod -R 755 data/
```

---

## Configuration Errors

### Issue 16: `KeyError` in configuration

**Error Message**:
```
KeyError: 'epsilon'
```

**Cause**: Configuration file missing required parameter.

**Solution**: Verify all required parameters exist:

```yaml
# Minimum required configuration (geolife.yaml)
dataset:
  name: "geolife"
  epsilon: 20          # Required for DBSCAN
  previous_day: [7]

preprocessing:
  staypoint:
    dist_threshold: 200
    time_threshold: 1800
    gap_threshold: 86400
  
  activity:
    dur_threshold: 1500
  
  quality:
    min_days: 50
  
  location:
    num_samples: 2
    distance_metric: "haversine"
    agg_level: "dataset"
```

---

### Issue 17: Configuration not found

**Error Message**:
```
FileNotFoundError: config/preprocessing/geolife.yaml not found
```

**Solution**:
```bash
# Check config directory
ls -la config/preprocessing/

# Run from correct directory
cd /path/to/next_loc_clean_v2
python preprocessing/geolife_1_raw_to_interim.py
```

---

## Common Warnings

### Warning 1: `FutureWarning` from pandas

**Warning**:
```
FutureWarning: A value is trying to be set on a copy of a slice from a DataFrame
```

**Cause**: Pandas copy-on-write behavior.

**Impact**: Usually harmless, but may indicate potential bugs.

**Solution**: The scripts use `.copy()` where needed. If you modify the code:
```python
# Use explicit copy
sp = sp.loc[mask].copy()

# Or use .loc for assignment
sp.loc[mask, 'column'] = value
```

---

### Warning 2: `SettingWithCopyWarning`

**Warning**:
```
SettingWithCopyWarning: A value is trying to be set on a copy of a slice
```

**Cause**: Chained indexing in pandas.

**Solution**:
```python
# Wrong (triggers warning)
sp[sp['user_id'] == 1]['location_id'] = 5

# Correct
sp.loc[sp['user_id'] == 1, 'location_id'] = 5
```

---

### Warning 3: `RuntimeWarning: divide by zero`

**Warning**:
```
RuntimeWarning: divide by zero encountered in log
```

**Cause**: Usually from data normalization with zero values.

**Impact**: May produce NaN values.

**Solution**: Check for zeros in data:
```python
# Check for zeros before normalization
print(f"Zero durations: {(sp['duration'] == 0).sum()}")

# Replace zeros if needed
sp['duration'] = sp['duration'].replace(0, 1)  # Or minimum value
```

---

## Performance Issues

### Issue 18: Script runs very slowly

**Symptom**: Processing takes hours instead of minutes.

**Causes and Solutions**:

1. **Too many n_jobs**: Reduce parallel jobs
   ```python
   sp, locs = sp.as_staypoints.generate_locations(
       ...,
       n_jobs=4  # Instead of -1 (all cores)
   )
   ```

2. **Large epsilon**: Reduce DBSCAN epsilon
   ```yaml
   epsilon: 20  # Instead of 100
   ```

3. **No spatial index**: trackintel should use spatial indexing automatically
   ```python
   # Verify spatial index
   print(sp.sindex)  # Should not be None
   ```

4. **Debug output**: Disable verbose logging
   ```python
   import logging
   logging.getLogger('trackintel').setLevel(logging.WARNING)
   ```

---

### Issue 19: High CPU usage but no progress

**Symptom**: CPU at 100% but no output.

**Cause**: Stuck in DBSCAN clustering with large clusters.

**Solution**:
1. Use smaller epsilon
2. Switch to H3 for large datasets
3. Add progress monitoring:

```python
from tqdm import tqdm

# Monitor progress by user
for user_id in tqdm(sp['user_id'].unique()):
    user_sp = sp[sp['user_id'] == user_id]
    # Process user...
```

---

## Debugging Guide

### Step-by-Step Debugging Process

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    DEBUGGING WORKFLOW                                            │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  1. CHECK INPUT DATA                                                             │
│     ─────────────────                                                            │
│     • Does raw data exist?                                                       │
│     • Are PLT files readable?                                                    │
│     • What's the data volume?                                                    │
│                                                                                  │
│  2. CHECK INTERMEDIATE OUTPUT                                                    │
│     ───────────────────────────                                                  │
│     • Were staypoints generated?                                                 │
│     • How many passed quality filter?                                            │
│     • Are locations assigned?                                                    │
│                                                                                  │
│  3. CHECK CONFIGURATION                                                          │
│     ─────────────────────                                                        │
│     • Are all required params present?                                           │
│     • Are values reasonable?                                                     │
│     • Is the right config file loaded?                                           │
│                                                                                  │
│  4. ADD LOGGING                                                                  │
│     ────────────                                                                 │
│     • Print record counts at each step                                           │
│     • Log filtering decisions                                                    │
│     • Time each operation                                                        │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Debug Code Template

```python
import pandas as pd
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def debug_pipeline(sp):
    """Add this to debug the pipeline."""
    
    logger.info(f"Step 1: Initial staypoints: {len(sp)}")
    logger.info(f"  Users: {sp['user_id'].nunique()}")
    logger.info(f"  Date range: {sp['started_at'].min()} to {sp['finished_at'].max()}")
    
    # After location generation
    if 'location_id' in sp.columns:
        logger.info(f"Step 2: After location gen: {len(sp)}")
        logger.info(f"  Locations: {sp['location_id'].nunique()}")
        logger.info(f"  Noise (NaN): {sp['location_id'].isna().sum()}")
    
    # After filtering
    logger.info(f"Step 3: After filtering: {len(sp)}")
    
    # User statistics
    for uid in sp['user_id'].unique()[:5]:  # First 5 users
        user_sp = sp[sp['user_id'] == uid]
        logger.debug(f"  User {uid}: {len(user_sp)} staypoints, "
                    f"{user_sp['location_id'].nunique()} locations")
    
    return sp
```

---

## FAQ

### Q1: Why are some users filtered out?

**Answer**: Users are filtered based on:
1. **Tracking duration**: Must have at least `min_days` of data
2. **Quality windows**: Must pass quality check in sliding windows
3. **Valid sequences**: Must have sequences with sufficient history

**Check with**:
```python
# Before and after each filter
print(f"Users before quality filter: {sp['user_id'].nunique()}")
# ... apply filter ...
print(f"Users after quality filter: {sp['user_id'].nunique()}")
```

---

### Q2: Why does H3 give different results than DBSCAN?

**Answer**: They use fundamentally different algorithms:
- **DBSCAN**: Data-driven clustering (adapts to data)
- **H3**: Fixed grid assignment (same grid everywhere)

See [11-COMPARISON-DBSCAN-VS-H3.md](11-COMPARISON-DBSCAN-VS-H3.md) for details.

---

### Q3: How do I choose epsilon/resolution?

**Answer**: 
- **epsilon=20**: Good for building-level precision
- **epsilon=50-100**: Good for block-level precision
- **H3 resolution=8**: Similar to epsilon~200m
- **H3 resolution=9**: Similar to epsilon~100m
- **H3 resolution=10**: Similar to epsilon~50m

---

### Q4: Why is location_id offset by 2?

**Answer**: To reserve special IDs:
- **0**: Padding token (for variable-length sequences)
- **1**: Unknown location (for locations not seen in training)
- **2+**: Actual location IDs

---

### Q5: Can I use my own dataset?

**Answer**: Yes! You need to:
1. Convert to trackintel format (positionfixes with lat, lng, tracked_at, user_id)
2. Modify the `read_geolife()` call to use your data loader
3. Adjust parameters for your data characteristics

---

### Q6: How do I add more features?

**Answer**: Modify `enrich_time_info()` in Script 1:
```python
def enrich_time_info(sp, my_new_feature=True):
    # Existing features...
    sp["weekday"] = sp["started_at"].dt.dayofweek
    sp["start_min"] = sp["started_at"].dt.hour * 60 + sp["started_at"].dt.minute
    
    # Add your feature
    if my_new_feature:
        sp["is_weekend"] = sp["weekday"].isin([5, 6]).astype(int)
        sp["hour"] = sp["started_at"].dt.hour
    
    return sp
```

Then update `generate_sequences()` in Script 2 to include the new feature.

---

### Q7: What if my GPS data has gaps?

**Answer**: The pipeline handles gaps via:
1. **gap_threshold** in staypoint generation (default: 1 day)
2. Gaps > threshold create new staypoint segments
3. Sequences only use data within `previous_day` window

For very gappy data, consider:
- Reducing `min_days` requirement
- Increasing `previous_day` window
- Filtering users with too many gaps

---

*Documentation Version: 1.0*
*For PhD Research Reference*
