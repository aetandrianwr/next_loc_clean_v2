# H3 Script 2: Interim to Processed - Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [Code Differences from Standard Script 2](#code-differences-from-standard-script-2)
3. [Complete Code Comparison](#complete-code-comparison)
4. [Output Files](#output-files)

---

## Overview

**Script**: `preprocessing/geolife_h3_2_interim_to_processed.py`

**Purpose**: Same as Script 2, but processes H3-based intermediate data.

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│              H3 SCRIPT 2: INTERIM TO PROCESSED                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  IMPORTANT: This script is ALMOST IDENTICAL to the standard Script 2            │
│                                                                                  │
│  The ONLY differences are:                                                       │
│  1. File paths use h3r{resolution} instead of eps{epsilon}                      │
│  2. Metadata includes h3_resolution instead of epsilon                           │
│  3. Default config path points to geolife_h3.yaml                               │
│                                                                                  │
│  All sequence generation logic is IDENTICAL                                      │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Code Differences from Standard Script 2

### Key Differences Summary

| Aspect | Standard Script 2 | H3 Script 2 |
|--------|------------------|-------------|
| Config parameter | `epsilon` | `h3_resolution` |
| Input folder | `geolife_eps{X}/interim/` | `geolife_h3r{X}/interim/` |
| Input file | `intermediate_eps{X}.csv` | `intermediate_h3r{X}.csv` |
| Output folder | `geolife_eps{X}/processed/` | `geolife_h3r{X}/processed/` |
| Output file prefix | `geolife_eps{X}_prev{Y}` | `geolife_h3r{X}_prev{Y}` |
| Metadata field | `"epsilon": X` | `"h3_resolution": X` |
| Default config | `config/geolife.yaml` | `config/preprocessing/geolife_h3.yaml` |

### Detailed Code Changes

#### Change 1: Parameter Name (Line 148)

**Standard:**
```python
def process_for_previous_day(sp, split_ratios, max_duration, previous_day, epsilon, dataset_name, output_base_path):
```

**H3:**
```python
def process_for_previous_day(sp, split_ratios, max_duration, previous_day, h3_resolution, dataset_name, output_base_path):
```

#### Change 2: Output Naming (Line 151)

**Standard:**
```python
output_name = f"{dataset_name}_eps{epsilon}_prev{previous_day}"
```

**H3:**
```python
output_name = f"{dataset_name}_h3r{h3_resolution}_prev{previous_day}"
```

#### Change 3: Metadata (Lines 266-269)

**Standard:**
```python
metadata = {
    "dataset_name": dataset_name,
    "output_dataset_name": output_name,
    "epsilon": epsilon,
    "previous_day": previous_day,
    ...
}
```

**H3:**
```python
metadata = {
    "dataset_name": dataset_name,
    "output_dataset_name": output_name,
    "h3_resolution": h3_resolution,
    "previous_day": previous_day,
    ...
}
```

#### Change 4: Config Parameter (Lines 300-304)

**Standard:**
```python
dataset_config = config["dataset"]
preproc_config = config["preprocessing"]

dataset_name = dataset_config["name"]
epsilon = dataset_config["epsilon"]
previous_day_list = dataset_config["previous_day"]
```

**H3:**
```python
dataset_config = config["dataset"]
preproc_config = config["preprocessing"]

dataset_name = dataset_config["name"]
h3_resolution = dataset_config["h3_resolution"]
previous_day_list = dataset_config["previous_day"]
```

#### Change 5: Path Construction (Lines 313-316)

**Standard:**
```python
output_folder = f"{dataset_name}_eps{epsilon}"
interim_path = os.path.join("data", output_folder, "interim")
output_base_path = os.path.join("data", output_folder)
```

**H3:**
```python
output_folder = f"{dataset_name}_h3r{h3_resolution}"
interim_path = os.path.join("data", output_folder, "interim")
output_base_path = os.path.join("data", output_folder)
```

#### Change 6: Loading Interim Data (Lines 331-332)

**Standard:**
```python
interim_file = os.path.join(interim_path, f"intermediate_eps{epsilon}.csv")
sp = pd.read_csv(interim_file)
```

**H3:**
```python
interim_file = os.path.join(interim_path, f"intermediate_h3r{h3_resolution}.csv")
sp = pd.read_csv(interim_file)
```

#### Change 7: Function Call (Lines 339-341)

**Standard:**
```python
metadata = process_for_previous_day(
    sp, split_ratios, max_duration, previous_day, 
    epsilon, dataset_name, output_base_path
)
```

**H3:**
```python
metadata = process_for_previous_day(
    sp, split_ratios, max_duration, previous_day, 
    h3_resolution, dataset_name, output_base_path
)
```

#### Change 8: Default Config Path (Line 366)

**Standard:**
```python
parser.add_argument(
    "--config",
    type=str,
    default="config/geolife.yaml",
    help="Path to dataset configuration file"
)
```

**H3:**
```python
parser.add_argument(
    "--config",
    type=str,
    default="config/preprocessing/geolife_h3.yaml",
    help="Path to dataset configuration file"
)
```

---

## Complete Code Comparison

### Side-by-Side Comparison of process_intermediate_to_final

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    COMPLETE FUNCTION COMPARISON                                  │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  STANDARD SCRIPT 2                        H3 SCRIPT 2                            │
│  ─────────────────                        ─────────────                          │
│                                                                                  │
│  def process_intermediate_to_final(       def process_intermediate_to_final(     │
│      config):                                 config):                           │
│                                                                                  │
│    dataset_config = config["dataset"]       dataset_config = config["dataset"]   │
│    preproc_config = config["..."]           preproc_config = config["..."]       │
│                                                                                  │
│    dataset_name = dataset_config["name"]    dataset_name = dataset_config["name"]│
│    epsilon = dataset_config["epsilon"]      h3_resolution = dataset_config[      │
│                                                   "h3_resolution"]               │
│    previous_day_list = dataset_config[      previous_day_list = dataset_config[  │
│        "previous_day"]                          "previous_day"]                  │
│                                                                                  │
│    # Paths                                  # Paths                              │
│    output_folder = f"{dataset_name}_        output_folder = f"{dataset_name}_    │
│        eps{epsilon}"                            h3r{h3_resolution}"              │
│    interim_path = os.path.join(...)         interim_path = os.path.join(...)     │
│    output_base_path = os.path.join(...)     output_base_path = os.path.join(...)│
│                                                                                  │
│    # Load intermediate data                 # Load intermediate data             │
│    interim_file = os.path.join(             interim_file = os.path.join(         │
│        interim_path,                            interim_path,                    │
│        f"intermediate_eps{epsilon}.csv")        f"intermediate_h3r{h3_res}.csv") │
│    sp = pd.read_csv(interim_file)           sp = pd.read_csv(interim_file)       │
│                                                                                  │
│    # Process each previous_day              # Process each previous_day          │
│    for previous_day in previous_day_list:   for previous_day in previous_day_lst:│
│        metadata = process_for_previous_day(     metadata = process_for_prev_day( │
│            sp, split_ratios, max_duration,          sp, split_ratios, max_dur,   │
│            previous_day, epsilon,                   previous_day, h3_resolution, │
│            dataset_name, output_base_path)          dataset_name, output_base)   │
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## Output Files

### Directory Structure Comparison

```
data/
├── geolife_eps20/                          # Standard version
│   ├── interim/
│   │   └── intermediate_eps20.csv
│   └── processed/
│       ├── geolife_eps20_prev7_train.pk
│       ├── geolife_eps20_prev7_validation.pk
│       ├── geolife_eps20_prev7_test.pk
│       └── geolife_eps20_prev7_metadata.json
│
└── geolife_h3r8/                           # H3 version
    ├── interim/
    │   └── intermediate_h3r8.csv
    └── processed/
        ├── geolife_h3r8_prev7_train.pk       # Note: h3r8 instead of eps20
        ├── geolife_h3r8_prev7_validation.pk
        ├── geolife_h3r8_prev7_test.pk
        └── geolife_h3r8_prev7_metadata.json
```

### Metadata Comparison

**Standard (geolife_eps20_prev7_metadata.json)**:
```json
{
  "dataset_name": "geolife",
  "output_dataset_name": "geolife_eps20_prev7",
  "epsilon": 20,
  "previous_day": 7,
  "total_user_num": 31,
  "total_loc_num": 245,
  ...
}
```

**H3 (geolife_h3r8_prev7_metadata.json)**:
```json
{
  "dataset_name": "geolife",
  "output_dataset_name": "geolife_h3r8_prev7",
  "h3_resolution": 8,
  "previous_day": 7,
  "total_user_num": 28,
  "total_loc_num": 312,
  ...
}
```

### Sequence Format (Identical)

Both versions produce the same sequence dictionary format:

```python
{
    "X": [4, 3, 4, 2, 4],           # Location history
    "user_X": [1, 1, 1, 1, 1],      # User IDs
    "weekday_X": [2, 3, 4, 5, 6],   # Weekdays
    "start_min_X": [540, 510, 480, 600, 720],  # Start times
    "dur_X": [480, 540, 300, 120, 180],        # Durations
    "diff": [7, 6, 5, 3, 1],        # Days ago
    "Y": 3                          # Target location
}
```

The only difference is in the location IDs:
- **Standard**: Location IDs from DBSCAN clustering
- **H3**: Location IDs from H3 hexagonal cells

---

## Running the H3 Pipeline

### Commands

```bash
# Step 1: Raw to Interim (H3)
python preprocessing/geolife_h3_1_raw_to_interim.py --config config/preprocessing/geolife_h3.yaml

# Step 2: Interim to Processed (H3)
python preprocessing/geolife_h3_2_interim_to_processed.py --config config/preprocessing/geolife_h3.yaml
```

### Using Different H3 Resolutions

```bash
# Create config for resolution 9 (smaller hexagons)
# Edit geolife_h3.yaml: h3_resolution: 9

python preprocessing/geolife_h3_1_raw_to_interim.py --config config/preprocessing/geolife_h3.yaml
python preprocessing/geolife_h3_2_interim_to_processed.py --config config/preprocessing/geolife_h3.yaml

# Output: data/geolife_h3r9/processed/
```

---

## Summary

The H3 Script 2 is essentially a copy of the standard Script 2 with:

1. **Parameter rename**: `epsilon` → `h3_resolution`
2. **Path changes**: `eps{X}` → `h3r{X}`
3. **Metadata update**: Include `h3_resolution` instead of `epsilon`

All sequence generation logic, splitting, encoding, and validation are **exactly the same**. This ensures consistency between the two approaches and allows direct comparison of results.

---

## Next Steps

- [08-DATA-STRUCTURES.md](08-DATA-STRUCTURES.md) - Detailed data format reference
- [11-COMPARISON-DBSCAN-VS-H3.md](11-COMPARISON-DBSCAN-VS-H3.md) - Compare DBSCAN vs H3 results

---

*Documentation Version: 1.0*
*For PhD Research Reference*
