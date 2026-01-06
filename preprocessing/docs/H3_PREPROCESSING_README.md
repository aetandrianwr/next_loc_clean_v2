# H3-Based Location Preprocessing

This document describes the H3 hexagonal grid-based preprocessing pipeline for mobility datasets. This approach replaces DBSCAN clustering with Uber's H3 hierarchical spatial indexing system for location assignment.

## Overview

The H3 preprocessing pipeline assigns staypoints to hexagonal cells instead of using DBSCAN clustering. This provides:
- **Deterministic results**: Same input always produces same output
- **Consistent spatial resolution**: All cells have approximately equal area
- **Hierarchical structure**: Resolution can be easily adjusted
- **Fast computation**: O(1) cell assignment per point

## Scripts

### DIY Dataset
| Script | Description |
|--------|-------------|
| `preprocessing/diy_h3_1_raw_to_interim.py` | Raw → Interim (H3 location assignment) |
| `preprocessing/diy_h3_2_interim_to_processed.py` | Interim → Processed (sequence generation) |

### GeoLife Dataset
| Script | Description |
|--------|-------------|
| `preprocessing/geolife_h3_1_raw_to_interim.py` | Raw → Interim (H3 location assignment) |
| `preprocessing/geolife_h3_2_interim_to_processed.py` | Interim → Processed (sequence generation) |

## Configuration Files

- `config/preprocessing/diy_h3.yaml` - DIY H3 configuration
- `config/preprocessing/geolife_h3.yaml` - GeoLife H3 configuration

### Key Configuration Parameters

```yaml
dataset:
  name: "diy"  # or "geolife"
  h3_resolution: 8  # H3 resolution (0-15)
  previous_day: [7]  # List of history window sizes

preprocessing:
  location:
    num_samples: 2  # Minimum staypoints per cell to be valid location
```

## H3 Resolution Reference

| Resolution | Edge Length | Cell Area |
|------------|-------------|-----------|
| 7 | ~1.22 km | ~5.16 km² |
| 8 | ~461 m | ~0.74 km² |
| 9 | ~174 m | ~0.11 km² |
| 10 | ~66 m | ~0.015 km² |

**Default: Resolution 8** (~461m edge length, comparable to typical city block)

## Usage

### Running the Pipeline

```bash
# DIY Dataset
python preprocessing/diy_h3_1_raw_to_interim.py --config config/preprocessing/diy_h3.yaml
python preprocessing/diy_h3_2_interim_to_processed.py --config config/preprocessing/diy_h3.yaml

# GeoLife Dataset
python preprocessing/geolife_h3_1_raw_to_interim.py --config config/preprocessing/geolife_h3.yaml
python preprocessing/geolife_h3_2_interim_to_processed.py --config config/preprocessing/geolife_h3.yaml
```

### Custom Resolution

To use a different H3 resolution, modify the config file:

```yaml
dataset:
  h3_resolution: 9  # Finer grid (~174m)
```

Or create a new config file and specify it:

```bash
python preprocessing/diy_h3_1_raw_to_interim.py --config config/preprocessing/diy_h3_r9.yaml
```

## Output Structure

```
data/
├── diy_h3r8/                           # DIY with H3 resolution 8
│   ├── interim/
│   │   ├── intermediate_h3r8.csv       # Processed staypoints
│   │   ├── locations_h3r8.csv          # Location definitions
│   │   ├── valid_users_h3r8.csv        # Valid user list
│   │   ├── staypoints_merged_h3r8.csv  # Merged staypoints
│   │   └── interim_stats_h3r8.json     # Statistics
│   └── processed/
│       ├── diy_h3r8_prev7_train.pk     # Training sequences
│       ├── diy_h3r8_prev7_validation.pk # Validation sequences
│       ├── diy_h3r8_prev7_test.pk      # Test sequences
│       └── diy_h3r8_prev7_metadata.json # Dataset metadata
│
└── geolife_h3r8/                       # GeoLife with H3 resolution 8
    ├── interim/
    │   └── ...
    └── processed/
        └── ...
```

## Key Differences from DBSCAN Version

| Aspect | DBSCAN Version | H3 Version |
|--------|----------------|------------|
| Parameter | `epsilon` (distance in meters) | `h3_resolution` (0-15) |
| Folder naming | `diy_eps50` | `diy_h3r8` |
| File naming | `intermediate_eps50.csv` | `intermediate_h3r8.csv` |
| Location assignment | Density-based clustering | Hexagonal grid cells |
| Noise handling | Points not in cluster = NaN | Cells with < num_samples = NaN |
| Reproducibility | May vary with random seed | Deterministic |

## Algorithm Details

### H3 Location Assignment (Script 1)

1. **Cell Assignment**: Each staypoint is assigned to an H3 cell based on its coordinates:
   ```python
   h3_cell = h3.latlng_to_cell(lat, lon, resolution)
   ```

2. **Minimum Samples Filter**: Cells with fewer than `num_samples` staypoints are filtered (similar to DBSCAN noise):
   ```python
   valid_cells = cell_counts[cell_counts >= num_samples].index
   ```

3. **Location ID Assignment**: Valid cells are mapped to sequential integer IDs

4. **Location Coordinates**: Cell center is used as location coordinates:
   ```python
   center_lat, center_lon = h3.cell_to_latlng(h3_cell)
   ```

### Sequence Generation (Script 2)

The sequence generation process is identical to the DBSCAN version:
1. Split data into train/val/test by user timeline
2. Encode location IDs (ordinal encoding + 2 for padding/unknown)
3. Filter valid sequences based on `previous_day` parameter
4. Generate sequence dictionaries with features (X, Y, weekday, duration, etc.)
5. Save as pickle files

## Dependencies

- `h3>=4.0.0` (Uber H3 Python bindings)
- `pandas`
- `numpy`
- `geopandas`
- `trackintel`
- `scikit-learn`
- `pyyaml`

Install H3:
```bash
pip install h3
```

## Metadata Output

The processed metadata includes:

```json
{
  "dataset_name": "diy",
  "output_dataset_name": "diy_h3r8_prev7",
  "h3_resolution": 8,
  "previous_day": 7,
  "total_user_num": 847,
  "total_loc_num": 2757,
  "train_sequences": 225182,
  "val_sequences": 14928,
  "test_sequences": 18491,
  "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
}
```

## Troubleshooting

### Common Issues

1. **H3 not installed**: `pip install h3`

2. **Resolution too fine**: Higher resolutions (>10) may create too many locations with few staypoints. Increase `num_samples` or decrease resolution.

3. **Memory issues**: For large datasets, consider processing in chunks or reducing resolution.

### Comparing with DBSCAN Results

H3 and DBSCAN produce different location definitions:
- H3: Grid-based, all cells same size
- DBSCAN: Cluster-based, varying cluster sizes

Expect different numbers of locations and sequence statistics between the two approaches.
