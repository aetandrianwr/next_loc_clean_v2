# GeoLife Dataset Preprocessing Documentation

## Comprehensive Documentation for Next Location Prediction

This documentation provides a complete, PhD-thesis-level reference for the GeoLife dataset preprocessing pipeline used in the Next Location Prediction project.

---

## 📚 Documentation Index

| Document | Description | Key Topics |
|----------|-------------|------------|
| [01-OVERVIEW.md](01-OVERVIEW.md) | Pipeline architecture and quick start | Two-stage pipeline, DBSCAN vs H3, flow diagrams |
| [02-GEOLIFE-DATASET.md](02-GEOLIFE-DATASET.md) | GeoLife dataset description | PLT format, data characteristics, user statistics |
| [03-CONFIGURATION.md](03-CONFIGURATION.md) | Configuration file reference | YAML parameters, tuning guide, visualizations |
| [04-SCRIPT1-RAW-TO-INTERIM.md](04-SCRIPT1-RAW-TO-INTERIM.md) | Raw to Interim (DBSCAN) | Line-by-line code walkthrough, 7 processing steps |
| [05-SCRIPT2-INTERIM-TO-PROCESSED.md](05-SCRIPT2-INTERIM-TO-PROCESSED.md) | Interim to Processed (DBSCAN) | Sequence generation, encoding, data splitting |
| [06-H3-SCRIPT1-RAW-TO-INTERIM.md](06-H3-SCRIPT1-RAW-TO-INTERIM.md) | Raw to Interim (H3) | H3 hexagonal grid, generate_h3_locations() |
| [07-H3-SCRIPT2-INTERIM-TO-PROCESSED.md](07-H3-SCRIPT2-INTERIM-TO-PROCESSED.md) | Interim to Processed (H3) | Differences from standard pipeline |
| [08-DATA-STRUCTURES.md](08-DATA-STRUCTURES.md) | Data format specifications | PLT, CSV, PK files, JSON schemas |
| [09-FUNCTIONS-REFERENCE.md](09-FUNCTIONS-REFERENCE.md) | API documentation | All functions with parameters and return types |
| [10-EXAMPLES.md](10-EXAMPLES.md) | Concrete examples | End-to-end walkthrough with Alice example |
| [11-COMPARISON-DBSCAN-VS-H3.md](11-COMPARISON-DBSCAN-VS-H3.md) | Method comparison | Pros/cons, when to use which, performance |
| [12-TROUBLESHOOTING.md](12-TROUBLESHOOTING.md) | Problem solving guide | Common errors, debugging, FAQ |

---

## 🎯 Quick Navigation

### For Understanding the Pipeline

1. Start with [01-OVERVIEW.md](01-OVERVIEW.md) for the big picture
2. Read [02-GEOLIFE-DATASET.md](02-GEOLIFE-DATASET.md) to understand the input data
3. Continue with [04-SCRIPT1-RAW-TO-INTERIM.md](04-SCRIPT1-RAW-TO-INTERIM.md) and [05-SCRIPT2-INTERIM-TO-PROCESSED.md](05-SCRIPT2-INTERIM-TO-PROCESSED.md) for detailed processing steps

### For Configuration

1. [03-CONFIGURATION.md](03-CONFIGURATION.md) - All parameters explained
2. [11-COMPARISON-DBSCAN-VS-H3.md](11-COMPARISON-DBSCAN-VS-H3.md) - Choosing between methods

### For Implementation

1. [09-FUNCTIONS-REFERENCE.md](09-FUNCTIONS-REFERENCE.md) - API documentation
2. [08-DATA-STRUCTURES.md](08-DATA-STRUCTURES.md) - Input/output formats
3. [10-EXAMPLES.md](10-EXAMPLES.md) - Concrete code examples

### For Troubleshooting

1. [12-TROUBLESHOOTING.md](12-TROUBLESHOOTING.md) - Common issues and solutions

---

## 🏗️ Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    GEOLIFE PREPROCESSING PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                  │
│  ┌─────────────┐     ┌─────────────────────┐     ┌──────────────────┐           │
│  │ RAW DATA    │     │ SCRIPT 1            │     │ INTERIM DATA     │           │
│  │ .plt files  │────▶│ Raw → Interim       │────▶│ .csv files       │           │
│  │ 182 users   │     │ • Staypoints        │     │ • sp_merged.csv  │           │
│  │ 24M points  │     │ • Locations         │     │ • loc.csv        │           │
│  └─────────────┘     │ • Quality filter    │     └──────────────────┘           │
│                      └─────────────────────┘              │                      │
│                                                           │                      │
│                                                           ▼                      │
│                      ┌─────────────────────┐     ┌──────────────────┐           │
│                      │ SCRIPT 2            │     │ PROCESSED DATA   │           │
│                      │ Interim → Processed │────▶│ .pk files        │           │
│                      │ • Encoding          │     │ • train.pk       │           │
│                      │ • Splitting         │     │ • val.pk         │           │
│                      │ • Sequences         │     │ • test.pk        │           │
│                      └─────────────────────┘     │ • config.json    │           │
│                                                  └──────────────────┘           │
│                                                                                  │
│  Two Methods Available:                                                          │
│  ─────────────────────                                                          │
│  • DBSCAN: geolife_1_raw_to_interim.py + geolife_2_interim_to_processed.py      │
│  • H3:     geolife_h3_1_raw_to_interim.py + geolife_h3_2_interim_to_processed.py│
│                                                                                  │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Step 1: Prepare Data
```bash
# Download GeoLife dataset and extract to:
# data/raw/Geolife Trajectories 1.3/
```

### Step 2: Install Dependencies
```bash
pip install trackintel h3 omegaconf pandas scikit-learn
```

### Step 3: Run DBSCAN Pipeline
```bash
# Stage 1: Raw to Interim
python preprocessing/geolife_1_raw_to_interim.py

# Stage 2: Interim to Processed
python preprocessing/geolife_2_interim_to_processed.py
```

### Step 4: Or Run H3 Pipeline
```bash
# Stage 1: Raw to Interim
python preprocessing/geolife_h3_1_raw_to_interim.py

# Stage 2: Interim to Processed
python preprocessing/geolife_h3_2_interim_to_processed.py
```

### Step 5: Use Output
```python
import pickle

# Load processed data
with open('data/processed/geolife_eps20/train.pk', 'rb') as f:
    train_data = pickle.load(f)

# Each item is a sequence dictionary
print(train_data[0].keys())
# dict_keys(['X', 'user_X', 'weekday_X', 'start_min_X', 'dur_X', 'diff', 'Y'])
```

---

## 📁 File Structure

```
next_loc_clean_v2/
├── config/
│   └── preprocessing/
│       ├── geolife.yaml          # DBSCAN configuration
│       └── geolife_h3.yaml       # H3 configuration
├── preprocessing/
│   ├── geolife_1_raw_to_interim.py        # DBSCAN Stage 1
│   ├── geolife_2_interim_to_processed.py  # DBSCAN Stage 2
│   ├── geolife_h3_1_raw_to_interim.py     # H3 Stage 1
│   └── geolife_h3_2_interim_to_processed.py # H3 Stage 2
├── data/
│   ├── raw/
│   │   └── Geolife Trajectories 1.3/      # Raw GPS data
│   ├── interim/
│   │   ├── geolife_eps20/                 # DBSCAN interim
│   │   └── geolife_h3r8/                  # H3 interim
│   └── processed/
│       ├── geolife_eps20/                 # DBSCAN processed
│       └── geolife_h3r8/                  # H3 processed
└── docs/
    └── docs_preprocessing_geolife/        # This documentation
```

---

## 📊 Output Data Format

### Sequence Dictionary
```python
{
    'X': [5, 12, 3, 8, 15, 12, 3],  # Location history (encoded IDs)
    'user_X': [2, 2, 2, 2, 2, 2, 2],  # User IDs
    'weekday_X': [0, 0, 1, 2, 3, 4, 5],  # Day of week (0=Mon)
    'start_min_X': [480, 720, 510, 750, 495, 735, 465],  # Start time (minutes)
    'dur_X': [120.5, 45.0, 180.2, 30.0, 90.0, 60.0, 240.0],  # Duration (minutes)
    'diff': [6, 5, 4, 3, 2, 1, 0],  # Days before target
    'Y': 5  # Target location ID
}
```

---

## 🔑 Key Concepts

| Concept | Description |
|---------|-------------|
| **Position Fix** | A single GPS measurement (lat, lon, timestamp) |
| **Staypoint** | A location where user stayed for ≥30 minutes |
| **Location** | A cluster of staypoints (DBSCAN) or grid cell (H3) |
| **Sequence** | Historical staypoints used to predict next location |
| **Previous Day** | Days of history to include (e.g., 7 = past week) |

---

## 📖 Citation

If you use this preprocessing pipeline in your research, please cite:

```bibtex
@misc{geolife_preprocessing,
  title={GeoLife Dataset Preprocessing for Next Location Prediction},
  author={Your Name},
  year={2024},
  howpublished={GitHub Repository}
}
```

---

## 📝 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2024 | Initial documentation |

---

## 🤝 Contributing

To improve this documentation:
1. Fork the repository
2. Edit documentation files
3. Submit a pull request

---

*Documentation for PhD Research Reference*
*Complete A-Z Guide for GeoLife Preprocessing*
