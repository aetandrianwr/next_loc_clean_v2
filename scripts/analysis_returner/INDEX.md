# Return Probability Distribution Analysis - Complete Index

## 📁 Directory Structure

```
/data/next_loc_clean_v2/scripts/analysis_returner/
│
├── 📜 Scripts (3 files)
│   ├── return_probability_analysis.py    Main analysis script (367 lines)
│   ├── compare_datasets.py               Dataset comparison script (93 lines)
│   └── run_analysis.sh                   Automated runner (executable)
│
├── 📖 Documentation (4 files)
│   ├── README.md                         User guide (148 lines)
│   ├── TECHNICAL_DETAILS.md              Implementation details (7.0 KB)
│   ├── RESULTS_SUMMARY.txt               Analysis results (8.4 KB)
│   ├── QUICK_REFERENCE.txt               Quick reference card (3.2 KB)
│   └── INDEX.md                          This file
│
├── 📊 Data Files (4 files)
│   ├── geolife_return_probability_data.csv          PDF data: t_hours, F_pt (121 rows)
│   ├── geolife_return_probability_data_returns.csv  Return times: user_id, delta_t (49 rows)
│   ├── diy_return_probability_data.csv              PDF data: t_hours, F_pt (121 rows)
│   └── diy_return_probability_data_returns.csv      Return times: user_id, delta_t (1091 rows)
│
└── 🎨 Plots (3 files)
    ├── geolife_return_probability.png               Geolife plot (177 KB, 2358×1771)
    ├── diy_return_probability.png                   DIY plot (186 KB, 2358×1771)
    └── comparison_return_probability.png            Comparison plot (263 KB, 2358×1771)
```

**Total: 14 files, 1.4 MB**

---

## 🚀 Quick Start

```bash
# 1. Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# 2. Navigate to project root
cd /data/next_loc_clean_v2

# 3. Run analysis with defaults
python scripts/analysis_returner/return_probability_analysis.py

# 4. Create comparison plot
cd scripts/analysis_returner
python compare_datasets.py

# Or run everything at once:
./scripts/analysis_returner/run_analysis.sh
```

---

## 📋 File Descriptions

### Scripts

#### `return_probability_analysis.py`
**Purpose:** Main analysis script that computes first-return time distributions  
**Input:** Intermediate CSV files from preprocessing pipeline  
**Output:** Plots, probability density data, individual return times  
**Key Features:**
- Fully vectorized (no per-row loops)
- Configurable parameters (bin width, max time)
- Validates probability mass conservation
- Handles both Geolife and DIY datasets

**Usage:**
```bash
python return_probability_analysis.py [--bin-width FLOAT] [--max-hours INT] [--output-dir PATH]
```

**Example:**
```bash
python return_probability_analysis.py --bin-width 1.0 --max-hours 240
```

#### `compare_datasets.py`
**Purpose:** Creates side-by-side comparison plot of both datasets  
**Input:** Generated probability density CSV files  
**Output:** Comparison plot with both curves  
**Key Features:**
- Overlays Geolife (blue dashed) and DIY (red solid) curves
- Prints statistical comparison table

**Usage:**
```bash
cd scripts/analysis_returner
python compare_datasets.py
```

#### `run_analysis.sh`
**Purpose:** Automated shell script to run complete analysis  
**Features:**
- Activates conda environment automatically
- Runs main analysis
- Creates comparison plot
- Prints summary

**Usage:**
```bash
./run_analysis.sh
```

---

### Documentation

#### `README.md`
**Purpose:** User guide and overview  
**Contents:**
- Analysis overview
- Usage instructions
- File descriptions
- Results summary
- Parameters and customization

**Audience:** Users who want to run the analysis

#### `TECHNICAL_DETAILS.md`
**Purpose:** Detailed implementation documentation  
**Contents:**
- Mathematical definitions
- Algorithm details
- Performance optimization
- Validation procedures
- Comparison with original paper

**Audience:** Developers and researchers

#### `RESULTS_SUMMARY.txt`
**Purpose:** Comprehensive analysis results  
**Contents:**
- Methodology description
- Complete statistics for both datasets
- Comparison and interpretation
- Validation results
- Recommendations

**Audience:** Data scientists and stakeholders

#### `QUICK_REFERENCE.txt`
**Purpose:** One-page reference card  
**Contents:**
- Key commands
- File locations
- Quick statistics
- Common use cases

**Audience:** Quick lookup during analysis

---

### Data Files

#### `{dataset}_return_probability_data.csv`
**Purpose:** Probability density function data  
**Format:** CSV with 2 columns
- `t_hours`: Time bin centers (1, 3, 5, ..., 239 hours)
- `F_pt`: Probability density at time t

**Properties:**
- 121 rows (120 bins of 2 hours each, 0-240h)
- Integrates to 1.0 (probability conservation)
- Can be used to recreate plots

**Example:**
```csv
t_hours,F_pt
1.0,0.02040816326530612
3.0,0.05102040816326531
5.0,0.0
...
```

#### `{dataset}_return_probability_data_returns.csv`
**Purpose:** Individual user return times  
**Format:** CSV with 2 columns
- `user_id`: User identifier
- `delta_t_hours`: First-return time in hours

**Properties:**
- One row per user with a return
- Geolife: 49 rows
- DIY: 1,091 rows
- Raw data for statistical analysis

**Example:**
```csv
user_id,delta_t_hours
0,35.283333333333
1,82.016666666667
...
```

---

### Plots

#### `{dataset}_return_probability.png`
**Purpose:** Return probability distribution visualization  
**Format:** PNG, 300 DPI, 2358×1771 pixels  
**Style:** Matches González et al. (2008) Figure 2c
- Blue dashed line for users curve
- X-axis: t (hours), ticks at 24h intervals
- Y-axis: F_pt(t) probability density
- Legend at top-right
- Clean appearance (hidden top/right spines)

#### `comparison_return_probability.png`
**Purpose:** Side-by-side comparison of both datasets  
**Format:** PNG, 300 DPI, 2358×1771 pixels  
**Features:**
- Geolife: Blue dashed line with circle markers
- DIY: Red solid line with square markers
- Same styling as individual plots

---

## 📊 Results Summary

### Geolife Dataset (eps=20)
| Metric | Value |
|--------|-------|
| Total events | 19,191 |
| Total users | 91 |
| Users with returns | 49 (53.85%) |
| Mean return time | 58.96 hours |
| Median return time | 35.28 hours |
| Peak at | 3 hours |
| Max F_pt(t) | 0.051020 |

### DIY Dataset (eps=50)
| Metric | Value |
|--------|-------|
| Total events | 265,621 |
| Total users | 1,306 |
| Users with returns | 1,091 (83.54%) |
| Mean return time | 60.02 hours |
| Median return time | 42.77 hours |
| Peak at | 23 hours |
| Max F_pt(t) | 0.024290 |

---

## ✅ Validation

All deliverables have been validated:
- ✓ Probability mass conservation (∫F_pt(t)dt = 1.0000)
- ✓ No negative values
- ✓ Data consistency checks passed
- ✓ Statistical summaries correct
- ✓ Plots generated successfully
- ✓ All files present

---

## 🔧 Technical Specifications

### Dependencies
- Python 3.x
- pandas (data manipulation)
- numpy (numerical operations)
- matplotlib (visualization)

All available in `mlenv` conda environment.

### Performance
- Geolife (19K events): < 1 second
- DIY (265K events): < 2 seconds
- Fully vectorized implementation
- Time complexity: O(N log N)

### Data Sources
- `data/geolife_eps20/interim/intermediate_eps20.csv`
- `data/diy_eps50/interim/intermediate_eps50.csv`

These are from Step 2/5 (Encode locations) of the preprocessing pipeline.

---

## 📚 Reference

González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008).  
**Understanding individual human mobility patterns.**  
*Nature*, 453(7196), 779-782.  
doi: 10.1038/nature06958

---

## 📝 Version History

**Version 1.0** (December 31, 2025)
- Initial implementation
- Both Geolife and DIY datasets analyzed
- Complete documentation
- Validation passed

---

## 👥 Contact

For questions or issues, contact the data science team.

**Last updated:** December 31, 2025
