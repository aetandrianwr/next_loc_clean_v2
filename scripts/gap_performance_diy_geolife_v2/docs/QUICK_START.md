# Quick Start Guide

Get started with the Gap Performance Analysis framework in 5 minutes.

---

## Prerequisites

### Required Software
- Python 3.8+
- PyTorch 1.9+
- NumPy, Pandas, Matplotlib, SciPy

### Required Files

**Data files** (must exist):
```
data/diy_eps50/processed/diy_eps50_prev7_test.pk
data/geolife_eps20/processed/geolife_eps20_prev7_test.pk
```

**Model files** (for model analysis only):
```
experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt
experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt
scripts/sci_hyperparam_tuning/configs/pointer_v45_diy_trial09.yaml
scripts/sci_hyperparam_tuning/configs/pointer_v45_geolife_trial01.yaml
```

---

## Quick Run

### Option 1: Run All Analyses (Recommended)

```bash
cd /data/next_loc_clean_v2
python scripts/gap_performance_diy_geolife_v2/run_all_experiments.py
```

**Time**: ~5-10 minutes (depending on GPU availability)

**Output**:
- 10 figures in `results/figures/`
- 3 tables in `results/tables/`
- 3 JSON files in `results/`

### Option 2: Run Individual Analyses

```bash
# Mobility patterns only (no model required)
python scripts/gap_performance_diy_geolife_v2/analyze_mobility_patterns.py

# Model behavior (requires trained models)
python scripts/gap_performance_diy_geolife_v2/analyze_model_pointer.py

# Recency patterns only (no model required)
python scripts/gap_performance_diy_geolife_v2/analyze_recency_patterns.py
```

---

## Understanding the Output

### Key Findings Location

After running, check these files for key results:

1. **Quick Summary**: Look at console output after running
2. **Detailed Numbers**: `results/tables/metric_comparison.csv`
3. **Visual Summary**: `results/figures/comprehensive_comparison.png`

### The One Number That Matters

Open `results/tables/recency_metrics.csv` and look for:

```
Target = Most Recent (%): DIY 18.56%, GeoLife 27.18%
```

This 8.6% difference explains why GeoLife drops 46.7% vs DIY's 8.3%.

---

## Key Files Reference

| File | Contents |
|------|----------|
| `comprehensive_comparison.png` | Best overview figure |
| `recency_pattern_analysis.png` | Explains the mechanism |
| `metric_comparison.csv` | All mobility metrics |
| `analysis_results.json` | Complete numerical results |

---

## Common Issues

### "Model file not found"

Skip model analysis by running only:
```bash
python scripts/gap_performance_diy_geolife_v2/analyze_mobility_patterns.py
python scripts/gap_performance_diy_geolife_v2/analyze_recency_patterns.py
```

### "Data file not found"

Ensure data files exist:
```bash
ls data/diy_eps50/processed/diy_eps50_prev7_test.pk
ls data/geolife_eps20/processed/geolife_eps20_prev7_test.pk
```

### "Import error for pointer_v45"

Add project root to Python path:
```bash
export PYTHONPATH=/data/next_loc_clean_v2:$PYTHONPATH
```

---

## Next Steps

1. **Read the findings**: See `README.md` for complete analysis
2. **Understand the code**: See `TECHNICAL_REFERENCE.md` for script details
3. **Interpret figures**: See `FIGURES_GALLERY.md` for plot explanations

---

## TL;DR

**Question**: Why does removing pointer hurt GeoLife 46.7% vs DIY 8.3%?

**Answer**: GeoLife users return to their most recent location 27.2% of the time (vs DIY's 18.6%). The pointer mechanism captures this; generation cannot.

**Run this to see**:
```bash
python scripts/gap_performance_diy_geolife_v2/analyze_recency_patterns.py
```

---

*Quick Start Version: 1.0*
