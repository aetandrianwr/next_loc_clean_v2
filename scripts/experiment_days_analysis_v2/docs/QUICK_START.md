# Quick Start Guide

## Day-of-Week Analysis Experiment V2

A fast introduction to running and understanding the experiment.

---

## What This Experiment Does

This experiment analyzes how **next location prediction accuracy varies by day of the week**. The key question: **Is weekend prediction harder than weekday prediction?**

### Quick Answer

| Dataset | Weekend Effect | Significant? |
|---------|----------------|--------------|
| **DIY** | -2.15% | No (p=0.24) |
| **GeoLife** | **-15.56%** | **Yes (p=0.001)** |

**Bottom line:** GeoLife shows dramatic weekend degradation; DIY shows minimal effect.

---

## 5-Minute Quick Start

### Step 1: Run the Analysis

```bash
cd /data/next_loc_clean_v2/scripts/experiment_days_analysis_v2
python run_days_analysis.py --dataset both
```

Expected output:
```
================================================================================
DAY-OF-WEEK ANALYSIS: DIY Dataset
================================================================================
Device: cuda
...
[1/7] Evaluating Monday...
  Samples: 2020
  Acc@1: 57.28%, Acc@5: 82.43%, MRR: 68.12%
...
```

### Step 2: Generate Visualizations

```bash
python generate_visualizations.py
```

Expected output:
```
================================================================================
GENERATING DAY-OF-WEEK ANALYSIS VISUALIZATIONS (V2 - Classic Style)
================================================================================
Loaded DIY results
Loaded GeoLife results
Saved: diy_accuracy_by_day
Saved: diy_weekday_weekend_comparison
...
```

### Step 3: View Results

```bash
# View main figure
open figures/combined_figure.pdf  # macOS
xdg-open figures/combined_figure.pdf  # Linux

# Or view PNG
ls figures/*.png
```

---

## Key Files

| File | Purpose |
|------|---------|
| `run_days_analysis.py` | Main experiment script |
| `generate_visualizations.py` | Create figures |
| `results/*.json` | Raw numerical results |
| `figures/combined_figure.pdf` | Publication-ready figure |
| `figures/days_analysis_summary.csv` | All data in spreadsheet format |

---

## Understanding the Results

### The Key Plot

Open `combined_figure.pdf` and look at:

1. **Panels (a) and (b)**: Bar charts showing Acc@1 for each day
   - Green bars = weekdays
   - Orange bars = weekends
   - **Lower weekend bars = weekend effect**

2. **Panel (c)**: Line plot comparing both datasets
   - DIY (blue) stays relatively flat
   - GeoLife (red) drops sharply on Sat/Sun

3. **Panel (i)**: Statistical summary
   - DIY: p = 0.24 (not significant)
   - GeoLife: p = 0.0015 (highly significant)

### Key Numbers

**DIY Dataset:**
- Weekday Acc@1: 57.24%
- Weekend Acc@1: 55.09%
- Difference: 2.15% (not statistically significant)

**GeoLife Dataset:**
- Weekday Acc@1: 55.26%
- Weekend Acc@1: 39.70%
- Difference: **15.56%** (highly significant, p < 0.01)

---

## Common Questions

### Q: Why is GeoLife's weekend effect so much larger?

**A:** GeoLife participants were researchers/students with very routine weekday schedules (commute to lab). Their weekends are unpredictable leisure time. DIY has more diverse users with less extreme weekday-weekend differences.

### Q: Which day is easiest to predict?

**A:** 
- DIY: Tuesday (61.53% Acc@1)
- GeoLife: Wednesday (59.88% Acc@1)

Mid-week days are most routine.

### Q: Which day is hardest to predict?

**A:** Saturday for both datasets (54.90% DIY, 37.58% GeoLife).

### Q: Can I run for just one dataset?

**A:** Yes:
```bash
python run_days_analysis.py --dataset diy
python run_days_analysis.py --dataset geolife
```

---

## Customization

### Change Output Directory

```bash
python run_days_analysis.py --output_dir ./my_results
python generate_visualizations.py --results_dir ./my_results --output_dir ./my_figures
```

### Change Random Seed

```bash
python run_days_analysis.py --seed 123
```

---

## Troubleshooting

### "CUDA out of memory"

The experiment uses GPU by default. If memory issues:
- Results should be the same on CPU (just slower)
- Reduce batch_size in `evaluate_on_day()` if needed

### "FileNotFoundError: No such file"

Check that the data and checkpoint paths in `CONFIGS` are correct:
```python
python -c "from run_days_analysis import CONFIGS; print(CONFIGS['diy']['checkpoint'])"
```

### Missing visualizations

Run both scripts in order:
```bash
python run_days_analysis.py --dataset both  # Creates results/*.json
python generate_visualizations.py            # Creates figures/*
```

---

## Next Steps

1. **Read the full documentation**: `docs/README.md`
2. **Understand the API**: `docs/TECHNICAL_API.md`
3. **Interpret visualizations**: `docs/VISUALIZATION_GUIDE.md`
4. **Cite in your paper**: Use tables from `figures/*_table.tex`

---

## Citation

If using these results in your research:

```
Day-of-Week Analysis Experiment for Next Location Prediction
Part of PhD Thesis Research on Human Mobility Prediction
2026
```

---

*Quick Start Guide - Last Updated: January 2, 2026*
