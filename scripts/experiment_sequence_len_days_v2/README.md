# Sequence Length Days Experiment (V2 - Publication Style)

## Overview

This experiment investigates **how the length of historical mobility data (in days) affects next location prediction performance**. The study systematically evaluates the impact of using 1 to 7 days of prior location history on the model's prediction accuracy.

**V2 Updates:** This version includes publication-quality visualizations matching the classic scientific journal style with:
- White background with black axis box (all 4 sides)
- Inside tick marks
- No grid lines
- Open markers (white fill with colored edges)
- Blue/red color scheme
- Solid/dashed line differentiation
- Hatched bar charts

## Research Question

> How does the temporal window of user mobility history affect prediction accuracy?

## Key Findings

| Finding | DIY | GeoLife |
|---------|-----|---------|
| Best performance | 7 days | 7 days |
| Acc@1 improvement (1→7) | +13.2% | +7.4% |
| Acc@5 improvement (1→7) | +13.3% | +16.0% |
| Optimal trade-off | 3-4 days | 3-4 days |

## Files

```
experiment_sequence_len_days_v2/
├── evaluate_sequence_length.py  # Main evaluation script
├── visualize_results.py         # Visualization generation (V2 style)
├── run_experiment.sh            # Master run script
├── README.md                    # This file
└── results/
    ├── diy_sequence_length_results.json
    ├── geolife_sequence_length_results.json
    ├── full_results.csv
    ├── summary_statistics.csv
    ├── improvement_analysis.csv
    ├── results_table.tex        # LaTeX table
    ├── statistics_table.tex
    ├── combined_figure.pdf      # Main publication figure
    ├── performance_comparison.{pdf,png,svg}
    ├── accuracy_heatmap.{pdf,png}
    ├── improvement_comparison.{pdf,png}
    ├── loss_curve.{pdf,png}
    ├── radar_comparison.{pdf,png}
    ├── sequence_length_distribution.{pdf,png}
    └── samples_vs_performance.{pdf,png}
```

## Quick Start

```bash
# Run complete experiment
./run_experiment.sh

# Or run individual components:
# 1. Run evaluation
python evaluate_sequence_length.py --dataset all

# 2. Generate visualizations
python visualize_results.py
```

## Visualization Style

The V2 visualizations follow classic scientific publication style (matching reference images):

| Feature | Implementation |
|---------|---------------|
| Background | White |
| Axes | Black box (all 4 sides) |
| Ticks | Inside, on all sides |
| Grid | None |
| Markers | Open (white fill) |
| Colors | Blue (DIY), Red (GeoLife) |
| Line styles | Solid (DIY), Dashed (GeoLife) |
| Bar charts | White fill with hatching |
| Font | Serif (Times-like) |

## Configuration

The experiment uses pre-trained checkpoints:

| Dataset | Checkpoint | Config |
|---------|------------|--------|
| DIY | `experiments/diy_pointer_v45_20260101_155348/` | `pointer_v45_diy_trial09.yaml` |
| GeoLife | `experiments/geolife_pointer_v45_20260101_151038/` | `pointer_v45_geolife_trial01.yaml` |

## Results Summary

### DIY Dataset

| Prev Days | Acc@1 | Acc@5 | Acc@10 | MRR | Loss |
|-----------|-------|-------|--------|-----|------|
| 1 | 50.00% | 72.55% | 74.65% | 59.97% | 3.763 |
| 7 | **56.58%** | **82.18%** | **85.16%** | **67.67%** | **2.874** |

### GeoLife Dataset

| Prev Days | Acc@1 | Acc@5 | Acc@10 | MRR | Loss |
|-----------|-------|-------|--------|-----|------|
| 1 | 47.84% | 70.00% | 74.32% | 57.83% | 3.492 |
| 7 | **51.40%** | **81.18%** | **85.04%** | **64.55%** | **2.630** |

## Documentation

See full documentation: [`docs/experiment_sequence_length_days.md`](../../docs/experiment_sequence_length_days.md)

## Environment

- Python 3.9+
- PyTorch 2.0+
- Matplotlib, Seaborn
- Conda environment: `mlenv`
- Seed: 42 (for reproducibility)
