# Sequence Length Days Experiment

## Overview

This experiment investigates **how the length of historical mobility data (in days) affects next location prediction performance**. The study systematically evaluates the impact of using 1 to 7 days of prior location history on the model's prediction accuracy.

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
experiment_sequence_len_days/
├── evaluate_sequence_length.py  # Main evaluation script
├── visualize_results.py         # Visualization generation
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

## Citation

If you use this experiment in your research, please cite:

```bibtex
@misc{sequence_length_experiment_2026,
  title={Impact of Historical Sequence Length on Next Location Prediction},
  author={PhD Research Team},
  year={2026},
  note={Ablation Study - Sequence Length Analysis}
}
```

## Environment

- Python 3.9+
- PyTorch 2.0+
- Conda environment: `mlenv`
- Seed: 42 (for reproducibility)
