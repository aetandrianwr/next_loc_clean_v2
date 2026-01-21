# Scientific Hyperparameter Tuning - Overview

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Statement](#problem-statement)
3. [Why Hyperparameter Tuning?](#why-hyperparameter-tuning)
4. [Project Goals](#project-goals)
5. [Key Contributions](#key-contributions)
6. [Directory Structure](#directory-structure)

---

## Introduction

This documentation provides a comprehensive guide to the **Scientific Hyperparameter Tuning** experiments conducted for next location prediction models. The goal is to fairly compare three deep learning architectures—**Pointer Generator Transformer** (proposed model), **Multi-Head Self-Attention (MHSA)**, and **LSTM** (baseline models)—across two mobility datasets: **Geolife** and **DIY**.

Hyperparameter tuning is a critical step in machine learning research. Without proper tuning, model comparisons can be misleading because poorly-tuned models may underperform due to suboptimal configurations rather than fundamental architectural limitations. This project implements a rigorous, reproducible hyperparameter tuning methodology following best practices from the machine learning research community.

---

## Problem Statement

### Next Location Prediction

**Next location prediction** is the task of predicting where a user will go next based on their historical mobility patterns. This is a fundamental problem in:

- **Urban Computing**: Understanding human mobility for city planning
- **Location-Based Services**: Recommending places, targeted advertising
- **Transportation**: Traffic prediction, ride-sharing optimization
- **Public Health**: Epidemic modeling, contact tracing

Given a user's sequence of visited locations with associated temporal information (time, day, duration), the model predicts the probability distribution over all possible next locations.

### Mathematical Formulation

Let:
- $U$ = set of users
- $L$ = set of locations (vocabulary)
- $H_u = [(l_1, t_1), (l_2, t_2), ..., (l_n, t_n)]$ = location history for user $u$

The goal is to learn a function:
$$f(H_u) \rightarrow P(l_{n+1} | H_u)$$

where $P(l_{n+1})$ is a probability distribution over all locations in $L$.

---

## Why Hyperparameter Tuning?

### The Problem with Default Hyperparameters

Deep learning models are sensitive to hyperparameter choices. Consider these scenarios:

1. **Learning Rate Too High**: Model diverges, loss explodes
2. **Learning Rate Too Low**: Training is slow, gets stuck in local minima
3. **Model Too Small**: Underfitting, cannot capture complex patterns
4. **Model Too Large**: Overfitting, poor generalization
5. **Dropout Too Low**: Overfitting
6. **Dropout Too High**: Underfitting, model cannot learn

### Fair Comparison Requirements

To claim that Model A is better than Model B, we must ensure:
1. Both models are optimally tuned on the same dataset
2. Same computational budget for tuning
3. Same evaluation protocol
4. Reproducible experiments

Without proper tuning, a poorly-configured baseline might unfairly lose to a well-tuned proposed model.

### Academic Standards

Following **Bergstra & Bengio (2012)**, we adopt **Random Search** over grid search because:
- More efficient: Random search explores more unique hyperparameter values
- Handles importance hierarchy: Important hyperparameters are explored more effectively
- Reproducible: Fixed random seed ensures identical configurations

---

## Project Goals

### Primary Objectives

1. **Fair Model Comparison**: Tune all three models (Pointer Generator Transformer, MHSA, LSTM) with equal effort
2. **Reproducibility**: All experiments use fixed seeds and logged configurations
3. **Scientific Rigor**: Follow established hyperparameter tuning methodology
4. **Comprehensive Analysis**: Report not just best results, but variance across trials

### Secondary Objectives

1. Identify optimal hyperparameter ranges for each model
2. Understand hyperparameter sensitivity (which parameters matter most?)
3. Generate publishable results with statistical significance

---

## Key Contributions

### Technical Contributions

1. **Automated Tuning Pipeline**
   - Generate 120 configurations automatically
   - Parallel execution with 5 concurrent jobs
   - Automatic result logging to CSV files
   - Resume capability for interrupted experiments

2. **Comprehensive Search Spaces**
   - Architecture parameters: model size, depth, attention heads
   - Optimization parameters: learning rate, weight decay, batch size
   - Regularization: dropout rates, label smoothing

3. **Reproducibility Infrastructure**
   - Fixed random seed (42) for all experiments
   - YAML configuration files for every trial
   - Complete result logging with timestamps

### Scientific Contributions

1. **Validated Pointer Generator Transformer Architecture**: Demonstrated consistent superiority over baselines
2. **Dataset-Specific Insights**: Different performance gaps on Geolife vs. DIY
3. **Hyperparameter Sensitivity Analysis**: Identified critical hyperparameters

---

## Directory Structure

```
sci_hyperparam_tuning/
├── README.md                           # Quick start guide
├── RESULTS_SUMMARY.md                  # High-level results summary
├── hyperparam_search_space.py          # Search space definitions
├── generate_configs.py                 # Configuration generation script
├── run_hyperparam_tuning.py            # Parallel tuning manager
├── run_final_evaluation.py             # Final evaluation with multiple seeds
├── tuning_log.txt                      # Detailed execution log
├── final_eval_log.txt                  # Final evaluation log
│
├── configs/                            # Generated configurations (120+ YAML files)
│   ├── pointer_v45_geolife_trial00.yaml
│   ├── pointer_v45_geolife_trial01.yaml
│   ├── ...
│   ├── lstm_diy_trial19.yaml
│   └── all_configs_summary.yaml        # Summary of all configurations
│
├── results/                            # Experimental results (12 CSV files)
│   ├── pointer_v45_geolife_val_results.csv
│   ├── pointer_v45_geolife_test_results.csv
│   ├── pointer_v45_diy_val_results.csv
│   ├── pointer_v45_diy_test_results.csv
│   ├── mhsa_geolife_val_results.csv
│   ├── mhsa_geolife_test_results.csv
│   ├── mhsa_diy_val_results.csv
│   ├── mhsa_diy_test_results.csv
│   ├── lstm_geolife_val_results.csv
│   ├── lstm_geolife_test_results.csv
│   ├── lstm_diy_val_results.csv
│   └── lstm_diy_test_results.csv
│
└── docs/                               # This documentation
    ├── 01_OVERVIEW.md                  # Introduction (this file)
    ├── 02_METHODOLOGY.md               # Scientific methodology
    ├── 03_SEARCH_SPACE.md              # Hyperparameter search spaces
    ├── 04_IMPLEMENTATION.md            # Code implementation details
    ├── 05_MODELS.md                    # Model architecture details
    ├── 06_RESULTS.md                   # Comprehensive results
    ├── 07_INTERPRETATION.md            # Analysis and interpretation
    └── 08_USAGE.md                     # How to use this system
```

---

## Quick Results Preview

### Geolife Dataset (Best Val Acc@1)

| Rank | Model | Val Acc@1 | Parameters | Improvement |
|------|-------|-----------|------------|-------------|
| 🥇 1 | **Pointer Generator Transformer** | **49.25%** | 443,404 | - |
| 🥈 2 | MHSA | 42.38% | 281,251 | +6.87% |
| 🥉 3 | LSTM | 40.58% | 467,683 | +8.67% |

### DIY Dataset (Best Val Acc@1)

| Rank | Model | Val Acc@1 | Parameters | Improvement |
|------|-------|-----------|------------|-------------|
| 🥇 1 | **Pointer Generator Transformer** | **54.92%** | 1,081,554 | - |
| 🥈 2 | LSTM | 53.90% | 3,564,990 | +1.02% |
| 🥉 3 | MHSA | 53.69% | 797,982 | +1.23% |

**Key Finding**: Pointer Generator Transformer consistently outperforms both baselines across all datasets and metrics, validating the proposed architecture's effectiveness.

---

## Next Steps

Continue reading the documentation in order:
1. **[02_METHODOLOGY.md](02_METHODOLOGY.md)** - Understand the scientific approach
2. **[03_SEARCH_SPACE.md](03_SEARCH_SPACE.md)** - Explore hyperparameter definitions
3. **[04_IMPLEMENTATION.md](04_IMPLEMENTATION.md)** - Dive into the code
4. **[05_MODELS.md](05_MODELS.md)** - Understand model architectures
5. **[06_RESULTS.md](06_RESULTS.md)** - Analyze experimental results
6. **[07_INTERPRETATION.md](07_INTERPRETATION.md)** - Draw conclusions
7. **[08_USAGE.md](08_USAGE.md)** - Reproduce the experiments

---

## References

1. Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research*, 13(2).
2. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
3. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks. *NeurIPS*.
