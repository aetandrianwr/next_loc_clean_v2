# 1st-Order Markov Chain Model for Next Location Prediction

## Comprehensive Documentation Overview

**Version:** 1.0  
**Last Updated:** January 2, 2026  
**Author:** Documentation Team  
**Repository:** `next_loc_clean_v2/src/models/baseline/markov_ori/`

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [What is This Model?](#what-is-this-model)
3. [Why Use a Markov Chain for Location Prediction?](#why-use-a-markov-chain-for-location-prediction)
4. [Key Characteristics](#key-characteristics)
5. [Documentation Structure](#documentation-structure)
6. [Quick Start](#quick-start)
7. [Performance Summary](#performance-summary)

---

## Executive Summary

The **1st-Order Markov Chain Model** is a statistical baseline model for predicting a user's next location based on their mobility history. This model implements the fundamental assumption that **the probability of visiting a location depends only on the current location**, disregarding all prior history.

This documentation covers `run_markov_ori.py`, which is a faithful reproduction of the original Markov baseline from `location-prediction-ori-freeze/baselines/markov.py`.

### Key Facts at a Glance

| Aspect | Value |
|--------|-------|
| **Model Type** | Statistical / Probabilistic |
| **Order** | 1st-Order (memoryless beyond current state) |
| **Learning Paradigm** | Frequency counting (no gradient-based optimization) |
| **User Personalization** | Per-user transition matrices |
| **Training Time** | ~5 seconds (GeoLife), ~43 seconds (DIY) |
| **Primary Metric** | Acc@1: 24.18% (GeoLife), 44.13% (DIY) |

---

## What is This Model?

### The Core Idea

Imagine you're trying to predict where someone will go next. The Markov Chain approach says:

> *"To predict where someone will go next, I only need to know where they are right now."*

This is called the **Markov Property** or **memorylessness**. The model learns transition patterns like:
- "When at location A, user X usually goes to location B (70% of the time) or location C (30% of the time)"

### Visual Intuition

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MARKOV CHAIN CONCEPT                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│    User's Location History:                                         │
│    ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐                 │
│    │Home │ → │Work │ → │Cafe │ → │Work │ → │ ??? │                 │
│    └─────┘   └─────┘   └─────┘   └─────┘   └─────┘                 │
│                                      ↓                              │
│                              ONLY THIS MATTERS                      │
│                                      ↓                              │
│    1st-Order Markov:  P(Next | Work) = ?                           │
│                                                                     │
│    From training data:                                              │
│    • Work → Home:   45%                                            │
│    • Work → Cafe:   30%                                            │
│    • Work → Gym:    25%                                            │
│                                                                     │
│    Prediction: HOME (highest probability)                           │
└─────────────────────────────────────────────────────────────────────┘
```

### What Makes This "Original" Implementation Special?

The `markov_ori` implementation differs from other variants by:
1. **Using raw CSV data** (not preprocessed pickle files)
2. **Original time-based splitting** (60% train, 20% validation, 20% test by tracked days)
3. **Consecutive pair evaluation** (evaluates all consecutive location pairs in trajectories)
4. **Original metric functions** (self-contained evaluation, not using external `metrics.py`)

---

## Why Use a Markov Chain for Location Prediction?

### Strengths

| Strength | Explanation |
|----------|-------------|
| **Simplicity** | No neural networks, no hyperparameter tuning, just counting |
| **Interpretability** | You can directly inspect transition probabilities |
| **Speed** | Training takes seconds, inference is instant |
| **No GPU Required** | Runs on any hardware |
| **Baseline Quality** | Establishes a meaningful lower bound for comparison |
| **Personalization** | Learns individual user patterns naturally |

### Limitations

| Limitation | Explanation |
|------------|-------------|
| **Limited Memory** | Only considers current location (ignores trajectory patterns) |
| **No Temporal Awareness** | Doesn't distinguish weekday vs weekend, morning vs evening |
| **Cold Start Problem** | Cannot predict for unseen locations or users |
| **Sparsity** | Rare transitions may not be captured |

### When to Use This Model

✅ **Use as a baseline** when developing more complex models  
✅ **Use for interpretability** when you need to explain predictions  
✅ **Use for quick prototyping** before investing in neural approaches  
❌ **Don't use** if temporal patterns are critical  
❌ **Don't use** if you need to capture long-range dependencies  

---

## Key Characteristics

### Mathematical Foundation

The 1st-Order Markov Chain models location transitions as:

```
P(L_{t+1} | L_t, L_{t-1}, ..., L_1) ≈ P(L_{t+1} | L_t)
```

Where:
- `L_t` = Location at time t
- `L_{t+1}` = Next location (what we want to predict)

The transition probability is estimated from frequency counts:

```
P(L_{t+1} = j | L_t = i) = Count(i → j) / Count(i → *)
```

### Per-User Personalization

The model maintains **separate transition matrices for each user**:

```
User 1:          User 2:
     A   B   C        A   B   C
A  [ 0  .8  .2 ]  A  [ 0  .5  .5 ]
B  [.3   0  .7 ]  B  [.6   0  .4 ]
C  [.5  .5   0 ]  C  [.2  .8   0 ]
```

This captures that different users have different mobility patterns.

---

## Documentation Structure

This comprehensive documentation is organized into the following documents:

| Document | Description |
|----------|-------------|
| **01_OVERVIEW.md** | This file - high-level introduction and summary |
| **02_THEORY_BACKGROUND.md** | Mathematical theory, probability concepts, Markov property |
| **03_TECHNICAL_IMPLEMENTATION.md** | Code architecture, data flow, functions |
| **04_COMPONENTS_DEEP_DIVE.md** | Detailed explanation of each component |
| **05_DIAGRAMS_VISUALIZATIONS.md** | Visual diagrams at multiple detail levels |
| **06_RESULTS_ANALYSIS.md** | Performance metrics, interpretation, comparison |
| **07_WALKTHROUGH_LINE_BY_LINE.md** | Step-by-step example with actual code execution |

---

## Quick Start

### Prerequisites

```bash
# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Navigate to project root
cd /data/next_loc_clean_v2
```

### Running the Model

**GeoLife Dataset:**
```bash
python src/models/baseline/markov_ori/run_markov_ori.py \
    --config config/models/config_markov_ori_geolife.yaml
```

**DIY Dataset:**
```bash
python src/models/baseline/markov_ori/run_markov_ori.py \
    --config config/models/config_markov_ori_diy.yaml
```

### Output Location

Results are saved to:
```
experiments/{dataset}_markov_ori_{yyyyMMdd_hhmmss}/
├── checkpoints/           # (empty for Markov model)
├── training.log           # Complete training log
├── config.yaml            # Flattened configuration
├── config_original.yaml   # Copy of input config
├── val_results.json       # Validation metrics
└── test_results.json      # Test metrics
```

---

## Performance Summary

### GeoLife Dataset

| Metric | Validation | Test |
|--------|------------|------|
| **Acc@1** | 33.57% | **24.18%** |
| Acc@5 | 47.43% | 37.87% |
| Acc@10 | 48.59% | 38.76% |
| MRR | 39.91% | 30.34% |
| F1 | 32.87% | 23.38% |
| NDCG@10 | 42.01% | 32.38% |

**Dataset Statistics:**
- Total records: 16,600
- Filtered training records: 7,424
- Validation samples: 3,289
- Test samples: 3,457
- Number of users: 45
- Total parameters (transitions): 166,309
- Training time: ~5 seconds

### DIY Dataset

| Metric | Validation | Test |
|--------|------------|------|
| **Acc@1** | 48.10% | **44.13%** |
| Acc@5 | 67.24% | 62.56% |
| Acc@10 | 69.48% | 64.80% |
| MRR | 56.30% | 52.13% |
| F1 | 46.01% | 42.68% |
| NDCG@10 | 59.51% | 55.22% |

**Dataset Statistics:**
- Validation samples: 26,499
- Test samples: 26,872
- Number of users: 692
- Total parameters (transitions): 366,338
- Training time: ~43 seconds

### Why DIY Performs Better

The DIY dataset shows higher accuracy because:
1. **More data per user** on average
2. **More regular mobility patterns** in the DIY population
3. **Higher location revisitation** (people return to same places more often)

---

## File Dependencies

```
run_markov_ori.py
├── Input
│   ├── config/models/config_markov_ori_*.yaml
│   ├── data/*/markov_ori_data/dataset_*.csv
│   └── data/*/markov_ori_data/valid_ids_*.pk (or generated)
├── External Libraries
│   ├── pandas (data manipulation)
│   ├── numpy (numerical operations)
│   ├── sklearn (F1 score, recall)
│   └── tqdm (progress bars)
└── Output
    └── experiments/*/
        ├── test_results.json
        ├── val_results.json
        └── training.log
```

---

## Navigation

| Previous | Next |
|----------|------|
| - | [02_THEORY_BACKGROUND.md](02_THEORY_BACKGROUND.md) |

---

## References

1. **Original Implementation:** `location-prediction-ori-freeze/baselines/markov.py`
2. **Alternative Implementation:** `src/models/baseline/markov1st.py` (uses preprocessed data)
3. **Evaluation Metrics:** `src/evaluation/metrics.py`
4. **Related Documentation:** `docs/markov1st_baseline.md`
