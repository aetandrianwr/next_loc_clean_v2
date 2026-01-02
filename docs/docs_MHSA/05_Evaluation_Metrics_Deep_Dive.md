# Evaluation Metrics Deep Dive

## Understanding the Metrics Used in Next Location Prediction

---

## Table of Contents

1. [Overview](#1-overview)
2. [Accuracy@k (Acc@k)](#2-accuracyk-acck)
3. [Mean Reciprocal Rank (MRR)](#3-mean-reciprocal-rank-mrr)
4. [Normalized Discounted Cumulative Gain (NDCG)](#4-normalized-discounted-cumulative-gain-ndcg)
5. [F1 Score](#5-f1-score)
6. [Metric Comparison](#6-metric-comparison)
7. [Implementation Details](#7-implementation-details)
8. [Interpreting Results](#8-interpreting-results)

---

## 1. Overview

### 1.1 Why Multiple Metrics?

Next location prediction is a **ranking problem** where we want the correct location to appear high in our predictions. Different metrics capture different aspects:

| Metric | Measures | Use Case |
|--------|----------|----------|
| Acc@1 | Exact accuracy | How often we're exactly right |
| Acc@5 | Top-5 accuracy | How often correct is in top 5 |
| MRR | Average rank quality | Overall ranking performance |
| NDCG@10 | Position-weighted ranking | Quality of top-10 rankings |
| F1 | Classification quality | Per-class performance balance |

### 1.2 Metric Summary

For a prediction ranking locations from most to least likely:

```
Prediction Ranking: [Cafe, Work, Home, Gym, Mall, ...]
                     1st   2nd   3rd   4th   5th

If correct answer is "Home" (rank 3):
- Acc@1: 0 (not in top 1)
- Acc@3: 1 (in top 3)
- Acc@5: 1 (in top 5)
- MRR: 1/3 = 0.333
- NDCG: 1/log₂(3+1) = 0.5
```

---

## 2. Accuracy@k (Acc@k)

### 2.1 Definition

Accuracy@k measures the proportion of test samples where the correct location appears in the top-k predictions.

```
Acc@k = (# samples with correct in top k) / (total samples) × 100%
```

### 2.2 Mathematical Formulation

For a batch of N samples with predictions P and ground truth Y:

```
Acc@k = (1/N) × Σᵢ 𝟙[yᵢ ∈ topk(Pᵢ)]

where:
- topk(Pᵢ) = set of k locations with highest predicted scores
- 𝟙[condition] = 1 if condition is true, 0 otherwise
```

### 2.3 Calculation Example

```
Sample 1: Predictions = [Cafe:0.4, Work:0.3, Home:0.2, Gym:0.1]
          Ground Truth = Work
          Top-1 = {Cafe}     → Acc@1 contribution: 0
          Top-3 = {Cafe, Work, Home} → Acc@3 contribution: 1

Sample 2: Predictions = [Home:0.5, Work:0.3, Cafe:0.1, Gym:0.1]
          Ground Truth = Home
          Top-1 = {Home}     → Acc@1 contribution: 1
          Top-3 = {Home, Work, Cafe} → Acc@3 contribution: 1

Overall:
Acc@1 = (0 + 1) / 2 = 50%
Acc@3 = (1 + 1) / 2 = 100%
```

### 2.4 Properties

- **Range**: 0% to 100%
- **Higher is better**
- **Acc@k ≥ Acc@(k-1)** always (larger k is easier)
- **Acc@V = 100%** for V = vocabulary size

### 2.5 Interpretation Guide

| Acc@1 | Interpretation |
|-------|----------------|
| < 10% | Random baseline level (depends on num locations) |
| 10-30% | Weak model |
| 30-50% | Moderate model |
| 50-70% | Good model |
| > 70% | Excellent model |

---

## 3. Mean Reciprocal Rank (MRR)

### 3.1 Definition

MRR is the average of reciprocal ranks of the correct answer across all samples.

```
MRR = (1/N) × Σᵢ (1 / rankᵢ)

where rankᵢ = position of correct answer in sorted predictions (1-indexed)
```

### 3.2 Intuition

- If correct answer is **1st**: reciprocal rank = 1/1 = 1.0
- If correct answer is **2nd**: reciprocal rank = 1/2 = 0.5
- If correct answer is **3rd**: reciprocal rank = 1/3 = 0.33
- If correct answer is **10th**: reciprocal rank = 1/10 = 0.1

MRR gives **more weight to higher rankings** - getting it in top 3 matters a lot, getting it in rank 50 vs 100 matters little.

### 3.3 Calculation Example

```
Sample 1: Correct = "Work", Rank = 2
          RR = 1/2 = 0.5

Sample 2: Correct = "Home", Rank = 1
          RR = 1/1 = 1.0

Sample 3: Correct = "Gym", Rank = 5
          RR = 1/5 = 0.2

MRR = (0.5 + 1.0 + 0.2) / 3 = 0.567
```

### 3.4 Properties

- **Range**: 0 to 1 (reported as 0% to 100%)
- **Higher is better**
- **MRR = 1** means correct answer is always rank 1
- **MRR approaches 0** as ranks get worse

### 3.5 Relationship to Accuracy

```
If MRR = 40%, typical interpretation:
- Correct answers are usually in top 2-3 positions
- ~40% chance the correct is rank 1, ~60% spread across lower ranks
```

---

## 4. Normalized Discounted Cumulative Gain (NDCG)

### 4.1 Definition

NDCG measures ranking quality with a logarithmic discount for lower positions.

```
DCG@k = Σᵢ (relᵢ) / log₂(i + 1)   for i = 1 to k

NDCG@k = DCG@k / IDCG@k

where IDCG is the ideal (best possible) DCG
```

For our binary relevance (correct/incorrect):

```
If correct answer is at rank r (where r ≤ k):
    NDCG = 1 / log₂(r + 1)
    
If correct answer is beyond rank k:
    NDCG = 0
```

### 4.2 Intuition

NDCG uses a **logarithmic discount**:

| Rank | Discount Factor | NDCG |
|------|-----------------|------|
| 1 | 1/log₂(2) = 1.0 | 1.000 |
| 2 | 1/log₂(3) = 0.631 | 0.631 |
| 3 | 1/log₂(4) = 0.5 | 0.500 |
| 5 | 1/log₂(6) = 0.387 | 0.387 |
| 10 | 1/log₂(11) = 0.289 | 0.289 |
| >10 | - | 0.000 |

The discount is **less aggressive** than MRR's 1/rank.

### 4.3 Calculation Example

```
Sample 1: Correct at rank 1
          NDCG = 1/log₂(2) = 1.0

Sample 2: Correct at rank 3
          NDCG = 1/log₂(4) = 0.5

Sample 3: Correct at rank 7
          NDCG = 1/log₂(8) = 0.333

Sample 4: Correct at rank 15 (beyond k=10)
          NDCG = 0

Average NDCG@10 = (1.0 + 0.5 + 0.333 + 0) / 4 = 0.458
```

### 4.4 NDCG vs MRR

| Aspect | MRR (1/rank) | NDCG (1/log₂) |
|--------|--------------|---------------|
| Rank 1 | 1.00 | 1.00 |
| Rank 2 | 0.50 | 0.63 |
| Rank 3 | 0.33 | 0.50 |
| Rank 5 | 0.20 | 0.39 |
| Rank 10 | 0.10 | 0.29 |
| Decay | Aggressive | Gradual |

NDCG is **more forgiving** of lower ranks than MRR.

---

## 5. F1 Score

### 5.1 Definition

F1 Score is the harmonic mean of precision and recall. For multi-class classification, we use **weighted F1**:

```
F1_weighted = Σc (supportc / total) × F1c

where F1c = 2 × (Precisionc × Recallc) / (Precisionc + Recallc)
```

### 5.2 Per-Class Precision and Recall

For each location class c:

```
Precision_c = True Positives_c / Predicted as c
Recall_c = True Positives_c / Actually class c
```

### 5.3 Example

```
Confusion Matrix (simplified 3 classes):
              Predicted
              Home  Work  Cafe
Actual Home    50    10    5     (65 total Home samples)
Actual Work     5    80   15     (100 total Work samples)
Actual Cafe    10    20   35     (65 total Cafe samples)

Home:
  Precision = 50 / (50+5+10) = 0.77
  Recall = 50 / 65 = 0.77
  F1_Home = 2 × 0.77 × 0.77 / (0.77 + 0.77) = 0.77

Work:
  Precision = 80 / (10+80+20) = 0.73
  Recall = 80 / 100 = 0.80
  F1_Work = 2 × 0.73 × 0.80 / (0.73 + 0.80) = 0.76

Cafe:
  Precision = 35 / (5+15+35) = 0.64
  Recall = 35 / 65 = 0.54
  F1_Cafe = 2 × 0.64 × 0.54 / (0.64 + 0.54) = 0.59

Weighted F1 = (65×0.77 + 100×0.76 + 65×0.59) / 230 = 0.71
```

### 5.4 Why Weighted F1?

- Handles **class imbalance** - frequent locations don't dominate
- Balances precision and recall
- Useful when all classes matter equally

### 5.5 Typical F1 Ranges

| F1 Score | Interpretation |
|----------|----------------|
| < 0.2 | Poor |
| 0.2-0.4 | Fair |
| 0.4-0.6 | Moderate |
| 0.6-0.8 | Good |
| > 0.8 | Excellent |

---

## 6. Metric Comparison

### 6.1 When to Use Each Metric

| Metric | Best For | Limitation |
|--------|----------|------------|
| Acc@1 | Exact prediction tasks | Ignores near-misses |
| Acc@5 | Recommendation systems | Doesn't distinguish ranks |
| MRR | Search/retrieval | Harsh on lower ranks |
| NDCG | Ranked recommendations | Cutoff at k ignores rest |
| F1 | Class balance analysis | Only top-1 prediction |

### 6.2 Typical Relationships

For a well-calibrated model:

```
Acc@1 < MRR < NDCG < Acc@5 < Acc@10

Example:
Acc@1 = 30%
MRR = 42%
NDCG = 46%
Acc@5 = 55%
Acc@10 = 62%
```

### 6.3 Performance Profiles

**Profile A: High Acc@1, Close MRR**
```
Acc@1=40%, MRR=45%, NDCG=50%
Interpretation: Model often gets exact answer or close second
```

**Profile B: Low Acc@1, Higher MRR**
```
Acc@1=20%, MRR=35%, NDCG=40%
Interpretation: Model rarely gets exact, but correct in top 3-5
```

**Profile C: Flat metrics**
```
Acc@1=25%, MRR=30%, NDCG=35%
Interpretation: Predictions spread across many positions
```

---

## 7. Implementation Details

### 7.1 Code Structure

**Location:** `src/evaluation/metrics.py`

```python
def calculate_correct_total_prediction(logits, true_y):
    """
    Calculate all metrics for a batch.
    
    Args:
        logits: [batch_size, num_locations] - model predictions
        true_y: [batch_size] - ground truth indices
    
    Returns:
        result_array: [correct@1, correct@3, correct@5, correct@10, rr_sum, ndcg_sum, total]
        true_labels: for F1 calculation
        top1_predictions: for F1 calculation
    """
```

### 7.2 Accuracy@k Implementation

```python
def top_k_accuracy(logits, targets, k):
    # Get top-k predictions
    _, top_k_preds = torch.topk(logits, k, dim=-1)
    
    # Check if target is in top-k
    correct = (targets.unsqueeze(-1) == top_k_preds).any(dim=-1)
    
    return correct.sum().item()
```

### 7.3 MRR Implementation

```python
def get_mrr(prediction, targets):
    # Sort predictions descending
    sorted_indices = torch.argsort(prediction, dim=-1, descending=True)
    
    # Find rank of correct answer
    hits = (targets.unsqueeze(-1) == sorted_indices).nonzero()
    ranks = (hits[:, -1] + 1).float()  # 1-indexed
    
    # Reciprocal rank
    return torch.sum(1.0 / ranks).item()
```

### 7.4 NDCG Implementation

```python
def get_ndcg(prediction, targets, k=10):
    # Get ranks
    sorted_indices = torch.argsort(prediction, dim=-1, descending=True)
    hits = (targets.unsqueeze(-1) == sorted_indices).nonzero()
    ranks = (hits[:, -1] + 1).float().numpy()
    
    # NDCG with log discount
    ndcg = 1 / np.log2(ranks + 1)
    
    # Zero out beyond k
    ndcg[ranks > k] = 0
    
    return ndcg.sum()
```

---

## 8. Interpreting Results

### 8.1 Sample Results Analysis

```json
{
  "acc@1": 29.61,
  "acc@5": 54.48,
  "acc@10": 58.94,
  "mrr": 40.84,
  "ndcg": 44.96,
  "f1": 21.20
}
```

**Interpretation:**
- **Acc@1=30%**: Correct prediction ~1 in 3 times
- **Acc@5=54%**: Correct in top 5 ~half the time
- **MRR=41%**: Average rank is around 2.4 (1/0.41)
- **NDCG=45%**: Moderate ranking quality
- **F1=21%**: Lower due to many location classes with imbalance

### 8.2 Comparing Models

When comparing two models:

```
Model A: Acc@1=30%, MRR=42%, NDCG=46%
Model B: Acc@1=32%, MRR=40%, NDCG=44%

Analysis:
- Model B is better at exact predictions (+2% Acc@1)
- Model A is better at ranking (higher MRR, NDCG)
- Choose based on use case:
  - Exact prediction needed → Model B
  - Top-k suggestions okay → Model A
```

### 8.3 Statistical Significance

For robust comparison:
- Run with multiple seeds (3-5)
- Report mean ± std
- Use t-test for significance

```
Model A: Acc@1 = 30.2 ± 0.8%
Model B: Acc@1 = 31.5 ± 0.6%
Difference: 1.3% (may or may not be significant)
```

### 8.4 Baseline Comparisons

**Random Baseline:**
```
Acc@1 = 1 / num_locations
MRR ≈ 1 / log(num_locations)
```

For GeoLife (1187 locations):
```
Random Acc@1 ≈ 0.08%
Random MRR ≈ 14%
```

**Majority Class Baseline:**
```
Acc@1 = frequency of most common location
```

---

## Appendix: Quick Reference

### Metric Formulas

| Metric | Formula |
|--------|---------|
| Acc@k | Σ 𝟙[y ∈ topk(pred)] / N |
| MRR | Σ (1/rank) / N |
| NDCG@k | Σ (1/log₂(rank+1)) / N, for rank ≤ k |
| F1 | 2PR/(P+R) weighted by class support |

### Expected Ranges (Good Model)

| Dataset | Acc@1 | MRR | NDCG |
|---------|-------|-----|------|
| GeoLife | 28-35% | 38-45% | 42-48% |
| DIY | 50-55% | 60-65% | 65-70% |

---

*Evaluation metrics documentation for MHSA model - next_loc_clean_v2 project*
