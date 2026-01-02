# 08. Evaluation Metrics

## Comprehensive Guide to Performance Metrics

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Metric Documentation |
| **Audience** | Researchers, ML Engineers |
| **Reading Time** | 15-18 minutes |
| **Prerequisites** | Basic probability and statistics |

---

## 1. Overview of Metrics

### 1.1 Why Multiple Metrics?

No single metric captures all aspects of prediction quality:

| Metric | What It Measures | When It's Important |
|--------|-----------------|---------------------|
| Acc@1 | Exact match | Single best prediction |
| Acc@5 | Top-5 inclusion | Short list of suggestions |
| Acc@10 | Top-10 inclusion | Browsable recommendation list |
| MRR | Rank of correct answer | Ranked search results |
| NDCG@10 | Ranking quality | Quality of full ranking |
| F1 | Class-balanced accuracy | Imbalanced location distribution |
| Loss | Probability calibration | Confidence of predictions |

### 1.2 Metric Categories

```
┌─────────────────────────────────────────────────────────────┐
│                    EVALUATION METRICS                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Classification Metrics        Ranking Metrics               │
│  ┌───────────────────┐        ┌───────────────────┐         │
│  │ • Accuracy@k      │        │ • MRR             │         │
│  │ • F1 Score        │        │ • NDCG@10         │         │
│  └───────────────────┘        └───────────────────┘         │
│                                                              │
│  Probabilistic Metrics                                       │
│  ┌───────────────────┐                                      │
│  │ • Cross-Entropy   │                                      │
│  │   Loss            │                                      │
│  └───────────────────┘                                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Top-K Accuracy (Acc@k)

### 2.1 Definition

**Accuracy@k** is the percentage of samples where the correct location appears in the top-k predictions.

**Mathematical Formula**:

$$\text{Acc@}k = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[y_i \in \text{TopK}(\hat{\mathbf{p}}_i, k)]$$

where:
- $N$ = number of samples
- $y_i$ = true location for sample $i$
- $\hat{\mathbf{p}}_i$ = predicted probability distribution for sample $i$
- $\text{TopK}(\hat{\mathbf{p}}_i, k)$ = set of k locations with highest predicted probabilities
- $\mathbb{1}[\cdot]$ = indicator function (1 if true, 0 if false)

### 2.2 Implementation

```python
def accuracy_at_k(logits, targets, k):
    """
    Calculate Accuracy@k.
    
    Args:
        logits: [N, num_locations] predicted scores
        targets: [N] true location IDs
        k: number of top predictions to consider
    
    Returns:
        float: Accuracy@k as percentage
    """
    # Get top-k predictions
    _, top_k_indices = torch.topk(logits, k=k, dim=-1)  # [N, k]
    
    # Check if target is in top-k
    targets_expanded = targets.unsqueeze(1)  # [N, 1]
    hits = (targets_expanded == top_k_indices).any(dim=1)  # [N]
    
    # Calculate accuracy
    accuracy = hits.float().mean() * 100
    return accuracy
```

### 2.3 Interpretation Guide

| Acc@k Value | Interpretation |
|-------------|----------------|
| 0-30% | Poor - near random guessing |
| 30-50% | Below average |
| 50-70% | Good - better than majority baseline |
| 70-85% | Very good |
| 85-95% | Excellent |
| 95-100% | Near perfect |

### 2.4 Results from This Experiment

**DIY Dataset**:
| Metric | prev1 | prev7 | Improvement |
|--------|-------|-------|-------------|
| Acc@1 | 50.00% | 56.58% | +6.58 pp |
| Acc@5 | 72.55% | 82.18% | +9.63 pp |
| Acc@10 | 74.65% | 85.16% | +10.50 pp |

**GeoLife Dataset**:
| Metric | prev1 | prev7 | Improvement |
|--------|-------|-------|-------------|
| Acc@1 | 47.84% | 51.40% | +3.56 pp |
| Acc@5 | 70.00% | 81.18% | +11.19 pp |
| Acc@10 | 74.32% | 85.04% | +10.72 pp |

### 2.5 Why Acc@5 and Acc@10 Show Larger Improvements

**Observation**: Acc@5 improves more than Acc@1 (relative to baseline).

**Explanation**:
1. More history helps identify likely candidates even if not the top choice
2. The model learns better ranking of candidates
3. Second and third choices become more accurate with more context

### 2.6 Practical Applications

| Use Case | Recommended Metric |
|----------|-------------------|
| Navigation "Turn left in 100m" | Acc@1 |
| "Suggested destinations" list | Acc@5 |
| "Places you might like" | Acc@10 |

---

## 3. Mean Reciprocal Rank (MRR)

### 3.1 Definition

**MRR** is the average of reciprocal ranks of the correct answer across all samples.

**Mathematical Formula**:

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position (1-indexed) of the correct location in the sorted prediction list.

### 3.2 Implementation

```python
def mean_reciprocal_rank(logits, targets):
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        logits: [N, num_locations] predicted scores
        targets: [N] true location IDs
    
    Returns:
        float: MRR as percentage
    """
    # Sort predictions in descending order
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)  # [N, num_locations]
    
    # Find rank of each target
    targets_expanded = targets.unsqueeze(1)  # [N, 1]
    matches = (sorted_indices == targets_expanded)  # [N, num_locations]
    
    # Get rank (1-indexed)
    ranks = matches.nonzero()[:, 1] + 1  # [N]
    
    # Calculate MRR
    reciprocal_ranks = 1.0 / ranks.float()
    mrr = reciprocal_ranks.mean() * 100
    
    return mrr
```

### 3.3 Understanding Reciprocal Rank

| Rank | Reciprocal Rank | Interpretation |
|------|-----------------|----------------|
| 1 | 1.000 | Perfect - first prediction correct |
| 2 | 0.500 | Good - second prediction correct |
| 3 | 0.333 | Fair - third prediction correct |
| 5 | 0.200 | Mediocre |
| 10 | 0.100 | Poor |
| 100 | 0.010 | Very poor |
| 1000 | 0.001 | Nearly wrong |

### 3.4 Properties of MRR

**Advantages**:
- Considers the full ranking, not just top-k
- Rewards higher rankings more than lower ones
- Continuous metric (vs. discrete Acc@k)
- Gives partial credit for near-correct predictions

**Disadvantages**:
- Doesn't consider positions beyond the correct one
- Heavily influenced by top ranks
- May not reflect user experience for list-based recommendations

### 3.5 MRR vs Accuracy Relationship

```
MRR is always between Acc@1 and 100%:
   Acc@1 ≤ MRR ≤ 100%

If all predictions are rank 1: MRR = 100% = Acc@1
If all predictions are rank 2: MRR = 50%, Acc@1 = 0%
```

### 3.6 Results from This Experiment

| Dataset | prev1 MRR | prev7 MRR | Improvement |
|---------|-----------|-----------|-------------|
| DIY | 59.97% | 67.67% | +7.70 pp (+12.8%) |
| GeoLife | 57.83% | 64.55% | +6.72 pp (+11.6%) |

**Interpretation**:
- MRR of 67.67% means average rank is ~1.48 (since 1/1.48 ≈ 0.68)
- Most predictions are rank 1 or 2

---

## 4. Normalized Discounted Cumulative Gain (NDCG@k)

### 4.1 Definition

**NDCG** measures the quality of ranking with position-dependent discounting.

**Discounted Cumulative Gain (DCG)**:

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{2^{rel_i} - 1}{\log_2(i + 1)}$$

For binary relevance (correct/incorrect):

$$\text{DCG@}k = \sum_{i=1}^{k} \frac{rel_i}{\log_2(i + 1)}$$

**NDCG** normalizes by ideal DCG:

$$\text{NDCG@}k = \frac{\text{DCG@}k}{\text{IDCG@}k}$$

For single-item relevance: $\text{IDCG} = 1$ (best case: correct item at rank 1)

### 4.2 Simplified Formula for Our Case

Since we have binary relevance (one correct location):

$$\text{NDCG@}k = \begin{cases}
\frac{1}{\log_2(\text{rank} + 1)} & \text{if rank} \leq k \\
0 & \text{if rank} > k
\end{cases}$$

### 4.3 Implementation

```python
def ndcg_at_k(logits, targets, k=10):
    """
    Calculate NDCG@k.
    
    Args:
        logits: [N, num_locations] predicted scores
        targets: [N] true location IDs
        k: cutoff for ranking consideration
    
    Returns:
        float: NDCG@k as percentage
    """
    # Sort predictions
    sorted_indices = torch.argsort(logits, dim=-1, descending=True)
    
    # Find ranks
    targets_expanded = targets.unsqueeze(1)
    matches = (sorted_indices == targets_expanded)
    ranks = matches.nonzero()[:, 1] + 1  # 1-indexed
    
    # Calculate NDCG
    ndcg = 1.0 / torch.log2(ranks.float() + 1)
    
    # Zero out ranks beyond k
    ndcg[ranks > k] = 0
    
    return ndcg.mean() * 100
```

### 4.4 NDCG Discount Table

| Rank | Discount | NDCG Value |
|------|----------|------------|
| 1 | 1/log₂(2) | 1.000 |
| 2 | 1/log₂(3) | 0.631 |
| 3 | 1/log₂(4) | 0.500 |
| 4 | 1/log₂(5) | 0.431 |
| 5 | 1/log₂(6) | 0.387 |
| 10 | 1/log₂(11) | 0.289 |
| 100 | 1/log₂(101) | 0.150 |

### 4.5 NDCG vs MRR Comparison

| Property | NDCG@10 | MRR |
|----------|---------|-----|
| Discount function | Logarithmic | Inverse |
| Position cutoff | Yes (k=10) | No |
| Penalization | Gentler at high ranks | Harsher at high ranks |
| Best for | Ranking quality | Finding correct item |

### 4.6 Results from This Experiment

| Dataset | prev1 NDCG | prev7 NDCG | Improvement |
|---------|------------|------------|-------------|
| DIY | 63.47% | 71.88% | +8.41 pp (+13.2%) |
| GeoLife | 61.60% | 69.46% | +7.86 pp (+12.8%) |

---

## 5. F1 Score (Weighted)

### 5.1 Definition

**Weighted F1** is the weighted average of per-class F1 scores, weighted by class frequency.

**Per-class F1**:
$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

**Weighted F1**:
$$F1_{\text{weighted}} = \sum_{c=1}^{C} \frac{n_c}{N} \cdot F1_c$$

where $n_c$ is the number of samples with true class $c$.

### 5.2 Why Weighted F1?

**Problem with Standard Accuracy**:
- Location distribution is highly imbalanced
- A model predicting "home" always might have 30% accuracy
- Weighted F1 accounts for per-class performance

**Example**:
```
Location A: 1000 samples, 60% correct → Precision=60%, Recall=60%
Location B: 10 samples, 10% correct → Precision=10%, Recall=10%

Standard Acc: (600 + 1) / 1010 ≈ 59.5%
Weighted F1: (1000/1010)*0.6 + (10/1010)*0.1 ≈ 59.5%

Now if model ignores Location B entirely:
Standard Acc: 600/1010 ≈ 59.4%  (similar!)
Weighted F1: Lower (penalized for 0% recall on B)
```

### 5.3 Implementation

```python
from sklearn.metrics import f1_score

def weighted_f1(logits, targets):
    """
    Calculate weighted F1 score.
    
    Args:
        logits: [N, num_locations] predicted scores
        targets: [N] true location IDs
    
    Returns:
        float: Weighted F1 as percentage
    """
    # Get top-1 predictions
    predictions = torch.argmax(logits, dim=-1)  # [N]
    
    # Calculate weighted F1
    f1 = f1_score(
        targets.cpu().numpy(),
        predictions.cpu().numpy(),
        average='weighted'
    )
    
    return f1 * 100
```

### 5.4 Results from This Experiment

| Dataset | prev1 F1 | prev7 F1 | Improvement |
|---------|----------|----------|-------------|
| DIY | 46.73% | 51.91% | +5.18 pp (+11.1%) |
| GeoLife | 45.51% | 46.97% | +1.46 pp (+3.2%) |

**Observation**: F1 improves less than Acc@1 for GeoLife.

**Interpretation**: 
- DIY improvement is uniform across location classes
- GeoLife improvement is concentrated on high-frequency locations
- Rare locations remain difficult even with more history

---

## 6. Cross-Entropy Loss

### 6.1 Definition

**Cross-Entropy Loss** measures the average negative log-likelihood of correct predictions.

**Mathematical Formula**:

$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P(y_i | x_i)$$

where $P(y_i | x_i)$ is the predicted probability of the correct location.

### 6.2 Implementation

```python
def cross_entropy_loss(logits, targets):
    """
    Calculate cross-entropy loss.
    
    Args:
        logits: [N, num_locations] predicted log-probabilities
        targets: [N] true location IDs
    
    Returns:
        float: Average cross-entropy loss
    """
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, targets)
    return loss.item()
```

### 6.3 Understanding Loss Values

| Loss | Interpretation |
|------|----------------|
| 0.0 | Perfect (100% confidence correct) |
| 0.69 | ~50% confidence on correct class (2-class) |
| 2.3 | ~10% confidence on correct class |
| 3.0 | ~5% confidence on correct class |
| 4.6 | ~1% confidence on correct class |
| 6.9 | ~0.1% confidence on correct class |

**Relationship to Probability**:
$$\text{Loss} = -\log(p) \implies p = e^{-\text{Loss}}$$

| Loss | Average Probability |
|------|---------------------|
| 2.5 | e^(-2.5) ≈ 8.2% |
| 2.8 | e^(-2.8) ≈ 6.1% |
| 3.0 | e^(-3.0) ≈ 5.0% |
| 3.5 | e^(-3.5) ≈ 3.0% |

### 6.4 Why Loss is Important

**Beyond Accuracy**:
- A model with 50% accuracy could have:
  - High confidence when correct, low when wrong (good)
  - Low confidence always (bad)
- Loss distinguishes these cases

**Calibration**:
- Well-calibrated models have probabilities that match empirical frequencies
- Lower loss generally means better calibration

### 6.5 Results from This Experiment

| Dataset | prev1 Loss | prev7 Loss | Improvement |
|---------|------------|------------|-------------|
| DIY | 3.763 | 2.874 | -0.889 (-23.6%) |
| GeoLife | 3.492 | 2.630 | -0.862 (-24.7%) |

**Interpretation**:
- DIY prev1: ~2.3% average probability on correct location
- DIY prev7: ~5.6% average probability on correct location
- More history → more confident predictions

**Why GeoLife Has Lower Loss**:
- Despite lower accuracy, GeoLife model is more confident
- Suggests "cleaner" patterns or better calibration

---

## 7. Metric Relationships

### 7.1 Theoretical Relationships

```
Acc@1 ≤ Acc@5 ≤ Acc@10    (by definition)
Acc@1 ≤ MRR ≤ 100%        (MRR ≥ Acc@1 always)
NDCG@10 ≥ 0               (0 if all correct items beyond rank 10)
```

### 7.2 Empirical Relationships (This Experiment)

From our results:
```
DIY prev7:
Acc@1 (56.58%) < MRR (67.67%) < Acc@5 (82.18%) < Acc@10 (85.16%)
NDCG@10 (71.88%) between MRR and Acc@5
F1 (51.91%) < Acc@1 (due to class imbalance)

GeoLife prev7:
Acc@1 (51.40%) < MRR (64.55%) < Acc@5 (81.18%) < Acc@10 (85.04%)
NDCG@10 (69.46%) between MRR and Acc@5
F1 (46.97%) < Acc@1
```

### 7.3 When to Use Which Metric

| Scenario | Primary Metric | Secondary Metrics |
|----------|---------------|-------------------|
| Single prediction | Acc@1 | MRR, Loss |
| Short list recommendation | Acc@5 | NDCG@10, MRR |
| Research comparison | All metrics | - |
| Class imbalance concern | F1 | Acc@1 |
| Model calibration | Loss | Acc@1 |

---

## 8. Metric Summary Card

```
┌─────────────────────────────────────────────────────────────┐
│                METRIC QUICK REFERENCE                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  ACCURACY@k                                                  │
│  Formula: % of samples with correct in top-k                │
│  Range: 0-100%, higher is better                            │
│  Best for: k-item recommendation lists                      │
│                                                              │
│  MRR (Mean Reciprocal Rank)                                 │
│  Formula: average of 1/rank                                 │
│  Range: 0-100%, higher is better                            │
│  Best for: Ranked search results                            │
│                                                              │
│  NDCG@10                                                     │
│  Formula: 1/log₂(rank+1) averaged                           │
│  Range: 0-100%, higher is better                            │
│  Best for: Ranking quality assessment                       │
│                                                              │
│  F1 (Weighted)                                              │
│  Formula: Weighted average of per-class F1                  │
│  Range: 0-100%, higher is better                            │
│  Best for: Imbalanced classification                        │
│                                                              │
│  LOSS (Cross-Entropy)                                       │
│  Formula: -avg(log(P(correct)))                             │
│  Range: 0-∞, lower is better                                │
│  Best for: Probability calibration                          │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,500 |
| **Status** | Final |

---

**Navigation**: [← Datasets](./07_datasets.md) | [Index](./INDEX.md) | [Next: Results & Analysis →](./09_results_and_analysis.md)
