# Evaluation Metrics Documentation

This document provides a comprehensive explanation of all evaluation metrics used to assess the Pointer Network V45 model, including their mathematical formulations, intuitions, and interpretations.

---

## 1. Overview of Metrics

### 1.1 Metric Categories

| Category | Metrics | Purpose |
|----------|---------|---------|
| **Accuracy** | Acc@1, Acc@5, Acc@10 | Measure if correct answer is in top-k |
| **Ranking** | MRR, NDCG | Measure quality of ranking |
| **Classification** | F1 Score | Measure precision-recall balance |
| **Training** | Cross-Entropy Loss | Optimization objective |

### 1.2 Why Multiple Metrics?

Different metrics capture different aspects:
- **Acc@1**: Exact prediction accuracy
- **Acc@5/10**: Relaxed accuracy (model suggests multiple options)
- **MRR**: How high is the correct answer ranked?
- **NDCG**: Quality of the entire ranking
- **F1**: Balance of precision and recall per class

---

## 2. Accuracy Metrics (Acc@k)

### 2.1 Definition

**Acc@k** measures the percentage of samples where the correct answer appears in the top-k predictions.

### 2.2 Mathematical Formula

```
Acc@k = (1/N) × Σᵢ 𝟙[yᵢ ∈ top_k(ŷᵢ)]

Where:
- N = total number of samples
- yᵢ = true label for sample i
- ŷᵢ = predicted probabilities for sample i
- top_k(ŷᵢ) = k indices with highest probabilities
- 𝟙[·] = indicator function (1 if true, 0 if false)
```

### 2.3 Implementation

```python
def calculate_topk_accuracy(logits, targets, k):
    """
    Calculate top-k accuracy.
    
    Args:
        logits: [batch_size, num_classes] - model predictions
        targets: [batch_size] - true labels
    
    Returns:
        Number of correct predictions
    """
    # Get top-k predictions
    prediction = torch.topk(logits, k=k, dim=-1).indices  # [batch, k]
    
    # Check if target is in top-k
    correct = torch.eq(targets[:, None], prediction).any(dim=1)  # [batch]
    
    return correct.sum().item()
```

### 2.4 Intuition

- **Acc@1**: "Did we guess exactly right?"
  - Most strict metric
  - Directly usable for single predictions
  
- **Acc@5**: "Is the correct answer in our top 5 suggestions?"
  - Useful when showing multiple options to users
  - More forgiving than Acc@1
  
- **Acc@10**: "Is the correct answer in our top 10?"
  - Even more relaxed
  - Upper bound on useful predictions

### 2.5 Interpretation Guide

| Acc@1 | Interpretation |
|-------|----------------|
| < 30% | Poor - random guessing level |
| 30-50% | Moderate - learning patterns |
| 50-60% | Good - strong predictions |
| > 60% | Excellent - very accurate |

### 2.6 Example

```python
# Example predictions
logits = torch.tensor([
    [0.5, 0.3, 0.1, 0.1],  # Sample 0: predict class 0
    [0.1, 0.5, 0.3, 0.1],  # Sample 1: predict class 1
    [0.2, 0.2, 0.4, 0.2],  # Sample 2: predict class 2
])

targets = torch.tensor([0, 0, 2])  # True labels

# Acc@1:
# Sample 0: top-1 = [0], target 0 ✓
# Sample 1: top-1 = [1], target 0 ✗
# Sample 2: top-1 = [2], target 2 ✓
# Acc@1 = 2/3 = 66.67%

# Acc@2:
# Sample 0: top-2 = [0, 1], target 0 ✓
# Sample 1: top-2 = [1, 2], target 0 ✗
# Sample 2: top-2 = [2, 0], target 2 ✓
# Acc@2 = 2/3 = 66.67%
```

---

## 3. Mean Reciprocal Rank (MRR)

### 3.1 Definition

**MRR** is the average of reciprocal ranks across all samples. The reciprocal rank is 1/r where r is the rank of the correct answer.

### 3.2 Mathematical Formula

```
MRR = (1/N) × Σᵢ (1/rankᵢ)

Where:
- N = total number of samples
- rankᵢ = position of correct answer in sorted predictions (1-indexed)
```

### 3.3 Implementation

```python
def get_mrr(prediction, targets):
    """
    Calculate MRR score.
    
    Args:
        prediction: [batch, num_classes] - prediction scores
        targets: [batch] - true labels
    
    Returns:
        Sum of reciprocal ranks
    """
    # Sort predictions in descending order
    index = torch.argsort(prediction, dim=-1, descending=True)
    
    # Find where target appears in sorted predictions
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    
    # Get ranks (1-indexed)
    ranks = (hits[:, -1] + 1).float()
    
    # Calculate reciprocal ranks
    rranks = torch.reciprocal(ranks)
    
    return torch.sum(rranks).cpu().numpy()
```

### 3.4 Intuition

MRR rewards predictions that rank the correct answer high:
- Correct answer at rank 1: RR = 1.0
- Correct answer at rank 2: RR = 0.5
- Correct answer at rank 3: RR = 0.33
- Correct answer at rank 10: RR = 0.1

The reciprocal means:
- Top-ranked correct predictions contribute much more
- Lower ranks contribute diminishing amounts

### 3.5 Interpretation Guide

| MRR | Interpretation |
|-----|----------------|
| < 40% | Poor - correct answer ranks low |
| 40-60% | Moderate - average rank ~2-3 |
| 60-70% | Good - often in top 2 |
| > 70% | Excellent - usually top prediction |

### 3.6 Example

```python
# Three samples with different rankings
sample_1_rank = 1  # Correct answer is top prediction
sample_2_rank = 3  # Correct answer is 3rd
sample_3_rank = 5  # Correct answer is 5th

# Reciprocal ranks
rr_1 = 1/1 = 1.000
rr_2 = 1/3 = 0.333
rr_3 = 1/5 = 0.200

# MRR
mrr = (1.000 + 0.333 + 0.200) / 3 = 0.511 = 51.1%
```

---

## 4. Normalized Discounted Cumulative Gain (NDCG)

### 4.1 Definition

**NDCG@k** measures the quality of ranking by considering both relevance and position, normalized by the ideal ranking.

### 4.2 Mathematical Formula

For our binary relevance case (relevant = 1, not relevant = 0):

```
DCG@k = Σᵢ₌₁ᵏ (relᵢ / log₂(i + 1))

IDCG@k = 1 / log₂(2) = 1  (single relevant item at position 1)

NDCG@k = DCG@k / IDCG@k = 1 / log₂(rank + 1)  (if rank ≤ k, else 0)
```

### 4.3 Implementation

```python
def get_ndcg(prediction, targets, k=10):
    """
    Calculate NDCG@k score.
    
    Args:
        prediction: [batch, num_classes] - prediction scores
        targets: [batch] - true labels
        k: Consider only top-k predictions
    
    Returns:
        Sum of NDCG scores
    """
    # Sort predictions in descending order
    index = torch.argsort(prediction, dim=-1, descending=True)
    
    # Find where target appears
    hits = (targets.unsqueeze(-1).expand_as(index) == index).nonzero()
    
    # Get ranks (1-indexed)
    ranks = (hits[:, -1] + 1).float().cpu().numpy()
    
    # Calculate NDCG with logarithmic discount
    ndcg = 1 / np.log2(ranks + 1)
    
    # Zero out ranks beyond k
    ndcg[ranks > k] = 0
    
    return np.sum(ndcg)
```

### 4.4 Intuition

NDCG applies a **logarithmic discount** based on position:
- Position 1: Discount = 1/log₂(2) = 1.0
- Position 2: Discount = 1/log₂(3) = 0.63
- Position 3: Discount = 1/log₂(4) = 0.5
- Position 10: Discount = 1/log₂(11) = 0.29

Key properties:
- Higher positions contribute more (like MRR)
- Discount is smoother than MRR's 1/r
- Standard metric in information retrieval

### 4.5 Interpretation Guide

| NDCG@10 | Interpretation |
|---------|----------------|
| < 45% | Poor ranking quality |
| 45-60% | Moderate ranking |
| 60-70% | Good ranking |
| > 70% | Excellent ranking |

### 4.6 NDCG vs MRR

| Aspect | MRR | NDCG |
|--------|-----|------|
| Discount | 1/rank | 1/log₂(rank+1) |
| Rank 1 | 1.0 | 1.0 |
| Rank 2 | 0.5 | 0.63 |
| Rank 5 | 0.2 | 0.39 |
| Use case | Single correct answer | Ranking quality |

---

## 5. F1 Score

### 5.1 Definition

**Weighted F1 Score** is the harmonic mean of precision and recall, weighted by class frequency.

### 5.2 Mathematical Formula

For each class c:
```
Precision_c = TP_c / (TP_c + FP_c)
Recall_c = TP_c / (TP_c + FN_c)
F1_c = 2 × (Precision_c × Recall_c) / (Precision_c + Recall_c)

Weighted F1 = Σ_c (weight_c × F1_c)
Where weight_c = N_c / N  (fraction of samples in class c)
```

### 5.3 Implementation

```python
from sklearn.metrics import f1_score

# Get top-1 predictions
predictions = logits.argmax(dim=-1).cpu().numpy()
true_labels = targets.cpu().numpy()

# Calculate weighted F1
f1 = f1_score(true_labels, predictions, average='weighted', zero_division=0)
```

### 5.4 Intuition

F1 Score balances:
- **Precision**: Of predicted class c, how many were correct?
- **Recall**: Of true class c, how many were predicted?

**Weighted** averaging accounts for class imbalance - frequent locations contribute more.

### 5.5 Interpretation Guide

| F1 | Interpretation |
|----|----------------|
| < 30% | Poor - many misclassifications |
| 30-50% | Moderate |
| 50-60% | Good |
| > 60% | Excellent |

### 5.6 Note on Location Prediction

F1 is often lower than Acc@1 because:
- Many locations are rare (low recall on them)
- Weighted average still penalizes mistakes on rare locations
- F1 is more sensitive to class imbalance

---

## 6. Cross-Entropy Loss

### 6.1 Definition

**Cross-Entropy Loss** measures the difference between predicted probabilities and true labels.

### 6.2 Mathematical Formula

```
L = -Σᵢ Σⱼ yᵢⱼ × log(pᵢⱼ)

For one-hot labels:
L = -(1/N) × Σᵢ log(pᵢ,yᵢ)

Where:
- N = batch size
- yᵢ = true class for sample i
- pᵢ,yᵢ = predicted probability for true class
```

### 6.3 Implementation

```python
criterion = nn.CrossEntropyLoss(
    ignore_index=0,           # Ignore padding
    label_smoothing=0.03,     # Regularization
)

loss = criterion(logits, targets)
```

### 6.4 With Label Smoothing

```
Smoothed target: y_smooth = (1 - ε) × y_onehot + ε / C

Where:
- ε = smoothing factor (e.g., 0.03)
- C = number of classes
```

### 6.5 Interpretation

| Loss | Interpretation |
|------|----------------|
| > 5.0 | Very high - model not learning |
| 3.0-5.0 | High - early training |
| 2.5-3.0 | Moderate - learning |
| < 2.5 | Good convergence |

---

## 7. Metrics Aggregation

### 7.1 Batch-Level Computation

```python
def calculate_correct_total_prediction(logits, true_y):
    """
    Calculate all metrics for a batch.
    
    Returns:
        result_array: [correct@1, correct@3, correct@5, correct@10, rr_sum, ndcg_sum, total]
        true_labels: CPU tensor
        top1_predictions: CPU tensor
    """
    result_ls = []
    
    # Top-k accuracy
    for k in [1, 3, 5, 10]:
        prediction = torch.topk(logits, k=k, dim=-1).indices
        top_k = torch.eq(true_y[:, None], prediction).any(dim=1).sum().cpu().numpy()
        result_ls.append(top_k)
    
    # MRR
    result_ls.append(get_mrr(logits, true_y))
    
    # NDCG
    result_ls.append(get_ndcg(logits, true_y))
    
    # Total samples
    result_ls.append(true_y.shape[0])
    
    return np.array(result_ls), true_y.cpu(), top1_predictions
```

### 7.2 Dataset-Level Aggregation

```python
# Accumulate across batches
total_results = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

for batch in dataloader:
    batch_results, _, _ = calculate_correct_total_prediction(logits, targets)
    total_results += batch_results

# Convert to percentages
metrics = {
    "acc@1": total_results[0] / total_results[6] * 100,
    "acc@5": total_results[2] / total_results[6] * 100,
    "acc@10": total_results[3] / total_results[6] * 100,
    "mrr": total_results[4] / total_results[6] * 100,
    "ndcg": total_results[5] / total_results[6] * 100,
}
```

---

## 8. Metrics Summary Table

### 8.1 All Metrics

| Metric | Range | Higher Better? | Primary Use |
|--------|-------|----------------|-------------|
| Acc@1 | 0-100% | Yes | Exact accuracy |
| Acc@5 | 0-100% | Yes | Top-5 inclusion |
| Acc@10 | 0-100% | Yes | Top-10 inclusion |
| MRR | 0-100% | Yes | Ranking quality |
| NDCG@10 | 0-100% | Yes | Ranking quality |
| F1 | 0-100% | Yes | Classification balance |
| Loss | 0-∞ | No | Training objective |

### 8.2 Metric Relationships

```
Generally: Acc@10 > Acc@5 > Acc@1
           NDCG ≥ MRR (due to discount function)
           Acc@1 ≈ F1 (for balanced classes)
```

### 8.3 Typical Results

| Metric | GeoLife | DIY |
|--------|---------|-----|
| Acc@1 | 53.97% | 56.89% |
| Acc@5 | 81.10% | 82.24% |
| Acc@10 | 84.38% | 86.14% |
| MRR | 65.82% | 68.00% |
| NDCG | 70.23% | 72.31% |
| F1 | ~50% | ~50% |

---

## 9. Using Metrics for Model Comparison

### 9.1 Primary Metrics

For location prediction, prioritize:
1. **Acc@1**: Most important - exact predictions
2. **MRR**: Ranking quality matters
3. **Acc@5**: Useful for suggestion systems

### 9.2 Statistical Significance

When comparing models:
- Run multiple seeds (e.g., 3-5 runs)
- Report mean ± standard deviation
- Use statistical tests for significance

### 9.3 Trade-offs

Sometimes metrics can be at odds:
- Model A: 55% Acc@1, 60% MRR
- Model B: 52% Acc@1, 65% MRR

Model B ranks correct answers higher on average, but Model A gets more exact hits. Choose based on application:
- Navigation: Acc@1 matters most
- Suggestions: MRR/NDCG matter more

---

## 10. Code Reference

### 10.1 Complete Metrics Module

Location: `src/evaluation/metrics.py`

Key functions:
- `get_mrr()`: Calculate MRR
- `get_ndcg()`: Calculate NDCG@10
- `calculate_correct_total_prediction()`: Batch metrics
- `get_performance_dict()`: Convert to percentages
- `calculate_metrics()`: High-level API
- `evaluate_model()`: Full dataset evaluation

### 10.2 Usage Example

```python
from src.evaluation.metrics import calculate_metrics, evaluate_model

# Option 1: Single batch
metrics = calculate_metrics(logits, targets)
print(f"Acc@1: {metrics['acc@1']:.2f}%")
print(f"MRR: {metrics['mrr']:.2f}%")

# Option 2: Full dataset
metrics = evaluate_model(model, test_loader, device, verbose=True)
```

---

*Next: [07_CONFIGURATION_GUIDE.md](07_CONFIGURATION_GUIDE.md) - Configuration Options*
