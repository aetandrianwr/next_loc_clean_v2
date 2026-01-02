# 04. Experimental Methodology

## Scientific Design and Approach

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Methodology |
| **Audience** | Researchers, Reviewers |
| **Reading Time** | 12-15 minutes |
| **Prerequisites** | Understanding of experimental design |

---

## 1. Research Design Overview

### 1.1 Experiment Type

This is a **controlled ablation study** that systematically varies one factor (temporal window) while holding all others constant.

**Design Classification**:
- **Type**: Single-factor within-subjects design
- **Factor**: Number of previous days (7 levels: 1-7)
- **Replication**: Two datasets (DIY, GeoLife)
- **Measurement**: Multiple metrics (Acc@1, Acc@5, Acc@10, MRR, NDCG, F1, Loss)

### 1.2 Variables

#### Independent Variable (IV)
| Variable | Description | Levels |
|----------|-------------|--------|
| **previous_days** | Number of days of historical location data | 1, 2, 3, 4, 5, 6, 7 |

#### Dependent Variables (DVs)
| Variable | Type | Range | Interpretation |
|----------|------|-------|----------------|
| **Accuracy@1** | Percentage | 0-100% | Exact match accuracy |
| **Accuracy@5** | Percentage | 0-100% | Correct in top-5 |
| **Accuracy@10** | Percentage | 0-100% | Correct in top-10 |
| **MRR** | Percentage | 0-100% | Mean reciprocal rank |
| **NDCG@10** | Percentage | 0-100% | Ranking quality |
| **F1** | Percentage | 0-100% | Class-weighted F1 |
| **Loss** | Real | 0-∞ | Cross-entropy loss (lower better) |

#### Control Variables (Held Constant)
| Variable | Value | Why Controlled |
|----------|-------|----------------|
| Model architecture | PointerNetworkV45 | Isolate data effect |
| Model weights | Pre-trained on prev7 | Same model for all |
| Hyperparameters | Fixed (from tuning) | No confounding |
| Random seed | 42 | Reproducibility |
| Test data source | Same test split | Fair comparison |
| Batch size | 64 | Consistent evaluation |

### 1.3 Experimental Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL PIPELINE                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │   Original   │    │   Model      │    │   Test       │      │
│  │   Test Data  │    │  (Pre-trained│    │   Metrics    │      │
│  │   (prev7)    │    │   on prev7)  │    │              │      │
│  └──────┬───────┘    └──────────────┘    └──────────────┘      │
│         │                   │                    │               │
│         ▼                   │                    │               │
│  ┌──────────────┐          │                    │               │
│  │   Filter by  │──────────┤                    │               │
│  │   prev_days  │          │                    │               │
│  └──────┬───────┘          │                    │               │
│         │                   │                    │               │
│    ┌────┴────┐             │                    │               │
│    ▼         ▼             ▼                    ▼               │
│ ┌─────┐ ┌─────┐        ┌────────┐        ┌──────────┐         │
│ │prev1│ │prev7│ ────▶  │Evaluate│ ────▶  │ Metrics  │         │
│ └─────┘ └─────┘        └────────┘        └──────────┘         │
│   ...                                                           │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. Pre-Trained Model Strategy

### 2.1 Rationale for Using a Single Pre-Trained Model

We deliberately use a **single model trained on 7-day data** (prev7) and evaluate it with varying input lengths, rather than training separate models for each configuration.

**Why This Approach**:

1. **Isolates Data Effect**: 
   - Training introduces variance (different local minima, initialization effects)
   - Same model ensures differences are due to input data, not training

2. **Reflects Practical Scenario**:
   - Production systems train once and deploy
   - Test scenario: "Given a trained model, how does prediction quality degrade with less history?"

3. **Fair Capacity Comparison**:
   - Different training data might need different architectures
   - Same architecture ensures capacity is not a confound

4. **Computational Efficiency**:
   - 1 training run instead of 7 per dataset
   - Faster iteration during experiment development

### 2.2 Alternative Approaches (Not Used)

| Approach | Description | Why Not Used |
|----------|-------------|--------------|
| **Per-config training** | Train 7 models, one for each prev_days | Introduces training variance |
| **Multi-task training** | Train one model with multiple outputs | More complex, harder to interpret |
| **Online adaptation** | Fine-tune for each config | Computationally expensive |

### 2.3 Potential Concerns with Our Approach

**Concern 1**: Model trained on prev7 might not be optimal for shorter sequences.

**Response**: This is actually informative—it shows the robustness of the model to reduced input.

**Concern 2**: prev7 model might rely on patterns not available in shorter windows.

**Response**: If so, we'd see large performance drops, which is a valid finding.

---

## 3. Data Filtering Methodology

### 3.1 Filter Algorithm

For each sample and each previous_days value, we apply:

```python
def filter_sequence(sample, previous_days):
    """
    Filter a sample to include only visits within the temporal window.
    
    Args:
        sample: dict with keys 'X' (locations), 'diff' (days ago), etc.
        previous_days: int, number of days to include (1-7)
    
    Returns:
        Filtered sample or None if too few visits remain
    """
    diff = sample['diff']  # Array: days ago for each visit
    
    # Create mask for visits within window
    mask = diff <= previous_days
    
    # Check minimum sequence length
    if mask.sum() < MIN_SEQUENCE_LENGTH:
        return None  # Exclude this sample
    
    # Apply mask to all features
    filtered_sample = {
        'X': sample['X'][mask],           # Location IDs
        'user_X': sample['user_X'][mask], # User IDs (repeated)
        'weekday_X': sample['weekday_X'][mask],
        'start_min_X': sample['start_min_X'][mask],
        'dur_X': sample['dur_X'][mask],
        'diff': sample['diff'][mask],
        'Y': sample['Y'],                  # Target unchanged
    }
    return filtered_sample
```

### 3.2 The `diff` Field Explained

Each location visit has an associated `diff` value indicating how many days ago the visit occurred:

| diff Value | Interpretation |
|------------|----------------|
| 0 | Today (same day as prediction) |
| 1 | Yesterday |
| 2 | 2 days ago |
| ... | ... |
| 7 | 7 days ago |

**Example**:
If predicting at Wednesday 3 PM:
- Visit at Wednesday 10 AM → diff = 0
- Visit at Tuesday 6 PM → diff = 1
- Visit at previous Wednesday 9 AM → diff = 7

### 3.3 Filtering Semantics

**prev_days = d** includes all visits where:
$$\text{diff} \leq d$$

| prev_days | Includes diff values | Time window |
|-----------|---------------------|-------------|
| 1 | 0, 1 | Last ~24-48 hours |
| 2 | 0, 1, 2 | Last ~48-72 hours |
| 3 | 0, 1, 2, 3 | Last ~72-96 hours |
| ... | ... | ... |
| 7 | 0, 1, 2, 3, 4, 5, 6, 7 | Last ~168-192 hours |

### 3.4 Sample Retention

Some samples may be excluded after filtering if they have too few visits:

**DIY Dataset**:
| prev_days | Valid Samples | % of prev7 |
|-----------|---------------|------------|
| 1 | 11,532 | 93.2% |
| 2 | 12,068 | 97.6% |
| 3 | 12,235 | 98.9% |
| 4 | 12,311 | 99.5% |
| 5 | 12,351 | 99.9% |
| 6 | 12,365 | 99.97% |
| 7 | 12,368 | 100% |

**GeoLife Dataset**:
| prev_days | Valid Samples | % of prev7 |
|-----------|---------------|------------|
| 1 | 3,263 | 93.2% |
| 2 | 3,398 | 97.0% |
| 3 | 3,458 | 98.7% |
| 4 | 3,487 | 99.6% |
| 5 | 3,494 | 99.8% |
| 6 | 3,499 | 99.9% |
| 7 | 3,502 | 100% |

**Observation**: ~7% of samples have insufficient data with only 1 day of history.

---

## 4. Evaluation Protocol

### 4.1 Evaluation Pipeline

For each dataset and each prev_days configuration:

```python
def evaluate_configuration(dataset, prev_days, model):
    """Evaluate model on filtered dataset."""
    
    # Step 1: Create filtered dataset
    filtered_data = []
    for sample in dataset.test_data:
        filtered = filter_sequence(sample, prev_days)
        if filtered is not None:
            filtered_data.append(filtered)
    
    # Step 2: Create DataLoader
    dataloader = DataLoader(filtered_data, batch_size=64, shuffle=False)
    
    # Step 3: Run inference
    all_logits = []
    all_targets = []
    all_losses = []
    
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch)
            loss = compute_loss(logits, batch['Y'])
            
            all_logits.append(logits)
            all_targets.append(batch['Y'])
            all_losses.append(loss)
    
    # Step 4: Compute metrics
    metrics = calculate_metrics(all_logits, all_targets)
    metrics['loss'] = average(all_losses)
    
    return metrics
```

### 4.2 Batch Processing Details

- **Batch size**: 64 (same as training)
- **Padding**: Variable-length sequences padded to batch max
- **Masking**: Attention masks for padded positions
- **Order**: Sequential (no shuffling for reproducibility)

### 4.3 No Evaluation Variance

Unlike training, evaluation is deterministic:
- No dropout (model.eval())
- No random sampling
- No gradient updates
- Same batch order each run

This means **results are exactly reproducible** given the same checkpoint.

---

## 5. Statistical Considerations

### 5.1 Sample Size Adequacy

| Dataset | Samples (prev7) | Confidence |
|---------|-----------------|------------|
| DIY | 12,368 | Very High |
| GeoLife | 3,502 | High |

**Standard Error for Accuracy**:
$$SE = \sqrt{\frac{p(1-p)}{n}}$$

For DIY with p=0.56, n=12,368:
$$SE = \sqrt{\frac{0.56 \times 0.44}{12368}} \approx 0.0045 = 0.45\%$$

This means 95% CI is approximately ±0.9 percentage points.

### 5.2 Effect Size Calculations

The improvement from prev1 to prev7 can be characterized by effect size:

**Cohen's h** for proportion differences:
$$h = 2 \cdot \arcsin(\sqrt{p_1}) - 2 \cdot \arcsin(\sqrt{p_2})$$

For DIY (Acc@1: 50% → 56.58%):
$$h = 2 \cdot \arcsin(\sqrt{0.5658}) - 2 \cdot \arcsin(\sqrt{0.50}) = 0.13$$

This is a "small to medium" effect size, but practically significant.

### 5.3 No Statistical Tests Performed

We do not perform significance tests because:
1. **Not sampling from a population**: We evaluate on the full test set
2. **No randomness in evaluation**: Results are deterministic
3. **Practical significance over statistical**: A 6.58 pp improvement is practically significant regardless of p-value

However, for reference:
- With n=12,368 and Δ=6.58pp, any standard test would show p << 0.001

---

## 6. Addressing Potential Confounds

### 6.1 Sample Set Variation

**Concern**: Different prev_days have different sample sets (due to filtering).

**Mitigation**: 
- Analyze sample retention rates (shown above)
- 93%+ retention at all levels
- Similar sample composition across levels

### 6.2 Sequence Length Confound

**Concern**: Longer prev_days → longer sequences → different model behavior.

**Observation**: This is part of what we're measuring. Longer history provides more tokens to attend to.

**Analysis**: We report sequence length statistics alongside performance.

### 6.3 Time-of-Day Distribution

**Concern**: Different filtering might change the distribution of prediction times.

**Verification**: Target distribution (Y values) is identical across prev_days since we only filter input, not output.

### 6.4 Model Capacity Matching

**Concern**: Model capacity might be better suited for longer sequences.

**Acknowledgment**: This is a limitation. The model was designed for prev7 and might underperform on shorter sequences.

**Future Work**: Test with models trained on each prev_days value.

---

## 7. Reproducibility Measures

### 7.1 Fixed Random Seeds

```python
import random
import numpy as np
import torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 7.2 Version Control

| Component | Version/Reference |
|-----------|------------------|
| PyTorch | 2.0+ |
| Python | 3.9+ |
| CUDA | 11.7+ |
| Model Checkpoint | `best.pt` from specified experiment |
| Config | YAML files in `sci_hyperparam_tuning/configs/` |

### 7.3 Data Integrity

| Item | Verification |
|------|-------------|
| Test data | Loaded from pickle files with fixed paths |
| Checkpoints | Loaded with `torch.load(strict=True)` |
| Configs | YAML files version-controlled |

---

## 8. Limitations and Validity Threats

### 8.1 Internal Validity

| Threat | Mitigation | Residual Risk |
|--------|------------|---------------|
| Training variance | Single pre-trained model | None |
| Evaluation variance | Deterministic evaluation | None |
| Data leakage | Separate test split | Low |
| Selection bias | Full test set evaluation | Low |

### 8.2 External Validity

| Threat | Assessment |
|--------|------------|
| **Different architectures** | Results may not generalize to RNN/Markov models |
| **Different datasets** | Two datasets provide some generalization |
| **Different time periods** | Both datasets from similar era |
| **Production conditions** | Lab evaluation vs real-time serving may differ |

### 8.3 Construct Validity

| Threat | Assessment |
|--------|------------|
| **Metric choice** | Multiple metrics provide comprehensive view |
| **"Days" as a measure** | Standardized and interpretable |
| **Clustering parameters** | Different ε values between datasets |

---

## 9. Chapter Summary

### Methodological Strengths

1. **Controlled design**: Single-factor ablation with all else held constant
2. **Multiple DVs**: Comprehensive metric suite
3. **Replication**: Two independent datasets
4. **Reproducibility**: Fixed seeds, version control, deterministic evaluation

### Methodological Limitations

1. **Single model architecture**: Results specific to PointerNetworkV45
2. **Fixed hyperparameters**: Model optimized for prev7 only
3. **Limited window range**: Only 1-7 days tested

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Pre-trained model | Isolates input data effect |
| Filter-based evaluation | Simulates reduced history availability |
| Multiple metrics | Comprehensive performance view |
| Two datasets | Generalization check |

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,100 |
| **Status** | Final |

---

**Navigation**: [← Theoretical Foundation](./03_theoretical_foundation.md) | [Index](./INDEX.md) | [Next: Technical Implementation →](./05_technical_implementation.md)
