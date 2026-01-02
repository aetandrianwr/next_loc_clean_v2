# 4. Methodology

## Scientific Methodology for Ablation Study

---

## 4.1 Research Design Philosophy

### The Scientific Method Applied

This ablation study follows the rigorous scientific method:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     SCIENTIFIC METHOD                                │
│                                                                      │
│   1. OBSERVATION                                                     │
│      PointerNetworkV45 achieves good performance                     │
│                           ↓                                          │
│   2. QUESTION                                                        │
│      Which components contribute to this performance?                │
│                           ↓                                          │
│   3. HYPOTHESIS                                                      │
│      Each component has specific contribution                        │
│                           ↓                                          │
│   4. EXPERIMENT                                                      │
│      Systematically remove each component                            │
│                           ↓                                          │
│   5. ANALYSIS                                                        │
│      Compare performance with baseline                               │
│                           ↓                                          │
│   6. CONCLUSION                                                      │
│      Quantify each component's importance                            │
└─────────────────────────────────────────────────────────────────────┘
```

### Why Ablation?

Ablation studies are the gold standard for:

1. **Validating architectural choices**: Proving each component helps
2. **Understanding model behavior**: Seeing how components interact
3. **Identifying redundancy**: Finding unnecessary complexity
4. **Guiding optimization**: Knowing where to focus efforts

---

## 4.2 Experimental Design

### 4.2.1 Controlled Experiment Framework

We use a **controlled experiment** design where:

- **Control Group**: Full model (baseline)
- **Treatment Groups**: Models with one component removed
- **Independent Variable**: Component presence/absence
- **Dependent Variables**: Performance metrics

```
┌────────────────────────────────────────────────────────────────┐
│                CONTROLLED EXPERIMENT DESIGN                     │
│                                                                 │
│   Control (Baseline):                                           │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  Full Model: All components enabled                      │  │
│   │  [Pointer ✓] [Gen ✓] [Temporal ✓] [User ✓] [Gate ✓] ... │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Treatment 1: No Pointer                                       │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  [Pointer ✗] [Gen ✓] [Temporal ✓] [User ✓] [Gate ✓] ... │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   Treatment 2: No Generation                                    │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │  [Pointer ✓] [Gen ✗] [Temporal ✓] [User ✓] [Gate ✓] ... │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│   ... (6 more treatments)                                       │
└────────────────────────────────────────────────────────────────┘
```

### 4.2.2 Single Variable Change

**Critical Principle**: Only one component is changed at a time.

This isolates the effect of each component:

```
❌ WRONG: Removing pointer AND temporal together
   - Cannot determine which caused the performance change

✅ CORRECT: Removing only pointer
   - Performance change is attributed to pointer mechanism
```

### 4.2.3 Fair Comparison

All experiments use **identical**:
- Training data
- Validation data
- Test data
- Optimizer settings
- Learning rate schedule
- Early stopping criteria
- Random seeds
- Hardware

---

## 4.3 Reproducibility Framework

### 4.3.1 Random Seed Control

```python
def set_seed(seed: int = 42):
    """Set random seed for complete reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

**Why seed=42?**
- It's a convention in ML research
- Makes results reproducible by others
- Ensures fair comparison between ablations

### 4.3.2 Deterministic Operations

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

These settings ensure:
- Same input always produces same output
- No randomness in CUDA operations
- Slightly slower but reproducible

### 4.3.3 Documentation Requirements

For full reproducibility, we document:

| Category | What We Document |
|----------|------------------|
| **Environment** | Python version, PyTorch version, CUDA version |
| **Hardware** | GPU model, memory, compute capability |
| **Data** | Preprocessing steps, train/val/test splits |
| **Model** | Architecture, hyperparameters, initialization |
| **Training** | Optimizer, LR schedule, batch size, epochs |
| **Evaluation** | Metrics, evaluation protocol |

---

## 4.4 Evaluation Protocol

### 4.4.1 Train-Validation-Test Split

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATASET SPLIT                             │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                    Training Set                           │  │
│   │   Used for: Learning model parameters                     │  │
│   │   GeoLife: 7,672 samples | DIY: 193,510 samples          │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                   Validation Set                          │  │
│   │   Used for: Early stopping, hyperparameter selection      │  │
│   │   GeoLife: 3,485 samples | DIY: 13,147 samples           │  │
│   └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                      Test Set                             │  │
│   │   Used for: Final evaluation (NEVER used for training)    │  │
│   │   GeoLife: 3,686 samples | DIY: 16,348 samples           │  │
│   └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4.2 Evaluation Metrics

We use comprehensive metrics that capture different aspects of prediction quality:

#### Accuracy Metrics (Acc@k)

```
Acc@k = (# predictions where correct answer is in top-k) / (total predictions) × 100%
```

| Metric | Description | When It's Important |
|--------|-------------|---------------------|
| **Acc@1** | Top-1 accuracy | Exact prediction needed |
| **Acc@5** | Top-5 accuracy | Short list of suggestions |
| **Acc@10** | Top-10 accuracy | Longer recommendation list |

#### Ranking Metrics

**Mean Reciprocal Rank (MRR)**:
```
MRR = (1/N) × Σ (1/rank_i)

Example:
- Correct at rank 1: 1/1 = 1.0
- Correct at rank 2: 1/2 = 0.5
- Correct at rank 5: 1/5 = 0.2
```

**Normalized Discounted Cumulative Gain (NDCG)**:
```
NDCG = (1/N) × Σ (1/log₂(rank_i + 1))

- Rewards correct predictions at higher ranks
- Penalizes correct predictions at lower ranks more gradually than MRR
```

#### Classification Metrics

**F1 Score (Weighted)**:
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)

- Weighted by class frequency
- Accounts for class imbalance
```

#### Loss Metric

**Cross-Entropy Loss**:
```
Loss = -Σ y_true × log(y_pred)

- Lower is better
- Used for early stopping
```

### 4.4.3 Early Stopping Protocol

```python
# Early stopping logic
patience = 5  # Stop if no improvement for 5 epochs
min_epochs = 8  # Train at least 8 epochs

for epoch in range(max_epochs):
    train_loss = train_epoch()
    val_metrics = evaluate(val_loader)
    
    if val_metrics['loss'] < best_val_loss:
        best_val_loss = val_metrics['loss']
        patience_counter = 0
        save_checkpoint("best.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience and epoch >= min_epochs:
            break  # Early stop
```

**Why patience=5?**
- Balances training time with finding optimal point
- Allows temporary dips in performance
- Prevents premature stopping

---

## 4.5 Statistical Considerations

### 4.5.1 Baseline Validation

Before running ablations, we validate that the baseline matches expected performance:

| Dataset | Expected Acc@1 | Achieved Acc@1 | Status |
|---------|---------------|----------------|--------|
| GeoLife | 51.39% | 51.43% | ✅ +0.04% |
| DIY | 56.58% | 56.57% | ✅ -0.01% |

**Tolerance**: ±0.5% considered acceptable for validation.

### 4.5.2 Effect Size Calculation

We calculate both absolute and relative effects:

```
Absolute Effect = Baseline_Acc@1 - Ablation_Acc@1

Relative Effect = (Baseline_Acc@1 - Ablation_Acc@1) / Baseline_Acc@1 × 100%
```

**Example**:
```
Baseline: 51.43%
No Pointer: 27.41%

Absolute Effect: 51.43 - 27.41 = 24.02 percentage points
Relative Effect: 24.02 / 51.43 × 100% = 46.7%
```

### 4.5.3 Significance Thresholds

We categorize component importance:

| Relative Impact | Category | Interpretation |
|-----------------|----------|----------------|
| > 10% | **Critical** | Essential component |
| 2-10% | **Important** | Significant contribution |
| 0.5-2% | **Minor** | Small but measurable |
| < 0.5% | **Negligible** | Little to no impact |
| Negative | **Redundant** | Removing improves performance |

---

## 4.6 Limitations of the Methodology

### 4.6.1 Single Run

**Limitation**: Each ablation is run once with seed=42.

**Implication**: Results may vary with different seeds.

**Mitigation**: Fixed seed ensures reproducibility; future work could add multiple runs.

### 4.6.2 Component Interactions

**Limitation**: We test single-component removals only.

**Implication**: Interactions between components not captured.

**Example**: Removing A and B together might have different effect than A alone + B alone.

### 4.6.3 Hyperparameter Dependence

**Limitation**: Hyperparameters optimized for full model.

**Implication**: Ablated models might perform better with re-tuned hyperparameters.

**Mitigation**: We use consistent hyperparameters to isolate component effects.

---

## 4.7 Quality Assurance

### 4.7.1 Sanity Checks

Before trusting results, we verify:

1. **Training converged**: Loss decreases over epochs
2. **No NaN/Inf values**: Gradients are stable
3. **Metrics are reasonable**: Values in expected ranges
4. **Baseline matches**: Confirms correct implementation

### 4.7.2 Code Review

All scripts are reviewed for:
- Correctness of ablation implementation
- Proper random seed setting
- Correct metric calculation
- Appropriate logging

### 4.7.3 Result Verification

Results are cross-checked:
- Training logs match reported metrics
- CSV exports match calculated values
- Multiple aggregation methods agree

---

## 4.8 Ethical Considerations

### 4.8.1 Reproducibility as Ethics

We consider reproducibility an ethical obligation:
- Other researchers can verify our claims
- Results are trustworthy
- Science advances through replication

### 4.8.2 Honest Reporting

We report:
- All ablation results (not just favorable ones)
- Negative results (components that don't help)
- Limitations of our methodology

### 4.8.3 Resource Usage

We document:
- Computational resources used
- Time required for experiments
- Environmental impact considerations

---

*Next: [05_ablation_design.md](05_ablation_design.md) - Detailed explanation of each ablation variant*
