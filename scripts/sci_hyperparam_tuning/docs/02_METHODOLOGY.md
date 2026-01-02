# Scientific Methodology

## Table of Contents

1. [Theoretical Foundation](#theoretical-foundation)
2. [Experimental Design](#experimental-design)
3. [Random Search Strategy](#random-search-strategy)
4. [Training Protocol](#training-protocol)
5. [Evaluation Protocol](#evaluation-protocol)
6. [Statistical Considerations](#statistical-considerations)

---

## Theoretical Foundation

### Hyperparameter Optimization Background

Hyperparameter optimization (HPO) is the process of finding the best configuration of hyperparameters for a machine learning model. Unlike model parameters (weights, biases) that are learned during training, hyperparameters are set before training begins and control the learning process itself.

#### Types of Hyperparameters

1. **Architecture Hyperparameters**
   - Model dimension (`d_model`)
   - Number of layers (`num_layers`)
   - Number of attention heads (`nhead`)
   - Feedforward dimension (`dim_feedforward`)
   - Hidden sizes for recurrent networks

2. **Optimization Hyperparameters**
   - Learning rate
   - Weight decay (L2 regularization)
   - Batch size
   - Number of warmup epochs

3. **Regularization Hyperparameters**
   - Dropout rate
   - Label smoothing

### Why Random Search Over Grid Search?

**Grid Search** exhaustively evaluates all combinations of predefined hyperparameter values:
```
learning_rate: [0.0001, 0.001, 0.01]
batch_size: [32, 64, 128]
→ 9 combinations
```

**Problems with Grid Search**:
1. Exponential growth: Adding parameters multiplies combinations
2. Wasted computation: Many similar values explored
3. Assumes all parameters equally important

**Random Search** samples hyperparameter values randomly from defined distributions:

```python
learning_rate ~ Uniform(0.0001, 0.01)
batch_size ~ Choice([32, 64, 128, 256])
```

**Advantages of Random Search** (Bergstra & Bengio, 2012):

1. **More Efficient Exploration**: Explores more unique values of important hyperparameters
2. **Handles Varying Importance**: If `learning_rate` matters more than `batch_size`, random search explores more `learning_rate` values
3. **Better Coverage**: Grid search may miss optimal regions between grid points
4. **Reproducible**: With fixed seed, experiments are exactly reproducible

### Visual Intuition

Consider optimizing over 2 parameters where only one matters:

```
Grid Search (9 points):         Random Search (9 points):
• • •                          •    •  •
• • •                            •    •
• • •                          •  •   •  •

Only 3 unique values of         9 unique values of the
the important parameter!        important parameter explored!
```

---

## Experimental Design

### Overall Framework

```
┌─────────────────────────────────────────────────────────────────┐
│                    EXPERIMENTAL DESIGN                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │   Geolife   │    │     DIY     │    │   Dataset   │         │
│  │   Dataset   │    │   Dataset   │    │   Splits    │         │
│  │  46 users   │    │  693 users  │    │ Train/Val/  │         │
│  │ 1187 locs   │    │ 7038 locs   │    │    Test     │         │
│  └──────┬──────┘    └──────┬──────┘    └─────────────┘         │
│         │                  │                                    │
│         └────────┬─────────┘                                    │
│                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Generate Configurations                      │   │
│  │    Random Search: 20 trials × 3 models × 2 datasets      │   │
│  │                   = 120 configurations                    │   │
│  └─────────────────────────────────────────────────────────┘   │
│                  │                                              │
│                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Parallel Training                           │   │
│  │         5 concurrent jobs, auto-resume                   │   │
│  │         Fixed seed=42 for reproducibility                │   │
│  └─────────────────────────────────────────────────────────┘   │
│                  │                                              │
│                  ▼                                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Evaluation & Logging                        │   │
│  │    Val Acc@1 for model selection (primary metric)        │   │
│  │    Test metrics for final reporting                      │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Experimental Matrix

| Factor | Values | Description |
|--------|--------|-------------|
| Models | Pointer V45, MHSA, LSTM | 3 architectures to compare |
| Datasets | Geolife, DIY | 2 mobility datasets |
| Trials | 20 per model-dataset | Random search iterations |
| Total | 120 experiments | 3 × 2 × 20 = 120 |

### Design Principles

1. **Equal Computational Budget**
   - Same number of trials (20) for each model-dataset pair
   - Same maximum epochs (50) for all models
   - Same early stopping patience (5 epochs)

2. **Reproducibility**
   - Fixed random seed: 42
   - Deterministic seed derivation for each trial
   - All configurations saved as YAML files

3. **Fair Comparison**
   - Same evaluation metrics for all models
   - Same train/val/test splits
   - Same preprocessing pipeline

---

## Random Search Strategy

### Seed Management

Each hyperparameter configuration is generated using a deterministic seed derived from:

```python
seed = hash(f"{model_name}_{dataset}_{base_seed}_{trial_idx}") % (2**32)
```

This ensures:
- Same seed always generates same configuration
- Different trials get different configurations
- Full reproducibility across runs

### Configuration Generation Process

```python
def sample_hyperparameters(search_space, seed):
    """Sample a random hyperparameter configuration."""
    random.seed(seed)
    config = {}
    for param, values in search_space.items():
        config[param] = random.choice(values)
    return config
```

The `random.choice()` function samples uniformly from the discrete set of values defined in the search space.

### Example: Pointer V45 Configuration Generation

```python
POINTER_V45_SEARCH_SPACE = {
    # Architecture
    'd_model': [64, 96, 128],           # Model dimension
    'nhead': [2, 4, 8],                  # Attention heads
    'num_layers': [2, 3, 4],             # Encoder layers
    'dim_feedforward': [128, 192, 256],  # FFN dimension
    'dropout': [0.1, 0.15, 0.2, 0.25],   # Dropout rate
    
    # Optimization
    'learning_rate': [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 0.01, 0.015],
    'batch_size': [64, 128, 256],
    'label_smoothing': [0.0, 0.01, 0.03, 0.05],
    'warmup_epochs': [3, 5, 7],
}

# Total combinations: 3×3×3×3×4×5×7×3×4×3 = 816,480 possible configs
# We sample 20 randomly per dataset
```

### Search Space Size Analysis

| Model | Parameters | Possible Combinations |
|-------|------------|----------------------|
| Pointer V45 | 10 | 816,480 |
| MHSA | 9 | 20,736 |
| LSTM | 10 | 41,472 |

With only 20 trials per model-dataset, we sample ~0.002% to ~0.1% of the search space. Random search is effective because most hyperparameters have low importance—performance is dominated by a few key parameters.

---

## Training Protocol

### Training Configuration

| Setting | Value | Rationale |
|---------|-------|-----------|
| Maximum Epochs | 50 | Sufficient for convergence |
| Early Stopping Patience | 5 | Prevent overfitting |
| Minimum Epochs | 8 | Ensure learning before early stop |
| Gradient Clipping | 0.8 | Pointer V45 only, stabilize training |
| Mixed Precision | Yes | Faster training, less memory |

### Learning Rate Schedule

**Pointer V45**: Warmup + Cosine Annealing
```
LR
^
|   /\
|  /  \___________________
| /                       \
|/                         \
+----------------------------> Epoch
  warmup   cosine annealing
```

**MHSA & LSTM**: Linear Warmup + Decay
```
LR
^
|   /--------\
|  /          \
| /            \
|/              \_______
+-------------------------> Epoch
  warmup    linear decay
```

### Early Stopping Logic

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.counter = 0
        self.best_loss = None
        
    def step(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # Continue training
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False
```

---

## Evaluation Protocol

### Metrics

All models are evaluated using the same metrics from `src/evaluation/metrics.py`:

#### Primary Metric: Accuracy@1 (Acc@1)

$$\text{Acc@1} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{top-1 prediction}_i = \text{target}_i] \times 100\%$$

This is the **primary metric for model selection**—we choose the hyperparameters that maximize Val Acc@1.

#### Secondary Metrics

**Accuracy@k (k=5, 10)**:
$$\text{Acc@k} = \frac{1}{N} \sum_{i=1}^{N} \mathbb{1}[\text{target}_i \in \text{top-k predictions}_i] \times 100\%$$

**Mean Reciprocal Rank (MRR)**:
$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \frac{1}{\text{rank}_i}$$

where $\text{rank}_i$ is the position of the correct target in the sorted predictions.

**Normalized Discounted Cumulative Gain (NDCG@10)**:
$$\text{NDCG@10}_i = \frac{1}{\log_2(\text{rank}_i + 1)} \text{ if rank}_i \leq 10 \text{ else } 0$$

**F1 Score**: Weighted F1 for multi-class classification

### Evaluation Procedure

```python
def evaluate(model, dataloader, device):
    model.eval()
    all_metrics = []
    
    with torch.no_grad():
        for batch in dataloader:
            logits = model(batch)
            targets = batch['target']
            
            # Calculate all metrics
            metrics = calculate_correct_total_prediction(logits, targets)
            all_metrics.append(metrics)
    
    # Aggregate and compute final metrics
    total_metrics = sum(all_metrics)
    return get_performance_dict(total_metrics)
```

---

## Statistical Considerations

### Why 20 Trials Per Model-Dataset?

1. **Computational Budget**: ~120 experiments × 15-60 min each ≈ 30-40 hours GPU time
2. **Statistical Significance**: 20 trials provide reasonable variance estimation
3. **PhD-Level Standard**: Consistent with published research practices

### Variance Analysis

Each model-dataset combination has 20 different hyperparameter configurations. We report:
- **Best**: Maximum Val Acc@1 across all 20 trials
- **Mean**: Average Val Acc@1 across all 20 trials
- **Std**: Standard deviation of Val Acc@1

This shows not just the best achievable performance, but also:
- How sensitive the model is to hyperparameters
- How robust the model is across different configurations

### Fair Comparison Criteria

| Criterion | Implementation |
|-----------|---------------|
| Same search budget | 20 trials for each model-dataset |
| Same training budget | Max 50 epochs, early stopping patience=5 |
| Same evaluation | Identical metrics, same test set |
| Same data | Same preprocessed data files |
| Reproducibility | Fixed seed=42, all configs logged |

---

## Summary

The methodology follows these scientific principles:

1. **Random Search** over grid search for efficiency (Bergstra & Bengio, 2012)
2. **Fixed Seed** (42) for complete reproducibility
3. **Equal Budget** for fair model comparison
4. **Comprehensive Logging** of all configurations and results
5. **Validation-based Selection** to avoid test set leakage
6. **Multiple Metrics** for thorough evaluation

This approach ensures that any observed performance differences between models are due to architectural differences, not tuning artifacts.

---

## Next: [03_SEARCH_SPACE.md](03_SEARCH_SPACE.md) - Hyperparameter Search Space Design
