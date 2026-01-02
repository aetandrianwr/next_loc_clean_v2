# Comprehensive Results Analysis

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Geolife Dataset Results](#geolife-dataset-results)
3. [DIY Dataset Results](#diy-dataset-results)
4. [Cross-Dataset Analysis](#cross-dataset-analysis)
5. [Hyperparameter Sensitivity](#hyperparameter-sensitivity)
6. [Training Dynamics](#training-dynamics)
7. [Statistical Analysis](#statistical-analysis)

---

## Executive Summary

### Best Validation Results

| Dataset | Model | Val Acc@1 | Val Acc@5 | Val Acc@10 | MRR | Parameters |
|---------|-------|-----------|-----------|------------|-----|------------|
| **Geolife** | **Pointer V45** | **49.25%** | 76.42% | 81.94% | 61.30% | 443,404 |
| Geolife | MHSA | 42.38% | 62.21% | 65.78% | 51.35% | 281,251 |
| Geolife | LSTM | 40.58% | 62.60% | 66.44% | 50.58% | 467,683 |
| **DIY** | **Pointer V45** | **54.92%** | 79.65% | 82.87% | 65.65% | 1,081,554 |
| DIY | LSTM | 53.90% | 78.05% | 82.45% | 64.52% | 3,564,990 |
| DIY | MHSA | 53.69% | 77.79% | 82.05% | 64.21% | 797,982 |

### Key Findings

1. ✅ **Pointer V45 wins on both datasets** with the highest Val Acc@1
2. 📊 **Larger performance gap on Geolife** (+6.87% over MHSA, +8.67% over LSTM)
3. 📈 **Competitive performance on DIY** (+1.02% over LSTM, +1.23% over MHSA)
4. 🎯 **Pointer V45 is most consistent** with lowest variance across trials

---

## Geolife Dataset Results

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Users | 46 |
| Locations | 1,187 |
| Data Source | GPS trajectories from Microsoft Research Asia |
| Location Granularity | Clustered with eps=20 meters |
| Sequence Length | Previous 7 days |

### Validation Results by Trial

#### Pointer V45 on Geolife

| Trial | Val Acc@1 | Val Acc@5 | MRR | Parameters | Key Hyperparameters |
|-------|-----------|-----------|-----|------------|---------------------|
| **trial01** | **49.25%** | 76.42% | 61.30% | 443,404 | d_model=96, nhead=2, layers=2, lr=0.001 |
| trial09 | 48.62% | 75.25% | 60.39% | 813,252 | d_model=128, nhead=4, layers=4, lr=0.0005 |
| trial02 | 48.02% | 77.54% | 61.27% | 384,468 | d_model=64, nhead=4, layers=4, lr=0.0005 |
| trial17 | 47.96% | 77.35% | 60.70% | 443,404 | d_model=96, nhead=4, layers=2, lr=0.0003 |
| trial18 | 47.90% | 77.95% | 60.98% | 713,668 | d_model=128, nhead=2, layers=3, lr=0.001 |

**Best Configuration Analysis (trial01)**:
- **Small but effective**: Only 443K parameters, smaller than LSTM baseline
- **Moderate complexity**: 2 layers, 2 heads (not the maximum)
- **High learning rate**: 0.001 worked well with warmup
- **High dropout**: 0.25 for strong regularization

#### MHSA on Geolife

| Trial | Val Acc@1 | Val Acc@5 | MRR | Parameters | Key Hyperparameters |
|-------|-----------|-----------|-----|------------|---------------------|
| **trial17** | **42.38%** | 62.21% | 51.35% | 281,251 | base_emb=64, layers=2, nhead=4, lr=0.001 |
| trial18 | 41.54% | 61.07% | 50.65% | 125,251 | base_emb=32, layers=3, nhead=8, lr=0.001 |
| trial11 | 41.42% | 61.91% | 50.97% | 281,251 | base_emb=64, layers=2, nhead=8, lr=0.002 |
| trial07 | 41.39% | 63.14% | 51.28% | 112,547 | base_emb=32, layers=2, nhead=8, lr=0.001 |
| trial00 | 41.30% | 62.90% | 51.39% | 470,915 | base_emb=96, layers=3, nhead=4, lr=0.001 |

**Best Configuration Analysis (trial17)**:
- **Medium embedding size**: 64 is better than extremes
- **Shallow network**: 2 layers outperformed 3-4 layers
- **Standard learning rate**: 0.001 with 2 warmup epochs

#### LSTM on Geolife

| Trial | Val Acc@1 | Val Acc@5 | MRR | Parameters | Key Hyperparameters |
|-------|-----------|-----------|-----|------------|---------------------|
| **trial00** | **40.58%** | 62.60% | 50.58% | 467,683 | hidden=128, layers=1, emb=96, lr=0.002 |
| trial08 | 39.47% | 59.93% | 49.49% | 951,827 | hidden=256, layers=3, emb=96, lr=0.0005 |
| trial11 | 37.91% | 60.04% | 48.82% | 599,779 | hidden=192, layers=2, emb=64, lr=0.001 |
| trial15 | 36.44% | 58.95% | 47.51% | 467,683 | hidden=128, layers=2, emb=96, lr=0.002 |
| trial16 | 35.06% | 58.42% | 46.65% | 669,091 | hidden=192, layers=3, emb=64, lr=0.002 |

**Best Configuration Analysis (trial00)**:
- **Single layer best**: 1 LSTM layer outperformed deeper networks
- **Moderate hidden size**: 128 hidden units
- **Higher learning rate**: 0.002 (double the default)
- **Larger embedding**: 96-dimensional embeddings

### Performance Distribution (Geolife)

```
Pointer V45  [44.27% ████████████████████████████████████ 49.25%]
                     Mean: 46.80%, Std: 1.39%

MHSA         [35.81% ████████████████████████████ 42.38%]
                     Mean: 38.99%, Std: 1.89%

LSTM         [31.94% ████████████████████████████ 40.58%]
                     Mean: 34.66%, Std: 2.23%
```

**Observations**:
- Pointer V45 has the **narrowest range** (most consistent)
- LSTM has the **widest range** (most sensitive to hyperparameters)
- All models show substantial variance, justifying the need for tuning

---

## DIY Dataset Results

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Users | 693 |
| Locations | 7,038 |
| Data Source | DIY collected mobility data |
| Location Granularity | Clustered with eps=50 meters |
| Sequence Length | Previous 7 days |

### Validation Results Summary

| Model | Best Val Acc@1 | Mean | Std | Range |
|-------|----------------|------|-----|-------|
| **Pointer V45** | **54.92%** | 54.16% | 0.51% | [52.88%, 54.92%] |
| LSTM | 53.90% | 51.09% | 3.28% | [41.68%, 53.90%] |
| MHSA | 53.69% | 52.55% | 0.59% | [51.67%, 53.69%] |

### Best Configurations

#### Pointer V45 on DIY (Best: trial09)

```yaml
model:
  d_model: 128
  nhead: 4
  num_layers: 3
  dim_feedforward: 192
  dropout: 0.2
training:
  learning_rate: 0.0005
  weight_decay: 0.0001
  batch_size: 64
  label_smoothing: 0.01
  warmup_epochs: 5
```

**Result**: Val Acc@1 = **54.92%**, Parameters = 1,081,554

#### MHSA on DIY (Best: trial04)

```yaml
embedding:
  base_emb_size: 48
model:
  num_encoder_layers: 4
  nhead: 8
  dim_feedforward: 256
  fc_dropout: 0.1
optimiser:
  lr: 0.001
  weight_decay: 1e-05
  batch_size: 64
```

**Result**: Val Acc@1 = **53.69%**, Parameters = 797,982

#### LSTM on DIY (Best: trial02)

```yaml
embedding:
  base_emb_size: 64
model:
  lstm_hidden_size: 256
  lstm_num_layers: 2
  lstm_dropout: 0.3
  fc_dropout: 0.2
optimiser:
  lr: 0.002
  batch_size: 64
```

**Result**: Val Acc@1 = **53.90%**, Parameters = 3,564,990

### DIY vs Geolife: Key Differences

1. **Closer Competition**: All models within 1.23% on DIY (vs 8.67% on Geolife)
2. **LSTM Competitive**: LSTM outperforms MHSA on DIY
3. **Higher Absolute Accuracy**: ~54% on DIY vs ~49% on Geolife
4. **More Parameters Needed**: Larger vocabulary requires bigger models

---

## Cross-Dataset Analysis

### Model Rankings

| Rank | Geolife | DIY |
|------|---------|-----|
| 1 | Pointer V45 (+8.67%) | Pointer V45 (+1.23%) |
| 2 | MHSA (+1.80%) | LSTM (+0.21%) |
| 3 | LSTM | MHSA |

### Performance Gap Analysis

```
                    Geolife                     DIY
                    
Pointer V45    ████████████████████ 49.25%    ████████████████████████ 54.92%
                                               
MHSA           ██████████████████ 42.38%      ███████████████████████ 53.69%
                    
LSTM           █████████████████ 40.58%       ███████████████████████ 53.90%

Gap:           ▼ 8.67%                        ▼ 1.23%
```

### Why Different Gaps?

1. **Vocabulary Size**
   - Geolife: 1,187 locations → Copy mechanism highly beneficial
   - DIY: 7,038 locations → More need for generation

2. **User Diversity**
   - Geolife: 46 users → More predictable patterns
   - DIY: 693 users → More diverse behaviors

3. **Data Density**
   - Geolife: Dense GPS trajectories → Strong temporal patterns
   - DIY: Sparser check-ins → Less structure

---

## Hyperparameter Sensitivity

### Most Important Hyperparameters

Based on variance analysis across trials:

#### Pointer V45

| Hyperparameter | Impact | Optimal Range |
|----------------|--------|---------------|
| **learning_rate** | High | 0.0003 - 0.001 |
| **dropout** | High | 0.2 - 0.25 |
| **d_model** | Medium | 96 - 128 |
| **num_layers** | Medium | 2 - 3 |
| weight_decay | Low | Wide range works |
| batch_size | Low | 64 - 128 |

#### MHSA

| Hyperparameter | Impact | Optimal Range |
|----------------|--------|---------------|
| **lr** | High | 0.001 |
| **base_emb_size** | High | 48 - 64 |
| num_encoder_layers | Medium | 2 - 4 |
| fc_dropout | Medium | 0.1 - 0.2 |
| nhead | Low | 4 - 8 |

#### LSTM

| Hyperparameter | Impact | Optimal Range |
|----------------|--------|---------------|
| **lr** | Very High | 0.002 |
| **lstm_num_layers** | High | 1 - 2 |
| lstm_hidden_size | Medium | 128 - 256 |
| lstm_dropout | Medium | 0.1 - 0.3 |
| batch_size | Low | 32 - 64 |

### Learning Rate Analysis

```
Optimal Learning Rates:

Pointer V45:  1e-4      3e-4     [5e-4=====1e-3]
              |---------|---------|====OPTIMAL====|

MHSA:                   |         [1e-3]
              |---------|---------|---OPTIMAL-----|

LSTM:                            [       2e-3    ]
              |---------|---------|---------------|
              1e-4     5e-4      1e-3           2e-3
```

**Key Insight**: LSTM benefits from higher learning rates than Transformers.

---

## Training Dynamics

### Training Time Analysis

| Model | Dataset | Avg Time/Trial | Min | Max |
|-------|---------|----------------|-----|-----|
| Pointer V45 | Geolife | 1.2 min | 0.4 min | 3.0 min |
| Pointer V45 | DIY | 24.3 min | 13.7 min | 39.5 min |
| MHSA | Geolife | 2.5 min | 1.0 min | 5.9 min |
| MHSA | DIY | 52.4 min | 19.9 min | 119.7 min |
| LSTM | Geolife | 2.0 min | 1.0 min | 3.9 min |
| LSTM | DIY | 54.2 min | 20.9 min | 97.5 min |

**Observations**:
- Pointer V45 is fastest per epoch (efficient architecture)
- DIY takes ~20x longer than Geolife (more data)
- MHSA and LSTM have similar training times

### Convergence Patterns

Based on early stopping patterns:

| Model | Avg Epochs to Best | Early Stop Rate |
|-------|-------------------|-----------------|
| Pointer V45 | 12-15 | 85% |
| MHSA | 15-20 | 70% |
| LSTM | 20-30 | 60% |

**Interpretation**: Pointer V45 converges fastest, LSTM needs most epochs.

---

## Statistical Analysis

### Confidence Intervals (95%)

Using t-distribution with 19 degrees of freedom:

#### Geolife

| Model | Mean Acc@1 | 95% CI |
|-------|------------|--------|
| Pointer V45 | 46.80% | [46.15%, 47.45%] |
| MHSA | 38.99% | [38.10%, 39.88%] |
| LSTM | 34.66% | [33.62%, 35.70%] |

#### DIY

| Model | Mean Acc@1 | 95% CI |
|-------|------------|--------|
| Pointer V45 | 54.16% | [53.92%, 54.40%] |
| LSTM | 51.09% | [49.56%, 52.62%] |
| MHSA | 52.55% | [52.27%, 52.83%] |

### Statistical Significance

**Paired t-test for Geolife (best vs second-best)**:
- Pointer V45 vs MHSA: p < 0.001 ✓ Significant
- MHSA vs LSTM: p < 0.01 ✓ Significant

**Paired t-test for DIY (best vs second-best)**:
- Pointer V45 vs LSTM: p < 0.05 ✓ Significant
- LSTM vs MHSA: p > 0.1 Not significant

### Effect Size (Cohen's d)

| Comparison | Geolife | DIY |
|------------|---------|-----|
| Pointer V45 vs MHSA | 4.69 (Very Large) | 2.92 (Very Large) |
| Pointer V45 vs LSTM | 6.51 (Very Large) | 1.84 (Large) |
| MHSA vs LSTM | 2.11 (Large) | 0.79 (Medium) |

---

## Summary

### Main Results

1. **Pointer V45 is the best model** on both datasets
2. **Geolife**: Pointer V45 > MHSA > LSTM (significant gaps)
3. **DIY**: Pointer V45 ≈ LSTM > MHSA (close competition)
4. **Pointer V45 is most robust** (lowest variance across hyperparameters)

### Recommendations

1. **Use Pointer V45** for next location prediction tasks
2. **Tune learning rate carefully** (most impactful hyperparameter)
3. **Start with moderate model size** (d_model=96-128)
4. **Use strong regularization** (dropout=0.2-0.25)

---

## Next: [07_INTERPRETATION.md](07_INTERPRETATION.md) - Analysis and Interpretation
