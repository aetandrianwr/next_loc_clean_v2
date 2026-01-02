# LSTM vs MHSA Comparison for prev_day=7

**Pipeline Used:** Find best config on validation set (highest Acc@1) → Report test set performance

---

## Results Summary

### DIY Dataset (prev_day=7)

#### LSTM - Best Configuration
**Selected by:** Highest Val Acc@1 = 52.55%

**Hyperparameters:**
- base_emb_size: 128
- lstm_hidden_size: 64
- lstm_num_layers: 4
- dim_feedforward: 256
- learning_rate: 0.0015

**Test Set Performance:**
- **Acc@1: 52.86%** ← Final result
- Acc@5: 77.20%
- Acc@10: 80.90%
- MRR: 63.53
- NDCG: 67.53

**Trials:** 20/20 completed successfully

---

#### MHSA - Best Configuration
**Selected by:** Highest Val Acc@1 = 52.55%

**Hyperparameters:**
- base_emb_size: 128
- nhead: 8
- num_encoder_layers: 2
- dim_feedforward: 128
- learning_rate: 0.002

**Test Set Performance:**
- **Acc@1: 53.08%** ← Final result
- Acc@5: 76.79%
- Acc@10: 81.06%
- MRR: 63.57
- NDCG: 67.60

**Trials:** 20/20 completed successfully

---

#### DIY Comparison (prev_day=7)

| Metric | LSTM | MHSA | Difference | Winner |
|--------|------|------|------------|--------|
| **Acc@1** | 52.86% | **53.08%** | -0.22% | **MHSA ✓** |
| Acc@5 | **77.20%** | 76.79% | +0.41% | LSTM ✓ |
| Acc@10 | 80.90% | **81.06%** | -0.16% | MHSA ✓ |
| MRR | 63.53 | **63.57** | -0.04 | MHSA ✓ |
| NDCG | 67.53 | **67.60** | -0.07 | MHSA ✓ |

**Winner:** MHSA (4 out of 5 metrics)  
**Acc@1 Advantage:** MHSA by 0.22%

---

### GeoLife Dataset (prev_day=7)

#### LSTM - Best Configuration
**Selected by:** Highest Val Acc@1 = 35.45%

**Hyperparameters:**
- base_emb_size: 32
- lstm_hidden_size: 32
- lstm_num_layers: 2
- dim_feedforward: 128
- learning_rate: 0.002

**Test Set Performance:**
- **Acc@1: 32.04%** ← Final result
- Acc@5: 54.31%
- Acc@10: 58.02%
- MRR: 42.37
- NDCG: 45.94

**Trials:** 20/20 completed successfully

---

#### MHSA - Best Configuration
**Selected by:** Highest Val Acc@1 = 42.95%

**Hyperparameters:**
- base_emb_size: 64
- nhead: 2
- num_encoder_layers: 2
- dim_feedforward: 512
- learning_rate: 0.002

**Test Set Performance:**
- **Acc@1: 31.35%** ← Final result
- Acc@5: 55.45%
- Acc@10: 58.37%
- MRR: 42.12
- NDCG: 45.89

**Trials:** 20/20 completed successfully

---

#### GeoLife Comparison (prev_day=7)

| Metric | LSTM | MHSA | Difference | Winner |
|--------|------|------|------------|--------|
| **Acc@1** | **32.04%** | 31.35% | +0.69% | **LSTM ✓** |
| Acc@5 | 54.31% | **55.45%** | -1.14% | MHSA ✓ |
| Acc@10 | 58.02% | **58.37%** | -0.34% | MHSA ✓ |
| MRR | **42.37** | 42.12 | +0.25 | **LSTM ✓** |
| NDCG | **45.94** | 45.89 | +0.06 | **LSTM ✓** |

**Winner:** LSTM (3 out of 5 metrics, including Acc@1)  
**Acc@1 Advantage:** LSTM by 0.69%

---

## Overall Conclusion for prev_day=7

```
┌─────────────┬────────────────┬────────────────┬──────────────┬─────────┐
│   Dataset   │  LSTM Acc@1    │  MHSA Acc@1    │  Difference  │  Winner │
├─────────────┼────────────────┼────────────────┼──────────────┼─────────┤
│     DIY     │     52.86%     │     53.08%     │    -0.22%    │  MHSA ✓ │
│   GeoLife   │     32.04%     │     31.35%     │    +0.69%    │  LSTM ✓ │
└─────────────┴────────────────┴────────────────┴──────────────┴─────────┘
```

### Key Findings:

1. **Each model wins on one dataset:**
   - MHSA is better on DIY (53.08% vs 52.86%)
   - LSTM is better on GeoLife (32.04% vs 31.35%)

2. **Differences are minimal:**
   - DIY: 0.22% difference (MHSA advantage)
   - GeoLife: 0.69% difference (LSTM advantage)
   - Both differences are < 1%, practically negligible

3. **Validation-Test Gap:**
   - DIY LSTM: Very small gap (52.55% → 52.86%, +0.31%)
   - DIY MHSA: Very small gap (52.55% → 53.08%, +0.53%)
   - GeoLife LSTM: Large gap (35.45% → 32.04%, -3.41%)
   - GeoLife MHSA: Large gap (42.95% → 31.35%, -11.60%)
   - **Note:** GeoLife shows significant overfitting to validation set

4. **Architecture Differences:**
   - LSTM prefers deeper networks (4 layers for DIY)
   - MHSA prefers shallow networks (2 layers for both datasets)
   - DIY benefits from larger embeddings (128) for both models
   - GeoLife works with smaller models (32-64 base_emb_size)

5. **Verdict:**
   - **No clear winner** - each model has strengths on different datasets
   - Performance differences are too small to matter in practice
   - Both are viable choices for prev_day=7

### Recommendation:

For **prev_day=7**, choose based on:
- **DIY dataset:** Slight preference for MHSA (+0.22%)
- **GeoLife dataset:** Slight preference for LSTM (+0.69%)
- **Overall:** Either model works well; differences are negligible

In practice, consider other factors like:
- Training time and computational cost
- Model interpretability needs
- Deployment constraints
- Preference for attention mechanisms vs sequential processing
