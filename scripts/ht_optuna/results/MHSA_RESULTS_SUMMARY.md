# MHSA Hyperparameter Tuning Results Summary

**Date:** December 31, 2025 - January 1, 2026  
**Command:** `python scripts/ht_optuna/run_optuna_ht.py --model MHSA --n_jobs 10`  
**Methodology:** Best configuration selected by highest Acc@1 on validation set  

---

## Summary Statistics

- **Total Trials Attempted:** 160 (2 datasets × 4 prev_days × 20 trials)
- **Successful Trials:** 138 (86.25%)
- **Failed Trials:** 22 (13.75%)
  - DIY prev_day=10: 5 failed (25% failure rate)
  - DIY prev_day=14: 17 failed (85% failure rate)
  - Root cause: GPU OOM errors due to n_jobs=10 parallel execution

---

## DIY Dataset Results

### Test Set Performance (Best Configurations)

| prev_day | Acc@1  | Acc@5  | Acc@10 | MRR   | NDCG  | Trials |
|----------|--------|--------|--------|-------|-------|--------|
| **3**    | 53.19% | 77.23% | 81.43% | 63.83 | 67.89 | 20/20  |
| **7**    | 53.08% | 76.79% | 81.06% | 63.57 | 67.60 | 20/20  |
| **10**   | 51.52% | 76.16% | 80.35% | 62.33 | 66.48 | 15/20  |
| **14**   | 49.66% | 75.81% | 80.08% | 61.00 | 65.42 | 3/20   |

### Best Hyperparameters by prev_day

#### prev_day = 3 (Best Overall: Acc@1 = 53.19%)
- base_emb_size: **128**
- nhead: **2**
- num_encoder_layers: **4**
- dim_feedforward: **128**
- learning_rate: **0.001**

#### prev_day = 7 (Acc@1 = 53.08%)
- base_emb_size: **128**
- nhead: **8**
- num_encoder_layers: **2**
- dim_feedforward: **128**
- learning_rate: **0.002**

#### prev_day = 10 (Acc@1 = 51.52%)
- base_emb_size: **128**
- nhead: **2**
- num_encoder_layers: **4**
- dim_feedforward: **512**
- learning_rate: **0.001**

#### prev_day = 14 (Acc@1 = 49.66%)
- base_emb_size: **128**
- nhead: **2**
- num_encoder_layers: **8**
- dim_feedforward: **512**
- learning_rate: **0.0005**

### DIY Observations
- **Trend:** Performance decreases as prev_day increases (3 → 7 → 10 → 14)
- **Best Setting:** prev_day=3 achieves highest Acc@1 (53.19%)
- **Preferred Architecture:** base_emb_size=128 (all best configs)
- **Val-Test Gap:** Small (1-3%), indicating good generalization

---

## GeoLife Dataset Results

### Test Set Performance (Best Configurations)

| prev_day | Acc@1  | Acc@5  | Acc@10 | MRR   | NDCG  | Trials |
|----------|--------|--------|--------|-------|-------|--------|
| **3**    | 33.34% | 56.28% | 59.43% | 43.45 | 47.11 | 20/20  |
| **7**    | 31.35% | 55.45% | 58.37% | 42.12 | 45.89 | 20/20  |
| **10**   | 31.31% | 55.32% | 58.13% | 41.97 | 45.67 | 20/20  |
| **14**   | 35.57% | 56.37% | 58.70% | 45.05 | 48.20 | 20/20  |

### Best Hyperparameters by prev_day

#### prev_day = 3 (Acc@1 = 33.34%)
- base_emb_size: **128**
- nhead: **2**
- num_encoder_layers: **2**
- dim_feedforward: **256**
- learning_rate: **0.002**

#### prev_day = 7 (Acc@1 = 31.35%)
- base_emb_size: **64**
- nhead: **2**
- num_encoder_layers: **2**
- dim_feedforward: **512**
- learning_rate: **0.002**

#### prev_day = 10 (Acc@1 = 31.31%)
- base_emb_size: **128**
- nhead: **8**
- num_encoder_layers: **2**
- dim_feedforward: **512**
- learning_rate: **0.001**

#### prev_day = 14 (Best Overall: Acc@1 = 35.57%)
- base_emb_size: **32**
- nhead: **2**
- num_encoder_layers: **2**
- dim_feedforward: **256**
- learning_rate: **0.002**

### GeoLife Observations
- **Trend:** prev_day=14 achieves best performance (35.57% Acc@1)
- **Stability:** All trials completed successfully (100% success rate)
- **Preferred Architecture:** Shallow models (2 encoder layers for all)
- **Val-Test Gap:** Larger (7-12%), suggesting overfitting on small validation set

---

## Key Insights

### Hyperparameter Patterns

1. **Encoder Layers:**
   - DIY: Varies (2-8 layers), deeper for longer history
   - GeoLife: Consistently 2 layers (shallow models)

2. **Attention Heads:**
   - Most configs prefer **nhead=2** (simpler attention)
   - DIY prev_day=7 and GeoLife prev_day=10 use nhead=8

3. **Embedding Size:**
   - DIY: Prefers larger (base_emb_size=128)
   - GeoLife: More flexible (32-128)

4. **Learning Rate:**
   - Range: 0.0005 - 0.002
   - Higher LR (0.002) common for GeoLife

### Dataset Differences

| Aspect | DIY | GeoLife |
|--------|-----|---------|
| Best prev_day | 3 (53.19%) | 14 (35.57%) |
| Model Depth | Deeper (4-8 layers) | Shallow (2 layers) |
| Trial Success | 75-100% | 100% |
| Val-Test Gap | Small (1-3%) | Larger (7-12%) |

### Recommendations

1. **For DIY Dataset:**
   - Use prev_day=3 for best performance
   - Use base_emb_size=128, nhead=2, 4 layers, dim_feedforward=128
   - Be cautious with large models when using parallel jobs

2. **For GeoLife Dataset:**
   - Use prev_day=14 if longer history preferred
   - Keep models shallow (2 encoder layers)
   - Use smaller base_emb_size (32-64 works well)

3. **For Future Hyperparameter Tuning:**
   - Reduce n_jobs (use 2-4 instead of 10) to avoid OOM errors
   - Focus search on successful ranges:
     - DIY: base_emb=128, layers=2-4, ff=128-512
     - GeoLife: base_emb=32-128, layers=2, ff=256-512

---

## Files Generated

1. **MHSA_val_results_20251231_085019.csv** - All validation results (138 trials)
2. **MHSA_test_results_20251231_085019.csv** - All test results (138 trials)
3. **MHSA_failed_trials_20251231_085019.csv** - Failed trial details (22 trials)
4. **MHSA_best_configs_summary.csv** - Best configurations per dataset/prev_day (8 configs)
5. **MHSA_RESULTS_SUMMARY.md** - This document

---

## Contact & Reproducibility

**Best Config Paths:**
- DIY prev3: `config/ht_optuna/optuna_MHSA_diy_prev3_t15_20251231_085020_28a1fe93.yaml`
- DIY prev7: `config/ht_optuna/optuna_MHSA_diy_prev7_t8_20251231_103724_fe98f18a.yaml`
- DIY prev10: `config/ht_optuna/optuna_MHSA_diy_prev10_t6_20251231_124302_de9ea08d.yaml`
- DIY prev14: `config/ht_optuna/optuna_MHSA_diy_prev14_t7_20251231_143153_7740b5f0.yaml`
- GeoLife prev3: `config/ht_optuna/optuna_MHSA_geolife_prev3_t17_20251231_151924_d562d5d3.yaml`
- GeoLife prev7: `config/ht_optuna/optuna_MHSA_geolife_prev7_t10_20251231_153533_53515622.yaml`
- GeoLife prev10: `config/ht_optuna/optuna_MHSA_geolife_prev10_t0_20251231_154749_a3c2cc92.yaml`
- GeoLife prev14: `config/ht_optuna/optuna_MHSA_geolife_prev14_t16_20251231_161445_ce287941.yaml`

All configurations can be re-run using:
```bash
python src/training/train_MHSA.py --config <config_path>
```
