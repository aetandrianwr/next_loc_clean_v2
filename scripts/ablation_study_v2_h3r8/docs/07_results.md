# 7. Results

## Complete Experimental Results and Data

---

## 7.1 Baseline Validation

Before running ablations, we verified that our baseline (full model) matches expected performance:

### GeoLife Baseline
```
Expected Acc@1: 51.39%
Achieved Acc@1: 51.43%
Difference:     +0.04% ✅ (within tolerance)
```

### DIY Baseline
```
Expected Acc@1: 56.58%
Achieved Acc@1: 56.57%
Difference:     -0.01% ✅ (within tolerance)
```

**Conclusion**: Baseline validated. Proceeding with ablation study.

---

## 7.2 GeoLife Dataset Results

### 7.2.1 Complete Results Table

| Ablation | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG | F1 | Loss | ΔAcc@1 |
|----------|-------|-------|--------|-----|------|-----|------|--------|
| **full** | **51.43%** | 81.18% | 85.04% | 64.57% | 69.48% | 0.47% | 2.6300 | — |
| no_generation | 51.86% | 82.41% | 85.38% | 64.95% | 69.58% | 0.48% | 4.5829 | +0.43 |
| single_layer | 51.68% | 81.70% | 85.01% | 64.96% | 69.81% | 0.47% | 2.5142 | +0.26 |
| no_position_bias | 51.48% | 81.21% | 84.98% | 64.61% | 69.49% | 0.47% | 2.6297 | +0.06 |
| no_gate | 49.54% | 81.64% | 84.67% | 63.57% | 68.67% | 0.45% | 2.6445 | -1.88 |
| no_pos_from_end | 49.34% | 80.87% | 84.75% | 63.38% | 68.53% | 0.44% | 2.6639 | -2.08 |
| no_user | 49.11% | 81.10% | 84.12% | 63.27% | 68.33% | 0.44% | 2.6091 | -2.31 |
| no_temporal | 47.40% | 81.47% | 85.09% | 62.56% | 68.03% | 0.42% | 2.6342 | -4.03 |
| no_pointer | 27.41% | 54.14% | 58.65% | 38.88% | 43.43% | 0.20% | 4.8392 | **-24.01** |

### 7.2.2 Sorted by Impact (Most to Least Important)

| Rank | Component | Acc@1 Drop | Relative Drop | Category |
|------|-----------|------------|---------------|----------|
| 1 | Pointer Mechanism | -24.01% | 46.7% | **Critical** |
| 2 | Temporal Embeddings | -4.03% | 7.8% | Important |
| 3 | User Embedding | -2.31% | 4.5% | Important |
| 4 | Position-from-End | -2.08% | 4.1% | Important |
| 5 | Adaptive Gate | -1.88% | 3.7% | Minor |
| 6 | Position Bias | +0.06% | -0.1% | Negligible |
| 7 | Transformer Depth | +0.26% | -0.5% | Redundant |
| 8 | Generation Head | +0.43% | -0.8% | Redundant |

### 7.2.3 Visual Comparison

```
GeoLife Acc@1 by Ablation
                                                 
full            ████████████████████████████████████████████████████ 51.43%
no_generation   █████████████████████████████████████████████████████ 51.86%
single_layer    ████████████████████████████████████████████████████ 51.68%
no_position_bias████████████████████████████████████████████████████ 51.48%
no_gate         ███████████████████████████████████████████████████ 49.54%
no_pos_from_end ██████████████████████████████████████████████████ 49.34%
no_user         ██████████████████████████████████████████████████ 49.11%
no_temporal     ████████████████████████████████████████████████ 47.40%
no_pointer      ████████████████████████████ 27.41%  ← DRAMATIC DROP!
                0        10        20        30        40        50%
```

---

## 7.3 DIY Dataset Results

### 7.3.1 Complete Results Table

| Ablation | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG | F1 | Loss | ΔAcc@1 |
|----------|-------|-------|--------|-----|------|-----|------|--------|
| no_generation | 57.41% | 81.80% | 84.48% | 67.88% | 71.92% | 0.52% | 5.3675 | +0.84 |
| no_pos_from_end | 56.74% | 82.28% | 85.27% | 67.82% | 72.03% | 0.52% | 2.8800 | +0.16 |
| no_position_bias | 56.65% | 82.14% | 85.16% | 67.70% | 71.90% | 0.52% | 2.8733 | +0.08 |
| single_layer | 56.65% | 81.90% | 85.04% | 67.58% | 71.78% | 0.52% | 2.8829 | +0.08 |
| **full** | **56.57%** | 82.16% | 85.16% | 67.66% | 71.88% | 0.52% | 2.8736 | — |
| no_user | 56.27% | 81.98% | 84.89% | 67.31% | 71.57% | 0.51% | 2.9132 | -0.31 |
| no_gate | 56.08% | 81.90% | 85.28% | 67.22% | 71.56% | 0.51% | 2.9430 | -0.49 |
| no_temporal | 55.95% | 82.03% | 85.24% | 67.24% | 71.56% | 0.50% | 2.8855 | -0.62 |
| no_pointer | 51.90% | 75.59% | 78.27% | 62.21% | 66.05% | 0.46% | 3.5887 | **-4.67** |

### 7.3.2 Sorted by Impact (Most to Least Important)

| Rank | Component | Acc@1 Drop | Relative Drop | Category |
|------|-----------|------------|---------------|----------|
| 1 | Pointer Mechanism | -4.67% | 8.3% | Important |
| 2 | Temporal Embeddings | -0.62% | 1.1% | Minor |
| 3 | Adaptive Gate | -0.49% | 0.9% | Minor |
| 4 | User Embedding | -0.31% | 0.5% | Minor |
| 5 | Position Bias | +0.08% | -0.1% | Negligible |
| 6 | Transformer Depth | +0.08% | -0.1% | Negligible |
| 7 | Position-from-End | +0.16% | -0.3% | Redundant |
| 8 | Generation Head | +0.84% | -1.5% | Redundant |

### 7.3.3 Visual Comparison

```
DIY Acc@1 by Ablation
                                                 
no_generation   █████████████████████████████████████████████████████████ 57.41%
no_pos_from_end █████████████████████████████████████████████████████████ 56.74%
no_position_bias████████████████████████████████████████████████████████ 56.65%
single_layer    ████████████████████████████████████████████████████████ 56.65%
full            ████████████████████████████████████████████████████████ 56.57%
no_user         ████████████████████████████████████████████████████████ 56.27%
no_gate         ███████████████████████████████████████████████████████ 56.08%
no_temporal     ███████████████████████████████████████████████████████ 55.95%
no_pointer      █████████████████████████████████████████████████████ 51.90%  ← DROP
                0        10        20        30        40        50%
```

---

## 7.4 Cross-Dataset Comparison

### 7.4.1 Impact Comparison

| Component | GeoLife Drop | DIY Drop | More Important For |
|-----------|--------------|----------|-------------------|
| Pointer Mechanism | **-24.01%** | -4.67% | GeoLife (6× more) |
| Temporal Embeddings | **-4.03%** | -0.62% | GeoLife (6× more) |
| User Embedding | **-2.31%** | -0.31% | GeoLife (7× more) |
| Position-from-End | **-2.08%** | +0.16% | GeoLife only |
| Adaptive Gate | **-1.88%** | -0.49% | GeoLife (4× more) |
| Position Bias | +0.06% | +0.08% | Neither |
| Transformer Depth | +0.26% | +0.08% | Neither |
| Generation Head | +0.43% | **+0.84%** | Redundant for both |

### 7.4.2 Insights from Comparison

```
┌──────────────────────────────────────────────────────────────────────┐
│                    CROSS-DATASET PATTERNS                             │
│                                                                       │
│  1. POINTER MECHANISM                                                 │
│     GeoLife: -24.01% (Catastrophic)                                   │
│     DIY:     -4.67%  (Significant)                                    │
│     → GeoLife users revisit locations more predictably                │
│                                                                       │
│  2. TEMPORAL FEATURES                                                 │
│     GeoLife: -4.03% (Important)                                       │
│     DIY:     -0.62% (Minor)                                           │
│     → GeoLife has stronger time patterns (research data)              │
│                                                                       │
│  3. GENERATION HEAD                                                   │
│     GeoLife: +0.43% (Better without)                                  │
│     DIY:     +0.84% (Better without)                                  │
│     → Copy mechanism alone is sufficient                              │
│                                                                       │
│  4. MODEL DEPTH                                                       │
│     GeoLife: +0.26% (Better with 1 layer)                             │
│     DIY:     +0.08% (Better with 1 layer)                             │
│     → Task doesn't need deep models                                   │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 7.5 Detailed Metrics Analysis

### 7.5.1 Acc@k Breakdown

**GeoLife**:
| Ablation | Acc@1 | Acc@3 (est.) | Acc@5 | Acc@10 | Gap (1→10) |
|----------|-------|--------------|-------|--------|------------|
| full | 51.43% | ~70% | 81.18% | 85.04% | +33.61% |
| no_pointer | 27.41% | ~45% | 54.14% | 58.65% | +31.24% |

**DIY**:
| Ablation | Acc@1 | Acc@3 (est.) | Acc@5 | Acc@10 | Gap (1→10) |
|----------|-------|--------------|-------|--------|------------|
| full | 56.57% | ~72% | 82.16% | 85.16% | +28.59% |
| no_pointer | 51.90% | ~66% | 75.59% | 78.27% | +26.37% |

**Insight**: The gap between Acc@1 and Acc@10 is ~30% for all variants, meaning top-10 always captures most correct answers.

### 7.5.2 MRR Analysis

MRR measures the average position of the correct answer.

| Dataset | Full Model MRR | No Pointer MRR | Drop |
|---------|----------------|----------------|------|
| GeoLife | 64.57% | 38.88% | -25.69% |
| DIY | 67.66% | 62.21% | -5.45% |

**Interpretation**: Without pointer, correct answers are ranked much lower.

### 7.5.3 NDCG Analysis

NDCG is more forgiving of lower-ranked correct predictions.

| Dataset | Full Model NDCG | No Pointer NDCG | Drop |
|---------|-----------------|-----------------|------|
| GeoLife | 69.48% | 43.43% | -26.05% |
| DIY | 71.88% | 66.05% | -5.83% |

### 7.5.4 Loss Analysis

| Dataset | Full Model Loss | No Pointer Loss | Increase |
|---------|-----------------|-----------------|----------|
| GeoLife | 2.6300 | 4.8392 | +84% |
| DIY | 2.8736 | 3.5887 | +25% |

**Insight**: Loss increases significantly without pointer, indicating harder optimization.

---

## 7.6 LaTeX Tables for Publication

### 7.6.1 GeoLife Table

```latex
\begin{table*}[htbp]
\centering
\small
\caption{Ablation Study Results on GEOLIFE Dataset. Baseline Acc@1: 51.43\%.}
\label{tab:ablation_geolife}
\begin{tabular}{l|ccccc|c}
\toprule
\textbf{Model Variant} & \textbf{Acc@1} & \textbf{Acc@5} & \textbf{Acc@10} & \textbf{MRR} & \textbf{NDCG} & \textbf{$\Delta$Acc@1} \\
\midrule
\textbf{Full Model (Baseline)} & \textbf{51.43} & 81.18 & 85.04 & 64.57 & 69.48 & — \\
w/o Generation Head & 51.86 & 82.41 & 85.38 & 64.95 & 69.58 & +0.43 \\
Single Transformer Layer & 51.68 & 81.70 & 85.01 & 64.96 & 69.81 & +0.26 \\
w/o Position Bias & 51.48 & 81.21 & 84.98 & 64.61 & 69.49 & +0.06 \\
w/o Adaptive Gate & 49.54 & 81.64 & 84.67 & 63.57 & 68.67 & -1.88 \\
w/o Position-from-End & 49.34 & 80.87 & 84.75 & 63.38 & 68.53 & -2.08 \\
w/o User Embedding & 49.11 & 81.10 & 84.12 & 63.27 & 68.33 & -2.31 \\
w/o Temporal Embeddings & 47.40 & 81.47 & 85.09 & 62.56 & 68.03 & -4.03 \\
w/o Pointer Mechanism & 27.41 & 54.14 & 58.65 & 38.88 & 43.43 & -24.01 \\
\bottomrule
\end{tabular}
\end{table*}
```

### 7.6.2 DIY Table

```latex
\begin{table*}[htbp]
\centering
\small
\caption{Ablation Study Results on DIY Dataset. Baseline Acc@1: 56.57\%.}
\label{tab:ablation_diy}
\begin{tabular}{l|ccccc|c}
\toprule
\textbf{Model Variant} & \textbf{Acc@1} & \textbf{Acc@5} & \textbf{Acc@10} & \textbf{MRR} & \textbf{NDCG} & \textbf{$\Delta$Acc@1} \\
\midrule
w/o Generation Head & 57.41 & 81.80 & 84.48 & 67.88 & 71.92 & +0.84 \\
w/o Position-from-End & 56.74 & 82.28 & 85.27 & 67.82 & 72.03 & +0.16 \\
w/o Position Bias & 56.65 & 82.14 & 85.16 & 67.70 & 71.90 & +0.08 \\
Single Transformer Layer & 56.65 & 81.90 & 85.04 & 67.58 & 71.78 & +0.08 \\
\textbf{Full Model (Baseline)} & \textbf{56.57} & 82.16 & 85.16 & 67.66 & 71.88 & — \\
w/o User Embedding & 56.27 & 81.98 & 84.89 & 67.31 & 71.57 & -0.31 \\
w/o Adaptive Gate & 56.08 & 81.90 & 85.28 & 67.22 & 71.56 & -0.49 \\
w/o Temporal Embeddings & 55.95 & 82.03 & 85.24 & 67.24 & 71.56 & -0.62 \\
w/o Pointer Mechanism & 51.90 & 75.59 & 78.27 & 62.21 & 66.05 & -4.67 \\
\bottomrule
\end{tabular}
\end{table*}
```

---

## 7.7 Data Files

### 7.7.1 CSV Export Location

```
scripts/ablation_study_v2/results/geolife/ablation_results.csv
scripts/ablation_study_v2/results/diy/ablation_results.csv
```

### 7.7.2 CSV Format

```csv
ablation,description,parameters,val_acc@1,val_acc@5,test_acc@1,test_acc@5,...,delta_acc1
full,Complete model (baseline),200000,49.25,78.52,51.43,81.18,...,0.00
no_pointer,w/o Pointer Mechanism,180000,26.85,52.34,27.41,54.14,...,-24.01
...
```

---

## 7.8 Raw Training Logs

### 7.8.1 Log Location

```
scripts/ablation_study_v2/logs/
├── geolife_full_baseline.log
├── geolife_no_pointer.log
├── geolife_no_generation.log
├── diy_full_baseline.log
├── diy_no_pointer.log
└── ...
```

### 7.8.2 Sample Log Output

```
2026-01-02 08:45:52,338 - INFO - ============================================================
2026-01-02 08:45:52,338 - INFO - FINAL RESULTS - FULL
2026-01-02 08:45:52,338 - INFO - ============================================================
2026-01-02 08:45:52,338 - INFO - Validation Acc@1: 49.25%
2026-01-02 08:45:52,338 - INFO - Test Acc@1: 51.43%
2026-01-02 08:45:52,339 - INFO - Test Acc@5: 81.18%
2026-01-02 08:45:52,339 - INFO - Test Acc@10: 85.04%
2026-01-02 08:45:52,339 - INFO - Test MRR: 64.57%
2026-01-02 08:45:52,339 - INFO - Test NDCG: 69.48%
2026-01-02 08:45:52,339 - INFO - Test F1: 0.47%
2026-01-02 08:45:52,339 - INFO - ============================================================
```

---

*Next: [08_analysis_discussion.md](08_analysis_discussion.md) - In-depth analysis and interpretation*
