# Results Tables and Data Export

## Complete Data Reference for Day-of-Week Analysis Experiment

**Document Version:** 1.0  
**Date:** January 2, 2026

This document provides all numerical results in various formats for easy reference, citation, and further analysis.

---

## Table of Contents

1. [DIY Dataset Complete Results](#1-diy-dataset-complete-results)
2. [GeoLife Dataset Complete Results](#2-geolife-dataset-complete-results)
3. [Comparative Summary Tables](#3-comparative-summary-tables)
4. [Statistical Test Results](#4-statistical-test-results)
5. [LaTeX-Ready Tables](#5-latex-ready-tables)
6. [CSV Export Format](#6-csv-export-format)
7. [Markdown Tables for Reports](#7-markdown-tables-for-reports)

---

## 1. DIY Dataset Complete Results

### 1.1 Individual Day Performance

| Day | Samples | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR | NDCG@10 | F1 | Loss |
|-----|---------|-------|-------|-------|--------|-----|---------|-----|------|
| Monday | 2,020 | 57.28 | 77.72 | 82.43 | 85.20 | 68.12 | 72.23 | 53.01 | 2.436 |
| Tuesday | 1,227 | 61.53 | 80.44 | 85.82 | 88.43 | 71.85 | 75.87 | 57.67 | 2.137 |
| Wednesday | 1,660 | 57.47 | 77.71 | 82.47 | 84.16 | 68.12 | 71.98 | 53.08 | 2.460 |
| Thursday | 1,721 | 55.08 | 75.48 | 80.48 | 84.08 | 66.21 | 70.50 | 50.34 | 2.571 |
| Friday | 1,950 | 56.21 | 77.64 | 82.82 | 86.36 | 67.76 | 72.24 | 52.25 | 2.353 |
| Saturday | 1,938 | 54.90 | 75.70 | 80.03 | 83.85 | 66.08 | 70.35 | 50.78 | 2.621 |
| Sunday | 1,852 | 55.29 | 77.00 | 82.40 | 84.94 | 66.90 | 71.26 | 50.68 | 2.519 |

### 1.2 Raw Count Data

| Day | Correct@1 | Correct@3 | Correct@5 | Correct@10 | RR Sum | NDCG Sum | Total |
|-----|-----------|-----------|-----------|------------|--------|----------|-------|
| Monday | 1,157 | 1,570 | 1,665 | 1,721 | 1,376.05 | 1,459.05 | 2,020 |
| Tuesday | 755 | 987 | 1,053 | 1,085 | 881.55 | 930.94 | 1,227 |
| Wednesday | 954 | 1,290 | 1,369 | 1,397 | 1,130.86 | 1,194.89 | 1,660 |
| Thursday | 948 | 1,299 | 1,385 | 1,447 | 1,139.52 | 1,213.31 | 1,721 |
| Friday | 1,096 | 1,514 | 1,615 | 1,684 | 1,321.40 | 1,408.74 | 1,950 |
| Saturday | 1,064 | 1,467 | 1,551 | 1,625 | 1,280.58 | 1,363.51 | 1,938 |
| Sunday | 1,024 | 1,426 | 1,526 | 1,573 | 1,239.03 | 1,319.42 | 1,852 |
| **Total** | **6,998** | **9,553** | **10,164** | **10,532** | **8,368.99** | **8,889.86** | **12,368** |

### 1.3 Aggregated Statistics

| Category | Samples | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG | F1 | Loss |
|----------|---------|-------|-------|--------|-----|------|-----|------|
| Weekday Average | 8,578 | 57.24 | 82.62 | 85.50 | 68.19 | 72.36 | 52.98 | 2.406 |
| Weekend Average | 3,790 | 55.09 | 81.19 | 84.38 | 66.48 | 70.80 | 50.73 | 2.571 |
| Overall | 12,368 | 56.58 | 82.18 | 85.16 | 67.67 | 71.88 | 51.91 | 2.446 |

### 1.4 Day Rankings (DIY)

**By Acc@1:**
1. Tuesday: 61.53%
2. Wednesday: 57.47%
3. Monday: 57.28%
4. Friday: 56.21%
5. Sunday: 55.29%
6. Thursday: 55.08%
7. Saturday: 54.90%

**By Loss (lower is better):**
1. Tuesday: 2.137
2. Friday: 2.353
3. Monday: 2.436
4. Wednesday: 2.460
5. Sunday: 2.519
6. Thursday: 2.571
7. Saturday: 2.621

---

## 2. GeoLife Dataset Complete Results

### 2.1 Individual Day Performance

| Day | Samples | Acc@1 | Acc@3 | Acc@5 | Acc@10 | MRR | NDCG@10 | F1 | Loss |
|-----|---------|-------|-------|-------|--------|-----|---------|-----|------|
| Monday | 538 | 53.53 | 83.64 | 89.78 | 91.82 | 69.28 | 74.86 | 50.62 | 2.143 |
| Tuesday | 528 | 53.22 | 79.55 | 83.33 | 86.55 | 67.18 | 71.84 | 48.68 | 2.292 |
| Wednesday | 516 | 59.88 | 81.98 | 85.08 | 88.76 | 71.15 | 75.38 | 55.94 | 2.173 |
| Thursday | 537 | 56.24 | 78.96 | 84.92 | 87.90 | 68.41 | 73.15 | 52.01 | 2.360 |
| Friday | 514 | 53.50 | 76.85 | 81.91 | 86.38 | 66.09 | 70.96 | 49.48 | 2.684 |
| Saturday | 463 | 37.58 | 59.61 | 67.39 | 73.43 | 50.12 | 55.49 | 35.14 | 3.945 |
| Sunday | 406 | 42.12 | 68.97 | 71.92 | 77.09 | 55.86 | 60.83 | 36.57 | 3.356 |

### 2.2 Raw Count Data

| Day | Correct@1 | Correct@3 | Correct@5 | Correct@10 | RR Sum | NDCG Sum | Total |
|-----|-----------|-----------|-----------|------------|--------|----------|-------|
| Monday | 288 | 450 | 483 | 494 | 372.75 | 402.75 | 538 |
| Tuesday | 281 | 420 | 440 | 457 | 354.70 | 379.32 | 528 |
| Wednesday | 309 | 423 | 439 | 458 | 367.12 | 388.96 | 516 |
| Thursday | 302 | 424 | 456 | 472 | 367.37 | 392.82 | 537 |
| Friday | 275 | 395 | 421 | 444 | 339.68 | 364.74 | 514 |
| Saturday | 174 | 276 | 312 | 340 | 232.04 | 256.94 | 463 |
| Sunday | 171 | 280 | 292 | 313 | 226.79 | 246.98 | 406 |
| **Total** | **1,800** | **2,668** | **2,843** | **2,978** | **2,260.45** | **2,432.51** | **3,502** |

### 2.3 Aggregated Statistics

| Category | Samples | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG | F1 | Loss |
|----------|---------|-------|-------|--------|-----|------|-----|------|
| Weekday Average | 2,633 | 55.26 | 85.04 | 88.30 | 68.42 | 73.25 | 51.34 | 2.329 |
| Weekend Average | 869 | 39.70 | 69.51 | 75.14 | 52.80 | 57.99 | 35.81 | 3.670 |
| Overall | 3,502 | 51.40 | 81.18 | 85.04 | 64.55 | 69.46 | 46.97 | 2.630 |

### 2.4 Day Rankings (GeoLife)

**By Acc@1:**
1. Wednesday: 59.88%
2. Thursday: 56.24%
3. Monday: 53.53%
4. Friday: 53.50%
5. Tuesday: 53.22%
6. Sunday: 42.12%
7. Saturday: 37.58%

**By Loss (lower is better):**
1. Monday: 2.143
2. Wednesday: 2.173
3. Tuesday: 2.292
4. Thursday: 2.360
5. Friday: 2.684
6. Sunday: 3.356
7. Saturday: 3.945

---

## 3. Comparative Summary Tables

### 3.1 Weekend Effect Comparison

| Metric | DIY Weekday | DIY Weekend | DIY Drop | GeoLife Weekday | GeoLife Weekend | GeoLife Drop |
|--------|-------------|-------------|----------|-----------------|-----------------|--------------|
| Acc@1 | 57.24% | 55.09% | 2.15 pp | 55.26% | 39.70% | **15.56 pp** |
| Acc@5 | 82.62% | 81.19% | 1.43 pp | 85.04% | 69.51% | **15.53 pp** |
| Acc@10 | 85.50% | 84.38% | 1.12 pp | 88.30% | 75.14% | **13.16 pp** |
| MRR | 68.19% | 66.48% | 1.71 pp | 68.42% | 52.80% | **15.62 pp** |
| NDCG | 72.36% | 70.80% | 1.56 pp | 73.25% | 57.99% | **15.26 pp** |
| F1 | 52.98% | 50.73% | 2.25 pp | 51.34% | 35.81% | **15.53 pp** |
| Loss | 2.406 | 2.571 | +0.165 | 2.329 | 3.670 | **+1.341** |

*(pp = percentage points)*

### 3.2 Best and Worst Days Comparison

| Aspect | DIY | GeoLife |
|--------|-----|---------|
| Best Day | Tuesday | Wednesday |
| Best Acc@1 | 61.53% | 59.88% |
| Worst Day | Saturday | Saturday |
| Worst Acc@1 | 54.90% | 37.58% |
| Best-Worst Gap | 6.63 pp | **22.30 pp** |
| Relative Gap | 12.1% | **59.3%** |

### 3.3 Sample Distribution Comparison

| Day | DIY Samples | DIY % | GeoLife Samples | GeoLife % |
|-----|-------------|-------|-----------------|-----------|
| Monday | 2,020 | 16.33% | 538 | 15.36% |
| Tuesday | 1,227 | 9.92% | 528 | 15.08% |
| Wednesday | 1,660 | 13.42% | 516 | 14.73% |
| Thursday | 1,721 | 13.91% | 537 | 15.33% |
| Friday | 1,950 | 15.77% | 514 | 14.68% |
| Saturday | 1,938 | 15.67% | 463 | 13.22% |
| Sunday | 1,852 | 14.97% | 406 | 11.59% |
| **Total** | **12,368** | 100% | **3,502** | 100% |
| **Weekend %** | 3,790 | **30.64%** | 869 | **24.81%** |

---

## 4. Statistical Test Results

### 4.1 DIY Dataset T-Test

| Parameter | Value |
|-----------|-------|
| Test Type | Independent Samples t-test (Welch) |
| Comparison | Weekday Acc@1 vs Weekend Acc@1 |
| n₁ (weekday) | 5 |
| n₂ (weekend) | 2 |
| Mean₁ (weekday) | 57.51% |
| Mean₂ (weekend) | 55.10% |
| Difference | 2.42 pp |
| t-statistic | 1.321 |
| p-value | **0.2436** |
| Significant at α=0.05 | **No** |
| Significant at α=0.01 | **No** |

### 4.2 GeoLife Dataset T-Test

| Parameter | Value |
|-----------|-------|
| Test Type | Independent Samples t-test (Welch) |
| Comparison | Weekday Acc@1 vs Weekend Acc@1 |
| n₁ (weekday) | 5 |
| n₂ (weekend) | 2 |
| Mean₁ (weekday) | 55.28% |
| Mean₂ (weekend) | 39.85% |
| Difference | 15.43 pp |
| t-statistic | 6.297 |
| p-value | **0.0015** |
| Significant at α=0.05 | **Yes** |
| Significant at α=0.01 | **Yes** |

---

## 5. LaTeX-Ready Tables

### 5.1 DIY Results Table

```latex
\begin{table}[htbp]
\centering
\caption{Day-of-Week Performance Metrics - DIY Dataset}
\label{tab:diy_results}
\begin{tabular}{lrrrrrrr}
\toprule
Day & Samples & Acc@1 & Acc@5 & Acc@10 & MRR & NDCG & F1 \\
\midrule
Monday & 2,020 & 57.28 & 82.43 & 85.20 & 68.12 & 72.23 & 0.530 \\
Tuesday & 1,227 & \textbf{61.53} & \textbf{85.82} & \textbf{88.43} & \textbf{71.85} & \textbf{75.87} & \textbf{0.577} \\
Wednesday & 1,660 & 57.47 & 82.47 & 84.16 & 68.12 & 71.98 & 0.531 \\
Thursday & 1,721 & 55.08 & 80.48 & 84.08 & 66.21 & 70.50 & 0.503 \\
Friday & 1,950 & 56.21 & 82.82 & 86.36 & 67.76 & 72.24 & 0.523 \\
\rowcolor{gray!15} Saturday & 1,938 & 54.90 & 80.03 & 83.85 & 66.08 & 70.35 & 0.508 \\
\rowcolor{gray!15} Sunday & 1,852 & 55.29 & 82.40 & 84.94 & 66.90 & 71.26 & 0.507 \\
\midrule
\textit{Weekday Avg} & 8,578 & 57.24 & 82.62 & 85.50 & 68.19 & 72.36 & 0.530 \\
\textit{Weekend Avg} & 3,790 & 55.09 & 81.19 & 84.38 & 66.48 & 70.80 & 0.507 \\
\textit{Overall} & 12,368 & 56.58 & 82.18 & 85.16 & 67.67 & 71.88 & 0.519 \\
\bottomrule
\end{tabular}
\end{table}
```

### 5.2 GeoLife Results Table

```latex
\begin{table}[htbp]
\centering
\caption{Day-of-Week Performance Metrics - GeoLife Dataset}
\label{tab:geolife_results}
\begin{tabular}{lrrrrrrr}
\toprule
Day & Samples & Acc@1 & Acc@5 & Acc@10 & MRR & NDCG & F1 \\
\midrule
Monday & 538 & 53.53 & \textbf{89.78} & \textbf{91.82} & 69.28 & 74.86 & 0.506 \\
Tuesday & 528 & 53.22 & 83.33 & 86.55 & 67.18 & 71.84 & 0.487 \\
Wednesday & 516 & \textbf{59.88} & 85.08 & 88.76 & \textbf{71.15} & \textbf{75.38} & \textbf{0.559} \\
Thursday & 537 & 56.24 & 84.92 & 87.90 & 68.41 & 73.15 & 0.520 \\
Friday & 514 & 53.50 & 81.91 & 86.38 & 66.09 & 70.96 & 0.495 \\
\rowcolor{gray!15} Saturday & 463 & \underline{37.58} & \underline{67.39} & \underline{73.43} & \underline{50.12} & \underline{55.49} & \underline{0.351} \\
\rowcolor{gray!15} Sunday & 406 & 42.12 & 71.92 & 77.09 & 55.86 & 60.83 & 0.366 \\
\midrule
\textit{Weekday Avg} & 2,633 & 55.26 & 85.04 & 88.30 & 68.42 & 73.25 & 0.513 \\
\textit{Weekend Avg} & 869 & 39.70 & 69.51 & 75.14 & 52.80 & 57.99 & 0.358 \\
\textit{Overall} & 3,502 & 51.40 & 81.18 & 85.04 & 64.55 & 69.46 & 0.470 \\
\bottomrule
\multicolumn{8}{l}{\footnotesize \textbf{Bold}: best value; \underline{Underline}: worst value; Shaded: weekend rows}
\end{tabular}
\end{table}
```

### 5.3 Statistical Comparison Table

```latex
\begin{table}[htbp]
\centering
\caption{Weekday vs Weekend Statistical Comparison}
\label{tab:statistical}
\begin{tabular}{lcccc}
\toprule
Dataset & Weekday Mean & Weekend Mean & Difference & p-value \\
\midrule
DIY & 57.51\% & 55.10\% & 2.42 pp & 0.2436 \\
GeoLife & 55.28\% & 39.85\% & 15.43 pp$^{**}$ & 0.0015 \\
\bottomrule
\multicolumn{5}{l}{\footnotesize $^{**}$ Significant at $p < 0.01$}
\end{tabular}
\end{table}
```

---

## 6. CSV Export Format

### 6.1 Full Results CSV

The file `days_analysis_summary.csv` contains:

```csv
Dataset,Day,Type,Samples,Acc@1,Acc@5,Acc@10,MRR,NDCG,F1,Loss
DIY,Monday,Weekday,2020,57.2772,82.4257,85.1980,68.1211,72.2284,0.5301,2.4360
DIY,Tuesday,Weekday,1227,61.5322,85.8191,88.4271,71.8461,75.8701,0.5767,2.1372
DIY,Wednesday,Weekday,1660,57.4699,82.4699,84.1566,68.1243,71.9823,0.5308,2.4600
DIY,Thursday,Weekday,1721,55.0843,80.4765,84.0790,66.2128,70.4962,0.5034,2.5706
DIY,Friday,Weekday,1950,56.2051,82.8205,86.3590,67.7642,72.2431,0.5225,2.3526
DIY,Saturday,Weekend,1938,54.9020,80.0310,83.8493,66.0772,70.3466,0.5078,2.6214
DIY,Sunday,Weekend,1852,55.2916,82.3974,84.9352,66.9021,71.2644,0.5068,2.5191
DIY,Weekday Average,Summary,8578,57.2394,82.6183,85.4978,68.1906,72.3575,0.5298,2.4060
DIY,Weekend Average,Summary,3790,55.0923,81.1873,84.3799,66.4803,70.7951,0.5073,2.5714
DIY,Overall,Summary,12368,56.5815,82.1798,85.1552,67.6665,71.8787,0.5191,2.4463
GeoLife,Monday,Weekday,538,53.5316,89.7770,91.8216,69.2840,74.8613,0.5062,2.1433
GeoLife,Tuesday,Weekday,528,53.2197,83.3333,86.5530,67.1785,71.8434,0.4868,2.2925
GeoLife,Wednesday,Weekday,516,59.8837,85.0775,88.7597,71.1470,75.3784,0.5594,2.1726
GeoLife,Thursday,Weekday,537,56.2384,84.9162,87.8957,68.4115,73.1549,0.5201,2.3603
GeoLife,Friday,Weekday,514,53.5019,81.9066,86.3813,66.0856,70.9592,0.4948,2.6844
GeoLife,Saturday,Weekend,463,37.5810,67.3866,73.4341,50.1160,55.4934,0.3514,3.9451
GeoLife,Sunday,Weekend,406,42.1182,71.9212,77.0936,55.8603,60.8344,0.3657,3.3563
GeoLife,Weekday Average,Summary,2633,55.2602,85.0361,88.3023,68.4246,73.2477,0.5134,2.3288
GeoLife,Weekend Average,Summary,869,39.7008,69.5052,75.1438,52.7997,57.9887,0.3581,3.6700
GeoLife,Overall,Summary,3502,51.3992,81.1822,85.0371,64.5474,69.4613,0.4697,2.6301
```

### 6.2 Loading in Python

```python
import pandas as pd

# Load data
df = pd.read_csv('figures/days_analysis_summary.csv')

# Filter to specific dataset
diy = df[df['Dataset'] == 'DIY']
geolife = df[df['Dataset'] == 'GeoLife']

# Get individual days only
diy_days = diy[diy['Type'].isin(['Weekday', 'Weekend'])]

# Calculate statistics
print(f"DIY Acc@1 Mean: {diy_days['Acc@1'].mean():.2f}%")
print(f"DIY Acc@1 Std: {diy_days['Acc@1'].std():.2f}%")
```

### 6.3 Loading in R

```r
library(readr)
library(dplyr)

# Load data
df <- read_csv("figures/days_analysis_summary.csv")

# Filter by dataset
diy <- df %>% filter(Dataset == "DIY")
geolife <- df %>% filter(Dataset == "GeoLife")

# Calculate weekend effect
diy %>%
  filter(Type %in% c("Weekday", "Weekend")) %>%
  group_by(Type) %>%
  summarize(mean_acc1 = mean(`Acc@1`))
```

---

## 7. Markdown Tables for Reports

### 7.1 Executive Summary Table

| Finding | DIY Dataset | GeoLife Dataset |
|---------|-------------|-----------------|
| Total Samples | 12,368 | 3,502 |
| Overall Acc@1 | 56.58% | 51.40% |
| Best Day | Tuesday (61.53%) | Wednesday (59.88%) |
| Worst Day | Saturday (54.90%) | Saturday (37.58%) |
| Weekend Drop | 2.15 pp | **15.56 pp** |
| Statistically Significant? | No (p=0.24) | **Yes (p=0.001)** |

### 7.2 Detailed Performance Table

#### DIY Dataset

| Day | Type | Samples | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|-----|------|---------|-------|-------|--------|-----|------|
| Monday | Weekday | 2,020 | 57.28% | 82.43% | 85.20% | 68.12% | 72.23% |
| Tuesday | Weekday | 1,227 | **61.53%** | **85.82%** | **88.43%** | **71.85%** | **75.87%** |
| Wednesday | Weekday | 1,660 | 57.47% | 82.47% | 84.16% | 68.12% | 71.98% |
| Thursday | Weekday | 1,721 | 55.08% | 80.48% | 84.08% | 66.21% | 70.50% |
| Friday | Weekday | 1,950 | 56.21% | 82.82% | 86.36% | 67.76% | 72.24% |
| Saturday | Weekend | 1,938 | *54.90%* | *80.03%* | *83.85%* | *66.08%* | *70.35%* |
| Sunday | Weekend | 1,852 | 55.29% | 82.40% | 84.94% | 66.90% | 71.26% |

*Bold = best, Italic = worst*

#### GeoLife Dataset

| Day | Type | Samples | Acc@1 | Acc@5 | Acc@10 | MRR | NDCG |
|-----|------|---------|-------|-------|--------|-----|------|
| Monday | Weekday | 538 | 53.53% | **89.78%** | **91.82%** | 69.28% | 74.86% |
| Tuesday | Weekday | 528 | 53.22% | 83.33% | 86.55% | 67.18% | 71.84% |
| Wednesday | Weekday | 516 | **59.88%** | 85.08% | 88.76% | **71.15%** | **75.38%** |
| Thursday | Weekday | 537 | 56.24% | 84.92% | 87.90% | 68.41% | 73.15% |
| Friday | Weekday | 514 | 53.50% | 81.91% | 86.38% | 66.09% | 70.96% |
| Saturday | Weekend | 463 | *37.58%* | *67.39%* | *73.43%* | *50.12%* | *55.49%* |
| Sunday | Weekend | 406 | 42.12% | 71.92% | 77.09% | 55.86% | 60.83% |

*Bold = best, Italic = worst*

---

## Appendix: Precision and Rounding

All percentages in this document are:
- Computed as: `(count / total) × 100`
- Displayed with 2 decimal places
- Internal precision: float64 (15-17 significant digits)

For exact values, refer to the JSON files:
- `results/diy_days_results.json`
- `results/geolife_days_results.json`

---

*End of Results Tables and Data Export*
