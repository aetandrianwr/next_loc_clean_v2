# Day-of-Week Impact Analysis on Next Location Prediction

## Experiment Overview

**Title:** Temporal Periodicity in Human Mobility: Day-of-Week Impact on Location Prediction Performance

**Research Question:** Does the day of the week significantly affect next location prediction accuracy, and specifically, does weekend behavior lead to reduced prediction performance due to less routine human mobility patterns?

**Hypothesis:** Weekend predictions (Saturday and Sunday) will show lower accuracy compared to weekdays (Monday-Friday) because human mobility on weekends tends to be more exploratory and less routine-based, making next location prediction inherently more difficult.

---

## Scientific Background

### Human Mobility Patterns

Human mobility exhibits strong temporal periodicity driven by social and economic factors:

1. **Weekday Patterns**: Dominated by routine activities such as:
   - Commuting to work
   - Regular lunch locations
   - School drop-off/pick-up
   - Evening activities at fixed venues

2. **Weekend Patterns**: Characterized by:
   - Exploratory behavior (shopping, tourism, socializing)
   - Irregular sleep schedules affecting departure times
   - Novel location visits
   - Less predictable sequences

### Implications for Prediction Models

Location prediction models inherently learn from historical patterns. When human behavior deviates from learned routines, prediction difficulty increases. This experiment quantifies this phenomenon across different datasets and user populations.

---

## Methodology

### Experimental Design

**Type:** Evaluation-only experiment (no retraining required)

**Datasets:**
- **DIY Dataset**: Urban mobility data with 12,368 test samples across 693 users and 7,038 unique locations
- **Geolife Dataset**: GPS trajectory data with 3,502 test samples across 46 users and 1,187 unique locations

**Model:** PointerGeneratorTransformer (Pre-trained on prev7 historical context)
- DIY model: 1,081,554 parameters
- Geolife model: 443,404 parameters

**Pre-trained Checkpoints:**
- DIY: `experiments/diy_pointer_v45_20260101_155348/checkpoints/best.pt`
- Geolife: `experiments/geolife_pointer_v45_20260101_151038/checkpoints/best.pt`

### Day Classification

Samples were categorized by the day of week of the **target prediction** (Y):
- **Weekdays**: Monday (0), Tuesday (1), Wednesday (2), Thursday (3), Friday (4)
- **Weekend**: Saturday (5), Sunday (6)

The target day was computed from the data as:
```python
y_weekday = (weekday_X[-1] + diff[-1]) % 7
```

### Evaluation Metrics

All standard metrics from the evaluation framework were computed:

| Metric | Description |
|--------|-------------|
| **Acc@k** | Top-k accuracy (k=1,5,10) - % of predictions where correct location is in top k |
| **MRR** | Mean Reciprocal Rank - average of 1/rank of correct location |
| **NDCG** | Normalized Discounted Cumulative Gain - ranking quality measure |
| **F1** | Weighted F1 score for top-1 predictions |
| **Loss** | Cross-entropy loss |

### Statistical Analysis

**Test:** Independent samples t-test comparing weekday vs weekend Acc@1 values
- α = 0.05 for significance
- α = 0.01 for high significance

---

## Results

### DIY Dataset Results

#### Performance by Day of Week

| Day       | Type    | Samples | Acc@1 (%) | Acc@5 (%) | Acc@10 (%) | MRR (%) | NDCG (%) | F1 Score | Loss |
|-----------|---------|---------|-----------|-----------|------------|---------|----------|----------|------|
| Monday    | Weekday | 2,020   | 57.28     | 82.43     | 85.20      | 68.12   | 72.23    | 0.53     | 2.44 |
| Tuesday   | Weekday | 1,227   | **61.53** | **85.82** | **88.43**  | **71.85**| **75.87**| **0.58** | **2.14**|
| Wednesday | Weekday | 1,660   | 57.47     | 82.47     | 84.16      | 68.12   | 71.98    | 0.53     | 2.46 |
| Thursday  | Weekday | 1,721   | 55.08     | 80.48     | 84.08      | 66.21   | 70.50    | 0.50     | 2.57 |
| Friday    | Weekday | 1,950   | 56.21     | 82.82     | 86.36      | 67.76   | 72.24    | 0.52     | 2.35 |
| Saturday  | Weekend | 1,938   | 54.90     | 80.03     | 83.85      | 66.08   | 70.35    | 0.51     | 2.62 |
| Sunday    | Weekend | 1,852   | 55.29     | 82.40     | 84.94      | 66.90   | 71.26    | 0.51     | 2.52 |

#### Aggregate Comparison

| Category     | Samples | Acc@1 (%) | Acc@5 (%) | Acc@10 (%) | MRR (%) | NDCG (%) |
|--------------|---------|-----------|-----------|------------|---------|----------|
| **Weekday**  | 8,578   | 57.24     | 82.62     | 85.50      | 68.19   | 72.36    |
| **Weekend**  | 3,790   | 55.09     | 81.19     | 84.38      | 66.48   | 70.80    |
| **Δ (W-We)** | -       | +2.15     | +1.43     | +1.12      | +1.71   | +1.56    |

#### Statistical Test (DIY)
- **Weekday Mean Acc@1:** 57.51%
- **Weekend Mean Acc@1:** 55.10%
- **Difference:** 2.42%
- **t-statistic:** 1.32
- **p-value:** 0.244
- **Result:** Not statistically significant (p > 0.05)

---

### Geolife Dataset Results

#### Performance by Day of Week

| Day       | Type    | Samples | Acc@1 (%) | Acc@5 (%) | Acc@10 (%) | MRR (%) | NDCG (%) | F1 Score | Loss |
|-----------|---------|---------|-----------|-----------|------------|---------|----------|----------|------|
| Monday    | Weekday | 538     | 53.53     | **89.78** | **91.82**  | 69.28   | 74.86    | 0.51     | 2.14 |
| Tuesday   | Weekday | 528     | 53.22     | 83.33     | 86.55      | 67.18   | 71.84    | 0.49     | 2.29 |
| Wednesday | Weekday | 516     | **59.88** | 85.08     | 88.76      | **71.15**| **75.38**| **0.56** | 2.17 |
| Thursday  | Weekday | 537     | 56.24     | 84.92     | 87.90      | 68.41   | 73.15    | 0.52     | 2.36 |
| Friday    | Weekday | 514     | 53.50     | 81.91     | 86.38      | 66.09   | 70.96    | 0.49     | 2.68 |
| Saturday  | Weekend | 463     | 37.58     | 67.39     | 73.43      | 50.12   | 55.49    | 0.35     | **3.95** |
| Sunday    | Weekend | 406     | 42.12     | 71.92     | 77.09      | 55.86   | 60.83    | 0.37     | 3.36 |

#### Aggregate Comparison

| Category     | Samples | Acc@1 (%) | Acc@5 (%) | Acc@10 (%) | MRR (%) | NDCG (%) |
|--------------|---------|-----------|-----------|------------|---------|----------|
| **Weekday**  | 2,633   | 55.26     | 85.04     | 88.30      | 68.42   | 73.25    |
| **Weekend**  | 869     | 39.70     | 69.51     | 75.14      | 52.80   | 57.99    |
| **Δ (W-We)** | -       | **+15.56**| +15.53    | +13.16     | +15.62  | +15.26   |

#### Statistical Test (Geolife)
- **Weekday Mean Acc@1:** 55.28%
- **Weekend Mean Acc@1:** 39.85%
- **Difference:** 15.43%
- **t-statistic:** 6.30
- **p-value:** 0.0015
- **Result:** **Highly significant (p < 0.01)** ✓✓

---

## Key Findings

### 1. Weekend Performance Drop is Dataset-Dependent

The magnitude of weekend performance degradation varies significantly:

| Dataset | Weekday Acc@1 | Weekend Acc@1 | Drop (%) | Relative Drop |
|---------|---------------|---------------|----------|---------------|
| DIY     | 57.24%        | 55.09%        | 2.15%    | 3.8%          |
| Geolife | 55.26%        | 39.70%        | 15.56%   | **28.2%**     |

### 2. Geolife Shows Dramatic Weekend Effect

The Geolife dataset exhibits a substantial and statistically significant weekend performance drop:
- **Top-1 accuracy drops by 15.56 percentage points** on weekends
- **Saturday shows the worst performance** (37.58% Acc@1)
- All metrics consistently degrade on weekends
- The effect is highly significant (p = 0.0015)

### 3. DIY Dataset Shows Mild Weekend Effect

The DIY dataset shows a smaller, non-significant weekend effect:
- Top-1 accuracy drops by only 2.15 percentage points
- The drop is consistent but modest across all metrics
- Not statistically significant (p = 0.244)

### 4. Day-by-Day Patterns

**Best Performance Days:**
- DIY: Tuesday (61.53% Acc@1)
- Geolife: Wednesday (59.88% Acc@1)

**Worst Performance Days:**
- DIY: Saturday (54.90% Acc@1)
- Geolife: Saturday (37.58% Acc@1)

---

## Analysis and Discussion

### Why Geolife Shows Stronger Weekend Effect

1. **User Demographics:** Geolife contains mobility data from Microsoft Research employees and Beijing residents. These users may have highly structured work schedules during weekdays.

2. **Dataset Size:** With only 46 users, individual behavioral variations have more impact on aggregate statistics.

3. **Geographic Context:** Beijing's urban structure may enforce more routine-based weekday behavior.

4. **User Activity Profile:** Research employees may exhibit particularly exploratory behavior on weekends (tourism, city exploration).

### Why DIY Shows Weaker Weekend Effect

1. **Diverse User Base:** 693 users provide more behavioral diversity, smoothing out individual variations.

2. **Mixed Activity Types:** The DIY dataset may capture users with less weekday/weekend differentiation in their mobility patterns.

3. **Different Location Granularity:** More unique locations (7,038) may dilute the weekend effect through increased prediction difficulty overall.

### Implications for Model Development

1. **Day-aware Models:** Incorporating day-of-week as an explicit feature could improve predictions, especially for weekend forecasting.

2. **Adaptive Prediction:** Models could adjust confidence thresholds based on the predicted day.

3. **Weekend-specific Training:** Separate models or transfer learning approaches for weekend prediction may improve performance.

4. **User Segmentation:** Identifying users with high weekday/weekend behavioral variance could enable personalized strategies.

---

## Conclusion

This experiment provides strong evidence that **weekend human mobility is harder to predict** than weekday mobility, supporting the hypothesis that reduced routine behavior on weekends increases prediction difficulty.

**Key takeaways:**

1. The weekend effect is **statistically significant in the Geolife dataset** (p < 0.01) with a 15.56% Acc@1 drop.

2. The DIY dataset shows a **consistent but smaller** weekend effect (2.15% Acc@1 drop), not reaching statistical significance.

3. **Saturday consistently shows the lowest prediction accuracy** across both datasets.

4. The effect magnitude varies by dataset characteristics, suggesting **user demographics and geographic context** play important roles.

5. Future work should explore **day-aware model architectures** and **weekend-specific training strategies** to address this temporal variability.

---

## Files and Artifacts

### Scripts
- `scripts/experiment_days_analysis/run_days_analysis.py` - Main experiment script
- `scripts/experiment_days_analysis/generate_visualizations.py` - Visualization generation

### Results
- `scripts/experiment_days_analysis/results/diy_days_results.json`
- `scripts/experiment_days_analysis/results/geolife_days_results.json`

### Figures
- `scripts/experiment_days_analysis/figures/combined_comparison.png` - Cross-dataset comparison
- `scripts/experiment_days_analysis/figures/{dataset}_accuracy_by_day.png` - Daily accuracy charts
- `scripts/experiment_days_analysis/figures/{dataset}_weekday_weekend_comparison.png` - Weekday vs weekend bars
- `scripts/experiment_days_analysis/figures/{dataset}_metrics_heatmap.png` - Full metrics heatmap
- `scripts/experiment_days_analysis/figures/{dataset}_performance_trend.png` - Weekly trend lines
- `scripts/experiment_days_analysis/figures/{dataset}_sample_distribution.png` - Sample counts

### LaTeX Tables
- `scripts/experiment_days_analysis/figures/diy_table.tex`
- `scripts/experiment_days_analysis/figures/geolife_table.tex`

### CSV Summary
- `scripts/experiment_days_analysis/figures/days_analysis_summary.csv`

---

## Reproducibility

To reproduce this experiment:

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Run analysis
cd /data/next_loc_clean_v2
python scripts/experiment_days_analysis/run_days_analysis.py --dataset both --seed 42

# Generate visualizations
python scripts/experiment_days_analysis/generate_visualizations.py
```

**Requirements:**
- Pre-trained model checkpoints must exist
- Test data files must be in the expected locations
- Python environment with PyTorch, NumPy, Pandas, Matplotlib, Seaborn, SciPy

---

## Citation

If using this experiment in academic work, please cite the model architecture and dataset sources appropriately.

---

*Experiment conducted: January 2, 2026*
*Model: PointerGeneratorTransformer*
*Random Seed: 42*
