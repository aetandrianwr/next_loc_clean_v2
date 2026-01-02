# 10. Visualization Guide

## Complete Guide to Experiment Visualizations

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Visualization Documentation |
| **Audience** | Researchers, Data Analysts, Presentation Preparers |
| **Reading Time** | 25-30 minutes |
| **Prerequisites** | Understanding of results (see 09_results_and_analysis.md) |

---

## 1. Visualization Overview

### 1.1 Complete List of Generated Figures

| Figure | File Names | Purpose |
|--------|------------|---------|
| Performance Comparison | `performance_comparison.{pdf,png,svg}` | 6-panel metric comparison |
| Accuracy Heatmap | `accuracy_heatmap.{pdf,png}` | Matrix view of metrics |
| Loss Curve | `loss_curve.{pdf,png}` | Cross-entropy loss trends |
| Radar Comparison | `radar_comparison.{pdf,png}` | Prev1 vs prev7 shape comparison |
| Improvement Comparison | `improvement_comparison.{pdf,png}` | Relative improvement bars |
| Sequence Length Distribution | `sequence_length_distribution.{pdf,png}` | Avg seq length bars |
| Samples vs Performance | `samples_vs_performance.{pdf,png}` | Sample count scatter |
| Combined Figure | `combined_figure.{pdf,png,svg}` | Publication-ready composite |

### 1.2 Visual Style Guide

All figures follow a classic scientific journal style:

```
Style Element          Specification
─────────────────────────────────────────────────
Background:           White
Axis box:             Black, all 4 sides
Tick direction:       Inside
Grid:                 None (clean look)
DIY color:            Blue (#1f77b4)
GeoLife color:        Red (#d62728)
DIY marker:           Circle (○), white fill
GeoLife marker:       Square (□), white fill
Font:                 Serif (Times-like)
```

---

## 2. Performance Comparison Plot

### 2.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `performance_comparison.{pdf,png,svg}` |
| **Dimensions** | 12 × 8 inches |
| **Resolution** | 300 DPI |
| **Panels** | 2 rows × 3 columns = 6 |

### 2.2 Visual Description

```
┌─────────────────┬─────────────────┬─────────────────┐
│   Accuracy@1    │   Accuracy@5    │   Accuracy@10   │
│      (%)        │      (%)        │       (%)       │
│                 │                 │                 │
│   ○─○─○─○─○─○─○ │   ○─○─○─○─○─○─○ │   ○─○─○─○─○─○─○ │
│   □─□─□─□─□─□─□ │   □─□─□─□─□─□─□ │   □─□─□─□─□─□─□ │
│                 │                 │                 │
├─────────────────┼─────────────────┼─────────────────┤
│      MRR        │    NDCG@10      │    F1 Score     │
│      (%)        │      (%)        │       (%)       │
│                 │                 │                 │
│   ○─○─○─○─○─○─○ │   ○─○─○─○─○─○─○ │   ○─○─○─○─○─○─○ │
│   □─□─□─□─□─□─□ │   □─□─□─□─□─□─□ │   □─□─□─□─□─□─□ │
│                 │                 │                 │
└─────────────────┴─────────────────┴─────────────────┘

Legend: ○ DIY (blue)  □ GeoLife (red)
X-axis: t (days) from 1 to 7
```

### 2.3 How to Read This Plot

**X-Axis**: `t (days)` - Number of previous days of history (1-7)

**Y-Axis**: Metric value as percentage

**Lines**:
- Blue circles connected by blue line: DIY dataset
- Red squares connected by red line: GeoLife dataset

**Reading a single panel**:
1. Find the leftmost point (prev1) - this is the baseline
2. Follow the line to the right - observe the slope
3. Steeper slope = larger improvement
4. Flattening slope = diminishing returns

### 2.4 Key Observations from This Plot

**Panel 1 (Accuracy@1)**:
- DIY starts at 50.0%, ends at 56.6% (steep rise then plateau)
- GeoLife starts at 47.8%, ends at 51.4% (gentler rise)
- Gap between datasets increases with more days

**Panel 2 (Accuracy@5)**:
- Both datasets show stronger improvement than Acc@1
- Lines are more parallel (similar relative improvement)
- GeoLife nearly catches up to DIY at prev7

**Panel 3 (Accuracy@10)**:
- Highest values for both datasets (~85%)
- Nearly identical at prev7 (85.16% vs 85.04%)
- Curves almost converge

**Panel 4 (MRR)**:
- Similar shape to Acc@1
- Consistent ~5 pp gap between datasets

**Panel 5 (NDCG@10)**:
- Similar to MRR pattern
- Values between MRR and Acc@5

**Panel 6 (F1)**:
- DIY shows clear improvement
- GeoLife nearly flat (46.5% → 47.0%)
- Largest difference between datasets

### 2.5 Interpretation

The plot demonstrates:
1. **Universal improvement**: All metrics improve with more data
2. **Diminishing returns**: Curves flatten after day 3-4
3. **Dataset differences**: DIY consistently outperforms except for Acc@10 at prev7
4. **Top-k scaling**: Higher k → higher absolute values, larger improvements

---

## 3. Accuracy Heatmap

### 3.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `accuracy_heatmap.{pdf,png}` |
| **Dimensions** | 14 × 5 inches |
| **Panels** | 2 (DIY left, GeoLife right) |

### 3.2 Visual Description

```
        DIY Dataset                      GeoLife Dataset
┌─────────────────────────────┐  ┌─────────────────────────────┐
│     1   2   3   4   5   6   7 │  │     1   2   3   4   5   6   7 │
│   ┌───┬───┬───┬───┬───┬───┬───┤  │   ┌───┬───┬───┬───┬───┬───┬───┤
│a@1│50 │54 │55 │56 │56 │57 │57 │  │a@1│48 │49 │49 │51 │50 │51 │51 │
│   ├───┼───┼───┼───┼───┼───┼───┤  │   ├───┼───┼───┼───┼───┼───┼───┤
│a@5│73 │77 │79 │80 │81 │82 │82 │  │a@5│70 │74 │77 │78 │79 │80 │81 │
│   ├───┼───┼───┼───┼───┼───┼───┤  │   ├───┼───┼───┼───┼───┼───┼───┤
│a10│75 │80 │82 │83 │84 │85 │85 │  │a10│74 │78 │80 │82 │83 │84 │85 │
│   ├───┼───┼───┼───┼───┼───┼───┤  │   ├───┼───┼───┼───┼───┼───┼───┤
│mrr│60 │64 │66 │67 │67 │67 │68 │  │mrr│58 │60 │61 │63 │63 │64 │65 │
│   ├───┼───┼───┼───┼───┼───┼───┤  │   ├───┼───┼───┼───┼───┼───┼───┤
│ndg│63 │68 │70 │71 │71 │72 │72 │  │ndg│62 │64 │66 │67 │68 │69 │69 │
│   └───┴───┴───┴───┴───┴───┴───┘  │   └───┴───┴───┴───┴───┴───┴───┘
└─────────────────────────────┘  └─────────────────────────────┘

Color scale: Light (low ~50%) ──────▶ Dark (high ~85%)
```

### 3.3 How to Read This Plot

**Axes**:
- X-axis: prev_days (1-7)
- Y-axis: Metric name (acc@1, acc@5, acc@10, mrr, ndcg)

**Color Scale**:
- Lighter gray: Lower values
- Darker gray: Higher values
- Color intensity represents metric magnitude

**Cell Values**: Numeric metric values (rounded)

### 3.4 Key Observations

1. **Horizontal gradient**: Clear left-to-right darkening shows improvement with more days
2. **Vertical pattern**: Top row (acc@1) is lightest, bottom rows (acc@10) are darkest
3. **DIY is darker overall**: Reflects higher performance
4. **Gradient steepness**: Steeper change in days 1-3, gentler in days 5-7

### 3.5 Specific Values to Note

**DIY Heatmap**:
- Lightest cell: acc@1, day 1 = 50.0%
- Darkest cell: acc@10, day 7 = 85.2%
- Range: 35.2 percentage points

**GeoLife Heatmap**:
- Lightest cell: acc@1, day 1 = 47.8%
- Darkest cell: acc@10, day 7 = 85.0%
- Range: 37.2 percentage points

---

## 4. Loss Curve

### 4.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `loss_curve.{pdf,png}` |
| **Dimensions** | 8 × 6 inches |
| **Type** | Line plot |

### 4.2 Visual Description

```
Cross-Entropy Loss
       │
  3.8  │  ○
       │   ╲
  3.5  │    ○ □
       │     ╲ ╲
  3.2  │      ○ □
       │       ╲ ╲
  2.9  │        ○─○─○ □
       │              ╲ □─□─□
  2.6  │                    □
       │
       └────┬────┬────┬────┬────┬────┬────▶
            1    2    3    4    5    6    7
                     t (days)

○ DIY (blue)    □ GeoLife (red)
```

### 4.3 How to Read This Plot

**X-Axis**: `t (days)` - Number of previous days (1-7)

**Y-Axis**: Cross-Entropy Loss (lower is better)

**Interpretation**:
- Downward slope = improving (lower loss)
- Steeper descent = faster improvement
- Curves converging = similar loss at endpoint

### 4.4 Key Observations

1. **Both curves descend**: Consistent improvement
2. **GeoLife always lower**: Better calibrated predictions
3. **Parallel descent**: Similar relative improvement rate
4. **Steepest at start**: Days 1-3 show fastest loss reduction

### 4.5 Numerical Analysis

| Segment | DIY ΔLoss | GeoLife ΔLoss |
|---------|-----------|---------------|
| Days 1-2 | -0.421 | -0.277 |
| Days 2-3 | -0.202 | -0.200 |
| Days 3-4 | -0.101 | -0.168 |
| Days 4-7 | -0.165 | -0.217 |

**Total reduction**: DIY -0.889 (23.6%), GeoLife -0.862 (24.7%)

### 4.6 Why GeoLife Has Lower Loss

Despite lower accuracy, GeoLife has lower loss because:
1. More confident predictions when correct
2. Fewer location classes (simpler distribution)
3. Cleaner GPS data from dedicated loggers

---

## 5. Radar Comparison

### 5.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `radar_comparison.{pdf,png}` |
| **Dimensions** | 12 × 5 inches |
| **Panels** | 2 (DIY left, GeoLife right) |

### 5.2 Visual Description

```
        DIY Dataset                    GeoLife Dataset
           ACC@1                          ACC@1
             ▲                              ▲
            ╱│╲                            ╱│╲
           ╱ │ ╲                          ╱ │ ╲
    NDCG  ╱──┼──╲ ACC@5           NDCG  ╱──┼──╲ ACC@5
          ╲  │  ╱                       ╲  │  ╱
           ╲ │ ╱                         ╲ │ ╱
            ╲│╱                           ╲│╱
            MRR                           MRR
           ╱   ╲                         ╱   ╲
    ACC@10      ACC@10            ACC@10      ACC@10

    --- prev1 (black dashed)
    ─── prev7 (blue/red solid)
```

### 5.3 How to Read This Plot

**Axes**: Five metrics arranged radially (ACC@1, ACC@5, ACC@10, MRR, NDCG)

**Polygons**:
- Black dashed line with circles: prev1 (1-day history)
- Colored solid line with squares: prev7 (7-day history)

**Interpretation**:
- Larger polygon = better overall performance
- Area difference = improvement magnitude
- Shape = relative metric strengths

### 5.4 Key Observations

1. **prev7 polygon is larger**: Clear improvement across all metrics
2. **Uniform expansion**: All vertices move outward similarly
3. **ACC@5/ACC@10 corners**: Largest outward movement (biggest improvements)
4. **ACC@1 vertex**: Smallest outward movement (modest improvement)

### 5.5 Area Comparison (Approximate)

| Dataset | prev1 Area | prev7 Area | Increase |
|---------|------------|------------|----------|
| DIY | ~0.68 | ~0.82 | +20.6% |
| GeoLife | ~0.64 | ~0.78 | +21.9% |

*Note: Areas normalized to unit circle*

---

## 6. Improvement Comparison

### 6.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `improvement_comparison.{pdf,png}` |
| **Dimensions** | 10 × 6 inches |
| **Type** | Grouped bar chart |

### 6.2 Visual Description

```
Relative Improvement (%)
    │
16% │                    ▓▓
    │         ▒▒         ▓▓
14% │    ░░   ▒▒    ░░   ▓▓    ░░        ░░
    │    ░░   ▒▒    ░░   ▓▓    ░░   ░░   ░░
12% │    ░░   ▒▒    ░░   ▓▓    ░░   ░░   ░░
    │    ░░   ▒▒    ░░   ▓▓    ░░   ░░   ░░
10% │    ░░   ▒▒    ░░   ▓▓    ░░   ░░   ░░
    │    ░░   ▒▒    ░░   ▓▓    ░░   ░░   ░░
 8% │    ░░   ▒▒         ▓▓    ░░   ░░   
    │    ░░   ▒▒              ░░   ░░   
 6% │    ░░                    ░░   ░░   
    │    ░░                    ░░   ░░   
 4% │    ░░                    ░░   ░░        ▒▒
    │    ░░                    ░░   ░░        ▒▒
 2% │    ░░                    ░░   ░░        ▒▒
    │    ░░                    ░░   ░░        ▒▒
 0% └────┴──────┴──────┴──────┴──────┴──────┴──────▶
        ACC@1   ACC@5  ACC@10   MRR   NDCG    F1

░░ DIY (blue with diagonal hatch)
▒▒ GeoLife (red with dots)
```

### 6.3 How to Read This Plot

**X-Axis**: Metric names

**Y-Axis**: Relative improvement from prev1 to prev7 (%)

**Bar Pairs**: Left bar (DIY, blue), Right bar (GeoLife, red)

### 6.4 Key Observations

1. **ACC@5 shows largest GeoLife improvement**: +16.0% vs +13.3%
2. **ACC@1 shows largest DIY advantage**: +13.2% vs +7.4%
3. **F1 shows biggest disparity**: DIY +11.1% vs GeoLife +3.2%
4. **MRR, NDCG, ACC@10 are similar**: ~12-14% for both

### 6.5 Interpretation

- **DIY excels at top-1 prediction**: Strong habitual patterns
- **GeoLife excels at candidate ranking**: Better at "in the list" metrics
- **F1 tells the class balance story**: DIY improvement is uniform, GeoLife is concentrated

---

## 7. Sequence Length Distribution

### 7.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `sequence_length_distribution.{pdf,png}` |
| **Dimensions** | 12 × 5 inches |
| **Panels** | 2 (DIY left, GeoLife right) |

### 7.2 Visual Description

```
        DIY Dataset                    GeoLife Dataset
Avg Seq Length                    Avg Seq Length
    │                                  │
 25 │              ▓▓▓                 │
    │           ▓▓▓▓▓▓              20 │              ▓▓▓
 20 │        ▓▓▓▓▓▓▓▓▓                 │           ▓▓▓▓▓▓
    │     ▓▓▓▓▓▓▓▓▓▓▓▓              15 │        ▓▓▓▓▓▓▓▓▓
 15 │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 │     ▓▓▓▓▓▓▓▓▓▓▓▓
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓              10 │  ▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
 10 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓               5 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
  5 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓                 │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓
    └─┬──┬──┬──┬──┬──┬──▶               └─┬──┬──┬──┬──┬──┬──▶
      1  2  3  4  5  6  7                 1  2  3  4  5  6  7
           t (days)                            t (days)

Error bars show ±1 standard deviation
```

### 7.3 How to Read This Plot

**X-Axis**: `t (days)` - Number of previous days (1-7)

**Y-Axis**: Average sequence length (number of location visits)

**Bars**: Height = mean sequence length

**Error Bars**: ±1 standard deviation

### 7.4 Key Observations

1. **Linear growth**: Sequence length grows linearly with days
2. **DIY is longer**: ~30% more visits per time period
3. **Variability increases**: Error bars grow with days
4. **Approximate rates**:
   - DIY: ~3.4 visits per day
   - GeoLife: ~2.6 visits per day

### 7.5 Numerical Values

| prev_days | DIY Avg (±Std) | GeoLife Avg (±Std) |
|-----------|----------------|-------------------|
| 1 | 5.6 (±4.1) | 4.1 (±2.7) |
| 2 | 8.8 (±6.3) | 6.5 (±4.1) |
| 3 | 11.9 (±8.4) | 8.9 (±5.5) |
| 4 | 14.9 (±10.3) | 11.2 (±6.9) |
| 5 | 17.9 (±12.2) | 13.6 (±8.3) |
| 6 | 20.9 (±14.1) | 15.9 (±9.7) |
| 7 | 24.0 (±15.8) | 18.4 (±11.1) |

---

## 8. Samples vs Performance

### 8.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `samples_vs_performance.{pdf,png}` |
| **Dimensions** | 8 × 6 inches |
| **Type** | Scatter plot with labels |

### 8.2 Visual Description

```
Accuracy@1 (%)
    │
 57 │                              ⁷○
    │                           ⁵⁶○○
 55 │                         ⁴○
    │                        ³○
 53 │                       ²○
    │
 51 │         ⁷□
    │      ⁴⁵⁶□□□
 50 │      ²³□□
    │    ¹□
 48 │   ¹○
    │
    └────┬────┬────┬────┬────┬────┬────▶
        3K   4K   8K   10K  11K  12K  13K
              Number of Test Samples

○ DIY (blue)   □ GeoLife (red)
Numbers 1-7 indicate prev_days
```

### 8.3 How to Read This Plot

**X-Axis**: Number of test samples (after filtering)

**Y-Axis**: Accuracy@1 (%)

**Points**: Each point represents one configuration

**Labels**: Numbers near points indicate prev_days (1-7)

### 8.4 Key Observations

1. **Two clusters**: DIY (right, higher) and GeoLife (left, lower)
2. **Within-cluster positive correlation**: More samples → higher accuracy
3. **Label progression**: Higher numbers (more days) are higher and to the right
4. **Scale difference**: DIY has 3.5× more samples

### 8.5 Interpretation

- Sample count and performance are correlated within each dataset
- Adding days increases both sample count (more complete sequences qualify) AND performance
- The 3.5× dataset size difference partially explains DIY's advantage

---

## 9. Combined Figure

### 9.1 File Information

| Property | Value |
|----------|-------|
| **Filename** | `combined_figure.{pdf,png,svg}` |
| **Dimensions** | 14 × 10 inches |
| **Purpose** | Publication-ready multi-panel figure |

### 9.2 Panel Layout

```
┌─────────────────────────────────────────────────────────────┐
│  (a) Acc@1         │  (b) Acc@5         │  (c) MRR          │
│  [Line plot]       │  [Line plot]       │  [Line plot]      │
├────────────────────┼────────────────────┼───────────────────┤
│  (d) NDCG@10       │  (e) Loss          │  (f) Seq Length   │
│  [Line plot]       │  [Line plot]       │  [Bar chart]      │
├────────────────────┴────────────────────┼───────────────────┤
│  (g) Relative Improvement               │  (h) Sample Count │
│  [Grouped bar chart]                    │  [Line plot]      │
└─────────────────────────────────────────┴───────────────────┘
```

### 9.3 Panel Descriptions

**(a) Accuracy@1**: Line plot of Acc@1 vs days
**(b) Accuracy@5**: Line plot of Acc@5 vs days
**(c) MRR**: Line plot of MRR vs days
**(d) NDCG@10**: Line plot of NDCG vs days
**(e) Loss**: Line plot of loss vs days (inverted Y for consistency)
**(f) Seq Length**: Bar chart with error bars
**(g) Improvement**: Grouped bars for relative improvement
**(h) Sample Count**: Line plot of samples vs days

### 9.4 Usage

This combined figure is designed for:
- Academic paper submission
- Conference presentations
- Technical reports
- Thesis chapters

All panels share consistent styling, colors, and legends.

---

## 10. Color and Style Reference

### 10.1 Color Palette

| Element | Color | Hex Code | RGB |
|---------|-------|----------|-----|
| DIY primary | Blue | #1f77b4 | (31, 119, 180) |
| GeoLife primary | Red | #d62728 | (214, 39, 40) |
| Background | White | #ffffff | (255, 255, 255) |
| Axis/Text | Black | #000000 | (0, 0, 0) |
| Grid (when used) | Light gray | #cccccc | (204, 204, 204) |

### 10.2 Marker Specifications

| Dataset | Marker | Size | Fill | Edge Width |
|---------|--------|------|------|------------|
| DIY | Circle (○) | 8 pt | White | 1.5 pt |
| GeoLife | Square (□) | 8 pt | White | 1.5 pt |

### 10.3 Line Specifications

| Element | Width | Style |
|---------|-------|-------|
| Data lines | 1.5 pt | Solid |
| Prev1 reference | 1.0 pt | Dashed |
| Axis | 1.0 pt | Solid |

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,600 |
| **Status** | Final |

---

**Navigation**: [← Results & Analysis](./09_results_and_analysis.md) | [Index](./INDEX.md) | [Next: Interpretation & Insights →](./11_interpretation_and_insights.md)
