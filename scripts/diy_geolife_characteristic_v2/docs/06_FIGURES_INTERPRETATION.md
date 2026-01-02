# Figure Interpretation Guide

## Comprehensive Documentation - Part 6

---

## Table of Contents

1. [Overview](#overview)
2. [Figure 1: Target-in-History Analysis](#figure-1-target-in-history-analysis)
3. [Figure 2: Repetition Patterns](#figure-2-repetition-patterns)
4. [Figure 3: Vocabulary and User Patterns](#figure-3-vocabulary-and-user-patterns)
5. [Figure 4: Radar Comparison](#figure-4-radar-comparison)
6. [Figure 5: Gate Analysis](#figure-5-gate-analysis)
7. [Figure 6: Pointer vs Generation Performance](#figure-6-pointer-vs-generation-performance)
8. [Figure 7: Vocabulary Effect](#figure-7-vocabulary-effect)
9. [Experiment Figures](#experiment-figures)
10. [Summary Root Cause Figure](#summary-root-cause-figure)

---

## Overview

All figures follow a **classic scientific publication style**:
- White background with black axis borders
- Inside tick marks on all four sides
- No grid lines
- Consistent color coding: Blue (DIY), Red (GeoLife)
- Open markers for scatter/line plots
- Hatched bars for bar charts
- Panel labels (a), (b), (c) for multi-panel figures

**File Formats:** All figures are saved as both PNG (300 DPI) and PDF (vector).

---

## Figure 1: Target-in-History Analysis

**File:** `fig1_target_in_history.png`

### Panel (a): Overall Target-in-History Rate

**What it shows:** Bar chart comparing the percentage of test samples where the target location appears somewhere in the input sequence.

**Axes:**
- X-axis: Dataset (DIY, GeoLife)
- Y-axis: Target-in-History Rate (%)

**Values shown:**
- DIY: 84.1%
- GeoLife: 83.8%

**Interpretation:** Both datasets have nearly identical target-in-history rates (~84%). This means the **opportunity for the pointer mechanism to be useful is equal** for both datasets. The differential ablation impact (46.7% vs 8.3%) is NOT due to different copy applicability.

**Key Insight:** If this metric differed significantly, it would explain differential impact. Since it doesn't, we must look elsewhere.

### Panel (b): Target Position Distribution

**What it shows:** Step histogram (density plot) of where in the sequence the target appears, measured as "position from end."

**Axes:**
- X-axis: Position from End (1 = immediately previous, 7 = 7 positions ago, etc.)
- Y-axis: Density (normalized frequency)

**Visual elements:**
- Blue solid line: DIY distribution
- Red dashed line: GeoLife distribution

**Interpretation:** 
- Both distributions show most targets appear in recent positions
- GeoLife has slightly higher concentration at position 1 (27.2% vs 18.6%)
- This means GeoLife users more often return to their immediately previous location
- Peaks at positions 1, 3, and 5 suggest periodic patterns (home-work-home cycles)

### Panel (c): Target-in-Last-k Comparison

**What it shows:** Grouped bar chart showing what percentage of targets appear in the last 1, 3, 5, or 7 positions.

**Axes:**
- X-axis: Last k positions (k = 1, 3, 5, 7)
- Y-axis: Rate (%)

**Values:**
| k | DIY | GeoLife |
|---|-----|---------|
| 1 | 18.6% | 27.2% |
| 3 | 64.9% | 65.5% |
| 5 | 73.6% | 73.7% |
| 7 | 77.2% | 77.1% |

**Interpretation:**
- GeoLife has 8.6% more targets in the immediately previous position
- By k=3, both datasets converge to ~65%
- By k=7, both reach ~77%
- This suggests pointer attention should weight recent positions similarly for both

---

## Figure 2: Repetition Patterns

**File:** `fig2_repetition_patterns.png`

### Panel (a): Repetition Rate Distribution

**What it shows:** Step histogram of repetition rates across all test samples.

**Formula:**
```
Repetition Rate = (Sequence Length - Unique Locations) / Sequence Length
```

**Axes:**
- X-axis: Repetition Rate (0-1)
- Y-axis: Density

**Visual elements:**
- Blue solid line: DIY distribution
- Red dashed line: GeoLife distribution
- Dotted vertical lines: Mean values for each dataset

**Values:**
- DIY Mean: 0.6865 (blue dotted line)
- GeoLife Mean: 0.6596 (red dotted line)

**Interpretation:**
- Both datasets show high repetition (65-69% of locations are repeats)
- DIY has slightly higher average repetition
- Distributions largely overlap
- This supports the finding that both datasets have similar mobility patterns

### Panel (b): Average Repetition Metrics

**What it shows:** Grouped bar chart comparing repetition metrics.

**Axes:**
- X-axis: Metric type (Avg Repetition Rate, Avg Consecutive Repetition)
- Y-axis: Rate (0-1)

**Values:**
| Metric | DIY | GeoLife |
|--------|-----|---------|
| Avg Repetition Rate | 0.687 | 0.660 |
| Avg Consecutive Repetition | 0.179 | 0.269 |

**Interpretation:**
- DIY has slightly higher overall repetition
- GeoLife has significantly higher **consecutive** repetition (0.269 vs 0.179)
- Consecutive repetition = staying in same location back-to-back
- This explains GeoLife's higher "last-1" target rate

### Panel (c): Unique Location Ratio Boxplot

**What it shows:** Boxplots of unique location ratio per sequence.

**Formula:**
```
Unique Ratio = Unique Locations / Sequence Length
```

**Axes:**
- X-axis: Dataset
- Y-axis: Unique Location Ratio (0-1)

**Interpretation:**
- Both datasets show similar distributions
- Medians around 0.3-0.35 (30-35% of locations are unique)
- GeoLife has slightly more variance
- Confirms high repetition in both datasets

---

## Figure 3: Vocabulary and User Patterns

**File:** `fig3_vocabulary_user_patterns.png`

### Panel (a): Unique Location Count

**What it shows:** Bar chart comparing the number of unique locations in test sequences.

**Axes:**
- X-axis: Dataset
- Y-axis: Count

**Values:**
- DIY: 2,346 unique locations
- GeoLife: 347 unique locations

**Interpretation:**
- **DIY has 6.8× more unique locations** in test sequences
- This is the fundamental structural difference between datasets
- More locations = harder generation problem

### Panel (b): Coverage Comparison

**What it shows:** Grouped bar chart comparing top-10 and top-50 location coverage.

**Axes:**
- X-axis: Coverage type (Top-10, Top-50)
- Y-axis: Coverage (%)

**Values:**
| Coverage | DIY | GeoLife |
|----------|-----|---------|
| Top-10 | 42.6% | 69.3% |
| Top-50 | 56.8% | 90.0% |

**Interpretation:**
- **GeoLife top-10 locations cover 69% of visits** (vs 43% for DIY)
- GeoLife top-50 covers 90% of visits
- This concentration makes generation easier for GeoLife
- DIY's more uniform distribution makes generation harder

**This is a key finding supporting the root cause explanation.**

### Panel (c): User Target Revisit Rate Boxplot

**What it shows:** Boxplots of per-user target revisit rates.

**Definition:** How often each user's target is a previously visited location.

**Axes:**
- X-axis: Dataset
- Y-axis: Target Revisit Rate (0-1)

**Values:**
- DIY Mean: 0.974 (97.4%)
- GeoLife Mean: 0.958 (95.8%)

**Interpretation:**
- Both datasets show users overwhelmingly revisit known locations
- Slight variation between users
- Supports the general utility of copy mechanisms

---

## Figure 4: Radar Comparison

**File:** `fig4_radar_comparison.png`

### What it shows

A radar (spider) chart comparing normalized metrics across 6 dimensions.

**Dimensions:**
1. Target-in-History Rate
2. Repetition Rate
3. Top-10 Coverage
4. User Revisit Rate
5. Avg Seq Length (normalized)
6. Location Concentration (inverse of vocabulary)

**Visual elements:**
- Blue line with circle markers: DIY
- Red line with square markers: GeoLife
- Values normalized to 0-1 scale

**Interpretation:**
- Both datasets have similar shapes for target-in-history and repetition
- **GeoLife dominates in Top-10 Coverage** (more concentrated)
- **DIY has higher location concentration (normalized inverse)** = less concentrated
- This visual summary highlights the vocabulary/concentration as the key difference

---

## Figure 5: Gate Analysis

**File:** `fig5_gate_analysis.png`

### Panel (a): Gate Value Distribution

**What it shows:** Step histogram of learned gate values across all test samples.

**Axes:**
- X-axis: Gate Value (0 = all generation, 1 = all pointer)
- Y-axis: Density

**Visual elements:**
- Blue solid line: DIY distribution
- Red dashed line: GeoLife distribution
- Dotted vertical lines: Mean values

**Values:**
- DIY Mean Gate: 0.787
- GeoLife Mean Gate: 0.627

**Interpretation:**
- **DIY distribution is shifted right** (higher gate values)
- DIY model relies ~79% on pointer, ~21% on generation
- GeoLife model uses ~63% pointer, ~37% generation
- This shows DIY is more pointer-dependent

**Key Insight:** The model learned to rely more on pointer for DIY because generation is weak.

### Panel (b): Mean Gate by Target-in-History Status

**What it shows:** Grouped bar chart of average gate values for samples where target IS vs IS NOT in history.

**Axes:**
- X-axis: Target in History, Target NOT in History
- Y-axis: Mean Gate Value

**Values:**
| Condition | DIY | GeoLife |
|-----------|-----|---------|
| Target IN History | 0.803 | 0.637 |
| Target NOT in History | Lower | Lower |

**Interpretation:**
- Both models correctly increase gate when target is in history
- This shows learned adaptive behavior
- Even when pointer can't help (target not in history), DIY still has higher gate

### Panel (c): Gate vs Target Position Scatter

**What it shows:** Scatter plot of gate values vs target position from end.

**Axes:**
- X-axis: Target Position from End
- Y-axis: Gate Value

**Visual elements:**
- Blue open circles: DIY samples
- Red open squares: GeoLife samples

**Interpretation:**
- No strong correlation between position and gate
- Gate is more influenced by overall dataset characteristics than per-sample recency

---

## Figure 6: Pointer vs Generation Performance

**File:** `fig6_ptr_vs_gen.png`

### Top Row: Accuracy Breakdown

**Panels (0,0) and (0,1):** Per-dataset accuracy breakdown

**Axes:**
- X-axis: Categories (Target in Hist, Target NOT in Hist, Overall)
- Y-axis: Accuracy (%)

**Bar colors:**
- Green: Pointer head only
- Purple: Generation head only
- Orange: Combined model

**Key Values (DIY - Panel 0,0):**
| Category | Pointer | Generation | Combined |
|----------|---------|------------|----------|
| Target IN | 67.2% | 5.7% | 67.2% |
| Target OUT | 0.0% | 5.1% | 0.2% |
| Overall | 56.5% | 5.6% | 56.6% |

**Key Values (GeoLife - Panel 0,1):**
| Category | Pointer | Generation | Combined |
|----------|---------|------------|----------|
| Target IN | 61.6% | 13.9% | 61.3% |
| Target OUT | 0.0% | 3.5% | 0.4% |
| Overall | 51.6% | 12.2% | 51.4% |

**Critical Observation:** GeoLife generation (13.9% when target in history) is 2.4× better than DIY generation (5.7%).

**Panel (0,2):** Direct comparison across datasets

Shows side-by-side comparison of all components for both datasets.

### Bottom Row: Detailed Analysis

**Panel (1,0): MRR Comparison**
- Pointer MRR: ~68-70% for both
- Generation MRR: 10.1% (DIY) vs 18.7% (GeoLife)
- Generation MRR differs by 1.9×

**Panel (1,1): Probability on Target Boxplots**
- Shows distribution of P(correct answer) for each component
- Pointer assigns higher probabilities to correct answer

**Panel (1,2): Pointer Advantage Distribution**
- Shows P_pointer(target) - P_generation(target)
- Both distributions centered above 0
- Mean advantage: ~0.43-0.47 for both

---

## Figure 7: Vocabulary Effect

**File:** `fig7_vocabulary_effect.png`

### Panel (a): Generation Accuracy vs Target Frequency

**What it shows:** Scatter plot of generation head accuracy by how frequently each target appears.

**Axes:**
- X-axis: Target Frequency in Test Set (log scale)
- Y-axis: Generation Head Accuracy (%)

**Visual elements:**
- Blue open circles: DIY targets
- Red open squares: GeoLife targets

**Interpretation:**
- Higher frequency targets → Higher generation accuracy (positive correlation)
- GeoLife has more high-frequency targets
- This explains why GeoLife generation performs better overall

### Panel (b): Unique Target Count

**What it shows:** Bar chart of number of unique target locations.

**Axes:**
- X-axis: Dataset
- Y-axis: Number of Unique Targets

**Values:**
- DIY: 1,713 unique targets
- GeoLife: 315 unique targets

**Interpretation:**
- **DIY has 5.4× more unique targets**
- Generation must predict over larger space for DIY
- This is the root cause of generation performance difference

---

## Experiment Figures

### exp1_stratified_analysis.png

Shows accuracy breakdown by target-in-history status with Δ annotations showing pointer benefit.

### exp2_ablation_simulation.png

Shows four gate configurations (Pointer Only, Gen Only, Combined, Fixed 50-50) with accuracy comparisons.

### exp3_generation_difficulty.png

Four-panel figure showing:
- (a) DIY target frequency distribution
- (b) GeoLife target frequency distribution  
- (c) Cumulative coverage curves
- (d) Summary metrics comparison

### exp5_target_in_history_ablation.png

Performance on full test vs filtered subsets (target in/not in history).

### exp5_recency_effect.png

Performance by target recency (last 1, 2, 3, 5, 10 positions).

---

## Summary Root Cause Figure

**File:** `fig_summary_root_cause.png`

### Overview

A comprehensive 6-panel figure plus text summary that synthesizes all findings.

### Panel (a): Component Performance

Grouped bar chart showing Pointer, Generation, and Combined accuracy for both datasets.

**Key numbers visible:**
- DIY Pointer: ~56%
- DIY Generation: ~6%
- GeoLife Pointer: ~52%
- GeoLife Generation: ~12%

### Panel (b): Pointer Advantage

Bar chart showing (Pointer - Generation) accuracy difference.

**Values:**
- DIY: 50.9%
- GeoLife: 39.4%

**Insight:** DIY has LARGER pointer advantage but SMALLER ablation impact.

### Panel (c): Target Vocabulary Size

Bar chart of unique target counts.

**Values:**
- DIY: 1,713
- GeoLife: 315

### Panel (d) & (e): Stratified Performance

Per-dataset breakdown showing Pointer, Generation, Combined performance for Target IN vs OUT of history.

### Panel (f): Simulated Ablation Impact

Bar chart showing relative performance drop when using generation only.

**Values:**
- DIY: 90.0% relative drop
- GeoLife: 76.3% relative drop

### Text Summary Box

Contains the complete root cause explanation:
- DIY: Larger vocabulary → Generation struggles → Model becomes pointer-dependent → Small relative ablation impact
- GeoLife: Smaller vocabulary → Generation works → Model uses both → Large relative ablation impact

---

*All figures are designed for publication quality and scientific clarity.*
