# Descriptive Analysis Results

## Comprehensive Documentation - Part 3

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset Overview](#dataset-overview)
3. [Target-in-History Analysis](#target-in-history-analysis)
4. [Repetition Pattern Analysis](#repetition-pattern-analysis)
5. [Vocabulary Utilization Analysis](#vocabulary-utilization-analysis)
6. [User Pattern Analysis](#user-pattern-analysis)
7. [Sequence Characteristics](#sequence-characteristics)
8. [Temporal Pattern Analysis](#temporal-pattern-analysis)
9. [Summary of Key Differences](#summary-of-key-differences)

---

## Introduction

This section presents the descriptive analysis of DIY and GeoLife test datasets. The goal is to identify fundamental structural differences that may explain the differential pointer mechanism impact.

**Script:** `01_descriptive_analysis.py`

---

## Dataset Overview

### Basic Statistics

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Total Test Samples | 12,368 | 3,502 | -8,866 |
| Number of Users | 692 | 45 | -647 |
| Unique Locations in Test | 2,346 | 347 | -1,999 |
| Unique Target Locations | 1,713 | 315 | -1,398 |
| Train Vocabulary Size | 7,017 | 1,156 | -5,861 |

**Key Insight:** DIY is a much larger dataset with 3.5× more samples, 15× more users, and 5.4× more unique target locations.

---

## Target-in-History Analysis

### What This Measures

Target-in-history rate measures how often the next location (target) has been visited in the recent input sequence. This directly determines when the pointer mechanism can be useful—it can only "copy" a location that exists in the history.

### Results

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Target-in-History Rate | **84.12%** | **83.81%** | -0.31% |
| Target in Last-1 Position | 18.56% | 27.18% | +8.63% |
| Target in Last-3 Positions | 64.89% | 65.53% | +0.64% |
| Target in Last-5 Positions | 73.59% | 73.73% | +0.14% |
| Target in Last-7 Positions | 77.23% | 77.07% | -0.16% |

### Detailed Breakdown

**DIY Dataset:**
- Total samples: 12,368
- Target in history: 10,404 samples (84.12%)
- Target NOT in history: 1,964 samples (15.88%)

**GeoLife Dataset:**
- Total samples: 3,502
- Target in history: 2,935 samples (83.81%)
- Target NOT in history: 567 samples (16.19%)

### Interpretation

**Critical Finding:** Both datasets have nearly identical target-in-history rates (~84%). This means:

1. The **opportunity** for the pointer mechanism is equal for both datasets
2. The differential ablation impact is **NOT** due to copy applicability
3. We must look elsewhere for the explanation

The slightly higher "Target in Last-1" rate for GeoLife (27.18% vs 18.56%) suggests GeoLife users more frequently return to their immediately previous location, but this difference alone doesn't explain the ablation impact differential.

### Figure Reference

**Figure 1: Target-in-History Analysis** (`fig1_target_in_history.png`)

- **Panel (a):** Bar chart comparing overall target-in-history rates
- **Panel (b):** Distribution of target position from end
- **Panel (c):** Target-in-last-k comparison for k=1,3,5,7

---

## Repetition Pattern Analysis

### What This Measures

Repetition patterns measure how often locations are revisited within a single sequence. Higher repetition suggests more predictable mobility patterns that might benefit from copy mechanisms.

### Results

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Average Repetition Rate | 0.6865 | 0.6596 | -0.027 |
| Std Repetition Rate | 0.1756 | 0.1618 | -0.014 |
| Avg Consecutive Repetition | 0.1794 | 0.2687 | +0.0893 |
| Average Unique Ratio | 0.3135 | 0.3404 | +0.027 |

### Interpretation

**Repetition Rate Formula:**
```
Repetition Rate = (Sequence Length - Unique Locations) / Sequence Length
```

A rate of 0.6865 (DIY) means about 69% of locations in sequences are repeats of previous locations.

**Key Observations:**

1. **DIY has slightly higher repetition** (0.6865 vs 0.6596), meaning DIY sequences have more repeated visits
2. **GeoLife has higher consecutive repetition** (0.2687 vs 0.1794), meaning users more often stay in the same location back-to-back
3. Both datasets show high repetition (~65-69%), indicating strong revisitation patterns

**Implication:** The similar repetition rates suggest both datasets should benefit similarly from a copy mechanism.

### Figure Reference

**Figure 2: Repetition Patterns** (`fig2_repetition_patterns.png`)

- **Panel (a):** Distribution of repetition rates (step histogram)
- **Panel (b):** Comparison of average repetition and consecutive repetition
- **Panel (c):** Boxplot of unique location ratio

---

## Vocabulary Utilization Analysis

### What This Measures

Vocabulary analysis examines how many unique locations exist and how concentrated the distribution is. A more concentrated distribution (fewer locations covering most visits) makes generation easier.

### Results

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Unique Locations in Test | 2,346 | 347 | -1,999 |
| Unique Targets | **1,713** | **315** | -1,398 |
| Top-10 Sequence Coverage | 42.61% | 69.34% | +26.73% |
| Top-50 Sequence Coverage | 56.78% | 90.03% | +33.25% |
| Top-10 Target Coverage | **41.75%** | **67.13%** | +25.38% |
| Top-50 Target Coverage | 52.43% | 88.15% | +35.72% |
| Sequence Entropy | 4.956 | 3.376 | -1.580 |
| Target Entropy | 5.022 | 3.539 | -1.483 |

### Interpretation

**This is the critical difference that explains everything.**

#### Vocabulary Size
- DIY has **5.4× more unique target locations** (1,713 vs 315)
- This means the generation head must predict over a much larger space

#### Location Concentration
- **GeoLife Top-10 Coverage: 67.13%** - Just 10 locations account for 67% of targets
- **DIY Top-10 Coverage: 41.75%** - Top 10 locations only cover 42% of targets
- This 25% difference is substantial

#### Entropy Analysis
- **Higher entropy = more uniform distribution = harder to predict**
- DIY target entropy: 5.022 (harder)
- GeoLife target entropy: 3.539 (easier)

**Why This Matters:**

1. **For Generation Head:**
   - GeoLife: Predict among ~315 options with 67% concentrated in top 10
   - DIY: Predict among ~1,713 options with only 42% in top 10
   - Result: GeoLife generation is fundamentally easier

2. **For Model Training:**
   - GeoLife model can learn effective generation patterns
   - DIY model struggles with generation → relies more on pointer

### Figure Reference

**Figure 3: Vocabulary and User Patterns** (`fig3_vocabulary_user_patterns.png`)

- **Panel (a):** Comparison of unique location counts
- **Panel (b):** Top-10 and Top-50 coverage comparison
- **Panel (c):** User target revisit rate distribution

---

## User Pattern Analysis

### What This Measures

User-level analysis examines individual mobility patterns and diversity.

### Results

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Number of Users | 692 | 45 | -647 |
| Avg Samples per User | 17.87 | 77.82 | +59.95 |
| Std Samples per User | 18.95 | 112.16 | +93.21 |
| Avg Unique Locations per User | 7.06 | 12.56 | +5.50 |
| Avg Location Diversity | 0.0958 | 0.1039 | +0.008 |
| Avg Target Revisit Rate | 0.9738 | 0.9584 | -0.0154 |

### Interpretation

**Key Observations:**

1. **More data per user in GeoLife** (77.82 vs 17.87 samples/user)
   - GeoLife has fewer users but more data per user
   - This allows better user-specific pattern learning

2. **High target revisit rate for both** (~96-97%)
   - Users overwhelmingly revisit known locations
   - Supports the pointer mechanism's utility

3. **DIY has more users, less data per user**
   - Greater user diversity
   - Less opportunity to learn individual patterns

---

## Sequence Characteristics

### Results

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Average Sequence Length | 23.98 | 18.37 | -5.61 |
| Std Sequence Length | 15.77 | 11.08 | -4.69 |
| Min Sequence Length | 3 | 3 | 0 |
| Max Sequence Length | 99 | 46 | -53 |
| Median Sequence Length | 21.0 | 14.0 | -7.0 |

### Interpretation

**DIY has longer sequences** (avg 24 vs 18), which provides:
- More context for the model
- More positions from which to copy
- Potentially more challenging generation (more history to consider)

The longer sequences in DIY may contribute to the higher repetition rate and more complex generation task.

---

## Temporal Pattern Analysis

### Results

| Metric | DIY | GeoLife |
|--------|-----|---------|
| Morning (6-12) | 34.48% | 36.39% |
| Afternoon (12-18) | 30.81% | 25.71% |
| Evening (18-22) | 18.07% | 8.77% |
| Night (22-6) | 16.64% | 29.12% |
| Avg Duration (min) | 357.60 | 355.37 |
| Avg Recency (days) | 3.62 | 3.68 |

### Interpretation

**Key Differences:**
- GeoLife has significantly more night activity (29% vs 17%)
- DIY has more evening activity (18% vs 9%)
- Both have similar average duration and recency

These temporal differences reflect the different contexts (Beijing vs general urban) but don't directly explain the ablation impact differential.

---

## Summary of Key Differences

### Factors That Are SIMILAR (Don't Explain Differential Impact)

| Factor | DIY | GeoLife | Conclusion |
|--------|-----|---------|------------|
| Target-in-History Rate | 84.12% | 83.81% | Equal copy opportunity |
| Repetition Rate | 0.6865 | 0.6596 | Similar patterns |
| Target Revisit Rate | 97.38% | 95.84% | Both highly revisit |
| Avg Duration | 357.6 min | 355.4 min | Similar |
| Avg Recency | 3.62 days | 3.68 days | Similar |

### Factors That Are DIFFERENT (May Explain Differential Impact)

| Factor | DIY | GeoLife | Implication |
|--------|-----|---------|-------------|
| **Unique Targets** | 1,713 | 315 | DIY 5.4× harder for generation |
| **Top-10 Target Coverage** | 41.75% | 67.13% | GeoLife more concentrated |
| **Target Entropy** | 5.022 | 3.539 | DIY more uniform (harder) |
| Number of Users | 692 | 45 | DIY more diverse |
| Samples per User | 17.87 | 77.82 | GeoLife more per-user data |

### Primary Hypothesis Emerging from Descriptive Analysis

**The vocabulary size and concentration difference is the root cause:**

1. DIY has 5.4× more unique target locations
2. This makes generation fundamentally harder for DIY
3. The model compensates by relying more on pointer
4. When pointer is ablated, the impact differs because the generation "backup" has different baseline performance

**This hypothesis is tested in the diagnostic and hypothesis testing phases.**

---

*All data presented are from actual experimental runs and are reproducible.*
