# Overview: Return Probability Distribution Analysis

## 1. Executive Summary

This document provides a comprehensive analysis of **human mobility return patterns**, specifically measuring **how likely and when users return to their first observed location**. This analysis reproduces and extends the seminal work by González et al. (2008), published in *Nature*, which revealed fundamental patterns in human movement behavior.

### Key Findings at a Glance

| Finding | Implication |
|---------|-------------|
| **83.5% of DIY users return** to their first location within 10 days | Human mobility is highly predictable |
| **24-hour peaks** in return probability | Daily routines dominate movement patterns |
| **Mean return time ~60 hours** | Users revisit important locations within 2-3 days |
| **Non-random behavior** deviates from Random Walk (RW) model | Simple models fail to capture human mobility |

---

## 2. What is Return Probability Distribution?

### 2.1 Intuitive Explanation

Imagine you're tracking a person's movements through a city:

```
Day 1, 8:00 AM: User starts at HOME (Location A)
Day 1, 9:00 AM: User goes to WORK (Location B)
Day 1, 6:00 PM: User returns to HOME (Location A) ← FIRST RETURN!
```

**First-Return Time (Δt)** = 6:00 PM - 8:00 AM = **10 hours**

The **Return Probability Distribution F_pt(t)** answers the question:

> *"What is the probability that a user returns to their starting location after exactly t hours?"*

### 2.2 Visual Concept

```
                   FIRST RETURN
                       ↓
Timeline: ─────●───────●───────●───────●───────●─────→ time
              t₀      t₁      t₂      t₃      t₄
              │       │
              │       └── Return to L₀ at t₁
              │           Δt = t₁ - t₀
              │
              └── First observation at location L₀

For this user: Δt = t₁ - t₀ hours
```

### 2.3 Mathematical Definition

For each user *u*:

1. **First Location**: L₀(u) = location at first observation
2. **First Time**: t₀(u) = timestamp of first observation  
3. **First Return Time**: t₁(u) = first time where location = L₀(u) AND time > t₀(u)
4. **Return Interval**: Δt(u) = t₁(u) - t₀(u)

The probability density function is:

```
           Number of users with return time in [t, t+Δt]
F_pt(t) = ─────────────────────────────────────────────────
                    N_total × Δt (bin width)
```

---

## 3. Why This Analysis Matters

### 3.1 Scientific Significance

The return probability distribution reveals fundamental aspects of human behavior:

| Aspect | What It Reveals |
|--------|-----------------|
| **Periodicity** | Daily, weekly patterns (circadian rhythms) |
| **Predictability** | How foreseeable human movement is |
| **Memory** | Whether people "remember" to return to important places |
| **Regularity** | Consistency of routines across populations |

### 3.2 Practical Applications

| Application | Use Case |
|-------------|----------|
| **Location Prediction** | Predicting where users will go next |
| **Urban Planning** | Understanding traffic patterns and facility usage |
| **Epidemiology** | Modeling disease spread through movement |
| **Recommendation Systems** | Location-based services and advertising |
| **Transportation** | Optimizing public transit schedules |

### 3.3 Connection to Machine Learning

📌 **Key Insight for Model Design**:

The high return probability (~80%+) suggests that:
- **Pointer mechanisms** are ideal for next location prediction
- Models should prioritize copying from history over generating new locations
- Temporal features (especially 24-hour cycles) are critical

---

## 4. Datasets Analyzed

### 4.1 Geolife Dataset

```
┌─────────────────────────────────────────────────────┐
│ GEOLIFE DATASET                                     │
├─────────────────────────────────────────────────────┤
│ Source:      Microsoft Research Asia                │
│ Period:      April 2007 - August 2012               │
│ Location:    Beijing, China (primarily)             │
│ Collection:  GPS trajectories                       │
│ Epsilon:     20 meters (clustering parameter)       │
├─────────────────────────────────────────────────────┤
│ Total Events:     19,191                            │
│ Total Users:      91                                │
│ Unique Locations: 2,049                             │
│ Return Rate:      53.85% (49 users)                 │
│ Mean Return Time: 58.96 hours                       │
└─────────────────────────────────────────────────────┘
```

### 4.2 DIY Dataset

```
┌─────────────────────────────────────────────────────┐
│ DIY DATASET                                         │
├─────────────────────────────────────────────────────┤
│ Source:      Custom data collection                 │
│ Period:      ~7.6 months                            │
│ Collection:  GPS/Location data                      │
│ Epsilon:     50 meters (clustering parameter)       │
├─────────────────────────────────────────────────────┤
│ Total Events:     265,621                           │
│ Total Users:      1,306                             │
│ Unique Locations: 8,439                             │
│ Return Rate:      83.54% (1,091 users)              │
│ Mean Return Time: 60.02 hours                       │
└─────────────────────────────────────────────────────┘
```

---

## 5. Analysis Overview

### 5.1 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ANALYSIS PIPELINE                                 │
└─────────────────────────────────────────────────────────────────────┘

 ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
 │  Raw GPS     │    │ Intermediate │    │   Return     │
 │  Trajectory  │───►│    CSV       │───►│   Times      │
 │  Data        │    │  (cleaned)   │    │  (per user)  │
 └──────────────┘    └──────────────┘    └──────────────┘
                                                │
                                                ▼
                     ┌──────────────┐    ┌──────────────┐
                     │   Plots      │◄───│  Histogram   │
                     │   (PNG)      │    │  & PDF       │
                     └──────────────┘    └──────────────┘

Step 1: Load intermediate data (preprocessed trajectories)
Step 2: Compute first return time for each user
Step 3: Build histogram with 2-hour bins
Step 4: Normalize to probability density
Step 5: Generate publication-quality plots
```

### 5.2 Output Files Generated

| File | Description | Format |
|------|-------------|--------|
| `*_return_probability.png` | Probability distribution plot | PNG (300 DPI) |
| `*_return_probability_v2.png` | Enhanced plot with RW baseline | PNG (300 DPI) |
| `*_return_probability_data.csv` | Histogram data (t, F_pt) | CSV |
| `*_return_probability_data_returns.csv` | Per-user return times | CSV |
| `comparison_return_probability.png` | Cross-dataset comparison | PNG (300 DPI) |

---

## 6. Key Results Summary

### 6.1 Return Statistics Comparison

```
┌─────────────────────────────────────────────────────────────────────┐
│                    RETURN STATISTICS                                 │
├───────────────────────┬─────────────────┬─────────────────┬─────────┤
│ Metric                │ Geolife         │ DIY             │ Diff    │
├───────────────────────┼─────────────────┼─────────────────┼─────────┤
│ Users with returns    │ 49 (53.85%)     │ 1,091 (83.54%)  │ +29.69% │
│ Mean return time      │ 58.96 h         │ 60.02 h         │ +1.06 h │
│ Median return time    │ 35.28 h         │ 42.77 h         │ +7.49 h │
│ Standard deviation    │ 65.62 h         │ 54.48 h         │ -11.14 h│
│ Peak probability at   │ 3 h             │ 23 h            │ +20 h   │
└───────────────────────┴─────────────────┴─────────────────┴─────────┘
```

### 6.2 Key Observations

1. **High Return Rate**: Over 80% of DIY users return to their first location within 10 days

2. **Daily Periodicity**: DIY shows strong peak at ~24 hours, indicating daily routines

3. **Consistent Mean**: Both datasets show ~60-hour mean return time (~2.5 days)

4. **Non-Random Behavior**: Real user data significantly deviates from Random Walk model

5. **Periodic Spikes**: Clear 24-hour periodicity visible in Geolife data

---

## 7. Reading the Plots

### 7.1 Main Elements

```
                        ┌──────────┐
                        │ Legend   │
                        │ -------- │
                        │ --- Users│
                        │ ── RW    │
                        └──────────┘
     ▲
F_pt(t)
0.025│     
     │   ∿∿∿     Users (observed data)
0.020│  ∿    ∿   - Blue dashed line
     │ ∿      ∿  - Shows actual return probability
0.015│∿        ∿ 
     │          ∿∿∿∿
0.010│╲              ∿∿∿∿∿
     │ ╲  RW baseline       ∿∿∿∿∿∿
0.005│  ╲ - Black solid line        ∿∿∿
     │   ╲╲- Exponential decay model    ∿∿∿
     │     ╲╲╲_______________
0.000├─────────────────────────────────────► t (h)
     0    24    48    72    96   120  ...  240
```

### 7.2 How to Interpret

| Pattern | Meaning |
|---------|---------|
| **Peaks at 24h, 48h, 72h...** | Daily routine patterns |
| **Higher than RW baseline** | Non-random, intentional returns |
| **Decay over time** | Returns become less likely as time passes |
| **Sharp spikes** | Strong periodic behavior |

---

## 8. Connection to Proposed Model

### 8.1 Why This Matters for Pointer Networks

The analysis reveals that:

```
┌─────────────────────────────────────────────────────────────────────┐
│ INSIGHT: Human mobility is dominated by RETURNS to known places    │
└─────────────────────────────────────────────────────────────────────┘

This directly supports the Pointer Network architecture:

  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                  │
  │  Observation: ~80% of next locations are RETURNS               │
  │                                                                  │
  │  ───────────────────────────────────────────────────────────►  │
  │                                                                  │
  │  Model Design: Pointer mechanism to "copy" from history         │
  │                                                                  │
  └─────────────────────────────────────────────────────────────────┘
```

### 8.2 Design Justification

The Pointer Network V45 model includes:

1. **Pointer Mechanism**: Copies locations from user history (justified by high return rate)
2. **Position-from-End Embedding**: Encodes recency (recent locations more likely to be revisited)
3. **Temporal Features**: Captures 24-hour periodicity (justified by daily return peaks)
4. **Adaptive Gate**: Balances between copying and generation (handles both returns and new visits)

---

## 9. Next Steps

Continue reading:

1. **[02_THEORETICAL_BACKGROUND.md](02_THEORETICAL_BACKGROUND.md)** - Deep dive into the science
2. **[03_CODE_WALKTHROUGH.md](03_CODE_WALKTHROUGH.md)** - Understand the implementation
3. **[07_PLOT_ANALYSIS.md](07_PLOT_ANALYSIS.md)** - Detailed plot interpretation
4. **[08_MODEL_JUSTIFICATION.md](08_MODEL_JUSTIFICATION.md)** - Full model justification

---

## 10. Quick Reference

### Running the Analysis

```bash
# Activate environment
conda activate mlenv

# Navigate to project root
cd /data/next_loc_clean_v2

# Run analysis
python scripts/analysis_returner/return_probability_analysis_v2.py

# Create comparison plot
cd scripts/analysis_returner
python compare_datasets.py
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--bin-width` | 2.0 | Histogram bin width in hours |
| `--max-hours` | 240 | Maximum return time (10 days) |
| `--output-dir` | `scripts/analysis_returner` | Output directory |

---

*← Back to [Index](00_INDEX.md) | Continue to [Theoretical Background](02_THEORETICAL_BACKGROUND.md) →*
