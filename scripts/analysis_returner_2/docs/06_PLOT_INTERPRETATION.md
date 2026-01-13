# Plot Interpretation Guide

## 1. Overview of Generated Plots

This analysis generates three types of plots:

| Plot | File | Purpose |
|------|------|---------|
| Geolife Zipf | `geolife_zipf_location_frequency.png` | Zipf analysis for Geolife |
| DIY Zipf | `diy_zipf_location_frequency.png` | Zipf analysis for DIY |
| Comparison | `comparison_zipf_location_frequency.png` | Side-by-side comparison |

---

## 2. Understanding the Main Plot

### 2.1 Plot Structure

Each individual Zipf plot (Geolife and DIY) has two components:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                          │
│                              ┌──────────────────────┐                   │
│                              │      INSET           │                   │
│                              │   (Linear scale)     │                   │
│                              │   Ranks 1-6 only     │                   │
│                              │   With error bars    │                   │
│                              └──────────────────────┘                   │
│                                                                          │
│   MAIN PANEL (Log-log scale)                                            │
│   All ranks shown                                                        │
│   Reference line L^(-1)                                                  │
│                                                                          │
│   Legend:                                                                │
│   ○ 5 loc.  □ 10 loc.  ◇ 30 loc.  △ 50 loc.                            │
│   ─── Reference line                                                    │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Axis Interpretation

**Main Panel (Log-Log):**

```
         Y-axis: P(L) - Probability of visiting rank-L location
         │
    10⁰ ─┤  (= 1.0 = 100%)
         │
   10⁻¹ ─┤  (= 0.1 = 10%)    ← Most data points here
         │
   10⁻² ─┤  (= 0.01 = 1%)
         │
   10⁻³ ─┤  (= 0.001 = 0.1%)
         │
         └───────────────────────────────────────
              10⁰    10¹    10²
              (1)   (10)   (100)
                    
         X-axis: L - Rank of location (1 = most visited)
```

**Reading the axes:**
- **X-axis (L)**: Location rank. L=1 is the most visited location.
- **Y-axis (P(L))**: Probability of visiting that location. Higher = more visits.

---

## 3. How to Read the Main Panel

### 3.1 The Data Points

Each data point represents:
- **X-coordinate**: Rank L (1, 2, 3, ...)
- **Y-coordinate**: Average P(L) across all users in that group

```
Example: Point at (L=3, P=0.1) means:
"On average, users visit their 3rd most frequent location 10% of the time"
```

### 3.2 The Groups (Different Markers)

| Marker | Color | Group | Users with... |
|--------|-------|-------|---------------|
| ○ (circle) | Black | 5 loc. | 4-6 unique locations |
| □ (square) | Red | 10 loc. | 8-12 unique locations |
| ◇ (diamond) | Green | 30 loc. | 25-35 unique locations |
| △ (triangle) | Blue | 50 loc. | 45-55 unique locations |

### 3.3 The Reference Line

The solid black line represents **Zipf's Law**: P(L) = c × L^(-1)

```
Reference Line Interpretation:
───────────────────────────────────────────────────────
If data follows line → Zipf's Law holds
If data above line  → Higher than Zipf predicts
If data below line  → Lower than Zipf predicts
───────────────────────────────────────────────────────
```

### 3.4 What Good Fit Looks Like

```
LOG-LOG PLOT: Good Zipf Fit         LOG-LOG PLOT: Poor Fit

P(L)                                P(L)
 │●                                  │   ●
 │ ●                                 │  ●
 │  ●                                │ ●
 │   ●                               │●
 │    ●────── Good: follows line     │●
 │     ●                             │ ●●───── Poor: curved
 │      ●                            │    ●●●
 │       ●                           │        ●●●
 └─────────────                      └─────────────────
       L (log scale)                       L (log scale)
```

---

## 4. How to Read the Inset

### 4.1 Inset Purpose

The inset shows the **top 6 locations** (L=1 to L=6) on a **linear scale** with **error bars**.

```
Why an inset?
──────────────────────────────────────────────────────────────
Log-log plots compress high-probability values near P=1
The inset lets you see:
  • Actual P(L) values (not compressed)
  • Differences between groups clearly
  • Error bars (uncertainty)
──────────────────────────────────────────────────────────────
```

### 4.2 Reading the Inset

```
       INSET (Linear Scale)
     
 P(L)
 0.6 ┤●───┐    
     │    │    Error bar
 0.5 ┤    └──● (standard error)
     │
 0.4 ┤  □────□
     │
 0.3 ┤     ◇────◇
     │
 0.2 ┤        △────△
     │
 0.1 ┤ ●──□──◇──△─────
     │                 
 0.0 ┼─────────────────
     1    2    3    4    5    6
              L (rank)
```

### 4.3 Interpreting Error Bars

```
Error bars show STANDARD ERROR: SE = σ / √n

Large error bar:    Small error bar:
       ●                  ●
       │                  │
       │                  │
       │                 Meaning:
       │                 • More users in group
       │                 • More precise estimate
Meaning:              
• Fewer users in group
• Less precise estimate
• Results may vary
```

---

## 5. Reading the Actual Plots

### 5.1 DIY Zipf Plot Analysis

Looking at `diy_zipf_location_frequency.png`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  DIY ZIPF PLOT INTERPRETATION                                            │
│                                                                          │
│  MAIN PANEL OBSERVATIONS:                                               │
│  ─────────────────────────                                              │
│  1. All four groups (○□◇△) follow approximately the same line          │
│     → Zipf's Law holds across all group sizes                           │
│                                                                          │
│  2. Data points are ABOVE reference line at L=1                         │
│     → Top location is even MORE visited than Zipf predicts              │
│     → This is the "home bias" effect                                    │
│                                                                          │
│  3. Data points converge at higher ranks (L>10)                         │
│     → All groups show similar behavior for less-visited places          │
│                                                                          │
│  4. Points follow line well for L=3 to L=30                             │
│     → Good fit in the "mid-rank" region                                 │
│                                                                          │
│  INSET OBSERVATIONS:                                                    │
│  ───────────────────                                                    │
│  1. ○ (5 loc.) has highest P(1) ≈ 0.64                                  │
│     → Users with few locations concentrate on home                      │
│                                                                          │
│  2. Error bars are SMALL for DIY                                        │
│     → Large sample sizes (95-230 users per group)                       │
│     → High confidence in estimates                                      │
│                                                                          │
│  3. All curves decay steeply from L=1 to L=2                            │
│     → Confirms Zipf-like decay                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.2 Geolife Zipf Plot Analysis

Looking at `geolife_zipf_location_frequency.png`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  GEOLIFE ZIPF PLOT INTERPRETATION                                        │
│                                                                          │
│  MAIN PANEL OBSERVATIONS:                                               │
│  ─────────────────────────                                              │
│  1. More scattered than DIY (fewer users → more noise)                  │
│  2. Still follows approximate L^(-1) trend                              │
│  3. Some outlier points (especially □ 10 loc. group)                    │
│     → Small group sizes cause variability                               │
│                                                                          │
│  INSET OBSERVATIONS:                                                    │
│  ───────────────────                                                    │
│  1. LARGE error bars (especially ○ 5 loc. with only 4 users)           │
│     → Results are less certain than DIY                                 │
│                                                                          │
│  2. P(1) values are LOWER than DIY                                      │
│     → Geolife users more exploratory                                    │
│                                                                          │
│  3. Despite noise, same general pattern visible                         │
│     → Zipf's Law is robust                                              │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 Comparison Plot Analysis

Looking at `comparison_zipf_location_frequency.png`:

```
┌─────────────────────────────────────────────────────────────────────────┐
│  COMPARISON PLOT INTERPRETATION                                          │
│                                                                          │
│  LEFT PANEL (Geolife):          RIGHT PANEL (DIY):                      │
│  • Coefficient c = 0.222        • Coefficient c = 0.150                 │
│  • More scatter                 • Less scatter                          │
│  • Fewer users                  • More users                            │
│                                                                          │
│  KEY COMPARISON:                                                        │
│  ─────────────────                                                      │
│  1. BOTH follow L^(-1) pattern                                          │
│     → Universal mobility law confirmed                                  │
│                                                                          │
│  2. DIY has STEEPER decay (lower c)                                     │
│     → More concentration on top locations                               │
│                                                                          │
│  3. DIY has SMOOTHER curves                                             │
│     → Better statistics, more reliable                                  │
│                                                                          │
│  4. SAME SHAPE in both panels                                           │
│     → Fundamental pattern is universal                                  │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Key Visual Patterns to Look For

### 6.1 Signs of Zipf's Law

```
✓ GOOD: Linear trend on log-log scale
✓ GOOD: Slope approximately -1
✓ GOOD: All groups follow similar pattern
✓ GOOD: High P(1) decaying to low P(large L)
```

### 6.2 Expected Deviations

```
⚠ EXPECTED: P(1) above reference line (home bias)
⚠ EXPECTED: More scatter at high ranks (few data points)
⚠ EXPECTED: Cutoff at high ranks (finite vocabulary)
⚠ EXPECTED: Groups start at different P(1) but converge
```

### 6.3 Warning Signs (Not Seen Here)

```
✗ BAD: Curved line on log-log scale (not power-law)
✗ BAD: Flat regions (saturation)
✗ BAD: Groups not following same pattern
✗ BAD: P(1) < P(2) (impossible by definition)
```

---

## 7. Extracting Numbers from Plots

### 7.1 Reading P(L) Values

To read approximate values from the main panel:

```
Step 1: Find the data point
Step 2: Drop vertical line to X-axis → Read L
Step 3: Draw horizontal line to Y-axis → Read P(L)

Example: For DIY, n_L=5 group (○):
  - Point at L=1: P(1) ≈ 0.6 (60%)
  - Point at L=2: P(2) ≈ 0.25 (25%)
  - Point at L=3: P(3) ≈ 0.08 (8%)
```

### 7.2 Exact Values from Data Files

For precise numbers, use the CSV files:

```python
# Read exact values
import pandas as pd
stats = pd.read_csv('diy_zipf_data_stats.csv')

# Get P(1) for 5 loc. group
p1 = stats[(stats['n_locations_group'] == 5) & (stats['rank'] == 1)]
print(p1['mean_prob'])  # → 0.6426
```

---

## 8. Summary: What the Plots Tell Us

### 8.1 Main Conclusions

```
┌─────────────────────────────────────────────────────────────────────────┐
│  WHAT THE PLOTS SHOW                                                     │
│                                                                          │
│  1. ZIPF'S LAW IS REAL                                                  │
│     → Log-log linearity with slope ≈ -1                                 │
│     → Confirmed for both datasets                                       │
│                                                                          │
│  2. TOP LOCATION DOMINATES                                              │
│     → P(1) = 30-65% of visits                                           │
│     → Clear separation from other ranks                                 │
│                                                                          │
│  3. PATTERN IS UNIVERSAL                                                │
│     → All groups (5, 10, 30, 50 loc.) follow same law                  │
│     → Both datasets show same behavior                                  │
│                                                                          │
│  4. PREDICTABLE MOBILITY                                                │
│     → Most visits to few locations                                      │
│     → Supports pointer mechanism design                                 │
└─────────────────────────────────────────────────────────────────────────┘
```

### 8.2 How to Report These Results

For PhD thesis or publication:

```
"Figure X shows that location visit frequency follows Zipf's law 
(P(L) ∝ L^(-1)) in both datasets. The most visited location accounts 
for 31-64% of visits, with the top 3 locations covering 65-75% of 
all visits. This concentration of visits on a small number of 
locations motivates the use of a pointer mechanism that can directly 
copy from the user's location history."
```

---

*Next: [07_GEOLIFE_RESULTS.md](./07_GEOLIFE_RESULTS.md) - Detailed Geolife results*
