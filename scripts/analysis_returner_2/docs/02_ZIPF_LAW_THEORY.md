# Zipf's Law Theory: Mathematical Foundation

## 1. Introduction to Zipf's Law

### 1.1 Historical Background

**Zipf's Law** was originally discovered by George Kingsley Zipf in 1949 while studying word frequencies in natural language. He found that the frequency of any word is inversely proportional to its rank in the frequency table.

```
┌─────────────────────────────────────────────────────────────┐
│  ZIPF'S LAW (General Form)                                  │
│                                                              │
│  f(r) ∝ r^(-α)                                              │
│                                                              │
│  Where:                                                      │
│    f(r) = frequency/probability of rank r                   │
│    r    = rank (1, 2, 3, ...)                               │
│    α    = Zipf exponent (typically ≈ 1)                     │
└─────────────────────────────────────────────────────────────┘
```

### 1.2 Examples in Nature

Zipf's Law appears in many natural and social phenomena:

| Domain | Observation |
|--------|-------------|
| **Language** | Word frequency follows rank^(-1) |
| **Cities** | Population follows rank^(-1) |
| **Websites** | Page visits follow rank^(-1) |
| **Income** | Wealth distribution (Pareto) |
| **Earthquakes** | Magnitude frequency |
| **Human Mobility** | Location visits follow rank^(-1) |

---

## 2. Zipf's Law in Human Mobility

### 2.1 González et al. (2008) Discovery

The seminal paper "Understanding Individual Human Mobility Patterns" by González, Hidalgo, and Barabási (Nature, 2008) discovered that **location visit frequency follows Zipf's Law**.

Using mobile phone data from 100,000 users over 6 months, they found:

```
P(L) ≈ c × L^(-1)
```

Where:
- **P(L)** = probability of visiting the L-th most visited location
- **L** = rank of location by visit frequency
- **c** = normalization constant

### 2.2 Physical Interpretation

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  WHAT ZIPF'S LAW MEANS FOR HUMAN MOBILITY                       │
│                                                                  │
│  • L=1 (Home/Work): Users spend MOST of their time here         │
│    → Typical: 30-65% of all visits                              │
│                                                                  │
│  • L=2,3 (Regular places): Significant but less frequent        │
│    → Typical: 15-25% combined                                   │
│                                                                  │
│  • L=4+ (Occasional places): Long tail of infrequent visits     │
│    → Many locations with few visits each                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 Mathematical Properties

#### Property 1: Heavy Tail
The L^(-1) decay is slow enough that rare locations are still visited:

```
   Probability
   1.0 │●
       │ ●
   0.5 │  ●
       │   ●
   0.2 │    ●●
       │      ●●●
   0.1 │         ●●●●●●●●●●●●●●●●●●●●  ← Long tail
       └──────────────────────────────
       1  2  3  4  5 ...  20    50
                 Rank L
```

#### Property 2: Concentration
Top locations dominate total visits:

| Top N Locations | Cumulative Probability |
|-----------------|----------------------|
| Top 1 | ~30-65% |
| Top 3 | ~60-80% |
| Top 5 | ~75-90% |
| Top 10 | ~85-95% |

#### Property 3: Scale Invariance
The pattern holds regardless of how many locations a user visits:

```
Users with 5 locations:   P(L) ≈ c₁ × L^(-1)
Users with 10 locations:  P(L) ≈ c₂ × L^(-1)
Users with 50 locations:  P(L) ≈ c₃ × L^(-1)

Same exponent (-1), different constants (c₁, c₂, c₃)
```

---

## 3. Mathematical Formulation

### 3.1 Per-User Probability

For user **u** with locations ranked by visit frequency:

```
         visits to location at rank L for user u
p_u(L) = ─────────────────────────────────────────
              total visits for user u
```

**Properties:**
- Σ_L p_u(L) = 1 (probabilities sum to 1)
- p_u(1) ≥ p_u(2) ≥ p_u(3) ≥ ... (by definition of ranking)

### 3.2 Group Averaging

Users are grouped by **n_L** (number of unique locations visited):

For group G with target n_L:

```
              1
P_G(L) = ────────── × Σ p_u(L)
          |G|       u∈G
```

**Standard Error:**

```
             σ_G(L)
SE_G(L) = ──────────
          √|G|

where σ_G(L) = standard deviation of p_u(L) across users in G
```

### 3.3 Reference Line Fitting

We fit the model P(L) = c × L^(-1) in log-space:

```
log(P(L)) = log(c) - log(L)

Solving by least squares:
log(c) = mean[log(P) + log(L)]
c = exp(log(c))
```

**Fit range:** L = 3 to 10 (avoids outliers at L=1 and noise at large L)

---

## 4. Derivation from First Principles

### 4.1 Why L^(-1)?

Several theories explain why human mobility follows Zipf's Law:

#### Theory 1: Exploration-Exploitation Tradeoff

Humans balance:
- **Exploitation**: Return to known, preferred locations
- **Exploration**: Discover new locations occasionally

This produces a preferential return pattern that naturally generates power-law distributions.

#### Theory 2: Time Budget Constraints

Given limited time:
- High-priority locations (home, work) get most visits
- Remaining time distributed among decreasing-priority locations
- This hierarchical allocation produces Zipf-like behavior

#### Theory 3: Gravity Model

Visit frequency to location L depends on:
- Attractiveness of L (utility)
- Distance/cost to reach L

When attractiveness follows a power-law distribution, so do visits.

### 4.2 Connection to Other Mobility Laws

Zipf's Law connects to other established mobility patterns:

```
┌─────────────────────────────────────────────────────────────┐
│  HUMAN MOBILITY LAWS                                         │
│                                                              │
│  1. ZIPF'S LAW:        P(L) ∝ L^(-1)                        │
│     Location visit frequency                                 │
│                                                              │
│  2. DISPLACEMENT:      P(Δr) ∝ Δr^(-β) × exp(-Δr/κ)         │
│     Travel distance distribution                             │
│                                                              │
│  3. RADIUS OF GYRATION: P(r_g) ∝ r_g^(-γ)                   │
│     Characteristic travel distance per user                  │
│                                                              │
│  4. WAITING TIME:       P(Δt) ∝ Δt^(-δ)                     │
│     Time between movements                                   │
│                                                              │
│  All follow power-law distributions!                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 5. Implications for Prediction

### 5.1 Why This Matters for ML

The Zipf's Law finding has **critical implications** for machine learning:

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  IF: Most visits (60-80%) go to top 3 locations             │
│                                                              │
│  THEN: Effective prediction should:                          │
│        1. PRIORITIZE previously visited locations            │
│        2. Give HIGHER WEIGHT to frequently visited           │
│        3. Use RECENT HISTORY as primary signal               │
│                                                              │
│  SOLUTION: POINTER MECHANISM                                 │
│        → Copies directly from input sequence                 │
│        → Naturally favors recent/frequent locations          │
│        → Perfect fit for Zipf-distributed data!              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Prediction Strategies

| Strategy | Description | Effectiveness |
|----------|-------------|---------------|
| **Random baseline** | Predict any location uniformly | Very poor |
| **Global frequency** | Predict most popular globally | Poor |
| **User frequency** | Predict user's most visited | Good (~30-40% acc) |
| **Sequential** | Use sequence patterns | Better (~40-50% acc) |
| **Pointer Network** | Copy from history with attention | **Best** (~50-60% acc) |

The pointer mechanism aligns perfectly with Zipf's Law by allowing the model to **directly select from the user's visited locations**.

---

## 6. Statistical Verification

### 6.1 Goodness of Fit

To verify Zipf's Law, we check:

1. **Log-log linearity**: Plot should be approximately linear
2. **Slope ≈ -1**: The exponent should be close to -1
3. **R² value**: High R² indicates good fit

**Results from our analysis:**

| Dataset | Coefficient c | Approximate R² |
|---------|--------------|----------------|
| Geolife | 0.222 | ~0.85 |
| DIY | 0.150 | ~0.92 |

### 6.2 Deviations from Pure Zipf

The data shows slight deviations:

1. **L=1 excess**: Top location often has higher P(L) than predicted
2. **Tail cutoff**: Very high ranks show faster decay (finite vocabulary effect)
3. **Group variation**: Different n_L groups have different intercepts

These deviations are typical and expected for empirical Zipf distributions.

---

## 7. Summary

### Key Takeaways

1. **Zipf's Law** describes location visit frequency: P(L) ∝ L^(-1)

2. **Top-heavy distribution**: Most visits go to few locations

3. **Universal pattern**: Holds across users, datasets, and time periods

4. **Model implication**: Pointer mechanisms are ideal for this distribution

### Mathematical Summary

```
┌─────────────────────────────────────────────────────────────┐
│                                                              │
│  ZIPF'S LAW FOR LOCATION VISITS                             │
│                                                              │
│  P(L) = c × L^(-1)                                          │
│                                                              │
│  Where:                                                      │
│    P(L) = probability of visiting rank-L location           │
│    L    = rank (1=most visited)                             │
│    c    = dataset-specific constant (0.15-0.22)             │
│                                                              │
│  Consequence:                                                │
│    Top 3 locations ≈ 60-80% of visits                       │
│    → POINTER MECHANISM is well-suited!                      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

---

*Next: [03_CODE_WALKTHROUGH.md](./03_CODE_WALKTHROUGH.md) - Detailed code explanation*
