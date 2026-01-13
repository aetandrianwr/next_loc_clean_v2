# Appendix: Mathematical Foundations, Glossary, and References

## A. Mathematical Foundations

### A.1 Probability Density Function (PDF)

📐 **Definition**: A probability density function f(x) is a function that describes the relative likelihood of a continuous random variable X taking on a given value x.

**Properties**:
1. Non-negativity: f(x) ≥ 0 for all x
2. Normalization: ∫_{-∞}^{∞} f(x) dx = 1
3. Probability calculation: P(a ≤ X ≤ b) = ∫_{a}^{b} f(x) dx

**In our context**:
- F_pt(t) is the PDF of first-return times
- F_pt(t) ≥ 0 for all t ≥ 0
- ∫_{0}^{∞} F_pt(t) dt = 1

### A.2 Histogram to PDF Conversion

📐 **Formula**:

Given N observations {x₁, x₂, ..., xₙ} and bins of width Δx:

```
           count_i
f(x_i) = ─────────────
          N × Δx

Where:
  count_i = number of observations in bin i
  N = total number of observations
  Δx = bin width
```

**Proof of normalization**:
```
Σᵢ f(x_i) × Δx = Σᵢ (count_i / (N × Δx)) × Δx
               = Σᵢ count_i / N
               = N / N
               = 1 ✓
```

### A.3 Exponential Distribution (RW Baseline)

📐 **Definition**: The exponential distribution with rate parameter λ has PDF:

```
f(t) = λ × exp(-λt)   for t ≥ 0
```

**Properties**:
- Mean: E[T] = 1/λ
- Variance: Var(T) = 1/λ²
- Memoryless: P(T > t + s | T > s) = P(T > t)

**In our context**:
- RW baseline uses τ = 1/λ = 30 hours
- F_RW(t) = P₀ × exp(-t/τ) where P₀ is a scaling factor

### A.4 Kolmogorov-Smirnov Test

📐 **Definition**: The KS test compares the empirical CDF with a theoretical CDF.

**Test Statistic**:
```
D = max_x |F_empirical(x) - F_theoretical(x)|
```

**Interpretation**:
- Small D → data matches theoretical distribution
- Large D → data differs from theoretical distribution
- p-value < 0.05 → reject null hypothesis (distributions differ)

---

## B. Statistical Formulas

### B.1 Central Tendency

```
Mean:       μ = (1/n) × Σᵢ xᵢ

Median:     M = x_{(n+1)/2}          if n is odd
            M = (x_{n/2} + x_{n/2+1})/2   if n is even

Mode:       Most frequent value (or bin center for continuous data)
```

### B.2 Dispersion

```
Variance:   σ² = (1/(n-1)) × Σᵢ (xᵢ - μ)²

Standard Deviation: σ = √σ²

Interquartile Range: IQR = Q3 - Q1

Coefficient of Variation: CV = σ / μ
```

### B.3 Quantiles

```
Q1 (25th percentile): Value below which 25% of data falls
Q2 (50th percentile): Median
Q3 (75th percentile): Value below which 75% of data falls
```

### B.4 Skewness

```
Skewness = E[(X - μ)³] / σ³

Interpretation:
  > 0: Right-skewed (tail extends right)
  = 0: Symmetric
  < 0: Left-skewed (tail extends left)

Our data: Both datasets show positive skewness (right-skewed)
```

---

## C. Algorithm Complexity Analysis

### C.1 Time Complexity Derivations

**Sorting** (Step 1):
```
Using TimSort (Python's default): O(n log n)
Where n = total number of events
```

**GroupBy** (Steps 2, 6):
```
Pandas groupby: O(n) amortized
Uses hash-based grouping
```

**Merge** (Step 3):
```
Pandas merge: O(n + m) where n, m are table sizes
Uses sort-merge or hash-merge algorithm
```

**Filter** (Steps 4, 5):
```
Boolean indexing: O(n)
Single pass through data
```

**Histogram** (Step 7):
```
NumPy histogram: O(n + b)
Where b = number of bins
```

**Total**: O(n log n) dominated by sorting

### C.2 Space Complexity

```
Input DataFrame:          O(n) for n events
Sorted DataFrame:         O(n) copy
First Events Table:       O(u) for u users
Merged DataFrame:         O(n)
Filtered DataFrames:      O(n) worst case
Return Times:             O(u)
Histogram:                O(b) for b bins

Total: O(n) dominated by DataFrames
```

---

## D. Glossary of Terms

### D.1 General Terms

| Term | Definition |
|------|------------|
| **GPS** | Global Positioning System; satellite-based navigation system |
| **Trajectory** | Sequence of time-stamped locations for a single user |
| **Staypoint** | Location where user spent significant time (not just passing through) |
| **DBSCAN** | Density-Based Spatial Clustering; used to identify staypoints |
| **Epsilon (ε)** | Distance parameter for DBSCAN clustering |

### D.2 Analysis-Specific Terms

| Term | Definition |
|------|------------|
| **First Location (L₀)** | The first recorded location for a user |
| **First Return Time** | Time until user returns to their first location |
| **Return Rate** | Percentage of users who return within observation window |
| **F_pt(t)** | Probability density of first return at time t |
| **RW (Random Walk)** | Baseline model assuming random movement |
| **Bin Width** | Width of histogram bins (default: 2 hours) |

### D.3 Model-Specific Terms

| Term | Definition |
|------|------------|
| **Pointer Network** | Neural architecture that "points" to input elements |
| **Pointer Mechanism** | Attention over input sequence to copy locations |
| **Generation Head** | Network component that predicts from full vocabulary |
| **Pointer-Gen Gate** | Learned parameter balancing pointer vs generation |
| **Temporal Embedding** | Vector representation of time-related features |
| **Position-from-End** | Encoding of recency (distance from sequence end) |

### D.4 Statistical Terms

| Term | Definition |
|------|------------|
| **PDF** | Probability Density Function |
| **CDF** | Cumulative Distribution Function |
| **IQR** | Interquartile Range (Q3 - Q1) |
| **CV** | Coefficient of Variation (σ/μ) |
| **KS Test** | Kolmogorov-Smirnov test for distribution comparison |
| **Circadian** | Related to 24-hour biological rhythms |

---

## E. Code Reference

### E.1 Key Functions

```python
# Load data
def load_intermediate_data(dataset_path):
    """Load and preprocess intermediate CSV data."""
    
# Compute return times
def compute_first_return_times(df, bin_width_hours, max_hours):
    """Compute first-return time for each user."""
    
# Build histogram
def compute_probability_density(delta_t_values, bin_width_hours, max_hours):
    """Convert return times to probability density."""
    
# Random walk baseline
def compute_random_walk_baseline(bin_centers, initial_prob):
    """Compute exponential decay baseline."""
    
# Plotting
def plot_return_probability(bin_centers, pdf, dataset_name, output_path):
    """Generate publication-quality plot."""
```

### E.2 Command Line Usage

```bash
# Basic usage
python return_probability_analysis_v2.py

# Custom parameters
python return_probability_analysis_v2.py \
    --bin-width 1.0 \
    --max-hours 120 \
    --output-dir results/

# Create comparison plot
python compare_datasets.py
```

### E.3 Configuration Parameters

| Parameter | Type | Default | Range |
|-----------|------|---------|-------|
| `bin_width` | float | 2.0 | 0.5-10.0 |
| `max_hours` | int | 240 | 24-720 |
| `output_dir` | str | `scripts/analysis_returner` | any path |
| `tau` (RW) | float | 30.0 | 10-100 |
| `P0` (RW) | float | 0.01 | 0.001-0.1 |

---

## F. Data Format Specifications

### F.1 Input CSV Schema

```
Column      Type      Range           Description
────────────────────────────────────────────────────────────
user_id     int/str   any             Unique user identifier
location_id int       1 to N_locs     Encoded location (0=padding)
start_day   int       0 to N_days     Day count from start
start_min   int       0 to 1439       Minute of day
```

### F.2 Output CSV Schemas

**Return Times CSV**:
```
Column         Type    Description
────────────────────────────────────────────────
user_id        int     User identifier
delta_t_hours  float   First return time in hours
```

**Probability Density CSV**:
```
Column    Type    Description
────────────────────────────────────────────────
t_hours   float   Bin center (time in hours)
F_pt      float   Probability density value
```

---

## G. References

### G.1 Primary Literature

1. **González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008)**.
   Understanding individual human mobility patterns.
   *Nature*, 453(7196), 779-782.
   https://doi.org/10.1038/nature06958

2. **Song, C., Qu, Z., Blumm, N., & Barabási, A.-L. (2010)**.
   Limits of predictability in human mobility.
   *Science*, 327(5968), 1018-1021.
   https://doi.org/10.1126/science.1177170

3. **Pappalardo, L., Simini, F., Rinzivillo, S., et al. (2015)**.
   Returners and explorers dichotomy in human mobility.
   *Nature Communications*, 6, 8166.
   https://doi.org/10.1038/ncomms9166

### G.2 Method References

4. **Vinyals, O., Fortunato, M., & Jaitly, N. (2015)**.
   Pointer Networks.
   *Advances in Neural Information Processing Systems*, 28.

5. **Vaswani, A., Shazeer, N., Parmar, N., et al. (2017)**.
   Attention is All You Need.
   *Advances in Neural Information Processing Systems*, 30.

6. **See, A., Liu, P. J., & Manning, C. D. (2017)**.
   Get To The Point: Summarization with Pointer-Generator Networks.
   *ACL 2017*.

### G.3 Dataset References

7. **Zheng, Y., Li, Q., Chen, Y., et al. (2008)**.
   Understanding mobility based on GPS data.
   *UbiComp 2008*.

8. **Zheng, Y., Xie, X., & Ma, W.-Y. (2010)**.
   GeoLife: A Collaborative Social Networking Service.
   *IEEE Internet Computing*.

### G.4 Location Prediction Literature

9. **Feng, J., Li, Y., Zhang, C., et al. (2018)**.
   DeepMove: Predicting Human Mobility with Attentional Recurrent Networks.
   *WWW 2018*.

10. **Luca, M., Barlacchi, G., Lepri, B., & Pappalardo, L. (2021)**.
    A Survey on Deep Learning for Human Mobility.
    *ACM Computing Surveys*, 55(1).

---

## H. Troubleshooting Guide

### H.1 Common Issues

**Issue**: Empty return times DataFrame
```
Cause: No users returned to first location
Solution: Check data quality, extend max_hours, verify location encoding
```

**Issue**: Probability doesn't sum to 1
```
Cause: Floating point precision or bin edge issues
Solution: Check bin edges, verify histogram computation
```

**Issue**: Plot axis limits incorrect
```
Cause: Outliers or mismatched parameters
Solution: Check ylim, adjust max_hours parameter
```

### H.2 Data Quality Checks

```python
# Check for missing values
assert df[['user_id', 'location_id', 'start_day', 'start_min']].notna().all().all()

# Check for valid ranges
assert df['start_min'].between(0, 1439).all()
assert df['start_day'].ge(0).all()
assert df['location_id'].gt(0).all()  # 0 is padding

# Check for sufficient data
assert len(df) > 0
assert df['user_id'].nunique() > 0
```

---

## I. Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | Jan 2026 | Initial documentation |
| 1.0.1 | Jan 2026 | Added model justification details |
| 1.0.2 | Jan 2026 | Added worked examples |

---

## J. Contact and Support

For questions about this analysis:
1. Review this documentation thoroughly
2. Check the troubleshooting guide
3. Examine the source code comments
4. Consult the primary literature references

---

*← Back to [Examples](09_EXAMPLES.md) | Return to [Index](00_INDEX.md) →*
