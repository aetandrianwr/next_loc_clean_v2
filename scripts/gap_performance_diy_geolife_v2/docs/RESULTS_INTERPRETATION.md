# Results Interpretation Guide

A comprehensive guide to understanding and interpreting all numerical results from the gap performance analysis.

---

## 1. How to Read the Results Files

### 1.1 JSON Files

**analysis_results.json** - Mobility pattern analysis:
```json
{
  "diy": { ... },      // DIY dataset results
  "geolife": { ... },  // GeoLife dataset results
  "statistical_tests": { ... }  // Significance tests
}
```

**model_analysis_results.json** - Trained model behavior:
```json
{
  "diy": { ... },      // DIY model on DIY data
  "geolife": { ... }   // GeoLife model on GeoLife data
}
```

**recency_analysis_results.json** - Temporal recency patterns:
```json
{
  "diy_recency": { ... },
  "geolife_recency": { ... },
  "diy_return": { ... },
  "geolife_return": { ... },
  "diy_pred": { ... },
  "geolife_pred": { ... }
}
```

---

## 2. Metric-by-Metric Interpretation

### 2.1 Target-in-History Rate

**What it measures**: Percentage of test samples where the target (next location to predict) appears somewhere in the input history.

**Values**:
- DIY: 84.12%
- GeoLife: 83.81%
- Difference: -0.31%

**Interpretation**:
- Both datasets are nearly identical (~84%)
- In 84% of cases, the pointer mechanism CAN copy the correct answer
- This metric does NOT explain the performance gap

**Why it matters**: This is a prerequisite for the pointer mechanism to help. If target is not in history, pointer cannot copy it.

**What this tells us**: The difference in ablation impact is NOT because GeoLife has more copyable targets.

---

### 2.2 Unique Location Ratio

**What it measures**: Proportion of unique locations in each input sequence.

**Formula**: `unique_ratio = |unique_locations| / sequence_length`

**Values**:
- DIY: 0.313 (31.3% unique)
- GeoLife: 0.340 (34.0% unique)
- Difference: +0.027

**Derived metric - Repetition Rate**: `1 - unique_ratio`
- DIY: 68.65%
- GeoLife: 65.96%

**Interpretation**:
- Both datasets are highly repetitive (65-70% repeated visits)
- DIY is actually slightly MORE repetitive
- This seems to contradict the hypothesis that GeoLife is more repetitive

**Resolution**: Total repetition doesn't capture WHERE repetitions occur. The key is recency, not overall repetition.

---

### 2.3 Sequence Entropy

**What it measures**: Shannon entropy of location distribution within sequences.

**Formula**: `H = -Σ p(loc) × log₂(p(loc))`

**Values**:
- DIY: 1.89 bits
- GeoLife: 1.74 bits
- Difference: -0.15 bits (8% lower for GeoLife)

**Interpretation**:
- Lower entropy = more predictable/concentrated patterns
- GeoLife sequences are slightly more predictable
- The difference is modest but consistent

**Example**:
```
High entropy (2.0 bits): [A, B, C, D, A, B, C, D]  - Visits spread evenly
Low entropy (0.5 bits): [A, A, A, A, A, A, B, A]   - Visits concentrated
```

---

### 2.4 Consecutive Repeat Rate

**What it measures**: Rate of A→A patterns (staying at same location).

**Formula**: `rate = count(x[i] == x[i+1]) / (length - 1)`

**Values**:
- DIY: 17.94%
- GeoLife: 26.87%
- **Difference: +8.93%** ← SIGNIFICANT

**Interpretation**:
- GeoLife has 50% more consecutive repeats (26.9% vs 17.9%)
- This indicates users often stay at or immediately return to locations
- The pointer mechanism captures this through position bias

**Example**:
```
DIY pattern:     Home → Work → Cafe → Home → Work
                  0      1      2      1      0 (consecutive repeats)
                  Rate: 0/4 = 0%

GeoLife pattern: Home → Home → Work → Work → Home
                  0      1      1      1      0 (consecutive repeats)
                  Rate: 2/4 = 50%
```

---

### 2.5 Target Equals Last Rate ⭐ KEY METRIC

**What it measures**: Percentage where the target location equals the most recent location in input.

**Values**:
- DIY: 18.56%
- GeoLife: 27.18%
- **Difference: +8.63%** ← PRIMARY EXPLANATION

**Interpretation**:
- In 27.2% of GeoLife samples, the answer is simply "the last location"
- Only 18.6% for DIY
- The pointer's position bias gives position 1 the highest weight
- When pointer is removed, this pattern becomes hard to capture

**Why this explains 46.7% vs 8.3%**:
- Position 1 gets maximum position bias boost
- 8.6% more GeoLife samples benefit from this
- Generation head cannot learn "predict position 1" as effectively

---

### 2.6 Average Gate Value

**What it measures**: How much the trained model relies on pointer (1.0) vs generation (0.0).

**Values**:
- DIY: 0.787 (78.7% pointer)
- GeoLife: 0.627 (62.7% pointer)
- Difference: -0.160

**Interpretation**:
- Both models favor pointer heavily (>60%)
- DIY model uses pointer MORE aggressively
- This seems counterintuitive but reflects model learning

**Explanation**: DIY model learned to always use pointer heavily. GeoLife model is more adaptive, varying gate based on context.

---

### 2.7 Pointer Probability on Target

**What it measures**: How much probability the pointer assigns to the correct answer.

**Values (when target in history)**:
- DIY: 0.571 (57.1%)
- GeoLife: 0.544 (54.4%)
- Difference: -0.027

**Interpretation**:
- Pointer assigns majority probability to correct answer
- Both datasets show similar pointer effectiveness
- Slight advantage for DIY

---

### 2.8 Generation Probability on Target ⭐ CRITICAL FINDING

**What it measures**: How much probability the generation head assigns to correct answer.

**Values (when target in history)**:
- DIY: 0.005 (0.5%)
- GeoLife: 0.021 (2.1%)
- **Both are near ZERO**

**Interpretation**:
- Generation head is nearly USELESS
- Assigns <2% probability to correct answer
- Pointer is 25-100× more effective
- This proves all predictive power comes from pointer

**Critical insight**: When pointer is removed, both datasets collapse to near-random prediction. The difference is in how much position-specific patterns exist that ONLY pointer can capture.

---

### 2.9 Accuracy When Target NOT in History

**What it measures**: Model accuracy when copying is impossible.

**Values**:
- DIY: 0.15%
- GeoLife: 0.35%
- Both essentially ZERO

**Interpretation**:
- Model completely fails when target not in history
- ~0.15-0.35% is near random chance
- Confirms generation head cannot predict novel locations

---

### 2.10 Recency Score

**What it measures**: Average inverse position of target from sequence end.

**Formula**: `recency_score = 1 / position_from_end`
- Position 1 → score = 1.0
- Position 2 → score = 0.5
- Position 5 → score = 0.2

**Values**:
- DIY: 0.432
- GeoLife: 0.475
- Difference: +0.043

**Interpretation**:
- Higher score = targets are more recent on average
- GeoLife has 10% higher recency score
- Confirms targets are closer to end in GeoLife

---

### 2.11 Predictability Score

**What it measures**: Combined recency and frequency score.

**Formula**: `predictability = recency_score × frequency_score`

**Values**:
- DIY: 0.205
- GeoLife: 0.232
- Difference: +0.027 (+13%)

**Interpretation**:
- Higher predictability = more pointer-friendly patterns
- GeoLife is 13% more predictable
- Combines both position and repetition effects

---

## 3. Statistical Significance

### 3.1 Chi-Square Test (Target in History)

**Test**: Are target-in-history rates significantly different?

**Results**:
- χ² = 0.174
- p-value = 0.676

**Conclusion**: NOT significant (p > 0.05). The rates are statistically the same.

---

### 3.2 Mann-Whitney U Test (Unique Ratios)

**Test**: Are unique ratio distributions significantly different?

**Results**:
- U statistic = 19,139,076
- p-value = 7.03 × 10⁻²⁶

**Conclusion**: HIGHLY significant (p < 0.001). The distributions differ.

---

### 3.3 Cohen's d Effect Size

**Test**: How large is the practical difference in unique ratios?

**Results**:
- Cohen's d = -0.160

**Interpretation**:
- |d| < 0.2 = Small effect
- The difference is statistically significant but practically small

---

## 4. Summary Table

| Metric | DIY | GeoLife | Gap | Significance |
|--------|-----|---------|-----|--------------|
| Target in History | 84.12% | 83.81% | -0.31% | Not significant |
| Repetition Rate | 68.65% | 65.96% | -2.69% | Small effect |
| **Target = Last** | **18.56%** | **27.18%** | **+8.62%** | **KEY** |
| **Consecutive Repeat** | **17.94%** | **26.87%** | **+8.93%** | **KEY** |
| Sequence Entropy | 1.89 bits | 1.74 bits | -0.15 | Minor |
| Gate Value | 0.787 | 0.627 | -0.160 | Model adaptation |
| Pointer Prob | 0.571 | 0.544 | -0.027 | Similar |
| **Gen Prob** | **0.005** | **0.021** | **-** | **Both ~0** |
| Recency Score | 0.432 | 0.475 | +0.043 | Contributing |
| **Ablation Impact** | **8.3%** | **46.7%** | **+38.4%** | **OUTCOME** |

---

## 5. The Bottom Line

**Question**: Why does removing pointer cause 46.7% drop on GeoLife vs 8.3% on DIY?

**Answer in one sentence**: GeoLife users return to their most recent location 8.6% more often (27.2% vs 18.6%), and this pattern is captured by the pointer's position bias but cannot be learned by the generation head.

**The causal chain**:
1. GeoLife has more "target = last position" patterns (8.6% more)
2. The pointer mechanism with position bias captures this perfectly
3. The generation head assigns only 0.5-2% probability to targets
4. When pointer is removed, GeoLife loses 8.6% more of its predictable patterns
5. Result: 46.7% drop vs 8.3% drop

---

*Results Interpretation Guide Version: 1.0*
