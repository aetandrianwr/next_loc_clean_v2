# Deep Dive Analysis: Understanding the Pointer Mechanism Performance Gap

## A Comprehensive Scientific Investigation

This document provides an exhaustive, in-depth analysis of why the pointer mechanism in the PointerNetworkV45 model shows dramatically different impacts when removed from two human mobility datasets: **DIY (8.3% relative accuracy drop)** versus **GeoLife (46.7% relative accuracy drop)**. This 5.6× difference requires careful scientific explanation.

---

# Part I: Foundation and Context

## Chapter 1: The Research Problem

### 1.1 What We Observed

In ablation studies conducted on the PointerNetworkV45 model, we systematically removed components to measure their contribution. The results were striking:

**DIY Dataset Ablation Results:**
```
Full Model Accuracy:     56.58%
Without Pointer:         51.88%
Absolute Drop:           4.70 percentage points
Relative Drop:           8.3%
```

**GeoLife Dataset Ablation Results:**
```
Full Model Accuracy:     53.97%
Without Pointer:         28.76%
Absolute Drop:           25.21 percentage points
Relative Drop:           46.7%
```

### 1.2 The Central Question

**Why does removing the same architectural component (the pointer mechanism) cause such dramatically different impacts on two datasets that both involve human mobility prediction?**

This is not a trivial difference. A 46.7% relative drop means the model loses almost half its predictive power, while an 8.3% drop is relatively modest. Understanding this difference has profound implications for:

1. **Model Architecture Design**: Should we always include pointer mechanisms?
2. **Dataset Selection**: Which datasets are appropriate for which models?
3. **Scientific Understanding**: What fundamental differences exist in human mobility patterns across populations?

### 1.3 Our Hypothesis

We hypothesize that **GeoLife users exhibit fundamentally different mobility patterns** that make them more dependent on the pointer (copy) mechanism. Specifically:

- GeoLife users may revisit recent locations more frequently
- GeoLife patterns may be more concentrated and predictable
- The temporal recency of target locations may differ between datasets

---

## Chapter 2: The PointerNetworkV45 Architecture

### 2.1 Conceptual Overview

The PointerNetworkV45 is a hybrid neural network that combines two prediction strategies:

```
                    ┌─────────────────────────────────────────────┐
                    │         PointerNetworkV45 Architecture       │
                    └─────────────────────────────────────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    │                                          │
                    ▼                                          ▼
        ┌───────────────────┐                      ┌───────────────────┐
        │  POINTER MECHANISM │                      │  GENERATION HEAD  │
        │                    │                      │                    │
        │  "Copy from input" │                      │  "Generate new"   │
        │                    │                      │                    │
        │  Attends to input  │                      │  Predicts over    │
        │  sequence and      │                      │  entire location  │
        │  copies locations  │                      │  vocabulary       │
        │  directly          │                      │                    │
        └─────────┬──────────┘                      └─────────┬──────────┘
                  │                                            │
                  │         P_pointer(location)                │   P_generation(location)
                  │                                            │
                  └──────────────────┬─────────────────────────┘
                                     │
                                     ▼
                           ┌─────────────────┐
                           │   LEARNED GATE  │
                           │                 │
                           │  g = σ(W·h + b) │
                           │                 │
                           │  g ∈ [0, 1]     │
                           └────────┬────────┘
                                    │
                                    ▼
                    ┌───────────────────────────────┐
                    │     FINAL DISTRIBUTION        │
                    │                               │
                    │  P(loc) = g·P_ptr + (1-g)·P_gen │
                    └───────────────────────────────┘
```

### 2.2 The Pointer Mechanism in Detail

The pointer mechanism works by attending to the input sequence and "pointing" to positions that should be copied.

**Mathematical Formulation:**

```python
# Step 1: Compute attention scores between context and each input position
query = W_query @ context                    # [d_model]
keys = W_key @ encoded_sequence              # [seq_len, d_model]
scores = (query @ keys.T) / sqrt(d_model)    # [seq_len]

# Step 2: Add position bias (learned preference for recent positions)
scores = scores + position_bias[pos_from_end]  # [seq_len]

# Step 3: Mask padding and compute attention
scores[padding_mask] = -inf
attention = softmax(scores)                  # [seq_len], sums to 1

# Step 4: Scatter attention to location vocabulary
P_pointer = zeros(num_locations)
for i, loc in enumerate(input_sequence):
    P_pointer[loc] += attention[i]           # Accumulate probability
```

**Example:**
```
Input sequence:  [Home, Work, Cafe, Work, Home]
                   ↓     ↓     ↓     ↓     ↓
Attention:      [0.05, 0.10, 0.05, 0.15, 0.65]  (learned to favor recent)
                                            ↑
                                    Most recent = highest

P_pointer[Home] = 0.05 + 0.65 = 0.70  (appears at positions 0 and 4)
P_pointer[Work] = 0.10 + 0.15 = 0.25  (appears at positions 1 and 3)
P_pointer[Cafe] = 0.05               (appears at position 2)
```

### 2.3 The Position Bias: Why Recency Matters

The model includes a **learned position bias** that adds a preference for certain positions:

```python
# position_bias is a learnable parameter of shape [max_seq_len]
# pos_from_end tells us how recent each position is (1 = most recent)

scores_with_bias = attention_scores + position_bias[pos_from_end]
```

This allows the model to learn that recent positions are often more predictive. After training, the position bias typically looks like:

```
Position from end:  1     2     3     4     5     6     7
Position bias:    +1.2  +0.8  +0.4  +0.1  -0.1  -0.3  -0.5
                   ↑
           Most recent gets highest boost
```

**Why this matters for our analysis:**
- If a dataset has strong recency patterns (target often = recent location), the position bias is CRITICAL
- When the pointer is removed, this position-aware copying is lost
- The generation head must learn these patterns from scratch, which is much harder

### 2.4 The Generation Head

The generation head is a simple feedforward network that predicts over the full vocabulary:

```python
# context: the final hidden state [d_model]
logits = W_output @ context + bias           # [num_locations]
P_generation = softmax(logits)               # [num_locations]
```

**Key limitation:** The generation head has NO direct access to the input sequence. It only sees the encoded context. It cannot "look up" what locations were in the history.

### 2.5 The Adaptive Gate

The gate learns when to use pointer vs generation:

```python
# Simplified gate computation
hidden = GELU(W1 @ context)                  # [d_model/2]
gate = sigmoid(W2 @ hidden)                  # [1], range [0,1]

# Final probability
P_final = gate * P_pointer + (1 - gate) * P_generation
```

**What the gate learns:**
- High gate (→1): "The answer is probably in the input, use pointer"
- Low gate (→0): "The answer might be new, use generation"

---

## Chapter 3: The Datasets

### 3.1 DIY Dataset (diy_eps50)

**Origin:** Custom check-in dataset collected from a social media platform

**Preprocessing:**
- Spatial clustering with epsilon=50 meters
- Temporal segmentation with prev7 (7-day history window)

**Statistics from our analysis:**
```
Total test samples:           12,368
Number of unique users:       692
Average sequence length:      23.98 locations
Average unique locations:     6.19 per sequence
Maximum sequence length:      ~48 locations

Location vocabulary size:     Varies by user
User activity level:          Moderate to high
Geographic coverage:          Urban areas
```

**Characteristics:**
- More users (692) means more diverse behavior patterns
- Longer sequences provide more context
- Higher unique location ratio suggests more diverse mobility

### 3.2 GeoLife Dataset (geolife_eps20)

**Origin:** Microsoft Research GeoLife GPS Trajectory Dataset (2007-2012)

**Preprocessing:**
- Spatial clustering with epsilon=20 meters (tighter clustering)
- Temporal segmentation with prev7 (7-day history window)

**Statistics from our analysis:**
```
Total test samples:           3,502
Number of unique users:       45
Average sequence length:      18.37 locations
Average unique locations:     5.41 per sequence
Maximum sequence length:      ~40 locations

Location vocabulary size:     Varies by user
User activity level:          High (GPS tracking)
Geographic coverage:          Beijing, China
```

**Characteristics:**
- Fewer users (45) means more homogeneous behavior
- Shorter sequences but denser temporal coverage
- Lower unique location ratio suggests more repetitive patterns

### 3.3 Key Differences Summary

| Aspect | DIY | GeoLife | Implication |
|--------|-----|---------|-------------|
| Users | 692 | 45 | DIY has more behavioral diversity |
| Samples | 12,368 | 3,502 | DIY has more data |
| Avg Seq Length | 23.98 | 18.37 | DIY has longer history |
| Epsilon | 50m | 20m | GeoLife has finer spatial resolution |
| Collection | Check-ins | GPS | Different mobility capture methods |

---

# Part II: Experiment Design and Methodology

## Chapter 4: The Analysis Framework

### 4.1 Three-Pronged Approach

We designed three complementary analyses to understand the performance gap:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ANALYSIS FRAMEWORK                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ EXPERIMENT 1: MOBILITY PATTERN ANALYSIS                      │   │
│  │                                                               │   │
│  │ Purpose: Understand fundamental mobility characteristics     │   │
│  │                                                               │   │
│  │ Questions answered:                                           │   │
│  │ • How often does the target appear in input history?         │   │
│  │ • How diverse are location visits within sequences?          │   │
│  │ • How concentrated are visits to frequent locations?         │   │
│  │ • What is the entropy (randomness) of mobility patterns?     │   │
│  │                                                               │   │
│  │ Script: analyze_mobility_patterns.py                         │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ EXPERIMENT 2: MODEL BEHAVIOR ANALYSIS                        │   │
│  │                                                               │   │
│  │ Purpose: Understand how trained models use the pointer       │   │
│  │                                                               │   │
│  │ Questions answered:                                           │   │
│  │ • What gate values do models learn for each dataset?         │   │
│  │ • How much probability does pointer assign to targets?       │   │
│  │ • How much probability does generation assign to targets?    │   │
│  │ • When does pointer help vs hurt predictions?                │   │
│  │                                                               │   │
│  │ Script: analyze_model_pointer.py                             │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │ EXPERIMENT 3: RECENCY PATTERN ANALYSIS                       │   │
│  │                                                               │   │
│  │ Purpose: Understand temporal recency effects                 │   │
│  │                                                               │   │
│  │ Questions answered:                                           │   │
│  │ • How recent is the target when it appears in history?       │   │
│  │ • How often is the target the MOST RECENT location?          │   │
│  │ • What return patterns exist (A→B→A)?                        │   │
│  │ • How does recency correlate with pointer benefit?           │   │
│  │                                                               │   │
│  │ Script: analyze_recency_patterns.py                          │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 4.2 Why These Three Experiments?

**Experiment 1 (Mobility Patterns)** answers: "Are the datasets fundamentally different?"

**Experiment 2 (Model Behavior)** answers: "How do trained models actually use the pointer?"

**Experiment 3 (Recency Patterns)** answers: "Is the difference about WHEN targets appear in history?"

Together, they provide a complete picture connecting:
```
Dataset Properties → Model Behavior → Performance Impact
```

---

## Chapter 5: Experiment 1 - Mobility Pattern Analysis

### 5.1 Target-in-History Analysis

**Purpose:** Determine how often the target (next location) appears somewhere in the input history.

**Why this matters:** If the target is in history, the pointer CAN copy it. If not, only generation can predict it.

**Implementation (from analyze_mobility_patterns.py):**

```python
def analyze_target_in_history(self, data: list, name: str) -> dict:
    """
    For each sample, check if target Y appears anywhere in input X.
    """
    target_in_history = []
    target_position_from_end = []
    target_frequency_in_history = []
    
    for sample in data:
        x = sample['X']  # Input sequence, e.g., [1, 5, 3, 5, 2]
        y = sample['Y']  # Target, e.g., 5
        
        # Check if target is anywhere in input
        is_in_history = y in x  # True if 5 is in [1, 5, 3, 5, 2]
        target_in_history.append(is_in_history)
        
        if is_in_history:
            # Find all positions where target appears
            positions = np.where(x == y)[0]  # [1, 3] (indices where 5 appears)
            
            # Convert to position from end (1 = most recent)
            # If sequence length is 5, position 3 → pos_from_end = 5-3 = 2
            pos_from_end = len(x) - positions[-1]  # Most recent occurrence
            target_position_from_end.append(pos_from_end)
            
            # Count frequency
            target_frequency_in_history.append(len(positions))
```

**Results:**

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Total samples | 12,368 | 3,502 | - |
| Target in history count | 10,404 | 2,935 | - |
| **Target in history rate** | **84.12%** | **83.81%** | **-0.31%** |
| Avg position from end | 3.37 | 3.33 | -0.04 |
| Avg target frequency | 9.26 | 7.39 | -1.87 |

**Interpretation:**

The target-in-history rates are nearly identical (~84%). This tells us:

1. **The pointer CAN help both datasets equally** - in 84% of cases, the answer is available to copy
2. **This does NOT explain the performance gap** - if both have 84%, why does removing pointer hurt GeoLife 5× more?
3. **The difference must be in HOW targets appear, not WHETHER they appear**

**Statistical Test:**
```
Chi-square test for target-in-history:
χ² = 0.174
p-value = 0.676
Conclusion: NOT statistically significant
```

### 5.2 Unique Location Ratio Analysis

**Purpose:** Measure the diversity of locations within sequences.

**Formula:**
```
unique_ratio = |{unique locations in X}| / |X|

Examples:
X = [1, 2, 3, 4, 5]     → unique_ratio = 5/5 = 1.0  (all unique)
X = [1, 1, 1, 1, 1]     → unique_ratio = 1/5 = 0.2  (all same)
X = [1, 2, 1, 2, 1]     → unique_ratio = 2/5 = 0.4  (repetitive)
```

**Implementation:**

```python
def analyze_unique_location_ratio(self, data: list, name: str) -> dict:
    unique_ratios = []
    
    for sample in data:
        x = sample['X']
        seq_len = len(x)
        n_unique = len(np.unique(x))  # Count unique locations
        ratio = n_unique / seq_len
        unique_ratios.append(ratio)
    
    # Also compute repetition rate = 1 - unique_ratio
    # This tells us what proportion of visits are to repeated locations
```

**Results:**

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| **Avg unique ratio** | **0.313** | **0.340** | **+0.027** |
| Std unique ratio | 0.176 | 0.162 | - |
| Median unique ratio | 0.286 | 0.308 | - |
| Min unique ratio | 0.021 | 0.025 | - |
| Max unique ratio | 1.000 | 1.000 | - |
| Avg sequence length | 23.98 | 18.37 | - |
| Avg unique count | 6.19 | 5.41 | - |
| **Repetition rate** | **68.65%** | **65.96%** | **-2.70%** |

**Interpretation:**

1. **Both datasets are highly repetitive** - only ~31-34% of locations are unique
2. **DIY is slightly MORE repetitive** (68.65% vs 65.96%)
3. **This seems counterintuitive** - if DIY is more repetitive, shouldn't pointer help DIY more?

**Resolution of the paradox:**
The total repetition rate doesn't capture WHERE the repetitions occur. GeoLife may have more repetitions in RECENT positions, which is what the pointer with position bias captures.

### 5.3 Location Entropy Analysis

**Purpose:** Measure the randomness/predictability of location choices using Shannon entropy.

**Shannon Entropy Formula:**
```
H = -Σ p(location) × log₂(p(location))

Where p(location) = count(location) / total_visits

Properties:
- H = 0: Only one location ever visited (completely predictable)
- H = log₂(n): All n locations visited equally (maximum randomness)
```

**Example:**
```
Sequence: [A, A, A, A, B]
Counts: A=4, B=1, Total=5
Probabilities: p(A)=0.8, p(B)=0.2

H = -[0.8 × log₂(0.8) + 0.2 × log₂(0.2)]
H = -[0.8 × (-0.322) + 0.2 × (-2.322)]
H = -[-0.258 + (-0.464)]
H = 0.722 bits

Maximum possible = log₂(2) = 1 bit
Normalized entropy = 0.722 / 1.0 = 0.722 (72.2% of maximum randomness)
```

**Implementation:**

```python
def calculate_entropy(counts):
    """Calculate Shannon entropy from counts."""
    total = sum(counts)
    if total == 0:
        return 0
    probs = np.array([c / total for c in counts if c > 0])
    return -np.sum(probs * np.log2(probs))

def analyze_location_entropy(self, data: list, name: str) -> dict:
    sequence_entropies = []
    
    for sample in data:
        x = sample['X']
        counter = Counter(x)  # Count each location
        entropy = calculate_entropy(counter.values())
        sequence_entropies.append(entropy)
    
    # Also compute normalized entropy
    normalized_entropies = []
    for sample in data:
        x = sample['X']
        counter = Counter(x)
        n_unique = len(counter)
        if n_unique > 1:
            max_entropy = np.log2(n_unique)
            entropy = calculate_entropy(counter.values())
            normalized_entropies.append(entropy / max_entropy)
        else:
            normalized_entropies.append(0)  # Only one location
```

**Results:**

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| **Avg sequence entropy** | **1.89 bits** | **1.74 bits** | **-0.15 bits** |
| Std sequence entropy | 0.81 | 0.72 | - |
| **Avg normalized entropy** | **0.785** | **0.775** | **-0.010** |
| Std normalized entropy | 0.196 | 0.207 | - |
| Avg user entropy | 1.83 bits | 2.14 bits | +0.31 bits |
| Number of users | 692 | 45 | - |

**Interpretation:**

1. **GeoLife has lower sequence entropy** (1.74 vs 1.89 bits)
   - GeoLife sequences are more predictable/concentrated
   - Fewer bits needed to describe GeoLife mobility

2. **Normalized entropy is similar** (~0.78 for both)
   - When accounting for vocabulary size, diversity is similar
   - The raw entropy difference is partly due to different vocabularies

3. **User entropy is HIGHER for GeoLife** (2.14 vs 1.83 bits)
   - Individual GeoLife users visit more diverse locations overall
   - But within sequences, visits are more concentrated

**Key insight:** GeoLife users have larger "location vocabularies" but visit them more predictably within sequences. This is exactly the pattern where position-biased attention helps most.

### 5.4 Consecutive Repeat Analysis

**Purpose:** Measure A→A patterns (staying at or returning to the same location).

**Types of patterns:**
```
Consecutive repeat (A→A):
[Home, Home, Work, ...]  - Stayed at Home
         ↑↑
     Same location

Target equals last:
Sequence: [A, B, C, D]
Target: D
Result: Target = Last position ✓
```

**Implementation:**

```python
def analyze_consecutive_repeats(self, data: list, name: str) -> dict:
    consecutive_repeat_rates = []
    target_equals_last = []
    
    for sample in data:
        x = sample['X']
        y = sample['Y']
        
        if len(x) < 2:
            consecutive_repeat_rates.append(0)
            continue
        
        # Count consecutive repeats (A→A patterns)
        n_consecutive = sum(1 for i in range(len(x)-1) if x[i] == x[i+1])
        rate = n_consecutive / (len(x) - 1)  # Normalize by possible pairs
        consecutive_repeat_rates.append(rate)
        
        # Check if target equals the last location in sequence
        target_equals_last.append(y == x[-1])
```

**Results:**

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| **Avg consecutive repeat rate** | **17.94%** | **26.87%** | **+8.93%** |
| Std consecutive repeat rate | 25.35% | 25.26% | - |
| % with any consecutive | 73.03% | 86.78% | +13.75% |
| **Target equals last rate** | **18.56%** | **27.18%** | **+8.63%** |

**THIS IS THE KEY FINDING!**

**Interpretation:**

1. **GeoLife has 8.93% more consecutive repeats**
   - GeoLife users stay at or immediately return to locations more often
   - This is exactly what the pointer's position bias captures

2. **GeoLife has 8.63% higher "target = last" rate**
   - In 27.18% of GeoLife samples, the answer is the most recent location
   - In only 18.56% of DIY samples
   - The pointer with position bias DIRECTLY captures this pattern

3. **86.78% of GeoLife sequences have at least one consecutive repeat**
   - Only 73.03% for DIY
   - GeoLife mobility is more "sticky"

**Why this explains the ablation gap:**
- The pointer mechanism gives position-1 (most recent) the highest position bias
- When target = last 27.18% of the time, this bias is crucial
- When pointer is removed, the model loses this position-specific pattern capture
- The generation head cannot easily learn "last position is often the answer"

### 5.5 Most Frequent Location Analysis

**Purpose:** Analyze concentration of visits to top locations.

**Implementation:**

```python
def analyze_most_frequent_location(self, data: list, name: str) -> dict:
    most_freq_ratios = []
    target_is_most_freq = []
    target_is_top3 = []
    
    for sample in data:
        x = sample['X']
        y = sample['Y']
        counter = Counter(x)
        total = len(x)
        
        # Most frequent location
        most_common = counter.most_common(3)  # Top 3 locations
        
        # Ratio of visits to most frequent
        most_freq_ratio = most_common[0][1] / total
        most_freq_ratios.append(most_freq_ratio)
        
        # Is target the most frequent?
        target_is_most_freq.append(y == most_common[0][0])
        
        # Is target in top 3?
        top3_locs = [loc for loc, count in most_common]
        target_is_top3.append(y in top3_locs)
```

**Results:**

| Metric | DIY | GeoLife | Difference |
|--------|-----|---------|------------|
| Avg most frequent ratio | 47.33% | 51.49% | +4.16% |
| Avg top-3 ratio | 83.69% | 87.12% | +3.44% |
| Target is most frequent | 41.99% | 44.20% | +2.22% |
| **Target in top-3** | **75.23%** | **78.47%** | **+3.24%** |

**Interpretation:**

1. **GeoLife visits are more concentrated**
   - 51.49% of GeoLife visits go to the single most frequent location
   - Only 47.33% for DIY

2. **Top-3 locations dominate**
   - 87.12% of GeoLife visits are to top-3 locations
   - 83.69% for DIY

3. **Target predictability from frequency**
   - 78.47% of GeoLife targets are in the top-3 most frequent locations
   - 75.23% for DIY

**Connection to pointer mechanism:**
More concentrated visits mean the pointer attention can focus on fewer locations, making the copy operation more accurate.

---

## Chapter 6: Statistical Significance Tests

### 6.1 Chi-Square Test for Target-in-History

**Purpose:** Test if target-in-history rates are significantly different.

**Setup:**
```
Contingency table:
                    In History    Not In History
DIY                   10,404          1,964
GeoLife                2,935            567
```

**Results:**
```python
chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
# chi2 = 0.174
# p_value = 0.676
# Degrees of freedom = 1
```

**Conclusion:** p > 0.05, so the difference is NOT statistically significant. Both datasets have essentially the same target-in-history rate.

### 6.2 Mann-Whitney U Test for Unique Ratios

**Purpose:** Test if unique ratio distributions are significantly different.

**Why Mann-Whitney?**
- Non-parametric test (doesn't assume normal distribution)
- Appropriate for comparing distributions
- Works well with skewed data

**Results:**
```python
u_stat, p_value = stats.mannwhitneyu(diy_unique_ratios, geolife_unique_ratios)
# U statistic = 19,139,076
# p_value = 7.03 × 10⁻²⁶
```

**Conclusion:** p < 0.001, so the difference IS statistically significant. The unique ratio distributions are genuinely different.

### 6.3 Cohen's d Effect Size

**Purpose:** Measure the magnitude of the difference (not just significance).

**Formula:**
```
Cohen's d = (mean1 - mean2) / pooled_std

pooled_std = sqrt[(std1² + std2²) / 2]
```

**Interpretation:**
- |d| < 0.2: Small effect
- |d| ~ 0.5: Medium effect
- |d| > 0.8: Large effect

**Results:**
```python
pooled_std = np.sqrt((0.176**2 + 0.162**2) / 2)  # = 0.169
cohens_d = (0.313 - 0.340) / 0.169  # = -0.160
```

**Conclusion:** Small effect size (d = -0.16). The difference is statistically significant but practically small for unique ratios. The meaningful differences are in OTHER metrics (consecutive repeats, target=last).

---
