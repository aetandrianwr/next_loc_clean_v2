# Mathematical Foundations

A rigorous mathematical treatment of the concepts underlying the gap performance analysis.

---

## 1. The Pointer-Generator Framework

### 1.1 Problem Formulation

**Input**: 
- Location sequence X = (x₁, x₂, ..., xₙ) where xᵢ ∈ {1, 2, ..., V}
- V = total number of unique locations (vocabulary size)
- n = sequence length

**Output**: 
- Probability distribution P(Y) over next location Y ∈ {1, 2, ..., V}

**Objective**: Learn P(Y | X) to predict the next location.

### 1.2 Pointer Mechanism

The pointer mechanism computes a distribution over input positions, then scatters to vocabulary.

**Step 1: Attention Scores**
```
e_i = (W_q · h)ᵀ · (W_k · h_i) / √d
```

Where:
- h = context vector (final hidden state)
- h_i = hidden state at position i
- W_q, W_k = learned projection matrices
- d = model dimension

**Step 2: Position Bias**
```
ê_i = e_i + b_{n-i+1}
```

Where:
- b = learned position bias vector
- n-i+1 = position from end (1 = most recent)

**Step 3: Attention Distribution**
```
α = softmax(ê)
α_i = exp(ê_i) / Σⱼ exp(ê_j)
```

**Step 4: Scatter to Vocabulary**
```
P_ptr(v) = Σᵢ α_i · 𝟙[x_i = v]
```

Where 𝟙[·] is the indicator function.

**Properties**:
- Σᵥ P_ptr(v) = 1 (valid probability distribution)
- P_ptr(v) > 0 only if v ∈ {x₁, ..., xₙ}
- P_ptr is a weighted sum of one-hot vectors

### 1.3 Generation Mechanism

The generation head computes a distribution over the full vocabulary:

```
P_gen(v) = softmax(W_o · h + b_o)
P_gen(v) = exp(w_v · h + b_v) / Σᵤ exp(w_u · h + b_u)
```

**Properties**:
- Σᵥ P_gen(v) = 1
- P_gen(v) > 0 for all v ∈ {1, ..., V}
- Can predict locations not in input

### 1.4 Gating Mechanism

The gate combines pointer and generation:

```
g = σ(W_g · [h; MLP(h)])
P_final(v) = g · P_ptr(v) + (1-g) · P_gen(v)
```

Where σ is the sigmoid function.

**Properties**:
- g ∈ [0, 1]
- If g = 1: fully pointer (copy only)
- If g = 0: fully generation (generate only)
- Model learns optimal g for each input

---

## 2. Shannon Entropy

### 2.1 Definition

For a discrete probability distribution P over outcomes X:

```
H(X) = -Σₓ P(x) log₂ P(x)
```

**Unit**: bits (when using log base 2)

### 2.2 Properties

1. **Non-negativity**: H(X) ≥ 0
2. **Maximum**: H(X) ≤ log₂(|X|) with equality when P is uniform
3. **Additivity**: H(X,Y) = H(X) + H(Y) for independent X, Y

### 2.3 Application to Mobility

For a sequence of location visits:

```
p(loc) = count(loc) / total_visits
H = -Σ_loc p(loc) log₂ p(loc)
```

**Interpretation**:
- Low entropy: visits concentrated on few locations
- High entropy: visits spread across many locations

### 2.4 Normalized Entropy

To compare across different vocabulary sizes:

```
H_norm = H / H_max = H / log₂(n_unique)
```

**Range**: [0, 1]
- 0: only one location visited (minimum diversity)
- 1: all locations visited equally (maximum diversity)

---

## 3. Statistical Tests

### 3.1 Chi-Square Test for Independence

**Null hypothesis**: Target-in-history rate is independent of dataset.

**Contingency table**:
```
                 In History    Not In History
DIY                 a              b
GeoLife             c              d
```

**Test statistic**:
```
χ² = Σᵢⱼ (O_ij - E_ij)² / E_ij
```

Where:
- O_ij = observed count
- E_ij = expected count = (row_i total × col_j total) / grand total

**Decision**: Reject H₀ if χ² > χ²_critical(α, df=1)

### 3.2 Mann-Whitney U Test

**Null hypothesis**: Unique ratio distributions are the same.

**Procedure**:
1. Combine samples and rank them
2. Sum ranks for each group: R₁, R₂
3. Compute U statistics:
   ```
   U₁ = n₁n₂ + n₁(n₁+1)/2 - R₁
   U₂ = n₁n₂ + n₂(n₂+1)/2 - R₂
   U = min(U₁, U₂)
   ```

**For large samples**: U is approximately normal
```
z = (U - μ_U) / σ_U
μ_U = n₁n₂ / 2
σ_U = √(n₁n₂(n₁+n₂+1) / 12)
```

### 3.3 Cohen's d Effect Size

**Definition**:
```
d = (μ₁ - μ₂) / s_pooled
s_pooled = √[(s₁² + s₂²) / 2]
```

**Interpretation**:
- |d| < 0.2: Small effect
- |d| ≈ 0.5: Medium effect
- |d| > 0.8: Large effect

---

## 4. Position Bias Analysis

### 4.1 Expected Benefit from Position Bias

Let:
- f_k = fraction of samples where target is at position k from end
- b_k = position bias for position k
- A_k = base attention at position k (without bias)

**Expected attention boost**:
```
E[boost] = Σₖ f_k · exp(b_k) / Σⱼ exp(b_j + A_j)
```

### 4.2 Differential Impact

**DIY**: f₁ = 0.186 (target at position 1)
**GeoLife**: f₁ = 0.272

**Difference in position-1 benefit**:
```
Δf₁ = 0.272 - 0.186 = 0.086
```

This 8.6% difference directly maps to differential pointer benefit.

### 4.3 Why Generation Cannot Compensate

The generation head operates on context h, which is computed as:

```
h = TransformerEncoder(embeddings + positional_encoding)[-1]
```

**Information bottleneck**:
- Input: n × d dimensional sequence
- Output: d dimensional context (single vector)
- Position-specific information is compressed

**What is lost**:
- Explicit position indices
- Direct access to "what was at position 1"
- Ability to say "copy from position k"

The pointer mechanism PRESERVES position information through attention:
```
α_i ∝ exp(attention_score_i + position_bias_i)
```

---

## 5. Probability Theory Perspective

### 5.1 Conditional Probability Decomposition

```
P(Y = v | X) = P(Y = v | Y ∈ X) · P(Y ∈ X | X) 
             + P(Y = v | Y ∉ X) · P(Y ∉ X | X)
```

**Empirical estimates**:
- P(Y ∈ X | X) ≈ 0.84 (target in history)
- P(Y ∉ X | X) ≈ 0.16

### 5.2 Pointer Effectiveness

When Y ∈ X (target in history):
```
P_ptr(Y = y | Y ∈ X) = attention_weight_on_y
                     ≈ 0.57 (from data)
```

When Y ∉ X (target not in history):
```
P_ptr(Y = y | Y ∉ X) = 0
```

### 5.3 Generation Effectiveness

When Y ∈ X:
```
P_gen(Y = y | Y ∈ X) ≈ 0.005 to 0.021
```

When Y ∉ X:
```
P_gen(Y = y | Y ∉ X) ≈ 1/V (near random)
```

**Conclusion**: P_gen << P_ptr regardless of condition.

---

## 6. Information-Theoretic Analysis

### 6.1 Mutual Information

**Definition**: Information shared between X and Y.

```
I(X; Y) = H(Y) - H(Y | X)
```

**Higher I(X; Y)** means X is more informative about Y.

### 6.2 Conditional Entropy

```
H(Y | X) = -E[log P(Y | X)]
```

**Lower H(Y|X)** means Y is more predictable given X.

### 6.3 Dataset Comparison

**GeoLife has lower H(Y|X)**:
- More predictable next locations
- Higher I(X; Y)
- More benefit from pointer mechanism

**DIY has higher H(Y|X)**:
- Less predictable next locations
- Lower I(X; Y)
- Less dependent on pointer mechanism

---

*Mathematical Foundations Version: 1.0*
