# 03. Theoretical Foundation

## The Science Behind Temporal Context in Mobility Prediction

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Theoretical Background |
| **Audience** | Researchers, Graduate Students |
| **Reading Time** | 15-20 minutes |
| **Prerequisites** | Basic probability, information theory concepts |

---

## 1. Human Mobility Theory

### 1.1 Fundamental Laws of Human Movement

Human mobility follows several well-documented statistical laws that form the foundation of our understanding:

#### Lévy Flight Hypothesis
Early research suggested human movements follow a Lévy flight pattern—a random walk where step lengths are drawn from a heavy-tailed distribution:

$$P(\Delta r) \sim |\Delta r|^{-\beta}$$

where $\Delta r$ is the displacement and $\beta \approx 1.75$ in empirical studies.

**Implication**: Humans make many short trips (local movements) punctuated by occasional long trips (inter-city travel, vacations).

#### Truncated Lévy Flight (More Accurate)
Subsequent research showed a better fit with truncated Lévy flights, where the distribution has an exponential cutoff:

$$P(\Delta r) \sim |\Delta r|^{-\beta} \exp\left(-\frac{|\Delta r|}{\kappa}\right)$$

The cutoff $\kappa$ reflects practical constraints (country borders, maximum daily travel distance).

### 1.2 Exploration vs Exploitation Model

Human mobility exhibits a dual nature captured by the exploration-exploitation model:

**Exploitation (Return to Familiar Places)**:
- Humans repeatedly visit a small set of "anchor points"
- Home, work, favorite restaurant, gym
- ~80% of time spent at ~20% of locations (Pareto-like distribution)

**Exploration (Visit New Places)**:
- Occasional visits to never-before-seen locations
- Rate of exploration decreases over time (sublinear growth)
- New location visits follow: $S(t) \propto t^{\mu}$ where $\mu < 1$

**Mathematical Model**:
The probability of visiting location $i$ at time $t$ can be decomposed:

$$P(l_t = i) = \rho \cdot P_{\text{return}}(i) + (1-\rho) \cdot P_{\text{explore}}(i)$$

where $\rho$ is the return probability (typically 0.6-0.8).

### 1.3 Preferential Return Model

The probability of returning to a location depends on:

1. **Visit frequency**: More visits → higher return probability
2. **Recency**: Recent visits → higher return probability
3. **Rank**: Higher rank locations → higher return probability

**Song et al. (2010) Model**:
$$P(\text{return to } i) \propto f_i^\alpha$$

where $f_i$ is the frequency of visits to location $i$ and $\alpha \approx 1$.

**Relevance to This Experiment**:
A longer temporal window captures more visits, providing better estimates of $f_i$ for each location.

---

## 2. Temporal Periodicity in Human Behavior

### 2.1 Multiple Temporal Scales

Human mobility exhibits periodicity at multiple scales:

```
Scale           Period      Examples
─────────────────────────────────────────────────────────
Circadian       24 hours    Sleep-wake cycle, meals
Weekly          7 days      Work week vs weekend
Seasonal        ~90 days    Summer vacation, holidays
Annual          365 days    Birthday trips, anniversaries
```

### 2.2 Daily (Circadian) Patterns

The 24-hour cycle strongly influences mobility:

```
Time of Day     Typical Behavior            Prediction Difficulty
────────────────────────────────────────────────────────────────
6-9 AM          Home → Work commute         Low (routine)
9 AM - 12 PM    At work                     Very Low
12-2 PM         Lunch break                 Medium (choices)
2-6 PM          At work or meetings         Low-Medium
6-9 PM          Work → Home or activities   Medium-High
9 PM - 6 AM     Home (mostly)               Very Low
```

**Key Insight**: Time of day alone explains ~30-40% of location variance.

### 2.3 Weekly Patterns

The 7-day cycle introduces significant variation:

| Day | Mobility Characteristics |
|-----|--------------------------|
| Monday | Work routine begins, predictable |
| Tuesday-Thursday | Most predictable (full work routine) |
| Friday | Work + social activities |
| Saturday | High exploration, low predictability |
| Sunday | Mixed (rest, preparation, social) |

**Empirical Observation**:
- Weekday prediction accuracy: ~55-60%
- Weekend prediction accuracy: ~45-50%
- Difference: ~10 percentage points

### 2.4 Why 7 Days Matters

A 7-day window captures:
1. **One complete weekly cycle**: All day-of-week patterns
2. **Same-day recurrence**: "Last Monday's pattern predicts this Monday"
3. **Weekend vs weekday transitions**: Understanding mode switches
4. **Robust frequency estimates**: Multiple samples per location

**Mathematical Justification**:
If location visits follow day-of-week patterns:
$$P(l|t) = P(l|d(t), h(t))$$

where $d(t) \in \{0,...,6\}$ is day-of-week and $h(t)$ is hour.

With 7 days of history, we observe each day-of-week at least once, enabling:
- Same-day predictions: $P(l|d=\text{Monday}, h=9\text{AM})$
- Cross-day patterns: Weekend→Weekday transitions

---

## 3. Information-Theoretic Perspective

### 3.1 Entropy of Human Mobility

The predictability of mobility can be quantified using entropy:

**Random Entropy** (upper bound):
$$S^{\text{rand}} = \log_2(N)$$

where $N$ is the number of distinct locations visited.

**Temporal-Uncorrelated Entropy**:
$$S^{\text{unc}} = -\sum_{i=1}^{N} p_i \log_2(p_i)$$

where $p_i$ is the probability of visiting location $i$.

**Actual Entropy** (considering temporal correlations):
$$S = \lim_{n\to\infty} H(L_n | L_1, ..., L_{n-1})$$

This is estimated using compression algorithms (Lempel-Ziv).

### 3.2 Predictability Upper Bound

Song et al. (2010) showed that maximum predictability $\Pi^{\text{max}}$ satisfies:

$$S = H(\Pi^{\text{max}}) + (1-\Pi^{\text{max}}) \log_2(N-1)$$

**Key Finding**: Human mobility is 93% predictable in theory, meaning $\Pi^{\text{max}} \approx 0.93$.

**Practical Gap**: Current models achieve 50-60% accuracy, leaving significant room for improvement.

### 3.3 Mutual Information Framework

The mutual information between history and next location quantifies predictive value:

$$I(L_{n+1}; L_1, ..., L_n) = H(L_{n+1}) - H(L_{n+1} | L_1, ..., L_n)$$

**Decomposition by Time Lag**:
$$I(L_{n+1}; L_1, ..., L_n) = \sum_{k=1}^{n} I(L_{n+1}; L_{n+1-k} | L_{n+2-k}, ..., L_n)$$

This decomposition shows how much information each historical time step contributes.

**Expected Pattern**:
```
Mutual Information
        │
        │█████████
        │████████
        │███████
        │██████
        │█████
        │████
        │███
        │██
        │█
        │
        └──────────────────────────▶
         t-1  t-2  t-3  ...  t-n
                Time Lag
```

More recent visits typically provide more information, but periodic patterns create spikes at 7-day, 14-day, etc.

---

## 4. Markov Chain Theory

### 4.1 First-Order Markov Assumption

The simplest mobility model assumes first-order Markov property:

$$P(L_{n+1} | L_1, ..., L_n) = P(L_{n+1} | L_n)$$

**Transition Matrix**:
$$A_{ij} = P(L_{n+1} = j | L_n = i)$$

**Limitation**: Cannot capture multi-step dependencies like "Home → Work → Lunch → Work → Home"

### 4.2 Higher-Order Markov Models

A k-th order Markov model considers k previous states:

$$P(L_{n+1} | L_1, ..., L_n) = P(L_{n+1} | L_{n-k+1}, ..., L_n)$$

**Trade-off**:
- Higher k → captures longer patterns
- Higher k → exponentially more parameters: $O(N^{k+1})$
- Higher k → requires more data to estimate

### 4.3 Variable-Order Markov Models

In practice, optimal order varies by context:

- Morning commute: First-order sufficient (Home → Work)
- Evening activities: Higher-order needed (Work → Gym vs Work → Home depends on whether user already went to gym this week)

**Relevance to Experiment**:
Our experiment effectively tests what temporal depth is needed to capture relevant dependencies, without explicitly parameterizing Markov order.

---

## 5. Sequence Modeling Theory

### 5.1 Recurrent Neural Networks (RNNs)

RNNs model sequences by maintaining a hidden state:

$$h_t = f(W_h h_{t-1} + W_x x_t + b)$$
$$y_t = g(W_y h_t + b_y)$$

**Theoretical Properties**:
- Can capture arbitrarily long dependencies (in theory)
- In practice, suffer from vanishing gradients
- LSTM/GRU variants address gradient issues

**Effective Context Length**:
- Standard RNN: ~5-10 time steps
- LSTM: ~20-50 time steps
- GRU: ~20-50 time steps

### 5.2 Transformer Architecture

Transformers use self-attention to model dependencies:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Theoretical Advantages**:
- Direct connection between any two positions
- Parallel computation (no sequential bottleneck)
- Learnable attention patterns

**Context Length**:
- Theoretically unlimited within memory constraints
- Typically 512-2048 tokens in practice
- Our sequences: 5-99 tokens (well within range)

### 5.3 Pointer Networks

Pointer networks are specialized for selecting from input sequences:

$$P(\text{output}_i = j) \propto \exp(u_i \cdot e_j)$$

where $u_i$ is the decoder state and $e_j$ is the j-th encoder output.

**Why Pointer Networks for Mobility**:
- Most next locations are from user's history
- Explicit copy mechanism for known locations
- Generation head for novel locations

### 5.4 The Attention-History Relationship

Self-attention allows the model to learn what history is relevant:

**Hypothesis**: If day-7 data is useful, attention should show:
- Strong attention to same time-of-day positions
- Strong attention to same day-of-week positions
- Periodic attention patterns at 7-step intervals

---

## 6. The Diminishing Returns Hypothesis

### 6.1 Mathematical Formalization

Let $A(n)$ denote accuracy with $n$ days of history. We hypothesize:

$$A(n) = A_{\max} - C \cdot e^{-\lambda n}$$

where:
- $A_{\max}$ is asymptotic maximum accuracy
- $C$ is the gap between 1-day and maximum
- $\lambda$ is the decay rate

**Implication**: 
$$\frac{dA}{dn} = C\lambda e^{-\lambda n}$$

Marginal improvement decreases exponentially with more history.

### 6.2 Why Diminishing Returns Occur

1. **Information Redundancy**:
   - Weekly patterns repeat
   - Day-8 ≈ Day-1 in terms of information content

2. **Finite User Vocabulary**:
   - Users visit finite set of locations
   - Beyond a point, no new locations appear in history

3. **Model Capacity**:
   - Fixed model capacity limits how much history can be utilized
   - Information bottleneck in hidden representations

4. **Relevance Decay**:
   - Older data less relevant (habits change)
   - Noise accumulates over time

### 6.3 Optimal Window Size

The optimal window balances:

$$n^* = \arg\max_n \left[ A(n) - \text{Cost}(n) \right]$$

where Cost includes:
- Computational cost (processing longer sequences)
- Storage cost (storing more history)
- Latency cost (loading more data)

---

## 7. Dataset Considerations

### 7.1 Spatial Resolution Impact

Different spatial clustering affects sequence characteristics:

| Clustering | Effect on Sequences | Effect on Prediction |
|------------|--------------------|--------------------|
| Fine (ε=20m) | More locations, shorter sequences | Harder prediction |
| Coarse (ε=50m) | Fewer locations, longer sequences | Easier prediction |

**DIY (ε=50m)**: Coarser clustering → longer effective sequences
**GeoLife (ε=20m)**: Finer clustering → more locations to choose from

### 7.2 User Activity Variability

Users differ in activity levels:
- **Active users**: 20+ visits/day, rich patterns
- **Moderate users**: 5-10 visits/day, regular patterns
- **Sparse users**: 1-3 visits/day, limited data

**Impact on This Experiment**:
- Active users: Benefit more from longer windows
- Sparse users: May not have enough data even with 7 days

### 7.3 Geographic and Cultural Context

Mobility patterns vary by context:

| Factor | DIY (Indonesia) | GeoLife (Beijing) |
|--------|-----------------|-------------------|
| Primary transport | Mixed (car, motorcycle, transit) | Mixed (transit, walking, car) |
| Urban density | Moderate-high | Very high |
| Work patterns | Standard business hours | Extended hours |
| Weekend behavior | Family activities | Mixed |

---

## 8. Mathematical Model of the Experiment

### 8.1 Formal Problem Setup

Let:
- $\mathcal{D} = \{(x^{(i)}, y^{(i)})\}_{i=1}^{N}$ be the dataset
- $x^{(i)} = (l_1^{(i)}, l_2^{(i)}, ..., l_{n_i}^{(i)})$ be the input sequence
- $y^{(i)}$ be the target next location
- $\theta$ be model parameters

The model computes:
$$P_\theta(y | x) = \text{PointerGeneratorTransformer}(x; \theta)$$

### 8.2 Filtering Function

For each previous_days value $d$, we define filter function $F_d$:

$$F_d(x) = (l_j : \text{diff}(l_j) \leq d)$$

where $\text{diff}(l_j)$ is the number of days ago for location $l_j$.

### 8.3 Experiment Protocol

For each $d \in \{1, 2, ..., 7\}$:

1. **Filter**: $x_d^{(i)} = F_d(x^{(i)})$
2. **Evaluate**: $\hat{y}^{(i)} = \arg\max P_\theta(y | x_d^{(i)})$
3. **Compute Metrics**: $\text{Acc}_d = \frac{1}{N_d} \sum_{i} \mathbf{1}[\hat{y}^{(i)} = y^{(i)}]$

Where $N_d$ is the number of valid samples after filtering.

---

## 9. Chapter Summary

### Key Theoretical Points

1. **Human mobility follows statistical laws** (Lévy flights, preferential return)
2. **Multiple temporal scales exist** (circadian, weekly, seasonal)
3. **7 days captures one complete weekly cycle**
4. **Information-theoretic limits** suggest 93% predictability is possible
5. **Diminishing returns are expected** due to redundancy and model capacity
6. **Transformers can model long-range dependencies** relevant to weekly patterns

### Connection to Experiment

| Theory | Experimental Test |
|--------|------------------|
| Weekly periodicity | Compare prev1 vs prev7 |
| Diminishing returns | Examine marginal gains day by day |
| Information content | Loss reduction with more history |
| Predictability limits | Compare achieved vs theoretical accuracy |

### What's Next

The following document ([Experimental Methodology](./04_experimental_methodology.md)) details how we design the experiment to test these theoretical predictions, including:
- Controlled variables
- Measurement approach
- Statistical considerations

---

## References

1. Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. *Science*, 327(5968), 1018-1021.

2. Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008). Understanding individual human mobility patterns. *Nature*, 453(7196), 779-782.

3. Pappalardo, L., et al. (2015). Returners and explorers dichotomy in human mobility. *Nature Communications*, 6, 8166.

4. Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer networks. *NeurIPS*.

5. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~2,400 |
| **Status** | Final |

---

**Navigation**: [← Introduction](./02_introduction_and_motivation.md) | [Index](./INDEX.md) | [Next: Experimental Methodology →](./04_experimental_methodology.md)
