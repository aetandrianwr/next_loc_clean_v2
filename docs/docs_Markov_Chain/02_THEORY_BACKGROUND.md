# Theoretical Background: Markov Chains for Location Prediction

## Table of Contents

1. [Introduction to Markov Chains](#1-introduction-to-markov-chains)
2. [The Markov Property Explained](#2-the-markov-property-explained)
3. [Mathematical Foundations](#3-mathematical-foundations)
4. [Application to Location Prediction](#4-application-to-location-prediction)
5. [Transition Probability Estimation](#5-transition-probability-estimation)
6. [Prediction Mechanism](#6-prediction-mechanism)
7. [Theoretical Justification](#7-theoretical-justification)
8. [Comparison with Higher-Order Models](#8-comparison-with-higher-order-models)
9. [Limitations from a Theoretical Perspective](#9-limitations-from-a-theoretical-perspective)

---

## 1. Introduction to Markov Chains

### What is a Markov Chain?

A **Markov Chain** is a mathematical system that models a sequence of events where the probability of each event depends only on the state attained in the previous event. Named after Russian mathematician Andrey Markov (1856-1922), it is one of the most fundamental concepts in probability theory and stochastic processes.

### Intuitive Understanding

Think of a Markov Chain as a "forgetful" system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MARKOV CHAIN INTUITION                           │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Imagine a frog jumping on lily pads:                               │
│                                                                     │
│       [A]  ─── 0.7 ───>  [B]                                       │
│        │                  │                                         │
│       0.3               0.4                                         │
│        ↓                  ↓                                         │
│       [C]  <─── 0.6 ─── [B]                                        │
│                                                                     │
│  The frog only "remembers" which pad it's currently on.            │
│  It doesn't remember the path it took to get there.                │
│                                                                     │
│  From pad B:                                                        │
│    • 40% chance to stay at B                                       │
│    • 60% chance to jump to C                                       │
│                                                                     │
│  This probability is THE SAME regardless of whether the frog       │
│  arrived at B from A, from C, or started at B.                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Historical Context

Markov Chains were developed to analyze the interdependence of letters in Russian literature. Today, they are used in:
- **Weather prediction** (sunny → rainy, rainy → sunny)
- **Page ranking** (Google's original PageRank algorithm)
- **Speech recognition** (word sequences)
- **Genetics** (DNA sequence analysis)
- **Finance** (stock price movements)
- **Location prediction** (our use case)

---

## 2. The Markov Property Explained

### Formal Definition

The **Markov Property** (also called "memorylessness") states:

> The future state depends only on the present state, not on the sequence of events that preceded it.

Mathematically:

```
P(X_{n+1} = x | X_n = x_n, X_{n-1} = x_{n-1}, ..., X_0 = x_0) = P(X_{n+1} = x | X_n = x_n)
```

### Why "Memoryless"?

```
┌─────────────────────────────────────────────────────────────────────┐
│                    MEMORYLESSNESS ILLUSTRATED                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Non-Markov System (Has Memory):                                    │
│  ────────────────────────────────                                   │
│  "I went Home → Work → Gym → Work. Because I already went to       │
│   the gym today, I probably won't go again. I'll go Home."         │
│                                                                     │
│  Past matters! The decision depends on the full history.            │
│                                                                     │
│  ─────────────────────────────────────────────────────────────────  │
│                                                                     │
│  Markov System (No Memory):                                         │
│  ────────────────────────────                                       │
│  "I'm at Work. Based on my general patterns from Work, there's a   │
│   40% chance I go Home, 35% Gym, 25% Restaurant."                  │
│                                                                     │
│  Only current state matters! History is ignored.                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Is the Markov Property Realistic for Human Mobility?

**Partially.** Human mobility has both:

1. **Markovian aspects:**
   - Immediate context matters (you're more likely to go home from work than from home)
   - Many transitions are habitual and predictable

2. **Non-Markovian aspects:**
   - Time of day matters (not captured in basic Markov)
   - Day of week matters (weekday vs weekend patterns)
   - Long-term dependencies (already visited gym today)

The 1st-order Markov model is a **simplification** that captures dominant patterns while ignoring temporal and historical nuances.

---

## 3. Mathematical Foundations

### State Space

The **state space** S is the set of all possible locations:

```
S = {L_1, L_2, L_3, ..., L_N}
```

Where N is the total number of unique locations in the dataset.

### Transition Probability Matrix

For a Markov Chain with N states, we define an N×N **transition probability matrix** P:

```
        To:    L_1   L_2   L_3   ...   L_N
      ┌──────────────────────────────────────┐
From: │                                      │
L_1   │  p_11  p_12  p_13  ...  p_1N        │
L_2   │  p_21  p_22  p_23  ...  p_2N        │
L_3   │  p_31  p_32  p_33  ...  p_3N        │
...   │  ...   ...   ...   ...  ...         │
L_N   │  p_N1  p_N2  p_N3  ...  p_NN        │
      └──────────────────────────────────────┘
```

Where `p_ij = P(next location = L_j | current location = L_i)`

### Properties of the Transition Matrix

1. **Non-negativity:** All entries are ≥ 0
   ```
   p_ij ≥ 0  for all i, j
   ```

2. **Row stochasticity:** Each row sums to 1
   ```
   Σ_j p_ij = 1  for all i
   ```
   (From any state, you must go somewhere)

3. **Sparsity:** Most real transition matrices are sparse
   - Users don't transition between all possible location pairs
   - Many p_ij = 0 (impossible or never-observed transitions)

### Example Transition Matrix

For a user with 4 frequent locations: Home (H), Work (W), Gym (G), Restaurant (R)

```
        To:    H     W     G     R
      ┌────────────────────────────────┐
From: │                                │
H     │  0.0   0.6   0.2   0.2        │  (From Home: 60% Work, 20% Gym, 20% Restaurant)
W     │  0.5   0.0   0.3   0.2        │  (From Work: 50% Home, 30% Gym, 20% Restaurant)
G     │  0.4   0.4   0.0   0.2        │  (From Gym: equal Home/Work, some Restaurant)
R     │  0.7   0.2   0.1   0.0        │  (From Restaurant: mostly Home)
      └────────────────────────────────┘
```

---

## 4. Application to Location Prediction

### Problem Formulation

**Given:**
- A sequence of visited locations: L_1, L_2, ..., L_t
- Transition probabilities learned from historical data

**Goal:**
- Predict L_{t+1} (the next location)

### 1st-Order Assumption

The 1st-order Markov assumption simplifies prediction to:

```
argmax_L P(L_{t+1} = L | L_t)
```

We only need to look up the current location L_t in our transition matrix and find the most probable next location.

### Per-User Personalization

In practice, different users have different mobility patterns. The model maintains **separate transition matrices per user**:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    PER-USER TRANSITION MATRICES                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  User 1 (Office Worker):           User 2 (Student):               │
│  ─────────────────────             ─────────────────                │
│        H    W    G    R                 H    U    L    C           │
│    H [ 0   .8   .1   .1 ]          H [ 0   .7   .2   .1 ]         │
│    W [.6   0    .3   .1 ]          U [.3   0    .4   .3 ]         │
│    G [.7  .2    0    .1 ]          L [.4  .3    0    .3 ]         │
│    R [.8  .1   .1    0  ]          C [.5  .2   .3    0  ]         │
│                                                                     │
│  H=Home, W=Work, G=Gym, R=Restaurant                               │
│  U=University, L=Library, C=Cafe                                    │
│                                                                     │
│  Same current location → Different predictions for different users │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 5. Transition Probability Estimation

### Maximum Likelihood Estimation (MLE)

Transition probabilities are estimated using **Maximum Likelihood Estimation**:

```
p̂_ij = Count(transitions from i to j) / Count(transitions from i to any)

      = n_ij / Σ_k n_ik
```

Where:
- `n_ij` = number of times we observed transition from location i to location j
- `Σ_k n_ik` = total number of transitions from location i

### Example Calculation

**Training data for User 1:**
```
Day 1: Home → Work → Gym → Home
Day 2: Home → Work → Restaurant → Home
Day 3: Home → Work → Work → Gym → Home
```

**Count transitions:**
```
From Home:    Home → Work: 3       Total from Home: 3
From Work:    Work → Gym: 2        Total from Work: 4
              Work → Restaurant: 1
              Work → Work: 1
From Gym:     Gym → Home: 2        Total from Gym: 2
From Restaurant: Restaurant → Home: 1  Total from Restaurant: 1
```

**Calculate probabilities:**
```
P(Work | Home) = 3/3 = 1.00
P(Gym | Work) = 2/4 = 0.50
P(Restaurant | Work) = 1/4 = 0.25
P(Work | Work) = 1/4 = 0.25
P(Home | Gym) = 2/2 = 1.00
P(Home | Restaurant) = 1/1 = 1.00
```

### Handling Zero Counts (Smoothing)

What if we never observed a transition from A to B in training, but it occurs in testing?

The original implementation handles this through **fallback mechanisms**:

1. **User-specific fallback:** If current location not found for user, use all user's transitions
2. **Global fallback:** If user has no relevant data, use global transition statistics
3. **Frequency fallback:** If no transition data at all, use most frequent locations

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FALLBACK HIERARCHY                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Query: Predict next location for User 1 at Location X             │
│                                                                     │
│  Step 1: Check User 1's transitions from Location X                │
│          Found? → Return ranked destinations                        │
│          Not found? ↓                                               │
│                                                                     │
│  Step 2: Reduce history length (for higher-order)                  │
│          Found match? → Return ranked destinations                 │
│          Still not found? ↓                                         │
│                                                                     │
│  Step 3: Return zero predictions (cold start scenario)              │
│          This indicates unseen location for user                    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. Prediction Mechanism

### Top-K Prediction

The model doesn't just predict the single most likely location; it produces a **ranked list** of predictions:

```python
# Given current location and user
# Sort all possible next locations by transition count
predictions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)

# Return top-K predictions
return [loc for loc, count in predictions[:K]]
```

### Prediction Algorithm (Pseudocode)

```
FUNCTION predict(current_location, user_id, K):
    
    # Step 1: Look up user's transition table
    user_transitions = transition_table[user_id]
    
    # Step 2: Get transitions from current location
    IF current_location IN user_transitions:
        candidates = user_transitions[current_location]
        
        # Step 3: Sort by count (descending)
        sorted_candidates = SORT(candidates, by=count, descending=True)
        
        # Step 4: Return top K
        RETURN sorted_candidates[:K]
    
    ELSE:
        # Fallback: Return zeros (no prediction possible)
        RETURN [0] * K
```

### Handling Ties

When multiple destinations have the same count:
- The implementation uses `drop_duplicates()` which keeps the first occurrence
- In practice, the order depends on the data structure (pandas DataFrame order)

---

## 7. Theoretical Justification

### Why Does This Work?

1. **Regularity of Human Mobility:**
   Human movement is highly predictable. Studies show that 93% of human mobility can be predicted with the right model (Song et al., 2010).

2. **Location Correlation:**
   Certain locations naturally lead to others:
   - Home → Work (morning commute)
   - Work → Home (evening return)
   - Restaurant → Home (after dinner)

3. **Habit Formation:**
   People develop routines. The same user tends to follow similar patterns day after day.

### Information-Theoretic Perspective

From an information theory standpoint, the Markov model captures the **mutual information** between consecutive locations:

```
I(L_{t+1}; L_t) = H(L_{t+1}) - H(L_{t+1} | L_t)
```

Where:
- `H(L_{t+1})` = entropy (uncertainty) of next location without any information
- `H(L_{t+1} | L_t)` = conditional entropy given current location
- `I(...)` = reduction in uncertainty by knowing current location

A high mutual information indicates that knowing the current location significantly reduces uncertainty about the next location—which is exactly what makes Markov models effective.

### Occam's Razor

The 1st-order Markov model embodies the principle of parsimony:
- **Simple model** with minimal assumptions
- **Few parameters** (just transition counts)
- **Interpretable** predictions
- Serves as a strong baseline before adding complexity

---

## 8. Comparison with Higher-Order Models

### What is Model Order?

The **order** of a Markov model refers to how many previous states influence the next state:

| Order | Dependency | Example |
|-------|------------|---------|
| 0 | None | P(L_{t+1}) - independent of history |
| **1** | **Current only** | **P(L_{t+1} \| L_t) - our model** |
| 2 | Current + previous | P(L_{t+1} \| L_t, L_{t-1}) |
| n | Last n states | P(L_{t+1} \| L_t, L_{t-1}, ..., L_{t-n+1}) |

### Trade-offs

```
┌─────────────────────────────────────────────────────────────────────┐
│              ORDER vs COMPLEXITY TRADE-OFF                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  Higher Order:                                                      │
│  ├── ✅ Captures longer patterns (Home→Work→Gym different from     │
│  │       Home→Gym→Gym)                                             │
│  ├── ❌ Exponentially more parameters (N^order possible states)    │
│  ├── ❌ More data needed to estimate reliably                      │
│  └── ❌ Higher risk of overfitting                                 │
│                                                                     │
│  1st Order (Our Model):                                             │
│  ├── ✅ Simple and interpretable                                   │
│  ├── ✅ Few parameters (N^2 at most)                              │
│  ├── ✅ Robust with limited data                                   │
│  ├── ✅ Fast training and inference                                │
│  └── ❌ Cannot capture multi-step patterns                         │
│                                                                     │
│  Parameters by Order:                                               │
│  Order 1: ~N² parameters (e.g., 1000² = 1M)                       │
│  Order 2: ~N³ parameters (e.g., 1000³ = 1B) ← Often infeasible    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### The Code's Flexibility

The `run_markov_ori.py` implementation supports higher orders through the `markov_order` config parameter:

```python
n = config.get("markov_order", 1)  # Default is 1
```

With order n, the model tracks sequences of n locations:
```
n=1: loc_1 → toLoc
n=2: loc_1, loc_2 → toLoc
n=3: loc_1, loc_2, loc_3 → toLoc
```

---

## 9. Limitations from a Theoretical Perspective

### Fundamental Limitations

1. **Time Invariance Assumption**
   - The model assumes transition probabilities don't change over time
   - Reality: Morning commute ≠ Evening commute ≠ Weekend

2. **Homogeneous Markov Chain**
   - Same transition matrix applies at all time steps
   - Reality: Behavior varies by context (weekday/weekend, season)

3. **Discrete State Space**
   - Locations must be discretized (via DBSCAN clustering)
   - Reality: GPS coordinates are continuous; discretization loses information

4. **Independence of Time Gaps**
   - Model ignores how long ago previous location was visited
   - Reality: Transition 5 minutes ago vs 5 hours ago may have different implications

### Data Sparsity Issues

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SPARSITY PROBLEM                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  With 1000 locations:                                               │
│  • Possible transitions: 1000 × 1000 = 1,000,000                   │
│  • Observed transitions: Maybe 10,000 (1%)                         │
│                                                                     │
│  Most entries in transition matrix = 0                              │
│                                                                     │
│  Problem: New location pairs in test set may have zero probability │
│  Solution: Fallback mechanisms, but predictions become uninformed  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### What the Model Cannot Capture

| Pattern | Example | Why Markov Fails |
|---------|---------|------------------|
| Temporal regularity | Go to work at 9am, not at midnight | No time awareness |
| Weekly patterns | Gym on Monday, Church on Sunday | No day awareness |
| Activity completion | Already visited gym today | No history awareness |
| Trip chains | Home→Gas→Grocery→Home is one "trip" | Sees as independent transitions |
| External events | Meeting at new location | No context awareness |

### Theoretical Bounds

The **entropy rate** of a 1st-order Markov chain provides a theoretical lower bound on prediction uncertainty:

```
H_rate = - Σ_i π_i Σ_j p_ij log(p_ij)
```

Where π_i is the stationary distribution. This represents the irreducible uncertainty in the system.

---

## Summary

The 1st-Order Markov Chain model for location prediction is based on solid probabilistic foundations:

1. **Mathematical basis:** Transition probability matrices estimated via MLE
2. **Markov property:** Next location depends only on current location
3. **Personalization:** Separate matrices per user
4. **Prediction:** Rank candidates by transition frequency

**Strengths:** Simplicity, interpretability, speed, baseline quality
**Limitations:** No temporal awareness, limited memory, sparsity challenges

Despite its simplicity, the Markov model remains a valuable baseline because it captures the dominant pattern in human mobility: **where you are strongly predicts where you'll go next.**

---

## Navigation

| Previous | Next |
|----------|------|
| [01_OVERVIEW.md](01_OVERVIEW.md) | [03_TECHNICAL_IMPLEMENTATION.md](03_TECHNICAL_IMPLEMENTATION.md) |

---

## References

1. Markov, A. A. (1906). "Extension of the law of large numbers to dependent quantities"
2. Song, C., et al. (2010). "Limits of Predictability in Human Mobility" Science
3. Gambs, S., et al. (2012). "Next place prediction using mobility Markov chains"
4. Lu, X., et al. (2013). "Approaching the Limit of Predictability in Human Mobility"
