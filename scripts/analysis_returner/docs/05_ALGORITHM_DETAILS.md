# Algorithm Details: First-Return Time Analysis

## 1. Algorithm Overview

This document provides a detailed explanation of the **First-Return Time Analysis** algorithm, including mathematical foundations, pseudocode, complexity analysis, and edge case handling.

---

## 2. Problem Definition

### 2.1 Formal Problem Statement

**Given**: A set of users U, where each user u ∈ U has a trajectory:
```
T(u) = {(l₁, t₁), (l₂, t₂), ..., (lₙ, tₙ)}
```
where lᵢ is a location and tᵢ is a timestamp (in hours).

**Compute**: The first-return time distribution F_pt(t), which measures the probability density of users returning to their first observed location after time t.

### 2.2 Key Definitions

📐 **Definition 1 (First Location)**:
For user u with trajectory T(u) = {(l₁, t₁), ..., (lₙ, tₙ)} sorted by time:
```
L₀(u) = l₁ (the first observed location)
t₀(u) = t₁ (the time of first observation)
```

📐 **Definition 2 (First Return Event)**:
The first return event for user u is:
```
(L₀(u), t_return(u)) where:
  t_return(u) = min{tᵢ : lᵢ = L₀(u) AND tᵢ > t₀(u)}
```

📐 **Definition 3 (First-Return Time)**:
```
Δt(u) = t_return(u) - t₀(u)
```

📐 **Definition 4 (Return Probability Density)**:
```
              |{u : Δt(u) ∈ [t, t+Δt)}|
F_pt(t) = ─────────────────────────────────
                N_returns × Δt

Where N_returns = number of users who returned
```

---

## 3. Algorithm Pseudocode

### 3.1 Main Algorithm

```
Algorithm: COMPUTE_RETURN_PROBABILITY_DISTRIBUTION
═══════════════════════════════════════════════════════════════════════

Input:
  - D: Dataset of events {(user_id, location_id, timestamp)}
  - bin_width: Width of histogram bins (default: 2 hours)
  - max_time: Maximum return time to consider (default: 240 hours)

Output:
  - bin_centers: Array of time values (x-axis)
  - F_pt: Array of probability density values (y-axis)

Procedure:

1. SORT D by (user_id, timestamp)

2. FOR each user u in D:
     2.1 first_location[u] ← location of first event for u
     2.2 first_time[u] ← timestamp of first event for u

3. FOR each user u in D:
     3.1 returns[u] ← []
     3.2 FOR each event (loc, time) in trajectory of u:
         IF loc == first_location[u] AND time > first_time[u]:
             returns[u].append(time)
             BREAK  // Only need first return

4. delta_t_values ← []
   FOR each user u where returns[u] is not empty:
     delta_t ← returns[u][0] - first_time[u]
     IF delta_t ≤ max_time:
       delta_t_values.append(delta_t)

5. bins ← [0, bin_width, 2×bin_width, ..., max_time]
   counts ← HISTOGRAM(delta_t_values, bins)

6. N_returns ← length(delta_t_values)
   F_pt ← counts / (N_returns × bin_width)

7. bin_centers ← [(bins[i] + bins[i+1]) / 2 for i in range(len(bins)-1)]

8. RETURN (bin_centers, F_pt)
```

### 3.2 Vectorized Implementation

The actual implementation uses pandas vectorized operations for efficiency:

```
Algorithm: VECTORIZED_COMPUTE_RETURN_TIMES
═══════════════════════════════════════════════════════════════════════

1. df_sorted ← SORT(df, by=['user_id', 'timestamp'])

2. first_events ← GROUP_BY(df_sorted, 'user_id').FIRST()
   // Creates lookup: user_id → (first_location, first_time)

3. df_with_first ← MERGE(df_sorted, first_events, on='user_id')
   // Adds first_location, first_time columns to all rows

4. df_later ← FILTER(df_with_first, timestamp > first_time)
   // Keep only events after first observation

5. df_returns ← FILTER(df_later, location == first_location)
   // Keep only returns to first location

6. first_returns ← GROUP_BY(df_returns, 'user_id').FIRST()
   // Get earliest return for each user

7. delta_t ← first_returns.timestamp - first_returns.first_time
   // Compute return times

8. RETURN delta_t
```

---

## 4. Visual Algorithm Walkthrough

### 4.1 Example Dataset

Consider 3 users with the following trajectories:

```
USER 1 (Alice):
────────────────────────────────────────────────────────────────────
Time(h):   8.0      9.0      12.0     18.0     32.0     44.0
           │        │        │        │        │        │
Location:  HOME     WORK     CAFE     HOME     WORK     HOME
           100      200      300      100      200      100
           ↑                          ↑
        L₀=100                    First Return
        t₀=8.0                    Δt=10.0h

USER 2 (Bob):
────────────────────────────────────────────────────────────────────
Time(h):   5.0      9.0      17.0     29.0
           │        │        │        │
Location:  OFFICE   LUNCH    GYM      OFFICE
           300      400      500      300
           ↑                          ↑
        L₀=300                    First Return
        t₀=5.0                    Δt=24.0h

USER 3 (Carol):
────────────────────────────────────────────────────────────────────
Time(h):   10.0     15.0     20.0     25.0
           │        │        │        │
Location:  PARK     MALL     REST     MOVIE
           600      700      800      900
           ↑
        L₀=600                    NO RETURN
        t₀=10.0
```

### 4.2 Step-by-Step Processing

**Step 1: Sort by user and time**
```
Already sorted in example
```

**Step 2: Extract first events**
```
┌─────────────────────────────────────┐
│ USER    FIRST_LOC    FIRST_TIME    │
├─────────────────────────────────────┤
│ Alice      100          8.0        │
│ Bob        300          5.0        │
│ Carol      600         10.0        │
└─────────────────────────────────────┘
```

**Step 3: Find returns**
```
Alice: Events after t₀=8.0 with loc=100: [18.0, 44.0] → First: 18.0
Bob:   Events after t₀=5.0 with loc=300: [29.0] → First: 29.0
Carol: Events after t₀=10.0 with loc=600: [] → No return
```

**Step 4: Compute delta_t**
```
Alice: Δt = 18.0 - 8.0 = 10.0 hours
Bob:   Δt = 29.0 - 5.0 = 24.0 hours
Carol: No return
```

**Step 5: Build histogram (bin_width=2)**
```
delta_t_values = [10.0, 24.0]

Bins:    [0-2)  [2-4)  ...  [8-10)  [10-12)  ...  [22-24)  [24-26)
Counts:    0      0    ...    0        1      ...     0        1

         [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, ...]
Counts:  [0, 0, 0, 0, 0,  1,  0,  0,  0,  0,  0,  0,  1,  0, ...]
                          ↑                              ↑
                       Alice                            Bob
```

**Step 6: Normalize to probability density**
```
N_returns = 2 (Alice and Bob)
bin_width = 2

F_pt = counts / (N_returns × bin_width)
     = [0, 0, ..., 1, 0, ..., 1, ...] / (2 × 2)
     = [0, 0, ..., 0.25, 0, ..., 0.25, ...]

At t=11 hours: F_pt(11) = 0.25
At t=25 hours: F_pt(25) = 0.25
```

### 4.3 Final Output

```
┌────────────────────────────────┐
│ bin_centers    F_pt           │
├────────────────────────────────┤
│     1          0.000          │
│     3          0.000          │
│     5          0.000          │
│     7          0.000          │
│     9          0.000          │
│    11          0.250  ← Alice │
│    13          0.000          │
│    ...         ...            │
│    23          0.000          │
│    25          0.250  ← Bob   │
│    ...         ...            │
└────────────────────────────────┘
```

---

## 5. Complexity Analysis

### 5.1 Time Complexity

| Step | Operation | Complexity |
|------|-----------|------------|
| 1 | Sort by user and time | O(N log N) |
| 2 | Group by user (first event) | O(N) |
| 3 | Merge | O(N) |
| 4-5 | Filter operations | O(N) |
| 6 | Group by (first return) | O(N) |
| 7 | Histogram | O(N) |

**Total Time Complexity**: O(N log N)

Where N = total number of events

**Actual Runtimes**:
- Geolife (19K events): ~0.5 seconds
- DIY (265K events): ~1.5 seconds

### 5.2 Space Complexity

| Component | Space |
|-----------|-------|
| Input DataFrame | O(N) |
| Sorted DataFrame | O(N) |
| First events table | O(U) |
| Merged DataFrame | O(N) |
| Return times | O(U) |
| Histogram | O(B) |

**Total Space Complexity**: O(N)

Where:
- N = number of events
- U = number of users (U << N typically)
- B = number of bins (fixed at ~120 for 240h/2h)

---

## 6. Edge Cases and Handling

### 6.1 Users with No Returns

**Scenario**: User visits their first location only once.

```
Carol's trajectory: PARK → MALL → REST → MOVIE
                    (never returns to PARK)
```

**Handling**: User is excluded from the return time distribution.

```python
# In code: Carol is filtered out because df_returns is empty for her
first_returns = df_returns.groupby('user_id').first()
# Carol not in first_returns
```

**Impact**: Return rate calculation accounts for this:
```
Return rate = Users with returns / Total users × 100%
            = 2/3 × 100% = 66.67%
```

### 6.2 Very Short Return Times

**Scenario**: User returns almost immediately (same hour).

```
Alice: HOME (8:00) → HOME (8:15)  → Δt = 0.25 hours
```

**Handling**: These are captured in the first bin [0, 2).

**Interpretation**: Could indicate:
- Data quality issues (duplicate records)
- Very short trips (forgot something at home)
- GPS noise

### 6.3 Return Time Exceeds Max Hours

**Scenario**: User returns after more than 240 hours.

```
Bob: First location at t=0, returns at t=500 hours
```

**Handling**: Filtered out by max_hours parameter.

```python
first_returns = first_returns[first_returns['delta_t_hours'] <= max_hours]
```

**Rationale**: Focus on short-to-medium term return patterns (10 days).

### 6.4 Single Event Users

**Scenario**: User has only one recorded event.

```
Dave: HOME (10:00) → [end of data]
```

**Handling**: No events exist after first_time, so no return possible.

```python
df_later = df_with_first[df_with_first['timestamp_hours'] > df_with_first['first_time']]
# Dave's trajectory has no events satisfying this condition
```

### 6.5 Multiple Returns to Same Location

**Scenario**: User returns to first location multiple times.

```
Alice: HOME → WORK → HOME → WORK → HOME
             (18h)         (44h)
```

**Handling**: Only the FIRST return is counted.

```python
first_returns = df_returns.groupby('user_id').first()  # Takes earliest
```

**Rationale**: F_pt(t) measures FIRST-return time, not all returns.

---

## 7. Probability Density Properties

### 7.1 Normalization Proof

The probability density F_pt(t) satisfies:

```
∫₀^∞ F_pt(t) dt = 1
```

**Proof**:
```
∫₀^∞ F_pt(t) dt ≈ Σᵢ F_pt(tᵢ) × Δt
                = Σᵢ (countᵢ / (N × Δt)) × Δt
                = Σᵢ countᵢ / N
                = N / N
                = 1
```

### 7.2 Verification in Code

```python
# After computing pdf
total_mass = (pdf * bin_width).sum()
print(f"Total probability mass: {total_mass:.6f}")
# Should print: Total probability mass: 1.000000
```

### 7.3 Relationship to CDF

The cumulative distribution function (CDF) can be computed as:

```
F(t) = ∫₀^t F_pt(τ) dτ ≈ Σᵢ F_pt(tᵢ) × Δt  for tᵢ ≤ t
```

**Interpretation**: F(t) = probability of returning within t hours.

---

## 8. Random Walk Baseline Derivation

### 8.1 Theoretical Model

For a simple random walk on a lattice with n locations:

```
P(return at step k) ∝ 1/k^(d/2)
```

where d is the dimensionality.

### 8.2 Simplified Exponential Model

We use a simplified exponential decay model:

```
F_RW(t) = P₀ × exp(-t/τ)
```

**Parameters**:
- P₀ = 0.01 (initial probability, fitted to approximate scale)
- τ = 30 hours (decay constant)

### 8.3 Justification for Comparison

The RW baseline serves as a **null model**:
- If humans moved randomly, we'd expect exponential decay
- Real data shows periodic peaks (24h, 48h, etc.)
- Deviation from RW indicates intentional, planned movement

---

## 9. Algorithm Variations

### 9.1 All-Returns Analysis (Alternative)

Instead of first-return time, analyze ALL returns:

```
Modified Step 3: Don't break after first return
  FOR each event (loc, time) in trajectory of u:
      IF loc == first_location[u] AND time > first_time[u]:
          returns[u].append(time - first_time[u])
          // Don't break - collect all returns
```

**Use case**: Analyzing recurrence patterns, periodicity strength.

### 9.2 Nth-Return Analysis (Alternative)

Analyze return to Nth most visited location:

```
Modified Step 2:
  top_locations[u] ← most_frequent_locations(trajectory of u, N=3)

Modified Step 3:
  FOR each event (loc, time) in trajectory of u:
      IF loc in top_locations[u] AND time > first_time[u]:
          ...
```

**Use case**: Understanding returns to important places (home, work, etc.).

---

## 10. Summary

### 10.1 Algorithm Characteristics

| Property | Value |
|----------|-------|
| Time complexity | O(N log N) |
| Space complexity | O(N) |
| Deterministic | Yes |
| Parallelizable | Yes (by user) |

### 10.2 Key Parameters

| Parameter | Default | Range | Impact |
|-----------|---------|-------|--------|
| bin_width | 2.0 | 0.5-10 | Resolution vs. noise tradeoff |
| max_hours | 240 | 24-720 | Time window for analysis |

### 10.3 Assumptions

1. Trajectories are sorted chronologically
2. Location IDs are consistent (same place = same ID)
3. Timestamps are in hours from a common reference
4. Users have at least one recorded event

---

*← Back to [Data Pipeline](04_DATA_PIPELINE.md) | Continue to [Results Interpretation](06_RESULTS_INTERPRETATION.md) →*
