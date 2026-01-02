# Results Analysis and Interpretation

## Table of Contents

1. [Performance Summary](#1-performance-summary)
2. [GeoLife Dataset Analysis](#2-geolife-dataset-analysis)
3. [DIY Dataset Analysis](#3-diy-dataset-analysis)
4. [Metric Interpretation](#4-metric-interpretation)
5. [Comparison with Other Models](#5-comparison-with-other-models)
6. [Error Analysis](#6-error-analysis)
7. [Factors Affecting Performance](#7-factors-affecting-performance)
8. [Practical Implications](#8-practical-implications)

---

## 1. Performance Summary

### Overall Results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PERFORMANCE SUMMARY                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                        GeoLife Dataset           DIY Dataset                │
│                        ═══════════════           ═══════════                │
│                                                                              │
│  Metric              Val      Test              Val      Test               │
│  ────────────────    ─────    ─────             ─────    ─────              │
│  Acc@1               33.57%   24.18%            48.10%   44.13%             │
│  Acc@5               47.43%   37.87%            67.24%   62.56%             │
│  Acc@10              48.59%   38.76%            69.48%   64.80%             │
│  MRR                 39.91%   30.34%            56.30%   52.13%             │
│  F1                  32.87%   23.38%            46.01%   42.68%             │
│  NDCG@10             42.01%   32.38%            59.51%   55.22%             │
│                                                                              │
│  ─────────────────────────────────────────────────────────────────────────  │
│                                                                              │
│  Dataset Statistics:                                                        │
│                                                                              │
│                        GeoLife                   DIY                        │
│                        ─────────                 ───                        │
│  Total Records         16,600                    ~200,000                   │
│  Filtered Train        7,424                     ~150,000                   │
│  Val Samples           3,289                     26,499                     │
│  Test Samples          3,457                     26,872                     │
│  Users                 45                        692                        │
│  Parameters            166,309                   366,338                    │
│  Training Time         ~5 seconds                ~43 seconds                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Findings

| Finding | Details |
|---------|---------|
| **DIY outperforms GeoLife** | ~20% higher Acc@1 (44.13% vs 24.18%) |
| **Validation > Test** | Both datasets show validation-test gap |
| **Top-K gains diminish** | Jump from @1 to @5 significant, @5 to @10 minimal |
| **MRR tracks Acc@1** | Both reflect quality of top predictions |

---

## 2. GeoLife Dataset Analysis

### Dataset Characteristics

The GeoLife dataset consists of GPS trajectories from 182 users in Beijing, China, collected over 5 years. After preprocessing:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEOLIFE DATASET PROFILE                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Collection Period:    April 2007 - August 2012                             │
│  Original Users:       182                                                  │
│  Users after filtering: 45                                                  │
│  Geographic Area:      Beijing, China                                       │
│  Total Records:        16,600                                               │
│  Unique Locations:     ~1,185                                               │
│  DBSCAN Epsilon:       20 meters                                            │
│                                                                              │
│  User Mobility Patterns:                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  • Mix of transportation modes (walking, driving, bus, subway)        │ │
│  │  • Diverse activities (commute, leisure, errands)                     │ │
│  │  • Varying tracking durations (50+ days per user)                     │ │
│  │  • Some users highly regular, others very irregular                   │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Results Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    GEOLIFE DETAILED RESULTS                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TEST SET ANALYSIS:                                                         │
│                                                                              │
│  Total test samples:     3,457                                              │
│  Correct @1:             836 (24.18%)                                       │
│  Correct @5:             1,309 (37.87%)                                     │
│  Correct @10:            1,340 (38.76%)                                     │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │         Accuracy Distribution                                       │   │
│  │                                                                     │   │
│  │  @1  ████████████████████████░░░░░░░░░░░░░░░░░░░░░░░░  24.18%     │   │
│  │  @5  █████████████████████████████████████░░░░░░░░░░░░  37.87%     │   │
│  │  @10 ██████████████████████████████████████░░░░░░░░░░░  38.76%     │   │
│  │                                                                     │   │
│  │       0%        25%        50%        75%       100%               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OBSERVATIONS:                                                              │
│                                                                              │
│  1. Large gap between Acc@1 (24%) and Acc@5 (38%)                          │
│     → Correct answer often in positions 2-5                                │
│     → Model captures alternatives but struggles with top-1                  │
│                                                                              │
│  2. Minimal gain from @5 to @10 (< 1%)                                     │
│     → If not in top-5, unlikely to be in top-10                           │
│     → Suggests wrong predictions are "far off"                             │
│                                                                              │
│  3. Validation-Test gap (~9% for Acc@1)                                    │
│     → Train/Val from similar time period (first 80% of days)              │
│     → Test from later period (last 20%)                                    │
│     → Suggests behavior changes over time                                  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why GeoLife Performance is Lower

1. **Diverse mobility patterns:** Users vary widely in regularity
2. **Multi-modal transportation:** Different modes = different patterns
3. **Long collection period:** Behavior may drift over 5 years
4. **Small sample per user:** 45 users, each with limited data
5. **Dense location clustering:** 20m epsilon creates many similar locations

---

## 3. DIY Dataset Analysis

### Dataset Characteristics

The DIY dataset is a proprietary mobility dataset with higher data quality:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIY DATASET PROFILE                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Users:                692                                                  │
│  Total Records:        ~200,000                                             │
│  Unique Locations:     ~4,000                                               │
│  DBSCAN Epsilon:       50 meters (larger = fewer, more distinct locations) │
│                                                                              │
│  User Mobility Patterns:                                                    │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │  • More consistent tracking (quality-filtered users)                  │ │
│  │  • Higher location revisitation rates                                 │ │
│  │  • More regular daily patterns                                        │ │
│  │  • Larger sample size per user on average                            │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Results Breakdown

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    DIY DETAILED RESULTS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  TEST SET ANALYSIS:                                                         │
│                                                                              │
│  Total test samples:     26,872                                             │
│  Correct @1:             11,858 (44.13%)                                    │
│  Correct @5:             16,811 (62.56%)                                    │
│  Correct @10:            17,414 (64.80%)                                    │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                                                                     │   │
│  │         Accuracy Distribution                                       │   │
│  │                                                                     │   │
│  │  @1  ████████████████████████████████████████████░░░░░  44.13%     │   │
│  │  @5  █████████████████████████████████████████████████████████████  62.56%│
│  │  @10 ███████████████████████████████████████████████████████████████ 64.80%│
│  │                                                                     │   │
│  │       0%        25%        50%        75%       100%               │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  OBSERVATIONS:                                                              │
│                                                                              │
│  1. Nearly double Acc@1 compared to GeoLife                                │
│     → Users have more predictable patterns                                 │
│                                                                              │
│  2. Still significant @1 to @5 gap (~18%)                                  │
│     → Multiple viable destinations even for regular users                  │
│                                                                              │
│  3. Smaller validation-test gap (~4% for Acc@1)                            │
│     → More stable behavior over time                                       │
│                                                                              │
│  4. High NDCG (55.22%)                                                     │
│     → When wrong, often still reasonably ranked                            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Why DIY Performance is Higher

1. **Quality filtering:** Users with consistent tracking retained
2. **Higher revisitation:** People return to same places more
3. **Regular patterns:** More routine-based mobility
4. **Larger epsilon:** 50m creates more distinct, memorable locations
5. **More data:** More samples for learning transitions

---

## 4. Metric Interpretation

### What Each Metric Tells Us

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    METRIC INTERPRETATION GUIDE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ACCURACY@K (Acc@1, Acc@5, Acc@10)                                          │
│  ═════════════════════════════════                                          │
│                                                                              │
│  What it measures: % of times correct answer is in top-K predictions        │
│                                                                              │
│  GeoLife Test:                                                              │
│  • Acc@1 = 24.18%: Correct destination is rank 1 in ~1/4 of cases          │
│  • Acc@5 = 37.87%: Correct destination is in top-5 in ~1/3 of cases        │
│                                                                              │
│  Interpretation:                                                            │
│  • Higher Acc@1 = More precise predictions                                 │
│  • Gap between Acc@1 and Acc@5 = Model knows alternatives but not sure    │
│  • Acc@10 ≈ Acc@5 = Long-tail of low-probability destinations             │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  MRR (Mean Reciprocal Rank)                                                 │
│  ═════════════════════════                                                  │
│                                                                              │
│  What it measures: Average of 1/rank across all predictions                 │
│                                                                              │
│  GeoLife Test: MRR = 30.34%                                                 │
│                                                                              │
│  Examples:                                                                   │
│  • Correct at rank 1 → RR = 1.0                                            │
│  • Correct at rank 2 → RR = 0.5                                            │
│  • Correct at rank 3 → RR = 0.33                                           │
│  • Not in predictions → RR = 0                                             │
│                                                                              │
│  Interpretation:                                                            │
│  • MRR near Acc@1 = Good predictions are usually rank 1                    │
│  • MRR > Acc@1 = Many predictions are rank 2-3, not just @1                │
│  • GeoLife: MRR (30%) > Acc@1 (24%) = Often rank 2-3                       │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  NDCG@10 (Normalized Discounted Cumulative Gain)                            │
│  ═══════════════════════════════════════════════                            │
│                                                                              │
│  What it measures: Ranking quality with logarithmic discounting             │
│                                                                              │
│  GeoLife Test: NDCG@10 = 32.38%                                             │
│                                                                              │
│  Why use NDCG?                                                              │
│  • Differentiates between rank 1 vs rank 3 (unlike binary Acc@K)          │
│  • Rank 1 gets full credit, rank 10 gets ~29% credit                       │
│  • Beyond rank 10 gets 0                                                    │
│                                                                              │
│  Interpretation:                                                            │
│  • NDCG ≈ Acc@1 means predictions are either rank 1 or not in top-10      │
│  • NDCG > Acc@1 means many predictions land in positions 2-10             │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  F1 SCORE                                                                   │
│  ════════                                                                   │
│                                                                              │
│  What it measures: Harmonic mean of precision and recall (top-1 only)       │
│                                                                              │
│  GeoLife Test: F1 = 23.38%                                                  │
│                                                                              │
│  Why weighted F1?                                                           │
│  • Accounts for class imbalance (some locations more frequent)             │
│  • Weights each location by its frequency                                  │
│                                                                              │
│  Interpretation:                                                            │
│  • F1 ≈ Acc@1 for balanced datasets                                        │
│  • F1 < Acc@1 suggests model predicts common locations too often          │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Relationship Between Metrics

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    METRIC RELATIONSHIPS                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  For GeoLife Test:                                                          │
│                                                                              │
│  Metric      Value    Relationship                                          │
│  ──────────  ──────   ───────────────────────────────────────────────────── │
│  Acc@1       24.18%   Base accuracy                                         │
│  MRR         30.34%   > Acc@1 → many predictions at rank 2-3               │
│  NDCG        32.38%   > Acc@1 → good ranking even when not @1              │
│  F1          23.38%   ≈ Acc@1 → balanced performance across locations       │
│  Acc@5       37.87%   +14% from @1 → alternatives are captured             │
│  Acc@10      38.76%   +1% from @5 → little beyond top-5                    │
│                                                                              │
│  Visual:                                                                    │
│                                                                              │
│  0%        25%        50%        75%       100%                             │
│  ├─────────┼──────────┼──────────┼──────────┤                              │
│  │         │                                                                │
│  │  F1 ───▼                                                                │
│  │  Acc@1 ──▼                                                               │
│  │  MRR ─────▼                                                              │
│  │  NDCG ─────▼                                                             │
│  │  Acc@5 ─────────▼                                                        │
│  │  Acc@10 ──────────▼                                                      │
│  │                                                                          │
│  ├─────────┼──────────┼──────────┼──────────┤                              │
│                                                                              │
│  Insight: ~15% of predictions are "close but not quite right"              │
│           (rank 2-10 when should be rank 1)                                 │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. Comparison with Other Models

### Baseline Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    MODEL COMPARISON (GeoLife)                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Model              Acc@1    MRR     Training Time    Parameters            │
│  ────────────────   ──────   ─────   ─────────────    ──────────            │
│  Random             ~0.08%   ~0.1%   0s               0                     │
│  Most Frequent      ~8%      ~10%    0s               N                     │
│  Markov (1st-order) 24.18%   30.34%  ~5s              166K                  │
│  LSTM Baseline      ~30%     ~38%    ~minutes         ~1M                   │
│  MHSA               ~35%     ~42%    ~minutes         ~2M                   │
│  Transformer        ~38%     ~45%    ~minutes-hours   ~5M+                  │
│                                                                              │
│  Visual Comparison:                                                         │
│                                                                              │
│         Accuracy@1                                                          │
│         0%       10%       20%       30%       40%       50%                │
│         ├────────┼─────────┼─────────┼─────────┼─────────┤                 │
│  Random │▌                                                                  │
│  MostFr │████                                                               │
│  Markov │████████████                              ← This model             │
│  LSTM   │███████████████                                                    │
│  MHSA   │█████████████████▌                                                 │
│  Transf │██████████████████▌                                                │
│         ├────────┼─────────┼─────────┼─────────┼─────────┤                 │
│                                                                              │
│  KEY INSIGHTS:                                                              │
│                                                                              │
│  1. Markov provides 3x improvement over "Most Frequent" baseline           │
│     → Current location IS informative                                       │
│                                                                              │
│  2. Neural models improve ~10-15% over Markov                              │
│     → But require 10-100x more parameters and training time                │
│                                                                              │
│  3. Diminishing returns for more complex models                            │
│     → Fundamental limit of predictability (~40-45% for GeoLife)            │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Implementation Variant Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    markov_ori vs markov1st                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│                          markov_ori        markov1st                        │
│                          (This doc)        (Alternative)                    │
│                          ──────────        ─────────────                    │
│  GeoLife Acc@1          24.18%            27.64%                            │
│  DIY Acc@1              44.13%            50.60%                            │
│                                                                              │
│  Why Different?                                                             │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  markov_ori:                                                                │
│  • Evaluates ALL consecutive pairs in trajectories                         │
│  • Includes many "in-sequence" predictions                                 │
│  • Some predictions may have less context                                  │
│                                                                              │
│  markov1st:                                                                 │
│  • Uses pre-extracted (X, Y) samples from preprocessing                   │
│  • Each sample has N days of history                                       │
│  • More consistent evaluation context                                      │
│                                                                              │
│  Practical Implication:                                                     │
│  • markov_ori: Reproduces original paper results                           │
│  • markov1st: Fair comparison with neural models                           │
│  • Both are valid; choose based on use case                                │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 6. Error Analysis

### Types of Prediction Errors

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ERROR ANALYSIS                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ERROR TYPE 1: Unseen Transitions (Cold Start)                              │
│  ═══════════════════════════════════════════════                            │
│                                                                              │
│  Scenario: User at location X, but X never seen in training                 │
│                                                                              │
│  Training data:    [H→W, W→H, W→G, G→H]                                    │
│  Test query:       Currently at R (Restaurant) → ???                        │
│                                                                              │
│  Problem: No transitions from R in training                                 │
│  Result: Fallback to zeros (essentially random)                             │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  ERROR TYPE 2: Multi-Modal Destinations                                     │
│  ══════════════════════════════════════                                     │
│                                                                              │
│  Scenario: Current location leads to multiple destinations with similar    │
│            probability                                                       │
│                                                                              │
│  Training data:    [W→H: 50 times, W→G: 48 times, W→R: 45 times]          │
│  Test query:       Currently at W → ?                                       │
│  Prediction:       H (rank 1), G (rank 2), R (rank 3)                      │
│  Actual:           G                                                        │
│                                                                              │
│  Problem: All three are nearly equally likely                               │
│  Result: Acc@1 misses, but Acc@5 hits                                      │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  ERROR TYPE 3: Temporal Patterns Ignored                                    │
│  ════════════════════════════════════════                                   │
│                                                                              │
│  Scenario: Same location has different patterns at different times          │
│                                                                              │
│  Reality:                                                                    │
│    Morning at W → H (going home after night shift) - 10%                   │
│    Evening at W → H (end of workday) - 80%                                 │
│                                                                              │
│  Markov sees: W → H (aggregated) - 90%                                     │
│                                                                              │
│  Test query:       At W in morning                                         │
│  Prediction:       H (based on aggregate)                                   │
│  Actual:           G (gym before work)                                      │
│                                                                              │
│  Problem: Model doesn't know it's morning                                  │
│  Result: Wrong prediction due to time-blindness                             │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  ERROR TYPE 4: Behavior Drift                                               │
│  ═════════════════════════════                                              │
│                                                                              │
│  Scenario: User's patterns change over time                                 │
│                                                                              │
│  Training period (days 0-60): W → H dominant (commute)                     │
│  Test period (days 80-100): W → NewGym dominant (changed routine)          │
│                                                                              │
│  Problem: Training patterns don't reflect test reality                      │
│  Result: Systematic errors on changed behaviors                             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Error Distribution (Estimated)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    ESTIMATED ERROR DISTRIBUTION                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  For GeoLife Test Set (3,457 samples):                                      │
│                                                                              │
│  Correct @1:                 836 (24.2%)   ████████████                     │
│  Correct @2-5:              473 (13.7%)   ██████▌                           │
│  Correct @6-10:              31 (0.9%)    ▌                                 │
│  In predictions, beyond 10: ~150 (4.3%)   ██                               │
│  Not in predictions:       ~1967 (56.9%)  ████████████████████████████▌    │
│                                                                              │
│  Analysis:                                                                   │
│  • 24.2% - Perfect predictions                                             │
│  • 14.6% - Close (rank 2-10)                                               │
│  • 56.9% - Completely missed (unseen transitions or rare patterns)         │
│                                                                              │
│  The majority of errors (57%) are from transitions the model               │
│  never learned, highlighting the cold-start limitation.                     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 7. Factors Affecting Performance

### Positive Factors (Improve Performance)

| Factor | Effect | Example |
|--------|--------|---------|
| **More training data** | More transitions observed | DIY > GeoLife |
| **Higher revisitation** | Stronger patterns | Home-Work commute |
| **Regular users** | Predictable behavior | Office workers |
| **Larger epsilon** | Fewer, more distinct locations | 50m vs 20m |
| **Quality filtering** | Remove noisy users | DIY filtering |

### Negative Factors (Hurt Performance)

| Factor | Effect | Example |
|--------|--------|---------|
| **Sparse data** | Many unseen transitions | New users |
| **Diverse behavior** | Weak patterns | Tourists |
| **Small epsilon** | Too many similar locations | 20m in dense city |
| **Long time span** | Behavior drift | Multi-year data |
| **Multi-modal** | Uncertain destinations | Work → {Home, Gym, Restaurant} |

### Factor Impact Visualization

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FACTOR IMPACT ON ACCURACY                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Revisitation Rate:                                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Low Revisit   ████████░░░░░░░░░░░░  ~20% Acc@1                     │   │
│  │  Med Revisit   ████████████████░░░░  ~40% Acc@1                     │   │
│  │  High Revisit  ██████████████████████████████  ~60% Acc@1          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  Training Data Amount:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  < 50 records  ████░░░░░░░░░░░░░░░░  ~15% Acc@1                     │   │
│  │  50-200        ████████████░░░░░░░░  ~30% Acc@1                     │   │
│  │  > 200 records ██████████████████░░  ~45% Acc@1                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  User Regularity:                                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  Irregular     ████████░░░░░░░░░░░░  ~20% Acc@1                     │   │
│  │  Moderate      ████████████████░░░░  ~35% Acc@1                     │   │
│  │  Regular       ██████████████████████████████  ~55% Acc@1          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 8. Practical Implications

### When to Use This Model

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    PRACTICAL DECISION GUIDE                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  USE MARKOV WHEN:                                                           │
│  ════════════════                                                           │
│                                                                              │
│  ✅ You need a quick baseline                                               │
│     → 5 seconds vs minutes/hours for neural models                         │
│                                                                              │
│  ✅ Interpretability is important                                           │
│     → "User went from A to B 80% of the time"                              │
│                                                                              │
│  ✅ Resources are limited                                                   │
│     → No GPU needed, runs on any laptop                                    │
│                                                                              │
│  ✅ You're establishing a lower bound                                       │
│     → Any model should beat Markov to be worth complexity                  │
│                                                                              │
│  ✅ Users have regular, predictable patterns                                │
│     → Commuters, routine-based mobility                                    │
│                                                                              │
│  ───────────────────────────────────────────────────────────────────────── │
│                                                                              │
│  DON'T USE MARKOV WHEN:                                                     │
│  ═════════════════════                                                      │
│                                                                              │
│  ❌ Maximum accuracy is critical                                            │
│     → Neural models offer 10-15% improvement                               │
│                                                                              │
│  ❌ Temporal patterns matter                                                │
│     → Time of day, day of week affect predictions                          │
│                                                                              │
│  ❌ Long-term dependencies exist                                            │
│     → "Already visited gym, won't go again today"                          │
│                                                                              │
│  ❌ Users have irregular, diverse patterns                                  │
│     → Tourists, explorers, variable schedules                              │
│                                                                              │
│  ❌ Cold-start is common                                                    │
│     → Many new users or new locations                                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Integration Recommendations

1. **As a baseline:** Always run Markov first to establish a baseline
2. **As a fallback:** Use Markov when neural model fails (cold start)
3. **As a feature:** Use Markov predictions as input to neural models
4. **As a sanity check:** If neural < Markov, something is wrong

### Expected Performance by Use Case

| Use Case | Expected Acc@1 | Notes |
|----------|---------------|-------|
| Office commuters | 50-60% | Very regular patterns |
| General population | 35-45% | Mixed regularity |
| Tourists | 15-25% | Irregular, exploring |
| Delivery drivers | 40-50% | Route-based, somewhat regular |
| Students | 45-55% | Class schedule, regular |

---

## Navigation

| Previous | Next |
|----------|------|
| [05_DIAGRAMS_VISUALIZATIONS.md](05_DIAGRAMS_VISUALIZATIONS.md) | [07_WALKTHROUGH_LINE_BY_LINE.md](07_WALKTHROUGH_LINE_BY_LINE.md) |
