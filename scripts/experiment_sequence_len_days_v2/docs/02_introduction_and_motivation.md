# 02. Introduction and Motivation

## Why Study Sequence Length in Location Prediction?

---

## Document Overview

| Item | Details |
|------|---------|
| **Document Type** | Introduction & Background |
| **Audience** | Researchers, Graduate Students, Practitioners |
| **Reading Time** | 10-12 minutes |
| **Prerequisites** | Basic understanding of machine learning |

---

## 1. The Next Location Prediction Problem

### 1.1 Problem Definition

**Next location prediction** is the task of predicting where a user will go next based on their historical movement patterns and contextual information.

**Formal Definition**:
Given:
- A sequence of historical locations: $L = \{l_1, l_2, ..., l_n\}$
- Associated timestamps: $T = \{t_1, t_2, ..., t_n\}$
- User identity: $u$
- Contextual features: $C$ (time of day, day of week, etc.)

Predict:
- The next location: $l_{n+1}$

**Mathematical Formulation**:
$$\hat{l}_{n+1} = \arg\max_{l \in \mathcal{L}} P(l_{n+1} = l | L, T, u, C)$$

where $\mathcal{L}$ is the set of all possible locations.

### 1.2 Why This Problem Matters

Next location prediction has profound implications across multiple domains:

#### Transportation and Urban Planning
- **Traffic prediction**: Anticipating congestion by predicting vehicle destinations
- **Public transit optimization**: Adjusting bus/train frequencies based on predicted demand
- **Infrastructure planning**: Understanding population flow for new road/rail construction
- **Parking management**: Predicting parking demand at different locations

#### Location-Based Services
- **Personalized recommendations**: Suggesting relevant restaurants, shops, or services
- **Proactive assistance**: "You usually get coffee at 9 AM - want directions to the nearest café?"
- **Advertising**: Targeted location-aware marketing
- **Social networking**: Suggesting meetup locations based on friends' predicted movements

#### Public Health and Safety
- **Epidemic tracking**: Modeling disease spread through population movement
- **Emergency response**: Predicting evacuation patterns during disasters
- **Crime prevention**: Anticipating high-risk areas based on movement patterns
- **Contact tracing**: Understanding potential exposure networks

#### Personal Productivity
- **Calendar integration**: Auto-suggesting travel times for appointments
- **Smart home automation**: Pre-heating home based on predicted arrival
- **Battery optimization**: Managing device resources based on predicted locations

### 1.3 The Challenge

Despite decades of research, next location prediction remains challenging due to:

1. **Sparsity**: Users visit many locations infrequently
2. **Individuality**: Each user has unique patterns
3. **Context-dependence**: Same user, same time → different decisions on different days
4. **Exploration**: Users occasionally visit new places
5. **External factors**: Weather, events, social influence

---

## 2. The Temporal Context Question

### 2.1 Central Research Question

This experiment addresses a fundamental question in mobility prediction:

> **How much historical data should we use to predict the next location?**

This seemingly simple question has no obvious answer:

**Arguments for MORE history**:
- Captures weekly patterns (e.g., "every Monday I go to the gym")
- Reveals habitual locations (frequently visited places)
- Provides context about user's typical movement range
- Enables detection of periodic behaviors

**Arguments for LESS history**:
- Recent behavior may be more predictive than old patterns
- Habits change over time (concept drift)
- Computational efficiency (shorter sequences = faster processing)
- Data storage considerations (less history to store)

### 2.2 What "History Length" Means

In this experiment, we measure history length in **days**, specifically:

- **1 day (prev1)**: Only locations visited in the last 24 hours
- **2 days (prev2)**: Locations from the last 48 hours
- **...**
- **7 days (prev7)**: A full week of location history

**Important Clarification**: 
Days are counted backward from each prediction point. If we're predicting where a user will go at 3 PM on Wednesday:
- prev1 = Tuesday 3 PM to Wednesday 3 PM
- prev7 = Previous Wednesday 3 PM to this Wednesday 3 PM

### 2.3 The Trade-off Landscape

```
                    ▲ Performance
                    │
                    │                    ┌── Saturation
                    │                   ╱
                    │                  ╱
                    │                 ╱
                    │                ╱
                    │               ╱
                    │              ╱
                    │             ╱ ← Optimal region
                    │            ╱
                    │           ╱
                    │          ╱
                    │         ╱
                    │        ╱ ← Rapid improvement
                    │       ╱
                    │      ╱
                    │_____╱_________________________________▶
                         1d  2d  3d  4d  5d  6d  7d    History Length

Legend:
- Rapid improvement: Each day adds significant new information
- Optimal region: Best performance/cost trade-off
- Saturation: Diminishing returns, more data adds little value
```

---

## 3. Research Gap and Motivation

### 3.1 What We Already Know

Previous research has established:

1. **Markov Models work**: First-order Markov chains can predict next locations with ~40-50% accuracy
2. **RNNs improve on Markov**: LSTM and GRU models capture longer dependencies, reaching ~50-60% accuracy
3. **Transformers excel**: Self-attention mechanisms handle variable-length sequences effectively
4. **Temporal features help**: Time of day, day of week are strong predictors
5. **Personalization matters**: User-specific patterns significantly improve predictions

### 3.2 What We Don't Know

However, several questions remain underexplored:

1. **Optimal window size**: What's the ideal amount of historical data?
2. **Diminishing returns**: When does adding more history stop helping?
3. **Dataset dependence**: Do different datasets have different optimal windows?
4. **Metric dependence**: Does the optimal window differ for different evaluation metrics?

### 3.3 Why This Experiment

This experiment addresses the gap by:

| What We Do | Why It Matters |
|------------|----------------|
| Fix the model architecture | Isolate the effect of data length |
| Vary only the temporal window | Controlled experimental design |
| Test multiple metrics | Comprehensive evaluation |
| Use two datasets | Generalizability check |
| Provide detailed analysis | Actionable insights |

---

## 4. Real-World Implications

### 4.1 For System Designers

**Data Retention Policy**:
If 7 days provides optimal accuracy, systems should store at least 7 days of history. If 3 days provides 90% of the benefit, a 3-day retention policy might be more cost-effective.

**Example Calculation**:
- 1M users × 20 visits/day × 7 days = 140M records
- 1M users × 20 visits/day × 3 days = 60M records
- Storage savings: 57%

**Cold Start Strategy**:
New users have limited history. Understanding how performance degrades with less data informs:
- When to show predictions (minimum viable history)
- How to set user expectations
- Alternative strategies for new users

### 4.2 For Researchers

**Baseline Comparisons**:
Many papers don't report the temporal window used, making comparisons difficult. This experiment provides reference points for different window sizes.

**Hypothesis Generation**:
If 7 days provides significant improvement over 1 day, this suggests:
- Weekly periodicity is important
- Models can learn temporal patterns spanning days
- History compression might be valuable

**Future Directions**:
Understanding the saturation point guides research into:
- Attention mechanisms for long sequences
- Temporal decay weighting
- Hierarchical temporal modeling

### 4.3 For End Users

**Privacy Considerations**:
Less required history = less sensitive data stored = better privacy.

If 3 days provides sufficient accuracy:
- Systems can delete older data
- Users have less exposure to data breaches
- Compliance with data minimization principles (GDPR)

---

## 5. Scope and Boundaries

### 5.1 What This Experiment Covers

✅ **Included**:
- Temporal window from 1 to 7 days
- Two real-world datasets (DIY, GeoLife)
- One well-tuned model architecture (PointerGeneratorTransformer)
- Multiple evaluation metrics (Acc@k, MRR, NDCG, F1, Loss)
- Detailed statistical analysis
- Publication-quality visualizations

### 5.2 What This Experiment Does Not Cover

❌ **Not Included**:
- Windows longer than 7 days (14 days, 30 days)
- Other model architectures (comparison with RNN, Markov)
- Real-time vs batch prediction scenarios
- Per-user optimal window analysis
- Seasonal or monthly patterns
- Computational cost analysis

### 5.3 Assumptions

1. **Stationarity**: User patterns don't change significantly within the test period
2. **Completeness**: The datasets capture most of users' mobility
3. **Independence**: Test samples are independent (no overlap)
4. **Representativeness**: The two datasets represent typical urban mobility

---

## 6. Hypotheses

Before running the experiment, we formulated these hypotheses:

### Hypothesis 1: More History Improves Accuracy
**H1**: Prediction accuracy will increase as more historical data is provided.

**Rationale**: More data provides richer context for the model to identify patterns.

### Hypothesis 2: Diminishing Returns
**H2**: The improvement will show diminishing returns, with marginal gains decreasing as history length increases.

**Rationale**: Beyond a certain point, additional information becomes redundant.

### Hypothesis 3: Weekly Cycle Matters
**H3**: Seven days will provide meaningful improvement over one day, as it captures a full weekly cycle.

**Rationale**: Human behavior is strongly influenced by weekly schedules (work week vs weekend).

### Hypothesis 4: Top-K Benefits More
**H4**: Top-k accuracy (k > 1) will benefit more from additional history than top-1 accuracy.

**Rationale**: More history helps the model learn better rankings, even if the top prediction doesn't change.

### Hypothesis 5: Dataset-Specific Effects
**H5**: The optimal window size may differ between datasets due to different mobility patterns.

**Rationale**: Different cultural and geographic contexts may have different temporal dependencies.

---

## 7. Connection to Broader Research

### 7.1 Related Fields

This experiment connects to:

| Field | Connection |
|-------|------------|
| **Human Mobility Modeling** | Understanding temporal scales of human movement |
| **Sequence Modeling** | Optimal context length for sequential prediction |
| **Information Theory** | Entropy and predictability of human behavior |
| **Urban Computing** | Data-driven approaches to city understanding |
| **Privacy Research** | Data minimization and retention policies |

### 7.2 Related Problems

Similar "how much history" questions arise in:

- **Language Modeling**: How much text context improves next word prediction?
- **Stock Prediction**: How many days of prices to use for forecasting?
- **Weather Forecasting**: How far back should models look?
- **Recommendation Systems**: How much user history improves recommendations?

### 7.3 Transferable Insights

Findings from this experiment may transfer to:
- Next POI prediction (semantic location prediction)
- Transportation mode prediction
- Dwell time prediction
- Trajectory forecasting

---

## 8. Chapter Summary

### Key Points

1. **Next location prediction** is a fundamental problem with wide applications
2. **Temporal window selection** is a crucial but underexplored design decision
3. **Trade-offs exist** between more history (better patterns) and less (efficiency, privacy)
4. **This experiment** systematically evaluates 1-7 day windows on two datasets
5. **Results will inform** both system design and research directions

### What's Next

The following document ([Theoretical Foundation](./03_theoretical_foundation.md)) provides the scientific basis for understanding why temporal context matters, including:
- Human mobility theory
- Temporal periodicity in behavior
- Mathematical foundations of sequence modeling

---

## References

1. Gonzalez, M. C., Hidalgo, C. A., & Barabasi, A. L. (2008). Understanding individual human mobility patterns. *Nature*, 453(7196), 779-782.

2. Song, C., Qu, Z., Blumm, N., & Barabási, A. L. (2010). Limits of predictability in human mobility. *Science*, 327(5968), 1018-1021.

3. Feng, J., et al. (2018). DeepMove: Predicting human mobility with attentional recurrent networks. *WWW*.

4. Luca, M., et al. (2021). A survey on deep learning for human mobility. *ACM Computing Surveys*.

---

## Document Metadata

| Field | Value |
|-------|-------|
| **Version** | 1.0 |
| **Created** | 2026-01-02 |
| **Word Count** | ~1,800 |
| **Status** | Final |

---

**Navigation**: [← Executive Summary](./01_executive_summary.md) | [Index](./INDEX.md) | [Next: Theoretical Foundation →](./03_theoretical_foundation.md)
