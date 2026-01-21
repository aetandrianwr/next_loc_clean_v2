# Theoretical Background and Motivation

## 1. Introduction to Next Location Prediction

### 1.1 The Problem Domain

**Next location prediction** is a fundamental problem in human mobility analysis that aims to forecast where an individual will visit next based on their historical movement patterns. This task sits at the intersection of:

- **Spatial Data Mining**: Analyzing location sequences and geographic patterns
- **Temporal Sequence Modeling**: Capturing time-dependent behaviors
- **Personalization**: Learning individual user preferences and routines

### 1.2 Why is This Problem Hard?

Human mobility is influenced by many factors:

1. **Temporal Patterns**
   - Time of day (work during business hours, home at night)
   - Day of week (weekday vs. weekend routines)
   - Seasonal variations

2. **Personal Preferences**
   - Individual habits and routines
   - Preference for certain types of locations
   - Social connections

3. **Contextual Factors**
   - Weather conditions
   - Special events
   - Transportation availability

4. **Spatial Constraints**
   - Distance limitations
   - Geographic barriers
   - Urban structure

---

## 2. Key Observations About Human Mobility

### 2.1 Repetitiveness of Movement

**Critical Insight**: Human mobility is highly repetitive.

Studies have shown that:
- People visit only **25-50 unique locations** regularly
- **70-80% of visits** are to locations previously visited
- The most frequently visited locations (home, work) account for the majority of time

This observation is fundamental to the Pointer Generator Transformer design - the pointer mechanism allows the model to "copy" locations from the user's history, which is often the correct strategy.

### 2.2 Recency Effect

Recent visits are more predictive than older visits:
- If you went to a coffee shop yesterday, you're more likely to go there today
- Locations visited long ago may no longer be relevant

The model encodes this through:
- **Position-from-end embedding**: Captures how recently each location was visited
- **Recency embedding**: Explicit encoding of days since each visit

### 2.3 Temporal Periodicity

Human activities follow temporal cycles:
- **Daily cycles**: Sleep at night, work during day
- **Weekly cycles**: Different patterns on weekends vs. weekdays
- **Hourly patterns**: Lunch breaks, commute times

The model captures this through:
- **Time-of-day embedding**: 96 intervals (15-minute slots)
- **Weekday embedding**: 7 days of the week

### 2.4 User-Specific Patterns

Different users have different routines:
- Some users have highly regular patterns (same locations daily)
- Others have more diverse mobility

The model handles this through:
- **User embedding**: Captures user-specific preferences
- **Adaptive gate**: Learns user-specific pointer-generation balance

---

## 3. Evolution of Location Prediction Models

### 3.1 Traditional Approaches

**Markov Models**
- Model transitions between locations as a Markov chain
- P(l_{n+1} | l_n) - only depends on current location
- Limitation: Cannot capture long-range dependencies

**Probabilistic Models**
- Learn location probability distributions per user
- Can capture user preferences but not sequential patterns

### 3.2 Deep Learning Approaches

**RNN/LSTM Models**
- Process sequences recurrently
- Can capture sequential dependencies
- Limitation: May struggle with very long sequences

**Transformer Models (MHSA)**
- Use self-attention to capture dependencies
- Can attend to any position in the sequence
- Limitation: Predicts from full vocabulary - inefficient for repetitive patterns

**Pointer Networks (Our Approach)**
- Explicitly model the "copy from history" strategy
- Hybrid approach: pointer + generation
- Advantage: Directly exploits mobility repetitiveness

### 3.3 Why Pointer Networks?

The key insight is that location prediction often involves **selecting from a known set** (previous visits) rather than **generating from scratch**.

Consider predicting a person's next location:
- **Scenario A**: They're going home (visited 1000 times before)
  - Pointer mechanism: Point to "home" in history ✓
  - Generation: Find "home" in vocabulary of 10,000 locations ✗
  
- **Scenario B**: They're visiting a new restaurant
  - Pointer mechanism: Cannot point (never visited) ✗
  - Generation: Generate from vocabulary ✓

The adaptive gate learns when each strategy is appropriate.

---

## 4. Pointer Networks: Theory

### 4.1 Original Pointer Networks (Vinyals et al., 2015)

Pointer Networks were introduced for problems where the output is a permutation or subset of the input:
- Traveling Salesman Problem
- Convex Hull
- Sorting

**Key Innovation**: Instead of selecting from a fixed vocabulary, point to positions in the input sequence.

**Attention as Pointing**:
```
attention_score(query, key_i) = v^T tanh(W_q * query + W_k * key_i)
pointer_prob(i) = softmax(attention_scores)
```

### 4.2 Copy Mechanism in Sequence-to-Sequence Models

The copy mechanism was later applied to text generation:
- **See et al., 2017**: "Get To The Point: Summarization with Pointer-Generator Networks"
- Combines pointing (copy from source) with generation (from vocabulary)
- Uses a gate to balance copy vs. generate

### 4.3 Our Adaptation for Location Prediction

We adapt the pointer-generator framework for location prediction:

1. **Pointer Distribution**: Attention over the input location sequence
2. **Generation Distribution**: Softmax over full location vocabulary
3. **Adaptive Gate**: Learned sigmoid function to blend distributions

```
P(location) = gate * P_pointer(location) + (1 - gate) * P_generation(location)
```

---

## 5. Transformer Architecture: Theory

### 5.1 Self-Attention Mechanism

The core of the Transformer is **scaled dot-product attention**:

```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

Where:
- **Q** (Query): What am I looking for?
- **K** (Key): What do I have?
- **V** (Value): What do I return?
- **d_k**: Dimension of keys (for scaling)

### 5.2 Multi-Head Attention

Multiple attention heads allow the model to attend to different aspects:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O

where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
```

**Intuition**: Different heads can learn:
- Head 1: Focus on recent locations
- Head 2: Focus on similar temporal contexts
- Head 3: Focus on frequently visited locations

### 5.3 Positional Encoding

Since attention is permutation-invariant, we need to encode position:

**Sinusoidal Positional Encoding**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why Sinusoidal?**
1. Each position gets a unique encoding
2. Can generalize to longer sequences
3. Relative positions can be computed via linear transformation

### 5.4 Pre-Norm vs. Post-Norm

**Post-Norm (Original)**:
```
output = LayerNorm(x + SubLayer(x))
```

**Pre-Norm (Our Approach)**:
```
output = x + SubLayer(LayerNorm(x))
```

**Why Pre-Norm?**
- More stable training
- Gradients flow more easily
- Often leads to better performance

---

## 6. Embedding Theory

### 6.1 Why Embeddings?

Categorical variables (locations, users, time) cannot be directly used as neural network inputs. Embeddings map discrete values to continuous vector spaces:

```
embedding: {0, 1, 2, ..., N-1} → R^d
```

**Properties of Good Embeddings**:
- Similar items have similar embeddings
- Capture semantic relationships
- Learn from data

### 6.2 Location Embeddings

Each location ID maps to a dense vector:
```
loc_embedding(location_id) → R^{d_model}
```

During training, the model learns:
- Similar locations (same type, nearby) have similar embeddings
- Frequently co-occurring locations have related embeddings

### 6.3 Temporal Embeddings

**Time-of-Day Embedding**:
- Discretize time into 96 intervals (15-minute slots)
- Learn embedding for each interval
- Captures daily patterns (morning, afternoon, evening, night)

**Weekday Embedding**:
- 7 embeddings for days of the week
- Captures weekly patterns (weekday vs. weekend)

**Recency Embedding**:
- How many days ago was this visit?
- Captures the recency effect
- More recent visits are more predictive

**Duration Embedding**:
- How long was the visit?
- Short visits vs. long stays have different characteristics

### 6.4 Position-from-End Embedding

**Key Innovation**: Embed how far each position is from the sequence end.

```
pos_from_end = sequence_length - position
```

**Why?**
- The last few locations are most predictive
- This embedding helps the pointer mechanism focus on recent positions
- Complements sinusoidal positional encoding

---

## 7. The Pointer-Generation Gate

### 7.1 Motivation

Not all predictions should use the same strategy:
- **High pointer**: Next location is likely from history (going home)
- **High generation**: Next location might be new (visiting new restaurant)

### 7.2 Mathematical Formulation

The gate is computed from the context vector:

```python
gate = sigmoid(MLP(context))
# MLP: Linear(d_model → d_model/2) → GELU → Linear(d_model/2 → 1) → Sigmoid
```

The final distribution:
```
P_final(loc) = gate * P_pointer(loc) + (1 - gate) * P_generation(loc)
```

### 7.3 Interpretation

- **gate ≈ 1**: Model is confident location is from history
- **gate ≈ 0**: Model thinks it's a new location
- **gate ≈ 0.5**: Uncertain, hedging between both

### 7.4 Learning the Gate

The gate is learned end-to-end through backpropagation:
- If pointer is usually correct → gate learns to be high
- If generation is usually correct → gate learns to be low
- The model adapts to the data distribution

---

## 8. Position Bias in Pointer Mechanism

### 8.1 The Concept

Beyond attention scores, we add a learnable **position bias**:

```
pointer_scores = attention_scores + position_bias[pos_from_end]
```

### 8.2 Why Position Bias?

Pure attention might not capture the strong recency preference:
- The model might attend to similar locations regardless of position
- Position bias explicitly encourages attending to recent positions

### 8.3 What the Model Learns

After training, the position bias typically shows:
- High values for recent positions (position 0, 1, 2 from end)
- Lower values for older positions
- This aligns with the recency effect in human mobility

---

## 9. Mathematical Foundations

### 9.1 The Complete Forward Pass

Given:
- Location sequence: `x = [x_1, ..., x_n]`
- User: `u`
- Temporal features: `time, weekday, recency, duration`

**Step 1: Embedding**
```
loc_emb = LocationEmbedding(x)           # [n, d_model]
user_emb = UserEmbedding(u)               # [d_model]
temp_emb = concat([time_emb, weekday_emb, recency_emb, duration_emb, pos_from_end_emb])
combined = concat([loc_emb, user_emb, temp_emb])
hidden = LayerNorm(Linear(combined))
```

**Step 2: Add Positional Encoding**
```
hidden = hidden + sinusoidal_pos_encoding
```

**Step 3: Transformer Encoding**
```
encoded = TransformerEncoder(hidden, mask)
context = encoded[last_valid_position]    # [d_model]
```

**Step 4: Pointer Distribution**
```
query = Linear_Q(context)                 # [d_model]
keys = Linear_K(encoded)                  # [n, d_model]
scores = (query @ keys.T) / sqrt(d_model) # [n]
scores = scores + position_bias
pointer_probs = softmax(scores)           # [n]
P_pointer = scatter_add(pointer_probs to locations)  # [num_locations]
```

**Step 5: Generation Distribution**
```
P_generation = softmax(Linear(context))   # [num_locations]
```

**Step 6: Combine with Gate**
```
gate = sigmoid(MLP(context))              # [1]
P_final = gate * P_pointer + (1 - gate) * P_generation
output = log(P_final)                     # [num_locations]
```

### 9.2 Loss Function

**Cross-Entropy Loss with Label Smoothing**:

```
L = -sum(y_smoothed * log(P_final))

where y_smoothed = (1 - ε) * y_onehot + ε / num_classes
```

Label smoothing (ε = 0.03) helps:
- Prevent overconfident predictions
- Improve generalization
- Smooth the loss landscape

---

## 10. References

### Foundational Papers

1. **Pointer Networks**
   - Vinyals, O., Fortunato, M., & Jaitly, N. (2015). Pointer Networks. NeurIPS.
   - Introduced the concept of pointing to input positions

2. **Attention Is All You Need**
   - Vaswani, A., et al. (2017). Attention Is All You Need. NeurIPS.
   - Introduced the Transformer architecture

3. **Pointer-Generator Networks**
   - See, A., Liu, P. J., & Manning, C. D. (2017). Get To The Point: Summarization with Pointer-Generator Networks. ACL.
   - Combined copying and generation

### Location Prediction Literature

4. **DeepMove**
   - Feng, J., et al. (2018). DeepMove: Predicting Human Mobility with Attentional Recurrent Networks. WWW.
   - Attention-based RNN for location prediction

5. **Context-aware Multi-head Self-attention**
   - Hong et al. (2023). Context-aware multi-head self-attentional neural network model for next location prediction.
   - Transformer-based location prediction with context

### Datasets

6. **GeoLife GPS Trajectories**
   - Zheng, Y., et al. (2009). GeoLife: A Collaborative Social Networking Service among User, Location and Trajectory. IEEE MDM.
   - GPS trajectory dataset from Microsoft Research

---

*Next: [03_MODEL_ARCHITECTURE.md](03_MODEL_ARCHITECTURE.md) - Detailed Architecture Documentation*
