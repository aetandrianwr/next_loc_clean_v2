# Components Deep Dive

This document provides an in-depth analysis of each component in the Pointer Generator Transformer, explaining the intuition, implementation, input/output, and justification for each design choice.

---

## 1. Location Embedding

### 1.1 What It Does

Maps discrete location IDs to continuous vector representations.

### 1.2 Implementation

```python
self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
```

### 1.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Location IDs: `[batch_size, seq_len]` dtype=long |
| **Output** | Embeddings: `[batch_size, seq_len, d_model]` dtype=float |

### 1.4 Intuition

- Each location (e.g., "Home", "Office", "Grocery Store") gets a unique vector
- Similar locations (by visitation patterns) will have similar embeddings after training
- The embedding space captures semantic relationships

### 1.5 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| `padding_idx=0` | Location ID 0 reserved for padding, embedding is zero |
| Dimension `d_model` | Match model dimension for easy combination |
| Learnable | Let the model discover location relationships |

### 1.6 What the Model Learns

After training, the location embeddings encode:
- **Functional similarity**: Coffee shops cluster together
- **Spatial proximity**: Nearby locations may have similar embeddings
- **Temporal patterns**: Locations visited at similar times cluster

---

## 2. User Embedding

### 2.1 What It Does

Maps user IDs to continuous vectors capturing user-specific preferences.

### 2.2 Implementation

```python
self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
```

### 2.3 Input/Output

| | Description |
|--|-------------|
| **Input** | User ID: `[batch_size]` dtype=long |
| **Output** | User embedding: `[batch_size, d_model]` → expanded to `[batch_size, seq_len, d_model]` |

### 2.4 Intuition

- Different users have different mobility patterns
- User A might prefer restaurants, User B might prefer parks
- The embedding captures these individual preferences

### 2.5 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| Expand to seq_len | Apply same user context to all positions |
| Same dimension as location | Easy concatenation and combination |
| Learnable | Learn user preferences from data |

### 2.6 What the Model Learns

The user embedding captures:
- **Location preferences**: Which locations this user typically visits
- **Temporal preferences**: When this user is typically active
- **Mobility style**: Regular vs. exploratory behavior

### 2.7 Ablation Impact

Removing user embedding:
- GeoLife: -3.94% Acc@1
- DIY: -0.77% Acc@1

**Interpretation**: User personalization matters more when users have distinct patterns (GeoLife).

---

## 3. Time-of-Day Embedding

### 3.1 What It Does

Captures the time of day when each location was visited.

### 3.2 Implementation

```python
self.time_emb = nn.Embedding(97, d_model // 4)  # 96 intervals + 1 padding
```

### 3.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Time indices: `[batch_size, seq_len]` values 0-96 |
| **Output** | Time embeddings: `[batch_size, seq_len, d_model // 4]` |

### 3.4 Intuition

- Human activities follow daily cycles
- 8 AM: Commute to work
- 12 PM: Lunch
- 6 PM: Return home
- 10 PM: Sleep

The model needs to understand these patterns.

### 3.5 Time Discretization

```
Minutes 0-14    → Index 1
Minutes 15-29   → Index 2
...
Minutes 1425-1439 → Index 96
```

15-minute intervals capture most temporal patterns without being too fine-grained.

### 3.6 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| 96 intervals | 15-minute granularity balances precision and generalization |
| Dimension d/4 | Temporal features are auxiliary, not primary |
| Learnable | Capture non-linear temporal patterns |

### 3.7 Ablation Impact

Removing time-of-day embedding:
- GeoLife: -2.03% Acc@1
- DIY: -0.71% Acc@1

---

## 4. Weekday Embedding

### 4.1 What It Does

Captures the day of the week for each visit.

### 4.2 Implementation

```python
self.weekday_emb = nn.Embedding(8, d_model // 4)  # 7 days + 1 padding
```

### 4.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Weekday indices: `[batch_size, seq_len]` values 0-7 |
| **Output** | Weekday embeddings: `[batch_size, seq_len, d_model // 4]` |

### 4.4 Intuition

- Weekly patterns differ significantly
- Weekdays: Work, regular routines
- Weekends: Leisure, different locations

### 4.5 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| 7+1 categories | 7 days plus padding |
| Dimension d/4 | Auxiliary feature |
| Learnable | Capture day-specific patterns |

### 4.6 Ablation Impact

Removing weekday embedding:
- GeoLife: -4.34% Acc@1
- DIY: -0.53% Acc@1

**Interpretation**: Weekday patterns are particularly important in GeoLife (GPS tracking shows stronger work patterns).

---

## 5. Recency Embedding

### 5.1 What It Does

Captures how many days ago each visit occurred relative to the prediction target.

### 5.2 Implementation

```python
self.recency_emb = nn.Embedding(9, d_model // 4)  # 8 recency levels + 1 padding
```

### 5.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Days ago: `[batch_size, seq_len]` values 0-8 (clamped) |
| **Output** | Recency embeddings: `[batch_size, seq_len, d_model // 4]` |

### 5.4 Intuition

**Key Insight**: Recent visits are more predictive than older visits.

- Visit from yesterday: Highly relevant
- Visit from last week: Somewhat relevant
- Visit from 3 weeks ago: Less relevant

The model needs to weight recent information more heavily.

### 5.5 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| 8 levels | Distinguish recent (0-2 days) from older (7+ days) |
| Clamping at 8 | Visits older than 8 days treated equally |
| Learnable | Let model determine recency importance |

### 5.6 Ablation Impact

Removing recency embedding:
- GeoLife: **-6.45%** Acc@1 (highest among temporal features!)
- DIY: -1.08% Acc@1

**Interpretation**: Recency is the most important temporal signal. The model heavily relies on knowing which visits are recent.

---

## 6. Duration Embedding

### 6.1 What It Does

Captures how long the user stayed at each location.

### 6.2 Implementation

```python
self.duration_emb = nn.Embedding(100, d_model // 4)  # 100 duration buckets
```

### 6.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Duration buckets: `[batch_size, seq_len]` values 0-99 |
| **Output** | Duration embeddings: `[batch_size, seq_len, d_model // 4]` |

### 6.4 Duration Discretization

```python
duration_bucket = duration_minutes // 30  # 30-minute buckets
```

- Bucket 0: 0-29 minutes (quick visit)
- Bucket 1: 30-59 minutes (short stay)
- Bucket 2+: 60+ minutes (longer stay)

### 6.5 Intuition

Duration reveals the nature of the visit:
- **Quick stop** (5 min): Coffee pickup, ATM
- **Medium stay** (1 hour): Restaurant, shopping
- **Long stay** (8+ hours): Work, home

### 6.6 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| 100 buckets | Cover up to 50 hours (99 × 30 min) |
| 30-min granularity | Balance between precision and generalization |
| Learnable | Capture duration-location relationships |

### 6.7 Ablation Impact

Removing duration embedding:
- GeoLife: -2.26% Acc@1
- DIY: -1.46% Acc@1

---

## 7. Position-from-End Embedding

### 7.1 What It Does

Captures the position of each location relative to the end of the sequence.

### 7.2 Implementation

```python
self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
```

### 7.3 Computation

```python
positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
pos_emb = self.pos_from_end_emb(pos_from_end)
```

### 7.4 Input/Output

| | Description |
|--|-------------|
| **Input** | Computed from positions and lengths |
| **Output** | Position embeddings: `[batch_size, seq_len, d_model // 4]` |

### 7.5 Intuition

**Key Insight**: The most recent positions are most predictive for pointing.

Position-from-end ensures:
- Position 1 from end = most recent visit
- Position 2 from end = second most recent
- etc.

This is different from position-from-start which depends on sequence length.

### 7.6 Example

```
Sequence: [A, B, C, D, E]  (length=5)

Position from start: [0, 1, 2, 3, 4]
Position from end:   [4, 3, 2, 1, 0]  ← E is position 0 from end
```

### 7.7 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| From end | Recent positions are more important |
| Learnable | Model can learn position-specific biases |
| Combined with sinusoidal | Captures both absolute and relative position |

### 7.8 Ablation Impact

Removing position-from-end embedding:
- GeoLife: -2.80% Acc@1
- DIY: -0.63% Acc@1

---

## 8. Input Projection Layer

### 8.1 What It Does

Combines and projects all embeddings to the model dimension.

### 8.2 Implementation

```python
input_dim = d_model * 2 + d_model // 4 * 5  # loc + user + 5 temporal
self.input_proj = nn.Linear(input_dim, d_model)
self.input_norm = nn.LayerNorm(d_model)
```

### 8.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Concatenated embeddings: `[batch_size, seq_len, input_dim]` |
| **Output** | Projected: `[batch_size, seq_len, d_model]` |

### 8.4 Intuition

- Different embeddings have different scales and semantics
- Linear projection learns how to combine them
- LayerNorm stabilizes the combined representation

### 8.5 Why This Design?

| Design Choice | Justification |
|---------------|---------------|
| Linear projection | Learn optimal combination weights |
| LayerNorm after | Stabilize for Transformer input |
| Single layer | Sufficient for embedding combination |

---

## 9. Sinusoidal Positional Encoding

### 9.1 What It Does

Adds absolute position information to the sequence.

### 9.2 Implementation

```python
def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d_model)
    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # [1, max_len, d_model]
```

### 9.3 Mathematical Formula

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 9.4 Intuition

- Transformers are permutation-invariant without position encoding
- Sinusoidal encoding provides unique position signatures
- Smooth transitions between positions
- Generalizes to longer sequences

### 9.5 Why Sinusoidal (Not Learned)?

| Approach | Pros | Cons |
|----------|------|------|
| Sinusoidal | Generalizes to longer sequences | Fixed patterns |
| Learned | Optimal for training data | May not generalize |

We use sinusoidal because:
- Sequences can vary significantly in length
- The position-from-end embedding provides learned position info

### 9.6 Ablation Impact

Removing sinusoidal encoding:
- GeoLife: -0.43% Acc@1 (minimal!)
- DIY: +0.18% Acc@1 (slightly better without)

**Interpretation**: Position-from-end embedding already captures sufficient positional information. Sinusoidal encoding is somewhat redundant.

---

## 10. Transformer Encoder

### 10.1 What It Does

Processes the sequence to capture dependencies between positions.

### 10.2 Implementation

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=d_model,
    nhead=nhead,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    activation='gelu',
    batch_first=True,
    norm_first=True  # Pre-norm
)
self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
```

### 10.3 Input/Output

| | Description |
|--|-------------|
| **Input** | Hidden states: `[batch_size, seq_len, d_model]`, Mask: `[batch_size, seq_len]` |
| **Output** | Encoded states: `[batch_size, seq_len, d_model]` |

### 10.4 Intuition

The Transformer encoder allows each position to attend to all other positions:
- "When I visited the gym, I usually go to the smoothie shop after"
- "Visits on Monday correlate with visits on the next Monday"
- "Long stays at work are followed by short dinner visits"

### 10.5 Pre-Norm Architecture

```
Standard (Post-Norm):
    x → SubLayer → Add → LayerNorm → output

Our (Pre-Norm):
    x → LayerNorm → SubLayer → Add → output
```

**Why Pre-Norm?**
- More stable gradients
- Better training dynamics
- Can use larger learning rates

### 10.6 GELU Activation

```
GELU(x) = x × Φ(x)
```

Where Φ is the standard Gaussian CDF.

**Why GELU?**
- Smoother than ReLU
- Better gradient flow
- Standard in modern Transformers (BERT, GPT)

### 10.7 Ablation Impact (Single Layer)

Using only 1 Transformer layer:
- GeoLife: -2.91% Acc@1
- DIY: -0.78% Acc@1

**Interpretation**: Multiple layers help, but even a single layer captures most of the benefit.

---

## 11. Pointer Mechanism

### 11.1 What It Does

Computes attention over the input sequence and converts it to a probability distribution over locations.

### 11.2 Implementation

```python
self.pointer_query = nn.Linear(d_model, d_model)
self.pointer_key = nn.Linear(d_model, d_model)
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
```

### 11.3 Computation

```python
# 1. Project context to query
query = self.pointer_query(context).unsqueeze(1)  # [B, 1, d]

# 2. Project encoded sequence to keys
keys = self.pointer_key(encoded)  # [B, seq, d]

# 3. Compute attention scores
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
# [B, seq]

# 4. Add learnable position bias
ptr_scores = ptr_scores + self.position_bias[pos_from_end]

# 5. Mask padding
ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))

# 6. Softmax
ptr_probs = F.softmax(ptr_scores, dim=-1)  # [B, seq]

# 7. Scatter to location vocabulary
ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
ptr_dist.scatter_add_(1, x, ptr_probs)  # [B, V]
```

### 11.4 Input/Output

| | Description |
|--|-------------|
| **Input** | Context: `[B, d]`, Encoded: `[B, seq, d]`, x: `[B, seq]` |
| **Output** | Pointer distribution: `[B, num_locations]` |

### 11.5 Intuition

The pointer mechanism asks: "Which position in my history should I copy from?"

1. **Query**: "What am I looking for?" (from context)
2. **Keys**: "What do I have at each position?" (from encoded)
3. **Attention**: "How relevant is each position?"
4. **Position bias**: "Prefer recent positions"
5. **Scatter**: "Map positions to locations"

### 11.6 Position Bias

```python
self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
```

This learnable vector adds bias based on position-from-end:
- After training, recent positions (low index) get higher bias
- This explicitly encourages attending to recent visits

### 11.7 Scatter Operation

Why scatter? Because multiple positions might have the same location:

```
Sequence: [Home, Work, Gym, Work, Home]
Positions:   0     1    2    3     4

If attention weights are [0.1, 0.2, 0.1, 0.2, 0.4]:
- Home gets: 0.1 + 0.4 = 0.5
- Work gets: 0.2 + 0.2 = 0.4
- Gym gets: 0.1
```

### 11.8 Ablation Impact

Removing pointer mechanism:
- GeoLife: **-20.96%** Acc@1 (CRITICAL!)
- DIY: **-5.64%** Acc@1

**Interpretation**: The pointer mechanism is the most critical component. Without it, performance drops dramatically, especially on GeoLife.

---

## 12. Generation Head

### 12.1 What It Does

Predicts a probability distribution over the full location vocabulary.

### 12.2 Implementation

```python
self.gen_head = nn.Linear(d_model, num_locations)
```

### 12.3 Computation

```python
gen_probs = F.softmax(self.gen_head(context), dim=-1)  # [B, V]
```

### 12.4 Input/Output

| | Description |
|--|-------------|
| **Input** | Context: `[B, d_model]` |
| **Output** | Generation distribution: `[B, num_locations]` |

### 12.5 Intuition

The generation head can predict:
- Locations never visited before
- Locations not in the current sequence
- Based on learned location-context associations

### 12.6 Why Both Pointer and Generation?

| Scenario | Best Strategy |
|----------|---------------|
| Going home (visited 100x) | Pointer |
| Going to new restaurant | Generation |
| Going to place visited once long ago | Either |

The model needs both strategies to handle all scenarios.

### 12.7 Ablation Impact

Removing generation head:
- GeoLife: -4.28% Acc@1
- DIY: +0.04% Acc@1

**Interpretation**: Generation is important for GeoLife (more location exploration), but DIY users stick to their history (pointer is sufficient).

---

## 13. Pointer-Generation Gate

### 13.1 What It Does

Learns to balance pointer and generation distributions.

### 13.2 Implementation

```python
self.ptr_gen_gate = nn.Sequential(
    nn.Linear(d_model, d_model // 2),
    nn.GELU(),
    nn.Linear(d_model // 2, 1),
    nn.Sigmoid()
)
```

### 13.3 Computation

```python
gate = self.ptr_gen_gate(context)  # [B, 1]
final_probs = gate * ptr_dist + (1 - gate) * gen_probs  # [B, V]
```

### 13.4 Input/Output

| | Description |
|--|-------------|
| **Input** | Context: `[B, d_model]` |
| **Output** | Gate value: `[B, 1]` (0 to 1) |

### 13.5 Intuition

The gate learns context-dependent strategies:
- **High gate (→1)**: "This looks like a repeat visit, use pointer"
- **Low gate (→0)**: "This might be somewhere new, use generation"
- **Medium gate**: "Hedge between both"

### 13.6 What Influences the Gate?

The gate is computed from the context, which encodes:
- User identity (some users are more exploratory)
- Temporal context (weekends might have more exploration)
- Recent history (familiar patterns → higher gate)

### 13.7 Ablation Impact

Removing gate (using fixed 0.5 blend):
- GeoLife: -4.88% Acc@1
- DIY: -1.54% Acc@1

**Interpretation**: Adaptive gating significantly outperforms fixed blending.

---

## 14. Final Output Computation

### 14.1 Implementation

```python
final_probs = gate * ptr_dist + (1 - gate) * gen_probs
return torch.log(final_probs + 1e-10)
```

### 14.2 Why Log Probabilities?

1. **Numerical stability**: Avoid underflow with small probabilities
2. **Cross-entropy loss**: Works with log probabilities
3. **Gradient flow**: Better gradient magnitude

### 14.3 Why Add ε (1e-10)?

Prevent log(0) = -inf:
- If a location has zero probability, log would be undefined
- Small epsilon ensures all log values are finite

---

## Component Summary Table

| Component | Dimension | Parameters | Ablation Impact (GeoLife) | Ablation Impact (DIY) |
|-----------|-----------|------------|---------------------------|------------------------|
| Location Embedding | d_model | V × d | Essential | Essential |
| User Embedding | d_model | U × d | -3.94% | -0.77% |
| Time Embedding | d/4 | 97 × d/4 | -2.03% | -0.71% |
| Weekday Embedding | d/4 | 8 × d/4 | -4.34% | -0.53% |
| Recency Embedding | d/4 | 9 × d/4 | **-6.45%** | -1.08% |
| Duration Embedding | d/4 | 100 × d/4 | -2.26% | -1.46% |
| Pos-from-End Embedding | d/4 | L × d/4 | -2.80% | -0.63% |
| Sinusoidal PE | d | Fixed | -0.43% | +0.18% |
| Transformer | d | ~100K/layer | -2.91% (single) | -0.78% (single) |
| Pointer Mechanism | d | 2 × d² | **-20.96%** | **-5.64%** |
| Generation Head | V | d × V | -4.28% | +0.04% |
| Gate | 1 | d × d/2 + d/2 | -4.88% | -1.54% |

---

*Next: [05_TRAINING_PIPELINE.md](05_TRAINING_PIPELINE.md) - Training Process Documentation*
