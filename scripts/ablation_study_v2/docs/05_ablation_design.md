# 5. Ablation Study Design

## Detailed Explanation of Each Ablation Variant

---

## 5.1 Overview of Ablation Variants

We evaluate 9 model configurations, systematically testing each major component:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         ABLATION VARIANTS OVERVIEW                            │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  1. Full Model (Baseline)     - All components enabled                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  CORE MECHANISM ABLATIONS:                                              │ │
│  │  2. No Pointer Mechanism      - Tests pointer's importance              │ │
│  │  3. No Generation Head        - Tests generation's importance           │ │
│  │  4. No Adaptive Gate          - Tests dynamic blending                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  FEATURE ABLATIONS:                                                     │ │
│  │  5. No Temporal Embeddings    - Tests time features                     │ │
│  │  6. No User Embedding         - Tests personalization                   │ │
│  │  7. No Position-from-End      - Tests recency encoding                  │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐ │
│  │  ARCHITECTURAL ABLATIONS:                                               │ │
│  │  8. No Position Bias          - Tests pointer attention bias            │ │
│  │  9. Single Transformer Layer  - Tests model depth                       │ │
│  └─────────────────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 5.2 Variant 1: Full Model (Baseline)

### Description
The complete PointerGeneratorTransformer model with all components enabled. This serves as the reference point for all comparisons.

### Configuration
```python
ablation_type = 'full'
```

### Components Enabled
| Component | Status |
|-----------|--------|
| Location Embedding | ✅ |
| User Embedding | ✅ |
| Temporal Embeddings | ✅ |
| Position-from-End | ✅ |
| Transformer Encoder (N layers) | ✅ |
| Pointer Mechanism | ✅ |
| Position Bias | ✅ |
| Generation Head | ✅ |
| Adaptive Gate | ✅ |

### Expected Performance
- **GeoLife**: ~51.39% Acc@1
- **DIY**: ~56.58% Acc@1

### Purpose
Establishes the baseline performance against which all ablations are compared.

---

## 5.3 Variant 2: No Pointer Mechanism

### Description
Removes the pointer mechanism entirely, leaving only the generation head for prediction.

### What Changes
```python
# Forward pass without pointer
if ablation_type == 'no_pointer':
    # Only generation distribution
    final_probs = F.softmax(self.gen_head(context), dim=-1)
```

### Visual
```
FULL MODEL:                          NO POINTER:
┌─────────────────────┐              ┌─────────────────────┐
│     Transformer     │              │     Transformer     │
└─────────┬───────────┘              └─────────┬───────────┘
          │                                    │
    ┌─────┴─────┐                              │
    │           │                              │
┌───┴───┐   ┌───┴───┐                      ┌───┴───┐
│Pointer│   │  Gen  │                      │  Gen  │ (only)
└───┬───┘   └───┬───┘                      └───┬───┘
    │           │                              │
    └─────┬─────┘                              │
          │                                    │
      [Gate]                              [No Gate]
          │                                    │
      Output                               Output
```

### Hypothesis
**High Impact Expected**: The pointer mechanism is the core innovation for capturing repetitive mobility patterns.

### What We're Testing
- Is the copy mechanism essential?
- Can generation alone match the full model?
- How much do users revisit known locations?

### Expected Outcome
Significant performance drop because:
1. Most locations are revisits
2. Generation must predict across entire vocabulary
3. Historical context is not directly leveraged

---

## 5.4 Variant 3: No Generation Head

### Description
Removes the generation head, relying solely on the pointer mechanism.

### What Changes
```python
# Forward pass without generation
if ablation_type == 'no_generation':
    # Only pointer distribution
    final_probs = self._compute_pointer_dist(x, encoded, context, mask, pos_from_end)
```

### Visual
```
FULL MODEL:                          NO GENERATION:
┌─────────────────────┐              ┌─────────────────────┐
│     Transformer     │              │     Transformer     │
└─────────┬───────────┘              └─────────┬───────────┘
          │                                    │
    ┌─────┴─────┐                              │
    │           │                              │
┌───┴───┐   ┌───┴───┐                      ┌───┴───┐
│Pointer│   │  Gen  │                      │Pointer│ (only)
└───┬───┘   └───┬───┘                      └───┬───┘
    │           │                              │
    └─────┬─────┘                              │
          │                                    │
      [Gate]                              [No Gate]
          │                                    │
      Output                               Output
```

### Hypothesis
**Moderate Impact Expected**: Generation provides vocabulary-wide prediction capability.

### What We're Testing
- Can copying alone handle all predictions?
- Are novel locations (not in history) common?
- Is vocabulary-wide prediction necessary?

### Expected Outcome
Possibly minor impact if:
- Most predictions are revisits
- Novel locations are rare in test set
- Pointer captures sufficient distribution

---

## 5.5 Variant 4: No Adaptive Gate (Fixed 0.5)

### Description
Replaces the learned adaptive gate with a fixed 0.5 blending.

### What Changes
```python
# Forward pass with fixed gate
if ablation_type == 'no_gate':
    gate = torch.full((batch_size, 1), 0.5, device=device)  # Fixed 0.5
else:
    gate = self.ptr_gen_gate(context)  # Learned gate

final_probs = gate * ptr_dist + (1 - gate) * gen_probs
```

### Visual
```
FULL MODEL:                          NO GATE:
┌─────────────┐  ┌─────────────┐     ┌─────────────┐  ┌─────────────┐
│   Pointer   │  │ Generation  │     │   Pointer   │  │ Generation  │
└──────┬──────┘  └──────┬──────┘     └──────┬──────┘  └──────┬──────┘
       │                │                   │                │
       └────────┬───────┘                   └────────┬───────┘
                │                                    │
        ┌───────┴───────┐                    ┌───────┴───────┐
        │ Learned Gate  │                    │ Fixed = 0.5   │
        │ g = σ(W·ctx)  │                    │               │
        └───────┬───────┘                    └───────┬───────┘
                │                                    │
    P = g·Ptr + (1-g)·Gen                P = 0.5·Ptr + 0.5·Gen
```

### Hypothesis
**Moderate Impact Expected**: The gate learns context-dependent blending.

### What We're Testing
- Does learned blending outperform fixed?
- Is the optimal blend always 0.5?
- Does context affect the pointer/gen balance?

### Expected Outcome
Performance drop because:
- Some contexts favor pointer (repetitive)
- Some contexts favor generation (novel)
- Fixed blend cannot adapt

---

## 5.6 Variant 5: No Temporal Embeddings

### Description
Removes all temporal features (time, weekday, duration, recency).

### What Changes
```python
# Building embeddings without temporal
embeddings = [self.loc_emb(x)]  # Location

if self.use_user:
    embeddings.append(user_emb)

# SKIP temporal embeddings
# if self.use_temporal:
#     embeddings.append(temporal)

if self.use_pos_from_end:
    embeddings.append(pos_emb)
```

### Components Removed
| Temporal Feature | Purpose | Removed |
|------------------|---------|---------|
| Time of Day | Captures daily patterns | ✅ Removed |
| Weekday | Captures weekly patterns | ✅ Removed |
| Duration | Captures activity type | ✅ Removed |
| Recency | Captures visit freshness | ✅ Removed |

### Hypothesis
**High Impact Expected**: Time is a strong predictor of location.

### What We're Testing
- Do time-based patterns exist?
- How important is temporal context?
- Can location alone predict next location?

### Examples of Temporal Patterns
```
Time-based:
- 8:00 AM → High probability of "Work"
- 12:00 PM → High probability of "Restaurant"
- 6:00 PM → High probability of "Gym" or "Home"

Weekday-based:
- Saturday → High probability of "Shopping"
- Sunday → High probability of "Church" or "Home"
```

### Expected Outcome
Performance drop because temporal patterns are strong predictors.

---

## 5.7 Variant 6: No User Embedding

### Description
Removes user-specific embeddings, making predictions user-agnostic.

### What Changes
```python
# Building embeddings without user
embeddings = [self.loc_emb(x)]  # Location

# SKIP user embedding
# if self.use_user:
#     embeddings.append(user_emb)

if self.use_temporal:
    embeddings.append(temporal)
```

### Visual
```
WITH USER EMBEDDING:                 WITHOUT USER EMBEDDING:
┌─────────────────────────────────┐  ┌─────────────────────────────────┐
│  User A: [Home→Work→Gym pattern]│  │  [Generic pattern for all users]│
│  User B: [Home→School→Home]     │  │                                 │
│  User C: [Home→Work→Bar]        │  │                                 │
└─────────────────────────────────┘  └─────────────────────────────────┘
```

### Hypothesis
**Moderate Impact Expected**: Users have individual habits.

### What We're Testing
- Do individual mobility patterns matter?
- Is personalization important?
- Can generic patterns suffice?

### Expected Outcome
Performance drop for diverse user populations, less drop for homogeneous populations.

---

## 5.8 Variant 7: No Position-from-End

### Description
Removes the position-from-end embedding that encodes recency.

### What Changes
```python
# Building embeddings without position-from-end
embeddings = [self.loc_emb(x)]

if self.use_user:
    embeddings.append(user_emb)

if self.use_temporal:
    embeddings.append(temporal)

# SKIP position-from-end
# if self.use_pos_from_end:
#     embeddings.append(pos_emb)
```

### Visual Explanation
```
Sequence: [A, B, C, D, E] → Predict F

WITH POS-FROM-END:                   WITHOUT POS-FROM-END:
Position:       1  2  3  4  5        Position:       1  2  3  4  5
Pos-from-end:   5  4  3  2  1        No recency info, all positions equal
                ↓  ↓  ↓  ↓  ↓        
                More recent = 
                smaller pos_from_end
```

### Hypothesis
**Minor to Moderate Impact Expected**: Recency is encoded elsewhere too.

### What We're Testing
- Is explicit recency encoding needed?
- Does position bias in pointer capture this?
- Does sinusoidal position encoding suffice?

### Expected Outcome
Moderate drop if recency matters and isn't captured elsewhere.

---

## 5.9 Variant 8: No Position Bias

### Description
Removes the learnable position bias from the pointer attention.

### What Changes
```python
# Pointer attention WITHOUT position bias
ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)

# SKIP position bias
# ptr_scores = ptr_scores + self.position_bias[pos_from_end]

ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
```

### Visual
```
WITH POSITION BIAS:                  WITHOUT POSITION BIAS:
Attention scores: [0.1, 0.2, 0.3, 0.4, 0.5]     [0.1, 0.2, 0.3, 0.4, 0.5]
Position bias:    [0.0, 0.0, 0.1, 0.2, 0.3]     [0.0, 0.0, 0.0, 0.0, 0.0]
                  ─────────────────────────     ─────────────────────────
Final scores:     [0.1, 0.2, 0.4, 0.6, 0.8]     [0.1, 0.2, 0.3, 0.4, 0.5]

→ Position bias adds learned preference for certain positions (typically recent)
```

### Hypothesis
**Minor Impact Expected**: Other recency mechanisms exist.

### What We're Testing
- Is explicit position bias needed?
- Does position-from-end capture this?
- Does the model learn recency preference elsewhere?

### Expected Outcome
Minor drop if redundant with other position encodings.

---

## 5.10 Variant 9: Single Transformer Layer

### Description
Reduces the transformer encoder from N layers to just 1 layer.

### What Changes
```python
# Adjust number of layers
actual_num_layers = 1 if ablation_type == 'single_layer' else num_layers
self.transformer = nn.TransformerEncoder(encoder_layer, actual_num_layers)
```

### Visual
```
FULL MODEL (2 layers):               SINGLE LAYER:
┌─────────────────────┐              ┌─────────────────────┐
│  Encoder Layer 1    │              │  Encoder Layer 1    │
├─────────────────────┤              └─────────────────────┘
│  Encoder Layer 2    │              (Only 1 layer)
└─────────────────────┘              
```

### Hypothesis
**Minor Impact Expected**: Sequence modeling may not need depth.

### What We're Testing
- Is deep modeling necessary?
- Can shallow networks suffice?
- What's the complexity-performance tradeoff?

### Expected Outcome
Minimal drop if:
- Sequences are short
- Patterns are simple
- Single layer captures enough

---

## 5.11 Summary Table

| Variant | Type | Component Removed | Expected Impact |
|---------|------|-------------------|-----------------|
| Full | Baseline | None | Reference |
| No Pointer | Core | Pointer mechanism | **Critical** |
| No Generation | Core | Generation head | Moderate |
| No Gate | Core | Adaptive gate | Moderate |
| No Temporal | Feature | Time embeddings | **High** |
| No User | Feature | User embedding | Moderate |
| No Pos-from-End | Feature | Recency embedding | Minor |
| No Position Bias | Arch | Pointer bias | Minor |
| Single Layer | Arch | Transformer depth | Minor |

---

## 5.12 Implementation Details

### Component Enabling Logic

```python
class PointerGeneratorTransformerAblation(nn.Module):
    def __init__(self, ..., ablation_type='full'):
        # Feature flags based on ablation type
        self.use_user = ablation_type != 'no_user'
        self.use_temporal = ablation_type != 'no_temporal'
        self.use_pos_from_end = ablation_type != 'no_pos_from_end'
        self.use_pointer = ablation_type != 'no_pointer'
        self.use_generation = ablation_type != 'no_generation'
        self.use_gate = ablation_type != 'no_gate'
        self.use_position_bias = ablation_type != 'no_position_bias'
        
        # Adjust layers for single_layer ablation
        actual_num_layers = 1 if ablation_type == 'single_layer' else num_layers
```

### Dynamic Input Dimension

```python
# Calculate input dimension based on enabled components
input_dim = d_model  # Location (always)

if self.use_user:
    input_dim += d_model

if self.use_temporal:
    input_dim += d_model // 4 * 4  # 4 temporal features

if self.use_pos_from_end:
    input_dim += d_model // 4

# Project to model dimension
self.input_proj = nn.Linear(input_dim, d_model)
```

---

*Next: [06_experimental_setup.md](06_experimental_setup.md) - Datasets, hyperparameters, and training protocol*
