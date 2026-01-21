# Adaptation Guide: From Text Summarization to Other Domains

This guide explains how to adapt the pointer-generator architecture to domains beyond text summarization. We use next location prediction as a concrete example, referencing the implementation in `pgt.py`.

## Table of Contents
1. [Understanding the Core Abstraction](#understanding-the-core-abstraction)
2. [Domain Comparison](#domain-comparison)
3. [Key Adaptations Required](#key-adaptations-required)
4. [Case Study: Next Location Prediction](#case-study-next-location-prediction)
5. [Step-by-Step Adaptation Process](#step-by-step-adaptation-process)
6. [PyTorch Implementation](#pytorch-implementation)
7. [Common Pitfalls](#common-pitfalls)
8. [Other Domain Applications](#other-domain-applications)

---

## Understanding the Core Abstraction

The pointer-generator network has a fundamental abstraction that transfers to many domains:

```
┌────────────────────────────────────────────────────────────────────────────┐
│                    CORE POINTER-GENERATOR ABSTRACTION                       │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   Given an INPUT SEQUENCE and a VOCABULARY:                                │
│                                                                             │
│   The model can either:                                                     │
│   1. GENERATE from vocabulary (new items not in input)                     │
│   2. COPY from input sequence (items seen before)                          │
│                                                                             │
│   The p_gen GATE learns when to use each strategy.                         │
│                                                                             │
└────────────────────────────────────────────────────────────────────────────┘
```

### What Makes a Problem Suitable for Pointer-Generator?

A problem is well-suited for pointer-generator when:

| Criterion | Text Summarization | Next Location Prediction |
|-----------|-------------------|-------------------------|
| Input is a sequence | ✓ Article (word sequence) | ✓ Location history (location sequence) |
| Output comes from limited set | ✓ Words | ✓ Locations |
| Copying from input is useful | ✓ Copy proper nouns | ✓ Revisit past locations |
| Novel items also needed | ✓ Generate common words | ✓ Visit new locations |
| Attention reveals relevance | ✓ Which sentences matter | ✓ Which past visits matter |

---

## Domain Comparison

```
┌────────────────────────────────────────────────────────────────────────────────────────────┐
│                           TEXT SUMMARIZATION vs LOCATION PREDICTION                         │
├────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                             │
│   TEXT SUMMARIZATION                    LOCATION PREDICTION                                 │
│   ══════════════════                    ═══════════════════                                 │
│                                                                                             │
│   Input:                                Input:                                              │
│   "Germany beat Argentina               [Home, Work, Cafe, Gym, Home]                      │
│    in the World Cup final"              (Past location visits)                             │
│                                                                                             │
│   Output:                               Output:                                             │
│   "Germany won the Cup"                 "Work" (Next location)                             │
│                                                                                             │
│   ─────────────────────────────────────────────────────────────────────────────────────    │
│                                                                                             │
│   VOCABULARY                            VOCABULARY                                          │
│   ══════════                            ══════════                                          │
│   50,000 words                          ~1,000-10,000 locations                            │
│   (Common English words)                (POIs in a city/region)                            │
│                                                                                             │
│   ─────────────────────────────────────────────────────────────────────────────────────    │
│                                                                                             │
│   COPY BENEFIT                          COPY BENEFIT                                        │
│   ═══════════                           ═══════════                                         │
│   Copy proper nouns:                    Copy frequent locations:                           │
│   "Musk", "Tesla", "SpaceX"             Home, Work, Favorite Cafe                          │
│   (Not in vocab but in article)         (User revisits same places)                        │
│                                                                                             │
│   ─────────────────────────────────────────────────────────────────────────────────────    │
│                                                                                             │
│   GENERATE BENEFIT                      GENERATE BENEFIT                                    │
│   ════════════════                      ════════════════                                    │
│   Generate common words:                Generate new destinations:                         │
│   "the", "won", "was"                   New restaurant, Airport                            │
│   (Not in article but needed)           (Never visited before)                             │
│                                                                                             │
│   ─────────────────────────────────────────────────────────────────────────────────────    │
│                                                                                             │
│   ATTENTION MEANING                     ATTENTION MEANING                                   │
│   ═════════════════                     ═════════════════                                   │
│   "Which words in article               "Which past visits                                 │
│    are relevant to summary?"             are relevant to next?"                            │
│                                                                                             │
│   Example: Attending to                 Example: Attending to                              │
│   "Germany" when generating             "Work" on weekday mornings                         │
│   subject of summary                    to predict commute                                 │
│                                                                                             │
└────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Key Adaptations Required

### 1. Encoder Architecture

| Original (Text) | Adapted (Location) |
|-----------------|-------------------|
| Bidirectional LSTM | Transformer Encoder |
| Word embeddings only | Location + User + Temporal embeddings |
| Sequential processing | Parallel attention |

**Why Change?**

1. **Transformer vs LSTM**: Locations have complex dependencies (e.g., "went to gym after work on Tuesdays") that benefit from global attention.

2. **Multiple Embeddings**: Locations have rich context:
   - **Who**: Different users have different patterns
   - **When**: Time of day, day of week matter
   - **Recency**: Recent visits are more relevant

```python
# Original: Single word embedding
emb = embedding_layer(word_ids)

# Adapted: Multiple embeddings combined
loc_emb = self.loc_emb(locations)
user_emb = self.user_emb(user_id)
time_emb = self.time_emb(time_of_day)
weekday_emb = self.weekday_emb(day_of_week)
combined = torch.cat([loc_emb, user_emb, time_emb, weekday_emb], dim=-1)
```

### 2. Vocabulary Handling

| Original (Text) | Adapted (Location) |
|-----------------|-------------------|
| 50,000 words | ~1,000-10,000 locations |
| OOV handling complex | All locations known |
| Extended vocabulary needed | Fixed vocabulary sufficient |

**Why Simpler?**

In text, OOV words are common (names, new terms). In location prediction, the set of POIs is typically fixed and known in advance.

```python
# Original: Extended vocabulary for OOVs
extended_vocab_size = vocab_size + max_article_oovs

# Adapted: Fixed vocabulary
num_locations = len(all_pois)  # No OOV handling needed
```

### 3. Decoder Structure

| Original (Text) | Adapted (Location) |
|-----------------|-------------------|
| Auto-regressive LSTM | Single prediction |
| Generate sequence of words | Predict single next location |
| Multiple decoding steps | One forward pass |

**Why Simpler?**

Text summarization generates variable-length output. Location prediction typically asks "what is the NEXT location?" - a single prediction.

```python
# Original: Loop over decoder steps
for step in range(max_dec_steps):
    decoder_output, state = decoder(input, state)
    word = sample(decoder_output)

# Adapted: Single prediction
context = encoded[batch_idx, last_idx]  # Last encoder output
prediction = model(context)  # Single location
```

### 4. Attention Mechanism

| Original (Text) | Adapted (Location) |
|-----------------|-------------------|
| Bahdanau (additive) attention | Scaled dot-product attention |
| Decoder state as query | Learned query from context |
| Coverage to prevent repetition | Position bias for recency |

**Why Different?**

1. **Scaled dot-product** is more efficient and works well with Transformers.
2. **Position bias** encodes that recent visits matter more (no coverage needed since we predict once).

```python
# Original: Bahdanau attention
e_i = v^T * tanh(W_h * h_i + W_s * s_t + w_c * coverage + b)

# Adapted: Scaled dot-product with position bias
ptr_scores = (query @ keys.T) / sqrt(d_model)
ptr_scores = ptr_scores + position_bias[pos_from_end]
```

### 5. Pointer-Generator Gate

| Original (Text) | Adapted (Location) |
|-----------------|-------------------|
| Uses context + state + input | Uses encoded context only |
| Single scalar output | Single scalar output |
| Sigmoid activation | Sigmoid activation |

**Core mechanism preserved!**

```python
# Original
p_gen = sigmoid(w_c * context + w_s * state + w_x * input + b)

# Adapted
gate = self.ptr_gen_gate(context)  # MLP with sigmoid
final = gate * ptr_dist + (1 - gate) * gen_dist
```

---

## Case Study: Next Location Prediction

### Problem Definition

```
Given: User U's location history [L₁, L₂, ..., Lₙ] with timestamps
Predict: Next location Lₙ₊₁
```

### Why Pointer-Generator?

1. **Copy (Pointer)**: Users often revisit the same locations
   - Home → Work → Home → Work (daily commute)
   - Home → Gym → Home (weekly routine)

2. **Generate**: Users also visit new places
   - First-time visits to restaurants
   - Travel to new areas

3. **Adaptive Gate**: Different users/times need different strategies
   - Routine times (morning): High copy probability → Work
   - Weekends: Higher generate probability → New places

### Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                        POINTER NETWORK V45 ARCHITECTURE                                       │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                               │
│   INPUT: Location History                                                                     │
│   ─────────────────────────                                                                   │
│                                                                                               │
│   Locations:  [Home]   [Work]   [Cafe]   [Gym]   [Home]                                     │
│   Times:       8:00    9:00    12:00    18:00   19:00                                       │
│   Weekday:     Mon     Mon      Mon      Mon     Mon                                        │
│   Recency:     5d      5d       5d       5d      5d                                         │
│                                                                                               │
│                ▼        ▼        ▼        ▼        ▼                                        │
│   ─────────────────────────────────────────────────────────────────────────────────────      │
│                                                                                               │
│   EMBEDDING LAYER                                                                             │
│   ═══════════════                                                                             │
│                                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐       │
│   │                                                                                  │       │
│   │  loc_emb(Home)  + user_emb(U1) + time_emb(8:00) + weekday_emb(Mon) + ...       │       │
│   │      [128]           [128]          [32]              [32]                      │       │
│   │                                                                                  │       │
│   │  Combined: [128 + 128 + 32 + 32 + 32 + 32 + 32] = [416]                        │       │
│   │  Projected: Linear(416, 128) → [128]                                           │       │
│   │                                                                                  │       │
│   └─────────────────────────────────────────────────────────────────────────────────┘       │
│                                                                                               │
│                ▼        ▼        ▼        ▼        ▼                                        │
│   ─────────────────────────────────────────────────────────────────────────────────────      │
│                                                                                               │
│   TRANSFORMER ENCODER                                                                         │
│   ═══════════════════                                                                         │
│                                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐       │
│   │                                                                                  │       │
│   │    ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐                        │       │
│   │    │  h₁  │   │  h₂  │   │  h₃  │   │  h₄  │   │  h₅  │                        │       │
│   │    └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘                        │       │
│   │       │          │          │          │          │                             │       │
│   │       └──────────┴──────────┴──────────┴──────────┘                             │       │
│   │                    Self-Attention                                               │       │
│   │                                                                                  │       │
│   │    3 Transformer Layers with:                                                   │       │
│   │    - 4 attention heads                                                          │       │
│   │    - GELU activation                                                            │       │
│   │    - Pre-LayerNorm                                                              │       │
│   │                                                                                  │       │
│   └─────────────────────────────────────────────────────────────────────────────────┘       │
│                                                                                               │
│                ▼        ▼        ▼        ▼        ▼                                        │
│   ─────────────────────────────────────────────────────────────────────────────────────      │
│                                                                                               │
│   OUTPUT HEADS                                                                                │
│   ════════════                                                                                │
│                                                                                               │
│   Context = encoded[last_position] = h₅                                                      │
│                                                                                               │
│   ┌──────────────────────────┐     ┌──────────────────────────┐                             │
│   │     POINTER HEAD         │     │    GENERATION HEAD        │                             │
│   │                          │     │                           │                             │
│   │   query = W_q × context  │     │   logits = W_g × context  │                             │
│   │   keys = W_k × encoded   │     │   gen_dist = softmax(logits)│                           │
│   │                          │     │                           │                             │
│   │   scores = query @ keys^T│     │   Shape: [num_locations]  │                             │
│   │   + position_bias        │     │                           │                             │
│   │                          │     │                           │                             │
│   │   ptr_dist = scatter_add │     │                           │                             │
│   │   Shape: [num_locations] │     │                           │                             │
│   │                          │     │                           │                             │
│   └────────────┬─────────────┘     └────────────┬──────────────┘                             │
│                │                                │                                            │
│                │                                │                                            │
│                ▼                                ▼                                            │
│   ─────────────────────────────────────────────────────────────────────────────────────      │
│                                                                                               │
│   POINTER-GENERATOR GATE                                                                      │
│   ══════════════════════                                                                      │
│                                                                                               │
│   ┌─────────────────────────────────────────────────────────────────────────────────┐       │
│   │                                                                                  │       │
│   │   gate = MLP(context) → sigmoid → [0.7]                                         │       │
│   │                                                                                  │       │
│   │   final_dist = gate × ptr_dist + (1 - gate) × gen_dist                         │       │
│   │              = 0.7 × ptr_dist + 0.3 × gen_dist                                 │       │
│   │                                                                                  │       │
│   │   Output: log(final_dist)  →  [batch_size, num_locations]                      │       │
│   │                                                                                  │       │
│   └─────────────────────────────────────────────────────────────────────────────────┘       │
│                                                                                               │
│   PREDICTION: "Work" (location with highest probability)                                     │
│                                                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Adaptation Process

### Step 1: Define Your Domain Mapping

```
TEXT SUMMARIZATION              YOUR DOMAIN
═══════════════════             ═══════════
Article                    →    ?
Words                      →    ?
Vocabulary                 →    ?
Summary                    →    ?
```

**Example (Location Prediction)**:
```
Article                    →    Location history
Words                      →    Locations (POIs)
Vocabulary                 →    All known POIs
Summary                    →    Next location
```

### Step 2: Identify Copy vs Generate Trade-off

Ask yourself:
1. When should the model copy from input? (Pointer)
2. When should the model generate something new? (Generator)

**Example**:
- Copy: Revisiting Home, Work, favorite places
- Generate: New restaurants, travel destinations

### Step 3: Design Your Embeddings

What information is relevant to your prediction?

```python
# Text: Just words
input_features = word_embedding

# Location: Rich context
input_features = concat([
    location_embedding,    # What place
    user_embedding,        # Who
    time_embedding,        # When (time of day)
    weekday_embedding,     # When (day)
    recency_embedding,     # How recent
    duration_embedding,    # How long they stayed
])
```

### Step 4: Choose Your Encoder

| Task Characteristics | Recommended Encoder |
|---------------------|-------------------|
| Sequential, local dependencies | LSTM/GRU |
| Long-range dependencies | Transformer |
| Very long sequences | Longformer/BigBird |
| Graph structure | Graph Neural Network |

**Example**: Location patterns have complex temporal dependencies → Transformer

### Step 5: Decide Decoder Strategy

| Output Type | Decoder Strategy |
|-------------|-----------------|
| Variable-length sequence | Auto-regressive LSTM |
| Fixed-length sequence | Parallel decoder |
| Single item | No decoder (direct from encoder) |

**Example**: Next location is single item → Use last encoder state directly

### Step 6: Implement Pointer Mechanism

```python
# Core pointer mechanism (domain-agnostic)
def pointer_attention(query, keys, values, mask=None):
    """
    query: [batch, d_model] - from context
    keys: [batch, seq_len, d_model] - from encoder
    values: [batch, seq_len] - item IDs to copy
    """
    # Compute attention scores
    scores = query @ keys.T / sqrt(d_model)
    
    # Apply mask
    if mask is not None:
        scores = scores.masked_fill(mask, -inf)
    
    # Softmax over positions
    attn_weights = softmax(scores, dim=-1)
    
    # Scatter to vocabulary
    ptr_dist = zeros(batch, vocab_size)
    ptr_dist.scatter_add_(1, values, attn_weights)
    
    return ptr_dist
```

### Step 7: Implement Generation Head

```python
# Simple linear projection to vocabulary
def generation_head(context):
    """
    context: [batch, d_model] - encoded representation
    returns: [batch, vocab_size] - probability distribution
    """
    logits = linear(context)  # [batch, vocab_size]
    gen_dist = softmax(logits, dim=-1)
    return gen_dist
```

### Step 8: Implement Gate

```python
# Gate decides pointer vs generation
def ptr_gen_gate(context):
    """
    context: [batch, d_model]
    returns: [batch, 1] - probability of using pointer
    """
    gate = MLP(context)  # [batch, 1]
    gate = sigmoid(gate)
    return gate
```

### Step 9: Combine Distributions

```python
# Final prediction
ptr_dist = pointer_attention(query, keys, input_ids)
gen_dist = generation_head(context)
gate = ptr_gen_gate(context)

final_dist = gate * ptr_dist + (1 - gate) * gen_dist
```

---

## PyTorch Implementation

Here's the complete adapted implementation with annotations:

```python
class PointerGeneratorTransformer(nn.Module):
    """
    Pointer-Generator for Next Location Prediction.
    
    Adapted from text summarization pointer-generator:
    - LSTM encoder → Transformer encoder
    - Word embeddings → Location + temporal embeddings
    - Sequence generation → Single prediction
    """
    
    def __init__(self, num_locations, num_users, d_model=128, ...):
        super().__init__()
        
        # ═══════════════════════════════════════════════════════════════
        # ADAPTATION 1: Rich Embeddings
        # Original: Single word embedding
        # Adapted: Location + User + Temporal features
        # ═══════════════════════════════════════════════════════════════
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
        self.time_emb = nn.Embedding(97, d_model // 4)     # Time of day
        self.weekday_emb = nn.Embedding(8, d_model // 4)   # Day of week
        self.recency_emb = nn.Embedding(9, d_model // 4)   # Days ago
        self.duration_emb = nn.Embedding(100, d_model // 4) # Stay duration
        
        # ═══════════════════════════════════════════════════════════════
        # ADAPTATION 2: Position-from-End Embedding
        # Original: Coverage mechanism to track attention
        # Adapted: Position bias favoring recent visits
        # ═══════════════════════════════════════════════════════════════
        self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
        
        # ═══════════════════════════════════════════════════════════════
        # ADAPTATION 3: Transformer Encoder
        # Original: Bidirectional LSTM
        # Adapted: Transformer for global attention
        # ═══════════════════════════════════════════════════════════════
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-LayerNorm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # ═══════════════════════════════════════════════════════════════
        # POINTER MECHANISM (Core - mostly preserved)
        # ═══════════════════════════════════════════════════════════════
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_key = nn.Linear(d_model, d_model)
        
        # Position bias instead of coverage
        self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
        
        # ═══════════════════════════════════════════════════════════════
        # GENERATION HEAD (Simplified)
        # Original: Two-layer projection
        # Adapted: Single linear layer
        # ═══════════════════════════════════════════════════════════════
        self.gen_head = nn.Linear(d_model, num_locations)
        
        # ═══════════════════════════════════════════════════════════════
        # POINTER-GENERATOR GATE (Core - preserved)
        # ═══════════════════════════════════════════════════════════════
        self.ptr_gen_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, x_dict):
        """
        Args:
            x: Location sequence [seq_len, batch]
            x_dict: Temporal and user features
        
        Returns:
            Log probabilities [batch, num_locations]
        """
        # Prepare features
        x = x.T  # [batch, seq_len]
        batch_size, seq_len = x.shape
        lengths = x_dict['len']
        
        # ═══════════════════════════════════════════════════════════════
        # BUILD RICH EMBEDDINGS
        # ═══════════════════════════════════════════════════════════════
        loc_emb = self.loc_emb(x)
        user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
        
        temporal = torch.cat([
            self.time_emb(x_dict['time'].T),
            self.weekday_emb(x_dict['weekday'].T),
            self.recency_emb(x_dict['diff'].T),
            self.duration_emb(x_dict['duration'].T)
        ], dim=-1)
        
        # Position from end (recency feature)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        pos_from_end = lengths.unsqueeze(1) - positions
        pos_emb = self.pos_from_end_emb(pos_from_end.clamp(0, self.max_seq_len - 1))
        
        # Combine and project
        combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
        hidden = self.input_norm(self.input_proj(combined))
        hidden = hidden + self.pos_encoding[:, :seq_len, :]
        
        # ═══════════════════════════════════════════════════════════════
        # ENCODE (Transformer instead of LSTM)
        # ═══════════════════════════════════════════════════════════════
        mask = positions >= lengths.unsqueeze(1)
        encoded = self.transformer(hidden, src_key_padding_mask=mask)
        
        # Extract context from LAST position (no decoder needed)
        batch_idx = torch.arange(batch_size, device=x.device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]
        
        # ═══════════════════════════════════════════════════════════════
        # POINTER ATTENTION
        # ═══════════════════════════════════════════════════════════════
        query = self.pointer_query(context).unsqueeze(1)  # [B, 1, D]
        keys = self.pointer_key(encoded)                   # [B, S, D]
        
        # Scaled dot-product attention
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1)
        ptr_scores = ptr_scores / math.sqrt(self.d_model)
        
        # Add position bias (favors recent positions)
        ptr_scores = ptr_scores + self.position_bias[pos_from_end]
        
        # Mask padding
        ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
        
        # Softmax to get attention weights
        ptr_probs = F.softmax(ptr_scores, dim=-1)
        
        # Scatter to location vocabulary
        # This is the COPY distribution
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=x.device)
        ptr_dist.scatter_add_(1, x, ptr_probs)
        
        # ═══════════════════════════════════════════════════════════════
        # GENERATION HEAD
        # ═══════════════════════════════════════════════════════════════
        gen_probs = F.softmax(self.gen_head(context), dim=-1)
        
        # ═══════════════════════════════════════════════════════════════
        # COMBINE WITH GATE (Core mechanism preserved!)
        # ═══════════════════════════════════════════════════════════════
        gate = self.ptr_gen_gate(context)
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs
        
        return torch.log(final_probs + 1e-10)
```

---

## Common Pitfalls

### 1. Forgetting Numerical Stability

```python
# ❌ Bad: Can produce NaN
log_probs = torch.log(final_probs)

# ✓ Good: Add small epsilon
log_probs = torch.log(final_probs + 1e-10)
```

### 2. Not Handling Variable Lengths

```python
# ❌ Bad: Ignores padding
context = encoded[:, -1, :]

# ✓ Good: Use actual last position
last_idx = (lengths - 1).clamp(min=0)
context = encoded[batch_idx, last_idx]
```

### 3. Scatter Add vs Scatter

```python
# ❌ Bad: Overwrites repeated items
ptr_dist.scatter_(1, x, ptr_probs)

# ✓ Good: Accumulates for repeated items
ptr_dist.scatter_add_(1, x, ptr_probs)
```

If a location appears multiple times in history, we want to SUM its attention weights.

### 4. Gate Range Issues

```python
# ❌ Bad: Gate might be 0 or 1 exactly
gate = torch.sigmoid(logits)
final = gate * ptr_dist + (1 - gate) * gen_dist
# If gate=1 exactly and ptr_dist has zeros, final has zeros → log(0) = -inf

# ✓ Good: Ensure mixture
final = gate * ptr_dist + (1 - gate) * gen_dist + 1e-10
```

### 5. Position Bias Direction

```python
# Think carefully: higher position = more recent or less recent?

# If pos_from_end = 0 means LAST (most recent):
# Positive bias[0] → favors recent
position_bias = [0.5, 0.3, 0.1, ...]  # Decaying with distance

# If pos_from_end = 0 means FIRST (least recent):
# Opposite interpretation needed
```

---

## Other Domain Applications

The pointer-generator pattern applies to many domains:

### Code Completion
```
Input: [import, numpy, as, np, x, =, np, .]
Output: array (generate) or numpy (copy "np" refers to)
```

### Product Recommendation
```
Input: User's purchase history [iPhone, Case, AirPods, ...]
Output: Apple Watch (generate) or AirPods (buy again)
```

### Dialogue Systems
```
Input: "My name is John and I work at Google"
Output: "Nice to meet you, John" (copy name)
```

### Music Playlist
```
Input: Recently played [Song_A, Song_B, Song_C, ...]
Output: Song_B (replay) or Song_D (new recommendation)
```

### Email Reply
```
Input: Received email mentioning "Project Alpha deadline"
Output: Reply copying "Project Alpha" instead of generic text
```

---

## Summary

Adapting the pointer-generator to new domains requires:

1. **Identify the abstraction**: Input sequence → Copy or Generate → Output
2. **Map domain concepts**: What are your "words" and "vocabulary"?
3. **Design embeddings**: What context matters for your prediction?
4. **Choose encoder**: LSTM vs Transformer vs other
5. **Simplify decoder**: Often single prediction is enough
6. **Preserve core mechanism**: The p_gen gate and distribution combination

The key insight is that the **pointer-generator pattern** (copy vs generate with learned gating) is domain-agnostic and applies whenever:
- Output can come from input (copy)
- Output can come from vocabulary (generate)
- Different situations call for different strategies

---

*Next: [README.md](README.md) - Documentation Index*
