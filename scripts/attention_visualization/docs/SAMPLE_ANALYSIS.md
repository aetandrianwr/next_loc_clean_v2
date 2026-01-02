# Sample-by-Sample Analysis

## In-Depth Analysis of Every Selected Sample

This document provides a comprehensive analysis of each of the 10 selected samples from both the DIY and Geolife datasets, explaining exactly what happened in each prediction.

---

## Table of Contents

1. [How to Read This Document](#how-to-read-this-document)
2. [DIY Dataset Samples](#diy-dataset-samples)
3. [Geolife Dataset Samples](#geolife-dataset-samples)
4. [Cross-Sample Patterns](#cross-sample-patterns)
5. [What These Samples Teach Us](#what-these-samples-teach-us)

---

## How to Read This Document

For each sample, we provide:

1. **Basic Information**: Sequence length, target, prediction, gate value
2. **The Story**: A narrative explanation of what the model "sees"
3. **Attention Breakdown**: Which positions got attention and why
4. **Score Analysis**: How raw scores and position bias combined
5. **Self-Attention Insights**: What the transformer layers captured
6. **Key Takeaway**: The main lesson from this sample

---

## DIY Dataset Samples

### DIY Sample 1

**File Reference**: `results/diy/sample_01_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 7254 |
| Sequence Length | 29 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9718 |
| Max Pointer Attention | 0.1529 |
| Confidence | 97.18% |

#### The Story

This is a user who visits location L17 **extremely frequently**. Out of 29 positions in their history, L17 appears in most of them. The model looks at this history and essentially says: "This person ALWAYS goes to L17. I'll copy from history with 97% confidence."

#### Attention Breakdown

With 29 positions and many L17 occurrences, the attention is **distributed** across multiple positions:
- No single position dominates (max is only 15.29%)
- Multiple L17 positions each get some attention
- The attention "spreads out" because many positions are equally relevant

**Why distributed?** When the same location appears many times, each occurrence provides similar information. The model hedges by attending to multiple positions rather than picking one.

#### Score Analysis

Looking at the Score Decomposition panel:
- **Raw scores** (blue bars): Relatively uniform across positions
- **Position bias** (orange bars): Slight boost to recent positions
- The combination doesn't create a single dominant peak

#### Self-Attention Insights

- **Layer 1**: Shows local patterns with diagonal dominance
- **Layer 2**: More globally integrated, helping understand the "L17 everywhere" pattern

#### Key Takeaway

**Lesson**: When a location dominates the history, attention distributes across its occurrences. The model doesn't need to focus on ONE position because they all point to the same answer.

---

### DIY Sample 2

**File Reference**: `results/diy/sample_02_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 4314 |
| Sequence Length | 12 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9716 |
| Max Pointer Attention | 0.2819 |
| Confidence | 97.16% |

#### The Story

A shorter sequence (12 positions) that's entirely composed of L17 visits. This user has an even more concentrated pattern than Sample 1.

#### Attention Breakdown

Examining the visualization:
- The tallest bar is at position 9 (about 28% attention)
- Several other positions also have notable attention (positions 7, 8, 10, 11)
- The attention is more peaked than Sample 1 because there are fewer positions

**Why more peaked?** With only 12 positions vs. 29, each position represents a larger "share" of the total attention. The maximum attention is naturally higher.

#### Score Analysis

From the Score Decomposition:
- **Raw scores**: All negative (around -0.7 to -1.2)
- **Position bias**: Near zero with slight positive values for recent positions
- Position 9's raw score is least negative → becomes the peak after softmax

**Important insight**: Even with all negative raw scores, softmax converts them to positive probabilities. It's the *relative* differences that matter!

#### Self-Attention Insights

Looking at the Multi-Head Attention comparison:
- **Head 1**: Focuses on position 11 (most recent)
- **Head 2**: Peaks at positions 1 and 4
- **Head 3**: Broad focus on positions 3-6
- **Head 4**: Emphasizes positions 0-1

Each head captures different temporal patterns!

#### Key Takeaway

**Lesson**: In shorter sequences, attention naturally becomes more concentrated. Different attention heads learn to focus on different parts of the sequence, providing diverse information.

---

### DIY Sample 3

**File Reference**: `results/diy/sample_03_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 4331 |
| Sequence Length | 13 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9678 |
| Max Pointer Attention | 0.2129 |
| Confidence | 96.78% |

#### The Story

Similar to Sample 2 but with one additional position. All 13 positions contain L17.

#### Attention Breakdown

- Peak attention at position 11 (about 21%)
- Secondary peak at position 12 (about 11%)
- Remaining attention distributed across earlier positions

The two-peak pattern suggests the model finds both position 11 and 12 particularly informative.

#### Score Analysis

- Raw scores range from about -1.2 to +0.2
- Positions 11 and 12 have the highest (least negative/most positive) raw scores
- Position bias adds a small boost to recent positions

#### Self-Attention Insights

Layer 1 self-attention shows:
- Strong diagonal (self-attention)
- Particularly bright column at position 4-6 (key positions)

Layer 2 self-attention:
- More uniform distribution
- Integrates information globally

#### Key Takeaway

**Lesson**: Even with homogeneous sequences (all same location), the model still has attention preferences based on position and temporal patterns encoded in the sequence.

---

### DIY Sample 4

**File Reference**: `results/diy/sample_04_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 4313 |
| Sequence Length | 11 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9677 |
| Max Pointer Attention | 0.2443 |
| Confidence | 96.77% |

#### The Story

An 11-position sequence, again dominated by L17.

#### Attention Breakdown

- Highest attention at position 9 (about 24%)
- Clear recency preference: positions 7-10 get most attention
- Earlier positions (0-4) get less attention

#### Score Analysis

The score decomposition reveals:
- Recent positions have higher raw scores
- Position bias reinforces this recency preference
- Combined effect creates the peak at position 9

#### Key Takeaway

**Lesson**: The model exhibits recency bias - more recent positions tend to get higher attention, even when all positions contain the same location.

---

### DIY Sample 5

**File Reference**: `results/diy/sample_05_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 4326 |
| Sequence Length | 10 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9658 |
| Max Pointer Attention | 0.2281 |
| Confidence | 96.58% |

#### The Story

A 10-position sequence with L17 throughout.

#### Attention Breakdown

- More evenly distributed than Sample 4
- Positions 6-9 each get 15-23% attention
- Flatter profile suggests less certainty about which position is "best"

#### Key Takeaway

**Lesson**: Shorter sequences with uniform content tend to have more even attention distribution, as the model has less information to differentiate positions.

---

### DIY Sample 6

**File Reference**: `results/diy/sample_06_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 7252 |
| Sequence Length | 29 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9683 |
| Max Pointer Attention | 0.1728 |
| Confidence | 96.83% |

#### The Story

Another long sequence (29 positions) like Sample 1. Very similar pattern.

#### Attention Breakdown

- Max attention is only 17.28% (distributed across many positions)
- Multiple peaks throughout the sequence
- No single dominant position

#### Key Takeaway

**Lesson**: Long sequences with repeated locations show highly distributed attention. The model relies on the aggregate effect rather than any single position.

---

### DIY Sample 7

**File Reference**: `results/diy/sample_07_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 4315 |
| Sequence Length | 11 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9669 |
| Max Pointer Attention | 0.2037 |
| Confidence | 96.69% |

#### The Story

Standard 11-position L17 sequence.

#### Key Takeaway

**Lesson**: Consistent pattern across similar sequences - the model has learned robust features for predicting frequent locations.

---

### DIY Sample 8

**File Reference**: `results/diy/sample_08_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 4319 |
| Sequence Length | 6 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9651 |
| Max Pointer Attention | 0.3233 |
| Confidence | 96.51% |

#### The Story

The **shortest** selected sample with only 6 positions. This is interesting because it shows how the model handles minimal history.

#### Attention Breakdown

- Highest max attention among DIY samples (32.33%)
- With only 6 positions, each one represents ~17% if uniform
- The peak at 32% is about 2× the uniform baseline

#### Score Analysis

- Raw scores are relatively similar across positions
- Position bias has more impact with fewer positions
- Recent positions get clear preference

#### Key Takeaway

**Lesson**: Shorter sequences result in more concentrated attention. With less data, the model must make stronger commitments to specific positions.

---

### DIY Sample 9

**File Reference**: `results/diy/sample_09_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 9076 |
| Sequence Length | 13 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9649 |
| Max Pointer Attention | 0.2530 |
| Confidence | 96.49% |

#### The Story

A 13-position sequence, similar to Sample 3.

#### Key Takeaway

**Lesson**: Consistent behavior across similar sequence lengths and compositions.

---

### DIY Sample 10

**File Reference**: `results/diy/sample_10_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 9075 |
| Sequence Length | 14 |
| Target Location | L17 |
| Prediction | L17 ✓ |
| Gate Value | 0.9644 |
| Max Pointer Attention | 0.2227 |
| Confidence | 96.44% |

#### The Story

A 14-position sequence with all L17.

#### Key Takeaway

**Lesson**: The model maintains high confidence (96%+) across varying sequence lengths when the pattern is clear.

---

## Geolife Dataset Samples

### Geolife Sample 1

**File Reference**: `results/geolife/sample_01_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 596 |
| Sequence Length | 41 |
| Target Location | L14 |
| Prediction | L14 ✓ |
| Gate Value | 0.9607 |
| Max Pointer Attention | 0.2614 |
| Confidence | 94.49% |

#### The Story

The longest selected Geolife sample with 41 positions. Unlike DIY samples, this sequence likely contains **multiple different locations**, making prediction more challenging.

#### Attention Breakdown

- Max attention of 26.14% is notable for a 41-position sequence
- This indicates the model found specific positions particularly relevant
- L14 appears multiple times, and the model focuses on key occurrences

#### Why Different from DIY?

GPS data captures continuous movement. A 41-position trajectory might include:
- Starting location
- Transit points
- Intermediate stops
- Destination
- Return journey

The model must identify which positions correspond to the target L14.

#### Key Takeaway

**Lesson**: Longer, more diverse sequences require the model to be more selective in attention, focusing on positions that actually contain relevant information.

---

### Geolife Sample 2

**File Reference**: `results/geolife/sample_02_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 1984 |
| Sequence Length | 14 |
| Target Location | L7 |
| Prediction | L7 ✓ |
| Gate Value | 0.9421 |
| Max Pointer Attention | 0.5419 |
| Confidence | 93.05% |

#### The Story

This is **the most interesting sample** in the entire analysis. With 54% attention on a single position, it shows extremely focused attention.

#### Detailed Sequence Analysis

From the visualization, the sequence contains:
```
Position:  0    1    2    3     4     5     6     7     8    9    10    11   12    13
Location:  L7   L7   L7   L939  L582  L7   L582  L582  L7  L582  L7   L582  L7   L582
```

#### Attention Breakdown

- Position 0 (first L7): **54.19%** attention
- Position 1 (second L7): ~24% attention  
- Position 2 (third L7): ~5% attention
- Other positions: minimal attention

**Why position 0?**

Looking at the Score Decomposition:
- Position 0 has a raw score of ~6.0 (extremely high!)
- Other positions have raw scores of 0-4
- This massive difference drives the focus

#### What Makes Position 0 Special?

The model has learned that the **first occurrence** of a frequently visited location is particularly informative. Position 0 represents:
- The starting point of a journey
- A "home base" location
- A reference point for the trajectory

#### Self-Attention Analysis

Layer 1 shows:
- Very bright column at position 0 (everyone attends to position 0)
- Strong diagonal pattern
- Clear hierarchical structure

Multi-Head Comparison:
- Head 1: Focuses strongly on position 10 (a later L7)
- Head 2: More uniform, with peaks at positions 3-4

**Interpretation**: Different heads learned different patterns:
- One head tracks the "anchor" position (first L7)
- Another tracks transitions between locations

#### Key Takeaway

**Lesson**: In mixed-location sequences, the model learns to focus heavily on the most informative occurrence of the target location, often the first or most contextually significant one.

---

### Geolife Sample 3

**File Reference**: `results/geolife/sample_03_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 574 |
| Sequence Length | 35 |
| Target Location | L14 |
| Prediction | L14 ✓ |
| Gate Value | 0.9361 |
| Max Pointer Attention | 0.2174 |
| Confidence | 92.32% |

#### The Story

A 35-position trajectory predicting L14. More distributed attention than Sample 2.

#### Key Takeaway

**Lesson**: Attention distribution depends on how the target location is distributed in the sequence and its relationship with surrounding locations.

---

### Geolife Sample 4

**File Reference**: `results/geolife/sample_04_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 2699 |
| Sequence Length | 12 |
| Target Location | L1151 |
| Prediction | L1151 ✓ |
| Gate Value | 0.9242 |
| Max Pointer Attention | **0.7570** |
| Confidence | 92.04% |

#### The Story

This sample has the **highest max attention** of all samples (75.70%!). The model is extremely confident about which position to attend to.

#### Why Such High Focus?

L1151 is a specific location that likely appears in a very distinctive context:
- Perhaps it only appears once or twice in the sequence
- The surrounding context makes it unmistakable
- The model has strong evidence for this prediction

#### Attention Breakdown

- One position dominates with 75.70% attention
- Remaining positions share the other 24.30%
- This is almost like "hard attention" (binary decision)

#### Key Takeaway

**Lesson**: When evidence is overwhelming, the model can achieve near-deterministic attention. This represents maximum confidence in the pointer mechanism.

---

### Geolife Sample 5

**File Reference**: `results/geolife/sample_05_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 3330 |
| Sequence Length | 8 |
| Target Location | L336 |
| Prediction | L336 ✓ |
| Gate Value | 0.9175 |
| Max Pointer Attention | 0.7287 |
| Confidence | 90.97% |

#### The Story

Short sequence (8 positions) with very high max attention (72.87%).

#### Key Takeaway

**Lesson**: Short sequences with distinctive patterns allow the model to focus attention strongly on the relevant position.

---

### Geolife Sample 6

**File Reference**: `results/geolife/sample_06_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 1987 |
| Sequence Length | 15 |
| Target Location | L7 |
| Prediction | L7 ✓ |
| Gate Value | 0.9580 |
| Max Pointer Attention | 0.3801 |
| Confidence | 90.42% |

#### The Story

Another L7 prediction (like Sample 2) but with different attention distribution.

#### Comparison with Sample 2

| Aspect | Sample 2 | Sample 6 |
|--------|----------|----------|
| Length | 14 | 15 |
| Max Attention | 54.19% | 38.01% |
| Gate | 0.9421 | 0.9580 |

Despite lower max attention, Sample 6 has higher gate value. This shows that confidence comes from the overall pattern, not just attention concentration.

#### Key Takeaway

**Lesson**: The same target location (L7) can produce different attention patterns depending on the specific sequence context.

---

### Geolife Sample 7

**File Reference**: `results/geolife/sample_07_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 2701 |
| Sequence Length | 12 |
| Target Location | L1151 |
| Prediction | L1151 ✓ |
| Gate Value | 0.9053 |
| Max Pointer Attention | 0.6118 |
| Confidence | 90.41% |

#### The Story

Another L1151 prediction (like Sample 4) with high but slightly lower max attention.

#### Key Takeaway

**Lesson**: Location L1151 consistently produces focused attention, suggesting it has distinctive contextual patterns.

---

### Geolife Sample 8

**File Reference**: `results/geolife/sample_08_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 2771 |
| Sequence Length | 9 |
| Target Location | L1151 |
| Prediction | L1151 ✓ |
| Gate Value | 0.9132 |
| Max Pointer Attention | 0.4264 |
| Confidence | 90.40% |

#### The Story

Short sequence (9 positions) predicting L1151.

#### Key Takeaway

**Lesson**: Shorter sequences of the same target location (L1151) show more distributed attention than longer ones.

---

### Geolife Sample 9

**File Reference**: `results/geolife/sample_09_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 570 |
| Sequence Length | 36 |
| Target Location | L14 |
| Prediction | L14 ✓ |
| Gate Value | 0.9219 |
| Max Pointer Attention | 0.2283 |
| Confidence | 90.18% |

#### The Story

Long trajectory (36 positions) predicting L14.

#### Key Takeaway

**Lesson**: Longer GPS trajectories show more distributed attention as there are more positions to consider.

---

### Geolife Sample 10

**File Reference**: `results/geolife/sample_10_attention.png`

#### Basic Information
| Property | Value |
|----------|-------|
| Original Index | 1292 |
| Sequence Length | 12 |
| Target Location | L553 |
| Prediction | L553 ✓ |
| Gate Value | 0.9335 |
| Max Pointer Attention | 0.5213 |
| Confidence | 90.05% |

#### The Story

Predicting a unique location (L553) with focused attention.

#### Key Takeaway

**Lesson**: Diverse target locations in Geolife each produce their own characteristic attention patterns.

---

## Cross-Sample Patterns

### DIY vs Geolife Comparison

| Pattern | DIY Samples | Geolife Samples |
|---------|-------------|-----------------|
| Target Diversity | All predict L17 | 5 different locations |
| Max Attention Range | 0.15 - 0.32 | 0.22 - 0.76 |
| Gate Range | 0.964 - 0.972 | 0.905 - 0.961 |
| Confidence Range | 96.4% - 97.2% | 90.0% - 94.5% |

### Relationship: Sequence Length vs Max Attention

| Dataset | Correlation | Interpretation |
|---------|-------------|----------------|
| DIY | Negative | Longer sequences → more distributed attention |
| Geolife | Weak negative | Similar trend but with more variance |

### Relationship: Max Attention vs Confidence

| Dataset | Correlation | Interpretation |
|---------|-------------|----------------|
| DIY | Weak | Confidence from aggregate, not single position |
| Geolife | Positive | Focused attention → higher confidence |

---

## What These Samples Teach Us

### 1. Homogeneous vs Heterogeneous Sequences

**DIY** sequences are mostly homogeneous (same location repeated):
- Attention distributes across multiple occurrences
- Max attention is moderate (15-32%)
- Prediction confidence comes from consistency

**Geolife** sequences are heterogeneous (mixed locations):
- Attention focuses on specific key positions
- Max attention can be very high (up to 76%)
- Prediction confidence comes from identification of relevant positions

### 2. The Role of Sequence Length

**Short sequences** (6-12 positions):
- Each position has more "weight" in the attention distribution
- Max attention tends to be higher
- Model must make decisions with limited information

**Long sequences** (29-41 positions):
- Attention naturally becomes more distributed
- Max attention is lower but still meaningful
- Model benefits from more context

### 3. Gate Value Interpretation

**Very high gate** (>0.95):
- Model is extremely confident in copying from history
- Target location appears clearly in the sequence
- Generation head is almost unused

**Moderately high gate** (0.90-0.95):
- Still predominantly pointer-based
- Some consideration of generation head
- Slightly less certain about copying

### 4. The Importance of Context

The same location can have different attention patterns depending on:
- Where it appears in the sequence
- What other locations surround it
- The temporal features (time, day, duration)
- The user's overall history

### 5. What Makes a "Good" Sample for Visualization

The selected samples all have:
- Correct predictions (we're studying success cases)
- High confidence (>90%)
- High gate values (>0.90)

This selection reveals how the model behaves when it's confident and correct. A complementary analysis of incorrect predictions would reveal failure modes.

---

*Sample Analysis - Version 1.0*
*Every sample tells a story about how the model makes decisions*
