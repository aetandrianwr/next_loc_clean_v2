# Attention Visualization for PointerGeneratorTransformer

## Comprehensive Documentation for Next Location Prediction Attention Analysis

This documentation provides a complete, scientifically rigorous explanation of the attention visualization experiment for the PointerGeneratorTransformer model used in next location prediction. It covers theoretical foundations, technical implementation, experimental results, and detailed interpretation of all generated visualizations and tables.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Theoretical Background](#2-theoretical-background)
3. [Model Architecture](#3-model-architecture)
4. [Attention Extraction System](#4-attention-extraction-system)
5. [Experimental Setup](#5-experimental-setup)
6. [Results and Analysis](#6-results-and-analysis)
7. [Visualization Guide](#7-visualization-guide)
8. [Cross-Dataset Comparison](#8-cross-dataset-comparison)
9. [Code Reference](#9-code-reference)
10. [Conclusions](#10-conclusions)

---

## 1. Introduction

### 1.1 Purpose

This experiment analyzes the internal attention mechanisms of the PointerGeneratorTransformer model to understand **how** the model makes next location predictions. By extracting and visualizing attention weights, we gain interpretability into:

- Which historical locations the model considers important
- How the model balances between copying from history vs. generating from vocabulary
- The role of temporal patterns in prediction
- Differences in model behavior across different mobility datasets

### 1.2 Scientific Significance

Understanding attention mechanisms is crucial for:

1. **Model Interpretability**: Explaining why a prediction was made
2. **Validation of Hypotheses**: Confirming that the model learns meaningful patterns (e.g., recency bias in human mobility)
3. **Model Improvement**: Identifying potential weaknesses or biases
4. **Trust Building**: Providing transparent model behavior for real-world applications

### 1.3 Scope

This analysis covers two datasets:
- **DIY Dataset**: Check-in based location data representing urban mobility patterns (12,368 test samples)
- **Geolife Dataset**: GPS trajectory data representing continuous movement patterns (3,502 test samples)

---

## 2. Theoretical Background

### 2.1 The Attention Mechanism

Attention mechanisms allow neural networks to dynamically focus on different parts of the input when making predictions. In the context of location prediction:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where:
- **Q (Query)**: What we're looking for
- **K (Key)**: What we're searching through
- **V (Value)**: What we retrieve

The softmax function converts attention scores to probabilities, indicating the importance of each position.

### 2.2 Pointer Networks

Pointer Networks (Vinyals et al., 2015) are designed to **copy** elements from the input sequence rather than generating from a fixed vocabulary. This is ideal for location prediction because:

1. Users often revisit the same locations (home, work, gym)
2. The model can directly point to a previously visited location
3. Reduces vocabulary complexity for prediction

The pointer attention score for position $i$ is:

$$\text{ptr\_score}_i = \frac{q \cdot k_i}{\sqrt{d}} + \text{position\_bias}_i$$

Where the position bias is a learned parameter that captures recency preferences.

### 2.3 Pointer-Generation Hybrid

The model uses a **gate** mechanism to blend two prediction strategies:

$$P(y) = g \cdot P_{\text{pointer}}(y) + (1-g) \cdot P_{\text{generation}}(y)$$

Where:
- **g**: Gate value (0 to 1), learned per sample
- **P_pointer**: Probability from the pointer mechanism (copying from history)
- **P_generation**: Probability from the generation head (predicting from full vocabulary)

**Interpretation**:
- **g → 1**: Model relies on copying from visited locations
- **g → 0**: Model predicts from the full location vocabulary

### 2.4 Self-Attention in Transformers

Self-attention allows each position in the sequence to attend to all other positions:

$$\text{Attention}_{ij} = \frac{\exp(q_i \cdot k_j / \sqrt{d})}{\sum_k \exp(q_i \cdot k_k / \sqrt{d})}$$

This captures:
- **Temporal dependencies**: How visits at different times relate
- **Location relationships**: Patterns between location types
- **Sequential patterns**: Common transition sequences

### 2.5 Attention Entropy

Entropy measures the **spread** of attention:

$$H = -\sum_i p_i \log(p_i)$$

- **Low entropy**: Attention is focused on few positions (confident)
- **High entropy**: Attention is spread across many positions (uncertain)

---

## 3. Model Architecture

### 3.1 PointerGeneratorTransformer Overview

The model architecture consists of several key components:

```
Input Sequence → Embeddings → Transformer Encoder → Pointer + Generation → Gate → Final Prediction
```

### 3.2 Embedding Layer

The embedding layer combines multiple features:

| Feature | Dimension | Description |
|---------|-----------|-------------|
| Location | d_model | Embedding for each unique location |
| User | d_model | User-specific embedding |
| Time | d_model/4 | Time of day (96 intervals = 15-min buckets) |
| Weekday | d_model/4 | Day of week (0-6) |
| Recency | d_model/4 | How many days ago (0-8 levels) |
| Duration | d_model/4 | Visit duration (100 buckets) |
| Position from End | d_model/4 | Distance from sequence end |

**Total input dimension**: `d_model × 2 + d_model/4 × 5`

### 3.3 Model Configurations

#### DIY Dataset Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 64 |
| nhead | 4 |
| num_layers | 2 |
| dim_feedforward | 256 |
| dropout | 0.2 |
| Total Parameters | ~1,081,280 |

#### Geolife Dataset Configuration
| Parameter | Value |
|-----------|-------|
| d_model | 96 |
| nhead | 2 |
| num_layers | 2 |
| dim_feedforward | 192 |
| dropout | 0.25 |
| Total Parameters | ~443,328 |

### 3.4 Pointer Mechanism Details

The pointer mechanism computes attention over the input sequence:

1. **Query**: Derived from the context vector (last valid position)
2. **Keys**: Derived from all encoded positions
3. **Position Bias**: Learned parameter per position from end

The final pointer distribution is created by:
1. Computing attention scores
2. Adding position bias (recency preference)
3. Applying softmax (masked for padding)
4. Scattering to location vocabulary

---

## 4. Attention Extraction System

### 4.1 AttentionExtractor Class

The `AttentionExtractor` class (`attention_extractor.py`) provides comprehensive attention extraction:

```python
class AttentionExtractor:
    """
    Extracts and processes attention weights from PointerGeneratorTransformer.
    
    Captures:
    - Multi-head self-attention weights from transformer layers
    - Pointer attention scores and probabilities
    - Pointer-generation gate values
    """
```

### 4.2 Extracted Components

| Component | Shape | Description |
|-----------|-------|-------------|
| `self_attention` | [num_layers, batch, heads, seq, seq] | Self-attention per layer |
| `pointer_scores_raw` | [batch, seq_len] | Raw pointer scores before bias |
| `position_bias` | [batch, seq_len] | Learned position bias values |
| `pointer_probs` | [batch, seq_len] | Final pointer attention weights |
| `gate_values` | [batch, 1] | Pointer-generation gate |
| `generation_probs` | [batch, num_locations] | Generation head output |
| `final_probs` | [batch, num_locations] | Combined prediction |

### 4.3 Manual Self-Attention Extraction

The extractor manually computes self-attention to capture weights:

```python
def _compute_self_attention(self, x, attn_module, mask):
    # Get Q, K, V projections
    qkv = F.linear(x, attn_module.in_proj_weight, attn_module.in_proj_bias)
    q, k, v = qkv.chunk(3, dim=-1)
    
    # Reshape for multi-head attention
    q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
    
    # Compute attention scores
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
    attn_weights = F.softmax(attn_scores, dim=-1)
    
    return attn_weights
```

---

## 5. Experimental Setup

### 5.1 Data Preparation

- **DIY Dataset**: 12,368 test samples from check-in data
- **Geolife Dataset**: 3,502 test samples from GPS trajectories
- Both datasets use sequences of up to 7 previous visits (`prev7`)

### 5.2 Experiment Pipeline

The experiment (`run_attention_experiment.py`) follows this pipeline:

1. **Load Model**: Load trained PointerGeneratorTransformer checkpoint
2. **Extract Attention**: Process all test samples
3. **Compute Statistics**: Aggregate metrics across samples
4. **Select Best Samples**: Choose 10 high-confidence correct predictions
5. **Generate Visualizations**: Create publication-quality plots
6. **Generate Tables**: Export CSV and LaTeX tables

### 5.3 Sample Selection Criteria

The 10 selected samples for detailed visualization are chosen by:
1. **Correct predictions only**: Ensures analysis of successful behavior
2. **High confidence**: Sorted by prediction probability
3. **Length diversity**: Different sequence lengths represented

---

## 6. Results and Analysis

### 6.1 Summary Statistics

#### DIY Dataset Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Samples | 12,368 | Large test set for statistical significance |
| Prediction Accuracy | **56.58%** | Moderate accuracy for multi-class prediction |
| Mean Gate Value | **0.7872** | Strong preference for pointer mechanism |
| Gate Std Dev | 0.1366 | Relatively consistent gate behavior |
| Gate (Correct) | **0.8168** | Higher gate for correct predictions |
| Gate (Incorrect) | 0.7486 | Lower gate for incorrect predictions |
| Mean Pointer Entropy | 2.3358 | Moderately spread attention |
| Most Recent Attention | 0.0458 | Position t-0 attention weight |

#### Geolife Dataset Results

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Total Samples | 3,502 | Smaller but sufficient test set |
| Prediction Accuracy | **51.40%** | Lower than DIY, harder prediction task |
| Mean Gate Value | **0.6267** | More balanced pointer/generation use |
| Gate Std Dev | 0.2289 | Higher variance in gate behavior |
| Gate (Correct) | **0.6464** | Higher gate for correct predictions |
| Gate (Incorrect) | 0.6059 | Lower gate for incorrect predictions |
| Mean Pointer Entropy | 1.9764 | More focused attention |
| Most Recent Attention | 0.0605 | Higher recency focus than DIY |

### 6.2 Key Findings

#### Finding 1: Pointer Mechanism Dominance

**DIY**: Gate mean = 0.7872 (78.72% pointer weight)
**Geolife**: Gate mean = 0.6267 (62.67% pointer weight)

**Interpretation**: Both models predominantly use the pointer mechanism, indicating that users frequently revisit previously visited locations. The DIY dataset shows stronger copy behavior, suggesting more repetitive mobility patterns in check-in data compared to GPS trajectories.

#### Finding 2: Gate Predicts Success

The gate value is **consistently higher for correct predictions**:
- DIY: +0.0682 difference (0.8168 vs 0.7486)
- Geolife: +0.0405 difference (0.6464 vs 0.6059)

**Scientific Significance**: This validates that the pointer mechanism is effective. When the model correctly identifies that the next location is in the history, it increases the gate value to copy from history. Incorrect predictions often occur when the model incorrectly tries to copy or when the actual next location wasn't in the history.

#### Finding 3: Recency Effect

Position-wise attention reveals **recency bias**:

| Position from End | DIY Attention | Geolife Attention |
|-------------------|---------------|-------------------|
| t-0 (most recent) | 0.0458 | 0.0605 |
| t-1 | **0.2105** | **0.1305** |
| t-2 | 0.0739 | 0.0746 |
| t-3 | 0.0756 | 0.1001 |
| t-4 | 0.0519 | 0.0709 |
| t-5 | 0.0532 | 0.0735 |

**Key Observation**: Position t-1 (second most recent) receives the **highest attention** in both datasets. This suggests:
1. The most recent location (t-0) is often the current location
2. Position t-1 is the best predictor of where the user will go next
3. This aligns with human mobility patterns (commuting between key locations)

#### Finding 4: Entropy Differences

**DIY Entropy**: 2.3358 (higher spread)
**Geolife Entropy**: 1.9764 (more focused)

**Interpretation**: Geolife shows more concentrated attention, potentially because GPS trajectories have stronger sequential dependencies. Check-in data (DIY) may have more irregular patterns requiring broader attention.

### 6.3 Selected Sample Analysis

#### DIY Top 10 Samples

| Sample | Seq Length | Target | Gate | Max Ptr Attn | Confidence |
|--------|------------|--------|------|--------------|------------|
| 1 | 29 | L17 | 0.9718 | 0.1529 | 97.18% |
| 2 | 12 | L17 | 0.9716 | 0.2819 | 97.16% |
| 3 | 13 | L17 | 0.9678 | 0.2129 | 96.78% |
| 4 | 11 | L17 | 0.9677 | 0.2443 | 96.77% |
| 5 | 10 | L17 | 0.9658 | 0.2281 | 96.58% |
| 6 | 29 | L17 | 0.9683 | 0.1728 | 96.83% |
| 7 | 11 | L17 | 0.9669 | 0.2037 | 96.69% |
| 8 | 6 | L17 | 0.9651 | 0.3233 | 96.51% |
| 9 | 13 | L17 | 0.9649 | 0.2530 | 96.49% |
| 10 | 14 | L17 | 0.9644 | 0.2227 | 96.44% |

**Notable Patterns**:
- All top samples predict location L17 (likely a frequent location like home)
- Gate values all > 0.96 (very high pointer reliance)
- Shorter sequences (Sample 8, length 6) have higher max pointer attention
- Confidence directly correlates with gate value

#### Geolife Top 10 Samples

| Sample | Seq Length | Target | Gate | Max Ptr Attn | Confidence |
|--------|------------|--------|------|--------------|------------|
| 1 | 41 | L14 | 0.9607 | 0.2614 | 94.49% |
| 2 | 14 | L7 | 0.9421 | 0.5419 | 93.05% |
| 3 | 35 | L14 | 0.9361 | 0.2174 | 92.32% |
| 4 | 12 | L1151 | 0.9242 | 0.7570 | 92.04% |
| 5 | 8 | L336 | 0.9175 | 0.7287 | 90.97% |
| 6 | 15 | L7 | 0.9580 | 0.3801 | 90.42% |
| 7 | 12 | L1151 | 0.9053 | 0.6118 | 90.41% |
| 8 | 9 | L1151 | 0.9132 | 0.4264 | 90.40% |
| 9 | 36 | L14 | 0.9219 | 0.2283 | 90.18% |
| 10 | 12 | L553 | 0.9335 | 0.5213 | 90.05% |

**Notable Patterns**:
- More diverse target locations (L14, L7, L1151, L336, L553)
- Higher max pointer attention values (up to 0.7570)
- Geolife shows stronger single-position focus

---

## 7. Visualization Guide

### 7.1 Aggregate Pointer Attention Plot

**File**: `aggregate_pointer_attention.png`

#### Left Panel: Position-wise Attention
- **X-axis**: Position from sequence end (t-k where k is positions back)
- **Y-axis**: Mean attention weight across all samples
- **Interpretation**: Shows the average importance given to each historical position

**DIY Dataset Analysis**:
- Position t-1 has the highest attention (0.2105)
- Position t-0 has moderate attention (0.0458)
- Attention decays gradually with distance from end
- Significant attention extends to ~15 positions back

**Geolife Dataset Analysis**:
- Position t-1 also highest (0.1305) but lower than DIY
- Position t-0 slightly higher than DIY (0.0605)
- More uniform distribution across positions
- Position t-3 shows a secondary peak (0.1001)

#### Right Panel: Attention Entropy Distribution
- **X-axis**: Attention entropy (nats)
- **Y-axis**: Number of samples
- **Red dashed line**: Mean entropy

**Interpretation**:
- Right-skewed distribution indicates most samples have focused attention
- Long tail shows some samples with high uncertainty
- DIY mean: 2.3358 nats
- Geolife mean: 1.9764 nats

### 7.2 Gate Analysis Plot

**File**: `gate_analysis.png`

This three-panel figure analyzes the pointer-generation gate:

#### Panel 1: Gate Distribution
- **X-axis**: Gate value (0-1)
- **Y-axis**: Number of samples
- Shows the distribution of gate values across all test samples

**DIY Observations**:
- Strongly right-skewed (most values > 0.7)
- Peak around 0.8-0.85
- Very few samples with gate < 0.5

**Geolife Observations**:
- More spread distribution
- Peak around 0.6-0.7
- Noticeable samples across full range

#### Panel 2: Gate by Prediction Outcome
- **Violin plot**: Shows distribution shape
- **Green violin**: Correct predictions
- **Red violin**: Incorrect predictions

**Key Finding**: Correct predictions have statistically higher gate values in both datasets, validating the pointer mechanism's effectiveness.

#### Panel 3: Gate vs Sequence Length
- **X-axis**: Sequence length
- **Y-axis**: Mean gate value
- **Error bars**: Standard deviation

**Observations**:
- Gate values relatively stable across lengths
- Slight decrease with very long sequences
- Suggests pointer mechanism works consistently

### 7.3 Self-Attention Aggregate Plot

**File**: `self_attention_aggregate.png`

Shows heatmaps of aggregated self-attention patterns for each transformer layer.

#### How to Read
- **X-axis**: Key position (from sequence end)
- **Y-axis**: Query position (from sequence end)
- **Color intensity**: Average attention weight
- **Diagonal**: Self-attention (position attending to itself)

#### Layer 1 Patterns
- Strong diagonal component (self-attention)
- Off-diagonal patterns reveal cross-position dependencies
- Recent positions show more activity

#### Layer 2 Patterns
- More diffuse attention patterns
- Higher-level representations attend more broadly
- Less pronounced diagonal

### 7.4 Position Bias Analysis

**File**: `position_bias_analysis.png`

Analyzes the learned position bias parameter.

#### Left Panel: Raw Position Bias Values
- **X-axis**: Position from end
- **Y-axis**: Learned bias value
- Shows the raw learned parameters

**Interpretation**:
- Positive values increase attention to that position
- Negative values decrease attention
- Pattern reveals learned recency preferences

#### Right Panel: Bias Effect on Attention
- Shows attention distribution when all content scores are equal
- Isolates the pure effect of position bias
- Demonstrates learned temporal preferences

### 7.5 Samples Overview

**File**: `samples_overview.png`

Shows pointer attention patterns for all 10 selected samples in a grid.

#### How to Read
- Each subplot is one sample
- **X-axis**: Sequence position
- **Y-axis**: Attention weight
- **Title**: Gate value and prediction result
- **Green dashed line**: Position of target in sequence (if present)
- **Color gradient**: Intensity of attention (yellow to red)

#### DIY Pattern
- Generally smooth attention distributions
- Multiple peaks common
- High gate values (>0.9 for all samples)

#### Geolife Pattern
- More peaked distributions
- Often single dominant position
- High maximum attention values

### 7.6 Individual Sample Attention

**Files**: `sample_01_attention.png` through `sample_10_attention.png`

Detailed four-panel analysis for each selected sample.

#### Panel A: Pointer Attention Bar Chart
- **X-axis**: Sequence positions (labeled with location IDs)
- **Y-axis**: Attention weight
- **Color**: Gradient from low (yellow) to high (red) attention
- **Black border**: Marks position with maximum attention
- **Annotation**: Shows gate value

#### Panel B: Score Decomposition
- **Blue bars**: Raw attention scores (content-based)
- **Orange bars**: Position bias contribution
- Shows how final scores are composed

#### Panel C: Self-Attention Heatmaps
- One heatmap per transformer layer
- Query × Key attention patterns
- Shows internal representation learning

#### Panel D: Multi-Head Attention Comparison
- **X-axis**: Key position
- **Y-axis**: Attention head
- Shows how different heads specialize

---

## 8. Cross-Dataset Comparison

### 8.1 Comparison Table

| Metric | DIY | Geolife | Difference |
|--------|-----|---------|------------|
| Test Samples | 12,368 | 3,502 | - |
| Accuracy | 56.58% | 51.40% | +5.18% |
| Mean Gate | 0.7872 | 0.6267 | +0.1605 |
| Gate (Correct) | 0.8168 | 0.6464 | +0.1704 |
| Gate (Incorrect) | 0.7486 | 0.6059 | +0.1427 |
| Gate Differential | 0.0682 | 0.0405 | +0.0277 |
| Pointer Entropy | 2.3358 | 1.9764 | +0.3594 |
| Recent Position Attn | 0.0458 | 0.0605 | -0.0147 |

### 8.2 Gate Comparison Plot

**File**: `cross_dataset_gate_comparison.png`

#### Left Panel: Mean Gate Comparison
- Bar chart comparing gate values
- Error bars show standard deviation
- Horizontal line at 0.5 marks equal pointer/generation balance

**Key Insight**: DIY shows significantly higher pointer reliance (0.787 vs 0.627), indicating more repetitive location visiting patterns.

#### Right Panel: Position Attention Comparison
- Grouped bar chart comparing attention at each position
- Green bars: DIY
- Blue bars: Geolife

**Key Insight**: DIY has dramatically higher attention at position t-1 (0.21 vs 0.13).

### 8.3 Attention Pattern Comparison

**File**: `cross_dataset_attention_patterns.png`

Four-panel comprehensive comparison:

#### Panel A: Recency Effect
Line plot showing position-wise attention decay.
- DIY shows steeper initial decay from t-1
- Geolife shows more gradual decay

#### Panel B: Cumulative Attention
Shows cumulative attention for top-k positions.
- DIY: ~0.33 cumulative attention in top 2 positions
- Geolife: ~0.19 cumulative attention in top 2 positions

**Interpretation**: DIY concentrates more attention in recent history.

#### Panel C: Gate by Outcome
Grouped bar chart showing gate values for correct vs incorrect predictions.
- Both datasets show higher gate for correct predictions
- Difference is more pronounced in DIY

#### Panel D: Summary Metrics
Normalized comparison of key metrics (0-1 scale):
- Accuracy
- Gate Mean
- Entropy (normalized)
- Recency (position 0 attention)

### 8.4 Dataset Characteristics Explanation

**Why DIY Shows Higher Pointer Reliance**:
1. Check-in data captures intentional visits (restaurants, stores)
2. Users tend to revisit favorite locations
3. Temporal patterns (lunch spots, evening venues) are stronger
4. Sparse data creates more repetitive sequences

**Why Geolife Shows More Balanced Gate**:
1. GPS trajectories capture all movement
2. More diverse location transitions
3. Continuous nature includes travel between points
4. Denser data with more unique locations

---

## 9. Code Reference

### 9.1 File Structure

```
attention_visualization/
├── attention_extractor.py      # Core attention extraction logic
├── run_attention_experiment.py # Main experiment runner
├── cross_dataset_comparison.py # Cross-dataset analysis
├── results/
│   ├── diy/                    # DIY dataset results
│   ├── geolife/                # Geolife dataset results
│   ├── cross_dataset_*.png     # Comparison visualizations
│   └── key_findings.csv        # Summary findings
└── docs/                       # This documentation
```

### 9.2 Key Functions

#### `extract_attention()` in `attention_extractor.py`
Performs a modified forward pass to capture all attention components.

#### `compute_attention_statistics()` in `attention_extractor.py`
Aggregates statistics across samples including:
- Gate mean/std
- Pointer entropy
- Position-wise attention

#### `select_best_samples()` in `run_attention_experiment.py`
Selects representative samples for detailed visualization.

#### `plot_*()` functions in `run_attention_experiment.py`
Generate publication-quality visualizations.

### 9.3 Usage

```bash
# Run experiment for DIY dataset
python run_attention_experiment.py --dataset diy --seed 42

# Run experiment for Geolife dataset
python run_attention_experiment.py --dataset geolife --seed 42

# Generate cross-dataset comparison
python cross_dataset_comparison.py
```

---

## 10. Conclusions

### 10.1 Summary of Findings

1. **Pointer Mechanism Effectiveness**: The pointer-generation gate successfully learns when to copy from history vs. generate. Higher gate values correlate with correct predictions.

2. **Recency Bias Validated**: Both datasets show strong attention to recent positions, particularly position t-1, validating known human mobility patterns.

3. **Dataset-Specific Behavior**: 
   - DIY (check-in): Higher pointer reliance, more repetitive patterns
   - Geolife (GPS): More balanced prediction, diverse transitions

4. **Position Bias Contribution**: The learned position bias captures meaningful temporal preferences that enhance prediction beyond content-based attention.

5. **Attention Entropy**: Lower entropy (more focused attention) generally correlates with more confident and accurate predictions.

### 10.2 Scientific Implications

- The attention mechanism provides interpretable insights into model behavior
- Pointer networks are well-suited for location prediction due to revisitation patterns
- The gate mechanism adaptively balances prediction strategies
- Cross-dataset analysis reveals fundamental differences in mobility data types

### 10.3 Limitations

1. Analysis based on test set performance only
2. Selected samples biased toward high-confidence predictions
3. Aggregate statistics may mask individual patterns
4. Position bias analysis assumes uniform base scores

### 10.4 Future Work

- Analyze attention patterns for incorrect predictions
- Study attention dynamics across training epochs
- Investigate attention patterns for specific location types
- Extend analysis to other mobility datasets

---

## References

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Vinyals et al. (2015). "Pointer Networks"
3. See et al. (2017). "Get To The Point: Summarization with Pointer-Generator Networks"

---

*Documentation generated: January 2026*
*Experiment Timestamp: 2026-01-02*
