# Glossary of Terms

## Complete Reference of All Technical Terms Used in Attention Visualization

This glossary provides definitions for every technical term used throughout the documentation, organized alphabetically with cross-references.

---

## A

### Aggregate Statistics
Summary measures computed across all samples in a dataset, such as mean, standard deviation, or percentiles. Used to understand overall model behavior rather than individual predictions.

### Attention
A mechanism that allows neural networks to dynamically focus on different parts of the input when making predictions. See also: [Self-Attention](#self-attention), [Pointer Attention](#pointer-attention).

### Attention Entropy
A measure of how "spread out" attention is across positions. Calculated as $H = -\sum_i p_i \log p_i$ where $p_i$ is the attention weight at position $i$. Low entropy means focused attention; high entropy means distributed attention.

### Attention Heatmap
A 2D visualization where color intensity represents attention weights. Rows typically represent query positions, columns represent key positions. Darker colors indicate higher attention.

### Attention Weight
A non-negative number between 0 and 1 assigned to each position in a sequence, indicating how much the model "focuses" on that position. All weights sum to 1.

---

## B

### Batch
A group of samples processed together through the neural network. Using batches improves computational efficiency and training stability. Batch size in this experiment is 64.

### Batch Size
The number of samples processed in one forward/backward pass. Denoted as $B$ in tensor shapes like $[B, T, d]$.

---

## C

### Check-in Data
Location data collected when users voluntarily "check in" at venues (like Foursquare). Captures intentional visits to meaningful locations. The DIY dataset uses this format.

### Confidence
The probability assigned to the predicted class. In this context, it equals the probability of the target location in the final distribution. Higher confidence indicates the model is more certain.

### Context Vector
A fixed-size representation of the entire input sequence, extracted from the last valid position of the transformer encoder output. Used to generate queries for attention mechanisms.

### Cross-Entropy Loss
The training objective function: $\mathcal{L} = -\log P(y_{true})$. Measures how well the predicted distribution matches the true label.

---

## D

### d_model
The dimensionality of the model's internal representations. All embeddings and hidden states have this dimension. DIY uses d_model=64; Geolife uses d_model=96.

### Decoder
The part of a sequence-to-sequence model that generates output. In PointerGeneratorTransformer, the pointer mechanism and generation head together form the decoder.

### Diagonal (in attention)
In self-attention heatmaps, the diagonal represents each position attending to itself. Strong diagonal patterns indicate self-attention is important.

### Duration
How long a user stayed at a location. One of the temporal features encoded in the model (100 buckets representing different stay lengths).

---

## E

### Embedding
A learned vector representation of a discrete input (like a location ID or user ID). Embeddings map categorical data to continuous vectors.

### Encoder
The part of the model that processes the input sequence. In PointerGeneratorTransformer, this is the Transformer encoder that produces contextualized representations.

### Entropy
See [Attention Entropy](#attention-entropy). In information theory, entropy measures uncertainty or randomness in a probability distribution.

### Effective Positions
The approximate number of positions that attention is spread across, calculated as $e^{H}$ where $H$ is entropy. For example, entropy of 2.0 means ~7.4 effective positions.

---

## F

### Feed-Forward Network (FFN)
A component of transformer layers that applies position-wise transformations. Consists of two linear layers with GELU activation in between.

### Final Distribution
The probability distribution over locations after combining pointer and generation distributions using the gate: $P_{final} = g \cdot P_{ptr} + (1-g) \cdot P_{gen}$.

---

## G

### Gate / Gate Value
A learned scalar between 0 and 1 that determines how to blend pointer and generation predictions. High gate (→1) means trust pointer; low gate (→0) means trust generation.

### Gate Differential
The difference in average gate values between correct and incorrect predictions. Positive differential indicates higher gate correlates with correct predictions.

### GELU (Gaussian Error Linear Unit)
An activation function used in the model: $\text{GELU}(x) = x \cdot \Phi(x)$ where $\Phi$ is the standard Gaussian CDF. Smoother than ReLU.

### Generation Head
A linear layer that produces probability distribution over the full location vocabulary. Used when the target might not be in history.

### Geolife Dataset
A GPS trajectory dataset collected by Microsoft Research in Beijing. Contains continuous location tracking data.

### GPS Data
Location data from Global Positioning System tracking. Captures all movement continuously, unlike check-in data.

---

## H

### Head (Attention)
One of multiple parallel attention mechanisms in multi-head attention. Each head has its own learned projections, allowing it to focus on different patterns. See [Multi-Head Attention](#multi-head-attention).

### Hidden State
The internal representation at each position after processing through neural network layers.

### History
The sequence of previously visited locations used to predict the next location. Also called the input sequence.

### Hook (PyTorch)
A mechanism to access intermediate values during forward pass. Used in attention extraction to capture attention weights.

---

## I

### Input Sequence
The sequence of location IDs representing a user's visit history. Shape: $[T, B]$ or $[B, T]$ depending on format.

---

## K

### Key (K)
In attention mechanisms, keys are vectors that queries are compared against. High query-key similarity means high attention.

### Key Position
In attention heatmaps, the position being attended TO (usually on x-axis).

---

## L

### Layer
A single processing unit in the neural network. PointerGeneratorTransformer uses 2 transformer encoder layers.

### LayerNorm (Layer Normalization)
A normalization technique that normalizes across features for each sample: $\text{LN}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta$.

### Location ID
A unique identifier for each distinct location in the vocabulary. Ranges from 0 to num_locations-1.

---

## M

### Mask / Padding Mask
A tensor indicating which positions are valid (True) vs padded (False). Masked positions get $-\infty$ attention score, resulting in 0 attention weight after softmax.

### Max Attention
The highest attention weight assigned to any single position. Indicates how focused the attention is.

### Mean Attention
The average attention weight at a particular position across all samples.

### Multi-Head Attention
Attention with multiple parallel "heads," each with different learned projections. Allows capturing different types of relationships. Output is concatenation of all heads.

---

## N

### nhead
Number of attention heads in multi-head attention. DIY uses 4 heads; Geolife uses 2 heads.

### num_layers
Number of transformer encoder layers stacked. Both DIY and Geolife use 2 layers.

### num_locations
Total number of unique locations in the vocabulary (the "V" in vocabulary size).

---

## P

### Padding
Adding placeholder values to make all sequences in a batch the same length. Padded positions are masked during attention.

### Perplexity
A measure related to entropy: $\text{Perplexity} = e^H$. Represents the "effective vocabulary size" of a distribution.

### Pointer Attention
The attention mechanism that "points" to positions in the input sequence to copy locations. Core of the pointer network.

### Pointer Distribution
Probability distribution over locations derived from pointer attention by scattering position probabilities to their corresponding locations.

### Pointer Network
A neural network architecture that can output pointers to input positions rather than generating from a fixed vocabulary.

### PointerGeneratorTransformer
The specific model architecture used in this experiment, combining transformer encoder with pointer mechanism and generation head.

### Position Bias
A learned parameter vector that adds position-dependent bias to pointer attention scores. Captures recency preferences.

### Position Encoding
Fixed sinusoidal vectors added to embeddings to provide position information. Uses sin/cos at different frequencies.

### Position from End
Distance from the end of the sequence. Position 0 = most recent, Position 1 = second most recent, etc.

### Pre-Norm
Architecture where LayerNorm is applied BEFORE the sublayer (attention or FFN), rather than after. Improves training stability.

---

## Q

### Query (Q)
In attention mechanisms, the query is what we're looking for. It's compared against keys to compute attention weights.

### Query Position
In attention heatmaps, the position doing the attending (usually on y-axis).

---

## R

### Raw Score
The attention score before softmax normalization. Can be positive or negative.

### Recency
How recently something occurred. In the model, encoded as "days ago" (0-8 levels).

### Recency Effect
The tendency to give more weight to recent events. Observed in position t-1 receiving highest attention.

### Residual Connection
Adding the input of a layer to its output: $y = x + f(x)$. Helps gradient flow in deep networks.

---

## S

### Sample
A single data point consisting of input sequence, temporal features, and target location.

### Scaled Dot-Product Attention
Attention where scores are computed as dot products scaled by $\sqrt{d_k}$: $\text{Attention} = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.

### Scatter Operation
Operation that distributes values from positions to locations based on the sequence: summing attention at all positions with the same location.

### Self-Attention
Attention where queries, keys, and values all come from the same sequence. Each position can attend to all other positions.

### Sequence Length
The number of positions in an input sequence. Varies per sample, denoted as $T$ or $L$.

### Sigmoid
Activation function that squashes values to (0, 1): $\sigma(x) = \frac{1}{1+e^{-x}}$. Used for gate output.

### Softmax
Function that converts raw scores to probability distribution: $\text{softmax}(z)_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$.

---

## T

### Target
The true next location that the model should predict. Also called label or ground truth.

### Temporal Features
Time-related information: time of day, day of week, recency, duration. Encoded as embeddings.

### Test Set
Data held out from training, used only for evaluation. DIY has 12,368 test samples; Geolife has 3,502.

### Transformer
Neural network architecture based on self-attention, introduced in "Attention Is All You Need" (2017).

### Transformer Encoder
The encoding part of a transformer that processes input sequences using self-attention and feed-forward layers.

---

## V

### Value (V)
In attention mechanisms, values are what gets retrieved/aggregated based on attention weights.

### Vocabulary
The set of all possible locations. Size denoted as $V$ or num_locations.

---

## W

### Weight (Model)
Learnable parameters in the neural network, optimized during training.

### Weekday
Day of the week (0-6), one of the temporal features.

---

## Symbols and Notation

| Symbol | Meaning |
|--------|---------|
| $B$ | Batch size |
| $T$ | Sequence length |
| $d$ | Model dimension (d_model) |
| $h$ | Number of attention heads |
| $d_k$ | Key/query dimension per head |
| $V$ | Vocabulary size |
| $\alpha$ | Attention weights |
| $g$ | Gate value |
| $H$ | Entropy |
| $\sigma$ | Sigmoid function |
| $\mathbf{Q}, \mathbf{K}, \mathbf{V}$ | Query, Key, Value matrices |
| $\odot$ | Element-wise multiplication |

---

## Acronyms

| Acronym | Full Form |
|---------|-----------|
| DIY | Do-It-Yourself (dataset name) |
| FFN | Feed-Forward Network |
| GELU | Gaussian Error Linear Unit |
| GPS | Global Positioning System |
| MHA | Multi-Head Attention |
| MLP | Multi-Layer Perceptron |
| ReLU | Rectified Linear Unit |

---

*Glossary - Version 1.0*
*Quick reference for all terminology*
