# Glossary of Key Terms

A comprehensive reference of terminology used in the pointer-generator network and sequence-to-sequence models.

## Table of Contents
1. [Core Concepts](#core-concepts)
2. [Neural Network Terms](#neural-network-terms)
3. [Attention and Pointer Mechanisms](#attention-and-pointer-mechanisms)
4. [Data Processing Terms](#data-processing-terms)
5. [Training and Optimization](#training-and-optimization)
6. [Evaluation and Inference](#evaluation-and-inference)
7. [Mathematical Notation](#mathematical-notation)

---

## Core Concepts

### Abstractive Summarization
**Definition**: A text summarization approach that generates new phrases and sentences not present in the original text, as opposed to extractive summarization which only selects existing sentences.

**Example**:
- Source: "Germany defeated Argentina 1-0 in the World Cup final held in Brazil"
- Abstractive: "Germany won the World Cup" (rephrased)
- Extractive: "Germany defeated Argentina 1-0" (directly copied)

---

### Sequence-to-Sequence (Seq2Seq)
**Definition**: A neural network architecture that transforms an input sequence into an output sequence of potentially different length. Consists of an encoder that processes the input and a decoder that generates the output.

**Formula**:
```
Input: [x₁, x₂, ..., xₙ] → Encoder → Hidden State → Decoder → [y₁, y₂, ..., yₘ]
```

**Use Cases**: Machine translation, text summarization, dialogue systems, question answering.

---

### Encoder
**Definition**: The component that processes the input sequence and compresses it into a fixed-size representation (context vector) or a sequence of hidden states.

**In Pointer-Generator**:
- Uses bidirectional LSTM
- Input: Token embeddings
- Output: Encoder states [batch, seq_len, 512] and initial decoder state

---

### Decoder
**Definition**: The component that generates the output sequence one token at a time, conditioned on the encoder output and previously generated tokens.

**In Pointer-Generator**:
- Single-layer LSTM
- Uses attention over encoder outputs
- Generates tokens via pointer-generator mechanism

---

### Context Vector
**Definition**: A fixed-size vector summarizing the relevant information from the input sequence for generating the current output token. Computed as a weighted sum of encoder hidden states.

**Formula**:
```
c_t = Σᵢ αᵢ × hᵢ
```
Where αᵢ is the attention weight for encoder position i.

---

## Neural Network Terms

### LSTM (Long Short-Term Memory)
**Definition**: A type of recurrent neural network (RNN) designed to learn long-term dependencies. Uses gates (forget, input, output) to control information flow.

**Gates**:
1. **Forget Gate**: Decides what to discard from cell state
2. **Input Gate**: Decides what new information to store
3. **Output Gate**: Decides what to output based on cell state

**Equations**:
```
fₜ = σ(Wf·[hₜ₋₁, xₜ] + bf)        # Forget gate
iₜ = σ(Wi·[hₜ₋₁, xₜ] + bi)        # Input gate
C̃ₜ = tanh(Wc·[hₜ₋₁, xₜ] + bc)    # Candidate cell state
Cₜ = fₜ ⊙ Cₜ₋₁ + iₜ ⊙ C̃ₜ         # New cell state
oₜ = σ(Wo·[hₜ₋₁, xₜ] + bo)        # Output gate
hₜ = oₜ ⊙ tanh(Cₜ)                 # Hidden state
```

---

### Bidirectional LSTM (BiLSTM)
**Definition**: An LSTM that processes the input sequence in both forward and backward directions, capturing both past and future context.

**Output**:
```
hᵢ = [h⃗ᵢ; h⃖ᵢ]  # Concatenation of forward and backward states
```

**In Pointer-Generator**: Each direction has hidden_dim=256, concatenated to 512.

---

### Embedding Layer
**Definition**: A lookup table that converts discrete token IDs to continuous dense vectors (embeddings).

**Parameters**:
- `vocab_size`: Number of unique tokens
- `embedding_dim`: Dimension of each embedding vector

**In Pointer-Generator**: vocab_size=50000, emb_dim=128

---

### Hidden State
**Definition**: The internal representation maintained by a recurrent neural network at each timestep. Encodes information about previously processed tokens.

**Symbol**: Usually denoted as `h` or `s`

**Dimensions in Pointer-Generator**: 256 for decoder, 512 for concatenated encoder states

---

### Cell State
**Definition**: The LSTM's "memory" component that carries information across timesteps. Modified by forget and input gates.

**Symbol**: Usually denoted as `c` or `C`

---

### Activation Functions

#### Sigmoid (σ)
**Definition**: Squashes input to range [0, 1]. Used for gates and probability outputs.
```
σ(x) = 1 / (1 + e⁻ˣ)
```

#### Tanh
**Definition**: Squashes input to range [-1, 1]. Used for cell state candidate and final output transformations.
```
tanh(x) = (eˣ - e⁻ˣ) / (eˣ + e⁻ˣ)
```

#### Softmax
**Definition**: Converts a vector of real numbers into a probability distribution.
```
softmax(xᵢ) = eˣⁱ / Σⱼ eˣʲ
```

#### ReLU (Rectified Linear Unit)
**Definition**: Returns max(0, x). Not used in standard pointer-generator but common in other architectures.

---

## Attention and Pointer Mechanisms

### Attention Mechanism
**Definition**: A mechanism that allows the decoder to selectively focus on different parts of the input sequence when generating each output token.

**Types**:
1. **Bahdanau Attention** (Additive): Used in pointer-generator
2. **Luong Attention** (Multiplicative): Simpler, often faster
3. **Scaled Dot-Product Attention**: Used in Transformers

---

### Bahdanau Attention
**Definition**: An additive attention mechanism that computes attention scores using a learned alignment function.

**Formula**:
```
eᵢⱼ = v^T · tanh(W_h·hᵢ + W_s·sⱼ + b)
αᵢⱼ = softmax(eᵢⱼ)
```

**Characteristics**:
- Uses feedforward network to compute alignment
- Considers both encoder state and decoder state
- More expressive than dot-product attention

---

### Attention Weights (α)
**Definition**: A probability distribution over input positions indicating how much attention to pay to each position.

**Properties**:
- Sum to 1 (after softmax)
- Non-negative
- Higher weight = more attention

---

### Attention Score (e)
**Definition**: The raw, unnormalized "energy" or compatibility score between decoder state and each encoder position.

**Before softmax**: Can be any real number
**After softmax**: Becomes attention weight α

---

### Pointer Network
**Definition**: A neural network architecture that uses attention as a pointer to select positions from the input sequence, rather than generating from a fixed vocabulary.

**Original Paper**: "Pointer Networks" (Vinyals et al., 2015)

**Use Case**: Selecting from variable-length input (e.g., copying words, solving combinatorial problems)

---

### Pointer-Generator Network
**Definition**: A hybrid architecture combining:
1. A **generator** that produces words from a fixed vocabulary
2. A **pointer** that copies words from the input

**Key Innovation**: The p_gen gate decides the mixture between generating and copying.

---

### Generation Probability (p_gen)
**Definition**: A scalar probability (0 to 1) indicating whether to generate from vocabulary (p_gen) or copy from input (1 - p_gen).

**Formula**:
```
p_gen = σ(w_c^T·cₜ + w_s^T·sₜ + w_x^T·xₜ + b_ptr)
```

**Components**:
- `cₜ`: Context vector (from attention)
- `sₜ`: Decoder state
- `xₜ`: Decoder input embedding

---

### Copy Distribution
**Definition**: A probability distribution over input tokens, determined by attention weights. Allows generating OOV words by copying.

**Formula**:
```
P_copy(w) = Σᵢ:wᵢ=w αᵢ
```
Sum of attention weights for all positions containing word w.

---

### Vocabulary Distribution
**Definition**: A probability distribution over the fixed vocabulary, computed via softmax over the output projection.

**Formula**:
```
P_vocab = softmax(V × [sₜ; cₜ] + b)
```

---

### Final Distribution
**Definition**: The combined distribution used for word prediction, mixing vocabulary and copy distributions.

**Formula**:
```
P(w) = p_gen × P_vocab(w) + (1 - p_gen) × P_copy(w)
```

---

### Coverage Mechanism
**Definition**: A mechanism to prevent repetition by tracking which input positions have been attended to and penalizing re-attending.

**Coverage Vector**:
```
coverage_t = Σₜ'<ₜ αₜ'
```
Sum of all previous attention distributions.

---

### Coverage Loss
**Definition**: A loss term that penalizes attending to already-attended positions.

**Formula**:
```
L_cov = Σᵢ min(αₜᵢ, coverageₜᵢ)
```

**Effect**: Encourages attending to new positions at each step.

---

## Data Processing Terms

### Vocabulary (Vocab)
**Definition**: The set of unique tokens the model can recognize and generate. Typically includes the most frequent words in the training data.

**In Pointer-Generator**: 50,000 words by default

---

### Out-of-Vocabulary (OOV)
**Definition**: Words not in the vocabulary. Handled differently in encoding vs. decoding.

**Handling**:
- **Encoding**: Mapped to [UNK] for embedding, but assigned temporary IDs for copying
- **Decoding**: Can be generated via the copy mechanism

---

### UNK Token
**Definition**: A special token representing unknown/OOV words. Allows the model to handle any input text.

**ID**: Typically 0 or 1 (depending on vocabulary structure)

---

### START Token
**Definition**: A special token marking the beginning of a sequence. Used to initialize decoder input.

**Symbol**: `<s>`, `[START]`, or `[BOS]` (Beginning Of Sequence)

---

### STOP Token
**Definition**: A special token marking the end of a sequence. Signals the decoder to stop generating.

**Symbol**: `</s>`, `[STOP]`, or `[EOS]` (End Of Sequence)

---

### PAD Token
**Definition**: A special token used to fill sequences to a fixed length for batching.

**ID**: Typically 0
**Handling**: Masked out in attention and loss computation

---

### Tokenization
**Definition**: The process of splitting text into individual tokens (words, subwords, or characters).

**Example**:
- Input: "Germany won"
- Tokens: ["Germany", "won"]

---

### Extended Vocabulary
**Definition**: The vocabulary extended with temporary IDs for OOV words from the current article.

**Size**: vocab_size + max_article_oovs (e.g., 50000 + dynamic OOVs)

---

### Dual Encoding
**Definition**: The strategy of encoding input tokens twice:
1. `enc_input`: OOVs mapped to [UNK] (for embedding)
2. `enc_input_extend_vocab`: OOVs mapped to temporary IDs (for copying)

---

### Batch
**Definition**: A collection of examples processed together for efficiency.

**In Pointer-Generator**: batch_size=16

---

### Padding
**Definition**: Adding PAD tokens to make all sequences in a batch the same length.

**Example**:
- Sequence 1: [1, 2, 3] → [1, 2, 3, 0, 0]
- Sequence 2: [4, 5, 6, 7, 8] → [4, 5, 6, 7, 8]

---

### Padding Mask
**Definition**: A binary mask indicating which positions are real tokens (1) vs. padding (0).

**Usage**: Zero out attention scores and loss for padded positions.

---

## Training and Optimization

### Loss Function
**Definition**: A function measuring how well the model's predictions match the targets. Minimized during training.

---

### Negative Log-Likelihood (NLL) Loss
**Definition**: The standard loss for sequence generation, measuring the negative log probability of the correct token.

**Formula**:
```
L_NLL = -Σₜ log P(y*ₜ)
```
Where y*ₜ is the target token at step t.

---

### Cross-Entropy Loss
**Definition**: Equivalent to NLL for classification/generation tasks. Measures the "distance" between predicted and true distributions.

**Formula**:
```
CE = -Σᵢ yᵢ × log(ŷᵢ)
```

---

### Gradient
**Definition**: The partial derivative of the loss with respect to a parameter. Indicates the direction and magnitude of change to reduce loss.

**Symbol**: ∇L or ∂L/∂θ

---

### Backpropagation
**Definition**: The algorithm for computing gradients by propagating errors backward through the network.

---

### Gradient Clipping
**Definition**: Scaling down gradients when their norm exceeds a threshold. Prevents exploding gradients.

**In Pointer-Generator**: max_grad_norm=2.0

**Formula**:
```
if ||g|| > max_norm:
    g = g × (max_norm / ||g||)
```

---

### Optimizer
**Definition**: An algorithm that updates model parameters using gradients to minimize the loss function.

---

### Adagrad
**Definition**: An optimizer that adapts the learning rate for each parameter based on historical gradients.

**Formula**:
```
accumulator += gradient²
param -= lr × gradient / √(accumulator + ε)
```

**Characteristics**:
- Good for sparse features
- Learning rate decreases over time
- Used in original pointer-generator

---

### Learning Rate (lr)
**Definition**: A hyperparameter controlling the step size of parameter updates.

**In Pointer-Generator**: lr=0.15 (relatively high for Adagrad)

---

### Epoch
**Definition**: One complete pass through the entire training dataset.

---

### Step (Iteration)
**Definition**: One parameter update, typically processing one batch.

---

### Checkpoint
**Definition**: A saved snapshot of model parameters at a particular training step.

**Usage**: Resume training, evaluate at different stages, deploy best model.

---

## Evaluation and Inference

### Inference
**Definition**: Using a trained model to make predictions on new data.

**Modes**:
- **Training**: Uses teacher forcing
- **Inference**: Uses model's own predictions

---

### Teacher Forcing
**Definition**: During training, using the ground truth previous token as input to the decoder, rather than the model's prediction.

**Advantage**: Faster training convergence
**Disadvantage**: Exposure bias (train/test mismatch)

---

### Greedy Decoding
**Definition**: At each step, selecting the token with highest probability.

**Formula**:
```
yₜ = argmax P(y|y<t, x)
```

**Pros**: Fast, simple
**Cons**: May miss globally optimal sequences

---

### Beam Search
**Definition**: A search algorithm that maintains multiple hypotheses (beams) and expands the most promising ones.

**Parameters**:
- `beam_size` (K): Number of hypotheses to maintain
- `max_dec_steps`: Maximum decoding length

---

### Beam
**Definition**: A single partial hypothesis during beam search.

**Contents**:
- Token sequence generated so far
- Cumulative log probability
- Decoder states
- Coverage (if applicable)

---

### Hypothesis
**Definition**: A candidate output sequence during decoding, with associated probability.

---

### Length Normalization
**Definition**: Dividing log probability by sequence length to avoid favoring shorter sequences.

**Formula**:
```
score = log_prob / length
```

---

### ROUGE Score
**Definition**: Recall-Oriented Understudy for Gisting Evaluation. A metric comparing generated summaries to reference summaries.

**Variants**:
- **ROUGE-1**: Unigram overlap
- **ROUGE-2**: Bigram overlap
- **ROUGE-L**: Longest common subsequence

---

### Perplexity
**Definition**: A measure of how well a probability model predicts a sample. Lower is better.

**Formula**:
```
PPL = exp(-1/N × Σᵢ log P(wᵢ))
```

---

## Mathematical Notation

### Common Symbols

| Symbol | Meaning |
|--------|---------|
| `x` | Input sequence |
| `y` | Output sequence |
| `h` | Hidden state (encoder or general) |
| `s` | Decoder hidden state |
| `c` | Context vector or cell state |
| `α` | Attention weights |
| `e` | Attention scores (energy) |
| `W` | Weight matrix |
| `b` | Bias vector |
| `v` | Attention parameter vector |
| `θ` | Model parameters (general) |
| `σ` | Sigmoid activation |
| `⊙` | Element-wise multiplication |
| `×` | Matrix multiplication |
| `^T` | Transpose |
| `Σ` | Summation |
| `∂` | Partial derivative |
| `∇` | Gradient |

---

### Dimensions Reference

| Tensor | Shape | Description |
|--------|-------|-------------|
| `enc_batch` | [B, enc_len] | Encoder input IDs |
| `dec_batch` | [B, dec_len] | Decoder input IDs |
| `embeddings` | [vocab, emb_dim] | Embedding matrix |
| `encoder_outputs` | [B, enc_len, 512] | Encoder hidden states |
| `decoder_state` | [B, 256] | Decoder hidden state |
| `attention_dist` | [B, enc_len] | Attention weights |
| `context_vector` | [B, 512] | Weighted encoder sum |
| `vocab_dist` | [B, vocab_size] | Vocabulary distribution |
| `final_dist` | [B, vocab_size + max_oovs] | Final distribution |
| `p_gen` | [B, 1] | Generation probability |
| `coverage` | [B, enc_len] | Cumulative attention |

---

### Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 256 | LSTM hidden size |
| `emb_dim` | 128 | Embedding dimension |
| `vocab_size` | 50000 | Vocabulary size |
| `batch_size` | 16 | Training batch size |
| `max_enc_steps` | 400 | Max encoder sequence length |
| `max_dec_steps` | 100 | Max decoder sequence length |
| `beam_size` | 4 | Beam search width |
| `lr` | 0.15 | Learning rate |
| `max_grad_norm` | 2.0 | Gradient clipping threshold |
| `cov_loss_wt` | 1.0 | Coverage loss weight |

---

## Summary

This glossary provides definitions for all key terms used in the pointer-generator documentation. Use it as a reference when reading other documents or implementing your own version.

---

*Next: [17_adaptation_guide.md](17_adaptation_guide.md) - How to Adapt to Other Domains*
