# Deep Dive: Mathematical Foundations

## Complete Mathematical Treatment of Attention Mechanisms

This document provides rigorous mathematical explanations of every component in the attention visualization system, including derivations, proofs of properties, and detailed computational examples.

---

## Table of Contents

1. [Notation and Preliminaries](#1-notation-and-preliminaries)
2. [Softmax and Attention Fundamentals](#2-softmax-and-attention-fundamentals)
3. [Scaled Dot-Product Attention](#3-scaled-dot-product-attention)
4. [Multi-Head Attention Mathematics](#4-multi-head-attention-mathematics)
5. [Pointer Network Mathematics](#5-pointer-network-mathematics)
6. [Gate Mechanism Analysis](#6-gate-mechanism-analysis)
7. [Entropy and Information Theory](#7-entropy-and-information-theory)
8. [Position Encoding Mathematics](#8-position-encoding-mathematics)
9. [Gradient Flow Analysis](#9-gradient-flow-analysis)
10. [Numerical Examples](#10-numerical-examples)

---

## 1. Notation and Preliminaries

### 1.1 Notation Table

| Symbol | Meaning | Typical Shape |
|--------|---------|---------------|
| $B$ | Batch size | scalar |
| $T$ | Sequence length | scalar |
| $d$ | Model dimension (d_model) | scalar |
| $h$ | Number of attention heads | scalar |
| $d_k$ | Key/Query dimension per head | $d/h$ |
| $d_v$ | Value dimension per head | $d/h$ |
| $V$ | Vocabulary size (num_locations) | scalar |
| $\mathbf{X}$ | Input sequence | $B \times T \times d$ |
| $\mathbf{Q}$ | Query matrix | $B \times T \times d$ |
| $\mathbf{K}$ | Key matrix | $B \times T \times d$ |
| $\mathbf{V}$ | Value matrix | $B \times T \times d$ |
| $\boldsymbol{\alpha}$ | Attention weights | $B \times T \times T$ |
| $g$ | Gate value | $B \times 1$ |

### 1.2 Key Definitions

**Definition 1.1 (Attention Function)**
An attention function maps a query and a set of key-value pairs to an output:
$$\text{Attention}: (\mathbf{q}, \{(\mathbf{k}_i, \mathbf{v}_i)\}_{i=1}^T) \mapsto \sum_{i=1}^T \alpha_i \mathbf{v}_i$$

where $\alpha_i$ are attention weights satisfying:
$$\alpha_i \geq 0 \quad \text{and} \quad \sum_{i=1}^T \alpha_i = 1$$

**Definition 1.2 (Pointer Distribution)**
A pointer distribution over input positions is a probability distribution:
$$P_{\text{ptr}}: \{1, 2, ..., T\} \rightarrow [0, 1], \quad \sum_{i=1}^T P_{\text{ptr}}(i) = 1$$

---

## 2. Softmax and Attention Fundamentals

### 2.1 Softmax Function

**Definition 2.1 (Softmax)**
For a vector $\mathbf{z} = (z_1, ..., z_n) \in \mathbb{R}^n$:
$$\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_{j=1}^n \exp(z_j)}$$

**Properties of Softmax**:

1. **Output range**: $\text{softmax}(\mathbf{z})_i \in (0, 1)$ for all $i$

2. **Sum to one**: $\sum_{i=1}^n \text{softmax}(\mathbf{z})_i = 1$

3. **Translation invariance**: $\text{softmax}(\mathbf{z} + c) = \text{softmax}(\mathbf{z})$ for any scalar $c$

4. **Monotonicity**: If $z_i > z_j$, then $\text{softmax}(\mathbf{z})_i > \text{softmax}(\mathbf{z})_j$

**Proof of Property 3**:
$$\text{softmax}(\mathbf{z} + c)_i = \frac{\exp(z_i + c)}{\sum_j \exp(z_j + c)} = \frac{\exp(z_i) \cdot \exp(c)}{\exp(c) \cdot \sum_j \exp(z_j)} = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$$

### 2.2 Softmax with Temperature

**Definition 2.2 (Temperature-Scaled Softmax)**
$$\text{softmax}_\tau(\mathbf{z})_i = \frac{\exp(z_i / \tau)}{\sum_{j=1}^n \exp(z_j / \tau)}$$

**Temperature Effects**:

| $\tau$ | Effect | Limit Case |
|--------|--------|------------|
| $\tau \to 0^+$ | Approaches one-hot (argmax) | $\lim_{\tau \to 0^+} \text{softmax}_\tau = \text{one-hot}(\arg\max)$ |
| $\tau = 1$ | Standard softmax | - |
| $\tau \to \infty$ | Approaches uniform | $\lim_{\tau \to \infty} \text{softmax}_\tau(\mathbf{z}) = \frac{1}{n}\mathbf{1}$ |

### 2.3 Softmax Gradient

**Theorem 2.1 (Softmax Jacobian)**
Let $\mathbf{p} = \text{softmax}(\mathbf{z})$. The Jacobian matrix is:
$$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

where $\delta_{ij}$ is the Kronecker delta.

**In matrix form**:
$$\frac{\partial \mathbf{p}}{\partial \mathbf{z}} = \text{diag}(\mathbf{p}) - \mathbf{p}\mathbf{p}^T$$

**Proof**:
$$\frac{\partial p_i}{\partial z_j} = \frac{\partial}{\partial z_j}\left(\frac{\exp(z_i)}{\sum_k \exp(z_k)}\right)$$

Using quotient rule:
$$= \frac{\delta_{ij}\exp(z_i) \sum_k \exp(z_k) - \exp(z_i) \exp(z_j)}{(\sum_k \exp(z_k))^2}$$

$$= \frac{\exp(z_i)}{\sum_k \exp(z_k)} \cdot \delta_{ij} - \frac{\exp(z_i)}{\sum_k \exp(z_k)} \cdot \frac{\exp(z_j)}{\sum_k \exp(z_k)}$$

$$= p_i \delta_{ij} - p_i p_j = p_i(\delta_{ij} - p_j)$$

---

## 3. Scaled Dot-Product Attention

### 3.1 Definition

**Definition 3.1 (Scaled Dot-Product Attention)**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$$

Where:
- $\mathbf{Q} \in \mathbb{R}^{T_q \times d_k}$ (queries)
- $\mathbf{K} \in \mathbb{R}^{T_k \times d_k}$ (keys)
- $\mathbf{V} \in \mathbb{R}^{T_k \times d_v}$ (values)
- Output $\in \mathbb{R}^{T_q \times d_v}$

### 3.2 Why Scale by $\sqrt{d_k}$?

**Theorem 3.1 (Variance of Dot Products)**
If $\mathbf{q}, \mathbf{k} \in \mathbb{R}^{d_k}$ with components independently drawn from $\mathcal{N}(0, 1)$, then:
$$\mathbb{E}[\mathbf{q}^T\mathbf{k}] = 0, \quad \text{Var}(\mathbf{q}^T\mathbf{k}) = d_k$$

**Proof**:
$$\mathbf{q}^T\mathbf{k} = \sum_{i=1}^{d_k} q_i k_i$$

Since $q_i, k_i$ are independent with mean 0:
$$\mathbb{E}[q_i k_i] = \mathbb{E}[q_i]\mathbb{E}[k_i] = 0$$

$$\text{Var}(q_i k_i) = \mathbb{E}[q_i^2 k_i^2] - (\mathbb{E}[q_i k_i])^2 = \mathbb{E}[q_i^2]\mathbb{E}[k_i^2] = 1 \cdot 1 = 1$$

By independence:
$$\text{Var}(\mathbf{q}^T\mathbf{k}) = \sum_{i=1}^{d_k} \text{Var}(q_i k_i) = d_k$$

**Consequence**: Without scaling, dot products grow with $d_k$, pushing softmax into saturation regions where gradients vanish. Dividing by $\sqrt{d_k}$ normalizes variance to 1.

### 3.3 Attention with Masking

For variable-length sequences with padding:

$$\text{MaskedAttention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}, \mathbf{M}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

Where mask $\mathbf{M}_{ij} = \begin{cases} 0 & \text{if position } j \text{ is valid} \\ -\infty & \text{if position } j \text{ is padded} \end{cases}$

After softmax, masked positions have weight 0: $\exp(-\infty) = 0$.

---

## 4. Multi-Head Attention Mathematics

### 4.1 Definition

**Definition 4.1 (Multi-Head Attention)**
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, ..., \text{head}_h)\mathbf{W}^O$$

where each head is:
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

Projection matrices:
- $\mathbf{W}_i^Q \in \mathbb{R}^{d \times d_k}$
- $\mathbf{W}_i^K \in \mathbb{R}^{d \times d_k}$
- $\mathbf{W}_i^V \in \mathbb{R}^{d \times d_v}$
- $\mathbf{W}^O \in \mathbb{R}^{hd_v \times d}$

### 4.2 Parameter Count

For multi-head attention with $d_{model} = d$, $h$ heads:

$$\text{Parameters} = 3 \cdot h \cdot d \cdot d_k + h \cdot d_v \cdot d = 3d^2 + d^2 = 4d^2$$

(assuming $d_k = d_v = d/h$)

### 4.3 Head Independence

**Theorem 4.1 (Head Subspace Independence)**
Each attention head operates in an independent $d_k$-dimensional subspace of the $d$-dimensional embedding space.

**Proof Sketch**: The projection matrices $\mathbf{W}_i^Q, \mathbf{W}_i^K, \mathbf{W}_i^V$ define a linear map from $\mathbb{R}^d \to \mathbb{R}^{d_k}$. Different heads use different projections, so they attend to different aspects of the representation.

### 4.4 DIY vs Geolife Multi-Head Configuration

| Parameter | DIY | Geolife |
|-----------|-----|---------|
| $d_{model}$ | 64 | 96 |
| $h$ (heads) | 4 | 2 |
| $d_k$ | 16 | 48 |
| Parameters (MHA) | $4 \times 64^2 = 16,384$ | $4 \times 96^2 = 36,864$ |

**Implication**: 
- DIY: 4 heads with 16-dim subspaces → more specialization
- Geolife: 2 heads with 48-dim subspaces → richer per-head representations

---

## 5. Pointer Network Mathematics

### 5.1 Pointer Attention Computation

In PointerNetworkV45, the pointer mechanism computes:

**Step 1: Context Extraction**
$$\mathbf{c} = \mathbf{H}[b, L_b - 1, :] \quad \text{(last valid position)}$$

where $\mathbf{H} \in \mathbb{R}^{B \times T \times d}$ is the encoder output and $L_b$ is sequence length for batch $b$.

**Step 2: Query and Key Projection**
$$\mathbf{q} = \mathbf{c}\mathbf{W}_{\text{ptr}}^Q \in \mathbb{R}^{B \times d}$$
$$\mathbf{K}_{\text{ptr}} = \mathbf{H}\mathbf{W}_{\text{ptr}}^K \in \mathbb{R}^{B \times T \times d}$$

**Step 3: Raw Attention Scores**
$$s_i^{\text{raw}} = \frac{\mathbf{q} \cdot \mathbf{K}_{\text{ptr}}[:, i, :]}{\sqrt{d}}$$

**Step 4: Position Bias Addition**
$$s_i = s_i^{\text{raw}} + b_{\text{pos}}[\text{pos\_from\_end}[i]]$$

where $b_{\text{pos}} \in \mathbb{R}^{T_{max}}$ is a learnable parameter.

**Step 5: Masking and Softmax**
$$s_i^{\text{masked}} = \begin{cases} s_i & \text{if } i < L_b \\ -\infty & \text{otherwise} \end{cases}$$

$$\alpha_i^{\text{ptr}} = \text{softmax}(\mathbf{s}^{\text{masked}})_i$$

### 5.2 Scatter Operation for Location Distribution

**Definition 5.1 (Scatter-Add Operation)**
Given pointer attention $\boldsymbol{\alpha}^{\text{ptr}} \in \mathbb{R}^T$ over positions and input sequence $\mathbf{x} \in \{1, ..., V\}^T$:

$$P_{\text{ptr}}(\ell) = \sum_{i: x_i = \ell} \alpha_i^{\text{ptr}}$$

**In tensor notation**:
$$\mathbf{P}_{\text{ptr}} = \text{scatter\_add}(\mathbf{0}_V, \mathbf{x}, \boldsymbol{\alpha}^{\text{ptr}})$$

**Example**:
```
Sequence x:     [L5, L17, L5, L8]
Attention α:    [0.1, 0.3, 0.2, 0.4]

P_ptr(L5)  = α[0] + α[2] = 0.1 + 0.2 = 0.3
P_ptr(L17) = α[1] = 0.3
P_ptr(L8)  = α[3] = 0.4
P_ptr(other) = 0
```

### 5.3 Properties of Pointer Distribution

**Theorem 5.1 (Valid Probability Distribution)**
The pointer distribution $P_{\text{ptr}}$ is a valid probability distribution over locations.

**Proof**:
1. Non-negativity: $P_{\text{ptr}}(\ell) = \sum_{i: x_i = \ell} \alpha_i^{\text{ptr}} \geq 0$ (sum of non-negative terms)

2. Sum to one:
$$\sum_{\ell=1}^V P_{\text{ptr}}(\ell) = \sum_{\ell=1}^V \sum_{i: x_i = \ell} \alpha_i^{\text{ptr}} = \sum_{i=1}^T \alpha_i^{\text{ptr}} = 1$$

**Theorem 5.2 (Support of Pointer Distribution)**
The pointer distribution has support only on locations in the input sequence:
$$\text{supp}(P_{\text{ptr}}) = \{x_1, x_2, ..., x_T\} \subseteq \{1, ..., V\}$$

---

## 6. Gate Mechanism Analysis

### 6.1 Gate Computation

The gate is computed as:
$$g = \sigma(f_{\text{gate}}(\mathbf{c}))$$

where $f_{\text{gate}}$ is an MLP:
$$f_{\text{gate}}(\mathbf{c}) = \mathbf{W}_2 \cdot \text{GELU}(\mathbf{W}_1 \mathbf{c} + \mathbf{b}_1) + b_2$$

With:
- $\mathbf{W}_1 \in \mathbb{R}^{(d/2) \times d}$
- $\mathbf{W}_2 \in \mathbb{R}^{1 \times (d/2)}$
- $\sigma$ is the sigmoid function

### 6.2 Sigmoid Function Properties

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

**Properties**:
1. Range: $\sigma(x) \in (0, 1)$
2. Symmetry: $\sigma(-x) = 1 - \sigma(x)$
3. Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$
4. Fixed point: $\sigma(0) = 0.5$

### 6.3 Final Distribution Computation

$$P_{\text{final}}(\ell) = g \cdot P_{\text{ptr}}(\ell) + (1 - g) \cdot P_{\text{gen}}(\ell)$$

**Theorem 6.1 (Convex Combination)**
$P_{\text{final}}$ is a convex combination of $P_{\text{ptr}}$ and $P_{\text{gen}}$.

**Proof**: Since $g \in (0, 1)$:
1. Coefficients are non-negative: $g \geq 0$ and $(1-g) \geq 0$
2. Coefficients sum to one: $g + (1-g) = 1$
3. $P_{\text{final}}$ is in the convex hull of $\{P_{\text{ptr}}, P_{\text{gen}}\}$

**Corollary**: $P_{\text{final}}$ is a valid probability distribution (sum = 1, non-negative).

### 6.4 Gate Gradient Analysis

For loss $\mathcal{L} = -\log P_{\text{final}}(y)$ (cross-entropy with true label $y$):

$$\frac{\partial \mathcal{L}}{\partial g} = -\frac{1}{P_{\text{final}}(y)} \cdot (P_{\text{ptr}}(y) - P_{\text{gen}}(y))$$

**Interpretation**:
- If $P_{\text{ptr}}(y) > P_{\text{gen}}(y)$: Gradient is negative → increase $g$
- If $P_{\text{ptr}}(y) < P_{\text{gen}}(y)$: Gradient is positive → decrease $g$

The gate learns to favor the mechanism that gives higher probability to the correct answer!

---

## 7. Entropy and Information Theory

### 7.1 Shannon Entropy

**Definition 7.1 (Entropy)**
For discrete distribution $P$ over $n$ outcomes:
$$H(P) = -\sum_{i=1}^n p_i \log p_i$$

(with convention $0 \log 0 = 0$)

### 7.2 Entropy Bounds

**Theorem 7.1 (Entropy Bounds)**
For distribution over $n$ outcomes:
$$0 \leq H(P) \leq \log n$$

- Minimum $H = 0$: When $P$ is a point mass (all probability on one outcome)
- Maximum $H = \log n$: When $P$ is uniform

**Proof of upper bound (via Jensen's inequality)**:
$$H(P) = \sum_i p_i \log \frac{1}{p_i} \leq \log\left(\sum_i p_i \cdot \frac{1}{p_i}\right) = \log n$$

### 7.3 Effective Number of States

**Definition 7.2 (Perplexity)**
$$\text{Perplexity}(P) = 2^{H(P)} \quad \text{or} \quad e^{H(P)} \text{ (natural log)}$$

This gives the "effective number of states" the distribution spans.

**From our experiments**:
- DIY entropy: 2.34 nats → $e^{2.34} \approx 10.4$ effective positions
- Geolife entropy: 1.98 nats → $e^{1.98} \approx 7.2$ effective positions

### 7.4 Attention Entropy Computation

For pointer attention $\boldsymbol{\alpha}^{\text{ptr}}$ over valid positions:

$$H(\boldsymbol{\alpha}^{\text{ptr}}) = -\sum_{i=1}^{L} \alpha_i^{\text{ptr}} \log \alpha_i^{\text{ptr}}$$

**Numerical stability**: Use $\log(\alpha_i + \epsilon)$ with $\epsilon = 10^{-10}$ to avoid $\log(0)$.

### 7.5 Cross-Entropy Loss

**Definition 7.3 (Cross-Entropy)**
$$H(P, Q) = -\sum_i p_i \log q_i$$

For training with one-hot target $\mathbf{y}$ (where $y_c = 1$ for correct class $c$):
$$\mathcal{L} = -\log P_{\text{final}}(c)$$

---

## 8. Position Encoding Mathematics

### 8.1 Sinusoidal Positional Encoding

**Definition 8.1**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

### 8.2 Properties of Sinusoidal Encoding

**Theorem 8.1 (Relative Position as Linear Transform)**
For any fixed offset $k$, there exists a linear transformation $\mathbf{M}_k$ such that:
$$PE_{pos+k} = \mathbf{M}_k \cdot PE_{pos}$$

**Proof**:
Using trigonometric identities, for dimension pair $(2i, 2i+1)$:
$$\sin(\omega(pos+k)) = \sin(\omega \cdot pos)\cos(\omega k) + \cos(\omega \cdot pos)\sin(\omega k)$$
$$\cos(\omega(pos+k)) = \cos(\omega \cdot pos)\cos(\omega k) - \sin(\omega \cdot pos)\sin(\omega k)$$

where $\omega = 1/10000^{2i/d}$.

This can be written as:
$$\begin{pmatrix} PE_{pos+k, 2i} \\ PE_{pos+k, 2i+1} \end{pmatrix} = \begin{pmatrix} \cos(\omega k) & \sin(\omega k) \\ -\sin(\omega k) & \cos(\omega k) \end{pmatrix} \begin{pmatrix} PE_{pos, 2i} \\ PE_{pos, 2i+1} \end{pmatrix}$$

### 8.3 Position-from-End Embedding

In PointerNetworkV45, an additional learnable embedding based on position from sequence end:
$$\text{pos\_from\_end}_i = L - i$$

This is a **learnable** embedding, unlike sinusoidal encoding.

### 8.4 Combined Position Information

The model uses BOTH:
1. Sinusoidal encoding (absolute position)
2. Position-from-end embedding (relative to sequence end)
3. Position bias in pointer attention (learned preference)

This provides rich positional information for the model.

---

## 9. Gradient Flow Analysis

### 9.1 Backpropagation Through Attention

For attention output $\mathbf{O} = \text{softmax}(\mathbf{S})\mathbf{V}$ where $\mathbf{S} = \frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}$:

**Given**: $\frac{\partial \mathcal{L}}{\partial \mathbf{O}}$

**Compute**: $\frac{\partial \mathcal{L}}{\partial \mathbf{Q}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{K}}$, $\frac{\partial \mathcal{L}}{\partial \mathbf{V}}$

Let $\mathbf{A} = \text{softmax}(\mathbf{S})$, then:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}^T \frac{\partial \mathcal{L}}{\partial \mathbf{O}}$$

For attention weights:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \frac{\partial \mathcal{L}}{\partial \mathbf{O}} \mathbf{V}^T$$

Through softmax Jacobian (row-wise):
$$\frac{\partial \mathcal{L}}{\partial \mathbf{S}} = \mathbf{A} \odot \left(\frac{\partial \mathcal{L}}{\partial \mathbf{A}} - \text{rowsum}\left(\frac{\partial \mathcal{L}}{\partial \mathbf{A}} \odot \mathbf{A}\right)\right)$$

Finally:
$$\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \frac{1}{\sqrt{d_k}} \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \mathbf{K}$$
$$\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial \mathbf{S}}\right)^T \mathbf{Q}$$

### 9.2 Gradient Flow Through Gate

The gate creates two gradient pathways:

$$\frac{\partial \mathcal{L}}{\partial \theta_{\text{ptr}}} = g \cdot \frac{\partial \mathcal{L}}{\partial P_{\text{ptr}}} \cdot \frac{\partial P_{\text{ptr}}}{\partial \theta_{\text{ptr}}}$$

$$\frac{\partial \mathcal{L}}{\partial \theta_{\text{gen}}} = (1-g) \cdot \frac{\partial \mathcal{L}}{\partial P_{\text{gen}}} \cdot \frac{\partial P_{\text{gen}}}{\partial \theta_{\text{gen}}}$$

**Implication**: Higher gate value → stronger gradients to pointer mechanism → faster learning of pointer patterns.

### 9.3 Vanishing Gradient in Deep Attention

**Problem**: With many layers, gradients can vanish or explode.

**Solutions implemented in PointerNetworkV45**:
1. **Residual connections**: $\mathbf{X}_{l+1} = \mathbf{X}_l + \text{Sublayer}(\mathbf{X}_l)$
2. **Pre-norm**: $\text{Sublayer}(\text{LayerNorm}(\mathbf{X}_l))$
3. **Scaled attention**: Division by $\sqrt{d_k}$

---

## 10. Numerical Examples

### 10.1 Complete Pointer Attention Calculation

**Setup**:
- Sequence: [L5, L17, L5] (3 positions)
- d_model = 4
- Context vector $\mathbf{c} = [0.5, -0.3, 0.8, 0.1]$
- Encoder outputs $\mathbf{H}$:
  - Position 0: $[0.2, 0.4, -0.1, 0.3]$
  - Position 1: $[0.6, -0.2, 0.5, 0.1]$
  - Position 2: $[-0.1, 0.3, 0.4, -0.2]$
- Position bias: $[0.5, 0.3, 0.1]$ (for positions from end: 2, 1, 0)

**Step 1: Query projection** (assume $\mathbf{W}^Q = \mathbf{I}$ for simplicity)
$$\mathbf{q} = \mathbf{c} = [0.5, -0.3, 0.8, 0.1]$$

**Step 2: Dot products** (with $\mathbf{K} = \mathbf{H}$)
$$s_0^{\text{raw}} = \frac{\mathbf{q} \cdot \mathbf{H}_0}{\sqrt{4}} = \frac{0.5(0.2) + (-0.3)(0.4) + 0.8(-0.1) + 0.1(0.3)}{2}$$
$$= \frac{0.1 - 0.12 - 0.08 + 0.03}{2} = \frac{-0.07}{2} = -0.035$$

$$s_1^{\text{raw}} = \frac{0.5(0.6) + (-0.3)(-0.2) + 0.8(0.5) + 0.1(0.1)}{2}$$
$$= \frac{0.3 + 0.06 + 0.4 + 0.01}{2} = \frac{0.77}{2} = 0.385$$

$$s_2^{\text{raw}} = \frac{0.5(-0.1) + (-0.3)(0.3) + 0.8(0.4) + 0.1(-0.2)}{2}$$
$$= \frac{-0.05 - 0.09 + 0.32 - 0.02}{2} = \frac{0.16}{2} = 0.08$$

**Step 3: Add position bias**
$$s_0 = -0.035 + 0.5 = 0.465$$
$$s_1 = 0.385 + 0.3 = 0.685$$
$$s_2 = 0.08 + 0.1 = 0.18$$

**Step 4: Softmax**
$$\exp(0.465) = 1.592, \quad \exp(0.685) = 1.984, \quad \exp(0.18) = 1.197$$
$$\text{sum} = 1.592 + 1.984 + 1.197 = 4.773$$

$$\alpha_0 = 1.592/4.773 = 0.334$$
$$\alpha_1 = 1.984/4.773 = 0.416$$
$$\alpha_2 = 1.197/4.773 = 0.251$$

**Step 5: Scatter to locations**
$$P_{\text{ptr}}(L5) = \alpha_0 + \alpha_2 = 0.334 + 0.251 = 0.585$$
$$P_{\text{ptr}}(L17) = \alpha_1 = 0.416$$

**Result**: Model assigns 58.5% probability to L5, 41.6% to L17.

### 10.2 Gate Blending Example

**Given**:
- $P_{\text{ptr}}(L5) = 0.585$, $P_{\text{ptr}}(L17) = 0.416$
- $P_{\text{gen}}(L5) = 0.1$, $P_{\text{gen}}(L17) = 0.05$, $P_{\text{gen}}(\text{other}) = 0.85$
- Gate $g = 0.8$

**Final distribution**:
$$P_{\text{final}}(L5) = 0.8 \times 0.585 + 0.2 \times 0.1 = 0.468 + 0.02 = 0.488$$
$$P_{\text{final}}(L17) = 0.8 \times 0.416 + 0.2 \times 0.05 = 0.333 + 0.01 = 0.343$$
$$P_{\text{final}}(\text{other}) = 0.8 \times 0 + 0.2 \times 0.85 = 0 + 0.17 = 0.17$$

**Prediction**: L5 (highest probability at 48.8%)

### 10.3 Entropy Calculation

For attention $\boldsymbol{\alpha} = [0.334, 0.416, 0.251]$:

$$H = -(0.334 \log 0.334 + 0.416 \log 0.416 + 0.251 \log 0.251)$$
$$= -(0.334 \times (-1.097) + 0.416 \times (-0.877) + 0.251 \times (-1.382))$$
$$= -(-0.366 - 0.365 - 0.347)$$
$$= 1.078 \text{ nats}$$

**Effective positions**: $e^{1.078} = 2.94 \approx 3$ (matches our 3-position sequence)

---

## Summary of Key Equations

| Component | Equation |
|-----------|----------|
| Softmax | $\text{softmax}(\mathbf{z})_i = \frac{\exp(z_i)}{\sum_j \exp(z_j)}$ |
| Scaled Attention | $\text{Attention} = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right)\mathbf{V}$ |
| Pointer Scores | $s_i = \frac{\mathbf{q} \cdot \mathbf{k}_i}{\sqrt{d}} + b_{\text{pos}}[i]$ |
| Scatter | $P_{\text{ptr}}(\ell) = \sum_{i: x_i = \ell} \alpha_i$ |
| Gate Blend | $P_{\text{final}} = g \cdot P_{\text{ptr}} + (1-g) \cdot P_{\text{gen}}$ |
| Entropy | $H = -\sum_i p_i \log p_i$ |

---

*Mathematical Foundations - Version 1.0*
*For the rigorous reader who wants to understand every detail*
