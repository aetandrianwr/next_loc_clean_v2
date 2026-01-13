# Pointer-Generator Network Documentation

Comprehensive documentation for the pointer-generator network architecture, designed as a PhD-level reference for understanding and adapting the model.

## Overview

This documentation covers the pointer-generator network for abstractive text summarization, as implemented in the reference codebase at `/workspace/pointer-generator/`. The documentation is structured to serve as a source of truth for understanding the architecture and adapting it to other domains, such as next location prediction.

```
┌──────────────────────────────────────────────────────────────────────────────────────────────┐
│                           POINTER-GENERATOR NETWORK                                           │
├──────────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                               │
│   Key Innovation: Combines GENERATION from vocabulary with COPYING from input                │
│                                                                                               │
│   P(word) = p_gen × P_vocab(word) + (1 - p_gen) × P_copy(word)                              │
│                                                                                               │
│   • Handles out-of-vocabulary words (names, technical terms)                                 │
│   • Reduces factual errors by copying facts from source                                      │
│   • Coverage mechanism prevents repetition                                                   │
│                                                                                               │
└──────────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Documentation Structure

### Foundational Concepts

| Document | Description |
|----------|-------------|
| [01_overview.md](01_overview.md) | High-level architecture overview, file structure, key components |
| [02_theoretical_foundations.md](02_theoretical_foundations.md) | Mathematical foundations: Seq2Seq, LSTM, attention theory |
| [03_architecture_deep_dive.md](03_architecture_deep_dive.md) | Complete architecture with tensor shapes and data flow |

### Core Mechanisms

| Document | Description |
|----------|-------------|
| [04_attention_mechanism.md](04_attention_mechanism.md) | Bahdanau attention deep dive with equations |
| [05_pointer_mechanism.md](05_pointer_mechanism.md) | Pointer network and copy mechanism explained |
| [06_coverage_mechanism.md](06_coverage_mechanism.md) | Coverage vector and repetition prevention |

### Data and Training

| Document | Description |
|----------|-------------|
| [07_data_pipeline.md](07_data_pipeline.md) | batcher.py deep dive, Example/Batch classes |
| [08_vocabulary.md](08_vocabulary.md) | Vocab class, OOV handling, dual encoding |
| [10_training_pipeline.md](10_training_pipeline.md) | Training loop, Adagrad optimization |
| [11_loss_functions.md](11_loss_functions.md) | NLL loss, coverage loss, masking |

### Inference

| Document | Description |
|----------|-------------|
| [09_beam_search.md](09_beam_search.md) | Hypothesis class, beam search algorithm |

### Code Walkthroughs

| Document | Description |
|----------|-------------|
| [12_model_py_walkthrough.md](12_model_py_walkthrough.md) | Line-by-line model.py analysis |
| [13_attention_decoder_walkthrough.md](13_attention_decoder_walkthrough.md) | Line-by-line attention_decoder.py analysis |

### Worked Examples and Reference

| Document | Description |
|----------|-------------|
| [14_running_example.md](14_running_example.md) | End-to-end worked example with consistent data |
| [15_diagrams.md](15_diagrams.md) | All visual diagrams consolidated |
| [16_glossary.md](16_glossary.md) | Key terms and definitions |
| [17_adaptation_guide.md](17_adaptation_guide.md) | How to adapt to other domains |

---

## Quick Start Reading Path

### Path 1: Conceptual Understanding
If you want to understand the theory:
```
01_overview → 02_theoretical_foundations → 03_architecture_deep_dive
```

### Path 2: Implementation Focus
If you want to implement the model:
```
12_model_py_walkthrough → 13_attention_decoder_walkthrough → 07_data_pipeline
```

### Path 3: Deep Dive into Mechanisms
If you want to understand specific components:
```
04_attention_mechanism → 05_pointer_mechanism → 06_coverage_mechanism
```

### Path 4: Adaptation to Other Domains
If you want to adapt the model:
```
01_overview → 05_pointer_mechanism → 17_adaptation_guide
```

### Path 5: Complete Reference
For comprehensive understanding:
```
01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13 → 14 → 15 → 16 → 17
```

---

## Key Equations Quick Reference

### Attention Scores
```
e_ti = v^T · tanh(W_h · h_i + W_s · s_t + w_c · c_t^i + b_attn)
α_ti = softmax(e_ti)
```

### Context Vector
```
c_t = Σ_i α_ti · h_i
```

### Generation Probability
```
p_gen = σ(w_c^T · c_t + w_s^T · s_t + w_x^T · x_t + b_ptr)
```

### Final Distribution
```
P(w) = p_gen × P_vocab(w) + (1 - p_gen) × Σ_{i:w_i=w} α_ti
```

### Coverage Update
```
coverage_t = Σ_{t'<t} α_t'
```

### Coverage Loss
```
L_cov = Σ_i min(α_ti, coverage_t^i)
```

---

## Source Code Reference

| File | Purpose | Lines |
|------|---------|-------|
| `model.py` | Main model class, encoder, loss | 481 |
| `attention_decoder.py` | Attention mechanism, decoder | 229 |
| `batcher.py` | Data loading, batching | 375 |
| `data.py` | Vocabulary, OOV handling | 277 |
| `beam_search.py` | Beam search decoding | 167 |
| `decode.py` | Decoding pipeline, ROUGE | 254 |
| `run_summarization.py` | Entry point, hyperparameters | 325 |

---

## Default Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `hidden_dim` | 256 | LSTM hidden size |
| `emb_dim` | 128 | Embedding dimension |
| `vocab_size` | 50,000 | Vocabulary size |
| `batch_size` | 16 | Training batch size |
| `max_enc_steps` | 400 | Max encoder length |
| `max_dec_steps` | 100 | Max decoder length |
| `beam_size` | 4 | Beam search width |
| `lr` | 0.15 | Learning rate (Adagrad) |
| `max_grad_norm` | 2.0 | Gradient clipping |
| `cov_loss_wt` | 1.0 | Coverage loss weight |

---

## Example Throughout Documentation

All documents use a consistent running example:

```
Article: "Germany beat Argentina 1-0 in the World Cup final. 
          Mario Götze scored the winning goal."

Summary: "Germany won the World Cup."
```

This example demonstrates:
- **Copying**: "Germany" (proper noun in article)
- **Generating**: "won" (not in article but common word)
- **OOV handling**: "Götze" (not in vocabulary but can be copied)
- **Factual accuracy**: Copying entities ensures correctness

---

## Adapted Implementation

The documentation also references an adapted implementation for next location prediction:

| File | Description |
|------|-------------|
| `pointer_v45.py` | PyTorch implementation for location prediction |
| `train_pointer_v45.py` | Training script for adapted model |

Key adaptations:
- Transformer encoder (instead of LSTM)
- Location + User + Temporal embeddings
- Single prediction (instead of sequence generation)
- Position bias (instead of coverage)

See [17_adaptation_guide.md](17_adaptation_guide.md) for detailed comparison.

---

## Citation

If using this documentation for academic work:

```bibtex
@article{see2017get,
  title={Get To The Point: Summarization with Pointer-Generator Networks},
  author={See, Abigail and Liu, Peter J and Manning, Christopher D},
  journal={arXiv preprint arXiv:1704.04368},
  year={2017}
}
```

---

## Navigation

| Previous | Current | Next |
|----------|---------|------|
| - | README.md | [01_overview.md](01_overview.md) |

---

*This documentation was created to provide a comprehensive reference for understanding the pointer-generator network architecture and its applications.*
