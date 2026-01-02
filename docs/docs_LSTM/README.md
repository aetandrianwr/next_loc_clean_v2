# LSTM Model Documentation

## Overview

This directory contains comprehensive documentation for the LSTM (Long Short-Term Memory) model implementation for next location prediction.

---

## Documentation Structure

### 📖 Main Documentation

| Document | Description | Audience |
|----------|-------------|----------|
| **[01_LSTM_Model_Complete_Documentation.md](01_LSTM_Model_Complete_Documentation.md)** | Comprehensive technical documentation covering theory, architecture, training, and evaluation | Researchers, ML Engineers |
| **[02_LSTM_Code_Walkthrough_With_Examples.md](02_LSTM_Code_Walkthrough_With_Examples.md)** | Line-by-line code explanation with concrete numerical examples | Developers, Students |
| **[03_Architecture_Diagrams_and_Visualizations.md](03_Architecture_Diagrams_and_Visualizations.md)** | Visual diagrams from simplified to detailed, ASCII art architecture diagrams | All audiences |
| **[04_Quick_Reference_Guide.md](04_Quick_Reference_Guide.md)** | Quick reference for commands, hyperparameters, and common tasks | Practitioners |

---

## Quick Navigation

### For Beginners
1. Start with **04_Quick_Reference_Guide.md** for a high-level overview
2. Look at **03_Architecture_Diagrams_and_Visualizations.md** Section 1 (Simplified Overview)
3. Read **01_LSTM_Model_Complete_Documentation.md** Sections 1-2

### For Understanding the Code
1. Read **02_LSTM_Code_Walkthrough_With_Examples.md** end-to-end
2. Refer to **03_Architecture_Diagrams_and_Visualizations.md** for visual understanding

### For Implementation
1. **04_Quick_Reference_Guide.md** for commands and configurations
2. **01_LSTM_Model_Complete_Documentation.md** Section 12 (Usage Guide)

### For Research/Paper Writing
1. **01_LSTM_Model_Complete_Documentation.md** Sections 3 (Theoretical Background), 9 (Results)
2. **03_Architecture_Diagrams_and_Visualizations.md** for figure references

---

## Key Topics Covered

### Theoretical Foundation
- RNN fundamentals and limitations
- LSTM architecture and gating mechanisms
- Why LSTM works for location prediction
- Embedding theory

### Architecture Details
- AllEmbeddingLSTM (feature combination)
- TemporalEmbedding (time encoding)
- LSTM Encoder (sequence processing)
- FullyConnected Output (prediction layer)
- Weight initialization strategies

### Data Pipeline
- Preprocessing workflow
- Sequence generation
- Variable-length handling
- Batch processing

### Training Process
- Optimizer configuration
- Learning rate scheduling
- Early stopping strategy
- Gradient clipping

### Evaluation
- Accuracy@K metrics
- Mean Reciprocal Rank (MRR)
- NDCG@10
- F1 Score

---

## Related Source Code

```
next_loc_clean_v2/
├── src/
│   ├── models/baseline/LSTM.py       # Model definition
│   ├── training/train_LSTM.py        # Training script
│   └── evaluation/metrics.py         # Evaluation metrics
├── config/models/
│   ├── config_LSTM_geolife.yaml      # GeoLife config
│   └── config_LSTM_diy.yaml          # DIY config
└── preprocessing/
    └── geolife_2_interim_to_processed.py  # Data preprocessing
```

---

## Quick Commands

```bash
# Train LSTM on GeoLife dataset
python src/training/train_LSTM.py --config config/models/config_LSTM_geolife.yaml

# Train LSTM on DIY dataset
python src/training/train_LSTM.py --config config/models/config_LSTM_diy.yaml
```

---

## Model Performance Summary

| Dataset | Acc@1 | Acc@5 | Acc@10 | MRR | Parameters |
|---------|-------|-------|--------|-----|------------|
| GeoLife | ~32% | ~56% | ~60% | ~43% | ~481K |
| DIY | ~50% | ~77% | ~81% | ~63% | ~2.85M |

---

## Document Statistics

| Document | Lines | Size |
|----------|-------|------|
| 01_Complete_Documentation | ~1200 | ~47KB |
| 02_Code_Walkthrough | ~900 | ~33KB |
| 03_Architecture_Diagrams | ~800 | ~35KB |
| 04_Quick_Reference | ~200 | ~4KB |

---

## References

1. **Hochreiter, S., & Schmidhuber, J. (1997)**. "Long short-term memory." Neural computation, 9(8), 1735-1780.

2. **Hong, Y., et al. (2023)**. "Context-aware multi-head self-attentional neural network model for next location prediction." Transportation Research Part C, 156, 104315.

---

## Version Information

- **Documentation Version**: 1.0
- **Last Updated**: January 2026
- **Model Version**: Compatible with PyTorch 1.x and 2.x
- **Python Version**: 3.8+

---

## Contributing

To update this documentation:
1. Ensure code changes are reflected in documentation
2. Update examples if hyperparameters change
3. Keep diagrams synchronized with architecture

---

**Author**: Generated from source code analysis
**Contact**: See repository maintainers
