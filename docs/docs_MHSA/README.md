# MHSA Model Documentation

## Multi-Head Self-Attention Model for Next Location Prediction

---

## Documentation Overview

This comprehensive documentation covers the MHSA (Multi-Head Self-Attention) model implementation for next location prediction. The documentation is organized into modular files for easy navigation.

### Quick Navigation

| Document | Description | Best For |
|----------|-------------|----------|
| [01_Complete_Documentation](01_MHSA_Complete_Documentation.md) | Full technical documentation | Complete understanding |
| [02_Line_by_Line_Walkthrough](02_MHSA_Line_by_Line_Walkthrough.md) | Code walkthrough with examples | Learning the code |
| [03_Architecture_Diagrams](03_Architecture_Diagrams.md) | Visual diagrams at all levels | Visual learners |
| [04_Training_Guide](04_Training_and_Configuration_Guide.md) | Training and configuration | Running experiments |
| [05_Evaluation_Metrics](05_Evaluation_Metrics_Deep_Dive.md) | Deep dive into metrics | Understanding results |

---

## Key Files Reference

### Model Code
- **Main Model:** `src/models/baseline/MHSA.py`
- **Training Script:** `src/training/train_MHSA.py`
- **Metrics:** `src/evaluation/metrics.py`

### Configuration
- **GeoLife Config:** `config/models/config_MHSA_geolife.yaml`
- **DIY Config:** `config/models/config_MHSA_diy.yaml`

### Data
- **GeoLife Data:** `data/geolife_eps20/processed/`
- **DIY Data:** `data/diy_eps50/processed/`

---

## Quick Start

### 1. Train on GeoLife Dataset

```bash
cd /data/next_loc_clean_v2
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv
python src/training/train_MHSA.py --config config/models/config_MHSA_geolife.yaml
```

### 2. Train on DIY Dataset

```bash
python src/training/train_MHSA.py --config config/models/config_MHSA_diy.yaml
```

### 3. View Results

```bash
cat experiments/geolife_MHSA_*/test_results.json
```

---

## Model Summary

### Architecture

```
Input → Embedding Layer → Transformer Encoder → Output Layer → Predictions
        (Loc+Time+Dur)    (Self-Attention)      (User+FC)    (Per-location)
```

### Key Components

1. **AllEmbedding**: Combines location, temporal, and duration embeddings
2. **TransformerEncoder**: Multi-head self-attention with causal masking
3. **FullyConnected**: User embedding + residual block + classification

### Performance

| Dataset | Acc@1 | Acc@5 | MRR | NDCG@10 |
|---------|-------|-------|-----|---------|
| GeoLife | ~30% | ~51% | ~40% | ~44% |
| DIY | ~53% | ~77% | ~63% | ~68% |

---

## Document Contents

### 01_MHSA_Complete_Documentation.md
- Executive summary
- Problem formulation
- Theoretical background (self-attention, positional encoding)
- Complete architecture description
- Component deep dive with math
- Data pipeline explanation
- Training process details
- Evaluation metrics
- Experimental results
- Configuration reference
- Usage guide
- Troubleshooting

### 02_MHSA_Line_by_Line_Walkthrough.md
- Sample input data explanation
- Step-by-step code walkthrough
- Numerical examples for each component
- Complete forward pass trace
- Training step breakdown
- Shape annotations throughout

### 03_Architecture_Diagrams.md
- High-level overview diagram
- Moderate detail diagram
- Detailed component diagrams:
  - AllEmbedding layer
  - Transformer encoder
  - FullyConnected layer
- Data flow diagram
- Parameter count breakdown
- Attention visualization examples

### 04_Training_and_Configuration_Guide.md
- Quick start commands
- Environment setup
- Data preparation checklist
- Configuration template with all options
- Training workflow diagram
- Monitoring and debugging tips
- Hyperparameter tuning guide
- Best practices

### 05_Evaluation_Metrics_Deep_Dive.md
- Accuracy@k explanation
- MRR definition and examples
- NDCG formulation
- F1 score for multi-class
- Metric comparison guide
- Implementation details
- Results interpretation guide

---

## Frequently Asked Questions

### Q: How do I change the embedding dimension?

Edit the config file:
```yaml
embedding:
  base_emb_size: 64  # Change from 32 to 64
```

### Q: Why is my accuracy low?

Check:
1. Data preprocessing is correct
2. `total_loc_num` matches metadata
3. Try different learning rates
4. Increase model capacity if underfitting

### Q: How do I know if training is working?

Look for:
- Decreasing training loss
- Validation loss following (with gap)
- Increasing validation accuracy
- Typical convergence in 10-30 epochs

### Q: Can I use this for a different dataset?

Yes, with modifications:
1. Preprocess data into `.pk` format
2. Update config with new `total_loc_num` and `total_user_num`
3. Adjust model size based on dataset complexity

---

## Contributing

When modifying the model or documentation:
1. Update relevant documentation files
2. Add numerical examples for new features
3. Update diagrams if architecture changes
4. Test on both datasets

---

## Citation

If using this model for research, please cite:

```bibtex
@article{hong2023context,
  title={Context-aware multi-head self-attentional neural network model for next location prediction},
  author={Hong, et al.},
  journal={Taylor & Francis},
  year={2023}
}
```

---

*Documentation for MHSA model - next_loc_clean_v2 project*
*Last updated: January 2026*
