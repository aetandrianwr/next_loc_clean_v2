# LSTM Model: Quick Reference Guide

## 1. One-Minute Overview

**What**: LSTM-based neural network for predicting the next location a user will visit.

**Input**: Sequence of visited locations with time and duration features.

**Output**: Probability distribution over all possible locations.

**Key Files**:
- Model: `src/models/baseline/LSTM.py`
- Training: `src/training/train_LSTM.py`
- Config: `config/models/config_LSTM_*.yaml`

---

## 2. Quick Start Commands

```bash
# Activate environment
conda activate mlenv

# Train on GeoLife
python src/training/train_LSTM.py --config config/models/config_LSTM_geolife.yaml

# Train on DIY
python src/training/train_LSTM.py --config config/models/config_LSTM_diy.yaml
```

---

## 3. Architecture Summary

```
Location + Time + Duration → Embeddings → LSTM (2 layers) → FC + User Emb → Prediction
```

| Component | Input Dim | Output Dim | Parameters |
|-----------|-----------|------------|------------|
| Embedding | N/A | 32 | ~42K |
| LSTM | 32 | 128 | ~215K |
| FC Output | 128 | 1187 | ~224K |
| **Total** | - | - | **~481K** |

---

## 4. Key Hyperparameters

### GeoLife Dataset

| Parameter | Value |
|-----------|-------|
| `base_emb_size` | 32 |
| `lstm_hidden_size` | 128 |
| `lstm_num_layers` | 2 |
| `lstm_dropout` | 0.2 |
| `batch_size` | 32 |
| `learning_rate` | 0.001 |
| `patience` | 3 |

### DIY Dataset

| Parameter | Value |
|-----------|-------|
| `base_emb_size` | 96 |
| `lstm_hidden_size` | 192 |
| `lstm_num_layers` | 2 |
| `lstm_dropout` | 0.2 |
| `batch_size` | 256 |
| `learning_rate` | 0.001 |
| `patience` | 3 |

---

## 5. Expected Performance

| Dataset | Acc@1 | Acc@5 | MRR | Train Time |
|---------|-------|-------|-----|------------|
| GeoLife | ~32% | ~56% | ~43% | ~4 min |
| DIY | ~50% | ~77% | ~63% | ~15 min |

---

## 6. Input Format

Each training sample is a dictionary:
```python
{
    "X": [102, 45, 103, ...],      # Location sequence
    "Y": 89,                        # Target location
    "user_X": [5, 5, 5, ...],      # User ID (repeated)
    "weekday_X": [0, 0, 1, ...],   # Day of week (0=Mon)
    "start_min_X": [540, 720, ...],# Time in minutes
    "dur_X": [120, 60, ...],       # Duration in minutes
    "diff": [6, 5, 4, ...]         # Days until target
}
```

---

## 7. Model Usage Code

```python
from src.models.baseline.LSTM import LSTMModel

# Load model
model = LSTMModel(config=config, total_loc_num=1187)
model.load_state_dict(torch.load("checkpoint.pt"))
model.eval()

# Inference
with torch.no_grad():
    logits = model(x, x_dict, device)
    prediction = torch.argmax(logits, dim=-1)
```

---

## 8. Output Files

Training creates:
```
experiments/{dataset}_{model}_{timestamp}/
├── checkpoints/checkpoint.pt    # Best model weights
├── training.log                 # Training logs
├── config.yaml                  # Used configuration
├── val_results.json             # Validation metrics
└── test_results.json            # Test metrics
```

---

## 9. Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| CUDA out of memory | Reduce `batch_size` |
| Training stuck | Check `learning_rate`, reduce if needed |
| Poor accuracy | Increase `lstm_hidden_size` or `base_emb_size` |
| Overfitting | Increase `lstm_dropout` or `fc_dropout` |

---

## 10. Evaluation Metrics

| Metric | Description | Good Value |
|--------|-------------|------------|
| **Acc@1** | Top-1 accuracy | >30% |
| **Acc@5** | Target in top 5 | >55% |
| **MRR** | Mean Reciprocal Rank | >40% |
| **NDCG@10** | Ranking quality | >45% |
| **F1** | Weighted F1 score | >0.20 |

---

## 11. Important Code Snippets

### Forward Pass
```python
def forward(self, src, context_dict, device):
    emb = self.Embedding(src, context_dict)           # [seq, B, 32]
    packed = pack_padded_sequence(emb, seq_len)       # Efficient processing
    output, _ = self.lstm(packed)                     # LSTM
    output, _ = pad_packed_sequence(output)           # Unpack
    out = self.layer_norm(last_valid_output)          # [B, 128]
    return self.FC(out, context_dict["user"])         # [B, 1187]
```

### Loss Calculation
```python
CEL = nn.CrossEntropyLoss(ignore_index=0)
loss = CEL(logits, target)
```

---

## 12. Related Documentation

1. **Complete Documentation**: `01_LSTM_Model_Complete_Documentation.md`
2. **Code Walkthrough**: `02_LSTM_Code_Walkthrough_With_Examples.md`
3. **Architecture Diagrams**: `03_Architecture_Diagrams_and_Visualizations.md`

---

**Last Updated**: January 2026
