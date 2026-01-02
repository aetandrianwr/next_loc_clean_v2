# Markov Chain Model Documentation

## Comprehensive Documentation for 1st-Order Markov Chain Location Prediction

This folder contains complete documentation for the Markov Chain baseline model implemented in `next_loc_clean_v2/src/models/baseline/markov_ori/run_markov_ori.py`.

---

## 📚 Documentation Index

| Document | Description | Reading Time |
|----------|-------------|--------------|
| [01_OVERVIEW.md](01_OVERVIEW.md) | Executive summary, quick start, and key facts | 10 min |
| [02_THEORY_BACKGROUND.md](02_THEORY_BACKGROUND.md) | Mathematical foundations, Markov property, probability theory | 25 min |
| [03_TECHNICAL_IMPLEMENTATION.md](03_TECHNICAL_IMPLEMENTATION.md) | Code architecture, data flow, function details | 30 min |
| [04_COMPONENTS_DEEP_DIVE.md](04_COMPONENTS_DEEP_DIVE.md) | Detailed analysis of each module and component | 35 min |
| [05_DIAGRAMS_VISUALIZATIONS.md](05_DIAGRAMS_VISUALIZATIONS.md) | Visual explanations at multiple detail levels | 20 min |
| [06_RESULTS_ANALYSIS.md](06_RESULTS_ANALYSIS.md) | Performance metrics, interpretation, comparisons | 25 min |
| [07_WALKTHROUGH_LINE_BY_LINE.md](07_WALKTHROUGH_LINE_BY_LINE.md) | Step-by-step code execution with examples | 30 min |

**Total Reading Time:** ~3 hours for complete documentation

---

## 🎯 Quick Navigation by Goal

### "I want to understand the basics"
→ Start with [01_OVERVIEW.md](01_OVERVIEW.md)

### "I need to understand the theory"
→ Read [02_THEORY_BACKGROUND.md](02_THEORY_BACKGROUND.md)

### "I want to modify the code"
→ Study [03_TECHNICAL_IMPLEMENTATION.md](03_TECHNICAL_IMPLEMENTATION.md) and [04_COMPONENTS_DEEP_DIVE.md](04_COMPONENTS_DEEP_DIVE.md)

### "I learn better with visuals"
→ Browse [05_DIAGRAMS_VISUALIZATIONS.md](05_DIAGRAMS_VISUALIZATIONS.md)

### "I need to interpret results"
→ Read [06_RESULTS_ANALYSIS.md](06_RESULTS_ANALYSIS.md)

### "I want to trace through with examples"
→ Follow [07_WALKTHROUGH_LINE_BY_LINE.md](07_WALKTHROUGH_LINE_BY_LINE.md)

---

## 🚀 Quick Start

```bash
# Activate environment
source ~/miniconda3/etc/profile.d/conda.sh && conda activate mlenv

# Navigate to project root
cd /data/next_loc_clean_v2

# Run on GeoLife dataset
python src/models/baseline/markov_ori/run_markov_ori.py \
    --config config/models/config_markov_ori_geolife.yaml

# Run on DIY dataset
python src/models/baseline/markov_ori/run_markov_ori.py \
    --config config/models/config_markov_ori_diy.yaml
```

---

## 📊 Performance Summary

| Dataset | Acc@1 | MRR | Training Time |
|---------|-------|-----|---------------|
| GeoLife | 24.18% | 30.34% | ~5 seconds |
| DIY | 44.13% | 52.13% | ~43 seconds |

---

## 📁 Related Files

```
next_loc_clean_v2/
├── src/models/baseline/markov_ori/
│   ├── run_markov_ori.py          ← Main implementation
│   └── README.md                   ← Quick reference
├── config/models/
│   ├── config_markov_ori_geolife.yaml
│   └── config_markov_ori_diy.yaml
├── data/
│   ├── geolife_eps20/markov_ori_data/
│   └── diy_eps50/interim/
└── experiments/
    └── {dataset}_markov_ori_{timestamp}/
```

---

## 📖 Documentation Highlights

### Theory (Document 02)
- Markov property and memorylessness
- Transition probability matrices
- Maximum likelihood estimation
- Mathematical justification

### Implementation (Document 03)
- Complete data flow pipeline
- Function-by-function breakdown
- Input/output formats
- Error handling

### Components (Document 04)
- Configuration system
- Data splitting logic
- Transition building
- Prediction mechanism
- Evaluation metrics

### Visualizations (Document 05)
- Simplified conceptual diagrams
- Moderate detail diagrams
- Detailed technical diagrams
- Data structure visualizations
- Algorithm flowcharts

### Results (Document 06)
- Metric interpretation guide
- GeoLife analysis
- DIY analysis
- Error analysis
- Comparison with neural models

### Walkthrough (Document 07)
- Line-by-line code trace
- Concrete numerical examples
- End-to-end execution trace
- Timing breakdown

---

## 🔗 Related Documentation

- [docs/markov1st_baseline.md](../markov1st_baseline.md) - Alternative implementation
- [docs/preprocessing.md](../preprocessing.md) - Data preprocessing
- [src/evaluation/metrics.py](../../src/evaluation/metrics.py) - Metrics module

---

## 📝 Version Information

- **Documentation Version:** 1.0
- **Last Updated:** January 2, 2026
- **Code Version:** run_markov_ori.py (faithful reproduction of original)

---

## 🤝 Contributing

If you find errors or have suggestions for improving this documentation:
1. Check if the issue is already documented
2. Verify against the source code
3. Propose clear, specific improvements
