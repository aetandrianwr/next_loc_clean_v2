# Sequence Length Days Experiment (V2) - Documentation Index

## 📚 Complete Documentation Suite

Welcome to the comprehensive documentation for the **Sequence Length Days Experiment V2**. This experiment investigates how the length of historical mobility data (in days) affects next location prediction performance.

---

## 📖 Documentation Files

### Core Documents

| # | Document | Description | Reading Time |
|---|----------|-------------|--------------|
| 1 | [Executive Summary](./01_executive_summary.md) | High-level overview, key findings, and conclusions | 5 min |
| 2 | [Introduction & Motivation](./02_introduction_and_motivation.md) | Background, research questions, and why this matters | 10 min |
| 3 | [Theoretical Foundation](./03_theoretical_foundation.md) | Scientific basis, human mobility theory, mathematical foundations | 15 min |
| 4 | [Experimental Methodology](./04_experimental_methodology.md) | Research design, controls, and scientific approach | 12 min |
| 5 | [Technical Implementation](./05_technical_implementation.md) | Code walkthrough, scripts, and execution details | 20 min |
| 6 | [Model Architecture](./06_model_architecture.md) | PointerGeneratorTransformer deep dive, components, and design | 18 min |
| 7 | [Datasets](./07_datasets.md) | DIY and GeoLife dataset details, statistics, preprocessing | 12 min |
| 8 | [Evaluation Metrics](./08_evaluation_metrics.md) | Acc@k, MRR, NDCG, F1, Loss - formulas and interpretation | 15 min |
| 9 | [Results & Analysis](./09_results_and_analysis.md) | Complete experimental results with statistical analysis | 20 min |
| 10 | [Visualization Guide](./10_visualization_guide.md) | Detailed explanation of all plots and figures | 25 min |
| 11 | [Interpretation & Insights](./11_interpretation_and_insights.md) | What the results mean, practical implications | 15 min |
| 12 | [Reproducibility Guide](./12_reproducibility.md) | How to reproduce results, environment setup | 10 min |

### Quick Reference

| Document | Description |
|----------|-------------|
| [COMPREHENSIVE_DOCUMENTATION.md](./COMPREHENSIVE_DOCUMENTATION.md) | All-in-one document (for quick reference) |

---

## 🎯 Quick Navigation by Use Case

### "I want to understand the experiment quickly"
→ Start with [Executive Summary](./01_executive_summary.md)

### "I want to understand the scientific motivation"
→ Read [Introduction](./02_introduction_and_motivation.md) → [Theoretical Foundation](./03_theoretical_foundation.md)

### "I want to reproduce the results"
→ Read [Reproducibility Guide](./12_reproducibility.md) → [Technical Implementation](./05_technical_implementation.md)

### "I want to understand the results"
→ Read [Results & Analysis](./09_results_and_analysis.md) → [Visualization Guide](./10_visualization_guide.md)

### "I want to understand the model"
→ Read [Model Architecture](./06_model_architecture.md)

### "I want to use these findings in my work"
→ Read [Interpretation & Insights](./11_interpretation_and_insights.md)

---

## 📊 Key Results at a Glance

| Dataset | Acc@1 (prev1→prev7) | Acc@5 (prev1→prev7) | Loss (prev1→prev7) |
|---------|---------------------|---------------------|-------------------|
| **DIY** | 50.00% → 56.58% (+13.2%) | 72.55% → 82.18% (+13.3%) | 3.763 → 2.874 (-23.6%) |
| **GeoLife** | 47.84% → 51.40% (+7.4%) | 70.00% → 81.18% (+16.0%) | 3.492 → 2.630 (-24.7%) |

**Main Finding**: Using 7 days of historical data instead of 1 day improves prediction accuracy by 7-13% across both datasets.

---

## 📁 File Structure

```
experiment_sequence_len_days_v2/
├── docs/                                    # 📚 Documentation
│   ├── INDEX.md                            # This file
│   ├── COMPREHENSIVE_DOCUMENTATION.md      # All-in-one reference
│   ├── 01_executive_summary.md
│   ├── 02_introduction_and_motivation.md
│   ├── 03_theoretical_foundation.md
│   ├── 04_experimental_methodology.md
│   ├── 05_technical_implementation.md
│   ├── 06_model_architecture.md
│   ├── 07_datasets.md
│   ├── 08_evaluation_metrics.md
│   ├── 09_results_and_analysis.md
│   ├── 10_visualization_guide.md
│   ├── 11_interpretation_and_insights.md
│   └── 12_reproducibility.md
├── results/                                 # 📈 Output files
│   ├── *.json                              # Raw results
│   ├── *.csv                               # Tabular data
│   ├── *.tex                               # LaTeX tables
│   └── *.{pdf,png,svg}                     # Visualizations
├── evaluate_sequence_length.py             # 🔬 Evaluation script
├── visualize_results.py                    # 📊 Visualization script
├── run_experiment.sh                       # 🚀 Master script
└── README.md                               # Quick start guide
```

---

## 📅 Document Information

- **Experiment Date**: January 2, 2026
- **Documentation Version**: 2.0
- **Total Documentation**: ~150,000 characters across 13 files
- **Author**: PhD Research - Next Location Prediction

---

## 🔗 Related Resources

- **Model Code**: `src/models/proposed/pgt.py`
- **Evaluation Metrics**: `src/evaluation/metrics.py`
- **Experiment Configs**: `scripts/sci_hyperparam_tuning/configs/`
- **Pre-trained Checkpoints**: `experiments/*/checkpoints/best.pt`

---

*Navigate to any document above to begin reading the detailed documentation.*
