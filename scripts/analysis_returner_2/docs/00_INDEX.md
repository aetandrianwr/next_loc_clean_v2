# Zipf's Law Location Frequency Analysis - Documentation Index

## 📚 Complete Documentation for PhD-Level Research

This directory contains comprehensive documentation for the **Zipf's Law Location Visit Frequency Analysis**, which reproduces Figure 2d from González et al. (2008) "Understanding Individual Human Mobility Patterns" (Nature).

---

## 📋 Table of Contents

### Part I: Foundations
| Document | Description |
|----------|-------------|
| [01_OVERVIEW.md](./01_OVERVIEW.md) | Executive summary and key findings |
| [02_ZIPF_LAW_THEORY.md](./02_ZIPF_LAW_THEORY.md) | Theoretical background on Zipf's Law in human mobility |

### Part II: Implementation
| Document | Description |
|----------|-------------|
| [03_CODE_WALKTHROUGH.md](./03_CODE_WALKTHROUGH.md) | Line-by-line code explanation with examples |
| [04_DATA_PIPELINE.md](./04_DATA_PIPELINE.md) | Input/output data formats and processing steps |

### Part III: Results & Analysis
| Document | Description |
|----------|-------------|
| [05_RESULTS_ANALYSIS.md](./05_RESULTS_ANALYSIS.md) | Comprehensive analysis of all results |
| [06_PLOT_INTERPRETATION.md](./06_PLOT_INTERPRETATION.md) | How to read and interpret the Zipf plots |
| [07_GEOLIFE_RESULTS.md](./07_GEOLIFE_RESULTS.md) | Detailed Geolife dataset results |
| [08_DIY_RESULTS.md](./08_DIY_RESULTS.md) | Detailed DIY dataset results |
| [09_COMPARISON_ANALYSIS.md](./09_COMPARISON_ANALYSIS.md) | Cross-dataset comparison |

### Part IV: Connection to Proposed Model
| Document | Description |
|----------|-------------|
| [10_MODEL_JUSTIFICATION.md](./10_MODEL_JUSTIFICATION.md) | **How Zipf's Law justifies the Pointer Network model** |

### Part V: Visual Learning
| Document | Description |
|----------|-------------|
| [11_EXAMPLES_AND_DIAGRAMS.md](./11_EXAMPLES_AND_DIAGRAMS.md) | Step-by-step examples with ASCII diagrams |

---

## 🎯 Quick Navigation by Use Case

### If you want to understand...

#### The Theory
→ Start with [02_ZIPF_LAW_THEORY.md](./02_ZIPF_LAW_THEORY.md)

#### The Code
→ Read [03_CODE_WALKTHROUGH.md](./03_CODE_WALKTHROUGH.md) and [04_DATA_PIPELINE.md](./04_DATA_PIPELINE.md)

#### The Results
→ See [05_RESULTS_ANALYSIS.md](./05_RESULTS_ANALYSIS.md) and [06_PLOT_INTERPRETATION.md](./06_PLOT_INTERPRETATION.md)

#### How to Justify Your Model
→ **[10_MODEL_JUSTIFICATION.md](./10_MODEL_JUSTIFICATION.md)** is essential reading

#### Examples with Diagrams
→ [11_EXAMPLES_AND_DIAGRAMS.md](./11_EXAMPLES_AND_DIAGRAMS.md)

---

## 📊 Generated Outputs

### Plots
| File | Description |
|------|-------------|
| `geolife_zipf_location_frequency.png` | Zipf plot for Geolife dataset |
| `diy_zipf_location_frequency.png` | Zipf plot for DIY dataset |
| `comparison_zipf_location_frequency.png` | Side-by-side comparison |

### Data Files
| File | Description |
|------|-------------|
| `*_zipf_data_stats.csv` | Group statistics (rank, P(L), SE) |
| `*_zipf_data_user_groups.csv` | User group assignments |
| `*_zipf_data.csv` | Detailed per-user probabilities |

---

## 📖 Reference

**González, M. C., Hidalgo, C. A., & Barabási, A.-L. (2008)**. Understanding individual human mobility patterns. *Nature*, 453(7196), 779-782. doi:10.1038/nature06958

---

## 🔗 Related Files

- **Analysis Scripts**: `../zipf_location_frequency_analysis.py`, `../compare_datasets.py`
- **Proposed Model**: `/workspace/next_loc_clean_v2/src/models/proposed/pgt.py`
- **Training Script**: `/workspace/next_loc_clean_v2/src/training/train_pgt.py`

---

*Documentation generated for PhD thesis reference*
*Last updated: January 2026*
