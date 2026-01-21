# DIY vs GeoLife Pointer Mechanism Analysis - Complete Documentation

## Documentation Index

This folder contains comprehensive documentation for the analysis of differential pointer mechanism impact on DIY and GeoLife datasets.

---

## Document Structure

| File | Content | Pages |
|------|---------|-------|
| `01_OVERVIEW.md` | Executive summary, key findings, navigation guide | ~5 |
| `02_METHODOLOGY.md` | Experimental design, data sources, model architecture, scripts | ~9 |
| `03_DESCRIPTIVE_ANALYSIS.md` | Dataset characteristics, statistics, comparisons | ~10 |
| `04_DIAGNOSTIC_ANALYSIS.md` | Model behavior, gate values, component performance | ~10 |
| `05_HYPOTHESIS_TESTING.md` | Experiments 1-5, causal proofs, evidence synthesis | ~12 |
| `06_FIGURES_INTERPRETATION.md` | Detailed guide to all figures with axis descriptions | ~14 |
| `07_CONCLUSIONS.md` | Root cause, implications, recommendations, future work | ~13 |

**Total Documentation:** ~73 pages of comprehensive analysis

---

## Quick Reference

### The Core Finding

**Why does pointer ablation cause 46.7% drop on GeoLife but only 8.3% on DIY?**

```
Root Cause: Vocabulary Size
     ↓
DIY: 1,713 targets → Generation 5.64% → High pointer reliance (0.787)
GeoLife: 315 targets → Generation 12.19% → Balanced use (0.627)
     ↓
DIY already pointer-dependent → Small relative change
GeoLife balanced → Removing pointer is disruptive
```

### Key Metrics Summary

| Metric | DIY | GeoLife | Significance |
|--------|-----|---------|--------------|
| Ablation Impact | 8.3% | 46.7% | The puzzle |
| Unique Targets | 1,713 | 315 | Root cause |
| Generation Acc | 5.64% | 12.19% | Key difference |
| Mean Gate | 0.787 | 0.627 | Adaptation |
| Target-in-History | 84.12% | 83.81% | NOT cause |

---

## Reading Guide

### For a Quick Understanding (15 min)
1. Read `01_OVERVIEW.md` completely
2. Read "Root Cause Identification" in `07_CONCLUSIONS.md`
3. View `fig_summary_root_cause.png` in `../results/`

### For Methodology Understanding (30 min)
1. `01_OVERVIEW.md`
2. `02_METHODOLOGY.md`
3. `07_CONCLUSIONS.md` → Recommendations section

### For Complete Understanding (2 hours)
Read all documents in order: 01 → 02 → 03 → 04 → 05 → 06 → 07

### For Figure Reference
Use `06_FIGURES_INTERPRETATION.md` as a reference when viewing figures in `../results/`

---

## Figure Inventory

All figures are in `../results/` folder in PNG and PDF formats.

### Main Figures
| Figure | Content |
|--------|---------|
| `fig1_target_in_history.*` | Copy applicability analysis |
| `fig2_repetition_patterns.*` | Repetition analysis |
| `fig3_vocabulary_user_patterns.*` | Vocabulary and user analysis |
| `fig4_radar_comparison.*` | Multi-dimensional comparison |
| `fig5_gate_analysis.*` | Gate behavior analysis |
| `fig6_ptr_vs_gen.*` | Component performance |
| `fig7_vocabulary_effect.*` | Vocabulary effect |

### Experiment Figures
| Figure | Content |
|--------|---------|
| `exp1_stratified_analysis.*` | Stratified by target-in-history |
| `exp2_ablation_simulation.*` | Gate configuration simulation |
| `exp3_generation_difficulty.*` | Generation difficulty analysis |
| `exp5_target_in_history_ablation.*` | Test set manipulation |
| `exp5_recency_effect.*` | Recency analysis |
| `fig_summary_root_cause.png` | Complete summary |

---

## Data Files

All result data is in `../results/` folder.

| File | Format | Content |
|------|--------|---------|
| `descriptive_analysis_results.*` | CSV, MD, JSON | Dataset statistics |
| `diagnostic_analysis_results.*` | CSV, MD | Model behavior |
| `diagnostic_summary.json` | JSON | Summary metrics |
| `hypothesis_testing_results.json` | JSON | Experiment results |
| `test_manipulation_results.json` | JSON | Manipulation results |
| `exp4_root_cause_synthesis.csv` | CSV | Evidence table |
| `final_summary.*` | MD, CSV | Final conclusions |

---

## Reproduction

### Requirements
- Python 3.x with PyTorch, NumPy, Pandas, Matplotlib
- Trained model checkpoints (see `02_METHODOLOGY.md`)
- Dataset files (see `02_METHODOLOGY.md`)

### Commands
```bash
cd /data/next_loc_clean_v2
conda activate mlenv

# Run all analyses
python scripts/diy_geolife_characteristic_v2/01_descriptive_analysis.py
python scripts/diy_geolife_characteristic_v2/02_diagnostic_analysis.py
python scripts/diy_geolife_characteristic_v2/03_hypothesis_testing.py
python scripts/diy_geolife_characteristic_v2/04_test_manipulation.py
```

---

## Citation

If using this analysis, please cite:
- The DIY and GeoLife datasets appropriately
- The PointerGeneratorTransformer model architecture
- This analysis documentation

---

## Contact

For questions about this documentation or analysis, refer to the main project repository.

---

*Documentation generated: January 2, 2026*
*Version: 1.0*
