# DIY vs GeoLife Pointer Mechanism Characteristic Analysis (V2)

This folder contains comprehensive scientific experiments analyzing why the pointer mechanism has different impact on DIY (8.3% drop) vs GeoLife (46.7% drop) datasets.

## Visualization Style

All visualizations follow the **Classic Scientific Publication Style** matching the reference images:
- White background with black axis box (all 4 sides)
- Inside tick marks on all sides
- No grid lines
- Simple colors: blue (DIY), red (GeoLife), green (Pointer), purple (Generation)
- Open markers: circles (DIY), squares (GeoLife)
- Hatched bars with white fill
- Step histograms for distributions
- Panel labels (a), (b), (c) for multi-panel figures

## Quick Summary

**Root Cause:** Vocabulary size (1,713 vs 315 unique targets) causes generation head performance difference (5.64% vs 12.19%), which determines the relative pointer dependency and thus the differential ablation impact.

## Experiment Scripts

| Script | Purpose |
|--------|---------|
| `01_descriptive_analysis.py` | Dataset characteristics comparison |
| `02_diagnostic_analysis.py` | Model behavior analysis (gate, components) |
| `03_hypothesis_testing.py` | Hypothesis testing and root cause synthesis |
| `04_test_manipulation.py` | Causal experiments with test set manipulation |
| `publication_style.py` | Shared visualization style configuration |

## How to Run

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv
cd /data/next_loc_clean_v2

# Run all experiments in order
python scripts/diy_geolife_characteristic_v2/01_descriptive_analysis.py
python scripts/diy_geolife_characteristic_v2/02_diagnostic_analysis.py
python scripts/diy_geolife_characteristic_v2/03_hypothesis_testing.py
python scripts/diy_geolife_characteristic_v2/04_test_manipulation.py
```

## Key Results

### Component Performance

| Component | DIY | GeoLife |
|-----------|-----|---------|
| Pointer Head Acc@1 | 56.53% | 51.63% |
| Generation Head Acc@1 | **5.64%** | **12.19%** |
| Combined Acc@1 | 56.58% | 51.40% |
| Mean Gate Value | 0.787 | 0.627 |

### Dataset Characteristics

| Metric | DIY | GeoLife |
|--------|-----|---------|
| Target-in-History Rate | 84.12% | 83.81% |
| Unique Target Locations | 1,713 | 315 |
| Top-10 Target Coverage | 41.75% | 67.13% |

## Results Directory

See `results/` folder for:
- All figures (PNG and PDF) with classic scientific style
- Summary tables (CSV)
- Full analysis reports (JSON, Markdown)

## Documentation

Full Nature Journal standard documentation available at:
`docs/diy_geolife_pointer_impact_analysis.md`

---

*Generated: January 2, 2026*  
*Style: Classic Scientific Publication (matching reference images)*
