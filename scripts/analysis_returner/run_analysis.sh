#!/bin/bash
#
# Run Return Probability Analysis
# Demonstrates different usage scenarios
#
# Author: Data Scientist
# Date: 2025-12-31

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mlenv

# Navigate to project root
cd /data/next_loc_clean_v2

echo "=========================================================================="
echo "RETURN PROBABILITY DISTRIBUTION ANALYSIS"
echo "Reproducing González et al. (2008) Figure 2c"
echo "=========================================================================="

# Scenario 1: Default parameters (bin_width=2h, max_hours=240)
echo -e "\n[Scenario 1] Running with default parameters..."
python scripts/analysis_returner/return_probability_analysis.py

# Scenario 2: Higher resolution (bin_width=1h)
# Uncomment to run:
# echo -e "\n[Scenario 2] Running with 1-hour bins..."
# python scripts/analysis_returner/return_probability_analysis.py \
#     --bin-width 1.0 \
#     --max-hours 240 \
#     --output-dir scripts/analysis_returner/high_res

# Scenario 3: Shorter time window (0-120h)
# Uncomment to run:
# echo -e "\n[Scenario 3] Running with 120-hour window..."
# python scripts/analysis_returner/return_probability_analysis.py \
#     --bin-width 2.0 \
#     --max-hours 120 \
#     --output-dir scripts/analysis_returner/short_window

# Create comparison plot
echo -e "\n[Comparison] Creating comparison plot..."
cd scripts/analysis_returner
python compare_datasets.py

echo -e "\n=========================================================================="
echo "ANALYSIS COMPLETE"
echo "=========================================================================="
echo "Results saved to: scripts/analysis_returner/"
echo ""
echo "Generated files:"
echo "  - geolife_return_probability.png"
echo "  - geolife_return_probability_data.csv"
echo "  - geolife_return_probability_data_returns.csv"
echo "  - diy_return_probability.png"
echo "  - diy_return_probability_data.csv"
echo "  - diy_return_probability_data_returns.csv"
echo "  - comparison_return_probability.png"
echo "=========================================================================="
