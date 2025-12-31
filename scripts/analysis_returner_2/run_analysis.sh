#!/bin/bash
#
# Run Zipf Plot Analysis
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
echo "ZIPF PLOT - LOCATION VISIT FREQUENCY ANALYSIS"
echo "Reproducing González et al. (2008) Figure 2d"
echo "=========================================================================="

# Scenario 1: Default parameters (target n_L = 5, 10, 30, 50)
echo -e "\n[Scenario 1] Running with default parameters..."
python scripts/analysis_returner_2/zipf_location_frequency_analysis.py

# Scenario 2: Custom grouping (different n_L values)
# Uncomment to run:
# echo -e "\n[Scenario 2] Running with custom grouping..."
# python scripts/analysis_returner_2/zipf_location_frequency_analysis.py \
#     --target-n 5 15 25 40 \
#     --bin-widths 2 3 5 5 \
#     --output-dir scripts/analysis_returner_2/custom

# Scenario 3: Different fit range for reference line
# Uncomment to run:
# echo -e "\n[Scenario 3] Running with different fit range..."
# python scripts/analysis_returner_2/zipf_location_frequency_analysis.py \
#     --fit-range 2 15 \
#     --output-dir scripts/analysis_returner_2/alt_fit

# Create comparison plot
echo -e "\n[Comparison] Creating comparison plot..."
cd scripts/analysis_returner_2
python compare_datasets.py

echo -e "\n=========================================================================="
echo "ANALYSIS COMPLETE"
echo "=========================================================================="
echo "Results saved to: scripts/analysis_returner_2/"
echo ""
echo "Generated files:"
echo "  - geolife_zipf_location_frequency.png"
echo "  - geolife_zipf_data_stats.csv"
echo "  - geolife_zipf_data_user_groups.csv"
echo "  - geolife_zipf_data.csv"
echo "  - diy_zipf_location_frequency.png"
echo "  - diy_zipf_data_stats.csv"
echo "  - diy_zipf_data_user_groups.csv"
echo "  - diy_zipf_data.csv"
echo "  - comparison_zipf_location_frequency.png"
echo "=========================================================================="
