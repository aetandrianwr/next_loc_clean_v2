# Return Probability Analysis V2 - Style Comparison

## Changes Made from V1 to V2

### Visual Styling Updates
1. **Added RW (Random Walk) baseline**: Black solid line showing exponential decay model
2. **Legend order**: Changed to "Users" first (dashed), then "RW" (solid)
3. **Removed title**: Cleaner appearance matching reference
4. **Increased font sizes**: 
   - Axis labels: 20pt (was 12pt)
   - Tick labels: 16pt (was default)
   - Legend: 18pt (was 11pt)
5. **Thicker borders**: 2.5pt frame (was thin with hidden top/right spines)
6. **Fixed Y-axis range**: 0 to 0.025 (was auto-scaled)
7. **Removed grid**: Clean background (was light grid)
8. **Enhanced legend**: Box with thick border, no transparency

### Technical Details
- **Blue dashed line**: #3366CC color, 2.5pt width, rounded dash caps
- **Black solid line**: Standard black, 2pt width
- **RW model**: Exponential decay with tau=30h
- **Output filenames**: `*_v2.png` to distinguish from V1

## Files Generated
- `geolife_return_probability_v2.png`
- `diy_return_probability_v2.png`
- `geolife_return_probability_data_v2.csv`
- `diy_return_probability_data_v2.csv`
- `*_returns.csv` files with individual user return times

## Script Location
`scripts/analysis_returner/return_probability_analysis_v2.py`

## Style Match
✅ Matches reference plot "Screenshot 2026-01-02 140753.png"
