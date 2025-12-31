"""
Compare Return Probability Distributions
Plot both Geolife and DIY datasets on the same figure for comparison.

Author: Data Scientist
Date: 2025-12-31
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_comparison():
    """Create comparison plot with both datasets."""
    
    # Load data
    geolife_pdf = pd.read_csv('geolife_return_probability_data.csv')
    diy_pdf = pd.read_csv('diy_return_probability_data.csv')
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot both curves
    plt.plot(geolife_pdf['t_hours'], geolife_pdf['F_pt'], 
             'b--', linewidth=2, label='Geolife', alpha=0.8, marker='o', 
             markersize=3, markevery=5)
    plt.plot(diy_pdf['t_hours'], diy_pdf['F_pt'], 
             'r-', linewidth=2, label='DIY', alpha=0.8, marker='s', 
             markersize=3, markevery=5)
    
    # Styling
    plt.xlabel('t (h)', fontsize=12)
    plt.ylabel('F$_{pt}$(t)', fontsize=12)
    plt.title('Return Probability Distribution - Dataset Comparison', 
              fontsize=14, pad=15)
    
    # Set x-axis ticks at 24-hour intervals
    x_ticks = np.arange(0, 241, 24)
    plt.xticks(x_ticks)
    
    # Set axis limits
    plt.xlim(0, 240)
    plt.ylim(bottom=0)
    
    # Grid
    plt.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    # Legend
    plt.legend(loc='upper right', fontsize=11, framealpha=0.9)
    
    # Clean spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig('comparison_return_probability.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to: comparison_return_probability.png")
    
    plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    geolife_returns = pd.read_csv('geolife_return_probability_data_returns.csv')
    diy_returns = pd.read_csv('diy_return_probability_data_returns.csv')
    
    print(f"\n{'Metric':<30} {'Geolife':>15} {'DIY':>15}")
    print("-"*70)
    print(f"{'Users with returns':<30} {len(geolife_returns):>15,} {len(diy_returns):>15,}")
    print(f"{'Mean return time (h)':<30} {geolife_returns['delta_t_hours'].mean():>15.2f} {diy_returns['delta_t_hours'].mean():>15.2f}")
    print(f"{'Median return time (h)':<30} {geolife_returns['delta_t_hours'].median():>15.2f} {diy_returns['delta_t_hours'].median():>15.2f}")
    print(f"{'Std dev (h)':<30} {geolife_returns['delta_t_hours'].std():>15.2f} {diy_returns['delta_t_hours'].std():>15.2f}")
    print(f"{'Min return time (h)':<30} {geolife_returns['delta_t_hours'].min():>15.2f} {diy_returns['delta_t_hours'].min():>15.2f}")
    print(f"{'Max return time (h)':<30} {geolife_returns['delta_t_hours'].max():>15.2f} {diy_returns['delta_t_hours'].max():>15.2f}")
    print(f"{'Max F_pt(t)':<30} {geolife_pdf['F_pt'].max():>15.6f} {diy_pdf['F_pt'].max():>15.6f}")
    
    # Peak locations
    geolife_peak_t = geolife_pdf.loc[geolife_pdf['F_pt'].idxmax(), 't_hours']
    diy_peak_t = diy_pdf.loc[diy_pdf['F_pt'].idxmax(), 't_hours']
    print(f"{'Peak at t (h)':<30} {geolife_peak_t:>15.1f} {diy_peak_t:>15.1f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    plot_comparison()
