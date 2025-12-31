"""
Compare Zipf Plots
Create side-by-side comparison of Geolife and DIY datasets.

Author: Data Scientist
Date: 2025-12-31
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def plot_comparison():
    """Create comparison plot with both datasets."""
    
    # Load data
    geolife_stats = pd.read_csv('geolife_zipf_data_stats.csv')
    diy_stats = pd.read_csv('diy_zipf_data_stats.csv')
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Define styles
    group_styles = {
        5:  {'color': 'blue',   'marker': 'o', 'label': '5 loc.'},
        10: {'color': 'green',  'marker': 's', 'label': '10 loc.'},
        30: {'color': 'red',    'marker': '^', 'label': '30 loc.'},
        50: {'color': 'purple', 'marker': 'D', 'label': '50 loc.'}
    }
    
    # Plot Geolife
    for target_n in sorted(geolife_stats['n_locations_group'].unique()):
        data = geolife_stats[geolife_stats['n_locations_group'] == target_n]
        style = group_styles[target_n]
        ax1.loglog(data['rank'], data['mean_prob'],
                   marker=style['marker'], color=style['color'],
                   label=style['label'], markersize=6, linewidth=1.5, alpha=0.8)
    
    # Add reference line for Geolife
    max_rank_geo = geolife_stats['rank'].max()
    L_ref = np.arange(1, max_rank_geo + 1)
    P_ref = 0.222 / L_ref  # Fitted coefficient
    ax1.loglog(L_ref, P_ref, 'k--', linewidth=2, label='$L^{-1}$ (c=0.222)', alpha=0.7)
    
    ax1.set_xlabel('L (rank)', fontsize=12)
    ax1.set_ylabel('P(L)', fontsize=12)
    ax1.set_title('Geolife Dataset', fontsize=14, pad=15)
    ax1.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax1.legend(loc='lower left', fontsize=10)
    
    # Plot DIY
    for target_n in sorted(diy_stats['n_locations_group'].unique()):
        data = diy_stats[diy_stats['n_locations_group'] == target_n]
        style = group_styles[target_n]
        ax2.loglog(data['rank'], data['mean_prob'],
                   marker=style['marker'], color=style['color'],
                   label=style['label'], markersize=6, linewidth=1.5, alpha=0.8)
    
    # Add reference line for DIY
    max_rank_diy = diy_stats['rank'].max()
    L_ref = np.arange(1, max_rank_diy + 1)
    P_ref = 0.150 / L_ref  # Fitted coefficient
    ax2.loglog(L_ref, P_ref, 'k--', linewidth=2, label='$L^{-1}$ (c=0.150)', alpha=0.7)
    
    ax2.set_xlabel('L (rank)', fontsize=12)
    ax2.set_ylabel('P(L)', fontsize=12)
    ax2.set_title('DIY Dataset', fontsize=14, pad=15)
    ax2.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax2.legend(loc='lower left', fontsize=10)
    
    # Overall title
    fig.suptitle('Zipf Plot Comparison: Location Visit Frequency', 
                 fontsize=16, y=1.02)
    
    plt.tight_layout()
    plt.savefig('comparison_zipf_location_frequency.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to: comparison_zipf_location_frequency.png")
    plt.close()
    
    # Print statistics
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    geo_users = pd.read_csv('geolife_zipf_data_user_groups.csv')
    diy_users = pd.read_csv('diy_zipf_data_user_groups.csv')
    
    print(f"\n{'Group':<10} {'Geolife Users':>15} {'DIY Users':>15} {'Difference':>15}")
    print("-"*70)
    for n in [5, 10, 30, 50]:
        geo_count = len(geo_users[geo_users['n_locations_group'] == n])
        diy_count = len(diy_users[diy_users['n_locations_group'] == n])
        print(f"{n} loc.    {geo_count:>15,} {diy_count:>15,} {diy_count - geo_count:>15,}")
    
    print(f"\n{'Group':<10} {'Geolife P(1)':>15} {'DIY P(1)':>15} {'Difference':>15}")
    print("-"*70)
    for n in [5, 10, 30, 50]:
        geo_p1 = geolife_stats[(geolife_stats['n_locations_group'] == n) & 
                                (geolife_stats['rank'] == 1)]['mean_prob'].values
        diy_p1 = diy_stats[(diy_stats['n_locations_group'] == n) & 
                           (diy_stats['rank'] == 1)]['mean_prob'].values
        if len(geo_p1) > 0 and len(diy_p1) > 0:
            print(f"{n} loc.    {geo_p1[0]:>15.4f} {diy_p1[0]:>15.4f} {diy_p1[0] - geo_p1[0]:>15.4f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    plot_comparison()
