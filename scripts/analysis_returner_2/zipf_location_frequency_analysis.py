"""
Zipf Plot of Location Visit Frequency
Reproduces Figure 2d from González et al. (2008)

Analyzes the frequency distribution of location visits following Zipf's law.
For each user, ranks locations by visit frequency and plots P(L) vs L on log-log scale,
grouped by the number of distinct locations visited (n_L = 5, 10, 30, 50).

Author: Data Scientist
Date: 2025-12-31
"""

import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def load_intermediate_data(dataset_path):
    """
    Load intermediate CSV data from preprocessing.
    
    Parameters
    ----------
    dataset_path : str
        Path to the intermediate CSV file
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, location_id
    """
    print(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # We only need user_id and location_id for visit counts
    df_visits = df[['user_id', 'location_id']].copy()
    
    print(f"Loaded {len(df_visits):,} visits from {df_visits['user_id'].nunique():,} users")
    print(f"Unique locations: {df_visits['location_id'].nunique():,}")
    
    return df_visits


def compute_user_location_frequencies(df_visits):
    """
    For each user, compute visit frequency per location and rank them.
    
    This function:
    1. Counts visits per location for each user
    2. Ranks locations by visit count (descending)
    3. Converts counts to probabilities p_u(L)
    
    Parameters
    ----------
    df_visits : pd.DataFrame
        DataFrame with columns: user_id, location_id
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, location_id, visit_count, rank, probability
    """
    print("\n" + "="*60)
    print("Computing location visit frequencies per user...")
    print("="*60)
    
    # Count visits per location per user
    location_counts = df_visits.groupby(['user_id', 'location_id']).size().reset_index(name='visit_count')
    
    # Sort by user and visit count (descending) to assign ranks
    location_counts = location_counts.sort_values(['user_id', 'visit_count'], 
                                                   ascending=[True, False])
    
    # Assign rank within each user (L = 1, 2, 3, ...)
    # rank=1 means most visited location
    location_counts['rank'] = location_counts.groupby('user_id').cumcount() + 1
    
    # Compute total visits per user
    user_totals = location_counts.groupby('user_id')['visit_count'].sum().reset_index()
    user_totals = user_totals.rename(columns={'visit_count': 'total_visits'})
    
    # Merge and compute probabilities
    location_counts = location_counts.merge(user_totals, on='user_id')
    location_counts['probability'] = location_counts['visit_count'] / location_counts['total_visits']
    
    # Compute number of unique locations per user
    n_locations = location_counts.groupby('user_id')['location_id'].nunique().reset_index()
    n_locations = n_locations.rename(columns={'location_id': 'n_unique_locations'})
    
    location_counts = location_counts.merge(n_locations, on='user_id')
    
    print(f"Processed {location_counts['user_id'].nunique():,} users")
    print(f"Total location-user pairs: {len(location_counts):,}")
    
    return location_counts


def assign_user_groups(location_counts, target_n_locations=[5, 10, 30, 50], 
                       bin_widths=[1, 2, 5, 5]):
    """
    Group users by number of unique locations visited.
    
    Uses binning approach: for target n_L, include users with 
    n_unique_locations in [n_L - bin_width, n_L + bin_width].
    
    Parameters
    ----------
    location_counts : pd.DataFrame
        Output from compute_user_location_frequencies
    target_n_locations : list
        Target number of locations [5, 10, 30, 50]
    bin_widths : list
        Bin width for each target (e.g., 5±1 means [4, 6])
        
    Returns
    -------
    dict
        Maps target_n -> DataFrame of users in that group
    """
    print("\n" + "="*60)
    print("Grouping users by number of distinct locations...")
    print("="*60)
    
    user_groups = {}
    
    # Get unique users with their n_unique_locations
    users_info = location_counts[['user_id', 'n_unique_locations']].drop_duplicates()
    
    for target_n, bin_width in zip(target_n_locations, bin_widths):
        # Define range
        min_n = target_n - bin_width
        max_n = target_n + bin_width
        
        # Filter users in this range
        users_in_group = users_info[
            (users_info['n_unique_locations'] >= min_n) & 
            (users_info['n_unique_locations'] <= max_n)
        ]['user_id'].values
        
        # Get all location data for these users
        group_data = location_counts[location_counts['user_id'].isin(users_in_group)].copy()
        
        user_groups[target_n] = group_data
        
        print(f"Group n_L={target_n} (range [{min_n}, {max_n}]): {len(users_in_group)} users")
        if len(users_in_group) > 0:
            actual_n = users_info[users_info['user_id'].isin(users_in_group)]['n_unique_locations']
            print(f"  Actual n_unique range: [{actual_n.min()}, {actual_n.max()}]")
            print(f"  Mean n_unique: {actual_n.mean():.1f}")
    
    return user_groups


def compute_group_statistics(user_groups):
    """
    For each group and each rank L, compute mean P(L) and standard error.
    
    P_G(L) = mean_{u in G}[p_u(L)]
    SE_G(L) = std_{u in G}[p_u(L)] / sqrt(|G|)
    
    Parameters
    ----------
    user_groups : dict
        Maps target_n -> DataFrame of users in that group
        
    Returns
    -------
    dict
        Maps target_n -> DataFrame with columns: rank, mean_prob, std_error, n_users
    """
    print("\n" + "="*60)
    print("Computing group statistics (mean and SE)...")
    print("="*60)
    
    group_stats = {}
    
    for target_n, group_data in user_groups.items():
        if len(group_data) == 0:
            print(f"Group n_L={target_n}: No data, skipping")
            continue
        
        # For each rank, compute mean and SE across users
        # Use pivot to get user x rank matrix of probabilities
        pivot = group_data.pivot_table(
            index='user_id', 
            columns='rank', 
            values='probability',
            aggfunc='first'  # Each user-rank pair should be unique
        )
        
        # Compute statistics across users (axis=0)
        mean_prob = pivot.mean(axis=0)
        std_prob = pivot.std(axis=0, ddof=1)  # Sample std dev
        n_users = pivot.count(axis=0)  # Number of users with this rank
        
        # Standard error = std / sqrt(n)
        std_error = std_prob / np.sqrt(n_users)
        
        # Combine into DataFrame
        stats_df = pd.DataFrame({
            'rank': mean_prob.index,
            'mean_prob': mean_prob.values,
            'std_error': std_error.values,
            'n_users': n_users.values
        })
        
        # Filter out ranks where we have too few users (optional)
        # stats_df = stats_df[stats_df['n_users'] >= 3]
        
        group_stats[target_n] = stats_df
        
        print(f"Group n_L={target_n}:")
        print(f"  Max rank: {stats_df['rank'].max()}")
        print(f"  Users per rank (range): [{stats_df['n_users'].min():.0f}, {stats_df['n_users'].max():.0f}]")
    
    return group_stats


def fit_reference_line(group_stats, fit_rank_range=(3, 10)):
    """
    Fit a reference line proportional to L^{-1}.
    
    We fit c * L^{-1} in log space using least squares on mid-rank data.
    log(P(L)) = log(c) - log(L)
    
    Parameters
    ----------
    group_stats : dict
        Output from compute_group_statistics
    fit_rank_range : tuple
        Range of ranks to use for fitting (avoid rank 1 which can be outlier)
        
    Returns
    -------
    float
        Fitted coefficient c
    """
    print("\n" + "="*60)
    print("Fitting reference line: c * L^{-1}...")
    print("="*60)
    
    # Collect all data from all groups for fitting
    all_ranks = []
    all_probs = []
    
    for target_n, stats_df in group_stats.items():
        # Filter to fit range
        fit_data = stats_df[
            (stats_df['rank'] >= fit_rank_range[0]) & 
            (stats_df['rank'] <= fit_rank_range[1])
        ]
        
        all_ranks.extend(fit_data['rank'].values)
        all_probs.extend(fit_data['mean_prob'].values)
    
    if len(all_ranks) == 0:
        print("Warning: No data in fit range, using default c=0.5")
        return 0.5
    
    all_ranks = np.array(all_ranks)
    all_probs = np.array(all_probs)
    
    # Fit in log space: log(P) = log(c) - log(L)
    log_L = np.log(all_ranks)
    log_P = np.log(all_probs)
    
    # Least squares: minimize sum((log(P) - (log(c) - log(L)))^2)
    # log(c) = mean(log(P) + log(L))
    log_c = np.mean(log_P + log_L)
    c = np.exp(log_c)
    
    print(f"Fitted coefficient: c = {c:.4f}")
    print(f"Fit using ranks {fit_rank_range[0]} to {fit_rank_range[1]}")
    print(f"Reference line: P(L) = {c:.4f} * L^(-1)")
    
    return c


def plot_zipf(group_stats, c_ref, dataset_name, output_path, max_rank=None):
    """
    Create the Zipf plot with main panel (log-log) and inset (linear).
    
    Parameters
    ----------
    group_stats : dict
        Maps target_n -> DataFrame with rank, mean_prob, std_error
    c_ref : float
        Coefficient for reference line c * L^{-1}
    dataset_name : str
        Name of dataset for title
    output_path : str
        Path to save the figure
    max_rank : int, optional
        Maximum rank to plot on main panel
    """
    # Define colors and markers for each group
    group_styles = {
        5:  {'color': 'blue',   'marker': 'o', 'label': '5 loc.'},
        10: {'color': 'green',  'marker': 's', 'label': '10 loc.'},
        30: {'color': 'red',    'marker': '^', 'label': '30 loc.'},
        50: {'color': 'purple', 'marker': 'D', 'label': '50 loc.'}
    }
    
    # Create figure with GridSpec for inset
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(3, 3, figure=fig)
    
    # Main panel (log-log)
    ax_main = fig.add_subplot(gs[:, :])
    
    # Plot each group
    for target_n in sorted(group_stats.keys()):
        stats_df = group_stats[target_n]
        
        if len(stats_df) == 0:
            continue
        
        style = group_styles.get(target_n, {'color': 'gray', 'marker': 'x', 'label': f'{target_n} loc.'})
        
        # Apply max_rank filter if specified
        if max_rank is not None:
            stats_df = stats_df[stats_df['rank'] <= max_rank]
        
        # Plot on log-log scale
        ax_main.loglog(
            stats_df['rank'], 
            stats_df['mean_prob'],
            marker=style['marker'],
            color=style['color'],
            label=style['label'],
            markersize=6,
            linewidth=1.5,
            alpha=0.8
        )
    
    # Add reference line: c * L^{-1}
    if max_rank is None:
        max_rank = max([stats_df['rank'].max() for stats_df in group_stats.values() if len(stats_df) > 0])
    
    L_ref = np.arange(1, max_rank + 1)
    P_ref = c_ref / L_ref
    ax_main.loglog(L_ref, P_ref, 'k--', linewidth=2, label=f'$L^{{-1}}$ (c={c_ref:.3f})', alpha=0.7)
    
    # Styling main panel
    ax_main.set_xlabel('L (rank)', fontsize=12)
    ax_main.set_ylabel('P(L)', fontsize=12)
    ax_main.set_title(f'Location Visit Frequency Distribution - {dataset_name}', fontsize=14, pad=15)
    ax_main.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_main.legend(loc='lower left', fontsize=10, framealpha=0.9)
    
    # Set reasonable axis limits
    ax_main.set_xlim(0.8, max_rank * 1.2)
    
    # Inset (linear scale for small ranks)
    ax_inset = fig.add_axes([0.55, 0.5, 0.33, 0.33])  # [left, bottom, width, height]
    
    inset_max_rank = 6  # Show ranks 1-6 in inset
    
    for target_n in sorted(group_stats.keys()):
        stats_df = group_stats[target_n]
        
        if len(stats_df) == 0:
            continue
        
        style = group_styles.get(target_n, {'color': 'gray', 'marker': 'x', 'label': f'{target_n} loc.'})
        
        # Filter to small ranks
        inset_data = stats_df[stats_df['rank'] <= inset_max_rank].copy()
        
        if len(inset_data) == 0:
            continue
        
        # Plot with error bars
        ax_inset.errorbar(
            inset_data['rank'],
            inset_data['mean_prob'],
            yerr=inset_data['std_error'],
            marker=style['marker'],
            color=style['color'],
            markersize=5,
            linewidth=1,
            capsize=3,
            alpha=0.8
        )
    
    # Styling inset
    ax_inset.set_xlabel('L', fontsize=9)
    ax_inset.set_ylabel('P(L)', fontsize=9)
    ax_inset.set_title('Top locations (linear scale)', fontsize=9)
    ax_inset.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    ax_inset.tick_params(labelsize=8)
    ax_inset.set_xlim(0.5, inset_max_rank + 0.5)
    ax_inset.set_ylim(bottom=0)
    
    # Set integer x-ticks on inset
    ax_inset.set_xticks(range(1, inset_max_rank + 1))
    
    # Tight layout
    plt.tight_layout()
    
    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved plot to: {output_path}")
    
    plt.close()


def save_results_data(group_stats, user_groups, output_path):
    """
    Save the computed data for reproducibility.
    
    Parameters
    ----------
    group_stats : dict
        Group statistics (rank, mean_prob, std_error)
    user_groups : dict
        User group assignments
    output_path : str
        Base path for output files
    """
    # Save group statistics (for plotting)
    stats_file = output_path.replace('.csv', '_stats.csv')
    all_stats = []
    for target_n, stats_df in group_stats.items():
        stats_df_copy = stats_df.copy()
        stats_df_copy['n_locations_group'] = target_n
        all_stats.append(stats_df_copy)
    
    if all_stats:
        all_stats_df = pd.concat(all_stats, ignore_index=True)
        all_stats_df = all_stats_df[['n_locations_group', 'rank', 'mean_prob', 'std_error', 'n_users']]
        all_stats_df.to_csv(stats_file, index=False)
        print(f"✓ Saved group statistics to: {stats_file}")
    
    # Save user-level data (which users in which groups)
    user_file = output_path.replace('.csv', '_user_groups.csv')
    all_user_data = []
    for target_n, group_data in user_groups.items():
        if len(group_data) == 0:
            continue
        user_summary = group_data.groupby('user_id').agg({
            'n_unique_locations': 'first',
            'visit_count': 'sum'
        }).reset_index()
        user_summary['n_locations_group'] = target_n
        all_user_data.append(user_summary)
    
    if all_user_data:
        all_user_df = pd.concat(all_user_data, ignore_index=True)
        all_user_df = all_user_df[['n_locations_group', 'user_id', 'n_unique_locations', 'visit_count']]
        all_user_df.to_csv(user_file, index=False)
        print(f"✓ Saved user groups to: {user_file}")
    
    # Save detailed per-user location probabilities
    detail_file = output_path
    all_details = []
    for target_n, group_data in user_groups.items():
        if len(group_data) == 0:
            continue
        detail_df = group_data[['user_id', 'location_id', 'rank', 'probability', 'n_unique_locations']].copy()
        detail_df['n_locations_group'] = target_n
        all_details.append(detail_df)
    
    if all_details:
        all_details_df = pd.concat(all_details, ignore_index=True)
        all_details_df = all_details_df[['n_locations_group', 'user_id', 'location_id', 
                                          'rank', 'probability', 'n_unique_locations']]
        all_details_df.to_csv(detail_file, index=False)
        print(f"✓ Saved detailed location probabilities to: {detail_file}")


def analyze_dataset(dataset_path, dataset_name, output_dir,
                    target_n_locations=[5, 10, 30, 50],
                    bin_widths=[1, 2, 5, 5],
                    fit_rank_range=(3, 10)):
    """
    Complete analysis pipeline for one dataset.
    
    Parameters
    ----------
    dataset_path : str
        Path to intermediate CSV file
    dataset_name : str
        Name of dataset (e.g., 'Geolife', 'DIY')
    output_dir : str
        Directory to save outputs
    target_n_locations : list
        Target number of locations for grouping
    bin_widths : list
        Bin width for each target
    fit_rank_range : tuple
        Range of ranks for fitting reference line
    """
    print("\n" + "="*80)
    print(f"ANALYZING: {dataset_name}")
    print("="*80)
    
    # Load data
    df_visits = load_intermediate_data(dataset_path)
    
    # Compute location frequencies per user
    location_counts = compute_user_location_frequencies(df_visits)
    
    # Group users by n_unique_locations
    user_groups = assign_user_groups(location_counts, target_n_locations, bin_widths)
    
    # Compute group statistics
    group_stats = compute_group_statistics(user_groups)
    
    # Fit reference line
    c_ref = fit_reference_line(group_stats, fit_rank_range)
    
    # Create plot
    plot_file = os.path.join(output_dir, f'{dataset_name.lower()}_zipf_location_frequency.png')
    plot_zipf(group_stats, c_ref, dataset_name, plot_file)
    
    # Save data
    data_file = os.path.join(output_dir, f'{dataset_name.lower()}_zipf_data.csv')
    save_results_data(group_stats, user_groups, data_file)
    
    print(f"\n✓ Analysis complete for {dataset_name}")
    
    return location_counts, user_groups, group_stats, c_ref


def main():
    parser = argparse.ArgumentParser(
        description='Compute Zipf plot of location visit frequency (González et al. 2008, Figure 2d)'
    )
    parser.add_argument(
        '--target-n',
        type=int,
        nargs='+',
        default=[5, 10, 30, 50],
        help='Target number of locations for grouping (default: 5 10 30 50)'
    )
    parser.add_argument(
        '--bin-widths',
        type=int,
        nargs='+',
        default=[1, 2, 5, 5],
        help='Bin widths for each target (default: 1 2 5 5)'
    )
    parser.add_argument(
        '--fit-range',
        type=int,
        nargs=2,
        default=[3, 10],
        help='Rank range for fitting reference line (default: 3 10)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts/analysis_returner_2',
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Dataset paths
    geolife_path = 'data/geolife_eps20/interim/intermediate_eps20.csv'
    diy_path = 'data/diy_eps50/interim/intermediate_eps50.csv'
    
    print("="*80)
    print("ZIPF PLOT - LOCATION VISIT FREQUENCY ANALYSIS")
    print("Reproducing González et al. (2008) Figure 2d")
    print("="*80)
    print(f"Target n_locations: {args.target_n}")
    print(f"Bin widths: {args.bin_widths}")
    print(f"Fit range: ranks {args.fit_range[0]} to {args.fit_range[1]}")
    print(f"Output directory: {output_dir}")
    
    # Analyze Geolife
    if os.path.exists(geolife_path):
        geolife_results = analyze_dataset(
            geolife_path, 'Geolife', output_dir,
            args.target_n, args.bin_widths, tuple(args.fit_range)
        )
    else:
        print(f"\n⚠ Geolife data not found at: {geolife_path}")
    
    # Analyze DIY
    if os.path.exists(diy_path):
        diy_results = analyze_dataset(
            diy_path, 'DIY', output_dir,
            args.target_n, args.bin_widths, tuple(args.fit_range)
        )
    else:
        print(f"\n⚠ DIY data not found at: {diy_path}")
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"All results saved to: {output_dir}/")
    print("="*80)


if __name__ == '__main__':
    main()
