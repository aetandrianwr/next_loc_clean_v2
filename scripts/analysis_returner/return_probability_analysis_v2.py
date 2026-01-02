"""
Return Probability Distribution Analysis
Reproduces Figure 2c from González et al. (2008)

Computes the first-return time distribution F_pt(t) for users.
For each user, finds their first observed location L0 at time t0,
then finds the first later event where the user returns to L0 (time t1 > t0).
The first-return time is Δt = (t1 - t0) in hours.

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
        DataFrame with columns: user_id, location_id, timestamp (in minutes)
    """
    print(f"Loading data from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Compute timestamp in minutes from start_day and start_min
    # timestamp = start_day * 1440 (minutes per day) + start_min
    df['timestamp_minutes'] = df['start_day'] * 1440 + df['start_min']
    
    # Convert to hours for easier interpretation
    df['timestamp_hours'] = df['timestamp_minutes'] / 60.0
    
    print(f"Loaded {len(df):,} events from {df['user_id'].nunique():,} users")
    print(f"Time range: {df['timestamp_hours'].min():.2f}h to {df['timestamp_hours'].max():.2f}h")
    print(f"Unique locations: {df['location_id'].nunique():,}")
    
    return df[['user_id', 'location_id', 'timestamp_hours']].copy()


def compute_first_return_times(df, bin_width_hours=2.0, max_hours=240):
    """
    Compute first-return time distribution for all users.
    
    For each user:
    - Identify first location L0 at time t0
    - Find first later event where location == L0 (time t1 > t0)
    - Record Δt = t1 - t0 in hours
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns: user_id, location_id, timestamp_hours
    bin_width_hours : float
        Width of histogram bins in hours
    max_hours : int
        Maximum return time to consider in hours
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: user_id, first_location, first_time, 
        return_time, delta_t_hours
    """
    print("\n" + "="*60)
    print("Computing first-return times...")
    print("="*60)
    
    # Sort by user_id and timestamp
    df_sorted = df.sort_values(['user_id', 'timestamp_hours']).reset_index(drop=True)
    
    # Group by user and get first location for each user
    first_events = df_sorted.groupby('user_id').first().reset_index()
    first_events = first_events.rename(columns={
        'location_id': 'first_location',
        'timestamp_hours': 'first_time'
    })
    
    print(f"Total users: {len(first_events):,}")
    
    # Merge to get first location info for all events
    df_with_first = df_sorted.merge(
        first_events[['user_id', 'first_location', 'first_time']],
        on='user_id',
        how='left'
    )
    
    # Filter to events after first event (timestamp > first_time)
    df_later = df_with_first[df_with_first['timestamp_hours'] > df_with_first['first_time']].copy()
    
    # Filter to returns (location == first_location)
    df_returns = df_later[df_later['location_id'] == df_later['first_location']].copy()
    
    # Get first return for each user (earliest timestamp after first event)
    first_returns = df_returns.groupby('user_id').first().reset_index()
    
    # Compute delta_t
    first_returns['delta_t_hours'] = first_returns['timestamp_hours'] - first_returns['first_time']
    
    # Filter to max_hours
    first_returns = first_returns[first_returns['delta_t_hours'] <= max_hours].copy()
    
    print(f"Users with returns: {len(first_returns):,}")
    print(f"Return rate: {len(first_returns) / len(first_events) * 100:.2f}%")
    print(f"Mean return time: {first_returns['delta_t_hours'].mean():.2f}h")
    print(f"Median return time: {first_returns['delta_t_hours'].median():.2f}h")
    print(f"Min return time: {first_returns['delta_t_hours'].min():.2f}h")
    print(f"Max return time: {first_returns['delta_t_hours'].max():.2f}h")
    
    return first_returns


def compute_probability_density(delta_t_values, bin_width_hours=2.0, max_hours=240):
    """
    Convert return times to probability density F_pt(t).
    
    Parameters
    ----------
    delta_t_values : np.array
        Array of first-return times in hours
    bin_width_hours : float
        Width of histogram bins in hours
    max_hours : int
        Maximum time in hours
        
    Returns
    -------
    bin_centers : np.array
        Centers of histogram bins (x-axis)
    pdf : np.array
        Probability density values (y-axis)
    """
    # Create bins
    bins = np.arange(0, max_hours + bin_width_hours, bin_width_hours)
    
    # Compute histogram
    counts, bin_edges = np.histogram(delta_t_values, bins=bins)
    
    # Convert to probability density: count / (N_returns * bin_width)
    n_returns = len(delta_t_values)
    pdf = counts / (n_returns * bin_width_hours)
    
    # Bin centers for plotting
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    
    return bin_centers, pdf


def compute_random_walk_baseline(bin_centers, initial_prob=0.01):
    """
    Compute a simple exponential decay baseline (RW - Random Walk model).
    
    Parameters
    ----------
    bin_centers : np.array
        Time bins in hours
    initial_prob : float
        Initial probability value
        
    Returns
    -------
    np.array
        Random walk baseline probability values
    """
    # Simple exponential decay: P(t) = P0 * exp(-t/tau)
    tau = 30.0  # decay constant in hours
    rw = initial_prob * np.exp(-bin_centers / tau)
    return rw


def plot_return_probability(bin_centers, pdf, dataset_name, output_path, 
                            max_hours=240, bin_width_hours=2.0):
    """
    Create the return probability distribution plot matching reference style.
    
    Parameters
    ----------
    bin_centers : np.array
        x-axis values (time in hours)
    pdf : np.array
        y-axis values (probability density)
    dataset_name : str
        Name of dataset for title
    output_path : str
        Path to save the figure
    max_hours : int
        Maximum x-axis value
    bin_width_hours : float
        Bin width used
    """
    # Create figure with exact reference styling
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Compute RW baseline
    rw_baseline = compute_random_walk_baseline(bin_centers)
    
    # Plot Users data (blue dashed line) - plot first for legend order
    ax.plot(bin_centers, pdf, color='#3366CC', linestyle='--', 
            linewidth=2.5, label='Users', dash_capstyle='round')

#     ax.step(bin_centers, pdf, where='mid',
#             color='#3366CC', linestyle='--',
#             linewidth=2.5, label='Users')

    
    # Plot RW baseline (black solid line)
    ax.plot(bin_centers, rw_baseline, 'k-', linewidth=2, label='RW')
    
    # Axis labels with larger font
    ax.set_xlabel('$t$ (h)', fontsize=20, labelpad=8)
    ax.set_ylabel('$F_{pt}(t)$', fontsize=20, labelpad=8)
    
    # Set x-axis ticks at 24-hour intervals
    x_ticks = np.arange(0, max_hours + 1, 24)
    ax.set_xticks(x_ticks)
    ax.tick_params(axis='both', which='major', labelsize=16, 
                   width=1.5, length=6)
    
    # Set axis limits with extra padding at top to prevent cutoff
    ax.set_xlim(0, max_hours)
    ax.set_ylim(0, 0.026)
    
    # No grid
    ax.grid(False)
    
    # Legend with custom styling
    legend = ax.legend(loc='upper right', fontsize=18, frameon=True, 
                      framealpha=1.0, edgecolor='black', fancybox=False)
    legend.get_frame().set_linewidth(1.5)
    
    # Thicker spines (all visible)
    for spine in ax.spines.values():
        spine.set_linewidth(2.5)
        spine.set_edgecolor('black')
    
    # White background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Tight layout with padding to prevent cutoff
    plt.tight_layout(pad=1.5)
    
    # Save with proper padding
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', pad_inches=0.1)
    print(f"\n✓ Saved plot to: {output_path}")
    
    plt.close()


def save_results_data(first_returns, bin_centers, pdf, output_path):
    """
    Save the computed data for reproducibility.
    
    Parameters
    ----------
    first_returns : pd.DataFrame
        DataFrame with first-return times per user
    bin_centers : np.array
        Histogram bin centers
    pdf : np.array
        Probability density values
    output_path : str
        Path to save the CSV file
    """
    # Save individual return times
    returns_file = output_path.replace('.csv', '_returns.csv')
    first_returns[['user_id', 'delta_t_hours']].to_csv(returns_file, index=False)
    print(f"✓ Saved return times to: {returns_file}")
    
    # Save probability density data
    pdf_data = pd.DataFrame({
        't_hours': bin_centers,
        'F_pt': pdf
    })
    pdf_data.to_csv(output_path, index=False)
    print(f"✓ Saved probability density to: {output_path}")


def analyze_dataset(dataset_path, dataset_name, output_dir, 
                    bin_width_hours=2.0, max_hours=240):
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
    bin_width_hours : float
        Histogram bin width in hours
    max_hours : int
        Maximum return time to analyze
    """
    print("\n" + "="*80)
    print(f"ANALYZING: {dataset_name}")
    print("="*80)
    
    # Load data
    df = load_intermediate_data(dataset_path)
    
    # Compute first-return times
    first_returns = compute_first_return_times(df, bin_width_hours, max_hours)
    
    # Compute probability density
    bin_centers, pdf = compute_probability_density(
        first_returns['delta_t_hours'].values,
        bin_width_hours,
        max_hours
    )
    
    # Plot
    plot_file = os.path.join(output_dir, f'{dataset_name.lower()}_return_probability_v2.png')
    plot_return_probability(bin_centers, pdf, dataset_name, plot_file, 
                           max_hours, bin_width_hours)
    
    # Save data
    data_file = os.path.join(output_dir, f'{dataset_name.lower()}_return_probability_data_v2.csv')
    save_results_data(first_returns, bin_centers, pdf, data_file)
    
    print(f"\n✓ Analysis complete for {dataset_name}")
    
    return first_returns, bin_centers, pdf


def main():
    parser = argparse.ArgumentParser(
        description='Compute return probability distribution (González et al. 2008, Figure 2c)'
    )
    parser.add_argument(
        '--bin-width',
        type=float,
        default=2.0,
        help='Histogram bin width in hours (default: 2.0)'
    )
    parser.add_argument(
        '--max-hours',
        type=int,
        default=240,
        help='Maximum return time in hours (default: 240)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='scripts/analysis_returner',
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
    print("RETURN PROBABILITY DISTRIBUTION ANALYSIS")
    print("Reproducing González et al. (2008) Figure 2c")
    print("="*80)
    print(f"Bin width: {args.bin_width} hours")
    print(f"Max time: {args.max_hours} hours")
    print(f"Output directory: {output_dir}")
    
    # Analyze Geolife
    if os.path.exists(geolife_path):
        geolife_results = analyze_dataset(
            geolife_path, 'Geolife', output_dir,
            args.bin_width, args.max_hours
        )
    else:
        print(f"\n⚠ Geolife data not found at: {geolife_path}")
    
    # Analyze DIY
    if os.path.exists(diy_path):
        diy_results = analyze_dataset(
            diy_path, 'DIY', output_dir,
            args.bin_width, args.max_hours
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
