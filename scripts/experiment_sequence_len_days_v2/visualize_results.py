#!/usr/bin/env python3
"""
Sequence Length Days Experiment - Visualization and Analysis (V2)

This script generates Nature Journal standard visualizations and statistical
analyses for the sequence length experiment.

Style Reference: Classic scientific publication style with:
- White background, black axis box (all 4 sides)
- Inside tick marks
- No grid lines
- Simple colors: black, blue, red, green
- Open markers: circles, squares, diamonds, triangles

Output:
1. Publication-quality figures (PDF, PNG, SVG)
2. LaTeX-formatted tables
3. Statistical analysis reports
4. CSV exports for further analysis

Author: PhD Research - Next Location Prediction
Date: 2026-01-02
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import seaborn as sns

# Set classic scientific publication style (matching reference images)
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    
    # Figure settings
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    
    # Axes settings - box style (all 4 sides visible)
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,  # No grid
    'axes.spines.top': True,  # Show all 4 sides
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    
    # Tick settings - inside ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.top': True,  # Ticks on all sides
    'xtick.bottom': True,
    'ytick.left': True,
    'ytick.right': True,
    
    # Line settings
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    
    # Legend settings
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})

# Classic scientific color palette (matching reference)
COLORS = {
    'diy': 'blue',          # Blue (solid)
    'geolife': 'red',       # Red
    'black': 'black',
    'green': 'green',
}

# Marker styles (open markers like in reference)
MARKERS = {
    'diy': 'o',        # Circle
    'geolife': 's',    # Square
}

METRIC_LABELS = {
    'acc@1': 'Accuracy@1 (%)',
    'acc@5': 'Accuracy@5 (%)',
    'acc@10': 'Accuracy@10 (%)',
    'mrr': 'MRR (%)',
    'ndcg': 'NDCG@10 (%)',
    'f1': 'F1 Score (%)',
    'loss': 'Cross-Entropy Loss',
}


def load_results(results_dir: str) -> Dict:
    """Load experiment results from JSON files."""
    results = {}
    
    for dataset in ['diy', 'geolife']:
        result_file = os.path.join(results_dir, f'{dataset}_sequence_length_results.json')
        if os.path.exists(result_file):
            with open(result_file, 'r') as f:
                results[dataset] = json.load(f)
    
    return results


def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert results dictionary to pandas DataFrame."""
    rows = []
    
    for dataset, data in results.items():
        dataset_name = data.get('dataset', dataset.upper())
        
        for prev_days, values in data['results'].items():
            row = {
                'dataset': dataset_name,
                'prev_days': int(prev_days),
                'num_samples': values['num_samples'],
                'avg_seq_len': values['avg_seq_len'],
                'std_seq_len': values['std_seq_len'],
                'max_seq_len': values['max_seq_len'],
            }
            # Add all metrics
            for metric, value in values['metrics'].items():
                if metric == 'f1':
                    row[metric] = value * 100  # Convert to percentage
                else:
                    row[metric] = value
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.sort_values(['dataset', 'prev_days'])
    return df


def setup_classic_axes(ax):
    """Configure axes to match classic scientific publication style."""
    # Ensure all spines visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Inside ticks on all sides
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='major', length=5, width=1)
    ax.tick_params(axis='both', which='minor', length=3, width=0.5)


def create_performance_comparison_plot(df: pd.DataFrame, output_dir: str):
    """Create multi-panel figure comparing metrics across sequence lengths."""
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    
    datasets = df['dataset'].unique()
    
    # Define line styles for each dataset
    line_styles = {'DIY': '-', 'GeoLife': '-'}
    
    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for dataset in datasets:
            data = df[df['dataset'] == dataset]
            color = COLORS.get(dataset.lower(), 'black')
            marker = MARKERS.get(dataset.lower(), 'o')
            linestyle = line_styles.get(dataset, '-')
            
            ax.plot(
                data['prev_days'], 
                data[metric],
                marker=marker,
                markersize=7,
                linewidth=1.5,
                label=dataset,
                color=color,
                linestyle=linestyle,
                markerfacecolor='white',  # Open markers
                markeredgecolor=color,
                markeredgewidth=1.5
            )
        
        ax.set_xlabel(r'$t$ (days)')
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        
        # Apply classic style
        setup_classic_axes(ax)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(
            os.path.join(output_dir, f'performance_comparison.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print(f"Saved performance comparison plot")


def create_accuracy_heatmap(df: pd.DataFrame, output_dir: str):
    """Create heatmap visualization of accuracy metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        
        # Prepare data for heatmap
        metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
        heatmap_data = data.set_index('prev_days')[metrics].T
        
        ax = axes[idx]
        # Use a simple grayscale colormap for classic look
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt='.1f',
            cmap='Greys',
            ax=ax,
            cbar_kws={'label': 'Performance (%)'},
            linewidths=0.5,
            linecolor='black'
        )
        
        ax.set_title(f'{dataset} Dataset')
        ax.set_xlabel(r'$t$ (days)')
        ax.set_ylabel('Metric')
        
        # Style adjustments
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(output_dir, f'accuracy_heatmap.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print(f"Saved accuracy heatmap")


def create_sequence_length_distribution(df: pd.DataFrame, output_dir: str):
    """Create bar chart showing sequence length statistics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        ax = axes[idx]
        
        x = data['prev_days']
        y = data['avg_seq_len']
        err = data['std_seq_len']
        
        color = COLORS.get(dataset.lower(), 'black')
        
        # Classic style bars with error bars
        bars = ax.bar(x, y, yerr=err, capsize=4, 
                      color='white', edgecolor=color, linewidth=1.5,
                      error_kw={'ecolor': 'black', 'capthick': 1, 'elinewidth': 1})
        
        ax.set_xlabel(r'$t$ (days)')
        ax.set_ylabel('Average Sequence Length')
        ax.set_title(f'{dataset}')
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # Apply classic style
        setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(output_dir, f'sequence_length_distribution.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print(f"Saved sequence length distribution plot")


def create_loss_curve(df: pd.DataFrame, output_dir: str):
    """Create loss curve across sequence lengths."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    line_styles = {'DIY': '-', 'GeoLife': '-'}
    
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset]
        color = COLORS.get(dataset.lower(), 'black')
        marker = MARKERS.get(dataset.lower(), 'o')
        linestyle = line_styles.get(dataset, '-')
        
        ax.plot(
            data['prev_days'],
            data['loss'],
            marker=marker,
            markersize=7,
            linewidth=1.5,
            label=dataset,
            color=color,
            linestyle=linestyle,
            markerfacecolor='white',
            markeredgecolor=color,
            markeredgewidth=1.5
        )
    
    ax.set_xlabel(r'$t$ (days)')
    ax.set_ylabel('Cross-Entropy Loss')
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    
    # Apply classic style
    setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(output_dir, f'loss_curve.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print(f"Saved loss curve plot")


def create_samples_vs_performance(df: pd.DataFrame, output_dir: str):
    """Create scatter plot of sample count vs performance."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    marker_list = ['o', 's', '^', 'D']
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        color = COLORS.get(dataset.lower(), 'black')
        marker = MARKERS.get(dataset.lower(), 'o')
        
        ax.scatter(
            data['num_samples'],
            data['acc@1'],
            s=80,
            label=dataset,
            facecolors='white',
            edgecolors=color,
            linewidths=1.5,
            marker=marker
        )
        
        # Add prev_days labels
        for _, row in data.iterrows():
            ax.annotate(
                f"{int(row['prev_days'])}",
                (row['num_samples'], row['acc@1']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=9
            )
    
    ax.set_xlabel('Number of Test Samples')
    ax.set_ylabel('Accuracy@1 (%)')
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    
    # Apply classic style
    setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(output_dir, f'samples_vs_performance.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print(f"Saved samples vs performance plot")


def create_radar_chart(df: pd.DataFrame, output_dir: str):
    """Create radar chart comparing prev1 vs prev7."""
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), subplot_kw=dict(polar=True))
    
    for idx, dataset in enumerate(df['dataset'].unique()):
        data = df[df['dataset'] == dataset]
        ax = axes[idx]
        
        # Get data for prev1 and prev7
        prev1 = data[data['prev_days'] == 1][metrics].values.flatten()
        prev7 = data[data['prev_days'] == 7][metrics].values.flatten()
        
        # Normalize to 0-100 range for visualization
        prev1_norm = prev1
        prev7_norm = prev7
        
        # Number of variables
        num_vars = len(metrics)
        
        # Compute angle for each axis
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # Complete the loop
        
        prev1_norm = np.concatenate((prev1_norm, [prev1_norm[0]]))
        prev7_norm = np.concatenate((prev7_norm, [prev7_norm[0]]))
        
        # Classic style: solid vs dashed lines, black and blue
        ax.plot(angles, prev1_norm, 'o--', linewidth=1.5, label='t=1', color='black',
                markerfacecolor='white', markeredgecolor='black', markersize=6)
        
        ax.plot(angles, prev7_norm, 's-', linewidth=1.5, label='t=7', color='blue',
                markerfacecolor='white', markeredgecolor='blue', markersize=6)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.upper() for m in metrics])
        ax.set_title(f'{dataset}', y=1.08)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), frameon=True, edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(output_dir, f'radar_comparison.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print(f"Saved radar chart")


def generate_latex_tables(df: pd.DataFrame, output_dir: str):
    """Generate LaTeX-formatted tables for publication."""
    
    # Main results table
    latex_content = []
    latex_content.append(r"\begin{table}[htbp]")
    latex_content.append(r"\centering")
    latex_content.append(r"\caption{Impact of Sequence Length on Next Location Prediction Performance}")
    latex_content.append(r"\label{tab:sequence_length_results}")
    latex_content.append(r"\begin{tabular}{llcccccc}")
    latex_content.append(r"\toprule")
    latex_content.append(r"Dataset & Prev Days & Acc@1 & Acc@5 & Acc@10 & MRR & NDCG & F1 \\")
    latex_content.append(r"\midrule")
    
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        for _, row in data.iterrows():
            line = f"{dataset} & {int(row['prev_days'])} & {row['acc@1']:.2f} & {row['acc@5']:.2f} & {row['acc@10']:.2f} & {row['mrr']:.2f} & {row['ndcg']:.2f} & {row['f1']:.2f} \\\\"
            latex_content.append(line)
        latex_content.append(r"\midrule")
    
    latex_content.append(r"\bottomrule")
    latex_content.append(r"\end{tabular}")
    latex_content.append(r"\end{table}")
    
    with open(os.path.join(output_dir, 'results_table.tex'), 'w') as f:
        f.write('\n'.join(latex_content))
    
    # Summary statistics table
    latex_summary = []
    latex_summary.append(r"\begin{table}[htbp]")
    latex_summary.append(r"\centering")
    latex_summary.append(r"\caption{Dataset Statistics by Sequence Length}")
    latex_summary.append(r"\label{tab:sequence_length_stats}")
    latex_summary.append(r"\begin{tabular}{llcccc}")
    latex_summary.append(r"\toprule")
    latex_summary.append(r"Dataset & Prev Days & Samples & Avg Len & Std Len & Max Len \\")
    latex_summary.append(r"\midrule")
    
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        for _, row in data.iterrows():
            line = f"{dataset} & {int(row['prev_days'])} & {int(row['num_samples'])} & {row['avg_seq_len']:.1f} & {row['std_seq_len']:.1f} & {int(row['max_seq_len'])} \\\\"
            latex_summary.append(line)
        latex_summary.append(r"\midrule")
    
    latex_summary.append(r"\bottomrule")
    latex_summary.append(r"\end{tabular}")
    latex_summary.append(r"\end{table}")
    
    with open(os.path.join(output_dir, 'statistics_table.tex'), 'w') as f:
        f.write('\n'.join(latex_summary))
    
    print("Saved LaTeX tables")


def generate_csv_exports(df: pd.DataFrame, output_dir: str):
    """Export results to CSV for further analysis."""
    # Full results
    df.to_csv(os.path.join(output_dir, 'full_results.csv'), index=False)
    
    # Summary by dataset
    summary = df.groupby('dataset').agg({
        'acc@1': ['min', 'max', 'mean', 'std'],
        'acc@5': ['min', 'max', 'mean', 'std'],
        'acc@10': ['min', 'max', 'mean', 'std'],
        'mrr': ['min', 'max', 'mean', 'std'],
        'ndcg': ['min', 'max', 'mean', 'std'],
        'num_samples': ['sum', 'mean'],
    })
    summary.to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    print("Saved CSV exports")


def compute_improvement_analysis(df: pd.DataFrame, output_dir: str):
    """Compute performance improvement analysis."""
    analysis = []
    
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        
        prev1 = data[data['prev_days'] == 1].iloc[0]
        prev7 = data[data['prev_days'] == 7].iloc[0]
        
        metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
        
        for metric in metrics:
            abs_improvement = prev7[metric] - prev1[metric]
            rel_improvement = (prev7[metric] - prev1[metric]) / prev1[metric] * 100
            
            analysis.append({
                'dataset': dataset,
                'metric': metric,
                'prev1_value': prev1[metric],
                'prev7_value': prev7[metric],
                'absolute_improvement': abs_improvement,
                'relative_improvement_pct': rel_improvement,
            })
    
    analysis_df = pd.DataFrame(analysis)
    analysis_df.to_csv(os.path.join(output_dir, 'improvement_analysis.csv'), index=False)
    
    # Create improvement summary
    print("\n" + "="*80)
    print("IMPROVEMENT ANALYSIS: prev1 → prev7")
    print("="*80)
    
    for dataset in df['dataset'].unique():
        ds_analysis = analysis_df[analysis_df['dataset'] == dataset]
        print(f"\n{dataset}:")
        for _, row in ds_analysis.iterrows():
            print(f"  {row['metric']:10s}: {row['prev1_value']:6.2f} → {row['prev7_value']:6.2f} "
                  f"(+{row['absolute_improvement']:5.2f}, +{row['relative_improvement_pct']:5.1f}%)")
    
    return analysis_df


def create_improvement_bar_chart(df: pd.DataFrame, output_dir: str):
    """Create bar chart showing relative improvement from prev1 to prev7."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    
    improvements = {}
    for dataset in df['dataset'].unique():
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        prev1 = data[data['prev_days'] == 1][metrics].values.flatten()
        prev7 = data[data['prev_days'] == 7][metrics].values.flatten()
        rel_imp = (prev7 - prev1) / prev1 * 100
        improvements[dataset] = rel_imp
    
    datasets = list(improvements.keys())
    hatches = ['///', '...']  # Classic hatch patterns
    
    for idx, dataset in enumerate(datasets):
        offset = width * (idx - 0.5)
        color = COLORS.get(dataset.lower(), 'black')
        bars = ax.bar(x + offset, improvements[dataset], width, label=dataset, 
                      color='white', edgecolor=color, linewidth=1.5,
                      hatch=hatches[idx % len(hatches)])
    
    ax.set_ylabel('Relative Improvement (%)')
    ax.set_xlabel('Metric')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in metrics])
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    
    # Apply classic style
    setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        fig.savefig(
            os.path.join(output_dir, f'improvement_comparison.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print("Saved improvement comparison chart")


def create_combined_figure(df: pd.DataFrame, output_dir: str):
    """Create a single comprehensive figure for publication."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    datasets = df['dataset'].unique()
    line_styles = {'DIY': '-', 'GeoLife': '-'}
    
    # Helper function to plot with classic style
    def plot_metric(ax, metric, ylabel, panel_label):
        for dataset in datasets:
            data = df[df['dataset'] == dataset]
            color = COLORS.get(dataset.lower(), 'black')
            marker = MARKERS.get(dataset.lower(), 'o')
            linestyle = line_styles.get(dataset, '-')
            ax.plot(data['prev_days'], data[metric], marker=marker, linewidth=1.5, 
                    label=dataset, color=color, markersize=6, linestyle=linestyle,
                    markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
        ax.set_xlabel(r'$t$ (days)')
        ax.set_ylabel(ylabel)
        ax.text(0.02, 0.98, panel_label, transform=ax.transAxes, fontsize=12, 
                fontweight='bold', va='top', ha='left')
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        setup_classic_axes(ax)
    
    # (a) Acc@1 comparison
    ax1 = fig.add_subplot(gs[0, 0])
    plot_metric(ax1, 'acc@1', 'Accuracy@1 (%)', '(a)')
    
    # (b) Acc@5 comparison
    ax2 = fig.add_subplot(gs[0, 1])
    plot_metric(ax2, 'acc@5', 'Accuracy@5 (%)', '(b)')
    
    # (c) MRR comparison
    ax3 = fig.add_subplot(gs[0, 2])
    plot_metric(ax3, 'mrr', 'MRR (%)', '(c)')
    
    # (d) NDCG comparison
    ax4 = fig.add_subplot(gs[1, 0])
    plot_metric(ax4, 'ndcg', 'NDCG@10 (%)', '(d)')
    
    # (e) Loss comparison
    ax5 = fig.add_subplot(gs[1, 1])
    plot_metric(ax5, 'loss', 'Cross-Entropy Loss', '(e)')
    
    # (f) Sequence length distribution
    ax6 = fig.add_subplot(gs[1, 2])
    x = np.arange(7)
    width = 0.35
    hatches = ['///', '...']
    for idx, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        color = COLORS.get(dataset.lower(), 'black')
        offset = width * (idx - 0.5)
        ax6.bar(x + offset + 1, data['avg_seq_len'], width, label=dataset, 
                color='white', edgecolor=color, linewidth=1.5,
                yerr=data['std_seq_len'], capsize=2, hatch=hatches[idx],
                error_kw={'ecolor': 'black', 'capthick': 1, 'elinewidth': 1})
    ax6.set_xlabel(r'$t$ (days)')
    ax6.set_ylabel('Avg Sequence Length')
    ax6.text(0.02, 0.98, '(f)', transform=ax6.transAxes, fontsize=12, 
             fontweight='bold', va='top', ha='left')
    ax6.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax6.set_xticks(range(1, 8))
    setup_classic_axes(ax6)
    
    # (g) Improvement analysis
    ax7 = fig.add_subplot(gs[2, :2])
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    x = np.arange(len(metrics))
    width = 0.35
    hatches = ['///', '...']
    
    for idx, dataset in enumerate(datasets):
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        prev1 = data[data['prev_days'] == 1][metrics].values.flatten()
        prev7 = data[data['prev_days'] == 7][metrics].values.flatten()
        rel_imp = (prev7 - prev1) / prev1 * 100
        
        color = COLORS.get(dataset.lower(), 'black')
        offset = width * (idx - 0.5)
        ax7.bar(x + offset, rel_imp, width, label=dataset, 
                color='white', edgecolor=color, linewidth=1.5, hatch=hatches[idx])
    
    ax7.set_xlabel('Metric')
    ax7.set_ylabel('Relative Improvement (%)')
    ax7.text(0.02, 0.98, '(g)', transform=ax7.transAxes, fontsize=12, 
             fontweight='bold', va='top', ha='left')
    ax7.set_xticks(x)
    ax7.set_xticklabels([m.upper() for m in metrics])
    ax7.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax7.axhline(y=0, color='black', linestyle='-', linewidth=1.0)
    setup_classic_axes(ax7)
    
    # (h) Sample size
    ax8 = fig.add_subplot(gs[2, 2])
    for dataset in datasets:
        data = df[df['dataset'] == dataset].sort_values('prev_days')
        color = COLORS.get(dataset.lower(), 'black')
        marker = MARKERS.get(dataset.lower(), 'o')
        linestyle = line_styles.get(dataset, '-')
        ax8.plot(data['prev_days'], data['num_samples'], marker=marker, linewidth=1.5,
                 label=dataset, color=color, markersize=6, linestyle=linestyle,
                 markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
    ax8.set_xlabel(r'$t$ (days)')
    ax8.set_ylabel('Number of Samples')
    ax8.text(0.02, 0.98, '(h)', transform=ax8.transAxes, fontsize=12, 
             fontweight='bold', va='top', ha='left')
    ax8.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    ax8.xaxis.set_major_locator(MaxNLocator(integer=True))
    setup_classic_axes(ax8)
    
    # Save
    for fmt in ['pdf', 'png', 'svg']:
        fig.savefig(
            os.path.join(output_dir, f'combined_figure.{fmt}'),
            format=fmt,
            dpi=300,
            bbox_inches='tight',
            facecolor='white'
        )
    
    plt.close()
    print("Saved combined publication figure")


def main():
    parser = argparse.ArgumentParser(
        description="Generate visualizations and analysis for sequence length experiment"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_sequence_len_days_v2/results',
        help='Directory containing experiment results'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_sequence_len_days_v2/results',
        help='Output directory for visualizations'
    )
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading experiment results...")
    results = load_results(args.results_dir)
    
    if not results:
        print("ERROR: No results found. Run evaluate_sequence_length.py first.")
        return
    
    # Convert to DataFrame
    df = results_to_dataframe(results)
    print(f"Loaded {len(df)} data points")
    print(df.to_string())
    
    # Generate all visualizations
    print("\nGenerating visualizations...")
    
    create_performance_comparison_plot(df, args.output_dir)
    create_accuracy_heatmap(df, args.output_dir)
    create_sequence_length_distribution(df, args.output_dir)
    create_loss_curve(df, args.output_dir)
    create_samples_vs_performance(df, args.output_dir)
    create_radar_chart(df, args.output_dir)
    create_improvement_bar_chart(df, args.output_dir)
    create_combined_figure(df, args.output_dir)
    
    # Generate tables and exports
    print("\nGenerating tables and exports...")
    generate_latex_tables(df, args.output_dir)
    generate_csv_exports(df, args.output_dir)
    
    # Compute improvement analysis
    compute_improvement_analysis(df, args.output_dir)
    
    print("\n" + "="*80)
    print("VISUALIZATION COMPLETE")
    print(f"All outputs saved to: {args.output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
