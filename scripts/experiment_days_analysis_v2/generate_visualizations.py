#!/usr/bin/env python
"""
Visualization Script for Day-of-Week Analysis Experiment (V2).

This script generates publication-quality figures for the day-of-week
analysis experiment. Designed for Nature Journal standards.

Style Reference: Classic scientific publication style with:
- White background, black axis box (all 4 sides)
- Inside tick marks
- No grid lines
- Simple colors: black, blue, red, green
- Open markers: circles, squares, diamonds, triangles

Visualizations:
1. Performance by Day of Week (bar chart)
2. Weekday vs Weekend Comparison (grouped bar)
3. All Metrics Heatmap
4. Performance Pattern Line Plot
5. Statistical Summary Table
6. Combined Publication Figure

Author: Experiment Script for PhD Thesis
Date: 2026-01-02
"""

import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, AutoMinorLocator
import seaborn as sns
from scipy import stats

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
    'weekday': 'green',        # Blue for weekday
    'weekend': 'orange',         # Red for weekend
    'diy': 'blue',            # Blue for DIY dataset
    'geolife': 'red',         # Red for Geolife dataset
    'black': 'black',
    'green': 'green',
}

# Marker styles (open markers like in reference)
MARKERS = {
    'diy': 'o',        # Circle
    'geolife': 's',    # Square
}

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_ABBREV = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

METRIC_LABELS = {
    'acc@1': 'Accuracy@1 (%)',
    'acc@5': 'Accuracy@5 (%)',
    'acc@10': 'Accuracy@10 (%)',
    'mrr': 'MRR (%)',
    'ndcg': 'NDCG@10 (%)',
    'f1': 'F1 Score (%)',
    'loss': 'Cross-Entropy Loss',
}


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


def load_results(results_dir: str, dataset: str):
    """Load results from JSON file."""
    filepath = os.path.join(results_dir, f'{dataset}_days_results.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_by_day(results: dict, dataset_name: str, output_dir: str):
    """
    Plot Acc@1, Acc@5, Acc@10 by day of week.
    
    Creates a grouped bar chart showing accuracy metrics for each day.
    Classic style with open bars and hatch patterns.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    days = DAY_NAMES
    x = np.arange(len(days))
    width = 0.7
    
    metrics = ['acc@1', 'acc@5', 'acc@10']
    titles = ['Accuracy@1', 'Accuracy@5', 'Accuracy@10']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [results[day][metric] for day in days]
        
        # Classic style: white fill with edge color based on weekday/weekend
        colors = [COLORS['weekday']] * 5 + [COLORS['weekend']] * 2
        hatches = ['\\\\'] * 5 + ['///'] * 2  # Hatch pattern for weekend
        
        bars = ax.bar(x, values, width, 
                      color='white', 
                      edgecolor=colors, 
                      linewidth=1.5)
        
        # Add hatches for weekend
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        # Add weekday/weekend average lines
#         weekday_avg = results['Weekday_Avg'][metric]
#         weekend_avg = results['Weekend_Avg'][metric]
        
#         ax.axhline(y=weekday_avg, color=COLORS['weekday'], linestyle='--', 
#                    linewidth=1.5, label=f'Weekday: {weekday_avg:.1f}%')
#         ax.axhline(y=weekend_avg, color=COLORS['weekend'], linestyle=':', 
#                    linewidth=1.5, label=f'Weekend: {weekend_avg:.1f}%')
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel(f'{title} (%)')
        ax.set_xticks(x)
        ax.set_xticklabels(DAY_ABBREV, rotation=0)
#         ax.legend(loc='lower left', fontsize=9, frameon=True, edgecolor='black', fancybox=False)
        
        # Set y-axis limits with some padding
        min_val = min(values) * 0.92
        max_val = max(values) * 1.08
        ax.set_ylim(min_val, max_val)
        
        # Apply classic style
        setup_classic_axes(ax)
    
    plt.tight_layout()
    
    # Save in multiple formats
    for fmt in ['pdf', 'png', 'svg']:
        output_path = os.path.join(output_dir, f'{dataset_name.lower()}_accuracy_by_day.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"Saved: {dataset_name.lower()}_accuracy_by_day")


def plot_weekday_weekend_comparison(results: dict, dataset_name: str, output_dir: str):
    """
    Plot grouped bar chart comparing weekday vs weekend performance.
    Classic style with hatched bars.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    metric_labels = ['ACC@1', 'ACC@5', 'ACC@10', 'MRR', 'NDCG', 'F1']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    weekday_vals = [results['Weekday_Avg'][m] for m in metrics]
    weekend_vals = [results['Weekend_Avg'][m] for m in metrics]
    
    # Classic style: white fill with different edge colors and hatches
    bars1 = ax.bar(x - width/2, weekday_vals, width, label='Weekday', 
                   color='white', edgecolor=COLORS['weekday'], linewidth=1.5,
                   hatch='\\\\')
    bars2 = ax.bar(x + width/2, weekend_vals, width, label='Weekend', 
                   color='white', edgecolor=COLORS['weekend'], linewidth=1.5,
                   hatch='///')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=10)
    
    # Add difference annotations
    for i, (wd, we) in enumerate(zip(weekday_vals, weekend_vals)):
        diff = wd - we
        color = 'black'
        ax.annotate(f'Δ{diff:+.1f}', xy=(x[i], max(wd, we) + 2),
                   ha='center', fontsize=9, color=color, fontweight='bold')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    
    # Add significance annotation if available
    if 'Statistical_Test' in results:
        st = results['Statistical_Test']
        sig_text = f"t-test p = {st['p_value']:.4f}"
        if st['significant_at_001']:
            sig_text += " **"
        elif st['significant_at_005']:
            sig_text += " *"
        ax.text(0.98, 0.02, sig_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=10, style='italic')
    
    # Apply classic style
    setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        output_path = os.path.join(output_dir, f'{dataset_name.lower()}_weekday_weekend_comparison.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"Saved: {dataset_name.lower()}_weekday_weekend_comparison")


def plot_performance_heatmap(results: dict, dataset_name: str, output_dir: str):
    """
    Create heatmap of all metrics by day of week.
    Classic style with grayscale colormap.
    """
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    metric_labels = ['ACC@1', 'ACC@5', 'ACC@10', 'MRR', 'NDCG', 'F1']
    
    data = []
    for day in DAY_NAMES:
        row = [results[day][m] for m in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, columns=metric_labels, index=DAY_NAMES)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Classic grayscale heatmap
    sns.heatmap(df, annot=True, fmt='.1f', cmap='Greys', 
                ax=ax, cbar_kws={'label': 'Score (%)'}, 
                linewidths=0.5, linecolor='black')
    
    # Highlight weekend rows with red border
    for i in [5, 6]:
        ax.add_patch(plt.Rectangle((0, i), len(metrics), 1, 
                                    fill=False, edgecolor='red', linewidth=2))
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Day of Week')
    
    # Style adjustments
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        output_path = os.path.join(output_dir, f'{dataset_name.lower()}_metrics_heatmap.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"Saved: {dataset_name.lower()}_metrics_heatmap")


def plot_performance_trend(results: dict, dataset_name: str, output_dir: str):
    """
    Create line plot showing performance trend across days.
    Classic style with open markers.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = DAY_NAMES
    x = np.arange(len(days))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr']
    labels = ['ACC@1', 'ACC@5', 'ACC@10', 'MRR']
    markers = ['o', 's', '^', 'D']
    colors = ['black', 'blue', 'red', 'green']
    linestyles = ['-', '-', '-', '-']
    
    for metric, label, marker, color, ls in zip(metrics, labels, markers, colors, linestyles):
        values = [results[day][metric] for day in days]
        ax.plot(x, values, marker=marker, label=label, color=color, 
                linewidth=1.5, markersize=7, linestyle=ls,
                markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
    
    # Add weekend shading (subtle)
    ax.axvspan(4.5, 6.5, alpha=0.1, color='red', label='Weekend')
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Score (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(DAY_ABBREV)
    ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
    
    # Apply classic style
    setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        output_path = os.path.join(output_dir, f'{dataset_name.lower()}_performance_trend.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"Saved: {dataset_name.lower()}_performance_trend")


def plot_sample_distribution(results: dict, dataset_name: str, output_dir: str):
    """
    Plot sample distribution by day of week.
    Classic style with hatched bars for weekend.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    days = DAY_NAMES
    samples = [results[day]['samples'] for day in days]
    
    colors = [COLORS['weekday']] * 5 + [COLORS['weekend']] * 2
    hatches = ['\\\\'] * 5 + ['///'] * 2
    
    bars = ax.bar(days, samples, color='white', edgecolor=colors, linewidth=1.5)
    
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    
    # Add value labels
    for bar, val in zip(bars, samples):
        height = bar.get_height()
        ax.annotate(f'{val:,}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=10)
    
    total = sum(samples)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Samples')
    ax.text(0.02, 0.98, f'Total: {total:,}', transform=ax.transAxes, 
            ha='left', va='top', fontsize=11)
    
    plt.xticks(rotation=45, ha='right')
    
    # Apply classic style
    setup_classic_axes(ax)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png']:
        output_path = os.path.join(output_dir, f'{dataset_name.lower()}_sample_distribution.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"Saved: {dataset_name.lower()}_sample_distribution")


def plot_combined_comparison(diy_results: dict, geolife_results: dict, output_dir: str):
    """
    Create combined visualization comparing both datasets.
    Classic style matching reference.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, results, name in zip(axes, [diy_results, geolife_results], ['DIY', 'GeoLife']):
        days = DAY_NAMES
        x = np.arange(len(days))
        
        values = [results[day]['acc@1'] for day in days]
        
        colors = [COLORS['weekday']] * 5 + [COLORS['weekend']] * 2
        hatches = ['\\\\'] * 5 + ['///'] * 2
        
        bars = ax.bar(x, values, color='white', edgecolor=colors, linewidth=1.5)
        
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
        
        # Average lines
        weekday_avg = results['Weekday_Avg']['acc@1']
        weekend_avg = results['Weekend_Avg']['acc@1']
        ax.axhline(y=weekday_avg, color=COLORS['weekday'], linestyle='--', linewidth=1.5)
        ax.axhline(y=weekend_avg, color=COLORS['weekend'], linestyle=':', linewidth=1.5)
        
        # Difference annotation
        diff = weekday_avg - weekend_avg
        ax.text(0.02, 0.98, f'Δ = {diff:.2f}%', 
                transform=ax.transAxes, ha='left', va='top', fontsize=11,
                bbox=dict(boxstyle='square', facecolor='white', edgecolor='black'))
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Accuracy@1 (%)')
        ax.set_title(f'{name}')
        ax.set_xticks(x)
        ax.set_xticklabels(DAY_ABBREV)
        
        # Apply classic style
        setup_classic_axes(ax)
    
    # Legend
    weekday_patch = mpatches.Patch(facecolor='white', edgecolor=COLORS['weekday'], 
                                    linewidth=1.5, label='Weekday')
    weekend_patch = mpatches.Patch(facecolor='white', edgecolor=COLORS['weekend'], 
                                    linewidth=1.5, hatch='///', label='Weekend')
    fig.legend(handles=[weekday_patch, weekend_patch], 
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.02),
               frameon=True, edgecolor='black', fancybox=False)
    
    plt.tight_layout()
    
    for fmt in ['pdf', 'png', 'svg']:
        output_path = os.path.join(output_dir, f'combined_comparison.{fmt}')
        plt.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print(f"Saved: combined_comparison")


def create_combined_figure(diy_results: dict, geolife_results: dict, output_dir: str):
    """Create a single comprehensive figure for publication."""
    fig = plt.figure(figsize=(16, 12))
    
    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    results_list = [('DIY', diy_results), ('GeoLife', geolife_results)]
    
    # (a) DIY Acc@1 by day
    ax1 = fig.add_subplot(gs[0, 0])
    results = diy_results
    days = DAY_NAMES
    x = np.arange(len(days))
    values = [results[day]['acc@1'] for day in days]
    colors = [COLORS['weekday']] * 5 + [COLORS['weekend']] * 2
    hatches = ['\\\\'] * 5 + ['///'] * 2
    bars = ax1.bar(x, values, color='white', edgecolor=colors, linewidth=1.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax1.set_xlabel('Day of Week')
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(DAY_ABBREV)
#     ax1.text(0.02, 0.98, '(a) DIY', transform=ax1.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax1)
    
    # (b) GeoLife Acc@1 by day
    ax2 = fig.add_subplot(gs[0, 1])
    results = geolife_results
    values = [results[day]['acc@1'] for day in days]
    bars = ax2.bar(x, values, color='white', edgecolor=colors, linewidth=1.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax2.set_xlabel('Day of Week')
    ax2.set_ylabel('Accuracy@1 (%)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(DAY_ABBREV)
#     ax2.text(0.02, 0.98, '(b) GeoLife', transform=ax2.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax2)
    
    # (c) Both datasets trend comparison
    ax3 = fig.add_subplot(gs[0, 2])
    for name, results in results_list:
        color = COLORS['diy'] if name == 'DIY' else COLORS['geolife']
        marker = MARKERS['diy'] if name == 'DIY' else MARKERS['geolife']
        values = [results[day]['acc@1'] for day in days]
        ax3.plot(x, values, marker=marker, label=name, color=color, 
                 linewidth=1.5, markersize=7,
                 markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)
    ax3.axvspan(4.5, 6.5, alpha=0.1, color='red')
    ax3.set_xlabel('Day of Week')
    ax3.set_ylabel('Accuracy@1 (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(DAY_ABBREV)
    ax3.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
#     ax3.text(0.02, 0.98, '(c) Comparison', transform=ax3.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax3)
    
    # (d) DIY Weekday vs Weekend
    ax4 = fig.add_subplot(gs[1, 0])
    results = diy_results
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    metric_labels = ['ACC@1', 'ACC@5', 'ACC@10', 'MRR', 'NDCG']
    mx = np.arange(len(metrics))
    width = 0.35
    weekday_vals = [results['Weekday_Avg'][m] for m in metrics]
    weekend_vals = [results['Weekend_Avg'][m] for m in metrics]
    ax4.bar(mx - width/2, weekday_vals, width, label='Weekday', 
            color='white', edgecolor=COLORS['weekday'], linewidth=1.5)
    ax4.bar(mx + width/2, weekend_vals, width, label='Weekend', 
            color='white', edgecolor=COLORS['weekend'], linewidth=1.5, hatch='///')
    ax4.set_xlabel('Metric')
    ax4.set_ylabel('Score (%)')
    ax4.set_xticks(mx)
    ax4.set_xticklabels(metric_labels, fontsize=10)
    ax4.legend(loc='best', frameon=True, edgecolor='black', fancybox=False, fontsize=9)
#     ax4.text(0.02, 0.98, '(d) DIY', transform=ax4.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax4)
    
    # (e) GeoLife Weekday vs Weekend
    ax5 = fig.add_subplot(gs[1, 1])
    results = geolife_results
    weekday_vals = [results['Weekday_Avg'][m] for m in metrics]
    weekend_vals = [results['Weekend_Avg'][m] for m in metrics]
    ax5.bar(mx - width/2, weekday_vals, width, label='Weekday', 
            color='white', edgecolor=COLORS['weekday'], linewidth=1.5)
    ax5.bar(mx + width/2, weekend_vals, width, label='Weekend', 
            color='white', edgecolor=COLORS['weekend'], linewidth=1.5, hatch='///')
    ax5.set_xlabel('Metric')
    ax5.set_ylabel('Score (%)')
    ax5.set_xticks(mx)
    ax5.set_xticklabels(metric_labels, fontsize=10)
    ax5.legend(loc='best', frameon=True, edgecolor='black', fancybox=False, fontsize=9)
#     ax5.text(0.02, 0.98, '(e) GeoLife', transform=ax5.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax5)
    
    # (f) Weekend performance drop comparison
    ax6 = fig.add_subplot(gs[1, 2])
    metrics_drop = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    mx = np.arange(len(metrics_drop))
    width = 0.35
    
    diy_drop = [(diy_results['Weekday_Avg'][m] - diy_results['Weekend_Avg'][m]) 
                for m in metrics_drop]
    geo_drop = [(geolife_results['Weekday_Avg'][m] - geolife_results['Weekend_Avg'][m]) 
                for m in metrics_drop]
    
    ax6.bar(mx - width/2, diy_drop, width, label='DIY', 
            color='white', edgecolor=COLORS['diy'], linewidth=1.5)
    ax6.bar(mx + width/2, geo_drop, width, label='GeoLife', 
            color='white', edgecolor=COLORS['geolife'], linewidth=1.5, hatch='///')
    ax6.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax6.set_xlabel('Metric')
    ax6.set_ylabel('Weekend Drop (%)')
    ax6.set_xticks(mx)
    ax6.set_xticklabels(metric_labels, fontsize=10)
    ax6.legend(loc='best', frameon=True, edgecolor='black', fancybox=False, fontsize=9)
#     ax6.text(0.02, 0.98, '(f) Weekend Drop', transform=ax6.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax6)
    
    # (g) DIY Sample distribution
    ax7 = fig.add_subplot(gs[2, 0])
    results = diy_results
    samples = [results[day]['samples'] for day in days]
    bars = ax7.bar(x, samples, color='white', edgecolor=colors, linewidth=1.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax7.set_xlabel('Day of Week')
    ax7.set_ylabel('Samples')
    ax7.set_xticks(x)
    ax7.set_xticklabels(DAY_ABBREV)
#     ax7.text(0.02, 0.98, '(g) DIY Samples', transform=ax7.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax7)
    
    # (h) GeoLife Sample distribution
    ax8 = fig.add_subplot(gs[2, 1])
    results = geolife_results
    samples = [results[day]['samples'] for day in days]
    bars = ax8.bar(x, samples, color='white', edgecolor=colors, linewidth=1.5)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    ax8.set_xlabel('Day of Week')
    ax8.set_ylabel('Samples')
    ax8.set_xticks(x)
    ax8.set_xticklabels(DAY_ABBREV)
#     ax8.text(0.02, 0.98, '(h) GeoLife Samples', transform=ax8.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')
    setup_classic_axes(ax8)
    
    # (i) Statistical significance summary
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis('off')
    
    # Create summary text
    summary_text = "Statistical Analysis Summary\n"
    summary_text += "=" * 35 + "\n\n"
    
    for name, results in results_list:
        st = results.get('Statistical_Test', {})
        summary_text += f"{name} Dataset:\n"
        summary_text += f"  Weekday Mean: {st.get('weekday_mean', 0):.2f}%\n"
        summary_text += f"  Weekend Mean: {st.get('weekend_mean', 0):.2f}%\n"
        summary_text += f"  Difference: {st.get('difference', 0):.2f}%\n"
        summary_text += f"  p-value: {st.get('p_value', 1):.4f}"
        if st.get('significant_at_001'):
            summary_text += " **\n"
        elif st.get('significant_at_005'):
            summary_text += " *\n"
        else:
            summary_text += "\n"
        summary_text += "\n"
    
    summary_text += "* p < 0.05, ** p < 0.01"
    
    ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='square', facecolor='white', edgecolor='black'))
#     ax9.text(0.02, 0.98, '(i) Statistics', transform=ax9.transAxes, fontsize=12, 
#              fontweight='bold', va='top', ha='left')

    panel_labels = [
        '(a) DIY Acc@1',
        '(b) GeoLife Acc@1',
        '(c) Acc@1 Comparison',
        '(d) DIY Metrics',
        '(e) GeoLife Metrics',
        '(f) Weekend Drop',
        '(g) DIY Samples',
        '(h) GeoLife Samples',
        '(i) Statistical Summary'
    ]

    axes = [ax1, ax2, ax3,
            ax4, ax5, ax6,
            ax7, ax8, ax9]

    for label, ax in zip(panel_labels, axes):
        bbox = ax.get_position()
        fig.text(
            bbox.x0 - 0.02,   # left offset (outside axis)
            bbox.y1 + 0.01,   # above axis
            label,
            fontsize=13,
            fontweight='bold',
            ha='left',
            va='bottom'
        )
    
    # Save
    for fmt in ['pdf', 'png', 'svg']:
        output_path = os.path.join(output_dir, f'combined_figure.{fmt}')
        fig.savefig(output_path, format=fmt, dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.close()
    print("Saved: combined_figure")


def generate_latex_table(results: dict, dataset_name: str, output_dir: str):
    """
    Generate LaTeX table for publication.
    """
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{Performance Metrics by Day of Week - {dataset_name} Dataset}}")
    latex.append(r"\label{tab:" + dataset_name.lower() + "_days}")
    latex.append(r"\begin{tabular}{lcccccc}")
    latex.append(r"\toprule")
    latex.append(r"Day & Acc@1 & Acc@5 & Acc@10 & MRR & NDCG & F1 \\")
    latex.append(r"\midrule")
    
    # Individual days
    for day in DAY_NAMES:
        r = results[day]
        vals = " & ".join([f"{r[m]:.2f}" for m in metrics])
        highlight = r" \rowcolor{gray!20}" if r['is_weekend'] else ""
        latex.append(f"{highlight}{day} & {vals} \\\\")
    
    latex.append(r"\midrule")
    
    # Averages
    for key, label in [('Weekday_Avg', 'Weekday Avg'), ('Weekend_Avg', 'Weekend Avg')]:
        r = results[key]
        vals = " & ".join([f"{r[m]:.2f}" for m in metrics])
        latex.append(f"\\textbf{{{label}}} & {vals} \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    # Save to file
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_table.tex')
    with open(output_path, 'w') as f:
        f.write('\n'.join(latex))
    print(f"Saved: {dataset_name.lower()}_table.tex")


def create_summary_csv(diy_results: dict, geolife_results: dict, output_dir: str):
    """
    Create comprehensive CSV summary.
    """
    rows = []
    
    for dataset, results in [('DIY', diy_results), ('GeoLife', geolife_results)]:
        if results is None:
            continue
        
        # Individual days
        for day in DAY_NAMES:
            r = results[day]
            rows.append({
                'Dataset': dataset,
                'Day': day,
                'Type': 'Weekend' if r['is_weekend'] else 'Weekday',
                'Samples': r['samples'],
                'Acc@1': r['acc@1'],
                'Acc@5': r['acc@5'],
                'Acc@10': r['acc@10'],
                'MRR': r['mrr'],
                'NDCG': r['ndcg'],
                'F1': r['f1'],
                'Loss': r['loss'],
            })
        
        # Aggregates
        for key, label in [('Weekday_Avg', 'Weekday Average'), 
                           ('Weekend_Avg', 'Weekend Average'),
                           ('Overall', 'Overall')]:
            if key in results:
                r = results[key]
                rows.append({
                    'Dataset': dataset,
                    'Day': label,
                    'Type': 'Summary',
                    'Samples': r['samples'],
                    'Acc@1': r.get('acc@1', 0),
                    'Acc@5': r.get('acc@5', 0),
                    'Acc@10': r.get('acc@10', 0),
                    'MRR': r.get('mrr', 0),
                    'NDCG': r.get('ndcg', 0),
                    'F1': r.get('f1', 0),
                    'Loss': r.get('loss', 0),
                })
    
    df = pd.DataFrame(rows)
    output_path = os.path.join(output_dir, 'days_analysis_summary.csv')
    df.to_csv(output_path, index=False, float_format='%.4f')
    print(f"Saved: days_analysis_summary.csv")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Day-of-Week Analysis Visualizations (V2)")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_days_analysis_v2/results',
        help='Directory containing results JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_days_analysis_v2/figures',
        help='Output directory for figures'
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING DAY-OF-WEEK ANALYSIS VISUALIZATIONS (V2 - Classic Style)")
    print("=" * 80)
    
    # Load results
    diy_results = None
    geolife_results = None
    
    try:
        diy_results = load_results(args.results_dir, 'diy')
        print("Loaded DIY results")
    except FileNotFoundError:
        print("DIY results not found, skipping")
    
    try:
        geolife_results = load_results(args.results_dir, 'geolife')
        print("Loaded GeoLife results")
    except FileNotFoundError:
        print("GeoLife results not found, skipping")
    
    # Generate visualizations for each dataset
    for results, name in [(diy_results, 'DIY'), (geolife_results, 'GeoLife')]:
        if results is None:
            continue
        
        print(f"\nGenerating visualizations for {name}...")
        
        plot_accuracy_by_day(results, name, args.output_dir)
        plot_weekday_weekend_comparison(results, name, args.output_dir)
        plot_performance_heatmap(results, name, args.output_dir)
        plot_performance_trend(results, name, args.output_dir)
        plot_sample_distribution(results, name, args.output_dir)
        generate_latex_table(results, name, args.output_dir)
    
    # Combined comparison
    if diy_results and geolife_results:
        print("\nGenerating combined comparison...")
        plot_combined_comparison(diy_results, geolife_results, args.output_dir)
        create_combined_figure(diy_results, geolife_results, args.output_dir)
        

        
        
    
    # Create summary CSV
    print("\nCreating summary CSV...")
    create_summary_csv(diy_results, geolife_results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE (V2 - Classic Style)")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
