#!/usr/bin/env python
"""
Visualization Script for Day-of-Week Analysis Experiment.

This script generates publication-quality figures for the day-of-week
analysis experiment. Designed for Nature Journal standards.

Visualizations:
1. Performance by Day of Week (bar chart)
2. Weekday vs Weekend Comparison (grouped bar)
3. All Metrics Heatmap
4. Performance Pattern Line Plot
5. Statistical Summary Table

Author: Experiment Script for PhD Thesis
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
import seaborn as sns
from scipy import stats

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# Color scheme for publication
COLORS = {
    'weekday': '#2E86AB',     # Blue
    'weekend': '#E94F37',     # Red
    'overall': '#4A4E69',     # Dark gray
    'monday': '#264653',
    'tuesday': '#2A9D8F',
    'wednesday': '#E9C46A',
    'thursday': '#F4A261',
    'friday': '#E76F51',
    'saturday': '#BC4749',
    'sunday': '#A44A3F',
}

DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DAY_COLORS = [COLORS['weekday']] * 5 + [COLORS['weekend']] * 2


def load_results(results_dir: str, dataset: str):
    """Load results from JSON file."""
    filepath = os.path.join(results_dir, f'{dataset}_days_results.json')
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_accuracy_by_day(results: dict, dataset_name: str, output_dir: str):
    """
    Plot Acc@1, Acc@5, Acc@10 by day of week.
    
    Creates a grouped bar chart showing accuracy metrics for each day.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    days = DAY_NAMES
    x = np.arange(len(days))
    width = 0.7
    
    metrics = ['acc@1', 'acc@5', 'acc@10']
    titles = ['Top-1 Accuracy', 'Top-5 Accuracy', 'Top-10 Accuracy']
    
    for ax, metric, title in zip(axes, metrics, titles):
        values = [results[day][metric] for day in days]
        
        # Color bars by weekday/weekend
        bars = ax.bar(x, values, width, color=DAY_COLORS, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # Add weekday/weekend average lines
        weekday_avg = results['Weekday_Avg'][metric]
        weekend_avg = results['Weekend_Avg'][metric]
        
        ax.axhline(y=weekday_avg, color=COLORS['weekday'], linestyle='--', 
                   alpha=0.7, label=f'Weekday Avg: {weekday_avg:.1f}%')
        ax.axhline(y=weekend_avg, color=COLORS['weekend'], linestyle='--', 
                   alpha=0.7, label=f'Weekend Avg: {weekend_avg:.1f}%')
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([d[:3] for d in days], rotation=45, ha='right')
        ax.legend(loc='lower left', fontsize=8)
        
        # Set y-axis limits with some padding
        min_val = min(values) * 0.9
        max_val = max(values) * 1.08
        ax.set_ylim(min_val, max_val)
    
    # Add legend for weekday/weekend colors
    weekday_patch = mpatches.Patch(color=COLORS['weekday'], label='Weekday')
    weekend_patch = mpatches.Patch(color=COLORS['weekend'], label='Weekend')
    fig.legend(handles=[weekday_patch, weekend_patch], 
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.08))
    
    plt.suptitle(f'{dataset_name} Dataset: Accuracy by Day of Week', fontsize=14, y=1.12)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_accuracy_by_day.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_weekday_weekend_comparison(results: dict, dataset_name: str, output_dir: str):
    """
    Plot grouped bar chart comparing weekday vs weekend performance.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    metric_labels = ['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG', 'F1']
    
    x = np.arange(len(metrics))
    width = 0.35
    
    weekday_vals = [results['Weekday_Avg'][m] for m in metrics]
    weekend_vals = [results['Weekend_Avg'][m] for m in metrics]
    
    bars1 = ax.bar(x - width/2, weekday_vals, width, label='Weekday', 
                   color=COLORS['weekday'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, weekend_vals, width, label='Weekend', 
                   color=COLORS['weekend'], edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    # Add difference annotations
    for i, (wd, we) in enumerate(zip(weekday_vals, weekend_vals)):
        diff = wd - we
        color = 'green' if diff > 0 else 'red'
        ax.annotate(f'Δ{diff:+.1f}', xy=(x[i], max(wd, we) + 1.5),
                   ha='center', fontsize=8, color=color, fontweight='bold')
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'{dataset_name} Dataset: Weekday vs Weekend Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    
    # Add significance annotation if available
    if 'Statistical_Test' in results:
        st = results['Statistical_Test']
        sig_text = f"t-test p-value: {st['p_value']:.4f}"
        if st['significant_at_001']:
            sig_text += " **"
        elif st['significant_at_005']:
            sig_text += " *"
        ax.text(0.98, 0.02, sig_text, transform=ax.transAxes, 
                ha='right', va='bottom', fontsize=9, style='italic')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_weekday_weekend_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_performance_heatmap(results: dict, dataset_name: str, output_dir: str):
    """
    Create heatmap of all metrics by day of week.
    """
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1']
    metric_labels = ['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG', 'F1']
    
    data = []
    for day in DAY_NAMES:
        row = [results[day][m] for m in metrics]
        data.append(row)
    
    df = pd.DataFrame(data, columns=metric_labels, index=DAY_NAMES)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Normalize each column for better visualization
    df_normalized = (df - df.min()) / (df.max() - df.min())
    
    sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn', 
                ax=ax, cbar_kws={'label': 'Score (%)'}, 
                linewidths=0.5, linecolor='white')
    
    # Highlight weekend rows
    for i in [5, 6]:
        ax.add_patch(plt.Rectangle((0, i), len(metrics), 1, 
                                    fill=False, edgecolor='red', linewidth=2))
    
    ax.set_title(f'{dataset_name} Dataset: Performance Metrics by Day of Week')
    ax.set_xlabel('Metric')
    ax.set_ylabel('Day of Week')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_performance_trend(results: dict, dataset_name: str, output_dir: str):
    """
    Create line plot showing performance trend across days.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    days = DAY_NAMES
    x = np.arange(len(days))
    
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr']
    labels = ['Acc@1', 'Acc@5', 'Acc@10', 'MRR']
    markers = ['o', 's', '^', 'D']
    colors = ['#264653', '#2A9D8F', '#E9C46A', '#E76F51']
    
    for metric, label, marker, color in zip(metrics, labels, markers, colors):
        values = [results[day][metric] for day in days]
        ax.plot(x, values, marker=marker, label=label, color=color, 
                linewidth=2, markersize=8)
    
    # Add weekend shading
    ax.axvspan(4.5, 6.5, alpha=0.2, color='red', label='Weekend')
    
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'{dataset_name} Dataset: Performance Trend Across Week')
    ax.set_xticks(x)
    ax.set_xticklabels(days, rotation=45, ha='right')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_performance_trend.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_sample_distribution(results: dict, dataset_name: str, output_dir: str):
    """
    Plot sample distribution by day of week.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    
    days = DAY_NAMES
    samples = [results[day]['samples'] for day in days]
    
    bars = ax.bar(days, samples, color=DAY_COLORS, edgecolor='black', linewidth=0.5)
    
    # Add value labels
    for bar, val in zip(bars, samples):
        height = bar.get_height()
        ax.annotate(f'{val:,}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3),
                   textcoords="offset points",
                   ha='center', va='bottom', fontsize=9)
    
    total = sum(samples)
    ax.set_xlabel('Day of Week')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'{dataset_name} Dataset: Sample Distribution (Total: {total:,})')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{dataset_name.lower()}_sample_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_combined_comparison(diy_results: dict, geolife_results: dict, output_dir: str):
    """
    Create combined visualization comparing both datasets.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, results, name in zip(axes, [diy_results, geolife_results], ['DIY', 'Geolife']):
        days = DAY_NAMES
        x = np.arange(len(days))
        
        values = [results[day]['acc@1'] for day in days]
        bars = ax.bar(x, values, color=DAY_COLORS, edgecolor='black', linewidth=0.5)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=8)
        
        # Average lines
        weekday_avg = results['Weekday_Avg']['acc@1']
        weekend_avg = results['Weekend_Avg']['acc@1']
        ax.axhline(y=weekday_avg, color=COLORS['weekday'], linestyle='--', alpha=0.7)
        ax.axhline(y=weekend_avg, color=COLORS['weekend'], linestyle='--', alpha=0.7)
        
        # Difference annotation
        diff = weekday_avg - weekend_avg
        ax.text(0.02, 0.98, f'Weekday-Weekend Δ: {diff:.2f}%', 
                transform=ax.transAxes, ha='left', va='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax.set_xlabel('Day of Week')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'{name} Dataset')
        ax.set_xticks(x)
        ax.set_xticklabels([d[:3] for d in days], rotation=45, ha='right')
    
    # Legend
    weekday_patch = mpatches.Patch(color=COLORS['weekday'], label='Weekday')
    weekend_patch = mpatches.Patch(color=COLORS['weekend'], label='Weekend')
    fig.legend(handles=[weekday_patch, weekend_patch], 
               loc='upper center', ncol=2, bbox_to_anchor=(0.5, 1.05))
    
    plt.suptitle('Top-1 Accuracy by Day of Week: Dataset Comparison', fontsize=14, y=1.1)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'combined_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.savefig(output_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


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
    print(f"Saved: {output_path}")


def create_summary_csv(diy_results: dict, geolife_results: dict, output_dir: str):
    """
    Create comprehensive CSV summary.
    """
    rows = []
    
    for dataset, results in [('DIY', diy_results), ('Geolife', geolife_results)]:
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
    print(f"Saved: {output_path}")
    
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate Day-of-Week Analysis Visualizations")
    parser.add_argument(
        '--results_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_days_analysis/results',
        help='Directory containing results JSON files'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/data/next_loc_clean_v2/scripts/experiment_days_analysis/figures',
        help='Output directory for figures'
    )
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("=" * 80)
    print("GENERATING DAY-OF-WEEK ANALYSIS VISUALIZATIONS")
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
        print("Loaded Geolife results")
    except FileNotFoundError:
        print("Geolife results not found, skipping")
    
    # Generate visualizations for each dataset
    for results, name in [(diy_results, 'DIY'), (geolife_results, 'Geolife')]:
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
    
    # Create summary CSV
    print("\nCreating summary CSV...")
    create_summary_csv(diy_results, geolife_results, args.output_dir)
    
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION COMPLETE")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
