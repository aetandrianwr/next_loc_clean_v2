"""
Results Visualization Script for Next Location Prediction.

This script creates publication-quality visualizations for experimental results:
1. Performance comparison bar charts
2. Confusion matrices
3. Per-user performance distributions
4. Sequence length analysis plots
5. Confidence distribution plots

Designed for Nature Journal standard publications.

Author: Research Team
Date: 2026-01-02
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.ticker import FuncFormatter

# Set publication-quality defaults
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Color palette
COLORS = {
    'diy': '#8468F5',      # Purple
    'geolife': '#46B6E8',  # Blue
    'positive': '#2FD4A1', # Green
    'negative': '#FF6B6B', # Red
    'neutral': '#B2B2B2',  # Gray
}


# =============================================================================
# Performance Visualization
# =============================================================================

def plot_performance_comparison(metrics_dict: Dict[str, Dict], output_path: str = None):
    """Create bar chart comparing performance metrics across datasets."""
    datasets = list(metrics_dict.keys())
    metrics_to_plot = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    metric_labels = ['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG@10']
    
    x = np.arange(len(metric_labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, dataset in enumerate(datasets):
        values = [metrics_dict[dataset][m] for m in metrics_to_plot]
        
        # Get confidence intervals
        ci_lower = []
        ci_upper = []
        for m in metrics_to_plot:
            if f'{m}_ci_lower' in metrics_dict[dataset]:
                ci_lower.append(metrics_dict[dataset][f'{m}_ci_lower'])
                ci_upper.append(metrics_dict[dataset][f'{m}_ci_upper'])
            else:
                ci_lower.append(values[metrics_to_plot.index(m)])
                ci_upper.append(values[metrics_to_plot.index(m)])
        
        errors = [(v - l, u - v) for v, l, u in zip(values, ci_lower, ci_upper)]
        yerr = np.array(errors).T
        
        offset = width * (i - (len(datasets) - 1) / 2)
        bars = ax.bar(x + offset, values, width, label=dataset.upper(), 
                     color=COLORS.get(dataset, COLORS['neutral']),
                     yerr=yerr, capsize=3, alpha=0.8)
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(f'{val:.1f}',
                       xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Metric')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison: PointerNetwork V45')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(loc='upper right')
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f'{y:.0f}%'))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


def plot_metrics_radar(metrics_dict: Dict[str, Dict], output_path: str = None):
    """Create radar chart for multi-metric comparison."""
    datasets = list(metrics_dict.keys())
    metrics = ['acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg']
    metric_labels = ['Acc@1', 'Acc@5', 'Acc@10', 'MRR', 'NDCG@10']
    
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for dataset in datasets:
        values = [metrics_dict[dataset][m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, 
               label=dataset.upper(), color=COLORS.get(dataset, COLORS['neutral']))
        ax.fill(angles, values, alpha=0.25, color=COLORS.get(dataset, COLORS['neutral']))
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0, 100)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    ax.set_title('Multi-Metric Performance Comparison', pad=20)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


# =============================================================================
# User and Sequence Analysis
# =============================================================================

def plot_user_performance_distribution(user_stats_path: str, dataset_name: str,
                                       output_path: str = None):
    """Plot distribution of per-user performance."""
    df = pd.read_csv(user_stats_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Acc@1 distribution
    ax = axes[0]
    ax.hist(df['acc@1'], bins=20, color=COLORS.get(dataset_name, COLORS['neutral']), 
           alpha=0.7, edgecolor='white')
    ax.axvline(df['acc@1'].mean(), color='red', linestyle='--', 
              label=f'Mean: {df["acc@1"].mean():.1f}%')
    ax.set_xlabel('Accuracy@1 (%)')
    ax.set_ylabel('Number of Users')
    ax.set_title('Acc@1 Distribution by User')
    ax.legend()
    
    # Acc@5 distribution
    ax = axes[1]
    ax.hist(df['acc@5'], bins=20, color=COLORS.get(dataset_name, COLORS['neutral']), 
           alpha=0.7, edgecolor='white')
    ax.axvline(df['acc@5'].mean(), color='red', linestyle='--', 
              label=f'Mean: {df["acc@5"].mean():.1f}%')
    ax.set_xlabel('Accuracy@5 (%)')
    ax.set_ylabel('Number of Users')
    ax.set_title('Acc@5 Distribution by User')
    ax.legend()
    
    # MRR distribution
    ax = axes[2]
    ax.hist(df['mrr'], bins=20, color=COLORS.get(dataset_name, COLORS['neutral']), 
           alpha=0.7, edgecolor='white')
    ax.axvline(df['mrr'].mean(), color='red', linestyle='--', 
              label=f'Mean: {df["mrr"].mean():.1f}%')
    ax.set_xlabel('MRR (%)')
    ax.set_ylabel('Number of Users')
    ax.set_title('MRR Distribution by User')
    ax.legend()
    
    plt.suptitle(f'Per-User Performance Distribution: {dataset_name.upper()}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, axes


def plot_sequence_length_analysis(seq_analysis_path: str, dataset_name: str,
                                  output_path: str = None):
    """Plot performance by sequence length."""
    df = pd.read_csv(seq_analysis_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot lines for different metrics
    ax.plot(df['seq_len'], df['acc@1'], 'o-', label='Acc@1', 
           color=COLORS['positive'], linewidth=2, markersize=6)
    ax.plot(df['seq_len'], df['acc@5'], 's-', label='Acc@5', 
           color=COLORS.get(dataset_name, COLORS['neutral']), linewidth=2, markersize=6)
    ax.plot(df['seq_len'], df['mrr'], '^-', label='MRR', 
           color=COLORS['negative'], linewidth=2, markersize=6)
    
    ax.set_xlabel('Sequence Length')
    ax.set_ylabel('Score (%)')
    ax.set_title(f'Performance by Sequence Length: {dataset_name.upper()}')
    ax.legend(loc='best')
    ax.set_ylim(0, 100)
    
    # Add sample count as bar chart on secondary axis
    ax2 = ax.twinx()
    ax2.bar(df['seq_len'], df['n_samples'], alpha=0.2, color='gray', label='Sample Count')
    ax2.set_ylabel('Number of Samples', color='gray')
    ax2.tick_params(axis='y', labelcolor='gray')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


# =============================================================================
# Confidence Analysis
# =============================================================================

def plot_confidence_analysis(sample_results_path: str, dataset_name: str,
                            output_path: str = None):
    """Plot confidence distribution and calibration."""
    df = pd.read_csv(sample_results_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Confidence distribution by correctness
    ax = axes[0]
    correct_conf = df[df['is_correct_1']]['pred_prob']
    incorrect_conf = df[~df['is_correct_1']]['pred_prob']
    
    ax.hist(correct_conf, bins=30, alpha=0.6, label='Correct', 
           color=COLORS['positive'], edgecolor='white')
    ax.hist(incorrect_conf, bins=30, alpha=0.6, label='Incorrect', 
           color=COLORS['negative'], edgecolor='white')
    ax.set_xlabel('Prediction Confidence')
    ax.set_ylabel('Count')
    ax.set_title('Confidence Distribution by Prediction Correctness')
    ax.legend()
    
    # Calibration plot
    ax = axes[1]
    df['conf_bin'] = pd.cut(df['pred_prob'], bins=10)
    calibration = df.groupby('conf_bin', observed=True).agg({
        'is_correct_1': 'mean',
        'pred_prob': 'mean'
    }).reset_index()
    
    ax.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    ax.plot(calibration['pred_prob'], calibration['is_correct_1'], 'o-', 
           color=COLORS.get(dataset_name, COLORS['neutral']), linewidth=2, markersize=8,
           label='Model')
    ax.set_xlabel('Mean Predicted Confidence')
    ax.set_ylabel('Actual Accuracy')
    ax.set_title('Calibration Plot')
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.suptitle(f'Confidence Analysis: {dataset_name.upper()}', fontsize=14)
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, axes


def plot_rank_distribution(sample_results_path: str, dataset_name: str,
                          output_path: str = None):
    """Plot distribution of prediction ranks."""
    df = pd.read_csv(sample_results_path)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Filter ranks <= 20 for visibility
    ranks = df['rank'].clip(upper=20)
    
    counts, bins, patches = ax.hist(ranks, bins=range(1, 22), 
                                    color=COLORS.get(dataset_name, COLORS['neutral']),
                                    alpha=0.7, edgecolor='white', align='left')
    
    # Color top-1 predictions differently
    if len(patches) > 0:
        patches[0].set_facecolor(COLORS['positive'])
    
    ax.set_xlabel('Rank of True Label')
    ax.set_ylabel('Count')
    ax.set_title(f'Distribution of Prediction Ranks: {dataset_name.upper()}')
    ax.set_xticks(range(1, 21))
    
    # Add percentage annotations for top ranks
    total = len(df)
    for i, count in enumerate(counts[:10]):
        if count > 0:
            pct = count / total * 100
            ax.annotate(f'{pct:.1f}%', 
                       xy=(i + 1, count), 
                       xytext=(0, 5), 
                       textcoords='offset points',
                       ha='center', fontsize=8)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {output_path}")
    
    return fig, ax


# =============================================================================
# Summary Tables
# =============================================================================

def create_latex_table(metrics_dict: Dict[str, Dict], output_path: str = None) -> str:
    """Create LaTeX-formatted results table."""
    datasets = list(metrics_dict.keys())
    
    latex = r"""
\begin{table}[h]
\centering
\caption{Performance Evaluation of PointerNetwork V45 on Next Location Prediction}
\label{tab:results}
\begin{tabular}{lccccc}
\hline
Dataset & Acc@1 (\%) & Acc@5 (\%) & Acc@10 (\%) & MRR (\%) & NDCG@10 (\%) \\
\hline
"""
    
    for dataset in datasets:
        m = metrics_dict[dataset]
        row = f"{dataset.upper()} & "
        row += f"{m['acc@1']:.2f} & "
        row += f"{m['acc@5']:.2f} & "
        row += f"{m['acc@10']:.2f} & "
        row += f"{m['mrr']:.2f} & "
        row += f"{m['ndcg']:.2f} \\\\\n"
        latex += row
    
    latex += r"""\hline
\end{tabular}
\end{table}
"""
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(latex)
        print(f"Saved: {output_path}")
    
    return latex


def create_markdown_table(metrics_dict: Dict[str, Dict], output_path: str = None) -> str:
    """Create Markdown-formatted results table."""
    datasets = list(metrics_dict.keys())
    
    md = "| Dataset | Acc@1 (%) | Acc@5 (%) | Acc@10 (%) | MRR (%) | NDCG@10 (%) | F1 (%) | N |\n"
    md += "|---------|-----------|-----------|------------|---------|-------------|--------|---|\n"
    
    for dataset in datasets:
        m = metrics_dict[dataset]
        row = f"| {dataset.upper()} | "
        row += f"{m['acc@1']:.2f} | "
        row += f"{m['acc@5']:.2f} | "
        row += f"{m['acc@10']:.2f} | "
        row += f"{m['mrr']:.2f} | "
        row += f"{m['ndcg']:.2f} | "
        row += f"{m['f1']*100:.2f} | "
        row += f"{int(m['total'])} |\n"
        md += row
    
    if output_path:
        with open(output_path, 'w') as f:
            f.write(md)
        print(f"Saved: {output_path}")
    
    return md


# =============================================================================
# Main Visualization Pipeline
# =============================================================================

def generate_all_visualizations(results_dir: str, output_dir: str):
    """Generate all visualizations from results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all metrics
    metrics_dict = {}
    datasets = ['diy', 'geolife']
    
    for dataset in datasets:
        metrics_path = os.path.join(results_dir, f'{dataset}_metrics.json')
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics_dict[dataset] = json.load(f)
    
    if not metrics_dict:
        print("No metrics files found!")
        return
    
    print(f"Generating visualizations for: {list(metrics_dict.keys())}")
    
    # 1. Performance comparison
    plot_performance_comparison(
        metrics_dict, 
        os.path.join(output_dir, 'performance_comparison.png')
    )
    
    # 2. Radar chart
    plot_metrics_radar(
        metrics_dict,
        os.path.join(output_dir, 'performance_radar.png')
    )
    
    # 3. Per-dataset visualizations
    for dataset in metrics_dict.keys():
        # User performance distribution
        user_stats_path = os.path.join(results_dir, f'{dataset}_user_statistics.csv')
        if os.path.exists(user_stats_path):
            plot_user_performance_distribution(
                user_stats_path, dataset,
                os.path.join(output_dir, f'{dataset}_user_distribution.png')
            )
        
        # Sequence length analysis
        seq_analysis_path = os.path.join(results_dir, f'{dataset}_sequence_analysis.csv')
        if os.path.exists(seq_analysis_path):
            plot_sequence_length_analysis(
                seq_analysis_path, dataset,
                os.path.join(output_dir, f'{dataset}_sequence_analysis.png')
            )
        
        # Confidence analysis
        sample_results_path = os.path.join(results_dir, f'{dataset}_sample_results.csv')
        if os.path.exists(sample_results_path):
            plot_confidence_analysis(
                sample_results_path, dataset,
                os.path.join(output_dir, f'{dataset}_confidence_analysis.png')
            )
            
            plot_rank_distribution(
                sample_results_path, dataset,
                os.path.join(output_dir, f'{dataset}_rank_distribution.png')
            )
    
    # 4. Tables
    create_latex_table(metrics_dict, os.path.join(output_dir, 'results_table.tex'))
    create_markdown_table(metrics_dict, os.path.join(output_dir, 'results_table.md'))
    
    print(f"\nAll visualizations saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations for experiment results")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing result files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for visualizations")
    args = parser.parse_args()
    
    generate_all_visualizations(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
