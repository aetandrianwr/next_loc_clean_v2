#!/usr/bin/env python3
"""
Comprehensive Results Collection and Analysis for Ablation Study.

This script collects all ablation study results, generates summary tables,
statistical analysis, and exports data for documentation.

Usage:
    python collect_results.py
"""

import os
import json
import glob
from pathlib import Path
from datetime import datetime, timezone, timedelta

import pandas as pd
import numpy as np


# Configuration
RESULTS_DIR = Path(__file__).parent / 'results'
OUTPUT_DIR = Path(__file__).parent / 'results'

# Ablation descriptions
ABLATION_DESCRIPTIONS = {
    'full': 'Complete model (baseline)',
    'no_pointer': 'w/o Pointer Mechanism',
    'no_generation': 'w/o Generation Head',
    'no_position_bias': 'w/o Position Bias',
    'no_temporal': 'w/o Temporal Embeddings',
    'no_user': 'w/o User Embedding',
    'no_pos_from_end': 'w/o Position-from-End',
    'single_layer': 'Single Transformer Layer',
    'no_gate': 'w/o Adaptive Gate (Fixed 0.5)',
}

# Metrics to collect
METRICS = [
    'correct@1', 'correct@3', 'correct@5', 'correct@10',
    'acc@1', 'acc@5', 'acc@10', 'mrr', 'ndcg', 'f1', 'loss', 'total', 'rr'
]


def collect_dataset_results(dataset_dir):
    """Collect results from all ablation experiments for a dataset."""
    results = []
    
    for exp_dir in sorted(dataset_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        
        results_file = exp_dir / 'results.json'
        if not results_file.exists():
            continue
        
        with open(results_file, 'r') as f:
            data = json.load(f)
        
        ablation_type = data.get('ablation_type', 'unknown')
        
        result = {
            'ablation': ablation_type,
            'description': ABLATION_DESCRIPTIONS.get(ablation_type, ablation_type),
            'parameters': data.get('parameters', 0),
        }
        
        # Add validation metrics
        val = data.get('validation', {})
        for metric in METRICS:
            if metric in val:
                result[f'val_{metric}'] = val[metric]
        
        # Add test metrics
        test = data.get('test', {})
        for metric in METRICS:
            if metric in test:
                result[f'test_{metric}'] = test[metric]
        
        results.append(result)
    
    return results


def calculate_delta(df, baseline_metric='test_acc@1'):
    """Calculate delta from baseline for each ablation."""
    baseline = df[df['ablation'] == 'full'][baseline_metric].values[0]
    df['delta_acc1'] = df[baseline_metric] - baseline
    return df, baseline


def generate_latex_table(df, dataset_name, baseline_acc1):
    """Generate publication-quality LaTeX table."""
    latex = []
    latex.append(r"\begin{table*}[htbp]")
    latex.append(r"\centering")
    latex.append(r"\small")
    latex.append(f"\\caption{{Ablation Study Results on {dataset_name.upper()} Dataset. "
                 f"Baseline Acc@1: {baseline_acc1:.2f}\\%. "
                 r"$\Delta$Acc@1 shows performance change relative to the full model.}}")
    latex.append(f"\\label{{tab:ablation_{dataset_name}}}")
    latex.append(r"\begin{tabular}{l|ccccc|c}")
    latex.append(r"\toprule")
    latex.append(r"\textbf{Model Variant} & \textbf{Acc@1} & \textbf{Acc@5} & \textbf{Acc@10} & \textbf{MRR} & \textbf{NDCG} & \textbf{$\Delta$Acc@1} \\")
    latex.append(r"\midrule")
    
    # Sort by delta
    df_sorted = df.sort_values('delta_acc1', ascending=False)
    
    for _, row in df_sorted.iterrows():
        ablation = row['ablation']
        desc = row['description']
        acc1 = row['test_acc@1']
        acc5 = row['test_acc@5']
        acc10 = row['test_acc@10']
        mrr = row['test_mrr']
        ndcg = row['test_ndcg']
        delta = row['delta_acc1']
        
        # Format delta
        if ablation == 'full':
            delta_str = "—"
            desc = r"\textbf{" + desc + r"}"
            acc1_str = f"\\textbf{{{acc1:.2f}}}"
        else:
            delta_str = f"{delta:+.2f}"
            acc1_str = f"{acc1:.2f}"
        
        latex.append(f"{desc} & {acc1_str} & {acc5:.2f} & {acc10:.2f} & {mrr:.2f} & {ndcg:.2f} & {delta_str} \\\\")
    
    latex.append(r"\bottomrule")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    
    return '\n'.join(latex)


def generate_summary_statistics(df, dataset_name, baseline_acc1):
    """Generate summary statistics for the ablation study."""
    summary = []
    summary.append(f"\n{'='*70}")
    summary.append(f"ABLATION STUDY SUMMARY - {dataset_name.upper()} DATASET")
    summary.append(f"{'='*70}")
    summary.append(f"Baseline (Full Model) Test Acc@1: {baseline_acc1:.2f}%")
    summary.append("")
    
    # Sort ablations by impact
    df_sorted = df[df['ablation'] != 'full'].copy()
    df_sorted['impact'] = baseline_acc1 - df_sorted['test_acc@1']
    df_sorted = df_sorted.sort_values('impact', ascending=False)
    
    summary.append("Component Impact Ranking (by Acc@1 drop):")
    summary.append("-" * 50)
    for i, (_, row) in enumerate(df_sorted.iterrows(), 1):
        impact = row['impact']
        pct_drop = (impact / baseline_acc1) * 100
        summary.append(f"{i}. {row['description']}: -{impact:.2f}% ({pct_drop:.1f}% relative drop)")
    
    summary.append("")
    summary.append("Key Insights:")
    summary.append("-" * 50)
    
    # Find most important component
    most_important = df_sorted.iloc[0]
    summary.append(f"• Most Critical Component: {most_important['description']}")
    summary.append(f"  - Removing it causes {most_important['impact']:.2f}% Acc@1 drop")
    
    # Find least important component
    least_important = df_sorted.iloc[-1]
    if least_important['impact'] < 0:
        summary.append(f"• Redundant Component: {least_important['description']}")
        summary.append(f"  - Removing it actually improves Acc@1 by {-least_important['impact']:.2f}%")
    else:
        summary.append(f"• Least Critical Component: {least_important['description']}")
        summary.append(f"  - Only causes {least_important['impact']:.2f}% Acc@1 drop")
    
    return '\n'.join(summary)


def main():
    """Main function to collect and analyze all results."""
    all_results = {}
    all_summaries = []
    
    gmt7 = timezone(timedelta(hours=7))
    timestamp = datetime.now(gmt7).strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"{'='*70}")
    print("ABLATION STUDY RESULTS COLLECTION")
    print(f"Timestamp: {timestamp} (GMT+7)")
    print(f"{'='*70}")
    
    for dataset_name in ['geolife', 'diy']:
        dataset_dir = RESULTS_DIR / dataset_name
        if not dataset_dir.exists():
            print(f"[WARN] No results found for {dataset_name}")
            continue
        
        print(f"\nProcessing {dataset_name.upper()} dataset...")
        
        # Collect results
        results = collect_dataset_results(dataset_dir)
        if not results:
            print(f"[WARN] No valid results for {dataset_name}")
            continue
        
        # Create DataFrame
        df = pd.DataFrame(results)
        
        # Calculate delta
        df, baseline_acc1 = calculate_delta(df)
        
        # Store results
        all_results[dataset_name] = df
        
        # Save CSV
        csv_path = dataset_dir / 'ablation_results.csv'
        df.to_csv(csv_path, index=False)
        print(f"  Saved CSV: {csv_path}")
        
        # Generate LaTeX table
        latex_table = generate_latex_table(df, dataset_name, baseline_acc1)
        latex_path = dataset_dir / 'ablation_table.tex'
        with open(latex_path, 'w') as f:
            f.write(latex_table)
        print(f"  Saved LaTeX: {latex_path}")
        
        # Generate summary
        summary = generate_summary_statistics(df, dataset_name, baseline_acc1)
        all_summaries.append(summary)
        print(summary)
        
        # Print detailed results
        print(f"\n{'='*70}")
        print(f"DETAILED RESULTS - {dataset_name.upper()}")
        print(f"{'='*70}")
        print(f"{'Ablation':<25} {'Acc@1':>8} {'Acc@5':>8} {'Acc@10':>8} {'MRR':>8} {'NDCG':>8} {'F1':>8} {'Loss':>8}")
        print("-" * 95)
        
        for _, row in df.sort_values('test_acc@1', ascending=False).iterrows():
            print(f"{row['ablation']:<25} "
                  f"{row['test_acc@1']:>7.2f}% "
                  f"{row['test_acc@5']:>7.2f}% "
                  f"{row['test_acc@10']:>7.2f}% "
                  f"{row['test_mrr']:>7.2f}% "
                  f"{row['test_ndcg']:>7.2f}% "
                  f"{row['test_f1']:>7.2f}% "
                  f"{row['test_loss']:>7.4f}")
    
    # Generate combined summary report
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE ABLATION STUDY REPORT")
    report.append("Pointer Network V45 - Next Location Prediction")
    report.append("=" * 80)
    report.append(f"\nGenerated: {timestamp} (GMT+7)")
    report.append("Random Seed: 42")
    report.append("Early Stopping Patience: 5")
    report.append("")
    
    for summary in all_summaries:
        report.append(summary)
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    report_text = '\n'.join(report)
    
    report_path = OUTPUT_DIR / 'ablation_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report_text)
    
    print(f"\n{'='*70}")
    print(f"Summary report saved to: {report_path}")
    print(f"{'='*70}")
    
    return all_results


if __name__ == "__main__":
    main()
