#!/usr/bin/env python3
"""
Comprehensive Ablation Study Runner for Pointer Generator Transformer.

This script orchestrates a full ablation study following Nature Journal standards
for scientific rigor and reproducibility. It runs multiple ablation experiments
in parallel to systematically evaluate the contribution of each model component.

Ablation Variants:
1. Full Model (baseline) - Complete PointerGeneratorTransformer
2. No Pointer Mechanism - Removes copy mechanism, only generation
3. No Generation Head - Removes vocabulary prediction, only copying
4. No Position Bias - Removes position bias in pointer attention
5. No Temporal Embeddings - Removes time/weekday/duration/recency
6. No User Embedding - Removes user personalization
7. No Position-from-End - Removes position-from-end embeddings
8. Single Transformer Layer - Reduces model depth
9. No Gate (Fixed 0.5) - Removes adaptive pointer-generation blending

Usage:
    # Run full ablation study on both datasets
    python run_ablation_study.py
    
    # Run on specific dataset
    python run_ablation_study.py --dataset geolife
    python run_ablation_study.py --dataset diy

Author: Ablation Study Script
Date: 2026-01-02
"""

import os
import sys
import json
import yaml
import time
import subprocess
import argparse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import numpy as np

# Configuration
SEED = 42
PATIENCE = 5
MAX_PARALLEL_JOBS = 3
JOB_DELAY_SECONDS = 5

# Ablation types to evaluate
ABLATION_TYPES = [
    'full',              # Baseline (complete model)
    'no_pointer',        # Remove pointer mechanism
    'no_generation',     # Remove generation head
    'no_position_bias',  # Remove position bias
    'no_temporal',       # Remove temporal embeddings
    'no_user',           # Remove user embeddings
    'no_pos_from_end',   # Remove position-from-end
    'single_layer',      # Single transformer layer
    'no_gate',           # Fixed 0.5 gate
]

# Dataset configurations
DATASET_CONFIGS = {
    'geolife': {
        'config_path': 'config/models/config_pgt_geolife_h3r8.yaml',
        'expected_acc1': 42.59,  # Expected Acc@1 for validation
    },
    'diy': {
        'config_path': 'config/models/config_pgt_diy_h3r8.yaml',
        'expected_acc1': 44.41,  # Expected Acc@1 for validation
    },
}

# Ablation descriptions for documentation
ABLATION_DESCRIPTIONS = {
    'full': 'Complete model with all components (baseline)',
    'no_pointer': 'Remove pointer mechanism (copy from history)',
    'no_generation': 'Remove generation head (vocabulary prediction)',
    'no_position_bias': 'Remove position bias in pointer attention',
    'no_temporal': 'Remove temporal embeddings (time, weekday, duration, recency)',
    'no_user': 'Remove user embedding (personalization)',
    'no_pos_from_end': 'Remove position-from-end embedding (recency awareness)',
    'single_layer': 'Reduce to single transformer encoder layer',
    'no_gate': 'Replace adaptive gate with fixed 0.5 blending',
}


def get_script_dir():
    """Get the directory containing this script."""
    return Path(__file__).parent.absolute()


def run_single_ablation(config_path, ablation_type, output_dir, seed=42):
    """Run a single ablation experiment."""
    script_dir = get_script_dir()
    train_script = script_dir / 'train_ablation.py'
    
    cmd = [
        'python', str(train_script),
        '--config', config_path,
        '--ablation', ablation_type,
        '--output_dir', output_dir,
        '--seed', str(seed),
    ]
    
    print(f"\n[INFO] Starting ablation: {ablation_type}")
    print(f"[INFO] Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(script_dir.parent.parent),  # Project root
        )
        
        if result.returncode != 0:
            print(f"[ERROR] Ablation {ablation_type} failed:")
            print(result.stderr)
            return None
        
        print(f"[SUCCESS] Completed ablation: {ablation_type}")
        return ablation_type
        
    except Exception as e:
        print(f"[ERROR] Exception in {ablation_type}: {e}")
        return None


def collect_results(output_dir, dataset_name):
    """Collect results from all ablation experiments."""
    results = []
    
    for ablation_type in ABLATION_TYPES:
        # Find the latest experiment directory for this ablation
        pattern = f"ablation_{dataset_name}_{ablation_type}_*"
        matching_dirs = sorted(Path(output_dir).glob(pattern), reverse=True)
        
        if not matching_dirs:
            print(f"[WARN] No results found for {ablation_type}")
            continue
        
        exp_dir = matching_dirs[0]
        results_file = exp_dir / 'results.json'
        
        if not results_file.exists():
            print(f"[WARN] No results.json in {exp_dir}")
            continue
        
        with open(results_file, 'r') as f:
            result = json.load(f)
        
        results.append({
            'ablation': ablation_type,
            'description': ABLATION_DESCRIPTIONS.get(ablation_type, ''),
            'parameters': result.get('parameters', 0),
            **{f'val_{k}': v for k, v in result.get('validation', {}).items()},
            **{f'test_{k}': v for k, v in result.get('test', {}).items()},
        })
    
    return results


def generate_latex_table(results_df, dataset_name):
    """Generate LaTeX table for Nature Journal format."""
    latex = []
    latex.append(r"\begin{table}[htbp]")
    latex.append(r"\centering")
    latex.append(f"\\caption{{Ablation Study Results on {dataset_name.upper()} Dataset}}")
    latex.append(f"\\label{{tab:ablation_{dataset_name}}}")
    latex.append(r"\begin{tabular}{l|ccccc|c}")
    latex.append(r"\hline")
    latex.append(r"Ablation & Acc@1 & Acc@5 & Acc@10 & MRR & NDCG & $\Delta$Acc@1 \\")
    latex.append(r"\hline")
    
    # Get baseline accuracy
    baseline_acc1 = results_df[results_df['ablation'] == 'full']['test_acc@1'].values[0]
    
    for _, row in results_df.iterrows():
        ablation = row['ablation']
        acc1 = row['test_acc@1']
        acc5 = row['test_acc@5']
        acc10 = row['test_acc@10']
        mrr = row['test_mrr']
        ndcg = row['test_ndcg']
        delta = acc1 - baseline_acc1
        
        # Format ablation name
        ablation_display = ablation.replace('_', ' ').title()
        if ablation == 'full':
            ablation_display = r"\textbf{Full Model (Baseline)}"
        
        # Format delta
        delta_str = f"{delta:+.2f}" if delta != 0 else "—"
        
        latex.append(f"{ablation_display} & {acc1:.2f} & {acc5:.2f} & {acc10:.2f} & {mrr:.2f} & {ndcg:.2f} & {delta_str} \\\\")
    
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table}")
    
    return '\n'.join(latex)


def generate_summary_report(all_results, output_dir):
    """Generate comprehensive summary report."""
    report = []
    report.append("=" * 80)
    report.append("ABLATION STUDY SUMMARY REPORT")
    report.append("Pointer Generator Transformer - Next Location Prediction")
    report.append("=" * 80)
    report.append("")
    
    gmt7 = timezone(timedelta(hours=7))
    report.append(f"Generated: {datetime.now(gmt7).strftime('%Y-%m-%d %H:%M:%S')} (GMT+7)")
    report.append(f"Random Seed: {SEED}")
    report.append(f"Early Stopping Patience: {PATIENCE}")
    report.append("")
    
    for dataset_name, results in all_results.items():
        if not results:
            continue
        
        df = pd.DataFrame(results)
        
        report.append(f"\n{'='*80}")
        report.append(f"DATASET: {dataset_name.upper()}")
        report.append(f"{'='*80}\n")
        
        # Get baseline
        baseline = df[df['ablation'] == 'full']
        if not baseline.empty:
            baseline_acc1 = baseline['test_acc@1'].values[0]
            report.append(f"Baseline (Full Model) Test Acc@1: {baseline_acc1:.2f}%")
            report.append("")
        
        # Results table
        report.append("Test Set Results:")
        report.append("-" * 80)
        report.append(f"{'Ablation':<20} {'Acc@1':>8} {'Acc@5':>8} {'Acc@10':>8} {'MRR':>8} {'NDCG':>8} {'F1':>8} {'ΔAcc@1':>8}")
        report.append("-" * 80)
        
        for _, row in df.iterrows():
            delta = row['test_acc@1'] - baseline_acc1 if not baseline.empty else 0
            delta_str = f"{delta:+.2f}" if row['ablation'] != 'full' else "—"
            
            report.append(
                f"{row['ablation']:<20} "
                f"{row['test_acc@1']:>7.2f}% "
                f"{row['test_acc@5']:>7.2f}% "
                f"{row['test_acc@10']:>7.2f}% "
                f"{row['test_mrr']:>7.2f}% "
                f"{row['test_ndcg']:>7.2f}% "
                f"{row['test_f1']:>7.2f}% "
                f"{delta_str:>8}"
            )
        
        report.append("-" * 80)
        report.append("")
        
        # Component importance ranking
        if not baseline.empty:
            report.append("Component Importance Ranking (by Acc@1 drop):")
            report.append("-" * 50)
            
            importance = []
            for _, row in df.iterrows():
                if row['ablation'] != 'full':
                    drop = baseline_acc1 - row['test_acc@1']
                    importance.append((row['ablation'], drop))
            
            importance.sort(key=lambda x: x[1], reverse=True)
            
            for i, (ablation, drop) in enumerate(importance, 1):
                report.append(f"{i}. {ablation}: -{drop:.2f}%")
            
            report.append("")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    return '\n'.join(report)


def run_ablation_study(datasets=None, output_base_dir=None):
    """Run the complete ablation study."""
    script_dir = get_script_dir()
    project_root = script_dir.parent.parent
    
    if output_base_dir is None:
        output_base_dir = script_dir / 'results'
    else:
        output_base_dir = Path(output_base_dir)
    
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    if datasets is None:
        datasets = list(DATASET_CONFIGS.keys())
    
    all_results = {}
    
    for dataset_name in datasets:
        if dataset_name not in DATASET_CONFIGS:
            print(f"[ERROR] Unknown dataset: {dataset_name}")
            continue
        
        dataset_config = DATASET_CONFIGS[dataset_name]
        config_path = str(project_root / dataset_config['config_path'])
        output_dir = str(output_base_dir / dataset_name)
        
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"RUNNING ABLATION STUDY FOR: {dataset_name.upper()}")
        print(f"Config: {config_path}")
        print(f"Output: {output_dir}")
        print(f"Expected Acc@1: {dataset_config['expected_acc1']}%")
        print(f"{'='*60}\n")
        
        # Run ablations in batches of MAX_PARALLEL_JOBS
        completed = []
        
        for batch_start in range(0, len(ABLATION_TYPES), MAX_PARALLEL_JOBS):
            batch = ABLATION_TYPES[batch_start:batch_start + MAX_PARALLEL_JOBS]
            
            print(f"\n[INFO] Running batch: {batch}")
            
            processes = []
            for i, ablation_type in enumerate(batch):
                if i > 0:
                    print(f"[INFO] Waiting {JOB_DELAY_SECONDS}s before starting next job...")
                    time.sleep(JOB_DELAY_SECONDS)
                
                # Start subprocess
                train_script = script_dir / 'train_ablation.py'
                cmd = [
                    'python', str(train_script),
                    '--config', config_path,
                    '--ablation', ablation_type,
                    '--output_dir', output_dir,
                    '--seed', str(SEED),
                ]
                
                print(f"[INFO] Starting: {ablation_type}")
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    cwd=str(project_root),
                )
                processes.append((ablation_type, proc))
            
            # Wait for all processes in batch to complete
            for ablation_type, proc in processes:
                stdout, stderr = proc.communicate()
                
                if proc.returncode == 0:
                    print(f"[SUCCESS] Completed: {ablation_type}")
                    completed.append(ablation_type)
                else:
                    print(f"[ERROR] Failed: {ablation_type}")
                    print(stderr[:500] if stderr else "No error message")
        
        # Collect results for this dataset
        results = collect_results(output_dir, dataset_name)
        all_results[dataset_name] = results
        
        # Save individual dataset results
        if results:
            df = pd.DataFrame(results)
            df.to_csv(Path(output_dir) / 'ablation_results.csv', index=False)
            
            # Generate LaTeX table
            latex_table = generate_latex_table(df, dataset_name)
            with open(Path(output_dir) / 'ablation_table.tex', 'w') as f:
                f.write(latex_table)
    
    # Generate summary report
    report = generate_summary_report(all_results, output_base_dir)
    report_path = output_base_dir / 'ablation_summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"\n{'='*60}")
    print("ABLATION STUDY COMPLETE")
    print(f"{'='*60}")
    print(f"Summary report saved to: {report_path}")
    print(report)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive ablation study for Pointer Generator Transformer")
    parser.add_argument("--dataset", type=str, choices=['geolife', 'diy', 'all'], default='all',
                       help="Dataset to run ablation on")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for results")
    args = parser.parse_args()
    
    if args.dataset == 'all':
        datasets = None
    else:
        datasets = [args.dataset]
    
    run_ablation_study(datasets=datasets, output_base_dir=args.output_dir)


if __name__ == "__main__":
    main()
