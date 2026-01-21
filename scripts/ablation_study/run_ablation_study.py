#!/usr/bin/env python3
"""
Comprehensive Ablation Study Runner for Pointer Generator Transformer.

This script orchestrates a complete ablation study following Nature Journal standards.
It runs multiple ablation experiments in parallel (3 at a time) to evaluate the
contribution of each model component to the overall performance.

Features:
- Parallel execution of 3 training sessions with 5-second delays
- Systematic evaluation of all model components
- Comprehensive metrics collection (Acc@1, Acc@5, Acc@10, MRR, NDCG, F1, Loss)
- Aggregated results in publication-ready format
- Reproducibility with seed=42 and patience=5

Usage:
    # Run full ablation study on both datasets
    python scripts/ablation_study/run_ablation_study.py
    
    # Run on specific dataset
    python scripts/ablation_study/run_ablation_study.py --dataset geolife
    python scripts/ablation_study/run_ablation_study.py --dataset diy

Author: Ablation Study Framework
Date: December 2024
"""

import os
import sys
import json
import yaml
import subprocess
import time
import argparse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# =============================================================================
# Configuration
# =============================================================================

# Ablation experiments to run
ABLATION_EXPERIMENTS = [
    'full_model',           # Baseline: Full model
    'no_user_emb',          # Component 1: User Embedding
    'no_time_emb',          # Component 2: Time-of-Day Embedding
    'no_weekday_emb',       # Component 3: Weekday Embedding
    'no_recency_emb',       # Component 4: Recency Embedding
    'no_duration_emb',      # Component 5: Duration Embedding
    'no_pos_from_end',      # Component 6: Position-from-End Embedding
    'no_sinusoidal_pos',    # Component 7: Sinusoidal Positional Encoding
    'no_temporal',          # Component Group: All Temporal Features
    'no_pointer',           # Component 8: Pointer Mechanism
    'no_generation',        # Component 9: Generation Head
    'no_gate',              # Component 10: Pointer-Generation Gate
    'single_layer',         # Architecture: Single Transformer Layer
]

# Dataset configurations
DATASET_CONFIGS = {
    'geolife': {
        'config_path': 'config/models/config_pgt_geolife.yaml',
        'expected_acc1': 53.96,
        'description': 'GeoLife GPS Trajectory Dataset',
    },
    'diy': {
        'config_path': 'config/models/config_pgt_diy.yaml',
        'expected_acc1': 56.88,
        'description': 'DIY Location Check-in Dataset',
    },
}

# Ablation descriptions for documentation
ABLATION_DESCRIPTIONS = {
    'full_model': 'Complete Pointer Generator Transformer model with all components enabled',
    'no_user_emb': 'Model without user embedding (removes personalization)',
    'no_time_emb': 'Model without time-of-day embedding (removes circadian patterns)',
    'no_weekday_emb': 'Model without weekday embedding (removes weekly patterns)',
    'no_recency_emb': 'Model without recency embedding (removes temporal decay)',
    'no_duration_emb': 'Model without duration embedding (removes visit duration features)',
    'no_pos_from_end': 'Model without position-from-end embedding (removes sequence position awareness)',
    'no_sinusoidal_pos': 'Model without sinusoidal positional encoding (removes absolute position)',
    'no_temporal': 'Model without all temporal features (time, weekday, recency, duration)',
    'no_pointer': 'Model without pointer mechanism (generation head only)',
    'no_generation': 'Model without generation head (pointer mechanism only)',
    'no_gate': 'Model without adaptive gate (fixed 50-50 blend of pointer and generation)',
    'single_layer': 'Model with single transformer layer (reduced model capacity)',
}

# Number of parallel jobs
MAX_PARALLEL_JOBS = 3
DELAY_BETWEEN_JOBS = 5  # seconds


# =============================================================================
# Utility Functions
# =============================================================================

def get_timestamp():
    """Get current timestamp in GMT+7."""
    gmt7 = timezone(timedelta(hours=7))
    return datetime.now(gmt7).strftime("%Y%m%d_%H%M%S")


def run_single_ablation(config_path: str, ablation_name: str, job_id: int, 
                        results_dir: str, conda_env: str = "mlenv") -> Dict:
    """
    Run a single ablation experiment.
    
    Args:
        config_path: Path to config YAML
        ablation_name: Name of ablation experiment
        job_id: Unique job identifier
        results_dir: Directory to store results
        conda_env: Conda environment name
        
    Returns:
        Dictionary with results and metadata
    """
    start_time = time.time()
    
    # Build command
    script_path = "scripts/ablation_study/train_ablation.py"
    cmd = f"""
    source ~/miniconda3/etc/profile.d/conda.sh && \
    conda activate {conda_env} && \
    python {script_path} --config {config_path} --ablation {ablation_name}
    """
    
    print(f"[Job {job_id}] Starting: {ablation_name}")
    
    # Run experiment
    result = subprocess.run(
        cmd,
        shell=True,
        executable='/bin/bash',
        capture_output=True,
        text=True,
        cwd=str(Path(__file__).parent.parent.parent),
    )
    
    elapsed_time = time.time() - start_time
    
    # Parse results
    success = result.returncode == 0
    
    result_dict = {
        'ablation_name': ablation_name,
        'job_id': job_id,
        'success': success,
        'elapsed_time': elapsed_time,
        'stdout': result.stdout,
        'stderr': result.stderr,
    }
    
    if success:
        print(f"[Job {job_id}] Completed: {ablation_name} ({elapsed_time:.1f}s)")
    else:
        print(f"[Job {job_id}] FAILED: {ablation_name}")
        print(f"Error: {result.stderr[-500:] if result.stderr else 'No error output'}")
    
    return result_dict


def collect_results(results_base_dir: str, dataset_name: str) -> Dict:
    """
    Collect all ablation results from experiment directories.
    
    Args:
        results_base_dir: Base directory containing ablation results
        dataset_name: Name of dataset (geolife or diy)
        
    Returns:
        Dictionary mapping ablation names to their results
    """
    results = {}
    ablation_dir = Path(results_base_dir) / "ablation_study"
    
    if not ablation_dir.exists():
        return results
    
    for exp_dir in ablation_dir.iterdir():
        if not exp_dir.is_dir():
            continue
        
        # Check if this is for the correct dataset
        if dataset_name not in exp_dir.name:
            continue
        
        # Load test results
        test_results_path = exp_dir / "test_results.json"
        ablation_info_path = exp_dir / "ablation_info.json"
        
        if test_results_path.exists() and ablation_info_path.exists():
            with open(test_results_path, 'r') as f:
                test_results = json.load(f)
            with open(ablation_info_path, 'r') as f:
                ablation_info = json.load(f)
            
            ablation_name = ablation_info['ablation_name']
            
            # Store most recent result for each ablation
            if ablation_name not in results or exp_dir.name > results[ablation_name]['exp_dir']:
                results[ablation_name] = {
                    'test_results': test_results,
                    'ablation_info': ablation_info,
                    'exp_dir': exp_dir.name,
                }
    
    return results


def format_results_table(results: Dict, dataset_name: str, full_model_results: Dict) -> str:
    """
    Format results as a publication-ready table.
    
    Args:
        results: Dictionary of ablation results
        dataset_name: Name of dataset
        full_model_results: Results from full model for comparison
        
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 120)
    lines.append(f"ABLATION STUDY RESULTS - {dataset_name.upper()}")
    lines.append("=" * 120)
    lines.append("")
    
    # Header
    header = f"{'Ablation':<30} {'Acc@1':>8} {'Acc@5':>8} {'Acc@10':>8} {'MRR':>8} {'NDCG':>8} {'F1':>8} {'Loss':>8} {'Δ Acc@1':>10}"
    lines.append(header)
    lines.append("-" * 120)
    
    # Get full model accuracy for delta calculation
    full_acc1 = full_model_results.get('test_results', {}).get('acc@1', 0) if full_model_results else 0
    
    # Sort by ablation type
    sorted_ablations = sorted(results.keys(), key=lambda x: ABLATION_EXPERIMENTS.index(x) if x in ABLATION_EXPERIMENTS else 999)
    
    for ablation_name in sorted_ablations:
        data = results[ablation_name]
        test = data['test_results']
        
        acc1 = test.get('acc@1', 0)
        acc5 = test.get('acc@5', 0)
        acc10 = test.get('acc@10', 0)
        mrr = test.get('mrr', 0)
        ndcg = test.get('ndcg', 0)
        f1 = test.get('f1', 0) * 100 if test.get('f1', 0) < 1 else test.get('f1', 0)
        loss = test.get('loss', 0)
        
        delta = acc1 - full_acc1 if ablation_name != 'full_model' else 0
        delta_str = f"{delta:+.2f}%" if ablation_name != 'full_model' else "-"
        
        row = f"{ablation_name:<30} {acc1:>8.2f} {acc5:>8.2f} {acc10:>8.2f} {mrr:>8.2f} {ndcg:>8.2f} {f1:>8.2f} {loss:>8.4f} {delta_str:>10}"
        lines.append(row)
    
    lines.append("=" * 120)
    lines.append("")
    lines.append("Notes:")
    lines.append("- Δ Acc@1: Change in Accuracy@1 compared to full model (negative = worse)")
    lines.append("- All metrics except Loss are percentages")
    lines.append("- Experiments conducted with seed=42 and patience=5")
    lines.append("")
    
    return "\n".join(lines)


def generate_summary_report(geolife_results: Dict, diy_results: Dict, output_dir: str) -> str:
    """
    Generate comprehensive summary report.
    
    Args:
        geolife_results: Results for GeoLife dataset
        diy_results: Results for DIY dataset
        output_dir: Directory to save report
        
    Returns:
        Path to generated report
    """
    report_lines = []
    report_lines.append("=" * 120)
    report_lines.append("COMPREHENSIVE ABLATION STUDY REPORT")
    report_lines.append("Pointer Generator Transformer for Next Location Prediction")
    report_lines.append(f"Generated: {get_timestamp()}")
    report_lines.append("=" * 120)
    report_lines.append("")
    
    # GeoLife Results
    if geolife_results:
        full_geo = geolife_results.get('full_model', {})
        report_lines.append(format_results_table(geolife_results, 'geolife', full_geo))
    
    # DIY Results
    if diy_results:
        full_diy = diy_results.get('full_model', {})
        report_lines.append(format_results_table(diy_results, 'diy', full_diy))
    
    # Component Importance Analysis
    report_lines.append("")
    report_lines.append("=" * 120)
    report_lines.append("COMPONENT IMPORTANCE ANALYSIS")
    report_lines.append("=" * 120)
    report_lines.append("")
    
    for dataset_name, results in [('GeoLife', geolife_results), ('DIY', diy_results)]:
        if not results:
            continue
            
        report_lines.append(f"\n{dataset_name} Dataset:")
        report_lines.append("-" * 60)
        
        full_acc = results.get('full_model', {}).get('test_results', {}).get('acc@1', 0)
        
        # Calculate importance (delta from full model)
        importance = []
        for abl, data in results.items():
            if abl == 'full_model':
                continue
            acc = data.get('test_results', {}).get('acc@1', 0)
            delta = full_acc - acc
            importance.append((abl, delta))
        
        # Sort by importance (largest drop = most important)
        importance.sort(key=lambda x: -x[1])
        
        report_lines.append(f"  Full Model Acc@1: {full_acc:.2f}%")
        report_lines.append("")
        report_lines.append("  Component Importance Ranking (by Acc@1 drop):")
        for i, (abl, delta) in enumerate(importance, 1):
            report_lines.append(f"    {i:2d}. {abl:<25} Δ = {delta:+.2f}%")
    
    report = "\n".join(report_lines)
    
    # Save report
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, f"ablation_study_report_{get_timestamp()}.txt")
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Also save as JSON for programmatic access
    json_results = {
        'geolife': {k: v['test_results'] for k, v in geolife_results.items()} if geolife_results else {},
        'diy': {k: v['test_results'] for k, v in diy_results.items()} if diy_results else {},
        'timestamp': get_timestamp(),
    }
    json_path = os.path.join(output_dir, f"ablation_study_results_{get_timestamp()}.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    return report_path


# =============================================================================
# Main Runner
# =============================================================================

def run_ablation_study(dataset: str = 'all', conda_env: str = 'mlenv'):
    """
    Run the complete ablation study.
    
    Args:
        dataset: 'geolife', 'diy', or 'all'
        conda_env: Conda environment name
    """
    print("=" * 80)
    print("COMPREHENSIVE ABLATION STUDY FOR POINTER NETWORK V45")
    print("Nature Journal Standard Ablation Analysis")
    print("=" * 80)
    print(f"Timestamp: {get_timestamp()}")
    print(f"Parallel Jobs: {MAX_PARALLEL_JOBS}")
    print(f"Delay Between Jobs: {DELAY_BETWEEN_JOBS}s")
    print(f"Seed: 42, Patience: 5")
    print("=" * 80)
    
    # Determine which datasets to run
    datasets_to_run = []
    if dataset == 'all':
        datasets_to_run = ['geolife', 'diy']
    else:
        datasets_to_run = [dataset]
    
    project_root = Path(__file__).parent.parent.parent
    results_dir = project_root / "experiments"
    
    all_results = {'geolife': {}, 'diy': {}}
    
    for ds_name in datasets_to_run:
        print(f"\n{'='*80}")
        print(f"Running ablation study on {ds_name.upper()} dataset")
        print(f"{'='*80}")
        
        config_path = DATASET_CONFIGS[ds_name]['config_path']
        expected_acc1 = DATASET_CONFIGS[ds_name]['expected_acc1']
        
        print(f"Config: {config_path}")
        print(f"Expected Full Model Acc@1: {expected_acc1}%")
        print(f"Ablation Experiments: {len(ABLATION_EXPERIMENTS)}")
        
        # Create job queue
        jobs = []
        for i, ablation in enumerate(ABLATION_EXPERIMENTS):
            jobs.append({
                'config_path': config_path,
                'ablation_name': ablation,
                'job_id': i,
            })
        
        # Run experiments in parallel batches
        completed_jobs = []
        job_counter = [0]  # Use list to allow modification in closure
        lock = threading.Lock()
        
        def run_with_delay(job):
            # Add staggered delay based on job position within batch
            with lock:
                delay = (job_counter[0] % MAX_PARALLEL_JOBS) * DELAY_BETWEEN_JOBS
                job_counter[0] += 1
            
            if delay > 0:
                time.sleep(delay)
            
            return run_single_ablation(
                config_path=job['config_path'],
                ablation_name=job['ablation_name'],
                job_id=job['job_id'],
                results_dir=str(results_dir),
                conda_env=conda_env,
            )
        
        print(f"\nStarting {len(jobs)} experiments with {MAX_PARALLEL_JOBS} parallel workers...")
        
        with ThreadPoolExecutor(max_workers=MAX_PARALLEL_JOBS) as executor:
            futures = {executor.submit(run_with_delay, job): job for job in jobs}
            
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                    completed_jobs.append(result)
                except Exception as e:
                    print(f"Job {job['ablation_name']} generated an exception: {e}")
                    completed_jobs.append({
                        'ablation_name': job['ablation_name'],
                        'success': False,
                        'error': str(e),
                    })
        
        # Collect results
        print(f"\nCollecting results for {ds_name}...")
        all_results[ds_name] = collect_results(str(results_dir), ds_name)
        
        print(f"Collected {len(all_results[ds_name])} ablation results")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("GENERATING SUMMARY REPORT")
    print("=" * 80)
    
    report_dir = project_root / "experiments" / "ablation_study" / "reports"
    report_path = generate_summary_report(
        all_results.get('geolife', {}),
        all_results.get('diy', {}),
        str(report_dir),
    )
    
    print(f"\nReport saved to: {report_path}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY COMPLETE")
    print("=" * 80)
    
    for ds_name in datasets_to_run:
        results = all_results.get(ds_name, {})
        if results:
            full_result = results.get('full_model', {}).get('test_results', {})
            if full_result:
                print(f"\n{ds_name.upper()} Full Model Results:")
                print(f"  Acc@1:  {full_result.get('acc@1', 0):.2f}%")
                print(f"  Acc@5:  {full_result.get('acc@5', 0):.2f}%")
                print(f"  Acc@10: {full_result.get('acc@10', 0):.2f}%")
                print(f"  MRR:    {full_result.get('mrr', 0):.2f}%")
                print(f"  NDCG:   {full_result.get('ndcg', 0):.2f}%")


def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive ablation study for Pointer Generator Transformer model"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=['geolife', 'diy', 'all'],
        default='all',
        help="Dataset to run ablation on (default: all)"
    )
    parser.add_argument(
        "--conda-env",
        type=str,
        default='mlenv',
        help="Conda environment name (default: mlenv)"
    )
    args = parser.parse_args()
    
    run_ablation_study(dataset=args.dataset, conda_env=args.conda_env)


if __name__ == "__main__":
    main()
