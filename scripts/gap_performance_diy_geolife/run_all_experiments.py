"""
Master Script for Gap Performance Analysis.

This script orchestrates all experiments and generates the final comprehensive report.

Experiments:
1. Mobility Pattern Analysis - Location revisit, entropy, frequency
2. Model-Based Pointer Analysis - Gate values, attention patterns
3. Recency Pattern Analysis - Position from end, return patterns

Author: Gap Performance Analysis Framework
Date: January 2, 2026
Seed: 42
"""

import os
import sys
import subprocess
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent

def run_script(script_name: str):
    """Run a Python script and return success status."""
    script_path = PROJECT_ROOT / 'scripts' / 'gap_performance_diy_geolife' / script_name
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print('='*70)
    
    result = subprocess.run(
        ['python', str(script_path)],
        cwd=str(PROJECT_ROOT),
        capture_output=False
    )
    
    return result.returncode == 0


def main():
    """Run all analyses."""
    print("="*70)
    print("GAP PERFORMANCE ANALYSIS - MASTER SCRIPT")
    print("Why Pointer Mechanism Benefits GeoLife More Than DIY")
    print("="*70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scripts = [
        'analyze_mobility_patterns.py',
        'analyze_model_pointer.py',
        'analyze_recency_patterns.py',
    ]
    
    results = {}
    for script in scripts:
        success = run_script(script)
        results[script] = success
        if not success:
            print(f"WARNING: {script} failed!")
    
    print("\n" + "="*70)
    print("ALL ANALYSES COMPLETE")
    print("="*70)
    
    print("\nResults Summary:")
    for script, success in results.items():
        status = "✓ Success" if success else "✗ Failed"
        print(f"  {script}: {status}")
    
    # List generated files
    output_dir = PROJECT_ROOT / 'scripts' / 'gap_performance_diy_geolife' / 'results'
    
    print(f"\nGenerated Files in {output_dir}:")
    
    # Tables
    tables_dir = output_dir / 'tables'
    if tables_dir.exists():
        print("\n  Tables:")
        for f in tables_dir.glob('*'):
            print(f"    - {f.name}")
    
    # Figures
    figures_dir = output_dir / 'figures'
    if figures_dir.exists():
        print("\n  Figures:")
        for f in figures_dir.glob('*'):
            print(f"    - {f.name}")
    
    # JSON results
    print("\n  JSON Results:")
    for f in output_dir.glob('*.json'):
        print(f"    - {f.name}")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)


if __name__ == '__main__':
    main()
