"""
Run All Analyses

This script runs all analysis modules and generates a comprehensive report.
"""

import os
import sys
from pathlib import Path

# Add the scripts directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import analysis modules
import importlib.util

def run_analysis(script_name):
    """Run an analysis script and return results."""
    print(f"\n{'='*70}")
    print(f"Running: {script_name}")
    print(f"{'='*70}")
    
    spec = importlib.util.spec_from_file_location(
        script_name.replace('.py', ''),
        Path(__file__).parent / script_name
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    if hasattr(module, 'main'):
        return module.main()
    return None


def main():
    """Run all analyses."""
    print("=" * 70)
    print("COMPREHENSIVE ANALYSIS: GEOLIFE vs DIY IMPROVEMENT GAP")
    print("=" * 70)
    print("""
Research Question:
Why does PGT show a larger improvement over MHSA on Geolife (+20.78%)
compared to DIY (+3.71%)?

Analysis Approach:
1. Data Characteristics Analysis - Understanding dataset differences
2. Pointer Mechanism Analysis - Why Pointer works better on Geolife
3. Model Performance Analysis - Deep dive into performance metrics
""")
    
    # Run all analyses
    analyses = [
        '01_data_characteristics_analysis.py',
        '02_pointer_mechanism_analysis.py',
        '03_model_performance_analysis.py',
    ]
    
    results = {}
    for script in analyses:
        try:
            results[script] = run_analysis(script)
        except Exception as e:
            print(f"Error running {script}: {e}")
            import traceback
            traceback.print_exc()
    
    # Print final summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nResults saved to: scripts/analysis_geolife_vs_diy/results/")
    print("\nKey files generated:")
    
    results_dir = Path("scripts/analysis_geolife_vs_diy/results")
    if results_dir.exists():
        for f in sorted(results_dir.glob("*")):
            print(f"  - {f.name}")
    
    print("\n" + "=" * 70)
    
    return results


if __name__ == "__main__":
    main()
