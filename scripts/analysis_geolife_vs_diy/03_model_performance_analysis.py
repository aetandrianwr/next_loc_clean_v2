"""
Model Performance Deep Dive Analysis

This script analyzes model performance differences between MHSA and PGT
on both datasets to understand the improvement gap.

Analysis includes:
1. Accuracy breakdown by sample characteristics
2. Performance on copyable vs non-copyable samples
3. Error analysis
4. Per-user performance comparison
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path("scripts/analysis_geolife_vs_diy/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_experiment_results():
    """Load experiment results from both models."""
    results = {}
    
    # Geolife MHSA
    with open("experiments/geolife_MHSA_20251228_230813/test_results.json", "r") as f:
        results['geolife_mhsa'] = json.load(f)
    
    # DIY MHSA
    with open("experiments/diy_MHSA_20251226_192959/test_results.json", "r") as f:
        results['diy_mhsa'] = json.load(f)
    
    return results


def load_test_data():
    """Load test data for both datasets."""
    with open("data/geolife_eps20/processed/geolife_eps20_prev7_test.pk", "rb") as f:
        geolife_test = pickle.load(f)
    with open("data/diy_eps50/processed/diy_eps50_prev7_test.pk", "rb") as f:
        diy_test = pickle.load(f)
    
    return {'geolife': geolife_test, 'diy': diy_test}


def analyze_sample_difficulty(test_data):
    """
    Categorize samples by difficulty based on characteristics.
    
    Difficulty factors:
    1. Target in history (easy for Pointer)
    2. Target frequency in history
    3. Target recency
    4. Sequence length
    5. Number of unique locations in history
    """
    print("\n" + "=" * 70)
    print("SAMPLE DIFFICULTY ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    for name, data in test_data.items():
        easy_samples = []  # High pointer benefit potential
        medium_samples = []
        hard_samples = []  # Low pointer benefit potential
        
        sample_details = []
        
        for i, seq in enumerate(data):
            history = list(seq['X'])
            target = seq['Y']
            
            # Calculate difficulty factors
            target_in_history = target in history
            target_freq = history.count(target) if target_in_history else 0
            
            if target_in_history:
                positions = [j for j, x in enumerate(history) if x == target]
                recency = len(history) - max(positions) - 1
            else:
                recency = float('inf')
            
            seq_length = len(history)
            unique_locs = len(set(history))
            repetition_ratio = 1 - (unique_locs / seq_length)
            
            # Calculate difficulty score (lower = easier for pointer)
            if not target_in_history:
                difficulty = 'hard'
                hard_samples.append(i)
            elif recency <= 2 and target_freq >= 2:
                difficulty = 'easy'
                easy_samples.append(i)
            elif recency <= 5:
                difficulty = 'medium'
                medium_samples.append(i)
            else:
                difficulty = 'medium'
                medium_samples.append(i)
            
            sample_details.append({
                'idx': i,
                'target_in_history': target_in_history,
                'target_freq': target_freq,
                'recency': recency if recency != float('inf') else -1,
                'seq_length': seq_length,
                'unique_locs': unique_locs,
                'repetition_ratio': repetition_ratio,
                'difficulty': difficulty,
            })
        
        df_details = pd.DataFrame(sample_details)
        
        results[name] = {
            'total_samples': len(data),
            'easy_samples': len(easy_samples),
            'medium_samples': len(medium_samples),
            'hard_samples': len(hard_samples),
            'easy_ratio': len(easy_samples) / len(data),
            'medium_ratio': len(medium_samples) / len(data),
            'hard_ratio': len(hard_samples) / len(data),
            'avg_seq_length': df_details['seq_length'].mean(),
            'avg_unique_locs': df_details['unique_locs'].mean(),
            'avg_repetition_ratio': df_details['repetition_ratio'].mean(),
            'details': df_details,
        }
    
    # Print comparison
    print("\nSample Difficulty Distribution:")
    print("-" * 50)
    print(f"{'Category':<20} {'Geolife':<15} {'DIY':<15}")
    print("-" * 50)
    
    for cat in ['easy', 'medium', 'hard']:
        geo_val = results['geolife'][f'{cat}_ratio'] * 100
        diy_val = results['diy'][f'{cat}_ratio'] * 100
        print(f"{cat.capitalize():<20} {geo_val:>6.2f}%{'':<8} {diy_val:>6.2f}%")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Difficulty distribution comparison
    ax1 = axes[0]
    categories = ['Easy\n(Recent + Frequent)', 'Medium\n(In History)', 'Hard\n(Not in History)']
    geo_vals = [results['geolife']['easy_ratio']*100, 
                results['geolife']['medium_ratio']*100,
                results['geolife']['hard_ratio']*100]
    diy_vals = [results['diy']['easy_ratio']*100,
                results['diy']['medium_ratio']*100,
                results['diy']['hard_ratio']*100]
    
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, geo_vals, width, label='Geolife', color='#3498db')
    ax1.bar(x + width/2, diy_vals, width, label='DIY', color='#e74c3c')
    ax1.set_ylabel('Percentage of Samples')
    ax1.set_title('Sample Difficulty Distribution')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    
    # Add percentage labels
    for i, (g, d) in enumerate(zip(geo_vals, diy_vals)):
        ax1.text(i - width/2, g + 1, f'{g:.1f}%', ha='center', fontsize=9)
        ax1.text(i + width/2, d + 1, f'{d:.1f}%', ha='center', fontsize=9)
    
    # 2. Repetition ratio distribution
    ax2 = axes[1]
    ax2.hist(results['geolife']['details']['repetition_ratio'], bins=30, alpha=0.7, 
             label='Geolife', color='#3498db')
    ax2.hist(results['diy']['details']['repetition_ratio'], bins=30, alpha=0.7,
             label='DIY', color='#e74c3c')
    ax2.set_xlabel('Repetition Ratio (1 - unique/total)')
    ax2.set_ylabel('Frequency')
    ax2.set_title('Sequence Repetition Pattern')
    ax2.legend()
    
    # 3. Unique locations per sequence
    ax3 = axes[2]
    ax3.hist(results['geolife']['details']['unique_locs'], bins=30, alpha=0.7,
             label='Geolife', color='#3498db')
    ax3.hist(results['diy']['details']['unique_locs'], bins=30, alpha=0.7,
             label='DIY', color='#e74c3c')
    ax3.set_xlabel('Unique Locations in Sequence')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Sequence Diversity')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "11_sample_difficulty_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save summary
    summary = {
        'geolife': {k: v for k, v in results['geolife'].items() if k != 'details'},
        'diy': {k: v for k, v in results['diy'].items() if k != 'details'}
    }
    with open(OUTPUT_DIR / "11_sample_difficulty.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return results


def analyze_theoretical_accuracy_breakdown():
    """
    Break down expected accuracy improvement by sample category.
    
    Hypothesis:
    - Easy samples: Pointer should significantly outperform MHSA
    - Medium samples: Pointer should moderately outperform MHSA
    - Hard samples: Both models should perform similarly (pure generation)
    """
    print("\n" + "=" * 70)
    print("THEORETICAL ACCURACY BREAKDOWN")
    print("=" * 70)
    
    # Actual results from experiments
    actual_results = {
        'geolife': {
            'mhsa_acc': 33.18,
            'pointer_acc': 53.96,
            'improvement': 20.78,
        },
        'diy': {
            'mhsa_acc': 53.17,
            'pointer_acc': 56.88,
            'improvement': 3.71,
        }
    }
    
    # Load test data for sample distribution
    test_data = load_test_data()
    difficulty_results = analyze_sample_difficulty(test_data)
    
    print("\nAccuracy Breakdown Analysis:")
    print("=" * 70)
    
    for name in ['geolife', 'diy']:
        print(f"\n{name.upper()}:")
        print("-" * 50)
        
        easy_ratio = difficulty_results[name]['easy_ratio']
        medium_ratio = difficulty_results[name]['medium_ratio']
        hard_ratio = difficulty_results[name]['hard_ratio']
        
        mhsa_acc = actual_results[name]['mhsa_acc']
        pointer_acc = actual_results[name]['pointer_acc']
        improvement = actual_results[name]['improvement']
        
        print(f"  Sample Distribution:")
        print(f"    Easy (high pointer benefit):   {easy_ratio*100:.1f}%")
        print(f"    Medium (moderate benefit):     {medium_ratio*100:.1f}%")
        print(f"    Hard (no pointer benefit):     {hard_ratio*100:.1f}%")
        print(f"  Actual Performance:")
        print(f"    MHSA Accuracy:    {mhsa_acc:.2f}%")
        print(f"    Pointer Accuracy: {pointer_acc:.2f}%")
        print(f"    Improvement:      +{improvement:.2f}%")
        
        # Calculate expected improvement given sample distribution
        # Assume: Easy samples get ~30% boost, Medium ~15%, Hard ~0%
        # These are rough estimates based on typical pointer performance
        expected_boost_easy = 0.30
        expected_boost_medium = 0.15
        expected_boost_hard = 0.0
        
        expected_improvement = (
            easy_ratio * expected_boost_easy + 
            medium_ratio * expected_boost_medium + 
            hard_ratio * expected_boost_hard
        ) * 100
        
        print(f"  Expected Improvement (theoretical): ~{expected_improvement:.1f}%")
        print(f"  Efficiency: {improvement / expected_improvement * 100:.1f}%" if expected_improvement > 0 else "  N/A")
    
    return actual_results, difficulty_results


def create_comprehensive_comparison():
    """
    Create comprehensive comparison showing why improvements differ.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE COMPARISON: WHY IMPROVEMENTS DIFFER")
    print("=" * 70)
    
    comparison = {
        'Factor': [
            'MHSA Baseline Accuracy',
            'PGT Accuracy',
            'Actual Improvement',
            '---',
            'Target in History Ratio',
            'Target in Recent 5 Ratio',
            'Easy Samples Ratio',
            'Hard Samples Ratio',
            '---',
            'Num Locations',
            'Num Users',
            'Avg Seq Length',
        ],
        'Geolife': [
            '33.18%',
            '53.96%',
            '+20.78%',
            '---',
            '~79%',  # From earlier analysis
            '~65%',
            '~25%',
            '~21%',
            '---',
            '~1,187',
            '~45',
            '~9',
        ],
        'DIY': [
            '53.17%',
            '56.88%',
            '+3.71%',
            '---',
            '~71%',
            '~55%',
            '~18%',
            '~29%',
            '---',
            '~7,038',
            '~692',
            '~12',
        ],
        'Impact on Pointer': [
            'Lower baseline = more room to improve',
            'Higher with more pointer opportunity',
            'Gap explained by factors below',
            '---',
            'Higher = more copy opportunities',
            'Higher = easier copy decisions',
            'Higher = more improvement potential',
            'Higher = less improvement possible',
            '---',
            'More = harder classification',
            'More diverse patterns',
            'Longer = more context but diluted signal',
        ]
    }
    
    df_comparison = pd.DataFrame(comparison)
    print("\n" + df_comparison.to_string(index=False))
    df_comparison.to_csv(OUTPUT_DIR / "12_comprehensive_comparison.csv", index=False)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    models = ['MHSA', 'PGT']
    geo_accs = [33.18, 53.96]
    diy_accs = [53.17, 56.88]
    
    x = np.arange(len(models))
    width = 0.35
    bars1 = ax1.bar(x - width/2, geo_accs, width, label='Geolife', color='#3498db')
    bars2 = ax1.bar(x + width/2, diy_accs, width, label='DIY', color='#e74c3c')
    
    ax1.set_ylabel('Accuracy@1 (%)')
    ax1.set_title('Model Accuracy Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.legend()
    ax1.set_ylim(0, 70)
    
    # Add improvement arrows
    ax1.annotate('', xy=(0 + width/2, geo_accs[1]), xytext=(0 - width/2, geo_accs[0]),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    ax1.text(0, (geo_accs[0] + geo_accs[1])/2, f'+{geo_accs[1]-geo_accs[0]:.1f}%',
            ha='center', fontweight='bold', color='#27ae60')
    
    ax1.annotate('', xy=(1 + width/2, diy_accs[1]), xytext=(1 - width/2, diy_accs[0]),
                arrowprops=dict(arrowstyle='->', color='#27ae60', lw=2))
    ax1.text(1, (diy_accs[0] + diy_accs[1])/2 + 2, f'+{diy_accs[1]-diy_accs[0]:.1f}%',
            ha='center', fontweight='bold', color='#27ae60')
    
    # 2. Key factors comparison
    ax2 = axes[0, 1]
    factors = ['Target in\nHistory', 'Recent Target\n(< 5 pos)', 'Easy\nSamples']
    geo_factors = [79, 65, 25]
    diy_factors = [71, 55, 18]
    
    x = np.arange(len(factors))
    ax2.bar(x - width/2, geo_factors, width, label='Geolife', color='#3498db')
    ax2.bar(x + width/2, diy_factors, width, label='DIY', color='#e74c3c')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Key Factors Enabling Pointer Mechanism')
    ax2.set_xticks(x)
    ax2.set_xticklabels(factors)
    ax2.legend()
    
    # 3. Dataset characteristics
    ax3 = axes[1, 0]
    chars = ['Locations\n(÷100)', 'Users\n(÷10)', 'Sequences\n(÷1000)']
    geo_chars = [1187/100, 45/10, 3502/1000]
    diy_chars = [7038/100, 692/10, 12368/1000]
    
    x = np.arange(len(chars))
    ax3.bar(x - width/2, geo_chars, width, label='Geolife', color='#3498db')
    ax3.bar(x + width/2, diy_chars, width, label='DIY', color='#e74c3c')
    ax3.set_ylabel('Value (scaled)')
    ax3.set_title('Dataset Scale Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(chars)
    ax3.legend()
    
    # 4. Improvement explanation
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    explanation = """
SUMMARY: Why Geolife Improvement is Larger

1. HIGHER COPYABLE RATIO
   Geolife: ~79% of targets appear in history
   DIY: ~71% of targets appear in history
   → More opportunities for Pointer to copy

2. MORE RECENT TARGETS
   Geolife: ~65% of targets in last 5 positions
   DIY: ~55% of targets in last 5 positions
   → Easier for Pointer to attend to recent history

3. LOWER BASELINE ACCURACY
   Geolife MHSA: 33.18% (much room to improve)
   DIY MHSA: 53.17% (already decent)
   → More headroom for improvement on Geolife

4. SMALLER VOCABULARY
   Geolife: ~1,187 locations
   DIY: ~7,038 locations
   → Easier classification problem for Geolife

5. CONCLUSION
   The Pointer mechanism works effectively on BOTH datasets,
   but Geolife provides more favorable conditions for the
   copy-from-history strategy, resulting in larger gains.
"""
    ax4.text(0.05, 0.95, explanation, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "12_comprehensive_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    return comparison


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("MODEL PERFORMANCE DEEP DIVE ANALYSIS")
    print("=" * 70)
    
    # Run analyses
    test_data = load_test_data()
    difficulty_results = analyze_sample_difficulty(test_data)
    actual_results, difficulty_results = analyze_theoretical_accuracy_breakdown()
    comparison = create_comprehensive_comparison()
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
