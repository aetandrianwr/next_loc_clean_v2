"""
Pointer Mechanism Effectiveness Analysis

This script analyzes why the Pointer mechanism in PGT is more effective 
on Geolife than DIY by examining:
1. Copy vs Generate behavior
2. Target location position patterns
3. Pointer-friendly vs Pointer-unfriendly samples

Key Hypothesis:
The Pointer mechanism benefits from:
- High "target in history" ratio
- Recent target appearances
- Repetitive user behavior patterns
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


def load_data():
    """Load test datasets for both Geolife and DIY."""
    with open("data/geolife_eps20/processed/geolife_eps20_prev7_train.pk", "rb") as f:
        geolife_train = pickle.load(f)
    with open("data/geolife_eps20/processed/geolife_eps20_prev7_test.pk", "rb") as f:
        geolife_test = pickle.load(f)
    
    with open("data/diy_eps50/processed/diy_eps50_prev7_train.pk", "rb") as f:
        diy_train = pickle.load(f)
    with open("data/diy_eps50/processed/diy_eps50_prev7_test.pk", "rb") as f:
        diy_test = pickle.load(f)
    
    return {
        'geolife': {'train': geolife_train, 'test': geolife_test, 'all': geolife_train + geolife_test},
        'diy': {'train': diy_train, 'test': diy_test, 'all': diy_train + diy_test}
    }


def analyze_pointer_opportunities(data):
    """
    Analyze opportunities where Pointer mechanism can help.
    
    For each sequence, categorize:
    1. Copy-able: Target appears in history
    2. Generate-only: Target does NOT appear in history
    
    For copy-able cases, analyze:
    - Position of target in history
    - Frequency of target in history
    - Recency of target (distance from end)
    """
    print("\n" + "=" * 70)
    print("POINTER MECHANISM OPPORTUNITY ANALYSIS")
    print("=" * 70)
    
    results = {}
    detailed_stats = {}
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    for idx, (name, d) in enumerate(data.items()):
        all_seqs = d['all']
        
        # Categorize sequences
        copyable = []
        generate_only = []
        
        target_positions = []  # All positions where target appears
        target_frequencies = []  # How often target appears
        target_recencies = []  # Distance from end (most recent occurrence)
        target_first_occurrence = []  # Position of first occurrence
        
        for seq in all_seqs:
            history = list(seq['X'])
            target = seq['Y']
            
            if target in history:
                copyable.append(seq)
                
                # Find all positions
                positions = [i for i, x in enumerate(history) if x == target]
                target_positions.extend(positions)
                target_frequencies.append(len(positions))
                
                # Recency: distance from end (0 = last position)
                most_recent_pos = max(positions)
                recency = len(history) - most_recent_pos - 1
                target_recencies.append(recency)
                
                # First occurrence
                target_first_occurrence.append(min(positions))
            else:
                generate_only.append(seq)
        
        copyable_ratio = len(copyable) / len(all_seqs)
        
        results[name] = {
            'total_sequences': len(all_seqs),
            'copyable_sequences': len(copyable),
            'generate_only_sequences': len(generate_only),
            'copyable_ratio': copyable_ratio,
            'generate_only_ratio': 1 - copyable_ratio,
            'avg_target_frequency_in_history': np.mean(target_frequencies) if target_frequencies else 0,
            'avg_target_recency': np.mean(target_recencies) if target_recencies else 0,
            'target_in_last_3_ratio': np.mean([1 if r < 3 else 0 for r in target_recencies]) if target_recencies else 0,
            'target_in_last_5_ratio': np.mean([1 if r < 5 else 0 for r in target_recencies]) if target_recencies else 0,
        }
        
        detailed_stats[name] = {
            'target_frequencies': target_frequencies,
            'target_recencies': target_recencies,
            'target_first_occurrence': target_first_occurrence,
        }
        
        # Visualizations
        # 1. Pie chart: copyable vs generate-only
        ax1 = axes[idx, 0]
        sizes = [len(copyable), len(generate_only)]
        labels = ['Copy-able\n(Target in History)', 'Generate-only\n(Target NOT in History)']
        colors = ['#2ecc71', '#e74c3c']
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title(f'{name.upper()}: Pointer Opportunity')
        
        # 2. Target frequency distribution
        ax2 = axes[idx, 1]
        if target_frequencies:
            max_freq = min(max(target_frequencies), 20)
            ax2.hist(target_frequencies, bins=range(1, max_freq+2), alpha=0.7, edgecolor='black', align='left')
            ax2.set_xlabel('Target Frequency in History')
            ax2.set_ylabel('Number of Sequences')
            ax2.set_title(f'{name.upper()}: Target Frequency\n(when in history)')
        
        # 3. Target recency distribution
        ax3 = axes[idx, 2]
        if target_recencies:
            max_recency = min(max(target_recencies), 30)
            ax3.hist(target_recencies, bins=range(0, max_recency+2), alpha=0.7, edgecolor='black', align='left')
            ax3.axvline(np.mean(target_recencies), color='r', linestyle='--', 
                       label=f'Mean: {np.mean(target_recencies):.1f}')
            ax3.set_xlabel('Target Recency (0 = most recent)')
            ax3.set_ylabel('Number of Sequences')
            ax3.set_title(f'{name.upper()}: Target Recency\n(when in history)')
            ax3.legend()
        
        # 4. Cumulative recency
        ax4 = axes[idx, 3]
        if target_recencies:
            sorted_recencies = np.sort(target_recencies)
            cumulative = np.arange(1, len(sorted_recencies)+1) / len(sorted_recencies)
            ax4.plot(sorted_recencies, cumulative, linewidth=2)
            ax4.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
            ax4.axvline(np.median(target_recencies), color='r', linestyle='--',
                       label=f'Median: {np.median(target_recencies):.1f}')
            ax4.set_xlabel('Recency Threshold')
            ax4.set_ylabel('Cumulative Proportion')
            ax4.set_title(f'{name.upper()}: Cumulative Recency\n(target in recent N positions)')
            ax4.legend()
            ax4.set_xlim(0, min(30, max(target_recencies)))
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "07_pointer_opportunity_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print results
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.to_string())
    df_results.to_csv(OUTPUT_DIR / "07_pointer_opportunity_analysis.csv")
    
    return results, detailed_stats


def analyze_copy_benefit_by_recency(data):
    """
    Analyze how Pointer benefit varies by target recency.
    
    Hypothesis: Pointer is most effective when target appeared recently.
    """
    print("\n" + "=" * 70)
    print("COPY BENEFIT BY RECENCY ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    for name, d in data.items():
        all_seqs = d['all']
        
        # Bin sequences by recency
        recency_bins = {
            '0 (most recent)': [],
            '1-2': [],
            '3-5': [],
            '6-10': [],
            '>10': [],
            'not_in_history': []
        }
        
        for seq in all_seqs:
            history = list(seq['X'])
            target = seq['Y']
            
            if target in history:
                positions = [i for i, x in enumerate(history) if x == target]
                recency = len(history) - max(positions) - 1
                
                if recency == 0:
                    recency_bins['0 (most recent)'].append(seq)
                elif recency <= 2:
                    recency_bins['1-2'].append(seq)
                elif recency <= 5:
                    recency_bins['3-5'].append(seq)
                elif recency <= 10:
                    recency_bins['6-10'].append(seq)
                else:
                    recency_bins['>10'].append(seq)
            else:
                recency_bins['not_in_history'].append(seq)
        
        results[name] = {k: len(v) for k, v in recency_bins.items()}
        results[name]['total'] = len(all_seqs)
        
        # Calculate proportions
        for k in recency_bins:
            results[name][f'{k}_ratio'] = len(recency_bins[k]) / len(all_seqs)
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    recency_order = ['0 (most recent)', '1-2', '3-5', '6-10', '>10', 'not_in_history']
    
    for idx, (name, res) in enumerate(results.items()):
        ax = axes[idx]
        counts = [res[k] for k in recency_order]
        percentages = [res[f'{k}_ratio'] * 100 for k in recency_order]
        
        bars = ax.bar(recency_order, counts, color=plt.cm.RdYlGn(np.linspace(0.8, 0.2, len(recency_order))))
        ax.set_xlabel('Target Recency (positions from end)')
        ax.set_ylabel('Number of Sequences')
        ax.set_title(f'{name.upper()}: Sequences by Target Recency')
        ax.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        for bar, pct in zip(bars, percentages):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{pct:.1f}%',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "08_recency_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print comparison
    print("\nRecency Distribution Comparison:")
    print("-" * 60)
    print(f"{'Recency':<20} {'Geolife':<20} {'DIY':<20}")
    print("-" * 60)
    for k in recency_order:
        geo_val = results['geolife'][f'{k}_ratio'] * 100
        diy_val = results['diy'][f'{k}_ratio'] * 100
        print(f"{k:<20} {geo_val:>6.2f}%{'':>12} {diy_val:>6.2f}%")
    
    # Save results
    with open(OUTPUT_DIR / "08_recency_distribution.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results


def analyze_theoretical_pointer_ceiling(data):
    """
    Calculate theoretical ceiling of Pointer mechanism improvement.
    
    If Pointer could perfectly predict ALL copy-able samples,
    what would be the maximum possible improvement?
    """
    print("\n" + "=" * 70)
    print("THEORETICAL POINTER CEILING ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    for name, d in data.items():
        all_seqs = d['all']
        
        copyable = 0
        generate_only = 0
        
        # Additional analysis: what if we could copy perfectly for recent targets?
        recent_copyable = 0  # Target in last 5 positions
        very_recent_copyable = 0  # Target in last 3 positions
        
        for seq in all_seqs:
            history = list(seq['X'])
            target = seq['Y']
            
            if target in history:
                copyable += 1
                positions = [i for i, x in enumerate(history) if x == target]
                recency = len(history) - max(positions) - 1
                
                if recency < 5:
                    recent_copyable += 1
                if recency < 3:
                    very_recent_copyable += 1
            else:
                generate_only += 1
        
        total = len(all_seqs)
        
        # Actual performance from experiments
        if name == 'geolife':
            mhsa_acc = 33.18
            pointer_acc = 53.96
        else:  # diy
            mhsa_acc = 53.17
            pointer_acc = 56.88
        
        actual_improvement = pointer_acc - mhsa_acc
        
        # Theoretical ceiling: if Pointer got 100% on copyable samples
        # Assume MHSA's accuracy on generate-only samples remains the same
        # This is a rough estimate
        theoretical_ceiling_copyable = (copyable / total) * 100
        
        results[name] = {
            'total_samples': total,
            'copyable_samples': copyable,
            'copyable_ratio': copyable / total,
            'recent_copyable_ratio': recent_copyable / total,
            'very_recent_copyable_ratio': very_recent_copyable / total,
            'mhsa_accuracy': mhsa_acc,
            'pointer_accuracy': pointer_acc,
            'actual_improvement': actual_improvement,
            'theoretical_max_pointer_benefit': theoretical_ceiling_copyable,
            'improvement_efficiency': actual_improvement / theoretical_ceiling_copyable if theoretical_ceiling_copyable > 0 else 0,
        }
    
    # Print results
    print("\nTheoretical Pointer Ceiling Analysis:")
    print("-" * 70)
    df_results = pd.DataFrame(results).T
    print(df_results.to_string())
    
    print("\n" + "=" * 70)
    print("KEY INSIGHT:")
    print("=" * 70)
    
    geo_copyable = results['geolife']['copyable_ratio'] * 100
    diy_copyable = results['diy']['copyable_ratio'] * 100
    geo_improve = results['geolife']['actual_improvement']
    diy_improve = results['diy']['actual_improvement']
    
    print(f"""
The MAXIMUM benefit Pointer can provide is bounded by the copyable ratio:
- Geolife: {geo_copyable:.1f}% of samples are copyable
- DIY: {diy_copyable:.1f}% of samples are copyable

Actual improvement achieved:
- Geolife: +{geo_improve:.2f}% (from {results['geolife']['mhsa_accuracy']:.2f}% to {results['geolife']['pointer_accuracy']:.2f}%)
- DIY: +{diy_improve:.2f}% (from {results['diy']['mhsa_accuracy']:.2f}% to {results['diy']['pointer_accuracy']:.2f}%)

This shows:
1. Geolife has ~{geo_copyable - diy_copyable:.1f}% MORE copyable samples than DIY
2. The improvement gap ({geo_improve - diy_improve:.2f}%) is primarily due to this difference
3. The Pointer mechanism is doing its job effectively on both datasets,
   but there are simply fewer opportunities to "copy" in DIY
""")
    
    # Save results
    df_results.to_csv(OUTPUT_DIR / "09_theoretical_pointer_ceiling.csv")
    
    return results


def analyze_user_repetition_patterns(data):
    """
    Analyze per-user repetition patterns that enable Pointer mechanism.
    """
    print("\n" + "=" * 70)
    print("USER REPETITION PATTERN ANALYSIS")
    print("=" * 70)
    
    results = {}
    
    for name, d in data.items():
        all_seqs = d['all']
        
        # Group sequences by user
        user_seqs = defaultdict(list)
        for seq in all_seqs:
            user = seq['user_X'][0]
            user_seqs[user].append(seq)
        
        user_stats = []
        for user, seqs in user_seqs.items():
            copyable = 0
            for seq in seqs:
                history = list(seq['X'])
                target = seq['Y']
                if target in history:
                    copyable += 1
            
            user_stats.append({
                'user': user,
                'num_sequences': len(seqs),
                'copyable': copyable,
                'copyable_ratio': copyable / len(seqs) if seqs else 0
            })
        
        user_df = pd.DataFrame(user_stats)
        
        results[name] = {
            'num_users': len(user_df),
            'avg_copyable_ratio_per_user': user_df['copyable_ratio'].mean(),
            'median_copyable_ratio_per_user': user_df['copyable_ratio'].median(),
            'std_copyable_ratio': user_df['copyable_ratio'].std(),
            'users_with_high_copyable_ratio': (user_df['copyable_ratio'] > 0.8).sum() / len(user_df),
            'users_with_low_copyable_ratio': (user_df['copyable_ratio'] < 0.5).sum() / len(user_df),
        }
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (name, d) in enumerate(data.items()):
        all_seqs = d['all']
        
        user_seqs = defaultdict(list)
        for seq in all_seqs:
            user = seq['user_X'][0]
            user_seqs[user].append(seq)
        
        copyable_ratios = []
        for user, seqs in user_seqs.items():
            copyable = sum(1 for seq in seqs if seq['Y'] in list(seq['X']))
            copyable_ratios.append(copyable / len(seqs) if seqs else 0)
        
        ax = axes[idx]
        ax.hist(copyable_ratios, bins=20, alpha=0.7, edgecolor='black')
        ax.axvline(np.mean(copyable_ratios), color='r', linestyle='--',
                  label=f'Mean: {np.mean(copyable_ratios):.3f}')
        ax.set_xlabel('Copyable Ratio per User')
        ax.set_ylabel('Number of Users')
        ax.set_title(f'{name.upper()}: User Copyable Ratio Distribution')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "10_user_repetition_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print results
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.to_string())
    df_results.to_csv(OUTPUT_DIR / "10_user_repetition_patterns.csv")
    
    return results


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("POINTER MECHANISM EFFECTIVENESS ANALYSIS")
    print("Why is Pointer more effective on Geolife than DIY?")
    print("=" * 70)
    
    # Load data
    data = load_data()
    
    # Run analyses
    opportunity_results, detailed_stats = analyze_pointer_opportunities(data)
    recency_results = analyze_copy_benefit_by_recency(data)
    ceiling_results = analyze_theoretical_pointer_ceiling(data)
    user_pattern_results = analyze_user_repetition_patterns(data)
    
    # Combine all results
    all_results = {
        'opportunity': opportunity_results,
        'recency': recency_results,
        'ceiling': ceiling_results,
        'user_patterns': user_pattern_results,
    }
    
    # Save combined results
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable = {}
    for k1, v1 in all_results.items():
        serializable[k1] = {}
        for k2, v2 in v1.items():
            if isinstance(v2, dict):
                serializable[k1][k2] = {k3: convert_to_serializable(v3) for k3, v3 in v2.items()}
            else:
                serializable[k1][k2] = convert_to_serializable(v2)
    
    with open(OUTPUT_DIR / "pointer_mechanism_analysis.json", "w") as f:
        json.dump(serializable, f, indent=2)
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
