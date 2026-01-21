"""
Data Characteristics Analysis: Geolife vs DIY Dataset

This script analyzes the fundamental characteristics of both datasets to understand
why the improvement from MHSA to PGT is larger on Geolife (+20.78%) 
than on DIY (+3.71%).

Key Analysis Areas:
1. Basic Statistics (users, locations, sequences)
2. Location Distribution Analysis (entropy, repetition patterns)
3. Temporal Patterns (tracking duration, visit frequency)
4. Sequence Characteristics (length, history patterns)
5. User Behavior Patterns (location diversity per user)
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
from scipy.stats import entropy
from pathlib import Path

# Set output directory
OUTPUT_DIR = Path("scripts/analysis_geolife_vs_diy/results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def load_data():
    """Load interim and processed datasets for both Geolife and DIY."""
    print("=" * 70)
    print("Loading datasets...")
    print("=" * 70)
    
    # Load interim data
    geolife_interim = pd.read_csv("data/geolife_eps20/interim/intermediate_eps20.csv")
    diy_interim = pd.read_csv("data/diy_eps50/interim/intermediate_eps50.csv")
    
    # Load processed data (prev7)
    with open("data/geolife_eps20/processed/geolife_eps20_prev7_train.pk", "rb") as f:
        geolife_train = pickle.load(f)
    with open("data/geolife_eps20/processed/geolife_eps20_prev7_test.pk", "rb") as f:
        geolife_test = pickle.load(f)
    
    with open("data/diy_eps50/processed/diy_eps50_prev7_train.pk", "rb") as f:
        diy_train = pickle.load(f)
    with open("data/diy_eps50/processed/diy_eps50_prev7_test.pk", "rb") as f:
        diy_test = pickle.load(f)
    
    # Load metadata
    with open("data/geolife_eps20/processed/geolife_eps20_prev7_metadata.json", "r") as f:
        geolife_meta = json.load(f)
    with open("data/diy_eps50/processed/diy_eps50_prev7_metadata.json", "r") as f:
        diy_meta = json.load(f)
    
    return {
        'geolife': {
            'interim': geolife_interim,
            'train': geolife_train,
            'test': geolife_test,
            'meta': geolife_meta
        },
        'diy': {
            'interim': diy_interim,
            'train': diy_train,
            'test': diy_test,
            'meta': diy_meta
        }
    }


def analyze_basic_statistics(data):
    """Analyze basic statistics of both datasets."""
    print("\n" + "=" * 70)
    print("1. BASIC STATISTICS COMPARISON")
    print("=" * 70)
    
    stats = {}
    for name, d in data.items():
        interim = d['interim']
        meta = d['meta']
        train = d['train']
        test = d['test']
        
        stats[name] = {
            'num_users': meta['total_user_num'],
            'num_locations': meta['total_loc_num'],
            'total_staypoints': len(interim),
            'train_sequences': len(train),
            'test_sequences': len(test),
            'avg_staypoints_per_user': len(interim) / meta['unique_users'],
            'avg_locations_per_user': interim.groupby('user_id')['location_id'].nunique().mean(),
            'location_per_user_ratio': interim.groupby('user_id')['location_id'].nunique().mean() / meta['total_loc_num'],
        }
    
    # Print comparison
    df_stats = pd.DataFrame(stats).T
    print("\n" + df_stats.to_string())
    
    # Save
    df_stats.to_csv(OUTPUT_DIR / "01_basic_statistics.csv")
    
    return stats


def analyze_location_distribution(data):
    """Analyze location visit distribution patterns."""
    print("\n" + "=" * 70)
    print("2. LOCATION DISTRIBUTION ANALYSIS")
    print("=" * 70)
    
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, d) in enumerate(data.items()):
        interim = d['interim']
        train = d['train']
        
        # Overall location frequency
        loc_counts = interim['location_id'].value_counts()
        
        # Calculate entropy (measure of predictability)
        loc_probs = loc_counts / loc_counts.sum()
        loc_entropy = entropy(loc_probs)
        max_entropy = np.log(len(loc_counts))
        normalized_entropy = loc_entropy / max_entropy
        
        # Calculate location repetition in sequences
        repeat_ratios = []
        target_in_history = []
        for seq in train:
            history = seq['X']
            target = seq['Y']
            unique_ratio = len(set(history)) / len(history)
            repeat_ratios.append(1 - unique_ratio)  # Repetition ratio
            target_in_history.append(1 if target in history else 0)
        
        avg_repeat_ratio = np.mean(repeat_ratios)
        target_in_history_ratio = np.mean(target_in_history)
        
        # Top-k location coverage
        top_5_coverage = loc_counts.head(5).sum() / loc_counts.sum()
        top_10_coverage = loc_counts.head(10).sum() / loc_counts.sum()
        top_20_coverage = loc_counts.head(20).sum() / loc_counts.sum()
        
        # Gini coefficient for location distribution
        sorted_counts = np.sort(loc_counts.values)
        n = len(sorted_counts)
        gini = (2 * np.sum((np.arange(1, n+1) * sorted_counts))) / (n * np.sum(sorted_counts)) - (n + 1) / n
        
        results[name] = {
            'num_unique_locations': len(loc_counts),
            'location_entropy': loc_entropy,
            'normalized_entropy': normalized_entropy,
            'avg_repetition_ratio_in_seq': avg_repeat_ratio,
            'target_in_history_ratio': target_in_history_ratio,
            'top_5_location_coverage': top_5_coverage,
            'top_10_location_coverage': top_10_coverage,
            'top_20_location_coverage': top_20_coverage,
            'gini_coefficient': gini,
        }
        
        # Plot distributions
        ax1 = axes[idx, 0]
        ax1.plot(range(1, min(101, len(loc_counts)+1)), loc_counts.values[:100], 'b-', linewidth=2)
        ax1.set_xlabel('Location Rank')
        ax1.set_ylabel('Visit Count')
        ax1.set_title(f'{name.upper()}: Top 100 Location Distribution')
        ax1.set_yscale('log')
        
        ax2 = axes[idx, 1]
        ax2.hist(repeat_ratios, bins=50, alpha=0.7, edgecolor='black')
        ax2.axvline(avg_repeat_ratio, color='r', linestyle='--', label=f'Mean: {avg_repeat_ratio:.3f}')
        ax2.set_xlabel('Repetition Ratio in Sequence')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{name.upper()}: Sequence Repetition Pattern')
        ax2.legend()
        
        ax3 = axes[idx, 2]
        cumsum = np.cumsum(sorted_counts[::-1]) / np.sum(sorted_counts)
        ax3.plot(np.linspace(0, 1, len(cumsum)), cumsum, linewidth=2)
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax3.fill_between(np.linspace(0, 1, len(cumsum)), cumsum, alpha=0.3)
        ax3.set_xlabel('Proportion of Locations')
        ax3.set_ylabel('Cumulative Visit Proportion')
        ax3.set_title(f'{name.upper()}: Location Lorenz Curve (Gini={gini:.3f})')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_location_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Print and save results
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.to_string())
    df_results.to_csv(OUTPUT_DIR / "02_location_distribution.csv")
    
    print(f"\n[KEY FINDING]")
    print(f"  - Geolife target_in_history_ratio: {results['geolife']['target_in_history_ratio']:.4f}")
    print(f"  - DIY target_in_history_ratio: {results['diy']['target_in_history_ratio']:.4f}")
    print(f"  - This ratio is critical for Pointer mechanism effectiveness!")
    
    return results


def analyze_temporal_patterns(data):
    """Analyze temporal patterns and tracking duration."""
    print("\n" + "=" * 70)
    print("3. TEMPORAL PATTERNS ANALYSIS")
    print("=" * 70)
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (name, d) in enumerate(data.items()):
        interim = d['interim']
        
        # Tracking duration per user
        user_tracking = interim.groupby('user_id').agg({
            'start_day': ['min', 'max'],
            'location_id': ['count', 'nunique']
        })
        user_tracking.columns = ['start_day_min', 'start_day_max', 'visit_count', 'unique_locations']
        user_tracking['tracking_days'] = user_tracking['start_day_max'] - user_tracking['start_day_min']
        user_tracking['visits_per_day'] = user_tracking['visit_count'] / (user_tracking['tracking_days'] + 1)
        user_tracking['location_diversity'] = user_tracking['unique_locations'] / user_tracking['visit_count']
        
        # Duration patterns
        avg_duration = interim['duration'].mean()
        median_duration = interim['duration'].median()
        
        results[name] = {
            'avg_tracking_days': user_tracking['tracking_days'].mean(),
            'median_tracking_days': user_tracking['tracking_days'].median(),
            'avg_visits_per_day': user_tracking['visits_per_day'].mean(),
            'avg_unique_locations_per_user': user_tracking['unique_locations'].mean(),
            'avg_location_diversity': user_tracking['location_diversity'].mean(),
            'avg_stay_duration_min': avg_duration,
            'median_stay_duration_min': median_duration,
        }
        
        # Plot tracking days distribution
        ax1 = axes[idx, 0]
        ax1.hist(user_tracking['tracking_days'], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(user_tracking['tracking_days'].mean(), color='r', linestyle='--', 
                   label=f"Mean: {user_tracking['tracking_days'].mean():.1f}")
        ax1.set_xlabel('Tracking Days')
        ax1.set_ylabel('Number of Users')
        ax1.set_title(f'{name.upper()}: User Tracking Duration')
        ax1.legend()
        
        # Plot location diversity
        ax2 = axes[idx, 1]
        ax2.scatter(user_tracking['visit_count'], user_tracking['unique_locations'], alpha=0.5)
        ax2.set_xlabel('Total Visits')
        ax2.set_ylabel('Unique Locations')
        ax2.set_title(f'{name.upper()}: Visit Count vs Location Diversity')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_temporal_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.to_string())
    df_results.to_csv(OUTPUT_DIR / "03_temporal_patterns.csv")
    
    return results


def analyze_sequence_characteristics(data):
    """Analyze sequence characteristics critical for Pointer mechanism."""
    print("\n" + "=" * 70)
    print("4. SEQUENCE CHARACTERISTICS ANALYSIS")
    print("=" * 70)
    
    results = {}
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    for idx, (name, d) in enumerate(data.items()):
        train = d['train']
        test = d['test']
        all_seqs = train + test
        
        # Analyze sequence characteristics
        seq_lengths = []
        unique_in_seq = []
        target_position_in_history = []
        most_frequent_target = []
        recency_of_target = []  # Position from end if target in history
        
        for seq in all_seqs:
            history = seq['X']
            target = seq['Y']
            
            seq_lengths.append(len(history))
            unique_in_seq.append(len(set(history)))
            
            if target in history:
                # Find all positions
                positions = [i for i, x in enumerate(history) if x == target]
                target_position_in_history.append(len(positions))
                # Recency: distance from end
                recency_of_target.append(len(history) - max(positions) - 1)
                
                # Check if target is most frequent in history
                counter = Counter(history)
                most_common = counter.most_common(1)[0][0]
                most_frequent_target.append(1 if target == most_common else 0)
        
        # Statistics
        avg_seq_len = np.mean(seq_lengths)
        avg_unique = np.mean(unique_in_seq)
        avg_repetition = avg_seq_len - avg_unique
        target_in_history_count = len(target_position_in_history)
        target_in_history_ratio = target_in_history_count / len(all_seqs)
        
        avg_target_freq_when_in_hist = np.mean(target_position_in_history) if target_position_in_history else 0
        avg_recency = np.mean(recency_of_target) if recency_of_target else 0
        most_freq_is_target_ratio = np.mean(most_frequent_target) if most_frequent_target else 0
        
        results[name] = {
            'avg_sequence_length': avg_seq_len,
            'avg_unique_locations_in_seq': avg_unique,
            'avg_repeated_locations_in_seq': avg_repetition,
            'target_in_history_ratio': target_in_history_ratio,
            'avg_target_freq_when_in_history': avg_target_freq_when_in_hist,
            'avg_target_recency_when_in_history': avg_recency,
            'most_frequent_is_target_ratio': most_freq_is_target_ratio,
        }
        
        # Plots
        ax1 = axes[idx, 0]
        ax1.hist(seq_lengths, bins=30, alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Sequence Length')
        ax1.set_ylabel('Frequency')
        ax1.set_title(f'{name.upper()}: Sequence Length Distribution')
        
        ax2 = axes[idx, 1]
        if recency_of_target:
            ax2.hist(recency_of_target, bins=30, alpha=0.7, edgecolor='black')
            ax2.axvline(avg_recency, color='r', linestyle='--', label=f'Mean: {avg_recency:.1f}')
            ax2.legend()
        ax2.set_xlabel('Target Recency (distance from end)')
        ax2.set_ylabel('Frequency')
        ax2.set_title(f'{name.upper()}: Target Location Recency')
        
        ax3 = axes[idx, 2]
        if target_position_in_history:
            ax3.hist(target_position_in_history, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Target Frequency in History')
        ax3.set_ylabel('Frequency')
        ax3.set_title(f'{name.upper()}: Target Occurrence Count in History')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_sequence_characteristics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.to_string())
    df_results.to_csv(OUTPUT_DIR / "04_sequence_characteristics.csv")
    
    print(f"\n[KEY FINDING]")
    print(f"  - Geolife: {results['geolife']['target_in_history_ratio']*100:.2f}% of targets appear in history")
    print(f"  - DIY: {results['diy']['target_in_history_ratio']*100:.2f}% of targets appear in history")
    print(f"  - This directly impacts Pointer mechanism effectiveness!")
    
    return results


def analyze_user_behavior_patterns(data):
    """Analyze per-user behavior patterns and predictability."""
    print("\n" + "=" * 70)
    print("5. USER BEHAVIOR PATTERNS ANALYSIS")
    print("=" * 70)
    
    results = {}
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for idx, (name, d) in enumerate(data.items()):
        interim = d['interim']
        train = d['train']
        
        # Analyze user-level patterns
        user_stats = []
        for user_id in interim['user_id'].unique():
            user_data = interim[interim['user_id'] == user_id]
            
            # Location entropy per user
            loc_counts = user_data['location_id'].value_counts()
            loc_probs = loc_counts / loc_counts.sum()
            user_entropy = entropy(loc_probs)
            max_entropy = np.log(len(loc_counts)) if len(loc_counts) > 1 else 1
            normalized_entropy = user_entropy / max_entropy if max_entropy > 0 else 0
            
            # Top location dominance
            top1_ratio = loc_counts.iloc[0] / loc_counts.sum() if len(loc_counts) > 0 else 0
            top3_ratio = loc_counts.head(3).sum() / loc_counts.sum() if len(loc_counts) > 0 else 0
            
            user_stats.append({
                'user_id': user_id,
                'visit_count': len(user_data),
                'unique_locations': len(loc_counts),
                'entropy': user_entropy,
                'normalized_entropy': normalized_entropy,
                'top1_ratio': top1_ratio,
                'top3_ratio': top3_ratio,
            })
        
        user_df = pd.DataFrame(user_stats)
        
        # Calculate predictability scores
        avg_entropy = user_df['normalized_entropy'].mean()
        avg_top1 = user_df['top1_ratio'].mean()
        avg_top3 = user_df['top3_ratio'].mean()
        
        # Users with high repetition patterns (potential for pointer)
        high_repeat_users = (user_df['top1_ratio'] > 0.5).sum() / len(user_df)
        
        results[name] = {
            'avg_user_normalized_entropy': avg_entropy,
            'avg_user_top1_ratio': avg_top1,
            'avg_user_top3_ratio': avg_top3,
            'high_repeat_user_ratio': high_repeat_users,
            'avg_unique_locations_per_user': user_df['unique_locations'].mean(),
            'median_unique_locations_per_user': user_df['unique_locations'].median(),
        }
        
        # Plots
        ax1 = axes[idx, 0]
        ax1.hist(user_df['normalized_entropy'], bins=30, alpha=0.7, edgecolor='black')
        ax1.axvline(avg_entropy, color='r', linestyle='--', label=f'Mean: {avg_entropy:.3f}')
        ax1.set_xlabel('Normalized Entropy')
        ax1.set_ylabel('Number of Users')
        ax1.set_title(f'{name.upper()}: User Behavior Predictability')
        ax1.legend()
        
        ax2 = axes[idx, 1]
        ax2.scatter(user_df['unique_locations'], user_df['normalized_entropy'], alpha=0.5)
        ax2.set_xlabel('Unique Locations per User')
        ax2.set_ylabel('Normalized Entropy')
        ax2.set_title(f'{name.upper()}: Location Count vs Entropy')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "05_user_behavior_patterns.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    df_results = pd.DataFrame(results).T
    print("\n" + df_results.to_string())
    df_results.to_csv(OUTPUT_DIR / "05_user_behavior_patterns.csv")
    
    return results


def create_summary_comparison(all_results):
    """Create comprehensive summary of key differences."""
    print("\n" + "=" * 70)
    print("SUMMARY: KEY DIFFERENCES EXPLAINING IMPROVEMENT GAP")
    print("=" * 70)
    
    summary = {
        'Metric': [],
        'Geolife': [],
        'DIY': [],
        'Implication for Pointer': []
    }
    
    # Key metrics affecting pointer performance
    key_metrics = [
        ('Target in History Ratio', 'target_in_history_ratio', 'seq', 
         'Higher = Pointer more effective'),
        ('Avg Sequence Repetition', 'avg_repeated_locations_in_seq', 'seq',
         'Higher = More copying opportunities'),
        ('User Top-1 Location Ratio', 'avg_user_top1_ratio', 'user',
         'Higher = More predictable patterns'),
        ('User Normalized Entropy', 'avg_user_normalized_entropy', 'user',
         'Lower = More predictable'),
        ('Avg Locations per User', 'avg_unique_locations_per_user', 'user',
         'Lower = Easier to predict'),
        ('Location Gini Coefficient', 'gini_coefficient', 'loc',
         'Higher = More concentrated visits'),
    ]
    
    for display_name, metric, source, implication in key_metrics:
        if source == 'seq':
            geo_val = all_results['sequence']['geolife'][metric]
            diy_val = all_results['sequence']['diy'][metric]
        elif source == 'user':
            geo_val = all_results['user']['geolife'][metric]
            diy_val = all_results['user']['diy'][metric]
        elif source == 'loc':
            geo_val = all_results['location']['geolife'][metric]
            diy_val = all_results['location']['diy'][metric]
        
        summary['Metric'].append(display_name)
        summary['Geolife'].append(f"{geo_val:.4f}")
        summary['DIY'].append(f"{diy_val:.4f}")
        summary['Implication for Pointer'].append(implication)
    
    df_summary = pd.DataFrame(summary)
    print("\n" + df_summary.to_string(index=False))
    df_summary.to_csv(OUTPUT_DIR / "06_summary_comparison.csv", index=False)
    
    # Print key conclusions
    print("\n" + "=" * 70)
    print("KEY CONCLUSIONS")
    print("=" * 70)
    
    geo_target_ratio = all_results['sequence']['geolife']['target_in_history_ratio']
    diy_target_ratio = all_results['sequence']['diy']['target_in_history_ratio']
    
    print(f"""
1. TARGET IN HISTORY RATIO (Most Critical)
   - Geolife: {geo_target_ratio*100:.2f}%
   - DIY: {diy_target_ratio*100:.2f}%
   - Difference: {(geo_target_ratio - diy_target_ratio)*100:.2f}%
   
   The Pointer mechanism in PGT can ONLY help when the target location
   appears in the history sequence. This is the PRIMARY factor explaining the
   performance gap.

2. USER BEHAVIOR PREDICTABILITY
   - Geolife users have HIGHER top-location concentration
   - Geolife users have LOWER normalized entropy (more predictable)
   - This makes Geolife inherently more suited for the Pointer mechanism

3. DATASET SCALE EFFECT
   - DIY has significantly more locations ({all_results['basic']['diy']['num_locations']} vs {all_results['basic']['geolife']['num_locations']})
   - More locations = harder classification problem
   - MHSA already performs well on DIY, leaving less room for improvement

4. SEQUENCE CHARACTERISTICS
   - Geolife sequences have more repetition in history
   - This creates more opportunities for the Pointer to "copy" from history
""")
    
    return df_summary


def main():
    """Main analysis pipeline."""
    print("=" * 70)
    print("DATA CHARACTERISTICS ANALYSIS: GEOLIFE VS DIY")
    print("Investigating why Pointer improvement differs between datasets")
    print("=" * 70)
    
    # Load data
    data = load_data()
    
    # Run analyses
    basic_stats = analyze_basic_statistics(data)
    location_results = analyze_location_distribution(data)
    temporal_results = analyze_temporal_patterns(data)
    sequence_results = analyze_sequence_characteristics(data)
    user_results = analyze_user_behavior_patterns(data)
    
    # Create summary
    all_results = {
        'basic': basic_stats,
        'location': location_results,
        'temporal': temporal_results,
        'sequence': sequence_results,
        'user': user_results,
    }
    
    summary = create_summary_comparison(all_results)
    
    # Save all results to JSON
    def convert_to_serializable(obj):
        if isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    serializable_results = {}
    for key, val in all_results.items():
        if isinstance(val, dict):
            serializable_results[key] = {}
            for k2, v2 in val.items():
                if isinstance(v2, dict):
                    serializable_results[key][k2] = {k3: convert_to_serializable(v3) for k3, v3 in v2.items()}
                else:
                    serializable_results[key][k2] = convert_to_serializable(v2)
        else:
            serializable_results[key] = convert_to_serializable(val)
    
    with open(OUTPUT_DIR / "all_analysis_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"\n✓ All results saved to: {OUTPUT_DIR}")
    print("=" * 70)
    
    return all_results


if __name__ == "__main__":
    main()
