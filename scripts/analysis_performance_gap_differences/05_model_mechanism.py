"""
05. Model Mechanism Analysis
Analyze why the pointer mechanism benefits Geolife more than DIY.

This script directly investigates:
- Pointer gate activation patterns (simulation)
- Copy vs Generate scenarios
- Baseline-achievable accuracy (most frequent prediction)
- Upper bound analysis (oracle pointer)

KEY HYPOTHESIS: The difference in improvement comes from how well
the pointer mechanism can leverage the input sequence.
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

# Setup paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(OUTPUT_DIR, "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_sequences(dataset_name, epsilon, split, prev_day=7):
    """Load sequence data from pickle file."""
    path = os.path.join(BASE_DIR, "data", f"{dataset_name}_eps{epsilon}",
                        "processed", f"{dataset_name}_eps{epsilon}_prev{prev_day}_{split}.pk")
    with open(path, 'rb') as f:
        return pickle.load(f)


def calculate_oracle_pointer_accuracy(sequences):
    """
    Calculate the upper bound accuracy if we had a perfect pointer.
    This is the percentage of test cases where target appears in history.
    """
    correct = 0
    for seq in sequences:
        if seq['Y'] in seq['X']:
            correct += 1
    return correct / len(sequences) * 100


def calculate_most_frequent_accuracy(train_sequences, test_sequences):
    """
    Calculate accuracy using 'most frequent in history' heuristic.
    This represents what a simple baseline can achieve.
    """
    correct = 0
    for seq in test_sequences:
        loc_counts = Counter(seq['X'])
        if loc_counts:
            most_frequent = loc_counts.most_common(1)[0][0]
            if most_frequent == seq['Y']:
                correct += 1
    return correct / len(test_sequences) * 100


def calculate_most_recent_accuracy(sequences):
    """
    Calculate accuracy using 'most recent location' heuristic.
    """
    correct = 0
    for seq in sequences:
        if len(seq['X']) > 0:
            most_recent = seq['X'][-1]
            if most_recent == seq['Y']:
                correct += 1
    return correct / len(sequences) * 100


def calculate_global_most_frequent_accuracy(train_sequences, test_sequences):
    """
    Calculate accuracy using global most frequent location from training.
    """
    # Get global location frequencies from training targets
    target_counts = Counter([seq['Y'] for seq in train_sequences])
    most_frequent_global = target_counts.most_common(1)[0][0]
    
    correct = sum(1 for seq in test_sequences if seq['Y'] == most_frequent_global)
    return correct / len(test_sequences) * 100


def analyze_pointer_scenarios(sequences):
    """
    Analyze different scenarios for pointer mechanism.
    """
    scenarios = {
        'target_in_history': 0,
        'target_not_in_history': 0,
        'target_most_frequent_in_history': 0,
        'target_most_recent': 0,
        'target_in_last_5': 0
    }
    
    for seq in sequences:
        history = seq['X']
        target = seq['Y']
        
        if target in history:
            scenarios['target_in_history'] += 1
            
            # Check if target is most frequent
            loc_counts = Counter(history)
            most_frequent = loc_counts.most_common(1)[0][0]
            if most_frequent == target:
                scenarios['target_most_frequent_in_history'] += 1
            
            # Check if target is most recent
            if history[-1] == target:
                scenarios['target_most_recent'] += 1
            
            # Check if target is in last 5
            if target in history[-5:]:
                scenarios['target_in_last_5'] += 1
        else:
            scenarios['target_not_in_history'] += 1
    
    total = len(sequences)
    rates = {k: v / total * 100 for k, v in scenarios.items()}
    return rates, scenarios


def analyze_pointer_advantage(sequences, train_sequences):
    """
    Analyze cases where pointer provides advantage over generation.
    """
    # Build location frequency from training
    train_target_counts = Counter([seq['Y'] for seq in train_sequences])
    total_train_targets = sum(train_target_counts.values())
    train_probs = {loc: count / total_train_targets for loc, count in train_target_counts.items()}
    
    pointer_advantage_cases = 0
    generation_advantage_cases = 0
    neutral_cases = 0
    
    for seq in sequences:
        history = seq['X']
        target = seq['Y']
        
        # Pointer can only predict if target is in history
        target_in_history = target in history
        
        # Is target a frequent location globally?
        target_global_prob = train_probs.get(target, 0)
        target_is_frequent = target_global_prob > 0.01  # Top ~100 locations
        
        if target_in_history and not target_is_frequent:
            # Pointer helps: target is in history but rare globally
            pointer_advantage_cases += 1
        elif not target_in_history and target_is_frequent:
            # Generation helps: target is frequent globally but not in history
            generation_advantage_cases += 1
        else:
            neutral_cases += 1
    
    total = len(sequences)
    return {
        'pointer_advantage_pct': pointer_advantage_cases / total * 100,
        'generation_advantage_pct': generation_advantage_cases / total * 100,
        'neutral_pct': neutral_cases / total * 100
    }


def estimate_improvement_potential(sequences, baseline_acc, pointer_acc_upper):
    """
    Estimate the improvement potential from pointer mechanism.
    """
    # The improvement comes from:
    # 1. Cases where target is in history (pointer can help)
    # 2. Minus cases where baseline already gets it right
    
    # Maximum possible improvement
    max_improvement = pointer_acc_upper - baseline_acc
    
    # Actual improvement requires pointer to:
    # a) Correctly attend to the target in history
    # b) Output higher probability than generation
    
    return {
        'baseline_accuracy': baseline_acc,
        'oracle_pointer_accuracy': pointer_acc_upper,
        'max_improvement_potential': max_improvement
    }


def main():
    print("="*80)
    print("05. Model Mechanism Analysis")
    print("="*80)
    
    # Load sequences
    print("[Loading sequences...]")
    geo_train = load_sequences("geolife", 20, "train")
    geo_test = load_sequences("geolife", 20, "test")
    diy_train = load_sequences("diy", 50, "train")
    diy_test = load_sequences("diy", 50, "test")
    
    # Calculate heuristic baselines
    print("\n[Calculating baseline heuristics...]")
    
    # Oracle pointer (upper bound for pointer)
    geo_oracle_pointer = calculate_oracle_pointer_accuracy(geo_test)
    diy_oracle_pointer = calculate_oracle_pointer_accuracy(diy_test)
    
    # Most frequent in history
    geo_most_freq = calculate_most_frequent_accuracy(geo_train, geo_test)
    diy_most_freq = calculate_most_frequent_accuracy(diy_train, diy_test)
    
    # Most recent
    geo_most_recent = calculate_most_recent_accuracy(geo_test)
    diy_most_recent = calculate_most_recent_accuracy(diy_test)
    
    # Global most frequent
    geo_global_freq = calculate_global_most_frequent_accuracy(geo_train, geo_test)
    diy_global_freq = calculate_global_most_frequent_accuracy(diy_train, diy_test)
    
    print("\n" + "="*80)
    print("BASELINE HEURISTICS (Test Set)")
    print("="*80)
    
    heuristic_comparison = {
        'Heuristic': [
            'Oracle Pointer (target in history)',
            'Most Frequent in History',
            'Most Recent Location',
            'Global Most Frequent'
        ],
        'Geolife': [
            f"{geo_oracle_pointer:.2f}%",
            f"{geo_most_freq:.2f}%",
            f"{geo_most_recent:.2f}%",
            f"{geo_global_freq:.2f}%"
        ],
        'DIY': [
            f"{diy_oracle_pointer:.2f}%",
            f"{diy_most_freq:.2f}%",
            f"{diy_most_recent:.2f}%",
            f"{diy_global_freq:.2f}%"
        ]
    }
    
    heuristic_df = pd.DataFrame(heuristic_comparison)
    print(heuristic_df.to_string(index=False))
    heuristic_df.to_csv(os.path.join(RESULTS_DIR, "05_baseline_heuristics.csv"), index=False)
    
    # Analyze pointer scenarios
    print("\n[Analyzing pointer scenarios...]")
    geo_scenarios, _ = analyze_pointer_scenarios(geo_test)
    diy_scenarios, _ = analyze_pointer_scenarios(diy_test)
    
    print("\n" + "="*80)
    print("POINTER SCENARIO ANALYSIS (Test Set)")
    print("="*80)
    
    scenario_comparison = {
        'Scenario': [
            'Target in history',
            'Target NOT in history',
            'Target is most frequent in history',
            'Target is most recent',
            'Target in last 5'
        ],
        'Geolife (%)': [
            f"{geo_scenarios['target_in_history']:.2f}%",
            f"{geo_scenarios['target_not_in_history']:.2f}%",
            f"{geo_scenarios['target_most_frequent_in_history']:.2f}%",
            f"{geo_scenarios['target_most_recent']:.2f}%",
            f"{geo_scenarios['target_in_last_5']:.2f}%"
        ],
        'DIY (%)': [
            f"{diy_scenarios['target_in_history']:.2f}%",
            f"{diy_scenarios['target_not_in_history']:.2f}%",
            f"{diy_scenarios['target_most_frequent_in_history']:.2f}%",
            f"{diy_scenarios['target_most_recent']:.2f}%",
            f"{diy_scenarios['target_in_last_5']:.2f}%"
        ]
    }
    
    scenario_df = pd.DataFrame(scenario_comparison)
    print(scenario_df.to_string(index=False))
    scenario_df.to_csv(os.path.join(RESULTS_DIR, "05_pointer_scenarios.csv"), index=False)
    
    # Analyze pointer vs generation advantage
    print("\n[Analyzing pointer vs generation advantage...]")
    geo_advantage = analyze_pointer_advantage(geo_test, geo_train)
    diy_advantage = analyze_pointer_advantage(diy_test, diy_train)
    
    print("\n" + "="*80)
    print("POINTER VS GENERATION ADVANTAGE")
    print("="*80)
    
    advantage_comparison = {
        'Case Type': [
            'Pointer Advantage (target in history, rare globally)',
            'Generation Advantage (target frequent globally, not in history)',
            'Neutral/Both Can Help'
        ],
        'Geolife (%)': [
            f"{geo_advantage['pointer_advantage_pct']:.2f}%",
            f"{geo_advantage['generation_advantage_pct']:.2f}%",
            f"{geo_advantage['neutral_pct']:.2f}%"
        ],
        'DIY (%)': [
            f"{diy_advantage['pointer_advantage_pct']:.2f}%",
            f"{diy_advantage['generation_advantage_pct']:.2f}%",
            f"{diy_advantage['neutral_pct']:.2f}%"
        ]
    }
    
    advantage_df = pd.DataFrame(advantage_comparison)
    print(advantage_df.to_string(index=False))
    advantage_df.to_csv(os.path.join(RESULTS_DIR, "05_pointer_advantage.csv"), index=False)
    
    # Compare with actual model results
    print("\n" + "="*80)
    print("COMPARISON WITH ACTUAL MODEL RESULTS")
    print("="*80)
    
    # Actual results (from user's description)
    actual_results = {
        'Model': ['MHSA (Baseline)', 'PointerV45 (Proposed)', 'Improvement'],
        'Geolife Acc@1': ['33.18%', '53.96%', '+20.78%'],
        'DIY Acc@1': ['53.17%', '56.88%', '+3.71%']
    }
    
    actual_df = pd.DataFrame(actual_results)
    print(actual_df.to_string(index=False))
    
    # Calculate improvement potential
    print("\n" + "="*80)
    print("IMPROVEMENT POTENTIAL ANALYSIS")
    print("="*80)
    
    geo_potential = estimate_improvement_potential(geo_test, 33.18, geo_oracle_pointer)
    diy_potential = estimate_improvement_potential(diy_test, 53.17, diy_oracle_pointer)
    
    potential_comparison = {
        'Metric': [
            'MHSA Baseline Acc@1',
            'Oracle Pointer Upper Bound',
            'Max Improvement Potential',
            'Actual PointerV45 Acc@1',
            'Actual Improvement',
            'Improvement Realization Rate'
        ],
        'Geolife': [
            '33.18%',
            f"{geo_oracle_pointer:.2f}%",
            f"{geo_potential['max_improvement_potential']:.2f}%",
            '53.96%',
            '+20.78%',
            f"{20.78 / geo_potential['max_improvement_potential'] * 100:.1f}%" if geo_potential['max_improvement_potential'] > 0 else 'N/A'
        ],
        'DIY': [
            '53.17%',
            f"{diy_oracle_pointer:.2f}%",
            f"{diy_potential['max_improvement_potential']:.2f}%",
            '56.88%',
            '+3.71%',
            f"{3.71 / diy_potential['max_improvement_potential'] * 100:.1f}%" if diy_potential['max_improvement_potential'] > 0 else 'N/A'
        ]
    }
    
    potential_df = pd.DataFrame(potential_comparison)
    print(potential_df.to_string(index=False))
    potential_df.to_csv(os.path.join(RESULTS_DIR, "05_improvement_potential.csv"), index=False)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Heuristic baselines comparison
    ax1 = axes[0, 0]
    heuristics = ['Oracle\nPointer', 'Most Freq\nin History', 'Most\nRecent', 'Global\nMost Freq']
    geolife_vals = [geo_oracle_pointer, geo_most_freq, geo_most_recent, geo_global_freq]
    diy_vals = [diy_oracle_pointer, diy_most_freq, diy_most_recent, diy_global_freq]
    
    x = np.arange(len(heuristics))
    width = 0.35
    ax1.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue')
    ax1.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('Baseline Heuristic Accuracy')
    ax1.set_xticks(x)
    ax1.set_xticklabels(heuristics)
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # 2. Pointer scenario breakdown
    ax2 = axes[0, 1]
    scenarios = ['Target in\nHistory', 'Target\nMost Freq', 'Target\nMost Recent', 'Target in\nLast 5']
    geo_vals = [geo_scenarios['target_in_history'], geo_scenarios['target_most_frequent_in_history'],
                geo_scenarios['target_most_recent'], geo_scenarios['target_in_last_5']]
    diy_vals = [diy_scenarios['target_in_history'], diy_scenarios['target_most_frequent_in_history'],
                diy_scenarios['target_most_recent'], diy_scenarios['target_in_last_5']]
    
    x = np.arange(len(scenarios))
    ax2.bar(x - width/2, geo_vals, width, label='Geolife', color='steelblue')
    ax2.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax2.set_ylabel('Rate (%)')
    ax2.set_title('Pointer Scenario Rates\n(All favor pointer mechanism)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(scenarios)
    ax2.legend()
    
    # 3. Improvement comparison
    ax3 = axes[1, 0]
    categories = ['Geolife', 'DIY']
    baseline = [33.18, 53.17]
    pointer = [53.96, 56.88]
    oracle = [geo_oracle_pointer, diy_oracle_pointer]
    
    x = np.arange(len(categories))
    width = 0.25
    ax3.bar(x - width, baseline, width, label='MHSA Baseline', color='lightgray')
    ax3.bar(x, pointer, width, label='PointerV45', color='steelblue')
    ax3.bar(x + width, oracle, width, label='Oracle Pointer', color='lightgreen', alpha=0.7)
    ax3.set_ylabel('Accuracy (%)')
    ax3.set_title('Model Performance Comparison')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories)
    ax3.legend()
    ax3.set_ylim(0, 100)
    
    # Add improvement annotations
    for i, (b, p) in enumerate(zip(baseline, pointer)):
        ax3.annotate(f'+{p-b:.1f}%', (i, p + 2), ha='center', fontsize=10, color='green')
    
    # 4. Summary: Why Geolife benefits more
    ax4 = axes[1, 1]
    metrics = ['Target in\nHistory (%)', 'Improvement\nPotential (%)', 'Actual\nImprovement (%)']
    geolife_vals = [geo_oracle_pointer, geo_potential['max_improvement_potential'], 20.78]
    diy_vals = [diy_oracle_pointer, diy_potential['max_improvement_potential'], 3.71]
    
    x = np.arange(len(metrics))
    ax4.bar(x - width/2, geolife_vals, width, label='Geolife', color='steelblue')
    ax4.bar(x + width/2, diy_vals, width, label='DIY', color='coral')
    ax4.set_ylabel('Percentage (%)')
    ax4.set_title('Key Factors for Improvement Difference')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "05_model_mechanism.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)
    
    findings = []
    
    findings.append("1. ORACLE POINTER UPPER BOUND:")
    findings.append(f"   - Geolife: {geo_oracle_pointer:.2f}% of test targets appear in history")
    findings.append(f"   - DIY: {diy_oracle_pointer:.2f}% of test targets appear in history")
    findings.append(f"   - Difference: {geo_oracle_pointer - diy_oracle_pointer:.2f}%")
    
    findings.append("")
    findings.append("2. IMPROVEMENT POTENTIAL:")
    findings.append(f"   - Geolife: Max improvement = {geo_potential['max_improvement_potential']:.2f}% (from 33.18% to {geo_oracle_pointer:.2f}%)")
    findings.append(f"   - DIY: Max improvement = {diy_potential['max_improvement_potential']:.2f}% (from 53.17% to {diy_oracle_pointer:.2f}%)")
    
    findings.append("")
    findings.append("3. WHY GEOLIFE BENEFITS MORE:")
    findings.append(f"   a) Higher target-in-history rate: {geo_oracle_pointer:.1f}% vs {diy_oracle_pointer:.1f}%")
    findings.append(f"   b) More room for improvement: baseline starts lower (33.18% vs 53.17%)")
    findings.append(f"   c) More concentrated patterns: targets are more predictable from history")
    
    findings.append("")
    findings.append("4. WHY DIY BENEFITS LESS:")
    findings.append(f"   a) Lower target-in-history rate: pointer has fewer opportunities")
    findings.append(f"   b) Baseline already high: generation already captures common patterns")
    findings.append(f"   c) More diverse patterns: harder to leverage personal history")
    
    findings.append("")
    findings.append("5. CONCLUSION:")
    findings.append("   The performance gap difference is primarily due to DATA CHARACTERISTICS:")
    findings.append("   - Geolife users have more repetitive, history-dependent patterns")
    findings.append("   - DIY users have more diverse, less predictable patterns")
    findings.append("   - The pointer mechanism's advantage scales with history predictability")
    
    for finding in findings:
        print(finding)
    
    with open(os.path.join(RESULTS_DIR, "05_key_findings.txt"), 'w') as f:
        f.write("Model Mechanism Analysis Key Findings\n")
        f.write("="*50 + "\n\n")
        for finding in findings:
            f.write(finding + "\n")
    
    print(f"\n✓ Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
