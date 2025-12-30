"""
Visualization Script for Improvement Difference Analysis

This script generates visualizations to illustrate the key findings
about why Pointer V45 shows different improvements on Geolife vs DIY.

Outputs:
- Bar charts comparing key metrics
- Pie charts showing prediction categories
- Performance comparison charts
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter, defaultdict
from pathlib import Path

# Paths
GEOLIFE_TRAIN = "data/geolife_eps20/processed/geolife_eps20_prev7_train.pk"
GEOLIFE_TEST = "data/geolife_eps20/processed/geolife_eps20_prev7_test.pk"
DIY_TRAIN = "data/diy_eps50/processed/diy_eps50_prev7_train.pk"
DIY_TEST = "data/diy_eps50/processed/diy_eps50_prev7_test.pk"

OUTPUT_DIR = "scripts/analysis_improvement_differences_ok/results"

# Model performance
GEOLIFE_MHSA_ACC = 33.18
GEOLIFE_POINTER_ACC = 53.97
DIY_MHSA_ACC = 53.17
DIY_POINTER_ACC = 56.85


def load_data():
    """Load all datasets."""
    with open(GEOLIFE_TRAIN, 'rb') as f:
        geo_train = pickle.load(f)
    with open(GEOLIFE_TEST, 'rb') as f:
        geo_test = pickle.load(f)
    with open(DIY_TRAIN, 'rb') as f:
        diy_train = pickle.load(f)
    with open(DIY_TEST, 'rb') as f:
        diy_test = pickle.load(f)
    return geo_train, geo_test, diy_train, diy_test


def plot_performance_comparison(output_dir):
    """Plot performance comparison between MHSA and Pointer V45."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Absolute performance
    ax1 = axes[0]
    x = np.arange(2)
    width = 0.35
    
    mhsa_accs = [GEOLIFE_MHSA_ACC, DIY_MHSA_ACC]
    pointer_accs = [GEOLIFE_POINTER_ACC, DIY_POINTER_ACC]
    
    bars1 = ax1.bar(x - width/2, mhsa_accs, width, label='MHSA (Baseline)', color='#3498db', alpha=0.8)
    bars2 = ax1.bar(x + width/2, pointer_accs, width, label='Pointer V45 (Proposed)', color='#e74c3c', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=12)
    ax1.set_ylabel('Acc@1 (%)', fontsize=12)
    ax1.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Geolife', 'DIY'])
    ax1.legend(loc='upper left')
    ax1.set_ylim(0, 70)
    
    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    # Add improvement annotations
    ax1.annotate('', xy=(0 + width/2, GEOLIFE_POINTER_ACC), xytext=(0 - width/2, GEOLIFE_MHSA_ACC),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.annotate(f'+{GEOLIFE_POINTER_ACC - GEOLIFE_MHSA_ACC:.1f}pp', 
                xy=(0, (GEOLIFE_MHSA_ACC + GEOLIFE_POINTER_ACC)/2 + 3), ha='center', fontsize=11, color='green', fontweight='bold')
    
    ax1.annotate('', xy=(1 + width/2, DIY_POINTER_ACC), xytext=(1 - width/2, DIY_MHSA_ACC),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax1.annotate(f'+{DIY_POINTER_ACC - DIY_MHSA_ACC:.1f}pp', 
                xy=(1, (DIY_MHSA_ACC + DIY_POINTER_ACC)/2 + 3), ha='center', fontsize=11, color='green', fontweight='bold')
    
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Theoretical potential utilization
    ax2 = axes[1]
    
    geo_theoretical = 89.18
    diy_theoretical = 92.56
    
    categories = ['MHSA', 'Pointer V45', 'Remaining\nPotential']
    geo_values = [GEOLIFE_MHSA_ACC, GEOLIFE_POINTER_ACC - GEOLIFE_MHSA_ACC, geo_theoretical - GEOLIFE_POINTER_ACC]
    diy_values = [DIY_MHSA_ACC, DIY_POINTER_ACC - DIY_MHSA_ACC, diy_theoretical - DIY_POINTER_ACC]
    
    x = np.arange(3)
    bars1 = ax2.bar(x - width/2, geo_values, width, label='Geolife', color=['#3498db', '#e74c3c', '#95a5a6'], alpha=0.8)
    bars2 = ax2.bar(x + width/2, diy_values, width, label='DIY', color=['#2980b9', '#c0392b', '#7f8c8d'], alpha=0.8)
    
    ax2.set_xlabel('Performance Component', fontsize=12)
    ax2.set_ylabel('Percentage Points', fontsize=12)
    ax2.set_title('Theoretical Potential Breakdown', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories)
    
    # Add value labels
    for i, (b1, b2) in enumerate(zip(bars1, bars2)):
        ax2.annotate(f'{b1.get_height():.1f}', xy=(b1.get_x() + b1.get_width()/2, b1.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
        ax2.annotate(f'{b2.get_height():.1f}', xy=(b2.get_x() + b2.get_width()/2, b2.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=9)
    
    # Custom legend
    colors = ['#3498db', '#e74c3c', '#95a5a6']
    labels = ['MHSA Baseline', 'Pointer V45 Gain', 'Remaining Potential']
    patches = [mpatches.Patch(color=c, label=l, alpha=0.8) for c, l in zip(colors, labels)]
    ax2.legend(handles=patches, loc='upper right')
    
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_performance_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/05_performance_comparison.png")


def plot_key_factors(geo_train, geo_test, diy_train, diy_test, output_dir):
    """Plot key factors explaining the difference."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Calculate metrics
    def markov_acc(train, test):
        transitions = defaultdict(Counter)
        for s in train:
            full_seq = list(s['X']) + [s['Y']]
            for i in range(len(full_seq) - 1):
                transitions[full_seq[i]][full_seq[i+1]] += 1
        
        correct = 0
        for s in test:
            last_loc = s['X'][-1]
            if last_loc in transitions and transitions[last_loc]:
                pred = transitions[last_loc].most_common(1)[0][0]
                if pred == s['Y']:
                    correct += 1
        return correct / len(test) * 100
    
    def unseen_trans_pct(train, test):
        transitions = defaultdict(Counter)
        for s in train:
            full_seq = list(s['X']) + [s['Y']]
            for i in range(len(full_seq) - 1):
                transitions[full_seq[i]][full_seq[i+1]] += 1
        
        unseen = sum(1 for s in test if s['X'][-1] not in transitions)
        return unseen / len(test) * 100
    
    geo_markov = markov_acc(geo_train, geo_test)
    diy_markov = markov_acc(diy_train, diy_test)
    geo_unseen = unseen_trans_pct(geo_train, geo_test)
    diy_unseen = unseen_trans_pct(diy_train, diy_test)
    
    geo_in_hist = sum(1 for s in geo_test if s['Y'] in s['X']) / len(geo_test) * 100
    diy_in_hist = sum(1 for s in diy_test if s['Y'] in s['X']) / len(diy_test) * 100
    
    geo_last = sum(1 for s in geo_test if s['Y'] == s['X'][-1]) / len(geo_test) * 100
    diy_last = sum(1 for s in diy_test if s['Y'] == s['X'][-1]) / len(diy_test) * 100
    
    # Plot 1: Markov Accuracy
    ax1 = axes[0, 0]
    x = np.arange(2)
    bars = ax1.bar(x, [geo_markov, diy_markov], color=['#3498db', '#e74c3c'], alpha=0.8)
    ax1.set_xlabel('Dataset')
    ax1.set_ylabel('Accuracy (%)')
    ax1.set_title('1st-Order Markov Accuracy\n(Pattern Predictability)', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(['Geolife', 'DIY'])
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
    ax1.set_ylim(0, 50)
    ax1.grid(axis='y', alpha=0.3)
    ax1.annotate('DIY 1.6x more\npredictable', xy=(1, diy_markov + 5), ha='center', 
                fontsize=10, color='green', fontweight='bold')
    
    # Plot 2: Unseen Transitions
    ax2 = axes[0, 1]
    bars = ax2.bar(x, [geo_unseen, diy_unseen], color=['#3498db', '#e74c3c'], alpha=0.8)
    ax2.set_xlabel('Dataset')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_title('Unseen Transitions in Test\n(Learning Difficulty)', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Geolife', 'DIY'])
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
    ax2.set_ylim(0, 35)
    ax2.grid(axis='y', alpha=0.3)
    ax2.annotate('Geolife 5.7x more\nunseen transitions', xy=(0, geo_unseen + 3), ha='center', 
                fontsize=10, color='red', fontweight='bold')
    
    # Plot 3: Target in History
    ax3 = axes[1, 0]
    bars = ax3.bar(x, [geo_in_hist, diy_in_hist], color=['#3498db', '#e74c3c'], alpha=0.8)
    ax3.set_xlabel('Dataset')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Target in Sequence History\n(Pointer Applicability)', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(['Geolife', 'DIY'])
    for bar in bars:
        height = bar.get_height()
        ax3.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=11)
    ax3.set_ylim(0, 100)
    ax3.grid(axis='y', alpha=0.3)
    ax3.annotate('Similar (~84%)', xy=(0.5, 95), ha='center', fontsize=10, color='gray')
    
    # Plot 4: MHSA % of Theoretical
    ax4 = axes[1, 1]
    geo_mhsa_util = GEOLIFE_MHSA_ACC / 89.18 * 100
    diy_mhsa_util = DIY_MHSA_ACC / 92.56 * 100
    geo_ptr_util = GEOLIFE_POINTER_ACC / 89.18 * 100
    diy_ptr_util = DIY_POINTER_ACC / 92.56 * 100
    
    width = 0.35
    bars1 = ax4.bar(x - width/2, [geo_mhsa_util, diy_mhsa_util], width, label='MHSA', color='#3498db', alpha=0.8)
    bars2 = ax4.bar(x + width/2, [geo_ptr_util, diy_ptr_util], width, label='Pointer V45', color='#e74c3c', alpha=0.8)
    
    ax4.set_xlabel('Dataset')
    ax4.set_ylabel('% of Theoretical Potential')
    ax4.set_title('Utilization of Theoretical Potential\n(KEY INSIGHT)', fontsize=12, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Geolife', 'DIY'])
    ax4.legend(loc='upper left')
    
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}%', xy=(bar.get_x() + bar.get_width()/2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', fontsize=10)
    
    ax4.set_ylim(0, 80)
    ax4.grid(axis='y', alpha=0.3)
    ax4.axhline(y=60, color='gray', linestyle='--', alpha=0.5)
    ax4.annotate('Both reach ~60%\nbut from different baselines', xy=(0.5, 65), ha='center', 
                fontsize=10, color='green', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_key_factors.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/05_key_factors.png")


def plot_root_cause_diagram(output_dir):
    """Create a visual diagram explaining the root cause."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(8, 9.5, 'Root Cause Analysis: Why Different Improvements?', 
            fontsize=16, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    # Geolife box
    rect1 = plt.Rectangle((0.5, 5), 7, 3.5, linewidth=2, edgecolor='#3498db', facecolor='#ebf5fb', alpha=0.8)
    ax.add_patch(rect1)
    ax.text(4, 8, 'GEOLIFE', fontsize=14, fontweight='bold', ha='center', color='#3498db')
    ax.text(4, 7.2, 'MHSA: 33.18%', fontsize=11, ha='center')
    ax.text(4, 6.6, 'Pointer V45: 53.97%', fontsize=11, ha='center')
    ax.text(4, 6.0, 'Improvement: +20.79pp', fontsize=12, ha='center', fontweight='bold', color='green')
    ax.text(4, 5.3, 'MHSA uses only 37.2% of potential', fontsize=10, ha='center', color='red')
    
    # DIY box
    rect2 = plt.Rectangle((8.5, 5), 7, 3.5, linewidth=2, edgecolor='#e74c3c', facecolor='#fdedec', alpha=0.8)
    ax.add_patch(rect2)
    ax.text(12, 8, 'DIY', fontsize=14, fontweight='bold', ha='center', color='#e74c3c')
    ax.text(12, 7.2, 'MHSA: 53.17%', fontsize=11, ha='center')
    ax.text(12, 6.6, 'Pointer V45: 56.85%', fontsize=11, ha='center')
    ax.text(12, 6.0, 'Improvement: +3.68pp', fontsize=12, ha='center', fontweight='bold', color='green')
    ax.text(12, 5.3, 'MHSA already uses 57.4% of potential', fontsize=10, ha='center', color='blue')
    
    # Why box - Geolife
    rect3 = plt.Rectangle((0.5, 2), 7, 2.5, linewidth=2, edgecolor='#3498db', facecolor='white', alpha=0.8)
    ax.add_patch(rect3)
    ax.text(4, 4.1, 'Why MHSA struggles:', fontsize=11, fontweight='bold', ha='center', color='#3498db')
    ax.text(4, 3.5, '• 22.99% unseen transitions', fontsize=10, ha='center')
    ax.text(4, 3.0, '• Only 21.25% Markov accuracy', fontsize=10, ha='center')
    ax.text(4, 2.5, '• Less predictable patterns', fontsize=10, ha='center')
    
    # Why box - DIY
    rect4 = plt.Rectangle((8.5, 2), 7, 2.5, linewidth=2, edgecolor='#e74c3c', facecolor='white', alpha=0.8)
    ax.add_patch(rect4)
    ax.text(12, 4.1, 'Why MHSA succeeds:', fontsize=11, fontweight='bold', ha='center', color='#e74c3c')
    ax.text(12, 3.5, '• Only 4.04% unseen transitions', fontsize=10, ha='center')
    ax.text(12, 3.0, '• 34.49% Markov accuracy', fontsize=10, ha='center')
    ax.text(12, 2.5, '• More predictable patterns', fontsize=10, ha='center')
    
    # Conclusion box
    rect5 = plt.Rectangle((3, 0.3), 10, 1.3, linewidth=3, edgecolor='green', facecolor='#e8f8f5', alpha=0.8)
    ax.add_patch(rect5)
    ax.text(8, 1.2, 'CONCLUSION', fontsize=12, fontweight='bold', ha='center', color='green')
    ax.text(8, 0.6, 'The improvement difference is due to MHSA baseline performance, not pointer mechanism.', 
            fontsize=11, ha='center')
    
    # Arrows
    ax.annotate('', xy=(4, 4.5), xytext=(4, 5),
                arrowprops=dict(arrowstyle='->', color='#3498db', lw=2))
    ax.annotate('', xy=(12, 4.5), xytext=(12, 5),
                arrowprops=dict(arrowstyle='->', color='#e74c3c', lw=2))
    ax.annotate('', xy=(8, 1.6), xytext=(4, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    ax.annotate('', xy=(8, 1.6), xytext=(12, 2),
                arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/05_root_cause_diagram.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/05_root_cause_diagram.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # Load data
    geo_train, geo_test, diy_train, diy_test = load_data()
    
    # Generate plots
    print("\n1. Performance Comparison Plot")
    plot_performance_comparison(OUTPUT_DIR)
    
    print("\n2. Key Factors Plot")
    plot_key_factors(geo_train, geo_test, diy_train, diy_test, OUTPUT_DIR)
    
    print("\n3. Root Cause Diagram")
    plot_root_cause_diagram(OUTPUT_DIR)
    
    print(f"\nAll visualizations saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
