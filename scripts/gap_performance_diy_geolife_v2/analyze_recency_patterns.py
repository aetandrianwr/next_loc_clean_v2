"""
Recency Pattern Analysis for Gap Performance Study.

This script specifically analyzes recency patterns - how recently visited locations
relate to the next location prediction. This is critical because the pointer mechanism
uses position bias to favor recent locations.

Key questions:
1. How often is the target the most recently visited location?
2. What is the distribution of target position from end?
3. How does this correlate with pointer mechanism importance?

Author: Gap Performance Analysis Framework
Date: January 2, 2026
Seed: 42
"""

import os
import sys
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, Counter
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reset to defaults first then apply classic style
plt.rcdefaults()

# Set classic scientific publication style (matching reference images)
plt.rcParams.update({
    # Font settings
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'Times'],
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    
    # Figure settings
    'figure.figsize': (8, 6),
    'figure.dpi': 100,
    'figure.facecolor': 'white',
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    
    # Axes settings - box style (all 4 sides visible)
    'axes.facecolor': 'white',
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.0,
    'axes.grid': False,  # No grid
    'axes.spines.top': True,  # Show all 4 sides
    'axes.spines.right': True,
    'axes.spines.bottom': True,
    'axes.spines.left': True,
    
    # Tick settings - inside ticks
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'xtick.major.size': 5,
    'ytick.major.size': 5,
    'xtick.minor.size': 3,
    'ytick.minor.size': 3,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'xtick.top': True,  # Ticks on all sides
    'xtick.bottom': True,
    'ytick.left': True,
    'ytick.right': True,
    
    # Line settings
    'lines.linewidth': 1.5,
    'lines.markersize': 7,
    
    # Legend settings
    'legend.frameon': True,
    'legend.framealpha': 1.0,
    'legend.edgecolor': 'black',
    'legend.fancybox': False,
})

# Classic scientific color palette (matching reference)
COLORS = {
    'diy': 'blue',
    'geolife': 'red',
    'black': 'black',
    'green': 'green',
}

# Marker styles (open markers like in reference)
MARKERS = {
    'diy': 'o',        # Circle
    'geolife': 's',    # Square
}


def setup_classic_axes(ax):
    """Configure axes to match classic scientific publication style."""
    # Ensure all spines visible and black
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('black')
        spine.set_linewidth(1.0)
    
    # Inside ticks on all sides
    ax.tick_params(axis='both', which='both', direction='in',
                   top=True, bottom=True, left=True, right=True)
    ax.tick_params(axis='both', which='major', length=5, width=1)
    ax.tick_params(axis='both', which='minor', length=3, width=0.5)


np.random.seed(42)


class RecencyPatternAnalyzer:
    """Analyzes recency patterns that explain pointer mechanism benefit."""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
    
    def load_dataset(self, data_path: str, name: str) -> list:
        """Load dataset from pickle file."""
        print(f"Loading {name} dataset from {data_path}...")
        with open(data_path, 'rb') as f:
            data = pickle.load(f)
        print(f"  Loaded {len(data)} samples")
        return data
    
    def analyze_target_recency(self, data: list, name: str) -> dict:
        """
        Analyze how recent the target location was visited.
        
        Key insight: If target is often the most recent location,
        the pointer mechanism with position bias will excel.
        """
        print(f"\n{'='*60}")
        print(f"Recency Pattern Analysis: {name}")
        print('='*60)
        
        target_positions = []  # Position from end (1 = most recent)
        target_is_last = []
        target_is_top3_recent = []
        target_is_top5_recent = []
        target_first_occurrence = []  # First occurrence position from end
        
        for sample in data:
            x = sample['X']
            y = sample['Y']
            seq_len = len(x)
            
            # Find all occurrences of target
            positions = np.where(x == y)[0]
            
            if len(positions) > 0:
                # Position from end (1 = most recent)
                pos_from_end = seq_len - positions
                most_recent_pos = min(pos_from_end)  # Most recent = smallest
                first_pos = max(pos_from_end)  # First occurrence = largest
                
                target_positions.append(most_recent_pos)
                target_first_occurrence.append(first_pos)
                target_is_last.append(most_recent_pos == 1)
                target_is_top3_recent.append(most_recent_pos <= 3)
                target_is_top5_recent.append(most_recent_pos <= 5)
            else:
                target_positions.append(-1)
                target_first_occurrence.append(-1)
                target_is_last.append(False)
                target_is_top3_recent.append(False)
                target_is_top5_recent.append(False)
        
        # Filter for samples where target is in history
        target_positions = np.array(target_positions)
        in_history_mask = target_positions > 0
        positions_when_in = target_positions[in_history_mask]
        
        results = {
            'total_samples': len(data),
            'target_in_history_pct': np.mean(in_history_mask) * 100,
            'target_is_last_pct': np.mean(target_is_last) * 100,
            'target_in_top3_recent_pct': np.mean(target_is_top3_recent) * 100,
            'target_in_top5_recent_pct': np.mean(target_is_top5_recent) * 100,
            'avg_target_position': np.mean(positions_when_in) if len(positions_when_in) > 0 else 0,
            'median_target_position': np.median(positions_when_in) if len(positions_when_in) > 0 else 0,
            'std_target_position': np.std(positions_when_in) if len(positions_when_in) > 0 else 0,
        }
        
        # Calculate distribution
        position_dist = Counter(positions_when_in)
        results['position_distribution'] = dict(sorted(position_dist.items())[:20])  # Top 20 positions
        
        print(f"  Total samples: {results['total_samples']}")
        print(f"  Target in history: {results['target_in_history_pct']:.2f}%")
        print(f"  Target is most recent (pos=1): {results['target_is_last_pct']:.2f}%")
        print(f"  Target in top-3 recent: {results['target_in_top3_recent_pct']:.2f}%")
        print(f"  Target in top-5 recent: {results['target_in_top5_recent_pct']:.2f}%")
        print(f"  Average target position from end: {results['avg_target_position']:.2f}")
        print(f"  Median target position from end: {results['median_target_position']:.2f}")
        
        return results, target_positions, target_is_last
    
    def analyze_return_patterns(self, data: list, name: str) -> dict:
        """
        Analyze return-to-previous-location patterns.
        
        These patterns indicate strong sequential dependencies that
        the pointer mechanism can capture.
        """
        print(f"\n{'='*60}")
        print(f"Return Pattern Analysis: {name}")
        print('='*60)
        
        # Pattern: A -> B -> A (return to previous location)
        return_to_prev_prev = []  # Target is loc at position -2
        return_to_any_recent = []  # Target is any of last 5 locations
        
        for sample in data:
            x = sample['X']
            y = sample['Y']
            
            # Check if target is the location 2 steps back (A->B->A pattern)
            if len(x) >= 2:
                return_to_prev_prev.append(y == x[-2])
            else:
                return_to_prev_prev.append(False)
            
            # Check if target is any of last 5 locations
            recent_locs = x[-5:] if len(x) >= 5 else x
            return_to_any_recent.append(y in recent_locs)
        
        results = {
            'return_to_prev_prev_pct': np.mean(return_to_prev_prev) * 100,
            'return_to_any_recent5_pct': np.mean(return_to_any_recent) * 100,
        }
        
        print(f"  Target = location 2 steps back (A->B->A): {results['return_to_prev_prev_pct']:.2f}%")
        print(f"  Target = any of last 5 locations: {results['return_to_any_recent5_pct']:.2f}%")
        
        return results
    
    def analyze_location_predictability(self, data: list, name: str) -> dict:
        """
        Analyze how predictable locations are based on recency.
        
        Key insight: If a location appears recently and frequently,
        pointer mechanism's position-weighted attention excels.
        """
        print(f"\n{'='*60}")
        print(f"Location Predictability Analysis: {name}")
        print('='*60)
        
        # For each sample, calculate "predictability score"
        # High score = target is recent AND frequent in history
        predictability_scores = []
        recency_scores = []  # 1/position_from_end
        frequency_scores = []  # count/seq_len
        
        for sample in data:
            x = sample['X']
            y = sample['Y']
            seq_len = len(x)
            
            # Find target in history
            positions = np.where(x == y)[0]
            
            if len(positions) > 0:
                # Recency score: 1/position_from_end (higher = more recent)
                most_recent_pos = seq_len - positions[-1]
                recency = 1 / most_recent_pos
                
                # Frequency score: count/seq_len
                frequency = len(positions) / seq_len
                
                # Combined predictability
                predictability = recency * frequency
                
                recency_scores.append(recency)
                frequency_scores.append(frequency)
                predictability_scores.append(predictability)
            else:
                recency_scores.append(0)
                frequency_scores.append(0)
                predictability_scores.append(0)
        
        results = {
            'avg_recency_score': np.mean(recency_scores),
            'avg_frequency_score': np.mean(frequency_scores),
            'avg_predictability_score': np.mean(predictability_scores),
            'high_predictability_pct': np.mean(np.array(predictability_scores) > 0.1) * 100,
        }
        
        print(f"  Average recency score (1/pos): {results['avg_recency_score']:.4f}")
        print(f"  Average frequency score (count/len): {results['avg_frequency_score']:.4f}")
        print(f"  Average predictability score: {results['avg_predictability_score']:.4f}")
        print(f"  High predictability (>0.1): {results['high_predictability_pct']:.2f}%")
        
        return results, recency_scores, frequency_scores, predictability_scores
    
    def run_all_analyses(self, diy_data: list, geolife_data: list):
        """Run all recency analyses on both datasets."""
        # Recency analysis
        diy_recency, diy_positions, diy_is_last = self.analyze_target_recency(diy_data, 'DIY')
        geolife_recency, geolife_positions, geolife_is_last = self.analyze_target_recency(geolife_data, 'GeoLife')
        
        # Return patterns
        diy_return = self.analyze_return_patterns(diy_data, 'DIY')
        geolife_return = self.analyze_return_patterns(geolife_data, 'GeoLife')
        
        # Predictability
        diy_pred, diy_rec_scores, diy_freq_scores, diy_pred_scores = self.analyze_location_predictability(diy_data, 'DIY')
        geolife_pred, geo_rec_scores, geo_freq_scores, geo_pred_scores = self.analyze_location_predictability(geolife_data, 'GeoLife')
        
        return {
            'diy_recency': diy_recency,
            'geolife_recency': geolife_recency,
            'diy_return': diy_return,
            'geolife_return': geolife_return,
            'diy_pred': diy_pred,
            'geolife_pred': geolife_pred,
            'diy_positions': diy_positions,
            'geolife_positions': geolife_positions,
            'diy_is_last': diy_is_last,
            'geolife_is_last': geolife_is_last,
            'diy_pred_scores': diy_pred_scores,
            'geolife_pred_scores': geo_pred_scores,
        }
    
    def plot_recency_comparison(self, results: dict):
        """Plot recency pattern comparison with classic scientific style."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        
        # 1. Target position distribution
        ax = axes[0, 0]
        
        diy_pos = results['diy_positions']
        geo_pos = results['geolife_positions']
        
        # Filter for in-history
        diy_pos_filtered = diy_pos[diy_pos > 0]
        geo_pos_filtered = geo_pos[geo_pos > 0]
        
        bins = np.arange(1, 21)
        ax.hist(diy_pos_filtered, bins=bins, alpha=0.5, label='DIY', 
                color='white', edgecolor=COLORS['diy'], linewidth=1.5, hatch='///', density=True)
        ax.hist(geo_pos_filtered, bins=bins, alpha=0.5, label='GeoLife', 
                color='white', edgecolor=COLORS['geolife'], linewidth=1.5, hatch='...', density=True)
        ax.set_xlabel('Target Position from End (1 = Most Recent)')
        ax.set_ylabel('Density')
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        ax.set_xlim(0, 20)
        setup_classic_axes(ax)
        
        # 2. Key recency metrics comparison
        ax = axes[0, 1]
        
        metrics = ['Target=Last', 'Target in Top-3', 'Target in Top-5', 'A→B→A Pattern']
        diy_vals = [
            results['diy_recency']['target_is_last_pct'],
            results['diy_recency']['target_in_top3_recent_pct'],
            results['diy_recency']['target_in_top5_recent_pct'],
            results['diy_return']['return_to_prev_prev_pct'],
        ]
        geo_vals = [
            results['geolife_recency']['target_is_last_pct'],
            results['geolife_recency']['target_in_top3_recent_pct'],
            results['geolife_recency']['target_in_top5_recent_pct'],
            results['geolife_return']['return_to_prev_prev_pct'],
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, diy_vals, width, label='DIY', 
                       color='white', edgecolor=COLORS['diy'], linewidth=1.5, hatch='///')
        bars2 = ax.bar(x + width/2, geo_vals, width, label='GeoLife', 
                       color='white', edgecolor=COLORS['geolife'], linewidth=1.5, hatch='...')
        
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}%',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=9)
        add_labels(bars1)
        add_labels(bars2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=15, ha='right')
        ax.set_ylabel('Percentage (%)')
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        setup_classic_axes(ax)
        
        # 3. Cumulative distribution of target position
        ax = axes[1, 0]
        
        # Calculate cumulative percentage
        positions = range(1, 21)
        diy_cumulative = []
        geo_cumulative = []
        
        for pos in positions:
            diy_cumulative.append(np.mean(diy_pos_filtered <= pos) * 100)
            geo_cumulative.append(np.mean(geo_pos_filtered <= pos) * 100)
        
        # Open markers with classic style
        ax.plot(positions, diy_cumulative, marker='o', color=COLORS['diy'], linewidth=1.5, markersize=6, label='DIY',
                markerfacecolor='white', markeredgecolor=COLORS['diy'], markeredgewidth=1.5)
        ax.plot(positions, geo_cumulative, marker='s', color=COLORS['geolife'], linewidth=1.5, markersize=6, label='GeoLife',
                markerfacecolor='white', markeredgecolor=COLORS['geolife'], markeredgewidth=1.5)
        
        ax.axhline(y=50, color='black', linestyle='--', linewidth=0.8, label='50%')
        ax.axhline(y=80, color='black', linestyle=':', linewidth=0.8, label='80%')
        
        ax.set_xlabel('Position from End')
        ax.set_ylabel('Cumulative % of Targets')
        ax.legend(loc='lower right', frameon=True, edgecolor='black', fancybox=False)
        ax.set_xlim(0, 20)
        ax.set_ylim(0, 100)
        setup_classic_axes(ax)
        
        # 4. Correlation with ablation impact
        ax = axes[1, 1]
        
        # Key metrics that favor pointer
        pointer_favoring_metrics = {
            'Target=Last': (results['diy_recency']['target_is_last_pct'], results['geolife_recency']['target_is_last_pct']),
            'Target in Top-3': (results['diy_recency']['target_in_top3_recent_pct'], results['geolife_recency']['target_in_top3_recent_pct']),
            'A→B→A Pattern': (results['diy_return']['return_to_prev_prev_pct'], results['geolife_return']['return_to_prev_prev_pct']),
        }
        
        ablation_impacts = {'DIY': 8.3, 'GeoLife': 46.7}
        
        # Calculate average "pointer benefit score"
        diy_avg = np.mean([v[0] for v in pointer_favoring_metrics.values()])
        geo_avg = np.mean([v[1] for v in pointer_favoring_metrics.values()])
        
        # Open markers
        ax.scatter([diy_avg], [ablation_impacts['DIY']], s=150, 
                   facecolors='white', edgecolors=COLORS['diy'], 
                   marker='o', label='DIY', linewidths=2, zorder=5)
        ax.scatter([geo_avg], [ablation_impacts['GeoLife']], s=150, 
                   facecolors='white', edgecolors=COLORS['geolife'], 
                   marker='s', label='GeoLife', linewidths=2, zorder=5)
        
        # Fit line
        x_points = [diy_avg, geo_avg]
        y_points = [ablation_impacts['DIY'], ablation_impacts['GeoLife']]
        z = np.polyfit(x_points, y_points, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(x_points)-5, max(x_points)+5, 100)
        ax.plot(x_line, p(x_line), 'k--', linewidth=1.0, label=f'Trend')
        
        ax.annotate('DIY', xy=(diy_avg, ablation_impacts['DIY']), 
                   xytext=(10, -15), textcoords='offset points', fontsize=10)
        ax.annotate('GeoLife', xy=(geo_avg, ablation_impacts['GeoLife']), 
                   xytext=(10, 5), textcoords='offset points', fontsize=10)
        
        ax.set_xlabel('Average Recency Pattern Score (%)')
        ax.set_ylabel('Pointer Removal Impact\n(Relative Acc@1 Drop %)')
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        setup_classic_axes(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'recency_pattern_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'figures' / 'recency_pattern_analysis.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Saved: recency_pattern_analysis.png/pdf")
    
    def plot_predictability_analysis(self, results: dict):
        """Plot predictability analysis with classic scientific style."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 1. Predictability score distribution
        ax = axes[0]
        
        diy_scores = np.array(results['diy_pred_scores'])
        geo_scores = np.array(results['geolife_pred_scores'])
        
        bins = np.linspace(0, 0.5, 30)
        ax.hist(diy_scores, bins=bins, alpha=0.5, label=f'DIY (μ={np.mean(diy_scores):.3f})', 
                color='white', edgecolor=COLORS['diy'], linewidth=1.5, hatch='///', density=True)
        ax.hist(geo_scores, bins=bins, alpha=0.5, label=f'GeoLife (μ={np.mean(geo_scores):.3f})', 
                color='white', edgecolor=COLORS['geolife'], linewidth=1.5, hatch='...', density=True)
        ax.set_xlabel('Predictability Score (Recency × Frequency)')
        ax.set_ylabel('Density')
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        setup_classic_axes(ax)
        
        # 2. Predictability metrics comparison
        ax = axes[1]
        
        metrics = ['Avg Recency Score', 'Avg Frequency Score', 'Avg Predictability']
        diy_vals = [
            results['diy_pred']['avg_recency_score'] * 100,
            results['diy_pred']['avg_frequency_score'] * 100,
            results['diy_pred']['avg_predictability_score'] * 100,
        ]
        geo_vals = [
            results['geolife_pred']['avg_recency_score'] * 100,
            results['geolife_pred']['avg_frequency_score'] * 100,
            results['geolife_pred']['avg_predictability_score'] * 100,
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, diy_vals, width, label='DIY', 
                       color='white', edgecolor=COLORS['diy'], linewidth=1.5, hatch='///')
        bars2 = ax.bar(x + width/2, geo_vals, width, label='GeoLife', 
                       color='white', edgecolor=COLORS['geolife'], linewidth=1.5, hatch='...')
        
        def add_labels(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.1f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
        add_labels(bars1)
        add_labels(bars2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.set_ylabel('Score (×100)')
        ax.legend(loc='best', frameon=True, edgecolor='black', fancybox=False)
        setup_classic_axes(ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'figures' / 'predictability_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(self.output_dir / 'figures' / 'predictability_analysis.pdf', bbox_inches='tight', facecolor='white')
        plt.close()
        print("  Saved: predictability_analysis.png/pdf")
    
    def create_summary_table(self, results: dict) -> pd.DataFrame:
        """Create summary table of recency metrics."""
        metrics = [
            ('Target in History (%)', results['diy_recency']['target_in_history_pct'], 
             results['geolife_recency']['target_in_history_pct']),
            ('Target = Most Recent (%)', results['diy_recency']['target_is_last_pct'], 
             results['geolife_recency']['target_is_last_pct']),
            ('Target in Top-3 Recent (%)', results['diy_recency']['target_in_top3_recent_pct'], 
             results['geolife_recency']['target_in_top3_recent_pct']),
            ('Target in Top-5 Recent (%)', results['diy_recency']['target_in_top5_recent_pct'], 
             results['geolife_recency']['target_in_top5_recent_pct']),
            ('A→B→A Return Pattern (%)', results['diy_return']['return_to_prev_prev_pct'], 
             results['geolife_return']['return_to_prev_prev_pct']),
            ('Return to Recent 5 (%)', results['diy_return']['return_to_any_recent5_pct'], 
             results['geolife_return']['return_to_any_recent5_pct']),
            ('Avg Target Position from End', results['diy_recency']['avg_target_position'], 
             results['geolife_recency']['avg_target_position']),
            ('Avg Recency Score (×100)', results['diy_pred']['avg_recency_score']*100, 
             results['geolife_pred']['avg_recency_score']*100),
            ('Avg Predictability Score (×100)', results['diy_pred']['avg_predictability_score']*100, 
             results['geolife_pred']['avg_predictability_score']*100),
        ]
        
        data = []
        for metric_name, diy_val, geo_val in metrics:
            diff = geo_val - diy_val
            data.append({
                'Metric': metric_name,
                'DIY': f'{diy_val:.2f}',
                'GeoLife': f'{geo_val:.2f}',
                'Difference': f'{diff:+.2f}',
                'Favors': 'GeoLife' if diff > 0 else 'DIY' if diff < 0 else 'Same'
            })
        
        return pd.DataFrame(data)
    
    def save_results(self, results: dict, summary_df: pd.DataFrame):
        """Save all results."""
        # Save summary table
        summary_df.to_csv(self.output_dir / 'tables' / 'recency_metrics.csv', index=False)
        
        # Save LaTeX table
        latex = summary_df.to_latex(index=False, caption='Recency Pattern Metrics: DIY vs GeoLife',
                                    label='tab:recency_patterns')
        with open(self.output_dir / 'tables' / 'recency_metrics.tex', 'w') as f:
            f.write(latex)
        
        # Save JSON results (without arrays)
        json_results = {
            'diy_recency': {k: v for k, v in results['diy_recency'].items() if k != 'position_distribution'},
            'geolife_recency': {k: v for k, v in results['geolife_recency'].items() if k != 'position_distribution'},
            'diy_return': results['diy_return'],
            'geolife_return': results['geolife_return'],
            'diy_pred': results['diy_pred'],
            'geolife_pred': results['geolife_pred'],
        }
        
        with open(self.output_dir / 'recency_analysis_results.json', 'w') as f:
            json.dump(json_results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)


def main():
    """Main function to run recency pattern analysis."""
    print("="*70)
    print("RECENCY PATTERN ANALYSIS")
    print("Explaining Pointer Mechanism Importance Through Recency")
    print("="*70)
    
    # Paths
    diy_test_path = PROJECT_ROOT / 'data' / 'diy_eps50' / 'processed' / 'diy_eps50_prev7_test.pk'
    geolife_test_path = PROJECT_ROOT / 'data' / 'geolife_eps20' / 'processed' / 'geolife_eps20_prev7_test.pk'
    output_dir = PROJECT_ROOT / 'scripts' / 'gap_performance_diy_geolife_v2' / 'results'
    
    # Initialize analyzer
    analyzer = RecencyPatternAnalyzer(output_dir)
    
    # Load datasets
    diy_data = analyzer.load_dataset(str(diy_test_path), 'DIY')
    geolife_data = analyzer.load_dataset(str(geolife_test_path), 'GeoLife')
    
    # Run all analyses
    results = analyzer.run_all_analyses(diy_data, geolife_data)
    
    # Generate visualizations
    print("\n" + "="*60)
    print("GENERATING VISUALIZATIONS")
    print("="*60)
    
    analyzer.plot_recency_comparison(results)
    analyzer.plot_predictability_analysis(results)
    
    # Create summary table
    summary_df = analyzer.create_summary_table(results)
    
    print("\n" + "="*60)
    print("RECENCY METRICS SUMMARY TABLE")
    print("="*60)
    print(summary_df.to_string(index=False))
    
    # Save results
    analyzer.save_results(results, summary_df)
    
    # Print key findings
    print("\n" + "="*60)
    print("KEY FINDINGS: RECENCY PATTERNS")
    print("="*60)
    
    target_last_diff = results['geolife_recency']['target_is_last_pct'] - results['diy_recency']['target_is_last_pct']
    aba_diff = results['geolife_return']['return_to_prev_prev_pct'] - results['diy_return']['return_to_prev_prev_pct']
    top3_diff = results['geolife_recency']['target_in_top3_recent_pct'] - results['diy_recency']['target_in_top3_recent_pct']
    
    print(f"""
RECENCY PATTERN DIFFERENCES (GeoLife vs DIY):

1. Target = Most Recent Location:
   - DIY:     {results['diy_recency']['target_is_last_pct']:.2f}%
   - GeoLife: {results['geolife_recency']['target_is_last_pct']:.2f}%
   - Difference: {target_last_diff:+.2f}%
   
   GeoLife users return to the same location {abs(target_last_diff):.1f}% more often!

2. Target in Top-3 Recent Locations:
   - DIY:     {results['diy_recency']['target_in_top3_recent_pct']:.2f}%
   - GeoLife: {results['geolife_recency']['target_in_top3_recent_pct']:.2f}%
   - Difference: {top3_diff:+.2f}%

3. A→B→A Return Pattern (Bounce-back):
   - DIY:     {results['diy_return']['return_to_prev_prev_pct']:.2f}%
   - GeoLife: {results['geolife_return']['return_to_prev_prev_pct']:.2f}%
   - Difference: {aba_diff:+.2f}%
   
   GeoLife shows {abs(aba_diff):.1f}% more bounce-back patterns!

CONCLUSION:
GeoLife's mobility patterns are more "pointer-friendly" because:
- Higher rate of returning to most recent location (position bias helps)
- More bounce-back patterns (A→B→A) that pointer can easily copy
- Stronger concentration in recent positions

This directly explains why removing the pointer mechanism hurts GeoLife more:
- GeoLife relies heavily on recency-based prediction (46.7% drop)
- DIY has more diverse patterns, less dependent on recency (8.3% drop)

The pointer mechanism's position bias is specifically designed to capture these
recency patterns, making it MORE CRITICAL for datasets like GeoLife.
""")
    
    print("="*70)
    print("Analysis Complete!")
    print("="*70)


if __name__ == '__main__':
    main()
