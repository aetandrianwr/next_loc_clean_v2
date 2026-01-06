#!/usr/bin/env python3
"""
H3 Dataset Statistics and Exploratory Data Analysis

This script generates comprehensive statistics and EDA for H3-preprocessed datasets.
Outputs statistics summary and optional visualizations.

Usage:
    python preprocessing/docs/h3_dataset_eda.py
"""

import os
import json
import pickle
import pandas as pd
import numpy as np
from collections import Counter

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")

def load_dataset_stats(dataset_name, h3_resolution=8):
    """Load all statistics for a dataset."""
    folder = f"{dataset_name}_h3r{h3_resolution}"
    base_path = os.path.join(DATA_DIR, folder)
    
    stats = {
        "dataset_name": dataset_name,
        "h3_resolution": h3_resolution,
    }
    
    # Load interim stats
    interim_stats_file = os.path.join(base_path, "interim", f"interim_stats_h3r{h3_resolution}.json")
    if os.path.exists(interim_stats_file):
        with open(interim_stats_file, 'r') as f:
            stats["interim"] = json.load(f)
    
    # Load processed metadata
    metadata_file = os.path.join(base_path, "processed", f"{folder}_prev7_metadata.json")
    if os.path.exists(metadata_file):
        with open(metadata_file, 'r') as f:
            stats["processed"] = json.load(f)
    
    # Load intermediate CSV for detailed analysis
    intermediate_file = os.path.join(base_path, "interim", f"intermediate_h3r{h3_resolution}.csv")
    if os.path.exists(intermediate_file):
        stats["intermediate_df"] = pd.read_csv(intermediate_file)
    
    # Load locations
    locations_file = os.path.join(base_path, "interim", f"locations_h3r{h3_resolution}.csv")
    if os.path.exists(locations_file):
        stats["locations_df"] = pd.read_csv(locations_file)
    
    # Load sequences for analysis
    for split in ["train", "validation", "test"]:
        pk_file = os.path.join(base_path, "processed", f"{folder}_prev7_{split}.pk")
        if os.path.exists(pk_file):
            with open(pk_file, 'rb') as f:
                stats[f"{split}_sequences"] = pickle.load(f)
    
    return stats


def analyze_sequences(sequences):
    """Analyze sequence statistics."""
    if not sequences:
        return {}
    
    seq_lengths = [len(s["X"]) for s in sequences]
    target_locs = [s["Y"] for s in sequences]
    
    return {
        "count": len(sequences),
        "seq_length_mean": np.mean(seq_lengths),
        "seq_length_std": np.std(seq_lengths),
        "seq_length_min": np.min(seq_lengths),
        "seq_length_max": np.max(seq_lengths),
        "seq_length_median": np.median(seq_lengths),
        "unique_targets": len(set(target_locs)),
    }


def analyze_staypoints(df):
    """Analyze staypoint statistics."""
    if df is None or df.empty:
        return {}
    
    stats = {
        "total_staypoints": len(df),
        "total_users": df["user_id"].nunique(),
        "total_locations": df["location_id"].nunique(),
    }
    
    # Per-user statistics
    user_stats = df.groupby("user_id").agg({
        "id": "count",
        "location_id": "nunique",
        "start_day": lambda x: x.max() - x.min()
    }).rename(columns={"id": "staypoints", "location_id": "unique_locs", "start_day": "days_tracked"})
    
    stats["staypoints_per_user"] = {
        "mean": user_stats["staypoints"].mean(),
        "std": user_stats["staypoints"].std(),
        "min": user_stats["staypoints"].min(),
        "max": user_stats["staypoints"].max(),
        "median": user_stats["staypoints"].median(),
    }
    
    stats["locations_per_user"] = {
        "mean": user_stats["unique_locs"].mean(),
        "std": user_stats["unique_locs"].std(),
        "min": user_stats["unique_locs"].min(),
        "max": user_stats["unique_locs"].max(),
        "median": user_stats["unique_locs"].median(),
    }
    
    stats["days_tracked_per_user"] = {
        "mean": user_stats["days_tracked"].mean(),
        "std": user_stats["days_tracked"].std(),
        "min": user_stats["days_tracked"].min(),
        "max": user_stats["days_tracked"].max(),
        "median": user_stats["days_tracked"].median(),
    }
    
    # Duration statistics
    if "duration" in df.columns:
        stats["duration_minutes"] = {
            "mean": df["duration"].mean(),
            "std": df["duration"].std(),
            "min": df["duration"].min(),
            "max": df["duration"].max(),
            "median": df["duration"].median(),
        }
    
    # Temporal distribution
    if "weekday" in df.columns:
        weekday_counts = df["weekday"].value_counts().sort_index()
        stats["weekday_distribution"] = weekday_counts.to_dict()
    
    if "start_min" in df.columns:
        # Hour distribution (start_min / 60)
        df["hour"] = (df["start_min"] / 60).astype(int)
        hour_counts = df["hour"].value_counts().sort_index()
        stats["hour_distribution"] = hour_counts.to_dict()
    
    # Location frequency distribution
    loc_counts = df["location_id"].value_counts()
    stats["location_frequency"] = {
        "mean": loc_counts.mean(),
        "std": loc_counts.std(),
        "min": loc_counts.min(),
        "max": loc_counts.max(),
        "median": loc_counts.median(),
        "top_10_percent_threshold": loc_counts.quantile(0.9),
    }
    
    return stats


def generate_report(dataset_stats, output_file=None):
    """Generate a formatted report."""
    report = []
    report.append("=" * 80)
    report.append(f"H3 DATASET STATISTICS AND EDA")
    report.append(f"Dataset: {dataset_stats['dataset_name']}")
    report.append(f"H3 Resolution: {dataset_stats['h3_resolution']}")
    report.append("=" * 80)
    
    # Interim statistics
    if "interim" in dataset_stats:
        interim = dataset_stats["interim"]
        report.append("\n" + "-" * 40)
        report.append("INTERIM DATA STATISTICS")
        report.append("-" * 40)
        report.append(f"Total Staypoints: {interim.get('total_staypoints', 'N/A'):,}")
        report.append(f"Total Users: {interim.get('total_users', 'N/A'):,}")
        report.append(f"Total Locations (H3 cells): {interim.get('total_locations', 'N/A'):,}")
        report.append(f"Mean Staypoints/User: {interim.get('staypoints_per_user_mean', 'N/A'):.1f}")
        report.append(f"Mean Duration (min): {interim.get('duration_mean_min', 'N/A'):.1f}")
        report.append(f"Median Duration (min): {interim.get('duration_median_min', 'N/A'):.1f}")
        report.append(f"Mean Days Tracked: {interim.get('days_tracked_mean', 'N/A'):.1f}")
    
    # Processed statistics
    if "processed" in dataset_stats:
        proc = dataset_stats["processed"]
        report.append("\n" + "-" * 40)
        report.append("PROCESSED DATA STATISTICS")
        report.append("-" * 40)
        report.append(f"Previous Day Window: {proc.get('previous_day', 'N/A')}")
        report.append(f"Final Users: {proc.get('unique_users', 'N/A'):,}")
        report.append(f"Final Locations: {proc.get('unique_locations', 'N/A'):,}")
        report.append(f"Total Staypoints: {proc.get('total_staypoints', 'N/A'):,}")
        report.append(f"  Train: {proc.get('train_staypoints', 'N/A'):,}")
        report.append(f"  Val: {proc.get('val_staypoints', 'N/A'):,}")
        report.append(f"  Test: {proc.get('test_staypoints', 'N/A'):,}")
        report.append(f"\nTotal Sequences: {proc.get('total_sequences', 'N/A'):,}")
        report.append(f"  Train: {proc.get('train_sequences', 'N/A'):,}")
        report.append(f"  Val: {proc.get('val_sequences', 'N/A'):,}")
        report.append(f"  Test: {proc.get('test_sequences', 'N/A'):,}")
    
    # Detailed staypoint analysis
    if "intermediate_df" in dataset_stats:
        sp_stats = analyze_staypoints(dataset_stats["intermediate_df"])
        report.append("\n" + "-" * 40)
        report.append("DETAILED STAYPOINT ANALYSIS")
        report.append("-" * 40)
        
        if "staypoints_per_user" in sp_stats:
            spu = sp_stats["staypoints_per_user"]
            report.append(f"\nStaypoints per User:")
            report.append(f"  Mean: {spu['mean']:.1f}, Std: {spu['std']:.1f}")
            report.append(f"  Min: {spu['min']}, Max: {spu['max']}, Median: {spu['median']:.1f}")
        
        if "locations_per_user" in sp_stats:
            lpu = sp_stats["locations_per_user"]
            report.append(f"\nUnique Locations per User:")
            report.append(f"  Mean: {lpu['mean']:.1f}, Std: {lpu['std']:.1f}")
            report.append(f"  Min: {lpu['min']}, Max: {lpu['max']}, Median: {lpu['median']:.1f}")
        
        if "duration_minutes" in sp_stats:
            dur = sp_stats["duration_minutes"]
            report.append(f"\nDuration (minutes):")
            report.append(f"  Mean: {dur['mean']:.1f}, Std: {dur['std']:.1f}")
            report.append(f"  Min: {dur['min']:.1f}, Max: {dur['max']:.1f}, Median: {dur['median']:.1f}")
        
        if "location_frequency" in sp_stats:
            lf = sp_stats["location_frequency"]
            report.append(f"\nLocation Visit Frequency:")
            report.append(f"  Mean visits/location: {lf['mean']:.1f}")
            report.append(f"  Std: {lf['std']:.1f}")
            report.append(f"  Most visited location: {lf['max']} visits")
            report.append(f"  Least visited location: {lf['min']} visits")
        
        if "weekday_distribution" in sp_stats:
            wd = sp_stats["weekday_distribution"]
            weekday_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            report.append(f"\nWeekday Distribution:")
            for day, count in sorted(wd.items()):
                pct = count / sum(wd.values()) * 100
                report.append(f"  {weekday_names[day]}: {count:,} ({pct:.1f}%)")
    
    # Sequence analysis
    for split in ["train", "validation", "test"]:
        key = f"{split}_sequences"
        if key in dataset_stats:
            seq_stats = analyze_sequences(dataset_stats[key])
            report.append(f"\n{split.upper()} Sequence Statistics:")
            report.append(f"  Count: {seq_stats['count']:,}")
            report.append(f"  Length Mean: {seq_stats['seq_length_mean']:.1f}")
            report.append(f"  Length Std: {seq_stats['seq_length_std']:.1f}")
            report.append(f"  Length Range: {seq_stats['seq_length_min']} - {seq_stats['seq_length_max']}")
            report.append(f"  Unique Targets: {seq_stats['unique_targets']}")
    
    report.append("\n" + "=" * 80)
    
    report_text = "\n".join(report)
    
    if output_file:
        with open(output_file, 'w') as f:
            f.write(report_text)
        print(f"Report saved to: {output_file}")
    
    return report_text


def main():
    """Generate EDA for all H3 datasets."""
    
    datasets = [
        ("diy", 8),
        ("geolife", 8),
    ]
    
    all_reports = []
    
    for dataset_name, h3_resolution in datasets:
        print(f"\nAnalyzing {dataset_name} (H3 resolution {h3_resolution})...")
        
        try:
            stats = load_dataset_stats(dataset_name, h3_resolution)
            report = generate_report(stats)
            all_reports.append(report)
            print(report)
        except Exception as e:
            print(f"Error analyzing {dataset_name}: {e}")
    
    # Save combined report
    docs_dir = os.path.dirname(os.path.abspath(__file__))
    output_file = os.path.join(docs_dir, "H3_DATASET_STATISTICS.md")
    
    with open(output_file, 'w') as f:
        f.write("# H3 Dataset Statistics and EDA\n\n")
        f.write("Generated statistics for H3-preprocessed mobility datasets.\n\n")
        f.write("```\n")
        f.write("\n\n".join(all_reports))
        f.write("\n```\n")
    
    print(f"\nCombined report saved to: {output_file}")


if __name__ == "__main__":
    main()
