"""
DIY Dataset Preprocessing - Script 1: Raw to Interim
Processes raw DIY staypoint data to interim dataset with locations.

This script:
1. Reads preprocessed staypoints from raw CSV files
2. Filters to valid users based on quality criteria
3. Filters to activity staypoints only
4. Generates locations using DBSCAN clustering
5. Merges consecutive staypoints at same location
6. Enriches with temporal information
7. Saves interim dataset for further processing

Input: data/raw_diy/
Output: data/diy_eps{epsilon}/interim/
"""

import os
import sys
import json
import pickle
import argparse
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

import trackintel as ti

# Set random seed
RANDOM_SEED = 42


def _get_time(df):
    """Extract temporal features from timestamps."""
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)

    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["end_day"] = (df["finished_at"] - min_day).dt.days

    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["end_min"] = df["finished_at"].dt.hour * 60 + df["finished_at"].dt.minute
    df.loc[df["end_min"] == 0, "end_min"] = 24 * 60

    df["weekday"] = df["started_at"].dt.weekday
    return df


def enrich_time_info(sp):
    """Add temporal features to staypoints."""
    sp = sp.groupby("user_id", group_keys=False).apply(_get_time)
    sp.drop(columns={"finished_at", "started_at"}, inplace=True)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp = sp.reset_index(drop=True)

    # Convert user_id to integer if it's not already
    if sp["user_id"].dtype == 'object' or sp["user_id"].dtype == 'string':
        unique_users = sp["user_id"].unique()
        user_mapping = {user: idx for idx, user in enumerate(unique_users)}
        sp["user_id"] = sp["user_id"].map(user_mapping)
    else:
        sp["user_id"] = sp["user_id"].astype(int)
    
    sp["location_id"] = sp["location_id"].astype(int)

    # final cleaning, reassign ids
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def load_raw_data(config):
    """Load raw DIY staypoints and valid users."""
    print("\n" + "="*70)
    print("STAGE 1: Loading Raw Data")
    print("="*70)
    
    dataset_name = config['dataset']['name']
    raw_path = f"data/raw_{dataset_name}"
    
    print(f"\n[1/2] Reading preprocessed staypoints from {raw_path}...")
    # Read staypoints
    sp = ti.read_staypoints_csv(
        f'{raw_path}/3_staypoints_fun_generate_trips.csv',
        columns={'geometry': 'geom'},
        index_col='id'
    )
    print(f"  Loaded {len(sp):,} staypoints")
    
    # Read valid users
    print("\n[2/2] Reading valid users...")
    valid_user_df = pd.read_csv(f'{raw_path}/10_filter_after_user_quality_DIY_slide_filteres.csv')
    valid_user = valid_user_df["user_id"].values
    print(f"  Loaded {len(valid_user):,} valid users")

    # Filter to valid users
    sp = sp.loc[sp["user_id"].isin(valid_user)]
    print(f"  Valid users after quality filter: {len(valid_user):,}")

    # Filter to activity staypoints
    sp = sp.loc[sp["is_activity"] == True]
    print(f"  Activity staypoints: {len(sp):,}")
    
    if len(sp) == 0:
        print("\n❌ Error: No valid staypoints found after quality filtering.")
        sys.exit(1)
    
    return sp, valid_user


def generate_locations(sp, config, interim_dir, epsilon):
    """Generate locations from staypoints using DBSCAN clustering."""
    print("\n" + "="*70)
    print("STAGE 2: Generating Locations")
    print("="*70)
    
    loc_params = config['preprocessing']['location']
    
    print(f"\n[1/1] Clustering staypoints with epsilon={epsilon}m...")
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=epsilon,
        num_samples=loc_params['num_samples'],
        distance_metric=loc_params['distance_metric'],
        agg_level=loc_params['agg_level'],
        n_jobs=-1
    )
    
    # Filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print(f"  After filtering non-location staypoints: {len(sp):,}")

    # Save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    
    locations_file = os.path.join(interim_dir, f"locations_eps{epsilon}.csv")
    filtered_locs.as_locations.to_csv(locations_file)
    print(f"  Saved {len(filtered_locs):,} unique locations to: {locations_file}")
    
    return sp, filtered_locs


def merge_staypoints(sp, config, interim_dir, epsilon):
    """Merge consecutive staypoints at the same location."""
    print("\n" + "="*70)
    print("STAGE 3: Merging Staypoints")
    print("="*70)
    
    merge_params = config['preprocessing']['staypoint_merging']
    
    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    
    # Reset index to ensure it's named 'id'
    if sp.index.name != 'id':
        sp.index.name = 'id'
    
    print(f"\n[1/1] Merging consecutive staypoints (max gap: {merge_params['max_time_gap']})...")
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap=merge_params['max_time_gap'],
        agg={"location_id": "first"}
    )
    print(f"  After merging: {len(sp_merged):,} staypoints")
    
    # Save merged staypoints
    sp_merged_file = os.path.join(interim_dir, f"staypoints_merged_eps{epsilon}.csv")
    sp_merged.to_csv(sp_merged_file)
    print(f"  Saved merged staypoints to: {sp_merged_file}")
    
    # Recalculate staypoint duration
    sp_merged["duration"] = (
        sp_merged["finished_at"] - sp_merged["started_at"]
    ).dt.total_seconds() // 60
    
    return sp_merged


def process_temporal_features(sp, config, interim_dir, epsilon):
    """Add temporal features to staypoints."""
    print("\n" + "="*70)
    print("STAGE 4: Enriching Temporal Features")
    print("="*70)
    
    print("\n[1/1] Extracting temporal features (day, time, weekday)...")
    sp_time = enrich_time_info(sp)
    print(f"  Users with temporal features: {sp_time['user_id'].nunique():,}")
    
    # Save interim results - main output for Script 2
    interim_file = os.path.join(interim_dir, f"intermediate_eps{epsilon}.csv")
    sp_time.to_csv(interim_file, index=False)
    print(f"  Saved interim data to: {interim_file}")
    
    # Save interim statistics for EDA
    interim_stats = {
        "epsilon": epsilon,
        "total_staypoints": len(sp_time),
        "total_users": sp_time['user_id'].nunique(),
        "total_locations": sp_time['location_id'].nunique(),
        "staypoints_per_user_mean": len(sp_time) / sp_time['user_id'].nunique(),
        "duration_mean_min": float(sp_time['duration'].mean()),
        "duration_median_min": float(sp_time['duration'].median()),
        "duration_max_min": float(sp_time['duration'].max()),
        "days_tracked_mean": float(sp_time.groupby('user_id')['start_day'].max().mean()),
    }
    interim_stats_file = os.path.join(interim_dir, f"interim_stats_eps{epsilon}.json")
    with open(interim_stats_file, 'w') as f:
        json.dump(interim_stats, f, indent=2)
    print(f"  Saved interim statistics to: {interim_stats_file}")
    
    return sp_time


def main():
    parser = argparse.ArgumentParser(
        description="DIY Dataset Preprocessing - Script 1: Raw to Interim"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/diy.yaml",
        help="Path to dataset configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    np.random.seed(config.get('random_seed', RANDOM_SEED))
    
    dataset_name = config['dataset']['name']
    epsilon = config['dataset']['epsilon']
    
    # Create output directories
    output_folder = f"{dataset_name}_eps{epsilon}"
    interim_dir = os.path.join("data", output_folder, "interim")
    os.makedirs(interim_dir, exist_ok=True)
    
    raw_path = f"data/raw_{dataset_name}"
    
    print("=" * 80)
    print("DIY PREPROCESSING - Script 1: Raw to Interim")
    print("=" * 80)
    print(f"[INPUT]  Raw data: {raw_path}")
    print(f"[OUTPUT] Interim folder: {interim_dir}")
    print(f"[CONFIG] Dataset: {dataset_name}, Epsilon: {epsilon}")
    print(f"[CONFIG] Random seed: {config.get('random_seed', RANDOM_SEED)}")
    print("=" * 80)
    
    # Execute pipeline
    sp, valid_users = load_raw_data(config)
    
    # Save valid users info
    valid_users_file = os.path.join(interim_dir, f"valid_users_eps{epsilon}.csv")
    pd.DataFrame({"user_id": valid_users}).to_csv(valid_users_file, index=False)
    print(f"  Saved valid users to: {valid_users_file}")
    
    sp, locs = generate_locations(sp, config, interim_dir, epsilon)
    sp_merged = merge_staypoints(sp, config, interim_dir, epsilon)
    sp_time = process_temporal_features(sp_merged, config, interim_dir, epsilon)
    
    print("\n" + "=" * 80)
    print("SCRIPT 1 COMPLETE: Raw to Interim")
    print("=" * 80)
    print(f"Output folder: {interim_dir}")
    print(f"Main output: {os.path.join(interim_dir, f'intermediate_eps{epsilon}.csv')}")
    print("=" * 80)


if __name__ == "__main__":
    main()
