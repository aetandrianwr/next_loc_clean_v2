"""
Geolife Dataset Preprocessing - H3 Version - Script 1: Raw to Interim
Processes raw Geolife trajectory data to intermediate staypoint dataset using H3-based locations.

This script uses Uber H3 hexagonal grid for location assignment instead of DBSCAN clustering.

This script:
1. Reads raw Geolife GPS trajectories
2. Generates staypoints from position fixes
3. Creates activity flags
4. Filters users based on quality metrics
5. Generates locations using H3 hexagonal grid (instead of DBSCAN)
6. Filters locations with minimum samples (like num_samples in DBSCAN)
7. Enriches with temporal information
8. Saves intermediate dataset for sequence generation

Input: data/raw_geolife/
Output: data/geolife_h3r{resolution}/interim/
"""

import os
import sys
import json
import pickle
import argparse
import datetime
from pathlib import Path

import yaml
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder

# Trackintel for mobility data processing
from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
import trackintel as ti
import h3

# Set random seed
RANDOM_SEED = 42


def calculate_user_quality(sp, trips, quality_file, quality_filter):
    """Calculate user quality based on temporal tracking coverage."""
    trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
    sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)

    print("Starting merge", sp.shape, trips.shape)
    sp["type"] = "sp"
    trips["type"] = "tpl"
    df_all = pd.concat([sp, trips])
    df_all = _split_overlaps(df_all, granularity="day")
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
    print("Finished merge", df_all.shape)
    print("*" * 50)

    print(f"Total users: {len(df_all['user_id'].unique())}")

    # Get quality
    total_quality = temporal_tracking_quality(df_all, granularity="all")
    # Get tracking days
    total_quality["days"] = (
        df_all.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
    )
    # Filter based on days
    user_filter_day = (
        total_quality.loc[(total_quality["days"] > quality_filter["day_filter"])]
        .reset_index(drop=True)["user_id"]
        .unique()
    )

    sliding_quality = (
        df_all.groupby("user_id")
        .apply(_get_tracking_quality, window_size=quality_filter["window_size"])
        .reset_index(drop=True)
    )

    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]
    filter_after_user_quality = filter_after_day.groupby("user_id", as_index=False)["quality"].mean()

    print(f"Final selected users: {filter_after_user_quality.shape[0]}")
    
    # Save quality file
    os.makedirs(os.path.dirname(quality_file), exist_ok=True)
    filter_after_user_quality.to_csv(quality_file, index=False)
    
    return filter_after_user_quality["user_id"].values


def _get_tracking_quality(df, window_size):
    """Calculate tracking quality using sliding window."""
    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        total_sec = (curr_end - curr_start).total_seconds()

        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret


def _get_time(df):
    """Calculate time features for a user."""
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

    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)

    # Reassign ids
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    return sp


def generate_h3_locations(sp, h3_resolution, num_samples):
    """
    Generate locations from staypoints using H3 hexagonal grid.
    
    This function replaces the DBSCAN-based generate_locations with H3-based location assignment.
    Each staypoint is assigned to an H3 cell based on its coordinates.
    Locations with fewer than num_samples staypoints are filtered out (similar to DBSCAN noise filtering).
    
    Args:
        sp: GeoDataFrame with staypoints
        h3_resolution: H3 resolution level (0-15)
        num_samples: Minimum samples required in H3 cell to be valid location
    
    Returns:
        sp: GeoDataFrame with location_id column added
        locs: DataFrame with location information
    """
    print(f"\n  Assigning staypoints to H3 cells (resolution={h3_resolution})...")
    
    # Extract coordinates from geometry
    sp = sp.copy()
    
    # Get lat/lon from geometry
    if hasattr(sp, 'geom'):
        sp['lat'] = sp['geom'].y
        sp['lon'] = sp['geom'].x
    elif hasattr(sp, 'geometry'):
        sp['lat'] = sp['geometry'].y
        sp['lon'] = sp['geometry'].x
    else:
        raise ValueError("No geometry column found in staypoints")
    
    # Assign H3 cell to each staypoint
    sp['h3_cell'] = sp.apply(lambda row: h3.latlng_to_cell(row['lat'], row['lon'], h3_resolution), axis=1)
    
    print(f"  Assigned {len(sp):,} staypoints to {sp['h3_cell'].nunique():,} unique H3 cells")
    
    # Filter cells with at least num_samples staypoints (similar to DBSCAN num_samples)
    print(f"  Filtering locations with min {num_samples} staypoints...")
    cell_counts = sp['h3_cell'].value_counts()
    valid_cells = cell_counts[cell_counts >= num_samples].index
    
    # Mark invalid cells as NaN (similar to DBSCAN noise points)
    sp.loc[~sp['h3_cell'].isin(valid_cells), 'h3_cell'] = None
    
    # Filter out noise staypoints
    sp = sp.loc[~sp["h3_cell"].isna()].copy()
    print(f"  After filtering non-location staypoints: {len(sp):,}")
    print(f"  Valid H3 cells (locations): {sp['h3_cell'].nunique():,}")
    
    # Create integer location_id from H3 cells
    print("  Creating location IDs...")
    unique_cells = sp['h3_cell'].unique()
    cell_to_id = {cell: idx for idx, cell in enumerate(unique_cells)}
    sp['location_id'] = sp['h3_cell'].map(cell_to_id)
    
    # Create locations DataFrame with H3 cell center coordinates
    print("  Creating locations dataframe...")
    locs_data = []
    for cell, loc_id in cell_to_id.items():
        lat, lng = h3.cell_to_latlng(cell)
        locs_data.append({
            'location_id': loc_id,
            'h3_cell': cell,
            'center_lat': lat,
            'center_lng': lng,
            'h3_resolution': h3_resolution
        })
    
    locs = pd.DataFrame(locs_data)
    locs = locs.set_index('location_id')
    
    # Clean up temporary columns
    sp = sp.drop(columns=['lat', 'lon', 'h3_cell'], errors='ignore')
    
    return sp, locs


def process_raw_to_intermediate(config):
    """Main processing function: raw trajectories to intermediate staypoint dataset using H3."""
    
    dataset_config = config["dataset"]
    preproc_config = config["preprocessing"]
    
    dataset_name = dataset_config["name"]
    h3_resolution = dataset_config["h3_resolution"]
    
    # Paths - use h3r{resolution} naming
    raw_path = os.path.join("data", f"raw_{dataset_name}")
    output_folder = f"{dataset_name}_h3r{h3_resolution}"
    interim_path = os.path.join("data", output_folder, "interim")
    
    os.makedirs(interim_path, exist_ok=True)
    
    print("=" * 80)
    print(f"GEOLIFE PREPROCESSING (H3) - Script 1: Raw to Interim")
    print("=" * 80)
    print(f"[INPUT]  Raw data: {raw_path}")
    print(f"[OUTPUT] Interim folder: {interim_path}")
    print(f"[CONFIG] Dataset: {dataset_name}, H3 Resolution: {h3_resolution}")
    print("=" * 80)
    
    # 1. Read raw Geolife data
    print("\n[1/7] Reading raw Geolife trajectories...")
    pfs, _ = read_geolife(raw_path, print_progress=True)
    print(f"Loaded {len(pfs)} position fixes from {len(pfs['user_id'].unique())} users")
    
    # Save raw statistics
    raw_stats = {
        "total_position_fixes": len(pfs),
        "total_users": len(pfs['user_id'].unique()),
    }
    stats_file = os.path.join(interim_path, f"raw_stats_h3r{h3_resolution}.json")
    with open(stats_file, 'w') as f:
        json.dump(raw_stats, f, indent=2)
    print(f"Saved raw statistics to: {stats_file}")
    
    # 2. Generate staypoints
    print("\n[2/7] Generating staypoints...")
    sp_config = preproc_config["staypoint"]
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        gap_threshold=sp_config["gap_threshold"],
        include_last=True,
        print_progress=True,
        dist_threshold=sp_config["dist_threshold"],
        time_threshold=sp_config["time_threshold"],
        n_jobs=-1
    )
    print(f"Generated {len(sp)} staypoints")
    
    # Save staypoints before filtering
    sp_before_file = os.path.join(interim_path, f"staypoints_all_h3r{h3_resolution}.csv")
    sp.to_csv(sp_before_file)
    print(f"Saved all staypoints to: {sp_before_file}")
    
    # 3. Create activity flag
    print("\n[3/7] Creating activity flags...")
    sp = sp.as_staypoints.create_activity_flag(
        method="time_threshold",
        time_threshold=sp_config["activity_time_threshold"]
    )
    
    # 4. Filter valid users based on quality
    print("\n[4/7] Filtering valid users based on quality...")
    quality_path = os.path.join(interim_path, "quality")
    quality_file = os.path.join(quality_path, f"user_quality_h3r{h3_resolution}.csv")
    
    if Path(quality_file).is_file():
        print(f"Loading existing quality file: {quality_file}")
        valid_user = pd.read_csv(quality_file)["user_id"].values
    else:
        print("Calculating user quality (this may take a while)...")
        # Generate triplegs for quality calculation
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        # Generate trips
        sp_temp, tpls_temp, trips = generate_trips(sp.copy(), tpls, add_geometry=False)
        
        quality_filter = preproc_config["quality_filter"]
        valid_user = calculate_user_quality(sp_temp.copy(), trips.copy(), quality_file, quality_filter)
    
    print(f"Valid users after quality filter: {len(valid_user)}")
    sp = sp.loc[sp["user_id"].isin(valid_user)]
    
    # Save user quality info
    valid_users_file = os.path.join(interim_path, f"valid_users_h3r{h3_resolution}.csv")
    pd.DataFrame({"user_id": valid_user}).to_csv(valid_users_file, index=False)
    print(f"Saved valid users to: {valid_users_file}")
    
    # 5. Filter activity staypoints
    print("\n[5/7] Filtering activity staypoints...")
    sp = sp.loc[sp["is_activity"] == True]
    print(f"Activity staypoints: {len(sp)}")
    
    # 6. Generate locations using H3 (instead of DBSCAN)
    print("\n[6/7] Generating locations using H3 hexagonal grid...")
    loc_config = preproc_config["location"]
    sp, locs = generate_h3_locations(
        sp, 
        h3_resolution=h3_resolution, 
        num_samples=loc_config["num_samples"]
    )
    
    # Save locations
    locations_file = os.path.join(interim_path, f"locations_h3r{h3_resolution}.csv")
    locs.to_csv(locations_file)
    print(f"Unique locations: {sp['location_id'].nunique()}")
    print(f"Saved locations to: {locations_file}")
    
    # 7. Merge consecutive staypoints and enrich temporal info
    print("\n[7/7] Merging staypoints and enriching temporal information...")
    sp = sp[["user_id", "started_at", "finished_at", "geom", "location_id"]]
    
    # Merge staypoints
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap="1min",
        agg={"location_id": "first"}
    )
    print(f"Staypoints after merging: {len(sp_merged)}")
    
    # Save merged staypoints before time enrichment
    sp_merged_file = os.path.join(interim_path, f"staypoints_merged_h3r{h3_resolution}.csv")
    sp_merged.to_csv(sp_merged_file)
    print(f"Saved merged staypoints to: {sp_merged_file}")
    
    # Recalculate duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60
    
    # Add time features
    sp_time = enrich_time_info(sp_merged)
    
    print(f"Final users in intermediate data: {sp_time['user_id'].nunique()}")
    print(f"Final staypoints: {len(sp_time)}")
    
    # Save intermediate results - this is the main output for Script 2
    interim_file = os.path.join(interim_path, f"intermediate_h3r{h3_resolution}.csv")
    sp_time.to_csv(interim_file, index=False)
    print(f"\n✓ Saved intermediate dataset to: {interim_file}")
    
    # Save interim statistics for EDA
    interim_stats = {
        "h3_resolution": h3_resolution,
        "total_staypoints": len(sp_time),
        "total_users": sp_time['user_id'].nunique(),
        "total_locations": sp_time['location_id'].nunique(),
        "staypoints_per_user_mean": len(sp_time) / sp_time['user_id'].nunique(),
        "duration_mean_min": sp_time['duration'].mean(),
        "duration_median_min": sp_time['duration'].median(),
        "duration_max_min": sp_time['duration'].max(),
        "days_tracked_mean": sp_time.groupby('user_id')['start_day'].max().mean(),
    }
    interim_stats_file = os.path.join(interim_path, f"interim_stats_h3r{h3_resolution}.json")
    with open(interim_stats_file, 'w') as f:
        json.dump(interim_stats, f, indent=2)
    print(f"Saved interim statistics to: {interim_stats_file}")
    
    print("\n" + "=" * 80)
    print("SCRIPT 1 COMPLETE: Raw to Interim (H3)")
    print("=" * 80)
    print(f"Output folder: {interim_path}")
    print(f"Main output: {interim_file}")
    print("=" * 80)
    
    return sp_time


def main():
    parser = argparse.ArgumentParser(
        description="Geolife preprocessing (H3) - Script 1: Raw to Intermediate"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocessing/geolife_h3.yaml",
        help="Path to dataset configuration file"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    np.random.seed(config.get("random_seed", RANDOM_SEED))
    
    # Process
    process_raw_to_intermediate(config)


if __name__ == "__main__":
    main()
