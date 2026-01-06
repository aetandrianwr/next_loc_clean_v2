"""
DIY Dataset Preprocessing (H3 Version) - Script 2: Interim to Processed
Processes intermediate staypoint data to final sequence .pk files.

This script:
1. Loads intermediate staypoint dataset from Script 1 (H3 version)
2. Splits data into train/val/test per user
3. Encodes location IDs
4. For each previous_day value in config:
   - Filters valid sequences based on previous_day parameter
   - Generates sequences with features (X, user_X, weekday_X, etc.)
   - Saves train/validation/test .pk files
   - Saves metadata.pk

Input: data/diy_h3r{resolution}/interim/
Output: data/diy_h3r{resolution}/processed/
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
from tqdm import tqdm
from sklearn.preprocessing import OrdinalEncoder
from joblib import Parallel, delayed, parallel_backend

# Set random seed
RANDOM_SEED = 42


def split_dataset(totalData, split_ratios):
    """Split dataset into train, val and test per user."""
    totalData = totalData.groupby("user_id", group_keys=False).apply(
        _get_split_days_user, split_ratios=split_ratios
    )

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # Final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def _get_split_days_user(df, split_ratios):
    """Split the dataset according to the tracked day of each user."""
    maxDay = df["start_day"].max()
    train_split = maxDay * split_ratios["train"]
    validation_split = maxDay * (split_ratios["train"] + split_ratios["val"])

    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & (df["start_day"] < validation_split), "Dataset"] = "vali"

    return df


def get_valid_sequence(input_df, previous_day=7, min_length=3):
    """Get valid sequence IDs based on previous_day requirement."""
    valid_id = []
    for user in input_df["user_id"].unique():
        df = input_df.loc[input_df["user_id"] == user].copy().reset_index(drop=True)

        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # Exclude the first records
            if row["diff_day"] < previous_day:
                continue

            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
            if len(hist) < min_length:
                continue

            valid_id.append(row["id"])

    return valid_id


def _get_valid_sequence_user(args):
    """Get valid sequences per user - for parallel processing."""
    df, previous_day, valid_ids = args
    df = df.reset_index(drop=True)
    data_single_user = []
    
    # Get the day of tracking
    min_days = df["start_day"].min()
    df["diff_day"] = df["start_day"] - min_days
    
    for index, row in df.iterrows():
        # Exclude the first records that do not include enough previous_day
        if row["diff_day"] < previous_day:
            continue
        
        # Get the history records [curr-previous, curr]
        hist = df.iloc[:index]
        hist = hist.loc[(hist["start_day"] >= (row["start_day"] - previous_day))]
        
        # Should be in the valid user ids
        if not (row["id"] in valid_ids):
            continue
        
        data_dict = {}
        # Get features: location, user, weekday, start time, duration, diff to curr day
        data_dict["X"] = hist["location_id"].values
        data_dict["user_X"] = hist["user_id"].values
        data_dict["weekday_X"] = hist["weekday"].values
        data_dict["start_min_X"] = hist["start_min"].values
        data_dict["dur_X"] = hist["duration"].values
        data_dict["diff"] = (row["diff_day"] - hist["diff_day"]).astype(int).values
        
        # The next location is the target
        data_dict["Y"] = int(row["location_id"])
        
        # Append the single sample to list
        data_single_user.append(data_dict)
    
    return data_single_user


def generate_sequences(data, valid_ids, previous_day, split_name):
    """Generate sequences from data using parallel processing."""
    print(f"  Processing {split_name} sequences...")
    
    valid_ids_set = set(valid_ids)
    
    # Prepare arguments for parallel processing
    user_groups = [(group.copy(), previous_day, valid_ids_set) for _, group in data.groupby("user_id")]
    
    # Use parallel processing
    with parallel_backend("threading", n_jobs=-1):
        valid_user_ls = Parallel()(
            delayed(_get_valid_sequence_user)(args) 
            for args in tqdm(user_groups, desc=f"    {split_name}")
        )
    
    # Flatten the list of lists
    valid_records = [item for sublist in valid_user_ls for item in sublist]
    
    return valid_records


def process_for_previous_day(sp, split_ratios, max_duration, previous_day, h3_resolution, dataset_name, output_base_path, min_length=3):
    """Process data for a specific previous_day value."""
    
    output_name = f"{dataset_name}_h3r{h3_resolution}_prev{previous_day}"
    processed_path = os.path.join(output_base_path, "processed")
    os.makedirs(processed_path, exist_ok=True)
    
    print("\n" + "-" * 60)
    print(f"Processing for previous_day = {previous_day}")
    print("-" * 60)
    
    # Truncate too long duration
    sp_copy = sp.copy()
    sp_copy.loc[sp_copy["duration"] > max_duration - 1, "duration"] = max_duration - 1
    
    # 1. Split dataset
    print("\n[1/5] Splitting dataset into train/val/test...")
    train_data, vali_data, test_data = split_dataset(sp_copy, split_ratios)
    print(f"Train: {len(train_data)}, Val: {len(vali_data)}, Test: {len(test_data)}")
    
    # 2. Encode locations
    print("\n[2/5] Encoding location IDs...")
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    # Add 2 to account for unseen locations (1) and padding (0)
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2
    
    print(f"Max location ID: {train_data['location_id'].max()}")
    print(f"Unique locations in train: {train_data['location_id'].nunique()}")
    
    # 3. Get valid sequences
    print(f"\n[3/5] Filtering valid sequences (previous_day={previous_day})...")
    
    all_ids = sp_copy[["id"]].copy()
    
    valid_ids = get_valid_sequence(train_data, previous_day=previous_day, min_length=min_length)
    valid_ids.extend(get_valid_sequence(vali_data, previous_day=previous_day, min_length=min_length))
    valid_ids.extend(get_valid_sequence(test_data, previous_day=previous_day, min_length=min_length))
    
    all_ids[f"{previous_day}"] = 0
    all_ids.loc[all_ids["id"].isin(valid_ids), f"{previous_day}"] = 1
    
    # Get final valid staypoint IDs
    all_ids.set_index("id", inplace=True)
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values
    
    print(f"Valid staypoints: {len(final_valid_id)}")
    
    # 4. Filter users based on final_valid_id
    print("\n[4/5] Filtering users with valid sequences in all splits...")
    valid_users_train = train_data.loc[train_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_vali = vali_data.loc[vali_data["id"].isin(final_valid_id), "user_id"].unique()
    valid_users_test = test_data.loc[test_data["id"].isin(final_valid_id), "user_id"].unique()
    
    valid_users = set.intersection(set(valid_users_train), set(valid_users_vali), set(valid_users_test))
    print(f"Valid users (in all splits): {len(valid_users)}")
    
    filtered_sp = sp_copy.loc[sp_copy["user_id"].isin(valid_users)].copy()
    
    # Re-split with filtered users
    train_data, vali_data, test_data = split_dataset(filtered_sp, split_ratios)
    
    # Re-encode locations
    enc = OrdinalEncoder(dtype=np.int64, handle_unknown="use_encoded_value", unknown_value=-1).fit(
        train_data["location_id"].values.reshape(-1, 1)
    )
    train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
    vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
    test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2
    
    # Re-encode user IDs to be continuous
    user_enc = OrdinalEncoder(dtype=np.int64)
    filtered_sp["user_id"] = user_enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1
    
    train_data["user_id"] = user_enc.transform(train_data["user_id"].values.reshape(-1, 1)) + 1
    vali_data["user_id"] = user_enc.transform(vali_data["user_id"].values.reshape(-1, 1)) + 1
    test_data["user_id"] = user_enc.transform(test_data["user_id"].values.reshape(-1, 1)) + 1
    
    print(f"Final max location ID: {train_data['location_id'].max()}")
    print(f"Final unique locations: {train_data['location_id'].nunique()}")
    print(f"Final user count: {filtered_sp['user_id'].nunique()}")
    
    # 5. Generate sequences and save .pk files
    print("\n[5/5] Generating sequences and saving .pk files...")
    
    train_sequences = generate_sequences(train_data, final_valid_id, previous_day, "train")
    print(f"  Generated {len(train_sequences)} train sequences")
    
    val_sequences = generate_sequences(vali_data, final_valid_id, previous_day, "validation")
    print(f"  Generated {len(val_sequences)} validation sequences")
    
    test_sequences = generate_sequences(test_data, final_valid_id, previous_day, "test")
    print(f"  Generated {len(test_sequences)} test sequences")
    
    # Save pickle files
    train_pk_file = os.path.join(processed_path, f"{output_name}_train.pk")
    val_pk_file = os.path.join(processed_path, f"{output_name}_validation.pk")
    test_pk_file = os.path.join(processed_path, f"{output_name}_test.pk")
    
    with open(train_pk_file, "wb") as f:
        pickle.dump(train_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved train sequences to: {train_pk_file}")
    
    with open(val_pk_file, "wb") as f:
        pickle.dump(val_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved validation sequences to: {val_pk_file}")
    
    with open(test_pk_file, "wb") as f:
        pickle.dump(test_sequences, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"✓ Saved test sequences to: {test_pk_file}")
    
    # Generate and save metadata
    metadata = {
        "dataset_name": dataset_name,
        "output_dataset_name": output_name,
        "h3_resolution": h3_resolution,
        "previous_day": previous_day,
        "total_user_num": int(train_data["user_id"].max() + 1),
        "total_loc_num": int(train_data["location_id"].max() + 1),
        "unique_users": int(train_data["user_id"].nunique()),
        "unique_locations": int(train_data["location_id"].nunique()),
        "total_staypoints": int(len(filtered_sp)),
        "valid_staypoints": int(len(final_valid_id)),
        "train_staypoints": int(len(train_data)),
        "val_staypoints": int(len(vali_data)),
        "test_staypoints": int(len(test_data)),
        "train_sequences": len(train_sequences),
        "val_sequences": len(val_sequences),
        "test_sequences": len(test_sequences),
        "total_sequences": len(train_sequences) + len(val_sequences) + len(test_sequences),
        "split_ratios": split_ratios,
        "max_duration_minutes": max_duration,
    }
    
    metadata_file = os.path.join(processed_path, f"{output_name}_metadata.json")
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved metadata to: {metadata_file}")
    
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="DIY Dataset Preprocessing (H3 Version) - Script 2: Interim to Processed"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/preprocessing/diy_h3.yaml",
        help="Path to dataset configuration file"
    )
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    np.random.seed(config.get('random_seed', RANDOM_SEED))
    
    dataset_name = config['dataset']['name']
    h3_resolution = config['dataset']['h3_resolution']
    previous_day_list = config['dataset']['previous_day']  # Now a list
    
    # Ensure previous_day is a list
    if not isinstance(previous_day_list, list):
        previous_day_list = [previous_day_list]
    
    # Paths
    output_folder = f"{dataset_name}_h3r{h3_resolution}"
    interim_path = os.path.join("data", output_folder, "interim")
    output_base_path = os.path.join("data", output_folder)
    
    split_ratios = config['preprocessing']['split']
    max_duration = config['preprocessing'].get('max_duration', 2880)
    min_length = config['preprocessing'].get('min_sequence_length', 3)
    
    print("=" * 80)
    print("DIY PREPROCESSING (H3) - Script 2: Interim to Processed")
    print("=" * 80)
    print(f"[INPUT]  Interim folder: {interim_path}")
    print(f"[OUTPUT] Processed folder: {output_base_path}/processed/")
    print(f"[CONFIG] Dataset: {dataset_name}, H3 Resolution: {h3_resolution}")
    print(f"[CONFIG] Previous days: {previous_day_list}")
    print("=" * 80)
    
    # Load intermediate data
    print("\n[LOAD] Loading intermediate dataset...")
    interim_file = os.path.join(interim_path, f"intermediate_h3r{h3_resolution}.csv")
    sp = pd.read_csv(interim_file)
    print(f"Loaded {len(sp)} staypoints from {sp['user_id'].nunique()} users")
    print(f"Input file: {interim_file}")
    
    # Process for each previous_day value
    all_metadata = {}
    for previous_day in previous_day_list:
        metadata = process_for_previous_day(
            sp, split_ratios, max_duration, previous_day, 
            h3_resolution, dataset_name, output_base_path, min_length
        )
        all_metadata[previous_day] = metadata
    
    print("\n" + "=" * 80)
    print("SCRIPT 2 COMPLETE: Interim to Processed")
    print("=" * 80)
    print(f"Output folder: {output_base_path}/processed/")
    for prev_day, meta in all_metadata.items():
        print(f"\nprevious_day={prev_day}:")
        print(f"  Train: {meta['train_sequences']}, Val: {meta['val_sequences']}, Test: {meta['test_sequences']}")
        print(f"  Total users: {meta['total_user_num']}, Total locations: {meta['total_loc_num']}")
    print("=" * 80)


if __name__ == "__main__":
    main()
