"""
1st-Order Markov Chain Model - Faithful reproduction of original markov.py

This is an exact reproduction of location-prediction-ori-freeze/baselines/markov.py
adapted to work within the next_loc_clean_v2 project structure.

The original script:
1. Reads CSV data from interim folder (intermediate_eps{X}.csv)
2. Splits data by user's tracked days (60% train, 20% val, 20% test)
3. Filters by valid_ids (generated from preprocessed data)
4. Trains per-user Markov transition matrices
5. Evaluates on test set using original metric functions

Usage (from next_loc_clean_v2 root):
    python src/models/baseline/markov_ori/run_markov_ori.py --config config/models/config_markov_ori_geolife.yaml
    python src/models/baseline/markov_ori/run_markov_ori.py --config config/models/config_markov_ori_diy.yaml

Output:
    Results are saved to experiments/{dataset}_markov_ori_{timestamp}/
"""

import pickle
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import sys
import argparse
import json
import yaml
import time
import shutil
from datetime import datetime, timezone, timedelta
from pathlib import Path

from sklearn.metrics import f1_score, recall_score
from timeit import default_timer as timer

# Set random seed as in original (seed=0)
np.random.seed(0)


def load_config(path):
    """Load configuration from YAML file."""
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        if isinstance(value, dict):
            for k, v in value.items():
                config[k] = v
        else:
            config[key] = value

    return config


class EasyDict(dict):
    """Dictionary with attribute-style access."""
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name)

    def __setattr__(self, name, value):
        self[name] = value


def splitDataset(totalData):
    """Split dataset into train, vali and test.
    
    Exactly as in original markov.py
    """
    totalData = totalData.groupby("user_id").apply(getSplitDaysUser)

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def getSplitDaysUser(df):
    """Split the dataset according to the tracked day of each user.
    
    Exactly as in original markov.py
    """
    maxDay = df["start_day"].max()
    train_split = maxDay * 0.6
    vali_split = maxDay * 0.8

    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[
        (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
        "Dataset",
    ] = "vali"

    return df


def markov_transition_prob(df, n=1):
    """Build transition probability table.
    
    Exactly as in original markov.py
    """
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    for i in range(n):
        locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i]["location_id"].values
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()


def get_true_pred_pair(locSequence, df, n=1):
    """Get true and predicted pairs from test data.
    
    Exactly as in original markov.py
    """
    testSeries = df["location_id"].values

    true_ls = []
    pred_ls = []

    time_ls = []
    for i in range(testSeries.shape[0] - n):
        locCurr = testSeries[i : i + n + 1]
        numbLoc = n

        start = timer()
        # loop until finds a match
        while True:
            res_df = locSequence
            for j in range(n - numbLoc, n):
                res_df = res_df.loc[res_df[f"loc_{j+1}"] == locCurr[j]]
            res_df = res_df.sort_values(by="size", ascending=False)

            if res_df.shape[0]:  # if the dataframe contains entry, stop finding
                # choose the location which are visited most often for the matches
                pred = res_df["toLoc"].drop_duplicates().values
                break
            # decrese the number of location history considered
            numbLoc -= 1
            if numbLoc == 0:
                pred = np.zeros(10)
                break

        time_ls.append(timer() - start)
        true_ls.append(locCurr[-1])
        pred_ls.append(pred)

    return true_ls, pred_ls, time_ls


def get_performance_measure(true_ls, pred_ls):
    """Calculate performance metrics.
    
    Exactly as in original markov.py
    """
    acc_ls = [1, 5, 10]

    res = []
    ndcg_ls = []
    # total number
    res.append(len(true_ls))
    for top_acc in acc_ls:
        correct = 0
        for true, pred in zip(true_ls, pred_ls):
            if true in pred[:top_acc]:
                correct += 1

            # ndcg calculation
            if top_acc == 10:
                idx = np.where(true == pred[:top_acc])[0]
                if len(idx) == 0:
                    ndcg_ls.append(0)
                else:
                    ndcg_ls.append(1 / np.log2(idx[0] + 1 + 1))

        res.append(correct)

    top1 = [pred[0] for pred in pred_ls]
    f1 = f1_score(true_ls, top1, average="weighted")
    recall = recall_score(true_ls, top1, average="weighted")

    res.append(f1)
    res.append(recall)
    res.append(np.mean(ndcg_ls))

    # rr
    rank_ls = []
    for true, pred in zip(true_ls, pred_ls):
        rank = np.where(pred == true)[0] + 1
        if len(rank):
            rank_ls.append(rank[0])
        else:
            rank_ls.append(0)
    rank = np.array(rank_ls, dtype=float)

    rank = np.divide(1.0, rank, out=np.zeros_like(rank), where=rank != 0)
    res.append(rank.sum())

    return pd.Series(res, index=["total", "correct@1", "correct@5", "correct@10", "f1", "recall", "ndcg", "rr"])


def get_markov_res(train, test, n=2):
    """Get Markov results for a user.
    
    Exactly as in original markov.py
    """
    locSeq_df = markov_transition_prob(train, n=n)
    return get_true_pred_pair(locSeq_df, test, n=n)


def generate_valid_ids(preprocessed_dir, dataset_prefix, csv_data):
    """
    Generate valid_ids by matching preprocessed data users.
    
    Args:
        preprocessed_dir: Directory containing preprocessed .pk files
        dataset_prefix: Prefix for the preprocessed files
        csv_data: DataFrame with the CSV data
    
    Returns:
        numpy.ndarray: Array of valid record IDs
    """
    # Load all preprocessed data
    train_path = os.path.join(preprocessed_dir, f"{dataset_prefix}_train.pk")
    val_path = os.path.join(preprocessed_dir, f"{dataset_prefix}_validation.pk")
    test_path = os.path.join(preprocessed_dir, f"{dataset_prefix}_test.pk")
    
    train_data = pickle.load(open(train_path, "rb"))
    val_data = pickle.load(open(val_path, "rb"))
    test_data = pickle.load(open(test_path, "rb"))
    
    # Get unique users from preprocessed data
    users_train = set(sample['user_X'][0] for sample in train_data)
    users_val = set(sample['user_X'][0] for sample in val_data)
    users_test = set(sample['user_X'][0] for sample in test_data)
    all_users = users_train | users_val | users_test
    
    # Filter CSV records to only include these users
    valid_records = csv_data[csv_data['user_id'].isin(all_users)]
    valid_ids = valid_records['id'].values
    
    return valid_ids


def init_experiment_dir(config, dataset_name, model_name="markov_ori"):
    """Create experiment directory with dataset name, model name, and timestamp."""
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_{model_name}_{timestamp}"
    experiment_dir = os.path.join(config.experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir


def run_evaluation(config, train_data, eval_data, eval_name, n=1):
    """Run evaluation on a dataset split."""
    true_all_ls = []
    pred_all_ls = []
    time_all_ls = []
    total_parameter = 0
    
    users = train_data["user_id"].unique()
    
    for user in tqdm(users, desc=f"Evaluating on {eval_name}"):
        curr_train = train_data.loc[train_data["user_id"] == user]
        curr_eval = eval_data.loc[eval_data["user_id"] == user]
        
        total_parameter += curr_train["location_id"].unique().shape[0] ** 2
        true_ls, pred_ls, time_ls = get_markov_res(curr_train, curr_eval, n=n)
        time_all_ls.extend(time_ls)
        true_all_ls.extend(true_ls)
        pred_all_ls.extend(pred_ls)
    
    if not true_all_ls:
        return None, total_parameter
    
    result = get_performance_measure(true_all_ls, pred_all_ls)
    
    acc1 = result["correct@1"] / result["total"] * 100
    acc5 = result["correct@5"] / result["total"] * 100
    acc10 = result["correct@10"] / result["total"] * 100
    mrr = result["rr"] / result["total"] * 100
    f1 = result["f1"] * 100
    recall = result["recall"] * 100
    ndcg = result["ndcg"] * 100
    
    results = {
        "total_samples": int(result["total"]),
        "correct@1": int(result["correct@1"]),
        "correct@5": int(result["correct@5"]),
        "correct@10": int(result["correct@10"]),
        "acc@1": acc1,
        "acc@5": acc5,
        "acc@10": acc10,
        "mrr": mrr,
        "f1": f1,
        "recall": recall,
        "ndcg": ndcg,
    }
    
    return results, total_parameter


def main():
    parser = argparse.ArgumentParser(
        description="1st-Order Markov Chain Model - Faithful to original markov.py"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    config = EasyDict(config)
    
    n = config.get("markov_order", 1)
    
    print("=" * 60)
    print("1st-Order Markov Chain Model - Original Implementation")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Dataset: {config.dataset}")
    print(f"Markov order: {n}")
    print("=" * 60)
    
    # Get dataset name
    dataset_name = config.dataset
    
    # Try to load metadata if available
    processed_dir = config.get("processed_dir", None)
    dataset_prefix = config.get("dataset_prefix", None)
    if processed_dir and dataset_prefix:
        metadata_path = os.path.join(processed_dir, f"{dataset_prefix}_metadata.json")
        if os.path.exists(metadata_path):
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            dataset_name = metadata.get("dataset_name", config.dataset)
    
    # Initialize experiment directory
    experiment_dir = init_experiment_dir(config, dataset_name, model_name="markov_ori")
    print(f"Experiment directory: {experiment_dir}")
    
    # Create log file
    log_path = os.path.join(experiment_dir, "training.log")
    log_file = open(log_path, "w")
    log_file.write("=" * 60 + "\n")
    log_file.write("1st-Order Markov Chain Model - Original Implementation\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.write(f"Dataset: {dataset_name}\n")
    log_file.write(f"Config: {args.config}\n")
    log_file.write(f"Markov order: {n}\n")
    log_file.write(f"Experiment directory: {experiment_dir}\n")
    log_file.write("=" * 60 + "\n")
    
    # Load CSV data
    csv_path = config.data_csv
    print(f"\nLoading data from: {csv_path}")
    inputData = pd.read_csv(csv_path)
    inputData.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    print(f"Total records: {len(inputData)}")
    log_file.write(f"\nLoaded {len(inputData)} records from {csv_path}\n")
    
    # Split data - exactly as in original
    train_data, vali_data, test_data = splitDataset(inputData)
    print(f"After split: train={train_data.shape}, vali={vali_data.shape}, test={test_data.shape}")
    log_file.write(f"After split: train={train_data.shape}, vali={vali_data.shape}, test={test_data.shape}\n")
    
    # Load or generate valid_ids
    valid_ids_file = config.get("valid_ids_file", None)
    if valid_ids_file and os.path.exists(valid_ids_file):
        print(f"\nLoading valid_ids from: {valid_ids_file}")
        valid_ids = pickle.load(open(valid_ids_file, "rb"))
        log_file.write(f"Loaded valid_ids from: {valid_ids_file}\n")
    else:
        print("\nGenerating valid_ids from preprocessed data...")
        valid_ids = generate_valid_ids(config.processed_dir, config.dataset_prefix, inputData)
        log_file.write(f"Generated valid_ids from preprocessed data\n")
    print(f"Valid IDs: {len(valid_ids)}")
    log_file.write(f"Valid IDs: {len(valid_ids)}\n")
    
    # Filter data
    train_data = train_data.loc[train_data["id"].isin(valid_ids)]
    vali_data = vali_data.loc[vali_data["id"].isin(valid_ids)]
    test_data = test_data.loc[test_data["id"].isin(valid_ids)]
    print(f"After filtering: train={train_data.shape}, vali={vali_data.shape}, test={test_data.shape}")
    log_file.write(f"After filtering: train={train_data.shape}, vali={vali_data.shape}, test={test_data.shape}\n")
    
    # Training and evaluation
    print("\n" + "=" * 60)
    print("Training and Evaluation")
    print("=" * 60)
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write("Training and Evaluation\n")
    log_file.write("=" * 60 + "\n")
    
    training_start_time = time.time()
    
    # Evaluate on validation set
    print(f"\nNumber of users: {train_data['user_id'].nunique()}")
    val_results, total_params = run_evaluation(config, train_data, vali_data, "validation", n=n)
    
    # Evaluate on test set
    test_results, _ = run_evaluation(config, train_data, test_data, "test", n=n)
    
    training_time = time.time() - training_start_time
    print(f"\nTraining finished. Time: {training_time:.2f}s")
    print(f"Total parameters: {total_params}")
    log_file.write(f"\nTraining finished. Time: {training_time:.2f}s\n")
    log_file.write(f"Total parameters: {total_params}\n")
    
    # Print results
    print("\n" + "=" * 60)
    print("Validation Results")
    print("=" * 60)
    if val_results:
        print(f"Acc@1:  {val_results['acc@1']:.2f}%")
        print(f"Acc@5:  {val_results['acc@5']:.2f}%")
        print(f"Acc@10: {val_results['acc@10']:.2f}%")
        print(f"MRR:    {val_results['mrr']:.2f}%")
        print(f"F1:     {val_results['f1']:.2f}%")
        print(f"NDCG:   {val_results['ndcg']:.2f}%")
        print(f"Total:  {val_results['total_samples']}")
        
        log_file.write("\nValidation Results:\n")
        log_file.write(f"  Acc@1:  {val_results['acc@1']:.2f}%\n")
        log_file.write(f"  Acc@5:  {val_results['acc@5']:.2f}%\n")
        log_file.write(f"  Acc@10: {val_results['acc@10']:.2f}%\n")
        log_file.write(f"  MRR:    {val_results['mrr']:.2f}%\n")
        log_file.write(f"  F1:     {val_results['f1']:.2f}%\n")
        log_file.write(f"  NDCG:   {val_results['ndcg']:.2f}%\n")
    
    print("\n" + "=" * 60)
    print("Test Results")
    print("=" * 60)
    if test_results:
        print(f"Acc@1:  {test_results['acc@1']:.2f}%")
        print(f"Acc@5:  {test_results['acc@5']:.2f}%")
        print(f"Acc@10: {test_results['acc@10']:.2f}%")
        print(f"MRR:    {test_results['mrr']:.2f}%")
        print(f"F1:     {test_results['f1']:.2f}%")
        print(f"NDCG:   {test_results['ndcg']:.2f}%")
        print(f"Total:  {test_results['total_samples']}")
        
        log_file.write("\nTest Results:\n")
        log_file.write(f"  Acc@1:  {test_results['acc@1']:.2f}%\n")
        log_file.write(f"  Acc@5:  {test_results['acc@5']:.2f}%\n")
        log_file.write(f"  Acc@10: {test_results['acc@10']:.2f}%\n")
        log_file.write(f"  MRR:    {test_results['mrr']:.2f}%\n")
        log_file.write(f"  F1:     {test_results['f1']:.2f}%\n")
        log_file.write(f"  NDCG:   {test_results['ndcg']:.2f}%\n")
    
    # Save results
    # Save config
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        yaml.dump(dict(config), f, default_flow_style=False)
    
    # Copy original config
    if os.path.exists(args.config):
        shutil.copy(args.config, os.path.join(experiment_dir, "config_original.yaml"))
    
    # Save validation results
    if val_results:
        val_results["dataset"] = dataset_name
        val_results["markov_order"] = n
        val_results["total_parameters"] = total_params
        val_results["training_time_seconds"] = training_time
        val_results["num_users"] = train_data['user_id'].nunique()
        
        val_path = os.path.join(experiment_dir, "val_results.json")
        with open(val_path, "w") as f:
            json.dump(val_results, f, indent=2)
    
    # Save test results
    if test_results:
        test_results["dataset"] = dataset_name
        test_results["markov_order"] = n
        test_results["total_parameters"] = total_params
        test_results["training_time_seconds"] = training_time
        test_results["num_users"] = train_data['user_id'].nunique()
        
        test_path = os.path.join(experiment_dir, "test_results.json")
        with open(test_path, "w") as f:
            json.dump(test_results, f, indent=2)
    
    log_file.write("\n=== Training Complete ===\n")
    log_file.close()
    
    print(f"\nResults saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
