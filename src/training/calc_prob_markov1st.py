"""
Training script for 1st-Order Markov Chain Model.

This script handles:
- Configuration loading from YAML files
- Data loading from preprocessed pickle files
- Transition probability calculation (model fitting)
- Evaluation on validation and test sets using src/evaluation/metrics.py
- Saving checkpoints, logs, and metrics to experiments/ directory

Usage:
    python src/training/calc_prob_markov1st.py --config config/models/config_markov1st_geolife.yaml
    python src/training/calc_prob_markov1st.py --config config/models/config_markov1st_diy.yaml

Output folder structure:
    experiments/{dataset_name}_markov1st_{yyyyMMdd_hhmmss}/
        ├── checkpoints/
        │   └── markov1st_model.pkl
        ├── training.log
        ├── config.yaml
        ├── config_original.yaml
        ├── val_results.json
        └── test_results.json
"""

import os
import sys
import json
import yaml
import random
import argparse
import pickle
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import shutil

import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.models.baseline.markov1st import Markov1stModel
from src.evaluation.metrics import calculate_metrics


def setup_seed(seed):
    """Fix random seed for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path):
    """
    Load configuration from YAML file.
    
    Args:
        path: Path to YAML config file
    
    Returns:
        Flattened configuration dictionary
    """
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

    def __delattr__(self, name):
        del self[name]


def load_data(config):
    """
    Load train, validation, and test data from pickle files.
    
    Args:
        config: Configuration dictionary with data_dir and dataset_prefix.
    
    Returns:
        tuple: (train_data, val_data, test_data) as lists of dictionaries.
    """
    data_dir = config.data_dir
    dataset_prefix = config.dataset_prefix

    train_path = os.path.join(data_dir, f"{dataset_prefix}_train.pk")
    val_path = os.path.join(data_dir, f"{dataset_prefix}_validation.pk")
    test_path = os.path.join(data_dir, f"{dataset_prefix}_test.pk")

    with open(train_path, "rb") as f:
        train_data = pickle.load(f)
    with open(val_path, "rb") as f:
        val_data = pickle.load(f)
    with open(test_path, "rb") as f:
        test_data = pickle.load(f)

    return train_data, val_data, test_data


def load_metadata(config):
    """
    Load dataset metadata from JSON file.
    
    Args:
        config: Configuration dictionary with data_dir and dataset_prefix.
    
    Returns:
        dict: Metadata dictionary containing dataset_name, num_locations, etc.
    """
    metadata_path = os.path.join(config.data_dir, f"{config.dataset_prefix}_metadata.json")
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    return metadata


def init_experiment_dir(config, dataset_name, model_name="markov1st"):
    """
    Create experiment directory with dataset name, model name, and timestamp.
    
    Format: experiments/{dataset_name}_{model_name}_{yyyyMMdd_hhmmss}/
    
    Args:
        config: Configuration dictionary with experiment_root.
        dataset_name: Name of the dataset (from metadata).
        model_name: Name of the model.
    
    Returns:
        str: Path to the experiment directory.
    """
    # Get current time in GMT+7
    gmt7 = timezone(timedelta(hours=7))
    now = datetime.now(gmt7)
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    
    experiment_name = f"{dataset_name}_{model_name}_{timestamp}"
    experiment_dir = os.path.join(config.experiment_root, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, "checkpoints"), exist_ok=True)
    
    return experiment_dir


def evaluate_model(model, data, num_locations, split_name="test", log_file=None, verbose=True):
    """
    Evaluate the Markov model on a dataset split.
    
    Uses src/evaluation/metrics.py for calculating all metrics.
    
    Args:
        model: Fitted Markov1stModel.
        data: List of data samples.
        num_locations: Total number of locations.
        split_name: Name of the split (for logging).
        log_file: File object for logging.
        verbose: Whether to print progress.
    
    Returns:
        dict: Performance metrics dictionary.
    """
    if verbose:
        print(f"\nEvaluating on {split_name} set ({len(data)} samples)...")
    if log_file:
        log_file.write(f"\nEvaluating on {split_name} set ({len(data)} samples)...\n")
    
    start_time = time.time()
    
    # Get predictions as logits for compatibility with metrics module
    logits, targets = model.predict_as_logits(data)
    
    # Calculate metrics using the standard metrics module
    metrics = calculate_metrics(logits, targets)
    
    eval_time = time.time() - start_time
    
    # Log results
    result_msg = (
        f"{split_name.capitalize()} Results:\n"
        f"  Acc@1:  {metrics['acc@1']:.2f}%\n"
        f"  Acc@5:  {metrics['acc@5']:.2f}%\n"
        f"  Acc@10: {metrics['acc@10']:.2f}%\n"
        f"  MRR:    {metrics['mrr']:.2f}%\n"
        f"  NDCG:   {metrics['ndcg']:.2f}%\n"
        f"  F1:     {metrics['f1']:.2f}%\n"
        f"  Total:  {int(metrics['total'])}\n"
        f"  Evaluation time: {eval_time:.2f}s\n"
    )
    
    if verbose:
        print(result_msg)
    if log_file:
        log_file.write(result_msg)
    
    return metrics


def save_results(experiment_dir, config, val_metrics, test_metrics, config_path, model):
    """
    Save all results to experiment directory.
    
    Args:
        experiment_dir: Path to experiment directory.
        config: Configuration dictionary.
        val_metrics: Validation metrics dictionary.
        test_metrics: Test metrics dictionary.
        config_path: Path to original config file.
        model: Trained model to save.
    """
    # Save configuration
    config_save_path = os.path.join(experiment_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        config_dict = dict(config)
        yaml.dump(config_dict, f, default_flow_style=False)
    
    # Copy the original config file
    if os.path.exists(config_path):
        shutil.copy(config_path, os.path.join(experiment_dir, "config_original.yaml"))
    
    # Save validation results
    val_results_path = os.path.join(experiment_dir, "val_results.json")
    val_results = {
        k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v 
        for k, v in val_metrics.items()
    }
    with open(val_results_path, "w") as f:
        json.dump(val_results, f, indent=2)
    
    # Save test results
    test_results_path = os.path.join(experiment_dir, "test_results.json")
    test_results = {
        k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v 
        for k, v in test_metrics.items()
    }
    with open(test_results_path, "w") as f:
        json.dump(test_results, f, indent=2)
    
    # Save model checkpoint
    model_path = os.path.join(experiment_dir, "checkpoints", "markov1st_model.pkl")
    model.save(model_path)


def main():
    """Main entry point for training 1st-order Markov model."""
    parser = argparse.ArgumentParser(
        description="Calculate transition probabilities for 1st-order Markov Chain model"
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

    # Setup seed
    seed = config.get("seed", 42)
    setup_seed(seed)
    print(f"Random seed: {seed}")

    # Load metadata to get dataset name and number of locations
    metadata = load_metadata(config)
    dataset_name = metadata["dataset_name"]
    num_locations = metadata["total_loc_num"]
    
    print(f"Dataset: {dataset_name}")
    print(f"Number of locations: {num_locations}")

    # Initialize experiment directory
    experiment_dir = init_experiment_dir(config, dataset_name, model_name="markov1st")
    print(f"Experiment directory: {experiment_dir}")

    # Create log file
    log_path = os.path.join(experiment_dir, "training.log")
    log_file = open(log_path, "w")
    log_file.write("=" * 60 + "\n")
    log_file.write("1st-Order Markov Chain Model Training Log\n")
    log_file.write("=" * 60 + "\n\n")
    log_file.write(f"Dataset: {dataset_name}\n")
    log_file.write(f"Number of locations: {num_locations}\n")
    log_file.write(f"Config: {args.config}\n")
    log_file.write(f"Seed: {seed}\n")
    log_file.write(f"Experiment directory: {experiment_dir}\n")
    log_file.write("=" * 60 + "\n")

    # Load data
    print("\nLoading data...")
    train_data, val_data, test_data = load_data(config)
    print(f"Train: {len(train_data)} samples")
    print(f"Val:   {len(val_data)} samples")
    print(f"Test:  {len(test_data)} samples")
    
    log_file.write(f"\nData loaded:\n")
    log_file.write(f"  Train: {len(train_data)} samples\n")
    log_file.write(f"  Val:   {len(val_data)} samples\n")
    log_file.write(f"  Test:  {len(test_data)} samples\n")

    # Create and fit model
    print("\n" + "=" * 60)
    print("Training 1st-Order Markov Chain Model")
    print("=" * 60)
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write("Training 1st-Order Markov Chain Model\n")
    log_file.write("=" * 60 + "\n")
    
    training_start_time = time.time()
    
    model = Markov1stModel(num_locations=num_locations, random_seed=seed)
    model.fit(train_data)
    
    training_time = time.time() - training_start_time
    total_params = model.get_total_parameters()
    
    print(f"\nTraining completed in {training_time:.2f}s")
    print(f"Total parameters (unique transitions): {total_params}")
    
    log_file.write(f"\nTraining completed in {training_time:.2f}s\n")
    log_file.write(f"Total parameters (unique transitions): {total_params}\n")

    # Evaluate on validation set
    print("\n" + "=" * 60)
    print("Evaluation")
    print("=" * 60)
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write("Evaluation\n")
    log_file.write("=" * 60 + "\n")
    
    val_metrics = evaluate_model(
        model, val_data, num_locations, 
        split_name="validation", 
        log_file=log_file, 
        verbose=True
    )

    # Evaluate on test set
    test_metrics = evaluate_model(
        model, test_data, num_locations, 
        split_name="test", 
        log_file=log_file, 
        verbose=True
    )

    # Save all results
    print("\nSaving results...")
    save_results(experiment_dir, config, val_metrics, test_metrics, args.config, model)
    
    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(f"Experiment directory: {experiment_dir}")
    print(f"Test Acc@1: {test_metrics['acc@1']:.2f}%")
    print(f"Test MRR:   {test_metrics['mrr']:.2f}%")
    print(f"Test NDCG:  {test_metrics['ndcg']:.2f}%")
    print(f"Test F1:    {test_metrics['f1']:.2f}%")
    
    log_file.write("\n" + "=" * 60 + "\n")
    log_file.write("FINAL SUMMARY\n")
    log_file.write("=" * 60 + "\n")
    log_file.write(f"Test Acc@1: {test_metrics['acc@1']:.2f}%\n")
    log_file.write(f"Test MRR:   {test_metrics['mrr']:.2f}%\n")
    log_file.write(f"Test NDCG:  {test_metrics['ndcg']:.2f}%\n")
    log_file.write(f"Test F1:    {test_metrics['f1']:.2f}%\n")
    log_file.write("\n=== Training Complete ===\n")
    
    log_file.close()
    
    print(f"\nAll results saved to: {experiment_dir}")


if __name__ == "__main__":
    main()
