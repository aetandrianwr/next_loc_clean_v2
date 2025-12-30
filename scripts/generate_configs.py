#!/usr/bin/env python3
"""
Generate config YAML files for previous days analysis.
Creates configs for both Geolife and DIY datasets with various hyperparameter combinations.
"""

import os
import yaml

# Previous days to test
PREV_DAYS = [3, 7, 10, 14, 17, 21]

# Geolife configurations (6 configs)
GEOLIFE_CONFIGS = {
    "baseline_d64_L2": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "learning_rate": 6.5e-4},
    "d80_L3_deeper": {"d_model": 80, "nhead": 4, "num_layers": 3, "dim_feedforward": 160, "learning_rate": 6.0e-4},
    "d64_L2_ff192_highLR": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 192, "learning_rate": 8.0e-4},
    "d64_L3_lowLR_highDrop": {"d_model": 64, "nhead": 4, "num_layers": 3, "dim_feedforward": 128, "learning_rate": 5.0e-4},
    "d64_L2_lowDropout": {"d_model": 64, "nhead": 4, "num_layers": 2, "dim_feedforward": 128, "learning_rate": 6.0e-4},
    "d72_L2": {"d_model": 72, "nhead": 4, "num_layers": 2, "dim_feedforward": 144, "learning_rate": 6.5e-4},
}

# DIY configurations (5 configs)
DIY_CONFIGS = {
    "baseline_d128_L3": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 7.0e-4},
    "d128_L4_deeper": {"d_model": 128, "nhead": 4, "num_layers": 4, "dim_feedforward": 256, "learning_rate": 6.0e-4},
    "d128_L3_highLR": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 9.0e-4},
    "d144_L3_largerEmb": {"d_model": 144, "nhead": 4, "num_layers": 3, "dim_feedforward": 288, "learning_rate": 7.0e-4},
    "d128_L3_lowerLR": {"d_model": 128, "nhead": 4, "num_layers": 3, "dim_feedforward": 256, "learning_rate": 6.0e-4},
}


def generate_config(dataset_type, prev_days, config_name, config_params, num_epochs=50, patience=5, seed=42):
    """Generate a single config YAML file."""
    
    if dataset_type == "geolife":
        data_dir = "data/geolife_eps20/processed"
        dataset_prefix = f"geolife_eps20_prev{prev_days}"
    else:  # diy
        data_dir = "data/diy_eps50/processed"
        dataset_prefix = f"diy_eps50_prev{prev_days}"
    
    config = {
        "seed": seed,
        "data": {
            "data_dir": data_dir,
            "dataset_prefix": dataset_prefix,
            "dataset": dataset_type,
            "experiment_root": "experiments",
            "num_workers": 0,
        },
        "model": {
            "d_model": config_params["d_model"],
            "nhead": config_params["nhead"],
            "num_layers": config_params["num_layers"],
            "dim_feedforward": config_params["dim_feedforward"],
            "dropout": 0.15,
        },
        "training": {
            "batch_size": 128,
            "num_epochs": num_epochs,
            "learning_rate": config_params["learning_rate"],
            "weight_decay": 0.015,
            "label_smoothing": 0.03,
            "grad_clip": 0.8,
            "patience": patience,
            "min_epochs": 8,
            "warmup_epochs": 5,
            "use_amp": True,
            "min_lr": 1e-6,
        },
    }
    
    return config


def main():
    output_dir = "config/analysis_prev_days"
    os.makedirs(output_dir, exist_ok=True)
    
    generated_files = []
    
    # Generate Geolife configs
    for prev_days in PREV_DAYS:
        for config_name, config_params in GEOLIFE_CONFIGS.items():
            filename = f"geolife_prev{prev_days}_{config_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            config = generate_config("geolife", prev_days, config_name, config_params)
            
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(filepath)
            print(f"Generated: {filepath}")
    
    # Generate DIY configs
    for prev_days in PREV_DAYS:
        for config_name, config_params in DIY_CONFIGS.items():
            filename = f"diy_prev{prev_days}_{config_name}.yaml"
            filepath = os.path.join(output_dir, filename)
            
            config = generate_config("diy", prev_days, config_name, config_params)
            
            with open(filepath, "w") as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            
            generated_files.append(filepath)
            print(f"Generated: {filepath}")
    
    print(f"\nTotal configs generated: {len(generated_files)}")
    print(f"  Geolife: {len(PREV_DAYS) * len(GEOLIFE_CONFIGS)} configs")
    print(f"  DIY: {len(PREV_DAYS) * len(DIY_CONFIGS)} configs")


if __name__ == "__main__":
    main()
