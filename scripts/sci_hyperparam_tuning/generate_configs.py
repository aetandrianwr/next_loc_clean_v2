"""
Configuration Generator for Hyperparameter Tuning.

This script generates YAML configuration files for each hyperparameter trial.
It creates configs for all three models (Pointer Generator Transformer, MHSA, LSTM) on both datasets
(Geolife and DIY).

Usage:
    python scripts/sci_hyperparam_tuning/generate_configs.py
"""

import os
import sys
import yaml
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from hyperparam_search_space import (
    generate_all_configs, 
    get_config_name, 
    NUM_TRIALS, 
    RANDOM_SEED,
    set_seed
)


# Dataset-specific settings
DATASET_CONFIGS = {
    'geolife': {
        'data_dir': 'data/geolife_eps20/processed',
        'dataset_prefix': 'geolife_eps20_prev7',
        'total_loc_num': 1187,
        'total_user_num': 46,
    },
    'diy': {
        'data_dir': 'data/diy_eps50/processed',
        'dataset_prefix': 'diy_eps50_prev7',
        'total_loc_num': 7038,
        'total_user_num': 693,
    }
}


def generate_pgt_config(hp_config: dict, dataset: str) -> dict:
    """Generate Pointer Generator Transformer YAML config from hyperparameters."""
    ds_cfg = DATASET_CONFIGS[dataset]
    
    return {
        'seed': 42,
        'data': {
            'data_dir': ds_cfg['data_dir'],
            'dataset_prefix': ds_cfg['dataset_prefix'],
            'dataset': dataset,
            'experiment_root': 'experiments',
            'num_workers': 0,
        },
        'model': {
            'd_model': hp_config['d_model'],
            'nhead': hp_config['nhead'],
            'num_layers': hp_config['num_layers'],
            'dim_feedforward': hp_config['dim_feedforward'],
            'dropout': hp_config['dropout'],
        },
        'training': {
            'batch_size': hp_config['batch_size'],
            'num_epochs': 50,
            'learning_rate': hp_config['learning_rate'],
            'weight_decay': hp_config['weight_decay'],
            'label_smoothing': hp_config['label_smoothing'],
            'grad_clip': 0.8,
            'patience': 5,
            'min_epochs': 8,
            'warmup_epochs': hp_config['warmup_epochs'],
            'use_amp': True,
            'min_lr': 1e-6,
        },
    }


def generate_mhsa_config(hp_config: dict, dataset: str) -> dict:
    """Generate MHSA YAML config from hyperparameters."""
    ds_cfg = DATASET_CONFIGS[dataset]
    
    return {
        'seed': 42,
        'data': {
            'data_dir': ds_cfg['data_dir'],
            'dataset_prefix': ds_cfg['dataset_prefix'],
            'dataset': dataset,
            'experiment_root': 'experiments',
        },
        'training': {
            'if_embed_user': True,
            'if_embed_poi': False,
            'if_embed_time': True,
            'if_embed_duration': True,
            'previous_day': 7,
            'verbose': True,
            'debug': False,
            'batch_size': hp_config['batch_size'],
            'print_step': 20 if dataset == 'geolife' else 50,
            'num_workers': 0,
            'day_selection': 'default',
        },
        'dataset_info': {
            'total_loc_num': ds_cfg['total_loc_num'],
            'total_user_num': ds_cfg['total_user_num'],
        },
        'embedding': {
            'base_emb_size': hp_config['base_emb_size'],
            'poi_original_size': 16,
        },
        'model': {
            'networkName': 'transformer',
            'num_encoder_layers': hp_config['num_encoder_layers'],
            'nhead': hp_config['nhead'],
            'dim_feedforward': hp_config['dim_feedforward'],
            'fc_dropout': hp_config['fc_dropout'],
        },
        'optimiser': {
            'optimizer': 'Adam',
            'max_epoch': 100,
            'lr': hp_config['lr'],
            'weight_decay': hp_config['weight_decay'],
            'beta1': 0.9,
            'beta2': 0.999,
            'momentum': 0.98,
            'num_warmup_epochs': hp_config['num_warmup_epochs'],
            'num_training_epochs': 50,
            'patience': 5,
            'lr_step_size': 1,
            'lr_gamma': 0.1,
        },
    }


def generate_lstm_config(hp_config: dict, dataset: str) -> dict:
    """Generate LSTM YAML config from hyperparameters."""
    ds_cfg = DATASET_CONFIGS[dataset]
    
    return {
        'seed': 42,
        'data': {
            'data_dir': ds_cfg['data_dir'],
            'dataset_prefix': ds_cfg['dataset_prefix'],
            'dataset': dataset,
            'experiment_root': 'experiments',
        },
        'training': {
            'if_embed_user': True,
            'if_embed_poi': False,
            'if_embed_time': True,
            'if_embed_duration': True,
            'previous_day': 7,
            'verbose': True,
            'debug': False,
            'batch_size': hp_config['batch_size'],
            'print_step': 20 if dataset == 'geolife' else 50,
            'num_workers': 0,
            'day_selection': 'default',
        },
        'dataset_info': {
            'total_loc_num': ds_cfg['total_loc_num'],
            'total_user_num': ds_cfg['total_user_num'],
        },
        'embedding': {
            'base_emb_size': hp_config['base_emb_size'],
            'poi_original_size': 16,
        },
        'model': {
            'networkName': 'lstm',
            'lstm_hidden_size': hp_config['lstm_hidden_size'],
            'lstm_num_layers': hp_config['lstm_num_layers'],
            'lstm_dropout': hp_config['lstm_dropout'],
            'fc_dropout': hp_config['fc_dropout'],
        },
        'optimiser': {
            'optimizer': 'Adam',
            'max_epoch': 100,
            'lr': hp_config['lr'],
            'weight_decay': hp_config['weight_decay'],
            'beta1': 0.9,
            'beta2': 0.999,
            'momentum': 0.98,
            'num_warmup_epochs': hp_config['num_warmup_epochs'],
            'num_training_epochs': 50,
            'patience': 5,
            'lr_step_size': 1,
            'lr_gamma': 0.1,
        },
    }


def main():
    """Generate all configuration files for hyperparameter tuning."""
    set_seed(RANDOM_SEED)
    
    # Output directory
    config_dir = Path(__file__).parent / 'configs'
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Config generators by model
    generators = {
        "pgt": generate_pgt_config,
        'mhsa': generate_mhsa_config,
        'lstm': generate_lstm_config,
    }
    
    # Generate configs for all model-dataset combinations
    all_configs = []
    
    for model_name in ["pgt", 'mhsa', 'lstm']:
        for dataset in ['geolife', 'diy']:
            hp_configs = generate_all_configs(model_name, dataset, num_trials=NUM_TRIALS)
            
            for hp_config in hp_configs:
                # Generate config name
                config_name = get_config_name(model_name, dataset, hp_config['trial_idx'])
                
                # Generate YAML config
                yaml_config = generators[model_name](hp_config, dataset)
                
                # Save config
                config_path = config_dir / f'{config_name}.yaml'
                with open(config_path, 'w') as f:
                    # Add header comment
                    f.write(f"# Hyperparameter Tuning Config: {config_name}\n")
                    f.write(f"# Model: {model_name}, Dataset: {dataset}, Trial: {hp_config['trial_idx']}\n")
                    f.write(f"# Hyperparameters: {hp_config}\n\n")
                    yaml.dump(yaml_config, f, default_flow_style=False, sort_keys=False)
                
                all_configs.append({
                    'config_name': config_name,
                    'config_path': str(config_path),
                    'model_name': model_name,
                    'dataset': dataset,
                    'trial_idx': hp_config['trial_idx'],
                    'hyperparameters': hp_config,
                })
    
    print(f"Generated {len(all_configs)} configuration files in {config_dir}")
    
    # Save all configs summary
    summary_path = config_dir / 'all_configs_summary.yaml'
    with open(summary_path, 'w') as f:
        yaml.dump(all_configs, f, default_flow_style=False)
    
    print(f"Saved configs summary to {summary_path}")
    
    return all_configs


if __name__ == "__main__":
    main()
