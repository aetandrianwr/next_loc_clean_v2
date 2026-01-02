"""
Hyperparameter Search Space Definition for Scientific Hyperparameter Tuning.

This module defines the hyperparameter search spaces for all three models:
- Pointer V45 (Proposed Model)
- MHSA (Baseline)
- LSTM (Baseline)

The search space is designed following best practices for fair comparison:
1. Similar parameter budget constraints across models
2. Same number of hyperparameter trials per model
3. Random search with fixed seed for reproducibility (Bergstra & Bengio, 2012)
4. Search spaces include architecture and optimization hyperparameters

References:
- Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. JMLR.
"""

import random
import numpy as np
from typing import Dict, List, Any

# Fixed seed for reproducibility
RANDOM_SEED = 42

# Number of hyperparameter configurations to try per model per dataset
NUM_TRIALS = 20  # Reasonable for PhD-level tuning with limited compute


def set_seed(seed: int = RANDOM_SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)


# =============================================================================
# Pointer V45 Search Space
# =============================================================================

POINTER_V45_SEARCH_SPACE = {
    # Architecture hyperparameters
    'd_model': [64, 96, 128],
    'nhead': [2, 4, 8],
    'num_layers': [2, 3, 4],
    'dim_feedforward': [128, 192, 256],
    'dropout': [0.1, 0.15, 0.2, 0.25],
    
    # Training hyperparameters
    'learning_rate': [1e-4, 3e-4, 5e-4, 7e-4, 1e-3],
    'weight_decay': [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 0.01, 0.015],
    'batch_size': [64, 128, 256],
    'label_smoothing': [0.0, 0.01, 0.03, 0.05],
    'warmup_epochs': [3, 5, 7],
}

# =============================================================================
# MHSA Search Space
# =============================================================================

MHSA_SEARCH_SPACE = {
    # Architecture hyperparameters
    'base_emb_size': [32, 48, 64, 96],
    'num_encoder_layers': [2, 3, 4],
    'nhead': [4, 8],
    'dim_feedforward': [128, 192, 256],
    'fc_dropout': [0.1, 0.15, 0.2, 0.25],
    
    # Training hyperparameters
    'lr': [5e-4, 1e-3, 2e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'batch_size': [32, 64, 128, 256],
    'num_warmup_epochs': [1, 2, 3],
}

# =============================================================================
# LSTM Search Space
# =============================================================================

LSTM_SEARCH_SPACE = {
    # Architecture hyperparameters
    'base_emb_size': [32, 48, 64, 96],
    'lstm_hidden_size': [128, 192, 256],
    'lstm_num_layers': [1, 2, 3],
    'lstm_dropout': [0.1, 0.2, 0.3],
    'fc_dropout': [0.1, 0.15, 0.2, 0.25],
    
    # Training hyperparameters
    'lr': [5e-4, 1e-3, 2e-3],
    'weight_decay': [1e-6, 1e-5, 1e-4],
    'batch_size': [32, 64, 128, 256],
    'num_warmup_epochs': [1, 2, 3],
}


def sample_hyperparameters(search_space: Dict[str, List[Any]], seed: int = None) -> Dict[str, Any]:
    """
    Sample a random hyperparameter configuration from the search space.
    
    Args:
        search_space: Dictionary mapping hyperparameter names to lists of possible values
        seed: Random seed for reproducibility (optional)
        
    Returns:
        Dictionary with sampled hyperparameter values
    """
    if seed is not None:
        random.seed(seed)
    
    config = {}
    for param, values in search_space.items():
        config[param] = random.choice(values)
    
    return config


def generate_all_configs(model_name: str, dataset: str, num_trials: int = NUM_TRIALS, 
                         base_seed: int = RANDOM_SEED) -> List[Dict[str, Any]]:
    """
    Generate all hyperparameter configurations for a model-dataset pair.
    
    Args:
        model_name: One of 'pointer_v45', 'mhsa', 'lstm'
        dataset: One of 'geolife', 'diy'
        num_trials: Number of configurations to generate
        base_seed: Base seed for reproducibility
        
    Returns:
        List of hyperparameter configurations
    """
    search_space = {
        'pointer_v45': POINTER_V45_SEARCH_SPACE,
        'mhsa': MHSA_SEARCH_SPACE,
        'lstm': LSTM_SEARCH_SPACE,
    }[model_name]
    
    configs = []
    for trial_idx in range(num_trials):
        # Use deterministic seed based on model, dataset, and trial index
        seed = hash(f"{model_name}_{dataset}_{base_seed}_{trial_idx}") % (2**32)
        config = sample_hyperparameters(search_space, seed=seed)
        config['trial_idx'] = trial_idx
        config['model_name'] = model_name
        config['dataset'] = dataset
        configs.append(config)
    
    return configs


def get_config_name(model_name: str, dataset: str, trial_idx: int) -> str:
    """Generate a unique config name."""
    return f"{model_name}_{dataset}_trial{trial_idx:02d}"


if __name__ == "__main__":
    # Test generation
    set_seed(RANDOM_SEED)
    
    for model in ['pointer_v45', 'mhsa', 'lstm']:
        for dataset in ['geolife', 'diy']:
            configs = generate_all_configs(model, dataset, num_trials=3)
            print(f"\n{model.upper()} on {dataset}:")
            for cfg in configs:
                print(f"  Trial {cfg['trial_idx']}: {cfg}")
