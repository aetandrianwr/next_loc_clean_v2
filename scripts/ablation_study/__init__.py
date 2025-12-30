"""
Ablation Study Package for Pointer Network V45.

This package provides tools for systematic ablation analysis of the
Pointer Network V45 model for next location prediction.

Modules:
    - pointer_v45_ablation: Model with configurable component ablation
    - train_ablation: Training script for individual ablation experiments
    - run_ablation_study: Main runner for comprehensive ablation study

Usage:
    # Run complete ablation study
    python scripts/ablation_study/run_ablation_study.py --dataset all
    
    # Run single ablation experiment
    python scripts/ablation_study/train_ablation.py --config <config> --ablation <name>
"""

from .pointer_v45_ablation import PointerNetworkV45Ablation
from .train_ablation import ABLATION_CONFIGS

__all__ = ['PointerNetworkV45Ablation', 'ABLATION_CONFIGS']
