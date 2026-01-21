"""
Ablation Study Package for Pointer Generator Transformer.

This package provides tools for systematic ablation analysis of the
Pointer Generator Transformer model for next location prediction.

Modules:
    - pgt_ablation: Model with configurable component ablation
    - train_ablation: Training script for individual ablation experiments
    - run_ablation_study: Main runner for comprehensive ablation study

Usage:
    # Run complete ablation study
    python scripts/ablation_study/run_ablation_study.py --dataset all
    
    # Run single ablation experiment
    python scripts/ablation_study/train_ablation.py --config <config> --ablation <name>
"""

from .pgt_ablation import PointerGeneratorTransformerAblation
from .train_ablation import ABLATION_CONFIGS

__all__ = ['PointerGeneratorTransformerAblation', 'ABLATION_CONFIGS']
