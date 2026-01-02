"""
Baseline models for next location prediction.

This package contains baseline LSTM and RNN models that serve as
comparison baselines for the proposed Pointer Network model.
"""

from .lstm_baseline import LSTMBaseline
from .rnn_baseline import RNNBaseline

__all__ = ['LSTMBaseline', 'RNNBaseline']
