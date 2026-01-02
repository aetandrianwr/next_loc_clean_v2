"""
RNN Baseline Model for Next Location Prediction.

This module implements a vanilla RNN-based model for next location prediction,
designed as a baseline comparison against the Pointer Network V45.

Architecture:
- Location embedding only (standard RNN approach)
- Multi-layer vanilla RNN (Elman RNN) encoder  
- Classification head over full vocabulary

Key Design Principles for Scientific Comparison:
1. Standard RNN approach without advanced feature engineering
2. Only uses location sequence and user ID (no temporal features)
3. This shows the baseline performance of a "standard" sequence model
4. Expected to perform WORSE than LSTM due to:
   - Vanishing gradient problem
   - Limited long-term dependency modeling
   - No gating mechanism

Usage:
    from src.models.baselines.rnn_baseline import RNNBaseline
    
    model = RNNBaseline(
        num_locations=1000,
        num_users=100,
        d_model=64,
        hidden_size=128,
        num_layers=2,
        dropout=0.15,
    )
    
    # Forward pass
    # x: [seq_len, batch_size] - location sequence
    # x_dict: dictionary with 'user', 'len'
    logits = model(x, x_dict)  # [batch_size, num_locations]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RNNBaseline(nn.Module):
    """
    Vanilla RNN Baseline Model for Next Location Prediction.
    
    This model predicts the next location using a vanilla RNN (Elman RNN)
    architecture with location embeddings and user embeddings. It does NOT 
    use the rich temporal features of Pointer V45.
    
    The vanilla RNN uses simple recurrence without gating:
        h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b)
    
    Key Difference from Pointer V45:
    - No temporal embeddings (time, weekday, duration, recency)
    - No position-from-end embedding
    - No attention mechanism
    - No pointer network
    - No LSTM gating
    
    This represents the simplest possible recurrent baseline.
    
    Args:
        num_locations (int): Total number of unique locations in vocabulary
        num_users (int): Total number of unique users
        d_model (int): Dimension of embeddings (default: 64)
        hidden_size (int): RNN hidden state dimension (default: 128)
        num_layers (int): Number of RNN layers (default: 2)
        dropout (float): Dropout probability (default: 0.15)
        max_seq_len (int): Maximum sequence length (default: 150)
    
    Input:
        x (torch.Tensor): Location sequence tensor of shape [seq_len, batch_size]
        x_dict (dict): Dictionary containing:
            - 'user': User IDs [batch_size]
            - 'len': Sequence lengths [batch_size]
    
    Output:
        torch.Tensor: Logits over locations [batch_size, num_locations]
    """
    
    def __init__(
        self,
        num_locations: int,
        num_users: int,
        d_model: int = 64,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.15,
        max_seq_len: int = 150,
    ):
        super().__init__()
        
        self.num_locations = num_locations
        self.d_model = d_model
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Core embeddings (location only - standard RNN baseline)
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 2, padding_idx=0)
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # Vanilla RNN encoder
        rnn_dropout = dropout if num_layers > 1 else 0
        self.rnn = nn.RNN(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=rnn_dropout,
            batch_first=True,
            bidirectional=False,  # Unidirectional for causal prediction
            nonlinearity='tanh'   # Standard tanh activation
        )
        
        # Output layers: combine RNN output with user embedding
        self.output_norm = nn.LayerNorm(hidden_size + d_model // 2)
        self.output_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size + d_model // 2, num_locations)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization for better gradients
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: Zero initialization
                nn.init.zeros_(param)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param)
    
    def forward(self, x: torch.Tensor, x_dict: dict) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Location sequence [seq_len, batch_size]
            x_dict: Dictionary with 'user', 'len' (other features ignored)
            
        Returns:
            Logits [batch_size, num_locations]
        """
        x = x.T  # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        device = x.device
        lengths = x_dict['len']
        
        # Location embedding only (standard RNN baseline)
        loc_emb = self.loc_emb(x)  # [batch_size, seq_len, d_model]
        
        # Apply normalization and dropout
        hidden = self.input_norm(loc_emb)
        hidden = self.input_dropout(hidden)
        
        # Pack padded sequence for RNN
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = pack_padded_sequence(
            hidden, 
            lengths_cpu, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # RNN forward
        packed_output, h_n = self.rnn(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get last valid output for each sequence
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = output[batch_idx, last_idx]  # [batch_size, hidden_size]
        
        # Get user embedding
        user_emb = self.user_emb(x_dict['user'])  # [batch_size, d_model // 2]
        
        # Concatenate RNN output with user embedding
        context = torch.cat([context, user_emb], dim=-1)
        
        # Output projection
        context = self.output_norm(context)
        context = self.output_dropout(context)
        logits = self.classifier(context)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
