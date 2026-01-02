"""
LSTM Baseline Model for Next Location Prediction.

This module implements an LSTM-based model for next location prediction,
designed as a baseline comparison against the Pointer Network V45.

Architecture:
- Location embedding only (standard LSTM approach)
- Multi-layer LSTM encoder  
- Classification head over full vocabulary

Key Design Principles for Scientific Comparison:
1. Standard LSTM approach without advanced feature engineering
2. Only uses location sequence and user ID (no temporal features)
3. This shows the baseline performance of a "standard" sequence model
4. Pointer V45's advantage comes from:
   - Pointer mechanism for copy
   - Rich temporal features
   - Attention mechanism

Usage:
    from src.models.baselines.lstm_baseline import LSTMBaseline
    
    model = LSTMBaseline(
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


class LSTMBaseline(nn.Module):
    """
    LSTM Baseline Model for Next Location Prediction.
    
    This model predicts the next location using a standard LSTM architecture
    with location embeddings and user embeddings. It does NOT use the rich
    temporal features of Pointer V45 to demonstrate the advantage of the
    proposed model's design choices.
    
    Key Difference from Pointer V45:
    - No temporal embeddings (time, weekday, duration, recency)
    - No position-from-end embedding
    - No attention mechanism
    - No pointer network
    
    This represents a "standard" LSTM baseline as commonly used in sequence
    prediction tasks.
    
    Args:
        num_locations (int): Total number of unique locations in vocabulary
        num_users (int): Total number of unique users
        d_model (int): Dimension of embeddings (default: 64)
        hidden_size (int): LSTM hidden state dimension (default: 128)
        num_layers (int): Number of LSTM layers (default: 2)
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
        
        # Core embeddings (location only - standard LSTM baseline)
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model // 2, padding_idx=0)
        
        # Input layer normalization
        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)
        
        # LSTM encoder
        lstm_dropout = dropout if num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=False  # Unidirectional for causal prediction
        )
        
        # Output layers: combine LSTM output with user embedding
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
                # Hidden-hidden weights: Orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name and 'lstm' in name:
                # LSTM biases: Zero initialization with forget gate bias set to 1
                nn.init.zeros_(param)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
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
        
        # Location embedding only (standard LSTM baseline)
        loc_emb = self.loc_emb(x)  # [batch_size, seq_len, d_model]
        
        # Apply normalization and dropout
        hidden = self.input_norm(loc_emb)
        hidden = self.input_dropout(hidden)
        
        # Pack padded sequence for LSTM
        lengths_cpu = lengths.cpu().clamp(min=1)
        packed = pack_padded_sequence(
            hidden, 
            lengths_cpu, 
            batch_first=True, 
            enforce_sorted=False
        )
        
        # LSTM forward
        packed_output, (h_n, c_n) = self.lstm(packed)
        
        # Unpack
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Get last valid output for each sequence
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = output[batch_idx, last_idx]  # [batch_size, hidden_size]
        
        # Get user embedding
        user_emb = self.user_emb(x_dict['user'])  # [batch_size, d_model // 2]
        
        # Concatenate LSTM output with user embedding
        context = torch.cat([context, user_emb], dim=-1)
        
        # Output projection
        context = self.output_norm(context)
        context = self.output_dropout(context)
        logits = self.classifier(context)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
