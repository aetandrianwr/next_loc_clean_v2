"""
LSTM Baseline Model for Next Location Prediction.

This module implements an LSTM-based baseline model for next location prediction,
following the architecture described in:
    Hong et al., 2023 - "Context-aware multi-head self-attentional neural network 
    model for next location prediction" (Transportation Research Part C)

The LSTM baseline in the paper achieves 28.4% Acc@1 on Geolife dataset.

Architecture:
- Location embedding + Temporal embeddings (time, weekday, duration)
- Multi-layer LSTM for sequence modeling
- Fully connected output layer with optional user embedding and residual connection

Key design choices for fair comparison:
- Same input features as MHSA model (location, time, weekday, duration, user)
- Same embedding approach (additive embeddings)
- Same output layer structure (FC residual block with user embedding)
- Same training setup (Adam optimizer, early stopping, learning rate schedule)

Usage:
    from src.models.baseline.LSTM_Baseline import LSTMBaseline
    
    model = LSTMBaseline(config=config, total_loc_num=1187)
    logits = model(x, x_dict, device)  # [batch_size, total_loc_num]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time-related features.
    
    Embeds minute (quarter-hour), hour, and weekday separately and combines them.
    This matches the MHSA baseline implementation for fair comparison.
    
    Args:
        d_input (int): Embedding dimension
    """
    def __init__(self, d_input):
        super(TemporalEmbedding, self).__init__()
        
        # Quarter of an hour (4 intervals)
        self.minute_size = 4
        hour_size = 24
        weekday_size = 7
        
        self.minute_embed = nn.Embedding(self.minute_size, d_input)
        self.hour_embed = nn.Embedding(hour_size, d_input)
        self.weekday_embed = nn.Embedding(weekday_size, d_input)

    def forward(self, time, weekday):
        """
        Args:
            time: Time in 15-min intervals [seq_len, batch_size]
            weekday: Day of week [seq_len, batch_size]
        
        Returns:
            Combined temporal embedding [seq_len, batch_size, d_input]
        """
        hour = torch.div(time, self.minute_size, rounding_mode="floor")
        minutes = time % 4
        
        # Clamp values to valid ranges
        hour = torch.clamp(hour, 0, 23)
        minutes = torch.clamp(minutes, 0, 3)
        weekday = torch.clamp(weekday, 0, 6)

        minute_x = self.minute_embed(minutes)
        hour_x = self.hour_embed(hour)
        weekday_x = self.weekday_embed(weekday)

        return hour_x + minute_x + weekday_x


class AllEmbedding(nn.Module):
    """
    Combined embedding layer for all input features.
    
    Combines location embedding with temporal embeddings for time and duration.
    Uses additive combination as in the MHSA paper.
    
    Args:
        d_input (int): Base embedding dimension
        config: Configuration object with embedding settings
        total_loc_num (int): Total number of unique locations
    """
    def __init__(self, d_input, config, total_loc_num):
        super(AllEmbedding, self).__init__()
        self.d_input = d_input

        # Location embedding
        self.emb_loc = nn.Embedding(total_loc_num, d_input, padding_idx=0)

        # Time embedding
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            self.temporal_embedding = TemporalEmbedding(d_input)

        # Duration embedding (in minutes, max 2 days, 30-min buckets)
        self.if_include_duration = config.if_embed_duration
        if self.if_include_duration:
            self.emb_duration = nn.Embedding(60 * 24 * 2 // 30, d_input)

        self.dropout = nn.Dropout(0.1)

    def forward(self, src, context_dict) -> Tensor:
        """
        Args:
            src: Location sequence [seq_len, batch_size]
            context_dict: Dictionary with time, weekday, duration
        
        Returns:
            Combined embedding [seq_len, batch_size, d_input]
        """
        emb = self.emb_loc(src)

        if self.if_include_time:
            emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday"])

        if self.if_include_duration:
            duration = torch.clamp(context_dict["duration"], 0, 60 * 24 * 2 // 30 - 1)
            emb = emb + self.emb_duration(duration)

        return self.dropout(emb)


class FullyConnected(nn.Module):
    """
    Output layer with user embedding and residual connections.
    
    Matches the MHSA paper's FC residual block for fair comparison.
    
    Args:
        d_input (int): Input dimension
        config: Configuration object
        total_loc_num (int): Total number of locations (output dimension)
    """
    def __init__(self, d_input, config, total_loc_num):
        super(FullyConnected, self).__init__()

        fc_dim = d_input
        
        # User embedding
        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, fc_dim)

        self.fc_loc = nn.Linear(fc_dim, total_loc_num)
        self.emb_dropout = nn.Dropout(p=0.1)

        # Residual block
        self.linear1 = nn.Linear(fc_dim, fc_dim * 2)
        self.linear2 = nn.Linear(fc_dim * 2, fc_dim)

        self.norm1 = nn.BatchNorm1d(fc_dim)
        self.fc_dropout1 = nn.Dropout(p=config.fc_dropout)
        self.fc_dropout2 = nn.Dropout(p=config.fc_dropout)

    def forward(self, out, user) -> Tensor:
        """
        Args:
            out: LSTM output [batch_size, d_input]
            user: User IDs [batch_size]
        
        Returns:
            Logits [batch_size, total_loc_num]
        """
        if self.if_embed_user:
            out = out + self.emb_user(user)

        out = self.emb_dropout(out)
        out = self.norm1(out + self._res_block(out))

        return self.fc_loc(out)

    def _res_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
        return self.fc_dropout2(x)


class LSTMBaseline(nn.Module):
    """
    LSTM Baseline Model for Next Location Prediction.
    
    This is a standard LSTM-based model that serves as a baseline for comparison
    with more advanced architectures like MHSA and Pointer Networks.
    
    Architecture:
    1. AllEmbedding: Combines location, time, and duration embeddings
    2. LSTM: Multi-layer LSTM for sequence modeling
    3. FullyConnected: Output layer with user embedding and residual connection
    
    From the Hong et al. 2023 paper:
    - LSTM achieves 28.4% Acc@1 on Geolife dataset
    - Uses same input features as MHSA for fair comparison
    - Hidden state from last time step is used for prediction
    
    Args:
        config: Configuration object containing model hyperparameters
        total_loc_num (int): Total number of unique locations
    
    Config requirements:
        - base_emb_size: Base embedding dimension
        - hidden_size: LSTM hidden state dimension
        - num_layers: Number of LSTM layers
        - lstm_dropout: Dropout between LSTM layers
        - if_embed_user: Whether to embed user IDs
        - if_embed_time: Whether to embed time features
        - if_embed_duration: Whether to embed duration
        - fc_dropout: Dropout rate for FC layer
        - total_user_num: Total number of users
    """
    def __init__(self, config, total_loc_num) -> None:
        super(LSTMBaseline, self).__init__()

        self.d_input = config.base_emb_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        
        # Embedding layer
        self.Embedding = AllEmbedding(self.d_input, config, total_loc_num)

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.d_input,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=config.lstm_dropout if self.num_layers > 1 else 0,
            batch_first=False,  # Input shape: [seq_len, batch, features]
        )

        # Output layer
        self.FC = FullyConnected(self.hidden_size, config, total_loc_num)

        # Initialize parameters
        self._init_weights()

    def forward(self, src, context_dict, device) -> Tensor:
        """
        Forward pass for next location prediction.
        
        Args:
            src: Input sequence tensor [seq_len, batch_size]
            context_dict: Dictionary containing context features:
                - len: Sequence lengths
                - user: User IDs
                - time: Time features
                - weekday: Weekday features
                - duration: Duration features
            device: Device to run on
        
        Returns:
            Logits tensor [batch_size, total_loc_num]
        """
        # Get embeddings [seq_len, batch_size, d_input]
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]
        batch_size = src.shape[1]

        # Pack padded sequence for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, 
            seq_len.cpu(), 
            batch_first=False, 
            enforce_sorted=False
        )
        
        # LSTM forward pass
        packed_output, (hidden, cell) = self.lstm(packed)
        
        # Unpack sequence
        output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=False)
        
        # Get the output at the last valid timestep for each sequence
        # output shape: [seq_len, batch_size, hidden_size]
        out = output.gather(
            0,
            (seq_len - 1).view([1, -1, 1]).expand([1, batch_size, self.hidden_size]).to(device)
        ).squeeze(0)

        # Pass through FC layer with user embedding
        return self.FC(out, context_dict["user"])

    def _init_weights(self):
        """Initialize parameters with Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                # Hidden-hidden weights - use orthogonal initialization
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                # Biases - initialize forget gate bias to 1 for better gradient flow
                param.data.fill_(0)
                n = param.size(0)
                # LSTM bias is [4*hidden_size], forget gate is at [hidden_size:2*hidden_size]
                param.data[n//4:n//2].fill_(1.0)
            elif param.dim() > 1:
                nn.init.xavier_uniform_(param.data)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
