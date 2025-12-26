"""
LSTM Model for Next Location Prediction.

This module contains the LSTM-based model for next location prediction,
adapted from the MHSA model by replacing the Transformer Encoder with LSTM layers.

Components:
- TemporalEmbedding: Embedding for time features (hour, minute, weekday)
- POINet: Neural network for POI feature processing
- AllEmbedding: Combined embedding layer for all input features (without positional encoding)
- FullyConnected: Output layer with optional residual connections
- LSTMModel: Main LSTM-based model

The LSTM replaces the Transformer Encoder with causal masking and sinusoidal position
embedding. This serves as a baseline model for comparison.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import math


class TemporalEmbedding(nn.Module):
    """
    Temporal embedding for time-related features.
    
    Supports three modes:
    - "all": Separate embeddings for minute (quarter-hour), hour, and weekday
    - "time": Single embedding for time slots
    - "weekday": Only weekday embedding
    
    Args:
        d_input (int): Embedding dimension
        emb_info (str): Embedding mode ("all", "time", "weekday")
    """
    def __init__(self, d_input, emb_info="all"):
        super(TemporalEmbedding, self).__init__()

        self.emb_info = emb_info
        # quarter of an hour
        self.minute_size = 4
        hour_size = 24
        weekday = 7

        if self.emb_info == "all":
            self.minute_embed = nn.Embedding(self.minute_size, d_input)
            self.hour_embed = nn.Embedding(hour_size, d_input)
            self.weekday_embed = nn.Embedding(weekday, d_input)
        elif self.emb_info == "time":
            self.time_embed = nn.Embedding(self.minute_size * hour_size, d_input)
        elif self.emb_info == "weekday":
            self.weekday_embed = nn.Embedding(weekday, d_input)

    def forward(self, time, weekday):
        if self.emb_info == "all":
            hour = torch.div(time, self.minute_size, rounding_mode="floor")
            minutes = time % 4

            minute_x = self.minute_embed(minutes)
            hour_x = self.hour_embed(hour)
            weekday_x = self.weekday_embed(weekday)

            return hour_x + minute_x + weekday_x
        elif self.emb_info == "time":
            return self.time_embed(time)
        elif self.emb_info == "weekday":
            return self.weekday_embed(weekday)


class POINet(nn.Module):
    """
    Neural network for processing POI (Point of Interest) features.
    
    Processes POI vectors through a series of transformations with 
    residual connections and layer normalization.
    
    Args:
        poi_vector_size (int): Size of POI feature vectors
        out (int): Output dimension
    """
    def __init__(self, poi_vector_size, out):
        super(POINet, self).__init__()

        self.buffer_num = 11

        # 11 -> poi_vector_size*2 -> 11
        if self.buffer_num == 11:
            self.linear1 = torch.nn.Linear(self.buffer_num, poi_vector_size * 2)
            self.linear2 = torch.nn.Linear(poi_vector_size * 2, self.buffer_num)
            self.dropout2 = nn.Dropout(p=0.1)
            self.norm1 = nn.LayerNorm(self.buffer_num)

            # 11*poi_vector_size -> poi_vector_size
            self.dense = torch.nn.Linear(self.buffer_num * poi_vector_size, poi_vector_size)
            self.dropout_dense = nn.Dropout(p=0.1)

        # poi_vector_size -> poi_vector_size*4 -> poi_vector_size
        self.linear3 = torch.nn.Linear(poi_vector_size, poi_vector_size * 4)
        self.linear4 = torch.nn.Linear(poi_vector_size * 4, poi_vector_size)
        self.dropout3 = nn.Dropout(p=0.1)
        self.dropout4 = nn.Dropout(p=0.1)
        self.norm2 = nn.LayerNorm(poi_vector_size)

        # poi_vector_size -> out
        self.fc = nn.Linear(poi_vector_size, out)

    def forward(self, x):
        # first
        if self.buffer_num == 11:
            x = self.norm1(x + self._ff_block(x))
        # flat
        x = x.view([x.shape[0], x.shape[1], x.shape[2] * x.shape[3]])
        if self.buffer_num == 11:
            x = self.dropout_dense(F.relu(self.dense(x)))
        # second
        x = self.norm2(x + self._dense_block(x))
        return self.fc(x)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(F.relu(self.linear1(x)))
        return self.dropout2(x)

    def _dense_block(self, x: Tensor) -> Tensor:
        x = self.linear4(self.dropout3(F.relu(self.linear3(x))))
        return self.dropout4(x)


class AllEmbeddingLSTM(nn.Module):
    """
    Combined embedding layer for all input features (LSTM version without positional encoding).
    
    Combines location embedding with optional embeddings for:
    - Time (temporal embedding)
    - Duration
    - POI features
    
    Note: Unlike the Transformer version, this does NOT use positional encoding
    since LSTM inherently captures sequence order.
    
    Args:
        d_input (int): Base embedding dimension
        config: Configuration object with embedding settings
        total_loc_num (int): Total number of unique locations
        emb_info (str): Temporal embedding mode
        emb_type (str): How to combine embeddings ("add" or "concat")
    """
    def __init__(self, d_input, config, total_loc_num, emb_info="all", emb_type="add"):
        super(AllEmbeddingLSTM, self).__init__()
        self.d_input = d_input
        self.emb_type = emb_type

        # location embedding
        if self.emb_type == "add":
            self.emb_loc = nn.Embedding(total_loc_num, d_input)
        else:
            self.emb_loc = nn.Embedding(total_loc_num, d_input - config.time_emb_size)

        # time embedding
        self.if_include_time = config.if_embed_time
        if self.if_include_time:
            if self.emb_type == "add":
                self.temporal_embedding = TemporalEmbedding(d_input, emb_info)
            else:
                self.temporal_embedding = TemporalEmbedding(config.time_emb_size, emb_info)

        # duration embedding (in minutes, max 2 days)
        self.if_include_duration = config.if_embed_duration
        if self.if_include_duration:
            self.emb_duration = nn.Embedding(60 * 24 * 2 // 30, d_input)

        # POI embedding
        self.if_include_poi = config.if_embed_poi
        if self.if_include_poi:
            self.poi_net = POINet(config.poi_original_size, d_input)

        # Dropout (no positional encoding for LSTM)
        self.dropout = nn.Dropout(0.1)

    def forward(self, src, context_dict) -> Tensor:
        emb = self.emb_loc(src)

        if self.if_include_time:
            if self.emb_type == "add":
                emb = emb + self.temporal_embedding(context_dict["time"], context_dict["weekday"])
            else:
                emb = torch.cat([emb, self.temporal_embedding(context_dict["time"], context_dict["weekday"])], dim=-1)

        if self.if_include_duration:
            emb = emb + self.emb_duration(context_dict["duration"])

        if self.if_include_poi:
            emb = emb + self.poi_net(context_dict["poi"])

        return self.dropout(emb)


class FullyConnected(nn.Module):
    """
    Output layer with optional user embedding and residual connections.
    
    Args:
        d_input (int): Input dimension
        config: Configuration object
        total_loc_num (int): Total number of locations (output dimension)
        if_residual_layer (bool): Whether to use residual layer
    """
    def __init__(self, d_input, config, total_loc_num, if_residual_layer=True):
        super(FullyConnected, self).__init__()

        fc_dim = d_input
        self.if_embed_user = config.if_embed_user
        if self.if_embed_user:
            self.emb_user = nn.Embedding(config.total_user_num, fc_dim)

        self.fc_loc = nn.Linear(fc_dim, total_loc_num)
        self.emb_dropout = nn.Dropout(p=0.1)

        self.if_residual_layer = if_residual_layer
        if self.if_residual_layer:
            self.linear1 = nn.Linear(fc_dim, fc_dim * 2)
            self.linear2 = nn.Linear(fc_dim * 2, fc_dim)

            self.norm1 = nn.BatchNorm1d(fc_dim)
            self.fc_dropout1 = nn.Dropout(p=config.fc_dropout)
            self.fc_dropout2 = nn.Dropout(p=config.fc_dropout)

    def forward(self, out, user) -> Tensor:
        if self.if_embed_user:
            out = out + self.emb_user(user)

        out = self.emb_dropout(out)

        if self.if_residual_layer:
            out = self.norm1(out + self._res_block(out))

        return self.fc_loc(out)

    def _res_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.fc_dropout1(F.relu(self.linear1(x))))
        return self.fc_dropout2(x)


class LSTMModel(nn.Module):
    """
    LSTM Model for Next Location Prediction.
    
    This model replaces the Transformer Encoder (with causal masking and sinusoidal
    positional embedding) from MHSA with LSTM layers. The LSTM naturally handles
    sequential dependencies without requiring explicit causal masking.
    
    Architecture:
    1. AllEmbeddingLSTM: Combines location, time, duration, and optional POI embeddings
    2. LSTM: Multi-layer LSTM for sequence processing
    3. FullyConnected: Output layer with optional user embedding and residual
    
    Args:
        config: Configuration object containing model hyperparameters
        total_loc_num (int): Total number of unique locations
    
    Config requirements:
        - base_emb_size: Base embedding dimension
        - lstm_hidden_size: LSTM hidden dimension
        - lstm_num_layers: Number of LSTM layers
        - lstm_dropout: Dropout between LSTM layers
        - if_embed_user: Whether to embed user IDs
        - if_embed_time: Whether to embed time features
        - if_embed_duration: Whether to embed duration
        - if_embed_poi: Whether to embed POI features
        - fc_dropout: Dropout rate for FC layer
        - total_user_num: Total number of users
    """
    def __init__(self, config, total_loc_num) -> None:
        super(LSTMModel, self).__init__()

        self.d_input = config.base_emb_size
        self.hidden_size = config.lstm_hidden_size
        self.num_layers = config.lstm_num_layers
        
        # Embedding layer (without positional encoding)
        self.Embedding = AllEmbeddingLSTM(self.d_input, config, total_loc_num)

        # LSTM encoder
        lstm_dropout = config.lstm_dropout if self.num_layers > 1 else 0
        self.lstm = nn.LSTM(
            input_size=self.d_input,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=lstm_dropout,
            batch_first=False,  # Input format: [seq_len, batch, features]
            bidirectional=False  # Unidirectional for causal/autoregressive behavior
        )
        
        # Layer norm after LSTM
        self.layer_norm = nn.LayerNorm(self.hidden_size)
        
        # Fully connected output layer
        self.FC = FullyConnected(self.hidden_size, config, if_residual_layer=True, total_loc_num=total_loc_num)

        # init parameters
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
                - poi (optional): POI features
            device: Device to run on
        
        Returns:
            Logits tensor [batch_size, total_loc_num]
        """
        # Get embeddings
        emb = self.Embedding(src, context_dict)  # [seq_len, batch, d_input]
        seq_len = context_dict["len"]
        
        # Pack padded sequence for efficient LSTM processing
        packed_emb = pack_padded_sequence(
            emb, 
            seq_len.cpu(), 
            batch_first=False, 
            enforce_sorted=False
        )
        
        # Pass through LSTM
        packed_output, (h_n, c_n) = self.lstm(packed_emb)
        
        # Unpack sequence
        output, _ = pad_packed_sequence(packed_output, batch_first=False)
        
        # Get the last valid output for each sequence
        # output shape: [seq_len, batch, hidden_size]
        out = output.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, output.shape[1], output.shape[-1]]).to(device) - 1,
        ).squeeze(0)
        
        # Apply layer normalization
        out = self.layer_norm(out)

        return self.FC(out, context_dict["user"])

    def _init_weights(self):
        """Initialize parameters."""
        for name, param in self.named_parameters():
            if 'weight_ih' in name:
                # Input-hidden weights: Xavier uniform
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                # Hidden-hidden weights: Orthogonal initialization
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                # Biases: Zero initialization with forget gate bias set to 1
                nn.init.zeros_(param)
                # Set forget gate bias to 1 for better gradient flow
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)
            elif param.dim() > 1:
                # Other parameters: Xavier uniform
                nn.init.xavier_uniform_(param)
