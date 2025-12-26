"""
Multi-Head Self-Attention (MHSA) Model for Next Location Prediction.

This module contains the MHSA/Transformer-based model for next location prediction,
originally from location-prediction-ori-freeze. All model components are consolidated
into a single file for easy maintenance and deployment.

Components:
- PositionalEncoding: Standard sinusoidal positional encoding for transformers
- TemporalEmbedding: Embedding for time features (hour, minute, weekday)
- POINet: Neural network for POI feature processing
- AllEmbedding: Combined embedding layer for all input features
- FullyConnected: Output layer with optional residual connections
- TransEncoder (MHSA): Main transformer encoder model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding for transformer models.
    
    Args:
        emb_size (int): Embedding dimension size
        dropout (float): Dropout rate
        maxlen (int): Maximum sequence length (default: 5000)
    """
    def __init__(self, emb_size: int, dropout: float, maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(-torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("pos_embedding", pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[: token_embedding.size(0), :])


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


class AllEmbedding(nn.Module):
    """
    Combined embedding layer for all input features.
    
    Combines location embedding with optional embeddings for:
    - Time (temporal embedding)
    - Duration
    - POI features
    - Positional encoding (for transformer)
    
    Args:
        d_input (int): Base embedding dimension
        config: Configuration object with embedding settings
        total_loc_num (int): Total number of unique locations
        if_pos_encoder (bool): Whether to use positional encoding
        emb_info (str): Temporal embedding mode
        emb_type (str): How to combine embeddings ("add" or "concat")
    """
    def __init__(self, d_input, config, total_loc_num, if_pos_encoder=True, emb_info="all", emb_type="add"):
        super(AllEmbedding, self).__init__()
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

        # position encoder for transformer
        self.if_pos_encoder = if_pos_encoder
        if self.if_pos_encoder:
            self.pos_encoder = PositionalEncoding(d_input, dropout=0.1)
        else:
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

        if self.if_pos_encoder:
            return self.pos_encoder(emb * math.sqrt(self.d_input))
        else:
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


class MHSA(nn.Module):
    """
    Multi-Head Self-Attention (MHSA) Model for Next Location Prediction.
    
    This is a Transformer Encoder-based model that predicts the next location
    based on a sequence of historical locations and associated context features.
    
    Architecture:
    1. AllEmbedding: Combines location, time, duration, and optional POI embeddings
    2. TransformerEncoder: Multi-layer transformer encoder with self-attention
    3. FullyConnected: Output layer with optional user embedding and residual
    
    Args:
        config: Configuration object containing model hyperparameters
        total_loc_num (int): Total number of unique locations
    
    Config requirements:
        - base_emb_size: Base embedding dimension
        - nhead: Number of attention heads
        - dim_feedforward: FFN dimension
        - num_encoder_layers: Number of encoder layers
        - if_embed_user: Whether to embed user IDs
        - if_embed_time: Whether to embed time features
        - if_embed_duration: Whether to embed duration
        - if_embed_poi: Whether to embed POI features
        - fc_dropout: Dropout rate for FC layer
        - total_user_num: Total number of users
    """
    def __init__(self, config, total_loc_num) -> None:
        super(MHSA, self).__init__()

        self.d_input = config.base_emb_size
        self.Embedding = AllEmbedding(self.d_input, config, total_loc_num)

        # encoder
        encoder_layer = torch.nn.TransformerEncoderLayer(
            self.d_input, nhead=config.nhead, activation="gelu", dim_feedforward=config.dim_feedforward
        )
        encoder_norm = torch.nn.LayerNorm(self.d_input)
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
        )

        self.FC = FullyConnected(self.d_input, config, if_residual_layer=True, total_loc_num=total_loc_num)

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
        emb = self.Embedding(src, context_dict)
        seq_len = context_dict["len"]

        # positional encoding, dropout performed inside
        src_mask = self._generate_square_subsequent_mask(src.shape[0]).to(device)
        src_padding_mask = (src == 0).transpose(0, 1).to(device)
        out = self.encoder(emb, mask=src_mask, src_key_padding_mask=src_padding_mask)

        # only take the last timestep
        out = out.gather(
            0,
            seq_len.view([1, -1, 1]).expand([1, out.shape[1], out.shape[-1]]) - 1,
        ).squeeze(0)

        return self.FC(out, context_dict["user"])

    def _generate_square_subsequent_mask(self, sz):
        """Generate causal attention mask."""
        return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)

    def _init_weights(self):
        """Initiate parameters with Xavier uniform initialization."""
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @torch.no_grad()
    def get_attention_maps(self, x, context_dict, device):
        """
        Extract attention maps from all encoder layers.
        
        Useful for model interpretation and visualization.
        
        Args:
            x: Input sequence tensor
            context_dict: Context dictionary
            device: Device to run on
        
        Returns:
            List of attention maps, one per encoder layer
        """
        emb = self.Embedding(x, context_dict)
        seq_len = context_dict["len"]

        src_mask = self._generate_square_subsequent_mask(x.shape[0]).to(device)
        src_padding_mask = (x == 0).transpose(0, 1).to(device)

        attention_maps = []
        for layer in self.encoder.layers:
            _, attn_map = layer.self_attn(
                emb, emb, emb, attn_mask=src_mask, key_padding_mask=src_padding_mask, need_weights=True
            )
            attn_map = attn_map.gather(
                1, seq_len.view([-1, 1, 1]).expand([attn_map.shape[0], 1, attn_map.shape[-1]]) - 1
            ).squeeze(1)
            attention_maps.append(attn_map)
            emb = layer(emb)

        return attention_maps


# Alias for backward compatibility
TransEncoder = MHSA
