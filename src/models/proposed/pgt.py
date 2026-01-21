"""
Pointer Generator Transformer - Clean and Lean Version.

This module implements the PointerGeneratorTransformer model for next location prediction.
The model combines a Transformer encoder with a pointer mechanism and a generation
head, using a learned gate to blend the two prediction strategies.

Architecture:
- Location + User + Temporal embeddings
- Transformer encoder with pre-norm and GELU activation
- Pointer mechanism with position bias
- Generation head with full vocabulary prediction
- Pointer-Generation gate for adaptive blending

Key Features:
- Sinusoidal positional encoding
- Position-from-end embedding for recency awareness
- Temporal features: time of day, weekday, recency, duration
- Mixed precision training support

Usage:
    from src.models.proposed.pgt import PointerGeneratorTransformer

    model = PointerGeneratorTransformer(
        num_locations=1000,
        num_users=100,
        d_model=128,
        nhead=4,
        num_layers=3,
        dim_feedforward=256,
        dropout=0.15,
    )

    # Forward pass
    # x: [seq_len, batch_size] - location sequence
    # x_dict: dictionary with 'user', 'time', 'weekday', 'diff', 'duration', 'len'
    log_probs = model(x, x_dict)  # [batch_size, num_locations]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGeneratorTransformer(nn.Module):
    """
    Pointer Generator Transformer for Next Location Prediction.

    This model predicts the next location a user will visit based on their
    location history and temporal context. It uses a hybrid approach combining:

    1. Pointer Mechanism: Attends to input sequence and copies from history
    2. Generation Head: Generates prediction over full vocabulary
    3. Adaptive Gate: Learns to blend pointer and generation distributions

    Args:
        num_locations (int): Total number of unique locations in vocabulary
        num_users (int): Total number of unique users
        d_model (int): Dimension of model embeddings (default: 128)
        nhead (int): Number of attention heads (default: 4)
        num_layers (int): Number of transformer encoder layers (default: 3)
        dim_feedforward (int): Feedforward network dimension (default: 256)
        dropout (float): Dropout probability (default: 0.15)
        max_seq_len (int): Maximum sequence length (default: 150)

    Input:
        x (torch.Tensor): Location sequence tensor of shape [seq_len, batch_size]
        x_dict (dict): Dictionary containing:
            - 'user': User IDs [batch_size]
            - 'time': Time of day in 15-min intervals [seq_len, batch_size]
            - 'weekday': Day of week [seq_len, batch_size]
            - 'diff': Days ago for each visit [seq_len, batch_size]
            - 'duration': Duration in 30-min buckets [seq_len, batch_size]
            - 'len': Sequence lengths [batch_size]

    Output:
        torch.Tensor: Log probabilities over locations [batch_size, num_locations]
    """

    def __init__(
        self,
        num_locations: int,
        num_users: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.15,
        max_seq_len: int = 150,
    ):
        super().__init__()

        self.num_locations = num_locations
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Core embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)

        # Temporal embeddings (time, weekday, recency, duration)
        self.time_emb = nn.Embedding(97, d_model // 4)  # 96 intervals + 1 padding
        self.weekday_emb = nn.Embedding(8, d_model // 4)  # 7 days + 1 padding
        self.recency_emb = nn.Embedding(9, d_model // 4)  # 8 recency levels + 1 padding
        self.duration_emb = nn.Embedding(100, d_model // 4)  # 100 duration buckets

        # Position from end embedding (important for pointer mechanism)
        self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)

        # Input projection: loc + user + 5 temporal features
        # Location (d_model) + User (d_model) + 5 temporal features (d_model // 4 each)
        input_dim = d_model * 2 + d_model // 4 * 5
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)

        # Sinusoidal positional encoding
        self.register_buffer(
            "pos_encoding", self._create_pos_encoding(max_seq_len, d_model)
        )

        # Transformer encoder with pre-norm
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # Pointer mechanism
        self.pointer_query = nn.Linear(d_model, d_model)
        self.pointer_key = nn.Linear(d_model, d_model)
        self.position_bias = nn.Parameter(torch.zeros(max_seq_len))

        # Generation head
        self.gen_head = nn.Linear(d_model, num_locations)

        # Pointer-Generation gate
        self.ptr_gen_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.

        Uses the standard positional encoding formula from "Attention is All You Need":
        PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

        Args:
            max_len: Maximum sequence length
            d_model: Model dimension

        Returns:
            Positional encoding tensor [1, max_len, d_model]
        """
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)

    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x: torch.Tensor, x_dict: dict) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Location sequence [seq_len, batch_size]
            x_dict: Dictionary with 'user', 'time', 'weekday', 'diff', 'duration', 'len'

        Returns:
            Log probabilities [batch_size, num_locations]
        """
        x = x.T  # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        device = x.device
        lengths = x_dict["len"]

        # Embeddings
        loc_emb = self.loc_emb(x)
        user_emb = self.user_emb(x_dict["user"]).unsqueeze(1).expand(-1, seq_len, -1)

        # Temporal features (clamped to valid ranges)
        time = torch.clamp(x_dict["time"].T, 0, 96)
        weekday = torch.clamp(x_dict["weekday"].T, 0, 7)
        recency = torch.clamp(x_dict["diff"].T, 0, 8)
        duration = torch.clamp(x_dict["duration"].T, 0, 99)

        temporal = torch.cat(
            [
                self.time_emb(time),
                self.weekday_emb(weekday),
                self.recency_emb(recency),
                self.duration_emb(duration),
            ],
            dim=-1,
        )

        # Position from end
        positions = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )
        pos_from_end = torch.clamp(
            lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1
        )
        pos_emb = self.pos_from_end_emb(pos_from_end)

        # Combine features
        combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
        hidden = self.input_norm(self.input_proj(combined))
        hidden = hidden + self.pos_encoding[:, :seq_len, :]

        # Transformer encoding
        mask = positions >= lengths.unsqueeze(1)
        encoded = self.transformer(hidden, src_key_padding_mask=mask)

        # Extract context from last valid position
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]

        # Pointer attention
        query = self.pointer_query(context).unsqueeze(1)
        keys = self.pointer_key(encoded)
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(
            self.d_model
        )
        ptr_scores = ptr_scores + self.position_bias[pos_from_end]
        ptr_scores = ptr_scores.masked_fill(mask, float("-inf"))
        ptr_probs = F.softmax(ptr_scores, dim=-1)

        # Scatter pointer probabilities to location vocabulary
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
        ptr_dist.scatter_add_(1, x, ptr_probs)

        # Generation distribution
        gen_probs = F.softmax(self.gen_head(context), dim=-1)

        # Gate and combine
        gate = self.ptr_gen_gate(context)
        final_probs = gate * ptr_dist + (1 - gate) * gen_probs

        return torch.log(final_probs + 1e-10)

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
