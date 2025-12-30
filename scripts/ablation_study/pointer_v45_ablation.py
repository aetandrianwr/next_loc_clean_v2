"""
Ablation Model for Pointer Network V45.

This module implements a modified PointerNetworkV45 with configurable
component ablation for systematic evaluation of individual components.

Ablation Components:
1. Location Embedding - Core location representation
2. User Embedding - User-specific personalization
3. Time Embedding - Time-of-day awareness
4. Weekday Embedding - Day-of-week patterns
5. Recency Embedding - Temporal decay awareness
6. Duration Embedding - Visit duration features
7. Position-from-End Embedding - Sequence position awareness
8. Sinusoidal Positional Encoding - Absolute position information
9. Pointer Mechanism - Copy mechanism from input sequence
10. Generation Head - Full vocabulary prediction
11. Pointer-Generation Gate - Adaptive blending mechanism
12. Transformer Encoder - Multi-layer attention (ablate layers)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class PointerNetworkV45Ablation(nn.Module):
    """
    Pointer Network V45 with Ablation Support.
    
    Each component can be selectively disabled to measure its contribution
    to the overall model performance.
    
    Args:
        num_locations (int): Total number of unique locations
        num_users (int): Total number of unique users
        d_model (int): Model dimension (default: 128)
        nhead (int): Number of attention heads (default: 4)
        num_layers (int): Number of transformer layers (default: 3)
        dim_feedforward (int): FFN dimension (default: 256)
        dropout (float): Dropout rate (default: 0.15)
        max_seq_len (int): Maximum sequence length (default: 150)
        ablation_config (dict): Dictionary specifying which components to ablate
    
    Ablation Config Keys:
        - use_user_emb: bool (default: True)
        - use_time_emb: bool (default: True)
        - use_weekday_emb: bool (default: True)
        - use_recency_emb: bool (default: True)
        - use_duration_emb: bool (default: True)
        - use_pos_from_end_emb: bool (default: True)
        - use_sinusoidal_pos: bool (default: True)
        - use_pointer: bool (default: True)
        - use_generation: bool (default: True)
        - use_gate: bool (default: True)
        - num_transformer_layers: int (override num_layers if specified)
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
        ablation_config: Optional[Dict] = None,
    ):
        super().__init__()
        
        # Parse ablation config
        self.ablation_config = ablation_config or {}
        self.use_user_emb = self.ablation_config.get('use_user_emb', True)
        self.use_time_emb = self.ablation_config.get('use_time_emb', True)
        self.use_weekday_emb = self.ablation_config.get('use_weekday_emb', True)
        self.use_recency_emb = self.ablation_config.get('use_recency_emb', True)
        self.use_duration_emb = self.ablation_config.get('use_duration_emb', True)
        self.use_pos_from_end_emb = self.ablation_config.get('use_pos_from_end_emb', True)
        self.use_sinusoidal_pos = self.ablation_config.get('use_sinusoidal_pos', True)
        self.use_pointer = self.ablation_config.get('use_pointer', True)
        self.use_generation = self.ablation_config.get('use_generation', True)
        self.use_gate = self.ablation_config.get('use_gate', True)
        
        # Override num_layers if specified in ablation config
        effective_num_layers = self.ablation_config.get('num_transformer_layers', num_layers)
        
        self.num_locations = num_locations
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        
        # Core location embedding (always required)
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # User embedding (ablatable)
        if self.use_user_emb:
            self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
        
        # Temporal embeddings (ablatable)
        temporal_dim = d_model // 4
        if self.use_time_emb:
            self.time_emb = nn.Embedding(97, temporal_dim)
        if self.use_weekday_emb:
            self.weekday_emb = nn.Embedding(8, temporal_dim)
        if self.use_recency_emb:
            self.recency_emb = nn.Embedding(9, temporal_dim)
        if self.use_duration_emb:
            self.duration_emb = nn.Embedding(100, temporal_dim)
        
        # Position from end embedding (ablatable)
        if self.use_pos_from_end_emb:
            self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, temporal_dim)
        
        # Calculate input dimension based on active components
        input_dim = d_model  # Location embedding always included
        if self.use_user_emb:
            input_dim += d_model
        if self.use_time_emb:
            input_dim += temporal_dim
        if self.use_weekday_emb:
            input_dim += temporal_dim
        if self.use_recency_emb:
            input_dim += temporal_dim
        if self.use_duration_emb:
            input_dim += temporal_dim
        if self.use_pos_from_end_emb:
            input_dim += temporal_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Sinusoidal positional encoding (ablatable)
        if self.use_sinusoidal_pos:
            self.register_buffer('pos_encoding', self._create_pos_encoding(max_seq_len, d_model))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, effective_num_layers)
        
        # Pointer mechanism (ablatable)
        if self.use_pointer:
            self.pointer_query = nn.Linear(d_model, d_model)
            self.pointer_key = nn.Linear(d_model, d_model)
            self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
        
        # Generation head (ablatable)
        if self.use_generation:
            self.gen_head = nn.Linear(d_model, num_locations)
        
        # Pointer-Generation gate (ablatable)
        if self.use_gate and self.use_pointer and self.use_generation:
            self.ptr_gen_gate = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, 1),
                nn.Sigmoid()
            )
        
        self._init_weights()
    
    def _create_pos_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
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
        Forward pass with ablation support.
        
        Args:
            x: Location sequence [seq_len, batch_size]
            x_dict: Dictionary with temporal and user features
            
        Returns:
            Log probabilities [batch_size, num_locations]
        """
        x = x.T  # [batch_size, seq_len]
        batch_size, seq_len = x.shape
        device = x.device
        lengths = x_dict['len']
        
        # Build feature list
        features = []
        
        # Location embedding (always included)
        loc_emb = self.loc_emb(x)
        features.append(loc_emb)
        
        # User embedding
        if self.use_user_emb:
            user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
            features.append(user_emb)
        
        # Temporal features
        time = torch.clamp(x_dict['time'].T, 0, 96)
        weekday = torch.clamp(x_dict['weekday'].T, 0, 7)
        recency = torch.clamp(x_dict['diff'].T, 0, 8)
        duration = torch.clamp(x_dict['duration'].T, 0, 99)
        
        if self.use_time_emb:
            features.append(self.time_emb(time))
        if self.use_weekday_emb:
            features.append(self.weekday_emb(weekday))
        if self.use_recency_emb:
            features.append(self.recency_emb(recency))
        if self.use_duration_emb:
            features.append(self.duration_emb(duration))
        
        # Position from end
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
        
        if self.use_pos_from_end_emb:
            features.append(self.pos_from_end_emb(pos_from_end))
        
        # Combine features
        combined = torch.cat(features, dim=-1)
        hidden = self.input_norm(self.input_proj(combined))
        
        # Add sinusoidal positional encoding
        if self.use_sinusoidal_pos:
            hidden = hidden + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        mask = positions >= lengths.unsqueeze(1)
        encoded = self.transformer(hidden, src_key_padding_mask=mask)
        
        # Extract context from last valid position
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]
        
        # Compute pointer distribution
        ptr_dist = None
        if self.use_pointer:
            query = self.pointer_query(context).unsqueeze(1)
            keys = self.pointer_key(encoded)
            ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
            ptr_scores = ptr_scores + self.position_bias[pos_from_end]
            ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
            ptr_probs = F.softmax(ptr_scores, dim=-1)
            
            ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
            ptr_dist.scatter_add_(1, x, ptr_probs)
        
        # Compute generation distribution
        gen_probs = None
        if self.use_generation:
            gen_probs = F.softmax(self.gen_head(context), dim=-1)
        
        # Combine distributions
        if self.use_pointer and self.use_generation:
            if self.use_gate:
                gate = self.ptr_gen_gate(context)
                final_probs = gate * ptr_dist + (1 - gate) * gen_probs
            else:
                # Fixed 50-50 blend without gate
                final_probs = 0.5 * ptr_dist + 0.5 * gen_probs
        elif self.use_pointer:
            final_probs = ptr_dist
        elif self.use_generation:
            final_probs = gen_probs
        else:
            raise ValueError("At least one of pointer or generation must be enabled")
        
        return torch.log(final_probs + 1e-10)
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_ablation_summary(self) -> str:
        """Return a summary of ablated components."""
        summary = []
        if not self.use_user_emb:
            summary.append("w/o User Embedding")
        if not self.use_time_emb:
            summary.append("w/o Time Embedding")
        if not self.use_weekday_emb:
            summary.append("w/o Weekday Embedding")
        if not self.use_recency_emb:
            summary.append("w/o Recency Embedding")
        if not self.use_duration_emb:
            summary.append("w/o Duration Embedding")
        if not self.use_pos_from_end_emb:
            summary.append("w/o Position-from-End Embedding")
        if not self.use_sinusoidal_pos:
            summary.append("w/o Sinusoidal Positional Encoding")
        if not self.use_pointer:
            summary.append("w/o Pointer Mechanism")
        if not self.use_generation:
            summary.append("w/o Generation Head")
        if not self.use_gate:
            summary.append("w/o Pointer-Generation Gate")
        
        if not summary:
            return "Full Model (No Ablation)"
        return ", ".join(summary)
