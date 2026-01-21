"""
Position-Aware Pointer Generator Transformer - Ablation Study Variant.

This module implements ablation variants of the PointerGeneratorTransformer model for 
systematic component analysis. Each ablation variant allows selective disabling
of specific model components to quantify their contribution to performance.

Ablation Variants:
1. Full Model (baseline) - All components enabled
2. No Pointer Mechanism - Only generation head (no copy mechanism)
3. No Generation Head - Only pointer mechanism (copy only)
4. No Position Bias - Pointer without position bias
5. No Temporal Embeddings - Remove time/weekday/duration/recency
6. No User Embedding - Remove user personalization
7. No Position-from-End - Remove position-from-end embeddings
8. Single Transformer Layer - Reduce depth
9. No Pointer-Gen Gate (Fixed 0.5) - Remove adaptive gating

Usage:
    from pgt_ablation import PointerGeneratorTransformerAblation
    
    model = PointerGeneratorTransformerAblation(
        num_locations=1000,
        num_users=100,
        ablation_type='no_pointer',
        ...
    )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PointerGeneratorTransformerAblation(nn.Module):
    """
    Ablation variant of PointerGeneratorTransformer for systematic component analysis.
    
    This model supports various ablation configurations to evaluate the
    contribution of each component to the overall model performance.
    
    Args:
        num_locations (int): Total number of unique locations
        num_users (int): Total number of unique users
        d_model (int): Model embedding dimension
        nhead (int): Number of attention heads
        num_layers (int): Number of transformer layers
        dim_feedforward (int): FFN dimension
        dropout (float): Dropout probability
        max_seq_len (int): Maximum sequence length
        ablation_type (str): Type of ablation:
            - 'full': Full model (baseline)
            - 'no_pointer': Remove pointer mechanism
            - 'no_generation': Remove generation head
            - 'no_position_bias': Remove position bias in pointer
            - 'no_temporal': Remove temporal embeddings
            - 'no_user': Remove user embeddings
            - 'no_pos_from_end': Remove position-from-end embeddings
            - 'single_layer': Use single transformer layer
            - 'no_gate': Fixed 0.5 gate (no adaptive blending)
    """
    
    VALID_ABLATIONS = [
        'full',
        'no_pointer',
        'no_generation',
        'no_position_bias',
        'no_temporal',
        'no_user',
        'no_pos_from_end',
        'single_layer',
        'no_gate',
    ]
    
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
        ablation_type: str = 'full',
    ):
        super().__init__()
        
        if ablation_type not in self.VALID_ABLATIONS:
            raise ValueError(f"Invalid ablation type: {ablation_type}. "
                           f"Valid types: {self.VALID_ABLATIONS}")
        
        self.num_locations = num_locations
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.ablation_type = ablation_type
        
        # Adjust num_layers for single_layer ablation
        actual_num_layers = 1 if ablation_type == 'single_layer' else num_layers
        
        # Core embeddings
        self.loc_emb = nn.Embedding(num_locations, d_model, padding_idx=0)
        
        # User embedding (conditionally used)
        self.use_user = ablation_type != 'no_user'
        if self.use_user:
            self.user_emb = nn.Embedding(num_users, d_model, padding_idx=0)
        
        # Temporal embeddings (conditionally used)
        self.use_temporal = ablation_type != 'no_temporal'
        if self.use_temporal:
            self.time_emb = nn.Embedding(97, d_model // 4)
            self.weekday_emb = nn.Embedding(8, d_model // 4)
            self.recency_emb = nn.Embedding(9, d_model // 4)
            self.duration_emb = nn.Embedding(100, d_model // 4)
        
        # Position from end embedding (conditionally used)
        self.use_pos_from_end = ablation_type != 'no_pos_from_end'
        if self.use_pos_from_end:
            self.pos_from_end_emb = nn.Embedding(max_seq_len + 1, d_model // 4)
        
        # Calculate input dimension based on ablation
        input_dim = d_model  # Location embedding
        if self.use_user:
            input_dim += d_model  # User embedding
        if self.use_temporal:
            input_dim += d_model // 4 * 4  # 4 temporal features
        if self.use_pos_from_end:
            input_dim += d_model // 4  # Position from end
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.input_norm = nn.LayerNorm(d_model)
        
        # Sinusoidal positional encoding
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
        self.transformer = nn.TransformerEncoder(encoder_layer, actual_num_layers)
        
        # Pointer mechanism (conditionally used)
        self.use_pointer = ablation_type not in ['no_pointer']
        if self.use_pointer:
            self.pointer_query = nn.Linear(d_model, d_model)
            self.pointer_key = nn.Linear(d_model, d_model)
            
            # Position bias (conditionally used)
            self.use_position_bias = ablation_type != 'no_position_bias'
            if self.use_position_bias:
                self.position_bias = nn.Parameter(torch.zeros(max_seq_len))
        
        # Generation head (conditionally used)
        self.use_generation = ablation_type not in ['no_generation']
        if self.use_generation:
            self.gen_head = nn.Linear(d_model, num_locations)
        
        # Pointer-Generation gate (conditionally used)
        self.use_gate = ablation_type != 'no_gate'
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
        Forward pass with ablation-specific behavior.
        
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
        
        # Build embeddings based on ablation configuration
        embeddings = [self.loc_emb(x)]
        
        if self.use_user:
            user_emb = self.user_emb(x_dict['user']).unsqueeze(1).expand(-1, seq_len, -1)
            embeddings.append(user_emb)
        
        if self.use_temporal:
            time = torch.clamp(x_dict['time'].T, 0, 96)
            weekday = torch.clamp(x_dict['weekday'].T, 0, 7)
            recency = torch.clamp(x_dict['diff'].T, 0, 8)
            duration = torch.clamp(x_dict['duration'].T, 0, 99)
            
            temporal = torch.cat([
                self.time_emb(time),
                self.weekday_emb(weekday),
                self.recency_emb(recency),
                self.duration_emb(duration)
            ], dim=-1)
            embeddings.append(temporal)
        
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, self.max_seq_len - 1)
        
        if self.use_pos_from_end:
            pos_emb = self.pos_from_end_emb(pos_from_end)
            embeddings.append(pos_emb)
        
        # Combine and project
        combined = torch.cat(embeddings, dim=-1)
        hidden = self.input_norm(self.input_proj(combined))
        hidden = hidden + self.pos_encoding[:, :seq_len, :]
        
        # Transformer encoding
        mask = positions >= lengths.unsqueeze(1)
        encoded = self.transformer(hidden, src_key_padding_mask=mask)
        
        # Extract context
        batch_idx = torch.arange(batch_size, device=device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]
        
        # Compute distributions based on ablation
        if self.use_pointer and self.use_generation:
            # Both pointer and generation
            ptr_dist = self._compute_pointer_dist(x, encoded, context, mask, pos_from_end)
            gen_probs = F.softmax(self.gen_head(context), dim=-1)
            
            if self.use_gate:
                gate = self.ptr_gen_gate(context)
            else:
                gate = torch.full((batch_size, 1), 0.5, device=device)
            
            final_probs = gate * ptr_dist + (1 - gate) * gen_probs
            
        elif self.use_pointer:
            # Pointer only (no generation)
            final_probs = self._compute_pointer_dist(x, encoded, context, mask, pos_from_end)
            
        else:
            # Generation only (no pointer)
            final_probs = F.softmax(self.gen_head(context), dim=-1)
        
        return torch.log(final_probs + 1e-10)
    
    def _compute_pointer_dist(self, x, encoded, context, mask, pos_from_end):
        """Compute pointer distribution over vocabulary."""
        batch_size = x.shape[0]
        device = x.device
        
        query = self.pointer_query(context).unsqueeze(1)
        keys = self.pointer_key(encoded)
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(self.d_model)
        
        if self.use_position_bias:
            ptr_scores = ptr_scores + self.position_bias[pos_from_end]
        
        ptr_scores = ptr_scores.masked_fill(mask, float('-inf'))
        ptr_probs = F.softmax(ptr_scores, dim=-1)
        
        ptr_dist = torch.zeros(batch_size, self.num_locations, device=device)
        ptr_dist.scatter_add_(1, x, ptr_probs)
        
        return ptr_dist
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
