"""
Attention Extraction Module for PointerGeneratorTransformer.

This module provides comprehensive attention extraction capabilities for
the Pointer Generator Transformer model. It extracts multiple types of attention:

1. **Transformer Self-Attention**: Captures relationships between positions
   in the input sequence through multi-head self-attention.

2. **Pointer Attention**: The core mechanism that determines which historical
   locations to "copy" for the next prediction.

3. **Pointer-Generation Gate**: A scalar gate that balances between the
   pointer mechanism (copying from history) and generation (predicting from
   full vocabulary).

Scientific Significance:
- Self-attention reveals temporal dependencies in location sequences
- Pointer attention shows which historical visits influence predictions
- Gate values indicate when the model relies on repetitive vs. novel behavior

Author: PhD Thesis Experiment
Date: 2026
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class AttentionExtractor:
    """
    Extracts and processes attention weights from PointerGeneratorTransformer.
    
    This class hooks into the model to capture:
    - Multi-head self-attention weights from transformer layers
    - Pointer attention scores
    - Pointer-generation gate values
    
    Methods:
        extract_attention: Main method to extract all attention components
        register_hooks: Register forward hooks on model layers
        clear_hooks: Remove all registered hooks
    """
    
    def __init__(self, model: nn.Module, device: torch.device):
        """
        Initialize the attention extractor.
        
        Args:
            model: PointerGeneratorTransformer model instance
            device: Torch device (cuda/cpu)
        """
        self.model = model
        self.device = device
        self.hooks = []
        self.attention_weights = {}
        
    def register_hooks(self):
        """
        Register forward hooks to capture attention weights.
        
        Hooks are placed on:
        - Each transformer encoder layer's self-attention module
        """
        self.clear_hooks()
        self.attention_weights = {}
        
        # Hook for transformer self-attention layers
        for i, layer in enumerate(self.model.transformer.layers):
            def get_hook(layer_idx):
                def hook(module, input, output):
                    # TransformerEncoderLayer stores attention in output
                    # We need to modify the layer to return attention weights
                    pass
                return hook
            handle = layer.register_forward_hook(get_hook(i))
            self.hooks.append(handle)
    
    def clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.attention_weights = {}
    
    @torch.no_grad()
    def extract_attention(
        self,
        x: torch.Tensor,
        x_dict: Dict[str, torch.Tensor],
        return_predictions: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Extract all attention components from a forward pass.
        
        This method performs a modified forward pass that captures:
        1. Self-attention weights from each transformer layer
        2. Pointer attention scores and probabilities
        3. Pointer-generation gate values
        4. Final predictions
        
        Args:
            x: Location sequence tensor [seq_len, batch_size]
            x_dict: Dictionary with temporal features
            return_predictions: Whether to return model predictions
            
        Returns:
            Dictionary containing:
                - 'self_attention': List of attention tensors per layer
                - 'pointer_attention': Pointer mechanism attention
                - 'pointer_probs': Softmax of pointer attention
                - 'gate_values': Pointer-generation gate outputs
                - 'predictions': Model output log-probs (optional)
                - 'context': Final context vector
        """
        self.model.eval()
        results = {}
        
        # Get model components
        model = self.model
        
        # Prepare input (same as model forward)
        x_t = x.T.to(self.device)  # [batch_size, seq_len]
        batch_size, seq_len = x_t.shape
        lengths = x_dict['len'].to(self.device)
        
        # Compute embeddings
        loc_emb = model.loc_emb(x_t)
        user_emb = model.user_emb(x_dict['user'].to(self.device)).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Temporal features
        time = torch.clamp(x_dict['time'].T.to(self.device), 0, 96)
        weekday = torch.clamp(x_dict['weekday'].T.to(self.device), 0, 7)
        recency = torch.clamp(x_dict['diff'].T.to(self.device), 0, 8)
        duration = torch.clamp(x_dict['duration'].T.to(self.device), 0, 99)
        
        temporal = torch.cat([
            model.time_emb(time),
            model.weekday_emb(weekday),
            model.recency_emb(recency),
            model.duration_emb(duration)
        ], dim=-1)
        
        # Position from end
        positions = torch.arange(seq_len, device=self.device).unsqueeze(0).expand(batch_size, -1)
        pos_from_end = torch.clamp(lengths.unsqueeze(1) - positions, 0, model.max_seq_len - 1)
        pos_emb = model.pos_from_end_emb(pos_from_end)
        
        # Combine features
        combined = torch.cat([loc_emb, user_emb, temporal, pos_emb], dim=-1)
        hidden = model.input_norm(model.input_proj(combined))
        hidden = hidden + model.pos_encoding[:, :seq_len, :]
        
        # Create padding mask
        mask = positions >= lengths.unsqueeze(1)
        
        # Extract self-attention from transformer layers
        self_attentions = []
        current_hidden = hidden
        
        for layer in model.transformer.layers:
            # Manual self-attention extraction
            # Pre-norm
            normed = layer.norm1(current_hidden)
            
            # Self-attention (need to compute manually for weights)
            attn_weights = self._compute_self_attention(
                normed, layer.self_attn, mask
            )
            self_attentions.append(attn_weights)
            
            # Forward through the layer normally
            current_hidden = layer(current_hidden, src_key_padding_mask=mask)
        
        encoded = current_hidden
        results['self_attention'] = self_attentions
        
        # Extract context from last valid position
        batch_idx = torch.arange(batch_size, device=self.device)
        last_idx = (lengths - 1).clamp(min=0)
        context = encoded[batch_idx, last_idx]
        results['context'] = context
        
        # Pointer attention computation
        query = model.pointer_query(context).unsqueeze(1)
        keys = model.pointer_key(encoded)
        ptr_scores = torch.bmm(query, keys.transpose(1, 2)).squeeze(1) / math.sqrt(model.d_model)
        
        # Add position bias
        ptr_scores_with_bias = ptr_scores + model.position_bias[pos_from_end]
        
        # Store raw scores before masking
        results['pointer_scores_raw'] = ptr_scores.clone()
        results['pointer_scores_with_bias'] = ptr_scores_with_bias.clone()
        results['position_bias'] = model.position_bias[pos_from_end].clone()
        
        # Mask and softmax
        ptr_scores_masked = ptr_scores_with_bias.masked_fill(mask, float('-inf'))
        ptr_probs = F.softmax(ptr_scores_masked, dim=-1)
        
        results['pointer_attention'] = ptr_scores_masked
        results['pointer_probs'] = ptr_probs
        
        # Gate values
        gate = model.ptr_gen_gate(context)
        results['gate_values'] = gate
        
        # Generation distribution
        gen_logits = model.gen_head(context)
        gen_probs = F.softmax(gen_logits, dim=-1)
        results['generation_probs'] = gen_probs
        
        # Full prediction
        if return_predictions:
            ptr_dist = torch.zeros(batch_size, model.num_locations, device=self.device)
            ptr_dist.scatter_add_(1, x_t, ptr_probs)
            final_probs = gate * ptr_dist + (1 - gate) * gen_probs
            results['predictions'] = torch.log(final_probs + 1e-10)
            results['final_probs'] = final_probs
            results['pointer_distribution'] = ptr_dist
        
        # Store input sequence and metadata
        results['input_sequence'] = x_t
        results['lengths'] = lengths
        results['mask'] = mask
        results['pos_from_end'] = pos_from_end
        
        return results
    
    def _compute_self_attention(
        self,
        x: torch.Tensor,
        attn_module: nn.MultiheadAttention,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute self-attention weights manually.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_module: MultiheadAttention module
            mask: Padding mask [batch_size, seq_len]
            
        Returns:
            Attention weights [batch_size, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Get Q, K, V projections
        # PyTorch's MultiheadAttention uses in_proj for combined QKV
        if attn_module._qkv_same_embed_dim:
            qkv = F.linear(x, attn_module.in_proj_weight, attn_module.in_proj_bias)
            q, k, v = qkv.chunk(3, dim=-1)
        else:
            q = F.linear(x, attn_module.q_proj_weight, attn_module.in_proj_bias[:d_model])
            k = F.linear(x, attn_module.k_proj_weight, attn_module.in_proj_bias[d_model:2*d_model])
            v = F.linear(x, attn_module.v_proj_weight, attn_module.in_proj_bias[2*d_model:])
        
        # Reshape for multi-head attention
        num_heads = attn_module.num_heads
        head_dim = d_model // num_heads
        
        q = q.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)
        
        # Apply mask
        if mask is not None:
            # Expand mask for heads: [batch, 1, 1, seq_len]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask_expanded, float('-inf'))
            # Also mask query positions
            mask_query = mask.unsqueeze(1).unsqueeze(-1)
            attn_scores = attn_scores.masked_fill(mask_query, float('-inf'))
        
        # Softmax
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = torch.nan_to_num(attn_weights, nan=0.0)
        
        return attn_weights


def extract_batch_attention(
    extractor: AttentionExtractor,
    dataloader,
    num_samples: int = None,
    device: torch.device = None
) -> List[Dict]:
    """
    Extract attention for multiple samples from a dataloader.
    
    Args:
        extractor: AttentionExtractor instance
        dataloader: PyTorch DataLoader
        num_samples: Maximum number of samples to extract (None for all)
        device: Torch device
        
    Returns:
        List of attention dictionaries for each sample
    """
    all_results = []
    sample_count = 0
    
    for x, y, x_dict in dataloader:
        if device:
            x = x.to(device)
            y = y.to(device)
            x_dict = {k: v.to(device) for k, v in x_dict.items()}
        
        attention_data = extractor.extract_attention(x, x_dict)
        
        # Process batch into individual samples
        batch_size = x.shape[1]
        for i in range(batch_size):
            sample_result = {
                'input_sequence': attention_data['input_sequence'][i].cpu(),
                'length': attention_data['lengths'][i].cpu().item(),
                'pointer_probs': attention_data['pointer_probs'][i].cpu(),
                'pointer_scores_raw': attention_data['pointer_scores_raw'][i].cpu(),
                'position_bias': attention_data['position_bias'][i].cpu(),
                'gate_value': attention_data['gate_values'][i].cpu().item(),
                'self_attention': [sa[i].cpu() for sa in attention_data['self_attention']],
                'target': y[i].cpu().item(),
                'prediction': attention_data['predictions'][i].argmax().cpu().item(),
                'final_probs': attention_data['final_probs'][i].cpu(),
                'pointer_distribution': attention_data['pointer_distribution'][i].cpu(),
                'generation_probs': attention_data['generation_probs'][i].cpu(),
            }
            all_results.append(sample_result)
            sample_count += 1
            
            if num_samples and sample_count >= num_samples:
                return all_results
    
    return all_results


def compute_attention_statistics(
    attention_results: List[Dict]
) -> Dict[str, np.ndarray]:
    """
    Compute aggregate statistics over attention results.
    
    Computes:
    - Mean attention patterns
    - Attention entropy
    - Position-wise attention distribution
    - Gate value distribution
    
    Args:
        attention_results: List of attention dictionaries
        
    Returns:
        Dictionary of statistics
    """
    stats = {}
    
    # Gate values
    gate_values = np.array([r['gate_value'] for r in attention_results])
    stats['gate_mean'] = np.mean(gate_values)
    stats['gate_std'] = np.std(gate_values)
    stats['gate_values'] = gate_values
    
    # Pointer attention entropy
    entropies = []
    for r in attention_results:
        probs = r['pointer_probs'][:r['length']].numpy()
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        entropies.append(entropy)
    stats['pointer_entropy_mean'] = np.mean(entropies)
    stats['pointer_entropy_std'] = np.std(entropies)
    stats['pointer_entropies'] = np.array(entropies)
    
    # Position-wise attention (relative to sequence end)
    max_len = max(r['length'] for r in attention_results)
    position_attention = np.zeros(max_len)
    position_counts = np.zeros(max_len)
    
    for r in attention_results:
        length = r['length']
        probs = r['pointer_probs'][:length].numpy()
        # Align to end of sequence
        for pos in range(length):
            rel_pos = length - 1 - pos  # Position from end
            if rel_pos < max_len:
                position_attention[rel_pos] += probs[pos]
                position_counts[rel_pos] += 1
    
    position_attention = np.divide(
        position_attention,
        position_counts,
        out=np.zeros_like(position_attention),
        where=position_counts > 0
    )
    stats['position_attention'] = position_attention
    stats['position_counts'] = position_counts
    
    # Correct prediction statistics
    correct_mask = np.array([r['prediction'] == r['target'] for r in attention_results])
    stats['accuracy'] = np.mean(correct_mask)
    stats['correct_gate_mean'] = np.mean(gate_values[correct_mask]) if correct_mask.sum() > 0 else 0
    stats['incorrect_gate_mean'] = np.mean(gate_values[~correct_mask]) if (~correct_mask).sum() > 0 else 0
    
    return stats
