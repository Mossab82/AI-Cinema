"""
Cultural attention mechanisms for AI-Cinema.
Implements the attention components described in the paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

class CulturalAttention(nn.Module):
    """
    Cultural attention mechanism that prioritizes cultural elements.
    """
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 bias: bool = True):
        """
        Initialize cultural attention.
        
        Args:
            embed_dim: Dimension of input embeddings
            num_heads: Number of attention heads
            dropout: Dropout probability
            bias: Whether to include bias terms
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        # Cultural bias terms
        self.cultural_bias = nn.Parameter(torch.zeros(1, num_heads, 1, self.head_dim))
        
        self.dropout_module = nn.Dropout(dropout)
        self.scaling = self.head_dim ** -0.5

    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                cultural_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of cultural attention.
        
        Args:
            query: Query tensor [batch_size, tgt_len, embed_dim]
            key: Key tensor [batch_size, src_len, embed_dim]
            value: Value tensor [batch_size, src_len, embed_dim]
            cultural_mask: Optional mask for cultural elements
            attention_mask: Optional general attention mask
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        batch_size, tgt_len, embed_dim = query.size()
        src_len = key.size(1)
        
        scaling = self.scaling
        
        # Linear projections and reshape for attention heads
        q = self.q_proj(query).reshape(batch_size, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).reshape(batch_size, src_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Add cultural bias
        q = q + self.cultural_bias
        
        # Calculate attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * scaling
        
        # Apply cultural mask if provided
        if cultural_mask is not None:
            cultural_mask = cultural_mask.unsqueeze(1)  # Add head dimension
            attn_weights = attn_weights + cultural_mask
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout_module(attn_weights)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project output
        output = output.transpose(1, 2).reshape(batch_size, tgt_len, embed_dim)
        output = self.out_proj(output)
        
        return output, attn_weights

class MultiHeadCulturalAttention(nn.Module):
    """
    Multi-head version of cultural attention with specialized cultural heads.
    """
    def __init__(self, 
                 config: Dict,
                 embed_dim: int,
                 num_heads: int = 8,
                 cultural_heads: int = 2):
        """
        Initialize multi-head cultural attention.
        
        Args:
            config: Model configuration
            embed_dim: Dimension of input embeddings
            num_heads: Total number of attention heads
            cultural_heads: Number of heads dedicated to cultural attention
        """
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.cultural_heads = cultural_heads
        assert cultural_heads <= num_heads, "cultural_heads must be <= num_heads"
        
        # Regular attention for non-cultural heads
        self.regular_attention = CulturalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads - cultural_heads,
            dropout=config.get('attention_dropout', 0.1)
        )
        
        # Cultural attention for cultural heads
        if cultural_heads > 0:
            self.cultural_attention = CulturalAttention(
                embed_dim=embed_dim,
                num_heads=cultural_heads,
                dropout=config.get('attention_dropout', 0.1)
            )
        
        self.output_layer = nn.Linear(embed_dim, embed_dim)

    def forward(self,
                hidden_states: torch.Tensor,
                cultural_mask: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass combining regular and cultural attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, embed_dim]
            cultural_mask: Optional mask for cultural elements
            attention_mask: Optional general attention mask
            
        Returns:
            Output tensor after attention
        """
        outputs = []
        
        # Regular attention
        if self.num_heads > self.cultural_heads:
            regular_output, _ = self.regular_attention(
                hidden_states, hidden_states, hidden_states,
                attention_mask=attention_mask
            )
            outputs.append(regular_output)
        
        # Cultural attention
        if self.cultural_heads > 0:
            cultural_output, _ = self.cultural_attention(
                hidden_states, hidden_states, hidden_states,
                cultural_mask=cultural_mask,
                attention_mask=attention_mask
            )
            outputs.append(cultural_output)
        
        # Combine outputs
        combined_output = torch.cat(outputs, dim=-1)
        return self.output_layer(combined_output)

def create_cultural_mask(tokens: List[str],
                        cultural_elements: Dict[str, List[str]],
                        max_len: int) -> torch.Tensor:
    """
    Create attention mask for cultural elements.
    
    Args:
        tokens: Input tokens
        cultural_elements: Dictionary of cultural elements
        max_len: Maximum sequence length
        
    Returns:
        Cultural attention mask tensor
    """
    mask = torch.zeros(max_len, max_len)
    
    # Find positions of cultural elements
    cultural_positions = []
    for i, token in enumerate(tokens):
        if any(token in elements for elements in cultural_elements.values()):
            cultural_positions.append(i)
    
    # Enhance attention for cultural elements
    for pos in cultural_positions:
        mask[pos, :] = 1.0  # Attend to all positions
        mask[:, pos] = 1.0  # All positions attend to cultural element
    
    return mask
