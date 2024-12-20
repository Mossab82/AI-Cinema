"""
Embedding layers for AI-Cinema.
Includes cultural, dialectal, and positional embeddings.
"""

import math
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

class CulturalEmbedding(nn.Module):
    """
    Embedding layer with cultural markers.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 padding_idx: Optional[int] = None,
                 cultural_dim: int = 64):
        """
        Initialize cultural embedding.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            padding_idx: Index for padding token
            cultural_dim: Dimension of cultural features
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.cultural_dim = cultural_dim
        
        # Main token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim - cultural_dim,
            padding_idx=padding_idx
        )
        
        # Cultural feature embedding
        self.cultural_embedding = nn.Embedding(
            vocab_size, cultural_dim,
            padding_idx=padding_idx
        )
        
        # Cultural attention weights
        self.cultural_attention = nn.Parameter(
            torch.ones(cultural_dim)
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.cultural_embedding.weight, mean=0, std=0.02)
        
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()
            self.cultural_embedding.weight.data[self.cultural_embedding.padding_idx].zero_()

    def forward(self,
                input_ids: torch.Tensor,
                cultural_markers: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of cultural embedding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            cultural_markers: Optional tensor marking cultural tokens
            
        Returns:
            Combined embeddings [batch_size, seq_len, embed_dim]
        """
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        cultural_embeds = self.cultural_embedding(input_ids)
        
        # Apply cultural attention if markers provided
        if cultural_markers is not None:
            cultural_weights = torch.sigmoid(cultural_markers) * self.cultural_attention
            cultural_embeds = cultural_embeds * cultural_weights.unsqueeze(-1)
        
        # Combine embeddings
        return torch.cat([token_embeds, cultural_embeds], dim=-1)

class DialectalEmbedding(nn.Module):
    """
    Embedding layer with dialectal features.
    """
    def __init__(self,
                 vocab_size: int,
                 embed_dim: int,
                 num_dialects: int,
                 padding_idx: Optional[int] = None):
        """
        Initialize dialectal embedding.
        
        Args:
            vocab_size: Size of vocabulary
            embed_dim: Dimension of embeddings
            num_dialects: Number of supported dialects
            padding_idx: Index for padding token
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_dialects = num_dialects
        
        # Token embedding
        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim,
            padding_idx=padding_idx
        )
        
        # Dialect embedding
        self.dialect_embedding = nn.Embedding(
            num_dialects, embed_dim
        )
        
        # Dialectal adaptation layer
        self.dialect_adapter = nn.Linear(
            embed_dim * 2, embed_dim
        )
        
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding parameters."""
        nn.init.normal_(self.token_embedding.weight, mean=0, std=0.02)
        nn.init.normal_(self.dialect_embedding.weight, mean=0, std=0.02)
        
        if self.token_embedding.padding_idx is not None:
            self.token_embedding.weight.data[self.token_embedding.padding_idx].zero_()

    def forward(self,
                input_ids: torch.Tensor,
                dialect_ids: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of dialectal embedding.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            dialect_ids: Dialect IDs [batch_size]
            
        Returns:
            Dialectally adapted embeddings
        """
        # Get token embeddings
        token_embeds = self.token_embedding(input_ids)
        
        # Get dialect embeddings
        dialect_embeds = self.dialect_embedding(dialect_ids)
        
        # Expand dialect embeddings to match sequence length
        dialect_embeds = dialect_embeds.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        
        # Combine token and dialect embeddings
        combined = torch.cat([token_embeds, dialect_embeds], dim=-1)
        
        # Apply dialectal adaptation
        return self.dialect_adapter(combined)

class PositionalEmbedding(nn.Module):
    """
    Sinusoidal positional embedding layer.
    """
    def __init__(self, 
                 embed_dim: int,
                 max_seq_length: int = 5000):
        """
        Initialize positional embedding.
        
        Args:
            embed_dim: Dimension of embeddings
            max_seq_length: Maximum sequence length
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2) * (-math.log(10000.0) / embed_dim))
        
        pe = torch.zeros(max_seq_length, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register buffer (not a parameter)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of positional embedding.
        
        Args:
            x: Input tensor [batch_size, seq_len, embed_dim]
            
        Returns:
            Tensor with added positional encodings
        """
        seq_len = x.size(1)
        return x + self.pe[:seq_len]

def create_sinusoidal_positions(n_position: int, 
                              d_hid: int) -> torch.Tensor:
    """
    Create sinusoidal position encoding table.
    
    Args:
        n_position: Maximum sequence length
        d_hid: Hidden dimension size
        
    Returns:
        Position encoding tensor
    """
    position = torch.arange(n_position).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_hid, 2) * (-math.log(10000.0) / d_hid))
    
    pos_table = torch.zeros(n_position, d_hid)
    pos_table[:, 0::2] = torch.sin(position * div_term)
    pos_table[:, 1::2] = torch.cos(position * div_term)
    
    return pos_table
