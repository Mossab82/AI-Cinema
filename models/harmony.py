"""
Harmony Function implementation for AI-Cinema.
Implements the balanced scoring mechanism described in the paper.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
import math

class CulturalPreservationMetric(nn.Module):
    """
    Measures cultural preservation in generated text.
    """
    def __init__(self, cultural_weights: Optional[Dict[str, float]] = None):
        """
        Initialize cultural preservation metric.
        
        Args:
            cultural_weights: Optional weights for different cultural elements
        """
        super().__init__()
        
        # Default weights if none provided
        self.weights = cultural_weights or {
            'proverbs': 1.0,
            'customs': 0.8,
            'values': 0.9,
            'archetypes': 0.7,
            'settings': 0.6
        }
        
        # Create learnable weight parameters
        self.weight_params = nn.ParameterDict({
            k: nn.Parameter(torch.tensor(v))
            for k, v in self.weights.items()
        })

    def forward(self, 
                cultural_elements: Dict[str, List[str]],
                expected_elements: Dict[str, List[str]]) -> torch.Tensor:
        """
        Compute cultural preservation score.
        
        Args:
            cultural_elements: Detected cultural elements in text
            expected_elements: Expected cultural elements
            
        Returns:
            Cultural preservation score
        """
        scores = []
        
        for category, weight in self.weight_params.items():
            if category in cultural_elements and category in expected_elements:
                detected = set(cultural_elements[category])
                expected = set(expected_elements[category])
                
                if expected:  # Avoid division by zero
                    preservation = len(detected & expected) / len(expected)
                    scores.append(weight * preservation)
        
        if not scores:
            return torch.tensor(0.0)
        
        return torch.stack(scores).mean()

class LinguisticFluencyMetric(nn.Module):
    """
    Measures linguistic fluency of generated text.
    """
    def __init__(self, 
                 model_dim: int = 768,
                 dropout: float = 0.1):
        """
        Initialize linguistic fluency metric.
        
        Args:
            model_dim: Model embedding dimension
            dropout: Dropout probability
        """
        super().__init__()
        
        self.fluency_scorer = nn.Sequential(
            nn.Linear(model_dim, model_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute linguistic fluency score.
        
        Args:
            hidden_states: Model hidden states
            attention_mask: Optional attention mask
            
        Returns:
            Fluency score
        """
        # Pool hidden states
        if attention_mask is not None:
            masked_states = hidden_states * attention_mask.unsqueeze(-1)
            pooled = masked_states.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True)
        else:
            pooled = hidden_states.mean(dim=1)
        
        return self.fluency_scorer(pooled).squeeze(-1)

class DialectalAuthenticityMetric(nn.Module):
    """
    Measures dialectal authenticity of generated text.
    """
    def __init__(self, 
                 num_dialects: int,
                 model_dim: int = 768):
        """
        Initialize dialectal authenticity metric.
        
        Args:
            num_dialects: Number of supported dialects
            model_dim: Model embedding dimension
        """
        super().__init__()
        
        self.dialect_classifier = nn.Linear(model_dim, num_dialects)
        self.authenticity_scorer = nn.Linear(model_dim, 1)

    def forward(self, 
                hidden_states: torch.Tensor,
                target_dialect: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute dialectal authenticity score.
        
        Args:
            hidden_states: Model hidden states
            target_dialect: Target dialect labels
            
        Returns:
            Tuple of (authenticity score, dialect predictions)
        """
        # Pool hidden states
        pooled = hidden_states.mean(dim=1)
        
        # Compute dialect probabilities
        dialect_logits = self.dialect_classifier(pooled)
        dialect_probs = torch.softmax(dialect_logits, dim=-1)
        
        # Compute authenticity score
        authenticity = torch.sigmoid(self.authenticity_scorer(pooled)).squeeze(-1)
        
        # Combine with target dialect accuracy
        target_probs = dialect_probs.gather(1, target_dialect.unsqueeze(-1)).squeeze(-1)
        score = authenticity * target_probs
        
        return score, dialect_probs

class HarmonyFunction(nn.Module):
    """
    Harmony Function combining cultural preservation, linguistic fluency,
    and dialectal authenticity.
    """
    def __init__(self,
                 model_dim: int = 768,
                 num_dialects: int = 25,
                 alpha: float = 0.6,
                 beta: float = 0.2):
        """
        Initialize harmony function.
        
        Args:
            model_dim: Model embedding dimension
            num_dialects: Number of supported dialects
            alpha: Weight for cultural preservation
            beta: Weight for dialectal authenticity
        """
        super().__init__()
        
        # Component metrics
        self.cultural_metric = CulturalPreservationMetric()
        self.linguistic_metric = LinguisticFluencyMetric(model_dim)
        self.dialectal_metric = DialectalAuthenticityMetric(num_dialects, model_dim)
        
        # Learnable weights
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        
        # Ensure weights stay in valid range
        self.register_buffer('zero', torch.tensor(0.0))
        self.register_buffer('one', torch.tensor(1.0))

    def forward(self,
                hidden_states: torch.Tensor,
                cultural_elements: Dict[str, List[str]],
                expected_elements: Dict[str, List[str]],
                target_dialect: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Compute harmony score.
        
        Args:
            hidden_states: Model hidden states
            cultural_elements: Detected cultural elements
            expected_elements: Expected cultural elements
            target_dialect: Target dialect labels
            attention_mask: Optional attention mask
            
        Returns:
            Tuple of (harmony score, component scores)
        """
        # Clamp weights to [0, 1]
        alpha = torch.clamp(self.alpha, min=self.zero, max=self.one)
        beta = torch.clamp(self.beta, min=self.zero, max=self.one)
        
        # Compute component scores
        cultural_score = self.cultural_metric(cultural_elements, expected_elements)
        linguistic_score = self.linguistic_metric(hidden_states, attention_mask)
        dialectal_score, dialect_probs = self.dialectal_metric(hidden_states, target_dialect)
        
        # Combine scores using harmony function
        harmony_score = (
            alpha * cultural_score +
            (1 - alpha) * linguistic_score +
            beta * dialectal_score
        )
        
        scores = {
            'cultural': cultural_score,
            'linguistic': linguistic_score,
            'dialectal': dialectal_score,
            'dialect_probs': dialect_probs
        }
        
        return harmony_score, scores
