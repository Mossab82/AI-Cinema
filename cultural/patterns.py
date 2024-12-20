"""
Cultural pattern recognition system for AI-Cinema.
Implements pattern detection and matching algorithms.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

@dataclass
class PatternMatch:
    """Represents a matched cultural pattern."""
    pattern_type: str
    content: str
    confidence: float
    context: str
    start_idx: int
    end_idx: int

class PatternMatcher(nn.Module):
    """Base class for pattern matching."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__()
        self.tokenizer = tokenizer
    
    def match(self, text: str) -> List[PatternMatch]:
        """Match patterns in text."""
        raise NotImplementedError

class MoralPatternMatcher(PatternMatcher):
    """Matches moral teachings and values."""
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 moral_patterns: Dict[str, List[str]]):
        """
        Initialize moral pattern matcher.
        
        Args:
            tokenizer: Tokenizer for text processing
            moral_patterns: Dictionary of moral patterns
        """
        super().__init__(tokenizer)
        self.patterns = moral_patterns
        
        # Pattern classification head
        self.classifier = nn.Linear(768, len(moral_patterns))
    
    def match(self, text: str) -> List[PatternMatch]:
        """
        Match moral patterns in text.
        
        Args:
            text: Input text
            
        Returns:
            List of matched patterns
        """
        matches = []
        tokens = self.tokenizer.tokenize(text)
        
        for pattern_type, examples in self.patterns.items():
            for example in examples:
                # Find pattern occurrences
                example_tokens = self.tokenizer.tokenize(example)
                for i in range(len(tokens) - len(example_tokens) + 1):
                    if self._is_pattern_match(tokens[i:i+len(example_tokens)], example_tokens):
                        matches.append(PatternMatch(
                            pattern_type="moral",
                            content=example,
                            confidence=self._compute_confidence(tokens[i:i+len(example_tokens)]),
                            context=self._get_context(tokens, i, len(example_tokens)),
                            start_idx=i,
                            end_idx=i+len(example_tokens)
                        ))
        
        return matches
    
    def _is_pattern_match(self, text_tokens: List[str], pattern_tokens: List[str]) -> bool:
        """Check if tokens match pattern."""
        return text_tokens == pattern_tokens
    
    def _compute_confidence(self, tokens: List[str]) -> float:
        """Compute confidence score for match."""
        # Placeholder for actual confidence computation
        return 0.8
    
    def _get_context(self, tokens: List[str], start: int, length: int, context_size: int = 5) -> str:
        """Get surrounding context for pattern."""
        ctx_start = max(0, start - context_size)
        ctx_end = min(len(tokens), start + length + context_size)
        return self.tokenizer.convert_tokens_to_string(tokens[ctx_start:ctx_end])

class ArchetypePatternMatcher(PatternMatcher):
    """Matches character and situation archetypes."""
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 archetypes: Dict[str, Dict[str, List[str]]]):
        """
        Initialize archetype pattern matcher.
        
        Args:
            tokenizer: Tokenizer for text processing
            archetypes: Dictionary of character and situation archetypes
        """
        super().__init__(tokenizer)
        self.archetypes = archetypes
        
        # Archetype classification head
        self.classifier = nn.Linear(768, len(archetypes))
    
    def match(self, text: str) -> List[PatternMatch]:
        """
        Match archetypal patterns in text.
        
        Args:
            text: Input text
            
        Returns:
            List of matched patterns
        """
        matches = []
        tokens = self.tokenizer.tokenize(text)
        
        for archetype_type, patterns in self.archetypes.items():
            for pattern_name, examples in patterns.items():
                for example in examples:
                    example_tokens = self.tokenizer.tokenize(example)
                    for i in range(len(tokens) - len(example_tokens) + 1):
                        if self._is_archetype_match(tokens[i:i+len(example_tokens)], example_tokens):
                            matches.append(PatternMatch(
                                pattern_type="archetype",
                                content=f"{archetype_type}:{pattern_name}",
                                confidence=self._compute_confidence(tokens[i:i+len(example_tokens)]),
                                context=self._get_context(tokens, i, len(example_tokens)),
                                start_idx=i,
                                end_idx=i+len(example_tokens)
                            ))
        
        return matches
    
    def _is_archetype_match(self, text_tokens: List[str], pattern_tokens: List[str]) -> bool:
        """Check if tokens match archetypal pattern."""
        return text_tokens == pattern_tokens
    
    def _compute_confidence(self, tokens: List[str]) -> float:
        """Compute confidence score for archetype match."""
        # Placeholder for actual confidence computation
        return 0.8
    
    def _get_context(self, tokens: List[str], start: int, length: int, context_size: int = 5) -> str:
        """Get surrounding context for archetype."""
        ctx_start = max(0, start - context_size)
        ctx_end = min(len(tokens), start + length + context_size)
        return self.tokenizer.convert_tokens_to_string(tokens[ctx_start:ctx_end])

class DialectalPatternMatcher(PatternMatcher):
    """Matches dialect-specific patterns and phrases."""
    
    def __init__(self, 
                 tokenizer: PreTrainedTokenizer,
                 dialectal_patterns: Dict[str, Dict[str, List[str]]]):
        """
        Initialize dialectal pattern matcher.
        
        Args:
            tokenizer: Tokenizer for text processing
            dialectal_patterns: Dictionary of dialect-specific patterns
        """
        super().__init__(tokenizer)
        self.dialectal_patterns = dialectal_patterns
        
        # Dialect classification head
        self.dialect_classifier = nn.Linear(768, len(dialectal_patterns))
    
    def match(self, text: str, dialect: Optional[str] = None) -> List[PatternMatch]:
        """
        Match dialectal patterns in text.
        
        Args:
            text: Input text
            dialect: Optional target dialect to match
            
        Returns:
            List of matched patterns
        """
        matches = []
        tokens = self.tokenizer.tokenize(text)
        
        # Filter patterns by dialect if specified
        patterns_to_check = (
            {dialect: self.dialectal_patterns[dialect]}
            if dialect and dialect in self.dialectal_patterns
            else self.dialectal_patterns
        )
        
        for dialect_name, patterns in patterns_to_check.items():
            for pattern_type, examples in patterns.items():
                for example in examples:
                    example_tokens = self.tokenizer.tokenize(example)
                    for i in range(len(tokens) - len(example_tokens) + 1):
                        if self._is_dialectal_match(tokens[i:i+len(example_tokens)], example_tokens):
                            matches.append(PatternMatch(
                                pattern_type="dialectal",
                                content=f"{dialect_name}:{pattern_type}",
                                confidence=self._compute_confidence(tokens[i:i+len(example_tokens)]),
                                context=self._get_context(tokens, i, len(example_tokens)),
                                start_idx=i,
                                end_idx=i+len(example_tokens)
                            ))
        
        return matches
    
    def _is_dialectal_match(self, text_tokens: List[str], pattern_tokens: List[str]) -> bool:
        """Check if tokens match dialectal pattern."""
        return text_tokens == pattern_tokens
    
    def _compute_confidence(self, tokens: List[str]) -> float:
        """Compute confidence score for dialectal match."""
        # Placeholder for actual confidence computation
        return 0.8
    
    def _get_context(self, tokens: List[str], start: int, length: int, context_size: int = 5) -> str:
        """Get surrounding context for dialectal pattern."""
        ctx_start = max(0, start - context_size)
        ctx_end = min(len(tokens), start + length + context_size)
        return self.tokenizer.convert_tokens_to_string(tokens[ctx_start:ctx_end])

class CulturalPatternDetector(nn.Module):
    """Main pattern detection system combining all matchers."""
    
    def __init__(self,
                 tokenizer: PreTrainedTokenizer,
                 moral_patterns: Dict[str, List[str]],
                 archetypes: Dict[str, Dict[str, List[str]]],
                 dialectal_patterns: Dict[str, Dict[str, List[str]]]):
        """
        Initialize cultural pattern detector.
        
        Args:
            tokenizer: Tokenizer for text processing
            moral_patterns: Dictionary of moral patterns
            archetypes: Dictionary of character and situation archetypes
            dialectal_patterns: Dictionary of dialect-specific patterns
        """
        super().__init__()
        
        # Initialize pattern matchers
        self.moral_matcher = MoralPatternMatcher(tokenizer, moral_patterns)
        self.archetype_matcher = ArchetypePatternMatcher(tokenizer, archetypes)
        self.dialectal_matcher = DialectalPatternMatcher(tokenizer, dialectal_patterns)
    
    def forward(self, 
                text: str,
                dialect: Optional[str] = None) -> Dict[str, List[PatternMatch]]:
        """
        Detect all cultural patterns in text.
        
        Args:
            text: Input text
            dialect: Optional target dialect
            
        Returns:
            Dictionary of pattern matches by category
        """
        return {
            'moral': self.moral_matcher.match(text),
            'archetype': self.archetype_matcher.match(text),
            'dialectal': self.dialectal_matcher.match(text, dialect)
        }
    
    def filter_matches(self,
                      matches: Dict[str, List[PatternMatch]],
                      min_confidence: float = 0.5) -> Dict[str, List[PatternMatch]]:
        """
        Filter matches by confidence threshold.
        
        Args:
            matches: Dictionary of pattern matches
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered matches
        """
        return {
            category: [m for m in category_matches if m.confidence >= min_confidence]
            for category, category_matches in matches.items()
        }
