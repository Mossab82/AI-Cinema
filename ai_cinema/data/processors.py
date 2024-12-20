"""
Text processing utilities for AI-Cinema.
Handles Arabic text preprocessing, cultural element extraction, and dialectal processing.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
import logging
from pathlib import Path
import numpy as np
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.tokenizers.word import simple_word_tokenize
from camel_tools.disambig.mle import MLEDisambiguator
from camel_tools.tagger.default import DefaultTagger

logger = logging.getLogger(__name__)

class ArabicTextProcessor:
    """Processor for Arabic text normalization and cleaning."""
    
    def __init__(self):
        self.mle_disambig = MLEDisambiguator.pretrained()
        self.pos_tagger = DefaultTagger.pretrained()
        
        # Common Arabic diacritics to remove
        self.diacritics = re.compile(r'[\u064B-\u065F\u0670]')
        
        # Punctuation normalization map
        self.punct_map = {
            '،': ',',
            '؛': ';',
            '؟': '?',
            '٪': '%'
        }
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Arabic text.
        
        Args:
            text: Input text
            
        Returns:
            Normalized text
        """
        # Remove diacritics
        text = self.diacritics.sub('', text)
        
        # Normalize Unicode
        text = normalize_unicode(text)
        
        # Normalize punctuation
        for ar, en in self.punct_map.items():
            text = text.replace(ar, en)
        
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize Arabic text.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        return simple_word_tokenize(self.normalize_text(text))
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Get POS tags for text.
        
        Args:
            text: Input text
            
        Returns:
            List of (token, tag) tuples
        """
        tokens = self.tokenize(text)
        analyses = self.mle_disambig.disambiguate(tokens)
        return [(token, analysis.pos) for token, analysis in zip(tokens, analyses)]

class CulturalElementExtractor:
    """Extractor for cultural elements from text."""
    
    def __init__(self, resources: Dict[str, Any]):
        self.resources = resources
        self.text_processor = ArabicTextProcessor()
    
    def extract_elements(self, text: str) -> Dict[str, List[str]]:
        """
        Extract cultural elements from text.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary of extracted elements by category
        """
        elements = {
            'proverbs': self._extract_proverbs(text),
            'customs': self._extract_customs(text),
            'values': self._extract_values(text),
            'archetypes': self._extract_archetypes(text),
            'settings': self._extract_settings(text)
        }
        return elements
    
    def _extract_proverbs(self, text: str) -> List[str]:
        """Extract proverbs from text."""
        proverbs = []
        text_tokens = set(self.text_processor.tokenize(text))
        
        for proverb in self.resources['cultural_knowledge']['proverbs']:
            proverb_tokens = set(self.text_processor.tokenize(proverb))
            # Check if significant portion of proverb tokens are present
            if len(proverb_tokens & text_tokens) / len(proverb_tokens) > 0.7:
                proverbs.append(proverb)
        
        return proverbs
    
    def _extract_customs(self, text: str) -> List[str]:
        """Extract cultural customs from text."""
        customs = []
        for custom in self.resources['cultural_knowledge']['customs']:
            if custom in text:
                customs.append(custom)
        return customs
    
    def _extract_values(self, text: str) -> List[str]:
        """Extract cultural values from text."""
        values = []
        tokens = self.text_processor.tokenize(text)
        pos_tags = self.text_processor.get_pos_tags(text)
        
        # Look for value-related patterns
        for value in self.resources['cultural_knowledge']['values']:
            value_tokens = self.text_processor.tokenize(value)
            if all(token in tokens for token in value_tokens):
                values.append(value)
        
        return values
    
    def _extract_archetypes(self, text: str) -> List[str]:
        """Extract character archetypes from text."""
        archetypes = []
        for archetype in self.resources['cultural_knowledge']['archetypes']:
            if archetype in text:
                archetypes.append(archetype)
        return archetypes
    
    def _extract_settings(self, text: str) -> List[str]:
        """Extract traditional settings from text."""
        settings = []
        for setting in self.resources['cultural_knowledge']['settings']:
            if setting in text:
                settings.append(setting)
        return settings

class DialectDetector:
    """Detector for Arabic dialects in text."""
    
    def __init__(self, dialectal_corpus: Dict[str, List[str]]):
        self.dialectal_corpus = dialectal_corpus
        self.text_processor = ArabicTextProcessor()
    
    def detect_dialect(self, text: str) -> Tuple[str, float]:
        """
        Detect the dialect of text.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (dialect_name, confidence_score)
        """
        text = self.text_processor.normalize_text(text)
        tokens = self.text_processor.tokenize(text)
        
        # Calculate dialect scores
        scores = {}
        for dialect, corpus in self.dialectal_corpus.items():
            score = self._calculate_dialect_score(tokens, corpus)
            scores[dialect] = score
        
        # Get most likely dialect
        best_dialect = max(scores.items(), key=lambda x: x[1])
        return best_dialect[0], best_dialect[1]
    
    def _calculate_dialect_score(self, tokens: List[str], corpus: List[str]) -> float:
        """Calculate dialect likelihood score."""
        corpus_tokens = set(' '.join(corpus).split())
        matches = sum(1 for token in tokens if token in corpus_tokens)
        return matches / len(tokens) if tokens else 0.0

def preprocess_text(text: str) -> str:
    """
    Preprocess Arabic text for model input.
    
    Args:
        text: Raw input text
        
    Returns:
        Preprocessed text
    """
    processor = ArabicTextProcessor()
    return processor.normalize_text(text)

def extract_cultural_elements(text: str, resources: Dict[str, Any]) -> Dict[str, List[str]]:
    """
    Extract cultural elements from text.
    
    Args:
        text: Input text
        resources: Cultural resources dictionary
        
    Returns:
        Dictionary of extracted elements
    """
    extractor = CulturalElementExtractor(resources)
    return extractor.extract_elements(text)

def detect_dialect(text: str, dialectal_corpus: Dict[str, List[str]]) -> Tuple[str, float]:
    """
    Detect dialect of text.
    
    Args:
        text: Input text
        dialectal_corpus: Dialectal corpus dictionary
        
    Returns:
        Tuple of (dialect_name, confidence_score)
    """
    detector = DialectDetector(dialectal_corpus)
    return detector.detect_dialect(text)

def normalize_arabic(text: str) -> str:
    """
    Normalize Arabic text.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    processor = ArabicTextProcessor()
    return processor.normalize_text(text)

def tokenize_arabic(text: str) -> List[str]:
    """
    Tokenize Arabic text.
    
    Args:
        text: Input text
        
    Returns:
        List of tokens
    """
    processor = ArabicTextProcessor()
    return processor.tokenize(text)
