"""
Dialectal processing utilities for AI-Cinema.
"""

import re
from typing import Dict, List, Optional, Set, Tuple
from pathlib import Path
import json
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.tokenizers.word import simple_word_tokenize

# Dialect mappings from the MADAR corpus
DIALECT_MAPPINGS = {
    'gulf': {
        'pronouns': {'أنا': 'انا', 'نحن': 'احنا', 'أنت': 'انت'},
        'verbs': {'يذهب': 'يروح', 'يأتي': 'يجي', 'يريد': 'يبي'},
        'markers': {'هذا': 'هالـ', 'ماذا': 'وش', 'كيف': 'كيف'}
    },
    'egyptian': {
        'pronouns': {'أنا': 'انا', 'نحن': 'احنا', 'أنت': 'انت'},
        'verbs': {'يذهب': 'يروح', 'يأتي': 'ييجي', 'يريد': 'عايز'},
        'markers': {'هذا': 'دا', 'ماذا': 'ايه', 'كيف': 'ازاي'}
    },
    'levantine': {
        'pronouns': {'أنا': 'انا', 'نحن': 'نحنا', 'أنت': 'انت'},
        'verbs': {'يذهب': 'يروح', 'يأتي': 'يجي', 'يريد': 'بدو'},
        'markers': {'هذا': 'هاد', 'ماذا': 'شو', 'كيف': 'كيف'}
    }
}

def detect_dialect(text: str) -> Tuple[str, float]:
    """
    Detect Arabic dialect in text.
    
    Args:
        text: Input text
        
    Returns:
        Tuple of (dialect_name, confidence_score)
    """
    # Normalize text
    text = normalize_dialectal_text(text)
    tokens = simple_word_tokenize(text)
    
    # Calculate dialect scores
    scores = {}
    for dialect, mappings in DIALECT_MAPPINGS.items():
        score = _calculate_dialect_score(tokens, mappings)
        scores[dialect] = score
    
    # Get most likely dialect
    best_dialect = max(scores.items(), key=lambda x: x[1])
    return best_dialect

def _calculate_dialect_score(tokens: List[str], mappings: Dict) -> float:
    """Calculate score for a specific dialect."""
    score = 0
    total_markers = 0
    
    for category in mappings.values():
        for msa, dialect in category.items():
            total_markers += 1
            if dialect in tokens:
                score += 1
            elif msa in tokens:  # MSA form indicates non-dialect
                score -= 0.5
    
    return score / total_markers if total_markers > 0 else 0

def convert_dialect(text: str, 
                   source_dialect: str,
                   target_dialect: str) -> str:
    """
    Convert text from one dialect to another.
    
    Args:
        text: Input text
        source_dialect: Source dialect
        target_dialect: Target dialect
        
    Returns:
        Converted text
    """
    if source_dialect == target_dialect:
        return text
    
    # Normalize text
    text = normalize_dialectal_text(text)
    tokens = simple_word_tokenize(text)
    
    # Convert to MSA first if source is not MSA
    if source_dialect != 'msa':
        tokens = _convert_to_msa(tokens, source_dialect)
    
    # Convert to target dialect if not MSA
    if target_dialect != 'msa':
        tokens = _convert_to_dialect(tokens, target_dialect)
    
    return ' '.join(tokens)

def _convert_to_msa(tokens: List[str], 
                   source_dialect: str) -> List[str]:
    """Convert dialectal tokens to MSA."""
    converted = []
    mappings = DIALECT_MAPPINGS.get(source_dialect, {})
    
    # Create reverse mapping
    reverse_map = {}
    for category in mappings.values():
        for msa, dialect in category.items():
            reverse_map[dialect] = msa
    
    # Convert tokens
    for token in tokens:
        converted.append(reverse_map.get(token, token))
    
    return converted

def _convert_to_dialect(tokens: List[str], 
                       target_dialect: str) -> List[str]:
    """Convert MSA tokens to target dialect."""
    converted = []
    mappings = DIALECT_MAPPINGS.get(target_dialect, {})
    
    # Create forward mapping
    forward_map = {}
    for category in mappings.values():
        forward_map.update(category)
    
    # Convert tokens
    for token in tokens:
        converted.append(forward_map.get(token, token))
    
    return converted

def normalize_dialectal_text(text: str) -> str:
    """
    Normalize Arabic text for dialectal processing.
    
    Args:
        text: Input text
        
    Returns:
        Normalized text
    """
    # Basic normalization
    text = normalize_unicode(text)
    
    # Remove diacritics
    text = re.sub(r'[\u064B-\u065F\u0670]', '', text)
    
    # Normalize hamza forms
    text = re.sub(r'[إأٱآا]', 'ا', text)
    text = re.sub(r'[ؤئ]', 'ء', text)
    
    # Normalize taa marbouta
    text = text.replace('ة', 'ه')
    
    # Normalize alef maksura
    text = text.replace('ى', 'ي')
    
    return text.strip()

def validate_dialect_compatibility(text: str, 
                                dialect: str,
                                threshold: float = 0.8) -> bool:
    """
    Validate if text matches expected dialect.
    
    Args:
        text: Input text
        dialect: Expected dialect
        threshold: Minimum compatibility score
        
    Returns:
        True if text is compatible with dialect
    """
    detected_dialect, score = detect_dialect(text)
    return detected_dialect == dialect and score >= threshold

def get_dialectal_features(dialect: str) -> Dict[str, List[str]]:
    """
    Get distinctive features for a dialect.
    
    Args:
        dialect: Target dialect
        
    Returns:
        Dictionary of dialectal features
    """
    if dialect not in DIALECT_MAPPINGS:
        raise ValueError(f"Unsupported dialect: {dialect}")
    
    features = DIALECT_MAPPINGS[dialect]
    return {
        'pronouns': list(features['pronouns'].values()),
        'verbs': list(features['verbs'].values()),
        'markers': list(features['markers'].values())
    }
