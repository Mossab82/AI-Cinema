"""
Evaluation metrics for AI-Cinema.
"""

import numpy as np
from typing import Dict, List, Optional, Set, Tuple
from collections import Counter
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from .dialectal import detect_dialect, normalize_dialectal_text

def calculate_cpm(generated_text: str,
                 cultural_elements: Dict[str, List[str]]) -> float:
    """
    Calculate Cultural Preservation Metric (CPM).
    
    Args:
        generated_text: Generated text
        cultural_elements: Expected cultural elements
        
    Returns:
        CPM score (0-1)
    """
    total_elements = sum(len(elements) for elements in cultural_elements.values())
    if total_elements == 0:
        return 1.0
    
    matched_elements = 0
    for category_elements in cultural_elements.values():
        for element in category_elements:
            if element in generated_text:
                matched_elements += 1
    
    return matched_elements / total_elements

def calculate_bleu(hypothesis: str,
                  reference: str,
                  weights: Optional[Tuple[float, ...]] = None) -> float:
    """
    Calculate BLEU score.
    
    Args:
        hypothesis: Generated text
        reference: Reference text
        weights: Optional n-gram weights
        
    Returns:
        BLEU score (0-1)
    """
    if weights is None:
        weights = (0.25, 0.25, 0.25, 0.25)  # Default to BLEU-4
    
    # Normalize and tokenize
    hypothesis = normalize_dialectal_text(hypothesis).split()
    reference = normalize_dialectal_text(reference).split()
    
    # Calculate BLEU score
    smoothing = SmoothingFunction().method1
    return sentence_bleu([reference], hypothesis, weights=weights, smoothing_function=smoothing)

def calculate_dialectal_accuracy(text: str,
                               target_dialect: str,
                               dialectal_features: Dict[str, List[str]]) -> float:
    """
    Calculate dialectal accuracy.
    
    Args:
        text: Generated text
        target_dialect: Target dialect
        dialectal_features: Expected dialectal features
        
    Returns:
        Dialectal accuracy score (0-1)
    """
    # Detect dialect
    detected_dialect, confidence = detect_dialect(text)
    
    # Base score from dialect detection
    base_score = 1.0 if detected_dialect == target_dialect else 0.0
    
    # Feature matching score
    feature_score = 0.0
    total_features = 0
    
    for feature_list in dialectal_features.values():
        total_features += len(feature_list)
        for feature in feature_list:
            if feature in text:
                feature_score += 1
    
    feature_score = feature_score / total_features if total_features > 0 else 0.0
    
    # Combine scores
    return 0.7 * base_score + 0.3 * feature_score

def evaluate_cultural_preservation(generated_text: str,
                                 reference_text: str,
                                 cultural_elements: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Comprehensive cultural preservation evaluation.
    
    Args:
        generated_text: Generated text
        reference_text: Reference text
        cultural_elements: Expected cultural elements
        
    Returns:
        Dictionary of evaluation metrics
    """
    return {
        'cpm': calculate_cpm(generated_text, cultural_elements),
        'bleu': calculate_bleu(generated_text, reference_text),
        'category_scores': _calculate_category_scores(generated_text, cultural_elements)
    }

def _calculate_category_scores(text: str,
                             cultural_elements: Dict[str, List[str]]) -> Dict[str, float]:
    """Calculate preservation scores by category."""
    return {
        category: calculate_cpm(text, {category: elements})
        for category, elements in cultural_elements.items()
    }

def evaluate_linguistic_fluency(text: str,
                              reference: Optional[str] = None) -> Dict[str, float]:
    """
    Evaluate linguistic fluency.
    
    Args:
        text: Generated text
        reference: Optional reference text
        
    Returns:
        Dictionary of fluency metrics
    """
    metrics = {
        'avg_sentence_length': _calculate_avg_sentence_length(text),
        'vocabulary_diversity': _calculate_vocabulary_diversity(text),
        'punctuation_balance': _check_punctuation_balance(text)
    }
    
    if reference:
        metrics['bleu'] = calculate_bleu(text, reference)
    
    return metrics

def _calculate_avg_sentence_length(text: str) -> float:
    """Calculate average sentence length."""
    sentences = text.split('.')
    lengths = [len(s.split()) for s in sentences if s.strip()]
    return np.mean(lengths) if lengths else 0

def _calculate_vocabulary_diversity(text: str) -> float:
    """Calculate vocabulary diversity score."""
    words = normalize_dialectal_text(text).split()
    if not words:
        return 0.0
    return len(set(words)) / len(words)

def _check_punctuation_balance(text: str) -> float:
    """Check balance of punctuation marks."""
    paired_marks = {
        '(': ')',
        '[': ']',
        '{': '}',
        '"': '"',
        "'": "'"
    }
    
    stack = []
    errors = 0
    total_marks = 0
    
    for char in text:
        if char in paired_marks:
            stack.append(char)
            total_marks += 1
        elif char in paired_marks.values():
            if not stack or paired_marks[stack[-1]] != char:
                errors += 1
            else:
                stack.pop()
            total_marks += 1
    
    errors += len(stack)  # Count unclosed marks
    return 1 - (errors / total_marks if total_marks > 0 else 0)
