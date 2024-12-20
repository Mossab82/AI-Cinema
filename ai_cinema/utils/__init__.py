"""
Utility functions for AI-Cinema.
"""

from .dialectal import (
    detect_dialect,
    convert_dialect,
    normalize_dialectal_text,
    DIALECT_MAPPINGS
)

from .evaluation import (
    calculate_cpm,
    calculate_bleu,
    calculate_dialectal_accuracy,
    evaluate_cultural_preservation,
    evaluate_linguistic_fluency
)

__all__ = [
    'detect_dialect',
    'convert_dialect',
    'normalize_dialectal_text',
    'DIALECT_MAPPINGS',
    'calculate_cpm',
    'calculate_bleu',
    'calculate_dialectal_accuracy',
    'evaluate_cultural_preservation',
    'evaluate_linguistic_fluency'
]
