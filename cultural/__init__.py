"""
Cultural components for AI-Cinema.
Handles pattern recognition and cultural integration.
"""

from .patterns import (
    CulturalPatternDetector,
    PatternMatcher,
    MoralPatternMatcher,
    ArchetypePatternMatcher,
    DialectalPatternMatcher
)

from .integration import (
    CulturalIntegrator,
    CulturalAdapter,
    CulturalContext,
    CulturalElement
)

__all__ = [
    'CulturalPatternDetector',
    'PatternMatcher',
    'MoralPatternMatcher',
    'ArchetypePatternMatcher',
    'DialectalPatternMatcher',
    'CulturalIntegrator',
    'CulturalAdapter',
    'CulturalContext',
    'CulturalElement'
]
