"""Tests for the Cultural Pattern Recognition system."""

import pytest
import torch
from ai_cinema.cultural.patterns import CulturalPatternDetector, CulturalPattern

# Sample test data
SAMPLE_MORAL_TEXT = """
يا ولدي خالد، أدري إن قراراتي ما عجبتك، بس لازم تفهم إن الأصول ما تنكسر
"""

SAMPLE_ARCHETYPE_TEXT = """
الشيخ الحكيم جلس في مجلسه، محاطاً بالمريدين، يروي لهم قصص الأولين
"""

SAMPLE_DIALECTAL_TEXT = """
مو معقول يا خوي، هالسالفة ما تمشي جذي
"""

@pytest.fixture
def pattern_detector():
    """Fixture for pattern detector instance."""
    return CulturalPatternDetector()

@pytest.fixture
def cultural_knowledge():
    """Fixture for cultural knowledge base."""
    return {
        'moral_teachings': [
            'الأصول ما تنكسر',
            'الصبر مفتاح الفرج'
        ],
        'archetypes': [
            'الشيخ الحكيم',
            'الأب المتسلط'
        ],
        'dialectal_phrases': [
            'مو معقول',
            'هالسالفة'
        ]
    }

def test_pattern_detector_initialization(pattern_detector):
    """Test proper initialization of pattern detector."""
    assert pattern_detector is not None
    assert isinstance(pattern_detector, CulturalPatternDetector)
    assert hasattr(pattern_detector, 'moral_attention')
    assert hasattr(pattern_detector, 'archetype_attention')
    assert hasattr(pattern_detector, 'dialectal_attention')

def test_moral_pattern_detection(pattern_detector, cultural_knowledge):
    """Test detection of moral teachings and values."""
    patterns = pattern_detector(SAMPLE_MORAL_TEXT, cultural_knowledge)
    
    # Should detect at least one moral pattern
    moral_patterns = [p for p in patterns if p.pattern_type == 'moral']
    assert len(moral_patterns) > 0
    
    # Check pattern properties
    pattern = moral_patterns[0]
    assert isinstance(pattern, CulturalPattern)
    assert pattern.confidence > 0.5
    assert 'الأصول' in pattern.content

def test_archetype_detection(pattern_detector, cultural_knowledge):
    """Test detection of character and situation archetypes."""
    patterns = pattern_detector(SAMPLE_ARCHETYPE_TEXT, cultural_knowledge)
    
    # Should detect the wise elder archetype
    archetype_patterns = [p for p in patterns if p.pattern_type == 'archetype']
    assert len(archetype_patterns) > 0
    
    pattern = archetype_patterns[0]
    assert 'الشيخ الحكيم' in pattern.content
    assert pattern.confidence > 0.5

def test_dialectal_pattern_detection(pattern_detector, cultural_knowledge):
    """Test detection of dialect-specific patterns."""
    patterns = pattern_detector(SAMPLE_DIALECTAL_TEXT, cultural_knowledge)
    
    # Should detect Gulf Arabic dialectal patterns
    dialectal_patterns = [p for p in patterns if p.pattern_type == 'dialectal']
    assert len(dialectal_patterns) > 0
    
    pattern = dialectal_patterns[0]
    assert any(phrase in pattern.content for phrase in ['مو معقول', 'هالسالفة'])
    assert pattern.confidence > 0.5

def test_cultural_mask_creation(pattern_detector):
    """Test creation of cultural attention mask."""
    # Create sample input
    text = "يجلس الشيخ في مجلسه"
    inputs = pattern_detector.tokenizer(text, return_tensors="pt")
    embeddings = pattern_detector.base_model(**inputs).last_hidden_state
    
    cultural_context = {
        'traditional_settings': [2, 3],  # Position of "مجلس" tokens
        'archetypes': [1]  # Position of "الشيخ" token
    }
    
    mask = pattern_detector.create_cultural_mask(embeddings, cultural_context)
    
    # Check mask properties
    assert isinstance(mask, torch.Tensor)
    assert mask.shape[-2:] == embeddings.shape[:2]  # Correct attention mask shape
    assert torch.any(mask > 0)  # Mask contains positive values for cultural elements

def test_pattern_confidence_thresholds(pattern_detector, cultural_knowledge):
    """Test confidence thresholds for pattern detection."""
    # Text with weak/ambiguous patterns
    weak_text = "اجتمع الناس في المكان"
    patterns = pattern_detector(weak_text, cultural_knowledge)
    
    # Should not detect patterns with low confidence
    assert all(p.confidence > 0.5 for p in patterns)

def test_multiple_pattern_detection(pattern_detector, cultural_knowledge):
    """Test detection of multiple patterns in the same text."""
    complex_text = """
    جلس الشيخ الحكيم في مجلسه وقال: "الصبر مفتاح الفرج، يا ولدي"
    """
    patterns = pattern_detector(complex_text, cultural_knowledge)
    
    # Should detect both archetype and moral patterns
    pattern_types = [p.pattern_type for p in patterns]
    assert 'archetype' in pattern_types
    assert 'moral' in pattern_types

def test_pattern_context_extraction(pattern_detector, cultural_knowledge):
    """Test extraction of contextual information around patterns."""
    text = SAMPLE_MORAL_TEXT
    patterns = pattern_detector(text, cultural_knowledge)
    
    for pattern in patterns:
        # Context should be non-empty and contain the pattern
        assert pattern.context
        assert pattern.content in pattern.context
        assert len(pattern.context) >= len(pattern.content)

@pytest.mark.parametrize("dialect,expected_phrases", [
    ("gulf", ["مو معقول", "هالسالفة"]),
    ("egyptian", ["مش معقول", "الحكاية دي"]),
    ("levantine", ["مش معقول", "هالقصة"])
])
def test_dialect_specific_patterns(pattern_detector, cultural_knowledge, dialect, expected_phrases):
    """Test detection of dialect-specific patterns across different dialects."""
    cultural_knowledge['dialectal_phrases'] = expected_phrases
    text = f"قال: {expected_phrases[0]}، {expected_phrases[1]}"
    
    patterns = pattern_detector(text, cultural_knowledge)
    dialectal_patterns = [p for p in patterns if p.pattern_type == 'dialectal']
    
    assert len(dialectal_patterns) > 0
    assert any(phrase in dialectal_patterns[0].content for phrase in expected_phrases)
