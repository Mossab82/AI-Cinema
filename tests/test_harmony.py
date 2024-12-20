"""Tests for the Harmony Function component."""

import pytest
import torch
from ai_cinema.models.harmony import HarmonyFunction

@pytest.fixture
def harmony_function():
    """Fixture for harmony function instance."""
    return HarmonyFunction(
        initial_alpha=0.6,
        initial_beta=0.2,
        learning_rate=0.001
    )

@pytest.fixture
def sample_data():
    """Fixture for sample test data."""
    return {
        'text': """
        يجلس الأب في المجلس، محاطاً بأبنائه. يقول بحكمة: "يا أولادي، الأصول ما تنكسر."
        """,
        'cultural_elements': {
            'settings': ['مجلس', 'ديوانية'],
            'proverbs': ['الأصول ما تنكسر', 'الصبر مفتاح الفرج'],
            'relationships': ['الأب', 'الابن', 'العائلة']
        },
        'dialect': 'gulf',
        'dialectal_features': {
            'pronouns': ['انته', 'انتي', 'احنا'],
            'verbs': ['يبي', 'يروح', 'يجي'],
            'expressions': ['ما عليه', 'ان شاء الله']
        }
    }

def test_harmony_function_initialization(harmony_function):
    """Test proper initialization of harmony function."""
    assert harmony_function is not None
    assert isinstance(harmony_function.alpha, torch.nn.Parameter)
    assert isinstance(harmony_function.beta, torch.nn.Parameter)
    assert 0 <= harmony_function.alpha.item() <= 1
    assert 0 <= harmony_function.beta.item() <= 1

def test_cultural_score_computation(harmony_function, sample_data):
    """Test computation of cultural preservation score."""
    score = harmony_function.compute_cultural_score(
        sample_data['text'],
        sample_data['cultural_elements']
    )
    
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1
    
    # Score should be higher for text with more cultural elements
    text_with_fewer_elements = "اجتمع الناس في المكان"
    lower_score = harmony_function.compute_cultural_score(
        text_with_fewer_elements,
        sample_data['cultural_elements']
    )
    
    assert score > lower_score

def test_linguistic_score_computation(harmony_function, sample_data):
    """Test computation of linguistic fluency score."""
    score = harmony_function.compute_linguistic_score(
        sample_data['text']
    )
    
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1
    
    # Score should be lower for text with grammatical errors
    text_with_errors = "الأب يجلسون في المجلس"  # Agreement error
    lower_score = harmony_function.compute_linguistic_score(text_with_errors)
    
    assert score > lower_score

def test_dialectal_score_computation(harmony_function, sample_data):
    """Test computation of dialectal authenticity score."""
    score = harmony_function.compute_dialectal_score(
        sample_data['text'],
        sample_data['dialect'],
        sample_data['dialectal_features']
    )
    
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1

def test_harmony_score_computation(harmony_function, sample_data):
    """Test computation of overall harmony score."""
    score, components = harmony_function(
        text=sample_data['text'],
        cultural_elements=sample_data['cultural_elements'],
        dialect=sample_data['dialect'],
        dialectal_features=sample_data['dialectal_features']
    )
    
    assert isinstance(score, torch.Tensor)
    assert 0 <= score.item() <= 1
    
    # Check score components
    assert 'cultural_score' in components
    assert 'linguistic_score' in components
    assert 'dialectal_score' in components
    
    # Verify weighted combination
    expected_score = (
        harmony_function.alpha.item() * components['cultural_score'] +
        (1 - harmony_function.alpha.item()) * components['linguistic_score'] +
        harmony_function.beta.item() * components['dialectal_score']
    )
    
    assert abs(score.item() - expected_score) < 1e-5

def test_parameter_updates(harmony_function, sample_data):
    """Test parameter updates during training."""
    initial_alpha = harmony_function.alpha.item()
    initial_beta = harmony_function.beta.item()
    
    # Simulate parameter update
    optimizer = torch.optim.Adam(harmony_function.parameters(), lr=0.01)
    
    score, _ = harmony_function(
        text=sample_data['text'],
        cultural_elements=sample_data['cultural_elements'],
        dialect=sample_data['dialect'],
        dialectal_features=sample_data['dialectal_features']
    )
    
    # Use score for optimization
    loss = 1 - score  # Maximize score
    loss.backward()
    optimizer.step()
    
    # Parameters should be updated
    assert harmony_function.alpha.item() != initial_alpha
    assert harmony_function.beta.item() != initial_beta

@pytest.mark.parametrize("alpha,beta", [
    (0.3, 0.3),
    (0.5, 0.2),
    (0.7, 0.1)
])
def test_different_weight_configurations(alpha, beta, sample_data):
    """Test harmony function with different weight configurations."""
    harmony_fn = HarmonyFunction(
        initial_alpha=alpha,
        initial_beta=beta
    )
    
    score, components = harmony_fn(
        text=sample_data['text'],
        cultural_elements=sample_data['cultural_elements'],
        dialect=sample_data['dialect'],
        dialectal_features=sample_data['dialectal_features']
    )
    
    # Verify score reflects weight configuration
    weighted_sum = (
        alpha * components['cultural_score'] +
        (1 - alpha) * components['linguistic_score'] +
        beta * components['dialectal_score']
    )
    
    assert abs(score.item() - weighted_sum) < 1e
