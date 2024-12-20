"""Tests for the Scenario Generation system."""

import pytest
import torch
import json
from pathlib import Path
from ai_cinema.generation.scenario import ScenarioGenerator
from ai_cinema.generation.dialogue import DialogueGenerator
from ai_cinema.models.harmony import HarmonyFunction

@pytest.fixture
def scenario_generator():
    """Fixture for scenario generator instance."""
    return ScenarioGenerator()

@pytest.fixture
def dialogue_generator():
    """Fixture for dialogue generator instance."""
    return DialogueGenerator()

@pytest.fixture
def sample_scenario_request():
    """Fixture for sample scenario generation request."""
    return {
        "theme": "family_conflict",
        "setting": "traditional_house",
        "characters": [
            {"name": "سلطان", "role": "father", "age": 60},
            {"name": "خالد", "role": "son", "age": 35},
            {"name": "الجد", "role": "grandfather", "age": 80}
        ],
        "dialect": "gulf",
        "cultural_context": {
            "proverbs": ["اللي ما يعرف أصله ضايع فرعه"],
            "customs": ["مجلس", "قهوة عربية", "تقبيل رأس الوالدين"],
            "values": ["احترام الكبير", "الحفاظ على التقاليد"]
        }
    }

@pytest.fixture
def expected_scene_structure():
    """Fixture for expected scene structure."""
    return {
        "scene_heading": "مجلس عائلي - داخلي - مساءً",
        "action": str,  # Type check for string
        "characters": list,  # Type check for list
        "dialogue": list,  # Type check for list
        "transitions": str  # Type check for string
    }

def test_scenario_generator_initialization(scenario_generator):
    """Test proper initialization of scenario generator."""
    assert hasattr(scenario_generator, 'harmony_function')
    assert isinstance(scenario_generator.harmony_function, HarmonyFunction)
    assert hasattr(scenario_generator, 'model')
    assert hasattr(scenario_generator, 'tokenizer')

def test_basic_scenario_generation(scenario_generator, sample_scenario_request):
    """Test basic scenario generation functionality."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    # Check basic structure
    assert isinstance(scenario, dict)
    assert 'scenes' in scenario
    assert len(scenario['scenes']) > 0
    
    # Check first scene structure
    first_scene = scenario['scenes'][0]
    assert 'heading' in first_scene
    assert 'action' in first_scene
    assert 'dialogue' in first_scene

def test_cultural_element_integration(scenario_generator, sample_scenario_request):
    """Test integration of cultural elements in generated scenario."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    # Check for cultural elements presence
    full_text = ' '.join([
        scene['heading'] + ' ' + scene['action'] + ' ' + 
        ' '.join(d['text'] for d in scene['dialogue'])
        for scene in scenario['scenes']
    ])
    
    # Verify cultural elements are present
    cultural_elements = (
        sample_scenario_request['cultural_context']['proverbs'] +
        sample_scenario_request['cultural_context']['customs'] +
        sample_scenario_request['cultural_context']['values']
    )
    
    matches = sum(1 for element in cultural_elements if element in full_text)
    assert matches >= len(cultural_elements) * 0.7  # At least 70% should be present

def test_dialectal_consistency(scenario_generator, sample_scenario_request):
    """Test dialectal consistency in generated dialogue."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    # Gulf Arabic dialectal markers
    gulf_markers = [
        'يا ولدي',
        'يبه',
        'ما عليه',
        'إن شاء الله',
        'هالشي'
    ]
    
    # Extract all dialogue
    dialogue_texts = []
    for scene in scenario['scenes']:
        dialogue_texts.extend(d['text'] for d in scene['dialogue'])
    
    # Check for dialectal markers
    total_markers = sum(1 for marker in gulf_markers if any(marker in text for text in dialogue_texts))
    assert total_markers >= len(gulf_markers) * 0.6  # At least 60% should be present

def test_narrative_coherence(scenario_generator, sample_scenario_request):
    """Test narrative coherence in generated scenario."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    # Check character consistency
    character_names = set(char['name'] for char in sample_scenario_request['characters'])
    speaking_characters = set()
    
    for scene in scenario['scenes']:
        for dialogue in scene['dialogue']:
            speaking_characters.add(dialogue['character'])
    
    # All defined characters should speak at least once
    assert character_names.issubset(speaking_characters)
    
    # Check scene progression
    assert len(scenario['scenes']) >= 3  # At least setup, conflict, resolution

def test_formatting_requirements(scenario_generator, sample_scenario_request):
    """Test screenplay formatting requirements."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    for scene in scenario['scenes']:
        # Scene heading format
        assert any(keyword in scene['heading'] for keyword in ['داخلي', 'خارجي'])
        assert any(time in scene['heading'] for time in ['صباحاً', 'مساءً', 'ليلاً'])
        
        # Action paragraphs
        assert len(scene['action']) <= 400  # Maximum action block length
        
        # Dialogue format
        for dialogue in scene['dialogue']:
            assert 'character' in dialogue
            assert 'text' in dialogue
            assert len(dialogue['text']) <= 200  # Maximum dialogue length

def test_character_arc_development(scenario_generator, sample_scenario_request):
    """Test character development and arc progression."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    # Track character dialogue across scenes
    character_lines = {char['name']: [] for char in sample_scenario_request['characters']}
    
    for scene in scenario['scenes']:
        for dialogue in scene['dialogue']:
            if dialogue['character'] in character_lines:
                character_lines[dialogue['character']].append(dialogue['text'])
    
    # Each main character should have significant dialogue
    for char_name, lines in character_lines.items():
        assert len(lines) >= 3  # At least 3 lines per main character

def test_generation_with_constraints(scenario_generator):
    """Test scenario generation with specific constraints."""
    constrained_request = {
        "theme": "family_conflict",
        "max_scenes": 3,
        "max_dialogue_per_scene": 5,
        "required_elements": ["مجلس", "شاي", "مصالحة"],
        "dialect": "gulf"
    }
    
    scenario = scenario_generator.generate(constrained_request)
    
    # Check constraints are met
    assert len(scenario['scenes']) <= constrained_request['max_scenes']
    
    for scene in scenario['scenes']:
        assert len(scene['dialogue']) <= constrained_request['max_dialogue_per_scene']
    
    # Check required elements
    full_text = ' '.join(str(item) for scene in scenario['scenes'] 
                        for item in [scene['heading'], scene['action']] + 
                        [d['text'] for d in scene['dialogue']])
    
    for element in constrained_request['required_elements']:
        assert element in full_text

def test_error_handling(scenario_generator):
    """Test error handling in scenario generation."""
    # Test with invalid dialect
    with pytest.raises(ValueError):
        scenario_generator.generate({"dialect": "invalid_dialect"})
    
    # Test with missing required fields
    with pytest.raises(ValueError):
        scenario_generator.generate({})
    
    # Test with invalid character configuration
    with pytest.raises(ValueError):
        scenario_generator.generate({
            "theme": "family_conflict",
            "characters": [{"name": "سلطان"}]  # Missing required character fields
        })

def test_generation_reproducibility(scenario_generator, sample_scenario_request):
    """Test reproducibility of scenario generation with same seed."""
    # Set seed
    torch.manual_seed(42)
    scenario1 = scenario_generator.generate(sample_scenario_request)
    
    # Reset seed
    torch.manual_seed(42)
    scenario2 = scenario_generator.generate(sample_scenario_request)
    
    # Check scenarios are identical
    assert scenario1 == scenario2

def test_integration_with_harmony_function(scenario_generator, sample_scenario_request):
    """Test integration with harmony function during generation."""
    scenario = scenario_generator.generate(sample_scenario_request)
    
    # Get harmony scores for generated content
    harmony_scores = []
    for scene in scenario['scenes']:
        score = scenario_generator.harmony_function(
            text=' '.join([scene['heading'], scene['action']] + 
                         [d['text'] for d in scene['dialogue']]),
            cultural_elements=sample_scenario_request['cultural_context'],
            dialect=sample_scenario_request['dialect'],
            dialectal_features={}  # Add appropriate dialectal features
        )
        harmony_scores.append(score)
    
    # Average harmony score should be above threshold
    avg_score = sum(score.item() for score in harmony_scores) / len(harmony_scores)
    assert avg_score >= 0.7  # Minimum harmony score threshold
