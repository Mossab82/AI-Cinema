# Getting Started with AI-Cinema

This tutorial will guide you through the basic concepts and usage of AI-Cinema.

## Installation

### 2. Dialectal Accuracy

```python
from ai_cinema.utils.evaluation import calculate_dialectal_accuracy

# Define dialectal features
gulf_features = {
    'pronouns': ['انا', 'احنا', 'انت'],
    'verbs': ['يروح', 'يجي', 'يبي'],
    'markers': ['هالـ', 'وش', 'كيف']
}

accuracy = calculate_dialectal_accuracy(
    text=scenario['scenes'][0]['dialogue'],
    target_dialect="gulf",
    dialectal_features=gulf_features
)

print(f"Dialectal Accuracy: {accuracy}")
```

## Advanced Usage

### 1. Custom Cultural Patterns

```python
from ai_cinema.cultural.patterns import PatternMatcher

# Define custom patterns
custom_patterns = {
    'business_wisdom': [
        'التاجر الشاطر يخلي رزقه دايم',
        'الربح في الحركة والبركة في السكون'
    ],
    'modern_values': [
        'التطور مع الحفاظ على الهوية',
        'الأصالة والمعاصرة'
    ]
}

matcher = PatternMatcher(custom_patterns)
```

### 2. Scene Configuration

```python
# Configure scene generation
scene_config = {
    "min_length": 200,
    "max_length": 1000,
    "required_elements": [
        "dialogue",
        "setting_description",
        "character_action"
    ],
    "style": "contemporary"
}

scenario = cinema.generate_scenario(
    theme="family_business",
    config=scene_config,
    dialect="gulf",
    cultural_context=cultural_context
)
```

### 3. Character Voices

```python
from ai_cinema.generation import CharacterVoice, DialogueGenerator

# Create character voices
father_voice = CharacterVoice(
    name="سلطان",
    age=60,
    role="father",
    dialect="gulf",
    speech_patterns=["formal", "traditional"],
    cultural_background={
        "values": ["tradition", "authority"],
        "expressions": ["proverbs", "religious"]
    }
)

son_voice = CharacterVoice(
    name="خالد",
    age=35,
    role="son",
    dialect="gulf",
    speech_patterns=["modern", "business"],
    cultural_background={
        "values": ["progress", "innovation"],
        "expressions": ["technical", "english_borrowing"]
    }
)

# Generate dialogue with character voices
dialogue_gen = DialogueGenerator()
dialogue = dialogue_gen.generate(
    characters=[father_voice, son_voice],
    context="business_meeting",
    num_turns=5
)
```

## Best Practices

### 1. Cultural Context

- Always provide comprehensive cultural context
- Include both traditional and modern elements
- Balance between different types of cultural markers

```python
cultural_context = {
    'traditional': {
        'proverbs': [...],
        'customs': [...],
        'values': [...]
    },
    'modern': {
        'expressions': [...],
        'situations': [...],
        'concepts': [...]
    }
}
```

### 2. Dialectal Usage

- Be consistent with dialect choice
- Consider audience and setting
- Mix MSA where appropriate

```python
# Mixed dialect example
scenario = cinema.generate_scenario(
    theme="business_meeting",
    dialect={
        "narrative": "msa",
        "dialogue": "gulf",
        "formal_speech": "msa"
    },
    cultural_context=cultural_context
)
```

### 3. Scene Structure

- Follow proper screenplay format
- Balance dialogue and action
- Maintain cultural authenticity

```python
scene_structure = {
    "heading": "INT. مجلس العائلة - مساءً",
    "description": "وصف المشهد وتفاصيل المكان",
    "action": [
        "وصف الحركة والأحداث",
        "تفاصيل تعبيرات الوجه والحركات"
    ],
    "dialogue": [
        {"character": "سلطان", "text": "..."},
        {"character": "خالد", "text": "..."}
    ]
}
```

## Common Issues and Solutions

### 1. Cultural Integration

Problem: Weak cultural presence in generated content

Solution:
```python
from ai_cinema.cultural import CulturalEnhancer

enhancer = CulturalEnhancer(
    min_elements_per_scene=3,
    balance_traditional_modern=True
)

enhanced_scenario = enhancer.enhance(
    scenario,
    cultural_context,
    preserve_structure=True
)
```

### 2. Dialect Consistency

Problem: Mixed or inconsistent dialects

Solution:
```python
from ai_cinema.utils import DialectNormalizer

normalizer = DialectNormalizer(target_dialect="gulf")
normalized_text = normalizer.normalize(
    text,
    preserve_formal=True,
    handle_code_switching=True
)
```

### 3. Character Voice

Problem: Weak character differentiation

Solution:
```python
from ai_cinema.generation import VoiceEnhancer

voice_enhancer = VoiceEnhancer()
enhanced_dialogue = voice_enhancer.enhance(
    dialogue,
    character_profiles=character_voices,
    strengthen_traits=True
)
```

## Next Steps

1. Explore advanced features in the [API Reference](../api/README.md)
2. Check out more examples in the [Examples](../examples/README.md) directory
3. Join our community:
   - GitHub Discussions
   - Discord Server
   - Contributing GuidelinesPrerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- 8GB+ RAM recommended

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install AI-Cinema
pip install ai-cinema
```

## Basic Concepts

### 1. Cultural Elements
AI-Cinema works with several types of cultural elements:
- Proverbs (أمثال)
- Customs (عادات)
- Values (قيم)
- Character archetypes (شخصيات نمطية)
- Settings (أماكن)

### 2. Dialectal Support
The framework supports major Arabic dialects:
- Modern Standard Arabic (MSA)
- Gulf Arabic
- Egyptian Arabic
- Levantine Arabic

### 3. Generation Components
- Scenario Generation
- Scene Creation
- Dialogue Generation
- Cultural Integration

## Your First Scenario

### 1. Initialize the Framework
```python
from ai_cinema import AICinema

cinema = AICinema()
```

### 2. Define Cultural Context
```python
cultural_context = {
    'proverbs': [
        'الصبر مفتاح الفرج',
        'العين بصيرة واليد قصيرة'
    ],
    'customs': [
        'مجلس',
        'قهوة عربية'
    ],
    'values': [
        'احترام الكبير',
        'الحفاظ على التقاليد'
    ]
}
```

### 3. Generate Scenario
```python
scenario = cinema.generate_scenario(
    theme="family_reconciliation",
    dialect="gulf",
    cultural_context=cultural_context
)
```

### 4. Working with the Output
```python
# Access individual scenes
for scene in scenario['scenes']:
    print(f"Scene: {scene['heading']}")
    print(f"Action: {scene['action']}")
    print("Dialogue:")
    for line in scene['dialogue']:
        print(f"{line['character']}: {line['text']}")
```

## Customization

### 1. Character Creation
```python
characters = [
    {
        "name": "سلطان",
        "role": "father",
        "age": 60,
        "background": {
            "occupation": "تاجر تقليدي",
            "values": ["التمسك بالتقاليد"]
        }
    }
]

scenario = cinema.generate_scenario(
    theme="family_conflict",
    characters=characters,
    dialect="gulf",
    cultural_context=cultural_context
)
```

### 2. Dialectal Adaptation
```python
# Generate in Gulf dialect
gulf_scenario = cinema.generate_scenario(
    theme="family_conflict",
    dialect="gulf",
    cultural_context=cultural_context
)

# Adapt to Egyptian dialect
egyptian_scenario = cinema.adapt_scenario(
    scenario=gulf_scenario,
    target_dialect="egyptian"
)
```

### 3. Cultural Enhancement
```python
from ai_cinema.cultural import CulturalIntegrator

integrator = CulturalIntegrator()

# Enhance cultural elements
enhanced_scene = integrator.enhance(
    text=original_scene,
    cultural_context=cultural_context,
    min_elements=3
)
```

## Evaluation

### 1. Cultural Preservation
```python
from ai_cinema.utils.evaluation import evaluate_cultural_preservation

scores = evaluate_cultural_preservation(
    generated_text=scenario['scenes'][0]['text'],
    cultural_elements=cultural_context
)

print(f"Cultural Preservation Score: {scores['cpm']}")
```

###
