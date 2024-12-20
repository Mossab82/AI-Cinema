# AI-Cinema Examples

This directory contains practical examples of using AI-Cinema for various scenarios.

## Basic Usage

### Generating a Simple Scenario

```python
from ai_cinema import AICinema

# Initialize the framework
cinema = AICinema()

# Define cultural context
cultural_context = {
    'proverbs': [
        'الصبر مفتاح الفرج',
        'العين بصيرة واليد قصيرة'
    ],
    'customs': [
        'مجلس',
        'قهوة عربية',
        'تقبيل رأس الوالدين'
    ],
    'values': [
        'احترام الكبير',
        'الحفاظ على التقاليد'
    ]
}

# Generate scenario
scenario = cinema.generate_scenario(
    theme="family_reconciliation",
    dialect="gulf",
    cultural_context=cultural_context
)

print(scenario)
```

### Adapting to Different Dialects

```python
# Original Gulf scenario
gulf_scenario = cinema.generate_scenario(
    theme="family_conflict",
    dialect="gulf",
    cultural_context=cultural_context
)

# Adapt to Egyptian dialect
egyptian_scenario = cinema.adapt_scenario(
    scenario=gulf_scenario,
    target_dialect="egyptian",
    preserve_cultural_elements=True
)
```

## Advanced Examples

### Complex Character Interactions

```python
# Define characters
characters = [
    {
        "name": "سلطان",
        "role": "father",
        "age": 60,
        "background": {
            "occupation": "تاجر تقليدي",
            "values": ["التمسك بالتقاليد", "العمل الجاد"]
        }
    },
    {
        "name": "خالد",
        "role": "son",
        "age": 35,
        "background": {
            "occupation": "رجل أعمال عصري",
            "values": ["التطور", "الانفتاح"]
        }
    }
]

# Generate scenario with complex interactions
scenario = cinema.generate_scenario(
    theme="generational_conflict",
    dialect="gulf",
    cultural_context=cultural_context,
    characters=characters,
    min_scenes=5,
    max_scenes=8
)
```

### Cultural Integration Examples

```python
from ai_cinema.cultural import CulturalIntegrator

integrator = CulturalIntegrator()

# Define traditional elements
traditional_elements = {
    'proverbs': ['الأصول ما تنكسر'],
    'customs': ['مجلس العائلة'],
    'values': ['احترام الكبير']
}

# Modern context
modern_context = {
    'setting': 'شركة حديثة',
    'time_period': 'معاصر',
    'social_context': 'عمل'
}

# Integrate traditional elements in modern context
integrated_scene = integrator.integrate(
    text=modern_scene,
    traditional_elements=traditional_elements,
    modern_context=modern_context
)
```

### Dialogue Generation

```python
from ai_cinema.generation import DialogueGenerator

dialogue_gen = DialogueGenerator()

# Define character voices
voices = {
    'سلطان': {
        'dialect': 'gulf',
        'patterns': ['استخدام الأمثال', 'نبرة حكيمة'],
        'background': traditional_elements
    },
    'خالد': {
        'dialect': 'gulf_modern',
        'patterns': ['مصطلحات عصرية', 'لغة عملية'],
        'background': {'values': ['التطور', 'الحداثة']}
    }
}

# Generate dialogue
dialogue = dialogue_gen.generate(
    characters=voices,
    context=modern_context,
    num_turns=5
)
```

## Evaluation Examples

### Measuring Cultural Preservation

```python
from ai_cinema.utils.evaluation import evaluate_cultural_preservation

# Evaluate generated content
scores = evaluate_cultural_preservation(
    generated_text=scenario['scenes'][0]['text'],
    reference_text=reference_scene,
    cultural_elements=cultural_context
)

print(f"Cultural Preservation Score: {scores['cpm']}")
print(f"BLEU Score: {scores['bleu']}")
```

### Dialectal Accuracy Evaluation

```python
from ai_cinema.utils.evaluation import calculate_dialectal_accuracy

# Define dialectal features
gulf_features = {
    'pronouns': ['انا', 'احنا', 'انت'],
    'verbs': ['يروح', 'يجي', 'يبي'],
    'markers': ['هالـ', 'وش', 'كيف']
}

# Evaluate dialectal accuracy
accuracy = calculate_dialectal_accuracy(
    text=scenario['scenes'][0]['dialogue'],
    target_dialect="gulf",
    dialectal_features=gulf_features
)

print(f"Dialectal Accuracy: {accuracy}")
```

## Advanced Customization

### Custom Cultural Patterns

```python
from ai_cinema.cultural import PatternMatcher

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

# Create custom matcher
matcher = PatternMatcher(custom_patterns)

# Use in generation
scenario = cinema.generate_scenario(
    theme="business_family",
    dialect="gulf",
    cultural_context={**cultural_context, **custom_patterns},
    pattern_matcher=matcher
)
```

### Custom Evaluation Metrics

```python
def calculate_custom_score(text: str, criteria: dict) -> float:
    # Custom evaluation logic
    return score

# Use in evaluation
custom_evaluation = {
    'cultural_score': calculate_cpm(text, cultural_elements),
    'dialect_score': calculate_dialectal_accuracy(text, dialect, features),
    'custom_score': calculate_custom_score(text, custom_criteria)
}
```
