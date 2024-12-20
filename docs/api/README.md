# AI-Cinema API Reference

## Core Components

### AICinema Class

The main class for interacting with the AI-Cinema framework.

```python
from ai_cinema import AICinema

cinema = AICinema()
```

#### Methods

##### generate_scenario
```python
def generate_scenario(
    theme: str,
    dialect: str,
    cultural_context: dict,
    **kwargs
) -> dict
```

Generates a complete movie scenario.

**Parameters:**
- `theme` (str): Main theme of the scenario
- `dialect` (str): Target Arabic dialect
- `cultural_context` (dict): Dictionary of cultural elements to incorporate
- `**kwargs`: Additional generation parameters

**Returns:**  
Dictionary containing the generated scenario

##### adapt_scenario
```python
def adapt_scenario(
    scenario: dict,
    target_dialect: str,
    preserve_cultural_elements: bool = True
) -> dict
```

Adapts an existing scenario to a different dialect.

**Parameters:**
- `scenario` (dict): Original scenario
- `target_dialect` (str): Target dialect for adaptation
- `preserve_cultural_elements` (bool): Whether to maintain cultural markers

### Cultural Components

#### CulturalPatternDetector

Detects cultural patterns in text.

```python
from ai_cinema.cultural import CulturalPatternDetector

detector = CulturalPatternDetector()
```

#### Methods

##### detect_patterns
```python
def detect_patterns(
    text: str,
    cultural_context: dict
) -> List[PatternMatch]
```

**Parameters:**
- `text` (str): Input text
- `cultural_context` (dict): Cultural context dictionary

**Returns:**  
List of detected cultural patterns

### Generation Components

#### ScenarioGenerator

Generates complete movie scenarios.

```python
from ai_cinema.generation import ScenarioGenerator

generator = ScenarioGenerator()
```

#### Methods

##### generate
```python
def generate(
    theme: str,
    cultural_context: dict,
    characters: List[dict],
    dialect: str,
    **kwargs
) -> ScenarioStructure
```

**Parameters:**
- `theme` (str): Main theme
- `cultural_context` (dict): Cultural elements
- `characters` (List[dict]): Character descriptions
- `dialect` (str): Target dialect

### Utility Functions

#### Dialectal Processing

```python
from ai_cinema.utils import (
    detect_dialect,
    convert_dialect,
    normalize_dialectal_text
)
```

#### Evaluation Metrics

```python
from ai_cinema.utils import (
    calculate_cpm,
    calculate_bleu,
    calculate_dialectal_accuracy
)
```

## Configuration

### Configuration File Structure

```yaml
resources:
  arabic_wordnet: "path/to/wordnet"
  arabic_verbnet: "path/to/verbnet"
  madar_corpus: "path/to/madar"

models:
  base_model: "aubmindlab/bert-base-arabert"
  tokenizer: "aubmindlab/bert-base-arabert"

generation:
  max_length: 1000
  num_beams: 5

dialects:
  supported:
    - "msa"
    - "gulf"
    - "egyptian"
    - "levantine"
  default: "msa"

cultural:
  min_elements: 3
  max_elements: 10
```

## Error Handling

### Common Errors

```python
from ai_cinema.exceptions import (
    CulturalContextError,
    DialectError,
    GenerationError
)
```

### Error Types

- `CulturalContextError`: Invalid cultural context
- `DialectError`: Unsupported dialect or conversion error
- `GenerationError`: Error during scenario generation

## Data Structures

### ScenarioStructure

```python
@dataclass
class ScenarioStructure:
    scenes: List[Dict[str, str]]
    characters: List[Dict[str, str]]
    settings: List[str]
    timeline: List[str]
```

### CharacterVoice

```python
@dataclass
class CharacterVoice:
    name: str
    age: int
    role: str
    dialect: str
    speech_patterns: List[str]
    cultural_background: Dict[str, List[str]]
```

### CulturalContext

```python
@dataclass
class CulturalContext:
    setting: str
    time_period: str
    social_context: str
    cultural_markers: List[str]
    dialect: str
```
