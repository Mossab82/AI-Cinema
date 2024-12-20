# AI-Cinema

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![AraBERT](https://img.shields.io/badge/AraBERT-base-red.svg)](https://github.com/aub-mind/arabert)

A hybrid framework for generating culturally authentic Arabic movie scenarios through neural-symbolic integration. AI-Cinema combines traditional storytelling elements with modern AI techniques to create compelling narratives that preserve cultural authenticity while meeting contemporary production standards.

## Features

### Cultural Integration
- Comprehensive cultural knowledge bases with 12,500+ annotated verbs
- Advanced pattern recognition for preserving traditional elements
- Dynamic cultural marker integration
- Traditional storytelling pattern preservation

### Dialectal Support
- Support for 25 Arabic regional dialects
- City-level dialectal adaptation
- Code-switching handling
- Regional idiom preservation

### Generation Capabilities
- Complete movie scenario generation
- Scene and dialogue creation
- Character development
- Production-ready formatting

## Performance Metrics

- Cultural Preservation: 82.3% CPM
- Linguistic Fluency: 32.76 BLEU-4
- Dialectal Accuracy: 92.3%
- Human Evaluation: 4.3/5.0

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.9 or higher
- 8GB+ RAM recommended

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-cinema.git
cd ai-cinema

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from ai_cinema import AICinema

# Initialize the framework
cinema = AICinema()

# Generate a basic scenario
scenario = cinema.generate_scenario(
    theme="family_conflict",
    dialect="gulf",
    cultural_context="traditional",
    output_format="screenplay"
)

# Print the generated scenario
print(scenario)
```

## Advanced Usage

### Customizing Cultural Elements

```python
cultural_elements = {
    "proverbs": ["الصبر مفتاح الفرج", "العين بصيرة واليد قصيرة"],
    "archetypes": ["الحكيم", "الأب المتسلط", "الابن العاق"],
    "settings": ["مجلس", "ديوانية", "بيت عربي تقليدي"]
}

scenario = cinema.generate_scenario(
    theme="family_reconciliation",
    cultural_elements=cultural_elements,
    min_length=1000,
    max_length=2000
)
```

### Dialectal Adaptation

```python
# Generate same scenario in different dialects
dialects = ["gulf", "egyptian", "levantine"]

for dialect in dialects:
    localized_scenario = cinema.adapt_scenario(
        scenario,
        target_dialect=dialect,
        preserve_cultural_elements=True
    )
```

## Resource Management

AI-Cinema uses several key resources:

- **ArabicVerbNet**: 12,500 contextually annotated verbs
- **ArabicNameNet**: 3,653 culturally annotated names
- **MADAR Corpus**: 49,000 dialectal sentences
- **Contemporary Scripts**: 1,200 annotated scenarios

## Model Architecture

The framework implements a three-tier architecture:

1. **Data Layer**
   - Resource management
   - Cultural knowledge retrieval
   - Dialectal adaptation

2. **Cultural Embedding Layer**
   - Pattern recognition
   - Cultural integration
   - Context maintenance

3. **Generation Layer**
   - Harmony Function
   - Neural-symbolic processing
   - Scenario formatting

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest tests/

# Run linting
flake8 ai_cinema/
```

## Citation

If you use AI-Cinema in your research, please cite our paper:

```bibtex
@inproceedings{ibrahim2024ai-cinema,
  title={AI-Cinema: A Hybrid Framework for Arabic Movie Scenario Generation with Traditional Storytelling and Cultural Dialogues},
  author={Ibrahim, Mossab and Gervás, Pablo and Méndez, Gonzalo},
  booktitle={Proceedings of the Twenty-Fifth International Joint Conference on Artificial Intelligence (IJCAI-25)},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- The AraBERT team for their foundational work in Arabic NLP
- The MADAR project for their comprehensive dialectal corpus
- Our cultural consultants and screenwriting experts for their valuable feedback

## Contact

For questions and feedback:
- **Mossab Ibrahim** - [mibrahim@ucm.es](mailto:mibrahim@ucm.es)
- **Project Website**: [https://ai-cinema.github.io](https://ai-cinema.github.io)
