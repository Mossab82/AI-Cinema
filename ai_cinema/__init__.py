
from ai_cinema.models.harmony import HarmonyFunction
from ai_cinema.cultural.patterns import CulturalPatternDetector
from ai_cinema.generation.scenario import ScenarioGenerator
from ai_cinema.generation.dialogue import DialogueGenerator

__version__ = "0.1.0"

class AICinema:
    """Main class for the AI-Cinema framework."""
    
    def __init__(self, config_path=None):
        """
        Initialize AI-Cinema framework.
        
        Args:
            config_path: Optional path to configuration file
        """
        from ai_cinema.config import load_config
        self.config = load_config(config_path) if config_path else {}
        
        # Initialize core components
        self.harmony_function = HarmonyFunction()
        self.pattern_detector = CulturalPatternDetector()
        self.scenario_generator = ScenarioGenerator()
        self.dialogue_generator = DialogueGenerator()
    
    def generate_scenario(self, 
                         theme: str,
                         dialect: str,
                         cultural_context: dict,
                         **kwargs) -> dict:
        """
        Generate a complete movie scenario.
        
        Args:
            theme: Main theme of the scenario
            dialect: Target Arabic dialect
            cultural_context: Dictionary of cultural elements to incorporate
            **kwargs: Additional generation parameters
            
        Returns:
            Dictionary containing the generated scenario
        """
        return self.scenario_generator.generate(
            theme=theme,
            dialect=dialect,
            cultural_context=cultural_context,
            **kwargs
        )
    
    def adapt_scenario(self,
                      scenario: dict,
                      target_dialect: str,
                      preserve_cultural_elements: bool = True) -> dict:
        """
        Adapt an existing scenario to a different dialect.
        
        Args:
            scenario: Original scenario
            target_dialect: Target dialect for adaptation
            preserve_cultural_elements: Whether to maintain cultural markers
            
        Returns:
            Adapted scenario in target dialect
        """
        return self.scenario_generator.adapt(
            scenario=scenario,
            target_dialect=target_dialect,
            preserve_cultural_elements=preserve_cultural_elements
        )
