"""
Dialogue generation module for AI-Cinema.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizer

from ..cultural import CulturalPatternDetector, CulturalIntegrator
from ..models import HarmonyFunction

@dataclass
class DialogueConfig:
    """Configuration for dialogue generation."""
    max_length: int = 200
    num_beams: int = 5
    min_turns: int = 2
    max_turns: int = 10
    cultural_threshold: float = 0.7
    dialect_threshold: float = 0.8

@dataclass
class CharacterVoice:
    """Character voice profile."""
    name: str
    age: int
    role: str
    dialect: str
    speech_patterns: List[str]
    cultural_background: Dict[str, List[str]]

@dataclass
class DialogueContext:
    """Context for dialogue generation."""
    scene_description: str
    emotional_state: Dict[str, str]
    relationship_dynamics: Dict[Tuple[str, str], str]
    cultural_markers: List[str]

class DialogueGenerator(nn.Module):
    """Generates culturally authentic dialogue."""
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 cultural_detector: CulturalPatternDetector,
                 cultural_integrator: CulturalIntegrator,
                 harmony_function: HarmonyFunction,
                 config: Optional[DialogueConfig] = None):
        """
        Initialize dialogue generator.
        
        Args:
            model: Base language model
            tokenizer: Tokenizer
            cultural_detector: Cultural pattern detector
            cultural_integrator: Cultural integrator
            harmony_function: Harmony scoring function
            config: Generator configuration
        """
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.cultural_detector = cultural_detector
        self.cultural_integrator = cultural_integrator
        self.harmony_function = harmony_function
        self.config = config or DialogueConfig()
        
        # Dialogue-specific components
        self.character_encoder = nn.Linear(model.config.hidden_size, model.config.hidden_size // 2)
        self.context_encoder = nn.Linear(model.config.hidden_size, model.config.hidden_size // 2)
        self.dialogue_decoder = nn.GRU(
            model.config.hidden_size,
            model.config.hidden_size,
            num_layers=2,
            batch_first=True
        )
        self.response_projector = nn.Linear(model.config.hidden_size, model.config.vocab_size)
    
    def generate_dialogue(self,
                         characters: List[CharacterVoice],
                         context: DialogueContext,
                         cultural_context: Dict[str, List[str]],
                         num_turns: Optional[int] = None) -> List[Dict[str, str]]:
        """
        Generate dialogue sequence.
        
        Args:
            characters: List of characters
            context: Dialogue context
            cultural_context: Cultural elements to incorporate
            num_turns: Optional number of dialogue turns
            
        Returns:
            List of dialogue turns
        """
        if num_turns is None:
            num_turns = torch.randint(
                self.config.min_turns,
                self.config.max_turns + 1,
                (1,)
            ).item()
        
        # Initialize dialogue state
        dialogue_state = self._initialize_state(characters, context)
        
        # Generate dialogue turns
        dialogue = []
        for _ in range(num_turns):
            # Select next speaker
            speaker = self._select_next_speaker(dialogue_state, dialogue)
            
            # Generate response
            response = self._generate_response(
                speaker,
                dialogue_state,
                cultural_context
            )
            
            # Verify cultural authenticity
            score, _ = self.harmony_function(
                response['text'],
                cultural_context,
                speaker.dialect
            )
            
            if score < self.config.cultural_threshold:
                response = self._enhance_cultural_elements(
                    response,
                    cultural_context,
                    speaker
                )
            
            dialogue.append(response)
            
            # Update dialogue state
            dialogue_state = self._update_state(
                dialogue_state,
                response,
                speaker
            )
        
        return dialogue
    
    def _initialize_state(self,
                         characters: List[CharacterVoice],
                         context: DialogueContext) -> Dict:
        """Initialize dialogue generation state."""
        # Encode characters
        character_encodings = {}
        for char in characters:
            char_desc = (
                f"{char.name} ({char.role}, {char.age}): "
                f"{' '.join(char.speech_patterns)}"
            )
            char_input = self.tokenizer(char_desc, return_tensors="pt")
            char_encoding = self.model(**char_input).last_hidden_state.mean(dim=1)
            character_encodings[char.name] = self.character_encoder(char_encoding)
        
        # Encode context
        context_text = (
            f"{context.scene_description} "
            f"Emotional states: {str(context.emotional_state)} "
            f"Cultural markers: {' '.join(context.cultural_markers)}"
        )
        context_input = self.tokenizer(context_text, return_tensors="pt")
        context_encoding = self.model(**context_input).last_hidden_state.mean(dim=1)
        context_encoding = self.context_encoder(context_encoding)
        
        return {
            'character_encodings': character_encodings,
            'context_encoding': context_encoding,
            'dialogue_history': [],
            'speaker_history': [],
            'emotional_states': context.emotional_state.copy(),
            'relationship_dynamics': context.relationship_dynamics.copy()
        }
    
    def _select_next_speaker(self,
                           state: Dict,
                           dialogue: List[Dict[str, str]]) -> CharacterVoice:
        """Select next speaker based on dialogue state."""
        if not dialogue:
            # First turn - select based on context
            speaker_scores = {
                name: self._compute_speaker_score(encoding, state)
                for name, encoding in state['character_encodings'].items()
            }
            return max(speaker_scores.items(), key=lambda x: x[1])[0]
        else:
            # Select based on dialogue flow
            # Avoid same speaker twice in a row
            last_speaker = dialogue[-1]['speaker']
            available_speakers = [
                name for name in state['character_encodings'].keys()
                if name != last_speaker
            ]
            speaker_scores = {
                name: self._compute_speaker_score(
                    state['character_encodings'][name],
                    state,
                    last_speaker=last_speaker
                )
                for name in available_speakers
            }
            return max(speaker_scores.items(), key=lambda x: x[1])[0]
    
    def _compute_speaker_score(self,
                             char_encoding: torch.Tensor,
                             state: Dict,
                             last_speaker: Optional[str] = None) -> float:
        """Compute score for potential next speaker."""
        # Base score from character encoding
        score = torch.sigmoid(
            torch.matmul(
                char_encoding,
                state['context_encoding'].transpose(-1, -2)
            )
        ).mean().item()
        
        # Adjust based on dialogue history
        if last_speaker and (last_speaker, char_encoding) in state['relationship_dynamics']:
            relationship = state['relationship_dynamics'][(last_speaker, char_encoding)]
            if relationship == "conflict":
                score *= 1.2  # Increase probability for dramatic tension
            elif relationship == "alliance":
                score *= 1.1  # Slight increase for allied characters
        
        return score
    
    def _generate_response(self,
                          speaker: CharacterVoice,
                          state: Dict,
                          cultural_context: Dict[str, List[str]]) -> Dict[str, str]:
        """Generate character response."""
        # Prepare input context
        dialogue_history = state['dialogue_history'][-3:]  # Last 3 turns for context
        history_text = " ".join(
            f"{turn['speaker']}: {turn['text']}"
            for turn in dialogue_history
        )
        
        input_text = (
            f"Speaker: {speaker.name}\n"
            f"Dialect: {speaker.dialect}\n"
            f"History: {history_text}\n"
            f"Cultural context: {str(cultural_context)}"
        )
        
        # Generate response
        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            early_stopping=True
        )
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Adapt to dialect
        response_text = self._adapt_to_dialect(response_text, speaker.dialect)
        
        return {
            'speaker': speaker.name,
            'text': response_text,
            'dialect': speaker.dialect,
            'emotion': state['emotional_states'].get(speaker.name, 'neutral')
        }
    
    def _adapt_to_dialect(self, text: str, dialect: str) -> str:
        """Adapt text to specific dialect."""
        # Process through cultural integrator
        adapted_text, _ = self.cultural_integrator(
            text,
            {'dialect': dialect},
            {}  # No patterns to avoid
        )
        return adapted_text
    
    def _enhance_cultural_elements(self,
                                 response: Dict[str, str],
                                 cultural_context: Dict[str, List[str]],
                                 speaker: CharacterVoice) -> Dict[str, str]:
        """Enhance cultural elements in response."""
        enhanced_text, _ = self.cultural_integrator(
            response['text'],
            cultural_context,
            {}  # No patterns to avoid
        )
        
        response['text'] = enhanced_text
        return response
    
    def _update_state(self,
                     state: Dict,
                     response: Dict[str, str],
                     speaker: CharacterVoice) -> Dict:
        """Update dialogue state after response."""
        # Update history
        state['dialogue_history'].append(response)
        state['speaker_history'].append(speaker.name)
        
        # Update emotional states based on response
        # This is a placeholder - implement actual emotion analysis
        state['emotional_states'][speaker.name] = 'neutral'
        
        return state
