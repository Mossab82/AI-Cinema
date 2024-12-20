"""
Configuration management for AI-Cinema.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

# Default paths
DEFAULT_CONFIG = {
    "resources": {
        "arabic_wordnet": "data/corpora/arabic_wordnet",
        "arabic_verbnet": "data/corpora/arabic_verbnet",
        "madar_corpus": "data/corpora/madar_corpus",
        "emotion_lexicon": "data/lexicons/emotion_lexicon",
        "name_lexicon": "data/lexicons/name_lexicon",
        "contemporary_scripts": "data/scripts/contemporary",
        "traditional_scripts": "data/scripts/traditional"
    },
    "models": {
        "base_model": "aubmindlab/bert-base-arabert",
        "tokenizer": "aubmindlab/bert-base-arabert",
    },
    "generation": {
        "max_length": 1000,
        "num_beams": 5,
        "early_stopping": True
    },
    "dialects": {
        "supported": [
            "msa",
            "gulf",
            "egyptian",
            "levantine",
            "maghrebi"
        ],
        "default": "msa"
    },
    "cultural": {
        "min_elements": 3,
        "max_elements": 10,
        "required_categories": [
            "proverbs",
            "customs",
            "values"
        ]
    }
}

def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from file or use defaults.
    
    Args:
        config_path: Optional path to YAML configuration file
        
    Returns:
        Configuration dictionary
    """
    config = DEFAULT_CONFIG.copy()
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            user_config = yaml.safe_load(f)
            _deep_update(config, user_config)
    
    return config

def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_keys = [
        "resources",
        "models",
        "generation",
        "dialects",
        "cultural"
    ]
    
    # Check required top-level keys
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate resource paths
    for resource, path in config["resources"].items():
        if not os.path.exists(path) and not os.path.exists(os.path.join(os.getcwd(), path)):
            raise ValueError(f"Resource path not found: {path}")
    
    # Validate dialect configuration
    if "default" not in config["dialects"]:
        raise ValueError("No default dialect specified")
    if config["dialects"]["default"] not in config["dialects"]["supported"]:
        raise ValueError("Default dialect not in supported dialects")
    
    return True

def _deep_update(base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
    """
    Recursively update a dictionary.
    
    Args:
        base_dict: Base dictionary to update
        update_dict: Dictionary with updates
    """
    for key, value in update_dict.items():
        if isinstance(value, dict) and key in base_dict:
            _deep_update(base_dict[key], value)
        else:
            base_dict[key] = value

def get_resource_path(resource_name: str, config: Dict[str, Any]) -> Path:
    """
    Get absolute path for a resource.
    
    Args:
        resource_name: Name of the resource
        config: Configuration dictionary
        
    Returns:
        Path object for the resource
        
    Raises:
        ValueError: If resource not found in configuration
    """
    if resource_name not in config["resources"]:
        raise ValueError(f"Resource not found in config: {resource_name}")
    
    path = config["resources"][resource_name]
    if os.path.isabs(path):
        return Path(path)
    else:
        return Path(os.getcwd()) / path
