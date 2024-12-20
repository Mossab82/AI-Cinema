"""
Data loading utilities for AI-Cinema.
Handles loading of cultural resources, corpora, and lexicons.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import yaml
from camel_tools.utils.normalize import normalize_unicode
from camel_tools.tokenizers.word import simple_word_tokenize

logger = logging.getLogger(__name__)

def load_cultural_knowledge(path: Path) -> Dict[str, Any]:
    """
    Load cultural knowledge base.
    
    Args:
        path: Path to cultural knowledge directory
        
    Returns:
        Dictionary containing cultural knowledge elements
    """
    knowledge = {
        'proverbs': [],
        'customs': [],
        'values': [],
        'archetypes': [],
        'settings': []
    }
    
    try:
        # Load each category from its file
        for category in knowledge.keys():
            category_path = path / f"{category}.json"
            if category_path.exists():
                with open(category_path, 'r', encoding='utf-8') as f:
                    knowledge[category] = json.load(f)
            else:
                logger.warning(f"Missing cultural knowledge file: {category_path}")
        
        return knowledge
    
    except Exception as e:
        logger.error(f"Error loading cultural knowledge: {str(e)}")
        raise

def load_dialectal_corpus(path: Path, dialect: str) -> Dict[str, List[str]]:
    """
    Load dialectal corpus for specific dialect.
    
    Args:
        path: Path to MADAR corpus directory
        dialect: Target dialect
        
    Returns:
        Dictionary containing dialectal phrases and patterns
    """
    try:
        corpus_path = path / f"{dialect}_corpus.txt"
        phrases_path = path / f"{dialect}_phrases.json"
        
        corpus = []
        if corpus_path.exists():
            with open(corpus_path, 'r', encoding='utf-8') as f:
                corpus = f.readlines()
        
        phrases = {}
        if phrases_path.exists():
            with open(phrases_path, 'r', encoding='utf-8') as f:
                phrases = json.load(f)
        
        return {
            'corpus': [line.strip() for line in corpus],
            'phrases': phrases
        }
    
    except Exception as e:
        logger.error(f"Error loading dialectal corpus: {str(e)}")
        raise

def load_wordnet(path: Path) -> Dict[str, Any]:
    """
    Load Arabic WordNet data.
    
    Args:
        path: Path to WordNet directory
        
    Returns:
        Dictionary containing WordNet data
    """
    try:
        with open(path / "wordnet.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading WordNet: {str(e)}")
        raise

def load_verbnet(path: Path) -> Dict[str, Any]:
    """
    Load Arabic VerbNet data.
    
    Args:
        path: Path to VerbNet directory
        
    Returns:
        Dictionary containing VerbNet data
    """
    try:
        with open(path / "verbnet.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading VerbNet: {str(e)}")
        raise

def load_emotion_lexicon(path: Path) -> Dict[str, str]:
    """
    Load emotion lexicon.
    
    Args:
        path: Path to emotion lexicon file
        
    Returns:
        Dictionary mapping words to emotions
    """
    try:
        with open(path / "emotions.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading emotion lexicon: {str(e)}")
        raise

def load_name_lexicon(path: Path) -> Dict[str, Dict[str, Any]]:
    """
    Load name lexicon with cultural annotations.
    
    Args:
        path: Path to name lexicon file
        
    Returns:
        Dictionary containing name data and cultural annotations
    """
    try:
        with open(path / "names.json", 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading name lexicon: {str(e)}")
        raise

def load_scripts(path: Path, script_type: str = "contemporary") -> List[Dict[str, Any]]:
    """
    Load script examples.
    
    Args:
        path: Path to scripts directory
        script_type: Type of scripts to load ("contemporary" or "traditional")
        
    Returns:
        List of script dictionaries
    """
    scripts = []
    script_dir = path / script_type
    
    try:
        for script_file in script_dir.glob("*.json"):
            with open(script_file, 'r', encoding='utf-8') as f:
                script_data = json.load(f)
                scripts.append(script_data)
        
        return scripts
    
    except Exception as e:
        logger.error(f"Error loading scripts: {str(e)}")
        raise

def load_parallel_corpus(path: Path) -> pd.DataFrame:
    """
    Load parallel corpus with multiple dialects.
    
    Args:
        path: Path to parallel corpus file
        
    Returns:
        DataFrame containing parallel text in different dialects
    """
    try:
        return pd.read_csv(path / "parallel_corpus.csv", encoding='utf-8')
    except Exception as e:
        logger.error(f"Error loading parallel corpus: {str(e)}")
        raise
def load_cultural_patterns(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load cultural pattern definitions.
    
    Args:
        path: Path to patterns directory
        
    Returns:
        Dictionary of cultural patterns by category
    """
    try:
        with open(path / "patterns.json", 'r', encoding='utf-8') as f:
            patterns = json.load(f)
            
        # Validate pattern structure
        for category, pattern_list in patterns.items():
            for pattern in pattern_list:
                if not all(k in pattern for k in ['name', 'description', 'examples']):
                    logger.warning(f"Invalid pattern structure in {category}")
        
        return patterns
    
    except Exception as e:
        logger.error(f"Error loading cultural patterns: {str(e)}")
        raise

def load_dialectal_mapping(path: Path) -> Dict[str, Dict[str, str]]:
    """
    Load dialect-to-dialect mapping tables.
    
    Args:
        path: Path to mapping files
        
    Returns:
        Dictionary of dialect mapping tables
    """
    try:
        mappings = {}
        mapping_files = path.glob("*_to_*.json")
        
        for file_path in mapping_files:
            source, target = file_path.stem.split('_to_')
            with open(file_path, 'r', encoding='utf-8') as f:
                mappings[f"{source}_to_{target}"] = json.load(f)
        
        return mappings
    
    except Exception as e:
        logger.error(f"Error loading dialectal mappings: {str(e)}")
        raise

def cache_resource(func):
    """
    Decorator to cache loaded resources.
    """
    _cache = {}
    
    def wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key not in _cache:
            _cache[key] = func(*args, **kwargs)
        return _cache[key]
    
    return wrapper

@cache_resource
def load_resource_bundle(config: Dict[str, Any], resource_types: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load multiple resources at once.
    
    Args:
        config: Configuration dictionary
        resource_types: Optional list of specific resources to load
        
    Returns:
        Dictionary containing all loaded resources
    """
    resources = {}
    resource_loaders = {
        'wordnet': load_wordnet,
        'verbnet': load_verbnet,
        'emotion_lexicon': load_emotion_lexicon,
        'name_lexicon': load_name_lexicon,
        'cultural_knowledge': load_cultural_knowledge,
        'cultural_patterns': load_cultural_patterns,
        'dialectal_mappings': load_dialectal_mapping
    }
    
    try:
        # Load specified resources or all if none specified
        types_to_load = resource_types or resource_loaders.keys()
        
        for resource_type in types_to_load:
            if resource_type in resource_loaders:
                path = Path(config['resources'][resource_type])
                resources[resource_type] = resource_loaders[resource_type](path)
            else:
                logger.warning(f"Unknown resource type: {resource_type}")
        
        return resources
    
    except Exception as e:
        logger.error(f"Error loading resource bundle: {str(e)}")
        raise

def verify_resource_integrity(resources: Dict[str, Any]) -> bool:
    """
    Verify integrity of loaded resources.
    
    Args:
        resources: Dictionary of loaded resources
        
    Returns:
        True if all resources are valid
        
    Raises:
        ValueError: If resource integrity check fails
    """
    try:
        # Check WordNet
        if 'wordnet' in resources:
            if not resources['wordnet'].get('synsets'):
                raise ValueError("WordNet resource missing synsets")
        
        # Check VerbNet
        if 'verbnet' in resources:
            if not resources['verbnet'].get('verbs'):
                raise ValueError("VerbNet resource missing verbs")
        
        # Check cultural knowledge
        if 'cultural_knowledge' in resources:
            required_categories = ['proverbs', 'customs', 'values']
            for category in required_categories:
                if category not in resources['cultural_knowledge']:
                    raise ValueError(f"Cultural knowledge missing category: {category}")
        
        # Check dialectal mappings
        if 'dialectal_mappings' in resources:
            if not any(key.endswith('_to_msa') for key in resources['dialectal_mappings']):
                raise ValueError("Dialectal mappings missing MSA mappings")
        
        return True
    
    except Exception as e:
        logger.error(f"Resource integrity check failed: {str(e)}")
        raise
