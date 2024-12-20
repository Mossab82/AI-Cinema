"""Test suite for AI-Cinema framework."""

import os
import sys

# Add project root to path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Test data paths
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
SAMPLE_TEXTS_PATH = os.path.join(TEST_DATA_DIR, 'sample_texts.json')
CULTURAL_PATTERNS_PATH = os.path.join(TEST_DATA_DIR, 'cultural_patterns.json')
DIALECTAL_SAMPLES_PATH = os.path.join(TEST_DATA_DIR, 'dialectal_samples.json')
