"""
Filter Framework for Medical Benchmark Datasets

This package provides a comprehensive filtering system for medical benchmark datasets,
supporting both keyword-based and AI model-based content filtering.

Main Components:
- Filter: Main filtering class
- GPT5V: GPT client for AI model-based filtering
- keywords: Predefined medical keyword sets for different categories
- prompt: Jinja2 templates for AI model prompts

Available Keyword Categories:
- dental: Dental and oral health related keywords
- cardiac: Cardiovascular related keywords
- cancer: Cancer and oncology related keywords
- neurological: Neurological related keywords
- respiratory: Respiratory related keywords
- general: General medical terms
"""

from .basic import Filter
from .gpt_client import GPT5V
from .keywords import get_keywords_for_category, get_all_keywords, KEYWORD_SETS

__all__ = [
    'Filter',
    'GPT5V',
    'get_keywords_for_category',
    'get_all_keywords',
    'KEYWORD_SETS'
]

__version__ = "1.0.0"