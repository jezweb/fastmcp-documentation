"""
Prompts Module
==============
Prompt templates for various use cases.
"""

from .templates import (
    analysis_prompt,
    summary_prompt,
    debug_prompt,
    optimization_prompt,
    code_review_prompt,
    documentation_prompt
)

__all__ = [
    'analysis_prompt',
    'summary_prompt',
    'debug_prompt',
    'optimization_prompt',
    'code_review_prompt',
    'documentation_prompt'
]