"""
Agent package for the coding agent benchmark suite.

This package contains the main coding agent that uses tools to execute
coding tasks and optimize its context.
"""

from .coding_agent import CodingAgent
from .llm_agent import (
    LLMCodingAgent, 
    LLMConfig, 
    PromptConfig, 
    ParserConfig,
    LLMPromptEngine,
    LLMResponseParser,
    LLMClient
)

__all__ = [
    'CodingAgent',
    'LLMCodingAgent',
    'LLMConfig',
    'PromptConfig', 
    'ParserConfig',
    'LLMPromptEngine',
    'LLMResponseParser',
    'LLMClient'
] 