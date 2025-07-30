"""
Benchmarks package for the coding agent benchmark suite.

This package contains various benchmarks to test the agent's capabilities
and measure its performance.
"""

from .benchmark_1 import FileCreationBenchmark
from .benchmark_2 import LLMAgentBenchmark

__all__ = ['FileCreationBenchmark', 'LLMAgentBenchmark'] 