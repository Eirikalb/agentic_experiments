"""
Evaluation package for LLM agent testing and optimization.

This package contains tools for evaluating LLM agent performance
and collecting detailed traces for prompt optimization.
"""

from .task_suite import TaskSuite, TaskDefinition
from .evaluator import LLMAgentEvaluator, TaskResult, ExecutionStep
from .storage import EvaluationStorage, EnhancedEvaluator, RunMetadata

__all__ = [
    'TaskSuite',
    'TaskDefinition', 
    'LLMAgentEvaluator',
    'TaskResult',
    'ExecutionStep',
    'EvaluationStorage',
    'EnhancedEvaluator',
    'RunMetadata'
] 