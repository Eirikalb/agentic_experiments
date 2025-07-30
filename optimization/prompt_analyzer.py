"""
Prompt Analysis and Optimization Framework

This module analyzes evaluation results to identify prompt optimization
opportunities and generate improved prompts.
"""

import json
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FailurePattern:
    """Pattern of failure identified in evaluation results."""
    pattern_type: str  # parsing_error, tool_execution_error, content_missing, etc.
    description: str
    frequency: int
    examples: List[Dict[str, Any]]
    suggested_fixes: List[str]


@dataclass
class OptimizationSuggestion:
    """Suggestion for prompt optimization."""
    category: str  # system_prompt, task_prompt, parsing, etc.
    current_issue: str
    suggested_improvement: str
    expected_impact: str
    priority: int  # 1-5, higher is more important


class PromptAnalyzer:
    """Analyzes evaluation results to identify prompt optimization opportunities."""
    
    def __init__(self, results_file: str):
        self.results_file = results_file
        self.results = self._load_results()
        self.failure_patterns = []
        self.optimization_suggestions = []
    
    def _load_results(self) -> Dict[str, Any]:
        """Load evaluation results from JSON file."""
        with open(self.results_file, 'r') as f:
            return json.load(f)
    
    def analyze_failures(self) -> List[FailurePattern]:
        """Analyze failure patterns in the evaluation results."""
        patterns = []
        
        # Analyze detailed results
        for result in self.results.get("detailed_results", []):
            if not result.get("success", False):
                self._analyze_single_failure(result, patterns)
        
        self.failure_patterns = patterns
        return patterns
    
    def _analyze_single_failure(self, result: Dict[str, Any], patterns: List[FailurePattern]):
        """Analyze a single failed task result."""
        task_id = result.get("task_id", "unknown")
        task_name = result.get("task_name", "unknown")
        
        # Check for missing files
        missing_files = result.get("missing_files", [])
        if missing_files:
            pattern = self._find_or_create_pattern(patterns, "missing_files", "Files not created as expected")
            pattern.frequency += 1
            pattern.examples.append({
                "task": f"{task_id}: {task_name}",
                "missing_files": missing_files,
                "steps": result.get("steps_taken", 0)
            })
        
        # Check for parsing errors
        execution_steps = result.get("execution_steps", [])
        for step in execution_steps:
            parsed_tool_call = step.get("parsed_tool_call", {})
            if parsed_tool_call.get("error"):
                pattern = self._find_or_create_pattern(patterns, "parsing_error", "LLM response parsing failed")
                pattern.frequency += 1
                pattern.examples.append({
                    "task": f"{task_id}: {task_name}",
                    "error": parsed_tool_call.get("error"),
                    "llm_response": step.get("llm_response", "")[:200]
                })
        
        # Check for tool execution errors
        for step in execution_steps:
            tool_result = step.get("tool_execution_result", {})
            if tool_result and not tool_result.get("success", False):
                pattern = self._find_or_create_pattern(patterns, "tool_execution_error", "Tool execution failed")
                pattern.frequency += 1
                pattern.examples.append({
                    "task": f"{task_id}: {task_name}",
                    "tool": step.get("parsed_tool_call", {}).get("tool"),
                    "error": tool_result.get("message", "Unknown error")
                })
        
        # Check for API errors
        for step in execution_steps:
            llm_response = step.get("llm_response", "")
            if "429" in llm_response or "401" in llm_response or "402" in llm_response:
                pattern = self._find_or_create_pattern(patterns, "api_error", "API rate limiting or authentication issues")
                pattern.frequency += 1
                pattern.examples.append({
                    "task": f"{task_id}: {task_name}",
                    "response": llm_response[:200]
                })
    
    def _find_or_create_pattern(self, patterns: List[FailurePattern], pattern_type: str, description: str) -> FailurePattern:
        """Find existing pattern or create new one."""
        for pattern in patterns:
            if pattern.pattern_type == pattern_type:
                return pattern
        
        # Create new pattern
        pattern = FailurePattern(
            pattern_type=pattern_type,
            description=description,
            frequency=0,
            examples=[],
            suggested_fixes=[]
        )
        patterns.append(pattern)
        return pattern
    
    def generate_optimization_suggestions(self) -> List[OptimizationSuggestion]:
        """Generate optimization suggestions based on failure analysis."""
        suggestions = []
        
        # Analyze failure patterns
        for pattern in self.failure_patterns:
            if pattern.pattern_type == "missing_files":
                suggestions.append(OptimizationSuggestion(
                    category="system_prompt",
                    current_issue="Agent not creating all required files",
                    suggested_improvement="Add explicit instructions about creating all specified files and checking completion",
                    expected_impact="Higher file creation success rate",
                    priority=4
                ))
            
            elif pattern.pattern_type == "parsing_error":
                suggestions.append(OptimizationSuggestion(
                    category="parsing",
                    current_issue="LLM response format not properly parsed",
                    suggested_improvement="Enhance XML parsing for Kimi-K2 responses",
                    expected_impact="Better tool call extraction",
                    priority=3
                ))
            
            elif pattern.pattern_type == "tool_execution_error":
                suggestions.append(OptimizationSuggestion(
                    category="system_prompt",
                    current_issue="Tool parameters not correctly specified",
                    suggested_improvement="Add parameter validation and retry logic",
                    expected_impact="More robust tool execution",
                    priority=3
                ))
        
        # Add general suggestions based on success rate
        success_rate = self.results.get("summary", {}).get("success_rate", 0)
        if success_rate < 50:
            suggestions.append(OptimizationSuggestion(
                category="system_prompt",
                current_issue="Low overall success rate",
                suggested_improvement="Add more explicit step-by-step instructions and completion criteria",
                expected_impact="Higher overall success rate",
                priority=5
            ))
        
        # Add suggestions for multi-step tasks
        multi_step_failures = self._count_multi_step_failures()
        if multi_step_failures > 0:
            suggestions.append(OptimizationSuggestion(
                category="task_prompt",
                current_issue="Multi-step tasks not completing",
                suggested_improvement="Add explicit continuation instructions and progress tracking",
                expected_impact="Better multi-step task completion",
                priority=4
            ))
        
        self.optimization_suggestions = suggestions
        return suggestions
    
    def _count_multi_step_failures(self) -> int:
        """Count failures in tasks that should require multiple steps."""
        multi_step_tasks = ["task_5", "task_7", "task_10"]  # Tasks that should require multiple steps
        count = 0
        
        for result in self.results.get("detailed_results", []):
            if (result.get("task_id") in multi_step_tasks and 
                not result.get("success", False) and 
                result.get("steps_taken", 0) < 2):
                count += 1
        
        return count
    
    def generate_improved_prompt(self) -> Dict[str, str]:
        """Generate improved prompts based on analysis."""
        current_system_prompt = """You are a coding agent that can manipulate files and write code.

Available tools:
- create_file: Create a new file with optional content
- read_file: Read the content of a file
- edit_file: Replace the content of an existing file
- list_files: List files and directories in a directory
- delete_file: Delete a file
- apply_diff: Apply a diff to a file
- read_diff: Show file differences

You should:
1. Analyze the user's request
2. Choose the appropriate tool(s) to use
3. Execute the tool(s) with correct parameters
4. Provide clear feedback about what you did

Always respond in the specified format and be precise with your tool selections."""

        # Improved system prompt based on analysis
        improved_system_prompt = """You are a coding agent that can manipulate files and write code.

Available tools:
- create_file: Create a new file with optional content
- read_file: Read the content of a file
- edit_file: Replace the content of an existing file
- list_files: List files and directories in a directory
- delete_file: Delete a file
- apply_diff: Apply a diff to a file
- read_diff: Show file differences

IMPORTANT INSTRUCTIONS:
1. Always create ALL files mentioned in the task description
2. For multi-step tasks, complete each step before moving to the next
3. Verify that files are created successfully before considering the task complete
4. Use the exact file names specified in the task
5. Include appropriate content based on the task requirements
6. If a task requires multiple files, create them all
7. For complex tasks, break them down into individual steps

You should:
1. Analyze the user's request carefully
2. Identify ALL required files and content
3. Choose the appropriate tool(s) to use
4. Execute the tool(s) with correct parameters
5. Verify completion of each step
6. Only indicate task completion when ALL requirements are met

Always respond in the specified format and be precise with your tool selections."""

        return {
            "current_system_prompt": current_system_prompt,
            "improved_system_prompt": improved_system_prompt,
            "key_improvements": [
                "Added explicit instruction to create ALL files mentioned",
                "Added multi-step task guidance",
                "Added verification requirements",
                "Added content requirements guidance",
                "Added completion criteria"
            ]
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive analysis report."""
        self.analyze_failures()
        self.generate_optimization_suggestions()
        
        return {
            "summary": {
                "total_tasks": self.results.get("summary", {}).get("total_tasks", 0),
                "success_rate": self.results.get("summary", {}).get("success_rate", 0),
                "failure_patterns_found": len(self.failure_patterns),
                "optimization_suggestions": len(self.optimization_suggestions)
            },
            "failure_patterns": [
                {
                    "type": pattern.pattern_type,
                    "description": pattern.description,
                    "frequency": pattern.frequency,
                    "examples": pattern.examples[:3]  # Top 3 examples
                }
                for pattern in self.failure_patterns
            ],
            "optimization_suggestions": [
                {
                    "category": suggestion.category,
                    "current_issue": suggestion.current_issue,
                    "suggested_improvement": suggestion.suggested_improvement,
                    "expected_impact": suggestion.expected_impact,
                    "priority": suggestion.priority
                }
                for suggestion in sorted(self.optimization_suggestions, key=lambda x: x.priority, reverse=True)
            ],
            "improved_prompts": self.generate_improved_prompt()
        }


if __name__ == "__main__":
    # Test the analyzer
    analyzer = PromptAnalyzer("full_evaluation_results.json")
    report = analyzer.generate_report()
    
    print("🔍 Prompt Analysis Report")
    print("=" * 50)
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Failure Patterns Found: {report['summary']['failure_patterns_found']}")
    print(f"Optimization Suggestions: {report['summary']['optimization_suggestions']}")
    
    print("\n📊 Top Failure Patterns:")
    for pattern in report['failure_patterns'][:3]:
        print(f"  {pattern['type']}: {pattern['frequency']} occurrences")
    
    print("\n💡 Top Optimization Suggestions:")
    for suggestion in report['optimization_suggestions'][:3]:
        print(f"  Priority {suggestion['priority']}: {suggestion['suggested_improvement']}") 