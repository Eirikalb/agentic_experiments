"""
Meta-LLM optimization utilities for improving prompts and parsing strategies.

This module provides interfaces for using a meta-LLM to optimize the
prompting and parsing strategies of the main LLM agent.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our LLM agent components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agent.llm_agent import LLMConfig, PromptConfig, ParserConfig, LLMClient


class MetaOptimizer:
    """
    Meta-LLM optimizer for improving prompt and parsing strategies.
    """
    
    def __init__(self, meta_llm_config: LLMConfig):
        """
        Initialize the meta-optimizer.
        
        Args:
            meta_llm_config: Configuration for the meta-LLM
        """
        self.meta_llm = LLMClient(meta_llm_config)
        self.optimization_history = []
    
    def optimize_prompts(self, optimization_data: Dict[str, Any]) -> PromptConfig:
        """
        Use meta-LLM to optimize prompt configuration.
        
        Args:
            optimization_data: Data from the main agent for optimization
            
        Returns:
            PromptConfig: Optimized prompt configuration
        """
        # Extract relevant data
        prompt_history = optimization_data.get("prompt_history", [])
        action_history = optimization_data.get("action_history", [])
        
        # Create optimization prompt
        optimization_prompt = self._create_prompt_optimization_prompt(
            prompt_history, action_history
        )
        
        # Get meta-LLM response
        messages = [
            {
                "role": "system", 
                "content": "You are an expert at optimizing LLM prompts for coding agents."
            },
            {
                "role": "user", 
                "content": optimization_prompt
            }
        ]
        
        try:
            response = self.meta_llm.generate_response(messages)
            
            # Parse the response to extract optimized config
            optimized_config = self._parse_prompt_optimization_response(response)
            
            # Record optimization
            self.optimization_history.append({
                "timestamp": time.time(),
                "type": "prompt_optimization",
                "input_data": optimization_data,
                "response": response,
                "optimized_config": optimized_config
            })
            
            return optimized_config
            
        except Exception as e:
            print(f"Meta-optimization failed: {e}")
            # Return default config on failure
            return PromptConfig()
    
    def optimize_parsing(self, optimization_data: Dict[str, Any]) -> ParserConfig:
        """
        Use meta-LLM to optimize parsing configuration.
        
        Args:
            optimization_data: Data from the main agent for optimization
            
        Returns:
            ParserConfig: Optimized parser configuration
        """
        # Extract relevant data
        parsing_history = optimization_data.get("parsing_history", [])
        action_history = optimization_data.get("action_history", [])
        
        # Create optimization prompt
        optimization_prompt = self._create_parsing_optimization_prompt(
            parsing_history, action_history
        )
        
        # Get meta-LLM response
        messages = [
            {
                "role": "system", 
                "content": "You are an expert at optimizing LLM response parsing strategies."
            },
            {
                "role": "user", 
                "content": optimization_prompt
            }
        ]
        
        try:
            response = self.meta_llm.generate_response(messages)
            
            # Parse the response to extract optimized config
            optimized_config = self._parse_parsing_optimization_response(response)
            
            # Record optimization
            self.optimization_history.append({
                "timestamp": time.time(),
                "type": "parsing_optimization",
                "input_data": optimization_data,
                "response": response,
                "optimized_config": optimized_config
            })
            
            return optimized_config
            
        except Exception as e:
            print(f"Meta-optimization failed: {e}")
            # Return default config on failure
            return ParserConfig()
    
    def _create_prompt_optimization_prompt(self, prompt_history: List[Dict], action_history: List[Dict]) -> str:
        """Create a prompt for optimizing prompt strategies."""
        return f"""
Analyze the following data from a coding agent and suggest optimizations for the prompt strategy.

PROMPT HISTORY (last 10 entries):
{json.dumps(prompt_history[-10:], indent=2)}

ACTION HISTORY (last 10 entries):
{json.dumps(action_history[-10:], indent=2)}

Based on this data, suggest optimizations for the PromptConfig. Consider:
1. Which prompts led to successful tool executions?
2. Which prompts failed or led to parsing errors?
3. What context information was most useful?
4. How can we improve the system prompt or task prompt template?

Respond with a JSON object containing the optimized PromptConfig:
{{
    "system_prompt": "optimized system prompt",
    "task_prompt_template": "optimized task template",
    "context_inclusion": true/false,
    "action_history_inclusion": true/false
}}
"""
    
    def _create_parsing_optimization_prompt(self, parsing_history: List[Dict], action_history: List[Dict]) -> str:
        """Create a prompt for optimizing parsing strategies."""
        return f"""
Analyze the following data from a coding agent and suggest optimizations for the parsing strategy.

PARSING HISTORY (last 10 entries):
{json.dumps(parsing_history[-10:], indent=2)}

ACTION HISTORY (last 10 entries):
{json.dumps(action_history[-10:], indent=2)}

Based on this data, suggest optimizations for the ParserConfig. Consider:
1. Which parsing strategies were most successful?
2. What types of LLM responses were hardest to parse?
3. How can we improve the response format or extraction methods?
4. What fallback strategies work best?

Respond with a JSON object containing the optimized ParserConfig:
{{
    "response_format": "json/structured_text/freeform",
    "tool_call_extraction": "regex/json_parsing/llm_parsing",
    "fallback_strategy": "heuristic/retry/abort"
}}
"""
    
    def _parse_prompt_optimization_response(self, response: str) -> PromptConfig:
        """Parse meta-LLM response to extract optimized PromptConfig."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                config_data = json.loads(json_str)
                
                return PromptConfig(
                    system_prompt=config_data.get("system_prompt", ""),
                    task_prompt_template=config_data.get("task_prompt_template", ""),
                    context_inclusion=config_data.get("context_inclusion", True),
                    action_history_inclusion=config_data.get("action_history_inclusion", True)
                )
        except Exception as e:
            print(f"Failed to parse prompt optimization response: {e}")
        
        # Return default config on failure
        return PromptConfig()
    
    def _parse_parsing_optimization_response(self, response: str) -> ParserConfig:
        """Parse meta-LLM response to extract optimized ParserConfig."""
        try:
            # Try to extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                config_data = json.loads(json_str)
                
                return ParserConfig(
                    response_format=config_data.get("response_format", "json"),
                    tool_call_extraction=config_data.get("tool_call_extraction", "regex"),
                    fallback_strategy=config_data.get("fallback_strategy", "heuristic")
                )
        except Exception as e:
            print(f"Failed to parse parsing optimization response: {e}")
        
        # Return default config on failure
        return ParserConfig()
    
    def get_optimization_history(self) -> List[Dict[str, Any]]:
        """Get the history of optimizations performed."""
        return self.optimization_history


class OptimizationManager:
    """
    High-level manager for coordinating optimization processes.
    """
    
    def __init__(self, meta_llm_config: LLMConfig):
        """
        Initialize the optimization manager.
        
        Args:
            meta_llm_config: Configuration for the meta-LLM
        """
        self.optimizer = MetaOptimizer(meta_llm_config)
        self.optimization_sessions = []
    
    def run_optimization_session(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a complete optimization session.
        
        Args:
            agent_data: Data from the main agent for optimization
            
        Returns:
            dict: Optimization results
        """
        session_id = len(self.optimization_sessions) + 1
        session_start = time.time()
        
        print(f"🔄 Starting optimization session {session_id}")
        
        # Optimize prompts
        print("📝 Optimizing prompts...")
        optimized_prompts = self.optimizer.optimize_prompts(agent_data)
        
        # Optimize parsing
        print("🔍 Optimizing parsing...")
        optimized_parsing = self.optimizer.optimize_parsing(agent_data)
        
        session_duration = time.time() - session_start
        
        session_result = {
            "session_id": session_id,
            "timestamp": session_start,
            "duration": session_duration,
            "optimized_prompts": optimized_prompts,
            "optimized_parsing": optimized_parsing,
            "input_data_summary": {
                "prompt_history_size": len(agent_data.get("prompt_history", [])),
                "parsing_history_size": len(agent_data.get("parsing_history", [])),
                "action_history_size": len(agent_data.get("action_history", []))
            }
        }
        
        self.optimization_sessions.append(session_result)
        
        print(f"✅ Optimization session {session_id} completed in {session_duration:.2f}s")
        
        return session_result
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of all optimization sessions."""
        return {
            "total_sessions": len(self.optimization_sessions),
            "total_duration": sum(session["duration"] for session in self.optimization_sessions),
            "sessions": self.optimization_sessions,
            "optimization_history": self.optimizer.get_optimization_history()
        } 