"""
LLM-powered coding agent with configurable prompting and parsing.

This agent uses an LLM to orchestrate tool calling while keeping
prompting and parsing logic separate for optimization.
"""

import json
import time
import os
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import requests
from dataclasses import dataclass

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available, continue without it
    pass

# Import our file tools
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.file_tools import (
    create_file, read_file, edit_file, list_files, 
    delete_file, apply_diff, read_diff
)


@dataclass
class LLMConfig:
    """Configuration for LLM interaction."""
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "moonshotai/kimi-k2"
    api_key: Optional[str] = None
    max_tokens: int = 4000
    temperature: float = 0.6
    timeout: int = 60
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv('OPENROUTER_API_KEY')


@dataclass
class PromptConfig:
    """Configuration for prompting strategy."""
    system_prompt: str = ""
    task_prompt_template: str = ""
    tool_selection_prompt: str = ""
    context_inclusion: bool = True
    action_history_inclusion: bool = True


@dataclass
class ParserConfig:
    """Configuration for response parsing."""
    response_format: str = "structured_text"  # json, structured_text, freeform
    tool_call_extraction: str = "regex"  # regex, json_parsing, llm_parsing
    fallback_strategy: str = "heuristic"  # heuristic, retry, abort


class LLMPromptEngine:
    """Handles prompt generation and management."""
    
    def __init__(self, config: PromptConfig):
        self.config = config
        self.prompt_history = []
    
    def generate_system_prompt(self, available_tools: List[str]) -> str:
        """Generate the system prompt for the LLM."""
        if self.config.system_prompt:
            return self.config.system_prompt
        
        # Default system prompt
        tools_description = "\n".join([
            f"- {tool}: {self._get_tool_description(tool)}"
            for tool in available_tools
        ])
        
        return f"""You are a coding agent that can manipulate files and write code.

Available tools:
{tools_description}

You should:
1. Analyze the user's request
2. Choose the appropriate tool(s) to use
3. Execute the tool(s) with correct parameters
4. Provide clear feedback about what you did

Always respond in the specified format and be precise with your tool selections."""

    def generate_task_prompt(self, task: str, context: Dict[str, Any]) -> str:
        """Generate the task-specific prompt."""
        if self.config.task_prompt_template:
            return self.config.task_prompt_template.format(
                task=task,
                context=json.dumps(context, indent=2)
            )
        
        # Default task prompt
        prompt_parts = [f"Task: {task}"]
        
        if self.config.context_inclusion and context:
            prompt_parts.append(f"\nCurrent Context:\n{json.dumps(context, indent=2)}")
        
        if self.config.action_history_inclusion and context.get("recent_actions"):
            actions = context["recent_actions"][-5:]  # Last 5 actions
            action_summary = "\n".join([
                f"- {action['tool']}: {action['result']['success']}"
                for action in actions
            ])
            prompt_parts.append(f"\nRecent Actions:\n{action_summary}")
        
        return "\n".join(prompt_parts)
    
    def _get_tool_description(self, tool_name: str) -> str:
        """Get description for a specific tool."""
        tool_descriptions = {
            "create_file": "Create a new file with optional content",
            "read_file": "Read the content of a file",
            "edit_file": "Replace the content of an existing file",
            "list_files": "List files and directories in a directory",
            "delete_file": "Delete a file",
            "apply_diff": "Apply a diff to a file",
            "read_diff": "Show file differences"
        }
        return tool_descriptions.get(tool_name, "Unknown tool")
    
    def record_prompt(self, prompt: str, response: str, success: bool):
        """Record prompt-response pairs for optimization."""
        self.prompt_history.append({
            "timestamp": time.time(),
            "prompt": prompt,
            "response": response,
            "success": success
        })


class LLMResponseParser:
    """Handles parsing of LLM responses into tool calls."""
    
    def __init__(self, config: ParserConfig):
        self.config = config
        self.parsing_history = []
    
    def parse_response(self, response: str, available_tools: List[str]) -> Dict[str, Any]:
        """Parse LLM response into a tool call."""
        try:
            if self.config.response_format == "json":
                return self._parse_json_response(response, available_tools)
            elif self.config.response_format == "structured_text":
                return self._parse_structured_text(response, available_tools)
            else:
                return self._parse_freeform(response, available_tools)
        except Exception as e:
            return self._fallback_parsing(response, available_tools, str(e))
    
    def _parse_json_response(self, response: str, available_tools: List[str]) -> Dict[str, Any]:
        """Parse JSON-formatted response."""
        try:
            # Try to extract JSON from the response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                json_str = response[json_start:json_end]
                parsed = json.loads(json_str)
                
                # Validate the parsed response
                if self._validate_tool_call(parsed, available_tools):
                    return parsed
            
            # If no valid JSON found, try to extract tool call info
            return self._extract_tool_call_from_text(response, available_tools)
            
        except json.JSONDecodeError:
            return self._extract_tool_call_from_text(response, available_tools)
    
    def _parse_structured_text(self, response: str, available_tools: List[str]) -> Dict[str, Any]:
        """Parse structured text response."""
        # Look for patterns like "TOOL: create_file" or "ACTION: read_file"
        # Also handle Kimi-K2's XML-style format: <tool_use><tool_name>create_file</tool_name>
        lines = response.split('\n')
        tool_call = {}
        
        # First try XML-style parsing (Kimi-K2 format)
        if '<tool_use>' in response or '<use_tool>' in response:
            import re
            # Extract tool name
            tool_match = re.search(r'<tool_name>([^<]+)</tool_name>', response)
            if tool_match:
                tool_name = tool_match.group(1).strip()
                if tool_name in available_tools:
                    tool_call['tool'] = tool_name
            
            # Extract parameters - try both <parameters> and <param> tags
            params_match = re.search(r'<parameters>\s*(\{.*?\})\s*</parameters>', response, re.DOTALL)
            if not params_match:
                params_match = re.search(r'<param[^>]*>\s*(\{.*?\})\s*</param>', response, re.DOTALL)
            if not params_match:
                # Try to extract individual parameters from XML
                params = {}
                path_match = re.search(r'<param[^>]*name=["\']path["\'][^>]*>\s*([^<]+)\s*</param>', response, re.IGNORECASE)
                if path_match:
                    params['path'] = path_match.group(1).strip()
                content_match = re.search(r'<param[^>]*name=["\']content["\'][^>]*>\s*([^<]+)\s*</param>', response, re.IGNORECASE)
                if content_match:
                    params['content'] = content_match.group(1).strip()
                
                if params:
                    tool_call['parameters'] = params
            else:
                try:
                    params_str = params_match.group(1)
                    params = json.loads(params_str)
                    tool_call['parameters'] = params
                except:
                    pass
        
        # Fallback to line-based parsing
        if not tool_call.get('tool'):
            for line in lines:
                line = line.strip()
                if line.startswith('TOOL:') or line.startswith('ACTION:'):
                    tool_name = line.split(':', 1)[1].strip()
                    if tool_name in available_tools:
                        tool_call['tool'] = tool_name
                elif line.startswith('PARAMS:') or line.startswith('ARGUMENTS:'):
                    try:
                        params_str = line.split(':', 1)[1].strip()
                        params = json.loads(params_str)
                        tool_call['parameters'] = params
                    except:
                        pass
        
        if tool_call.get('tool'):
            # Map Kimi-K2 parameters to our tool parameters
            if tool_call.get('parameters'):
                params = tool_call['parameters']
                # Map file_path to path for compatibility
                if 'file_path' in params and 'path' not in params:
                    params['path'] = params.pop('file_path')
            return tool_call
        else:
            return self._extract_tool_call_from_text(response, available_tools)
    
    def _parse_freeform(self, response: str, available_tools: List[str]) -> Dict[str, Any]:
        """Parse freeform text response."""
        return self._extract_tool_call_from_text(response, available_tools)
    
    def _extract_tool_call_from_text(self, text: str, available_tools: List[str]) -> Dict[str, Any]:
        """Extract tool call information from freeform text."""
        text_lower = text.lower()
        
        # Look for tool mentions
        for tool in available_tools:
            if tool in text_lower:
                # Try to extract parameters
                params = self._extract_parameters_from_text(text, tool)
                return {
                    "tool": tool,
                    "parameters": params,
                    "reasoning": text
                }
        
        # No tool found
        return {
            "tool": None,
            "parameters": {},
            "reasoning": text,
            "error": "No valid tool found in response"
        }
    
    def _extract_parameters_from_text(self, text: str, tool: str) -> Dict[str, Any]:
        """Extract parameters for a specific tool from text."""
        params = {}
        
        if tool == "create_file":
            # Look for file path and content
            import re
            # Handle both "path" and "file_path" parameters
            path_match = re.search(r'file_path["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            if not path_match:
                path_match = re.search(r'path["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            if not path_match:
                path_match = re.search(r'file[:\s]+([^\s]+\.\w+)', text, re.IGNORECASE)
            
            if path_match:
                params['path'] = path_match.group(1)
            
            # Look for content
            content_match = re.search(r'content["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            if content_match:
                params['content'] = content_match.group(1)
        
        elif tool == "read_file":
            # Look for file path
            import re
            path_match = re.search(r'file_path["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            if not path_match:
                path_match = re.search(r'path["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            if not path_match:
                path_match = re.search(r'file[:\s]+([^\s]+\.\w+)', text, re.IGNORECASE)
            
            if path_match:
                params['path'] = path_match.group(1)
        
        elif tool == "list_files":
            # Look for directory
            import re
            dir_match = re.search(r'directory["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            if not dir_match:
                dir_match = re.search(r'directory[:\s]+([^\s]+)', text, re.IGNORECASE)
            
            if dir_match:
                params['directory'] = dir_match.group(1)
        
        return params
    
    def _validate_tool_call(self, parsed: Dict[str, Any], available_tools: List[str]) -> bool:
        """Validate a parsed tool call."""
        if not isinstance(parsed, dict):
            return False
        
        tool = parsed.get('tool')
        if not tool or tool not in available_tools:
            return False
        
        return True
    
    def _fallback_parsing(self, response: str, available_tools: List[str], error: str) -> Dict[str, Any]:
        """Fallback parsing when primary method fails."""
        if self.config.fallback_strategy == "heuristic":
            return self._heuristic_parsing(response, available_tools)
        elif self.config.fallback_strategy == "retry":
            # Could implement retry logic here
            return self._heuristic_parsing(response, available_tools)
        else:
            return {
                "tool": None,
                "parameters": {},
                "reasoning": response,
                "error": f"Parsing failed: {error}"
            }
    
    def _heuristic_parsing(self, response: str, available_tools: List[str]) -> Dict[str, Any]:
        """Heuristic-based parsing as fallback."""
        # Simple keyword-based parsing
        response_lower = response.lower()
        
        if "create" in response_lower and "file" in response_lower:
            return {
                "tool": "create_file",
                "parameters": {"path": "default.txt", "content": ""},
                "reasoning": "Heuristic: create file detected"
            }
        elif "read" in response_lower and "file" in response_lower:
            return {
                "tool": "read_file",
                "parameters": {"path": "default.txt"},
                "reasoning": "Heuristic: read file detected"
            }
        elif "list" in response_lower and "file" in response_lower:
            return {
                "tool": "list_files",
                "parameters": {"directory": "."},
                "reasoning": "Heuristic: list files detected"
            }
        
        return {
            "tool": None,
            "parameters": {},
            "reasoning": response,
            "error": "Heuristic parsing failed"
        }
    
    def record_parsing(self, response: str, parsed: Dict[str, Any], success: bool):
        """Record parsing attempts for optimization."""
        self.parsing_history.append({
            "timestamp": time.time(),
            "response": response,
            "parsed": parsed,
            "success": success
        })


class LLMClient:
    """Handles communication with the LLM API."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from LLM."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/agentic_experiments",
            "X-Title": "Coding Agent Benchmark Suite"
        }
        
        payload = {
            "model": self.config.model,
            "messages": messages,
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature
        }
        
        try:
            response = requests.post(
                self.config.api_url,
                headers=headers,
                json=payload,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            
            result = response.json()
            return result['choices'][0]['message']['content']
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"LLM API request failed: {e}")
        except KeyError as e:
            raise Exception(f"Unexpected LLM API response format: {e}")


class LLMCodingAgent:
    """
    LLM-powered coding agent with configurable prompting and parsing.
    """
    
    def __init__(self, 
                 workspace_dir: str = ".",
                 llm_config: Optional[LLMConfig] = None,
                 prompt_config: Optional[PromptConfig] = None,
                 parser_config: Optional[ParserConfig] = None):
        """
        Initialize the LLM coding agent.
        
        Args:
            workspace_dir: Directory to work in
            llm_config: LLM configuration
            prompt_config: Prompt configuration
            parser_config: Parser configuration
        """
        self.workspace_dir = Path(workspace_dir)
        
        # Initialize configurations
        self.llm_config = llm_config or LLMConfig()
        self.prompt_config = prompt_config or PromptConfig()
        self.parser_config = parser_config or ParserConfig()
        
        # Initialize components
        self.prompt_engine = LLMPromptEngine(self.prompt_config)
        self.response_parser = LLMResponseParser(self.parser_config)
        self.llm_client = LLMClient(self.llm_config)
        
        # Tools and state
        self.tools = {
            "create_file": create_file,
            "read_file": read_file,
            "edit_file": edit_file,
            "list_files": list_files,
            "delete_file": delete_file,
            "apply_diff": apply_diff,
            "read_diff": read_diff
        }
        
        self.action_history = []
        self.context = {
            "current_files": [],
            "recent_actions": [],
            "task_description": "",
            "current_state": {}
        }
    
    def update_context(self):
        """Update the agent's context with current file system state."""
        try:
            files_result = list_files(str(self.workspace_dir))
            if files_result["success"]:
                self.context["current_files"] = files_result["items"]
            else:
                self.context["current_files"] = []
                
            self.context["recent_actions"] = self.action_history[-10:]
            
        except Exception as e:
            print(f"Warning: Failed to update context: {e}")
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool with the given parameters."""
        if tool_name not in self.tools:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}",
                "error": "UnknownToolError"
            }
        
        try:
            # Change to workspace directory for file operations
            original_cwd = os.getcwd()
            os.chdir(str(self.workspace_dir))
            
            try:
                # Execute the tool
                result = self.tools[tool_name](**kwargs)
                
                # Record the action
                action_record = {
                    "timestamp": time.time(),
                    "tool": tool_name,
                    "parameters": kwargs,
                    "result": result
                }
                self.action_history.append(action_record)
                
                # Update context after tool execution
                self.update_context()
                
                return result
                
            finally:
                # Always restore original working directory
                os.chdir(original_cwd)
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"Tool execution failed: {str(e)}",
                "error": str(e)
            }
            
            action_record = {
                "timestamp": time.time(),
                "tool": tool_name,
                "parameters": kwargs,
                "result": error_result
            }
            self.action_history.append(action_record)
            
            return error_result
    
    def execute_task(self, task_description: str, max_steps: int = 10) -> Dict[str, Any]:
        """
        Execute a coding task using LLM orchestration.
        
        Args:
            task_description: Natural language description of the task
            max_steps: Maximum number of tool executions allowed
            
        Returns:
            dict: Summary of the task execution
        """
        self.context["task_description"] = task_description
        self.update_context()
        
        print(f"🤖 LLM Agent executing: {task_description}")
        print(f"📁 Working in: {self.workspace_dir.absolute()}")
        
        step_count = 0
        task_success = False
        
        try:
            while step_count < max_steps:
                # Generate prompt
                system_prompt = self.prompt_engine.generate_system_prompt(list(self.tools.keys()))
                task_prompt = self.prompt_engine.generate_task_prompt(task_description, self.context)
                
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": task_prompt}
                ]
                
                # Get LLM response
                print(f"\n🔄 Step {step_count + 1}: Consulting LLM...")
                llm_response = self.llm_client.generate_response(messages)
                print(f"📝 LLM Response: {llm_response[:200]}...")
                
                # Parse response
                parsed = self.response_parser.parse_response(llm_response, list(self.tools.keys()))
                
                # Record for optimization
                self.prompt_engine.record_prompt(task_prompt, llm_response, True)
                self.response_parser.record_parsing(llm_response, parsed, True)
                
                # Check if task is complete
                if parsed.get("tool") is None:
                    print("✅ LLM indicates task is complete")
                    task_success = True
                    break
                
                # Execute tool
                tool_name = parsed["tool"]
                parameters = parsed.get("parameters", {})
                
                print(f"🔧 Executing: {tool_name} with {parameters}")
                result = self.execute_tool(tool_name, **parameters)
                
                if result["success"]:
                    print(f"✓ {tool_name} succeeded: {result['message']}")
                else:
                    print(f"✗ {tool_name} failed: {result['message']}")
                    # Could implement retry logic here
                
                step_count += 1
                
                # Check if we should continue
                if not result["success"]:
                    print("❌ Tool execution failed, stopping")
                    break
        
        except Exception as e:
            print(f"❌ Error during task execution: {e}")
            return {
                "success": False,
                "message": f"Task execution failed: {str(e)}",
                "steps_taken": step_count
            }
        
        # Final context update
        self.update_context()
        
        return {
            "success": task_success,
            "message": f"Task completed with {'success' if task_success else 'failure'}",
            "steps_taken": step_count,
            "final_context": self.get_context_summary()
        }
    
    def get_context_summary(self) -> str:
        """Get a human-readable summary of the current context."""
        summary_parts = []
        
        if self.context["task_description"]:
            summary_parts.append(f"Task: {self.context['task_description']}")
        
        files = self.context["current_files"]
        if files:
            file_list = [f"{item['name']} ({'file' if item['is_file'] else 'dir'})" 
                        for item in files[:10]]
            summary_parts.append(f"Files in workspace: {', '.join(file_list)}")
            if len(files) > 10:
                summary_parts.append(f"... and {len(files) - 10} more items")
        else:
            summary_parts.append("No files in workspace")
        
        recent = self.context["recent_actions"]
        if recent:
            action_summaries = []
            for action in recent[-5:]:
                tool = action["tool"]
                success = action["result"]["success"]
                status = "✓" if success else "✗"
                action_summaries.append(f"{status} {tool}")
            summary_parts.append(f"Recent actions: {' → '.join(action_summaries)}")
        
        return "\n".join(summary_parts)
    
    def get_optimization_data(self) -> Dict[str, Any]:
        """Get data for prompt and parser optimization."""
        return {
            "prompt_history": self.prompt_engine.prompt_history,
            "parsing_history": self.response_parser.parsing_history,
            "action_history": self.action_history,
            "context": self.context
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def reset_context(self):
        """Reset the agent's context and action history."""
        self.action_history = []
        self.context = {
            "current_files": [],
            "recent_actions": [],
            "task_description": "",
            "current_state": {}
        }
        self.update_context()