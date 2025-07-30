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
from tools.execution_tools import (
    run_python_file, detect_imports, install_package,
    install_requirements, run_command, check_python_version
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
class ExecutionConfig:
    """Configuration for execution environment."""
    python_executable: Optional[str] = None  # Path to Python interpreter
    enable_execution: bool = True  # Whether to enable execution tools
    execution_timeout: int = 30  # Default timeout for execution
    package_install_timeout: int = 60  # Timeout for package installation


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

IMPORTANT INSTRUCTIONS:
1. Always create ALL files mentioned in the task description
2. For multi-step tasks, complete each step before moving to the next
3. Verify that files are created successfully before considering the task complete
4. Use the exact file names specified in the task
5. Include appropriate content based on the task requirements
6. If a task requires multiple files, create them all
7. For complex tasks, break them down into individual steps

RESPONSE FORMAT:
Use this exact format for tool calls:
<use_tool>
<tool_name>create_file</tool_name>
<parameters>
{{
  "path": "filename.py",
  "content": "file content here"
}}
</parameters>
</use_tool>

You should:
1. Analyze the user's request carefully
2. Identify ALL required files and content
3. Choose the appropriate tool(s) to use
4. Execute the tool(s) with correct parameters
5. Verify completion of each step
6. Only indicate task completion when ALL requirements are met

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
            "read_diff": "Show file differences",
            "run_python_file": "Execute a Python file and return the output",
            "detect_imports": "Analyze a Python file to detect its imports",
            "install_package": "Install a Python package using pip",
            "install_requirements": "Install packages from a requirements.txt file",
            "run_command": "Execute a shell command",
            "check_python_version": "Check the Python version of the interpreter"
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
        if ('<tool_use>' in response or '<use_tool>' in response or 
            any(f'<{tool}>' in response for tool in available_tools)):
            import re
            
            # Extract tool name - try multiple patterns
            tool_name = None
            
            # Pattern 1: <tool_name>create_file</tool_name>
            tool_match = re.search(r'<tool_name>([^<]+)</tool_name>', response)
            if tool_match:
                tool_name = tool_match.group(1).strip()
            
            # Pattern 2: <create_file> or <use_tool><tool_name>create_file</tool_name>
            if not tool_name:
                for tool in available_tools:
                    if f'<{tool}>' in response or f'<{tool} ' in response:
                        tool_name = tool
                        break
            
            # Pattern 3: <use_tool><tool_name>create_file</tool_name>
            if not tool_name:
                tool_match = re.search(r'<use_tool>\s*<tool_name>([^<]+)</tool_name>', response, re.DOTALL)
                if tool_match:
                    tool_name = tool_match.group(1).strip()
            
            if tool_name and tool_name in available_tools:
                tool_call['tool'] = tool_name
            
            # Extract parameters - try multiple patterns
            params = {}
            
            # Pattern 1: <parameters>{...}</parameters>
            params_match = re.search(r'<parameters>\s*(\{.*?\})\s*</parameters>', response, re.DOTALL)
            if params_match:
                try:
                    params_str = params_match.group(1)
                    params = json.loads(params_str)
                except:
                    pass
            
            # Pattern 2: <param name="path">value</param>
            if not params:
                path_match = re.search(r'<param[^>]*name=["\']path["\'][^>]*>\s*([^<]+)\s*</param>', response, re.IGNORECASE)
                if path_match:
                    params['path'] = path_match.group(1).strip()
                
                content_match = re.search(r'<param[^>]*name=["\']content["\'][^>]*>\s*([^<]+)\s*</param>', response, re.IGNORECASE)
                if content_match:
                    params['content'] = content_match.group(1).strip()
            
            # Pattern 3: <path>value</path> and <content>value</content>
            if not params:
                path_match = re.search(r'<path>\s*([^<]+)\s*</path>', response, re.IGNORECASE)
                if path_match:
                    params['path'] = path_match.group(1).strip()
                
                content_match = re.search(r'<content>\s*(.*?)\s*</content>', response, re.IGNORECASE | re.DOTALL)
                if content_match:
                    params['content'] = content_match.group(1).strip()
            
            # Pattern 4: Direct tool call format like <create_file><path>file.py</path><content>...</content></create_file>
            if not params:
                for tool in available_tools:
                    if f'<{tool}>' in response:
                        # Extract path and content from within the tool tags
                        tool_pattern = f'<{tool}>(.*?)</{tool}>'
                        tool_match = re.search(tool_pattern, response, re.DOTALL)
                        if tool_match:
                            tool_content = tool_match.group(1)
                            
                            # Extract path
                            path_match = re.search(r'<path>\s*([^<]+)\s*</path>', tool_content, re.IGNORECASE)
                            if path_match:
                                params['path'] = path_match.group(1).strip()
                            
                            # Extract content
                            content_match = re.search(r'<content>\s*(.*?)\s*</content>', tool_content, re.IGNORECASE | re.DOTALL)
                            if content_match:
                                params['content'] = content_match.group(1).strip()
                            else:
                                # If no content tag, the rest might be content
                                content_parts = re.split(r'<path>[^<]*</path>', tool_content, flags=re.IGNORECASE)
                                if len(content_parts) > 1:
                                    params['content'] = content_parts[1].strip()
                            break
            
            if params:
                tool_call['parameters'] = params
        
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
                
                # Map path to directory for list_files
                if tool_call['tool'] == 'list_files' and 'path' in params and 'directory' not in params:
                    params['directory'] = params.pop('path')
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
            if not dir_match:
                # Also look for path parameter (common mistake)
                dir_match = re.search(r'path["\']?\s*:\s*["\']([^"\']+)["\']', text, re.IGNORECASE)
            
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
    LLM-powered coding agent that can manipulate files and execute code.
    
    This agent uses an LLM to orchestrate tool calling while keeping
    prompting and parsing logic separate for optimization.
    """
    
    def __init__(self, 
                 workspace_dir: str = ".",
                 llm_config: Optional[LLMConfig] = None,
                 prompt_config: Optional[PromptConfig] = None,
                 parser_config: Optional[ParserConfig] = None,
                 execution_config: Optional[ExecutionConfig] = None):
        """
        Initialize the LLM coding agent.
        
        Args:
            workspace_dir: Directory where the agent will work
            llm_config: Configuration for LLM interaction
            prompt_config: Configuration for prompting strategy
            parser_config: Configuration for response parsing
            execution_config: Configuration for execution environment
        """
        self.workspace_dir = Path(workspace_dir)
        self.llm_config = llm_config or LLMConfig()
        self.prompt_config = prompt_config or PromptConfig()
        self.parser_config = parser_config or ParserConfig()
        self.execution_config = execution_config or ExecutionConfig()
        
        # Initialize components
        self.prompt_engine = LLMPromptEngine(self.prompt_config)
        self.response_parser = LLMResponseParser(self.parser_config)
        self.llm_client = LLMClient(self.llm_config)
        
        # Context management
        self.context = {
            "current_files": [],
            "recent_actions": [],
            "task_description": "",
            "current_state": {}
        }
        
        # Execution history for logging
        self.execution_history = []
        
        # Initialize tools
        self.tools = {
            "create_file": create_file,
            "read_file": read_file,
            "edit_file": edit_file,
            "list_files": list_files,
            "delete_file": delete_file,
            "apply_diff": apply_diff,
            "read_diff": read_diff
        }
        
        # Add execution tools if enabled
        if self.execution_config.enable_execution:
            self.tools.update({
                "run_python_file": self._wrap_execution_tool(run_python_file),
                "detect_imports": self._wrap_execution_tool(detect_imports),
                "install_package": self._wrap_execution_tool(install_package),
                "install_requirements": self._wrap_execution_tool(install_requirements),
                "run_command": self._wrap_execution_tool(run_command),
                "check_python_version": self._wrap_execution_tool(check_python_version)
            })
    
    def _wrap_execution_tool(self, tool_func):
        """Wrap execution tools to include Python executable configuration."""
        def wrapped_tool(*args, **kwargs):
            # Add Python executable to kwargs if not already present
            if self.execution_config.python_executable and 'python_executable' not in kwargs:
                kwargs['python_executable'] = self.execution_config.python_executable
            
            # Add timeout configuration
            if 'timeout' not in kwargs:
                kwargs['timeout'] = self.execution_config.execution_timeout
            
            return tool_func(*args, **kwargs)
        return wrapped_tool
    
    def update_context(self):
        """Update the agent's context with current file system state."""
        try:
            # Get current file listing
            files_result = list_files(str(self.workspace_dir))
            if files_result["success"]:
                self.context["current_files"] = files_result["items"]
            else:
                self.context["current_files"] = []
                
            # Keep only recent actions (last 10)
            self.context["recent_actions"] = self.context.get("recent_actions", [])[-10:]
            
        except Exception as e:
            print(f"Warning: Failed to update context: {e}")
    
    def execute_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool with the given parameters.
        
        Args:
            tool_name: Name of the tool to execute
            **kwargs: Parameters to pass to the tool
            
        Returns:
            dict: Result of the tool execution
        """
        if tool_name not in self.tools:
            return {
                "success": False,
                "message": f"Unknown tool: {tool_name}",
                "error": "UnknownToolError"
            }
        
        try:
            # Handle parameter mapping for specific tools
            if tool_name == "list_files":
                # Map 'path' parameter to 'directory' for list_files
                if "path" in kwargs and "directory" not in kwargs:
                    kwargs["directory"] = kwargs.pop("path")
            
            # Handle file operations to ensure they work in the workspace directory
            if tool_name in ["create_file", "read_file", "edit_file", "delete_file", "list_files"]:
                # For file operations, ensure paths are relative to workspace
                if "path" in kwargs and not Path(kwargs["path"]).is_absolute():
                    kwargs["path"] = str(self.workspace_dir / kwargs["path"])
                if "directory" in kwargs and not Path(kwargs["directory"]).is_absolute():
                    kwargs["directory"] = str(self.workspace_dir / kwargs["directory"])
            
            # For execution tools, ensure they run in the workspace directory
            if tool_name in ["run_python_file"]:
                if "path" in kwargs and not Path(kwargs["path"]).is_absolute():
                    kwargs["path"] = str(self.workspace_dir / kwargs["path"])
            
            # Execute the tool
            result = self.tools[tool_name](**kwargs)
            
            # Record the action in context
            action_record = {
                "timestamp": time.time(),
                "tool": tool_name,
                "parameters": kwargs,
                "result": result
            }
            
            if "recent_actions" not in self.context:
                self.context["recent_actions"] = []
            self.context["recent_actions"].append(action_record)
            
            # Record detailed execution step
            execution_step = {
                "timestamp": time.time(),
                "tool": tool_name,
                "parameters": kwargs,
                "result": result,
                "success": result.get("success", False)
            }
            self.execution_history.append(execution_step)
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"Tool execution failed: {str(e)}",
                "error": str(e)
            }
            
            # Record the error
            action_record = {
                "timestamp": time.time(),
                "tool": tool_name,
                "parameters": kwargs,
                "result": error_result
            }
            
            if "recent_actions" not in self.context:
                self.context["recent_actions"] = []
            self.context["recent_actions"].append(action_record)
            
            # Record detailed execution step
            execution_step = {
                "timestamp": time.time(),
                "tool": tool_name,
                "parameters": kwargs,
                "result": error_result,
                "success": False,
                "error": str(e)
            }
            self.execution_history.append(execution_step)
            
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
        self.execution_history = []  # Reset execution history for new task
        self.update_context()
        
        print(f"🤖 LLM Agent executing: {task_description}")
        print(f"📁 Working in: {self.workspace_dir.absolute()}")
        
        step_count = 0
        task_success = False
        start_time = time.time()
        
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
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": False,
                "message": f"Task execution failed: {str(e)}",
                "steps_taken": step_count,
                "max_steps": max_steps,
                "duration": duration,
                "execution_history": self.execution_history,
                "error": str(e)
            }
        
        # Final context update
        self.update_context()
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        
        return {
            "success": task_success,
            "message": f"Task completed with {'success' if task_success else 'failure'}",
            "steps_taken": step_count,
            "max_steps": max_steps,
            "duration": duration,
            "execution_history": self.execution_history,
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
        
        recent = self.context.get("recent_actions", [])
        if recent:
            action_summaries = []
            for action in recent[-5:]:  # Show last 5 actions
                tool = action["tool"]
                success = action["result"]["success"]
                status = "✓" if success else "✗"
                action_summaries.append(f"{status} {tool}")
            summary_parts.append(f"Recent actions: {' → '.join(action_summaries)}")
        
        return "\n".join(summary_parts)
    
    def get_optimization_data(self) -> Dict[str, Any]:
        """Get data for prompt optimization."""
        return {
            "prompt_history": self.prompt_engine.prompt_history,
            "parsing_history": self.response_parser.parsing_history,
            "context": self.context
        }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def reset_context(self):
        """Reset the agent's context and history."""
        self.context = {
            "current_files": [],
            "recent_actions": [],
            "task_description": "",
            "current_state": {}
        }
        self.execution_history = []
        self.update_context()