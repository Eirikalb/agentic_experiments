"""
Coding Agent for automated code generation and file manipulation.

This agent uses a set of file manipulation tools to execute coding tasks
and optimize its context to improve success rates.
"""

import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import our file tools
import sys
sys.path.append(str(Path(__file__).parent.parent))
from tools.file_tools import (
    create_file, read_file, edit_file, list_files, 
    delete_file, apply_diff, read_diff
)


class CodingAgent:
    """
    An agent that can perform coding tasks using file manipulation tools.
    
    The agent maintains context about the current state and uses this
    to make informed decisions about which tools to use.
    """
    
    def __init__(self, workspace_dir: str = "."):
        """
        Initialize the coding agent.
        
        Args:
            workspace_dir: The directory where the agent will work
        """
        self.workspace_dir = Path(workspace_dir)
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
        self.execution_history = []  # Track detailed execution steps
        self.context = {
            "current_files": [],
            "recent_actions": [],
            "task_description": "",
            "current_state": {}
        }
        
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
            self.context["recent_actions"] = self.action_history[-10:]
            
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
            self.action_history.append(action_record)
            
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
    
    def get_context_summary(self) -> str:
        """
        Get a summary of the current context.
        
        Returns:
            str: Summary of current state
        """
        summary_parts = []
        
        # Task description
        if self.context["task_description"]:
            summary_parts.append(f"Task: {self.context['task_description']}")
        
        # Current files
        if self.context["current_files"]:
            file_list = ", ".join([f["name"] for f in self.context["current_files"]])
            summary_parts.append(f"Files in workspace: {file_list}")
        else:
            summary_parts.append("Workspace is empty")
        
        # Recent actions
        if self.context["recent_actions"]:
            recent_tools = [action["tool"] for action in self.context["recent_actions"][-3:]]
            summary_parts.append(f"Recent actions: {', '.join(recent_tools)}")
        
        return "\n".join(summary_parts)
    
    def execute_task(self, task_description: str, max_steps: int = 20) -> Dict[str, Any]:
        """
        Execute a coding task with automatic tool selection.
        
        This is a simplified version that uses basic heuristics to choose tools.
        In a more sophisticated implementation, this would use an LLM to make decisions.
        
        Args:
            task_description: Natural language description of the task
            max_steps: Maximum number of tool executions allowed
            
        Returns:
            dict: Summary of the task execution
        """
        self.context["task_description"] = task_description
        self.execution_history = []  # Reset execution history for new task
        self.update_context()
        
        print(f"Starting task: {task_description}")
        print(f"Working in: {self.workspace_dir.absolute()}")
        
        step_count = 0
        task_success = False
        start_time = time.time()
        
        try:
            # Simple heuristic-based task execution
            # This is where you'd integrate with an LLM for smarter decision making
            
            # First, let's see what's in the workspace
            print("\n1. Exploring workspace...")
            files_result = self.execute_tool("list_files", directory=str(self.workspace_dir))
            if not files_result["success"]:
                return {
                    "success": False,
                    "message": f"Failed to explore workspace: {files_result['message']}",
                    "steps_taken": step_count,
                    "execution_history": self.execution_history
                }
            
            step_count += 1
            
            # For now, we'll implement a very basic task parser
            # This is where the LLM integration would happen
            task_lower = task_description.lower()
            
            if "create" in task_lower and "file" in task_lower:
                # Extract filenames from task description
                # Look for patterns like "file called X" or "file X"
                words = task_description.split()
                filenames = []
                
                # Pattern 1: "file called filename.ext"
                for i, word in enumerate(words):
                    if word.lower() == "called" and i > 0 and i + 1 < len(words):
                        if words[i-1].lower() == "file":
                            potential_filename = words[i + 1]
                            if "." in potential_filename:  # Likely a filename with extension
                                filenames.append(potential_filename)
                
                # Pattern 2: "file filename.ext"
                for i, word in enumerate(words):
                    if word.lower() == "file" and i + 1 < len(words):
                        potential_filename = words[i + 1]
                        if "." in potential_filename:  # Likely a filename with extension
                            if potential_filename not in filenames:
                                filenames.append(potential_filename)
                
                # Pattern 3: "another called filename.ext"
                for i, word in enumerate(words):
                    if word.lower() == "called" and i > 0 and i + 1 < len(words):
                        if words[i-1].lower() == "another":
                            potential_filename = words[i + 1]
                            if "." in potential_filename:  # Likely a filename with extension
                                if potential_filename not in filenames:
                                    filenames.append(potential_filename)
                
                if filenames:
                    print(f"\n2. Creating {len(filenames)} file(s): {', '.join(filenames)}")
                    
                    for filename in filenames:
                        content = ""
                        if "hello" in task_lower and "world" in task_lower and filename.endswith('.py'):
                            content = 'print("Hello, World!")'
                        elif "function" in task_lower and filename.endswith('.py'):
                            content = 'def main():\n    pass\n\nif __name__ == "__main__":\n    main()'
                        elif filename.endswith('.txt'):
                            content = "This is a text file created by the coding agent."
                        elif filename.endswith('.json'):
                            content = '{\n    "name": "config",\n    "version": "1.0.0"\n}'
                        elif filename.endswith('.py'):
                            content = 'def main():\n    print("Python file created by the coding agent")\n\nif __name__ == "__main__":\n    main()'
                        
                        # Create the file in the workspace directory
                        full_path = str(self.workspace_dir / filename)
                        result = self.execute_tool("create_file", path=full_path, content=content)
                        step_count += 1
                        
                        if result["success"]:
                            task_success = True
                            print(f"✓ Successfully created {filename}")
                        else:
                            print(f"✗ Failed to create {filename}: {result['message']}")
                
                else:
                    print("Could not identify files to create from task description")
                    return {
                        "success": False,
                        "message": "Could not identify files to create from task description",
                        "steps_taken": step_count,
                        "execution_history": self.execution_history
                    }
            
            else:
                print("Task not recognized. This agent currently only handles file creation tasks.")
                return {
                    "success": False,
                    "message": "Task not recognized. This agent currently only handles file creation tasks.",
                    "steps_taken": step_count,
                    "execution_history": self.execution_history
                }
            
            # Calculate duration
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": task_success,
                "message": "Task completed with success" if task_success else "Task failed",
                "steps_taken": step_count,
                "max_steps": max_steps,
                "duration": duration,
                "execution_history": self.execution_history
            }
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            return {
                "success": False,
                "message": f"Task execution failed with exception: {str(e)}",
                "steps_taken": step_count,
                "max_steps": max_steps,
                "duration": duration,
                "execution_history": self.execution_history,
                "error": str(e)
            }
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description of a specific tool."""
        if tool_name == "create_file":
            return "Create a new file with optional content"
        elif tool_name == "read_file":
            return "Read and return file content"
        elif tool_name == "edit_file":
            return "Replace entire file content"
        elif tool_name == "list_files":
            return "List files and directories"
        elif tool_name == "delete_file":
            return "Delete a file"
        elif tool_name == "apply_diff":
            return "Apply a diff to a file"
        elif tool_name == "read_diff":
            return "Show file differences"
        else:
            return "Unknown tool"
    
    def reset_context(self):
        """Reset the agent's context and history."""
        self.action_history = []
        self.execution_history = []
        self.context = {
            "current_files": [],
            "recent_actions": [],
            "task_description": "",
            "current_state": {}
        } 