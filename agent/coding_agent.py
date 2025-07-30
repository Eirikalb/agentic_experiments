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
            
            # Update context after tool execution
            self.update_context()
            
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"Tool execution failed: {str(e)}",
                "error": str(e)
            }
            
            # Record the failed action
            action_record = {
                "timestamp": time.time(),
                "tool": tool_name,
                "parameters": kwargs,
                "result": error_result
            }
            self.action_history.append(action_record)
            
            return error_result
    
    def get_context_summary(self) -> str:
        """
        Get a human-readable summary of the current context.
        
        Returns:
            str: Summary of the current state
        """
        summary_parts = []
        
        # Task description
        if self.context["task_description"]:
            summary_parts.append(f"Task: {self.context['task_description']}")
        
        # Current files
        files = self.context["current_files"]
        if files:
            file_list = [f"{item['name']} ({'file' if item['is_file'] else 'dir'})" 
                        for item in files[:10]]  # Show first 10
            summary_parts.append(f"Files in workspace: {', '.join(file_list)}")
            if len(files) > 10:
                summary_parts.append(f"... and {len(files) - 10} more items")
        else:
            summary_parts.append("No files in workspace")
        
        # Recent actions
        recent = self.context["recent_actions"]
        if recent:
            action_summaries = []
            for action in recent[-5:]:  # Show last 5 actions
                tool = action["tool"]
                success = action["result"]["success"]
                status = "✓" if success else "✗"
                action_summaries.append(f"{status} {tool}")
            summary_parts.append(f"Recent actions: {' → '.join(action_summaries)}")
        
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
        self.update_context()
        
        print(f"Starting task: {task_description}")
        print(f"Working in: {self.workspace_dir.absolute()}")
        
        step_count = 0
        task_success = False
        
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
                    "steps_taken": step_count
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
                            task_success = False
                else:
                    print("Could not determine filename from task description")
            
            elif "read" in task_lower and "file" in task_lower:
                # Extract filename from task description
                words = task_description.split()
                filename = None
                for i, word in enumerate(words):
                    if word.lower() in ["read", "show", "display"] and i + 1 < len(words):
                        potential_filename = words[i + 1]
                        if "." in potential_filename:
                            filename = potential_filename
                            break
                
                if filename:
                    print(f"\n2. Reading file: {filename}")
                    result = self.execute_tool("read_file", path=filename)
                    step_count += 1
                    
                    if result["success"]:
                        task_success = True
                        print(f"✓ Successfully read {filename}")
                        print(f"Content:\n{result['content']}")
                    else:
                        print(f"✗ Failed to read {filename}: {result['message']}")
                else:
                    print("Could not determine filename from task description")
            
            else:
                print("Task not recognized. This is where LLM integration would help.")
                print("Available tools:", list(self.tools.keys()))
        
        except Exception as e:
            print(f"Error during task execution: {e}")
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
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return list(self.tools.keys())
    
    def get_tool_description(self, tool_name: str) -> str:
        """Get description of a specific tool."""
        if tool_name in self.tools:
            return self.tools[tool_name].__doc__ or f"No description available for {tool_name}"
        return f"Unknown tool: {tool_name}"
    
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