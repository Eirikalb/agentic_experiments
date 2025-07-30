"""
Verification Agent for LLM-powered task completion verification.

This agent uses an LLM to intelligently verify that tasks have been completed
successfully by checking file existence and execution.
"""

import json
import time
import os
from typing import Dict, List, Any, Optional
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
from tools.file_tools import list_files, read_file
from tools.execution_tools import run_python_file


@dataclass
class VerificationConfig:
    """Configuration for verification LLM interaction."""
    api_url: str = "https://openrouter.ai/api/v1/chat/completions"
    model: str = "moonshotai/kimi-k2"
    api_key: Optional[str] = None
    max_tokens: int = 2000
    temperature: float = 0.1  # Lower temperature for more consistent verification
    timeout: int = 30
    
    def __post_init__(self):
        """Load API key from environment if not provided."""
        if self.api_key is None:
            self.api_key = os.getenv('OPENROUTER_API_KEY')


class VerificationLLMClient:
    """Handles communication with the LLM API for verification."""
    
    def __init__(self, config: VerificationConfig):
        self.config = config
    
    def generate_response(self, messages: List[Dict[str, str]]) -> str:
        """Generate response from LLM."""
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/your-repo/agentic_experiments",
            "X-Title": "Task Verification Agent"
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


class VerificationAgent:
    """
    LLM-powered verification agent that intelligently checks task completion.
    
    This agent uses an LLM to analyze the task description and verify that
    all required files exist and can be executed successfully.
    """
    
    def __init__(self, workspace_dir: str = ".", config: Optional[VerificationConfig] = None):
        """
        Initialize the verification agent.
        
        Args:
            workspace_dir: Directory where the agent will work
            config: Configuration for LLM interaction
        """
        self.workspace_dir = Path(workspace_dir)
        self.config = config or VerificationConfig()
        self.llm_client = VerificationLLMClient(self.config)
    
    def verify_task_completion(self, task_description: str, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that a task has been completed successfully.
        
        Args:
            task_description: The original task description
            execution_results: Results from the task execution
            
        Returns:
            dict: Verification results with success status and details
        """
        print(f"🔍 Verifying task completion: {task_description}")
        
        try:
            # Step 1: Get current file system state
            files_state = self._get_files_state()
            
            # Step 2: Generate verification prompt
            verification_prompt = self._generate_verification_prompt(
                task_description, files_state, execution_results
            )
            
            # Step 3: Get LLM verification response
            messages = [
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": verification_prompt}
            ]
            
            llm_response = self.llm_client.generate_response(messages)
            print(f"📝 Verification LLM Response: {llm_response[:200]}...")
            
            # Step 4: Parse verification response
            verification_result = self._parse_verification_response(llm_response)
            
            # Step 5: Execute any required verification actions
            if verification_result.get("needs_execution_verification"):
                execution_verification = self._verify_execution(verification_result.get("files_to_execute", []))
                verification_result["execution_verification"] = execution_verification
            
            return verification_result
            
        except Exception as e:
            print(f"❌ Verification failed: {e}")
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "error": str(e)
            }
    
    def _get_files_state(self) -> Dict[str, Any]:
        """Get current state of files in the workspace."""
        try:
            files_result = list_files(str(self.workspace_dir))
            if files_result["success"]:
                return {
                    "success": True,
                    "files": files_result["items"],
                    "total_files": files_result["total_files"],
                    "total_dirs": files_result["total_dirs"]
                }
            else:
                return {
                    "success": False,
                    "message": files_result["message"],
                    "files": []
                }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to get files state: {str(e)}",
                "files": []
            }
    
    def _generate_verification_prompt(self, task_description: str, files_state: Dict[str, Any], execution_results: Dict[str, Any]) -> str:
        """Generate the verification prompt for the LLM."""
        
        # Format files list
        files_list = []
        if files_state.get("success") and files_state.get("files"):
            for item in files_state["files"]:
                file_type = "file" if item["is_file"] else "directory"
                files_list.append(f"- {item['name']} ({file_type})")
        else:
            files_list.append("- No files found")
        
        files_text = "\n".join(files_list)
        
        # Format execution history
        execution_history = []
        if execution_results.get("execution_history"):
            for step in execution_results["execution_history"][-5:]:  # Last 5 steps
                tool = step.get("tool", "unknown")
                success = step.get("success", False)
                status = "✓" if success else "✗"
                execution_history.append(f"{status} {tool}")
        
        execution_text = "\n".join(execution_history) if execution_history else "No execution history"
        
        prompt = f"""Task Description: {task_description}

Current Files in Workspace:
{files_text}

Recent Execution History:
{execution_text}

Please analyze this task and verify completion by:

1. **File Requirements Analysis**: What files should exist based on the task description?
2. **File Existence Check**: Are all required files present in the workspace?
3. **Execution Requirements**: What files should be executable and what should they produce?
4. **Execution Verification**: Which files should be executed to verify the task works?

Respond in this JSON format:
{{
  "task_analysis": {{
    "required_files": ["list", "of", "required", "files"],
    "optional_files": ["list", "of", "optional", "files"],
    "executable_files": ["list", "of", "files", "that", "should", "run"],
    "expected_outputs": ["list", "of", "expected", "outputs"]
  }},
  "verification": {{
    "missing_files": ["list", "of", "missing", "files"],
    "present_files": ["list", "of", "files", "that", "exist"],
    "needs_execution_verification": true/false,
    "files_to_execute": ["list", "of", "files", "to", "execute", "for", "verification"]
  }},
  "success": true/false,
  "message": "Detailed explanation of verification result"
}}"""
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for verification."""
        return """You are a task verification expert. Your job is to analyze coding tasks and verify that they have been completed successfully.

You should:
1. Analyze the task description to understand what files should be created
2. Check if all required files exist in the workspace
3. Identify which files should be executable
4. Determine what execution verification is needed
5. Provide clear, detailed feedback about what's missing or working

Be thorough but fair. Consider:
- File naming conventions and extensions
- Common patterns for different types of tasks (Python scripts, multi-file projects, etc.)
- Execution requirements (imports, dependencies, etc.)
- Expected outputs and behaviors

Respond with structured JSON that clearly indicates success/failure and provides actionable feedback."""
    
    def _parse_verification_response(self, response: str) -> Dict[str, Any]:
        """Parse the LLM verification response."""
        try:
            # Try to extract JSON from the response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                parsed = json.loads(json_str)
                return parsed
            else:
                # Fallback parsing
                return {
                    "success": False,
                    "message": "Could not parse verification response",
                    "error": "JSON parsing failed",
                    "raw_response": response
                }
        except json.JSONDecodeError as e:
            return {
                "success": False,
                "message": f"Invalid JSON in verification response: {str(e)}",
                "error": str(e),
                "raw_response": response
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to parse verification response: {str(e)}",
                "error": str(e),
                "raw_response": response
            }
    
    def _verify_execution(self, files_to_execute: List[str]) -> Dict[str, Any]:
        """Execute files to verify they work correctly."""
        execution_results = {}
        
        for file_path in files_to_execute:
            try:
                # Ensure the file path is relative to workspace
                if not Path(file_path).is_absolute():
                    full_path = str(self.workspace_dir / file_path)
                else:
                    full_path = file_path
                
                print(f"🔧 Executing {file_path} for verification...")
                result = run_python_file(path=file_path)
                
                execution_results[file_path] = {
                    "success": result.get("success", False),
                    "stdout": result.get("stdout", ""),
                    "stderr": result.get("stderr", ""),
                    "return_code": result.get("return_code", -1),
                    "duration": result.get("duration", 0)
                }
                
                if result.get("success"):
                    print(f"✓ {file_path} executed successfully")
                else:
                    print(f"✗ {file_path} execution failed: {result.get('message', 'Unknown error')}")
                    
            except Exception as e:
                execution_results[file_path] = {
                    "success": False,
                    "error": str(e),
                    "stdout": "",
                    "stderr": str(e),
                    "return_code": -1,
                    "duration": 0
                }
                print(f"✗ {file_path} execution error: {e}")
        
        return execution_results
    
    def get_verification_summary(self, verification_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of verification results."""
        summary_parts = []
        
        # Overall success
        success = verification_result.get("success", False)
        summary_parts.append(f"Verification: {'✓ PASSED' if success else '✗ FAILED'}")
        
        # Task analysis
        task_analysis = verification_result.get("task_analysis", {})
        if task_analysis:
            required_files = task_analysis.get("required_files", [])
            if required_files:
                summary_parts.append(f"Required files: {', '.join(required_files)}")
        
        # Verification details
        verification = verification_result.get("verification", {})
        if verification:
            missing_files = verification.get("missing_files", [])
            if missing_files:
                summary_parts.append(f"Missing files: {', '.join(missing_files)}")
            
            present_files = verification.get("present_files", [])
            if present_files:
                summary_parts.append(f"Present files: {', '.join(present_files)}")
        
        # Execution verification
        execution_verification = verification_result.get("execution_verification", {})
        if execution_verification:
            successful_executions = [f for f, r in execution_verification.items() if r.get("success")]
            failed_executions = [f for f, r in execution_verification.items() if not r.get("success")]
            
            if successful_executions:
                summary_parts.append(f"Successful executions: {', '.join(successful_executions)}")
            if failed_executions:
                summary_parts.append(f"Failed executions: {', '.join(failed_executions)}")
        
        # Message
        message = verification_result.get("message", "")
        if message:
            summary_parts.append(f"Details: {message}")
        
        return "\n".join(summary_parts) 