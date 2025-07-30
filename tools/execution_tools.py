"""
Execution tools for the coding agent.

This module provides tools for executing Python files and managing packages
in a controlled environment using a provided virtual environment.
"""

import os
import sys
import subprocess
import time
import ast
from pathlib import Path
from typing import Dict, Any, List, Optional, Set


def run_python_file(path: str, python_executable: str = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Execute a Python file using the specified Python interpreter.
    
    Args:
        path: Path to the Python file to execute
        python_executable: Path to Python interpreter (default: sys.executable)
        timeout: Maximum execution time in seconds
    
    Returns:
        dict: Execution results including stdout, stderr, return_code, duration
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {
                "success": False,
                "message": f"File '{path}' does not exist",
                "error": "FileNotFoundError"
            }
        
        if not file_path.is_file():
            return {
                "success": False,
                "message": f"'{path}' is not a file",
                "error": "NotAFileError"
            }
        
        # Use provided Python executable or default
        python_cmd = python_executable or sys.executable
        
        # Prepare command
        cmd = [python_cmd, str(file_path.absolute())]
        
        # Execute with timeout
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=file_path.parent  # Run from the file's directory
            )
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "message": f"File '{path}' executed successfully" if result.returncode == 0 else f"File '{path}' execution failed",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": duration,
                "path": str(file_path.absolute())
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "message": f"Execution of '{path}' timed out after {timeout} seconds",
                "error": "TimeoutExpired",
                "duration": duration,
                "path": str(file_path.absolute())
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to execute file '{path}': {str(e)}",
            "error": str(e)
        }


def detect_imports(python_file: str) -> Dict[str, Any]:
    """
    Detect Python imports from a file using AST parsing.
    
    Args:
        python_file: Path to the Python file to analyze
    
    Returns:
        dict: List of detected imports and status information
    """
    try:
        file_path = Path(python_file)
        if not file_path.exists():
            return {
                "success": False,
                "message": f"File '{python_file}' does not exist",
                "error": "FileNotFoundError"
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Parse AST to find imports
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            return {
                "success": False,
                "message": f"Syntax error in '{python_file}': {str(e)}",
                "error": "SyntaxError",
                "imports": []
            }
        
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
        
        # Filter out standard library modules (basic list)
        stdlib_modules = {
            'os', 'sys', 'time', 'datetime', 'pathlib', 'json', 'argparse',
            'tempfile', 'shutil', 'subprocess', 'typing', 'dataclasses',
            'collections', 'itertools', 'functools', 're', 'math', 'random'
        }
        
        external_imports = [imp for imp in imports if imp not in stdlib_modules]
        
        return {
            "success": True,
            "message": f"Imports detected in '{python_file}'",
            "all_imports": list(imports),
            "external_imports": external_imports,
            "stdlib_imports": list(imports & stdlib_modules),
            "path": str(file_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to detect imports in '{python_file}': {str(e)}",
            "error": str(e)
        }


def install_package(package: str, python_executable: str = None, timeout: int = 60) -> Dict[str, Any]:
    """
    Install a Python package using pip.
    
    Args:
        package: Package name to install (e.g., "requests", "numpy==1.21.0")
        python_executable: Path to Python interpreter (default: sys.executable)
        timeout: Maximum installation time in seconds
    
    Returns:
        dict: Installation results
    """
    try:
        # Use provided Python executable or default
        python_cmd = python_executable or sys.executable
        
        # Prepare pip install command
        cmd = [python_cmd, "-m", "pip", "install", package]
        
        # Execute with timeout
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "message": f"Package '{package}' installed successfully" if result.returncode == 0 else f"Failed to install package '{package}'",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": duration,
                "package": package
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "message": f"Installation of '{package}' timed out after {timeout} seconds",
                "error": "TimeoutExpired",
                "duration": duration,
                "package": package
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to install package '{package}': {str(e)}",
            "error": str(e),
            "package": package
        }


def install_requirements(requirements_file: str, python_executable: str = None, timeout: int = 120) -> Dict[str, Any]:
    """
    Install packages from a requirements.txt file.
    
    Args:
        requirements_file: Path to requirements.txt file
        python_executable: Path to Python interpreter (default: sys.executable)
        timeout: Maximum installation time in seconds
    
    Returns:
        dict: Installation results
    """
    try:
        req_path = Path(requirements_file)
        if not req_path.exists():
            return {
                "success": False,
                "message": f"Requirements file '{requirements_file}' does not exist",
                "error": "FileNotFoundError"
            }
        
        # Use provided Python executable or default
        python_cmd = python_executable or sys.executable
        
        # Prepare pip install command
        cmd = [python_cmd, "-m", "pip", "install", "-r", str(req_path.absolute())]
        
        # Execute with timeout
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "message": f"Requirements from '{requirements_file}' installed successfully" if result.returncode == 0 else f"Failed to install requirements from '{requirements_file}'",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": duration,
                "requirements_file": str(req_path.absolute())
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "message": f"Installation of requirements from '{requirements_file}' timed out after {timeout} seconds",
                "error": "TimeoutExpired",
                "duration": duration,
                "requirements_file": str(req_path.absolute())
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to install requirements from '{requirements_file}': {str(e)}",
            "error": str(e),
            "requirements_file": requirements_file
        }


def run_command(command: str, cwd: str = None, timeout: int = 30, python_executable: str = None) -> Dict[str, Any]:
    """
    Execute a shell command.
    
    Args:
        command: Command to execute
        cwd: Working directory for execution (default: current directory)
        timeout: Maximum execution time in seconds
        python_executable: Path to Python interpreter (for Python-specific commands)
    
    Returns:
        dict: Execution results
    """
    try:
        # Execute with timeout
        start_time = time.time()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=cwd
            )
            duration = time.time() - start_time
            
            return {
                "success": result.returncode == 0,
                "message": f"Command executed successfully" if result.returncode == 0 else f"Command failed with return code {result.returncode}",
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode,
                "duration": duration,
                "command": command,
                "cwd": cwd or os.getcwd()
            }
            
        except subprocess.TimeoutExpired:
            duration = time.time() - start_time
            return {
                "success": False,
                "message": f"Command timed out after {timeout} seconds",
                "error": "TimeoutExpired",
                "duration": duration,
                "command": command,
                "cwd": cwd or os.getcwd()
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to execute command: {str(e)}",
            "error": str(e),
            "command": command,
            "cwd": cwd or os.getcwd()
        }


def check_python_version(python_executable: str = None) -> Dict[str, Any]:
    """
    Check the Python version of the specified interpreter.
    
    Args:
        python_executable: Path to Python interpreter (default: sys.executable)
    
    Returns:
        dict: Version information
    """
    try:
        # Use provided Python executable or default
        python_cmd = python_executable or sys.executable
        
        # Get version info
        result = subprocess.run(
            [python_cmd, "--version"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            version_output = result.stdout.strip()
            return {
                "success": True,
                "message": f"Python version: {version_output}",
                "version": version_output,
                "executable": python_cmd
            }
        else:
            return {
                "success": False,
                "message": f"Failed to get Python version: {result.stderr}",
                "error": result.stderr,
                "executable": python_cmd
            }
            
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to check Python version: {str(e)}",
            "error": str(e),
            "executable": python_executable or sys.executable
        } 