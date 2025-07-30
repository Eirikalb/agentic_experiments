"""
File manipulation tools for the coding agent.

This module provides a set of tools that allow the agent to interact with
the file system in a controlled and safe manner.
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional


def create_file(path: str, content: str = "") -> dict:
    """
    Create a new file with optional content.
    
    Args:
        path: The file path to create
        content: The content to write to the file (default: empty string)
    
    Returns:
        dict: Status information about the operation
    """
    try:
        # Ensure the directory exists
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create the file with content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return {
            "success": True,
            "message": f"File '{path}' created successfully",
            "path": str(file_path.absolute())
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to create file '{path}': {str(e)}",
            "error": str(e)
        }


def read_file(path: str) -> dict:
    """
    Read the content of a file.
    
    Args:
        path: The file path to read
    
    Returns:
        dict: File content and status information
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {
                "success": False,
                "message": f"File '{path}' does not exist",
                "error": "FileNotFoundError"
            }
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            "success": True,
            "message": f"File '{path}' read successfully",
            "content": content,
            "path": str(file_path.absolute()),
            "size": len(content)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to read file '{path}': {str(e)}",
            "error": str(e)
        }


def edit_file(path: str, new_content: str) -> dict:
    """
    Replace the content of an existing file.
    
    Args:
        path: The file path to edit
        new_content: The new content to write to the file
    
    Returns:
        dict: Status information about the operation
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {
                "success": False,
                "message": f"File '{path}' does not exist",
                "error": "FileNotFoundError"
            }
        
        # Read the original content for backup
        with open(file_path, 'r', encoding='utf-8') as f:
            original_content = f.read()
        
        # Write the new content
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(new_content)
        
        return {
            "success": True,
            "message": f"File '{path}' edited successfully",
            "path": str(file_path.absolute()),
            "original_size": len(original_content),
            "new_size": len(new_content)
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to edit file '{path}': {str(e)}",
            "error": str(e)
        }


def list_files(directory: str = ".") -> dict:
    """
    List all files and directories in a given directory.
    
    Args:
        directory: The directory to list (default: current directory)
    
    Returns:
        dict: List of files and directories with status information
    """
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return {
                "success": False,
                "message": f"Directory '{directory}' does not exist",
                "error": "DirectoryNotFoundError"
            }
        
        if not dir_path.is_dir():
            return {
                "success": False,
                "message": f"'{directory}' is not a directory",
                "error": "NotADirectoryError"
            }
        
        items = []
        for item in dir_path.iterdir():
            item_info = {
                "name": item.name,
                "path": str(item.relative_to(dir_path)),
                "is_file": item.is_file(),
                "is_dir": item.is_dir(),
                "size": item.stat().st_size if item.is_file() else None
            }
            items.append(item_info)
        
        # Sort items: directories first, then files, both alphabetically
        items.sort(key=lambda x: (not x["is_dir"], x["name"].lower()))
        
        return {
            "success": True,
            "message": f"Directory '{directory}' listed successfully",
            "items": items,
            "total_files": len([item for item in items if item["is_file"]]),
            "total_dirs": len([item for item in items if item["is_dir"]])
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to list directory '{directory}': {str(e)}",
            "error": str(e)
        }


def delete_file(path: str) -> dict:
    """
    Delete a file.
    
    Args:
        path: The file path to delete
    
    Returns:
        dict: Status information about the operation
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
        
        file_path.unlink()
        
        return {
            "success": True,
            "message": f"File '{path}' deleted successfully"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to delete file '{path}': {str(e)}",
            "error": str(e)
        }


def apply_diff(path: str, diff_content: str) -> dict:
    """
    Apply a diff to a file.
    
    Args:
        path: The file path to apply the diff to
        diff_content: The diff content in unified diff format
    
    Returns:
        dict: Status information about the operation
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {
                "success": False,
                "message": f"File '{path}' does not exist",
                "error": "FileNotFoundError"
            }
        
        # For now, we'll implement a simple diff application
        # In a more sophisticated version, we'd use a proper diff library
        lines = diff_content.split('\n')
        current_content = read_file(path)
        
        if not current_content["success"]:
            return current_content
        
        file_lines = current_content["content"].split('\n')
        
        # Simple diff parsing and application
        # This is a basic implementation - in practice you'd want a proper diff library
        new_lines = file_lines.copy()
        
        for line in lines:
            if line.startswith('+') and not line.startswith('+++'):
                # Add line
                new_lines.append(line[1:])
            elif line.startswith('-') and not line.startswith('---'):
                # Remove line (simplified - would need line numbers in real implementation)
                pass
        
        new_content = '\n'.join(new_lines)
        
        # Apply the changes
        return edit_file(path, new_content)
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to apply diff to file '{path}': {str(e)}",
            "error": str(e)
        }


def read_diff(path: str) -> dict:
    """
    Read the diff of a file (show uncommitted changes).
    
    Args:
        path: The file path to get diff for
    
    Returns:
        dict: Diff content and status information
    """
    try:
        file_path = Path(path)
        if not file_path.exists():
            return {
                "success": False,
                "message": f"File '{path}' does not exist",
                "error": "FileNotFoundError"
            }
        
        # For now, we'll return a simple diff format
        # In a more sophisticated version, we'd compare with git or previous versions
        content = read_file(path)
        
        if not content["success"]:
            return content
        
        # Create a simple diff showing the current content
        diff_content = f"""--- {path} (original)
+++ {path} (current)
@@ -1,{len(content['content'].split(chr(10)))} +1,{len(content['content'].split(chr(10)))} @@
{chr(10).join('+' + line for line in content['content'].split(chr(10)))}
"""
        
        return {
            "success": True,
            "message": f"Diff for '{path}' generated successfully",
            "diff": diff_content,
            "path": str(file_path.absolute())
        }
        
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to read diff for file '{path}': {str(e)}",
            "error": str(e)
        } 