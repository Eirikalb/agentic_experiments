"""
Task Suite for LLM Agent Evaluation

This module contains 10 diverse coding tasks to evaluate the LLM agent's
performance and identify areas for prompt optimization.
"""

from typing import Dict, List, Any
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TaskDefinition:
    """Definition of a single evaluation task."""
    id: str
    name: str
    description: str
    expected_files: List[str]
    expected_content_patterns: List[str] = None
    difficulty: str = "medium"  # easy, medium, hard
    category: str = "file_creation"  # file_creation, file_modification, complex_task
    max_steps: int = 5


class TaskSuite:
    """Collection of evaluation tasks for LLM agent testing."""
    
    def __init__(self):
        self.tasks = self._create_tasks()
    
    def _create_tasks(self) -> List[TaskDefinition]:
        """Create the 10 evaluation tasks."""
        return [
            # Task 1: Simple file creation
            TaskDefinition(
                id="task_1",
                name="Simple Python File",
                description="Create a file called hello.py with a hello world function",
                expected_files=["hello.py"],
                expected_content_patterns=["def hello", "print", "Hello"],
                difficulty="easy",
                category="file_creation"
            ),
            
            # Task 2: Multiple file creation
            TaskDefinition(
                id="task_2", 
                name="Multiple Files",
                description="Create a file called main.py and another called config.json",
                expected_files=["main.py", "config.json"],
                difficulty="easy",
                category="file_creation"
            ),
            
            # Task 3: File with specific content
            TaskDefinition(
                id="task_3",
                name="Calculator Class",
                description="Create a Python file called calculator.py with a simple calculator class that has add, subtract, multiply, and divide methods",
                expected_files=["calculator.py"],
                expected_content_patterns=["class Calculator", "def add", "def subtract", "def multiply", "def divide"],
                difficulty="medium",
                category="file_creation"
            ),
            
            # Task 4: Complex file structure
            TaskDefinition(
                id="task_4",
                name="Web Server",
                description="Create a Python file called server.py with a simple Flask web server that serves a hello world page",
                expected_files=["server.py"],
                expected_content_patterns=["from flask", "app = Flask", "@app.route", "hello world"],
                difficulty="medium",
                category="file_creation"
            ),
            
            # Task 5: File modification
            TaskDefinition(
                id="task_5",
                name="File Modification",
                description="Create a file called data.txt with some content, then read it and create a summary file called summary.txt",
                expected_files=["data.txt", "summary.txt"],
                difficulty="medium",
                category="file_modification"
            ),
            
            # Task 6: Configuration file
            TaskDefinition(
                id="task_6",
                name="JSON Config",
                description="Create a config.json file with settings for a web application including port, host, and debug mode",
                expected_files=["config.json"],
                expected_content_patterns=["port", "host", "debug"],
                difficulty="easy",
                category="file_creation"
            ),
            
            # Task 7: Multi-step task
            TaskDefinition(
                id="task_7",
                name="Project Setup",
                description="Create a basic Python project structure with main.py, requirements.txt, and a README.md file",
                expected_files=["main.py", "requirements.txt", "README.md"],
                difficulty="medium",
                category="complex_task"
            ),
            
            # Task 8: Error handling
            TaskDefinition(
                id="task_8",
                name="Error Handling",
                description="Create a Python file called error_handler.py with a function that demonstrates try-catch error handling",
                expected_files=["error_handler.py"],
                expected_content_patterns=["try:", "except", "def"],
                difficulty="medium",
                category="file_creation"
            ),
            
            # Task 9: Data structure
            TaskDefinition(
                id="task_9",
                name="Data Structure",
                description="Create a Python file called data_structures.py with implementations of a stack and queue class",
                expected_files=["data_structures.py"],
                expected_content_patterns=["class Stack", "class Queue", "def push", "def pop"],
                difficulty="hard",
                category="file_creation"
            ),
            
            # Task 10: Execution-focused task
            TaskDefinition(
                id="task_10",
                name="Executable Calculator",
                description="Create a file called calculator.py with a Calculator class that has add, subtract, multiply, and divide methods. Include a main function that creates an instance and tests all methods, then run it to verify it works",
                expected_files=["calculator.py"],
                expected_content_patterns=["class Calculator", "def add", "def subtract", "def multiply", "def divide", "if __name__"],
                difficulty="medium",
                category="execution_task",
                max_steps=8
            ),
            
            # Task 11: Dependency management
            TaskDefinition(
                id="task_11",
                name="HTTP Client",
                description="Create a file called http_client.py that uses the 'requests' library to make a simple HTTP request. The file should import requests, define a function that makes a GET request to 'https://httpbin.org/get', and print the status code. Then install the requests package and run the file to verify it works",
                expected_files=["http_client.py"],
                expected_content_patterns=["import requests", "def", "requests.get", "print"],
                difficulty="medium",
                category="execution_task",
                max_steps=8
            ),
            
            # Task 12: Multi-file execution
            TaskDefinition(
                id="task_12",
                name="Multi-file Project",
                description="Create a simple project with main.py, utils.py, and test_main.py. main.py should import from utils.py and have a main function. utils.py should have some utility functions. test_main.py should test the main function. Then run test_main.py to verify everything works",
                expected_files=["main.py", "utils.py", "test_main.py"],
                expected_content_patterns=["import", "def", "if __name__"],
                difficulty="hard",
                category="execution_task",
                max_steps=10
            ),
            
            # Task 13: Error handling execution
            TaskDefinition(
                id="task_13",
                name="Error Handler",
                description="Create a file called error_demo.py with a function that demonstrates proper error handling using try-except blocks. The function should attempt to divide by zero, catch the exception, and print a meaningful error message. Then run it to verify the error handling works",
                expected_files=["error_demo.py"],
                expected_content_patterns=["try:", "except", "ZeroDivisionError", "print"],
                difficulty="medium",
                category="execution_task",
                max_steps=8
            ),
            
            # Task 14: Complex execution task
            TaskDefinition(
                id="task_14",
                name="Data Processor",
                description="Create a file called data_processor.py that processes a list of numbers. The file should have functions to calculate mean, median, and standard deviation. Include a main function that creates sample data and tests all functions, then run it to verify it works",
                expected_files=["data_processor.py"],
                expected_content_patterns=["def mean", "def median", "def std_dev", "import math", "if __name__"],
                difficulty="hard",
                category="execution_task",
                max_steps=10
            )
        ]
    
    def get_task(self, task_id: str) -> TaskDefinition:
        """Get a specific task by ID."""
        for task in self.tasks:
            if task.id == task_id:
                return task
        raise ValueError(f"Task {task_id} not found")
    
    def get_tasks_by_difficulty(self, difficulty: str) -> List[TaskDefinition]:
        """Get all tasks of a specific difficulty."""
        return [task for task in self.tasks if task.difficulty == difficulty]
    
    def get_tasks_by_category(self, category: str) -> List[TaskDefinition]:
        """Get all tasks of a specific category."""
        return [task for task in self.tasks if task.category == category]
    
    def get_all_tasks(self) -> List[TaskDefinition]:
        """Get all tasks."""
        return self.tasks.copy()


if __name__ == "__main__":
    # Test the task suite
    suite = TaskSuite()
    print(f"Created {len(suite.tasks)} evaluation tasks:")
    for task in suite.tasks:
        print(f"  {task.id}: {task.name} ({task.difficulty})") 