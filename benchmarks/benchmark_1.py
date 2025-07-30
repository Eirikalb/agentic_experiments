"""
Benchmark 1: Basic File Creation

This benchmark tests the agent's ability to create files with different content types.
It's a simple test to verify the foundational tools are working correctly.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any

# Import our agent
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agent.coding_agent import CodingAgent


class FileCreationBenchmark:
    """
    Benchmark for testing file creation capabilities.
    """
    
    def __init__(self):
        self.test_results = []
        self.workspace_dir = None
    
    def setup(self) -> Dict[str, Any]:
        """
        Set up the benchmark environment.
        
        Returns:
            dict: Setup status information
        """
        try:
            # Create a temporary workspace
            self.workspace_dir = tempfile.mkdtemp(prefix="benchmark_1_")
            
            return {
                "success": True,
                "message": f"Benchmark workspace created: {self.workspace_dir}",
                "workspace": self.workspace_dir
            }
        except Exception as e:
            return {
                "success": False,
                "message": f"Failed to setup benchmark: {str(e)}",
                "error": str(e)
            }
    
    def cleanup(self):
        """Clean up the benchmark environment."""
        if self.workspace_dir and os.path.exists(self.workspace_dir):
            shutil.rmtree(self.workspace_dir)
    
    def run_test(self, test_name: str, task_description: str, expected_files: list) -> Dict[str, Any]:
        """
        Run a single test case.
        
        Args:
            test_name: Name of the test
            task_description: Task to give to the agent
            expected_files: List of files that should be created
            
        Returns:
            dict: Test result information
        """
        print(f"\n{'='*50}")
        print(f"Running test: {test_name}")
        print(f"Task: {task_description}")
        print(f"{'='*50}")
        
        # Create a fresh agent for this test
        agent = CodingAgent(self.workspace_dir)
        
        # Execute the task
        result = agent.execute_task(task_description)
        
        # Verify the results
        verification_result = self.verify_test(expected_files)
        
        test_result = {
            "test_name": test_name,
            "task_description": task_description,
            "agent_result": result,
            "verification": verification_result,
            "success": result["success"] and verification_result["success"]
        }
        
        self.test_results.append(test_result)
        
        # Print summary
        status = "✓ PASS" if test_result["success"] else "✗ FAIL"
        print(f"\n{status}: {test_name}")
        if not test_result["success"]:
            print(f"  Agent result: {result['message']}")
            print(f"  Verification: {verification_result['message']}")
        
        return test_result
    
    def verify_test(self, expected_files: list) -> Dict[str, Any]:
        """
        Verify that the expected files were created.
        
        Args:
            expected_files: List of expected file paths
            
        Returns:
            dict: Verification result
        """
        try:
            missing_files = []
            existing_files = []
            
            for file_path in expected_files:
                full_path = os.path.join(self.workspace_dir, file_path)
                if os.path.exists(full_path):
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            if missing_files:
                return {
                    "success": False,
                    "message": f"Missing files: {missing_files}",
                    "existing_files": existing_files,
                    "missing_files": missing_files
                }
            else:
                return {
                    "success": True,
                    "message": f"All expected files found: {existing_files}",
                    "existing_files": existing_files
                }
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "error": str(e)
            }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete benchmark suite.
        
        Returns:
            dict: Benchmark results summary
        """
        print("Starting File Creation Benchmark")
        print("=" * 50)
        
        # Setup
        setup_result = self.setup()
        if not setup_result["success"]:
            return {
                "success": False,
                "message": f"Benchmark setup failed: {setup_result['message']}",
                "setup_error": setup_result
            }
        
        try:
            # Test 1: Create a simple Python file
            self.run_test(
                "Create Python Hello World",
                "Create a file called hello.py with a hello world function",
                ["hello.py"]
            )
            
            # Test 2: Create a text file
            self.run_test(
                "Create Text File",
                "Create a file called notes.txt with some content",
                ["notes.txt"]
            )
            
            # Test 3: Create multiple files
            self.run_test(
                "Create Multiple Files",
                "Create a file called main.py and another called config.json",
                ["main.py", "config.json"]
            )
            
            # Calculate results
            total_tests = len(self.test_results)
            passed_tests = sum(1 for result in self.test_results if result["success"])
            failed_tests = total_tests - passed_tests
            
            benchmark_result = {
                "success": failed_tests == 0,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "test_results": self.test_results,
                "workspace": self.workspace_dir
            }
            
            # Print summary
            print(f"\n{'='*50}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*50}")
            print(f"Total tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Success rate: {benchmark_result['success_rate']:.1%}")
            
            if failed_tests > 0:
                print(f"\nFailed tests:")
                for result in self.test_results:
                    if not result["success"]:
                        print(f"  - {result['test_name']}: {result['agent_result']['message']}")
            
            return benchmark_result
            
        finally:
            # Cleanup
            self.cleanup()


def main():
    """Run the benchmark."""
    benchmark = FileCreationBenchmark()
    result = benchmark.run_benchmark()
    
    if result["success"]:
        print(f"\n🎉 All tests passed! Success rate: {result['success_rate']:.1%}")
    else:
        print(f"\n❌ Some tests failed. Success rate: {result['success_rate']:.1%}")
    
    return result


if __name__ == "__main__":
    main() 