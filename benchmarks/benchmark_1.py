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
from datetime import datetime

# Import our agent
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agent.coding_agent import CodingAgent

# Import storage system
from benchmarks.benchmark_storage import EnhancedBenchmark, BenchmarkMetadata


class FileCreationBenchmark:
    """
    Benchmark for testing file creation capabilities with automatic storage and logging.
    """
    
    def __init__(self, enable_storage: bool = True, run_id: str = None):
        self.test_results = []
        self.workspace_dir = None
        self.enable_storage = enable_storage
        
        if enable_storage:
            self.enhanced_benchmark = EnhancedBenchmark("file_creation")
            self.run_id = self.enhanced_benchmark.start_run(run_id)
        else:
            self.enhanced_benchmark = None
            self.run_id = None
    
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
        Run a single test case with complete logging.
        
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
        
        # Get logger for this test
        test_logger = None
        if self.enable_storage and self.enhanced_benchmark:
            test_logger = self.enhanced_benchmark.get_test_logger(test_name)
            test_logger.start_test()
            test_logger.log_info(f"Starting test: {test_name}")
            test_logger.log_info(f"Task: {task_description}")
            test_logger.log_info(f"Expected files: {expected_files}")
        
        try:
            # Create a fresh agent for this test
            agent = CodingAgent(self.workspace_dir)
            
            # Log agent creation
            if test_logger:
                test_logger.log_info(f"Created agent with workspace: {self.workspace_dir}")
            
            # Execute the task
            result = agent.execute_task(task_description)
            
            # Log agent result
            if test_logger:
                test_logger.log_info(f"Agent execution completed: {result['success']}")
                test_logger.log_info(f"Agent message: {result.get('message', 'No message')}")
                test_logger.log_info(f"Steps taken: {result.get('steps_taken', 0)}")
                
                # Log agent steps if available
                execution_history = result.get('execution_history', [])
                if execution_history:
                    for i, step in enumerate(execution_history, 1):
                        test_logger.log_agent_step(
                            step_number=i,
                            prompt=f"Executing tool: {step.get('tool', 'unknown')}",
                            response=f"Tool result: {step.get('result', {}).get('message', 'No message')}",
                            tool_call={
                                "tool": step.get('tool', 'unknown'),
                                "parameters": step.get('parameters', {})
                            },
                            tool_result=step.get('result', {})
                        )
                else:
                    # Fallback: create a single step log
                    test_logger.log_agent_step(
                        step_number=1,
                        prompt=f"Executing task: {task_description}",
                        response=f"Task result: {result.get('message', 'No message')}",
                        tool_call={"tool": "execute_task", "parameters": {"task": task_description}},
                        tool_result=result
                    )
            
            # Verify the results
            verification_result = self.verify_test(expected_files)
            
            # Log verification
            if test_logger:
                test_logger.log_verification(verification_result)
            
            test_result = {
                "test_name": test_name,
                "task_description": task_description,
                "agent_result": result,
                "verification": verification_result,
                "success": result["success"] and verification_result["success"]
            }
            
            self.test_results.append(test_result)
            
            # Store the test result if storage is enabled
            if self.enable_storage and self.enhanced_benchmark:
                workspace_path = Path(self.workspace_dir) if self.workspace_dir else None
                self.enhanced_benchmark.store_test_result(test_result, workspace_path)
            
            # End logging
            if test_logger:
                test_logger.end_test()
            
            # Print summary
            status = "✓ PASS" if test_result["success"] else "✗ FAIL"
            print(f"\n{status}: {test_name}")
            if not test_result["success"]:
                print(f"  Agent result: {result['message']}")
                print(f"  Verification: {verification_result['message']}")
            
            return test_result
            
        except Exception as e:
            # Log error
            if test_logger:
                test_logger.log_error(f"Test failed with exception: {str(e)}", "exception")
                test_logger.end_test()
            
            # Create error result
            error_result = {
                "test_name": test_name,
                "task_description": task_description,
                "agent_result": {"success": False, "message": f"Exception: {str(e)}"},
                "verification": {"success": False, "message": f"Exception: {str(e)}"},
                "success": False
            }
            
            self.test_results.append(error_result)
            
            # Store error result
            if self.enable_storage and self.enhanced_benchmark:
                workspace_path = Path(self.workspace_dir) if self.workspace_dir else None
                self.enhanced_benchmark.store_test_result(error_result, workspace_path)
            
            print(f"\n✗ FAIL: {test_name}")
            print(f"  Exception: {str(e)}")
            
            return error_result
    
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
                "workspace": self.workspace_dir,
                "run_id": self.run_id
            }
            
            # Store the benchmark results if storage is enabled
            if self.enable_storage and self.enhanced_benchmark:
                metadata = BenchmarkMetadata(
                    run_id=self.run_id,
                    timestamp=datetime.now().isoformat(),
                    benchmark_name="file_creation",
                    total_tests=total_tests,
                    passed_tests=passed_tests,
                    success_rate=benchmark_result["success_rate"],
                    workspace=self.workspace_dir
                )
                
                self.enhanced_benchmark.end_run(metadata, benchmark_result)
            
            # Print summary
            print(f"\n{'='*50}")
            print("BENCHMARK SUMMARY")
            print(f"{'='*50}")
            print(f"Total tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Success rate: {benchmark_result['success_rate']:.1%}")
            
            if self.run_id:
                print(f"Run ID: {self.run_id}")
            
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
    import argparse
    
    parser = argparse.ArgumentParser(description="Run File Creation Benchmark")
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Disable automatic storage of results"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run ID for storage"
    )
    parser.add_argument(
        "--list-runs",
        action="store_true",
        help="List all previous benchmark runs"
    )
    parser.add_argument(
        "--show-run",
        type=str,
        help="Show details of a specific run"
    )
    
    args = parser.parse_args()
    
    if args.list_runs:
        from benchmarks.benchmark_storage import BenchmarkStorage
        storage = BenchmarkStorage()
        runs = storage.list_benchmark_runs()
        
        print("\n📁 Available Benchmark Runs:")
        print("=" * 50)
        if runs:
            for run in runs:
                print(f"  {run['run_id']}: {run['success_rate']:.1f}% success rate")
                print(f"    Benchmark: {run['benchmark_name']}, Tests: {run['total_tests']}")
                print(f"    Timestamp: {run['timestamp']}")
                print(f"    Path: {run['path']}")
                print()
        else:
            print("  No previous runs found.")
        return
    
    if args.show_run:
        from benchmarks.benchmark_storage import BenchmarkStorage
        storage = BenchmarkStorage()
        try:
            details = storage.get_benchmark_run_details(args.show_run)
            metadata = details['metadata']
            print(f"\n📋 Run Details: {args.show_run}")
            print("=" * 50)
            print(f"Benchmark: {metadata.get('benchmark_name', 'unknown')}")
            print(f"Success Rate: {metadata.get('success_rate', 0):.1f}%")
            print(f"Total Tests: {metadata.get('total_tests', 0)}")
            print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
            
            print("\nTest Results:")
            for test in details['tests']:
                status = "PASS" if test.get('success', False) else "FAIL"
                log_summary = test.get('log_summary', {})
                duration = log_summary.get('duration', 0) if log_summary else 0
                steps = log_summary.get('agent_steps', 0) if log_summary else 0
                print(f"  {test.get('test_name', 'unknown')}: {status} ({duration:.2f}s, {steps} steps)")
        except ValueError as e:
            print(f"❌ Error: {e}")
        return
    
    # Run benchmark
    benchmark = FileCreationBenchmark(
        enable_storage=not args.no_storage,
        run_id=args.run_id
    )
    result = benchmark.run_benchmark()
    
    if result["success"]:
        print(f"\n🎉 All tests passed! Success rate: {result['success_rate']:.1%}")
    else:
        print(f"\n❌ Some tests failed. Success rate: {result['success_rate']:.1%}")
    
    return result


if __name__ == "__main__":
    main() 