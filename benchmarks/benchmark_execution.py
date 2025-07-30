"""
Benchmark: Execution Capabilities

This benchmark tests the agent's ability to create Python files, execute them,
install dependencies, and verify that the code actually works correctly.
"""

import os
import tempfile
import shutil
import json
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

# Import our agent
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agent.llm_agent import LLMCodingAgent, ExecutionConfig
from agent.verification_agent import VerificationAgent, VerificationConfig

# Import storage system
from benchmarks.benchmark_storage import EnhancedBenchmark, BenchmarkMetadata


class ExecutionBenchmark:
    """
    Benchmark for testing execution capabilities including file creation,
    execution, dependency management, and result verification with automatic storage and logging.
    """
    
    def __init__(self, python_executable: str = None, enable_storage: bool = True, run_id: str = None):
        self.test_results = []
        self.workspace_dir = None
        self.python_executable = python_executable
        self.enable_storage = enable_storage
        
        # Configure execution settings
        self.execution_config = ExecutionConfig(
            python_executable=python_executable,
            enable_execution=True,
            execution_timeout=30,
            package_install_timeout=60
        )
        
        if enable_storage:
            self.enhanced_benchmark = EnhancedBenchmark("execution")
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
            self.workspace_dir = tempfile.mkdtemp(prefix="execution_benchmark_")
            
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
    
    def run_test(self, test_name: str, task_description: str, expected_files: List[str], 
                 expected_execution: Dict[str, Any] = None, use_llm_verification: bool = True) -> Dict[str, Any]:
        """
        Run a single test case with complete logging.
        
        Args:
            test_name: Name of the test
            task_description: Task to give to the agent
            expected_files: List of files that should be created
            expected_execution: Expected execution results (optional)
            
        Returns:
            dict: Test result information
        """
        print(f"\n{'='*60}")
        print(f"Running test: {test_name}")
        print(f"Task: {task_description}")
        print(f"{'='*60}")
        
        # Get logger for this test
        test_logger = None
        if self.enable_storage and self.enhanced_benchmark:
            test_logger = self.enhanced_benchmark.get_test_logger(test_name)
            test_logger.start_test()
            test_logger.log_info(f"Starting test: {test_name}")
            test_logger.log_info(f"Task: {task_description}")
            test_logger.log_info(f"Expected files: {expected_files}")
            if expected_execution:
                test_logger.log_info(f"Expected execution: {expected_execution}")
        
        try:
            # Create a fresh agent for this test
            agent = LLMCodingAgent(
                workspace_dir=str(self.workspace_dir),
                execution_config=self.execution_config
            )
            
            # Log agent creation
            if test_logger:
                test_logger.log_info(f"Created execution agent with workspace: {self.workspace_dir}")
                test_logger.log_info(f"Python executable: {self.python_executable or 'default'}")
                test_logger.log_info(f"Execution enabled: {self.execution_config.enable_execution}")
                test_logger.log_info(f"Execution timeout: {self.execution_config.execution_timeout}s")
            
            # Execute the task
            result = agent.execute_task(task_description, max_steps=8)
            
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
            
            # Verify the results using LLM-powered verification
            verification_result = self.verify_test(
                expected_files=expected_files, 
                expected_execution=expected_execution,
                task_description=task_description if use_llm_verification else "",
                agent_result=result if use_llm_verification else None
            )
            
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
    
    def verify_test(self, expected_files: List[str], expected_execution: Dict[str, Any] = None, task_description: str = "", agent_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify that the expected files were created and optionally verify execution.
        Uses LLM-powered verification agent for intelligent verification.
        
        Args:
            expected_files: List of expected file paths (legacy support)
            expected_execution: Expected execution results (optional, legacy support)
            task_description: The original task description for LLM verification
            agent_result: Results from the agent execution for LLM verification
            
        Returns:
            dict: Verification result
        """
        try:
            # Use LLM verification if task description is provided
            if task_description and agent_result:
                verification_agent = VerificationAgent(
                    workspace_dir=str(self.workspace_dir),
                    config=VerificationConfig()
                )
                
                verification_result = verification_agent.verify_task_completion(task_description, agent_result)
                
                # Add legacy compatibility fields
                if verification_result.get("verification", {}).get("present_files"):
                    verification_result["existing_files"] = verification_result["verification"]["present_files"]
                if verification_result.get("verification", {}).get("missing_files"):
                    verification_result["missing_files"] = verification_result["verification"]["missing_files"]
                
                # Add execution verification if available
                if verification_result.get("execution_verification"):
                    execution_results = {}
                    for file_path, result in verification_result["execution_verification"].items():
                        execution_results[file_path] = {
                            "success": result.get("success", False),
                            "stdout": result.get("stdout", ""),
                            "stderr": result.get("stderr", ""),
                            "return_code": result.get("return_code", -1),
                            "duration": result.get("duration", 0)
                        }
                    verification_result["execution_results"] = execution_results
                    verification_result["execution_success"] = all(r.get("success", False) for r in execution_results.values())
                
                return verification_result
            
            # Fallback to legacy verification
            missing_files = []
            existing_files = []
            
            # Check for expected files
            for file_path in expected_files:
                full_path = os.path.join(self.workspace_dir, file_path)
                if os.path.exists(full_path):
                    existing_files.append(file_path)
                else:
                    missing_files.append(file_path)
            
            verification_result = {
                "success": len(missing_files) == 0,
                "existing_files": existing_files,
                "missing_files": missing_files
            }
            
            # If execution verification is requested, check the files
            if expected_execution and existing_files:
                execution_verification = self._verify_execution(existing_files, expected_execution)
                verification_result.update(execution_verification)
                verification_result["success"] = verification_result["success"] and execution_verification.get("execution_success", True)
            
            if missing_files:
                verification_result["message"] = f"Missing files: {missing_files}"
            else:
                verification_result["message"] = f"All expected files found: {existing_files}"
                if expected_execution:
                    verification_result["message"] += f" and execution verified"
            
            return verification_result
                
        except Exception as e:
            return {
                "success": False,
                "message": f"Verification failed: {str(e)}",
                "error": str(e)
            }
    
    def _verify_execution(self, existing_files: List[str], expected_execution: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify that the created files can be executed successfully.
        
        Args:
            existing_files: List of existing files
            expected_execution: Expected execution results
            
        Returns:
            dict: Execution verification result
        """
        try:
            # Find Python files to execute
            python_files = [f for f in existing_files if f.endswith('.py')]
            
            if not python_files:
                return {
                    "execution_success": True,
                    "message": "No Python files to execute"
                }
            
            # Try to execute each Python file
            execution_results = {}
            all_successful = True
            
            for py_file in python_files:
                file_path = os.path.join(self.workspace_dir, py_file)
                
                # Use the agent's execution tools to run the file
                agent = LLMCodingAgent(
                    workspace_dir=str(self.workspace_dir),
                    execution_config=self.execution_config
                )
                
                # Execute the file
                result = agent.tools["run_python_file"](file_path)
                execution_results[py_file] = result
                
                if not result.get("success", False):
                    all_successful = False
                    print(f"  ❌ Execution failed for {py_file}: {result.get('message', 'Unknown error')}")
                    if result.get("stderr"):
                        print(f"    Error: {result['stderr']}")
                else:
                    print(f"  ✅ {py_file} executed successfully")
                    if result.get("stdout"):
                        print(f"    Output: {result['stdout'].strip()}")
            
            return {
                "execution_success": all_successful,
                "execution_results": execution_results,
                "message": f"Execution verification: {'All successful' if all_successful else 'Some failed'}"
            }
            
        except Exception as e:
            return {
                "execution_success": False,
                "error": str(e),
                "message": f"Execution verification failed: {str(e)}"
            }
    
    def run_benchmark(self) -> Dict[str, Any]:
        """
        Run the complete execution benchmark suite.
        
        Returns:
            dict: Benchmark results summary
        """
        print("Starting Execution Capabilities Benchmark")
        print("=" * 60)
        
        # Setup
        setup_result = self.setup()
        if not setup_result["success"]:
            return {
                "success": False,
                "message": f"Benchmark setup failed: {setup_result['message']}",
                "setup_error": setup_result
            }
        
        try:
            # Test 1: Create and execute a simple Python file
            self.run_test(
                "Simple Python Execution",
                "Create a file called hello.py with a function that prints 'Hello, World!' and returns 'Success!', then run it to verify it works",
                ["hello.py"],
                {"should_execute": True}
            )
            
            # Test 2: Create a calculator with multiple functions
            self.run_test(
                "Calculator with Functions",
                "Create a file called calculator.py with a Calculator class that has add, subtract, multiply, and divide methods. Include a test function that creates an instance and tests all methods, then run it to verify it works",
                ["calculator.py"],
                {"should_execute": True}
            )
            
            # Test 3: Create a file with external dependencies
            self.run_test(
                "External Dependencies",
                "Create a file called data_processor.py that uses the 'requests' library to make a simple HTTP request. The file should import requests, define a function that makes a GET request to 'https://httpbin.org/get', and print the status code. Then install the requests package and run the file to verify it works",
                ["data_processor.py"],
                {"should_execute": True, "requires_dependencies": ["requests"]}
            )
            
            # Test 4: Create a multi-file project
            self.run_test(
                "Multi-file Project",
                "Create a simple project with main.py, utils.py, and test_main.py. main.py should import from utils.py and have a main function. utils.py should have some utility functions. test_main.py should test the main function. Then run test_main.py to verify everything works",
                ["main.py", "utils.py", "test_main.py"],
                {"should_execute": True}
            )
            
            # Test 5: Create a file with error handling
            self.run_test(
                "Error Handling",
                "Create a file called error_demo.py with a function that demonstrates proper error handling using try-except blocks. The function should attempt to divide by zero, catch the exception, and print a meaningful error message. Then run it to verify the error handling works",
                ["error_demo.py"],
                {"should_execute": True}
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
                "run_id": self.run_id,
                "execution_config": {
                    "python_executable": self.python_executable,
                    "enable_execution": self.execution_config.enable_execution,
                    "execution_timeout": self.execution_config.execution_timeout,
                    "package_install_timeout": self.execution_config.package_install_timeout
                }
            }
            
            # Store the benchmark results if storage is enabled
            if self.enable_storage and self.enhanced_benchmark:
                metadata = BenchmarkMetadata(
                    run_id=self.run_id,
                    timestamp=datetime.now().isoformat(),
                    benchmark_name="execution",
                    total_tests=total_tests,
                    passed_tests=passed_tests,
                    success_rate=benchmark_result["success_rate"],
                    workspace=self.workspace_dir
                )
                
                self.enhanced_benchmark.end_run(metadata, benchmark_result)
            
            # Print summary
            print(f"\n{'='*60}")
            print("EXECUTION BENCHMARK SUMMARY")
            print(f"{'='*60}")
            print(f"Total tests: {total_tests}")
            print(f"Passed: {passed_tests}")
            print(f"Failed: {failed_tests}")
            print(f"Success rate: {benchmark_result['success_rate']:.1%}")
            print(f"Python executable: {self.python_executable or 'default'}")
            
            if self.run_id:
                print(f"Run ID: {self.run_id}")
            
            if failed_tests > 0:
                print(f"\nFailed tests:")
                for result in self.test_results:
                    if not result["success"]:
                        print(f"  - {result['test_name']}: {result['agent_result']['message']}")
                        if result['verification'].get('missing_files'):
                            print(f"    Missing: {result['verification']['missing_files']}")
                        if result['verification'].get('execution_results'):
                            for file, exec_result in result['verification']['execution_results'].items():
                                if not exec_result.get('success'):
                                    print(f"    Execution failed for {file}: {exec_result.get('message', 'Unknown error')}")
            
            return benchmark_result
            
        finally:
            # Cleanup
            self.cleanup()


def main():
    """Run the execution benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run Execution Capabilities Benchmark")
    parser.add_argument(
        "--python-executable",
        type=str,
        help="Path to Python executable to use for execution"
    )
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
    parser.add_argument(
        "--output",
        type=str,
        default="execution_benchmark_results.json",
        help="Output file for results (legacy option)"
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
                errors = log_summary.get('errors', 0) if log_summary else 0
                
                print(f"  {test.get('test_name', 'unknown')}: {status}")
                if log_summary:
                    print(f"    Duration: {duration:.2f}s, Steps: {steps}, Errors: {errors}")
                
                # Show agent result details
                agent_result = test.get('agent_result', {})
                if agent_result:
                    message = agent_result.get('message', 'No message')
                    steps_taken = agent_result.get('steps_taken', 0)
                    print(f"    Agent: {message} ({steps_taken} steps)")
                
                # Show verification details
                verification = test.get('verification', {})
                if verification:
                    v_message = verification.get('message', 'No verification message')
                    print(f"    Verification: {v_message}")
                
                print()  # Empty line between tests
        except ValueError as e:
            print(f"❌ Error: {e}")
        return
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Check API key
    import os
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: No OPENROUTER_API_KEY found in environment")
        print("   Set the API key in your .env file or environment variables")
        return {"success": False, "error": "No API key"}
    
    print("🔑 API key loaded successfully")
    
    # Run benchmark
    benchmark = ExecutionBenchmark(
        python_executable=args.python_executable,
        enable_storage=not args.no_storage,
        run_id=args.run_id
    )
    result = benchmark.run_benchmark()
    
    # Save results (legacy support)
    if args.output and not args.no_storage:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Legacy results saved to: {args.output}")
    
    if result["success"]:
        print(f"\n🎉 All tests passed! Success rate: {result['success_rate']:.1%}")
    else:
        print(f"\n❌ Some tests failed. Success rate: {result['success_rate']:.1%}")
    
    return result


if __name__ == "__main__":
    main() 