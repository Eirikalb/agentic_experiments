#!/usr/bin/env python3
"""
Reliable Benchmark Runner with LLM-Powered Verification

This script runs benchmarks with intelligent verification using the verification agent,
ensuring reliable and accurate assessment of task completion.
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, Any, List

# Import benchmarks and verification
from benchmarks.benchmark_execution import ExecutionBenchmark
from agent.verification_agent import VerificationAgent, VerificationConfig

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ReliableBenchmarkRunner:
    """
    Reliable benchmark runner that uses LLM-powered verification
    for accurate task completion assessment.
    """
    
    def __init__(self, python_executable: str = None, enable_storage: bool = True, run_id: str = None):
        self.python_executable = python_executable
        self.enable_storage = enable_storage
        self.run_id = run_id
        self.results = {}
        
        # Initialize verification agent
        self.verification_agent = None  # Will be initialized per workspace
        
        # Check API key
        api_key = os.getenv('OPENROUTER_API_KEY')
        if not api_key:
            raise ValueError("No OPENROUTER_API_KEY found in environment")
    
    def run_execution_benchmark(self) -> Dict[str, Any]:
        """Run the execution benchmark with reliable verification."""
        print("\n⚡ Running Execution Benchmark with LLM Verification")
        print("=" * 60)
        
        # Create benchmark
        benchmark = ExecutionBenchmark(
            python_executable=self.python_executable,
            enable_storage=self.enable_storage,
            run_id=f"{self.run_id}_execution" if self.run_id else None
        )
        
        # Override the verification method to use our verification agent
        original_verify_test = benchmark.verify_test
        
        def enhanced_verify_test(expected_files, expected_execution=None, task_description="", agent_result=None):
            """Enhanced verification using LLM-powered verification agent."""
            if task_description and agent_result:
                # Initialize verification agent for this workspace
                verification_agent = VerificationAgent(
                    workspace_dir=str(benchmark.workspace_dir),
                    config=VerificationConfig()
                )
                
                # Use LLM verification
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
                
                # Print verification summary
                summary = verification_agent.get_verification_summary(verification_result)
                print(f"\n🔍 Verification Summary:")
                print(summary)
                
                return verification_result
            
            # Fallback to original verification
            return original_verify_test(expected_files, expected_execution, task_description, agent_result)
        
        # Replace the verification method
        benchmark.verify_test = enhanced_verify_test
        
        # Run the benchmark
        try:
            result = benchmark.run_benchmark()
            self.results["execution"] = result
            return result
        except Exception as e:
            error_result = {
                "success": False,
                "message": f"Execution benchmark failed: {str(e)}",
                "error": str(e)
            }
            self.results["execution"] = error_result
            return error_result
    
    def run_custom_tasks(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run custom tasks with reliable verification."""
        print("\n🧪 Running Custom Tasks with LLM Verification")
        print("=" * 60)
        
        results = []
        
        for i, task in enumerate(tasks, 1):
            print(f"\n{'='*60}")
            print(f"Task {i}: {task['name']}")
            print(f"Description: {task['description']}")
            print(f"{'='*60}")
            
            # Create temporary workspace
            import tempfile
            import shutil
            
            workspace_dir = tempfile.mkdtemp(prefix=f"reliable_task_{i}_")
            
            try:
                # Create verification agent
                verification_agent = VerificationAgent(
                    workspace_dir=workspace_dir,
                    config=VerificationConfig()
                )
                
                # Create LLM agent
                from agent.llm_agent import LLMCodingAgent, ExecutionConfig
                agent = LLMCodingAgent(
                    workspace_dir=workspace_dir,
                    execution_config=ExecutionConfig(
                        python_executable=self.python_executable,
                        enable_execution=True
                    )
                )
                
                # Execute task
                print(f"🤖 Executing task...")
                agent_result = agent.execute_task(task['description'], max_steps=10)
                
                # Verify task completion
                print(f"🔍 Verifying task completion...")
                verification_result = verification_agent.verify_task_completion(
                    task['description'], 
                    agent_result
                )
                
                # Create result
                task_result = {
                    "task_name": task['name'],
                    "task_description": task['description'],
                    "agent_result": agent_result,
                    "verification": verification_result,
                    "success": agent_result.get("success", False) and verification_result.get("success", False)
                }
                
                results.append(task_result)
                
                # Print summary
                status = "✓ PASS" if task_result["success"] else "✗ FAIL"
                print(f"\n{status}: {task['name']}")
                if not task_result["success"]:
                    print(f"  Agent result: {agent_result.get('message', 'No message')}")
                    print(f"  Verification: {verification_result.get('message', 'No message')}")
                
            finally:
                # Cleanup
                shutil.rmtree(workspace_dir)
        
        # Calculate summary
        total_tasks = len(results)
        passed_tasks = sum(1 for r in results if r["success"])
        failed_tasks = total_tasks - passed_tasks
        
        summary_result = {
            "success": failed_tasks == 0,
            "total_tasks": total_tasks,
            "passed_tasks": passed_tasks,
            "failed_tasks": failed_tasks,
            "success_rate": passed_tasks / total_tasks if total_tasks > 0 else 0,
            "task_results": results
        }
        
        self.results["custom_tasks"] = summary_result
        return summary_result
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks with reliable verification."""
        print("🚀 Starting Reliable Benchmark Suite")
        print("=" * 60)
        
        # Run execution benchmark
        execution_result = self.run_execution_benchmark()
        
        # Run custom tasks
        custom_tasks = [
            {
                "name": "Simple Python File",
                "description": "Create a file called hello.py with a function that prints 'Hello, World!' and returns 'Success!', then run it to verify it works"
            },
            {
                "name": "Calculator Class",
                "description": "Create a file called calculator.py with a Calculator class that has add, subtract, multiply, and divide methods. Include a test function that creates an instance and tests all methods, then run it to verify it works"
            },
            {
                "name": "Multi-file Project",
                "description": "Create a simple project with main.py, utils.py, and test_main.py. main.py should import from utils.py and have a main function. utils.py should have some utility functions. test_main.py should test the main function. Then run test_main.py to verify everything works"
            },
            {
                "name": "Error Handling",
                "description": "Create a file called error_demo.py with a function that demonstrates proper error handling using try-except blocks. The function should attempt to divide by zero, catch the exception, and print a meaningful error message. Then run it to verify the error handling works"
            }
        ]
        
        custom_result = self.run_custom_tasks(custom_tasks)
        
        # Print overall summary
        print("\n" + "=" * 60)
        print("📊 RELIABLE BENCHMARK SUMMARY")
        print("=" * 60)
        
        print(f"Execution Benchmark: {execution_result.get('success_rate', 0):.1f}% success rate")
        print(f"Custom Tasks: {custom_result.get('success_rate', 0):.1f}% success rate")
        
        overall_success = execution_result.get("success", False) and custom_result.get("success", False)
        print(f"\nOverall: {'✅ PASS' if overall_success else '❌ FAIL'}")
        
        return {
            "success": overall_success,
            "execution_benchmark": execution_result,
            "custom_tasks": custom_result
        }


def main():
    parser = argparse.ArgumentParser(description="Run Reliable Benchmarks with LLM Verification")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["execution", "custom", "all"],
        default="all",
        help="Which benchmark to run"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Custom run ID for storage"
    )
    parser.add_argument(
        "--no-storage",
        action="store_true",
        help="Disable automatic storage of results"
    )
    parser.add_argument(
        "--python-executable",
        type=str,
        help="Path to Python executable to use for execution"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reliable_benchmark_results.json",
        help="Output file for results"
    )
    
    args = parser.parse_args()
    
    # Check API key
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key:
        print("❌ Error: No OPENROUTER_API_KEY found in environment")
        print("   Set the API key in your .env file or environment variables")
        sys.exit(1)
    
    print("🔑 API key loaded successfully")
    
    # Create runner
    runner = ReliableBenchmarkRunner(
        python_executable=args.python_executable,
        enable_storage=not args.no_storage,
        run_id=args.run_id
    )
    
    # Run benchmarks
    if args.benchmark == "execution":
        result = runner.run_execution_benchmark()
    elif args.benchmark == "custom":
        custom_tasks = [
            {
                "name": "Simple Python File",
                "description": "Create a file called hello.py with a function that prints 'Hello, World!' and returns 'Success!', then run it to verify it works"
            },
            {
                "name": "Calculator Class", 
                "description": "Create a file called calculator.py with a Calculator class that has add, subtract, multiply, and divide methods. Include a test function that creates an instance and tests all methods, then run it to verify it works"
            }
        ]
        result = runner.run_custom_tasks(custom_tasks)
    else:  # all
        result = runner.run_all_benchmarks()
    
    # Save results
    if not args.no_storage:
        with open(args.output, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\n💾 Results saved to: {args.output}")
    
    if result.get("success"):
        print(f"\n🎉 All benchmarks passed!")
    else:
        print(f"\n❌ Some benchmarks failed.")
    
    return result


if __name__ == "__main__":
    main() 