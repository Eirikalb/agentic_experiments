"""
Benchmark 2: LLM Agent Testing

This benchmark tests the LLM-powered agent's ability to execute tasks
using LLM orchestration. It requires an OpenRouter API key to run.
"""

import os
import tempfile
import shutil
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # If dotenv is not available, continue without it
    pass

# Import our agents
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agent.coding_agent import CodingAgent
from agent.llm_agent import LLMCodingAgent, LLMConfig

# Import storage system
from benchmarks.benchmark_storage import EnhancedBenchmark, BenchmarkMetadata


class LLMAgentBenchmark:
    """
    Benchmark for testing LLM agent capabilities with automatic storage and logging.
    """
    
    def __init__(self, api_key: str = None, enable_storage: bool = True, run_id: str = None):
        self.test_results = []
        self.workspace_dir = None
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        self.enable_storage = enable_storage
        
        if not self.api_key:
            print("⚠️  Warning: No OpenRouter API key found.")
            print("   Set OPENROUTER_API_KEY environment variable or create a .env file")
            print("   Get your API key from: https://openrouter.ai")
        else:
            print(f"✅ API key loaded successfully")
        
        self.llm_config = LLMConfig(
            api_key=self.api_key,
            model="moonshotai/kimi-k2"
        )
        
        if enable_storage:
            self.enhanced_benchmark = EnhancedBenchmark("llm_agent")
            self.run_id = self.enhanced_benchmark.start_run(run_id)
        else:
            self.enhanced_benchmark = None
            self.run_id = None
    
    def setup(self):
        """Set up the benchmark workspace."""
        self.workspace_dir = Path(tempfile.mkdtemp(prefix="benchmark_2_"))
        print(f"📁 Workspace: {self.workspace_dir}")
        print(f"🔑 API Key: {'✅ Available' if self.api_key else '❌ Not available'}")
    
    def cleanup(self):
        """Clean up the benchmark workspace."""
        if self.workspace_dir and self.workspace_dir.exists():
            shutil.rmtree(self.workspace_dir)
    
    def run_test(self, test_name: str, task_description: str, expected_files: list) -> bool:
        """Run a single test case with complete logging."""
        print(f"\n🧪 Running test: {test_name}")
        print(f"📝 Task: {task_description}")
        
        # Get logger for this test
        test_logger = None
        if self.enable_storage and self.enhanced_benchmark:
            test_logger = self.enhanced_benchmark.get_test_logger(test_name)
            test_logger.start_test()
            test_logger.log_info(f"Starting test: {test_name}")
            test_logger.log_info(f"Task: {task_description}")
            test_logger.log_info(f"Expected files: {expected_files}")
        
        try:
            # Create LLM agent
            agent = LLMCodingAgent(
                workspace_dir=str(self.workspace_dir),
                llm_config=self.llm_config
            )
            
            # Log agent creation
            if test_logger:
                test_logger.log_info(f"Created LLM agent with workspace: {self.workspace_dir}")
                test_logger.log_info(f"Model: {self.llm_config.model}")
                test_logger.log_info(f"API key available: {bool(self.api_key)}")
            
            # Execute task
            print(f"🤖 LLM Agent executing: {task_description}")
            print(f"📁 Working in: {self.workspace_dir}")
            
            result = agent.execute_task(task_description, max_steps=5)
            
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
            
            # Verify results
            success = self.verify_test(expected_files)
            
            # Log verification
            if test_logger:
                verification_result = {
                    "success": success,
                    "expected_files": expected_files,
                    "message": "All files found" if success else "Missing files"
                }
                test_logger.log_verification(verification_result)
            
            # Create test result
            test_result = {
                "test_name": test_name,
                "task_description": task_description,
                "agent_result": result,
                "verification": {
                    "success": success,
                    "expected_files": expected_files,
                    "message": "All files found" if success else "Missing files"
                },
                "success": result["success"] and success
            }
            
            self.test_results.append(test_result)
            
            # Store the test result if storage is enabled
            if self.enable_storage and self.enhanced_benchmark:
                self.enhanced_benchmark.store_test_result(test_result, self.workspace_dir)
            
            # End logging
            if test_logger:
                test_logger.end_test()
            
            if success:
                print(f"✅ {test_name}: PASSED")
            else:
                print(f"❌ {test_name}: FAILED")
                print(f"   Reason: Missing files: {expected_files}")
            
            return success
            
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
                self.enhanced_benchmark.store_test_result(error_result, self.workspace_dir)
            
            print(f"❌ {test_name}: FAILED")
            print(f"   Exception: {str(e)}")
            
            return False
    
    def verify_test(self, expected_files: list) -> bool:
        """Verify that expected files were created."""
        for filename in expected_files:
            file_path = self.workspace_dir / filename
            if not file_path.exists():
                return False
        return True
    
    def run_benchmark(self) -> Dict[str, Any]:
        """Run the complete LLM agent benchmark."""
        print("Starting LLM Agent Benchmark")
        print("=" * 50)
        
        try:
            self.setup()
            
            # Test cases
            test_cases = [
                {
                    "name": "LLM Create Python File",
                    "task": "Create a file called hello.py with a hello world function",
                    "expected_files": ["hello.py"]
                },
                {
                    "name": "LLM Create Multiple Files", 
                    "task": "Create a file called main.py and another called config.json",
                    "expected_files": ["main.py", "config.json"]
                },
                {
                    "name": "LLM Complex Task",
                    "task": "Create a Python file called calculator.py with a simple calculator class",
                    "expected_files": ["calculator.py"]
                }
            ]
            
            passed = 0
            total = len(test_cases)
            
            for test_case in test_cases:
                success = self.run_test(
                    test_case["name"],
                    test_case["task"], 
                    test_case["expected_files"]
                )
                if success:
                    passed += 1
            
            success_rate = (passed / total) * 100 if total > 0 else 0
            
            benchmark_result = {
                "benchmark_name": "llm_agent",
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": success_rate,
                "api_key_available": bool(self.api_key),
                "test_results": self.test_results,
                "run_id": self.run_id
            }
            
            # Store the benchmark results if storage is enabled
            if self.enable_storage and self.enhanced_benchmark:
                metadata = BenchmarkMetadata(
                    run_id=self.run_id,
                    timestamp=datetime.now().isoformat(),
                    benchmark_name="llm_agent",
                    total_tests=total,
                    passed_tests=passed,
                    success_rate=success_rate,
                    workspace=str(self.workspace_dir)
                )
                
                self.enhanced_benchmark.end_run(metadata, benchmark_result)
            
            print("\n" + "=" * 50)
            print("📊 LLM Benchmark Results:")
            print(f"   Total Tests: {total}")
            print(f"   Passed: {passed}")
            print(f"   Failed: {total - passed}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   API Key: {'✅ Available' if self.api_key else '❌ Not available'}")
            
            if self.run_id:
                print(f"   Run ID: {self.run_id}")
            
            if not self.api_key:
                print("\n💡 Note: Run with OPENROUTER_API_KEY environment variable for real LLM testing")
            
            return benchmark_result
            
        finally:
            self.cleanup()


def main():
    """Run the LLM agent benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run LLM Agent Benchmark")
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
    
    # Run benchmark
    benchmark = LLMAgentBenchmark(
        enable_storage=not args.no_storage,
        run_id=args.run_id
    )
    result = benchmark.run_benchmark()
    
    print(f"\nFinal result: {result}")


if __name__ == "__main__":
    main() 