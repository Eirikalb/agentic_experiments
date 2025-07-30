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


class LLMAgentBenchmark:
    """
    Benchmark for testing LLM agent capabilities.
    """
    
    def __init__(self, api_key: str = None):
        self.test_results = []
        self.workspace_dir = None
        self.api_key = api_key or os.getenv('OPENROUTER_API_KEY')
        
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
        """Run a single test case."""
        print(f"\n🧪 Running test: {test_name}")
        print(f"📝 Task: {task_description}")
        
        # Create LLM agent
        agent = LLMCodingAgent(
            workspace_dir=str(self.workspace_dir),
            llm_config=self.llm_config
        )
        
        # Execute task
        print(f"🤖 LLM Agent executing: {task_description}")
        print(f"📁 Working in: {self.workspace_dir}")
        
        result = agent.execute_task(task_description, max_steps=5)
        
        # Verify results
        success = self.verify_test(expected_files)
        
        if success:
            print(f"✅ {test_name}: PASSED")
        else:
            print(f"❌ {test_name}: FAILED")
            print(f"   Reason: Missing files: {expected_files}")
        
        return success
    
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
                self.test_results.append({
                    "name": test_case["name"],
                    "success": success
                })
            
            success_rate = (passed / total) * 100 if total > 0 else 0
            
            print("\n" + "=" * 50)
            print("📊 LLM Benchmark Results:")
            print(f"   Total Tests: {total}")
            print(f"   Passed: {passed}")
            print(f"   Failed: {total - passed}")
            print(f"   Success Rate: {success_rate:.1f}%")
            print(f"   API Key: {'✅ Available' if self.api_key else '❌ Not available'}")
            
            if not self.api_key:
                print("\n💡 Note: Run with OPENROUTER_API_KEY environment variable for real LLM testing")
            
            return {
                "benchmark_name": "llm_agent",
                "total_tests": total,
                "passed": passed,
                "failed": total - passed,
                "success_rate": success_rate,
                "api_key_available": bool(self.api_key),
                "test_results": self.test_results
            }
            
        finally:
            self.cleanup()


if __name__ == "__main__":
    benchmark = LLMAgentBenchmark()
    result = benchmark.run_benchmark()
    print(f"\nFinal result: {result}") 