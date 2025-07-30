"""
LLM Agent Evaluator

This module runs evaluation tasks and collects detailed traces for
prompt optimization analysis.
"""

import json
import time
import tempfile
import shutil
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import our components
import sys
sys.path.append(str(Path(__file__).parent.parent))
from agent.llm_agent import LLMCodingAgent, LLMConfig
from evaluation.task_suite import TaskSuite, TaskDefinition


@dataclass
class ExecutionStep:
    """Record of a single execution step."""
    step_number: int
    prompt: str
    llm_response: str
    parsed_tool_call: Dict[str, Any]
    tool_execution_result: Dict[str, Any]
    timestamp: float
    duration: float


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    task_name: str
    success: bool
    steps_taken: int
    max_steps: int
    execution_steps: List[ExecutionStep]
    final_files: List[str]
    expected_files: List[str]
    missing_files: List[str]
    extra_files: List[str]
    content_analysis: Dict[str, Any]
    error_message: Optional[str] = None
    total_duration: float = 0.0


class LLMAgentEvaluator:
    """Evaluates LLM agent performance on a suite of tasks."""
    
    def __init__(self, workspace_base: str = None):
        self.workspace_base = workspace_base or tempfile.mkdtemp(prefix="evaluation_")
        self.task_suite = TaskSuite()
        self.results = []
        
    def run_single_task(self, task_id: str) -> TaskResult:
        """Run a single task and collect detailed traces."""
        task = self.task_suite.get_task(task_id)
        
        # Create isolated workspace for this task
        workspace_dir = Path(self.workspace_base) / f"task_{task_id}"
        workspace_dir.mkdir(exist_ok=True)
        
        print(f"\n🧪 Running Task: {task.name}")
        print(f"📝 Description: {task.description}")
        print(f"📁 Workspace: {workspace_dir}")
        
        # Initialize agent
        agent = LLMCodingAgent(workspace_dir=str(workspace_dir))
        
        # Collect execution steps
        execution_steps = []
        start_time = time.time()
        
        # Override the execute_task method to capture traces
        original_execute_task = agent.execute_task
        
        def traced_execute_task(task_description: str, max_steps: int = 10):
            """Execute task with detailed tracing."""
            agent.context["task_description"] = task_description
            agent.update_context()
            
            step_count = 0
            task_success = False
            
            try:
                while step_count < max_steps:
                    step_start = time.time()
                    
                    # Generate prompt
                    system_prompt = agent.prompt_engine.generate_system_prompt(list(agent.tools.keys()))
                    task_prompt = agent.prompt_engine.generate_task_prompt(task_description, agent.context)
                    
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": task_prompt}
                    ]
                    
                    # Get LLM response
                    llm_response = agent.llm_client.generate_response(messages)
                    
                    # Parse response
                    parsed = agent.response_parser.parse_response(llm_response, list(agent.tools.keys()))
                    
                    # Execute tool if needed
                    tool_result = None
                    if parsed.get("tool"):
                        tool_name = parsed["tool"]
                        parameters = parsed.get("parameters", {})
                        tool_result = agent.execute_tool(tool_name, **parameters)
                    else:
                        task_success = True
                        break
                    
                    step_duration = time.time() - step_start
                    
                    # Record step
                    step = ExecutionStep(
                        step_number=step_count + 1,
                        prompt=task_prompt,
                        llm_response=llm_response,
                        parsed_tool_call=parsed,
                        tool_execution_result=tool_result,
                        timestamp=time.time(),
                        duration=step_duration
                    )
                    execution_steps.append(step)
                    
                    step_count += 1
                    
                    if tool_result and not tool_result.get("success", False):
                        break
                        
            except Exception as e:
                print(f"❌ Error during task execution: {e}")
                return {
                    "success": False,
                    "message": f"Task execution failed: {str(e)}",
                    "steps_taken": step_count
                }
            
            return {
                "success": task_success,
                "message": f"Task completed with {'success' if task_success else 'failure'}",
                "steps_taken": step_count
            }
        
        # Replace the execute_task method temporarily
        agent.execute_task = traced_execute_task
        
        # Execute the task
        result = agent.execute_task(task.description, max_steps=task.max_steps)
        
        # Analyze results - check the actual workspace directory
        final_files = self._get_files_in_workspace(workspace_dir)
        missing_files = [f for f in task.expected_files if f not in final_files]
        extra_files = [f for f in final_files if f not in task.expected_files]
        
        # Also check if files were created in the current directory (fallback)
        if missing_files:
            current_dir_files = self._get_files_in_workspace(Path("."))
            for missing_file in missing_files[:]:  # Create a copy to iterate
                if missing_file in current_dir_files:
                    # File was created in current directory instead of workspace
                    missing_files.remove(missing_file)
                    print(f"⚠️  Note: {missing_file} was created in current directory instead of workspace")
        
        # Analyze content if patterns are specified
        content_analysis = {}
        if task.expected_content_patterns:
            content_analysis = self._analyze_content(workspace_dir, task)
        
        total_duration = time.time() - start_time
        
        task_result = TaskResult(
            task_id=task.id,
            task_name=task.name,
            success=result["success"] and len(missing_files) == 0,
            steps_taken=result["steps_taken"],
            max_steps=task.max_steps,
            execution_steps=execution_steps,
            final_files=final_files,
            expected_files=task.expected_files,
            missing_files=missing_files,
            extra_files=extra_files,
            content_analysis=content_analysis,
            error_message=result.get("message") if not result["success"] else None,
            total_duration=total_duration
        )
        
        # Cleanup
        shutil.rmtree(workspace_dir, ignore_errors=True)
        
        return task_result
    
    def run_all_tasks(self) -> List[TaskResult]:
        """Run all tasks in the suite."""
        print("🚀 Starting LLM Agent Evaluation")
        print("=" * 50)
        
        all_tasks = self.task_suite.get_all_tasks()
        results = []
        
        for i, task in enumerate(all_tasks, 1):
            print(f"\n📋 Task {i}/{len(all_tasks)}: {task.name}")
            try:
                result = self.run_single_task(task.id)
                results.append(result)
                
                # Print summary
                status = "✅ PASS" if result.success else "❌ FAIL"
                print(f"{status} - {result.steps_taken}/{result.max_steps} steps")
                if result.missing_files:
                    print(f"   Missing: {result.missing_files}")
                if result.extra_files:
                    print(f"   Extra: {result.extra_files}")
                    
            except Exception as e:
                print(f"❌ Error running task {task.id}: {e}")
                # Create error result
                error_result = TaskResult(
                    task_id=task.id,
                    task_name=task.name,
                    success=False,
                    steps_taken=0,
                    max_steps=task.max_steps,
                    execution_steps=[],
                    final_files=[],
                    expected_files=task.expected_files,
                    missing_files=task.expected_files,
                    extra_files=[],
                    content_analysis={},
                    error_message=str(e),
                    total_duration=0.0
                )
                results.append(error_result)
        
        self.results = results
        return results
    
    def _get_files_in_workspace(self, workspace_dir: Path) -> List[str]:
        """Get list of files in workspace."""
        files = []
        if workspace_dir.exists():
            for item in workspace_dir.iterdir():
                if item.is_file():
                    files.append(item.name)
        return files
    
    def _analyze_content(self, workspace_dir: Path, task: TaskDefinition) -> Dict[str, Any]:
        """Analyze content of created files against expected patterns."""
        analysis = {}
        
        for pattern in task.expected_content_patterns:
            found_in_files = []
            for file_path in workspace_dir.iterdir():
                if file_path.is_file():
                    try:
                        content = file_path.read_text(encoding='utf-8')
                        if pattern.lower() in content.lower():
                            found_in_files.append(file_path.name)
                    except:
                        pass
            
            analysis[pattern] = {
                "found": len(found_in_files) > 0,
                "files": found_in_files
            }
        
        return analysis
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report."""
        if not self.results:
            return {"error": "No results available"}
        
        total_tasks = len(self.results)
        successful_tasks = sum(1 for r in self.results if r.success)
        success_rate = (successful_tasks / total_tasks) * 100 if total_tasks > 0 else 0
        
        # Analyze by difficulty
        difficulty_stats = {}
        for difficulty in ["easy", "medium", "hard"]:
            tasks = self.task_suite.get_tasks_by_difficulty(difficulty)
            task_ids = [t.id for t in tasks]
            difficulty_results = [r for r in self.results if r.task_id in task_ids]
            if difficulty_results:
                difficulty_stats[difficulty] = {
                    "total": len(difficulty_results),
                    "successful": sum(1 for r in difficulty_results if r.success),
                    "success_rate": (sum(1 for r in difficulty_results if r.success) / len(difficulty_results)) * 100
                }
        
        # Analyze common failure patterns
        failure_analysis = self._analyze_failures()
        
        report = {
            "summary": {
                "total_tasks": total_tasks,
                "successful_tasks": successful_tasks,
                "success_rate": success_rate,
                "average_steps": sum(r.steps_taken for r in self.results) / total_tasks if total_tasks > 0 else 0,
                "average_duration": sum(r.total_duration for r in self.results) / total_tasks if total_tasks > 0 else 0
            },
            "difficulty_breakdown": difficulty_stats,
            "failure_analysis": failure_analysis,
            "detailed_results": [asdict(r) for r in self.results]
        }
        
        return report
    
    def _analyze_failures(self) -> Dict[str, Any]:
        """Analyze common failure patterns."""
        failed_results = [r for r in self.results if not r.success]
        
        failure_patterns = {
            "missing_files": {},
            "parsing_errors": 0,
            "tool_execution_errors": 0,
            "api_errors": 0
        }
        
        for result in failed_results:
            # Analyze missing files
            for missing_file in result.missing_files:
                failure_patterns["missing_files"][missing_file] = failure_patterns["missing_files"].get(missing_file, 0) + 1
            
            # Analyze execution steps for errors
            for step in result.execution_steps:
                if step.parsed_tool_call.get("error"):
                    failure_patterns["parsing_errors"] += 1
                if step.tool_execution_result and not step.tool_execution_result.get("success"):
                    failure_patterns["tool_execution_errors"] += 1
                if "429" in str(step.llm_response) or "401" in str(step.llm_response):
                    failure_patterns["api_errors"] += 1
        
        return failure_patterns
    
    def save_results(self, output_file: str):
        """Save evaluation results to a JSON file."""
        report = self.generate_report()
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {output_file}")


if __name__ == "__main__":
    # Test the evaluator
    evaluator = LLMAgentEvaluator()
    results = evaluator.run_all_tasks()
    evaluator.save_results("evaluation_results.json") 