"""
Storage system for evaluation runs and tasks.

This module provides structured storage for evaluation results,
including raw logs, prompts, and generated files for each task.
"""

import json
import shutil
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from .evaluator import TaskResult, ExecutionStep
from .task_suite import TaskDefinition


@dataclass
class RunMetadata:
    """Metadata for an evaluation run."""
    run_id: str
    timestamp: str
    model: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    average_steps: float
    average_duration: float
    failure_patterns: Dict[str, Any]
    notes: Optional[str] = None


class EvaluationStorage:
    """Manages storage of evaluation runs and tasks."""
    
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
    
    def create_run_directory(self, run_id: Optional[str] = None) -> Path:
        """Create a new run directory."""
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"run_{timestamp}"
        
        run_dir = self.base_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (run_dir / "tasks").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        
        return run_dir
    
    def store_task_result(self, 
                         run_dir: Path, 
                         task_result: TaskResult, 
                         workspace_dir: Path,
                         raw_logs: List[Dict[str, Any]]) -> Path:
        """Store a single task result with all associated files."""
        task_dir = run_dir / "tasks" / task_result.task_id
        task_dir.mkdir(exist_ok=True)
        
        # Store task metadata
        task_metadata = {
            "task_id": task_result.task_id,
            "task_name": task_result.task_name,
            "success": task_result.success,
            "steps_taken": task_result.steps_taken,
            "max_steps": task_result.max_steps,
            "total_duration": task_result.total_duration,
            "expected_files": task_result.expected_files,
            "missing_files": task_result.missing_files,
            "extra_files": task_result.extra_files,
            "content_analysis": task_result.content_analysis,
            "error_message": task_result.error_message,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(task_dir / "metadata.json", "w") as f:
            json.dump(task_metadata, f, indent=2, default=str)
        
        # Store execution steps
        steps_data = []
        for step in task_result.execution_steps:
            step_data = {
                "step_number": step.step_number,
                "prompt": step.prompt,
                "llm_response": step.llm_response,
                "parsed_tool_call": step.parsed_tool_call,
                "tool_execution_result": step.tool_execution_result,
                "timestamp": step.timestamp,
                "duration": step.duration
            }
            steps_data.append(step_data)
        
        with open(task_dir / "execution_steps.json", "w") as f:
            json.dump(steps_data, f, indent=2, default=str)
        
        # Store raw logs
        with open(task_dir / "raw_logs.json", "w") as f:
            json.dump(raw_logs, f, indent=2, default=str)
        
        # Copy generated files from workspace
        if workspace_dir.exists():
            files_dir = task_dir / "files"
            files_dir.mkdir(exist_ok=True)
            
            for file_path in workspace_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, files_dir / file_path.name)
        
        # Create a summary file
        summary_lines = [
            f"Task: {task_result.task_name}",
            f"Status: {'SUCCESS' if task_result.success else 'FAILURE'}",
            f"Steps: {task_result.steps_taken}/{task_result.max_steps}",
            f"Duration: {task_result.total_duration:.2f}s",
            f"Expected Files: {task_result.expected_files}",
            f"Missing Files: {task_result.missing_files}",
            f"Extra Files: {task_result.extra_files}",
            "",
            "Execution Summary:",
            "-" * 30
        ]
        
        for i, step in enumerate(task_result.execution_steps, 1):
            tool = step.parsed_tool_call.get("tool", "None")
            success = step.tool_execution_result.get("success", False) if step.tool_execution_result else False
            status = "PASS" if success else "FAIL"
            summary_lines.append(f"Step {i}: {status} {tool}")
        
        with open(task_dir / "summary.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        
        return task_dir
    
    def store_run_metadata(self, run_dir: Path, metadata: RunMetadata) -> None:
        """Store run metadata."""
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def store_run_report(self, run_dir: Path, report: Dict[str, Any]) -> None:
        """Store the full evaluation report."""
        with open(run_dir / "reports" / "evaluation_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_lines = [
            "Evaluation Run Summary",
            "=" * 50,
            f"Run ID: {report.get('summary', {}).get('run_id', 'unknown')}",
            f"Total Tasks: {report.get('summary', {}).get('total_tasks', 0)}",
            f"Successful: {report.get('summary', {}).get('successful_tasks', 0)}",
            f"Success Rate: {report.get('summary', {}).get('success_rate', 0):.1f}%",
            f"Average Steps: {report.get('summary', {}).get('average_steps', 0):.1f}",
            f"Average Duration: {report.get('summary', {}).get('average_duration', 0):.2f}s",
            "",
            "Task Results:",
            "-" * 20
        ]
        
        for result in report.get("detailed_results", []):
            status = "PASS" if result.get("success", False) else "FAIL"
            summary_lines.append(f"{result.get('task_id', 'unknown')}: {status}")
        
        with open(run_dir / "reports" / "summary.txt", "w") as f:
            f.write("\n".join(summary_lines))
    
    def list_runs(self) -> List[Dict[str, Any]]:
        """List all evaluation runs."""
        runs = []
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("run_"):
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    runs.append({
                        "run_id": metadata.get("run_id", run_dir.name),
                        "timestamp": metadata.get("timestamp", ""),
                        "success_rate": metadata.get("success_rate", 0),
                        "total_tasks": metadata.get("total_tasks", 0),
                        "path": str(run_dir)
                    })
        
        return sorted(runs, key=lambda x: x["timestamp"], reverse=True)
    
    def get_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Run {run_id} not found")
        
        # Load metadata
        metadata_file = run_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        # Load report
        report_file = run_dir / "reports" / "evaluation_report.json"
        report = {}
        if report_file.exists():
            with open(report_file, "r") as f:
                report = json.load(f)
        
        # List tasks
        tasks = []
        tasks_dir = run_dir / "tasks"
        if tasks_dir.exists():
            for task_dir in tasks_dir.iterdir():
                if task_dir.is_dir():
                    task_metadata_file = task_dir / "metadata.json"
                    if task_metadata_file.exists():
                        with open(task_metadata_file, "r") as f:
                            task_metadata = json.load(f)
                        tasks.append(task_metadata)
        
        return {
            "metadata": metadata,
            "report": report,
            "tasks": tasks,
            "path": str(run_dir)
        }
    
    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple runs."""
        runs_data = {}
        for run_id in run_ids:
            try:
                runs_data[run_id] = self.get_run_details(run_id)
            except ValueError:
                continue
        
        comparison = {
            "runs": runs_data,
            "summary": {
                "total_runs": len(runs_data),
                "best_success_rate": max(
                    (data["metadata"].get("success_rate", 0) for data in runs_data.values()),
                    default=0
                ),
                "average_success_rate": sum(
                    data["metadata"].get("success_rate", 0) for data in runs_data.values()
                ) / len(runs_data) if runs_data else 0
            }
        }
        
        return comparison


class EnhancedEvaluator:
    """Enhanced evaluator with storage capabilities."""
    
    def __init__(self, storage: EvaluationStorage):
        self.storage = storage
        self.current_run_dir = None
        self.current_run_id = None
    
    def start_run(self, run_id: Optional[str] = None) -> str:
        """Start a new evaluation run."""
        self.current_run_dir = self.storage.create_run_directory(run_id)
        self.current_run_id = self.current_run_dir.name
        
        print(f"🚀 Started evaluation run: {self.current_run_id}")
        print(f"📁 Run directory: {self.current_run_dir}")
        
        return self.current_run_id
    
    def store_task_with_logs(self, 
                            task_result: TaskResult, 
                            workspace_dir: Path,
                            raw_logs: List[Dict[str, Any]]) -> Path:
        """Store a task result with all associated logs and files."""
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")
        
        return self.storage.store_task_result(
            self.current_run_dir, 
            task_result, 
            workspace_dir, 
            raw_logs
        )
    
    def end_run(self, metadata: RunMetadata, report: Dict[str, Any]) -> None:
        """End the current run and store final data."""
        if not self.current_run_dir:
            raise RuntimeError("No active run to end.")
        
        # Update metadata with run_id
        metadata.run_id = self.current_run_id
        
        # Store metadata and report
        self.storage.store_run_metadata(self.current_run_dir, metadata)
        self.storage.store_run_report(self.current_run_dir, report)
        
        print(f"✅ Completed evaluation run: {self.current_run_id}")
        print(f"📊 Success rate: {metadata.success_rate:.1f}%")
        
        # Reset
        self.current_run_dir = None
        self.current_run_id = None


if __name__ == "__main__":
    # Test the storage system
    storage = EvaluationStorage()
    runs = storage.list_runs()
    
    print("📁 Available Evaluation Runs:")
    print("=" * 50)
    for run in runs:
        print(f"  {run['run_id']}: {run['success_rate']:.1f}% success rate")
        print(f"    Tasks: {run['total_tasks']}, Timestamp: {run['timestamp']}")
        print() 