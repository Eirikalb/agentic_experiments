"""
Storage system for benchmark runs and results.

This module provides structured storage for benchmark results,
integrating with the existing evaluation storage system.
"""

import json
import shutil
import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

# Import the evaluation storage system
import sys
sys.path.append(str(Path(__file__).parent.parent))
from evaluation.storage import EvaluationStorage, RunMetadata


@dataclass
class BenchmarkMetadata:
    """Metadata for a benchmark run."""
    run_id: str
    timestamp: str
    benchmark_name: str
    total_tests: int
    passed_tests: int
    success_rate: float
    workspace: str
    notes: Optional[str] = None


class TestLogger:
    """Logger for capturing complete test execution details."""
    
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.logs = []
        self.start_time = None
        self.end_time = None
        
    def start_test(self):
        """Start logging a test."""
        self.start_time = datetime.now()
        self.logs.append({
            "timestamp": self.start_time.isoformat(),
            "type": "test_start",
            "test_name": self.test_name,
            "message": f"Starting test: {self.test_name}"
        })
    
    def end_test(self):
        """End logging a test."""
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds() if self.start_time else 0
        self.logs.append({
            "timestamp": self.end_time.isoformat(),
            "type": "test_end",
            "test_name": self.test_name,
            "duration": duration,
            "message": f"Completed test: {self.test_name} (duration: {duration:.2f}s)"
        })
    
    def log_agent_step(self, step_number: int, prompt: str, response: str, tool_call: Dict[str, Any], tool_result: Dict[str, Any]):
        """Log an agent execution step."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "agent_step",
            "step_number": step_number,
            "prompt": prompt,
            "response": response,
            "tool_call": tool_call,
            "tool_result": tool_result
        })
    
    def log_verification(self, verification_result: Dict[str, Any]):
        """Log verification results."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "verification",
            "verification_result": verification_result
        })
    
    def log_error(self, error: str, error_type: str = "error"):
        """Log an error."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": error_type,
            "error": error
        })
    
    def log_info(self, message: str):
        """Log an info message."""
        self.logs.append({
            "timestamp": datetime.now().isoformat(),
            "type": "info",
            "message": message
        })
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get all logs for this test."""
        return self.logs
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the test execution."""
        if not self.start_time or not self.end_time:
            return {}
        
        duration = (self.end_time - self.start_time).total_seconds()
        agent_steps = [log for log in self.logs if log["type"] == "agent_step"]
        errors = [log for log in self.logs if log["type"] in ["error", "exception"]]
        
        return {
            "test_name": self.test_name,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration": duration,
            "total_logs": len(self.logs),
            "agent_steps": len(agent_steps),
            "errors": len(errors),
            "success": len(errors) == 0
        }


class BenchmarkStorage:
    """Manages storage of benchmark runs and results."""
    
    def __init__(self, base_dir: str = "runs"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        
        # Reuse the evaluation storage system
        self.evaluation_storage = EvaluationStorage(base_dir)
    
    def create_benchmark_run_directory(self, benchmark_name: str, run_id: Optional[str] = None) -> Path:
        """Create a new benchmark run directory."""
        if run_id is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_id = f"benchmark_{benchmark_name}_{timestamp}"
        
        run_dir = self.base_dir / run_id
        run_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (run_dir / "tests").mkdir(exist_ok=True)
        (run_dir / "logs").mkdir(exist_ok=True)
        (run_dir / "reports").mkdir(exist_ok=True)
        (run_dir / "workspace").mkdir(exist_ok=True)
        
        return run_dir
    
    def store_test_result(self, 
                         run_dir: Path, 
                         test_result: Dict[str, Any],
                         workspace_dir: Optional[Path] = None,
                         test_logger: Optional[TestLogger] = None) -> Path:
        """Store a single test result with all associated files and logs."""
        test_name = test_result.get("test_name", "unknown_test")
        test_dir = run_dir / "tests" / test_name.replace(" ", "_").lower()
        test_dir.mkdir(exist_ok=True)
        
        # Store test metadata
        test_metadata = {
            "test_name": test_result.get("test_name"),
            "task_description": test_result.get("task_description"),
            "success": test_result.get("success", False),
            "agent_result": test_result.get("agent_result", {}),
            "verification": test_result.get("verification", {}),
            "timestamp": datetime.now().isoformat()
        }
        
        with open(test_dir / "metadata.json", "w") as f:
            json.dump(test_metadata, f, indent=2, default=str)
        
        # Store detailed test result
        with open(test_dir / "test_result.json", "w") as f:
            json.dump(test_result, f, indent=2, default=str)
        
        # Store complete logs if available
        if test_logger:
            logs = test_logger.get_logs()
            with open(test_dir / "complete_logs.json", "w") as f:
                json.dump(logs, f, indent=2, default=str)
            
            # Store log summary
            log_summary = test_logger.get_summary()
            with open(test_dir / "log_summary.json", "w") as f:
                json.dump(log_summary, f, indent=2, default=str)
            
            # Create human-readable log file
            self._create_readable_log_file(test_dir, logs, log_summary)
        
        # Copy workspace files if available
        if workspace_dir and workspace_dir.exists():
            files_dir = test_dir / "workspace"
            files_dir.mkdir(exist_ok=True)
            
            for file_path in workspace_dir.iterdir():
                if file_path.is_file():
                    shutil.copy2(file_path, files_dir / file_path.name)
        
        # Create a summary file
        summary_lines = [
            f"Test: {test_result.get('test_name', 'Unknown')}",
            f"Status: {'SUCCESS' if test_result.get('success', False) else 'FAILURE'}",
            f"Task: {test_result.get('task_description', 'No description')}",
            "",
            "Agent Result:",
            "-" * 20
        ]
        
        agent_result = test_result.get("agent_result", {})
        summary_lines.extend([
            f"Success: {agent_result.get('success', False)}",
            f"Message: {agent_result.get('message', 'No message')}",
            f"Steps: {agent_result.get('steps_taken', 0)}/{agent_result.get('max_steps', 0)}",
            f"Duration: {agent_result.get('duration', 0):.2f}s"
        ])
        
        verification = test_result.get("verification", {})
        summary_lines.extend([
            "",
            "Verification:",
            "-" * 20,
            f"Success: {verification.get('success', False)}",
            f"Message: {verification.get('message', 'No message')}"
        ])
        
        if verification.get("existing_files"):
            summary_lines.append(f"Existing files: {verification['existing_files']}")
        if verification.get("missing_files"):
            summary_lines.append(f"Missing files: {verification['missing_files']}")
        
        # Add log summary if available
        if test_logger:
            log_summary = test_logger.get_summary()
            summary_lines.extend([
                "",
                "Log Summary:",
                "-" * 20,
                f"Total Logs: {log_summary.get('total_logs', 0)}",
                f"Agent Steps: {log_summary.get('agent_steps', 0)}",
                f"Errors: {log_summary.get('errors', 0)}",
                f"Duration: {log_summary.get('duration', 0):.2f}s"
            ])
        
        with open(test_dir / "summary.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(summary_lines))
        
        return test_dir
    
    def _create_readable_log_file(self, test_dir: Path, logs: List[Dict[str, Any]], summary: Dict[str, Any]):
        """Create a human-readable log file."""
        log_lines = [
            f"Complete Test Log: {summary.get('test_name', 'Unknown')}",
            "=" * 60,
            f"Start Time: {summary.get('start_time', 'Unknown')}",
            f"End Time: {summary.get('end_time', 'Unknown')}",
            f"Duration: {summary.get('duration', 0):.2f}s",
            f"Total Logs: {summary.get('total_logs', 0)}",
            f"Agent Steps: {summary.get('agent_steps', 0)}",
            f"Errors: {summary.get('errors', 0)}",
            "",
            "Detailed Log:",
            "=" * 60
        ]
        
        for i, log_entry in enumerate(logs, 1):
            timestamp = log_entry.get("timestamp", "Unknown")
            log_type = log_entry.get("type", "unknown")
            
            log_lines.append(f"\n[{i}] {timestamp} - {log_type.upper()}")
            log_lines.append("-" * 40)
            
            if log_type == "agent_step":
                step_num = log_entry.get("step_number", "?")
                tool_call = log_entry.get("tool_call", {})
                tool_name = tool_call.get("tool", "unknown")
                tool_success = log_entry.get("tool_result", {}).get("success", False)
                
                log_lines.extend([
                    f"Step {step_num}: {tool_name}",
                    f"Success: {tool_success}",
                    f"Tool Call: {json.dumps(tool_call, indent=2)}",
                    f"Tool Result: {json.dumps(log_entry.get('tool_result', {}), indent=2)}"
                ])
                
                # Include prompt and response (truncated for readability)
                prompt = log_entry.get("prompt", "")
                response = log_entry.get("response", "")
                
                if prompt:
                    log_lines.append(f"Prompt (first 200 chars): {prompt[:200]}...")
                if response:
                    log_lines.append(f"Response (first 200 chars): {response[:200]}...")
                    
            elif log_type == "verification":
                verification = log_entry.get("verification_result", {})
                log_lines.append(f"Verification: {json.dumps(verification, indent=2)}")
                
            elif log_type in ["error", "exception"]:
                error = log_entry.get("error", "Unknown error")
                log_lines.append(f"ERROR: {error}")
                
            elif log_type == "info":
                message = log_entry.get("message", "")
                log_lines.append(f"INFO: {message}")
                
            else:
                # Generic log entry
                for key, value in log_entry.items():
                    if key not in ["timestamp", "type"]:
                        if isinstance(value, dict):
                            log_lines.append(f"{key}: {json.dumps(value, indent=2)}")
                        else:
                            log_lines.append(f"{key}: {value}")
        
        with open(test_dir / "detailed_log.txt", "w", encoding='utf-8') as f:
            f.write("\n".join(log_lines))
    
    def store_benchmark_metadata(self, run_dir: Path, metadata: BenchmarkMetadata) -> None:
        """Store benchmark metadata."""
        with open(run_dir / "metadata.json", "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
    
    def store_benchmark_report(self, run_dir: Path, report: Dict[str, Any]) -> None:
        """Store the full benchmark report."""
        with open(run_dir / "reports" / "benchmark_report.json", "w") as f:
            json.dump(report, f, indent=2, default=str)
        
        # Create human-readable summary
        summary_lines = [
            "Benchmark Run Summary",
            "=" * 50,
            f"Run ID: {report.get('run_id', 'unknown')}",
            f"Benchmark: {report.get('benchmark_name', 'unknown')}",
            f"Total Tests: {report.get('total_tests', 0)}",
            f"Passed: {report.get('passed_tests', 0)}",
            f"Failed: {report.get('failed_tests', 0)}",
            f"Success Rate: {report.get('success_rate', 0):.1f}%",
            f"Workspace: {report.get('workspace', 'unknown')}",
            "",
            "Test Results:",
            "-" * 20
        ]
        
        for result in report.get("test_results", []):
            status = "PASS" if result.get("success", False) else "FAIL"
            summary_lines.append(f"{result.get('test_name', 'unknown')}: {status}")
        
        with open(run_dir / "reports" / "summary.txt", "w") as f:
            f.write("\n".join(summary_lines))
    
    def list_benchmark_runs(self) -> List[Dict[str, Any]]:
        """List all benchmark runs."""
        runs = []
        for run_dir in self.base_dir.iterdir():
            if run_dir.is_dir() and run_dir.name.startswith("benchmark_"):
                metadata_file = run_dir / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, "r") as f:
                        metadata = json.load(f)
                    runs.append({
                        "run_id": metadata.get("run_id", run_dir.name),
                        "benchmark_name": metadata.get("benchmark_name", "unknown"),
                        "timestamp": metadata.get("timestamp", ""),
                        "success_rate": metadata.get("success_rate", 0),
                        "total_tests": metadata.get("total_tests", 0),
                        "path": str(run_dir)
                    })
        
        return sorted(runs, key=lambda x: x["timestamp"], reverse=True)
    
    def get_benchmark_run_details(self, run_id: str) -> Dict[str, Any]:
        """Get detailed information about a specific benchmark run."""
        run_dir = self.base_dir / run_id
        if not run_dir.exists():
            raise ValueError(f"Benchmark run {run_id} not found")
        
        # Load metadata
        metadata_file = run_dir / "metadata.json"
        metadata = {}
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                metadata = json.load(f)
        
        # Load report
        report_file = run_dir / "reports" / "benchmark_report.json"
        report = {}
        if report_file.exists():
            with open(report_file, "r") as f:
                report = json.load(f)
        
        # List tests
        tests = []
        tests_dir = run_dir / "tests"
        if tests_dir.exists():
            for test_dir in tests_dir.iterdir():
                if test_dir.is_dir():
                    test_metadata_file = test_dir / "metadata.json"
                    if test_metadata_file.exists():
                        with open(test_metadata_file, "r") as f:
                            test_metadata = json.load(f)
                        
                        # Check for log summary
                        log_summary_file = test_dir / "log_summary.json"
                        if log_summary_file.exists():
                            with open(log_summary_file, "r") as f:
                                log_summary = json.load(f)
                            test_metadata["log_summary"] = log_summary
                        
                        tests.append(test_metadata)
        
        return {
            "metadata": metadata,
            "report": report,
            "tests": tests,
            "path": str(run_dir)
        }
    
    def compare_benchmark_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """Compare multiple benchmark runs."""
        runs_data = {}
        for run_id in run_ids:
            try:
                runs_data[run_id] = self.get_benchmark_run_details(run_id)
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


class EnhancedBenchmark:
    """Enhanced benchmark with storage capabilities."""
    
    def __init__(self, benchmark_name: str, storage: BenchmarkStorage = None):
        self.benchmark_name = benchmark_name
        self.storage = storage or BenchmarkStorage()
        self.current_run_dir = None
        self.current_run_id = None
        self.test_results = []
        self.test_loggers = {}  # Store loggers for each test
    
    def start_run(self, run_id: Optional[str] = None) -> str:
        """Start a new benchmark run."""
        self.current_run_dir = self.storage.create_benchmark_run_directory(
            self.benchmark_name, run_id
        )
        self.current_run_id = self.current_run_dir.name
        self.test_results = []
        self.test_loggers = {}
        
        print(f"🚀 Started benchmark run: {self.current_run_id}")
        print(f"📁 Run directory: {self.current_run_dir}")
        
        return self.current_run_id
    
    def get_test_logger(self, test_name: str) -> TestLogger:
        """Get or create a logger for a specific test."""
        if test_name not in self.test_loggers:
            self.test_loggers[test_name] = TestLogger(test_name)
        return self.test_loggers[test_name]
    
    def store_test_result(self, test_result: Dict[str, Any], workspace_dir: Optional[Path] = None) -> Path:
        """Store a test result with all associated files and logs."""
        if not self.current_run_dir:
            raise RuntimeError("No active run. Call start_run() first.")
        
        test_name = test_result.get("test_name", "unknown_test")
        test_logger = self.test_loggers.get(test_name)
        
        return self.storage.store_test_result(
            self.current_run_dir, 
            test_result, 
            workspace_dir,
            test_logger
        )
    
    def end_run(self, metadata: BenchmarkMetadata, report: Dict[str, Any]) -> None:
        """End the current run and store final data."""
        if not self.current_run_dir:
            raise RuntimeError("No active run to end.")
        
        # Update metadata with run_id
        metadata.run_id = self.current_run_id
        
        # Store metadata and report
        self.storage.store_benchmark_metadata(self.current_run_dir, metadata)
        self.storage.store_benchmark_report(self.current_run_dir, report)
        
        print(f"✅ Completed benchmark run: {self.current_run_id}")
        print(f"📊 Success rate: {metadata.success_rate:.1f}%")
        
        # Reset
        self.current_run_dir = None
        self.current_run_id = None
        self.test_loggers = {}


if __name__ == "__main__":
    # Test the benchmark storage system
    storage = BenchmarkStorage()
    runs = storage.list_benchmark_runs()
    
    print("📁 Available Benchmark Runs:")
    print("=" * 50)
    for run in runs:
        print(f"  {run['run_id']}: {run['success_rate']:.1f}% success rate")
        print(f"    Benchmark: {run['benchmark_name']}, Tests: {run['total_tests']}")
        print(f"    Timestamp: {run['timestamp']}")
        print() 