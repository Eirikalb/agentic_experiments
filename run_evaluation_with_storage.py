#!/usr/bin/env python3
"""
Enhanced LLM Agent Evaluation Runner with Storage

This script runs the 10-task evaluation suite and stores results
in a structured folder hierarchy for analysis and comparison.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Import evaluation components
from evaluation.evaluator import LLMAgentEvaluator
from evaluation.storage import EvaluationStorage, EnhancedEvaluator, RunMetadata
from evaluation.task_suite import TaskSuite
from agent.llm_agent import LLMConfig


def main():
    parser = argparse.ArgumentParser(description="Run LLM Agent Evaluation with Storage")
    parser.add_argument(
        "--run-id", 
        type=str, 
        help="Custom run ID (default: auto-generated timestamp)"
    )
    parser.add_argument(
        "--task", 
        type=str, 
        help="Run a specific task (e.g., task_1, task_2)"
    )
    parser.add_argument(
        "--difficulty", 
        type=str, 
        choices=["easy", "medium", "hard"],
        help="Run only tasks of a specific difficulty"
    )
    parser.add_argument(
        "--category", 
        type=str, 
        choices=["file_creation", "file_modification", "complex_task"],
        help="Run only tasks of a specific category"
    )
    parser.add_argument(
        "--list-runs", 
        action="store_true",
        help="List all previous evaluation runs"
    )
    parser.add_argument(
        "--compare-runs", 
        nargs="+",
        help="Compare multiple runs (e.g., --compare-runs run_1 run_2 run_3)"
    )
    parser.add_argument(
        "--show-run", 
        type=str,
        help="Show details of a specific run"
    )
    
    args = parser.parse_args()
    
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
        sys.exit(1)
    
    print("🔑 API key loaded successfully")
    
    # Initialize storage
    storage = EvaluationStorage()
    
    if args.list_runs:
        print("\n📁 Available Evaluation Runs:")
        print("=" * 50)
        runs = storage.list_runs()
        if runs:
            for run in runs:
                print(f"  {run['run_id']}: {run['success_rate']:.1f}% success rate")
                print(f"    Tasks: {run['total_tasks']}, Timestamp: {run['timestamp']}")
                print(f"    Path: {run['path']}")
                print()
        else:
            print("  No previous runs found.")
        return
    
    if args.compare_runs:
        print(f"\n📊 Comparing Runs: {', '.join(args.compare_runs)}")
        print("=" * 50)
        comparison = storage.compare_runs(args.compare_runs)
        
        print(f"Total Runs: {comparison['summary']['total_runs']}")
        print(f"Best Success Rate: {comparison['summary']['best_success_rate']:.1f}%")
        print(f"Average Success Rate: {comparison['summary']['average_success_rate']:.1f}%")
        
        print("\nDetailed Comparison:")
        for run_id, data in comparison['runs'].items():
            metadata = data['metadata']
            print(f"  {run_id}: {metadata.get('success_rate', 0):.1f}% ({metadata.get('total_tasks', 0)} tasks)")
        return
    
    if args.show_run:
        print(f"\n📋 Run Details: {args.show_run}")
        print("=" * 50)
        try:
            details = storage.get_run_details(args.show_run)
            metadata = details['metadata']
            print(f"Success Rate: {metadata.get('success_rate', 0):.1f}%")
            print(f"Total Tasks: {metadata.get('total_tasks', 0)}")
            print(f"Model: {metadata.get('model', 'unknown')}")
            print(f"Timestamp: {metadata.get('timestamp', 'unknown')}")
            
            print("\nTask Results:")
            for task in details['tasks']:
                status = "PASS" if task.get('success', False) else "FAIL"
                print(f"  {task.get('task_id', 'unknown')}: {status}")
        except ValueError as e:
            print(f"❌ Error: {e}")
        return
    
    # Initialize enhanced evaluator
    enhanced_evaluator = EnhancedEvaluator(storage)
    
    # Start new run
    run_id = enhanced_evaluator.start_run(args.run_id)
    
    # Initialize regular evaluator for task execution
    evaluator = LLMAgentEvaluator()
    
    # Run evaluation
    if args.task:
        # Run single task
        print(f"🧪 Running single task: {args.task}")
        try:
            result = evaluator.run_single_task(args.task)
            evaluator.results = [result]
            
            # Store the result
            workspace_dir = Path(evaluator.workspace_base) / f"task_{args.task}"
            raw_logs = []  # Could capture additional logs here
            enhanced_evaluator.store_task_with_logs(result, workspace_dir, raw_logs)
            
            print(f"\n📊 Task Result:")
            print(f"  Success: {result.success}")
            print(f"  Steps: {result.steps_taken}/{result.max_steps}")
            print(f"  Duration: {result.total_duration:.2f}s")
            if result.missing_files:
                print(f"  Missing files: {result.missing_files}")
            if result.extra_files:
                print(f"  Extra files: {result.extra_files}")
        except ValueError as e:
            print(f"❌ Error: {e}")
            return
    else:
        # Run all tasks or filtered tasks
        if args.difficulty or args.category:
            # Filter tasks
            all_tasks = evaluator.task_suite.get_all_tasks()
            filtered_tasks = []
            
            for task in all_tasks:
                include = True
                if args.difficulty and task.difficulty != args.difficulty:
                    include = False
                if args.category and task.category != args.category:
                    include = False
                if include:
                    filtered_tasks.append(task)
            
            print(f"🧪 Running {len(filtered_tasks)} filtered tasks...")
            results = []
            for task in filtered_tasks:
                try:
                    result = evaluator.run_single_task(task.id)
                    results.append(result)
                    
                    # Store each result
                    workspace_dir = Path(evaluator.workspace_base) / f"task_{task.id}"
                    raw_logs = []
                    enhanced_evaluator.store_task_with_logs(result, workspace_dir, raw_logs)
                    
                except Exception as e:
                    print(f"❌ Error running task {task.id}: {e}")
            
            evaluator.results = results
        else:
            # Run all tasks
            print("🧪 Running all 10 evaluation tasks...")
            evaluator.run_all_tasks()
            
            # Store all results
            for result in evaluator.results:
                workspace_dir = Path(evaluator.workspace_base) / f"task_{result.task_id}"
                raw_logs = []
                enhanced_evaluator.store_task_with_logs(result, workspace_dir, raw_logs)
    
    # Generate and store report
    if evaluator.results:
        report = evaluator.generate_report()
        
        # Create run metadata
        metadata = RunMetadata(
            run_id=run_id,
            timestamp=datetime.now().isoformat(),
            model="moonshotai/kimi-k2",
            total_tasks=len(evaluator.results),
            successful_tasks=sum(1 for r in evaluator.results if r.success),
            success_rate=report["summary"]["success_rate"],
            average_steps=report["summary"]["average_steps"],
            average_duration=report["summary"]["average_duration"],
            failure_patterns=report.get("failure_analysis", {})
        )
        
        # End the run
        enhanced_evaluator.end_run(metadata, report)
        
        # Print summary
        print("\n📊 Evaluation Summary")
        print("=" * 50)
        summary = report["summary"]
        print(f"Total Tasks: {summary['total_tasks']}")
        print(f"Successful: {summary['successful_tasks']}")
        print(f"Success Rate: {summary['success_rate']:.1f}%")
        print(f"Average Steps: {summary['average_steps']:.1f}")
        print(f"Average Duration: {summary['average_duration']:.2f}s")
        
        # Print difficulty breakdown
        if "difficulty_breakdown" in report:
            print("\n📈 Difficulty Breakdown:")
            for difficulty, stats in report["difficulty_breakdown"].items():
                print(f"  {difficulty.capitalize()}: {stats['success_rate']:.1f}% ({stats['successful']}/{stats['total']})")
        
        # Print failure analysis
        if "failure_analysis" in report:
            failure_analysis = report["failure_analysis"]
            print("\n🔍 Failure Analysis:")
            if failure_analysis["missing_files"]:
                print("  Most missing files:")
                for file, count in sorted(failure_analysis["missing_files"].items(), key=lambda x: x[1], reverse=True)[:3]:
                    print(f"    {file}: {count} times")
            if failure_analysis["parsing_errors"] > 0:
                print(f"  Parsing errors: {failure_analysis['parsing_errors']}")
            if failure_analysis["tool_execution_errors"] > 0:
                print(f"  Tool execution errors: {failure_analysis['tool_execution_errors']}")
            if failure_analysis["api_errors"] > 0:
                print(f"  API errors: {failure_analysis['api_errors']}")
        
        print(f"\n💾 Results stored in: runs/{run_id}/")
        print(f"📁 Task details: runs/{run_id}/tasks/")
        print(f"📊 Reports: runs/{run_id}/reports/")
    else:
        print("❌ No results to report")


if __name__ == "__main__":
    main() 