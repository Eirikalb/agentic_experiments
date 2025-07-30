#!/usr/bin/env python3
"""
LLM Agent Evaluation Runner

This script runs the 10-task evaluation suite and generates detailed reports
for prompt optimization analysis.
"""

import argparse
import json
import sys
from pathlib import Path

# Import evaluation components
from evaluation.evaluator import LLMAgentEvaluator
from evaluation.task_suite import TaskSuite


def main():
    parser = argparse.ArgumentParser(description="Run LLM Agent Evaluation")
    parser.add_argument(
        "--task", 
        type=str, 
        help="Run a specific task (e.g., task_1, task_2)"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="evaluation_results.json",
        help="Output file for results (default: evaluation_results.json)"
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
        "--list-tasks", 
        action="store_true",
        help="List all available tasks"
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
    
    # Initialize evaluator
    evaluator = LLMAgentEvaluator()
    
    if args.list_tasks:
        print("\n📋 Available Tasks:")
        print("=" * 50)
        for task in evaluator.task_suite.get_all_tasks():
            print(f"  {task.id}: {task.name}")
            print(f"    Difficulty: {task.difficulty}")
            print(f"    Category: {task.category}")
            print(f"    Description: {task.description}")
            print()
        return
    
    # Run evaluation
    if args.task:
        # Run single task
        print(f"🧪 Running single task: {args.task}")
        try:
            result = evaluator.run_single_task(args.task)
            evaluator.results = [result]
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
                except Exception as e:
                    print(f"❌ Error running task {task.id}: {e}")
            
            evaluator.results = results
        else:
            # Run all tasks
            print("🧪 Running all 10 evaluation tasks...")
            evaluator.run_all_tasks()
    
    # Generate and save report
    if evaluator.results:
        report = evaluator.generate_report()
        
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
        
        # Save detailed results
        evaluator.save_results(args.output)
        print(f"\n💾 Detailed results saved to: {args.output}")
        
        # Save a human-readable summary
        summary_file = args.output.replace('.json', '_summary.txt')
        with open(summary_file, 'w') as f:
            f.write("LLM Agent Evaluation Summary\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Tasks: {summary['total_tasks']}\n")
            f.write(f"Successful: {summary['successful_tasks']}\n")
            f.write(f"Success Rate: {summary['success_rate']:.1f}%\n")
            f.write(f"Average Steps: {summary['average_steps']:.1f}\n")
            f.write(f"Average Duration: {summary['average_duration']:.2f}s\n\n")
            
            f.write("Task Results:\n")
            f.write("-" * 20 + "\n")
            for result in evaluator.results:
                status = "PASS" if result.success else "FAIL"
                f.write(f"{result.task_id}: {result.task_name} - {status}\n")
                if result.missing_files:
                    f.write(f"  Missing: {result.missing_files}\n")
                if result.error_message:
                    f.write(f"  Error: {result.error_message}\n")
        
        print(f"📝 Summary saved to: {summary_file}")
    else:
        print("❌ No results to report")


if __name__ == "__main__":
    main() 