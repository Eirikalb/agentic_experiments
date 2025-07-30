#!/usr/bin/env python3
"""
Unified Benchmark Runner

This script runs all benchmarks with automatic storage and provides
a consistent interface for managing benchmark runs.
"""

import argparse
import sys
from pathlib import Path

# Import benchmarks
from benchmarks.benchmark_1 import FileCreationBenchmark
from benchmarks.benchmark_2 import LLMAgentBenchmark
from benchmarks.benchmark_execution import ExecutionBenchmark
from benchmarks.benchmark_storage import BenchmarkStorage


def run_all_benchmarks(run_id: str = None, enable_storage: bool = True):
    """Run all benchmarks with storage."""
    results = {}
    
    print("🚀 Starting All Benchmarks")
    print("=" * 60)
    
    # Run File Creation Benchmark
    print("\n📁 Running File Creation Benchmark...")
    try:
        benchmark1 = FileCreationBenchmark(enable_storage=enable_storage, run_id=f"{run_id}_file_creation" if run_id else None)
        results["file_creation"] = benchmark1.run_benchmark()
    except Exception as e:
        print(f"❌ File Creation Benchmark failed: {e}")
        results["file_creation"] = {"success": False, "error": str(e)}
    
    # Run LLM Agent Benchmark
    print("\n🤖 Running LLM Agent Benchmark...")
    try:
        benchmark2 = LLMAgentBenchmark(enable_storage=enable_storage, run_id=f"{run_id}_llm_agent" if run_id else None)
        results["llm_agent"] = benchmark2.run_benchmark()
    except Exception as e:
        print(f"❌ LLM Agent Benchmark failed: {e}")
        results["llm_agent"] = {"success": False, "error": str(e)}
    
    # Run Execution Benchmark
    print("\n⚡ Running Execution Benchmark...")
    try:
        benchmark3 = ExecutionBenchmark(enable_storage=enable_storage, run_id=f"{run_id}_execution" if run_id else None)
        results["execution"] = benchmark3.run_benchmark()
    except Exception as e:
        print(f"❌ Execution Benchmark failed: {e}")
        results["execution"] = {"success": False, "error": str(e)}
    
    # Print overall summary
    print("\n" + "=" * 60)
    print("📊 OVERALL BENCHMARK SUMMARY")
    print("=" * 60)
    
    total_benchmarks = len(results)
    successful_benchmarks = sum(1 for r in results.values() if r.get("success", False))
    
    for benchmark_name, result in results.items():
        status = "✅ PASS" if result.get("success", False) else "❌ FAIL"
        success_rate = result.get("success_rate", 0)
        print(f"{status} {benchmark_name.replace('_', ' ').title()}: {success_rate:.1f}%")
    
    print(f"\nOverall: {successful_benchmarks}/{total_benchmarks} benchmarks passed")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run All Benchmarks with Storage")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["file_creation", "llm_agent", "execution", "all"],
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
        "--compare-runs",
        nargs="+",
        help="Compare multiple runs"
    )
    
    args = parser.parse_args()
    
    # Load environment variables
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    # Check API key for LLM benchmarks
    import os
    api_key = os.getenv('OPENROUTER_API_KEY')
    if not api_key and args.benchmark in ["llm_agent", "all"]:
        print("⚠️  Warning: No OPENROUTER_API_KEY found in environment")
        print("   LLM agent benchmarks will fail without an API key")
        print("   Set the API key in your .env file or environment variables")
    
    if args.list_runs:
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
    
    if args.compare_runs:
        storage = BenchmarkStorage()
        print(f"\n📊 Comparing Runs: {', '.join(args.compare_runs)}")
        print("=" * 50)
        comparison = storage.compare_benchmark_runs(args.compare_runs)
        
        print(f"Total Runs: {comparison['summary']['total_runs']}")
        print(f"Best Success Rate: {comparison['summary']['best_success_rate']:.1f}%")
        print(f"Average Success Rate: {comparison['summary']['average_success_rate']:.1f}%")
        
        print("\nDetailed Comparison:")
        for run_id, data in comparison['runs'].items():
            metadata = data['metadata']
            print(f"  {run_id}: {metadata.get('success_rate', 0):.1f}% ({metadata.get('total_tests', 0)} tests)")
        return
    
    # Run selected benchmark(s)
    if args.benchmark == "all":
        results = run_all_benchmarks(args.run_id, not args.no_storage)
    else:
        results = {}
        
        if args.benchmark == "file_creation":
            print("📁 Running File Creation Benchmark...")
            benchmark = FileCreationBenchmark(
                enable_storage=not args.no_storage,
                run_id=args.run_id
            )
            results["file_creation"] = benchmark.run_benchmark()
            
        elif args.benchmark == "llm_agent":
            print("🤖 Running LLM Agent Benchmark...")
            benchmark = LLMAgentBenchmark(
                enable_storage=not args.no_storage,
                run_id=args.run_id
            )
            results["llm_agent"] = benchmark.run_benchmark()
            
        elif args.benchmark == "execution":
            print("⚡ Running Execution Benchmark...")
            benchmark = ExecutionBenchmark(
                enable_storage=not args.no_storage,
                run_id=args.run_id
            )
            results["execution"] = benchmark.run_benchmark()
    
    # Print final summary
    if results:
        print(f"\n🎯 Benchmark run completed!")
        if not args.no_storage:
            print("💾 Results stored automatically in runs/ directory")
        
        # Check if any benchmarks failed
        failed_benchmarks = [name for name, result in results.items() if not result.get("success", False)]
        if failed_benchmarks:
            print(f"⚠️  Failed benchmarks: {', '.join(failed_benchmarks)}")
        else:
            print("✅ All benchmarks completed successfully!")
    
    return results


if __name__ == "__main__":
    main() 