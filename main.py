"""
Main entry point for the Coding Agent Benchmark Suite.

This script provides a command-line interface to run benchmarks and test
the coding agent's capabilities.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Dict, Any

# Import our components
from agent.coding_agent import CodingAgent
from benchmarks.benchmark_1 import FileCreationBenchmark


def run_benchmark(benchmark_name: str) -> Dict[str, Any]:
    """
    Run a specific benchmark.
    
    Args:
        benchmark_name: Name of the benchmark to run
        
    Returns:
        dict: Benchmark results
    """
    if benchmark_name == "file_creation":
        benchmark = FileCreationBenchmark()
        return benchmark.run_benchmark()
    else:
        return {
            "success": False,
            "message": f"Unknown benchmark: {benchmark_name}",
            "available_benchmarks": ["file_creation"]
        }


def run_interactive_mode():
    """Run the agent in interactive mode for manual testing."""
    print("🤖 Coding Agent Interactive Mode")
    print("=" * 40)
    print("Type 'help' for available commands, 'quit' to exit")
    print()
    
    agent = CodingAgent()
    
    while True:
        try:
            command = input("agent> ").strip()
            
            if command.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            elif command.lower() == 'help':
                print("\nAvailable commands:")
                print("  <task description>  - Execute a coding task")
                print("  tools               - List available tools")
                print("  context             - Show current context")
                print("  reset               - Reset agent context")
                print("  help                - Show this help")
                print("  quit                - Exit interactive mode")
                print()
            elif command.lower() == 'tools':
                tools = agent.get_available_tools()
                print(f"\nAvailable tools ({len(tools)}):")
                for tool in tools:
                    description = agent.get_tool_description(tool)
                    print(f"  {tool}: {description.split('.')[0]}")
                print()
            elif command.lower() == 'context':
                print(f"\nCurrent Context:")
                print(agent.get_context_summary())
                print()
            elif command.lower() == 'reset':
                agent.reset_context()
                print("Agent context reset.")
                print()
            elif command:
                print(f"\nExecuting: {command}")
                result = agent.execute_task(command)
                print(f"\nResult: {result['message']}")
                if result.get('final_context'):
                    print(f"\nFinal Context:")
                    print(result['final_context'])
                print()
            else:
                continue
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def run_single_task(task_description: str, workspace_dir: str = ".") -> Dict[str, Any]:
    """
    Run a single task with the agent.
    
    Args:
        task_description: The task to execute
        workspace_dir: Directory to work in
        
    Returns:
        dict: Task execution result
    """
    print(f"🤖 Executing task: {task_description}")
    print(f"📁 Working in: {workspace_dir}")
    print("=" * 50)
    
    agent = CodingAgent(workspace_dir)
    result = agent.execute_task(task_description)
    
    print(f"\n📊 Task Result:")
    print(f"  Success: {result['success']}")
    print(f"  Message: {result['message']}")
    print(f"  Steps taken: {result['steps_taken']}")
    
    if result.get('final_context'):
        print(f"\n📋 Final Context:")
        print(result['final_context'])
    
    return result


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Coding Agent Benchmark Suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --benchmark file_creation
  python main.py --task "Create a file called test.py"
  python main.py --interactive
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        '--benchmark', 
        choices=['file_creation'],
        help='Run a specific benchmark'
    )
    group.add_argument(
        '--task',
        help='Execute a single task'
    )
    group.add_argument(
        '--interactive', '-i',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--workspace', '-w',
        default='.',
        help='Workspace directory (default: current directory)'
    )
    
    parser.add_argument(
        '--output', '-o',
        help='Output file for results (JSON format)'
    )
    
    args = parser.parse_args()
    
    try:
        if args.benchmark:
            print(f"🏃 Running benchmark: {args.benchmark}")
            result = run_benchmark(args.benchmark)
            
        elif args.task:
            result = run_single_task(args.task, args.workspace)
            
        elif args.interactive:
            run_interactive_mode()
            return
            
        # Save results if output file specified
        if args.output and 'result' in locals():
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\n💾 Results saved to: {args.output}")
            
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 