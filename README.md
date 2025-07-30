# Coding Agent Benchmark Suite

A comprehensive benchmark suite for testing and improving code-generating agents. This project provides a framework for building agents that can manipulate files and write code, with built-in benchmarks to measure and improve their performance.

## 🎯 Project Overview

This suite consists of:

1. **File Manipulation Tools** - A set of Python functions for file operations
2. **Coding Agent** - An agent that uses these tools to execute coding tasks
3. **Benchmarks** - Tests to measure agent performance and capabilities
4. **Interactive Interface** - Command-line tools for testing and experimentation

## 🏗️ Architecture

```
agentic_experiments/
├── tools/
│   ├── __init__.py
│   └── file_tools.py          # File manipulation tools
├── agent/
│   ├── __init__.py
│   └── coding_agent.py        # Main coding agent
├── benchmarks/
│   ├── __init__.py
│   └── benchmark_1.py         # File creation benchmark
├── main.py                    # Main entry point
├── requirements.txt           # Dependencies
└── README.md                 # This file
```

## 🛠️ Available Tools

The agent has access to the following tools:

- **`create_file(path, content)`** - Create a new file with optional content
- **`read_file(path)`** - Read the content of a file
- **`edit_file(path, new_content)`** - Replace file content
- **`list_files(directory)`** - List files and directories
- **`delete_file(path)`** - Delete a file
- **`apply_diff(path, diff_content)`** - Apply a diff to a file
- **`read_diff(path)`** - Show file differences

## 🚀 Quick Start

### Prerequisites

- Python 3.7 or higher
- No external dependencies required for basic functionality

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd agentic_experiments
```

2. The project uses only built-in Python modules, so no installation is required.

### Running Benchmarks

Run the file creation benchmark:
```bash
python main.py --benchmark file_creation
```

### Interactive Mode

Test the agent interactively:
```bash
python main.py --interactive
```

Available commands in interactive mode:
- `<task description>` - Execute a coding task
- `tools` - List available tools
- `context` - Show current context
- `reset` - Reset agent context
- `help` - Show help
- `quit` - Exit

### Single Task Execution

Run a specific task:
```bash
python main.py --task "Create a file called hello.py with a hello world function"
```

## 📊 Benchmarks

### Benchmark 1: File Creation

Tests the agent's ability to create files with different content types:

1. **Create Python Hello World** - Creates a Python file with a hello world function
2. **Create Text File** - Creates a text file with content
3. **Create Multiple Files** - Creates multiple files in one task

## 🔧 Development

### Adding New Tools

To add a new tool:

1. Add the function to `tools/file_tools.py`
2. Update the `__all__` list in `tools/__init__.py`
3. Add the tool to the agent's tools dictionary in `agent/coding_agent.py`

### Adding New Benchmarks

To add a new benchmark:

1. Create a new file in the `benchmarks/` directory
2. Implement a benchmark class with `setup()`, `run_test()`, and `run_benchmark()` methods
3. Add the benchmark to the `run_benchmark()` function in `main.py`

### LLM Integration

The current agent uses basic heuristics for task execution. To integrate with an LLM:

1. Add LLM client dependencies to `requirements.txt`
2. Modify the `execute_task()` method in `CodingAgent` to use LLM for decision making
3. Implement proper prompt engineering for tool selection

## 🎯 Future Enhancements

- **LLM Integration** - Connect to OpenAI, Anthropic, or other LLM providers
- **Advanced Diff Tools** - Implement proper diff parsing and application
- **Git Integration** - Add version control capabilities
- **Code Analysis** - Add tools for code quality assessment
- **Multi-step Tasks** - Support for complex, multi-step coding tasks
- **Performance Metrics** - Detailed timing and success rate tracking
- **Web Interface** - GUI for easier interaction and visualization

## 📝 Example Usage

### Basic File Creation
```python
from agent.coding_agent import CodingAgent

agent = CodingAgent()
result = agent.execute_task("Create a file called test.py with a main function")
print(result)
```

### Running a Benchmark
```python
from benchmarks.benchmark_1 import FileCreationBenchmark

benchmark = FileCreationBenchmark()
result = benchmark.run_benchmark()
print(f"Success rate: {result['success_rate']:.1%}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is open source. Please check the license file for details.

## 🐛 Troubleshooting

### Import Errors
If you encounter import errors, make sure you're running from the project root directory and that all `__init__.py` files are present.

### File Permission Errors
The agent needs write permissions in the workspace directory. Make sure the directory is writable.

### Tool Execution Failures
Check that the file paths are valid and that the agent has the necessary permissions to perform the requested operations.