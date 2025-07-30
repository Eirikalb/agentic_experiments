# Coding Agent Benchmark Suite

A comprehensive framework for building, testing, and optimizing AI coding agents with both heuristic and LLM-powered approaches.

## Overview

This project provides a complete suite for developing coding agents that can manipulate files and write code. It includes:

- **File manipulation tools**: Create, read, edit, delete, and diff files
- **Heuristic agent**: Rule-based agent for basic file operations
- **LLM-powered agent**: AI-driven agent using OpenRouter for intelligent task execution
- **Benchmarking system**: Automated testing and performance measurement
- **Meta-optimization framework**: Tools for optimizing LLM prompts and parsing strategies

## Architecture

### Core Components

1. **Tools** (`tools/`): File manipulation utilities
2. **Agents** (`agent/`): Both heuristic and LLM-powered agents
3. **Benchmarks** (`benchmarks/`): Performance testing suites
4. **Optimization** (`optimization/`): Meta-LLM optimization utilities

### LLM Integration

The system supports LLM-powered agents with:
- **Configurable prompting**: Control system prompts, task templates, and context inclusion
- **Flexible parsing**: Multiple response parsing strategies (JSON, structured text, freeform)
- **OpenRouter integration**: Easy access to multiple LLM models
- **Meta-optimization**: Separate LLM for optimizing prompts and parsing strategies

## Quick Start

### Basic Usage

```bash
# Run the heuristic agent benchmark
python main.py --benchmark file_creation

# Run the LLM agent benchmark (requires API key)
python main.py --benchmark llm_agent

# Execute a single task with heuristic agent
python main.py --task "Create a file called hello.py"

# Execute a single task with LLM agent
python main.py --task "Create a file called hello.py" --llm

# Interactive mode with heuristic agent
python main.py --interactive

# Interactive mode with LLM agent
python main.py --interactive --llm
```

### LLM Setup

1. **Get an OpenRouter API key** from [openrouter.ai](https://openrouter.ai)
2. **Set up your API key** using one of these methods:
   
   **Option A: Environment variable**
   ```bash
   export OPENROUTER_API_KEY="your_api_key_here"
   ```
   
   **Option B: .env file (recommended)**
   ```bash
   # Create a .env file in the project root
   echo "OPENROUTER_API_KEY=your_api_key_here" > .env
   ```
   
   The system automatically loads environment variables from `.env` files using python-dotenv.
   
3. **Install dependencies** (if not already installed):
   ```bash
   pip install python-dotenv requests
   ```
   
4. **Run LLM benchmarks**:
   ```bash
   python main.py --benchmark llm_agent
   ```

### Model Configuration

The system is configured to use **Kimi-K2** (`moonshotai/kimi-k2:free`) by default, which offers:

- **Free tier available** with rate limits
- **XML-style tool calling** format
- **Good performance** for coding tasks
- **Automatic parameter mapping** from `file_path` to `path`

To use a different model, modify the `LLMConfig`:

```python
llm_config = LLMConfig(
    model="anthropic/claude-3.5-sonnet",  # or any other model
    max_tokens=4000,
    temperature=0.1
)
```

## Available Tools

The agent has access to these file manipulation tools:

- `create_file(path, content)`: Create a new file with optional content
- `read_file(path)`: Read and return file content
- `edit_file(path, new_content)`: Replace entire file content
- `list_files(directory)`: List files and directories
- `delete_file(path)`: Delete a file
- `apply_diff(path, diff_content)`: Apply a diff to a file
- `read_diff(path)`: Show file differences

## Benchmarks

### File Creation Benchmark (`benchmark_1.py`)
Tests basic file creation capabilities:
- Create Python Hello World
- Create Text File
- Create Multiple Files

### LLM Agent Benchmark (`benchmark_2.py`)
Tests LLM-powered task execution:
- LLM Create Python File
- LLM Create Multiple Files
- LLM Complex Task

## LLM Configuration

### LLM Agent Configuration

```python
from agent.llm_agent import LLMCodingAgent, LLMConfig, PromptConfig, ParserConfig

# Configure the LLM
llm_config = LLMConfig(
    model="moonshotai/kimi-k2",
    max_tokens=4000,
    temperature=0.6
)

# Configure prompting strategy
prompt_config = PromptConfig(
    system_prompt="You are a helpful coding assistant.",
    context_inclusion=True,
    action_history_inclusion=True
)

# Configure response parsing
parser_config = ParserConfig(
    response_format="structured_text",
    tool_call_extraction="regex"
)

# Create the agent
agent = LLMCodingAgent(
    workspace_dir="./workspace",
    llm_config=llm_config,
    prompt_config=prompt_config,
    parser_config=parser_config
)
```

### Meta-Optimization

The system supports using a separate LLM to optimize prompts and parsing strategies:

```python
from optimization.optimizer import MetaOptimizer

# Create meta-optimizer
meta_optimizer = MetaOptimizer(meta_llm_config)

# Optimize prompts based on performance data
optimized_prompt = meta_optimizer.optimize_prompt(
    current_prompt_config,
    performance_data
)

# Optimize parsing strategies
optimized_parser = meta_optimizer.optimize_parser(
    current_parser_config,
    parsing_history
)
```

## Development

### Adding New Tools

1. Add the tool function to `tools/file_tools.py`
2. Update the tool registration in the agent classes
3. Add tool descriptions to the prompt engine

### Adding New Benchmarks

1. Create a new benchmark class in `benchmarks/`
2. Implement `setup()`, `run_test()`, `verify_test()`, and `run_benchmark()` methods
3. Add the benchmark to `main.py`

### LLM Integration

The LLM integration is designed for easy customization:

- **Prompt Engine**: Control how prompts are generated
- **Response Parser**: Control how LLM responses are interpreted
- **LLM Client**: Control which LLM service to use
- **Configuration**: Separate configs for different components

## Future Enhancements

- [ ] Support for more LLM providers (OpenAI, Anthropic, etc.)
- [ ] Advanced prompt optimization strategies
- [ ] Multi-agent collaboration
- [ ] Code quality metrics and optimization
- [ ] Integration with version control systems
- [ ] Web-based interface for benchmark visualization

## Troubleshooting

### Common Issues

1. **"No OpenRouter API key"**: Set the `OPENROUTER_API_KEY` environment variable or create a `.env` file
2. **Import errors**: Ensure you're running from the project root directory
3. **File permission errors**: Check that the workspace directory is writable
4. **"402 Payment Required"**: Your OpenRouter account needs credits. Add funds at [openrouter.ai](https://openrouter.ai)
5. **"401 Unauthorized"**: Check that your API key is correct and active
6. **"429 Too Many Requests"**: You've hit the rate limit. Wait a moment and try again
7. **"Kimi-K2 rate limits"**: The free tier has strict rate limits. Consider upgrading or using a different model
8. **Parameter mapping issues**: The system automatically maps `file_path` to `path` for Kimi-K2 compatibility

### Performance Tips

- Use smaller models for faster responses
- Adjust `max_tokens` based on task complexity
- Enable context inclusion for better task understanding
- Use JSON response format for more reliable parsing

## Example Usage

```python
from agent.llm_agent import LLMCodingAgent

# Create an LLM agent
agent = LLMCodingAgent(workspace_dir="./my_project")

# Execute a complex task
result = agent.execute_task(
    "Create a Python web server with Flask that serves a simple API"
)

print(f"Task completed: {result['success']}")
print(f"Steps taken: {result['steps_taken']}")
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## License

This project is open source and available under the MIT License.