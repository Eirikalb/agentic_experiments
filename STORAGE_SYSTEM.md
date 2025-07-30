# LLM Agent Evaluation Storage System

## 🎯 **Overview**

The storage system provides structured organization for evaluation runs and tasks, storing all relevant data including raw logs, prompts, generated files, and execution traces. This enables detailed analysis, debugging, and comparison of different evaluation runs.

## 📁 **Directory Structure**

```
runs/
├── run_20250730_121523/           # Individual evaluation run
│   ├── metadata.json              # Run metadata (success rate, model, etc.)
│   ├── tasks/                     # Task results
│   │   ├── task_1/               # Individual task
│   │   │   ├── metadata.json     # Task metadata
│   │   │   ├── execution_steps.json # Detailed execution traces
│   │   │   ├── raw_logs.json     # Raw LLM logs
│   │   │   ├── summary.txt       # Human-readable summary
│   │   │   └── files/            # Generated files
│   │   │       └── hello.py      # Actual files created by agent
│   │   └── task_2/               # Another task...
│   ├── reports/                   # Run reports
│   │   ├── evaluation_report.json # Full evaluation report
│   │   └── summary.txt           # Human-readable run summary
│   └── logs/                      # Additional logs (future use)
└── run_20250730_121449/          # Another run...
```

## 🛠 **Key Components**

### **1. EvaluationStorage**
- **Purpose**: Manages storage of evaluation runs and tasks
- **Features**: 
  - Creates structured directory hierarchy
  - Stores task results with all associated files
  - Provides run comparison and analysis tools

### **2. EnhancedEvaluator**
- **Purpose**: Enhanced evaluator with storage capabilities
- **Features**:
  - Manages run lifecycle (start/end)
  - Stores task results with logs and files
  - Integrates with regular evaluator

### **3. RunMetadata**
- **Purpose**: Stores metadata about evaluation runs
- **Fields**: run_id, timestamp, model, success_rate, etc.

## 📊 **Stored Data**

### **Task Level**
- **metadata.json**: Task success/failure, steps taken, duration
- **execution_steps.json**: Detailed step-by-step execution traces
- **raw_logs.json**: Raw LLM responses and API calls
- **summary.txt**: Human-readable task summary
- **files/**: All generated files from the task

### **Run Level**
- **metadata.json**: Overall run statistics and configuration
- **reports/evaluation_report.json**: Full evaluation report
- **reports/summary.txt**: Human-readable run summary

## 🚀 **Usage Examples**

### **Run a Single Task**
```bash
python run_evaluation_with_storage.py --task task_1
```

### **Run All Tasks**
```bash
python run_evaluation_with_storage.py
```

### **Run with Custom Run ID**
```bash
python run_evaluation_with_storage.py --run-id baseline_run_1
```

### **List All Runs**
```bash
python run_evaluation_with_storage.py --list-runs
```

### **Show Run Details**
```bash
python run_evaluation_with_storage.py --show-run run_20250730_121523
```

### **Compare Multiple Runs**
```bash
python run_evaluation_with_storage.py --compare-runs run_1 run_2 run_3
```

## 📈 **Analysis Capabilities**

### **1. Task-Level Analysis**
- **Execution Traces**: Step-by-step analysis of agent behavior
- **File Generation**: Verification of created files and content
- **Performance Metrics**: Duration, success rate, step efficiency

### **2. Run-Level Analysis**
- **Overall Performance**: Success rates across all tasks
- **Failure Patterns**: Common failure modes and their frequency
- **Model Comparison**: Performance across different models/configurations

### **3. Cross-Run Comparison**
- **Trend Analysis**: Performance improvements over time
- **Prompt Optimization**: Impact of prompt changes on performance
- **Model Selection**: Comparison of different LLM models

## 🔍 **Data Access**

### **Programmatic Access**
```python
from evaluation.storage import EvaluationStorage

# List all runs
storage = EvaluationStorage()
runs = storage.list_runs()

# Get run details
details = storage.get_run_details("run_20250730_121523")

# Compare runs
comparison = storage.compare_runs(["run_1", "run_2", "run_3"])
```

### **File Access**
```python
# Access task execution steps
with open("runs/run_20250730_121523/tasks/task_1/execution_steps.json") as f:
    steps = json.load(f)

# Access generated files
generated_files = list(Path("runs/run_20250730_121523/tasks/task_1/files").iterdir())
```

## 📋 **Stored Information**

### **Task Metadata**
```json
{
  "task_id": "task_1",
  "task_name": "Simple Python File",
  "success": true,
  "steps_taken": 1,
  "max_steps": 5,
  "total_duration": 15.71,
  "expected_files": ["hello.py"],
  "missing_files": [],
  "extra_files": [],
  "content_analysis": {...},
  "error_message": null,
  "timestamp": "2025-07-30T12:15:39.035166"
}
```

### **Execution Steps**
```json
[
  {
    "step_number": 1,
    "prompt": "Task: Create a file called hello.py...",
    "llm_response": "I'll create a file called hello.py...",
    "parsed_tool_call": {
      "tool": "create_file",
      "parameters": {"path": "hello.py", "content": "..."}
    },
    "tool_execution_result": {
      "success": true,
      "message": "File 'hello.py' created successfully"
    },
    "timestamp": 1753869339.035166,
    "duration": 15.71
  }
]
```

### **Run Metadata**
```json
{
  "run_id": "run_20250730_121523",
  "timestamp": "2025-07-30T12:15:39.035166",
  "model": "moonshotai/kimi-k2",
  "total_tasks": 1,
  "successful_tasks": 1,
  "success_rate": 100.0,
  "average_steps": 1.0,
  "average_duration": 15.71,
  "failure_patterns": {...}
}
```

## 🎯 **Benefits**

### **1. Reproducibility**
- Complete execution traces for debugging
- All generated files preserved
- Raw LLM responses for analysis

### **2. Analysis**
- Detailed failure analysis
- Performance trend tracking
- Prompt optimization insights

### **3. Comparison**
- Cross-run performance comparison
- Model selection insights
- Optimization impact measurement

### **4. Debugging**
- Step-by-step execution traces
- Raw LLM responses for error analysis
- Generated files for verification

## 🔧 **Future Enhancements**

### **1. Advanced Analytics**
- Performance trend analysis
- Failure pattern clustering
- Automated optimization suggestions

### **2. Visualization**
- Performance dashboards
- Failure pattern visualization
- Comparison charts

### **3. Integration**
- CI/CD pipeline integration
- Automated testing
- Performance regression detection

## 📝 **Best Practices**

### **1. Run Naming**
- Use descriptive run IDs: `baseline_run_1`, `optimized_prompts_v2`
- Include date/time for chronological ordering
- Add notes for context

### **2. Analysis Workflow**
1. Run baseline evaluation
2. Analyze failure patterns
3. Implement optimizations
4. Run comparison evaluation
5. Measure improvements

### **3. Storage Management**
- Regular cleanup of old runs
- Archive important runs
- Backup critical data

The storage system provides a comprehensive foundation for iterative LLM agent optimization and analysis. 