# Metacontext v0.3.0 - Two-Tier Architecture

[![Tests](https://img.shields.io/badge/tests-49%20passing-brightgreen)](tests/)
[![Coverage](https://img.shields.io/badge/coverage-90%25-brightgreen)](tests/)
[![Python](https://img.shields.io/badge/python-3.12%2B-blue)](#installation)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)

**Metacontext generates intelligent metadata with revolutionary two-tier architecture: deterministic analysis + AI enrichment.**

## 🎯 Mission: Production-Ready Intelligent Metadata

Traditional file metadata tells you *what* a file is. **Metacontext tells you *why* it exists, *how* it was created, and *what* it means** - with guaranteed availability and smart AI enhancement.

**Two-Tier Approach:** 
- **Tier 1 (Deterministic)**: Always available structural analysis  
- **Tier 2 (AI Enrichment)**: Best-effort intelligent interpretation with token cost tracking

**The simple approach:** Add one function call after your data exports. Rich context is generated instantly with comprehensive cost monitoring.

## 🚀 Quick Start

```python
from metacontext.metacontextualize import metacontextualize
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Your existing model workflow (unchanged)
df = pd.DataFrame({'features': [1,2,3], 'target': [0,1,0]})
model = RandomForestClassifier().fit(df[['features']], df['target'])
joblib.dump(model, 'model.pkl')

# Generate two-tier intelligent context (one line)
metacontextualize(model, 'model.pkl')
# Creates: model_metacontext.yaml with:
# ✓ Deterministic model metadata (always available)
# ✓ AI analysis of training process (when LLM available)
# ✓ Token usage and cost tracking
```

**Example Output:**
```yaml
metacontext_version: 0.3.0
token_usage:
  total_tokens: 31761
  estimated_cost_usd: 0.0025
data_structure:
  deterministic_metadata:
    model_type: RandomForestClassifier
    hyperparameters: {...}
  ai_enrichment:
    training_data_analysis: "Analysis of feature engineering patterns..."
    performance_predictions: "Expected performance characteristics..."
```

## � Key Features

- **🏗️ Two-Tier Architecture**: Deterministic metadata + AI enrichment with graceful degradation
- **💰 Cost Tracking**: Real-time token usage and cost estimation across providers (OpenAI, Anthropic, Gemini)
- **🔍 Code Analysis**: Automatically discovers and analyzes training scripts and related code
- **📊 Smart Handlers**: Specialized analysis for models (.pkl), tabular data (.csv), and more
- **🎛️ Production Ready**: Graceful error handling, confidence assessments, fallback strategies

## 🆕 Advanced Configuration

```python
# Custom LLM configuration with cost monitoring
config = {
    'llm_provider': 'anthropic',  # or 'openai', 'gemini'
    'model_name': 'claude-3-5-haiku-20241022'
}

result = metacontextualize(
    data_object, 
    'file.csv',
    config=config,
    include_llm_analysis=True
)
```

**Features:**
- 🎨 **Custom YAML Structure**: Define your ideal metacontext format
- 🤖 **AI-Driven Content**: Each field generated using specific AI prompts
- 🔧 **VS Code Integration**: Works with GitHub Copilot and other AI assistants
- 📝 **Smart Column Analysis**: Intelligent interpretation of cryptic column names
- 🎯 **Confidence Assessment**: AI provides confidence levels for interpretations

## 🚀 Project Status

**Status**: ✅ **Revolutionary System Ready** - Code-Aware Intelligence with Universal File Intelligence

- 🧠 **Code-Aware Model Intelligence**: Revolutionary approach analyzing source code that created models
- 🚀 **Universal File Intelligence**: Specialized handlers for different file types (.pkl, .csv, .xlsx, etc.)
- ✅ **Simple API**: One-line `metacontextualize()` function with sophisticated AI analysis
- 🤖 **Real AI Integration**: Gemini and OpenAI integration for understanding code intent
- 📊 **Comprehensive Analysis**: Full context generation from training scripts, not just files
- ✅ **Production Ready**: Clean, tested, and optimized for real-world usage
- 📁 **Automatic Discovery**: Finds and analyzes training scripts automatically
- � **Business Context**: AI-powered interpretation of why models/data exist

## Overview

## ✨ Key Benefits

- **🚀 Simple & Explicit**: One function call after data export - you control when context is generated
- **⚡ Lightweight**: No import overhead, method patching, or performance impact  
- **🧠 Intelligent**: Rich data analysis with optional AI-powered business context
- **📊 Comprehensive**: Data structure, statistics, samples, and business insights
- **🔄 Portable**: Context files travel with your data automatically
- **⏰ Time-Saving**: 2-3 hours of manual documentation → 5 seconds of automation

## Installation

### Using Poetry (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd metacontext

# Configure Poetry to use .venv in project directory (optional - already configured)
poetry config virtualenvs.in-project true

# Install using Poetry (creates .venv folder in project)
poetry install

# Activate the virtual environment
poetry shell
```

### Using pip

```bash
pip install metacontext
```

### Local Development

To use metacontext as a local dependency in another project:

```bash
# In your project's pyproject.toml
[tool.poetry.dependencies]
metacontext = {path = "../path/to/metacontext", develop = true}

# Then install dependencies
poetry install
```

## Usage

## Usage

### Simple Data Context Generation

```python
import metacontext
import pandas as pd

# Step 1: Your normal data workflow (unchanged)
df = pd.DataFrame({
    'product': ['Widget A', 'Widget B'],
    'sales': [100, 150],
    'region': ['North', 'South']
})
df.to_csv('sales_report.csv')

# Step 2: Generate rich context (one function call)
context_file = metacontext.metacontextualize(df, 'sales_report.csv')
# Creates: sales_report.csv.metacontext.yaml

print(f"Context saved to: {context_file}")
```

### Advanced Usage

```python
# Custom output location
metacontext.metacontextualize(
    df, 'data.csv', 
    output_path='docs/data_context.yaml'
)

# Disable LLM analysis for faster processing
metacontext.metacontextualize(
    df, 'data.csv',
    include_llm_analysis=False
)

# Custom configuration
config = {'model': 'gpt-4', 'custom_key': 'value'}
metacontext.metacontextualize(df, 'data.csv', config=config)
```

### Command Line Interface

```bash
# Check system status
metacontext status

# Analyze entire projects (legacy functionality)
metacontext analyze /path/to/project
```

## LLM Integration (Optional)

**For enhanced AI-powered business context**, set up your LLM API key:

```bash
export OPENAI_API_KEY="sk-your-openai-key-here"
# or  
export ANTHROPIC_API_KEY="sk-ant-your-anthropic-key"
```

📖 **[Complete API Setup Guide](docs/api_setup_guide.md)** - Detailed instructions for OpenAI and Anthropic

### Configuration

Create a `config.yaml` file to customize the behavior:

```yaml
llm:
  provider: openai  # or anthropic
  model: gpt-3.5-turbo
  api_key: your-api-key-here

output:
  format: yaml  # or json
  include_content: true
  max_file_size: 5000

parsing:
  exclude_patterns:
    - "*.pyc"
    - "__pycache__"
    - "node_modules"
  include_extensions:
    - ".py"
    - ".js"
    - ".ts"
    - ".md"
```

### Environment Variables

Set your API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
# or
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Features

- **Multi-language Support**: Analyzes Python, JavaScript, TypeScript, and more
- **Git Integration**: Includes repository information and current state
- **Dependency Analysis**: Extracts and analyzes project dependencies
- **LLM Integration**: Uses OpenAI or Anthropic models for intelligent analysis
- **Structured Output**: Generates YAML or JSON formatted context
- **Configurable**: Customizable parsing and output options

## 🧪 Testing

The project has comprehensive test coverage with 41 tests across all modules:

```bash
# Run all tests
make test

# Run tests with coverage report
make test-cov

# Quick test run (stops on first failure)
make q-test

# Run specific test file
poetry run pytest tests/test_cli.py -v

# View coverage report (after make test-cov)
open htmlcov/index.html
```

### Test Structure
```
tests/
├── test_cli.py           # CLI interface tests (11 tests)
├── test_core.py          # Core functionality tests (5 tests)  
├── test_llm_handler.py   # LLM integration tests (10 tests)
├── test_parser.py        # Code parser tests (8 tests)
├── test_prompt_loader.py # Prompt system tests (5 tests)
└── test_integration.py   # End-to-end tests (2 tests)
```

**Coverage by Module:**
- CLI: 95% coverage
- Core: 100% coverage  
- LLM Handler: 87% coverage
- Parser: 88% coverage
- Prompt Loader: 89% coverage
- **Overall: 90% coverage**

## 🛠️ Development

### Quick Start

```bash
# Clone and setup
git clone <repository-url>
cd metacontext

# Complete development setup
make dev-setup

# Start working
make shell    # Activate Poetry shell
make test     # Verify everything works
make run      # Test the tool on current project
```

### Development Workflow

```bash
# Code quality checks
make check           # Run all checks (lint, format, type)
make lint           # Check code with ruff
make lint-fix       # Auto-fix linting issues  
make format         # Format code
make type-check     # Run mypy type checker

# Testing
make test           # Run all tests
make test-cov       # Run tests with coverage
make ci             # Run full CI checks locally

# Shortcuts for quick development
make fix            # Fix linting and format code
make pre-commit     # Prepare for commit (fix + test)
```

### Using the Makefile

The project includes a comprehensive Makefile with convenient shortcuts:

```bash
# Show all available commands
make help

# Development setup
make dev-setup          # Install dependencies and set up environment

# Code quality
make lint               # Check code with ruff
make lint-fix           # Auto-fix linting issues
make format             # Format code
make format-check       # Check formatting without changes
make type-check         # Run mypy type checker
make fix                # Fix linting and format code
make check              # Run all checks (lint, format, type)

# Testing
make test               # Run tests
make test-cov           # Run tests with coverage report
make q-test             # Quick test run

# Running the tool
make run                # Run metacontext on current project
make analyze            # Analyze current project structure
make version            # Show version information

# Development workflow
make ci                 # Run all CI checks locally
make pre-commit         # Prepare for commit (fix + test)

# Package management
make build              # Build the package
make clean              # Clean up cache files
```

### Project Structure

```
metacontext/
├── src/                     # Source code (importable package)
│   └── metacontext/
│       ├── __init__.py      # Package initialization  
│       ├── core.py          # Main MetacontextGenerator class
│       ├── parser.py        # CodebaseParser for multi-language analysis
│       ├── llm_handler.py   # LLMHandler for OpenAI/Anthropic integration
│       ├── prompt_loader.py # PromptLoader for template management
│       └── cli.py           # Command-line interface (analyze/main commands)
├── tests/                   # Comprehensive test suite (41 tests, 90% coverage)
│   ├── test_cli.py          # CLI tests (11 tests)
│   ├── test_core.py         # Core functionality tests (5 tests)
│   ├── test_llm_handler.py  # LLM integration tests (10 tests)  
│   ├── test_parser.py       # Parser tests (8 tests)
│   ├── test_prompt_loader.py # Prompt loader tests (5 tests)
│   └── test_integration.py  # Integration tests (2 tests)
├── docs/                    # Documentation
│   ├── technical_architecture.md
│   ├── implementation_checklist.md
│   ├── project_structure.md
│   └── ...
├── prompts/                 # LLM prompt templates
├── .github/workflows/       # CI/CD configuration  
├── Makefile                 # Development commands
├── pyproject.toml          # Project configuration & dependencies
├── example.py              # Usage example
└── README.md               # This file
```

## 📖 Documentation

- 📋 **[API Setup Guide](docs/api_setup_guide.md)** - Configure OpenAI or Anthropic API keys
- 🧪 **[Testing Guide](docs/testing_guide.md)** - Comprehensive testing documentation  
- 👨‍💻 **[Developer Guide](docs/developer_guide.md)** - Complete development workflow
- ✅ **[Implementation Status](docs/implementation_checklist.md)** - Current project status
- 🏗️ **[Project Structure](docs/project_structure.md)** - Architecture and organization
- 🎯 **[Technical Architecture](docs/technical_architecture.md)** - System design
- 💡 **[Product Vision](docs/product_vision.md)** - Goals and objectives

## Contributing

We welcome contributions! The project is well-structured with comprehensive testing and documentation.

**Quick Start:**
```bash
git clone <repository-url>
cd metacontext
make dev-setup
make test  # Verify everything works
```

**Development Workflow:**
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes following our [Developer Guide](docs/developer_guide.md)
4. Add tests for your changes (maintain >90% coverage)
5. Run quality checks (`make ci`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)  
8. Open a Pull Request

**Development Commands:**
```bash
make help          # Show all available commands
make test          # Run tests (41 tests, 90% coverage)
make check         # Run all quality checks
make fix           # Auto-fix code issues
make ci            # Full CI pipeline locally
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🗺️ Roadmap: Establishing Metacontext as the New Metadata Standard

**Vision: Make metacontext files as ubiquitous as traditional file metadata - but infinitely more valuable.**

### 🎯 **Phase 1: Foundation (COMPLETE)**
- ✅ **Core Analysis Engine**: Full project context generation with 90% test coverage
- ✅ **LLM Integration**: OpenAI/Anthropic API support for intelligent context
- ✅ **Professional Quality**: Comprehensive testing, documentation, and development tools

### 🚧 **Phase 2: Wrapper Implementation (IN PROGRESS)**
**Core Goal: Automatic metacontext generation alongside every data export**

```python
import pandas as pd
import metacontext  # Automatically patches save functions

df.to_csv('results.csv')
# ✅ Auto-generates: results.csv.metacontext.yaml
```

**Target Integration:**
- [ ] **Pandas**: `to_csv()`, `to_pickle()`, `to_parquet()`, `to_excel()`, etc.
- [ ] **GeoPandas**: `to_file()` (.gpkg, .geojson, .shp, etc.)  
- [ ] **NumPy/SciPy**: `numpy.save()`, `scipy.io.savemat()`
- [ ] **General Python**: `pickle.dump()`, `joblib.dump()`

### 🚀 **Phase 3: Ecosystem Expansion**
- [ ] **Multi-Language Support**: R, Julia, MATLAB wrappers
- [ ] **IDE Integration**: VS Code, PyCharm, Jupyter extensions
- [ ] **CI/CD Integration**: Automatic context validation and updates
- [ ] **Cloud Platform Support**: AWS, GCP, Azure native integration

### 🌟 **Phase 4: Industry Standard**
- [ ] **Community Adoption**: Open-source ecosystem and plugin architecture
- [ ] **Enterprise Features**: Security, compliance, audit trails
- [ ] **Standards Compliance**: Integration with existing metadata standards
- [ ] **Universal Recognition**: Metacontext files recognized by all major data tools

### 📋 **Implementation Roadmap**
See detailed technical roadmap: **[Wrapper Implementation Roadmap](docs/wrapper_implementation_roadmap.md)**

### Current Development Priority
1. **🎯 Build pandas wrapper** - Most critical data export use case
2. **🧪 Proof of concept testing** - Validate automatic context generation
3. **📦 Core library integration** - Seamless import and patching system
4. **📈 Performance optimization** - Ensure minimal impact on existing workflows

**The goal is simple but revolutionary: Every data file should automatically have rich, intelligent context - making metacontext the new standard for data intelligence.**
