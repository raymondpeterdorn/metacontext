# Metacontext Implementation Guides

## API Setup Guide

### Installation

To get started with the Metacontext API:

```bash
pip install metacontext
```

### Configuration

Create a configuration file at `~/.metacontext/config.yaml`:

```yaml
llm_providers:
  openai:
    api_key: "your-openai-api-key-here"
    model: "gpt-4"
    temperature: 0.0
    
  anthropic:
    api_key: "your-anthropic-api-key-here"
    model: "claude-3-opus-20240229"
    temperature: 0.0
    
default_provider: "openai"
```

### Basic Usage

```python
from metacontext import metacontextualize

# Generate metadata for a file
metadata = metacontextualize("path/to/file.csv")

# Save metadata to a YAML file
metadata.save("path/to/output.yaml")

# Or access as a Python dict
data_dict = metadata.to_dict()
```

### Advanced Usage

```python
from metacontext import metacontextualize, CodebaseScanner

# Scan codebase for context
scanner = CodebaseScanner()
context = scanner.scan_for_file_context("data/birds.csv")

# Generate metadata with codebase context
metadata = metacontextualize(
    "data/birds.csv",
    codebase_context=context,
    llm_provider="anthropic",
    override_template="custom_template.yaml"
)
```

## Project Structure

The Metacontext project is organized as follows:

```
metacontext/
├── docs/                     # Documentation
├── examples/                 # Example usage and demo projects
├── src/                      # Source code
│   └── metacontext/          # Main package
│       ├── __init__.py       # Package initialization
│       ├── metacontextualize.py    # Main entry point
│       ├── ai/               # AI-related code
│       │   ├── __init__.py
│       │   ├── llm_handler.py   # LLM API integration
│       │   └── prompts/      # Prompt templates and utilities
│       ├── handlers/         # File type handlers
│       │   ├── base.py       # Base handler class
│       │   ├── tabular.py    # CSV/DataFrame handler
│       │   └── ...           # Other handlers
│       ├── schemas/          # Pydantic schemas
│       │   ├── core.py       # Core schema
│       │   └── extensions/   # File type-specific schemas
│       └── utils/            # Utility functions
├── tests/                    # Test suite
│   ├── __init__.py
│   └── test_*.py             # Test modules
├── pyproject.toml            # Project configuration
└── README.md                 # Project README
```

### Key Components

- **metacontextualize.py**: Main entry point that orchestrates the metadata generation process
- **ai/**: Contains all AI-related code, including LLM handlers and prompt templates
- **handlers/**: File type-specific handlers for processing different types of files
- **schemas/**: Pydantic schemas for validating metadata structure
- **utils/**: Utility functions used throughout the codebase

## LLM Provider Plugin Architecture

### Overview

Metacontext uses a plugin architecture for LLM providers, allowing easy integration with different LLM APIs:

- **Built-in Providers**: OpenAI, Anthropic, Azure OpenAI
- **Custom Providers**: Add your own provider implementations
- **Fallback Mechanism**: Configure backup providers for reliability

### Provider Interface

Each provider must implement the following interface:

```python
class BaseLLMProvider(ABC):
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> None:
        """Initialize the provider with configuration."""
        pass
        
    @abstractmethod
    async def generate_completion(
        self, 
        prompt: str, 
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate a completion from the given prompt."""
        pass
```

### Configuration

Configure providers in your `~/.metacontext/config.yaml`:

```yaml
llm_providers:
  openai:
    api_key: "your-openai-api-key-here"
    model: "gpt-4"
    temperature: 0.0
    
  anthropic:
    api_key: "your-anthropic-api-key-here"
    model: "claude-3-opus-20240229"
    temperature: 0.0
    
  custom_provider:
    provider_class: "mymodule.CustomProvider"
    api_key: "your-custom-api-key"
    other_param: "value"
    
default_provider: "openai"
fallback_providers: ["anthropic", "custom_provider"]
```

### Implementing a Custom Provider

Create a new provider by implementing the `BaseLLMProvider` interface:

```python
from metacontext.ai.providers import BaseLLMProvider

class CustomProvider(BaseLLMProvider):
    def initialize(self, config):
        self.api_key = config.get("api_key")
        self.other_param = config.get("other_param")
        # Setup your API client here
        
    async def generate_completion(self, prompt, temperature=None, max_tokens=None):
        # Implement API call logic here
        response = your_api_client.generate(
            prompt=prompt,
            temperature=temperature or 0.0,
            max_tokens=max_tokens or 1024
        )
        return response.text
```

### Using Providers

```python
from metacontext import metacontextualize

# Use default provider
metadata = metacontextualize("data/birds.csv")

# Specify provider
metadata = metacontextualize("data/birds.csv", llm_provider="anthropic")

# Or use the handler directly
from metacontext.ai import get_llm_handler

handler = get_llm_handler("custom_provider")
response = await handler.generate_completion("Hello, world!")
```