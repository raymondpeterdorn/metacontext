# Metacontext Prompt Templates

This directory contains prompt templates used by the Metacontext system for AI analysis.

## Directory Structure

- `general/`: General-purpose templates used across multiple domains
- `tabular/`: Prompts for analyzing tabular data (CSV, Excel, etc.)
- `model/`: Prompts for analyzing ML models and their attributes
- `code/`: Prompts for analyzing code and repository structure
- `media/`: Prompts for analyzing media files (images, audio, video) - future

## How to Use

Prompts are loaded using the `PromptLoader` class:

```python
from metacontext.ai.prompts.prompt_loader import PromptLoader

# Initialize the prompt loader
prompt_loader = PromptLoader()

# Load and render a prompt with context
context = {
    "file_name": "my_data.csv",
    "columns_data": "...",
    # Other variables used in the template
}
prompt = prompt_loader.render_prompt("templates/tabular/column_analysis", context)
```

## Template Format

Templates are defined in YAML files with the following structure:

```yaml
system: |
  System instructions for the AI model (optional)

instruction: |
  Main instruction text with ${variable} placeholders

json_schema: |
  JSON schema for the expected response format (optional)
```

Variables in templates use the `${variable_name}` syntax and are replaced with 
values from the context dictionary when rendered.