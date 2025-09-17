"""Prompt loading and templating utilities."""

import importlib
from pathlib import Path
from string import Template
from typing import Any

import yaml

from metacontext.ai.prompts.schema_utils import generate_prompt_from_schema


class PromptLoader:
    """Loads and processes prompt templates from YAML files."""

    def __init__(self, prompts_dir: Path | None = None) -> None:
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt templates.
                        Defaults to 'prompts/templates' in the ai directory.

        """
        if prompts_dir is None:
            # Try to find prompts directory relative to this file
            current_dir = Path(__file__).parent
            self.prompts_dir = current_dir / "templates"
        else:
            self.prompts_dir = prompts_dir

        if not self.prompts_dir.exists():
            msg = f"Prompts directory not found: {self.prompts_dir}"
            raise ValueError(msg)

    def load_prompt(self, prompt_name: str) -> dict[str, Any]:
        """Load a prompt template from a YAML file.

        Args:
            prompt_name: Name of the prompt file (with or without .yaml extension)
                       Can include subdirectories like "templates/tabular_column_analysis"

        Returns:
            Dictionary containing the prompt template structure

        Raises:
            FileNotFoundError: If the prompt file doesn't exist
            yaml.YAMLError: If the YAML is invalid

        """
        # Make sure we have a .yaml extension
        if not prompt_name.endswith(".yaml"):
            prompt_name += ".yaml"

        # Construct the prompt path
        if "/" in prompt_name or "\\" in prompt_name:
            # If prompt_name includes a path, use it relative to prompts_dir
            parts = Path(prompt_name).parts
            prompt_path = self.prompts_dir.parent / Path(*parts)
        else:
            # Otherwise look directly in prompts_dir
            prompt_path = self.prompts_dir / prompt_name

        if not prompt_path.exists():
            msg = f"Prompt file not found: {prompt_path}"
            raise FileNotFoundError(msg)

        with prompt_path.open(encoding="utf-8") as f:
            prompt_data: dict[str, Any] = yaml.safe_load(f) or {}

        return prompt_data

    def render_prompt(self, prompt_name: str, context: dict[str, Any]) -> str:
        """Load and render a prompt template with the given context, using dynamic structure from YAML."""
        prompt_data = self.load_prompt(prompt_name)

        # If schema-based, delegate to schema logic
        if "schema_class" in prompt_data:
            schema_class_path = prompt_data["schema_class"]
            system_message = prompt_data.get("system", "")
            instruction_template = prompt_data.get("instruction", "")
            return self.load_schema_prompt(
                schema_class_path,
                system_message=system_message,
                instruction_template=instruction_template,
                **context,
            )

        # Dynamically assemble prompt from YAML keys
        prompt_parts = []
        for key, value in prompt_data.items():
            # Skip keys that are not part of the prompt (e.g., metadata)
            if key in {"schema_class"}:
                continue
            # If value is a string, treat as a template
            if isinstance(value, str):
                rendered = Template(value).safe_substitute(context)
                if rendered:
                    prompt_parts.append(rendered)
            # If value is a dict or list, format as YAML for readability
            elif isinstance(value, (dict, list)):
                formatted = yaml.dump(value, default_flow_style=False, indent=2)
                prompt_parts.append(f"{key} (YAML):\n{formatted}")

        # Add context information if present
        if context:
            context_section = self._format_context(context)
            prompt_parts.append(context_section)

        return "\n\n".join(prompt_parts)

    def _format_context(self, context: dict[str, Any]) -> str:
        """Format context data for inclusion in the prompt."""
        context_lines = ["Context Information:"]

        for key, value in context.items():
            if isinstance(value, (dict, list)):
                # For complex objects, format as YAML for readability
                formatted_value = yaml.dump(value, default_flow_style=False, indent=2)
                context_lines.append(f"\n{key}:\n{formatted_value}")
            else:
                context_lines.append(f"{key}: {value}")

        return "\n".join(context_lines)

    def list_available_prompts(self) -> list[str]:
        """List all available prompt templates."""
        if not self.prompts_dir.exists():
            return []

        yaml_files = list(self.prompts_dir.glob("*.yaml"))
        return [f.stem for f in yaml_files]

    def load_schema_prompt(
        self,
        schema_class_path: str,
        system_message: str = "",
        instruction_template: str = "",
        **context: dict[str, Any],
    ) -> str:
        """Generate a prompt directly from a schema class.

        Args:
            schema_class_path: String path to the Pydantic model class (e.g., "metacontext.schemas.extensions.models.ModelAIEnrichment")
            system_message: Optional system message
            instruction_template: Optional instruction template
            **context: Additional context variables

        Returns:
            Rendered prompt string ready to send to LLM

        """
        # Import the schema class
        schema_path = schema_class_path.split(".")
        module_path, class_name = ".".join(schema_path[:-1]), schema_path[-1]

        try:
            module = importlib.import_module(module_path)
            schema_class = getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            msg = f"Error importing schema class {schema_class_path}: {e}"
            raise ImportError(msg) from e

        # Generate the prompt from the schema
        prompt_data = generate_prompt_from_schema(
            schema_class,
            system_message=system_message,
            instruction_template=instruction_template,
            extra_context=context,
        )

        # Build the final prompt
        prompt_parts = []

        if prompt_data["system"]:
            prompt_parts.append(prompt_data["system"])

        if prompt_data["instruction"]:
            prompt_parts.append(prompt_data["instruction"])

        # Add context information
        if context:
            context_section = self._format_context(context)
            prompt_parts.append(context_section)

        # Add schema
        if prompt_data["json_schema"]:
            prompt_parts.append(f"\nRequired JSON schema:\n{prompt_data['json_schema']}")

        return "\n\n".join(prompt_parts)
