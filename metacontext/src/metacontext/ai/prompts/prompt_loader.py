"""Prompt loading and templating utilities."""

import importlib
import time
from pathlib import Path
from string import Template
from typing import Any, Optional

import yaml

from metacontext.ai.prompts.schema_utils import (
    compact_schema_hint,
    generate_prompt_from_schema,
)
from metacontext.ai.prompts.performance_monitor import PerformanceMonitor


class PromptLoader:
    """Loads and processes prompt templates from YAML files."""

    def __init__(self, prompts_dir: Path | None = None, enable_monitoring: bool = False) -> None:
        """Initialize the prompt loader.

        Args:
            prompts_dir: Directory containing prompt templates.
                        Defaults to 'prompts/templates' in the ai directory.
            enable_monitoring: Whether to enable performance monitoring

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
        
        # Initialize performance monitoring if enabled
        self.performance_monitor = None
        if enable_monitoring:
            metrics_file = current_dir / "metrics" / "prompt_performance.json"
            self.performance_monitor = PerformanceMonitor(metrics_file)

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

        # Construct the prompt path - always relative to prompts_dir
        prompt_path = self.prompts_dir / prompt_name

        if not prompt_path.exists():
            msg = f"Prompt file not found: {prompt_path}"
            raise FileNotFoundError(msg)

        with prompt_path.open(encoding="utf-8") as f:
            prompt_data: dict[str, Any] = yaml.safe_load(f) or {}

        return prompt_data

    def render_prompt(self, prompt_name: str, context: dict[str, Any]) -> str:
        """Load and render a prompt template with the given context, using dynamic structure from YAML.
        
        If the template contains a schema_class, automatically inject a compact schema hint.
        """
        prompt_data = self.load_prompt(prompt_name)

        # If schema-based, enhance context with schema hint and delegate to schema logic
        if "schema_class" in prompt_data:
            schema_class_path = prompt_data["schema_class"]
            system_message = prompt_data.get("system", "")
            instruction_template = prompt_data.get("instruction", "")
            
            # Import and generate schema hint
            schema_class = self._import_schema_class(schema_class_path)
            schema_hint = compact_schema_hint(schema_class)  # type: ignore[arg-type]

            # Add schema hint to context
            enhanced_context = {**context, "schema_hint": schema_hint}
            
            return self.load_schema_prompt(
                schema_class_path,
                system_message=system_message,
                instruction_template=instruction_template,
                **enhanced_context,
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

    def _import_schema_class(self, schema_class_path: str) -> object:
        """Import a schema class from its string path.

        Args:
            schema_class_path: String path to the Pydantic model class

        Returns:
            The imported schema class

        Raises:
            ImportError: If the schema class cannot be imported

        """
        schema_path = schema_class_path.split(".")
        module_path, class_name = ".".join(schema_path[:-1]), schema_path[-1]

        try:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)
        except (ImportError, AttributeError) as e:
            msg = f"Error importing schema class {schema_class_path}: {e}"
            raise ImportError(msg) from e

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
        """Generate a prompt directly from a schema class using compact schema hints.

        Args:
            schema_class_path: String path to the Pydantic model class
            system_message: Optional system message
            instruction_template: Optional instruction template
            **context: Additional context variables

        Returns:
            Rendered prompt string ready to send to LLM

        """
        # Import the schema class
        schema_class = self._import_schema_class(schema_class_path)

        # If schema_hint is not already in context, generate it
        enhanced_context = dict(context)
        if "schema_hint" not in enhanced_context:
            enhanced_context["schema_hint"] = compact_schema_hint(schema_class)  # type: ignore[arg-type]

        # Generate the prompt from the schema using compact hints
        prompt_data = generate_prompt_from_schema(
            schema_class,  # type: ignore[arg-type]
            system_message=system_message,
            instruction_template=instruction_template,
            extra_context=enhanced_context,
        )

        # Build the final prompt
        prompt_parts = []

        if prompt_data["system"]:
            prompt_parts.append(prompt_data["system"])

        if prompt_data["instruction"]:
            prompt_parts.append(prompt_data["instruction"])

        # Add context information (but skip schema_hint since it's already embedded)
        filtered_context = {k: v for k, v in enhanced_context.items() if k != "schema_hint"}
        if filtered_context:
            context_section = self._format_context(filtered_context)
            prompt_parts.append(context_section)

        # Note: We no longer add the full JSON schema since we use compact hints
        return "\n\n".join(prompt_parts)
