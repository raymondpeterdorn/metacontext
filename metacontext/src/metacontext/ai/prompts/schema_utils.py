"""Utilities for generating prompts from Pydantic schemas."""

import json
import string
from typing import Any

from pydantic import BaseModel

# Constants for description truncation
MAX_DESCRIPTION_LENGTH = 80
MAX_NESTED_DESCRIPTION_LENGTH = 60


def compact_schema_hint(schema_class: type[BaseModel]) -> str:
    """Generate a compact, low-token summary of a Pydantic schema.

    This function creates a minimal field hint that uses 80% fewer tokens
    than a full JSON schema while preserving essential structure information.

    Args:
        schema_class: The Pydantic model class to generate hints for

    Returns:
        A compact string listing field names, types, and brief descriptions

    Example:
        - project_purpose: string  # Summarized purpose of the project
        - architecture_summary: string  # Key design patterns and decisions

    """
    schema = schema_class.model_json_schema()
    fields = schema.get("properties", {})

    lines = []
    for field_name, field_data in fields.items():
        # Get description if available, truncate to MAX_DESCRIPTION_LENGTH chars max
        desc = field_data.get("description", "")
        if desc:
            desc = (
                desc[:MAX_DESCRIPTION_LENGTH] + "..."
                if len(desc) > MAX_DESCRIPTION_LENGTH
                else desc
            )

        # Get field type, defaulting to 'any' if not specified
        field_type = field_data.get("type", "any")

        # Handle array types
        if field_type == "array" and "items" in field_data:
            item_type = field_data["items"].get("type", "any")
            field_type = f"array[{item_type}]"

        # Compact line for readability + low token cost
        if desc:
            lines.append(f"- {field_name}: {field_type}  # {desc}")
        else:
            lines.append(f"- {field_name}: {field_type}")

    return "\n".join(lines)


def compact_schema_hint_nested(schema_class: type[BaseModel], depth: int = 0) -> str:
    """Generate a compact schema hint with support for nested objects.

    Args:
        schema_class: The Pydantic model class to generate hints for
        depth: Current nesting depth (for indentation)

    Returns:
        A compact string with nested object support

    """
    schema = schema_class.model_json_schema()
    fields = schema.get("properties", {})
    lines = []

    indent = "  " * depth
    for name, info in fields.items():
        field_type = info.get("type", "object")
        desc = info.get("description", "")
        if desc:
            desc = (
                desc[:MAX_DESCRIPTION_LENGTH] + "..."
                if len(desc) > MAX_DESCRIPTION_LENGTH
                else desc
            )

        if desc:
            lines.append(f"{indent}- {name}: {field_type}  # {desc}")
        else:
            lines.append(f"{indent}- {name}: {field_type}")

        # Handle nested objects
        if field_type == "object" and "properties" in info:
            nested_lines = []
            nested_indent = "  " * (depth + 1)
            for nested_name, nested_info in info["properties"].items():
                nested_type = nested_info.get("type", "any")
                nested_desc = nested_info.get("description", "")
                if nested_desc:
                    nested_desc = (
                        nested_desc[:MAX_NESTED_DESCRIPTION_LENGTH] + "..."
                        if len(nested_desc) > MAX_NESTED_DESCRIPTION_LENGTH
                        else nested_desc
                    )
                    nested_lines.append(
                        f"{nested_indent}- {nested_name}: {nested_type}  # {nested_desc}",
                    )
                else:
                    nested_lines.append(
                        f"{nested_indent}- {nested_name}: {nested_type}",
                    )

            if nested_lines:
                lines.extend(nested_lines)

    return "\n".join(lines)


def generate_field_descriptions(model_class: type[BaseModel]) -> str:
    """Generate a numbered list of field descriptions from a Pydantic model."""
    descriptions = []
    for i, (field_name, field) in enumerate(model_class.model_fields.items(), 1):
        description = (
            field.description or f"The {field_name.replace('_', ' ')} of the item"
        )
        descriptions.append(f"{i}. {field_name}: {description}")

    return "\n".join(descriptions)


def generate_json_schema(model_class: type[BaseModel]) -> str:
    """Generate a JSON schema string from a Pydantic model."""
    schema = model_class.model_json_schema()
    # Remove the schema definition and title for cleaner output
    if "$schema" in schema:
        del schema["$schema"]
    if "title" in schema:
        del schema["title"]

    return json.dumps(schema, indent=2)


def generate_prompt_from_schema(
    model_class: type[BaseModel],
    system_message: str = "",
    instruction_template: str = "",
    extra_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    """Generate a complete prompt template from a Pydantic model.

    Args:
        model_class: The Pydantic model class to use for schema generation
        system_message: Optional custom system message
        instruction_template: Optional instruction template with ${field_descriptions} placeholder
        extra_context: Additional context variables for the instruction template

    Returns:
        A dictionary with system, instruction, and json_schema keys

    """
    field_descriptions = generate_field_descriptions(model_class)
    schema = generate_json_schema(model_class)

    # Default system message if none provided
    if not system_message:
        system_message = (
            "You are an AI assistant tasked with analyzing data and providing "
            "structured information about it according to a specific schema."
        )

    # Default instruction template if none provided
    if not instruction_template:
        instruction_template = (
            "Analyze the following information and provide details about:\n"
            "${field_descriptions}\n\n"
            "Respond with a properly structured JSON object."
        )

    # Prepare context for template substitution
    context = {"field_descriptions": field_descriptions}
    if extra_context:
        context.update(extra_context)

    # Substitute variables in the instruction template
    instruction = string.Template(instruction_template).safe_substitute(**context)

    return {
        "system": system_message,
        "instruction": instruction,
        "json_schema": schema,
    }
