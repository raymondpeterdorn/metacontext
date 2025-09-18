"""Utilities for generating prompts from Pydantic schemas."""

import json
import string
from typing import Any

from pydantic import BaseModel


def generate_field_descriptions(model_class: type[BaseModel]) -> str:
    """Generate a numbered list of field descriptions from a Pydantic model."""
    descriptions = []
    for i, (field_name, field) in enumerate(model_class.model_fields.items(), 1):
        description = field.description or f"The {field_name.replace('_', ' ')} of the item"
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
