"""Schema-driven template generator for GitHub Copilot integration.

This module generates template configurations directly from Pydantic schemas,
eliminating the need for hardcoded YAML templates and ensuring single source of truth.
"""

import logging
from pathlib import Path
from typing import Any, get_args, get_origin

import yaml
from pydantic import BaseModel

from metacontext.schemas.extensions.geospatial import (
    GeospatialRasterContext,
    GeospatialVectorContext,
)
from metacontext.schemas.extensions.tabular import DataAIEnrichment

logger = logging.getLogger(__name__)


class SchemaTemplateGenerator:
    """Generates GitHub Copilot templates directly from Pydantic schema definitions."""

    def __init__(self) -> None:
        """Initialize the schema template generator with type mappings."""
        self.type_mapping = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "list",
            dict: "object",
        }

    def extract_field_info(self, model: type[BaseModel]) -> dict[str, Any]:
        """Extract field information from a Pydantic model."""
        fields = {}

        # Constants for Union type checking
        union_arg_count = 2

        for field_name, field_info in model.model_fields.items():
            # Get the field type
            field_type = field_info.annotation

            # Handle Optional types (Union[T, None])
            if get_origin(field_type) is type(str | None):
                args = get_args(field_type)
                if len(args) == union_arg_count and type(None) in args:
                    field_type = next(arg for arg in args if arg is not type(None))

            # Handle list types
            if get_origin(field_type) is list:
                yaml_type = "list"
            elif get_origin(field_type) is dict:
                yaml_type = "object"
            else:
                yaml_type = self.type_mapping.get(field_type, "string")

            # Extract description from Field
            description = (
                getattr(field_info, "description", None)
                or f"AI analysis for {field_name}"
            )

            fields[field_name] = {
                "type": yaml_type,
                "description": description,
            }

        return fields

    def generate_template_config(
        self,
        ai_enrichment_model: type[BaseModel],
        root_key: str,
        deterministic_mapping: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """Generate complete template configuration from schema models."""
        # Extract AI enrichment fields
        ai_fields = self.extract_field_info(ai_enrichment_model)

        # Build template config
        return {
            "root_key": root_key,
            "deterministic_mapping": deterministic_mapping or {},
            "ai_enrichment": {
                "fields": ai_fields,
            },
        }

    def generate_csv_template(self) -> dict[str, Any]:
        """Generate CSV handler template from DataAIEnrichment schema."""
        return self.generate_template_config(
            ai_enrichment_model=DataAIEnrichment,
            root_key="data_structure",
            deterministic_mapping={
                "type": "type",
                "memory_usage_bytes": "memory_usage_bytes",
                "shape": "shape",
                "columns": "columns",
            },
        )

    def generate_geospatial_raster_template(self) -> dict[str, Any]:
        """Generate geospatial raster template from schema."""
        return self.generate_template_config(
            ai_enrichment_model=GeospatialRasterContext,  # Uses embedded AI enrichment
            root_key="geospatial_raster_context",
        )

    def generate_geospatial_vector_template(self) -> dict[str, Any]:
        """Generate geospatial vector template from schema."""
        return self.generate_template_config(
            ai_enrichment_model=GeospatialVectorContext,  # Uses embedded AI enrichment
            root_key="geospatial_vector_context",
        )

    def save_template(self, config: dict[str, Any], output_path: Path) -> None:
        """Save generated template configuration to YAML file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    def generate_all_templates(self, output_dir: Path) -> None:
        """Generate all template files from schemas."""
        templates = {
            "csvhandler_template.yaml": self.generate_csv_template(),
            "geospatial_raster_template.yaml": self.generate_geospatial_raster_template(),
            "geospatial_vector_template.yaml": self.generate_geospatial_vector_template(),
        }

        for filename, config in templates.items():
            self.save_template(config, output_dir / filename)


def main() -> None:
    """Generate all template files from Pydantic schemas."""
    generator = SchemaTemplateGenerator()

    # Generate templates in the templates directory
    templates_dir = Path(__file__).parent / "templates"
    generator.generate_all_templates(templates_dir)


if __name__ == "__main__":
    main()
