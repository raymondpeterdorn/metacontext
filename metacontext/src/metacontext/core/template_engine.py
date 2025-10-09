"""Template configuration system for GitHub Copilot integration.

This module provides a modular, configuration-driven approach to generating
GitHub Copilot templates for different file types without hard-coding.
Now uses schema-driven template generation to eliminate YAML duplication.
"""

import logging
from pathlib import Path
from typing import Any, Protocol

from metacontext.core.schema_template_generator import SchemaTemplateGenerator

logger = logging.getLogger(__name__)


class TemplateConfig(Protocol):
    """Protocol for template configuration providers."""

    def get_ai_enrichment_fields(self) -> dict[str, Any]:
        """Get the AI enrichment field structure for this file type."""
        ...

    def get_prompt_components(self) -> dict[str, str]:
        """Get prompt component templates for this file type."""
        ...

    def get_schema_requirements(self) -> dict[str, Any]:
        """Get schema validation requirements."""
        ...


class ConfigurableTemplateEngine:
    """Template engine that uses configuration instead of hard-coding."""

    def __init__(self, config_dir: Path | None = None) -> None:
        """Initialize with optional config directory and schema generator."""
        self.config_dir = config_dir or Path(__file__).parent / "templates"
        self._template_cache: dict[str, dict[str, Any]] = {}
        self._schema_generator = SchemaTemplateGenerator()

    def load_template_config(self, handler_name: str) -> dict[str, Any]:
        """Load template configuration for a specific handler.

        Now generates templates from schemas instead of loading hardcoded YAML files.

        Args:
            handler_name: Name of the handler (e.g., 'csvhandler', 'geospatial_raster')

        Returns:
            Template configuration dictionary generated from Pydantic schemas

        """
        if handler_name in self._template_cache:
            return self._template_cache[handler_name]

        # Generate template from schema instead of loading static YAML
        try:
            if handler_name.lower() == "csvhandler":
                config = self._schema_generator.generate_csv_template()
            elif handler_name.lower() == "geospatial_raster":
                config = self._schema_generator.generate_geospatial_raster_template()
            elif handler_name.lower() == "geospatial_vector":
                config = self._schema_generator.generate_geospatial_vector_template()
            else:
                logger.warning(
                    "Unknown handler %s, using default template",
                    handler_name,
                )
                config = self._get_default_template_config()

            # Cache the generated config
            self._template_cache[handler_name] = config
            return config

        except Exception:
            logger.exception("Failed to generate template config for %s", handler_name)
            return self._get_default_template_config()

    def generate_template_structure(
        self,
        handler_name: str,
        deterministic_data: dict[str, Any],
        schema_class: type | None = None,
    ) -> dict[str, Any]:
        """Generate template structure using configuration.

        Args:
            handler_name: Name of the handler
            deterministic_data: Results from deterministic analysis
            schema_class: Optional schema class for validation

        Returns:
            Template structure with AI enrichment placeholders

        """
        config = self.load_template_config(handler_name)

        # Generate deterministic metadata structure
        deterministic_structure = self._create_deterministic_structure(
            deterministic_data,
            config.get("deterministic_mapping", {}),
        )

        # Generate AI enrichment structure from config
        ai_enrichment_structure = self._create_ai_enrichment_structure(
            deterministic_data,
            config.get("ai_enrichment", {}),
        )

        # Combine into final template
        template = {
            config.get("root_key", "data_structure"): {
                "deterministic_metadata": deterministic_structure,
                "ai_enrichment": ai_enrichment_structure,
            },
        }

        return template

    def generate_prompt_context(
        self,
        handler_name: str,
        template: dict[str, Any],
        file_path: Path,
    ) -> str:
        """Generate prompt context using configuration.

        Args:
            handler_name: Name of the handler
            template: Template structure with deterministic data
            file_path: Path to the file being analyzed

        Returns:
            Context string for GitHub Copilot prompts

        """
        config = self.load_template_config(handler_name)
        prompt_config = config.get("prompt", {})

        # Extract data for prompt variables
        prompt_variables = self._extract_prompt_variables(
            template,
            file_path,
            prompt_config,
            handler_name,
        )

        # Load and format prompt template
        prompt_template = prompt_config.get(
            "template",
            self._get_default_prompt_template(),
        )

        try:
            return prompt_template.format(**prompt_variables)
        except KeyError as e:
            logger.warning("Missing prompt variable %s, using fallback", e)
            return self._get_fallback_prompt(handler_name, file_path)

    def _create_deterministic_structure(
        self,
        deterministic_data: dict[str, Any],
        mapping_config: dict[str, str],
    ) -> dict[str, Any]:
        """Create deterministic structure using field mapping config."""
        if not mapping_config:
            return deterministic_data

        mapped_structure = {}
        for config_field, data_field in mapping_config.items():
            if data_field in deterministic_data:
                mapped_structure[config_field] = deterministic_data[data_field]

        return mapped_structure

    def _create_ai_enrichment_structure(
        self,
        deterministic_data: dict[str, Any],
        ai_config: dict[str, Any],
    ) -> dict[str, Any]:
        """Create AI enrichment structure using configuration."""
        structure = {}

        # Add top-level AI fields
        for field_name, field_config in ai_config.get("fields", {}).items():
            if field_config.get("type") == "string":
                structure[field_name] = ""
            elif field_config.get("type") == "list":
                structure[field_name] = []
            elif field_config.get("type") == "dict":
                structure[field_name] = {}

        # Add column interpretations if this is tabular data
        if "columns" in deterministic_data and ai_config.get(
            "include_column_analysis",
            False,
        ):
            column_template = ai_config.get("column_template", {})
            columns_dict = deterministic_data["columns"]
            structure["column_interpretations"] = {
                col_name: self._create_column_structure(column_template)
                for col_name in columns_dict
            }

        return structure

    def _create_column_structure(
        self,
        column_template: dict[str, Any],
    ) -> dict[str, Any]:
        """Create column structure from template configuration."""
        structure = {"deterministic": {}}

        ai_fields = {}
        for field_name, field_config in column_template.items():
            if field_config.get("type") == "string":
                ai_fields[field_name] = ""
            elif field_config.get("type") == "list":
                ai_fields[field_name] = []

        structure["ai_enrichment"] = ai_fields
        return structure

    def _extract_prompt_variables(
        self,
        template: dict[str, Any],
        file_path: Path,
        prompt_config: dict[str, Any],
        handler_name: str,
    ) -> dict[str, str]:
        """Extract variables for prompt template."""
        variables = {
            "filename": file_path.name,
            "handler_name": handler_name,
        }

        # Extract deterministic metadata (look for the configured root key)
        config = self.load_template_config(handler_name)
        root_key = config.get("root_key", "data_structure")

        # Try the configured root key first, fall back to "data_structure"
        structure_data = template.get(root_key, template.get("data_structure", {}))
        deterministic = structure_data.get("deterministic_metadata", {})

        variables.update(
            {
                "data_type": deterministic.get("type", "unknown"),
                "file_size": str(deterministic.get("memory_usage_bytes", 0)),
            },
        )

        # Extract shape information if available
        shape = deterministic.get("shape", [])
        if len(shape) >= 2:
            variables.update(
                {
                    "rows": str(shape[0]),
                    "columns": str(shape[1]),
                },
            )

        # Extract column information if available
        ai_enrichment = structure_data.get("ai_enrichment", {})
        column_interp = ai_enrichment.get("column_interpretations", {})
        if column_interp:
            column_names = list(column_interp.keys())
            variables["column_list"] = "\n".join(
                f"  - {col}" for col in column_names[:10]
            )
            if len(column_names) > 10:
                variables["column_list"] += "\n  - ... and more"

        return variables

    def _get_default_template_config(self) -> dict[str, Any]:
        """Get default template configuration."""
        return {
            "root_key": "data_structure",
            "deterministic_mapping": {},
            "ai_enrichment": {
                "fields": {
                    "domain_analysis": {"type": "string"},
                    "data_quality_assessment": {"type": "string"},
                    "business_value_assessment": {"type": "string"},
                },
            },
            "prompt": {"template": self._get_default_prompt_template()},
        }

    def _get_default_prompt_template(self) -> str:
        """Get default prompt template."""
        return """
🔍 DATA ANALYSIS - {filename}

📊 File Overview:
- Type: {data_type}
- Handler: {handler_name}

🎯 Your Task: Analyze this data and provide insights for the AI enrichment sections.

Please maintain the exact YAML structure provided in the template.
        """.strip()

    def _get_fallback_prompt(self, handler_name: str, file_path: Path) -> str:
        """Get fallback prompt when template formatting fails."""
        return f"""
🔍 DATA ANALYSIS - {file_path.name}

📊 Handler: {handler_name}

🎯 Please analyze this file and provide insights for the AI enrichment sections.
        """.strip()
