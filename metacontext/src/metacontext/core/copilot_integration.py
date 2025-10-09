"""GitHub Copilot integration layer for metacontext generation.

This module manages the GitHub Copilot template generation and AI enrichment
process, ensuring consistent structure and schema compliance.
"""

import logging
from pathlib import Path
from typing import Any

import yaml

from metacontext.core.deterministic_utils import (
    create_base_structure,
    extract_deterministic_metadata,
)
from metacontext.handlers.base import BaseFileHandler

logger = logging.getLogger(__name__)


class CopilotTemplateEngine:
    """Manages GitHub Copilot template generation and AI enrichment."""

    def generate_template(
        self,
        handler: BaseFileHandler,
        file_path: Path,
        data_object: object = None,
    ) -> dict[str, Any]:
        """Generate template with deterministic data pre-populated.

        Args:
            handler: File handler instance for the specific file type
            file_path: Path to the file being analyzed
            data_object: Optional data object associated with the file

        Returns:
            Template structure with deterministic metadata populated

        """
        logger.info("🏗️ Generating GitHub Copilot template for %s", file_path.name)

        # Phase 1: Deterministic analysis (shared with API mode)
        deterministic_data = extract_deterministic_metadata(
            handler,
            file_path,
            data_object,
        )

        # Phase 2: Create base structure
        template = create_base_structure(deterministic_data, file_path)

        # Phase 3: Add handler-specific structure with AI enrichment placeholders
        if hasattr(handler, "create_copilot_template"):
            handler_template = handler.create_copilot_template(
                deterministic_data,
                file_path,
            )
            template.update(handler_template)
        else:
            logger.warning(
                "⚠️ Handler %s does not support GitHub Copilot templates",
                handler.__class__.__name__,
            )

        logger.info("✅ Template generated successfully")
        return template

    def create_copilot_prompt(
        self,
        handler: BaseFileHandler,
        template: dict[str, Any],
        file_path: Path,
    ) -> str:
        """Create GitHub Copilot prompt with template and context.

        Args:
            handler: File handler instance
            template: Template structure with deterministic data
            file_path: Path to the file being analyzed

        Returns:
            Formatted prompt for GitHub Copilot

        """
        logger.info(
            "📝 Creating GitHub Copilot prompt for %s",
            handler.__class__.__name__,
        )

        # Get handler-specific context if available
        context = ""
        if hasattr(handler, "get_copilot_prompt_context"):
            context = handler.get_copilot_prompt_context(template, file_path)
        else:
            context = f"File: {file_path.name}\nType: {handler.__class__.__name__}"

        # Create the structured prompt
        prompt = f"""
🔍 METACONTEXT AI ENRICHMENT TASK

{context}

🎯 CRITICAL INSTRUCTIONS:
1. Complete ONLY the AI enrichment sections in this metacontext template
2. DO NOT modify the deterministic_metadata sections - they are already populated
3. Fill in the ai_enrichment sections with your analysis
4. Maintain exact YAML structure and field names
5. Follow schema requirements exactly

📋 TEMPLATE STRUCTURE:
```yaml
{yaml.dump(template, default_flow_style=False, sort_keys=False)}
```

⚠️ VALIDATION REQUIREMENTS:
- Maintain exact YAML structure
- Fill only ai_enrichment sections
- Keep deterministic_metadata unchanged
- Follow schema requirements exactly
- Output valid YAML only

📁 Save your completed analysis as: {file_path.stem}_metacontext.yaml
"""

        logger.info("✅ Prompt created successfully")
        return prompt

    def save_template(
        self,
        template: dict[str, Any],
        file_path: Path,
    ) -> Path:
        """Save template to file for GitHub Copilot processing.

        Args:
            template: Template structure to save
            file_path: Original file path (template path will be derived)

        Returns:
            Path to the saved template file

        """
        template_file = file_path.with_suffix(
            f"{file_path.suffix}.copilot_template.yaml",
        )

        try:
            with open(template_file, "w", encoding="utf-8") as f:
                yaml.dump(template, f, default_flow_style=False, sort_keys=False)

            logger.info("✅ Template saved to %s", template_file)
            return template_file

        except (OSError, yaml.YAMLError) as e:
            logger.exception("❌ Failed to save template")
            raise RuntimeError(f"Template save failed: {e!s}") from e

    def validate_copilot_output(
        self,
        output_file: Path,
        expected_schema: type | None = None,
    ) -> bool:
        """Validate GitHub Copilot output for schema compliance.

        Args:
            output_file: Path to the GitHub Copilot output file
            expected_schema: Optional Pydantic schema to validate against

        Returns:
            True if validation passes, False otherwise

        """
        logger.info("🔍 Validating GitHub Copilot output: %s", output_file)

        try:
            with open(output_file, encoding="utf-8") as f:
                data = yaml.safe_load(f)

            # Basic structure validation
            required_top_level = [
                "metacontext_version",
                "generation_info",
                "file_info",
                "system_info",
                "confidence_assessment",
            ]

            for field in required_top_level:
                if field not in data:
                    logger.error("❌ Missing required field: %s", field)
                    return False

            # Check for deterministic metadata preservation
            if "data_structure" in data:
                if "deterministic_metadata" not in data["data_structure"]:
                    logger.error("❌ Missing deterministic_metadata section")
                    return False

            # Schema validation if provided
            if expected_schema:
                try:
                    expected_schema(**data)
                    logger.info("✅ Schema validation passed")
                except Exception as e:
                    logger.error("❌ Schema validation failed: %s", e)
                    return False

            logger.info("✅ Validation completed successfully")
            return True

        except (OSError, yaml.YAMLError):
            logger.exception("❌ Validation failed")
            return False


def generate_copilot_context(
    file_path: Path,
    handler: BaseFileHandler,
    data_object: object = None,
) -> dict[str, Any]:
    """Main entry point for GitHub Copilot context generation.

    This function orchestrates the entire GitHub Copilot integration process.

    Args:
        file_path: Path to the file being analyzed
        handler: File handler instance for the specific file type
        data_object: Optional data object associated with the file

    Returns:
        Template structure ready for GitHub Copilot processing

    """
    logger.info("🚀 Starting GitHub Copilot context generation for %s", file_path.name)

    template_engine = CopilotTemplateEngine()

    try:
        # Generate template with deterministic data
        template = template_engine.generate_template(handler, file_path, data_object)

        # Create and log the prompt (for debugging/development)
        prompt = template_engine.create_copilot_prompt(handler, template, file_path)
        logger.debug("📝 Generated prompt:\n%s", prompt)

        # Save template for GitHub Copilot processing
        template_file = template_engine.save_template(template, file_path)
        logger.info("📁 Template saved to: %s", template_file)

        logger.info("✅ GitHub Copilot context generation completed")
        return template

    except Exception as e:
        logger.exception("❌ GitHub Copilot context generation failed")
        raise RuntimeError(f"Copilot context generation failed: {e!s}") from e
