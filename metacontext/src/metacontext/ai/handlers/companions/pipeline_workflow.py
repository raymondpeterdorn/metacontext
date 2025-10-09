"""Enhanced companion workflow using the standard Pydantic-to-YAML pipeline.

This module implements a new approach for companion integration that:
1. Uses the same Pydantic validation pipeline as the main workflow
2. Generates complete YAML structures with null AI enrichment fields
3. Asks companions to fill in the missing values
4. Validates and merges the response back

This eliminates template duplication while maintaining type safety.
"""

import logging
import tempfile
from pathlib import Path

from metacontext.schemas.core.core import (
    ConfidenceAssessment,
    create_base_metacontext,
)
from metacontext.schemas.core.interfaces import ConfidenceLevel

logger = logging.getLogger(__name__)


def generate_companion_context_with_pipeline(
    file_path: Path,
    data_object: object,
    handler: object,
    ai_companion: object,
    *,
    verbose: bool = False,
) -> dict:
    """Generate companion context using the standard Pydantic-to-YAML pipeline.

    This approach:
    1. Runs deterministic analysis to get baseline metadata
    2. Creates a complete Pydantic model with null AI enrichment fields
    3. Converts to YAML using the standard pipeline
    4. Includes that YAML in a prompt asking companion to fill nulls
    5. Parses response and validates with Pydantic

    Args:
        file_path: Path to the file being analyzed
        data_object: The data object (DataFrame, etc.)
        handler: File handler instance
        ai_companion: Companion provider
        verbose: Enable verbose logging

    Returns:
        Dictionary containing validated context ready for YAML output

    """
    try:

        deterministic_result = handler.analyze_deterministic(file_path, data_object)

        # Phase 2: Create base metacontext with deterministic data
        base_context = create_base_metacontext(
            filename=file_path.name,
            file_purpose="Generated with companion-enhanced pipeline",
            project_context_summary=f"Analysis with {handler.__class__.__name__}",
        )

        # Phase 3: Populate deterministic extension data
        extension_data = _create_extension_with_null_ai_enrichment(
            handler,
            deterministic_result,
        )

        # Add extension to base context
        for key, value in extension_data.items():
            if hasattr(base_context, key):
                setattr(base_context, key, value)

        # Phase 4: Convert to YAML using standard pipeline
        partial_yaml = base_context.model_dump(
            mode="json",
            by_alias=True,
            exclude_none=False,  # Keep nulls for companion
        )

        # Phase 5: Create companion prompt with the YAML
        companion_prompt = _create_pipeline_companion_prompt(
            file_path,
            partial_yaml,
            handler.__class__.__name__,
        )

        # Phase 6: Send to companion
        # Create response file
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
            response_file_path = Path(tmp_file.name)

        # Get companion response
        companion_response = ai_companion.display_prompt_and_wait(
            companion_prompt,
            response_file_path,
        )

        if companion_response:
            # Phase 7: Validate and merge response
            enhanced_context = _merge_companion_response(
                base_context,
                companion_response,
                verbose,
            )

            # Phase 8: Final validation and confidence assessment
            enhanced_context.confidence_assessment = ConfidenceAssessment(
                overall=ConfidenceLevel("HIGH"),  # Companion + deterministic
            )

            return enhanced_context.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
            )
        if verbose:
            logger.warning("⚠️ No companion response, using deterministic only")
        return partial_yaml

    except Exception:
        logger.exception("❌ Pipeline companion workflow failed")
        # Return deterministic data as fallback
        return base_context.model_dump(mode="json", by_alias=True, exclude_none=True)


def _create_extension_with_null_ai_enrichment(
    handler: object,
    deterministic_result: dict,
) -> dict:
    """Create extension data with deterministic metadata and null AI enrichment.

    This creates the appropriate extension structure (DataStructure, ModelContext, etc.)
    with real deterministic data but null/placeholder AI enrichment fields.

    Args:
        handler: File handler instance
        deterministic_result: Results from deterministic analysis

    Returns:
        Dictionary with extension data ready for Pydantic validation

    """
    # Import here to avoid circular imports
    from metacontext.schemas.extensions.geospatial import (
        GeospatialRasterContext,
        GeospatialVectorContext,
    )
    from metacontext.schemas.extensions.media import MediaContext
    from metacontext.schemas.extensions.models import ModelContext
    from metacontext.schemas.extensions.tabular import DataStructure

    handler_name = handler.__class__.__name__

    if handler_name == "CSVHandler":
        # Create DataStructure with deterministic metadata and null AI enrichment
        return {
            "data_structure": DataStructure(
                deterministic_metadata=deterministic_result,
                ai_enrichment=None,  # Will be filled by companion
            ),
        }
    if handler_name == "ModelHandler":
        return {
            "model_context": ModelContext(
                deterministic_metadata=deterministic_result,
                ai_enrichment=None,  # Will be filled by companion
            ),
        }
    if handler_name == "MediaHandler":
        return {
            "media_context": MediaContext(
                deterministic_metadata=deterministic_result,
                ai_enrichment=None,  # Will be filled by companion
            ),
        }
    if handler_name == "GeospatialHandler":
        # Determine vector vs raster based on deterministic result
        if "geometry_type" in deterministic_result:
            return {
                "geospatial_vector_context": GeospatialVectorContext(
                    deterministic_metadata=deterministic_result,
                    ai_enrichment=None,  # Will be filled by companion
                ),
            }
        return {
            "geospatial_raster_context": GeospatialRasterContext(
                deterministic_metadata=deterministic_result,
                ai_enrichment=None,  # Will be filled by companion
            ),
        }
    # Fallback for unknown handlers
    return {
        "data_structure": DataStructure(
            deterministic_metadata=deterministic_result,
            ai_enrichment=None,
        ),
    }


def _create_pipeline_companion_prompt(
    file_path: Path,
    partial_yaml: dict,
    handler_name: str,
) -> str:
    """Create a companion prompt that includes the partial YAML structure.

    Args:
        file_path: Path to the file being analyzed
        partial_yaml: YAML structure with null AI enrichment fields
        handler_name: Name of the handler for context

    Returns:
        Complete prompt for the companion

    """
    import yaml

    # Convert dict back to YAML string for the prompt
    yaml_str = yaml.dump(
        partial_yaml,
        default_flow_style=False,
        sort_keys=False,
        indent=2,
        allow_unicode=True,
    )

    return f"""# Metacontext AI Enrichment Task

You are analyzing the file: `{file_path.name}`
Handler: {handler_name}

## Task
Below is a complete metacontext structure with deterministic metadata already populated.
Your job is to **fill in the null AI enrichment fields** with intelligent analysis.

## Instructions
1. **Preserve all existing structure** - only replace `null` values in AI enrichment sections
2. **Keep all deterministic metadata unchanged** - these are factual measurements
3. **Focus on semantic interpretation** - what does the data mean, not just what it contains
4. **Provide confidence assessments** - HIGH/MEDIUM/LOW based on certainty
5. **Return the complete YAML** - entire structure with your additions

## Current Structure (fill in null AI enrichment fields):

```yaml
{yaml_str}
```

## Key AI Enrichment Fields to Complete:
- `ai_enrichment` sections (wherever they appear as `null`)
- Semantic interpretations and domain context
- Quality assessments and recommendations
- Business value and usage guidance
- Relationships and patterns
- Confidence levels for your assessments

## Output Requirements:
- Return the **complete YAML structure** with AI enrichment fields filled
- Maintain exact field names and overall structure
- Use appropriate data types (strings, lists, dicts)
- Be specific and actionable in your analysis
- Focus on insights that aren't obvious from the raw data

**Important**: Save your response as `{file_path.stem}_metacontext.yaml` when complete.
"""


def _merge_companion_response(
    base_context: object,
    companion_response: dict,
    verbose: bool = False,
) -> object:
    """Merge companion response back into the base Pydantic model.

    Args:
        base_context: Original Pydantic model with deterministic data
        companion_response: Dictionary from companion with AI enrichment
        verbose: Enable verbose logging

    Returns:
        Updated Pydantic model with companion enhancements

    """
    try:
        # The companion should return a complete structure - we can validate it
        # directly by creating a new instance from the response

        # Import the core model for validation
        from metacontext.schemas.core.core import Metacontext

        # Create new validated instance from companion response
        enhanced_context = Metacontext.model_validate(companion_response)

        return enhanced_context

    except Exception as e:
        if verbose:
            logger.warning("⚠️ Companion response validation failed: %s", e)
            logger.info("🔄 Falling back to deterministic context")

        # Return original context if companion response is invalid
        return base_context
