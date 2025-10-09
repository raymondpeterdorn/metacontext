"""Shared deterministic metadata utilities for API and GitHub Copilot integration.

This module provides reusable deterministic analysis functions that ensure
consistency between API mode and GitHub Copilot mode metacontext generation.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from metacontext.handlers.base import BaseFileHandler

logger = logging.getLogger(__name__)


def extract_deterministic_metadata(
    handler: BaseFileHandler,
    file_path: Path,
    data_object: object = None,
) -> dict[str, Any]:
    """Shared deterministic analysis between API and GitHub Copilot modes.

    This function ensures that both API and GitHub Copilot modes generate
    identical deterministic metadata, maintaining consistency across approaches.

    Args:
        handler: File handler instance for the specific file type
        file_path: Path to the file being analyzed
        data_object: Optional data object associated with the file

    Returns:
        Dictionary containing deterministic analysis results

    """

    try:
        # Use the handler's existing deterministic analysis
        deterministic_data = handler.analyze_deterministic(file_path, data_object)
    except (AttributeError, ValueError, TypeError, OSError) as e:
        logger.exception("❌ Deterministic analysis failed")
        # Return minimal fallback structure
        return {
            "type": "unknown",
            "error": f"Deterministic analysis failed: {e!s}",
            "fallback": True,
        }
    else:
        return deterministic_data


def create_base_structure(
    deterministic_data: dict[str, Any],
    file_path: Path,
    generation_method: str = "github_copilot_integration",
    file_purpose: str | None = None,
    project_context_summary: str | None = None,
) -> dict[str, Any]:
    """Create base metacontext structure with deterministic metadata populated.

    This creates the foundational metacontext structure that is shared between
    API and GitHub Copilot modes, with deterministic metadata pre-populated.

    Args:
        deterministic_data: Results from deterministic analysis
        file_path: Path to the file being analyzed
        generation_method: Method used for generation
        file_purpose: Optional description of file purpose
        project_context_summary: Optional project context

    Returns:
        Base metacontext structure with core fields populated

    """

    # Current timestamp
    now = datetime.now(UTC)

    # Create base structure matching existing schema
    base_structure = {
        "metacontext_version": "0.3.0",
        "generation_info": {
            "generated_at": now.isoformat(),
            "generation_method": generation_method,
            "function_call": "metacontext.metacontextualize()",
            "token_usage": {
                "total_tokens": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_api_calls": 0,
                "provider": "github_copilot"
                if "copilot" in generation_method
                else "api",
                "model": "gpt-4" if "copilot" in generation_method else None,
            },
        },
        "file_info": {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "file_purpose": file_purpose
            or "Generated file with deterministic analysis",
            "project_context_summary": project_context_summary
            or "Analysis with shared deterministic infrastructure",
            "creation_timestamp": now.isoformat(),
        },
        "system_info": {
            "working_directory": str(Path.cwd()),
        },
        "confidence_assessment": {
            "overall": "HIGH" if not deterministic_data.get("fallback") else "LOW",
        },
    }

    return base_structure


def validate_deterministic_consistency(
    api_data: dict[str, Any],
    copilot_data: dict[str, Any],
) -> bool:
    """Validate that API and GitHub Copilot modes produce consistent deterministic data.

    This is used for testing to ensure the shared infrastructure works correctly.

    Args:
        api_data: Deterministic data from API mode
        copilot_data: Deterministic data from GitHub Copilot mode

    Returns:
        True if data is consistent, False otherwise

    """
    # Define keys that should be identical between modes
    critical_keys = ["type", "shape", "memory_usage_bytes"]

    for key in critical_keys:
        if api_data.get(key) != copilot_data.get(key):
            logger.warning(
                "❌ Inconsistency detected in key '%s': API=%s, Copilot=%s",
                key,
                api_data.get(key),
                copilot_data.get(key),
            )
            return False

    return True
