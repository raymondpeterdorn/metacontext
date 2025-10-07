"""Base file handler class for universal file intelligence.

This module defines the abstract base class for file handlers.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from metacontext.core.registry import HandlerRegistry
from metacontext.schemas.core.core import Metacontext, create_base_metacontext

logger = logging.getLogger(__name__)


class AnalysisDepth(Enum):
    """Analysis depth options for handlers."""

    DETERMINISTIC_ONLY = "deterministic_only"
    INCLUDE_DEEP = "include_deep"


@dataclass
class MetacontextArgs:
    """Arguments for creating metacontext."""

    data_object: object = None
    codebase_context: dict[str, object] | None = None
    ai_companion: object | None = None
    file_purpose: str | None = None
    project_context_summary: str | None = None


class BaseFileHandler(ABC):
    """Abstract base class for file type handlers.

    Each handler is responsible for:
    1. Fast probing to check file compatibility (fast_probe)
    2. Deterministic analysis without AI (analyze_deterministic)
    3. Deep analysis with AI if enabled (analyze_deep)
    4. Providing bulk prompts for AI analysis
    """

    # Subclasses should override these
    supported_extensions: ClassVar[list[str]] = []
    required_schema_extensions: ClassVar[list[str]] = []
    PROMPT_CONFIG: ClassVar[dict[str, str]] = {}

    @abstractmethod
    def fast_probe(self, file_path: Path) -> dict[str, object]:
        """Fast probe to check file compatibility and get basic metadata.

        This method should be cheap (<50ms) and only check:
        - File extension match
        - Basic file existence/readability
        - MIME type detection
        - File size

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with basic file metadata:
            {
                "can_handle": bool,
                "file_size": int,
                "mime_type": str,
                "extension": str,
            }

        """

    @abstractmethod
    def analyze_deterministic(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> dict[str, object]:
        """Analyze file without AI - deterministic analysis only.

        This includes schema detection, basic statistics, structure analysis,
        but NO AI prompts or heavy computation.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file

        Returns:
            Dictionary with deterministic analysis results

        """

    @abstractmethod
    def analyze_deep(
        self,
        file_path: Path,
        data_object: object = None,
        ai_companion: object | None = None,
        deterministic_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Deep analysis using AI and heavy computation.

        This method can use AI prompts, training script inspection,
        complex pattern analysis, etc.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file
            ai_companion: AI companion for intelligent analysis
            deterministic_context: Results from analyze_deterministic

        Returns:
            Dictionary with deep analysis results

        """

    @abstractmethod
    def can_handle(self, file_path: Path, data_object: object = None) -> bool:
        """Determine if this handler can process the given file.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file

        Returns:
            True if this handler can process the file

        """

    @abstractmethod
    def get_required_extensions(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> list[str]:
        """Get the schema extensions required for this file type.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file

        Returns:
            List of extension names (e.g., ['data_structure', 'model_context'])

        """

    def generate_context(
        self,
        file_path: Path,
        data_object: object = None,
        ai_companion: object | None = None,
        analysis_depth: AnalysisDepth = AnalysisDepth.INCLUDE_DEEP,
    ) -> dict[str, Any]:
        """Generate file-type-specific context using the new lifecycle.

        This method now orchestrates the analysis phases:
        1. Fast probe (always)
        2. Deterministic analysis (always)
        3. Deep analysis (if enabled)

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file
            ai_companion: AI companion for intelligent analysis
            analysis_depth: How deep to analyze (deterministic only or include AI)

        Returns:
            Dictionary containing extension-specific context

        """
        # Phase 1: Fast probe (always required)
        probe_result = self.fast_probe(file_path)
        if not probe_result.get("can_handle", False):
            return {"error": "Handler cannot process this file"}

        # Phase 2: Deterministic analysis (always)
        deterministic_result = self.analyze_deterministic(file_path, data_object)

        # Combine results so far
        combined_context: dict[str, Any] = {
            "probe": probe_result,
            "deterministic": deterministic_result,
        }

        # Phase 3: Deep analysis (optional with graceful fallback)
        if analysis_depth == AnalysisDepth.INCLUDE_DEEP and ai_companion:
            try:
                deep_result = self.analyze_deep(
                    file_path=file_path,
                    data_object=data_object,
                    ai_companion=ai_companion,
                    deterministic_context=deterministic_result,
                )
                combined_context["deep"] = deep_result
            except (RuntimeError, ValueError, ImportError, AttributeError) as e:
                # Graceful fallback: log error but continue with deterministic analysis
                logger.warning(
                    "AI analysis failed for %s, continuing with deterministic analysis: %s",
                    file_path.name,
                    e,
                )
                combined_context["ai_analysis_error"] = str(e)
                combined_context["analysis_mode"] = "deterministic_only"
        else:
            combined_context["analysis_mode"] = "deterministic_only"

        return combined_context

    def get_bulk_prompts(
        self,
        file_path: Path,  # noqa: ARG002
        data_object: object = None,  # noqa: ARG002
    ) -> dict[str, str]:
        """Get bulk prompts for AI analysis of this file type.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file

        Returns:
            Dictionary mapping prompt names to prompt template paths

        Recommended pattern: Subclasses should set PROMPT_CONFIG as a class attribute
        mapping prompt types to template/config paths, e.g.:

            PROMPT_CONFIG: ClassVar[dict[str, str]] = {
                "column_analysis": "templates/tabular/column_analysis.yaml",
                "schema_interpretation": "templates/tabular/schema_analysis.yaml",
            }

        """
        return self.PROMPT_CONFIG.copy() if hasattr(self, "PROMPT_CONFIG") else {}

    def create_metacontext(
        self,
        file_path: Path,
        args: MetacontextArgs,
    ) -> Metacontext:
        """Create a complete metacontext for this file.

        Args:
            file_path: Path to the file
            args: Dataclass containing all optional arguments.

        Returns:
            Complete Metacontext instance

        """
        # Create base metacontext with core fields
        metacontext = create_base_metacontext(
            filename=str(file_path),
            file_purpose=args.file_purpose,
            project_context_summary=args.project_context_summary,
        )

        # Generate file-type-specific context
        extension_context = self.generate_context(
            file_path=file_path,
            data_object=args.data_object,
            ai_companion=args.ai_companion,
        )

        # Add extension context to metacontext
        for extension_name, extension_data in extension_context.items():
            if hasattr(metacontext, extension_name):
                setattr(metacontext, extension_name, extension_data)

        return metacontext


def register_handler(handler_class: type[BaseFileHandler]) -> type[BaseFileHandler]:
    """Register a file handler class.

    Args:
        handler_class: Handler class to register

    Returns:
        The handler class (for use as decorator)

    """
    return HandlerRegistry.register(handler_class)


# Utility function for easy access
def metacontextualize_universal(
    file_path: Path,
    args: MetacontextArgs,
) -> Metacontext | None:
    """Universal metacontextualize function that works with any file type.

    Args:
        file_path: Path to the file
        args: Dataclass containing all optional arguments.

    Returns:
        Metacontext instance if a handler is found, otherwise None

    """
    handler = HandlerRegistry.get_handler(file_path, args.data_object)
    if handler:
        return handler.create_metacontext(file_path, args)
    return None
