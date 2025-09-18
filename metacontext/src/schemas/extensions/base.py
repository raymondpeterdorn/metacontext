"""Base classes for extensions using the new architecture.

This module provides base classes for schema extensions, deriving from the
core interfaces to ensure proper dependency direction and avoid circular imports.
"""

from abc import ABC
from typing import ClassVar

from pydantic import BaseModel

from src.schemas.core.interfaces import (
    EnrichmentProvider,
    MetadataProvider,
)


class DeterministicMetadata(MetadataProvider, ABC):
    """Base class for deterministic metadata in extensions.

    Inherits from the core MetadataProvider interface.
    """


class AIEnrichment(EnrichmentProvider, ABC):
    """Base class for AI-generated enrichment in extensions.

    Inherits from the core EnrichmentProvider interface.
    """


class ExtensionContext(BaseModel, ABC):
    """Base class for all extension contexts with deterministic + AI pattern."""

    schema_name: ClassVar[str] = "extension_context"

    deterministic_metadata: DeterministicMetadata | None = None
    ai_enrichment: AIEnrichment | None = None

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of file extensions this schema component supports.

        Override this in subclasses to specify supported file extensions.

        Returns:
            List of supported file extensions

        """
        return []

    @classmethod
    def get_schema_name(cls) -> str:
        """Get the unique name of this schema component.

        Returns:
            Schema name

        """
        return cls.schema_name
