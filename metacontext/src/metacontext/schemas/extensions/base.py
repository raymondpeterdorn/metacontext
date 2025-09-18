"""Base classes for extensions using the new architecture.

This module provides base classes for schema extensions, deriving from the
core interfaces to ensure proper dependency direction and avoid circular imports.
"""

from abc import ABC
from typing import ClassVar

from pydantic import BaseModel, Field

from metacontext.schemas.core.interfaces import (
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


class ForensicAIEnrichment(AIEnrichment, ABC):
    """Enhanced AI enrichment that forces forensic/investigative analysis.

    This base class adds fields that push LLMs to act like forensic investigators,
    digging deep into code context to find hidden meanings, suspicious patterns,
    and cross-references rather than just surface-level descriptions.
    """

    hidden_meaning: str | None = Field(
        default=None,
        description="Explanations, business logic, or context buried in code comments, variable names, or logic that clarify what this really represents beyond surface appearance.",
    )

    suspicious_patterns: list[str] | None = Field(
        default=None,
        description="List of oddities found: poorly named variables/columns, magic numbers, confusing transformations, misleading names, or hacky implementations.",
    )

    cross_references: dict[str, str] | None = Field(
        default=None,
        description="Mapping of confusing field/variable names to where in the codebase they are created, defined, calculated, or explained.",
    )

    detective_insights: str | None = Field(
        default=None,
        description="Reverse-engineered understanding of developer intent, business requirements, or domain knowledge that explains why this exists and how it connects to the larger system.",
    )


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
