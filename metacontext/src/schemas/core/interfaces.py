"""Core schema interfaces.

This module provides the base interfaces and abstract classes for the schema system,
separating core functionality from extensions to eliminate circular dependencies.
"""

from abc import ABC
from enum import Enum
from typing import Any, ClassVar, Protocol, TypeVar

from pydantic import BaseModel


class ConfidenceLevel(str, Enum):
    """Confidence levels for AI-generated content."""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


class SchemaComponent(Protocol):
    """Protocol for schema components that can be registered and retrieved."""

    schema_name: ClassVar[str]


class MetadataProvider(BaseModel, ABC):
    """Base class for deterministic metadata - facts extracted through code execution.

    Characteristics:
    - Always succeeds (no external dependencies)
    - 100% reliable and reproducible
    - Direct measurement/inspection only
    - Never contains interpretive analysis
    """


class EnrichmentProvider(BaseModel, ABC):
    """Base class for AI-generated enrichment - interpretive analysis via LLM.

    Standard fields for all AI analysis:
    - ai_interpretation: Comprehensive analysis
    - ai_confidence: Reliability indicator
    - ai_domain_context: Domain-specific context
    - usage_guidance: Practical recommendations
    """

    ai_interpretation: str | None = None
    ai_confidence: ConfidenceLevel | None = None
    ai_domain_context: str | None = None
    usage_guidance: str | None = None


class ExtensionProtocol(Protocol):
    """Protocol defining the extension interface for modular schema components."""

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get list of file extensions this schema component supports."""
        ...

    @classmethod
    def get_schema_name(cls) -> str:
        """Get the unique name of this schema component."""
        ...


# Type variable for extension registration
T = TypeVar("T")


class SchemaRegistry:
    """Registry for schema extensions with automatic detection and routing."""

    # Class variable to store registered extensions
    _extensions: ClassVar[dict[str, type[Any]]] = {}

    @classmethod
    def register(cls, extension_class: type[T]) -> type[T]:
        """Register a schema extension class.

        Args:
            extension_class: Extension class to register

        Returns:
            The extension class (for decorator support)

        """
        schema_name = getattr(extension_class, "schema_name", None)
        if schema_name and schema_name not in cls._extensions:
            cls._extensions[schema_name] = extension_class
        return extension_class

    @classmethod
    def get_extension(cls, schema_name: str) -> type[Any] | None:
        """Get a schema extension by name.

        Args:
            schema_name: Name of the schema extension

        Returns:
            Extension class if found, None otherwise

        """
        return cls._extensions.get(schema_name)

    @classmethod
    def get_extensions_for_file(cls, file_extension: str) -> list[str]:
        """Get extension names supported for a file type.

        Args:
            file_extension: File extension (e.g., '.csv', '.pkl')

        Returns:
            List of schema extension names that support this file type

        """
        supported_extensions = []
        for ext_name, ext_class in cls._extensions.items():
            if hasattr(ext_class, "get_supported_extensions") and file_extension in ext_class.get_supported_extensions():
                supported_extensions.append(ext_name)
        return supported_extensions
