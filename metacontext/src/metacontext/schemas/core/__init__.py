"""Core schema system for metacontext.

This package provides the core schema classes and interfaces for the metacontext system.
"""

from metacontext.schemas.core.interfaces import (
    ConfidenceLevel,
    EnrichmentProvider,
    ExtensionProtocol,
    MetadataProvider,
    SchemaComponent,
    SchemaRegistry,
)

__all__ = [
    "ConfidenceLevel",
    "EnrichmentProvider",
    "ExtensionProtocol",
    "MetadataProvider",
    "SchemaComponent",
    "SchemaRegistry",
]
