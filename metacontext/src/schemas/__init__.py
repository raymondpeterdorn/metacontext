"""Metacontext schemas package.

This package provides the core schemas and extensions for the metacontext system.
Commonly used types are re-exported at the package level for convenience.
"""

from metacontext.schemas.core.codebase import CodebaseContext
from metacontext.schemas.core.core import (
    ConfidenceAssessment,
    FileInfo,
    GenerationInfo,
    GenerationMethod,
    Metacontext,
    SystemInfo,
    TokenUsage,
    create_base_metacontext,
)
from metacontext.schemas.core.interfaces import ConfidenceLevel
from metacontext.schemas.extensions.models import (
    ModelAIEnrichment,
    ModelContext,
    ModelDeterministicMetadata,
    TrainingData,
)
from metacontext.schemas.extensions.tabular import (
    ColumnAIEnrichment,
    ColumnDeterministicInfo,
    ColumnInfo,
    DataAIEnrichment,
    DataDeterministicMetadata,
    DataStructure,
)

__all__ = [
    # Core schemas
    "CodebaseContext",
    # Tabular extension
    "ColumnAIEnrichment",
    "ColumnDeterministicInfo",
    "ColumnInfo",
    "ConfidenceAssessment",
    # Base extensions
    "ConfidenceLevel",
    "DataAIEnrichment",
    "DataDeterministicMetadata",
    "DataStructure",
    "FileInfo",
    "GenerationInfo",
    "GenerationMethod",
    "Metacontext",
    # Model extension
    "ModelAIEnrichment",
    "ModelContext",
    "ModelDeterministicMetadata",
    "SystemInfo",
    "TokenUsage",
    "TrainingData",
    "create_base_metacontext",
]
