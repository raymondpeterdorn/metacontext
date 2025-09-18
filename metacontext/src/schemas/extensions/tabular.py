"""Tabular data extension schemas."""

from typing import Any

from pydantic import BaseModel, Field

from src.schemas.extensions.base import (
    AIEnrichment,
    DeterministicMetadata,
    ExtensionContext,
)

# ========== DATA STRUCTURE EXTENSION ==========


class ColumnDeterministicInfo(DeterministicMetadata):
    """Deterministic facts about a column extracted through code execution."""

    dtype: str | None = None
    null_count: int | None = None
    null_percentage: float | None = None
    unique_count: int | None = None
    sample_values: list[Any] | None = None


class ColumnAIEnrichment(AIEnrichment):
    """AI-generated insights about a column."""

    semantic_meaning: str | None = Field(
        default=None,
        description="What this column represents, including business meaning and units if applicable.",
    )
    data_quality_assessment: str | None = Field(
        default=None,
        description="Assessment of data quality issues, patterns, and anomalies in this column.",
    )
    domain_context: str | None = Field(
        default=None,
        description="Domain-specific context and significance of this column in the dataset.",
    )
    relationship_to_other_columns: list[str] | None = Field(
        default=None,
        description="How this column relates to or depends on other columns in the dataset.",
    )


class ColumnInfo(BaseModel):
    """Complete column information with deterministic + AI structure."""

    deterministic: ColumnDeterministicInfo | None = None
    ai_enrichment: ColumnAIEnrichment | None = None


class DataDeterministicMetadata(DeterministicMetadata):
    """Deterministic facts about tabular data."""

    type: str | None = None
    shape: list[int] | None = None  # [rows, cols]
    memory_usage_bytes: int | None = None
    column_dtypes: dict[str, str] | None = None


class DataAIEnrichment(AIEnrichment):
    """AI-generated insights about tabular data."""

    domain_analysis: str | None = Field(
        default=None,
        description="The domain this data represents and its key characteristics in a business context.",
    )
    data_quality_assessment: str | None = Field(
        default=None,
        description="Assessment of how well-structured and clear the overall schema is.",
    )
    column_interpretations: dict[str, ColumnInfo] | None = Field(
        default=None,
        description="Detailed interpretations of each column in the dataset.",
    )
    business_value_assessment: str | None = Field(
        default=None,
        description="How this data might provide business value and insights.",
    )


class DataStructure(ExtensionContext):
    """Extension for tabular data.

    See: architecture_reference.ArchitecturalComponents.TWO_TIER_METADATA
    """

    deterministic_metadata: DataDeterministicMetadata | None = None
    ai_enrichment: DataAIEnrichment | None = None
