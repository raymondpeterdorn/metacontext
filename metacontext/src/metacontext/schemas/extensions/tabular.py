"""Tabular data extension schemas."""

from pydantic import BaseModel, Field

from metacontext.schemas.extensions.base import (
    DeterministicMetadata,
    ExtensionContext,
    ForensicAIEnrichment,
)

# ========== DATA STRUCTURE EXTENSION ==========


class ColumnDeterministicInfo(DeterministicMetadata):
    """Deterministic facts about a column extracted through code execution."""

    null_count: int | None = None
    unique_count: int | None = None


class ColumnAIEnrichment(ForensicAIEnrichment):
    """AI-generated forensic insights about a column.

    Inherits forensic capabilities to dig deep and find hidden meaning
    behind column names, transformations, and business logic.
    """

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
    derived_from: list[str] | None = Field(
        default=None,
        description="Source columns used in calculation (only for code-derived columns).",
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


class DataAIEnrichment(ForensicAIEnrichment):
    """AI-generated forensic insights about tabular data.

    Inherits forensic capabilities to investigate the true purpose
    of the dataset and uncover hidden business logic.
    """

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
