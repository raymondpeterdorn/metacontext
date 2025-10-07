"""Codebase context schema with clear boundaries between different context types.

This module defines the schema for codebase context information that is used
across the system, avoiding overlapping concepts.
"""

from typing import Any

from pydantic import BaseModel, Field

from metacontext.schemas.extensions.base import (
    DeterministicMetadata,
    ForensicAIEnrichment,
)


class CodeDeterministicMetadata(DeterministicMetadata):
    """Deterministic facts about code files."""

    project_root: str | None = None
    language: str | None = None
    file_size_bytes: int | None = None
    file_count: int | None = Field(
        default=None,
        description="Total number of code files found in the repository",
    )
    file_types: list[str] | None = Field(
        default=None,
        description="List of file extensions found in the repository",
    )
    readme_path: str | None = Field(
        default=None,
        description="Path to the README file if found",
    )
    config_files: list[str] | None = Field(
        default=None,
        description="List of configuration files found in the repository",
    )


class CodeAIEnrichment(ForensicAIEnrichment):
    """AI-generated forensic insights about code files.

    Inherits forensic capabilities to perform code archaeology,
    tracing function purposes and uncovering hidden business logic.
    """

    purpose_analysis: str | None = Field(
        None,
        description="Analysis of the code's purpose, function, and role within the overall project.",
    )
    readme_summary: str | None = Field(
        None,
        description="Summary of the project's README file, including key points about usage, installation, and purpose.",
    )
    project_documentation: str | None = Field(
        None,
        description="Overview of the project's documentation, including availability, quality, and areas for improvement.",
    )
    complexity_assessment: str | None = Field(
        None,
        description="Assessment of the code's complexity, including algorithmic complexity, structure, and readability.",
    )
    maintainability_score: str | None = Field(
        None,
        description="A qualitative or quantitative assessment of the code's maintainability and suggestions for improvement.",
    )
    architecture_summary: str | None = Field(
        None,
        description="Summary of the codebase architecture and design patterns",
    )
    semantic_knowledge: dict[str, Any] | None = Field(
        None,
        description="Extracted semantic knowledge including column meanings, business logic, and data relationships from code analysis",
    )


class FileRelationship(BaseModel):
    """Relationship between the target file and a related file in the codebase."""

    related_file_path: str = Field(
        ...,
        description="Path to the related file",
    )
    relationship_type: str = Field(
        ...,
        description="Type of relationship (e.g., 'imports', 'uses', 'creates', 'reads')",
    )
    relevance_score: float = Field(
        default=0.0,
        description="Score indicating the strength of the relationship (0.0-1.0)",
    )
    relationship_evidence: str | None = Field(
        default=None,
        description="Evidence supporting the relationship claim (e.g., code snippet)",
    )


class CodebaseRelationships(BaseModel):
    """Relationships between the target file and other files in the codebase."""

    file_path: str = Field(
        ...,
        description="Path to the target file",
    )
    related_files: list[FileRelationship] = Field(
        default_factory=list,
        description="List of related files with relationship information",
    )
    related_config_files: list[FileRelationship] = Field(
        default_factory=list,
        description="Configuration files related to the target file",
    )
    related_documentation: list[FileRelationship] = Field(
        default_factory=list,
        description="Documentation files related to the target file",
    )


class CodebaseContext(BaseModel):
    """Context derived from codebase analysis.

    This schema defines the structure for all codebase-related context information,
    separating deterministic metadata from AI-generated enrichment and relationship
    information to avoid conceptual overlapping.
    """

    deterministic_metadata: CodeDeterministicMetadata | None = Field(
        default=None,
        description="Factual information about the codebase",
    )
    ai_enrichment: CodeAIEnrichment | None = Field(
        default=None,
        description="AI-generated insights about the codebase",
    )
    file_relationships: CodebaseRelationships | None = Field(
        default=None,
        description="Relationships between the target file and other files",
    )
    context_summary: str | None = Field(
        default=None,
        description="Concise summary of the codebase context for the target file",
    )
