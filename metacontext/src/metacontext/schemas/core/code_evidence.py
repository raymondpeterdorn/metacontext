"""Code evidence schema for capturing source code context."""

from typing import Any

from pydantic import BaseModel, Field


class CodeSnippet(BaseModel):
    """Represents a code snippet with context information."""

    content: str = Field(
        description="The actual source code snippet",
    )
    file_path: str = Field(
        description="Path to the file containing this code",
    )
    start_line: int = Field(
        description="Starting line number of the snippet",
    )
    end_line: int = Field(
        description="Ending line number of the snippet",
    )
    function_name: str | None = Field(
        default=None,
        description="Name of the function or method containing this code",
    )
    class_name: str | None = Field(
        default=None,
        description="Name of the class containing this code",
    )


class FieldDescription(BaseModel):
    """Represents a Pydantic field description with source context."""

    field_name: str = Field(
        description="Name of the field",
    )
    description: str = Field(
        description="The field description/documentation",
    )
    field_type: str | None = Field(
        default=None,
        description="Type annotation of the field",
    )
    source_context: CodeSnippet | None = Field(
        default=None,
        description="Source code context where this field is defined",
    )


class CommentWithContext(BaseModel):
    """Represents a comment with surrounding code context."""

    comment_text: str = Field(
        description="The actual comment text",
    )
    comment_type: str = Field(
        description="Type of comment: inline, block, docstring",
    )
    related_code: CodeSnippet | None = Field(
        default=None,
        description="Code that this comment is associated with",
    )


class CodeEvidence(BaseModel):
    """Container for all code evidence related to a data structure or field."""

    related_snippets: list[CodeSnippet] = Field(
        default_factory=list,
        description="Code snippets that reference or manipulate this data",
    )
    field_descriptions: list[FieldDescription] = Field(
        default_factory=list,
        description="Pydantic field descriptions with source context",
    )
    associated_comments: list[CommentWithContext] = Field(
        default_factory=list,
        description="Comments that provide context about this data",
    )
    data_transformations: list[CodeSnippet] = Field(
        default_factory=list,
        description="Code snippets showing how this data is transformed or processed",
    )
    validation_logic: list[CodeSnippet] = Field(
        default_factory=list,
        description="Code snippets showing validation or business rules",
    )
    cross_references: dict[str, Any] = Field(
        default_factory=dict,
        description="Cross-references to other parts of the codebase",
    )
