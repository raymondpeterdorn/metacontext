"""Core metacontext schema."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, Field

from metacontext.schemas.core.interfaces import ConfidenceLevel
from metacontext.schemas.extensions.geospatial import (
    GeospatialRasterContext,
    GeospatialVectorContext,
)
from metacontext.schemas.extensions.media import MediaContext
from metacontext.schemas.extensions.models import ModelContext
from metacontext.schemas.extensions.spatial import SpatialExtension
from metacontext.schemas.extensions.tabular import DataStructure


class GenerationMethod(str, Enum):
    """Method used to generate the metacontext."""

    EXPLICIT_FUNCTION = "explicit_function"
    AUTOMATIC = "automatic"
    MANUAL = "manual"


# ========== CORE ARCHITECTURE ==========


class TokenUsage(BaseModel):
    """Token usage statistics for LLM API calls."""

    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_api_calls: int | None = None
    provider: str | None = None
    model: str | None = None


class GenerationInfo(BaseModel):
    """Information about how the metacontext was generated."""

    generated_at: datetime
    generation_method: GenerationMethod | None = None
    function_call: str | None = None
    token_usage: TokenUsage | None = None


class FileInfo(BaseModel):
    """Core file information - always present regardless of file type."""

    filename: str
    extension: str
    file_purpose: str | None = None
    source_script: str | None = None
    project_context_summary: str | None = None
    creation_timestamp: datetime | None = None


class FileContext(BaseModel):
    """Universal context about the file's nature, creation, and intended use."""

    file_summary: str = Field(
        description="What this file is, how it was created, and what it's useful for",
    )
    creation_method: str | None = Field(
        default=None,
        description="How this file was generated (script, manual, tool, export, etc.)",
    )
    intended_use: str | None = Field(
        default=None,
        description="Primary purpose and applications for this file",
    )
    data_lineage: str | None = Field(
        default=None,
        description="Source data and transformation history",
    )


class SystemInfo(BaseModel):
    """System information about the generation environment."""

    python_version: str | None = None
    platform: str | None = None
    working_directory: str | None = None


class ConfidenceAssessment(BaseModel):
    """Overall confidence in the generated context."""

    file_purpose: ConfidenceLevel | None = None
    schema_interpretation: ConfidenceLevel | None = None
    business_context: ConfidenceLevel | None = None
    overall: ConfidenceLevel | None = None


class Metacontext(BaseModel):
    """Root metacontext schema with core + extensions architecture.

    Core fields are always present, extension fields are added based on file type.
    """

    # Core (always present)
    metacontext_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    generation_info: GenerationInfo
    file_info: FileInfo
    file_context: FileContext
    system_info: SystemInfo | None = None
    # NOTE: codebase_context removed as it provides no valuable information in output
    business_context: str | None = None
    usage_patterns: str | None = None
    confidence_assessment: ConfidenceAssessment | None = None

    # Extensions (conditionally present based on file type)
    data_structure: DataStructure | None = None
    geospatial_raster_context: GeospatialRasterContext | None = None
    geospatial_vector_context: GeospatialVectorContext | None = None
    spatial_extension: SpatialExtension | None = None  # Lightweight spatial metadata
    model_context: ModelContext | None = None
    media_context: MediaContext | None = None

    # AI companion integration
    ai_companion_status: str | None = None
    generation_method: str | None = None
    available_companions: dict[str, bool] | None = None
    codebase_scan: dict[str, Any] | None = None

    class Config:
        """Pydantic configuration for strict schema validation."""

        extra = "forbid"  # Prevents additional fields not defined in schema
        validate_assignment = True  # Validates on assignment after creation
        use_enum_values = True  # Use enum values in serialization


# ========== UTILITY FUNCTIONS ==========


def create_base_metacontext(
    filename: str,
    file_purpose: str | None = None,
    project_context_summary: str | None = None,
    file_summary: str | None = None,
    creation_method: str | None = None,
    intended_use: str | None = None,
) -> Metacontext:
    """Create a base metacontext with core fields populated."""
    file_path = Path(filename)

    # Generate default file summary if not provided
    if not file_summary:
        file_summary = f"A {file_path.suffix} file containing data analysis and metadata context"

    return Metacontext(
        metacontext_version="0.3.0",
        generation_info=GenerationInfo(
            generated_at=datetime.now(UTC),
            generation_method=GenerationMethod.EXPLICIT_FUNCTION,
            function_call="metacontext.metacontextualize()",
        ),
        file_info=FileInfo(
            filename=file_path.name,
            extension=file_path.suffix,
            file_purpose=file_purpose,
            project_context_summary=project_context_summary,
            creation_timestamp=datetime.now(UTC),
        ),
        file_context=FileContext(
            file_summary=file_summary,
            creation_method=creation_method or "automatic_generation",
            intended_use=intended_use or "data_analysis_and_context",
        ),
        system_info=SystemInfo(working_directory=str(Path.cwd())),
    )


def get_schema_for_prompt(schema_class: type[BaseModel]) -> dict[str, Any]:
    """Generate JSON schema for LLM prompts with validation.

    This enables schema-first prompt engineering where LLM prompts
    are automatically generated from Pydantic schema definitions.
    """
    return schema_class.model_json_schema()


# ========== EXTENSION MANAGEMENT ==========


class ExtensionCategory(str, Enum):
    """Categories of file extensions for more efficient mapping."""

    TABULAR = "tabular"
    ML_MODEL = "ml_model"
    GEOSPATIAL_RASTER = "geospatial_raster"
    GEOSPATIAL_VECTOR = "geospatial_vector"
    MEDIA = "media"
    CODE = "code"
    DOCUMENT = "document"
    UNKNOWN = "unknown"


class ExtensionMapper:
    """Efficient file extension mapping using categorization.

    This class replaces the previous dictionary-based approach with a more
    organized and maintainable mapping system.
    """

    # Map categories to extension schemas
    _CATEGORY_TO_SCHEMAS: ClassVar[dict[ExtensionCategory, list[str]]] = {
        ExtensionCategory.TABULAR: ["data_structure"],
        ExtensionCategory.ML_MODEL: ["model_context"],
        ExtensionCategory.GEOSPATIAL_RASTER: ["geospatial_context"],
        ExtensionCategory.GEOSPATIAL_VECTOR: ["geospatial_context"],
        ExtensionCategory.MEDIA: ["media_context"],
        ExtensionCategory.CODE: ["code_context"],
        ExtensionCategory.DOCUMENT: ["document_context"],
        ExtensionCategory.UNKNOWN: [],
    }

    # Map file extensions to categories
    _EXTENSION_TO_CATEGORY: ClassVar[dict[str, ExtensionCategory]] = {
        # Tabular data
        ".csv": ExtensionCategory.TABULAR,
        ".xlsx": ExtensionCategory.TABULAR,
        ".parquet": ExtensionCategory.TABULAR,
        ".tsv": ExtensionCategory.TABULAR,
        ".feather": ExtensionCategory.TABULAR,
        # ML models
        ".pkl": ExtensionCategory.ML_MODEL,
        ".joblib": ExtensionCategory.ML_MODEL,
        ".onnx": ExtensionCategory.ML_MODEL,
        ".h5": ExtensionCategory.ML_MODEL,
        ".pt": ExtensionCategory.ML_MODEL,
        ".pth": ExtensionCategory.ML_MODEL,
        ".keras": ExtensionCategory.ML_MODEL,
        ".tflite": ExtensionCategory.ML_MODEL,
        # Geospatial Raster
        ".tif": ExtensionCategory.GEOSPATIAL_RASTER,
        ".tiff": ExtensionCategory.GEOSPATIAL_RASTER,
        ".nc": ExtensionCategory.GEOSPATIAL_RASTER,
        ".asc": ExtensionCategory.GEOSPATIAL_RASTER,
        # Geospatial Vector
        ".shp": ExtensionCategory.GEOSPATIAL_VECTOR,
        ".geojson": ExtensionCategory.GEOSPATIAL_VECTOR,
        ".gpkg": ExtensionCategory.GEOSPATIAL_VECTOR,
        ".geoparquet": ExtensionCategory.GEOSPATIAL_VECTOR,
        # Media
        ".jpg": ExtensionCategory.MEDIA,
        ".jpeg": ExtensionCategory.MEDIA,
        ".png": ExtensionCategory.MEDIA,
        ".gif": ExtensionCategory.MEDIA,
        ".mp4": ExtensionCategory.MEDIA,
        ".mp3": ExtensionCategory.MEDIA,
        ".wav": ExtensionCategory.MEDIA,
        # Code
        ".py": ExtensionCategory.CODE,
        ".js": ExtensionCategory.CODE,
        ".ts": ExtensionCategory.CODE,
        ".java": ExtensionCategory.CODE,
        ".cpp": ExtensionCategory.CODE,
        ".c": ExtensionCategory.CODE,
        ".cs": ExtensionCategory.CODE,
        ".go": ExtensionCategory.CODE,
        ".rb": ExtensionCategory.CODE,
        ".php": ExtensionCategory.CODE,
        ".swift": ExtensionCategory.CODE,
        ".rs": ExtensionCategory.CODE,
        # Documents
        ".md": ExtensionCategory.DOCUMENT,
        ".txt": ExtensionCategory.DOCUMENT,
        ".pdf": ExtensionCategory.DOCUMENT,
        ".docx": ExtensionCategory.DOCUMENT,
        ".json": ExtensionCategory.DOCUMENT,
        ".yaml": ExtensionCategory.DOCUMENT,
        ".yml": ExtensionCategory.DOCUMENT,
    }

    @classmethod
    def get_category(cls, extension: str) -> ExtensionCategory:
        """Get the category for a file extension.

        Args:
            extension: File extension (with or without leading dot)

        Returns:
            ExtensionCategory: The category of the file

        """
        # Ensure extension has a leading dot
        if not extension.startswith("."):
            extension = f".{extension}"

        # Convert to lowercase for case-insensitive matching
        extension_lower = extension.lower()

        # Return the category or UNKNOWN
        return cls._EXTENSION_TO_CATEGORY.get(
            extension_lower,
            ExtensionCategory.UNKNOWN,
        )

    @classmethod
    def get_file_type_extension(cls, file_extension: str) -> list[str]:
        """Determine which extension schemas should be included based on file extension.

        This replaces the previous dictionary-based approach with a more efficient
        categorization system.

        Args:
            file_extension: The file extension (with or without leading dot)

        Returns:
            List of extension schema names to use

        """
        category = cls.get_category(file_extension)
        return cls._CATEGORY_TO_SCHEMAS.get(category, [])

    @classmethod
    def register_extension(cls, extension: str, category: ExtensionCategory) -> None:
        """Register a new file extension to a category.

        Args:
            extension: The file extension (with or without leading dot)
            category: The category to assign this extension to

        """
        # Ensure extension has a leading dot
        if not extension.startswith("."):
            extension = f".{extension}"

        # Add to the mapping
        cls._EXTENSION_TO_CATEGORY[extension.lower()] = category

    @classmethod
    def get_all_extensions_by_category(cls) -> dict[ExtensionCategory, list[str]]:
        """Get all registered extensions grouped by category.

        Returns:
            Dictionary of categories mapping to lists of extensions

        """
        result: dict[ExtensionCategory, list[str]] = {
            category: [] for category in ExtensionCategory
        }

        for ext, category in cls._EXTENSION_TO_CATEGORY.items():
            result[category].append(ext)

        return result


def get_extensions_for_file_type(file_extension: str) -> list[str]:
    """Determine which extensions should be included based on file extension.

    This function is maintained for backward compatibility but now uses
    the more efficient ExtensionMapper implementation.

    Args:
        file_extension: The file extension

    Returns:
        List of extension schema names to use

    """
    return ExtensionMapper.get_file_type_extension(file_extension)
