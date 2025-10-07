"""Schema extension factory for modular schema composition.

This module provides a factory pattern for creating and composing schema
extensions dynamically, eliminating circular dependencies between schema modules.
"""

from pathlib import Path
from typing import ClassVar

from metacontext.schemas.core.interfaces import SchemaRegistry


class SchemaExtensionFactory:
    """Factory for creating and composing schema extensions.

    This class handles the discovery, instantiation, and composition of schema
    extensions, providing a unified interface for generating complete schemas
    based on file type.
    """

    # Mapping of file extensions to schema components
    _extension_mapping: ClassVar[dict[str, list[str]]] = {
        # Tabular data
        ".csv": ["data_structure"],
        ".xlsx": ["data_structure"],
        ".parquet": ["data_structure"],
        # ML models
        ".pkl": ["model_context"],
        ".joblib": ["model_context"],
        ".onnx": ["model_context"],
        ".h5": ["model_context"],
        ".pt": ["model_context"],
        # Geospatial Raster
        ".tif": ["geospatial_raster_context"],
        ".tiff": ["geospatial_raster_context"],
        ".nc": ["geospatial_raster_context"],
        # Geospatial Vector
        ".shp": ["geospatial_vector_context"],
        ".geojson": ["geospatial_vector_context"],
        ".gpkg": ["geospatial_vector_context"],
        ".geoparquet": ["geospatial_vector_context"],
        # Media
        ".jpg": ["media_context"],
        ".png": ["media_context"],
        ".mp4": ["media_context"],
        ".wav": ["media_context"],
        # Code
        ".py": ["code_context"],
        ".js": ["code_context"],
        ".ts": ["code_context"],
        ".java": ["code_context"],
    }

    @classmethod
    def get_extensions_for_file(cls, file_path: Path) -> list[str]:
        """Get schema extensions needed for a file.

        Args:
            file_path: Path to the file

        Returns:
            List of schema extension names to use for this file

        """
        file_ext = file_path.suffix.lower()

        # First try the registry for dynamic discovery
        extensions = SchemaRegistry.get_extensions_for_file(file_ext)

        # Fall back to static mapping if needed
        if not extensions:
            extensions = cls._extension_mapping.get(file_ext, [])

        return extensions

    @classmethod
    def create_schema_component(cls, schema_name: str, **kwargs: object) -> object:
        """Create a schema component instance.

        Args:
            schema_name: Name of the schema component to create
            **kwargs: Arguments to pass to the component constructor

        Returns:
            Schema component instance

        """
        extension_class = SchemaRegistry.get_extension(schema_name)
        if extension_class:
            return extension_class(**kwargs)
        return None

    @classmethod
    def register_extension_mapping(
        cls,
        file_extension: str,
        schema_names: list[str],
    ) -> None:
        """Register a new file extension to schema mapping.

        Args:
            file_extension: File extension (e.g., '.csv')
            schema_names: List of schema names to use for this extension

        """
        cls._extension_mapping[file_extension.lower()] = schema_names
