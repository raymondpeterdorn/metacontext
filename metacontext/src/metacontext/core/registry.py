"""Unified registry for file handlers with automatic detection and routing.

This module provides a central registry system for file handlers that can
be used across the codebase, eliminating duplicate implementations.
"""

import logging
from pathlib import Path
from typing import Any, ClassVar, Protocol, TypeVar, cast

logger = logging.getLogger(__name__)


# Define a Protocol for handlers to avoid circular imports
class HandlerProtocol(Protocol):
    """Protocol defining the minimum interface for handlers."""

    @property
    def supported_extensions(self) -> list[str]:
        """Get supported file extensions."""
        ...

    def can_handle(self, file_path: Path, data_object: object = None) -> bool:
        """Check if handler can handle the given file."""
        ...

    def create_metacontext(self, file_path: Path, args: Any) -> Any:
        """Create metacontext for the given file."""
        ...


# Type variable for handler classes
T = TypeVar("T")


class HandlerRegistry:
    """Unified registry for file type handlers with automatic detection and routing.

    This implementation replaces duplicate registries in metacontextualize.py
    and handlers/base.py.
    """

    # Class variable to store registered handlers
    _handlers: ClassVar[list[type[Any]]] = []

    @classmethod
    def register(cls, handler_class: type[T]) -> type[T]:
        """Register a file handler class.

        Args:
            handler_class: Handler class to register

        Returns:
            The handler class (for decorator support)

        """
        if handler_class not in cls._handlers:
            cls._handlers.append(handler_class)
            logger.debug("Registered handler: %s", handler_class.__name__)
        return handler_class

    @classmethod
    def get_handler(
        cls,
        file_path: Path,
        data_object: object = None,
    ) -> "HandlerProtocol | None":
        """Get the appropriate handler for a file.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file

        Returns:
            Handler instance if found, None otherwise

        """
        for handler_class in cls._handlers:
            try:
                handler = handler_class()
                # Cast to Protocol to satisfy type checking
                handler_protocol = cast("HandlerProtocol", handler)
                if handler_protocol.can_handle(file_path, data_object):
                    logger.debug(
                        "Found handler %s for %s",
                        handler_class.__name__,
                        file_path.name,
                    )
                    
                    # Enhanced logging for Tasks 2.6-2.8
                    file_ext = file_path.suffix.lower()
                    logger.info("🎯 TASK-2.8 Handler Selection: %s selected for %s (ext: %s)", 
                               handler_class.__name__, file_path.name, file_ext)
                    
                    # Log extension pattern validation
                    if handler_class.__name__ == "CSVHandler" and file_ext in [".gpkg", ".geojson", ".shp"]:
                        logger.info("✅ TASK-2.6 Extension Pattern: Vector geospatial file routed to TabularHandler")
                    elif handler_class.__name__ == "MediaHandler" and file_ext in [".tif", ".tiff"]:
                        logger.info("✅ TASK-2.6 Extension Pattern: Raster geospatial file routed to MediaHandler") 
                    elif handler_class.__name__ == "GeospatialHandler" and file_ext in [".kml", ".kmz"]:
                        logger.info("✅ TASK-2.6 Extension Pattern: Pure geospatial file routed to GeospatialHandler")
                    
                    return cast("HandlerProtocol", handler)
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(
                    "Handler %s failed with %s: %s",
                    handler_class.__name__,
                    type(e).__name__,
                    e,
                )
                continue

        logger.debug("No handler found for %s", file_path.name)
        return None
    
    @classmethod
    def get_applicable_extensions(
        cls,
        file_path: Path,
        data_object: object = None,
    ) -> list[object]:
        """Get applicable extensions for a file/data object.
        
        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file
            
        Returns:
            List of extension instances that can enhance the base handler
        """
        extensions = []
        
        # Import here to avoid circular imports
        try:
            from metacontext.extensions.geospatial import GeospatialExtension
            
            geospatial_ext = GeospatialExtension()
            if geospatial_ext.can_extend(file_path, data_object):
                extensions.append(geospatial_ext)
                logger.debug(
                    "GeospatialExtension applicable for %s",
                    file_path.name,
                )
        except ImportError:
            logger.debug("GeospatialExtension not available")
            
        return extensions

    @classmethod
    def execute_composition_workflow(
        cls,
        file_path: Path,
        base_handler: "HandlerProtocol",
        data_object: object = None,
        ai_companion: object = None,
        codebase_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Execute composition workflow: base handler + applicable extensions.
        
        This implements TASK 2.5: Handler Composition Workflow.
        
        Args:
            file_path: Path to the file being processed
            base_handler: The primary handler (e.g., CSVHandler, MediaHandler)
            data_object: Optional data object
            ai_companion: Optional AI companion
            codebase_context: Optional codebase context
            
        Returns:
            Combined results from base handler and all applicable extensions
        """
        result = {}
        
        # Step 1: Get base handler results
        try:
            logger.info("🔄 TASK-2.5 Composition: Starting base handler analysis with %s", 
                      base_handler.__class__.__name__)
            
            # Call base handler's generate_context method
            base_context = base_handler.generate_context(
                file_path=file_path,
                data_object=data_object,
                ai_companion=ai_companion,
            )
            
            if base_context:
                result.update(base_context)
                logger.debug("✅ Base handler analysis successful")
        except Exception as e:
            logger.warning("❌ Base handler analysis failed: %s", e)
        
        # Step 2: Get and apply extensions
        extensions = cls.get_applicable_extensions(file_path, data_object)
        if extensions:
            logger.info("🔄 TASK-2.5 Composition: Found %d applicable extensions", len(extensions))
            
            for extension in extensions:
                try:
                    logger.debug("Applying %s to %s", 
                               extension.__class__.__name__, file_path.name)
                    
                    # Extract extension metadata
                    extension_data = extension.extract_spatial_metadata(file_path, data_object)
                    
                    if extension_data:
                        # Get extension key for result placement
                        extension_key = getattr(extension, "result_key", 
                                               extension.__class__.__name__.lower())
                        
                        # Add extension data to result
                        result[extension_key] = extension_data
                        
                        logger.info("✅ TASK-2.5 Composition: Successfully applied %s",
                                  extension.__class__.__name__)
                    else:
                        logger.warning("No data extracted by %s extension", 
                                     extension.__class__.__name__)
                        
                except Exception as e:
                    logger.warning("❌ Extension %s failed: %s", 
                                 extension.__class__.__name__, e)
        
        return result

    @classmethod
    def get_all_handlers(cls) -> list["HandlerProtocol"]:
        """Get instances of all registered handlers.

        Returns:
            List of all handler instances

        """
        return [handler_class() for handler_class in cls._handlers]

    @classmethod
    def get_supported_extensions(cls) -> list[str]:
        """Get all supported file extensions across all handlers.

        Returns:
            List of supported file extensions

        """
        extensions = []
        for handler_class in cls._handlers:
            try:
                handler = handler_class()
                # Cast to Protocol to satisfy type checking
                handler_protocol = cast("HandlerProtocol", handler)
                extensions.extend(handler_protocol.supported_extensions)
            except (AttributeError, TypeError) as e:
                logger.warning(
                    "Failed to get extensions from %s: %s",
                    handler_class.__name__,
                    e,
                )
                continue

        return list(set(extensions))  # Remove duplicates

    @classmethod
    def clear(cls) -> None:
        """Clear all registered handlers.

        This is primarily useful for testing.
        """
        cls._handlers.clear()

    @classmethod
    def get_required_extensions(
        cls,
        file_path: Path,
        data_object: object = None,
    ) -> list[str]:
        """Get schema extensions required for a file using dynamic discovery.

        This replaces hardcoded ExtensionMapper with dynamic discovery from handlers.

        Args:
            file_path: Path to the file
            data_object: Optional data object associated with the file

        Returns:
            List of extension names (e.g., ['data_structure', 'model_context'])

        """
        handler = cls.get_handler(file_path, data_object)
        if handler and hasattr(handler, "get_required_extensions"):
            try:
                extensions = handler.get_required_extensions(file_path, data_object)
                return list(extensions) if extensions else []
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(
                    "Handler %s failed to get extensions: %s",
                    handler.__class__.__name__,
                    e,
                )
        return []

    @classmethod
    def get_all_supported_extensions(cls) -> dict[str, list[str]]:
        """Get all supported file extensions mapped to their schema requirements.

        Returns:
            Dictionary mapping file extensions to required schema extensions

        """
        extension_map = {}
        for handler_class in cls._handlers:
            try:
                handler = handler_class()
                if hasattr(handler, "supported_extensions") and hasattr(
                    handler,
                    "required_schema_extensions",
                ):
                    for ext in handler.supported_extensions:
                        extension_map[ext] = handler.required_schema_extensions
            except (AttributeError, TypeError, ValueError) as e:
                logger.warning(
                    "Failed to get extensions for handler %s: %s",
                    handler_class.__name__,
                    e,
                )
                continue
        return extension_map
