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
