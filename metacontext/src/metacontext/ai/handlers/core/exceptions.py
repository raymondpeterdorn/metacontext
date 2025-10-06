"""Enhanced exception hierarchy for better error handling.

This module provides a comprehensive exception hierarchy for handling errors
in the metacontext system, with specific exception types for different error
categories to enable more precise error handling and recovery.
"""


class MetacontextError(Exception):
    """Base class for all metacontext-related exceptions."""

    def __init__(self, message: str, *args: object, **kwargs: object) -> None:
        """Initialize with a descriptive message."""
        self.message = message
        super().__init__(message, *args, **kwargs)


# ---- LLM-related Exceptions ----


class LLMError(MetacontextError):
    """Base exception for LLM-related errors."""


class ValidationRetryError(LLMError):
    """Raised when LLM response fails validation after max retries."""


class ProviderNotAvailableError(LLMError):
    """Raised when LLM provider is not available or not configured."""


class UnsupportedProviderError(LLMError):
    """Raised when an unsupported LLM provider is specified."""


class ProviderConfigError(LLMError):
    """Raised when there's an issue with provider configuration."""


class APIRateLimitError(LLMError):
    """Raised when an API rate limit is exceeded."""


class APIConnectionError(LLMError):
    """Raised when connection to the API fails."""


class TokenLimitExceededError(LLMError):
    """Raised when a token limit is exceeded."""


class JSONParsingError(LLMError):
    """Raised when parsing JSON from an LLM response fails."""


class SchemaValidationError(LLMError):
    """Raised when a response fails schema validation."""


# ---- Schema-related Exceptions ----


class SchemaError(MetacontextError):
    """Base exception for schema-related errors."""


class InvalidSchemaError(SchemaError):
    """Raised when a schema is invalid or malformed."""


class SchemaConflictError(SchemaError):
    """Raised when there are conflicts between schemas."""


# ---- File Handler Exceptions ----


class HandlerError(MetacontextError):
    """Base exception for handler-related errors."""


class UnsupportedFileTypeError(HandlerError):
    """Raised when a file type is not supported by any handler."""


class DataExtractionError(HandlerError):
    """Raised when data extraction from a file fails."""


class FileProcessingError(HandlerError):
    """Raised when processing a file fails."""


# ---- Codebase Context Exceptions ----


class CodebaseError(MetacontextError):
    """Base exception for codebase-related errors."""


class RepositoryError(CodebaseError):
    """Raised when there are issues with a git repository."""


class DependencyAnalysisError(CodebaseError):
    """Raised when analyzing dependencies fails."""
