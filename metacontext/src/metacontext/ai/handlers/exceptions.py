"""Exception classes for LLM handlers."""


class LLMError(Exception):
    """Base exception for LLM-related errors."""


class ValidationRetryError(LLMError):
    """Raised when LLM response fails validation after max retries."""


class ProviderNotAvailableError(LLMError):
    """Raised when LLM provider is not available or not configured."""


class UnsupportedProviderError(LLMError):
    """Raised when an unsupported LLM provider is specified."""
