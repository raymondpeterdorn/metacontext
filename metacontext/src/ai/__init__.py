"""AI integration package for metacontext.

This package provides LLM integration, code companion support, and codebase scanning
functionality for intelligent metadata generation.
"""

from src.ai.codebase_scanner import CodebaseScanner, scan_codebase_context
from src.ai.handlers import (
    AbstractLLMProvider,
    BaseCodeCompanionProvider,
    BaseLLMProviderConfig,
    LLMError,
    LLMProvider,
    OpenAIProvider,
    ProviderFactory,
    ProviderRegistry,
    SimplifiedLLMProvider,
    ValidationRetryError,
)

__all__ = [
    "AbstractLLMProvider",
    "BaseCodeCompanionProvider",
    "BaseLLMProviderConfig",
    "CodebaseScanner",
    "LLMError",
    "LLMProvider",
    "OpenAIProvider",
    "ProviderFactory",
    "ProviderRegistry",
    "SimplifiedLLMProvider",
    "ValidationRetryError",
    "scan_codebase_context",
]
