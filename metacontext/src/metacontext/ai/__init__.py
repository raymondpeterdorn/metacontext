"""AI integration package for metacontext.

This package provides LLM integration, code companion support, and codebase scanning
functionality for intelligent metadata generation.
"""

from metacontext.ai.codebase_scanner import CodebaseScanner, scan_codebase_context
from metacontext.ai.handlers import (
    AbstractLLMProvider,
    BaseCompanionProvider,
    BaseLLMProviderConfig,
    CompanionProviderFactory,
    GitHubCopilotProvider,
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
    "BaseCompanionProvider",
    "BaseLLMProviderConfig",
    "CodebaseScanner",
    "CompanionProviderFactory",
    "GitHubCopilotProvider",
    "LLMError",
    "LLMProvider",
    "OpenAIProvider",
    "ProviderFactory",
    "ProviderRegistry",
    "SimplifiedLLMProvider",
    "ValidationRetryError",
    "scan_codebase_context",
]
