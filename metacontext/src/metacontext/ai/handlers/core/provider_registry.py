"""Provider registry for LLM providers."""

from typing import ClassVar

from metacontext.ai.handlers.llms.provider_interface import LLMProvider


class ProviderRegistry:
    """Central registry for LLM providers."""

    _providers: ClassVar[dict[str, type[LLMProvider]]] = {}

    @classmethod
    def register(cls, name: str, provider_cls: type[LLMProvider]) -> None:
        """Register a new provider.

        Args:
            name: Provider name (e.g., "openai", "anthropic", "code_companion")
            provider_cls: Provider class that extends BaseLLMProvider

        Raises:
            ValueError: If provider name is already registered

        """
        if name in cls._providers:
            msg = f"Provider '{name}' already registered"
            raise ValueError(msg)
        cls._providers[name] = provider_cls

    @classmethod
    def get(cls, name: str) -> type[LLMProvider]:
        """Get a provider class by name.

        Args:
            name: Provider name

        Returns:
            Provider class

        Raises:
            ValueError: If provider name is not found

        """
        if name not in cls._providers:
            available = list(cls._providers.keys())
            msg = f"Provider '{name}' not found. Available: {available}"
            raise ValueError(msg)
        return cls._providers[name]

    @classmethod
    def list_providers(cls) -> list[str]:
        """List all registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def is_registered(cls, name: str) -> bool:
        """Check if a provider is registered."""
        return name in cls._providers

    @classmethod
    def unregister(cls, name: str) -> None:
        """Unregister a provider (useful for testing)."""
        if name in cls._providers:
            del cls._providers[name]
