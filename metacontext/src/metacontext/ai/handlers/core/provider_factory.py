"""Factory for creating LLM provider instances."""

from metacontext.ai.handlers.llms.provider_interface import LLMProvider
from metacontext.core.config import get_config

from .provider_registry import ProviderRegistry


class ProviderFactory:
    """Factory to create LLM provider instances."""

    @staticmethod
    def create(
        provider_name: str | None = None,
        model: str | None = None,
        **kwargs: dict[str, object],
    ) -> LLMProvider:
        """Create a provider instance.

        Args:
            provider_name: Name of the provider (e.g., "openai", "anthropic").
                          If None, uses the value from central configuration.
            model: Model name/identifier. If None, uses the value from central configuration.
            **kwargs: Additional configuration passed to the provider

        Returns:
            Configured provider instance

        Raises:
            ValueError: If provider is not registered

        """
        # Get config from central configuration
        config = get_config()

        # If provider_name not specified, use configured provider
        if provider_name is None:
            provider_name = config.llm.provider

        # If model not specified, use configured model
        if model is None:
            model = config.llm.model

        # Create provider configuration with defaults from central config
        provider_config = {
            "model": model,
            "api_key": config.llm.api_key,
            "temperature": config.llm.temperature,
            "max_tokens": config.llm.max_tokens,
            **kwargs,  # Any overrides passed directly
        }

        # Get provider class and create instance
        provider_cls = ProviderRegistry.get(provider_name)

        # Use the create_from_config factory method if available, otherwise fallback
        if hasattr(provider_cls, "create_from_config"):
            return provider_cls.create_from_config(provider_config)

        return provider_cls(**provider_config)
        return provider_cls(**provider_config)

    @staticmethod
    def list_providers() -> list[str]:
        """List all registered provider names."""
        return ProviderRegistry.list_providers()

    @staticmethod
    def is_available(provider_name: str) -> bool:
        """Check if a provider is available."""
        return ProviderRegistry.is_registered(provider_name)
