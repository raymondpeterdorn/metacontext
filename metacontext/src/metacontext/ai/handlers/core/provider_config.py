"""Provider configuration models.

This module provides Pydantic models for LLM provider configurations to replace
complex dictionary-based configurations with type-safe models.
"""

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    """Supported LLM provider types."""

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    CODE_COMPANION = "code_companion"


class ProviderConfig(BaseModel):
    """Base configuration for all LLM providers."""

    provider_type: ProviderType
    model: str
    api_key: str | None = None
    temperature: float = 0.1

    # Additional provider-specific parameters with defaults
    max_tokens: int | None = None
    system_prompt: str | None = None

    # Other parameters that don't fit the standard pattern
    extra_params: dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic model configuration."""

        extra = "allow"  # Allow extra fields not in the model


class OpenAIConfig(ProviderConfig):
    """OpenAI-specific provider configuration."""

    provider_type: ProviderType = ProviderType.OPENAI
    model: str = "gpt-4o"

    # OpenAI-specific parameters
    response_format: dict[str, str] | None = None
    frequency_penalty: float | None = None
    presence_penalty: float | None = None

    @classmethod
    def create_default(cls, api_key: str | None = None) -> "OpenAIConfig":
        """Create default OpenAI configuration.

        Args:
            api_key: Optional API key override

        Returns:
            Default OpenAI configuration

        """
        return cls(api_key=api_key)


class AnthropicConfig(ProviderConfig):
    """Anthropic-specific provider configuration."""

    provider_type: ProviderType = ProviderType.ANTHROPIC
    model: str = "claude-3-opus-20240229"

    # Anthropic-specific parameters
    max_tokens_to_sample: int | None = None

    @classmethod
    def create_default(cls, api_key: str | None = None) -> "AnthropicConfig":
        """Create default Anthropic configuration.

        Args:
            api_key: Optional API key override

        Returns:
            Default Anthropic configuration

        """
        return cls(api_key=api_key)


class GeminiConfig(ProviderConfig):
    """Gemini-specific provider configuration."""

    provider_type: ProviderType = ProviderType.GEMINI
    model: str = "gemini-1.5-pro"

    # Gemini-specific parameters
    top_p: float | None = None
    top_k: int | None = None

    @classmethod
    def create_default(cls, api_key: str | None = None) -> "GeminiConfig":
        """Create default Gemini configuration.

        Args:
            api_key: Optional API key override

        Returns:
            Default Gemini configuration

        """
        return cls(api_key=api_key)


class CodeCompanionConfig(ProviderConfig):
    """Code companion provider configuration."""

    provider_type: ProviderType = ProviderType.CODE_COMPANION
    model: str = "companion"

    # Code companion specific parameters
    companion_name: str = "copilot"
    timeout_seconds: int = 60

    @classmethod
    def create_default(cls, companion_name: str = "copilot") -> "CodeCompanionConfig":
        """Create default code companion configuration.

        Args:
            companion_name: Name of the code companion (e.g., "copilot", "codeium")

        Returns:
            Default code companion configuration

        """
        return cls(companion_name=companion_name)


# Factory function to create appropriate config
def create_provider_config(
    provider_type: str | ProviderType,
    model: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> ProviderConfig:
    """Create a provider configuration based on provider type.

    Args:
        provider_type: Type of provider (e.g., "openai", "anthropic")
        model: Optional model name (uses default if not provided)
        api_key: Optional API key
        **kwargs: Additional provider-specific parameters

    Returns:
        Provider configuration appropriate for the specified provider type

    Raises:
        ValueError: If provider_type is not recognized

    """
    if isinstance(provider_type, str):
        try:
            provider_type = ProviderType(provider_type.lower())
        except ValueError as e:
            valid_types = [pt.value for pt in ProviderType]
            msg = f"Unsupported provider type: {provider_type}. Valid types: {valid_types}"
            raise ValueError(msg) from e

    # Select the appropriate config class
    config_class = {
        ProviderType.OPENAI: OpenAIConfig,
        ProviderType.ANTHROPIC: AnthropicConfig,
        ProviderType.GEMINI: GeminiConfig,
        ProviderType.CODE_COMPANION: CodeCompanionConfig,
    }[provider_type]

    # Create the config with provided parameters
    config_params = {"api_key": api_key, **kwargs}
    if model:
        config_params["model"] = model

    return config_class(**config_params)
