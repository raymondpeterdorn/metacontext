"""Enhanced provider manager with priority-based selection.

This module implements the provider priority system that prefers code companions
(Copilot, Cursor, Codeium) over external API providers when available.
"""

import logging
import subprocess
from typing import Any

from metacontext.ai.handlers.core.provider_factory import ProviderFactory
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.provider_interface import LLMProvider
from metacontext.core.config import get_config

logger = logging.getLogger(__name__)


class ProviderManager:
    """Enhanced provider manager with intelligent selection based on priority."""

    # Priority order: Code companions first, then API providers
    PROVIDER_PRIORITY = [
        # Code companions (free, context-aware)
        "cursor",  # Cursor AI
        "copilot",  # GitHub Copilot
        "codeium",  # Codeium
        "tabnine",  # TabNine
        # API providers (require keys, may have costs)
        "openai",  # OpenAI GPT
        "anthropic",  # Claude
        "gemini",  # Google Gemini
        "ollama",  # Local Ollama
    ]

    @classmethod
    def get_best_available_provider(
        cls,
        preferred_provider: str | None = None,
        **kwargs: dict[str, Any],
    ) -> LLMProvider:
        """Get the best available provider based on priority.

        Args:
            preferred_provider: Specific provider to try first (optional)
            **kwargs: Additional configuration for the provider

        Returns:
            Best available LLM provider instance

        Raises:
            RuntimeError: If no providers are available

        """
        config = get_config()

        # If user specifies a preferred provider, try it first
        if preferred_provider:
            if cls._is_provider_available(preferred_provider):
                logger.info("✓ Using preferred provider: %s", preferred_provider)
                return ProviderFactory.create(preferred_provider, **kwargs)
            logger.warning(
                "⚠ Preferred provider %s not available, falling back",
                preferred_provider,
            )

        # Try providers in priority order
        for provider_name in cls.PROVIDER_PRIORITY:
            if cls._is_provider_available(provider_name):
                logger.info("✓ Using %s as primary provider", provider_name)
                return ProviderFactory.create(provider_name, **kwargs)

        # Fallback to configured provider if none of the priority providers work
        fallback_provider = config.llm.provider
        if cls._is_provider_available(fallback_provider):
            logger.info("✓ Using configured fallback provider: %s", fallback_provider)
            return ProviderFactory.create(fallback_provider, **kwargs)

        # Last resort: try any registered provider
        available_providers = ProviderRegistry.list_providers()
        for provider_name in available_providers:
            if cls._is_provider_available(provider_name):
                logger.warning("⚠ Using last resort provider: %s", provider_name)
                return ProviderFactory.create(provider_name, **kwargs)

        # If we get here, no providers are available
        msg = f"No LLM providers available. Registered: {available_providers}"
        raise RuntimeError(msg)

    @classmethod
    def _is_provider_available(cls, provider_name: str) -> bool:
        """Check if a provider is available and can be used.

        Args:
            provider_name: Name of the provider to check

        Returns:
            True if provider is available and functional

        """
        if not ProviderRegistry.is_registered(provider_name):
            return False

        # Check code companions first
        cli_commands = {
            "copilot": ["gh", "copilot", "--version"],
            "cursor": ["cursor", "--version"],
            "codeium": ["codeium", "--version"],
            "tabnine": ["tabnine", "--version"],
        }

        if provider_name in cli_commands:
            return cls._check_cli_available(cli_commands[provider_name])

        # For API providers, check if they can actually be instantiated
        try:
            provider = ProviderFactory.create(provider_name)
            return provider.is_available()
        except (ImportError, RuntimeError, ValueError) as e:
            logger.debug("Provider %s not available: %s", provider_name, e)
            return False

    @classmethod
    def _check_cli_available(cls, command: list[str]) -> bool:
        """Check if a CLI command is available.

        Args:
            command: Command to test (e.g., ["gh", "copilot", "--version"])

        Returns:
            True if command is available and works

        """
        try:
            subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=5,
                check=False,  # Don't raise on non-zero exit
            )
            # Consider available if command runs (regardless of exit code)
            return True
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        except Exception as e:
            logger.debug("CLI check failed for %s: %s", command[0], e)
            return False

    @classmethod
    def list_available_providers(cls) -> list[tuple[str, bool]]:
        """List all providers with their availability status.

        Returns:
            List of tuples (provider_name, is_available)

        """
        all_providers = ProviderRegistry.list_providers()
        return [
            (provider, cls._is_provider_available(provider))
            for provider in all_providers
        ]

    @classmethod
    def get_provider_info(cls, provider_name: str) -> dict[str, Any]:
        """Get detailed information about a provider.

        Args:
            provider_name: Name of the provider

        Returns:
            Dictionary with provider information

        """
        if not ProviderRegistry.is_registered(provider_name):
            return {
                "name": provider_name,
                "registered": False,
                "available": False,
                "type": "unknown",
            }

        is_companion = provider_name in ["copilot", "cursor", "codeium", "tabnine"]

        return {
            "name": provider_name,
            "registered": True,
            "available": cls._is_provider_available(provider_name),
            "type": "code_companion" if is_companion else "api_provider",
            "priority": cls.PROVIDER_PRIORITY.index(provider_name)
            if provider_name in cls.PROVIDER_PRIORITY
            else 999,
        }
