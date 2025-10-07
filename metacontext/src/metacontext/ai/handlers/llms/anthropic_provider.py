"""Anthropic LLM provider implementation using the simplified architecture.

This module provides an Anthropic provider implementation with the new simplified
architecture that reduces code duplication and improves maintainability.
"""

import contextlib
import logging
import os
from typing import Any

import anthropic

from metacontext.ai.handlers.core.exceptions import LLMError
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.base import SimplifiedLLMProvider

logger = logging.getLogger(__name__)


class AnthropicProvider(SimplifiedLLMProvider):
    """Anthropic LLM provider with simplified implementation."""

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "anthropic"

    def is_available(self) -> bool:
        """Check if provider is available for use."""
        if not self.api_key:
            return False

        try:
            self._initialize_client()
        except Exception:
            logger.exception("Failed to initialize Anthropic client")
            return False
        else:
            return self.client is not None

    def _initialize_client(self) -> anthropic.Anthropic | None:
        """Initialize Anthropic client if not already done."""
        if not hasattr(self, "client") or self.client is None:
            if not self.api_key:
                logger.warning("No API key found for Anthropic")
                self.client = None
                return None

            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                logger.exception("Anthropic library not installed")
                self.client = None
                return None

        return self.client

    def _call_llm(self, prompt: str) -> str:
        """Make call to Anthropic and track token usage."""
        client = self._initialize_client()
        if not client:
            msg = "No Anthropic client available"
            raise LLMError(msg)

        try:
            # Track API call using TokenTracker
            self._token_tracker.track_api_call()

            # Claude models expect a specific format
            response = client.messages.create(
                model=self.model,
                max_tokens=self.DEFAULT_CONFIG["max_tokens"],
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt},
                ],
            )

            # Track token usage when available
            if hasattr(response, "usage"):
                input_tokens = getattr(response.usage, "input_tokens", 0)
                output_tokens = getattr(response.usage, "output_tokens", 0)

                self._token_tracker.add_prompt_tokens(input_tokens)
                self._token_tracker.add_completion_tokens(output_tokens)

            # Extract content from the Claude response
            if hasattr(response, "content") and response.content:
                # Handle the list of content blocks
                for content_block in response.content:
                    if content_block.type == "text":
                        return content_block.text
        except Exception as e:
            logger.exception("Anthropic call failed")
            msg = f"Failed to call Anthropic: {e!s}"
            raise LLMError(msg) from e
        # If no text content was found
        return ""

    @classmethod
    def get_default_model(cls) -> str:
        """Get the default model for Anthropic."""
        return "claude-3-haiku-20240307"

    @classmethod
    def get_api_key_env_var(cls) -> str:
        """Get the environment variable name for API key."""
        return "ANTHROPIC_API_KEY"

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> "AnthropicProvider":
        """Create Anthropic provider from configuration."""
        model = config.get("model", cls.get_default_model())
        api_key = config.get("api_key") or os.getenv(cls.get_api_key_env_var())
        temperature = config.get(
            "temperature",
            cls.DEFAULT_CONFIG["default_temperature"],
        )

        return cls(model=model, api_key=api_key, temperature=temperature)


# Auto-register when imported
def _register_provider() -> None:
    """Register the AnthropicProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("anthropic", AnthropicProvider)


_register_provider()
