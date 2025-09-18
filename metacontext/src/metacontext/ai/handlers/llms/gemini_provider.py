"""Gemini LLM provider implementation using the simplified architecture.

This module provides a Gemini provider implementation with the new simplified
architecture that reduces code duplication and improves maintainability.
"""

import contextlib
import logging
from typing import Any

import google.generativeai as genai

from metacontext.ai.handlers.core.exceptions import LLMError
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.base import SimplifiedLLMProvider
from metacontext.core.config import get_config

logger = logging.getLogger(__name__)


class GeminiProvider(SimplifiedLLMProvider):
    """Gemini LLM provider with simplified implementation."""

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "gemini"

    def is_available(self) -> bool:
        """Check if provider is available for use."""
        if not self.api_key:
            return False

        try:
            self._initialize_client()
        except Exception:
            logger.exception("Failed to initialize Gemini client")
            return False
        else:
            return self.client is not None

    def _initialize_client(self) -> object | None:
        """Initialize Gemini client if not already done."""
        if not hasattr(self, "client") or self.client is None:
            if not self.api_key:
                logger.warning("No API key found for Gemini")
                self.client = None
                return None

            try:
                genai.configure(api_key=self.api_key)
                # Store genai module as the client for simplicity
                self.client = genai
            except ImportError:
                logger.exception("Google generative AI library not installed")
                self.client = None
                return None

        return self.client

    def _call_llm(self, prompt: str) -> str:
        """Make call to Gemini and track token usage."""
        client = self._initialize_client()
        if not client:
            msg = "No Gemini client available"
            raise LLMError(msg)

        try:
            # Track API call using TokenTracker
            self._token_tracker.track_api_call()

            logger.info("🔍 Calling Gemini with model: %s", self.model)
            logger.debug("Prompt: %s", prompt[:200] + "..." if len(prompt) > 200 else prompt)

            # Configure generation for JSON format
            generation_config = {
                "temperature": self.temperature,
                "response_mime_type": "application/json",
            }

            model = client.GenerativeModel(
                model_name=self.model,
                generation_config=generation_config,
            )
            response = model.generate_content(prompt)

            logger.info("✓ Gemini response received")

            # Track token usage when available
            # Try multiple ways to extract token usage from Gemini response
            if hasattr(response, "usage_metadata"):
                usage = response.usage_metadata
                if hasattr(usage, "prompt_token_count") and hasattr(usage, "candidates_token_count"):
                    prompt_tokens = getattr(usage, "prompt_token_count", 0)
                    completion_tokens = getattr(usage, "candidates_token_count", 0)
                    total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens)

                    self._token_tracker.track_response(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                    )
                    logger.info("Token usage: %d prompt + %d completion = %d total",
                              prompt_tokens, completion_tokens, total_tokens)
                elif hasattr(usage, "total_token_count"):
                    total_tokens = getattr(usage, "total_token_count", 0)
                    self._token_tracker.add_total_tokens(total_tokens)
                    logger.info("Total tokens: %d", total_tokens)
            else:
                # Fallback: try older API format
                token_count = getattr(response, "usage", {}).get("total_tokens", 0)
                if token_count > 0:
                    self._token_tracker.add_total_tokens(token_count)
                    logger.info("Token count (fallback): %d", token_count)

            if hasattr(response, "text"):
                logger.info("Response text available via .text")
                response_text = str(response.text)
                logger.debug("Raw response: %s", response_text[:500] + "..." if len(response_text) > 500 else response_text)
                return response_text

            if hasattr(response, "parts") and response.parts:
                logger.info("Response text available via .parts[0].text")
                response_text = str(response.parts[0].text)
                logger.debug("Raw response: %s", response_text[:500] + "..." if len(response_text) > 500 else response_text)
                return response_text

            logger.warning("⚠️ No text found in Gemini response")
            return ""

        except Exception as e:
            logger.exception("Gemini call failed")
            msg = f"Failed to call Gemini: {e!s}"
            raise LLMError(msg) from e

    @classmethod
    def get_default_model(cls) -> str:
        """Get the default model for Gemini."""
        return "gemini-1.5-flash"

    @classmethod
    def get_api_key_env_var(cls) -> str:
        """Get the environment variable name for API key."""
        return "GEMINI_API_KEY"

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> "GeminiProvider":
        """Create Gemini provider from configuration."""
        # Get central configuration
        central_config = get_config().llm

        # Use provided config with fallbacks to central config
        model = config.get("model", central_config.model)
        api_key = config.get("api_key", central_config.api_key)
        temperature = config.get("temperature", central_config.temperature)

        return cls(model=model, api_key=api_key, temperature=temperature)


# Auto-register when imported
def _register_provider() -> None:
    """Register the GeminiProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("gemini", GeminiProvider)


_register_provider()
