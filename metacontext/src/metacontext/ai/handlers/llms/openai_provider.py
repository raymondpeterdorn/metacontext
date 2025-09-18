"""OpenAI LLM provider implementation using the simplified architecture.

This module provides an OpenAI provider implementation with the new simplified
architecture that reduces code duplication and improves maintainability.
"""

import contextlib
import logging
import os
from typing import Any

from openai import OpenAI
from pydantic import BaseModel

from metacontext.ai.handlers.core.exceptions import LLMError
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.base import SimplifiedLLMProvider
from metacontext.ai.handlers.llms.provider_interface import parse_json_response

logger = logging.getLogger(__name__)


class OpenAIProvider(SimplifiedLLMProvider):
    """OpenAI LLM provider with simplified implementation."""

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "openai"

    def is_available(self) -> bool:
        """Check if provider is available for use."""
        if not self.api_key:
            return False

        try:
            self._get_client()
        except Exception:
            logger.exception("Failed to initialize OpenAI client")
            return False
        else:
            return True

    def _get_client(self) -> OpenAI:
        """Get or initialize OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            LLMError: If client cannot be initialized

        """
        if not hasattr(self, "client") or not self.client:
            if not self.api_key:
                msg = "OpenAI API key not provided"
                raise LLMError(msg)

            try:
                self.client = OpenAI(api_key=self.api_key)
            except Exception as e:
                msg = f"Failed to initialize OpenAI client: {e}"
                raise LLMError(msg) from e

        return self.client

    def generate_json(self, prompt: str, schema_class: type[BaseModel] | None = None) -> dict[str, Any]:
        """Generate a JSON response from the OpenAI API.

        Args:
            prompt: The prompt to send to the LLM
            schema_class: Optional Pydantic model to validate against

        Returns:
            JSON response as a dictionary

        Raises:
            LLMError: For API or validation errors

        """
        client = self._get_client()

        # Track the API call
        self._token_tracker.track_api_call()

        try:
            response = client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                response_format={"type": "json_object"},
            )

            if hasattr(response, "usage"):
                # Track token usage
                self._token_tracker.add_prompt_tokens(response.usage.prompt_tokens or 0)
                self._token_tracker.add_completion_tokens(
                    response.usage.completion_tokens or 0,
                )
                self._token_tracker.add_total_tokens(response.usage.total_tokens or 0)

            response_text = response.choices[0].message.content or ""

            # Parse JSON response
            response_data = parse_json_response(response_text)

            # Validate against schema if provided
            if schema_class is not None:
                try:
                    validated = schema_class.model_validate(response_data)
                    return validated.model_dump()
                except Exception as e:
                    msg = f"Failed to validate response against schema: {e}"
                    raise LLMError(msg) from e

            return response_data
        except Exception as e:
            if not isinstance(e, LLMError):
                msg = f"OpenAI API error: {e}"
                raise LLMError(msg) from e
            raise

    def _get_system_prompt(self) -> str:
        """Get system prompt for structured generation."""
        return (
            "You are a precise AI assistant specialized in generating structured JSON responses. "
            "Follow these guidelines:\n"
            "1. Generate ONLY valid JSON based on the given schema or requirements\n"
            "2. Be factual, comprehensive, and concise\n"
            "3. Format your response as valid JSON without any surrounding text\n"
            "4. Include detailed, specific information wherever possible\n"
            "5. Avoid markdown formatting, explanations, or non-JSON content\n"
            "6. When describing data columns, provide thorough explanations of each column's meaning\n"
            "7. For model analysis, include detailed assessments of model capabilities\n"
            "8. Return ONLY valid JSON - no markdown formatting or additional text"
        )

    @classmethod
    def get_default_model(cls) -> str:
        """Get the default model for OpenAI."""
        return "gpt-4o-mini"

    @classmethod
    def get_api_key_env_var(cls) -> str:
        """Get the environment variable name for API key."""
        return "OPENAI_API_KEY"

    @classmethod
    def create_from_config(cls, config: dict[str, Any]) -> "OpenAIProvider":
        """Create OpenAI provider from configuration."""
        model = config.get("model", cls.get_default_model())
        api_key = config.get("api_key") or os.getenv(cls.get_api_key_env_var())
        temperature = config.get("temperature", 0.1)

        return cls(model=model, api_key=api_key, temperature=temperature)


# Auto-register when imported
def _register_provider() -> None:
    """Register the OpenAIProvider when module is imported."""
    with contextlib.suppress(ValueError):
        ProviderRegistry.register("openai", OpenAIProvider)


_register_provider()
