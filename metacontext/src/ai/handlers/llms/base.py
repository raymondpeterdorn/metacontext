"""Simplified base LLM provider implementation.

This module implements the AbstractLLMProvider interface with standard functionality
for all LLM providers, focusing on a cleaner architecture with less redundancy.
"""

import logging
from typing import Any

from pydantic import BaseModel, ValidationError

from src.ai.handlers.core.exceptions import LLMError, ValidationRetryError
from src.ai.handlers.core.token_tracker import TokenTracker
from src.ai.handlers.llms.provider_interface import (
    AbstractLLMProvider,
    parse_json_response,
)
from src.core.config import get_config

logger = logging.getLogger(__name__)


class SimplifiedLLMProvider(AbstractLLMProvider):
    """Simplified base implementation for LLM providers.

    This class provides a standard implementation with cleaner separation of concerns
    and less redundant code compared to the original BaseLLMProvider.
    """

    def __init__(
        self, model: str, api_key: str | None, temperature: float = 0.1,
    ) -> None:
        """Initialize the base provider.

        Args:
            model: The model identifier to use
            api_key: API key for the provider
            temperature: Temperature parameter for generation

        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

        # Use TokenTracker to track token usage
        self._token_tracker = TokenTracker(
            provider_name=self.provider_name,
            model_name=self.model,
        )

        logger.info(
            "Initialized %s provider with model %s", self.provider_name, self.model,
        )

    def _call_llm(self, prompt: str) -> str:
        """Make a call to the LLM.

        This method should be implemented by specific provider subclasses.
        """
        msg = "Subclasses must implement _call_llm"
        raise NotImplementedError(msg)

    def generate_with_schema(
        self,
        schema_class: type[BaseModel],
        context_data: dict[str, Any],
        instruction: str | None = None,
        max_retries: int | None = None,
    ) -> BaseModel:
        """Generate and validate LLM response using schema-first approach.

        This is the main method for schema-first LLM interaction:
        1. Generate prompt from schema
        2. Call LLM
        3. Parse and validate response
        4. Retry on validation failures
        """
        if instruction is None:
            instruction = f"Generate {schema_class.__name__} analysis"

        if max_retries is None:
            # Get default retries from central configuration
            max_retries = get_config().llm.max_retries

        prompt = self.generate_schema_prompt(schema_class, context_data, instruction)

        for attempt in range(max_retries):
            try:
                logger.debug("Attempt %s for %s", attempt + 1, schema_class.__name__)

                # Call LLM
                response = self._call_llm(prompt)

                # Parse JSON
                response_data = parse_json_response(response)

                # Validate against schema
                validated_instance = schema_class(**response_data)

            except ValidationError as e:
                logger.warning("Validation failed on attempt %s: %s", attempt + 1, e)
                if attempt == max_retries - 1:
                    msg = f"Failed to generate valid {schema_class.__name__} after {max_retries} attempts: {e}"
                    raise ValidationRetryError(msg) from e
                # Continue to next attempt

            except Exception as e:
                logger.exception("Generation failed on attempt %s", attempt + 1)
                if attempt == max_retries - 1:
                    msg = f"Failed to generate {schema_class.__name__}: {e!s}"
                    raise LLMError(msg) from e
                # Continue to next attempt
            else:
                logger.info(
                    "Successfully generated %s with %s fields",
                    schema_class.__name__,
                    len(response_data),
                )
                return validated_instance

        # This part should be unreachable, but linters might complain
        msg = "Exited generation loop unexpectedly"
        raise LLMError(msg)

    def get_token_usage(self) -> dict[str, Any]:
        """Get token usage statistics."""
        return self._token_tracker.get_usage()

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about the current provider configuration."""
        return {
            "provider": self.provider_name,
            "model": self.model,
            "available": self.is_available(),
            "temperature": self.temperature,
        }

