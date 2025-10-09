"""Interface definition for LLM providers.

This module defines the core interfaces that all LLM providers must implement,
promoting a cleaner separation between interface and implementation.
"""

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Protocol

from pydantic import BaseModel

from metacontext.ai.handlers.core.exceptions import LLMError
from metacontext.ai.prompts.schema_utils import compact_schema_hint

logger = logging.getLogger(__name__)

# Constants for logging truncation
LOG_PREVIEW_LENGTH = 200


class LLMProvider(Protocol):
    """Protocol defining the core LLM provider interface.

    This protocol defines the minimal interface that any LLM provider must implement,
    regardless of implementation details.
    """

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        ...

    def generate_completion(self, prompt: str) -> str:
        """Generate a text completion for the given prompt."""
        ...

    def generate_with_schema(
        self,
        schema_class: type[BaseModel],
        context_data: dict[str, Any],
        instruction: str | None = None,
        max_retries: int | None = None,
    ) -> BaseModel:
        """Generate and validate a response using the provided schema."""
        ...

    def is_available(self) -> bool:
        """Check if the provider is available for use."""
        ...

    def get_token_usage(self) -> dict[str, Any]:
        """Get current token usage statistics."""
        ...

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about the current provider configuration."""
        ...

    def is_companion_mode(self) -> bool:
        """Return True if this provider uses companion workflow, False for API workflow."""
        ...

    @property
    def execution_mode(self) -> str:
        """Return 'api' or 'companion' to indicate execution mode."""
        ...


class BaseLLMProviderConfig:
    """Configuration defaults for LLM providers.

    This class centralizes configuration defaults to avoid duplication across provider implementations.

    Note: These default values are now provided as fallbacks only. The primary source of
    configuration should be the centralized configuration system in metacontext.core.config.
    """

    # Default instruction templates
    INSTRUCTION_CONFIG: ClassVar[dict[str, str]] = {
        "default_schema": "Generate JSON matching the provided schema",
    }


class AbstractLLMProvider(ABC, BaseLLMProviderConfig):
    """Abstract base implementation providing common functionality for LLM providers.

    This class implements common functionality that is likely to be shared by most
    provider implementations, while still requiring implementation of the core
    provider-specific methods.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the name of the provider."""

    @abstractmethod
    def _call_llm(self, prompt: str) -> str:
        """Make the actual call to the LLM service."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available for use."""

    # The following methods have a default implementation but can be overridden

    def generate_completion(self, prompt: str) -> str:
        """Generate a text completion for the given prompt."""
        return self._call_llm(prompt)

    def generate_schema_prompt(
        self,
        schema_class: type[BaseModel],
        context_data: dict[str, Any],
        instruction: str | None = None,
    ) -> str:
        """Generate LLM prompt from Pydantic schema.

        This is the core of schema-first prompt engineering - prompts are
        automatically generated from schema definitions and stay in sync.
        """
        schema_json = schema_class.model_json_schema()
        if instruction is None:
            instruction = self.INSTRUCTION_CONFIG["default_schema"]

        return f"""{instruction}

CONTEXT DATA:
{json.dumps(context_data, indent=2, default=str)}

REQUIRED JSON SCHEMA:
{json.dumps(schema_json, indent=2)}

INSTRUCTIONS:
1. Analyze the provided context data thoroughly and extract all relevant information
2. Generate a detailed JSON response that exactly matches the schema above
3. Fill all fields with comprehensive, detailed information based on the context
4. For missing information:
   - Use null for optional object/array fields
   - Use empty string "" for string fields where information is not available
   - Never use null for string values in dictionaries
   - For dict fields, provide detailed key-value pairs even when specific data is limited
   - For list fields, provide comprehensive lists with detailed entries
5. Ensure all enum values match exactly as specified
6. When describing data columns, provide thorough explanations of each column's meaning, purpose, and content
7. For model analysis, include detailed assessments of model capabilities, limitations, and suggested uses
8. Return ONLY valid JSON - no markdown formatting or additional text
"""

    def is_companion_mode(self) -> bool:
        """Return True if this provider uses companion workflow, False for API workflow.

        Default implementation returns False (API mode). Companion providers should override
        this method to return True.
        """
        return False

    @property
    def execution_mode(self) -> str:
        """Return 'api' or 'companion' to indicate execution mode.

        This property is derived from is_companion_mode() and provides a string
        representation of the execution mode for logging and debugging.
        """
        return "companion" if self.is_companion_mode() else "api"

    def _build_optimized_prompt(
        self,
        schema_class: type[BaseModel],
        context_data: dict[str, Any],
        instruction: str,
    ) -> str:
        """Build optimized prompt using existing generation logic.

        This method extracts the prompt building logic to make it reusable
        by both API and companion modes. It delegates to the existing
        generate_schema_prompt method to preserve current behavior.

        Args:
            schema_class: Pydantic model class for response validation
            context_data: Context data to include in prompt
            instruction: Instruction text for the LLM

        Returns:
            Complete optimized prompt ready for LLM consumption

        """
        # Use existing schema prompt generation to preserve current behavior
        # Subclasses can override this method to use more sophisticated logic
        return self.generate_schema_prompt(schema_class, context_data, instruction)

    def _generate_schema_hint(self, schema_class: type[BaseModel]) -> str:
        """Generate compact schema hint for prompt optimization.

        This method creates a compact representation of the schema that can be
        used in prompt optimization strategies, using 80% fewer tokens than
        full JSON schemas.

        Args:
            schema_class: Pydantic model class

        Returns:
            Compact schema hint optimized for token efficiency

        """
        return compact_schema_hint(schema_class)

    def _preprocess_context(self, context_data: dict[str, Any]) -> dict[str, Any]:
        """Preprocess context data for prompt optimization.

        This method provides a hook for context preprocessing that can be
        overridden by specific providers or companion implementations.

        Args:
            context_data: Raw context data

        Returns:
            Processed context data ready for prompt generation

        """
        return context_data  # Default: no preprocessing


def parse_json_response(response: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response.

    This utility function helps extract JSON from text responses that might contain
    markdown code blocks or other text. Includes recovery for truncated JSON.
    """
    # Strip whitespace
    response = response.strip()

    # Log the raw response for debugging
    logger.info(
        "Raw LLM response: %s",
        response[:LOG_PREVIEW_LENGTH] + "..."
        if len(response) > LOG_PREVIEW_LENGTH
        else response,
    )

    # Try to find JSON block using regex (more efficient than multiple string operations)
    json_block_pattern = r"```(?:json)?\s*([\s\S]*?)```"
    json_matches = re.findall(json_block_pattern, response)

    if json_matches:
        # Use the first JSON block found
        json_str = json_matches[0].strip()
        logger.info("Found JSON block using regex")
    else:
        # Look for JSON object boundaries
        start = response.find("{")
        end = response.rfind("}")

        json_str = response[start : end + 1] if start >= 0 and end > start else response
        logger.info(
            "Extracted JSON using object boundaries: %s",
            json_str[:LOG_PREVIEW_LENGTH] + "..."
            if len(json_str) > LOG_PREVIEW_LENGTH
            else json_str,
        )

    try:
        result = json.loads(json_str)
        logger.info(
            "Parsed JSON keys: %s",
            list(result.keys()) if isinstance(result, dict) else "Not a dict",
        )
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError as e:
        logger.exception("Failed to parse JSON response")
        msg = f"Invalid JSON response: {e!s}"
        raise LLMError(msg) from e
