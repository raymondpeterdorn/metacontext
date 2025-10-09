"""Companion LLM provider for clipboard-based workflow.

This module implements the companion-specific LLM provider that follows the unified
interface while implementing clipboard-based workflow instead of direct API calls.
"""

import logging
from typing import Any

from pydantic import BaseModel

from metacontext.ai.handlers.companions.clipboard_manager import ClipboardManager
from metacontext.ai.handlers.core.exceptions import LLMError
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.provider_interface import (
    AbstractLLMProvider,
    parse_json_response,
)

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)


class CompanionLLMProvider(AbstractLLMProvider):
    """Companion LLM provider implementing clipboard-based workflow.

    This provider follows the unified pipeline (steps B→H) but differs in step I
    by using clipboard interaction instead of direct API calls.
    """

    def __init__(
        self,
        model: str = "companion",
        api_key: str | None = None,
        temperature: float = 0.1,
    ) -> None:
        """Initialize companion LLM provider.

        Args:
            model: Model identifier (not used for companion mode)
            api_key: API key (not used for companion mode)
            temperature: Temperature setting (not used for companion mode)

        """
        # Store configuration for compatibility
        self.model = model
        self.api_key = api_key
        self.temperature = temperature

        # Initialize companion-specific state
        self._last_prompt: str | None = None
        self._clipboard_interactions = 0

        logger.info("Initialized companion LLM provider")

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "companion"

    def is_companion_mode(self) -> bool:
        """Return True for companion mode."""
        return True

    @property
    def execution_mode(self) -> str:
        """Return 'companion' execution mode."""
        return "companion"

    def is_available(self) -> bool:
        """Check if companion provider is available.

        For companion mode, we assume it's always available since it doesn't
        require network connectivity or API keys.
        """
        return True

    def _call_llm(self, prompt: str) -> str:
        """Handle companion workflow with clipboard interaction.

        This method implements the clipboard-based workflow:
        1. Copy prompt to clipboard
        2. Wait for user to paste to AI tool and get response
        3. Get response from clipboard

        Args:
            prompt: The prompt to send to companion AI

        Returns:
            Response text from companion AI

        Raises:
            LLMError: If companion workflow fails

        """
        try:
            # Store the prompt for debugging/logging
            self._last_prompt = prompt
            self._clipboard_interactions += 1

            logger.info(
                "🤖 Starting companion workflow interaction #%d",
                self._clipboard_interactions,
            )

            # Import clipboard functionality
            clipboard = ClipboardManager()

            # Step 1: Copy prompt to clipboard
            logger.info("📋 Copying prompt to clipboard...")
            clipboard.copy_to_clipboard(prompt)

            # Step 2: Wait for user interaction
            logger.info(
                "⏳ Waiting for user to paste prompt to AI tool and copy response..."
            )
            response = clipboard.wait_for_response(
                prompt_instruction="Please paste the prompt to your AI tool and copy the response back to clipboard",
            )

            # Step 3: Validate response
            if not response or not response.strip():
                raise LLMError("No response received from companion AI")

            logger.info(
                "✅ Received response from companion AI (%d characters)", len(response)
            )
            return response.strip()

        except Exception as e:
            if not isinstance(e, LLMError):
                msg = f"Companion workflow error: {e}"
                raise LLMError(msg) from e
            raise

    def generate_with_schema(
        self,
        schema_class: type[BaseModel],
        context_data: dict[str, Any],
        instruction: str | None = None,
        max_retries: int | None = None,
    ) -> BaseModel:
        """Generate and validate LLM response using companion clipboard workflow.

        This method follows the unified pipeline (steps B→H) but differs in step I
        by using clipboard interaction instead of direct API calls.

        Args:
            schema_class: Pydantic model class for response validation
            context_data: Context data to include in prompt
            instruction: Instruction text for the LLM
            max_retries: Maximum number of retries (not used in companion mode)

        Returns:
            Validated Pydantic model instance

        Raises:
            LLMError: If companion workflow or validation fails

        """
        if instruction is None:
            instruction = f"Generate {schema_class.__name__} analysis"

        # Check if we have a pre-rendered prompt (for compatibility with existing handlers)
        if "rendered_prompt" in context_data:
            prompt = context_data["rendered_prompt"]
            logger.debug("Using pre-rendered prompt from context_data")
        else:
            # Steps B→H: Use unified pipeline for prompt generation
            processed_context = self._preprocess_context(context_data)
            prompt = self._build_optimized_prompt(
                schema_class, processed_context, instruction
            )

        # Step I: Execute companion workflow instead of API call
        response_text = self._call_llm(prompt)

        # Steps J→K: Parse and validate response (shared with API mode)
        try:
            response_data = parse_json_response(response_text)
            validated_instance = schema_class(**response_data)
        except Exception as e:
            msg = f"Failed to validate companion response for {schema_class.__name__}: {e}"
            raise LLMError(msg) from e
        else:
            logger.info(
                "Successfully generated %s with %s fields via companion workflow",
                schema_class.__name__,
                len(response_data),
            )
            return validated_instance

    def get_token_usage(self) -> dict[str, Any]:
        """Get token usage statistics for companion mode.

        Since companion mode doesn't have direct token tracking,
        we return interaction-based metrics.
        """
        return {
            "total_interactions": self._clipboard_interactions,
            "last_prompt_length": len(self._last_prompt) if self._last_prompt else 0,
            "provider": "companion",
            "mode": "clipboard_based",
        }

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about the companion provider configuration."""
        return {
            "provider_name": self.provider_name,
            "execution_mode": self.execution_mode,
            "is_companion_mode": self.is_companion_mode(),
            "model": self.model,
            "total_interactions": self._clipboard_interactions,
            "is_available": self.is_available(),
        }


def _register_provider() -> None:
    """Register the CompanionLLMProvider when module is imported."""
    try:
        ProviderRegistry.register("companion", CompanionLLMProvider)
        logger.debug("Registered companion provider")
    except ValueError:
        # Provider already registered, ignore
        pass


_register_provider()
