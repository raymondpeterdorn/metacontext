"""GitHub Copilot code companion provider."""

import contextlib
import logging
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ValidationError

from metacontext.ai.handlers.companions.base_companion import BaseCodeCompanionProvider
from metacontext.ai.handlers.core.exceptions import LLMError, ValidationRetryError
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.provider_interface import parse_json_response
from metacontext.core.config import get_config

logger = logging.getLogger(__name__)


class CopilotProvider(BaseCodeCompanionProvider):
    """Provider for GitHub Copilot integration."""

    def __init__(self, model: str = "default", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the GitHub Copilot provider.

        Args:
            model: The model name/identifier
            **kwargs: Additional configuration options

        """
        # Store model (copilot doesn't use different models)
        self.model = model
        self._name = "copilot"
        # Keep kwargs for interface compatibility but don't use them
        _ = kwargs

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return self._name

    @property
    def companion_name(self) -> str:
        """Return the human-readable name of the companion."""
        return "GitHub Copilot"

    @property
    def companion_shortcuts(self) -> str:
        """Return the keyboard shortcuts for this companion."""
        return "Ctrl+I for GitHub Copilot Chat"

    @property
    def cli_commands(self) -> list[str] | None:
        """Return the CLI command to check availability."""
        return ["gh", "copilot", "--version"]

    def _call_llm(self, prompt: str) -> str:
        """Make the actual call to the GitHub Copilot CLI service."""
        response_dict = self._cli_interaction(prompt)
        if "error" in response_dict:
            raise RuntimeError(response_dict["error"])
        return str(response_dict.get("content", ""))

    def is_available(self) -> bool:
        """Check if GitHub Copilot CLI is available."""
        if self.cli_commands:
            try:
                subprocess.run(  # nosec B603
                    self.cli_commands,
                    capture_output=True,
                    text=True,
                    check=False,  # Don't raise on non-zero exit
                    timeout=5,
                )
            except (subprocess.TimeoutExpired, FileNotFoundError):
                return False
            else:
                return True
        return False

    def _cli_interaction(self, prompt: str) -> dict[str, Any]:
        """Handle CLI interaction with GitHub Copilot."""
        try:
            # Use gh copilot suggest command with shell target and output to file for non-interactive mode
            with tempfile.NamedTemporaryFile(mode="w+", suffix=".txt", delete=False) as f:
                temp_file = f.name

            cmd = ["gh", "copilot", "suggest", "-t", "shell", "-s", temp_file, prompt]

            # Note: The CLI will timeout because it's interactive, but it still writes to the output file
            # So we'll catch the timeout and read from the file anyway
            with contextlib.suppress(subprocess.TimeoutExpired):
                subprocess.run(  # nosec B603
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=10,  # Shorter timeout since we expect it to fail
                    check=False,  # Don't raise on non-zero exit
                )

            # Read the suggestion from the temp file (should be there even if command timed out)
            try:
                if Path(temp_file).exists():
                    with open(temp_file) as f:  # noqa: PTH123
                        suggestion = f.read().strip()
                    if suggestion:
                        return {"content": suggestion}

                return {"error": "No suggestion generated"}
            finally:
                # Clean up temp file
                with contextlib.suppress(OSError):
                    Path(temp_file).unlink()

        except (
            FileNotFoundError,
            subprocess.CalledProcessError,
        ) as e:
            logger.exception("Copilot CLI interaction failed")
            return {"error": f"Copilot CLI failed: {e!s}"}

    def generate_with_schema(
        self,
        schema_class: type[BaseModel],
        context_data: dict[str, Any],
        instruction: str | None = None,
        max_retries: int | None = None,
    ) -> BaseModel:
        """Generate and validate LLM response using schema-first approach.

        Args:
            schema_class: Pydantic model class to validate against
            context_data: Data to include in the prompt context
            instruction: Optional instruction override
            max_retries: Maximum retry attempts for validation failures

        Returns:
            Validated instance of schema_class

        Raises:
            ValidationRetryError: If validation fails after max retries
            LLMError: If LLM call fails

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
        """Get token usage statistics.

        Returns:
            Dictionary with token usage information. Since GitHub Copilot doesn't
            provide token usage data, returns placeholder information.

        """
        return {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "provider": self.provider_name,
            "note": "GitHub Copilot does not provide token usage statistics",
        }

    def get_provider_info(self) -> dict[str, Any]:
        """Get information about the current provider configuration.

        Returns:
            Dictionary containing provider information for compatibility with LLM interface.

        """
        return {
            "provider": self.provider_name,
            "model": self.model,
            "available": self.is_available(),
            "temperature": 0.0,  # Copilot doesn't use temperature
            "type": "companion",
        }

    @classmethod
    def create(
        cls, model: str = "default", **kwargs: dict[str, Any],
    ) -> "CopilotProvider":
        """Create a new CopilotProvider instance.

        Args:
            model: Model identifier (for compatibility)
            **kwargs: Additional configuration

        Returns:
            Configured CopilotProvider instance

        """
        return cls(model=model, **kwargs)


# Auto-register when imported
def _register_provider() -> None:
    """Register the CopilotProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("copilot", CopilotProvider)


_register_provider()
