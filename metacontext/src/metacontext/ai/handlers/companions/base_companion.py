"""Base code companion provider for editor-integrated AI assistants.

This module provides the base class for all code companion providers.
"""

import logging
import shlex
import subprocess
import tempfile
import time
from abc import abstractmethod
from pathlib import Path
from typing import Any

from metacontext.ai.handlers.llms.provider_interface import AbstractLLMProvider

logger = logging.getLogger(__name__)


class BaseCodeCompanionProvider(AbstractLLMProvider):
    """Base provider for code companions that integrate with editors.

    This provider facilitates interaction with AI assistants that are built
    into code editors rather than accessed via HTTP APIs. It provides both
    CLI integration and manual workflow fallback.
    """

    @property
    @abstractmethod
    def companion_name(self) -> str:
        """Return the human-readable name of the companion."""

    @property
    @abstractmethod
    def companion_shortcuts(self) -> str:
        """Return the keyboard shortcuts for this companion."""

    @property
    @abstractmethod
    def cli_commands(self) -> list[str] | None:
        """Return the CLI command to check availability, or None if no CLI."""

    def __init__(self, model: str = "default", **kwargs: Any) -> None:  # noqa: ANN401
        """Initialize the code companion provider.

        Args:
            model: The model name/identifier
            **kwargs: Additional configuration options

        """
        # Code companions don't need API keys
        api_key = kwargs.pop("api_key", None)
        temperature = kwargs.pop("temperature", 0.1)
        if not isinstance(temperature, float):
            temperature = 0.1
        super().__init__(model=model, api_key=api_key, temperature=temperature)

    @property
    def provider_name(self) -> str:
        """Return the name of the provider."""
        return "code_companion"

    def _initialize_sync_client(self) -> None:
        """Initialize synchronous client (not needed for code companions)."""
        return

    async def _get_async_client(self) -> None:
        """Get async client (not needed for code companions)."""
        return

    def _call_sync_llm(self, prompt: str) -> str:
        """Make synchronous call to code companion."""
        # Get full response and extract content
        response_dict = self._get_companion_response(prompt)
        return response_dict.get("content", "")

    async def _call_async_llm(self, prompt: str) -> str:
        """Make asynchronous call to code companion."""
        # Code companions are inherently synchronous (manual interaction)
        return self._call_sync_llm(prompt)

    def is_available(self) -> bool:
        """Check if code companion is available (always True for manual interaction)."""
        return True

    def get_available_models(self) -> list[str]:
        """Get list of available models for the code companion."""
        # Code companions don't have a queryable model list
        return [self.model]

    def check_health(self) -> dict[str, Any]:
        """Check if the code companion is available."""
        try:
            # Check if the companion CLI is available
            is_available = self._check_companion_availability()
        except OSError:
            logger.exception("Health check failed")
            return {"type": "manual", "companion": self.companion_name}
        else:
            return {
                "status": "healthy" if is_available else "unavailable",
                "companion": self.companion_name,
            }

    def _check_companion_availability(self) -> bool:
        """Check if the code companion CLI is available."""
        if not self.cli_commands:
            return False

        # Security: Ensure commands are safe by splitting a hardcoded string.
        # This is safer than passing a list that might be constructed from user input.
        command_to_run = " ".join(self.cli_commands)
        safe_commands = ["gh", "cursor"]
        if self.cli_commands[0] not in safe_commands:
            logger.warning(
                "Attempted to run an untrusted command: %s", self.cli_commands,
            )
            return False

        try:
            # Use shell=False for security
            result = subprocess.run(  # nosec B603
                shlex.split(command_to_run),
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        else:
            return result.returncode == 0

    def generate_completion(self, prompt: str) -> str:
        """Make a synchronous call to the code companion and return the completion.

        This is a public wrapper around the _call_sync_llm method to maintain compatibility
        with the BaseLLMProvider class.

        Args:
            prompt: The prompt to send to the code companion

        Returns:
            The text completion from the code companion

        """
        return self._call_sync_llm(prompt)

    def _get_companion_response(
        self,
        prompt: str,
        # Note: Previously had max_tokens parameter, removed to avoid lint warnings
    ) -> dict[str, Any]:
        """Generate a structured response from the code companion.

        This method first tries CLI interaction, then falls back to manual workflow.

        Args:
            prompt: The prompt to send to the code companion

        Returns:
            Dictionary containing response content, model info, and token usage estimates

        """
        # Check if CLI interaction is available
        if self._check_companion_availability():
            logger.info("Code companion CLI available for %s", self.companion_name)
            try:
                return self._cli_interaction(prompt)
            except OSError as e:
                logger.warning(
                    "CLI interaction failed for %s: %s, falling back to manual mode",
                    self.companion_name,
                    e,
                )
                # Rather than silently falling back, propagate a more specific exception
                # with the original exception context preserved
                msg = f"CLI interaction failed for {self.companion_name}, see logs for details"
                raise RuntimeError(msg) from e

        logger.info("Using manual interaction mode for %s", self.companion_name)
        return self._manual_interaction(prompt)

    @abstractmethod
    def _cli_interaction(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """Handle CLI interaction with the companion."""

    def _manual_interaction(
        self,
        prompt: str,
    ) -> dict[str, Any]:
        """Handle manual interaction with the code companion."""
        # Create temporary files for communication
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Generate unique filenames
            timestamp = int(time.time() * 1000)
            prompt_file = temp_path / f"prompt_{timestamp}.txt"
            response_file = temp_path / f"response_{timestamp}.txt"

            # Write the prompt to a file
            prompt_file.write_text(prompt, encoding="utf-8")

            # Generate instructions for the user
            self._generate_interaction_instructions(prompt_file, response_file)
            logger.info("User instructions generated for code companion")

            # Wait for user response
            response = self._wait_for_user_response(response_file)

            return {
                "content": response,
                "model": self.model,
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response.split()) if response else 0,
                    "total_tokens": len(prompt.split())
                    + (len(response.split()) if response else 0),
                },
            }

    def _generate_interaction_instructions(
        self, prompt_file: Path, response_file: Path,
    ) -> str:
        """Generate instructions for manual interaction."""
        # or write to a dedicated output panel
        return f"""
=== CODE COMPANION INTERACTION REQUIRED ===

A prompt has been prepared for your {self.companion_name}.

1. Prompt file: {prompt_file}
2. Please copy the prompt content and use it with your code companion
3. Shortcut: {self.companion_shortcuts}
4. After getting the response, save it to: {response_file}
5. The system will continue automatically once the response file is created

Press Enter when you've saved the response...
"""

    def _wait_for_user_response(self, response_file: Path) -> str:
        """Wait for the user to create a response file."""
        while not response_file.exists():
            time.sleep(1)

        # Give a moment for the file to be fully written
        time.sleep(0.5)

        try:
            content = response_file.read_text(encoding="utf-8")
            return content.strip()
        except OSError:
            logger.exception("Error reading response file")
            return ""

    @staticmethod
    def estimate_cost() -> float:
        """Estimate the cost of a completion.

        For code companions, we assume the cost is always zero as it's a local/free service.

        Returns:
            The estimated cost (always 0.0).

        """
        # Unused arguments are kept for signature compatibility.
        return 0.0
