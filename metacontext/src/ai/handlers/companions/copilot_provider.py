"""GitHub Copilot code companion provider."""

import contextlib
import logging
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from metacontext.ai.handlers.companions.base_companion import BaseCodeCompanionProvider
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry

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
        return response_dict.get("content", "")

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
