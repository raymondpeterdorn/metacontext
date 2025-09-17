"""Cursor AI Editor code companion provider."""

import contextlib
import logging
import subprocess
from typing import Any

from metacontext.ai.handlers.companions.base_companion import BaseCodeCompanionProvider
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry

logger = logging.getLogger(__name__)


class CursorProvider(BaseCodeCompanionProvider):
    """Provider for Cursor AI Editor integration."""

    @property
    def companion_name(self) -> str:
        """Return the human-readable name of the companion."""
        return "Cursor AI Editor"

    @property
    def companion_shortcuts(self) -> str:
        """Return the keyboard shortcuts for this companion."""
        return "Ctrl+K for Cursor Composer or Ctrl+L for Chat"

    @property
    def cli_commands(self) -> list[str] | None:
        """Return the CLI command to check availability."""
        return ["cursor", "--version"]

    def _cli_interaction(self, prompt: str) -> dict[str, Any]:
        """Handle CLI interaction with Cursor."""
        try:
            # Use cursor --chat command (if available)
            cmd = ["cursor", "--chat", "--prompt", prompt]
            result = subprocess.run(  # nosec B603
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,
            )
            return {"content": result.stdout.strip()}
        except (
            subprocess.TimeoutExpired,
            FileNotFoundError,
            subprocess.CalledProcessError,
        ) as e:
            logger.exception("Cursor CLI interaction failed")
            return {"error": f"Cursor CLI failed: {e!s}"}

    @classmethod
    def create(
        cls, model: str = "default", **kwargs: dict[str, Any],
    ) -> "CursorProvider":
        """Create a new CursorProvider instance.

        Args:
            model: Model identifier (for compatibility)
            **kwargs: Additional configuration

        Returns:
            Configured CursorProvider instance

        """
        return cls(model=model, **kwargs)


# Auto-register when imported
def _register_provider() -> None:
    """Register the CursorProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("cursor", CursorProvider)


_register_provider()
