"""Codeium AI Assistant code companion provider."""

import contextlib
from typing import Any

from src.ai.handlers.companions.base_companion import BaseCodeCompanionProvider
from src.ai.handlers.core.provider_registry import ProviderRegistry


class CodeiumProvider(BaseCodeCompanionProvider):
    """Provider for Codeium AI Assistant integration."""

    @property
    def companion_name(self) -> str:
        """Return the human-readable name of the companion."""
        return "Codeium AI Assistant"

    @property
    def companion_shortcuts(self) -> str:
        """Return the keyboard shortcuts for this companion."""
        return "Use Codeium chat interface"

    @property
    def cli_commands(self) -> list[str] | None:
        """Return the CLI command to check availability."""
        return ["codeium", "--version"]

    @classmethod
    def create(
        cls, model: str = "default", **kwargs: dict[str, Any],
    ) -> "CodeiumProvider":
        """Create a new CodeiumProvider instance.

        Args:
            model: Model identifier (for compatibility)
            **kwargs: Additional configuration

        Returns:
            Configured CodeiumProvider instance

        """
        return cls(model=model, **kwargs)


# Auto-register when imported
def _register_provider() -> None:
    """Register the CodeiumProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("codeium", CodeiumProvider)


_register_provider()
