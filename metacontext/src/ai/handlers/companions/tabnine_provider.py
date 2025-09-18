"""TabNine AI code companion provider."""

import contextlib

from src.ai.handlers.companions.base_companion import BaseCodeCompanionProvider
from src.ai.handlers.core.provider_registry import ProviderRegistry


class TabnineProvider(BaseCodeCompanionProvider):
    """Provider for TabNine AI integration."""

    @property
    def companion_name(self) -> str:
        """Return the human-readable name of the companion."""
        return "TabNine AI"

    @property
    def companion_shortcuts(self) -> str:
        """Return the keyboard shortcuts for this companion."""
        return "Use TabNine chat interface"

    @property
    def cli_commands(self) -> list[str] | None:
        """Return the CLI command to check availability."""
        return None  # No CLI available for TabNine


# Auto-register when imported
def _register_provider() -> None:
    """Register the TabnineProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("tabnine", TabnineProvider)


_register_provider()
