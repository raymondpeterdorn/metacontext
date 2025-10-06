"""Generic code companion provider for unknown companion types."""

import contextlib

from metacontext.ai.handlers.companions.base_companion import BaseCodeCompanionProvider
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry


class GenericProvider(BaseCodeCompanionProvider):
    """Generic provider for unknown or custom code companions."""

    def __init__(
        self,
        model: str = "generic",
        companion_name: str = "Code Companion",
    ) -> None:
        """Initialize the generic provider with custom companion name."""
        super().__init__(model=model)
        self._companion_name = companion_name

    @property
    def companion_name(self) -> str:
        """Return the human-readable name of the companion."""
        return self._companion_name

    @property
    def companion_shortcuts(self) -> str:
        """Return the keyboard shortcuts for this companion."""
        return f"Use {self._companion_name} interface"

    @property
    def cli_commands(self) -> list[str] | None:
        """Return the CLI command to check availability."""
        return None  # No CLI available for generic companions


# Auto-register when imported
def _register_provider() -> None:
    """Register the GenericProvider when module is imported."""
    with contextlib.suppress(ImportError, ValueError):
        ProviderRegistry.register("generic", GenericProvider)


_register_provider()
