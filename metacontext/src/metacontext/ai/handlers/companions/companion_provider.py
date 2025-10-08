"""Base companion provider interface for IDE-integrated code companions."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import pyperclip

    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False


@dataclass
class TemplateContext:
    """Type-safe container for template generation parameters."""

    file_path: Path
    file_type: str  # "tabular", "model", "media", "geospatial"
    semantic_knowledge: str = ""
    project_context: dict[str, Any] | None = None
    deterministic_metadata: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for template variable substitution."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_path.name,
            "file_type": self.file_type,
            "semantic_knowledge": self.semantic_knowledge,
            "project_context": self.project_context or {},
            "deterministic_metadata": self.deterministic_metadata or {},
        }


class BaseCompanionProvider(ABC):
    """Abstract base class for IDE-integrated code companion providers."""

    def __init__(self, companion_type: str) -> None:
        """Initialize the companion provider.

        Args:
            companion_type: Type of companion (e.g., "github_copilot", "cursor")

        """
        self.companion_type = companion_type
        self._availability_cache: bool | None = None

    @abstractmethod
    def is_available(self) -> bool:
        """Check if the companion is available and properly configured.

        This combines detection and availability checking in one method.
        Results are cached to avoid repeated expensive checks.

        Returns:
            True if companion is available and ready to use

        """

    @abstractmethod
    def generate_prompt(self, context: TemplateContext) -> str:
        """Generate a companion-friendly prompt from template context.

        Args:
            context: Template context with all necessary parameters

        Returns:
            Formatted prompt string ready for companion chat

        """

    @abstractmethod
    def wait_for_response(self, response_file_path: Path) -> dict[str, Any] | None:
        """Wait for and parse companion response.

        Args:
            response_file_path: Path where user should save companion response

        Returns:
            Parsed response data or None if failed/cancelled

        """

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard if available.

        Args:
            text: Text to copy to clipboard

        Returns:
            True if successfully copied, False otherwise

        """
        if not PYPERCLIP_AVAILABLE:
            return False

        try:
            pyperclip.copy(text)
        except (OSError, RuntimeError):
            return False
        else:
            return True

    def create_response_instructions(self, response_file_path: Path) -> str:
        """Generate user instructions for saving companion response.

        Args:
            response_file_path: Path where response should be saved

        Returns:
            Formatted instructions for user

        """
        return f"""
📋 **RESPONSE INSTRUCTIONS**

After the companion provides its analysis:

1. **Copy the entire response** from the companion chat
2. **Save it to this file**: `{response_file_path}`
3. **Make sure it's valid YAML** format
4. **Return to terminal** - metacontext will automatically process it

The response file should contain the analysis in the exact structure requested in the prompt above.
"""

    def get_companion_info(self) -> dict[str, Any]:
        """Get information about this companion provider.

        Returns:
            Dictionary with companion metadata

        """
        return {
            "type": self.companion_type,
            "available": self.is_available(),
            "clipboard_support": PYPERCLIP_AVAILABLE,
        }
