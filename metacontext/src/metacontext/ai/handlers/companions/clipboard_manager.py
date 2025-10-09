"""Simple clipboard manager for companion LLM workflow.

This module provides a simplified clipboard interface specifically designed
for the unified LLM provider architecture.
"""

import logging
from typing import Any

try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

logger = logging.getLogger(__name__)


class ClipboardManager:
    """Simplified clipboard manager for companion LLM workflow."""

    def __init__(self) -> None:
        """Initialize clipboard manager."""
        self.clipboard_available = CLIPBOARD_AVAILABLE
        if not self.clipboard_available:
            logger.warning("pyperclip not available - clipboard functionality limited")

    def copy_to_clipboard(self, text: str) -> bool:
        """Copy text to system clipboard.

        Args:
            text: Text to copy to clipboard

        Returns:
            True if successful, False otherwise

        """
        if not self.clipboard_available:
            logger.error("Clipboard not available - pyperclip not installed")
            return False

        try:
            pyperclip.copy(text)
        except (OSError, RuntimeError):
            logger.exception("Failed to copy to clipboard")
            return False
        else:
            logger.info("✅ Prompt copied to clipboard (%d characters)", len(text))
            return True

    def wait_for_response(self, prompt_instruction: str) -> str:
        """Wait for user to provide response via clipboard.

        This method implements a simple user interaction workflow:
        1. Display instructions to user
        2. Wait for user to copy response to clipboard
        3. Return the response

        Args:
            prompt_instruction: Instructions to display to user

        Returns:
            Response text from clipboard

        """
        print("\n" + "=" * 80)  # noqa: T201
        print("🤖 COMPANION AI WORKFLOW")  # noqa: T201
        print("=" * 80)  # noqa: T201
        print(prompt_instruction)  # noqa: T201
        print("\nSteps:")  # noqa: T201
        print("1. ✅ Prompt is already copied to your clipboard")  # noqa: T201
        print("2. 📋 Paste it into your AI tool (ChatGPT, Claude, etc.)")  # noqa: T201
        print("3. 🤖 Get the AI response")  # noqa: T201
        print("4. 📋 Copy the AI response to your clipboard")  # noqa: T201
        print("5. ⏎  Press Enter here when response is copied")  # noqa: T201
        print("=" * 80)  # noqa: T201

        # Wait for user to press Enter
        input("\nPress Enter when you have copied the AI response to clipboard...")

        # Get response from clipboard
        if not self.clipboard_available:
            # Fallback: prompt user to paste response
            print("\n📝 Please paste the AI response below and press Enter:")  # noqa: T201
            return input().strip()

        try:
            response = pyperclip.paste()
            if not response or not response.strip():
                # Fallback: prompt user to paste response
                print(
                    "\n⚠️  No content found in clipboard. Please paste the response below:"
                )  # noqa: T201
                response = input().strip()
            else:
                logger.info(
                    "✅ Retrieved response from clipboard (%d characters)",
                    len(response),
                )
            return response.strip()
        except (OSError, RuntimeError):
            logger.exception("Failed to read from clipboard")
            # Fallback: prompt user to paste response
            print("\n⚠️  Clipboard error occurred")  # noqa: T201
            print("Please paste the AI response below:")  # noqa: T201
            return input().strip()

    def get_clipboard_status(self) -> dict[str, Any]:
        """Get clipboard manager status information.

        Returns:
            Status information dictionary

        """
        return {
            "clipboard_available": self.clipboard_available,
            "pyperclip_available": CLIPBOARD_AVAILABLE,
        }
