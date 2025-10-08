"""IDE integration utilities for companion providers.

This module provides cross-platform utilities for detecting and interacting
with IDEs, with a focus on VS Code integration for GitHub Copilot.
"""

import logging
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

try:
    import pyperclip

    PYPERCLIP_AVAILABLE = True
except ImportError:
    PYPERCLIP_AVAILABLE = False

logger = logging.getLogger(__name__)


def detect_vscode() -> dict[str, Any]:
    """Check if VS Code is running and available.

    Phase 3.2 Implementation: VS Code detection across platforms

    Returns:
        Dictionary with VS Code detection information

    """
    detection_info = {
        "available": False,
        "running": False,
        "command_available": False,
        "version": None,
        "extensions": [],
    }

    # Check if 'code' command is available
    detection_info["command_available"] = shutil.which("code") is not None

    if detection_info["command_available"]:
        try:
            # Get VS Code version
            result = subprocess.run(
                ["code", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            if result.returncode == 0:
                detection_info["available"] = True
                detection_info["version"] = result.stdout.strip().split("\n")[0]

            # Get installed extensions (limited list for performance)
            result = subprocess.run(
                ["code", "--list-extensions"],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
            if result.returncode == 0:
                extensions = [
                    ext.strip() for ext in result.stdout.split("\n") if ext.strip()
                ]
                detection_info["extensions"] = extensions[:20]  # Limit for performance

        except (subprocess.TimeoutExpired, OSError) as e:
            logger.debug("VS Code detection failed: %s", e)

    # Check if VS Code is running (platform-specific)
    detection_info["running"] = _is_vscode_running()

    return detection_info


def _is_vscode_running() -> bool:
    """Check if VS Code is currently running.

    Returns:
        True if VS Code process is detected

    """
    try:
        system = platform.system().lower()

        if system == "darwin":  # macOS
            result = subprocess.run(
                ["pgrep", "-f", "Visual Studio Code"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0

        if system == "linux":
            result = subprocess.run(
                ["pgrep", "-f", "code"],
                capture_output=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0

        if system == "windows":
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq Code.exe"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            return "Code.exe" in result.stdout if result.returncode == 0 else False

    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Process detection failed: %s", e)

    return False


def copy_to_clipboard(text: str) -> bool:
    """Cross-platform clipboard automation.

    Phase 3.2 Implementation: Optional clipboard automation

    Args:
        text: Text to copy to clipboard

    Returns:
        True if successfully copied, False otherwise

    """
    if not PYPERCLIP_AVAILABLE:
        logger.debug("pyperclip not available for clipboard operations")
        return False

    try:
        pyperclip.copy(text)
        logger.info("Text copied to clipboard successfully")
        return True
    except (OSError, RuntimeError) as e:
        logger.debug("Clipboard operation failed: %s", e)
        return False


def focus_vscode() -> bool:
    """Bring VS Code to front across platforms.

    Phase 3.2 Implementation: Cross-platform VS Code focus

    Returns:
        True if VS Code was successfully focused, False otherwise

    """
    try:
        system = platform.system().lower()

        if system == "darwin":  # macOS
            # Use AppleScript to focus VS Code
            script = """
                tell application "Visual Studio Code"
                    activate
                end tell
            """
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True,
                timeout=5,
                check=False,
            )
            return result.returncode == 0

        if system == "linux":
            # Try wmctrl first, then xdotool as fallback
            focus_commands = [
                ["wmctrl", "-a", "Visual Studio Code"],
                ["xdotool", "search", "--name", "Visual Studio Code", "windowactivate"],
            ]

            for cmd in focus_commands:
                if shutil.which(cmd[0]):
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        timeout=5,
                        check=False,
                    )
                    if result.returncode == 0:
                        return True

        elif system == "windows":
            # Use PowerShell to focus VS Code
            script = """
                $proc = Get-Process -Name "Code" -ErrorAction SilentlyContinue
                if ($proc) {
                    $proc.MainWindowHandle | ForEach-Object { 
                        [Microsoft.VisualBasic.Interaction]::AppActivate($_) 
                    }
                }
            """
            result = subprocess.run(
                ["powershell", "-Command", script],
                capture_output=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0

    except (subprocess.TimeoutExpired, OSError) as e:
        logger.debug("Focus operation failed: %s", e)

    return False


def create_response_file(
    target_file: Path,
    content_hint: str = "",
    suffix: str = "_response.yaml",
) -> Path:
    """Create a temporary response file for user responses.

    Phase 3.2 Implementation: Response file management

    Args:
        target_file: The file being analyzed
        content_hint: Hint about expected content structure
        suffix: File suffix for the response file

    Returns:
        Path to the created response file

    """
    # Create response file in same directory as target
    response_dir = target_file.parent
    response_dir.mkdir(parents=True, exist_ok=True)

    # Generate response filename
    base_name = target_file.stem
    response_file = response_dir / f"{base_name}{suffix}"

    # Create file with helpful template if it doesn't exist
    if not response_file.exists():
        template_content = _generate_response_template(content_hint)
        try:
            response_file.write_text(template_content, encoding="utf-8")
            logger.info("Created response file template: %s", response_file)
        except OSError as e:
            logger.warning("Could not create response file template: %s", e)

    return response_file


def _generate_response_template(content_hint: str) -> str:
    """Generate a helpful template for response files.

    Args:
        content_hint: Hint about expected content

    Returns:
        YAML template content

    """
    template = f"""# Metacontext Companion Response
# {content_hint if content_hint else "Replace this with your companion analysis"}
#
# This file should contain valid YAML matching the metacontext schema.
# The companion (GitHub Copilot, etc.) should provide structured analysis here.
#
# Example structure:
# ai_interpretation: "Your analysis here"
# ai_confidence: "HIGH"  # HIGH, MEDIUM, LOW
# ai_domain_context: "Domain-specific context"
# usage_guidance: "How this data should be used"
# hidden_meaning: "Patterns or insights not immediately obvious"
# suspicious_patterns: []
# cross_references: {{}}
# detective_insights: "Forensic-level analysis insights"

# TODO: Replace this template with actual companion response
ai_interpretation: "Placeholder - replace with actual analysis"
ai_confidence: "LOW"
ai_domain_context: "Placeholder"
usage_guidance: "Placeholder"
hidden_meaning: "Placeholder"
suspicious_patterns: []
cross_references: {{}}
detective_insights: "Placeholder"
"""
    return template


def create_temp_response_file(prefix: str = "metacontext_response") -> Path:
    """Create a temporary response file for short-term use.

    Phase 3.2 Utility: Temporary file management (defer file watchers to post-MVP)

    Args:
        prefix: Prefix for the temporary file name

    Returns:
        Path to the temporary response file

    """
    # Create temporary file with .yaml extension
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".yaml",
        prefix=f"{prefix}_",
        delete=False,
        encoding="utf-8",
    )

    temp_path = Path(temp_file.name)
    temp_file.close()

    logger.debug("Created temporary response file: %s", temp_path)
    return temp_path


def cleanup_response_file(file_path: Path, keep_on_error: bool = True) -> bool:
    """Clean up temporary response files.

    Args:
        file_path: Path to the response file to clean up
        keep_on_error: Whether to keep file if cleanup fails

    Returns:
        True if successfully cleaned up, False otherwise

    """
    try:
        if file_path.exists():
            file_path.unlink()
            logger.debug("Cleaned up response file: %s", file_path)
            return True
    except OSError as e:
        logger.warning("Could not clean up response file %s: %s", file_path, e)
        if not keep_on_error:
            # Force cleanup attempt
            try:
                file_path.unlink(missing_ok=True)
            except OSError:
                pass

    return False


def get_ide_environment_info() -> dict[str, Any]:
    """Get comprehensive IDE environment information.

    Phase 3.2 Diagnostic: Environment information gathering

    Returns:
        Dictionary with IDE environment details

    """
    env_info = {
        "platform": platform.system(),
        "platform_version": platform.version(),
        "python_version": platform.python_version(),
        "vscode": detect_vscode(),
        "clipboard_available": PYPERCLIP_AVAILABLE,
        "focus_capabilities": {},
    }

    # Test focus capabilities per platform
    system = platform.system().lower()
    if system == "darwin":
        env_info["focus_capabilities"]["method"] = "applescript"
        env_info["focus_capabilities"]["available"] = (
            shutil.which("osascript") is not None
        )
    elif system == "linux":
        env_info["focus_capabilities"]["method"] = "wmctrl/xdotool"
        env_info["focus_capabilities"]["available"] = (
            shutil.which("wmctrl") is not None or shutil.which("xdotool") is not None
        )
    elif system == "windows":
        env_info["focus_capabilities"]["method"] = "powershell"
        env_info["focus_capabilities"]["available"] = (
            shutil.which("powershell") is not None
        )

    return env_info


# Simplified automation interface (defer complex file watching to post-MVP)
class IDEAutomation:
    """Simplified IDE automation interface for companion providers.

    Phase 3.2: Simplified automation without complex file watching
    """

    def __init__(self) -> None:
        """Initialize IDE automation."""
        self.env_info = get_ide_environment_info()

    def is_ide_available(self, ide_name: str = "vscode") -> bool:
        """Check if specified IDE is available.

        Args:
            ide_name: Name of IDE to check (currently only 'vscode' supported)

        Returns:
            True if IDE is available

        """
        if ide_name.lower() == "vscode":
            return self.env_info["vscode"]["available"]
        return False

    def prepare_workspace(self, target_file: Path) -> dict[str, Any]:
        """Prepare workspace for companion analysis.

        Args:
            target_file: File to be analyzed

        Returns:
            Dictionary with workspace preparation results

        """
        result = {
            "target_file": str(target_file),
            "response_file": None,
            "workspace_ready": False,
        }

        try:
            # Create response file
            response_file = create_response_file(target_file)
            result["response_file"] = str(response_file)

            # Check if we can focus IDE
            if self.env_info["vscode"]["available"]:
                result["workspace_ready"] = True

            logger.info("Workspace prepared for analysis of %s", target_file)

        except Exception as e:
            logger.error("Workspace preparation failed: %s", e)

        return result
