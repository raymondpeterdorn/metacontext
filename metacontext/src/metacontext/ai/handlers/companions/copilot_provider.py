"""GitHub Copilot companion provider implementation."""

import json
import logging
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import yaml

try:
    import pyperclip

    CLIPBOARD_AVAILABLE = True
except ImportError:
    CLIPBOARD_AVAILABLE = False

from metacontext.ai.handlers.companions.companion_provider import (
    BaseCompanionProvider,
    TemplateContext,
)

logger = logging.getLogger(__name__)


class GitHubCopilotProvider(BaseCompanionProvider):
    """GitHub Copilot companion provider with multi-method detection."""

    def __init__(self) -> None:
        """Initialize the GitHub Copilot provider."""
        super().__init__("github_copilot")

    def is_available(self) -> bool:
        """Check if GitHub Copilot is available through multiple detection methods.

        Returns:
            True if Copilot is available, False otherwise

        """
        detection_methods = [
            self._detect_gh_cli_copilot,
            self._detect_vscode_copilot,
            self._detect_github_cli_auth,
        ]

        for method in detection_methods:
            try:
                if method():
                    return True
            except Exception as e:  # noqa: BLE001
                # Continue to next detection method if this one fails
                logger.debug("Detection method failed: %s", e)
                continue

        return False

    def _detect_gh_cli_copilot(self) -> bool:
        """Detect GitHub Copilot via GitHub CLI.

        Returns:
            True if gh copilot command is available

        """
        # Check if gh CLI is available
        if not shutil.which("gh"):
            return False

        try:
            # Check if copilot extension is available
            result = subprocess.run(
                ["gh", "copilot", "--version"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def _detect_vscode_copilot(self) -> bool:
        """Detect VS Code with Copilot extension installed.

        Enhanced detection for macOS, Windows, and Linux including cases
        where 'code' command is not in PATH.

        Returns:
            True if VS Code with Copilot extension is detected

        """
        # Method 1: Try standard 'code' command
        if shutil.which("code"):
            try:
                result = subprocess.run(
                    ["code", "--list-extensions"],
                    capture_output=True,
                    text=True,
                    timeout=15,
                    check=False,
                )

                if result.returncode == 0:
                    extensions = result.stdout.lower()
                    copilot_extensions = [
                        "github.copilot",
                        "github.copilot-chat",
                        "github.copilot-labs",
                    ]
                    if any(ext in extensions for ext in copilot_extensions):
                        return True

            except (subprocess.TimeoutExpired, OSError):
                pass

        # Method 2: Platform-specific VS Code detection
        system = platform.system().lower()

        if system == "darwin":  # macOS
            return self._detect_vscode_macos()
        if system == "windows":
            return self._detect_vscode_windows()
        if system == "linux":
            return self._detect_vscode_linux()

        return False

    def _detect_vscode_macos(self) -> bool:
        """Detect VS Code on macOS using multiple methods."""
        try:
            # Check if VS Code app exists in Applications
            vscode_app_path = Path("/Applications/Visual Studio Code.app")
            if not vscode_app_path.exists():
                return False

            # Try to find extensions directory
            extensions_paths = [
                Path.home() / ".vscode" / "extensions",
                Path.home()
                / "Library"
                / "Application Support"
                / "Code"
                / "User"
                / "extensions",
            ]

            for ext_path in extensions_paths:
                if ext_path.exists():
                    # Look for Copilot extensions in directory names
                    extension_dirs = [
                        d.name.lower() for d in ext_path.iterdir() if d.is_dir()
                    ]
                    copilot_patterns = ["github.copilot", "copilot"]

                    if any(
                        pattern in ext_dir
                        for pattern in copilot_patterns
                        for ext_dir in extension_dirs
                    ):
                        return True

            # If VS Code exists but no extensions found, try alternative code command
            code_paths = [
                "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
                "/usr/local/bin/code",
            ]

            for code_path in code_paths:
                if Path(code_path).exists():
                    try:
                        result = subprocess.run(
                            [code_path, "--list-extensions"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            check=False,
                        )
                        if result.returncode == 0:
                            extensions = result.stdout.lower()
                            copilot_extensions = [
                                "github.copilot",
                                "github.copilot-chat",
                            ]
                            if any(ext in extensions for ext in copilot_extensions):
                                return True
                    except (subprocess.TimeoutExpired, OSError):
                        continue

        except Exception:
            pass

        return False

    def _detect_vscode_windows(self) -> bool:
        """Detect VS Code on Windows."""
        try:
            # Common VS Code installation paths on Windows
            possible_paths = [
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs"
                / "Microsoft VS Code"
                / "bin"
                / "code.cmd",
                Path(os.environ.get("PROGRAMFILES", ""))
                / "Microsoft VS Code"
                / "bin"
                / "code.cmd",
                Path(os.environ.get("PROGRAMFILES(X86)", ""))
                / "Microsoft VS Code"
                / "bin"
                / "code.cmd",
            ]

            for code_path in possible_paths:
                if code_path.exists():
                    try:
                        result = subprocess.run(
                            [str(code_path), "--list-extensions"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            check=False,
                        )
                        if result.returncode == 0:
                            extensions = result.stdout.lower()
                            copilot_extensions = [
                                "github.copilot",
                                "github.copilot-chat",
                            ]
                            if any(ext in extensions for ext in copilot_extensions):
                                return True
                    except (subprocess.TimeoutExpired, OSError):
                        continue

        except Exception:
            pass

        return False

    def _detect_vscode_linux(self) -> bool:
        """Detect VS Code on Linux."""
        try:
            # Common VS Code installation paths on Linux
            possible_paths = [
                "/usr/bin/code",
                "/usr/local/bin/code",
                "/snap/bin/code",
                "/opt/visual-studio-code/bin/code",
            ]

            for code_path in possible_paths:
                if Path(code_path).exists():
                    try:
                        result = subprocess.run(
                            [code_path, "--list-extensions"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            check=False,
                        )
                        if result.returncode == 0:
                            extensions = result.stdout.lower()
                            copilot_extensions = [
                                "github.copilot",
                                "github.copilot-chat",
                            ]
                            if any(ext in extensions for ext in copilot_extensions):
                                return True
                    except (subprocess.TimeoutExpired, OSError):
                        continue

        except Exception:
            pass

        return False

    def _detect_vscode(self) -> bool:
        """Detect VS Code installation without checking for Copilot extensions.

        Returns:
            True if VS Code is detected on the system

        """
        # Method 1: Check if 'code' command is available
        if shutil.which("code"):
            return True

        # Method 2: Platform-specific VS Code detection
        system = platform.system().lower()

        if system == "darwin":  # macOS
            return Path("/Applications/Visual Studio Code.app").exists()
        if system == "windows":
            # Common VS Code installation paths on Windows
            possible_paths = [
                Path(os.environ.get("LOCALAPPDATA", ""))
                / "Programs"
                / "Microsoft VS Code",
                Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft VS Code",
                Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft VS Code",
            ]
            return any(path.exists() for path in possible_paths)
        if system == "linux":
            # Common VS Code installation paths on Linux
            possible_paths = [
                Path("/usr/bin/code"),
                Path("/usr/local/bin/code"),
                Path("/snap/bin/code"),
                Path("/opt/visual-studio-code"),
            ]
            return any(path.exists() for path in possible_paths)

        return False

    def _detect_github_cli_auth(self) -> bool:
        """Check for authenticated GitHub CLI session.

        Returns:
            True if GitHub CLI is authenticated

        """
        if not shutil.which("gh"):
            return False

        try:
            result = subprocess.run(
                ["gh", "auth", "status"],
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
            # gh auth status returns 0 if authenticated
            return result.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    def generate_prompt(self, context: TemplateContext) -> str:
        """Generate a Copilot-optimized prompt from template context.

        Args:
            context: Template context with file and analysis parameters

        Returns:
            Formatted prompt for GitHub Copilot chat

        """
        context_dict = context.to_dict()

        prompt_parts = [
            "🔍 **METACONTEXT ANALYSIS REQUEST**",
            "",
            f"**File:** `{context_dict['file_name']}`",
            f"**Type:** {context_dict['file_type']} analysis",
            "",
            "📋 **INSTRUCTIONS:**",
            "",
            "1. **ANALYZE THE ENTIRE WORKSPACE** - examine all files, not just the target file",
            "2. **USE VS CODE'S CONTEXT** - leverage imports, references, and code intelligence",
            "3. **PROVIDE STRUCTURED OUTPUT** - save response as YAML in exact metacontext format",
            "",
            "📊 **ANALYSIS SCOPE:**",
        ]

        # Add semantic knowledge if available
        if context_dict.get("semantic_knowledge"):
            prompt_parts.extend(
                [
                    "",
                    "🧠 **SEMANTIC CONTEXT:**",
                    f"{context_dict['semantic_knowledge']}",
                ],
            )

        # Add project context if available
        if context_dict.get("project_context"):
            prompt_parts.extend(
                [
                    "",
                    "🏗️ **PROJECT CONTEXT:**",
                    "```json",
                    f"{json.dumps(context_dict['project_context'], indent=2)}",
                    "```",
                ],
            )

        # Add deterministic metadata if available
        if context_dict.get("deterministic_metadata"):
            prompt_parts.extend(
                [
                    "",
                    "📈 **FILE METADATA:**",
                    "```json",
                    f"{json.dumps(context_dict['deterministic_metadata'], indent=2)}",
                    "```",
                ],
            )

        prompt_parts.extend(
            [
                "",
                "⚡ **KEY REQUIREMENTS:**",
                "- Examine ALL workspace files for relationships and context",
                "- Use cross-file analysis to understand data flow and dependencies",
                "- Provide forensic-level analysis with investigative depth",
                "- Output must be valid YAML matching metacontext schema exactly",
                "- Include detailed reasoning and evidence for all findings",
                "",
                "💾 **RESPONSE FORMAT:**",
                "Save your complete analysis as `metacontext_response.yaml` with proper YAML structure.",
            ],
        )

        return "\n".join(prompt_parts)

    def display_prompt_and_wait(
        self,
        prompt: str,
        response_file_path: Path,
    ) -> dict[str, Any] | None:
        """Display a ready-made prompt and wait for user response.

        Args:
            prompt: Ready-made prompt string to display
            response_file_path: Path where response should be saved

        Returns:
            Parsed response data or None if failed/cancelled

        """
        # Try to copy prompt to clipboard
        clipboard_success = False
        if CLIPBOARD_AVAILABLE:
            try:
                pyperclip.copy(prompt)
                clipboard_success = True
            except Exception:
                logger.warning("Failed to copy prompt to clipboard")

        print("\n" + "=" * 80)
        print("🤖 GITHUB COPILOT PROMPT")
        print("=" * 80)

        if clipboard_success:
            print(
                "✅ Prompt copied to clipboard-- paste in chat and send to generate metacontext",
            )
            print("=" * 80)
        else:
            print("⚠️  Clipboard not available - copy the prompt below manually")
            print("=" * 80)
            print(prompt)
            print("=" * 80)

        return self.wait_for_response(response_file_path)

    def wait_for_response(self, response_file_path: Path) -> dict[str, Any] | None:
        """Wait for user to provide companion response via file.

        Args:
            response_file_path: Path where response should be saved

        Returns:
            Parsed response data or None if failed/cancelled

        """
        import yaml

        print("\n" + "=" * 60)
        print("🤖 GITHUB COPILOT WORKFLOW")
        print("=" * 60)
        print(f"📁 Response file: {response_file_path}")
        print("\n📋 NEXT STEPS:")
        print(
            "1. The prompt is copied to your clipboard. Paste & send in GitHub Copilot Chat",
        )
        print("2. Save Copilot's response to the file path shown above")
        print("3. Press Enter when ready...")

        # Wait for user acknowledgment before starting to poll
        try:
            input("Press Enter to continue...")
        except KeyboardInterrupt:
            print("\n❌ Operation cancelled by user")
            return None

        print("\n⏱️  Waiting for your response file...")

        # Simple polling approach - defer watchdog to post-MVP
        max_wait_time = 300  # 5 minutes
        poll_interval = 2  # Check every 2 seconds
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            if response_file_path.exists():
                try:
                    with response_file_path.open(encoding="utf-8") as f:
                        response_data = yaml.safe_load(f)
                    print(f"✅ Response received from {response_file_path}")
                    return response_data
                except (yaml.YAMLError, OSError) as e:
                    print(f"⚠️  Error reading response file: {e}")
                    print("Please check the YAML format and try again...")

            time.sleep(poll_interval)
            elapsed_time += poll_interval

            # Show progress every 30 seconds
            if elapsed_time % 30 == 0:
                remaining = max_wait_time - elapsed_time
                print(f"⏳ Still waiting... ({remaining}s remaining)")

        print("⏰ Timeout reached. Response collection cancelled.")
        return None

    def get_detection_info(self) -> dict[str, Any]:
        """Get detailed information about Copilot detection methods.

        Returns:
            Dictionary with detection method results

        """
        detection_results = {
            "gh_cli_available": shutil.which("gh") is not None,
            "gh_copilot_available": False,
            "vscode_available": False,
            "vscode_copilot_extensions": [],
            "github_cli_authenticated": False,
        }

        # Test gh copilot specifically
        try:
            detection_results["gh_copilot_available"] = self._detect_gh_cli_copilot()
        except Exception:  # noqa: BLE001
            pass

        # Test VS Code with enhanced detection
        try:
            detection_results["vscode_available"] = self._detect_vscode_copilot()

            # If VS Code with Copilot is detected, extract extension info
            if detection_results["vscode_available"]:
                # Try to get extension list using enhanced detection
                system = platform.system().lower()
                if system == "darwin":
                    extensions_path = Path.home() / ".vscode" / "extensions"
                    if extensions_path.exists():
                        extension_dirs = [
                            d.name for d in extensions_path.iterdir() if d.is_dir()
                        ]
                        copilot_extensions = [
                            ext for ext in extension_dirs if "copilot" in ext.lower()
                        ]
                        detection_results["vscode_copilot_extensions"] = (
                            copilot_extensions
                        )
                else:
                    # For other platforms, try the command line approach
                    code_path = shutil.which("code")
                    if code_path:
                        result = subprocess.run(
                            [code_path, "--list-extensions"],
                            capture_output=True,
                            text=True,
                            timeout=15,
                            check=False,
                        )
                        if result.returncode == 0:
                            extensions = [
                                ext.strip()
                                for ext in result.stdout.split("\n")
                                if ext.strip() and "copilot" in ext.lower()
                            ]
                            detection_results["vscode_copilot_extensions"] = extensions
        except Exception:  # noqa: BLE001
            pass

        # Test GitHub CLI auth
        try:
            detection_results["github_cli_authenticated"] = (
                self._detect_github_cli_auth()
            )
        except Exception:  # noqa: BLE001
            pass

        return detection_results

    def send_to_copilot(self, prompt: str, clipboard_fallback: bool = True) -> bool:
        """Send prompt to Copilot via clipboard automation with fallback.

        Phase 3.1 Implementation: Clipboard automation with fallback for systems without VS Code

        Args:
            prompt: The generated prompt to send
            clipboard_fallback: Whether to use clipboard if direct sending fails

        Returns:
            True if prompt was successfully sent/copied, False otherwise

        """
        # Try clipboard automation first
        if clipboard_fallback and self.copy_to_clipboard(prompt):
            print("✅ Prompt copied to clipboard!")
            print("📋 Paste it into GitHub Copilot Chat in VS Code")
            return True

        # If clipboard fails, provide manual instructions
        print("⚠️  Clipboard automation unavailable")
        print("📝 Please manually copy the following prompt:")
        print("-" * 60)
        print(prompt)
        print("-" * 60)
        return False

    def parse_response(self, response_file_path: Path) -> dict[str, Any] | None:
        """Extract structured data from YAML response.

        Phase 3.1 Implementation: Parse and validate companion response

        Args:
            response_file_path: Path to the response file

        Returns:
            Parsed response dictionary or None if parsing failed

        """
        if not response_file_path.exists():
            logger.error("Response file not found: %s", response_file_path)
            return None

        try:
            with response_file_path.open(encoding="utf-8") as f:
                response_data = yaml.safe_load(f)

            if not isinstance(response_data, dict):
                logger.error("Response is not a valid dictionary")
                return None

            logger.info("Successfully parsed response from %s", response_file_path)
            return response_data

        except yaml.YAMLError as e:
            logger.error("YAML parsing error: %s", e)
            return None
        except (OSError, UnicodeDecodeError) as e:
            logger.error("File reading error: %s", e)
            return None

    def format_prompt_for_copilot_chat(self, context: TemplateContext) -> str:
        """Format prompts specifically for Copilot chat interface.

        Phase 3.1 Enhancement: Optimized formatting for Copilot chat

        Args:
            context: Template context with analysis parameters

        Returns:
            Copilot Chat optimized prompt

        """
        # Use the existing generate_prompt method but add Copilot-specific enhancements
        base_prompt = self.generate_prompt(context)

        # Add Copilot-specific chat formatting
        copilot_enhancements = [
            "",
            "🔧 **COPILOT CHAT SPECIFIC INSTRUCTIONS:**",
            "- Use @workspace to access the entire workspace context",
            "- Leverage VS Code's code intelligence features",
            "- Provide step-by-step reasoning for your analysis",
            "- Include code examples and references where relevant",
            "- Format the final output as valid YAML exactly as specified",
            "",
            "🎯 **SAVE YOUR RESPONSE:**",
            f"When complete, save your entire analysis to: `{context.file_path.parent}/metacontext_response.yaml`",
        ]

        return base_prompt + "\n" + "\n".join(copilot_enhancements)

    def create_response_file_path(self, target_file: Path) -> Path:
        """Create appropriate response file path for companion responses.

        Phase 3.1 Utility: Generate response file paths

        Args:
            target_file: The file being analyzed

        Returns:
            Path where companion response should be saved

        """
        # Save response in the same directory as target file
        response_dir = target_file.parent
        response_file = response_dir / "metacontext_response.yaml"

        # Ensure directory exists
        response_dir.mkdir(parents=True, exist_ok=True)

        return response_file

    def validate_copilot_environment(self) -> dict[str, Any]:
        """Validate the GitHub Copilot environment setup.

        Phase 3.1 Diagnostic: Environment validation

        Returns:
            Dictionary with validation results and recommendations

        """
        validation = {
            "overall_status": "unknown",
            "checks": {},
            "recommendations": [],
        }

        # Check GitHub CLI
        validation["checks"]["gh_cli"] = shutil.which("gh") is not None
        if not validation["checks"]["gh_cli"]:
            validation["recommendations"].append(
                "Install GitHub CLI: https://cli.github.com/",
            )

        # Check GitHub CLI authentication
        validation["checks"]["gh_auth"] = self._detect_github_cli_auth()
        if not validation["checks"]["gh_auth"]:
            validation["recommendations"].append(
                "Authenticate GitHub CLI: run 'gh auth login'",
            )

        # Check GitHub Copilot CLI
        validation["checks"]["gh_copilot"] = self._detect_gh_cli_copilot()
        if not validation["checks"]["gh_copilot"]:
            validation["recommendations"].append(
                "Install Copilot CLI: run 'gh extension install github/gh-copilot'",
            )

        # Check VS Code using enhanced detection
        validation["checks"]["vscode"] = self._detect_vscode()
        if not validation["checks"]["vscode"]:
            validation["recommendations"].append("Install VS Code")

        # Check VS Code Copilot extensions
        validation["checks"]["vscode_copilot"] = self._detect_vscode_copilot()
        if not validation["checks"]["vscode_copilot"]:
            validation["recommendations"].append(
                "Install GitHub Copilot extension in VS Code",
            )

        # Determine overall status
        critical_checks = ["gh_cli", "gh_auth", "gh_copilot"]
        if all(validation["checks"].get(check, False) for check in critical_checks):
            validation["overall_status"] = "ready"
        elif any(validation["checks"].get(check, False) for check in critical_checks):
            validation["overall_status"] = "partial"
        else:
            validation["overall_status"] = "not_ready"

        return validation

    def get_detailed_diagnostics(self) -> dict[str, Any]:
        """Get comprehensive diagnostics with troubleshooting information.

        This method provides detailed information about VS Code and Copilot
        detection failures, including platform-specific installation paths
        and step-by-step troubleshooting instructions.

        Returns:
            Dictionary with detailed diagnostic information and solutions

        """
        diagnostics = {
            "platform": platform.system(),
            "platform_version": platform.release(),
            "detection_results": {},
            "issues_found": [],
            "recommended_solutions": [],
            "troubleshooting_steps": [],
        }

        # Detailed VS Code detection
        vscode_details = self._diagnose_vscode()
        diagnostics["detection_results"]["vscode"] = vscode_details

        # Detailed GitHub CLI detection
        gh_details = self._diagnose_github_cli()
        diagnostics["detection_results"]["github_cli"] = gh_details

        # Detailed Copilot detection
        copilot_details = self._diagnose_copilot()
        diagnostics["detection_results"]["copilot"] = copilot_details

        # Generate issues and solutions based on detection results
        self._analyze_issues_and_solutions(diagnostics)

        return diagnostics

    def _diagnose_vscode(self) -> dict[str, Any]:
        """Comprehensive VS Code diagnostic information."""
        diagnosis = {
            "installed": False,
            "code_command_available": False,
            "installation_paths_checked": [],
            "extensions_directory": None,
            "copilot_extensions": [],
            "issues": [],
            "platform_specific_info": {},
        }

        # Check if 'code' command is available
        diagnosis["code_command_available"] = shutil.which("code") is not None

        system = platform.system().lower()

        if system == "darwin":  # macOS
            diagnosis["platform_specific_info"] = self._diagnose_vscode_macos(diagnosis)
        elif system == "windows":
            diagnosis["platform_specific_info"] = self._diagnose_vscode_windows(
                diagnosis,
            )
        elif system == "linux":
            diagnosis["platform_specific_info"] = self._diagnose_vscode_linux(diagnosis)

        # Check for extensions if VS Code is found
        if diagnosis["installed"]:
            self._check_vscode_extensions(diagnosis)

        return diagnosis

    def _diagnose_vscode_macos(self, diagnosis: dict[str, Any]) -> dict[str, Any]:
        """macOS-specific VS Code diagnostics."""
        macos_info = {
            "app_bundle_path": "/Applications/Visual Studio Code.app",
            "app_bundle_exists": False,
            "code_binary_path": "/Applications/Visual Studio Code.app/Contents/Resources/app/bin/code",
            "code_binary_exists": False,
            "installation_method": "unknown",
        }

        # Check standard installation
        app_path = Path(macos_info["app_bundle_path"])
        macos_info["app_bundle_exists"] = app_path.exists()
        diagnosis["installation_paths_checked"].append(str(app_path))

        if macos_info["app_bundle_exists"]:
            diagnosis["installed"] = True
            macos_info["installation_method"] = "standard_installer"

            # Check if code binary exists
            code_binary = Path(macos_info["code_binary_path"])
            macos_info["code_binary_exists"] = code_binary.exists()

            if (
                not diagnosis["code_command_available"]
                and macos_info["code_binary_exists"]
            ):
                diagnosis["issues"].append(
                    "VS Code installed but 'code' command not in PATH",
                )
        else:
            # Check homebrew installation
            homebrew_path = Path("/opt/homebrew/bin/code")
            if homebrew_path.exists():
                diagnosis["installed"] = True
                macos_info["installation_method"] = "homebrew"
                diagnosis["installation_paths_checked"].append(str(homebrew_path))
            else:
                diagnosis["issues"].append("VS Code not found in standard locations")

        return macos_info

    def _diagnose_vscode_windows(self, diagnosis: dict[str, Any]) -> dict[str, Any]:
        """Windows-specific VS Code diagnostics."""
        windows_info = {
            "user_installation_path": None,
            "system_installation_path": None,
            "installation_method": "unknown",
        }

        # Check common installation paths
        possible_paths = [
            Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Microsoft VS Code",
            Path(os.environ.get("PROGRAMFILES", "")) / "Microsoft VS Code",
            Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Microsoft VS Code",
        ]

        for path in possible_paths:
            diagnosis["installation_paths_checked"].append(str(path))
            if path.exists():
                diagnosis["installed"] = True
                if "LOCALAPPDATA" in str(path):
                    windows_info["user_installation_path"] = str(path)
                    windows_info["installation_method"] = "user_installer"
                else:
                    windows_info["system_installation_path"] = str(path)
                    windows_info["installation_method"] = "system_installer"
                break

        if not diagnosis["installed"]:
            diagnosis["issues"].append(
                "VS Code not found in standard Windows locations",
            )
        elif not diagnosis["code_command_available"]:
            diagnosis["issues"].append(
                "VS Code installed but 'code' command not in PATH",
            )

        return windows_info

    def _diagnose_vscode_linux(self, diagnosis: dict[str, Any]) -> dict[str, Any]:
        """Linux-specific VS Code diagnostics."""
        linux_info = {
            "installation_method": "unknown",
            "package_managers_checked": [],
        }

        # Check common installation paths
        possible_paths = [
            Path("/usr/bin/code"),
            Path("/usr/local/bin/code"),
            Path("/snap/bin/code"),
            Path("/opt/visual-studio-code"),
            Path("/usr/share/code"),
        ]

        for path in possible_paths:
            diagnosis["installation_paths_checked"].append(str(path))
            if path.exists():
                diagnosis["installed"] = True
                if "snap" in str(path):
                    linux_info["installation_method"] = "snap"
                elif "opt" in str(path):
                    linux_info["installation_method"] = "manual_extract"
                else:
                    linux_info["installation_method"] = "package_manager"
                break

        # Check for flatpak
        try:
            result = subprocess.run(
                ["flatpak", "list", "--app", "--columns=application"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if "com.visualstudio.code" in result.stdout:
                diagnosis["installed"] = True
                linux_info["installation_method"] = "flatpak"
                linux_info["package_managers_checked"].append("flatpak")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass

        if not diagnosis["installed"]:
            diagnosis["issues"].append("VS Code not found in standard Linux locations")
        elif not diagnosis["code_command_available"]:
            diagnosis["issues"].append(
                "VS Code installed but 'code' command not in PATH",
            )

        return linux_info

    def _check_vscode_extensions(self, diagnosis: dict[str, Any]) -> None:
        """Check VS Code extensions directory for Copilot extensions."""
        system = platform.system().lower()

        # Determine extensions directory based on platform
        if system == "darwin":
            extensions_dir = Path.home() / ".vscode" / "extensions"
        elif system == "windows":
            extensions_dir = (
                Path(os.environ.get("USERPROFILE", "")) / ".vscode" / "extensions"
            )
        else:  # Linux
            extensions_dir = Path.home() / ".vscode" / "extensions"

        diagnosis["extensions_directory"] = str(extensions_dir)

        if extensions_dir.exists():
            # Look for Copilot extensions
            copilot_patterns = [
                "github.copilot",
                "github.copilot-chat",
                "github.copilot-labs",
            ]
            for ext_dir in extensions_dir.iterdir():
                if ext_dir.is_dir():
                    for pattern in copilot_patterns:
                        if pattern in ext_dir.name.lower():
                            diagnosis["copilot_extensions"].append(ext_dir.name)

            if not diagnosis["copilot_extensions"]:
                diagnosis["issues"].append(
                    "No GitHub Copilot extensions found in VS Code",
                )
        else:
            diagnosis["issues"].append("VS Code extensions directory not found")

    def _diagnose_github_cli(self) -> dict[str, Any]:
        """Comprehensive GitHub CLI diagnostic information."""
        diagnosis = {
            "installed": False,
            "version": None,
            "authenticated": False,
            "copilot_extension_installed": False,
            "installation_paths_checked": [],
            "issues": [],
        }

        # Check if gh command is available
        gh_path = shutil.which("gh")
        if gh_path:
            diagnosis["installed"] = True
            diagnosis["installation_paths_checked"].append(gh_path)

            # Get version
            try:
                result = subprocess.run(
                    ["gh", "--version"],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    diagnosis["version"] = result.stdout.strip()
            except (subprocess.TimeoutExpired, OSError):
                pass

            # Check authentication
            diagnosis["authenticated"] = self._detect_github_cli_auth()
            if not diagnosis["authenticated"]:
                diagnosis["issues"].append("GitHub CLI not authenticated")

            # Check Copilot extension
            diagnosis["copilot_extension_installed"] = self._detect_gh_cli_copilot()
            if not diagnosis["copilot_extension_installed"]:
                diagnosis["issues"].append("GitHub Copilot CLI extension not installed")
        else:
            diagnosis["issues"].append("GitHub CLI not found in PATH")

        return diagnosis

    def _diagnose_copilot(self) -> dict[str, Any]:
        """Comprehensive Copilot availability diagnosis."""
        diagnosis = {
            "overall_available": False,
            "vscode_copilot_available": False,
            "cli_copilot_available": False,
            "issues": [],
        }

        # Check VS Code Copilot
        diagnosis["vscode_copilot_available"] = self._detect_vscode_copilot()
        if not diagnosis["vscode_copilot_available"]:
            diagnosis["issues"].append(
                "GitHub Copilot extension not detected in VS Code",
            )

        # Check CLI Copilot
        diagnosis["cli_copilot_available"] = self._detect_gh_cli_copilot()
        if not diagnosis["cli_copilot_available"]:
            diagnosis["issues"].append("GitHub Copilot CLI extension not available")

        diagnosis["overall_available"] = (
            diagnosis["vscode_copilot_available"] or diagnosis["cli_copilot_available"]
        )

        return diagnosis

    def _analyze_issues_and_solutions(self, diagnostics: dict[str, Any]) -> None:
        """Analyze diagnostic results and generate solutions."""
        vscode_result = diagnostics["detection_results"]["vscode"]
        gh_result = diagnostics["detection_results"]["github_cli"]
        copilot_result = diagnostics["detection_results"]["copilot"]

        # VS Code issues and solutions
        if not vscode_result["installed"]:
            diagnostics["issues_found"].append(
                {
                    "component": "VS Code",
                    "issue": "VS Code is not installed",
                    "severity": "high",
                },
            )
            self._add_vscode_installation_solution(diagnostics)
        elif vscode_result["issues"]:
            for issue in vscode_result["issues"]:
                diagnostics["issues_found"].append(
                    {
                        "component": "VS Code",
                        "issue": issue,
                        "severity": "medium",
                    },
                )

                if "PATH" in issue:
                    self._add_vscode_path_solution(diagnostics, vscode_result)
                elif "extensions" in issue:
                    self._add_copilot_extension_solution(diagnostics)

        # GitHub CLI issues and solutions
        if not gh_result["installed"]:
            diagnostics["issues_found"].append(
                {
                    "component": "GitHub CLI",
                    "issue": "GitHub CLI is not installed",
                    "severity": "high",
                },
            )
            self._add_github_cli_installation_solution(diagnostics)
        elif gh_result["issues"]:
            for issue in gh_result["issues"]:
                diagnostics["issues_found"].append(
                    {
                        "component": "GitHub CLI",
                        "issue": issue,
                        "severity": "medium",
                    },
                )

                if "authenticated" in issue:
                    self._add_github_auth_solution(diagnostics)
                elif "extension" in issue:
                    self._add_copilot_cli_solution(diagnostics)

    def _add_vscode_installation_solution(self, diagnostics: dict[str, Any]) -> None:
        """Add VS Code installation solution based on platform."""
        system = platform.system().lower()

        if system == "darwin":
            solution = {
                "title": "Install VS Code on macOS",
                "description": "Download and install Visual Studio Code for macOS",
                "steps": [
                    "1. Visit https://code.visualstudio.com/download",
                    "2. Download 'Mac' version (Universal or Intel based on your Mac)",
                    "3. Open the downloaded .zip file",
                    "4. Drag 'Visual Studio Code.app' to Applications folder",
                    "5. Open VS Code from Applications",
                    "6. Optional: Add 'code' command to PATH via Command Palette (Cmd+Shift+P) → 'Shell Command: Install code command in PATH'",
                ],
                "alternative": "Install via Homebrew: brew install --cask visual-studio-code",
            }
        elif system == "windows":
            solution = {
                "title": "Install VS Code on Windows",
                "description": "Download and install Visual Studio Code for Windows",
                "steps": [
                    "1. Visit https://code.visualstudio.com/download",
                    "2. Download 'Windows' version (User or System installer)",
                    "3. Run the downloaded installer",
                    "4. Follow installation wizard (recommended: check 'Add to PATH')",
                    "5. Launch VS Code from Start Menu or desktop shortcut",
                ],
                "alternative": "Install via winget: winget install Microsoft.VisualStudioCode",
            }
        else:  # Linux
            solution = {
                "title": "Install VS Code on Linux",
                "description": "Install Visual Studio Code using your distribution's package manager",
                "steps": [
                    "Ubuntu/Debian:",
                    "1. wget -qO- https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor > packages.microsoft.gpg",
                    "2. sudo install -o root -g root -m 644 packages.microsoft.gpg /etc/apt/trusted.gpg.d/",
                    "3. sudo sh -c 'echo \"deb [arch=amd64,arm64,armhf signed-by=/etc/apt/trusted.gpg.d/packages.microsoft.gpg] https://packages.microsoft.com/repos/code stable main\" > /etc/apt/sources.list.d/vscode.list'",
                    "4. sudo apt update && sudo apt install code",
                    "",
                    "Arch Linux: yay -S visual-studio-code-bin",
                    "Fedora: sudo dnf install code",
                ],
                "alternative": "Install via Snap: sudo snap install code --classic",
            }

        diagnostics["recommended_solutions"].append(solution)

    def _add_vscode_path_solution(
        self,
        diagnostics: dict[str, Any],
        vscode_result: dict[str, Any],
    ) -> None:
        """Add solution for VS Code PATH issues."""
        system = platform.system().lower()

        if system == "darwin":
            solution = {
                "title": "Add VS Code to PATH on macOS",
                "description": "Make the 'code' command available in terminal",
                "steps": [
                    "Method 1 (Recommended):",
                    "1. Open VS Code",
                    "2. Press Cmd+Shift+P to open Command Palette",
                    "3. Type 'Shell Command: Install code command in PATH'",
                    "4. Select the command and press Enter",
                    "",
                    "Method 2 (Manual):",
                    "1. Add this to your ~/.zshrc or ~/.bash_profile:",
                    '   export PATH="/Applications/Visual Studio Code.app/Contents/Resources/app/bin:$PATH"',
                    "2. Restart your terminal or run: source ~/.zshrc",
                ],
            }
        elif system == "windows":
            solution = {
                "title": "Add VS Code to PATH on Windows",
                "description": "Make the 'code' command available in Command Prompt/PowerShell",
                "steps": [
                    "1. Reinstall VS Code with 'Add to PATH' option checked, OR",
                    "2. Manually add to PATH:",
                    "   - Open System Properties → Advanced → Environment Variables",
                    "   - Edit 'Path' in System Variables",
                    f"   - Add: {vscode_result.get('platform_specific_info', {}).get('user_installation_path', 'C:\\\\Users\\\\[username]\\\\AppData\\\\Local\\\\Programs\\\\Microsoft VS Code')}\\\\bin",
                    "   - Restart Command Prompt/PowerShell",
                ],
            }
        else:  # Linux
            solution = {
                "title": "Add VS Code to PATH on Linux",
                "description": "Make the 'code' command available in terminal",
                "steps": [
                    "If installed via package manager, 'code' should be in PATH automatically.",
                    "If manually installed:",
                    "1. Create symlink: sudo ln -s /opt/visual-studio-code/bin/code /usr/local/bin/code",
                    '2. Or add to ~/.bashrc: export PATH="/opt/visual-studio-code/bin:$PATH"',
                    "3. Restart terminal or run: source ~/.bashrc",
                ],
            }

        diagnostics["recommended_solutions"].append(solution)

    def _add_copilot_extension_solution(self, diagnostics: dict[str, Any]) -> None:
        """Add GitHub Copilot extension installation solution."""
        solution = {
            "title": "Install GitHub Copilot Extension in VS Code",
            "description": "Add GitHub Copilot to VS Code for AI assistance",
            "steps": [
                "1. Open VS Code",
                "2. Click Extensions icon in sidebar (Ctrl+Shift+X / Cmd+Shift+X)",
                "3. Search for 'GitHub Copilot'",
                "4. Install 'GitHub Copilot' extension by GitHub",
                "5. Install 'GitHub Copilot Chat' extension by GitHub (recommended)",
                "6. Sign in to GitHub when prompted",
                "7. Ensure you have an active GitHub Copilot subscription",
            ],
            "note": "GitHub Copilot requires a paid subscription or qualifying free access",
        }
        diagnostics["recommended_solutions"].append(solution)

    def _add_github_cli_installation_solution(
        self,
        diagnostics: dict[str, Any],
    ) -> None:
        """Add GitHub CLI installation solution."""
        system = platform.system().lower()

        if system == "darwin":
            solution = {
                "title": "Install GitHub CLI on macOS",
                "description": "Install the GitHub command-line tool",
                "steps": [
                    "Method 1 (Homebrew - Recommended):",
                    "1. brew install gh",
                    "",
                    "Method 2 (Download):",
                    "1. Visit https://cli.github.com/",
                    "2. Download macOS installer",
                    "3. Run installer package",
                ],
            }
        elif system == "windows":
            solution = {
                "title": "Install GitHub CLI on Windows",
                "description": "Install the GitHub command-line tool",
                "steps": [
                    "Method 1 (winget):",
                    "1. winget install --id GitHub.cli",
                    "",
                    "Method 2 (Download):",
                    "1. Visit https://cli.github.com/",
                    "2. Download Windows installer",
                    "3. Run the .msi installer",
                ],
            }
        else:  # Linux
            solution = {
                "title": "Install GitHub CLI on Linux",
                "description": "Install the GitHub command-line tool",
                "steps": [
                    "Ubuntu/Debian:",
                    "1. type -p curl >/dev/null || sudo apt install curl -y",
                    "2. curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | sudo dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg",
                    "3. sudo chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg",
                    '4. echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null',
                    "5. sudo apt update && sudo apt install gh",
                    "",
                    "Arch Linux: pacman -S github-cli",
                    "Fedora: dnf install gh",
                ],
            }

        diagnostics["recommended_solutions"].append(solution)

    def _add_github_auth_solution(self, diagnostics: dict[str, Any]) -> None:
        """Add GitHub authentication solution."""
        solution = {
            "title": "Authenticate GitHub CLI",
            "description": "Sign in to GitHub using the command line",
            "steps": [
                "1. Run: gh auth login",
                "2. Choose 'GitHub.com'",
                "3. Choose 'HTTPS' (recommended) or 'SSH'",
                "4. Choose 'Y' to authenticate Git with GitHub credentials",
                "5. Choose 'Login with a web browser' (recommended)",
                "6. Copy the one-time code shown",
                "7. Press Enter to open browser",
                "8. Paste the code and sign in to GitHub",
                "9. Authorize GitHub CLI access",
            ],
        }
        diagnostics["recommended_solutions"].append(solution)

    def _add_copilot_cli_solution(self, diagnostics: dict[str, Any]) -> None:
        """Add GitHub Copilot CLI extension solution."""
        solution = {
            "title": "Install GitHub Copilot CLI Extension",
            "description": "Add Copilot support to GitHub CLI",
            "steps": [
                "1. Ensure GitHub CLI is installed and authenticated",
                "2. Run: gh extension install github/gh-copilot",
                "3. Test with: gh copilot --version",
                "4. Use 'gh copilot suggest' for command suggestions",
                "5. Use 'gh copilot explain' for command explanations",
            ],
            "note": "Requires active GitHub Copilot subscription",
        }
        diagnostics["recommended_solutions"].append(solution)

    def format_user_friendly_error(
        self,
        validation_result: dict[str, Any] | None = None,
    ) -> str:
        """Format a user-friendly error message with troubleshooting steps.

        Args:
            validation_result: Optional validation result from validate_copilot_environment()

        Returns:
            Formatted error message with clear instructions

        """
        if validation_result is None:
            validation_result = self.validate_copilot_environment()

        # If everything is working, return success message
        if validation_result["overall_status"] == "ready":
            return """
✅ GitHub Copilot Environment: READY!

All components are properly installed and configured:
• GitHub CLI: ✅ Installed and authenticated  
• GitHub Copilot CLI: ✅ Available
• VS Code: ✅ Installed
• Copilot Extensions: ✅ Available

You're all set to use GitHub Copilot with metacontext!
"""

        # Get detailed diagnostics for comprehensive error reporting
        diagnostics = self.get_detailed_diagnostics()

        error_msg = [""]
        error_msg.append("❌ GitHub Copilot Environment Setup Issues Detected")
        error_msg.append("=" * 60)
        error_msg.append("")

        # Quick Fix Summary for immediate action
        error_msg.append("🚀 QUICK FIX SUMMARY:")
        missing_components = []
        if not diagnostics["detection_results"]["vscode"]["installed"]:
            missing_components.append("VS Code")
        if not diagnostics["detection_results"]["github_cli"]["installed"]:
            missing_components.append("GitHub CLI")
        if not diagnostics["detection_results"]["copilot"]["overall_available"]:
            missing_components.append("GitHub Copilot")

        if missing_components:
            error_msg.append(f"   You need to install: {', '.join(missing_components)}")
        else:
            error_msg.append(
                "   Configuration issues detected - quick fixes available below",
            )
        error_msg.append("")

        # Summary of issues
        if diagnostics["issues_found"]:
            error_msg.append("🔍 Issues Found:")
            for issue in diagnostics["issues_found"]:
                severity_icon = "🚨" if issue["severity"] == "high" else "⚠️"
                error_msg.append(
                    f"  {severity_icon} {issue['component']}: {issue['issue']}",
                )
            error_msg.append("")

        # Platform information
        error_msg.append(
            f"🖥️  Platform: {diagnostics['platform']} {diagnostics['platform_version']}",
        )
        error_msg.append("")

        # Quick status overview
        vscode_status = (
            "✅" if diagnostics["detection_results"]["vscode"]["installed"] else "❌"
        )
        gh_status = (
            "✅"
            if diagnostics["detection_results"]["github_cli"]["installed"]
            else "❌"
        )
        copilot_status = (
            "✅"
            if diagnostics["detection_results"]["copilot"]["overall_available"]
            else "❌"
        )

        error_msg.append("📊 Component Status:")
        error_msg.append(f"  {vscode_status} VS Code")
        error_msg.append(f"  {gh_status} GitHub CLI")
        error_msg.append(f"  {copilot_status} GitHub Copilot")
        error_msg.append("")

        # Solutions with enhanced actionable instructions
        if diagnostics["recommended_solutions"]:
            error_msg.append("🛠️  Step-by-Step Solutions (in priority order):")
            error_msg.append("")

            for i, solution in enumerate(diagnostics["recommended_solutions"], 1):
                error_msg.append(f"═══ SOLUTION {i}: {solution['title']} ═══")
                error_msg.append(f"📝 {solution['description']}")
                error_msg.append("")

                # Add time estimate and priority
                if "VS Code" in solution["title"]:
                    error_msg.append("⏱️  Time: 5-10 minutes | 🚨 HIGH PRIORITY")
                elif "GitHub CLI" in solution["title"]:
                    error_msg.append("⏱️  Time: 3-5 minutes | 🚨 HIGH PRIORITY")
                else:
                    error_msg.append("⏱️  Time: 2-3 minutes | ⚠️  MEDIUM PRIORITY")
                error_msg.append("")

                # Enhanced step formatting with copy-paste commands
                for step in solution["steps"]:
                    if step.strip():
                        # Highlight commands that can be copy-pasted
                        if any(
                            cmd in step
                            for cmd in [
                                "brew install",
                                "gh auth",
                                "winget install",
                                "sudo apt",
                                "gh extension",
                            ]
                        ):
                            error_msg.append(f"   💻 COPY & RUN: {step}")
                        elif step.startswith(
                            ("1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.", "9."),
                        ):
                            error_msg.append(f"   📋 {step}")
                        else:
                            error_msg.append(f"      {step}")
                error_msg.append("")

                # Add verification step for each solution
                if "VS Code" in solution["title"]:
                    error_msg.append(
                        "   ✅ VERIFY: Run 'code --version' in terminal to confirm installation",
                    )
                elif (
                    "GitHub CLI" in solution["title"] and "Install" in solution["title"]
                ):
                    error_msg.append(
                        "   ✅ VERIFY: Run 'gh --version' to confirm installation",
                    )
                elif "auth" in solution["title"].lower():
                    error_msg.append(
                        "   ✅ VERIFY: Run 'gh auth status' to confirm authentication",
                    )
                elif "extension" in solution["title"].lower():
                    error_msg.append(
                        "   ✅ VERIFY: Run 'gh copilot --version' to confirm extension",
                    )
                error_msg.append("")

                if "alternative" in solution:
                    error_msg.append(f"   🔄 ALTERNATIVE: {solution['alternative']}")
                    error_msg.append("")

                if "note" in solution:
                    error_msg.append(f"   📝 IMPORTANT: {solution['note']}")
                    error_msg.append("")

                error_msg.append("─" * 50)
                error_msg.append("")

        # Enhanced additional help and next steps
        error_msg.append("📚 Additional Resources:")
        error_msg.append(
            "• VS Code Setup Guide: https://code.visualstudio.com/docs/setup/setup-overview",
        )
        error_msg.append("• GitHub CLI Manual: https://cli.github.com/manual/")
        error_msg.append(
            "• GitHub Copilot Docs: https://docs.github.com/en/copilot/getting-started-with-github-copilot",
        )
        error_msg.append("")

        error_msg.append("🎯 NEXT STEPS:")
        error_msg.append("1. ✅ Complete the solutions above in the order shown")
        error_msg.append("2. 🔄 Run this command again to verify your setup:")
        error_msg.append(
            "   💻 python -m metacontext.ai.handlers.companions.copilot_provider",
        )
        error_msg.append(
            "3. 🚀 Once all components show ✅, you're ready to use GitHub Copilot!",
        )
        error_msg.append("")

        error_msg.append("💡 TROUBLESHOOTING TIPS:")
        error_msg.append(
            "• If commands don't work, restart your terminal after installation",
        )
        error_msg.append(
            "• On macOS, you may need to run VS Code once to install command line tools",
        )
        error_msg.append(
            "• GitHub Copilot requires an active subscription or free access",
        )
        error_msg.append("")

        return "\n".join(error_msg)
