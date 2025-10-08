#!/usr/bin/env python3
"""Command Line Interface for Metacontext.

This module provides a lightweight CLI for generating metacontext from files.
Users can run metacontext directly from the terminal without writing Python code.

Example usage:
    metacontext data.csv
    metacontext data.csv --output json
    metacontext data.csv --output yaml --deep
"""

import argparse
import sys
import traceback
from pathlib import Path

from metacontext.ai.handlers.companions.companion_factory import (
    CompanionProviderFactory,
)
from metacontext.metacontextualize import metacontextualize

# Constants for file size calculations
BYTES_PER_KB = 1024
BYTES_PER_MB = 1024 * 1024


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the argument parser."""
    parser = argparse.ArgumentParser(
        prog="metacontext",
        description="Generate intelligent metacontext for any file",
        epilog="Example: metacontext data.csv --output yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional argument for the file
    parser.add_argument(
        "file",
        type=str,
        help="Path to the file to analyze",
    )

    # Output format option
    parser.add_argument(
        "--output",
        "-o",
        choices=["yaml", "json"],
        default="yaml",
        help="Output format (default: yaml)",
    )

    # Deep analysis option
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Enable deep analysis with AI enrichment (requires API key)",
    )

    # Companion mode option
    parser.add_argument(
        "--companion",
        action="store_true",
        help="Use IDE-integrated companions like GitHub Copilot instead of API",
    )

    # Force API mode option
    parser.add_argument(
        "--force-api",
        action="store_true",
        help="Force API mode even if companions are available",
    )

    # Output file option
    parser.add_argument(
        "--output-file",
        "-f",
        type=str,
        help="Output file path (default: auto-generated based on input file)",
    )

    # Verbose option
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )

    # Version option
    parser.add_argument(
        "--version",
        action="version",
        version="metacontext 0.3.0",
    )

    return parser


def validate_file_path(file_path: str) -> Path:
    """Validate and convert file path to Path object."""
    path = Path(file_path)

    if not path.exists():
        sys.exit(1)

    if not path.is_file():
        sys.exit(1)

    return path


def main() -> None:
    """Provide main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Validate input file
    try:
        file_path = validate_file_path(args.file)
    except KeyboardInterrupt:
        sys.exit(1)

    try:
        # Determine AI companion/provider strategy
        ai_companion = None
        use_deep_analysis = args.deep

        if args.companion or (not args.force_api and not args.deep):
            # Try to detect and use companion providers
            try:
                factory = CompanionProviderFactory()
                companion_provider = factory.detect_available_companion()
                if companion_provider:
                    print(f"🤖 Using {companion_provider.companion_type} for analysis")
                    ai_companion = companion_provider
                    use_deep_analysis = True  # Enable deep analysis for companions
                elif args.companion:
                    # User explicitly requested companion mode but none available
                    print(
                        "❌ No companion providers available. Install GitHub Copilot or use --force-api",
                    )
                    sys.exit(1)
                else:
                    print("🔗 No companions detected, using API mode")
            except Exception as e:
                if args.companion:
                    print(f"❌ Companion detection failed: {e}")
                    sys.exit(1)
                print(f"⚠️ Companion detection failed, falling back to API mode: {e}")

        # Generate metacontext using simplified interface
        output_path = metacontextualize(
            data_object=None,  # We're analyzing a file directly
            file_path=file_path,
            output_format=args.output,
            include_llm_analysis=use_deep_analysis,
            ai_companion=ai_companion,  # Pass companion provider
            output_path=args.output_file,
            verbose=args.verbose,
        )

        # Success message

        if args.verbose:
            # Show file size
            size_bytes = output_path.stat().st_size
            if size_bytes < BYTES_PER_KB:
                pass
            elif size_bytes < BYTES_PER_MB:
                f"{size_bytes / BYTES_PER_KB:.1f} KB"
            else:
                f"{size_bytes / BYTES_PER_MB:.1f} MB"

    except KeyboardInterrupt:
        sys.exit(1)
    except (FileNotFoundError, PermissionError, OSError):
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
