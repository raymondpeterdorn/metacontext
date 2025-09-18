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

from src.metacontextualize import MetacontextualizeArgs, metacontextualize

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

    # Create metacontextualize arguments
    metacontext_args = MetacontextualizeArgs(
        output_format=args.output,
        include_llm_analysis=args.deep,
        output_path=args.output_file,
        verbose=args.verbose,
    )

    # Show what we're doing
    if args.verbose and args.output_file:
        pass

    try:
        # Generate metacontext
        output_path = metacontextualize(
            data_object=None,  # We're analyzing a file directly
            file_path=file_path,
            args=metacontext_args,
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
