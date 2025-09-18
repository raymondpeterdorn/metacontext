"""Output utilities for writing metacontext in different formats.

This module provides writers for various output formats including YAML, JSON,
and custom metacontext formats.
"""

import json
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


def write_yaml(data: dict[str, Any], output_path: Path) -> None:
    """Write data to YAML format.

    Args:
        data: Dictionary data to write
        output_path: Path to write the YAML file

    """
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            default_flow_style=False,
            sort_keys=False,
            indent=2,
            allow_unicode=True,
        )
    logger.debug("Written YAML format to %s", output_path)


def write_json(data: dict[str, Any], output_path: Path) -> None:
    """Write data to JSON format.

    Args:
        data: Dictionary data to write
        output_path: Path to write the JSON file

    """
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
            ensure_ascii=False,
            default=str,  # Handle non-serializable objects
        )
    logger.debug("Written JSON format to %s", output_path)


def write_mcntxt(data: dict[str, Any], output_path: Path) -> None:
    """Write data to .mcntxt (metacontext) format.

    This is a custom text format that's more readable than JSON/YAML.

    Args:
        data: Dictionary data to write
        output_path: Path to write the .mcntxt file

    """
    with output_path.open("w", encoding="utf-8") as f:
        f.write("# METACONTEXT FILE\n")
        f.write(f"# Generated for: {data.get('filename', 'unknown')}\n")
        f.write(f"# Architecture: {data.get('architecture_version', 'unknown')}\n")
        f.write("\n")

        # Write key sections in a readable format
        _write_section(f, "FILE_INFO", data.get("file_info", {}))
        _write_section(f, "DATA_STRUCTURE", data.get("data_structure", {}))
        _write_section(f, "ANALYSIS_METADATA", data.get("analysis_metadata", {}))

        if "codebase_context" in data:
            _write_section(f, "CODEBASE_CONTEXT", data["codebase_context"])

        if "generation_info" in data:
            _write_section(f, "GENERATION_INFO", data["generation_info"])

    logger.debug("Written .mcntxt format to %s", output_path)


def _write_section(f, section_name: str, section_data: Any) -> None:
    """Write a section to the .mcntxt file.

    Args:
        f: File handle
        section_name: Name of the section
        section_data: Data for the section

    """
    f.write(f"[{section_name}]\n")

    if isinstance(section_data, dict):
        for key, value in section_data.items():
            if isinstance(value, (dict, list)):
                f.write(f"{key}: {json.dumps(value, default=str)}\n")
            else:
                f.write(f"{key}: {value}\n")
    else:
        f.write(f"data: {json.dumps(section_data, default=str)}\n")

    f.write("\n")


def write_output(data: dict[str, Any], output_path: Path, output_format: str) -> None:
    """Write data in the specified format.

    Args:
        data: Dictionary data to write
        output_path: Path to write the file
        output_format: Format to write ('yaml', 'json', 'mcntxt', 'metacontext')

    Raises:
        ValueError: If output_format is not supported

    """
    # Normalize format names
    format_map = {
        "yaml": write_yaml,
        "yaml_clean": write_yaml,
        "json": write_json,
        "mcntxt": write_mcntxt,
        "metacontext": write_mcntxt,  # Alias for mcntxt
    }

    writer_func = format_map.get(output_format)
    if not writer_func:
        supported_formats = list(format_map.keys())
        msg = f"Unsupported output format: {output_format}. Supported: {supported_formats}"
        raise ValueError(msg)

    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write using the appropriate writer
    writer_func(data, output_path)

    logger.info("✓ Output written in %s format to %s", output_format, output_path.name)
