"""Context preprocessing utilities for reducing token usage in prompts.

This module provides utilities to intelligently truncate and prioritize file content
to reduce token usage while preserving semantic information.
"""

import re
from pathlib import Path
from typing import Any

# Constants for content limits
MAX_FILE_CONTENT_LENGTH = 800  # Maximum characters per file
MAX_CONFIG_LINE_LENGTH = 100  # Maximum length for configuration lines
PRIORITY_FILE_PATTERNS = [
    r"readme\.md",
    r"readme\.txt",
    r"pyproject\.toml",
    r"package\.json",
    r"setup\.py",
    r"requirements\.txt",
    r"dockerfile",
    r"makefile",
    r"\.gitignore",
    r"config\.(yaml|yml|json|toml)",
]

# Patterns for preserving important content
IMPORTANT_CONTENT_PATTERNS = [
    r"purpose",
    r"description",
    r"usage",
    r"installation",
    r"features",
    r"requirements",
    r"dependencies",
    r"configuration",
    r"model",
    r"dataset",
    r"training",
    r"evaluation",
]

# Domain-specific keyword sets for semantic filtering
DOMAIN_KEYWORDS = {
    "machine_learning": [
        "model",
        "training",
        "dataset",
        "features",
        "algorithm",
        "neural",
        "deep learning",
        "classification",
        "regression",
        "clustering",
        "validation",
        "accuracy",
        "loss",
        "hyperparameter",
        "epoch",
        "batch",
        "gradient",
        "optimizer",
        "tensorflow",
        "pytorch",
    ],
    "data_science": [
        "analysis",
        "visualization",
        "pandas",
        "numpy",
        "statistics",
        "correlation",
        "distribution",
        "outlier",
        "preprocessing",
        "cleaning",
        "eda",
        "exploration",
    ],
    "geospatial": [
        "gis",
        "spatial",
        "coordinate",
        "projection",
        "raster",
        "vector",
        "geometry",
        "map",
        "location",
        "latitude",
        "longitude",
        "crs",
        "geojson",
        "shapefile",
    ],
    "web_development": [
        "api",
        "endpoint",
        "route",
        "middleware",
        "database",
        "authentication",
        "session",
        "request",
        "response",
        "http",
        "rest",
        "graphql",
        "frontend",
        "backend",
    ],
    "configuration": [
        "config",
        "settings",
        "environment",
        "variable",
        "parameter",
        "option",
        "default",
        "override",
        "production",
        "development",
        "staging",
    ],
}


def calculate_semantic_relevance(content: str, keywords: list[str]) -> float:
    """Calculate semantic relevance score based on keyword presence.

    Args:
        content: Text content to analyze
        keywords: List of relevant keywords

    Returns:
        Relevance score between 0.0 and 1.0

    """
    if not content or not keywords:
        return 0.0

    content_lower = content.lower()
    keyword_matches = 0
    total_keywords = len(keywords)

    for keyword in keywords:
        if keyword.lower() in content_lower:
            keyword_matches += 1

    return keyword_matches / total_keywords


def filter_content_by_keywords(
    content: str, keywords: list[str], min_relevance: float = 0.1
) -> str:
    """Filter content to include only sections with sufficient keyword relevance.

    Args:
        content: Content to filter
        keywords: Relevant keywords for filtering
        min_relevance: Minimum relevance score to include content

    Returns:
        Filtered content

    """
    if not keywords:
        return content

    # Split content into meaningful chunks (paragraphs or sections)
    chunks = []
    current_chunk = []

    for line in content.split("\n"):
        if line.strip() == "" and current_chunk:
            # End of chunk
            chunks.append("\n".join(current_chunk))
            current_chunk = []
        else:
            current_chunk.append(line)

    # Add final chunk
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    # Filter chunks by relevance
    relevant_chunks = []
    for chunk in chunks:
        relevance = calculate_semantic_relevance(chunk, keywords)
        if (
            relevance >= min_relevance or len(chunk) < 100
        ):  # Always include short chunks
            relevant_chunks.append(chunk)

    return "\n\n".join(relevant_chunks)


def extract_file_type_specific_content(
    content: str, file_path: str, max_length: int
) -> str:
    """Extract content using file type-specific rules.

    Args:
        content: File content
        file_path: Path to the file
        max_length: Maximum length to extract

    Returns:
        Extracted content optimized for file type

    """
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name.lower()

    # README files - focus on description, usage, installation
    if "readme" in file_name:
        readme_keywords = [
            "description",
            "usage",
            "installation",
            "getting started",
            "overview",
            "purpose",
        ]
        filtered_content = filter_content_by_keywords(
            content, readme_keywords, min_relevance=0.05
        )
        return extract_important_sections(filtered_content, max_length)

    # Configuration files - extract business logic indicators
    if file_ext in {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}:
        config_keywords = DOMAIN_KEYWORDS["configuration"]
        return filter_content_by_keywords(content, config_keywords, min_relevance=0.1)[
            :max_length
        ]

    # Code files - extract signatures and docstrings
    if file_ext in {".py", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".rs", ".go"}:
        return extract_code_signatures(content, max_length)

    # Documentation files
    if file_ext in {".md", ".rst", ".txt"}:
        doc_keywords = [
            "overview",
            "architecture",
            "design",
            "implementation",
            "usage",
            "api",
        ]
        filtered_content = filter_content_by_keywords(
            content, doc_keywords, min_relevance=0.05
        )
        return extract_important_sections(filtered_content, max_length)

    # Default: extract important sections
    return extract_important_sections(content, max_length)


def prioritize_files(file_contents: dict[str, str]) -> dict[str, str]:
    """Prioritize files based on importance for analysis.

    Args:
        file_contents: Dictionary mapping file paths to content

    Returns:
        Reordered dictionary with important files first

    """
    priority_files = {}
    regular_files = {}

    for file_path, content in file_contents.items():
        file_name = Path(file_path).name.lower()

        # Check if this is a priority file
        is_priority = any(
            re.search(pattern, file_name, re.IGNORECASE)
            for pattern in PRIORITY_FILE_PATTERNS
        )

        if is_priority:
            priority_files[file_path] = content
        else:
            regular_files[file_path] = content

    # Return priority files first, then regular files
    return {**priority_files, **regular_files}


def extract_important_sections(
    content: str, max_length: int = MAX_FILE_CONTENT_LENGTH
) -> str:
    """Extract important sections from file content while respecting length limits.

    Args:
        content: Full file content
        max_length: Maximum characters to return

    Returns:
        Truncated content focusing on important sections

    """
    if len(content) <= max_length:
        return content

    lines = content.split("\n")
    important_lines = []
    current_length = 0

    # First pass: collect lines with important keywords
    for line in lines:
        line_lower = line.lower()

        # Check if line contains important patterns
        is_important = any(
            pattern in line_lower for pattern in IMPORTANT_CONTENT_PATTERNS
        )

        # Also include headers, function definitions, and class definitions
        is_structural = (
            line.strip().startswith("#")  # Markdown headers
            or line.strip().startswith("def ")  # Python functions
            or line.strip().startswith("class ")  # Python classes
            or line.strip().startswith("function ")  # JavaScript functions
            or ("=" in line and len(line.strip()) < 100)  # Configuration entries
        )

        if is_important or is_structural:
            if current_length + len(line) + 1 <= max_length:
                important_lines.append(line)
                current_length += len(line) + 1
            else:
                break

    # If we haven't filled our quota, add more lines from the beginning
    if current_length < max_length * 0.8:  # Use 80% threshold
        remaining_space = max_length - current_length

        for line in lines:
            if line not in important_lines:
                if len(line) + 1 <= remaining_space:
                    important_lines.append(line)
                    remaining_space -= len(line) + 1
                else:
                    break

    result = "\n".join(important_lines)

    # If still too long, truncate and add ellipsis
    if len(result) > max_length:
        result = result[: max_length - 3] + "..."

    return result


def extract_code_signatures(
    content: str, max_length: int = MAX_FILE_CONTENT_LENGTH
) -> str:
    """Extract function signatures and class definitions from code files.

    Args:
        content: Full code content
        max_length: Maximum characters to return

    Returns:
        Content with only signatures and docstrings

    """
    if len(content) <= max_length:
        return content

    lines = content.split("\n")
    signature_lines = []
    current_length = 0
    in_docstring = False
    docstring_char = None

    for line in lines:
        stripped = line.strip()

        # Track docstrings
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if '"""' in line else "'''"
            elif docstring_char in line:
                in_docstring = False
                docstring_char = None

        # Include function/class definitions and docstrings
        should_include = (
            stripped.startswith(
                ("def ", "class ", "function ", "async def ", "export ")
            )
            or in_docstring
        )

        if should_include:
            if current_length + len(line) + 1 <= max_length:
                signature_lines.append(line)
                current_length += len(line) + 1
            else:
                break

    result = "\n".join(signature_lines)

    if len(result) > max_length:
        result = result[: max_length - 3] + "..."

    return result


def preprocess_file_contents(
    file_contents: dict[str, str],
    total_limit: int = 3000,
) -> dict[str, str]:
    """Preprocess file contents to reduce token usage while preserving important information.

    Args:
        file_contents: Dictionary mapping file paths to content
        total_limit: Total character limit across all files

    Returns:
        Preprocessed file contents dictionary

    """
    if not file_contents:
        return {}

    # Prioritize files by importance
    prioritized_files = prioritize_files(file_contents)

    # Calculate per-file limit based on number of files
    num_files = len(prioritized_files)
    per_file_limit = min(MAX_FILE_CONTENT_LENGTH, total_limit // num_files)

    processed_contents = {}
    total_used = 0

    for file_path, content in prioritized_files.items():
        if total_used >= total_limit:
            break

        # Use enhanced file type-specific extraction
        processed_content = extract_file_type_specific_content(
            content, file_path, per_file_limit
        )

        # Ensure we don't exceed total limit
        remaining_space = total_limit - total_used
        if len(processed_content) > remaining_space:
            processed_content = processed_content[: remaining_space - 3] + "..."

        processed_contents[file_path] = processed_content
        total_used += len(processed_content)

    return processed_contents


def format_preprocessed_context(
    preprocessed_contents: dict[str, str],
    *,
    include_file_info: bool = True,
) -> str:
    """Format preprocessed file contents for inclusion in prompts.

    Args:
        preprocessed_contents: Preprocessed file contents
        include_file_info: Whether to include file path information

    Returns:
        Formatted string for prompt inclusion

    """
    if not preprocessed_contents:
        return "No file contents available."

    sections = []

    for file_path, content in preprocessed_contents.items():
        if include_file_info:
            sections.append(f"--- {file_path} ---")
        sections.append(content.strip())
        if include_file_info:
            sections.append("")  # Add blank line between files

    return "\n".join(sections)


def smart_context_preprocessing(context_data: dict[str, Any]) -> dict[str, Any]:
    """Smart preprocessing of context data to reduce token usage.

    This function identifies and preprocesses large text content in context data
    while preserving other data types and small content unchanged.

    Args:
        context_data: Raw context data dictionary

    Returns:
        Preprocessed context data with reduced token usage

    """
    processed_context = context_data.copy()

    # Look for file content keys that might need preprocessing
    file_content_keys = [
        "key_files_content",
        "file_contents",
        "source_code",
        "training_scripts",
        "code_content",
    ]

    for key in file_content_keys:
        if key in processed_context:
            value = processed_context[key]

            # If it's a dictionary (file_path -> content mapping)
            if isinstance(value, dict):
                processed_context[key] = format_preprocessed_context(
                    preprocess_file_contents(value),
                )
            # If it's a large string
            elif isinstance(value, str) and len(value) > MAX_FILE_CONTENT_LENGTH:
                processed_context[key] = extract_important_sections(value)

    return processed_context
