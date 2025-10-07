"""Context preprocessing utilities for reducing token usage in prompts.

This module provides utilities to intelligently truncate and prioritize file content
to reduce token usage while preserving semantic information.
"""

import ast
import hashlib
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any

from .knowledge_graph import SemanticKnowledgeGraph

# Performance optimization constants
ENABLE_CACHING = True
ENABLE_PARALLEL_PROCESSING = True
MAX_WORKERS = 4  # Number of parallel workers
CACHE_SIZE = 256  # LRU cache size for frequently accessed functions

# Constants for content limits
MAX_FILE_CONTENT_LENGTH = 800  # Maximum characters per file
MAX_CONFIG_LINE_LENGTH = 100  # Maximum length for configuration lines
SHORT_CHUNK_THRESHOLD = 100

# File filtering constants
MAX_FILE_SIZE = 1024 * 1024  # 1MB limit for individual files
MAX_BINARY_FILE_SIZE = 10 * 1024  # 10KB limit for binary files

# File type categories
RELEVANT_EXTENSIONS = {
    "code": [
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cpp",
        ".c",
        ".h",
        ".rs",
        ".go",
        ".rb",
        ".php",
        ".scala",
        ".kt",
    ],
    "config": [".yaml", ".yml", ".json", ".toml", ".ini", ".cfg", ".conf", ".env"],
    "documentation": [".md", ".rst", ".txt", ".adoc", ".org"],
    "data": [".csv", ".xlsx", ".parquet", ".feather", ".jsonl", ".tsv"],
    "geospatial": [".geojson", ".gpkg", ".shp", ".kml", ".gml"],
    "raster": [".tiff", ".tif", ".nc", ".hdf", ".grib"],
    "models": [".pkl", ".joblib", ".h5", ".keras", ".onnx", ".pt", ".pth"],
    "media": [".jpg", ".jpeg", ".png", ".gif", ".svg", ".mp4", ".avi", ".wav", ".mp3"],
    "notebooks": [".ipynb"],
    "sql": [".sql", ".ddl", ".dml"],
}

# File patterns to ignore (infrastructure, build artifacts, etc.)
IGNORE_PATTERNS = [
    r"\.git/",
    r"\.venv/",
    r"venv/",
    r"__pycache__/",
    r"\.pytest_cache/",
    r"\.mypy_cache/",
    r"node_modules/",
    r"\.vscode/",
    r"\.idea/",
    r"build/",
    r"dist/",
    r"target/",
    r"\.DS_Store",
    r"Thumbs\.db",
    r"\.coverage",
    r"coverage\.xml",
    r"\.tox/",
    r"\.nox/",
    r"Dockerfile.*",
    r"docker-compose.*",
    r"Makefile",
    r"\.lock$",
    r"\.log$",
    r"\.tmp$",
    r"\.temp$",
    r"\.bak$",
    r"\.swp$",
    r"~$",
]

# Priority patterns for important files
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

# Patterns for detecting column derivations and definitions
COLUMN_DERIVATION_PATTERNS = [
    r"df\[[\"\'](\w+)[\"\']\]\s*=\s*(.+)",  # df["col"] = expression
    r"(\w+_df)\[[\"\'](\w+)[\"\']\]\s*=\s*(.+)",  # named_df["col"] = expression
    r"(\w+)\[[\"\'](\w+)[\"\']\]\s*=\s*(.+)",  # any_var["col"] = expression
    r"#\s*(?:derived|feature|column):\s*(\w+)\s*=\s*(.+)",  # explicit comments
    r"(\w+)\s*=\s*\(.+\)\.astype\(",  # var = (condition).astype(int)
]

# Enhanced comment pattern recognition for semantic extraction
MULTILINE_COMMENT_PATTERNS = [
    r"\"\"\"(.*?[Dd]ata [Dd]ictionary.*?)\"\"\"",  # Data Dictionary in docstrings
    r"\"\"\"(.*?[Cc]olumn [Dd]efinitions.*?)\"\"\"",  # Column Definitions in docstrings
    r"#\s*=+\s*\n((?:#.*\n)*)",  # Comment blocks with separators
    r"#\s*-+\s*\n((?:#.*\n)*)",  # Comment blocks with dashes
]

BUSINESS_LOGIC_PATTERNS = [
    r"#\s*[Bb]usiness [Rr]ule:\s*(.+)",  # Business rule comments
    r"#\s*[Aa]lgorithm:\s*(.+)",  # Algorithm descriptions
    r"#\s*[Ll]ogic:\s*(.+)",  # Logic explanations
    r"#\s*[Nn]ote:\s*(.+)",  # Important notes
    r"#\s*[Ii]mportant:\s*(.+)",  # Important information
    r"#\s*[Ww]arning:\s*(.+)",  # Warnings
]

CONSTANT_DEFINITION_PATTERNS = [
    r"([A-Z_][A-Z0-9_]*)\s*=\s*([^#\n]+)\s*#\s*(.+)",  # CONSTANT = value  # explanation
    r"(\w+_(?:THRESHOLD|LIMIT|MAX|MIN|SIZE))\s*=\s*([^#\n]+)\s*#\s*(.+)",  # Threshold constants
    r"(\w+_(?:CONFIG|SETTING|PARAM))\s*=\s*([^#\n]+)\s*#\s*(.+)",  # Configuration constants
]

DATA_DICTIONARY_PATTERNS = [
    r"#\s*[Cc]olumn [Dd]efinitions?\s*:?\s*\n((?:#.*\n)*)",  # Column definition blocks
    r"#\s*[Dd]ata [Dd]ictionary\s*:?\s*\n((?:#.*\n)*)",  # Data dictionary blocks
    r"#\s*[Ff]ield [Dd]escriptions?\s*:?\s*\n((?:#.*\n)*)",  # Field description blocks
    r"#\s*[Vv]ariable [Dd]efinitions?\s*:?\s*\n((?:#.*\n)*)",  # Variable definition blocks
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
    content: str,
    keywords: list[str],
    min_relevance: float = 0.1,
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
    current_chunk: list[str] = []

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
            relevance >= min_relevance or len(chunk) < SHORT_CHUNK_THRESHOLD
        ):  # Always include short chunks
            relevant_chunks.append(chunk)

    return "\n\n".join(relevant_chunks)


def extract_file_type_specific_content(
    content: str,
    file_path: str,
    max_length: int,
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
            content,
            readme_keywords,
            min_relevance=0.05,
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
            content,
            doc_keywords,
            min_relevance=0.05,
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


def is_file_relevant(file_path: str) -> bool:
    """Check if a file is relevant for metacontext analysis.

    Args:
        file_path: Path to the file to check

    Returns:
        True if file should be processed, False if it should be ignored

    """
    file_path_str = str(file_path)
    file_name = Path(file_path).name.lower()
    file_ext = Path(file_path).suffix.lower()

    # Check ignore patterns first
    for pattern in IGNORE_PATTERNS:
        if re.search(pattern, file_path_str, re.IGNORECASE):
            return False

    # Check if extension is in our relevant categories
    all_relevant_extensions = set()
    for category_exts in RELEVANT_EXTENSIONS.values():
        all_relevant_extensions.update(category_exts)

    if file_ext in all_relevant_extensions:
        return True

    # Special cases for files without extensions but important names
    important_names = [
        "readme",
        "license",
        "changelog",
        "authors",
        "contributors",
        "makefile",
        "dockerfile",
        "requirements",
        "setup",
    ]

    for name in important_names:
        if name in file_name:
            return True

    return False


def check_file_size_limits(file_path: str, content: str) -> tuple[bool, str]:
    """Check if file meets size requirements for processing.

    Args:
        file_path: Path to the file
        content: File content

    Returns:
        Tuple of (should_process, reason_if_skipped)

    """
    file_size = len(content.encode("utf-8"))
    file_ext = Path(file_path).suffix.lower()

    # Check if it's a binary file type that should have strict limits
    binary_extensions = RELEVANT_EXTENSIONS["media"] + RELEVANT_EXTENSIONS["models"]

    if file_ext in binary_extensions:
        if file_size > MAX_BINARY_FILE_SIZE:
            return (
                False,
                f"Binary file too large: {file_size} bytes > {MAX_BINARY_FILE_SIZE} limit",
            )
    elif file_size > MAX_FILE_SIZE:
        return False, f"File too large: {file_size} bytes > {MAX_FILE_SIZE} limit"

    return True, ""


def detect_file_type(file_path: str, content: str) -> str:
    """Detect the type/category of a file for specialized processing.

    Args:
        file_path: Path to the file
        content: File content (first few lines for detection)

    Returns:
        File category string

    """
    file_ext = Path(file_path).suffix.lower()
    file_name = Path(file_path).name.lower()

    # Check extension-based categories first
    for category, extensions in RELEVANT_EXTENSIONS.items():
        if file_ext in extensions:
            return category

    # Check content-based detection for files without clear extensions
    content_preview = content[:500].lower()

    # Check for special file types by content
    if any(
        keyword in content_preview
        for keyword in ["select", "insert", "create table", "drop table"]
    ):
        return "sql"

    if any(
        keyword in content_preview
        for keyword in ["#!/bin/bash", "#!/bin/sh", "bash", "shell"]
    ):
        return "script"

    if "<?xml" in content_preview or "<html" in content_preview:
        return "markup"

    # Default fallback
    if file_ext == "":
        return "text"

    return "unknown"


def filter_files_by_relevance(file_contents: dict[str, str]) -> dict[str, str]:
    """Filter files to only include those relevant for metacontext analysis.

    Args:
        file_contents: Dictionary mapping file paths to content

    Returns:
        Filtered dictionary with only relevant files

    """
    filtered_files = {}
    skipped_files = []

    for file_path, content in file_contents.items():
        # Check if file is relevant by extension and patterns
        if not is_file_relevant(file_path):
            skipped_files.append(f"{file_path} (not relevant)")
            continue

        # Check size limits
        should_process, skip_reason = check_file_size_limits(file_path, content)
        if not should_process:
            skipped_files.append(f"{file_path} ({skip_reason})")
            continue

        # File passes all filters
        filtered_files[file_path] = content

    # Log filtering results if there were skipped files
    if skipped_files:
        print(f"Filtered out {len(skipped_files)} files:")
        for skipped in skipped_files[:10]:  # Show first 10
            print(f"  - {skipped}")
        if len(skipped_files) > 10:
            print(f"  ... and {len(skipped_files) - 10} more")

    return filtered_files


def categorize_files_by_type(
    file_contents: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Categorize files by their detected type for specialized processing.

    Args:
        file_contents: Dictionary mapping file paths to content

    Returns:
        Dictionary mapping categories to file dictionaries

    """
    categorized_files = {}

    for file_path, content in file_contents.items():
        file_type = detect_file_type(file_path, content)

        if file_type not in categorized_files:
            categorized_files[file_type] = {}

        categorized_files[file_type][file_path] = content

    return categorized_files


# Performance optimization utilities


def _get_content_hash(content: str) -> str:
    """Generate a hash for content to enable caching."""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


@lru_cache(maxsize=CACHE_SIZE)
def _cached_ast_parse(content_hash: str, content: str) -> ast.AST | None:
    """Parse AST with caching to avoid re-parsing the same content."""
    if not ENABLE_CACHING:
        try:
            return ast.parse(content)
        except SyntaxError:
            return None

    try:
        return ast.parse(content)
    except SyntaxError:
        return None


def _process_file_worker(args: tuple[str, str]) -> tuple[str, dict[str, Any]]:
    """Worker function for parallel file processing."""
    file_path, content = args

    # Extract semantic information for this file
    semantic_results = {
        "constants": [],
        "enums": [],
        "magic_numbers": [],
        "lookup_tables": [],
        "functions": [],
        "data_transformations": [],
        "business_logic": [],
        "scoring_algorithms": [],
    }

    # Only process Python files for advanced semantic extraction
    if not file_path.endswith(".py"):
        return file_path, semantic_results

    try:
        content_hash = _get_content_hash(content)
        tree = _cached_ast_parse(content_hash, content)

        if tree:
            lines = content.split("\n")
            # Import here to avoid circular import
            extractor = AdvancedSemanticExtractor(file_path, lines)
            extractor.visit(tree)

            semantic_results["constants"].extend(extractor.constants)
            semantic_results["enums"].extend(extractor.enums)
            semantic_results["magic_numbers"].extend(extractor.magic_numbers)
            semantic_results["lookup_tables"].extend(extractor.lookup_tables)
            semantic_results["functions"].extend(extractor.functions)
            semantic_results["data_transformations"].extend(
                extractor.data_transformations,
            )
            semantic_results["business_logic"].extend(extractor.business_logic)
            semantic_results["scoring_algorithms"].extend(extractor.scoring_algorithms)

    except Exception:
        # Continue with empty results if processing fails
        pass

    return file_path, semantic_results


def process_files_parallel(
    file_contents: dict[str, str],
    progress_callback=None,
) -> dict[str, dict[str, Any]]:
    """Process multiple files in parallel for better performance.

    Args:
        file_contents: Dictionary mapping file paths to content
        progress_callback: Optional callback function for progress updates

    Returns:
        Dictionary mapping file paths to semantic extraction results

    """
    if not ENABLE_PARALLEL_PROCESSING or len(file_contents) < 3:
        # Fall back to sequential processing for small numbers of files
        return _process_files_sequential(file_contents, progress_callback)

    results = {}
    total_files = len(file_contents)
    completed = 0

    # Prepare worker arguments
    worker_args = list(file_contents.items())

    # Use ThreadPoolExecutor for I/O-bound tasks
    with ThreadPoolExecutor(
        max_workers=min(MAX_WORKERS, len(file_contents)),
    ) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(_process_file_worker, args): args[0] for args in worker_args
        }

        # Collect results as they complete
        for future in as_completed(future_to_file):
            file_path = future_to_file[future]
            try:
                file_path, semantic_results = future.result()
                results[file_path] = semantic_results
                completed += 1

                # Update progress if callback provided
                if progress_callback:
                    progress_callback(completed, total_files, file_path)

            except Exception as e:
                # Handle individual file processing errors gracefully
                results[file_path] = {
                    "constants": [],
                    "enums": [],
                    "magic_numbers": [],
                    "lookup_tables": [],
                    "functions": [],
                    "data_transformations": [],
                    "business_logic": [],
                    "scoring_algorithms": [],
                }
                completed += 1

                if progress_callback:
                    progress_callback(completed, total_files, file_path, error=str(e))

    return results


def _process_files_sequential(
    file_contents: dict[str, str],
    progress_callback=None,
) -> dict[str, dict[str, Any]]:
    """Sequential fallback for file processing."""
    results = {}
    total_files = len(file_contents)

    for i, (file_path, content) in enumerate(file_contents.items(), 1):
        file_path, semantic_results = _process_file_worker((file_path, content))
        results[file_path] = semantic_results

        if progress_callback:
            progress_callback(i, total_files, file_path)

    return results


class ProgressTracker:
    """Simple progress tracker for long operations."""

    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = time.time()
        self.last_update = self.start_time

    def update(
        self,
        completed: int,
        total: int,
        current_item: str = "",
        error: str = "",
    ):
        """Update progress with current status."""
        current_time = time.time()

        # Only update every second to avoid spam
        if current_time - self.last_update < 1.0 and completed < total:
            return

        self.last_update = current_time
        elapsed = current_time - self.start_time
        percent = (completed / total) * 100 if total > 0 else 0

        if error:
            print(
                f"\r{self.operation_name}: {completed}/{total} ({percent:.1f}%) - Error in {current_item}: {error[:50]}",
            )
        else:
            print(
                f"\r{self.operation_name}: {completed}/{total} ({percent:.1f}%) - Processing {Path(current_item).name}",
            )

        if completed >= total:
            print(f"\n{self.operation_name} completed in {elapsed:.2f}s")


# Optimized regex patterns (compiled for better performance)
_COMPILED_PATTERNS = {}


def get_compiled_pattern(pattern: str) -> re.Pattern:
    """Get compiled regex pattern with caching."""
    if pattern not in _COMPILED_PATTERNS:
        _COMPILED_PATTERNS[pattern] = re.compile(pattern, re.IGNORECASE)
    return _COMPILED_PATTERNS[pattern]


def extract_important_sections(
    content: str,
    max_length: int = MAX_FILE_CONTENT_LENGTH,
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
            or (
                "=" in line and len(line.strip()) < MAX_CONFIG_LINE_LENGTH
            )  # Configuration entries
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
    content: str,
    max_length: int = MAX_FILE_CONTENT_LENGTH,
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
    docstring_char: str | None = None

    for line in lines:
        stripped = line.strip()

        # Track docstrings
        if '"""' in line or "'''" in line:
            if not in_docstring:
                in_docstring = True
                docstring_char = '"""' if '"""' in line else "'''"
            elif docstring_char and docstring_char in line:
                in_docstring = False
                docstring_char = None

        # Include function/class definitions and docstrings
        should_include = (
            stripped.startswith(
                ("def ", "class ", "function ", "async def ", "export "),
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
            content,
            file_path,
            per_file_limit,
        )

        if total_used + len(processed_content) > total_limit:
            processed_content = (
                processed_content[: total_limit - total_used - 3] + "..."
            )

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


def extract_column_derivations(
    file_contents: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Extract column derivations and definitions from Python code files.

    This addresses the LLM's difficulty in finding column meaning hints by
    preprocessing code to extract explicit column→definition mappings.

    Args:
        file_contents: Dictionary mapping file paths to file content

    Returns:
        Dictionary mapping column names to their derivation info:
        {
            "column_name": {
                "definition": "df['col'] = expression",
                "source_file": "path/to/file.py",
                "comment": "# explanatory comment if found",
                "line_number": "42"
            }
        }

    """
    derivations = {}
    min_groups_for_assignment = 2  # Constant for magic number
    min_groups_for_named_df = 3  # Constant for named dataframe patterns

    for file_path, content in file_contents.items():
        if not file_path.endswith(".py"):
            continue

        lines = content.split("\n")

        for i, original_line in enumerate(lines, 1):
            current_line = original_line.strip()

            # Check each pattern for column derivations
            for pattern in COLUMN_DERIVATION_PATTERNS:
                match = re.search(pattern, current_line, re.IGNORECASE)
                if match:
                    if pattern.startswith("#"):
                        # Comment-based derivation: # derived: col = expression
                        column_name = match.group(1)
                        definition = match.group(2)
                        comment = current_line
                    elif "df[" in pattern:
                        # DataFrame assignment: df["col"] = expression
                        column_name = match.group(1)
                        definition = match.group(2)
                        comment = _find_nearby_comment(lines, i)
                    elif len(match.groups()) >= min_groups_for_named_df:
                        # Named dataframe: var_df["col"] = expression or var["col"] = expression
                        column_name = match.group(2)  # Column name is second group
                        definition = match.group(3)  # Definition is third group
                        comment = _find_nearby_comment(lines, i)
                    else:
                        # Simple assignment or astype: var = expression
                        column_name = match.group(1)
                        definition = f"{match.group(2) if len(match.groups()) > min_groups_for_assignment else ''}"
                        comment = _find_nearby_comment(lines, i)

                    derivations[column_name] = {
                        "definition": definition.strip(),
                        "source_file": file_path,
                        "comment": comment or "",
                        "line_number": str(i),
                        "full_line": current_line,
                    }

    return derivations


def _find_nearby_comment(
    lines: list[str],
    line_index: int,
    search_range: int = 3,
) -> str:
    """Find explanatory comments near a line of code."""
    # Check lines before and after for comments
    start = max(0, line_index - search_range - 1)
    end = min(len(lines), line_index + search_range)

    for i in range(start, end):
        line = lines[i].strip()
        if line.startswith("#") and any(
            keyword in line.lower()
            for keyword in ["derived", "feature", "column", "calculated", "represents"]
        ):
            return line

    return ""


def format_column_derivations_for_llm(derivations: dict[str, dict[str, str]]) -> str:
    """Format column derivations in a structure optimized for LLM consumption.

    Args:
        derivations: Output from extract_column_derivations()

    Returns:
        Formatted string that clearly shows column meanings for LLM analysis

    """
    if not derivations:
        return "No column derivations found in codebase."

    sections = ["=== COLUMN DERIVATIONS & DEFINITIONS ==="]

    for column_name, info in derivations.items():
        sections.append(f"\n🔸 Column: {column_name}")
        sections.append(f"   Definition: {info['definition']}")
        sections.append(f"   Source: {info['source_file']}:{info['line_number']}")

        if info["comment"]:
            sections.append(f"   Comment: {info['comment']}")

        sections.append(f"   Full line: {info['full_line']}")

    sections.append("\n=== END COLUMN DERIVATIONS ===")
    return "\n".join(sections)


def _extract_business_logic_comments(
    lines: list[str],
    file_path: str,
) -> list[dict[str, Any]]:
    """Extract business logic comments from file lines."""
    business_logic_comments = []
    for i, line in enumerate(lines, 1):
        for pattern in BUSINESS_LOGIC_PATTERNS:
            match = re.search(pattern, line)
            if match:
                # Extract type from the actual matched text
                pattern_type = "comment"
                line_lower = line.lower()
                if "business rule:" in line_lower:
                    pattern_type = "business_rule"
                elif "algorithm:" in line_lower:
                    pattern_type = "algorithm"
                elif "logic:" in line_lower:
                    pattern_type = "logic"
                elif "note:" in line_lower:
                    pattern_type = "note"
                elif "important:" in line_lower:
                    pattern_type = "important"
                elif "warning:" in line_lower:
                    pattern_type = "warning"

                business_logic_comments.append(
                    {
                        "file": file_path,
                        "line": i,
                        "content": match.group(1).strip(),
                        "type": pattern_type,
                    },
                )

    return business_logic_comments


def extract_comment_patterns(file_contents: dict[str, str]) -> dict[str, list[str]]:
    """Extract various comment patterns from file contents.

    Args:
        file_contents: Dictionary mapping file paths to their content

    Returns:
        Dictionary with extracted comment information categorized by type

    """
    extracted_comments = {
        "multiline_comments": [],
        "business_logic": [],
        "constants": [],
        "data_dictionaries": [],
    }

    for file_path, content in file_contents.items():
        lines = content.split("\n")

        # Extract multiline comment blocks
        for pattern in MULTILINE_COMMENT_PATTERNS:
            matches = re.finditer(pattern, content, re.DOTALL | re.IGNORECASE)
            for match in matches:
                extracted_comments["multiline_comments"].append(
                    {
                        "file": file_path,
                        "content": match.group(1).strip(),
                        "type": "docstring" if '"""' in pattern else "comment_block",
                    },
                )

        # Extract business logic comments using helper function
        extracted_comments["business_logic"].extend(
            _extract_business_logic_comments(lines, file_path),
        )

        # Extract constant definitions with explanations
        for i, line in enumerate(lines, 1):
            found_constant = False
            for pattern in CONSTANT_DEFINITION_PATTERNS:
                if found_constant:
                    break
                match = re.search(pattern, line)
                if match:
                    extracted_comments["constants"].append(
                        {
                            "file": file_path,
                            "line": i,
                            "name": match.group(1).strip(),
                            "value": match.group(2).strip(),
                            "explanation": match.group(3).strip(),
                        },
                    )
                    found_constant = True

        # Extract data dictionary blocks
        for pattern in DATA_DICTIONARY_PATTERNS:
            matches = re.finditer(pattern, content, re.MULTILINE | re.IGNORECASE)
            for match in matches:
                block_content = match.group(1).strip()
                if block_content:
                    extracted_comments["data_dictionaries"].append(
                        {
                            "file": file_path,
                            "content": block_content,
                            "type": "data_dictionary",
                        },
                    )

    return extracted_comments


def format_comment_patterns_for_llm(comment_data: dict[str, list[str]]) -> str:
    """Format extracted comment patterns for LLM consumption.

    Args:
        comment_data: Dictionary of extracted comment patterns

    Returns:
        Formatted string suitable for LLM prompts

    """
    # Minimum content length for multiline comments to include
    min_comment_length = 100

    if not any(comment_data.values()):
        return ""

    sections = ["\n=== EXTRACTED SEMANTIC INFORMATION ==="]

    # Format business logic comments
    if comment_data["business_logic"]:
        sections.append("\n--- Business Logic & Rules ---")
        for item in comment_data["business_logic"]:
            sections.append(f"• {item['type'].title()}: {item['content']}")
            sections.append(f"  Source: {item['file']}:{item['line']}")

    # Format constants with explanations
    if comment_data["constants"]:
        sections.append("\n--- Important Constants ---")
        for item in comment_data["constants"]:
            sections.append(f"• {item['name']} = {item['value']}")
            sections.append(f"  Purpose: {item['explanation']}")
            sections.append(f"  Source: {item['file']}:{item['line']}")

    # Format data dictionary blocks
    if comment_data["data_dictionaries"]:
        sections.append("\n--- Data Dictionaries & Field Definitions ---")
        for item in comment_data["data_dictionaries"]:
            sections.append(f"• From {item['file']}:")
            # Clean up comment block content
            content_lines = [
                line.lstrip("# ").strip()
                for line in item["content"].split("\n")
                if line.strip()
            ]
            sections.extend(f"  {line}" for line in content_lines if line)

    # Format multiline comments/docstrings
    if comment_data["multiline_comments"]:
        sections.append("\n--- Documentation & Comments ---")
        for item in comment_data["multiline_comments"]:
            if (
                len(item["content"]) > min_comment_length
            ):  # Only include substantial content
                sections.append(f"• {item['type'].title()} from {item['file']}:")
                sections.append(f"  {item['content'][:200]}...")

    sections.append("\n=== END SEMANTIC INFORMATION ===")
    return "\n".join(sections)


class PydanticSchemaExtractor(ast.NodeVisitor):
    """AST visitor for extracting Pydantic schema information."""

    def __init__(self, file_path: str, lines: list[str]) -> None:
        """Initialize the Pydantic schema extractor.

        Args:
            file_path: Path to the file being analyzed
            lines: List of lines from the file for line number mapping

        """
        self.file_path = file_path
        self.lines = lines
        self.models: list[dict[str, Any]] = []
        self.enums: list[dict[str, Any]] = []
        self.current_class: dict[str, Any] | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract Pydantic model class definitions."""
        # Check if this is a Pydantic model
        is_pydantic_model = any(self._is_base_model(base) for base in node.bases)

        if is_pydantic_model:
            docstring = ast.get_docstring(node)
            base_classes = [ast.unparse(base) for base in node.bases]

            model_info = {
                "name": node.name,
                "line": node.lineno,
                "docstring": docstring,
                "base_classes": base_classes,
                "fields": [],
                "validators": [],
                "config": {},
                "file": self.file_path,
            }

            self.current_class = model_info

            # Process class body to extract fields and validators
            for child in node.body:
                if isinstance(child, ast.AnnAssign) and isinstance(
                    child.target,
                    ast.Name,
                ):
                    field_info = self._extract_field_info(child)
                    if field_info:
                        model_info["fields"].append(field_info)
                elif isinstance(child, ast.FunctionDef):
                    # Check if function has validator decorators
                    validator_info = self._extract_validator_info(child)
                    if validator_info:
                        model_info["validators"].append(validator_info)

            self.models.append(model_info)
            self.current_class = None

        # Check if this is an Enum
        elif any("Enum" in ast.unparse(base) for base in node.bases):
            enum_info = self._extract_enum_info(node)
            if enum_info:
                self.enums.append(enum_info)

        self.generic_visit(node)

    def _is_base_model(self, base: ast.expr) -> bool:
        """Check if a base class is BaseModel or derived from it."""
        base_str = ast.unparse(base)
        # More comprehensive check for Pydantic-related base classes
        pydantic_indicators = [
            "BaseModel",
            "ExtensionContext",  # Our custom base
            "AIEnrichment",  # Another custom base in our schema
            "EnrichmentProvider",  # Another custom base
            "MetadataProvider",  # Another custom base
        ]
        return any(indicator in base_str for indicator in pydantic_indicators)

    def _extract_field_info(self, node: ast.AnnAssign) -> dict[str, Any] | None:
        """Extract information about a Pydantic field."""
        field_name = node.target.id
        type_hint = ast.unparse(node.annotation) if node.annotation else None

        field_info = {
            "name": field_name,
            "type_hint": type_hint,
            "line": node.lineno,
            "description": None,
            "default_value": None,
            "alias": None,
            "validation_rules": [],
        }

        # Extract Field() information if present
        if node.value:
            field_call_info = self._extract_field_call_info(node.value)
            field_info.update(field_call_info)

        return field_info

    def _extract_field_call_info(self, value_node: ast.expr) -> dict[str, Any]:
        """Extract information from Field() call."""
        field_info = {
            "description": None,
            "default_value": None,
            "alias": None,
            "validation_rules": [],
        }

        if isinstance(value_node, ast.Call):
            func_name = (
                ast.unparse(value_node.func) if hasattr(value_node, "func") else ""
            )

            if "Field" in func_name:
                # Extract arguments from Field() call
                for keyword in value_node.keywords:
                    if keyword.arg == "description" and isinstance(
                        keyword.value,
                        ast.Constant,
                    ):
                        field_info["description"] = keyword.value.value
                    elif keyword.arg == "alias" and isinstance(
                        keyword.value,
                        ast.Constant,
                    ):
                        field_info["alias"] = keyword.value.value
                    elif keyword.arg == "default":
                        field_info["default_value"] = ast.unparse(keyword.value)
                    elif keyword.arg in [
                        "gt",
                        "ge",
                        "lt",
                        "le",
                        "min_length",
                        "max_length",
                        "regex",
                    ]:
                        rule_value = ast.unparse(keyword.value)
                        field_info["validation_rules"].append(
                            f"{keyword.arg}: {rule_value}",
                        )

                # Also check positional arguments (first one is usually default)
                if value_node.args and len(value_node.args) > 0:
                    first_arg = value_node.args[0]
                    if (
                        field_info["default_value"] is None
                    ):  # Only if not set by keyword
                        field_info["default_value"] = ast.unparse(first_arg)
        else:
            # Simple default value (not a Field call)
            field_info["default_value"] = ast.unparse(value_node)

        return field_info

    def _extract_validator_info(self, node: ast.FunctionDef) -> dict[str, Any] | None:
        """Extract information about Pydantic validators."""
        docstring = ast.get_docstring(node)

        # Extract field names from decorator
        field_names = []
        has_validator_decorator = False
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Call):
                func_name = ast.unparse(decorator.func)
                if "validator" in func_name.lower():
                    has_validator_decorator = True
                    field_names.extend(
                        [
                            arg.value
                            for arg in decorator.args
                            if isinstance(arg, ast.Constant)
                        ],
                    )
            elif isinstance(decorator, ast.Name):
                if "validator" in decorator.id.lower():
                    has_validator_decorator = True

        # Only return validator info if this function actually has validator decorators
        if not has_validator_decorator:
            return None

        return {
            "name": node.name,
            "line": node.lineno,
            "docstring": docstring,
            "field_names": field_names,
            "file": self.file_path,
        }

    def _extract_enum_info(self, node: ast.ClassDef) -> dict[str, Any] | None:
        """Extract information about Enum classes."""
        docstring = ast.get_docstring(node)
        values = []

        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        value_name = target.id
                        value_repr = ast.unparse(child.value)

                        # Look for inline comment
                        comment = None
                        if child.lineno <= len(self.lines):
                            line = self.lines[child.lineno - 1]
                            if "#" in line:
                                comment = line.split("#", 1)[1].strip()

                        values.append(
                            {
                                "name": value_name,
                                "value": value_repr,
                                "comment": comment,
                                "line": child.lineno,
                            },
                        )

        return {
            "name": node.name,
            "line": node.lineno,
            "docstring": docstring,
            "values": values,
            "file": self.file_path,
        }


def _extract_model_relationships(models: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract relationships between Pydantic models.

    Args:
        models: List of model dictionaries from PydanticSchemaExtractor

    Returns:
        List of relationship dictionaries with type, source, target, and description

    """
    relationships = []

    # Create a set of all model names for quick lookup
    model_names = {model["name"] for model in models}

    for model in models:
        model_name = model["name"]

        # 1. Inheritance relationships
        inheritance_relationships = [
            {
                "type": "inheritance",
                "source": model_name,
                "target": base_class,
                "description": f"{model_name} inherits from {base_class}",
                "file": model["file"],
            }
            for base_class in model["base_classes"]
            if base_class in model_names
        ]
        relationships.extend(inheritance_relationships)

        # 2. Composition relationships (nested models in fields)
        for field in model["fields"]:
            field_type = field["type_hint"]
            if field_type:
                # Extract model names from type hints (handle Union types, Optional, etc.)
                referenced_models = _extract_referenced_models(field_type, model_names)
                composition_relationships = [
                    {
                        "type": "composition",
                        "source": model_name,
                        "target": ref_model,
                        "field": field["name"],
                        "description": f"{model_name}.{field['name']} contains {ref_model}",
                        "file": model["file"],
                    }
                    for ref_model in referenced_models
                ]
                relationships.extend(composition_relationships)

    return relationships


def _extract_referenced_models(type_hint: str, model_names: set[str]) -> list[str]:
    """Extract model names referenced in a type hint.

    Args:
        type_hint: String representation of the type hint
        model_names: Set of all available model names

    Returns:
        List of model names found in the type hint

    """
    referenced = []

    # Split by common type separators and check each part
    # Handle cases like: "Optional[ModelName]", "Union[Model1, Model2]", "list[ModelName]"
    parts = (
        type_hint.replace("|", " ")
        .replace(",", " ")
        .replace("[", " ")
        .replace("]", " ")
        .split()
    )

    for part in parts:
        # Clean up the part (remove quotes, whitespace, etc.)
        clean_part = part.strip().strip("'\"")
        if clean_part in model_names:
            referenced.append(clean_part)

    return referenced


def extract_pydantic_schema_information(
    file_contents: dict[str, str],
) -> dict[str, Any]:
    """Extract Pydantic schema information using AST parsing.

    Args:
        file_contents: Dictionary mapping file paths to their content

    Returns:
        Dictionary with extracted Pydantic schema information

    """
    schema_info = {
        "models": [],
        "enums": [],
        "field_descriptions": [],
        "validation_rules": [],
        "model_relationships": [],
    }

    for file_path, content in file_contents.items():
        # Only process Python files
        file_path_str = str(file_path)
        if not file_path_str.endswith(".py"):
            continue

        try:
            tree = ast.parse(content)
            lines = content.split("\n")

            extractor = PydanticSchemaExtractor(file_path_str, lines)
            extractor.visit(tree)

            # Collect results
            schema_info["models"].extend(extractor.models)
            schema_info["enums"].extend(extractor.enums)

            # Extract field descriptions for easy access
            for model in extractor.models:
                for field in model["fields"]:
                    if field["description"]:
                        schema_info["field_descriptions"].append(
                            {
                                "model": model["name"],
                                "field": field["name"],
                                "description": field["description"],
                                "type": field["type_hint"],
                                "file": file_path,
                            },
                        )

                    if field["validation_rules"]:
                        schema_info["validation_rules"].append(
                            {
                                "model": model["name"],
                                "field": field["name"],
                                "rules": field["validation_rules"],
                                "type": "field_constraint",
                                "file": file_path,
                            },
                        )

                # Also collect model-level validators
                for validator in model["validators"]:
                    schema_info["validation_rules"].append(
                        {
                            "model": model["name"],
                            "validator_name": validator["name"],
                            "field_names": validator["field_names"],
                            "docstring": validator["docstring"],
                            "type": "validator_function",
                            "line": validator["line"],
                            "file": file_path,
                        },
                    )

        except SyntaxError:
            # Skip files with syntax errors
            continue

    # Extract model relationships after all models are collected
    schema_info["model_relationships"] = _extract_model_relationships(
        schema_info["models"],
    )

    return schema_info


class AdvancedSemanticExtractor(ast.NodeVisitor):
    """AST visitor for advanced semantic extraction: enums, constants, and magic numbers."""

    def __init__(self, file_path: str, lines: list[str]) -> None:
        """Initialize the advanced semantic extractor.

        Args:
            file_path: Path to the file being analyzed
            lines: List of lines from the file for context extraction

        """
        self.file_path = file_path
        self.lines = lines
        self.constants: list[dict[str, Any]] = []
        self.enums: list[dict[str, Any]] = []
        self.magic_numbers: list[dict[str, Any]] = []
        self.lookup_tables: list[dict[str, Any]] = []
        self.functions: list[dict[str, Any]] = []
        self.data_transformations: list[dict[str, Any]] = []
        self.business_logic: list[dict[str, Any]] = []
        self.scoring_algorithms: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions with data transformation analysis."""
        function_info = self._extract_function_analysis(node)
        if function_info:
            self.functions.append(function_info)

        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        """Extract conditional logic patterns and business rules."""
        logic_info = self._extract_business_logic(node)
        if logic_info:
            self.business_logic.append(logic_info)

        self.generic_visit(node)

    def visit_Compare(self, node: ast.Compare) -> None:
        """Extract comparison operations that may represent scoring logic."""
        scoring_info = self._extract_scoring_logic(node)
        if scoring_info:
            self.scoring_algorithms.append(scoring_info)

        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract enhanced enum definitions."""
        # Check if this is an Enum class
        is_enum = any("Enum" in ast.unparse(base) for base in node.bases)

        if is_enum:
            enum_info = self._extract_enhanced_enum_info(node)
            if enum_info:
                self.enums.append(enum_info)

        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract constants, magic numbers, and lookup tables."""
        # Process each target in the assignment
        for target in node.targets:
            if isinstance(target, ast.Name):
                var_name = target.id

                # Check if this looks like a constant
                if self._is_constant_name(var_name):
                    constant_info = self._extract_constant_info(node, var_name)
                    if constant_info:
                        self.constants.append(constant_info)

                # Check if this is a lookup table (dict assignment)
                elif isinstance(node.value, ast.Dict):
                    lookup_info = self._extract_lookup_table_info(node, var_name)
                    if lookup_info:
                        self.lookup_tables.append(lookup_info)

        # Check for magic numbers in any assignment
        self._extract_magic_numbers_from_node(node)
        self.generic_visit(node)

    def _is_constant_name(self, name: str) -> bool:
        """Determine if a variable name follows constant naming conventions."""
        # All uppercase with underscores
        if name.isupper() and "_" in name:
            return True
        # Common constant patterns
        constant_patterns = [
            "_THRESHOLD",
            "_LIMIT",
            "_MAX",
            "_MIN",
            "_SIZE",
            "_CONFIG",
            "_SETTING",
            "_PARAM",
            "_DEFAULT",
            "_DIR",
            "_PATH",
            "_URL",
            "_KEY",
        ]
        return any(pattern in name.upper() for pattern in constant_patterns)

    def _extract_constant_info(
        self,
        node: ast.Assign,
        var_name: str,
    ) -> dict[str, Any] | None:
        """Extract detailed information about a constant definition."""
        line_num = node.lineno

        # Get the value representation
        try:
            value_repr = ast.unparse(node.value)
        except Exception:
            value_repr = "<complex_expression>"

        # Extract comment from the same line or nearby lines
        comment = self._get_comment_for_line(line_num)

        # Determine constant type and semantic meaning
        constant_type = self._classify_constant_type(var_name, node.value)
        semantic_meaning = self._infer_constant_meaning(var_name, value_repr, comment)

        return {
            "name": var_name,
            "value": value_repr,
            "line": line_num,
            "comment": comment,
            "type": constant_type,
            "semantic_meaning": semantic_meaning,
            "file": self.file_path,
        }

    def _extract_lookup_table_info(
        self,
        node: ast.Assign,
        var_name: str,
    ) -> dict[str, Any] | None:
        """Extract information about dictionary-based lookup tables."""
        if not isinstance(node.value, ast.Dict):
            return None

        line_num = node.lineno
        comment = self._get_comment_for_line(line_num)

        # Extract key-value pairs with their semantic meanings
        mappings = []
        for key_node, value_node in zip(
            node.value.keys,
            node.value.values,
            strict=False,
        ):
            if key_node and value_node:
                try:
                    key_repr = ast.unparse(key_node)
                    value_repr = ast.unparse(value_node)
                    mappings.append(
                        {
                            "key": key_repr,
                            "value": value_repr,
                            "key_type": type(key_node).__name__,
                            "value_type": type(value_node).__name__,
                        },
                    )
                except Exception:
                    continue

        return {
            "name": var_name,
            "line": line_num,
            "comment": comment,
            "mappings": mappings[:10],  # Limit to first 10 entries
            "total_entries": len(node.value.keys),
            "purpose": self._infer_lookup_purpose(var_name, mappings, comment),
            "file": self.file_path,
        }

    def _extract_enhanced_enum_info(self, node: ast.ClassDef) -> dict[str, Any] | None:
        """Extract enhanced enum information with semantic meanings."""
        docstring = ast.get_docstring(node)
        values = []

        for child in node.body:
            if isinstance(child, ast.Assign):
                for target in child.targets:
                    if isinstance(target, ast.Name):
                        value_name = target.id

                        # Skip special enum methods
                        if value_name.startswith("_"):
                            continue

                        try:
                            value_repr = ast.unparse(child.value)
                        except Exception:
                            value_repr = "<complex_expression>"

                        # Get comment for this enum value
                        comment = self._get_comment_for_line(child.lineno)

                        # Infer semantic meaning from name and comment
                        semantic_meaning = self._infer_enum_value_meaning(
                            node.name,
                            value_name,
                            value_repr,
                            comment,
                        )

                        values.append(
                            {
                                "name": value_name,
                                "value": value_repr,
                                "comment": comment,
                                "semantic_meaning": semantic_meaning,
                                "line": child.lineno,
                            },
                        )

        return {
            "name": node.name,
            "line": node.lineno,
            "docstring": docstring,
            "values": values,
            "domain": self._infer_enum_domain(node.name, docstring, values),
            "file": self.file_path,
        }

    def _extract_magic_numbers_from_node(self, node: ast.AST) -> None:
        """Extract magic numbers from an AST node."""
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(
                child.value,
                (int, float),
            ):
                # Skip obvious non-magic numbers
                if child.value in [0, 1, -1, 0.0, 1.0]:
                    continue

                line_num = getattr(child, "lineno", node.lineno)
                comment = self._get_comment_for_line(line_num)

                # Get context by looking at the surrounding code
                context = self._get_magic_number_context(child, node)

                magic_info = {
                    "value": child.value,
                    "line": line_num,
                    "comment": comment,
                    "context": context,
                    "semantic_guess": self._guess_magic_number_meaning(
                        child.value,
                        context,
                        comment,
                    ),
                    "file": self.file_path,
                }

                # Avoid duplicates
                if not any(
                    m["value"] == magic_info["value"]
                    and m["line"] == magic_info["line"]
                    for m in self.magic_numbers
                ):
                    self.magic_numbers.append(magic_info)

    def _get_comment_for_line(self, line_num: int) -> str:
        """Extract comment from a specific line or nearby lines."""
        if line_num <= len(self.lines):
            line = self.lines[line_num - 1]  # Convert to 0-based indexing
            if "#" in line:
                return line.split("#", 1)[1].strip()

        # Check the next line for comments
        if line_num < len(self.lines):
            next_line = self.lines[line_num]
            if next_line.strip().startswith("#"):
                return next_line.strip()[1:].strip()

        return ""

    def _classify_constant_type(self, name: str, value_node: ast.AST) -> str:
        """Classify the type of constant based on name and value."""
        name_upper = name.upper()

        if any(pattern in name_upper for pattern in ["_PATH", "_DIR", "_FILE"]):
            return "path_constant"
        if any(pattern in name_upper for pattern in ["_URL", "_ENDPOINT", "_HOST"]):
            return "url_constant"
        if any(
            pattern in name_upper
            for pattern in ["_THRESHOLD", "_LIMIT", "_MAX", "_MIN"]
        ):
            return "threshold_constant"
        if any(pattern in name_upper for pattern in ["_CONFIG", "_SETTING", "_PARAM"]):
            return "configuration_constant"
        if isinstance(value_node, ast.Constant) and isinstance(
            value_node.value,
            (int, float),
        ):
            return "numeric_constant"
        if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str):
            return "string_constant"
        return "general_constant"

    def _infer_constant_meaning(self, name: str, value: str, comment: str) -> str:
        """Infer the semantic meaning of a constant."""
        if comment:
            return comment

        # Generate meaning from name patterns
        name_parts = name.lower().split("_")

        if "threshold" in name_parts:
            return f"Threshold value for {' '.join(name_parts[:-1])}"
        if "limit" in name_parts:
            return f"Limit for {' '.join(name_parts[:-1])}"
        if "size" in name_parts:
            return f"Size specification for {' '.join(name_parts[:-1])}"
        if "path" in name_parts or "dir" in name_parts:
            return f"File system path: {value}"
        if "config" in name_parts:
            return f"Configuration setting: {' '.join(name_parts)}"
        return f"Constant value: {value}"

    def _infer_lookup_purpose(
        self,
        name: str,
        mappings: list[dict],
        comment: str,
    ) -> str:
        """Infer the purpose of a lookup table."""
        if comment:
            return comment

        name_lower = name.lower()
        if "mapping" in name_lower or "map" in name_lower:
            return f"Mapping table: {name}"
        if "lookup" in name_lower:
            return f"Lookup table: {name}"
        if "config" in name_lower:
            return f"Configuration mapping: {name}"
        if len(mappings) > 0:
            first_mapping = mappings[0]
            return f"Dictionary mapping {first_mapping['key_type']} to {first_mapping['value_type']}"
        return f"Data structure: {name}"

    def _infer_enum_value_meaning(
        self,
        enum_name: str,
        value_name: str,
        value_repr: str,
        comment: str,
    ) -> str:
        """Infer the semantic meaning of an enum value."""
        if comment:
            return comment

        # Generate meaning from enum and value names
        enum_lower = enum_name.lower()
        value_lower = value_name.lower()

        if "type" in enum_lower:
            return f"Type: {value_name.replace('_', ' ').title()}"
        if "status" in enum_lower:
            return f"Status: {value_name.replace('_', ' ').title()}"
        if "method" in enum_lower:
            return f"Method: {value_name.replace('_', ' ').title()}"
        if "level" in enum_lower:
            return f"Level: {value_name.replace('_', ' ').title()}"
        return f"{enum_name} option: {value_name.replace('_', ' ').title()}"

    def _infer_enum_domain(
        self,
        enum_name: str,
        docstring: str,
        values: list[dict],
    ) -> str:
        """Infer the domain/category of an enum."""
        if docstring:
            first_line = docstring.split("\n")[0].strip()
            return first_line

        name_lower = enum_name.lower()
        if "type" in name_lower:
            return "Type classification"
        if "status" in name_lower:
            return "Status tracking"
        if "method" in name_lower:
            return "Method selection"
        if "level" in name_lower:
            return "Level specification"
        if "config" in name_lower:
            return "Configuration options"
        return f"Enumerated values for {enum_name}"

    def _get_magic_number_context(
        self,
        number_node: ast.Constant,
        parent_node: ast.AST,
    ) -> str:
        """Get context information for a magic number."""
        try:
            # Get a small snippet of the code containing the number
            if hasattr(parent_node, "lineno"):
                line_idx = parent_node.lineno - 1
                if 0 <= line_idx < len(self.lines):
                    return self.lines[line_idx].strip()

            return ast.unparse(parent_node)[:50] + "..."
        except Exception:
            return "unknown_context"

    def _guess_magic_number_meaning(
        self,
        value: float,
        context: str,
        comment: str,
    ) -> str:
        """Guess the semantic meaning of a magic number."""
        if comment:
            return comment

        # Common magic number patterns
        if value == 255:
            return "Maximum RGB color value"
        if value == 100:
            return "Percentage maximum or iteration count"
        if value == 1000:
            return "Milliseconds per second or large iteration count"
        if value == 3600:
            return "Seconds per hour"
        if value == 24:
            return "Hours per day"
        if value == 7:
            return "Days per week"
        if value == 365:
            return "Days per year"
        if isinstance(value, float) and 0 < value < 1:
            return "Probability, fraction, or rate"
        if value > 1000000:
            return "Large scale factor or byte size"
        # Try to infer from context
        context_lower = context.lower()
        if "width" in context_lower or "height" in context_lower:
            return f"Dimension value: {value}"
        if "size" in context_lower:
            return f"Size specification: {value}"
        if "threshold" in context_lower:
            return f"Threshold value: {value}"
        if "limit" in context_lower:
            return f"Limit value: {value}"
        return f"Numeric literal: {value}"

    def _extract_function_analysis(
        self,
        node: ast.FunctionDef,
    ) -> dict[str, Any] | None:
        """Extract function analysis focusing on data transformations and business logic."""
        try:
            docstring = ast.get_docstring(node)

            # Skip private functions unless they have significant business logic
            if node.name.startswith("_") and len(node.body) < 3:
                return None

            # Extract function signature information
            args = []
            for arg in node.args.args:
                arg_info = {"name": arg.arg}
                if arg.annotation:
                    try:
                        arg_info["type"] = ast.unparse(arg.annotation)
                    except Exception:
                        arg_info["type"] = "unknown"
                args.append(arg_info)

            # Extract return type annotation
            return_type = None
            if node.returns:
                try:
                    return_type = ast.unparse(node.returns)
                except Exception:
                    return_type = "unknown"

            # Parse docstring for data transformation patterns
            creates_data = []
            returns_data = []
            modifies_data = []

            if docstring:
                lines = docstring.split("\n")
                current_section = None

                for line in lines:
                    line = line.strip()
                    line_lower = line.lower()

                    if line_lower.startswith("creates:") or "creates " in line_lower:
                        current_section = "creates"
                        content = line.split(":", 1)
                        if len(content) > 1:
                            creates_data.append(content[1].strip())
                    elif line_lower.startswith("returns:") or "returns " in line_lower:
                        current_section = "returns"
                        content = line.split(":", 1)
                        if len(content) > 1:
                            returns_data.append(content[1].strip())
                    elif (
                        line_lower.startswith("modifies:") or "modifies " in line_lower
                    ):
                        current_section = "modifies"
                        content = line.split(":", 1)
                        if len(content) > 1:
                            modifies_data.append(content[1].strip())
                    elif (
                        current_section
                        and line
                        and not line_lower.startswith(
                            (
                                "args:",
                                "arguments:",
                                "parameters:",
                                "raises:",
                                "examples:",
                            ),
                        )
                    ):
                        # Continue current section
                        if current_section == "creates":
                            creates_data.append(line)
                        elif current_section == "returns":
                            returns_data.append(line)
                        elif current_section == "modifies":
                            modifies_data.append(line)
                    else:
                        current_section = None

            # Analyze function body for data transformation patterns
            transforms_data = self._analyze_data_transformations(node)

            # Check if this is a data transformation function
            is_data_transform = bool(
                creates_data
                or returns_data
                or modifies_data
                or transforms_data
                or any(
                    keyword in node.name.lower()
                    for keyword in [
                        "transform",
                        "convert",
                        "process",
                        "parse",
                        "format",
                        "create",
                        "generate",
                        "build",
                        "make",
                        "extract",
                    ]
                ),
            )

            function_info = {
                "name": node.name,
                "line": node.lineno,
                "args": args,
                "return_type": return_type,
                "docstring": docstring[:200]
                if docstring
                else None,  # Truncate for token efficiency
                "creates": creates_data,
                "returns": returns_data,
                "modifies": modifies_data,
                "data_transformations": transforms_data,
                "is_data_transform": is_data_transform,
                "complexity": len(node.body),  # Simple complexity metric
                "file": self.file_path,
            }

            # Track as data transformation if it meets criteria
            if is_data_transform:
                self.data_transformations.append(
                    {
                        "function": node.name,
                        "type": self._classify_transformation_type(
                            node,
                            creates_data,
                            returns_data,
                            modifies_data,
                        ),
                        "input_data": [arg["name"] for arg in args],
                        "output_data": returns_data or [return_type]
                        if return_type
                        else [],
                        "line": node.lineno,
                        "file": self.file_path,
                    },
                )

            return function_info

        except Exception:
            return None

    def _analyze_data_transformations(self, node: ast.FunctionDef) -> list[str]:
        """Analyze function body for data transformation operations."""
        transformations = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call) and hasattr(child.func, "attr"):
                method_name = child.func.attr

                # Common data transformation methods
                if method_name in [
                    "map",
                    "filter",
                    "reduce",
                    "apply",
                    "transform",
                    "groupby",
                    "merge",
                    "join",
                    "concat",
                    "pivot",
                    "melt",
                    "sort_values",
                    "drop",
                    "fillna",
                    "replace",
                    "rename",
                    "astype",
                ]:
                    transformations.append(f"DataFrame.{method_name}()")
                elif method_name in ["append", "extend", "insert", "remove", "pop"]:
                    transformations.append(f"List.{method_name}()")
                elif method_name in ["update", "pop", "get", "setdefault"]:
                    transformations.append(f"Dict.{method_name}()")

            elif isinstance(child, ast.ListComp):
                transformations.append("List comprehension")
            elif isinstance(child, ast.DictComp):
                transformations.append("Dict comprehension")
            elif isinstance(child, ast.GeneratorExp):
                transformations.append("Generator expression")

        return list(set(transformations))  # Remove duplicates

    def _classify_transformation_type(
        self,
        node: ast.FunctionDef,
        creates: list,
        returns: list,
        modifies: list,
    ) -> str:
        """Classify the type of data transformation."""
        name_lower = node.name.lower()

        if (
            creates
            or "create" in name_lower
            or "build" in name_lower
            or "make" in name_lower
        ):
            return "creator"
        if (
            modifies
            or "update" in name_lower
            or "modify" in name_lower
            or "change" in name_lower
        ):
            return "modifier"
        if "parse" in name_lower or "extract" in name_lower or "load" in name_lower:
            return "extractor"
        if (
            "format" in name_lower
            or "convert" in name_lower
            or "transform" in name_lower
        ):
            return "transformer"
        if returns or "get" in name_lower or "fetch" in name_lower:
            return "accessor"
        return "processor"

    def _extract_business_logic(self, node: ast.If) -> dict[str, Any] | None:
        """Extract business logic from conditional statements."""
        try:
            # Get the condition and line info
            condition_code = ast.unparse(node.test)
            line = node.lineno

            # Skip simple/trivial conditions
            if len(condition_code) < 10:
                return None

            # Analyze condition complexity
            complexity = self._analyze_condition_complexity(node.test)
            if complexity < 2:  # Skip simple boolean checks
                return None

            # Extract threshold values from conditions
            thresholds = self._extract_thresholds_from_condition(node.test)

            # Analyze the consequences - what happens in if/else branches
            if_actions = self._extract_branch_actions(node.body)
            else_actions = []
            if node.orelse:
                else_actions = self._extract_branch_actions(node.orelse)

            # Determine business logic type
            logic_type = self._classify_business_logic(
                condition_code,
                if_actions,
                else_actions,
            )

            return {
                "condition": condition_code[:100],  # Truncate for token efficiency
                "line": line,
                "complexity": complexity,
                "thresholds": thresholds,
                "if_actions": if_actions,
                "else_actions": else_actions,
                "logic_type": logic_type,
                "file": self.file_path,
            }

        except Exception:
            return None

    def _extract_scoring_logic(self, node: ast.Compare) -> dict[str, Any] | None:
        """Extract scoring and calculation logic from comparison operations."""
        try:
            # Get the comparison expression
            comparison_code = ast.unparse(node)
            line = node.lineno

            # Skip simple variable comparisons
            if not any(op in comparison_code for op in ["*", "+", "-", "/", "**"]):
                return None

            # Extract the left side (often the calculated value)
            left_side = ast.unparse(node.left)

            # Extract comparison operators and values
            ops = []
            comparators = []
            for op, comp in zip(node.ops, node.comparators, strict=False):
                ops.append(type(op).__name__)
                try:
                    comparators.append(ast.unparse(comp))
                except Exception:
                    comparators.append("complex_expression")

            # Determine if this looks like a scoring algorithm
            is_scoring = self._is_scoring_calculation(left_side, comparison_code)

            if not is_scoring:
                return None

            return {
                "expression": comparison_code[:100],  # Truncate for token efficiency
                "left_side": left_side[:50],
                "operators": ops,
                "comparators": comparators,
                "line": line,
                "scoring_type": self._classify_scoring_type(left_side, comparison_code),
                "file": self.file_path,
            }

        except Exception:
            return None

    def _analyze_condition_complexity(self, node: ast.AST) -> int:
        """Analyze the complexity of a conditional statement."""
        complexity = 0

        for child in ast.walk(node):
            if (
                isinstance(child, (ast.BoolOp, ast.Compare))
                or isinstance(child, ast.Call)
                or isinstance(child, (ast.BinOp, ast.UnaryOp))
            ):
                complexity += 1

        return complexity

    def _extract_thresholds_from_condition(self, node: ast.AST) -> list[dict[str, Any]]:
        """Extract threshold values from conditional expressions."""
        thresholds = []

        for child in ast.walk(node):
            if isinstance(child, ast.Compare):
                for op, comp in zip(child.ops, child.comparators, strict=False):
                    if isinstance(comp, ast.Constant) and isinstance(
                        comp.value,
                        (int, float),
                    ):
                        threshold_info = {
                            "value": comp.value,
                            "operator": type(op).__name__,
                            "context": ast.unparse(child)[:50],
                        }
                        thresholds.append(threshold_info)

        return thresholds

    def _extract_branch_actions(self, body: list[ast.stmt]) -> list[str]:
        """Extract high-level actions from if/else branch bodies."""
        actions = []

        for stmt in body[:3]:  # Limit to first 3 statements
            if isinstance(stmt, ast.Assign):
                actions.append(
                    f"assign_to_{stmt.targets[0].id if hasattr(stmt.targets[0], 'id') else 'variable'}",
                )
            elif isinstance(stmt, ast.Return):
                actions.append("return_value")
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                func_name = getattr(stmt.value.func, "id", "function")
                actions.append(f"call_{func_name}")
            elif isinstance(stmt, ast.Raise):
                actions.append("raise_exception")

        return actions

    def _classify_business_logic(
        self,
        condition: str,
        if_actions: list[str],
        else_actions: list[str],
    ) -> str:
        """Classify the type of business logic pattern."""
        condition_lower = condition.lower()

        if "threshold" in condition_lower or any(
            ">" in condition or "<" in condition for condition in [condition]
        ):
            return "threshold_check"
        if "status" in condition_lower or "state" in condition_lower:
            return "status_validation"
        if "permission" in condition_lower or "auth" in condition_lower:
            return "access_control"
        if "valid" in condition_lower or "error" in condition_lower:
            return "validation_rule"
        if len(if_actions) > 0 and "return" in str(if_actions):
            return "decision_gate"
        return "conditional_logic"

    def _is_scoring_calculation(self, left_side: str, full_expression: str) -> bool:
        """Determine if this looks like a scoring or calculation algorithm."""
        scoring_indicators = [
            "*",
            "+",
            "-",
            "/",
            "**",  # Mathematical operations
            "score",
            "rating",
            "weight",
            "factor",
            "coefficient",
            "calculate",
            "compute",
            "sum",
            "total",
            "average",
            "max",
            "min",
            "abs",
            "round",
        ]

        combined = (left_side + " " + full_expression).lower()
        return any(indicator in combined for indicator in scoring_indicators)

    def _classify_scoring_type(self, left_side: str, expression: str) -> str:
        """Classify the type of scoring algorithm."""
        combined = (left_side + " " + expression).lower()

        if "weight" in combined or "*" in expression:
            return "weighted_calculation"
        if "sum" in combined or "total" in combined:
            return "aggregation"
        if "average" in combined or "mean" in combined:
            return "averaging"
        if "max" in combined or "min" in combined:
            return "optimization"
        if "threshold" in combined or ">" in expression or "<" in expression:
            return "threshold_scoring"
        return "calculation"


def extract_advanced_semantic_information(
    file_contents: dict[str, str],
) -> dict[str, Any]:
    """Extract advanced semantic information: enums, constants, magic numbers, lookup tables.

    Args:
        file_contents: Dictionary mapping file paths to their content

    Returns:
        Dictionary with extracted semantic information

    """
    semantic_info = {
        "constants": [],
        "enums": [],
        "magic_numbers": [],
        "lookup_tables": [],
        "functions": [],
        "data_transformations": [],
        "business_logic": [],
        "scoring_algorithms": [],
    }

    for file_path, content in file_contents.items():
        # Only process Python files
        file_path_str = str(file_path)
        if not file_path_str.endswith(".py"):
            continue

        try:
            tree = ast.parse(content)
            lines = content.split("\n")

            extractor = AdvancedSemanticExtractor(file_path_str, lines)
            extractor.visit(tree)

            # Collect results
            semantic_info["constants"].extend(extractor.constants)
            semantic_info["enums"].extend(extractor.enums)
            semantic_info["magic_numbers"].extend(extractor.magic_numbers)
            semantic_info["lookup_tables"].extend(extractor.lookup_tables)
            semantic_info["functions"].extend(extractor.functions)
            semantic_info["data_transformations"].extend(extractor.data_transformations)
            semantic_info["business_logic"].extend(extractor.business_logic)
            semantic_info["scoring_algorithms"].extend(extractor.scoring_algorithms)

        except SyntaxError:
            # Skip files with syntax errors
            continue

    return semantic_info


def format_advanced_semantic_for_llm(semantic_info: dict[str, Any]) -> str:
    """Format extracted advanced semantic information for LLM consumption.

    Args:
        semantic_info: Dictionary of extracted semantic information

    Returns:
        Formatted string suitable for LLM prompts

    """
    if not any(semantic_info.values()):
        return ""

    sections = ["\n=== ADVANCED SEMANTIC ANALYSIS ==="]

    # Use helper functions to reduce complexity
    sections.extend(_format_constants(semantic_info["constants"]))
    sections.extend(_format_enhanced_enums(semantic_info["enums"]))
    sections.extend(_format_magic_numbers(semantic_info["magic_numbers"]))
    sections.extend(_format_lookup_tables(semantic_info["lookup_tables"]))
    sections.extend(_format_functions(semantic_info["functions"]))
    sections.extend(_format_data_transformations(semantic_info["data_transformations"]))
    sections.extend(_format_business_logic(semantic_info["business_logic"]))
    sections.extend(_format_scoring_algorithms(semantic_info["scoring_algorithms"]))

    sections.append("\n=== END ADVANCED SEMANTIC ANALYSIS ===")
    return "\n".join(sections)


def _format_constants(constants: list[dict[str, Any]]) -> list[str]:
    """Format constants for LLM output."""
    sections = []
    if constants:
        sections.append("\n--- Configuration Constants & Settings ---")

        # Group by type
        by_type = {}
        for const in constants:
            const_type = const.get("type", "general_constant")
            if const_type not in by_type:
                by_type[const_type] = []
            by_type[const_type].append(const)

        for const_type, const_list in by_type.items():
            type_name = const_type.replace("_", " ").title()
            sections.append(f"{type_name}:")
            for const in const_list[:5]:  # Limit to 5 per type
                sections.append(f"  • {const['name']} = {const['value']}")
                if const.get("semantic_meaning"):
                    sections.append(f"    Purpose: {const['semantic_meaning']}")
                if const.get("comment"):
                    sections.append(f"    Note: {const['comment']}")

    return sections


def _format_enhanced_enums(enums: list[dict[str, Any]]) -> list[str]:
    """Format enhanced enum definitions for LLM output."""
    sections = []
    if enums:
        sections.append("\n--- Enumerated Types & Categories ---")
        for enum in enums[:5]:  # Limit to 5 enums
            sections.append(f"• {enum['name']} at line {enum['line']}")
            if enum.get("domain"):
                sections.append(f"  Domain: {enum['domain']}")

            if enum["values"]:
                sections.append(f"  Values: {len(enum['values'])} defined")
                for value in enum["values"][:4]:  # Show first 4 values
                    value_str = f"    - {value['name']} = {value['value']}"
                    if value.get("semantic_meaning"):
                        value_str += f" # {value['semantic_meaning']}"
                    sections.append(value_str)
                if len(enum["values"]) > 4:
                    sections.append(
                        f"    ... and {len(enum['values']) - 4} more values",
                    )

    return sections


def _format_magic_numbers(magic_numbers: list[dict[str, Any]]) -> list[str]:
    """Format magic numbers for LLM output."""
    sections = []
    if magic_numbers:
        sections.append("\n--- Magic Numbers & Literals ---")
        # Sort by value to group similar numbers
        sorted_numbers = sorted(magic_numbers, key=lambda x: x["value"])

        for magic in sorted_numbers[:8]:  # Limit to 8 magic numbers
            sections.append(f"• {magic['value']} at line {magic['line']}")
            if magic.get("semantic_guess"):
                sections.append(f"  Likely meaning: {magic['semantic_guess']}")
            if magic.get("context"):
                sections.append(f"  Context: {magic['context'][:60]}...")

    return sections


def _format_lookup_tables(lookup_tables: list[dict[str, Any]]) -> list[str]:
    """Format lookup tables for LLM output."""
    sections = []
    if lookup_tables:
        sections.append("\n--- Lookup Tables & Mappings ---")
        for table in lookup_tables[:4]:  # Limit to 4 tables
            sections.append(f"• {table['name']} at line {table['line']}")
            if table.get("purpose"):
                sections.append(f"  Purpose: {table['purpose']}")
            sections.append(f"  Entries: {table.get('total_entries', 0)}")

            # Show a few example mappings
            if table.get("mappings"):
                sections.append("  Sample mappings:")
                for mapping in table["mappings"][:3]:
                    sections.append(f"    {mapping['key']} → {mapping['value']}")

    return sections


def _format_pydantic_models(models: list[dict[str, Any]]) -> list[str]:
    """Format Pydantic model information for LLM output."""
    sections = []
    if models:
        sections.append("\n--- Pydantic Model Definitions ---")
        for model in models[:5]:  # Limit to top 5 models
            sections.append(f"• {model['name']} at line {model['line']}")
            if model["docstring"]:
                first_line = model["docstring"].split("\n")[0].strip()
                sections.append(f"  Purpose: {first_line}")

            if model["base_classes"]:
                bases = [b for b in model["base_classes"] if b != "BaseModel"]
                if bases:
                    sections.append(f"  Inherits from: {', '.join(bases)}")

            if model["fields"]:
                sections.append(f"  Fields: {len(model['fields'])} defined")
                for field in model["fields"][:5]:  # Show first 5 fields
                    field_str = f"    - {field['name']}: {field['type_hint'] or 'Any'}"
                    if field["description"]:
                        field_str += f" # {field['description'][:50]}..."
                    sections.append(field_str)
    return sections


def _format_field_descriptions(field_descriptions: list[dict[str, Any]]) -> list[str]:
    """Format field descriptions for LLM output."""
    sections = []
    if field_descriptions:
        sections.append("\n--- Field Descriptions & Documentation ---")
        for field_desc in field_descriptions[:10]:  # Limit to 10 descriptions
            sections.append(
                f"• {field_desc['model']}.{field_desc['field']}: {field_desc['type']}",
            )
            sections.append(f"  Description: {field_desc['description']}")
    return sections


def _format_validation_rules(validation_rules: list[dict[str, Any]]) -> list[str]:
    """Format validation rules for LLM output."""
    sections = []
    if validation_rules:
        sections.append("\n--- Validation Rules & Constraints ---")

        # Group by type
        field_constraints = [
            v for v in validation_rules if v.get("type") == "field_constraint"
        ]
        validator_functions = [
            v for v in validation_rules if v.get("type") == "validator_function"
        ]

        # Show field constraints
        if field_constraints:
            sections.append("Field Constraints:")
            for validation in field_constraints[:8]:  # Limit to 8 rules
                sections.append(f"  • {validation['model']}.{validation['field']}")
                sections.append(f"    Rules: {', '.join(validation['rules'])}")

        # Show validator functions
        if validator_functions:
            sections.append("Validator Functions:")
            for validation in validator_functions[:6]:  # Limit to 6 validators
                field_list = (
                    ", ".join(validation["field_names"])
                    if validation["field_names"]
                    else "model-level"
                )
                sections.append(
                    f"  • {validation['model']}.{validation['validator_name']} (line {validation['line']})",
                )
                sections.append(f"    Validates: {field_list}")
                if validation.get("docstring"):
                    first_line = validation["docstring"].split("\n")[0].strip()
                    sections.append(f"    Purpose: {first_line}")

    return sections


def _format_enums(enums: list[dict[str, Any]]) -> list[str]:
    """Format enum definitions for LLM output."""
    sections = []
    if enums:
        sections.append("\n--- Enum Definitions ---")
        for enum in enums[:5]:  # Limit to 5 enums
            sections.append(f"• {enum['name']} at line {enum['line']}")
            if enum["docstring"]:
                first_line = enum["docstring"].split("\n")[0].strip()
                sections.append(f"  Purpose: {first_line}")

            if enum["values"]:
                sections.append(f"  Values: {len(enum['values'])} defined")
                for value in enum["values"][:3]:  # Show first 3 values
                    value_str = f"    - {value['name']} = {value['value']}"
                    if value["comment"]:
                        value_str += f"  # {value['comment']}"
                    sections.append(value_str)
    return sections


def _format_model_relationships(relationships: list[dict[str, Any]]) -> list[str]:
    """Format model relationships for LLM output."""
    sections = []
    if relationships:
        sections.append("\n--- Model Relationships ---")

        # Group by relationship type
        inheritance_rels = [r for r in relationships if r["type"] == "inheritance"]
        composition_rels = [r for r in relationships if r["type"] == "composition"]

        if inheritance_rels:
            sections.append("Inheritance:")
            for rel in inheritance_rels[:8]:  # Limit to 8 relationships
                sections.append(f"  • {rel['source']} inherits from {rel['target']}")

        if composition_rels:
            sections.append("Composition:")
            for rel in composition_rels[:8]:  # Limit to 8 relationships
                field_info = f" (field: {rel['field']})" if rel.get("field") else ""
                sections.append(
                    f"  • {rel['source']} contains {rel['target']}{field_info}",
                )

    return sections


def format_pydantic_schema_for_llm(schema_info: dict[str, Any]) -> str:
    """Format extracted Pydantic schema information for LLM consumption.

    Args:
        schema_info: Dictionary of extracted Pydantic schema information

    Returns:
        Formatted string suitable for LLM prompts

    """
    if not any(schema_info.values()):
        return ""

    sections = ["\n=== PYDANTIC SCHEMA ANALYSIS ==="]

    # Use helper functions to reduce complexity
    sections.extend(_format_pydantic_models(schema_info["models"]))
    sections.extend(
        _format_model_relationships(schema_info.get("model_relationships", [])),
    )
    sections.extend(_format_field_descriptions(schema_info["field_descriptions"]))
    sections.extend(_format_validation_rules(schema_info["validation_rules"]))
    sections.extend(_format_enums(schema_info["enums"]))

    sections.append("\n=== END PYDANTIC SCHEMA ANALYSIS ===")
    return "\n".join(sections)


def _format_functions(functions: list[dict[str, Any]]) -> list[str]:
    """Format function analysis for LLM output."""
    sections = []
    if functions:
        sections.append("\n--- Function Analysis & Data Flow ---")

        # Group by data transformation status
        data_transforms = [f for f in functions if f.get("is_data_transform")]
        regular_functions = [f for f in functions if not f.get("is_data_transform")]

        if data_transforms:
            sections.append("Data Transformation Functions:")
            for func in data_transforms[:6]:  # Limit to 6 functions
                sections.append(f"  • {func['name']}() at line {func['line']}")
                if func.get("creates"):
                    sections.append(f"    Creates: {', '.join(func['creates'][:2])}")
                if func.get("returns"):
                    sections.append(f"    Returns: {', '.join(func['returns'][:2])}")
                if func.get("data_transformations"):
                    sections.append(
                        f"    Operations: {', '.join(func['data_transformations'][:3])}",
                    )

        if regular_functions:
            sections.append("Other Key Functions:")
            for func in regular_functions[:4]:  # Limit to 4 functions
                sections.append(f"  • {func['name']}() at line {func['line']}")
                if func.get("docstring"):
                    sections.append(f"    Purpose: {func['docstring'][:60]}...")

    return sections


def _format_data_transformations(transformations: list[dict[str, Any]]) -> list[str]:
    """Format data transformation pipeline information for LLM output."""
    sections = []
    if transformations:
        sections.append("\n--- Data Transformation Pipelines ---")

        # Group by transformation type
        by_type = {}
        for transform in transformations:
            trans_type = transform.get("type", "processor")
            if trans_type not in by_type:
                by_type[trans_type] = []
            by_type[trans_type].append(transform)

        for trans_type, trans_list in by_type.items():
            type_name = trans_type.replace("_", " ").title()
            sections.append(f"{type_name} Functions:")
            for transform in trans_list[:4]:  # Limit to 4 per type
                sections.append(
                    f"  • {transform['function']}() at line {transform['line']}",
                )
                if transform.get("input_data"):
                    sections.append(
                        f"    Input: {', '.join(transform['input_data'][:3])}",
                    )
                if transform.get("output_data"):
                    sections.append(
                        f"    Output: {', '.join(transform['output_data'][:2])}",
                    )

    return sections


def _format_business_logic(business_logic: list[dict[str, Any]]) -> list[str]:
    """Format business logic patterns for LLM output."""
    sections = []
    if business_logic:
        sections.append("\n--- Business Logic & Conditional Patterns ---")

        # Group by logic type
        by_type = {}
        for logic in business_logic:
            logic_type = logic.get("logic_type", "conditional_logic")
            if logic_type not in by_type:
                by_type[logic_type] = []
            by_type[logic_type].append(logic)

        for logic_type, logic_list in by_type.items():
            type_name = logic_type.replace("_", " ").title()
            sections.append(f"{type_name}:")
            for logic in logic_list[:4]:  # Limit to 4 per type
                sections.append(f"  • Line {logic['line']}: {logic['condition']}")
                if logic.get("thresholds"):
                    threshold_info = ", ".join(
                        [
                            f"{t['value']} ({t['operator']})"
                            for t in logic["thresholds"][:2]
                        ],
                    )
                    sections.append(f"    Thresholds: {threshold_info}")
                if logic.get("complexity", 0) > 3:
                    sections.append(f"    Complexity: {logic['complexity']} conditions")

    return sections


def _format_scoring_algorithms(scoring_algorithms: list[dict[str, Any]]) -> list[str]:
    """Format scoring and calculation algorithms for LLM output."""
    sections = []
    if scoring_algorithms:
        sections.append("\n--- Scoring & Calculation Algorithms ---")

        # Group by scoring type
        by_type = {}
        for scoring in scoring_algorithms:
            scoring_type = scoring.get("scoring_type", "calculation")
            if scoring_type not in by_type:
                by_type[scoring_type] = []
            by_type[scoring_type].append(scoring)

        for scoring_type, scoring_list in by_type.items():
            type_name = scoring_type.replace("_", " ").title()
            sections.append(f"{type_name}:")
            for scoring in scoring_list[:4]:  # Limit to 4 per type
                sections.append(f"  • Line {scoring['line']}: {scoring['expression']}")
                if scoring.get("left_side"):
                    sections.append(f"    Calculation: {scoring['left_side']}")
                if scoring.get("operators"):
                    op_info = ", ".join(scoring["operators"])
                    sections.append(f"    Operations: {op_info}")

    return sections


class SemanticExtractor(ast.NodeVisitor):
    """AST visitor for extracting semantic information from Python code."""

    def __init__(self, file_path: str, lines: list[str]) -> None:
        """Initialize the semantic extractor.

        Args:
            file_path: Path to the file being analyzed
            lines: List of lines from the file for line number mapping

        """
        self.file_path = file_path
        self.lines = lines
        self.functions: list[dict[str, Any]] = []
        self.classes: list[dict[str, Any]] = []
        self.assignments: list[dict[str, Any]] = []
        self.decorators: list[dict[str, Any]] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Extract function definitions and docstrings."""
        docstring = ast.get_docstring(node)

        # Extract parameters with type hints
        params = []
        for arg in node.args.args:
            param_info = {
                "name": arg.arg,
                "type_hint": None,
            }
            if arg.annotation:
                param_info["type_hint"] = ast.unparse(arg.annotation)
            params.append(param_info)

        # Extract return type hint
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        function_info = {
            "name": node.name,
            "line": node.lineno,
            "docstring": docstring,
            "parameters": params,
            "return_type": return_type,
            "decorators": [ast.unparse(dec) for dec in node.decorator_list],
            "file": self.file_path,
        }

        self.functions.append(function_info)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Extract class definitions and their purposes."""
        docstring = ast.get_docstring(node)

        # Extract base classes
        bases = [ast.unparse(base) for base in node.bases]

        # Extract class fields (assignments within class body)
        class_fields = []
        for child in node.body:
            if isinstance(child, ast.AnnAssign) and isinstance(child.target, ast.Name):
                field_info = {
                    "name": child.target.id,
                    "type_hint": ast.unparse(child.annotation)
                    if child.annotation
                    else None,
                    "default_value": ast.unparse(child.value) if child.value else None,
                    "line": child.lineno,
                }
                class_fields.append(field_info)

        class_info = {
            "name": node.name,
            "line": node.lineno,
            "docstring": docstring,
            "base_classes": bases,
            "fields": class_fields,
            "decorators": [ast.unparse(dec) for dec in node.decorator_list],
            "file": self.file_path,
        }

        self.classes.append(class_info)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        """Extract variable assignments with type hints."""
        # Handle simple variable assignments
        for target in node.targets:
            if isinstance(target, ast.Name):
                assignment_info = {
                    "variable": target.id,
                    "value": ast.unparse(node.value),
                    "line": node.lineno,
                    "file": self.file_path,
                    "nearby_comment": self._get_nearby_comment(node.lineno),
                }
                self.assignments.append(assignment_info)

            # Handle DataFrame column assignments like df["col"] = value
            elif isinstance(target, ast.Subscript):
                if isinstance(target.value, ast.Name) and isinstance(
                    target.slice,
                    ast.Constant,
                ):
                    assignment_info = {
                        "variable": f"{target.value.id}[{target.slice.value!r}]",
                        "value": ast.unparse(node.value),
                        "line": node.lineno,
                        "file": self.file_path,
                        "nearby_comment": self._get_nearby_comment(node.lineno),
                        "is_column_assignment": True,
                        "dataframe": target.value.id,
                        "column_name": target.slice.value,
                    }
                    self.assignments.append(assignment_info)

        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Extract annotated assignments (variable: type = value)."""
        if isinstance(node.target, ast.Name):
            assignment_info = {
                "variable": node.target.id,
                "type_hint": ast.unparse(node.annotation),
                "value": ast.unparse(node.value) if node.value else None,
                "line": node.lineno,
                "file": self.file_path,
                "nearby_comment": self._get_nearby_comment(node.lineno),
            }
            self.assignments.append(assignment_info)

        self.generic_visit(node)

    def _get_nearby_comment(self, line_num: int) -> str | None:
        """Get comment from the same line or nearby lines."""
        # Check same line
        if line_num <= len(self.lines):
            line = self.lines[line_num - 1]
            if "#" in line:
                comment_part = line.split("#", 1)[1].strip()
                if comment_part:
                    return comment_part

        # Check line after
        if line_num < len(self.lines):
            next_line = self.lines[line_num].strip()
            if next_line.startswith("#"):
                return next_line[1:].strip()

        return None


def extract_ast_information(file_contents: dict[str, str]) -> dict[str, Any]:
    """Extract semantic information using AST parsing.

    Args:
        file_contents: Dictionary mapping file paths to their content

    Returns:
        Dictionary with extracted AST information categorized by type

    """
    ast_info = {
        "functions": [],
        "classes": [],
        "assignments": [],
        "column_assignments": [],
    }

    for file_path, content in file_contents.items():
        # Only process Python files
        if not file_path.endswith(".py"):
            continue

        try:
            tree = ast.parse(content)
            lines = content.split("\n")

            extractor = SemanticExtractor(file_path, lines)
            extractor.visit(tree)

            # Collect results
            ast_info["functions"].extend(extractor.functions)
            ast_info["classes"].extend(extractor.classes)

            # Separate regular assignments from column assignments
            for assignment in extractor.assignments:
                if assignment.get("is_column_assignment"):
                    ast_info["column_assignments"].append(assignment)
                else:
                    ast_info["assignments"].append(assignment)

        except SyntaxError:
            # Skip files with syntax errors
            continue

    return ast_info


def _format_functions_for_llm(functions: list[dict[str, Any]]) -> list[str]:
    """Format function information for LLM output."""
    sections = []
    if functions:
        sections.append("\n--- Function Definitions ---")
        for func in functions[:10]:  # Limit to top 10 functions
            sections.append(f"• {func['name']}() at line {func['line']}")
            if func["docstring"]:
                # Show first line of docstring
                first_line = func["docstring"].split("\n")[0].strip()
                sections.append(f"  Purpose: {first_line}")
            if func["parameters"]:
                param_strs = []
                for param in func["parameters"][:5]:  # Limit parameters shown
                    param_str = param["name"]
                    if param["type_hint"]:
                        param_str += f": {param['type_hint']}"
                    param_strs.append(param_str)
                sections.append(f"  Parameters: {', '.join(param_strs)}")
            if func["return_type"]:
                sections.append(f"  Returns: {func['return_type']}")
    return sections


def _format_classes_for_llm(classes: list[dict[str, Any]]) -> list[str]:
    """Format class information for LLM output."""
    sections = []
    if classes:
        sections.append("\n--- Class Definitions ---")
        for cls in classes[:10]:  # Limit to top 10 classes
            sections.append(f"• {cls['name']} at line {cls['line']}")
            if cls["docstring"]:
                first_line = cls["docstring"].split("\n")[0].strip()
                sections.append(f"  Purpose: {first_line}")
            if cls["base_classes"]:
                sections.append(f"  Inherits from: {', '.join(cls['base_classes'])}")
            if cls["fields"]:
                sections.append(f"  Fields: {len(cls['fields'])} defined")
                for field in cls["fields"][:3]:  # Show first 3 fields
                    field_str = field["name"]
                    if field["type_hint"]:
                        field_str += f": {field['type_hint']}"
                    sections.append(f"    - {field_str}")
    return sections


def _format_assignments_for_llm(ast_info: dict[str, Any]) -> list[str]:
    """Format assignment information for LLM output."""
    sections = []

    # Format column assignments (most relevant for data analysis)
    if ast_info["column_assignments"]:
        sections.append("\n--- Column Creation & Assignment ---")
        for assignment in ast_info["column_assignments"][
            :15
        ]:  # Show more column assignments
            sections.append(
                f"• {assignment['dataframe']}['{assignment['column_name']}'] = {assignment['value']}",
            )
            sections.append(f"  Line: {assignment['line']} in {assignment['file']}")
            if assignment["nearby_comment"]:
                sections.append(f"  Comment: {assignment['nearby_comment']}")

    # Format other assignments (constants, variables)
    if ast_info["assignments"]:
        sections.append("\n--- Variable Assignments ---")
        # Filter for likely constants and important variables
        important_assignments = [
            a
            for a in ast_info["assignments"]
            if (
                a["variable"].isupper()  # Constants
                or a.get("type_hint")  # Type-annotated variables
                or a.get("nearby_comment")
            )  # Variables with comments
        ]

        for assignment in important_assignments[:10]:
            var_str = assignment["variable"]
            if assignment.get("type_hint"):
                var_str += f": {assignment['type_hint']}"
            sections.append(f"• {var_str} = {assignment['value']}")
            if assignment["nearby_comment"]:
                sections.append(f"  Comment: {assignment['nearby_comment']}")

    return sections


def format_ast_information_for_llm(ast_info: dict[str, Any]) -> str:
    """Format extracted AST information for LLM consumption.

    Args:
        ast_info: Dictionary of extracted AST information

    Returns:
        Formatted string suitable for LLM prompts

    """
    if not any(ast_info.values()):
        return ""

    sections = ["\n=== AST-BASED CODE ANALYSIS ==="]

    # Use helper functions to reduce complexity
    sections.extend(_format_functions_for_llm(ast_info["functions"]))
    sections.extend(_format_classes_for_llm(ast_info["classes"]))
    sections.extend(_format_assignments_for_llm(ast_info))

    sections.append("\n=== END AST ANALYSIS ===")
    return "\n".join(sections)


def _link_business_logic_to_code(
    comment_data: dict[str, list[dict[str, Any]]],
    line_to_functions: dict[int, dict[str, Any]],
    line_to_columns: dict[int, dict[str, Any]],
    proximity_lines: int,
) -> list[dict[str, Any]]:
    """Link business logic comments to nearby code elements."""
    business_rules = []

    for business_comment in comment_data.get("business_logic", []):
        comment_line = business_comment["line"]
        linked_elements = []

        # Find nearby functions
        for func_line, func in line_to_functions.items():
            if abs(comment_line - func_line) <= proximity_lines:
                linked_elements.append(("function", func))

        # Find nearby column assignments
        for col_line, col in line_to_columns.items():
            if abs(comment_line - col_line) <= proximity_lines:
                linked_elements.append(("column", col))

        if linked_elements:
            business_rules.append(
                {
                    "comment": business_comment,
                    "linked_elements": linked_elements,
                },
            )

    return business_rules


def _link_functions_to_columns(
    ast_info: dict[str, Any],
    proximity_lines: int,
) -> list[dict[str, Any]]:
    """Link function docstrings to column creation."""
    function_contexts = []

    for func in ast_info["functions"]:
        func_line = func["line"]

        # Find columns created within reasonable distance of function
        related_columns = [
            col
            for col in ast_info["column_assignments"]
            if abs(col["line"] - func_line) <= proximity_lines * 3
        ]

        if related_columns or func["docstring"]:
            function_contexts.append(
                {
                    "function": func,
                    "related_columns": related_columns,
                    "creates_columns": len(related_columns) > 0,
                },
            )

    return function_contexts


def _link_constants_to_columns(
    comment_data: dict[str, list[dict[str, Any]]],
    ast_info: dict[str, Any],
    proximity_lines: int,
) -> list[dict[str, Any]]:
    """Link constants to nearby column assignments."""
    column_contexts = []

    for constant in comment_data.get("constants", []):
        constant_line = constant["line"]

        nearby_columns = [
            col
            for col in ast_info["column_assignments"]
            if abs(col["line"] - constant_line) <= proximity_lines * 2
        ]

        if nearby_columns:
            column_contexts.append(
                {
                    "constant": constant,
                    "related_columns": nearby_columns,
                },
            )

    return column_contexts


def link_comments_to_code(
    comment_data: dict[str, list[dict[str, Any]]],
    ast_info: dict[str, Any],
    proximity_lines: int = 3,
) -> dict[str, Any]:
    """Link comments to nearby code elements within N lines.

    Args:
        comment_data: Dictionary of extracted comment patterns
        ast_info: Dictionary of extracted AST information
        proximity_lines: Number of lines to consider for proximity linking

    Returns:
        Dictionary with linked context information

    """
    # Create line-to-element mappings for quick lookup
    line_to_functions = {func["line"]: func for func in ast_info["functions"]}
    line_to_columns = {col["line"]: col for col in ast_info["column_assignments"]}

    return {
        "function_contexts": _link_functions_to_columns(ast_info, proximity_lines),
        "class_contexts": [],  # Placeholder for future implementation
        "column_contexts": _link_constants_to_columns(
            comment_data,
            ast_info,
            proximity_lines,
        ),
        "business_rules": _link_business_logic_to_code(
            comment_data,
            line_to_functions,
            line_to_columns,
            proximity_lines,
        ),
    }


def _format_business_rules(business_rules: list[dict[str, Any]]) -> list[str]:
    """Format business rules linked to code."""
    sections = []
    if business_rules:
        sections.append("\n--- Business Rules → Code Connections ---")
        for rule_context in business_rules:
            comment = rule_context["comment"]
            sections.append(f"• {comment['type'].title()}: {comment['content']}")
            sections.append(f"  Source: {comment['file']}:{comment['line']}")

            for element_type, element in rule_context["linked_elements"]:
                if element_type == "function":
                    sections.append(
                        f"  → Affects function: {element['name']}() at line {element['line']}",
                    )
                elif element_type == "column":
                    sections.append(
                        f"  → Affects column: {element['dataframe']}['{element['column_name']}'] at line {element['line']}",
                    )
    return sections


def _format_function_contexts(function_contexts: list[dict[str, Any]]) -> list[str]:
    """Format function-column relationships."""
    sections = []
    if function_contexts:
        sections.append("\n--- Function → Column Creation ---")
        for func_context in function_contexts:
            func = func_context["function"]
            if func_context["creates_columns"]:
                sections.append(
                    f"• {func['name']}() creates {len(func_context['related_columns'])} columns",
                )
                if func["docstring"]:
                    first_line = func["docstring"].split("\n")[0].strip()
                    sections.append(f"  Purpose: {first_line}")

                # Show first 3 columns
                column_descriptions = [
                    f"  → Creates: {col['dataframe']}['{col['column_name']}']"
                    for col in func_context["related_columns"][:3]
                ]
                sections.extend(column_descriptions)
    return sections


def _format_column_contexts(column_contexts: list[dict[str, Any]]) -> list[str]:
    """Format constant-column relationships."""
    sections = []
    if column_contexts:
        sections.append("\n--- Constants → Column Usage ---")
        for col_context in column_contexts:
            constant = col_context["constant"]
            sections.append(f"• {constant['name']} = {constant['value']}")
            sections.append(f"  Purpose: {constant['explanation']}")

            # Show first 2 related columns
            column_usages = [
                f"  → Used in: {col['dataframe']}['{col['column_name']}']"
                for col in col_context["related_columns"][:2]
            ]
            sections.extend(column_usages)
    return sections


def format_linked_context_for_llm(linked_context: dict[str, Any]) -> str:
    """Format linked context information for LLM consumption.

    Args:
        linked_context: Dictionary with linked context information

    Returns:
        Formatted string suitable for LLM prompts

    """
    if not any(linked_context.values()):
        return ""

    sections = ["\n=== CONTEXT LINKAGE ANALYSIS ==="]

    # Use helper functions to reduce complexity
    sections.extend(_format_business_rules(linked_context["business_rules"]))
    sections.extend(_format_function_contexts(linked_context["function_contexts"]))
    sections.extend(_format_column_contexts(linked_context["column_contexts"]))

    sections.append("\n=== END CONTEXT LINKAGE ===")
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


def build_semantic_knowledge_graph(files_content: dict[str, str]) -> dict[str, Any]:
    """Build a comprehensive semantic knowledge graph from codebase.

    This integrates Pydantic schema mining, semantic extraction, and cross-reference
    resolution to create a unified knowledge representation.

    Args:
        files_content: Dictionary mapping file paths to their content

    Returns:
        Dictionary containing the complete knowledge graph analysis

    """
    # Initialize knowledge graph
    knowledge_graph = SemanticKnowledgeGraph()

    # Extract Pydantic information using existing function
    pydantic_data = extract_pydantic_schema_information(files_content)

    # Extract semantic information using existing function
    semantic_data = extract_advanced_semantic_information(files_content)

    # Build knowledge graph
    knowledge_graph.add_pydantic_schema_info(pydantic_data)
    knowledge_graph.add_semantic_extraction_info(semantic_data)

    # Resolve aliases and cross-references (Task 5.1 & 5.2)
    knowledge_graph.resolve_column_aliases()
    knowledge_graph.build_cross_references()

    # Resolve conflicts and consolidate knowledge (Task 5.3)
    knowledge_graph.resolve_conflicts_and_consolidate()

    # Generate comprehensive summaries
    summary = knowledge_graph.generate_column_summary()
    cross_ref_summary = knowledge_graph.generate_cross_reference_summary()
    conflict_resolution_summary = knowledge_graph.get_conflict_resolution_summary()

    return {
        "knowledge_graph": knowledge_graph,
        "column_summary": summary,
        "cross_reference_summary": cross_ref_summary,
        "conflict_resolution_summary": conflict_resolution_summary,
        "high_confidence_columns": knowledge_graph.get_high_confidence_columns(),
        "total_columns_discovered": len(knowledge_graph.columns),
        "total_cross_references": len(knowledge_graph.cross_references),
        "pydantic_fields": len(pydantic_data.get("field_descriptions", [])),
        "semantic_functions": len(semantic_data.get("functions", [])),
        "semantic_constants": len(semantic_data.get("constants", [])),
        "business_logic_patterns": len(semantic_data.get("business_logic", [])),
    }
