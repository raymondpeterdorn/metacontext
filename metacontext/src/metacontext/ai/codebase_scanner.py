"""Codebase context scanner for metacontext generation.

Scans the current working directory and codebase to find relevant context
for data files being metacontextualized, including related code, documentation,
and project information.
"""

import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class CodebaseScanner:
    """Scans codebase for relevant context related to a data file."""

    MIN_PART_LEN = 2
    max_docs = 20
    max_config_files = 15

    def __init__(self, cwd: Path | None = None) -> None:
        """Initialize scanner with working directory."""
        self.cwd = Path(cwd) if cwd else Path.cwd()

        # File patterns to look for
        self.code_extensions = {
            ".py",
            ".r",
            ".R",
            ".sql",
            ".ipynb",
            ".js",
            ".ts",
            ".java",
            ".scala",
            ".go",
            ".cpp",
            ".c",
            ".h",
        }

        self.doc_extensions = {".md", ".rst", ".txt", ".pdf", ".docx", ".html"}

        self.config_extensions = {".yaml", ".yml", ".json", ".toml", ".ini", ".cfg"}

    def _collect_file_inventory(self, data_file: Path) -> dict[str, list[Path]]:
        """Single directory traversal to collect all relevant files by type.

        Only includes user-written project files, excludes dependencies and virtual environments.
        """
        inventory = {
            "code_files": [],
            "doc_files": [],
            "config_files": [],
            "data_files": [],
            "all_files": [],
        }

        # Directories to exclude (dependencies, virtual environments, caches)
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".coverage",
            ".tox",
            ".idea",
            ".vscode",
            ".DS_Store",
            "node_modules",
            ".venv",
            "venv",
            "env",
            ".env",
            "site-packages",
            "dist",
            "build",
            "pip-wheel-metadata",
            ".eggs",
            "*.egg-info",
            ".cache",
            ".local",
            ".ipynb_checkpoints",
            ".gitignore",
        }

        logger.info("🔍 DEBUG: Collecting file inventory from: %s", self.cwd)
        logger.info(
            "🔍 DEBUG: Excluding dependency/cache directories: %s", excluded_dirs
        )

        # Single directory traversal - only include user project files
        for file_path in self.cwd.rglob("*"):
            if not file_path.is_file() or file_path == data_file:
                continue

            # Skip files in excluded directories
            if any(excluded_dir in file_path.parts for excluded_dir in excluded_dirs):
                continue

            inventory["all_files"].append(file_path)

            suffix = file_path.suffix.lower()
            if suffix in self.code_extensions:
                inventory["code_files"].append(file_path)
                logger.info(
                    "🔍 DEBUG: Found code file: %s", file_path.relative_to(self.cwd)
                )
            elif suffix in self.doc_extensions:
                inventory["doc_files"].append(file_path)
            elif suffix in self.config_extensions:
                inventory["config_files"].append(file_path)
            elif suffix in {
                ".csv",
                ".json",
                ".yaml",
                ".yml",
                ".xlsx",
                ".xls",
                ".parquet",
                ".tsv",
            }:
                inventory["data_files"].append(file_path)

        logger.info("🔍 DEBUG: File inventory summary:")
        logger.info("   Code files: %d", len(inventory["code_files"]))
        logger.info("   Doc files: %d", len(inventory["doc_files"]))
        logger.info("   Config files: %d", len(inventory["config_files"]))
        logger.info("   Data files: %d", len(inventory["data_files"]))
        logger.info("   Total files: %d", len(inventory["all_files"]))

        return inventory

    def _aggregate_file_contents(
        self, file_inventory: dict[str, list[Path]]
    ) -> dict[str, str]:
        """Aggregate file contents into searchable documents."""
        aggregated = {
            "all_code": "",
            "all_docs": "",
            "all_configs": "",
            "file_map": {},  # Maps line ranges to original files
        }

        # Aggregate code files
        code_sections = []
        for file_path in file_inventory["code_files"]:
            try:
                with file_path.open(encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel_path = str(file_path.relative_to(self.cwd))
                section = f"\n# FILE: {rel_path}\n{content}\n# END FILE: {rel_path}\n"
                code_sections.append(section)
            except OSError:
                continue
        aggregated["all_code"] = "\n".join(code_sections)

        # Aggregate documentation
        doc_sections = []
        for file_path in file_inventory["doc_files"]:
            try:
                with file_path.open(encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel_path = str(file_path.relative_to(self.cwd))
                section = f"\n# DOC: {rel_path}\n{content}\n# END DOC: {rel_path}\n"
                doc_sections.append(section)
            except OSError:
                continue
        aggregated["all_docs"] = "\n".join(doc_sections)

        # Aggregate config files
        config_sections = []
        for file_path in file_inventory["config_files"]:
            try:
                with file_path.open(encoding="utf-8", errors="ignore") as f:
                    content = f.read()
                rel_path = str(file_path.relative_to(self.cwd))
                section = (
                    f"\n# CONFIG: {rel_path}\n{content}\n# END CONFIG: {rel_path}\n"
                )
                config_sections.append(section)
            except OSError:
                continue
        aggregated["all_configs"] = "\n".join(config_sections)

        return aggregated

    def _find_cross_references_from_aggregate(
        self,
        data_file: Path,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> dict[str, Any]:
        """Fast cross-reference analysis using aggregated content."""
        cross_refs = {
            "referenced_by": [],
            "imports_from": [],
            "data_dependencies": [],
            "summary": "",
        }

        data_filename = data_file.name
        data_stem = data_file.stem
        data_relative_path = (
            str(data_file.relative_to(self.cwd))
            if data_file.is_relative_to(self.cwd)
            else str(data_file)
        )

        # Fast search in aggregated code content
        all_code = aggregated_content["all_code"]

        # Build patterns for the specific file
        patterns = [
            (rf"""['"`]{re.escape(data_filename)}['"`]""", "direct_filename"),
            (rf"""['"`]{re.escape(data_stem)}['"`]""", "stem_reference"),
            (rf"""['"`]{re.escape(data_relative_path)}['"`]""", "path_reference"),
            (
                rf"""(?:read_csv|load|open|Path)\s*\(\s*['"`][^'"`]*{re.escape(data_stem)}[^'"`]*['"`]""",
                "data_loading",
            ),
        ]

        # Find which files contain references
        referenced_files = set()
        for pattern, ref_type in patterns:
            matches = re.finditer(pattern, all_code, re.IGNORECASE)
            for match in matches:
                # Find which file this match belongs to by looking for file markers
                file_start = all_code.rfind("# FILE:", 0, match.start())
                if file_start != -1:
                    file_end = all_code.find("\n", file_start)
                    if file_end != -1:
                        file_line = all_code[file_start:file_end]
                        file_path = file_line.replace("# FILE:", "").strip()
                        referenced_files.add((file_path, ref_type))

        # Convert to structured format
        for file_path, ref_type in referenced_files:
            cross_refs["referenced_by"].append(
                {
                    "file": file_path,
                    "file_type": Path(file_path).suffix,
                    "reference_type": ref_type,
                    "reference_count": 1,  # Simplified for performance
                }
            )

        # Generate summary
        ref_count = len(cross_refs["referenced_by"])
        cross_refs["summary"] = (
            f"Referenced by {ref_count} file(s)"
            if ref_count > 0
            else "No cross-references found"
        )

        return cross_refs

    def _find_related_code_from_aggregate(
        self,
        data_file: Path,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> list[dict[str, Any]]:
        """Fast related code analysis using aggregated content."""
        related_files = []
        data_stem = data_file.stem
        data_name_parts = re.split(r"[_\-\.]", data_stem.lower())

        # Search for relevance in aggregated content
        all_code = aggregated_content["all_code"]

        for file_path in file_inventory["code_files"]:
            rel_path = str(file_path.relative_to(self.cwd))

            # Calculate relevance score based on filename similarity
            relevance_score = 0.0
            file_stem = file_path.stem.lower()

            # Check for direct name matches
            for part in data_name_parts:
                if len(part) >= self.MIN_PART_LEN and part in file_stem:
                    relevance_score += 0.3

            # Check if file content appears in aggregated content around our data file references
            if data_stem.lower() in all_code.lower():
                file_marker = f"# FILE: {rel_path}"
                if file_marker in all_code:
                    relevance_score += 0.2

            if relevance_score > 0:
                file_info = {
                    "path": rel_path,
                    "relevance_score": relevance_score,
                    "file_type": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                    "preview": f"Found in aggregated scan (score: {relevance_score:.2f})",
                }
                related_files.append(file_info)

        # Sort by relevance and return top 10
        related_files.sort(key=lambda x: x["relevance_score"], reverse=True)
        return related_files[:10]

    def _analyze_project_info_from_aggregate(
        self,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> dict[str, Any]:
        """Fast project info analysis using aggregated content."""
        return {
            "total_files": len(file_inventory["all_files"]),
            "code_files": len(file_inventory["code_files"]),
            "doc_files": len(file_inventory["doc_files"]),
            "config_files": len(file_inventory["config_files"]),
            "has_readme": any(
                "readme" in f.name.lower() for f in file_inventory["doc_files"]
            ),
            "languages": list(set(f.suffix for f in file_inventory["code_files"])),
            "project_name": self.cwd.name,
        }

    def _find_documentation_from_aggregate(
        self,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> list[dict[str, Any]]:
        """Fast documentation analysis using aggregated content."""
        docs = []
        for file_path in file_inventory["doc_files"][:5]:  # Limit for performance
            docs.append(
                {
                    "path": str(file_path.relative_to(self.cwd)),
                    "file_type": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                }
            )
        return docs

    def _find_data_models_from_aggregate(
        self,
        data_file: Path,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> list[dict[str, Any]]:
        """Fast data models analysis using aggregated content."""
        model_files = []
        all_code = aggregated_content["all_code"]

        # Look for model-related patterns in aggregated content
        model_patterns = [
            "class.*Model",
            "BaseModel",
            "pydantic",
            "schema",
            "dataclass",
        ]

        found_models = set()
        for pattern in model_patterns:
            matches = re.finditer(pattern, all_code, re.IGNORECASE)
            for match in matches:
                # Find which file this match belongs to
                file_start = all_code.rfind("# FILE:", 0, match.start())
                if file_start != -1:
                    file_end = all_code.find("\n", file_start)
                    if file_end != -1:
                        file_line = all_code[file_start:file_end]
                        file_path = file_line.replace("# FILE:", "").strip()
                        found_models.add(file_path)

        for file_path in list(found_models)[:5]:  # Limit for performance
            try:
                full_path = self.cwd / file_path
                model_files.append(
                    {
                        "path": file_path,
                        "file_type": full_path.suffix,
                        "size_bytes": full_path.stat().st_size,
                        "relevance": "Contains model/schema definitions",
                    }
                )
            except OSError:
                continue

        return model_files

    def _find_config_files_from_aggregate(
        self,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> list[dict[str, Any]]:
        """Fast config files analysis using aggregated content."""
        configs = []
        for file_path in file_inventory["config_files"][:10]:  # Limit for performance
            configs.append(
                {
                    "path": str(file_path.relative_to(self.cwd)),
                    "file_type": file_path.suffix,
                    "size_bytes": file_path.stat().st_size,
                }
            )
        return configs

    def _extract_semantic_knowledge_from_aggregate(
        self,
        aggregated_content: dict[str, str],
        file_inventory: dict[str, list[Path]],
    ) -> dict[str, Any]:
        """Fast semantic knowledge extraction using aggregated content."""
        # Use the aggregated code content for semantic analysis
        all_code = aggregated_content["all_code"]

        if not all_code.strip():
            return {
                "semantic_knowledge": None,
                "message": "No code content found for semantic analysis",
            }

        python_files = [f for f in file_inventory["code_files"] if f.suffix == ".py"]

        if not python_files:
            return {
                "semantic_knowledge": None,
                "message": "No Python files found for semantic analysis",
            }

        # Debug logging to see what we're working with
        logger.info(
            "🔍 DEBUG: Found %d Python files for semantic analysis", len(python_files)
        )
        for f in python_files[:5]:  # Show first 5
            logger.info("🔍 DEBUG: Python file: %s", f.relative_to(self.cwd))

        logger.info("🔍 DEBUG: Aggregated code length: %d characters", len(all_code))
        logger.info("🔍 DEBUG: First 500 chars of aggregated code:")
        logger.info("🔍 DEBUG: %s", all_code[:500])

        # Extract semantic patterns from aggregated content
        semantic_patterns = self._extract_semantic_patterns_from_code(all_code)
        logger.info("🔍 DEBUG: Semantic patterns result: %s", semantic_patterns)

        if semantic_patterns:
            return {
                "semantic_knowledge": semantic_patterns,
                "message": f"Extracted from {len(python_files)} Python files via aggregation",
            }
        return {
            "semantic_knowledge": None,
            "message": "No semantic patterns found in aggregated code",
        }

    def _extract_semantic_patterns_from_code(
        self, all_code: str
    ) -> dict[str, Any] | None:
        """Extract semantic patterns from aggregated code content."""
        patterns = {
            "pydantic_fields": {},
            "class_definitions": [],
            "function_definitions": [],
            "variable_assignments": {},
            "comments": [],
        }

        lines = all_code.split("\n")
        current_file = None

        for line_num, line in enumerate(lines):
            # Track which file we're in
            if line.startswith("# FILE:"):
                current_file = line.replace("# FILE:", "").strip()
                continue
            if line.startswith("# END FILE:"):
                current_file = None
                continue

            if not current_file:
                continue

            stripped = line.strip()

            # Extract Pydantic field definitions with comments
            if ":" in stripped and not stripped.startswith("#"):
                # Look for patterns like: field_name: Type = Field(description="Description")
                # or: field_name: Type  # Comment about meaning
                field_match = re.match(
                    r"(\w+)\s*:\s*([^=]+)(?:=.*)?(?:\s*#\s*(.+))?", stripped
                )
                if field_match:
                    field_name, field_type, comment = field_match.groups()

                    # Look for Field() descriptions
                    description = None
                    if "Field(" in stripped:
                        desc_match = re.search(
                            r'description\s*=\s*["\']([^"\']+)["\']', stripped
                        )
                        if desc_match:
                            description = desc_match.group(1)

                    patterns["pydantic_fields"][field_name] = {
                        "type": field_type.strip(),
                        "comment": comment.strip() if comment else None,
                        "description": description,
                        "file": current_file,
                        "line": line_num,
                    }

            # Extract comments that might contain semantic information
            if stripped.startswith("#") and len(stripped) > 3:
                comment_text = stripped[1:].strip()
                if len(comment_text) > 10:  # Meaningful comments
                    patterns["comments"].append(
                        {
                            "text": comment_text,
                            "file": current_file,
                            "line": line_num,
                        }
                    )

        # Extract all semantic patterns found in the code
        found_columns = {}

        # Collect all Pydantic fields with descriptions
        for field_name, field_info in patterns["pydantic_fields"].items():
            if field_info.get("description") or field_info.get("comment"):
                found_columns[field_name] = {
                    "pydantic_definition": field_info,
                    "related_comments": [],
                    "inferred_meaning": field_info.get("description")
                    or field_info.get("comment"),
                }

        # Find comments that might relate to any discovered fields
        for comment in patterns["comments"]:
            comment_text = comment["text"].lower()
            # Look for comments that mention any of our discovered field names
            for field_name in found_columns:
                if field_name.lower() in comment_text:
                    found_columns[field_name]["related_comments"].append(comment)

        if found_columns:
            knowledge_graph = {
                "columns": found_columns,
                "total_fields_found": len(patterns["pydantic_fields"]),
                "total_comments_found": len(patterns["comments"]),
            }

            return {
                "knowledge_graph": knowledge_graph,
                "column_summary": {
                    "total_columns": len(found_columns),
                    "high_confidence_columns": sum(
                        1
                        for col in found_columns.values()
                        if col["pydantic_definition"].get("description")
                    ),
                },
            }

        return None

        return None

        # Special filenames to prioritize
        self.important_files = {
            "readme.md",
            "readme.txt",
            "readme.rst",
            "readme",
            "changelog.md",
            "changelog.txt",
            "changelog",
            "requirements.txt",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "pom.xml",
            "build.gradle",
            "makefile",
            "dockerfile",
            "docker-compose.yml",
        }

        # Documentation directories
        self.doc_directories = {"docs", "doc", "documentation", "wiki", "help"}

    def scan_for_context(self, data_file: Path) -> dict[str, Any]:
        """Scan codebase for context related to the data file using optimized aggregation.

        Args:
            data_file: Path to the data file being metacontextualized

        Returns:
            Dictionary containing discovered context information

        """
        scan_start = time.time()
        timing_log = {}

        # Step 1: Single directory traversal to collect all relevant files
        inventory_start = time.time()
        file_inventory = self._collect_file_inventory(data_file)
        timing_log["file_inventory"] = time.time() - inventory_start

        # Step 2: Aggregate content for fast searching
        aggregation_start = time.time()
        aggregated_content = self._aggregate_file_contents(file_inventory)
        timing_log["content_aggregation"] = time.time() - aggregation_start

        # Step 3: Fast analysis on aggregated content
        project_start = time.time()
        project_info = self._analyze_project_info_from_aggregate(
            aggregated_content, file_inventory
        )
        timing_log["project_info"] = time.time() - project_start

        code_start = time.time()
        related_code = self._find_related_code_from_aggregate(
            data_file, aggregated_content, file_inventory
        )
        timing_log["related_code"] = time.time() - code_start

        docs_start = time.time()
        documentation = self._find_documentation_from_aggregate(
            aggregated_content, file_inventory
        )
        timing_log["documentation"] = time.time() - docs_start

        models_start = time.time()
        data_models = self._find_data_models_from_aggregate(
            data_file, aggregated_content, file_inventory
        )
        timing_log["data_models"] = time.time() - models_start

        config_start = time.time()
        config_files = self._find_config_files_from_aggregate(
            aggregated_content, file_inventory
        )
        timing_log["config_files"] = time.time() - config_start

        refs_start = time.time()
        cross_references = self._find_cross_references_from_aggregate(
            data_file, aggregated_content, file_inventory
        )
        timing_log["cross_references"] = time.time() - refs_start

        semantic_start = time.time()
        semantic_knowledge = self._extract_semantic_knowledge_from_aggregate(
            aggregated_content, file_inventory
        )
        timing_log["semantic_knowledge"] = time.time() - semantic_start

        total_scan_time = time.time() - scan_start
        timing_log["total_scan"] = total_scan_time

        context: dict[str, Any] = {
            "project_info": project_info,
            "related_code": related_code,
            "documentation": documentation,
            "data_models": data_models,
            "config_files": config_files,
            "cross_references": cross_references,
            "semantic_knowledge": semantic_knowledge,
            "scan_summary": {},
        }

        # Add summary statistics
        semantic_summary = context["semantic_knowledge"]
        semantic_count = 0
        if isinstance(semantic_summary, dict) and "summary" in semantic_summary:
            try:
                semantic_count = len(
                    semantic_summary.get("summary", {}).get("column_knowledge", {}),
                )
            except (AttributeError, TypeError):
                semantic_count = 0

        context["scan_summary"] = {
            "total_files_scanned": self._count_files_scanned(),
            "related_code_files": len(context["related_code"]),
            "documentation_files": len(context["documentation"]),
            "data_model_files": len(context["data_models"]),
            "config_files": len(context["config_files"]),
            "cross_reference_files": len(context["cross_references"]["referenced_by"])
            + len(context["cross_references"]["imports_from"]),
            "semantic_columns_found": semantic_count,
            "scan_depth": self._get_scan_depth(),
            "timing": timing_log,  # Add timing info to summary
        }

        # Log timing information for slow operations
        if total_scan_time > 1.0:  # Only log if scan takes more than 1 second
            logger.info("🔍 CODEBASE SCAN TIMING:")
            for operation, duration in timing_log.items():
                if duration > 0.1:  # Only show operations that take > 100ms
                    logger.info("   %s: %.3fs", operation, duration)

        return context

    def _scan_project_info(self) -> dict[str, Any]:
        """Scan for project-level information."""
        project_info: dict[str, Any] = {
            "project_root": str(self.cwd),
            "readme_files": [],
            "changelog_files": [],
            "project_config": [],
        }

        # Look for README files
        for file_path in self.cwd.rglob("*"):
            if file_path.is_file():
                filename_lower = file_path.name.lower()

                if any(readme in filename_lower for readme in ["readme"]):
                    project_info["readme_files"].append(
                        {
                            "path": str(file_path.relative_to(self.cwd)),
                            "size_bytes": file_path.stat().st_size,
                            "preview": self._get_file_preview(file_path),
                        },
                    )

                elif any(
                    changelog in filename_lower
                    for changelog in ["changelog", "changes", "history"]
                ):
                    project_info["changelog_files"].append(
                        {
                            "path": str(file_path.relative_to(self.cwd)),
                            "size_bytes": file_path.stat().st_size,
                            "preview": self._get_file_preview(file_path),
                        },
                    )

                elif filename_lower in [
                    "setup.py",
                    "pyproject.toml",
                    "package.json",
                    "requirements.txt",
                ]:
                    project_info["project_config"].append(
                        {
                            "path": str(file_path.relative_to(self.cwd)),
                            "type": filename_lower,
                            "content": self._get_file_preview(file_path, max_lines=20),
                        },
                    )

        return project_info

    def _find_related_code(self, data_file: Path) -> list[dict[str, Any]]:
        """Find code files that might be related to the data file."""
        related_files = []
        data_stem = data_file.stem  # filename without extension
        data_name_parts = re.split(r"[_\-\.]", data_stem.lower())

        # Search for code files
        for file_path in self.cwd.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.code_extensions
                and file_path != data_file
            ):
                relevance_score = self._calculate_relevance(
                    file_path,
                    data_file,
                    data_name_parts,
                )

                if relevance_score > 0:
                    file_info = {
                        "path": str(file_path.relative_to(self.cwd)),
                        "relevance_score": relevance_score,
                        "file_type": file_path.suffix,
                        "size_bytes": file_path.stat().st_size,
                        "preview": self._get_code_preview(file_path),
                    }

                    # Add specific analysis for Python files
                    if file_path.suffix == ".py":
                        file_info.update(
                            self._analyze_python_file(file_path, data_stem),
                        )

                    related_files.append(file_info)

        # Sort by relevance score
        def get_relevance_score(x: dict[str, Any]) -> float:
            score = x.get("relevance_score", 0.0)
            return float(score) if isinstance(score, (int, float, str)) else 0.0

        related_files.sort(key=get_relevance_score, reverse=True)
        return related_files[:10]  # Limit to top 10 most relevant files

    def _find_data_models(self, data_file: Path) -> list[dict[str, Any]]:
        """Find data models, schemas, or type definitions that might relate to the data."""
        model_files = []
        data_stem = data_file.stem.lower()

        # Look for model-related files
        model_patterns = [
            r".*model.*\.py$",
            r".*schema.*\.py$",
            r".*types?.*\.py$",
            r".*entity.*\.py$",
            r".*dto.*\.py$",
            r".*pydantic.*\.py$",
        ]

        for file_path in self.cwd.rglob("*.py"):
            if file_path.is_file():
                filename_lower = file_path.name.lower()

                # Check if filename matches model patterns
                if any(re.match(pattern, filename_lower) for pattern in model_patterns):
                    content_analysis = self._analyze_model_file(file_path, data_stem)
                    if content_analysis["has_relevant_models"]:
                        model_files.append(
                            {
                                "path": str(file_path.relative_to(self.cwd)),
                                "type": "data_model",
                                "models_found": content_analysis["models"],
                                "preview": self._get_code_preview(file_path),
                            },
                        )

        return model_files

    def _scan_documentation(self) -> list[Document]:
        """Scan for documentation files and directories."""
        docs: list[Document] = []

        # Check for documentation directories
        for dir_name in self.doc_directories:
            doc_dir = self.cwd / dir_name
            if doc_dir.exists() and doc_dir.is_dir():
                docs.extend(
                    Document(
                        page_content=self._get_file_preview(file_path),
                        metadata={
                            "path": str(file_path.relative_to(self.cwd)),
                            "type": "documentation",
                            "directory": dir_name,
                            "size_bytes": file_path.stat().st_size,
                        },
                    )
                    for file_path in doc_dir.rglob("*")
                    if file_path.is_file()
                    and file_path.suffix.lower() in self.doc_extensions
                )

        # Look for scattered documentation files
        docs.extend(
            Document(
                page_content=self._get_file_preview(file_path),
                metadata={
                    "path": str(file_path.relative_to(self.cwd)),
                    "type": "documentation",
                    "directory": "root",
                    "size_bytes": file_path.stat().st_size,
                },
            )
            for file_path in self.cwd.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in self.doc_extensions
            and not any(doc_dir in file_path.parts for doc_dir in self.doc_directories)
        )

        return docs[: self.max_docs]  # Limit to prevent overwhelming output

    def _scan_config_files(self) -> list[dict[str, Any]]:
        """Scan for configuration files that might be relevant."""
        config_files = [
            {
                "path": str(file_path.relative_to(self.cwd)),
                "type": "configuration",
                "size_bytes": file_path.stat().st_size,
                "preview": self._get_file_preview(file_path, max_lines=10),
            }
            for file_path in self.cwd.rglob("*")
            if file_path.is_file()
            and (
                file_path.suffix.lower() in self.config_extensions
                or file_path.name.lower() in self.important_files
            )
        ]

        return config_files[
            : self.max_config_files
        ]  # Limit to prevent overwhelming output

    def _find_cross_references(self, data_file: Path) -> dict[str, Any]:
        """Find cross-file references - which files reference this file and what it imports.

        Args:
            data_file: Path to the data file being analyzed

        Returns:
            Dictionary containing cross-reference information

        """
        cross_refs: dict[str, Any] = {
            "referenced_by": [],  # Files that reference/import/load this data file
            "imports_from": [],  # Files that this data file imports/depends on
            "data_dependencies": [],  # Other data files this file might depend on
            "summary": "",
        }

        data_filename = data_file.name
        data_stem = data_file.stem
        data_relative_path = (
            str(data_file.relative_to(self.cwd))
            if data_file.is_relative_to(self.cwd)
            else str(data_file)
        )

        # Search through all code files for references to this data file
        for file_path in self.cwd.rglob("*"):
            if (
                file_path.is_file()
                and file_path.suffix.lower() in self.code_extensions
                and file_path != data_file
            ):
                try:
                    with file_path.open(encoding="utf-8", errors="ignore") as f:
                        content = f.read()

                    # Look for references to this data file
                    references = self._find_file_references_in_content(
                        content,
                        data_filename,
                        data_stem,
                        data_relative_path,
                    )

                    if references:
                        ref_info = {
                            "file": str(file_path.relative_to(self.cwd)),
                            "file_type": file_path.suffix,
                            "references": references,
                            "reference_count": len(references),
                        }
                        cross_refs["referenced_by"].append(ref_info)

                except OSError:
                    continue

        # If this is a code file, find what it imports/depends on
        if data_file.suffix.lower() in self.code_extensions:
            try:
                with data_file.open(encoding="utf-8", errors="ignore") as f:
                    content = f.read()

                imports = self._extract_imports_from_content(content, data_file.suffix)
                cross_refs["imports_from"] = imports

                # Look for data file dependencies
                data_deps = self._find_data_dependencies_in_content(content)
                cross_refs["data_dependencies"] = data_deps

            except OSError:
                pass

        # Generate summary
        ref_count = len(cross_refs["referenced_by"])
        import_count = len(cross_refs["imports_from"])
        data_dep_count = len(cross_refs["data_dependencies"])

        summary_parts = []
        if ref_count > 0:
            summary_parts.append(f"Referenced by {ref_count} file(s)")
        if import_count > 0:
            summary_parts.append(f"Imports from {import_count} module(s)")
        if data_dep_count > 0:
            summary_parts.append(f"Depends on {data_dep_count} data file(s)")

        cross_refs["summary"] = (
            "; ".join(summary_parts) if summary_parts else "No cross-references found"
        )

        return cross_refs

    def _find_file_references_in_content(
        self,
        content: str,
        filename: str,
        file_stem: str,
        relative_path: str,
    ) -> list[dict[str, Any]]:
        """Find references to a specific file in the content."""
        references: list[dict[str, Any]] = []
        lines = content.split("\n")

        # Patterns to look for file references
        patterns = [
            # Direct filename references
            (rf"""['"`]{re.escape(filename)}['"`]""", "direct_filename"),
            (rf"""['"`]{re.escape(file_stem)}['"`]""", "stem_reference"),
            (rf"""['"`]{re.escape(relative_path)}['"`]""", "path_reference"),
            # Common data loading patterns
            (
                rf"""(?:read_csv|load|open|Path)\s*\(\s*['"`][^'"`]*{re.escape(file_stem)}[^'"`]*['"`]""",
                "data_loading",
            ),
            # Import patterns for Python modules
            (rf"""(?:from|import)\s+.*{re.escape(file_stem)}""", "import_statement"),
        ]

        for line_num, line in enumerate(lines, 1):
            for pattern, ref_type in patterns:
                matches = re.finditer(pattern, line, re.IGNORECASE)
                references.extend(
                    {
                        "line_number": line_num,
                        "line_content": line.strip(),
                        "match": match.group(),
                        "reference_type": ref_type,
                        "context": self._get_line_context(
                            lines,
                            line_num - 1,
                            context_lines=2,
                        ),
                    }
                    for match in matches
                )

        return references

    def _extract_imports_from_content(
        self,
        content: str,
        file_extension: str,
    ) -> list[dict[str, Any]]:
        """Extract import statements from code content."""
        imports = []
        lines = content.split("\n")

        if file_extension == ".py":
            # Python import patterns
            import_patterns = [
                (r"^import\s+([^\s#]+)", "standard_import"),
                (r"^from\s+([^\s#]+)\s+import", "from_import"),
            ]

            for line_num, line in enumerate(lines, 1):
                stripped_line = line.strip()
                for pattern, import_type in import_patterns:
                    match = re.match(pattern, stripped_line)
                    if match:
                        imports.append(
                            {
                                "module": match.group(1),
                                "line_number": line_num,
                                "import_type": import_type,
                                "full_line": stripped_line,
                            },
                        )

        return imports

    def _find_data_dependencies_in_content(self, content: str) -> list[dict[str, Any]]:
        """Find data file dependencies in code content."""
        dependencies: list[dict[str, Any]] = []
        lines = content.split("\n")

        # Common data file extensions
        data_extensions = [
            ".csv",
            ".json",
            ".xlsx",
            ".parquet",
            ".pkl",
            ".h5",
            ".nc",
            ".geojson",
        ]

        for line_num, line in enumerate(lines, 1):
            for ext in data_extensions:
                # Look for file paths with data extensions
                pattern = rf"""['"`]([^'"`]*\{ext})['"`]"""
                matches = re.finditer(pattern, line, re.IGNORECASE)
                dependencies.extend(
                    {
                        "file_path": match.group(1),
                        "file_extension": ext,
                        "line_number": line_num,
                        "line_content": line.strip(),
                        "context": self._get_line_context(
                            lines,
                            line_num - 1,
                            context_lines=1,
                        ),
                    }
                    for match in matches
                )

        return dependencies

    def _get_line_context(
        self,
        lines: list[str],
        line_index: int,
        context_lines: int = 2,
    ) -> list[str]:
        """Get surrounding lines for context."""
        start = max(0, line_index - context_lines)
        end = min(len(lines), line_index + context_lines + 1)
        return [lines[i].strip() for i in range(start, end)]

    def _calculate_relevance(
        self,
        file_path: Path,
        data_file: Path,
        data_name_parts: list[str],
    ) -> float:
        """Calculate relevance score between a code file and the data file."""
        score = 0.0
        filename_lower = file_path.name.lower()

        # Exact name match (highest score)
        if data_file.stem.lower() in filename_lower:
            score += 10.0

        # Partial name matches
        for part in data_name_parts:
            if len(part) > self.MIN_PART_LEN and part in filename_lower:
                score += 2.0

        # Directory proximity
        try:
            common_path = os.path.commonpath([file_path, data_file])
            if common_path:
                # Files in the same directory get higher score
                if file_path.parent == data_file.parent:
                    score += 3.0
                elif len(Path(common_path).parts) > len(self.cwd.parts):
                    score += 1.0
        except ValueError:
            pass

        # File type bonuses
        if file_path.suffix == ".py":
            score += 1.0

        # Check file content for references to the data file
        content_score = 0.0
        try:
            content_score = self._scan_file_content_for_references(file_path, data_file)
        except OSError as e:
            logger.warning("Could not scan file %s for relevance: %s", file_path, e)
            msg = f"Failed to read file content for scanning: {file_path}"
            raise OSError(msg) from e
        score += content_score
        return score

    def _scan_file_content_for_references(
        self,
        file_path: Path,
        data_file: Path,
    ) -> float:
        """Scan file content for references to the data file."""
        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                content = f.read()

            score = 0.0
            data_filename = data_file.name
            data_stem = data_file.stem

            # Direct filename references
            if data_filename in content:
                score += 5.0

            # Stem references
            if data_stem in content:
                score += 2.0

            # Path references
            if str(data_file) in content:
                score += 3.0

        except OSError:
            return 0.0
        else:
            return score

    def _analyze_python_file(self, file_path: Path, data_stem: str) -> dict[str, Any]:
        """Analyze Python file for data-related patterns."""
        analysis = {
            "has_pandas": False,
            "has_data_io": False,
            "has_relevant_functions": [],
            "imports": [],
        }

        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Check for pandas usage
            if "pandas" in content or "pd." in content:
                analysis["has_pandas"] = True

            # Check for data I/O operations
            io_patterns = [r"\.read_csv", r"\.to_csv", r"\.read_", r"\.to_", r"open\("]
            if any(re.search(pattern, content) for pattern in io_patterns):
                analysis["has_data_io"] = True

            # Find function definitions that might relate to the data
            func_pattern = r"def\s+(\w*" + re.escape(data_stem) + r"\w*)\s*\("
            matches = re.findall(func_pattern, content, re.IGNORECASE)
            analysis["has_relevant_functions"] = matches

            # Extract imports
            import_pattern = r"^(?:from\s+\S+\s+)?import\s+(.+)$"
            imports = re.findall(import_pattern, content, re.MULTILINE)
            analysis["imports"] = [imp.strip() for imp in imports[:10]]  # Limit imports

        except (OSError, ValueError) as e:
            logger.debug("Failed to analyze python file %s: %s", file_path, e)

        return analysis

    def _analyze_model_file(self, file_path: Path, data_stem: str) -> dict[str, Any]:
        """Analyze a potential model file for relevant data structures."""
        analysis: dict[str, Any] = {"has_relevant_models": False, "models": []}

        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                content = f.read()

            # Look for class definitions that might be models
            class_pattern = r"class\s+(\w+).*?:"
            classes = re.findall(class_pattern, content)

            # Look for Pydantic models or dataclasses
            model_indicators = ["BaseModel", "dataclass", "NamedTuple"]
            for indicator in model_indicators:
                if indicator in content:
                    analysis["has_relevant_models"] = True
                    break

            # Check if any class names relate to the data
            for class_name in classes:
                if data_stem.lower() in class_name.lower() or any(
                    part in class_name.lower() for part in data_stem.lower().split("_")
                ):
                    analysis["models"].append(class_name)
                    analysis["has_relevant_models"] = True

        except (OSError, ValueError) as e:
            logger.debug("Failed to analyze model file %s: %s", file_path, e)

        return analysis

    def _get_file_preview(self, file_path: Path, max_lines: int = 10) -> str:
        """Get a preview of the file content."""
        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
                return "\n".join(lines)
        except OSError:
            return "Could not read file content"

    def _get_code_preview(self, file_path: Path, max_lines: int = 25) -> str:
        """Get a preview of code file, focusing on important parts and semantic content."""
        try:
            with file_path.open(encoding="utf-8", errors="ignore") as f:
                content = f.read()

            lines = content.split("\n")
            preview_lines = []

            # Prioritize structural elements AND semantic content
            important_patterns = [
                r"^import\s+",
                r"^from\s+.+import",
                r"^class\s+",
                r"^def\s+",
                r"^async\s+def\s+",
                r".*Field\s*\([^)]*description\s*=",  # Pydantic Field descriptions
                r"^\s*\"\"\".*\"\"\"",  # Docstrings
                r"^\s*#.*",  # Comments
                r".*:\s*[^=]*=.*Field\(",  # Field definitions
            ]

            # First, get important lines
            for line in lines:
                if any(
                    re.match(pattern, line.strip()) for pattern in important_patterns
                ):
                    preview_lines.append(line.rstrip())
                    if len(preview_lines) >= max_lines * 2 // 3:
                        break

            # Then add some regular content
            if len(preview_lines) < max_lines:
                remaining = max_lines - len(preview_lines)
                preview_lines.extend(line.rstrip() for line in lines[:remaining])

            return "\n".join(preview_lines[:max_lines])

        except (OSError, ValueError) as e:
            logger.debug("Failed to get code preview for %s: %s", file_path, e)
            return "Could not read file content"

    def _count_files_scanned(self) -> int:
        """Count total files scanned in the codebase."""
        try:
            return sum(1 for _ in self.cwd.rglob("*") if _.is_file())
        except OSError:
            return 0

    def _get_scan_depth(self) -> int:
        """Get the maximum directory depth scanned."""
        max_depth = 0
        try:
            for file_path in self.cwd.rglob("*"):
                if file_path.is_file():
                    depth = len(file_path.relative_to(self.cwd).parts)
                    max_depth = max(max_depth, depth)
        except (OSError, ValueError):
            return 0
        return max_depth

    def _extract_semantic_knowledge(self) -> dict[str, Any]:
        """Extract semantic knowledge using our comprehensive analysis system."""
        logger.info("🔍 DEBUG: Starting semantic knowledge extraction")
        try:
            # Import here to avoid circular imports
            from metacontext.ai.prompts.context_preprocessor import (
                build_semantic_knowledge_graph,
            )

            # Collect all Python files for semantic analysis
            files_content = {}
            for file_path in self.cwd.rglob("*.py"):
                if file_path.is_file() and self._is_relevant_code_file(file_path):
                    try:
                        with file_path.open(encoding="utf-8") as f:
                            relative_path = str(file_path.relative_to(self.cwd))
                            files_content[relative_path] = f.read()
                            logger.info(
                                "🔍 DEBUG: Added file for semantic analysis: %s",
                                relative_path,
                            )
                    except (UnicodeDecodeError, OSError) as e:
                        logger.warning("Could not read file %s: %s", file_path, e)
                        continue

            logger.info(
                "🔍 DEBUG: Found %d Python files for semantic analysis",
                len(files_content),
            )

            # Run semantic analysis if we have files
            if files_content:
                logger.info("🔍 DEBUG: Running semantic knowledge extraction on files")
                result = build_semantic_knowledge_graph(files_content)
                logger.info(
                    "🔍 DEBUG: Semantic knowledge result: %s",
                    str(result)[:200] + "...",
                )
                return result
            logger.info("🔍 DEBUG: No Python files found for semantic analysis")
            return {
                "semantic_knowledge": None,
                "message": "No Python files found for semantic analysis",
            }

        except (ImportError, AttributeError) as e:
            logger.warning("Semantic knowledge extraction failed: %s", e)
            return {"semantic_knowledge": None, "error": str(e)}

    def _is_relevant_code_file(self, file_path: Path) -> bool:
        """Check if a code file is relevant for semantic analysis."""
        # Skip test files, cache, and other non-relevant files
        path_str = str(file_path)
        skip_patterns = [
            "__pycache__",
            ".pytest_cache",
            ".git",
            "node_modules",
            ".venv",
            ".env",
            "test_",
            "_test.py",
            "/tests/",
        ]
        return not any(pattern in path_str for pattern in skip_patterns)


def scan_codebase_context(
    data_file: Path,
    cwd: Path | None = None,
) -> dict[str, Any]:
    """Scan codebase for context related to a data file.

    Args:
        data_file: Path to the data file being metacontextualized
        cwd: Optional working directory (defaults to current directory)

    Returns:
        Dictionary containing discovered context information

    """
    scanner = CodebaseScanner(cwd)
    return scanner.scan_for_context(data_file)
