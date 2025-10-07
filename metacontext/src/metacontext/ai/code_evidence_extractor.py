"""Code evidence extractor for Phase 3 enhancement.

This module extracts actual code snippets, comments, and field descriptions
to provide concrete code context for enhanced semantic analysis.
"""

import ast
import logging
import re
from pathlib import Path

from metacontext.schemas.core.code_evidence import (
    CodeEvidence,
    CodeSnippet,
    CommentWithContext,
    FieldDescription,
)

logger = logging.getLogger(__name__)


class CodeEvidenceExtractor:
    """Extracts code evidence from source files for semantic analysis enhancement."""

    def __init__(self, cwd: Path | None = None) -> None:
        """Initialize the code evidence extractor."""
        self.cwd = Path(cwd) if cwd else Path.cwd()
        self.python_files: list[Path] = []
        self._scan_python_files()

    def _scan_python_files(self) -> None:
        """Scan for Python files in the codebase."""
        # Directories to exclude
        excluded_dirs = {
            "__pycache__",
            ".git",
            ".pytest_cache",
            ".mypy_cache",
            ".ruff_cache",
            ".venv",
            "venv",
            "env",
            "node_modules",
            "site-packages",
            "dist",
            "build",
            ".ipynb_checkpoints",
        }

        for file_path in self.cwd.rglob("*.py"):
            # Check if any parent directory is in excluded_dirs
            if not any(part in excluded_dirs for part in file_path.parts):
                self.python_files.append(file_path)

        logger.debug(
            f"Found {len(self.python_files)} Python files for code evidence extraction",
        )

    def extract_for_column(self, column_name: str) -> CodeEvidence:
        """Extract code evidence for a specific column name.

        Args:
            column_name: Name of the column to find evidence for

        Returns:
            CodeEvidence object containing related code snippets and context

        """
        evidence = CodeEvidence()

        for file_path in self.python_files:
            try:
                file_evidence = self._extract_from_file(file_path, column_name)
                self._merge_evidence(evidence, file_evidence)
            except Exception as e:
                logger.debug(f"Error extracting evidence from {file_path}: {e}")

        return evidence

    def extract_for_dataset(self, data_reference: str | None = None) -> CodeEvidence:
        """Extract code evidence for an entire dataset.

        Args:
            data_reference: Optional reference to the dataset (filename, variable name, etc.)

        Returns:
            CodeEvidence object containing related code snippets and context

        """
        evidence = CodeEvidence()

        for file_path in self.python_files:
            try:
                file_evidence = self._extract_dataset_evidence(
                    file_path,
                    data_reference,
                )
                self._merge_evidence(evidence, file_evidence)
            except Exception as e:
                logger.debug(f"Error extracting dataset evidence from {file_path}: {e}")

        return evidence

    def _extract_from_file(self, file_path: Path, column_name: str) -> CodeEvidence:
        """Extract code evidence from a single file for a column."""
        evidence = CodeEvidence()

        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            tree = ast.parse(content)

            # Extract snippets mentioning the column
            evidence.related_snippets.extend(
                self._find_column_references(content, file_path, column_name),
            )

            # Extract Pydantic field descriptions
            evidence.field_descriptions.extend(
                self._extract_pydantic_field_descriptions(
                    tree,
                    content,
                    file_path,
                    column_name,
                ),
            )

            # Extract comments mentioning the column
            evidence.associated_comments.extend(
                self._extract_comments_for_column(content, file_path, column_name),
            )

            # Extract data transformations
            evidence.data_transformations.extend(
                self._find_data_transformations(tree, content, file_path, column_name),
            )

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")

        return evidence

    def _extract_dataset_evidence(
        self,
        file_path: Path,
        data_reference: str | None,
    ) -> CodeEvidence:
        """Extract code evidence for a dataset from a single file."""
        evidence = CodeEvidence()

        try:
            with file_path.open("r", encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            # Extract general data processing snippets
            evidence.related_snippets.extend(
                self._find_data_processing_code(tree, content, file_path),
            )

            # Extract comments about data processing
            evidence.associated_comments.extend(
                self._extract_data_processing_comments(content, file_path),
            )

        except Exception as e:
            logger.debug(f"Error parsing {file_path}: {e}")

        return evidence

    def _find_column_references(
        self,
        content: str,
        file_path: Path,
        column_name: str,
    ) -> list[CodeSnippet]:
        """Find specific defining code snippets for a column (creation, assignment, definition)."""
        snippets = []
        lines = content.split("\n")

        # Look for DEFINING patterns only - where the column is created or explained
        defining_patterns = [
            # DataFrame column assignment: df["column"] = ... or df['column'] = ...
            rf"[a-zA-Z_][a-zA-Z0-9_]*\s*\[\s*[\"'{re.escape(column_name)}[\"']\s*\]\s*=",
            # Pydantic field definition: column_name: Type = Field(...)
            rf"^\s*{re.escape(column_name)}\s*:\s*[^=]*=.*Field\(",
            # Direct variable assignment: column_name = ...
            rf"^\s*{re.escape(column_name)}\s*[:=]",
            # Column with inline comment: column_name ... # explanation
            rf"{re.escape(column_name)}.*#.*",
        ]

        for i, line in enumerate(lines):
            for pattern in defining_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    # For defining lines, we want just the key line, not lots of context
                    snippet = CodeSnippet(
                        content=line.strip(),
                        file_path=str(file_path.relative_to(self.cwd)),
                        start_line=i + 1,
                        end_line=i + 1,
                    )
                    snippets.append(snippet)
                    break  # Only add one snippet per line

        return snippets

    def _extract_pydantic_field_descriptions(
        self,
        tree: ast.AST,
        content: str,
        file_path: Path,
        column_name: str,
    ) -> list[FieldDescription]:
        """Extract Pydantic field descriptions that match the column name."""
        descriptions = []
        lines = content.split("\n")
        cwd = self.cwd  # Capture cwd for nested class

        class FieldVisitor(ast.NodeVisitor):
            def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
                """Visit annotated assignments (Pydantic fields)."""
                if isinstance(node.target, ast.Name) and node.target.id == column_name:
                    if isinstance(node.value, ast.Call) and isinstance(
                        node.value.func,
                        ast.Name,
                    ):
                        if node.value.func.id == "Field":
                            # Extract Field arguments
                            field_desc = self._extract_field_description(
                                node,
                                lines,
                                file_path,
                            )
                            if field_desc:
                                descriptions.append(field_desc)

            def _extract_field_description(
                self,
                node: ast.AnnAssign,
                lines: list[str],
                file_path: Path,
            ) -> FieldDescription | None:
                """Extract description from Field call."""
                if not isinstance(node.value, ast.Call):
                    return None

                description_text = None
                field_type = None

                # Extract description from Field kwargs
                for keyword in node.value.keywords:
                    if keyword.arg == "description" and isinstance(
                        keyword.value,
                        ast.Constant,
                    ):
                        description_text = keyword.value.value

                # Extract type annotation
                if node.annotation:
                    field_type = ast.unparse(node.annotation)

                if description_text:
                    # Get just the defining line, not lots of context
                    line_content = lines[node.lineno - 1].strip()

                    source_context = CodeSnippet(
                        content=line_content,
                        file_path=str(file_path.relative_to(cwd)),
                        start_line=node.lineno,
                        end_line=node.lineno,
                    )

                    return FieldDescription(
                        field_name=column_name,
                        description=description_text,
                        field_type=field_type,
                        source_context=source_context,
                    )

                return None

        visitor = FieldVisitor()
        visitor.visit(tree)
        return descriptions

    def _extract_comments_for_column(
        self,
        content: str,
        file_path: Path,
        column_name: str,
    ) -> list[CommentWithContext]:
        """Extract comments that mention the column name."""
        comments = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            # Check for comments mentioning the column
            if "#" in line and column_name.lower() in line.lower():
                comment_match = re.search(r"#\s*(.+)", line)
                if comment_match:
                    comment_text = comment_match.group(1).strip()

                    # Get surrounding code context
                    start_line = max(0, i - 2)
                    end_line = min(len(lines), i + 3)
                    context_content = "\n".join(lines[start_line:end_line])

                    related_code = CodeSnippet(
                        content=context_content,
                        file_path=str(file_path.relative_to(self.cwd)),
                        start_line=start_line + 1,
                        end_line=end_line,
                    )

                    comment = CommentWithContext(
                        comment_text=comment_text,
                        comment_type="inline",
                        related_code=related_code,
                    )
                    comments.append(comment)

        return comments

    def _find_data_transformations(
        self,
        tree: ast.AST,
        content: str,
        file_path: Path,
        column_name: str,
    ) -> list[CodeSnippet]:
        """Find code snippets showing data transformations for the column."""
        snippets = []
        lines = content.split("\n")
        cwd = self.cwd  # Capture cwd for nested class

        class TransformVisitor(ast.NodeVisitor):
            def visit_Assign(self, node: ast.Assign) -> None:
                """Visit assignments that might transform the column."""
                # Look for assignments involving the column
                node_str = ast.unparse(node)
                if column_name in node_str:
                    start_line = max(0, node.lineno - 2)
                    end_line = min(
                        len(lines),
                        node.end_lineno + 2 if node.end_lineno else node.lineno + 2,
                    )
                    snippet_content = "\n".join(lines[start_line:end_line])

                    snippet = CodeSnippet(
                        content=snippet_content,
                        file_path=str(file_path.relative_to(cwd)),
                        start_line=start_line + 1,
                        end_line=end_line,
                    )
                    snippets.append(snippet)

        visitor = TransformVisitor()
        visitor.visit(tree)
        return snippets

    def _find_data_processing_code(
        self,
        tree: ast.AST,
        content: str,
        file_path: Path,
    ) -> list[CodeSnippet]:
        """Find general data processing code snippets."""
        snippets = []
        lines = content.split("\n")
        cwd = self.cwd  # Capture cwd for nested class

        class DataProcessingVisitor(ast.NodeVisitor):
            def visit_Call(self, node: ast.Call) -> None:
                """Visit function calls that look like data processing."""
                # Look for common data processing functions
                data_functions = {
                    "read_csv",
                    "to_csv",
                    "read_excel",
                    "to_excel",
                    "DataFrame",
                    "groupby",
                    "merge",
                    "join",
                    "apply",
                    "transform",
                    "agg",
                    "aggregate",
                }

                if isinstance(node.func, ast.Attribute):
                    func_name = node.func.attr
                elif isinstance(node.func, ast.Name):
                    func_name = node.func.id
                else:
                    return

                if func_name in data_functions:
                    # Get just the line with the function call
                    line_content = lines[node.lineno - 1].strip()

                    snippet = CodeSnippet(
                        content=line_content,
                        file_path=str(file_path.relative_to(cwd)),
                        start_line=node.lineno,
                        end_line=node.lineno,
                    )
                    snippets.append(snippet)

        visitor = DataProcessingVisitor()
        visitor.visit(tree)
        return snippets

    def _extract_data_processing_comments(
        self,
        content: str,
        file_path: Path,
    ) -> list[CommentWithContext]:
        """Extract comments related to data processing."""
        comments = []
        lines = content.split("\n")

        # Keywords that suggest data processing comments
        data_keywords = {
            "data",
            "dataset",
            "csv",
            "excel",
            "dataframe",
            "column",
            "field",
            "process",
            "transform",
            "clean",
            "filter",
        }

        for i, line in enumerate(lines):
            if "#" in line:
                comment_match = re.search(r"#\s*(.+)", line)
                if comment_match:
                    comment_text = comment_match.group(1).strip().lower()

                    # Check if comment mentions data processing
                    if any(keyword in comment_text for keyword in data_keywords):
                        # Get surrounding code context
                        start_line = max(0, i - 2)
                        end_line = min(len(lines), i + 3)
                        context_content = "\n".join(lines[start_line:end_line])

                        related_code = CodeSnippet(
                            content=context_content,
                            file_path=str(file_path.relative_to(self.cwd)),
                            start_line=start_line + 1,
                            end_line=end_line,
                        )

                        comment = CommentWithContext(
                            comment_text=comment_match.group(1).strip(),
                            comment_type="inline",
                            related_code=related_code,
                        )
                        comments.append(comment)

        return comments

    def _merge_evidence(self, target: CodeEvidence, source: CodeEvidence) -> None:
        """Merge source evidence into target evidence."""
        target.related_snippets.extend(source.related_snippets)
        target.field_descriptions.extend(source.field_descriptions)
        target.associated_comments.extend(source.associated_comments)
        target.data_transformations.extend(source.data_transformations)
        target.validation_logic.extend(source.validation_logic)
        target.cross_references.update(source.cross_references)
