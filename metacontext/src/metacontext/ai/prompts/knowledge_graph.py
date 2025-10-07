"""Knowledge graph construction for semantic codebase relationships.

This module builds relationships between extracted semantic information to create
a cohesive knowledge system for LLM context enhancement.
"""

import re
from dataclasses import dataclass, field
from typing import Any

# Constants
MIN_COLUMN_NAME_LENGTH = 2
HIGH_CONFIDENCE_THRESHOLD = 0.7
MEDIUM_CONFIDENCE_THRESHOLD = 0.4
MIN_MATCH_LENGTH = 2

# Cross-reference relationship types
RELATIONSHIP_DERIVES_FROM = "derives_from"
RELATIONSHIP_MAPS_TO = "maps_to"
RELATIONSHIP_VALIDATES = "validates"
RELATIONSHIP_TRANSFORMS = "transforms"
RELATIONSHIP_REFERENCES = "references"
RELATIONSHIP_INHERITS = "inherits"

# Conflict resolution constants
CONFLICT_HIGH_CONFIDENCE_THRESHOLD = 0.8
CONFLICT_MEDIUM_CONFIDENCE_THRESHOLD = 0.5
NAME_SIMILARITY_THRESHOLD = 0.8
DEFINITION_SIMILARITY_THRESHOLD = 0.9


@dataclass
class ColumnUpdateData:
    """Data structure for column update parameters."""

    aliases: list[str] | None = None
    definition: str | None = None
    inline_comment: str | None = None
    docstring_reference: str | None = None
    pydantic_description: str | None = None
    validation_rule: str | None = None
    business_context: str | None = None
    source_file: str = ""
    line_number: int = 0


@dataclass
class ColumnKnowledge:
    """Represents comprehensive knowledge about a data column."""

    name: str
    aliases: list[str] = field(default_factory=list)
    definition: str | None = None
    inline_comments: list[str] = field(default_factory=list)
    docstring_references: list[str] = field(default_factory=list)
    pydantic_description: str | None = None
    validation_rules: list[str] = field(default_factory=list)
    business_context: list[str] = field(default_factory=list)
    confidence_score: float = 0.0
    source_files: list[str] = field(default_factory=list)
    line_references: list[dict[str, str | int]] = field(default_factory=list)


@dataclass
class CrossReference:
    """Represents a cross-reference between different code elements."""

    source_type: str  # 'function_param', 'model_field', 'variable', 'constant'
    source_name: str
    source_file: str
    source_line: int
    target_type: str
    target_name: str
    target_file: str
    target_line: int
    relationship: str  # 'derives_from', 'maps_to', 'validates', 'transforms'
    confidence: float


class SemanticKnowledgeGraph:
    """Builds and manages semantic relationships in codebase."""

    def __init__(self) -> None:
        """Initialize the semantic knowledge graph."""
        self.columns: dict[str, ColumnKnowledge] = {}
        self.cross_references: list[CrossReference] = []
        self.pydantic_info: dict[str, Any] = {}
        self.semantic_info: dict[str, Any] = {}

    def add_pydantic_schema_info(self, pydantic_data: dict[str, Any]) -> None:
        """Add Pydantic schema information to the knowledge graph."""
        self.pydantic_info = pydantic_data

        # Extract field descriptions for column knowledge
        for field_desc in pydantic_data.get("field_descriptions", []):
            column_name = field_desc.get("field", "")
            if column_name:
                self._add_or_update_column(
                    name=column_name,
                    update_data=ColumnUpdateData(
                        pydantic_description=field_desc.get("description"),
                        source_file=field_desc.get("file", ""),
                        line_number=field_desc.get("line", 0),
                    ),
                )

        # Extract validation rules
        for validation in pydantic_data.get("validation_rules", []):
            field_names = validation.get("field_names", [])
            rule_description = validation.get("docstring", "No description")

            for field_name in field_names:
                if field_name in self.columns:
                    self.columns[field_name].validation_rules.append(rule_description)

    def add_semantic_extraction_info(self, semantic_data: dict[str, Any]) -> None:
        """Add semantic extraction information to the knowledge graph."""
        self.semantic_info = semantic_data

        # Extract function information for column relationships
        for function in semantic_data.get("functions", []):
            self._analyze_function_for_columns(function)

        # Extract constants that might be column-related
        for constant in semantic_data.get("constants", []):
            self._analyze_constant_for_columns(constant)

        # Extract business logic for column context
        for logic in semantic_data.get("business_logic", []):
            self._analyze_business_logic_for_columns(logic)

    def _add_or_update_column(self, name: str, update_data: ColumnUpdateData) -> None:
        """Add or update column knowledge using ColumnUpdateData."""
        if name not in self.columns:
            self.columns[name] = ColumnKnowledge(name=name)

        column = self.columns[name]

        # Update fields if provided
        if update_data.aliases:
            column.aliases.extend(
                [a for a in update_data.aliases if a not in column.aliases],
            )
        if update_data.definition and not column.definition:
            column.definition = update_data.definition
        if update_data.inline_comment:
            column.inline_comments.append(update_data.inline_comment)
        if update_data.docstring_reference:
            column.docstring_references.append(update_data.docstring_reference)
        if update_data.pydantic_description and not column.pydantic_description:
            column.pydantic_description = update_data.pydantic_description
        if update_data.validation_rule:
            column.validation_rules.append(update_data.validation_rule)
        if update_data.business_context:
            column.business_context.append(update_data.business_context)

        # Track source information
        if (
            update_data.source_file
            and update_data.source_file not in column.source_files
        ):
            column.source_files.append(update_data.source_file)
        if update_data.line_number > 0:
            column.line_references.append(
                {
                    "file": update_data.source_file,
                    "line": update_data.line_number,
                },
            )

        # Update confidence score
        column.confidence_score = self._calculate_confidence_score(column)

    def _calculate_confidence_score(self, column: ColumnKnowledge) -> float:
        """Calculate confidence score based on available information."""
        score = 0.0

        # Base score for having a name
        score += 0.1

        # Pydantic description is highly valuable
        if column.pydantic_description:
            score += 0.4

        # Definition from docstring or comment
        if column.definition:
            score += 0.3

        # Inline comments provide context
        score += min(len(column.inline_comments) * 0.1, 0.2)

        # Validation rules show business importance
        score += min(len(column.validation_rules) * 0.1, 0.2)

        # Business context adds understanding
        score += min(len(column.business_context) * 0.05, 0.15)

        # Multiple sources increase confidence
        if len(column.source_files) > 1:
            score += 0.1

        # Aliases show broader usage
        if column.aliases:
            score += 0.05

        return min(score, 1.0)

    def _analyze_function_for_columns(self, function: dict[str, Any]) -> None:
        """Analyze function information for column relationships."""
        func_name = function.get("name", "")
        docstring = function.get("docstring", "")
        creates = function.get("creates", [])
        returns = function.get("returns", [])
        modifies = function.get("modifies", [])

        # Look for column-related patterns in function name
        column_patterns = [
            r"create_(\w+)_column",
            r"(\w+)_derivation",
            r"calculate_(\w+)",
            r"extract_(\w+)",
            r"transform_(\w+)",
        ]

        for pattern in column_patterns:
            match = re.search(pattern, func_name, re.IGNORECASE)
            if match:
                potential_column = match.group(1)
                self._add_or_update_column(
                    name=potential_column,
                    update_data=ColumnUpdateData(
                        definition=f"Created/processed by {func_name}()",
                        docstring_reference=docstring[:100] if docstring else "",
                        source_file=function.get("file", ""),
                        line_number=function.get("line", 0),
                    ),
                )

        # Analyze creates/returns/modifies for column information
        all_outputs = creates + returns + modifies
        for output in all_outputs:
            if self._looks_like_column_name(output):
                self._add_or_update_column(
                    name=output,
                    update_data=ColumnUpdateData(
                        business_context=f"Function {func_name}: {output}",
                        source_file=function.get("file", ""),
                        line_number=function.get("line", 0),
                    ),
                )

    def _analyze_constant_for_columns(self, constant: dict[str, Any]) -> None:
        """Analyze constants for column-related information."""
        const_name = constant.get("name", "")
        const_value = constant.get("value", "")
        comment = constant.get("comment", "")

        # Check if constant name suggests column relationship
        column_related_patterns = [
            r"(\w+)_COLUMN",
            r"(\w+)_FIELD",
            r"(\w+)_ATTR",
            r"DEFAULT_(\w+)",
        ]

        for pattern in column_related_patterns:
            match = re.search(pattern, const_name, re.IGNORECASE)
            if match:
                potential_column = match.group(1).lower()
                context = f"Constant {const_name} = {const_value}"
                if comment:
                    context += f" ({comment})"

                self._add_or_update_column(
                    name=potential_column,
                    update_data=ColumnUpdateData(
                        business_context=context,
                        source_file=constant.get("file", ""),
                        line_number=constant.get("line", 0),
                    ),
                )

    def _analyze_business_logic_for_columns(self, logic: dict[str, Any]) -> None:
        """Analyze business logic for column-related context."""
        condition = logic.get("condition", "")
        logic_type = logic.get("logic_type", "")
        thresholds = logic.get("thresholds", [])

        # Extract potential column names from conditions
        # Look for patterns like: column_name > threshold, df['column'], etc.
        column_patterns = [
            r"(\w+)\s*[><=!]+\s*[\d.]+",  # column > threshold
            r"df\[[\"\'](\w+)[\"\']\]",  # df['column']
            r"(\w+)\.(\w+)",  # object.attribute
        ]

        for pattern in column_patterns:
            matches = re.findall(pattern, condition, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    # Handle tuple matches from grouped patterns
                    potential_columns = [
                        m for m in match if m and self._looks_like_column_name(m)
                    ]
                else:
                    potential_columns = (
                        [match] if self._looks_like_column_name(match) else []
                    )

                for column_name in potential_columns:
                    context = f"Business logic ({logic_type}): {condition[:50]}..."
                    if thresholds:
                        threshold_info = ", ".join(
                            [
                                f"{t.get('value')} ({t.get('operator')})"
                                for t in thresholds[:2]
                            ],
                        )
                        context += f" [Thresholds: {threshold_info}]"

                    self._add_or_update_column(
                        name=column_name,
                        update_data=ColumnUpdateData(
                            business_context=context,
                            source_file=logic.get("file", ""),
                            line_number=logic.get("line", 0),
                        ),
                    )

    def _looks_like_column_name(self, name: str) -> bool:
        """Check if a name looks like a data column name."""
        if not name or len(name) < MIN_COLUMN_NAME_LENGTH:
            return False

        # Common column name patterns
        column_indicators = [
            r"^[a-z][a-z0-9_]*$",  # snake_case
            r".*_id$",  # ends with _id
            r".*_name$",  # ends with _name
            r".*_date$",  # ends with _date
            r".*_time$",  # ends with _time
            r".*_count$",  # ends with _count
            r".*_amount$",  # ends with _amount
            r".*_score$",  # ends with _score
        ]

        return any(
            re.match(pattern, name, re.IGNORECASE) for pattern in column_indicators
        )

    def resolve_column_aliases(self) -> None:
        """Resolve and merge columns that are likely aliases of each other."""
        # Simple alias resolution based on name similarity
        column_names = list(self.columns.keys())

        for i, name1 in enumerate(column_names):
            for name2 in column_names[i + 1 :]:
                if self._are_likely_aliases(name1, name2):
                    # Merge the columns
                    self._merge_columns(name1, name2)

    def _are_likely_aliases(self, name1: str, name2: str) -> bool:
        """Check if two names are likely aliases of the same column."""
        # Remove common prefixes/suffixes
        clean1 = re.sub(r"^(df_|data_|col_)", "", name1, flags=re.IGNORECASE)
        clean2 = re.sub(r"^(df_|data_|col_)", "", name2, flags=re.IGNORECASE)

        clean1 = re.sub(r"(_col|_field|_attr)$", "", clean1, flags=re.IGNORECASE)
        clean2 = re.sub(r"(_col|_field|_attr)$", "", clean2, flags=re.IGNORECASE)

        # Check for exact match after cleaning
        if clean1.lower() == clean2.lower():
            return True

        # Check for underscore vs camelCase variations
        return clean1.lower().replace("_", "") == clean2.lower().replace("_", "")

    def _merge_columns(self, primary_name: str, alias_name: str) -> None:
        """Merge information from alias column into primary column."""
        if primary_name not in self.columns or alias_name not in self.columns:
            return

        primary = self.columns[primary_name]
        alias = self.columns[alias_name]

        # Add alias name to aliases list
        if alias_name not in primary.aliases:
            primary.aliases.append(alias_name)

        # Merge information, preferring primary when conflicts exist
        primary.aliases.extend([a for a in alias.aliases if a not in primary.aliases])
        primary.inline_comments.extend(alias.inline_comments)
        primary.docstring_references.extend(alias.docstring_references)
        primary.validation_rules.extend(alias.validation_rules)
        primary.business_context.extend(alias.business_context)
        primary.source_files.extend(
            [f for f in alias.source_files if f not in primary.source_files],
        )
        primary.line_references.extend(alias.line_references)

        # Use better definition if primary doesn't have one
        if not primary.definition and alias.definition:
            primary.definition = alias.definition
        if not primary.pydantic_description and alias.pydantic_description:
            primary.pydantic_description = alias.pydantic_description

        # Recalculate confidence score
        primary.confidence_score = self._calculate_confidence_score(primary)

        # Remove the alias column
        del self.columns[alias_name]

    def get_high_confidence_columns(
        self,
        min_confidence: float = 0.5,
    ) -> dict[str, ColumnKnowledge]:
        """Get columns with confidence score above threshold."""
        return {
            name: column
            for name, column in self.columns.items()
            if column.confidence_score >= min_confidence
        }

    def generate_column_summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of column knowledge."""
        total_columns = len(self.columns)
        high_confidence = len(self.get_high_confidence_columns())

        # Group by confidence ranges
        confidence_distribution = {
            "high (0.7+)": len(
                [
                    c
                    for c in self.columns.values()
                    if c.confidence_score >= HIGH_CONFIDENCE_THRESHOLD
                ],
            ),
            "medium (0.4-0.7)": len(
                [
                    c
                    for c in self.columns.values()
                    if MEDIUM_CONFIDENCE_THRESHOLD
                    <= c.confidence_score
                    < HIGH_CONFIDENCE_THRESHOLD
                ],
            ),
            "low (<0.4)": len(
                [
                    c
                    for c in self.columns.values()
                    if c.confidence_score < MEDIUM_CONFIDENCE_THRESHOLD
                ],
            ),
        }

        # Top columns by confidence
        top_columns = sorted(
            self.columns.items(),
            key=lambda x: x[1].confidence_score,
            reverse=True,
        )[:10]

        return {
            "total_columns": total_columns,
            "high_confidence_columns": high_confidence,
            "confidence_distribution": confidence_distribution,
            "top_columns": [
                {
                    "name": name,
                    "confidence": round(column.confidence_score, 3),
                    "sources": len(column.source_files),
                    "description": column.pydantic_description
                    or column.definition
                    or "No description",
                }
                for name, column in top_columns
            ],
        }

    def build_cross_references(self) -> None:
        """Build cross-references between code elements and columns.

        This method implements Task 5.2: Cross-Reference Resolution by:
        - Linking function parameters to model fields
        - Resolving variable assignments and transformations
        - Building dependency graphs between code elements
        - Detecting column lineage through data processing pipelines
        """
        self._resolve_function_parameter_mappings()
        self._resolve_variable_assignments()
        self._build_column_dependency_graphs()
        self._detect_data_pipeline_lineage()

    def _resolve_function_parameter_mappings(self) -> None:
        """Resolve mappings between function parameters and model fields."""
        for function in self.semantic_info.get("functions", []):
            func_name = function.get("name", "")
            file_path = function.get("file", "")
            line_number = function.get("line", 0)

            # Extract parameter names and types
            parameters = function.get("parameters", [])
            creates = function.get("creates", [])
            modifies = function.get("modifies", [])
            returns = function.get("returns", [])

            # Map function parameters to known columns
            for param in parameters:
                param_name = param.get("name", "")

                # Check if parameter name matches a column
                if param_name in self.columns:
                    self.cross_references.append(
                        CrossReference(
                            source_type="function_param",
                            source_name=f"{func_name}({param_name})",
                            source_file=file_path,
                            source_line=line_number,
                            target_type="column",
                            target_name=param_name,
                            target_file=self.columns[param_name].source_files[0]
                            if self.columns[param_name].source_files
                            else "",
                            target_line=0,
                            relationship=RELATIONSHIP_REFERENCES,
                            confidence=0.8,
                        ),
                    )

            # Map function outputs to columns
            all_outputs = creates + returns + modifies
            for output in all_outputs:
                if output in self.columns:
                    relationship = (
                        RELATIONSHIP_TRANSFORMS
                        if output in modifies
                        else RELATIONSHIP_DERIVES_FROM
                    )
                    self.cross_references.append(
                        CrossReference(
                            source_type="function",
                            source_name=func_name,
                            source_file=file_path,
                            source_line=line_number,
                            target_type="column",
                            target_name=output,
                            target_file=self.columns[output].source_files[0]
                            if self.columns[output].source_files
                            else "",
                            target_line=0,
                            relationship=relationship,
                            confidence=0.7,
                        ),
                    )

    def _resolve_variable_assignments(self) -> None:
        """Resolve variable assignments and transformations."""
        # Look for patterns in business logic that show variable transformations
        for logic in self.semantic_info.get("business_logic", []):
            condition = logic.get("condition", "")
            file_path = logic.get("file", "")
            line_number = logic.get("line", 0)

            # Extract assignment patterns like "new_column = transform(old_column)"
            assignment_patterns = [
                r"(\w+)\s*=\s*transform\w*\(.*?(\w+)",  # new = transform(old)
                r"(\w+)\s*=\s*calculate\w*\(.*?(\w+)",  # new = calculate(old)
                r"(\w+)\s*=\s*(\w+)\s*[\+\-\*\/]",  # new = old + something
                r"df\[[\"\'](\w+)[\"\']\]\s*=.*?df\[[\"\'](\w+)[\"\']\]",  # df['new'] = df['old']
            ]

            for pattern in assignment_patterns:
                matches = re.findall(pattern, condition, re.IGNORECASE)
                for match in matches:
                    if len(match) >= MIN_MATCH_LENGTH:
                        new_col, old_col = match[0], match[1]
                        if (
                            self._looks_like_column_name(new_col)
                            and self._looks_like_column_name(old_col)
                            and old_col in self.columns
                        ):
                            # Add the new column if it doesn't exist
                            if new_col not in self.columns:
                                self._add_or_update_column(
                                    name=new_col,
                                    update_data=ColumnUpdateData(
                                        definition=f"Derived from {old_col}",
                                        business_context=f"Variable assignment: {condition[:50]}...",
                                        source_file=file_path,
                                        line_number=line_number,
                                    ),
                                )

                            # Create cross-reference
                            self.cross_references.append(
                                CrossReference(
                                    source_type="variable",
                                    source_name=old_col,
                                    source_file=file_path,
                                    source_line=line_number,
                                    target_type="variable",
                                    target_name=new_col,
                                    target_file=file_path,
                                    target_line=line_number,
                                    relationship=RELATIONSHIP_DERIVES_FROM,
                                    confidence=0.6,
                                ),
                            )

    def _build_column_dependency_graphs(self) -> None:
        """Build dependency graphs showing how columns depend on each other."""
        # Create a dependency map based on cross-references
        self.dependency_graph: dict[str, list[str]] = {}

        for ref in self.cross_references:
            if ref.relationship in [RELATIONSHIP_DERIVES_FROM, RELATIONSHIP_TRANSFORMS]:
                target = ref.target_name
                source = ref.source_name

                if target not in self.dependency_graph:
                    self.dependency_graph[target] = []
                if source not in self.dependency_graph[target]:
                    self.dependency_graph[target].append(source)

    def _detect_data_pipeline_lineage(self) -> None:
        """Detect column lineage through data processing pipelines."""
        # Look for data pipeline patterns in function names and operations
        pipeline_patterns = [
            r"extract_(\w+)",  # extract operations
            r"transform_(\w+)",  # transform operations
            r"load_(\w+)",  # load operations
            r"process_(\w+)",  # process operations
            r"clean_(\w+)",  # cleaning operations
            r"validate_(\w+)",  # validation operations
        ]

        for function in self.semantic_info.get("functions", []):
            func_name = function.get("name", "")
            file_path = function.get("file", "")
            line_number = function.get("line", 0)

            for pattern in pipeline_patterns:
                match = re.search(pattern, func_name, re.IGNORECASE)
                if match:
                    column_name = match.group(1)
                    if self._looks_like_column_name(column_name):
                        # Add pipeline context to column
                        if column_name in self.columns:
                            pipeline_stage = (
                                pattern.split("(")[0].replace("_", " ").title()
                            )
                            context = f"{pipeline_stage} pipeline stage: {func_name}()"
                            self.columns[column_name].business_context.append(context)

                        # Create pipeline cross-reference
                        self.cross_references.append(
                            CrossReference(
                                source_type="pipeline_function",
                                source_name=func_name,
                                source_file=file_path,
                                source_line=line_number,
                                target_type="column",
                                target_name=column_name,
                                target_file=self.columns.get(
                                    column_name,
                                    ColumnKnowledge(name=""),
                                ).source_files[0]
                                if self.columns.get(column_name)
                                and self.columns[column_name].source_files
                                else "",
                                target_line=0,
                                relationship=RELATIONSHIP_TRANSFORMS,
                                confidence=0.5,
                            ),
                        )

    def get_column_lineage(self, column_name: str) -> dict[str, Any]:
        """Get the complete lineage (dependencies and dependents) for a column."""
        if not hasattr(self, "dependency_graph"):
            self._build_column_dependency_graphs()

        # Find what this column depends on
        dependencies = self.dependency_graph.get(column_name, [])

        # Find what depends on this column
        dependents = [
            col for col, deps in self.dependency_graph.items() if column_name in deps
        ]

        # Get related cross-references
        related_refs = [
            ref
            for ref in self.cross_references
            if column_name in (ref.source_name, ref.target_name)
        ]

        return {
            "column": column_name,
            "dependencies": dependencies,
            "dependents": dependents,
            "cross_references": [
                {
                    "source": f"{ref.source_type}:{ref.source_name}",
                    "target": f"{ref.target_type}:{ref.target_name}",
                    "relationship": ref.relationship,
                    "confidence": ref.confidence,
                    "file": ref.source_file,
                }
                for ref in related_refs
            ],
        }

    def resolve_conflicts_and_consolidate(self) -> None:
        """Task 5.3: Resolve conflicts and consolidate knowledge.

        Handles cases where multiple sources provide different information
        about the same column and consolidates for better accuracy.
        """
        # Identify conflicting definitions
        conflicts = self._identify_definition_conflicts()

        # Resolve conflicts using various strategies
        for conflict in conflicts:
            self._resolve_definition_conflict(conflict)

        # Consolidate aliases and remove duplicates
        self._consolidate_aliases()

        # Merge similar columns with high confidence
        self._merge_similar_columns()

        # Update confidence scores based on consolidation
        self._update_consolidation_confidence()

    def _identify_definition_conflicts(self) -> list[dict[str, Any]]:
        """Identify columns with conflicting definitions."""
        conflicts = []

        for column_name, column in self.columns.items():
            # Check for conflicting definitions
            definitions = []
            if column.definition:
                definitions.append(("direct", column.definition))
            if column.pydantic_description:
                definitions.append(("pydantic", column.pydantic_description))
            if column.docstring_references:
                definitions.extend(
                    ("docstring", doc_ref) for doc_ref in column.docstring_references
                )

            # If we have multiple different definitions, it's a conflict
            if len({def_text for _, def_text in definitions}) > 1:
                conflicts.append(
                    {
                        "column_name": column_name,
                        "definitions": definitions,
                        "confidence_score": column.confidence_score,
                        "sources": len(definitions),
                    },
                )

        return conflicts

    def _resolve_definition_conflict(self, conflict: dict[str, Any]) -> None:
        """Resolve a specific definition conflict using priority rules."""
        column_name = conflict["column_name"]
        definitions = conflict["definitions"]

        # Priority order: pydantic > docstring > direct
        priority_order = ["pydantic", "docstring", "direct"]

        # Find the highest priority definition
        best_definition = None
        best_source = None

        for source_type in priority_order:
            for source, definition in definitions:
                if source == source_type:
                    best_definition = definition
                    best_source = source
                    break
            if best_definition:
                break

        if best_definition and column_name in self.columns:
            # Update the column with the resolved definition
            column = self.columns[column_name]
            column.definition = best_definition

            # Boost confidence for resolved conflicts
            if best_source == "pydantic":
                column.confidence_score = min(1.0, column.confidence_score + 0.2)
            elif best_source == "docstring":
                column.confidence_score = min(1.0, column.confidence_score + 0.1)

            # Add conflict resolution metadata
            column.line_references.append(
                {
                    "type": "conflict_resolution",
                    "resolved_source": best_source,
                    "total_sources": len(definitions),
                    "confidence_boost": 0.2 if best_source == "pydantic" else 0.1,
                },
            )

    def _consolidate_aliases(self) -> None:
        """Consolidate and deduplicate column aliases."""
        for column_name, column in self.columns.items():
            # Remove duplicate aliases
            unique_aliases = list(set(column.aliases))

            # Remove aliases that are the same as the column name
            unique_aliases = [alias for alias in unique_aliases if alias != column_name]

            # Sort for consistency
            unique_aliases.sort()

            column.aliases = unique_aliases

    def _merge_similar_columns(self) -> None:
        """Merge columns that are very similar with high confidence."""
        columns_to_merge = []
        processed = set()

        for column_name, column in self.columns.items():
            if column_name in processed:
                continue

            # Find similar columns based on name similarity and aliases
            similar_columns = []

            for other_name, other_column in self.columns.items():
                if other_name == column_name or other_name in processed:
                    continue

                # Check if columns are similar enough to merge
                if self._are_columns_similar(
                    column,
                    other_column,
                    column_name,
                    other_name,
                ):
                    similar_columns.append((other_name, other_column))

            if similar_columns:
                # Create merge group
                merge_group = {
                    "primary": (column_name, column),
                    "similar": similar_columns,
                }
                columns_to_merge.append(merge_group)

                # Mark as processed
                processed.add(column_name)
                for other_name, _ in similar_columns:
                    processed.add(other_name)

        # Execute merges
        for merge_group in columns_to_merge:
            self._execute_column_merge(merge_group)

    def _are_columns_similar(
        self,
        col1: ColumnKnowledge,
        col2: ColumnKnowledge,
        name1: str,
        name2: str,
    ) -> bool:
        """Determine if two columns are similar enough to merge."""
        # Check name similarity (edit distance)
        name_similarity = self._calculate_name_similarity(name1, name2)

        # Check if one is an alias of the other
        is_alias = name2 in col1.aliases or name1 in col2.aliases

        # Check definition similarity if both have definitions
        definition_similarity = 0.0
        if col1.definition and col2.definition:
            definition_similarity = self._calculate_text_similarity(
                col1.definition,
                col2.definition,
            )

        # Merge criteria: high name similarity OR alias relationship OR high definition similarity
        return (
            name_similarity > NAME_SIMILARITY_THRESHOLD
            or is_alias
            or definition_similarity > DEFINITION_SIMILARITY_THRESHOLD
        )

    def _calculate_name_similarity(self, name1: str, name2: str) -> float:
        """Calculate similarity between two column names."""
        # Simple edit distance based similarity
        max_len = max(len(name1), len(name2))
        if max_len == 0:
            return 1.0

        # Calculate common prefix/suffix and substring matches
        common_chars = set(name1.lower()) & set(name2.lower())
        char_similarity = len(common_chars) / max(
            len(set(name1.lower())),
            len(set(name2.lower())),
        )

        # Check for common patterns
        name1_lower = name1.lower()
        name2_lower = name2.lower()

        if name1_lower in name2_lower or name2_lower in name1_lower:
            return 0.9

        return char_similarity

    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two text strings."""
        # Simple word-based similarity
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _execute_column_merge(self, merge_group: dict[str, Any]) -> None:
        """Execute the merging of similar columns."""
        primary_name, primary_column = merge_group["primary"]
        similar_columns = merge_group["similar"]

        # Merge information from similar columns into primary
        for other_name, other_column in similar_columns:
            # Merge aliases
            primary_column.aliases.extend(other_column.aliases)
            if other_name not in primary_column.aliases:
                primary_column.aliases.append(other_name)

            # Merge comments and references
            primary_column.inline_comments.extend(other_column.inline_comments)
            primary_column.docstring_references.extend(
                other_column.docstring_references,
            )
            primary_column.line_references.extend(other_column.line_references)

            # Use better definition if available
            if other_column.definition and not primary_column.definition:
                primary_column.definition = other_column.definition
            elif (
                other_column.pydantic_description
                and not primary_column.pydantic_description
            ):
                primary_column.pydantic_description = other_column.pydantic_description

            # Boost confidence for merged columns
            primary_column.confidence_score = min(
                1.0,
                primary_column.confidence_score + 0.1,
            )

            # Remove the merged column
            del self.columns[other_name]

        # Clean up aliases after merge
        primary_column.aliases = list(set(primary_column.aliases))
        primary_column.aliases = [
            alias for alias in primary_column.aliases if alias != primary_name
        ]
        primary_column.aliases.sort()

    def _update_consolidation_confidence(self) -> None:
        """Update confidence scores based on consolidation results."""
        for column in self.columns.values():
            # Boost confidence for columns with multiple information sources
            source_count = 0
            if column.definition:
                source_count += 1
            if column.pydantic_description:
                source_count += 1
            if column.docstring_references:
                source_count += len(column.docstring_references)
            if column.inline_comments:
                source_count += len(column.inline_comments)

            # Apply consolidation bonus
            if source_count > 1:
                consolidation_bonus = min(0.3, source_count * 0.05)
                column.confidence_score = min(
                    1.0,
                    column.confidence_score + consolidation_bonus,
                )

    def get_conflict_resolution_summary(self) -> dict[str, Any]:
        """Generate summary of conflict resolution and consolidation."""
        # Count columns by confidence level
        high_confidence = [
            col
            for col in self.columns.values()
            if col.confidence_score >= CONFLICT_HIGH_CONFIDENCE_THRESHOLD
        ]
        medium_confidence = [
            col
            for col in self.columns.values()
            if CONFLICT_MEDIUM_CONFIDENCE_THRESHOLD
            <= col.confidence_score
            < CONFLICT_HIGH_CONFIDENCE_THRESHOLD
        ]
        low_confidence = [
            col
            for col in self.columns.values()
            if col.confidence_score < CONFLICT_MEDIUM_CONFIDENCE_THRESHOLD
        ]

        # Count consolidated information
        consolidated_columns = [
            col
            for col in self.columns.values()
            if len(col.aliases) > 1 or len(col.line_references) > 1
        ]

        # Count columns with multiple sources
        multi_source_columns = [
            col
            for col in self.columns.values()
            if sum(
                [
                    bool(col.definition),
                    bool(col.pydantic_description),
                    len(col.docstring_references) > 0,
                    len(col.inline_comments) > 0,
                ],
            )
            > 1
        ]

        return {
            "total_columns": len(self.columns),
            "high_confidence_columns": len(high_confidence),
            "medium_confidence_columns": len(medium_confidence),
            "low_confidence_columns": len(low_confidence),
            "consolidated_columns": len(consolidated_columns),
            "multi_source_columns": len(multi_source_columns),
            "average_confidence": round(
                sum(col.confidence_score for col in self.columns.values())
                / len(self.columns),
                3,
            )
            if self.columns
            else 0.0,
            "consolidation_rate": round(
                len(consolidated_columns) / len(self.columns),
                3,
            )
            if self.columns
            else 0.0,
        }

    def generate_cross_reference_summary(self) -> dict[str, Any]:
        """Generate a comprehensive summary of cross-references."""
        # Group cross-references by type
        by_relationship = {}
        for ref in self.cross_references:
            if ref.relationship not in by_relationship:
                by_relationship[ref.relationship] = []
            by_relationship[ref.relationship].append(ref)

        # Calculate statistics
        total_refs = len(self.cross_references)
        avg_confidence = (
            sum(ref.confidence for ref in self.cross_references) / total_refs
            if total_refs > 0
            else 0
        )

        # Find most connected columns
        connection_counts = {}
        for ref in self.cross_references:
            for name in [ref.source_name, ref.target_name]:
                if name in self.columns:
                    connection_counts[name] = connection_counts.get(name, 0) + 1

        most_connected = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        return {
            "total_cross_references": total_refs,
            "average_confidence": round(avg_confidence, 3),
            "references_by_type": {
                rel_type: len(refs) for rel_type, refs in by_relationship.items()
            },
            "most_connected_columns": [
                {"column": name, "connections": count} for name, count in most_connected
            ],
            "dependency_graph_size": len(getattr(self, "dependency_graph", {})),
        }
