"""Phase 6: LLM-Optimized Output Generation.

This module creates specialized output formats optimized for different LLM contexts
and use cases, leveraging the semantic knowledge graph built in Phase 5.
"""

from dataclasses import dataclass
from enum import Enum

from .knowledge_graph import SemanticKnowledgeGraph

# Constants
HIGH_CONFIDENCE_THRESHOLD = 0.8
MEDIUM_CONFIDENCE_THRESHOLD = 0.5


class OutputFormat(Enum):
    """Supported LLM-optimized output formats."""

    CONTEXT_INJECTION = "context_injection"
    CODE_DOCUMENTATION = "code_documentation"
    API_SPECIFICATION = "api_specification"
    PROMPT_TEMPLATE = "prompt_template"
    STRUCTURED_SUMMARY = "structured_summary"
    DEBUGGING_GUIDE = "debugging_guide"


class LLMContext(Enum):
    """Target LLM context types."""

    GENERAL_PURPOSE = "general_purpose"
    CODE_COMPLETION = "code_completion"
    DOCUMENTATION = "documentation"
    DEBUGGING = "debugging"
    API_DESIGN = "api_design"
    DATA_ANALYSIS = "data_analysis"


@dataclass
class OutputConfiguration:
    """Configuration for LLM-optimized output generation."""

    format_type: OutputFormat
    llm_context: LLMContext
    include_confidence_scores: bool = True
    include_cross_references: bool = True
    include_examples: bool = True
    max_tokens: int | None = None
    prioritize_high_confidence: bool = True
    include_metadata: bool = True


class LLMOptimizedOutputGenerator:
    """Generates LLM-optimized outputs from semantic knowledge graphs."""

    def __init__(self, knowledge_graph: SemanticKnowledgeGraph) -> None:
        """Initialize the output generator."""
        self.knowledge_graph = knowledge_graph
        self.output_templates = self._load_output_templates()

    def generate_output(self, config: OutputConfiguration) -> str:
        """Generate LLM-optimized output based on configuration."""
        if config.format_type == OutputFormat.CONTEXT_INJECTION:
            return self._generate_context_injection(config)
        if config.format_type == OutputFormat.CODE_DOCUMENTATION:
            return self._generate_code_documentation(config)
        if config.format_type == OutputFormat.API_SPECIFICATION:
            return self._generate_api_specification(config)
        if config.format_type == OutputFormat.PROMPT_TEMPLATE:
            return self._generate_prompt_template(config)
        if config.format_type == OutputFormat.STRUCTURED_SUMMARY:
            return self._generate_structured_summary(config)
        if config.format_type == OutputFormat.DEBUGGING_GUIDE:
            return self._generate_debugging_guide(config)

        msg = f"Unsupported output format: {config.format_type}"
        raise ValueError(msg)

    def _generate_context_injection(self, config: OutputConfiguration) -> str:
        """Generate context optimized for injection into LLM prompts."""
        sections = []

        # Header
        sections.append("# Codebase Context Information")
        sections.append("")

        # High-confidence columns first
        high_conf_columns = self.knowledge_graph.get_high_confidence_columns()
        if high_conf_columns and config.prioritize_high_confidence:
            sections.append("## High-Confidence Data Elements")
            for col_name in high_conf_columns[:10]:  # Top 10
                col = self.knowledge_graph.columns.get(col_name)
                if col:
                    sections.append(
                        f"- **{col_name}**: {col.definition or 'Data element'}",
                    )
                    if col.aliases and config.include_cross_references:
                        sections.append(f"  - Aliases: {', '.join(col.aliases[:3])}")
                    if config.include_confidence_scores:
                        sections.append(f"  - Confidence: {col.confidence_score:.2f}")
            sections.append("")

        # Cross-references
        if config.include_cross_references and self.knowledge_graph.cross_references:
            sections.append("## Data Relationships")
            top_refs = sorted(
                self.knowledge_graph.cross_references,
                key=lambda x: x.confidence,
                reverse=True,
            )[:5]

            for ref in top_refs:
                sections.append(
                    f"- {ref.source_name} {ref.relationship} {ref.target_name}",
                )
                if config.include_confidence_scores:
                    sections.append(f"  - Confidence: {ref.confidence:.2f}")
            sections.append("")

        # Metadata
        if config.include_metadata:
            sections.append("## Codebase Statistics")
            sections.append(
                f"- Total data elements: {len(self.knowledge_graph.columns)}",
            )
            sections.append(
                f"- Cross-references: {len(self.knowledge_graph.cross_references)}",
            )
            sections.append(f"- High-confidence elements: {len(high_conf_columns)}")

        return "\n".join(sections)

    def _generate_code_documentation(self, config: OutputConfiguration) -> str:
        """Generate comprehensive code documentation."""
        sections = []

        sections.append("# Code Documentation")
        sections.append("")
        sections.append(
            "This documentation was automatically generated from codebase analysis.",
        )
        sections.append("")

        # Data Model Overview
        sections.append("## Data Model Overview")
        sections.append("")

        # Group columns by confidence and type
        high_conf = [
            col
            for col in self.knowledge_graph.columns.values()
            if col.confidence_score >= 0.8
        ]
        medium_conf = [
            col
            for col in self.knowledge_graph.columns.values()
            if 0.5 <= col.confidence_score < 0.8
        ]

        if high_conf:
            sections.append("### Core Data Elements")
            for col in sorted(
                high_conf, key=lambda x: x.confidence_score, reverse=True,
            )[:15]:
                col_name = next(
                    (
                        name
                        for name, c in self.knowledge_graph.columns.items()
                        if c == col
                    ),
                    "unknown",
                )
                sections.append(f"#### {col_name}")
                if col.definition:
                    sections.append(f"{col.definition}")
                if col.pydantic_description:
                    sections.append(f"*Pydantic: {col.pydantic_description}*")
                if col.aliases:
                    sections.append(f"**Aliases:** {', '.join(col.aliases[:5])}")
                sections.append("")

        if medium_conf and config.llm_context != LLMContext.CODE_COMPLETION:
            sections.append("### Supporting Data Elements")
            for col in sorted(
                medium_conf, key=lambda x: x.confidence_score, reverse=True,
            )[:10]:
                col_name = next(
                    (
                        name
                        for name, c in self.knowledge_graph.columns.items()
                        if c == col
                    ),
                    "unknown",
                )
                sections.append(f"- **{col_name}**: {col.definition or 'Data element'}")
            sections.append("")

        # Cross-references
        if config.include_cross_references:
            sections.append("## Data Relationships")
            sections.append("")
            relationship_groups = {}
            for ref in self.knowledge_graph.cross_references:
                if ref.relationship not in relationship_groups:
                    relationship_groups[ref.relationship] = []
                relationship_groups[ref.relationship].append(ref)

            for rel_type, refs in relationship_groups.items():
                sections.append(
                    f"### {rel_type.replace('_', ' ').title()} Relationships",
                )
                for ref in sorted(refs, key=lambda x: x.confidence, reverse=True)[:5]:
                    sections.append(f"- `{ref.source_name}` → `{ref.target_name}`")
                sections.append("")

        return "\n".join(sections)

    def _generate_api_specification(self, config: OutputConfiguration) -> str:
        """Generate API specification documentation."""
        sections = []

        sections.append("# API Specification")
        sections.append("")

        # Find Pydantic models and high-confidence data elements
        pydantic_columns = [
            col
            for col in self.knowledge_graph.columns.values()
            if col.pydantic_description
        ]

        if pydantic_columns:
            sections.append("## Data Models")
            sections.append("")

            for col in sorted(
                pydantic_columns, key=lambda x: x.confidence_score, reverse=True,
            ):
                col_name = next(
                    (
                        name
                        for name, c in self.knowledge_graph.columns.items()
                        if c == col
                    ),
                    "unknown",
                )
                sections.append(f"### {col_name}")
                sections.append("**Type:** Data Model Field")
                sections.append(f"**Description:** {col.pydantic_description}")

                if col.definition and col.definition != col.pydantic_description:
                    sections.append(f"**Additional Context:** {col.definition}")

                if config.include_confidence_scores:
                    sections.append(f"**Confidence:** {col.confidence_score:.2f}")

                if col.aliases:
                    sections.append(f"**Alternative Names:** {', '.join(col.aliases)}")

                sections.append("")

        # High-confidence non-Pydantic columns
        other_high_conf = [
            col
            for col in self.knowledge_graph.columns.values()
            if col.confidence_score >= 0.8 and not col.pydantic_description
        ]

        if other_high_conf:
            sections.append("## Additional Data Elements")
            sections.append("")

            for col in sorted(
                other_high_conf, key=lambda x: x.confidence_score, reverse=True,
            )[:10]:
                col_name = next(
                    (
                        name
                        for name, c in self.knowledge_graph.columns.items()
                        if c == col
                    ),
                    "unknown",
                )
                sections.append(f"- **{col_name}**: {col.definition or 'Data element'}")
            sections.append("")

        return "\n".join(sections)

    def _generate_prompt_template(self, config: OutputConfiguration) -> str:
        """Generate a prompt template for LLM interactions."""
        template_parts = []

        if config.llm_context == LLMContext.CODE_COMPLETION:
            template_parts.append(
                "You are a code completion assistant. Use this context:",
            )
        elif config.llm_context == LLMContext.DEBUGGING:
            template_parts.append(
                "You are a debugging assistant. Here's the codebase context:",
            )
        elif config.llm_context == LLMContext.DATA_ANALYSIS:
            template_parts.append(
                "You are a data analysis assistant. Use this data context:",
            )
        else:
            template_parts.append(
                "You are a helpful coding assistant. Here's the codebase context:",
            )

        template_parts.append("")

        # Key data elements
        high_conf_columns = self.knowledge_graph.get_high_confidence_columns()
        if high_conf_columns:
            template_parts.append("Key Data Elements:")
            for col_name in high_conf_columns[:8]:
                col = self.knowledge_graph.columns.get(col_name)
                if col:
                    template_parts.append(
                        f"- {col_name}: {col.definition or 'Data element'}",
                    )

        template_parts.append("")
        template_parts.append("User Question: {user_question}")
        template_parts.append("")
        template_parts.append(
            "Please provide a helpful response using the above context.",
        )

        return "\n".join(template_parts)

    def _generate_structured_summary(self, config: OutputConfiguration) -> str:
        """Generate a structured summary of the codebase."""
        sections = []

        sections.append("# Codebase Analysis Summary")
        sections.append("")

        # Statistics
        total_columns = len(self.knowledge_graph.columns)
        high_conf = len(self.knowledge_graph.get_high_confidence_columns())
        cross_refs = len(self.knowledge_graph.cross_references)

        sections.append("## Overview")
        sections.append(f"- **Total Data Elements:** {total_columns}")
        sections.append(f"- **High-Confidence Elements:** {high_conf}")
        sections.append(f"- **Cross-References:** {cross_refs}")
        sections.append(
            f"- **Analysis Confidence:** {high_conf / total_columns * 100:.1f}%"
            if total_columns > 0
            else "- **Analysis Confidence:** 0%",
        )
        sections.append("")

        # Top elements by confidence
        all_columns = list(self.knowledge_graph.columns.items())
        top_columns = sorted(
            all_columns, key=lambda x: x[1].confidence_score, reverse=True,
        )[:10]

        sections.append("## Top Data Elements")
        for col_name, col in top_columns:
            sections.append(
                f"- **{col_name}** ({col.confidence_score:.2f}): {col.definition or 'Data element'}",
            )
        sections.append("")

        # Relationship summary
        if self.knowledge_graph.cross_references:
            rel_counts = {}
            for ref in self.knowledge_graph.cross_references:
                rel_counts[ref.relationship] = rel_counts.get(ref.relationship, 0) + 1

            sections.append("## Relationship Types")
            for rel_type, count in sorted(
                rel_counts.items(), key=lambda x: x[1], reverse=True,
            ):
                sections.append(f"- **{rel_type.replace('_', ' ').title()}:** {count}")
            sections.append("")

        return "\n".join(sections)

    def _generate_debugging_guide(self, config: OutputConfiguration) -> str:
        """Generate a debugging guide based on the codebase analysis."""
        sections = []

        sections.append("# Debugging Guide")
        sections.append("")
        sections.append(
            "This guide highlights key data elements and relationships for debugging.",
        )
        sections.append("")

        # Critical data elements
        high_conf_columns = self.knowledge_graph.get_high_confidence_columns()
        if high_conf_columns:
            sections.append("## Critical Data Elements")
            sections.append(
                "Monitor these high-confidence data elements during debugging:",
            )
            sections.append("")

            for col_name in high_conf_columns[:12]:
                col = self.knowledge_graph.columns.get(col_name)
                if col:
                    sections.append(f"### {col_name}")
                    sections.append(
                        f"**Description:** {col.definition or 'Data element'}",
                    )
                    if col.aliases:
                        sections.append(
                            f"**Also known as:** {', '.join(col.aliases[:3])}",
                        )
                    if col.source_file:
                        sections.append(f"**Source:** {col.source_file}")
                    sections.append("")

        # Data flow relationships
        if self.knowledge_graph.cross_references:
            transform_refs = [
                ref
                for ref in self.knowledge_graph.cross_references
                if ref.relationship == "transforms"
            ]
            derive_refs = [
                ref
                for ref in self.knowledge_graph.cross_references
                if ref.relationship == "derives_from"
            ]

            if transform_refs or derive_refs:
                sections.append("## Data Flow Analysis")
                sections.append("Trace data transformations and derivations:")
                sections.append("")

                for ref in sorted(
                    transform_refs + derive_refs,
                    key=lambda x: x.confidence,
                    reverse=True,
                )[:8]:
                    sections.append(
                        f"- `{ref.source_name}` → `{ref.target_name}` ({ref.relationship})",
                    )
                sections.append("")

        return "\n".join(sections)

    def _load_output_templates(self) -> dict[str, str]:
        """Load output templates for different formats."""
        # In a real implementation, these could be loaded from files
        return {
            "context_header": "# Codebase Context\n\n",
            "api_header": "# API Documentation\n\n",
            "debug_header": "# Debugging Guide\n\n",
        }

    def generate_multiple_formats(
        self,
        formats: list[OutputFormat],
        llm_context: LLMContext = LLMContext.GENERAL_PURPOSE,
    ) -> dict[str, str]:
        """Generate multiple output formats at once."""
        results = {}

        for format_type in formats:
            config = OutputConfiguration(
                format_type=format_type,
                llm_context=llm_context,
            )
            results[format_type.value] = self.generate_output(config)

        return results
