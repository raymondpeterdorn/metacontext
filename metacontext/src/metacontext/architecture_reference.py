"""Architecture reference documentation embedded in code.

This module serves as a reference point connecting the code implementation to its
corresponding technical documentation. It contains no functional code but provides
annotated documentation references that can be used to navigate between code and docs.

This helps maintain the connection between code and documentation, making it easier
to keep them in sync and understand the architectural principles behind the implementation.
"""

from dataclasses import dataclass


@dataclass
class ArchitectureReference:
    """Reference to a section in the technical architecture documentation."""

    section: str
    file: str = "technical_architecture.md"
    line_number: int | None = None
    description: str | None = None

    def __str__(self) -> str:
        """Return a string representation of the reference."""
        result = f"[{self.section}] in {self.file}"
        if self.line_number:
            result += f" (line {self.line_number})"
        if self.description:
            result += f": {self.description}"
        return result


class ArchitecturalComponents:
    """References to the main architectural components described in the documentation."""

    TWO_TIER_ARCHITECTURE = ArchitectureReference(
        section="Two-Tier Architecture",
        line_number=17,
        description="Guaranteed reliability - Deterministic metadata always available, "
        "AI enhancement when possible",
    )

    SCHEMA_FIRST_LLM = ArchitectureReference(
        section="Schema-First LLM Integration",
        line_number=20,
        description="Structured AI calls - Pydantic schemas drive prompts with "
        "automatic validation",
    )

    COST_TRACKING = ArchitectureReference(
        section="Comprehensive Cost Tracking",
        line_number=23,
        description="Production monitoring - Real-time token usage and cost estimation "
        "across providers",
    )

    HANDLER_SPECIALIZATION = ArchitectureReference(
        section="Handler Specialization",
        line_number=26,
        description="File-type expertise - Specialized handlers for models, tabular data, and more",
    )

    FILE_TYPE_HANDLER = ArchitectureReference(
        section="File Type Handler System",
        line_number=52,
        description="Auto-detects file types and routes to appropriate handlers",
    )

    SCHEMA_SYSTEM = ArchitectureReference(
        section="Core + Extensions Schema System",
        line_number=67,
        description="Modular metadata structure with optional file-type extensions",
    )

    TWO_TIER_METADATA = ArchitectureReference(
        section="Two-Tier Metadata Generation",
        line_number=83,
        description="Tier 1 (Deterministic) and Tier 2 (AI Enrichment) components",
    )

    SCHEMA_FIRST_PROMPT = ArchitectureReference(
        section="Schema-First Prompt Engineering",
        line_number=96,
        description="Uses Pydantic schema definitions to generate LLM prompts",
    )

    AI_COMPANION = ArchitectureReference(
        section="AI Companion Integration",
        line_number=110,
        description="Leverages available AI assistants for intelligent content generation",
    )

    UNIVERSAL_FILE = ArchitectureReference(
        section="Universal File Intelligence",
        line_number=117,
        description="File-type-agnostic metadata generation",
    )

    OUTPUT_HANDLER = ArchitectureReference(
        section="Automatic Output Handler",
        line_number=132,
        description="Takes the final, validated YAML data and writes it to a file",
    )


class ConfigurationArchitecture:
    """References to the configuration system architecture."""

    CONFIG_HIERARCHY = ArchitectureReference(
        section="Configuration System",
        file="implementation_guides.md",
        description="Configuration hierarchy: defaults, env vars, config files, runtime",
    )

    ENV_VARS = ArchitectureReference(
        section="Environment Variables",
        file="implementation_guides.md",
        description="Environment variables used for configuration",
    )

    CONFIG_FILES = ArchitectureReference(
        section="Configuration Files",
        file="implementation_guides.md",
        description="Configuration file formats and locations",
    )


class HandlerArchitecture:
    """References to the handler system architecture."""

    HANDLER_REGISTRY = ArchitectureReference(
        section="Handler Registry",
        description="Central registry for file type handlers",
    )

    HANDLER_INTERFACE = ArchitectureReference(
        section="Handler Interface",
        description="Common interface for all file type handlers",
    )

    MODEL_HANDLER = ArchitectureReference(
        section="Model Handler",
        description="Specialized handler for machine learning models",
    )

    TABULAR_HANDLER = ArchitectureReference(
        section="Tabular Handler",
        description="Specialized handler for tabular data (CSV, DataFrames)",
    )


class LLMArchitecture:
    """References to the LLM system architecture."""

    PROVIDER_INTERFACE = ArchitectureReference(
        section="LLM Provider Interface",
        description="Common interface for all LLM providers",
    )

    PROVIDER_REGISTRY = ArchitectureReference(
        section="Provider Registry",
        description="Central registry for LLM providers",
    )

    PROVIDER_FACTORY = ArchitectureReference(
        section="Provider Factory",
        description="Factory for creating LLM provider instances",
    )

    TOKEN_TRACKER = ArchitectureReference(
        section="Token Tracker",
        description="Tracks token usage and cost estimation",
    )


class SchemaArchitecture:
    """References to the schema system architecture."""

    CORE_SCHEMAS = ArchitectureReference(
        section="Core Schemas",
        description="Base schema components used across all file types",
    )

    EXTENSION_SCHEMAS = ArchitectureReference(
        section="Extension Schemas",
        description="File-type specific schema extensions",
    )

    CONFIDENCE_LEVELS = ArchitectureReference(
        section="Confidence Levels",
        description="Confidence assessment for AI-generated content",
    )
