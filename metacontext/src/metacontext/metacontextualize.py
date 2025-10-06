"""Main API for metacontext generation with file handler routing.

This module provides the main entry point for metacontextualize() which
intelligently routes files to appropriate handlers and generates metadata.
It implements the architectural patterns documented in the architecture reference.

See:
- architecture_reference.ArchitecturalComponents.TWO_TIER_ARCHITECTURE
- architecture_reference.ArchitecturalComponents.SCHEMA_SYSTEM
- architecture_reference.ArchitecturalComponents.UNIVERSAL_FILE
"""

import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from metacontext.ai.codebase_scanner import scan_codebase_context
from metacontext.ai.handlers.core.provider_manager import ProviderManager
from metacontext.ai.handlers.llms.provider_interface import LLMProvider
from metacontext.core.config import get_config
from metacontext.core.output_utils import write_output
from metacontext.core.registry import HandlerRegistry
from metacontext.handlers.geospatial import GeospatialHandler
from metacontext.handlers.media import MediaHandler
from metacontext.handlers.model import ModelHandler
from metacontext.handlers.tabular import CSVHandler
from metacontext.inspectors.file_inspector import FileInspector
from metacontext.schemas.core.codebase import (
    CodebaseContext,
    CodebaseRelationships,
    FileRelationship,
)
from metacontext.schemas.core.core import (
    ConfidenceAssessment,
    TokenUsage,
    create_base_metacontext,
)
from metacontext.schemas.core.interfaces import ConfidenceLevel

logger = logging.getLogger(__name__)


# Register available handlers
HandlerRegistry.register(ModelHandler)
HandlerRegistry.register(CSVHandler)
HandlerRegistry.register(GeospatialHandler)
HandlerRegistry.register(MediaHandler)


@dataclass
class MetacontextualizeArgs:
    """Arguments for the metacontextualize function."""

    output_path: str | Path | None = None
    config: dict | None = None
    include_llm_analysis: bool = True
    output_format: str = "yaml"
    verbose: bool = False


def _merge_config_with_args(args: MetacontextualizeArgs) -> dict[str, Any]:
    """Merge centralized config with runtime arguments.

    Precedence: CLI args > config file > env vars > defaults

    Args:
        args: Runtime arguments from function call

    Returns:
        Merged configuration dictionary

    """
    # Get centralized config (already handles config file > env vars > defaults)
    central_config = get_config()

    # Start with centralized config values
    merged_config = {
        "llm_provider": central_config.llm.provider,
        "llm_model": central_config.llm.model,
        "llm_api_key": central_config.llm.api_key,
        "llm_temperature": central_config.llm.temperature,
        "llm_max_retries": central_config.llm.max_retries,
        "scan_codebase": central_config.scan_codebase,
        "scan_depth": central_config.scan_depth,
    }

    # Override with runtime args.config if provided (CLI args have highest precedence)
    if args.config:
        merged_config.update(args.config)

    # Override specific args that are provided via the MetacontextualizeArgs
    if hasattr(args, "output_format") and args.output_format != "yaml":
        # If output_format was explicitly set (not default), use it
        merged_config["output_format"] = args.output_format
    else:
        # Use config system default
        merged_config["output_format"] = central_config.output_format

    # Verbosity from args takes precedence
    merged_config["verbose"] = (
        args.verbose if hasattr(args, "verbose") else central_config.verbosity
    )

    return merged_config


def metacontextualize(
    data_object: object,
    file_path: str | Path,
    args: MetacontextualizeArgs,
) -> Path:
    """Generate intelligent metacontext with revolutionary two-tier architecture.

    Args:
        data_object: The data object to analyze
        file_path: Path to the file being analyzed
        args: Dataclass containing all optional arguments.

    Returns:
        Path to the generated metacontext file

    Architecture:
        - Tier 1: Deterministic metadata (always succeeds)
        - Tier 2: AI enrichment (best effort, graceful degradation)
        - Schema-first LLM integration with automatic validation
        - Universal file format support through handler registry

    """
    file_path = Path(file_path)

    # Merge centralized config with runtime arguments
    merged_config = _merge_config_with_args(args)

    start_time = time.time()

    # Generate output path
    if args.output_path is None:
        output_path = _generate_output_path(file_path, args.output_format)
    else:
        output_path = Path(args.output_path)

    logger.info("\n🚀 METACONTEXT v0.3.0 - Two-Tier Architecture")
    logger.info("📁 File: %s", file_path.name)
    logger.info("💾 Output: %s", output_path.name)
    logger.info("=" * 60)

    # Generate context using two-tier architecture
    logger.info("🔍 Starting file analysis...")
    analysis_start = time.time()

    # Always perform universal file inspection for baseline metadata
    file_inspector = FileInspector()
    universal_metadata = file_inspector.inspect(file_path)

    context = _generate_context(
        data_object=data_object,
        file_path=file_path,
        config=merged_config,
        include_llm_analysis=args.include_llm_analysis,
        verbose=args.verbose,
        universal_metadata=universal_metadata,
    )

    analysis_time = time.time() - analysis_start
    if args.verbose:
        logger.info("✓ Analysis completed in %.2f seconds", analysis_time)

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output in specified format
    logger.info("💾 Writing output file...")
    write_start = time.time()

    write_output(context, output_path, args.output_format)

    write_time = time.time() - write_start
    total_time = time.time() - start_time

    if args.verbose:
        logger.info("✓ Output written in %.2f seconds", write_time)

    logger.info("\n✅ Metacontext generated: %s", output_path)
    logger.info("   🏗️  Architecture: Two-tier (deterministic + AI enrichment)")
    logger.info("   📊 Schema: Core + Extensions pattern")
    logger.info(
        "   🤖 AI Analysis: %s",
        "Enabled" if args.include_llm_analysis else "Disabled",
    )
    if args.verbose:
        logger.info("   ⏱️  Total time: %.2f seconds", total_time)

    return output_path


def _initialize_llm_handler(config: dict) -> LLMProvider | None:
    """Initialize the LLM handler."""
    if not config.get("llm_api_key"):
        logger.warning("⚠️ No LLM API key provided - skipping AI enrichment")
        return None

    try:
        llm_config = {
            "provider": config.get("llm_provider", "gemini"),
            "model": config.get("llm_model"),  # No fallback - let provider decide
            "api_key": config.get("llm_api_key"),
            "temperature": config.get("llm_temperature", 0.1),
            "max_retries": config.get("llm_max_retries", 3),
        }

        logger.info("🔍 Initializing LLM provider with config: %s", llm_config)

        # Use enhanced provider manager for intelligent provider selection
        llm_handler = ProviderManager.get_best_available_provider(
            preferred_provider=llm_config.get("provider"),
            model=llm_config.get("model"),
            api_key=llm_config.get("api_key"),
            temperature=llm_config.get("temperature"),
        )
        if llm_handler.is_available():
            provider_info = llm_handler.get_provider_info()
            logger.info(
                "✓ LLM enabled: %s (%s)",
                provider_info["provider"],
                provider_info["model"],
            )
            return llm_handler
        logger.warning(
            "⚠️  LLM configuration invalid - continuing with deterministic analysis only",
        )
    except (ValueError, ImportError) as e:
        logger.warning("⚠️  LLM initialization failed: %s", e)
        logger.info("   Continuing with deterministic analysis only")
    return None


def _convert_codebase_context_to_schema(
    raw_context: dict[str, Any], file_path: Path
) -> CodebaseContext:
    """Convert raw codebase context to structured CodebaseContext schema."""
    # Extract cross-references if they exist
    cross_refs = raw_context.get("cross_references", {})
    file_relationships = None

    if cross_refs:
        related_files = []

        # Convert "referenced_by" to FileRelationship objects
        for ref_info in cross_refs.get("referenced_by", []):
            relationship = FileRelationship(
                file_path=ref_info["file"],
                relationship_type="references_target",
                description=f"References this file {ref_info['reference_count']} time(s)",
                confidence_score=0.9,  # High confidence for direct references
                evidence=f"Found {ref_info['reference_count']} references in {ref_info['file_type']} file",
            )
            related_files.append(relationship)

        # Convert data dependencies to FileRelationship objects
        for dep_info in cross_refs.get("data_dependencies", []):
            relationship = FileRelationship(
                file_path=dep_info["file_path"],
                relationship_type="depends_on",
                description=f"Data dependency ({dep_info['file_extension']} file)",
                confidence_score=0.8,
                evidence=f"Referenced on line {dep_info['line_number']}: {dep_info['line_content'][:50]}",
            )
            related_files.append(relationship)

        if related_files:
            file_relationships = CodebaseRelationships(
                file_path=str(file_path),
                related_files=related_files,
            )

    # Create the CodebaseContext with our structured data
    return CodebaseContext(
        file_relationships=file_relationships,
        context_summary=cross_refs.get("summary", "No cross-references found")
        if cross_refs
        else None,
    )


def _scan_codebase(config: dict, file_path: Path) -> CodebaseContext | None:
    """Scan the codebase for context."""
    if not config.get("scan_codebase", True):
        return None

    try:
        context = scan_codebase_context(file_path)
    except OSError as e:
        logger.warning("⚠️  Codebase scanning failed: %s", e)
        msg = f"Failed to scan codebase for context around {file_path.name}"
        raise RuntimeError(msg) from e
    else:
        logger.info("✓ Codebase context scanned")
        return context


def _generate_output_path(file_path: Path, output_format: str) -> Path:
    """Generate output path for metacontext file."""
    stem = file_path.stem

    format_mapping = {
        "yaml": f"{stem}_metacontext.yaml",
        "yaml_clean": f"{stem}_metacontext.yaml",
        "json": f"{stem}.metacontext.json",
        "metacontext": f"{stem}.metacontext",
        "mcntxt": f"{stem}.mcntxt",
    }

    filename = format_mapping.get(output_format, f"{stem}_metacontext.yaml")
    return file_path.parent / filename


def _generate_context(
    data_object: object,
    file_path: Path,
    config: dict,
    *,
    include_llm_analysis: bool,
    verbose: bool = False,
    universal_metadata: dict[str, Any] | None = None,
) -> dict:
    """Generate complete metacontext using two-tier architecture."""
    try:
        handler = HandlerRegistry.get_handler(file_path, data_object)
        if not handler:
            logger.warning("⚠️  No specific handler found - using fallback analysis")
            return _generate_fallback_context(data_object, file_path)

        logger.info(
            "✓ Using %s for %s files",
            handler.__class__.__name__,
            file_path.suffix,
        )

        # Initialize LLM handler if needed
        llm_start = time.time()
        llm_handler = _initialize_llm_handler(config) if include_llm_analysis else None
        if llm_handler:
            handler.llm_handler = llm_handler
            if verbose:
                llm_time = time.time() - llm_start
                logger.info("✓ LLM initialized in %.2f seconds", llm_time)

        # Scan codebase context
        if verbose:
            logger.info("🔍 Scanning codebase context...")
        scan_start = time.time()
        codebase_context = _scan_codebase(config, file_path)
        if verbose:
            scan_time = time.time() - scan_start
            logger.info("✓ Codebase scan completed in %.2f seconds", scan_time)

        # Generate file-specific context
        if verbose:
            logger.info("🔍 Analyzing file content...")
        context_start = time.time()
        file_specific_context = handler.generate_context(
            file_path=file_path,
            data_object=data_object,
            codebase_context=codebase_context,
            ai_companion=None,
        )
        if verbose:
            context_time = time.time() - context_start
            logger.info("✓ File analysis completed in %.2f seconds", context_time)

        # Add universal file metadata if available
        if universal_metadata and "statistics" in universal_metadata:
            if verbose:
                logger.info("📊 Adding enhanced statistics to context")
            # Add statistics to the existing data structure if it exists
            if "data_structure" in file_specific_context:
                data_struct = file_specific_context["data_structure"]
                if (
                    hasattr(data_struct, "deterministic_metadata")
                    and data_struct.deterministic_metadata
                ):
                    # Add as dict to deterministic metadata (it's flexible)
                    metadata_dict = data_struct.deterministic_metadata
                    if isinstance(metadata_dict, dict):
                        metadata_dict["file_statistics"] = universal_metadata[
                            "statistics"
                        ]
                        if "schema" in universal_metadata:
                            metadata_dict["enhanced_schema"] = universal_metadata[
                                "schema"
                            ]

        base_context = create_base_metacontext(
            filename=file_path.name,
            file_purpose="Generated file with two-tier architecture",
            project_context_summary=f"Analysis with {handler.__class__.__name__}",
        )

        if llm_handler:
            token_usage = llm_handler.get_token_usage()
            base_context.generation_info.token_usage = TokenUsage(**token_usage)
            if token_usage["total_api_calls"] > 0:
                logger.info(
                    "🔢 Token Usage: %s total (%s API calls)",
                    token_usage["total_tokens"],
                    token_usage["total_api_calls"],
                )

        for key, value in file_specific_context.items():
            if hasattr(base_context, key):
                setattr(base_context, key, value)

        # NOTE: codebase_context removed from output as it provides no valuable information
        # and wastes space in the metacontext YAML files. The context is still available
        # internally for LLM processing but not included in the final output.

        overall_conf = _assess_overall_confidence(
            file_specific_context,
            has_llm=llm_handler is not None,
        )
        base_context.confidence_assessment = ConfidenceAssessment(
            overall=ConfidenceLevel(overall_conf),
        )

        logger.info("✅ Two-tier context generation complete")
        return base_context.model_dump(mode="json", by_alias=True, exclude_none=True)

    except Exception:
        logger.exception("❌ Handler system failed")
        logger.info("   Falling back to basic analysis")
        return _generate_fallback_context(data_object, file_path)


def _assess_overall_confidence(file_specific_context: dict, *, has_llm: bool) -> str:
    """Assess overall confidence in the generated context."""
    if not has_llm:
        return "MEDIUM"  # Deterministic only

    # Look for AI confidence indicators in extensions
    ai_confidences = []
    for context_data in file_specific_context.values():
        if isinstance(context_data, dict):
            ai_enrichment = context_data.get("ai_enrichment", {})
            if isinstance(ai_enrichment, dict):
                ai_confidence = ai_enrichment.get("ai_confidence")
                if ai_confidence:
                    ai_confidences.append(ai_confidence)

    if not ai_confidences:
        return "MEDIUM"

    # Simple confidence aggregation
    if all(conf == "HIGH" for conf in ai_confidences):
        return "HIGH"
    if any(conf == "LOW" for conf in ai_confidences):
        return "LOW"
    return "MEDIUM"


def _generate_fallback_context(data_object: object, file_path: Path) -> dict:
    """Generate basic fallback context when handlers fail."""
    # Get codebase context even in fallback mode
    # NOTE: codebase_context collection removed since it's not included in output
    # but could be re-enabled if needed for internal processing

    return {
        "metacontext_version": "0.3.0",
        "generation_info": {
            "generated_at": datetime.now(UTC).isoformat(),
            "generation_method": "fallback",
            "function_call": "metacontext.metacontextualize()",
        },
        "file_info": {
            "filename": file_path.name,
            "extension": file_path.suffix,
            "absolute_path": str(file_path.absolute()),
            "directory": str(file_path.parent),
        },
        "system_info": {"working_directory": str(Path.cwd())},
        "data_analysis": {
            "type": "generic_object",
            "python_type": type(data_object).__name__,
            "note": "Fallback analysis - specialized handler not available",
        },
        "confidence_assessment": {"overall": "LOW"},
    }

    # NOTE: codebase_context removed from output as it provides no valuable information
    # and wastes space in the metacontext YAML files. The context is still available
    # internally for LLM processing but not included in the final output.
