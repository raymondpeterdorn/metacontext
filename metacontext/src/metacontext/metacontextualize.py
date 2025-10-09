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
from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
from metacontext.ai.handlers.llms.companion_provider import CompanionLLMProvider
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
    CodeAIEnrichment,
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
HandlerRegistry.register(GeospatialHandler)  # Check geospatial first
HandlerRegistry.register(CSVHandler)  # CSV handler is more generic
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
    args: MetacontextualizeArgs | None = None,
    *,
    output_path: str | Path | None = None,
    output_format: str = "yaml",
    include_llm_analysis: bool = True,
    ai_companion: bool = True,
    verbose: bool = False,
    scan_codebase: bool = True,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_temperature: float | None = None,
    llm_max_retries: int | None = None,
) -> Path:
    """Generate intelligent metacontext with revolutionary two-tier architecture.

    This function supports both legacy and simplified interfaces for backward compatibility:

    Legacy interface (backward compatible):
        metacontextualize(data_object, file_path, MetacontextualizeArgs(...))

    Simplified interface (new):
        metacontextualize(
            data_object,
            file_path,
            output_format="yaml",
            llm_provider="gemini",
            llm_api_key="...",
            include_llm_analysis=True
        )

    Args:
        data_object: The data object to analyze
        file_path: Path to the file being analyzed
        args: (Legacy) Dataclass containing all optional arguments. If provided, other
              keyword arguments are ignored for backward compatibility.
        output_path: Path for output file (if None, auto-generated)
        output_format: Output format ("yaml", "json", etc.)
        include_llm_analysis: Enable AI-powered analysis
        ai_companion: Enable AI companion mode (default: True). When True, uses
                      clipboard-based companion workflow. When False, uses API mode.
                      If llm_api_key is provided, API mode takes precedence.
        verbose: Enable verbose logging
        scan_codebase: Enable codebase context scanning
        llm_provider: LLM provider ("gemini", "openai", etc.)
        llm_model: Specific model to use
        llm_api_key: API key for LLM provider. When provided, forces API mode even
                     if ai_companion=True.
        llm_temperature: Temperature for LLM generation (0.0-1.0)
        llm_max_retries: Max retry attempts for LLM calls

    Returns:
        Path to the generated metacontext file

    Architecture:
        - Tier 1: Deterministic metadata (always succeeds)
        - Tier 2: AI enrichment (best effort, graceful degradation)
        - Schema-first LLM integration with automatic validation
        - Universal file format support through handler registry

    """
    file_path = Path(file_path)

    # Handle backward compatibility vs. new simplified interface
    if args is not None:
        # Legacy interface - use provided args as-is
        effective_args = args
    else:
        # New simplified interface - convert parameters to MetacontextualizeArgs
        config = {"scan_codebase": scan_codebase}

        # Add LLM configuration if provided
        if llm_provider is not None:
            config["llm_provider"] = llm_provider
        if llm_model is not None:
            config["llm_model"] = llm_model
        if llm_api_key is not None:
            config["llm_api_key"] = llm_api_key
        if llm_temperature is not None:
            config["llm_temperature"] = llm_temperature
        if llm_max_retries is not None:
            config["llm_max_retries"] = llm_max_retries

        # Create MetacontextualizeArgs from simplified parameters
        effective_args = MetacontextualizeArgs(
            output_path=output_path,
            config=config,
            include_llm_analysis=include_llm_analysis,
            output_format=output_format,
            verbose=verbose,
        )

    # Merge centralized config with runtime arguments
    merged_config = _merge_config_with_args(effective_args)

    start_time = time.time()

    # Generate output path
    if effective_args.output_path is None:
        output_path = _generate_output_path(file_path, effective_args.output_format)
    else:
        output_path = Path(effective_args.output_path)

    logger.info("\n🚀 METACONTEXT v0.3.0")
    logger.info("📁 File: %s", file_path.name)
    logger.info("💾 Output: %s", output_path.name)
    logger.info("=" * 60)


    # Always perform universal file inspection for baseline metadata
    file_inspector = FileInspector()
    universal_metadata = file_inspector.inspect(file_path)

    # Initialize AI companion if requested
    actual_ai_companion = None
    if include_llm_analysis:
        # Simple logic: if llm_api_key is provided (as parameter), always use API mode
        # Otherwise, use companion mode unless explicitly ai_companion=False
        if llm_api_key is not None:
            if effective_args.verbose:
                logger.info("🔑 LLM API key provided - using API mode")
            actual_ai_companion = None  # Use API mode
        elif ai_companion is False:
            if effective_args.verbose:
                logger.info("� AI companion explicitly disabled - using API mode")
            actual_ai_companion = None  # Use API mode
        else:
            try:
                # Check if companion provider is available and create unified provider
                if ProviderRegistry.is_registered("companion"):
                    companion_provider = CompanionLLMProvider()
                    if companion_provider.is_available():
                        actual_ai_companion = companion_provider
                    else:
                        actual_ai_companion = None
                else:
                    actual_ai_companion = None
            except (ImportError, ValueError, AttributeError) as e:
                if effective_args.verbose:
                    logger.warning("⚠️  Error initializing AI companion: %s, falling back to API mode", e)
                actual_ai_companion = None

    context = _generate_context(
        data_object=data_object,
        file_path=file_path,
        config=merged_config,
        include_llm_analysis=effective_args.include_llm_analysis,
        verbose=effective_args.verbose,
        universal_metadata=universal_metadata,
        ai_companion=actual_ai_companion,
    )

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Write output in specified format
    write_start = time.time()

    write_output(context, output_path, effective_args.output_format)

    logger.info("\n✅ Metacontext generated: %s", output_path)

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

        # Use enhanced provider manager for intelligent provider selection
        provider_kwargs = {}
        model = llm_config.get("model")
        if model:
            provider_kwargs["model"] = model
        api_key = llm_config.get("api_key")
        if api_key:
            provider_kwargs["api_key"] = api_key
        temperature = llm_config.get("temperature")
        if temperature:
            provider_kwargs["temperature"] = temperature

        llm_handler = ProviderManager.get_best_available_provider(
            preferred_provider=llm_config.get("provider"),
            **provider_kwargs,
        )
        if llm_handler.is_available():
            provider_info = llm_handler.get_provider_info()
            return llm_handler
        logger.warning(
            "⚠️  LLM configuration invalid - continuing with deterministic analysis only",
        )
    except (ValueError, ImportError) as e:
        logger.warning("⚠️  LLM initialization failed: %s", e)
        logger.info("   Continuing with deterministic analysis only")
    return None


def _convert_codebase_context_to_schema(
    raw_context: dict[str, Any],
    file_path: Path,
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
                related_file_path=ref_info["file"],
                relationship_type="references_target",
                relevance_score=0.9,  # High confidence for direct references
                relationship_evidence=f"Found {ref_info['reference_count']} references in {ref_info['file_type']} file",
            )
            related_files.append(relationship)

        # Convert data dependencies to FileRelationship objects
        for dep_info in cross_refs.get("data_dependencies", []):
            relationship = FileRelationship(
                related_file_path=dep_info["file_path"],
                relationship_type="depends_on",
                relevance_score=0.8,
                relationship_evidence=f"Referenced on line {dep_info['line_number']}: {dep_info['line_content'][:50]}",
            )
            related_files.append(relationship)

        if related_files:
            file_relationships = CodebaseRelationships(
                file_path=str(file_path),
                related_files=related_files,
            )

    # Extract semantic knowledge if available
    semantic_knowledge = raw_context.get("semantic_knowledge")
    ai_enrichment = None

    # Extract the nested semantic knowledge if it exists
    nested_semantic_knowledge = (
        semantic_knowledge.get("semantic_knowledge") if semantic_knowledge else None
    )

    if nested_semantic_knowledge and nested_semantic_knowledge.get("knowledge_graph"):
        # Convert the SemanticKnowledgeGraph object to a dictionary
        knowledge_graph = nested_semantic_knowledge.get("knowledge_graph")

        # Create a dictionary that preserves the structure the tabular handler expects
        semantic_knowledge_dict = {
            "columns": {},
            "summary": nested_semantic_knowledge.get("column_summary", {}),
            "cross_references": nested_semantic_knowledge.get(
                "cross_reference_summary",
                {},
            ),
            "total_columns": nested_semantic_knowledge.get(
                "total_columns_discovered",
                0,
            ),
        }

        # Extract column knowledge from the knowledge graph
        if isinstance(knowledge_graph, dict) and "columns" in knowledge_graph:
            for col_name, col_knowledge in knowledge_graph["columns"].items():
                # Extract the description from the pydantic_definition
                pydantic_def = col_knowledge.get("pydantic_definition", {})
                description = pydantic_def.get("description") or col_knowledge.get(
                    "inferred_meaning",
                )

                semantic_knowledge_dict["columns"][col_name] = {
                    "pydantic_description": description,
                    "definition": col_knowledge.get("inferred_meaning"),
                    "aliases": col_knowledge.get("aliases", []),
                    "confidence_score": 1.0 if description else 0.5,
                    "source_files": [pydantic_def.get("file")]
                    if pydantic_def.get("file")
                    else [],
                }

        ai_enrichment = CodeAIEnrichment(
            semantic_knowledge=semantic_knowledge_dict,
        )

    # Create the CodebaseContext with our structured data
    return CodebaseContext(
        ai_enrichment=ai_enrichment,
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
        raw_context = scan_codebase_context(file_path)
    except OSError as e:
        logger.warning("⚠️  Codebase scanning failed: %s", e)
        msg = f"Failed to scan codebase for context around {file_path.name}"
        raise RuntimeError(msg) from e
    else:
        return _convert_codebase_context_to_schema(raw_context, file_path)


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


def _generate_companion_context(
    file_path: Path,
    data_object: object,
    handler: object,
    ai_companion: object,
    *,
    verbose: bool = False,
) -> dict:
    """Generate context using companion workflow with unified schema wrapper.

    TASK 1 FIX: Ensure companion outputs use the same schema construction
    as API mode, wrapping handler responses in complete Metacontext schema.
    """
    try:

        if not hasattr(handler, "generate_context"):
            # Fallback if handler doesn't support generate_context
            return {
                "error": "Handler does not support generate_context",
                "handler_type": handler.__class__.__name__,
            }

        # Use composition workflow to combine base handler + extensions
        file_specific_context = HandlerRegistry.execute_composition_workflow(
            file_path=file_path,
            base_handler=handler,
            data_object=data_object,
            ai_companion=ai_companion,
        )

        # TASK 1 FIX: Use same schema builder as API mode
        # Create the base metacontext schema with core fields
        base_context = create_base_metacontext(
            filename=file_path.name,
            file_purpose="Generated file with two-tier architecture",
            project_context_summary=f"Analysis with {handler.__class__.__name__}",
        )

        # Add companion-specific generation info
        base_context.generation_info.generation_method = "explicit_function"
        base_context.generation_info.function_call = "metacontext.metacontextualize()"

        # Add handler-specific context to base schema (same as API mode)
        for key, value in file_specific_context.items():
            if hasattr(base_context, key):
                setattr(base_context, key, value)

        # Set confidence assessment
        overall_conf = _assess_overall_confidence(
            file_specific_context,
            has_llm=True,  # Companion is considered an LLM mode
        )
        base_context.confidence_assessment = ConfidenceAssessment(
            overall=ConfidenceLevel(overall_conf),
        )

        # Return in same format as API mode with custom serialization for geometry objects
        try:
            return base_context.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                serialize_as_any=True,  # Allow custom serialization
            )
        except Exception as serialization_error:
            logger.warning(
                "❌ Serialization failed, attempting with fallback: %s",
                str(serialization_error),
            )
            # Fallback: Convert problematic objects to strings
            return base_context.model_dump(
                mode="json",
                by_alias=True,
                exclude_none=True,
                serialize_as_any=True,
                fallback=str,  # Convert non-serializable objects to strings
            )

    except Exception:
        logger.exception("❌ Handler companion workflow failed")
        logger.info("   Falling back to basic analysis")
        return _generate_fallback_context(data_object, file_path)


def _generate_context(
    data_object: object,
    file_path: Path,
    config: dict,
    *,
    include_llm_analysis: bool,
    verbose: bool = False,
    universal_metadata: dict[str, Any] | None = None,
    ai_companion: object | None = None,
) -> dict:
    """Generate complete metacontext using two-tier architecture."""
    try:
        handler = HandlerRegistry.get_handler(file_path, data_object)
        if not handler:
            logger.warning("⚠️  No specific handler found - using fallback analysis")
            return _generate_fallback_context(data_object, file_path)

        # Initialize LLM handler if needed
        llm_start = time.time()
        llm_handler = _initialize_llm_handler(config) if include_llm_analysis else None
        if llm_handler and hasattr(handler, "llm_handler"):
            handler.llm_handler = llm_handler

        scan_start = time.time()
        codebase_context = _scan_codebase(config, file_path)
        # Provide codebase context to LLM handler if available
        if llm_handler and codebase_context:
            llm_handler.codebase_context = codebase_context

        # Generate file-specific context
        try:
            # Check if we're in companion mode
            if ai_companion is not None:

                # TASK 3 FIX: Provide codebase context to companion mode
                if codebase_context:
                    ai_companion.codebase_context = codebase_context

                # Route to companion workflow with the existing ai_companion
                return _generate_companion_context(
                    file_path=file_path,
                    data_object=data_object,
                    handler=handler,
                    ai_companion=ai_companion,
                    verbose=verbose,
                )

            if hasattr(handler, "generate_context"):
                # Use composition workflow to combine base handler + extensions
                file_specific_context = HandlerRegistry.execute_composition_workflow(
                    file_path=file_path,
                    base_handler=handler,
                    data_object=data_object,
                    ai_companion=llm_handler,
                )
            else:
                file_specific_context = {
                    "error": "Handler does not support generate_context",
                }

            # Add universal file metadata if available
            if universal_metadata and "statistics" in universal_metadata:
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

            overall_conf = _assess_overall_confidence(
                file_specific_context,
                has_llm=llm_handler is not None,
            )
            base_context.confidence_assessment = ConfidenceAssessment(
                overall=ConfidenceLevel(overall_conf),
            )

            return base_context.model_dump(
                mode="json", by_alias=True, exclude_none=True,
            )

        except Exception:
            logger.exception("❌ API analysis failed")
            raise  # Re-raise to be caught by outer try-except

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
