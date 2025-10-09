"""Handler integration helper for unified LLM pipeline.

This module provides reusable integration logic for all handlers to use the
unified pipeline where both API and companion modes follow identical workflow
steps B→H, with only step I (LLM Analysis) differing.

See:
- architecture_reference.ArchitecturalComponents.UNIFIED_PIPELINE
- architecture_reference.HandlerArchitecture.UNIFIED_LLM_INTEGRATION
"""

import logging
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


@runtime_checkable
class LLMProvider(Protocol):
    """Protocol for LLM providers in unified pipeline."""

    def is_available(self) -> bool:
        """Check if the LLM provider is available for use."""
        ...

    def is_companion_mode(self) -> bool:
        """Check if this is a companion provider (IDE-integrated mode)."""
        ...

    @property
    def execution_mode(self) -> str:
        """Get execution mode: 'api' or 'companion'."""
        ...


class UnifiedLLMIntegration:
    """Helper class for integrating handlers with unified LLM pipeline.

    This class provides common patterns for handler integration that eliminates
    code duplication between ModelHandler, CSVHandler, and other handlers.
    """

    @staticmethod
    def extract_semantic_knowledge(ai_companion: object) -> str:
        """Extract semantic knowledge from codebase context (Step D).

        Args:
            ai_companion: LLM provider instance that may contain codebase context

        Returns:
            Formatted semantic knowledge text or default message

        """
        semantic_knowledge_text = "No semantic knowledge extracted from codebase."

        if not (
            hasattr(ai_companion, "codebase_context") and ai_companion.codebase_context
        ):
            logger.info("🔍 DEBUG: No codebase context found on ai_companion")
            return semantic_knowledge_text

        logger.info("🔍 DEBUG: Codebase context found on ai_companion")

        try:
            # Check if we have semantic knowledge available
            if not (
                hasattr(ai_companion.codebase_context, "ai_enrichment")
                and ai_companion.codebase_context.ai_enrichment
                and hasattr(
                    ai_companion.codebase_context.ai_enrichment, "semantic_knowledge"
                )
            ):
                logger.info("🔍 DEBUG: No semantic knowledge found in ai_enrichment")
                return semantic_knowledge_text

            logger.info("🔍 DEBUG: Found semantic knowledge in ai_enrichment")
            semantic_knowledge = (
                ai_companion.codebase_context.ai_enrichment.semantic_knowledge
            )

            # Handle different semantic knowledge formats
            if hasattr(semantic_knowledge, "columns"):
                # Column-based semantic knowledge (for tabular data)
                return UnifiedLLMIntegration._format_column_knowledge(
                    semantic_knowledge
                )
            if hasattr(semantic_knowledge, "model_fields"):
                # Model field-based semantic knowledge (for model files)
                return UnifiedLLMIntegration._format_model_field_knowledge(
                    semantic_knowledge
                )
            if isinstance(semantic_knowledge, dict) and "columns" in semantic_knowledge:
                # Dictionary format column knowledge
                return UnifiedLLMIntegration._format_dict_column_knowledge(
                    semantic_knowledge
                )
            logger.info(
                "🔍 DEBUG: Semantic knowledge exists but has unknown format: %s",
                type(semantic_knowledge),
            )
            return semantic_knowledge_text

        except (AttributeError, KeyError, TypeError) as e:
            logger.warning("Error extracting semantic knowledge: %s", e)
            return semantic_knowledge_text

    @staticmethod
    def _format_column_knowledge(semantic_knowledge: Any) -> str:
        """Format column-based semantic knowledge."""
        logger.info(
            "🔍 DEBUG: Semantic knowledge has %d columns",
            len(semantic_knowledge.columns),
        )

        column_descriptions = []
        for col_name, col_info in semantic_knowledge.columns.items():
            logger.info(
                "🔍 DEBUG: Column %s: pydantic='%s', definition='%s'",
                col_name,
                col_info.pydantic_description,
                col_info.definition,
            )

            if col_info.pydantic_description:
                column_descriptions.append(
                    f"- {col_name}: {col_info.pydantic_description}"
                )
            elif col_info.definition:
                column_descriptions.append(f"- {col_name}: {col_info.definition}")

        if column_descriptions:
            formatted_knowledge = "Extracted column meanings:\n" + "\n".join(
                column_descriptions
            )
            logger.info(
                "🔍 DEBUG: Formatted semantic knowledge: %s", formatted_knowledge
            )
            return formatted_knowledge

        return "No semantic knowledge extracted from codebase."

    @staticmethod
    def _format_model_field_knowledge(semantic_knowledge: Any) -> str:
        """Format model field-based semantic knowledge."""
        logger.info(
            "🔍 DEBUG: Semantic knowledge has %d model fields",
            len(semantic_knowledge.model_fields),
        )

        field_descriptions = []
        for field_name, field_info in semantic_knowledge.model_fields.items():
            logger.info(
                "🔍 DEBUG: Field %s: pydantic='%s', definition='%s'",
                field_name,
                field_info.pydantic_description,
                field_info.definition,
            )

            if field_info.pydantic_description:
                field_descriptions.append(
                    f"- {field_name}: {field_info.pydantic_description}"
                )
            elif field_info.definition:
                field_descriptions.append(f"- {field_name}: {field_info.definition}")

        if field_descriptions:
            formatted_knowledge = "Semantic knowledge from codebase:\n" + "\n".join(
                field_descriptions
            )
            logger.info(
                "🔍 DEBUG: Using semantic knowledge: %s",
                formatted_knowledge[:200] + "...",
            )
            return formatted_knowledge

        return "No semantic knowledge extracted from codebase."

    @staticmethod
    def _format_dict_column_knowledge(semantic_knowledge: dict[str, Any]) -> str:
        """Format dictionary-based column knowledge."""
        columns = semantic_knowledge["columns"]
        logger.info(
            "🔍 DEBUG: Semantic knowledge is dict with %d columns",
            len(columns),
        )

        column_descriptions = []
        for col_name, col_info in columns.items():
            logger.info(
                "🔍 DEBUG: Dict column %s: pydantic='%s', definition='%s'",
                col_name,
                col_info.get("pydantic_description"),
                col_info.get("definition"),
            )

            if col_info.get("pydantic_description"):
                column_descriptions.append(
                    f"- {col_name}: {col_info['pydantic_description']}"
                )
            elif col_info.get("definition"):
                column_descriptions.append(f"- {col_name}: {col_info['definition']}")

        if column_descriptions:
            formatted_knowledge = "Extracted column meanings:\n" + "\n".join(
                column_descriptions
            )
            logger.info(
                "🔍 DEBUG: Formatted semantic knowledge: %s", formatted_knowledge
            )
            return formatted_knowledge

        return "No semantic knowledge extracted from codebase."

    @staticmethod
    def log_execution_mode(ai_companion: object) -> None:
        """Log which execution mode is being used (Step I routing).

        Args:
            ai_companion: LLM provider instance

        """
        if (
            hasattr(ai_companion, "is_companion_mode")
            and ai_companion.is_companion_mode()
        ):
            logger.info("🤖 Using companion mode for analysis")
        else:
            logger.info("🔗 Using API LLM provider for analysis")

    @staticmethod
    def execute_unified_analysis(
        deterministic_metadata: Any,
        file_path: Any,
        ai_companion: object,
        analysis_method: callable,
        codebase_context: dict[str, object] | None = None,
    ) -> Any:
        """Execute unified AI analysis with error handling and fallback logic.

        This is the main integration point that handlers can use to execute
        the unified pipeline without duplicating error handling logic.

        Args:
            deterministic_metadata: Results from deterministic analysis (Step B)
            file_path: Path to the file being analyzed
            ai_companion: LLM provider instance
            analysis_method: Handler's AI analysis method to call
            codebase_context: Optional codebase context

        Returns:
            AI analysis results or fallback enrichment on error

        """
        if not ai_companion:
            logger.info("i  No LLM handler provided - using fallback analysis")
            return None

        if not (hasattr(ai_companion, "is_available") and ai_companion.is_available()):
            logger.warning("AI companion not available - using fallback analysis")
            return None

        # Extract semantic knowledge (Step D)
        semantic_knowledge_text = UnifiedLLMIntegration.extract_semantic_knowledge(
            ai_companion
        )

        # Log execution mode (Step I routing)
        UnifiedLLMIntegration.log_execution_mode(ai_companion)

        try:
            # Execute the handler's specific AI analysis method (Steps E→I)
            return analysis_method(
                deterministic_metadata,
                file_path,
                codebase_context,
                semantic_knowledge_text,
            )
        except Exception as e:
            logger.exception("AI analysis failed, using fallback: %s", e)
            return None
