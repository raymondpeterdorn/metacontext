"""Tabular data handler for CSV, Excel, Parquet files."""

import json
import logging
import re
from pathlib import Path
from typing import Any, ClassVar

import pandas as pd

from metacontext.ai.handlers.core.exceptions import LLMError, ValidationRetryError
from metacontext.ai.handlers.llms.prompt_constraints import (
    COMMON_FIELD_CONSTRAINTS,
    build_schema_constraints,
    calculate_response_limits,
)
from metacontext.ai.handlers.llms.provider_interface import parse_json_response
from metacontext.ai.prompts.prompt_loader import PromptLoader
from metacontext.handlers.base import BaseFileHandler, register_handler
from metacontext.schemas.extensions.tabular import (
    ColumnAIEnrichment,
    ColumnDeterministicInfo,
    ColumnInfo,
    DataAIEnrichment,
    DataDeterministicMetadata,
    DataStructure,
)

try:
    import magic
except ImportError:
    magic = None

logger = logging.getLogger(__name__)


@register_handler
class CSVHandler(BaseFileHandler):
    """Handler for CSV and tabular data files.

    Supports: CSV, Excel, Parquet, Feather files
    Extensions: data_structure
    """

    supported_extensions: ClassVar[list[str]] = [
        ".csv",
        ".xlsx",
        ".xls",
        ".parquet",
        ".feather",
    ]
    required_schema_extensions: ClassVar[list[str]] = ["data_structure"]

    def __init__(self) -> None:
        """Initialize the tabular handler."""
        # No prompt loader needed - using schema-first prompting

    def can_handle(self, file_path: Path, data_object: object | None = None) -> bool:
        """Check if this is a tabular data file."""
        if file_path.suffix.lower() in self.supported_extensions:
            return True

        # Check if data_object is a pandas DataFrame
        if data_object is not None:
            try:
                return isinstance(data_object, pd.DataFrame)
            except ImportError:
                pass

        return False

    def get_required_extensions(self, file_path: Path, data_object: object = None) -> list[str]:  # noqa: ARG002
        """Return required extensions for tabular data."""
        return self.required_schema_extensions

    def fast_probe(self, file_path: Path) -> dict[str, object]:
        """Fast probe to check file compatibility and get basic metadata."""
        # Basic file checks
        if not file_path.exists():
            return {"can_handle": False, "error": "File does not exist"}

        file_size = file_path.stat().st_size
        extension = file_path.suffix.lower()

        # Check if we support this extension
        can_handle = extension in self.supported_extensions

        # Try to detect MIME type
        mime_type = "unknown"
        if magic:
            try:
                mime_type = magic.from_file(str(file_path), mime=True)
            except (OSError, TypeError):
                mime_type = "unknown"

        return {
            "can_handle": can_handle,
            "file_size": file_size,
            "mime_type": mime_type,
            "extension": extension,
        }

    def analyze_deterministic(self, file_path: Path, data_object: object = None) -> dict[str, object]:
        """Analyze file without AI - deterministic analysis only."""
        # Basic data analysis (descriptive layer) - no AI needed
        if data_object is not None:
            return self._analyze_dataframe(data_object)
        return self._analyze_file(file_path)

    def analyze_deep(
        self,
        file_path: Path,
        data_object: object = None,
        ai_companion: object | None = None,
        deterministic_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Deep analysis using AI and heavy computation."""
        if not ai_companion or not hasattr(ai_companion, "is_available") or not ai_companion.is_available():
            return {"error": "AI companion not available for deep analysis"}

        # Use deterministic context as base
        data_analysis = deterministic_context.copy() if deterministic_context else {}

        # AI-powered interpretation (interpretive + contextual layers)
        return self._generate_ai_analysis(
            data_analysis,
            file_path,
            None,  # codebase_context not used in new structure
            ai_companion,
        )

    def generate_context(
        self,
        file_path: Path,
        data_object: object | None = None,
        codebase_context: dict[str, object] | None = None,
        ai_companion: object | None = None,  # noqa: ARG002
    ) -> dict[str, object]:
        """Generate data_structure context using bulk AI prompts.

        This method analyzes the tabular data and generates comprehensive
        column analysis using efficient bulk prompting instead of individual
        field-by-field prompts.
        """
        context: dict[str, object] = {}

        # Basic data analysis (descriptive layer)
        if data_object is not None:
            data_analysis = self._analyze_dataframe(data_object)
        else:
            data_analysis = self._analyze_file(file_path)

        # AI-powered interpretation (interpretive + contextual layers)
        if (
            hasattr(self, "llm_handler")
            and self.llm_handler
            and self.llm_handler.is_available()
        ):
            ai_analysis = self._generate_ai_analysis(
                data_analysis,
                file_path,
                codebase_context,
                self.llm_handler,
            )
            data_analysis.update(ai_analysis)

        # Create DataStructure schema
        context["data_structure"] = self._create_data_structure(data_analysis)

        return context

    def _analyze_dataframe(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyze a pandas DataFrame (descriptive layer).

        Args:
            df: Pandas DataFrame to analyze

        Returns:
            Dictionary containing basic data analysis

        """
        try:
            if not isinstance(df, pd.DataFrame):
                return {"error": "Object is not a pandas DataFrame"}

            analysis: dict[str, Any] = {
                "type": "pandas_dataframe",
                "shape": list(df.shape),
                "memory_usage_bytes": int(df.memory_usage(deep=True).sum()),
                "columns": {},
            }

            # Analyze each column
            for col_name in df.columns:
                col_data = df[col_name]
                analysis["columns"][col_name] = {
                    "dtype": str(col_data.dtype),
                    "null_count": int(col_data.isna().sum()),
                    "null_percentage": float(
                        col_data.isna().sum() / len(col_data) * 100,
                    ),
                    "unique_count": int(col_data.nunique()),
                    "sample_values": col_data.dropna().head(3).tolist(),
                }
        except ImportError:
            return {"error": "pandas not available for DataFrame analysis"}
        except (AttributeError, ValueError, TypeError) as e:
            return {"error": f"DataFrame analysis failed: {e!s}"}
        else:
            return analysis

    def _analyze_file(self) -> dict[str, Any]:
        """Analyze a file without the data object (fallback).

        Args:
            file_path: Path to the file

        Returns:
            Dictionary containing basic file analysis

        """
        return {
            "type": "file_only_analysis",
            "shape": [],
            "memory_usage_bytes": 0,
            "columns": {},
            "note": "Limited analysis - data object not provided",
        }

    def _generate_ai_analysis(
        self,
        data_analysis: dict[str, Any],
        file_path: Path,
        codebase_context: dict[str, Any] | None,
        llm_handler: object,
    ) -> dict[str, Any]:
        """Generate AI-powered analysis using bulk prompts.

        This replaces the old field-by-field approach with efficient bulk analysis.
        """
        ai_analysis = {}

        # Bulk column analysis prompt
        if data_analysis.get("columns"):
            column_analysis = self._bulk_analyze_columns(
                data_analysis["columns"],
                file_path,
                codebase_context,
                llm_handler,
            )
            ai_analysis["column_analysis"] = column_analysis

        # Schema interpretation prompt
        schema_interpretation = self._bulk_analyze_schema(
            data_analysis,
            file_path,
            codebase_context,
            llm_handler,
        )
        ai_analysis["domain_summary"] = schema_interpretation

        return ai_analysis

    def _bulk_analyze_columns(
        self,
        columns_data: dict[str, Any],
        file_path: Path,
        codebase_context: dict[str, Any] | None,
        llm_handler: object,
    ) -> dict[str, dict[str, Any]]:
        """Bulk analysis of all columns using constraint-aware template prompting."""
        try:
            from metacontext.ai.prompts.prompt_loader import PromptLoader
            
            # Prepare context for template-based analysis
            context_summary = self._prepare_context_summary(codebase_context)

            template_context = {
                "file_name": file_path.name,
                "project_summary": context_summary.get("project_summary", "Unknown project"),
                "code_summary": context_summary.get("code_summary", "Limited context"),
                "columns_data": columns_data,
            }

            # Use new constraint-aware template approach
            prompt_loader = PromptLoader()
            rendered_prompt = prompt_loader.render_prompt(
                "tabular/column_analysis",
                template_context
            )
            
            # Call LLM directly with the template-generated prompt
            response = llm_handler._call_llm(rendered_prompt)
            
            # Parse and validate the response manually since we bypassed generate_with_schema
            response_data = parse_json_response(response)
            
            # Convert response to legacy format expected by existing code
            return self._convert_column_response_to_legacy_format(response_data)

        except Exception:
            logger.exception("Error during template-based column analysis")
            # Fallback to empty result instead of raising
            return {}

    def _convert_column_response_to_legacy_format(self, response_data: dict) -> dict[str, dict[str, Any]]:
        """Convert raw LLM column response to legacy format for compatibility."""
        result = {}
        
        # Direct column response should already be in the expected format
        for col_name, col_data in response_data.items():
            if isinstance(col_data, dict):
                result[col_name] = col_data
            else:
                # Fallback for unexpected format
                result[col_name] = {
                    "ai_interpretation": str(col_data),
                    "ai_confidence": "MEDIUM",
                    "ai_domain_context": "",
                    "usage_guidance": "",
                    "semantic_meaning": str(col_data),
                    "data_quality_assessment": "",
                    "domain_context": "",
                    "relationship_to_other_columns": [],
                }
        
        return result

    def _convert_ai_enrichment_to_legacy_format(self, ai_enrichment: DataAIEnrichment) -> dict[str, dict[str, Any]]:
        """Convert schema-first AI enrichment to legacy format for compatibility."""
        result = {}

        if ai_enrichment.column_interpretations:
            for col_name, col_info in ai_enrichment.column_interpretations.items():
                result[col_name] = {
                    "ai_interpretation": col_info.semantic_meaning or "",
                    "ai_confidence": "HIGH",  # Default confidence
                    "ai_domain_context": col_info.domain_context or "",
                    "usage_guidance": col_info.usage_guidance or "",
                    "semantic_meaning": col_info.semantic_meaning or "",
                    "data_quality_assessment": col_info.data_quality_assessment or "",
                    "domain_context": col_info.domain_context or "",
                    "relationship_to_other_columns": col_info.relationship_to_other_columns or [],
                }

        return result

    def _convert_ai_schema_to_legacy_format(self, ai_enrichment: DataAIEnrichment) -> dict[str, Any]:
        """Convert schema-first AI enrichment to legacy format for backward compatibility."""
        result = {}

        # Map top-level fields
        if ai_enrichment.domain_analysis:
            result["domain_analysis"] = ai_enrichment.domain_analysis
        if ai_enrichment.data_quality_assessment:
            result["data_quality_assessment"] = ai_enrichment.data_quality_assessment
        if ai_enrichment.business_value_assessment:
            result["business_value_assessment"] = ai_enrichment.business_value_assessment

        return result

    def _bulk_analyze_schema(
        self,
        data_analysis: dict[str, Any],
        file_path: Path,
        codebase_context: dict[str, Any] | None,
        llm_handler: object,
    ) -> dict[str, Any]:
        """Analyze the overall schema characteristics and business context."""
        # Prepare context for schema analysis
        context_summary = self._prepare_context_summary(codebase_context)

        # Extract key information for schema-level analysis
        num_columns = len(data_analysis.get("columns", {}))
        data_shape = data_analysis.get("shape", [])
        rows = data_shape[0] if len(data_shape) > 0 else "unknown"

        # Default fallback response using schema-driven keys
        fallback_response = {
            key: f"{key.replace('_', ' ').capitalize()} unavailable"
            for key in DataAIEnrichment.model_fields
            if key in ["domain_analysis", "data_quality_assessment", "business_value_assessment"]
        }

        try:
            # Use new constraint-aware template approach
            # Create context for the template using available variables
            template_context = {
                "file_name": file_path.name,
                "rows": rows,
                "num_columns": num_columns,
                "project_summary": context_summary.get("project_summary", "Dataset analysis for business insights"),
                "columns": list(data_analysis.get("columns", {}).keys()),
            }

            # Render the constraint-aware template
            rendered_prompt = PromptLoader().render_prompt(
                "tabular/schema_analysis",
                template_context,
            )
            
            # Call LLM with the constraint-aware prompt
            response = llm_handler._call_llm(rendered_prompt)
            
            # Parse the JSON response
            response_data = parse_json_response(response)
            
            # Validate against schema
            ai_enrichment = DataAIEnrichment(**response_data)

            # Convert to legacy format for backward compatibility
            return self._convert_ai_schema_to_legacy_format(ai_enrichment)

        except (LLMError, ValidationRetryError):
            return fallback_response

    def _prepare_context_summary(
        self,
        codebase_context: dict[str, Any] | None,
    ) -> dict[str, str]:
        """Prepare a summary of available context for AI prompts."""
        if not codebase_context:
            return {
                "project_summary": "No project context available",
                "code_summary": "No code context",
            }

        return {
            "project_summary": codebase_context.get(
                "readme_summary",
                "No README available",
            ),
            "code_summary": str(
                codebase_context.get("scan_summary", "No code scan available"),
            ),
        }

    def _query_llm_handler(self, llm_handler: object, prompt: str) -> str | dict[str, Any]:
        """Query the LLM handler with appropriate error handling."""
        result: str | dict[str, Any] = {}
        try:
            if hasattr(llm_handler, "call_sync_llm"):
                result = llm_handler.call_sync_llm(prompt)
            elif hasattr(llm_handler, "call"):
                result = llm_handler.call(prompt)
        except (json.JSONDecodeError, AttributeError):
            pass
        return result

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse response from LLM to extract JSON structure.

        Args:
            response: The raw response from LLM

        Returns:
            Dictionary parsed from JSON in response

        """
        return self._extract_json_from_response(response)

    def _extract_json_from_response(self, response: str) -> dict[str, Any]:
        """Extract JSON from markdown-wrapped responses."""
        try:
            # First try direct JSON parsing
            result = json.loads(response)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks

            # Look for ```json ... ``` or ``` ... ``` blocks
            json_pattern = r"```(?:json)?\s*(\{.*?\})\s*```"
            match = re.search(json_pattern, response, re.DOTALL)

            if match:
                try:
                    result = json.loads(match.group(1))
                    return result if isinstance(result, dict) else {}
                except json.JSONDecodeError:
                    pass

            # If no markdown blocks, try to find JSON-like content
            # Look for content between first { and last }
            start_idx = response.find("{")
            end_idx = response.rfind("}")

            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx : end_idx + 1]
                try:
                    result = json.loads(json_str)
                    return result if isinstance(result, dict) else {}
                except json.JSONDecodeError:
                    pass

            return {}

    def _fallback_column_analysis(
        self,
        columns_data: dict[str, Any],
    ) -> dict[str, dict[str, Any]]:
        """Fallback analysis when AI is not available."""
        return {
            col_name: {
                "semantic_meaning": f"Column analysis unavailable for {col_name}",
                "data_quality_assessment": "No AI companion available for analysis",
                "domain_context": "Unknown domain context",
                "relationship_to_other_columns": [],
            }
            for col_name in columns_data
        }

    def _create_data_structure(self, data_analysis: dict[str, Any]) -> DataStructure:
        """Create the DataStructure schema from analysis data."""
        # Deterministic info
        deterministic_info = DataDeterministicMetadata(
            type=data_analysis.get("type", "unknown"),
            shape=data_analysis.get("shape", []),
            memory_usage_bytes=data_analysis.get("memory_usage_bytes", 0),
        )

        # AI enrichment
        ai_enrichment = self._create_ai_enrichment(data_analysis)

        # Return DataStructure object
        return DataStructure(
            deterministic_metadata=deterministic_info,
            ai_enrichment=ai_enrichment,
        )

    def _build_constrained_instruction(self, schema_context: dict[str, Any]) -> str:
        """Build instruction with strict response size constraints."""
        column_count = len(schema_context.get("column_details", {}).get("columns", []))

        # Calculate limits using the utility function
        max_total_chars, max_field_chars = calculate_response_limits(
            base_fields=7,  # ForensicAIEnrichment base fields
            extended_fields=4,  # DataAIEnrichment specific fields
            complexity_factor=max(0.5, min(2.0, column_count / 10.0)),  # Scale with column count
        )

        # Shorter interpretations for many columns
        max_interpretation_length = min(max_field_chars, 500 // max(column_count, 1))

        # Build field-specific constraints
        field_constraints = {
            **COMMON_FIELD_CONSTRAINTS,
            "domain_analysis": "Business domain + purpose in 1 sentence",
            "data_quality_assessment": "Quality level + main issues in 1 sentence",
            "column_interpretations": f"Per column: max {max_interpretation_length} chars each",
            "business_value_assessment": "Value proposition in 1 sentence",
        }

        base_instruction = "Analyze this tabular dataset and provide insights"
        constraints = build_schema_constraints(
            max_total_chars=max_total_chars,
            max_field_chars=max_field_chars,
            field_descriptions=field_constraints,
            complexity_context=f"Dataset has {column_count} columns",
        )

        return f"{base_instruction} that fit within these STRICT LIMITS:\n\n{constraints}"

    def _create_ai_enrichment(
        self,
        data_analysis: dict[str, Any],
    ) -> DataAIEnrichment:
        """Create the DataAIEnrichment schema."""
        schema_interpretation = data_analysis.get("domain_summary", {})
        column_analysis = data_analysis.get("column_analysis", {})

        # Create column interpretations dictionary
        column_interpretations = {}
        for col_name, col_data in column_analysis.items():
            # Only add columns that have analysis
            if col_data:
                det_info = ColumnDeterministicInfo()
                ai_info = ColumnAIEnrichment(
                    semantic_meaning=col_data.get("semantic_meaning", ""),
                    data_quality_assessment=col_data.get("supporting_evidence", ""),
                    domain_context=col_data.get("domain_context", ""),
                    relationship_to_other_columns=col_data.get("relationship_to_other_columns", []),
                )
                column_interpretations[col_name] = ColumnInfo(
                    deterministic=det_info,
                    ai_enrichment=ai_info,
                )

        return DataAIEnrichment(
            domain_analysis=schema_interpretation.get("domain_summary", ""),
            data_quality_assessment=schema_interpretation.get("overall_clarity", "unknown"),
            column_interpretations=column_interpretations,
            business_value_assessment=schema_interpretation.get("business_impact", ""),
        )

    def _simple_extract_json(self, response: str) -> dict[str, Any] | None:
        """Extract JSON from a string that might contain other text."""
        # Find the start and end of the JSON object
        match = re.search(r"\{.*\}", response, re.DOTALL)
        if match:
            json_str = match.group(0)
            try:
                result = json.loads(json_str)
                if isinstance(result, dict):
                    return result
            except json.JSONDecodeError:
                logger.warning("Failed to decode JSON from LLM response.")
        return None

    PROMPT_CONFIG: ClassVar[dict[str, str]] = {
        "column_analysis": "templates/tabular/column_analysis.yaml",
        "schema_interpretation": "templates/tabular/schema_analysis.yaml",
    }

    def get_bulk_prompts(self, file_path: Path, data_object: object = None) -> dict[str, str]:
        """Get bulk prompts for this file type from config."""
        return self.PROMPT_CONFIG.copy()
