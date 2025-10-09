"""Model handler for machine learning model files.

This handler processes machine learning model files to extract metadata using
both deterministic techniques and AI enrichment. It implements the architectural
patterns defined in the central architecture reference.

See:
- architecture_reference.ArchitecturalComponents.TWO_TIER_ARCHITECTURE
- architecture_reference.ArchitecturalComponents.SCHEMA_FIRST_LLM
- architecture_reference.HandlerArchitecture.MODEL_HANDLER
"""

import logging
import pickle
import re
from pathlib import Path
from typing import Any, ClassVar

from metacontext.ai.handlers.exceptions import LLMError, ValidationRetryError
from metacontext.ai.handlers.llms.prompt_constraints import (
    COMMON_FIELD_CONSTRAINTS,
    build_schema_constraints,
    calculate_response_limits,
)
from metacontext.ai.handlers.llms.provider_interface import LLMProvider
from metacontext.handlers.base import BaseFileHandler, register_handler
from metacontext.schemas.core.interfaces import ConfidenceLevel
from metacontext.schemas.extensions.models import (
    ModelAIEnrichment,
    ModelContext,
    ModelDeterministicMetadata,
    TrainingData,
)

logger = logging.getLogger(__name__)

# Constants for response size constraints
LARGE_MODEL_SIZE_MB = 100
MEDIUM_MODEL_SIZE_MB = 10
DEBUG_FIELD_PREVIEW_LENGTH = 100


@register_handler
class ModelHandler(BaseFileHandler):
    """Handler for machine learning model files (e.g., .pkl, .h5)."""

    supported_extensions: ClassVar[list[str]] = [".pkl", ".joblib", ".h5", ".keras"]
    required_schema_extensions: ClassVar[list[str]] = ["model_context"]
    llm_handler: LLMProvider | None

    def __init__(self, llm_handler: LLMProvider | None = None) -> None:
        """Initialize the model handler."""
        self.llm_handler = llm_handler

    def can_handle(self, file_path: Path, data_object: object | None = None) -> bool:
        """Check if the handler can process the given file."""
        return file_path.suffix.lower() in self.supported_extensions

    def get_required_extensions(
        self,
        file_path: Path,
        data_object: object | None = None,
    ) -> list[str]:
        """Get required schema extensions for this file type."""
        return ["model_context"]

    def fast_probe(self, file_path: Path) -> dict[str, object]:
        """Fast probe to check file compatibility and get basic metadata."""
        # Basic file checks
        if not file_path.exists():
            return {"can_handle": False, "error": "File does not exist"}

        file_size = file_path.stat().st_size
        extension = file_path.suffix.lower()

        # Check if we support this extension
        can_handle = extension in self.supported_extensions

        return {
            "can_handle": can_handle,
            "file_size": file_size,
            "mime_type": "application/octet-stream",  # Most model files are binary
            "extension": extension,
        }

    def analyze_deterministic(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> dict[str, object]:
        """Analyze file without AI - deterministic analysis only."""
        # Basic model file analysis - no AI needed
        return {
            "model_path": str(file_path),
            "model_filename": file_path.name,
            "model_stem": file_path.stem,
            "file_size_bytes": file_path.stat().st_size,
            "file_extension": file_path.suffix.lower(),
        }

    def analyze_deep(
        self,
        file_path: Path,
        data_object: object = None,
        ai_companion: object | None = None,
        deterministic_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Deep analysis using AI and heavy computation."""
        if not ai_companion:
            return {"error": "AI companion not available for deep analysis"}

        # Use deterministic context as base
        deep_analysis = deterministic_context.copy() if deterministic_context else {}

        # Add training script discovery and AI analysis
        training_scripts = self._discover_training_scripts(file_path)
        training_analysis = {}

        if training_scripts:
            # TODO: Implement training script analysis
            training_analysis = {
                "analysis": "Training script analysis not yet implemented",
            }

        deep_analysis.update(
            {
                "training_scripts": [str(script) for script in training_scripts],
                "training_analysis": training_analysis,
            },
        )

        return deep_analysis

    def _discover_training_scripts(self, model_path: Path) -> list[Path]:
        """Discover training scripts related to the model using intelligent search.

        Uses heuristics to find Python files that likely created or trained the model:
        - Files in same directory or parent directories
        - Files containing model filename or stem
        - Files with common ML training patterns
        """
        logger.info("🔍 Discovering training scripts for %s...", model_path.name)

        search_paths = [
            model_path.parent,
            model_path.parent.parent,
            model_path.parent.parent / "example",
            model_path.parent.parent / "scripts",
            model_path.parent.parent / "src",
        ]

        training_scripts = []
        model_stem = model_path.stem.lower()

        for search_path in search_paths:
            if not search_path.exists():
                continue

            for py_file in search_path.rglob("*.py"):
                if "__pycache__" in str(py_file) or py_file.name.startswith("."):
                    continue

                try:
                    with Path.open(py_file, encoding="utf-8") as f:
                        content = f.read().lower()

                    # Scoring system for relevance
                    score = 0

                    # Model-specific indicators
                    if model_path.name.lower() in content:
                        score += 10
                    if model_stem in content:
                        score += 5

                    # ML training indicators
                    ml_patterns = [
                        "pickle.dump",
                        "joblib.dump",
                        ".fit(",
                        ".train(",
                        "model.save",
                        "torch.save",
                        ".pkl",
                        "sklearn",
                        "train_test_split",
                        "model =",
                        "classifier =",
                        "regressor =",
                        ".predict(",
                    ]
                    score += sum(2 for pattern in ml_patterns if pattern in content)

                    # File structure indicators
                    training_indicators = [
                        "train",
                        "model",
                        "fit",
                        "classifier",
                        "regressor",
                    ]
                    filename_lower = py_file.name.lower()
                    score += sum(
                        3
                        for indicator in training_indicators
                        if indicator in filename_lower
                    )
                    min_score = 5
                    if score >= min_score:  # Minimum threshold
                        training_scripts.append((py_file, score))
                        logger.info(
                            "   📝 Found potential script: %s (score: %s)",
                            py_file.name,
                            score,
                        )

                except OSError:
                    continue

        # Sort by relevance and return top candidates
        training_scripts.sort(key=lambda x: x[1], reverse=True)
        return [script[0] for script in training_scripts[:3]]

    def _extract_deterministic_metadata(
        self,
        model_path: Path,
    ) -> ModelDeterministicMetadata:
        """Extract deterministic facts about the model through direct inspection.

        This tier always succeeds and provides 100% reliable information:
        - Model type from object inspection
        - Hyperparameters from model attributes
        - File size and technical specs
        - Framework detection
        """
        logger.info("📊 Extracting deterministic model metadata...")

        try:
            # Basic file information
            model_size_bytes = model_path.stat().st_size

            # Load and inspect model
            with Path.open(model_path, "rb") as f:
                model_data = pickle.load(f)  # noqa: S301

            # Handle different storage formats
            if isinstance(model_data, dict) and "model" in model_data:
                actual_model = model_data["model"]
            else:
                actual_model = model_data

            # Extract model type and hyperparameters
            model_type = type(actual_model).__name__
            framework = self._determine_framework(model_type)

            hyperparameters = {}
            if hasattr(actual_model, "get_params"):
                try:
                    params = actual_model.get_params()
                    hyperparameters = {k: v for k, v in params.items() if v is not None}
                    logger.info(
                        "   ⚙️  Extracted %s hyperparameters",
                        len(hyperparameters),
                    )
                except AttributeError:
                    logger.warning("   ⚠️  Could not extract hyperparameters")

            # Extract shapes if available
            input_shape = None
            output_shape = None

            if hasattr(actual_model, "n_features_in_"):
                input_shape = [actual_model.n_features_in_]
            if hasattr(actual_model, "classes_"):
                output_shape = [len(actual_model.classes_)]

            logger.info("   📊 Model type: %s", model_type)
            logger.info("   🏗️  Framework: %s", framework)
            logger.info("   💾 File size: %s bytes", f"{model_size_bytes:,}")

            return ModelDeterministicMetadata(
                framework=framework,
                model_type=model_type,
                hyperparameters=hyperparameters,
                input_shape=input_shape,
                output_shape=output_shape,
                model_size_bytes=model_size_bytes,
            )

        except (pickle.UnpicklingError, AttributeError, ImportError) as e:
            logger.warning("   ⚠️ Error loading model: %s", e)

            # Fallback with minimal information
            return ModelDeterministicMetadata(
                framework="unknown",
                model_type="unknown",
                model_size_bytes=model_path.stat().st_size,
            )

    def _determine_framework(self, model_type: str) -> str:
        """Determine ML framework from model type."""
        model_type_lower = str(model_type).lower()

        if any(keyword in model_type_lower for keyword in ["tensorflow", "keras"]):
            return "tensorflow"
        if any(keyword in model_type_lower for keyword in ["pytorch", "torch"]):
            return "pytorch"
        if "xgboost" in model_type_lower:
            return "xgboost"
        if "lightgbm" in model_type_lower:
            return "lightgbm"
        return "scikit-learn"

    INSTRUCTION_CONFIG: ClassVar[dict[str, str]] = {
        "ai_enrichment": "Analyze this ML model and generate comprehensive AI enrichment including specific training features used",
    }

    def _build_constrained_model_instruction(self, context_data: dict[str, Any]) -> str:
        """Build instruction with strict response size constraints for model analysis."""
        # Estimate complexity based on available metadata
        model_size = context_data.get("model_info", {}).get("file_size_mb", 0)

        # Calculate limits using the utility function
        if model_size > LARGE_MODEL_SIZE_MB:  # Large model
            complexity_factor = 1.5
        elif model_size > MEDIUM_MODEL_SIZE_MB:  # Medium model
            complexity_factor = 1.0
        else:  # Small model or unknown size
            complexity_factor = 0.8

        max_total_chars, max_field_chars = calculate_response_limits(
            base_fields=7,  # ForensicAIEnrichment base fields
            extended_fields=6,  # ModelAIEnrichment specific fields
            complexity_factor=complexity_factor,
        )

        # Build field-specific constraints
        field_constraints = {
            **COMMON_FIELD_CONSTRAINTS,
            "model_type_analysis": f"Architecture + algorithm (max {max_field_chars} chars)",
            "purpose": f"Business problem solved (max {max_field_chars} chars)",
            "training_approach": f"Training method summary (max {max_field_chars} chars)",
            "expected_inputs": f"Input format description (max {max_field_chars} chars)",
            "expected_outputs": f"Output format description (max {max_field_chars} chars)",
            "limitations": f"Key constraints only (max {max_field_chars} chars)",
        }

        base_instruction = (
            "Analyze this ML model and generate comprehensive AI enrichment"
        )
        constraints = build_schema_constraints(
            max_total_chars=max_total_chars,
            max_field_chars=max_field_chars,
            field_descriptions=field_constraints,
            complexity_context=f"Model size: {model_size}MB",
        )

        return (
            f"{base_instruction} that fit within these STRICT LIMITS:\n\n{constraints}"
        )

    def _generate_ai_enrichment(
        self,
        deterministic_metadata: ModelDeterministicMetadata,
        model_path: Path,
        codebase_context: dict | None = None,
        semantic_knowledge: str | None = None,
    ) -> ModelAIEnrichment | None:
        """Generate AI enrichment through schema-first LLM analysis.

        This tier provides interpretive analysis but gracefully degrades if unavailable:
        - Training data analysis from script inspection
        - Domain context inference
        - Usage guidance and recommendations
        - Performance insights
        """
        if not self.llm_handler or not self.llm_handler.is_available():
            logger.info("🤖 No LLM available - skipping AI enrichment")
            return None

        logger.info("🤖 Generating AI enrichment through schema-first analysis...")

        # Discover and analyze training scripts
        training_scripts = self._discover_training_scripts(model_path)

        # Prepare context data for LLM
        context_data: dict[str, Any] = {
            "deterministic_metadata": deterministic_metadata.dict(),
            "model_file_path": str(model_path),
            "training_scripts_found": [],
            "codebase_context": codebase_context or {},
            "semantic_knowledge": semantic_knowledge
            or "No semantic knowledge available from codebase",
        }

        # Add training script content and extract features
        for script_path in training_scripts[:2]:  # Limit to prevent token overflow
            try:
                with Path.open(script_path, encoding="utf-8") as f:
                    content = f.read()

                # Extract potential training features from script
                training_features = self._extract_training_features(content)

                context_data["training_scripts_found"].append(
                    {
                        "filename": script_path.name,
                        "path": str(script_path),
                        "content_preview": content[:3000],  # Limit content size
                        "extracted_features": training_features,
                    },
                )
            except OSError as e:
                logger.warning("   ⚠️ Could not read %s: %s", script_path.name, e)
                # Store the error message in a variable
                msg = f"Failed to read training script: {script_path.name}"
                # Raise with context to preserve the stack trace
                raise ValueError(msg) from e

        try:
            # Use schema-first generation with constrained instruction
            instruction = self._build_constrained_model_instruction(context_data)
            ai_enrichment = self.llm_handler.generate_with_schema(
                schema_class=ModelAIEnrichment,
                context_data=context_data,
                instruction=instruction,
            )

            # Debug: print the generated AI enrichment
            if ai_enrichment:
                logger.info("AI Enrichment Content:")
                for field_name, field_value in ai_enrichment.model_dump().items():
                    if field_value:  # Only show non-empty fields
                        logger.info(
                            "  %s: %s",
                            field_name,
                            field_value[:DEBUG_FIELD_PREVIEW_LENGTH] + "..."
                            if isinstance(field_value, str)
                            and len(field_value) > DEBUG_FIELD_PREVIEW_LENGTH
                            else field_value,
                        )

            logger.info("✅ AI enrichment generated successfully")
            return (
                ai_enrichment if isinstance(ai_enrichment, ModelAIEnrichment) else None
            )

        except (LLMError, ValidationRetryError) as e:
            logger.warning("⚠️  AI enrichment failed: %s", e)
            return self._create_fallback_ai_enrichment(
                deterministic_metadata,
                training_scripts,
            )

    def _create_fallback_ai_enrichment(
        self,
        deterministic_metadata: ModelDeterministicMetadata,
        training_scripts: list[Path],
    ) -> ModelAIEnrichment:
        """Create basic AI enrichment when LLM is unavailable, with feature extraction."""
        ai_interpretation = f"Analysis of {deterministic_metadata.model_type} model"

        # Extract training features from scripts
        all_features: dict[str, list[str]] = {
            "potential_columns": [],
            "target_variables": [],
            "preprocessing_steps": [],
        }

        for script_path in training_scripts[:2]:  # Limit to prevent overhead
            try:
                with Path.open(script_path, encoding="utf-8") as f:
                    content = f.read()
                features = self._extract_training_features(content)
                all_features["potential_columns"].extend(
                    features.get("potential_columns", []),
                )
                all_features["target_variables"].extend(
                    features.get("target_variables", []),
                )
                all_features["preprocessing_steps"].extend(
                    features.get("preprocessing_steps", []),
                )
            except OSError:
                continue

        # Remove duplicates
        for key, value in all_features.items():
            all_features[key] = list(set(value))

        if training_scripts:
            ai_interpretation += (
                f" with {len(training_scripts)} training scripts discovered"
            )
            if all_features["potential_columns"]:
                ai_interpretation += f" using features: {', '.join(all_features['potential_columns'][:5])}"

        # Basic domain inference from framework
        domain_mapping = {
            "scikit-learn": "General machine learning",
            "tensorflow": "Deep learning",
            "pytorch": "Deep learning",
            "xgboost": "Gradient boosting",
            "lightgbm": "Gradient boosting",
        }

        ai_domain_context = domain_mapping.get(
            deterministic_metadata.framework or "unknown",
            "Machine learning model",
        )

        usage_guidance = f"Load model using appropriate library for {deterministic_metadata.framework}. "
        usage_guidance += "Ensure input data matches training format. "
        usage_guidance += "Evaluate performance before production deployment."

        # Create training data with extracted features
        training_data = None
        if all_features["potential_columns"] or all_features["target_variables"]:
            training_data = TrainingData(
                potential_columns=all_features["potential_columns"]
                if all_features["potential_columns"]
                else None,
                target_variables=all_features["target_variables"]
                if all_features["target_variables"]
                else None,
                preprocessing_steps=all_features["preprocessing_steps"]
                if all_features["preprocessing_steps"]
                else None,
            )

        # Schema-driven fallback: fill missing fields with default values
        fallback_fields = ModelAIEnrichment.model_fields
        enrichment_kwargs = {}

        # Required string fields that need default values to avoid validation errors
        required_string_fields = [
            "model_type_analysis",
            "purpose",
            "training_approach",
            "expected_inputs",
            "expected_outputs",
            "limitations",
            "ideal_use_cases",
            "data_requirements",
        ]

        # Set defaults for all fields based on their type to handle validation
        for field_name, field in fallback_fields.items():
            field_type = str(field.annotation)

            if field_name in required_string_fields or "str" in field_type:
                enrichment_kwargs[field_name] = (
                    ""  # Empty string instead of "Not available"
                )
            elif "dict" in field_type:
                enrichment_kwargs[field_name] = {}
            elif "list" in field_type:
                enrichment_kwargs[field_name] = []
            else:
                enrichment_kwargs[field_name] = None

        # Update with any actual values we have
        enrichment_kwargs.update(
            {
                "ai_interpretation": ai_interpretation,
                "ai_confidence": ConfidenceLevel.LOW,
                "ai_domain_context": ai_domain_context,
                "usage_guidance": usage_guidance,
                "training_data": training_data,
                "evaluation_metrics": {},  # Empty dict instead of None
                "deployment_recommendations": [],  # Empty list instead of None
            },
        )
        return ModelAIEnrichment(**enrichment_kwargs)

    def generate_context(
        self,
        file_path: Path,
        data_object: object | None = None,
        codebase_context: dict[str, object] | None = None,
        ai_companion: object | None = None,
    ) -> dict[str, object]:
        """Generate complete model context using unified pipeline architecture.

        Uses the unified pipeline where both API and companion modes follow identical
        workflow steps B→H, with only step I (LLM Analysis) differing.

        Returns:
            Dictionary with model_context containing deterministic_metadata and ai_enrichment

        """
        logger.info("\n🚀 TWO-TIER MODEL ANALYSIS")
        logger.info("📁 Analyzing: %s", file_path.name)
        logger.info("=" * 60)

        # Tier 1: Always succeeds - deterministic metadata
        deterministic_metadata = self._extract_deterministic_metadata(file_path)

        # Tier 2: Best effort - AI enrichment using unified pipeline
        ai_enrichment = None
        llm_handler = ai_companion or self.llm_handler

        if llm_handler:
            # Extract semantic knowledge for enhanced context (Step D)
            semantic_knowledge_text = "No semantic knowledge extracted from codebase."
            if (
                hasattr(ai_companion, "codebase_context")
                and ai_companion.codebase_context
            ):
                logger.info("🔍 DEBUG: Codebase context found on ai_companion")
                try:
                    # Check if we have semantic knowledge available
                    if (
                        hasattr(ai_companion.codebase_context, "ai_enrichment")
                        and ai_companion.codebase_context.ai_enrichment
                        and hasattr(
                            ai_companion.codebase_context.ai_enrichment,
                            "semantic_knowledge",
                        )
                    ):
                        logger.info(
                            "🔍 DEBUG: Found semantic knowledge in ai_enrichment",
                        )
                        semantic_knowledge = ai_companion.codebase_context.ai_enrichment.semantic_knowledge

                        # Format semantic knowledge for AI analysis
                        if semantic_knowledge and hasattr(
                            semantic_knowledge,
                            "model_fields",
                        ):
                            logger.info(
                                "🔍 DEBUG: Semantic knowledge has %d model fields",
                                len(semantic_knowledge.model_fields),
                            )
                            field_descriptions = []
                            for (
                                field_name,
                                field_info,
                            ) in semantic_knowledge.model_fields.items():
                                logger.info(
                                    "🔍 DEBUG: Field %s: pydantic='%s', definition='%s'",
                                    field_name,
                                    field_info.pydantic_description,
                                    field_info.definition,
                                )
                                if field_info.pydantic_description:
                                    field_descriptions.append(
                                        f"- {field_name}: {field_info.pydantic_description}",
                                    )
                                elif field_info.definition:
                                    field_descriptions.append(
                                        f"- {field_name}: {field_info.definition}",
                                    )

                            if field_descriptions:
                                semantic_knowledge_text = (
                                    "Semantic knowledge from codebase:\n"
                                    + "\n".join(field_descriptions)
                                )
                                logger.info(
                                    "🔍 DEBUG: Using semantic knowledge: %s",
                                    semantic_knowledge_text[:200] + "...",
                                )
                except (AttributeError, KeyError, TypeError) as e:
                    logger.warning("Error extracting semantic knowledge: %s", e)

            # Use unified pipeline for AI enrichment (Steps E→I)
            # The LLM provider type determines whether this goes to API or companion mode
            if (
                hasattr(llm_handler, "is_companion_mode")
                and llm_handler.is_companion_mode()
            ):
                logger.info("🤖 Using companion mode for model analysis")
            else:
                logger.info("🔗 Using API LLM provider for model analysis")

            ai_enrichment = self._generate_ai_enrichment(
                deterministic_metadata,
                file_path,
                codebase_context,
                semantic_knowledge_text,
            )
        else:
            logger.info("i  No LLM handler provided - creating basic AI enrichment")
            training_scripts = self._discover_training_scripts(file_path)
            ai_enrichment = self._create_fallback_ai_enrichment(
                deterministic_metadata,
                training_scripts,
            )

        # Create model context
        model_context = ModelContext(
            deterministic_metadata=deterministic_metadata,
            ai_enrichment=ai_enrichment,
        )

        logger.info("✅ Two-tier analysis complete!")
        logger.info(
            "   🔧 Deterministic fields: %s",
            len(deterministic_metadata.model_dump(exclude_none=True)),
        )
        if ai_enrichment:
            logger.info(
                "   🤖 AI enrichment fields: %s",
                len(ai_enrichment.model_dump(exclude_none=True)),
            )

        return {"model_context": model_context.model_dump()}

    def _extract_training_features(self, script_content: str) -> dict[str, Any]:
        """Extract potential training features from script content.

        Analyzes Python training scripts to identify:
        - Column names used in data loading/processing
        - Feature selection patterns
        - Target variable definitions
        - Data preprocessing steps
        """
        features_info: dict[str, list[str]] = {
            "potential_columns": [],
            "target_variables": [],
            "feature_patterns": [],
            "preprocessing_steps": [],
        }

        content_lower = script_content.lower()

        # Look for column definitions - simplified patterns
        column_patterns = [
            r"feature_columns\s*=\s*\[([^\]]+)\]",  # feature_columns = ['col1', 'col2', ...]
            r"df\[['\"]([\w_]+)['\"]\]",  # df['column_name']
            r"data\[['\"]([\w_]+)['\"]\]",  # data['column_name']
            r"X\s*=\s*df\[([^\]]+)\]",  # X = df[['col1', 'col2']]
        ]

        for pattern in column_patterns:
            matches = re.findall(pattern, script_content, re.IGNORECASE)
            for match in matches:
                if "feature_columns" in pattern or "X =" in pattern:
                    # Handle list definitions
                    columns = re.findall(r"['\"]([\w_]+)['\"]", match)
                    features_info["potential_columns"].extend(columns)
                else:
                    # Handle single column references
                    features_info["potential_columns"].append(match)

        # Look for target variables
        target_patterns = [
            r"target\s*=\s*['\"]([^'\"]+)['\"]",
            r"y\s*=\s*.*\[(['\"][^'\"]+['\"])\]",
            r"label[s]?\s*=\s*['\"]([^'\"]+)['\"]",
        ]

        for pattern in target_patterns:
            matches = re.findall(pattern, script_content, re.IGNORECASE)
            for match in matches:
                features_info["target_variables"].append(match.strip("'\""))

        # Look for feature engineering patterns
        if "standardscaler" in content_lower or "minmaxscaler" in content_lower:
            features_info["preprocessing_steps"].append("feature_scaling")
        if "labelencoder" in content_lower or "onehotencoder" in content_lower:
            features_info["preprocessing_steps"].append("categorical_encoding")
        if "train_test_split" in content_lower:
            features_info["preprocessing_steps"].append("train_test_split")
        if "pca" in content_lower:
            features_info["preprocessing_steps"].append("dimensionality_reduction")

    # Prompt configuration for bulk analysis
    PROMPT_CONFIG: ClassVar[dict[str, str]] = {
        "model_analysis": "templates/model/model_analysis.yaml",
        "training_data_analysis": "templates/model/training_data_analysis.yaml",
    }

    def get_bulk_prompts(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> dict[str, str]:
        """Get bulk prompts for this file type from config."""
        return self.PROMPT_CONFIG.copy()
