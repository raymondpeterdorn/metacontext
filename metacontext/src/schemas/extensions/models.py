"""Model extension schemas."""

from typing import Any

from pydantic import BaseModel, Field, model_validator

from metacontext.schemas.extensions.base import (
    AIEnrichment,
    DeterministicMetadata,
    ExtensionContext,
)

# ========== MODEL CONTEXT EXTENSION ==========


class ModelDeterministicMetadata(DeterministicMetadata):
    """Deterministic facts about ML models extracted through code execution."""

    framework: str | None = None
    model_type: str | None = None
    model_version: str | None = None
    hyperparameters: dict[str, Any] | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    model_size_bytes: int | None = None


class TrainingData(BaseModel):
    """Training data information - can contain both deterministic and AI fields."""

    # Deterministic fields
    source_file: str | None = None
    samples: int | None = None
    features_used: list[str] | None = None
    target_variable: str | None = None
    train_test_split: dict[str, float] | None = None

    # Enhanced feature information from script analysis
    potential_columns: list[str] | None = None
    target_variables: list[str] | None = None
    preprocessing_steps: list[str] | None = None

    # AI enrichment fields (more flexible to handle LLM responses)
    feature_descriptions: dict[str, str] | None = Field(default=None)

    @model_validator(mode="before")
    @classmethod
    def clean_feature_descriptions(cls, values: dict[str, Any]) -> dict[str, Any]:
        """Clean up feature_descriptions to remove None values that break validation."""
        if "feature_descriptions" in values:
            feature_desc = values["feature_descriptions"]
            if isinstance(feature_desc, dict):
                # Remove None values and replace with empty strings
                cleaned = {
                    k: v if v is not None else "" for k, v in feature_desc.items()
                }
                values["feature_descriptions"] = cleaned
        return values


class ModelAIEnrichment(AIEnrichment):
    """AI-generated insights about ML models."""

    model_type_analysis: str = Field(
        default="",
        description="Detailed assessment of the model architecture and type, including framework, algorithm family, and key architectural elements.",
    )
    purpose: str = Field(
        default="",
        description="The specific problem this model is designed to solve, described in business terms.",
    )
    training_approach: str = Field(
        default="",
        description="Description of the training methodology used, including optimization approach, regularization, and training stages.",
    )
    expected_inputs: str = Field(
        default="",
        description="Description of the expected input data format, including dimensions, types, and preprocessing requirements.",
    )
    expected_outputs: str = Field(
        default="",
        description="Description of the model's output format, including dimensions, types, and interpretation guidance.",
    )
    limitations: str = Field(
        default="",
        description="Potential limitations of this model, including edge cases, biases, and constraints.",
    )
    ideal_use_cases: str = Field(
        default="",
        description="Best scenarios for using this model, considering its strengths and design objectives.",
    )
    data_requirements: str = Field(
        default="",
        description="What kind of data this model needs to function well, including volume, quality, and diversity requirements.",
    )
    training_data: TrainingData | None = None
    evaluation_metrics: dict[str, Any] | None = Field(
        default=None,
        description="Metrics used to evaluate model performance, such as accuracy, precision, recall, F1 score, etc.",
    )
    performance_analysis: str | None = Field(
        default=None,
        description="Comprehensive analysis of the model's performance, including strengths, weaknesses, and comparisons to benchmarks.",
    )
    deployment_recommendations: list[str] | None = Field(
        default=None,
        description="Recommendations for deploying this model, including infrastructure, monitoring, and serving strategies.",
    )


class ModelContext(ExtensionContext):
    """Extension for machine learning models.

    See: architecture_reference.ArchitecturalComponents.TWO_TIER_METADATA
    """

    deterministic_metadata: ModelDeterministicMetadata | None = None
    ai_enrichment: ModelAIEnrichment | None = None
