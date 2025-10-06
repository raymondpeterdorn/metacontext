"""Utility functions for building constrained LLM prompts to prevent oversized responses.

This module provides reusable functions for creating size-constrained prompts
that prevent JSON truncation and ensure valid schema responses within token limits.
"""


def build_schema_constraints(
    max_total_chars: int,
    max_field_chars: int,
    field_descriptions: dict[str, str] | None = None,
    complexity_context: str = "",
) -> str:
    """Build a standardized constraint instruction for schema-based prompting.

    Args:
        max_total_chars: Maximum total response size in characters
        max_field_chars: Maximum size per field in characters
        field_descriptions: Optional mapping of field names to constraint descriptions
        complexity_context: Optional context about data complexity for scaling

    Returns:
        Formatted constraint instruction string

    """
    base_constraints = f"""RESPONSE SIZE CONSTRAINTS:
- Total response must be under {max_total_chars} characters
- Each field value: maximum {max_field_chars} characters
- Use concise, technical language
- Focus on accuracy over completeness"""

    if complexity_context:
        base_constraints += f"\n- {complexity_context}"

    field_guidance = ""
    if field_descriptions:
        field_guidance = "\n\nFIELD-SPECIFIC CONSTRAINTS:\n"
        for field, description in field_descriptions.items():
            field_guidance += f"- {field}: {description}\n"

    size_guidance = f"\nEXPECTED SIZE: Target ~{max_total_chars//2} characters total."

    return f"""Provide insights that fit within these STRICT LIMITS:

{base_constraints}{field_guidance}{size_guidance}

Omit optional fields if approaching size limits."""


def calculate_response_limits(
    base_fields: int = 4,
    extended_fields: int = 0,
    complexity_factor: float = 1.0,
    min_chars: int = 2000,
    max_chars: int = 8000,
) -> tuple[int, int]:
    """Calculate appropriate response size limits based on schema complexity.

    Args:
        base_fields: Number of base schema fields (e.g., ai_interpretation, etc.)
        extended_fields: Number of schema-specific fields
        complexity_factor: Multiplier for data complexity (0.5-2.0)
        min_chars: Minimum total response size
        max_chars: Maximum total response size

    Returns:
        Tuple of (max_total_chars, max_field_chars)

    """
    total_fields = base_fields + extended_fields

    # Base calculation: allocate ~150-300 chars per field
    base_allocation = total_fields * 200
    adjusted_total = int(base_allocation * complexity_factor)

    # Clamp to min/max bounds
    max_total_chars = max(min_chars, min(max_chars, adjusted_total))

    # Field size is typically 1/4 to 1/6 of total, depending on field count
    max_field_chars = max(50, min(400, max_total_chars // max(total_fields, 4)))

    return max_total_chars, max_field_chars


# Common field constraint templates
COMMON_FIELD_CONSTRAINTS = {
    "ai_interpretation": "One sentence summary only",
    "ai_confidence": "HIGH/MEDIUM/LOW only",
    "ai_domain_context": "Domain + key characteristics",
    "usage_guidance": "1-2 practical applications",
    "hidden_meaning": "Key insights only",
    "suspicious_patterns": "Main issues only",
    "cross_references": "Essential connections only",
}
