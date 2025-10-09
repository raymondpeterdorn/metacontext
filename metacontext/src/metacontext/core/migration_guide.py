"""Migration guide: From hardcoded YAML templates to schema-driven templates.

This document explains how to migrate from the current hardcoded template system
to the new schema-driven approach that eliminates code duplication.
"""

# BEFORE: Hardcoded YAML Templates (Current Problem)
"""
Problem with current approach in csvhandler_template.yaml:

ai_enrichment:
  fields:
    domain_analysis:
      type: "string"
      description: "Analysis of the domain/subject area this data represents"
    
    data_quality_assessment:
      type: "string"  
      description: "Assessment of data quality, completeness, and potential issues"

Issues:
1. ❌ Schema drift: YAML descriptions can diverge from Pydantic Field descriptions
2. ❌ Maintenance burden: Update field descriptions in multiple places
3. ❌ Single source of truth violation: Which is canonical - YAML or Pydantic?
4. ❌ Brittle: Adding new schema fields requires manual YAML updates
"""

# AFTER: Schema-Driven Templates (Solution)
"""
Solution with SchemaTemplateGenerator:

1. ✅ Single source of truth: Pydantic schema Field descriptions are authoritative
2. ✅ Zero duplication: Templates generated automatically from schemas
3. ✅ Consistent: Field descriptions always match between schema and templates
4. ✅ Extensible: New schema fields automatically appear in templates
5. ✅ Maintainable: Update field descriptions once in Pydantic model

Example DataAIEnrichment schema field:
    domain_analysis: str | None = Field(
        default=None,
        description="The domain this data represents and its key characteristics in a business context.",
    )

Automatically generates template entry:
    "domain_analysis": {
        "type": "string",
        "description": "The domain this data represents and its key characteristics in a business context."
    }
"""

# MIGRATION STEPS


def migration_step_1_update_template_engine():
    """Update ConfigurableTemplateEngine to use generated templates."""
    print("""
    Step 1: Update template_engine.py to use SchemaTemplateGenerator
    
    BEFORE:
    def load_template_config(self, handler_name: str) -> dict[str, Any]:
        template_path = self.templates_dir / f"{handler_name}_template.yaml"
        with open(template_path) as f:
            return yaml.safe_load(f)
    
    AFTER:
    def load_template_config(self, handler_name: str) -> dict[str, Any]:
        # Generate template from schema instead of loading static YAML
        generator = SchemaTemplateGenerator()
        
        if handler_name == "csvhandler":
            return generator.generate_csv_template()
        elif handler_name == "geospatial_raster":
            return generator.generate_geospatial_raster_template()
        elif handler_name == "geospatial_vector":
            return generator.generate_geospatial_vector_template()
        else:
            raise ValueError(f"Unknown handler: {handler_name}")
    """)


def migration_step_2_remove_hardcoded_yaml():
    """Remove hardcoded YAML template files."""
    print("""
    Step 2: Remove hardcoded YAML files
    
    Delete these files (they're now generated from schemas):
    - core/templates/csvhandler_template.yaml
    - core/templates/geospatial_raster_template.yaml  
    - core/templates/geospatial_vector_template.yaml
    
    Keep:
    - schema_template_generator.py (the new generator)
    """)


def migration_step_3_update_schema_descriptions():
    """Improve Pydantic Field descriptions since they're now the single source of truth."""
    print("""
    Step 3: Enhance Pydantic Field descriptions
    
    Since Field descriptions now drive GitHub Copilot prompts, make them more detailed:
    
    BEFORE:
    domain_analysis: str | None = None
    
    AFTER:
    domain_analysis: str | None = Field(
        default=None,
        description="Comprehensive analysis of the business domain this data represents, "
                   "including industry context, typical use cases, and domain-specific terminology."
    )
    """)


def migration_step_4_add_ci_validation():
    """Add CI checks to ensure schema and template consistency."""
    print("""
    Step 4: Add CI validation
    
    Add GitHub Actions check to ensure generated templates stay current:
    
    # .github/workflows/validate-templates.yml
    - name: Validate Templates
      run: |
        python -m metacontext.core.schema_template_generator
        git diff --exit-code  # Fail if generated templates differ
    """)


# BENEFITS ACHIEVED


def benefits_summary():
    """Summary of benefits from migration."""
    return {
        "maintainability": "Single source of truth for field descriptions",
        "consistency": "Templates always match schema definitions",
        "extensibility": "New schema fields automatically appear in templates",
        "reliability": "No more schema drift between YAML and Pydantic",
        "developer_experience": "Update descriptions once, propagates everywhere",
        "code_quality": "Eliminates hardcoded duplication anti-pattern",
    }


if __name__ == "__main__":
    print("🔄 Migration Guide: Hardcoded YAML → Schema-Driven Templates")
    print("=" * 60)

    migration_step_1_update_template_engine()
    migration_step_2_remove_hardcoded_yaml()
    migration_step_3_update_schema_descriptions()
    migration_step_4_add_ci_validation()

    print("\n✅ Benefits achieved:")
    for benefit, description in benefits_summary().items():
        print(f"  • {benefit}: {description}")
