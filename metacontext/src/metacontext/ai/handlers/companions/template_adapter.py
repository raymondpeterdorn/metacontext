"""Template adaptation system for companion providers.

This module provides the CompanionTemplateAdapter class that transforms
API-focused YAML templates into companion-friendly prompts while maintaining
100% schema compatibility.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from metacontext.schemas import (
    ColumnAIEnrichment,
    DataAIEnrichment,
    ModelAIEnrichment,
)
from metacontext.schemas.extensions.geospatial import (
    RasterAIEnrichment,
    VectorAIEnrichment,
)
from metacontext.schemas.extensions.media import MediaAIEnrichment


class CompanionTemplateAdapter:
    """Adapter for converting API templates to companion prompts.

    This class provides modular methods for transforming YAML templates
    designed for API providers into prompts suitable for IDE companions
    while preserving output structure compatibility.
    """

    def __init__(self, templates_base_path: Path | None = None) -> None:
        """Initialize the template adapter.

        Args:
            templates_base_path: Base path to template directory. If None,
                                uses the default location.

        """
        if templates_base_path is None:
            # Default to the templates directory in the package
            self.templates_base_path = (
                Path(__file__).parent.parent.parent / "prompts" / "templates"
            )
        else:
            self.templates_base_path = templates_base_path

    def load_api_template(self, template_path: str) -> dict[str, Any]:
        """Load and parse YAML template from the templates directory.

        Args:
            template_path: Relative path to template from templates base
                          (e.g., 'tabular/column_analysis.yaml')

        Returns:
            Parsed template dictionary with system, instruction, and schema_class

        Raises:
            FileNotFoundError: If template file doesn't exist
            yaml.YAMLError: If template is malformed YAML

        """
        full_path = self.templates_base_path / template_path

        if not full_path.exists():
            msg = f"Template not found: {full_path}"
            raise FileNotFoundError(msg)

        try:
            with full_path.open("r", encoding="utf-8") as file:
                template_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            msg = f"Failed to parse YAML template {template_path}: {e}"
            raise yaml.YAMLError(msg) from e

        # Validate required template fields
        required_fields = ["system", "instruction", "schema_class"]
        missing_fields = [
            field for field in required_fields if field not in template_data
        ]
        if missing_fields:
            msg = f"Template {template_path} missing required fields: {missing_fields}"
            raise ValueError(msg)

        return template_data

    def sanitize_api_elements(self, instruction: str) -> str:
        """Strip JSON schema requirements and API-specific instructions.

        Removes efficiency constraints, character limits, and JSON formatting
        requirements that are not suitable for interactive companions.

        Args:
            instruction: Original template instruction text

        Returns:
            Sanitized instruction with API-specific elements removed

        """
        # Remove character limit constraints
        instruction = re.sub(
            r"Maximum response:.*?characters.*?\n",
            "",
            instruction,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        instruction = re.sub(
            r"Per.*?limit:.*?characters.*?\n",
            "",
            instruction,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        # Remove efficiency requirements sections
        instruction = re.sub(
            r"🎯 EFFICIENCY REQUIREMENTS:.*?(?=\n\s*[🔍⚠️📋]|\n\s*$)",
            "",
            instruction,
            flags=re.DOTALL | re.MULTILINE,
        )

        # Remove JSON-only formatting constraints
        instruction = re.sub(
            r"Never include markdown.*?valid JSON only.*?\n",
            "",
            instruction,
            flags=re.IGNORECASE | re.MULTILINE,
        )
        instruction = re.sub(
            r"output must be \*\*valid JSON only\*\*",
            "output should be well-structured YAML",
            instruction,
            flags=re.IGNORECASE,
        )
        # Also handle variations without asterisks
        instruction = re.sub(
            r"valid JSON only",
            "well-structured YAML",
            instruction,
            flags=re.IGNORECASE,
        )

        # Remove example output sections (we'll replace with YAML format)
        instruction = re.sub(
            r"📋 Example Output Structure:.*?(?=\n\s*[🔍⚠️📋]|\n\s*$)",
            "",
            instruction,
            flags=re.DOTALL | re.MULTILINE,
        )

        # Clean up extra whitespace
        instruction = re.sub(r"\n\s*\n\s*\n", "\n\n", instruction)

        return instruction.strip()

    def convert_schema_to_yaml(self, schema_class_path: str, response_filename: str = "metacontext_response.yaml") -> str:
        """Convert Pydantic schema class to YAML output format template.

        Args:
            schema_class_path: Full import path to schema class
                             (e.g., 'metacontext.schemas.extensions.tabular.ColumnAIEnrichment')
            response_filename: Filename for the response file

        Returns:
            YAML format template showing expected output structure with validation

        Raises:
            ImportError: If schema class cannot be imported
            AttributeError: If schema class doesn't have expected attributes

        """
        # Map known schema classes to avoid dynamic imports
        schema_class_map = {
            "metacontext.schemas.extensions.tabular.ColumnAIEnrichment": ColumnAIEnrichment,
            "metacontext.schemas.extensions.tabular.DataAIEnrichment": DataAIEnrichment,
            "metacontext.schemas.extensions.models.ModelAIEnrichment": ModelAIEnrichment,
            "metacontext.schemas.extensions.media.MediaAIEnrichment": MediaAIEnrichment,
            "metacontext.schemas.extensions.geospatial.RasterAIEnrichment": RasterAIEnrichment,
            "metacontext.schemas.extensions.geospatial.VectorAIEnrichment": VectorAIEnrichment,
        }

        schema_class = schema_class_map.get(schema_class_path)
        if schema_class is None:
            msg = f"Unknown schema class: {schema_class_path}"
            raise ImportError(msg)

        # Generate comprehensive YAML template with validation
        return self._generate_comprehensive_yaml_template(
            schema_class,
            schema_class_path,
            response_filename,
        )

    def _generate_comprehensive_yaml_template(
        self,
        schema_class: type[BaseModel],
        schema_class_path: str,
        response_filename: str = "metacontext_response.yaml",
    ) -> str:
        """Generate comprehensive YAML template with validation and structure requirements.

        Args:
            schema_class: The Pydantic model class
            schema_class_path: Full import path for reference

        Returns:
            Complete YAML template with validation instructions

        """
        yaml_template = f"""# ⚠️ CRITICAL: Output Structure Preservation for {schema_class.__name__}
# This response MUST match the exact metacontext structure.
# Save this file as: {response_filename}

# 🏗️ Schema: {schema_class_path}
# 📋 All field names and types must be preserved exactly as specified below.

"""

        # Get field information from the Pydantic model
        fields = schema_class.model_fields
        example_data = {}

        # Add detailed field documentation
        yaml_template += "# 📖 Field Descriptions and Requirements:\n"
        for field_name, field_info in fields.items():
            field_description = getattr(
                field_info,
                "description",
                "No description available",
            )
            field_type = field_info.annotation

            # Get more detailed type information
            type_hint = self._get_detailed_type_hint(field_type)

            # Check if field is required or optional
            required_status = "REQUIRED" if field_info.is_required() else "OPTIONAL"

            yaml_template += f"# • {field_name} ({type_hint}) - {required_status}\n"
            yaml_template += f"#   {field_description}\n"

            # Generate appropriate example values with proper types
            example_data[field_name] = self._generate_typed_example_value(
                field_name,
                field_type,
                field_info,
            )

        yaml_template += "\n# 🎯 VALIDATION REQUIREMENTS:\n"
        yaml_template += "# 1. ALL field names must match exactly (case-sensitive)\n"
        yaml_template += "# 2. Data types must conform to the specifications above\n"
        yaml_template += "# 3. Required fields must not be null or missing\n"
        yaml_template += "# 4. Optional fields can be null or omitted\n"
        yaml_template += "# 5. List fields must contain appropriate item types\n"
        yaml_template += (
            "# 6. Dict fields must have string keys and appropriate values\n"
        )
        yaml_template += "\n# 📝 EXAMPLE OUTPUT (replace with your analysis):\n\n"

        # Generate the actual YAML structure
        yaml_template += yaml.dump(
            example_data,
            default_flow_style=False,
            sort_keys=False,
            allow_unicode=True,
            width=80,
        )

        yaml_template += "\n# ✅ VALIDATION CHECKLIST before saving:\n"
        yaml_template += "# □ All required fields are present and not null\n"
        yaml_template += "# □ Field names match schema exactly\n"
        yaml_template += "# □ Data types are correct (strings, lists, dicts, etc.)\n"
        yaml_template += f"# □ File is saved as '{response_filename}'\n"
        yaml_template += "# □ YAML syntax is valid (no parsing errors)\n"

        return yaml_template

    def _get_detailed_type_hint(self, field_type: Any) -> str:
        """Get detailed type hint string for field documentation.

        Args:
            field_type: The field type annotation

        Returns:
            Human-readable type hint string

        """
        type_str = str(field_type)

        # Clean up common type patterns
        if "typing.Union" in type_str and "NoneType" in type_str:
            # Optional field
            base_type = type_str.split(",")[0].split("[")[1]
            return f"Optional[{base_type.split('.')[-1]}]"
        if "list[" in type_str.lower():
            inner_type = type_str.split("[")[1].split("]")[0]
            return f"List[{inner_type.split('.')[-1]}]"
        if "dict[" in type_str.lower():
            return "Dict[str, Any]"
        if "str" in type_str:
            return "str"
        if "int" in type_str:
            return "int"
        if "float" in type_str:
            return "float"
        if "bool" in type_str:
            return "bool"
        # Extract the class name from full path
        return type_str.split(".")[-1].replace("'", "").replace(">", "")

    def _generate_typed_example_value(
        self,
        field_name: str,
        field_type: Any,
        field_info: Any,
    ) -> Any:
        """Generate properly typed example values for YAML template.

        Args:
            field_name: Name of the field
            field_type: Type annotation of the field
            field_info: Pydantic field info

        Returns:
            Appropriately typed example value

        """
        type_str = str(field_type)

        # Handle optional fields (Union with None)
        if "typing.Union" in type_str and "NoneType" in type_str:
            # For optional fields, show null as default but indicate they can have values
            if field_info.is_required():
                # Required field, generate appropriate default
                base_type = type_str.split(",")[0]
                return self._generate_example_for_type(field_name, base_type)
            # Optional field, can be null
            return None

        # Handle specific field names with special values
        if field_name == "ai_confidence":
            return "HIGH"
        if field_name == "ai_interpretation":
            return "Your comprehensive analysis and interpretation"
        if field_name == "ai_domain_context":
            return "Domain-specific context and background"
        if field_name == "usage_guidance":
            return "Practical recommendations for usage"
        if field_name == "hidden_meaning":
            return "Hidden insights or business logic discovered"
        if field_name == "suspicious_patterns":
            return ["pattern1", "pattern2"]
        if field_name == "cross_references":
            return {"element1": "explanation1", "element2": "explanation2"}
        if field_name == "detective_insights":
            return "Key forensic insights and discoveries"

        return self._generate_example_for_type(field_name, type_str)

    def _generate_example_for_type(self, field_name: str, type_str: str) -> Any:
        """Generate example value based on type string.

        Args:
            field_name: Name of the field for context
            type_str: String representation of the type

        Returns:
            Example value of the appropriate type

        """
        if "list[" in type_str.lower() or "List[" in type_str:
            if "str" in type_str:
                return [f"example_{field_name}_1", f"example_{field_name}_2"]
            return ["item1", "item2"]
        if "dict[" in type_str.lower() or "Dict[" in type_str:
            return {
                "key1": f"value_for_{field_name}_1",
                "key2": f"value_for_{field_name}_2",
            }
        if "str" in type_str:
            return f"Your analysis for {field_name}"
        if "int" in type_str:
            return 0
        if "float" in type_str:
            return 0.0
        if "bool" in type_str:
            return True
        # Default to string representation
        return f"Value for {field_name}"

    def validate_companion_response(
        self,
        response_yaml_path: Path,
        expected_schema_class_path: str,
    ) -> tuple[bool, list[str]]:
        """Validate a companion response YAML file against expected schema.

        Args:
            response_yaml_path: Path to the YAML response file
            expected_schema_class_path: Expected schema class path

        Returns:
            Tuple of (is_valid, list_of_errors)

        """
        errors = []

        # Check if file exists
        if not response_yaml_path.exists():
            return False, [f"Response file not found: {response_yaml_path}"]

        try:
            # Load YAML content
            with response_yaml_path.open("r", encoding="utf-8") as file:
                response_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            return False, [f"Invalid YAML syntax: {e}"]
        except Exception as e:
            return False, [f"Error reading file: {e}"]

        if not isinstance(response_data, dict):
            return False, ["Response must be a YAML dictionary/object"]

        # Get expected schema class
        schema_class_map = {
            "metacontext.schemas.extensions.tabular.ColumnAIEnrichment": ColumnAIEnrichment,
            "metacontext.schemas.extensions.tabular.DataAIEnrichment": DataAIEnrichment,
            "metacontext.schemas.extensions.models.ModelAIEnrichment": ModelAIEnrichment,
            "metacontext.schemas.extensions.media.MediaAIEnrichment": MediaAIEnrichment,
            "metacontext.schemas.extensions.geospatial.RasterAIEnrichment": RasterAIEnrichment,
            "metacontext.schemas.extensions.geospatial.VectorAIEnrichment": VectorAIEnrichment,
        }

        schema_class = schema_class_map.get(expected_schema_class_path)
        if schema_class is None:
            return False, [f"Unknown schema class: {expected_schema_class_path}"]

        # Validate against Pydantic schema
        try:
            validated_instance = schema_class(**response_data)
            # If we get here, validation passed
            return True, []
        except Exception as e:
            errors.append(f"Schema validation failed: {e}")

            # Try to provide more specific error details
            fields = schema_class.model_fields
            for field_name, field_info in fields.items():
                if field_name not in response_data and field_info.is_required():
                    errors.append(f"Missing required field: {field_name}")
                elif field_name in response_data:
                    # Check basic type compatibility
                    value = response_data[field_name]
                    expected_type = field_info.annotation
                    if not self._check_basic_type_compatibility(value, expected_type):
                        errors.append(
                            f"Type mismatch for {field_name}: expected {expected_type}, got {type(value)}",
                        )

            return False, errors

    def _check_basic_type_compatibility(self, value: Any, expected_type: Any) -> bool:
        """Basic type compatibility check for validation.

        Args:
            value: The actual value
            expected_type: The expected type annotation

        Returns:
            True if types are compatible

        """
        if value is None:
            # None is compatible with Optional types
            return "Union" in str(expected_type) and "NoneType" in str(expected_type)

        type_str = str(expected_type).lower()
        value_type = type(value)

        if ("str" in type_str and value_type == str) or (
            "int" in type_str and value_type == int
        ):
            return True
        if (
            ("float" in type_str and value_type in (int, float))
            or ("bool" in type_str and value_type == bool)
            or ("list" in type_str and value_type == list)
            or ("dict" in type_str and value_type == dict)
        ):
            return True

        return False

    def create_response_validation_instructions(self, schema_class_path: str) -> str:
        """Create detailed validation instructions for companion responses.

        Args:
            schema_class_path: Schema class path for validation

        Returns:
            Detailed validation instruction text

        """
        schema_class_map = {
            "metacontext.schemas.extensions.tabular.ColumnAIEnrichment": ColumnAIEnrichment,
            "metacontext.schemas.extensions.tabular.DataAIEnrichment": DataAIEnrichment,
            "metacontext.schemas.extensions.models.ModelAIEnrichment": ModelAIEnrichment,
            "metacontext.schemas.extensions.media.MediaAIEnrichment": MediaAIEnrichment,
            "metacontext.schemas.extensions.geospatial.RasterAIEnrichment": RasterAIEnrichment,
            "metacontext.schemas.extensions.geospatial.VectorAIEnrichment": VectorAIEnrichment,
        }

        schema_class = schema_class_map.get(schema_class_path)
        if schema_class is None:
            return f"# ERROR: Unknown schema class: {schema_class_path}"

        instructions = f"""
🔍 RESPONSE VALIDATION INSTRUCTIONS FOR {schema_class.__name__}

📋 CRITICAL REQUIREMENTS:
1. File must be saved as: metacontext_response.yaml
2. Content must be valid YAML (no syntax errors)
3. Root structure must be a dictionary/object (not list or string)
4. ALL field names must match schema exactly (case-sensitive)

📖 REQUIRED FIELDS:
"""

        fields = schema_class.model_fields
        required_fields = [name for name, info in fields.items() if info.is_required()]
        optional_fields = [
            name for name, info in fields.items() if not info.is_required()
        ]

        for field_name in required_fields:
            field_info = fields[field_name]
            field_type = self._get_detailed_type_hint(field_info.annotation)
            description = getattr(field_info, "description", "No description")
            instructions += f"• {field_name}: {field_type} - {description[:80]}...\n"

        instructions += f"\n📝 OPTIONAL FIELDS ({len(optional_fields)} total):\n"
        for field_name in optional_fields[:3]:  # Show first 3 as examples
            field_info = fields[field_name]
            field_type = self._get_detailed_type_hint(field_info.annotation)
            instructions += f"• {field_name}: {field_type} (can be null/omitted)\n"

        if len(optional_fields) > 3:
            instructions += (
                f"• ... and {len(optional_fields) - 3} more optional fields\n"
            )

        instructions += """
⚠️ COMMON VALIDATION ERRORS TO AVOID:
• Typos in field names (check spelling and case)
• Wrong data types (strings vs lists vs dicts)
• Missing required fields
• Invalid YAML syntax (check indentation)
• Null values for required fields

✅ BEFORE SUBMITTING, VERIFY:
• File is saved as metacontext_response.yaml
• YAML parses without errors
• All required fields are present and not null
• Field names match schema exactly
• Data types are appropriate for each field
"""

        return instructions

    def parse_companion_response(
        self,
        response_yaml_path: Path,
    ) -> tuple[dict[str, Any] | None, list[str]]:
        """Parse companion response YAML file and return data or errors.

        This method loads and parses a YAML response file from a companion,
        providing detailed error reporting for any parsing issues.

        Args:
            response_yaml_path: Path to the YAML response file

        Returns:
            Tuple of (parsed_data, list_of_errors). If parsing fails,
            parsed_data will be None and errors will contain details.

        """
        errors = []

        # Check if file exists
        if not response_yaml_path.exists():
            return None, [f"Response file not found: {response_yaml_path}"]

        try:
            # Load YAML content
            with response_yaml_path.open("r", encoding="utf-8") as file:
                response_data = yaml.safe_load(file)
        except yaml.YAMLError as e:
            return None, [f"Invalid YAML syntax: {e}"]
        except UnicodeDecodeError as e:
            return None, [f"File encoding error: {e}"]
        except Exception as e:
            return None, [f"Error reading file: {e}"]

        # Validate basic structure
        if response_data is None:
            return None, ["YAML file is empty or contains only null value"]

        if not isinstance(response_data, dict):
            return None, [
                f"Response must be a YAML dictionary/object, got {type(response_data).__name__}",
            ]

        # Check for empty dictionary
        if not response_data:
            return None, ["Response dictionary is empty"]

        return response_data, []

    def validate_response_structure(
        self,
        response_data: dict[str, Any],
        expected_schema_class_path: str,
    ) -> tuple[bool, list[str]]:
        """Validate response data structure against expected metacontext schema.

        This method performs comprehensive validation of response data against
        the expected Pydantic schema, providing detailed error reporting.

        Args:
            response_data: Parsed YAML response data
            expected_schema_class_path: Expected schema class path

        Returns:
            Tuple of (is_valid, list_of_errors)

        """
        errors = []

        # Get expected schema class
        schema_class_map = {
            "metacontext.schemas.extensions.tabular.ColumnAIEnrichment": ColumnAIEnrichment,
            "metacontext.schemas.extensions.tabular.DataAIEnrichment": DataAIEnrichment,
            "metacontext.schemas.extensions.models.ModelAIEnrichment": ModelAIEnrichment,
            "metacontext.schemas.extensions.media.MediaAIEnrichment": MediaAIEnrichment,
            "metacontext.schemas.extensions.geospatial.RasterAIEnrichment": RasterAIEnrichment,
            "metacontext.schemas.extensions.geospatial.VectorAIEnrichment": VectorAIEnrichment,
        }

        schema_class = schema_class_map.get(expected_schema_class_path)
        if schema_class is None:
            return False, [f"Unknown schema class: {expected_schema_class_path}"]

        # Validate required fields presence
        fields = schema_class.model_fields
        required_fields = [name for name, info in fields.items() if info.is_required()]

        for field_name in required_fields:
            if field_name not in response_data:
                errors.append(f"Missing required field: {field_name}")
            elif response_data[field_name] is None:
                errors.append(f"Required field cannot be null: {field_name}")

        # Check for unexpected fields (strict validation)
        expected_fields = set(fields.keys())
        actual_fields = set(response_data.keys())
        unexpected_fields = actual_fields - expected_fields

        if unexpected_fields:
            errors.append(
                f"Unexpected fields found: {', '.join(sorted(unexpected_fields))}",
            )

        # Basic type validation for present fields
        for field_name, field_info in fields.items():
            if field_name in response_data and response_data[field_name] is not None:
                value = response_data[field_name]
                expected_type = field_info.annotation
                if not self._check_basic_type_compatibility(value, expected_type):
                    errors.append(
                        f"Type mismatch for {field_name}: expected {expected_type}, got {type(value).__name__}",
                    )

        # If basic validation passed, try Pydantic validation
        if not errors:
            try:
                schema_class(**response_data)
                return True, []
            except Exception as e:
                errors.append(f"Schema validation failed: {e}")

        return False, errors

    def convert_yaml_to_pydantic(
        self,
        response_data: dict[str, Any],
        schema_class_path: str,
    ) -> tuple[BaseModel | None, list[str]]:
        """Convert validated YAML data back to Pydantic model instance.

        This method takes validated response data and creates a proper
        Pydantic model instance for integration with the metacontext system.

        Args:
            response_data: Validated YAML response data
            schema_class_path: Schema class path for instantiation

        Returns:
            Tuple of (pydantic_instance, list_of_errors). If conversion fails,
            instance will be None and errors will contain details.

        """
        errors = []

        # Get schema class
        schema_class_map = {
            "metacontext.schemas.extensions.tabular.ColumnAIEnrichment": ColumnAIEnrichment,
            "metacontext.schemas.extensions.tabular.DataAIEnrichment": DataAIEnrichment,
            "metacontext.schemas.extensions.models.ModelAIEnrichment": ModelAIEnrichment,
            "metacontext.schemas.extensions.media.MediaAIEnrichment": MediaAIEnrichment,
            "metacontext.schemas.extensions.geospatial.RasterAIEnrichment": RasterAIEnrichment,
            "metacontext.schemas.extensions.geospatial.VectorAIEnrichment": VectorAIEnrichment,
        }

        schema_class = schema_class_map.get(schema_class_path)
        if schema_class is None:
            return None, [f"Unknown schema class: {schema_class_path}"]

        try:
            # Create Pydantic instance with comprehensive validation
            instance = schema_class(**response_data)
            return instance, []
        except Exception as e:
            errors.append(f"Failed to create Pydantic instance: {e}")

            # Provide additional debugging information
            try:
                # Try to identify specific field issues
                fields = schema_class.model_fields
                for field_name, field_info in fields.items():
                    if field_name in response_data:
                        try:
                            # Attempt individual field validation
                            value = response_data[field_name]
                            if hasattr(field_info, "validate"):
                                field_info.validate(value)
                        except Exception as field_error:
                            errors.append(
                                f"Field '{field_name}' validation error: {field_error}",
                            )
            except Exception:
                # If we can't provide detailed field errors, just use the main error
                pass

            return None, errors

    def generate_companion_prompt(
        self,
        template_data: dict[str, Any],
        context_variables: dict[str, Any] | None = None,
        response_filename: str | None = None,
        include_workspace_context: bool = False,
    ) -> str:
        """Assemble final companion prompt from adapted template components.

        Args:
            template_data: Parsed template with system, instruction, schema_class
            context_variables: Variables to substitute in template (e.g., file_name)
            response_filename: Custom response filename (if None, uses default)
            include_workspace_context: Whether to include workspace analysis guidance

        Returns:
            Complete companion prompt ready for IDE integration

        """
        if context_variables is None:
            context_variables = {}

        # Calculate response filename if not provided
        if response_filename is None:
            # Extract source filename and create response filename
            source_filename = context_variables.get("file_name", "unknown")
            if source_filename and source_filename != "unknown":
                # Remove extension and add _metacontext.yaml
                base_name = source_filename.rsplit(".", 1)[0] if "." in source_filename else source_filename
                response_filename = f"{base_name}_metacontext.yaml"
            else:
                response_filename = "metacontext_response.yaml"

        # Extract template components
        system_message = template_data.get("system", "")
        instruction = template_data.get("instruction", "")
        schema_class_path = template_data.get("schema_class", "")

        # Sanitize API-specific elements
        clean_instruction = self.sanitize_api_elements(instruction)

        # Add workspace context if requested
        if include_workspace_context:
            workspace_enhancement = self._generate_workspace_enhancement_instructions()
            clean_instruction += workspace_enhancement

        # Generate YAML output format
        yaml_format = self.convert_schema_to_yaml(schema_class_path, response_filename)

        # Substitute context variables in system and instruction
        for var_name, var_value in context_variables.items():
            placeholder = f"${{{var_name}}}"
            system_message = system_message.replace(placeholder, str(var_value))
            clean_instruction = clean_instruction.replace(placeholder, str(var_value))

        # Assemble final prompt
        final_prompt = f"""{system_message}

{clean_instruction}

{yaml_format}

📁 IMPORTANT: Save your response as `{response_filename}` in the same directory as the source file.
The response will be automatically processed and integrated into the metacontext system.
"""

        return final_prompt

    def _generate_workspace_enhancement_instructions(self) -> str:
        """Generate minimal workspace context instructions for companions.

        Returns:
            Simple workspace context instructions when relevant

        """
        return """

💡 WORKSPACE CONTEXT:
- Consider any related files or configurations that might provide additional context
- Look for patterns or conventions used elsewhere in the project
- Use your IDE's code intelligence to understand relationships when relevant
"""

    # Helper functions for specific template types

    def extract_workspace_context(self, target_file: Path) -> dict[str, Any]:
        """Extract relevant workspace context for companion analysis.

        Phase 2.4 feature: Provides structured workspace context that companions
        can use for deeper analysis beyond just the target file.

        Args:
            target_file: The primary file being analyzed

        Returns:
            Dictionary containing workspace context information

        """
        context = {
            "target_file": str(target_file),
            "project_structure": self._analyze_project_structure(target_file),
            "related_files": self._find_related_files(target_file),
            "dependencies": self._extract_dependencies(target_file),
            "workspace_patterns": self._identify_workspace_patterns(target_file),
        }
        return context

    def _analyze_project_structure(self, target_file: Path) -> dict[str, Any]:
        """Analyze the overall project structure for context.

        Args:
            target_file: The file being analyzed

        Returns:
            Project structure information

        """
        # Find project root (look for common root indicators)
        project_root = self._find_project_root(target_file)

        structure = {
            "project_root": str(project_root)
            if project_root
            else str(target_file.parent),
            "config_files": [],
            "documentation": [],
            "test_directories": [],
            "package_structure": {},
        }

        if project_root:
            # Look for configuration files
            config_patterns = [
                "pyproject.toml",
                "setup.py",
                "requirements.txt",
                "package.json",
                "Cargo.toml",
                "go.mod",
                ".gitignore",
                "Makefile",
                "Dockerfile",
            ]
            for pattern in config_patterns:
                config_file = project_root / pattern
                if config_file.exists():
                    structure["config_files"].append(str(config_file))

            # Look for documentation
            doc_patterns = ["README*", "CHANGELOG*", "docs/", "documentation/"]
            for pattern in doc_patterns:
                matches = list(project_root.glob(pattern))
                structure["documentation"].extend([str(m) for m in matches])

            # Look for test directories
            test_patterns = ["tests/", "test/", "*_test.py", "test_*.py"]
            for pattern in test_patterns:
                matches = list(project_root.glob(f"**/{pattern}"))
                structure["test_directories"].extend([str(m) for m in matches])

        return structure

    def _find_related_files(self, target_file: Path) -> list[str]:
        """Find files related to the target file through various relationships.

        Args:
            target_file: The file being analyzed

        Returns:
            List of related file paths

        """
        related = []

        # Look for files with similar names
        stem = target_file.stem
        parent = target_file.parent

        # Test files
        test_patterns = [
            f"test_{stem}.py",
            f"{stem}_test.py",
            f"tests/test_{stem}.py",
            f"test/test_{stem}.py",
        ]

        for pattern in test_patterns:
            test_file = parent / pattern
            if test_file.exists():
                related.append(str(test_file))

        # Configuration files in same directory
        config_patterns = [f"{stem}.yaml", f"{stem}.json", f"{stem}.toml"]
        for pattern in config_patterns:
            config_file = parent / pattern
            if config_file.exists():
                related.append(str(config_file))

        return related

    def _extract_dependencies(self, target_file: Path) -> dict[str, list[str]]:
        """Extract dependency information from the target file.

        Args:
            target_file: The file being analyzed

        Returns:
            Dictionary with import and dependency information

        """
        dependencies = {
            "imports": [],
            "local_imports": [],
            "external_dependencies": [],
        }

        try:
            if target_file.suffix == ".py":
                content = target_file.read_text(encoding="utf-8")

                # Extract import statements
                import_pattern = r"^(?:from\s+([\w.]+)\s+)?import\s+([\w.,\s*]+)"
                imports = re.findall(import_pattern, content, re.MULTILINE)

                for from_module, import_items in imports:
                    if from_module:
                        full_import = f"from {from_module} import {import_items}"
                    else:
                        full_import = f"import {import_items}"

                    dependencies["imports"].append(full_import.strip())

                    # Classify as local vs external
                    module = (
                        from_module
                        if from_module
                        else import_items.split(",")[0].strip()
                    )
                    if module.startswith(".") or "metacontext" in module:
                        dependencies["local_imports"].append(module)
                    else:
                        dependencies["external_dependencies"].append(module)

        except Exception:
            # If we can't read the file, return empty dependencies
            pass

        return dependencies

    def _identify_workspace_patterns(self, target_file: Path) -> dict[str, Any]:
        """Identify common patterns and conventions in the workspace.

        Args:
            target_file: The file being analyzed

        Returns:
            Dictionary of identified patterns and conventions

        """
        patterns = {
            "naming_conventions": [],
            "architectural_patterns": [],
            "testing_patterns": [],
            "documentation_patterns": [],
        }

        project_root = self._find_project_root(target_file)
        if not project_root:
            return patterns

        # Look for common architectural patterns
        if (project_root / "src").exists():
            patterns["architectural_patterns"].append("src_layout")
        if (project_root / "lib").exists():
            patterns["architectural_patterns"].append("lib_layout")
        if (project_root / "app").exists():
            patterns["architectural_patterns"].append("app_layout")

        # Look for testing patterns
        if (project_root / "tests").exists():
            patterns["testing_patterns"].append("dedicated_test_directory")
        if list(project_root.glob("**/test_*.py")):
            patterns["testing_patterns"].append("test_prefix_convention")
        if list(project_root.glob("**/*_test.py")):
            patterns["testing_patterns"].append("test_suffix_convention")

        # Look for documentation patterns
        if (project_root / "docs").exists():
            patterns["documentation_patterns"].append("dedicated_docs_directory")
        if list(project_root.glob("README*")):
            patterns["documentation_patterns"].append("readme_documentation")
        if list(project_root.glob("**/*.md")):
            patterns["documentation_patterns"].append("markdown_documentation")

        return patterns

    def _find_project_root(self, target_file: Path) -> Path | None:
        """Find the project root directory by looking for common indicators.

        Args:
            target_file: Starting file to search from

        Returns:
            Project root path if found, None otherwise

        """
        current = target_file.parent

        # Common project root indicators
        root_indicators = [
            ".git",
            ".gitignore",
            "pyproject.toml",
            "setup.py",
            "package.json",
            "Cargo.toml",
            "go.mod",
            "Makefile",
        ]

        # Search up the directory tree
        for _ in range(10):  # Limit search depth
            for indicator in root_indicators:
                if (current / indicator).exists():
                    return current

            parent = current.parent
            if parent == current:  # Reached filesystem root
                break
            current = parent

        return None

    def create_tabular_companion_prompt(
        self,
        file_name: str,
        project_summary: str = "",
        code_summary: str = "",
        columns_data: str = "",
        semantic_column_knowledge: str = "",
        enable_workspace_context: bool = True,
    ) -> str:
        """Create companion prompt for tabular analysis.

        Phase 2.4 Enhancement: Now includes comprehensive workspace context
        for deeper analysis leveraging IDE workspace awareness.

        Args:
            file_name: Name of the file being analyzed
            project_summary: Summary of the project context
            code_summary: Summary of relevant code
            columns_data: Column information to analyze
            semantic_column_knowledge: Extracted semantic knowledge
            enable_workspace_context: Whether to include workspace context analysis

        Returns:
            Complete tabular analysis prompt for companions

        """
        template_data = self.load_api_template("tabular/column_analysis.yaml")

        # Build context variables
        context_vars = {
            "file_name": file_name,
            "project_summary": project_summary,
            "code_summary": code_summary,
            "columns_data": columns_data,
            "semantic_column_knowledge": semantic_column_knowledge,
            "field_descriptions": "Analyze each column for semantic meaning, quality, and relationships",
        }

        # Phase 2.4: Add workspace context if enabled
        if enable_workspace_context:
            try:
                target_file = Path(file_name)
                workspace_context = self.extract_workspace_context(target_file)
                context_vars["workspace_context"] = self._format_workspace_context(
                    workspace_context,
                )
            except Exception:
                # If workspace context extraction fails, continue without it
                context_vars["workspace_context"] = (
                    "Workspace context unavailable - analyze based on file content only"
                )

        return self.generate_companion_prompt(template_data, context_vars)

    def _format_workspace_context(self, workspace_context: dict[str, Any]) -> str:
        """Format workspace context information for inclusion in companion prompts.

        Args:
            workspace_context: Workspace context dictionary from extract_workspace_context

        Returns:
            Formatted string representation of workspace context

        """
        formatted_context = "\n🏗️ WORKSPACE CONTEXT:\n"

        # Project structure
        if workspace_context.get("project_structure"):
            structure = workspace_context["project_structure"]
            formatted_context += (
                f"\n📂 Project Root: {structure.get('project_root', 'Unknown')}\n"
            )

            if structure.get("config_files"):
                formatted_context += (
                    f"⚙️ Configuration Files: {', '.join(structure['config_files'])}\n"
                )

            if structure.get("documentation"):
                formatted_context += f"📚 Documentation: {', '.join(structure['documentation'][:3])}{'...' if len(structure['documentation']) > 3 else ''}\n"

            if structure.get("test_directories"):
                formatted_context += f"🧪 Test Files: {', '.join(structure['test_directories'][:3])}{'...' if len(structure['test_directories']) > 3 else ''}\n"

        # Related files
        if workspace_context.get("related_files"):
            related = workspace_context["related_files"]
            if related:
                formatted_context += f"\n🔗 Related Files: {', '.join(related)}\n"

        # Dependencies
        if workspace_context.get("dependencies"):
            deps = workspace_context["dependencies"]
            if deps.get("local_imports"):
                formatted_context += f"\n🏠 Local Dependencies: {', '.join(deps['local_imports'][:5])}{'...' if len(deps['local_imports']) > 5 else ''}\n"
            if deps.get("external_dependencies"):
                formatted_context += f"📦 External Dependencies: {', '.join(deps['external_dependencies'][:5])}{'...' if len(deps['external_dependencies']) > 5 else ''}\n"

        # Workspace patterns
        if workspace_context.get("workspace_patterns"):
            patterns = workspace_context["workspace_patterns"]
            pattern_summary = []
            for category, items in patterns.items():
                if items:
                    pattern_summary.append(f"{category}: {', '.join(items)}")

            if pattern_summary:
                formatted_context += (
                    f"\n🎯 Detected Patterns: {'; '.join(pattern_summary)}\n"
                )

        formatted_context += "\n💡 Use this context to provide deeper insights about the file's role in the project.\n"

        return formatted_context

    def create_model_companion_prompt(
        self,
        model_filename: str,
        **kwargs: str,
    ) -> str:
        """Create companion prompt for model analysis.

        Args:
            model_filename: Name of the model file
            **kwargs: Additional context variables (model_type, data_info, etc.)

        Returns:
            Complete model analysis prompt for companions

        """
        template_data = self.load_api_template("model/model_analysis.yaml")
        context_vars = {
            "model_filename": model_filename,
            "model_type": kwargs.get("model_type", ""),
            "data_info": kwargs.get("data_info", ""),
            "project_context": kwargs.get("project_context", ""),
            "schema_hint": kwargs.get("schema_hint", ""),
            "training_script_content": kwargs.get("training_script_content", ""),
        }
        return self.generate_companion_prompt(template_data, context_vars)

    def create_media_companion_prompt(
        self,
        file_name: str,
        **kwargs: str,
    ) -> str:
        """Create companion prompt for media analysis.

        Args:
            file_name: Name of the media file
            **kwargs: Additional context variables (media_type, file_size, etc.)

        Returns:
            Complete media analysis prompt for companions

        """
        template_data = self.load_api_template("media/media_analysis.yaml")
        context_vars = {
            "file_name": file_name,
            "media_type": kwargs.get("media_type", ""),
            "file_size": kwargs.get("file_size", ""),
            "dimensions": kwargs.get("dimensions", ""),
            "duration": kwargs.get("duration", ""),
            "format": kwargs.get("format", ""),
            "creation_date": kwargs.get("creation_date", ""),
            "technical_metadata": kwargs.get("technical_metadata", ""),
            "project_summary": kwargs.get("project_summary", ""),
            "schema_hint": kwargs.get("schema_hint", ""),
        }
        return self.generate_companion_prompt(template_data, context_vars)

    def create_geospatial_vector_companion_prompt(
        self,
        file_name: str,
        **kwargs: str,
    ) -> str:
        """Create companion prompt for geospatial vector analysis.

        Args:
            file_name: Name of the geospatial file
            **kwargs: Additional context variables (geometry_type, feature_count, etc.)

        Returns:
            Complete vector analysis prompt for companions

        """
        template_data = self.load_api_template("geospatial/vector_analysis.yaml")
        context_vars = {
            "file_name": file_name,
            "geometry_type": kwargs.get("geometry_type", ""),
            "feature_count": kwargs.get("feature_count", ""),
            "crs": kwargs.get("crs", ""),
            "spatial_bounds": kwargs.get("spatial_bounds", ""),
            "attribute_fields": kwargs.get("attribute_fields", ""),
            "project_summary": kwargs.get("project_summary", ""),
            "schema_hint": kwargs.get("schema_hint", ""),
        }
        return self.generate_companion_prompt(template_data, context_vars)

    def create_geospatial_raster_companion_prompt(
        self,
        file_name: str,
        **kwargs: str,
    ) -> str:
        """Create companion prompt for geospatial raster analysis.

        Args:
            file_name: Name of the raster file
            **kwargs: Additional context variables (width, height, band_count, etc.)

        Returns:
            Complete raster analysis prompt for companions

        """
        template_data = self.load_api_template("geospatial/raster_analysis.yaml")
        context_vars = {
            "file_name": file_name,
            "width": kwargs.get("width", ""),
            "height": kwargs.get("height", ""),
            "band_count": kwargs.get("band_count", ""),
            "crs": kwargs.get("crs", ""),
            "pixel_resolution": kwargs.get("pixel_resolution", ""),
            "data_type": kwargs.get("data_type", ""),
            "nodata_value": kwargs.get("nodata_value", ""),
            "project_summary": kwargs.get("project_summary", ""),
            "schema_hint": kwargs.get("schema_hint", ""),
        }
        return self.generate_companion_prompt(template_data, context_vars)
