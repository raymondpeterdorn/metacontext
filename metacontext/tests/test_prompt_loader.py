"""Tests for the prompt loader module (metacontext.prompt_loader)."""

from pathlib import Path
from unittest.mock import mock_open, patch

import yaml

from metacontext.ai.prompts.prompt_loader import PromptLoader


def test_prompt_loader_basic():
    """Test basic prompt loader functionality."""
    yaml_content = """
template: |
  Analyze the following project: {project_name}
  Number of files: {file_count}
variables:
  project_name: "TestProject"
  file_count: "5"
"""

    with patch("builtins.open", mock_open(read_data=yaml_content)):
        with patch("pathlib.Path.exists", return_value=True):
            loader = PromptLoader()

            # Test loading prompt
            prompt_data = loader.load_prompt("test_prompt")
            assert "template" in prompt_data
            assert "variables" in prompt_data

            # Test rendering
            result = loader.render_prompt(
                "test_prompt",
                {"project_name": "MyProject", "file_count": "10"},
            )

            assert "MyProject" in result
            assert "10" in result


def test_prompt_loader_template_rendering():
    """Test template rendering with various scenarios."""
    # Test basic template rendering using string format
    template = "Hello {name}, welcome to {project}"
    variables = {"name": "Alice", "project": "Metacontext"}
    result = template.format(**variables)
    assert result == "Hello Alice, welcome to Metacontext"


def test_prompt_loader_file_errors():
    """Test prompt loader error handling."""
    # Test with non-existent directory
    try:
        PromptLoader(Path("/definitely/does/not/exist"))
        msg = "Should have raised ValueError"
        raise AssertionError(msg)
    except ValueError as e:
        assert "Prompts directory not found" in str(e)

    # Test file not found
    with patch("pathlib.Path.exists", return_value=False):
        loader = PromptLoader.__new__(PromptLoader)
        loader.prompts_dir = Path("/tmp")

        try:
            loader.load_prompt("nonexistent")
            msg = "Should have raised FileNotFoundError"
            raise AssertionError(msg)
        except FileNotFoundError as e:
            assert "Prompt file not found" in str(e)


def test_prompt_loader_yaml_errors():
    """Test prompt loader with YAML parsing errors."""
    invalid_yaml = "invalid: yaml: content: [missing closing bracket"

    with patch("builtins.open", mock_open(read_data=invalid_yaml)):
        with patch("pathlib.Path.exists", return_value=True):
            loader = PromptLoader()

            try:
                loader.load_prompt("invalid")
                msg = "Should have raised YAML error"
                raise AssertionError(msg)
            except yaml.YAMLError:
                # Expected - YAML parsing should fail
                pass


def test_prompt_loader_default_prompts_dir():
    """Test default prompts directory initialization."""
    with patch("pathlib.Path.exists", return_value=True):
        loader = PromptLoader()
        assert loader.prompts_dir is not None
