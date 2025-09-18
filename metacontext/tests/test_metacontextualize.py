"""Test suite for the simple metacontextualize API."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.metacontextualize import metacontextualize


class TestMetacontextualizeAPI:
    """Test the simple metacontextualize function."""

    def test_basic_dataframe_contextualization(self):
        """Test basic functionality with pandas DataFrame."""
        df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_file = temp_path / "test.csv"

            # Save data
            df.to_csv(csv_file, index=False)

            # Generate metacontext
            metacontext_file = metacontextualize(df, csv_file)

            # Verify file was created (using new default clean naming)
            assert metacontext_file.exists()
            assert metacontext_file.name == "test_metacontext.yaml"

            # Verify content
            with open(metacontext_file) as f:
                context = yaml.safe_load(f)

            assert context["metacontext_version"] == "0.2.0"
            assert (
                context["generation_info"]["generation_method"] == "explicit_function"
            )
            assert context["file_info"]["filename"] == "test.csv"
            assert context["data_analysis"]["type"] == "pandas_dataframe"
            assert context["data_analysis"]["shape"] == [3, 2]

    def test_custom_output_path(self):
        """Test specifying custom output path."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_file = temp_path / "data.csv"
            custom_output = temp_path / "custom_context.yaml"

            df.to_csv(csv_file, index=False)

            # Generate with custom output path
            result_path = metacontextualize(df, csv_file, output_path=custom_output)

            assert result_path == custom_output
            assert custom_output.exists()

    def test_numpy_array_contextualization(self):
        """Test functionality with numpy arrays."""
        arr = np.array([[1, 2, 3], [4, 5, 6]])

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            npy_file = temp_path / "array.npy"

            # Save numpy array
            np.save(npy_file, arr)

            # Generate metacontext
            metacontext_file = metacontextualize(arr, npy_file)

            assert metacontext_file.exists()

            with open(metacontext_file) as f:
                context = yaml.safe_load(f)

            assert context["data_analysis"]["type"] == "numpy_array"
            assert context["data_analysis"]["shape"] == [2, 3]
            assert context["data_analysis"]["dtype"] == "int64"

    def test_generic_object_contextualization(self):
        """Test functionality with generic Python objects."""
        data = {"key1": "value1", "key2": [1, 2, 3]}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            json_file = temp_path / "data.json"

            # Simulate saving data
            json_file.write_text('{"key1": "value1", "key2": [1, 2, 3]}')

            # Generate metacontext
            metacontext_file = metacontextualize(data, json_file)

            assert metacontext_file.exists()

            with open(metacontext_file) as f:
                context = yaml.safe_load(f)

            assert context["data_analysis"]["type"] == "generic_object"
            assert context["data_analysis"]["python_type"] == "dict"

    def test_disable_llm_analysis(self):
        """Test disabling LLM analysis."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_file = temp_path / "test.csv"

            df.to_csv(csv_file, index=False)

            # Generate without LLM analysis
            metacontext_file = metacontextualize(
                df,
                csv_file,
                include_llm_analysis=False,
            )

            with open(metacontext_file) as f:
                context = yaml.safe_load(f)

            assert context["business_context"]["status"] == "skipped"
            assert (
                "LLM analysis disabled by user" in context["business_context"]["note"]
            )

    def test_with_custom_config(self):
        """Test passing custom configuration."""
        df = pd.DataFrame({"x": [1, 2, 3]})
        config = {"custom_key": "custom_value"}

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            csv_file = temp_path / "test.csv"

            df.to_csv(csv_file, index=False)

            # Generate with custom config
            metacontext_file = metacontextualize(df, csv_file, config=config)

            assert metacontext_file.exists()

            # Config affects LLM generation (tested indirectly through no errors)
            with open(metacontext_file) as f:
                context = yaml.safe_load(f)

            # Should complete without errors
            assert context["metacontext_version"] == "0.2.0"

    def test_directory_creation(self):
        """Test that output directories are created if they don't exist."""
        df = pd.DataFrame({"x": [1, 2, 3]})

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Use nested directory that doesn't exist
            nested_dir = temp_path / "nested" / "deep"
            csv_file = nested_dir / "test.csv"

            # Create the directory for the CSV file
            nested_dir.mkdir(parents=True)
            df.to_csv(csv_file, index=False)

            # Generate metacontext (should create output directory)
            metacontext_file = metacontextualize(df, csv_file)

            assert metacontext_file.exists()
            assert metacontext_file.parent == nested_dir

    def test_error_handling_data_analysis(self):
        """Test error handling in data analysis."""

        # Create an object that will cause analysis errors
        class ProblematicObject:
            def __getattr__(self, name: str) -> None:
                msg = "Analysis error"
                raise ValueError(msg)

        obj = ProblematicObject()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            file_path = temp_path / "test.txt"

            file_path.write_text("dummy content")

            # Should handle analysis errors gracefully
            metacontext_file = metacontextualize(obj, file_path)

            assert metacontext_file.exists()

            with open(metacontext_file) as f:
                context = yaml.safe_load(f)

            # Should have error information in data analysis
            assert "analysis_error" in context["data_analysis"]
