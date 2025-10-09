"""data_analysis.py.

This script performs a full data analysis workflow on bird observation data.
It includes data loading from a CSV string, validation using a Pydantic model,
and exploratory data analysis with summary statistics and a t-test.
"""
import ast
from datetime import date
from typing import Any

import pandas as pd
from pydantic import BaseModel, Field, ValidationError
from scipy import stats


# --- Pydantic Data Model for Validation ---
class BirdObservation(BaseModel):
    """Pydantic model for validating the bird observation data.

    This model defines the expected data types and structure for each
    column in the dataset, ensuring data integrity before analysis.
    """

    species_name: str
    taxonomic_family: str
    taxonomic_order: str
    asdawas: float | None = Field(description = "Wing Length")
    beak_length: float
    weight_g: float
    nocturnal_diurnal: str
    brrrrkk: date  # An unusual column name, representing a date
    diet_types: dict[str, float]
    closest_relatives: list[str]
    latitude: float
    longitude: float
    altitude_m: float
    habitat_type: str
    weather_condition: str
    temperature_c: float
    observation_time: str
    diet_dict: dict | None = None
    primary_diet: str | None = None
    is_nocturnal: bool | None = None
    blwl: float | None = None


def load_and_validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """Read a CSV string into a DataFrame and validates its contents.

    Args:
        df: A pandas DataFrame containing the bird observation data.

    Returns:
        A pandas DataFrame if validation is successful.

    Raises:
        ValueError: If data validation fails.

    """
    # Clean and preprocess columns with string-serialized data structures
    try:
        # Parse string-encoded structures only when necessary. Some rows may
        # already contain dict/list objects (e.g., when pandas infers them).
        df["diet_types"] = df["diet_types"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )
        df["closest_relatives"] = df["closest_relatives"].apply(
            lambda v: ast.literal_eval(v) if isinstance(v, str) else v
        )

        # Avoid double literal_eval: use the already-parsed diet_types as diet_dict
        df["diet_dict"] = df["diet_types"].apply(lambda v: dict(v) if isinstance(v, dict) else v)
        # primary_diet should be None for empty/missing diet dicts
        # Use a lambda key to make the type-checker happy (and handle empty dicts)
        df["primary_diet"] = df["diet_dict"].apply(
            lambda x: (max(x, key=lambda k: x[k]) if isinstance(x, dict) and x else None)
        )

        # Keep is_nocturnal as boolean (Pydantic model expects bool)
        df["is_nocturnal"] = (df["nocturnal_diurnal"] == "Nocturnal")
        df["blwl"] = df["beak_length"] / df["weight_g"]

        # Validate each row against the Pydantic model
        validated_records = []
        for index, row in df.iterrows():
            try:
                # Convert row to dict for validation
                record = row.to_dict()
                # Ensure the date format is compatible
                record["brrrrkk"] = pd.to_datetime(record["brrrrkk"]).date()
                BirdObservation(**record)
                validated_records.append(record)
            except ValidationError as e:
                print(f"Validation error on row {index}: {e}")
                msg = "Data validation failed."
                raise ValueError(msg) from e

        # If all records are valid, create the final DataFrame
        return pd.DataFrame(validated_records)

    except Exception as e:
        msg = f"Error during data processing or validation: {e}"
        raise ValueError(msg) from e


def get_summary_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate and return summary statistics for numerical columns.

    Args:
        df: The pandas DataFrame to analyze.

    Returns:
        A DataFrame containing the summary statistics (mean, std, min, max, etc.).

    """
    numerical_cols = ["beak_length", "weight_g", "temperature_c", "altitude_m", "asdawas"]
    return df[numerical_cols].describe().T


def perform_t_test_on_weights(df: pd.DataFrame) -> dict[str, Any]:
    """Perform an independent two-sample t-test to compare the mean weight of diurnal vs. nocturnal birds.

    Args:
        df: The pandas DataFrame containing the data.

    Returns:
        A dictionary with the t-statistic, p-value, and a conclusion string.
        Returns None if there are not enough samples for the test.

    """
    diurnal_weights = df[df["nocturnal_diurnal"] == "Diurnal"]["weight_g"]
    nocturnal_weights = df[df["nocturnal_diurnal"] == "Nocturnal"]["weight_g"]
    weights = 2
    if len(diurnal_weights) < weights or len(nocturnal_weights) < weights:
        return {
            "result": "Insufficient data to perform t-test. Requires at least 2 samples per group.",
            "t_statistic": None,
            "p_value": None,
        }

    t_stat, p_value = stats.ttest_ind(diurnal_weights, nocturnal_weights, equal_var=False)
    pval = 0.05
    conclusion = (
        "There is a statistically significant difference in the mean weights "
        "between diurnal and nocturnal birds."
    ) if p_value < pval else (
        "There is no statistically significant difference in the mean weights "
        "between diurnal and nocturnal birds."
    )

    return {
        "t_statistic": t_stat,
        "p_value": p_value,
        "conclusion": conclusion,
    }


def analyze_categorical_data(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Analyzes categorical data by providing value counts and percentages.

    Args:
        df: The pandas DataFrame.

    Returns:
        A dictionary where keys are column names and values are DataFrames
        with counts and percentages.

    """
    categorical_cols = ["species_name", "taxonomic_family", "habitat_type"]
    analysis_results = {}

    for col in categorical_cols:
        counts = df[col].value_counts().reset_index()
        counts.columns = [col, "count"]
        percentages = df[col].value_counts(normalize=True).mul(100).reset_index()
        percentages.columns = [col, "percentage"]

        # Merge counts and percentages into a single DataFrame
        analysis_df = counts.merge(percentages, on=col)
        analysis_results[col] = analysis_df.sort_values(by="count", ascending=False)

    return analysis_results

def create_analysis_report_df(df: pd.DataFrame) -> pd.DataFrame:
    """Consolidates all analysis results into a single, comprehensive DataFrame.

    Args:
        df: The validated pandas DataFrame.

    Returns:
        A single pandas DataFrame containing all analysis results.

    """
    # Get numerical summaries
    summary_df = get_summary_statistics(df)

    # Get categorical results
    categorical_analysis = analyze_categorical_data(df)

    # Get t-test results
    t_test_results = perform_t_test_on_weights(df)

    # Create the final report DataFrame using a dictionary
    report_dict: dict[str, list[Any]] = {
        "Analysis Type": [],
        "Attribute": [],
        "Value": [],
    }

    # Add numerical summary statistics
    for index, row in summary_df.iterrows():
        for col in summary_df.columns:
            report_dict["Analysis Type"].append(f"Numerical Summary: {col.title()}")
            report_dict["Attribute"].append(index)
            report_dict["Value"].append(row[col])

    # Add t-test results
    for key, value in t_test_results.items():
        if key != "conclusion":
            report_dict["Analysis Type"].append("T-Test")
            report_dict["Attribute"].append(key)
            report_dict["Value"].append(value)

    report_dict["Analysis Type"].append("T-Test")
    report_dict["Attribute"].append("Conclusion")
    report_dict["Value"].append(t_test_results.get("conclusion", "N/A"))

    # Add categorical analysis results
    for col_name, result_df in categorical_analysis.items():
        for _, row in result_df.iterrows():
            report_dict["Analysis Type"].append(f"Categorical Analysis: {col_name}")
            report_dict["Attribute"].append(f"{row[col_name]} (Count)")
            report_dict["Value"].append(row["count"])
            report_dict["Analysis Type"].append(f"Categorical Analysis: {col_name}")
            report_dict["Attribute"].append(f"{row[col_name]} (Percentage)")
            report_dict["Value"].append(row["percentage"])

    return pd.DataFrame(report_dict)


def eda(csv_data: pd.DataFrame) -> pd.DataFrame:
    """Run the full exploratory data analysis workflow."""
    print("--- Starting Data Analysis Workflow ---")
    try:
        # 1. Load and validate the data
        validated_data = load_and_validate_data(csv_data)
        print("\nData loaded and successfully validated! DataFrame shape:", validated_data.shape)

        return create_analysis_report_df(validated_data)

    except ValueError as e:
        print(f"\nAn error occurred: {e}")
