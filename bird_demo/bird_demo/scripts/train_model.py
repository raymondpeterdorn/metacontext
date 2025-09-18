"""Train a model on the bird dataset."""

import logging
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Default model configuration - makes parameters configurable
DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "test_size": 0.2,
    "random_state": 42,
    "n_estimators": 50,
}


def train_bird_classifier(
    file_path: str | Path, config: dict[str, Any] | None = None,
) -> RandomForestClassifier:
    """Train a Random Forest model to classify bird species.

    Args:
        file_path: Path to the CSV file with bird data
        config: Model configuration parameters (optional)

    Returns:
        Trained RandomForestClassifier model

    """
    # Use default config if none provided
    if config is None:
        config = DEFAULT_MODEL_CONFIG.copy()

    logger.info("--- Training ML Model ---")
    df = pd.read_csv(file_path)

    # Basic Feature Engineering
    df["observation_month"] = pd.to_datetime(df["observation_date"]).dt.month
    df["location_type"] = "urban"
    df.loc[
        df["location_description"].str.contains("park|reserve|forest"), "location_type",
    ] = "natural"

    # Prepare features and target
    features = df[["observation_month", "location_type"]]
    target = df["species_name"]

    # Convert categorical features to numeric
    features = pd.get_dummies(features, columns=["location_type"], drop_first=True)

    # Encode target variable
    label_encoder = LabelEncoder()
    target_encoded = label_encoder.fit_transform(target)

    # Split data
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target_encoded,
        test_size=config["test_size"],
        random_state=config["random_state"],
    )

    # Train a simple RandomForest model
    model = RandomForestClassifier(
        n_estimators=config["n_estimators"],
        random_state=config["random_state"],
    )
    model.fit(x_train, y_train)

    # Evaluate the model
    accuracy = model.score(x_test, y_test)
    logger.info("Model accuracy on test set: %.2f", accuracy)

    return model


if __name__ == "__main__":
    # Example usage
    data_file = Path(__file__).parent.parent / "data" / "birdos.csv"

    # Use default config or modify it as needed
    model_config = DEFAULT_MODEL_CONFIG.copy()
    # Configuration can be customized here

    trained_model = train_bird_classifier(data_file, model_config)

    # Export the model
    output_file = (
        Path(__file__).parent.parent / "output" / "bird_classification_model.pkl"
    )
    output_file.parent.mkdir(exist_ok=True)
    joblib.dump(trained_model, output_file)

    logger.info("\nModel exported to '%s'", output_file)
