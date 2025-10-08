#!/usr/bin/env python3
"""Create a simple bird classification model and save it as pkl for testing."""

import ast
import logging
import os
import pickle
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from metacontext.ai.handlers.companions.companion_factory import CompanionProviderFactory
from metacontext.metacontextualize import metacontextualize
from PIL import Image
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(level=logging.INFO)
logging.getLogger("metacontext").setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv(Path(__file__).parent.parent / ".env")

DATA_DIR = Path("bird_demo/data")
OUTPUT_DIR = Path("bird_demo/output")


# Default model configuration
DEFAULT_MODEL_CONFIG: dict[str, Any] = {
    "n_estimators": 10,
    "random_state": 42,
}

def get_ai_companion():
    """Create an AI companion for interactive analysis."""
    try:
        factory = CompanionProviderFactory()
        ai_companion = factory.detect_available_companion()
        if ai_companion:
            return ai_companion
        else:
            logger.warning("No AI companion detected, falling back to API mode")
            return None
    except Exception:
        logger.exception("Error creating AI companion, falling back to API mode")
        return None

def csv_and_xlsx() -> None:
    """Process birdos.csv and birdos_expanded.xlsx to create CSV and Excel files."""
    csv_path = Path(DATA_DIR / "birdos_expanded.csv")
    xlsx_path = Path(DATA_DIR / "birdos_expanded.xlsx")  # Use the existing file
    output_csv_path = Path(OUTPUT_DIR / "csv.csv")
    output_xlsx_path = Path(OUTPUT_DIR / "xlsx.xlsx")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Read CSV
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path)

        # Create some simple features from the data-- let's see if the LLM will figure what these mean
        try:
            df_csv["diet_dict"] = df_csv["diet_types"].apply(ast.literal_eval)
            df_csv["primary_diet"] = df_csv["diet_dict"].apply(lambda x: max(x, key=x.get))
            df_csv["is_nocturnal"] = (df_csv["nocturnal_diurnal"] == "Nocturnal").astype(int)
            processed_df = df_csv[
                [
                    "species_name",
                    "taxonomic_family",
                    "asdawas",
                    "beak_length",
                    "nocturnal_diurnal",
                    "is_nocturnal",
                    "primary_diet",
                ]
            ]
            processed_df.to_csv(output_csv_path, index=False)

            # Get AI companion for interactive analysis
            ai_companion = get_ai_companion()
            
            if ai_companion:
                # Use companion mode for interactive analysis
                metacontextualize(
                    processed_df, 
                    output_csv_path,
                    output_format="yaml",
                    scan_codebase=True,
                    ai_companion=ai_companion,
                    verbose=True,
                )
            else:
                # Fallback to API mode if companion not available
                metacontextualize(
                    processed_df, 
                    output_csv_path,
                    output_format="yaml",
                    scan_codebase=True,
                    llm_api_key=os.getenv("GEMINI_API_KEY"),
                    llm_provider="gemini",
                    include_llm_analysis=True,
                )
        except Exception:
            logger.exception("Error processing CSV")
        # try:
        #     eda_output = eda(df_csv)
        #     eda_output_path = OUTPUT_DIR / "eda_csv.csv"
        #     eda_output.to_csv(eda_output_path)
        #     metacontextualize(eda_output, eda_output_path, args)
        # except Exception:
        #     logger.exception("Error running EDA")
    else:
        logger.warning("CSV file not found: %s", csv_path)

    # Read Excel
    # if xlsx_path.exists():
    #     try:
    #         df_xlsx = pd.read_excel(xlsx_path, sheet_name=0)

    #         # Process Excel file if it has similar columns
    #         if all(col in df_xlsx.columns for col in ["diet_types", "nocturnal_diurnal"]):
    #             df_xlsx["diet_dict"] = df_xlsx["diet_types"].apply(ast.literal_eval)
    #             df_xlsx["primary_diet"] = df_xlsx["diet_dict"].apply(lambda x: max(x, key=x.get))
    #             df_xlsx["nocturn_alley"] = (df_xlsx["nocturnal_diurnal"] == "Nocturnal").astype(int)
    #             df_xlsx.to_excel(output_xlsx_path, index=False)

    #             args = MetacontextualizeArgs(
    #                 output_format="yaml",
    #                 config={
    #                     "scan_codebase": True,
    #                     "llm_api_key": os.getenv("GEMINI_API_KEY"),
    #                     "llm_provider": "gemini",
                        
    #                 },
    #                 include_llm_analysis=True,
    #             )
    #             metacontextualize(df_xlsx, output_xlsx_path, args)
    #         else:
    #             logger.warning("Excel file doesn't have expected columns")
    #     except Exception:
    #         logger.exception("Error processing Excel")
    # else:
    #     logger.warning("Excel file not found: %s", xlsx_path)

def ml_models() -> None:
    """Create and train a machine learning model for bird classification."""
    csv_path = DATA_DIR / "birdos.csv"

    if not csv_path.exists():
        logger.warning("CSV file not found for ML model: %s", csv_path)
        return

    df_csv = pd.read_csv(csv_path)

    # Preprocess data and calculate is_nocturnal if needed
    if "nocturnal_diurnal" in df_csv.columns and "is_nocturnal" not in df_csv.columns:
        df_csv["is_nocturnal"] = (df_csv["nocturnal_diurnal"] == "Nocturnal").astype(int)

    # Determine available feature columns
    available_features = []
    for col in ["asdawas", "beak_length", "is_nocturnal"]:
        if col in df_csv.columns:
            available_features.append(col)

    if not available_features:
        logger.warning("No suitable features found for ML model")
        return

    # Extract features and target
    try:
        x = df_csv[available_features]
        y = df_csv["species_name"]

        model_config = DEFAULT_MODEL_CONFIG.copy()
        # Create and train model
        model = RandomForestClassifier(
            n_estimators=model_config["n_estimators"],
            random_state=model_config["random_state"],
        )
        model.fit(x, y)

        # Ensure output directory exists
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

        # Save the model
        model_path = OUTPUT_DIR / "bird_classification_model.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

        # Get AI companion for interactive analysis
        ai_companion = get_ai_companion()
        
        if ai_companion:
            # Use companion mode for interactive analysis
            metacontextualize(
                model, 
                model_path,
                output_format="yaml",
                scan_codebase=True,
                ai_companion=ai_companion,
                verbose=True,
            )
        else:
            # Fallback to API mode if companion not available
            metacontextualize(
                model, 
                model_path,
                output_format="yaml",
                scan_codebase=True,
                llm_api_key=os.getenv("GEMINI_API_KEY"),
                llm_provider="gemini",
                include_llm_analysis=True,
            )
        logger.info("Machine learning model created successfully")
    except Exception:
        logger.exception("Error creating machine learning model")

def geospatial_data() -> None:
    """Process geospatial data and generate metacontext for it."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Define file paths
    geojson_path = DATA_DIR / "birdos_locations.geojson"
    gpkg_path = DATA_DIR / "birdos_locations.gpkg"
    geojson_output_path = OUTPUT_DIR / "buffered_locations.geojson"
    gpkg_output_path = OUTPUT_DIR / "filtered_locations.gpkg"

    # Process GeoJSON
    if geojson_path.exists():
        try:
            logger.info("Processing GeoJSON file")
            gjson = gpd.read_file(str(geojson_path), driver="GeoJSON")

            # Buffer the geometries in the GeoJSON
            geojson_buffered = gjson.copy()
            geojson_buffered["geometry"] = gjson.geometry.buffer(0.00001)

            # Save the buffered GeoJSON
            geojson_buffered.to_file(str(geojson_output_path), driver="GeoJSON")

            # Get AI companion for interactive analysis
            ai_companion = get_ai_companion()
            
            if ai_companion:
                # Use companion mode for interactive analysis
                metacontextualize(
                    geojson_buffered, 
                    geojson_output_path,
                    output_format="yaml",
                    scan_codebase=True,
                    ai_companion=ai_companion,
                    verbose=True,
                )
            else:
                # Fallback to API mode if companion not available
                metacontextualize(
                    geojson_buffered, 
                    geojson_output_path,
                    output_format="yaml",
                    scan_codebase=True,
                    llm_api_key=os.getenv("GEMINI_API_KEY"),
                    llm_provider="gemini",
                    include_llm_analysis=True,
                )
            logger.info("Created buffered GeoJSON at %s", geojson_output_path)
        except Exception:
            logger.exception("Error processing GeoJSON")
    else:
        logger.warning("GeoJSON file not found: %s", geojson_path)

    # Process GeoPackage
    if gpkg_path.exists():
        try:
            logger.info("Processing GeoPackage file")
            gpkg = gpd.read_file(str(gpkg_path), driver="GPKG")

            # Drop some columns
            columns_to_drop = [col for col in ["beak_length", "taxonomic_family"] if col in gpkg.columns]

            gpkg_filtered = gpkg.drop(columns=columns_to_drop) if columns_to_drop else gpkg.copy()

            # Save the filtered GeoPackage
            gpkg_filtered.to_file(str(gpkg_output_path), driver="GPKG")

            # Get AI companion for interactive analysis
            ai_companion = get_ai_companion()
            
            if ai_companion:
                # Use companion mode for interactive analysis
                metacontextualize(
                    gpkg_filtered, 
                    gpkg_output_path,
                    output_format="yaml",
                    scan_codebase=True,
                    ai_companion=ai_companion,
                    verbose=True,
                )
            else:
                # Fallback to API mode if companion not available
                metacontextualize(
                    gpkg_filtered, 
                    gpkg_output_path,
                    output_format="yaml",
                    scan_codebase=True,
                    llm_api_key=os.getenv("GEMINI_API_KEY"),
                    llm_provider="gemini",
                    include_llm_analysis=True,
                )
            logger.info("Created filtered GeoPackage at %s", gpkg_output_path)
        except Exception:
            logger.exception("Error processing GeoPackage")
    else:
        logger.warning("GeoPackage file not found: %s", gpkg_path)

def media_data() -> None:
    """Create a simple pixel art bird image and generate metacontext for it."""
    # Define the size of the image
    width, height = 20, 20

    # Create a blank white canvas
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define bird colors
    bird_body = [255, 200, 0]    # Yellow body
    bird_beak = [255, 100, 0]    # Orange beak
    bird_eye = [0, 0, 0]         # Black eye
    bird_wing = [200, 150, 0]    # Darker yellow wing

    # Define radius for body shape
    body_radius_squared = 25  # For checking if pixel is within body circle

    # Draw the bird body (simple oval shape)
    for y in range(8, 15):
        for x in range(5, 15):
            if (x - 10)**2 + (y - 12)**2 < body_radius_squared:
                img_array[y, x] = bird_body

    # Draw the beak
    for y in range(11, 13):
        for x in range(3, 6):
            img_array[y, x] = bird_beak

    # Draw the eye
    img_array[10, 7] = bird_eye

    # Draw the wing
    for y in range(10, 13):
        for x in range(12, 16):
            img_array[y, x] = bird_wing

    # Create the image from the numpy array
    bird_img = Image.fromarray(img_array)

    # Save the image
    img_path = OUTPUT_DIR / "pixel_bird.png"
    bird_img.save(img_path)

    # Generate metacontext for the image
    ai_companion = get_ai_companion()
    
    if ai_companion:
        # Use companion mode for interactive analysis
        metacontextualize(
            bird_img, 
            img_path,
            output_format="yaml",
            scan_codebase=True,
            ai_companion=ai_companion,
            verbose=True,
        )
    else:
        # Fallback to API mode if companion not available
        metacontextualize(
            bird_img, 
            img_path,
            output_format="yaml",
            scan_codebase=True,
            llm_api_key=os.getenv("GEMINI_API_KEY"),
            llm_provider="gemini",
            include_llm_analysis=True,
        )

def main() -> None:
    """Train and save a simple bird classification model."""
    # Use default config if none provided
    csv_and_xlsx()
    ml_models()
    geospatial_data()
    media_data()

if __name__ == "__main__":
    main()
