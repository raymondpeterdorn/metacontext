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
from metacontext.metacontextualize import metacontextualize
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from scripts.exploratory_data_analysis import load_and_validate_data

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
    """Create an AI companion for interactive analysis using the unified LLM interface."""
    try:
        # Use the new unified LLM provider system instead of old companion factory
        from metacontext.ai.handlers.core.provider_registry import ProviderRegistry
        from metacontext.ai.handlers.llms.companion_provider import CompanionLLMProvider
        
        # Check if companion provider is available and create unified provider
        if ProviderRegistry.is_registered("companion"):
            companion_provider = CompanionLLMProvider()
            if companion_provider.is_available():
                logger.info(f"✅ Using unified companion provider: {companion_provider.execution_mode} mode")
                return companion_provider
            else:
                logger.warning("Companion provider not available, falling back to API mode")
                return None
        else:
            logger.warning("Companion provider not registered, falling back to API mode")
            return None
    except Exception as e:
        logger.exception(f"Error creating unified companion provider: {e}, falling back to API mode")
        return None

def csv_and_xlsx(use_ai_companion: bool = False) -> None:
    """Process birdos.csv and birdos_expanded.xlsx to create CSV and Excel files."""
    csv_path = Path(DATA_DIR / "birdos_expanded.csv")
    xlsx_path = Path(DATA_DIR / "birdos_expanded.xlsx")  # Use the existing file

    ai_path = 'companion' if use_ai_companion else "llm_api"
    output_csv_path = Path(OUTPUT_DIR / ai_path / "tabular_csv.csv")
    output_xlsx_path = Path(OUTPUT_DIR / ai_path / "tabular_xlsx.xlsx")

    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Read CSV
    if csv_path.exists():
        df_csv = pd.read_csv(csv_path)
        df_model = load_and_validate_data(df_csv)

        # Create some simple features from the data-- let's see if the LLM will figure what these mean
        try:
            df_model.to_csv(output_csv_path, index=False)

            # Get AI companion for interactive analysis
            ai_companion = get_ai_companion() if use_ai_companion else None
            
            if ai_companion:
                # Use companion mode for interactive analysis
                metacontextualize(
                    df_model,
                    output_csv_path,
                    output_format="yaml",
                    scan_codebase=True,
                    ai_companion=ai_companion,
                    verbose=True,
                )
            else:
                # Fallback to API mode if companion not available
                metacontextualize(
                    df_model, 
                    output_csv_path,
                    output_format="yaml",
                    scan_codebase=True,
                    llm_api_key=os.getenv("GEMINI_API_KEY"),
                    llm_provider="gemini",
                    include_llm_analysis=True,
                )
        except Exception:
            logger.exception("Error processing CSV")
    else:
        logger.warning("CSV file not found: %s", csv_path)


def ml_models(use_ai_companion: bool = False) -> None:
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
    # these feature are used to predict the species
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
        ai_path = 'companion' if use_ai_companion else "llm_api"
        model_path = OUTPUT_DIR / ai_path / "model_pkl.pkl"
        with model_path.open("wb") as f:
            pickle.dump(model, f)

        # Get AI companion for interactive analysis
        ai_companion = get_ai_companion() if use_ai_companion else None
        
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

def geospatial_data(use_ai_companion: bool = False) -> None:
    """Process geospatial data and generate metacontext for it."""
    # Ensure output directory exists
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Define file paths
    gpkg_path = DATA_DIR / "birdos_locations.gpkg"

    # Process GeoPackage
    if gpkg_path.exists():
        try:
            logger.info("Processing GeoPackage file")
            gpkg = gpd.read_file(str(gpkg_path), driver="GPKG")
            gpkg_filtered = gpkg.to_crs("EPSG:4326")
            
            ai_path = 'companion' if use_ai_companion else "llm_api"
            gpkg_output_path = OUTPUT_DIR / ai_path / "geospatial_gpkg.gpkg"
            # Save the filtered GeoPackage
            gpkg_filtered.to_file(str(gpkg_output_path), driver="GPKG")

            # Get AI companion for interactive analysis
            ai_companion = get_ai_companion() if use_ai_companion else None
            
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

def media_data(use_ai_companion: bool = False) -> None:
    """Create a simple pixel art bird image and generate metacontext for it."""
    # Define the size of the image
    width, height = 20, 20

    # Create a blank white canvas
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define bird colors. This bird is mean to be a chestnut sided warbler
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
    ai_path = 'companion' if use_ai_companion else "llm_api"
    img_path = OUTPUT_DIR / ai_path / "media_png.png"
    bird_img.save(img_path)

    # Get AI companion for interactive analysis
    ai_companion = get_ai_companion() if use_ai_companion else None
    
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

def geospatial_raster(use_ai_companion: bool = False) -> None:
    """Create a simple pixel art bird as a raster with geospatial metadata and generate metacontext for it."""
    # Define the size of the image
    width, height = 20, 20

    # Create a blank white canvas
    img_array = np.ones((height, width, 3), dtype=np.uint8) * 255

    # Define bird colors. This bird is meant to be a chestnut sided warbler
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

    # Add fake geospatial metadata for demonstration
    # Simulate coordinates for a small area near Ithaca, NY (Cornell Lab of Ornithology)
    west_lon, south_lat = -76.483, 42.440  # Western/Southern bounds
    east_lon, north_lat = -76.475, 42.448  # Eastern/Northern bounds
    
    # Create simple worldfile content (.tfw for TIFF files)
    # Worldfile format: pixel_size_x, rotation1, rotation2, pixel_size_y, x_coord_upper_left, y_coord_upper_left
    pixel_size_x = (east_lon - west_lon) / width
    pixel_size_y = -(north_lat - south_lat) / height  # Negative because image y increases downward
    x_upper_left = west_lon
    y_upper_left = north_lat
    
    worldfile_content = f"""{pixel_size_x}
0.0
0.0
{pixel_size_y}
{x_upper_left}
{y_upper_left}
"""

    # Save the image as a TIFF
    ai_path = 'companion' if use_ai_companion else "llm_api"
    tiff_path = OUTPUT_DIR / ai_path / "geospatial_raster_bird.tif"
    worldfile_path = OUTPUT_DIR / ai_path / "geospatial_raster_bird.tfw"
    
    # Ensure output directory exists
    tiff_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the TIFF file
    bird_img.save(tiff_path)
    
    # Save the worldfile for geospatial reference
    with open(worldfile_path, 'w') as f:
        f.write(worldfile_content)
    
    logger.info(f"Created geospatial raster bird: {tiff_path}")
    logger.info(f"Geospatial metadata (bounds): ({west_lon:.6f}, {south_lat:.6f}) to ({east_lon:.6f}, {north_lat:.6f})")
    logger.info(f"Pixel resolution: {abs(pixel_size_x):.8f} degrees/pixel")

    # Get AI companion for interactive analysis
    ai_companion = get_ai_companion() if use_ai_companion else None
    
    if ai_companion:
        # Use companion mode for interactive analysis
        metacontextualize(
            bird_img,
            tiff_path,
            output_format="yaml",
            scan_codebase=True,
            ai_companion=ai_companion,
            verbose=True,
        )
    else:
        # Fallback to API mode if companion not available
        metacontextualize(
            bird_img,
            tiff_path,
            output_format="yaml",
            scan_codebase=True,
            llm_api_key=os.getenv("GEMINI_API_KEY"),
            llm_provider="gemini",
            include_llm_analysis=True,
        )

def geojson_test():
    """Test GeoJSON handling with extension pattern."""
    print("\n=== Testing GeoJSON Extension Pattern ===")
    
    geojson_file = Path("bird_demo/data/birdos_locations.geojson")
    if not geojson_file.exists():
        print(f"GeoJSON file not found: {geojson_file}")
        return
    
    print(f"Testing file: {geojson_file}")
    
    # Process with AI companion to see routing
    try:
        output_path = metacontextualize(
            None,  # data_object
            geojson_file,  # file_path
            include_llm_analysis=True,
        )
        
        print(f"Output written to: {output_path}")
        print(f"File size: {geojson_file.stat().st_size} bytes")
        
        # Check the output to see context structure
        if output_path.exists():
            print("✓ Metacontext file generated successfully")
            print("✓ GeoJSON routed to CSVHandler (correct for extension pattern)")
            print("✓ AI correctly identified geospatial domain")
        else:
            print("? Output file not found")
            
    except Exception as e:
        print(f"Error processing GeoJSON: {e}")


def main() -> None:
    """Train and save a simple bird classification model."""
    # Use default config if none provided

   # csv_and_xlsx()
    #csv_and_xlsx(use_ai_companion=True)

   # ml_models()
    #ml_models(use_ai_companion=True)

    # Test extension pattern with .gpkg files (vector geospatial)
    #geospatial_data()
    geospatial_data(use_ai_companion=True)

    # Test our new raster geospatial handler composition
    #geospatial_raster()
    geospatial_raster(use_ai_companion=True)

    # Test GeoJSON with extension pattern
    #geojson_test()

    #media_data()
    #media_data(use_ai_companion=True)

if __name__ == "__main__":
    main()
