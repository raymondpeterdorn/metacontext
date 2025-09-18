Bird Anatomy and Species Classification Project
This repository contains an example codebase for data analysis and machine learning. The primary goal of this project is to explore the relationships between various anatomical measurements of birds and to build a predictive model for species classification.

Business Context & Purpose
The data was collected by a team of ornithologists to better understand the physical characteristics that define different bird species. The key questions we aim to answer are:

What are the key statistical characteristics of the collected data?

Can we predict a bird's species based on its physical measurements alone?

How do physical traits differ between nocturnal and diurnal birds?

The derived wing_beak_ratio column is a core feature for this analysis, representing a key biomechanical property of the birds, and is expected to be a strong predictor of species.

Repository Structure
data/: Contains the raw input data.

scripts/: Holds the Python scripts for analysis and modeling.

output/: Where all generated files (summaries, processed data, and the trained model) are stored.

docs/: Contains additional project documentation and a data dictionary.

Running the Code
To run the full pipeline, execute the following scripts in order from the example_code directory:

python scripts/exploratory_data_analysis.py

This script cleans and validates the raw data.

It performs statistical analysis and creates a new derived column: wing_beak_ratio.

Exports an eda_summary.csv and a processed_bird_data.csv to the output/ directory.

python scripts/models/train_model.py

This script loads the processed data.

It trains a Random Forest model to predict the species.

Exports the trained model as bird_model.pkl to the output/ directory.

Known Data Issues & Caveats
The asdawas column is often subject to slight measurement errors.

The brrrrkk column is a simple YYYY-MM-DD date string for the observation date.

The diet_types column is a dictionary that represents the diet composition as a percentage, which must be handled during data loading.

The closest_relatives is a list of known related species and serves as a qualitative feature for species identification.