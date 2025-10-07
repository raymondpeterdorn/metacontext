# Metacontext - Universal File Intelligence

**Transform any data file into an intelligent, self-documenting artifact with rich metadata that travels with your data.**

## 🎯 Why Metacontext Exists

In the modern data landscape, files are created constantly but their context is lost. You know *what* a file contains, but not *why* it exists, *how* it was created, or *what* it means for your project.

**Traditional file metadata tells you:**
- File size, creation date, permissions
- Basic structure (rows, columns, file type)

**Metacontext tells you:**
- **Purpose**: Why was this file created? What problem does it solve?
- **Provenance**: How was it generated? What code created it?
- **Context**: What does it mean? How does it fit into your project?
- **Quality**: What are its strengths and limitations?
- **Usage**: How should others use this file?

## 🚀 The Vision

Every data file should be **self-documenting** and **intelligent**. Whether it's a CSV dataset, a trained ML model, a GeoJSON file, or a research image - the file should carry rich context that makes it immediately useful to anyone who encounters it.

**Before Metacontext:**
```
model.pkl          # What model? Trained on what? Good or bad?
results.csv        # Results of what? What do columns mean?
locations.geojson  # Locations for what purpose? What projection?
```

**After Metacontext:**
```
model.pkl                    # Your original file (unchanged)
model_metacontext.yaml       # Rich context with AI analysis

results.csv                  # Your original file (unchanged)
results_metacontext.yaml     # Data dictionary + business context

locations.geojson            # Your original file (unchanged)  
locations_metacontext.yaml   # Spatial analysis + usage guide
```

## 🏗️ How It Works

Metacontext uses a **two-tier architecture** to generate comprehensive metadata:

1. **Deterministic Analysis** (Always Available)
   - File structure, statistics, technical specifications
   - Guaranteed to work without any external dependencies
   - Fast, reliable, factual information

2. **AI Enrichment** (Best Effort)
   - Intelligent interpretation of what the data means
   - Business context and usage recommendations
   - Quality assessment and potential issues
   - Graceful degradation when AI is unavailable

## 🔧 Universal File Support

Metacontext automatically detects file types and provides specialized analysis:

- **📊 Tabular Data** (.csv, .xlsx, .parquet): Schema analysis, statistical profiling, column interpretation
- **🤖 Machine Learning** (.pkl, .h5): Model architecture, hyperparameters, training context discovery
- **🗺️ Geospatial** (.geojson, .gpkg, .shp): Coordinate systems, spatial relationships, geometric analysis
- **🖼️ Media** (.png, .jpg, .mp4): Technical metadata, content analysis, format specifications
- **📁 Any File Type**: Basic analysis with file inspection and contextual information

## 🤖 AI-Powered Intelligence

- **Dynamic Provider Selection**: Auto-detects best available LLM (Gemini, OpenAI, Anthropic)
- **Cost Optimization**: 66% token reduction through smart prompt engineering
- **Semantic Codebase Analysis**: Revolutionary code understanding system that extracts business context, column meanings, and data relationships from comments, docstrings, and Pydantic schemas
- **Forensic Analysis**: Deep understanding of data purpose and quality
- **Confidence Assessment**: AI provides confidence levels for its interpretations

## 🔍 Semantic Codebase Analysis

Metacontext includes a groundbreaking **semantic codebase flattening system** that automatically extracts business context and data relationships from your codebase. This addresses the critical problem where LLMs miss embedded hints in code comments, docstrings, and Pydantic models.

### Key Features

- **📝 Enhanced Comment Mining**: Extracts business logic, data dictionaries, and algorithm explanations from multi-line comments and docstrings
- **🏗️ Pydantic Schema Intelligence**: Mines field descriptions, validation rules, and model relationships to understand data structures
- **🧠 Advanced Semantic Extraction**: Detects constants, enums, magic numbers, and complex patterns with intelligent classification
- **🕸️ Knowledge Graph Construction**: Builds semantic relationships between columns, functions, and business logic with confidence scoring
- **⚡ LLM-Optimized Output**: Generates 6 specialized output formats for different AI contexts (debugging, documentation, API specs, etc.)

### How It Works

The system processes your codebase through 6 phases:

1. **File Discovery**: Smart filtering focuses on relevant code while ignoring build artifacts
2. **Content Preprocessing**: AST-based parsing extracts structured information
3. **Pydantic Mining**: Discovers data schemas and validation business rules
4. **Semantic Analysis**: Identifies patterns, constants, and business logic
5. **Knowledge Graph**: Builds relationships with conflict resolution and cross-referencing
6. **LLM Output Generation**: Creates optimized context for AI consumption

**Example Impact:** On the bird demo project, the system discovered 2,090 semantic relationships, extracted 31 Pydantic field descriptions, and identified 14 business logic patterns—dramatically improving LLM understanding of data context.

## 🎯 Perfect For

- **Data Scientists**: Automatically document datasets and model outputs
- **Research Teams**: Create self-documenting research artifacts
- **Data Engineers**: Add intelligence to data pipeline outputs
- **Analytics Teams**: Generate rich context for business stakeholders
- **Any Developer**: Make your data files speak for themselves

## 🚀 Get Started

Choose your workspace to dive deeper:

### 📚 [**metacontext/**](metacontext/) - Core Library
Complete documentation, installation, and usage guide for the metacontext library.

### 🐦 [**bird_demo/**](bird_demo/) - Live Example  
See metacontext in action with real-world data analysis workflows.

---

## 🐦 Bird Demo - Live Example

The `bird_demo/` workspace provides a comprehensive example of metacontext usage across multiple file types. This demo showcases real-world data analysis workflows and how metacontext generates intelligent metadata for each file type.

### Demo Contents

The bird demo processes ornithological data through various formats and generates metacontext for each:

- **📊 Tabular Data**: CSV and Excel files with bird measurements and taxonomic data
- **🤖 Machine Learning**: Trained RandomForest classifier models for species prediction  
- **🗺️ Geospatial Data**: GeoJSON and GeoPackage files with bird observation locations
- **🖼️ Media Files**: Pixel art bird images with embedded metadata analysis

### Quick Setup & Run

```bash
# Navigate to bird_demo workspace
cd bird_demo/

# Create .env file with your API key
echo "GEMINI_API_KEY=your-api-key-here" > .env
# Alternative: echo "OPENAI_API_KEY=your-api-key-here" > .env

# Install dependencies and run demo
make install
make demo
```

### What the Demo Does

The demo processes several bird-related files and generates comprehensive metacontext:

1. **CSV Processing** (`birdos_expanded.csv`):
   ```python
   # Creates processed dataset with features like:
   # - species_name, taxonomic_family  
   # - asdawas (wing measurement), beak_length
   # - nocturnal_diurnal behavior, primary_diet
   # - Generates: csv_metacontext.yaml
   ```

2. **Exploratory Data Analysis**:
   ```python
   # Statistical analysis of the dataset
   # - Column profiling and data quality assessment
   # - Generates: eda_csv_metacontext.yaml
   ```

3. **Machine Learning Models**:
   ```python
   # Trains RandomForest classifier for species prediction
   # - Model architecture and hyperparameter analysis
   # - Generates: bird_classification_model_metacontext.yaml
   ```

4. **Geospatial Analysis**:
   ```python
   # Processes bird observation locations
   # - Coordinate system analysis, spatial relationships
   # - Generates: filtered_locations_metacontext.yaml
   ```

5. **Media Processing**:
   ```python
   # Creates pixel art bird image
   # - Image metadata, dimensions, color analysis
   # - Generates: pixel_bird_metacontext.yaml
   ```

### Example Output Structure

Each generated metacontext file follows the universal schema:

```yaml
metacontext_version: 0.3.0
generation_info:
  generated_at: "2025-01-06T12:00:00Z"
  total_tokens: 1247
  estimated_cost_usd: 0.00015
file_info:
  filename: "processed_bird_data.csv"
  file_size_bytes: 15234
tabular_context:  # File-type specific extension
  deterministic_metadata:
    row_count: 150
    column_count: 8
    memory_usage_bytes: 9840
  ai_enrichment:
    data_purpose: "Ornithological measurements for species classification..."
    quality_assessment: "Clean dataset with no missing values..."
    business_context: "Research dataset for understanding bird morphology..."
```

### Demo File Structure

```
bird_demo/
├── Makefile              # Simple commands to run demo
├── .env                  # Your API keys (create this)
├── bird_demo/
│   ├── demo.py          # Main demo script
│   ├── models.py        # ML model definitions
│   └── data/            # Input datasets
│       ├── birdos_expanded.csv
│       ├── birdos_locations.geojson
│       └── ...
├── output/              # Generated files + metacontext
│   ├── csv_metacontext.yaml
│   ├── bird_classification_model_metacontext.yaml
│   ├── filtered_locations_metacontext.yaml
│   └── ...
└── scripts/             # Processing utilities
```

This demo demonstrates how metacontext automatically adapts to different file types and provides relevant, intelligent analysis for each domain (tabular, geospatial, ML, media).

---

## 🌟 Philosophy

**"Every data file should tell its own story."**

In a world where data is created faster than it can be documented, metacontext makes intelligence automatic. No more lost context, no more mystery files, no more "what was this for?" moments.

Your data becomes self-documenting, immediately useful, and ready to share with anyone who needs to understand it.

**Start making your data intelligent today.**