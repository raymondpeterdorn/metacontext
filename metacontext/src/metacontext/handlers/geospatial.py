"""Geospatial data handler for GeoJSON, Shapefile, GeoTIFF, etc.

This handler processes geospatial data files to extract metadata using
both deterministic techniques and AI enrichment. It implements the architectural
patterns defined in the central architecture reference and uses the unified
geospatial interface for consistent spatial analysis.

See:
- architecture_reference.ArchitecturalComponents.TWO_TIER_ARCHITECTURE
- architecture_reference.ArchitecturalComponents.SCHEMA_FIRST_LLM
- interfaces.geospatial.GeospatialAnalyzer for unified spatial operations
"""

import logging
import math
import tempfile
from pathlib import Path
from typing import Any, ClassVar

import numpy as np

from metacontext.ai.handlers.companions.template_adapter import (
    CompanionTemplateAdapter,
)
from metacontext.ai.handlers.llms.prompt_constraints import (
    COMMON_FIELD_CONSTRAINTS,
    build_schema_constraints,
    calculate_response_limits,
)
from metacontext.handlers.base import BaseFileHandler, register_handler
from metacontext.interfaces.geospatial import (
    GeospatialInterfaceFactory,
    VectorGeospatialAnalyzer,
    RasterGeospatialAnalyzer,
)
from metacontext.schemas.extensions.geospatial import (
    GeospatialRasterContext,
    GeospatialVectorContext,
    RasterAIEnrichment,
    RasterDeterministicMetadata,
    VectorAIEnrichment,
    VectorDeterministicMetadata,
)

logger = logging.getLogger(__name__)

# Try to import optional geospatial libraries
try:
    import rasterio

    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False
    rasterio = None

try:
    import geopandas as gpd

    GEOPANDAS_AVAILABLE = True
except ImportError:
    GEOPANDAS_AVAILABLE = False
    gpd = None

try:
    import fiona

    FIONA_AVAILABLE = True
except ImportError:
    FIONA_AVAILABLE = False
    fiona = None


@register_handler
class GeospatialHandler(BaseFileHandler):
    """Handler for geospatial data files using extension composition pattern.

    This handler implements the extension pattern where:
    - Pure geospatial files (.kml, .kmz) are handled directly
    - Vector geospatial files (.gpkg, .shp, .geojson) delegate to CSVHandler + geospatial extension
    - Raster geospatial files (.tif, .tiff, .nc) delegate to MediaHandler + geospatial extension
    
    Supports: GeoJSON, Shapefile, GeoTIFF, NetCDF, KML files
    Extensions: geospatial_raster_context, geospatial_vector_context
    """

    # Pure geospatial formats handled directly by this handler
    pure_geospatial_extensions: ClassVar[list[str]] = [
        ".kml",
        ".kmz",
    ]
    
    # Vector geospatial formats that should use tabular + geospatial composition
    vector_geospatial_extensions: ClassVar[list[str]] = [
        ".geojson",
        ".json",  # GeoJSON - has attributes like GPKG/SHP
        ".shp",
        ".gpkg",  # SQLite-based, needs special handling in composition
    ]
    
    # Raster geospatial formats that should use media + geospatial composition
    raster_geospatial_extensions: ClassVar[list[str]] = [
        ".tif",
        ".tiff",
        ".nc",
        ".hdf",
    ]

    supported_extensions: ClassVar[list[str]] = (
        pure_geospatial_extensions + 
        vector_geospatial_extensions + 
        raster_geospatial_extensions
    )

    def __init__(self) -> None:
        """Initialize the geospatial handler with unified interface."""
        self._vector_analyzer = VectorGeospatialAnalyzer()
        self._raster_analyzer = RasterGeospatialAnalyzer()

    def can_handle(self, file_path: Path, data_object: object | None = None) -> bool:
        """Check if this is a geospatial data file using extension pattern.
        
        This handler now follows the extension pattern:
        - Only handles pure geospatial files (.kml, .kmz) directly
        - Vector geospatial files (.geojson, .gpkg, .shp) should be handled by CSVHandler + geospatial extension
        - Raster geospatial files should be handled by MediaHandler + geospatial extension
        """
        file_ext = file_path.suffix.lower()
        logger.debug("🔍 TASK-2.8 GeospatialHandler.can_handle: Evaluating file %s (ext: %s)", file_path.name, file_ext)
        
        # Only handle pure geospatial formats directly (KML/KMZ only)
        can_handle_result = file_ext in self.pure_geospatial_extensions
        
        if can_handle_result:
            logger.debug("✓ TASK-2.8 Extension pattern: %s matches pure_geospatial_extensions %s", file_ext, self.pure_geospatial_extensions)
        else:
            logger.debug("✗ TASK-2.8 Extension pattern: %s not in pure_geospatial_extensions %s (should use composition)", file_ext, self.pure_geospatial_extensions)
            
        return can_handle_result

    def get_required_extensions(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> list[str]:
        """Return required extensions for pure geospatial data.
        
        Note: In the extension pattern, this only applies to pure geospatial files
        that are handled directly by this handler (.kml, .kmz, GeoJSON).
        Vector/raster geospatial files are handled by other handlers with extensions.
        """
        file_ext = file_path.suffix.lower()
        
        # Pure geospatial files handled directly
        if file_ext in self.pure_geospatial_extensions:
            return ["geospatial_vector_context"]  # KML/KMZ are vector-like
            
        # GeoJSON as pure geospatial
        if file_ext == ".json" and self.can_handle(file_path):
            return ["geospatial_vector_context"]
            
        # This handler no longer handles raster/vector files with composition
        return []

    def _is_raster_file(self, file_path: Path) -> bool:
        """Determine if file is raster or vector data."""
        raster_extensions = {".tif", ".tiff", ".nc", ".hdf"}
        return file_path.suffix.lower() in raster_extensions

    def fast_probe(self, file_path: Path) -> dict[str, object]:
        """Fast probe to check file compatibility and get basic metadata."""
        if not file_path.exists():
            return {"can_handle": False, "error": "File does not exist"}

        file_size = file_path.stat().st_size
        extension = file_path.suffix.lower()

        try:
            if self._is_raster_file(file_path):
                return self._probe_raster_file(file_path, file_size)
            return self._probe_vector_file(file_path, file_size)
        except Exception as e:
            logger.warning("Error probing geospatial file %s: %s", file_path, e)
            return {
                "can_handle": True,
                "file_size": file_size,
                "extension": extension,
                "probe_error": str(e),
            }

    def _probe_raster_file(self, file_path: Path, file_size: int) -> dict[str, object]:
        """Probe raster file for basic metadata."""
        probe_result = {
            "can_handle": True,
            "data_type": "raster",
            "file_size": file_size,
            "extension": file_path.suffix.lower(),
        }

        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(file_path) as dataset:
                    probe_result.update(
                        {
                            "dimensions": [dataset.width, dataset.height],
                            "band_count": dataset.count,
                            "crs": str(dataset.crs) if dataset.crs else None,
                            "bounds": list(dataset.bounds),
                        },
                    )
            except Exception as e:
                probe_result["probe_error"] = str(e)

        return probe_result

    def _probe_vector_file(self, file_path: Path, file_size: int) -> dict[str, object]:
        """Probe vector file for basic metadata."""
        probe_result = {
            "can_handle": True,
            "data_type": "vector",
            "file_size": file_size,
            "extension": file_path.suffix.lower(),
        }

        try:
            if file_path.suffix.lower() in [".geojson", ".json"]:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    if data.get("type") == "FeatureCollection":
                        probe_result["feature_count"] = len(data.get("features", []))
                        if data.get("features"):
                            probe_result["geometry_type"] = (
                                data["features"][0].get("geometry", {}).get("type")
                            )
            elif GEOPANDAS_AVAILABLE:
                # Use geopandas for other vector formats
                gdf = gpd.read_file(file_path)
                probe_result.update(
                    {
                        "feature_count": len(gdf),
                        "geometry_type": gdf.geometry.type.iloc[0]
                        if len(gdf) > 0
                        else None,
                        "crs": str(gdf.crs) if gdf.crs else None,
                    },
                )
        except Exception as e:
            probe_result["probe_error"] = str(e)

        return probe_result

    def generate_context(
        self,
        file_path: Path,
        data_object: object | None = None,
        codebase_context: dict[str, object] | None = None,
        ai_companion: object | None = None,
    ) -> dict[str, Any]:
        """Generate geospatial context with deterministic metadata and AI enrichment."""
        try:
            # Check for companion mode vs API mode
            if ai_companion and hasattr(ai_companion, "companion_type"):
                # Companion mode - analyze deterministic data and use simplified integration
                if self._is_raster_file(file_path):
                    deterministic_metadata = self._analyze_raster_deterministic(
                        file_path,
                    )
                    result = self._generate_raster_companion_context(
                        file_path,
                        deterministic_metadata,
                        ai_companion,
                    )
                else:
                    deterministic_metadata = self._analyze_vector_deterministic(file_path)
                    result = self._generate_vector_companion_context(
                        file_path,
                        deterministic_metadata,
                        ai_companion,
                    )
                    
                    # For vector data in companion mode, try to add tabular analysis
                    try:
                        tabular_context = self._add_tabular_analysis(
                            file_path, data_object, ai_companion
                        )
                        if tabular_context:
                            result.update(tabular_context)
                    except Exception as e:
                        logger.debug("Could not add tabular analysis in companion mode: %s", e)
                        
                return result

            # Extract semantic knowledge for enhanced context (API mode)
            semantic_knowledge_text = "No semantic knowledge extracted from codebase."
            if (
                ai_companion
                and hasattr(ai_companion, "codebase_context")
                and ai_companion.codebase_context
            ):
                try:
                    # Check if we have semantic knowledge available
                    if (
                        hasattr(ai_companion.codebase_context, "ai_enrichment")
                        and ai_companion.codebase_context.ai_enrichment
                        and hasattr(
                            ai_companion.codebase_context.ai_enrichment,
                            "semantic_knowledge",
                        )
                    ):
                        semantic_knowledge = ai_companion.codebase_context.ai_enrichment.semantic_knowledge

                        # Format semantic knowledge for AI analysis
                        if semantic_knowledge and hasattr(
                            semantic_knowledge,
                            "geospatial_fields",
                        ):
                            field_descriptions = []
                            for (
                                field_name,
                                field_info,
                            ) in semantic_knowledge.geospatial_fields.items():
                                if field_info.pydantic_description:
                                    field_descriptions.append(
                                        f"- {field_name}: {field_info.pydantic_description}",
                                    )
                                elif field_info.definition:
                                    field_descriptions.append(
                                        f"- {field_name}: {field_info.definition}",
                                    )

                            if field_descriptions:
                                semantic_knowledge_text = (
                                    "Semantic knowledge from codebase:\n"
                                    + "\n".join(field_descriptions)
                                )
                except (AttributeError, KeyError, TypeError):
                    pass  # Use default semantic knowledge text

            # Determine if raster or vector
            if self._is_raster_file(file_path):
                result = self._generate_raster_context(
                    file_path,
                    ai_companion,
                    semantic_knowledge_text,
                    data_object,
                )
            else:
                result = self._generate_vector_context(
                    file_path,
                    ai_companion,
                    semantic_knowledge_text,
                )
                
                # For vector data, try to add tabular analysis
                try:
                    tabular_context = self._add_tabular_analysis(
                        file_path, 
                        data_object, 
                        ai_companion,
                    )
                    if tabular_context:
                        result.update(tabular_context)
                except Exception as e:
                    logger.debug("Could not add tabular analysis: %s", e)
                    
            return result
        except Exception:
            logger.exception("Error generating geospatial context for %s", file_path)
            return {"error": "Failed to generate geospatial context"}

    def _add_tabular_analysis(
        self, 
        file_path: Path, 
        data_object: object | None, 
        ai_companion: object | None,
    ) -> dict[str, object] | None:
        """Add tabular analysis for vector data with attributes."""
        try:
            # Import CSVHandler for tabular analysis
            from metacontext.handlers.tabular import CSVHandler
            
            csv_handler = CSVHandler()
            tabular_data = None
            
            # Try to extract tabular data for analysis
            if data_object is not None and hasattr(data_object, 'drop'):
                # For GeoDataFrame, drop geometry column for tabular analysis
                tabular_data = data_object.drop(columns=['geometry'], errors='ignore')
                if len(tabular_data.columns) == 0:
                    return None  # No non-spatial columns to analyze
                    
                # Use CSVHandler to analyze the tabular portion
                return csv_handler.generate_context(
                    file_path, 
                    data_object=tabular_data,
                    ai_companion=ai_companion,
                )
            elif file_path.suffix.lower() in ['.gpkg', '.shp', '.geojson']:
                # For spatial files, check if CSVHandler can extract tabular data
                if csv_handler.can_handle(file_path, data_object):
                    return csv_handler.generate_context(
                        file_path, 
                        data_object=data_object,
                        ai_companion=ai_companion,
                    )
                    
            return None
            
        except Exception as e:
            logger.debug("Tabular analysis failed: %s", e)
            return None

    def _add_media_analysis(
        self, 
        file_path: Path, 
        data_object: object | None, 
        ai_companion: object | None,
    ) -> dict[str, object] | None:
        """Add media analysis for raster geospatial data."""
        try:
            # Import MediaHandler for media analysis
            from metacontext.handlers.media import MediaHandler
            
            media_handler = MediaHandler()
            
            # Check if MediaHandler can handle this file
            if media_handler.can_handle(file_path, data_object):
                return media_handler.generate_context(
                    file_path, 
                    data_object=data_object,
                    ai_companion=ai_companion,
                )
            
            return None
            
        except Exception as e:
            logger.debug("Media analysis failed: %s", e)
            return None

    def _generate_raster_context(
        self,
        file_path: Path,
        ai_companion: object | None,
        semantic_knowledge: str | None = None,
        data_object: object | None = None,
    ) -> dict[str, Any]:
        """Generate context for raster geospatial data with media analysis."""
        # Deterministic analysis
        deterministic_metadata = self._analyze_raster_deterministic(file_path)

        # AI enrichment
        ai_enrichment = None
        if ai_companion and hasattr(ai_companion, "generate_with_schema"):
            ai_enrichment = self._generate_raster_ai_enrichment(
                file_path,
                deterministic_metadata,
                ai_companion,
                semantic_knowledge,
            )

        # Prepare result with geospatial context
        result = {
            "geospatial_raster_context": GeospatialRasterContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=ai_enrichment,
            ).model_dump(),
        }
        
        # Add media analysis for comprehensive raster analysis
        try:
            media_context = self._add_media_analysis(
                file_path, 
                data_object, 
                ai_companion,
            )
            if media_context:
                result.update(media_context)
        except Exception as e:
            logger.debug("Could not add media analysis: %s", e)
        
        return result

    def _generate_vector_context(
        self,
        file_path: Path,
        ai_companion: object | None,
        semantic_knowledge: str | None = None,
    ) -> dict[str, Any]:
        """Generate context for vector geospatial data."""
        # Deterministic analysis
        deterministic_metadata = self._analyze_vector_deterministic(file_path)

        # AI enrichment
        ai_enrichment = None
        if ai_companion and hasattr(ai_companion, "generate_with_schema"):
            ai_enrichment = self._generate_vector_ai_enrichment(
                file_path,
                deterministic_metadata,
                ai_companion,
                semantic_knowledge,
            )

        return {
            "geospatial_vector_context": GeospatialVectorContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=ai_enrichment,
            ),
        }

    def _analyze_raster_deterministic(
        self,
        file_path: Path,
    ) -> RasterDeterministicMetadata:
        """Analyze raster file deterministically."""
        metadata = RasterDeterministicMetadata()

        if RASTERIO_AVAILABLE:
            try:
                with rasterio.open(file_path) as dataset:
                    metadata.format = dataset.driver
                    metadata.crs = str(dataset.crs) if dataset.crs else None
                    metadata.bounds = list(dataset.bounds)
                    metadata.pixel_dimensions = [dataset.width, dataset.height]
                    metadata.band_count = dataset.count
                    metadata.pixel_size = [
                        abs(dataset.transform[0]),
                        abs(dataset.transform[4]),
                    ]
                    metadata.data_type = (
                        str(dataset.dtypes[0]) if dataset.dtypes else None
                    )
                    metadata.nodata_value = dataset.nodata

                    # Enhanced spatial metadata
                    if dataset.crs:
                        crs_info = self._analyze_crs(str(dataset.crs))
                        metadata.crs_type = crs_info["type"]
                        metadata.crs_units = crs_info["units"]
                        metadata.bounds_crs = str(dataset.crs)
                        
                        # Calculate area coverage and resolution assessment
                        if dataset.bounds and metadata.pixel_size:
                            metadata.area_coverage_km2 = self._calculate_area_coverage(
                                dataset.bounds, str(dataset.crs),
                            )
                            metadata.spatial_resolution = self._assess_spatial_resolution(
                                metadata.pixel_size, crs_info["units"],
                            )

                    # Try to get compression info
                    if hasattr(dataset, "compression"):
                        metadata.compression = dataset.compression

            except Exception as e:
                logger.warning("Error analyzing raster file %s: %s", file_path, e)
        else:
            # Fallback: Use lightweight geospatial extension for worldfile reading
            try:
                from metacontext.extensions.geospatial import GeospatialExtension
                
                extension = GeospatialExtension()
                spatial_info = extension.extract_spatial_metadata(file_path)
                
                if spatial_info.get("crs"):
                    metadata.crs = spatial_info["crs"]
                    metadata.crs_type = "geographic" if "4326" in spatial_info["crs"] else "unknown"
                    metadata.crs_units = "degrees" if "4326" in spatial_info["crs"] else "unknown"
                    metadata.bounds_crs = spatial_info["crs"]
                
                if spatial_info.get("bounds_estimate"):
                    bounds = spatial_info["bounds_estimate"]
                    metadata.bounds = [
                        bounds["min_lon"], bounds["min_lat"],
                        bounds["max_lon"], bounds["max_lat"]
                    ]
                
                if spatial_info.get("pixel_dimensions"):
                    metadata.pixel_dimensions = spatial_info["pixel_dimensions"]
                    
                if spatial_info.get("pixel_size"):
                    metadata.pixel_size = spatial_info["pixel_size"]
                    # Estimate basic resolution
                    avg_pixel_size = sum(spatial_info["pixel_size"]) / 2
                    if metadata.crs_units == "degrees":
                        metadata.spatial_resolution = self._assess_spatial_resolution(
                            spatial_info["pixel_size"], "degrees"
                        )
                        
                # Set basic format info
                metadata.format = "TIFF" if file_path.suffix.lower() in {".tif", ".tiff"} else "unknown"
                
            except Exception as e:
                logger.debug("Worldfile fallback failed: %s", e)

        return metadata

    def _analyze_vector_deterministic(
        self,
        file_path: Path,
    ) -> VectorDeterministicMetadata:
        """Analyze vector file deterministically."""
        metadata = VectorDeterministicMetadata()

        try:
            if file_path.suffix.lower() in [".geojson", ".json"]:
                with file_path.open(encoding="utf-8") as f:
                    data = json.load(f)
                    metadata.format = "GeoJSON"
                    if data.get("type") == "FeatureCollection":
                        features = data.get("features", [])
                        metadata.feature_count = len(features)
                        if features:
                            metadata.geometry_type = (
                                features[0].get("geometry", {}).get("type")
                            )
                            # Get property names
                            properties = features[0].get("properties", {})
                            metadata.attribute_fields = list(properties.keys())
                            
                            # Enhanced spatial metadata for GeoJSON
                            metadata.coordinate_precision = self._analyze_coordinate_precision(features)
                            crs_info = data.get("crs")
                            if crs_info:
                                metadata.crs = str(crs_info)
                                crs_analysis = self._analyze_crs(str(crs_info))
                                metadata.crs_type = crs_analysis["type"]
                                metadata.crs_units = crs_analysis["units"]

            elif GEOPANDAS_AVAILABLE:
                gdf = gpd.read_file(file_path)
                metadata.format = file_path.suffix.upper().lstrip(".")
                metadata.feature_count = len(gdf)
                metadata.crs = str(gdf.crs) if gdf.crs else None
                if len(gdf) > 0:
                    metadata.geometry_type = gdf.geometry.type.iloc[0]
                    metadata.bounds = list(gdf.total_bounds)
                    metadata.attribute_fields = [
                        col for col in gdf.columns if col != "geometry"
                    ]
                    
                    # Enhanced spatial metadata
                    if gdf.crs:
                        crs_info = self._analyze_crs(str(gdf.crs))
                        metadata.crs_type = crs_info["type"]
                        metadata.crs_units = crs_info["units"]
                        metadata.bounds_crs = str(gdf.crs)
                        
                        # Calculate area coverage
                        if not np.isnan(gdf.total_bounds).any():
                            metadata.area_coverage_km2 = self._calculate_area_coverage(
                                gdf.total_bounds, str(gdf.crs),
                            )
                    
                    # Analyze coordinate precision
                    metadata.coordinate_precision = self._analyze_gdf_coordinate_precision(gdf)
                    
                    # Check for spatial index
                    metadata.spatial_index_available = hasattr(gdf, 'sindex') and gdf.sindex is not None

        except (OSError, ValueError, json.JSONDecodeError) as e:
            logger.warning("Error analyzing vector file %s: %s", file_path, e)

        return metadata

    def analyze_deterministic(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> dict[str, object]:
        """Analyze file without AI - deterministic analysis only."""
        if self._is_raster_file(file_path):
            metadata = self._analyze_raster_deterministic(file_path)
            return {"raster_metadata": metadata.model_dump()}
        metadata = self._analyze_vector_deterministic(file_path)
        return {"vector_metadata": metadata.model_dump()}

    def analyze_deep(
        self,
        file_path: Path,
        data_object: object = None,
        ai_companion: object | None = None,
        deterministic_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Deep analysis using AI and heavy computation."""
        if (
            not ai_companion
            or not hasattr(ai_companion, "is_available")
            or not ai_companion.is_available()
        ):
            return {"error": "AI companion not available for deep analysis"}

        try:
            if self._is_raster_file(file_path):
                metadata = (
                    deterministic_context.get("raster_metadata")
                    if deterministic_context
                    else None
                )
                if not metadata:
                    metadata = self._analyze_raster_deterministic(
                        file_path,
                    ).model_dump()

                ai_enrichment = self._generate_raster_ai_enrichment(
                    file_path,
                    RasterDeterministicMetadata.model_validate(metadata),
                    ai_companion,
                )
                return {
                    "raster_ai_enrichment": ai_enrichment.model_dump()
                    if ai_enrichment
                    else None,
                }
            metadata = (
                deterministic_context.get("vector_metadata")
                if deterministic_context
                else None
            )
            if not metadata:
                metadata = self._analyze_vector_deterministic(file_path).model_dump()

            ai_enrichment = self._generate_vector_ai_enrichment(
                file_path,
                VectorDeterministicMetadata.model_validate(metadata),
                ai_companion,
            )
            return {
                "vector_ai_enrichment": ai_enrichment.model_dump()
                if ai_enrichment
                else None,
            }
        except Exception:
            logger.exception("Error in deep analysis for %s", file_path)
            return {"error": "Deep analysis failed"}

    def _build_raster_constraints(
        self,
        file_path: Path,
        metadata: RasterDeterministicMetadata,
    ) -> str:
        """Build constraints for raster AI enrichment with codebase context scanning."""
        # Calculate complexity based on bands and dimensions
        complexity_factor = 1.0
        if metadata.band_count:
            complexity_factor *= min(2.0, 1.0 + (metadata.band_count / 10))
        if metadata.pixel_dimensions:
            total_pixels = metadata.pixel_dimensions[0] * metadata.pixel_dimensions[1]
            if total_pixels > 10_000_000:  # Large raster
                complexity_factor *= 1.5

        max_total_chars, max_field_chars = calculate_response_limits(
            base_fields=7,  # ForensicAIEnrichment base fields
            extended_fields=6,  # RasterAIEnrichment specific fields
            complexity_factor=complexity_factor,
        )

        field_constraints = {
            **COMMON_FIELD_CONSTRAINTS,
            "geographic_region": "Region name + key characteristics",
            "data_source_analysis": "Data source + collection method",
            "temporal_analysis": "Time period + temporal patterns",
            "spatial_resolution_assessment": "Resolution quality + limitations",
            "band_analysis": "Band purposes + spectral info",
            "applications": "2-3 key use cases",
        }

        # Enhanced instruction for AI-first raster analysis
        base_instruction = """Analyze this raster geospatial dataset using AI-first spatial inference.

RASTER ANALYSIS APPROACH:
Since raster files often lack embedded metadata, you should:

1. **CODEBASE SCANNING**: Look for spatial clues in:
   - Variable names containing coordinates, CRS codes, or geographic terms
   - Comments mentioning regions, projections, or coordinate systems  
   - File paths suggesting geographic areas or data sources
   - Function names related to spatial processing or coordinate transforms

2. **FILENAME ANALYSIS**: Extract spatial hints from:
   - Geographic region names (e.g., "california", "north_america", "arctic")
   - Coordinate system codes (e.g., "utm", "wgs84", "epsg4326")
   - Resolution indicators (e.g., "30m", "1km", "high_res")
   - Data source indicators (e.g., "landsat", "modis", "sentinel")

3. **CONTEXTUAL INFERENCE**: Use project context to determine:
   - Likely coordinate reference systems based on geographic focus
   - Probable data sources based on research domain
   - Expected spatial resolution based on analysis type
   - Geographic coverage based on study area

4. **WORLDFILE DETECTION**: Check if companion .tfw, .tifw, or .wld files exist
   that might contain georeferencing information

PRIORITY: Focus on extracting maximum spatial context even when technical metadata is absent."""
        
        constraints = build_schema_constraints(
            max_total_chars=max_total_chars,
            max_field_chars=max_field_chars,
            field_descriptions=field_constraints,
            complexity_context=f"Raster: {metadata.band_count or 'unknown'} bands",
        )

        return f"{base_instruction}\\n\\n{constraints}"

    def _build_vector_constraints(
        self,
        file_path: Path,
        metadata: VectorDeterministicMetadata,
    ) -> str:
        """Build constraints for vector AI enrichment."""
        # Calculate complexity based on features and properties
        complexity_factor = 1.0
        if metadata.feature_count:
            if metadata.feature_count > 10000:
                complexity_factor *= 1.5
            elif metadata.feature_count < 10:
                complexity_factor *= 0.8

        property_count = len(metadata.attribute_fields or [])
        complexity_factor *= min(2.0, 1.0 + (property_count / 20))

        max_total_chars, max_field_chars = calculate_response_limits(
            base_fields=7,  # ForensicAIEnrichment base fields
            extended_fields=6,  # VectorAIEnrichment specific fields
            complexity_factor=complexity_factor,
        )

        field_constraints = {
            **COMMON_FIELD_CONSTRAINTS,
            "geographic_region": "Region name + coverage area",
            "data_source_analysis": "Data source + collection agency",
            "attribute_analysis": "Key attributes + data quality",
            "spatial_accuracy_assessment": "Accuracy + precision notes",
            "geometry_analysis": "Geometry type + complexity",
            "applications": "2-3 practical applications",
        }

        base_instruction = "Analyze this vector geospatial dataset"
        constraints = build_schema_constraints(
            max_total_chars=max_total_chars,
            max_field_chars=max_field_chars,
            field_descriptions=field_constraints,
            complexity_context=f"Vector: {metadata.feature_count or 'unknown'} features",
        )

        return f"{base_instruction} and provide insights that fit within these STRICT LIMITS:\\n\\n{constraints}"

    def _generate_raster_ai_enrichment(
        self,
        file_path: Path,
        metadata: RasterDeterministicMetadata,
        ai_companion: object,
        semantic_knowledge: str | None = None,
    ) -> RasterAIEnrichment | None:
        """Generate AI enrichment for raster data with enhanced spatial context."""
        try:
            # Enhanced context data with spatial clues for AI-first analysis
            context_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "raster_metadata": metadata.model_dump(),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "semantic_knowledge": semantic_knowledge
                or "No semantic knowledge available from codebase",
            }
            
            # Add spatial context clues for AI analysis
            spatial_clues = self._extract_spatial_clues(file_path)
            context_data["spatial_clues"] = spatial_clues

            instruction = self._build_raster_constraints(file_path, metadata)

            # Type: ignore needed because ai_companion is typed as object for flexibility
            return ai_companion.generate_with_schema(  # type: ignore[attr-defined]
                schema_class=RasterAIEnrichment,
                context_data=context_data,
                instruction=instruction,
            )
        except Exception:
            logger.exception("Error generating raster AI enrichment")
            return None
    
    def _extract_spatial_clues(self, file_path: Path) -> dict[str, Any]:
        """Extract spatial context clues using unified interface.
        
        Args:
            file_path: Path to the geospatial file
            
        Returns:
            Dictionary of spatial context clues for AI enrichment
        """
        # Use the unified interface method for consistent spatial clue extraction
        data_type = GeospatialInterfaceFactory.determine_data_type(file_path)
        analyzer = GeospatialInterfaceFactory.create_analyzer(file_path, data_type)
        
        return analyzer.extract_spatial_clues(file_path)

    def _generate_vector_ai_enrichment(
        self,
        file_path: Path,
        metadata: VectorDeterministicMetadata,
        ai_companion: object,
        semantic_knowledge: str | None = None,
    ) -> VectorAIEnrichment | None:
        """Generate AI enrichment for vector data."""
        try:
            context_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "vector_metadata": metadata.model_dump(),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
                "semantic_knowledge": semantic_knowledge
                or "No semantic knowledge available from codebase",
            }

            instruction = self._build_vector_constraints(file_path, metadata)

            # Type: ignore needed because ai_companion is typed as object for flexibility
            return ai_companion.generate_with_schema(  # type: ignore[attr-defined]
                schema_class=VectorAIEnrichment,
                context_data=context_data,
                instruction=instruction,
            )
        except Exception:
            logger.exception("Error generating vector AI enrichment")
            return None

    # Prompt configuration for bulk analysis
    PROMPT_CONFIG: ClassVar[dict[str, str]] = {
        "vector_analysis": "templates/geospatial/vector_analysis.yaml",
        "raster_analysis": "templates/geospatial/raster_analysis.yaml",
    }

    def get_bulk_prompts(
        self,
        file_path: Path,
        data_object: object = None,
    ) -> dict[str, str]:
        """Get bulk prompts for this file type from config."""
        return self.PROMPT_CONFIG.copy()

    def _generate_companion_context(
        self,
        file_path: Path,
        deterministic_metadata: object,  # This will be either Raster or Vector metadata
        ai_companion: object,
    ) -> dict[str, Any]:
        """Generate context using companion mode with simplified integration."""
        try:
            # Determine if this is raster or vector data
            if self._is_raster_file(file_path):
                return self._generate_raster_companion_context(
                    file_path,
                    deterministic_metadata,
                    ai_companion,
                )
            return self._generate_vector_companion_context(
                file_path,
                deterministic_metadata,
                ai_companion,
            )

        except Exception as e:
            logger.warning(
                "Companion mode failed, falling back to deterministic: %s",
                str(e),
            )
            # Fallback to deterministic only
            if self._is_raster_file(file_path):
                return GeospatialRasterContext(
                    deterministic_metadata=deterministic_metadata,
                    ai_enrichment=None,
                ).model_dump()
            return GeospatialVectorContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=None,
            ).model_dump()

    def _generate_raster_companion_context(
        self,
        file_path: Path,
        deterministic_metadata: RasterDeterministicMetadata,
        ai_companion: object,
    ) -> dict[str, Any]:
        """Generate raster context using companion mode with template adaptation."""
        try:
            adapter = CompanionTemplateAdapter()

            # Create context variables for template substitution
            context_variables = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "data_type": "raster",
                "bands": deterministic_metadata.bands or 0,
                "width": deterministic_metadata.width or 0,
                "height": deterministic_metadata.height or 0,
                "crs": deterministic_metadata.crs or "unknown",
            }

            # Load and adapt the geospatial analysis template
            template_path = "geospatial/raster_analysis.yaml"
            template_data = adapter.load_api_template(template_path)

            # Generate companion prompt
            companion_prompt = adapter.generate_companion_prompt(
                template_data,
                context_variables,
            )

            # Create a temporary response file path
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
                response_file_path = Path(tmp_file.name)

            # Display prompt and wait for companion response
            parsed_data = ai_companion.display_prompt_and_wait(
                companion_prompt, response_file_path
            )

            if parsed_data:
                # Validate response structure
                schema_path = (
                    "metacontext.schemas.extensions.geospatial.RasterAIEnrichment"
                )
                is_valid, validation_errors = adapter.validate_response_structure(
                    parsed_data,
                    schema_path,
                )
                if not is_valid:
                    logger.error("Response validation failed: %s", validation_errors)
                    return self._generate_fallback_raster_context(
                        file_path, deterministic_metadata, ai_companion
                    )

                # Convert to Pydantic instance
                ai_enrichment, conversion_errors = adapter.convert_yaml_to_pydantic(
                    parsed_data,
                    schema_path,
                )
                if conversion_errors:
                    logger.error("Response conversion failed: %s", conversion_errors)
                    return self._generate_fallback_raster_context(
                        file_path, deterministic_metadata, ai_companion
                    )

                return GeospatialRasterContext(
                    deterministic_metadata=deterministic_metadata,
                    ai_enrichment=ai_enrichment,
                ).model_dump()

            return self._generate_fallback_raster_context(
                file_path, deterministic_metadata, ai_companion
            )

        except Exception as e:
            logger.exception("Companion analysis failed: %s", e)
            return self._generate_fallback_raster_context(
                file_path, deterministic_metadata, ai_companion
            )

    def _generate_fallback_raster_context(
        self,
        file_path: Path,
        deterministic_metadata: RasterDeterministicMetadata,
        ai_companion: object,
    ) -> dict[str, Any]:
        """Generate fallback raster context when companion fails."""

        # Return deterministic data with companion marker
        return {
            "geospatial_raster_context": GeospatialRasterContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=RasterAIEnrichment(
                    spatial_analysis={
                        "companion_mode": True,
                        "companion_type": ai_companion.companion_type,
                        "analysis_note": f"Raster analyzed using {ai_companion.companion_type} companion",
                    },
                    semantic_tags=[f"{ai_companion.companion_type}_analyzed"],
                    contextual_insights=[
                        f"Raster file processed via {ai_companion.companion_type} integration",
                    ],
                ),
            ).model_dump()
        }

    def _generate_vector_companion_context(
        self,
        file_path: Path,
        deterministic_metadata: VectorDeterministicMetadata,
        ai_companion: object,
    ) -> dict[str, Any]:
        """Generate vector context using companion mode with template adaptation."""
        try:
            adapter = CompanionTemplateAdapter()

            # Create context variables for template substitution
            context_variables = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "data_type": "vector",
                "feature_count": deterministic_metadata.feature_count or 0,
                "geometry_type": deterministic_metadata.geometry_type or "unknown",
                "crs": deterministic_metadata.crs or "unknown",
                "attribute_fields": deterministic_metadata.attribute_fields or [],
            }

            # Load and adapt the geospatial analysis template
            template_path = "geospatial/vector_analysis.yaml"
            template_data = adapter.load_api_template(template_path)

            # Generate companion prompt
            companion_prompt = adapter.generate_companion_prompt(
                template_data,
                context_variables,
            )

            # Create a temporary response file path
            with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as tmp_file:
                response_file_path = Path(tmp_file.name)

            # Display prompt and wait for companion response
            parsed_data = ai_companion.display_prompt_and_wait(
                companion_prompt, response_file_path
            )

            if parsed_data:
                # Validate response structure
                schema_path = (
                    "metacontext.schemas.extensions.geospatial.VectorAIEnrichment"
                )
                is_valid, validation_errors = adapter.validate_response_structure(
                    parsed_data,
                    schema_path,
                )
                if not is_valid:
                    logger.error("Response validation failed: %s", validation_errors)
                    return self._generate_fallback_vector_context(
                        file_path, deterministic_metadata, ai_companion
                    )

                # Convert to Pydantic instance
                ai_enrichment, conversion_errors = adapter.convert_yaml_to_pydantic(
                    parsed_data,
                    schema_path,
                )
                if conversion_errors:
                    logger.error("Response conversion failed: %s", conversion_errors)
                    return self._generate_fallback_vector_context(
                        file_path, deterministic_metadata, ai_companion
                    )

                return {
                    "geospatial_vector_context": GeospatialVectorContext(
                        deterministic_metadata=deterministic_metadata,
                        ai_enrichment=ai_enrichment,
                    )
                }

            return self._generate_fallback_vector_context(
                file_path, deterministic_metadata, ai_companion
            )

        except Exception:
            logger.exception("Companion analysis failed")
            return self._generate_fallback_vector_context(
                file_path, deterministic_metadata, ai_companion
            )

    def _generate_fallback_vector_context(
        self,
        file_path: Path,
        deterministic_metadata: VectorDeterministicMetadata,
        ai_companion: object,
    ) -> dict[str, Any]:
        """Generate fallback vector context when companion fails."""

        # Return deterministic data with companion marker
        return {
            "geospatial_vector_context": GeospatialVectorContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=VectorAIEnrichment(
                    spatial_analysis={
                        "companion_mode": True,
                        "companion_type": ai_companion.companion_type,
                        "analysis_note": f"Vector analyzed using {ai_companion.companion_type} companion",
                    },
                    semantic_tags=[f"{ai_companion.companion_type}_analyzed"],
                    contextual_insights=[
                        f"Vector file processed via {ai_companion.companion_type} integration",
                    ],
                ),
            ),
        }

    def _analyze_crs(self, crs_string: str) -> dict[str, str | None]:
        """Analyze coordinate reference system using unified interface.
        
        Args:
            crs_string: CRS specification string
            
        Returns:
            Dictionary with parsed CRS information
        """
        # Use the unified interface for consistent CRS analysis
        analyzer = self._vector_analyzer  # Both analyzers have the same CRS analysis
        return analyzer.analyze_crs_info(crs_string)

    def _calculate_area_coverage(self, bounds: list[float], crs: str) -> float | None:
        """Calculate approximate area coverage in square kilometers."""
        if bounds is None or len(bounds) != 4:
            return None
            
        try:
            xmin, ymin, xmax, ymax = bounds
            
            # For geographic coordinates (degrees), use rough conversion
            crs_info = self._analyze_crs(crs)
            if crs_info["units"] == "degrees":
                # Rough approximation: 1 degree ≈ 111 km at equator
                width_km = abs(xmax - xmin) * 111
                height_km = abs(ymax - ymin) * 111
                # Apply cosine correction for latitude (rough approximation)
                avg_lat = abs((ymax + ymin) / 2)
                width_km *= math.cos(math.radians(avg_lat))
                return width_km * height_km
            
            elif crs_info["units"] == "meters":
                # Convert square meters to square kilometers
                width_m = abs(xmax - xmin)
                height_m = abs(ymax - ymin)
                return (width_m * height_m) / 1_000_000
                
            elif crs_info["units"] == "feet":
                # Convert square feet to square kilometers
                width_ft = abs(xmax - xmin)
                height_ft = abs(ymax - ymin)
                area_ft2 = width_ft * height_ft
                return area_ft2 / 10_763_910.4  # sq ft to sq km conversion
                
        except (ValueError, TypeError, ZeroDivisionError):
            pass
            
        return None

    def _assess_spatial_resolution(self, pixel_size: list[float], units: str | None) -> str:
        """Assess spatial resolution using unified interface.
        
        Args:
            pixel_size: List of [x_size, y_size] in CRS units
            units: CRS units ("degrees", "meters", etc.)
            
        Returns:
            Resolution category string
        """
        # Use the unified interface for consistent resolution assessment
        analyzer = self._raster_analyzer  # Use raster analyzer for resolution assessment
        return analyzer.assess_spatial_resolution(pixel_size, units)

    def _analyze_coordinate_precision(self, features: list[dict]) -> int | None:
        """Analyze coordinate precision from GeoJSON features."""
        if not features:
            return None
            
        try:
            # Sample first feature's coordinates
            first_feature = features[0]
            geometry = first_feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [])
            
            if not coordinates:
                return None
                
            # Flatten coordinates to get individual numbers
            def flatten_coords(coords):
                if isinstance(coords, (int, float)):
                    return [coords]
                elif isinstance(coords, list):
                    result = []
                    for item in coords:
                        result.extend(flatten_coords(item))
                    return result
                return []
            
            flat_coords = flatten_coords(coordinates)
            if not flat_coords:
                return None
                
            # Find max decimal places
            max_precision = 0
            for coord in flat_coords[:10]:  # Sample first 10 coordinates
                if isinstance(coord, float):
                    coord_str = str(coord)
                    if "." in coord_str:
                        decimal_places = len(coord_str.split(".")[1])
                        max_precision = max(max_precision, decimal_places)
                        
            return max_precision if max_precision > 0 else None
            
        except (KeyError, TypeError, ValueError):
            return None

    def _analyze_gdf_coordinate_precision(self, gdf) -> int | None:
        """Analyze coordinate precision from GeoPandas GeoDataFrame."""
        if GEOPANDAS_AVAILABLE and len(gdf) > 0:
            try:
                # Sample first geometry
                first_geom = gdf.geometry.iloc[0]
                if first_geom and hasattr(first_geom, "coords"):
                    coords_list = list(first_geom.coords)
                    if coords_list:
                        # Check precision of first coordinate pair
                        x, y = coords_list[0][:2]
                        max_precision = 0
                        for coord in [x, y]:
                            coord_str = str(coord)
                            if "." in coord_str:
                                decimal_places = len(coord_str.split(".")[1])
                                max_precision = max(max_precision, decimal_places)
                        return max_precision if max_precision > 0 else None
            except (AttributeError, IndexError, TypeError):
                pass
        return None
