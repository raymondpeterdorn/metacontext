"""Geospatial data handler for GeoJSON, Shapefile, GeoTIFF, etc.

This handler processes geospatial data files to extract metadata using
both deterministic techniques and AI enrichment. It implements the architectural
patterns defined in the central architecture reference.

See:
                    metadata.attribute_fields = [col for col in gdf.columns if col != "geometry"] architecture_reference.ArchitecturalComponents.TWO_TIER_ARCHITECTURE
- architecture_reference.ArchitecturalComponents.SCHEMA_FIRST_LLM
"""

import json
import logging
from pathlib import Path
from typing import Any, ClassVar

from metacontext.ai.handlers.llms.prompt_constraints import (
    COMMON_FIELD_CONSTRAINTS,
    build_schema_constraints,
    calculate_response_limits,
)
from metacontext.handlers.base import BaseFileHandler, register_handler
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
    """Handler for geospatial data files.

    Supports: GeoJSON, Shapefile, GeoTIFF, NetCDF, KML files
    Extensions: raster_context, vector_context
    """

    supported_extensions: ClassVar[list[str]] = [
        ".geojson", ".json",  # Vector data
        ".shp", ".kml", ".kmz", ".gpkg",  # Vector data
        ".tif", ".tiff", ".nc", ".hdf",  # Raster data
    ]

    def __init__(self) -> None:
        """Initialize the geospatial handler."""
        pass

    def can_handle(self, file_path: Path, data_object: object | None = None) -> bool:
        """Check if this is a geospatial data file."""
        if file_path.suffix.lower() in self.supported_extensions:
            return True
        
        # Check if JSON file might be GeoJSON
        if file_path.suffix.lower() == ".json":
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check for GeoJSON structure
                    return data.get("type") in ["FeatureCollection", "Feature", "Point", "LineString", "Polygon"]
            except (json.JSONDecodeError, IOError):
                pass
        
        return False

    def get_required_extensions(self, file_path: Path, data_object: object = None) -> list[str]:
        """Return required extensions for geospatial data."""
        if self._is_raster_file(file_path):
            return ["raster_context"]
        else:
            return ["vector_context"]

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
            else:
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
                    probe_result.update({
                        "dimensions": [dataset.width, dataset.height],
                        "band_count": dataset.count,
                        "crs": str(dataset.crs) if dataset.crs else None,
                        "bounds": list(dataset.bounds),
                    })
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
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("type") == "FeatureCollection":
                        probe_result["feature_count"] = len(data.get("features", []))
                        if data.get("features"):
                            probe_result["geometry_type"] = data["features"][0].get("geometry", {}).get("type")
            elif GEOPANDAS_AVAILABLE:
                # Use geopandas for other vector formats
                gdf = gpd.read_file(file_path)
                probe_result.update({
                    "feature_count": len(gdf),
                    "geometry_type": gdf.geometry.type.iloc[0] if len(gdf) > 0 else None,
                    "crs": str(gdf.crs) if gdf.crs else None,
                })
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
            # Determine if raster or vector
            if self._is_raster_file(file_path):
                return self._generate_raster_context(file_path, ai_companion)
            return self._generate_vector_context(file_path, ai_companion)
        except Exception:
            logger.exception("Error generating geospatial context for %s", file_path)
            return {"error": "Failed to generate geospatial context"}

    def _generate_raster_context(self, file_path: Path, ai_companion: object | None) -> dict[str, Any]:
        """Generate context for raster geospatial data."""
        # Deterministic analysis
        deterministic_metadata = self._analyze_raster_deterministic(file_path)
        
        # AI enrichment
        ai_enrichment = None
        if ai_companion and hasattr(ai_companion, "generate_with_schema"):
            ai_enrichment = self._generate_raster_ai_enrichment(file_path, deterministic_metadata, ai_companion)

        return {
            "raster_context": GeospatialRasterContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=ai_enrichment,
            ).model_dump(),
        }

    def _generate_vector_context(self, file_path: Path, ai_companion: object | None) -> dict[str, Any]:
        """Generate context for vector geospatial data."""
        # Deterministic analysis  
        deterministic_metadata = self._analyze_vector_deterministic(file_path)
        
        # AI enrichment
        ai_enrichment = None
        if ai_companion and hasattr(ai_companion, "generate_with_schema"):
            ai_enrichment = self._generate_vector_ai_enrichment(file_path, deterministic_metadata, ai_companion)

        return {
            "vector_context": GeospatialVectorContext(
                deterministic_metadata=deterministic_metadata,
                ai_enrichment=ai_enrichment,
            ).model_dump(),
        }

    def _analyze_raster_deterministic(self, file_path: Path) -> RasterDeterministicMetadata:
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
                    metadata.pixel_size = [abs(dataset.transform[0]), abs(dataset.transform[4])]
                    metadata.data_type = str(dataset.dtypes[0]) if dataset.dtypes else None
                    metadata.nodata_value = dataset.nodata
                    
                    # Try to get compression info
                    if hasattr(dataset, 'compression'):
                        metadata.compression = dataset.compression
                        
            except Exception as e:
                logger.warning("Error analyzing raster file %s: %s", file_path, e)

        return metadata

    def _analyze_vector_deterministic(self, file_path: Path) -> VectorDeterministicMetadata:
        """Analyze vector file deterministically."""
        metadata = VectorDeterministicMetadata()
        
        try:
            if file_path.suffix.lower() in [".geojson", ".json"]:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    metadata.format = "GeoJSON"
                    if data.get("type") == "FeatureCollection":
                        features = data.get("features", [])
                        metadata.feature_count = len(features)
                        if features:
                            metadata.geometry_type = features[0].get("geometry", {}).get("type")
                            # Get property names
                            properties = features[0].get("properties", {})
                            metadata.attribute_fields = list(properties.keys())
                            
            elif GEOPANDAS_AVAILABLE:
                gdf = gpd.read_file(file_path)
                metadata.format = file_path.suffix.upper().lstrip('.')
                metadata.feature_count = len(gdf)
                metadata.crs = str(gdf.crs) if gdf.crs else None
                if len(gdf) > 0:
                    metadata.geometry_type = gdf.geometry.type.iloc[0]
                    metadata.bounds = list(gdf.total_bounds)
                    metadata.property_names = [col for col in gdf.columns if col != 'geometry']
                    
        except Exception as e:
            logger.warning("Error analyzing vector file %s: %s", file_path, e)

        return metadata

    def analyze_deterministic(self, file_path: Path, data_object: object = None) -> dict[str, object]:
        """Analyze file without AI - deterministic analysis only."""
        if self._is_raster_file(file_path):
            metadata = self._analyze_raster_deterministic(file_path)
            return {"raster_metadata": metadata.model_dump()}
        else:
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
        if not ai_companion or not hasattr(ai_companion, "is_available") or not ai_companion.is_available():
            return {"error": "AI companion not available for deep analysis"}

        try:
            if self._is_raster_file(file_path):
                metadata = deterministic_context.get("raster_metadata") if deterministic_context else None
                if not metadata:
                    metadata = self._analyze_raster_deterministic(file_path).model_dump()
                
                ai_enrichment = self._generate_raster_ai_enrichment(
                    file_path, 
                    RasterDeterministicMetadata.model_validate(metadata),
                    ai_companion
                )
                return {"raster_ai_enrichment": ai_enrichment.model_dump() if ai_enrichment else None}
            else:
                metadata = deterministic_context.get("vector_metadata") if deterministic_context else None
                if not metadata:
                    metadata = self._analyze_vector_deterministic(file_path).model_dump()
                
                ai_enrichment = self._generate_vector_ai_enrichment(
                    file_path,
                    VectorDeterministicMetadata.model_validate(metadata), 
                    ai_companion
                )
                return {"vector_ai_enrichment": ai_enrichment.model_dump() if ai_enrichment else None}
        except Exception:
            logger.exception("Error in deep analysis for %s", file_path)
            return {"error": "Deep analysis failed"}

    def _build_raster_constraints(self, file_path: Path, metadata: RasterDeterministicMetadata) -> str:
        """Build constraints for raster AI enrichment."""
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

        base_instruction = "Analyze this raster geospatial dataset"
        constraints = build_schema_constraints(
            max_total_chars=max_total_chars,
            max_field_chars=max_field_chars,
            field_descriptions=field_constraints,
            complexity_context=f"Raster: {metadata.band_count or 'unknown'} bands",
        )

        return f"{base_instruction} and provide insights that fit within these STRICT LIMITS:\\n\\n{constraints}"

    def _build_vector_constraints(self, file_path: Path, metadata: VectorDeterministicMetadata) -> str:
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
    ) -> RasterAIEnrichment | None:
        """Generate AI enrichment for raster data."""
        try:
            context_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "raster_metadata": metadata.model_dump(),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
            }

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

    def _generate_vector_ai_enrichment(
        self,
        file_path: Path,
        metadata: VectorDeterministicMetadata,
        ai_companion: object,
    ) -> VectorAIEnrichment | None:
        """Generate AI enrichment for vector data."""
        try:
            context_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "vector_metadata": metadata.model_dump(),
                "file_size_mb": file_path.stat().st_size / (1024 * 1024),
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

    def get_bulk_prompts(self, file_path: Path, data_object: object = None) -> dict[str, str]:  # noqa: ARG002
        """Get bulk prompts for this file type from config."""
        return self.PROMPT_CONFIG.copy()
