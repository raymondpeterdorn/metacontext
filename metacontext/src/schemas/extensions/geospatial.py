"""Geospatial extension schemas."""

from pydantic import BaseModel, Field

from src.schemas.extensions.base import (
    AIEnrichment,
    DeterministicMetadata,
    ExtensionContext,
)

# ========== RASTER DATA CLASSES ==========


class BandInfo(BaseModel):
    """Detailed information about a single band in raster data."""

    band_index: int
    description: str | None = None
    data_type: str | None = None  # "Float32", "UInt16", etc.
    value_range: list[float] | None = None  # [min, max]
    classification_scheme: dict[str, str] | None = None  # Value -> class mapping
    units: str | None = None  # "meters", "celsius", "reflectance", etc.
    nodata_value: int | float | None = None
    scale_factor: float | None = None
    offset: float | None = None
    spectral_purpose: str | None = None  # "vegetation", "water detection", etc.


class RasterDeterministicMetadata(DeterministicMetadata):
    """Deterministic facts about raster geospatial data (GeoTIFF, etc.)."""

    format: str | None = None  # "GeoTIFF", "NetCDF", etc.
    crs: str | None = None  # Coordinate Reference System
    bounds: list[float] | None = None  # [xmin, ymin, xmax, ymax]
    pixel_dimensions: list[int] | None = None  # [width, height]
    band_count: int | None = None
    pixel_size: list[float] | None = None  # [x_size, y_size] in CRS units
    nodata_value: int | float | None = None
    data_type: str | None = None  # "Float32", "UInt16", etc.
    compression: str | None = None  # "LZW", "DEFLATE", etc.


class RasterAIEnrichment(AIEnrichment):
    """AI-generated insights about raster geospatial data."""

    geographic_region: str | None = Field(
        None,
        description="The geographic area or region covered by the raster data, such as 'North America', 'Mediterranean', 'Arctic', etc.",
    )
    data_source_analysis: str | None = Field(
        None,
        description="Analysis of the data source, including satellite mission, sensor type, or collection methodology.",
    )
    temporal_context: str | None = Field(
        None,
        description="Temporal information about the data, including time period, seasonality, or temporal patterns.",
    )
    spatial_resolution_assessment: str | None = Field(
        None,
        description="Assessment of the spatial resolution, indicating if it's fine or coarse and implications for analysis.",
    )
    band_info: BandInfo | None = Field(
        None,
        description="Detailed information about raster bands, including meaning, purpose, and characteristics.",
    )


# ========== VECTOR DATA CLASSES ==========


class VectorDeterministicMetadata(DeterministicMetadata):
    """Deterministic facts about vector geospatial data (Shapefile, GeoJSON, etc.)."""

    format: str | None = None  # "Shapefile", "GeoJSON", "GeoPackage", etc.
    crs: str | None = None  # Coordinate Reference System
    bounds: list[float] | None = None  # [xmin, ymin, xmax, ymax]
    geometry_type: str | None = None  # "Point", "LineString", "Polygon", etc.
    feature_count: int | None = None
    attribute_fields: list[str] | None = None  # Field names in attribute table
    encoding: str | None = None  # "UTF-8", etc. for Shapefiles


class VectorAIEnrichment(AIEnrichment):
    """AI-generated insights about vector geospatial data."""

    geographic_region: str | None = Field(
        None,
        description="The geographic area or environment represented by the vector data, such as 'Urban areas', 'Natural boundaries', 'Coastlines', etc.",
    )
    data_source_analysis: str | None = Field(
        None,
        description="Analysis of the data source origin, such as administrative records, field surveys, or crowdsourced data collection.",
    )
    geometry_purpose: str | None = Field(
        None,
        description="The purpose or meaning of the geometries represented in the vector data, such as boundaries, routes, or points of interest.",
    )
    attribute_analysis: str | None = Field(
        None,
        description="Analysis of the attribute fields in the vector data, explaining their meaning, relationships, and potential usage.",
    )
    spatial_relationships: str | None = Field(
        None,
        description="Analysis of how features relate spatially to each other, including patterns, clustering, or network properties.",
    )


# ========== EXTENSION CONTEXT CLASSES ==========


class GeospatialRasterContext(ExtensionContext):
    """Extension context for raster geospatial data.

    See: architecture_reference.ArchitecturalComponents.TWO_TIER_METADATA
    """

    deterministic_metadata: RasterDeterministicMetadata | None = None
    ai_enrichment: RasterAIEnrichment | None = None


class GeospatialVectorContext(ExtensionContext):
    """Extension context for vector geospatial data.

    See: architecture_reference.ArchitecturalComponents.TWO_TIER_METADATA
    """

    deterministic_metadata: VectorDeterministicMetadata | None = None
    ai_enrichment: VectorAIEnrichment | None = None
