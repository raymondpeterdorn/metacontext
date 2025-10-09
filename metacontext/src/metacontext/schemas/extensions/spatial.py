"""Lightweight spatial extension schema for core metacontext.

This schema provides basic spatial metadata that can be added to any base handler
without requiring heavy geospatial dependencies.
"""

from pydantic import BaseModel, Field
from typing import Any


class SpatialExtension(BaseModel):
    """Lightweight spatial metadata extension.
    
    This extension can be applied to tabular, media, or other base contexts
    to add spatial awareness without handler conflicts.
    """
    
    # CRS Information
    crs: str | None = Field(
        default=None,
        description="Coordinate Reference System (e.g., 'EPSG:4326', 'EPSG:3857')"
    )
    crs_detected_method: str = Field(
        default="none",
        description="Method used to detect CRS: 'geodataframe_attribute', 'geojson_crs_field', 'geojson_default', 'ai_enrichment_needed', 'extraction_failed'"
    )
    
    # Geometry Information
    geometry_type: str | None = Field(
        default=None,
        description="Primary geometry type: 'Point', 'LineString', 'Polygon', 'MultiPoint', etc."
    )
    feature_count: int | None = Field(
        default=None,
        description="Number of spatial features/records"
    )
    
    # Spatial Extent
    bounds_estimate: dict[str, float] | None = Field(
        default=None,
        description="Approximate spatial bounds: {'min_lon', 'max_lon', 'min_lat', 'max_lat'}"
    )
    
    # AI Enrichment Indicator
    requires_ai_enrichment: bool = Field(
        default=False,
        description="Whether spatial metadata needs AI analysis due to detection failures"
    )
    
    # Optional: Additional spatial context
    spatial_notes: str | None = Field(
        default=None,
        description="Additional spatial context or detection notes"
    )
    
    # Raster-specific fields
    pixel_size: list[float] | None = Field(
        default=None,
        description="Pixel size in coordinate units [x_size, y_size] for raster data"
    )
    pixel_dimensions: list[int] | None = Field(
        default=None,
        description="Raster dimensions in pixels [width, height]"
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True


class SpatialEnrichmentPrompts(BaseModel):
    """AI prompts for spatial metadata enrichment."""
    
    crs_detection_prompt: str = Field(
        default="""Analyze this geospatial data and identify the Coordinate Reference System (CRS):
1. Look for coordinate value patterns (large numbers suggest projected coordinates)
2. Decimal degrees typically range -180 to 180 (lon), -90 to 90 (lat)
3. Consider geographic region from coordinate ranges
4. Provide your best CRS estimate and confidence level."""
    )
    
    spatial_analysis_prompt: str = Field(
        default="""Examine the spatial characteristics:
1. Identify geometry type and spatial distribution patterns
2. Estimate geographic extent and coverage area
3. Analyze coordinate precision and data quality
4. Provide comprehensive spatial context."""
    )

    class Config:
        """Pydantic configuration."""
        extra = "forbid"
        validate_assignment = True