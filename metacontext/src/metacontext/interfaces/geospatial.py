"""Unified geospatial interface for common spatial analysis functionality.

This module defines the unified interface that standardizes geospatial
operations across vector and raster data types, implementing the composition
pattern for spatial analysis.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol


class SpatialMetadata(Protocol):
    """Protocol for spatial metadata objects."""

    crs: str | None
    bounds: list[float] | None


class GeospatialAnalyzer(ABC):
    """Abstract base class for geospatial analysis operations.
    
    This interface unifies common functionality between vector and raster
    spatial analysis, ensuring consistent spatial metadata extraction and
    AI enrichment patterns.
    """

    @abstractmethod
    def extract_spatial_metadata(self, file_path: Path) -> SpatialMetadata:
        """Extract basic spatial metadata from geospatial file.
        
        Args:
            file_path: Path to the geospatial file
            
        Returns:
            SpatialMetadata containing CRS and bounds information
        """

    @abstractmethod
    def analyze_deterministic(self, file_path: Path) -> dict[str, Any]:
        """Perform deterministic spatial analysis.
        
        Args:
            file_path: Path to the geospatial file
            
        Returns:
            Dictionary of deterministic spatial metadata
        """

    @abstractmethod
    def build_ai_constraints(
        self,
        file_path: Path,
        deterministic_metadata: dict[str, Any],
        spatial_clues: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build AI enrichment constraints for spatial analysis.
        
        Args:
            file_path: Path to the geospatial file
            deterministic_metadata: Previously extracted deterministic metadata
            spatial_clues: Optional spatial context clues
            
        Returns:
            Dictionary of AI enrichment constraints
        """

    def extract_spatial_clues(self, file_path: Path) -> dict[str, Any]:
        """Extract spatial context clues from file and environment.
        
        This method provides AI-first spatial intelligence by gathering
        contextual information from filenames, directory structure, and
        codebase patterns.
        
        Args:
            file_path: Path to the geospatial file
            
        Returns:
            Dictionary of spatial context clues for AI enrichment
        """
        clues = {
            "filename_indicators": [],
            "directory_context": [],
            "geographic_hints": [],
            "coordinate_patterns": [],
        }

        # Extract geographic hints from filename
        filename = file_path.stem.lower()
        geographic_terms = [
            "ithaca", "ny", "usa", "world", "global", "state", "country",
            "city", "county", "region", "utm", "wgs84", "nad83", "epsg",
            "lat", "lon", "coord", "geo", "spatial", "map", "gis"
        ]
        
        for term in geographic_terms:
            if term in filename:
                clues["filename_indicators"].append(term)

        # Extract directory context
        parts = file_path.parts
        for part in parts[-3:]:  # Last 3 directory levels
            part_lower = part.lower()
            if any(geo_term in part_lower for geo_term in geographic_terms):
                clues["directory_context"].append(part)

        # Look for coordinate patterns in filename
        import re
        coord_patterns = [
            r'\d+\.\d+[ns]?\d+\.\d+[ew]?',  # lat/lon coordinates
            r'utm_?\d+[ns]?',                # UTM zones
            r'epsg_?\d+',                    # EPSG codes
            r'\d{4}_\d{4}',                  # tile coordinates
        ]
        
        for pattern in coord_patterns:
            matches = re.findall(pattern, filename)
            clues["coordinate_patterns"].extend(matches)

        return clues

    def analyze_crs_info(self, crs_string: str) -> dict[str, str | None]:
        """Analyze coordinate reference system information.
        
        Args:
            crs_string: CRS specification string
            
        Returns:
            Dictionary with parsed CRS information
        """
        if not crs_string:
            return {"type": None, "code": None, "units": None, "description": None}

        crs_info = {
            "type": None,
            "code": None,
            "units": None,
            "description": crs_string,
        }

        crs_lower = crs_string.lower()

        # Extract EPSG codes
        if "epsg" in crs_lower:
            import re
            epsg_match = re.search(r'epsg[:\s]*(\d+)', crs_lower)
            if epsg_match:
                crs_info["code"] = f"EPSG:{epsg_match.group(1)}"
                crs_info["type"] = "projected" if epsg_match.group(1) != "4326" else "geographic"

        # Determine units
        if any(term in crs_lower for term in ["degree", "decimal", "wgs84", "4326"]):
            crs_info["units"] = "degrees"
        elif any(term in crs_lower for term in ["meter", "metre", "utm", "projected"]):
            crs_info["units"] = "meters"

        return crs_info

    def assess_spatial_resolution(self, pixel_size: list[float], units: str | None) -> str:
        """Assess spatial resolution category from pixel size.
        
        Args:
            pixel_size: List of [x_size, y_size] in CRS units
            units: CRS units ("degrees", "meters", etc.)
            
        Returns:
            Resolution category string
        """
        if not pixel_size or len(pixel_size) < 2:
            return "Unknown"

        # Use the larger dimension for assessment
        resolution = max(abs(pixel_size[0]), abs(pixel_size[1]))

        if units == "degrees":
            # Degree-based assessment (rough approximation)
            if resolution <= 0.0001:  # ~10m at equator
                return "Very High"
            elif resolution <= 0.001:  # ~100m at equator
                return "High"
            elif resolution <= 0.01:  # ~1km at equator
                return "Medium"
            else:
                return "Low"
        elif units == "meters":
            # Meter-based assessment
            if resolution <= 3:  # ~1m
                return "Very High"
            elif resolution <= 30:
                return "High"
            elif resolution <= 100:  # ~30m
                return "Medium"
            else:
                return "Low"
        else:
            # Unknown units - use generic assessment
            if resolution <= 30:
                return "High"
            elif resolution <= 100:
                return "Medium"
            else:
                return "Low"


class VectorGeospatialAnalyzer(GeospatialAnalyzer):
    """Specialized analyzer for vector geospatial data."""

    def extract_spatial_metadata(self, file_path: Path) -> SpatialMetadata:
        """Extract basic spatial metadata from vector file (placeholder)."""
        # This is a placeholder - would be implemented with actual vector reading
        return type('SpatialMetadata', (), {'crs': None, 'bounds': None})()

    def analyze_deterministic(self, file_path: Path) -> dict[str, Any]:
        """Perform deterministic spatial analysis (placeholder)."""
        # This is a placeholder - would be implemented with actual analysis
        return {}

    def build_ai_constraints(
        self,
        file_path: Path,
        deterministic_metadata: dict[str, Any],
        spatial_clues: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build AI enrichment constraints (placeholder)."""
        # This is a placeholder - would be implemented with actual constraints
        return {}

    def get_geometry_types(self, data_object: Any) -> list[str]:
        """Extract geometry types from vector data.
        
        Args:
            data_object: Loaded vector data object
            
        Returns:
            List of geometry type strings
        """
        geometry_types = []
        
        if hasattr(data_object, 'geometry'):
            # GeoDataFrame or similar
            if hasattr(data_object.geometry, 'geom_type'):
                unique_types = data_object.geometry.geom_type.unique()
                geometry_types = [str(geom_type) for geom_type in unique_types]
        
        return geometry_types if geometry_types else ["Unknown"]

    def extract_attribute_fields(self, data_object: Any) -> list[str]:
        """Extract attribute field names from vector data.
        
        Args:
            data_object: Loaded vector data object
            
        Returns:
            List of attribute field names
        """
        if hasattr(data_object, 'columns'):
            # DataFrame-like object
            return [col for col in data_object.columns if col != 'geometry']
        
        return []


class RasterGeospatialAnalyzer(GeospatialAnalyzer):
    """Specialized analyzer for raster geospatial data."""

    def extract_spatial_metadata(self, file_path: Path) -> SpatialMetadata:
        """Extract basic spatial metadata from raster file (placeholder)."""
        # This is a placeholder - would be implemented with actual raster reading
        return type("SpatialMetadata", (), {"crs": None, "bounds": None})()

    def analyze_deterministic(self, file_path: Path) -> dict[str, Any]:
        """Perform deterministic spatial analysis (placeholder)."""
        # This is a placeholder - would be implemented with actual analysis
        return {}

    def build_ai_constraints(
        self,
        file_path: Path,
        deterministic_metadata: dict[str, Any],
        spatial_clues: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build AI enrichment constraints (placeholder)."""
        # This is a placeholder - would be implemented with actual constraints
        return {}

    def get_band_count(self, data_object: object | None = None) -> int:
        """Extract band count from raster data.
        
        Args:
            data_object: Optional loaded raster data object
            
        Returns:
            Number of bands
        """
        # This would be implemented with actual raster reading
        # For now, return a default
        return 1

    def extract_pixel_dimensions(self, file_path: Path) -> tuple[int, int] | None:
        """Extract pixel dimensions from raster file.
        
        Args:
            file_path: Path to raster file
            
        Returns:
            Tuple of (width, height) or None if cannot determine
        """
        try:
            # Try to use PIL for basic image files
            from PIL import Image
            with Image.open(file_path) as img:
                return img.size  # (width, height)
        except Exception:
            return None


class GeospatialInterfaceFactory:
    """Factory for creating appropriate geospatial analyzers."""

    @staticmethod
    def create_analyzer(file_path: Path, data_type: str) -> GeospatialAnalyzer:
        """Create appropriate geospatial analyzer based on data type.
        
        Args:
            file_path: Path to geospatial file
            data_type: Type of geospatial data ("vector" or "raster")
            
        Returns:
            Appropriate GeospatialAnalyzer instance
        """
        if data_type == "vector":
            return VectorGeospatialAnalyzer()
        elif data_type == "raster":
            return RasterGeospatialAnalyzer()
        else:
            raise ValueError(f"Unknown geospatial data type: {data_type}")

    @staticmethod
    def determine_data_type(file_path: Path) -> str:
        """Determine if file is vector or raster geospatial data.
        
        Args:
            file_path: Path to geospatial file
            
        Returns:
            "vector" or "raster" data type
        """
        raster_extensions = {".tif", ".tiff", ".nc", ".hdf"}
        vector_extensions = {".geojson", ".json", ".shp", ".kml", ".kmz", ".gpkg"}
        
        ext = file_path.suffix.lower()
        
        if ext in raster_extensions:
            return "raster"
        elif ext in vector_extensions:
            return "vector"
        else:
            # Default fallback - could be enhanced with file content analysis
            return "vector"