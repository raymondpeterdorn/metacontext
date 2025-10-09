"""Geospatial extension for adding spatial metadata to base handlers.

This extension provides lightweight geospatial metadata extraction that can be
applied to tabular or media handlers without requiring heavy dependencies.
"""

import json
import logging
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


class SpatialMetadata(Protocol):
    """Protocol for spatial metadata extraction results."""
    
    crs: str | None
    crs_detected_method: str
    geometry_type: str | None
    feature_count: int | None
    bounds_estimate: dict[str, float] | None
    requires_ai_enrichment: bool


class GeospatialExtension:
    """Lightweight geospatial metadata extraction extension.
    
    This extension can be applied to base handlers (tabular, media) to add
    spatial metadata without requiring heavy geospatial dependencies.
    """
    
    # Define the result key for spatial data
    result_key = "spatial_extension"
    
    # Strategy execution order constants
    # Lower numbers = higher priority (executed first)
    STRATEGY_RASTER_WORLDFILE = 1      # Lightweight worldfile reading for rasters
    STRATEGY_GEOJSON_PARSING = 2       # JSON parsing for GeoJSON files  
    STRATEGY_GEODATAFRAME_OBJECT = 3   # Extract from GeoDataFrame objects
    STRATEGY_SPATIAL_FILE_DIRECT = 4   # Direct spatial file reading (GPKG/SHP)
    STRATEGY_COORDINATE_COLUMNS = 5    # Coordinate column detection
    STRATEGY_AI_ENRICHMENT = 6         # Fallback to AI analysis
    
    # Geographic coordinate bounds constants
    MIN_LATITUDE = -90
    MAX_LATITUDE = 90
    MIN_LONGITUDE = -180
    MAX_LONGITUDE = 180
    
    # Standard error state constants for consistent error reporting
    ERROR_EXTRACTION_FAILED = "extraction_failed"
    ERROR_DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
    ERROR_PARSE_FAILED = "parse_failed"
    ERROR_CRS_UNKNOWN = "crs_unknown"
    ERROR_COORDINATE_EXTRACTION_FAILED = "coordinate_extraction_failed"
    ERROR_COULD_NOT_DETERMINE_CRS = "could_not_determine_crs"
    
    # Standard success state constants for CRS detection methods
    METHOD_NONE = "none"
    METHOD_AI_ENRICHMENT_RECOMMENDED = "ai_enrichment_recommended"
    METHOD_COORDINATE_COLUMNS_DETECTED = "coordinate_columns_detected"
    METHOD_DECIMAL_DEGREES_ASSUMED_WGS84 = "decimal_degrees_assumed_wgs84"
    METHOD_GEODATAFRAME_CRS_ATTRIBUTE = "geodataframe_crs_attribute"
    METHOD_WORLDFILE_ASSUMED_WGS84 = "worldfile_assumed_wgs84"
    METHOD_DIRECT_SPATIAL_FILE_READING = "direct_spatial_file_reading"
    METHOD_GEOPANDAS_NOT_AVAILABLE_AI_FALLBACK = "geopandas_not_available_ai_fallback"
    METHOD_GEOJSON_CRS_FIELD = "geojson_crs_field"
    METHOD_GEOJSON_DEFAULT = "geojson_default"
    
    def __init__(self) -> None:
        """Initialize the geospatial extension."""
        self.supported_vector_extensions = {".geojson", ".json", ".kml", ".gpkg", ".shp"}
        self.supported_raster_extensions = {".tiff", ".tif", ".nc", ".hdf"}
        
        # Cache dependency availability
        self._geopandas_available: bool | None = None
        self._pil_available: bool | None = None
        self._rasterio_available: bool | None = None
        self._fiona_available: bool | None = None
        self._magic_available: bool | None = None
    
    def _check_dependencies(self) -> dict[str, bool]:
        """Check availability of optional dependencies and cache results.
        
        Returns:
            Dict mapping dependency names to availability status
        """
        # Check geopandas availability (cached)
        if self._geopandas_available is None:
            try:
                import geopandas  # noqa: F401, PLC0415
                self._geopandas_available = True
                logger.debug("GeoPandas available for spatial file processing")
            except ImportError:
                self._geopandas_available = False
                logger.debug("GeoPandas not available - spatial file processing limited")
        
        # Check PIL availability (cached)
        if self._pil_available is None:
            try:
                from PIL import Image  # noqa: F401, PLC0415
                self._pil_available = True
                logger.debug("PIL available for image dimension reading")
            except ImportError:
                self._pil_available = False
                logger.debug("PIL not available - image dimension reading disabled")
        
        # Check rasterio availability (cached)
        if self._rasterio_available is None:
            try:
                import rasterio  # noqa: F401, PLC0415
                self._rasterio_available = True
                logger.debug("Rasterio available for advanced raster processing")
            except ImportError:
                self._rasterio_available = False
                logger.debug("Rasterio not available - advanced raster processing disabled")
        
        # Check fiona availability (cached)
        if self._fiona_available is None:
            try:
                import fiona  # noqa: F401, PLC0415
                self._fiona_available = True
                logger.debug("Fiona available for vector file I/O")
            except ImportError:
                self._fiona_available = False
                logger.debug("Fiona not available - vector file I/O limited")
        
        # Check magic availability (cached)
        if self._magic_available is None:
            try:
                import magic  # noqa: F401, PLC0415
                self._magic_available = True
                logger.debug("Python-magic available for file type detection")
            except ImportError:
                self._magic_available = False
                logger.debug("Python-magic not available - file type detection limited")
        
        return {
            "geopandas": self._geopandas_available,
            "pil": self._pil_available,
            "rasterio": self._rasterio_available,
            "fiona": self._fiona_available,
            "magic": self._magic_available,
        }
    
    def _get_dependency_error_message(self, dependency: str) -> str:
        """Get user-friendly error message for missing dependency.
        
        Args:
            dependency: Name of the missing dependency
            
        Returns:
            Descriptive error message for the missing dependency
        """
        messages = {
            "geopandas": "GeoPandas is required for reading spatial files (GPKG, SHP, etc.). Install with: pip install geopandas",
            "pil": "Pillow (PIL) is required for reading image dimensions. Install with: pip install Pillow",
            "rasterio": "Rasterio is required for advanced raster data processing. Install with: pip install rasterio",
            "fiona": "Fiona is required for vector file I/O operations. Install with: pip install fiona",
            "magic": "Python-magic is required for robust file type detection. Install with: pip install python-magic",
        }
        return messages.get(dependency, f"Optional dependency '{dependency}' is not available")
    
    
    def can_extend(self, file_path: Path, data_object: object | None = None) -> bool:
        """Check if this extension can add spatial metadata to the given file/object."""
        # Check file extension
        if file_path.suffix.lower() in (self.supported_vector_extensions | self.supported_raster_extensions):
            return True
            
        # Check for GeoJSON content in .json files
        if file_path.suffix.lower() == ".json":
            return self._is_geojson_file(file_path)
            
        return False
    
    def extract_spatial_metadata(self, file_path: Path, data_object: object | None = None) -> dict[str, Any]:
        """Extract basic spatial metadata using lightweight methods only.
        
        This method prioritizes lightweight detection and defers to AI enrichment
        for complex geospatial analysis to keep metacontext dependency-free.
        """
        spatial_info: dict[str, Any] = {
            "crs": None,
            "crs_detected_method": self.METHOD_NONE,
            "geometry_type": None,
            "feature_count": None,
            "bounds_estimate": None,
            "requires_ai_enrichment": False,
        }
        try:
            # Strategy selection based on file type first (deterministic approach)
            file_ext = file_path.suffix.lower()

            # Strategy 1: Read worldfiles for raster images (highest priority)
            if file_ext in self.supported_raster_extensions:
                logger.debug("Using raster worldfile strategy for %s", file_ext)
                worldfile_info = self._read_worldfile(file_path)
                spatial_info.update(worldfile_info)

            # Strategy 2: Parse GeoJSON files (lightweight JSON parsing)
            elif file_ext in {".geojson", ".json"}:
                logger.debug("Using GeoJSON strategy for %s", file_ext)
                geojson_info = self._extract_from_geojson(file_path)
                spatial_info.update(geojson_info)

            # Strategy 3: Direct file reading for spatial formats (GPKG/SHP priority)
            elif file_ext in {".gpkg", ".shp"}:
                logger.debug("Using direct spatial file reading for %s", file_ext)
                direct_info = self._extract_from_spatial_file(file_path)
                spatial_info.update(direct_info)

                # Fallback: If direct reading fails and we have GeoDataFrame, try it
                if (
                    direct_info.get("crs_detected_method")
                    in {self.ERROR_EXTRACTION_FAILED, self.METHOD_GEOPANDAS_NOT_AVAILABLE_AI_FALLBACK}
                    and self._is_valid_geodataframe_like(data_object)
                ):
                    logger.debug("Direct reading failed, trying GeoDataFrame extraction")
                    geodf_info = self._extract_from_geodataframe_like(data_object)
                    # Only use GeoDataFrame results if they're better than file reading
                    if geodf_info.get("crs") and not direct_info.get("crs"):
                        spatial_info.update(geodf_info)

            # Strategy 4: GeoDataFrame object analysis (when no specific file strategy applies)
            elif self._is_valid_geodataframe_like(data_object):
                logger.debug("Using GeoDataFrame-like spatial extraction")
                geodf_info = self._extract_from_geodataframe_like(data_object)
                spatial_info.update(geodf_info)

            # Strategy 5: Basic coordinate detection from any data object with columns
            elif data_object and hasattr(data_object, "columns") and self._has_coordinate_columns(data_object):
                logger.debug("Using coordinate column extraction")
                coord_info = self._extract_basic_coordinates(data_object)
                spatial_info.update(coord_info)

            # Strategy 6: All other cases - use AI enrichment (keeps it lightweight)
            else:
                spatial_info["requires_ai_enrichment"] = True
                spatial_info["crs_detected_method"] = self.METHOD_AI_ENRICHMENT_RECOMMENDED

        except (OSError, ValueError, TypeError, AttributeError) as e:
            # Log the issue and request AI enrichment as a fallback
            logger.warning("Error extracting spatial metadata: %s", str(e), extra={"file": file_path.name})
            spatial_info["requires_ai_enrichment"] = True
            spatial_info["crs_detected_method"] = self.ERROR_EXTRACTION_FAILED

        return spatial_info
    
    def _is_valid_geodataframe_like(self, data_object: object) -> bool:
        """Check if data object is a valid GeoDataFrame-like object with spatial data.

        Returns True only when the object has rows, columns and a non-null CRS attribute.
        """
        if data_object is None:
            return False

        is_valid = False
        try:
            if not (hasattr(data_object, "shape") and hasattr(data_object, "columns") and hasattr(data_object, "crs")):
                return False

            # Ensure there is at least one row
            try:
                nrows = data_object.shape[0]
            except (TypeError, IndexError):
                return False

            if nrows <= 0:
                return False

            # Verify CRS can be read and is not None
            try:
                crs_value = data_object.crs
            except AttributeError:
                crs_value = None

            if crs_value is not None:
                is_valid = True

        except (AttributeError, TypeError):
            is_valid = False

        return is_valid
    
    def _has_coordinate_columns(self, data_object: object) -> bool:
        """Check if data object has coordinate-like columns (lightweight detection)."""
        try:
            if not hasattr(data_object, "columns"):
                logger.debug("No columns attribute on data_object type: %s", type(data_object))
                return False

            columns = [str(col).lower() for col in data_object.columns]
            logger.debug("Found columns: %s", columns)
            has_lat = any(col in ["lat", "latitude", "lat_deg", "y", "y_coord"] for col in columns)
            has_lon = any(col in ["lon", "long", "longitude", "lon_deg", "x", "x_coord"] for col in columns)
            has_x = any(col in ["x", "x_coord", "easting"] for col in columns)
            has_y = any(col in ["y", "y_coord", "northing"] for col in columns)

            result = (has_lat and has_lon) or (has_x and has_y)
            logger.debug("Coordinate detection: lat=%s, lon=%s, x=%s, y=%s, result=%s", has_lat, has_lon, has_x, has_y, result)
            return result
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Exception in _has_coordinate_columns: %s", str(e))
            return False
    
    def _extract_basic_coordinates(self, data_object: object) -> dict[str, Any]:
        """Extract basic coordinate information without heavy dependencies."""
        info: dict[str, Any] = {}
        
        try:
            if not hasattr(data_object, "columns") or not hasattr(data_object, "shape"):
                return info
            
            # Get basic info
            info["feature_count"] = data_object.shape[0]
            info["geometry_type"] = "Point"  # Assume points for coordinate data
            info["crs_detected_method"] = self.METHOD_COORDINATE_COLUMNS_DETECTED
            
            # Find coordinate columns (more precise matching)
            columns = [str(col).lower() for col in data_object.columns]
            lat_cols = [col for col in data_object.columns if str(col).lower() in ["lat", "latitude", "lat_deg", "y", "y_coord"]]
            lon_cols = [col for col in data_object.columns if str(col).lower() in ["lon", "long", "longitude", "lon_deg", "x", "x_coord"]]
            
            # Extract bounds if we have lat/lon columns and non-empty data
            if lat_cols and lon_cols and data_object.shape[0] > 0:
                try:
                    lat_col, lon_col = lat_cols[0], lon_cols[0]
                    lat_values = data_object[lat_col].dropna()
                    lon_values = data_object[lon_col].dropna()
                    
                    if len(lat_values) > 0 and len(lon_values) > 0:
                        bounds = {
                            "min_lon": float(lon_values.min()),
                            "max_lon": float(lon_values.max()),
                            "min_lat": float(lat_values.min()),
                            "max_lat": float(lat_values.max()),
                        }
                        info["bounds_estimate"] = bounds
                        
                        # Decimal degrees within valid lat/lon range detected, but CRS cannot be determined
                        # Many coordinate systems use decimal degrees (WGS84, NAD83, etc.)
                        if (self.MIN_LATITUDE <= bounds["min_lat"] <= bounds["max_lat"] <= self.MAX_LATITUDE and
                            self.MIN_LONGITUDE <= bounds["min_lon"] <= bounds["max_lon"] <= self.MAX_LONGITUDE):
                            info["crs"] = None  # Cannot determine which decimal degree system
                            info["crs_detected_method"] = self.ERROR_COULD_NOT_DETERMINE_CRS
                            info["requires_ai_enrichment"] = True  # AI may infer from context
                except (AttributeError, TypeError, ValueError) as e:
                    # If coordinate extraction fails, defer to AI
                    logger.debug("Exception in coordinate bounds extraction: %s", str(e))
                    info["requires_ai_enrichment"] = True
                    info["crs_detected_method"] = self.ERROR_COORDINATE_EXTRACTION_FAILED
                        
        except (AttributeError, TypeError, ValueError) as e:
            logger.debug("Exception in _extract_basic_coordinates: %s", str(e))
            info["requires_ai_enrichment"] = True
            
        return info

    def _extract_from_geodataframe_like(self, data_object: object) -> dict[str, Any]:
        """Extract spatial metadata from GeoDataFrame-like objects without importing geopandas."""
        info: dict[str, Any] = {}
        
        try:
            # Extract CRS information (available without importing geopandas)
            if hasattr(data_object, "crs") and data_object.crs:
                info["crs"] = str(data_object.crs)
                info["crs_detected_method"] = self.METHOD_GEODATAFRAME_CRS_ATTRIBUTE
            
            # Get basic feature count
            if hasattr(data_object, "shape"):
                info["feature_count"] = data_object.shape[0]
            
            # Determine geometry type from geometry column
            if hasattr(data_object, "geometry") and hasattr(data_object, "shape") and data_object.shape[0] > 0:
                try:
                    # Get first non-null geometry to determine type
                    first_geom = None
                    for geom in data_object.geometry:
                        if geom is not None:
                            first_geom = geom
                            break
                    
                    if first_geom and hasattr(first_geom, "geom_type"):
                        info["geometry_type"] = first_geom.geom_type
                except (AttributeError, IndexError):
                    info["geometry_type"] = "Unknown"
            
            # Calculate bounds using the bounds property if available
            if hasattr(data_object, "bounds") and data_object.shape[0] > 0:
                try:
                    bounds_series = data_object.bounds
                    if hasattr(bounds_series, "min") and hasattr(bounds_series, "max"):
                        # GeoDataFrame.bounds returns a DataFrame with minx, miny, maxx, maxy
                        info["bounds_estimate"] = {
                            "min_lon": float(bounds_series["minx"].min()),
                            "max_lon": float(bounds_series["maxx"].max()), 
                            "min_lat": float(bounds_series["miny"].min()),
                            "max_lat": float(bounds_series["maxy"].max()),
                        }
                        info["requires_ai_enrichment"] = False
                except (KeyError, AttributeError, ValueError):
                    # Fall back to coordinate column approach
                    coord_info = self._extract_basic_coordinates(data_object)
                    info.update(coord_info)
                    
        except (AttributeError, TypeError, ValueError, ImportError) as e:
            logger.debug("Error extracting from GeoDataFrame-like object: %s", str(e), extra={"error": str(e)})
            info["requires_ai_enrichment"] = True
            info["crs_detected_method"] = self.ERROR_EXTRACTION_FAILED
            
        return info

    def _read_worldfile(self, image_path: Path) -> dict[str, Any]:
        """Read worldfile (.tfw, .tiff.world, etc.) for raster georeferencing."""
        info: dict[str, Any] = {}
        
        # Constants for worldfile validation
        worldfile_params_count = 6
        COORDINATE_COMPONENTS = 2
        
        # Common worldfile extensions for different image formats
        worldfile_extensions = [
            ".tfw",      # TIFF worldfile
            ".tifw",     # Alternative TIFF worldfile
            ".wld",      # Generic worldfile
            ".tiff.world", # Long form TIFF worldfile
            ".tif.world",  # Long form TIF worldfile
        ]
        
        worldfile_path = None
        for ext in worldfile_extensions:
            candidate = image_path.with_suffix(ext)
            if candidate.exists():
                worldfile_path = candidate
                break
                
        if not worldfile_path:
            return info
            
        try:
            with worldfile_path.open("r") as f:
                lines = [line.strip() for line in f.readlines()]
                
            if len(lines) >= worldfile_params_count:
                # Worldfile format: pixel_size_x, rotation1, rotation2, pixel_size_y, x_upper_left, y_upper_left
                pixel_size_x = float(lines[0])
                # rotation params not used but read for completeness
                float(lines[1])  # rotation1
                float(lines[2])  # rotation2
                pixel_size_y = float(lines[3])  # Usually negative for images
                x_upper_left = float(lines[4])
                y_upper_left = float(lines[5])
                
                # Try to get image dimensions for bounds calculation
                dependencies = self._check_dependencies()
                if dependencies["pil"]:
                    try:
                        from PIL import Image  # noqa: PLC0415
                        with Image.open(image_path) as img:
                            width, height = img.size
                            
                            # Calculate bounds from worldfile parameters
                            west = x_upper_left
                            north = y_upper_left  
                            east = west + (width * pixel_size_x)
                            south = north + (height * pixel_size_y)
                            
                            info["bounds_estimate"] = {
                                "min_lon": min(west, east),
                                "max_lon": max(west, east), 
                                "min_lat": min(south, north),
                                "max_lat": max(south, north),
                            }
                            
                            info["pixel_size"] = [abs(pixel_size_x), abs(pixel_size_y)]
                            info["pixel_dimensions"] = [width, height]
                            
                    except (OSError, ValueError) as e:
                        logger.debug("Error reading image dimensions", extra={"error": str(e)})
                else:
                    logger.debug("PIL not available for image dimension reading",
                               extra={"dependency_message": self._get_dependency_error_message("pil")})
                
                # Worldfiles don't contain CRS information - cannot determine without additional files
                info["crs"] = None
                info["crs_detected_method"] = self.ERROR_COULD_NOT_DETERMINE_CRS
                info["geometry_type"] = "Raster"
                info["requires_ai_enrichment"] = True  # AI may be able to infer from context
                
                logger.info("Successfully read worldfile (CRS unknown)", extra={"worldfile_name": worldfile_path.name})
                
        except (OSError, ValueError, IndexError) as e:
            logger.debug("Error reading worldfile", extra={"path": str(worldfile_path), "error": str(e)})
            
        return info

    def _extract_from_spatial_file(self, file_path: Path) -> dict[str, Any]:
        """Extract spatial metadata directly from spatial file when base handler fails."""
        info: dict[str, Any] = {}
        
        # Check dependency availability first
        dependencies = self._check_dependencies()
        
        if not dependencies["geopandas"]:
            # GeoPandas not available, fall back to AI enrichment
            info["requires_ai_enrichment"] = True
            info["crs_detected_method"] = self.METHOD_GEOPANDAS_NOT_AVAILABLE_AI_FALLBACK
            logger.info("GeoPandas not available for spatial file reading", 
                       extra={"file": file_path.name, "dependency_message": self._get_dependency_error_message("geopandas")})
            return info
        
        try:
            # Import geopandas (already verified as available)
            import geopandas as gpd  # noqa: PLC0415
            
            logger.debug("Attempting direct spatial file reading", extra={"file": file_path.name})
            
            # Read the spatial file directly
            gdf = gpd.read_file(file_path)
            
            logger.debug("Successfully read spatial file",
                       extra={"file": file_path.name, "shape": gdf.shape, "crs": str(gdf.crs) if gdf.crs else None})
            
            # Extract basic spatial metadata
            if hasattr(gdf, "crs") and gdf.crs:
                info["crs"] = str(gdf.crs)
                info["crs_detected_method"] = self.METHOD_DIRECT_SPATIAL_FILE_READING
            
            if hasattr(gdf, "shape"):
                info["feature_count"] = gdf.shape[0]
            
            # Get geometry type from first non-null geometry
            if hasattr(gdf, "geometry") and len(gdf) > 0:
                for geom in gdf.geometry:
                    if geom is not None and hasattr(geom, "geom_type"):
                        info["geometry_type"] = geom.geom_type
                        break
            
            # Calculate bounds
            if hasattr(gdf, "total_bounds") and len(gdf) > 0:
                bounds = gdf.total_bounds
                info["bounds_estimate"] = {
                    "min_lon": float(bounds[0]),
                    "min_lat": float(bounds[1]),
                    "max_lon": float(bounds[2]),
                    "max_lat": float(bounds[3]),
                }
            
            info["requires_ai_enrichment"] = False
            logger.debug("Successfully extracted spatial metadata via direct reading",
                       extra={"file": file_path.name, "crs": info.get("crs"), "feature_count": info.get("feature_count")})
                
        except Exception as e:
            # Catch all file reading errors (including pyogrio.errors.DataSourceError)
            info["requires_ai_enrichment"] = True
            info["crs_detected_method"] = self.ERROR_EXTRACTION_FAILED
            logger.debug("Direct spatial file reading failed: %s", str(e),
                       extra={"file": file_path.name})
        
        return info

    def _is_geojson_file(self, file_path: Path) -> bool:
        """Check if JSON file is actually GeoJSON."""
        try:
            with file_path.open(encoding="utf-8") as f:
                data = json.load(f)
                return data.get("type") in {
                    "FeatureCollection", "Feature", "Point",
                    "LineString", "Polygon", "MultiPoint",
                    "MultiLineString", "MultiPolygon", "GeometryCollection",
                }
        except (OSError, json.JSONDecodeError, KeyError):
            return False
    
    def _extract_from_geojson(self, file_path: Path) -> dict[str, Any]:
        """Extract spatial metadata from GeoJSON file using pure JSON parsing."""
        info: dict[str, Any] = {}
        
        try:
            with file_path.open(encoding="utf-8") as f:
                geojson_data = json.load(f)
            
            # Extract CRS information
            if "crs" in geojson_data:
                info["crs"] = str(geojson_data["crs"])
                info["crs_detected_method"] = self.METHOD_GEOJSON_CRS_FIELD
            else:
                # GeoJSON spec default
                info["crs"] = "EPSG:4326"
                info["crs_detected_method"] = self.METHOD_GEOJSON_DEFAULT
            
            # Extract feature information
            if geojson_data.get("type") == "FeatureCollection":
                features = geojson_data.get("features", [])
                info["feature_count"] = len(features)
                
                # Get geometry type from first feature
                if features and "geometry" in features[0]:
                    info["geometry_type"] = features[0]["geometry"].get("type")
                
                # Calculate approximate bounds from sample coordinates
                info["bounds_estimate"] = self._calculate_geojson_bounds(features)
                
            elif geojson_data.get("type") == "Feature":
                info["feature_count"] = 1
                if "geometry" in geojson_data:
                    info["geometry_type"] = geojson_data["geometry"].get("type")
                    
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as e:
            logger.debug("Error parsing GeoJSON %s: %s", file_path.name, str(e))
            info["requires_ai_enrichment"] = True
            info["crs_detected_method"] = self.ERROR_PARSE_FAILED
            
        return info
    
    def _calculate_geojson_bounds(self, features: list[dict[str, Any]]) -> dict[str, float]:
        """Calculate approximate bounds from GeoJSON features.
        
        Returns empty dict on failure for consistent return type handling.
        """
        try:
            coords = []
            # Sample first 20 features to avoid processing huge files
            for feature in features[:20]:
                if "geometry" in feature and "coordinates" in feature["geometry"]:
                    geom_coords = feature["geometry"]["coordinates"]
                    # Handle different geometry types
                    if feature["geometry"].get("type") == "Point":
                        coords.append(geom_coords)
                    elif feature["geometry"].get("type") in {"LineString", "MultiPoint"}:
                        coords.extend(geom_coords)
                    # For polygons, take first ring
                    elif feature["geometry"].get("type") in {"Polygon"} and geom_coords and geom_coords[0]:
                        coords.extend(geom_coords[0])
            
            if coords:
                lons = [c[0] for c in coords if len(c) >= 2]
                lats = [c[1] for c in coords if len(c) >= 2]
                
                if lons and lats:
                    return {
                        "min_lon": min(lons),
                        "max_lon": max(lons),
                        "min_lat": min(lats),
                        "max_lat": max(lats),
                    }
        except (TypeError, IndexError, KeyError, ValueError) as e:
            logger.debug("Error calculating GeoJSON bounds: %s", str(e))
            
        return {}  # Return empty dict instead of None for consistency
    
    def get_ai_enrichment_prompts(self) -> dict[str, str]:
        """Get AI prompts for spatial metadata extraction when deterministic methods fail."""
        return {
            "crs_detection": """
Analyze this geospatial data and identify:
1. Coordinate Reference System (CRS/EPSG code) based on coordinate value patterns
2. Geographic region or extent from coordinate ranges
3. Coordinate precision level (decimal places)

Look for patterns like:
- Large numbers (> 1000) suggest projected coordinates (meters/feet)
- Decimal degrees typically range -180 to 180 (longitude), -90 to 90 (latitude)
- Specific coordinate ranges can indicate regional projections

Provide your best estimate of the CRS and confidence level.
""",
            
            "spatial_analysis": """
Examine the spatial distribution and characteristics:
1. Geometry type (Point, LineString, Polygon, etc.)
2. Geographic clustering patterns
3. Spatial extent and coverage area
4. Data density and distribution

Analyze coordinate patterns, attribute names, and any spatial relationships
to provide comprehensive spatial context.
""",
        }