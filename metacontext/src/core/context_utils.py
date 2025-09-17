"""Context normalization utilities for ensuring JSON-serializable output."""

import hashlib
import json
from datetime import datetime, date, time
from decimal import Decimal
from typing import Any, Dict, List, Union
import numpy as np
import pandas as pd


class ContextNormalizer:
    """Utility class for normalizing context data to ensure JSON serialization."""
    
    MAX_TEXT_LENGTH = 2000
    MAX_LIST_ITEMS = 5
    LARGE_BLOB_THRESHOLD = 1024 * 1024  # 1MB
    
    @classmethod
    def normalize_context(cls, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize a context dictionary to ensure JSON-serializable values.
        
        Args:
            context: Raw context dictionary that may contain non-serializable objects
            
        Returns:
            Normalized context dictionary with JSON-serializable values
        """
        if not isinstance(context, dict):
            return cls._normalize_value(context)
        
        normalized = {}
        for key, value in context.items():
            normalized[str(key)] = cls._normalize_value(value)
        
        return normalized
    
    @classmethod
    def _normalize_value(cls, value: Any) -> Any:
        """Normalize a single value to be JSON-serializable."""
        
        # Handle None
        if value is None:
            return None
        
        # Handle basic JSON-serializable types
        if isinstance(value, (str, int, float, bool)):
            if isinstance(value, str) and len(value) > cls.MAX_TEXT_LENGTH:
                return cls._truncate_text(value)
            return value
        
        # Handle datetime objects
        if isinstance(value, (datetime, date, time)):
            return value.isoformat()
        
        # Handle Decimal
        if isinstance(value, Decimal):
            return float(value)
        
        # Handle NumPy types
        if hasattr(value, 'dtype'):  # NumPy scalar
            if np.issubdtype(value.dtype, np.integer):
                return int(value)
            elif np.issubdtype(value.dtype, np.floating):
                if np.isnan(value) or np.isinf(value):
                    return str(value)  # Convert NaN/inf to string
                return float(value)
            elif np.issubdtype(value.dtype, np.bool_):
                return bool(value)
            elif np.issubdtype(value.dtype, np.datetime64):
                return str(value)
            else:
                return str(value)
        
        # Handle NumPy arrays
        if isinstance(value, np.ndarray):
            return cls._normalize_array(value)
        
        # Handle pandas objects
        if hasattr(value, 'to_dict'):  # pandas DataFrame/Series
            return cls._normalize_pandas_object(value)
        
        # Handle pandas Timestamp
        if hasattr(value, 'isoformat') and hasattr(value, 'timestamp'):
            return value.isoformat()
        
        # Handle geospatial objects (Shapely geometries)
        if hasattr(value, '__geo_interface__'):
            try:
                return cls._normalize_geometry(value)
            except Exception:
                return str(value)
        
        # Handle lists and tuples
        if isinstance(value, (list, tuple)):
            return cls._normalize_list(list(value))
        
        # Handle dictionaries
        if isinstance(value, dict):
            return cls.normalize_context(value)
        
        # Handle sets
        if isinstance(value, set):
            return cls._normalize_list(list(value))
        
        # Handle bytes
        if isinstance(value, bytes):
            return cls._normalize_bytes(value)
        
        # Handle other complex objects by converting to string
        try:
            # Try to serialize to see if it's already JSON-serializable
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            # If not serializable, convert to string representation
            return str(value)
    
    @classmethod
    def _truncate_text(cls, text: str) -> str:
        """Truncate long text and add indicator."""
        if len(text) <= cls.MAX_TEXT_LENGTH:
            return text
        
        truncated = text[:cls.MAX_TEXT_LENGTH]
        return f"{truncated}... [TRUNCATED: {len(text)} total chars]"
    
    @classmethod
    def _normalize_list(cls, lst: List[Any]) -> Union[List[Any], Dict[str, Any]]:
        """Normalize a list, truncating if too long."""
        if len(lst) <= cls.MAX_LIST_ITEMS:
            return [cls._normalize_value(item) for item in lst]
        
        # For large lists, return first few items plus metadata
        normalized_items = [cls._normalize_value(item) for item in lst[:cls.MAX_LIST_ITEMS]]
        return {
            "preview_items": normalized_items,
            "total_length": len(lst),
            "truncated": True,
            "preview_count": cls.MAX_LIST_ITEMS
        }
    
    @classmethod
    def _normalize_array(cls, arr: np.ndarray) -> Union[List[Any], Dict[str, Any]]:
        """Normalize NumPy arrays."""
        # For small arrays, convert to list
        if arr.size <= cls.MAX_LIST_ITEMS:
            return arr.tolist()
        
        # For large arrays, provide summary
        return {
            "array_shape": list(arr.shape),
            "array_dtype": str(arr.dtype),
            "array_size": int(arr.size),
            "preview_values": arr.flat[:cls.MAX_LIST_ITEMS].tolist(),
            "truncated": True,
            "sha256_hash": cls._compute_hash(arr.tobytes())
        }
    
    @classmethod
    def _normalize_pandas_object(cls, obj) -> Dict[str, Any]:
        """Normalize pandas DataFrame or Series."""
        if hasattr(obj, 'shape'):  # DataFrame
            if obj.shape[0] <= cls.MAX_LIST_ITEMS:
                try:
                    return obj.to_dict('records')
                except Exception:
                    return str(obj)
            else:
                # Large DataFrame - provide summary
                preview_data = obj.head(cls.MAX_LIST_ITEMS)
                return {
                    "dataframe_shape": list(obj.shape),
                    "column_names": list(obj.columns) if hasattr(obj, 'columns') else None,
                    "dtypes": obj.dtypes.to_dict() if hasattr(obj, 'dtypes') else None,
                    "preview_records": preview_data.to_dict('records'),
                    "truncated": True,
                    "total_rows": len(obj)
                }
        else:  # Series
            if len(obj) <= cls.MAX_LIST_ITEMS:
                return obj.tolist()
            else:
                return {
                    "series_length": len(obj),
                    "series_dtype": str(obj.dtype),
                    "preview_values": obj.head(cls.MAX_LIST_ITEMS).tolist(),
                    "truncated": True
                }
    
    @classmethod
    def _normalize_geometry(cls, geom) -> Dict[str, Any]:
        """Normalize geospatial geometry objects."""
        try:
            # Use the __geo_interface__ which should be JSON-serializable
            geo_dict = geom.__geo_interface__
            return {
                "geometry_type": geo_dict.get("type", "unknown"),
                "coordinates": geo_dict.get("coordinates", []),
                "geometry_interface": geo_dict
            }
        except Exception:
            return {
                "geometry_type": str(type(geom).__name__),
                "geometry_string": str(geom)
            }
    
    @classmethod
    def _normalize_bytes(cls, data: bytes) -> Union[str, Dict[str, Any]]:
        """Normalize bytes objects."""
        if len(data) > cls.LARGE_BLOB_THRESHOLD:
            return {
                "data_type": "large_binary_blob",
                "size_bytes": len(data),
                "sha256_hash": cls._compute_hash(data),
                "preview_hex": data[:16].hex()
            }
        else:
            # For smaller byte objects, convert to hex string
            return data.hex()
    
    @classmethod
    def _compute_hash(cls, data: bytes) -> str:
        """Compute SHA256 hash of binary data."""
        return hashlib.sha256(data).hexdigest()


def normalize_context(context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convenience function to normalize context data.
    
    Args:
        context: Raw context dictionary
        
    Returns:
        Normalized context dictionary
    """
    return ContextNormalizer.normalize_context(context)


def safe_json_serialize(obj: Any) -> str:
    """
    Safely serialize an object to JSON string, normalizing if necessary.
    
    Args:
        obj: Object to serialize
        
    Returns:
        JSON string representation
    """
    try:
        return json.dumps(obj, indent=2)
    except (TypeError, ValueError):
        normalized = normalize_context(obj) if isinstance(obj, dict) else ContextNormalizer._normalize_value(obj)
        return json.dumps(normalized, indent=2)