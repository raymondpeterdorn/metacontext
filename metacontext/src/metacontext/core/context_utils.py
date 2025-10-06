"""Context normalization utilities for ensuring JSON-serializable output."""

import hashlib
import json
from datetime import date, datetime, time
from decimal import Decimal
from typing import Any, Protocol

import numpy as np

JsonSerializable = (
    str
    | int
    | float
    | bool
    | None
    | list["JsonSerializable"]
    | dict[str, "JsonSerializable"]
)


class TypeHandler(Protocol):
    """A protocol for type normalization handlers."""

    def __call__(self, value: Any) -> JsonSerializable: ...


class ContextNormalizer:
    """Utility class for normalizing context data to ensure JSON serialization."""

    MAX_TEXT_LENGTH = 2000
    MAX_LIST_ITEMS = 5
    LARGE_BLOB_THRESHOLD = 1024 * 1024  # 1MB

    @classmethod
    def normalize_context(cls, context: dict[str, Any]) -> dict[str, JsonSerializable]:
        """Normalize a context dictionary to ensure JSON-serializable values.

        Args:
            context: Raw context dictionary that may contain non-serializable objects.

        Returns:
            Normalized context dictionary with JSON-serializable values.

        """
        normalized = {}
        for key, value in context.items():
            normalized[str(key)] = cls.normalize_value(value)

        return normalized

    @classmethod
    def normalize_value(cls, value: Any) -> JsonSerializable:
        """Normalize a single value to be JSON-serializable."""
        if isinstance(value, (str, int, float, bool, type(None))):
            if isinstance(value, str) and len(value) > cls.MAX_TEXT_LENGTH:
                return cls._truncate_text(value)
            return value

        if isinstance(value, (datetime, date, time)):
            return str(value.isoformat())

        if isinstance(value, Decimal):
            return float(value)

        if hasattr(value, "dtype"):  # NumPy scalar
            return cls._normalize_numpy_scalar(value)

        if isinstance(value, np.ndarray):
            return cls._normalize_array(value)

        if hasattr(value, "to_dict"):  # pandas DataFrame/Series
            return cls._normalize_pandas_object(value)

        if hasattr(value, "isoformat") and hasattr(value, "timestamp"):
            return str(value.isoformat())

        if hasattr(value, "__geo_interface__"):
            try:
                return cls._normalize_geometry(value)
            except (AttributeError, TypeError, ValueError):
                return str(value)

        if isinstance(value, (list, tuple, set)):
            return cls._normalize_list(list(value))

        if isinstance(value, dict):
            return cls.normalize_context(value)

        if isinstance(value, bytes):
            return cls._normalize_bytes(value)

        try:
            json.dumps(value)
        except (TypeError, ValueError):
            return str(value)
        else:
            return str(value)

    @classmethod
    def _normalize_numpy_scalar(cls, value: np.generic) -> JsonSerializable:
        """Normalize a NumPy scalar value."""
        if np.issubdtype(value.dtype, np.integer):
            return int(value)
        if np.issubdtype(value.dtype, np.floating):
            if np.isnan(value) or np.isinf(value):
                return str(value)
            return float(value)
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value)
        if np.issubdtype(value.dtype, np.datetime64):
            return str(value)
        return str(value)

    @classmethod
    def _truncate_text(cls, text: str) -> str:
        """Truncate long text and add indicator."""
        if len(text) <= cls.MAX_TEXT_LENGTH:
            return text

        truncated = text[: cls.MAX_TEXT_LENGTH]
        return f"{truncated}... [TRUNCATED: {len(text)} total chars]"

    @classmethod
    def _normalize_list(
        cls, lst: list[Any]
    ) -> list[JsonSerializable] | dict[str, JsonSerializable]:
        """Normalize a list, truncating if too long."""
        if len(lst) <= cls.MAX_LIST_ITEMS:
            return [cls.normalize_value(item) for item in lst]

        # For large lists, return first few items plus metadata
        normalized_items = [
            cls.normalize_value(item) for item in lst[: cls.MAX_LIST_ITEMS]
        ]
        return {
            "preview_items": normalized_items,
            "total_length": len(lst),
            "truncated": True,
            "preview_count": cls.MAX_LIST_ITEMS,
        }

    @classmethod
    def _normalize_array(
        cls, arr: np.ndarray
    ) -> list[JsonSerializable] | dict[str, JsonSerializable]:
        """Normalize NumPy arrays."""
        # For small arrays, convert to list
        if arr.size <= cls.MAX_LIST_ITEMS:
            return [cls.normalize_value(item) for item in arr.tolist()]

        # For large arrays, provide summary
        return {
            "array_shape": list(arr.shape),
            "array_dtype": str(arr.dtype),
            "array_size": int(arr.size),
            "preview_values": [
                cls.normalize_value(item)
                for item in arr.flat[: cls.MAX_LIST_ITEMS].tolist()
            ],
            "truncated": True,
            "sha256_hash": cls._compute_hash(arr.tobytes()),
        }

    @classmethod
    def _normalize_pandas_object(cls, obj: Any) -> dict[str, Any] | list[Any]:
        """Normalize pandas DataFrame or Series."""
        if hasattr(obj, "shape"):  # DataFrame
            if obj.shape[0] <= cls.MAX_LIST_ITEMS:
                try:
                    return list(obj.to_dict("records"))
                except (AttributeError, TypeError, ValueError):
                    return [{"error": "Failed to serialize DataFrame to records"}]
            else:
                # Large DataFrame - provide summary
                preview_data = obj.head(cls.MAX_LIST_ITEMS)
                return {
                    "dataframe_shape": list(obj.shape),
                    "column_names": list(obj.columns)
                    if hasattr(obj, "columns")
                    else None,
                    "dtypes": obj.dtypes.to_dict() if hasattr(obj, "dtypes") else None,
                    "preview_records": list(preview_data.to_dict("records")),
                    "truncated": True,
                    "total_rows": len(obj),
                }
        elif len(obj) <= cls.MAX_LIST_ITEMS:
            return list(obj.tolist())
        else:
            return {
                "series_length": len(obj),
                "series_dtype": str(obj.dtype),
                "preview_values": list(obj.head(cls.MAX_LIST_ITEMS).tolist()),
                "truncated": True,
            }

    @classmethod
    def _normalize_geometry(cls, geom: Any) -> dict[str, Any]:
        """Normalize geospatial geometry objects."""
        try:
            # Use the __geo_interface__ which should be JSON-serializable
            geo_dict = geom.__geo_interface__
            return {
                "geometry_type": geo_dict.get("type", "unknown"),
                "coordinates": geo_dict.get("coordinates", []),
                "geometry_interface": geo_dict,
            }
        except (AttributeError, TypeError, ValueError):
            return {
                "geometry_type": str(type(geom).__name__),
                "geometry_string": str(geom),
            }

    @classmethod
    def _normalize_bytes(cls, data: bytes) -> str | dict[str, Any]:
        """Normalize bytes objects."""
        if len(data) > cls.LARGE_BLOB_THRESHOLD:
            return {
                "data_type": "large_binary_blob",
                "size_bytes": len(data),
                "sha256_hash": cls._compute_hash(data),
                "preview_hex": data[:16].hex(),
            }
        # For smaller byte objects, convert to hex string
        return data.hex()

    @classmethod
    def _compute_hash(cls, data: bytes) -> str:
        """Compute SHA256 hash of binary data."""
        return hashlib.sha256(data).hexdigest()


def normalize_context(context: Any) -> JsonSerializable:
    """Normalize context data.

    Args:
        context: Raw context dictionary.

    Returns:
        Normalized context dictionary.

    """
    if isinstance(context, dict):
        return ContextNormalizer.normalize_context(context)
    return ContextNormalizer.normalize_value(context)


def safe_json_serialize(obj: Any) -> str:
    """Safely serialize an object to JSON string, normalizing if necessary.

    Args:
        obj: Object to serialize.

    Returns:
        JSON string representation.

    """
    normalized_obj = normalize_context(obj)
    return json.dumps(
        normalized_obj,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
    )
