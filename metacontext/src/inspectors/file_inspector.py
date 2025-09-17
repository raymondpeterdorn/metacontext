"""Universal file inspector for baseline metadata extraction."""

import hashlib
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    import magic
    HAS_PYTHON_MAGIC = True
except ImportError:
    HAS_PYTHON_MAGIC = False

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

try:
    import pyarrow.parquet as pq
    HAS_PYARROW = True
except ImportError:
    HAS_PYARROW = False


class FileInspector:
    """Universal file inspector that provides baseline metadata for any file type."""
    
    # Text file extensions that we can safely read as text
    TEXT_EXTENSIONS = {
        '.txt', '.md', '.py', '.js', '.ts', '.html', '.css', '.json', 
        '.yaml', '.yml', '.xml', '.csv', '.tsv', '.log', '.sql', '.sh',
        '.bat', '.ps1', '.r', '.R', '.scala', '.java', '.cpp', '.c', '.h',
        '.hpp', '.go', '.rs', '.php', '.rb', '.pl', '.swift', '.kt', '.m',
        '.mm', '.vue', '.jsx', '.tsx', '.scss', '.less', '.toml', '.ini',
        '.cfg', '.conf', '.properties', '.gitignore', '.dockerignore'
    }
    
    # Structured data extensions
    STRUCTURED_EXTENSIONS = {
        '.csv', '.tsv', '.json', '.jsonl', '.parquet', '.xlsx', '.xls'
    }
    
    def __init__(self):
        """Initialize the FileInspector."""
        self.mime = magic.Magic(mime=True) if HAS_PYTHON_MAGIC else None
    
    def inspect(self, file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Inspect a file and return comprehensive metadata.
        
        Args:
            file_path: Path to the file to inspect
            
        Returns:
            Dictionary containing file metadata
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            return {
                'error': 'File does not exist',
                'file_path': str(file_path)
            }
        
        # Basic file metadata
        stat = file_path.stat()
        metadata = {
            'file_path': str(file_path),
            'file_name': file_path.name,
            'file_extension': file_path.suffix.lower(),
            'file_size_bytes': stat.st_size,
            'file_size_human': self._format_file_size(stat.st_size),
            'last_modified': stat.st_mtime,
            'is_directory': file_path.is_dir(),
            'is_file': file_path.is_file()
        }
        
        # Don't process directories further
        if file_path.is_dir():
            return metadata
        
        # MIME type detection
        metadata['mime_type'] = self._get_mime_type(file_path)
        
        # Read file content based on type (prioritize structured over text)
        if self._is_structured_file(file_path):
            structured_metadata = self._inspect_structured_file(file_path)
            metadata.update(structured_metadata)
        elif self._is_text_file(file_path):
            text_metadata = self._inspect_text_file(file_path)
            metadata.update(text_metadata)
        else:
            binary_metadata = self._inspect_binary_file(file_path)
            metadata.update(binary_metadata)
        
        return metadata
    
    def _get_mime_type(self, file_path: Path) -> str:
        """Get MIME type using python-magic if available, fallback to mimetypes."""
        if self.mime:
            try:
                return self.mime.from_file(str(file_path))
            except Exception:
                pass
        
        # Fallback to built-in mimetypes
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type or 'application/octet-stream'
    
    def _is_text_file(self, file_path: Path) -> bool:
        """Check if file is likely a text file."""
        return file_path.suffix.lower() in self.TEXT_EXTENSIONS
    
    def _is_structured_file(self, file_path: Path) -> bool:
        """Check if file is a structured data file."""
        return file_path.suffix.lower() in self.STRUCTURED_EXTENSIONS
    
    def _inspect_text_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect a text file and return preview and basic stats."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                lines = []
                total_lines = 0
                total_chars = 0
                
                for i, line in enumerate(f):
                    total_lines += 1
                    total_chars += len(line)
                    
                    # Store first 20 lines for preview
                    if i < 20:
                        lines.append(line.rstrip('\n\r'))
                    
                    # Limit reading for very large files
                    if total_lines > 10000:
                        break
                
                return {
                    'file_type': 'text',
                    'preview_lines': lines,
                    'total_lines': total_lines,
                    'total_characters': total_chars,
                    'encoding': 'utf-8'
                }
        except Exception as e:
            return {
                'file_type': 'text',
                'error': f"Failed to read text file: {str(e)}"
            }
    
    def _inspect_structured_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect structured data files (CSV, JSON, Parquet, etc.)."""
        extension = file_path.suffix.lower()
        
        if extension in ['.csv', '.tsv']:
            return self._inspect_csv_file(file_path)
        elif extension == '.json':
            return self._inspect_json_file(file_path)
        elif extension == '.jsonl':
            return self._inspect_jsonl_file(file_path)
        elif extension == '.parquet':
            return self._inspect_parquet_file(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self._inspect_excel_file(file_path)
        else:
            return {'file_type': 'structured', 'error': f'Unsupported structured format: {extension}'}
    
    def _inspect_csv_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect CSV/TSV files with descriptive statistics."""
        if not HAS_PANDAS:
            return self._inspect_text_file(file_path)  # Fallback to text inspection
        
        try:
            separator = '\t' if file_path.suffix.lower() == '.tsv' else ','
            
            # Read sample to get schema and stats (increase sample size for better stats)
            df_sample = pd.read_csv(file_path, sep=separator, nrows=5000)
            
            # Get full row count (more efficient)
            with open(file_path, 'r', encoding='utf-8') as f:
                total_rows = sum(1 for _ in f) - 1  # Subtract header
            
            # Limit to 20 columns max for performance
            columns_to_analyze = df_sample.columns[:20]
            df_limited = df_sample[columns_to_analyze]
            
            schema = []
            statistics = {}
            
            for col in df_limited.columns:
                col_info = {
                    'name': col,
                    'dtype': str(df_limited[col].dtype),
                    'non_null_count': int(df_limited[col].count()),
                    'null_count': int(df_limited[col].isnull().sum()),
                    'null_percentage': round((df_limited[col].isnull().sum() / len(df_limited)) * 100, 2)
                }
                
                # Add descriptive statistics based on data type
                if df_limited[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    # Numeric column statistics
                    numeric_stats = self._get_numeric_statistics(df_limited[col])
                    col_info.update(numeric_stats)
                    statistics[col] = numeric_stats
                    
                elif df_limited[col].dtype == 'object':
                    # Categorical column statistics  
                    categorical_stats = self._get_categorical_statistics(df_limited[col])
                    col_info.update(categorical_stats)
                    statistics[col] = categorical_stats
                
                schema.append(col_info)
            
            result = {
                'file_type': 'csv',
                'total_rows': total_rows,
                'total_columns': len(df_sample.columns),
                'analyzed_columns': len(columns_to_analyze),
                'column_names': list(df_sample.columns),
                'schema': schema,
                'statistics': statistics,
                'separator': separator,
                'preview_rows': df_limited.head(5).to_dict('records')
            }
            
            # Add warning if we truncated columns
            if len(df_sample.columns) > 20:
                result['column_truncation_warning'] = f"Only first 20 of {len(df_sample.columns)} columns analyzed"
            
            return result
            
        except Exception as e:
            return {
                'file_type': 'csv',
                'error': f"Failed to parse CSV: {str(e)}"
            }
    
    def _inspect_json_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect JSON files."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            metadata = {
                'file_type': 'json',
                'data_type': type(data).__name__,
            }
            
            if isinstance(data, dict):
                metadata['keys'] = list(data.keys())[:20]  # First 20 keys
                metadata['total_keys'] = len(data)
            elif isinstance(data, list):
                metadata['total_items'] = len(data)
                if data and isinstance(data[0], dict):
                    metadata['item_keys'] = list(data[0].keys())[:20]
            
            # Store small preview
            if len(str(data)) < 2000:
                metadata['preview'] = data
            else:
                metadata['preview_truncated'] = str(data)[:2000] + "..."
            
            return metadata
        except Exception as e:
            return {
                'file_type': 'json',
                'error': f"Failed to parse JSON: {str(e)}"
            }
    
    def _inspect_jsonl_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect JSONL (JSON Lines) files."""
        try:
            lines = []
            total_lines = 0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    total_lines += 1
                    if i < 5:  # Preview first 5 lines
                        try:
                            parsed = json.loads(line.strip())
                            lines.append(parsed)
                        except json.JSONDecodeError:
                            lines.append({"parse_error": line.strip()[:100]})
            
            return {
                'file_type': 'jsonl',
                'total_lines': total_lines,
                'preview_lines': lines
            }
        except Exception as e:
            return {
                'file_type': 'jsonl',
                'error': f"Failed to parse JSONL: {str(e)}"
            }
    
    def _inspect_parquet_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect Parquet files with statistics."""
        if not HAS_PYARROW:
            return {
                'file_type': 'parquet',
                'error': 'PyArrow not available for Parquet inspection'
            }
        
        try:
            table = pq.read_table(file_path)
            
            # Convert to pandas for statistics if available
            if HAS_PANDAS:
                df = table.to_pandas()
                
                # Limit to 20 columns for performance
                columns_to_analyze = df.columns[:20]
                df_limited = df[columns_to_analyze]
                
                schema_info = []
                statistics = {}
                
                for i, name in enumerate(columns_to_analyze):
                    col_type = str(table.schema.field(i).type)
                    col_info = {
                        'name': name,
                        'type': col_type,
                        'non_null_count': int(df_limited[name].count()),
                        'null_count': int(df_limited[name].isnull().sum()),
                        'null_percentage': round((df_limited[name].isnull().sum() / len(df_limited)) * 100, 2)
                    }
                    
                    # Add statistics based on data type
                    if df_limited[name].dtype in ['int64', 'float64', 'int32', 'float32']:
                        numeric_stats = self._get_numeric_statistics(df_limited[name])
                        col_info.update(numeric_stats)
                        statistics[name] = numeric_stats
                    elif df_limited[name].dtype == 'object':
                        categorical_stats = self._get_categorical_statistics(df_limited[name])
                        col_info.update(categorical_stats)
                        statistics[name] = categorical_stats
                    
                    schema_info.append(col_info)
                
                result = {
                    'file_type': 'parquet',
                    'total_rows': len(table),
                    'total_columns': len(table.columns),
                    'analyzed_columns': len(columns_to_analyze),
                    'column_names': table.column_names,
                    'schema': schema_info,
                    'statistics': statistics
                }
                
                # Add warning if we truncated columns
                if len(table.columns) > 20:
                    result['column_truncation_warning'] = f"Only first 20 of {len(table.columns)} columns analyzed"
                    
                return result
            else:
                # Fallback without statistics
                schema_info = []
                for i, name in enumerate(table.column_names):
                    col_type = str(table.schema.field(i).type)
                    schema_info.append({
                        'name': name,
                        'type': col_type
                    })
                
                return {
                    'file_type': 'parquet',
                    'total_rows': len(table),
                    'total_columns': len(table.columns),
                    'column_names': table.column_names,
                    'schema': schema_info
                }
        except Exception as e:
            return {
                'file_type': 'parquet',
                'error': f"Failed to parse Parquet: {str(e)}"
            }
    
    def _inspect_excel_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect Excel files."""
        if not HAS_PANDAS:
            return {
                'file_type': 'excel',
                'error': 'Pandas not available for Excel inspection'
            }
        
        try:
            # Get sheet names
            excel_file = pd.ExcelFile(file_path)
            sheet_names = excel_file.sheet_names
            
            # Read first sheet for schema
            df = pd.read_excel(file_path, sheet_name=sheet_names[0], nrows=100)
            
            schema = []
            for col in df.columns:
                schema.append({
                    'name': col,
                    'dtype': str(df[col].dtype)
                })
            
            return {
                'file_type': 'excel',
                'sheet_names': sheet_names,
                'first_sheet': {
                    'name': sheet_names[0],
                    'total_columns': len(df.columns),
                    'column_names': list(df.columns),
                    'schema': schema
                }
            }
        except Exception as e:
            return {
                'file_type': 'excel',
                'error': f"Failed to parse Excel: {str(e)}"
            }
    
    def _inspect_binary_file(self, file_path: Path) -> Dict[str, Any]:
        """Inspect binary files by reading magic header."""
        try:
            with open(file_path, 'rb') as f:
                header_bytes = f.read(16)
            
            # Convert to hex for readability
            header_hex = header_bytes.hex()
            
            # Try to guess file type from magic bytes
            file_type = self._guess_binary_type(header_bytes)
            
            return {
                'file_type': 'binary',
                'magic_header_hex': header_hex,
                'magic_header_bytes': list(header_bytes),
                'guessed_type': file_type
            }
        except Exception as e:
            return {
                'file_type': 'binary',
                'error': f"Failed to read binary file: {str(e)}"
            }
    
    def _guess_binary_type(self, header_bytes: bytes) -> str:
        """Guess binary file type from magic header bytes."""
        # Common magic bytes patterns
        magic_patterns = {
            b'\x89PNG\r\n\x1a\n': 'PNG image',
            b'\xff\xd8\xff': 'JPEG image',
            b'GIF8': 'GIF image',
            b'PK\x03\x04': 'ZIP archive',
            b'PK\x05\x06': 'ZIP archive (empty)',
            b'%PDF': 'PDF document',
            b'\x7fELF': 'ELF executable',
            b'MZ': 'Windows executable',
            b'\xca\xfe\xba\xbe': 'Java class file',
        }
        
        for magic, file_type in magic_patterns.items():
            if header_bytes.startswith(magic):
                return file_type
        
        return 'unknown'
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size in human-readable format."""
        if size_bytes == 0:
            return "0 B"
        
        units = ['B', 'KB', 'MB', 'GB', 'TB']
        unit_index = 0
        size = float(size_bytes)
        
        while size >= 1024 and unit_index < len(units) - 1:
            size /= 1024
            unit_index += 1
        
        return f"{size:.1f} {units[unit_index]}"
    
    def _get_numeric_statistics(self, series: 'pd.Series') -> Dict[str, Any]:
        """Calculate descriptive statistics for numeric columns."""
        try:
            # Remove null values for calculations
            clean_series = series.dropna()
            
            if len(clean_series) == 0:
                return {'statistics_note': 'No non-null values for statistics'}
            
            stats = {
                'min': float(clean_series.min()),
                'max': float(clean_series.max()),
                'mean': float(clean_series.mean()),
                'median': float(clean_series.median()),
                'std': float(clean_series.std()) if len(clean_series) > 1 else 0.0,
                'unique_count': int(clean_series.nunique()),
                'data_type': 'numeric'
            }
            
            # Add quartiles
            try:
                stats['q25'] = float(clean_series.quantile(0.25))
                stats['q75'] = float(clean_series.quantile(0.75))
            except Exception:
                pass  # Skip quantiles if calculation fails
                
            return stats
            
        except Exception as e:
            return {'statistics_error': f'Failed to calculate numeric stats: {str(e)}'}
    
    def _get_categorical_statistics(self, series: 'pd.Series') -> Dict[str, Any]:
        """Calculate descriptive statistics for categorical/text columns."""
        try:
            # Remove null values for calculations
            clean_series = series.dropna()
            
            if len(clean_series) == 0:
                return {'statistics_note': 'No non-null values for statistics'}
            
            # Get value counts (top 5)
            value_counts = clean_series.value_counts().head(5)
            
            stats = {
                'unique_count': int(clean_series.nunique()),
                'most_frequent': str(value_counts.index[0]) if len(value_counts) > 0 else None,
                'most_frequent_count': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                'top_5_values': [
                    {'value': str(val), 'count': int(count)} 
                    for val, count in value_counts.items()
                ],
                'data_type': 'categorical'
            }
            
            # Add average string length for text data
            if clean_series.dtype == 'object':
                try:
                    str_lengths = clean_series.astype(str).str.len()
                    stats['avg_string_length'] = float(str_lengths.mean())
                    stats['max_string_length'] = int(str_lengths.max())
                except Exception:
                    pass  # Skip string length stats if calculation fails
                    
            return stats
            
        except Exception as e:
            return {'statistics_error': f'Failed to calculate categorical stats: {str(e)}'}