"""Metacontext: Automatic Metadata Replacement System.

Replaces traditional file metadata with rich, AI-generated context that
travels with every data file.

Simple Usage:
    import metacontext

    # Your existing data export (unchanged)
    df.to_csv('data.csv')

    # Generate rich context (one line)
    metacontext.metacontextualize(df, 'data.csv')
    # Creates: data.csv.metacontext.yaml
"""

__version__ = "0.2.0"
__author__ = "raymondpeterdorn"
__email__ = "rpd346@gmail.com"

# Universal File Intelligence System (Current)
from src.metacontextualize import MetacontextualizeArgs, metacontextualize

# Core API
__all__ = [
    # LLM Support
    "LLMHandler",
    # Simple API (Recommended)
    "metacontextualize",
]
