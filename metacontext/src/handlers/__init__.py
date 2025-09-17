"""Universal file intelligence handlers package.

This package provides the handler system for generating intelligent metadata
for any file type using the Core + Extensions architecture.
"""

from metacontext.handlers.base import BaseFileHandler, MetacontextArgs, register_handler
from metacontext.handlers.model import ModelHandler
from metacontext.handlers.tabular import CSVHandler

__all__ = [
    "BaseFileHandler",
    "CSVHandler",
    "MetacontextArgs",
    "ModelHandler",
    "register_handler",
]
