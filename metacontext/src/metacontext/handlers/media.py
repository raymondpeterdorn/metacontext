"""Media handler for images, audio, and video files.

This handler processes media files to extract metadata using
both deterministic techniques and AI enrichment. It implements the architectural
patterns defined in the central architecture reference.

See:
- architecture_reference.ArchitecturalComponents.TWO_TIER_ARCHITECTURE
- architecture_reference.ArchitecturalComponents.SCHEMA_FIRST_LLM
"""

import logging
import mimetypes
from pathlib import Path
from typing import Any, ClassVar

from metacontext.ai.handlers.llms.prompt_constraints import (
    COMMON_FIELD_CONSTRAINTS,
    build_schema_constraints,
    calculate_response_limits,
)
from metacontext.handlers.base import BaseFileHandler, register_handler
from metacontext.schemas.extensions.media import (
    MediaAIEnrichment,
    MediaContext,
    MediaDeterministicMetadata,
    MediaInfo,
)

logger = logging.getLogger(__name__)

# Try to import optional media libraries
try:
    from PIL import Image

    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import mutagen

    MUTAGEN_AVAILABLE = True
except ImportError:
    MUTAGEN_AVAILABLE = False
    mutagen = None

try:
    import cv2

    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None


@register_handler
class MediaHandler(BaseFileHandler):
    """Handler for media files - images, audio, and video.

    Supports: JPEG, PNG, GIF, MP4, AVI, MP3, WAV, etc.
    Extensions: media_context
    """

    supported_extensions: ClassVar[list[str]] = [
        # Image formats
        ".jpg",
        ".jpeg",
        ".png",
        ".gif",
        ".bmp",
        ".tiff",
        ".tif",
        ".webp",
        ".svg",
        # Video formats
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".wmv",
        ".flv",
        ".webm",
        ".m4v",
        ".mpg",
        ".mpeg",
        # Audio formats
        ".mp3",
        ".wav",
        ".flac",
        ".aac",
        ".ogg",
        ".wma",
        ".m4a",
        ".opus",
    ]

    def __init__(self) -> None:
        """Initialize the media handler."""

    def can_handle(self, file_path: Path, data_object: object | None = None) -> bool:
        """Check if this is a media file."""
        if file_path.suffix.lower() in self.supported_extensions:
            return True

        # Check MIME type as additional verification
        mime_type, _ = mimetypes.guess_type(str(file_path))
        return mime_type and mime_type.startswith(("image/", "video/", "audio/"))

    def get_required_extensions(
        self, file_path: Path, data_object: object = None
    ) -> list[str]:
        """Return required extensions for media files."""
        return ["media_context"]

    def fast_probe(self, file_path: Path) -> dict[str, object]:
        """Fast probe to check file compatibility and get basic metadata."""
        if not file_path.exists():
            return {"can_handle": False, "error": "File does not exist"}

        file_size = file_path.stat().st_size
        extension = file_path.suffix.lower()
        mime_type, _ = mimetypes.guess_type(str(file_path))

        media_type = self._determine_media_type(file_path, mime_type)

        return {
            "can_handle": True,
            "file_size": file_size,
            "extension": extension,
            "mime_type": mime_type or "unknown",
            "media_type": media_type,
        }

    def _determine_media_type(self, file_path: Path, mime_type: str | None) -> str:
        """Determine the media type (image, audio, video)."""
        # Try MIME type first
        if mime_type:
            for media_prefix, media_type in [
                ("image/", "image"),
                ("audio/", "audio"),
                ("video/", "video"),
            ]:
                if mime_type.startswith(media_prefix):
                    return media_type

        # Fallback to extension-based detection
        extension = file_path.suffix.lower()
        extension_mapping = {
            **dict.fromkeys(
                {
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".gif",
                    ".bmp",
                    ".tiff",
                    ".tif",
                    ".webp",
                    ".svg",
                },
                "image",
            ),
            **dict.fromkeys(
                {".mp3", ".wav", ".flac", ".aac", ".ogg", ".wma", ".m4a", ".opus"},
                "audio",
            ),
            **dict.fromkeys(
                {
                    ".mp4",
                    ".avi",
                    ".mov",
                    ".mkv",
                    ".wmv",
                    ".flv",
                    ".webm",
                    ".m4v",
                    ".mpg",
                    ".mpeg",
                },
                "video",
            ),
        }

        return extension_mapping.get(extension, "unknown")

    def generate_context(
        self,
        file_path: Path,
        data_object: object | None = None,
        codebase_context: dict[str, object] | None = None,
        ai_companion: object | None = None,
    ) -> dict[str, Any]:
        """Generate media context with deterministic metadata and AI enrichment."""
        try:
            # Deterministic analysis
            deterministic_metadata = self._analyze_media_deterministic(file_path)

            # AI enrichment
            ai_enrichment = None
            if ai_companion and hasattr(ai_companion, "generate_with_schema"):
                ai_enrichment = self._generate_media_ai_enrichment(
                    file_path, deterministic_metadata, ai_companion
                )

            return {
                "media_context": MediaContext(
                    deterministic_metadata=deterministic_metadata,
                    ai_enrichment=ai_enrichment,
                ).model_dump(),
            }
        except Exception:
            logger.exception("Error generating media context for %s", file_path)
            return {"error": "Failed to generate media context"}

    def _analyze_media_deterministic(
        self, file_path: Path
    ) -> MediaDeterministicMetadata:
        """Analyze media file deterministically."""
        metadata = MediaDeterministicMetadata()

        # Basic file information
        metadata.file_size_bytes = file_path.stat().st_size

        mime_type, _ = mimetypes.guess_type(str(file_path))
        metadata.media_type = self._determine_media_type(file_path, mime_type)

        # Create media info based on type
        media_info = MediaInfo()

        # Type-specific analysis
        if metadata.media_type == "image":
            self._analyze_image_metadata(file_path, media_info)
        elif metadata.media_type == "audio":
            self._analyze_audio_metadata(file_path, media_info)
        elif metadata.media_type == "video":
            self._analyze_video_metadata(file_path, media_info)

        metadata.media_info = media_info
        return metadata

    def analyze_deterministic(
        self, file_path: Path, data_object: object = None
    ) -> dict[str, object]:
        """Analyze file without AI - deterministic analysis only."""
        metadata = self._analyze_media_deterministic(file_path)
        return {"media_metadata": metadata.model_dump()}

    def analyze_deep(
        self,
        file_path: Path,
        data_object: object = None,
        ai_companion: object | None = None,
        deterministic_context: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Deep analysis using AI and heavy computation."""
        if (
            not ai_companion
            or not hasattr(ai_companion, "is_available")
            or not ai_companion.is_available()
        ):
            return {"error": "AI companion not available for deep analysis"}

        try:
            # Get or generate deterministic metadata
            metadata = (
                deterministic_context.get("media_metadata")
                if deterministic_context
                else None
            )
            if not metadata:
                metadata = self._analyze_media_deterministic(file_path).model_dump()

            ai_enrichment = self._generate_media_ai_enrichment(
                file_path,
                MediaDeterministicMetadata.model_validate(metadata),
                ai_companion,
            )
            return {
                "media_ai_enrichment": ai_enrichment.model_dump()
                if ai_enrichment
                else None
            }
        except Exception:
            logger.exception("Error in deep analysis for %s", file_path)
            return {"error": "Deep analysis failed"}

    def _analyze_image_metadata(self, file_path: Path, media_info: MediaInfo) -> None:
        """Analyze image-specific metadata."""
        if PIL_AVAILABLE:
            try:
                with Image.open(file_path) as img:
                    media_info.dimensions = [img.width, img.height]
                    media_info.color_space = img.mode

            except Exception:
                logger.warning("Error analyzing image %s", file_path)

    def _analyze_audio_metadata(self, file_path: Path, media_info: MediaInfo) -> None:
        """Analyze audio-specific metadata."""
        if MUTAGEN_AVAILABLE:
            try:
                audio_file = mutagen.File(file_path)
                if audio_file and hasattr(audio_file, "info"):
                    media_info.duration = audio_file.info.length
                    media_info.bit_rate = getattr(audio_file.info, "bitrate", None)
                    media_info.sample_rate = getattr(
                        audio_file.info, "sample_rate", None
                    )
                    media_info.channels = getattr(audio_file.info, "channels", None)

            except Exception:
                logger.warning("Error analyzing audio %s", file_path)

    def _analyze_video_metadata(self, file_path: Path, media_info: MediaInfo) -> None:
        """Analyze video-specific metadata."""
        if OPENCV_AVAILABLE:
            try:
                cap = cv2.VideoCapture(str(file_path))
                if cap.isOpened():
                    media_info.dimensions = [
                        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    ]
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                    if fps > 0:
                        media_info.duration = frame_count / fps
                cap.release()
            except Exception:
                logger.warning("Error analyzing video %s", file_path)

    def _build_media_constraints(self, metadata: MediaDeterministicMetadata) -> str:
        """Build constraints for media AI enrichment."""
        # Calculate complexity based on media type and characteristics
        complexity_factor = 1.0

        if metadata.media_info and metadata.media_type == "image":
            # Images with high resolution are more complex
            if metadata.media_info.dimensions:
                total_pixels = (
                    metadata.media_info.dimensions[0]
                    * metadata.media_info.dimensions[1]
                )
                if total_pixels > 2_000_000:  # 2MP+
                    complexity_factor *= 1.3
        elif metadata.media_info and metadata.media_type == "video":
            # Video complexity based on duration and resolution
            if (
                metadata.media_info.duration and metadata.media_info.duration > 300
            ):  # 5+ minutes
                complexity_factor *= 1.4
            if metadata.media_info.dimensions:
                total_pixels = (
                    metadata.media_info.dimensions[0]
                    * metadata.media_info.dimensions[1]
                )
                if total_pixels > 1_000_000:  # HD+
                    complexity_factor *= 1.2
        elif metadata.media_info and metadata.media_type == "audio":
            # Audio complexity based on duration and quality
            if (
                metadata.media_info.duration and metadata.media_info.duration > 300
            ):  # 5+ minutes
                complexity_factor *= 1.3
            if (
                metadata.media_info.bit_rate and metadata.media_info.bit_rate > 256
            ):  # High quality
                complexity_factor *= 1.1

        max_total_chars, max_field_chars = calculate_response_limits(
            base_fields=7,  # ForensicAIEnrichment base fields
            extended_fields=5,  # MediaAIEnrichment specific fields
            complexity_factor=complexity_factor,
        )

        field_constraints = {
            **COMMON_FIELD_CONSTRAINTS,
            "content_description": f"Visual/audio content ({max_field_chars // 2} chars max)",
            "quality_assessment": "Technical quality + defects",
            "technical_analysis": "Format details + compression",
            "use_case_recommendations": "Likely purpose + applications",
            "applications": "Potential domains + usage",
        }

        base_instruction = f"Analyze this {metadata.media_type or 'media'} file"
        constraints = build_schema_constraints(
            max_total_chars=max_total_chars,
            max_field_chars=max_field_chars,
            field_descriptions=field_constraints,
            complexity_context=f"{metadata.media_type}: {metadata.file_size_bytes or 0} bytes",
        )

        return f"{base_instruction} and provide insights that fit within these STRICT LIMITS:\\n\\n{constraints}"

    def _generate_media_ai_enrichment(
        self,
        file_path: Path,
        metadata: MediaDeterministicMetadata,
        ai_companion: object,
    ) -> MediaAIEnrichment | None:
        """Generate AI enrichment for media files."""
        try:
            context_data = {
                "file_name": file_path.name,
                "file_path": str(file_path),
                "media_metadata": metadata.model_dump(),
                "file_size_mb": (metadata.file_size_bytes or 0) / (1024 * 1024),
            }

            instruction = self._build_media_constraints(metadata)

            # Type: ignore needed because ai_companion is typed as object for flexibility
            return ai_companion.generate_with_schema(  # type: ignore[attr-defined]
                schema_class=MediaAIEnrichment,
                context_data=context_data,
                instruction=instruction,
            )
        except Exception:
            logger.exception("Error generating media AI enrichment")
            return None

    # Prompt configuration for bulk analysis
    PROMPT_CONFIG: ClassVar[dict[str, str]] = {
        "media_analysis": "templates/media/media_analysis.yaml",
    }

    def get_bulk_prompts(
        self, file_path: Path, data_object: object = None
    ) -> dict[str, str]:  # noqa: ARG002
        """Get bulk prompts for this file type from config."""
        return self.PROMPT_CONFIG.copy()
