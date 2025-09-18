"""Media extension schemas."""

from pydantic import BaseModel, Field

from metacontext.schemas.extensions.base import (
    DeterministicMetadata,
    ForensicAIEnrichment,
)


class MediaInfo(BaseModel):
    """Information about media file properties."""

    duration: float | None = None  # seconds
    dimensions: list[int] | None = None  # [width, height] for images/video
    sample_rate: int | None = None  # Hz for audio
    bit_rate: int | None = None
    channels: int | None = None  # audio channels
    color_space: str | None = None  # for images/video


class MediaDeterministicMetadata(DeterministicMetadata):
    """Deterministic facts about media files."""

    media_type: str | None = None  # image, audio, video
    file_size_bytes: int | None = None
    media_info: MediaInfo | None = None


class MediaAIEnrichment(ForensicAIEnrichment):
    """AI-generated insights about media files."""

    content_description: str | None = Field(
        None,
        description="Detailed description of the media content, including subjects, themes, or audio content.",
    )
    quality_assessment: str | None = Field(
        None,
        description="Assessment of the media quality, including resolution, clarity, noise levels, or compression artifacts.",
    )
    technical_analysis: str | None = Field(
        None,
        description="Technical analysis of the media file, including encoding, format characteristics, or technical issues.",
    )
    use_case_recommendations: list[str] | None = Field(
        None,
        description="Recommended use cases or applications for the media file based on its content and quality.",
    )
    applications: list[str] | None = Field(
        None,
        description="Potential applications or domains where this media file would be useful or relevant.",
    )
    processing_recommendations: str | None = Field(
        None,
        description="Recommendations for processing or enhancing the media file to improve its quality or usability.",
    )


class MediaContext(BaseModel):
    """Context derived from media file analysis."""

    deterministic_metadata: MediaDeterministicMetadata | None = None
    ai_enrichment: MediaAIEnrichment | None = None
