# models.py
"""Pydantic models for the example."""

import datetime
import zoneinfo

from pydantic import BaseModel, Field, field_validator


class BirdData(BaseModel):
    """Represents a single observation of a bird."""

    species_name: str
    taxonomic_family: str
    taxonomic_order: str
    asdawas: str = Field(description = "Wing Length")
    beak_length: float
    nocturnal_diurnal: str
    diet_types: str
    closest_relatives: str
    observation_date: datetime.date
    location_description: str
    brrrrkk: str | None = None

    @field_validator("observation_date", mode="before")
    @classmethod
    def parse_date(cls, v: str) -> datetime.date:
        """Parse date from string."""
        return (
            datetime.datetime.strptime(v, "%Y-%m-%d")
            .replace(tzinfo=zoneinfo.ZoneInfo("UTC"))
            .date()
        )
