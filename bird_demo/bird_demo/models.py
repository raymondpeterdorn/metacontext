# models.py
"""Pydantic models for the example."""

import datetime
import zoneinfo

from pydantic import BaseModel, field_validator


class BirdData(BaseModel):
    """Represents a single observation of a bird."""

    species_name: str
    taxonomic_family: str
    observation_date: datetime.date
    location_description: str
    notes: str | None = None
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
