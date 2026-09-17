"""Stage 3 - raw provider output, *before* normalisation.

Providers deliberately emit loose values (strings, mixed units, vendor metric
names). Cleaning them up is the normalisation layer's job, and keeping the raw
shape here is what makes the provenance chain auditable.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .enums import Confidence, ProviderStatus, SourceType


@dataclass(frozen=True, slots=True)
class SourceRef:
    """Identifies where a raw datapoint came from."""

    source_id: str
    source_name: str
    source_type: SourceType
    source_url: str | None = None
    retrieval_provider: str | None = None
    retrieval_url: str | None = None
    original_source_name: str | None = None
    original_source_url: str | None = None
    original_publication_date: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "retrieval_provider": self.retrieval_provider,
            "retrieval_url": self.retrieval_url,
            "original_source_name": self.original_source_name,
            "original_source_url": self.original_source_url,
            "original_publication_date": self.original_publication_date,
        }


@dataclass(frozen=True, slots=True)
class RawObservation:
    """A single numeric/textual datapoint as the provider reported it."""

    metric: str                      # vendor metric name, not yet canonical
    value: Any                       # may be str, int, float, None
    source: SourceRef
    unit: str | None = None
    currency: str | None = None
    as_of: str | None = None         # provider-formatted date string
    period: str | None = None        # provider-formatted fiscal period
    period_end: str | None = None
    company: str | None = None
    ticker: str | None = None
    confidence: Confidence = Confidence.UNKNOWN
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "currency": self.currency,
            "as_of": self.as_of,
            "period": self.period,
            "period_end": self.period_end,
            "company": self.company,
            "ticker": self.ticker,
            "confidence": self.confidence.value,
            "source": self.source.to_dict(),
            "metadata": dict(self.metadata),
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class RawDocumentPassage:
    """An extracted passage/claim from a document, with full citation data."""

    title: str
    source: SourceRef
    published_at: str | None
    text: str
    company: str | None = None
    ticker: str | None = None
    section: str | None = None       # section heading or page number
    speaker: str | None = None       # for transcripts: who said it
    confidence: Confidence = Confidence.MEDIUM
    metadata: dict[str, Any] = field(default_factory=dict)
    retrieved_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "published_at": self.published_at,
            "text": self.text,
            "company": self.company,
            "ticker": self.ticker,
            "section": self.section,
            "speaker": self.speaker,
            "confidence": self.confidence.value,
            "source": self.source.to_dict(),
            "metadata": dict(self.metadata),
            "retrieved_at": self.retrieved_at.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class ProviderResult:
    """What one acquisition branch returned, including how it went.

    A failed provider yields an empty payload plus a populated ``errors`` list
    rather than raising: the pipeline degrades instead of dying.
    """

    provider_name: str
    branch: str                      # "market_data" | "fundamentals" | "documents"
    status: ProviderStatus
    observations: tuple[RawObservation, ...] = ()
    passages: tuple[RawDocumentPassage, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    duration_ms: float = 0.0

    @property
    def item_count(self) -> int:
        return len(self.observations) + len(self.passages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider_name": self.provider_name,
            "branch": self.branch,
            "status": self.status.value,
            "observation_count": len(self.observations),
            "passage_count": len(self.passages),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "duration_ms": round(self.duration_ms, 1),
        }
