"""Stage 5 - the canonical evidence item, the pipeline's unit of truth.

Everything a report claims must point at one or more of these. An item keeps
both the normalised value (for analytics) and the raw value/metric name (for
traceability back to the original source).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .enums import Confidence, EvidenceCategory, EvidenceStatus, FactType, SourceType


@dataclass(frozen=True, slots=True)
class FiscalPeriod:
    """A normalised fiscal period, e.g. FY2026 Q2."""

    label: str                 # canonical label, e.g. "FY2026 Q2"
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    period_end: dt.date | None = None
    is_annual: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "fiscal_year": self.fiscal_year,
            "fiscal_quarter": self.fiscal_quarter,
            "period_end": self.period_end.isoformat() if self.period_end else None,
            "is_annual": self.is_annual,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "FiscalPeriod | None":
        if not payload:
            return None
        end = payload.get("period_end")
        return cls(
            label=payload["label"],
            fiscal_year=payload.get("fiscal_year"),
            fiscal_quarter=payload.get("fiscal_quarter"),
            period_end=dt.date.fromisoformat(end) if end else None,
            is_annual=bool(payload.get("is_annual")),
        )


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """One normalised, provenance-carrying fact.

    Numeric evidence populates ``metric``/``value``; document evidence populates
    ``claim_text``. Both carry the same source metadata so citations are uniform.
    """

    evidence_id: str
    report_run_id: str
    company: str
    ticker: str | None
    category: EvidenceCategory
    source_id: str
    source_name: str
    source_type: SourceType
    retrieved_at: dt.datetime
    metric: str | None = None            # canonical metric id
    value: float | None = None           # normalised numeric value
    unit: str | None = None              # canonical unit, e.g. "USD", "pct", "x"
    currency: str | None = None
    period: FiscalPeriod | None = None
    as_of: dt.date | None = None
    source_url: str | None = None
    retrieval_provider: str | None = None
    retrieval_url: str | None = None
    original_source_name: str | None = None
    original_source_url: str | None = None
    original_publication_date: dt.date | None = None
    basis: str | None = None
    frequency: str | None = None
    period_start: dt.date | None = None
    fact_type: FactType = FactType.REPORTED_FACT
    status: EvidenceStatus = EvidenceStatus.UNVERIFIED
    is_canonical: bool = False
    reconciliation_key: str | None = None
    alternate_evidence_ids: tuple[str, ...] = ()
    validation_messages: tuple[str, ...] = ()
    claim_text: str | None = None        # document evidence
    document_title: str | None = None
    published_at: dt.date | None = None
    confidence: Confidence = Confidence.UNKNOWN
    raw_metric: str | None = None        # provider's original metric name
    raw_value: Any = None                # provider's original value
    is_mock: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def period_label(self) -> str | None:
        return self.period.label if self.period else None

    def citation(self) -> str:
        """Short human-readable citation used in the report's source list."""
        bits = [self.original_source_name or self.source_name]
        if self.document_title and self.document_title != self.source_name:
            bits.append(self.document_title)
        date = self.published_at or self.as_of
        if date:
            bits.append(date.isoformat())
        return " - ".join(bits)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "report_run_id": self.report_run_id,
            "company": self.company,
            "ticker": self.ticker,
            "category": self.category.value,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "currency": self.currency,
            "period": self.period.to_dict() if self.period else None,
            "as_of": self.as_of.isoformat() if self.as_of else None,
            "source_id": self.source_id,
            "source_name": self.source_name,
            "source_type": self.source_type.value,
            "source_url": self.source_url,
            "retrieval_provider": self.retrieval_provider,
            "retrieval_url": self.retrieval_url,
            "original_source_name": self.original_source_name,
            "original_source_url": self.original_source_url,
            "original_publication_date": self.original_publication_date.isoformat() if self.original_publication_date else None,
            "basis": self.basis,
            "frequency": self.frequency,
            "period_start": self.period_start.isoformat() if self.period_start else None,
            "fact_type": self.fact_type.value,
            "status": self.status.value,
            "is_canonical": self.is_canonical,
            "reconciliation_key": self.reconciliation_key,
            "alternate_evidence_ids": list(self.alternate_evidence_ids),
            "validation_messages": list(self.validation_messages),
            "claim_text": self.claim_text,
            "document_title": self.document_title,
            "published_at": self.published_at.isoformat() if self.published_at else None,
            "retrieved_at": self.retrieved_at.isoformat(),
            "confidence": self.confidence.value,
            "raw_metric": self.raw_metric,
            "raw_value": self.raw_value,
            "is_mock": self.is_mock,
            "metadata": dict(self.metadata),
        }


def make_evidence_id(report_run_id: str, *parts: Any) -> str:
    """Deterministic evidence id.

    Deterministic (rather than random) ids mean re-running the pipeline over the
    same inputs produces stable references, which makes diffing report runs and
    debugging citations much easier.
    """
    payload = json.dumps([report_run_id, *[str(p) for p in parts]], sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"ev_{digest}"
