"""Stage 7 - structured output of a segment agent (never rendered text)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ClaimType, Confidence, SegmentName


@dataclass(frozen=True, slots=True)
class KeyFinding:
    """One traceable statement made by an agent.

    ``claim_type`` separates a reported fact from an inference, and at least one
    of ``evidence_ids``/``analytics_ids`` must be present unless the claim is a
    pure interpretation built on other findings.
    """

    claim: str
    claim_type: ClaimType
    evidence_ids: tuple[str, ...] = ()
    analytics_ids: tuple[str, ...] = ()
    confidence: Confidence = Confidence.MEDIUM
    materiality: int = 2              # 1 = highly material, 3 = context
    tags: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim,
            "claim_type": self.claim_type.value,
            "evidence_ids": list(self.evidence_ids),
            "analytics_ids": list(self.analytics_ids),
            "confidence": self.confidence.value,
            "materiality": self.materiality,
            "tags": list(self.tags),
        }


@dataclass(frozen=True, slots=True)
class MetricHighlight:
    """A metric an agent wants surfaced in a table."""

    label: str
    value_text: str                   # already formatted for display
    period: str | None = None
    comparison: str | None = None     # e.g. "+56% YoY"
    evidence_ids: tuple[str, ...] = ()
    analytics_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "value_text": self.value_text,
            "period": self.period,
            "comparison": self.comparison,
            "evidence_ids": list(self.evidence_ids),
            "analytics_ids": list(self.analytics_ids),
        }


#: How much a gap limits the analysis. Assigned by the agent that raised it
#: (or defaulted to "medium"), so a report with dozens of gaps can be scanned
#: for the handful that actually matter rather than read as one undifferentiated
#: list - see rendering.pdf_renderer for how this groups the Data Gaps table.
DataGapPriority = str  # "high" | "medium" | "low"


@dataclass(frozen=True, slots=True)
class DataGap:
    """Something the agent needed but the Evidence Store could not supply."""

    description: str
    missing_metric: str | None = None
    impact: str = ""
    segment: SegmentName | None = None
    priority: DataGapPriority = "medium"

    def to_dict(self) -> dict[str, Any]:
        return {
            "description": self.description,
            "missing_metric": self.missing_metric,
            "impact": self.impact,
            "segment": self.segment.value if self.segment else None,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class SegmentResult:
    """The contract every segment agent returns."""

    segment: SegmentName
    headline: str
    key_findings: tuple[KeyFinding, ...] = ()
    important_metrics: tuple[MetricHighlight, ...] = ()
    open_questions: tuple[str, ...] = ()
    data_gaps: tuple[DataGap, ...] = ()
    draft_narrative: str = ""
    errors: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_evidence_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for f in self.key_findings:
            ids.extend(f.evidence_ids)
        for m in self.important_metrics:
            ids.extend(m.evidence_ids)
        return tuple(dict.fromkeys(ids))

    @property
    def all_analytics_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for f in self.key_findings:
            ids.extend(f.analytics_ids)
        for m in self.important_metrics:
            ids.extend(m.analytics_ids)
        return tuple(dict.fromkeys(ids))

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment.value,
            "headline": self.headline,
            "key_findings": [f.to_dict() for f in self.key_findings],
            "important_metrics": [m.to_dict() for m in self.important_metrics],
            "open_questions": list(self.open_questions),
            "data_gaps": [g.to_dict() for g in self.data_gaps],
            "draft_narrative": self.draft_narrative,
            "errors": list(self.errors),
            "metadata": dict(self.metadata),
        }
