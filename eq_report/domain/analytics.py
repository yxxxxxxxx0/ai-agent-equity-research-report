"""Stage 6 - deterministic analytics results."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

from .enums import Confidence


@dataclass(frozen=True, slots=True)
class AnalyticsResult:
    """One calculated number, with the evidence it was derived from.

    ``formula`` is stored as text so QA can recompute the result independently
    and so the report can show its working.
    """

    analytics_id: str
    report_run_id: str
    metric: str                       # e.g. "revenue_growth_yoy"
    value: float
    unit: str                         # "pct" | "x" | "USD" | "pp" ...
    formula: str
    input_evidence_ids: tuple[str, ...]
    inputs: dict[str, float] = field(default_factory=dict)
    label: str = ""                   # human-readable, e.g. "Revenue growth YoY"
    period: str | None = None
    comparison_period: str | None = None
    confidence: Confidence = Confidence.HIGH
    metadata: dict[str, Any] = field(default_factory=dict)
    generated_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    def to_dict(self) -> dict[str, Any]:
        return {
            "analytics_id": self.analytics_id,
            "report_run_id": self.report_run_id,
            "metric": self.metric,
            "value": self.value,
            "unit": self.unit,
            "formula": self.formula,
            "input_evidence_ids": list(self.input_evidence_ids),
            "inputs": dict(self.inputs),
            "label": self.label,
            "period": self.period,
            "comparison_period": self.comparison_period,
            "confidence": self.confidence.value,
            "metadata": dict(self.metadata),
            "generated_at": self.generated_at.isoformat(),
        }


def make_analytics_id(report_run_id: str, metric: str, *parts: Any) -> str:
    payload = json.dumps([report_run_id, metric, *[str(p) for p in parts]], sort_keys=True)
    return f"an_{hashlib.sha1(payload.encode('utf-8')).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class AnalyticsBundle:
    """All analytics for a run, with lookup helpers for the agents."""

    results: tuple[AnalyticsResult, ...] = ()
    errors: tuple[str, ...] = ()

    def by_metric(self, metric: str) -> tuple[AnalyticsResult, ...]:
        return tuple(r for r in self.results if r.metric == metric)

    def first(self, metric: str) -> AnalyticsResult | None:
        hits = self.by_metric(metric)
        return hits[0] if hits else None

    def get(self, analytics_id: str) -> AnalyticsResult | None:
        for r in self.results:
            if r.analytics_id == analytics_id:
                return r
        return None

    def with_prefix(self, prefix: str) -> tuple[AnalyticsResult, ...]:
        return tuple(r for r in self.results if r.metric.startswith(prefix))

    def to_dict(self) -> dict[str, Any]:
        return {
            "results": [r.to_dict() for r in self.results],
            "errors": list(self.errors),
        }
