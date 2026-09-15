"""Stage 12 - report run tracking."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .enums import RunStatus


@dataclass(slots=True)
class StageTiming:
    stage: str
    started_at: dt.datetime
    completed_at: dt.datetime | None = None
    error: str | None = None

    @property
    def duration_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": round(self.duration_ms, 1) if self.duration_ms is not None else None,
            "error": self.error,
        }


@dataclass(slots=True)
class ReportRun:
    """Execution metadata for one pipeline invocation.

    This is written to disk on every run (success or failure) so any stage can
    be inspected after the fact.
    """

    report_run_id: str
    request: dict[str, Any]
    started_at: dt.datetime
    completed_at: dt.datetime | None = None
    status: RunStatus | None = None
    plan: dict[str, Any] | None = None
    source_status: list[dict[str, Any]] = field(default_factory=list)
    evidence_count: int = 0
    analytics_count: int = 0
    agent_status: list[dict[str, Any]] = field(default_factory=list)
    qa: dict[str, Any] | None = None
    llm_usage: dict[str, Any] | None = None
    stages: list[StageTiming] = field(default_factory=list)
    report_json_path: str | None = None
    report_pdf_path: str | None = None
    errors: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> float | None:
        if self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_run_id": self.report_run_id,
            "status": self.status.value if self.status else None,
            "request": self.request,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_ms": round(self.duration_ms, 1) if self.duration_ms is not None else None,
            "plan": self.plan,
            "source_status": self.source_status,
            "evidence_count": self.evidence_count,
            "analytics_count": self.analytics_count,
            "agent_status": self.agent_status,
            "qa": self.qa,
            "llm_usage": self.llm_usage,
            "stages": [s.to_dict() for s in self.stages],
            "report_json_path": self.report_json_path,
            "report_pdf_path": self.report_pdf_path,
            "errors": self.errors,
            "warnings": self.warnings,
        }
