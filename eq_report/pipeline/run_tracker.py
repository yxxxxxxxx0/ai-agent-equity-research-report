"""Stage 12 - report run tracking.

Holds the ReportRun record while the pipeline executes and stamps each stage as
it starts and finishes. The record is written to disk whatever the outcome, so a
failed run is as inspectable as a successful one.
"""

from __future__ import annotations

import datetime as dt
import uuid
from contextlib import contextmanager
from typing import Any, Iterator

from ..domain.enums import RunStatus
from ..domain.run import ReportRun, StageTiming
from ..errors import PipelineError
from ..logging_setup import set_run_id, stage_context


def new_run_id() -> str:
    """A short, sortable, unique run id."""
    stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%dT%H%M%S")
    return f"run_{stamp}_{uuid.uuid4().hex[:6]}"


class RunTracker:
    """Accumulates execution metadata for one pipeline invocation."""

    def __init__(self, report_run_id: str, request_payload: dict[str, Any]) -> None:
        self.run = ReportRun(
            report_run_id=report_run_id,
            request=request_payload,
            started_at=dt.datetime.now(dt.UTC),
        )
        set_run_id(report_run_id)

    @contextmanager
    def stage(self, name: str) -> Iterator[StageTiming]:
        """Time a pipeline stage and tag its log records.

        An exception is recorded on the timing and on the run, then re-raised:
        the orchestrator decides whether a given stage failure is fatal.
        """
        timing = StageTiming(stage=name, started_at=dt.datetime.now(dt.UTC))
        self.run.stages.append(timing)
        with stage_context(name):
            try:
                yield timing
            except Exception as exc:
                timing.error = f"{type(exc).__name__}: {exc}"
                timing.completed_at = dt.datetime.now(dt.UTC)
                self.error(PipelineError.from_exception(name, exc))
                raise
            timing.completed_at = dt.datetime.now(dt.UTC)

    # -- recording -------------------------------------------------------
    def error(self, error: PipelineError) -> None:
        self.run.errors.append(error.to_dict())

    def errors(self, errors: tuple[PipelineError, ...]) -> None:
        for error in errors:
            self.error(error)

    def warn(self, message: str) -> None:
        if message not in self.run.warnings:
            self.run.warnings.append(message)

    def set_plan(self, plan_payload: dict[str, Any]) -> None:
        self.run.plan = plan_payload

    def set_source_status(self, statuses: list[dict[str, Any]]) -> None:
        self.run.source_status = statuses

    def set_counts(self, *, evidence: int, analytics: int) -> None:
        self.run.evidence_count = evidence
        self.run.analytics_count = analytics

    def set_agent_status(self, statuses: list[dict[str, Any]]) -> None:
        self.run.agent_status = statuses

    def set_qa(self, qa_payload: dict[str, Any]) -> None:
        self.run.qa = qa_payload

    def set_llm_usage(self, usage_payload: dict[str, Any]) -> None:
        self.run.llm_usage = usage_payload

    def set_outputs(self, *, json_path: str | None, pdf_path: str | None) -> None:
        self.run.report_json_path = json_path
        self.run.report_pdf_path = pdf_path

    def finish(self, status: RunStatus) -> ReportRun:
        self.run.status = status
        self.run.completed_at = dt.datetime.now(dt.UTC)
        return self.run
