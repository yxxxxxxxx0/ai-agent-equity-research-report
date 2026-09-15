"""Explicit exception types and the structured error object used for reporting.

Rule of thumb used throughout the pipeline: recoverable problems become
PipelineError records attached to the run; unrecoverable ones raise.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any


class EqReportError(Exception):
    """Base class for every error raised by this package."""


class ConfigurationError(EqReportError):
    """Settings are missing or inconsistent."""


class ProviderError(EqReportError):
    """A data provider failed. Caught per-branch so the run can continue."""

    def __init__(self, provider: str, message: str, *, recoverable: bool = True) -> None:
        super().__init__(f"{provider}: {message}")
        self.provider = provider
        self.recoverable = recoverable


class NormalisationError(EqReportError):
    """A raw value could not be normalised.

    Raised (and caught by the normaliser, which records a rejection) rather than
    silently coercing a malformed value into valid-looking evidence.
    """

    def __init__(self, field_name: str, value: Any, message: str) -> None:
        super().__init__(f"{field_name}={value!r}: {message}")
        self.field_name = field_name
        self.value = value


class EvidenceStoreError(EqReportError):
    """Persisting or querying evidence failed."""


class AnalyticsError(EqReportError):
    """A calculation could not be performed with the available inputs."""


class SegmentAgentError(EqReportError):
    """A segment agent failed. Isolated so other agents still produce output."""

    def __init__(self, segment: str, message: str) -> None:
        super().__init__(f"{segment}: {message}")
        self.segment = segment


class QACriticalFailure(EqReportError):
    """QA found blocking problems; the PDF must not be generated."""


class RenderingError(EqReportError):
    """PDF or JSON rendering failed."""


@dataclass(frozen=True, slots=True)
class PipelineError:
    """A recorded, non-fatal failure attached to a report run."""

    stage: str
    kind: str
    message: str
    recoverable: bool = True
    context: dict[str, Any] = field(default_factory=dict)
    occurred_at: dt.datetime = field(default_factory=lambda: dt.datetime.now(dt.UTC))

    @classmethod
    def from_exception(cls, stage: str, exc: BaseException, **context: Any) -> "PipelineError":
        return cls(
            stage=stage,
            kind=type(exc).__name__,
            message=str(exc),
            recoverable=getattr(exc, "recoverable", True),
            context=context,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "kind": self.kind,
            "message": self.message,
            "recoverable": self.recoverable,
            "context": dict(self.context),
            "occurred_at": self.occurred_at.isoformat(),
        }
