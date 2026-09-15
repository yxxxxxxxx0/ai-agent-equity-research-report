"""Stage 9 - QA findings and result object."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import Severity


@dataclass(frozen=True, slots=True)
class QAFinding:
    """One problem detected by a QA check."""

    check: str                        # e.g. "evidence.reference_exists"
    severity: Severity
    message: str
    section: str | None = None
    subject: str | None = None        # statement text / metric / evidence id
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check": self.check,
            "severity": self.severity.value,
            "message": self.message,
            "section": self.section,
            "subject": self.subject,
            "details": dict(self.details),
        }


@dataclass(frozen=True, slots=True)
class QAResult:
    """Aggregated QA outcome. Critical findings block PDF generation."""

    findings: tuple[QAFinding, ...] = ()
    checks_run: tuple[str, ...] = ()

    @property
    def critical(self) -> tuple[QAFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.CRITICAL)

    @property
    def warnings(self) -> tuple[QAFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.WARNING)

    @property
    def infos(self) -> tuple[QAFinding, ...]:
        return tuple(f for f in self.findings if f.severity is Severity.INFO)

    @property
    def has_critical_errors(self) -> bool:
        return bool(self.critical)

    @property
    def passed(self) -> bool:
        return not self.has_critical_errors

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "has_critical_errors": self.has_critical_errors,
            "counts": {
                "critical": len(self.critical),
                "warning": len(self.warnings),
                "info": len(self.infos),
            },
            "checks_run": list(self.checks_run),
            "findings": [f.to_dict() for f in self.findings],
        }
