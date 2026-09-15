"""Stage 1 - the structured user request that enters the pipeline."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Sequence

from .enums import DEFAULT_SECTIONS, ReportSection


@dataclass(frozen=True, slots=True)
class ResearchRequest:
    """A user's request for an equity research report.

    Only ``company`` is genuinely mandatory; everything else has a sensible
    default so that a thin request ("report on NVIDIA") still runs end to end.
    """

    company: str
    ticker: str | None = None
    objective: str = "company update"
    sections: tuple[ReportSection, ...] = DEFAULT_SECTIONS
    time_horizon: str = "latest"
    peers: tuple[str, ...] = ()
    focus: tuple[str, ...] = ()
    report_date: dt.date = field(default_factory=dt.date.today)
    raw_prompt: str | None = None

    def __post_init__(self) -> None:
        if not self.company or not self.company.strip():
            raise ValueError("ResearchRequest.company must be a non-empty string")
        if not self.sections:
            raise ValueError("ResearchRequest.sections must not be empty")

    # -- construction ----------------------------------------------------
    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchRequest":
        """Build a request from loose JSON-ish input (CLI, API body, file)."""
        sections = payload.get("sections") or DEFAULT_SECTIONS
        parsed_sections = tuple(
            s if isinstance(s, ReportSection) else ReportSection(str(s)) for s in sections
        )
        report_date = payload.get("report_date")
        if isinstance(report_date, str):
            report_date = dt.date.fromisoformat(report_date)
        elif report_date is None:
            report_date = dt.date.today()

        return cls(
            company=str(payload["company"]).strip(),
            ticker=_clean_ticker(payload.get("ticker")),
            objective=payload.get("objective") or "company update",
            sections=parsed_sections,
            time_horizon=payload.get("time_horizon") or "latest",
            peers=tuple(payload.get("peers") or ()),
            focus=tuple(payload.get("focus") or ()),
            report_date=report_date,
            raw_prompt=payload.get("raw_prompt"),
        )

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "company": self.company,
            "ticker": self.ticker,
            "objective": self.objective,
            "sections": [s.value for s in self.sections],
            "time_horizon": self.time_horizon,
            "peers": list(self.peers),
            "focus": list(self.focus),
            "report_date": self.report_date.isoformat(),
            "raw_prompt": self.raw_prompt,
        }

    def wants(self, section: ReportSection | Sequence[ReportSection]) -> bool:
        """True if the user asked for (any of) the given section(s)."""
        wanted = (section,) if isinstance(section, ReportSection) else tuple(section)
        return any(s in self.sections for s in wanted)


def _clean_ticker(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    return text or None
