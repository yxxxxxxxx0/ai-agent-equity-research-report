"""Stage 8 - the structured ReportDraft consumed by QA and the PDF renderer."""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any

from .enums import ClaimType, Confidence, ReportSection


@dataclass(frozen=True, slots=True)
class Citation:
    """A source reference rendered in the report's source list."""

    ref_number: int
    evidence_id: str
    text: str
    source_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ref_number": self.ref_number,
            "evidence_id": self.evidence_id,
            "text": self.text,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class Statement:
    """A single rendered sentence/bullet plus its provenance."""

    text: str
    claim_type: ClaimType
    evidence_ids: tuple[str, ...] = ()
    analytics_ids: tuple[str, ...] = ()
    citation_refs: tuple[int, ...] = ()
    confidence: Confidence = Confidence.MEDIUM

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "claim_type": self.claim_type.value,
            "evidence_ids": list(self.evidence_ids),
            "analytics_ids": list(self.analytics_ids),
            "citation_refs": list(self.citation_refs),
            "confidence": self.confidence.value,
        }


@dataclass(frozen=True, slots=True)
class MetricRow:
    """One row of a metric table: a row label plus one cell per data column.

    ``cells`` is positional against :attr:`MetricTable.columns` minus its first
    (row-label) entry. A cell the evidence cannot fill is the empty string
    rather than a placeholder, which is what lets the renderer drop a column
    that turned out to be empty everywhere.
    """

    label: str
    cells: tuple[str, ...] = ()
    emphasis: bool = False            # e.g. the subject company in a peer table

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cells": list(self.cells),
            "emphasis": self.emphasis,
        }


@dataclass(frozen=True, slots=True)
class MetricTable:
    """A small grid exhibit.

    ``columns[0]`` heads the row-label column and is often blank; the rest head
    the data columns, in the same order as each row's ``cells``.
    """

    title: str
    columns: tuple[str, ...]
    rows: tuple[MetricRow, ...]
    note: str = ""                    # source/units line printed under the grid

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "columns": list(self.columns),
            "rows": [r.to_dict() for r in self.rows],
            "note": self.note,
        }


@dataclass(frozen=True, slots=True)
class KeyDataItem:
    """One line of the first-page key-data panel."""

    label: str
    value: str

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "value": self.value}


@dataclass(frozen=True, slots=True)
class KeyDataGroup:
    """A titled block of key-data lines (market data, valuation, ...)."""

    title: str
    items: tuple[KeyDataItem, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "items": [i.to_dict() for i in self.items]}


@dataclass(frozen=True, slots=True)
class KeyDataPanel:
    """The company's headline financials, shown beside the opening argument.

    Modelled on the data box a sell-side first page carries in its right-hand
    column: how the market prices the company, what it last reported and what
    it has guided to - the figures a reader wants before reading a word of the
    analysis. It is built once from the Evidence Store, so no body table has to
    repeat these numbers.
    """

    groups: tuple[KeyDataGroup, ...] = ()
    as_of: str = ""

    @property
    def is_empty(self) -> bool:
        return not any(group.items for group in self.groups)

    def to_dict(self) -> dict[str, Any]:
        return {
            "groups": [g.to_dict() for g in self.groups],
            "as_of": self.as_of,
        }


@dataclass(frozen=True, slots=True)
class ChartSpec:
    """A minimal chart description; the renderer decides how to draw it."""

    title: str
    chart_type: str                   # "bar" | "line"
    categories: tuple[str, ...]
    values: tuple[float, ...]
    unit: str = ""
    evidence_ids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "chart_type": self.chart_type,
            "categories": list(self.categories),
            "values": list(self.values),
            "unit": self.unit,
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True, slots=True)
class ReportSectionDraft:
    """One rendered section of the report."""

    section: ReportSection
    title: str
    summary: str = ""
    statements: tuple[Statement, ...] = ()
    paragraphs: tuple[str, ...] = ()
    tables: tuple[MetricTable, ...] = ()
    charts: tuple[ChartSpec, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section.value,
            "title": self.title,
            "summary": self.summary,
            "statements": [s.to_dict() for s in self.statements],
            "paragraphs": list(self.paragraphs),
            "tables": [t.to_dict() for t in self.tables],
            "charts": [c.to_dict() for c in self.charts],
        }


@dataclass(frozen=True, slots=True)
class ReportDraft:
    """The complete structured report, ready for QA then rendering."""

    report_run_id: str
    company: str
    ticker: str | None
    report_date: dt.date
    objective: str
    title: str
    sections: tuple[ReportSectionDraft, ...]
    citations: tuple[Citation, ...] = ()
    key_data: KeyDataPanel | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def section(self, section: ReportSection) -> ReportSectionDraft | None:
        for s in self.sections:
            if s.section is section:
                return s
        return None

    @property
    def all_statements(self) -> tuple[Statement, ...]:
        out: list[Statement] = []
        for s in self.sections:
            out.extend(s.statements)
        return tuple(out)

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_run_id": self.report_run_id,
            "company": self.company,
            "ticker": self.ticker,
            "report_date": self.report_date.isoformat(),
            "objective": self.objective,
            "title": self.title,
            "sections": [s.to_dict() for s in self.sections],
            "citations": [c.to_dict() for c in self.citations],
            "key_data": self.key_data.to_dict() if self.key_data else None,
            "metadata": dict(self.metadata),
        }
