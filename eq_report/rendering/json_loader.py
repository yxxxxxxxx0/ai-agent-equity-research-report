"""Rehydrate persisted report JSON for deterministic re-rendering."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

from ..domain.enums import ClaimType, Confidence, ReportSection, Severity
from ..domain.qa import QAFinding, QAResult
from ..domain.report import (
    ChartSpec,
    Citation,
    KeyDataGroup,
    KeyDataItem,
    KeyDataPanel,
    MetricRow,
    MetricTable,
    ReportDraft,
    ReportSectionDraft,
    Statement,
)


def load_report_json(path: Path | str) -> tuple[ReportDraft, QAResult]:
    """Load the JSON written by ``write_report_json`` back into domain objects."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    sections = tuple(ReportSectionDraft(
        section=ReportSection(row["section"]), title=row["title"],
        summary=row.get("summary", ""),
        statements=tuple(Statement(
            text=item["text"], claim_type=ClaimType(item["claim_type"]),
            evidence_ids=tuple(item.get("evidence_ids", [])),
            analytics_ids=tuple(item.get("analytics_ids", [])),
            citation_refs=tuple(item.get("citation_refs", [])),
            confidence=Confidence(item.get("confidence", "medium")),
        ) for item in row.get("statements", [])),
        paragraphs=tuple(row.get("paragraphs", [])),
        tables=tuple(MetricTable(
            title=table["title"], columns=tuple(table.get("columns", [])),
            rows=tuple(MetricRow(
                label=item["label"], cells=tuple(item.get("cells", [])),
                emphasis=bool(item.get("emphasis", False)),
            ) for item in table.get("rows", [])), note=table.get("note", ""),
        ) for table in row.get("tables", [])),
        charts=tuple(ChartSpec(
            title=chart["title"], chart_type=chart["chart_type"],
            categories=tuple(chart.get("categories", [])),
            values=tuple(chart.get("values", [])), unit=chart.get("unit", ""),
            evidence_ids=tuple(chart.get("evidence_ids", [])),
        ) for chart in row.get("charts", [])),
    ) for row in payload.get("sections", []))
    panel_row = payload.get("key_data")
    panel = None if not panel_row else KeyDataPanel(
        groups=tuple(KeyDataGroup(
            title=group["title"],
            items=tuple(KeyDataItem(label=item["label"], value=item["value"])
                        for item in group.get("items", [])),
        ) for group in panel_row.get("groups", [])), as_of=panel_row.get("as_of", ""),
    )
    draft = ReportDraft(
        report_run_id=payload["report_run_id"], company=payload["company"],
        ticker=payload.get("ticker"), report_date=dt.date.fromisoformat(payload["report_date"]),
        objective=payload.get("objective", ""), title=payload["title"], sections=sections,
        citations=tuple(Citation(**row) for row in payload.get("citations", [])),
        key_data=panel, metadata=dict(payload.get("metadata", {})),
    )
    qa_row = payload.get("qa", {})
    qa = QAResult(
        findings=tuple(QAFinding(
            check=row["check"], severity=Severity(row["severity"]),
            message=row["message"], section=row.get("section"),
            subject=row.get("subject"), details=dict(row.get("details", {})),
        ) for row in qa_row.get("findings", [])),
        checks_run=tuple(qa_row.get("checks_run", [])),
    )
    return draft, qa
