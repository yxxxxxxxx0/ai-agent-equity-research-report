"""Stage 5 - the Evidence Store.

This is the architectural boundary of the system: everything upstream writes
here, everything downstream reads only from here. The Analytics Engine and the
segment agents are constructed with an EvidenceStore and have no provider or
network access at all, which is what guarantees every report claim is traceable.

SQLite via the standard library, deliberately: one file, inspectable with any
SQL client, no service to run.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from ..domain.analytics import AnalyticsResult
from ..domain.enums import Confidence, EvidenceCategory, EvidenceStatus, FactType, SourceType
from ..domain.evidence import EvidenceItem, FiscalPeriod
from ..errors import EvidenceStoreError
from ..logging_setup import get_logger, log_event

logger = get_logger("evidence.store")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS evidence (
    evidence_id     TEXT PRIMARY KEY,
    report_run_id   TEXT NOT NULL,
    company         TEXT NOT NULL,
    ticker          TEXT,
    category        TEXT NOT NULL,
    metric          TEXT,
    value           REAL,
    unit            TEXT,
    currency        TEXT,
    period_label    TEXT,
    fiscal_year     INTEGER,
    fiscal_quarter  INTEGER,
    period_end      TEXT,
    is_annual       INTEGER NOT NULL DEFAULT 0,
    as_of           TEXT,
    source_id       TEXT NOT NULL,
    source_name     TEXT NOT NULL,
    source_type     TEXT NOT NULL,
    source_url      TEXT,
    retrieval_provider TEXT,
    retrieval_url   TEXT,
    original_source_name TEXT,
    original_source_url TEXT,
    original_publication_date TEXT,
    basis           TEXT,
    frequency       TEXT,
    period_start    TEXT,
    fact_type       TEXT NOT NULL DEFAULT 'reported_fact',
    status          TEXT NOT NULL DEFAULT 'unverified',
    is_canonical    INTEGER NOT NULL DEFAULT 0,
    reconciliation_key TEXT,
    alternate_evidence_ids TEXT NOT NULL DEFAULT '[]',
    validation_messages TEXT NOT NULL DEFAULT '[]',
    claim_text      TEXT,
    document_title  TEXT,
    published_at    TEXT,
    retrieved_at    TEXT NOT NULL,
    confidence      TEXT NOT NULL,
    raw_metric      TEXT,
    raw_value       TEXT,
    is_mock         INTEGER NOT NULL DEFAULT 0,
    metadata        TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_evidence_run     ON evidence(report_run_id);
CREATE INDEX IF NOT EXISTS idx_evidence_ticker  ON evidence(report_run_id, ticker);
CREATE INDEX IF NOT EXISTS idx_evidence_metric  ON evidence(report_run_id, metric);
CREATE INDEX IF NOT EXISTS idx_evidence_cat     ON evidence(report_run_id, category);
CREATE INDEX IF NOT EXISTS idx_evidence_period  ON evidence(report_run_id, period_label);
CREATE INDEX IF NOT EXISTS idx_evidence_source  ON evidence(report_run_id, source_type);

CREATE TABLE IF NOT EXISTS analytics (
    analytics_id        TEXT PRIMARY KEY,
    report_run_id       TEXT NOT NULL,
    metric              TEXT NOT NULL,
    value               REAL NOT NULL,
    unit                TEXT NOT NULL,
    formula             TEXT NOT NULL,
    input_evidence_ids  TEXT NOT NULL,
    inputs              TEXT NOT NULL DEFAULT '{}',
    label               TEXT,
    period              TEXT,
    comparison_period   TEXT,
    confidence          TEXT NOT NULL,
    metadata            TEXT NOT NULL DEFAULT '{}',
    generated_at        TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analytics_run    ON analytics(report_run_id);
CREATE INDEX IF NOT EXISTS idx_analytics_metric ON analytics(report_run_id, metric);
"""


@dataclass(frozen=True, slots=True)
class EvidenceQuery:
    """Declarative query over the Evidence Store.

    Every field is optional; set fields are ANDed together. This is the whole
    read surface the downstream stages use.
    """

    report_run_id: str | None = None
    company: str | None = None
    ticker: str | None = None
    category: EvidenceCategory | None = None
    categories: Sequence[EvidenceCategory] | None = None
    metric: str | None = None
    metrics: Sequence[str] | None = None
    period_label: str | None = None
    fiscal_year: int | None = None
    fiscal_quarter: int | None = None
    source_type: SourceType | None = None
    source_types: Sequence[SourceType] | None = None
    as_of_from: str | None = None
    as_of_to: str | None = None
    has_value: bool | None = None
    status: EvidenceStatus | None = None
    canonical_only: bool | None = None
    limit: int | None = None
    order_by: str = "as_of DESC, period_end DESC, evidence_id ASC"

    def build(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def add(clause: str, *values: Any) -> None:
            clauses.append(clause)
            params.extend(values)

        if self.report_run_id:
            add("report_run_id = ?", self.report_run_id)
        if self.company:
            add("company = ?", self.company)
        if self.ticker:
            add("ticker = ?", self.ticker.upper())
        if self.category:
            add("category = ?", self.category.value)
        if self.categories:
            placeholders = ", ".join("?" for _ in self.categories)
            add(f"category IN ({placeholders})", *[c.value for c in self.categories])
        if self.metric:
            add("metric = ?", self.metric)
        if self.metrics:
            placeholders = ", ".join("?" for _ in self.metrics)
            add(f"metric IN ({placeholders})", *self.metrics)
        if self.period_label:
            add("period_label = ?", self.period_label)
        if self.fiscal_year is not None:
            add("fiscal_year = ?", self.fiscal_year)
        if self.fiscal_quarter is not None:
            add("fiscal_quarter = ?", self.fiscal_quarter)
        if self.source_type:
            add("source_type = ?", self.source_type.value)
        if self.source_types:
            placeholders = ", ".join("?" for _ in self.source_types)
            add(f"source_type IN ({placeholders})", *[s.value for s in self.source_types])
        if self.as_of_from:
            add("as_of >= ?", self.as_of_from)
        if self.as_of_to:
            add("as_of <= ?", self.as_of_to)
        if self.has_value is True:
            clauses.append("value IS NOT NULL")
        elif self.has_value is False:
            clauses.append("value IS NULL")
        if self.status:
            add("status = ?", self.status.value)
        if self.canonical_only is True:
            clauses.append("is_canonical = 1")

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM evidence{where} ORDER BY {self.order_by}"
        if self.limit:
            sql += f" LIMIT {int(self.limit)}"
        return sql, params


class EvidenceStore:
    """SQLite-backed store for evidence and analytics."""

    def __init__(self, database_path: Path | str) -> None:
        self.database_path = Path(database_path)
        if str(self.database_path) != ":memory:":
            self.database_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self._conn = sqlite3.connect(str(self.database_path))
        except sqlite3.Error as exc:
            raise EvidenceStoreError(f"could not open {self.database_path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        self._conn.executescript(_SCHEMA)
        self._migrate_schema()
        self._conn.commit()

    def _migrate_schema(self) -> None:
        """Add canonical-layer columns to databases created by older releases."""
        existing = {row[1] for row in self._conn.execute("PRAGMA table_info(evidence)")}
        additions = {
            "retrieval_provider": "TEXT", "retrieval_url": "TEXT",
            "original_source_name": "TEXT", "original_source_url": "TEXT",
            "original_publication_date": "TEXT", "basis": "TEXT",
            "frequency": "TEXT", "period_start": "TEXT",
            "fact_type": "TEXT NOT NULL DEFAULT 'reported_fact'",
            "status": "TEXT NOT NULL DEFAULT 'unverified'",
            "is_canonical": "INTEGER NOT NULL DEFAULT 0",
            "reconciliation_key": "TEXT",
            "alternate_evidence_ids": "TEXT NOT NULL DEFAULT '[]'",
            "validation_messages": "TEXT NOT NULL DEFAULT '[]'",
        }
        for name, definition in additions.items():
            if name not in existing:
                self._conn.execute(f"ALTER TABLE evidence ADD COLUMN {name} {definition}")

    # -- lifecycle -------------------------------------------------------
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "EvidenceStore":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -- writes ----------------------------------------------------------
    def save(self, items: Iterable[EvidenceItem]) -> int:
        """Insert evidence items, ignoring exact-id duplicates.

        Returns the number of rows actually written.
        """
        rows = [self._to_row(item) for item in items]
        if not rows:
            return 0
        columns = list(rows[0].keys())
        placeholders = ", ".join(":" + c for c in columns)
        sql = (
            f"INSERT OR IGNORE INTO evidence ({', '.join(columns)}) VALUES ({placeholders})"
        )
        try:
            cursor = self._conn.executemany(sql, rows)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise EvidenceStoreError(f"evidence insert failed: {exc}") from exc
        written = cursor.rowcount if cursor.rowcount is not None else 0
        log_event(logger, logging.INFO, "evidence saved",
                  submitted=len(rows), written=written)
        return written

    def save_analytics(self, results: Iterable[AnalyticsResult]) -> int:
        rows = [
            {
                "analytics_id": r.analytics_id,
                "report_run_id": r.report_run_id,
                "metric": r.metric,
                "value": r.value,
                "unit": r.unit,
                "formula": r.formula,
                "input_evidence_ids": json.dumps(list(r.input_evidence_ids)),
                "inputs": json.dumps(r.inputs),
                "label": r.label,
                "period": r.period,
                "comparison_period": r.comparison_period,
                "confidence": r.confidence.value,
                "metadata": json.dumps(r.metadata, default=str),
                "generated_at": r.generated_at.isoformat(),
            }
            for r in results
        ]
        if not rows:
            return 0
        columns = list(rows[0].keys())
        sql = (
            f"INSERT OR REPLACE INTO analytics ({', '.join(columns)}) "
            f"VALUES ({', '.join(':' + c for c in columns)})"
        )
        try:
            self._conn.executemany(sql, rows)
            self._conn.commit()
        except sqlite3.Error as exc:
            self._conn.rollback()
            raise EvidenceStoreError(f"analytics insert failed: {exc}") from exc
        log_event(logger, logging.INFO, "analytics saved", count=len(rows))
        return len(rows)

    # -- reads -----------------------------------------------------------
    def query(self, query: EvidenceQuery) -> tuple[EvidenceItem, ...]:
        sql, params = query.build()
        try:
            rows = self._conn.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise EvidenceStoreError(f"evidence query failed: {exc}") from exc
        return tuple(self._from_row(row) for row in rows)

    def get(self, evidence_id: str) -> EvidenceItem | None:
        row = self._conn.execute(
            "SELECT * FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return self._from_row(row) if row else None

    def exists(self, evidence_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM evidence WHERE evidence_id = ?", (evidence_id,)
        ).fetchone()
        return row is not None

    def count(self, report_run_id: str | None = None) -> int:
        if report_run_id:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM evidence WHERE report_run_id = ?", (report_run_id,)
            ).fetchone()
        else:
            row = self._conn.execute("SELECT COUNT(*) FROM evidence").fetchone()
        return int(row[0])

    def periods(self, report_run_id: str, ticker: str | None = None) -> tuple[str, ...]:
        """Distinct fiscal periods present for a run, most recent first."""
        sql = (
            "SELECT DISTINCT period_label, period_end FROM evidence "
            "WHERE report_run_id = ? AND period_label IS NOT NULL"
        )
        params: list[Any] = [report_run_id]
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        sql += " ORDER BY period_end DESC"
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(row["period_label"] for row in rows)

    def latest_reported_period(self, report_run_id: str, ticker: str | None = None) -> str | None:
        """The most recent period that has an actual reported revenue figure.

        Guidance and consensus rows carry future periods, so "latest period" is
        defined by reported fundamentals, not by the maximum date in the table.
        """
        # Some providers supply a valid fiscal year/quarter label without a
        # calendar period-end date.  Those rows are still reported results and
        # must remain visible to analytics and the LLM agents.  Requiring
        # ``period_end`` used to make a complete quarter disappear downstream,
        # leaving the narrative agents with little besides valuation snapshots.
        sql = (
            "SELECT period_label FROM evidence "
            "WHERE report_run_id = ? AND metric = 'revenue' AND category = 'fundamental' "
            "AND period_label IS NOT NULL "
            "AND (period_end IS NOT NULL OR fiscal_year IS NOT NULL)"
        )
        params: list[Any] = [report_run_id]
        if ticker:
            sql += " AND ticker = ?"
            params.append(ticker.upper())
        sql += (
            " ORDER BY fiscal_year DESC, COALESCE(fiscal_quarter, 5) DESC, "
            "period_end DESC LIMIT 1"
        )
        row = self._conn.execute(sql, params).fetchone()
        return row["period_label"] if row else None

    def analytics(self, report_run_id: str) -> tuple[AnalyticsResult, ...]:
        rows = self._conn.execute(
            "SELECT * FROM analytics WHERE report_run_id = ? ORDER BY metric", (report_run_id,)
        ).fetchall()
        return tuple(self._analytics_from_row(row) for row in rows)

    def analytics_exists(self, analytics_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM analytics WHERE analytics_id = ?", (analytics_id,)
        ).fetchone()
        return row is not None

    def source_summary(self, report_run_id: str) -> list[dict[str, Any]]:
        """Per-source evidence counts, used in the run manifest."""
        rows = self._conn.execute(
            "SELECT source_name, source_type, is_mock, COUNT(*) AS n FROM evidence "
            "WHERE report_run_id = ? GROUP BY source_name, source_type, is_mock "
            "ORDER BY n DESC",
            (report_run_id,),
        ).fetchall()
        return [
            {
                "source_name": row["source_name"],
                "source_type": row["source_type"],
                "is_mock": bool(row["is_mock"]),
                "evidence_count": int(row["n"]),
            }
            for row in rows
        ]

    # -- row mapping -----------------------------------------------------
    @staticmethod
    def _to_row(item: EvidenceItem) -> dict[str, Any]:
        period = item.period
        return {
            "evidence_id": item.evidence_id,
            "report_run_id": item.report_run_id,
            "company": item.company,
            "ticker": item.ticker,
            "category": item.category.value,
            "metric": item.metric,
            "value": item.value,
            "unit": item.unit,
            "currency": item.currency,
            "period_label": period.label if period else None,
            "fiscal_year": period.fiscal_year if period else None,
            "fiscal_quarter": period.fiscal_quarter if period else None,
            "period_end": (
                period.period_end.isoformat() if period and period.period_end else None
            ),
            "is_annual": int(bool(period.is_annual)) if period else 0,
            "as_of": item.as_of.isoformat() if item.as_of else None,
            "source_id": item.source_id,
            "source_name": item.source_name,
            "source_type": item.source_type.value,
            "source_url": item.source_url,
            "retrieval_provider": item.retrieval_provider,
            "retrieval_url": item.retrieval_url,
            "original_source_name": item.original_source_name,
            "original_source_url": item.original_source_url,
            "original_publication_date": item.original_publication_date.isoformat() if item.original_publication_date else None,
            "basis": item.basis,
            "frequency": item.frequency,
            "period_start": item.period_start.isoformat() if item.period_start else None,
            "fact_type": item.fact_type.value,
            "status": item.status.value,
            "is_canonical": int(item.is_canonical),
            "reconciliation_key": item.reconciliation_key,
            "alternate_evidence_ids": json.dumps(list(item.alternate_evidence_ids)),
            "validation_messages": json.dumps(list(item.validation_messages)),
            "claim_text": item.claim_text,
            "document_title": item.document_title,
            "published_at": item.published_at.isoformat() if item.published_at else None,
            "retrieved_at": item.retrieved_at.isoformat(),
            "confidence": item.confidence.value,
            "raw_metric": item.raw_metric,
            "raw_value": None if item.raw_value is None else str(item.raw_value),
            "is_mock": int(item.is_mock),
            "metadata": json.dumps(item.metadata, default=str),
        }

    @staticmethod
    def _from_row(row: sqlite3.Row) -> EvidenceItem:
        import datetime as dt

        period = None
        if row["period_label"]:
            period = FiscalPeriod(
                label=row["period_label"],
                fiscal_year=row["fiscal_year"],
                fiscal_quarter=row["fiscal_quarter"],
                period_end=(
                    dt.date.fromisoformat(row["period_end"]) if row["period_end"] else None
                ),
                is_annual=bool(row["is_annual"]),
            )
        return EvidenceItem(
            evidence_id=row["evidence_id"],
            report_run_id=row["report_run_id"],
            company=row["company"],
            ticker=row["ticker"],
            category=EvidenceCategory(row["category"]),
            metric=row["metric"],
            value=row["value"],
            unit=row["unit"],
            currency=row["currency"],
            period=period,
            as_of=dt.date.fromisoformat(row["as_of"]) if row["as_of"] else None,
            source_id=row["source_id"],
            source_name=row["source_name"],
            source_type=SourceType(row["source_type"]),
            source_url=row["source_url"],
            retrieval_provider=row["retrieval_provider"],
            retrieval_url=row["retrieval_url"],
            original_source_name=row["original_source_name"],
            original_source_url=row["original_source_url"],
            original_publication_date=(dt.date.fromisoformat(row["original_publication_date"]) if row["original_publication_date"] else None),
            basis=row["basis"],
            frequency=row["frequency"],
            period_start=dt.date.fromisoformat(row["period_start"]) if row["period_start"] else None,
            fact_type=FactType(row["fact_type"]),
            status=EvidenceStatus(row["status"]),
            is_canonical=bool(row["is_canonical"]),
            reconciliation_key=row["reconciliation_key"],
            alternate_evidence_ids=tuple(json.loads(row["alternate_evidence_ids"] or "[]")),
            validation_messages=tuple(json.loads(row["validation_messages"] or "[]")),
            claim_text=row["claim_text"],
            document_title=row["document_title"],
            published_at=(
                dt.date.fromisoformat(row["published_at"]) if row["published_at"] else None
            ),
            retrieved_at=dt.datetime.fromisoformat(row["retrieved_at"]),
            confidence=Confidence(row["confidence"]),
            raw_metric=row["raw_metric"],
            raw_value=row["raw_value"],
            is_mock=bool(row["is_mock"]),
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _analytics_from_row(row: sqlite3.Row) -> AnalyticsResult:
        import datetime as dt

        return AnalyticsResult(
            analytics_id=row["analytics_id"],
            report_run_id=row["report_run_id"],
            metric=row["metric"],
            value=row["value"],
            unit=row["unit"],
            formula=row["formula"],
            input_evidence_ids=tuple(json.loads(row["input_evidence_ids"])),
            inputs=json.loads(row["inputs"] or "{}"),
            label=row["label"] or "",
            period=row["period"],
            comparison_period=row["comparison_period"],
            confidence=Confidence(row["confidence"]),
            metadata=json.loads(row["metadata"] or "{}"),
            generated_at=dt.datetime.fromisoformat(row["generated_at"]),
        )
