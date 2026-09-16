"""Read-only convenience view over the Evidence Store.

The Analytics Engine and the segment agents are handed an EvidenceReader rather
than the store itself. It scopes every read to one report run and one primary
ticker, and offers the handful of access patterns the downstream stages actually
need - so no stage has to write SQL, and none of them can fetch anything that
did not come through normalisation.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Sequence

from ..domain.enums import EvidenceCategory, EvidenceStatus, SourceType
from ..domain.evidence import EvidenceItem
from .store import EvidenceQuery, EvidenceStore


@dataclass(frozen=True, slots=True)
class EvidenceReader:
    """Run-scoped, read-only access to evidence."""

    store: EvidenceStore
    report_run_id: str
    ticker: str | None
    company: str

    # -- generic ---------------------------------------------------------
    def query(self, **kwargs: object) -> tuple[EvidenceItem, ...]:
        """Run an EvidenceQuery scoped to this report run."""
        return self.store.query(
            EvidenceQuery(report_run_id=self.report_run_id, **kwargs)  # type: ignore[arg-type]
        )

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self.store.get(evidence_id)

    # -- numeric access --------------------------------------------------
    def numeric(
        self,
        metric: str,
        period_label: str | None = None,
        *,
        ticker: str | None = None,
        exclude_series: bool = True,
    ) -> EvidenceItem | None:
        """Single most recent numeric datapoint for a metric (and period).

        ``exclude_series=True`` skips historical series points, so asking for
        ``forward_pe`` returns the current multiple rather than a past one.
        """
        items = self.numeric_all(
            metric, period_label, ticker=ticker, exclude_series=exclude_series)
        return items[0] if items else None

    def numeric_all(
        self,
        metric: str,
        period_label: str | None = None,
        *,
        ticker: str | None = None,
        exclude_series: bool = True,
    ) -> tuple[EvidenceItem, ...]:
        items = self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            ticker=ticker or self.ticker,
            metric=metric,
            period_label=period_label,
            has_value=True,
            status=EvidenceStatus.VALIDATED,
            canonical_only=True,
        ))
        if exclude_series:
            items = tuple(i for i in items if not i.metadata.get("series"))
        return items

    def series(self, metric: str, series_name: str) -> tuple[EvidenceItem, ...]:
        """A labelled historical series, oldest first."""
        items = self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            ticker=self.ticker,
            metric=metric,
            has_value=True,
            status=EvidenceStatus.VALIDATED,
            canonical_only=True,
        ))
        selected = [i for i in items if i.metadata.get("series") == series_name]
        return tuple(sorted(selected, key=lambda i: i.as_of or dt.date.min))

    def price_history(self) -> tuple[EvidenceItem, ...]:
        """Historical closing prices, oldest first."""
        return self.series("price_history_point", "price_history")

    def value(self, metric: str, period_label: str | None = None) -> float | None:
        item = self.numeric(metric, period_label)
        return item.value if item else None

    # -- structured groups -----------------------------------------------
    def segment_rows(self, period_label: str) -> tuple[EvidenceItem, ...]:
        """Segment revenue rows for a period, largest first."""
        items = self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            ticker=self.ticker,
            metric="segment_revenue",
            period_label=period_label,
            has_value=True,
            status=EvidenceStatus.VALIDATED,
            canonical_only=True,
        ))
        return tuple(sorted(items, key=lambda i: -(i.value or 0)))

    def segment_names(self) -> tuple[str, ...]:
        items = self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            ticker=self.ticker,
            metric="segment_revenue",
        ))
        names: list[str] = []
        for item in items:
            name = item.metadata.get("segment_name")
            if name and name not in names:
                names.append(str(name))
        return tuple(names)

    def kpi_rows(self, period_label: str | None = None) -> tuple[EvidenceItem, ...]:
        return self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            ticker=self.ticker,
            metric="kpi",
            period_label=period_label,
            has_value=True,
            status=EvidenceStatus.VALIDATED,
            canonical_only=True,
        ))

    def kpi(self, kpi_name: str, period_label: str | None = None) -> EvidenceItem | None:
        for item in self.kpi_rows(period_label):
            if str(item.metadata.get("kpi_name", "")).lower() == kpi_name.lower():
                return item
        return None

    def guidance(self) -> tuple[EvidenceItem, ...]:
        return self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            ticker=self.ticker,
            category=EvidenceCategory.GUIDANCE,
            has_value=True,
            status=EvidenceStatus.VALIDATED,
            canonical_only=True,
        ))

    def peer_value(self, peer_ticker: str, metric: str) -> EvidenceItem | None:
        return self.numeric(metric, ticker=peer_ticker)

    # -- document access -------------------------------------------------
    def documents(
        self,
        source_types: Sequence[SourceType] | None = None,
        *,
        limit: int | None = None,
    ) -> tuple[EvidenceItem, ...]:
        """Document passages, most recently published first."""
        items = self.store.query(EvidenceQuery(
            report_run_id=self.report_run_id,
            category=EvidenceCategory.DOCUMENT,
            source_types=list(source_types) if source_types else None,
            limit=limit,
            order_by="published_at DESC, evidence_id ASC",
        ))
        return items

    def documents_matching(
        self,
        keywords: Sequence[str],
        source_types: Sequence[SourceType] | None = None,
        *,
        limit: int = 5,
    ) -> tuple[EvidenceItem, ...]:
        """Passages whose text contains any of the keywords.

        Deliberately a plain keyword filter: for the prototype, transparent and
        debuggable beats clever retrieval. This is the natural place to swap in
        embeddings later without changing any agent.
        """
        needles = [k.lower() for k in keywords if k]
        matches: list[EvidenceItem] = []
        for item in self.documents(source_types):
            text = (item.claim_text or "").lower()
            if any(needle in text for needle in needles):
                matches.append(item)
            if len(matches) >= limit:
                break
        return tuple(matches)

    # -- periods ---------------------------------------------------------
    def latest_reported_period(self) -> str | None:
        return self.store.latest_reported_period(self.report_run_id, self.ticker)

    def periods(self) -> tuple[str, ...]:
        return self.store.periods(self.report_run_id, self.ticker)
