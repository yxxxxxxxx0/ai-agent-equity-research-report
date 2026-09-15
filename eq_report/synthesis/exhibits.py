"""The report's tables and charts, built once from evidence.

Exhibits used to be assembled per section from whatever ``important_metrics``
each agent happened to surface. Because every agent is shown the same retrieved
pool, they all surfaced roughly the same rows, and the report ended up printing
the same twenty metrics in seven near-identical tables whose comparison column
was empty in every row.

They are built here instead, centrally and from the Evidence Store, on three
rules:

* **One exhibit per question.** Three tables answer three different questions -
  what the quarter delivered, where the revenue comes from, and how the company
  compares with its peers - so no number is printed twice.
* **A column earns its place.** Growth, mix and surprise columns are filled from
  the Analytics Engine, which already computed them. Where an analytic does not
  exist the cell is empty and the renderer drops the column, so the reader never
  meets a column of blanks.
* **An exhibit that cannot be filled is not drawn.** Each builder returns
  ``None`` when the evidence behind it is missing, which is what lets the report
  vary with the company rather than always printing the same skeleton.

Every figure is read off an evidence row or an analytics result; nothing here
calculates.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..domain.analytics import AnalyticsBundle, AnalyticsResult
from ..domain.enums import ReportSection
from ..domain.evidence import EvidenceItem
from ..domain.report import ChartSpec, MetricRow, MetricTable
from ..evidence.reader import EvidenceReader
from ..normalisation import canonical_metrics as cm
from ..normalisation.units import format_number

#: The reported-quarter table, in reading order: scale, then profitability,
#: then cash. ``growth`` names the analytic that fills the change column and
#: ``surprise`` the one that fills the versus-consensus column; ``None`` leaves
#: that cell empty for this row.
_FINANCIAL_ROWS: tuple[tuple[str, str, str | None, str | None], ...] = (
    ("Revenue", cm.REVENUE, "revenue_growth_yoy", "revenue_surprise_pct"),
    ("Gross profit", cm.GROSS_PROFIT, None, None),
    ("Gross margin", cm.GROSS_MARGIN, "gross_margin_change_yoy", None),
    ("Operating income", cm.OPERATING_INCOME, None, None),
    ("Operating margin", cm.OPERATING_MARGIN, "operating_margin_change_yoy", None),
    ("Net income", cm.NET_INCOME, "net_income_growth_yoy", None),
    ("Diluted EPS", cm.EPS_DILUTED, "eps_growth_yoy", "eps_surprise_pct"),
    ("Operating cash flow", cm.OPERATING_CASH_FLOW, None, None),
    ("Capex", cm.CAPEX, None, None),
    ("Free cash flow", cm.FREE_CASH_FLOW, "fcf_growth_yoy", None),
)

#: Beyond this many segments a mix table stops being readable; the remainder is
#: collapsed into one "Other segments" row so the total still reconciles.
_MAX_SEGMENT_ROWS = 7


@dataclass(frozen=True, slots=True)
class Exhibit:
    """A table or chart, plus the evidence that has to be cited for it."""

    section: ReportSection
    table: MetricTable | None = None
    chart: ChartSpec | None = None
    evidence_ids: tuple[str, ...] = ()


class ExhibitBuilder:
    """Builds every exhibit in the report, so none of them can collide."""

    def __init__(
        self, reader: EvidenceReader, analytics: AnalyticsBundle, peers: tuple[str, ...],
    ) -> None:
        self.reader = reader
        self.analytics = analytics
        self.peers = peers
        self.period = reader.latest_reported_period()

    def build(self) -> tuple[Exhibit, ...]:
        candidates = (
            self._price_chart(),
            self._financials_table(),
            self._segment_table(),
            self._peer_table(),
        )
        return tuple(e for e in candidates if e is not None)

    # -- Company snapshot: the price path ---------------------------------
    def _price_chart(self) -> Exhibit | None:
        """The one thing a table genuinely cannot show: the path, not the level."""
        history = self.reader.price_history()
        if len(history) < 6:
            return None
        return Exhibit(
            section=ReportSection.COMPANY_SNAPSHOT,
            chart=ChartSpec(
                title="Share price, last 12 months",
                chart_type="line",
                categories=tuple(
                    item.as_of.isoformat() if item.as_of else "" for item in history),
                values=tuple(float(item.value or 0) for item in history),
                unit="USD",
                evidence_ids=tuple(item.evidence_id for item in history),
            ),
            evidence_ids=tuple(item.evidence_id for item in history),
        )

    # -- Financials: what the quarter delivered ---------------------------
    def _financials_table(self) -> Exhibit | None:
        if not self.period:
            return None
        rows: list[MetricRow] = []
        evidence: list[str] = []
        for label, metric, growth_metric, surprise_metric in _FINANCIAL_ROWS:
            item = self.reader.numeric(metric, self.period)
            if item is None or item.value is None:
                continue
            evidence.append(item.evidence_id)
            rows.append(MetricRow(
                label=label,
                cells=(
                    format_number(item.value, item.unit or "", item.currency),
                    _analytic_text(self.analytics.first(growth_metric) if growth_metric else None),
                    _analytic_text(
                        self.analytics.first(surprise_metric) if surprise_metric else None),
                ),
            ))
        if len(rows) < 3:
            return None
        return Exhibit(
            section=ReportSection.FINANCIALS,
            table=MetricTable(
                title=f"Reported results, {self.period}",
                columns=("", self.period, "YoY", "vs consensus"),
                rows=tuple(rows),
            ),
            evidence_ids=tuple(evidence),
        )

    # -- Operating drivers: where the revenue comes from ------------------
    def _segment_table(self) -> Exhibit | None:
        """Segment revenue with its mix and growth.

        This replaces the segment bar chart the report used to draw beside it:
        the chart showed only the levels, which the first column already gives,
        while the mix and growth columns are the part a reader cannot eyeball.
        """
        if not self.period:
            return None
        segment_rows = self.reader.segment_rows(self.period)
        if len(segment_rows) < 2:
            return None

        shares = _by_segment(self.analytics, "segment_contribution_pct")
        growth = _by_segment(self.analytics, "segment_growth_yoy")

        shown = segment_rows[:_MAX_SEGMENT_ROWS]
        remainder = segment_rows[_MAX_SEGMENT_ROWS:]
        rows = [_segment_row(item, shares, growth) for item in shown]
        if remainder:
            rows.append(_other_segments_row(remainder, shares))

        return Exhibit(
            section=ReportSection.OPERATING_DRIVERS,
            table=MetricTable(
                title=f"Revenue by segment, {self.period}",
                columns=("Segment", "Revenue", "% of total", "YoY"),
                rows=tuple(rows),
            ),
            evidence_ids=tuple(item.evidence_id for item in segment_rows),
        )

    # -- Competitive landscape: how it compares ---------------------------
    def _peer_table(self) -> Exhibit | None:
        """The subject company against its peers on the two measures held for both.

        Peer evidence is sparse by nature - a peer is only ever fetched for a
        couple of metrics - so a column survives only if at least one peer has
        it, and the table is dropped entirely below two comparable companies.
        """
        if not self.peers:
            return None
        metrics = (
            ("Forward P/E", cm.FORWARD_PE),
            ("Revenue growth YoY", cm.REVENUE_GROWTH_YOY_REPORTED),
        )
        rows: list[MetricRow] = []
        evidence: list[str] = []

        own_cells, own_evidence = self._own_peer_cells(metrics)
        if any(own_cells):
            rows.append(MetricRow(
                label=self.reader.ticker or self.reader.company,
                cells=own_cells, emphasis=True))
            evidence.extend(own_evidence)

        for peer in self.peers:
            cells: list[str] = []
            found = False
            for _, metric in metrics:
                item = self.reader.peer_value(peer, metric)
                if item is None or item.value is None:
                    cells.append("")
                    continue
                found = True
                evidence.append(item.evidence_id)
                cells.append(format_number(item.value, item.unit or "", item.currency))
            if found:
                rows.append(MetricRow(label=peer, cells=tuple(cells)))

        if len(rows) < 3:
            return None

        average = self._peer_average_row()
        if average is not None:
            rows.append(average)

        return Exhibit(
            section=ReportSection.COMPETITIVE_LANDSCAPE,
            table=MetricTable(
                title="Peer comparison",
                columns=("", *(label for label, _ in metrics)),
                rows=tuple(rows),
            ),
            evidence_ids=tuple(evidence),
        )

    def _own_peer_cells(
        self, metrics: tuple[tuple[str, str], ...]
    ) -> tuple[tuple[str, ...], list[str]]:
        """The subject company's own row of the peer table.

        Its revenue growth is an Analytics Engine result rather than a reported
        observation (peers supply theirs pre-computed), so the two sources are
        reconciled here rather than leaving the company's own row blank.
        """
        cells: list[str] = []
        evidence: list[str] = []
        for _, metric in metrics:
            item = self.reader.numeric(metric)
            if item is not None and item.value is not None:
                evidence.append(item.evidence_id)
                cells.append(format_number(item.value, item.unit or "", item.currency))
                continue
            if metric == cm.REVENUE_GROWTH_YOY_REPORTED:
                own_growth = self.analytics.first("revenue_growth_yoy")
                cells.append(f"{own_growth.value:.1f}%" if own_growth else "")
                continue
            cells.append("")
        return tuple(cells), evidence

    def _peer_average_row(self) -> MetricRow | None:
        """The peer averages the Analytics Engine already computed for its
        premium and growth-gap results, rather than a second average of our own."""
        premium = self.analytics.first("valuation_vs_peers")
        gap = self.analytics.first("peer_growth_gap")
        multiple = premium.metadata.get("peer_average_forward_pe") if premium else None
        growth = gap.metadata.get("peer_average_growth_pct") if gap else None
        if multiple is None and growth is None:
            return None
        return MetricRow(
            label="Peer average",
            cells=(
                f"{float(multiple):.1f}x" if multiple is not None else "",
                f"{float(growth):.1f}%" if growth is not None else "",
            ),
        )


# -- helpers ----------------------------------------------------------------
def _analytic_text(result: AnalyticsResult | None) -> str:
    """An analytics value as a signed change, or empty so the cell stays blank."""
    if result is None:
        return ""
    if result.unit == "pp":
        return f"{result.value:+.1f}pp"
    if result.unit == "pct":
        return f"{result.value:+.1f}%"
    if result.unit == "x":
        return f"{result.value:.1f}x"
    return f"{result.value:,.2f}"


def _by_segment(analytics: AnalyticsBundle, metric: str) -> dict[str, AnalyticsResult]:
    return {
        str(result.metadata.get("segment_name", "")): result
        for result in analytics.by_metric(metric)
        if result.metadata.get("segment_name")
    }


def _segment_row(
    item: EvidenceItem,
    shares: dict[str, AnalyticsResult],
    growth: dict[str, AnalyticsResult],
) -> MetricRow:
    name = str(item.metadata.get("segment_name", "Unnamed segment"))
    share = shares.get(name)
    return MetricRow(
        label=name,
        cells=(
            format_number(item.value or 0, item.unit or "", item.currency),
            f"{share.value:.1f}%" if share else "",
            _analytic_text(growth.get(name)),
        ),
    )


def _other_segments_row(
    remainder: tuple[EvidenceItem, ...], shares: dict[str, AnalyticsResult]
) -> MetricRow:
    """Everything below the cut, summed, so the mix column still adds up."""
    first = remainder[0]
    total = sum(float(item.value or 0) for item in remainder)
    share_total = sum(
        shares[str(item.metadata.get("segment_name", ""))].value
        for item in remainder
        if str(item.metadata.get("segment_name", "")) in shares
    )
    return MetricRow(
        label=f"Other segments ({len(remainder)})",
        cells=(
            format_number(total, first.unit or "", first.currency),
            f"{share_total:.1f}%" if share_total else "",
            "",
        ),
    )
