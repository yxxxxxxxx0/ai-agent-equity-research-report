"""ArcticDB-backed fundamentals provider.

Combines several ``bbg-*`` libraries, all ticker-keyed with a plain ``date``
column, into one branch:

- ``bbg-income-statement`` - carries its own ``FISCAL_YEAR_PERIOD`` label
  (e.g. ``"2001 Q4"``), so its rows need no period inference at all.
- ``bbg-balance-sheet`` / ``bbg-cash-flow`` / ``bbg-financial-metrics`` - no
  period label of their own; each row is matched to the income statement's
  period for the same report date, falling back to a label derived from the
  company's fiscal year-end month (see :mod:`fiscal_period`) when no exact
  date match exists.
- ``bbg-eps-estimates`` / ``bbg-revenue-estimates`` - consensus estimates,
  each already carrying the fiscal period they forecast.
- ``bbg-price-targets`` - a daily series; only the most recent row is used,
  as a point-in-time observation rather than a per-period one.
- ``bbg-segment-revenue`` - named segment revenue (e.g. NVIDIA's GPU/PSB/MCP
  breakdown), period-matched the same way as the balance sheet/cash flow.
- ``ibes.recommendations`` - individual named-firm analyst ratings (firm,
  analyst, Buy/Hold/Sell text, announcement date) - not a single consensus
  figure but the dispersion of what other research firms actually rate the
  stock, each kept as its own ``consensus_rating`` claim rather than
  collapsed into one number.

Field names map closely onto the canonical metric vocabulary; anything not in
the alias table still comes through as ``unmapped.*`` rather than being
dropped.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import math
from typing import Any

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawObservation, SourceRef
from ...domain.plan import ResearchPlan
from ...llm.usage import UsageTracker
from ..base import FundamentalsProvider
from .client import ArcticDBConnection
from .fiscal_period import FiscalCalendarResolver, quarter_label

_PERIOD_LIBRARY = "bbg-income-statement"
_PERIOD_COLUMN = "FISCAL_YEAR_PERIOD"
#: (library, non-metric columns beyond the shared set).
_STATEMENT_LIBRARIES: tuple[tuple[str, frozenset[str]], ...] = (
    (_PERIOD_LIBRARY, frozenset()),
    ("bbg-balance-sheet", frozenset()),
    ("bbg-cash-flow", frozenset()),
    ("bbg-financial-metrics", frozenset()),
)
_ESTIMATE_LIBRARIES: tuple[tuple[str, str | None], ...] = (
    # (library, its own period column, if any)
    ("bbg-eps-estimates", "fiscal_year_period"),
    ("bbg-revenue-estimates", "period"),
)
_NON_METRIC_COLUMNS = {"date", "ticker", "symbol", "currency", "period",
                       "fiscal_year_period", "fiscal_period_override",
                       "best_analyst_name", "segment", "segment level"}
#: Most recent N reported quarters, so YoY/QoQ comparisons have something to
#: compare against without pulling decades of history per statement.
_QUARTERS_KEPT = 8
_ESTIMATE_ROWS_KEPT = 40
_SEGMENT_ROWS_KEPT = _QUARTERS_KEPT * 6  # a handful of segments per period

_RATINGS_LIBRARY = "ibes.recommendations"
_RATINGS_KEPT = 20  # most recent individual analyst ratings


class ArcticDBFundamentalsProvider(FundamentalsProvider):
    name = "arcticdb_fundamentals"
    is_mock = False

    def __init__(
        self,
        settings,
        *,
        connection: ArcticDBConnection | None = None,
        fiscal_resolver: FiscalCalendarResolver | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        super().__init__(settings)
        self._connection = connection or ArcticDBConnection(
            settings.credentials.arcticdb_uri or "")
        self._fiscal_resolver = fiscal_resolver or FiscalCalendarResolver(
            settings.model, connection=self._connection, tracker=tracker)

    def is_available(self) -> bool:
        return bool(self.settings.credentials.arcticdb_uri)

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        if not plan.ticker:
            return self.skipped("No ticker resolved; ArcticDB fundamentals need one.")

        fiscal_year_end_month = await self._fiscal_resolver.fiscal_year_end_month(
            plan.ticker, plan.company)
        period_map = self._period_map(plan.ticker)

        jobs = (
            *(("statement", library, non_metric) for library, non_metric in _STATEMENT_LIBRARIES),
            *(("estimate", library, period_col) for library, period_col in _ESTIMATE_LIBRARIES),
            ("price_target", "bbg-price-targets", None),
            ("segment", "bbg-segment-revenue", None),
            ("analyst_rating", _RATINGS_LIBRARY, None),
        )
        results = await asyncio.gather(
            *(asyncio.to_thread(
                self._run_job, kind, library, extra, plan, period_map, fiscal_year_end_month)
              for kind, library, extra in jobs),
            return_exceptions=True,
        )
        observations: list[RawObservation] = []
        errors: list[str] = []
        for (kind, library, _), result in zip(jobs, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{library}: {result}")
                continue
            observations.extend(result)
        return self.ok(tuple(observations), errors=tuple(errors))

    def _run_job(
        self, kind: str, library: str, extra: Any, plan: ResearchPlan,
        period_map: dict[dt.date, str], fiscal_year_end_month: int,
    ) -> list[RawObservation]:
        if kind == "statement":
            return self._statement(plan, library, period_map, fiscal_year_end_month)
        if kind == "estimate":
            return self._estimate(plan, library, extra)
        if kind == "price_target":
            return self._price_target(plan, library)
        if kind == "analyst_rating":
            return self._analyst_ratings(plan, library)
        return self._segment_revenue(plan, library, period_map, fiscal_year_end_month)

    # -- period resolution -------------------------------------------------
    def _period_map(self, ticker: str) -> dict[dt.date, str]:
        """{report date -> label} from the income statement's own labels."""
        df = self._connection.read(_PERIOD_LIBRARY, ticker)
        if df is None or df.empty or _PERIOD_COLUMN not in df.columns:
            return {}
        out: dict[dt.date, str] = {}
        for _, row in df.iterrows():
            date = _to_date(row.get("date"))
            label = row.get(_PERIOD_COLUMN)
            if date is not None and isinstance(label, str) and label.strip():
                out[date] = label.strip()
        return out

    #: How close a date must be to a real reported period-end (in the income
    #: statement's own labels) to borrow that label, rather than falling back
    #: to calendar arithmetic. NVIDIA's real fiscal quarters, like many
    #: companies', do not land on exact calendar-month boundaries even once
    #: the fiscal year-end month is known - a company's own recently-reported
    #: dates are a far more reliable guide than month arithmetic, so they are
    #: preferred whenever one exists nearby.
    _NEAREST_PERIOD_WINDOW_DAYS = 45

    @classmethod
    def _label_for(
        cls, date: dt.date, period_map: dict[dt.date, str], fiscal_year_end_month: int
    ) -> str:
        label = period_map.get(date)
        if label is not None:
            return label
        if period_map:
            nearest = min(period_map, key=lambda d: abs((d - date).days))
            if abs((nearest - date).days) <= cls._NEAREST_PERIOD_WINDOW_DAYS:
                return period_map[nearest]
        return quarter_label(date, fiscal_year_end_month)

    # -- quarterly statements ----------------------------------------------
    def _statement(
        self, plan: ResearchPlan, library: str,
        period_map: dict[dt.date, str], fiscal_year_end_month: int,
    ) -> list[RawObservation]:
        df = self._connection.read(library, plan.ticker)
        if df is None or df.empty or "date" not in df.columns:
            return []
        df = df.sort_values("date")

        metric_columns = [c for c in df.columns if str(c) not in _NON_METRIC_COLUMNS]
        has_own_label = _PERIOD_COLUMN in df.columns
        labels: list[str | None] = []
        for _, row in df.iterrows():
            date = _to_date(row.get("date"))
            own_label = str(row.get(_PERIOD_COLUMN) or "").strip() if has_own_label else ""
            labels.append(
                own_label or (self._label_for(date, period_map, fiscal_year_end_month)
                              if date else None)
            )
        df = df.assign(_period_label=labels)
        df = df[df["_period_label"].notna()]
        if df.empty:
            return []

        # Some bbg-* libraries repeat the same period's row across every
        # calendar day until the next report, with most fields NaN except on
        # the day actually reported. Blindly taking the most recent raw rows
        # would grab repeated near-empty days rather than distinct reported
        # periods, so instead: pick one representative row per period - the
        # fullest one, ties broken by the latest date - then keep the most
        # recent N periods.
        df = df.assign(_completeness=df[metric_columns].notna().sum(axis=1))
        representative = (
            df.sort_values(["_completeness", "date"])
            .groupby("_period_label", sort=False).tail(1)
            .sort_values("date")
        )

        source = SourceRef(
            source_id=f"arcticdb:{library}:{plan.ticker}", source_name=f"ArcticDB ({library})",
            source_type=SourceType.COMPANY_FILING,
        )
        out: list[RawObservation] = []
        for _, row in representative.tail(_QUARTERS_KEPT).iterrows():
            date = _to_date(row.get("date"))
            period_end_iso = date.isoformat() if date else None
            period_label = row["_period_label"]
            for column in metric_columns:
                numeric_value = _to_float(row.get(column))
                if numeric_value is None:
                    continue
                out.append(RawObservation(
                    metric=str(column), value=numeric_value, source=source,
                    period=period_label, period_end=period_end_iso, as_of=period_end_iso,
                    company=plan.company, ticker=plan.ticker, confidence=Confidence.HIGH,
                    metadata={"library": library},
                ))
        return out

    # -- consensus estimates -------------------------------------------------
    def _estimate(
        self, plan: ResearchPlan, library: str, period_column: str | None
    ) -> list[RawObservation]:
        """Consensus estimates for each forecast period.

        These libraries hold a revision history: the same forecast period
        legitimately gets a different value on every capture date as
        estimates are revised. Keeping the whole history under one ``period``
        label would make every revised figure look like a contradiction
        (QA's metric-agreement check has no way to tell a revision from a
        genuine sourcing conflict), so only the most recent reading per
        forecast period is kept - the current consensus, not its history.
        """
        df = self._connection.read(library, plan.ticker)
        if df is None or df.empty or "date" not in df.columns or not period_column:
            return []
        if period_column not in df.columns:
            return []

        df = df.assign(_period=df[period_column].apply(
            lambda p: p.strip() if isinstance(p, str) and p.strip() else None))
        df = df[df["_period"].notna()].sort_values("date")
        latest_per_period = df.groupby("_period", sort=False).tail(1)

        source = SourceRef(
            source_id=f"arcticdb:{library}:{plan.ticker}", source_name=f"ArcticDB ({library})",
            source_type=SourceType.SELL_SIDE_CONSENSUS,
        )
        out: list[RawObservation] = []
        for _, row in latest_per_period.tail(_ESTIMATE_ROWS_KEPT).iterrows():
            as_of = _as_of(row.get("date"))
            period = row["_period"]
            for column, value in row.items():
                if str(column) in _NON_METRIC_COLUMNS or str(column) == "_period":
                    continue
                numeric_value = _to_float(value)
                if numeric_value is None:
                    continue
                out.append(RawObservation(
                    metric=str(column), value=numeric_value, source=source,
                    period=period, as_of=as_of, confidence=Confidence.MEDIUM,
                    company=plan.company, ticker=plan.ticker, metadata={"library": library},
                ))
        return out

    # -- price targets (point-in-time, not period-based) --------------------
    def _price_target(self, plan: ResearchPlan, library: str) -> list[RawObservation]:
        df = self._connection.read(library, plan.ticker)
        if df is None or df.empty or "date" not in df.columns:
            return []
        df = df.sort_values("date")
        latest = df.iloc[-1]
        source = SourceRef(
            source_id=f"arcticdb:{library}:{plan.ticker}", source_name=f"ArcticDB ({library})",
            source_type=SourceType.SELL_SIDE_CONSENSUS,
        )
        as_of = _as_of(latest.get("date"))
        out: list[RawObservation] = []
        for column, value in latest.items():
            if str(column) in _NON_METRIC_COLUMNS:
                continue
            numeric_value = _to_float(value)
            if numeric_value is None:
                continue
            out.append(RawObservation(
                metric=str(column), value=numeric_value, source=source, as_of=as_of,
                confidence=Confidence.MEDIUM, company=plan.company, ticker=plan.ticker,
                metadata={"library": library},
            ))
        return out

    # -- individual analyst ratings (not collapsed into one consensus) ------
    def _analyst_ratings(self, plan: ResearchPlan, library: str) -> list[RawObservation]:
        df = self._connection.read(library, plan.ticker)
        if df is None or df.empty:
            return []
        # ibes.recommendations is indexed by announcement date (anndats),
        # not a 'date' column like the bbg-* libraries.
        df = df.sort_index()

        out: list[RawObservation] = []
        for anndats, row in df.tail(_RATINGS_KEPT).iterrows():
            rating = row.get("itext") or row.get("etext")
            rating_text = str(rating).strip() if rating is not None else ""
            if not rating_text:
                continue
            firm = str(row.get("estimid") or "").strip()
            analyst = str(row.get("analyst") or "").strip()
            source_name = f"{firm} ({analyst})" if firm and analyst else (firm or "IBES analyst")
            as_of = _as_of(anndats)
            source = SourceRef(
                source_id=f"arcticdb:{library}:{plan.ticker}:{firm}:{as_of}",
                source_name=source_name, source_type=SourceType.SELL_SIDE_CONSENSUS,
            )
            out.append(RawObservation(
                metric="consensus_rating", value=rating_text, source=source, as_of=as_of,
                confidence=Confidence.MEDIUM, company=plan.company, ticker=plan.ticker,
                metadata={"library": library, "firm": firm, "analyst": analyst},
            ))
        return out

    # -- named segment revenue -----------------------------------------------
    def _segment_revenue(
        self, plan: ResearchPlan, library: str,
        period_map: dict[dt.date, str], fiscal_year_end_month: int,
    ) -> list[RawObservation]:
        df = self._connection.read(library, plan.ticker)
        if df is None or df.empty or "date" not in df.columns or "segment" not in df.columns:
            return []
        df = df.sort_values("date")

        # A (period, segment) pair can appear more than once (restatements),
        # so keep only the latest-dated value for each - the same "one
        # representative row" rule applied to the other statements, to avoid
        # QA seeing a restatement as a sourcing conflict.
        latest: dict[tuple[str, str], tuple[dt.date, float]] = {}
        for _, row in df.tail(_SEGMENT_ROWS_KEPT).iterrows():
            date = _to_date(row.get("date"))
            numeric_value = _to_float(row.get("value"))
            segment_name = row.get("segment")
            if date is None or numeric_value is None or not segment_name:
                continue
            period_label = self._label_for(date, period_map, fiscal_year_end_month)
            key = (period_label, str(segment_name))
            if key not in latest or date > latest[key][0]:
                latest[key] = (date, numeric_value)

        source = SourceRef(
            source_id=f"arcticdb:{library}:{plan.ticker}", source_name=f"ArcticDB ({library})",
            source_type=SourceType.COMPANY_FILING,
        )
        out: list[RawObservation] = []
        for (period_label, segment_name), (date, numeric_value) in latest.items():
            period_end_iso = date.isoformat()
            out.append(RawObservation(
                metric="segment_revenue", value=numeric_value, source=source,
                period=period_label, period_end=period_end_iso, as_of=period_end_iso,
                company=plan.company, ticker=plan.ticker, confidence=Confidence.HIGH,
                metadata={"library": library, "segment_name": str(segment_name)},
            ))
        return out


def _to_date(value: object) -> dt.date | None:
    if value is None:
        return None
    return value.date() if hasattr(value, "date") else None


def _as_of(value: object) -> str | None:
    date = _to_date(value)
    return date.isoformat() if date else (str(value) if value is not None else None)


def _to_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric
