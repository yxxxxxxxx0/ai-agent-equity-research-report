"""ArcticDB-backed market data provider.

Reads price/volume from ``bbg-ohlcv``, market cap and shares outstanding from
``bbg-market-cap``, and trailing P/E from ``bbg-valuation-metrics`` - all
ticker-keyed with a plain ``date`` column (not a DatetimeIndex). These three
libraries mirror what the Research Planner's Megadata endpoint taxonomy
already expects for this branch (``/api/bbg/ohlcv/data``,
``/api/bbg/market-cap/data``). Every numeric column is emitted as an
observation; canonicalise_metric maps the ones it recognises and keeps the
rest visible as ``unmapped.*`` rather than dropping them.

No forward P/E, EV/Sales or consensus-rating library was found in this
instance, so analytics that need those (``forward_pe``, ``valuation_vs_peers``,
...) will skip with a documented data gap rather than being coerced from
what's available.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawObservation, SourceRef
from ...domain.plan import ResearchPlan
from ..base import MarketDataProvider
from .client import ArcticDBConnection

#: (library, date_column, {dataframe column -> canonical-ish metric name})
#: Columns not listed still come through under their own name.
_LIBRARIES: tuple[tuple[str, dict[str, str]], ...] = (
    ("bbg-ohlcv", {"PX_LAST": "share_price", "PX_VOLUME": "volume"}),
    ("bbg-market-cap", {}),  # market_cap / shares_outstanding already match canonically
    ("bbg-valuation-metrics", {}),  # PE_RATIO already canonicalises to trailing_pe
)
_NON_METRIC_COLUMNS = {"ticker", "date", "currency", "period"}

#: The library/column price history is built from, for the price-return analytics.
_PRICE_LIBRARY = "bbg-ohlcv"
_PRICE_COLUMN = "PX_LAST"
#: 1m/3m/YTD comparisons only need the trailing ~14 months, not the full
#: multi-decade history bbg-ohlcv holds.
_PRICE_HISTORY_ROWS = 420


class ArcticDBMarketProvider(MarketDataProvider):
    name = "arcticdb_market"
    is_mock = False

    def __init__(self, settings, *, connection: ArcticDBConnection | None = None) -> None:
        super().__init__(settings)
        self._connection = connection or ArcticDBConnection(
            settings.credentials.arcticdb_uri or "")

    def is_available(self) -> bool:
        return bool(self.settings.credentials.arcticdb_uri)

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        tickers = tuple(dict.fromkeys(t for t in (plan.ticker, *plan.peers) if t))
        if not tickers:
            return self.skipped("No ticker resolved; ArcticDB market data needs one.")

        results = await asyncio.gather(
            *(asyncio.to_thread(self._observations_for, ticker, plan) for ticker in tickers),
            return_exceptions=True,
        )
        observations: list[RawObservation] = []
        errors: list[str] = []
        for ticker, result in zip(tickers, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{ticker}: {result}")
                continue
            observations.extend(result)
        return self.ok(tuple(observations), errors=tuple(errors))

    def _observations_for(self, ticker: str, plan: ResearchPlan) -> list[RawObservation]:
        # The normaliser falls back to the plan's own company name for any
        # observation that doesn't set one, so a peer's observations must
        # carry a company name of their own - otherwise every peer ticker
        # would be recorded against the subject company (a real QA finding
        # this once produced: "NVIDIA is associated with several tickers").
        # The ticker itself stands in for a peer's company name here, since
        # this provider has no per-peer legal-name lookup.
        company_name = plan.company if ticker == plan.ticker else ticker

        out: list[RawObservation] = []
        for library, column_map in _LIBRARIES:
            df = self._connection.read(library, ticker)
            if df is None or df.empty or "date" not in df.columns:
                continue
            df = df.sort_values("date")
            source = SourceRef(
                source_id=f"arcticdb:{library}:{ticker}", source_name=f"ArcticDB ({library})",
                source_type=SourceType.MARKET_DATA,
            )
            latest = df.iloc[-1]
            as_of = _as_of(latest.get("date"))
            for column, value in latest.items():
                if str(column) in _NON_METRIC_COLUMNS:
                    continue
                numeric_value = _to_float(value)
                if numeric_value is None:
                    continue
                metric = column_map.get(str(column), str(column))
                out.append(RawObservation(
                    metric=metric, value=numeric_value, source=source, as_of=as_of,
                    company=company_name, ticker=ticker, confidence=Confidence.HIGH,
                    metadata={"arcticdb_column": str(column)},
                ))

            if library == _PRICE_LIBRARY and _PRICE_COLUMN in df.columns:
                out.extend(self._price_history(df, ticker, company_name, source))
        return out

    @staticmethod
    def _price_history(
        df: Any, ticker: str, company_name: str, source: SourceRef
    ) -> list[RawObservation]:
        history: list[RawObservation] = []
        for _, row in df.tail(_PRICE_HISTORY_ROWS).iterrows():
            numeric_value = _to_float(row.get(_PRICE_COLUMN))
            if numeric_value is None:
                continue
            history.append(RawObservation(
                metric="price_history_point", value=numeric_value, source=source,
                as_of=_as_of(row.get("date")), company=company_name, ticker=ticker,
                confidence=Confidence.HIGH, metadata={"series": "price_history"},
            ))
        return history


def _as_of(value: object) -> str | None:
    if value is None:
        return None
    return value.date().isoformat() if hasattr(value, "date") else str(value)


def _to_float(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(numeric) else numeric
