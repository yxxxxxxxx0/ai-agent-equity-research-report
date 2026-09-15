"""Fallback market-data provider backed by the illustrative sample dataset.

Used when no real market-data credentials are configured. Every observation it
emits carries ``is_mock=True`` so the label survives normalisation, the Evidence
Store, and the rendered PDF.
"""

from __future__ import annotations

import asyncio
from typing import Any

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawObservation, SourceRef
from ...domain.plan import ResearchPlan
from ..base import MarketDataProvider
from ..sample_data import MARKET_SNAPSHOT, MOCK_NOTICE, PEER_MARKET_SNAPSHOT


class MockMarketDataProvider(MarketDataProvider):
    """Deterministic sample market data. Clearly labelled, never presented as real."""

    name = "mock_market_data"
    is_mock = True

    def is_available(self) -> bool:
        return self.settings.allow_mock_providers

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        # Simulated I/O latency so the concurrency in the acquisition layer is
        # actually exercised in tests and local runs.
        await asyncio.sleep(0)

        ticker = plan.ticker
        if not ticker or ticker not in MARKET_SNAPSHOT:
            return self.ok(
                warnings=(
                    f"No sample market data available for ticker {ticker or '<unknown>'}.",
                ),
            )

        snapshot = MARKET_SNAPSHOT[ticker]
        as_of = snapshot["as_of"]
        observations: list[RawObservation] = []

        quote_source = self._source("quote", SourceType.MARKET_DATA)
        for metric, value in snapshot["quote"].items():
            observations.append(self._obs(metric, value, quote_source, plan, as_of=as_of))

        multiples_source = self._source("multiples", SourceType.MARKET_DATA)
        for metric, value in snapshot["multiples"].items():
            observations.append(self._obs(metric, value, multiples_source, plan, as_of=as_of))

        consensus_source = self._source("consensus", SourceType.SELL_SIDE_CONSENSUS)
        for metric, value in snapshot["consensus"].items():
            observations.append(
                self._obs(metric, value, consensus_source, plan, as_of=as_of,
                          confidence=Confidence.MEDIUM)
            )

        for estimate in snapshot["estimates"]:
            observations.append(
                self._obs(
                    estimate["metric"], estimate["value"], consensus_source, plan,
                    as_of=as_of, period=estimate.get("period"),
                    period_end=estimate.get("period_end"), confidence=Confidence.MEDIUM,
                )
            )

        history_source = self._source("price_history", SourceType.MARKET_DATA)
        for date, close in snapshot["price_history"]:
            observations.append(
                self._obs("historical_close", close, history_source, plan, as_of=date,
                          metadata={"series": "price_history"})
            )

        pe_history_source = self._source("multiple_history", SourceType.MARKET_DATA)
        for date, value in snapshot["forward_pe_history"]:
            observations.append(
                self._obs("forwardPE", value, pe_history_source, plan, as_of=date,
                          confidence=Confidence.MEDIUM,
                          metadata={"series": "historical"})
            )

        warnings: list[str] = []
        peer_source = self._source("peer_quote", SourceType.MARKET_DATA)
        for peer in plan.peers:
            peer_data = PEER_MARKET_SNAPSHOT.get(peer)
            if peer_data is None:
                warnings.append(f"No sample market data for peer {peer}.")
                continue
            for metric, value in peer_data.items():
                observations.append(
                    self._obs(
                        metric, value, peer_source, plan, as_of=as_of,
                        ticker_override=peer,
                        company_override=peer,
                        confidence=Confidence.MEDIUM,
                        metadata={"entity_role": "peer", "peer_ticker": peer},
                    )
                )

        return self.ok(tuple(observations), warnings=tuple(warnings))

    # -- helpers ---------------------------------------------------------
    def _source(self, dataset: str, source_type: SourceType) -> SourceRef:
        return SourceRef(
            source_id=f"{self.name}:{dataset}",
            source_name=f"Mock market data feed ({dataset})",
            source_type=source_type,
            source_url=None,
            is_mock=True,
        )

    def _obs(
        self,
        metric: str,
        value: Any,
        source: SourceRef,
        plan: ResearchPlan,
        *,
        as_of: str | None = None,
        period: str | None = None,
        period_end: str | None = None,
        confidence: Confidence = Confidence.HIGH,
        metadata: dict[str, Any] | None = None,
        ticker_override: str | None = None,
        company_override: str | None = None,
    ) -> RawObservation:
        meta = {"mock_notice": MOCK_NOTICE}
        if metadata:
            meta.update(metadata)
        return RawObservation(
            metric=metric,
            value=value,
            source=source,
            currency="USD",
            as_of=as_of,
            period=period,
            period_end=period_end,
            company=company_override or plan.company,
            ticker=ticker_override or plan.ticker,
            confidence=confidence,
            metadata=meta,
        )
