"""Fallback fundamentals provider backed by the illustrative sample dataset."""

from __future__ import annotations

import asyncio
from typing import Any

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawObservation, SourceRef
from ...domain.plan import ResearchPlan
from ..base import FundamentalsProvider
from ..sample_data import CONSENSUS, FUNDAMENTALS, GUIDANCE, KPIS, MOCK_NOTICE, SEGMENTS


class MockFundamentalsProvider(FundamentalsProvider):
    """Deterministic sample fundamentals, segment revenue, KPIs and guidance."""

    name = "mock_fundamentals"
    is_mock = True

    def is_available(self) -> bool:
        return self.settings.allow_mock_providers

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        await asyncio.sleep(0)

        ticker = plan.ticker
        if not ticker or ticker not in FUNDAMENTALS:
            return self.ok(
                warnings=(
                    f"No sample fundamentals available for ticker {ticker or '<unknown>'}.",
                ),
            )

        observations: list[RawObservation] = []
        warnings: list[str] = []

        filing_source = self._source("reported_financials", SourceType.COMPANY_FILING)
        for period, block in FUNDAMENTALS[ticker].items():
            period_end = block["period_end"]
            for metric, value in block["lines"].items():
                observations.append(
                    self._obs(metric, value, filing_source, plan,
                              period=period, period_end=period_end)
                )

        segment_source = self._source("segment_detail", SourceType.COMPANY_FILING)
        for period, segments in SEGMENTS.get(ticker, {}).items():
            period_end = FUNDAMENTALS[ticker].get(period, {}).get("period_end")
            for segment_name, value in segments.items():
                observations.append(
                    self._obs(
                        "segment_revenue", value, segment_source, plan,
                        period=period, period_end=period_end,
                        metadata={"segment_name": segment_name},
                    )
                )

        kpi_source = self._source("operating_kpis", SourceType.COMPANY_FILING)
        for kpi in KPIS.get(ticker, []):
            observations.append(
                self._obs(
                    "kpi", kpi["value"], kpi_source, plan,
                    period=kpi.get("period"), period_end=kpi.get("period_end"),
                    unit=kpi.get("unit"),
                    metadata={"kpi_name": kpi["name"]},
                )
            )

        consensus_source = self._source("consensus", SourceType.SELL_SIDE_CONSENSUS)
        consensus_rows = CONSENSUS.get(ticker, [])
        if not consensus_rows:
            warnings.append("No consensus estimates available in the sample dataset.")
        for row in consensus_rows:
            observations.append(
                self._obs(
                    row["metric"], row["value"], consensus_source, plan,
                    period=row.get("period"), period_end=row.get("period_end"),
                    confidence=Confidence.MEDIUM,
                )
            )

        guidance_source = self._source("guidance", SourceType.EARNINGS_RELEASE)
        for row in GUIDANCE.get(ticker, []):
            observations.append(
                self._obs(
                    row["metric"], row["value"], guidance_source, plan,
                    period=row.get("period"), period_end=row.get("period_end"),
                    confidence=Confidence.MEDIUM,
                    metadata={"guidance_note": row.get("note", "")},
                )
            )

        return self.ok(tuple(observations), warnings=tuple(warnings))

    # -- helpers ---------------------------------------------------------
    def _source(self, dataset: str, source_type: SourceType) -> SourceRef:
        return SourceRef(
            source_id=f"{self.name}:{dataset}",
            source_name=f"Mock fundamentals feed ({dataset})",
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
        period: str | None = None,
        period_end: str | None = None,
        unit: str | None = None,
        confidence: Confidence = Confidence.HIGH,
        metadata: dict[str, Any] | None = None,
    ) -> RawObservation:
        meta = {"mock_notice": MOCK_NOTICE}
        if metadata:
            meta.update(metadata)
        return RawObservation(
            metric=metric,
            value=value,
            source=source,
            unit=unit,
            currency="USD",
            period=period,
            period_end=period_end,
            company=plan.company,
            ticker=plan.ticker,
            confidence=confidence,
            metadata=meta,
        )
