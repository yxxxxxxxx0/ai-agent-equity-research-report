"""Company Snapshot agent: scale, market pricing and recent share performance."""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName
from ..domain.segment import SegmentResult
from ..normalisation.units import format_currency
from .base import AgentContext, SegmentAgent


class CompanySnapshotAgent(SegmentAgent):
    segment = SegmentName.COMPANY_SNAPSHOT

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics

        price = reader.numeric("share_price")
        market_cap = reader.numeric("market_cap")
        enterprise_value = reader.numeric("enterprise_value")
        revenue = reader.numeric("revenue", context.latest_period)
        # A direct provider field, if MegaAPI ever sends one, wins; otherwise
        # fall back to the Analytics Engine's rolling max/min over the daily
        # OHLC series (see analytics/engine.py::_price_52w_range).
        high_52w = reader.numeric("price_52w_high")
        low_52w = reader.numeric("price_52w_low")
        high_52w_analytic = analytics.first("price_52w_high") if high_52w is None else None
        low_52w_analytic = analytics.first("price_52w_low") if low_52w is None else None

        return_1m = analytics.first("price_return_1m")
        return_3m = analytics.first("price_return_3m")
        return_ytd = analytics.first("price_return_ytd")

        gaps = []
        if price is None or market_cap is None:
            gaps.append(self.gap(
                "No current share price or market capitalisation is available.",
                metric="share_price",
                impact="The snapshot cannot state how the market currently prices the company.",
                segment=self.segment,
            ))

        findings = self.compact([
            self.analytic_finding(
                return_ytd,
                "The shares have returned {value} year to date.",
                materiality=2, tags=("price",),
            ),
            self.analytic_finding(
                return_3m,
                "Over the last three months the shares have returned {value}.",
                materiality=3, tags=("price",),
            ),
            self.analytic_finding(
                return_1m,
                "Over the last month the shares have returned {value}.",
                materiality=3, tags=("price",),
            ),
        ])

        if price is not None and market_cap is not None:
            findings = (
                self._pricing_finding(context, price, market_cap, enterprise_value),
                *findings,
            )

        metrics = self.compact_metrics([
            self.metric_highlight(price, label="Share price"),
            self.metric_highlight(market_cap, label="Market capitalisation"),
            self.metric_highlight(enterprise_value, label="Enterprise value"),
            self.metric_highlight(revenue, label="Revenue (latest reported quarter)"),
            self.metric_highlight(high_52w, label="52-week high")
            or self.metric_highlight_from_analytic(high_52w_analytic, label="52-week high"),
            self.metric_highlight(low_52w, label="52-week low")
            or self.metric_highlight_from_analytic(low_52w_analytic, label="52-week low"),
        ])

        if revenue is None:
            gaps.append(self.gap(
                "No reported revenue figure is available for the latest period.",
                metric="revenue",
                impact="The snapshot cannot state the company's current revenue scale.",
                segment=self.segment,
            ))

        headline = self._headline(context, price, market_cap, return_ytd)
        narrative = self.narrative([
            f"{context.company}"
            + (f" ({context.ticker})" if context.ticker else "")
            + " is covered here as at "
            + (price.as_of.isoformat() if price and price.as_of else "the report date")
            + ".",
            f"Market capitalisation stands at {format_currency(market_cap.value or 0)}."
            if market_cap else "",
            f"The latest reported quarter, {context.latest_period}, produced revenue of "
            f"{format_currency(revenue.value or 0)}." if revenue and context.latest_period else "",
            f"The shares have returned {return_ytd.value:+.1f}% year to date."
            if return_ytd else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=findings,
            important_metrics=metrics,
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
        )

    # -- internals -------------------------------------------------------
    def _pricing_finding(self, context, price, market_cap, enterprise_value):
        from ..domain.segment import KeyFinding

        parts = [
            f"The shares trade at {format_currency(price.value or 0)}",
            f"for a market capitalisation of {format_currency(market_cap.value or 0)}",
        ]
        evidence = [price.evidence_id, market_cap.evidence_id]
        if enterprise_value is not None:
            parts.append(
                f"and an enterprise value of {format_currency(enterprise_value.value or 0)}")
            evidence.append(enterprise_value.evidence_id)
        as_of = price.as_of.isoformat() if price.as_of else "the latest available date"
        return KeyFinding(
            claim=" ".join(parts) + f", as at {as_of}.",
            claim_type=ClaimType.CONFIRMED_FACT,
            evidence_ids=tuple(evidence),
            confidence=price.confidence,
            materiality=1,
            tags=("valuation", "price"),
        )

    def _headline(self, context, price, market_cap, return_ytd) -> str:
        if price is None or market_cap is None:
            return f"{context.company}: market data unavailable"
        ytd = f", {return_ytd.value:+.1f}% year to date" if return_ytd else ""
        return (
            f"{context.company} trades at {format_currency(price.value or 0)} "
            f"for a {format_currency(market_cap.value or 0)} market capitalisation{ytd}"
        )
