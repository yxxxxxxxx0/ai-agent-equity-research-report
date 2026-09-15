"""Valuation agent: current multiples, relative valuation and embedded expectations."""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName
from ..domain.segment import KeyFinding, MetricHighlight, SegmentResult
from ..normalisation.units import format_currency
from .base import AgentContext, SegmentAgent


class ValuationAgent(SegmentAgent):
    segment = SegmentName.VALUATION

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics

        forward_pe = reader.numeric("forward_pe")
        trailing_pe = reader.numeric("trailing_pe")
        ev_sales = reader.numeric("ev_to_sales")
        price_target = reader.numeric("price_target")
        # The rating is text-valued evidence, so it is read as a claim, not a number.
        rating_item = next(iter(reader.query(metric="consensus_rating")), None)
        price = reader.numeric("share_price")

        vs_peers = analytics.first("valuation_vs_peers")
        vs_history = analytics.first("valuation_vs_history")
        ytd = analytics.first("price_return_ytd")
        growth_yoy = analytics.first("revenue_growth_yoy")
        guidance_gap = analytics.first("guidance_vs_consensus_pct")

        findings: list[KeyFinding | None] = []
        metrics: list[MetricHighlight | None] = []
        gaps = []

        if forward_pe is not None:
            findings.append(KeyFinding(
                claim=(
                    f"The shares trade on {forward_pe.value:.1f}x forward earnings"
                    + (f" and {ev_sales.value:.1f}x EV/sales" if ev_sales else "")
                    + "."
                ),
                claim_type=ClaimType.CONFIRMED_FACT,
                evidence_ids=tuple(
                    i.evidence_id for i in (forward_pe, ev_sales) if i is not None),
                confidence=forward_pe.confidence,
                materiality=1,
                tags=("valuation",),
            ))
        else:
            gaps.append(self.gap(
                "No forward P/E observation is available.",
                metric="forward_pe",
                impact="Valuation can only be described on trailing or absolute measures.",
                segment=self.segment,
            ))

        findings.append(self.analytic_finding(
            vs_peers,
            "That is a {value} forward P/E premium to the peer average.",
            claim_type=ClaimType.CALCULATED_OBSERVATION,
            materiality=1, tags=("valuation", "peers"),
        ))
        if vs_peers is None:
            gaps.append(self.gap(
                "No peer valuation multiples are available for comparison.",
                metric="forward_pe",
                impact="Relative valuation against peers cannot be quantified.",
                segment=self.segment,
            ))

        findings.append(self.analytic_finding(
            vs_history,
            "Relative to its own observed history the forward multiple is {value}.",
            claim_type=ClaimType.CALCULATED_OBSERVATION,
            materiality=2, tags=("valuation", "history"),
        ))
        if vs_history is None:
            gaps.append(self.gap(
                "Insufficient valuation history is held to compare the current multiple "
                "with its own range.",
                metric="forward_pe",
                impact="The report cannot say whether the shares are expensive versus history.",
                segment=self.segment,
            ))

        if price_target is not None and price is not None:
            implied = (price_target.value or 0) / (price.value or 1) - 1
            findings.append(KeyFinding(
                claim=(
                    f"The consensus price target of {format_currency(price_target.value or 0)} "
                    f"implies {implied * 100:+.0f}% versus the current share price."
                ),
                claim_type=ClaimType.MARKET_EXPECTATION,
                evidence_ids=(price_target.evidence_id, price.evidence_id),
                confidence=price_target.confidence,
                materiality=2,
                tags=("valuation", "consensus"),
            ))
        elif price_target is None:
            gaps.append(self.gap(
                "No consensus price target is available.",
                metric="price_target", segment=self.segment,
            ))

        if rating_item is not None and rating_item.claim_text:
            findings.append(KeyFinding(
                claim=f"The aggregate sell-side rating is {rating_item.claim_text}.",
                claim_type=ClaimType.MARKET_EXPECTATION,
                evidence_ids=(rating_item.evidence_id,),
                confidence=rating_item.confidence,
                materiality=3,
                tags=("consensus",),
            ))

        # The interpretive step: connect the multiple to what it implies about
        # expectations. Explicitly labelled as an inference.
        if forward_pe is not None and growth_yoy is not None:
            findings.append(KeyFinding(
                claim=(
                    f"On {forward_pe.value:.1f}x forward earnings against "
                    f"{growth_yoy.value:+.0f}% revenue growth in the latest quarter, the "
                    "multiple embeds continued high growth rather than a normalisation; "
                    "a deceleration towards peer growth rates would be the main source of "
                    "multiple risk."
                ),
                claim_type=ClaimType.INTERPRETATION,
                evidence_ids=(forward_pe.evidence_id,),
                analytics_ids=tuple(
                    a.analytics_id for a in (growth_yoy, vs_peers, guidance_gap)
                    if a is not None),
                confidence=forward_pe.confidence,
                materiality=1,
                tags=("valuation", "expectations"),
            ))

        metrics = self.compact_metrics([
            self.metric_highlight(forward_pe, label="Forward P/E", comparison=vs_peers),
            self.metric_highlight(trailing_pe, label="Trailing P/E"),
            self.metric_highlight(ev_sales, label="EV/Sales"),
            self.metric_highlight(price_target, label="Consensus price target"),
            self.metric_highlight(price, label="Share price", comparison=ytd),
        ])
        if vs_history is not None:
            mean_pe = vs_history.metadata.get("historical_mean_forward_pe")
            if mean_pe is not None:
                metrics = metrics + (MetricHighlight(
                    label="Forward P/E, historical mean of held observations",
                    value_text=f"{float(mean_pe):.1f}x",
                    period=(
                        f"{vs_history.metadata.get('window_from')} to "
                        f"{vs_history.metadata.get('window_to')}"
                    ),
                    comparison=f"{vs_history.value:+.1f}% versus current",
                    analytics_ids=(vs_history.analytics_id,),
                    evidence_ids=vs_history.input_evidence_ids,
                ),)

        headline = self._headline(forward_pe, vs_peers, vs_history)
        narrative = self.narrative([
            self._valuation_sentence(forward_pe, vs_peers, vs_history),
            (
                f"Management has guided next-quarter revenue {guidance_gap.value:+.1f}% "
                "versus consensus, which is the immediate gap between the reported outlook "
                "and the estimates the multiple is built on."
            ) if guidance_gap else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=self.compact(findings),
            important_metrics=metrics,
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
        )

    @staticmethod
    def _valuation_sentence(forward_pe, vs_peers, vs_history) -> str:
        """One sentence describing the multiple and how it sits relative to context."""
        if forward_pe is None:
            return "No forward multiple is available for this company."
        clauses = []
        if vs_peers is not None:
            clauses.append(f"a {vs_peers.value:+.1f}% premium to the peer average")
        if vs_history is not None:
            clauses.append(
                f"{vs_history.value:+.1f}% versus the mean of the forward multiples held "
                "for this company")
        base = f"The shares trade on {forward_pe.value:.1f}x forward earnings"
        if not clauses:
            return base + "."
        return base + ", " + " and ".join(clauses) + "."

    def _headline(self, forward_pe, vs_peers, vs_history) -> str:
        if forward_pe is None:
            return "Valuation: no forward multiple available"
        bits = [f"{forward_pe.value:.1f}x forward earnings"]
        if vs_peers:
            bits.append(f"{vs_peers.value:+.0f}% versus peers")
        if vs_history:
            bits.append(f"{vs_history.value:+.0f}% versus its own history")
        return ", ".join(bits)
