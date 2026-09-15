"""What Matters Next agent.

Produces the specific, observable items an investor should watch. Every item is
tied to a metric already in the Evidence Store or an analytic already computed,
so each is checkable at the next report rather than a general exhortation.
"""

from __future__ import annotations

from ..domain.enums import ClaimType, Confidence, SegmentName
from ..domain.segment import KeyFinding, SegmentResult
from ..normalisation.dates import next_quarter_period, normalise_fiscal_period
from ..normalisation.units import format_currency
from .base import AgentContext, SegmentAgent


class WhatMattersNextAgent(SegmentAgent):
    segment = SegmentName.WHAT_MATTERS_NEXT

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics
        latest = context.latest_period

        watch_items: list[KeyFinding] = []
        gaps = []

        next_period = None
        if latest:
            period = normalise_fiscal_period(latest)
            next_period = next_quarter_period(period) if period else None

        # 1. Guidance delivery.
        guidance = reader.guidance()
        guidance_gap = analytics.first("guidance_vs_consensus_pct")
        for item in guidance:
            watch_items.append(KeyFinding(
                claim=(
                    f"Whether {item.period_label} "
                    f"{(item.metric or '').replace('guidance_', '').replace('_', ' ')} lands at "
                    f"or above the guided {_guided_value(item)}"
                    + (f", against consensus that sits {guidance_gap.value:+.1f}% away."
                       if guidance_gap and item.metric == "guidance_revenue" else ".")
                ),
                claim_type=ClaimType.CALCULATED_OBSERVATION,
                evidence_ids=(item.evidence_id,),
                analytics_ids=(
                    (guidance_gap.analytics_id,)
                    if guidance_gap and item.metric == "guidance_revenue" else ()),
                confidence=item.confidence,
                materiality=1,
                tags=("monitor", "guidance"),
            ))

        # 2. The margin trend, which is where the ramp shows up.
        trend = analytics.first("trend_direction")
        gm_change = analytics.first("gross_margin_change_yoy")
        if trend is not None:
            direction = str(trend.metadata.get("direction", "unclear"))
            watch_items.append(KeyFinding(
                claim=(
                    f"Whether gross margin inflects: the observed trend across the periods "
                    f"held is {direction} at {trend.value:+.2f}pp per period"
                    + (f", and the year-on-year change was {gm_change.value:+.1f}pp."
                       if gm_change else ".")
                ),
                claim_type=ClaimType.CALCULATED_OBSERVATION,
                analytics_ids=tuple(
                    a.analytics_id for a in (trend, gm_change) if a is not None),
                evidence_ids=trend.input_evidence_ids,
                confidence=trend.confidence,
                materiality=1,
                tags=("monitor", "margin"),
            ))
        else:
            gaps.append(self.gap(
                "Not enough margin history is held to establish a trend to monitor.",
                metric="gross_margin", segment=self.segment,
            ))

        # 3. Whether the growth differential that supports the multiple persists.
        growth_gap = analytics.first("peer_growth_gap")
        vs_peers = analytics.first("valuation_vs_peers")
        if growth_gap is not None and vs_peers is not None:
            watch_items.append(KeyFinding(
                claim=(
                    f"Whether the {growth_gap.value:+.0f}pp revenue growth advantage over the "
                    f"peer average holds, since it is what currently supports the "
                    f"{vs_peers.value:+.0f}% forward P/E premium."
                ),
                claim_type=ClaimType.INTERPRETATION,
                analytics_ids=(growth_gap.analytics_id, vs_peers.analytics_id),
                evidence_ids=growth_gap.input_evidence_ids,
                confidence=Confidence.MEDIUM,
                materiality=1,
                tags=("monitor", "valuation"),
            ))

        # 4. Concentration in the mix.
        largest_share = max(
            analytics.by_metric("segment_contribution_pct"),
            key=lambda a: a.value, default=None)
        if largest_share is not None:
            name = largest_share.metadata.get("segment_name")
            watch_items.append(KeyFinding(
                claim=(
                    f"Whether the mix concentrates further: {name} already accounts for "
                    f"{largest_share.value:.0f}% of revenue, so group results now track that "
                    "one segment."
                ),
                claim_type=ClaimType.CALCULATED_OBSERVATION,
                analytics_ids=(largest_share.analytics_id,),
                evidence_ids=largest_share.input_evidence_ids,
                confidence=largest_share.confidence,
                materiality=2,
                tags=("monitor", "mix"),
            ))

        # 5. Whether inventory and commitments stay aligned with demand.
        inventory = reader.kpi("Inventory", latest)
        commitments = reader.kpi("Purchase commitments and supply obligations", latest)
        if inventory is not None or commitments is not None:
            parts = []
            evidence = []
            if inventory is not None:
                parts.append(f"inventory of {format_currency(inventory.value or 0)}")
                evidence.append(inventory.evidence_id)
            if commitments is not None:
                parts.append(f"commitments of {format_currency(commitments.value or 0)}")
                evidence.append(commitments.evidence_id)
            watch_items.append(KeyFinding(
                claim=(
                    "Whether " + " and ".join(parts)
                    + " continue to be matched by shipments, since both are disclosed ahead "
                    "of the revenue they support."
                ),
                claim_type=ClaimType.CALCULATED_OBSERVATION,
                evidence_ids=tuple(evidence),
                confidence=Confidence.MEDIUM,
                materiality=2,
                tags=("monitor", "supply"),
            ))

        # 6. Anything the user explicitly asked to emphasise.
        for focus in context.plan.request.focus:
            # The whole phrase first, then its significant words: an emphasis
            # like "gross margin trajectory" rarely appears verbatim in a filing.
            matches = reader.documents_matching(
                [focus, *_significant_words(focus)], limit=1)
            if matches:
                watch_items.append(self.document_finding(
                    matches[0], materiality=1, tags=("monitor", "user_emphasis"),
                    prefix=f"On the requested emphasis '{focus}': ",
                ))
            else:
                gaps.append(self.gap(
                    f"No evidence was retrieved covering the requested emphasis '{focus}'.",
                    impact="This emphasis area is not addressed in the report.",
                    segment=self.segment,
                ))

        headline = (
            f"{len(watch_items)} checkable items before the "
            f"{next_period or 'next'} result"
        )
        narrative = self.narrative([
            (
                f"The next scheduled datapoint is the {next_period} result."
            ) if next_period else "",
            (
                "The items below are stated so that each can be marked delivered or missed "
                "against the next report."
            ) if watch_items else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=tuple(watch_items),
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
            metadata={"next_period": next_period},
        )


def _guided_value(item) -> str:
    """Format a guidance figure according to its unit.

    Guidance covers both currency amounts and margins, so the unit decides;
    formatting a margin as a currency amount is a real bug, not a cosmetic one.
    """
    if item.unit == "pct":
        return f"{item.value:.1f}%"
    return format_currency(item.value or 0)


#: Words too generic to be worth matching a document on.
_FOCUS_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with",
    "trajectory", "outlook", "trend", "trends", "performance", "growth",
})


def _significant_words(phrase: str) -> list[str]:
    """The words in an emphasis phrase worth searching a document for."""
    words = [w.strip(".,;:").lower() for w in phrase.split()]
    return [w for w in words if len(w) > 3 and w not in _FOCUS_STOPWORDS]
