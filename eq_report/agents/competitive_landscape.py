"""Competitive Landscape agent: relative performance and competitive developments."""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName, SourceType
from ..domain.segment import KeyFinding, MetricHighlight, SegmentResult
from .base import AgentContext, SegmentAgent


class CompetitiveLandscapeAgent(SegmentAgent):
    segment = SegmentName.COMPETITIVE_LANDSCAPE

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics
        peers = context.plan.peers

        findings: list[KeyFinding | None] = []
        metrics: list[MetricHighlight | None] = []
        gaps = []

        if not peers:
            gaps.append(self.gap(
                "No peer set was supplied or resolved for this company.",
                impact="Relative competitive positioning cannot be quantified.",
                segment=self.segment,
            ))

        growth_gap = analytics.first("peer_growth_gap")
        vs_peers = analytics.first("valuation_vs_peers")

        findings.append(self.analytic_finding(
            growth_gap,
            "Revenue growth ran {value} versus the average of the peer set.",
            claim_type=ClaimType.CALCULATED_OBSERVATION,
            materiality=1, tags=("peers", "growth"),
        ))
        findings.append(self.analytic_finding(
            vs_peers,
            "The forward earnings multiple sits {value} versus the peer average.",
            claim_type=ClaimType.CALCULATED_OBSERVATION,
            materiality=2, tags=("peers", "valuation"),
        ))

        # A peer comparison table built only from evidence actually held.
        covered_peers = 0
        for peer in peers:
            peer_pe = reader.peer_value(peer, "forward_pe")
            peer_growth = reader.peer_value(peer, "revenue_growth_yoy_reported")
            if peer_pe is None and peer_growth is None:
                gaps.append(self.gap(
                    f"No market data was retrieved for peer {peer}.",
                    impact=f"{peer} is excluded from the peer comparison.",
                    segment=self.segment,
                ))
                continue
            covered_peers += 1
            metrics.append(MetricHighlight(
                label=peer,
                value_text=(f"{peer_pe.value:.1f}x" if peer_pe else "n/a"),
                period="Forward P/E",
                comparison=(
                    f"{peer_growth.value:+.1f}% revenue growth YoY" if peer_growth else None),
                evidence_ids=tuple(
                    i.evidence_id for i in (peer_pe, peer_growth) if i is not None),
            ))

        own_pe = reader.numeric("forward_pe")
        own_growth = analytics.first("revenue_growth_yoy")
        if own_pe is not None:
            metrics.insert(0, MetricHighlight(
                label=f"{context.ticker or context.company} (subject)",
                value_text=f"{own_pe.value:.1f}x",
                period="Forward P/E",
                comparison=(
                    f"{own_growth.value:+.1f}% revenue growth YoY" if own_growth else None),
                evidence_ids=(own_pe.evidence_id,),
                analytics_ids=(own_growth.analytics_id,) if own_growth else (),
            ))

        # Competitive and industry developments, quoted with their source.
        for item in reader.documents(
            (SourceType.COMPETITOR_FILING, SourceType.INDUSTRY_RESEARCH), limit=5
        ):
            findings.append(self.document_finding(
                item, materiality=2, tags=("competition", item.source_type.value)))

        # The interpretive step, explicitly labelled.
        if growth_gap is not None and vs_peers is not None:
            findings.append(KeyFinding(
                claim=(
                    f"The company is growing {growth_gap.value:+.0f} percentage points faster "
                    f"than the peer average while trading at a {vs_peers.value:+.0f}% forward "
                    "P/E premium, so the premium is currently underwritten by a growth "
                    "differential rather than by multiple expansion alone."
                    if growth_gap.value > 0 else
                    f"The company is growing {growth_gap.value:+.0f} percentage points versus "
                    f"the peer average while trading at a {vs_peers.value:+.0f}% forward P/E "
                    "premium, which leaves the premium exposed if relative growth does not "
                    "improve."
                ),
                claim_type=ClaimType.INTERPRETATION,
                analytics_ids=(growth_gap.analytics_id, vs_peers.analytics_id),
                evidence_ids=growth_gap.input_evidence_ids + vs_peers.input_evidence_ids,
                confidence=growth_gap.confidence,
                materiality=1,
                tags=("competition", "valuation"),
            ))

        headline = self._headline(growth_gap, vs_peers, covered_peers)
        narrative = self.narrative([
            (
                f"Against a peer set of {', '.join(peers)}, revenue growth ran "
                f"{growth_gap.value:+.0f} percentage points versus the peer average."
            ) if growth_gap and peers else "",
            (
                f"The forward earnings multiple stands {vs_peers.value:+.1f}% versus the "
                "peer average."
            ) if vs_peers else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=self.compact(findings),
            important_metrics=self.compact_metrics(metrics),
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
            metadata={"peers_covered": covered_peers, "peers_requested": len(peers)},
        )

    def _headline(self, growth_gap, vs_peers, covered_peers: int) -> str:
        if growth_gap is None and vs_peers is None:
            return "Competitive position: no peer data available"
        bits = []
        if growth_gap is not None:
            bits.append(f"growing {growth_gap.value:+.0f}pp versus peers")
        if vs_peers is not None:
            bits.append(f"valued {vs_peers.value:+.0f}% versus peers")
        return f"{'; '.join(bits)} across {covered_peers} covered peers"
