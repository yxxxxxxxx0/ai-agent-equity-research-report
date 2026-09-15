"""Recent Developments agent: material events, announcements and guidance changes."""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName, SourceType
from ..domain.segment import KeyFinding, MetricHighlight, SegmentResult
from ..normalisation.canonical_metrics import display_label
from ..normalisation.units import format_number
from .base import AgentContext, SegmentAgent

#: Document types that count as a "development", most authoritative first.
#: The earnings release itself is deliberately excluded: its result highlights
#: belong in the financial performance section, and its outlook already reaches
#: this section as guidance evidence.
_EVENT_SOURCES = (
    SourceType.COMPANY_ANNOUNCEMENT,
    SourceType.NEWS,
)


class RecentDevelopmentsAgent(SegmentAgent):
    segment = SegmentName.RECENT_DEVELOPMENTS

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics

        findings: list[KeyFinding | None] = []
        metrics: list[MetricHighlight | None] = []
        gaps = []

        # Guidance is the most decision-relevant recent development.
        guidance_items = reader.guidance()
        guidance_vs_consensus = analytics.first("guidance_vs_consensus_pct")

        for item in guidance_items:
            # display_label already reads "Revenue guidance"; the bare measure is
            # what the sentence needs so it does not say "revenue guidance to".
            label = display_label(item.metric or "guidance")
            measure = label.removesuffix(" guidance")
            value_text = format_number(item.value or 0, item.unit or "", item.currency)
            note = str(item.metadata.get("guidance_note", "")).strip()
            findings.append(KeyFinding(
                claim=(
                    f"Management guided {item.period_label} {measure.lower()} to {value_text}"
                    + (f" ({note})" if note else "")
                    + "."
                ),
                claim_type=ClaimType.MANAGEMENT_STATEMENT,
                evidence_ids=(item.evidence_id,),
                confidence=item.confidence,
                materiality=1,
                tags=("guidance",),
            ))
            metrics.append(MetricHighlight(
                label=f"{measure} guidance",
                value_text=value_text,
                period=item.period_label,
                evidence_ids=(item.evidence_id,),
            ))

        if not guidance_items:
            gaps.append(self.gap(
                "No management guidance is available in the Evidence Store.",
                metric="guidance_revenue",
                impact="The report cannot say whether the outlook changed.",
                segment=self.segment,
            ))

        findings.append(self.analytic_finding(
            guidance_vs_consensus,
            "Revenue guidance sits {value} versus the consensus estimate for the guided quarter.",
            claim_type=ClaimType.CALCULATED_OBSERVATION,
            materiality=1, tags=("guidance", "consensus"),
        ))
        if guidance_vs_consensus is None and guidance_items:
            gaps.append(self.gap(
                "No consensus estimate is available for the guided period.",
                metric="consensus_revenue",
                impact="Guidance cannot be compared with expectations.",
                segment=self.segment,
            ))

        revisions = analytics.first("estimate_revision_direction")
        findings.append(self.analytic_finding(
            revisions,
            "Recent estimate revisions have been net {value} in direction "
            "(positive means upward).",
            claim_type=ClaimType.MARKET_EXPECTATION,
            materiality=2, tags=("estimates",),
        ))

        # Corporate events, most recent first, each quoted with its source.
        events = reader.documents(_EVENT_SOURCES, limit=8)
        seen_titles: set[str] = set()
        for item in events:
            title = item.document_title or ""
            # One passage per document keeps the section readable.
            if title in seen_titles:
                continue
            seen_titles.add(title)
            findings.append(self.document_finding(
                item,
                materiality=1 if item.source_type != SourceType.NEWS else 2,
                tags=("event", item.source_type.value),
            ))

        if not events:
            gaps.append(self.gap(
                "No recent announcements or news items were retrieved.",
                impact="Recent developments are limited to guidance and reported results.",
                segment=self.segment,
            ))

        headline = self._headline(guidance_items, guidance_vs_consensus, len(seen_titles))
        narrative = self.narrative([
            self._guidance_sentence(guidance_items, guidance_vs_consensus),
            (
                f"Estimate revisions have been net {revisions.value:+.0f}% in direction."
            ) if revisions else "",
            (
                f"{len(seen_titles)} distinct corporate or market developments were "
                "retrieved for the period under review."
            ) if seen_titles else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=self.compact(findings),
            important_metrics=self.compact_metrics(metrics),
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
        )

    @staticmethod
    def _guidance_sentence(guidance_items, guidance_vs_consensus) -> str:
        """One sentence on the revenue outlook, or nothing if none is held."""
        revenue_guidance = next(
            (i for i in guidance_items if i.metric == "guidance_revenue"), None)
        if revenue_guidance is None:
            return ""
        value = format_number(
            revenue_guidance.value or 0, revenue_guidance.unit or "",
            revenue_guidance.currency)
        sentence = (
            f"Management guided {revenue_guidance.period_label} revenue to {value}")
        if guidance_vs_consensus is not None:
            sentence += f", which is {guidance_vs_consensus.value:+.1f}% versus consensus"
        return sentence + "."

    def _headline(self, guidance_items, guidance_vs_consensus, event_count: int) -> str:
        if guidance_items:
            period = guidance_items[0].period_label or "the next quarter"
            if guidance_vs_consensus:
                direction = "above" if guidance_vs_consensus.value >= 0 else "below"
                return (
                    f"{period} guidance set {abs(guidance_vs_consensus.value):.1f}% "
                    f"{direction} consensus, alongside {event_count} other developments"
                )
            return f"{period} guidance issued, alongside {event_count} other developments"
        return f"{event_count} recent developments retrieved; no guidance available"
