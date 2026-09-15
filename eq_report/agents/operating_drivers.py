"""Operating Drivers agent: which segments and KPIs drove the result."""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName, SourceType
from ..domain.segment import KeyFinding, MetricHighlight, SegmentResult
from ..normalisation.units import format_currency
from .base import AgentContext, SegmentAgent


class OperatingDriversAgent(SegmentAgent):
    segment = SegmentName.OPERATING_DRIVERS

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics
        period = context.latest_period

        gaps = []
        if period is None:
            return SegmentResult(
                segment=self.segment,
                headline="No reported period available for segment analysis",
                data_gaps=(self.gap(
                    "No reported period is available, so segment performance cannot be assessed.",
                    segment=self.segment,
                ),),
                open_questions=self.questions(context),
            )

        rows = reader.segment_rows(period)
        if not rows:
            gaps.append(self.gap(
                f"No segment revenue breakdown is available for {period}.",
                metric="segment_revenue",
                impact="Operating drivers can only be described at the total-company level.",
                segment=self.segment,
            ))

        growth_by_segment = {
            str(a.metadata.get("segment_name")): a
            for a in analytics.by_metric("segment_growth_yoy")
        }
        share_by_segment = {
            str(a.metadata.get("segment_name")): a
            for a in analytics.by_metric("segment_contribution_pct")
        }

        findings: list[KeyFinding | None] = []
        metrics: list[MetricHighlight | None] = []

        for row in rows:
            name = str(row.metadata.get("segment_name", "Unnamed segment"))
            growth = growth_by_segment.get(name)
            share = share_by_segment.get(name)

            analytics_ids = tuple(
                a.analytics_id for a in (growth, share) if a is not None)
            claim_parts = [f"{name} revenue was {format_currency(row.value or 0)} in {period}"]
            if growth is not None:
                claim_parts.append(f"{growth.value:+.0f}% year on year")
            if share is not None:
                claim_parts.append(f"representing {share.value:.0f}% of total revenue")
            findings.append(KeyFinding(
                claim=", ".join(claim_parts) + ".",
                claim_type=ClaimType.CONFIRMED_FACT,
                evidence_ids=(row.evidence_id,),
                analytics_ids=analytics_ids,
                confidence=row.confidence,
                materiality=1 if (share and share.value >= 20) else 3,
                tags=("segment", name.lower()),
            ))
            metrics.append(MetricHighlight(
                label=name,
                value_text=format_currency(row.value or 0),
                period=period,
                comparison=(f"{growth.value:+.0f}% YoY" if growth else None),
                evidence_ids=(row.evidence_id,),
                analytics_ids=analytics_ids,
            ))

        # The largest and fastest-growing segments carry the narrative.
        largest = rows[0] if rows else None
        fastest = max(
            growth_by_segment.values(), key=lambda a: a.value, default=None)
        slowest = min(
            growth_by_segment.values(), key=lambda a: a.value, default=None)

        if fastest is not None and slowest is not None and fastest is not slowest:
            findings.append(KeyFinding(
                claim=(
                    f"Growth is widely dispersed across the portfolio: "
                    f"{fastest.metadata.get('segment_name')} grew {fastest.value:+.0f}% "
                    f"while {slowest.metadata.get('segment_name')} grew "
                    f"{slowest.value:+.0f}%."
                ),
                claim_type=ClaimType.INTERPRETATION,
                analytics_ids=(fastest.analytics_id, slowest.analytics_id),
                evidence_ids=fastest.input_evidence_ids + slowest.input_evidence_ids,
                confidence=fastest.confidence,
                materiality=2,
                tags=("segment", "dispersion"),
            ))

        # Company-specific KPIs.
        kpi_rows = reader.kpi_rows(period)
        for item in kpi_rows:
            name = str(item.metadata.get("kpi_name", "KPI"))
            prior = self._prior_kpi(context, name)
            comparison = None
            if prior is not None and prior.value:
                change = (item.value or 0) / prior.value - 1
                comparison = f"{change * 100:+.0f}% YoY"
            metrics.append(MetricHighlight(
                label=name,
                value_text=format_currency(item.value or 0),
                period=period,
                comparison=comparison,
                evidence_ids=(item.evidence_id,) + (
                    (prior.evidence_id,) if prior is not None else ()),
            ))

        # Management's own account of the drivers, clearly labelled as such.
        for item in reader.documents_matching(
            ["data centre", "data center", "networking", "demand", "attach"],
            [SourceType.EARNINGS_CALL, SourceType.INVESTOR_PRESENTATION],
            limit=3,
        ):
            findings.append(self.document_finding(
                item, materiality=2, tags=("driver", "management")))

        if not kpi_rows:
            gaps.append(self.gap(
                f"No company-specific operating KPIs are available for {period}.",
                metric="kpi", segment=self.segment,
            ))

        headline = self._headline(period, largest, share_by_segment, fastest)
        narrative = self.narrative([
            f"Segment performance in {period} was led by "
            f"{largest.metadata.get('segment_name')} at "
            f"{format_currency(largest.value or 0)}" if largest else "",
            f"({share_by_segment[str(largest.metadata.get('segment_name'))].value:.0f}% of "
            f"total revenue)."
            if largest and str(largest.metadata.get("segment_name")) in share_by_segment else "",
            f"The fastest-growing reported segment was "
            f"{fastest.metadata.get('segment_name')} at {fastest.value:+.0f}% year on year."
            if fastest else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=self.compact(findings),
            important_metrics=self.compact_metrics(metrics),
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
            metadata={"period": period, "segment_count": len(rows)},
        )

    def _prior_kpi(self, context: AgentContext, name: str):
        """The same KPI one year earlier, if the store holds it."""
        for item in context.reader.kpi_rows():
            if item.period_label == context.latest_period:
                continue
            if str(item.metadata.get("kpi_name", "")).lower() == name.lower():
                return item
        return None

    def _headline(self, period, largest, share_by_segment, fastest) -> str:
        if largest is None:
            return f"{period}: no segment breakdown available"
        name = str(largest.metadata.get("segment_name"))
        share = share_by_segment.get(name)
        bits = [f"{name} drove the quarter at {format_currency(largest.value or 0)}"]
        if share:
            bits.append(f"{share.value:.0f}% of revenue")
        if fastest is not None and str(fastest.metadata.get("segment_name")) != name:
            bits.append(
                f"fastest growth in {fastest.metadata.get('segment_name')} "
                f"at {fastest.value:+.0f}%")
        return ", ".join(bits)
