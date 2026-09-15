"""Risk and Catalysts agent.

Risks are drawn from filed risk factors and from observable pressure in the
numbers; catalysts from guidance, announced events and the next reporting date.
Nothing here is invented: a risk with no supporting evidence is not listed.
"""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName, SourceType
from ..domain.segment import KeyFinding, SegmentResult
from ..normalisation.dates import next_quarter_period, normalise_fiscal_period
from ..normalisation.units import format_currency
from .base import AgentContext, SegmentAgent


class RiskCatalystAgent(SegmentAgent):
    segment = SegmentName.RISKS_CATALYSTS

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics

        risks: list[KeyFinding] = []
        catalysts: list[KeyFinding] = []
        gaps = []

        # -- risks from filed risk factors --------------------------------
        risk_passages = [
            item for item in reader.documents((SourceType.COMPANY_FILING,))
            if "risk factor" in str(item.metadata.get("section", "")).lower()
        ]
        for item in risk_passages:
            risks.append(self.document_finding(
                item, claim_type=ClaimType.CONFIRMED_FACT, materiality=1,
                tags=("risk", "filed_risk_factor"),
                prefix="Filed risk factor: ",
            ))
        if not risk_passages:
            gaps.append(self.gap(
                "No filed risk factors were retrieved.",
                impact="Risks are limited to those observable in the reported numbers.",
                segment=self.segment,
            ))

        # -- risks observable in the numbers ------------------------------
        gm_change = analytics.first("gross_margin_change_yoy")
        if gm_change is not None and gm_change.value < 0:
            risks.append(KeyFinding(
                claim=(
                    f"Gross margin contracted {abs(gm_change.value):.1f}pp year on year, so a "
                    "longer or deeper ramp cost than management expects would compound into "
                    "earnings."
                ),
                claim_type=ClaimType.INTERPRETATION,
                analytics_ids=(gm_change.analytics_id,),
                evidence_ids=gm_change.input_evidence_ids,
                confidence=gm_change.confidence,
                materiality=1,
                tags=("risk", "margin"),
            ))

        qoq = analytics.first("revenue_growth_qoq")
        yoy = analytics.first("revenue_growth_yoy")
        if qoq is not None and yoy is not None and qoq.value * 4 < yoy.value:
            risks.append(KeyFinding(
                claim=(
                    f"Sequential revenue growth of {qoq.value:+.1f}% annualises well below the "
                    f"{yoy.value:+.0f}% year-on-year rate, which is the arithmetic of a "
                    "decelerating base rather than evidence of a downturn."
                ),
                claim_type=ClaimType.CALCULATED_OBSERVATION,
                analytics_ids=(qoq.analytics_id, yoy.analytics_id),
                evidence_ids=qoq.input_evidence_ids + yoy.input_evidence_ids,
                confidence=qoq.confidence,
                materiality=2,
                tags=("risk", "growth"),
            ))

        # Concentration and supply-commitment risk, where the filings show it.
        commitments = reader.kpi("Purchase commitments and supply obligations")
        if commitments is not None:
            risks.append(KeyFinding(
                claim=(
                    f"Purchase commitments and supply obligations of "
                    f"{format_currency(commitments.value or 0)} were disclosed for "
                    f"{commitments.period_label}, which converts a demand disappointment into "
                    "an inventory and write-down exposure."
                ),
                claim_type=ClaimType.INTERPRETATION,
                evidence_ids=(commitments.evidence_id,),
                confidence=commitments.confidence,
                materiality=1,
                tags=("risk", "supply"),
            ))

        for item in reader.documents_matching(
            ["export", "licens", "regulat", "restrict"],
            (SourceType.NEWS, SourceType.COMPANY_FILING),
            limit=2,
        ):
            risks.append(self.document_finding(
                item, materiality=1, tags=("risk", "regulatory")))

        # -- catalysts ----------------------------------------------------
        guidance = reader.guidance()
        guidance_gap = analytics.first("guidance_vs_consensus_pct")
        # Specifically the revenue guidance: the guidance set also holds a margin
        # figure, and formatting that as a currency amount would be wrong.
        guided = next((i for i in guidance if i.metric == "guidance_revenue"), None)
        if guided is not None:
            next_period = guided.period_label
            catalysts.append(KeyFinding(
                claim=(
                    f"The {next_period} result is the next scheduled test of the guided "
                    f"{format_currency(guided.value or 0)} revenue outlook"
                    + (f", which sits {guidance_gap.value:+.1f}% versus consensus."
                       if guidance_gap else ".")
                ),
                claim_type=ClaimType.MANAGEMENT_STATEMENT,
                evidence_ids=(guided.evidence_id,),
                analytics_ids=(guidance_gap.analytics_id,) if guidance_gap else (),
                confidence=guided.confidence,
                materiality=1,
                tags=("catalyst", "earnings"),
            ))
        else:
            latest = context.latest_period
            if latest:
                period = normalise_fiscal_period(latest)
                following = next_quarter_period(period) if period else None
                if following:
                    # A pure period-math inference (next quarter after the
                    # latest reported one) with no evidence behind it, and no
                    # guidance to compare it against - not worth asserting as
                    # a checked claim. The underlying absence of guidance is
                    # already surfaced as a data gap; recording it here too
                    # would just be a second, unsupported way of saying it.
                    gaps.append(self.gap(
                        f"No guidance is held for {following}, the next scheduled "
                        "reporting period.",
                        impact="No guidance-based catalyst can be stated for that period.",
                        segment=self.segment,
                    ))

        # Announcements are cited as dated forward events, not re-quoted: the
        # Recent Developments section already carries the passage text, and a
        # catalyst is about the effect that has yet to reach reported results.
        seen_announcements: set[str] = set()
        for item in reader.documents((SourceType.COMPANY_ANNOUNCEMENT,), limit=6):
            title = item.document_title or ""
            if not title or title in seen_announcements:
                continue
            seen_announcements.add(title)
            announced = item.published_at.isoformat() if item.published_at else "recently"
            catalysts.append(KeyFinding(
                claim=(
                    f"Announced {announced}: {title}. The commercial effect is not yet "
                    "visible in reported results, so delivery against it is the next "
                    "scheduled test of that expectation."
                ),
                claim_type=ClaimType.MANAGEMENT_STATEMENT,
                evidence_ids=(item.evidence_id,),
                confidence=item.confidence,
                materiality=2,
                tags=("catalyst", "announcement"),
            ))

        for item in reader.documents_matching(
            ["capital expenditure", "capex", "capacity"],
            (SourceType.NEWS,),
            limit=2,
        ):
            published = (
                item.published_at.isoformat() if item.published_at else "undated")
            catalysts.append(KeyFinding(
                claim=(
                    f"External demand signal, {published}: {item.document_title}. "
                    "Confirmation or reversal of this would move the revenue outlook."
                ),
                claim_type=ClaimType.MARKET_EXPECTATION,
                evidence_ids=(item.evidence_id,),
                confidence=item.confidence,
                materiality=3,
                tags=("catalyst", "demand"),
            ))

        if not catalysts:
            gaps.append(self.gap(
                "No forward-looking events or guidance were retrieved.",
                impact="No upcoming event or monitoring indicator can be stated.",
                segment=self.segment,
            ))

        risk_themes = sorted({
            tag for finding in risks for tag in finding.tags
            if tag not in {"risk", "filed_risk_factor"}
        })
        # A count of findings tells the reader nothing they could not get by
        # looking at the list, so the headline names what the risks are about.
        headline = (
            "Principal risks concern " + ", ".join(risk_themes)
            if risk_themes else
            "No risk themes could be identified from the evidence retrieved"
        )
        risk_narrative = (
            "The principal evidenced risks concern " + ", ".join(risk_themes) + "."
        ) if risk_themes else ""
        catalyst_headline = (
            f"The {guided.period_label} result is the next scheduled update to "
            "the guided outlook"
            if guided is not None else ""
        )
        catalyst_narrative = (
            f"The next scheduled update is the {guided.period_label} result, "
            "measured against the revenue guidance."
        ) if guided is not None else ""

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=tuple(risks) + tuple(catalysts),
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=risk_narrative,
            metadata={
                "risk_count": len(risks),
                "catalyst_count": len(catalysts),
                # The synthesis layer renders these under the Catalysts section;
                # draft_narrative and headline serve the Risks section.
                "catalyst_narrative": catalyst_narrative,
                "catalyst_headline": catalyst_headline,
            },
        )
