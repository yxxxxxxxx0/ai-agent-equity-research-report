"""Financial Performance agent: what changed in the latest reported period."""

from __future__ import annotations

from ..domain.enums import ClaimType, SegmentName, SourceType
from ..domain.segment import KeyFinding, SegmentResult
from ..normalisation.units import format_currency
from .base import AgentContext, SegmentAgent


class FinancialPerformanceAgent(SegmentAgent):
    segment = SegmentName.FINANCIAL_PERFORMANCE

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader
        analytics = context.analytics
        period = context.latest_period

        if period is None:
            return SegmentResult(
                segment=self.segment,
                headline="No reported financial period is available",
                data_gaps=(self.gap(
                    "The Evidence Store contains no reported revenue for any period.",
                    metric="revenue",
                    impact="The financial performance section cannot be written.",
                    segment=self.segment,
                ),),
                open_questions=self.questions(context),
            )

        revenue = reader.numeric("revenue", period)
        eps = reader.numeric("eps_diluted", period)
        gross_margin = reader.numeric("gross_margin", period)
        operating_margin = reader.numeric("operating_margin", period)
        net_income = reader.numeric("net_income", period)
        fcf = reader.numeric("free_cash_flow", period)
        capex = reader.numeric("capex", period)
        cash = reader.numeric("cash_and_equivalents", period)
        debt = reader.numeric("total_debt", period)

        growth_yoy = analytics.first("revenue_growth_yoy")
        growth_qoq = analytics.first("revenue_growth_qoq")
        eps_growth = analytics.first("eps_growth_yoy")
        gm_change = analytics.first("gross_margin_change_yoy")
        om_change = analytics.first("operating_margin_change_yoy")
        revenue_surprise = analytics.first("revenue_surprise_pct")
        eps_surprise = analytics.first("eps_surprise_pct")
        fcf_growth = analytics.first("fcf_growth_yoy")
        conversion = analytics.first("cash_conversion")

        findings: list[KeyFinding | None] = []

        if revenue is not None:
            findings.append(KeyFinding(
                claim=(
                    f"{period} revenue was {format_currency(revenue.value or 0)}"
                    + (f", up {growth_yoy.value:.0f}% year on year" if growth_yoy
                       and growth_yoy.value >= 0 else
                       f", down {abs(growth_yoy.value):.0f}% year on year" if growth_yoy else "")
                    + (f" and {growth_qoq.value:+.0f}% sequentially" if growth_qoq else "")
                    + "."
                ),
                claim_type=ClaimType.CONFIRMED_FACT,
                evidence_ids=(revenue.evidence_id,),
                analytics_ids=tuple(
                    a.analytics_id for a in (growth_yoy, growth_qoq) if a is not None),
                confidence=revenue.confidence,
                materiality=1,
                tags=("revenue", "growth"),
            ))

        findings.append(self.analytic_finding(
            revenue_surprise,
            "Revenue came in {value} versus consensus for the quarter.",
            claim_type=ClaimType.CALCULATED_OBSERVATION, materiality=1,
            tags=("surprise", "consensus"),
        ))
        findings.append(self.analytic_finding(
            eps_surprise,
            "Diluted EPS came in {value} versus consensus.",
            claim_type=ClaimType.CALCULATED_OBSERVATION, materiality=1,
            tags=("surprise", "consensus"),
        ))
        findings.append(self.analytic_finding(
            eps_growth,
            "Diluted EPS grew {value} year on year.",
            materiality=2, tags=("earnings",),
        ))
        findings.append(self.analytic_finding(
            gm_change,
            "Gross margin moved {value} year on year.",
            materiality=1, tags=("margin",),
        ))
        findings.append(self.analytic_finding(
            om_change,
            "Operating margin moved {value} year on year.",
            materiality=2, tags=("margin",),
        ))
        findings.append(self.analytic_finding(
            fcf_growth,
            "Free cash flow grew {value} year on year.",
            materiality=2, tags=("cash",),
        ))
        findings.append(self.analytic_finding(
            conversion,
            "Free cash flow represented {value} of net income in the quarter.",
            materiality=3, tags=("cash",),
        ))

        # A management explanation for the margin move, if one exists in the
        # transcript. Quoted as a management statement, never as fact.
        margin_commentary = reader.documents_matching(
            ["gross margin", "margin"],
            [SourceType.EARNINGS_CALL],
            limit=1,
        )
        for item in margin_commentary:
            findings.append(self.document_finding(
                item, materiality=2, tags=("margin", "management"),
                prefix="Management commentary on margins: ",
            ))

        gaps = []
        if revenue_surprise is None:
            gaps.append(self.gap(
                "No consensus revenue estimate is available for the reported period.",
                metric="consensus_revenue",
                impact="The report cannot say whether revenue beat or missed expectations.",
                segment=self.segment,
            ))
        if eps_surprise is None:
            gaps.append(self.gap(
                "No consensus EPS estimate is available for the reported period.",
                metric="consensus_eps",
                impact="The report cannot quantify the earnings surprise.",
                segment=self.segment,
            ))
        if gross_margin is None:
            gaps.append(self.gap(
                f"No gross margin figure is available for {period}.",
                metric="gross_margin", segment=self.segment,
            ))

        metrics = self.compact_metrics([
            self.metric_highlight(revenue, label="Revenue", comparison=growth_yoy),
            self.metric_highlight(gross_margin, label="Gross margin", comparison=gm_change),
            self.metric_highlight(operating_margin, label="Operating margin",
                                  comparison=om_change),
            self.metric_highlight(net_income, label="Net income"),
            self.metric_highlight(eps, label="Diluted EPS", comparison=eps_growth),
            self.metric_highlight(fcf, label="Free cash flow", comparison=fcf_growth),
            self.metric_highlight(capex, label="Capex"),
            self.metric_highlight(cash, label="Cash and equivalents"),
            self.metric_highlight(debt, label="Total debt"),
        ])

        headline = self._headline(period, revenue, growth_yoy, revenue_surprise, gm_change)
        narrative = self.narrative([
            f"{context.company} reported {period} revenue of "
            f"{format_currency(revenue.value or 0)}" if revenue else "",
            f"representing {growth_yoy.value:+.1f}% year-on-year growth" if growth_yoy else "",
            f"and {growth_qoq.value:+.1f}% sequentially." if growth_qoq else ".",
            f"That was {revenue_surprise.value:+.1f}% versus consensus,"
            if revenue_surprise else "",
            f"with diluted EPS {eps_surprise.value:+.1f}% versus the consensus estimate."
            if eps_surprise else "",
            f"Gross margin moved {gm_change.value:+.1f}pp year on year"
            if gm_change else "",
            f"while operating margin moved {om_change.value:+.1f}pp." if om_change else "",
            f"Free cash flow of {format_currency(fcf.value or 0)} represented "
            f"{conversion.value:.0f}% of net income." if fcf and conversion else "",
        ])

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=self.compact(findings),
            important_metrics=metrics,
            open_questions=self.questions(context),
            data_gaps=tuple(gaps),
            draft_narrative=narrative,
            metadata={"period": period},
        )

    def _headline(self, period, revenue, growth_yoy, surprise, gm_change) -> str:
        if revenue is None:
            return f"{period}: no reported revenue available"
        bits = [f"{period} revenue {format_currency(revenue.value or 0)}"]
        if growth_yoy:
            bits.append(f"{growth_yoy.value:+.0f}% year on year")
        if surprise:
            verb = "ahead of" if surprise.value >= 0 else "below"
            bits.append(f"{abs(surprise.value):.1f}% {verb} consensus")
        if gm_change:
            bits.append(f"gross margin {gm_change.value:+.1f}pp")
        return ", ".join(bits)
