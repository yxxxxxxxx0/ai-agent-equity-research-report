"""Stage 7 - the segment agent framework.

An agent receives the research plan, a read-only EvidenceReader and the
analytics bundle. It has no provider access and no network access, so anything
it says must come from the Evidence Store. It returns a structured
SegmentResult - never rendered prose destined straight for the PDF.

The agents in this prototype are deterministic: they compose findings from
evidence and analytics using explicit rules. That choice buys reproducibility
and makes the "agents must not invent facts" requirement structural rather than
a matter of prompt discipline. Swapping one agent for a model-backed
implementation means subclassing SegmentAgent and returning the same
SegmentResult; nothing else in the pipeline changes.
"""

from __future__ import annotations

import abc
import logging
from dataclasses import dataclass
from typing import Sequence

from ..domain.analytics import AnalyticsBundle, AnalyticsResult
from ..domain.enums import ClaimType, Confidence, SegmentName, SourceType
from ..domain.evidence import EvidenceItem
from ..domain.plan import ResearchPlan, SegmentTask
from ..domain.segment import DataGap, KeyFinding, MetricHighlight, SegmentResult
from ..evidence.reader import EvidenceReader
from ..logging_setup import get_logger, log_event
from ..normalisation.canonical_metrics import display_label
from ..normalisation.units import format_number, format_signed_percent

logger = get_logger("agents")


@dataclass(frozen=True, slots=True)
class AgentContext:
    """Everything an agent is allowed to see."""

    plan: ResearchPlan
    reader: EvidenceReader
    analytics: AnalyticsBundle
    task: SegmentTask | None
    latest_period: str | None

    @property
    def company(self) -> str:
        return self.plan.company

    @property
    def ticker(self) -> str | None:
        return self.plan.ticker


class SegmentAgent(abc.ABC):
    """Base class for the report-analysis agents."""

    segment: SegmentName

    async def run(self, context: AgentContext) -> SegmentResult:
        """Execute the agent, converting a crash into an empty-but-honest result.

        One failing agent must not take the report down; the failure is recorded
        on the SegmentResult and surfaced by QA.
        """
        try:
            result = await self._analyse(context)
        except Exception as exc:  # noqa: BLE001 - agents are isolated by design
            log_event(logger, logging.ERROR, "segment agent failed",
                      segment=self.segment.value, error=f"{type(exc).__name__}: {exc}")
            return SegmentResult(
                segment=self.segment,
                headline=f"{self.segment.value.replace('_', ' ').title()} unavailable",
                errors=(f"{type(exc).__name__}: {exc}",),
                data_gaps=(DataGap(
                    description=f"The {self.segment.value} agent failed to complete.",
                    impact="This section is missing from the report.",
                    segment=self.segment,
                ),),
            )
        log_event(
            logger, logging.INFO, "segment agent complete",
            segment=self.segment.value, findings=len(result.key_findings),
            metrics=len(result.important_metrics), data_gaps=len(result.data_gaps),
        )
        return result

    @abc.abstractmethod
    async def _analyse(self, context: AgentContext) -> SegmentResult:
        """Produce the structured result for this segment."""

    # -- shared helpers --------------------------------------------------
    @staticmethod
    def analytic_finding(
        result: AnalyticsResult | None,
        template: str,
        *,
        claim_type: ClaimType = ClaimType.CALCULATED_OBSERVATION,
        materiality: int = 2,
        tags: tuple[str, ...] = (),
    ) -> KeyFinding | None:
        """Build a finding from an analytics result.

        ``template`` is formatted with ``{value}`` (already unit-formatted) and
        ``{raw}`` (the bare number). Returns None when the analytic is missing,
        so callers can record a data gap instead of writing an unsupported claim.
        """
        if result is None:
            return None
        return KeyFinding(
            claim=template.format(
                value=format_analytic(result),
                raw=f"{result.value:.1f}",
                label=result.label,
            ),
            claim_type=claim_type,
            analytics_ids=(result.analytics_id,),
            evidence_ids=result.input_evidence_ids,
            confidence=result.confidence,
            materiality=materiality,
            tags=tags,
        )

    #: A quoted finding is a report bullet, not the source document; a
    #: passage longer than this is truncated. Without this, a long claim_text
    #: (a full filing section, an uncommonly long transcript remark) can
    #: produce a paragraph the PDF renderer cannot lay out as one list item.
    _MAX_QUOTE_CHARS = 600

    @staticmethod
    def document_finding(
        item: EvidenceItem,
        *,
        claim_type: ClaimType | None = None,
        materiality: int = 2,
        tags: tuple[str, ...] = (),
        prefix: str = "",
    ) -> KeyFinding:
        """Build a finding that quotes a document passage.

        The claim type defaults to the epistemic status implied by the document:
        a transcript remark is a management statement, a filing is a confirmed
        fact, a news item is neither.
        """
        resolved_type = claim_type or _claim_type_for(item)
        text = item.claim_text or ""
        if len(text) > SegmentAgent._MAX_QUOTE_CHARS:
            text = text[: SegmentAgent._MAX_QUOTE_CHARS].rstrip() + "…"
        return KeyFinding(
            claim=f"{prefix}{text}" if prefix else text,
            claim_type=resolved_type,
            evidence_ids=(item.evidence_id,),
            confidence=item.confidence,
            materiality=materiality,
            tags=tags,
        )

    @staticmethod
    def metric_highlight(
        item: EvidenceItem | None,
        *,
        label: str | None = None,
        comparison: AnalyticsResult | None = None,
    ) -> MetricHighlight | None:
        """Turn an evidence item into a display-ready table row."""
        if item is None or item.value is None:
            return None
        return MetricHighlight(
            label=label or display_label(item.metric or "metric"),
            value_text=format_number(item.value, item.unit or "", item.currency),
            period=item.period_label or (item.as_of.isoformat() if item.as_of else None),
            comparison=format_analytic(comparison) if comparison else None,
            evidence_ids=(item.evidence_id,),
            analytics_ids=(comparison.analytics_id,) if comparison else (),
        )

    @staticmethod
    def gap(description: str, *, metric: str | None = None, impact: str = "",
            segment: SegmentName | None = None, priority: str = "medium") -> DataGap:
        return DataGap(description=description, missing_metric=metric, impact=impact,
                       segment=segment, priority=priority)

    @staticmethod
    def compact(findings: Sequence[KeyFinding | None]) -> tuple[KeyFinding, ...]:
        return tuple(f for f in findings if f is not None)

    @staticmethod
    def compact_metrics(metrics: Sequence[MetricHighlight | None]) -> tuple[MetricHighlight, ...]:
        return tuple(m for m in metrics if m is not None)

    @staticmethod
    def narrative(sentences: Sequence[str]) -> str:
        """Join non-empty sentences into a paragraph."""
        return " ".join(s.strip() for s in sentences if s and s.strip())

    def questions(self, context: AgentContext) -> tuple[str, ...]:
        """The plan's research questions assigned to this segment."""
        return tuple(q.text for q in context.plan.questions_for(self.segment))


def format_analytic(result: AnalyticsResult) -> str:
    """Format an analytics value according to its unit, with an explicit sign."""
    if result.unit == "pct":
        return format_signed_percent(result.value)
    if result.unit == "pp":
        return f"{result.value:+.1f}pp"
    if result.unit == "x":
        return f"{result.value:.1f}x"
    return f"{result.value:,.2f}"


def _claim_type_for(item: EvidenceItem) -> ClaimType:
    """Map a document's source type to the epistemic status of quoting it."""
    if item.source_type in {SourceType.COMPANY_FILING, SourceType.EARNINGS_RELEASE,
                            SourceType.COMPETITOR_FILING}:
        # A filing states facts, unless the passage is an outlook.
        if item.metadata.get("speaker") == "Company outlook":
            return ClaimType.MANAGEMENT_STATEMENT
        return ClaimType.CONFIRMED_FACT
    if item.source_type in {SourceType.EARNINGS_CALL, SourceType.INVESTOR_PRESENTATION,
                            SourceType.COMPANY_ANNOUNCEMENT}:
        return ClaimType.MANAGEMENT_STATEMENT
    if item.source_type in {SourceType.SELL_SIDE_CONSENSUS, SourceType.INDUSTRY_RESEARCH}:
        return ClaimType.MARKET_EXPECTATION
    return ClaimType.INTERPRETATION


def default_confidence(items: Sequence[EvidenceItem]) -> Confidence:
    """Weakest-link confidence across a set of evidence."""
    order = {Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1, Confidence.UNKNOWN: 0}
    if not items:
        return Confidence.UNKNOWN
    return min((i.confidence for i in items), key=lambda c: order[c])
