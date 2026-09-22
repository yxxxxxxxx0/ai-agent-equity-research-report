"""Builds and runs the segment agents concurrently.

Which agents run is decided by the research plan, not hard-coded here. Every
segment is written by :class:`~eq_report.agents.llm_agent.VerifiedSegmentAgent`;
deterministic code validates the resulting provenance and numbers.
"""

from __future__ import annotations

import asyncio
import logging

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import SegmentName
from ..domain.plan import ResearchPlan
from ..domain.segment import SegmentResult
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from ..pipeline.web_gap_fill import TOPIC_SEGMENTS
from .base import AgentContext, SegmentAgent
from .company_snapshot import CompanySnapshotAgent
from .competitive_landscape import CompetitiveLandscapeAgent
from .financial_performance import FinancialPerformanceAgent
from .llm_agent import VerifiedSegmentAgent
from .operating_drivers import OperatingDriversAgent
from .recent_developments import RecentDevelopmentsAgent
from .risks_catalysts import RiskCatalystAgent
from .valuation import ValuationAgent
from .what_matters_next import WhatMattersNextAgent

logger = get_logger("agents.runner")

#: Segment-specific deterministic analysis helpers. They are passed for API
#: compatibility but are not used to write fallback prose.
AGENT_REGISTRY: dict[SegmentName, type[SegmentAgent]] = {
    SegmentName.COMPANY_SNAPSHOT: CompanySnapshotAgent,
    SegmentName.FINANCIAL_PERFORMANCE: FinancialPerformanceAgent,
    SegmentName.OPERATING_DRIVERS: OperatingDriversAgent,
    SegmentName.RECENT_DEVELOPMENTS: RecentDevelopmentsAgent,
    SegmentName.VALUATION: ValuationAgent,
    SegmentName.COMPETITIVE_LANDSCAPE: CompetitiveLandscapeAgent,
    SegmentName.RISKS_CATALYSTS: RiskCatalystAgent,
    SegmentName.WHAT_MATTERS_NEXT: WhatMattersNextAgent,
}


def build_agents(
    plan: ResearchPlan, model_config: ModelConfig | None = None,
    tracker: UsageTracker | None = None, reader: EvidenceReader | None = None,
) -> tuple[SegmentAgent, ...]:
    """Instantiate the agents the plan assigned work to.

    A segment named in ``model_config.deterministic_segments`` runs its
    rule-based implementation directly - no LLM call, no evidence-pool cost -
    instead of the LLM writer every other segment gets. This is a cost
    control, not a capability gap in general: the deterministic agents
    already produce real, evidence-cited prose (see e.g.
    financial_performance.py) from canonical numeric evidence.

    It genuinely is a capability gap, though, for a segment
    pipeline.web_gap_fill just wrote document evidence for: every
    deterministic agent reads numeric evidence
    (``EvidenceReader.numeric``/``segment_rows``/``kpi_rows``) almost
    exclusively, so a claim-text-only fact it fetched specifically to fill
    that segment's gap would otherwise sit in the Evidence Store unused.
    When ``reader`` is supplied and a segment has such evidence, this routes
    it to the LLM writer instead - the one agent that actually reads
    document evidence - regardless of the cost-control default.
    """
    deterministic = model_config.deterministic_segments if model_config else frozenset()
    gap_filled = _gap_filled_segments(reader) if reader is not None else frozenset()
    agents: list[SegmentAgent] = []
    for task in plan.segment_tasks:
        agent_class = AGENT_REGISTRY.get(task.segment)
        if agent_class is None:
            log_event(logger, logging.WARNING, "no agent implements segment",
                      segment=task.segment.value)
            continue
        if task.segment.value in deterministic and task.segment not in gap_filled:
            agents.append(agent_class())
            continue
        agents.append(VerifiedSegmentAgent(
            task.segment, agent_class(), model_config, tracker=tracker))
    return tuple(agents)


def _gap_filled_segments(reader: EvidenceReader) -> frozenset[SegmentName]:
    return frozenset(
        segment for topic, segment in TOPIC_SEGMENTS.items()
        if reader.gap_fill_documents(topic)
    )


def build_context(
    agent: SegmentAgent,
    plan: ResearchPlan,
    reader: EvidenceReader,
    analytics: AnalyticsBundle,
    latest_period: str | None,
) -> AgentContext:
    """One context per agent, carrying that agent's task from the plan."""
    return AgentContext(
        plan=plan,
        reader=reader,
        analytics=analytics,
        task=plan.task_for(agent.segment),
        latest_period=latest_period,
    )


async def run_segment_agents(
    plan: ResearchPlan,
    reader: EvidenceReader,
    analytics: AnalyticsBundle,
    model_config: ModelConfig | None = None,
    tracker: UsageTracker | None = None,
) -> tuple[SegmentResult, ...]:
    """Run every planned segment agent concurrently and return their results."""
    agents = build_agents(plan, model_config, tracker, reader)
    # Resolved once and shared: the reader and analytics bundle are read-only,
    # so concurrent access from the agents is safe.
    latest_period = reader.latest_reported_period()

    results = await asyncio.gather(*(
        agent.run(build_context(agent, plan, reader, analytics, latest_period))
        for agent in agents
    ))

    log_event(
        logger, logging.INFO, "segment agents complete",
        agents=len(agents), results=len(results),
        failed=[r.segment.value for r in results if r.errors],
    )
    return tuple(results)
