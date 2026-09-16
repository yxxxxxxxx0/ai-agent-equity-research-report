"""Controlled vocabularies shared across every pipeline stage.

Keeping these as enums (rather than bare strings) is what makes the boundaries
between stages checkable: the normalisation layer can only emit categories the
evidence store understands, and QA can reason about claim provenance.
"""

from __future__ import annotations

from enum import Enum


class StrEnum(str, Enum):
    """String-valued enum so members serialise cleanly to JSON/SQLite."""

    def __str__(self) -> str:  # pragma: no cover - trivial
        return str(self.value)


class EvidenceCategory(StrEnum):
    """Top-level bucket an evidence item belongs to."""

    MARKET = "market"
    FUNDAMENTAL = "fundamental"
    DOCUMENT = "document"
    ESTIMATE = "estimate"
    GUIDANCE = "guidance"


class SourceType(StrEnum):
    """Where a datapoint originally came from."""

    MARKET_DATA = "market_data"
    COMPANY_FILING = "company_filing"
    EARNINGS_RELEASE = "earnings_release"
    EARNINGS_CALL = "earnings_call"
    INVESTOR_PRESENTATION = "investor_presentation"
    COMPANY_ANNOUNCEMENT = "company_announcement"
    NEWS = "news"
    SELL_SIDE_CONSENSUS = "sell_side_consensus"
    INDUSTRY_RESEARCH = "industry_research"
    COMPETITOR_FILING = "competitor_filing"
    DERIVED = "derived"


class Confidence(StrEnum):
    """How much weight downstream stages should put on a datapoint."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


class ClaimType(StrEnum):
    """Epistemic status of a statement made by a segment agent.

    Segment agents must label every finding so the synthesis layer and the
    reader can tell a reported number from an inference.
    """

    CONFIRMED_FACT = "confirmed_fact"
    MANAGEMENT_STATEMENT = "management_statement"
    MARKET_EXPECTATION = "market_expectation"
    CALCULATED_OBSERVATION = "calculated_observation"
    INTERPRETATION = "interpretation"


class FactType(StrEnum):
    """Epistemic class of an evidence observation, before prose is written."""

    REPORTED_FACT = "reported_fact"
    DERIVED_FACT = "derived_fact"
    MANAGEMENT_STATEMENT = "management_statement"
    EXTERNAL_FORECAST = "external_forecast"
    ANALYST_OPINION = "analyst_opinion"
    MODEL_INTERPRETATION = "model_interpretation"


class EvidenceStatus(StrEnum):
    """Validation state. Only VALIDATED canonical rows are publishable facts."""

    VALIDATED = "validated"
    CONFLICTED = "conflicted"
    UNVERIFIED = "unverified"
    REJECTED = "rejected"


class SegmentName(StrEnum):
    """The segment agents implemented in this prototype."""

    COMPANY_SNAPSHOT = "company_snapshot"
    FINANCIAL_PERFORMANCE = "financial_performance"
    OPERATING_DRIVERS = "operating_drivers"
    RECENT_DEVELOPMENTS = "recent_developments"
    VALUATION = "valuation"
    COMPETITIVE_LANDSCAPE = "competitive_landscape"
    RISKS_CATALYSTS = "risks_catalysts"
    WHAT_MATTERS_NEXT = "what_matters_next"


class ReportSection(StrEnum):
    """Sections a user may request; also the render order of the PDF."""

    COMPANY_SNAPSHOT = "company_snapshot"
    KEY_TAKEAWAYS = "key_takeaways"
    RECENT_DEVELOPMENTS = "recent_developments"
    FINANCIALS = "financials"
    OPERATING_DRIVERS = "operating_drivers"
    COMPETITIVE_LANDSCAPE = "competitive_landscape"
    VALUATION = "valuation"
    RISKS = "risks"
    CATALYSTS = "catalysts"
    WHAT_MATTERS_NEXT = "what_matters_next"
    #: Not requestable and never in DEFAULT_SECTIONS - built after synthesis,
    #: only when live web research (see pipeline.gap_research) fills a
    #: disclosed data gap. See Synthesizer/apply_gap_research for how it is
    #: inserted into an already-finished ReportDraft.
    WEB_RESEARCH = "web_research"
    SOURCES = "sources"


DEFAULT_SECTIONS: tuple[ReportSection, ...] = (
    ReportSection.COMPANY_SNAPSHOT,
    ReportSection.KEY_TAKEAWAYS,
    ReportSection.RECENT_DEVELOPMENTS,
    ReportSection.FINANCIALS,
    ReportSection.OPERATING_DRIVERS,
    ReportSection.COMPETITIVE_LANDSCAPE,
    ReportSection.VALUATION,
    ReportSection.RISKS,
    ReportSection.CATALYSTS,
    ReportSection.WHAT_MATTERS_NEXT,
    ReportSection.SOURCES,
)


class ProviderStatus(StrEnum):
    """Outcome of a single acquisition branch."""

    OK = "ok"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class Severity(StrEnum):
    """QA finding severity. CRITICAL blocks PDF generation."""

    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class RunStatus(StrEnum):
    SUCCEEDED = "succeeded"
    SUCCEEDED_WITH_WARNINGS = "succeeded_with_warnings"
    FAILED_QA = "failed_qa"
    FAILED = "failed"
