"""Typed domain models shared by every pipeline stage."""

from .analytics import AnalyticsBundle, AnalyticsResult, make_analytics_id
from .enums import (
    DEFAULT_SECTIONS,
    ClaimType,
    Confidence,
    EvidenceCategory,
    ProviderStatus,
    ReportSection,
    RunStatus,
    SegmentName,
    Severity,
    SourceType,
)
from .evidence import EvidenceItem, FiscalPeriod, make_evidence_id
from .observation import ProviderResult, RawDocumentPassage, RawObservation, SourceRef
from .plan import DocumentRequirement, ResearchPlan, ResearchQuestion, SegmentTask
from .qa import QAFinding, QAResult
from .report import (
    ChartSpec,
    Citation,
    KeyDataGroup,
    KeyDataItem,
    KeyDataPanel,
    MetricRow,
    MetricTable,
    ReportDraft,
    ReportSectionDraft,
    Statement,
)
from .request import ResearchRequest
from .run import ReportRun, StageTiming
from .segment import DataGap, KeyFinding, MetricHighlight, SegmentResult

__all__ = [
    "DEFAULT_SECTIONS",
    "AnalyticsBundle",
    "AnalyticsResult",
    "ChartSpec",
    "Citation",
    "ClaimType",
    "Confidence",
    "DataGap",
    "DocumentRequirement",
    "EvidenceCategory",
    "EvidenceItem",
    "FiscalPeriod",
    "KeyDataGroup",
    "KeyDataItem",
    "KeyDataPanel",
    "KeyFinding",
    "MetricHighlight",
    "MetricRow",
    "MetricTable",
    "ProviderResult",
    "ProviderStatus",
    "QAFinding",
    "QAResult",
    "RawDocumentPassage",
    "RawObservation",
    "ReportDraft",
    "ReportRun",
    "ReportSection",
    "ReportSectionDraft",
    "ResearchPlan",
    "ResearchQuestion",
    "ResearchRequest",
    "RunStatus",
    "SegmentName",
    "SegmentResult",
    "SegmentTask",
    "Severity",
    "SourceRef",
    "SourceType",
    "StageTiming",
    "Statement",
    "make_analytics_id",
    "make_evidence_id",
]
