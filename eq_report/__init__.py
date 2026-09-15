"""An automated equity research report generation prototype.

The package is organised as one module per pipeline stage:

    domain/         typed models shared by every stage
    planning/       the Research Planner
    providers/      market data, fundamentals and document providers
    acquisition/    the three concurrent acquisition branches
    normalisation/  canonical metrics, units, dates, the normaliser
    evidence/       the Evidence Store and its read-only reader
    analytics/      pure calculations and the deterministic engine
    agents/         the segment agents and their runner
    synthesis/      the synthesis layer
    qa/             the QA checks and engine
    rendering/      PDF and JSON output
    pipeline/       the orchestrator and run tracking
"""

from .config import Settings
from .domain.request import ResearchRequest
from .pipeline.orchestrator import ReportResult, generate_report, generate_report_sync

__all__ = [
    "ReportResult",
    "ResearchRequest",
    "Settings",
    "generate_report",
    "generate_report_sync",
]

__version__ = "0.1.0"
