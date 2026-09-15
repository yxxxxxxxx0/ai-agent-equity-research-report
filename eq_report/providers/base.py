"""Stage 3 - provider interfaces for the three acquisition branches.

A provider's only job is to return raw, provenance-tagged observations for a
plan. It does not normalise, does not calculate and does not write to the
Evidence Store. Providers are async so the three branches (and multiple
providers within a branch) can run concurrently.
"""

from __future__ import annotations

import abc
import time
from typing import Any

from ..config import Settings
from ..domain.enums import ProviderStatus
from ..domain.observation import ProviderResult, RawDocumentPassage, RawObservation
from ..domain.plan import ResearchPlan


class DataProvider(abc.ABC):
    """Common behaviour for every provider."""

    #: Stable identifier used in logs, run metadata and evidence source ids.
    name: str = "unnamed"
    #: One of "market_data" | "fundamentals" | "documents".
    branch: str = "unknown"
    #: True when the provider returns illustrative sample data rather than
    #: real observations. Mock data is labelled all the way to the PDF.
    is_mock: bool = False

    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    @abc.abstractmethod
    def is_available(self) -> bool:
        """Whether this provider can run given the current configuration."""

    async def fetch(self, plan: ResearchPlan) -> ProviderResult:
        """Run the provider, timing it and converting exceptions into a result.

        Subclasses implement :meth:`_fetch`; this wrapper guarantees the branch
        service always receives a ProviderResult rather than an exception, which
        is what lets one failing source degrade instead of killing the run.
        """
        started = time.perf_counter()
        try:
            payload = await self._fetch(plan)
        except Exception as exc:  # noqa: BLE001 - deliberate: providers never raise upward
            return ProviderResult(
                provider_name=self.name,
                branch=self.branch,
                status=ProviderStatus.FAILED,
                errors=(f"{type(exc).__name__}: {exc}",),
                is_mock=self.is_mock,
                duration_ms=(time.perf_counter() - started) * 1000,
            )
        return ProviderResult(
            provider_name=self.name,
            branch=self.branch,
            status=payload.status,
            observations=payload.observations,
            passages=payload.passages,
            errors=payload.errors,
            warnings=payload.warnings,
            is_mock=self.is_mock,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    @abc.abstractmethod
    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        """Provider-specific fetch. May raise; the wrapper converts failures."""

    # -- helpers for subclasses -----------------------------------------
    def ok(
        self,
        observations: tuple[RawObservation, ...] = (),
        passages: tuple[RawDocumentPassage, ...] = (),
        *,
        warnings: tuple[str, ...] = (),
        errors: tuple[str, ...] = (),
    ) -> ProviderResult:
        status = ProviderStatus.PARTIAL if errors else ProviderStatus.OK
        if not observations and not passages:
            status = ProviderStatus.FAILED if errors else ProviderStatus.PARTIAL
        return ProviderResult(
            provider_name=self.name,
            branch=self.branch,
            status=status,
            observations=observations,
            passages=passages,
            warnings=warnings,
            errors=errors,
            is_mock=self.is_mock,
        )

    def skipped(self, reason: str) -> ProviderResult:
        return ProviderResult(
            provider_name=self.name,
            branch=self.branch,
            status=ProviderStatus.SKIPPED,
            warnings=(reason,),
            is_mock=self.is_mock,
        )

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "branch": self.branch,
            "is_mock": self.is_mock,
            "available": self.is_available(),
        }


class MarketDataProvider(DataProvider):
    """Share price, market cap, trading data, multiples, consensus, estimates."""

    branch = "market_data"


class FundamentalsProvider(DataProvider):
    """Reported financials, segment detail, KPIs, guidance and consensus."""

    branch = "fundamentals"


class DocumentsProvider(DataProvider):
    """Filings, releases, transcripts, presentations, announcements and news."""

    branch = "documents"
