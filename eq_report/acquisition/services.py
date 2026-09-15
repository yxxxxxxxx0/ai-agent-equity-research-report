"""Stage 3 - the three parallel acquisition branches.

Each branch service owns one or more providers, runs them concurrently, and
returns an AcquisitionResult. A branch never raises: a dead provider becomes a
recorded failure and the run continues with whatever the other branches
returned. That is what makes the pipeline degrade gracefully.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Sequence

from ..config import Settings
from ..domain.enums import ProviderStatus
from ..domain.observation import ProviderResult, RawDocumentPassage, RawObservation
from ..domain.plan import ResearchPlan
from ..errors import PipelineError
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from ..providers.base import DataProvider
from ..providers.registry import ProviderRegistry

logger = get_logger("acquisition")


@dataclass(frozen=True, slots=True)
class AcquisitionResult:
    """Everything one branch produced, plus how each provider fared."""

    branch: str
    observations: tuple[RawObservation, ...] = ()
    passages: tuple[RawDocumentPassage, ...] = ()
    provider_results: tuple[ProviderResult, ...] = ()
    errors: tuple[PipelineError, ...] = ()

    @property
    def status(self) -> ProviderStatus:
        if not self.provider_results:
            return ProviderStatus.FAILED
        statuses = {r.status for r in self.provider_results}
        if statuses == {ProviderStatus.OK}:
            return ProviderStatus.OK
        if ProviderStatus.OK in statuses or ProviderStatus.PARTIAL in statuses:
            return ProviderStatus.PARTIAL
        if statuses == {ProviderStatus.SKIPPED}:
            return ProviderStatus.SKIPPED
        return ProviderStatus.FAILED

    @property
    def used_mock_data(self) -> bool:
        return any(r.is_mock and r.item_count for r in self.provider_results)

    @property
    def item_count(self) -> int:
        return len(self.observations) + len(self.passages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "status": self.status.value,
            "item_count": self.item_count,
            "used_mock_data": self.used_mock_data,
            "providers": [r.to_dict() for r in self.provider_results],
            "errors": [e.to_dict() for e in self.errors],
        }


class AcquisitionService:
    """Base class: runs a branch's providers concurrently and merges the output.

    ``fallback`` is a mock provider tried when the *configured* real
    provider(s) ran without raising but still returned zero items (e.g. an
    unreachable host that the provider itself catches and reports as an
    error rather than an exception - see ``MegadataMarketProvider``). This is
    distinct from - and in addition to - ``ProviderRegistry`` already
    choosing a mock provider outright when no real provider is configured at
    all: this branch keeps "use mock data when the API can't be reached"
    true at runtime, not only at startup.
    """

    branch = "unknown"

    def __init__(
        self, providers: Sequence[DataProvider], *,
        fallback: DataProvider | None = None, allow_mock: bool = True,
    ) -> None:
        self.providers = tuple(providers)
        self.fallback = fallback
        self.allow_mock = allow_mock

    async def fetch(self, plan: ResearchPlan) -> AcquisitionResult:
        if not self.providers:
            return AcquisitionResult(
                branch=self.branch,
                errors=(PipelineError(
                    stage=f"acquisition.{self.branch}",
                    kind="NoProviderConfigured",
                    message=f"No provider is configured for the {self.branch} branch.",
                ),),
            )

        results = await asyncio.gather(
            *(provider.fetch(plan) for provider in self.providers),
            return_exceptions=True,
        )

        observations: list[RawObservation] = []
        passages: list[RawDocumentPassage] = []
        provider_results: list[ProviderResult] = []
        errors: list[PipelineError] = []

        for provider, result in zip(self.providers, results, strict=True):
            if isinstance(result, BaseException):
                # DataProvider.fetch already traps exceptions; reaching here means
                # something escaped the wrapper, so record it rather than hide it.
                errors.append(PipelineError.from_exception(
                    f"acquisition.{self.branch}", result, provider=provider.name))
                provider_results.append(ProviderResult(
                    provider_name=provider.name, branch=self.branch,
                    status=ProviderStatus.FAILED,
                    errors=(f"{type(result).__name__}: {result}",),
                    is_mock=provider.is_mock,
                ))
                continue

            provider_results.append(result)
            observations.extend(result.observations)
            passages.extend(result.passages)
            for message in result.errors:
                errors.append(PipelineError(
                    stage=f"acquisition.{self.branch}",
                    kind="ProviderError", message=message,
                    context={"provider": provider.name},
                ))
            for message in result.warnings:
                log_event(logger, logging.WARNING, "provider warning",
                          branch=self.branch, provider=provider.name, detail=message)

        acquisition = AcquisitionResult(
            branch=self.branch,
            observations=tuple(observations),
            passages=tuple(passages),
            provider_results=tuple(provider_results),
            errors=tuple(errors),
        )

        used_real_provider = any(not p.is_mock for p in self.providers)
        if acquisition.item_count == 0 and used_real_provider and self.fallback is not None:
            if not self.allow_mock:
                errors.append(PipelineError(
                    stage=f"acquisition.{self.branch}",
                    kind="NoDataAndMockDisabled",
                    message=(
                        f"The configured {self.branch} provider(s) returned no data "
                        "and mock providers are disabled (EQR_ALLOW_MOCK_PROVIDERS=false)."
                    ),
                ))
                acquisition = AcquisitionResult(
                    branch=self.branch, provider_results=tuple(provider_results),
                    errors=tuple(errors),
                )
            else:
                log_event(
                    logger, logging.WARNING,
                    "real provider returned no data; falling back to mock provider",
                    branch=self.branch,
                    providers=[p.name for p in self.providers],
                )
                mock_result = await self.fallback.fetch(plan)
                acquisition = AcquisitionResult(
                    branch=self.branch,
                    observations=tuple(mock_result.observations),
                    passages=tuple(mock_result.passages),
                    provider_results=(*provider_results, mock_result),
                    errors=tuple(errors),
                )

        log_event(
            logger, logging.INFO, "acquisition branch complete",
            branch=self.branch, status=acquisition.status.value,
            observations=len(acquisition.observations), passages=len(acquisition.passages),
            providers=[r.provider_name for r in acquisition.provider_results],
            used_mock_data=acquisition.used_mock_data,
        )
        return acquisition


class MarketDataService(AcquisitionService):
    branch = "market_data"


class FundamentalsService(AcquisitionService):
    branch = "fundamentals"


class DocumentsService(AcquisitionService):
    branch = "documents"


@dataclass(frozen=True, slots=True)
class AcquisitionBundle:
    """The output of all three branches for one run."""

    market_data: AcquisitionResult
    fundamentals: AcquisitionResult
    documents: AcquisitionResult

    @property
    def branches(self) -> tuple[AcquisitionResult, ...]:
        return (self.market_data, self.fundamentals, self.documents)

    @property
    def used_mock_data(self) -> bool:
        return any(b.used_mock_data for b in self.branches)

    @property
    def errors(self) -> tuple[PipelineError, ...]:
        out: list[PipelineError] = []
        for branch in self.branches:
            out.extend(branch.errors)
        return tuple(out)

    def to_dict(self) -> list[dict[str, Any]]:
        return [b.to_dict() for b in self.branches]


def build_services(
    settings: Settings, tracker: UsageTracker | None = None,
) -> tuple[MarketDataService, FundamentalsService, DocumentsService]:
    """Wire the three branch services from configuration."""
    registry = ProviderRegistry(settings, tracker=tracker)
    allow_mock = settings.allow_mock_providers
    return (
        MarketDataService(
            registry.market_data_providers(),
            fallback=registry.market_data_fallback(), allow_mock=allow_mock),
        FundamentalsService(
            registry.fundamentals_providers(),
            fallback=registry.fundamentals_fallback(), allow_mock=allow_mock),
        DocumentsService(
            registry.documents_providers(),
            fallback=registry.documents_fallback(), allow_mock=allow_mock),
    )


async def acquire_all(
    plan: ResearchPlan,
    market_data: MarketDataService,
    fundamentals: FundamentalsService,
    documents: DocumentsService,
) -> AcquisitionBundle:
    """Run all three branches concurrently."""
    market_result, fundamentals_result, documents_result = await asyncio.gather(
        market_data.fetch(plan),
        fundamentals.fetch(plan),
        documents.fetch(plan),
    )
    return AcquisitionBundle(
        market_data=market_result,
        fundamentals=fundamentals_result,
        documents=documents_result,
    )
