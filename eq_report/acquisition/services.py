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
    def item_count(self) -> int:
        return len(self.observations) + len(self.passages)

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "status": self.status.value,
            "item_count": self.item_count,
            "providers": [r.to_dict() for r in self.provider_results],
            "errors": [e.to_dict() for e in self.errors],
        }


class AcquisitionService:
    """Runs one branch's MegadataAPI provider and records any failure."""

    branch = "unknown"

    def __init__(self, providers: Sequence[DataProvider]) -> None:
        self.providers = tuple(providers)

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

        log_event(
            logger, logging.INFO, "acquisition branch complete",
            branch=self.branch, status=acquisition.status.value,
            observations=len(acquisition.observations), passages=len(acquisition.passages),
            providers=[r.provider_name for r in acquisition.provider_results],
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
    return (
        MarketDataService(registry.market_data_providers()),
        FundamentalsService(registry.fundamentals_providers()),
        DocumentsService(registry.documents_providers()),
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
