"""Chooses which providers run for each acquisition branch.

Selection is purely configuration-driven: a real provider is used when its
credentials are present, otherwise the clearly-labelled mock provider is used.
No business logic reads credentials directly. There is no LLM/web-search tier:
a real vendor feed either works or the branch falls straight back to mock data.
"""

from __future__ import annotations

import logging

from ..config import Settings
from ..errors import ConfigurationError
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from .base import DocumentsProvider, FundamentalsProvider, MarketDataProvider
from .documents.mock_provider import MockDocumentsProvider
from .fundamentals.mock_provider import MockFundamentalsProvider
from .market_data.http_provider import HttpMarketDataProvider
from .market_data.mock_provider import MockMarketDataProvider
from .online_sources import OnlineDocumentsProvider, OnlineFundamentalsProvider, OnlineMarketProvider
from .megadata import (
    MegadataDocumentsProvider,
    MegadataFundamentalsProvider,
    MegadataMarketProvider,
)

logger = get_logger("providers.registry")


class ProviderRegistry:
    """Resolves the provider list for each branch."""

    def __init__(self, settings: Settings, *, tracker: UsageTracker | None = None) -> None:
        self.settings = settings
        self._tracker = tracker

    def market_data_providers(self) -> tuple[MarketDataProvider, ...]:
        if self.settings.credentials.has_megadata():
            return (MegadataMarketProvider(self.settings),)
        if self.settings.online_sources:
            return (OnlineMarketProvider(self.settings),)
        real = HttpMarketDataProvider(self.settings)
        if real.is_available():
            return (real,)
        return (self._mock(MockMarketDataProvider(self.settings), "market_data"),)

    def fundamentals_providers(self) -> tuple[FundamentalsProvider, ...]:
        if self.settings.credentials.has_megadata():
            return (MegadataFundamentalsProvider(self.settings),)
        if self.settings.online_sources:
            return (OnlineFundamentalsProvider(self.settings),)
        # No other real fundamentals vendor is wired up yet; see README "Next
        # to productionise". The interface is the extension point.
        return (self._mock(MockFundamentalsProvider(self.settings), "fundamentals"),)

    def documents_providers(self) -> tuple[DocumentsProvider, ...]:
        if self.settings.credentials.has_megadata():
            return (MegadataDocumentsProvider(self.settings),)
        if self.settings.online_sources:
            return (OnlineDocumentsProvider(self.settings),)
        return (self._mock(MockDocumentsProvider(self.settings), "documents"),)

    # -- fallback mock providers -------------------------------------------
    # Used by AcquisitionService when a *configured* real provider (its
    # credentials are present) runs but comes back with no data at all - e.g.
    # unreachable host, empty response. Selection above only picks mock when
    # no real provider is configured in the first place; this covers the
    # "configured but actually failed" case, so "use mock data if the API
    # can't be pulled from" holds at runtime, not just at startup.
    def market_data_fallback(self) -> MarketDataProvider:
        return MockMarketDataProvider(self.settings)

    def fundamentals_fallback(self) -> FundamentalsProvider:
        return MockFundamentalsProvider(self.settings)

    def documents_fallback(self) -> DocumentsProvider:
        return MockDocumentsProvider(self.settings)

    def describe(self) -> list[dict[str, object]]:
        return [
            provider.describe()
            for provider in (
                *self.market_data_providers(),
                *self.fundamentals_providers(),
                *self.documents_providers(),
            )
        ]

    # -- internals -------------------------------------------------------
    def _mock[T](self, provider: T, branch: str) -> T:
        if not self.settings.allow_mock_providers:
            raise ConfigurationError(
                f"No real provider is configured for the {branch} branch and mock "
                f"providers are disabled (EQR_ALLOW_MOCK_PROVIDERS=false)."
            )
        log_event(
            logger, logging.WARNING,
            "falling back to mock provider",
            branch=branch, provider=getattr(provider, "name", "?"),
        )
        return provider
