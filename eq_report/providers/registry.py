"""MegadataAPI-only provider registry."""

from __future__ import annotations

from ..config import Settings
from ..llm.usage import UsageTracker
from .base import DocumentsProvider, FundamentalsProvider, MarketDataProvider
from .megadata import (
    MegadataDocumentsProvider,
    MegadataFundamentalsProvider,
    MegadataMarketProvider,
)


class ProviderRegistry:
    """Expose MegadataAPI for every acquisition branch, with no fallback."""

    def __init__(self, settings: Settings, *, tracker: UsageTracker | None = None) -> None:
        self.settings = settings
        self._tracker = tracker

    def market_data_providers(self) -> tuple[MarketDataProvider, ...]:
        if self.settings.credentials.has_megadata():
            return (MegadataMarketProvider(self.settings),)
        return ()

    def fundamentals_providers(self) -> tuple[FundamentalsProvider, ...]:
        if self.settings.credentials.has_megadata():
            return (MegadataFundamentalsProvider(self.settings),)
        return ()

    def documents_providers(self) -> tuple[DocumentsProvider, ...]:
        if self.settings.credentials.has_megadata():
            return (MegadataDocumentsProvider(self.settings),)
        return ()

    def describe(self) -> list[dict[str, object]]:
        return [
            provider.describe()
            for provider in (
                *self.market_data_providers(),
                *self.fundamentals_providers(),
                *self.documents_providers(),
            )
        ]
