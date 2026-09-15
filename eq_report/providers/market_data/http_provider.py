"""Real market-data provider over a generic REST endpoint.

This is the seam for a real vendor. It is only enabled when
``EQR_MARKET_DATA_API_KEY`` and ``EQR_MARKET_DATA_BASE_URL`` are configured; the
registry falls back to the mock provider otherwise, so the prototype runs out of
the box without credentials.

It expects the endpoint to return a flat JSON object of metric -> value, which
is the shape most quote APIs offer. Adapting to a specific vendor means changing
``_parse`` only - normalisation, canonical metric names and evidence handling
are all downstream and unchanged.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode

import requests

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawObservation, SourceRef
from ...domain.plan import ResearchPlan
from ...errors import ProviderError
from ..base import MarketDataProvider


class HttpMarketDataProvider(MarketDataProvider):
    """Fetches a quote snapshot from a configurable REST endpoint."""

    name = "http_market_data"
    is_mock = False

    def is_available(self) -> bool:
        creds = self.settings.credentials
        return bool(creds.market_data_api_key and creds.market_data_base_url)

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        if not self.is_available():
            return self.skipped("Market-data credentials are not configured.")
        if not plan.ticker:
            return self.skipped("No ticker resolved; cannot query the market-data endpoint.")

        # requests is synchronous; run it off the event loop so the three
        # acquisition branches still overlap.
        payload = await asyncio.to_thread(self._get_quote, plan.ticker)
        return self.ok(self._parse(payload, plan))

    # -- internals -------------------------------------------------------
    def _get_quote(self, ticker: str) -> dict[str, Any]:
        creds = self.settings.credentials
        base = str(creds.market_data_base_url).rstrip("/")
        query = urlencode({"symbol": ticker})
        url = f"{base}/quote?{query}"
        try:
            response = requests.get(
                url,
                headers={"Authorization": f"Bearer {creds.market_data_api_key}"},
                timeout=self.settings.provider_timeout_seconds,
            )
        except requests.RequestException as exc:
            raise ProviderError(self.name, f"request failed: {exc}") from exc

        if response.status_code != 200:
            raise ProviderError(
                self.name, f"endpoint returned HTTP {response.status_code}"
            )
        try:
            body = response.json()
        except json.JSONDecodeError as exc:
            raise ProviderError(self.name, f"response was not valid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise ProviderError(self.name, "expected a JSON object of metric -> value")
        return body

    def _parse(self, payload: dict[str, Any], plan: ResearchPlan) -> tuple[RawObservation, ...]:
        """Vendor-specific step: turn the response body into raw observations.

        Metric names are passed through verbatim - the normalisation layer owns
        the mapping to canonical ids.
        """
        as_of = payload.get("as_of") or payload.get("date")
        source = SourceRef(
            source_id=f"{self.name}:{plan.ticker}",
            source_name=str(self.settings.credentials.market_data_base_url),
            source_type=SourceType.MARKET_DATA,
            source_url=None,
            is_mock=False,
        )
        observations: list[RawObservation] = []
        for metric, value in payload.items():
            if metric in {"as_of", "date", "symbol"}:
                continue
            if isinstance(value, (dict, list)):
                continue  # nested structures need a vendor-specific reader
            observations.append(
                RawObservation(
                    metric=metric,
                    value=value,
                    source=source,
                    currency=payload.get("currency") or "USD",
                    as_of=as_of if isinstance(as_of, str) else None,
                    company=plan.company,
                    ticker=plan.ticker,
                    confidence=Confidence.HIGH,
                )
            )
        return tuple(observations)
