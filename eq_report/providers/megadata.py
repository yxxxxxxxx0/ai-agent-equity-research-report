"""Plan-driven providers for the local MegadataAPI service."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.parse import urlencode

import requests

from ..domain.enums import Confidence, SourceType
from ..domain.observation import (
    ProviderResult,
    RawDocumentPassage,
    RawObservation,
    SourceRef,
)
from ..domain.plan import DataRequest, ResearchPlan, SearchRequest
from ..errors import ProviderError
from .base import DocumentsProvider, FundamentalsProvider, MarketDataProvider


class _MegadataMixin:
    name = "megadata"
    is_mock = False

    def is_available(self) -> bool:
        return self.settings.credentials.has_megadata()

    async def _get(self, endpoint: str, params: dict[str, str]) -> tuple[Any, str]:
        return await asyncio.to_thread(self._get_sync, endpoint, params)

    def _get_sync(self, endpoint: str, params: dict[str, str]) -> tuple[Any, str]:
        base = str(self.settings.credentials.megadata_base_url).rstrip("/")
        url = f"{base}{endpoint}"
        headers = {"accept": "application/json"}
        api_key = self.settings.credentials.megadata_api_key
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.get(
                url, params=params, headers=headers,
                timeout=self.settings.provider_timeout_seconds,
            )
            response.raise_for_status()
            return response.json(), f"{url}?{urlencode(params)}"
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise ProviderError(self.name, f"GET {endpoint} failed: {exc}") from exc

    async def _retrieve_many(
        self, requests_: tuple[DataRequest, ...]
    ) -> list[tuple[DataRequest, Any, str] | BaseException]:
        results = await asyncio.gather(
            *(self._get(req.endpoint, req.params) for req in requests_),
            return_exceptions=True,
        )
        return [
            result if isinstance(result, BaseException) else (req, result[0], result[1])
            for req, result in zip(requests_, results, strict=True)
        ]


class MegadataMarketProvider(_MegadataMixin, MarketDataProvider):
    name = "megadata_market"

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        requests_ = tuple(r for r in plan.data_requests if r.branch == self.branch)
        return await self._observations(plan, requests_)

    async def _observations(
        self, plan: ResearchPlan, requests_: tuple[DataRequest, ...]
    ) -> ProviderResult:
        observations: list[RawObservation] = []
        errors: list[str] = []
        for result in await self._retrieve_many(requests_):
            if isinstance(result, BaseException):
                errors.append(str(result))
                continue
            req, payload, url = result
            observations.extend(_flatten_observations(payload, plan, req, url))
        return self.ok(tuple(observations), errors=tuple(errors))


class MegadataFundamentalsProvider(MegadataMarketProvider, FundamentalsProvider):
    name = "megadata_fundamentals"
    branch = "fundamentals"


class MegadataDocumentsProvider(_MegadataMixin, DocumentsProvider):
    name = "megadata_documents"
    branch = "documents"

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        data = tuple(r for r in plan.data_requests if r.branch == self.branch)
        searches = plan.search_requests
        data_results, search_results = await asyncio.gather(
            self._retrieve_many(data), self._search_many(searches))
        passages: list[RawDocumentPassage] = []
        errors: list[str] = []
        for result in [*data_results, *search_results]:
            if isinstance(result, BaseException):
                errors.append(str(result))
                continue
            req, payload, url = result
            passages.extend(_extract_passages(payload, plan, req, url))
        return self.ok(passages=tuple(passages), errors=tuple(errors))

    async def _search_many(
        self, searches: tuple[SearchRequest, ...]
    ) -> list[tuple[SearchRequest, Any, str] | BaseException]:
        results = await asyncio.gather(
            *(self._get(req.endpoint, {"query": req.query}) for req in searches),
            return_exceptions=True,
        )
        return [
            result if isinstance(result, BaseException) else (req, result[0], result[1])
            for req, result in zip(searches, results, strict=True)
        ]


def _flatten_observations(
    payload: Any, plan: ResearchPlan, request: DataRequest, url: str
) -> tuple[RawObservation, ...]:
    source = SourceRef(
        source_id=f"megadata:{request.request_id}", source_name="MegadataAPI",
        source_type=SourceType.MARKET_DATA if request.branch == "market_data"
        else SourceType.COMPANY_FILING,
        source_url=url,
    )
    out: list[RawObservation] = []

    def walk(value: Any, path: tuple[str, ...]) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(child, (*path, str(key)))
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)))
        elif value is not None and not isinstance(value, bool):
            metric = ".".join(path[-3:]) or request.request_id
            date = next((part for part in reversed(path) if len(part) >= 10 and part[4:5] == "-"), None)
            out.append(RawObservation(
                metric=metric, value=value, source=source, as_of=date,
                company=plan.company, ticker=plan.ticker, confidence=Confidence.HIGH,
                metadata={"request_id": request.request_id, "purpose": request.purpose,
                          "json_path": ".".join(path)},
            ))

    walk(payload, ())
    return tuple(out)


def _extract_passages(
    payload: Any, plan: ResearchPlan, request: DataRequest | SearchRequest, url: str
) -> tuple[RawDocumentPassage, ...]:
    records = payload if isinstance(payload, list) else (
        payload.get("data") or payload.get("results") or [payload]
        if isinstance(payload, dict) else []
    )
    if isinstance(records, dict):
        records = list(records.values())
    out: list[RawDocumentPassage] = []
    for index, record in enumerate(records if isinstance(records, list) else []):
        if not isinstance(record, dict):
            continue
        text = record.get("text") or record.get("content") or record.get("summary")
        if not text:
            text = json.dumps(record, ensure_ascii=False, default=str)
        source_url = record.get("url") or url
        source = SourceRef(
            source_id=f"megadata:{request.request_id}:{index}",
            source_name=str(record.get("source") or record.get("publisher") or "MegadataAPI"),
            source_type=SourceType.NEWS,
            source_url=str(source_url),
        )
        out.append(RawDocumentPassage(
            title=str(record.get("title") or request.purpose or request.request_id),
            source=source,
            published_at=record.get("published_at") or record.get("published") or record.get("date"),
            text=str(text), company=plan.company, ticker=plan.ticker,
            section=record.get("section"), speaker=record.get("speaker"),
            confidence=Confidence.MEDIUM,
            metadata={"request_id": request.request_id, "purpose": request.purpose},
        ))
    return tuple(out)
