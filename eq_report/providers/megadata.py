"""Plan-driven providers for the local MegadataAPI service."""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import timedelta
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
    def is_available(self) -> bool:
        return self.settings.credentials.has_megadata()

    async def _get(self, endpoint: str, params: dict[str, str]) -> tuple[Any, str]:
        return await asyncio.to_thread(self._get_sync, endpoint, params)

    def _get_sync(self, endpoint: str, params: dict[str, str]) -> tuple[Any, str]:
        base = str(self.settings.credentials.megadata_base_url).rstrip("/")
        url = f"{base}{endpoint}"
        headers = {"accept": "application/json"}
        auth: tuple[str, str] | None = None
        username = self.settings.credentials.megadata_username
        password = self.settings.credentials.megadata_password
        if username and password:
            auth = (username, password)
        api_key = self.settings.credentials.megadata_api_key
        if api_key and auth is None:
            headers["Authorization"] = f"Bearer {api_key}"
        try:
            response = requests.get(
                url, params=params, headers=headers, auth=auth,
                timeout=self.settings.provider_timeout_seconds,
            )
            response.raise_for_status()
            return response.json(), f"{url}?{urlencode(params)}"
        except (requests.RequestException, json.JSONDecodeError) as exc:
            raise ProviderError(self.name, f"GET {endpoint} failed: {exc}") from exc

    async def _retrieve_many(
        self, requests_: tuple[DataRequest, ...], plan: ResearchPlan,
    ) -> list[tuple[DataRequest, Any, str] | BaseException]:
        # The LAN-hosted Flask deployment serves several Arctic-backed routes
        # serially. Bursting a full research plan at it causes otherwise-fast
        # calls to queue until every request hits its timeout. Keep calls
        # sequential within each branch; the three acquisition branches still
        # run concurrently, giving bounded concurrency without overwhelming it.
        out: list[tuple[DataRequest, Any, str] | BaseException] = []
        for original in requests_:
            req = _bounded_request(original, plan)
            try:
                payload, url = await self._get(req.endpoint, req.params)
                if req.endpoint.startswith("/api/bbg/") and not _has_useful_payload(payload):
                    symbols = req.params.get("symbols", "")
                    qualified = ",".join(
                        symbol if " " in symbol.strip() else f"{symbol.strip()} US Equity"
                        for symbol in symbols.split(",") if symbol.strip()
                    )
                    if qualified and qualified != symbols:
                        retry = replace(req, params={**req.params, "symbols": qualified})
                        payload, url = await self._get(retry.endpoint, retry.params)
                out.append((req, payload, url))
            except BaseException as exc:  # preserve per-request degradation
                out.append(exc)
        return out


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
        for result in await self._retrieve_many(requests_, plan):
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
            self._retrieve_many(data, plan), self._search_many(searches, plan))
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
        self, searches: tuple[SearchRequest, ...], plan: ResearchPlan,
    ) -> list[tuple[SearchRequest, Any, str] | BaseException]:
        out: list[tuple[SearchRequest, Any, str] | BaseException] = []
        for req in searches:
            try:
                if req.endpoint == "/api/news/data":
                    params = {
                        "keyword": req.query, "size": "10",
                        "start_date": (plan.request.report_date - timedelta(days=365)).isoformat(),
                        "end_date": plan.request.report_date.isoformat(),
                    }
                else:
                    params = {"query": req.query}
                payload, url = await self._get(req.endpoint, params)
                out.append((req, payload, url))
            except BaseException as exc:
                out.append(exc)
        return out


def _bounded_request(request: DataRequest, plan: ResearchPlan) -> DataRequest:
    """Apply safe, report-oriented bounds to model-generated API requests."""
    params = dict(request.params)
    report_date = plan.request.report_date
    if request.endpoint == "/api/bbg/ohlcv/data":
        params.setdefault("trading_days", "120")
    dated = {
        "/api/bbg/market-cap/data": 730,
        "/api/market/bbg/ohlcv": 180,
        "/api/bbg/indicators/data": 1095,
        "/api/bbg/estimates/data": 730,
        "/api/bbg/segment-revenue/data": 1095,
        "/api/bbg/implied-move/data": 730,
        "/api/news/filings": 730,
        "/api/news/filings-by-form": 730,
        "/api/alpha-vantage/earning-call-transcripts": 730,
        "/api/alpha-vantage/earning-call-historical": 730,
    }
    if request.endpoint in dated:
        params.setdefault("from_date", (report_date - timedelta(days=dated[request.endpoint])).isoformat())
        params.setdefault("to_date", report_date.isoformat())
    if request.endpoint == "/api/bbg/supply-chain/data":
        params.setdefault("start_date", (report_date - timedelta(days=730)).isoformat())
        params.setdefault("end_date", report_date.isoformat())
        try:
            params["depth"] = str(min(15, max(1, int(params.get("depth", "2")))))
        except ValueError:
            params["depth"] = "2"
    return replace(request, params=params)


def _has_useful_payload(payload: Any) -> bool:
    if payload is None or payload == {} or payload == []:
        return False
    if isinstance(payload, dict):
        values = list(payload.values())
        if values and all(isinstance(value, dict) and set(value) <= {"error"} for value in values):
            return False
    return True


def _is_date_key(key: str) -> bool:
    """True for a dict key shaped like "YYYY-MM-DD" (a date-keyed series)."""
    return len(key) >= 10 and key[4:5] == "-" and key[7:8] == "-" and key[:4].isdigit()


def _flatten_observations(
    payload: Any, plan: ResearchPlan, request: DataRequest, url: str
) -> tuple[RawObservation, ...]:
    source = SourceRef(
        source_id=f"megadata:{request.request_id}", source_name="MegadataAPI",
        source_type=SourceType.MARKET_DATA if request.branch == "market_data"
        else SourceType.COMPANY_FILING,
        source_url=url, retrieval_provider="MegaAPI", retrieval_url=url,
    )
    out: list[RawObservation] = []

    # Tickers this specific request could plausibly key its payload by -
    # the primary ticker, its peers/benchmark from the plan, and whatever
    # symbols the request itself asked for. A leaf under any other key (e.g.
    # a peer the plan didn't know about) falls back to plan.ticker, so this
    # must be derived per-request rather than hardcoded: a literal set drawn
    # from one company's peers silently mislabels every other company's peer
    # data as the primary ticker's.
    known_tickers = {plan.ticker, plan.benchmark, *plan.peers}
    known_tickers.update(
        s.strip() for s in request.params.get("symbols", "").split(",") if s.strip())
    known_tickers = frozenset(t.upper() for t in known_tickers if t)

    def walk(value: Any, path: tuple[str, ...], record_date: str | None = None) -> None:
        if isinstance(value, dict):
            # A day-indexed record (e.g. one entry of a daily OHLCV list) carries
            # its own date as a sibling field rather than in the JSON path. Pull
            # it out once per record and thread it down, instead of leaving
            # every field in that record dateless (the previous behaviour: the
            # date-in-path heuristic below only ever finds a date that was
            # already a path *key*, never a sibling value like this).
            own_date = next(
                (str(value[k])[:10] for k in ("date", "datetime", "timestamp")
                 if isinstance(value.get(k), str) and len(str(value[k])) >= 10
                 and str(value[k])[4:5] == "-"),
                record_date,
            )
            for key, child in value.items():
                if key in ("date", "datetime", "timestamp"):
                    continue  # metadata for the record, not a metric in its own right
                if _is_date_key(key):
                    # A date-keyed dict is the same shape problem as a
                    # day-indexed list (e.g. fundamentals returned as
                    # {"2026-01-22": {"revenue": ..., "eps": ...}}), just with
                    # the date as a dict key instead of a list index or a
                    # sibling field. Treat it the same way: it supplies the
                    # date, not a piece of the metric's name, or "revenue"
                    # would become a distinct fake metric per reporting date
                    # and never match a canonical alias.
                    walk(child, path, str(key)[:10])
                else:
                    walk(child, (*path, str(key)), own_date)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, (*path, str(index)), record_date)
        elif value is not None and not isinstance(value, bool):
            # A ticker or a bare list index carries no metric identity of its
            # own - the ticker is already tracked separately below, and an
            # index is positional, not a name. Left in, either one turns a
            # single field (e.g. Bloomberg's PX_HIGH) into a distinct
            # "metric" per ticker/day, which can never match a canonical
            # alias and floods the Evidence Store with one-off names instead
            # of one real metric with many dated observations.
            name_parts = [p for p in path if p.upper() not in known_tickers and not p.isdigit()]
            metric = ".".join(name_parts[-3:]) or request.request_id
            date = record_date or next(
                (part for part in reversed(path) if len(part) >= 10 and part[4:5] == "-"), None)
            path_ticker = next(
                (part.upper() for part in path if str(part).upper() in known_tickers),
                plan.ticker)
            metadata: dict[str, Any] = {
                "request_id": request.request_id, "purpose": request.purpose,
                "json_path": ".".join(path),
            }
            if record_date:
                # This leaf came from one dated record in a list (a day of an
                # OHLCV-style series) rather than a one-off flat field. Tag it
                # so EvidenceReader.series()/daily_highs()/daily_lows() can
                # pull the whole dated history for a metric, the same way
                # price_history() already does for "price_history".
                metadata["series"] = "daily_ohlc"
            out.append(RawObservation(
                metric=metric, value=value, source=source, as_of=date,
                company=plan.company if path_ticker == plan.ticker else str(path_ticker),
                ticker=path_ticker, confidence=Confidence.HIGH,
                metadata=metadata,
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
        text = (record.get("text") or record.get("content") or record.get("summary")
                or record.get("story_content"))
        if not text:
            text = json.dumps(record, ensure_ascii=False, default=str)
        story_id = record.get("story_id")
        original_url = record.get("url") or (
            str(story_id).rsplit(":", 1)[0]
            if story_id and str(story_id).rsplit(":", 1)[-1].isdigit()
            else story_id
        )
        original_name = (record.get("source") or record.get("publisher")
                         or record.get("source_code"))
        published = (record.get("published_at") or record.get("published")
                     or record.get("date") or record.get("created_at"))
        source = SourceRef(
            source_id=f"megadata:{request.request_id}:{index}",
            source_name=str(original_name or "MegaAPI retrieved source"),
            source_type=SourceType.NEWS,
            source_url=str(original_url) if original_url else None,
            retrieval_provider="MegaAPI", retrieval_url=url,
            original_source_name=str(original_name) if original_name else None,
            original_source_url=str(original_url) if original_url else None,
            original_publication_date=str(published) if published else None,
        )
        out.append(RawDocumentPassage(
            title=str(record.get("title") or record.get("headline") or request.purpose or request.request_id),
            source=source,
            published_at=published,
            text=str(text), company=plan.company, ticker=plan.ticker,
            section=record.get("section"), speaker=record.get("speaker"),
            confidence=Confidence.MEDIUM,
            metadata={"request_id": request.request_id, "purpose": request.purpose},
        ))
    return tuple(out)
