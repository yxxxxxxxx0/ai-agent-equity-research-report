"""Opt-in public-web providers for supported issuers.

These adapters preserve the pipeline boundary: external data becomes raw,
provenance-tagged observations/passages and is then normalised like any other
provider. They are deliberately opt-in because public endpoints have licensing,
rate-limit and availability constraints. Apple is the initial supported issuer.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import html
import re
from typing import Any

import requests

from ..domain.enums import Confidence, SourceType
from ..domain.observation import RawDocumentPassage, RawObservation, SourceRef
from ..domain.plan import ResearchPlan
from .base import DocumentsProvider, FundamentalsProvider, MarketDataProvider

_CIKS = {"AAPL": "0000320193"}
_SEC_HEADERS = {"User-Agent": "EQ Report research@example.com", "Accept-Encoding": "gzip, deflate"}
_SEC_FACTS = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
_SEC_SUBMISSIONS = "https://data.sec.gov/submissions/CIK{cik}.json"
_YAHOO_CHART = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def _supported(plan: ResearchPlan) -> bool:
    return bool(plan.ticker and plan.ticker.upper() in _CIKS)


def _source(name: str, url: str, kind: SourceType) -> SourceRef:
    return SourceRef(source_id=f"online:{name}", source_name=name,
                     source_type=kind, source_url=url, is_mock=False)


class OnlineMarketProvider(MarketDataProvider):
    name = "online_yahoo_market"

    def is_available(self) -> bool:
        return self.settings.online_sources

    async def _fetch(self, plan: ResearchPlan):
        if not _supported(plan):
            return self.skipped("Public-web market source currently supports AAPL only.")
        return self.ok(await asyncio.to_thread(self._load, plan))

    def _load(self, plan: ResearchPlan) -> tuple[RawObservation, ...]:
        end = plan.request.report_date + dt.timedelta(days=1)
        start = end - dt.timedelta(days=430)
        response = requests.get(_YAHOO_CHART.format(ticker=plan.ticker), params={
            "period1": int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp()),
            "period2": int(dt.datetime.combine(end, dt.time(), dt.UTC).timestamp()), "interval": "1d"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=self.settings.provider_timeout_seconds)
        response.raise_for_status()
        body = response.json()["chart"]["result"][0]
        quote, meta = body["indicators"]["quote"][0], body["meta"]
        source = _source("Yahoo Finance daily chart", _YAHOO_CHART.format(ticker=plan.ticker), SourceType.MARKET_DATA)
        out: list[RawObservation] = []
        for i, timestamp in enumerate(body["timestamp"]):
            if quote["close"][i] is None or quote["volume"][i] is None:
                continue
            date = dt.datetime.fromtimestamp(timestamp, dt.UTC).date().isoformat()
            out.extend((
                RawObservation("historical_close", quote["close"][i], source, currency="USD", as_of=date,
                               company="Apple Inc.", ticker=plan.ticker, confidence=Confidence.MEDIUM,
                               metadata={"series": "price_history"}),
                RawObservation("historical_volume", quote["volume"][i], source, unit="count", as_of=date,
                               company="Apple Inc.", ticker=plan.ticker, confidence=Confidence.MEDIUM,
                               metadata={"series": "volume_history"}),
            ))
        if out:
            latest = out[-2]
            out.append(RawObservation("share_price", latest.value, source, currency="USD", as_of=latest.as_of,
                                      company="Apple Inc.", ticker=plan.ticker, confidence=Confidence.MEDIUM))
        market_cap = meta.get("marketCap")
        if market_cap:
            out.append(RawObservation("marketCap", market_cap, source, currency="USD", as_of=out[-1].as_of,
                                      company="Apple Inc.", ticker=plan.ticker, confidence=Confidence.MEDIUM))
        return tuple(out)


class OnlineFundamentalsProvider(FundamentalsProvider):
    name = "online_sec_facts"
    _METRICS = {
        "RevenueFromContractWithCustomerExcludingAssessedTax": "revenue",
        "GrossProfit": "gross_profit", "OperatingIncomeLoss": "operating_income",
        "NetIncomeLoss": "net_income", "EarningsPerShareDiluted": "eps_diluted",
        "NetCashProvidedByUsedInOperatingActivities": "operating_cash_flow",
        "PaymentsToAcquirePropertyPlantAndEquipment": "capex",
        "CashAndCashEquivalentsAtCarryingValue": "cash_and_equivalents",
        "LongTermDebtCurrent": "total_debt",
    }

    def is_available(self) -> bool:
        return self.settings.online_sources

    async def _fetch(self, plan: ResearchPlan):
        if not _supported(plan):
            return self.skipped("Public-web SEC facts source currently supports AAPL only.")
        return self.ok(await asyncio.to_thread(self._load, plan))

    def _load(self, plan: ResearchPlan) -> tuple[RawObservation, ...]:
        url = _SEC_FACTS.format(cik=_CIKS[plan.ticker or ""])
        body = requests.get(url, headers=_SEC_HEADERS, timeout=self.settings.provider_timeout_seconds).json()
        facts = body.get("facts", {}).get("us-gaap", {})
        source = _source("SEC Company Facts", url, SourceType.COMPANY_FILING)
        out: list[RawObservation] = []
        for tag, metric in self._METRICS.items():
            units = facts.get(tag, {}).get("units", {})
            values = next(iter(units.values()), [])
            # The latest reported facts only; the normaliser keeps dates and fiscal labels.
            seen: set[tuple[str, str]] = set()
            for row in reversed(values):
                if row.get("form") not in {"10-Q", "10-K"} or not row.get("fy") or not row.get("fp"):
                    continue
                period = f"FY{row['fy']} {row['fp']}"
                key = (period, str(row.get("end")))
                if key in seen:
                    continue
                seen.add(key)
                out.append(RawObservation(metric, row.get("val"), source, currency="USD",
                    as_of=row.get("filed"), period=period, period_end=row.get("end"), company="Apple Inc.",
                    ticker=plan.ticker, confidence=Confidence.HIGH,
                    metadata={"sec_tag": tag, "form": row.get("form"), "accession": row.get("accn")}))
                if len(seen) == 8:
                    break
        return tuple(out)


class OnlineDocumentsProvider(DocumentsProvider):
    name = "online_sec_documents"

    def is_available(self) -> bool:
        return self.settings.online_sources

    async def _fetch(self, plan: ResearchPlan):
        if not _supported(plan):
            return self.skipped("Public-web SEC documents source currently supports AAPL only.")
        return self.ok(passages=await asyncio.to_thread(self._load, plan))

    def _load(self, plan: ResearchPlan) -> tuple[RawDocumentPassage, ...]:
        cik = _CIKS[plan.ticker or ""]
        url = _SEC_SUBMISSIONS.format(cik=cik)
        recent = requests.get(url, headers=_SEC_HEADERS, timeout=self.settings.provider_timeout_seconds).json()["filings"]["recent"]
        source = _source("SEC EDGAR filings", url, SourceType.COMPANY_FILING)
        out: list[RawDocumentPassage] = []
        for i, form in enumerate(recent["form"]):
            if form not in {"10-Q", "10-K", "8-K"}:
                continue
            accession = recent["accessionNumber"][i].replace("-", "")
            document = recent["primaryDocument"][i]
            filing_url = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{document}"
            text = f"Apple Inc. {form} filed {recent['filingDate'][i]}. Source filing: {filing_url}"
            out.append(RawDocumentPassage(title=f"Apple {form} filing", source=SourceRef(
                source_id=f"sec:{accession}", source_name="SEC EDGAR", source_type=SourceType.COMPANY_FILING,
                source_url=filing_url), published_at=recent["filingDate"][i], text=text,
                company="Apple Inc.", ticker=plan.ticker, confidence=Confidence.HIGH,
                metadata={"form": form, "accession": recent["accessionNumber"][i]}))
            if len(out) == 12:
                break
        return tuple(out)
