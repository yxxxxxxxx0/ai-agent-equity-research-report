"""Company fiscal-calendar resolution, for ArcticDB rows dated by calendar
quarter-end rather than labelled with the company's own fiscal period.

Resolution goes through three layers, each one only used when the layer
before it can't answer:

1. Real data - ``bbg-company-details`` carries a ``fiscal_year_end`` field
   (e.g. ``"01/2026"`` for NVIDIA) for the tickers it covers. This is an
   authoritative source, not a guess, and is always tried first.
2. The model - for a ticker ``bbg-company-details`` doesn't cover, one small
   JSON call asks directly what month the company's fiscal year ends in.
   The answer is validated to be an integer 1-12 and cached for the run.
3. Calendar-aligned (month 12) - the fallback whenever neither of the above
   is available or usable. A lookup failure degrades the label; it never
   blocks ingestion.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Protocol

from ...config import ModelConfig
from ...llm.client import LLMJSONResponse, OpenRouterJSONClient
from ...llm.usage import UsageTracker
from ...logging_setup import get_logger, log_event
from .client import ArcticDBConnection

logger = get_logger("providers.arcticdb.fiscal_period")

_COMPANY_DETAILS_LIBRARY = "bbg-company-details"

#: Calendar-aligned fiscal year, used when no other layer can answer - the
#: same assumption this provider made before fiscal-calendar resolution
#: existed.
_DEFAULT_FISCAL_YEAR_END_MONTH = 12

_SYSTEM_PROMPT = """You answer one factual question about a public company's reporting
calendar, from general knowledge. Return JSON only, matching the requested schema."""


class FiscalCalendarLLMClient(Protocol):
    async def complete_json(
        self, system_prompt: str, user_prompt: str, *, stage: str = "",
    ) -> LLMJSONResponse: ...


class FiscalCalendarResolver:
    """Resolves and caches each ticker's fiscal year-end month for one run."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        connection: ArcticDBConnection | None = None,
        client: FiscalCalendarLLMClient | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self._connection = connection
        self._client = client
        if (self._client is None and model_config and model_config.enabled
                and model_config.provider.lower() == "openrouter"):
            self._client = OpenRouterJSONClient(model_config, tracker=tracker)
        self._cache: dict[str, int] = {}

    async def fiscal_year_end_month(self, ticker: str, company: str) -> int:
        if ticker not in self._cache:
            self._cache[ticker] = self._from_company_details(ticker) or await self._ask(
                ticker, company)
        return self._cache[ticker]

    def _from_company_details(self, ticker: str) -> int | None:
        if self._connection is None:
            return None
        try:
            df = self._connection.read(_COMPANY_DETAILS_LIBRARY, ticker)
        except Exception as exc:  # noqa: BLE001 - fall through to the next layer
            log_event(logger, logging.WARNING, "bbg-company-details lookup failed",
                      ticker=ticker, error=f"{type(exc).__name__}: {exc}")
            return None
        if df is None or df.empty or "fiscal_year_end" not in df.columns:
            return None
        raw = df.iloc[0].get("fiscal_year_end")
        match = re.match(r"^\s*(\d{1,2})\s*/\s*\d{4}\s*$", str(raw or ""))
        if not match:
            return None
        month = int(match.group(1))
        if not 1 <= month <= 12:
            return None
        log_event(logger, logging.INFO, "fiscal year-end resolved from bbg-company-details",
                  ticker=ticker, month=month)
        return month

    async def _ask(self, ticker: str, company: str) -> int:
        if self._client is None:
            return _DEFAULT_FISCAL_YEAR_END_MONTH
        user_prompt = (
            f"What calendar month does {company} ({ticker})'s fiscal year end in?\n"
            'Return this JSON shape: {"fiscal_year_end_month": "integer 1-12"}'
        )
        try:
            response = await self._client.complete_json(
                _SYSTEM_PROMPT, user_prompt, stage="fiscal_period")
            month = int(response.payload["fiscal_year_end_month"])
        except Exception as exc:  # noqa: BLE001 - a lookup failure degrades, never blocks
            log_event(
                logger, logging.WARNING,
                "fiscal year-end lookup failed; assuming a calendar fiscal year",
                ticker=ticker, error=f"{type(exc).__name__}: {exc}",
            )
            return _DEFAULT_FISCAL_YEAR_END_MONTH
        if not 1 <= month <= 12:
            log_event(
                logger, logging.WARNING,
                "fiscal year-end lookup out of range; assuming a calendar fiscal year",
                ticker=ticker, value=month,
            )
            return _DEFAULT_FISCAL_YEAR_END_MONTH
        log_event(logger, logging.INFO, "fiscal year-end resolved by the model",
                  ticker=ticker, month=month)
        return month


def quarter_label(period_end: dt.date, fiscal_year_end_month: int) -> str:
    """"FY{year} Q{quarter}" for a period-end date, given the fiscal year-end month.

    With ``fiscal_year_end_month=12`` this reduces to a plain calendar
    quarter, so the same formula covers both calendar- and offset-fiscal-year
    companies without a separate code path.
    """
    month, year = period_end.month, period_end.year
    fiscal_year = year + 1 if month > fiscal_year_end_month else year
    months_after_fye = (month - fiscal_year_end_month - 1) % 12 + 1
    quarter = (months_after_fye - 1) // 3 + 1
    return f"FY{fiscal_year} Q{quarter}"
