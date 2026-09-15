"""ArcticDB-backed documents provider: earnings-call transcripts plus filings.

Two independent sources feed this branch, run concurrently:

- ``alpha-vantage-transcripts`` holds real earnings-call transcripts (speaker,
  turn, content) - this is the primary source for ``EARNINGS_CALL`` evidence,
  and needs no network access since the text is already in ArcticDB.
- ``sec-gov-filings`` holds only filing *metadata* - form type, filing date,
  and a ``file_url`` pointing at the actual document on SEC EDGAR, not the
  filing text itself. This is therefore the one part of this provider that
  talks to the network beyond ArcticDB: it fetches each candidate filing's
  ``file_url`` over HTTP and extracts its prose - HTML via a lightweight,
  dependency-free tag-stripping pass, or PDF via ``pypdf`` when the response
  is one (some issuers only host a PDF rendition).

SEC EDGAR itself rejects automated requests without a descriptive
``User-Agent`` identifying the requester (``EQR_SEC_USER_AGENT``); when that
is not configured, or EDGAR still refuses the request, a company's own
investor-relations mirror of the *same* filing is tried instead, for the
handful of tickers/forms in ``_COMPANY_IR_FALLBACK_URLS``. That table is a
tiny, explicit stopgap - the same shape as the planner's ``_KNOWN_TICKERS``
lookup - covering only each company's current 10-K/10-Q; it is not a general
company-website search, since IR sites have no consistent URL pattern or
listing format to crawl generically.

A failure in either source - one filing's HTTP fetch, or the transcripts
library being unavailable - is recorded as a provider error for that one
item rather than failing the branch; everything else still comes through.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import io
import logging
from html.parser import HTMLParser
from typing import Any

import requests
from pypdf import PdfReader

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawDocumentPassage, SourceRef
from ...domain.plan import ResearchPlan
from ...errors import ProviderError
from ...logging_setup import get_logger, log_event
from ..base import DocumentsProvider
from .client import ArcticDBConnection

logger = get_logger("providers.arcticdb.documents")

_FILINGS_LIBRARY = "sec-gov-filings"
_TRANSCRIPTS_LIBRARY = "alpha-vantage-transcripts"
_MAX_CHARS = 20_000
_DEFAULT_LIMIT = 5
_DEFAULT_LOOKBACK_DAYS = 365
_DEFAULT_TRANSCRIPT_LIMIT = 1
#: A single speaker turn can still run to several thousand words (a CFO's
#: full prepared remarks); capped well below _MAX_CHARS since a transcript
#: passage is meant to be quotable as one excerpt, not a multi-page block.
_MAX_TRANSCRIPT_TURN_CHARS = 1_500
#: Short interjections ("Thank you.", operator boilerplate) add index noise
#: without evidence value.
_MIN_TRANSCRIPT_TURN_CHARS = 40
#: Caps passage count per quarter - a call can run to 60+ turns.
_MAX_TRANSCRIPT_TURNS = 30

#: Filing forms treated as narrative content. Ownership forms (3/4/5) are
#: index noise for this purpose, not prose.
_RELEVANT_FORMS: dict[str, SourceType] = {
    "10-K": SourceType.COMPANY_FILING,
    "10-Q": SourceType.COMPANY_FILING,
    "8-K": SourceType.COMPANY_ANNOUNCEMENT,
}

#: (ticker, form) -> a company-hosted mirror of its *current* filing of that
#: form, tried only when SEC EDGAR itself refuses the request. See the module
#: docstring: this is a narrow, explicit stopgap, not a general lookup.
_COMPANY_IR_FALLBACK_URLS: dict[tuple[str, str], str] = {
    ("NVDA", "10-K"): "https://investor.nvidia.com/files/doc_financials/2026/q4/10K-NVDA.pdf",
    ("NVDA", "10-Q"):
        "https://investor.nvidia.com/files/doc_financials/2026/q3/"
        "13e6981b-95ed-4aac-a602-ebc5865d0590.pdf",
}


class _TextExtractor(HTMLParser):
    """Strips markup, keeping prose text and skipping script/style content."""

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.chunks: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self._skip_depth and data.strip():
            self.chunks.append(data.strip())


def extract_text(html: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(html)
    except Exception as exc:  # noqa: BLE001 - malformed markup should not crash the run
        log_event(logger, logging.WARNING, "filing HTML parse error; using partial text",
                  error=f"{type(exc).__name__}: {exc}")
    return " ".join(parser.chunks)


def extract_pdf_text(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    pages: list[str] = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception as exc:  # noqa: BLE001 - one bad page should not lose the rest
            log_event(logger, logging.WARNING, "filing PDF page extraction error; skipping page",
                      error=f"{type(exc).__name__}: {exc}")
    return " ".join(" ".join(p.split()) for p in pages if p)


class ArcticDBDocumentsProvider(DocumentsProvider):
    name = "arcticdb_documents"
    is_mock = False

    def __init__(self, settings, *, connection: ArcticDBConnection | None = None) -> None:
        super().__init__(settings)
        self._connection = connection or ArcticDBConnection(
            settings.credentials.arcticdb_uri or "")

    def is_available(self) -> bool:
        return bool(self.settings.credentials.arcticdb_uri)

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        if not plan.ticker:
            return self.skipped("No ticker resolved; ArcticDB documents need one.")

        transcripts_task = asyncio.to_thread(self._transcripts, plan)
        filings_task = self._fetch_filings(plan)
        transcripts_result, filings_result = await asyncio.gather(
            transcripts_task, filings_task, return_exceptions=True)

        passages: list[RawDocumentPassage] = []
        errors: list[str] = []
        if isinstance(transcripts_result, BaseException):
            errors.append(f"{_TRANSCRIPTS_LIBRARY}: {transcripts_result}")
        else:
            passages.extend(transcripts_result)
        if isinstance(filings_result, BaseException):
            errors.append(f"{_FILINGS_LIBRARY}: {filings_result}")
        else:
            filing_passages, filing_errors = filings_result
            passages.extend(filing_passages)
            errors.extend(filing_errors)

        if not passages and not errors:
            return self.ok(passages=(), warnings=("No documents found for the ticker.",))
        return self.ok(passages=tuple(passages), errors=tuple(errors))

    # -- earnings-call transcripts (already text, no network needed) --------
    def _transcripts(self, plan: ResearchPlan) -> list[RawDocumentPassage]:
        """One passage per speaker turn, not one blob per whole call.

        A quarter's transcript is dozens of turns; concatenating all of them
        into a single passage produced a "document" of tens of thousands of
        characters that downstream agents would then quote wholesale as one
        finding - which the PDF renderer cannot even lay out on one page.
        Each turn is a properly-sized excerpt on its own, the same shape as
        any other document passage in the system.
        """
        df = self._connection.read(_TRANSCRIPTS_LIBRARY, plan.ticker)
        if df is None or df.empty or "quarter" not in df.columns:
            return []

        limit = self._transcript_limit(plan)
        quarters = sorted((q for q in df["quarter"].dropna().unique()), reverse=True)[:limit]
        passages: list[RawDocumentPassage] = []
        for quarter in quarters:
            rows = df[df["quarter"] == quarter].sort_values("turn_index")
            turns_kept = 0
            for _, row in rows.iterrows():
                if turns_kept >= _MAX_TRANSCRIPT_TURNS:
                    break
                content = str(row.get("content") or "").strip()
                # Short interjections ("Thank you.", operator boilerplate)
                # add index noise without adding evidence value.
                if len(content) < _MIN_TRANSCRIPT_TURN_CHARS:
                    continue
                speaker = str(row.get("speaker") or "").strip() or "Unknown speaker"
                truncated = len(content) > _MAX_TRANSCRIPT_TURN_CHARS
                text = content[:_MAX_TRANSCRIPT_TURN_CHARS]
                turn_index = row.get("turn_index")
                source = SourceRef(
                    source_id=(f"arcticdb:{_TRANSCRIPTS_LIBRARY}:{plan.ticker}:"
                               f"{quarter}:{turn_index}"),
                    source_name="Alpha Vantage (earnings call transcript)",
                    source_type=SourceType.EARNINGS_CALL,
                )
                passages.append(RawDocumentPassage(
                    title=f"{plan.ticker} {quarter} earnings call - {speaker}",
                    source=source, published_at=None, text=text,
                    company=plan.company, ticker=plan.ticker, speaker=speaker,
                    confidence=Confidence.HIGH,
                    metadata={"quarter": str(quarter), "truncated": truncated},
                ))
                turns_kept += 1
        return passages

    @staticmethod
    def _transcript_limit(plan: ResearchPlan) -> int:
        reqs = [r for r in plan.required_documents if r.source_type is SourceType.EARNINGS_CALL]
        if not reqs:
            return _DEFAULT_TRANSCRIPT_LIMIT
        return max(r.max_documents for r in reqs)

    # -- SEC filings (metadata in ArcticDB, text fetched over HTTP) ---------
    async def _fetch_filings(
        self, plan: ResearchPlan
    ) -> tuple[list[RawDocumentPassage], list[str]]:
        df = await asyncio.to_thread(self._connection.read, _FILINGS_LIBRARY, plan.ticker)
        if df is None or df.empty:
            return [], []

        earliest = self._earliest_allowed(plan)
        candidates = [
            row for _, row in df.iterrows()
            if str(row.get("form")) in _RELEVANT_FORMS and _filed_on_or_after(row, earliest)
        ]
        candidates.sort(key=lambda row: row.get("filing_date"), reverse=True)
        candidates = candidates[: self._document_limit(plan)]

        results = await asyncio.gather(
            *(asyncio.to_thread(self._fetch_filing, plan, row) for row in candidates),
            return_exceptions=True,
        )
        passages: list[RawDocumentPassage] = []
        errors: list[str] = []
        for row, result in zip(candidates, results, strict=True):
            if isinstance(result, BaseException):
                errors.append(f"{row.get('file_name', '?')}: {result}")
                continue
            if result is not None:
                passages.append(result)
        return passages, errors

    @staticmethod
    def _document_limit(plan: ResearchPlan) -> int:
        filing_reqs = [
            r for r in plan.required_documents if r.source_type in _RELEVANT_FORMS.values()
        ]
        if not filing_reqs:
            return _DEFAULT_LIMIT
        return max(r.max_documents for r in filing_reqs)

    @staticmethod
    def _earliest_allowed(plan: ResearchPlan) -> dt.date:
        lookback = max(
            (r.lookback_days for r in plan.required_documents), default=_DEFAULT_LOOKBACK_DAYS)
        return plan.request.report_date - dt.timedelta(days=lookback)

    def _fetch_filing(self, plan: ResearchPlan, row: Any) -> RawDocumentPassage | None:
        url = str(row.get("file_url") or "")
        if not url:
            return None
        form = str(row.get("form"))

        try:
            response = self._get(url)
        except ProviderError as primary_error:
            fallback_url = _COMPANY_IR_FALLBACK_URLS.get((str(plan.ticker), form))
            if fallback_url is None:
                raise
            log_event(logger, logging.INFO, "SEC EDGAR fetch failed; trying company IR mirror",
                      ticker=plan.ticker, form=form, error=str(primary_error))
            response = self._get(fallback_url)
            url = fallback_url

        content_type = response.headers.get("content-type", "")
        if "pdf" in content_type.lower() or url.lower().endswith(".pdf"):
            text = extract_pdf_text(response.content)
        else:
            text = extract_text(response.text)
        truncated = len(text) > _MAX_CHARS
        text = text[:_MAX_CHARS]

        filed_iso = _as_of(row.get("filing_date"))
        source = SourceRef(
            source_id=f"arcticdb:{_FILINGS_LIBRARY}:{row.get('accession')}",
            source_name="SEC EDGAR" if url == str(row.get("file_url")) else "Company IR site",
            source_type=_RELEVANT_FORMS.get(form, SourceType.COMPANY_FILING),
            source_url=url,
        )
        return RawDocumentPassage(
            title=f"{plan.ticker} {form} filed {filed_iso}",
            source=source, published_at=filed_iso, text=text,
            company=plan.company, ticker=plan.ticker, confidence=Confidence.HIGH,
            metadata={
                "form": form, "accession": str(row.get("accession")), "truncated": truncated,
            },
        )

    def _get(self, url: str) -> requests.Response:
        headers = {}
        user_agent = self.settings.credentials.sec_user_agent
        if user_agent:
            headers["User-Agent"] = user_agent
        try:
            response = requests.get(
                url, headers=headers, timeout=self.settings.provider_timeout_seconds)
            response.raise_for_status()
        except requests.RequestException as exc:
            raise ProviderError(self.name, f"GET {url} failed: {exc}") from exc
        return response


def _filed_on_or_after(row: Any, earliest: dt.date) -> bool:
    filed = row.get("filing_date")
    if filed is None:
        return False
    filed_date = filed.date() if hasattr(filed, "date") else filed
    return filed_date >= earliest


def _as_of(value: object) -> str | None:
    if value is None:
        return None
    return value.date().isoformat() if hasattr(value, "date") else str(value)
