"""Best-effort live web research to fill a section MegadataAPI left thin.

Runs once, after the Evidence Store holds MegadataAPI's normalised evidence
and before analysis or synthesis touch it - only when a model is configured
*and* ``EQR_WEB_FILL_GAPS`` is enabled (see ``config.Settings.web_fill_gaps``;
a genuine per-run cost, so it needs its own opt-in, separate from
``pipeline.freshness_check``'s and ``qa.auditor``'s).

The design mirrors the auditor's stance on live web verification: an LLM
proposing a fact is never itself trusted. Every candidate here goes through
two independent gates before it becomes an Evidence Store row a segment agent
can cite:

1. It must resolve to a real URL. A domain on the configured allow-list
   (regulatory filings, company newswires, major financial press) or
   recognisable as the subject company's own by name is fast-tracked; there
   is no way to list every company's real domain in advance, though (a
   ticker like "AAPL" shares no letters with "apple.com"), so anything else
   is not rejected here - it is simply not yet trusted.
2. A second, independent model call must confirm - via its own live search,
   not a re-read of the first call's answer - that the URL is genuine and
   its content actually supports the claim. For a candidate that skipped the
   fast track, this call is also where source legitimacy itself gets
   decided: the verifier is asked to confirm the source is the company's own
   official site or a reputable outlet, never a blog, forum or content farm.
   A claim that fails either check is dropped, never written.

Accepted claims are written as ordinary ``EvidenceCategory.DOCUMENT`` rows,
so they flow through the exact same path as MegadataAPI's own document
evidence: a segment agent's shared document pool
(``EvidenceReader.documents``/``documents_matching``) picks them up, cites
them like anything else, and the report's Sources section prints their real
URL - nothing downstream needs to know a claim originated here rather than
from a provider.
"""

from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from ..config import ModelConfig
from ..domain.enums import (
    Confidence,
    EvidenceCategory,
    EvidenceStatus,
    FactType,
    SegmentName,
    SourceType,
)
from ..domain.evidence import EvidenceItem, make_evidence_id
from ..evidence.reader import EvidenceReader
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event

logger = get_logger("pipeline.web_gap_fill")

_SEARCH_SYSTEM_PROMPT = """You are a conservative equity-research assistant with live web
search, filling specific factual gaps a primary data provider left empty. For each topic
given, search only reputable, official or major-press sources - company filings, company
press releases or investor-relations pages, and major financial news wires - never a blog,
forum, social media post, or aggregator. Reply only with what you can attribute to an actual
page you found: never invent a fact, a publisher name, a URL, or a date. If you cannot find a
specific, dated source for a topic, omit it rather than guessing. Each claim must be a single,
short, factual statement (one sentence) - not a summary of the whole page. Return JSON only,
matching the requested schema."""

_VERIFY_SYSTEM_PROMPT = """You are a conservative fact-checker with live web search,
independently verifying a claim someone else already proposed. Search the web yourself and
confirm whether the given URL is a real, reachable page whose content actually supports the
given claim and date. Do not simply trust the claim - if your own search cannot confirm it,
or finds the source says something materially different, set verified=false. Return JSON
only, matching the requested schema."""

#: Bounds the cost of one run regardless of how many topics are thin.
_MAX_TOPICS = 4

#: A topic's need, in the search agent's own words.
_TOPICS: dict[str, str] = {
    "company_snapshot": (
        "A brief factual overview: what the company does, its main business "
        "lines or segments, and its headquarters/industry."
    ),
    "financials": (
        "The most recently reported quarterly or annual revenue, net income, "
        "or EPS figure, with the fiscal period it covers."
    ),
    "operating_drivers": (
        "A specific operating metric or business driver management has "
        "recently disclosed (e.g. user/customer counts, capacity, unit "
        "volumes, segment performance)."
    ),
    "catalysts": (
        "A specific, dated upcoming event: a product launch, regulatory "
        "decision, earnings date, or other near-term development the "
        "company or credible press has disclosed."
    ),
}

#: Which segment agent each topic's evidence belongs to (see
#: EvidenceReader.gap_fill_documents and agents/llm_agent.py's use of it) -
#: the same ReportSection -> SegmentName mapping synthesis.synthesizer uses.
TOPIC_SEGMENTS: dict[str, SegmentName] = {
    "company_snapshot": SegmentName.COMPANY_SNAPSHOT,
    "financials": SegmentName.FINANCIAL_PERFORMANCE,
    "operating_drivers": SegmentName.OPERATING_DRIVERS,
    "catalysts": SegmentName.RISKS_CATALYSTS,
}


@dataclass(frozen=True, slots=True)
class WebGapFillResult:
    """What one gap-fill pass did. ``attempted=False`` means it did not run -
    not that MegadataAPI's evidence was confirmed sufficient."""

    attempted: bool
    topics_checked: tuple[str, ...] = ()
    claims_proposed: int = 0
    claims_verified: int = 0
    claims_added: int = 0
    rejected_reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "topics_checked": list(self.topics_checked),
            "claims_proposed": self.claims_proposed,
            "claims_verified": self.claims_verified,
            "claims_added": self.claims_added,
            "rejected_reasons": list(self.rejected_reasons),
        }


def _thin_topics(reader: EvidenceReader) -> list[str]:
    """Which topics MegadataAPI left too little evidence to write about.

    Deliberately conservative: a topic is only "thin" when the specific
    evidence that segment needs is genuinely absent, so a well-served report
    never pays for a search it does not need. "Any document exists at all"
    is not that signal - a run can hold dozens of general news items with
    nothing specific to a given topic, so each check below tests for the
    concrete anchor that topic's segment actually needs instead.
    """
    thin: list[str] = []
    latest = reader.latest_reported_period()
    if reader.numeric("market_cap") is None and reader.numeric("share_price") is None:
        thin.append("company_snapshot")
    if not latest or reader.numeric("revenue", latest) is None:
        thin.append("financials")
    if not latest or (not reader.segment_rows(latest) and not reader.kpi_rows(latest)):
        thin.append("operating_drivers")
    # No reliable evidence-side signal distinguishes "no upcoming events
    # exist" from "MegaAPI just didn't surface any" - so catalysts is always
    # a candidate; the search-then-verify gates below are what keep this
    # from fabricating one where none is real.
    thin.append("catalysts")
    return thin


#: Company-name words too generic to safely match a domain against (a
#: legal-form suffix, or a word short enough to false-positive on an
#: unrelated site).
_GENERIC_COMPANY_WORDS = {
    "the", "inc", "incorporated", "corp", "corporation", "co", "company",
    "group", "holdings", "holding", "limited", "ltd", "plc", "llc", "sa",
    "ag", "nv", "se", "class", "common", "stock", "shares",
}


def _is_company_domain(netloc: str, company: str) -> bool:
    """Best-effort check that a domain is the subject company's own site.

    A company's own investor-relations or newsroom page is at least as
    reputable a source for its own numbers as a third-party wire, but there
    is no fixed list of "every company's official domain" to allow-list
    ahead of time - so this matches the domain's registrable label against a
    distinctive word from the company's name instead (e.g. "apple.com" and
    "investor.apple.com" both match "Apple Inc."). The match is exact, not
    substring, so an unrelated site that merely contains the word (e.g.
    "appleinsider.com") is not mistaken for the company's own domain. This
    is a heuristic, not a security boundary: the independent verification
    pass still has to confirm the claim itself before anything is written.
    """
    labels = netloc.split(".")
    if len(labels) < 2:
        return False
    registrable = labels[-2]
    words = [
        w.lower() for w in re.findall(r"[A-Za-z]{4,}", company)
        if w.lower() not in _GENERIC_COMPANY_WORDS
    ]
    return registrable in words


def _domain_allowed(url: str, allowed_domains: tuple[str, ...], company: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower()
    except ValueError:
        return False
    netloc = netloc.split(":")[0]
    if any(netloc == d or netloc.endswith(f".{d}") for d in allowed_domains):
        return True
    return _is_company_domain(netloc, company)


def _source_type_for(url: str, company: str) -> SourceType:
    netloc = urlparse(url).netloc.lower()
    if "sec.gov" in netloc:
        return SourceType.COMPANY_FILING
    if any(w in netloc for w in ("prnewswire", "businesswire", "globenewswire")):
        return SourceType.COMPANY_ANNOUNCEMENT
    if _is_company_domain(netloc, company):
        return SourceType.COMPANY_ANNOUNCEMENT
    return SourceType.NEWS


def _parse_date(raw: Any) -> dt.date | None:
    if not raw:
        return None
    try:
        return dt.date.fromisoformat(str(raw)[:10])
    except ValueError:
        return None


async def fill_evidence_gaps(
    report_run_id: str,
    company: str,
    ticker: str | None,
    reader: EvidenceReader,
    model_config: ModelConfig | None,
    *,
    allowed_domains: tuple[str, ...],
    max_claims: int,
    tracker: UsageTracker | None = None,
    client: Any | None = None,
) -> tuple[WebGapFillResult, list[EvidenceItem]]:
    """Best-effort: never raises. Returns the evidence items to persist
    alongside a summary of what happened, so the caller decides how to
    write/log/disclose it. ``client`` is an injection point for tests; the
    normal caller leaves it unset and gets a real ``OpenRouterJSONClient``."""
    if client is None:
        if model_config is None or not model_config.enabled or max_claims <= 0:
            return WebGapFillResult(attempted=False), []
        client = OpenRouterJSONClient(model_config, tracker=tracker)
    elif max_claims <= 0:
        return WebGapFillResult(attempted=False), []

    topics = _thin_topics(reader)[:_MAX_TOPICS]
    if not topics:
        return WebGapFillResult(attempted=False), []
    schema = {
        "claims": [{
            "topic": "exactly one of: " + "|".join(sorted(topics)),
            "claim_text": "one short factual sentence",
            "source_name": "the publisher or site name",
            "source_url": "the exact URL of the page you found this on",
            "published_date": "YYYY-MM-DD the source states, if any",
        }],
    }
    prompt = (
        f"Company: {company} ({ticker or 'ticker unknown'})\n"
        f"Topics needing source-backed facts: "
        f"{[{'topic': t, 'need': _TOPICS[t]} for t in topics]}\n"
        "For each topic, return every distinct, source-backed fact you can find that "
        "addresses it - up to 3 per topic - so the topic has enough independent material "
        "to write from; never invent extra claims just to reach that count, and a topic "
        "with only one real source-backed fact should return only that one. "
        f"Return this JSON shape: {schema}"
    )
    try:
        response = await client.complete_json(
            _SEARCH_SYSTEM_PROMPT, prompt, web_search=True, stage="web_gap_fill_search")
    except Exception as exc:  # noqa: BLE001 - best-effort, must not sink the run
        log_event(logger, logging.WARNING, "web gap-fill search failed",
                  error=f"{type(exc).__name__}: {exc}")
        return WebGapFillResult(attempted=True, topics_checked=tuple(topics)), []

    payload = response.payload if isinstance(response.payload, dict) else {}
    raw_claims = payload.get("claims", []) if isinstance(payload.get("claims"), list) else []

    candidates: list[dict[str, Any]] = []
    rejected: list[str] = []
    for row in raw_claims[:max_claims]:
        if not isinstance(row, dict):
            continue
        raw_topic = row.get("topic", "")
        # Defensive: a model has been observed echoing the schema's own
        # enum-hint list back as the value instead of picking one member.
        if isinstance(raw_topic, list):
            raw_topic = next((t for t in raw_topic if t in topics), "")
        topic = str(raw_topic).strip()
        claim_text = str(row.get("claim_text", "")).strip()
        url = str(row.get("source_url", "")).strip()
        source_name = str(row.get("source_name", "")).strip()
        if topic not in topics or not claim_text or not source_name:
            rejected.append(f"{topic or 'unknown'}: incomplete candidate")
            continue
        if not url.startswith(("http://", "https://")):
            rejected.append(f"{topic}: no real URL returned")
            continue
        candidates.append({
            "topic": topic, "claim_text": claim_text, "source_name": source_name,
            "source_url": url, "published_date": row.get("published_date"),
            # A domain on the curated list, or recognisable as the subject
            # company's own site by name, needs no further scrutiny; anything
            # else still gets a chance, but the independent verifier below is
            # asked to confirm it is a legitimate source in its own right -
            # a static list or a name-matching heuristic can't know every
            # company's real domain (a ticker like "AAPL" shares no letters
            # with "apple.com"), but a live search knows what apple.com is.
            "domain_prevalidated": _domain_allowed(url, allowed_domains, company),
        })

    verified = await _verify_candidates(client, company, ticker, candidates)

    items = [
        _to_evidence_item(report_run_id, company, ticker, candidate)
        for candidate in verified
    ]
    log_event(
        logger, logging.INFO, "web gap-fill complete",
        topics=topics, proposed=len(candidates), verified=len(verified),
    )
    result = WebGapFillResult(
        attempted=True, topics_checked=tuple(topics),
        claims_proposed=len(candidates), claims_verified=len(verified),
        claims_added=len(items), rejected_reasons=tuple(rejected),
    )
    return result, items


async def _verify_candidates(
    client: Any, company: str, ticker: str | None,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """The second, independent check: each candidate is only kept if a fresh
    web search - not a re-read of the first call's own answer - confirms the
    URL is real and actually supports the claim. For a candidate whose
    domain isn't already on the curated allow-list or recognisable as the
    company's own by name, this call is also where source legitimacy itself
    gets decided - by a live search, not a static list."""
    schema = {
        "verified": "true or false",
        "confirmed_published_date": "YYYY-MM-DD if the page states one, else null",
        "note": "one short sentence on what you found",
    }
    legitimacy_instruction = (
        "Also confirm this source itself is legitimate: either the subject company's own "
        "official site (investor relations, newsroom, or a regulatory filing host), or a "
        "major, reputable financial/business news publisher. If it is a blog, forum, "
        "content farm, SEO aggregator, or any site you cannot confirm is one of those, set "
        "verified=false regardless of whether the claim text itself seems accurate.\n"
    )
    confirmed: list[dict[str, Any]] = []
    for candidate in candidates:
        prompt = (
            f"Company: {company} ({ticker or 'ticker unknown'})\n"
            f"Claim to verify: {candidate['claim_text']}\n"
            f"Claimed source: {candidate['source_name']} <{candidate['source_url']}>\n"
            f"{'' if candidate.get('domain_prevalidated') else legitimacy_instruction}"
            f"Return this JSON shape: {schema}"
        )
        try:
            response = await client.complete_json(
                _VERIFY_SYSTEM_PROMPT, prompt, web_search=True, stage="web_gap_fill_verify")
        except Exception as exc:  # noqa: BLE001 - one failed check must not sink the rest
            log_event(logger, logging.WARNING, "web gap-fill verification call failed",
                      topic=candidate["topic"], error=f"{type(exc).__name__}: {exc}")
            continue
        payload = response.payload if isinstance(response.payload, dict) else {}
        if not payload.get("verified"):
            continue
        confirmed_date = payload.get("confirmed_published_date") or candidate["published_date"]
        confirmed.append({**candidate, "published_date": confirmed_date})
    return confirmed


def _to_evidence_item(
    report_run_id: str, company: str, ticker: str | None, candidate: dict[str, Any],
) -> EvidenceItem:
    url = candidate["source_url"]
    published = _parse_date(candidate.get("published_date"))
    return EvidenceItem(
        evidence_id=make_evidence_id(
            report_run_id, "web_gap_fill", candidate["topic"], url, candidate["claim_text"]),
        report_run_id=report_run_id,
        company=company,
        ticker=ticker,
        category=EvidenceCategory.DOCUMENT,
        source_id=f"web:{urlparse(url).netloc}",
        source_name=candidate["source_name"],
        source_type=_source_type_for(url, company),
        retrieved_at=dt.datetime.now(dt.timezone.utc),
        claim_text=candidate["claim_text"],
        source_url=url,
        original_source_url=url,
        original_source_name=candidate["source_name"],
        published_at=published,
        as_of=published,
        retrieval_provider="web_gap_fill",
        fact_type=FactType.REPORTED_FACT,
        status=EvidenceStatus.VALIDATED,
        confidence=Confidence.MEDIUM,
        metadata={"gap_fill_topic": candidate["topic"]},
    )
