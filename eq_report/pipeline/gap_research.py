"""Best-effort live web research to fill the report's own disclosed data gaps.

Runs once, after synthesis, only when a model is configured *and*
``EQR_RESEARCH_DATA_GAPS`` is enabled (see ``config.Settings.research_data_gaps``
- it is a genuine extra cost per run, so it needs its own opt-in rather than
riding along with the model key). For each of the report's leading data gaps
it asks the model - with OpenRouter's web-search plugin turned on - to find a
real, dated, source-linked answer.

A gap is only accepted if the model returns an actual URL and a factual
answer; anything without one is treated as "not found" rather than trusted,
on the same principle used everywhere else in this pipeline: an unsupported
claim is a data gap, not a finding. This is what keeps a web-search-backed
stage from quietly starting to invent citations the moment a real source
cannot be found.

Accepted answers are written to the Evidence Store as ordinary, non-mock
evidence, so they carry the same provenance and QA scrutiny as everything
else the report cites - see :func:`apply_gap_research`, which turns them into
one addendum ``ReportSectionDraft`` and removes the now-filled gaps from the
report's disclosed gap list, rather than folding them silently into whichever
section first raised the gap (that would require re-running the segment
agents and synthesis over the new evidence, a much larger change).
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import replace
from typing import Any

from ..config import ModelConfig
from ..domain.enums import ClaimType, Confidence, EvidenceCategory, ReportSection, SourceType
from ..domain.evidence import EvidenceItem, make_evidence_id
from ..domain.report import Citation, ReportDraft, ReportSectionDraft, Statement
from ..domain.segment import DataGap
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event

logger = get_logger("pipeline.gap_research")

#: Bounds the cost of one run: this many extra model calls (each with the web
#: plugin enabled) at most, however many gaps the report disclosed.
MAX_GAPS_RESEARCHED = 8

_SYSTEM_PROMPT = """You are a research assistant with live web search, helping fill a named
gap in a company research report. Search the web for a real, current, dated source that
addresses the gap. Reply only with what you can attribute to an actual page you found: never
invent a fact, a publisher name, a URL, or a date. If you cannot find a specific source that
addresses the gap, set found=false rather than guessing or answering from general knowledge.
Return JSON only, matching the requested schema."""


async def research_gaps(
    report_run_id: str, company: str, ticker: str | None, report_date: dt.date,
    gaps: tuple[DataGap, ...], model_config: ModelConfig | None,
    *, tracker: UsageTracker | None = None, max_gaps: int = MAX_GAPS_RESEARCHED,
) -> tuple[EvidenceItem, ...]:
    """Evidence items for the gaps a real, source-linked answer was found for.

    Never raises: a failed or empty search here must not affect the rest of
    the run, since this is disclosure-improving, not report-critical.
    """
    if model_config is None or not model_config.enabled or not gaps:
        return ()
    client = OpenRouterJSONClient(model_config, tracker=tracker)
    schema = {
        "found": "true or false",
        "answer": "a factual paragraph, only if found is true, using only what the source states",
        "source_name": "the publisher or site name, only if found is true",
        "source_url": "the exact URL of the page you found this on, only if found is true",
        "published_date": "YYYY-MM-DD if stated on the page, else null",
    }
    items: list[EvidenceItem] = []
    for gap in gaps[:max_gaps]:
        prompt = (
            f"Company: {company} ({ticker or 'ticker unknown'})\n"
            f"Report as-of date: {report_date.isoformat()}\n"
            f"Data gap: {gap.description}\n"
            f"Why it matters: {gap.impact}\n"
            f"Return this JSON shape: {schema}"
        )
        try:
            response = await client.complete_json(
                _SYSTEM_PROMPT, prompt, web_search=True, stage="gap_research")
        except Exception as exc:  # noqa: BLE001 - one failed search must not sink the run
            log_event(logger, logging.WARNING, "gap research call failed",
                      gap=gap.description[:80], error=f"{type(exc).__name__}: {exc}")
            continue
        item = _item_from_response(report_run_id, company, ticker, gap, response.payload)
        if item is not None:
            items.append(item)

    log_event(logger, logging.INFO, "gap research complete",
              gaps_attempted=min(len(gaps), max_gaps), gaps_filled=len(items))
    return tuple(items)


def _item_from_response(
    report_run_id: str, company: str, ticker: str | None, gap: DataGap, payload: Any,
) -> EvidenceItem | None:
    if not isinstance(payload, dict) or not payload.get("found"):
        return None
    url = str(payload.get("source_url", "")).strip()
    answer = str(payload.get("answer", "")).strip()
    # A citation with no real URL is indistinguishable from an invented one,
    # so it is rejected rather than trusted - the same rule the rest of the
    # pipeline applies to any claim with no resolvable evidence id.
    if not answer or not (url.startswith("http://") or url.startswith("https://")):
        return None
    published = None
    raw_date = payload.get("published_date")
    if raw_date:
        try:
            published = dt.date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            published = None
    source_name = str(payload.get("source_name", "")).strip() or "Web search result"
    return EvidenceItem(
        evidence_id=make_evidence_id(report_run_id, "gap_research", gap.description, url),
        report_run_id=report_run_id,
        company=company,
        ticker=ticker,
        category=EvidenceCategory.DOCUMENT,
        source_id="web_search",
        source_name=source_name,
        source_type=SourceType.NEWS,
        retrieved_at=dt.datetime.now(dt.UTC),
        claim_text=answer,
        document_title=source_name,
        published_at=published,
        source_url=url,
        confidence=Confidence.MEDIUM,
        is_mock=False,
        metadata={"fills_gap": gap.description},
    )


# ---------------------------------------------------------------------------
# Folding the results into a finished ReportDraft
# ---------------------------------------------------------------------------

def apply_gap_research(draft: ReportDraft, filled: tuple[EvidenceItem, ...]) -> ReportDraft:
    """Return a new draft with an addendum section for ``filled`` and those
    gaps struck from the report's disclosed gap list.

    Each filled evidence item must already be written to the Evidence Store
    (``store.save``) before this runs, since QA re-resolves every evidence id
    a statement cites against the store, not against this function's return
    value.
    """
    if not filled:
        return draft

    filled_descriptions = {str(item.metadata.get("fills_gap", "")) for item in filled}
    next_ref = max((c.ref_number for c in draft.citations), default=0) + 1
    new_citations: list[Citation] = []
    new_statements: list[Statement] = []
    for item in filled:
        ref = next_ref
        next_ref += 1
        new_citations.append(Citation(
            ref_number=ref, evidence_id=item.evidence_id, text=item.citation(),
            source_url=item.source_url, is_mock=False,
        ))
        new_statements.append(Statement(
            text=str(item.claim_text or ""),
            claim_type=ClaimType.CONFIRMED_FACT,
            evidence_ids=(item.evidence_id,),
            citation_refs=(ref,),
            confidence=item.confidence,
        ))

    addendum = ReportSectionDraft(
        section=ReportSection.WEB_RESEARCH,
        title="Additional Research (Web-Verified)",
        summary=(
            f"{len(filled)} disclosed data gap(s) resolved by live web search, "
            "each carrying a real, dated source not supplied by the configured providers."
        ),
        statements=tuple(new_statements),
    )

    sections: list[ReportSectionDraft] = []
    for section in draft.sections:
        if section.section is not ReportSection.SOURCES:
            sections.append(section)
            continue
        remaining_gaps = tuple(
            g for g in section.data_gaps if g.description not in filled_descriptions)
        sections.append(replace(
            section,
            data_gaps=remaining_gaps,
            summary=(
                f"{len(draft.citations) + len(new_citations)} sources cited; "
                f"{len(remaining_gaps)} data gaps recorded."
            ),
        ))
        sections.append(addendum)

    remaining_top_gaps = tuple(
        g for g in draft.data_gaps if g.description not in filled_descriptions)

    return replace(
        draft,
        sections=tuple(sections),
        citations=(*draft.citations, *new_citations),
        data_gaps=remaining_top_gaps,
    )
