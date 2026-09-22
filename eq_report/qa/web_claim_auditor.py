"""Re-verifies web-sourced evidence immediately before publication.

``pipeline.web_gap_fill`` already runs a search-then-independently-verify
pair before a claim is written to the Evidence Store at all. This is a
third, later check, run as part of QA rather than acquisition: right before
the report is allowed to publish, every ``web_gap_fill`` claim that actually
made it into the final draft is re-confirmed with one more fresh,
independent web search - catching a fact that has gone stale since
acquisition, or one whose earlier verification was itself wrong - and
demoted out of the report if it can no longer be confirmed.

A demotion does not itself edit the draft: it marks the Evidence Store row
``REJECTED``, and :func:`eq_report.qa.checks.check_web_claims_reverified`
(a deterministic check, so this stays auditable) turns that into a CRITICAL
finding the same repair/omission path already used for every other
evidence-provenance failure handles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from ..config import ModelConfig
from ..domain.enums import EvidenceStatus
from ..domain.report import ReportDraft
from ..evidence.reader import EvidenceReader
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event

logger = get_logger("qa.web_claim_auditor")

_SYSTEM_PROMPT = """You are a conservative fact-checker with live web search, re-verifying a
previously-sourced claim immediately before publication. Search the web yourself right now and
confirm whether the claim is still accurate and the given source still supports it. Do not
assume an earlier verification is still correct - check again, independently. If your own
search cannot confirm it, set verified=false. Return JSON only, matching the requested schema."""

#: Bounds the cost of this recheck regardless of how many web claims a run produced.
_MAX_REVERIFICATIONS = 12


@dataclass(frozen=True, slots=True)
class WebClaimAuditResult:
    checked: int = 0
    rejected: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {"checked": self.checked, "rejected_evidence_ids": list(self.rejected)}


async def reverify_web_claims(
    draft: ReportDraft,
    reader: EvidenceReader,
    model_config: ModelConfig | None,
    *,
    tracker: UsageTracker | None = None,
    client: Any | None = None,
) -> WebClaimAuditResult:
    """Best-effort: never raises. Demotes a claim's Evidence Store row to
    REJECTED when a fresh, independent search can no longer confirm it - the
    caller reruns QA's deterministic checks afterward to see this reflected,
    the same "correct the store, then recheck" pattern qa.auditor uses."""
    cited_ids = {
        evidence_id
        for section in draft.sections
        for statement in section.statements
        for evidence_id in statement.evidence_ids
    }
    if not cited_ids:
        return WebClaimAuditResult()

    web_items = [
        item for evidence_id in cited_ids
        if (item := reader.get(evidence_id)) is not None
        and item.retrieval_provider == "web_gap_fill"
        and item.status is EvidenceStatus.VALIDATED
    ][:_MAX_REVERIFICATIONS]
    if not web_items:
        return WebClaimAuditResult()

    if client is None:
        if model_config is None or not model_config.enabled:
            return WebClaimAuditResult()
        client = OpenRouterJSONClient(model_config, tracker=tracker)

    schema = {"verified": "true or false", "note": "one short sentence on what you found"}
    rejected: list[str] = []
    for item in web_items:
        prompt = (
            f"Company: {item.company} ({item.ticker or 'ticker unknown'})\n"
            f"Claim: {item.claim_text}\n"
            f"Cited source: {item.source_name} <{item.source_url}>\n"
            f"Return this JSON shape: {schema}"
        )
        try:
            response = await client.complete_json(
                _SYSTEM_PROMPT, prompt, web_search=True, stage="qa_web_claim_reverify")
        except Exception as exc:  # noqa: BLE001 - one failed recheck must not sink the rest
            log_event(logger, logging.WARNING, "web claim re-verification call failed",
                      evidence_id=item.evidence_id, error=f"{type(exc).__name__}: {exc}")
            continue
        payload = response.payload if isinstance(response.payload, dict) else {}
        if payload.get("verified"):
            continue
        updated = replace(
            item, status=EvidenceStatus.REJECTED,
            validation_messages=item.validation_messages + (
                "qa: pre-publication re-verification could not confirm this claim "
                f"({payload.get('note') or 'no note'}).",
            ),
        )
        reader.store.replace([updated])
        rejected.append(item.evidence_id)

    log_event(logger, logging.INFO, "web claim re-verification complete",
              checked=len(web_items), rejected=len(rejected))
    return WebClaimAuditResult(checked=len(web_items), rejected=tuple(rejected))
