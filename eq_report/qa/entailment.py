"""Semantic claim-to-source verification before publication.

Provenance IDs establish where a sentence came from; this reviewer checks the
stronger requirement that the mapped material actually supports what the
sentence says.  It never repairs prose or changes citations.  A mismatch is a
critical QA finding and therefore blocks publication.
"""

from __future__ import annotations

from typing import Any

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import ReportSection, Severity
from ..domain.qa import QAFinding
from ..domain.report import ReportDraft
from ..evidence.reader import EvidenceReader
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..llm.verify import is_true

_SYSTEM_PROMPT = """You are a conservative citation-entailment reviewer. For each report
claim, decide whether its mapped source excerpts and calculations support the complete wording:
entity, period, direction, magnitude, comparison, causality, and conclusion. A source being
related to the topic is not enough. Mark supported=false if any material part is absent,
contradicted, overstated, or mapped to the wrong source. Do not use outside knowledge. Return
JSON only, matching the requested schema."""

_MAX_CLAIMS = 120
_MAX_EXCERPT_CHARS = 700


async def verify_claim_entailment(
    draft: ReportDraft,
    reader: EvidenceReader,
    analytics: AnalyticsBundle,
    model_config: ModelConfig | None,
    *,
    tracker: UsageTracker | None = None,
    client: Any | None = None,
) -> list[QAFinding]:
    """Return blocking findings for claims not supported by mapped sources.

    When no model is configured the deterministic provenance checks remain in
    force.  Runs with model-authored research use this independent semantic
    pass in addition to those checks.
    """
    if client is None:
        if model_config is None or not model_config.enabled:
            return []
        client = OpenRouterJSONClient(model_config, tracker=tracker)

    analytics_by_id = {item.analytics_id: item for item in analytics.results}
    claims: list[dict[str, Any]] = []
    claim_meta: dict[str, tuple[str, str]] = {}

    for section_index, section in enumerate(draft.sections):
        if section.section is ReportSection.SOURCES:
            continue
        rows: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        if section.summary:
            rows.append((section.summary, section.summary_evidence_ids,
                         section.summary_analytics_ids))
        rows.extend(
            (statement.text, statement.evidence_ids, statement.analytics_ids)
            for statement in section.statements
        )
        for row_index, (text, evidence_ids, analytics_ids) in enumerate(rows):
            claim_id = f"s{section_index}c{row_index}"
            excerpts = []
            for evidence_id in evidence_ids:
                item = reader.get(evidence_id)
                if item is None:
                    continue
                fact = item.claim_text
                if not fact and item.metric:
                    fact = (
                        f"{item.metric}={item.value} {item.unit or ''}; "
                        f"period={item.period_label or item.as_of or 'unspecified'}"
                    )
                excerpts.append({
                    "evidence_id": evidence_id,
                    "source": item.original_source_name or item.source_name,
                    "excerpt": str(fact or item.raw_value or "")[:_MAX_EXCERPT_CHARS],
                })
            calculations = []
            for analytics_id in analytics_ids:
                result = analytics_by_id.get(analytics_id)
                if result is not None:
                    calculations.append({
                        "analytics_id": analytics_id,
                        "metric": result.metric,
                        "value": result.value,
                        "unit": result.unit,
                        "formula": result.formula,
                        "inputs": result.inputs,
                    })
            claims.append({
                "claim_id": claim_id,
                "claim": text,
                "mapped_sources": excerpts,
                "mapped_calculations": calculations,
            })
            claim_meta[claim_id] = (section.section.value, text)

    if not claims:
        return []
    if len(claims) > _MAX_CLAIMS:
        return [QAFinding(
            check="evidence.claim_entailment_capacity",
            severity=Severity.CRITICAL,
            message=f"Report has {len(claims)} claims, exceeding the semantic review cap of {_MAX_CLAIMS}.",
        )]

    schema = {
        "results": [{
            "claim_id": "copied claim_id",
            "supported": "JSON boolean true or false (not a string)",
            "reason": "short, specific explanation",
        }]
    }
    prompt = f"Claims and mapped material:\n{claims}\nReturn this JSON shape: {schema}"
    try:
        response = await client.complete_json(
            _SYSTEM_PROMPT, prompt, stage="qa_claim_entailment")
    except Exception as exc:  # noqa: BLE001 - a failed mandatory review cannot wave claims through
        return [QAFinding(
            check="evidence.claim_entailment_unavailable",
            severity=Severity.CRITICAL,
            message=f"Semantic claim-to-source review failed: {type(exc).__name__}: {exc}",
        )]

    payload = response.payload if isinstance(response.payload, dict) else {}
    reviewed: set[str] = set()
    findings: list[QAFinding] = []
    for row in payload.get("results", []):
        if not isinstance(row, dict):
            continue
        claim_id = str(row.get("claim_id", ""))
        if claim_id not in claim_meta or claim_id in reviewed:
            continue
        reviewed.add(claim_id)
        if is_true(row.get("supported")):
            continue
        section, text = claim_meta[claim_id]
        findings.append(QAFinding(
            check="evidence.claim_not_entailed",
            severity=Severity.CRITICAL,
            message=("Mapped source does not support the complete claim: "
                     + str(row.get("reason") or "no reason supplied")),
            section=section,
            subject=text[:200],
            details={"claim_id": claim_id},
        ))

    for claim_id in claim_meta.keys() - reviewed:
        section, text = claim_meta[claim_id]
        findings.append(QAFinding(
            check="evidence.claim_not_reviewed",
            severity=Severity.CRITICAL,
            message="Semantic reviewer did not return a verdict for this claim.",
            section=section,
            subject=text[:200],
            details={"claim_id": claim_id},
        ))
    return findings
