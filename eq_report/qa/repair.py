"""Bounded, auditable repair of narrative QA failures.

The model may edit prose, but it cannot change evidence, analytics, citations,
or the QA verdict. Every proposed edit is applied to the existing structured
draft and the normal deterministic QA suite is run again by the orchestrator.
If a local narrative error cannot be repaired safely, the offending statement
is omitted; evidence-level and arithmetic failures remain hard blockers.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.qa import QAFinding, QAResult
from ..domain.report import ReportDraft, Statement
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json
from .checks import has_asserted_numeric_fact

_REPAIRABLE_STATEMENT_CHECKS = frozenset({
    "evidence.reference_exists",
    "analytics.reference_exists",
    "evidence.citation_resolves",
    "evidence.no_unsupported_numbers",
    "evidence.claim_supported",
    "evidence.numeric_claim_not_canonical",
})

_REWRITEABLE_CHECKS = frozenset({"evidence.numeric_claim_not_canonical"})

_SYSTEM_PROMPT = """You repair a structured equity-research draft after deterministic QA.
You may only remove unsupported numerical assertions from prose. Preserve the meaning that is
supported by the supplied evidence excerpts, do not add facts, dates, quantities, citations,
or source ids, and do not guess. If a safe useful rewrite is impossible, choose omit. Return
JSON only in the requested shape."""


@dataclass(frozen=True, slots=True)
class RepairEvent:
    section: str
    original_text: str
    action: str
    checks: tuple[str, ...]
    repaired_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "section": self.section,
            "original_text": self.original_text,
            "action": self.action,
            "checks": list(self.checks),
            "repaired_text": self.repaired_text,
        }


@dataclass(frozen=True, slots=True)
class RepairOutcome:
    draft: ReportDraft
    changed: bool
    events: tuple[RepairEvent, ...] = ()
    llm_error: str | None = None


class DraftRepairer:
    """Repair statement-scoped failures without weakening deterministic QA."""

    def __init__(
        self, model_config: ModelConfig | None, *, tracker: UsageTracker | None = None,
        client: Any | None = None,
    ) -> None:
        self._model_config = model_config
        self._tracker = tracker
        self._client = client

    async def repair(
        self, draft: ReportDraft, qa: QAResult, reader: EvidenceReader,
        analytics: AnalyticsBundle,
    ) -> RepairOutcome:
        targets = _statement_targets(draft, qa.critical)
        proposals: dict[int, str] = {}
        llm_error: str | None = None

        rewriteable = [target for target in targets if target[2].issubset(_REWRITEABLE_CHECKS)]
        if rewriteable:
            prompt_rows = [
                _prompt_row(index, target, reader, analytics)
                for index, target in enumerate(rewriteable)
            ]
            response, llm_error = await safe_complete_json(
                self._model_config,
                _SYSTEM_PROMPT,
                "Repair these statements. Return JSON shaped as "
                "{'items':[{'index':0,'action':'rewrite|omit','rewritten_text':'...'}]}.\n"
                f"Statements: {prompt_rows}",
                client=self._client,
                tracker=self._tracker,
                stage="qa_repair",
            )
            if response is not None and isinstance(response.payload, dict):
                for row in response.payload.get("items", []):
                    if not isinstance(row, dict) or row.get("action") != "rewrite":
                        continue
                    try:
                        index = int(row.get("index"))
                    except (TypeError, ValueError):
                        continue
                    text = str(row.get("rewritten_text") or "").strip()
                    if 0 <= index < len(rewriteable) and text \
                            and not has_asserted_numeric_fact(text):
                        proposals[id(rewriteable[index][1])] = text

        repaired, events = _apply_statement_repairs(draft, targets, proposals)
        repaired = _repair_draft_metadata(repaired, qa.critical)
        return RepairOutcome(
            draft=repaired,
            changed=repaired != draft,
            events=events,
            llm_error=llm_error,
        )


def _statement_targets(
    draft: ReportDraft, findings: tuple[QAFinding, ...],
) -> list[tuple[str, Statement, frozenset[str]]]:
    matches: dict[tuple[str, str], tuple[Statement, set[str]]] = {}
    for finding in findings:
        if finding.check not in _REPAIRABLE_STATEMENT_CHECKS \
                or not finding.section or not finding.subject:
            continue
        for section in draft.sections:
            if section.section.value != finding.section:
                continue
            for statement in section.statements:
                if statement.text == finding.subject \
                        or statement.text.startswith(finding.subject) \
                        or finding.subject.startswith(statement.text):
                    key = (section.section.value, statement.text)
                    current, checks = matches.setdefault(key, (statement, set()))
                    checks.add(finding.check)
                    matches[key] = (current, checks)
                    break
    return [
        (section, statement, frozenset(checks))
        for (section, _text), (statement, checks) in matches.items()
    ]


def _prompt_row(
    index: int, target: tuple[str, Statement, frozenset[str]],
    reader: EvidenceReader, analytics: AnalyticsBundle,
) -> dict[str, Any]:
    section, statement, checks = target
    evidence = []
    for evidence_id in statement.evidence_ids:
        item = reader.get(evidence_id)
        if item is not None:
            evidence.append({
                "evidence_id": evidence_id,
                "claim_text": (item.claim_text or "")[:600],
                "metric": item.metric,
                "value": item.value,
                "unit": item.unit,
                "period": item.period_label,
                "is_canonical": item.is_canonical,
                "status": item.status.value,
            })
    analytics_by_id = {item.analytics_id: item for item in analytics.results}
    analytic_rows = [
        {
            "analytics_id": analytic_id,
            "label": analytics_by_id[analytic_id].label,
            "value": analytics_by_id[analytic_id].value,
            "unit": analytics_by_id[analytic_id].unit,
        }
        for analytic_id in statement.analytics_ids if analytic_id in analytics_by_id
    ]
    return {
        "index": index,
        "section": section,
        "text": statement.text,
        "qa_checks": sorted(checks),
        "evidence": evidence,
        "analytics": analytic_rows,
    }


def _apply_statement_repairs(
    draft: ReportDraft,
    targets: list[tuple[str, Statement, frozenset[str]]],
    proposals: dict[int, str],
) -> tuple[ReportDraft, tuple[RepairEvent, ...]]:
    by_statement = {id(statement): (section, checks) for section, statement, checks in targets}
    events: list[RepairEvent] = []
    sections = []
    for section in draft.sections:
        statements = []
        for statement in section.statements:
            target = by_statement.get(id(statement))
            if target is None:
                statements.append(statement)
                continue
            section_name, checks = target
            rewritten = proposals.get(id(statement))
            if rewritten:
                statements.append(replace(statement, text=rewritten))
                events.append(RepairEvent(
                    section=section_name,
                    original_text=statement.text,
                    action="llm_rewrite",
                    checks=tuple(sorted(checks)),
                    repaired_text=rewritten,
                ))
            else:
                events.append(RepairEvent(
                    section=section_name,
                    original_text=statement.text,
                    action="deterministic_omit",
                    checks=tuple(sorted(checks)),
                ))
        sections.append(replace(section, statements=tuple(statements)))
    return replace(draft, sections=tuple(sections)), tuple(events)


def _repair_draft_metadata(
    draft: ReportDraft, findings: tuple[QAFinding, ...],
) -> ReportDraft:
    metadata = dict(draft.metadata)
    remaining_ids = {
        evidence_id
        for statement in draft.all_statements
        for evidence_id in statement.evidence_ids
    }
    metadata["missing_evidence_references"] = [
        evidence_id for evidence_id in metadata.get("missing_evidence_references", [])
        if evidence_id in remaining_ids
    ]

    omissions = list(metadata.get("sections_omitted", []))
    known = {str(row.get("section")) for row in omissions if isinstance(row, dict)}
    for finding in findings:
        if finding.check == "narrative.section_present" and finding.section \
                and finding.section not in known:
            omissions.append({
                "section": finding.section,
                "reason": "automatic QA repair recorded the missing unsupported section",
            })
            known.add(finding.section)
    metadata["sections_omitted"] = omissions
    return replace(draft, metadata=metadata)
