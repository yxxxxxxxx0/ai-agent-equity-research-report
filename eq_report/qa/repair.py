"""Bounded, auditable repair of narrative QA failures.

The model may edit prose, but it cannot change evidence, analytics, citations,
or the QA verdict. Every proposed edit is applied to the existing structured
draft and the normal deterministic QA suite is run again by the orchestrator.
If a local narrative error cannot be repaired safely, the offending statement
is omitted; evidence-level and arithmetic failures remain hard blockers.

A statement the entailment reviewer rejected is repaired by subtraction only:
the model trims it to the parts the reviewer said its cited evidence supports,
and the orchestrator's QA re-run (entailment included) must then pass it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.qa import QAFinding, QAResult
from ..domain.report import ReportDraft, Statement
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json

_REPAIRABLE_STATEMENT_CHECKS = frozenset({
    "evidence.reference_exists",
    "analytics.reference_exists",
    "evidence.citation_resolves",
    "evidence.no_unsupported_numbers",
    "evidence.claim_supported",
    "evidence.numeric_claim_not_canonical",
    "evidence.web_claim_not_reverified",
    "evidence.claim_not_entailed",
})

_NUMERIC_CHECK = "evidence.numeric_claim_not_canonical"
_ENTAILMENT_CHECK = "evidence.claim_not_entailed"
_REWRITEABLE_CHECKS = frozenset({_NUMERIC_CHECK, _ENTAILMENT_CHECK})

_SYSTEM_PROMPT = """You repair a structured equity-research draft after deterministic QA.
Each statement lists the QA checks it failed. Repair only by removing content - never by
adding it:

* evidence.claim_not_entailed: the statement's reviewer_reasons say which parts of the claim
  the cited evidence supports and which it does not. Rewrite the statement to keep only the
  supported parts and delete the unsupported clauses entirely. Do not turn an unsupported
  clause into a hedge ("may", "could indicate", "will help assess") - delete it.
* evidence.numeric_claim_not_canonical: keep a figure only if it appears verbatim in one of
  the statement's evidence excerpts (and attribute it to that source, e.g. "Zacks cited
  33.54x"); remove every figure you computed or that no excerpt states.

Do not add facts, dates, quantities, causes, comparisons, citations or source ids that are
not already in the statement, and do not guess. The result must still tell the reader
something concrete about the company; if all that would remain is a bare date, a title, or a
sentence with no information, choose omit. Return JSON only in the requested shape."""

_NUMBER_TOKEN = re.compile(r"\d[\d,.]*")


def _numbers(text: str) -> set[str]:
    return {m.group().rstrip(".,") for m in _NUMBER_TOKEN.finditer(text)}


def _entailment_reason(finding: QAFinding) -> str:
    prefix = "Mapped source does not support the complete claim: "
    message = finding.message or ""
    return message[len(prefix):] if message.startswith(prefix) else message


def _safe_rewrite(original: str, rewritten: str, checks: frozenset[str]) -> bool:
    """A repair may only remove content: no figure the original did not state.

    Whether a kept figure is publishable (canonical, or quoted verbatim from a
    cited passage) is decided by the orchestrator's QA re-run, not here.
    """
    return _numbers(rewritten) <= _numbers(original)


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
                    if not (0 <= index < len(rewriteable)) or not text:
                        continue
                    _section, statement, checks, _reasons = rewriteable[index]
                    if _safe_rewrite(statement.text, text, checks):
                        proposals[id(statement)] = text

        repaired, events = _apply_statement_repairs(draft, targets, proposals)
        repaired, summary_events = _drop_unsupported_summaries(repaired, qa.critical)
        events = events + summary_events
        repaired = _repair_draft_metadata(repaired, qa.critical)
        return RepairOutcome(
            draft=repaired,
            changed=repaired != draft,
            events=events,
            llm_error=llm_error,
        )


_Target = tuple[str, Statement, frozenset[str], tuple[str, ...]]


def _statement_targets(
    draft: ReportDraft, findings: tuple[QAFinding, ...],
) -> list[_Target]:
    matches: dict[tuple[str, str], tuple[Statement, set[str], list[str]]] = {}
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
                    _stmt, checks, reasons = matches.setdefault(key, (statement, set(), []))
                    checks.add(finding.check)
                    if finding.check == _ENTAILMENT_CHECK:
                        reasons.append(_entailment_reason(finding))
                    break
    return [
        (section, statement, frozenset(checks), tuple(reasons))
        for (section, _text), (statement, checks, reasons) in matches.items()
    ]


def _prompt_row(
    index: int, target: _Target, reader: EvidenceReader, analytics: AnalyticsBundle,
) -> dict[str, Any]:
    section, statement, checks, reasons = target
    evidence = []
    for evidence_id in statement.evidence_ids:
        item = reader.get(evidence_id)
        if item is not None:
            evidence.append({
                "evidence_id": evidence_id,
                # Same excerpt length the entailment reviewer judged against.
                "claim_text": (item.claim_text or "")[:700],
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
    row: dict[str, Any] = {
        "index": index,
        "section": section,
        "text": statement.text,
        "qa_checks": sorted(checks),
        "evidence": evidence,
        "analytics": analytic_rows,
    }
    if reasons:
        row["reviewer_reasons"] = list(reasons)
    return row


def _apply_statement_repairs(
    draft: ReportDraft,
    targets: list[_Target],
    proposals: dict[int, str],
) -> tuple[ReportDraft, tuple[RepairEvent, ...]]:
    by_statement = {
        id(statement): (section, checks) for section, statement, checks, _reasons in targets
    }
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


def tidy_draft(draft: ReportDraft, qa: QAResult) -> ReportDraft:
    """Deterministic editorial cleanup after QA: no repeats, no empty headings.

    A statement QA flagged as a verbatim repeat is removed from the later
    section (the finding names that one), and a section left with neither
    statements nor a summary is dropped and recorded as omitted rather than
    printed as a bare heading.
    """
    repeats = {
        (f.section, f.subject) for f in qa.findings
        if f.check == "narrative.duplication" and f.section and f.subject
    }
    omitted = list(draft.metadata.get("sections_omitted", []))
    sections = []
    for section in draft.sections:
        statements = tuple(
            s for s in section.statements
            if not any(sec == section.section.value and s.text.startswith(subj)
                       for sec, subj in repeats)
        )
        section = replace(section, statements=statements)
        if not statements and not section.summary and section.section.value != "sources":
            omitted.append({"section": section.section.value,
                            "reason": "nothing publishable remained after QA repair"})
            continue
        sections.append(section)
    return replace(draft, sections=tuple(sections),
                   metadata={**draft.metadata, "sections_omitted": omitted})


_SUMMARY_CHECKS = _REPAIRABLE_STATEMENT_CHECKS | {
    "evidence.summary_supported", "evidence.summary_traceable",
}


def _drop_unsupported_summaries(
    draft: ReportDraft, findings: tuple[QAFinding, ...],
) -> tuple[ReportDraft, tuple[RepairEvent, ...]]:
    """A section standfirst QA rejected is removed; the section body stands on its own."""
    failing: dict[str, set[str]] = {}
    for finding in findings:
        if finding.check in _SUMMARY_CHECKS and finding.section and finding.subject:
            failing.setdefault(finding.section, set()).add(finding.check)
    events: list[RepairEvent] = []
    sections = []
    for section in draft.sections:
        checks = failing.get(section.section.value)
        subject_matches = section.summary and any(
            f.subject and (section.summary.startswith(f.subject) or f.subject.startswith(section.summary))
            for f in findings if f.section == section.section.value
        )
        if checks and subject_matches:
            events.append(RepairEvent(
                section=section.section.value, original_text=section.summary,
                action="summary_omit", checks=tuple(sorted(checks)),
            ))
            section = replace(section, summary="", summary_evidence_ids=(),
                              summary_analytics_ids=(), summary_citation_refs=())
        sections.append(section)
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
