"""Stage 9 - the QA engine.

Runs every deterministic check in :mod:`checks` over the ReportDraft and
aggregates the findings. A check that itself raises is reported as a critical
finding rather than allowed to abort QA: a broken checker must not be able to
wave a bad report through. These deterministic checks remain the ground
truth QA verdict; PDF generation is gated on their result alone (see
``QAResult.has_critical_errors``/``passed``, unaffected by anything below).

LLM-backed review, deterministic checks as ground truth
----------------------------------------------------------
When a model is configured, ``validate`` additionally asks the LLM to review
the same draft (statements, tables, cited counts) and produce its own QA
verdict and findings. This is purely an added, informational cross-check: it
can never suppress a deterministic finding, and it can never turn a run
without critical deterministic findings into a failed one. If the LLM's
verdict disagrees with the deterministic one (e.g. it reports no material
issues while the deterministic checks found critical problems, or vice
versa), that disagreement is recorded as its own WARNING-level
``qa.llm_disagreement`` finding so it is visible without ever being trusted
over the code-based checks. Any of the LLM's own findings are attached at
INFO severity, clearly labelled as an LLM opinion, never CRITICAL/WARNING,
since only the deterministic checks are allowed to block the PDF.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import Severity
from ..domain.plan import ResearchPlan
from ..domain.qa import QAFinding, QAResult
from ..domain.report import ReportDraft
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json
from ..logging_setup import get_logger, log_event
from .checks import ALL_CHECKS, QAContext

logger = get_logger("qa")

_SEVERITY_ORDER = {Severity.CRITICAL: 0, Severity.WARNING: 1, Severity.INFO: 2}

_QA_SYSTEM_PROMPT = """You are an independent QA and citation-entailment reviewer for a
neutral institutional equity research report. For every statement, compare its wording with
the supplied evidence excerpts and analytics labels. Check whether the cited material really
supports the entity, period, direction, magnitude, causality and conclusion—not merely whether
a citation exists. Also judge inconsistent numbers, missing sections and editorial problems.
Return JSON only, matching the requested schema."""


class QAEngine:
    """Validates a ReportDraft before it is allowed to become a PDF."""

    def __init__(
        self, checks=ALL_CHECKS, model_config: ModelConfig | None = None,
        *, tracker: UsageTracker | None = None,
    ) -> None:
        self.checks = tuple(checks)
        self._model_config = model_config
        self._tracker = tracker

    async def validate(
        self,
        draft: ReportDraft,
        plan: ResearchPlan,
        reader: EvidenceReader,
        analytics: AnalyticsBundle,
    ) -> QAResult:
        context = QAContext(draft=draft, plan=plan, reader=reader, analytics=analytics)

        findings: list[QAFinding] = []
        checks_run: list[str] = []

        for check in self.checks:
            name = check.__name__
            try:
                findings.extend(check(context))
            except Exception as exc:  # noqa: BLE001 - a broken check is a critical failure
                findings.append(QAFinding(
                    check=f"qa.check_failed.{name}",
                    severity=Severity.CRITICAL,
                    message=f"QA check {name} raised {type(exc).__name__}: {exc}",
                ))
                log_event(logger, logging.ERROR, "QA check raised",
                          check=name, error=str(exc))
            checks_run.append(name)

        # The deterministic critical/warning count is fixed before the LLM is
        # ever consulted, so nothing below can change what gates the PDF.
        deterministic_critical = sum(1 for f in findings if f.severity is Severity.CRITICAL)
        findings.extend(await self._llm_review(
            draft, findings, deterministic_critical, reader, analytics))
        checks_run.append("llm_review")

        findings.sort(key=lambda f: (_SEVERITY_ORDER[f.severity], f.check))
        result = QAResult(findings=tuple(findings), checks_run=tuple(checks_run))

        log_event(
            logger, logging.INFO if result.passed else logging.ERROR,
            "QA complete",
            passed=result.passed, checks=len(checks_run),
            critical=len(result.critical), warnings=len(result.warnings),
            info=len(result.infos),
        )
        for finding in result.critical:
            log_event(logger, logging.ERROR,
                      f"QA critical [{finding.check}] {finding.message}",
                      check=finding.check, subject=finding.subject)
        for finding in result.warnings:
            log_event(logger, logging.WARNING,
                      f"QA warning [{finding.check}] {finding.message}",
                      check=finding.check, subject=finding.subject)
        return result

    # -- LLM-backed overlay ------------------------------------------------
    async def _llm_review(
        self, draft: ReportDraft, deterministic_findings: list[QAFinding],
        deterministic_critical: int,
        reader: EvidenceReader, analytics: AnalyticsBundle,
    ) -> list[QAFinding]:
        analytics_by_id = {item.analytics_id: item for item in analytics.results}
        statements: list[dict[str, Any]] = []
        for section in draft.sections:
            for statement in section.statements:
                evidence: list[dict[str, Any]] = []
                for evidence_id in statement.evidence_ids:
                    item = reader.get(evidence_id)
                    if item is not None:
                        evidence.append({
                            "evidence_id": evidence_id,
                            "claim_text": (item.claim_text or "")[:500],
                            "metric": item.metric, "value": item.value,
                            "unit": item.unit, "period": item.period_label,
                            "source": item.source_name,
                        })
                analytic_rows: list[dict[str, Any]] = []
                for analytics_id in statement.analytics_ids:
                    item = analytics_by_id.get(analytics_id)
                    if item is not None:
                        analytic_rows.append({
                            "analytics_id": analytics_id, "metric": item.metric,
                            "value": item.value, "unit": item.unit,
                            "label": item.label, "period": item.period,
                            "comparison_period": item.comparison_period,
                        })
                statements.append({
                    "section": section.section.value,
                    "text": statement.text[:300],
                    "claim_type": statement.claim_type.value,
                    "supporting_refs": len(statement.evidence_ids) + len(statement.analytics_ids),
                    "evidence_ids": list(statement.evidence_ids),
                    "analytics_ids": list(statement.analytics_ids),
                    "evidence": evidence,
                    "analytics": analytic_rows,
                })
        deterministic_summary = {
            "critical_count": deterministic_critical,
            "warning_count": sum(
                1 for f in deterministic_findings if f.severity is Severity.WARNING),
            "critical_messages": [
                f.message for f in deterministic_findings
                if f.severity is Severity.CRITICAL
            ][:20],
        }
        schema = {
            "overall_verdict": ["pass", "material_issues_found"],
            "findings": [{
                "severity": ["info", "warning"],
                "message": "a specific, actionable observation",
                "section": "section name or null",
            }],
        }
        prompt = (
            f"Report: {draft.title}\n"
            f"Statements with cited support: {statements[:200]}\n"
            f"Deterministic rule-based QA already found: {deterministic_summary}\n"
            f"Return this JSON shape: {schema}"
        )
        response, error = await safe_complete_json(
            self._model_config, _QA_SYSTEM_PROMPT, prompt,
            tracker=self._tracker, stage="qa")
        if response is None:
            log_event(logger, logging.INFO, "QA LLM review skipped", reason=error)
            return []

        payload = response.payload if isinstance(response.payload, dict) else {}
        verdict = str(payload.get("overall_verdict", "")).strip().lower()
        out: list[QAFinding] = []

        llm_says_clean = verdict == "pass"
        deterministic_says_clean = deterministic_critical == 0
        if llm_says_clean != deterministic_says_clean:
            out.append(QAFinding(
                check="qa.llm_disagreement",
                severity=Severity.WARNING,
                message=(
                    f"The LLM QA review verdict ({verdict or 'unknown'}) disagrees with the "
                    f"deterministic checks ({deterministic_critical} critical finding(s)). "
                    "The deterministic checks remain authoritative for gating the PDF."
                ),
            ))

        for row in payload.get("findings", [])[:20]:
            if not isinstance(row, dict):
                continue
            message = str(row.get("message", "")).strip()
            if not message:
                continue
            out.append(QAFinding(
                check="qa.llm_review",
                # Never CRITICAL/WARNING from the LLM path alone: only the
                # deterministic checks are allowed to affect PDF gating.
                severity=Severity.INFO,
                message=f"LLM QA review: {message}",
                section=row.get("section") if isinstance(row.get("section"), str) else None,
            ))
        return out
