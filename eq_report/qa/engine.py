"""Stage 9 - the QA engine.

Runs every deterministic check in :mod:`checks` over the ReportDraft and
aggregates the findings. A check that itself raises is reported as a critical
finding rather than allowed to abort QA: a broken checker must not be able to
wave a bad report through. These deterministic checks remain the ground
truth QA verdict; PDF generation is gated on their result alone (see
``QAResult.has_critical_errors``/``passed``, unaffected by anything below).

Model assistance is limited to narrow conflict classification and the
separate repair loop. There is no informational whole-report LLM review.
"""

from __future__ import annotations

import logging
from dataclasses import replace

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import Severity
from ..domain.plan import ResearchPlan
from ..domain.qa import QAFinding, QAResult
from ..domain.report import ReportDraft
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from .auditor import QAAuditor
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
        *, tracker: UsageTracker | None = None, verify_conflicts: bool = False,
    ) -> None:
        self.checks = tuple(checks)
        self._model_config = model_config
        self._tracker = tracker
        self._verify_conflicts = verify_conflicts

    async def validate(
        self,
        draft: ReportDraft,
        plan: ResearchPlan,
        reader: EvidenceReader,
        analytics: AnalyticsBundle,
    ) -> QAResult:
        context = QAContext(draft=draft, plan=plan, reader=reader, analytics=analytics)

        findings, checks_run = self._run_checks(context)

        # A conflict is usually a real data disagreement, but market-data
        # payloads can also place adjusted, peer, option, and benchmark
        # series under similar field labels. The auditor (qa/auditor.py)
        # classifies which is which, and may resolve a genuine disagreement
        # only against a real, dated, source-linked answer - never by
        # picking a numerical winner itself (see that module's docstring).
        auditor = QAAuditor(
            self._model_config, tracker=self._tracker, verify_online=self._verify_conflicts)
        outcome = await auditor.audit(findings, reader)
        checks_run.append("qa_auditor")

        if outcome.corrected_metrics:
            # The auditor corrected the Evidence Store itself (promoted a
            # verified value, demoted what it contradicted) - re-run every
            # check fresh against that corrected store rather than patch the
            # stale findings list. This is the literal "solved, then
            # checked again": a metric that is now genuinely consistent
            # reports nothing new on this second pass; anything still wrong
            # elsewhere in the report is untouched and still flagged.
            findings, recheck_names = self._run_checks(context)
            checks_run.extend(f"{name}.recheck" for name in recheck_names)
            log_event(logger, logging.INFO, "QA auditor corrected evidence; re-ran all checks",
                      corrected_metrics=sorted(outcome.corrected_metrics))

        if outcome.downgrade_messages:
            findings = [
                replace(finding, severity=Severity.WARNING,
                        message=f"{finding.message} {outcome.downgrade_messages[key]}")
                if (key := (finding.check, finding.subject or "")) in outcome.downgrade_messages
                else finding
                for finding in findings
            ]

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

    def _run_checks(self, context: QAContext) -> tuple[list[QAFinding], list[str]]:
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
        return findings, checks_run

