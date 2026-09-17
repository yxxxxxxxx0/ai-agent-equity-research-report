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
from typing import Any

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import EvidenceStatus, Severity
from ..domain.plan import ResearchPlan
from ..domain.qa import QAFinding, QAResult
from ..domain.report import ReportDraft
from ..evidence.reader import EvidenceReader
from ..llm.client import OpenRouterJSONClient
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

_CONFLICT_VERIFY_SYSTEM_PROMPT = """You are a conservative financial-data verifier with live web
search, helping resolve a genuine disagreement between two data-vendor readings of the same
metric. Search the web for a real, current, dated source that states this metric's actual value.
Reply only with what you can attribute to an actual page you found: never invent a value, a
publisher name, a URL, or a date. If you cannot find a specific, dated source, set found=false
rather than guessing which candidate is more plausible. Return JSON only, matching the requested
schema."""

#: Bounds the cost of one run: at most this many extra, web-search-enabled
#: model calls to verify a genuine (not just differently-defined) metric
#: conflict.
_MAX_CONFLICT_VERIFICATIONS = 5


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

        # A conflict is usually deterministic, but market-data payloads can
        # place adjusted, peer, option, and benchmark series under similar
        # field labels. Give the LLM the *small* disputed group only to decide
        # whether definitions differ. It may de-escalate a false comparison;
        # it may never promote a value or choose a numerical winner.
        findings = await self._adjudicate_metric_conflicts(findings, reader)
        checks_run.append("llm_conflict_adjudication")

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

    async def _adjudicate_metric_conflicts(
        self, findings: list[QAFinding], reader: EvidenceReader,
    ) -> list[QAFinding]:
        conflicts = [f for f in findings if f.check == "consistency.metric_agreement"
                     and f.severity is Severity.CRITICAL]
        if not conflicts:
            return findings
        groups: list[dict[str, Any]] = []
        for index, finding in enumerate(conflicts[:10]):
            rows: list[dict[str, Any]] = []
            for evidence_id in finding.details.get("evidence_ids", []):
                item = reader.get(str(evidence_id))
                if item is None:
                    continue
                rows.append({
                    "evidence_id": item.evidence_id, "metric": item.metric,
                    "value": item.value, "unit": item.unit, "currency": item.currency,
                    "as_of": item.as_of.isoformat() if item.as_of else None,
                    "period": item.period_label, "basis": item.basis,
                    "source": item.original_source_name or item.source_name,
                    "raw_metric": item.raw_metric,
                    "json_path": item.metadata.get("json_path"),
                    "series": item.metadata.get("series"),
                })
            groups.append({"index": index, "metric": finding.subject, "rows": rows})
        prompt = (
            "Determine whether each candidate group represents the SAME comparable financial/"
            "market fact, DIFFERENT DEFINITIONS OR INSTRUMENTS, or is INSUFFICIENTLY IDENTIFIED. "
            "Do not select a numerical winner and do not infer missing metadata.\n"
            f"Groups: {groups}\nReturn JSON: {{'items':[{{'index':0,'verdict':'same_fact|different_definitions|insufficient','reason':'short'}}]}}"
        )
        response, _error = await safe_complete_json(
            self._model_config,
            "You are a conservative financial-data provenance reviewer. Return JSON only.",
            prompt, tracker=self._tracker, stage="qa_conflict_adjudication")
        if response is None:
            return findings
        downgraded: dict[int, str] = {}
        same_fact_positions: list[int] = []
        for row in response.payload.get("items", []) if isinstance(response.payload, dict) else []:
            if not isinstance(row, dict):
                continue
            verdict = str(row.get("verdict"))
            try:
                position = int(row.get("index"))
            except (TypeError, ValueError):
                continue
            if not (0 <= position < len(conflicts)):
                continue
            if verdict == "different_definitions":
                downgraded[id(conflicts[position])] = (
                    f"LLM provenance review: {row.get('reason') or 'definitions differ'}.")
            elif verdict == "same_fact":
                same_fact_positions.append(position)

        # A genuine same-fact disagreement (not just different definitions) is
        # never resolved by guessing which value looks more plausible - only
        # by finding a real, dated, source-linked answer, the same standard
        # pipeline.gap_research holds live web search to. Opt-in: a real extra
        # cost per conflict (see Settings.verify_metric_conflicts).
        verified_evidence_ids: set[str] = set()
        if self._verify_conflicts and same_fact_positions and self._model_config is not None \
                and self._model_config.enabled:
            online_downgrades, verified_evidence_ids = await self._verify_conflicts_online(
                conflicts, groups, same_fact_positions, reader)
            downgraded.update(online_downgrades)

        resolved_numeric_findings = {
            id(finding) for finding in findings
            if finding.check == "evidence.numeric_claim_not_canonical"
            and verified_evidence_ids.intersection(
                map(str, finding.details.get("evidence_ids", [])))
        }
        return [
            replace(finding, severity=Severity.WARNING,
                    message=f"{finding.message} {downgraded[id(finding)]}")
            if id(finding) in downgraded else finding
            for finding in findings if id(finding) not in resolved_numeric_findings
        ]

    async def _verify_conflicts_online(
        self, conflicts: list[QAFinding], groups: list[dict[str, Any]],
        same_fact_positions: list[int], reader: EvidenceReader,
    ) -> tuple[dict[int, str], set[str]]:
        """Best-effort live verification of a genuine same-fact disagreement.

        Only unblocks a conflict by matching one candidate to an actual,
        dated source the model found via web search - never by picking
        whichever value seems more plausible. No match, no URL, or a source
        that agrees with neither candidate all leave the conflict CRITICAL,
        exactly as if verification had not run.
        """
        client = OpenRouterJSONClient(self._model_config, tracker=self._tracker)
        schema = {
            "found": "true or false",
            "matched_value": "the number the source states, only if found is true",
            "source_name": "the publisher or site name, only if found is true",
            "source_url": "the exact URL of the page you found this on, only if found is true",
            "published_date": "YYYY-MM-DD if stated on the page, else null",
        }
        downgraded: dict[int, str] = {}
        verified_evidence_ids: set[str] = set()
        for position in same_fact_positions[:_MAX_CONFLICT_VERIFICATIONS]:
            finding = conflicts[position]
            group = groups[position]
            candidates = sorted({
                row["value"] for row in group["rows"] if row.get("value") is not None})
            if not candidates:
                continue
            prompt = (
                f"Company: {reader.company} ({reader.ticker or 'ticker unknown'})\n"
                f"Metric: {group['metric']}\n"
                f"Candidate observations (the source must match their period/as-of date): {group['rows']}\n"
                "Find the real value for that same period/as-of date from an actual, dated source, and "
                "report exactly what that source states.\n"
                f"Return this JSON shape: {schema}"
            )
            try:
                response = await client.complete_json(
                    _CONFLICT_VERIFY_SYSTEM_PROMPT, prompt, web_search=True,
                    stage="qa_conflict_verification")
            except Exception as exc:  # noqa: BLE001 - one failed lookup must not sink the rest
                log_event(logger, logging.WARNING, "conflict verification call failed",
                          metric=group["metric"], error=f"{type(exc).__name__}: {exc}")
                continue
            payload = response.payload if isinstance(response.payload, dict) else {}
            if not payload.get("found"):
                continue
            url = str(payload.get("source_url", "")).strip()
            source_name = str(payload.get("source_name", "")).strip()
            # A citation with no real URL is indistinguishable from an invented
            # one - the same rule pipeline.gap_research applies.
            if not source_name or not url.startswith(("http://", "https://")):
                continue
            try:
                matched_value = float(payload.get("matched_value"))
            except (TypeError, ValueError):
                continue
            winner = min(candidates, key=lambda v: abs(v - matched_value))
            if abs(winner - matched_value) / max(abs(matched_value), 1e-9) > 0.02:
                continue  # the verified source doesn't actually match either candidate
            date_bit = f", dated {payload['published_date']}" if payload.get("published_date") else ""
            winner_row = next(
                (row for row in group["rows"] if row.get("value") == winner), None)
            winner_id = str(winner_row.get("evidence_id")) if winner_row else ""
            item = reader.get(winner_id)
            if item is None:
                continue
            reader.store.replace([replace(
                item, status=EvidenceStatus.VALIDATED, is_canonical=True,
                validation_messages=item.validation_messages + ("web-verified canonical selection",),
                metadata={**item.metadata, "web_verification": {
                    "source_name": source_name, "source_url": url,
                    "published_date": payload.get("published_date"),
                    "matched_value": matched_value,
                }},
            )])
            verified_evidence_ids.add(winner_id)
            downgraded[id(finding)] = (
                f"Web verification: {source_name}{date_bit} reports {matched_value}, "
                f"matching the {winner} candidate; the other candidate(s) were not "
                f"corroborated by any dated source. <{url}>"
            )
        return downgraded, verified_evidence_ids
