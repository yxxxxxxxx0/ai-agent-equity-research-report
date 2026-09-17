"""The QA auditor: resolves a flagged data conflict, or leaves it blocked.

This is the one place in the pipeline where an LLM is allowed anywhere near a
QA-raised *data* problem (as opposed to a narrative-repair problem - see
``docs/QA_REPAIR.md`` for that separate, prose-only loop). Its authority is
deliberately narrow, because QA only ever proves internal consistency, never
ground truth: a model that could just pick which of two conflicting numbers
"looks right" would make a wrong report pass QA looking fully audited, which
is worse than an honest, visible block. So the auditor may resolve a
conflict *only* by finding a real, dated, source-linked answer via live web
search - never by guessing. No verifiable source, no resolution: the
conflict stays CRITICAL and blocks the PDF, exactly as if the auditor had
not run at all.

When it does find one, it does not merely relabel the QA finding - it
corrects the Evidence Store itself: the corroborated row is marked
``is_canonical=True``/``VALIDATED`` and every row it actually contradicted is
demoted (``is_canonical=False``), both carrying the verification's source,
URL and date in ``metadata``/``validation_messages``. That is what lets
:class:`~eq_report.qa.engine.QAEngine` re-run the deterministic check
afterward and get a genuinely fresh, honest verdict - "solved and
re-checked", not "silenced".
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Any

from ..config import ModelConfig
from ..domain.enums import EvidenceStatus
from ..domain.qa import QAFinding, Severity
from ..evidence.reader import EvidenceReader
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json
from ..logging_setup import get_logger, log_event

logger = get_logger("qa.auditor")

_CLASSIFY_SYSTEM_PROMPT = """You are a conservative financial-data provenance reviewer.
Return JSON only."""

_VERIFY_SYSTEM_PROMPT = """You are a conservative financial-data verifier with live web
search, helping resolve a genuine disagreement between two data-vendor readings of the same
metric. Search the web for a real, current, dated source that states this metric's actual value.
Reply only with what you can attribute to an actual page you found: never invent a value, a
publisher name, a URL, or a date. If you cannot find a specific, dated source, set found=false
rather than guessing which candidate is more plausible. Return JSON only, matching the requested
schema."""

#: Bounds the cost of one run: at most this many extra, web-search-enabled
#: model calls to verify a genuine (not just differently-defined) conflict.
_MAX_VERIFICATIONS = 5

#: A verified source must agree with the candidate it corroborates to within
#: this relative tolerance, or it is treated as matching neither - rounding
#: noise is fine, a different fact is not.
_AGREEMENT_TOLERANCE = 0.02


@dataclass(frozen=True, slots=True)
class AuditOutcome:
    """What the auditor found, kept separate from any particular findings
    list so the caller is free to re-run checks against the corrected store
    and re-apply these afterward, rather than patching a now-stale list."""

    #: Canonical metric ids the auditor corrected the Evidence Store for -
    #: the caller re-runs the checks that touch these and trusts that fresh
    #: result, rather than trusting the auditor's own bookkeeping.
    corrected_metrics: frozenset[str]
    #: (check, subject) -> reasoning, for a conflict the auditor downgraded
    #: without touching the store (e.g. "different definitions"). The caller
    #: re-applies these to whichever finding list is actually final.
    downgrade_messages: dict[tuple[str, str], str]


class QAAuditor:
    """Resolves ``consistency.metric_agreement`` conflicts, or blocks honestly.

    ``audit`` is the only entry point: hand it the findings a QA pass just
    produced and the reader those findings came from, and it reports what it
    found - it does not itself return a findings list, since a store
    correction means the caller should re-run the checks fresh rather than
    trust this module's own bookkeeping as the final word.
    """

    def __init__(
        self, model_config: ModelConfig | None, *,
        tracker: UsageTracker | None = None, verify_online: bool = False,
    ) -> None:
        self._model_config = model_config
        self._tracker = tracker
        # Live web-search verification is a genuine extra cost per conflict
        # (Settings.verify_metric_conflicts) - classification (same fact vs.
        # different definitions) is cheap and always runs, but actually
        # resolving a conflict against a real source stays opt-in.
        self._verify_online_enabled = verify_online

    async def audit(
        self, findings: list[QAFinding], reader: EvidenceReader,
    ) -> AuditOutcome:
        conflicts = [f for f in findings if f.check == "consistency.metric_agreement"
                     and f.severity is Severity.CRITICAL]
        if not conflicts or self._model_config is None or not self._model_config.enabled:
            return AuditOutcome(corrected_metrics=frozenset(), downgrade_messages={})

        groups = self._build_groups(conflicts)
        same_fact_positions, downgraded = await self._classify(conflicts, groups)

        corrected_metrics: set[str] = set()
        if same_fact_positions and self._verify_online_enabled:
            verify_messages, corrected = await self._verify_online(
                conflicts, groups, same_fact_positions, reader)
            corrected_metrics |= corrected
            for position, message in verify_messages.items():
                downgraded[id(conflicts[position])] = message

        downgrade_messages = {
            (finding.check, finding.subject or ""): downgraded[id(finding)]
            for finding in conflicts if id(finding) in downgraded
        }
        return AuditOutcome(
            corrected_metrics=frozenset(corrected_metrics),
            downgrade_messages=downgrade_messages,
        )

    # -- internals ---------------------------------------------------------
    @staticmethod
    def _build_groups(conflicts: list[QAFinding]) -> list[dict[str, Any]]:
        groups: list[dict[str, Any]] = []
        for index, finding in enumerate(conflicts[:10]):
            groups.append({
                "index": index, "metric": finding.subject,
                "evidence_ids": list(finding.details.get("evidence_ids", [])),
            })
        return groups

    async def _classify(
        self, conflicts: list[QAFinding], groups: list[dict[str, Any]],
    ) -> tuple[list[int], dict[int, str]]:
        """Same fact vs. different definitions - never a numerical winner."""
        prompt = (
            "Determine whether each candidate group represents the SAME comparable financial/"
            "market fact, DIFFERENT DEFINITIONS OR INSTRUMENTS, or is INSUFFICIENTLY IDENTIFIED. "
            "Do not select a numerical winner and do not infer missing metadata.\n"
            f"Groups: {groups}\nReturn JSON: "
            "{'items':[{'index':0,'verdict':'same_fact|different_definitions|insufficient','reason':'short'}]}"
        )
        response, _error = await safe_complete_json(
            self._model_config, _CLASSIFY_SYSTEM_PROMPT, prompt,
            tracker=self._tracker, stage="qa_conflict_adjudication")
        downgraded: dict[int, str] = {}
        same_fact: list[int] = []
        if response is None:
            return same_fact, downgraded
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
                    f"Auditor: {row.get('reason') or 'definitions differ'}.")
            elif verdict == "same_fact":
                same_fact.append(position)
        return same_fact, downgraded

    async def _verify_online(
        self, conflicts: list[QAFinding], groups: list[dict[str, Any]],
        same_fact_positions: list[int], reader: EvidenceReader,
    ) -> tuple[dict[int, str], set[str]]:
        """Resolve a same-fact conflict only against a real, dated source.

        On a match, corrects the Evidence Store: the corroborated row becomes
        canonical, and every row it actually contradicts is demoted - so a
        re-run of the deterministic check afterward sees a genuinely
        corrected store, not just a silenced finding.
        """
        client = OpenRouterJSONClient(self._model_config, tracker=self._tracker)
        schema = {
            "found": "true or false",
            "matched_value": "the number the source states, only if found is true",
            "source_name": "the publisher or site name, only if found is true",
            "source_url": "the exact URL of the page you found this on, only if found is true",
            "published_date": "YYYY-MM-DD if stated on the page, else null",
        }
        messages: dict[int, str] = {}
        corrected_metrics: set[str] = set()
        for position in same_fact_positions[:_MAX_VERIFICATIONS]:
            finding = conflicts[position]
            group = groups[position]
            rows = [reader.get(str(eid)) for eid in group["evidence_ids"]]
            rows = [r for r in rows if r is not None]
            candidates = sorted({r.value for r in rows if r.value is not None})
            if not candidates:
                continue
            prompt = (
                f"Company: {reader.company} ({reader.ticker or 'ticker unknown'})\n"
                f"Metric: {group['metric']}\n"
                f"Candidate observations (the source must match their period/as-of date): "
                f"{[{'evidence_id': r.evidence_id, 'value': r.value, 'as_of': r.as_of.isoformat() if r.as_of else None, 'source': r.original_source_name or r.source_name} for r in rows]}\n"
                "Find the real value for that same period/as-of date from an actual, dated source, "
                "and report exactly what that source states.\n"
                f"Return this JSON shape: {schema}"
            )
            try:
                response = await client.complete_json(
                    _VERIFY_SYSTEM_PROMPT, prompt, web_search=True,
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
            if abs(winner - matched_value) / max(abs(matched_value), 1e-9) > _AGREEMENT_TOLERANCE:
                continue  # the verified source doesn't actually match either candidate

            verification_meta = {
                "source_name": source_name, "source_url": url,
                "published_date": payload.get("published_date"),
                "matched_value": matched_value,
            }
            winner_row = next((r for r in rows if r.value == winner), None)
            losers = [r for r in rows if r is not winner_row]
            updated = []
            if winner_row is not None:
                updated.append(replace(
                    winner_row, status=EvidenceStatus.VALIDATED, is_canonical=True,
                    validation_messages=winner_row.validation_messages
                    + ("auditor: web-verified canonical selection",),
                    metadata={**winner_row.metadata, "web_verification": verification_meta},
                ))
            for loser in losers:
                updated.append(replace(
                    loser, status=EvidenceStatus.REJECTED, is_canonical=False,
                    validation_messages=loser.validation_messages
                    + (f"auditor: contradicted by a web-verified source ({url})",),
                    metadata={**loser.metadata, "superseded_by_verification": verification_meta},
                ))
            if not updated:
                continue
            reader.store.replace(updated)
            corrected_metrics.add(group["metric"])

            date_bit = f", dated {payload['published_date']}" if payload.get("published_date") else ""
            messages[position] = (
                f"Auditor: web verification via {source_name}{date_bit} reports {matched_value}, "
                f"matching the {winner} candidate; the other candidate(s) were demoted as "
                f"contradicted, not merely unconfirmed. <{url}>"
            )
        return messages, corrected_metrics
