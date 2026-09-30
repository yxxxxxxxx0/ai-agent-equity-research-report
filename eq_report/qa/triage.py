"""LLM triage of the *heuristic* number checks - loosen-only, code decides.

``evidence.no_unsupported_numbers`` and ``evidence.numeric_claim_not_canonical``
are regex-driven: they flag any figure with no id mapped to it, even when the
figure is simply correct and merely uncited. This step lets a model help, but
never judge:

1. Whitelist (idea 1): only those two checks are eligible, and the only change
   possible is CRITICAL -> WARNING. Integrity checks are never touched and the
   model can never make QA stricter or add ids.
2. Extract, then compare (idea 3): the model only reads the sentence and says
   which canonical metric/period each figure is. Code parses the figure itself,
   looks the metric up in the Evidence Store and compares within tolerance. A
   finding is downgraded only if *every* figure matches, in *both* runs.
3. Shadow mode (idea 8): ``EQR_QA_TRIAGE=shadow`` (default) logs what would be
   downgraded and changes nothing; ``on`` applies it; ``off`` skips the calls.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import replace
from typing import Any

from ..config import ModelConfig
from ..domain.enums import Severity
from ..domain.qa import QAFinding
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json
from ..logging_setup import get_logger, log_event
from ..normalisation import canonical_metrics as cm
from ..normalisation.units import normalise_percent, parse_number
from .checks import _DATE_OR_PERIOD_RE, _NUMBER_RE

logger = get_logger("qa.triage")

TRIAGEABLE = frozenset({"evidence.no_unsupported_numbers", "evidence.numeric_claim_not_canonical"})
_TOLERANCE = 0.01

_SYSTEM = """You read one sentence from an equity report and identify what each figure in it is.
For every figure (money, percentage, multiple, count) return the exact token as written
("$24.1bn", "18.4%", "31.4x"), the canonical metric id it describes from the vocabulary
(or null if none fits), and the fiscal period label if the sentence states one (else null).
Do not judge whether the figure is correct. Return JSON only:
{"facts":[{"number":"...","metric":"...|null","period":"...|null"}]}"""


def _figures(text: str) -> list[str]:
    """The figures the heuristic check saw: numbers left after dates/periods are removed."""
    return [m.group().strip() for m in _NUMBER_RE.finditer(_DATE_OR_PERIOD_RE.sub("", text))
            if any(c.isdigit() for c in m.group())]


def _matches(token: str, metric: str | None, period: str | None, reader: EvidenceReader) -> str | None:
    """Evidence id of a store row whose value equals the figure, else None."""
    if not metric or metric not in cm.all_canonical_metrics():
        return None
    try:
        parsed = parse_number(token.replace(" ", ""), field_name=metric)
    except Exception:  # noqa: BLE001 - any parse failure = not confirmed
        return None
    value = normalise_percent(parsed.value, source_looked_like_percent=True) if parsed.is_percent else parsed.value
    for item in reader.numeric_all(metric, period or None):
        if item.value is not None and math.isclose(item.value, value, rel_tol=_TOLERANCE, abs_tol=1e-9):
            return item.evidence_id
    return None


async def _confirm(statement: str, reader: EvidenceReader, config: ModelConfig | None,
                   tracker: UsageTracker | None, client: Any | None) -> dict[str, str] | None:
    """``{figure: evidence_id}`` if both runs match every figure to the store, else None."""
    figures = _figures(statement)
    if not figures:
        return None
    prompt = f"Vocabulary: {sorted(cm.all_canonical_metrics())}\n\nSENTENCE: {statement}"
    runs = await asyncio.gather(*(
        safe_complete_json(config, _SYSTEM, prompt, client=client, tracker=tracker, stage="qa:triage")
        for _ in range(2)))
    matched: list[dict[str, str]] = []
    for response, _error in runs:
        payload = response.payload if response else None
        facts = [f for f in payload.get("facts", []) if isinstance(f, dict)] if isinstance(payload, dict) else []
        found: dict[str, str] = {}
        for figure in figures:
            fact = next((f for f in facts if str(f.get("number", "")).strip() == figure), None)
            evidence_id = _matches(figure, fact and fact.get("metric"), fact and fact.get("period"), reader) if fact else None
            if evidence_id is None:
                return None
            found[figure] = evidence_id
        matched.append(found)
    return matched[0] if matched[0] == matched[1] else None


async def triage_findings(
    findings: list[QAFinding], reader: EvidenceReader, config: ModelConfig | None, mode: str, *,
    tracker: UsageTracker | None = None, client: Any | None = None,
    cache: dict[str, dict[str, str] | None] | None = None,
) -> list[QAFinding]:
    """Downgrade confirmed false positives (mode ``on``); log them (mode ``shadow``)."""
    if mode == "off" or (client is None and (config is None or not config.enabled)):
        return findings
    cache = {} if cache is None else cache
    out: list[QAFinding] = []
    for finding in findings:
        if finding.check not in TRIAGEABLE or finding.severity is not Severity.CRITICAL or not finding.subject:
            out.append(finding)
            continue
        if finding.subject not in cache:
            cache[finding.subject] = await _confirm(finding.subject, reader, config, tracker, client)
        confirmed = cache[finding.subject]
        if confirmed is None:
            out.append(finding)
            continue
        log_event(logger, logging.INFO, "QA triage: figures match the Evidence Store",
                  mode=mode, check=finding.check, subject=finding.subject[:120], matches=confirmed)
        if mode == "on":
            note = ("Every figure matches the Evidence Store (" + ", ".join(
                f"{k} = {v}" for k, v in confirmed.items()) + "); the statement lacks a citation.")
            finding = replace(finding, severity=Severity.WARNING, message=f"{finding.message} {note}",
                              details={**finding.details, "triage_matches": confirmed})
        out.append(finding)
    return out
