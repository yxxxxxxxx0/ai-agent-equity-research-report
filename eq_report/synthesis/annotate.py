"""Builds the companion annotation used by the "annotated" PDF.

For every paragraph and bullet in the finished report, this produces one short
label: the single piece of information or conclusion the sentence exists to
convey, as opposed to a restatement of its numbers. It exists so the report's
author can check, section by section, that every paragraph earns its place -
see the "each paragraph must have a meaning" editorial standard - without
having to re-read and re-derive the point of each one by eye.

This never changes the primary report. It runs once, after the ReportDraft is
final, and its output is only ever consumed by
:meth:`eq_report.rendering.pdf_renderer.PdfReportRenderer.render_annotated`,
which prints it as a companion PDF, not by anything that feeds back into the
draft. Because of that one-way relationship, an annotation is free to be
wrong or missing without weakening the primary report's evidentiary
guarantees - so unlike an ordinary segment finding, an annotation earns no
evidence_id and is never checked by QA.
"""

from __future__ import annotations

import logging
import re

from ..config import ModelConfig
from ..domain.report import ReportDraft
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event

logger = get_logger("synthesis.annotate")

#: Above this many sentences, the batched prompt risks running past a
#: reasonable single-call size; excess sentences fall back to the heuristic
#: rather than growing the request without bound.
_MAX_MODEL_ITEMS = 220

_SYSTEM_PROMPT = """You annotate sentences from a finished equity research report. For each
numbered sentence, write a short label (at most 12 words, no trailing period) stating the
single piece of information or conclusion the sentence exists to convey to the reader - not a
restatement of its numbers, and not a description of the sentence itself (never write things
like "states a fact" or "provides a metric"). Return JSON only, matching the requested schema."""


async def build_annotations(
    draft: ReportDraft, model_config: ModelConfig | None, *, tracker: UsageTracker | None = None,
) -> dict[str, str]:
    """One message per unique paragraph/bullet text in ``draft``.

    Falls back to a cheap heuristic label - never an empty message - when no
    model is configured or the call fails, following the same "verify or
    fall back to code" pattern used by every other LLM-backed stage: the
    companion PDF should still be useful even without a model configured.
    """
    items = _collect(draft)
    if not items:
        return {}
    if model_config is not None and model_config.enabled:
        try:
            return await _annotate_with_model(items, model_config, tracker)
        except Exception as exc:  # noqa: BLE001 - this is a review aid, never load-bearing
            log_event(logger, logging.WARNING, "annotation LLM call failed; using heuristic",
                      error=f"{type(exc).__name__}: {exc}")
    return {text: _heuristic(text) for text in items}


def _collect(draft: ReportDraft) -> list[str]:
    """Every distinct paragraph/bullet text across the body, in first-seen
    order, excluding Sources (a citation list has no "point" to extract)."""
    seen: dict[str, None] = {}
    for section in draft.sections:
        for statement in section.statements:
            seen.setdefault(statement.text, None)
        for paragraph in section.paragraphs:
            seen.setdefault(paragraph, None)
    return list(seen)


async def _annotate_with_model(
    items: list[str], model_config: ModelConfig, tracker: UsageTracker | None,
) -> dict[str, str]:
    client = OpenRouterJSONClient(model_config, tracker=tracker)
    modelled, overflow = items[:_MAX_MODEL_ITEMS], items[_MAX_MODEL_ITEMS:]
    schema = {"annotations": [{"id": "the sentence's number", "message": "<=12 word label"}]}
    prompt = (
        "Sentences:\n"
        + "\n".join(f"{i}. {text}" for i, text in enumerate(modelled))
        + f"\nReturn this JSON shape: {schema}"
    )
    response = await client.complete_json(_SYSTEM_PROMPT, prompt, stage="annotate")
    payload = response.payload if isinstance(response.payload, dict) else {}

    out: dict[str, str] = {}
    for row in payload.get("annotations", []):
        if not isinstance(row, dict):
            continue
        try:
            index = int(row.get("id"))
        except (TypeError, ValueError):
            continue
        if 0 <= index < len(modelled):
            message = str(row.get("message", "")).strip()
            if message:
                out[modelled[index]] = message

    # Anything the model skipped or an id it couldn't be matched to (plus any
    # sentence beyond the batch cap) still gets a heuristic label, so the
    # annotated report never shows a bullet with nothing under it.
    for text in (*modelled, *overflow):
        out.setdefault(text, _heuristic(text))
    return out


_BREAK_RE = re.compile(r"[:;,]| - | so | which | while ")


def _heuristic(text: str) -> str:
    """A cheap fallback label: the clause up to the first strong break,
    which is usually where a sentence's headline claim ends and its
    supporting detail or implication begins."""
    lead = _BREAK_RE.split(text, maxsplit=1)[0].strip()
    words = lead.split()
    return " ".join(words[:12]) + ("…" if len(words) > 12 else "")
