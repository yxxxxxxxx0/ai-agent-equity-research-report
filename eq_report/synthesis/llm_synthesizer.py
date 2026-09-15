"""LLM-backed synthesizer: the only synthesis path when a model is configured.

Subclasses ``Synthesizer`` and uses the model for Key Takeaways and a batched
cross-section editorial pass over every body finding; the remaining steps -
tables, charts, citations, the sources section, deduplication and citation
numbering - is inherited unchanged from :class:`Synthesizer`. Those steps are
exactly the safety-critical bookkeeping the task calls out (citation
numbering must resolve, duplicate claims must be suppressed consistently) and
are left as pure deterministic code rather than LLM output, on the same
principle used everywhere else in this pipeline: narrative and ordering
decisions may come from the model, but anything that must be trustworthy by
construction stays in code. Only the selection and ordering of the report's
Key Takeaways is delegated to a model.

The orchestrator now always constructs this class instead of ``Synthesizer``
whenever a model is configured; there is no more ``use_for_synthesis`` toggle.
If the LLM call fails or returns nothing usable, ``_build_takeaways`` already
falls back to the inherited deterministic ``Synthesizer._key_takeaways``, so
the "verify or fall back to code" pattern used by every other converted stage
holds here too.

The model is shown every key finding the segment agents already produced,
each already carrying its own evidence_id/analytics_id tags, and asked to
select and order the most material ones (optionally writing one merged
sentence that restates two or more of them together). It may only cite
evidence_ids/analytics_ids that already appear somewhere in the findings it
was shown - anything else is dropped - so a merged sentence ends up exactly
as traceable as the findings it was built from, and the resulting statements
still go through :meth:`Synthesizer._statements`, which resolves citation
numbers the same way every other section does.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import ClaimType, ReportSection
from ..domain.plan import ResearchPlan
from ..domain.report import ReportDraft, ReportSectionDraft
from ..domain.segment import KeyFinding, SegmentResult
from ..evidence.reader import EvidenceReader
from ..llm.client import LLMJSONResponse, OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from .synthesizer import Synthesizer

logger = get_logger("synthesis.llm_synthesizer")

_CLAIM_TYPES = {c.value for c in ClaimType}
_MAX_TAKEAWAYS = 10

_SYSTEM_PROMPT = """You are the synthesis layer of a neutral institutional company
research pipeline. You are given every key finding the segment agents produced, each
already tagged with the evidence_id/analytics_id rows it rests on. Select and order the
most material findings into the report's Key Takeaways, following this reasoning chain:
what changed -> why -> financial impact -> surprise versus expectations -> forward
guidance -> valuation implication -> competitive position -> open uncertainty -> what to
monitor next.

Produce 4-6 connected analytical takeaways, not a list of isolated facts. Each takeaway
should lead with its point, support it with the supplied evidence, explain why it matters,
and hand the reader naturally to the next point in the argument. Prefer a merged takeaway
when adjacent findings need to be combined to complete that reasoning. Avoid repeating a
metric unless its role in the argument changes - a takeaway that restates a fact another
takeaway (or another section) already made, without changing what it means, adds nothing
and must be dropped in favour of a genuinely different point.

Deduplication rule: Review the full report for repeated facts, metrics, conclusions, and
explanations. Keep the most complete or contextually appropriate occurrence of each fact.
Remove later repetitions unless they add a materially new comparison, implication,
calculation, uncertainty, or monitoring relevance. Do not remove necessary cross-references,
but avoid restating the same numeric values when a short reference is sufficient.

This layer is a selection and ordering pass, not licence to write new judgments the
underlying findings do not make. A merged sentence may combine facts, but must stay
strictly on the observation side of the line between observation and conclusion: state
what the data shows (a comparison, a change, a gap between two measures) rather than
whether that makes the company or its valuation good, bad, attractive, or justified. Never
assert causation ("because", "driven by", "due to") beyond what the underlying findings
themselves establish. Never give a Buy/Hold/Sell view, a price target, or a trade
instruction.

You may write one merged sentence that restates two or more given findings together, but
every evidence_id/analytics_id you cite must be copied verbatim from the findings you were
given - never invent, guess, or reuse an id from general knowledge. Return JSON only,
matching the requested schema."""


class SynthesisLLMClient(Protocol):
    async def complete_json(
        self, system_prompt: str, user_prompt: str, *, stage: str = "",
    ) -> LLMJSONResponse: ...


class LLMSynthesizer(Synthesizer):
    """Synthesizer whose Key Takeaways selection is written by an LLM."""

    def __init__(
        self,
        report_run_id: str,
        plan: ResearchPlan,
        reader: EvidenceReader,
        analytics: AnalyticsBundle,
        model_config: ModelConfig | None = None,
        *,
        client: SynthesisLLMClient | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        super().__init__(report_run_id, plan, reader, analytics)
        self._model_config = model_config
        self._client = client
        if self._client is None:
            if model_config is None or not model_config.enabled:
                raise ValueError(
                    "LLMSynthesizer requires an enabled ModelConfig or an injected client")
            if model_config.provider.lower() != "openrouter":
                raise ValueError("LLMSynthesizer supports MODEL_PROVIDER=openrouter")
            self._client = OpenRouterJSONClient(model_config, tracker=tracker)
        self._llm_takeaways: ReportSectionDraft | None = None
        self._llm_body: dict[tuple[str, int], KeyFinding] = {}
        self._llm_headlines: dict[str, str] = {}
        self._llm_dropped: set[tuple[str, int]] = set()

    async def synthesize_async(
        self, segment_results: tuple[SegmentResult, ...]
    ) -> ReportDraft:
        """Resolve the LLM takeaways first, then run the (sync) inherited pipeline.

        The network call has to happen before ``synthesize`` runs, since that
        method - and the ``_key_takeaways`` hook it calls - is synchronous by
        design in the base class (every other synthesis step is pure).
        """
        self._llm_takeaways = await self._build_takeaways(segment_results)
        await self._build_body_sections(segment_results)
        return self.synthesize(segment_results)

    def _key_takeaways(self, segment_results: tuple[SegmentResult, ...]) -> ReportSectionDraft:
        if self._llm_takeaways is not None:
            return self._llm_takeaways
        return super()._key_takeaways(segment_results)

    def _findings_for_section(self, section, result):
        """Preserve the reasoning order chosen by an LLM segment agent.

        The deterministic synthesizer sorts facts ahead of interpretations,
        which breaks a model-authored chain such as result -> driver ->
        implication.  The LLM path keeps the agent's order while retaining the
        risk/catalyst tag split required by the report structure.
        """
        findings = tuple(
            self._llm_body.get((result.segment.value, index), finding)
            for index, finding in enumerate(result.key_findings)
            if (result.segment.value, index) not in self._llm_dropped
        )
        if section is ReportSection.RISKS:
            findings = tuple(f for f in findings if "risk" in f.tags)
        elif section is ReportSection.CATALYSTS:
            findings = tuple(f for f in findings if "catalyst" in f.tags)
        return tuple(findings)

    def _section_summary(self, section: ReportSection, result: SegmentResult) -> str:
        if section is ReportSection.CATALYSTS:
            key = f"{result.segment.value}:catalysts"
        else:
            key = result.segment.value
        return self._llm_headlines.get(key) or super()._section_summary(section, result)

    async def _build_body_sections(self, segment_results: tuple[SegmentResult, ...]) -> None:
        """Rewrite all body sections in one cross-report editorial call.

        The model can improve flow and remove repetition, but each rewritten
        finding inherits the exact provenance and claim type of its source
        finding.  Missing/malformed rows simply retain the agent-authored text.
        """
        rows: list[dict[str, Any]] = []
        source: dict[tuple[str, int], KeyFinding] = {}
        for result in segment_results:
            for index, finding in enumerate(result.key_findings):
                key = (result.segment.value, index)
                source[key] = finding
                rows.append({
                    "segment": key[0], "index": index, "claim": finding.claim,
                    "claim_type": finding.claim_type.value,
                    "evidence_ids": list(finding.evidence_ids),
                    "analytics_ids": list(finding.analytics_ids),
                    "tags": list(finding.tags),
                })
        if not rows:
            return
        system = """You are the final document-level editor for a neutral institutional
company report. Treat the report as one coherent document, not a collection of independent
sections. Review the full body for repeated facts, metrics, conclusions and explanations.
Keep a repeated fact only when the occurrence adds a genuinely new comparison, implication,
calculation, uncertainty or monitoring purpose. Otherwise keep its clearest and most
contextually appropriate occurrence, and mark later repetitions as drop or shorten them to
a necessary cross-reference without restating the same values. Key Takeaways are edited in
a separate pass and may intentionally summarize facts that appear once in the body.

Preserve all material information, neutrality, citations, factual qualifiers, periods and
uncertainties. Do not introduce facts, investment recommendations, causal claims or evidence
ids. A drop action is permitted only for information fully preserved in another supplied
body finding. Every kept or shortened finding must retain its original provenance. Return
JSON only."""
        schema = {
            "headlines": [{"segment": "exact supplied segment", "headline": "one line"}],
            "findings": [{
                "segment": "exact supplied segment", "index": "exact supplied integer",
                "action": "keep, shorten, or drop",
                "claim": "edited claim retaining the source meaning; empty only for drop",
            }],
        }
        prompt = (
            f"Company: {self.plan.company} ({self.plan.ticker or 'ticker unknown'})\n"
            f"Body findings: {rows}\nReturn this JSON shape: {schema}"
        )
        try:
            response = await self._client.complete_json(
                system, prompt, stage="synthesis_body")
        except Exception as exc:  # noqa: BLE001 - original findings remain valid
            log_event(logger, logging.WARNING, "LLM body synthesis failed; retaining agent text",
                      error=f"{type(exc).__name__}: {exc}")
            return
        payload = response.payload if isinstance(response.payload, dict) else {}
        for row in payload.get("findings", []):
            if not isinstance(row, dict):
                continue
            try:
                key = (str(row.get("segment", "")), int(row.get("index")))
            except (TypeError, ValueError):
                continue
            original = source.get(key)
            action = str(row.get("action", "keep")).strip().lower()
            claim = str(row.get("claim", "")).strip()
            if original is not None and action == "drop":
                self._llm_dropped.add(key)
            elif original is not None and claim:
                self._llm_body[key] = KeyFinding(
                    claim=claim, claim_type=original.claim_type,
                    evidence_ids=original.evidence_ids, analytics_ids=original.analytics_ids,
                    confidence=original.confidence, materiality=original.materiality,
                    tags=original.tags,
                )
        for row in payload.get("headlines", []):
            if isinstance(row, dict):
                segment = str(row.get("segment", "")).strip()
                headline = str(row.get("headline", "")).strip()
                if segment and headline:
                    self._llm_headlines[segment] = headline

    # -- LLM takeaway selection -------------------------------------------
    async def _build_takeaways(
        self, segment_results: tuple[SegmentResult, ...]
    ) -> ReportSectionDraft | None:
        pool: dict[str, KeyFinding] = {}
        rows: list[dict[str, Any]] = []
        for result in segment_results:
            for index, finding in enumerate(result.key_findings):
                fid = f"{result.segment.value}.{index}"
                pool[fid] = finding
                rows.append({
                    "id": fid,
                    "segment": result.segment.value,
                    "claim": finding.claim,
                    "claim_type": finding.claim_type.value,
                    "evidence_ids": list(finding.evidence_ids),
                    "analytics_ids": list(finding.analytics_ids),
                    "materiality": finding.materiality,
                    "tags": list(finding.tags),
                })
        if not rows:
            return None

        allowed_ids = {i for f in pool.values() for i in (*f.evidence_ids, *f.analytics_ids)}
        schema = {
            "summary": "one-line statement of the report's conclusion",
            "takeaways": [{
                "type": "existing or merged",
                "id": "an id from the supplied findings (type=existing only)",
                "claim": "new merged sentence (type=merged only)",
                "claim_type": sorted(_CLAIM_TYPES) + ["(type=merged only)"],
                "evidence_ids": ["ids copied from the supplied findings (type=merged only)"],
                "analytics_ids": ["ids copied from the supplied findings (type=merged only)"],
            }],
        }
        user_prompt = (
            f"Company: {self.plan.company} ({self.plan.ticker or 'ticker unknown'})\n"
            f"Findings from every segment: {rows}\n"
            "Select 4-6 takeaways when the source findings support that many. "
            f"Never exceed {_MAX_TAKEAWAYS}.\n"
            f"Return this JSON shape: {schema}"
        )

        try:
            response = await self._client.complete_json(
                _SYSTEM_PROMPT, user_prompt, stage="synthesis")
        except Exception as exc:  # noqa: BLE001 - fall back to the deterministic takeaways
            log_event(logger, logging.ERROR, "LLM synthesis failed; falling back",
                      error=f"{type(exc).__name__}: {exc}")
            return None

        log_event(
            logger, logging.INFO, "LLM synthesis takeaways generated",
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        )
        selected, dropped = self._findings_from_model(response.payload, pool, allowed_ids)
        if not selected:
            return None

        statements = self._statements(tuple(selected), dedupe=False)
        summary = str(response.payload.get("summary", "")).strip()
        if not summary:
            summary = self._takeaway_summary({r.segment: r for r in segment_results})

        if dropped:
            log_event(logger, logging.WARNING, "LLM takeaway dropped: untagged or unknown",
                      dropped=dropped)

        return ReportSectionDraft(
            section=ReportSection.KEY_TAKEAWAYS,
            title="Key Takeaways",
            summary=summary,
            statements=statements,
        )

    @staticmethod
    def _findings_from_model(
        payload: dict[str, Any], pool: dict[str, KeyFinding], allowed_ids: set[str],
    ) -> tuple[list[KeyFinding], int]:
        selected: list[KeyFinding] = []
        dropped = 0
        for row in payload.get("takeaways", [])[: _MAX_TAKEAWAYS * 2]:
            if not isinstance(row, dict):
                dropped += 1
                continue
            kind = str(row.get("type", "existing")).strip().lower()
            if kind == "existing":
                finding = pool.get(str(row.get("id", "")))
                if finding is None:
                    dropped += 1
                    continue
                selected.append(finding)
                continue
            if kind == "merged":
                claim = str(row.get("claim", "")).strip()
                evidence_ids = tuple(
                    str(e) for e in row.get("evidence_ids", []) if str(e) in allowed_ids)
                analytics_ids = tuple(
                    str(a) for a in row.get("analytics_ids", []) if str(a) in allowed_ids)
                if not claim or (not evidence_ids and not analytics_ids):
                    dropped += 1
                    continue
                claim_type_raw = str(row.get("claim_type", "interpretation"))
                claim_type = (
                    ClaimType(claim_type_raw) if claim_type_raw in _CLAIM_TYPES
                    else ClaimType.INTERPRETATION)
                selected.append(KeyFinding(
                    claim=claim,
                    claim_type=claim_type,
                    evidence_ids=evidence_ids,
                    analytics_ids=analytics_ids,
                    materiality=1,
                    tags=("llm_merged",),
                ))
                continue
            dropped += 1
        return selected[:_MAX_TAKEAWAYS], dropped
