"""LLM-backed segment agent: evidence-grounded, evidence-tagged narrative.

Every segment agent is now LLM-backed by default (there is no config toggle
any more). The agent is handed only the evidence and analytics already
retrieved for its segment by the EvidenceReader/AnalyticsBundle - the same
boundary every deterministic agent respects, no provider or network access -
and it is required to attach the evidence_id/analytics_id backing every claim
it makes, exactly like the deterministic agents' KeyFinding objects already
do.

The deterministic agents get "no invented facts" structurally, because they
are plain code that can only emit what it read. The LLM agent gets the same
guarantee through post-hoc validation instead: any claim whose cited ids do
not resolve against the evidence/analytics it was actually shown is dropped,
never trusted. One generic class serves all eight segments, since the
prompt-and-validate mechanics are identical and only the evidence in view
differs per segment. Numeric values shown in ``important_metrics`` are always
read directly off the evidence/analytics the reader retrieved (see
``_labelled_metric``/``metric_highlight``), never invented or recomputed by
the model, so the only thing the LLM actually controls is prose and framing.

:class:`VerifiedSegmentAgent` (bottom of this module) is what
:mod:`eq_report.agents.runner` actually instantiates: it pairs this LLM agent
with the module's original deterministic implementation from
``AGENT_REGISTRY``, which is kept as the fallback used when the LLM is not
configured or a call fails/errors, exactly the "verify or fall back to code"
pattern used by every other converted stage.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from ..config import ModelConfig
from ..domain.enums import ClaimType, Confidence, SegmentName
from ..domain.evidence import EvidenceItem
from ..domain.segment import KeyFinding, MetricHighlight, SegmentResult
from ..llm.client import LLMJSONResponse, OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from ..normalisation.canonical_metrics import display_label
from .base import AgentContext, SegmentAgent

logger = get_logger("agents.llm_agent")

_CLAIM_TYPES = {c.value for c in ClaimType}
_CONFIDENCE_ORDER = {
    Confidence.HIGH: 3, Confidence.MEDIUM: 2, Confidence.LOW: 1, Confidence.UNKNOWN: 0,
}

_INLINE_ID_BLOCK = re.compile(
    r"\s*\[(?:\s*(?:ev|an)_[a-zA-Z0-9]+\s*,?)+\s*\]\s*"
)

#: Point-in-time metrics: looked up without a fiscal period.
_POINT_METRICS: tuple[str, ...] = (
    "share_price", "market_cap", "forward_pe", "trailing_pe", "ev_to_sales",
    "price_to_sales", "price_target", "consensus_rating",
)

#: Metrics reported for a fiscal period: looked up against the latest period.
_PERIOD_METRICS: tuple[str, ...] = (
    "revenue", "gross_profit", "gross_margin", "operating_income", "operating_margin",
    "net_income", "net_margin", "eps_diluted", "operating_cash_flow", "capex",
    "free_cash_flow", "cash_and_equivalents", "total_debt", "rnd_expense",
    "consensus_revenue", "consensus_eps", "estimate_revisions_up", "estimate_revisions_down",
)

# Give the segment model a broad, bounded document pool and let the same call
# that writes the findings perform semantic selection.  This avoids the old
# failure mode where an entire natural-language research question was treated
# as one literal substring, while avoiding a second model call per segment.
_MAX_DOCUMENT_CANDIDATES = 30

_SYSTEM_PROMPT = """You are one analytical segment agent inside a neutral institutional
company research pipeline. You are given a fixed set of retrieved evidence rows and
computed analytics for one report segment, and nothing else. You may only state facts
present in the evidence or analytics you were given; you may not use general knowledge,
invent figures, or assume anything not in the supplied data. Never give a Buy/Hold/Sell
view, a price target, or a trade instruction; treat any analyst target you see only as an
external market statistic.

When evidence contains commentary from a named analyst, industry specialist, executive or
other key opinion leader, preserve the person's name, role or organisation, statement date
and source attribution. Label it as that person's view, forecast or interpretation; never
convert it into an established fact. Prefer a concise view/counterview pairing when evidence
supports both sides, and do not use anonymous commentary as a KOL statement.

Write like a neutral financial analyst producing company research, not an investment pitch.
A finding is not a data dump: lead with the point, then use the evidence, explain why it
matters for earnings, expectations, or valuation mechanics, and end with the specific
variable that would resolve the open question. Use comparisons and calculations supplied in
the analytics where available. Connect findings in a deliberate order: change -> driver ->
financial consequence -> expectation or valuation implication -> what would confirm or
change that picture.

Stay strictly on the observation side of the line between observation and conclusion.
"NVDA trades at a 15.4% forward P/E premium to peers and its reported revenue growth is 39
points above the peer average" is a supported observation. "The higher growth justifies the
premium" is a value judgment those two facts alone do not establish, and must not be
written. Concretely:

* Never write language that asserts or implies whether the company, its shares, or its
  valuation are good, bad, attractive, cheap, expensive, justified, or a buying/selling
  opportunity - that is an investment judgment, not analysis. Banned constructions include
  "supports its valuation premium", "central risk", "elevated expectations overtaking
  momentum", or any sentence a reader could summarise as "so you should/shouldn't buy this".
* Prefer explicitly neutral constructions: "X is above/below Y", "the available evidence is
  consistent with...", "the data cannot distinguish between...", "this metric can be
  monitored to assess...". State the comparison and let the reader draw any investment
  conclusion themselves.
* Never assert that one thing caused another (via "because", "driven by", "due to", "led
  to", etc.) unless the evidence itself is a management/reported statement of that causal
  link or a calculation that isolates it. Where you see two things move together without
  evidence of which drove which, say they moved together or coincided - not that one caused
  the other.
* When faster/slower growth, higher/lower margin, or similar could be explained by more than
  one thing the evidence cannot separate (e.g. market-wide growth vs. share gain; pricing vs.
  volume), say so explicitly rather than picking one explanation. This is itself a finding
  worth stating, not a gap to bury.
* Never write about the evidence itself. Sentences of the form "the supplied evidence does
  not establish X", "no data was provided for Y", "this cannot be verified" tell the reader
  about our data set, not about the company. Anything of that kind belongs in data_gaps,
  which is printed separately - and a finding must not be padded with a trailing caveat
  clause either.
* Never restate a figure another finding already used, in this segment or any other segment
  of the report. A metric may reappear only when its role in the argument changes, and then
  the sentence must say what changed - not repeat the original statement of it.
* Never state a number without saying what it means. A figure with no comparison, no
  consequence and no implication attached is a table row, and the report already has tables.
* Prefer "during the reported period" over language implying a durable trend, unless the
  supplied evidence spans enough periods to support a trend claim.

Return 3-6 substantive findings when the evidence supports them - fewer, if that is all the
evidence carries. Each finding should be one compact analytical paragraph of 2-3 sentences
(roughly 45-100 words), with one main idea and a clear logical bridge between fact and
implication. A short factual sentence is acceptable only when the evidence cannot support
any interpretation beyond the fact. Returning three strong findings is better than padding
to six.

Every finding you return must cite the evidence_id and/or analytics_id rows it rests on,
copied verbatim from the ids supplied to you. A finding with no matching id will be
discarded rather than trusted. Put ids only in the evidence_ids and analytics_ids JSON
fields; never print ids inside claim prose. If something material cannot be supported by the supplied
evidence, report it as a data gap instead of asserting it. Return JSON only, matching the
requested schema."""


class AgentLLMClient(Protocol):
    async def complete_json(
        self, system_prompt: str, user_prompt: str, *, stage: str = "",
    ) -> LLMJSONResponse: ...


class LLMSegmentAgent(SegmentAgent):
    """Generic, evidence-tagged LLM implementation of any one segment."""

    def __init__(
        self,
        segment: SegmentName,
        model_config: ModelConfig | None = None,
        *,
        client: AgentLLMClient | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self.segment = segment
        self._model_config = model_config
        self._client = client
        if self._client is None:
            if model_config is None or not model_config.enabled:
                raise ValueError(
                    "LLMSegmentAgent requires an enabled ModelConfig or an injected client")
            if model_config.provider.lower() != "openrouter":
                raise ValueError("LLMSegmentAgent supports MODEL_PROVIDER=openrouter")
            self._client = OpenRouterJSONClient(model_config, tracker=tracker)

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        evidence_rows, evidence_by_id = self._evidence_for(context)
        analytics_rows, analytics_by_id = self._analytics_for(context)

        if not evidence_rows and not analytics_rows:
            return SegmentResult(
                segment=self.segment,
                headline=f"No evidence available for {self.segment.value.replace('_', ' ')}",
                data_gaps=(self.gap(
                    f"No evidence or analytics were retrieved for {self.segment.value}.",
                    impact="This section cannot be written.",
                    segment=self.segment,
                ),),
                open_questions=self.questions(context),
            )

        response = await self._client.complete_json(
            _SYSTEM_PROMPT, self._user_prompt(context, evidence_rows, analytics_rows),
            stage=f"agent:{self.segment.value}")
        log_event(
            logger, logging.INFO, "LLM segment agent responded",
            segment=self.segment.value,
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
        )
        return self._result_from_model(context, response.payload, evidence_by_id, analytics_by_id)

    # -- evidence scoping --------------------------------------------------
    def _evidence_for(
        self, context: AgentContext
    ) -> tuple[list[dict[str, Any]], dict[str, EvidenceItem]]:
        reader = context.reader
        task = context.task
        latest = context.latest_period
        by_id: dict[str, EvidenceItem] = {}
        rows: list[dict[str, Any]] = []

        def add(item: EvidenceItem | None) -> None:
            if item is None or item.evidence_id in by_id:
                return
            by_id[item.evidence_id] = item
            rows.append({
                "evidence_id": item.evidence_id,
                "metric": item.metric,
                "value": item.value,
                "unit": item.unit,
                "period": item.period_label,
                "as_of": item.as_of.isoformat() if item.as_of else None,
                "claim_text": item.claim_text,
                "document_title": item.document_title,
                "source_type": item.source_type.value,
                "source_name": item.source_name,
                "confidence": item.confidence.value,
            })

        wanted_metrics = dict.fromkeys((*_POINT_METRICS, *(task.required_metrics if task else ())))
        for metric in wanted_metrics:
            add(reader.numeric(metric))
        wanted_period_metrics = dict.fromkeys(_PERIOD_METRICS)
        if latest:
            for metric in wanted_period_metrics:
                add(reader.numeric(metric, latest))
            for row in reader.segment_rows(latest):
                add(row)
            for row in reader.kpi_rows(latest):
                add(row)

        for item in reader.guidance():
            add(item)

        for peer in context.plan.peers:
            add(reader.peer_value(peer, "forward_pe"))
            add(reader.peer_value(peer, "revenue_growth_yoy_reported"))

        # The model is the semantic reranker: include a recent, source-diverse
        # candidate pool, plus any exact lexical hits first.  It may cite only
        # ids from this pool, so retrieval remains auditable.
        questions = [q.text for q in context.plan.questions_for(self.segment)]
        for item in reader.documents_matching(questions, limit=10):
            add(item)
        for item in reader.documents(limit=_MAX_DOCUMENT_CANDIDATES):
            add(item)

        return rows, by_id

    def _analytics_for(self, context: AgentContext):
        task = context.task
        wanted = set(task.required_analytics) if task else set()
        by_id: dict[str, Any] = {}
        rows: list[dict[str, Any]] = []
        for result in context.analytics.results:
            if result.metric not in wanted:
                continue
            by_id[result.analytics_id] = result
            rows.append({
                "analytics_id": result.analytics_id,
                "metric": result.metric,
                "value": result.value,
                "unit": result.unit,
                "label": result.label,
                "period": result.period,
                "comparison_period": result.comparison_period,
                "confidence": result.confidence.value,
            })
        return rows, by_id

    def _labelled_metric(
        self, item: EvidenceItem, context: AgentContext
    ) -> MetricHighlight | None:
        """A metric_highlight with a label that distinguishes same-metric rows.

        Segment rows, KPI rows and peer values all share one canonical metric
        id ("segment_revenue", "kpi", "forward_pe", ...); without a
        distinguishing label they render as repeated, unreadable table rows
        ("Segment revenue", "Segment revenue", ...). The deterministic agents
        avoid this by building each MetricHighlight with an explicit label; the
        generic LLM agent has to derive the same distinction from metadata.
        """
        label = None
        if item.metric == "segment_revenue":
            name = item.metadata.get("segment_name")
            if name:
                label = f"{name} revenue"
        elif item.metric == "kpi":
            name = item.metadata.get("kpi_name")
            if name:
                label = str(name)
        elif item.ticker and item.ticker != context.ticker:
            label = f"{item.ticker} {display_label(item.metric or 'metric')}"
        return self.metric_highlight(item, label=label)

    def _user_prompt(
        self, context: AgentContext, evidence_rows: list[dict[str, Any]],
        analytics_rows: list[dict[str, Any]],
    ) -> str:
        schema = {
            "headline": "one-sentence analytical conclusion for this section",
            "findings": [{
                "claim": "2-3 sentence evidence-grounded analytical paragraph: point, evidence, why it matters/implication",
                "claim_type": sorted(_CLAIM_TYPES),
                "evidence_ids": ["ids copied from the supplied evidence, or []"],
                "analytics_ids": ["ids copied from the supplied analytics, or []"],
                "materiality": "integer 1-3, 1 = most material",
                "tags": ["free-form short tags, e.g. risk or catalyst where relevant"],
            }],
            "data_gaps": [{
                "description": "what could not be answered from the supplied data",
                "impact": "why it matters",
                "priority": (
                    "high (blocks a central question this section needs to answer), "
                    "medium (limits but does not block the analysis), or low "
                    "(context/detail only)"
                ),
            }],
        }
        return (
            f"Segment: {self.segment.value}\n"
            f"Company: {context.company} ({context.ticker or 'ticker unknown'})\n"
            f"Objective: {context.task.objective if context.task else ''}\n"
            f"Questions to address: {list(self.questions(context))}\n"
            f"Evidence candidates (semantically select and cite only the rows relevant "
            f"to this segment; only these ids may be cited): {evidence_rows}\n"
            f"Analytics rows (only these ids may be cited): {analytics_rows}\n"
            "Keep the findings in reading order; the report will preserve this order.\n"
            f"Return this JSON shape: {schema}"
        )

    # -- validation ----------------------------------------------------------
    def _result_from_model(
        self,
        context: AgentContext,
        payload: dict[str, Any],
        evidence_by_id: dict[str, EvidenceItem],
        analytics_by_id: dict[str, Any],
    ) -> SegmentResult:
        findings: list[KeyFinding] = []
        dropped = 0
        for row in payload.get("findings", []):
            if not isinstance(row, dict):
                dropped += 1
                continue
            claim = _INLINE_ID_BLOCK.sub(" ", str(row.get("claim", ""))).strip()
            evidence_ids = tuple(
                str(e) for e in row.get("evidence_ids", []) if str(e) in evidence_by_id)
            analytics_ids = tuple(
                str(a) for a in row.get("analytics_ids", []) if str(a) in analytics_by_id)
            if not claim or (not evidence_ids and not analytics_ids):
                dropped += 1
                continue
            claim_type_raw = str(row.get("claim_type", "interpretation"))
            claim_type = (
                ClaimType(claim_type_raw) if claim_type_raw in _CLAIM_TYPES
                else ClaimType.INTERPRETATION)
            confidences = [evidence_by_id[e].confidence for e in evidence_ids]
            try:
                materiality = max(1, min(3, int(row.get("materiality", 2))))
            except (TypeError, ValueError):
                materiality = 2
            findings.append(KeyFinding(
                claim=claim,
                claim_type=claim_type,
                evidence_ids=evidence_ids,
                analytics_ids=analytics_ids,
                confidence=(
                    min(confidences, key=lambda c: _CONFIDENCE_ORDER[c]) if confidences
                    else Confidence.MEDIUM),
                materiality=materiality,
                tags=tuple(str(t) for t in row.get("tags", [])),
            ))

        data_gaps = tuple(
            self.gap(str(row.get("description", "")).strip(),
                     impact=str(row.get("impact", "")), segment=self.segment,
                     priority=_validated_priority(row.get("priority")))
            for row in payload.get("data_gaps", [])
            if isinstance(row, dict) and str(row.get("description", "")).strip()
        )

        # Only the evidence a finding actually cites becomes a table row. The
        # full retrieved pool (evidence_by_id) is much broader than what any
        # one segment ends up discussing - point metrics and period metrics
        # are fetched for every segment regardless of relevance, so agents
        # would otherwise all print the same oversized, near-identical table.
        cited_evidence_ids = {e for f in findings for e in f.evidence_ids}
        metrics = self.compact_metrics(
            self._labelled_metric(item, context)
            for eid, item in evidence_by_id.items()
            if eid in cited_evidence_ids and item.value is not None and item.metric
        )

        if dropped:
            log_event(
                logger, logging.WARNING, "LLM finding dropped: untagged or empty claim",
                segment=self.segment.value, dropped=dropped,
            )

        metadata: dict[str, Any] = {}
        if dropped:
            metadata["llm_findings_dropped"] = dropped

        return SegmentResult(
            segment=self.segment,
            headline=str(payload.get("headline", "")).strip()
            or f"{self.segment.value.replace('_', ' ').title()}",
            key_findings=tuple(findings),
            important_metrics=metrics,
            data_gaps=data_gaps,
            open_questions=self.questions(context),
            metadata=metadata,
        )


def _validated_priority(raw: Any) -> str:
    """A model-proposed data-gap priority, or "medium" if it did not supply
    one of the three accepted values - never trust free text here, since the
    renderer groups the Data Gaps table by exactly these three."""
    value = str(raw or "").strip().lower()
    return value if value in {"high", "medium", "low"} else "medium"


class VerifiedSegmentAgent(SegmentAgent):
    """LLM-backed segment agent with a deterministic fallback.

    Used for every segment now that there is no deterministic/LLM toggle: the
    generic :class:`LLMSegmentAgent` above is the primary path, and the
    original deterministic rule-based agent for the segment (from
    ``AGENT_REGISTRY``) is kept as the fallback used when the LLM is not
    configured, or the call itself raises or returns something that cannot be
    turned into a ``SegmentResult``. The fallback is recorded on the result's
    metadata so it is visible in the run's QA trail, not silent.
    """

    def __init__(
        self,
        segment: SegmentName,
        deterministic: SegmentAgent,
        model_config: ModelConfig | None = None,
        *,
        client: AgentLLMClient | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self.segment = segment
        self._deterministic = deterministic
        self._llm_enabled = bool(
            client is not None or (model_config is not None and model_config.enabled))
        self._llm = (
            LLMSegmentAgent(segment, model_config, client=client, tracker=tracker)
            if self._llm_enabled else None
        )

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        if self._llm is None:
            result = await self._deterministic._analyse(context)
            result.metadata["llm_agent_skipped"] = "no LLM model configured"
            return result
        try:
            return await self._llm._analyse(context)
        except Exception as exc:  # noqa: BLE001 - fall back to the deterministic agent
            log_event(
                logger, logging.WARNING, "LLM segment agent failed; using deterministic agent",
                segment=self.segment.value, error=f"{type(exc).__name__}: {exc}",
            )
            result = await self._deterministic._analyse(context)
            result.metadata["llm_agent_fallback_reason"] = f"{type(exc).__name__}: {exc}"
            return result
