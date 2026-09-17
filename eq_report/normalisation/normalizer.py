"""Stage 4 - the normalisation layer.

Converts raw provider output into canonical EvidenceItems. This is the only
place that decides what a value *means*: which canonical metric it is, what unit
it is in, which fiscal period it belongs to, and which category of evidence it
becomes.

Two rules matter most:

1. The raw metric name and raw value are always preserved on the evidence item,
   so any number in the report can be traced back to what the provider actually
   said.
2. A value that cannot be normalised is *rejected*, with a reason, rather than
   coerced. Rejections are reported as data gaps; they never become evidence.

LLM-backed interpretation, with code verification
---------------------------------------------------
When a model is configured, normalisation asks it - in one batched call per
branch - to interpret each raw observation into a canonical metric id and a
clean numeric value. That LLM output is never trusted directly: the
pre-existing deterministic parsers in this module (``cm.canonicalise_metric``,
``parse_number``, ``normalise_percent``, ...) still run on every observation
regardless, and are what actually decides the stored value. The LLM's
proposed number is compared against that deterministic value with
:func:`eq_report.llm.verify.verify_number`; the two normally agree, and when
they do not, the deterministic number wins and the divergence is recorded on
the evidence item's metadata (``llm_value_overridden``) so it is visible to
QA and the run manifest. The LLM's proposed canonical metric id is used only
to *resolve* an otherwise-unmapped provider metric (never to override a metric
the deterministic alias table already recognised), and only when it names a
metric the deterministic vocabulary actually knows about
(``cm.all_canonical_metrics()``).

If no model is configured, or the batched call fails or returns unparseable
output, normalisation falls back to the deterministic parsers alone - exactly
the previous behaviour - and records why on the result's ``warnings``.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

from ..config import ModelConfig
from ..domain.enums import Confidence, EvidenceCategory, SourceType
from ..domain.evidence import EvidenceItem, make_evidence_id
from ..domain.observation import RawDocumentPassage, RawObservation
from ..domain.plan import ResearchPlan
from ..errors import NormalisationError
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json, verify_number
from ..logging_setup import get_logger, log_event
from . import canonical_metrics as cm
from .dates import normalise_date, normalise_fiscal_period
from .reconciliation import reconcile
from .units import (
    UNIT_COUNT,
    UNIT_CURRENCY,
    UNIT_MULTIPLE,
    UNIT_PER_SHARE,
    UNIT_PERCENT,
    UNIT_TEXT,
    normalise_currency,
    normalise_percent,
    parse_number,
)

logger = get_logger("normalisation")

_LLM_SYSTEM_PROMPT = """You classify raw provider metric names for a neutral institutional
equity-research pipeline. For each numbered metric shape, propose exactly one canonical metric
id from the supplied vocabulary, or the literal string "unmapped". Do not infer values, units,
periods, or accounting bases: deterministic code handles them. Return JSON only."""

# The API frequently returns tens of thousands of time-series leaves. Sending
# every leaf in one prompt caused OpenRouter HTTP 400 request/context failures.
# Metric *shapes* repeat across those leaves, so classify them in small sections.
_LLM_METRIC_SHAPES_PER_BATCH = 150
_LLM_MAX_SHAPES_PER_SECTION = 300
_LLM_METRIC_TERMS = (
    "revenue", "sales", "eps", "earning", "margin", "income", "cash", "debt",
    "capex", "capital", "price", "market", "enterprise", "share", "volume",
    "target", "estimate", "guidance", "multiple", "pe", "ev_",
)


@dataclass(frozen=True, slots=True)
class Rejection:
    """A raw datapoint that could not be normalised, and why."""

    branch: str
    raw_metric: str
    raw_value: Any
    reason: str
    source_name: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "branch": self.branch,
            "raw_metric": self.raw_metric,
            "raw_value": self.raw_value,
            "reason": self.reason,
            "source_name": self.source_name,
        }


@dataclass(frozen=True, slots=True)
class NormalisationResult:
    """Normalised evidence plus everything that was rejected or unmapped."""

    evidence: tuple[EvidenceItem, ...] = ()
    rejections: tuple[Rejection, ...] = ()
    unmapped_metrics: tuple[str, ...] = ()
    warnings: tuple[str, ...] = field(default=())

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_count": len(self.evidence),
            "rejections": [r.to_dict() for r in self.rejections],
            "unmapped_metrics": list(self.unmapped_metrics),
            "warnings": list(self.warnings),
        }


class Normalizer:
    """Turns raw observations and document passages into canonical evidence."""

    def __init__(
        self, report_run_id: str, plan: ResearchPlan,
        model_config: ModelConfig | None = None,
        *,
        tracker: UsageTracker | None = None,
    ) -> None:
        self.report_run_id = report_run_id
        self.plan = plan
        self.company = self._canonical_company(plan.company)
        self.ticker = plan.ticker
        self._model_config = model_config
        self._tracker = tracker

    # -- public API ------------------------------------------------------
    async def normalize(
        self,
        market_observations: tuple[RawObservation, ...] = (),
        fundamental_observations: tuple[RawObservation, ...] = (),
        document_passages: tuple[RawDocumentPassage, ...] = (),
    ) -> NormalisationResult:
        """Normalise all three branches into one evidence set."""
        evidence: list[EvidenceItem] = []
        rejections: list[Rejection] = []
        unmapped: list[str] = []
        extra_warnings: list[str] = []

        branches: tuple[tuple[str, tuple[RawObservation, ...], EvidenceCategory], ...] = (
            ("market_data", market_observations, EvidenceCategory.MARKET),
            ("fundamentals", fundamental_observations, EvidenceCategory.FUNDAMENTAL),
        )
        llm_hints, hint_note = await self._llm_hints(branches)
        if hint_note:
            extra_warnings.append(hint_note)
        overrides = 0
        reclassified = 0

        for branch, observations, default_category in branches:
            for index, observation in enumerate(observations):
                try:
                    item, was_known, flag = self._normalise_observation(
                        observation, default_category,
                        llm_hints.get((branch, index)),
                    )
                except NormalisationError as exc:
                    rejections.append(Rejection(
                        branch=branch,
                        raw_metric=observation.metric,
                        raw_value=observation.value,
                        reason=str(exc),
                        source_name=observation.source.source_name,
                    ))
                    continue
                if not was_known and observation.metric not in unmapped:
                    unmapped.append(observation.metric)
                if flag == "overridden":
                    overrides += 1
                elif flag == "reclassified":
                    reclassified += 1
                evidence.append(item)

        if overrides:
            extra_warnings.append(
                f"{overrides} LLM-proposed value(s) diverged from the deterministic "
                "recomputation beyond tolerance; the deterministic value was used in "
                "each case (see evidence metadata 'llm_value_overridden')."
            )
        if reclassified:
            extra_warnings.append(
                f"{reclassified} otherwise-unmapped provider metric(s) were resolved to a "
                "canonical metric using the LLM's proposal (see evidence metadata "
                "'llm_reclassified_metric')."
            )

        for passage in document_passages:
            try:
                evidence.append(self._normalise_passage(passage))
            except NormalisationError as exc:
                rejections.append(Rejection(
                    branch="documents",
                    raw_metric="document_passage",
                    raw_value=passage.title,
                    reason=str(exc),
                    source_name=passage.source.source_name,
                ))

        deduped = list(reconcile(self._dedupe(evidence)))
        warnings: list[str] = list(extra_warnings)
        if unmapped:
            warnings.append(
                "Unmapped provider metrics kept with an 'unmapped.' prefix: "
                + ", ".join(sorted(unmapped))
            )

        result = NormalisationResult(
            evidence=tuple(deduped),
            rejections=tuple(rejections),
            unmapped_metrics=tuple(unmapped),
            warnings=tuple(warnings),
        )
        log_event(
            logger, logging.INFO, "normalisation complete",
            evidence=len(result.evidence), rejected=len(result.rejections),
            unmapped=len(result.unmapped_metrics),
            deduplicated=len(evidence) - len(deduped),
        )
        for rejection in rejections:
            log_event(logger, logging.WARNING, "value rejected",
                      branch=rejection.branch, metric=rejection.raw_metric,
                      reason=rejection.reason)
        return result

    # -- observation normalisation ---------------------------------------
    def _normalise_observation(
        self, observation: RawObservation, default_category: EvidenceCategory,
        llm_hint: dict[str, Any] | None = None,
    ) -> tuple[EvidenceItem, bool, str | None]:
        """Normalise one observation.

        ``llm_hint`` (from :meth:`_llm_hints`, or ``None`` when the model was
        not consulted or gave no usable row for this item) may resolve an
        otherwise-unmapped metric and is cross-checked against the
        deterministic numeric value; see the module docstring. Returns
        ``(item, was_known, flag)`` where ``flag`` is ``"overridden"``,
        ``"reclassified"``, or ``None``.
        """
        canonical_metric, was_known = cm.canonicalise_metric(observation.metric)
        flag: str | None = None
        if not was_known and llm_hint:
            proposed = str(llm_hint.get("canonical_metric") or "").strip()
            if proposed and proposed != "unmapped" and proposed in cm.all_canonical_metrics():
                canonical_metric = proposed
                was_known = True
                flag = "reclassified"
        category = cm.category_for_metric(canonical_metric, default_category)
        period = normalise_fiscal_period(observation.period, observation.period_end)
        as_of = normalise_date(observation.as_of, field_name="as_of")
        currency = normalise_currency(observation.currency, "USD")
        company = self._canonical_company(observation.company or self.company)
        ticker = self._canonical_ticker(observation.ticker)

        if cm.is_text_metric(canonical_metric):
            text = self._require_text(observation.value, canonical_metric)
            return (
                EvidenceItem(
                    evidence_id=make_evidence_id(
                        self.report_run_id, ticker or company, canonical_metric,
                        period.label if period else "", as_of or "",
                        observation.source.source_id, text,
                    ),
                    report_run_id=self.report_run_id,
                    company=company,
                    ticker=ticker,
                    category=category,
                    metric=canonical_metric,
                    value=None,
                    unit=UNIT_TEXT,
                    currency=None,
                    period=period,
                    as_of=as_of,
                    source_id=observation.source.source_id,
                    source_name=observation.source.source_name,
                    source_type=observation.source.source_type,
                    source_url=observation.source.source_url,
                    retrieval_provider=observation.source.retrieval_provider,
                    retrieval_url=observation.source.retrieval_url,
                    original_source_name=observation.source.original_source_name,
                    original_source_url=observation.source.original_source_url,
                    original_publication_date=normalise_date(observation.source.original_publication_date, field_name="original_publication_date"),
                    claim_text=text,
                    retrieved_at=observation.retrieved_at,
                    confidence=self._confidence(observation.confidence),
                    raw_metric=observation.metric,
                    raw_value=observation.value,
                    metadata=dict(observation.metadata),
                ),
                was_known,
                flag,
            )

        parsed = parse_number(observation.value, field_name=observation.metric)
        unit = self._unit_for(canonical_metric, observation.unit, parsed.is_percent)
        value = parsed.value
        if unit == UNIT_PERCENT:
            value = normalise_percent(value, source_looked_like_percent=parsed.is_percent)
        if unit in {UNIT_MULTIPLE, UNIT_COUNT, UNIT_PERCENT}:
            currency = None
        elif parsed.currency:
            currency = parsed.currency

        metadata = dict(observation.metadata)
        if parsed.scale_applied != 1.0:
            metadata["scale_applied"] = parsed.scale_applied
        if flag == "reclassified":
            metadata["llm_reclassified_metric"] = canonical_metric

        if llm_hint is not None and "value" in llm_hint:
            verified = verify_number(
                llm_hint.get("value"), value,
                label=f"{canonical_metric} ({observation.source.source_name})",
            )
            if verified.overridden and verified.llm_value is not None:
                metadata["llm_value_overridden"] = verified.flag
                flag = flag or "overridden"
            # value itself is always verified.value == the deterministic number
            # here (verify_number only returns the LLM's number when it agreed
            # within tolerance, in which case it is numerically the same value
            # to within that tolerance) - the assignment is kept explicit for
            # clarity and so future tolerance changes stay correct.
            value = verified.value

        # segment_revenue / kpi rows are only meaningful with their label, so the
        # label participates in the evidence id to keep rows distinct.
        discriminator = metadata.get("segment_name") or metadata.get("kpi_name") or ""

        item = EvidenceItem(
            evidence_id=make_evidence_id(
                self.report_run_id, ticker or company, canonical_metric,
                period.label if period else "", as_of or "",
                observation.source.source_id, discriminator,
                metadata.get("series", ""),
            ),
            report_run_id=self.report_run_id,
            company=company,
            ticker=ticker,
            category=category,
            metric=canonical_metric,
            value=value,
            unit=unit,
            currency=currency,
            period=period,
            as_of=as_of,
            source_id=observation.source.source_id,
            source_name=observation.source.source_name,
            source_type=observation.source.source_type,
            source_url=observation.source.source_url,
            retrieval_provider=observation.source.retrieval_provider,
            retrieval_url=observation.source.retrieval_url,
            original_source_name=observation.source.original_source_name,
            original_source_url=observation.source.original_source_url,
            original_publication_date=normalise_date(observation.source.original_publication_date, field_name="original_publication_date"),
            frequency=("annual" if period and period.is_annual else "quarterly" if period else observation.metadata.get("frequency")),
            retrieved_at=observation.retrieved_at,
            confidence=self._confidence(observation.confidence),
            raw_metric=observation.metric,
            raw_value=observation.value,
            metadata=metadata,
        )
        return item, was_known, flag

    # -- LLM-assisted interpretation ---------------------------------------
    async def _llm_hints(
        self,
        branches: tuple[tuple[str, tuple[RawObservation, ...], EvidenceCategory], ...],
    ) -> tuple[dict[tuple[str, int], dict[str, Any]], str | None]:
        """Classify unresolved metric shapes in small, provider-separated batches.

        Returns ``({(branch, index): {...}}, note)``. ``note`` is a warning
        string when the model was not configured or the call failed; the
        empty dict is always a safe fallback (every caller treats a missing
        hint as "use the deterministic parsers alone", which is what already
        happened before this method existed).
        """
        if self._model_config is None or not self._model_config.enabled:
            return {}, "no LLM model configured; deterministic parsers ran alone."
        hints: dict[tuple[str, int], dict[str, Any]] = {}
        failures: list[str] = []
        known_metrics = sorted(cm.all_canonical_metrics())
        schema = {"items": [{
            "index": "the integer index supplied",
            "canonical_metric": known_metrics + ["unmapped"],
        }]}

        for branch, observations, _category in branches:
            # Map one representative metric shape to every unresolved row that
            # shares it. Array indices are intentionally collapsed: e.g.
            # ``NVDA.184.PX_LAST`` and ``NVDA.185.PX_LAST`` are one shape.
            grouped: dict[str, list[tuple[int, RawObservation]]] = {}
            for index, observation in enumerate(observations):
                _metric, known = cm.canonicalise_metric(observation.metric)
                if known:
                    continue
                shape = self._metric_shape(observation.metric)
                grouped.setdefault(shape, []).append((index, observation))

            representatives = sorted(
                grouped.items(), key=lambda entry: self._metric_priority(entry[0]), reverse=True)
            skipped = max(0, len(representatives) - _LLM_MAX_SHAPES_PER_SECTION)
            representatives = representatives[:_LLM_MAX_SHAPES_PER_SECTION]
            if skipped:
                failures.append(
                    f"{branch}: skipped LLM classification for {skipped} low-priority metric shapes "
                    "(deterministic unmapped handling retained them)")
            for offset in range(0, len(representatives), _LLM_METRIC_SHAPES_PER_BATCH):
                chunk = representatives[offset:offset + _LLM_METRIC_SHAPES_PER_BATCH]
                rows = [
                    {"index": position, "raw_metric": shape,
                     "example_unit": records[0][1].unit,
                     "example_value": str(records[0][1].value)[:80]}
                    for position, (shape, records) in enumerate(chunk)
                ]
                prompt = (
                    f"Company: {self.plan.company} (ticker {self.ticker or 'unknown'}).\n"
                    f"Provider section: {branch}; metric-shape batch "
                    f"{offset // _LLM_METRIC_SHAPES_PER_BATCH + 1}.\n"
                    f"Metric shapes: {rows}\nReturn this JSON shape: {schema}"
                )
                response, error = await safe_complete_json(
                    self._model_config, _LLM_SYSTEM_PROMPT, prompt,
                    tracker=self._tracker, stage=f"normalisation:{branch}")
                if response is None:
                    failures.append(f"{branch} batch {offset // _LLM_METRIC_SHAPES_PER_BATCH + 1}: {error}")
                    continue
                for row in response.payload.get("items", []) if isinstance(response.payload, dict) else []:
                    if not isinstance(row, dict):
                        continue
                    try:
                        position = int(row.get("index"))
                    except (TypeError, ValueError):
                        continue
                    if not (0 <= position < len(chunk)):
                        continue
                    _shape, records = chunk[position]
                    for observation_index, _observation in records:
                        hints[(branch, observation_index)] = row
        note = None
        if failures:
            note = ("Some LLM metric-classification batches failed; deterministic parsing ran for "
                    f"those rows ({'; '.join(failures[:3])}{'...' if len(failures) > 3 else ''}).")
        return hints, note

    @staticmethod
    def _metric_shape(metric: str) -> str:
        """Collapse array indexes without changing the provider's field names."""
        return re.sub(r"(?<=\.)\d+(?=\.|$)", "#", str(metric))

    @staticmethod
    def _metric_priority(shape: str) -> int:
        """Bound LLM work to fields plausibly useful to a research report."""
        lowered = shape.lower()
        return sum(1 for term in _LLM_METRIC_TERMS if term in lowered)

    # -- passage normalisation -------------------------------------------
    def _normalise_passage(self, passage: RawDocumentPassage) -> EvidenceItem:
        text = self._require_text(passage.text, "document_passage")
        published_at = normalise_date(passage.published_at, field_name="published_at")
        if not passage.title or not passage.title.strip():
            raise NormalisationError("title", passage.title, "document title is missing")

        metadata = dict(passage.metadata)
        if passage.speaker:
            metadata["speaker"] = passage.speaker
        metadata["document_type"] = passage.source.source_type.value

        return EvidenceItem(
            evidence_id=make_evidence_id(
                self.report_run_id, passage.source.source_id, passage.section or "",
                passage.speaker or "", text,
            ),
            report_run_id=self.report_run_id,
            company=self._canonical_company(passage.company or self.company),
            ticker=self._canonical_ticker(passage.ticker),
            category=EvidenceCategory.DOCUMENT,
            metric=None,
            value=None,
            unit=None,
            currency=None,
            period=None,
            as_of=published_at,
            source_id=passage.source.source_id,
            source_name=passage.source.source_name,
            source_type=passage.source.source_type,
            source_url=passage.source.source_url,
            retrieval_provider=passage.source.retrieval_provider,
            retrieval_url=passage.source.retrieval_url,
            original_source_name=passage.source.original_source_name,
            original_source_url=passage.source.original_source_url,
            original_publication_date=normalise_date(passage.source.original_publication_date, field_name="original_publication_date"),
            claim_text=self._collapse_whitespace(text),
            document_title=passage.title.strip(),
            published_at=published_at,
            retrieved_at=passage.retrieved_at,
            confidence=self._confidence(passage.confidence),
            raw_metric=None,
            raw_value=None,
            metadata={"section": passage.section, **metadata},
        )

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _require_text(value: Any, field_name: str) -> str:
        if value is None:
            raise NormalisationError(field_name, value, "text value is null")
        text = str(value).strip()
        if not text:
            raise NormalisationError(field_name, value, "text value is empty")
        return text

    @staticmethod
    def _canonical_company(name: str | None) -> str:
        """Strip common corporate suffixes so one company has one name.

        Prevents "NVIDIA", "NVIDIA Corp" and "NVIDIA Corporation" appearing as
        three different entities in the Evidence Store.
        """
        if not name:
            return ""
        text = re.sub(r"\s+", " ", str(name).strip())
        text = re.sub(
            r"[,]?\s+(inc|inc\.|corp|corp\.|corporation|co|co\.|ltd|ltd\.|plc|"
            r"llc|nv|sa|ag|holdings)$",
            "", text, flags=re.IGNORECASE,
        )
        return text.strip()

    @staticmethod
    def _canonical_ticker(ticker: str | None) -> str | None:
        if not ticker:
            return None
        text = str(ticker).strip().upper()
        # Strip an exchange prefix such as "NASDAQ:NVDA".
        if ":" in text:
            text = text.split(":")[-1].strip()
        return text or None

    @staticmethod
    def _confidence(value: Confidence | str | None) -> Confidence:
        if isinstance(value, Confidence):
            return value
        if value is None:
            return Confidence.UNKNOWN
        try:
            return Confidence(str(value).lower())
        except ValueError:
            return Confidence.UNKNOWN

    def _unit_for(
        self, canonical_metric: str, provider_unit: str | None, looked_like_percent: bool
    ) -> str:
        """Decide the canonical unit for a metric.

        Metric identity wins over the provider's unit string: a gross margin is a
        percentage whatever the vendor called it.
        """
        if cm.is_percent_metric(canonical_metric) or looked_like_percent:
            return UNIT_PERCENT
        if cm.is_multiple_metric(canonical_metric):
            return UNIT_MULTIPLE
        if canonical_metric in cm.COUNT_METRICS:
            return UNIT_COUNT
        if canonical_metric in cm.PER_SHARE_METRICS:
            return UNIT_PER_SHARE
        if provider_unit:
            cleaned = str(provider_unit).strip()
            if cleaned.upper() in {"USD", "EUR", "GBP", "JPY"}:
                return UNIT_CURRENCY
            if cleaned in {"%", "pct", "percent"}:
                return UNIT_PERCENT
            if cleaned in {"x", "X"}:
                return UNIT_MULTIPLE
        return UNIT_CURRENCY

    @staticmethod
    def _dedupe(items: list[EvidenceItem]) -> list[EvidenceItem]:
        """Drop exact duplicates, keeping the first (highest-priority) source."""
        seen: set[str] = set()
        out: list[EvidenceItem] = []
        for item in items:
            if item.evidence_id in seen:
                continue
            seen.add(item.evidence_id)
            out.append(item)
        return out


def normalise_source_type(raw: str | None, default: SourceType = SourceType.NEWS) -> SourceType:
    """Map a loose source-category string onto the SourceType vocabulary."""
    if not raw:
        return default
    slug = re.sub(r"[^a-z0-9]+", "_", str(raw).strip().lower()).strip("_")
    try:
        return SourceType(slug)
    except ValueError:
        pass
    aliases = {
        "10_q": SourceType.COMPANY_FILING,
        "10_k": SourceType.COMPANY_FILING,
        "8_k": SourceType.COMPANY_ANNOUNCEMENT,
        "sec_filing": SourceType.COMPANY_FILING,
        "filing": SourceType.COMPANY_FILING,
        "press_release": SourceType.COMPANY_ANNOUNCEMENT,
        "transcript": SourceType.EARNINGS_CALL,
        "conference_call": SourceType.EARNINGS_CALL,
        "deck": SourceType.INVESTOR_PRESENTATION,
        "presentation": SourceType.INVESTOR_PRESENTATION,
        "consensus": SourceType.SELL_SIDE_CONSENSUS,
        "quote": SourceType.MARKET_DATA,
    }
    return aliases.get(slug, default)
