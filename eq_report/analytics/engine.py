"""Stage 6 - the Analytics Engine.

Reads evidence and applies the pure functions in :mod:`calculations` to emit
AnalyticsResults that each carry the evidence ids they were derived from. Those
deterministic calculations are kept as the ground truth for every number in
the report - see ``compute_sync`` - because a financial-report engine getting
arithmetic wrong is exactly the failure mode this pipeline cannot afford.

LLM-backed cross-check
-----------------------
When a model is configured, ``compute`` additionally sends the whole batch of
computed (metric, formula, inputs, value) rows to the LLM in one call and asks
it to independently recompute each one. This is deliberately structured so the
LLM never gets to *set* a number: the deterministic value already is the
result before the LLM is consulted at all. The LLM's independent recomputation
is compared against the stored value with
:func:`eq_report.llm.verify.verify_number`; any disagreement beyond tolerance
is recorded on the result's ``metadata`` (``llm_disagreement``) as a QA-visible
flag rather than changing the number, since the deterministic figure is the
one that must always be trusted here. Results are also flagged
``llm_verified: True`` when the LLM agreed with the computed number, so a QA
reader can see an analytic was independently corroborated.

A missing input produces a recorded skip rather than an exception, so partial
data still yields partial analytics.
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Callable

from ..config import ModelConfig
from ..domain.analytics import AnalyticsBundle, AnalyticsResult, make_analytics_id
from ..domain.enums import Confidence
from ..domain.evidence import EvidenceItem
from ..domain.plan import ResearchPlan
from ..errors import AnalyticsError
from ..evidence.reader import EvidenceReader
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json, verify_number
from ..logging_setup import get_logger, log_event
from ..normalisation.dates import (
    normalise_fiscal_period,
    previous_quarter_period,
    year_ago_period,
)
from ..normalisation.units import (
    UNIT_COUNT,
    UNIT_MULTIPLE,
    UNIT_PER_SHARE,
    UNIT_PERCENT,
    UNIT_PERCENTAGE_POINTS,
)
from . import calculations as calc

logger = get_logger("analytics")

_CROSS_CHECK_SYSTEM_PROMPT = """You are the analytics verification layer of a neutral
institutional equity research pipeline. You are given a batch of already-computed analytics
rows, each with its formula (in plain terms) and the numeric inputs used. Independently
recompute the value implied by each row's formula and inputs, and report the number you get.
Never adjust the inputs or use outside knowledge - only recompute from what you were given.
Return JSON only, matching the requested schema."""


class AnalyticsEngine:
    """Computes the analytics the research plan asked for."""

    def __init__(
        self, report_run_id: str, reader: EvidenceReader,
        model_config: ModelConfig | None = None,
        *,
        tracker: UsageTracker | None = None,
    ) -> None:
        self.report_run_id = report_run_id
        self.reader = reader
        self._model_config = model_config
        self._tracker = tracker
        self._results: list[AnalyticsResult] = []
        self._skips: list[str] = []

    async def compute(self, plan: ResearchPlan) -> AnalyticsBundle:
        """Deterministic computation, then an LLM cross-check pass (see module docstring)."""
        bundle = self.compute_sync(plan)
        await self._llm_cross_check(bundle)
        return bundle

    async def _llm_cross_check(self, bundle: AnalyticsBundle) -> None:
        if not bundle.results:
            return
        rows = [{
            "index": i, "metric": r.metric, "formula": r.formula,
            "inputs": r.inputs, "computed_value": r.value,
        } for i, r in enumerate(bundle.results)]
        schema = {"items": [{"index": "integer index supplied",
                              "recomputed_value": "your independently recomputed number"}]}
        prompt = f"Rows: {rows}\nReturn this JSON shape: {schema}"
        response, error = await safe_complete_json(
            self._model_config, _CROSS_CHECK_SYSTEM_PROMPT, prompt,
            tracker=self._tracker, stage="analytics")
        if response is None:
            log_event(logger, logging.INFO,
                      "analytics LLM cross-check skipped", reason=error)
            return

        items = response.payload.get("items", []) if isinstance(response.payload, dict) else []
        by_index: dict[int, Any] = {}
        for row in items:
            if not isinstance(row, dict):
                continue
            try:
                by_index[int(row.get("index"))] = row.get("recomputed_value")
            except (TypeError, ValueError):
                continue

        disagreements = 0
        for i, result in enumerate(bundle.results):
            if i not in by_index:
                continue
            verified = verify_number(
                by_index[i], result.value, label=result.metric, rel_tolerance=0.01,
            )
            if verified.overridden and verified.llm_value is not None:
                disagreements += 1
                result.metadata["llm_disagreement"] = verified.flag
            elif verified.llm_value is not None:
                result.metadata["llm_verified"] = True
        if disagreements:
            log_event(logger, logging.WARNING,
                      "analytics LLM cross-check found disagreements",
                      disagreements=disagreements)

    def compute_sync(self, plan: ResearchPlan) -> AnalyticsBundle:
        self._results = []
        self._skips = []

        latest = self.reader.latest_reported_period()
        if latest is None:
            self._skips.append(
                "No reported period found in the Evidence Store; "
                "period-based analytics were skipped."
            )
        prior_year = prior_quarter = None
        if latest:
            period = normalise_fiscal_period(latest)
            if period:
                prior_year = year_ago_period(period)
                prior_quarter = previous_quarter_period(period)

        wanted = set(plan.required_analytics)

        # Price-based analytics.
        self._price_returns(wanted)
        self._price_52w_range()

        # Growth and margin analytics for the latest reported period.
        if latest:
            self._growth(wanted, latest, prior_year, prior_quarter)
            self._margins(wanted, latest, prior_year)
            self._cash(wanted, latest, prior_year)
            self._surprise(wanted, latest)
            self._segments(wanted, latest, prior_year)
            self._trends(wanted, latest, prior_year, prior_quarter)

        # Valuation analytics.
        self._valuation(wanted, plan)

        # Expectations analytics.
        self._guidance_vs_consensus(wanted)
        self._estimate_revisions(wanted)

        bundle = AnalyticsBundle(results=tuple(self._results), errors=tuple(self._skips))
        log_event(
            logger, logging.INFO, "analytics complete",
            requested=len(wanted), computed=len(bundle.results), skipped=len(bundle.errors),
            latest_period=latest,
        )
        for skip in self._skips:
            log_event(logger, logging.INFO, "analytic skipped", detail=skip)
        return bundle

    # -- calculation groups ----------------------------------------------
    def _price_returns(self, wanted: set[str]) -> None:
        history = self.reader.price_history()
        current = self.reader.numeric("share_price")
        if current is None or not history:
            if {"price_return_1m", "price_return_3m", "price_return_ytd"} & wanted:
                self._skips.append(
                    "Price history or current price unavailable; price returns skipped.")
            return

        as_of = current.as_of or history[-1].as_of
        if as_of is None:
            self._skips.append("Current price has no as-of date; price returns skipped.")
            return

        windows: tuple[tuple[str, str, dt.date | None], ...] = (
            ("price_return_1m", "Share price return, 1 month",
             _shift_days(as_of, 30)),
            ("price_return_3m", "Share price return, 3 months",
             _shift_days(as_of, 91)),
            ("price_return_ytd", "Share price return, year to date",
             dt.date(as_of.year - 1, 12, 31)),
        )
        for metric, label, target_date in windows:
            if metric not in wanted or target_date is None:
                continue
            baseline = _closest_to(history, target_date)
            if baseline is None or baseline.as_of is None:
                self._skips.append(f"No price observation near {target_date} for {metric}.")
                continue
            drift_days = abs((baseline.as_of - target_date).days)
            self._record(
                metric=metric,
                label=label,
                fn=lambda: calc.price_return_pct(current.value, baseline.value),
                unit=UNIT_PERCENT,
                formula=calc.F_PCT_CHANGE,
                evidence=(current, baseline),
                inputs={"current": current.value or 0.0, "prior": baseline.value or 0.0},
                metadata={
                    "baseline_date": baseline.as_of.isoformat(),
                    "target_date": target_date.isoformat(),
                    "baseline_drift_days": drift_days,
                },
            )

    def _price_52w_range(self) -> None:
        """52-week high/low, computed from the daily OHLC series.

        MegadataAPI's daily feed carries a high and a low for every trading
        day (Bloomberg's PX_HIGH/PX_LOW), not a ready-made 52-week aggregate -
        so this is a calculated observation over the trailing window actually
        held in the Evidence Store, not a value asserted by the provider.
        Company Snapshot and the key-data panel (agents/company_snapshot.py,
        synthesis/key_data.py) already read "price_52w_high"/"price_52w_low"
        from evidence first and fall back to this analytic when the provider
        has not sent it as a direct field.
        """
        highs, lows = self.reader.daily_highs(), self.reader.daily_lows()
        if not highs and not lows:
            self._skips.append(
                "No daily high/low series available; 52-week range skipped.")
            return
        # Cite the specific day the extreme occurred, not all ~250 daily rows
        # behind it - the window size and date range go in metadata for
        # transparency, and the formula/inputs still record the calculation.
        if highs:
            peak = max(highs, key=lambda i: i.value if i.value is not None else float("-inf"))
            self._record(
                metric="price_52w_high", label="52-week high",
                fn=lambda: calc.rolling_max([i.value for i in highs]),
                unit=UNIT_PER_SHARE, formula=calc.F_ROLLING_MAX,
                evidence=(peak,), inputs={"window_days": len(highs)},
                metadata={"window_start": (highs[0].as_of.isoformat() if highs[0].as_of else None),
                          "window_end": (highs[-1].as_of.isoformat() if highs[-1].as_of else None)},
            )
        if lows:
            trough = min(lows, key=lambda i: i.value if i.value is not None else float("inf"))
            self._record(
                metric="price_52w_low", label="52-week low",
                fn=lambda: calc.rolling_min([i.value for i in lows]),
                unit=UNIT_PER_SHARE, formula=calc.F_ROLLING_MIN,
                evidence=(trough,), inputs={"window_days": len(lows)},
                metadata={"window_start": (lows[0].as_of.isoformat() if lows[0].as_of else None),
                          "window_end": (lows[-1].as_of.isoformat() if lows[-1].as_of else None)},
            )

    def _growth(
        self, wanted: set[str], latest: str, prior_year: str | None, prior_quarter: str | None
    ) -> None:
        pairs = (
            ("revenue_growth_yoy", "Revenue growth, year on year", "revenue", prior_year),
            ("revenue_growth_qoq", "Revenue growth, quarter on quarter", "revenue", prior_quarter),
            ("eps_growth_yoy", "Diluted EPS growth, year on year", "eps_diluted", prior_year),
            ("net_income_growth_yoy", "Net income growth, year on year", "net_income", prior_year),
        )
        for metric, label, source_metric, comparison in pairs:
            if metric not in wanted or comparison is None:
                continue
            self._pct_change(metric, label, source_metric, latest, comparison)

    def _margins(self, wanted: set[str], latest: str, prior_year: str | None) -> None:
        pairs = (
            ("gross_margin_change_yoy", "Gross margin change, year on year", "gross_margin"),
            ("operating_margin_change_yoy", "Operating margin change, year on year",
             "operating_margin"),
        )
        for metric, label, source_metric in pairs:
            if metric not in wanted or prior_year is None:
                continue
            current = self.reader.numeric(source_metric, latest)
            prior = self.reader.numeric(source_metric, prior_year)
            if current is None or prior is None:
                self._skips.append(
                    f"{source_metric} unavailable for {latest} or {prior_year}; {metric} skipped.")
                continue
            self._record(
                metric=metric, label=label,
                fn=lambda c=current, p=prior: calc.margin_change_pp(c.value, p.value),
                unit=UNIT_PERCENTAGE_POINTS,
                formula=calc.F_MARGIN_CHANGE,
                evidence=(current, prior),
                inputs={"current_pct": current.value or 0.0, "prior_pct": prior.value or 0.0},
                period=latest, comparison_period=prior_year,
            )

    def _cash(self, wanted: set[str], latest: str, prior_year: str | None) -> None:
        if "fcf_growth_yoy" in wanted and prior_year:
            self._pct_change("fcf_growth_yoy", "Free cash flow growth, year on year",
                             "free_cash_flow", latest, prior_year)
        if "cash_conversion" in wanted:
            fcf = self.reader.numeric("free_cash_flow", latest)
            net_income = self.reader.numeric("net_income", latest)
            if fcf is None or net_income is None:
                self._skips.append(
                    f"Free cash flow or net income unavailable for {latest}; "
                    "cash conversion skipped.")
            else:
                self._record(
                    metric="cash_conversion",
                    label="Free cash flow conversion of net income",
                    fn=lambda: calc.cash_conversion(fcf.value, net_income.value),
                    unit=UNIT_PERCENT,
                    formula="free_cash_flow / net_income * 100",
                    evidence=(fcf, net_income),
                    inputs={"free_cash_flow": fcf.value or 0.0,
                            "net_income": net_income.value or 0.0},
                    period=latest,
                )

    def _surprise(self, wanted: set[str], latest: str) -> None:
        pairs = (
            ("revenue_surprise_pct", "Revenue versus consensus", "revenue", "consensus_revenue"),
            ("eps_surprise_pct", "Diluted EPS versus consensus", "eps_diluted", "consensus_eps"),
        )
        for metric, label, actual_metric, consensus_metric in pairs:
            if metric not in wanted:
                continue
            actual = self.reader.numeric(actual_metric, latest)
            consensus = self.reader.numeric(consensus_metric, latest)
            if actual is None:
                self._skips.append(f"No reported {actual_metric} for {latest}; {metric} skipped.")
                continue
            if consensus is None:
                self._skips.append(
                    f"No consensus {consensus_metric} for {latest}; {metric} skipped. "
                    "Report this as a data gap rather than inferring a surprise.")
                continue
            self._record(
                metric=metric, label=label,
                fn=lambda a=actual, c=consensus: calc.surprise_pct(a.value, c.value),
                unit=UNIT_PERCENT,
                formula=calc.F_SURPRISE,
                evidence=(actual, consensus),
                inputs={"actual": actual.value or 0.0, "consensus": consensus.value or 0.0},
                period=latest,
                confidence=Confidence.MEDIUM,
            )

    def _segments(self, wanted: set[str], latest: str, prior_year: str | None) -> None:
        rows = self.reader.segment_rows(latest)
        if not rows:
            if {"segment_growth_yoy", "segment_contribution_pct"} & wanted:
                self._skips.append(f"No segment revenue for {latest}; segment analytics skipped.")
            return

        total = self.reader.numeric("revenue", latest)
        prior_rows = {
            str(r.metadata.get("segment_name")): r
            for r in (self.reader.segment_rows(prior_year) if prior_year else ())
        }

        for row in rows:
            name = str(row.metadata.get("segment_name", "Unnamed segment"))
            if "segment_contribution_pct" in wanted and total is not None:
                self._record(
                    metric="segment_contribution_pct",
                    label=f"{name} share of revenue",
                    fn=lambda r=row, t=total: calc.contribution_pct(r.value, t.value),
                    unit=UNIT_PERCENT,
                    formula=calc.F_CONTRIBUTION,
                    evidence=(row, total),
                    inputs={"part": row.value or 0.0, "total": total.value or 0.0},
                    period=latest,
                    metadata={"segment_name": name},
                )
            if "segment_growth_yoy" in wanted:
                prior = prior_rows.get(name)
                if prior is None:
                    self._skips.append(
                        f"No prior-year revenue for segment {name}; growth skipped.")
                    continue
                self._record(
                    metric="segment_growth_yoy",
                    label=f"{name} revenue growth, year on year",
                    fn=lambda r=row, p=prior: calc.pct_change(r.value, p.value),
                    unit=UNIT_PERCENT,
                    formula=calc.F_PCT_CHANGE,
                    evidence=(row, prior),
                    inputs={"current": row.value or 0.0, "prior": prior.value or 0.0},
                    period=latest, comparison_period=prior_year,
                    metadata={"segment_name": name},
                )

    def _trends(
        self, wanted: set[str], latest: str, prior_year: str | None, prior_quarter: str | None
    ) -> None:
        if "trend_direction" not in wanted:
            return
        # Gross margin over the three periods we hold, oldest first.
        labels = [p for p in (prior_year, prior_quarter, latest) if p]
        items = [self.reader.numeric("gross_margin", label) for label in labels]
        present = [(label, item) for label, item in zip(labels, items, strict=True) if item]
        if len(present) < 2:
            self._skips.append("Not enough gross margin observations to detect a trend.")
            return
        ordered = sorted(present, key=lambda pair: pair[1].period.period_end or dt.date.min
                         if pair[1].period else dt.date.min)
        evidence = tuple(item for _, item in ordered)
        values = [item.value for _, item in ordered]
        try:
            trend = calc.detect_trend(values, stable_band=0.3, strong_band=1.5)
        except AnalyticsError as exc:
            self._skips.append(f"Gross margin trend skipped: {exc}")
            return
        self._record(
            metric="trend_direction",
            label="Gross margin trend",
            fn=lambda: trend.slope,
            unit=UNIT_PERCENTAGE_POINTS,
            formula="mean(step changes in gross margin)",
            evidence=evidence,
            inputs={f"p{i}": v or 0.0 for i, v in enumerate(values)},
            period=latest,
            metadata={"direction": trend.direction, "series": "gross_margin",
                      "points": trend.points},
        )

    def _valuation(self, wanted: set[str], plan: ResearchPlan) -> None:
        # Current multiples are observations, but the plan asks for them as
        # analytics so downstream stages can treat them uniformly.
        for metric, label in (("forward_pe", "Forward P/E"), ("ev_to_sales", "EV/Sales")):
            if metric not in wanted:
                continue
            item = self.reader.numeric(metric)
            if item is None:
                self._skips.append(f"No {metric} observation available.")
                continue
            self._record(
                metric=metric, label=label,
                fn=lambda i=item: float(i.value or 0.0),
                unit=UNIT_MULTIPLE,
                formula="reported multiple (pass-through)",
                evidence=(item,),
                inputs={"value": item.value or 0.0},
                confidence=item.confidence,
            )

        if "valuation_vs_peers" in wanted:
            self._valuation_vs_peers(plan)

        if "valuation_vs_history" in wanted:
            self._valuation_vs_history()

        if "peer_growth_gap" in wanted:
            self._peer_growth_gap(plan)

    def _valuation_vs_peers(self, plan: ResearchPlan) -> None:
        own = self.reader.numeric("forward_pe")
        if own is None:
            self._skips.append("No forward P/E for the subject company; peer comparison skipped.")
            return
        peer_items = [
            item for peer in plan.peers
            if (item := self.reader.peer_value(peer, "forward_pe")) is not None
        ]
        if not peer_items:
            self._skips.append(
                "No peer forward P/E observations available; peer valuation comparison skipped.")
            return
        peer_mean = calc.mean([item.value for item in peer_items])
        self._record(
            metric="valuation_vs_peers",
            label="Forward P/E premium versus the peer average",
            fn=lambda: calc.premium_pct(own.value, peer_mean),
            unit=UNIT_PERCENT,
            formula=calc.F_PREMIUM,
            evidence=(own, *peer_items),
            inputs={"value": own.value or 0.0, "benchmark": peer_mean},
            confidence=Confidence.MEDIUM,
            metadata={
                "peer_average_forward_pe": round(peer_mean, 2),
                "peers": [item.ticker for item in peer_items],
            },
        )

    def _valuation_vs_history(self) -> None:
        current = self.reader.numeric("forward_pe")
        history = self.reader.series("forward_pe", "historical")
        if current is None or len(history) < 2:
            self._skips.append(
                "Insufficient forward P/E history; valuation-versus-history skipped.")
            return
        historical_mean = calc.mean([item.value for item in history])
        self._record(
            metric="valuation_vs_history",
            label="Forward P/E versus its own observed history",
            fn=lambda: calc.premium_pct(current.value, historical_mean),
            unit=UNIT_PERCENT,
            formula=calc.F_PREMIUM,
            evidence=(current, *history),
            inputs={"value": current.value or 0.0, "benchmark": historical_mean},
            confidence=Confidence.MEDIUM,
            metadata={
                "historical_mean_forward_pe": round(historical_mean, 2),
                "observations": len(history),
                "window_from": history[0].as_of.isoformat() if history[0].as_of else None,
                "window_to": history[-1].as_of.isoformat() if history[-1].as_of else None,
            },
        )

    def _peer_growth_gap(self, plan: ResearchPlan) -> None:
        own_growth = next(iter(
            [r for r in self._results if r.metric == "revenue_growth_yoy"]), None)
        if own_growth is None:
            self._skips.append("Own revenue growth unavailable; peer growth gap skipped.")
            return
        peer_items = [
            item for peer in plan.peers
            if (item := self.reader.peer_value(peer, "revenue_growth_yoy_reported")) is not None
        ]
        if not peer_items:
            self._skips.append("No peer growth observations available; peer growth gap skipped.")
            return
        peer_mean = calc.mean([item.value for item in peer_items])
        self._record(
            metric="peer_growth_gap",
            label="Revenue growth gap versus the peer average",
            fn=lambda: own_growth.value - peer_mean,
            unit=UNIT_PERCENTAGE_POINTS,
            formula="own_growth_pct - peer_average_growth_pct",
            evidence=peer_items,
            analytics_inputs=(own_growth.analytics_id,),
            inputs={"own_growth_pct": own_growth.value, "peer_average_growth_pct": peer_mean},
            confidence=Confidence.MEDIUM,
            metadata={"peer_average_growth_pct": round(peer_mean, 1),
                      "peers": [item.ticker for item in peer_items]},
        )

    def _guidance_vs_consensus(self, wanted: set[str]) -> None:
        if "guidance_vs_consensus_pct" not in wanted:
            return
        guidance_items = [
            item for item in self.reader.guidance() if item.metric == "guidance_revenue"
        ]
        if not guidance_items:
            self._skips.append("No revenue guidance available; guidance-versus-consensus skipped.")
            return
        guidance = guidance_items[0]
        period = guidance.period_label
        consensus = self.reader.numeric("consensus_revenue", period)
        if consensus is None:
            self._skips.append(
                f"No consensus revenue for the guided period {period}; "
                "guidance-versus-consensus skipped.")
            return
        self._record(
            metric="guidance_vs_consensus_pct",
            label=f"Revenue guidance versus consensus for {period}",
            fn=lambda: calc.premium_pct(guidance.value, consensus.value),
            unit=UNIT_PERCENT,
            formula=calc.F_PREMIUM,
            evidence=(guidance, consensus),
            inputs={"value": guidance.value or 0.0, "benchmark": consensus.value or 0.0},
            period=period,
            confidence=Confidence.MEDIUM,
        )

    def _estimate_revisions(self, wanted: set[str]) -> None:
        if "estimate_revision_direction" not in wanted:
            return
        up = self.reader.numeric("estimate_revisions_up")
        down = self.reader.numeric("estimate_revisions_down")
        if up is None or down is None:
            self._skips.append(
                "Estimate revision counts unavailable; revision direction skipped.")
            return
        self._record(
            metric="estimate_revision_direction",
            label="Net direction of recent estimate revisions",
            fn=lambda: calc.net_revision_pct(up.value, down.value),
            unit=UNIT_PERCENT,
            formula=calc.F_REVISION_NET,
            evidence=(up, down),
            inputs={"up": up.value or 0.0, "down": down.value or 0.0},
            confidence=Confidence.MEDIUM,
            metadata={"revisions_up": up.value, "revisions_down": down.value,
                      "unit_note": "positive means net upward revisions"},
        )

    # -- plumbing --------------------------------------------------------
    def _pct_change(
        self, metric: str, label: str, source_metric: str, latest: str, comparison: str
    ) -> None:
        current = self.reader.numeric(source_metric, latest)
        prior = self.reader.numeric(source_metric, comparison)
        if current is None or prior is None:
            self._skips.append(
                f"{source_metric} unavailable for {latest} or {comparison}; {metric} skipped.")
            return
        self._record(
            metric=metric, label=label,
            fn=lambda: calc.pct_change(current.value, prior.value),
            unit=UNIT_PERCENT,
            formula=calc.F_PCT_CHANGE,
            evidence=(current, prior),
            inputs={"current": current.value or 0.0, "prior": prior.value or 0.0},
            period=latest, comparison_period=comparison,
        )

    def _record(
        self,
        *,
        metric: str,
        label: str,
        fn: Callable[[], float],
        unit: str,
        formula: str,
        evidence: tuple[EvidenceItem, ...] | list[EvidenceItem],
        inputs: dict[str, float],
        period: str | None = None,
        comparison_period: str | None = None,
        confidence: Confidence = Confidence.HIGH,
        metadata: dict[str, object] | None = None,
        analytics_inputs: tuple[str, ...] = (),
    ) -> None:
        """Run one calculation and record it, or record why it was skipped."""
        try:
            value = fn()
        except AnalyticsError as exc:
            self._skips.append(f"{metric}: {exc}")
            return

        evidence_ids = tuple(item.evidence_id for item in evidence)
        meta = dict(metadata or {})
        if analytics_inputs:
            meta["input_analytics_ids"] = list(analytics_inputs)
        self._results.append(AnalyticsResult(
            analytics_id=make_analytics_id(
                self.report_run_id, metric, period or "", comparison_period or "",
                meta.get("segment_name", ""), *evidence_ids,
            ),
            report_run_id=self.report_run_id,
            metric=metric,
            value=round(value, 6),
            unit=unit if unit != UNIT_COUNT else UNIT_COUNT,
            formula=formula,
            input_evidence_ids=evidence_ids,
            inputs={k: round(float(v), 6) for k, v in inputs.items()},
            label=label,
            period=period,
            comparison_period=comparison_period,
            confidence=confidence,
            metadata=meta,
        ))


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _shift_days(anchor: dt.date, days: int) -> dt.date:
    return anchor - dt.timedelta(days=days)


def _closest_to(history: tuple[EvidenceItem, ...], target: dt.date) -> EvidenceItem | None:
    """Price observation nearest to ``target``, in either direction.

    Price histories are often sparse (monthly closes here), so nearest-match
    beats on-or-before: the latter silently reaches back a whole extra period.
    The distance actually used is recorded on the analytics result.
    """
    candidates = [item for item in history if item.as_of is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda item: abs((item.as_of - target).days))  # type: ignore[operator]
