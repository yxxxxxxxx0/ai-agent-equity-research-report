"""Stage 11 - the pipeline orchestrator.

One entry point, one pass through the architecture:

    request -> plan
            -> market data | fundamentals | documents   (concurrent)
            -> normalisation
            -> evidence store
            -> analytics | segment agents               (agents concurrent)
            -> synthesis
            -> QA
            -> PDF

The orchestrator owns the wiring and the failure policy; it contains no
analytical logic of its own. Every stage's output is written to the run
directory so any point in the chain can be inspected afterwards.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ..acquisition.services import AcquisitionBundle, acquire_all, build_services
from ..agents.runner import run_segment_agents
from ..analytics.engine import AnalyticsEngine
from ..config import ModelConfig, Settings
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import EvidenceCategory, ProviderStatus, RunStatus
from ..domain.plan import ResearchPlan
from ..domain.qa import QAResult
from ..domain.report import ReportDraft
from ..domain.request import ResearchRequest
from ..domain.run import ReportRun
from ..domain.segment import SegmentResult
from ..errors import PipelineError
from ..evidence.reader import EvidenceReader
from ..evidence.store import EvidenceQuery, EvidenceStore
from ..normalisation.reconciliation import revalidate_qualitative
from ..llm.usage import UsageTracker
from ..logging_setup import configure_logging, get_logger, log_event
from ..normalisation.normalizer import NormalisationResult, Normalizer
from ..planning.research_planner import ResearchPlanner, _clean_company_name
from ..qa.engine import QAEngine
from ..qa.repair import DraftRepairer, tidy_draft
from ..rendering.compact_renderer import render_compact_report
from ..rendering.json_writer import write_json, write_report_json, write_run_manifest
from ..rendering.pdf_renderer import PdfReportRenderer
from ..rendering.technical_appendix import (
    build_technical_appendix_pdf,
    merge_technical_appendix,
)
from ..synthesis.llm_synthesizer import LLMSynthesizer
from .freshness_check import FreshnessResult, check_freshness
from .run_tracker import RUN_ID_PATTERN, RunTracker, new_run_id
from .web_gap_fill import WebGapFillResult, fill_evidence_gaps

logger = get_logger("pipeline")


@dataclass(frozen=True, slots=True)
class ReportResult:
    """What one pipeline invocation produced."""

    report_run_id: str
    status: RunStatus
    run: ReportRun
    draft: ReportDraft | None = None
    qa_result: QAResult | None = None
    pdf_path: Path | None = None
    compact_pdf_path: Path | None = None
    report_json_path: Path | None = None
    run_manifest_path: Path | None = None

    @property
    def succeeded(self) -> bool:
        return self.status in {RunStatus.SUCCEEDED, RunStatus.SUCCEEDED_WITH_WARNINGS}

    def summary(self) -> str:
        parts = [f"run {self.report_run_id}", self.status.value]
        if self.pdf_path:
            parts.append(f"pdf={self.pdf_path}")
        if self.compact_pdf_path:
            parts.append(f"compact_pdf={self.compact_pdf_path}")
        if self.qa_result:
            parts.append(
                f"qa: {len(self.qa_result.critical)} critical, "
                f"{len(self.qa_result.warnings)} warnings")
        return " | ".join(parts)


async def generate_report(
    request: ResearchRequest | None = None,
    settings: Settings | None = None,
    *,
    store: EvidenceStore | None = None,
    resume_from: str | None = None,
) -> ReportResult:
    """Run the full pipeline for one research request.

    ``store`` may be supplied by tests to use an in-memory Evidence Store; the
    default opens the configured SQLite file.

    ``resume_from``: an existing ``report_run_id`` produced by an earlier,
    successful call to this function. When set, ``request`` is ignored -
    planning, acquisition, normalisation, the freshness check and web
    gap-fill are all skipped, and the pipeline resumes directly against that
    run's plan (reloaded from its saved ``01_plan.json``) and its
    already-persisted Evidence Store rows. This is for iterating on
    synthesis/QA/rendering code without re-paying for the API/LLM calls the
    earlier stages already made; it writes its outputs into that same run's
    directory, so ``04_analytics.json`` onward and the run manifest are
    replaced by the new attempt while the plan/acquisition/normalisation
    snapshots are untouched.
    """
    settings = settings or Settings.from_env()
    settings.ensure_dirs()
    configure_logging(settings.log_level, as_json=settings.log_json)

    if not settings.model.enabled:
        raise RuntimeError(
            "LLM writing is required. Configure EQR_MODEL_API_KEY before generating a report.")

    carried_metadata: dict[str, Any] = {}
    if resume_from:
        # The id becomes a path component - accept only the shape new_run_id() makes.
        if not RUN_ID_PATTERN.fullmatch(resume_from):
            raise ValueError(f"not a valid report_run_id: {resume_from!r}")
        report_run_id = resume_from
        run_dir = settings.run_dir(report_run_id)
        plan_path = run_dir / "01_plan.json"
        if not plan_path.exists():
            raise RuntimeError(
                f"cannot resume run {resume_from!r}: no saved plan at {plan_path}")
        plan = ResearchPlan.from_dict(json.loads(plan_path.read_text(encoding="utf-8")))
        # Plans saved before the planner cleaned its company name still carry
        # the model's "(AAPL; identity to be verified)" note.
        plan = replace(plan, company=_clean_company_name(plan.company))
        request = plan.request
        # Disclosures produced by the skipped stages (page-1 freshness notice,
        # web-research note) must survive the resume.
        previous_draft = run_dir / "06_report_draft_initial.json"
        if previous_draft.exists():
            previous = json.loads(previous_draft.read_text(encoding="utf-8"))
            carried_metadata = {
                key: value for key, value in (previous.get("metadata") or {}).items()
                if key in {"freshness_check", "web_gap_fill"}
            }
        (run_dir / "validation_failure.json").unlink(missing_ok=True)
    else:
        if request is None:
            raise RuntimeError("request is required unless resume_from is set")
        report_run_id = new_run_id()
        run_dir = settings.run_dir(report_run_id)

    owns_store = store is None
    store = store or EvidenceStore(settings.database_path)

    tracker = RunTracker(report_run_id, request.to_dict())
    usage_tracker = UsageTracker()

    log_event(logger, logging.INFO, "report run started",
              company=request.company, ticker=request.ticker,
              sections=[s.value for s in request.sections],
              resume_from=resume_from,
              settings=settings.describe())

    try:
        if resume_from:
            # Skip stages 1-4.6 entirely: the plan came from disk above, and
            # the Evidence Store already holds this run_id's rows from the
            # original run - see the docstring.
            with tracker.stage("resume_from_evidence_store"):
                reader = EvidenceReader(
                    store=store, report_run_id=report_run_id,
                    ticker=plan.ticker, company=Normalizer(report_run_id, plan).company,
                )
                evidence_count = store.count(report_run_id)
                if evidence_count == 0:
                    raise RuntimeError(
                        f"cannot resume run {resume_from!r}: the Evidence Store has no "
                        "rows for it (it may have been cleared since that run)")
                # Rows keep whatever status the code of their day assigned;
                # re-apply today's qualitative-validation rule so a resume
                # sees the same evidence a fresh run would.
                revalidated = revalidate_qualitative(store.query(EvidenceQuery(
                    report_run_id=report_run_id, category=EvidenceCategory.DOCUMENT)))
                if revalidated:
                    store.replace(revalidated)
                tracker.set_plan(plan.to_dict())
                tracker.warn(
                    f"Resumed from run {resume_from}: reused its plan and "
                    f"{evidence_count} Evidence Store row(s); planning, acquisition, "
                    "normalisation, the freshness check and web gap-fill were skipped."
                )
            freshness = FreshnessResult(checked=False)
            web_gap_fill_result = WebGapFillResult(attempted=False)
        else:
            # 1. Planning ----------------------------------------------------
            with tracker.stage("planning"):
                # Planning is the only stage given its own reasoning-effort
                # setting (see Settings.planning_reasoning_effort): the one call
                # that shapes every downstream stage's scope. This is a copy of
                # settings.model with only ``reasoning_effort`` changed - every
                # other stage below still receives settings.model unmodified.
                planning_model = (
                    replace(settings.model, reasoning_effort=settings.planning_reasoning_effort)
                    if settings.planning_reasoning_effort else settings.model
                )
                plan = await ResearchPlanner(planning_model, tracker=usage_tracker).plan(request)
                tracker.set_plan(plan.to_dict())
                for note in plan.notes:
                    tracker.warn(note)
                write_json(run_dir / "01_plan.json", plan.to_dict())

            # 2. Parallel acquisition ---------------------------------------
            with tracker.stage("acquisition"):
                market_service, fundamentals_service, documents_service = build_services(
                    settings, usage_tracker)
                acquisition = await acquire_all(
                    plan, market_service, fundamentals_service, documents_service)
                tracker.set_source_status(acquisition.to_dict())
                tracker.errors(acquisition.errors)
                _warn_on_dead_branches(tracker, acquisition)
                write_json(run_dir / "02_acquisition.json", {"branches": acquisition.to_dict()})

            # 3. Normalisation -----------------------------------------------
            with tracker.stage("normalisation"):
                normalised = await Normalizer(
                    report_run_id, plan, settings.model, tracker=usage_tracker
                ).normalize(
                    acquisition.market_data.observations,
                    acquisition.fundamentals.observations,
                    acquisition.documents.passages,
                    document_observations=acquisition.documents.observations,
                )
                _record_rejections(tracker, normalised)
                write_json(run_dir / "03_normalisation.json", {
                    **normalised.to_dict(),
                    "evidence": [item.to_dict() for item in normalised.evidence],
                })

            # 4. Evidence store ----------------------------------------------
            with tracker.stage("evidence_ingestion"):
                written = store.save(normalised.evidence)
                reader = EvidenceReader(
                    store=store, report_run_id=report_run_id,
                    ticker=plan.ticker, company=Normalizer(report_run_id, plan).company,
                )
                if written == 0:
                    tracker.warn("No evidence was written to the Evidence Store.")

            # 4.5 Data freshness check (best-effort, opt-in) ------------------
            # Runs here, before analysis or synthesis touch the evidence at all -
            # "fix data freshness before anything else" - rather than as part of
            # the later gap-research addendum, so a stale dataset is disclosed
            # up front instead of discovered on the last page.
            freshness = FreshnessResult(checked=False)
            if settings.check_data_freshness and settings.model.enabled:
                with tracker.stage("freshness_check"):
                    try:
                        freshness = await check_freshness(
                            reader.company, plan.ticker, plan.request.report_date,
                            reader.latest_reported_period(), settings.model,
                            tracker=usage_tracker,
                        )
                        if freshness.mismatched:
                            tracker.warn(
                                "Live verification found a more recent public report "
                                f"({freshness.verified_period}) than this dataset is anchored "
                                f"on ({reader.latest_reported_period()}); see the notice on "
                                "page 1."
                            )
                    except Exception as exc:  # noqa: BLE001 - never sink the run over this
                        log_event(logger, logging.WARNING, "freshness check failed",
                                  error=f"{type(exc).__name__}: {exc}")
                        tracker.warn(
                            f"Data freshness check failed: {type(exc).__name__}: {exc}")

            # 4.6 Web gap-fill (best-effort, opt-in) ---------------------------
            # Also runs before analysis/synthesis, so a section MegadataAPI left
            # thin gets a chance to become evidence-backed rather than omitted -
            # see pipeline.web_gap_fill for the two-gate verification every
            # candidate fact must pass before it is written to the store.
            web_gap_fill_result = WebGapFillResult(attempted=False)
            if settings.web_fill_gaps and settings.model.enabled:
                with tracker.stage("web_gap_fill"):
                    try:
                        web_gap_fill_result, new_items = await fill_evidence_gaps(
                            report_run_id, reader.company, plan.ticker, reader, settings.model,
                            allowed_domains=settings.web_fill_allowed_domains,
                            max_claims=settings.web_fill_max_claims,
                            tracker=usage_tracker,
                        )
                        if new_items:
                            store.save(new_items)
                            tracker.warn(
                                f"Web research added {len(new_items)} source-verified fact(s) "
                                f"for: {', '.join(web_gap_fill_result.topics_checked)}."
                            )
                    except Exception as exc:  # noqa: BLE001 - never sink the run over this
                        log_event(logger, logging.WARNING, "web gap-fill failed",
                                  error=f"{type(exc).__name__}: {exc}")
                        tracker.warn(f"Web gap-fill failed: {type(exc).__name__}: {exc}")
                write_json(run_dir / "04_6_web_gap_fill.json", web_gap_fill_result.to_dict())

        # 5. Analytics and segment agents, technical appendix ------------
        # The technical appendix is a live Bloomberg OHLCV fetch plus a
        # standalone chart render - independent of the report draft, so it
        # is kicked off here to run concurrently with analysis rather than
        # waiting until after QA passes, when it used to start.
        technical_appendix_task: asyncio.Task | None = None
        existing_appendix_raw = (
            run_dir / f"{report_run_id}_technical_appendix_raw.pdf" if resume_from else None
        )
        if resume_from and existing_appendix_raw is not None and existing_appendix_raw.exists():
            # Reuse the prior run's raw appendix fetch (a live Bloomberg OHLCV
            # call) rather than repeating it purely to iterate on downstream
            # code; a fresh full run still regenerates it as usual.
            async def _reuse_technical_appendix() -> Path | None:
                return existing_appendix_raw
            technical_appendix_task = asyncio.create_task(_reuse_technical_appendix())
        elif settings.technical_appendix:
            async def _build_technical_appendix() -> Path | None:
                with tracker.stage("technical_appendix"):
                    try:
                        raw_path = run_dir / f"{report_run_id}_technical_appendix_raw.pdf"
                        return await asyncio.to_thread(
                            build_technical_appendix_pdf, plan.ticker, plan.request.report_date,
                            raw_path, credentials=settings.credentials,
                            timeout=settings.provider_timeout_seconds,
                            min_request_interval_seconds=settings.megadata_min_request_interval_seconds,
                        )
                    except Exception as exc:  # noqa: BLE001 - non-core supplement
                        log_event(logger, logging.WARNING, "technical appendix failed",
                                  error=f"{type(exc).__name__}: {exc}")
                        tracker.warn(
                            f"Technical appendix could not be generated: "
                            f"{type(exc).__name__}: {exc}")
                        return None
            technical_appendix_task = asyncio.create_task(_build_technical_appendix())

        with tracker.stage("analysis"):
            analytics, segment_results = await _run_analysis(
                report_run_id, plan, reader, store, settings.model, usage_tracker)
            tracker.set_counts(evidence=store.count(report_run_id),
                               analytics=len(analytics.results))
            tracker.set_agent_status([
                {
                    "segment": result.segment.value,
                    "status": "failed" if result.errors else "ok",
                    "findings": len(result.key_findings),
                    "metrics": len(result.important_metrics),
                    "data_gaps": len(result.data_gaps),
                    "errors": list(result.errors),
                }
                for result in segment_results
            ])
            for skip in analytics.errors:
                tracker.warn(f"Analytic skipped: {skip}")
            write_json(run_dir / "04_analytics.json", analytics.to_dict())
            write_json(run_dir / "05_segments.json",
                       {"segments": [r.to_dict() for r in segment_results]})

        # 6. Synthesis ---------------------------------------------------
        with tracker.stage("synthesis"):
            draft = await LLMSynthesizer(
                report_run_id, plan, reader, analytics, settings.model,
                tracker=usage_tracker,
            ).synthesize_async(segment_results)
            if freshness.checked:
                draft = replace(draft, metadata={
                    **draft.metadata, "freshness_check": freshness.to_dict()})
            if web_gap_fill_result.attempted:
                draft = replace(draft, metadata={
                    **draft.metadata, "web_gap_fill": web_gap_fill_result.to_dict()})
            if carried_metadata:
                draft = replace(draft, metadata={**draft.metadata, **carried_metadata})

        write_json(run_dir / "06_report_draft_initial.json", draft.to_dict())

        # 7. QA ----------------------------------------------------------
        with tracker.stage("qa"):
            qa_engine = QAEngine(
                model_config=settings.model, tracker=usage_tracker,
                verify_conflicts=settings.verify_metric_conflicts,
                verify_web_claims=settings.web_fill_gaps,
                triage_mode=settings.qa_triage,
            )
            qa_result = await qa_engine.validate(draft, plan, reader, analytics)

            repair_log: list[dict[str, Any]] = []

            async def _converge() -> None:
                """Rephrase (LLM, subtraction only), then drop, until QA has no critical."""
                nonlocal draft, qa_result
                repairer = DraftRepairer(settings.model, tracker=usage_tracker)
                for attempt in range(1, settings.qa_auto_repair_max_attempts + 1):
                    if not qa_result.has_critical_errors:
                        return
                    outcome = await repairer.repair(draft, qa_result, reader, analytics)
                    repair_log.append({
                        "attempt": attempt,
                        "critical_before": len(qa_result.critical),
                        "changed": outcome.changed,
                        "llm_error": outcome.llm_error,
                        "events": [event.to_dict() for event in outcome.events],
                    })
                    if not outcome.changed:
                        break
                    draft = outcome.draft
                    qa_result = await qa_engine.validate(draft, plan, reader, analytics)
                    repair_log[-1]["critical_after"] = len(qa_result.critical)

                # Last resort: a statement still failing is omitted deterministically
                # (no model), so the loop converges instead of blocking on one
                # sentence. Repeated because the entailment reviewer is a model: a
                # fresh QA pass can flag a sentence an earlier pass accepted.
                for _round in range(3):
                    if not qa_result.has_critical_errors:
                        return
                    outcome = await DraftRepairer(None).repair(draft, qa_result, reader, analytics)
                    if not outcome.changed:
                        return
                    draft = outcome.draft
                    qa_result = await qa_engine.validate(draft, plan, reader, analytics)
                    repair_log.append({
                        "attempt": "deterministic_omit",
                        "events": [event.to_dict() for event in outcome.events],
                        "critical_after": len(qa_result.critical),
                    })

            if settings.qa_auto_repair:
                if qa_result.has_critical_errors:
                    await _converge()
                tidied = tidy_draft(draft, qa_result)
                if tidied != draft:
                    draft = tidied
                    qa_result = await qa_engine.validate(draft, plan, reader, analytics)
                    repair_log.append({"attempt": "tidy",
                                       "critical_after": len(qa_result.critical)})
                    # Tidy's re-check can surface findings the earlier pass accepted;
                    # give them the same rephrase-then-drop pass before the gate closes.
                    if qa_result.has_critical_errors:
                        await _converge()

            draft = replace(draft, metadata={
                **draft.metadata,
                "qa_auto_repair": {
                    "enabled": settings.qa_auto_repair,
                    "max_attempts": settings.qa_auto_repair_max_attempts,
                    "attempts": repair_log,
                    "final_critical_count": len(qa_result.critical),
                    "publication_decision": "deterministic_qa",
                },
            })
            write_json(run_dir / "06_report_draft.json", draft.to_dict())
            write_json(run_dir / "07_qa_repair.json", {
                "authority": {
                    "llm": "may propose prose rewrites only",
                    "deterministic": "validates evidence, arithmetic, and publication",
                },
                "attempts": repair_log,
            })
            tracker.set_qa(qa_result.to_dict())
            write_json(run_dir / "07_qa.json", qa_result.to_dict())

        report_json_path = write_report_json(run_dir, draft, qa_result)

        if qa_result.has_critical_errors:
            log_event(logger, logging.ERROR, "PDF suppressed by QA",
                      critical=len(qa_result.critical))
            if technical_appendix_task is not None:
                technical_appendix_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await technical_appendix_task
            tracker.set_outputs(json_path=str(report_json_path), pdf_path=None)
            write_json(run_dir / "validation_failure.json", {
                "report_run_id": report_run_id,
                "publication_blocked": True,
                "p0_count": len(qa_result.critical),
                "p1_count": len(qa_result.warnings),
                "p2_count": len(qa_result.infos),
                "required_fixes": [finding.to_dict() for finding in qa_result.critical],
            })
            run, manifest = _finish_with_usage(
                tracker, usage_tracker, RunStatus.FAILED_QA, run_dir)
            return ReportResult(
                report_run_id=report_run_id, status=RunStatus.FAILED_QA, run=run,
                draft=draft, qa_result=qa_result,
                report_json_path=report_json_path, run_manifest_path=manifest,
            )

        # 8. PDF and compact PDF, in parallel ------------------------------
        # Neither depends on the other's output - the full PDF needs only
        # draft/qa_result, the compact PDF only report_json_path (already
        # written above) - so they render concurrently instead of back to
        # back.
        async def _render_pdf() -> Path:
            with tracker.stage("pdf"):
                return await asyncio.to_thread(
                    PdfReportRenderer(run_dir).render, draft, qa_result)

        async def _render_compact() -> Path | None:
            if not settings.compact_report:
                return None
            with tracker.stage("compact_pdf"):
                try:
                    compact_target = run_dir / (
                        f"{plan.ticker}_{plan.request.report_date:%Y%m%d}_"
                        f"{report_run_id}_compact_two_page.pdf"
                    )
                    return await asyncio.to_thread(
                        render_compact_report, report_json_path, compact_target,
                        credentials=settings.credentials,
                        timeout=settings.provider_timeout_seconds,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the long report final
                    log_event(logger, logging.WARNING, "compact PDF failed",
                              error=f"{type(exc).__name__}: {exc}")
                    tracker.warn(
                        f"Compact two-page report could not be generated: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    return None

        pdf_path, compact_pdf_path = await asyncio.gather(_render_pdf(), _render_compact())

        # The technical appendix was already fetched and rendered concurrently
        # with analysis (see above); merging it onto the now-finished PDF is
        # just a page concatenation, so a market-data outage there never
        # suppresses the evidence-backed report - it leaves a warning and
        # preserves the base PDF instead.
        if technical_appendix_task is not None:
            appendix_pdf_path = await technical_appendix_task
            if appendix_pdf_path is not None:
                try:
                    technical_pdf_path = run_dir / (
                        f"{plan.ticker}_{plan.request.report_date:%Y%m%d}_"
                        f"{report_run_id}_with_technical_appendix.pdf"
                    )
                    pdf_path = await asyncio.to_thread(
                        merge_technical_appendix, pdf_path, appendix_pdf_path,
                        technical_pdf_path)
                except Exception as exc:  # noqa: BLE001 - non-core supplement
                    log_event(logger, logging.WARNING, "technical appendix merge failed",
                              error=f"{type(exc).__name__}: {exc}")
                    tracker.warn(
                        f"Technical appendix could not be merged: "
                        f"{type(exc).__name__}: {exc}")
                finally:
                    # A reused prior-run appendix is a resume input, not this
                    # attempt's scratch output - keep it so later resumes can
                    # reuse it too; only a freshly-fetched one is disposable.
                    if appendix_pdf_path != existing_appendix_raw:
                        appendix_pdf_path.unlink(missing_ok=True)

        tracker.set_outputs(json_path=str(report_json_path), pdf_path=str(pdf_path))
        status = (
            RunStatus.SUCCEEDED_WITH_WARNINGS
            if (qa_result.warnings or tracker.run.errors or tracker.run.warnings)
            else RunStatus.SUCCEEDED
        )
        run, manifest = _finish_with_usage(tracker, usage_tracker, status, run_dir)

        log_event(logger, logging.INFO, "report run complete",
                  status=status.value, pdf=str(pdf_path),
                  compact_pdf=str(compact_pdf_path) if compact_pdf_path else None,
                  duration_ms=round(run.duration_ms or 0, 1))
        return ReportResult(
            report_run_id=report_run_id, status=status, run=run, draft=draft,
            qa_result=qa_result, pdf_path=pdf_path,
            compact_pdf_path=compact_pdf_path,
            report_json_path=report_json_path, run_manifest_path=manifest,
        )

    except Exception as exc:  # noqa: BLE001 - the run record must always be written
        log_event(logger, logging.ERROR, "report run failed",
                  error=f"{type(exc).__name__}: {exc}")
        run, manifest = _finish_with_usage(tracker, usage_tracker, RunStatus.FAILED, run_dir)
        return ReportResult(
            report_run_id=report_run_id, status=RunStatus.FAILED, run=run,
            run_manifest_path=manifest,
        )
    finally:
        if owns_store:
            store.close()


async def _run_analysis(
    report_run_id: str,
    plan: ResearchPlan,
    reader: EvidenceReader,
    store: EvidenceStore,
    model_config: ModelConfig | None = None,
    tracker: UsageTracker | None = None,
) -> tuple[AnalyticsBundle, tuple[SegmentResult, ...]]:
    """Analytics then agents.

    The two are drawn as parallel branches in the architecture because neither
    fetches data; in practice the agents *consume* analytics, so analytics runs
    first and the agents then run concurrently with each other. Both read only
    from the Evidence Store.
    """
    analytics = await AnalyticsEngine(
        report_run_id, reader, model_config, tracker=tracker).compute(plan)
    store.save_analytics(analytics.results)
    segment_results = await run_segment_agents(plan, reader, analytics, model_config, tracker)
    return analytics, segment_results


def _finish_with_usage(
    tracker: RunTracker, usage_tracker: UsageTracker, status: RunStatus, run_dir: Path,
) -> tuple[ReportRun, Path]:
    """Attach the LLM usage summary to the run record, log it once, and persist the manifest."""
    usage_summary = usage_tracker.summary()
    tracker.set_llm_usage(usage_summary)
    run = tracker.finish(status)
    manifest = write_run_manifest(run_dir, run)
    log_event(
        logger, logging.INFO, "LLM usage summary",
        calls=usage_summary.get("call_count", 0),
        total_input_tokens=usage_summary.get("total_input_tokens", 0),
        total_output_tokens=usage_summary.get("total_output_tokens", 0),
        total_cost_usd=round(usage_summary.get("total_cost_usd", 0.0), 4),
        calls_missing_cost=usage_summary.get("calls_missing_cost", 0),
    )
    # One line per stage, cheapest last, so the terminal shows where the run's
    # cost actually went without needing to open the run manifest.
    by_stage = usage_summary.get("by_stage") or {}
    for stage, row in sorted(by_stage.items(), key=lambda kv: -kv[1].get("cost_usd", 0.0)):
        log_event(
            logger, logging.INFO, f"LLM cost [{stage}]",
            calls=row.get("calls", 0),
            input_tokens=row.get("input_tokens", 0),
            output_tokens=row.get("output_tokens", 0),
            cost_usd=round(row.get("cost_usd", 0.0), 4),
        )
    return run, manifest


def _warn_on_dead_branches(tracker: RunTracker, acquisition: AcquisitionBundle) -> None:
    """A failed or empty branch is a documented gap, not a crash."""
    for branch in acquisition.branches:
        if branch.status is ProviderStatus.FAILED:
            tracker.warn(
                f"The {branch.branch} branch returned no data; the report proceeds "
                "without it and the affected sections record data gaps.")
        elif branch.status is ProviderStatus.SKIPPED:
            tracker.warn(f"The {branch.branch} branch was skipped.")
        elif branch.item_count == 0:
            tracker.warn(f"The {branch.branch} branch returned zero items.")


def _record_rejections(tracker: RunTracker, normalised: NormalisationResult) -> None:
    for rejection in normalised.rejections:
        tracker.error(PipelineError(
            stage="normalisation",
            kind="NormalisationError",
            message=rejection.reason,
            context={"branch": rejection.branch, "raw_metric": rejection.raw_metric,
                     "source": rejection.source_name},
        ))
    for warning in normalised.warnings:
        tracker.warn(warning)


def generate_report_sync(
    request: ResearchRequest | dict[str, Any] | None = None,
    settings: Settings | None = None,
    *,
    resume_from: str | None = None,
) -> ReportResult:
    """Blocking wrapper for scripts and tests."""
    import asyncio

    parsed = (
        None if request is None
        else request if isinstance(request, ResearchRequest)
        else ResearchRequest.from_dict(request)
    )
    return asyncio.run(generate_report(parsed, settings, resume_from=resume_from))
