"""A small local web interface for the eq_report pipeline.

Pick a ticker and a report date, run the real pipeline against it, watch it
move through each pipeline stage live, and view the resulting short
(compact) report layout in the browser.

Usage:
    python webui/app.py
    then open http://127.0.0.1:5050

Each run happens in a background thread so the page can poll progress
instead of holding the HTTP request open for however long the LLM stages
take. Live stage tracking is done by monkeypatching RunTracker.stage() from
here rather than editing the pipeline itself, so this file is the only place
that knows a web UI exists at all.

The visual language is deliberately the report's own, not a generic
dashboard theme: the same orange/ink/rust/teal palette and hairline-and-zebra
table style as eq_report/rendering/pdf_renderer.py, so this page reads as
the desk behind the printed note rather than an unrelated admin panel.
"""

from __future__ import annotations

import contextvars
import datetime as dt
import json
import os
import re
import sys
import threading
import time
import traceback
import urllib.request
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

from flask import (
    Flask,
    jsonify,
    redirect,
    render_template_string,
    request,
    send_file,
    url_for,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    # Run as `python webui/app.py`, so sys.path[0] is webui/, not the repo
    # root - the eq_report package needs the root added explicitly.
    sys.path.insert(0, str(REPO_ROOT))


def _load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from .env into the process environment.

    Mirrors what run_report.ps1 does for the PowerShell entry point, since
    Settings.from_env() only ever reads os.environ.
    """
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv(REPO_ROOT / ".env")

from eq_report.config import Settings  # noqa: E402 (must follow dotenv load)
from eq_report.domain.request import ResearchRequest  # noqa: E402
from eq_report.logging_setup import configure_logging  # noqa: E402
from eq_report.llm import usage as usage_module  # noqa: E402
from eq_report.pipeline import run_tracker as run_tracker_module  # noqa: E402
from eq_report.pipeline.orchestrator import generate_report_sync  # noqa: E402
from eq_report.pipeline.run_tracker import RUN_ID_PATTERN  # noqa: E402

TICKER_PATTERN = re.compile(r"[A-Z0-9.\-]{1,10}")
MODEL_PATTERN = re.compile(r"[a-z0-9][\w.\-]*/[\w.\-:]+")

# Models offered in the picker, cheapest-to-strongest within each lab. Prices
# and availability come live from OpenRouter; an id it no longer lists is hidden.
CURATED_MODELS = (
    "openai/gpt-5.6-luna", "openai/gpt-5.6-sol", "openai/gpt-5.6-terra", "openai/gpt-5.5",
    "anthropic/claude-haiku-4.5", "anthropic/claude-sonnet-5.5", "anthropic/claude-opus-5.5",
    "google/gemini-3.8-flash", "google/gemini-3.1-pro-preview", "x-ai/grok-4.7",
    "deepseek/deepseek-v4-pro", "qwen/qwen3.8-max-0902",
)
_MODEL_CACHE: dict = {"at": 0.0, "live": None}


def _live_models() -> dict[str, dict] | None:
    """OpenRouter's model list keyed by id, cached for an hour; None if unreachable."""
    if time.time() - _MODEL_CACHE["at"] < 3600 and _MODEL_CACHE["live"] is not None:
        return _MODEL_CACHE["live"]
    try:
        with urllib.request.urlopen("https://openrouter.ai/api/v1/models", timeout=8) as response:
            rows = json.load(response)["data"]
    except Exception:  # noqa: BLE001 - the picker degrades to the curated list
        return _MODEL_CACHE["live"]
    _MODEL_CACHE.update(at=time.time(), live={row["id"]: row for row in rows})
    return _MODEL_CACHE["live"]


def _model_options() -> list[dict]:
    live = _live_models()
    default = Settings.from_env().model.model
    ids = list(dict.fromkeys((default, *CURATED_MODELS)))
    options = []
    for model_id in ids:
        row = (live or {}).get(model_id)
        if live is not None and row is None:
            continue
        pricing = (row or {}).get("pricing") or {}
        try:
            price = [round(float(pricing[k]) * 1e6, 2) for k in ("prompt", "completion")]
        except (KeyError, TypeError, ValueError):
            price = None
        options.append({"id": model_id, "name": (row or {}).get("name") or model_id,
                        "price": price, "default": model_id == default})
    return options


def _valid_model(model: str) -> bool:
    if not MODEL_PATTERN.fullmatch(model):
        return False
    live = _live_models()
    return live is None or model in live

app = Flask(__name__)

JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()

# -- pipeline stage metadata --------------------------------------------
# Mirrors the order tracker.stage(...) is called in eq_report/pipeline/
# orchestrator.py. Kept here (not imported) because it is UI copy - the
# label/blurb a viewer sees - not pipeline logic.
STAGES: tuple[tuple[str, str, str], ...] = (
    ("planning", "Planning", "Scope what to research and where to pull it from"),
    ("acquisition", "Acquisition", "Pull market data, fundamentals and documents"),
    ("normalisation", "Normalisation", "Reconcile sources into canonical evidence"),
    ("evidence_ingestion", "Evidence store", "Write canonical evidence to the store"),
    ("analysis", "Analysis", "Run analytics and the segment research agents"),
    ("technical_appendix", "Technical appendix", "Fetch and chart Bloomberg OHLCV - runs "
     "alongside analysis, independent of the report draft"),
    ("synthesis", "Synthesis", "Draft the report's narrative and exhibits"),
    ("qa", "QA", "Check every claim against the evidence store; repair and re-check critical findings"),
    ("pdf", "Render PDF", "Lay out the full report"),
    ("compact_pdf", "Compact PDF", "Render the short version (brief, technical page, references) - runs alongside "
     "the full PDF"),
)
STAGE_KEYS = tuple(key for key, _, _ in STAGES)

# -- live stage tracking, without touching the pipeline itself -----------
# RunTracker.stage() is the pipeline's own instrumentation point (it already
# times every stage and tags log lines with it). We wrap it so the job
# dict updates the instant a stage starts/finishes, keyed by which
# background thread is currently inside it.
_THREAD_JOB: dict[int, str] = {}
_orig_stage = run_tracker_module.RunTracker.stage


@contextmanager
def _tracked_stage(self, name):  # noqa: ANN001 - mirrors RunTracker.stage's signature
    # A list, not a single value: the pipeline now genuinely runs some
    # stages concurrently (technical_appendix alongside analysis, pdf
    # alongside compact_pdf), so more than one can be "current" at once.
    job_id = _THREAD_JOB.get(threading.get_ident())
    if job_id:
        with _JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job.setdefault("active_stages", []).append(name)
                job["stage_started_at"] = time.time()
                # Recorded once, never overwritten: a client computing "how
                # long has this stage been running" from this timestamp gets
                # the same answer before and after a page refresh, unlike a
                # client-side Date.now() captured at first render, which a
                # refresh would reset to "just started".
                job.setdefault("stage_first_started_at", {}).setdefault(name, job["stage_started_at"])
    try:
        with _orig_stage(self, name) as timing:
            yield timing
    finally:
        if job_id:
            with _JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None:
                    active = job.setdefault("active_stages", [])
                    if name in active:
                        active.remove(name)
                    if name not in job["completed_stages"]:
                        job["completed_stages"].append(name)
                    completed_at = time.time()
                    job.setdefault("stage_completed_at", {})[name] = completed_at
                    started_at = job.setdefault("stage_first_started_at", {}).get(name)
                    if started_at is not None:
                        job.setdefault("stage_durations_ms", {})[name] = round(
                            (completed_at - started_at) * 1000, 1)


run_tracker_module.RunTracker.stage = _tracked_stage

# -- live cost/token tracking, without touching the pipeline itself -------
# UsageTracker.record() runs inside asyncio.to_thread worker threads (see
# eq_report/llm/usage.py's own docstring), which do NOT share a thread
# identity with the _run_job thread that _THREAD_JOB above is keyed by - so
# this uses a contextvars.ContextVar instead, which asyncio.to_thread
# explicitly propagates into the worker thread's context. Every completed
# LLM call updates the job dict immediately, so cost/tokens climb live
# instead of only appearing once the whole run finishes.
_JOB_CTX: contextvars.ContextVar[str | None] = contextvars.ContextVar("_webui_job_id", default=None)
_orig_record = usage_module.UsageTracker.record


def _tracked_record(self, event):  # noqa: ANN001 - mirrors UsageTracker.record's signature
    _orig_record(self, event)
    job_id = _JOB_CTX.get()
    if not job_id:
        return
    with _JOBS_LOCK:
        job = JOBS.get(job_id)
        if job is None:
            return
        job["cost_usd"] = round((job.get("cost_usd") or 0.0) + (event.cost_usd or 0.0), 6)
        job["input_tokens"] = (job.get("input_tokens") or 0) + event.input_tokens
        job["output_tokens"] = (job.get("output_tokens") or 0) + event.output_tokens
        job["call_count"] = (job.get("call_count") or 0) + 1
        row = job.setdefault("usage_by_stage", {}).setdefault(event.stage, {
            "calls": 0, "input_tokens": 0, "output_tokens": 0,
            "cost_usd": 0.0, "calls_missing_cost": 0,
        })
        row["calls"] += 1
        row["input_tokens"] += event.input_tokens
        row["output_tokens"] += event.output_tokens
        if event.cost_usd is not None:
            row["cost_usd"] = round(row["cost_usd"] + event.cost_usd, 6)
        else:
            row["calls_missing_cost"] += 1


usage_module.UsageTracker.record = _tracked_record

configure_logging("INFO")  # lock in handlers now so per-run calls don't reset them


def _run_job(
    job_id: str, ticker: str | None, report_date: dt.date | None,
    resume_from: str | None = None, model: str | None = None,
) -> None:
    _THREAD_JOB[threading.get_ident()] = job_id
    _JOB_CTX.set(job_id)
    with _JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = time.time()
    try:
        settings = Settings.from_env(output_dir=REPO_ROOT / "output_webui")
        if model:
            settings = replace(settings, model=replace(settings.model, model=model))
        if resume_from:
            # Re-runs analysis/synthesis/QA/rendering against that run's
            # already-persisted plan and Evidence Store rows - no planning
            # or acquisition call is repeated. See
            # eq_report.pipeline.orchestrator.generate_report's docstring.
            result = generate_report_sync(None, settings, resume_from=resume_from)
        else:
            research_request = ResearchRequest(
                company=ticker, ticker=ticker, report_date=report_date)
            result = generate_report_sync(research_request, settings)
        usage = (result.run.llm_usage or {}) if result.run else {}
        qa = result.qa_result
        run_dir = Path(result.report_json_path).parent if result.report_json_path else None
        artifact_candidates = {
            "compact_pdf": ("Compact PDF", result.compact_pdf_path),
            "full_pdf": ("Full PDF", result.pdf_path),
            "report_json": ("Report JSON", result.report_json_path),
            "qa_json": ("QA findings", run_dir / "07_qa.json" if run_dir else None),
            "repair_json": ("Repair log", run_dir / "07_qa_repair.json" if run_dir else None),
            "validation_json": (
                "Blocking details", run_dir / "validation_failure.json" if run_dir else None),
        }
        artifacts = {
            key: {"label": label, "path": str(path)}
            for key, (label, path) in artifact_candidates.items()
            if path is not None and Path(path).exists()
        }
        qa_reasons: list[dict] = []
        if qa is not None and qa.critical:
            counts: dict[str, int] = {}
            for finding in qa.critical:
                counts[finding.check] = counts.get(finding.check, 0) + 1
            qa_reasons = [
                {"check": check, "count": count}
                for check, count in sorted(counts.items(), key=lambda kv: -kv[1])
            ]
        with _JOBS_LOCK:
            # Prefer the pipeline tracker's own UTC timings once the run is
            # complete. The live wrapper above supplies the same values while
            # a job is running; these become the durable final record exposed
            # by /api/job/<id> and shown after refresh.
            stage_durations = {
                stage.stage: round(stage.duration_ms, 1)
                for stage in (result.run.stages if result.run else [])
                if stage.duration_ms is not None
            }
            resumed_ticker = (
                (result.run.request or {}).get("ticker") if (resume_from and result.run) else None
            )
            JOBS[job_id].update(
                status="done" if (result.succeeded and result.compact_pdf_path) else "failed",
                report_run_id=result.report_run_id,
                **({"ticker": f"{resumed_ticker} (resumed)"} if resumed_ticker else {}),
                run_status=result.status.value,
                compact_pdf=str(result.compact_pdf_path) if result.compact_pdf_path else None,
                pdf=str(result.pdf_path) if result.pdf_path else None,
                duration_ms=result.run.duration_ms if result.run else None,
                cost_usd=usage.get("total_cost_usd"),
                call_count=usage.get("call_count"),
                input_tokens=usage.get("total_input_tokens"),
                output_tokens=usage.get("total_output_tokens"),
                usage_by_stage=usage.get("by_stage") or {},
                stage_durations_ms=stage_durations,
                qa_critical=len(qa.critical) if qa else None,
                qa_warnings=len(qa.warnings) if qa else None,
                qa_reasons=qa_reasons,
                artifacts=artifacts,
                error=None if (result.succeeded and result.compact_pdf_path) else (
                    f"QA blocked publication: {len(qa.critical)} critical finding(s)."
                    if (qa is not None and qa.critical)
                    else result.summary()
                ),
            )
    except Exception as exc:  # keep the failure visible in the UI, not just the console
        with _JOBS_LOCK:
            JOBS[job_id].update(
                status="failed",
                error=f"{exc}\n\n{traceback.format_exc(limit=4)}",
            )
    finally:
        with _JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None and job.get("status") in {"done", "failed"}:
                job.setdefault("completed_at", time.time())
        _THREAD_JOB.pop(threading.get_ident(), None)


def _start_job(
    ticker: str | None, report_date: dt.date | None, *, resume_from: str | None = None,
    model: str | None = None,
) -> str:
    job_id = uuid.uuid4().hex[:8]
    with _JOBS_LOCK:
        JOBS[job_id] = {
            "ticker": ticker or (f"Resuming {resume_from}" if resume_from else ""),
            "report_date": report_date.isoformat() if report_date else None,
            "resume_from": resume_from,
            "model": model or Settings.from_env().model.model,
            "status": "queued",
            "active_stages": [],
            "completed_stages": [],
            "stage_first_started_at": {},
            "stage_completed_at": {},
            "stage_durations_ms": {},
        }
    threading.Thread(
        target=_run_job, args=(job_id, ticker, report_date, resume_from, model), daemon=True).start()
    return job_id


@app.route("/", methods=["GET"])
def index():
    return render_template_string(APP_HTML, initial_job_id=None)


@app.route("/generate", methods=["POST"])
def generate():
    resume_from = request.form.get("resume_from", "").strip()
    model = request.form.get("model", "").strip() or None
    if model and not _valid_model(model):
        return redirect(url_for("index"))
    if resume_from:
        if not RUN_ID_PATTERN.fullmatch(resume_from):
            return redirect(url_for("index"))
        job_id = _start_job(None, None, resume_from=resume_from, model=model)
        return redirect(url_for("job_page", job_id=job_id))
    ticker = request.form.get("ticker", "").strip().upper()
    date_str = request.form.get("report_date", "").strip()
    if not TICKER_PATTERN.fullmatch(ticker):
        return redirect(url_for("index"))
    try:
        report_date = dt.date.fromisoformat(date_str) if date_str else dt.date.today()
    except ValueError:
        report_date = dt.date.today()
    job_id = _start_job(ticker, report_date, model=model)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    resume_from = str(payload.get("resume_from", "")).strip()
    model = str(payload.get("model", "")).strip() or None
    if model and not _valid_model(model):
        return jsonify({"error": f"unknown model {model!r}; pick one from the list or "
                                 "enter an exact OpenRouter model id"}), 400
    if resume_from:
        if not RUN_ID_PATTERN.fullmatch(resume_from):
            return jsonify({"error": "resume_from must be a run id like "
                                     "run_20260929T060809_f9cebb"}), 400
        job_id = _start_job(None, None, resume_from=resume_from, model=model)
        return jsonify({"job_id": job_id})
    ticker = str(payload.get("ticker", "")).strip().upper()
    if not TICKER_PATTERN.fullmatch(ticker):
        return jsonify({"error": "a valid ticker is required"}), 400
    date_str = str(payload.get("report_date", "")).strip()
    try:
        report_date = dt.date.fromisoformat(date_str) if date_str else dt.date.today()
    except ValueError:
        report_date = dt.date.today()
    job_id = _start_job(ticker, report_date, model=model)
    return jsonify({"job_id": job_id})


@app.route("/api/models", methods=["GET"])
def api_models():
    return jsonify({"models": _model_options()})


@app.route("/assets/logo.png", methods=["GET"])
def logo():
    return send_file(REPO_ROOT / "eq_report" / "rendering" / "assets" / "megaannum_logo.png",
                     mimetype="image/png", max_age=86400)


@app.route("/job/<job_id>", methods=["GET"])
def job_page(job_id: str):
    if job_id not in JOBS:
        return "Unknown job.", 404
    return render_template_string(APP_HTML, initial_job_id=job_id)


@app.route("/api/job/<job_id>", methods=["GET"])
def job_status(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return jsonify({"error": "unknown job"}), 404
    return jsonify(job)


@app.route("/pdf/<job_id>", methods=["GET"])
def job_pdf(job_id: str):
    job = JOBS.get(job_id)
    if job is None or not job.get("compact_pdf"):
        return "No compact PDF for this job yet.", 404
    return send_file(job["compact_pdf"], mimetype="application/pdf")


@app.route("/document/<job_id>/<kind>", methods=["GET"])
def job_document(job_id: str, kind: str):
    """Open one allow-listed output produced by this in-memory job."""
    job = JOBS.get(job_id)
    artifact = (job or {}).get("artifacts", {}).get(kind)
    if not artifact:
        return "Document is not available for this job.", 404
    path = Path(artifact["path"])
    if not path.is_file():
        return "Document no longer exists.", 404
    mimetype = "application/pdf" if path.suffix.lower() == ".pdf" else "application/json"
    return send_file(path, mimetype=mimetype, as_attachment=False)


# -- visual system --------------------------------------------------------
# "EquityAI" shell: ticker input -> live step tracker -> report viewer,
# matching the product mockup the user supplied. Real wiring throughout -
# the five steps below are grouped from the pipeline's own tracked stages
# (see STAGE_KEYS/_tracked_stage above), not decorative placeholders, and
# the two report-viewer toggle buttons point at the two PDFs the pipeline
# actually produces (full_pdf/compact_pdf in job["artifacts"]).
STEP_GROUPS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("ingest", "Fetching company data", "Financials, filings, news, and market data...",
     ("planning", "acquisition", "normalisation", "evidence_ingestion")),
    ("analyze", "Analyzing with AI", "Identifying key insights...",
     ("analysis", "technical_appendix")),
    ("full_report", "Building full report", "Writing comprehensive analysis...",
     ("synthesis", "qa", "pdf")),
    ("compact_report", "Building compact report", "Creating executive summary...",
     ("compact_pdf",)),
    ("finalizing", "Finalizing", "Almost there...", ()),
)

APP_STYLE = """
    :root {
      --page-bg: #eef1f6; --card-bg: #ffffff; --border: #e4e8f0;
      --ink: #12172b; --ink-soft: #3d4459; --muted: #6b7280; --faint: #9aa2b1;
      --navy: #8f350b; --blue: #c04f15; --blue-soft: #f8e3d9; --blue-ring: #efc5b0;
      --step-line: #e4e8f0;
      --font-ui: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --font-mono: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
      /* Aliases for the workflow map (fc-*), carried over from the original
         report-styled palette so that markup can be reused verbatim. */
      --paper: var(--card-bg); --rule: var(--border); --hairline: #eef1f6;
      --accent: var(--navy); --accent-line: var(--blue); --accent-soft: var(--blue-soft);
      --pos: #1f9d6b; --pos-bg: #e6f7ef;
      --neg-border: #d64545; --neg-bg: #fdeeee; --neg-ink: #7a2020;
    }
    * { box-sizing: border-box; }
    body {
      font-family: var(--font-ui); margin: 0; color: var(--ink); background: var(--page-bg);
      -webkit-font-smoothing: antialiased;
    }
    .shell { max-width: 1180px; margin: 0 auto; padding: 36px 28px 56px; }

    .topbar { display: flex; align-items: flex-end; justify-content: space-between;
      margin-bottom: 28px; flex-wrap: wrap; gap: 10px; }
    .wordmark { font-size: 30px; font-weight: 800; letter-spacing: -0.01em; }
    .wordmark .hl { color: var(--blue); }
    .tagline-top { color: var(--muted); font-size: 13.5px; }
    .subtitle { color: var(--muted); font-size: 14px; margin-top: 2px; }

    .layout { display: grid; grid-template-columns: 380px 1fr; gap: 22px; align-items: start; }
    @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } }

    .card {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 14px;
      padding: 22px 22px 24px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
    }
    .card + .card { margin-top: 18px; }
    .card h2 { font-size: 16px; font-weight: 700; margin: 0 0 4px; color: var(--ink); }
    .card .hint { color: var(--muted); font-size: 13px; margin: 0 0 16px; line-height: 1.4; }

    .ticker-row { display: flex; gap: 10px; }
    .ticker-field {
      flex: 1; display: flex; align-items: center; gap: 8px; border: 1px solid var(--border);
      border-radius: 9px; padding: 0 12px; background: #fbfcfe;
    }
    .ticker-field svg { flex: none; color: var(--faint); }
    .ticker-field input {
      border: none; background: transparent; padding: 12px 0; font-size: 15px;
      font-weight: 700; letter-spacing: 0.02em; text-transform: uppercase; width: 100%;
      color: var(--ink); font-family: var(--font-ui);
    }
    .ticker-field input:focus { outline: none; }
    .ticker-field input::placeholder { color: var(--faint); font-weight: 600; }

    .btn {
      display: inline-flex; align-items: center; gap: 8px; border: none; border-radius: 9px;
      font-size: 14px; font-weight: 700; cursor: pointer; white-space: nowrap;
      font-family: var(--font-ui); transition: background 120ms, opacity 120ms, transform 120ms;
    }
    .btn:active { transform: scale(0.97); }
    .btn-primary { background: var(--navy); color: #fff; padding: 12px 18px; }
    .btn-primary:hover { background: #1b2947; }
    .btn-primary:disabled { opacity: 0.55; cursor: default; transform: none; }
    .btn-toggle {
      background: #fff; color: var(--ink-soft); border: 1px solid var(--border);
      padding: 9px 14px;
    }
    .btn-toggle.active { background: var(--navy); color: #fff; border-color: var(--navy); }
    .toggle-group { display: flex; gap: 8px; }

    .example-hint { color: var(--faint); font-size: 12.5px; margin-top: 12px; }
    .resume-toggle { margin: 10px 0 0; font-size: 12.5px; }
    .resume-toggle a { color: var(--blue); text-decoration: none; }
    .resume-toggle a:hover { text-decoration: underline; }
    .resume-row.ticker-row { margin-top: 10px; }
    #resume-hint.example-hint { margin-top: 8px; }

    /* Step tracker */
    .steps { position: relative; }
    .step { position: relative; display: flex; gap: 14px; padding-bottom: 22px; }
    .step:last-child { padding-bottom: 0; }
    .step::before {
      content: ""; position: absolute; left: 10px; top: 24px; bottom: -2px; width: 2px;
      background: var(--step-line);
    }
    .step:last-child::before { display: none; }
    .step-dot {
      flex: none; width: 21px; height: 21px; border-radius: 50%; margin-top: 1px;
      border: 2px solid var(--border); background: #fff; position: relative; z-index: 1;
      display: flex; align-items: center; justify-content: center;
      transition: border-color 160ms, background 160ms;
    }
    .step-dot::after {
      content: ""; width: 9px; height: 9px; border-radius: 50%; background: transparent;
      transition: background 160ms;
    }
    .step.current .step-dot { border-color: var(--blue); box-shadow: 0 0 0 3px var(--blue-ring); }
    .step.current .step-dot::after { background: var(--blue); }
    .step.done .step-dot { border-color: var(--blue); background: var(--blue); }
    .step.done .step-dot::after {
      background: transparent; width: 10px; height: 7px; border-radius: 0;
      border-left: 2px solid #fff; border-bottom: 2px solid #fff; transform: rotate(-45deg) translate(1px, -1px);
    }
    .step-body { flex: 1; padding-top: 0; }
    .step-title-row { display: flex; align-items: baseline; justify-content: space-between; gap: 10px; }
    .step-title { font-size: 14.5px; font-weight: 700; color: var(--faint); transition: color 160ms; }
    .step.current .step-title, .step.done .step-title { color: var(--ink); }
    .step-desc { font-size: 12.5px; color: var(--faint); margin-top: 2px; line-height: 1.4; }
    .step.current .step-desc { color: var(--muted); }
    .step-time {
      font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums;
      font-family: var(--font-mono); white-space: nowrap;
    }

    .error-note {
      margin-top: 16px; padding: 12px 14px; border-radius: 9px; background: #fdeeee;
      border: 1px solid #f3c8c8; color: #7a2020; font-size: 12.5px; line-height: 1.5;
      white-space: pre-wrap; font-family: var(--font-mono);
    }
    .qa-note {
      margin-top: 16px; padding: 12px 14px; border-radius: 9px; background: #fdeeee;
      border: 1px solid #f3c8c8; color: #7a2020; font-size: 13px; line-height: 1.5;
    }

    /* Report viewer */
    .viewer-card { min-height: 640px; display: flex; flex-direction: column; }
    .viewer-head { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
    .viewer-head h2 { margin: 0; }
    .viewer-pane {
      flex: 1; background: #f6f8fb; border: 1px solid var(--border); border-radius: 12px;
      display: flex; align-items: center; justify-content: center; min-height: 560px;
      overflow: hidden;
    }
    .viewer-pane embed { width: 100%; height: 100%; min-height: 560px; border: none; }
    .viewer-empty { text-align: center; padding: 40px; color: var(--muted); }
    .viewer-empty svg { color: var(--faint); margin-bottom: 14px; }
    .viewer-empty .big { font-size: 17px; font-weight: 700; color: var(--ink); margin-bottom: 6px; }
    .viewer-empty .small { font-size: 13.5px; color: var(--muted); }

    /* Workflow map - the pipeline's real branching shape (parallel
       acquisition sources, parallel analysis, the QA pass/fail gate), not a
       flattened step list. Nodes glow live from the same job data the step
       tracker above uses. */
    .map-card { margin-top: 18px; }
    .fc-scroll { overflow-x: auto; }
    .fc-wrap { position: relative; width: 680px; height: 716px; margin: 0 auto; }
    .fc-svg { position: absolute; top: 0; left: 0; width: 680px; height: 716px; pointer-events: none; }
    .fc-node {
      position: absolute; border: 1px solid var(--hairline); background: var(--paper);
      border-radius: 4px; padding: 8px 11px; display: flex; flex-direction: column;
      justify-content: center; box-sizing: border-box;
      transition: background 200ms, border-color 200ms, box-shadow 200ms;
    }
    .fc-node .fc-name { font-size: 12px; font-weight: 700; color: var(--ink-soft); line-height: 1.25; }
    .fc-node .fc-blurb { font-size: 10px; color: var(--muted); margin-top: 2px; line-height: 1.3; }
    .fc-node.fc-gate { border-color: var(--accent-line); background: var(--accent-soft); }
    .fc-node.fc-gate .fc-name { color: var(--accent); }
    .fc-node.fc-blocked { border-style: dashed; opacity: 0.5; }
    .fc-node.fc-blocked .fc-name { color: var(--neg-ink); }

    .fc-node.done { background: var(--pos-bg); border-color: var(--pos); }
    .fc-node.done .fc-name { color: var(--pos); }
    .fc-node.current {
      background: var(--accent-soft); border-color: var(--accent);
      animation: fc-pulse 1.6s cubic-bezier(0.23,1,0.32,1) infinite;
    }
    .fc-node.current .fc-name { color: var(--accent); }
    @keyframes fc-pulse {
      0%, 100% { box-shadow: 0 0 0 0 rgba(18, 57, 94, 0.38); }
      50% { box-shadow: 0 0 0 7px rgba(18, 57, 94, 0); }
    }
    .fc-node.fc-blocked.active {
      opacity: 1; background: var(--neg-bg); border-color: var(--neg-border); border-style: solid;
      animation: fc-pulse-red 1.6s cubic-bezier(0.23,1,0.32,1) infinite;
    }
    @keyframes fc-pulse-red {
      0%, 100% { box-shadow: 0 0 0 0 rgba(193, 68, 46, 0.38); }
      50% { box-shadow: 0 0 0 7px rgba(193, 68, 46, 0); }
    }
    @media (prefers-reduced-motion: reduce) {
      .fc-node.current, .fc-node.fc-blocked.active { animation: none; }
    }
    .fc-cost {
      position: absolute; top: -9px; right: -8px; background: var(--navy); color: #fff;
      font-family: var(--font-mono); font-size: 10px; font-weight: 700; padding: 2px 6px;
      border-radius: 99px; line-height: 1.3; box-shadow: 0 1px 2px rgba(16,24,40,0.25);
    }
    .fc-time {
      position: absolute; right: 6px; bottom: 4px; color: var(--faint);
      font-family: var(--font-mono); font-size: 9.5px; font-weight: 700;
      line-height: 1; font-variant-numeric: tabular-nums;
      background: color-mix(in srgb, var(--card-bg) 82%, transparent);
      padding-left: 3px;
    }
    .fc-node.current .fc-time { color: var(--accent); }
    .fc-node.done .fc-time { color: var(--pos); }

    /* Overall telemetry - elapsed/cost/tokens, anchored to server timestamps
       (see updateTelemetry) so a refresh never resets the clock. */
    .telemetry { display: flex; gap: 22px; margin: -6px 0 22px; flex-wrap: wrap; }
    .t-item { display: flex; align-items: baseline; gap: 6px; }
    .t-k { font-size: 10.5px; font-weight: 700; color: var(--faint); text-transform: uppercase; letter-spacing: 0.06em; }
    .t-v { font-size: 15px; font-weight: 700; color: var(--ink); font-variant-numeric: tabular-nums; font-family: var(--font-mono); }

    /* ---- refresh: brand header, stat cards, model picker, softer surfaces ---- */
    body { background: linear-gradient(180deg, #fbeee4 0, var(--page-bg) 360px) fixed; }
    .shell { padding-top: 26px; }
    .brandbar {
      display: flex; align-items: center; justify-content: space-between; gap: 14px;
      flex-wrap: wrap; margin-bottom: 20px;
    }
    .brand { display: flex; align-items: center; gap: 14px; }
    .brand-logo {
      width: 56px; height: 56px; border-radius: 14px; background: #fff; padding: 4px;
      box-shadow: 0 6px 18px rgba(143, 53, 11, 0.18), 0 0 0 1px rgba(143, 53, 11, 0.08);
    }
    .brand-co {
      font-size: 11px; font-weight: 800; letter-spacing: 0.16em; text-transform: uppercase;
      color: var(--blue);
    }
    .brand-title { font-size: 26px; font-weight: 800; letter-spacing: -0.02em; line-height: 1.15; }
    .tagline-top { font-size: 13.5px; color: var(--muted); font-style: italic; }

    .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 22px; }
    @media (max-width: 860px) { .stats { grid-template-columns: repeat(2, 1fr); } }
    .stat {
      background: var(--card-bg); border: 1px solid var(--border); border-radius: 12px;
      padding: 12px 16px; display: flex; flex-direction: column; gap: 3px;
      box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); min-width: 0;
    }
    .stat-k { font-size: 10.5px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--faint); }
    .stat-v {
      font-family: var(--font-mono); font-size: 18px; font-weight: 700; color: var(--ink);
      font-variant-numeric: tabular-nums; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
    }
    .stat-model .stat-v { font-size: 13.5px; line-height: 24px; }

    .card {
      border-radius: 16px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04), 0 8px 24px rgba(16, 24, 40, 0.04);
    }
    .card h2 { font-size: 17px; letter-spacing: -0.01em; }
    .btn-primary {
      background: linear-gradient(180deg, #cf5a1a, var(--navy)); box-shadow: 0 4px 12px rgba(143, 53, 11, 0.28);
    }
    .btn-primary:hover { background: linear-gradient(180deg, #d9631f, #9a3a0d); transform: translateY(-1px); }
    .btn-toggle:hover:not(.active) { border-color: var(--blue-ring); background: var(--blue-soft); }
    .ticker-field:focus-within, .select-wrap:focus-within, .model-custom:focus {
      border-color: var(--blue); box-shadow: 0 0 0 3px var(--blue-ring);
    }

    .model-block { margin-top: 16px; padding-top: 16px; border-top: 1px dashed var(--border); }
    .field-label { display: block; font-size: 11px; font-weight: 800; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin-bottom: 7px; }
    .select-wrap {
      position: relative; border: 1px solid var(--border); border-radius: 9px; background: #fbfcfe;
    }
    .select-wrap::after {
      content: ""; position: absolute; right: 14px; top: 50%; width: 7px; height: 7px;
      border-right: 2px solid var(--faint); border-bottom: 2px solid var(--faint);
      transform: translateY(-70%) rotate(45deg); pointer-events: none;
    }
    .select-wrap select {
      appearance: none; -webkit-appearance: none; width: 100%; border: none; background: transparent;
      padding: 11px 36px 11px 12px; font-size: 14px; font-weight: 600; color: var(--ink);
      font-family: var(--font-ui); cursor: pointer;
    }
    .select-wrap select:focus { outline: none; }
    .select-wrap select:disabled { opacity: 0.6; cursor: default; }
    .model-custom {
      width: 100%; margin-top: 8px; border: 1px solid var(--border); border-radius: 9px; background: #fbfcfe;
      padding: 10px 12px; font-size: 13.5px; font-family: var(--font-mono); color: var(--ink);
    }
    .model-custom:focus { outline: none; }
    .model-price { margin: 8px 0 0; font-size: 12.5px; color: var(--muted); line-height: 1.4; }

    .fc-node { border-radius: 8px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.05); }
    .map-card .hint { max-width: 760px; }
"""

# Absolute-positioned nodes + an SVG line layer, laid out to match the
# pipeline's real shape: request -> plan -> {market data | fundamentals |
# documents} -> normalisation -> evidence store -> {analytics | segment
# agents | technical appendix} -> synthesis -> QA -> (pass) {pdf | compact
# pdf} / (blocked) withheld. Technical appendix and compact/full PDF are
# genuinely concurrent in the pipeline now (see orchestrator.py), not just
# drawn that way - the diagram matches the real asyncio.gather() calls.
FLOWCHART_HTML = """
<div class="fc-scroll"><div class="fc-wrap">
  <svg class="fc-svg" viewBox="0 0 680 716">
    <defs>
      <marker id="fc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--rule)"></path>
      </marker>
      <marker id="fc-arrow-pos" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--pos)"></path>
      </marker>
      <marker id="fc-arrow-blue" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--blue)"></path>
      </marker>
      <marker id="fc-arrow-neg" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--neg-border)"></path>
      </marker>
    </defs>
    <g style="stroke:var(--rule);stroke-width:2;fill:none;">
      <path d="M340,72 L340,86"></path>
      <path d="M135,86 L545,86"></path>
      <path d="M135,86 L135,100"></path>
      <path d="M340,86 L340,100"></path>
      <path d="M545,86 L545,100"></path>

      <path d="M135,160 L135,174"></path>
      <path d="M340,160 L340,174"></path>
      <path d="M545,160 L545,174"></path>
      <path d="M135,174 L545,174"></path>
      <path d="M340,174 L340,186"></path>

      <path d="M340,242 L340,270"></path>

      <path d="M340,326 L340,340"></path>
      <path d="M135,340 L545,340"></path>
      <path d="M135,340 L135,354"></path>
      <path d="M340,340 L340,354"></path>
      <path d="M545,340 L545,354"></path>

      <path d="M135,414 L135,428"></path>
      <path d="M340,414 L340,428"></path>
      <path d="M545,414 L545,428"></path>
      <path d="M135,428 L545,428"></path>
      <path d="M340,428 L340,440"></path>

      <path d="M340,496 L340,524"></path>
    </g>
    <path d="M340,580 L340,612 L190,612 L190,634" style="stroke:var(--pos);stroke-width:2;fill:none;" marker-end="url(#fc-arrow-pos)"></path>
    <path d="M340,612 L490,612 L490,634" style="stroke:var(--pos);stroke-width:2;fill:none;" marker-end="url(#fc-arrow-pos)"></path>
    <rect x="294" y="584" width="92" height="18" rx="9" style="fill:var(--card-bg);stroke:var(--pos);stroke-width:1.5;"></rect>
    <text x="340" y="596.5" text-anchor="middle" style="font:700 10px var(--font-ui);fill:var(--pos);">clears QA</text>

    <path d="M240,538 L194,538" style="stroke:var(--neg-border);stroke-width:2;fill:none;" marker-end="url(#fc-arrow-neg)"></path>
    <text x="217" y="531" text-anchor="middle" style="font:700 9.5px var(--font-ui);fill:var(--neg-ink);">critical</text>
    <path d="M194,566 L238,566" style="stroke:var(--blue);stroke-width:2;fill:none;" marker-end="url(#fc-arrow-blue)"></path>
    <text x="217" y="581" text-anchor="middle" style="font:700 9.5px var(--font-ui);fill:var(--blue);">re-check</text>

    <path d="M440,552 L490,552" style="stroke:var(--neg-border);stroke-width:2;fill:none;stroke-dasharray:5 3;" marker-end="url(#fc-arrow-neg)"></path>
    <text x="465" y="545" text-anchor="middle" style="font:700 9.5px var(--font-ui);fill:var(--neg-ink);">unfixed</text>
  </svg>

  <div class="fc-node" data-key="planning" data-node="planning" style="left:240px;top:16px;width:200px;height:56px;">
    <div class="fc-name">Planning</div><div class="fc-blurb">Scope the research plan</div>
  </div>

  <div class="fc-node" data-key="acquisition" data-node="market_data" style="left:40px;top:100px;width:190px;height:60px;">
    <div class="fc-name">Market data</div><div class="fc-blurb">Price, cap, multiples</div>
  </div>
  <div class="fc-node" data-key="acquisition" data-node="fundamentals_data" style="left:245px;top:100px;width:190px;height:60px;">
    <div class="fc-name">Fundamentals</div><div class="fc-blurb">Financials &amp; KPIs</div>
  </div>
  <div class="fc-node" data-key="acquisition" data-node="documents_data" style="left:450px;top:100px;width:190px;height:60px;">
    <div class="fc-name">Documents</div><div class="fc-blurb">Filings, news, transcripts</div>
  </div>

  <div class="fc-node" data-key="normalisation" data-node="normalisation" style="left:240px;top:186px;width:200px;height:56px;">
    <div class="fc-name">Normalisation</div><div class="fc-blurb">Parsers first; verified LLM rescue for rejects</div>
  </div>

  <div class="fc-node" data-key="evidence_ingestion" data-node="evidence_store" style="left:240px;top:270px;width:200px;height:56px;">
    <div class="fc-name">Evidence store</div><div class="fc-blurb">Canonical facts, written</div>
  </div>

  <div class="fc-node" data-key="analysis" data-node="analytics_engine" style="left:40px;top:354px;width:190px;height:60px;">
    <div class="fc-name">Analytics engine</div><div class="fc-blurb">Growth, mix, surprise</div>
  </div>
  <div class="fc-node" data-key="analysis" data-node="segment_agents" style="left:245px;top:354px;width:190px;height:60px;">
    <div class="fc-name">Segment agents</div><div class="fc-blurb">Per-section research</div>
  </div>
  <div class="fc-node" data-key="technical_appendix" data-node="technical_appendix" style="left:450px;top:354px;width:190px;height:60px;">
    <div class="fc-name">Technical appendix</div><div class="fc-blurb">Live OHLCV chart, independent of the draft</div>
  </div>

  <div class="fc-node" data-key="synthesis" data-node="synthesis" style="left:240px;top:440px;width:200px;height:56px;">
    <div class="fc-name">Synthesis</div><div class="fc-blurb">Draft narrative &amp; exhibits</div>
  </div>

  <div class="fc-node fc-gate" data-key="qa" data-node="qa_gate" style="left:240px;top:524px;width:200px;height:56px;">
    <div class="fc-name">QA gate</div><div class="fc-blurb">Rules + LLM entailment + triage</div>
  </div>
  <div class="fc-node" data-key="qa" data-notime="1" data-node="qa_repair" style="left:8px;top:524px;width:186px;height:56px;">
    <div class="fc-name">QA repair loop</div><div class="fc-blurb">Trim or omit critical statements, then re-check</div>
  </div>
  <div class="fc-node fc-blocked" data-key="__blocked" style="left:492px;top:524px;width:170px;height:56px;">
    <div class="fc-name">Blocked</div><div class="fc-blurb">No PDF - fixes required</div>
  </div>

  <div class="fc-node" data-key="pdf" data-node="pdf" style="left:50px;top:636px;width:280px;height:60px;">
    <div class="fc-name">Render PDF</div><div class="fc-blurb">Full report, merged with the technical appendix</div>
  </div>
  <div class="fc-node" data-key="compact_pdf" data-node="compact_pdf" style="left:350px;top:636px;width:280px;height:60px;">
    <div class="fc-name">Compact PDF</div><div class="fc-blurb">Brief, technical page, references</div>
  </div>
</div></div>
"""

APP_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>EquityAI</title>
  <link rel="icon" href="data:,">
  <style>""" + APP_STYLE + """</style>
</head>
<body>
  <div class="shell">
    <header class="brandbar">
      <div class="brand">
        <img class="brand-logo" src="/assets/logo.png" alt="Megaannum">
        <div>
          <div class="brand-co">Megaannum Technology Limited</div>
          <div class="brand-title">Equity Research Studio</div>
        </div>
      </div>
      <div class="tagline-top">Faster insights. Deeper decisions.</div>
    </header>

    <div class="stats">
      <div class="stat"><span class="stat-k">Elapsed</span><span class="stat-v" id="t-elapsed">&mdash;</span></div>
      <div class="stat"><span class="stat-k">LLM cost</span><span class="stat-v" id="t-cost">$0.00</span></div>
      <div class="stat"><span class="stat-k">Tokens</span><span class="stat-v" id="t-tokens">&mdash;</span></div>
      <div class="stat stat-model"><span class="stat-k">Model</span><span class="stat-v" id="t-model">&mdash;</span></div>
    </div>

    <div class="layout">
      <div>
        <div class="card">
          <h2>1. Enter Ticker</h2>
          <p class="hint">Generate a full and compact equity research report.</p>
          <div class="ticker-row">
            <div class="ticker-field">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="7"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
              <input id="ticker" placeholder="AAPL" autocomplete="off" maxlength="10">
            </div>
            <button id="generate-btn" class="btn btn-primary" onclick="generateReports()">
              Generate Reports
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"><line x1="5" y1="12" x2="19" y2="12"></line><polyline points="12 5 19 12 12 19"></polyline></svg>
            </button>
          </div>
          <p class="example-hint">e.g. AAPL, MSFT, NVDA, TSLA</p>
          <div class="model-block">
            <label class="field-label" for="model-select">Model</label>
            <div class="select-wrap">
              <select id="model-select" onchange="onModelChange()"><option>Loading models&hellip;</option></select>
            </div>
            <input id="model-custom" class="model-custom" placeholder="provider/model-id, e.g. openai/gpt-5.6-sol" autocomplete="off" style="display:none;">
            <p class="model-price" id="model-price">Used for every LLM stage of the run.</p>
          </div>
          <p class="resume-toggle"><a href="#" onclick="toggleResumeRow(event)" id="resume-link">Resume a previous run instead</a></p>
          <div class="ticker-row resume-row" id="resume-row" style="display:none;">
            <div class="ticker-field">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><polyline points="1 4 1 10 7 10"></polyline><path d="M3.51 15a9 9 0 1 0 2.13-9.36L1 10"></path></svg>
              <input id="resume-run-id" placeholder="run_20260929T060809_f9cebb" autocomplete="off">
            </div>
            <button id="resume-btn" class="btn btn-primary" onclick="resumeReport()">
              Resume
            </button>
          </div>
          <p class="example-hint resume-row" id="resume-hint" style="display:none;">Re-runs analysis, synthesis, QA and rendering for an existing run_id, reusing its already-fetched evidence - no re-planning or re-acquisition.</p>
        </div>

        <div class="card">
          <h2>2. Generating Reports</h2>
          <div class="steps" id="steps"></div>
        </div>
      </div>

      <div class="card viewer-card">
        <div class="viewer-head">
          <h2>3. Report Viewer</h2>
          <div class="toggle-group">
            <button class="btn btn-toggle active" id="btn-full" onclick="selectViewer('full')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
              Full Report
            </button>
            <button class="btn btn-toggle" id="btn-compact" onclick="selectViewer('compact')">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
              Compact Report
            </button>
          </div>
        </div>
        <div class="viewer-pane" id="viewer-pane">
          <div class="viewer-empty">
            <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="9" y1="13" x2="15" y2="13"></line><line x1="9" y1="17" x2="13" y2="17"></line></svg>
            <div class="big">Your research report will appear here</div>
            <div class="small">Enter a ticker and click &ldquo;Generate Reports&rdquo; to get started.</div>
          </div>
        </div>
      </div>
    </div>

    <div class="card map-card">
      <h2>Workflow map</h2>
      <p class="hint">The pipeline's real shape - parallel acquisition sources, parallel analysis, the QA pass/fail gate. Nodes glow live as your report moves through it.</p>
      """ + FLOWCHART_HTML + """
    </div>
  </div>

  <script>
    const STEP_GROUPS = """ + json.dumps([[key, title, desc, list(stages)]
                                            for key, title, desc, stages in STEP_GROUPS]) + """;
    let jobId = """ + "{{ initial_job_id | tojson }}" + """;
    let viewerMode = "full";
    let pollTimer = null;
    let tickTimer = null;
    let MODELS = [];

    function fmtPrice(m) {
      return m.price ? `$${m.price[0]} in / $${m.price[1]} out per 1M tokens` : "price unavailable";
    }

    async function loadModels() {
      const sel = document.getElementById("model-select");
      try {
        MODELS = (await (await fetch("/api/models")).json()).models || [];
      } catch (err) { MODELS = []; }
      sel.innerHTML = MODELS.map(m =>
        `<option value="${m.id}">${m.id}${m.default ? "  (default)" : ""}</option>`).join("")
        + `<option value="__custom__">Custom model id&hellip;</option>`;
      const current = window.__lastJob && window.__lastJob.model;
      if (current && MODELS.some(m => m.id === current)) sel.value = current;
      onModelChange();
    }

    function selectedModel() {
      const sel = document.getElementById("model-select");
      return sel.value === "__custom__"
        ? document.getElementById("model-custom").value.trim() : sel.value;
    }

    function onModelChange() {
      const sel = document.getElementById("model-select");
      const custom = document.getElementById("model-custom");
      const price = document.getElementById("model-price");
      custom.style.display = sel.value === "__custom__" ? "block" : "none";
      const m = MODELS.find(x => x.id === sel.value);
      price.textContent = m
        ? `${m.name} - ${fmtPrice(m)}. Used for every LLM stage of the run.`
        : "Enter an exact OpenRouter model id. Used for every LLM stage of the run.";
    }

    // Every timestamp driving the UI's clocks comes from the server
    // (job.started_at, job.stage_first_started_at, job.stage_completed_at -
    // all wall-clock seconds recorded once, server-side, the first time each
    // stage starts). A tick just recomputes Date.now() - <server time>, so
    // refreshing the page mid-run shows the same true elapsed time instead
    // of restarting every clock at 0.
    function stageElapsedMs(job, stage) {
      if (!job) return null;
      const recorded = job.stage_durations_ms && job.stage_durations_ms[stage];
      if (recorded != null) return recorded;
      const started = job.stage_first_started_at && job.stage_first_started_at[stage];
      if (started == null) return null;
      const completed = job.stage_completed_at && job.stage_completed_at[stage];
      return Math.max(0, ((completed != null ? completed * 1000 : Date.now()) - started * 1000));
    }

    function groupElapsedMs(job, stages) {
      if (!job || !stages.length) return null;
      const starts = stages
        .map(stage => job.stage_first_started_at && job.stage_first_started_at[stage])
        .filter(value => value != null);
      if (!starts.length) return null;
      const allComplete = stages.every(stage =>
        job.stage_completed_at && job.stage_completed_at[stage] != null);
      const endMs = allComplete
        ? Math.max(...stages.map(stage => job.stage_completed_at[stage])) * 1000
        : Date.now();
      return Math.max(0, endMs - Math.min(...starts) * 1000);
    }

    function finalizingElapsedMs(job) {
      const started = job && job.stage_completed_at && job.stage_completed_at.compact_pdf;
      if (started == null) return null;
      const ended = job.completed_at != null ? job.completed_at * 1000 : Date.now();
      return Math.max(0, ended - started * 1000);
    }

    function renderSteps(job) {
      const completed = new Set((job && job.completed_stages) || []);
      const active = new Set((job && job.active_stages) || []);
      const done = job && (job.status === "done" || job.status === "failed");
      const html = STEP_GROUPS.map((group) => {
        const [key, title, desc, stages] = group;
        const isCurrent = !done && stages.length > 0 && stages.some(s => active.has(s));
        const groupDone = stages.length > 0 && stages.every(s => completed.has(s));
        const isFinalizing = key === "finalizing";
        const finalizingDone = isFinalizing && done;
        const finalizingCurrent = isFinalizing && !done && job && job.status === "running"
          && STEP_GROUPS.slice(0, -1).every(g => g[3].every(s => completed.has(s)));
        const cls = (groupDone || finalizingDone) ? "done" : (isCurrent || finalizingCurrent) ? "current" : "";
        const elapsedMs = isFinalizing
          ? finalizingElapsedMs(job)
          : groupElapsedMs(job, stages);
        return `<div class="step ${cls}" data-key="${key}">
          <div class="step-dot"></div>
          <div class="step-body">
            <div class="step-title-row">
              <span class="step-title">${title}</span>
              <span class="step-time" data-key="${key}">${elapsedMs != null ? fmtSecs(elapsedMs) : ""}</span>
            </div>
            <div class="step-desc">${desc}</div>
          </div>
        </div>`;
      }).join("");
      document.getElementById("steps").innerHTML = html;
    }

    function applyFlowchart(job) {
      const completed = new Set((job && job.completed_stages) || []);
      const active = new Set((job && job.active_stages) || []);
      document.querySelectorAll(".fc-node[data-key]").forEach(el => {
        const key = el.dataset.key;
        if (key === "__blocked") {
          el.classList.toggle("active", !!(job && job.qa_critical));
          return;
        }
        el.classList.remove("done", "current");
        if (completed.has(key)) {
          el.classList.add("done");
        } else if (active.has(key)) {
          el.classList.add("current");
        }
      });
      renderTimeByLayer(job);
    }

    function renderTimeByLayer(job) {
      document.querySelectorAll(".fc-node[data-key]").forEach(node => {
        let badge = node.querySelector(".fc-time");
        const key = node.dataset.key;
        const elapsed = key === "__blocked" || node.dataset.notime ? null : stageElapsedMs(job, key);
        if (elapsed == null) {
          if (badge) badge.remove();
          return;
        }
        if (!badge) {
          badge = document.createElement("div");
          badge.className = "fc-time";
          node.appendChild(badge);
        }
        badge.textContent = fmtSecs(elapsed);
      });
    }

    // Which usage_by_stage keys (see llm/usage.py's per-stage breakdown)
    // belong to each visual node, once the run is done. Nodes with no LLM
    // call in their stage (acquisition, evidence store, technical appendix,
    // PDF rendering) are left without a badge rather than shown as "$0.00"
    // everywhere, which would just be noise.
    const NODE_STAGE_MATCH = {
      planning: s => s === "planning",
      normalisation: s => s === "normalisation" || s.startsWith("normalisation:"),
      analytics_engine: s => s === "analytics",
      segment_agents: s => s.startsWith("agent:"),
      synthesis: s => s === "synthesis" || s === "synthesis_body",
      qa_gate: s => s === "qa" || s.startsWith("qa_") || s.startsWith("qa:"),
    };

    function renderCostByLayer(job) {
      document.querySelectorAll(".fc-cost").forEach(el => el.remove());
      const byStage = (job && job.usage_by_stage) || {};
      if (!Object.keys(byStage).length) return;
      document.querySelectorAll(".fc-node[data-node]").forEach(node => {
        const match = NODE_STAGE_MATCH[node.dataset.node];
        if (!match) return;
        const cost = Object.entries(byStage)
          .filter(([stage]) => match(stage))
          .reduce((sum, [, row]) => sum + (row.cost_usd || 0), 0);
        if (cost <= 0) return;
        node.style.position = "absolute"; // already true, kept explicit for the badge's anchor
        const badge = document.createElement("div");
        badge.className = "fc-cost";
        badge.textContent = "$" + cost.toFixed(2);
        node.appendChild(badge);
      });
    }

    function fmtSecs(ms) {
      const totalSeconds = Math.max(0, Math.round(ms / 1000));
      const minutes = Math.floor(totalSeconds / 60);
      const seconds = totalSeconds % 60;
      return minutes + "m " + String(seconds).padStart(2, "0") + "s";
    }

    function tickTimes() {
      const job = window.__lastJob || null;
      document.querySelectorAll(".step-time[data-key]").forEach(el => {
        const key = el.dataset.key;
        const group = STEP_GROUPS.find(g => g[0] === key);
        if (!group) return;
        const elapsed = key === "finalizing"
          ? finalizingElapsedMs(job)
          : groupElapsedMs(job, group[3]);
        if (elapsed != null) el.textContent = fmtSecs(elapsed);
      });
      renderTimeByLayer(job);
      updateTelemetry(job);
    }

    function updateTelemetry(job) {
      const elapsedEl = document.getElementById("t-elapsed");
      const costEl = document.getElementById("t-cost");
      const tokensEl = document.getElementById("t-tokens");
      if (!elapsedEl) return;
      if (job && job.started_at) {
        const endMs = job.duration_ms != null ? (job.started_at * 1000 + job.duration_ms) : Date.now();
        elapsedEl.textContent = fmtSecs(endMs - job.started_at * 1000);
      } else {
        elapsedEl.textContent = "—";
      }
      costEl.textContent = job && job.cost_usd != null ? "$" + job.cost_usd.toFixed(2) : "$0.00";
      const modelEl = document.getElementById("t-model");
      if (modelEl) modelEl.textContent = (job && job.model) || selectedModel() || "\u2014";
      const total = job ? (job.input_tokens || 0) + (job.output_tokens || 0) : 0;
      tokensEl.textContent = total ? total.toLocaleString() : "—";
    }

    function selectViewer(mode) {
      viewerMode = mode;
      document.getElementById("btn-full").classList.toggle("active", mode === "full");
      document.getElementById("btn-compact").classList.toggle("active", mode === "compact");
      renderViewer(window.__lastJob || null);
    }

    function renderViewer(job) {
      const pane = document.getElementById("viewer-pane");
      const artifacts = (job && job.artifacts) || {};
      const key = viewerMode === "full" ? "full_pdf" : "compact_pdf";
      if (job && job.status === "done" && artifacts[key]) {
        pane.innerHTML = `<embed src="/document/${jobId}/${key}" type="application/pdf">`;
      } else if (job && job.status === "failed") {
        const qaMsg = job.qa_critical
          ? `QA blocked publication: ${job.qa_critical} critical finding(s), ${job.qa_warnings} warning(s).`
          : (job.error || "The run failed.");
        pane.innerHTML = `<div class="viewer-empty">
          <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>
          <div class="big">This report could not be completed</div>
          <div class="small">${qaMsg.replace(/</g, "&lt;")}</div>
        </div>`;
      } else if (job) {
        pane.innerHTML = `<div class="viewer-empty">
          <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline></svg>
          <div class="big">Generating your report&hellip;</div>
          <div class="small">${job.ticker} &middot; this can take several minutes.</div>
        </div>`;
      } else {
        pane.innerHTML = `<div class="viewer-empty">
          <svg width="46" height="46" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="9" y1="13" x2="15" y2="13"></line><line x1="9" y1="17" x2="13" y2="17"></line></svg>
          <div class="big">Your research report will appear here</div>
          <div class="small">Enter a ticker and click &ldquo;Generate Reports&rdquo; to get started.</div>
        </div>`;
      }
    }

    function toggleResumeRow(evt) {
      evt.preventDefault();
      const row = document.getElementById("resume-row");
      const hint = document.getElementById("resume-hint");
      const shown = row.style.display !== "none";
      row.style.display = shown ? "none" : "flex";
      hint.style.display = shown ? "none" : "block";
      document.getElementById("resume-link").textContent =
        shown ? "Resume a previous run instead" : "Start a new report instead";
    }

    async function resumeReport() {
      const runId = document.getElementById("resume-run-id").value.trim();
      if (!runId) return;
      const btn = document.getElementById("resume-btn");
      btn.disabled = true;
      stepStartedAt = {};
      try {
        const res = await fetch("/api/generate", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({resume_from: runId, model: selectedModel()}),
        });
        const body = await res.json();
        if (!res.ok) {
          alert(body.error || "Could not start the resume run.");
          btn.disabled = false;
          return;
        }
        jobId = body.job_id;
        history.replaceState(null, "", `/job/${jobId}`);
        renderSteps(null);
        renderViewer({status: "running", ticker: `Resuming ${runId}`});
        poll();
      } catch (err) {
        alert("Could not reach the server: " + err);
        btn.disabled = false;
      }
    }

    async function generateReports() {
      const ticker = document.getElementById("ticker").value.trim().toUpperCase();
      if (!ticker) return;
      const btn = document.getElementById("generate-btn");
      btn.disabled = true;
      stepStartedAt = {};
      try {
        const res = await fetch("/api/generate", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ticker, model: selectedModel()}),
        });
        const body = await res.json();
        if (!res.ok) {
          alert(body.error || "Could not start the report.");
          btn.disabled = false;
          return;
        }
        jobId = body.job_id;
        history.replaceState(null, "", `/job/${jobId}`);
        renderSteps(null);
        renderViewer({status: "running", ticker});
        poll();
      } catch (err) {
        alert("Could not reach the server: " + err);
        btn.disabled = false;
      }
    }

    async function poll() {
      if (!jobId) return;
      const res = await fetch(`/api/job/${jobId}`);
      const job = await res.json();
      window.__lastJob = job;
      renderSteps(job);
      applyFlowchart(job);
      renderViewer(job);
      updateTelemetry(job);
      const btn = document.getElementById("generate-btn");
      const modelSel = document.getElementById("model-select");
      if (job.model && MODELS.some(m => m.id === job.model)) { modelSel.value = job.model; onModelChange(); }
      modelSel.disabled = job.status !== "done" && job.status !== "failed";
      if (job.ticker && !job.resume_from) document.getElementById("ticker").value = job.ticker;
      if (job.resume_from) document.getElementById("resume-run-id").value = job.resume_from;

      if (job.status === "done" || job.status === "failed") {
        btn.disabled = false;
        renderCostByLayer(job);
        clearTimeout(pollTimer);
        return;
      }
      btn.disabled = true;
      pollTimer = setTimeout(poll, 1200);
    }

    renderSteps(null);
    loadModels();
    tickTimer = setInterval(tickTimes, 1000);
    if (jobId) { renderViewer({status: "running"}); poll(); }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
