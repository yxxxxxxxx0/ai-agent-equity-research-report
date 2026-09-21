"""A small local web interface for the eq_report pipeline.

Pick a ticker and a report date, run the real pipeline against it, watch it
move through each pipeline stage live, and view the resulting short
(compact, two-page) report layout in the browser.

Usage:
    python webui/app.py
    then open http://127.0.0.1:5050

Each run happens in a background thread so the page can poll progress
instead of holding the HTTP request open for however long the LLM stages
take. Live stage tracking is done by monkeypatching RunTracker.stage() from
here rather than editing the pipeline itself, so this file is the only place
that knows a web UI exists at all.

The visual language is deliberately the report's own, not a generic
dashboard theme: the same navy/ink/rust/teal palette and hairline-and-zebra
table style as eq_report/rendering/pdf_renderer.py, so this page reads as
the desk behind the printed note rather than an unrelated admin panel.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
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
from eq_report.pipeline import run_tracker as run_tracker_module  # noqa: E402
from eq_report.pipeline.orchestrator import generate_report_sync  # noqa: E402

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
    ("qa", "QA", "Check every claim against the evidence store"),
    ("pdf", "Render PDF", "Lay out the full report"),
    ("compact_pdf", "Compact PDF", "Render the two-page short version - runs alongside "
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


run_tracker_module.RunTracker.stage = _tracked_stage

configure_logging("INFO")  # lock in handlers now so per-run calls don't reset them


def _run_job(job_id: str, ticker: str, report_date: dt.date) -> None:
    _THREAD_JOB[threading.get_ident()] = job_id
    with _JOBS_LOCK:
        JOBS[job_id]["status"] = "running"
        JOBS[job_id]["started_at"] = time.time()
    try:
        research_request = ResearchRequest(company=ticker, ticker=ticker, report_date=report_date)
        settings = Settings.from_env(output_dir=REPO_ROOT / "output_webui")
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
            JOBS[job_id].update(
                status="done" if (result.succeeded and result.compact_pdf_path) else "failed",
                report_run_id=result.report_run_id,
                run_status=result.status.value,
                compact_pdf=str(result.compact_pdf_path) if result.compact_pdf_path else None,
                pdf=str(result.pdf_path) if result.pdf_path else None,
                duration_ms=result.run.duration_ms if result.run else None,
                cost_usd=usage.get("total_cost_usd"),
                call_count=usage.get("call_count"),
                input_tokens=usage.get("total_input_tokens"),
                output_tokens=usage.get("total_output_tokens"),
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
        _THREAD_JOB.pop(threading.get_ident(), None)


def _start_job(ticker: str, report_date: dt.date) -> str:
    job_id = uuid.uuid4().hex[:8]
    with _JOBS_LOCK:
        JOBS[job_id] = {
            "ticker": ticker,
            "report_date": report_date.isoformat(),
            "status": "queued",
            "active_stages": [],
            "completed_stages": [],
        }
    threading.Thread(target=_run_job, args=(job_id, ticker, report_date), daemon=True).start()
    return job_id


@app.route("/", methods=["GET"])
def index():
    return render_template_string(APP_HTML, initial_job_id=None)


@app.route("/generate", methods=["POST"])
def generate():
    ticker = request.form.get("ticker", "").strip().upper()
    date_str = request.form.get("report_date", "").strip()
    if not ticker:
        return redirect(url_for("index"))
    try:
        report_date = dt.date.fromisoformat(date_str) if date_str else dt.date.today()
    except ValueError:
        report_date = dt.date.today()
    job_id = _start_job(ticker, report_date)
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/api/generate", methods=["POST"])
def api_generate():
    payload = request.get_json(silent=True) or {}
    ticker = str(payload.get("ticker", "")).strip().upper()
    if not ticker:
        return jsonify({"error": "ticker is required"}), 400
    date_str = str(payload.get("report_date", "")).strip()
    try:
        report_date = dt.date.fromisoformat(date_str) if date_str else dt.date.today()
    except ValueError:
        report_date = dt.date.today()
    job_id = _start_job(ticker, report_date)
    return jsonify({"job_id": job_id})


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
      --navy: #0f1b33; --blue: #2f5fff; --blue-soft: #eaf0ff; --blue-ring: #c7d7ff;
      --step-line: #e4e8f0;
      --font-ui: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --font-mono: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
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
    <div class="topbar">
      <div>
        <div class="wordmark">Equity<span class="hl">AI</span></div>
        <div class="subtitle">AI-Powered Equity Research</div>
      </div>
      <div class="tagline-top">Faster insights. Deeper decisions.</div>
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
  </div>

  <script>
    const STEP_GROUPS = """ + json.dumps([[key, title, desc, list(stages)]
                                            for key, title, desc, stages in STEP_GROUPS]) + """;
    let jobId = """ + "{{ initial_job_id | tojson }}" + """;
    let viewerMode = "full";
    let stepStartedAt = {};
    let pollTimer = null;
    let tickTimer = null;

    function renderSteps(job) {
      const completed = new Set((job && job.completed_stages) || []);
      const active = new Set((job && job.active_stages) || []);
      const done = job && (job.status === "done" || job.status === "failed");
      const html = STEP_GROUPS.map((group, index) => {
        const [key, title, desc, stages] = group;
        const isCurrent = !done && stages.length > 0 && stages.some(s => active.has(s));
        const groupDone = stages.length > 0 && stages.every(s => completed.has(s));
        const isFinalizing = key === "finalizing";
        const finalizingDone = isFinalizing && done;
        const finalizingCurrent = isFinalizing && !done && job && job.status === "running"
          && STEP_GROUPS.slice(0, -1).every(g => g[3].every(s => completed.has(s)));
        const cls = (groupDone || finalizingDone) ? "done" : (isCurrent || finalizingCurrent) ? "current" : "";
        if ((isCurrent || finalizingCurrent) && !stepStartedAt[key]) stepStartedAt[key] = Date.now();
        const showTime = cls === "current" && stepStartedAt[key];
        return `<div class="step ${cls}" data-key="${key}">
          <div class="step-dot"></div>
          <div class="step-body">
            <div class="step-title-row">
              <span class="step-title">${title}</span>
              <span class="step-time" data-key="${key}">${showTime ? fmtSecs(Date.now() - stepStartedAt[key]) : ""}</span>
            </div>
            <div class="step-desc">${desc}</div>
          </div>
        </div>`;
      }).join("");
      document.getElementById("steps").innerHTML = html;
    }

    function fmtSecs(ms) {
      return Math.max(0, Math.round(ms / 1000)) + "s";
    }

    function tickTimes() {
      document.querySelectorAll(".step-time[data-key]").forEach(el => {
        const key = el.dataset.key;
        const row = el.closest(".step");
        if (row.classList.contains("current") && stepStartedAt[key]) {
          el.textContent = fmtSecs(Date.now() - stepStartedAt[key]);
        }
      });
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

    async function generateReports() {
      const ticker = document.getElementById("ticker").value.trim().toUpperCase();
      if (!ticker) return;
      const btn = document.getElementById("generate-btn");
      btn.disabled = true;
      stepStartedAt = {};
      try {
        const res = await fetch("/api/generate", {
          method: "POST", headers: {"Content-Type": "application/json"},
          body: JSON.stringify({ticker}),
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
      renderViewer(job);
      const btn = document.getElementById("generate-btn");
      if (job.ticker) document.getElementById("ticker").value = job.ticker;

      if (job.status === "done" || job.status === "failed") {
        btn.disabled = false;
        clearTimeout(pollTimer);
        return;
      }
      btn.disabled = true;
      pollTimer = setTimeout(poll, 1200);
    }

    renderSteps(null);
    tickTimer = setInterval(tickTimes, 1000);
    if (jobId) { renderViewer({status: "running"}); poll(); }
  </script>
</body>
</html>
"""

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
