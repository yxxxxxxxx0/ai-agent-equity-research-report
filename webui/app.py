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
import os
import sys
import threading
import time
import traceback
import uuid
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, jsonify, redirect, render_template_string, request, send_file, url_for

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
    ("synthesis", "Synthesis", "Draft the report's narrative and exhibits"),
    ("qa", "QA", "Check every claim against the evidence store"),
    ("pdf", "Render PDF", "Lay out the full report"),
    ("technical_appendix", "Technical appendix", "Append the technical-analysis page"),
    ("compact_pdf", "Compact PDF", "Render the two-page short version"),
    ("annotate", "Annotate", "Build the reviewer-annotated companion PDF"),
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
    job_id = _THREAD_JOB.get(threading.get_ident())
    if job_id:
        with _JOBS_LOCK:
            job = JOBS.get(job_id)
            if job is not None:
                job["stage"] = name
                job["stage_started_at"] = time.time()
    try:
        with _orig_stage(self, name) as timing:
            yield timing
    finally:
        if job_id:
            with _JOBS_LOCK:
                job = JOBS.get(job_id)
                if job is not None and name not in job["completed_stages"]:
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


@app.route("/", methods=["GET"])
def index():
    jobs = list(reversed(list(JOBS.items())))
    return render_template_string(
        INDEX_HTML, jobs=jobs, today=dt.date.today().isoformat(), flowchart=FLOWCHART_HTML)


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

    job_id = uuid.uuid4().hex[:8]
    with _JOBS_LOCK:
        JOBS[job_id] = {
            "ticker": ticker,
            "report_date": report_date.isoformat(),
            "status": "queued",
            "stage": None,
            "completed_stages": [],
        }
    threading.Thread(target=_run_job, args=(job_id, ticker, report_date), daemon=True).start()
    return redirect(url_for("job_page", job_id=job_id))


@app.route("/job/<job_id>", methods=["GET"])
def job_page(job_id: str):
    job = JOBS.get(job_id)
    if job is None:
        return "Unknown job.", 404
    return render_template_string(
        JOB_HTML, job_id=job_id, job=job, stages=STAGES, stage_keys=STAGE_KEYS,
        flowchart=FLOWCHART_HTML)


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


# -- visual system --------------------------------------------------------
# Lifted directly from eq_report/rendering/pdf_renderer.py's own palette
# (ACCENT/POS/NEG/WARN/INK/...) so the control room shares one identity
# with the document it produces, rather than wearing a separate "app" skin.
BASE_STYLE = """
    :root {
      --ink: #1a1a1a; --ink-soft: #3a3f47; --muted: #5c6470; --faint: #8993a1;
      --rule: #c8ccd4; --hairline: #e6e9ee; --paper: #ffffff; --band: #eef1f5;
      --zebra: #f5f7fa;
      --accent: #12395e; --accent-soft: #e4ebf2; --accent-line: #9db6cc;
      --warn-bg: #fdf3e0; --warn-border: #d99a2b; --warn-ink: #7a4a00;
      --neg-bg: #fbe9e7; --neg-border: #c1442e; --neg-ink: #7a2a1a;
      --pos: #2f6f5e; --pos-bg: #e9f3f0;
      --font-ui: -apple-system, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
      --font-mono: "SFMono-Regular", ui-monospace, Menlo, Consolas, monospace;
    }
    * { box-sizing: border-box; }
    body {
      font-family: var(--font-ui); margin: 0; color: var(--ink); background: var(--paper);
      -webkit-font-smoothing: antialiased;
    }
    a { color: var(--accent); }
    /* 712px content column: wide enough that the 680px-wide workflow chart
       never triggers a scrollbar at a normal desktop width. */
    .wrap { max-width: 760px; margin: 0 auto; padding: 0 24px 64px; }

    /* Masthead - the same idea as the report's own masthead band */
    .masthead {
      background: var(--accent); color: #eaf0f6; padding: 20px 24px;
      margin-bottom: 34px; border-bottom: 3px solid var(--accent-line);
    }
    .masthead .wrap { padding: 0; display: flex; align-items: baseline; justify-content: space-between; }
    .masthead .mark {
      font-size: 10.5px; font-weight: 700; letter-spacing: 0.16em; text-transform: uppercase;
      color: #9db6cc;
    }
    .masthead h1 { font-size: 23px; margin: 3px 0 0; font-weight: 700; letter-spacing: -0.015em; color: #fff; }
    .masthead a.back { color: #cfdcea; text-decoration: none; font-size: 12px; font-weight: 600; }
    .masthead a.back:hover { color: #fff; }

    .lede { color: var(--muted); font-size: 13.5px; margin: 0 0 26px; line-height: 1.5; }

    .panel {
      background: var(--paper); border: 1px solid var(--hairline); border-radius: 3px;
      padding: 20px 22px; margin-bottom: 20px;
    }
    .panel + .panel { margin-top: -2px; }

    label {
      display: block; margin-top: 16px; font-size: 10.5px; font-weight: 700;
      color: var(--faint); text-transform: uppercase; letter-spacing: 0.08em;
    }
    input {
      font-size: 15px; padding: 9px 10px; border: 1px solid var(--rule); border-radius: 3px;
      width: 100%; margin-top: 7px; background: var(--zebra); color: var(--ink);
      font-variant-numeric: tabular-nums;
    }
    input:focus { outline: none; border-color: var(--accent); background: var(--paper);
      box-shadow: 0 0 0 3px var(--accent-soft); }
    button {
      margin-top: 22px; background: var(--accent); color: #fff; border: none;
      padding: 11px 20px; border-radius: 3px; font-size: 13.5px; font-weight: 600;
      letter-spacing: 0.01em; cursor: pointer;
      transition: transform 120ms cubic-bezier(0.23,1,0.32,1), background 120ms;
    }
    button:hover { background: #0d2c47; }
    button:active { transform: scale(0.97); }

    table { width: 100%; border-collapse: collapse; font-size: 13px; }
    th {
      text-align: left; padding: 7px 9px; color: var(--faint); font-weight: 700;
      font-size: 10.5px; text-transform: uppercase; letter-spacing: 0.06em;
      border-bottom: 1.5px solid var(--accent);
    }
    td { padding: 9px; border-bottom: 1px solid var(--hairline); font-variant-numeric: tabular-nums; }
    tr:last-child td { border-bottom: none; }
    tbody tr:nth-child(even) td { background: var(--zebra); }

    .tag {
      display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px 2px 7px;
      border-radius: 99px; font-size: 10.5px; font-weight: 700; text-transform: uppercase;
      letter-spacing: 0.05em; white-space: nowrap;
    }
    .tag::before { content: ""; width: 6px; height: 6px; border-radius: 50%; background: currentColor; }
    .tag-done, .tag-succeeded { background: var(--pos-bg); color: var(--pos); }
    .tag-failed, .tag-failed_qa { background: var(--neg-bg); color: var(--neg-ink); }
    .tag-running, .tag-queued { background: var(--warn-bg); color: var(--warn-ink); }

    .panel-title { font-size: 11px; font-weight: 700; color: var(--faint);
      text-transform: uppercase; letter-spacing: 0.07em; margin: 0 0 12px; }

    /* Workflow chart - the pipeline's real branching shape (parallel
       acquisition sources, parallel analysis, the QA pass/fail gate),
       not a flattened step list. Nodes glow live on the job page; the
       same markup sits inert on the home page as a map of how this works. */
    .fc-scroll { overflow-x: auto; margin-bottom: 20px; }
    .fc-wrap { position: relative; width: 680px; height: 940px; margin: 0 auto; }
    .fc-svg { position: absolute; top: 0; left: 0; width: 680px; height: 940px; pointer-events: none; }
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
"""

# Absolute-positioned nodes + an SVG line layer, laid out to match the
# pipeline's real shape: request -> plan -> {market data | fundamentals |
# documents} -> normalisation -> evidence store -> {analytics | segment
# agents} -> synthesis -> QA -> (pass) pdf.. / (blocked) withheld.
# See eq_report/pipeline/orchestrator.py's own module docstring for the
# canonical version of this diagram in prose.
FLOWCHART_HTML = """
<div class="fc-scroll"><div class="fc-wrap">
  <svg class="fc-svg" viewBox="0 0 680 940">
    <defs>
      <marker id="fc-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--rule)"></path>
      </marker>
      <marker id="fc-arrow-pos" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L10,5 L0,10 z" style="fill:var(--pos)"></path>
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
      <path d="M190,340 L490,340"></path>
      <path d="M190,340 L190,354"></path>
      <path d="M490,340 L490,354"></path>

      <path d="M190,414 L190,428"></path>
      <path d="M490,414 L490,428"></path>
      <path d="M190,428 L490,428"></path>
      <path d="M340,428 L340,440"></path>

      <path d="M340,496 L340,524"></path>

      <path d="M340,664 L340,692"></path>
      <path d="M340,748 L340,776"></path>
      <path d="M340,832 L340,860"></path>
    </g>
    <path d="M340,580 L340,605" style="stroke:var(--pos);stroke-width:2;fill:none;" marker-end="url(#fc-arrow-pos)"></path>
    <path d="M440,552 L462,552 L462,608 L477,608" style="stroke:var(--neg-border);stroke-width:2;fill:none;" marker-end="url(#fc-arrow-neg)"></path>
    <text x="346" y="596" style="font:700 9.5px var(--font-ui);fill:var(--pos);">clears QA</text>
    <text x="485" y="572" style="font:700 9.5px var(--font-ui);fill:var(--neg-ink);">critical findings</text>
  </svg>

  <div class="fc-node" data-key="planning" style="left:240px;top:16px;width:200px;height:56px;">
    <div class="fc-name">Planning</div><div class="fc-blurb">Scope the research plan</div>
  </div>

  <div class="fc-node" data-key="acquisition" style="left:40px;top:100px;width:190px;height:60px;">
    <div class="fc-name">Market data</div><div class="fc-blurb">Price, cap, multiples</div>
  </div>
  <div class="fc-node" data-key="acquisition" style="left:245px;top:100px;width:190px;height:60px;">
    <div class="fc-name">Fundamentals</div><div class="fc-blurb">Financials &amp; KPIs</div>
  </div>
  <div class="fc-node" data-key="acquisition" style="left:450px;top:100px;width:190px;height:60px;">
    <div class="fc-name">Documents</div><div class="fc-blurb">Filings, news, transcripts</div>
  </div>

  <div class="fc-node" data-key="normalisation" style="left:240px;top:186px;width:200px;height:56px;">
    <div class="fc-name">Normalisation</div><div class="fc-blurb">Reconcile into canonical facts</div>
  </div>

  <div class="fc-node" data-key="evidence_ingestion" style="left:240px;top:270px;width:200px;height:56px;">
    <div class="fc-name">Evidence store</div><div class="fc-blurb">Canonical facts, written</div>
  </div>

  <div class="fc-node" data-key="analysis" style="left:50px;top:354px;width:280px;height:60px;">
    <div class="fc-name">Analytics engine</div><div class="fc-blurb">Growth, mix, surprise</div>
  </div>
  <div class="fc-node" data-key="analysis" style="left:350px;top:354px;width:280px;height:60px;">
    <div class="fc-name">Segment agents</div><div class="fc-blurb">Per-section research</div>
  </div>

  <div class="fc-node" data-key="synthesis" style="left:240px;top:440px;width:200px;height:56px;">
    <div class="fc-name">Synthesis</div><div class="fc-blurb">Draft narrative &amp; exhibits</div>
  </div>

  <div class="fc-node fc-gate" data-key="qa" style="left:240px;top:524px;width:200px;height:56px;">
    <div class="fc-name">QA gate</div><div class="fc-blurb">Check every claim vs. evidence</div>
  </div>

  <div class="fc-node" data-key="pdf" style="left:240px;top:608px;width:200px;height:56px;">
    <div class="fc-name">Render PDF</div><div class="fc-blurb">Lay out the full report</div>
  </div>
  <div class="fc-node fc-blocked" data-key="__blocked" style="left:480px;top:580px;width:170px;height:60px;">
    <div class="fc-name">Blocked</div><div class="fc-blurb">No PDF - fixes required</div>
  </div>

  <div class="fc-node" data-key="technical_appendix" style="left:240px;top:692px;width:200px;height:56px;">
    <div class="fc-name">Technical appendix</div><div class="fc-blurb">Append the TA page</div>
  </div>
  <div class="fc-node" data-key="compact_pdf" style="left:240px;top:776px;width:200px;height:56px;">
    <div class="fc-name">Compact PDF</div><div class="fc-blurb">Two-page short version</div>
  </div>
  <div class="fc-node" data-key="annotate" style="left:240px;top:860px;width:200px;height:56px;">
    <div class="fc-name">Annotate</div><div class="fc-blurb">Reviewer companion PDF</div>
  </div>
</div></div>
"""

INDEX_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>eq_report</title>
  <link rel="icon" href="data:,">
  <style>""" + BASE_STYLE + """</style>
</head>
<body>
  <div class="masthead"><div class="wrap">
    <div><div class="mark">EQ Report &middot; Research Desk</div><h1>New report</h1></div>
  </div></div>
  <div class="wrap">
    <p class="lede">Pick a ticker, run the pipeline end to end, and preview the short
      (compact) layout the moment it clears QA.</p>

    <div class="panel">
      <form action="/generate" method="post">
        <label for="ticker">Ticker</label>
        <input id="ticker" name="ticker" placeholder="e.g. NVDA" autocomplete="off" required
               style="text-transform: uppercase; font-weight: 600; letter-spacing: 0.02em;">
        <label for="report_date">Report date</label>
        <input id="report_date" name="report_date" type="date" value="{{ today }}">
        <button type="submit">Generate report &rarr;</button>
      </form>
    </div>

    {% if jobs %}
    <div class="panel" style="padding: 0;">
      <table>
        <thead><tr><th style="padding-left: 22px;">Ticker</th><th>Date</th><th>Status</th><th>Cost</th><th style="padding-right: 22px;"></th></tr></thead>
        <tbody>
        {% for job_id, job in jobs %}
        <tr>
          <td style="padding-left: 22px; font-weight: 600;">{{ job.ticker }}</td>
          <td style="color: var(--muted);">{{ job.report_date }}</td>
          <td><span class="tag tag-{{ job.status }}">{{ job.status }}</span></td>
          <td>{{ "$%.2f"|format(job.cost_usd) if job.cost_usd is not none else "&mdash;" }}</td>
          <td style="padding-right: 22px; text-align: right;"><a href="/job/{{ job_id }}">view &rarr;</a></td>
        </tr>
        {% endfor %}
        </tbody>
      </table>
    </div>
    {% endif %}

    <p class="panel-title" style="margin-top: 30px;">How it works</p>
    {{ flowchart | safe }}
  </div>
</body>
</html>
"""

JOB_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>eq_report &middot; {{ job.ticker }}</title>
  <link rel="icon" href="data:,">
  <style>""" + BASE_STYLE + """
    .masthead .ticker { font-family: var(--font-mono); }

    /* Run telemetry - a single ticker-line strip (the report's own "Market
       data as at ..." note, not a row of boxed stat tiles), so it reads as
       a caption under the masthead rather than competing with the workflow
       chart for the page's one focal point. */
    .telemetry {
      display: flex; align-items: baseline; flex-wrap: wrap; gap: 3px 14px;
      padding: 0 0 16px; margin-bottom: 24px; border-bottom: 1px solid var(--hairline);
    }
    .telemetry .t-item { display: inline-flex; align-items: baseline; gap: 6px; }
    .telemetry .t-k { font-size: 10px; font-weight: 700; color: var(--faint);
      text-transform: uppercase; letter-spacing: 0.06em; }
    .telemetry .t-v { font-size: 14px; font-weight: 700; color: var(--ink-soft);
      font-variant-numeric: tabular-nums; letter-spacing: -0.005em; }

    .qa-panel { border: 1px solid var(--neg-border); background: var(--neg-bg);
      border-radius: 3px; padding: 16px 18px; margin-top: 20px; }
    .qa-panel .title { font-weight: 700; color: var(--neg-ink); font-size: 13px; margin-bottom: 10px; }
    .qa-panel table { font-size: 12.5px; }
    .qa-panel th { border-bottom-color: var(--neg-border); color: var(--neg-ink); opacity: 0.75; }
    .qa-panel td { border-bottom-color: rgba(193, 68, 46, 0.18); }
    .qa-panel tbody tr:nth-child(even) td { background: rgba(193, 68, 46, 0.05); }

    embed { width: 100%; height: 86vh; border: 1px solid var(--hairline); border-radius: 3px; margin-top: 20px; }
    pre { white-space: pre-wrap; background: var(--zebra); padding: 14px; border-radius: 3px;
      font-size: 11.5px; font-family: var(--font-mono); margin-top: 16px; border: 1px solid var(--hairline); }
  </style>
</head>
<body>
  <div class="masthead"><div class="wrap">
    <div>
      <a class="back" href="/">&larr; new report</a>
      <h1><span class="ticker">{{ job.ticker }}</span> <span style="opacity: 0.55; font-weight: 400;">&middot; {{ job.report_date }}</span></h1>
    </div>
    <span id="status-pill" class="tag tag-{{ job.status }}">{{ job.status }}</span>
  </div></div>

  <div class="wrap">
    <div class="telemetry">
      <span class="t-item"><span class="t-k">Elapsed</span><span class="t-v" id="m-time">0:00</span></span>
      <span class="t-item"><span class="t-k">LLM cost</span><span class="t-v" id="m-cost">$0.00</span></span>
      <span class="t-item"><span class="t-k">Tokens</span><span class="t-v" id="m-tokens">&mdash;</span></span>
    </div>

    <p class="panel-title">Pipeline</p>
    {{ flowchart | safe }}

    <div id="qa-wrap"></div>
    <div id="result"></div>
  </div>

  <script>
    const jobId = {{ job_id | tojson }};
    let clockTimer = null;
    // Anchored to the job's own started_at (set server-side, in seconds
    // since epoch) rather than the moment this page happened to load, so
    // refreshing the browser doesn't reset the clock to 0:00.
    let serverStartedAtMs = {{ (job.started_at * 1000) | tojson if job.started_at else "null" }};

    function fmtElapsed(ms) {
      const s = Math.max(0, Math.floor(ms / 1000));
      const m = Math.floor(s / 60);
      const r = s % 60;
      return m + ":" + String(r).padStart(2, "0");
    }

    function applyStages(job) {
      const completed = new Set(job.completed_stages || []);
      document.querySelectorAll(".fc-node[data-key]").forEach(el => {
        const key = el.dataset.key;
        if (key === "__blocked") {
          el.classList.toggle("active", !!job.qa_critical);
          return;
        }
        el.classList.remove("done", "current");
        if (completed.has(key)) {
          el.classList.add("done");
        } else if (job.stage === key) {
          el.classList.add("current");
        }
      });
    }

    function renderQa(job) {
      const wrap = document.getElementById("qa-wrap");
      if (job.qa_critical) {
        let rows = (job.qa_reasons || []).map(r =>
          `<tr><td>${r.check}</td><td>${r.count}</td></tr>`).join("");
        wrap.innerHTML = `
          <div class="qa-panel">
            <div class="title">QA blocked publication &middot; ${job.qa_critical} critical / ${job.qa_warnings} warning finding(s)</div>
            <table><thead><tr><th>Check</th><th>Count</th></tr></thead><tbody>${rows}</tbody></table>
          </div>`;
      } else {
        wrap.innerHTML = "";
      }
    }

    async function poll() {
      const res = await fetch(`/api/job/${jobId}`);
      const job = await res.json();

      document.getElementById("status-pill").textContent = job.status;
      document.getElementById("status-pill").className = "tag tag-" + job.status;
      applyStages(job);

      if (job.started_at) {
        serverStartedAtMs = job.started_at * 1000;
      }

      if (job.cost_usd != null) {
        document.getElementById("m-cost").textContent = "$" + job.cost_usd.toFixed(2);
      }
      if (job.input_tokens != null) {
        const total = (job.input_tokens || 0) + (job.output_tokens || 0);
        document.getElementById("m-tokens").textContent = total.toLocaleString();
      }
      if (job.duration_ms != null) {
        document.getElementById("m-time").textContent = fmtElapsed(job.duration_ms);
        clearInterval(clockTimer);
      }

      if (job.status === "done" && job.compact_pdf) {
        renderQa(job);
        document.getElementById("result").innerHTML =
          `<embed src="/pdf/${jobId}" type="application/pdf">`;
        return;
      }
      if (job.status === "failed") {
        renderQa(job);
        if (!job.qa_critical) {
          document.getElementById("result").innerHTML =
            `<pre>${(job.error || "Unknown error").replace(/</g, "&lt;")}</pre>`;
        }
        return;
      }
      setTimeout(poll, 1500);
    }

    clockTimer = setInterval(() => {
      if (serverStartedAtMs != null) {
        document.getElementById("m-time").textContent = fmtElapsed(Date.now() - serverStartedAtMs);
      }
    }, 1000);
    poll();
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5050, debug=False)
