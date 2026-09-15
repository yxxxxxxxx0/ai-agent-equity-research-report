# Automated Equity Research Report Generation

A local, end-to-end prototype that turns a research request into a draft equity
research PDF. The core stack is **Python, standard-library dataclasses for the
typed models, SQLite for the Evidence Store, `reportlab` for the PDF**.

It runs with no credentials and no network access: every branch falls back to a
clearly-labelled mock provider, and the label follows the data all the way onto
the front page of the PDF.

```bash
pip install -r requirements.txt
python -m eq_report --ticker NVDA --report-date 2026-09-02
```

> The report is a neutral analysis, not investment research: no
> Buy/Hold/Sell view, no price target, no bull/base/bear thesis, and no claim
> that a valuation is justified or unjustified - see "Neutral wording and
> analytical discipline" below for how that is enforced. `docs/PENDING_CHANGES.md`
> tracks a larger redesign in progress, split into what has shipped and what
> is still deferred.

Planning, the eight segment agents and Key Takeaways synthesis each have an
**optional GPT-backed implementation, called through OpenRouter**, that
replaces their deterministic counterpart when explicitly turned on via
`.env`/environment variables (see §5a). Every other stage — acquisition,
normalisation, the Evidence Store, the Analytics Engine, QA and rendering — is
always deterministic code with no model call, whichever mode is active. With
none of the model variables set, the pipeline behaves exactly as before: fully
deterministic, zero API cost.

---

## 1. Architecture implemented

The target architecture is preserved as one module per stage. No stage reaches
around another.

```
                                 ResearchRequest
                                        │
                              ResearchPlanner  ── ResearchPlan (typed)
                              (deterministic, or GPT via OpenRouter
                               when EQR_MODEL_API_KEY is set)
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        MarketDataService      FundamentalsService      DocumentsService     ← asyncio.gather
        (RawObservation)       (RawObservation)      (RawDocumentPassage)
                 └──────────────────────┼──────────────────────┘
                                        ▼
                              Normalisation layer          ← rejects, never coerces
                                        ▼
                        ╔═══════════════════════════╗
                        ║   Evidence Store (SQLite) ║      ← THE BOUNDARY
                        ╚═══════════════════════════╝
                                        ▼
                          EvidenceReader (read-only)
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
        AnalyticsEngine                              8 Segment Agents        ← asyncio.gather
        (deterministic, no LLM,                      (deterministic, or GPT via
         no config toggle)                            OpenRouter when
                                                       EQR_MODEL_USE_FOR_AGENTS=true)
                 └──────────────────────┬──────────────────────┘
                                        ▼
                              Synthesis layer  ── ReportDraft (typed)
                              (Key Takeaways deterministic, or GPT via
                               OpenRouter when EQR_MODEL_USE_FOR_SYNTHESIS=true;
                               every other section always deterministic)
                                        ▼
                                    QA engine  ── QAResult
                                        │
                        critical? ──────┴────── no critical
                            ▼                        ▼
                    no PDF, full record        PdfReportRenderer
                                                     ▼
                                            PDF + report JSON + run manifest
```

Whichever mode is active, the Analytics Engine and Evidence Store are never
touched by a model: an LLM-backed agent or synthesizer may only *select and
phrase* what to say, and only by citing `evidence_id`/`analytics_id` values it
was actually shown — any claim citing an unknown or missing id is dropped, not
trusted (`agents/llm_agent.py::_result_from_model`,
`synthesis/llm_synthesizer.py::_findings_from_model`). This is testable
end-to-end: running the same request once with the model off and once with it
on and diffing `04_analytics.json` between the two runs should show zero
numeric differences, since both paths compute analytics from the same
Evidence Store — only the narrative differs.

The **Evidence Store is a hard boundary**. The Analytics Engine and every
segment agent are constructed with an `EvidenceReader` and nothing else — no
provider, no HTTP client, no `requests` import anywhere downstream of it. The
architecture the spec forbids (`Segment Agent → external API → unsupported
claim`) is not merely discouraged here; there is no object in scope that could
do it. `tests/test_pipeline_e2e.py::test_agents_read_only_from_the_evidence_store`
hands the agents an empty store and asserts they produce data gaps rather than
content.

**Provenance chain.** Every printed sentence carries its own lineage:

```
Statement.text
  ├── claim_type          reported | management | market expectation | calculated | interpretation
  ├── citation_refs       [1][2]  →  Citation  →  evidence_id  →  EvidenceItem  →  source_name/url
  ├── evidence_ids        →  EvidenceItem.raw_metric / raw_value (what the provider literally said)
  └── analytics_ids       →  AnalyticsResult.formula + inputs + input_evidence_ids  →  EvidenceItem
```

`test_every_report_claim_traces_back_to_a_source` walks that chain for every
statement in a generated report.

---

## 2. Directory structure

```
eq_report/
├── config.py                    # all env-var reading; the only place credentials are read
├── errors.py                    # typed exceptions + PipelineError (structured, recorded)
├── logging_setup.py             # structured logging, run_id + stage on every record
├── cli.py / __main__.py         # entry point
│
├── domain/                      # stage-agnostic typed models (frozen dataclasses)
│   ├── enums.py                 #   controlled vocabularies: SourceType, ClaimType, Severity, …
│   ├── request.py               # 1. ResearchRequest
│   ├── plan.py                  # 2. ResearchPlan, ResearchQuestion, SegmentTask, DocumentRequirement
│   ├── observation.py           # 3. RawObservation, RawDocumentPassage, SourceRef, ProviderResult
│   ├── evidence.py              # 5. EvidenceItem, FiscalPeriod
│   ├── analytics.py             # 6. AnalyticsResult, AnalyticsBundle
│   ├── segment.py               # 7. SegmentResult, KeyFinding, MetricHighlight, DataGap
│   ├── report.py                # 8. ReportDraft, ReportSectionDraft, Statement, Citation, ChartSpec
│   ├── qa.py                    # 9. QAResult, QAFinding
│   └── run.py                   # 12. ReportRun, StageTiming
│
├── planning/
│   ├── research_planner.py          # 2. request → plan. Performs no research. GPT plan via
│   │                                 #   OpenRouter when EQR_MODEL_API_KEY is set.
│   └── openrouter_client.py         #   OpenRouter chat-completions client for the planner
│
├── llm/client.py                    #   generic OpenRouter JSON-mode client, shared by the
│                                     #   segment agents, synthesis and ArcticDB fiscal-period
│                                     #   resolution — output validated, never trusted directly
│
├── providers/                       # 3. acquisition interfaces + implementations
│   ├── base.py                      #   DataProvider / MarketData / Fundamentals / Documents ABCs
│   ├── registry.py                  #   config-driven selection; ArcticDB > Megadata > HTTP > mock
│   ├── sample_data.py               #   ★ ALL synthetic data lives here, nowhere else
│   ├── megadata.py                  #   real HTTP provider set (enabled by EQR_MEGADATA_BASE_URL)
│   ├── arcticdb/                    #   real provider set backed by ArcticDB (enabled by
│   │   ├── client.py · market_data.py · fundamentals.py · documents.py
│   │   └── fiscal_period.py         #   optionally uses the generic OpenRouter client to
│   │                                 #   resolve fiscal-period labels
│   ├── openrouter_search.py         #   last real-data tier: asks GPT (OpenRouter's web-search
│   │                                 #   plugin) to look up the plan's required figures online
│   │                                 #   when no other real provider is configured; falls back
│   │                                 #   to mock per-branch only if the search finds nothing
│   ├── market_data/http_provider.py #   generic real REST seam (enabled by credentials)
│   ├── market_data/mock_provider.py
│   ├── fundamentals/mock_provider.py
│   └── documents/mock_provider.py
│
├── acquisition/services.py          # 3. three branches, run concurrently, never raise
│
├── normalisation/
│   ├── canonical_metrics.py         # 4. alias table, metric properties, display labels
│   ├── units.py                     # 4. number/currency/percent parsing and formatting
│   ├── dates.py                     # 4. date + fiscal-period canonicalisation and arithmetic
│   └── normalizer.py                # 4. raw → EvidenceItem, with rejections
│
├── evidence/
│   ├── store.py                     # 5. SQLite store + EvidenceQuery (all required dimensions)
│   └── reader.py                    # 5. read-only, run-scoped view handed downstream
│
├── analytics/
│   ├── calculations.py              # 6. pure functions — no I/O, independently testable
│   └── engine.py                    # 6. evidence → AnalyticsResult, with evidence ids
│
├── agents/
│   ├── base.py                      # 7. SegmentAgent ABC, AgentContext, shared helpers
│   ├── runner.py                    # 7. plan-driven construction + concurrent execution;
│   │                                 #   picks LLMSegmentAgent for every segment when
│   │                                 #   EQR_MODEL_USE_FOR_AGENTS=true, else the deterministic ones
│   ├── llm_agent.py                 # 7. generic evidence-tagged GPT agent for any one segment
│   └── company_snapshot.py · financial_performance.py · operating_drivers.py
│       recent_developments.py · valuation.py · competitive_landscape.py
│       risks_catalysts.py · what_matters_next.py
│
├── synthesis/
│   ├── synthesizer.py               # 8. segment results → ReportDraft
│   ├── llm_synthesizer.py           # 8. GPT-selected Key Takeaways when
│   │                                 #   EQR_MODEL_USE_FOR_SYNTHESIS=true; every other section
│   │                                 #   inherited unchanged from Synthesizer
│   ├── terminology.py               # 8. house style + near-duplicate fingerprinting
│   └── citations.py                 # 8. reference numbering, first-use order
│
├── qa/
│   ├── checks.py                    # 9. 18 independent checks in five families
│   └── engine.py                    # 9. runs them all; a broken check is itself critical
│
├── rendering/
│   ├── pdf_renderer.py              # 10. consumes ReportDraft; computes nothing
│   └── json_writer.py               # 10. report JSON + run manifest
│
└── pipeline/
    ├── orchestrator.py              # 11. generate_report() — the single entry point
    └── run_tracker.py               # 12. ReportRun assembly and stage timing

examples/         nvidia_request.json — a structured request
output/runs/<id>/ per-run artefacts (see §4)
```

79 Python files, ~12,100 lines in `eq_report/` (there is currently no `tests/`
directory in this checkout).

---

## 3. Important design decisions

**Deterministic segment agents by default; GPT is a config-toggled swap.** Each
deterministic agent composes findings from evidence and analytics by explicit
rule, with no model call, which makes "agents must not invent facts"
*structural* rather than a matter of prompt discipline. `SegmentAgent` is an
ABC returning `SegmentResult`; `LLMSegmentAgent` (`agents/llm_agent.py`) is the
alternative implementation used for every segment when
`EQR_MODEL_USE_FOR_AGENTS=true` — it gets the same guarantee through
post-hoc validation instead of structure: any claim whose cited
`evidence_id`/`analytics_id` doesn't resolve against what it was actually shown
is dropped, never trusted. Key Takeaways synthesis has the same pair
(`Synthesizer` / `LLMSynthesizer`), toggled by `EQR_MODEL_USE_FOR_SYNTHESIS`.

**No LLM in the Analytics Engine, ever — no config toggle exists for it.**
Every number is computed by a pure function in `analytics/calculations.py`, and
the QA layer *re-derives* each one through a second explicit dispatch on the
stored formula string (`qa/checks.py::_recompute`). Tampering with a stored
value fails QA. This holds whether or not the segment agents or synthesis are
running on GPT — neither ever writes to the Evidence Store or the analytics
bundle, only reads from them.

**Rejection over coercion.** `units.parse_number` raises `NormalisationError`
for `None`, `""`, `"n/a"`, `True`, `NaN`, `"12.5 zorkmids"`. The normaliser
catches it, records a `Rejection` with a reason, and the value never becomes
evidence. Twelve parametrised cases cover this. (`"31.4x"` — a multiple written
the way it conventionally is — used to be one of the false-positive
rejections: `parse_number` treated the trailing `x` as an unrecognised
magnitude suffix rather than the identity scale a multiple needs. The mock
data always supplied bare floats for `forward_pe`/`trailing_pe`/`ev_to_sales`
so this never surfaced until the OpenRouter web-search provider — §5b — began
returning multiples the way a real source actually writes them. Fixed in
`units._SCALE_SUFFIXES`.)

**Exhibits are built centrally, not per agent.** `LLMSegmentAgent`'s
retrieved-evidence pool (point metrics, period metrics, segment/KPI rows,
guidance, peer values, matching documents) is deliberately broad and largely
the *same* broad pool for every one of the 8 segments — a segment only differs
in which findings and headline the model chooses to write from it. Earlier
versions rendered each section's table from that pool, so the report printed
much the same twenty metrics in seven near-identical tables, with a
`Comparison` column that was empty in every row because nothing ever filled it.
Tables and charts are now built once by `synthesis/exhibits.py` from the
Evidence Store, on three rules: one exhibit per question (what the quarter
delivered, where the revenue comes from, how it compares with peers), so no
figure is printed twice; growth/mix/surprise columns are filled from the
Analytics Engine, which had already computed them; and an exhibit whose
evidence is missing is not drawn at all. `important_metrics` remains on the
`SegmentResult` and in the run JSON as the record of what each agent judged
material.

**The first page carries a data panel.** `synthesis/key_data.py` builds the
company's headline figures — market data, valuation, the last reported quarter,
guidance — and the renderer sets them in a right-hand column beside the opening
section, as a sell-side first page does. Key Takeaways is hoisted to lead so
the reader meets the argument and the figures behind it together. Because the
panel states the price, scale and multiples, no body table has to repeat them.

**The report's shape follows the evidence.** The plan lists the sections a
reader asked about; `Synthesizer._prune` decides which of them earned a
heading. A section with fewer than two statements and no exhibit is folded into
its sibling where one is defined (catalysts into risks, retitled "Risks and
Catalysts") and otherwise dropped. Every omission is recorded on
`ReportDraft.metadata["sections_omitted"]` with its reason, printed in the back
matter, and checked by QA: an omission *with* a recorded reason is a disclosed
editorial decision (WARNING), while a section that vanished with no reason
recorded is still CRITICAL, because that is the signature of a section lost by
accident rather than dropped on purpose.

**A column has to earn its place.** The renderer drops any table column no row
filled (`pdf_renderer.py::_metric_table`). Emptiness is decided at render time
rather than upstream because it is a presentation question: synthesis is
entitled to ask for a "vs consensus" column and leave it blank where no
consensus exists, and a column of blanks reads as missing data rather than as
a question the report did not need to ask.

**Raw values are never destroyed.** `EvidenceItem.raw_metric` and `raw_value`
keep what the provider literally said (`"$62,300,000,000"`, `"totalRevenue"`)
alongside the normalised `62300000000.0` / `revenue`.

**Deterministic evidence ids.** `make_evidence_id` is a hash of
(run, entity, metric, period, date, source, discriminator) rather than a UUID.
Re-running over the same inputs yields the same ids, which makes diffing two
report runs and debugging a citation tractable. Exact duplicates collapse.

**Metric identity beats vendor unit strings.** A gross margin is a percentage
whatever the vendor called it; a margin *change* is percentage points and is
differenced, never divided (−0.9pp, not −1.2%). QA enforces both.

**"Latest period" means latest *reported* period.** Guidance and consensus rows
legitimately carry future periods, so `latest_reported_period()` is defined by
reported fundamentals, not by the maximum date in the table. Getting this wrong
would have anchored the whole report on a guided quarter.

**Duplicate detection is scoped to the report body.** Key Takeaways is a summary
layer and is *expected* to restate a body claim, so it renders against its own
discarded scope. Suppressing a body claim because the summary made it would gut
the detail sections. QA's duplication check skips Key Takeaways for the same
reason.

**One agent, two sections.** The risk/catalyst agent tags its findings `risk` /
`catalyst` and writes a narrative for each; the synthesis layer splits them.

**Sample data is quarantined and labelled.** All synthetic values live in
`providers/sample_data.py`. `is_mock` propagates provider → observation →
evidence → citation → `ReportDraft.contains_mock_data` → a banner on page 1, a
footer on every page, `[sample data]` on every source line, and a QA warning.
The values are deliberately *messy* (`"$62,300,000,000"`, `"62.3B"`, `"74.8%"`,
`"Aug 19, 2026"`, `"19/08/2026"`, `totalRevenue` / `Gross Margin` / `forwardPE`)
so normalisation is genuinely exercised rather than bypassed.

**Failure policy.** Recoverable problems become `PipelineError` records attached
to the run; only an unrecoverable stage failure raises. A provider that throws
is trapped by `DataProvider.fetch` and returned as a `FAILED` `ProviderResult`.
A branch that dies does not stop the run — the loss surfaces as data gaps.

---

## 4. How data flows, stage by stage

| Stage | Input → Output | Where it lands on disk |
|---|---|---|
| 1. Request | free text or JSON → `ResearchRequest` | `run.request` in the manifest |
| 2. Planner | `ResearchRequest` → `ResearchPlan` (questions, required metrics/documents/analytics, 8 segment tasks, source priorities) | `01_plan.json` |
| 3. Acquisition | `ResearchPlan` → 3 × `AcquisitionResult` **concurrently** (`asyncio.gather`), each with per-provider status, errors and warnings | `02_acquisition.json` |
| 4. Normalisation | `RawObservation` / `RawDocumentPassage` → `EvidenceItem` + `Rejection` list | `03_normalisation.json` (every evidence item, in full) |
| 5. Evidence Store | `EvidenceItem` → SQLite; downstream gets `EvidenceReader` | `output/evidence.sqlite3` |
| 6. Analytics | evidence → `AnalyticsResult` (value, unit, formula, inputs, `input_evidence_ids`) | `04_analytics.json` |
| 7. Agents | plan + reader + analytics → 8 × `SegmentResult` **concurrently** | `05_segments.json` |
| 8. Synthesis | segment results → `ReportDraft` (sections, statements, tables, charts, citations, gaps) | `06_report_draft.json` |
| 9. QA | `ReportDraft` → `QAResult` (18 checks, five families) | `07_qa.json` |
| 10. Render | `ReportDraft` → PDF + report JSON | `<TICKER>_<date>_<run>.pdf`, `report_<run>.json` |
| 12. Tracking | everything above → `ReportRun` | `run_<run>.json` |

Every intermediate object is dumped, so any stage can be inspected without
re-running the pipeline. A run that fails QA still writes the draft, the QA
result and the manifest — only the PDF is withheld.

**The reasoning chain** is explicit in `synthesizer._TAKEAWAY_ORDER`: what
changed → why → financial impact → surprise vs expectations → forward
expectations → implication for the multiple → relative position → what could
break it → what matters next.

**QA check families** (severity policy: `CRITICAL` blocks the PDF):

- *Evidence* — referenced ids exist; no unsupported numbers; reference numbers resolve; weak-evidence-only claims flagged.
- *Numerical* — every analytic independently recomputed; units known and semantically right; decimal-fraction-as-percentage detection.
- *Temporal* — periods comparable; no observation dated after the report; the analysed period is named in prose.
- *Consistency* — one name per ticker, one ticker per company, one currency; no contradictory values for the same metric and period (>1% spread is critical, rounding is a warning).
- *Narrative* — requested sections present and populated; unsourced causal claims; verbatim duplication; undisclosed data gaps; mock-data disclosure.

---

## 5. How to run the prototype

```bash
pip install -r requirements.txt

# ticker and report date
python -m eq_report --ticker NVDA --report-date 2026-09-02

# optional operational output location
python -m eq_report --ticker NVDA --report-date 2026-09-02 --output-dir output

# inspect the parsed request without running anything
python -m eq_report "…" --print-request
```

From Python:

```python
import asyncio
from eq_report import ResearchRequest, Settings, generate_report

request = ResearchRequest.from_dict({"company": "NVIDIA", "ticker": "NVDA"})
result = asyncio.run(generate_report(request, Settings.from_env()))
print(result.summary(), result.pdf_path)
```

Configuration is entirely environment-driven — see `.env.example`. Nothing reads
`os.environ` outside `config.py`, and `Settings.describe()` (what gets logged)
reports only *whether* a credential is present, never its value; there is a test
asserting that.

Exit code is `0` on success, `1` if QA blocked the PDF or a stage failed.

### 5a. Optional GPT-backed stages (OpenRouter)

Every variable below is optional; leaving them all unset keeps the pipeline
fully deterministic. Set them in `.env` (see `.env.example`) or as environment
variables:

```bash
EQR_MODEL_PROVIDER=openrouter
EQR_MODEL_NAME=openai/gpt-5              # any OpenRouter chat-completions model
EQR_MODEL_API_KEY=sk-or-v1-...           # or set OPENROUTER_API_KEY instead

# Setting only the key above turns on GPT planning. These are separate,
# independent opt-ins on top of that:
EQR_MODEL_USE_FOR_AGENTS=true            # swap all 8 segment agents for LLMSegmentAgent
EQR_MODEL_USE_FOR_SYNTHESIS=true         # swap Key Takeaways selection for LLMSynthesizer
```

Whichever stages are on, they only ever *select and phrase* — every claim they
return must cite an `evidence_id`/`analytics_id` copied verbatim from the rows
they were shown; anything else is dropped before it reaches the draft. A
useful sanity check after enabling these: run the same request once with the
variables unset and once with them set, then diff `04_analytics.json` between
the two run directories — the values should be identical, since the Analytics
Engine and Evidence Store are never in the model's path. A run's
`run_<id>.json` manifest and stdout logs record which mode produced it;
pass `EQR_LOG_JSON=true` to get per-call `input_tokens`/`output_tokens` in the
log stream for cost tracking.

### 5b. Data acquisition fallback: OpenRouter web search

Provider selection per branch (market data / fundamentals / documents) is:
**ArcticDB > Megadata > a vendor HTTP client (market data only) > OpenRouter
web search > mock**. The new tier, `providers/openrouter_search.py`, only
engages when no real feed above it is configured *and* an OpenRouter API key
is set (the same `EQR_MODEL_API_KEY`/`OPENROUTER_API_KEY` as §5a — no separate
toggle). It asks the model, with OpenRouter's `{"plugins": [{"id": "web"}]}`
search plugin turned on, to look up the plan's required fields online and
return only what it can cite a real URL for; anything it can't find is left
null rather than guessed. Its output goes through the same normalisation,
Evidence Store and QA path as every other provider — nothing downstream
treats it specially — and it is stamped `Confidence.MEDIUM` and
`metadata.acquired_via = "openrouter_web_search"` so it stays visibly
distinct from a licensed vendor feed in the evidence record. It is not
`is_mock`: it is real (if unverified) web data, so `contains_mock_data` is not
set from it. If the search call fails or finds nothing for a branch, that
branch falls back to the mock provider exactly as before, and the run is
labelled accordingly.

Trying this against NVDA with no other provider configured pulled real,
citable figures (e.g. share price and market cap from Yahoo Finance /
StockAnalysis.com, segment revenue and KPIs from NVIDIA's own FY2027 Q2 press
release and 10-Q on SEC EDGAR) — see `providers.registry` and
`providers.openrouter_search` log lines for which tier actually served a run.

## 6. Tests

This checkout does not currently include a `tests/` directory. Prior test
coverage exercised normalisation, analytics, the evidence store, QA, and an
end-to-end pipeline walk (plan → acquisition → evidence → analytics → agents
→ draft → QA → PDF, including provenance, concurrency timing, graceful
degradation under a failing provider, and QA-gated PDF suppression). Rebuild
that coverage before relying on this code for anything beyond local runs.

```bash
python -m ruff check --select F,E,W,I --line-length 100 eq_report
```

## 7. Example user request

```json
{
  "company": "NVIDIA",
  "ticker": "NVDA",
  "objective": "company update",
  "sections": ["company_snapshot", "key_takeaways", "recent_developments",
               "financials", "operating_drivers", "competitive_landscape",
               "valuation", "risks", "catalysts", "what_matters_next", "sources"],
  "time_horizon": "latest",
  "peers": [],
  "focus": ["data centre demand", "gross margin trajectory"],
  "report_date": "2026-09-02"
}
```

Only `company` is mandatory. The planner resolved `NVDA` from the name and
applied the default peer set `AMD, INTC, AVGO`, recording both as plan notes.
Each `focus` entry became a top-priority research question and a monitored item.

## 8. Example generated report

`python -m eq_report --ticker NVDA --report-date 2026-09-02` →
7-page PDF, 11 sections, 66 statements, 3 metric tables, 1 chart, 76 citations,
0 critical QA findings, 6 warnings. Page 1 is Key Takeaways beside the key-data
panel.

```
Key Takeaways                               9 statements   (+ key-data panel)
Company Snapshot                            4 statements  1 chart (price, 12 months)
Recent Developments                         8 statements
Latest Financial Performance                9 statements  1 table (results vs YoY vs consensus)
Operating Drivers and Segment Performance   9 statements  1 table (segment revenue, mix, growth)
Competitive Landscape                       6 statements  1 table (peers on multiple and growth)
Valuation and Market Expectations           6 statements
Risks                                       5 statements
Catalysts                                   4 statements
What Matters Next                           6 statements
Sources and Data Gaps                      76 citations
```

The same command for a ticker the providers hold little on shows the shape
adapting rather than printing empty headings: every section that the evidence
could not fill is dropped, and the report becomes a one-page statement of what
is missing and why, listed under "Sections not included".

Verbatim from the generated PDF (all figures synthetic — see §9). Reference
numbers are assigned in order of first use, so they shift if the request changes:

> **2. Company Snapshot** — NVIDIA trades at $187.42 for a $4.56tn market capitalisation, +19.2% year to date
>
> **1. Key Takeaways**
> - FY2026 Q2 revenue was $62.30bn, up 55% year on year and +12% sequentially. *(reported)*[1]
> - Data Center revenue was $54.20bn in FY2026 Q2, +60% year on year, representing 87% of total revenue. *(reported)*[2]
> - Gross margin moved -0.9pp year on year. *(calculated)*[3][4]
> - Revenue came in +4.2% versus consensus for the quarter. *(calculated)*[1][5]
> - The company is growing +39 percentage points faster than the peer average while trading at a +15% forward P/E premium, so the premium is currently underwritten by a growth differential rather than by multiple expansion alone. *(interpretation)*[9][10][11][7][12][13][14]
>
> **4. Latest Financial Performance** — FY2026 Q2 revenue $62.30bn, +55% year on year, 4.2% ahead of consensus, gross margin -0.9pp
> - Management commentary on margins: Gross margin declined sequentially and year on year as the new platform ramps; we expect margins to recover towards the high 70s as the ramp matures and yields improve. *(management)*[39]
>
> **7. Valuation and Market Expectations** — 31.4x forward earnings, +15% versus peers, +3% versus its own history
> - On 31.4x forward earnings against +55% revenue growth in the latest quarter, the multiple embeds continued high growth rather than a normalisation; a deceleration towards peer growth rates would be the main source of multiple risk. *(interpretation)*[7]
>
> **10. What Matters Next** — 8 checkable items before the FY2026 Q3 result
> - Whether FY2026 Q3 revenue lands at or above the guided $66.50bn, against consensus that sits +2.5% away. *(calculated)*[24]
> - Whether gross margin inflects: the observed trend across the periods held is slowing at -0.45pp per period, and the year-on-year change was -0.9pp. *(calculated)*[4][72][3]
> - On the requested emphasis 'gross margin trajectory': non-GAAP gross margin is expected to be 75.5% for the third quarter, plus or minus 50 basis points. *(management)*[73]
>
> **Sources**
> `[1] [sample data] Mock fundamentals feed (reported_financials), Revenue (FY2026 Q2)`
> `[15] [sample data] U.S. Securities and Exchange Commission (EDGAR) - Quarterly Report on Form 10-Q for the quarter ended July 31, 2026 - 2026-08-22, Item 1A - Risk Factors <https://example.invalid/edgar/nvda/10-q-fy2026-q2>`

Note the `(reported)` / `(calculated)` / `(management)` / `(market expectation)`
/ `(interpretation)` label on every line: fact, arithmetic, management assertion,
market expectation and inference stay separable on the page.

### Data freshness check (opt-in)

`EQR_CHECK_DATA_FRESHNESS=true` (default off) adds one stage right after the
Evidence Store is populated - before analysis or synthesis run at all - that
asks the model, with the same web-search plugin, what the most recent fiscal
period this company has actually publicly reported results for, as of the
report date (`pipeline/freshness_check.py`). This runs *before* gap research
specifically so a stale dataset is surfaced up front rather than only turning
up later, mixed into a general-purpose addendum. A confident mismatch prints
a distinct "DATA FRESHNESS NOTICE" banner on page 1 (its own colour, so it is
never mistaken for the illustrative-sample-data banner) and a
`temporal.dataset_stale` QA warning. It does not attempt to replace the
dataset - see the module docstring for why, and the freshness-vs-sample-data
tension noted under "Live web research for data gaps" above applies here too.

### Neutral wording and analytical discipline

This report states comparisons and lets the reader draw any investment
conclusion; it does not itself say whether a valuation is justified or a
stock is attractive. This is enforced at three layers, so no single point of
failure can let judgmental language through:

1. **Prompt-level.** The segment-agent and Key-Takeaways system prompts
   (`agents/llm_agent.py`, `synthesis/llm_synthesizer.py`) ban constructions
   like "supports its valuation premium" or "central risk", require the
   observation/conclusion distinction ("NVDA trades at a 15.4% premium and
   has higher growth" is supported; "the growth justifies the premium" is
   not), and require explicit "the data cannot distinguish between X and Y"
   framing when more than one explanation fits the evidence.
2. **Deterministic rewrite.** Every statement passes through
   `synthesis/terminology.py::soften_unsupported_causation` before it is
   finalised: an unsupported causal claim ("driven by", "due to", "thanks
   to", "on the back of", "as a result of") is rewritten to assert
   coincidence rather than causation, unless the claim type already justifies
   it (a management statement, a reported fact, or a calculation that
   isolates the driver). Markers that can introduce a full clause ("because",
   "led to", "caused") are not auto-rewritten, since the swap can break the
   sentence's grammar there - those stay behind the QA warning below.
3. **QA backstop.** `check_causal_claims` (existing) flags any remaining
   unsupported causal marker; `check_judgmental_language` (new) flags
   investment-judgment language directly. Neither blocks the PDF (both are
   WARNING severity) since wording style is not a provenance failure, but
   both are visible in the QA trail printed in the back matter.

See `docs/PENDING_CHANGES.md` for the larger neutral-analysis redesign this
is part of, including what is deferred.

### Live web research for data gaps (opt-in)

`EQR_RESEARCH_DATA_GAPS=true` (default off - see `.env.example`) adds one stage
between synthesis and QA: for each of the report's disclosed data gaps, up to
`EQR_RESEARCH_DATA_GAPS_MAX` (default 8), the model is asked - with
OpenRouter's web-search plugin turned on - to find a real, dated, source-linked
answer (`pipeline/gap_research.py`). A gap is only accepted if the model
returns an actual URL and a factual answer; anything without one is treated as
"not found" rather than trusted, the same rule the rest of the pipeline
applies to any claim with no resolvable evidence id. Accepted answers are
written to the Evidence Store as ordinary, non-mock evidence and surfaced as
their own "Additional Research (Web-Verified)" section, with the now-filled
gaps struck from the disclosed gap list - see `apply_gap_research`.

Because this is a genuinely live search against the real web, an accepted
answer will report NVIDIA's actual reported figures, which do **not** match
the illustrative sample data the rest of the report is built on (the sample
data does not correspond to any real fiscal period). This is expected: it is
what a live source is supposed to return. It is also why this only makes
sense as a demonstration of the mechanism, not a way to reconcile a
mock-data report with reality - a real deployment (real providers) would not
have this seam.

### Annotated companion PDF

Every run also produces a second PDF (`..._annotated.pdf`, same run
directory) built from the same story-assembly code as the primary report
(`PdfReportRenderer.render_annotated`, `synthesis/annotate.py`). Under every
paragraph and bullet it prints one short line - "conveys: ..." - naming the
single piece of information or conclusion the sentence exists to give the
reader, so a sentence that has drifted into restating a number or describing
itself rather than saying something is easy to spot. It costs one extra model
call for the whole report (batched, not per-sentence) and falls back to a
cheap heuristic label when no model is configured, so the companion PDF is
still produced either way. It never feeds back into the primary report or
into QA.

### Degraded run

`python -m eq_report --ticker ZZZZ --report-date 2026-09-02` — no sample
data exists for that ticker. The pipeline completes, the PDF is produced, and it
contains 15 recorded data gaps ("The Evidence Store contains no reported revenue
for any period. Impact: The financial performance section cannot be written.")
and **zero invented numbers**. There is a test asserting that every statement in
that run still carries evidence or analytics references.

## 9. Mocked components — clearly identified

| Component | Status | Notes |
|---|---|---|
| `providers/sample_data.py` | **entirely synthetic** | Every figure in the example report above. NVIDIA-shaped in scale and structure so real code paths are exercised; **not real reported figures**. Header says so. |
| `MockMarketDataProvider` | **mock** | `is_mock=True`. Quote, multiples, consensus, estimates, price history, forward-P/E history, peer data. |
| `MockFundamentalsProvider` | **mock** | `is_mock=True`. Three fiscal periods, segment revenue, KPIs, consensus, guidance. |
| `MockDocumentsProvider` | **mock** | `is_mock=True`. 10 documents / 21 passages: earnings release, 10-Q, transcript, deck, two announcements, two news items, a competitor filing, industry research. URLs are `example.invalid`. |
| `HttpMarketDataProvider` | **real, unexercised** | Genuine REST client. Disabled without `EQR_MARKET_DATA_API_KEY` + `EQR_MARKET_DATA_BASE_URL`; the registry then uses the mock. Tested for the *skip* path only — never run against a live endpoint. |
| `providers/megadata.py` | **real, network-dependent** | Real HTTP provider set, enabled by `EQR_MEGADATA_BASE_URL`. Falls back to mock if the endpoint is unreachable within `EQR_PROVIDER_TIMEOUT_SECONDS`. |
| `providers/arcticdb/` | **real, network-dependent** | Real provider set backed by ArcticDB, enabled by `EQR_ARCTICDB_URI`; checked first, ahead of Megadata. `fiscal_period.py` can optionally call the OpenRouter client to resolve fiscal-period labels. |
| `ResearchPlanner` ticker resolution | **13-entry lookup** | Not a security master. An unresolved name plans without a ticker and records it. |
| `EvidenceReader.documents_matching` | **substring keyword match** | Deliberately transparent. The natural place for embeddings later; no agent would change. |
| LLM usage | **optional, implemented, off by default** | `ModelConfig` is used: `ResearchPlanner` plans via GPT through OpenRouter whenever `EQR_MODEL_API_KEY` is set; `LLMSegmentAgent` and `LLMSynthesizer` additionally replace their deterministic counterparts under `EQR_MODEL_USE_FOR_AGENTS`/`EQR_MODEL_USE_FOR_SYNTHESIS`. All three are constrained to cite only ids they were actually shown. With no model variables set, behaviour is unchanged from a fully deterministic run. |

The mock label is load-bearing: it reaches the front page, every page footer,
every source line, and a QA warning. `contains_mock_data` is asserted in the
end-to-end test.

## 10. Next components to productionise

In the order I would tackle them.

1. **Real fundamentals** — SEC EDGAR XBRL company-facts for reported financials,
   with a filing-level cache. The biggest credibility gap: everything numeric in
   the report currently comes from `sample_data.py`.
2. **Real market data** — finish `HttpMarketDataProvider` against a chosen
   vendor. Only `_parse` needs writing; normalisation and everything downstream
   are already vendor-agnostic. Add response caching and rate limiting.
3. **Real documents** — EDGAR full-text search plus a PDF/HTML extractor that
   preserves section and page. `RawDocumentPassage` already carries the fields.
4. **Consensus** — the one input with no free source. Until it exists, every
   surprise and guidance-versus-consensus number is synthetic; the engine
   already degrades to a documented gap without it.
5. **Ticker and entity resolution** — replace the lookup dict with a security
   master, and add fiscal-calendar metadata per issuer. The prototype's fiscal
   convention is internally consistent but assumed, not looked up.
6. ~~**LLM-backed narrative agents**~~ — done: `LLMSegmentAgent` and
   `LLMSynthesizer`, gated by `EQR_MODEL_USE_FOR_AGENTS`/`_SYNTHESIS`, constrained
   to evidence/analytics ids they were actually shown. Still open: automated
   regression coverage now that `tests/` has been removed from this checkout
   (see §6), and a token-cost/latency budget per run now that real OpenRouter
   calls are in the critical path.
7. **Document retrieval** — swap keyword matching for embeddings once the corpus
   is real. Contained entirely within `EvidenceReader`.
8. **QA hardening** — a units/dimensional-analysis pass over composed metrics,
   cross-source reconciliation rules, and a claim-to-source entailment check for
   quoted passages.
9. **Report design** — the layout is deliberately plain per the brief. A real
   version needs a house template, a proper chart library, and an exhibit system.
10. **Evidence Store scale-up** — SQLite is right for one local run. Multiple
    concurrent runs, evidence reuse across runs and retention policy need
    Postgres and a migration path; `EvidenceQuery` is the seam.
11. **Operational surface** — a job queue for report runs, run history and diffs
    between two runs of the same company, and alerting on QA-blocked runs.

Explicitly *not* built, per the brief: frontend, microservices, Kubernetes,
vector database, auth, scheduling, deployment infrastructure.
