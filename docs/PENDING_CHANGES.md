# Pending changes: neutral-analysis report redesign

This tracks a large change list the user gave in one message, organised by
what shipped in this pass versus what is deferred. The user's own priority
order was: **(1) fix freshness/source reconciliation, (2) remove
cross-section repetition, (3) enforce neutral evidence-based wording** -
everything else is refinement after those three. All three shipped; see
below for exactly what and how, then the deferred list.

## Shipped this pass

**1. Data freshness / source reconciliation** - `pipeline/freshness_check.py`
(new, opt-in via `EQR_CHECK_DATA_FRESHNESS`). Runs early, right after the
Evidence Store is populated and before analysis/synthesis, so a stale dataset
is checked and disclosed *before* the narrative is written rather than
discovered later via `pipeline.gap_research`'s addendum. Compares
`EvidenceReader.latest_reported_period()` against a live, source-linked
web-search answer; on a confident mismatch it prints a distinct "DATA
FRESHNESS NOTICE" banner on page 1 (separate colour from the mock-data
banner) and a `temporal.dataset_stale` QA warning. It deliberately does
**not** attempt to replace the dataset with the newer figures - see the
module docstring for why that is a much larger change, and does not make
sense against illustrative sample data in the first place, since it does not
correspond to any real fiscal period.

**2. Cross-section repetition** - `synthesis/synthesizer.py::_statements` now
catches a repeat two independent ways: near-identical wording (existing), or
a high overlap in the underlying evidence/analytics ids two statements cite
(new - `synthesis/terminology.py::id_overlap`). The second catches two
segment agents independently restating the same fact in different words
(different number formatting alone, e.g. "$62.3 billion" vs "$62.3bn", already
defeats a wording-only comparison), which is the actual repetition pattern the
user described (revenue beat / EPS beat / Q3 guidance / Data Center
concentration / gross-margin recovery recurring across sections). A finding
that pulls in materially different evidence lowers its own overlap ratio and
still gets through, which is what lets a genuinely new angle on a
previously-used fact survive.

**3. Neutral, evidence-based wording** - several pieces:
- Rewrote the segment-agent (`agents/llm_agent.py`) and Key-Takeaways
  (`synthesis/llm_synthesizer.py`) system prompts: ban investment-judgment
  language with the user's own banned/preferred phrasings, require the
  observation/conclusion distinction with the user's own example, require
  explicit "the data cannot distinguish between X and Y" framing, and prefer
  "during the reported period" over trend language when only one period is
  held.
- New deterministic QA backstop, `qa/checks.py::check_judgmental_language`
  (`narrative.judgmental_language`), since a prompt is not a guarantee - it
  flags constructions like "supports its valuation premium", "justifies the
  premium", "attractive/unattractive", "should buy/sell", etc.
- New deterministic causal-language softening,
  `synthesis/terminology.py::soften_unsupported_causation`, applied to every
  statement before it is finalised: for the causal markers that always
  precede a noun phrase ("driven by", "due to", "thanks to", "on the back of",
  "as a result of"), an unsupported claim (not a management statement,
  reported fact, or isolating calculation) gets the marker swapped for
  "alongside" - asserting coincidence, not causation. The markers that can
  also introduce a full clause ("because", "led to", "caused", "drove",
  "resulted in") are deliberately **not** auto-rewritten, since the same swap
  can break the sentence's grammar there; those are left to the existing
  `check_causal_claims` QA warning, per the user's "keep this safeguard"
  instruction.
- Renamed the three most investment-toned section titles (display title
  only, not the internal `ReportSection` id): "Risks" -> "Uncertainties and
  Analytical Limitations", "Catalysts" -> "Upcoming Events and Monitoring
  Indicators", "What Matters Next" -> "Forward Monitoring Framework". Touched
  up a few investment-toned phrases in the deterministic fallback agent,
  `agents/risks_catalysts.py` ("near-term catalyst" -> "the next scheduled
  test of that expectation", etc.).

**Companion change: data-gap prioritisation and a compact table.** Not one of
the top three, but cheap and directly requested: `domain/segment.py::DataGap`
gained a `priority: "high"|"medium"|"low"` field (LLM-assigned via a new
schema field in `agents/llm_agent.py`, defaulting to "medium" when absent or
invalid); the renderer's Data Gaps disclosure
(`rendering/pdf_renderer.py::_gap_box`) is now a compact three-column table
(Priority | Missing data | Why it matters), sorted high-to-low, instead of a
wall of text paragraphs.

## Deferred - not attempted this pass

These are real, well-specified asks that did not make this pass, either
because they require a substantially larger architectural change than could
be responsibly built *and verified* alongside everything above in one
session, or because they depend on data the mock providers do not carry.
Listed in roughly the order the original message raised them.

- **A proper canonical-fact reconciliation stage** ("collect -> check latest
  period -> verify primary sources -> resolve conflicts -> one canonical
  dataset -> calculate -> write"). What shipped is a *check and disclose*
  stage for the single most important fact (the latest period), not a general
  multi-source conflict-resolution engine across every metric. Building the
  latter means every metric needs a source-quality/precedence rule and a
  merge step between normalisation and the Evidence Store - a substantially
  larger change than a single verification call.
- **Full section reorder to the suggested structure** (Executive Summary ->
  Results vs Prior Period/Consensus -> Financial Analysis -> Segment Analysis
  -> Operating Drivers -> Competitive Comparison -> Valuation Context ->
  Forward Monitoring -> Uncertainties/Data Gaps -> Sources). The three
  renamed sections keep their existing positions in `DEFAULT_SECTIONS`; a
  full reorder plus "one distinct analytical purpose per section" would mean
  rewriting the scope of what each of the eight segment-agent prompts is
  *allowed* to say (right now most receive the same broad evidence pool),
  which is a meaningfully larger prompt-engineering task warranting its own
  pass with real-model verification.
- **"Forward Monitoring" as a structured table** (Metric | Current value |
  Previous/reference value | Change | Why it's relevant). The section's
  underlying agent (`agents/what_matters_next.py`) currently emits free-text
  `KeyFinding`s; turning it into a table means giving `SegmentResult`/
  `KeyFinding` a structured alternative shape, which cascades into the QA
  checks that assume `statement.text` is prose. Left as free text with the
  cross-section dedup and neutral-wording changes still applying to it.
- **Competitive Landscape rework** to compare named metrics across
  competitors (margins, market share, product categories) rather than mostly
  NVIDIA-only analysis with peer multiples appended. The peer *table*
  (`synthesis/exhibits.py`) already compares forward P/E and revenue growth
  across NVDA/AMD/AVGO/INTC; broader coverage (margins, market share) is
  gated on the mock/real providers actually supplying comparable peer
  fundamentals, which they do not yet.
- **Actively research decision-relevant gaps beyond period freshness**
  (competitor performance, hyperscaler capex, HBM supply, packaging capacity,
  ASPs, market share, custom ASICs). `pipeline/gap_research.py` already does
  exactly this mechanism (opt-in web search per gap, real-URL-or-nothing) for
  *any* disclosed gap, up to `EQR_RESEARCH_DATA_GAPS_MAX` of them - it is not
  restricted to the categories named above, so no further work was needed to
  make this "automatic"; it already runs on whichever gaps the segment
  agents actually raised.
- **Record what was searched for each gap** (missing evidence -> why it
  matters -> sources checked -> found/not found -> resulting limitation).
  `pipeline/gap_research.py` currently records only the *accepted* answers as
  evidence; it does not yet log the attempts that came back empty. Would need
  a small new field on the run manifest.
- **Improve estimate-revision granularity** (30-day/90-day, revenue/EPS/margin
  breakdowns). Gated on the mock/real providers supplying that granularity;
  the current single aggregate revision-direction figure is what they hold.
- **Move the full source list to an appendix / machine-readable manifest,
  and summarise QA warnings to unresolved high-severity only in the
  reader-facing body.** Both sources and QA warnings already live in the back
  matter, not the analytical body - but the user's ask was to compress them
  further (a citation appendix separate from the main PDF, and a QA summary
  rather than a full list). Not attempted this pass; would shorten the report
  by 1-2 pages as a side effect, addressing the "8 pages could be 4-6" note.
- **Aim to shorten the reader-facing body to ~4-6 pages.** A likely
  consequence of the repetition fix plus moving sources/QA further out, but
  not independently pursued or measured this pass.
