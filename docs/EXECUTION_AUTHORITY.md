# Deterministic and LLM execution authority

`EQR_MODEL_API_KEY` is the master model switch. With a key, the pipeline uses
the LLM wherever listed below; without one, the deterministic/fallback column
applies. Additional paid web-search stages retain separate opt-in flags.

| Stage | LLM role | Deterministic role / authority |
|---|---|---|
| Planning | Primary planner when a model key exists | Rule-based plan when no model is configured |
| Acquisition | None | MegadataAPI is the sole provider; failures remain failures |
| Normalisation | Classifies unknown provider metric labels into the fixed canonical vocabulary | Parses values, units, currencies, dates and known metrics; rejects invalid values; verifies model-proposed numbers |
| Analytics | Recomputes results as a cross-check | Calculates every published value and overrides any LLM disagreement |
| Segment agents | Primary narrative analysis | Rule-based segment agent runs when the model is unavailable or fails; evidence/analytics IDs are validated |
| Synthesis | Rewrites and orders supported findings | Typed draft construction, citation registration, exhibits and deterministic fallback |
| QA | Conflict-definition classification | Every blocking check and the final publication verdict |
| QA repair | Proposes prose-only rewrites | Rejects rewrites containing unsupported numbers, reruns QA and omits unsafe statements |
| Annotation | Produces editorial companion labels | Heuristic labels if the model fails; never changes the report |
| Rendering | None | JSON and PDF output |

## Separate opt-ins

| Setting | Default | LLM action | Deterministic backstop |
|---|---:|---|---|
| `EQR_CHECK_DATA_FRESHNESS` | `false` | Web-searches for the latest reported fiscal period | Compares canonical periods; only emits a notice/warning and never replaces MegaAPI data |
| `EQR_VERIFY_METRIC_CONFLICTS` | `false` | Classifies definitions and web-verifies genuine same-fact conflicts | Detects the conflict, requires a dated URL matching a candidate, and retains the block if verification fails |
| `EQR_QA_AUTO_REPAIR` | `true` | Rewrites statement-scoped unsupported prose | Deterministically validates the rewrite; omits the unsafe statement when necessary; cannot alter evidence or the verdict |
| `EQR_WEB_FILL_GAPS` | `false` | Proposes facts for a thin section via web search | A second, independent web-search call must confirm each fact - and, for a source off the allow-list/company-domain fast track, confirm the source's own legitimacy too; a third re-confirms it again during QA, immediately before publication (`qa/web_claim_auditor.py`) - a fact unconfirmed at any check is dropped, never trusted |

The Analytics LLM pass is the clearest example of “LLM enabled but
deterministic result”: code calculates the number first, the model only checks
it, and code wins on every disagreement. Normalisation and QA repair follow the
same proposal-then-validation pattern.
