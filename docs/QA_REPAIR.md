# QA authority and automatic repair

One report request now contains a bounded repair loop:

`draft -> deterministic QA -> repair -> deterministic QA -> render or hard block`

The default is two repair attempts (`EQR_QA_AUTO_REPAIR_MAX_ATTEMPTS=2`).
Each run writes `07_qa_repair.json`, including every edited or omitted
statement and whether the action came from the LLM or deterministic fallback.

## Deterministic authority

Code, not the LLM, controls all publication-critical decisions:

- evidence/reference existence and citation numbering;
- canonical and validated evidence status;
- arithmetic recomputation, accounting identities, units and percentages;
- entity, ticker, currency, period and cross-source consistency;
- whether an LLM rewrite still contains an unsupported number;
- the final QA pass/fail verdict and PDF publication gate;
- omission of a localized unsafe statement when no safe rewrite is available.

These checks rerun after every repair. The model cannot suppress, downgrade or
override them.

## LLM-assisted work

The model is used for work requiring language or source interpretation:

- planning and research decomposition;
- classification of unknown provider metric labels, constrained to the known
  canonical vocabulary;
- segment analysis and narrative synthesis;
- prose repair for a statement that uses a number without canonical support;
- classification of whether conflicting rows represent the same fact or
  different definitions;
- optional web verification of same-fact conflicts.

During QA repair, the LLM can only return rewritten text or request omission.
It cannot add evidence ids, analytics ids, citations or canonical values.

## What happens when the LLM cannot repair a finding

A statement-scoped narrative failure is omitted and recorded as
`deterministic_omit`. This produces a smaller but safe report instead of asking
the user to rerun repeatedly. A non-local invariant failure—such as corrupt
analytics, mixed company identity, contradictory primary facts with no verified
winner, or a malformed evidence store—remains a hard block. Such a condition
cannot safely be converted into a publishable fact by language-model judgment.
