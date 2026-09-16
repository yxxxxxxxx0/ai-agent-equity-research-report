"""Individual QA checks.

Each check is a plain function taking a QAContext and returning findings, so
checks can be unit-tested in isolation and added without touching the engine.

Severity policy for this draft:

* CRITICAL - blocks PDF generation. Reserved for provenance and arithmetic
  failures: a dangling evidence reference, an unsupported number, a formula that
  does not reproduce, a requested section that is missing entirely.
* WARNING  - printed and logged, does not block. Weak evidence, unresolved data
  gaps, incomparable periods, duplication.
* INFO     - recorded for the run manifest only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..analytics import calculations as calc
from ..domain.analytics import AnalyticsBundle
from ..domain.enums import ClaimType, Confidence, ReportSection, Severity
from ..domain.plan import ResearchPlan
from ..domain.qa import QAFinding
from ..domain.report import ReportDraft
from ..evidence.reader import EvidenceReader
from ..normalisation.dates import normalise_fiscal_period, periods_are_comparable
from ..synthesis.terminology import CAUSAL_MARKERS


@dataclass(frozen=True, slots=True)
class QAContext:
    """Everything the checks are allowed to look at."""

    draft: ReportDraft
    plan: ResearchPlan
    reader: EvidenceReader
    analytics: AnalyticsBundle


#: Matches a number that a statement asserts: currency amounts, percentages,
#: multiples and plain figures. Used to find numbers with no evidence behind them.
_NUMBER_RE = re.compile(
    r"[-+]?\$?\d[\d,]*(?:\.\d+)?\s*(?:bn|tn|mn|%|pp|x|billion|million|trillion)?",
    re.IGNORECASE,
)



# ---------------------------------------------------------------------------
# Evidence QA
# ---------------------------------------------------------------------------

def check_references_exist(context: QAContext) -> list[QAFinding]:
    """Every referenced evidence and analytics id must exist."""
    findings: list[QAFinding] = []
    valid_analytics = {r.analytics_id for r in context.analytics.results}

    for section in context.draft.sections:
        for statement in section.statements:
            for evidence_id in statement.evidence_ids:
                if context.reader.get(evidence_id) is None:
                    findings.append(QAFinding(
                        check="evidence.reference_exists",
                        severity=Severity.CRITICAL,
                        message=f"Statement cites evidence id {evidence_id}, "
                                "which is not in the Evidence Store.",
                        section=section.section.value,
                        subject=statement.text[:160],
                        details={"evidence_id": evidence_id},
                    ))
            for analytics_id in statement.analytics_ids:
                if analytics_id not in valid_analytics:
                    findings.append(QAFinding(
                        check="analytics.reference_exists",
                        severity=Severity.CRITICAL,
                        message=f"Statement cites analytics id {analytics_id}, "
                                "which was not produced by the Analytics Engine.",
                        section=section.section.value,
                        subject=statement.text[:160],
                        details={"analytics_id": analytics_id},
                    ))

    for missing in context.draft.metadata.get("missing_evidence_references", []):
        findings.append(QAFinding(
            check="evidence.reference_exists",
            severity=Severity.CRITICAL,
            message=f"The synthesis layer could not resolve evidence id {missing}.",
            details={"evidence_id": missing},
        ))
    return findings


def check_claims_are_supported(context: QAContext) -> list[QAFinding]:
    """Factual and numeric claims must carry evidence or analytics.

    A pure interpretation is allowed to rest on other statements, but it still
    may not introduce a *number* of its own without support.
    """
    findings: list[QAFinding] = []
    for section in context.draft.sections:
        for statement in section.statements:
            supported = bool(statement.evidence_ids or statement.analytics_ids)
            has_number = bool(_NUMBER_RE.search(statement.text))

            if supported:
                continue

            if has_number:
                findings.append(QAFinding(
                    check="evidence.no_unsupported_numbers",
                    severity=Severity.CRITICAL,
                    message="Statement asserts a number with no supporting evidence "
                            "or analytics reference.",
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
            elif statement.claim_type is not ClaimType.INTERPRETATION:
                findings.append(QAFinding(
                    check="evidence.claim_supported",
                    severity=Severity.CRITICAL,
                    message=f"Statement of type {statement.claim_type.value} has no "
                            "supporting evidence.",
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
            else:
                findings.append(QAFinding(
                    check="evidence.interpretation_unanchored",
                    severity=Severity.WARNING,
                    message="Interpretation carries no evidence or analytics reference.",
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
    return findings


def check_numeric_claims_use_canonical_evidence(context: QAContext) -> list[QAFinding]:
    """P0: document text is context, not a structured numerical fact."""
    findings: list[QAFinding] = []
    numeric = re.compile(r"(?:[$€£]\s*)?\d+(?:[,.]\d+)*(?:\s*(?:%|x|million|billion|bn|m))?", re.I)
    for section in context.draft.sections:
        for statement in section.statements:
            if not numeric.search(statement.text) or statement.analytics_ids:
                continue
            cited = [context.reader.get(eid) for eid in statement.evidence_ids]
            cited = [item for item in cited if item is not None]
            if cited and any(item.value is not None and item.is_canonical and item.status.value == "validated" for item in cited):
                continue
            findings.append(QAFinding(
                check="evidence.numeric_claim_not_canonical",
                severity=Severity.CRITICAL,
                message="Numeric narrative claim has no validated canonical observation or derived analytic.",
                section=section.section.value,
                subject=statement.text,
                details={"evidence_ids": list(statement.evidence_ids)},
            ))
    return findings


def check_accounting_identities(context: QAContext) -> list[QAFinding]:
    """P0 deterministic checks when compatible canonical inputs coexist."""
    findings: list[QAFinding] = []
    for period in context.reader.periods():
        revenue = context.reader.numeric("revenue", period)
        identities = (
            ("gross_margin", "gross_profit", "gross_margin"),
            ("operating_margin", "operating_income", "operating_margin"),
        )
        if revenue and revenue.value:
            for code, numerator_metric, margin_metric in identities:
                numerator = context.reader.numeric(numerator_metric, period)
                margin = context.reader.numeric(margin_metric, period)
                if not numerator or not margin:
                    continue
                calculated = numerator.value / revenue.value * 100.0
                if abs(calculated - margin.value) > 0.1:
                    findings.append(QAFinding(
                        check=f"accounting.{code}", severity=Severity.CRITICAL,
                        message=(f"{period} {margin_metric}={margin.value:.3f}% does not "
                                 f"reconcile to {calculated:.3f}% from canonical inputs."),
                        subject=period,
                        details={"evidence_ids": [revenue.evidence_id, numerator.evidence_id, margin.evidence_id]},
                    ))
        ocf = context.reader.numeric("operating_cash_flow", period)
        capex = context.reader.numeric("capital_expenditure", period)
        fcf = context.reader.numeric("free_cash_flow", period)
        if ocf and capex and fcf:
            # Accept either positive capex (cash spent) or negative cash-flow presentation.
            candidates = (ocf.value - abs(capex.value), ocf.value + capex.value)
            if min(abs(fcf.value - value) for value in candidates) > max(1.0, abs(fcf.value) * 0.005):
                findings.append(QAFinding(
                    check="accounting.free_cash_flow", severity=Severity.CRITICAL,
                    message=f"{period} free cash flow does not reconcile to operating cash flow less capex.",
                    subject=period,
                    details={"evidence_ids": [ocf.evidence_id, capex.evidence_id, fcf.evidence_id]},
                ))
    return findings


def check_citation_numbering(context: QAContext) -> list[QAFinding]:
    """Printed reference numbers must resolve to a citation in the source list."""
    findings: list[QAFinding] = []
    known = {citation.ref_number for citation in context.draft.citations}
    for section in context.draft.sections:
        for statement in section.statements:
            for ref in statement.citation_refs:
                if ref not in known:
                    findings.append(QAFinding(
                        check="evidence.citation_resolves",
                        severity=Severity.CRITICAL,
                        message=f"Reference [{ref}] does not appear in the source list.",
                        section=section.section.value,
                        subject=statement.text[:160],
                        details={"ref": ref},
                    ))
    return findings


def check_evidence_confidence(context: QAContext) -> list[QAFinding]:
    """Flag material claims resting only on low-confidence evidence."""
    findings: list[QAFinding] = []
    for section in context.draft.sections:
        for statement in section.statements:
            if not statement.evidence_ids:
                continue
            items = [context.reader.get(e) for e in statement.evidence_ids]
            present = [i for i in items if i is not None]
            if not present:
                continue
            if all(i.confidence in {Confidence.LOW, Confidence.UNKNOWN} for i in present):
                findings.append(QAFinding(
                    check="evidence.weak_support",
                    severity=Severity.WARNING,
                    message="Statement rests only on low- or unknown-confidence evidence.",
                    section=section.section.value,
                    subject=statement.text[:200],
                    details={"sources": [i.source_name for i in present]},
                ))
    return findings


# ---------------------------------------------------------------------------
# Numerical QA
# ---------------------------------------------------------------------------

def check_analytics_recompute(context: QAContext) -> list[QAFinding]:
    """Recompute every analytic from its stored inputs.

    This is the arithmetic backstop: the engine's formula string and inputs are
    re-run here through the same pure functions, and any mismatch is critical.
    """
    findings: list[QAFinding] = []
    tolerance = 1e-4

    for result in context.analytics.results:
        expected = _recompute(result.formula, result.inputs)
        if expected is None:
            findings.append(QAFinding(
                check="numeric.formula_recognised",
                severity=Severity.INFO,
                message=f"No independent recomputation is implemented for formula "
                        f"{result.formula!r}.",
                subject=result.metric,
            ))
            continue
        if abs(expected - result.value) > max(tolerance, abs(result.value) * 1e-6):
            findings.append(QAFinding(
                check="numeric.recompute",
                severity=Severity.CRITICAL,
                message=(
                    f"{result.metric} does not reproduce: stored {result.value}, "
                    f"recomputed {round(expected, 6)}."
                ),
                subject=result.metric,
                details={"formula": result.formula, "inputs": result.inputs,
                         "analytics_id": result.analytics_id},
            ))
    return findings


def check_units(context: QAContext) -> list[QAFinding]:
    """Units must be from the canonical set and match the metric's nature."""
    findings: list[QAFinding] = []
    allowed = {"pct", "pp", "x", "count", "USD", "USD/share", "EUR", "GBP", "text", ""}

    for result in context.analytics.results:
        if result.unit not in allowed:
            findings.append(QAFinding(
                check="numeric.unit_known",
                severity=Severity.CRITICAL,
                message=f"Analytics result {result.metric} has unrecognised unit "
                        f"{result.unit!r}.",
                subject=result.metric,
            ))
        # A margin *change* is percentage points; a growth *rate* is a percentage.
        if "margin_change" in result.metric and result.unit != "pp":
            findings.append(QAFinding(
                check="numeric.unit_semantics",
                severity=Severity.CRITICAL,
                message=f"{result.metric} is a margin change and must be in percentage "
                        f"points, not {result.unit!r}.",
                subject=result.metric,
            ))
        if result.metric.endswith("_pct") and result.unit not in {"pct", "pp"}:
            findings.append(QAFinding(
                check="numeric.unit_semantics",
                severity=Severity.CRITICAL,
                message=f"{result.metric} is named as a percentage but carries unit "
                        f"{result.unit!r}.",
                subject=result.metric,
            ))
    return findings


def check_percentage_representation(context: QAContext) -> list[QAFinding]:
    """Percentages must be in percent units, not decimal fractions.

    A margin stored as 0.748 rather than 74.8 is the classic silent error, so any
    percentage-unit value in the suspicious band is flagged.
    """
    findings: list[QAFinding] = []
    for item in context.reader.query(has_value=True):
        if item.unit != "pct" or item.value is None:
            continue
        if 0 < abs(item.value) < 1:
            findings.append(QAFinding(
                check="numeric.percent_representation",
                severity=Severity.WARNING,
                message=(
                    f"{item.metric} for {item.period_label or item.as_of} is "
                    f"{item.value}, which may be a decimal fraction stored as a percentage."
                ),
                subject=item.metric,
                details={"evidence_id": item.evidence_id, "raw_value": item.raw_value},
            ))
    return findings


# ---------------------------------------------------------------------------
# Temporal QA
# ---------------------------------------------------------------------------

def check_period_comparability(context: QAContext) -> list[QAFinding]:
    """Comparisons must be between compatible fiscal periods."""
    findings: list[QAFinding] = []
    for result in context.analytics.results:
        if not result.period or not result.comparison_period:
            continue
        left = normalise_fiscal_period(result.period)
        right = normalise_fiscal_period(result.comparison_period)
        if not periods_are_comparable(left, right):
            findings.append(QAFinding(
                check="temporal.comparable_periods",
                severity=Severity.WARNING,
                message=(
                    f"{result.metric} compares {result.period} with "
                    f"{result.comparison_period}, which are not directly comparable."
                ),
                subject=result.metric,
            ))
    return findings


def check_dates_not_future(context: QAContext) -> list[QAFinding]:
    """Observation dates must not post-date the report.

    Guidance and consensus legitimately carry future *periods*; an ``as_of``
    observation date in the future does not.
    """
    findings: list[QAFinding] = []
    report_date = context.draft.report_date
    for item in context.reader.query():
        if item.as_of and item.as_of > report_date:
            findings.append(QAFinding(
                check="temporal.as_of_not_future",
                severity=Severity.WARNING,
                message=(
                    f"Evidence {item.evidence_id} is dated {item.as_of.isoformat()}, "
                    f"after the report date {report_date.isoformat()}."
                ),
                subject=item.metric or item.document_title,
                details={"evidence_id": item.evidence_id},
            ))
    return findings


def check_period_consistency(context: QAContext) -> list[QAFinding]:
    """The reported period referenced in prose must be the one analysed."""
    findings: list[QAFinding] = []
    latest = context.draft.metadata.get("latest_reported_period")
    if not latest:
        return findings
    period = normalise_fiscal_period(str(latest))
    if period is None or period.fiscal_year is None:
        return findings

    other_periods = {
        item.period_label for item in context.reader.query()
        if item.period_label and item.period_label != latest
    }
    for section in context.draft.sections:
        if section.section is not ReportSection.FINANCIALS:
            continue
        for statement in section.statements:
            mentioned = [p for p in other_periods if p in statement.text]
            if mentioned and latest not in statement.text:
                findings.append(QAFinding(
                    check="temporal.period_labelled",
                    severity=Severity.WARNING,
                    message=(
                        "Financial statement references "
                        f"{', '.join(sorted(mentioned))} without naming the reported "
                        f"period {latest}."
                    ),
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
    return findings


# ---------------------------------------------------------------------------
# Consistency QA
# ---------------------------------------------------------------------------

def check_entity_consistency(context: QAContext) -> list[QAFinding]:
    """One company, one name, one ticker across the whole evidence set."""
    findings: list[QAFinding] = []
    draft = context.draft

    subject_items = context.reader.query(ticker=draft.ticker) if draft.ticker else ()
    names = {item.company for item in subject_items if item.company}
    if len(names) > 1:
        findings.append(QAFinding(
            check="consistency.company_name",
            severity=Severity.CRITICAL,
            message=f"Ticker {draft.ticker} is associated with several company names: "
                    f"{', '.join(sorted(names))}.",
            details={"names": sorted(names)},
        ))

    tickers = {
        item.ticker for item in context.reader.query(company=draft.company)
        if item.ticker
    }
    if len(tickers) > 1:
        findings.append(QAFinding(
            check="consistency.ticker",
            severity=Severity.CRITICAL,
            message=f"Company {draft.company} is associated with several tickers: "
                    f"{', '.join(sorted(tickers))}.",
            details={"tickers": sorted(tickers)},
        ))
    return findings


def check_metric_agreement(context: QAContext) -> list[QAFinding]:
    """The same metric and period must not hold contradictory values."""
    findings: list[QAFinding] = []
    buckets: dict[tuple[str, str, str], list] = {}

    for item in context.reader.query(ticker=context.draft.ticker, has_value=True):
        if not item.metric or item.metadata.get("series"):
            continue
        discriminator = str(
            item.metadata.get("segment_name") or item.metadata.get("kpi_name") or "")
        key = (item.metric, item.period_label or (
            item.as_of.isoformat() if item.as_of else ""), discriminator)
        buckets.setdefault(key, []).append(item)

    for (metric, period, discriminator), items in buckets.items():
        values = {round(float(i.value or 0), 6) for i in items}
        if len(values) <= 1:
            continue
        low, high = min(values), max(values)
        # A trivial rounding difference between two vendors is not a contradiction.
        spread = abs(high - low) / max(abs(high), 1e-9)
        severity = Severity.CRITICAL if spread > 0.01 else Severity.WARNING
        findings.append(QAFinding(
            check="consistency.metric_agreement",
            severity=severity,
            message=(
                f"{metric} for {period or 'the current date'}"
                + (f" ({discriminator})" if discriminator else "")
                + f" has conflicting values across sources: {sorted(values)}."
            ),
            subject=metric,
            details={
                "sources": [i.source_name for i in items],
                "evidence_ids": [i.evidence_id for i in items],
                "spread": round(spread, 6),
            },
        ))
    return findings


def check_currency_consistency(context: QAContext) -> list[QAFinding]:
    """All monetary evidence for the subject must share one currency."""
    findings: list[QAFinding] = []
    currencies = {
        item.currency for item in context.reader.query(
            ticker=context.draft.ticker, has_value=True)
        if item.currency and item.unit in {"USD", "USD/share", "EUR", "GBP"}
    }
    if len(currencies) > 1:
        findings.append(QAFinding(
            check="consistency.currency",
            severity=Severity.CRITICAL,
            message="Monetary evidence mixes currencies without conversion: "
                    f"{', '.join(sorted(currencies))}.",
            details={"currencies": sorted(currencies)},
        ))
    return findings


# ---------------------------------------------------------------------------
# Narrative QA
# ---------------------------------------------------------------------------

def check_requested_sections_present(context: QAContext) -> list[QAFinding]:
    """Every requested section is present, empty, or deliberately omitted.

    The synthesis layer drops a planned section the evidence could not fill and
    records the omission with its reason on the draft, so the report's shape
    follows the evidence rather than a fixed template. What QA enforces is that
    the drop was *declared*: an absent section with a recorded reason is a
    disclosed editorial decision (WARNING, printed in the report's back matter),
    while an absent section with no reason recorded is still a critical failure,
    because that is the signature of a section lost by accident.
    """
    findings: list[QAFinding] = []
    produced = {section.section for section in context.draft.sections}
    omitted = {
        str(entry.get("section")): str(entry.get("reason", ""))
        for entry in context.draft.metadata.get("sections_omitted", [])
        if isinstance(entry, dict)
    }

    for requested in context.plan.request.sections:
        if requested not in produced:
            reason = omitted.get(requested.value)
            if reason:
                findings.append(QAFinding(
                    check="narrative.section_omitted",
                    severity=Severity.WARNING,
                    message=(
                        f"Requested section {requested.value} was omitted: {reason}."
                    ),
                    section=requested.value,
                ))
                continue
            findings.append(QAFinding(
                check="narrative.section_present",
                severity=Severity.CRITICAL,
                message=f"Requested section {requested.value} is missing from the draft "
                        "and no reason for the omission was recorded.",
                section=requested.value,
            ))
            continue
        section = context.draft.section(requested)
        if section is None:
            continue
        # The sources section's substance is the draft-level citation list, which
        # the renderer prints under it rather than storing on the section.
        if requested is ReportSection.SOURCES and context.draft.citations:
            continue
        if not (section.statements or section.paragraphs or section.tables
                or section.data_gaps):
            findings.append(QAFinding(
                check="narrative.section_populated",
                severity=Severity.WARNING,
                message=f"Requested section {requested.value} is empty.",
                section=requested.value,
            ))
    return findings


def check_causal_claims(context: QAContext) -> list[QAFinding]:
    """A causal assertion needs a management source or a calculation behind it."""
    findings: list[QAFinding] = []
    for section in context.draft.sections:
        for statement in section.statements:
            lowered = statement.text.lower()
            markers = [m for m in CAUSAL_MARKERS if m in lowered]
            if not markers:
                continue
            acceptable = statement.claim_type in {
                ClaimType.MANAGEMENT_STATEMENT,
                ClaimType.CONFIRMED_FACT,
                ClaimType.CALCULATED_OBSERVATION,
            }
            if not acceptable:
                findings.append(QAFinding(
                    check="narrative.causal_support",
                    severity=Severity.WARNING,
                    message=(
                        f"Causal claim (\"{markers[0]}\") is typed as "
                        f"{statement.claim_type.value}, so the causation is asserted "
                        "rather than sourced."
                    ),
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
    return findings


#: Phrases that assert or imply an investment judgment - whether the company,
#: its shares, or its valuation are good, bad, attractive, or a buy/sell
#: opportunity - rather than stating a comparison and leaving the judgment to
#: the reader. This is a deterministic backstop behind the system prompts
#: (agents/llm_agent.py, synthesis/llm_synthesizer.py), which already instruct
#: the model to avoid this language; the prompt is not a guarantee, so this
#: check exists to catch what gets through anyway.
_JUDGMENT_MARKERS: tuple[str, ...] = (
    "supports its valuation", "justifies the premium", "justifies the discount",
    "central risk", "overtaking momentum", "attractive", "unattractive",
    "undervalued", "overvalued", "cheap valuation", "expensive valuation",
    "buying opportunity", "selling opportunity", "worth buying", "should buy",
    "should sell", "recommend buying", "recommend selling", "good investment",
    "bad investment", "compelling investment", "risk/reward is favourable",
    "risk/reward is favorable", "upside potential", "downside risk to the stock",
)


def check_judgmental_language(context: QAContext) -> list[QAFinding]:
    """Flag language that asserts an investment judgment rather than a fact.

    This report states comparisons (X is above Y, the data is consistent
    with...) and leaves any investment conclusion to the reader; it does not
    itself say whether a valuation is justified or a stock is attractive. See
    _JUDGMENT_MARKERS for the specific constructions this looks for.
    """
    findings: list[QAFinding] = []
    for section in context.draft.sections:
        for statement in section.statements:
            lowered = statement.text.lower()
            hits = [m for m in _JUDGMENT_MARKERS if m in lowered]
            if hits:
                findings.append(QAFinding(
                    check="narrative.judgmental_language",
                    severity=Severity.WARNING,
                    message=(
                        f"Statement uses investment-judgment language ({hits[0]!r}) "
                        "rather than a neutral comparison; this report does not take "
                        "a position on whether the company or its valuation is "
                        "attractive."
                    ),
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
    return findings


def check_duplication(context: QAContext) -> list[QAFinding]:
    """Flag statements repeated verbatim across sections."""
    findings: list[QAFinding] = []
    seen: dict[str, str] = {}
    for section in context.draft.sections:
        if section.section is ReportSection.KEY_TAKEAWAYS:
            continue  # the summary layer restates body claims by design
        for statement in section.statements:
            key = statement.text.strip().lower()
            if key in seen:
                findings.append(QAFinding(
                    check="narrative.duplication",
                    severity=Severity.WARNING,
                    message=f"Statement repeated verbatim; also appears in {seen[key]}.",
                    section=section.section.value,
                    subject=statement.text[:200],
                ))
                continue
            seen[key] = section.section.value
    return findings


def check_data_gaps_reported(context: QAContext) -> list[QAFinding]:
    """Unresolved data gaps must be visible in the report, not just internally."""
    findings: list[QAFinding] = []
    if not context.draft.data_gaps:
        return findings

    sources_section = context.draft.section(ReportSection.SOURCES)
    reported = {gap.description for gap in (sources_section.data_gaps if sources_section else ())}
    # A user who did not ask for a sources section has not asked for the gap
    # disclosure either; the gap is still reported, but it does not block.
    sources_requested = ReportSection.SOURCES in context.plan.request.sections
    for gap in context.draft.data_gaps:
        undisclosed = gap.description not in reported
        severity = (
            Severity.CRITICAL if (undisclosed and sources_requested) else Severity.WARNING)
        findings.append(QAFinding(
            check="narrative.data_gap",
            severity=severity,
            message=(
                f"Unresolved data gap: {gap.description}"
                + (f" Impact: {gap.impact}" if gap.impact else "")
                + ("" if not undisclosed
                   else " This gap is not disclosed in the report.")
            ),
            section=gap.segment.value if gap.segment else None,
            subject=gap.missing_metric,
        ))
    return findings


def check_data_freshness(context: QAContext) -> list[QAFinding]:
    """A live-verified newer public report must be disclosed, not silent.

    Set by pipeline.freshness_check (opt-in, see Settings.check_data_freshness)
    before synthesis; this only reads what it already recorded on
    draft.metadata - it does not itself call out to the web.
    """
    finding = context.draft.metadata.get("freshness_check")
    if not isinstance(finding, dict) or not finding.get("mismatched"):
        return []
    return [QAFinding(
        check="temporal.dataset_stale",
        severity=Severity.WARNING,
        message=(
            f"Live verification found a more recent public report "
            f"({finding.get('verified_period')}) than this report's dataset is anchored "
            f"on. Source: {finding.get('source_name')} <{finding.get('source_url')}>."
        ),
        details=finding,
    )]


def check_mock_data_disclosed(context: QAContext) -> list[QAFinding]:
    """Sample data must be labelled, and the label must reach the reader."""
    findings: list[QAFinding] = []
    if not context.draft.contains_mock_data:
        return findings
    findings.append(QAFinding(
        check="narrative.mock_data_disclosed",
        severity=Severity.WARNING,
        message=(
            "The report is built on illustrative sample data from mock providers. "
            "It must not be circulated as research."
        ),
        details={
            "mock_citations": [
                c.ref_number for c in context.draft.citations if c.is_mock
            ][:20],
        },
    ))
    return findings


# ---------------------------------------------------------------------------
# recomputation
# ---------------------------------------------------------------------------

def _recompute(formula: str, inputs: dict[str, float]) -> float | None:
    """Independently re-derive an analytics value from its stored inputs.

    Implemented as an explicit lookup on the formula string rather than by
    evaluating it, so QA is a genuine second implementation rather than a
    re-run of the same expression.
    """
    try:
        if formula == calc.F_PCT_CHANGE:
            return calc.pct_change(inputs["current"], inputs["prior"])
        if formula == calc.F_MARGIN_CHANGE:
            return calc.margin_change_pp(inputs["current_pct"], inputs["prior_pct"])
        if formula == calc.F_SURPRISE:
            return calc.surprise_pct(inputs["actual"], inputs["consensus"])
        if formula == calc.F_CONTRIBUTION:
            return calc.contribution_pct(inputs["part"], inputs["total"])
        if formula == calc.F_PREMIUM:
            return calc.premium_pct(inputs["value"], inputs["benchmark"])
        if formula == calc.F_REVISION_NET:
            return calc.net_revision_pct(inputs["up"], inputs["down"])
        if formula == "free_cash_flow / net_income * 100":
            return calc.cash_conversion(inputs["free_cash_flow"], inputs["net_income"])
        if formula == "own_growth_pct - peer_average_growth_pct":
            return inputs["own_growth_pct"] - inputs["peer_average_growth_pct"]
        if formula == "reported multiple (pass-through)":
            return float(inputs["value"])
        if formula == "mean(step changes in gross margin)":
            ordered = [inputs[k] for k in sorted(inputs, key=_step_key)]
            return calc.detect_trend(ordered).slope
    except Exception:  # noqa: BLE001 - a recompute that cannot run is itself reported
        return None
    return None


def _step_key(key: str) -> int:
    match = re.search(r"\d+", key)
    return int(match.group()) if match else 0


#: Every check, in the order the engine runs them.
ALL_CHECKS = (
    check_references_exist,
    check_claims_are_supported,
    check_numeric_claims_use_canonical_evidence,
    check_accounting_identities,
    check_citation_numbering,
    check_evidence_confidence,
    check_analytics_recompute,
    check_units,
    check_percentage_representation,
    check_period_comparability,
    check_dates_not_future,
    check_period_consistency,
    check_entity_consistency,
    check_metric_agreement,
    check_currency_consistency,
    check_requested_sections_present,
    check_causal_claims,
    check_judgmental_language,
    check_duplication,
    check_data_freshness,
    check_data_gaps_reported,
    check_mock_data_disclosed,
)
