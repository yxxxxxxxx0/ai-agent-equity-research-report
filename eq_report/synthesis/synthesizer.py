"""Stage 8 - the Synthesis Layer.

Combines the segment results into one ReportDraft: deduplicates claims, applies
house terminology, prioritises by materiality, assigns citation references and
orders the argument along the intended reasoning chain

    what changed -> why -> financial impact -> surprise versus expectations
    -> implication for expectations -> what matters next

Evidence and analytics references are carried through untouched, so every
sentence in the draft still points at the evidence it came from.

The shape of the report is decided here rather than fixed in advance. The plan
lists the sections a reader asked about, but a section only reaches the PDF if
the evidence produced something to say in it: a section with nothing behind it
is dropped and the omission recorded on the draft, and a thin section is folded
into its natural sibling. That is why two reports on different companies do not
come out the same length or in the same shape. Exhibits are built centrally by
:mod:`eq_report.synthesis.exhibits` and the first-page data panel by
:mod:`eq_report.synthesis.key_data`, so no figure is printed in two places.
"""

from __future__ import annotations

import logging
import re
from dataclasses import replace

from ..domain.analytics import AnalyticsBundle
from ..domain.enums import ClaimType, ReportSection, SegmentName
from ..domain.plan import ResearchPlan
from ..domain.report import (
    ChartSpec,
    MetricTable,
    ReportDraft,
    ReportSectionDraft,
    Statement,
)
from ..domain.segment import KeyFinding, SegmentResult
from ..evidence.reader import EvidenceReader
from ..logging_setup import get_logger, log_event
from .citations import CitationRegistry
from .exhibits import ExhibitBuilder
from .key_data import build_key_data_panel
from .terminology import (
    claim_fingerprint,
    is_near_duplicate,
    normalise_terminology,
    soften_unsupported_causation,
)

logger = get_logger("synthesis")

#: A section needs at least this much to be worth a heading of its own. Below
#: it, the section is either folded into a sibling (see _FOLD_INTO) or dropped
#: and recorded, rather than printed as a heading over one stray sentence.
_MIN_STATEMENTS = 2

#: Sections thin enough to fold, and what they fold into. The two sections
#: this maps are two readings of the same evidence, so a report that found
#: only one or two forward-looking events reads better with them under a
#: combined heading than with a near-empty section of their own.
_FOLD_INTO: dict[ReportSection, ReportSection] = {
    ReportSection.CATALYSTS: ReportSection.RISKS,
}

#: Title used when a fold actually happened.
_FOLDED_TITLES: dict[ReportSection, str] = {
    ReportSection.RISKS: "Uncertainties, Analytical Limitations and Monitoring Indicators",
}

#: Claim types that already justify causal language ("X because Y") - a
#: management statement, a reported fact, or a calculation that isolates the
#: driver. Everything else gets its safe causal markers softened to neutral
#: language (see terminology.soften_unsupported_causation); kept in sync with
#: qa.checks.check_causal_claims' own acceptable set.
_CAUSATION_ACCEPTABLE_CLAIM_TYPES = frozenset({
    ClaimType.MANAGEMENT_STATEMENT, ClaimType.CONFIRMED_FACT,
    ClaimType.CALCULATED_OBSERVATION,
})

#: Which segment supplies each report section, and the section's printed
#: title. Titles avoid investment-toned framing ("Risks", "Catalysts"), which
#: implies a bull/bear stance this report does not take: each section names
#: what it actually contains - facts still unresolved, or events that will
#: update a specific figure - not what an investor should feel about them.
_SECTION_SPEC: dict[ReportSection, tuple[SegmentName | None, str]] = {
    ReportSection.COMPANY_SNAPSHOT: (SegmentName.COMPANY_SNAPSHOT, "Company Snapshot"),
    ReportSection.KEY_TAKEAWAYS: (None, "Key Takeaways"),
    ReportSection.RECENT_DEVELOPMENTS: (SegmentName.RECENT_DEVELOPMENTS, "Recent Developments"),
    ReportSection.FINANCIALS: (SegmentName.FINANCIAL_PERFORMANCE,
                               "Latest Financial Performance"),
    ReportSection.OPERATING_DRIVERS: (SegmentName.OPERATING_DRIVERS,
                                      "Operating Drivers and Segment Performance"),
    ReportSection.COMPETITIVE_LANDSCAPE: (SegmentName.COMPETITIVE_LANDSCAPE,
                                          "Competitive Landscape"),
    ReportSection.VALUATION: (SegmentName.VALUATION, "Valuation and Market Expectations"),
    ReportSection.RISKS: (SegmentName.RISKS_CATALYSTS, "Uncertainties and Analytical Limitations"),
    ReportSection.CATALYSTS: (SegmentName.RISKS_CATALYSTS, "Upcoming Events and Monitoring Indicators"),
    ReportSection.WHAT_MATTERS_NEXT: (SegmentName.WHAT_MATTERS_NEXT, "Forward Monitoring Framework"),
    ReportSection.SOURCES: (None, "Sources and Data Gaps"),
}

#: The order the key takeaways are argued in - the reasoning chain, made
#: explicit rather than left to whichever agent happened to run first.
_TAKEAWAY_ORDER: tuple[tuple[SegmentName, str], ...] = (
    (SegmentName.FINANCIAL_PERFORMANCE, "revenue"),      # what changed
    (SegmentName.OPERATING_DRIVERS, "segment"),          # why it changed
    (SegmentName.FINANCIAL_PERFORMANCE, "margin"),       # financial impact
    (SegmentName.FINANCIAL_PERFORMANCE, "surprise"),     # surprise vs expectations
    (SegmentName.RECENT_DEVELOPMENTS, "guidance"),       # forward expectations
    (SegmentName.VALUATION, "valuation"),                # implication for the multiple
    (SegmentName.COMPETITIVE_LANDSCAPE, "competition"),  # relative position
    (SegmentName.RISKS_CATALYSTS, "risk"),               # what could break it
    (SegmentName.WHAT_MATTERS_NEXT, "monitor"),          # what matters next
)


class Synthesizer:
    """Turns segment results into a single coherent ReportDraft."""

    def __init__(
        self,
        report_run_id: str,
        plan: ResearchPlan,
        reader: EvidenceReader,
        analytics: AnalyticsBundle,
    ) -> None:
        self.report_run_id = report_run_id
        self.plan = plan
        self.reader = reader
        self.analytics = analytics
        self.citations = CitationRegistry(reader)
        # Duplicate detection is scoped to the report body. The Key Takeaways
        # section is a summary layer and is expected to restate a body claim, so
        # it is rendered against its own (discarded) scope; suppressing a body
        # claim because the summary already made it would gut the detail
        # sections. Each entry pairs a statement's text fingerprint with the
        # evidence/analytics ids it cited. Duplicate suppression uses wording,
        # not shared provenance: the same source can legitimately support a
        # financial result, an operating driver, and a risk.
        self._seen_fingerprints: list[tuple[frozenset[str], frozenset[str]]] = []
        self._duplicates_removed = 0
        self._exhibits = ExhibitBuilder(reader, analytics, plan.peers).build()
        #: Sections the plan asked for that the evidence could not fill, and
        #: why. Recorded on the draft so an omission is a disclosed decision
        #: rather than a silent hole - QA reads this.
        self._omitted: list[dict[str, str]] = []

    def synthesize(self, segment_results: tuple[SegmentResult, ...]) -> ReportDraft:
        by_segment = {result.segment: result for result in segment_results}

        # Key takeaways are built first so the most material claims claim the
        # lowest citation numbers, but the section is inserted in plan order.
        takeaways = self._key_takeaways(segment_results)
        drafted = self._draft_sections(by_segment, takeaways)
        sections = self._prune(drafted)

        if ReportSection.SOURCES in self.plan.sections:
            sections.append(self._sources_section(segment_results))

        draft = ReportDraft(
            report_run_id=self.report_run_id,
            company=self.plan.company,
            ticker=self.plan.ticker,
            report_date=self.plan.request.report_date,
            objective=self.plan.request.objective,
            title=self._title(),
            sections=tuple(sections),
            citations=self.citations.citations,
            key_data=build_key_data_panel(self.reader),
            metadata={
                "latest_reported_period": self.reader.latest_reported_period(),
                "peers": list(self.plan.peers),
                "analytics_count": len(self.analytics.results),
                "analytics_skipped": list(self.analytics.errors),
                "segments_run": [r.segment.value for r in segment_results],
                "segments_failed": [r.segment.value for r in segment_results if r.errors],
                "missing_evidence_references": list(self.citations.missing_evidence_ids),
                "duplicate_claims_removed": self._duplicates_removed,
                "sections_omitted": list(self._omitted),
            },
        )
        log_event(
            logger, logging.INFO, "synthesis complete",
            sections=len(draft.sections), statements=len(draft.all_statements),
            citations=len(draft.citations),
            duplicates_removed=self._duplicates_removed,
            sections_omitted=len(self._omitted),
        )
        return draft

    # -- report shape ----------------------------------------------------
    def _draft_sections(
        self,
        by_segment: dict[SegmentName, SegmentResult],
        takeaways: ReportSectionDraft,
    ) -> list[ReportSectionDraft]:
        """Draft every planned section; deciding which survive comes next.

        Key Takeaways leads, whatever position the plan gave it. It is the
        report's argument, and the first page sets it beside the key-data
        panel - the reader gets the conclusion and the figures behind it
        together, which is the arrangement every research first page uses.
        Putting the snapshot there instead would spend the same space
        restating the panel in sentences.
        """
        drafted: list[ReportSectionDraft] = []
        for section in _lead_first(self.plan.sections):
            spec = _SECTION_SPEC.get(section)
            if spec is None:
                log_event(logger, logging.WARNING, "unknown report section requested",
                          section=str(section))
                continue
            segment_name, title = spec

            if section is ReportSection.KEY_TAKEAWAYS:
                drafted.append(takeaways)
                continue
            if section is ReportSection.SOURCES:
                continue  # appended last, once every other section has cited

            result = by_segment.get(segment_name) if segment_name else None
            if result is None:
                self._omit(section, "no segment agent output was produced for it")
                continue
            drafted.append(self._build_section(section, title, result))
        return drafted

    def _prune(self, drafted: list[ReportSectionDraft]) -> list[ReportSectionDraft]:
        """Keep the sections the evidence actually filled.

        A section survives on substance, not on having been planned: enough
        statements to be worth a heading, or an exhibit that carries the point
        on its own. What is left over is folded into a sibling section where
        one is defined, and otherwise dropped and recorded.
        """
        kept: list[ReportSectionDraft] = []
        folded: dict[ReportSection, list[Statement]] = {}

        for section in drafted:
            if self._is_substantive(section):
                kept.append(section)
                continue
            target = _FOLD_INTO.get(section.section)
            if target is not None and section.statements:
                folded.setdefault(target, []).extend(section.statements)
                self._omit(
                    section.section,
                    f"only {len(section.statements)} statement(s) were evidenced, so they "
                    f"were folded into {target.value}",
                )
                continue
            self._omit(
                section.section,
                "No validated evidence or deterministic analytic supported this "
                "section, so it was omitted rather than fabricated.",
            )

        return [self._apply_fold(section, folded) for section in kept]

    @staticmethod
    def _is_substantive(section: ReportSectionDraft) -> bool:
        return (
            len(section.statements) >= _MIN_STATEMENTS
            or bool(section.tables)
            or bool(section.charts)
        )

    @staticmethod
    def _apply_fold(
        section: ReportSectionDraft, folded: dict[ReportSection, list[Statement]]
    ) -> ReportSectionDraft:
        """Append a folded sibling's statements and retitle the host section."""
        extra = folded.get(section.section)
        if not extra:
            return section
        return replace(
            section,
            title=_FOLDED_TITLES.get(section.section, section.title),
            statements=(*section.statements, *extra),
        )

    def _omit(self, section: ReportSection, reason: str) -> None:
        self._omitted.append({"section": section.value, "reason": reason})
        log_event(logger, logging.INFO, "section omitted",
                  section=section.value, reason=reason)

    # -- section builders ------------------------------------------------
    def _build_section(
        self, section: ReportSection, title: str, result: SegmentResult
    ) -> ReportSectionDraft:
        findings = self._findings_for_section(section, result)
        if section is ReportSection.CATALYSTS:
            # Catalysts deliberately shares its source findings with Risks:
            # the shared risk/catalyst agent may tag one finding with both
            # labels so it can serve both sections (see agents/llm_agent.py's
            # system prompt). Risks builds first and registers its
            # statements in the cross-section dedup scope, so a double-tagged
            # finding would otherwise be stripped back out of Catalysts as a
            # "repeat of itself" - defeating the entire point of allowing
            # both tags, and the actual reason Catalysts kept coming back
            # empty even when the agent produced real, on-topic findings for
            # it. Un-registering exactly the fingerprints Risks contributed
            # from this same result (not a blanket dedupe=False) means
            # Catalysts' statements still register themselves normally
            # afterward, so a *later* section is still correctly deduped
            # against whatever Catalysts ends up printing.
            risk_fingerprints = {
                self._fingerprint_for(f)
                for f in self._findings_for_section(ReportSection.RISKS, result)
            }
            self._seen_fingerprints = [
                (fp, ids) for fp, ids in self._seen_fingerprints
                if fp not in risk_fingerprints
            ]
        statements = self._statements(findings)

        return ReportSectionDraft(
            section=section,
            title=title,
            summary=self._section_summary(section, result),
            statements=statements,
            paragraphs=self._paragraphs(section, result),
            tables=self._tables(section),
            charts=self._charts(section),
        )

    @staticmethod
    def _paragraphs(section: ReportSection, result: SegmentResult) -> tuple[str, ...]:
        """The narrative paragraph(s) for a section.

        The risk/catalyst agent writes one narrative per section it serves, so
        the Catalysts section reads its own rather than repeating the risk one.
        """
        if section is ReportSection.CATALYSTS:
            text = str(result.metadata.get("catalyst_narrative", ""))
        else:
            text = result.draft_narrative
        return (normalise_terminology(text),) if text.strip() else ()

    @staticmethod
    def _section_summary(section: ReportSection, result: SegmentResult) -> str:
        """The section's standfirst: one line saying what the section concludes.

        This used to print a count ("3 evidenced risks to the current
        trajectory"), which told the reader nothing they could not get by
        looking. The risk/catalyst agent serves two sections, so the catalyst
        side reads the headline the agent wrote for it and falls back to
        nothing rather than to the risk headline, which would be wrong there.
        """
        if section is ReportSection.CATALYSTS:
            return normalise_terminology(str(result.metadata.get("catalyst_headline", "")))
        return normalise_terminology(result.headline)

    def _findings_for_section(
        self, section: ReportSection, result: SegmentResult
    ) -> tuple[KeyFinding, ...]:
        """Select the findings that belong in this section.

        The risk/catalyst agent produces one result serving two sections, so the
        split happens here on the tags the agent applied.
        """
        findings = result.key_findings
        if section is ReportSection.RISKS:
            findings = tuple(f for f in findings if _tag_matches(f.tags, "risk"))
        elif section is ReportSection.CATALYSTS:
            findings = tuple(f for f in findings if _tag_matches(f.tags, "catalyst"))
        return tuple(sorted(findings, key=lambda f: (f.materiality, _claim_rank(f.claim_type))))

    @staticmethod
    def _softened_text(finding: KeyFinding) -> str:
        text = normalise_terminology(finding.claim)
        return soften_unsupported_causation(
            text, supported=finding.claim_type in _CAUSATION_ACCEPTABLE_CLAIM_TYPES)

    @classmethod
    def _fingerprint_for(cls, finding: KeyFinding) -> frozenset[str]:
        return claim_fingerprint(cls._softened_text(finding))

    def _statements(
        self, findings: tuple[KeyFinding, ...], *, dedupe: bool = True
    ) -> tuple[Statement, ...]:
        """Render findings as statements, dropping repeats and softening
        causal language the claim type does not support.

        ``dedupe=False`` renders without consulting or updating the body scope,
        which is what the Key Takeaways section needs. A repeat is caught two
        independent ways: near-identical wording (catches two findings that
        happen to be phrased alike), or a high overlap in the underlying
        evidence/analytics ids cited (catches two segment agents independently
        restating the same fact in *different* words - number formatting
        alone, e.g. "$62.3 billion" versus "$62.3bn", is already enough to
        defeat a wording-only comparison, while the two still cite the same
        handful of rows). Either signal alone is treated as a repeat; a
        finding that pulls in materially different evidence lowers its own
        overlap ratio and survives, which is what lets a genuinely new angle
        on a previously-used fact still get through.
        """
        statements: list[Statement] = []
        local_seen: list[tuple[frozenset[str], frozenset[str]]] = []
        for finding in findings:
            text = self._softened_text(finding)
            fingerprint = self._fingerprint_for(finding)
            ids = frozenset((*finding.evidence_ids, *finding.analytics_ids))
            scope = self._seen_fingerprints if dedupe else local_seen
            if any(
                is_near_duplicate(fingerprint, seen_text)
                for seen_text, seen_ids in scope
            ):
                if dedupe:
                    self._duplicates_removed += 1
                continue
            scope.append((fingerprint, ids))
            statements.append(Statement(
                text=text,
                claim_type=finding.claim_type,
                evidence_ids=finding.evidence_ids,
                analytics_ids=finding.analytics_ids,
                citation_refs=self.citations.refs_for(finding.evidence_ids),
                confidence=finding.confidence,
            ))
        return tuple(statements)

    def _tables(self, section: ReportSection) -> tuple[MetricTable, ...]:
        """This section's tables, from the single central set of exhibits.

        Tables are no longer assembled from ``result.important_metrics``. Every
        agent is shown the same retrieved evidence pool and therefore surfaced
        much the same metrics, which is why the report used to print the same
        figures in a table under almost every heading. ``important_metrics``
        remains on the SegmentResult, and in the run JSON, as the record of
        what each agent considered material.
        """
        tables = tuple(
            exhibit.table for exhibit in self._exhibits
            if exhibit.section is section and exhibit.table is not None
        )
        self._cite_exhibits(section)
        return tables

    def _charts(self, section: ReportSection) -> tuple[ChartSpec, ...]:
        return tuple(
            exhibit.chart for exhibit in self._exhibits
            if exhibit.section is section and exhibit.chart is not None
        )

    def _cite_exhibits(self, section: ReportSection) -> None:
        """Register the evidence behind this section's exhibits, so the figures
        in them resolve to the same source list as the prose."""
        for exhibit in self._exhibits:
            if exhibit.section is section:
                self.citations.refs_for(exhibit.evidence_ids)

    def _key_takeaways(self, segment_results: tuple[SegmentResult, ...]) -> ReportSectionDraft:
        """The report's argument, in the intended reasoning order."""
        by_segment = {result.segment: result for result in segment_results}
        selected: list[KeyFinding] = []

        for segment_name, tag in _TAKEAWAY_ORDER:
            result = by_segment.get(segment_name)
            if result is None:
                continue
            candidates = [
                f for f in result.key_findings
                if tag in f.tags and f.materiality == 1
            ]
            if not candidates:
                candidates = [f for f in result.key_findings if tag in f.tags]
            if candidates:
                selected.append(candidates[0])

        statements = self._statements(tuple(selected), dedupe=False)
        return ReportSectionDraft(
            section=ReportSection.KEY_TAKEAWAYS,
            title="Key Takeaways",
            summary=self._takeaway_summary(by_segment),
            statements=statements,
        )

    def _takeaway_summary(self, by_segment: dict[SegmentName, SegmentResult]) -> str:
        """One line stating the report's conclusion, built from headlines held."""
        financial = by_segment.get(SegmentName.FINANCIAL_PERFORMANCE)
        valuation = by_segment.get(SegmentName.VALUATION)
        parts = []
        if financial and financial.headline:
            parts.append(normalise_terminology(financial.headline))
        if valuation and valuation.headline:
            parts.append(normalise_terminology(valuation.headline))
        return " | ".join(parts) if parts else "See sections below."

    def _sources_section(
        self, segment_results: tuple[SegmentResult, ...]
    ) -> ReportSectionDraft:
        analytics_skipped = self.analytics.errors
        paragraphs = []
        if analytics_skipped:
            paragraphs.append(
                "Analytics not computed for want of inputs: "
                + "; ".join(analytics_skipped)
            )
        agent_errors = [
            f"{r.segment.value}: {error}"
            for r in segment_results for error in r.errors
        ]
        if agent_errors:
            paragraphs.append("Segment agent failures: " + "; ".join(agent_errors))

        return ReportSectionDraft(
            section=ReportSection.SOURCES,
            title="Sources",
            summary=f"{len(self.citations.citations)} sources cited.",
            paragraphs=tuple(paragraphs),
        )

    def _title(self) -> str:
        ticker = f" ({self.plan.ticker})" if self.plan.ticker else ""
        objective = self.plan.request.objective.strip().title()
        return f"{self.plan.company}{ticker} - {objective}"


def _tag_matches(tags: tuple[str, ...], label: str) -> bool:
    """Match semantic labels inside model-authored multi-word tags."""
    wanted = label.casefold()
    return any(
        wanted in re.findall(r"[a-z0-9]+", str(tag).casefold())
        for tag in tags
    )


def _lead_first(sections: tuple[ReportSection, ...]) -> tuple[ReportSection, ...]:
    """Plan order, with Key Takeaways moved to the front if it was requested."""
    if ReportSection.KEY_TAKEAWAYS not in sections:
        return sections
    return (
        ReportSection.KEY_TAKEAWAYS,
        *(s for s in sections if s is not ReportSection.KEY_TAKEAWAYS),
    )


def _claim_rank(claim_type: ClaimType) -> int:
    """Facts lead, interpretations follow, within the same materiality band."""
    order = {
        ClaimType.CONFIRMED_FACT: 0,
        ClaimType.CALCULATED_OBSERVATION: 1,
        ClaimType.MANAGEMENT_STATEMENT: 2,
        ClaimType.MARKET_EXPECTATION: 3,
        ClaimType.INTERPRETATION: 4,
    }
    return order.get(claim_type, 5)
