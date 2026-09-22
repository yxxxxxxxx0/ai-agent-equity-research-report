from types import SimpleNamespace

from eq_report.agents.llm_agent import _risk_catalyst_tags
from eq_report.domain.enums import ClaimType, ReportSection
from eq_report.domain.segment import KeyFinding, SegmentResult
from eq_report.synthesis.synthesizer import Synthesizer, _tag_matches


def test_multiword_model_tags_route_risks_and_catalysts():
    assert _tag_matches(("execution regulation demand risk register",), "risk")
    assert _tag_matches(("earnings guidance gross margin catalyst",), "catalyst")
    assert not _tag_matches(("earnings guidance gross margin catalyst",), "risk")


def test_free_form_risk_catalyst_tags_gain_stable_routing_labels():
    tags = _risk_catalyst_tags(
        "Capacity constraints remain a risk; the next update will test deployment.",
        ("cloud capacity",),
    )
    assert "risk" in tags
    assert "catalyst" in tags


def test_shared_evidence_does_not_delete_a_distinct_section_claim():
    synthesizer = object.__new__(Synthesizer)
    synthesizer.citations = SimpleNamespace(refs_for=lambda _ids: (1,))
    synthesizer._seen_fingerprints = []
    synthesizer._duplicates_removed = 0

    financial = KeyFinding(
        claim="Quarterly revenue increased while gross margin expanded.",
        claim_type=ClaimType.CONFIRMED_FACT,
        evidence_ids=("ev-shared",),
    )
    driver = KeyFinding(
        claim="Pricing and product mix remain the disclosed operating variables.",
        claim_type=ClaimType.MANAGEMENT_STATEMENT,
        evidence_ids=("ev-shared",),
    )

    assert len(synthesizer._statements((financial,))) == 1
    assert len(synthesizer._statements((driver,))) == 1
    assert synthesizer._duplicates_removed == 0


def test_a_finding_double_tagged_risk_and_catalyst_survives_in_both_sections():
    """Regression: Risks builds first and used to register its statements in
    the shared cross-section dedup scope, so a finding legitimately tagged
    both "risk" and "catalyst" (explicitly allowed by the risk/catalyst
    agent's own prompt, to serve both sections) was stripped back out of
    Catalysts as a "repeat of itself" - Catalysts came back empty even when
    the agent produced real, on-topic findings for it."""
    synthesizer = object.__new__(Synthesizer)
    synthesizer.citations = SimpleNamespace(refs_for=lambda _ids: (1,))
    synthesizer._seen_fingerprints = []
    synthesizer._duplicates_removed = 0
    synthesizer._exhibits = ()

    finding = KeyFinding(
        claim="Apple's October 29 earnings release is the next dated financial checkpoint.",
        claim_type=ClaimType.CONFIRMED_FACT,
        evidence_ids=("ev-1",),
        tags=("risk", "catalyst"),
    )
    result = SegmentResult(segment=None, headline="", key_findings=(finding,))

    risks = synthesizer._build_section(ReportSection.RISKS, "Risks", result)
    catalysts = synthesizer._build_section(ReportSection.CATALYSTS, "Catalysts", result)

    assert len(risks.statements) == 1
    assert len(catalysts.statements) == 1
    assert synthesizer._duplicates_removed == 0
