from types import SimpleNamespace

from eq_report.agents.llm_agent import _risk_catalyst_tags
from eq_report.domain.enums import ClaimType
from eq_report.domain.segment import KeyFinding
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
