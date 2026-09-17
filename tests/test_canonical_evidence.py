import datetime as dt
from dataclasses import replace
from types import SimpleNamespace

import pytest

from eq_report.domain.enums import (
    Confidence,
    EvidenceCategory,
    EvidenceStatus,
    SourceType,
)
from eq_report.domain.evidence import EvidenceItem, FiscalPeriod
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore
from eq_report.normalisation.normalizer import Normalizer
from eq_report.normalisation.reconciliation import reconcile
from eq_report.normalisation.units import (
    UNIT_PERCENT,
    UNIT_PERCENTAGE_POINTS,
    parse_number,
)
from eq_report.normalisation.validation import (
    detect_outlier,
    guidance_midpoint,
    validate_ratio,
)
from eq_report.providers.megadata import _extract_passages
from eq_report.qa.checks import has_asserted_numeric_fact


def item(eid, value, *, metric="revenue", basis=None, source=SourceType.COMPANY_FILING,
         period="FY2026 Q4", company="Sandisk", ticker="SNDK"):
    return EvidenceItem(
        evidence_id=eid, report_run_id="run", company=company, ticker=ticker,
        category=EvidenceCategory.FUNDAMENTAL, source_id=eid, source_name="source",
        source_type=source, retrieved_at=dt.datetime.now(dt.UTC), metric=metric,
        value=value, unit="USD/share" if metric and "eps" in metric else "USD", currency="USD",
        period=FiscalPeriod(period, 2026, 4), confidence=Confidence.HIGH,
        metadata={"basis": basis} if basis else {},
    )


def test_unit_normalization_billion_equals_millions():
    assert parse_number("$8.97B").value == parse_number("8970 million").value == 8_970_000_000


def test_percent_and_percentage_point_are_distinct_units():
    assert UNIT_PERCENT != UNIT_PERCENTAGE_POINTS


def test_gaap_and_non_gaap_are_distinct_not_conflicted():
    rows = reconcile([item("a", 38.82, metric="diluted_eps", basis="GAAP"),
                      item("b", 39.25, metric="diluted_eps", basis="non_GAAP")])
    assert sum(x.status is EvidenceStatus.VALIDATED and x.is_canonical for x in rows) == 2


def test_quarterly_and_annual_are_distinct():
    q = item("q", 10); a = replace(item("a", 40), period=FiscalPeriod("FY2026", 2026, None, is_annual=True), frequency="annual")
    rows = reconcile([q, a])
    assert sum(x.is_canonical for x in rows) == 2


def test_guidance_midpoint():
    assert guidance_midpoint(8.0, 10.0) == 9.0
    with pytest.raises(ValueError): guidance_midpoint(10, 8)


def test_same_basis_conflict_blocks_both():
    rows = reconcile([item("a", 39.25, metric="diluted_eps", basis="non_GAAP"),
                      item("b", 38.82, metric="diluted_eps", basis="non_GAAP")])
    assert all(x.status is EvidenceStatus.CONFLICTED and not x.is_canonical for x in rows)


def test_accounting_identity():
    assert validate_ratio("gross_margin", 50, 100, 50) is None
    assert validate_ratio("gross_margin", 40, 100, 50).severity == "P0"


def test_provenance_round_trip_and_reader_only_canonical(tmp_path):
    store = EvidenceStore(tmp_path / "e.sqlite")
    canonical = reconcile([replace(item("a", 8.12, metric="forward_pe"), unit="x", currency=None,
                                   metadata={"valuation_horizon": "NTM"},
                                   original_source_url="https://example.com/original",
                                   retrieval_provider="MegaAPI", retrieval_url="http://192.168.1.16/api")])[0]
    bypass = replace(item("b", 9.0, metric="forward_pe"), status=EvidenceStatus.UNVERIFIED)
    store.save([canonical, bypass])
    reader = EvidenceReader(store, "run", "SNDK", "Sandisk")
    assert reader.numeric("forward_pe").value == 8.12
    saved = store.get("a")
    assert saved.original_source_url == "https://example.com/original"
    assert saved.retrieval_provider == "MegaAPI"


def test_outlier_requests_verification_not_rejection():
    issue = detect_outlier("gross_margin", 95)
    assert issue.severity == "P1" and "VERIFY_PRIMARY_SOURCE" in issue.message


def test_sndk_identity_is_not_wdc():
    rows = reconcile([item("s", 1), item("w", 2, company="Western Digital", ticker="WDC")])
    assert sum(x.is_canonical for x in rows) == 2


def test_rounding_tolerance_selects_one_canonical_revenue():
    rows = reconcile([item("a", 8_970_000_000), item("b", 8_969_000_000, source=SourceType.NEWS)])
    assert sum(x.is_canonical for x in rows) == 1
    assert all(x.status is EvidenceStatus.VALIDATED for x in rows)


def test_unresolved_eps_basis_is_not_publishable():
    rows = reconcile([item("a", 38.82, metric="diluted_eps")])
    assert rows[0].status is EvidenceStatus.CONFLICTED and not rows[0].is_canonical


def test_original_source_replaces_vector_search_url():
    record = {"story_id": "https://publisher.example/story:0", "headline": "Title",
              "story_content": "SNDK context", "created_at": "2026-08-05",
              "source_code": "Publisher"}
    req = SimpleNamespace(request_id="search", purpose="research")
    plan = SimpleNamespace(company="Sandisk", ticker="SNDK")
    passage = _extract_passages([record], plan, req, "http://192.168.1.16/api/vector-search")[0]
    assert passage.source.source_url == "https://publisher.example/story"
    assert passage.source.retrieval_url.startswith("http://192.168.1.16")


def test_derived_dependency_ids_are_preserved_in_metadata():
    row = replace(item("d", 17.6, metric="guidance_growth"),
                  source_type=SourceType.DERIVED,
                  metadata={"dependencies": ["low", "high", "prior"]})
    result = reconcile([row])[0]
    assert result.metadata["dependencies"] == ["low", "high", "prior"]


def test_dates_and_fiscal_labels_are_not_numeric_claims():
    assert not has_asserted_numeric_fact(
        "The FY2026 Q2 filing dated August 1, 2026 discusses demand.")
    assert has_asserted_numeric_fact("Revenue was $28.2 billion in FY2026 Q2.")


def test_retrieved_document_number_cannot_enter_numeric_reader(tmp_path):
    store = EvidenceStore(tmp_path / "e.sqlite")
    doc = replace(item("doc", None, metric=None), category=EvidenceCategory.DOCUMENT,
                  claim_text="Forward P/E is 8.12x", unit=None, currency=None)
    store.save(reconcile([doc]))
    reader = EvidenceReader(store, "run", "SNDK", "Sandisk")
    assert reader.numeric("forward_pe") is None


def test_metric_shape_collapses_array_indexes_for_small_llm_batches():
    assert Normalizer._metric_shape("NVDA.184.PX_LAST") == "NVDA.#.PX_LAST"
    assert Normalizer._metric_shape("BBG.12.periods.7.revenue") == "BBG.#.periods.#.revenue"


def test_llm_metric_priority_prefers_report_relevant_fields():
    assert Normalizer._metric_priority("NVDA.#.revenue") > Normalizer._metric_priority("NVDA.#.EMA_19")
