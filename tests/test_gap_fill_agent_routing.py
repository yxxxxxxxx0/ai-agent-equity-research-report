import datetime as dt

from eq_report.agents.llm_agent import VerifiedSegmentAgent
from eq_report.agents.runner import build_agents
from eq_report.config import ModelConfig
from eq_report.domain.enums import (
    EvidenceCategory,
    EvidenceStatus,
    FactType,
    SegmentName,
    SourceType,
)
from eq_report.domain.evidence import EvidenceItem
from eq_report.domain.plan import ResearchPlan, SegmentTask
from eq_report.domain.request import ResearchRequest
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore

_DETERMINISTIC = frozenset({"company_snapshot", "financial_performance", "operating_drivers"})


def _plan() -> ResearchPlan:
    request = ResearchRequest(company="Example Corp", ticker="EX", report_date=dt.date(2026, 9, 22))
    return ResearchPlan(
        request=request, company="Example Corp", ticker="EX", peers=(),
        questions=(), required_market_metrics=(), required_fundamental_metrics=(),
        required_documents=(), sections=(), required_analytics=(),
        segment_tasks=(
            SegmentTask(segment=SegmentName.FINANCIAL_PERFORMANCE, objective="x"),
            SegmentTask(segment=SegmentName.OPERATING_DRIVERS, objective="x"),
            SegmentTask(segment=SegmentName.COMPANY_SNAPSHOT, objective="x"),
        ),
        source_priorities=(),
    )


def _gap_fill_item(topic: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=f"ev_{topic}",
        report_run_id="run",
        company="Example Corp",
        ticker="EX",
        category=EvidenceCategory.DOCUMENT,
        source_id="web:sec.gov",
        source_name="SEC EDGAR",
        source_type=SourceType.COMPANY_FILING,
        retrieved_at=dt.datetime.now(dt.timezone.utc),
        claim_text=f"a fact about {topic}",
        published_at=dt.date(2026, 8, 1),
        retrieval_provider="web_gap_fill",
        fact_type=FactType.REPORTED_FACT,
        status=EvidenceStatus.VALIDATED,
        metadata={"gap_fill_topic": topic},
    )


def _reader(items: list[EvidenceItem]) -> EvidenceReader:
    store = EvidenceStore(":memory:")
    store.save(items)
    return EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")


def test_deterministic_segments_stay_deterministic_without_gap_fill_material():
    model_config = ModelConfig(deterministic_segments=_DETERMINISTIC, api_key="x")
    reader = _reader([])
    agents = build_agents(_plan(), model_config, reader=reader)
    by_segment = {a.segment: a for a in agents}
    assert not isinstance(by_segment[SegmentName.FINANCIAL_PERFORMANCE], VerifiedSegmentAgent)
    assert not isinstance(by_segment[SegmentName.OPERATING_DRIVERS], VerifiedSegmentAgent)


def test_a_segment_with_gap_fill_material_is_routed_to_the_llm_writer():
    model_config = ModelConfig(deterministic_segments=_DETERMINISTIC, api_key="x")
    reader = _reader([_gap_fill_item("financials")])
    agents = build_agents(_plan(), model_config, reader=reader)
    by_segment = {a.segment: a for a in agents}

    # financials got gap-filled: it must reach the writer that actually
    # reads document evidence, overriding the deterministic-segment default.
    assert isinstance(by_segment[SegmentName.FINANCIAL_PERFORMANCE], VerifiedSegmentAgent)
    # operating_drivers got nothing: it stays on the cheaper deterministic path.
    assert not isinstance(by_segment[SegmentName.OPERATING_DRIVERS], VerifiedSegmentAgent)


def test_no_reader_falls_back_to_the_configured_deterministic_set():
    model_config = ModelConfig(deterministic_segments=_DETERMINISTIC, api_key="x")
    agents = build_agents(_plan(), model_config, reader=None)
    by_segment = {a.segment: a for a in agents}
    assert not isinstance(by_segment[SegmentName.FINANCIAL_PERFORMANCE], VerifiedSegmentAgent)
