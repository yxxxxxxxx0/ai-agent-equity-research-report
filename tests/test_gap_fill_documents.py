import datetime as dt

from eq_report.domain.enums import EvidenceCategory, EvidenceStatus, FactType, SourceType
from eq_report.domain.evidence import EvidenceItem
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore


def _document(evidence_id: str, *, published_at: dt.date, retrieval_provider: str = "megadata",
              gap_fill_topic: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        report_run_id="run",
        company="Example Corp",
        ticker="EX",
        category=EvidenceCategory.DOCUMENT,
        source_id="src",
        source_name="Some Source",
        source_type=SourceType.NEWS,
        retrieved_at=dt.datetime.now(dt.timezone.utc),
        claim_text=f"claim {evidence_id}",
        published_at=published_at,
        retrieval_provider=retrieval_provider,
        fact_type=FactType.REPORTED_FACT,
        status=EvidenceStatus.VALIDATED,
        metadata={"gap_fill_topic": gap_fill_topic} if gap_fill_topic else {},
    )


def test_gap_fill_documents_survive_a_recency_cutoff_that_would_bury_them():
    store = EvidenceStore(":memory:")
    # 40 fresher, unrelated MegaAPI documents - well past documents()'s
    # usual 30-item shared-pool cutoff used by agents/llm_agent.py.
    fresh = [
        _document(f"news{i}", published_at=dt.date(2026, 9, 1) - dt.timedelta(days=i))
        for i in range(40)
    ]
    old_gap_fill = _document(
        "gap1", published_at=dt.date(2026, 8, 1),
        retrieval_provider="web_gap_fill", gap_fill_topic="financials",
    )
    store.save(fresh + [old_gap_fill])
    reader = EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")

    # The plain recency-limited pool a segment agent's shared block uses
    # would not include it...
    top_30 = reader.documents(limit=30)
    assert old_gap_fill.evidence_id not in {i.evidence_id for i in top_30}

    # ...but the topic-targeted lookup finds it regardless.
    found = reader.gap_fill_documents("financials")
    assert [i.evidence_id for i in found] == ["gap1"]


def test_gap_fill_documents_only_returns_the_requested_topic():
    store = EvidenceStore(":memory:")
    store.save([
        _document("fin1", published_at=dt.date(2026, 8, 1),
                  retrieval_provider="web_gap_fill", gap_fill_topic="financials"),
        _document("ops1", published_at=dt.date(2026, 8, 1),
                  retrieval_provider="web_gap_fill", gap_fill_topic="operating_drivers"),
    ])
    reader = EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")
    assert [i.evidence_id for i in reader.gap_fill_documents("financials")] == ["fin1"]
    assert [i.evidence_id for i in reader.gap_fill_documents("operating_drivers")] == ["ops1"]
