import datetime as dt

import pytest

from eq_report.domain.analytics import AnalyticsBundle
from eq_report.domain.enums import (
    ClaimType, Confidence, EvidenceCategory, EvidenceStatus, FactType,
    ReportSection, SourceType,
)
from eq_report.domain.evidence import EvidenceItem
from eq_report.domain.report import ReportDraft, ReportSectionDraft, Statement
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore
from eq_report.llm.client import LLMJSONResponse
from eq_report.qa.entailment import verify_claim_entailment


def _fixture():
    evidence = EvidenceItem(
        evidence_id="ev_revenue", report_run_id="run", company="Example Corp", ticker="EX",
        category=EvidenceCategory.DOCUMENT, source_id="sec", source_name="SEC EDGAR",
        source_type=SourceType.COMPANY_FILING, retrieved_at=dt.datetime.now(dt.timezone.utc),
        claim_text="Example Corp reported fiscal-year revenue of $1 billion.",
        source_url="https://www.sec.gov/example", fact_type=FactType.REPORTED_FACT,
        status=EvidenceStatus.VALIDATED, confidence=Confidence.HIGH,
    )
    store = EvidenceStore(":memory:")
    store.save([evidence])
    reader = EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")
    draft = ReportDraft(
        report_run_id="run", company="Example Corp", ticker="EX",
        report_date=dt.date(2026, 9, 22), objective="test", title="Example",
        sections=(ReportSectionDraft(
            section=ReportSection.FINANCIALS, title="Financials",
            statements=(Statement(
                text="Example Corp reported fiscal-year revenue of $2 billion.",
                claim_type=ClaimType.CONFIRMED_FACT, evidence_ids=("ev_revenue",),
            ),),
        ),),
    )
    return draft, reader


class _Client:
    def __init__(self, supported):
        self.supported = supported

    async def complete_json(self, _system, _prompt, *, stage="", **_kwargs):
        return LLMJSONResponse(
            payload={"results": [{
                "claim_id": "s0c0", "supported": self.supported,
                "reason": "the source says $1 billion, not $2 billion",
            }]}, input_tokens=1, output_tokens=1,
        )


@pytest.mark.asyncio
async def test_entailment_rejects_a_citation_that_does_not_support_the_claim():
    draft, reader = _fixture()
    findings = await verify_claim_entailment(
        draft, reader, AnalyticsBundle(), None, client=_Client(False))
    assert [finding.check for finding in findings] == ["evidence.claim_not_entailed"]


@pytest.mark.asyncio
async def test_entailment_accepts_a_claim_when_mapped_source_supports_it():
    draft, reader = _fixture()
    findings = await verify_claim_entailment(
        draft, reader, AnalyticsBundle(), None, client=_Client(True))
    assert findings == []
