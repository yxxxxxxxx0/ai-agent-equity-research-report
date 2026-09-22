import datetime as dt
from dataclasses import replace

from eq_report.domain.enums import (
    ClaimType,
    Confidence,
    EvidenceCategory,
    EvidenceStatus,
    FactType,
    ReportSection,
    SourceType,
)
from eq_report.domain.evidence import EvidenceItem
from eq_report.domain.report import ReportDraft, ReportSectionDraft, Statement
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore
from eq_report.llm.client import LLMJSONResponse
from eq_report.qa.checks import QAContext, check_web_claims_reverified
from eq_report.qa.web_claim_auditor import reverify_web_claims


def _web_evidence(evidence_id: str) -> EvidenceItem:
    return EvidenceItem(
        evidence_id=evidence_id,
        report_run_id="run",
        company="Example Corp",
        ticker="EX",
        category=EvidenceCategory.DOCUMENT,
        source_id="web:sec.gov",
        source_name="SEC EDGAR",
        source_type=SourceType.COMPANY_FILING,
        retrieved_at=dt.datetime.now(dt.timezone.utc),
        claim_text="Example Corp reported $1B revenue.",
        source_url="https://www.sec.gov/example-10k",
        published_at=dt.date(2026, 8, 1),
        retrieval_provider="web_gap_fill",
        fact_type=FactType.REPORTED_FACT,
        status=EvidenceStatus.VALIDATED,
        confidence=Confidence.MEDIUM,
    )


def _draft_citing(evidence_id: str) -> ReportDraft:
    return ReportDraft(
        report_run_id="run",
        company="Example Corp",
        ticker="EX",
        report_date=dt.date(2026, 9, 22),
        objective="test",
        title="Example",
        sections=(ReportSectionDraft(
            section=ReportSection.FINANCIALS,
            title="Financials",
            statements=(Statement(
                text="Example Corp reported $1B revenue.",
                claim_type=ClaimType.CONFIRMED_FACT,
                evidence_ids=(evidence_id,),
            ),),
        ),),
    )


class _RejectClient:
    async def complete_json(self, _system, _prompt, *, web_search=False, stage="", **_kw):
        return LLMJSONResponse(
            payload={"verified": False, "note": "source no longer states this"},
            input_tokens=1, output_tokens=1,
        )


class _ConfirmClient:
    async def complete_json(self, _system, _prompt, *, web_search=False, stage="", **_kw):
        return LLMJSONResponse(payload={"verified": True}, input_tokens=1, output_tokens=1)


def _reader_with(item: EvidenceItem) -> EvidenceReader:
    store = EvidenceStore(":memory:")
    store.save([item])
    return EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")


async def test_a_claim_that_fails_reverification_is_demoted_and_blocks_qa():
    item = _web_evidence("ev1")
    reader = _reader_with(item)
    draft = _draft_citing("ev1")

    result = await reverify_web_claims(draft, reader, None, client=_RejectClient())
    assert result.rejected == ("ev1",)
    assert reader.get("ev1").status is EvidenceStatus.REJECTED

    findings = check_web_claims_reverified(
        QAContext(draft=draft, plan=None, reader=reader, analytics=None))
    assert len(findings) == 1
    assert findings[0].check == "evidence.web_claim_not_reverified"
    assert findings[0].severity.value == "critical"


async def test_a_claim_that_reconfirms_is_left_untouched():
    item = _web_evidence("ev2")
    reader = _reader_with(item)
    draft = _draft_citing("ev2")

    result = await reverify_web_claims(draft, reader, None, client=_ConfirmClient())
    assert result.rejected == ()
    assert reader.get("ev2").status is EvidenceStatus.VALIDATED

    findings = check_web_claims_reverified(
        QAContext(draft=draft, plan=None, reader=reader, analytics=None))
    assert findings == []


async def test_non_web_evidence_is_never_rechecked():
    item = replace(_web_evidence("ev3"), retrieval_provider="megadata")
    reader = _reader_with(item)
    draft = _draft_citing("ev3")

    class _ExplodingClient:
        async def complete_json(self, *args, **kwargs):
            raise AssertionError("should not be called for non-web evidence")

    result = await reverify_web_claims(draft, reader, None, client=_ExplodingClient())
    assert result.checked == 0
    assert result.rejected == ()
