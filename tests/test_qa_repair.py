import datetime as dt
from types import SimpleNamespace

from eq_report.domain.analytics import AnalyticsBundle
from eq_report.domain.enums import ClaimType, ReportSection, Severity
from eq_report.domain.qa import QAFinding, QAResult
from eq_report.domain.report import ReportDraft, ReportSectionDraft, Statement
from eq_report.llm.client import LLMJSONResponse
from eq_report.qa.repair import DraftRepairer


def _draft(text: str) -> ReportDraft:
    return ReportDraft(
        report_run_id="run",
        company="Example",
        ticker="EX",
        report_date=dt.date(2026, 9, 17),
        objective="test",
        title="Example",
        sections=(ReportSectionDraft(
            section=ReportSection.RECENT_DEVELOPMENTS,
            title="Recent Developments",
            statements=(Statement(
                text=text,
                claim_type=ClaimType.MANAGEMENT_STATEMENT,
                evidence_ids=("doc",),
            ),),
        ),),
    )


def _qa(text: str) -> QAResult:
    return QAResult(findings=(QAFinding(
        check="evidence.numeric_claim_not_canonical",
        severity=Severity.CRITICAL,
        message="unsupported numeric claim",
        section=ReportSection.RECENT_DEVELOPMENTS.value,
        subject=text,
        details={"evidence_ids": ["doc"]},
    ),))


class _RewriteClient:
    async def complete_json(self, _system: str, _prompt: str) -> LLMJSONResponse:
        return LLMJSONResponse(payload={"items": [{
            "index": 0,
            "action": "rewrite",
            "rewritten_text": "Management described margin as a monitoring priority.",
        }]}, input_tokens=1, output_tokens=1)


async def test_llm_rewrite_is_applied_without_changing_provenance():
    text = "Management expects margin to reach 25%."
    draft = _draft(text)
    outcome = await DraftRepairer(None, client=_RewriteClient()).repair(
        draft, _qa(text), SimpleNamespace(get=lambda _eid: None), AnalyticsBundle())

    statement = outcome.draft.all_statements[0]
    assert statement.text == "Management described margin as a monitoring priority."
    assert statement.evidence_ids == ("doc",)
    assert outcome.events[0].action == "llm_rewrite"


async def test_unavailable_llm_omits_only_the_unsafe_statement():
    text = "Management expects margin to reach 25%."
    draft = _draft(text)
    outcome = await DraftRepairer(None).repair(
        draft, _qa(text), SimpleNamespace(get=lambda _eid: None), AnalyticsBundle())

    assert outcome.draft.all_statements == ()
    assert outcome.events[0].action == "deterministic_omit"
    assert outcome.llm_error == "no LLM model configured"


async def test_llm_rewrite_with_a_new_number_is_rejected_and_omitted():
    class UnsafeClient:
        async def complete_json(self, _system: str, _prompt: str) -> LLMJSONResponse:
            return LLMJSONResponse(payload={"items": [{
                "index": 0,
                "action": "rewrite",
                "rewritten_text": "Management expects margin to reach 30%.",
            }]}, input_tokens=1, output_tokens=1)

    text = "Management expects margin to reach 25%."
    outcome = await DraftRepairer(None, client=UnsafeClient()).repair(
        _draft(text), _qa(text), SimpleNamespace(get=lambda _eid: None), AnalyticsBundle())

    assert outcome.draft.all_statements == ()
    assert outcome.events[0].action == "deterministic_omit"
