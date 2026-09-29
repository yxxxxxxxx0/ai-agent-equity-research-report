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


def _entailment_qa(text: str, reason: str) -> QAResult:
    return QAResult(findings=(QAFinding(
        check="evidence.claim_not_entailed",
        severity=Severity.CRITICAL,
        message=f"Mapped source does not support the complete claim: {reason}",
        section=ReportSection.RECENT_DEVELOPMENTS.value,
        subject=text[:200],
        details={"claim_id": "s0c0"},
    ),))


async def test_unentailed_claim_is_trimmed_to_its_supported_part():
    seen: dict[str, str] = {}

    class TrimClient:
        async def complete_json(self, system: str, prompt: str) -> LLMJSONResponse:
            seen["prompt"] = prompt
            return LLMJSONResponse(payload={"items": [{
                "index": 0,
                "action": "rewrite",
                "rewritten_text": "One direct customer represented 16% of quarterly revenue.",
            }]}, input_tokens=1, output_tokens=1)

    text = ("One direct customer represented 16% of quarterly revenue, making Data Center "
            "growth dependent on that customer's purchases.")
    reason = "The 16% share is supported, but no source ties that customer to Data Center."
    outcome = await DraftRepairer(None, client=TrimClient()).repair(
        _draft(text), _entailment_qa(text, reason),
        SimpleNamespace(get=lambda _eid: None), AnalyticsBundle())

    assert reason in seen["prompt"]
    statement = outcome.draft.all_statements[0]
    assert statement.text == "One direct customer represented 16% of quarterly revenue."
    assert statement.evidence_ids == ("doc",)
    assert outcome.events[0].action == "llm_rewrite"


async def test_entailment_trim_that_adds_a_figure_is_rejected():
    class InventingClient:
        async def complete_json(self, _system: str, _prompt: str) -> LLMJSONResponse:
            return LLMJSONResponse(payload={"items": [{
                "index": 0,
                "action": "rewrite",
                "rewritten_text": "One direct customer represented 18% of quarterly revenue.",
            }]}, input_tokens=1, output_tokens=1)

    text = "One direct customer represented 16% of revenue, a key Data Center risk."
    outcome = await DraftRepairer(None, client=InventingClient()).repair(
        _draft(text), _entailment_qa(text, "The 16% share is supported only."),
        SimpleNamespace(get=lambda _eid: None), AnalyticsBundle())

    assert outcome.draft.all_statements == ()
    assert outcome.events[0].action == "deterministic_omit"


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
