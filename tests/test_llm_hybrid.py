"""LLM rescue of rejected values and loosen-only QA triage: code, not the model, decides."""

import datetime as dt

import pytest

from eq_report.domain.enums import (
    Confidence, EvidenceCategory, EvidenceStatus, Severity, SourceType,
)
from eq_report.domain.evidence import EvidenceItem
from eq_report.domain.qa import QAFinding
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore
from eq_report.llm.client import LLMJSONResponse
from eq_report.normalisation.llm_rescue import rescue_values
from eq_report.qa.triage import triage_findings


class Client:
    def __init__(self, *payloads):
        self.payloads = list(payloads)

    async def complete_json(self, _system, _prompt, **_kw):
        return LLMJSONResponse(payload=self.payloads.pop(0), input_tokens=1, output_tokens=1)


def items(*rows):
    return {"items": [{"id": i, "span": s, "number": n} for i, s, n in rows]}


@pytest.mark.asyncio
async def test_rescue_accepts_verified_value_in_both_runs():
    good = items((0, "(215.4)M", -215_400_000), (1, "2.9 trillion dollars", 2.9e12))
    texts = {0: "(215.4)M", 1: "2.9 trillion dollars"}
    got = await rescue_values(texts, None, client=Client(good, good))
    assert got == {0: "-215400000", 1: "2900000000000"}


@pytest.mark.asyncio
async def test_rescue_rejects_invented_wrong_scale_span_and_disagreement():
    texts = {0: "about 12M", 1: "n/a 5", 2: "(3.0)M", 3: "7.5M"}
    run = items((0, "12M", 13_000_000),      # digits x scale != number
                (1, "five", 5),              # span not in text
                (2, "(3.0)M", 3_000_000),    # sign lost
                (3, "7.5M", 7_500_000))
    other = items((3, "7.5M", 7_000_000))    # second run disagrees
    assert await rescue_values(texts, None, client=Client(run, other)) == {}


def _reader():
    store = EvidenceStore(":memory:")
    store.save([EvidenceItem(
        evidence_id="ev_rev", report_run_id="run", company="Ex", ticker="EX",
        category=EvidenceCategory.FUNDAMENTAL, source_id="s", source_name="s",
        source_type=SourceType.COMPANY_FILING, retrieved_at=dt.datetime.now(dt.UTC),
        metric="revenue", value=24_100_000_000, unit="USD", currency="USD",
        confidence=Confidence.HIGH, status=EvidenceStatus.VALIDATED, is_canonical=True)])
    return EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Ex")


def finding(check="evidence.no_unsupported_numbers", text="Revenue was $24.1bn."):
    return QAFinding(check=check, severity=Severity.CRITICAL, message="m", subject=text)


def facts(number="$24.1bn", metric="revenue"):
    return {"facts": [{"number": number, "metric": metric, "period": None}]}


@pytest.mark.asyncio
async def test_triage_on_downgrades_only_when_figure_matches_store_in_both_runs():
    out = await triage_findings([finding()], _reader(), None, "on",
                                client=Client(facts(), facts()))
    assert out[0].severity is Severity.WARNING and "ev_rev" in out[0].message


@pytest.mark.asyncio
async def test_triage_shadow_and_mismatch_and_disagreement_change_nothing():
    reader = _reader()
    assert (await triage_findings([finding()], reader, None, "shadow",
                                  client=Client(facts(), facts())))[0].severity is Severity.CRITICAL
    wrong = finding(text="Revenue was $30bn.")
    assert (await triage_findings([wrong], reader, None, "on",
                                  client=Client(facts("$30bn"), facts("$30bn"))))[0].severity is Severity.CRITICAL
    assert (await triage_findings([finding()], reader, None, "on",
                                  client=Client(facts(), facts(metric=None))))[0].severity is Severity.CRITICAL


@pytest.mark.asyncio
async def test_triage_never_touches_integrity_checks():
    f = finding(check="numeric.recompute")
    out = await triage_findings([f], _reader(), None, "on", client=Client())
    assert out == [f]
