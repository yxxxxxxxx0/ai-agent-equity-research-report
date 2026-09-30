"""The web UI's QA review collects findings, repair events and pipeline notices."""

import json
from types import SimpleNamespace

from eq_report.domain.enums import Severity
from eq_report.domain.qa import QAFinding, QAResult
from webui.app import _qa_review


def test_qa_review_collects_findings_repairs_and_notices(tmp_path):
    (tmp_path / "07_qa_repair.json").write_text(json.dumps({"attempts": [{
        "attempt": 1, "events": [{"section": "financials", "action": "llm_rewrite",
                                  "checks": ["evidence.claim_not_entailed"],
                                  "original_text": "long", "repaired_text": "short"}]}]}), encoding="utf-8")
    qa = QAResult(findings=(
        QAFinding(check="a", severity=Severity.CRITICAL, message="m1", section="risks", subject="s"),
        QAFinding(check="b", severity=Severity.WARNING, message="m2"),
        QAFinding(check="c", severity=Severity.INFO, message="m3"),
    ))
    run = SimpleNamespace(warnings=["appendix skipped"], errors=[{"message": "branch died"}])
    review = _qa_review(SimpleNamespace(qa_result=qa, run=run), tmp_path)
    assert review["counts"] == {"critical": 1, "warning": 1, "info": 1}
    assert review["repairs"][0]["repaired"] == "short"
    assert review["notices"] == {"warnings": ["appendix skipped"], "errors": ["branch died"]}


def test_qa_review_tolerates_no_qa_and_no_repair_log(tmp_path):
    review = _qa_review(SimpleNamespace(qa_result=None, run=None), tmp_path)
    assert review["counts"] == {"critical": 0, "warning": 0, "info": 0} and review["repairs"] == []
