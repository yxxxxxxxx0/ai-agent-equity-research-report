"""Compact report: brief, technical appendix, then the cited references on their own page."""

import datetime as dt

from pypdf import PdfReader

from eq_report.domain.enums import ClaimType, ReportSection
from eq_report.domain.qa import QAResult
from eq_report.domain.report import Citation, ReportDraft, ReportSectionDraft, Statement
from eq_report.rendering import compact_renderer, technical_appendix
from eq_report.rendering.json_writer import write_report_json


def _report(tmp_path):
    draft = ReportDraft(
        report_run_id="run", company="Example Corp", ticker="EX",
        report_date=dt.date(2026, 9, 30), objective="test", title="Example",
        sections=(ReportSectionDraft(
            section=ReportSection.COMPANY_SNAPSHOT, title="Company Snapshot",
            summary="Example Corp makes widgets.", summary_citation_refs=(7,),
            statements=(Statement(text="Revenue grew 10%.", claim_type=ClaimType.CONFIRMED_FACT,
                                  citation_refs=(7,)),),
        ),),
        citations=(Citation(ref_number=7, evidence_id="ev", text="Example filing - 2026-08-01",
                            source_url="https://example.com/filing"),),
    )
    return write_report_json(tmp_path, draft, QAResult())


def _fake_fetch(*_a, **_k):
    day, rows = dt.date(2026, 5, 1), []
    for i in range(220):
        day += dt.timedelta(days=1)
        if day.weekday() < 5:
            price = 100 + i * 0.3
            rows.append((day, price, price + 1, price - 1, price + 0.5, 1_000_000))
    return rows, "test"


def _pages(pdf):
    return [p.extract_text() for p in PdfReader(str(pdf)).pages]


def test_references_land_on_page_three(tmp_path, monkeypatch):
    monkeypatch.setattr(technical_appendix, "_fetch_bloomberg_ohlcv", _fake_fetch)
    out = compact_renderer.render_compact_report(_report(tmp_path), tmp_path / "c.pdf")
    pages = _pages(out)
    assert len(pages) == 3
    assert "Megaannum Technology Limited" in pages[0].title() or "MEGAANNUM" in pages[0].upper()
    assert "References" in pages[2] and "https://example.com/filing" in pages[2]
    assert "Page 3 of 3" in pages[2]


def test_appendix_failure_still_ships_brief_and_references(tmp_path, monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("no market data")
    monkeypatch.setattr(technical_appendix, "_fetch_bloomberg_ohlcv", boom)
    out = compact_renderer.render_compact_report(_report(tmp_path), tmp_path / "c.pdf")
    pages = _pages(out)
    assert len(pages) == 2 and "References" in pages[1] and "Page 2 of 2" in pages[1]
    assert not list(tmp_path.glob("c_*.pdf"))
