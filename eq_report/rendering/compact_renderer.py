"""Render a dense two-page version of a completed report.

The first page is an investment brief built from the existing structured
ReportDraft.  The second is the standard technical dashboard, preserving the
same calculations and chart captions as the full report.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import simpleSplit
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from ..config import ProviderCredentials, Settings
from ..domain.enums import ReportSection
from ..domain.report import MetricTable, ReportDraft
from ..logging_setup import get_logger
from ..synthesis.terminology import claim_fingerprint
from .brand import BAND_H, brand_band, footer
from .json_loader import load_report_json
from .pdf_renderer import ACCENT, ACCENT_LINE, ACCENT_SOFT, HAIRLINE, INK, MUTED, ZEBRA
from .technical_appendix import build_technical_appendix_pdf, merge_technical_appendix

logger = get_logger("rendering.compact")

# The compact brief is a denser layout of the same report, so it draws from
# the full report's own palette (see pdf_renderer.py) rather than a second,
# unrelated colour system - the two are meant to look like one house style,
# not two different products.
PAPER = colors.white


#: One body size and leading for every block on the page, so no box reads
#: smaller than its neighbour; section bars and the masthead stay distinct.
BODY_SIZE = 8.8
BODY_LEADING = 12.0
#: Card padding: first baseline sits 13pt under the card top (see panel()),
#: so a card needs this much beyond its lines to keep descenders inside.
_CARD_PAD = 12.0
_BULLET_INDENT = 8.0
_ITEM_GAP = 3.5


def _lines(text: str, width: float, font: str = "Helvetica", size: float = BODY_SIZE) -> list[str]:
    return simpleSplit(text, font, size, width)


def _paragraph(c: Canvas, x: float, y: float, width: float, text: str, *,
               size: float = 7.2, leading: float = 9.0, max_lines: int | None = None,
               color: colors.Color = INK) -> float:
    lines = _lines(text, width, "Helvetica", size)
    if max_lines is not None and len(lines) > max_lines:
        lines = lines[:max_lines]
        lines[-1] = lines[-1].rstrip(".,;: ") + "..."
    c.setFillColor(color)
    c.setFont("Helvetica", size)
    for line in lines:
        c.drawString(x, y, line)
        y -= leading
    return y


def _overlaps(left: frozenset[str], right: frozenset[str]) -> bool:
    """Near-duplicate test for the compact page: similar overall, or one
    point largely contained in another (a short restatement of a long one)."""
    if not left or not right:
        return False
    shared = len(left & right)
    return shared / len(left | right) >= 0.5 or shared / min(len(left), len(right)) >= 0.7


def _cited_text(text: str, refs: tuple[int, ...]) -> str:
    """Append the full report's stable source references to compact prose."""
    return text + (" " + "".join(f"[{ref}]" for ref in sorted(set(refs))) if refs else "")


def _bullets(c: Canvas, x: float, y: float, width: float, texts: list[str]) -> float:
    """Render each point in full at the body size; the card is sized to fit."""
    for text in texts:
        if not text:
            continue
        c.setFillColor(ACCENT)
        # Align the marker to the first line's baseline instead of to the
        # preceding block's leading; this keeps every bullet visually level.
        c.circle(x + 2, y + 1.1, 1.1, fill=1, stroke=0)
        y = _paragraph(c, x + _BULLET_INDENT, y, width - _BULLET_INDENT, " ".join(text.split()),
                       size=BODY_SIZE, leading=BODY_LEADING)
        y -= _ITEM_GAP
    return y


#: Height of a filled section-title bar, and the light card background drawn
#: beneath it - the two together give each section the boxed-grid look
#: (solid title bar, framed content box) instead of a bare rule under text.
_BAR_H = 17.0


def _heading(c: Canvas, x: float, y: float, label: str, *, width: float = 535.0) -> float:
    """A solid accent title bar, the same identity colour as the full
    report's masthead band (see pdf_renderer._masthead_band) rather than a
    second, unrelated accent colour."""
    c.setFillColor(ACCENT)
    c.roundRect(x, y - _BAR_H + 3, width, _BAR_H, 3, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 9.5)
    c.drawString(x + 7, y - 9, label.upper())
    return y - _BAR_H - 5


def _card(c: Canvas, x: float, y_top: float, width: float, height: float) -> None:
    """A light bordered box under a section bar, framing its content -
    the same zebra tint and hairline rule the full report's tables use."""
    c.setFillColor(ZEBRA)
    c.setStrokeColor(ACCENT_LINE)
    c.setLineWidth(0.6)
    c.rect(x, y_top - height, width, height, fill=1, stroke=1)


def _bullet_lines_height(texts: list[str], width: float) -> float:
    """Exact vertical space `_bullets` uses at this width (0 if nothing to draw)."""
    items = [t for t in texts if t]
    if not items:
        return 0.0
    lines = sum(len(_lines(" ".join(t.split()), width - _BULLET_INDENT)) for t in items)
    return lines * BODY_LEADING + len(items) * _ITEM_GAP


def _bullet_block_height(texts: list[str], width: float) -> float:
    """Card height that contains every wrapped line of `_bullets(texts)`."""
    body = _bullet_lines_height(texts, width)
    return body + _CARD_PAD if body else 0.0


_ROW_H = 14.0


def _table_block_height(table: MetricTable | None) -> float:
    """Exact height `_table` descends by, for sizing its card ahead of time."""
    if table is None or not table.columns:
        return 0.0
    return 13 + _ROW_H + min(len(table.rows), 4) * _ROW_H + 4


def _paragraph_block_height(text: str, width: float) -> float:
    """Card height that contains every wrapped line of `_paragraph(text)`."""
    if not text or not text.strip():
        return 0.0
    return len(_lines(text, width)) * BODY_LEADING + _CARD_PAD


def _key_data(c: Canvas, draft: ReportDraft, y: float) -> float:
    """Draw the report's key data in a four-block grid."""
    if not draft.key_data:
        return y
    groups = list(draft.key_data.groups)
    x_positions = (30.0, 306.0)
    row_height = 58.0
    for index, group in enumerate(groups[:4]):
        x = x_positions[index % 2]
        top = y - (index // 2) * row_height
        c.setFillColor(ZEBRA)
        c.roundRect(x, top - 49, 258, 49, 3, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont("Helvetica-Bold", 7.0)
        c.drawString(x + 6, top - 10, group.title)
        for item_index, item in enumerate(group.items[:6]):
            column = item_index // 3
            row = item_index % 3
            item_x = x + 6 + column * 126
            item_y = top - 21 - row * 8.6
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 5.9)
            c.drawString(item_x, item_y, item.label)
            c.setFillColor(INK)
            c.setFont("Helvetica-Bold", 6.2)
            c.drawRightString(item_x + 118, item_y, item.value)
    return y - 122


def _fit_cell(value: object, width: float, *, font: str = "Helvetica", size: float = 5.8) -> str:
    """Shorten a table cell before it can intrude into its neighbour."""
    text = str(value)
    if stringWidth(text, font, size) <= width:
        return text
    while text and stringWidth(f"{text}...", font, size) > width:
        text = text[:-1]
    return f"{text.rstrip()}..." if text else "..."


def _table(c: Canvas, x: float, y: float, width: float, table: MetricTable) -> float:
    """Draw a deliberately small table with at most four rows."""
    columns = table.columns[:4]
    if not columns:
        return y
    c.setFillColor(ACCENT)
    c.setFont("Helvetica-Bold", BODY_SIZE)
    c.drawString(x, y, table.title)
    y -= 13
    if len(columns) == 4:
        col_widths = (width * 0.40, width * 0.20, width * 0.20, width * 0.20)
    elif len(columns) == 3:
        col_widths = (width * 0.36, width * 0.32, width * 0.32)
    else:
        col_widths = tuple(width / len(columns) for _ in columns)
    col_starts: list[float] = [x]
    for col_width in col_widths[:-1]:
        col_starts.append(col_starts[-1] + col_width)
    c.setFillColor(ACCENT)
    c.rect(x, y - _ROW_H, width, _ROW_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", BODY_SIZE)
    for index, column in enumerate(columns):
        c.drawString(col_starts[index] + 3, y - 10,
                     _fit_cell(column, col_widths[index] - 4, font="Helvetica-Bold", size=BODY_SIZE))
    y -= _ROW_H
    for row_index, row in enumerate(table.rows[:4]):
        if row.emphasis:
            c.setFillColor(ACCENT_SOFT)
            c.rect(x, y - _ROW_H, width, _ROW_H, fill=1, stroke=0)
        elif row_index % 2 == 0:
            c.setFillColor(ZEBRA)
            c.rect(x, y - _ROW_H, width, _ROW_H, fill=1, stroke=0)
        c.setFillColor(ACCENT if row.emphasis else INK)
        font = "Helvetica-Bold" if row.emphasis else "Helvetica"
        c.setFont(font, BODY_SIZE)
        values = (row.label, *row.cells)[:len(columns)]
        for index, value in enumerate(values):
            c.drawString(col_starts[index] + 3, y - 10,
                         _fit_cell(value, col_widths[index] - 4, font=font, size=BODY_SIZE))
        y -= _ROW_H
        if row_index < min(len(table.rows), 4) - 1:
            c.setStrokeColor(HAIRLINE)
            c.setLineWidth(0.4)
            c.line(x, y, x + width, y)
    return y - 4


#: Vertical gap left between one block and the next.
_BLOCK_GAP = 12.0


def _render_brief(draft: ReportDraft, output_pdf: Path, *, total_pages: int,
                  max_bullets: int = 8) -> tuple[list[int], bool]:
    """Render page one as a brief in the full report's own house style; returns the
    source numbers it cites (for the references page) and whether every block fit
    above the footer - if not, the caller retries with a smaller ``max_bullets``.

    Laid out as a top-down flow of independent blocks rather than a fixed
    grid: a block with no supporting evidence is skipped entirely - no
    heading, no empty framed box - and every block after it moves up to
    fill the gap, exactly like the full report drops a section it cannot
    write rather than printing a heading over a placeholder.
    """
    page_w, page_h = A4
    c = Canvas(str(output_pdf), pagesize=A4)
    c.setTitle(f"{draft.ticker} compact equity brief")
    left_x, gap, full_w = 18.0, 4.0, page_w - 36.0
    col_w = (full_w - gap) / 2
    right_x = left_x + col_w + gap

    company = next((s for s in draft.sections if s.title == "Company Snapshot"), None)
    # A thin/degraded run may omit Key Takeaways entirely - the brief should
    # still render with that block empty, not crash (StopIteration on a
    # generator with no default was the previous bug here).
    takeaways_section = next(
        (s.section for s in draft.sections if s.title == "Key Takeaways"), None)
    takeaways = draft.section(takeaways_section) if takeaways_section else None
    competitive = next((s for s in draft.sections if s.title == "Competitive Landscape"), None)
    financial = next((s for s in draft.sections if "Financial Performance" in s.title), None)
    drivers = next((s for s in draft.sections if "Operating Drivers" in s.title), None)
    # By .section id, not a title substring: RISKS, CATALYSTS and
    # WHAT_MATTERS_NEXT all happen to contain "Monitoring" in their printed
    # titles, so a substring match here would pick whichever came first.
    risks = next((s for s in draft.sections if s.section == ReportSection.RISKS), None)
    monitoring = next((s for s in draft.sections if s.section == ReportSection.WHAT_MATTERS_NEXT), None)
    recent = next((s for s in draft.sections if s.title == "Recent Developments"), None)

    brand_band(c, page_w, page_h, "Compact Equity Brief",
               f"REPORT DATE: {draft.report_date.isoformat()}")
    name = re.sub(rf"^{re.escape(draft.ticker or '')}\s*[\u2014\u2013-]\s*", "", draft.company)
    c.setFillColor(INK)
    c.setFont("Helvetica-Bold", 15)
    c.drawString(left_x, page_h - BAND_H - 24, f"{name} ({draft.ticker})")
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 8.8)
    c.drawString(left_x, page_h - BAND_H - 37, "Equity research  |  compact brief")
    c.setStrokeColor(ACCENT_LINE)
    c.setLineWidth(1.0)
    c.line(left_x, page_h - BAND_H - 44, page_w - left_x, page_h - BAND_H - 44)

    def panel(x: float, top: float, width: float, height: float, title: str) -> float:
        bar_y = _heading(c, x, top, title, width=width)
        _card(c, x, bar_y + 5, width, height)
        return bar_y - 8

    def draw_full(y_top: float, title: str, height: float, draw) -> float:
        """One full-width block, or nothing at all if it has no content."""
        if height <= 0:
            return y_top
        content_y = panel(left_x, y_top, full_w, height, title)
        draw(left_x + 5, content_y, full_w - 10)
        return y_top - _BAR_H - height - _BLOCK_GAP

    def draw_pair(
        y_top: float, title_l: str, height_l: float, draw_l,
        title_r: str, height_r: float, draw_r,
    ) -> float:
        """Two side-by-side blocks - only paired when both have content; a
        lone survivor renders full width, and an empty pair is skipped."""
        if height_l <= 0 and height_r <= 0:
            return y_top
        if height_l > 0 and height_r > 0:
            h = max(height_l, height_r)
            content_l = panel(left_x, y_top, col_w, h, title_l)
            draw_l(left_x + 5, content_l, col_w - 10)
            content_r = panel(right_x, y_top, col_w, h, title_r)
            draw_r(right_x + 5, content_r, col_w - 10)
            return y_top - _BAR_H - h - _BLOCK_GAP
        if height_l > 0:
            return draw_full(y_top, title_l, height_l, draw_l)
        return draw_full(y_top, title_r, height_r, draw_r)

    # -- content, and only the content that actually exists ----------------
    intro_texts = [t for t in [
        _cited_text(company.summary, company.summary_citation_refs) if company else "",
        _cited_text(company.statements[0].text, company.statements[0].citation_refs)
        if company and company.statements else "",
    ] if t]
    snapshot_texts = [_cited_text(s.text, s.citation_refs)
                      for s in (takeaways.statements[:2] if takeaways else ())]
    financial_table = financial.tables[0] if financial and financial.tables else None
    metric_notes = [_cited_text(s.text, s.citation_refs)
                    for s in (drivers.statements[:3] if drivers else ())]
    # One bullet per point: the financial standfirst, the first financial
    # statements, then the operating drivers behind them.
    financial_texts = [t for t in [
        _cited_text(financial.summary, financial.summary_citation_refs) if financial else "",
        *(_cited_text(s.text, s.citation_refs) for s in (financial.statements[:3] if financial else ())),
        *metric_notes,
    ] if t]
    competitive_texts = [t for t in [
        _cited_text(competitive.summary, competitive.summary_citation_refs) if competitive else "",
        _cited_text(competitive.statements[0].text, competitive.statements[0].citation_refs)
        if competitive and competitive.statements else "",
    ] if t]
    risk_texts = [t for t in [
        _cited_text(risks.statements[0].text, risks.statements[0].citation_refs)
        if risks and risks.statements else "",
        _cited_text(risks.statements[1].text, risks.statements[1].citation_refs)
        if risks and len(risks.statements) > 1 else "",
    ] if t]
    watch_texts = [_cited_text(s.text, s.citation_refs)
                   for s in (monitoring.statements[:2] if monitoring else ())]
    recent_texts = [_cited_text(s.text, s.citation_refs)
                    for s in (recent.statements[:2] if recent else ())]

    intro_texts, snapshot_texts = intro_texts[:max_bullets], snapshot_texts[:max_bullets]
    financial_texts = financial_texts[:max_bullets]

    shown = [
        t for t in intro_texts + snapshot_texts + metric_notes + competitive_texts
        + risk_texts + watch_texts + recent_texts + financial_texts if t
    ]
    fingerprints = [claim_fingerprint(t) for t in shown]

    def repeats_something(text: str) -> bool:
        fp = claim_fingerprint(text)
        return any(_overlaps(fp, other) for other in fingerprints)

    # Sections with no block of their own (e.g. Valuation) go first, from
    # their first statement; other sections contribute what their block
    # did not show. A point that restates one already on the page is skipped.
    featured = {ReportSection.COMPANY_SNAPSHOT, ReportSection.KEY_TAKEAWAYS,
                ReportSection.FINANCIALS, ReportSection.OPERATING_DRIVERS,
                ReportSection.COMPETITIVE_LANDSCAPE, ReportSection.RISKS,
                ReportSection.WHAT_MATTERS_NEXT, ReportSection.RECENT_DEVELOPMENTS}
    ordered = [s for s in draft.sections if s.section not in featured] + \
              [s for s in draft.sections if s.section in featured]
    candidates: list[str] = []
    for section in ordered:
        start = 0 if section.section not in featured else 2
        for statement in section.statements[start:]:
            text = _cited_text(statement.text, statement.citation_refs)
            if repeats_something(text):
                continue
            candidates.append(text)
            fingerprints.append(claim_fingerprint(text))
    candidates = candidates[:12]

    # -- layout: each block reports its own height, 0 meaning "skip me" ----
    y = page_h - BAND_H - 58

    full_inner, col_inner = full_w - 10, col_w - 10

    y = draw_full(
        y, "Company Overview", _bullet_block_height(intro_texts, full_inner),
        lambda x, yy, w: _bullets(c, x, yy, w, intro_texts),
    )
    y = draw_full(
        y, "Investment Snapshot", _bullet_block_height(snapshot_texts, full_inner),
        lambda x, yy, w: _bullets(c, x, yy, w, snapshot_texts),
    )

    def draw_financial_metrics(x: float, yy: float, w: float) -> None:
        _table(c, x, yy, w, financial_table)

    table_h = _table_block_height(financial_table)
    financial_metrics_h = table_h + _CARD_PAD if table_h else 0.0
    # A lone survivor renders full width, so size it for that width instead.
    bullets_w = col_inner if financial_metrics_h else full_inner
    y = draw_pair(
        y, "Key Financial Metrics", financial_metrics_h, draw_financial_metrics,
        "Financial Performance & Operating Drivers",
        _bullet_block_height(financial_texts, bullets_w),
        lambda x, yy, w: _bullets(c, x, yy, w, financial_texts),
    )

    competitive_risk_texts = (competitive_texts + risk_texts)[:max_bullets]
    watch_recent_texts = (watch_texts + recent_texts)[:max_bullets]
    both = bool(competitive_risk_texts) and bool(watch_recent_texts)
    pair_w = col_inner if both else full_inner
    y = draw_pair(
        y,
        "Competitive Landscape & Business Risk",
        _bullet_block_height(competitive_risk_texts, pair_w),
        lambda x, yy, w: _bullets(c, x, yy, w, competitive_risk_texts),
        "What Matters Next & Recent Developments",
        _bullet_block_height(watch_recent_texts, pair_w),
        lambda x, yy, w: _bullets(c, x, yy, w, watch_recent_texts),
    )

    # Fill the remaining page only with points that fit entirely in the box.
    bottom = 40.0
    fits = y + _BLOCK_GAP >= bottom
    available = y - _BAR_H - bottom
    fitted: list[str] = []
    for text in candidates:
        if _bullet_block_height(fitted + [text], full_inner) > available:
            break
        fitted.append(text)
    y = draw_full(
        y, "Additional Evidence", _bullet_block_height(fitted, full_inner),
        lambda x, yy, w: _bullets(c, x, yy, w, fitted),
    )

    footer(c, page_w, f"Page 1 of {total_pages} | Compact version of the full structured report")
    c.save()
    refs = sorted({int(n) for text in shown + fitted for n in re.findall(r"\[(\d+)\]", text)})
    return refs, fits


def _fit_brief(draft: ReportDraft, output_pdf: Path, *, total_pages: int) -> list[int]:
    """Render page one, trimming bullets per block until nothing runs into the footer."""
    for limit in range(8, 0, -1):
        refs, fits = _render_brief(draft, output_pdf, total_pages=total_pages, max_bullets=limit)
        if fits:
            break
    return refs


_REF_SIZE, _REF_LEADING, _REF_URL_LEADING = 8.6, 11.5, 9.5
_REF_TOP = 842.0 - BAND_H - 62
_REF_BOTTOM = 44.0


def _wrap_chars(text: str, width: float, font: str, size: float) -> list[str]:
    """Wrap on width alone (a URL has no spaces to break at)."""
    lines, line = [], ""
    for ch in text:
        if line and stringWidth(line + ch, font, size) > width:
            lines.append(line)
            line = ""
        line += ch
    return lines + ([line] if line else [])


def _reference_pages(draft: ReportDraft, refs: list[int]) -> list[list[tuple[int, list[str], list[str]]]]:
    """Cited sources laid out into pages of (number, text lines, url lines)."""
    by_number = {c.ref_number: c for c in draft.citations}
    text_w = A4[0] - 36 - 34
    pages: list[list[tuple[int, list[str], list[str]]]] = [[]]
    y = _REF_TOP
    for number in refs:
        cite = by_number.get(number)
        if cite is None:
            continue
        text = simpleSplit(" ".join(cite.text.split()), "Helvetica", _REF_SIZE, text_w)[:4]
        url = _wrap_chars(cite.source_url or "", text_w, "Helvetica", 7.4)[:2]
        height = len(text) * _REF_LEADING + len(url) * _REF_URL_LEADING + 9
        if y - height < _REF_BOTTOM and pages[-1]:
            pages.append([])
            y = _REF_TOP
        pages[-1].append((number, text, url))
        y -= height
    return pages if pages[0] else []


def _render_references(draft: ReportDraft, pages, output_pdf: Path, *,
                       first_page: int, total_pages: int) -> None:
    page_w, page_h = A4
    c = Canvas(str(output_pdf), pagesize=A4)
    c.setTitle(f"{draft.ticker} compact equity brief - references")
    for index, entries in enumerate(pages):
        brand_band(c, page_w, page_h, "References", "SOURCES CITED IN THIS BRIEF")
        c.setFillColor(MUTED)
        c.setFont("Helvetica", 8.8)
        c.drawString(18, page_h - BAND_H - 24,
                     f"{draft.ticker}  |  numbering matches the full report"
                     + (f"  |  continued ({index + 1} of {len(pages)})" if len(pages) > 1 else ""))
        y = _REF_TOP
        for row, (number, text, url) in enumerate(entries):
            height = len(text) * _REF_LEADING + len(url) * _REF_URL_LEADING + 9
            if row % 2 == 0:
                c.setFillColor(ZEBRA)
                c.rect(18, y - height + 5, page_w - 36, height, fill=1, stroke=0)
            c.setFillColor(ACCENT)
            c.setFont("Helvetica-Bold", _REF_SIZE)
            c.drawString(24, y - 6, f"[{number}]")
            c.setFillColor(INK)
            c.setFont("Helvetica", _REF_SIZE)
            ty = y - 6
            for line in text:
                c.drawString(56, ty, line)
                ty -= _REF_LEADING
            c.setFillColor(MUTED)
            c.setFont("Helvetica", 7.4)
            for line in url:
                c.drawString(56, ty + 1, line)
                ty -= _REF_URL_LEADING
            y -= height
        footer(c, page_w, f"Page {first_page + index} of {total_pages} | References")
        c.showPage()
    c.save()


def render_compact_report(
    report_json: Path | str,
    output_pdf: Path | str,
    *,
    credentials: ProviderCredentials | None = None,
    timeout: int = 30,
) -> Path:
    """Write the compact report: brief, technical analysis, then the references.

    The pages are independent: the brief is built entirely from the
    already-validated ReportDraft, while the technical appendix makes its own
    live OHLCV fetch (see technical_appendix.py) and can fail for reasons that
    have nothing to do with the brief - a data-availability gap at MegaAPI,
    not a defect in the report itself. A live-fetch failure there must not
    also destroy the brief; the references then follow the brief directly
    instead of raising, the same "partial data still yields partial output"
    policy the rest of the pipeline follows.
    """
    report_json = Path(report_json)
    output_pdf = Path(output_pdf)
    draft, _ = load_report_json(report_json)
    stem = output_pdf.stem
    brief_pdf = output_pdf.with_name(f"{stem}_brief.pdf")
    appendix_pdf = output_pdf.with_name(f"{stem}_appendix.pdf")
    refs_pdf = output_pdf.with_name(f"{stem}_refs.pdf")
    merged_pdf = output_pdf.with_name(f"{stem}_merged.pdf")
    parts = [brief_pdf]
    try:
        # Pass 1 finds which sources the brief cites; pass 2 draws it with the
        # true page count once the appendix and references are known.
        refs = _fit_brief(draft, brief_pdf, total_pages=1)
        pages = _reference_pages(draft, refs)
        has_appendix = True
        try:
            build_technical_appendix_pdf(
                draft.ticker or draft.company, draft.report_date, appendix_pdf,
                page_label=f"Page 2 of {2 + len(pages)}", credentials=credentials, timeout=timeout,
            )
            parts.append(appendix_pdf)
        except Exception as exc:  # noqa: BLE001 - the brief must still ship
            has_appendix = False
            logger.warning(
                "Technical appendix unavailable (%s: %s); the references follow the brief.",
                type(exc).__name__, exc,
            )
        total = 1 + has_appendix + len(pages)
        _fit_brief(draft, brief_pdf, total_pages=total)
        if pages:
            _render_references(draft, pages, refs_pdf, first_page=2 + has_appendix, total_pages=total)
            parts.append(refs_pdf)
        merged = parts[0]
        for extra in parts[1:]:
            merge_technical_appendix(merged, extra, merged_pdf)
            merged = merged_pdf
        merged.replace(output_pdf)
        return output_pdf
    finally:
        for path in (brief_pdf, appendix_pdf, refs_pdf, merged_pdf):
            path.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render a compact two-page equity brief.")
    parser.add_argument("report_json", type=Path)
    parser.add_argument("output_pdf", type=Path)
    args = parser.parse_args()
    settings = Settings.from_env()
    print(render_compact_report(
        args.report_json, args.output_pdf,
        credentials=settings.credentials,
        timeout=settings.provider_timeout_seconds,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
