"""Render a dense two-page version of a completed report.

The first page is an investment brief built from the existing structured
ReportDraft.  The second is the standard technical dashboard, preserving the
same calculations and chart captions as the full report.
"""

from __future__ import annotations

import argparse
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
from .json_loader import load_report_json
from .technical_appendix import build_technical_appendix_pdf, merge_technical_appendix

logger = get_logger("rendering.compact")

# Match the supplied NVDA teaser: deep ink typography, warm rust section
# frames, and almost-white content cards rather than a dashboard palette.
NAVY = colors.HexColor("#102e4a")
INK = colors.HexColor("#171717")
MUTED = colors.HexColor("#6a625c")
RUST = colors.HexColor("#c94f10")
RULE = RUST
PALE = colors.HexColor("#f5f5f3")
WARM_PALE = colors.HexColor("#fae2d5")
WARN = colors.HexColor("#fdf3e0")


def _lines(text: str, width: float, font: str = "Helvetica", size: float = 7.2) -> list[str]:
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


def _short(text: str, max_words: int = 34) -> str:
    """Keep a report statement readable in a compact bullet layout."""
    words = " ".join(text.split()).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words]).rstrip(".,;:") + "..."


def _bullets(c: Canvas, x: float, y: float, width: float, texts: list[str], *,
             max_words: int = 34, size: float = 6.35, max_lines: int = 2) -> float:
    """Render concise evidence-led points without long narrative blocks."""
    for text in texts:
        if not text:
            continue
        c.setFillColor(RUST)
        # Align the marker to the first line's baseline instead of to the
        # preceding block's leading; this keeps every bullet visually level.
        c.circle(x + 2, y + 1.1, 1.1, fill=1, stroke=0)
        y = _paragraph(c, x + 8, y, width - 8, _short(text, max_words),
                       size=size, leading=size + 1.25, max_lines=max_lines)
        y -= 2
    return y


#: Height of a filled section-title bar, and the light card background drawn
#: beneath it - the two together give each section the boxed-grid look
#: (solid title bar, framed content box) instead of a bare rule under text.
_BAR_H = 14.0


def _heading(c: Canvas, x: float, y: float, label: str, *, width: float = 535.0) -> float:
    """A warm, rounded title bar modelled on the supplied teaser."""
    c.setFillColor(RUST)
    c.roundRect(x, y - _BAR_H + 3, width, _BAR_H, 7, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8.6)
    c.drawCentredString(x + width / 2, y - 7, label)
    return y - _BAR_H - 5


def _card(c: Canvas, x: float, y_top: float, width: float, height: float) -> None:
    """A light bordered box under a section bar, framing its content."""
    c.setFillColor(PALE)
    c.setStrokeColor(RULE)
    c.setLineWidth(0.6)
    c.rect(x, y_top - height, width, height, fill=1, stroke=1)


def _bullet_block_height(
    texts: list[str], *, size: float = 6.35, max_lines: int = 2, extra: float = 12.0,
) -> float:
    """Worst-case height a `_bullets` call can take, for drawing its card first.

    `_bullets` never exceeds `max_lines` per item (it truncates with an
    ellipsis), so this upper bound is exact enough to size a background card
    that always fully contains the text drawn on top of it afterward.
    """
    n = sum(1 for t in texts if t)
    return n * (max_lines * (size + 1.25) + 2) + extra


def _table_block_height(table: MetricTable | None) -> float:
    """Exact height `_table` descends by, for sizing its card ahead of time."""
    if table is None or not table.columns:
        return 0.0
    return 10 + 10 + min(len(table.rows), 4) * 10 + 4


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
        c.setFillColor(PALE)
        c.roundRect(x, top - 49, 258, 49, 3, fill=1, stroke=0)
        c.setFillColor(NAVY)
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
    c.setFillColor(RUST)
    c.setFont("Helvetica-Bold", 6.7)
    c.drawString(x, y, table.title)
    y -= 10
    if len(columns) == 4:
        col_widths = (width * 0.40, width * 0.20, width * 0.20, width * 0.20)
    elif len(columns) == 3:
        col_widths = (width * 0.36, width * 0.32, width * 0.32)
    else:
        col_widths = tuple(width / len(columns) for _ in columns)
    col_starts: list[float] = [x]
    for col_width in col_widths[:-1]:
        col_starts.append(col_starts[-1] + col_width)
    c.setFillColor(RUST)
    c.rect(x, y - 10, width, 10, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 5.7)
    for index, column in enumerate(columns):
        c.drawString(col_starts[index] + 2, y - 7,
                     _fit_cell(column, col_widths[index] - 4, font="Helvetica-Bold", size=5.7))
    y -= 10
    for row_index, row in enumerate(table.rows[:4]):
        if row_index % 2 == 0:
            c.setFillColor(PALE)
            c.rect(x, y - 10, width, 10, fill=1, stroke=0)
        c.setFillColor(INK)
        c.setFont("Helvetica-Bold" if row.emphasis else "Helvetica", 5.8)
        values = (row.label, *row.cells)[:len(columns)]
        for index, value in enumerate(values):
            c.drawString(col_starts[index] + 2, y - 7,
                         _fit_cell(value, col_widths[index] - 4,
                                   font="Helvetica-Bold" if row.emphasis else "Helvetica"))
        y -= 10
    return y - 4


def _render_brief(draft: ReportDraft, output_pdf: Path) -> None:
    """Render page one as the supplied teaser's stacked, rust-framed brief."""
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

    # Header: deliberately light and editorial, as in the reference PDF.
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_x, page_h - 28, "Compact Equity Brief")
    c.setFont("Helvetica", 13.5)
    c.setFillColor(INK)
    c.drawString(left_x, page_h - 48, f"{draft.company} ({draft.ticker})")
    c.setFont("Helvetica", 8.2)
    c.setFillColor(MUTED)
    c.drawString(left_x, page_h - 62, "Compact equity brief")
    c.setFillColor(WARM_PALE)
    c.roundRect(page_w - 178, page_h - 59, 160, 37, 9, fill=1, stroke=0)
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8.5)
    c.drawCentredString(page_w - 98, page_h - 37, f"REPORT DATE: {draft.report_date.isoformat()}")
    c.setStrokeColor(RUST)
    c.setLineWidth(1.3)
    c.line(left_x, page_h - 70, page_w - left_x, page_h - 70)

    def panel(x: float, top: float, width: float, height: float, title: str) -> float:
        bar_y = _heading(c, x, top, title, width=width)
        _card(c, x, bar_y + 5, width, height)
        return bar_y - 8

    # Full-width introduction and summary mirror the two opening teaser boxes.
    intro_texts = [
        company.summary if company else "",
        company.statements[0].text if company and company.statements else "",
    ]
    y = panel(left_x, page_h - 80, full_w, 94, "Company Overview")
    _bullets(c, left_x + 5, y, full_w - 10, intro_texts, max_words=70, size=7.1, max_lines=3)

    snapshot_texts = [s.text for s in (takeaways.statements[:2] if takeaways else ())]
    y = panel(left_x, page_h - 194, full_w, 65, "Investment Snapshot")
    _bullets(c, left_x + 5, y, full_w - 10, snapshot_texts, max_words=66, size=6.65, max_lines=2)

    # Two taller, side-by-side panels make room for data on the left and the
    # financial narrative on the right, matching the reference's focal row.
    financial_table = financial.tables[0] if financial and financial.tables else None
    y_left = panel(left_x, page_h - 278, col_w, 166, "Key Financial Metrics")
    table_bottom = _table(c, left_x + 5, y_left, col_w - 10, financial_table) if financial_table else y_left
    metric_notes = [s.text for s in (drivers.statements[:3] if drivers else ())]
    _bullets(c, left_x + 5, table_bottom - 5, col_w - 10, metric_notes, max_words=28, size=5.85, max_lines=2)
    y_right = panel(right_x, page_h - 278, col_w, 166, "Financial Performance & Operating Drivers")
    financial_text = " ".join(filter(None, [
        financial.summary if financial else "",
        financial.statements[0].text if financial and financial.statements else "",
    ]))
    _paragraph(c, right_x + 5, y_right, col_w - 10, financial_text,
               size=7.15, leading=8.6, max_lines=15)

    # The second two-up row follows the reference: landscape/risk left, a
    # checkable forward-monitoring panel right (rather than unsourced sentiment).
    competitive_texts = [
        competitive.summary if competitive else "",
        competitive.statements[0].text if competitive and competitive.statements else "",
    ]
    risk_texts = [
        risks.statements[0].text if risks and risks.statements else "",
        risks.statements[1].text if risks and len(risks.statements) > 1 else "",
    ]
    y_left = panel(left_x, page_h - 464, col_w, 152, "Competitive Landscape & Business Risk")
    _bullets(c, left_x + 5, y_left, col_w - 10, competitive_texts + risk_texts,
             max_words=32, size=6.5, max_lines=3)

    watch_texts = [s.text for s in ((monitoring.statements[:2] if monitoring else ()))]
    recent_texts = [s.text for s in (recent.statements[:2] if recent else ())]
    y_right = panel(right_x, page_h - 464, col_w, 152, "What Matters Next & Recent Developments")
    _bullets(c, right_x + 5, y_right, col_w - 10, watch_texts + recent_texts,
             max_words=34, size=6.5, max_lines=3)

    # The bottom full-width box follows the reference's news treatment, but
    # is populated only by report statements already supported by evidence.
    used = {text for text in intro_texts + snapshot_texts + competitive_texts + risk_texts + watch_texts + recent_texts if text}
    candidates = [
        statement.text for section in draft.sections for statement in section.statements[2:]
        if statement.text not in used
    ]
    y = panel(left_x, page_h - 634, full_w, 126, "Additional Evidence")
    _bullets(c, left_x + 5, y, full_w - 10, candidates[:5], max_words=62, size=6.3, max_lines=2)

    c.setFillColor(MUTED)
    c.setFont("Helvetica", 5.8)
    c.drawRightString(page_w - left_x, 17, "Page 1 of 2 | Compact version of the full structured report")
    c.save()


def render_compact_report(
    report_json: Path | str,
    output_pdf: Path | str,
    *,
    credentials: ProviderCredentials | None = None,
    timeout: int = 30,
) -> Path:
    """Write a two-page brief: one dense research page plus technical analysis.

    The two pages are independent: the brief is built entirely from the
    already-validated ReportDraft, while the technical appendix makes its own
    live OHLCV fetch (see technical_appendix.py) and can fail for reasons that
    have nothing to do with the brief - a data-availability gap at MegaAPI,
    not a defect in the report itself. A live-fetch failure there must not
    also destroy the brief page; it degrades to a one-page compact report
    instead of raising, the same "partial data still yields partial output"
    policy the rest of the pipeline follows.
    """
    report_json = Path(report_json)
    output_pdf = Path(output_pdf)
    draft, _ = load_report_json(report_json)
    brief_pdf = output_pdf.with_name(f"{output_pdf.stem}_brief.pdf")
    appendix_pdf = output_pdf.with_name(f"{output_pdf.stem}_appendix.pdf")
    _render_brief(draft, brief_pdf)
    try:
        build_technical_appendix_pdf(
            draft.ticker or draft.company, draft.report_date, appendix_pdf,
            page_label="Page 2 of 2", credentials=credentials, timeout=timeout,
        )
        return merge_technical_appendix(brief_pdf, appendix_pdf, output_pdf)
    except Exception as exc:  # noqa: BLE001 - the brief page must still ship
        logger.warning(
            "Technical appendix unavailable (%s: %s); shipping the one-page brief alone.",
            type(exc).__name__, exc,
        )
        brief_pdf.replace(output_pdf)
        return output_pdf
    finally:
        brief_pdf.unlink(missing_ok=True)
        appendix_pdf.unlink(missing_ok=True)


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
