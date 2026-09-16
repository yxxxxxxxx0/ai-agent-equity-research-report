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
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen.canvas import Canvas

from ..domain.report import MetricTable, ReportDraft
from ..config import ProviderCredentials, Settings
from .json_loader import load_report_json
from .technical_appendix import append_technical_appendix


NAVY = colors.HexColor("#12395e")
INK = colors.HexColor("#1a1a1a")
MUTED = colors.HexColor("#5c6470")
RULE = colors.HexColor("#c8ccd4")
PALE = colors.HexColor("#eef1f5")
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
        c.setFillColor(NAVY)
        # Align the marker to the first line's baseline instead of to the
        # preceding block's leading; this keeps every bullet visually level.
        c.circle(x + 2, y + 1.1, 1.1, fill=1, stroke=0)
        y = _paragraph(c, x + 8, y, width - 8, _short(text, max_words),
                       size=size, leading=size + 1.25, max_lines=max_lines)
        y -= 2
    return y


def _heading(c: Canvas, x: float, y: float, label: str, *, width: float = 535.0) -> float:
    """A full-width section rule gives each two-column block the same rhythm."""
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 8.2)
    c.drawString(x, y, label.upper())
    c.setStrokeColor(RULE)
    c.line(x, y - 3, x + width, y - 3)
    return y - 13


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
    c.setFillColor(NAVY)
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
    c.setFillColor(NAVY)
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
    page_w, page_h = A4
    c = Canvas(str(output_pdf), pagesize=A4)
    c.setTitle(f"{draft.ticker} compact equity brief")

    c.setFillColor(NAVY)
    c.rect(0, page_h - 58, page_w, 58, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(30, page_h - 33, f"{draft.company} ({draft.ticker})")
    c.setFont("Helvetica", 7.5)
    c.drawRightString(page_w - 30, page_h - 31, f"Compact Equity Brief | {draft.report_date.isoformat()}")

    y = page_h - 72
    if draft.contains_mock_data:
        c.setFillColor(WARN)
        c.roundRect(30, y - 20, page_w - 60, 20, 3, fill=1, stroke=0)
        c.setFillColor(colors.HexColor("#7a4a00"))
        c.setFont("Helvetica-Bold", 6.8)
        c.drawString(36, y - 13, "ILLUSTRATIVE MOCK DATA - NOT INVESTMENT RESEARCH. Technical charts use Bloomberg OHLCV via MegadataAPI.")
        y -= 29

    y = _key_data(c, draft, y)

    takeaways = draft.section(next(s.section for s in draft.sections if s.title == "Key Takeaways"))
    y = _heading(c, 30, y, "Investment snapshot")
    if takeaways:
        y = _bullets(c, 30, y, page_w - 60,
                     [statement.text for statement in takeaways.statements[:3]],
                     max_words=42, size=6.55, max_lines=2)

    company = next((s for s in draft.sections if s.title == "Company Snapshot"), None)
    competitive = next((s for s in draft.sections if s.title == "Competitive Landscape"), None)
    y = _heading(c, 30, y, "Company snapshot and competitive landscape")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 6.8)
    c.drawString(30, y, "Company snapshot")
    c.drawString(306, y, "Competitive landscape")
    company_y = _bullets(c, 30, y - 10, 252, [
        company.summary if company else "",
        company.statements[0].text if company and company.statements else "",
        company.statements[1].text if company and len(company.statements) > 1 else "",
    ], max_words=28)
    competitive_y = _bullets(c, 306, y - 10, 252, [
        competitive.summary if competitive else "",
        competitive.statements[0].text if competitive and competitive.statements else "",
        competitive.statements[1].text if competitive and len(competitive.statements) > 1 else "",
    ], max_words=28)
    y = min(company_y, competitive_y) - 6

    financial = next((s for s in draft.sections if "Financial Performance" in s.title), None)
    drivers = next((s for s in draft.sections if "Operating Drivers" in s.title), None)
    left_y = _heading(c, 30, y, "Reported results and drivers")
    right_y = left_y
    if financial and financial.tables:
        left_y = _table(c, 30, left_y, 258, financial.tables[0])
    if drivers and drivers.tables:
        right_y = _table(c, 306, right_y, 258, drivers.tables[0])
    y = min(left_y, right_y) - 3

    valuation = next((s for s in draft.sections if "Valuation" in s.title), None)
    monitoring = next((s for s in draft.sections if "Monitoring" in s.title), None)
    y = _heading(c, 30, y, "Valuation and watch items")
    left_x, right_x = 30.0, 306.0
    if valuation:
        y_left = _bullets(c, left_x, y, 250, [
            valuation.summary,
            valuation.statements[0].text if valuation.statements else "",
            valuation.statements[1].text if len(valuation.statements) > 1 else "",
        ], max_words=27)
    else:
        y_left = y
    watch = monitoring.statements if monitoring else ()
    if watch:
        y_right = _bullets(c, right_x, y, 250, [statement.text for statement in watch[:2]],
                           max_words=29)
    else:
        y_right = y

    y = min(y_left, y_right) - 6
    recent = next((s for s in draft.sections if s.title == "Recent Developments"), None)
    y = _heading(c, 30, y, "Recent developments and risk frame")
    c.setFillColor(NAVY)
    c.setFont("Helvetica-Bold", 6.8)
    c.drawString(left_x, y, "Developments")
    c.drawString(right_x, y, "Risks and catalysts")
    recent_y = _bullets(c, left_x, y - 10, 250,
                        [item.text for item in (recent.statements if recent else ())],
                        max_words=34)
    risk_y = _bullets(c, right_x, y - 10, 250, [
        recent.statements[3].text if recent and len(recent.statements) > 3 else "",
        takeaways.statements[5].text if takeaways and len(takeaways.statements) > 5 else "",
        takeaways.statements[3].text if takeaways and len(takeaways.statements) > 3 else "",
    ], max_words=34)

    peer_table = competitive.tables[0] if competitive and competitive.tables else None
    if peer_table:
        peer_y = _heading(c, 30, min(recent_y, risk_y) - 6, "Peer comparison")
        peer_y = _table(c, 30, peer_y, page_w - 60, peer_table) - 5
        c.setFillColor(NAVY)
        c.setFont("Helvetica-Bold", 6.3)
        c.drawString(left_x, peer_y, "What the comparison shows")
        c.drawString(right_x, peer_y, "What it cannot establish")
        comparison_texts = (
            [valuation.statements[0].text, valuation.statements[1].text]
            if valuation and len(valuation.statements) > 1
            else ["The peer table is a directional valuation and growth comparison."]
        )
        limitations_text = (
            "The comparison is not like-for-like: the available evidence does not provide "
            "peer segment mix, margins, earnings growth, product exposure, regional mix or "
            "capital intensity. It therefore cannot attribute the valuation gap to durable "
            "competitive advantage, autonomy optionality, energy economics or operating execution."
        )
        comparison_y = _bullets(c, left_x, peer_y - 9, 250, comparison_texts,
                                max_words=34, size=5.95, max_lines=2)
        limitation_y = _bullets(c, right_x, peer_y - 9, 250, [limitations_text],
                                max_words=38, size=5.95, max_lines=3)
        footer_y = min(comparison_y, limitation_y) - 3
    else:
        footer_y = min(recent_y, risk_y) - 6

    # Use otherwise-empty lower-page space for additional evidence, including
    # attributed KOL commentary. Keep the footer clear and typography stable.
    if footer_y > 105:
        used = {
            statement.text for section in draft.sections
            for statement in section.statements[:2]
        }
        candidates = [
            statement.text for section in draft.sections
            for statement in section.statements[2:]
            if statement.text not in used
        ]
        if candidates:
            extra_y = _heading(c, 30, footer_y, "Additional evidence and KOL perspectives")
            capacity = max(1, min(6, int((extra_y - 48) / 18)))
            footer_y = _bullets(
                c, 30, extra_y, page_w - 60, candidates[:capacity],
                max_words=40, size=6.2, max_lines=2,
            )

    gaps = list(draft.data_gaps)[:2]
    if gaps:
        c.setFillColor(MUTED)
        c.setFont("Helvetica-Bold", 6.3)
        if footer_y < 42:
            c.drawString(30, 31, "Key limitations: detailed sources, data gaps and peer-comparison caveats remain in the full report.")
        else:
            c.drawString(30, footer_y, "Key disclosed limitations:")
            for index, gap in enumerate(gaps):
                _paragraph(c, 126, footer_y - index * 8, page_w - 156, gap.description,
                           size=5.8, leading=7.0, max_lines=1, color=MUTED)
    c.setFillColor(MUTED)
    c.setFont("Helvetica", 5.8)
    c.drawRightString(page_w - 30, 17, "Page 1 of 2 | Compact version of the full structured report")
    c.save()


def render_compact_report(
    report_json: Path | str,
    output_pdf: Path | str,
    *,
    credentials: ProviderCredentials | None = None,
    timeout: int = 30,
) -> Path:
    """Write a two-page brief: one dense research page plus technical analysis."""
    report_json = Path(report_json)
    output_pdf = Path(output_pdf)
    draft, _ = load_report_json(report_json)
    brief_pdf = output_pdf.with_name(f"{output_pdf.stem}_brief.pdf")
    _render_brief(draft, brief_pdf)
    try:
        return append_technical_appendix(
            brief_pdf, output_pdf, draft.ticker or draft.company, draft.report_date,
            page_label="Page 2 of 2",
            credentials=credentials,
            timeout=timeout,
        )
    finally:
        brief_pdf.unlink(missing_ok=True)


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
