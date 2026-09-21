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


#: Height of a filled section-title bar, and the light card background drawn
#: beneath it - the two together give each section the boxed-grid look
#: (solid title bar, framed content box) instead of a bare rule under text.
_BAR_H = 14.0


def _heading(c: Canvas, x: float, y: float, label: str, *, width: float = 535.0) -> float:
    """A solid-filled title bar spanning the section's full width."""
    c.setFillColor(NAVY)
    c.rect(x, y - _BAR_H + 3, width, _BAR_H, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8.0)
    c.drawString(x + 6, y - 7, label.upper())
    return y - _BAR_H - 5


def _card(c: Canvas, x: float, y_top: float, width: float, height: float) -> None:
    """A light bordered box under a section bar, framing its content."""
    c.setFillColor(colors.white)
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
    """A boxed-grid brief: a solid title bar plus a framed card per section,
    each card sized ahead of drawing so its background never clips the text
    painted on top of it afterward (see _bullet_block_height/_table_block_height)."""
    page_w, page_h = A4
    c = Canvas(str(output_pdf), pagesize=A4)
    c.setTitle(f"{draft.ticker} compact equity brief")
    left_x, right_x, col_w = 30.0, 306.0, 252.0
    full_w = page_w - 60

    c.setFillColor(NAVY)
    c.rect(0, page_h - 58, page_w, 58, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 17)
    c.drawString(30, page_h - 33, f"{draft.company} ({draft.ticker})")
    c.setFont("Helvetica", 7.5)
    c.drawRightString(page_w - 30, page_h - 31, f"Compact Equity Brief | {draft.report_date.isoformat()}")

    y = page_h - 72
    y = _key_data(c, draft, y)

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

    # -- Company overview (full width) - the reference's "Quick Introduction" -
    intro_texts = [
        company.summary if company else "",
        company.statements[0].text if company and company.statements else "",
    ]
    h = _bullet_block_height(intro_texts, size=6.55, max_lines=2)
    bar_y = _heading(c, left_x, y, "Company overview", width=full_w)
    _card(c, left_x, bar_y + 5, full_w, h)
    y = _bullets(c, left_x + 4, bar_y - 2, full_w - 8, intro_texts, max_words=42, size=6.55, max_lines=2)

    # -- Investment snapshot (full width) - the reference's "Summary" -
    snapshot_texts = [s.text for s in (takeaways.statements[:3] if takeaways else ())]
    h = _bullet_block_height(snapshot_texts, size=6.55, max_lines=2)
    bar_y = _heading(c, left_x, y - 4, "Investment snapshot", width=full_w)
    _card(c, left_x, bar_y + 5, full_w, h)
    y = _bullets(c, left_x + 4, bar_y - 2, full_w - 8, snapshot_texts, max_words=42, size=6.55, max_lines=2)

    # -- Key metrics & growth drivers (two columns of tables) -
    bar_y = _heading(c, left_x, y - 4, "Key metrics & growth drivers", width=full_w)
    financial_table = financial.tables[0] if financial and financial.tables else None
    drivers_table = drivers.tables[0] if drivers and drivers.tables else None
    h = max(_table_block_height(financial_table), _table_block_height(drivers_table), 14)
    _card(c, left_x, bar_y + 5, col_w, h)
    _card(c, right_x, bar_y + 5, col_w, h)
    left_y = _table(c, left_x + 4, bar_y - 2, col_w - 8, financial_table) if financial_table else bar_y
    right_y = _table(c, right_x + 4, bar_y - 2, col_w - 8, drivers_table) if drivers_table else bar_y
    y = min(left_y, right_y) - 6

    # -- Competitive landscape & business risk (two columns) -
    competitive_texts = [
        competitive.summary if competitive else "",
        competitive.statements[0].text if competitive and competitive.statements else "",
    ]
    risk_texts = [
        risks.statements[0].text if risks and risks.statements else "",
        risks.statements[1].text if risks and len(risks.statements) > 1 else "",
    ]
    h = max(_bullet_block_height(competitive_texts), _bullet_block_height(risk_texts))
    bar_y = _heading(c, left_x, y - 4, "Competitive landscape & business risk", width=full_w)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 6.6)
    c.drawString(left_x + 4, bar_y - 8, "Competitive landscape")
    c.drawString(right_x + 4, bar_y - 8, "Business risk")
    _card(c, left_x, bar_y + 5, col_w, h + 10)
    _card(c, right_x, bar_y + 5, col_w, h + 10)
    left_y = _bullets(c, left_x + 4, bar_y - 18, col_w - 8, competitive_texts, max_words=28)
    right_y = _bullets(c, right_x + 4, bar_y - 18, col_w - 8, risk_texts, max_words=28)
    y = min(left_y, right_y) - 6

    # -- What matters next & recent developments (two columns) -
    # "What matters next" (checkable, evidence-scoped forward catalysts) takes
    # the slot a sentiment/social box would otherwise occupy in this layout -
    # the report states no market sentiment gauge, by design.
    watch_texts = [s.text for s in ((monitoring.statements[:2] if monitoring else ()))]
    recent_texts = [s.text for s in (recent.statements[:2] if recent else ())]
    h = max(_bullet_block_height(watch_texts), _bullet_block_height(recent_texts))
    bar_y = _heading(c, left_x, y - 4, "What matters next & recent developments", width=full_w)
    c.setFillColor(NAVY); c.setFont("Helvetica-Bold", 6.6)
    c.drawString(left_x + 4, bar_y - 8, "What matters next")
    c.drawString(right_x + 4, bar_y - 8, "Recent developments")
    _card(c, left_x, bar_y + 5, col_w, h + 10)
    _card(c, right_x, bar_y + 5, col_w, h + 10)
    left_y = _bullets(c, left_x + 4, bar_y - 18, col_w - 8, watch_texts, max_words=29)
    right_y = _bullets(c, right_x + 4, bar_y - 18, col_w - 8, recent_texts, max_words=34)
    y = min(left_y, right_y) - 6

    # -- Peer comparison (full-width table, then a two-column caveat) -
    peer_table = competitive.tables[0] if competitive and competitive.tables else None
    if peer_table:
        bar_y = _heading(c, left_x, y - 4, "Peer comparison", width=full_w)
        limitations_text = (
            "The comparison is not like-for-like: the available evidence does not provide "
            "peer segment mix, margins, earnings growth, product exposure, regional mix or "
            "capital intensity, so it cannot attribute the valuation gap to any one driver."
        )
        h = _table_block_height(peer_table) + _bullet_block_height([limitations_text], size=5.95, max_lines=3)
        _card(c, left_x, bar_y + 5, full_w, h)
        table_y = _table(c, left_x + 4, bar_y - 2, full_w - 8, peer_table) - 3
        footer_y = _bullets(c, left_x + 4, table_y, full_w - 8, [limitations_text],
                            max_words=38, size=5.95, max_lines=3)
    else:
        footer_y = y

    # Use otherwise-empty lower-page space for additional cited evidence - the
    # reference's "News" box, filled from this report's own sourced statements
    # rather than an unattributed feed.
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
            capacity = max(1, min(6, int((footer_y - 48) / 18)))
            picked = candidates[:capacity]
            h = _bullet_block_height(picked, size=6.2, max_lines=2)
            bar_y = _heading(c, left_x, footer_y, "Additional evidence", width=full_w)
            _card(c, left_x, bar_y + 5, full_w, h)
            footer_y = _bullets(c, left_x + 4, bar_y - 2, full_w - 8, picked, max_words=40, size=6.2, max_lines=2)

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
