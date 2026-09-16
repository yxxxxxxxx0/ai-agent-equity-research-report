"""Stage 10 - the PDF renderer.

The renderer is deliberately dumb: it consumes a ReportDraft and lays it out. It
computes nothing, fetches nothing and writes no prose of its own, so a change in
the report's content is always a change in an upstream stage.

Layout is a small, named "template" system - a fixed palette and style set
(masthead band, callout boxes, chip-badged section headers, zebra-striped
metric tables, restrained charts, numbered footer) applied uniformly to
whatever the draft contains. Nothing here decides *what* is said - only how it
is dressed.

Two layout rules do real work for the reader and are worth naming:

* The first page is set in two columns, with the draft's opening section beside
  the key-data panel, the way a sell-side first page carries its data box. The
  reader gets price, scale, multiples, the last reported quarter and the
  guidance in view before the first paragraph.
* A table column is printed only if some row fills it. Growth and
  versus-consensus columns are genuinely absent for some metrics and some
  companies, and a column of blanks reads as missing data rather than as a
  question the report did not need to ask.
"""

from __future__ import annotations

import datetime as dt
import logging
from pathlib import Path

from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.shapes import Drawing, String
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus import (
    CondPageBreak,
    HRFlowable,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from ..domain.enums import ClaimType, ReportSection
from ..domain.qa import QAResult
from ..domain.report import (
    ChartSpec,
    KeyDataPanel,
    MetricTable,
    ReportDraft,
    ReportSectionDraft,
)
from ..errors import RenderingError
from ..logging_setup import get_logger, log_event

logger = get_logger("rendering.pdf")

# ---------------------------------------------------------------------------
# Palette & spacing scale
#
# A small, named set of tokens so every element below draws from the same
# system rather than choosing colours/spacing ad hoc. Two accents only
# (ACCENT for identity/structure, ACCENT_SOFT as its tint), plus a warm
# amber for advisory callouts and a muted teal/rust pair reserved for
# signed data in charts. Everything else is neutral ink/grey.
# ---------------------------------------------------------------------------
INK = colors.HexColor("#1a1a1a")
INK_SOFT = colors.HexColor("#3a3f47")
MUTED = colors.HexColor("#5c6470")
FAINT = colors.HexColor("#8993a1")
RULE = colors.HexColor("#c8ccd4")
HAIRLINE = colors.HexColor("#e6e9ee")
PAPER = colors.HexColor("#ffffff")
BAND = colors.HexColor("#eef1f5")
ZEBRA = colors.HexColor("#f5f7fa")

ACCENT = colors.HexColor("#12395e")          # deep navy - identity colour
ACCENT_SOFT = colors.HexColor("#e4ebf2")     # tint of ACCENT for chips/callouts
ACCENT_LINE = colors.HexColor("#9db6cc")

WARN_BG = colors.HexColor("#fdf3e0")
WARN_BORDER = colors.HexColor("#d99a2b")
WARN_INK = colors.HexColor("#7a4a00")

# A live-verified data-freshness mismatch (see pipeline.freshness_check) gets
# its own colour, distinct from the amber mock-data notice: the two are
# unrelated problems (synthetic data vs. a real but stale dataset) and must
# not be mistaken for each other at a glance.
STALE_BG = colors.HexColor("#fbe9e7")
STALE_BORDER = colors.HexColor("#c1442e")
STALE_INK = colors.HexColor("#7a2a1a")

QA_BG = colors.HexColor("#eef1f5")
QA_BORDER = colors.HexColor("#8993a1")
QA_INK = colors.HexColor("#3a3f47")

POS = colors.HexColor("#2f6f5e")             # muted teal/green - positive
NEG = colors.HexColor("#a04434")             # muted rust/red - negative

#: Used only by the annotated companion report (see render_annotated), for
#: the margin note under each paragraph/bullet - a colour that appears
#: nowhere else in the document, so an annotation is never mistaken for
#: report content.
ANNOTATION_INK = colors.HexColor("#7a5a9e")

PAGE_W, PAGE_H = A4
MARGIN_L = 18 * mm
MARGIN_R = 18 * mm
CONTENT_W = PAGE_W - MARGIN_L - MARGIN_R

#: First-page split: the opening argument beside the key-data panel.
PANEL_W = 54 * mm
PANEL_GUTTER = 6 * mm
LEAD_W = CONTENT_W - PANEL_W - PANEL_GUTTER

#: Width of one data column in a metric table. Fixed, so a two-column and a
#: four-column table share a right edge and read as one family.
DATA_COL_W = 26 * mm

#: Minimum space a new section needs below its heading before it is allowed to
#: start on the current page: the heading band itself plus roughly two lines
#: of body. A section that would start with less than this left on the page
#: opens a fresh page instead of stranding a heading with nothing under it.
SECTION_MIN_HEIGHT = 70

#: Short label printed against a statement so the reader can tell a reported
#: figure from an inference without following the citation.
_CLAIM_TAGS: dict[ClaimType, str] = {
    ClaimType.CONFIRMED_FACT: "reported",
    ClaimType.MANAGEMENT_STATEMENT: "management",
    ClaimType.MARKET_EXPECTATION: "market expectation",
    ClaimType.CALCULATED_OBSERVATION: "calculated",
    ClaimType.INTERPRETATION: "interpretation",
}


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "company": ParagraphStyle(
            "company", parent=base["Title"], fontName="Helvetica-Bold",
            fontSize=20, leading=23, textColor=PAPER, alignment=0, spaceAfter=0),
        "subtitle": ParagraphStyle(
            "subtitle", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=10, leading=13, textColor=colors.HexColor("#c9d6e3"),
            spaceBefore=2, spaceAfter=0),
        "meta_r": ParagraphStyle(
            "meta_r", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=12, textColor=colors.HexColor("#c9d6e3"),
            alignment=TA_RIGHT),
        "ticker_pill": ParagraphStyle(
            "ticker_pill", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=9, leading=11, textColor=ACCENT, alignment=1),
        "section_num": ParagraphStyle(
            "section_num", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=10.5, leading=13, textColor=PAPER, alignment=1),
        "section_title": ParagraphStyle(
            "section_title", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=12.5, leading=15, textColor=ACCENT),
        "summary": ParagraphStyle(
            "summary", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=9, leading=12.5, textColor=MUTED, spaceBefore=2, spaceAfter=6),
        "body": ParagraphStyle(
            "body", parent=base["Normal"], fontName="Helvetica",
            fontSize=9, leading=12.8, textColor=INK, alignment=TA_JUSTIFY,
            spaceAfter=5.5),
        # bulletFontName is set explicitly: reportlab's default is Times-Roman,
        # which would embed a second font family just to draw the bullets.
        "bullet": ParagraphStyle(
            "bullet", parent=base["Normal"], fontName="Helvetica",
            bulletFontName="Helvetica", bulletFontSize=7,
            fontSize=8.8, leading=12.2, textColor=INK, spaceAfter=3),
        "takeaway": ParagraphStyle(
            "takeaway", parent=base["Normal"], fontName="Helvetica",
            bulletFontName="Helvetica", bulletFontSize=7,
            fontSize=9.2, leading=12.8, textColor=INK, spaceAfter=4),
        "tabletitle": ParagraphStyle(
            "tabletitle", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.8, leading=11, textColor=ACCENT, spaceBefore=5, spaceAfter=3.5),
        "cell": ParagraphStyle(
            "cell", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.4, textColor=INK),
        "cell_num": ParagraphStyle(
            "cell_num", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.4, textColor=INK, alignment=TA_RIGHT),
        # The subject company's own row in a comparison table, so the reader
        # can find it without reading every label.
        "cellhead_row": ParagraphStyle(
            "cellhead_row", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10.4, textColor=ACCENT),
        "cell_num_bold": ParagraphStyle(
            "cell_num_bold", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10.4, textColor=ACCENT, alignment=TA_RIGHT),
        "cellhead": ParagraphStyle(
            "cellhead", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.6, leading=10, textColor=PAPER),
        "cellhead_num": ParagraphStyle(
            "cellhead_num", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.6, leading=10, textColor=PAPER, alignment=TA_RIGHT),
        "note": ParagraphStyle(
            "note", parent=base["Normal"], fontName="Helvetica",
            fontSize=6.8, leading=9, textColor=FAINT, spaceBefore=2, spaceAfter=4),
        "panel_group": ParagraphStyle(
            "panel_group", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.2, leading=9.5, textColor=ACCENT),
        "panel_label": ParagraphStyle(
            "panel_label", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.4, leading=9.6, textColor=INK_SOFT),
        "panel_value": ParagraphStyle(
            "panel_value", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=7.4, leading=9.6, textColor=INK, alignment=TA_RIGHT),
        "source": ParagraphStyle(
            "source", parent=base["Normal"], fontName="Helvetica",
            fontSize=7.4, leading=9.8, textColor=MUTED, spaceAfter=2),
        "gap": ParagraphStyle(
            "gap", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.6, textColor=WARN_INK, spaceAfter=2.5),
        # Data Gaps table priority column: bold and coloured for "high" so it
        # is scannable at a glance, plain for the other two tiers.
        "gap_priority_high": ParagraphStyle(
            "gap_priority_high", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8, leading=10.4, textColor=NEG),
        "gap_priority_medium": ParagraphStyle(
            "gap_priority_medium", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.4, textColor=INK_SOFT),
        "gap_priority_low": ParagraphStyle(
            "gap_priority_low", parent=base["Normal"], fontName="Helvetica",
            fontSize=8, leading=10.4, textColor=FAINT),
        "notice": ParagraphStyle(
            "notice", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.6, leading=11.5, textColor=WARN_INK),
        "notice_stale": ParagraphStyle(
            "notice_stale", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.6, leading=11.5, textColor=STALE_INK),
        "qa_head": ParagraphStyle(
            "qa_head", parent=base["Normal"], fontName="Helvetica-Bold",
            fontSize=8.6, leading=11.5, textColor=QA_INK),
        "footer": ParagraphStyle(
            "footer", parent=base["Normal"], fontName="Helvetica",
            fontSize=6.6, leading=8, textColor=MUTED),
        # Annotation-mode only: the extracted "point" of a paragraph/bullet,
        # printed directly beneath it. Deliberately not one of the report's
        # own voices (not a claim tag, not a data-gap warning) so it reads as
        # a margin note rather than as content.
        "annotation": ParagraphStyle(
            "annotation", parent=base["Normal"], fontName="Helvetica-Oblique",
            fontSize=7.6, leading=10, textColor=ANNOTATION_INK,
            leftIndent=13, spaceAfter=6),
    }


class PdfReportRenderer:
    """Renders a ReportDraft to a PDF file."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)

    def render(
        self, draft: ReportDraft, qa_result: QAResult | None = None,
        *, filename: str | None = None,
    ) -> Path:
        return self._render(draft, qa_result, filename=filename, annotations=None)

    def render_annotated(
        self, draft: ReportDraft, annotations: dict[str, str],
        qa_result: QAResult | None = None, *, filename: str | None = None,
    ) -> Path:
        """A companion PDF: the same report, with each paragraph and bullet
        followed by a short note on the point it exists to convey.

        Built from the same story-assembly code as the primary report (see
        ``_story``/``_section``) so the two never drift apart in structure -
        only the presence of ``annotations`` changes what gets rendered under
        each paragraph and statement. This is a review aid, never fed back
        into the draft and never a substitute for the primary PDF.
        """
        name = filename or self._filename(draft).replace(".pdf", "_annotated.pdf")
        return self._render(draft, qa_result, filename=name, annotations=annotations)

    def _render(
        self, draft: ReportDraft, qa_result: QAResult | None, *,
        filename: str | None, annotations: dict[str, str] | None,
    ) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        name = filename or self._filename(draft)
        path = self.output_dir / name
        styles = _styles()
        title = draft.title + (" (annotated)" if annotations is not None else "")

        try:
            document = SimpleDocTemplate(
                str(path), pagesize=A4,
                leftMargin=MARGIN_L, rightMargin=MARGIN_R,
                topMargin=10 * mm, bottomMargin=16 * mm,
                title=title,
                author="Automated Equity Research Prototype",
                subject=draft.objective,
            )
            story = self._story(draft, qa_result, styles, annotations=annotations)
            document.build(
                story,
                onFirstPage=lambda cv, doc: self._page_furniture(cv, doc, draft),
                onLaterPages=lambda cv, doc: self._page_furniture(cv, doc, draft),
                canvasmaker=_make_numbered_canvas(draft),
            )
        except Exception as exc:  # noqa: BLE001 - surfaced as a rendering error
            raise RenderingError(f"PDF generation failed: {exc}") from exc

        log_event(logger, logging.INFO, "PDF rendered",
                  path=str(path), sections=len(draft.sections),
                  annotated=annotations is not None,
                  size_bytes=path.stat().st_size if path.exists() else 0)
        return path

    # -- document assembly -----------------------------------------------
    def _story(
        self, draft: ReportDraft, qa_result: QAResult | None, styles,
        *, annotations: dict[str, str] | None = None,
    ) -> list:
        story: list = []
        story.append(self._masthead_band(draft, styles))
        story.append(Spacer(1, 9))

        if draft.contains_mock_data:
            story.append(self._callout(
                "▲",
                "ILLUSTRATIVE SAMPLE DATA",
                "This report was generated by a prototype using mock data "
                "providers. The figures are synthetic and must not be used "
                "for investment purposes.",
                WARN_BG, WARN_BORDER, WARN_INK, styles["notice"],
            ))
            story.append(Spacer(1, 7))

        freshness = draft.metadata.get("freshness_check")
        if isinstance(freshness, dict) and freshness.get("mismatched"):
            period = draft.metadata.get("latest_reported_period", "the dataset's period")
            story.append(self._callout(
                "⏱",
                "DATA FRESHNESS NOTICE",
                f"This report's dataset is anchored on {period}. A live check found "
                f"evidence of a more recent public report ({freshness.get('verified_period')}, "
                f"source: {freshness.get('source_name')}). The analysis below has not been "
                "updated to reflect it.",
                STALE_BG, STALE_BORDER, STALE_INK, styles["notice_stale"],
            ))
            story.append(Spacer(1, 7))

        # The sources section is rendered once, in the back matter, together
        # with the citation list it is a heading for.
        body = [s for s in draft.sections if s.section is not ReportSection.SOURCES]

        story.extend(self._opening(body, draft, styles, annotations=annotations))
        for index, section in enumerate(body[1:], start=2):
            story.extend(self._section(index, section, draft, styles, annotations=annotations))

        story.extend(self._back_matter(
            draft, qa_result, styles, number=len(body) + 1, has_body=bool(body)))
        return story

    def _opening(
        self, body: list[ReportSectionDraft], draft: ReportDraft, styles,
        *, annotations: dict[str, str] | None = None,
    ) -> list:
        """Render the key-data strip, then the opening section at full width.

        The former implementation nested the complete Key Takeaways table in
        the left cell of another two-column table.  ReportLab cannot reliably
        split a nested table cell across pages; sufficiently long takeaways
        therefore painted over one another.  A compact full-width data strip
        preserves the first-page figures while allowing every narrative
        flowable to paginate normally in the document frame.
        """
        if not body:
            return []
        lead = self._section(
            1, body[0], draft, styles, width=CONTENT_W, top_level=False,
            annotations=annotations)
        if draft.key_data is None or draft.key_data.is_empty:
            return lead
        return [self._key_data_strip(draft.key_data, styles), Spacer(1, 7), *lead]

    def _key_data_strip(self, panel: KeyDataPanel, styles) -> Table:
        """Compact two-up grid used above the opening narrative."""
        cards = []
        card_width = CONTENT_W / 2
        for group in panel.groups:
            rows = [[Paragraph(_escape(group.title), styles["panel_group"]), ""]]
            rows.extend([
                [Paragraph(_escape(item.label), styles["panel_label"]),
                 Paragraph(_escape(item.value), styles["panel_value"])]
                for item in group.items
            ])
            card = Table(rows, colWidths=[card_width * 0.58, card_width * 0.42])
            card.setStyle(TableStyle([
                ("SPAN", (0, 0), (1, 0)),
                ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
                ("LINEABOVE", (0, 0), (-1, 0), 1, ACCENT),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 2),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ]))
            cards.append(card)

        outer_rows = []
        for index in range(0, len(cards), 2):
            pair = cards[index:index + 2]
            if len(pair) == 1:
                pair.append("")
            outer_rows.append(pair)
        strip = Table(outer_rows, colWidths=[card_width, card_width], hAlign="LEFT")
        strip.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ]))
        return strip

    def _back_matter(
        self, draft: ReportDraft, qa_result: QAResult | None, styles,
        *, number: int, has_body: bool = True,
    ) -> list:
        """Sources, data gaps, what was left out, then the QA trail.

        All four are one apparatus under one heading. The sources section used
        to print its own numbered heading in the body and the citation list
        printed a second "Sources" heading behind it, so the report ended with
        two headings for one thing. The QA warnings, meanwhile, used to open
        the report, where dozens of internal findings pushed the analysis off
        the first page; they belong here, still in full, behind the argument
        they qualify rather than in front of it.
        """
        sources = draft.section(ReportSection.SOURCES)
        if sources is None and not draft.citations:
            return []

        # Only break when there is a body to break away from: a run where the
        # evidence filled no section at all would otherwise open on a blank page.
        story: list = [PageBreak()] if has_body else []
        story.extend(self._section_heading(
            number, sources.title if sources else "Sources", styles))
        if sources is not None:
            if sources.summary:
                story.append(Paragraph(_escape(sources.summary), styles["summary"]))
            for paragraph in sources.paragraphs:
                story.append(Paragraph(_escape(paragraph), styles["body"]))

        for citation in draft.citations:
            label = "[sample data] " if citation.is_mock else ""
            anchor = f'<a name="cite_{citation.ref_number}"/>'
            if citation.source_url:
                href = _escape(citation.source_url)
                marker = (
                    f'<a href="{href}" color="#12395e">[{citation.ref_number}]</a>'
                )
                url = f' <a href="{href}" color="#12395e">&lt;{href}&gt;</a>'
            else:
                marker = f"<font color='#12395e'>[{citation.ref_number}]</font>"
                url = ""
            story.append(Paragraph(
                f"{anchor}{marker} "
                f"{label}{_escape(citation.text)}{url}",
                styles["source"],
            ))

        if sources is not None and sources.data_gaps:
            story.append(Spacer(1, 6))
            story.extend(self._gap_box(sources.data_gaps, styles))

        omitted = draft.metadata.get("sections_omitted") or []
        if omitted:
            story.append(Spacer(1, 6))
            story.append(self._omissions_box(omitted, styles))

        if qa_result is not None and qa_result.warnings:
            story.append(Spacer(1, 6))
            story.append(self._qa_box(qa_result, styles))
        return story

    # -- key data panel --------------------------------------------------
    def _key_data_panel(self, panel: KeyDataPanel, styles) -> Table:
        """The first page's data box: label left, figure right, grouped."""
        rows: list[list] = []
        spans: list[tuple] = []
        rules: list[tuple] = []

        for group in panel.groups:
            if rows:
                spans.append(("TOPPADDING", (0, len(rows)), (-1, len(rows)), 7))
            rules.append(("LINEBELOW", (0, len(rows)), (-1, len(rows)), 0.6, ACCENT_LINE))
            spans.append(("SPAN", (0, len(rows)), (-1, len(rows))))
            rows.append([Paragraph(_escape(group.title), styles["panel_group"]), ""])
            for item in group.items:
                rows.append([
                    Paragraph(_escape(item.label), styles["panel_label"]),
                    Paragraph(_escape(item.value), styles["panel_value"]),
                ])

        if panel.as_of:
            spans.append(("SPAN", (0, len(rows)), (-1, len(rows))))
            spans.append(("TOPPADDING", (0, len(rows)), (-1, len(rows)), 6))
            rows.append([
                Paragraph(f"Market data as at {_escape(panel.as_of)}", styles["note"]), ""])

        table = Table(rows, colWidths=[PANEL_W * 0.52, PANEL_W * 0.48],
                      splitByRow=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ZEBRA),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 6),
            ("RIGHTPADDING", (0, 0), (-1, -1), 6),
            ("TOPPADDING", (0, 0), (-1, -1), 1.6),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 1.6),
            ("LINEABOVE", (0, 0), (-1, 0), 2.2, ACCENT),
            *spans,
            *rules,
        ]))
        return table

    def _omissions_box(self, omitted: list, styles) -> Table:
        """What the report did not print, and why.

        The section list adapts to the evidence, so the reader is told which
        planned sections were dropped instead of being left to notice a gap.
        """
        heading = Paragraph("Sections not included", styles["qa_head"])
        items = [
            Paragraph(
                f"● <b>{_escape(str(entry.get('section', '')).replace('_', ' ').title())}</b>"
                f" — {_escape(entry.get('reason', ''))}",
                styles["gap"])
            for entry in omitted
            if isinstance(entry, dict)
        ]
        return self._qa_callout(heading, items, styles)

    # -- masthead ----------------------------------------------------------
    def _masthead_band(self, draft: ReportDraft, styles) -> Table:
        """A dark identity band: title block left, key metadata right."""
        period = draft.metadata.get("latest_reported_period")

        left_flow = [Paragraph(_escape(draft.company), styles["company"])]
        left_flow.append(Paragraph(_escape(draft.title), styles["subtitle"]))

        right_lines = [f"Report date &nbsp; <b>{_escape(draft.report_date.isoformat())}</b>"]
        if period:
            right_lines.append(f"Latest period &nbsp; <b>{_escape(period)}</b>")
        right_lines.append(f"Run &nbsp; {_escape(draft.report_run_id)}")
        right_flow = [Paragraph(line, styles["meta_r"]) for line in right_lines]

        if draft.ticker:
            pill = Table([[Paragraph(_escape(draft.ticker), styles["ticker_pill"])]],
                         colWidths=[22 * mm], rowHeights=[6.4 * mm])
            pill.setStyle(TableStyle([
                ("BACKGROUND", (0, 0), (-1, -1), PAPER),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("ROUNDEDCORNERS", [3, 3, 3, 3]),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            left_flow.insert(1, Spacer(1, 3))
            top_row = Table([[left_flow[0], pill]],
                            colWidths=[None, 24 * mm])
            top_row.setStyle(TableStyle([
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "RIGHT"),
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
            ]))
            left_flow = [top_row] + left_flow[1:]

        objective_line = Paragraph(
            f"Objective &nbsp;—&nbsp; {_escape(draft.objective)}",
            styles["meta_r"])
        right_flow.insert(0, objective_line)
        right_flow.insert(1, Spacer(1, 3))

        band = Table([[left_flow, right_flow]],
                     colWidths=[112 * mm, (PAGE_W - MARGIN_L - MARGIN_R - 112 * mm)])
        band.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (0, 0), 10),
            ("RIGHTPADDING", (0, 0), (0, 0), 6),
            ("LEFTPADDING", (1, 0), (1, 0), 6),
            ("RIGHTPADDING", (1, 0), (1, 0), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LINEBELOW", (0, 0), (-1, -1), 2.2, colors.HexColor("#0c2846")),
        ]))
        return band

    # -- callouts ------------------------------------------------------
    def _callout(self, glyph: str, heading: str, body: str,
                 bg, border, ink, heading_style) -> Table:
        """Shared visual language for advisory boxes: tint + left accent bar."""
        body_style = ParagraphStyle(
            "callout_body", parent=heading_style, fontName="Helvetica",
            fontSize=8.4, leading=11.4, textColor=ink, spaceBefore=2)
        glyph_style = ParagraphStyle(
            "callout_glyph", parent=heading_style, fontSize=13, leading=13,
            textColor=ink, alignment=1)
        content = [Paragraph(_escape(heading), heading_style)]
        if body:
            content.append(Paragraph(_escape(body), body_style))
        table = Table(
            [[Paragraph(glyph, glyph_style), content]],
            colWidths=[9 * mm, None],
        )
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), bg),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ("LINEBEFORE", (0, 0), (0, -1), 3, border),
            ("LEFTPADDING", (0, 0), (0, 0), 4),
            ("RIGHTPADDING", (0, 0), (0, 0), 2),
            ("LEFTPADDING", (1, 0), (1, 0), 6),
            ("RIGHTPADDING", (1, 0), (1, 0), 9),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return table

    def _qa_box(self, qa_result: QAResult, styles) -> Table:
        heading = Paragraph(
            f"QA review — {len(qa_result.warnings)} warning(s) raised before "
            "publication", styles["qa_head"])
        # This box is a short preview near the top of the report, not the
        # authoritative disclosure (that's the Sources section's data-gap
        # box, and the full list is always in the run manifest) - so it
        # stays capped for readability rather than growing with the finding
        # count.
        items = [
            Paragraph(f"● <b>{_escape(finding.check)}</b> — {_escape(finding.message)}",
                      styles["gap"])
            for finding in qa_result.warnings[:6]
        ]
        if len(qa_result.warnings) > 6:
            items.append(Paragraph(
                f"● and {len(qa_result.warnings) - 6} further warning(s); "
                "see the run manifest.", styles["gap"]))
        return self._qa_callout(heading, items, styles)

    def _qa_callout(self, heading: Paragraph, items: list, styles) -> Table:
        """Shared warning-box visual language: tint + left accent bar.

        Built as one table row per item (heading row plus one row per item)
        rather than packing every item into a single cell, so a long list -
        e.g. dozens of data gaps on a real (non-mock-truncated) run - can
        split across a page break like any other table instead of becoming
        one oversized, unsplittable flowable that crashes rendering.
        """
        glyph_style = ParagraphStyle(
            "qa_glyph", fontName="Helvetica-Bold", fontSize=12, textColor=QA_INK,
            alignment=1)
        rows = [[Paragraph("⚠", glyph_style), heading]]
        rows.extend(["", item] for item in items)
        table = Table(rows, colWidths=[9 * mm, None])
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, -1), QA_BG),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ALIGN", (0, 0), (0, 0), "CENTER"),
            ("LINEBEFORE", (0, 0), (0, -1), 3, QA_BORDER),
            ("LEFTPADDING", (0, 0), (0, -1), 4),
            ("RIGHTPADDING", (0, 0), (0, -1), 2),
            ("LEFTPADDING", (1, 0), (1, -1), 6),
            ("RIGHTPADDING", (1, 0), (1, -1), 9),
            ("TOPPADDING", (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 3 if items else 7),
            ("TOPPADDING", (0, 1), (-1, -1), 1.5),
            ("BOTTOMPADDING", (0, 1), (-1, -1), 1.5),
        ]
        if items:
            style_cmds.append(("BOTTOMPADDING", (0, -1), (-1, -1), 7))
        table.setStyle(TableStyle(style_cmds))
        return table

    # -- sections ------------------------------------------------------
    def _section_heading(self, index: int, title: str, styles) -> list:
        """A numbered chip beside the title, with a coloured underrule."""
        # A two-digit number does not fit a 7mm square and would wrap onto a
        # second line inside the chip, so the chip grows into a pill instead.
        chip_w = 7 * mm if index < 10 else 10 * mm
        chip = Table([[Paragraph(str(index), styles["section_num"])]],
                     colWidths=[chip_w], rowHeights=[7 * mm])
        chip.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
            ("ROUNDEDCORNERS", [2, 2, 2, 2]),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        head = Table([[chip, Paragraph(_escape(title), styles["section_title"])]],
                     colWidths=[chip_w + 3 * mm, None])
        head.setStyle(TableStyle([
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 0),
            ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ("TOPPADDING", (0, 0), (-1, -1), 0),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ]))
        return [
            Spacer(1, 4),
            head,
            HRFlowable(width="100%", thickness=1.4, color=ACCENT_LINE,
                       spaceBefore=4, spaceAfter=7),
        ]

    def _section(
        self, index: int, section: ReportSectionDraft, draft: ReportDraft, styles,
        *, width: float = CONTENT_W, top_level: bool = True,
        annotations: dict[str, str] | None = None,
    ) -> list:
        """One section. ``width`` is narrowed for the two-column first page.

        ``top_level`` gates the orphaned-heading guard: it only makes sense
        against the main frame's remaining page space, so the lead section -
        embedded in the first page's two-column table alongside the key-data
        panel - opts out and relies on that table's own splitting instead.

        ``annotations`` is only set by :meth:`render_annotated`: when present,
        every paragraph and bullet is followed by the note recorded for it.
        """
        story: list = [CondPageBreak(SECTION_MIN_HEIGHT)] if top_level else []
        story += self._section_heading(index, section.title, styles)

        if section.summary:
            story.append(Paragraph(_escape(section.summary), styles["summary"]))

        for paragraph in section.paragraphs:
            block = [Paragraph(_escape(paragraph), styles["body"])]
            self._append_annotation(block, paragraph, annotations, styles)
            story.append(KeepTogether(block) if annotations is not None else block[0])

        if section.statements:
            is_takeaways = section.section is ReportSection.KEY_TAKEAWAYS
            if is_takeaways:
                story.append(self._takeaways_box(section, styles, annotations=annotations))
            elif annotations is not None:
                # One KeepTogether block per bullet rather than a shared
                # ListFlowable, so a bullet's note always lands directly
                # beneath it rather than at the mercy of the list's own
                # internal layout.
                for statement in section.statements:
                    block = [Paragraph(f"• {self._statement_html(statement)}", styles["bullet"])]
                    self._append_annotation(block, statement.text, annotations, styles)
                    story.append(KeepTogether(block))
                story.append(Spacer(1, 3))
            else:
                items = [
                    ListItem(
                        Paragraph(self._statement_html(statement), styles["bullet"]),
                        leftIndent=10, value=None,
                        bulletFontName="Helvetica", bulletFontSize=6.5,
                        bulletColor=ACCENT_LINE,
                    )
                    for statement in section.statements
                ]
                story.append(ListFlowable(
                    items, bulletType="bullet", bulletChar="•",
                    bulletFontName="Helvetica", bulletFontSize=6.5,
                    leftIndent=11, start=None,
                ))
                story.append(Spacer(1, 3))

        for table in section.tables:
            flowables = self._metric_table(table, styles, width=width)
            if flowables:
                story.append(KeepTogether(flowables))

        for chart in section.charts:
            drawing = self._chart(chart, width=width)
            if drawing is not None:
                # Kept together, or a chart whose title fits at the foot of a
                # page ends up captioning blank space with its plot overleaf.
                story.append(KeepTogether([
                    Spacer(1, 5),
                    Paragraph(_escape(chart.title), styles["tabletitle"]),
                    drawing,
                ]))

        # Detailed gap disclosure lives in the back matter with the sources;
        # repeating every gap beneath the analysis overwhelms the argument.
        return story

    @staticmethod
    def _append_annotation(
        block: list, text: str, annotations: dict[str, str] | None, styles,
    ) -> None:
        """Append the recorded note for ``text``, if annotation mode is on.

        Looked up by exact statement/paragraph text - the same key
        :func:`eq_report.synthesis.annotate.build_annotations` used to build
        the map - so nothing here can attach the wrong note to the wrong
        sentence. A sentence the annotation pass skipped (should not happen;
        see that module's fallback) is simply followed by nothing rather than
        a placeholder.
        """
        if annotations is None:
            return
        note = annotations.get(text, "")
        if note:
            block.append(Paragraph(f"↳ <i>conveys:</i> {_escape(note)}", styles["annotation"]))

    def _takeaways_box(
        self, section: ReportSectionDraft, styles,
        *, annotations: dict[str, str] | None = None,
    ) -> Table:
        """Key Takeaways rendered with the shared callout language.

        One table row per takeaway rather than all of them in a single cell, so
        the box can break across a page like any other table. Packed into one
        cell it was unsplittable, and a box that did not fit in the space left
        on the page jumped whole to the next one, leaving the first page half
        empty.
        """
        rows = []
        for statement in section.statements:
            block = [Paragraph(f"▸ {self._statement_html(statement)}", styles["takeaway"])]
            self._append_annotation(block, statement.text, annotations, styles)
            rows.append([block])  # one column; the cell holds the whole block
        table = Table(rows, colWidths=["100%"], splitByRow=1)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), ACCENT_SOFT),
            ("LINEBEFORE", (0, 0), (0, -1), 3, ACCENT),
            ("LEFTPADDING", (0, 0), (-1, -1), 10),
            ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ("TOPPADDING", (0, 0), (-1, -1), 2),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("TOPPADDING", (0, 0), (-1, 0), 7),
            ("BOTTOMPADDING", (0, -1), (-1, -1), 7),
        ]))
        return table

    #: Sort/print order and label for each priority. A gap outside this set
    #: (should not happen - see agents/llm_agent.py::_validated_priority) sorts
    #: last under "Medium" rather than being dropped.
    _GAP_PRIORITY_ORDER: dict[str, tuple[int, str]] = {
        "high": (0, "High"), "medium": (1, "Medium"), "low": (2, "Low"),
    }

    def _gap_box(self, gaps, styles) -> list:
        """The Data Gaps table: Priority | Missing Data | Why It Matters.

        Sorted by priority so the handful of gaps that actually limit the
        analysis lead, rather than reading as one undifferentiated wall of
        text with the same weight given to a missing footnote and a missing
        segment breakdown.
        """
        ordered = sorted(
            gaps,
            key=lambda g: self._GAP_PRIORITY_ORDER.get(g.priority, (1, "Medium"))[0])

        rows = [[
            Paragraph("Priority", styles["cellhead"]),
            Paragraph("Missing data", styles["cellhead"]),
            Paragraph("Why it matters", styles["cellhead"]),
        ]]
        for gap in ordered:
            _, label = self._GAP_PRIORITY_ORDER.get(gap.priority, (1, "Medium"))
            rows.append([
                Paragraph(label, styles.get(f"gap_priority_{gap.priority}", styles["cell"])),
                Paragraph(_escape(gap.description), styles["cell"]),
                Paragraph(_escape(gap.impact) if gap.impact else "—", styles["cell"]),
            ])

        table = Table(rows, colWidths=[20 * mm, 75 * mm, CONTENT_W - 95 * mm],
                      repeatRows=1, hAlign="LEFT")
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ]
        for i in range(1, len(rows)):
            if (i - 1) % 2 == 1:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
            if i < len(rows) - 1:
                style_cmds.append(("LINEBELOW", (0, i), (-1, i), 0.25, HAIRLINE))
        table.setStyle(TableStyle(style_cmds))
        return [Paragraph(f"Data gaps ({len(gaps)})", styles["tabletitle"]), table]

    def _statement_html(self, statement) -> str:
        # Each ref is an internal link to its anchor in the source list
        # (_back_matter), so clicking [n] in the body jumps straight to it.
        refs = "".join(
            f'<a href="#cite_{r}" color="#12395e">[{r}]</a>'
            for r in statement.citation_refs
        )
        tag = _CLAIM_TAGS.get(statement.claim_type, statement.claim_type.value)
        return (
            f"{_escape(statement.text)} "
            f"<font size=7 color='#5c6470'>({tag}){refs}</font>"
        )

    # -- tables ----------------------------------------------------------
    def _metric_table(self, table: MetricTable, styles, *, width: float = CONTENT_W) -> list:
        """A metric grid, minus any column no row filled.

        Emptiness is decided per column here rather than upstream because it is
        a presentation question: the synthesis layer is entitled to ask for a
        "vs consensus" column and leave it blank where no consensus exists, and
        the renderer is what decides that an entirely blank column should not
        take up space or imply missing data.
        """
        keep = [
            index for index in range(len(table.columns) - 1)
            if any(_cell(row, index).strip() for row in table.rows)
        ]
        if not keep or not table.rows:
            return []

        header = [Paragraph(_escape(table.columns[0]), styles["cellhead"])]
        header += [
            Paragraph(_escape(table.columns[index + 1]), styles["cellhead_num"])
            for index in keep
        ]
        rows = [header]
        for row in table.rows:
            label_style = "cellhead_row" if row.emphasis else "cell"
            value_style = "cell_num_bold" if row.emphasis else "cell_num"
            rows.append([
                Paragraph(_escape(row.label), styles.get(label_style, styles["cell"])),
                *[
                    Paragraph(_escape(_cell(row, index)),
                              styles.get(value_style, styles["cell_num"]))
                    for index in keep
                ],
            ])

        # Two-column tables carry longer headings ("Revenue growth YoY") than
        # four-column ones, so the data columns widen as they thin out.
        data_w = {1: 34 * mm, 2: 30 * mm}.get(len(keep), DATA_COL_W)
        col_widths = [max(width - data_w * len(keep), 40 * mm)]
        col_widths += [data_w] * len(keep)

        flowable = Table(rows, colWidths=col_widths, repeatRows=1, hAlign="LEFT")
        style_cmds = [
            ("BACKGROUND", (0, 0), (-1, 0), ACCENT),
            ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
            ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ("TOPPADDING", (0, 0), (-1, -1), 3.4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3.4),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, RULE),
        ]
        for i, row in enumerate(table.rows, start=1):
            if row.emphasis:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ACCENT_SOFT))
            elif (i - 1) % 2 == 1:
                style_cmds.append(("BACKGROUND", (0, i), (-1, i), ZEBRA))
            if i < len(rows) - 1:
                style_cmds.append(("LINEBELOW", (0, i), (-1, i), 0.25, HAIRLINE))
        flowable.setStyle(TableStyle(style_cmds))

        out = [Paragraph(_escape(table.title), styles["tabletitle"]), flowable]
        if table.note:
            out.append(Paragraph(_escape(table.note), styles["note"]))
        out.append(Spacer(1, 5))
        return out

    # -- charts ----------------------------------------------------------
    #
    # Both chart shapes are drawn stripped back: no boxed frame, one accent
    # colour, and the numbers written where they are read rather than
    # inferred off an axis. A chart in this report exists to show a shape a
    # table cannot - a path, or a set of relative sizes - so anything that is
    # not that shape or its scale is removed. The price path is the
    # exception on gridlines: a reader placing a value on the timeline needs
    # more than two dated ticks, so it carries a full set of x-axis ticks and
    # matching gridlines rather than just its endpoints.
    def _chart(self, spec: ChartSpec, *, width: float = CONTENT_W) -> Drawing | None:
        try:
            if spec.chart_type == "bar":
                return self._bar_chart(spec, width)
            if spec.chart_type == "line":
                return self._line_chart(spec, width)
        except Exception as exc:  # noqa: BLE001 - a chart is never worth failing over
            log_event(logger, logging.WARNING, "chart skipped",
                      title=spec.title, error=str(exc))
        return None

    def _bar_chart(self, spec: ChartSpec, width: float) -> Drawing:
        height = 128
        drawing = Drawing(width, height)
        chart = VerticalBarChart()
        chart.x, chart.y = 4, 26
        chart.width, chart.height = width - 12, height - 46
        chart.data = [list(spec.values)]

        chart.categoryAxis.categoryNames = [
            (c[:15] + "…" if len(c) > 16 else c) for c in spec.categories]
        chart.categoryAxis.labels.fontName = "Helvetica"
        chart.categoryAxis.labels.fontSize = 6.4
        chart.categoryAxis.labels.fillColor = MUTED
        chart.categoryAxis.strokeColor = RULE
        chart.categoryAxis.strokeWidth = 0.5

        low = min(spec.values) if spec.values else 0
        high = max(spec.values) if spec.values else 1
        span = (high - low) or (abs(high) or 1)
        chart.valueAxis.valueMin = min(0, low - span * 0.1)
        chart.valueAxis.valueMax = high + span * 0.18
        # The value is printed on each bar, so the axis itself is redundant.
        chart.valueAxis.visible = 0
        chart.valueAxis.visibleGrid = 0
        chart.barLabels.fontName = "Helvetica"
        chart.barLabels.fontSize = 6.4
        chart.barLabels.fillColor = INK_SOFT
        chart.barLabelFormat = "%0.1f"
        chart.barLabels.dy = 5

        if spec.values and min(spec.values) < 0 <= max(spec.values):
            chart.bars.strokeWidth = 0
            for i, v in enumerate(spec.values):
                chart.bars[(0, i)].fillColor = POS if v >= 0 else NEG
        else:
            chart.bars[0].fillColor = ACCENT
            chart.bars.strokeWidth = 0
        chart.barSpacing = 2

        drawing.add(chart)
        drawing.add(String(4, height - 9, spec.unit, fontSize=6.4,
                           fontName="Helvetica", fillColor=FAINT))
        return drawing

    #: Evenly spaced x-axis ticks on a price path. Enough to place a reader
    #: anywhere in the period without crowding a chart this width.
    _LINE_CHART_TICKS = 7

    def _line_chart(self, spec: ChartSpec, width: float) -> Drawing:
        """A price path: the line, its scale, and a dated x-axis."""
        height = 122
        drawing = Drawing(width, height)
        plot = LinePlot()
        plot.x, plot.y = 30, 24
        plot.width, plot.height = width - 44, height - 44
        plot.data = [[(i, v) for i, v in enumerate(spec.values)]]
        plot.lines[0].strokeColor = ACCENT
        plot.lines[0].strokeWidth = 1.5
        plot.lines[0].symbol = None

        last = max(len(spec.values) - 1, 1)
        tick_count = min(self._LINE_CHART_TICKS, last + 1)
        steps = sorted({round(last * i / (tick_count - 1)) for i in range(tick_count)}) \
            if tick_count > 1 else [0]
        plot.xValueAxis.valueMin = 0
        plot.xValueAxis.valueMax = last
        plot.xValueAxis.valueSteps = steps
        plot.xValueAxis.labelTextFormat = (
            lambda value: _chart_date_label(spec.categories, int(value)))
        plot.xValueAxis.labels.fontName = "Helvetica"
        plot.xValueAxis.labels.fontSize = 6.2
        plot.xValueAxis.labels.fillColor = MUTED
        plot.xValueAxis.labels.angle = 0
        plot.xValueAxis.strokeColor = RULE
        plot.xValueAxis.strokeWidth = 0.5
        plot.xValueAxis.visibleGrid = 1
        plot.xValueAxis.gridStrokeColor = HAIRLINE
        plot.xValueAxis.gridStrokeWidth = 0.4

        low, high = min(spec.values), max(spec.values)
        span = (high - low) or 1
        plot.yValueAxis.valueMin = low - span * 0.12
        plot.yValueAxis.valueMax = high + span * 0.12
        plot.yValueAxis.valueSteps = [round(low), round(high)]
        plot.yValueAxis.labels.fontName = "Helvetica"
        plot.yValueAxis.labels.fontSize = 6.4
        plot.yValueAxis.labels.fillColor = MUTED
        plot.yValueAxis.strokeColor = RULE
        plot.yValueAxis.strokeWidth = 0.5
        plot.yValueAxis.visibleGrid = 1
        plot.yValueAxis.gridStrokeColor = HAIRLINE
        plot.yValueAxis.gridStrokeWidth = 0.4

        drawing.add(plot)
        drawing.add(String(4, height - 9, spec.unit, fontSize=6.4,
                           fontName="Helvetica", fillColor=FAINT))
        # The latest value, printed where the line ends.
        drawing.add(String(
            plot.x + plot.width + 3,
            plot.y + plot.height * (spec.values[-1] - plot.yValueAxis.valueMin)
            / (plot.yValueAxis.valueMax - plot.yValueAxis.valueMin) - 2,
            f"{spec.values[-1]:,.0f}",
            fontSize=6.6, fontName="Helvetica-Bold", fillColor=ACCENT))
        return drawing

    # -- page furniture --------------------------------------------------
    def _page_furniture(self, canvas, doc, draft: ReportDraft) -> None:
        """Thin footer rule + identity line. Page numbering is drawn by
        NumberedCanvas once the total page count is known."""
        canvas.saveState()
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(MARGIN_L, 13.5 * mm, PAGE_W - MARGIN_R, 13.5 * mm)
        canvas.setFont("Helvetica", 6.6)
        canvas.setFillColor(MUTED)
        identity = f"{draft.company}" + (f" ({draft.ticker})" if draft.ticker else "")
        base_left = f"{identity} · {draft.report_date.isoformat()}"
        full_left = f"{base_left} · run {draft.report_run_id}"

        watermark = "ILLUSTRATIVE SAMPLE DATA" if draft.contains_mock_data else None
        left = full_left
        if watermark:
            # The identity line's length is unbounded (company name, run id),
            # while the watermark is drawn centred - on a long identity line
            # the two would otherwise overlap. Drop the run id (already shown
            # in the masthead) first if there isn't room; the run id is the
            # least essential part of the footer.
            gap = 6
            watermark_left_edge = (
                PAGE_W / 2 - canvas.stringWidth(watermark, "Helvetica", 6.6) / 2 - gap)
            if MARGIN_L + canvas.stringWidth(full_left, "Helvetica", 6.6) > watermark_left_edge:
                left = base_left
            if MARGIN_L + canvas.stringWidth(left, "Helvetica", 6.6) > watermark_left_edge:
                left = identity

        canvas.drawString(MARGIN_L, 9 * mm, left)
        if watermark:
            canvas.setFillColor(WARN_INK)
            canvas.drawCentredString(PAGE_W / 2, 9 * mm, watermark)
        canvas.restoreState()

    @staticmethod
    def _filename(draft: ReportDraft) -> str:
        stamp = dt.datetime.now(dt.UTC).strftime("%Y%m%d")
        key = (draft.ticker or draft.company).replace(" ", "_")
        return f"{key}_{stamp}_{draft.report_run_id}.pdf"


def _make_numbered_canvas(draft: ReportDraft):
    """Build a Canvas subclass bound to this draft so the footer can print
    "Page X of Y" - reportlab only knows the total page count once every
    page has already been drawn, so pages are buffered and stamped in
    ``save()``."""

    class NumberedCanvas(pdfcanvas.Canvas):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._saved_page_states: list[dict] = []

        def showPage(self) -> None:
            self._saved_page_states.append(dict(self.__dict__))
            self._startPage()

        def save(self) -> None:
            total = len(self._saved_page_states)
            for state in self._saved_page_states:
                self.__dict__.update(state)
                self._draw_page_number(total)
                super().showPage()
            super().save()

        def _draw_page_number(self, total: int) -> None:
            self.saveState()
            self.setFont("Helvetica", 6.6)
            self.setFillColor(MUTED)
            self.drawRightString(
                PAGE_W - MARGIN_R, 9 * mm, f"Page {self._pageNumber} of {total}")
            self.restoreState()

    return NumberedCanvas


def _chart_date_label(categories: tuple[str, ...], index: int) -> str:
    """An x-axis tick label for a price-history point.

    Categories are ISO dates (``item.as_of.isoformat()``); parsed into "Mon
    YYYY" so a reader can place a value on the calendar without counting
    weeks. Anything that fails to parse as a date - a defensive fallback,
    since every caller today supplies ISO dates - falls back to the raw
    string truncated the way the chart used to render every label.
    """
    if not (0 <= index < len(categories)):
        return ""
    raw = categories[index]
    try:
        return dt.date.fromisoformat(raw).strftime("%b %Y")
    except ValueError:
        return raw[:7]


def _cell(row, index: int) -> str:
    """One cell of a metric row, tolerating a row shorter than the header.

    A table is built column by column from independent lookups, so a row whose
    last lookup found nothing may simply be short rather than padded.
    """
    return row.cells[index] if index < len(row.cells) else ""


def _escape(text: object) -> str:
    """Escape the characters reportlab's mini-HTML would otherwise interpret."""
    return (
        str(text or "")
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )
