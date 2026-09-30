"""Company branding shared by the compact report's pages: logo, header band, footer."""

from __future__ import annotations

from pathlib import Path

from reportlab.lib import colors
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .pdf_renderer import ACCENT, ACCENT_DARK, HAIRLINE, MUTED

COMPANY = "Megaannum Technology Limited"
LOGO = Path(__file__).parent / "assets" / "megaannum_logo.png"
BAND_H = 64.0


def draw_logo(c: Canvas, x: float, y: float, size: float) -> None:
    """Draw the logo with its lower-left corner at (x, y); skipped if the file is missing."""
    if LOGO.exists():
        c.drawImage(str(LOGO), x, y, size, size, mask="auto", preserveAspectRatio=True)


def brand_band(c: Canvas, page_w: float, page_h: float, title: str, right_text: str = "") -> None:
    """Full-width accent band: logo tile, company name, page title, optional date pill."""
    c.setFillColor(ACCENT)
    c.rect(0, page_h - BAND_H, page_w, BAND_H, fill=1, stroke=0)
    c.setFillColor(ACCENT_DARK)
    c.rect(0, page_h - BAND_H - 2.5, page_w, 2.5, fill=1, stroke=0)
    c.setFillColor(colors.white)
    c.roundRect(18, page_h - BAND_H + 9, 46, 46, 5, fill=1, stroke=0)
    draw_logo(c, 21, page_h - BAND_H + 12, 40)
    c.setFillColor(colors.white)
    c.setFont("Helvetica-Bold", 8.5)
    x = 76.0
    for ch in COMPANY.upper():  # letter-spaced by hand: a text object's spacing would leak
        c.drawString(x, page_h - 23, ch)
        x += stringWidth(ch, "Helvetica-Bold", 8.5) + 1.3
    c.setFont("Helvetica-Bold", 19)
    c.drawString(76, page_h - 46, title)
    if right_text:
        c.setFillColor(colors.white)
        c.roundRect(page_w - 194, page_h - BAND_H + 17, 176, 30, 5, fill=1, stroke=0)
        c.setFillColor(ACCENT)
        c.setFont("Helvetica-Bold", 8.5)
        c.drawCentredString(page_w - 106, page_h - BAND_H + 28, right_text)


def footer(c: Canvas, page_w: float, label: str, *, rule: bool = True) -> None:
    """Company name with a small logo at the left, the page label at the right."""
    base = 14.0 if rule else 9.0
    if rule:
        c.setStrokeColor(HAIRLINE)
        c.setLineWidth(0.8)
        c.line(18, 30, page_w - 18, 30)
    draw_logo(c, 18, base - 4, 14)
    c.setFillColor(MUTED)
    c.setFont("Helvetica-Bold", 7.2)
    c.drawString(36, base, COMPANY)
    c.setFont("Helvetica", 7.2)
    c.drawRightString(page_w - 18, base, label)
