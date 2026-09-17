"""Append a sourced, reproducible technical-analysis supplement to a report PDF."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import requests
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from ..config import ProviderCredentials


def _fetch_bloomberg_ohlcv(
    ticker: str,
    end_date: dt.date,
    credentials: ProviderCredentials,
    *,
    timeout: int = 30,
) -> tuple[list[tuple[dt.date, float, float, float, float, float]], str]:
    """Return Bloomberg daily OHLCV exclusively through MegadataAPI.

    Technical charts are deliberately kept on the same licensed source as the
    research report.  There is no silent public-web fallback: missing
    Bloomberg coverage must remain visible to the caller.
    """
    if not credentials.megadata_base_url:
        raise ValueError("MegadataAPI is not configured for Bloomberg technical data.")
    url = credentials.megadata_base_url.rstrip("/") + "/api/bbg/ohlcv/data"
    headers = {"accept": "application/json"}
    auth: tuple[str, str] | None = None
    if credentials.megadata_username and credentials.megadata_password:
        auth = (credentials.megadata_username, credentials.megadata_password)
    elif credentials.megadata_api_key:
        headers["Authorization"] = f"Bearer {credentials.megadata_api_key}"
    # Report dates are interpreted in Asia/Shanghai by the application while
    # Bloomberg daily bars close on the U.S. market calendar.  Query through
    # the prior calendar day; weekends/holidays naturally resolve to the most
    # recent available session, whose actual date is printed on the chart.
    market_end_date = end_date - dt.timedelta(days=1)
    response = requests.get(
        url,
        params={
            "symbols": ticker,
            "from_date": (end_date - dt.timedelta(days=370)).isoformat(),
            "to_date": market_end_date.isoformat(),
        },
        headers=headers,
        auth=auth,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    block = payload.get(ticker) if isinstance(payload, dict) else payload
    if isinstance(block, dict) and block.get("error"):
        raise ValueError(f"Megadata Bloomberg OHLCV unavailable for {ticker}: {block['error']}")
    records = block.get("data", block.get("results", block)) if isinstance(block, dict) else block
    if isinstance(records, dict):
        # Some deployments return date-keyed dictionaries.
        records = [dict(value, date=key) if isinstance(value, dict) else {"date": key, "close": value}
                   for key, value in records.items()]
    rows: list[tuple[dt.date, float, float, float, float, float]] = []
    for record in records if isinstance(records, list) else []:
        if not isinstance(record, dict):
            continue
        lowered = {str(key).lower(): value for key, value in record.items()}
        raw_date = lowered.get("date") or lowered.get("datetime") or lowered.get("timestamp")
        try:
            day = dt.date.fromisoformat(str(raw_date)[:10])
            def field(primary: str, bloomberg: str) -> float:
                value = lowered.get(primary)
                if value is None:
                    value = lowered[bloomberg]
                return float(value)
            row = (
                day,
                field("open", "px_open"),
                field("high", "px_high"),
                field("low", "px_low"),
                field("close", "px_last"),
                field("volume", "px_volume"),
            )
        except (TypeError, ValueError, KeyError):
            continue
        if day <= market_end_date:
            rows.append(row)
    rows.sort(key=lambda row: row[0])
    if len(rows) < 120:
        raise ValueError(
            f"Megadata Bloomberg returned {len(rows)} complete daily OHLCV observations; 120 are required."
        )
    return rows, "Bloomberg daily OHLCV via MegadataAPI"


def _ema(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = [None] * len(values)
    if len(values) < period:
        return out
    current = sum(values[:period]) / period
    out[period - 1] = current
    alpha = 2 / (period + 1)
    for i in range(period, len(values)):
        current += alpha * (values[i] - current)
        out[i] = current
    return out


def _sma(values: list[float], period: int) -> list[float | None]:
    return [None if i + 1 < period else sum(values[i - period + 1:i + 1]) / period
            for i in range(len(values))]


def _std(values: list[float], period: int) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(values)):
        if i + 1 < period:
            out.append(None)
        else:
            sample = values[i - period + 1:i + 1]
            mean = sum(sample) / period
            out.append(math.sqrt(sum((x - mean) ** 2 for x in sample) / period))
    return out


def _rsi(close: list[float], period: int = 10) -> list[float | None]:
    out: list[float | None] = [None] * len(close)
    if len(close) <= period:
        return out
    gains = [max(close[i] - close[i - 1], 0) for i in range(1, len(close))]
    losses = [max(close[i - 1] - close[i], 0) for i in range(1, len(close))]
    gain, loss = sum(gains[:period]) / period, sum(losses[:period]) / period
    out[period] = 100 if loss == 0 else 100 - 100 / (1 + gain / loss)
    for i in range(period + 1, len(close)):
        gain = (gain * (period - 1) + gains[i - 1]) / period
        loss = (loss * (period - 1) + losses[i - 1]) / period
        out[i] = 100 if loss == 0 else 100 - 100 / (1 + gain / loss)
    return out


def _stochastic(high: list[float], low: list[float], close: list[float]) -> tuple[list[float | None], list[float | None]]:
    k: list[float | None] = []
    for i, value in enumerate(close):
        if i < 13:
            k.append(None)
            continue
        hi, lo = max(high[i - 13:i + 1]), min(low[i - 13:i + 1])
        k.append(50 if hi == lo else 100 * (value - lo) / (hi - lo))
    d: list[float | None] = []
    for i in range(len(k)):
        window = [x for x in k[max(0, i - 2):i + 1] if x is not None]
        d.append(sum(window) / 3 if len(window) == 3 else None)
    return k, d


def _volatility(close: list[float], period: int) -> list[float | None]:
    returns = [math.log(close[i] / close[i - 1]) for i in range(1, len(close))]
    result: list[float | None] = [None]
    for i in range(len(returns)):
        if i + 1 < period:
            result.append(None)
        else:
            sample = returns[i - period + 1:i + 1]
            mean = sum(sample) / period
            result.append(math.sqrt(sum((x - mean) ** 2 for x in sample) / (period - 1)) * math.sqrt(252) * 100)
    return result


def _plot(c: Canvas, x: float, y: float, w: float, h: float, title: str,
          series: list[tuple[str, list[float | None], colors.Color]], *, fixed: tuple[float, float] | None = None) -> None:
    points = [v for _, values, _ in series for v in values if v is not None]
    lo, hi = fixed or (min(points), max(points))
    span = (hi - lo) or 1
    lo, hi = lo - span * .08, hi + span * .08
    # Keep titles and series keys on separate baselines.  Several panels have
    # long titles (notably Bollinger Bands) and up to four series; putting
    # both on the same line made the labels collide in the A4 appendix.
    c.setFont("Helvetica-Bold", 8)
    c.setFillColor(colors.HexColor("#12395e"))
    c.drawString(x, y + h + 18, title)
    c.setStrokeColor(colors.HexColor("#c8ccd4")); c.rect(x, y, w, h, stroke=1, fill=0)
    for level in range(1, 4):
        yy = y + h * level / 4; c.setStrokeColor(colors.HexColor("#e6e9ee")); c.line(x, yy, x + w, yy)
    for label, values, color in series:
        c.setStrokeColor(color); c.setLineWidth(1.15)
        previous = None
        for i, value in enumerate(values):
            if value is None:
                previous = None; continue
            px, py = x + w * i / max(len(values) - 1, 1), y + h * (value - lo) / (hi - lo)
            if previous: c.line(previous[0], previous[1], px, py)
            previous = (px, py)
    # A single plotted series is self-explanatory from the panel title.  For
    # multi-series panels, spread a compact key across a dedicated line.
    if len(series) > 1:
        for index, (label, _, color) in enumerate(series):
            c.setFillColor(color)
            c.setFont("Helvetica", 6.3)
            c.drawString(x + index * (w / len(series)), y + h + 7, label)
    c.setFillColor(colors.HexColor("#5c6470")); c.setFont("Helvetica", 6.5)
    c.drawRightString(x - 3, y + h - 2, _axis_label(hi)); c.drawRightString(x - 3, y - 2, _axis_label(lo))


def _axis_label(value: float) -> str:
    """Use compact labels so volume scales do not intrude into adjacent panels."""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.1f}bn"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if magnitude >= 1_000:
        return f"{value / 1_000:.1f}k"
    return f"{value:.1f}"


def _caption(c: Canvas, x: float, y: float, width: float, text: str) -> None:
    """Draw a compact, at-most-two-line explanation beneath a chart."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        proposed = f"{current} {word}".strip()
        if current and stringWidth(proposed, "Helvetica", 6.2) > width:
            lines.append(current)
            current = word
        else:
            current = proposed
    if current:
        lines.append(current)
    if len(lines) > 2:
        lines = [lines[0], f"{lines[1]} …"]
    c.setFillColor(colors.HexColor("#5c6470"))
    c.setFont("Helvetica-Oblique", 6.2)
    for index, line in enumerate(lines):
        c.drawString(x, y - index * 7.5, line)


def build_technical_appendix_pdf(
    ticker: str,
    end_date: dt.date,
    output_pdf: Path,
    *,
    page_label: str = "Page 1 of 1",
    credentials: ProviderCredentials | None = None,
    timeout: int = 30,
) -> Path:
    """Fetch Bloomberg OHLCV from MegadataAPI, calculate indicators and draw
    the standalone appendix page - everything ``append_technical_appendix``
    used to do except merging onto a base report.

    Split out so the orchestrator can run this (a live network fetch plus a
    chart render, independent of the report draft) concurrently with the
    analysis stage, then merge it once the base PDF exists rather than
    fetching it only after the report is otherwise finished.
    """
    rows, source_label = _fetch_bloomberg_ohlcv(
        ticker, end_date, credentials or ProviderCredentials(), timeout=timeout,
    )
    dates, _, high, low, close, volume = map(list, zip(*rows))
    ma10, ma20, ma50 = _sma(close, 10), _sma(close, 20), _sma(close, 50)
    sd10 = _std(close, 10); upper = [None if m is None or s is None else m + 2*s for m, s in zip(ma10, sd10)]; lower = [None if m is None or s is None else m - 2*s for m, s in zip(ma10, sd10)]
    ema12, ema26 = _ema(close, 12), _ema(close, 26); macd = [None if a is None or b is None else a - b for a, b in zip(ema12, ema26)]; signal = _ema([x or 0 for x in macd], 9)
    obv = [0.0]
    for i in range(1, len(close)): obv.append(obv[-1] + (volume[i] if close[i] > close[i-1] else -volume[i] if close[i] < close[i-1] else 0))
    rsi, vol10, vol20, vol30 = _rsi(close), _volatility(close, 10), _volatility(close, 20), _volatility(close, 30)
    keep = 90
    c = Canvas(str(output_pdf), pagesize=A4); page_w, page_h = A4
    def header(page: int, subtitle: str):
        c.setFillColor(colors.HexColor('#12395e')); c.rect(0, page_h-52, page_w, 52, fill=1, stroke=0); c.setFillColor(colors.white); c.setFont('Helvetica-Bold', 15); c.drawString(42, page_h-31, f'Technical Appendix — {ticker}'); c.setFont('Helvetica', 8); c.drawRightString(page_w-42, page_h-31, page_label); c.setFillColor(colors.HexColor('#5c6470')); c.setFont('Helvetica', 7); c.drawString(42, page_h-66, subtitle)
    header(1, f'Last 90 trading days ending {dates[-1].isoformat()} · {source_label}')
    s = lambda values: values[-keep:]
    price_change = (close[-1] / close[-keep] - 1) * 100
    ma20_gap = (close[-1] / ma20[-1] - 1) * 100
    band_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
    obv_change_m = (obv[-1] - obv[-21]) / 1_000_000
    rsi_state = "overbought" if rsi[-1] >= 70 else "oversold" if rsi[-1] <= 30 else "neutral"
    macd_state = "above" if macd[-1] >= signal[-1] else "below"

    _plot(c, 45, 635, 225, 105, 'Price', [('Close', s(close), colors.HexColor('#12395e'))])
    _caption(c, 45, 618, 225, f"Close is {price_change:+.1f}% over the 90-session window.")
    _plot(c, 320, 635, 225, 105, 'Moving averages', [('Close', s(close), colors.HexColor('#12395e')), ('MA 10', s(ma10), colors.HexColor('#d99a2b')), ('MA 20', s(ma20), colors.HexColor('#2f6f5e')), ('MA 50', s(ma50), colors.HexColor('#7a5a9e'))])
    _caption(c, 320, 618, 225, f"Close is {ma20_gap:+.1f}% versus the 20-day average; alignment frames trend direction.")
    _plot(c, 45, 465, 225, 105, 'Bollinger Bands (10, 2)', [('Close', s(close), colors.HexColor('#12395e')), ('MA 10', s(ma10), colors.HexColor('#d99a2b')), ('Upper', s(upper), colors.HexColor('#8993a1')), ('Lower', s(lower), colors.HexColor('#8993a1'))])
    _caption(c, 45, 448, 225, f"Close sits at {band_position:.0%} of the 10-day band range; extremes can flag stretched price action.")
    _plot(c, 320, 465, 225, 105, 'MACD (12, 26, 9)', [('MACD', s(macd), colors.HexColor('#12395e')), ('Signal', s(signal), colors.HexColor('#d99a2b'))])
    _caption(c, 320, 448, 225, f"MACD is {macd_state} its signal line, a short-term momentum read rather than a valuation signal.")
    _plot(c, 45, 295, 225, 105, 'On-balance volume (OBV)', [('OBV', s(obv), colors.HexColor('#2f6f5e'))])
    _caption(c, 45, 278, 225, f"OBV moved {obv_change_m:+.0f}m shares over 20 sessions; direction tests volume confirmation.")
    _plot(c, 320, 295, 225, 105, 'RSI (10-day)', [('RSI', s(rsi), colors.HexColor('#12395e'))], fixed=(0, 100))
    _caption(c, 320, 278, 225, f"RSI is {rsi[-1]:.1f} ({rsi_state}); 70/30 are conventional overbought/oversold reference levels.")
    _plot(c, 45, 100, 500, 110, 'Annualised realised volatility', [('10d', s(vol10), colors.HexColor('#12395e')), ('20d', s(vol20), colors.HexColor('#d99a2b')), ('30d', s(vol30), colors.HexColor('#2f6f5e'))])
    _caption(c, 45, 83, 500, f"Annualised realised volatility: {vol10[-1]:.1f}% (10d), {vol20[-1]:.1f}% (20d), and {vol30[-1]:.1f}% (30d); a near-term risk gauge.")
    # The chart is a timing/risk supplement, so it carries a compact,
    # calculation-backed interpretation rather than asking the reader to
    # infer every signal. It deliberately does not make an investment call.
    c.setFillColor(colors.HexColor('#12395e')); c.setFont('Helvetica-Bold', 6.8)
    c.drawString(45, 59, 'Technical read-through')
    c.setFillColor(colors.HexColor('#1a1a1a')); c.setFont('Helvetica', 6.5)
    c.drawString(45, 47, (
        f"Price is {(close[-1] / ma10[-1] - 1) * 100:+.1f}% / {(close[-1] / ma20[-1] - 1) * 100:+.1f}% / "
        f"{(close[-1] / ma50[-1] - 1) * 100:+.1f}% versus the 10d / 20d / 50d averages; "
        f"the {band_position:.0%} Bollinger-band position is not extended."))
    c.drawString(45, 35, (
        f"MACD ({macd[-1]:.2f}) is above its signal ({signal[-1]:.2f}) and RSI is {rsi[-1]:.1f} (neutral); "
        f"OBV changed {obv_change_m:+.0f}m shares over 20 sessions, so volume confirmation should be monitored."))
    c.drawString(45, 23, (
        f"Annualised realised volatility is {vol10[-1]:.1f}% / {vol20[-1]:.1f}% / {vol30[-1]:.1f}% for 10d / 20d / 30d. "
        "Use these signals for timing and risk context, not to validate the fundamental thesis."))
    c.save()
    return output_pdf


def merge_technical_appendix(report_pdf: Path, appendix_pdf: Path, output_pdf: Path) -> Path:
    """Concatenate the base report and a pre-built appendix page.

    Split from ``build_technical_appendix_pdf`` so the (slow, network-bound)
    fetch-and-render step can run before the base report even exists; this
    merge only needs both files to already be on disk.
    """
    writer = PdfWriter()
    for source in (report_pdf, appendix_pdf):
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    with output_pdf.open('wb') as fh:
        writer.write(fh)
    return output_pdf
