"""Append a sourced, reproducible technical-analysis supplement to a report PDF."""

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path

import requests
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen.canvas import Canvas


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
    c.setFont("Helvetica-Bold", 8); c.setFillColor(colors.HexColor("#12395e")); c.drawString(x, y + h + 8, title)
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
        c.setFillColor(color); c.setFont("Helvetica", 6.5); c.drawString(x + 4 + series.index((label, values, color)) * 74, y + h + 8, label)
    c.setFillColor(colors.HexColor("#5c6470")); c.setFont("Helvetica", 6.5)
    c.drawRightString(x - 3, y + h - 2, f"{hi:.1f}"); c.drawRightString(x - 3, y - 2, f"{lo:.1f}")


def append_technical_appendix(report_pdf: Path, output_pdf: Path, ticker: str, end_date: dt.date) -> Path:
    """Fetch daily Yahoo OHLCV, calculate indicators, and append three PDF pages."""
    start = end_date - dt.timedelta(days=250)
    response = requests.get("https://query1.finance.yahoo.com/v8/finance/chart/" + ticker,
        params={"period1": int(dt.datetime.combine(start, dt.time(), dt.UTC).timestamp()),
                "period2": int(dt.datetime.combine(end_date + dt.timedelta(days=1), dt.time(), dt.UTC).timestamp()),
                "interval": "1d"}, headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    response.raise_for_status()
    payload = response.json()["chart"]["result"][0]
    quote = payload["indicators"]["quote"][0]
    rows = [(dt.datetime.fromtimestamp(ts, dt.UTC).date(), quote["open"][i], quote["high"][i], quote["low"][i], quote["close"][i], quote["volume"][i])
            for i, ts in enumerate(payload["timestamp"]) if all(quote[key][i] is not None for key in ("open", "high", "low", "close", "volume"))]
    if len(rows) < 120:
        raise ValueError("Yahoo Finance returned fewer than 120 complete daily OHLCV observations.")
    dates, _, high, low, close, volume = map(list, zip(*rows))
    ma10, ma20, ma50 = _sma(close, 10), _sma(close, 20), _sma(close, 50)
    sd10 = _std(close, 10); upper = [None if m is None or s is None else m + 2*s for m, s in zip(ma10, sd10)]; lower = [None if m is None or s is None else m - 2*s for m, s in zip(ma10, sd10)]
    ema12, ema26 = _ema(close, 12), _ema(close, 26); macd = [None if a is None or b is None else a - b for a, b in zip(ema12, ema26)]; signal = _ema([x or 0 for x in macd], 9)
    obv = [0.0]
    for i in range(1, len(close)): obv.append(obv[-1] + (volume[i] if close[i] > close[i-1] else -volume[i] if close[i] < close[i-1] else 0))
    rsi, vol10, vol20, vol30 = _rsi(close), _volatility(close, 10), _volatility(close, 20), _volatility(close, 30)
    keep = 90
    appendix = output_pdf.with_suffix('.appendix.pdf')
    c = Canvas(str(appendix), pagesize=A4); page_w, page_h = A4
    def header(page: int, subtitle: str):
        c.setFillColor(colors.HexColor('#12395e')); c.rect(0, page_h-52, page_w, 52, fill=1, stroke=0); c.setFillColor(colors.white); c.setFont('Helvetica-Bold', 15); c.drawString(42, page_h-31, f'Technical Appendix — {ticker}'); c.setFont('Helvetica', 8); c.drawRightString(page_w-42, page_h-31, f'Page {page} of 1'); c.setFillColor(colors.HexColor('#5c6470')); c.setFont('Helvetica', 7); c.drawString(42, page_h-66, subtitle)
    header(1, f'Last 90 trading days ending {dates[-1].isoformat()} · Daily OHLCV: Yahoo Finance chart endpoint')
    s=lambda values: values[-keep:]
    _plot(c, 45, 635, 225, 105, 'Price', [('Close',s(close),colors.HexColor('#12395e'))])
    _plot(c, 320, 635, 225, 105, 'Moving averages', [('Close',s(close),colors.HexColor('#12395e')),('MA 10',s(ma10),colors.HexColor('#d99a2b')),('MA 20',s(ma20),colors.HexColor('#2f6f5e')),('MA 50',s(ma50),colors.HexColor('#7a5a9e'))])
    _plot(c, 45, 465, 225, 105, 'Bollinger Bands (10, 2)', [('Close',s(close),colors.HexColor('#12395e')),('MA 10',s(ma10),colors.HexColor('#d99a2b')),('Upper',s(upper),colors.HexColor('#8993a1')),('Lower',s(lower),colors.HexColor('#8993a1'))])
    _plot(c, 320, 465, 225, 105, 'MACD (12, 26, 9)', [('MACD',s(macd),colors.HexColor('#12395e')),('Signal',s(signal),colors.HexColor('#d99a2b'))])
    _plot(c, 45, 295, 225, 105, 'On-balance volume (OBV)', [('OBV',s(obv),colors.HexColor('#2f6f5e'))])
    _plot(c, 320, 295, 225, 105, 'RSI (10-day)', [('RSI',s(rsi),colors.HexColor('#12395e'))], fixed=(0,100))
    _plot(c, 45, 90, 500, 120, 'Annualised realised volatility', [('10d',s(vol10),colors.HexColor('#12395e')),('20d',s(vol20),colors.HexColor('#d99a2b')),('30d',s(vol30),colors.HexColor('#2f6f5e'))])
    # The chart is a timing/risk supplement, so it carries a compact,
    # calculation-backed interpretation rather than asking the reader to
    # infer every signal. It deliberately does not make an investment call.
    band_position = (close[-1] - lower[-1]) / (upper[-1] - lower[-1])
    obv_change_m = (obv[-1] - obv[-21]) / 1_000_000
    c.setFillColor(colors.HexColor('#12395e')); c.setFont('Helvetica-Bold', 6.8)
    c.drawString(45, 69, 'Technical read-through')
    c.setFillColor(colors.HexColor('#1a1a1a')); c.setFont('Helvetica', 6.5)
    c.drawString(45, 57, (
        f"Price is {(close[-1] / ma10[-1] - 1) * 100:+.1f}% / {(close[-1] / ma20[-1] - 1) * 100:+.1f}% / "
        f"{(close[-1] / ma50[-1] - 1) * 100:+.1f}% versus the 10d / 20d / 50d averages; "
        f"the {band_position:.0%} Bollinger-band position is not extended."))
    c.drawString(45, 45, (
        f"MACD ({macd[-1]:.2f}) is above its signal ({signal[-1]:.2f}) and RSI is {rsi[-1]:.1f} (neutral); "
        f"OBV changed {obv_change_m:+.0f}m shares over 20 sessions, so volume confirmation should be monitored."))
    c.drawString(45, 33, (
        f"Annualised realised volatility is {vol10[-1]:.1f}% / {vol20[-1]:.1f}% / {vol30[-1]:.1f}% for 10d / 20d / 30d. "
        "Use these signals for timing and risk context, not to validate the fundamental thesis."))
    c.save()
    writer = PdfWriter()
    for source in (report_pdf, appendix):
        for page in PdfReader(str(source)).pages:
            writer.add_page(page)
    with output_pdf.open('wb') as fh: writer.write(fh)
    appendix.unlink(missing_ok=True)
    return output_pdf
