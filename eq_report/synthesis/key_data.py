"""The first-page key-data panel.

A sell-side first page carries the company's headline figures in a narrow
right-hand column - how the market prices it, what it last reported, what it
has guided to - so the reader has the numbers in view before reading a word of
the argument. This module builds the same thing from the Evidence Store.

It exists as its own module because it is the *only* place in the report where
a metric may appear without a sentence around it, and because it is what makes
the body tables shrinkable: once the panel states the price, the market
capitalisation and the multiples, no section table needs to repeat them. The
synthesizer therefore also uses :data:`PANEL_METRICS` to suppress those rows
downstream (see ``Synthesizer._tables``).

Nothing here computes: every value is read off an evidence row and formatted
by the same ``format_number`` the rest of the pipeline uses. A metric the
Evidence Store does not hold is simply left out, so the panel shrinks to fit
the evidence rather than printing blanks.
"""

from __future__ import annotations

from ..domain.evidence import EvidenceItem
from ..domain.report import KeyDataGroup, KeyDataItem, KeyDataPanel
from ..evidence.reader import EvidenceReader
from ..normalisation import canonical_metrics as cm
from ..normalisation.units import format_number

#: Metrics the panel already states, and which body tables must therefore not
#: repeat. Kept next to the panel definition so the two can never drift.
PANEL_METRICS: frozenset[str] = frozenset({
    cm.SHARE_PRICE, cm.MARKET_CAP, cm.ENTERPRISE_VALUE,
    cm.AVG_VOLUME_30D, cm.PRICE_52W_HIGH, cm.PRICE_52W_LOW,
    cm.FORWARD_PE, cm.TRAILING_PE, cm.EV_TO_SALES, cm.PRICE_TARGET,
    cm.CONSENSUS_RATING,
    cm.GUIDANCE_REVENUE, cm.GUIDANCE_GROSS_MARGIN,
})


def build_key_data_panel(reader: EvidenceReader) -> KeyDataPanel | None:
    """Assemble the panel from whatever the Evidence Store holds.

    Returns ``None`` when there is not enough to be worth printing, so a run
    with no market data does not render an empty box.
    """
    latest = reader.latest_reported_period()

    price = reader.numeric(cm.SHARE_PRICE)
    groups = [
        _group("Market data", [
            _line(reader, cm.SHARE_PRICE, "Price"),
            _line(reader, cm.MARKET_CAP, "Market cap"),
            _line(reader, cm.ENTERPRISE_VALUE, "Enterprise value"),
            _range_line(reader),
            _line(reader, cm.AVG_VOLUME_30D, "Avg volume (30d)"),
        ]),
        _group("Valuation", [
            _line(reader, cm.FORWARD_PE, "Forward P/E"),
            _line(reader, cm.TRAILING_PE, "Trailing P/E"),
            _line(reader, cm.EV_TO_SALES, "EV/Sales"),
            _line(reader, cm.PRICE_TARGET, "Consensus target"),
            _rating_line(reader),
        ]),
        _group(f"Reported {latest}" if latest else "Latest reported", [
            _line(reader, cm.REVENUE, "Revenue", period=latest),
            _line(reader, cm.GROSS_MARGIN, "Gross margin", period=latest),
            _line(reader, cm.OPERATING_MARGIN, "Operating margin", period=latest),
            _line(reader, cm.EPS_DILUTED, "Diluted EPS", period=latest),
            _line(reader, cm.FREE_CASH_FLOW, "Free cash flow", period=latest),
            _net_cash_line(reader, latest),
        ]),
        _guidance_group(reader),
    ]

    panel = KeyDataPanel(
        groups=tuple(g for g in groups if g is not None),
        as_of=price.as_of.isoformat() if price and price.as_of else "",
    )
    return None if panel.is_empty else panel


# -- group builders ---------------------------------------------------------
def _group(title: str, items: list[KeyDataItem | None]) -> KeyDataGroup | None:
    present = tuple(i for i in items if i is not None)
    return KeyDataGroup(title=title, items=present) if present else None


def _guidance_group(reader: EvidenceReader) -> KeyDataGroup | None:
    """Guidance, titled with the period actually being guided to.

    The guided period is read off the evidence rather than assumed to be the
    quarter after the last reported one, since a company may guide to a full
    year or skip a quarter.
    """
    revenue = reader.numeric(cm.GUIDANCE_REVENUE)
    margin = reader.numeric(cm.GUIDANCE_GROSS_MARGIN)
    period = next(
        (i.period_label for i in (revenue, margin) if i is not None and i.period_label), None)
    return _group(
        f"Guidance {period}" if period else "Guidance",
        [_item(revenue, "Revenue"), _item(margin, "Gross margin")],
    )


# -- line builders ----------------------------------------------------------
def _line(
    reader: EvidenceReader, metric: str, label: str, *, period: str | None = None
) -> KeyDataItem | None:
    return _item(reader.numeric(metric, period), label)


def _item(item: EvidenceItem | None, label: str) -> KeyDataItem | None:
    if item is None or item.value is None:
        return None
    return KeyDataItem(
        label=label,
        value=format_number(item.value, item.unit or "", item.currency),
    )


def _range_line(reader: EvidenceReader) -> KeyDataItem | None:
    """The 52-week high and low as one range, the way a data box prints it."""
    high = reader.numeric(cm.PRICE_52W_HIGH)
    low = reader.numeric(cm.PRICE_52W_LOW)
    if high is None or low is None or high.value is None or low.value is None:
        return None
    unit = high.unit or ""
    return KeyDataItem(
        label="52-week range",
        value=(
            f"{format_number(low.value, unit, low.currency)} - "
            f"{format_number(high.value, unit, high.currency)}"
        ),
    )


def _net_cash_line(reader: EvidenceReader, period: str | None) -> KeyDataItem | None:
    """Net cash (or net debt), which is what a reader actually wants from a
    balance sheet in a box this size - two separate gross lines are not."""
    cash = reader.numeric(cm.CASH_AND_EQUIVALENTS, period)
    debt = reader.numeric(cm.TOTAL_DEBT, period)
    if cash is None or debt is None or cash.value is None or debt.value is None:
        return None
    net = cash.value - debt.value
    return KeyDataItem(
        label="Net cash" if net >= 0 else "Net debt",
        value=format_number(abs(net), cash.unit or "", cash.currency),
    )


def _rating_line(reader: EvidenceReader) -> KeyDataItem | None:
    """The consensus rating, which is stored as text rather than a number."""
    items = reader.query(metric=cm.CONSENSUS_RATING, ticker=reader.ticker)
    for item in items:
        text = (item.claim_text or "").strip()
        if text:
            return KeyDataItem(label="Consensus rating", value=text)
    return None
