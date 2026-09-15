"""Numeric and unit normalisation.

Providers hand back numbers as ``62_300_000_000``, ``"$62,300,000,000"``,
``"62.3B"`` or ``"74.8%"``. Everything downstream needs one representation, so
this module is the single place that parses them - and it raises
NormalisationError rather than guessing, so a malformed value is rejected
instead of quietly becoming plausible-looking evidence.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from ..errors import NormalisationError

#: Canonical unit strings used throughout the Evidence Store.
UNIT_CURRENCY = "USD"
UNIT_PERCENT = "pct"
UNIT_MULTIPLE = "x"
UNIT_COUNT = "count"
UNIT_PER_SHARE = "USD/share"
UNIT_TEXT = "text"
UNIT_PERCENTAGE_POINTS = "pp"

_SCALE_SUFFIXES: dict[str, float] = {
    "k": 1e3, "thousand": 1e3,
    "m": 1e6, "mm": 1e6, "mn": 1e6, "million": 1e6,
    "b": 1e9, "bn": 1e9, "billion": 1e9,
    "t": 1e12, "tn": 1e12, "trillion": 1e12,
    # Not a magnitude at all - "31.4x" is how a multiple (P/E, EV/Sales, ...)
    # is conventionally written. Scale 1.0 is a no-op, so the digits pass
    # through unchanged; the caller decides the resulting unit is UNIT_MULTIPLE
    # from the metric's identity, exactly as it already does for a bare float.
    "x": 1.0,
}

_CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY"}

_NUMERIC_RE = re.compile(
    r"^\s*(?P<sign>[-+(]?)\s*"
    r"(?P<symbol>[$€£¥]?)\s*"
    r"(?P<number>\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d*\.?\d+)\s*"
    r"(?P<suffix>[a-zA-Z]{0,8})\s*"
    r"(?P<percent>%?)\s*\)?\s*$"
)


@dataclass(frozen=True, slots=True)
class ParsedNumber:
    """A parsed numeric value plus what the source string told us about it."""

    value: float
    is_percent: bool = False
    currency: str | None = None
    scale_applied: float = 1.0


def parse_number(raw: object, *, field_name: str = "value") -> ParsedNumber:
    """Parse a provider value into a float, or raise NormalisationError.

    Handles ints/floats, currency symbols, thousands separators, magnitude
    suffixes (``62.3B``), trailing percent signs and parenthesised negatives.
    """
    if raw is None:
        raise NormalisationError(field_name, raw, "value is null")

    if isinstance(raw, bool):
        raise NormalisationError(field_name, raw, "boolean is not a numeric measurement")

    if isinstance(raw, (int, float)):
        value = float(raw)
        if math.isnan(value) or math.isinf(value):
            raise NormalisationError(field_name, raw, "value is not finite")
        return ParsedNumber(value=value)

    if not isinstance(raw, str):
        raise NormalisationError(field_name, raw, f"unsupported type {type(raw).__name__}")

    text = raw.strip()
    if not text:
        raise NormalisationError(field_name, raw, "value is an empty string")

    match = _NUMERIC_RE.match(text)
    if match is None:
        raise NormalisationError(field_name, raw, "value is not parseable as a number")

    number_text = match.group("number").replace(",", "")
    if not number_text or number_text == ".":
        raise NormalisationError(field_name, raw, "value has no digits")
    value = float(number_text)

    sign = match.group("sign")
    negative = sign in {"-", "("} or (sign == "(" and text.endswith(")"))
    if text.startswith("(") and text.rstrip().endswith(")"):
        negative = True
    if negative:
        value = -value

    suffix = match.group("suffix").lower()
    scale = 1.0
    if suffix:
        if suffix not in _SCALE_SUFFIXES:
            raise NormalisationError(field_name, raw, f"unrecognised magnitude suffix {suffix!r}")
        scale = _SCALE_SUFFIXES[suffix]
        value *= scale

    currency = _CURRENCY_SYMBOLS.get(match.group("symbol")) if match.group("symbol") else None
    return ParsedNumber(
        value=value,
        is_percent=bool(match.group("percent")),
        currency=currency,
        scale_applied=scale,
    )


def normalise_percent(value: float, *, source_looked_like_percent: bool) -> float:
    """Return a percentage in *percent* units (74.8 means 74.8%).

    Providers are inconsistent about whether a margin is ``0.748`` or ``74.8``.
    The rule applied here: a value that arrived with an explicit ``%`` is already
    in percent units; a bare value in ``[-1, 1]`` is treated as a decimal
    fraction and scaled. Anything else is left alone.
    """
    if source_looked_like_percent:
        return value
    if -1.0 <= value <= 1.0 and value != 0:
        return value * 100.0
    return value


def normalise_currency(raw: object, default: str | None = None) -> str | None:
    """Uppercase an ISO currency code, accepting a symbol as a shorthand."""
    if raw is None:
        return default.upper() if default else None
    text = str(raw).strip()
    if not text:
        return default.upper() if default else None
    if text in _CURRENCY_SYMBOLS:
        return _CURRENCY_SYMBOLS[text]
    if re.fullmatch(r"[A-Za-z]{3}", text):
        return text.upper()
    return default.upper() if default else None


def format_currency(value: float, currency: str = "USD") -> str:
    """Compact currency formatting used in tables and narrative text."""
    symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency, "")
    magnitude = abs(value)
    sign = "-" if value < 0 else ""
    if magnitude >= 1e12:
        return f"{sign}{symbol}{magnitude / 1e12:.2f}tn"
    if magnitude >= 1e9:
        return f"{sign}{symbol}{magnitude / 1e9:.2f}bn"
    if magnitude >= 1e6:
        return f"{sign}{symbol}{magnitude / 1e6:.1f}mn"
    if magnitude >= 1e3:
        return f"{sign}{symbol}{magnitude:,.0f}"
    return f"{sign}{symbol}{magnitude:,.2f}"


def format_number(value: float, unit: str, currency: str | None = "USD") -> str:
    """Format a normalised value for display according to its canonical unit."""
    if unit == UNIT_PERCENT:
        return f"{value:.1f}%"
    if unit == UNIT_PERCENTAGE_POINTS:
        return f"{value:+.1f}pp"
    if unit == UNIT_MULTIPLE:
        return f"{value:.1f}x"
    if unit == UNIT_COUNT:
        return f"{value:,.0f}"
    if unit == UNIT_PER_SHARE:
        symbol = {"USD": "$", "EUR": "€", "GBP": "£"}.get(currency or "", "")
        return f"{symbol}{value:,.2f}"
    if unit == UNIT_CURRENCY:
        return format_currency(value, currency or "USD")
    return f"{value:,.2f}"


def format_signed_percent(value: float) -> str:
    return f"{value:+.1f}%"
