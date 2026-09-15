"""Date and fiscal-period normalisation.

Providers report dates as ``2026-07-31``, ``Aug 19, 2026`` or ``19/08/2026``,
and fiscal periods as ``FY2026 Q2``, ``Q2 FY26`` or ``2026Q2``. Everything
downstream works with ``datetime.date`` and a canonical FiscalPeriod so that
temporal QA can compare periods meaningfully.
"""

from __future__ import annotations

import datetime as dt
import re

from ..domain.evidence import FiscalPeriod
from ..errors import NormalisationError

_DATE_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%d-%m-%Y",
    "%b %d, %Y",
    "%B %d, %Y",
    "%d %b %Y",
    "%d %B %Y",
    "%Y%m%d",
)

#: "second quarter fiscal 2027", "second quarter of fiscal year 2027" -> the
#: ordinal word a company's own prose (a press release, a web search answer -
#: see pipeline.freshness_check) uses for a quarter it would write as "Q2" in
#: a table. Rewritten to the compact form below before pattern matching, so a
#: prose label parses exactly as its compact equivalent would - this is what
#: lets freshness_check compare "Second Quarter Fiscal 2027" against "FY2026
#: Q2" instead of failing to parse it and conservatively reporting no mismatch.
_ORDINAL_QUARTER: dict[str, str] = {
    "first": "1", "second": "2", "third": "3", "fourth": "4",
    "1st": "1", "2nd": "2", "3rd": "3", "4th": "4",
}
_PROSE_QUARTER_RE = re.compile(
    r"^(?P<ord>first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter\s+"
    r"(?:of\s+)?fiscal(?:\s+year)?\s+(?P<year>\d{2,4})$"
)

# "FY2026 Q2", "FY26 Q2", "Q2 FY2026", "Q2 2026", "2026Q2", "FY2026"
_PERIOD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^fy\s*(?P<year>\d{2,4})\s*q(?P<quarter>[1-4])$"),
    re.compile(r"^q(?P<quarter>[1-4])\s*fy\s*(?P<year>\d{2,4})$"),
    re.compile(r"^q(?P<quarter>[1-4])\s*(?P<year>\d{4})$"),
    re.compile(r"^(?P<year>\d{4})\s*q(?P<quarter>[1-4])$"),
    re.compile(r"^fy\s*(?P<year>\d{2,4})$"),
    re.compile(r"^(?P<year>\d{4})$"),
)


def normalise_date(raw: object, *, field_name: str = "date") -> dt.date | None:
    """Parse a provider date string into a date. ``None`` passes through.

    Raises NormalisationError for a non-empty value that cannot be parsed, so a
    bad date never silently becomes "today".
    """
    if raw is None:
        return None
    if isinstance(raw, dt.datetime):
        return raw.date()
    if isinstance(raw, dt.date):
        return raw

    text = str(raw).strip()
    if not text:
        return None

    # ISO datetimes, with or without a trailing Z.
    iso_candidate = text.replace("Z", "+00:00")
    try:
        return dt.datetime.fromisoformat(iso_candidate).date()
    except ValueError:
        pass

    for fmt in _DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue

    raise NormalisationError(field_name, raw, "value is not a recognised date format")


def normalise_timestamp(raw: object, *, field_name: str = "timestamp") -> dt.datetime:
    """Parse a timestamp into a timezone-aware UTC datetime."""
    if isinstance(raw, dt.datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=dt.UTC)
    if raw is None:
        return dt.datetime.now(dt.UTC)
    text = str(raw).strip().replace("Z", "+00:00")
    try:
        parsed = dt.datetime.fromisoformat(text)
    except ValueError as exc:
        raise NormalisationError(field_name, raw, "value is not a recognised timestamp") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=dt.UTC)


def _expand_year(year_text: str) -> int:
    year = int(year_text)
    if year < 100:
        # Two-digit fiscal years: 26 -> 2026. Fine for this prototype's horizon.
        return 2000 + year
    return year


def normalise_fiscal_period(
    raw_period: object,
    period_end: object = None,
    *,
    field_name: str = "period",
) -> FiscalPeriod | None:
    """Build a canonical FiscalPeriod from a provider period label and end date.

    A period end date alone is enough to produce a dated (if unlabelled) period,
    which keeps evidence usable when a provider omits the label.
    """
    end_date = normalise_date(period_end, field_name=f"{field_name}_end")

    if raw_period is None or not str(raw_period).strip():
        if end_date is None:
            return None
        return FiscalPeriod(label=f"Period ended {end_date.isoformat()}", period_end=end_date)

    text = re.sub(r"\s+", " ", str(raw_period).strip().lower())
    prose_match = _PROSE_QUARTER_RE.match(text)
    if prose_match:
        text = f"fy{prose_match.group('year')} q{_ORDINAL_QUARTER[prose_match.group('ord')]}"
    for pattern in _PERIOD_PATTERNS:
        match = pattern.match(text)
        if match is None:
            continue
        year = _expand_year(match.group("year"))
        groups = match.groupdict()
        quarter = int(groups["quarter"]) if groups.get("quarter") else None
        if quarter is None:
            return FiscalPeriod(
                label=f"FY{year}", fiscal_year=year, fiscal_quarter=None,
                period_end=end_date, is_annual=True,
            )
        return FiscalPeriod(
            label=f"FY{year} Q{quarter}", fiscal_year=year, fiscal_quarter=quarter,
            period_end=end_date, is_annual=False,
        )

    # Unrecognised but non-empty: keep the provider's label rather than discard
    # the period, and let temporal QA flag it as uncomparable.
    return FiscalPeriod(label=str(raw_period).strip(), period_end=end_date)


def periods_are_comparable(left: FiscalPeriod | None, right: FiscalPeriod | None) -> bool:
    """Whether two periods can be compared without a caveat.

    Comparable means the same quarter-vs-quarter or annual-vs-annual shape, and
    both sides fully parsed.
    """
    if left is None or right is None:
        return False
    if left.fiscal_year is None or right.fiscal_year is None:
        return False
    return left.is_annual == right.is_annual and (
        (left.fiscal_quarter is None) == (right.fiscal_quarter is None)
    )


def year_ago_period(period: FiscalPeriod) -> str | None:
    """Canonical label of the same period one fiscal year earlier."""
    if period.fiscal_year is None:
        return None
    if period.fiscal_quarter is None:
        return f"FY{period.fiscal_year - 1}"
    return f"FY{period.fiscal_year - 1} Q{period.fiscal_quarter}"


def previous_quarter_period(period: FiscalPeriod) -> str | None:
    """Canonical label of the immediately preceding fiscal quarter."""
    if period.fiscal_year is None or period.fiscal_quarter is None:
        return None
    if period.fiscal_quarter == 1:
        return f"FY{period.fiscal_year - 1} Q4"
    return f"FY{period.fiscal_year} Q{period.fiscal_quarter - 1}"


def next_quarter_period(period: FiscalPeriod) -> str | None:
    """Canonical label of the next fiscal quarter."""
    if period.fiscal_year is None or period.fiscal_quarter is None:
        return None
    if period.fiscal_quarter == 4:
        return f"FY{period.fiscal_year + 1} Q1"
    return f"FY{period.fiscal_year} Q{period.fiscal_quarter + 1}"
