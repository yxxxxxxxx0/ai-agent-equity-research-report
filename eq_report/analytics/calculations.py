"""Pure calculation functions used by the Analytics Engine.

Every function here is deterministic, takes plain numbers and returns a plain
number. No evidence objects, no store, no I/O - which is exactly what makes them
unit-testable and what lets the QA layer recompute them independently.

An invalid input raises AnalyticsError rather than returning a sentinel, so a
missing denominator becomes a documented data gap instead of a zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from ..errors import AnalyticsError

#: Formula strings are stored on every AnalyticsResult so the report can show
#: its working and QA can re-derive the number.
F_PCT_CHANGE = "(current - prior) / abs(prior) * 100"
F_MARGIN_CHANGE = "current_pct - prior_pct"
F_SURPRISE = "(actual - consensus) / abs(consensus) * 100"
F_RATIO = "numerator / denominator"
F_CONTRIBUTION = "part / total * 100"
F_PREMIUM = "(value - benchmark) / abs(benchmark) * 100"
F_MEAN = "sum(values) / count(values)"
F_REVISION_NET = "(up - down) / (up + down) * 100"
F_ROLLING_MAX = "max(values)"
F_ROLLING_MIN = "min(values)"


def _require_number(value: object, name: str) -> float:
    if value is None:
        raise AnalyticsError(f"{name} is missing")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AnalyticsError(f"{name} is not numeric: {value!r}")
    return float(value)


def _require_nonzero(value: float, name: str) -> float:
    if value == 0:
        raise AnalyticsError(f"{name} is zero; the ratio is undefined")
    return value


def pct_change(current: object, prior: object) -> float:
    """Percentage change from ``prior`` to ``current``, in percent units."""
    current_value = _require_number(current, "current")
    prior_value = _require_nonzero(_require_number(prior, "prior"), "prior")
    return (current_value - prior_value) / abs(prior_value) * 100.0


def margin_change_pp(current_pct: object, prior_pct: object) -> float:
    """Change in a margin, in percentage points.

    Margins are differenced, never divided: a move from 75.7% to 74.8% is
    -0.9pp, not -1.2%.
    """
    return _require_number(current_pct, "current_pct") - _require_number(prior_pct, "prior_pct")


def surprise_pct(actual: object, consensus: object) -> float:
    """Beat or miss versus consensus, in percent units."""
    actual_value = _require_number(actual, "actual")
    consensus_value = _require_nonzero(_require_number(consensus, "consensus"), "consensus")
    return (actual_value - consensus_value) / abs(consensus_value) * 100.0


def ratio(numerator: object, denominator: object) -> float:
    """A plain ratio, e.g. free cash flow conversion or a valuation multiple."""
    numerator_value = _require_number(numerator, "numerator")
    denominator_value = _require_nonzero(
        _require_number(denominator, "denominator"), "denominator")
    return numerator_value / denominator_value


def cash_conversion(free_cash_flow: object, net_income: object) -> float:
    """Free cash flow as a percentage of net income."""
    return ratio(free_cash_flow, net_income) * 100.0


def contribution_pct(part: object, total: object) -> float:
    """A component's share of a total, in percent units."""
    part_value = _require_number(part, "part")
    total_value = _require_nonzero(_require_number(total, "total"), "total")
    return part_value / total_value * 100.0


def premium_pct(value: object, benchmark: object) -> float:
    """Premium (positive) or discount (negative) versus a benchmark, in percent."""
    value_number = _require_number(value, "value")
    benchmark_number = _require_nonzero(_require_number(benchmark, "benchmark"), "benchmark")
    return (value_number - benchmark_number) / abs(benchmark_number) * 100.0


def mean(values: Sequence[object]) -> float:
    """Arithmetic mean; raises if the sequence is empty."""
    numbers = [_require_number(v, "value") for v in values]
    if not numbers:
        raise AnalyticsError("cannot take the mean of an empty sequence")
    return sum(numbers) / len(numbers)


def price_return_pct(current_price: object, past_price: object) -> float:
    """Simple price return between two closes, in percent units."""
    return pct_change(current_price, past_price)


def rolling_max(values: Sequence[object]) -> float:
    """Highest value in a window; raises on an empty window."""
    numbers = [_require_number(v, "value") for v in values]
    if not numbers:
        raise AnalyticsError("cannot take the max of an empty sequence")
    return max(numbers)


def rolling_min(values: Sequence[object]) -> float:
    """Lowest value in a window; raises on an empty window."""
    numbers = [_require_number(v, "value") for v in values]
    if not numbers:
        raise AnalyticsError("cannot take the min of an empty sequence")
    return min(numbers)


def net_revision_pct(up: object, down: object) -> float:
    """Net estimate-revision breadth, in percent units.

    ``+100`` means every revision was upward, ``-100`` every revision downward.
    """
    up_count = _require_number(up, "up")
    down_count = _require_number(down, "down")
    total = up_count + down_count
    if total == 0:
        raise AnalyticsError("no estimate revisions to measure")
    return (up_count - down_count) / total * 100.0


# ---------------------------------------------------------------------------
# Trend detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class Trend:
    """A classified trend over an ordered series."""

    direction: str          # "accelerating" | "improving" | "stable" | "slowing" | "deteriorating"
    slope: float            # average change per step, in the series' own units
    first: float
    last: float
    points: int

    @property
    def is_positive(self) -> bool:
        return self.direction in {"accelerating", "improving"}


def detect_trend(
    values: Sequence[object],
    *,
    stable_band: float = 1.0,
    strong_band: float = 5.0,
) -> Trend:
    """Classify the direction of an ordered series (oldest first).

    Deliberately simple: the average step change, bucketed against two
    thresholds expressed in the series' own units. Enough to say "margins are
    slipping" without pretending to a statistical model.
    """
    numbers = [_require_number(v, "value") for v in values]
    if len(numbers) < 2:
        raise AnalyticsError("a trend needs at least two observations")

    # Intentionally not strict: pairing a list with its own tail is offset by one.
    steps = [b - a for a, b in zip(numbers, numbers[1:])]
    slope = sum(steps) / len(steps)

    if abs(slope) <= stable_band:
        direction = "stable"
    elif slope > strong_band:
        direction = "accelerating"
    elif slope > 0:
        direction = "improving"
    elif slope < -strong_band:
        direction = "deteriorating"
    else:
        direction = "slowing"

    return Trend(
        direction=direction,
        slope=slope,
        first=numbers[0],
        last=numbers[-1],
        points=len(numbers),
    )


def growth_is_accelerating(current_growth_pct: object, prior_growth_pct: object) -> bool:
    """Whether the growth rate itself increased between two periods."""
    return _require_number(current_growth_pct, "current_growth_pct") > _require_number(
        prior_growth_pct, "prior_growth_pct")
