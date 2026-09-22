"""Canonical metric vocabulary.

Providers disagree about naming: ``Revenue``, ``revenue``, ``total_revenue`` and
``sales`` all mean the same thing. Everything downstream of normalisation refers
to metrics by the canonical ids defined here, while the evidence item keeps the
provider's original name in ``raw_metric`` so provenance is never lost.
"""

from __future__ import annotations

import re

from ..domain.enums import EvidenceCategory

# ---------------------------------------------------------------------------
# Canonical metric ids
# ---------------------------------------------------------------------------

# Market data
SHARE_PRICE = "share_price"
PREVIOUS_CLOSE = "previous_close"
MARKET_CAP = "market_cap"
ENTERPRISE_VALUE = "enterprise_value"
VOLUME = "volume"
AVG_VOLUME_30D = "avg_volume_30d"
SHARES_OUTSTANDING = "shares_outstanding"
PRICE_52W_HIGH = "price_52w_high"
PRICE_52W_LOW = "price_52w_low"
PRICE_HISTORY_POINT = "price_history_point"
# Raw daily bar high/low (one per trading day) - distinct from the 52-week
# aggregate above, which the Analytics Engine derives from a rolling window
# over these (see analytics/calculations.py::price_52w_range).
DAILY_HIGH = "daily_high"
DAILY_LOW = "daily_low"
FORWARD_PE = "forward_pe"
TRAILING_PE = "trailing_pe"
EV_TO_SALES = "ev_to_sales"
EV_TO_EBITDA = "ev_to_ebitda"
PRICE_TO_SALES = "price_to_sales"

# Consensus / estimates
CONSENSUS_REVENUE = "consensus_revenue"
CONSENSUS_EPS = "consensus_eps"
CONSENSUS_RATING = "consensus_rating"
REVENUE_GROWTH_YOY_REPORTED = "revenue_growth_yoy_reported"
PRICE_TARGET = "price_target"
ESTIMATE_REVISION_COUNT_UP = "estimate_revisions_up"
ESTIMATE_REVISION_COUNT_DOWN = "estimate_revisions_down"
FORWARD_REVENUE_ESTIMATE = "forward_revenue_estimate"
FORWARD_EPS_ESTIMATE = "forward_eps_estimate"

# Fundamentals
REVENUE = "revenue"
COST_OF_REVENUE = "cost_of_revenue"
GROSS_PROFIT = "gross_profit"
GROSS_MARGIN = "gross_margin"
OPERATING_INCOME = "operating_income"
OPERATING_MARGIN = "operating_margin"
NET_INCOME = "net_income"
NET_MARGIN = "net_margin"
EPS_DILUTED = "eps_diluted"
OPERATING_CASH_FLOW = "operating_cash_flow"
CAPEX = "capex"
FREE_CASH_FLOW = "free_cash_flow"
CASH_AND_EQUIVALENTS = "cash_and_equivalents"
TOTAL_DEBT = "total_debt"
RND_EXPENSE = "rnd_expense"

# Segment / KPI (segment name is carried in evidence metadata)
SEGMENT_REVENUE = "segment_revenue"
SEGMENT_OPERATING_INCOME = "segment_operating_income"
KPI = "kpi"

# Guidance
GUIDANCE_REVENUE = "guidance_revenue"
GUIDANCE_GROSS_MARGIN = "guidance_gross_margin"
GUIDANCE_OPEX = "guidance_opex"


# ---------------------------------------------------------------------------
# Alias table: provider name (normalised) -> canonical id
# ---------------------------------------------------------------------------

_ALIASES: dict[str, str] = {}


def _register(canonical: str, *aliases: str) -> None:
    _ALIASES[_slug(canonical)] = canonical
    for alias in aliases:
        _ALIASES[_slug(alias)] = canonical


def _slug(name: str) -> str:
    """Lowercase, strip punctuation, collapse separators.

    ``"Total Revenue"``, ``"total_revenue"`` and ``"totalRevenue"`` all slug to
    ``"total revenue"``.
    """
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(name))
    cleaned = re.sub(r"[^a-z0-9]+", " ", spaced.lower())
    return cleaned.strip()


_register(SHARE_PRICE, "price", "last_price", "close", "closing_price", "regularMarketPrice")
_register(PREVIOUS_CLOSE, "prev_close", "previousClose")
_register(MARKET_CAP, "marketcap", "market_capitalisation", "market_capitalization")
_register(ENTERPRISE_VALUE, "ev")
_register(VOLUME, "trading_volume", "regularMarketVolume")
_register(AVG_VOLUME_30D, "average_volume_30d", "avg_daily_volume_30d")
_register(SHARES_OUTSTANDING, "diluted_shares_outstanding", "sharesOutstanding")
_register(PRICE_52W_HIGH, "fifty_two_week_high", "52_week_high")
_register(PRICE_52W_LOW, "fifty_two_week_low", "52_week_low")
_register(PRICE_HISTORY_POINT, "historical_close", "price_history")
# MegadataAPI's daily OHLCV feed (/api/bbg/ohlcv/data) uses Bloomberg field
# codes rather than the aliases above; PX_LAST/PX_VOLUME are just another
# name for the same share price / volume, while PX_HIGH/PX_LOW are new
# canonical metrics (a day's high/low, not the 52-week aggregate).
_register(SHARE_PRICE, "px_last")
_register(VOLUME, "px_volume")
_register(DAILY_HIGH, "px_high")
_register(DAILY_LOW, "px_low")
_register(FORWARD_PE, "forward_p_e", "pe_forward", "forwardPE")
_register(TRAILING_PE, "pe_ratio", "trailing_p_e", "trailingPE", "pe")
_register(EV_TO_SALES, "ev_sales", "ev_revenue", "enterprise_value_to_revenue")
_register(EV_TO_EBITDA, "ev_ebitda")
_register(PRICE_TO_SALES, "ps_ratio", "p_s")

_register(CONSENSUS_REVENUE, "revenue_consensus", "street_revenue", "consensus_sales",
          "best_sales_mean")
_register(CONSENSUS_EPS, "eps_consensus", "street_eps", "best_eps_mean", "estimated_eps")
_register(CONSENSUS_RATING, "analyst_rating", "rating")
_register(REVENUE_GROWTH_YOY_REPORTED, "revenue_growth_yoy_pct", "revenue_growth",
          "sales_growth", "revenue_growth_pct")
_register(PRICE_TARGET, "target_price", "mean_price_target", "consensus_price_target",
          "best_target_price")
_register(ESTIMATE_REVISION_COUNT_UP, "revisions_up", "upward_revisions")
_register(ESTIMATE_REVISION_COUNT_DOWN, "revisions_down", "downward_revisions")
_register(FORWARD_REVENUE_ESTIMATE, "next_quarter_revenue_estimate", "revenue_estimate")
_register(FORWARD_EPS_ESTIMATE, "next_quarter_eps_estimate", "eps_estimate")

_register(REVENUE, "total_revenue", "sales", "net_revenue", "net_sales", "totalRevenue",
          "revenues", "turnover", "sales_rev_turn", "is_sales_and_srvices_revenues")
_register(COST_OF_REVENUE, "cogs", "cost_of_goods_sold", "cost_of_sales",
          "is_cog_and_services_sold")
_register(GROSS_PROFIT, "gross_income")
_register(GROSS_MARGIN, "gross_margin_pct", "gross_profit_margin", "gm")
_register(OPERATING_INCOME, "operating_profit", "ebit", "income_from_operations",
          "operating_income_loss", "is_oper_inc")
_register(OPERATING_MARGIN, "operating_margin_pct", "ebit_margin", "om", "oper_margin")
_register(NET_INCOME, "net_profit", "net_earnings", "profit_after_tax", "netIncome",
          "net_income_gaap")
_register(NET_MARGIN, "net_margin_pct", "net_profit_margin", "prof_margin")
_register(EPS_DILUTED, "eps", "diluted_eps", "earnings_per_share", "eps_diluted_gaap",
          "diluted_eps_gaap", "is_diluted_eps", "reported_eps")
_register(OPERATING_CASH_FLOW, "cash_flow_from_operations", "cfo", "operating_cashflow",
          "cash_from_operating_activities", "cf_cash_from_oper")
_register(CAPEX, "capital_expenditure", "capital_expenditures", "purchases_of_ppe",
          "cf_purchase_of_fixed_prod_assets")
_register(FREE_CASH_FLOW, "fcf", "free_cashflow", "cf_free_cash_flow")
_register(CASH_AND_EQUIVALENTS, "cash", "cash_and_cash_equivalents",
          "cash_and_marketable_securities", "bs_cash_near_cash_item")
_register(TOTAL_DEBT, "debt", "gross_debt", "total_borrowings")
_register(RND_EXPENSE, "r_and_d", "research_and_development", "rd_expense")

_register(SEGMENT_REVENUE, "segment_sales", "business_segment_revenue")
_register(SEGMENT_OPERATING_INCOME, "segment_operating_profit")
_register(KPI, "operating_kpi", "company_kpi")

_register(GUIDANCE_REVENUE, "revenue_guidance", "guided_revenue", "outlook_revenue")
_register(GUIDANCE_GROSS_MARGIN, "gross_margin_guidance", "guided_gross_margin")
_register(GUIDANCE_OPEX, "opex_guidance", "guided_operating_expenses")


# ---------------------------------------------------------------------------
# Metric properties
# ---------------------------------------------------------------------------

#: Canonical metrics whose natural unit is a percentage.
PERCENT_METRICS: frozenset[str] = frozenset({
    GROSS_MARGIN, OPERATING_MARGIN, NET_MARGIN, GUIDANCE_GROSS_MARGIN,
    REVENUE_GROWTH_YOY_REPORTED,
})

#: Canonical metrics whose value is text, not a number (e.g. a rating).
#: These are stored as document-style evidence with a claim_text.
TEXT_METRICS: frozenset[str] = frozenset({
    CONSENSUS_RATING,
})

#: Canonical metrics expressed as a multiple (``x``) rather than a currency.
MULTIPLE_METRICS: frozenset[str] = frozenset({
    FORWARD_PE, TRAILING_PE, EV_TO_SALES, EV_TO_EBITDA, PRICE_TO_SALES,
})

#: Canonical metrics that are counts, not money.
COUNT_METRICS: frozenset[str] = frozenset({
    VOLUME, AVG_VOLUME_30D, SHARES_OUTSTANDING,
    ESTIMATE_REVISION_COUNT_UP, ESTIMATE_REVISION_COUNT_DOWN,
})

#: Canonical metrics that are per-share currency amounts.
PER_SHARE_METRICS: frozenset[str] = frozenset({
    EPS_DILUTED, CONSENSUS_EPS, FORWARD_EPS_ESTIMATE, SHARE_PRICE, PREVIOUS_CLOSE,
    PRICE_TARGET, PRICE_52W_HIGH, PRICE_52W_LOW, PRICE_HISTORY_POINT,
    DAILY_HIGH, DAILY_LOW,
})

#: Which evidence category a canonical metric belongs to.
_CATEGORY_BY_METRIC: dict[str, EvidenceCategory] = {}
for _m in (SHARE_PRICE, PREVIOUS_CLOSE, MARKET_CAP, ENTERPRISE_VALUE, VOLUME, AVG_VOLUME_30D,
           SHARES_OUTSTANDING, PRICE_52W_HIGH, PRICE_52W_LOW, PRICE_HISTORY_POINT,
           DAILY_HIGH, DAILY_LOW, FORWARD_PE,
           TRAILING_PE, EV_TO_SALES, EV_TO_EBITDA, PRICE_TO_SALES,
           REVENUE_GROWTH_YOY_REPORTED):
    _CATEGORY_BY_METRIC[_m] = EvidenceCategory.MARKET
for _m in (CONSENSUS_REVENUE, CONSENSUS_EPS, CONSENSUS_RATING, PRICE_TARGET,
           ESTIMATE_REVISION_COUNT_UP, ESTIMATE_REVISION_COUNT_DOWN,
           FORWARD_REVENUE_ESTIMATE, FORWARD_EPS_ESTIMATE):
    _CATEGORY_BY_METRIC[_m] = EvidenceCategory.ESTIMATE
for _m in (GUIDANCE_REVENUE, GUIDANCE_GROSS_MARGIN, GUIDANCE_OPEX):
    _CATEGORY_BY_METRIC[_m] = EvidenceCategory.GUIDANCE


def canonicalise_metric(raw_metric: str) -> tuple[str, bool]:
    """Map a provider metric name to a canonical id.

    Returns ``(canonical_id, was_known)``. Unknown metrics are slugged with an
    ``unmapped.`` prefix rather than dropped, so the data still reaches the
    evidence store and shows up as an explicit gap instead of vanishing.
    """
    slug = _slug(raw_metric)
    if not slug:
        return "unmapped.unknown", False
    canonical = _ALIASES.get(slug)
    if canonical is not None:
        return canonical, True
    return f"unmapped.{slug.replace(' ', '_')}", False


def all_canonical_metrics() -> frozenset[str]:
    """Every canonical metric id the deterministic alias table knows about.

    Used by the LLM-assisted normalisation step to validate a model-proposed
    canonical id before accepting it in place of an "unmapped.*" fallback.
    """
    return frozenset(_ALIASES.values())


def category_for_metric(canonical_metric: str, default: EvidenceCategory) -> EvidenceCategory:
    """Category a canonical metric belongs to, falling back to the branch default."""
    return _CATEGORY_BY_METRIC.get(canonical_metric, default)


def is_text_metric(canonical_metric: str) -> bool:
    return canonical_metric in TEXT_METRICS


def is_percent_metric(canonical_metric: str) -> bool:
    return canonical_metric in PERCENT_METRICS


def is_multiple_metric(canonical_metric: str) -> bool:
    return canonical_metric in MULTIPLE_METRICS


def display_label(canonical_metric: str) -> str:
    """Human-readable label used in tables and narrative text."""
    return _DISPLAY_LABELS.get(canonical_metric, canonical_metric.replace("_", " ").title())


_DISPLAY_LABELS: dict[str, str] = {
    SHARE_PRICE: "Share price",
    MARKET_CAP: "Market cap",
    ENTERPRISE_VALUE: "Enterprise value",
    VOLUME: "Volume",
    AVG_VOLUME_30D: "Avg volume (30d)",
    SHARES_OUTSTANDING: "Diluted shares outstanding",
    PRICE_52W_HIGH: "52-week high",
    PRICE_52W_LOW: "52-week low",
    DAILY_HIGH: "Daily high",
    DAILY_LOW: "Daily low",
    FORWARD_PE: "Forward P/E",
    TRAILING_PE: "Trailing P/E",
    EV_TO_SALES: "EV/Sales",
    EV_TO_EBITDA: "EV/EBITDA",
    PRICE_TO_SALES: "P/S",
    CONSENSUS_REVENUE: "Consensus revenue",
    CONSENSUS_EPS: "Consensus EPS",
    CONSENSUS_RATING: "Consensus rating",
    REVENUE_GROWTH_YOY_REPORTED: "Reported revenue growth YoY",
    PRICE_TARGET: "Consensus price target",
    REVENUE: "Revenue",
    GROSS_PROFIT: "Gross profit",
    GROSS_MARGIN: "Gross margin",
    OPERATING_INCOME: "Operating income",
    OPERATING_MARGIN: "Operating margin",
    NET_INCOME: "Net income",
    NET_MARGIN: "Net margin",
    EPS_DILUTED: "Diluted EPS",
    OPERATING_CASH_FLOW: "Operating cash flow",
    CAPEX: "Capex",
    FREE_CASH_FLOW: "Free cash flow",
    CASH_AND_EQUIVALENTS: "Cash and equivalents",
    TOTAL_DEBT: "Total debt",
    RND_EXPENSE: "R&D expense",
    SEGMENT_REVENUE: "Segment revenue",
    GUIDANCE_REVENUE: "Revenue guidance",
    GUIDANCE_GROSS_MARGIN: "Gross margin guidance",
}


# ---------------------------------------------------------------------------
# Plan-facing metric requirement lists
# ---------------------------------------------------------------------------

#: What the Research Planner asks the market-data branch for.
MARKET_METRIC_PLAN: tuple[str, ...] = (
    SHARE_PRICE, PREVIOUS_CLOSE, MARKET_CAP, ENTERPRISE_VALUE, VOLUME, AVG_VOLUME_30D,
    SHARES_OUTSTANDING, PRICE_52W_HIGH, PRICE_52W_LOW, PRICE_HISTORY_POINT,
    FORWARD_PE, TRAILING_PE, EV_TO_SALES, CONSENSUS_RATING, PRICE_TARGET,
    FORWARD_REVENUE_ESTIMATE, FORWARD_EPS_ESTIMATE,
)

#: What the Research Planner asks the fundamentals branch for.
FUNDAMENTAL_METRIC_PLAN: tuple[str, ...] = (
    REVENUE, GROSS_PROFIT, GROSS_MARGIN, OPERATING_INCOME, OPERATING_MARGIN,
    NET_INCOME, EPS_DILUTED, OPERATING_CASH_FLOW, CAPEX, FREE_CASH_FLOW,
    CASH_AND_EQUIVALENTS, TOTAL_DEBT, RND_EXPENSE,
    SEGMENT_REVENUE, KPI,
    CONSENSUS_REVENUE, CONSENSUS_EPS,
    GUIDANCE_REVENUE, GUIDANCE_GROSS_MARGIN,
)
