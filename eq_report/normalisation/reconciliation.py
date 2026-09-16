"""Deterministic canonical-evidence selection and cross-source reconciliation."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

from ..domain.enums import EvidenceCategory, EvidenceStatus, FactType, SourceType
from ..domain.evidence import EvidenceItem


@dataclass(frozen=True, slots=True)
class MetricTolerance:
    absolute: float = 0.0
    relative: float = 0.005


TOLERANCES: dict[str, MetricTolerance] = {
    "eps": MetricTolerance(absolute=0.01, relative=0.001),
    "diluted_eps": MetricTolerance(absolute=0.01, relative=0.001),
    "revenue": MetricTolerance(relative=0.005),
    "gross_margin": MetricTolerance(absolute=0.1, relative=0.0),
    "operating_margin": MetricTolerance(absolute=0.1, relative=0.0),
}

SOURCE_PRIORITY = {
    SourceType.COMPANY_FILING: 100,
    SourceType.EARNINGS_RELEASE: 95,
    SourceType.INVESTOR_PRESENTATION: 90,
    SourceType.EARNINGS_CALL: 85,
    SourceType.MARKET_DATA: 80,
    SourceType.SELL_SIDE_CONSENSUS: 75,
    SourceType.INDUSTRY_RESEARCH: 70,
    SourceType.NEWS: 50,
    SourceType.DERIVED: 100,
}


def infer_basis(item: EvidenceItem) -> str | None:
    explicit = item.metadata.get("basis")
    if explicit:
        value = str(explicit).lower().replace("-", "_")
        return "non_GAAP" if value in {"nongaap", "non_gaap", "adjusted"} else value.upper()
    text = " ".join(str(x) for x in (item.raw_metric, item.document_title, item.claim_text) if x).lower()
    if "non-gaap" in text or "non gaap" in text or "adjusted eps" in text:
        return "non_GAAP"
    if "gaap" in text:
        return "GAAP"
    if item.category == EvidenceCategory.ESTIMATE:
        return "consensus"
    if item.category == EvidenceCategory.GUIDANCE:
        return "management_guidance"
    return None


def infer_fact_type(item: EvidenceItem) -> FactType:
    if item.source_type == SourceType.DERIVED:
        return FactType.DERIVED_FACT
    if item.category == EvidenceCategory.GUIDANCE or item.source_type == SourceType.EARNINGS_CALL:
        return FactType.MANAGEMENT_STATEMENT
    if item.category == EvidenceCategory.ESTIMATE:
        return FactType.EXTERNAL_FORECAST
    if item.source_type in {SourceType.NEWS, SourceType.INDUSTRY_RESEARCH} and item.category == EvidenceCategory.DOCUMENT:
        return FactType.ANALYST_OPINION
    return FactType.REPORTED_FACT


def _key(item: EvidenceItem, basis: str | None) -> str:
    period = item.period_label or (item.as_of.isoformat() if item.as_of else "undated")
    raw = "|".join(str(x or "unresolved") for x in (
        item.company.lower(), item.ticker, item.metric, period, basis,
        item.unit, item.currency, item.frequency,
    ))
    return "rg_" + hashlib.sha1(raw.encode()).hexdigest()[:16]


def _agrees(a: float, b: float, metric: str) -> bool:
    tolerance = TOLERANCES.get(metric, MetricTolerance())
    difference = abs(a - b)
    scale = max(abs(a), abs(b), 1e-12)
    return difference <= max(tolerance.absolute, tolerance.relative * scale)


def _score(item: EvidenceItem) -> tuple[int, str, str]:
    return (
        SOURCE_PRIORITY.get(item.source_type, 40),
        (item.published_at or item.as_of).isoformat() if (item.published_at or item.as_of) else "",
        item.evidence_id,
    )


#: Metrics where a provider bug or unit error produces a value no real
#: instrument could have - e.g. a listed company with 25,000 shares
#: outstanding. These floors are set far below any real listed company's
#: figures, so they only ever catch data errors, never a genuine small-cap.
_IMPLAUSIBLE_FLOOR: dict[str, float] = {
    "shares_outstanding": 100_000,
    "market_cap": 1_000_000,
    "volume": 0,
    "avg_volume_30d": 0,
}


def _implausible(item: EvidenceItem) -> bool:
    floor = _IMPLAUSIBLE_FLOOR.get(item.metric or "")
    return floor is not None and item.value is not None and item.value < floor


def reconcile(items: Iterable[EvidenceItem]) -> tuple[EvidenceItem, ...]:
    """Keep all observations, but mark exactly which numeric facts are canonical."""
    prepared: list[EvidenceItem] = []
    groups: dict[str, list[EvidenceItem]] = defaultdict(list)
    for item in items:
        basis = infer_basis(item)
        fact_type = infer_fact_type(item)
        if item.value is None:
            # Qualitative passages remain usable as context, but never as validated
            # numeric evidence. This closes the raw-document numeric bypass.
            prepared.append(replace(item, basis=basis, fact_type=fact_type,
                                    status=EvidenceStatus.UNVERIFIED, is_canonical=False))
            continue
        key = _key(item, basis)
        candidate = replace(item, basis=basis, fact_type=fact_type,
                            reconciliation_key=key)
        groups[key].append(candidate)

    for key, group in groups.items():
        ranked = sorted(group, key=_score, reverse=True)
        winner = ranked[0]
        conflicting = [x for x in ranked[1:] if not _agrees(winner.value or 0, x.value or 0, winner.metric or "")]
        ids = tuple(x.evidence_id for x in ranked[1:])
        # EPS without an identified basis is not safe to publish as canonical.
        unresolved_basis = (winner.metric in {"eps", "diluted_eps", "reported_eps"} and not winner.basis)
        missing_horizon = winner.metric == "forward_pe" and not winner.metadata.get("valuation_horizon")
        implausible = _implausible(winner)
        if conflicting or unresolved_basis or missing_horizon or implausible:
            reason = "unresolved basis" if unresolved_basis else "values exceed configured tolerance"
            if missing_horizon:
                reason = "forward valuation horizon unavailable"
            if implausible:
                reason = f"value {winner.value!r} is below the plausible floor for {winner.metric}"
            blocked_status = EvidenceStatus.UNVERIFIED if (missing_horizon or implausible) else EvidenceStatus.CONFLICTED
            for item in ranked:
                prepared.append(replace(item, status=blocked_status,
                                        is_canonical=False,
                                        alternate_evidence_ids=tuple(x.evidence_id for x in ranked if x != item),
                                        validation_messages=(reason,)))
        else:
            prepared.append(replace(winner, status=EvidenceStatus.VALIDATED,
                                    is_canonical=True, alternate_evidence_ids=ids))
            for item in ranked[1:]:
                prepared.append(replace(item, status=EvidenceStatus.VALIDATED,
                                        is_canonical=False,
                                        alternate_evidence_ids=(winner.evidence_id,),
                                        validation_messages=("alternate agreeing observation",)))
    return tuple(prepared)
