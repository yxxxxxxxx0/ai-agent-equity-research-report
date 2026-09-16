"""Small deterministic financial validations used before publication."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    code: str
    severity: str
    message: str
    evidence_ids: tuple[str, ...] = ()


def guidance_midpoint(low: float, high: float) -> float:
    if low > high:
        raise ValueError("guidance low exceeds guidance high")
    return (low + high) / 2.0


def validate_ratio(
    name: str, numerator: float, denominator: float, reported_pct: float,
    *, tolerance_pp: float = 0.1,
) -> ValidationIssue | None:
    if denominator == 0:
        return ValidationIssue(name, "P0", "denominator is zero")
    calculated = numerator / denominator * 100.0
    if abs(calculated - reported_pct) <= tolerance_pp:
        return None
    return ValidationIssue(
        name, "P0",
        f"reported {reported_pct:.3f}% does not reconcile to calculated {calculated:.3f}%",
    )


def detect_outlier(metric: str, value: float) -> ValidationIssue | None:
    limits = {"gross_margin": (-20.0, 90.0), "revenue_growth": (-100.0, 100.0),
              "fcf_margin": (-200.0, 100.0)}
    if metric not in limits:
        return None
    low, high = limits[metric]
    if low <= value <= high:
        return None
    return ValidationIssue("outlier", "P1", f"VERIFY_PRIMARY_SOURCE: unusual {metric}={value}")
