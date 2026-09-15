"""Shared "LLM primary, deterministic verification" plumbing.

Every stage converted to be LLM-backed follows the same shape: an LLM call
proposes a result, and the module's own pre-existing deterministic logic
(kept as an internal function, never deleted) recomputes the numeric parts of
that same result from the same inputs. When the two agree within tolerance,
the LLM's output is used as-is; when they disagree, the deterministic value
wins and a flag is recorded so the divergence is visible in the run's QA
trail rather than silently overwritten.

This module holds the handful of helpers that pattern needs in every stage:
a tolerance-based numeric comparator, and a small wrapper that turns "call the
LLM" into "call the LLM, or record why it could not be reached" without ever
raising out of the caller.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig
from .client import LLMJSONResponse, OpenRouterJSONClient
from .usage import UsageTracker

logger = logging.getLogger("eq_report.llm.verify")


@dataclass(frozen=True, slots=True)
class VerifiedNumber:
    """The outcome of checking one LLM-proposed number against code.

    ``value`` is always the number callers should actually use: the LLM's
    number when it agreed with the deterministic recomputation, otherwise the
    deterministic number. ``overridden`` and ``flag`` describe what happened
    so it can be surfaced to QA / the run manifest.
    """

    value: float
    llm_value: float | None
    deterministic_value: float
    overridden: bool
    flag: str | None = None


def verify_number(
    llm_value: Any,
    deterministic_value: float,
    *,
    label: str,
    rel_tolerance: float = 0.01,
    abs_tolerance: float = 0.01,
) -> VerifiedNumber:
    """Compare an LLM-proposed number with the code-computed ground truth.

    Numbers are always safety-critical, so the deterministic value is what
    gets used whenever the two disagree by more than the tolerance (or the
    LLM did not return a usable number at all).
    """
    try:
        llm_number = float(llm_value) if llm_value is not None else None
    except (TypeError, ValueError):
        llm_number = None

    if llm_number is None:
        return VerifiedNumber(
            value=deterministic_value, llm_value=None,
            deterministic_value=deterministic_value, overridden=True,
            flag=f"{label}: LLM did not return a usable number; used the computed value.",
        )

    tolerance = max(abs_tolerance, abs(deterministic_value) * rel_tolerance)
    if abs(llm_number - deterministic_value) <= tolerance:
        return VerifiedNumber(
            value=llm_number, llm_value=llm_number,
            deterministic_value=deterministic_value, overridden=False,
        )
    return VerifiedNumber(
        value=deterministic_value, llm_value=llm_number,
        deterministic_value=deterministic_value, overridden=True,
        flag=(
            f"{label}: LLM value {llm_number!r} diverged from the computed value "
            f"{deterministic_value!r} beyond tolerance; the computed value was used."
        ),
    )


async def safe_complete_json(
    model_config: ModelConfig | None,
    system_prompt: str,
    user_prompt: str,
    *,
    client: Any | None = None,
    tracker: UsageTracker | None = None,
    stage: str = "",
) -> tuple[LLMJSONResponse | None, str | None]:
    """Call the LLM JSON client, never raising.

    Returns ``(response, None)`` on success or ``(None, reason)`` when the
    model is not configured or the call failed/returned unparseable output -
    callers fall back to their deterministic result in that case. ``tracker``
    and ``stage`` are only used when this helper builds its own client (an
    injected ``client`` is assumed to already carry whatever tracking it
    needs).
    """
    if client is None:
        if model_config is None or not model_config.enabled:
            return None, "no LLM model configured"
        if model_config.provider.lower() != "openrouter":
            return None, f"unsupported model provider {model_config.provider!r}"
        client = OpenRouterJSONClient(model_config, tracker=tracker)
        try:
            response = await client.complete_json(system_prompt, user_prompt, stage=stage)
        except Exception as exc:  # noqa: BLE001 - degrade to the deterministic path
            logger.warning("LLM call failed; falling back to deterministic result: %s: %s",
                            type(exc).__name__, exc)
            return None, f"{type(exc).__name__}: {exc}"
        return response, None
    try:
        response = await client.complete_json(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 - degrade to the deterministic path
        logger.warning("LLM call failed; falling back to deterministic result: %s: %s",
                        type(exc).__name__, exc)
        return None, f"{type(exc).__name__}: {exc}"
    return response, None
