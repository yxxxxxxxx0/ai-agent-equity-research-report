"""Best-effort live verification that the report is anchored on the latest
publicly reported fiscal period.

Runs once, early - right after the Evidence Store is populated, before
analysis or synthesis - only when a model is configured *and*
``EQR_CHECK_DATA_FRESHNESS`` is enabled (see ``config.Settings.check_data_freshness``;
a genuine per-run cost, so it needs its own opt-in, separate from
``pipeline.gap_research``'s). It asks the model, with OpenRouter's web-search
plugin, what the most recent fiscal period this company has actually reported
results for, as of the report date - and compares that against what the
configured data providers gave the pipeline
(``EvidenceReader.latest_reported_period()``).

A mismatch does not attempt to replace the dataset. Reconciling every metric
onto a newer period would mean re-running acquisition, normalisation and
analytics against different underlying numbers - a much larger change than
one verification call, and not meaningful against illustrative sample data in
particular, since it does not correspond to any real fiscal period to begin
with. What this *does* do is surface the mismatch prominently and
structurally rather than let it pass silently or only turn up later, mixed in
with general data-gap research: a mismatch is recorded on the run and
reflected in the ReportDraft's metadata, so the renderer can print a banner
at the top of the report and QA can flag it - the reader is told which period
the analysis is anchored on, and that a newer public report exists and has
not been incorporated, before reading a word of the analysis itself.
"""

from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig
from ..llm.client import OpenRouterJSONClient
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from ..normalisation.dates import normalise_fiscal_period

logger = get_logger("pipeline.freshness_check")

_SYSTEM_PROMPT = """You are verifying a data-freshness question with live web search. Given a
company and an as-of date, search the web and identify the most recent fiscal period the
company has PUBLICLY reported financial results for, as of that date - not a period it is
merely expected or scheduled to report. Reply only with what you can attribute to an actual
page you found; never invent a period label, a filing date, a source name, or a URL. If you
cannot find a specific, dated source confirming the latest reported period, set found=false
rather than guessing. Return JSON only, matching the requested schema."""


@dataclass(frozen=True, slots=True)
class FreshnessResult:
    """The outcome of one freshness check. ``checked=False`` means the check
    did not run or could not produce a usable, source-linked answer - not
    that the dataset was confirmed current."""

    checked: bool
    verified_period: str | None = None
    verified_as_of: dt.date | None = None
    source_name: str | None = None
    source_url: str | None = None
    mismatched: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "checked": self.checked,
            "verified_period": self.verified_period,
            "verified_as_of": self.verified_as_of.isoformat() if self.verified_as_of else None,
            "source_name": self.source_name,
            "source_url": self.source_url,
            "mismatched": self.mismatched,
        }


async def check_freshness(
    company: str, ticker: str | None, report_date: dt.date,
    dataset_period: str | None, model_config: ModelConfig | None,
    *, tracker: UsageTracker | None = None,
) -> FreshnessResult:
    """Best-effort: never raises. ``checked=False`` when the check could not
    run or the model could not produce a source-linked answer."""
    if model_config is None or not model_config.enabled or not dataset_period:
        return FreshnessResult(checked=False)

    client = OpenRouterJSONClient(model_config, tracker=tracker)
    schema = {
        "found": "true or false",
        "period_label": (
            "the most recent reported fiscal period, only if found is true, in canonical "
            "form 'FYnnnn Qn' for a quarter or 'FYnnnn' for a full year (e.g. 'FY2027 Q2') "
            "- never a prose description like 'second quarter fiscal 2027'"
        ),
        "as_of_date": "the date results were reported/filed, YYYY-MM-DD, only if found is true",
        "source_name": "the publisher or site name, only if found is true",
        "source_url": "the exact URL of the page you found this on, only if found is true",
    }
    prompt = (
        f"Company: {company} ({ticker or 'ticker unknown'})\n"
        f"As-of date: {report_date.isoformat()}\n"
        f"This report's dataset is currently anchored on: {dataset_period}\n"
        f"Return this JSON shape: {schema}"
    )
    try:
        response = await client.complete_json(
            _SYSTEM_PROMPT, prompt, web_search=True, stage="freshness_check")
    except Exception as exc:  # noqa: BLE001 - best-effort, must not sink the run
        log_event(logger, logging.WARNING, "freshness check failed",
                  error=f"{type(exc).__name__}: {exc}")
        return FreshnessResult(checked=False)

    payload = response.payload if isinstance(response.payload, dict) else {}
    if not payload.get("found"):
        return FreshnessResult(checked=False)

    url = str(payload.get("source_url", "")).strip()
    label = str(payload.get("period_label", "")).strip()
    # Same rule as pipeline.gap_research: no real URL means the answer is
    # indistinguishable from an invented one, so it is rejected rather than
    # trusted.
    if not label or not (url.startswith("http://") or url.startswith("https://")):
        return FreshnessResult(checked=False)

    as_of = None
    raw_date = payload.get("as_of_date")
    if raw_date:
        try:
            as_of = dt.date.fromisoformat(str(raw_date)[:10])
        except ValueError:
            as_of = None

    mismatched = _is_confidently_newer(label, dataset_period)
    log_event(logger, logging.INFO, "freshness check complete",
              dataset_period=dataset_period, verified_period=label, mismatched=mismatched)
    return FreshnessResult(
        checked=True, verified_period=label, verified_as_of=as_of,
        source_name=str(payload.get("source_name", "")).strip() or "Web search result",
        source_url=url, mismatched=mismatched,
    )


def _is_confidently_newer(verified_label: str, dataset_label: str) -> bool:
    """True only when both periods parse and the verified one is strictly
    later. Deliberately conservative: two labels that fail to parse into a
    comparable (year, quarter) - e.g. one is an unusual format - return False
    rather than falling back to a plain string inequality, which would flag a
    mismatch for two labels that actually mean the same period."""
    verified = normalise_fiscal_period(verified_label)
    dataset = normalise_fiscal_period(dataset_label)
    if (
        verified is None or dataset is None
        or verified.fiscal_year is None or dataset.fiscal_year is None
    ):
        return False
    verified_key = (verified.fiscal_year, verified.fiscal_quarter or 0)
    dataset_key = (dataset.fiscal_year, dataset.fiscal_quarter or 0)
    return verified_key > dataset_key
