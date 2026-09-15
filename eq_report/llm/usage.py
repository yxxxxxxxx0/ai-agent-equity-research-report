"""Real USD cost and token tracking for LLM-backed pipeline stages.

OpenRouter can report the actual USD cost of a completion directly in the
response when the request body includes ``"usage": {"include": true}``
(``eq_report.llm.client.OpenRouterJSONClient`` and
``eq_report.planning.openrouter_client.OpenRouterPlannerClient`` both set
this). :class:`UsageTracker` accumulates one :class:`UsageEvent` per
successful LLM call across a whole pipeline run, so the run manifest and the
CLI can report the real total cost rather than an estimate from a hardcoded
price table.

Calls happen inside ``asyncio.to_thread`` - i.e. real OS threads running
concurrently (segment agents in particular) - so the tracker guards its state
with a plain :class:`threading.Lock` rather than relying on the asyncio event
loop for exclusion.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """One completed LLM call.

    ``cost_usd`` is ``None`` when OpenRouter did not report a cost for the
    call (e.g. an upstream provider that does not support cost reporting) -
    that is never fabricated from a price table, only recorded as missing.
    """

    stage: str
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float | None


class UsageTracker:
    """Thread-safe accumulator of :class:`UsageEvent` for one pipeline run."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: list[UsageEvent] = []

    def record(self, event: UsageEvent) -> None:
        with self._lock:
            self._events.append(event)

    @property
    def events(self) -> tuple[UsageEvent, ...]:
        """A read-only snapshot of every event recorded so far."""
        with self._lock:
            return tuple(self._events)

    @property
    def total_input_tokens(self) -> int:
        return sum(e.input_tokens for e in self.events)

    @property
    def total_output_tokens(self) -> int:
        return sum(e.output_tokens for e in self.events)

    @property
    def total_cost_usd(self) -> float:
        return sum(e.cost_usd for e in self.events if e.cost_usd is not None)

    @property
    def calls_missing_cost(self) -> int:
        return sum(1 for e in self.events if e.cost_usd is None)

    def summary(self) -> dict[str, Any]:
        """A JSON-serialisable summary, embeddable in the run manifest."""
        events = self.events
        by_stage: dict[str, dict[str, Any]] = {}
        for event in events:
            row = by_stage.setdefault(event.stage, {
                "calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cost_usd": 0.0,
                "calls_missing_cost": 0,
            })
            row["calls"] += 1
            row["input_tokens"] += event.input_tokens
            row["output_tokens"] += event.output_tokens
            if event.cost_usd is not None:
                row["cost_usd"] += event.cost_usd
            else:
                row["calls_missing_cost"] += 1

        return {
            "call_count": len(events),
            "total_input_tokens": sum(e.input_tokens for e in events),
            "total_output_tokens": sum(e.output_tokens for e in events),
            "total_cost_usd": sum(e.cost_usd for e in events if e.cost_usd is not None),
            "calls_missing_cost": sum(1 for e in events if e.cost_usd is None),
            "by_stage": by_stage,
        }
