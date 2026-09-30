"""LLM fallback for numeric strings the deterministic parsers reject.

The parsers stay authoritative for every format they know. Only a value they
*rejected* (e.g. ``"(215.4)M"``, ``"2.9 trillion dollars"``) may be rewritten
here, and the rewrite is trusted only if code can verify it:

* the quoted ``span`` occurs verbatim in the raw text,
* the number equals the span's own digits times a known scale (no invention),
* the sign matches the span, and
* two independent runs agree (an omission or drift in either drops the row).

An accepted value is a clean numeric string that goes back through the normal
deterministic path, so units, currency and percent handling are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any

from ..config import ModelConfig
from ..llm.usage import UsageTracker
from ..llm.verify import safe_complete_json

_SCALES = (1.0, 1e3, 1e6, 1e9, 1e12)
_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?|\.\d+")

_SYSTEM = """You clean up numeric strings a parser could not read. For each item return the
exact contiguous text you read the number from (span) and the plain number in base units
(dollars not millions; keep a percentage as written, e.g. 12.5 for 12.5%; negatives for
parentheses or a minus sign). Use only what the text says; never compute or guess. If the
text holds no single clear number, omit the item.
Return JSON only: {"items":[{"id":<int>,"span":"...","number":<number>}]}"""


def _verified(text: str, row: dict[str, Any]) -> str | None:
    span = str(row.get("span", ""))
    match = _NUMBER.search(span) if span and span in text else None
    try:
        value = float(row.get("number"))
    except (TypeError, ValueError):
        return None
    if match is None or not math.isfinite(value):
        return None
    mantissa = float(match.group().replace(",", ""))
    if (value < 0) != ("-" in span or "(" in span):
        return None
    if not any(math.isclose(abs(value), mantissa * s, rel_tol=1e-9) for s in _SCALES):
        return None
    return f"{value:.6f}".rstrip("0").rstrip(".") + ("%" if "%" in span else "")


async def rescue_values(
    texts: dict[int, str], model_config: ModelConfig | None, *,
    tracker: UsageTracker | None = None, client: Any | None = None,
) -> dict[int, str]:
    """Return ``{id: clean numeric string}`` for the rows two runs agree on."""
    if not texts or (client is None and (model_config is None or not model_config.enabled)):
        return {}
    prompt = json.dumps([{"id": i, "text": t[:200]} for i, t in texts.items()])
    runs = await asyncio.gather(*(
        safe_complete_json(model_config, _SYSTEM, prompt, client=client, tracker=tracker,
                           stage="normalisation:rescue")
        for _ in range(2)))
    outcomes: list[dict[int, str]] = []
    for response, _error in runs:
        got: dict[int, str] = {}
        payload = response.payload if response else None
        for row in payload.get("items", []) if isinstance(payload, dict) else []:
            try:
                i = int(row["id"])
            except (KeyError, TypeError, ValueError):
                continue
            if i in texts and (clean := _verified(texts[i], row)) is not None:
                got[i] = clean
        outcomes.append(got)
    return {i: v for i, v in outcomes[0].items() if outcomes[1].get(i) == v}
