"""OpenRouter client for the GPT-backed research planner.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint. The planner
uses JSON mode and still validates the returned object against its local typed
contract; model output is never executed directly.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

from ..config import ModelConfig
from ..errors import ConfigurationError
from ..llm.client import _post_with_hard_deadline
from ..llm.usage import UsageEvent, UsageTracker

logger = logging.getLogger("eq_report.planning.openrouter_client")


@dataclass(frozen=True, slots=True)
class PlannerModelResponse:
    payload: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None


class OpenRouterPlannerClient:
    def __init__(
        self, config: ModelConfig, *, timeout_seconds: int | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        if not config.api_key:
            raise ConfigurationError(
                "EQR_MODEL_API_KEY or OPENROUTER_API_KEY is required for GPT planning")
        self.config = config
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else config.timeout_seconds
        self._tracker = tracker

    async def create_plan(
        self, system_prompt: str, user_prompt: str
    ) -> PlannerModelResponse:
        return await asyncio.to_thread(
            self._create_plan_sync, system_prompt, user_prompt)

    def _create_plan_sync(
        self, system_prompt: str, user_prompt: str
    ) -> PlannerModelResponse:
        base_url = self.config.base_url.rstrip("/")
        headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
        }
        if self.config.app_url:
            headers["HTTP-Referer"] = self.config.app_url
        if self.config.app_name:
            headers["X-Title"] = self.config.app_name

        request_body: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": self.config.max_tokens,
            "temperature": self.config.temperature,
            "response_format": {"type": "json_object"},
            # See OpenRouterJSONClient (eq_report.llm.client) for why this
            # is always requested: it asks OpenRouter to report the real
            # USD cost of the call on usage.cost.
            "usage": {"include": True},
        }
        if self.config.reasoning_effort:
            # OpenRouter's unified reasoning control: same model, told to
            # spend more (or less) inference on this call - not a different
            # model id. Omitted entirely when unset, so an unsupported model
            # is never sent a reasoning block it would have to reject.
            request_body["reasoning"] = {"effort": self.config.reasoning_effort}

        logger.info(
            "LLM call starting: stage=planning model=%s max_tokens=%s timeout=%ss"
            "%s",
            self.config.model, self.config.max_tokens, self.timeout_seconds,
            f" reasoning_effort={self.config.reasoning_effort}"
            if self.config.reasoning_effort else "",
        )
        started = time.monotonic()
        try:
            response = _post_with_hard_deadline(
                f"{base_url}/chat/completions", headers, request_body, self.timeout_seconds,
            )
        except Exception as exc:
            logger.warning(
                "LLM call failed after %.1fs: stage=planning model=%s (%s: %s)",
                time.monotonic() - started, self.config.model, type(exc).__name__, exc,
            )
            raise
        elapsed = time.monotonic() - started
        logger.info(
            "LLM call finished in %.1fs: stage=planning model=%s status=%s",
            elapsed, self.config.model, response.status_code,
        )
        response.raise_for_status()
        body = response.json()
        try:
            content = body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                text = "".join(
                    str(part.get("text", "")) for part in content
                    if isinstance(part, dict)
                )
            else:
                text = str(content)
            payload = json.loads(text.strip())
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("OpenRouter planner did not return valid JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("OpenRouter planner JSON must be an object")

        usage = body.get("usage") or {}
        input_tokens = int(usage.get("prompt_tokens", 0) or 0)
        output_tokens = int(usage.get("completion_tokens", 0) or 0)
        raw_cost = usage.get("cost")
        try:
            cost_usd = float(raw_cost) if raw_cost is not None else None
        except (TypeError, ValueError):
            cost_usd = None

        if self._tracker is not None:
            try:
                self._tracker.record(UsageEvent(
                    stage="planning", model=self.config.model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=cost_usd,
                ))
            except Exception as exc:  # noqa: BLE001 - tracking must never break a call
                logger.warning("LLM usage tracking failed: %s: %s", type(exc).__name__, exc)

        return PlannerModelResponse(
            payload=payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
