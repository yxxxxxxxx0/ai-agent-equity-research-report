"""Generic OpenRouter JSON-mode chat client, shared across LLM-backed stages.

OpenRouter exposes an OpenAI-compatible Chat Completions endpoint. Every
caller uses JSON mode and treats the response the same way the Research
Planner already does: it is a proposal, never executed or trusted directly,
and the caller must validate it against its own typed contract before
accepting it.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from dataclasses import dataclass
from typing import Any

import requests

from ..config import ModelConfig
from ..errors import ConfigurationError
from .usage import UsageEvent, UsageTracker

logger = logging.getLogger("eq_report.llm.client")


@dataclass(frozen=True, slots=True)
class LLMJSONResponse:
    payload: dict[str, Any]
    input_tokens: int
    output_tokens: int
    cost_usd: float | None = None


class OpenRouterJSONClient:
    """Thin OpenAI-compatible chat-completions client, JSON mode only."""

    def __init__(
        self, config: ModelConfig, *, timeout_seconds: int | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        if not config.api_key:
            raise ConfigurationError(
                "EQR_MODEL_API_KEY or OPENROUTER_API_KEY is required for LLM-backed stages")
        self.config = config
        self.timeout_seconds = timeout_seconds if timeout_seconds is not None else config.timeout_seconds
        self._tracker = tracker

    async def complete_json(
        self, system_prompt: str, user_prompt: str, *, web_search: bool = False,
        stage: str = "", cache_key: str | None = None,
    ) -> LLMJSONResponse:
        return await asyncio.to_thread(
            self._complete_json_sync, system_prompt, user_prompt, web_search, stage, cache_key)

    def _complete_json_sync(
        self, system_prompt: str, user_prompt: str, web_search: bool = False,
        stage: str = "", cache_key: str | None = None,
    ) -> LLMJSONResponse:
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
            # Asks OpenRouter to report the actual USD cost of the call on
            # the response's usage object (usage.cost). This is a
            # response-shape hint, not a generation-mode setting, so it is
            # kept in both branches below - unlike response_format vs. the
            # web plugin (which really do conflict at generation time),
            # there is no documented interaction between usage.include and
            # the web plugin, and no basis here to assume OpenRouter rejects
            # the combination.
            "usage": {"include": True},
        }
        if cache_key:
            # OpenAI-compatible prompt-cache routing hint: calls sharing a key
            # are preferentially routed to the same cache-warm backend, so a
            # shared prompt prefix (see agents/llm_agent.py's cache_key use)
            # actually gets a cache hit instead of landing on a cold worker.
            # Providers/models that don't support it ignore the field.
            request_body["prompt_cache_key"] = cache_key
        if web_search:
            # OpenRouter's web plugin: augments the prompt with live search
            # results before the model answers, regardless of which underlying
            # model is configured. Used when a real data provider is not
            # available, so the model can look up current figures instead of
            # answering from (possibly stale) training data. Several upstream
            # providers reject "response_format: json_object" combined with
            # the web plugin ("Web Search cannot be used with JSON mode"), so
            # JSON mode is only requested when web search is off; the system
            # prompt is relied on instead to keep the reply to plain JSON.
            request_body["plugins"] = [{"id": "web"}]
        else:
            request_body["response_format"] = {"type": "json_object"}

        logger.info(
            "LLM call starting: stage=%s model=%s max_tokens=%s timeout=%ss",
            stage or "-", self.config.model, self.config.max_tokens, self.timeout_seconds,
        )
        started = time.monotonic()
        try:
            response = _post_with_hard_deadline(
                f"{base_url}/chat/completions", headers, request_body, self.timeout_seconds)
        except Exception as exc:
            logger.warning(
                "LLM call failed after %.1fs: stage=%s model=%s (%s: %s)",
                time.monotonic() - started, stage or "-", self.config.model,
                type(exc).__name__, exc,
            )
            raise
        elapsed = time.monotonic() - started
        logger.info(
            "LLM call finished in %.1fs: stage=%s model=%s status=%s",
            elapsed, stage or "-", self.config.model, response.status_code,
        )
        response.raise_for_status()
        response_body = response.json()
        try:
            content = response_body["choices"][0]["message"]["content"]
            if isinstance(content, list):
                text = "".join(
                    str(part.get("text", "")) for part in content
                    if isinstance(part, dict)
                )
            else:
                text = str(content)
            payload = _parse_json_object(text)
        except (KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise ValueError("LLM did not return valid JSON") from exc
        if not isinstance(payload, dict):
            raise TypeError("LLM JSON must be an object")

        usage = response_body.get("usage") or {}
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
                    stage=stage, model=self.config.model,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    cost_usd=cost_usd,
                ))
            except Exception as exc:  # noqa: BLE001 - tracking must never break a call
                logger.warning("LLM usage tracking failed: %s: %s", type(exc).__name__, exc)

        return LLMJSONResponse(
            payload=payload,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )


def _post_with_hard_deadline(
    url: str, headers: dict[str, str], json_body: dict[str, Any], timeout_seconds: int,
) -> requests.Response:
    """POST with a real total-call deadline, not just ``requests``' inter-read timeout.

    ``requests``' own ``timeout=`` only bounds the gap between successive
    socket reads, not the call's total duration - a server (or an
    intermediary) that trickles bytes slowly enough can keep resetting that
    gap timer and hang far past the configured value. Running the call in its
    own single-use thread and enforcing ``future.result(timeout=...)`` gives a
    hard wall-clock ceiling: if it fires, this raises and the caller's
    existing fallback logic takes over. The underlying thread is not
    forcibly killed (Python cannot do that safely) - it is abandoned to
    finish or die on its own; ``shutdown(wait=False)`` ensures this function
    does not itself block waiting for it.
    """
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(
        requests.post, url, headers=headers, json=json_body, timeout=timeout_seconds)
    try:
        return future.result(timeout=timeout_seconds)
    except concurrent.futures.TimeoutError as exc:
        raise TimeoutError(
            f"LLM call exceeded hard deadline of {timeout_seconds}s") from exc
    finally:
        executor.shutdown(wait=False)


def _parse_json_object(text: str) -> Any:
    """Parse a JSON object out of a model reply.

    JSON-mode replies are already exactly one JSON object. Without JSON mode
    (required when ``web_search=True``, see above) a model will often still
    wrap it in a ```json fenced code block or add a sentence either side, so
    this falls back to extracting the outermost ``{...}`` span before giving
    up.
    """
    stripped = text.strip()
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        pass
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        try:
            return json.loads(stripped.strip())
        except json.JSONDecodeError:
            pass
    start, end = stripped.find("{"), stripped.rfind("}")
    if start != -1 and end != -1 and end > start:
        return json.loads(stripped[start:end + 1])
    raise json.JSONDecodeError("no JSON object found", stripped, 0)
