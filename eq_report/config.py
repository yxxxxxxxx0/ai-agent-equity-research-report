"""Configuration, resolved from environment variables only.

Business logic never reads ``os.environ`` directly - it receives a
:class:`Settings` instance. Credentials are read here and nowhere else, and are
never written to logs or to the run manifest.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_ENV_PREFIX = "EQR_"


def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"{_ENV_PREFIX}{name}", default)


def _env_bool(name: str, default: bool) -> bool:
    raw = _env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = _env(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        return default


@dataclass(frozen=True, slots=True)
class ProviderCredentials:
    """API credentials. Empty values mean the provider is unavailable."""

    megadata_base_url: str | None = None
    megadata_api_key: str | None = None
    megadata_username: str | None = None
    megadata_password: str | None = None

    def has_megadata(self) -> bool:
        return bool(self.megadata_base_url)

    def has_megadata_basic_auth(self) -> bool:
        return bool(self.megadata_username and self.megadata_password)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """OpenRouter configuration for every LLM-backed stage of the pipeline.

    Every stage that can be LLM-backed (normalisation, analytics, segment
    agents, synthesis, QA) always uses the LLM path when a model is
    ``enabled`` (an API key is present); there is no separate per-stage
    opt-in any more. Each of those stages keeps its previous deterministic
    logic internally as a verification/fallback path: it is used to check the
    LLM's numeric output and to produce a result when the LLM is not
    configured or a call fails.
    """

    provider: str = "openrouter"
    model: str = "openai/gpt-5"
    # Segment agents select-and-phrase from evidence they are shown and are
    # validated post hoc (see agents/llm_agent.py); they do not need the same
    # reasoning depth as planning or synthesis. When set, this model is used
    # for the 8 segment-agent calls only, everything else keeps ``model``.
    agent_model: str | None = None
    # Segment names (SegmentName.value, comma-separated) that use their
    # deterministic, rule-based agent instead of the LLM writer, regardless of
    # whether a model is configured. These segments are fact-heavy and the
    # deterministic implementation already writes real prose from evidence
    # and analytics (see agents/financial_performance.py etc.) - routing them
    # away from the LLM cuts one full evidence-pool call per segment listed
    # here, at zero cost in what the section can say.
    deterministic_segments: frozenset[str] = frozenset()
    base_url: str = "https://openrouter.ai/api/v1"
    max_tokens: int = 16000
    temperature: float = 0.0
    # OpenRouter's reasoning-effort control: "high" | "medium" | "low", passed
    # as {"reasoning": {"effort": ...}} in the request body, not a different
    # model id. None means "let the model/provider default apply" - the
    # request omits the field entirely rather than sending a guessed default.
    reasoning_effort: str | None = None
    api_key: str | None = None
    app_url: str | None = None
    app_name: str = "EQ Report"
    # Some OpenRouter-routed models (large/heavy-reasoning ones especially)
    # can take several minutes per call, particularly for a batched prompt
    # covering many items (e.g. normalisation, analytics cross-check). This
    # is a per-HTTP-call timeout, not a total-run budget - each LLM-backed
    # stage still falls back to its deterministic logic if a call exceeds it.
    timeout_seconds: int = 900

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)


@dataclass(frozen=True, slots=True)
class Settings:
    """Everything the pipeline needs from the environment."""

    output_dir: Path
    database_path: Path
    log_level: str = "INFO"
    log_json: bool = False
    provider_timeout_seconds: int = 20
    # Best-effort live check (see pipeline.freshness_check) that the report's
    # dataset is anchored on the latest publicly reported fiscal period,
    # before analysis or synthesis run. A genuine extra cost per run (one
    # additional model call, using OpenRouter's web plugin), so it needs its
    # own opt-in.
    check_data_freshness: bool = False
    # Best-effort live web verification (see qa.engine._adjudicate_metric_conflicts)
    # of a metric the deterministic checks found two sources disagreeing on: the
    # model is asked to find the real value from an actual, dated, source-linked
    # page (OpenRouter's web plugin) and may only unblock QA by matching a
    # candidate to that verified source - it is never allowed to just pick
    # whichever value seems more plausible. A genuine extra cost per conflict,
    # so it needs its own opt-in.
    verify_metric_conflicts: bool = False
    # Repair statement-scoped QA failures inside the same run. The model may
    # rewrite prose, but deterministic QA remains the publication gate; an
    # unsafe/unrepairable statement is omitted rather than waved through.
    qa_auto_repair: bool = True
    qa_auto_repair_max_attempts: int = 2
    # Appends Bloomberg/MegadataAPI technical analysis to every full report.
    technical_appendix: bool = True
    # Produces a second, two-page investment brief from the same validated
    # report draft.  The normal PDF is always retained as the long report.
    compact_report: bool = True
    # Planning is the one stage that sets its own reasoning effort, since it is
    # a single call that shapes every downstream stage's scope (which sections,
    # which peers, which questions) - worth spending more inference on even
    # when other stages stay at the model's default. Every other stage keeps
    # using ``model`` unmodified; see ``pipeline.orchestrator`` for the copy
    # this produces.
    planning_reasoning_effort: str | None = None
    credentials: ProviderCredentials = field(default_factory=ProviderCredentials)
    model: ModelConfig = field(default_factory=ModelConfig)

    @classmethod
    def from_env(cls, *, output_dir: Path | str | None = None) -> Settings:
        root = Path(_env("OUTPUT_DIR", "output") or "output")
        if output_dir is not None:
            root = Path(output_dir)
        db_path = Path(_env("DB_PATH") or (root / "evidence.sqlite3"))
        return cls(
            output_dir=root,
            database_path=db_path,
            log_level=(_env("LOG_LEVEL", "INFO") or "INFO").upper(),
            log_json=_env_bool("LOG_JSON", False),
            provider_timeout_seconds=_env_int("PROVIDER_TIMEOUT_SECONDS", 20),
            check_data_freshness=_env_bool("CHECK_DATA_FRESHNESS", False),
            verify_metric_conflicts=_env_bool("VERIFY_METRIC_CONFLICTS", False),
            qa_auto_repair=_env_bool("QA_AUTO_REPAIR", True),
            qa_auto_repair_max_attempts=max(
                0, _env_int("QA_AUTO_REPAIR_MAX_ATTEMPTS", 2)),
            # Both deliverables are contractual pipeline outputs. Environment
            # flags are retained for backwards-compatible wrappers but may no
            # longer suppress either version.
            technical_appendix=True,
            compact_report=True,
            planning_reasoning_effort=_env("PLANNING_REASONING_EFFORT"),
            credentials=ProviderCredentials(
                megadata_base_url=_env("MEGADATA_BASE_URL"),
                megadata_api_key=_env("MEGADATA_API_KEY"),
                megadata_username=_env("MEGADATA_USERNAME"),
                megadata_password=_env("MEGADATA_PASSWORD"),
            ),
            model=ModelConfig(
                provider=_env("MODEL_PROVIDER", "openrouter") or "openrouter",
                model=_env("MODEL_NAME", "openai/gpt-5") or "openai/gpt-5",
                agent_model=_env("MODEL_NAME_AGENTS"),
                deterministic_segments=frozenset(
                    s.strip() for s in (
                        _env("DETERMINISTIC_SEGMENTS",
                             "company_snapshot,financial_performance,operating_drivers")
                        or ""
                    ).split(",")
                    if s.strip()
                ),
                base_url=(_env("MODEL_BASE_URL", "https://openrouter.ai/api/v1")
                          or "https://openrouter.ai/api/v1"),
                max_tokens=_env_int("MODEL_MAX_TOKENS", 16000),
                temperature=float(_env("MODEL_TEMPERATURE", "0.0") or 0.0),
                timeout_seconds=_env_int("MODEL_TIMEOUT_SECONDS", 900),
                api_key=_env("MODEL_API_KEY") or os.environ.get("OPENROUTER_API_KEY"),
                app_url=_env("MODEL_APP_URL"),
                app_name=_env("MODEL_APP_NAME", "EQ Report") or "EQ Report",
            ),
        )

    def run_dir(self, report_run_id: str) -> Path:
        return self.output_dir / "runs" / report_run_id

    def ensure_dirs(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict[str, object]:
        """Loggable view of the settings - never includes secrets."""
        return {
            "output_dir": str(self.output_dir),
            "database_path": str(self.database_path),
            "log_level": self.log_level,
            "provider_timeout_seconds": self.provider_timeout_seconds,
            "check_data_freshness": self.check_data_freshness,
            "qa_auto_repair": self.qa_auto_repair,
            "qa_auto_repair_max_attempts": self.qa_auto_repair_max_attempts,
            "verify_metric_conflicts": self.verify_metric_conflicts,
            "technical_appendix": self.technical_appendix,
            "compact_report": self.compact_report,
            "planning_reasoning_effort": self.planning_reasoning_effort,
            "credentials_present": {
                "megadata": self.credentials.has_megadata(),
            },
            "model": {
                "provider": self.model.provider,
                "model": self.model.model,
                "agent_model": self.model.agent_model or self.model.model,
                "deterministic_segments": sorted(self.model.deterministic_segments),
                "base_url": self.model.base_url,
                "enabled": self.model.enabled,
            },
        }
