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

    market_data_api_key: str | None = None
    market_data_base_url: str | None = None
    fundamentals_api_key: str | None = None
    fundamentals_base_url: str | None = None
    documents_api_key: str | None = None
    documents_base_url: str | None = None
    sec_user_agent: str | None = None
    megadata_base_url: str | None = None
    megadata_api_key: str | None = None
    arcticdb_uri: str | None = None

    def has_market_data(self) -> bool:
        return bool(self.market_data_api_key)

    def has_fundamentals(self) -> bool:
        return bool(self.fundamentals_api_key)

    def has_documents(self) -> bool:
        return bool(self.documents_api_key or self.sec_user_agent)

    def has_megadata(self) -> bool:
        return bool(self.megadata_base_url)

    def has_arcticdb(self) -> bool:
        return bool(self.arcticdb_uri)


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
    allow_mock_providers: bool = True
    provider_timeout_seconds: int = 20
    # Best-effort live web search (see pipeline.gap_research) to fill the
    # report's own disclosed data gaps. Off by default: it is a genuine extra
    # cost per run (one additional model call per gap, using OpenRouter's web
    # plugin) on top of whatever the model is already configured for, so it
    # needs its own opt-in rather than riding along with EQR_MODEL_API_KEY.
    research_data_gaps: bool = False
    research_data_gaps_max: int = 8
    # Best-effort live check (see pipeline.freshness_check) that the report's
    # dataset is anchored on the latest publicly reported fiscal period,
    # before analysis or synthesis run. A genuine extra cost per run (one
    # additional model call, using OpenRouter's web plugin), so - like
    # research_data_gaps - it needs its own opt-in.
    check_data_freshness: bool = False
    online_sources: bool = False
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
            allow_mock_providers=_env_bool("ALLOW_MOCK_PROVIDERS", True),
            provider_timeout_seconds=_env_int("PROVIDER_TIMEOUT_SECONDS", 20),
            research_data_gaps=_env_bool("RESEARCH_DATA_GAPS", False),
            research_data_gaps_max=_env_int("RESEARCH_DATA_GAPS_MAX", 8),
            check_data_freshness=_env_bool("CHECK_DATA_FRESHNESS", False),
            online_sources=_env_bool("ONLINE_SOURCES", False),
            planning_reasoning_effort=_env("PLANNING_REASONING_EFFORT"),
            credentials=ProviderCredentials(
                market_data_api_key=_env("MARKET_DATA_API_KEY"),
                market_data_base_url=_env("MARKET_DATA_BASE_URL"),
                fundamentals_api_key=_env("FUNDAMENTALS_API_KEY"),
                fundamentals_base_url=_env("FUNDAMENTALS_BASE_URL"),
                documents_api_key=_env("DOCUMENTS_API_KEY"),
                documents_base_url=_env("DOCUMENTS_BASE_URL"),
                sec_user_agent=_env("SEC_USER_AGENT"),
                megadata_base_url=_env("MEGADATA_BASE_URL"),
                megadata_api_key=_env("MEGADATA_API_KEY"),
                arcticdb_uri=_env("ARCTICDB_URI"),
            ),
            model=ModelConfig(
                provider=_env("MODEL_PROVIDER", "openrouter") or "openrouter",
                model=_env("MODEL_NAME", "openai/gpt-5") or "openai/gpt-5",
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
            "allow_mock_providers": self.allow_mock_providers,
            "provider_timeout_seconds": self.provider_timeout_seconds,
            "research_data_gaps": self.research_data_gaps,
            "research_data_gaps_max": self.research_data_gaps_max,
            "check_data_freshness": self.check_data_freshness,
            "online_sources": self.online_sources,
            "planning_reasoning_effort": self.planning_reasoning_effort,
            "credentials_present": {
                "market_data": self.credentials.has_market_data(),
                "fundamentals": self.credentials.has_fundamentals(),
                "documents": self.credentials.has_documents(),
                "megadata": self.credentials.has_megadata(),
                "arcticdb": self.credentials.has_arcticdb(),
            },
            "model": {
                "provider": self.model.provider,
                "model": self.model.model,
                "base_url": self.model.base_url,
                "enabled": self.model.enabled,
            },
        }
