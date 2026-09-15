"""Stage 2 - the Research Planner.

Turns a ResearchRequest into a typed ResearchPlan. The planner is deliberately
inert: it decides *what* must be researched and computed, and never fetches
anything itself. That keeps the acquisition branches replaceable and makes the
plan a reviewable artefact in its own right.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Protocol

from ..config import ModelConfig
from ..domain.enums import ReportSection, SegmentName, SourceType
from ..domain.plan import (
    DataRequest,
    DatasetRequirement,
    DocumentRequirement,
    ExhibitRequirement,
    ResearchPlan,
    ResearchQuestion,
    SearchRequest,
    SegmentTask,
)
from ..domain.request import ResearchRequest
from ..llm.usage import UsageTracker
from ..logging_setup import get_logger, log_event
from ..normalisation.canonical_metrics import (
    FUNDAMENTAL_METRIC_PLAN,
    MARKET_METRIC_PLAN,
)
from .openrouter_client import OpenRouterPlannerClient, PlannerModelResponse

logger = get_logger("planning.research_planner")

_REPORT_OUTLINE = (
    "0. Executive Dashboard — Why Should I Look at This Stock Now?",
    "1. Company & Economic Engine",
    "2. Material Developments — What Changed?",
    "3. Financial Performance — What Did the Numbers Show?",
    "4. Operating Drivers & Segment Economics — Why Did the Numbers Move?",
    "5. Demand, Industry, Competition & Supply — Can the Drivers Continue?",
    "6. Management Outlook — What Does the Company Expect Next?",
    "7. Consensus & Estimate Revisions — What Does the Market Expect?",
    "8. Valuation & Market-Implied Expectations — What Is Priced In?",
    "9. Market Debates & Positioning — Where Could Expectations Be Wrong?",
    "10. Catalysts, Risks & Observation Points — What Could Change Expectations?",
    "11. Monitoring Dashboard — What Should We Watch?",
    "12. Final — What Matters Next?",
)

_DATASETS = (
    DatasetRequirement("company_master", ("ticker", "exchange", "name", "sector", "industry", "peers"),
                       ("/api/bbg/company_details/data", "/api/bbg/sector_taxonomy/data", "/api/bbg/target_universe/data"),
                       "Resolve the issuer, classification, benchmark and relevant peers", 1),
    DatasetRequirement("market_data", ("ohlcv", "price", "market_cap", "adv", "returns", "volatility"),
                       ("/api/bbg/ohlcv/data", "/api/bbg/market-cap/data", "/api/market/bbg/data", "/api/market/tradestation/ohlcuvdv"),
                       "Measure company and benchmark performance, liquidity and positioning", 1),
    DatasetRequirement("financial_fundamentals", ("revenue", "eps", "margins", "ebit", "ebitda", "ocf", "fcf", "capex", "balance_sheet"),
                       ("/api/bbg/indicators/data", "/api/news/filings-by-form"),
                       "Measure results, history and cash conversion with filing verification", 1),
    DatasetRequirement("segments_and_kpis", ("segment_revenue", "growth", "mix", "units", "asp", "bookings", "backlog"),
                       ("/api/bbg/segment-revenue/data", "/api/bbg/indicators/data"),
                       "Identify the economic and operating drivers of reported changes", 1),
    DatasetRequirement("expectations", ("consensus", "high", "low", "median", "estimate_count", "revisions"),
                       ("/api/bbg/estimates/data",), "Compare actuals, guidance and changing market expectations", 1),
    DatasetRequirement("company_narrative_events", ("guidance", "products", "customers", "developments", "management_commentary"),
                       ("/api/news/data", "/api/news/item", "/api/alpha-vantage/earning-call-transcripts", "/api/news/filings-by-form"),
                       "Trace material events into economic drivers", 1),
    DatasetRequirement("industry_supply_chain", ("customers", "suppliers", "industry_growth", "competitors", "macro"),
                       ("/api/bbg/supply-chain/data", "/api/bea-gdp-by-industry/data", "/api/bbg/macroeconomic/data"),
                       "Externally test demand, competition and supply constraints", 2),
    DatasetRequirement("positioning_catalysts", ("implied_move", "earnings_dates", "corporate_actions", "events"),
                       ("/api/bbg/implied-move/data", "/api/bbg/corporate-actions/data", "/api/alpha-vantage/earning-call-future", "/api/economic-events/current-month"),
                       "Map sensitivity and dated events that could resolve debates", 2),
)


class PlannerClient(Protocol):
    async def create_plan(
        self, system_prompt: str, user_prompt: str
    ) -> PlannerModelResponse: ...


_SYSTEM_PROMPT = """You are the research-planning component for a neutral institutional
company intelligence brief. Plan research; do not write the report and do not invent data.
The audience is portfolio managers, traders and equity analysts. The causal flow must be:
why interesting now -> economic engine -> material changes -> financial impact -> operating
drivers -> external demand/competition/supply validation -> management outlook -> consensus
and revisions -> valuation/market-implied expectations -> debates/positioning -> catalysts and
risks -> monitoring KPIs -> what matters next.

Never plan a Buy/Hold/Sell view, directional price forecast, target price, position sizing or
trade instruction. Treat analyst targets only as external market statistics. Explicitly separate
facts, management views, market expectations, market data, derived figures, interpretations and
data gaps. Prefer primary/regulatory evidence, then external primary evidence, institutional
market data, specialist data and finally secondary sources. Every important number needs a
comparator. Plan 4-7 conclusion-bearing exhibits. Return JSON only, matching the requested
schema. Use only the allowed enum values supplied by the user. Company identity, exchange,
benchmark and peers are planning hypotheses that acquisition must verify."""

_ENDPOINT_BRANCH = {
    "/api/bbg/company_details/data": "market_data",
    "/api/bbg/sector_taxonomy/data": "market_data",
    "/api/bbg/target_universe/data": "market_data",
    "/api/bbg/ohlcv/data": "market_data",
    "/api/bbg/market-cap/data": "market_data",
    "/api/market/bbg/data": "market_data",
    "/api/market/bbg/ohlcv": "market_data",
    "/api/market-tradestation/ohlcuvdv": "market_data",
    "/api/bbg/implied-move/data": "market_data",
    "/api/bbg/corporate-actions/data": "market_data",
    "/api/bbg/indicators/data": "fundamentals",
    "/api/bbg/estimates/data": "fundamentals",
    "/api/bbg/segment-revenue/data": "fundamentals",
    "/api/bbg/supply-chain/data": "fundamentals",
    "/api/bbg/macroeconomic/data": "fundamentals",
    "/api/bea-gdp-by-industry/data": "fundamentals",
    "/api/news/data": "documents",
    "/api/news/item": "documents",
    "/api/news/filings": "documents",
    "/api/news/filings-by-form": "documents",
    "/api/alpha-vantage/earning-call-transcripts": "documents",
    "/api/alpha-vantage/earning-call-historical": "documents",
    "/api/alpha-vantage/earning-call-future": "documents",
    "/api/vector-search/search": "documents",
    "/api/economic-events/current-month": "documents",
}

# A tiny, explicit lookup. A production planner would resolve tickers against a
# security master; for the prototype an unknown company simply plans without a
# ticker and the acquisition branches record that as a data gap.
_KNOWN_TICKERS: dict[str, str] = {
    "nvidia": "NVDA",
    "nvidia corporation": "NVDA",
    "advanced micro devices": "AMD",
    "amd": "AMD",
    "intel": "INTC",
    "broadcom": "AVGO",
    "microsoft": "MSFT",
    "apple": "AAPL",
    "alphabet": "GOOGL",
    "google": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "tesla": "TSLA",
}

# Default peer sets used when the user does not supply peers.
_DEFAULT_PEERS: dict[str, tuple[str, ...]] = {
    "NVDA": ("AMD", "INTC", "AVGO"),
    "AMD": ("NVDA", "INTC", "AVGO"),
    "INTC": ("NVDA", "AMD", "AVGO"),
}

# Which segment agent owns which requested report section.
_SECTION_TO_SEGMENT: dict[ReportSection, SegmentName] = {
    ReportSection.COMPANY_SNAPSHOT: SegmentName.COMPANY_SNAPSHOT,
    ReportSection.FINANCIALS: SegmentName.FINANCIAL_PERFORMANCE,
    ReportSection.OPERATING_DRIVERS: SegmentName.OPERATING_DRIVERS,
    ReportSection.RECENT_DEVELOPMENTS: SegmentName.RECENT_DEVELOPMENTS,
    ReportSection.VALUATION: SegmentName.VALUATION,
    ReportSection.COMPETITIVE_LANDSCAPE: SegmentName.COMPETITIVE_LANDSCAPE,
    ReportSection.RISKS: SegmentName.RISKS_CATALYSTS,
    ReportSection.CATALYSTS: SegmentName.RISKS_CATALYSTS,
    ReportSection.WHAT_MATTERS_NEXT: SegmentName.WHAT_MATTERS_NEXT,
    # KEY_TAKEAWAYS and SOURCES are produced by the synthesis layer, not an agent.
}

_QUESTION_LIBRARY: tuple[tuple[str, str, SegmentName, int], ...] = (
    ("q_profile", "What does the company do and how is it currently priced by the market?",
     SegmentName.COMPANY_SNAPSHOT, 2),
    ("q_results_change", "What changed in the latest financial results?",
     SegmentName.FINANCIAL_PERFORMANCE, 1),
    ("q_consensus_surprise", "What surprised relative to consensus?",
     SegmentName.FINANCIAL_PERFORMANCE, 1),
    ("q_margin_trend", "How are margins and cash generation trending?",
     SegmentName.FINANCIAL_PERFORMANCE, 2),
    ("q_drivers", "Which operating drivers are accelerating or weakening?",
     SegmentName.OPERATING_DRIVERS, 1),
    ("q_segment_mix", "How is the revenue mix shifting across segments?",
     SegmentName.OPERATING_DRIVERS, 2),
    ("q_guidance", "Has management guidance changed, and how does it compare with consensus?",
     SegmentName.RECENT_DEVELOPMENTS, 1),
    ("q_recent_events", "What material events have occurred recently?",
     SegmentName.RECENT_DEVELOPMENTS, 1),
    ("q_valuation_change", "How has valuation changed?",
     SegmentName.VALUATION, 1),
    ("q_market_expectations", "What does the market currently appear to expect?",
     SegmentName.VALUATION, 1),
    ("q_competition", "What are the main competitive developments?",
     SegmentName.COMPETITIVE_LANDSCAPE, 1),
    ("q_relative_position",
     "How does the company compare with peers on growth and valuation?",
     SegmentName.COMPETITIVE_LANDSCAPE, 2),
    ("q_risks", "What are the main risks to the current trajectory?",
     SegmentName.RISKS_CATALYSTS, 1),
    ("q_catalysts", "What are the important near-term catalysts?",
     SegmentName.RISKS_CATALYSTS, 1),
    ("q_monitor", "What should investors monitor next?",
     SegmentName.WHAT_MATTERS_NEXT, 1),
)

# Analytics the engine should attempt, keyed by the segment that consumes them.
_SEGMENT_ANALYTICS: dict[SegmentName, tuple[str, ...]] = {
    SegmentName.COMPANY_SNAPSHOT: ("price_return_1m", "price_return_3m", "price_return_ytd"),
    SegmentName.FINANCIAL_PERFORMANCE: (
        "revenue_growth_yoy", "revenue_growth_qoq", "eps_growth_yoy",
        "gross_margin_change_yoy", "operating_margin_change_yoy",
        "revenue_surprise_pct", "eps_surprise_pct", "fcf_growth_yoy", "cash_conversion",
    ),
    SegmentName.OPERATING_DRIVERS: ("segment_growth_yoy", "segment_contribution_pct"),
    SegmentName.RECENT_DEVELOPMENTS: ("guidance_vs_consensus_pct", "estimate_revision_direction"),
    SegmentName.VALUATION: (
        "forward_pe", "ev_to_sales", "valuation_vs_peers", "valuation_vs_history",
        "price_return_ytd",
    ),
    SegmentName.COMPETITIVE_LANDSCAPE: ("peer_growth_gap", "valuation_vs_peers"),
    SegmentName.RISKS_CATALYSTS: ("revenue_growth_qoq", "gross_margin_change_yoy"),
    SegmentName.WHAT_MATTERS_NEXT: ("trend_direction", "guidance_vs_consensus_pct"),
}

_SEGMENT_OBJECTIVES: dict[SegmentName, str] = {
    SegmentName.COMPANY_SNAPSHOT:
        "Establish what the company is, its scale, and how the market currently prices it.",
    SegmentName.FINANCIAL_PERFORMANCE:
        "Describe what changed in the latest reported period and how it compared "
        "with expectations.",
    SegmentName.OPERATING_DRIVERS:
        "Identify which business segments and operating KPIs drove the reported result.",
    SegmentName.RECENT_DEVELOPMENTS:
        "Summarise material recent events, announcements and guidance changes.",
    SegmentName.VALUATION:
        "Assess current valuation against peers and history, and infer embedded expectations.",
    SegmentName.COMPETITIVE_LANDSCAPE:
        "Assess competitive position and relative performance against named peers.",
    SegmentName.RISKS_CATALYSTS:
        "Identify the material risks to the trajectory and the near-term catalysts.",
    SegmentName.WHAT_MATTERS_NEXT:
        "State the specific, observable items an investor should monitor next.",
}

_DEFAULT_SOURCE_PRIORITIES: tuple[SourceType, ...] = (
    SourceType.COMPANY_FILING,
    SourceType.EARNINGS_RELEASE,
    SourceType.EARNINGS_CALL,
    SourceType.INVESTOR_PRESENTATION,
    SourceType.MARKET_DATA,
    SourceType.SELL_SIDE_CONSENSUS,
    SourceType.COMPANY_ANNOUNCEMENT,
    SourceType.NEWS,
    SourceType.INDUSTRY_RESEARCH,
    SourceType.COMPETITOR_FILING,
)


class ResearchPlanner:
    """Builds the typed research plan, using GPT through OpenRouter when configured."""

    def __init__(
        self,
        model_config: ModelConfig | None = None,
        *,
        client: PlannerClient | None = None,
        logger_: logging.Logger | None = None,
        tracker: UsageTracker | None = None,
    ) -> None:
        self._logger = logger_ or logger
        self._model_config = model_config
        self._client = client
        if self._client is None and model_config and model_config.enabled:
            if model_config.provider.lower() != "openrouter":
                raise ValueError("ResearchPlanner supports MODEL_PROVIDER=openrouter")
            self._client = OpenRouterPlannerClient(model_config, tracker=tracker)

    async def plan(self, request: ResearchRequest) -> ResearchPlan:
        """Create a GPT plan when enabled; otherwise use the deterministic plan."""
        if self._client is None:
            return self.plan_sync(request)
        response = await self._client.create_plan(
            _SYSTEM_PROMPT, self._user_prompt(request))
        plan = self._plan_from_model(request, response.payload)
        log_event(
            self._logger, logging.INFO, "OpenRouter GPT research plan built",
            model=self._model_config.model if self._model_config else "injected",
            input_tokens=response.input_tokens, output_tokens=response.output_tokens,
            company=plan.company, ticker=plan.ticker,
        )
        return plan

    def plan_sync(self, request: ResearchRequest) -> ResearchPlan:
        notes: list[str] = []

        ticker = request.ticker or self._resolve_ticker(request.company)
        if request.ticker is None and ticker is not None:
            notes.append(f"Ticker {ticker} resolved from the company name.")
        elif ticker is None:
            notes.append(
                "No ticker supplied and none could be resolved; "
                "market data coverage may be limited."
            )

        peers = request.peers or _DEFAULT_PEERS.get(ticker or "", ())
        if not request.peers and peers:
            notes.append("Default peer set applied: " + ", ".join(peers) + ".")

        sections = request.sections
        segments = self._segments_for(sections)
        questions = self._questions_for(segments, request)
        analytics = self._analytics_for(segments)
        segment_tasks = self._tasks_for(segments, questions, analytics)

        plan = ResearchPlan(
            request=request,
            company=request.company,
            ticker=ticker,
            peers=tuple(peers),
            questions=questions,
            required_market_metrics=MARKET_METRIC_PLAN,
            required_fundamental_metrics=FUNDAMENTAL_METRIC_PLAN,
            required_documents=self._documents_for(sections),
            sections=sections,
            required_analytics=analytics,
            segment_tasks=segment_tasks,
            source_priorities=_DEFAULT_SOURCE_PRIORITIES,
            notes=tuple(notes),
            dataset_requirements=_DATASETS,
            report_outline=_REPORT_OUTLINE,
        )

        log_event(
            self._logger, logging.INFO, "research plan built",
            company=plan.company, ticker=plan.ticker, sections=len(plan.sections),
            questions=len(plan.questions), segment_tasks=len(plan.segment_tasks),
            analytics=len(plan.required_analytics), peers=list(plan.peers),
        )
        return plan

    def _user_prompt(self, request: ResearchRequest) -> str:
        schema = {
            "company": "resolved company name or ticker if uncertain",
            "exchange": "exchange or null",
            "benchmark": "most economically relevant index/sector index or null",
            "peers": ["ticker"],
            "questions": [{
                "question_id": "unique snake_case id",
                "text": "company-specific analytical question",
                "segment": "one allowed segment",
                "priority": "integer 1-3",
            }],
            "segment_tasks": [{
                "segment": "one allowed segment",
                "objective": "specific objective",
                "question_ids": ["existing question id"],
                "required_metrics": ["metric identifiers"],
                "required_analytics": ["allowed analytic"],
            }],
            "documents": [{
                "source_type": "one allowed source type",
                "lookback_days": "integer 1-1825",
                "max_documents": "integer 1-20",
            }],
            "exhibits": [{
                "title": "conclusion-bearing title",
                "analytical_question": "question answered",
                "metrics": ["metric identifiers"],
                "comparator": "benchmark/consensus/history/peer comparator",
            }],
            "data_requests": [{
                "request_id": "unique id",
                "endpoint": "one allowed Megadata endpoint",
                "params": {"symbols": "comma-separated tickers", "from_date": "YYYY-MM-DD"},
                "purpose": "specific question this retrieval answers",
            }],
            "search_requests": [{
                "request_id": "unique id",
                "endpoint": "/api/vector-search/search or /api/news/data",
                "query": "company-specific evidence query",
                "purpose": "driver, debate or event tested",
            }],
            "notes": ["verification caveat or company-specific planning note"],
        }
        return (
            f"Plan the brief as of {request.report_date.isoformat()} for ticker "
            f"{request.ticker or request.company}.\n"
            f"Allowed segments: {[s.value for s in SegmentName]}\n"
            f"Allowed analytics: {list(dict.fromkeys(m for values in _SEGMENT_ANALYTICS.values() for m in values))}\n"
            f"Allowed source types: {[s.value for s in _DEFAULT_SOURCE_PRIORITIES]}\n"
            f"Allowed Megadata endpoints and destination branches: {_ENDPOINT_BRANCH}\n"
            f"Required report outline: {list(_REPORT_OUTLINE)}\n"
            "Design questions and tasks that cover every outline section by mapping them onto "
            "the closest allowed segment. Identify 2-4 economic drivers, external validation, "
            "management guidance, consensus revisions, market-implied expectations, 2-3 debates, "
            "dated catalysts and 5-10 monitoring KPIs. Do not assert answers.\n"
            "Return the concrete data_requests and search_requests needed to answer the plan. "
            "Use API data for structured market/fundamental/filing fields and search only for "
            "qualitative developments, causal evidence, debates and missing context.\n"
            f"Return this JSON shape: {schema}"
        )

    def _plan_from_model(
        self, request: ResearchRequest, payload: dict[str, Any]
    ) -> ResearchPlan:
        ticker = (request.ticker or request.company).strip().upper()
        questions = tuple(
            ResearchQuestion(
                question_id=str(row["question_id"]),
                text=str(row["text"]),
                segment=SegmentName(str(row["segment"])),
                priority=max(1, min(3, int(row.get("priority", 2)))),
            )
            for row in payload.get("questions", [])
        )
        if not questions:
            raise ValueError("Model research plan contained no questions")
        question_ids = {q.question_id for q in questions}
        allowed_analytics = {
            metric for values in _SEGMENT_ANALYTICS.values() for metric in values
        }
        tasks: list[SegmentTask] = []
        seen_segments: set[SegmentName] = set()
        for row in payload.get("segment_tasks", []):
            segment = SegmentName(str(row["segment"]))
            if segment in seen_segments:
                raise ValueError(f"Model research plan duplicated segment {segment.value}")
            seen_segments.add(segment)
            qids = tuple(str(q) for q in row.get("question_ids", []) if str(q) in question_ids)
            analytics = tuple(
                str(metric) for metric in row.get("required_analytics", [])
                if str(metric) in allowed_analytics
            )
            tasks.append(SegmentTask(
                segment=segment,
                objective=str(row["objective"]),
                question_ids=qids,
                required_metrics=tuple(str(m) for m in row.get("required_metrics", [])),
                required_analytics=analytics,
            ))
        expected_segments = set(self._segments_for(request.sections))
        missing_segments = expected_segments - seen_segments
        if missing_segments:
            missing = ", ".join(sorted(segment.value for segment in missing_segments))
            raise ValueError(f"Model research plan omitted required segments: {missing}")

        documents = tuple(
            DocumentRequirement(
                source_type=SourceType(str(row["source_type"])),
                lookback_days=max(1, min(1825, int(row.get("lookback_days", 365)))),
                max_documents=max(1, min(20, int(row.get("max_documents", 5)))),
            )
            for row in payload.get("documents", [])
        ) or self._documents_for(request.sections)
        exhibits = tuple(
            ExhibitRequirement(
                title=str(row["title"]),
                analytical_question=str(row["analytical_question"]),
                metrics=tuple(str(m) for m in row.get("metrics", [])),
                comparator=str(row.get("comparator", "")),
            )
            for row in payload.get("exhibits", [])[:7]
        )
        if len(exhibits) < 4:
            raise ValueError("Model research plan must contain 4-7 analytical exhibits")
        analytics = tuple(dict.fromkeys(
            metric for task in tasks for metric in task.required_analytics
        ))
        market_metrics = tuple(dict.fromkeys(
            (*MARKET_METRIC_PLAN, *(m for task in tasks for m in task.required_metrics))
        ))
        data_requests: list[DataRequest] = []
        skipped_notes: list[str] = []
        for row in payload.get("data_requests", []):
            endpoint = str(row["endpoint"])
            branch = _ENDPOINT_BRANCH.get(endpoint)
            if branch is None:
                skipped_notes.append(
                    f"Model requested unsupported data endpoint {endpoint!r}; skipped.")
                continue
            params = {str(k): str(v) for k, v in dict(row.get("params", {})).items()}
            data_requests.append(DataRequest(
                request_id=str(row["request_id"]), branch=branch, endpoint=endpoint,
                params=params, purpose=str(row["purpose"]),
            ))
        search_requests: list[SearchRequest] = []
        for row in payload.get("search_requests", []):
            endpoint = str(row.get("endpoint", "/api/vector-search/search"))
            if endpoint in {"/api/vector-search/search", "/api/news/data"}:
                search_requests.append(SearchRequest(
                    request_id=str(row["request_id"]), query=str(row["query"]),
                    endpoint=endpoint, purpose=str(row.get("purpose", "")),
                ))
                continue
            branch = _ENDPOINT_BRANCH.get(endpoint)
            if branch is not None:
                # The model filed a structured-data endpoint as a "search" -
                # honour the intent as a data request rather than discarding
                # an otherwise-valid plan over a categorisation slip.
                data_requests.append(DataRequest(
                    request_id=str(row.get("request_id", endpoint)), branch=branch,
                    endpoint=endpoint, params={}, purpose=str(row.get("purpose", "")),
                ))
                continue
            skipped_notes.append(
                f"Model requested unsupported search endpoint {endpoint!r}; skipped.")
        if not data_requests:
            raise ValueError("Model research plan contained no Megadata API requests")
        return ResearchPlan(
            request=request,
            company=str(payload.get("company") or ticker),
            ticker=ticker,
            peers=tuple(str(p).upper() for p in payload.get("peers", [])),
            questions=questions,
            required_market_metrics=market_metrics,
            required_fundamental_metrics=FUNDAMENTAL_METRIC_PLAN,
            required_documents=documents,
            sections=request.sections,
            required_analytics=analytics,
            segment_tasks=tuple(tasks),
            source_priorities=_DEFAULT_SOURCE_PRIORITIES,
            notes=tuple(str(note) for note in payload.get("notes", ())) + tuple(skipped_notes),
            exchange=payload.get("exchange"),
            benchmark=payload.get("benchmark"),
            dataset_requirements=_DATASETS,
            report_outline=_REPORT_OUTLINE,
            exhibit_requirements=exhibits,
            planner_model=self._model_config.model if self._model_config else "injected",
            data_requests=tuple(data_requests),
            search_requests=tuple(search_requests),
        )

    # -- internals -------------------------------------------------------
    def _resolve_ticker(self, company: str) -> str | None:
        key = re.sub(r"[^a-z ]", "", company.strip().lower()).strip()
        if key in _KNOWN_TICKERS:
            return _KNOWN_TICKERS[key]
        # A bare all-caps token that looks like a ticker is accepted as one.
        if re.fullmatch(r"[A-Z]{1,5}", company.strip()):
            return company.strip()
        return None

    def _segments_for(self, sections: tuple[ReportSection, ...]) -> tuple[SegmentName, ...]:
        seen: list[SegmentName] = []
        for section in sections:
            seg = _SECTION_TO_SEGMENT.get(section)
            if seg and seg not in seen:
                seen.append(seg)
        return tuple(seen)

    def _questions_for(
        self, segments: tuple[SegmentName, ...], request: ResearchRequest
    ) -> tuple[ResearchQuestion, ...]:
        questions = [
            ResearchQuestion(qid, text, segment, priority)
            for qid, text, segment, priority in _QUESTION_LIBRARY
            if segment in segments
        ]
        # Each explicit emphasis area becomes an extra, top-priority question so
        # the user's steer is visible in the plan rather than buried in prose.
        for index, focus in enumerate(request.focus):
            questions.append(
                ResearchQuestion(
                    question_id=f"q_focus_{index}",
                    text=f"User emphasis: {focus}",
                    segment=SegmentName.WHAT_MATTERS_NEXT,
                    priority=1,
                )
            )
        return tuple(questions)

    def _analytics_for(self, segments: tuple[SegmentName, ...]) -> tuple[str, ...]:
        out: list[str] = []
        for segment in segments:
            for metric in _SEGMENT_ANALYTICS.get(segment, ()):
                if metric not in out:
                    out.append(metric)
        return tuple(out)

    def _tasks_for(
        self,
        segments: tuple[SegmentName, ...],
        questions: tuple[ResearchQuestion, ...],
        analytics: tuple[str, ...],
    ) -> tuple[SegmentTask, ...]:
        tasks: list[SegmentTask] = []
        for segment in segments:
            qids = tuple(q.question_id for q in questions if q.segment is segment)
            seg_analytics = tuple(
                m for m in _SEGMENT_ANALYTICS.get(segment, ()) if m in analytics
            )
            tasks.append(
                SegmentTask(
                    segment=segment,
                    objective=_SEGMENT_OBJECTIVES[segment],
                    question_ids=qids,
                    required_analytics=seg_analytics,
                )
            )
        return tuple(tasks)

    def _documents_for(
        self, sections: tuple[ReportSection, ...]
    ) -> tuple[DocumentRequirement, ...]:
        reqs = [
            DocumentRequirement(SourceType.EARNINGS_RELEASE, lookback_days=120, max_documents=2),
            DocumentRequirement(SourceType.COMPANY_FILING, lookback_days=365, max_documents=3),
            DocumentRequirement(SourceType.EARNINGS_CALL, lookback_days=120, max_documents=1),
        ]
        if ReportSection.RECENT_DEVELOPMENTS in sections:
            reqs.append(DocumentRequirement(
                SourceType.COMPANY_ANNOUNCEMENT, lookback_days=120, max_documents=4))
            reqs.append(DocumentRequirement(
                SourceType.NEWS, lookback_days=90, max_documents=6))
        if ReportSection.COMPETITIVE_LANDSCAPE in sections:
            reqs.append(DocumentRequirement(
                SourceType.INDUSTRY_RESEARCH, lookback_days=365, max_documents=3))
            reqs.append(DocumentRequirement(
                SourceType.COMPETITOR_FILING, lookback_days=365, max_documents=3))
        if ReportSection.OPERATING_DRIVERS in sections:
            reqs.append(DocumentRequirement(
                SourceType.INVESTOR_PRESENTATION, lookback_days=180, max_documents=2))
        return tuple(reqs)
