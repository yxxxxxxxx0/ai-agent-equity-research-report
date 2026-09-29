"""Stage 2 - the typed research plan produced by the Research Planner."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .enums import ReportSection, SegmentName, SourceType
from .request import ResearchRequest


@dataclass(frozen=True, slots=True)
class ResearchQuestion:
    """One analytical question the report must answer."""

    question_id: str
    text: str
    segment: SegmentName
    priority: int = 2  # 1 = must answer, 3 = nice to have

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchQuestion":
        return cls(
            question_id=str(payload["question_id"]),
            text=str(payload["text"]),
            segment=SegmentName(str(payload["segment"])),
            priority=int(payload.get("priority", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "text": self.text,
            "segment": self.segment.value,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class DocumentRequirement:
    """A class of document the documents branch should try to retrieve."""

    source_type: SourceType
    lookback_days: int = 180
    max_documents: int = 5

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DocumentRequirement":
        return cls(
            source_type=SourceType(str(payload["source_type"])),
            lookback_days=int(payload.get("lookback_days", 180)),
            max_documents=int(payload.get("max_documents", 5)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type.value,
            "lookback_days": self.lookback_days,
            "max_documents": self.max_documents,
        }


@dataclass(frozen=True, slots=True)
class SegmentTask:
    """A unit of work assigned to one segment agent."""

    segment: SegmentName
    objective: str
    question_ids: tuple[str, ...] = ()
    required_metrics: tuple[str, ...] = ()
    required_analytics: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SegmentTask":
        return cls(
            segment=SegmentName(str(payload["segment"])),
            objective=str(payload["objective"]),
            question_ids=tuple(payload.get("question_ids") or ()),
            required_metrics=tuple(payload.get("required_metrics") or ()),
            required_analytics=tuple(payload.get("required_analytics") or ()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment": self.segment.value,
            "objective": self.objective,
            "question_ids": list(self.question_ids),
            "required_metrics": list(self.required_metrics),
            "required_analytics": list(self.required_analytics),
        }


@dataclass(frozen=True, slots=True)
class DatasetRequirement:
    """A dataset/API family the acquisition layer should satisfy."""

    dataset: str
    fields: tuple[str, ...]
    endpoints: tuple[str, ...]
    purpose: str
    priority: int = 2

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DatasetRequirement":
        return cls(
            dataset=str(payload["dataset"]),
            fields=tuple(payload.get("fields") or ()),
            endpoints=tuple(payload.get("endpoints") or ()),
            purpose=str(payload.get("purpose", "")),
            priority=int(payload.get("priority", 2)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "fields": list(self.fields),
            "endpoints": list(self.endpoints),
            "purpose": self.purpose,
            "priority": self.priority,
        }


@dataclass(frozen=True, slots=True)
class ExhibitRequirement:
    """An analytical exhibit requested by the institutional framework."""

    title: str
    analytical_question: str
    metrics: tuple[str, ...]
    comparator: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExhibitRequirement":
        return cls(
            title=str(payload["title"]),
            analytical_question=str(payload.get("analytical_question", "")),
            metrics=tuple(payload.get("metrics") or ()),
            comparator=str(payload.get("comparator", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "analytical_question": self.analytical_question,
            "metrics": list(self.metrics),
            "comparator": self.comparator,
        }


@dataclass(frozen=True, slots=True)
class DataRequest:
    """One concrete, planner-selected Megadata API retrieval."""

    request_id: str
    branch: str
    endpoint: str
    params: dict[str, str]
    purpose: str

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "DataRequest":
        return cls(
            request_id=str(payload["request_id"]),
            branch=str(payload["branch"]),
            endpoint=str(payload["endpoint"]),
            params=dict(payload.get("params") or {}),
            purpose=str(payload.get("purpose", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "branch": self.branch,
            "endpoint": self.endpoint,
            "params": dict(self.params),
            "purpose": self.purpose,
        }


@dataclass(frozen=True, slots=True)
class SearchRequest:
    """One evidence search the documents branch should execute.

    ``date_to``, ``top_k`` and ``exclude_same_source`` map directly onto
    ``/api/vector-search/search``'s own query parameters (its OpenAPI schema
    supports date bounding, a result count and de-duplication by source
    natively). ``None``/default values leave that parameter unset so
    ``/api/news/data`` searches, which take their own date params, are
    unaffected.
    """

    request_id: str
    query: str
    endpoint: str = "/api/vector-search/search"
    purpose: str = ""
    date_to: str | None = None
    top_k: int | None = None
    exclude_same_source: bool = False

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "SearchRequest":
        return cls(
            request_id=str(payload["request_id"]),
            query=str(payload["query"]),
            endpoint=str(payload.get("endpoint") or "/api/vector-search/search"),
            purpose=str(payload.get("purpose", "")),
            date_to=payload.get("date_to"),
            top_k=payload.get("top_k"),
            exclude_same_source=bool(payload.get("exclude_same_source", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "query": self.query,
            "endpoint": self.endpoint,
            "purpose": self.purpose,
            "date_to": self.date_to,
            "top_k": self.top_k,
            "exclude_same_source": self.exclude_same_source,
        }


@dataclass(frozen=True, slots=True)
class ResearchPlan:
    """Everything downstream stages need, derived from the user request.

    The planner performs no research itself: this object is a pure declaration
    of what must be fetched, computed and written.
    """

    request: ResearchRequest
    company: str
    ticker: str | None
    peers: tuple[str, ...]
    questions: tuple[ResearchQuestion, ...]
    required_market_metrics: tuple[str, ...]
    required_fundamental_metrics: tuple[str, ...]
    required_documents: tuple[DocumentRequirement, ...]
    sections: tuple[ReportSection, ...]
    required_analytics: tuple[str, ...]
    segment_tasks: tuple[SegmentTask, ...]
    source_priorities: tuple[SourceType, ...]
    notes: tuple[str, ...] = field(default=())
    exchange: str | None = None
    benchmark: str | None = None
    dataset_requirements: tuple[DatasetRequirement, ...] = ()
    report_outline: tuple[str, ...] = ()
    exhibit_requirements: tuple[ExhibitRequirement, ...] = ()
    planner_model: str | None = None
    data_requests: tuple[DataRequest, ...] = ()
    search_requests: tuple[SearchRequest, ...] = ()

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ResearchPlan":
        """Rebuild a plan from its ``to_dict``/``01_plan.json`` shape.

        Lets a later stage (or a full pipeline re-run against the same
        Evidence Store rows) resume without re-invoking the Research
        Planner: every downstream stage only ever reads the plan through
        this object, never through the planner call that produced it.
        """
        return cls(
            request=ResearchRequest.from_dict(payload["request"]),
            company=str(payload["company"]),
            ticker=payload.get("ticker"),
            peers=tuple(payload.get("peers") or ()),
            questions=tuple(
                ResearchQuestion.from_dict(q) for q in payload.get("questions") or ()),
            required_market_metrics=tuple(payload.get("required_market_metrics") or ()),
            required_fundamental_metrics=tuple(
                payload.get("required_fundamental_metrics") or ()),
            required_documents=tuple(
                DocumentRequirement.from_dict(d) for d in payload.get("required_documents") or ()),
            sections=tuple(ReportSection(str(s)) for s in payload.get("sections") or ()),
            required_analytics=tuple(payload.get("required_analytics") or ()),
            segment_tasks=tuple(
                SegmentTask.from_dict(t) for t in payload.get("segment_tasks") or ()),
            source_priorities=tuple(
                SourceType(str(s)) for s in payload.get("source_priorities") or ()),
            notes=tuple(payload.get("notes") or ()),
            exchange=payload.get("exchange"),
            benchmark=payload.get("benchmark"),
            dataset_requirements=tuple(
                DatasetRequirement.from_dict(d) for d in payload.get("dataset_requirements") or ()),
            report_outline=tuple(payload.get("report_outline") or ()),
            exhibit_requirements=tuple(
                ExhibitRequirement.from_dict(e) for e in payload.get("exhibit_requirements") or ()),
            planner_model=payload.get("planner_model"),
            data_requests=tuple(
                DataRequest.from_dict(r) for r in payload.get("data_requests") or ()),
            search_requests=tuple(
                SearchRequest.from_dict(r) for r in payload.get("search_requests") or ()),
        )

    def questions_for(self, segment: SegmentName) -> tuple[ResearchQuestion, ...]:
        return tuple(q for q in self.questions if q.segment is segment)

    def task_for(self, segment: SegmentName) -> SegmentTask | None:
        for task in self.segment_tasks:
            if task.segment is segment:
                return task
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "company": self.company,
            "ticker": self.ticker,
            "peers": list(self.peers),
            "questions": [q.to_dict() for q in self.questions],
            "required_market_metrics": list(self.required_market_metrics),
            "required_fundamental_metrics": list(self.required_fundamental_metrics),
            "required_documents": [d.to_dict() for d in self.required_documents],
            "sections": [s.value for s in self.sections],
            "required_analytics": list(self.required_analytics),
            "segment_tasks": [t.to_dict() for t in self.segment_tasks],
            "source_priorities": [s.value for s in self.source_priorities],
            "notes": list(self.notes),
            "exchange": self.exchange,
            "benchmark": self.benchmark,
            "dataset_requirements": [d.to_dict() for d in self.dataset_requirements],
            "report_outline": list(self.report_outline),
            "exhibit_requirements": [e.to_dict() for e in self.exhibit_requirements],
            "planner_model": self.planner_model,
            "data_requests": [r.to_dict() for r in self.data_requests],
            "search_requests": [r.to_dict() for r in self.search_requests],
        }
