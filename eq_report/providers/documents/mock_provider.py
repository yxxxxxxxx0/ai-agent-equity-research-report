"""Fallback documents provider backed by the illustrative sample corpus.

Returns extracted passages, not whole documents. Each passage keeps the full
citation set the Evidence Store requires: title, document type, publisher,
publication date, URL, section and the extracted text itself.
"""

from __future__ import annotations

import asyncio

from ...domain.enums import Confidence, SourceType
from ...domain.observation import ProviderResult, RawDocumentPassage, SourceRef
from ...domain.plan import ResearchPlan
from ..base import DocumentsProvider
from ..sample_data import DOCUMENTS, MOCK_NOTICE


class MockDocumentsProvider(DocumentsProvider):
    """Deterministic sample filings, releases, transcripts, announcements and news."""

    name = "mock_documents"
    is_mock = True

    def is_available(self) -> bool:
        return self.settings.allow_mock_providers

    async def _fetch(self, plan: ResearchPlan) -> ProviderResult:
        await asyncio.sleep(0)

        ticker = plan.ticker
        corpus = DOCUMENTS.get(ticker or "", [])
        if not corpus:
            return self.ok(
                warnings=(
                    f"No sample documents available for ticker {ticker or '<unknown>'}.",
                ),
            )

        # The plan states which document types are wanted and how many of each.
        wanted: dict[SourceType, int] = {
            req.source_type: req.max_documents for req in plan.required_documents
        }
        taken: dict[SourceType, int] = {}

        passages: list[RawDocumentPassage] = []
        warnings: list[str] = []
        errors: list[str] = []

        for doc in corpus:
            try:
                source_type = SourceType(doc["source_type"])
            except ValueError:
                errors.append(f"Unknown document source_type {doc['source_type']!r}; skipped.")
                continue

            if wanted and source_type not in wanted:
                continue
            if taken.get(source_type, 0) >= wanted.get(source_type, 0):
                continue
            taken[source_type] = taken.get(source_type, 0) + 1

            source = SourceRef(
                source_id=f"{self.name}:{doc['url']}",
                source_name=doc["publisher"],
                source_type=source_type,
                source_url=doc["url"],
                is_mock=True,
            )
            for passage in doc["passages"]:
                passages.append(
                    RawDocumentPassage(
                        title=doc["title"],
                        source=source,
                        published_at=doc["published"],
                        text=passage["text"],
                        company=plan.company,
                        ticker=plan.ticker,
                        section=passage.get("section"),
                        speaker=passage.get("speaker"),
                        confidence=self._confidence_for(source_type),
                        metadata={"mock_notice": MOCK_NOTICE},
                    )
                )

        for source_type, limit in wanted.items():
            if taken.get(source_type, 0) == 0:
                warnings.append(
                    f"No {source_type.value} documents found "
                    f"(plan requested up to {limit})."
                )

        return self.ok(passages=tuple(passages), warnings=tuple(warnings),
                       errors=tuple(errors))

    @staticmethod
    def _confidence_for(source_type: SourceType) -> Confidence:
        """Filings are the most reliable source; news the least."""
        if source_type in {SourceType.COMPANY_FILING, SourceType.EARNINGS_RELEASE}:
            return Confidence.HIGH
        if source_type in {SourceType.NEWS, SourceType.INDUSTRY_RESEARCH}:
            return Confidence.LOW
        return Confidence.MEDIUM
