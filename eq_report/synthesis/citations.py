"""Citation numbering for the report.

Reference numbers are assigned in order of first use so the source list reads in
the order a reader meets the claims. The registry is the only place that maps an
evidence id to a printed reference, which keeps the provenance chain intact:
every printed number traces to a reference, a reference to an evidence id, and
an evidence id to a source.
"""

from __future__ import annotations

from ..domain.evidence import EvidenceItem
from ..domain.report import Citation
from ..evidence.reader import EvidenceReader
from ..normalisation.canonical_metrics import display_label


class CitationRegistry:
    """Assigns and holds reference numbers for cited evidence."""

    def __init__(self, reader: EvidenceReader) -> None:
        self._reader = reader
        self._numbers: dict[str, int] = {}
        self._citations: list[Citation] = []
        self._missing: list[str] = []

    def refs_for(self, evidence_ids: tuple[str, ...]) -> tuple[int, ...]:
        """Reference numbers for a set of evidence ids, registering new ones.

        An id that is not in the Evidence Store is recorded in ``missing`` rather
        than silently dropped - QA turns that into a critical finding.
        """
        refs: list[int] = []
        for evidence_id in evidence_ids:
            if evidence_id in self._numbers:
                refs.append(self._numbers[evidence_id])
                continue
            item = self._reader.get(evidence_id)
            if item is None:
                if evidence_id not in self._missing:
                    self._missing.append(evidence_id)
                continue
            number = len(self._citations) + 1
            self._numbers[evidence_id] = number
            self._citations.append(self._citation(number, item))
            refs.append(number)
        return tuple(dict.fromkeys(refs))

    @staticmethod
    def _citation(number: int, item: EvidenceItem) -> Citation:
        text = item.citation()
        section = item.metadata.get("section")
        if section:
            text = f"{text}, {section}"
        if item.metric:
            # Without the metric, several references to the same feed are
            # indistinguishable in the source list, which defeats the purpose.
            label = display_label(item.metric)
            qualifier = item.metadata.get("segment_name") or item.metadata.get("kpi_name")
            if qualifier:
                label = f"{label}: {qualifier}"
            text = f"{text}, {label}"
        if item.period_label:
            text = f"{text} ({item.period_label})"
        return Citation(
            ref_number=number,
            evidence_id=item.evidence_id,
            text=text,
            source_url=item.source_url,
            is_mock=item.is_mock,
        )

    @property
    def citations(self) -> tuple[Citation, ...]:
        return tuple(self._citations)

    @property
    def missing_evidence_ids(self) -> tuple[str, ...]:
        return tuple(self._missing)
