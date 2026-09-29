"""Market Commentary agent.

Surfaces named, attributable third-party commentary on the stock - sell-side
rating/price-target moves, and other identifiable named views - and
interprets what they collectively imply about market positioning. An
unattributed "analysts expect" passage is not admitted: a comment must name
who said it, or it is indistinguishable from filler.
"""

from __future__ import annotations

import re

from ..domain.enums import ClaimType, SegmentName, SourceType
from ..domain.segment import KeyFinding, SegmentResult
from .base import AgentContext, SegmentAgent

#: A plain, transparent keyword filter (see EvidenceReader.documents_matching),
#: not semantic retrieval. Deliberately narrow: this section exists only for
#: named, attributable views, not general market/price chatter.
_ANALYST_KEYWORDS = (
    "price target", "overweight", "underweight", "outperform", "underperform",
    "buy rating", "sell rating", "hold rating", "reiterat", "downgrade",
    "upgrade", "analyst",
)

#: Mention of one of these marks a passage as attributable rather than
#: anonymous "analysts say" chatter.
_NAMED_SOURCE_PATTERN = re.compile(
    r"\b(Morgan Stanley|Goldman Sachs|JPMorgan|J\.P\. Morgan|Citi|Citigroup|"
    r"Evercore|Wells Fargo|Bank of America|BofA|UBS|Barclays|Deutsche Bank|"
    r"Wedbush|KeyBanc|Piper Sandler|Jefferies|Bernstein|Baird|Needham|"
    r"Jim Cramer|Cathie Wood)\b"
)


class MarketCommentaryAgent(SegmentAgent):
    segment = SegmentName.MARKET_COMMENTARY

    async def _analyse(self, context: AgentContext) -> SegmentResult:
        reader = context.reader

        candidates = reader.documents_matching(
            list(_ANALYST_KEYWORDS), (SourceType.NEWS,), limit=20,
        )
        attributable = [
            item for item in candidates
            if _NAMED_SOURCE_PATTERN.search(item.claim_text or "")
        ]

        seen_titles: set[str] = set()
        findings: list[KeyFinding] = []

        for item in attributable:
            title = item.document_title or ""
            if title in seen_titles:
                continue
            seen_titles.add(title)

            text = item.claim_text or ""
            source_match = _NAMED_SOURCE_PATTERN.search(text)
            source_name = source_match.group(0) if source_match else "A named commentator"

            findings.append(self.document_finding(
                item, claim_type=ClaimType.MARKET_EXPECTATION, materiality=2,
                tags=("commentary", "sentiment"),
                prefix=f"{source_name}: ",
            ))

            if len(findings) >= 8:
                break

        if not findings:
            return SegmentResult(
                segment=self.segment,
                headline="No attributable named commentary was retrieved",
                data_gaps=(self.gap(
                    "No attributable named-analyst or named-commentator "
                    "coverage was retrieved.",
                    impact="Market positioning can only be inferred from price "
                           "and options-implied data, not from stated "
                           "third-party views.",
                    segment=self.segment,
                ),),
            )

        # No price-target arithmetic here: targets scraped from passage text
        # have no canonical numeric evidence, so QA would block every figure.
        headline = f"{len(findings)} attributable named view(s) were retrieved on the stock"

        return SegmentResult(
            segment=self.segment,
            headline=headline,
            key_findings=tuple(findings),
            draft_narrative=headline,
        )
