"""Terminology normalisation for report prose.

Different sources say "data centre" and "data center", "year over year" and
"year on year", "FY26" and "FY2026". The synthesis layer rewrites all of them to
one form so the report reads as one voice rather than a concatenation of feeds.

Rewrites are applied to the *rendered statement text* only. Quoted document
passages keep their citation, so a reader who follows the reference still sees
the original wording in the source.
"""

from __future__ import annotations

import re

#: (pattern, replacement) applied in order, case-insensitively where safe.
_REWRITES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdata cent(?:re|er)\b", re.IGNORECASE), "Data Center"),
    (re.compile(r"\byear[- ]over[- ]year\b", re.IGNORECASE), "year on year"),
    (re.compile(r"\bquarter[- ]over[- ]quarter\b", re.IGNORECASE), "quarter on quarter"),
    (re.compile(r"\bYoY\b"), "year on year"),
    (re.compile(r"\bQoQ\b"), "quarter on quarter"),
    (re.compile(r"\bfree cash[- ]flow\b", re.IGNORECASE), "free cash flow"),
    (re.compile(r"\bgross margins\b", re.IGNORECASE), "gross margin"),
    (re.compile(r"\bbasis points\b", re.IGNORECASE), "basis points"),
    (re.compile(r"\bnon[- ]GAAP\b", re.IGNORECASE), "non-GAAP"),
    # FY26 -> FY2026, but leave an already four-digit year alone.
    (re.compile(r"\bFY(\d{2})\b"), r"FY20\1"),
    (re.compile(r"\bfiscal (\d{4})\b", re.IGNORECASE), r"FY\1"),
    (re.compile(r"\s+([,.;])"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
)


def normalise_terminology(text: str) -> str:
    """Apply the house style rewrites to a sentence.

    Several rewrites are case-insensitive and would otherwise lowercase a term
    that legitimately opens a sentence, so the leading capital is restored
    afterwards.
    """
    result = text
    for pattern, replacement in _REWRITES:
        result = pattern.sub(replacement, result)
    result = result.strip()
    if text[:1].isupper() and result[:1].islower():
        result = result[0].upper() + result[1:]
    return result


def claim_fingerprint(text: str) -> frozenset[str]:
    """A bag of significant words, used to spot near-duplicate claims.

    Numbers are kept (they are what distinguishes two otherwise similar claims)
    and stopwords dropped.
    """
    tokens = re.findall(r"[a-z0-9.%+-]+", text.lower())
    return frozenset(t for t in tokens if t not in _STOPWORDS and len(t) > 1)


_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for", "with", "was", "were",
    "is", "are", "be", "been", "that", "this", "it", "its", "from", "by", "as", "which", "than",
    "versus", "vs", "up", "down", "over", "per", "has", "have", "had", "will", "would", "so",
    "while", "though", "but", "not", "no", "any", "all", "more", "most", "also", "into",
})


def is_near_duplicate(
    left: frozenset[str], right: frozenset[str], *, threshold: float = 0.72
) -> bool:
    """Jaccard-similarity duplicate test.

    Deliberately a plain set overlap: it catches the real duplication case (two
    agents reporting the same number from the same evidence) without collapsing
    two genuinely different claims that happen to share vocabulary.
    """
    if not left or not right:
        return False
    intersection = len(left & right)
    union = len(left | right)
    return union > 0 and intersection / union >= threshold


def id_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    """Jaccard overlap of two evidence/analytics id sets.

    Two statements that cite the same handful of ids are almost always the
    same underlying fact, however differently they are worded - this is what
    lets cross-section duplicate detection catch a restated fact that dodges
    the text-based fingerprint by using different vocabulary. See
    ``is_near_duplicate`` for the wording-based test this complements.
    """
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    union = len(left | right)
    return intersection / union if union else 0.0


#: Words asserting causation. A causal claim needs either a management source
#: or a calculation behind it, not just a correlation the agent noticed - see
#: qa.checks.check_causal_claims, which still flags every one of these when
#: the claim's type does not justify it. Kept here, not duplicated in qa.checks,
#: so the QA check and the rewrite below always agree on what counts as causal.
CAUSAL_MARKERS: tuple[str, ...] = (
    "because", "driven by", "drove", "due to", "as a result of", "led to",
    "caused", "thanks to", "on the back of", "resulted in",
)

#: The subset of CAUSAL_MARKERS safe to rewrite mechanically: each always
#: precedes a noun phrase ("driven by strong demand"), so swapping in a
#: neutral preposition cannot break the sentence's grammar. The remaining
#: markers ("because", "led to", "caused", "drove", "resulted in") can just as
#: easily introduce a full clause ("because demand increased"), where the same
#: swap would not parse - those are left to the QA warning instead of rewritten.
_SAFE_CAUSAL_REWRITES: dict[str, str] = {
    "driven by": "alongside",
    "due to": "alongside",
    "thanks to": "alongside",
    "on the back of": "alongside",
    "as a result of": "alongside",
}
_SAFE_CAUSAL_RE = re.compile(
    "|".join(re.escape(marker) for marker in _SAFE_CAUSAL_REWRITES), re.IGNORECASE)


def soften_unsupported_causation(text: str, *, supported: bool) -> str:
    """Rewrite a statement's safe causal markers to neutral language.

    ``supported`` should be true only when the statement's claim type already
    justifies causal language (a management statement, a confirmed fact, or a
    calculation that isolates the driver). When false, this asserts
    correlation/coincidence rather than causation for the markers that can be
    swapped without risking the sentence's grammar; the harder markers are
    left alone and still caught by check_causal_claims.
    """
    if supported:
        return text

    def _replace(match: re.Match[str]) -> str:
        replacement = _SAFE_CAUSAL_REWRITES[match.group(0).lower()]
        return replacement[0].upper() + replacement[1:] if match.group(0)[0].isupper() \
            else replacement

    return _SAFE_CAUSAL_RE.sub(_replace, text)
