import datetime as dt

from eq_report.domain.enums import EvidenceCategory, EvidenceStatus, FactType, SourceType
from eq_report.domain.evidence import EvidenceItem
from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore
from eq_report.llm.client import LLMJSONResponse
from eq_report.pipeline.web_gap_fill import (
    _is_company_domain,
    _on_curated_list,
    _thin_topics,
    fill_evidence_gaps,
)

_ALLOWED = ("sec.gov", "prnewswire.com")


def test_companys_own_domain_is_recognised_by_name():
    # A company's own investor-relations/newsroom page is a legitimate
    # primary source even though no fixed allow-list can name it in
    # advance - but the name match is a fuzzy heuristic, not curation, so
    # it must still go through independent legitimacy verification later.
    assert _is_company_domain("www.apple.com", "Apple Inc.")
    assert _is_company_domain("investor.apple.com", "Apple Inc.")
    assert not _on_curated_list("https://www.apple.com/newsroom/x", _ALLOWED)


def test_a_similarly_named_third_party_site_is_not_mistaken_for_the_company():
    # "appleinsider.com" merely contains the word "apple" - it is not
    # Apple's own domain, and must not be admitted on that basis alone.
    assert not _is_company_domain("appleinsider.com", "Apple Inc.")


def test_curated_domains_are_recognised_without_a_company_name_match():
    assert _on_curated_list("https://www.sec.gov/example-10k", _ALLOWED)
    assert not _on_curated_list("https://randomblog.example/leak", _ALLOWED)


def _reader() -> EvidenceReader:
    store = EvidenceStore(":memory:")
    return EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")


def test_company_snapshot_stays_thin_despite_unrelated_documents_existing():
    # Regression: a run can hold dozens of general news items with nothing
    # about the company's own profile - "any document exists" is not a
    # signal that company_snapshot has what it needs.
    store = EvidenceStore(":memory:")
    store.save([
        EvidenceItem(
            evidence_id=f"news{i}", report_run_id="run", company="Example Corp", ticker="EX",
            category=EvidenceCategory.DOCUMENT, source_id="src", source_name="Wire",
            source_type=SourceType.NEWS, retrieved_at=dt.datetime.now(dt.timezone.utc),
            claim_text=f"unrelated industry news {i}",
            published_at=dt.date(2026, 9, 1), fact_type=FactType.REPORTED_FACT,
            status=EvidenceStatus.VALIDATED,
        )
        for i in range(20)
    ])
    reader = EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")
    assert "company_snapshot" in _thin_topics(reader)
    assert "catalysts" in _thin_topics(reader)


class _FakeClient:
    """A search call proposing three candidates, then a verify call that only
    confirms the allow-listed one it is asked about."""

    def __init__(self, verify_all: bool = False) -> None:
        self.verify_all = verify_all
        self.verify_calls: list[str] = []

    async def complete_json(self, _system: str, prompt: str, *, web_search=False, stage="", **_kw):
        if stage == "web_gap_fill_search":
            return LLMJSONResponse(payload={"claims": [
                {
                    "topic": "company_snapshot",
                    "claim_text": "Example Corp designs enterprise software.",
                    "source_name": "SEC EDGAR",
                    "source_url": "https://www.sec.gov/example-10k",
                    "published_date": "2026-08-01",
                },
                {
                    "topic": "financials",
                    "claim_text": "Example Corp reported $1B revenue.",
                    "source_name": "Random Blog",
                    "source_url": "https://randomblog.example/leak",
                    "published_date": "2026-08-01",
                },
                {
                    "topic": "catalysts",
                    "claim_text": "Example Corp will launch a product in Q4.",
                    "source_name": "PR Newswire",
                    "source_url": "not-a-real-url",
                    "published_date": "2026-09-01",
                },
            ]}, input_tokens=1, output_tokens=1)
        assert stage == "web_gap_fill_verify"
        self.verify_calls.append(prompt)
        verified = self.verify_all or "SEC EDGAR" in prompt
        return LLMJSONResponse(
            payload={"verified": verified, "confirmed_published_date": "2026-08-01"},
            input_tokens=1, output_tokens=1,
        )


async def test_a_malformed_url_never_reaches_verification():
    reader = _reader()
    client = _FakeClient()
    result, items = await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=client,
    )

    # "not-a-real-url" has no http(s) scheme and is dropped before ever
    # reaching verification; the off-allow-list candidate still gets its
    # own independent check rather than being rejected on domain alone.
    assert len(client.verify_calls) == 2
    assert any("no real URL" in r for r in result.rejected_reasons)


async def test_an_off_allowlist_domain_must_pass_its_own_legitimacy_check():
    reader = _reader()
    client = _FakeClient()
    result, items = await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=client,
    )
    # The verify prompt for the off-allow-list candidate must ask the
    # independent verifier to judge source legitimacy itself.
    blog_prompt = next(p for p in client.verify_calls if "Random Blog" in p)
    assert "legitimate" in blog_prompt.lower()
    # Its verifier response (not on the allow-list, and this fake client
    # only confirms "SEC EDGAR" claims) correctly leaves it unconfirmed.
    assert result.claims_verified == 1


async def test_a_curated_domain_still_gets_its_claim_independently_verified():
    # Regression: an earlier version skipped the entire verification call
    # for a curated domain, not just the redundant legitimacy question -
    # weakening the "every claim is independently re-checked" guarantee.
    # The claim itself must still go through a fresh, independent check.
    reader = _reader()
    client = _FakeClient()
    await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=client,
    )
    sec_prompt = next(p for p in client.verify_calls if "SEC EDGAR" in p)
    assert "legitimate" not in sec_prompt.lower()


async def test_verified_candidate_becomes_citable_document_evidence():
    reader = _reader()
    client = _FakeClient()
    result, items = await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=client,
    )

    assert result.claims_verified == 1
    assert len(items) == 1
    item = items[0]
    assert item.source_url == "https://www.sec.gov/example-10k"
    assert item.claim_text == "Example Corp designs enterprise software."
    assert item.category.value == "document"

    reader.store.save(items)
    assert reader.documents()[0].evidence_id == item.evidence_id


async def test_unverified_candidate_is_never_written():
    reader = _reader()
    client = _FakeClient(verify_all=False)

    class _NoConfirmClient(_FakeClient):
        async def complete_json(self, _system, prompt, *, web_search=False, stage="", **kw):
            if stage == "web_gap_fill_verify":
                return LLMJSONResponse(
                    payload={"verified": False}, input_tokens=1, output_tokens=1)
            return await super().complete_json(
                _system, prompt, web_search=web_search, stage=stage, **kw)

    _, items = await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=_NoConfirmClient(),
    )
    assert items == []


async def test_topic_echoed_back_as_a_list_is_still_recognised():
    """Regression test: a model has been observed echoing the schema's own
    enum-hint list back as the value instead of picking one member."""
    reader = _reader()

    class _ListTopicClient:
        async def complete_json(self, _system, prompt, *, web_search=False, stage="", **_kw):
            if stage == "web_gap_fill_search":
                return LLMJSONResponse(payload={"claims": [{
                    "topic": ["company_snapshot"],
                    "claim_text": "Example Corp designs enterprise software.",
                    "source_name": "SEC EDGAR",
                    "source_url": "https://www.sec.gov/example-10k",
                    "published_date": "2026-08-01",
                }]}, input_tokens=1, output_tokens=1)
            return LLMJSONResponse(
                payload={"verified": True, "confirmed_published_date": "2026-08-01"},
                input_tokens=1, output_tokens=1,
            )

    result, items = await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=_ListTopicClient(),
    )
    assert result.claims_proposed == 1
    assert len(items) == 1
    assert items[0].metadata["gap_fill_topic"] == "company_snapshot"


async def test_well_served_report_never_calls_the_model():
    reader = _reader()
    reader.store.save([])  # nothing to seed; simulate a well-served run instead:

    class _ExplodingClient:
        async def complete_json(self, *args, **kwargs):
            raise AssertionError("should not be called when nothing is thin")

    from eq_report.pipeline import web_gap_fill as module
    original = module._thin_topics
    module._thin_topics = lambda _reader: []
    try:
        result, items = await fill_evidence_gaps(
            "run", "Example Corp", "EX", reader, None,
            allowed_domains=_ALLOWED, max_claims=12, client=_ExplodingClient(),
        )
    finally:
        module._thin_topics = original
    assert result.attempted is False
    assert items == []
