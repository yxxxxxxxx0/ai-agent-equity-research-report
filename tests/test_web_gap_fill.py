from eq_report.evidence.reader import EvidenceReader
from eq_report.evidence.store import EvidenceStore
from eq_report.llm.client import LLMJSONResponse
from eq_report.pipeline.web_gap_fill import _domain_allowed, fill_evidence_gaps

_ALLOWED = ("sec.gov", "prnewswire.com")


def test_companys_own_domain_is_allowed_without_being_on_the_list():
    # A company's own investor-relations/newsroom page is a legitimate
    # primary source even though no fixed allow-list can name it in advance.
    assert _domain_allowed("https://www.apple.com/newsroom/x", _ALLOWED, "Apple Inc.")
    assert _domain_allowed("https://investor.apple.com/x", _ALLOWED, "Apple Inc.")


def test_a_similarly_named_third_party_site_is_not_mistaken_for_the_company():
    # "appleinsider.com" merely contains the word "apple" - it is not
    # Apple's own domain, and must not be admitted on that basis alone.
    assert not _domain_allowed("https://appleinsider.com/x", _ALLOWED, "Apple Inc.")


def _reader() -> EvidenceReader:
    store = EvidenceStore(":memory:")
    return EvidenceReader(store=store, report_run_id="run", ticker="EX", company="Example Corp")


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


async def test_rejects_non_allowlisted_and_malformed_urls_before_verification():
    reader = _reader()
    client = _FakeClient()
    result, items = await fill_evidence_gaps(
        "run", "Example Corp", "EX", reader, None,
        allowed_domains=_ALLOWED, max_claims=12, client=client,
    )

    # Only the sec.gov candidate ever reaches the second, independent check.
    assert len(client.verify_calls) == 1
    assert "SEC EDGAR" in client.verify_calls[0]
    assert result.claims_proposed == 1
    assert any("not an allow-listed domain" in r for r in result.rejected_reasons)
    assert any("no real URL" in r for r in result.rejected_reasons)


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
