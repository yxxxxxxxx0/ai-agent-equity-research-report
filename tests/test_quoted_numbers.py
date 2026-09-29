from types import SimpleNamespace

from eq_report.qa.checks import _numbers_quoted_from_sources


def _passage(text: str, status: str = "validated"):
    return SimpleNamespace(value=None, status=SimpleNamespace(value=status), claim_text=text)


SOURCE = ("As of August 21, 2026, Zacks: Apple trades at 33.54x forward 12-month earnings "
          "versus 21.05x for the sector. The iPhone Duo lists at 15,999 yuan.")


def test_figure_quoted_verbatim_from_a_validated_passage_is_accepted():
    text = "Zacks cited Apple at 33.54x forward earnings versus 21.05x for the sector."
    assert _numbers_quoted_from_sources(text, [_passage(SOURCE)])


def test_thousands_separator_does_not_matter():
    assert _numbers_quoted_from_sources("The Duo lists at 15999 yuan.", [_passage(SOURCE)])


def test_a_derived_figure_is_rejected():
    text = "Apple's multiple was 59% above the sector's 21.05x."
    assert not _numbers_quoted_from_sources(text, [_passage(SOURCE)])


def test_unverified_passage_does_not_count():
    text = "Zacks cited Apple at 33.54x forward earnings."
    assert not _numbers_quoted_from_sources(text, [_passage(SOURCE, status="unverified")])


def test_a_digit_inside_a_larger_number_does_not_match():
    assert not _numbers_quoted_from_sources("The ratio was 3.54x.", [_passage(SOURCE)])


def test_model_booleans_are_read_strictly():
    from eq_report.llm.verify import is_true
    assert is_true(True) and is_true("true") and is_true(" True ")
    assert not is_true(False) and not is_true("false") and not is_true(None) and not is_true("yes")
