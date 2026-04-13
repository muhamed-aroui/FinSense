"""Unit tests for finsense.data.preprocessing."""

from finsense.data.preprocessing import preprocess


def test_strip_single_url():
    text = "Stock rises https://t.co/abc123 today"
    assert preprocess(text) == "Stock rises today"


def test_strip_multiple_urls():
    text = "See https://t.co/a and http://example.com/b for details"
    assert preprocess(text) == "See and for details"


def test_no_url():
    text = "$AAPL is bullish"
    assert preprocess(text) == "$AAPL is bullish"


def test_collapse_whitespace():
    text = "hello   world  \t here"
    assert preprocess(text) == "hello world here"


def test_strip_url_and_collapse():
    text = "before  https://t.co/x  after"
    assert preprocess(text) == "before after"


def test_preserves_cashtags():
    text = "$AAPL $TSLA going up"
    assert preprocess(text) == "$AAPL $TSLA going up"


def test_preserves_hashtags():
    text = "#marketscreener #stock news"
    assert preprocess(text) == "#marketscreener #stock news"


def test_preserves_case():
    text = "Apple AAPL BuLlIsH"
    assert preprocess(text) == "Apple AAPL BuLlIsH"


def test_empty_string():
    assert preprocess("") == ""


def test_url_only():
    assert preprocess("https://t.co/abc123") == ""
