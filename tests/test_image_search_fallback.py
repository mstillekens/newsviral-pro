"""Image search falls through cleanly when DuckDuckGo is unreachable."""
from unittest.mock import patch

import pytest

from image_search import search_news_image


def test_returns_first_url_on_success():
    fake_results = [
        {"image": "https://example.com/a.jpg", "title": "A"},
        {"image": "https://example.com/b.jpg", "title": "B"},
    ]
    with patch("image_search.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.images.return_value = iter(fake_results)
        url = search_news_image("AMLO conferencia Cancún")
    assert url == "https://example.com/a.jpg"


def test_returns_none_on_no_results():
    with patch("image_search.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.images.return_value = iter([])
        url = search_news_image("nada que se encuentre")
    assert url is None


def test_returns_none_when_ddg_raises():
    with patch("image_search.DDGS", side_effect=Exception("network down")):
        url = search_news_image("AMLO")
    assert url is None


def test_skips_obvious_junk_extensions():
    fake_results = [
        {"image": "https://example.com/icon.svg", "title": "junk"},
        {"image": "https://example.com/real.jpg", "title": "ok"},
    ]
    with patch("image_search.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.images.return_value = iter(fake_results)
        url = search_news_image("query")
    assert url == "https://example.com/real.jpg"
