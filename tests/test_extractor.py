"""Tests for M4 trafilatura extractor."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import extractor  # noqa: E402
from extractor import extract_article, ArticleContent  # noqa: E402


def test_extract_article_returns_content_on_success():
    with patch("extractor.trafilatura.fetch_url", return_value="<html><body>x</body></html>"):
        with patch("extractor.trafilatura.extract") as mock_ex:
            mock_ex.return_value = (
                '{"text": "This is the article body.", '
                '"title": "Breaking News", "author": "Reporter X", '
                '"date": "2026-05-17", "language": "es"}'
            )
            result = extract_article("https://example.com/news/1")
    assert isinstance(result, ArticleContent)
    assert result.text == "This is the article body."
    assert result.title == "Breaking News"
    assert result.author == "Reporter X"
    assert result.language == "es"


def test_extract_article_returns_none_on_no_html():
    with patch("extractor.trafilatura.fetch_url", return_value=None):
        assert extract_article("https://example.com/404") is None


def test_extract_article_returns_none_on_empty_extract():
    with patch("extractor.trafilatura.fetch_url", return_value="<html></html>"):
        with patch("extractor.trafilatura.extract", return_value=None):
            assert extract_article("https://x.com/a") is None


def test_extract_article_non_sipse_url_works():
    """Verify the old sipse domain-lock is gone — any URL is acceptable."""
    with patch("extractor.trafilatura.fetch_url", return_value="<html><p>X</p></html>"):
        with patch("extractor.trafilatura.extract") as mock_ex:
            mock_ex.return_value = '{"text": "Content", "title": "T"}'
            r = extract_article("https://animalpolitico.com/story/1")
    assert r is not None
    assert r.text == "Content"


def test_extract_article_handles_exceptions_gracefully():
    with patch("extractor.trafilatura.fetch_url", side_effect=RuntimeError("network")):
        assert extract_article("https://x.com/a") is None


def test_fetch_timeout_constant_is_sane():
    assert extractor.FETCH_TIMEOUT_SECONDS > 0
    assert extractor.FETCH_TIMEOUT_SECONDS <= 30


def test_extract_bodies_batch_skips_failures():
    with patch("extractor.trafilatura.fetch_url") as mock_fetch:
        with patch("extractor.trafilatura.extract") as mock_ex:
            def side(url, **kw):
                return "<html>x</html>" if "good" in url else None
            mock_fetch.side_effect = side
            mock_ex.return_value = '{"text": "body"}'
            r = extractor.extract_bodies_batch([
                "https://good.com/1", "https://bad.com/1", "https://good.com/2",
            ])
    assert "https://good.com/1" in r
    assert "https://good.com/2" in r
    assert "https://bad.com/1" not in r
