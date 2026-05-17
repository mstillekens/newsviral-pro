"""Tests for M2 SourceRegistry + multi-source RSS fetcher."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_sources import (  # noqa: E402
    NewsItem, Source, SourceRegistry, fetch_all_rss_sources,
    _canonical_link,
)


REQUIRED_FIELDS = {"name", "url", "type", "region", "method", "priority", "active"}
VALID_METHODS = {"rss", "sitemap", "homepage", "api"}
VALID_TYPES = {"local", "national", "government", "independent", "aggregator"}
VALID_REGIONS = {"qr", "national", "mx_southeast", "international"}
VALID_PRIORITIES = {"high", "medium", "low"}


# ---------- schema ----------

def _load_raw_registry() -> dict:
    return json.loads((ROOT / "source_registry.json").read_text())


def test_registry_has_at_least_50_sources():
    reg = _load_raw_registry()
    assert len(reg["sources"]) >= 50


def test_every_source_has_required_fields():
    reg = _load_raw_registry()
    for src in reg["sources"]:
        missing = REQUIRED_FIELDS - set(src.keys())
        assert not missing, f"{src.get('name')} missing: {missing}"


def test_every_active_rss_source_has_feed_url():
    reg = _load_raw_registry()
    for src in reg["sources"]:
        if src.get("active") and src.get("method") == "rss":
            assert src.get("feed_url"), f"{src['name']} RSS without feed_url"


def test_valid_method_values():
    reg = _load_raw_registry()
    for src in reg["sources"]:
        assert src["method"] in VALID_METHODS, f"{src['name']} bad method {src['method']}"


def test_valid_priority_values():
    reg = _load_raw_registry()
    for src in reg["sources"]:
        assert src["priority"] in VALID_PRIORITIES


# ---------- SourceRegistry class ----------

def test_source_registry_loads_active_rss_sources():
    reg = SourceRegistry.from_file()
    active_rss = reg.active_rss_sources()
    assert len(active_rss) >= 10


def test_source_dataclass_has_required_attributes():
    reg = SourceRegistry.from_file()
    src = reg.sources[0]
    for attr in ("name", "feed_url", "region", "priority",
                 "requires_region_filter", "crawl_delay_seconds"):
        assert hasattr(src, attr), f"missing attr {attr}"


def test_source_registry_by_region():
    reg = SourceRegistry.from_file()
    qr = reg.by_region("qr")
    nat = reg.by_region("national")
    assert len(qr) > 0
    assert len(nat) > 0


# ---------- NewsItem M2 fields ----------

def test_news_item_auto_fills_detected_at_and_url_hash():
    item = NewsItem(
        title="Test", url="https://example.com/news/1",
        source="Test", published_at="2026-05-17T10:00:00+00:00",
    )
    assert item.detected_at, "detected_at should auto-populate"
    assert item.url_hash, "url_hash should auto-populate"
    assert len(item.url_hash) == 32, "md5 hex is 32 chars"


def test_news_item_url_hash_is_deterministic():
    a = NewsItem(title="a", url="HTTPS://Example.com/1", source="s",
                 published_at="2026-05-17T10:00:00+00:00")
    b = NewsItem(title="b", url="https://example.com/1", source="s",
                 published_at="2026-05-17T10:00:00+00:00")
    assert a.url_hash == b.url_hash, "case + scheme normalized"


# ---------- fetch_all_rss_sources ----------

def test_fetch_all_rss_sources_deduplicates_by_canonical_url():
    """fetch_all_rss_sources must not return two items with the same URL."""
    fake_entry = MagicMock(
        title="Duplicate Story - Source A",
        link="https://example.com/story/1",
        summary="Same story",
        published_parsed=(2026, 5, 17, 10, 0, 0, 0, 0, 0),
        enclosures=[], media_thumbnail=None, media_content=None,
    )
    fake_entry.get = lambda k, default=None: {
        "title": "Duplicate Story - Source A",
        "link": "https://example.com/story/1",
        "summary": "Same story",
        "published_parsed": (2026, 5, 17, 10, 0, 0, 0, 0, 0),
        "enclosures": [],
        "media_thumbnail": None, "media_content": None,
        "author": "",
    }.get(k, default)

    with patch("news_sources.feedparser.parse") as mock_parse:
        mock_parse.return_value = MagicMock(entries=[fake_entry])
        sources = [
            Source(name="A", url="https://a.com", type="local", region="qr",
                   method="rss", priority="high", active=True,
                   feed_url="https://a.com/feed"),
            Source(name="B", url="https://b.com", type="local", region="qr",
                   method="rss", priority="high", active=True,
                   feed_url="https://b.com/feed"),
        ]
        items = fetch_all_rss_sources(sources, since_days=7)
        urls = [i.url for i in items]
        assert len(urls) == len(set(urls)), f"duplicates found: {urls}"


def test_fetch_all_rss_sources_logs_failures_without_breaking():
    """Failed sources are logged; following sources still get fetched."""
    fake_good = MagicMock()
    fake_good.get = lambda k, default=None: {
        "title": "Good story",
        "link": "https://b.com/story/1",
        "summary": "x",
        "published_parsed": (2026, 5, 17, 10, 0, 0, 0, 0, 0),
        "enclosures": [], "media_thumbnail": None, "media_content": None,
        "author": "",
    }.get(k, default)

    def side(url):
        if "a.com" in url:
            raise Exception("network blip")
        return MagicMock(entries=[fake_good])

    with patch("news_sources.feedparser.parse", side_effect=side):
        sources = [
            Source(name="A", url="https://a.com", type="local", region="qr",
                   method="rss", priority="high", active=True,
                   feed_url="https://a.com/feed"),
            Source(name="B", url="https://b.com", type="local", region="qr",
                   method="rss", priority="high", active=True,
                   feed_url="https://b.com/feed"),
        ]
        items = fetch_all_rss_sources(sources, since_days=7)
        assert len(items) == 1, "B should still succeed despite A's error"


# ---------- canonical link ----------

def test_canonical_link_strips_www_and_scheme():
    assert _canonical_link("https://www.example.com/a/") == "example.com/a"
    assert _canonical_link("HTTP://Example.com/a") == "example.com/a"
