"""Tests for M5 deduplicator (URL + title hash, in-session dedup)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from deduplicator import Deduplicator, title_hash, url_hash, _normalize  # noqa: E402
from news_sources import NewsItem  # noqa: E402


def _item(title, url, source="src"):
    return NewsItem(
        title=title, url=url, source=source,
        published_at="2026-05-17T10:00:00+00:00",
    )


# ---------- hashing primitives ----------

def test_normalize_strips_diacritics_and_lowercase():
    assert _normalize("  Cancún ") == "cancun"


def test_url_hash_normalizes_case():
    assert url_hash("HTTPS://x.com/A") == url_hash("https://x.com/a")


def test_title_hash_is_deterministic():
    assert title_hash("Balacera en Cancún", "Sipse") == title_hash("Balacera en Cancún", "Sipse")


def test_title_hash_different_sources_differ():
    assert title_hash("Same Headline", "A") != title_hash("Same Headline", "B")


def test_title_hash_normalizes_case_and_whitespace_and_diacritics():
    a = title_hash("  BALACERA EN CANCÚN  ", "Sipse")
    b = title_hash("balacera en cancun", "sipse")
    assert a == b


# ---------- Deduplicator ----------

def test_dedup_rejects_seen_url():
    d = Deduplicator()
    it = _item("Story", "https://example.com/1")
    assert d.is_new(it) is True
    assert d.is_new(it) is False


def test_dedup_rejects_same_title_different_url():
    d = Deduplicator()
    a = _item("Balacera en Cancún", "https://sipse.com/1", source="Sipse")
    b = _item("Balacera en Cancún", "https://google.com/news/abc", source="Sipse")
    d.is_new(a)
    assert d.is_new(b) is False


def test_dedup_allows_different_stories():
    d = Deduplicator()
    assert d.is_new(_item("A", "https://x.com/a")) is True
    assert d.is_new(_item("B", "https://x.com/b")) is True


def test_filter_new_removes_duplicates_in_order():
    d = Deduplicator()
    items = [
        _item("A", "https://x.com/a"),
        _item("A", "https://x.com/a"),     # exact dup
        _item("B", "https://x.com/b"),
        _item("A", "https://x.com/a2", "src"),  # title-source dup
    ]
    out = d.filter_new(items)
    assert len(out) == 2
    assert [i.title for i in out] == ["A", "B"]


def test_dedup_stats():
    d = Deduplicator()
    d.is_new(_item("A", "https://x.com/a"))
    d.is_new(_item("B", "https://x.com/b"))
    s = d.stats()
    assert s["unique_urls_seen"] == 2
    assert s["unique_title_keys_seen"] == 2
