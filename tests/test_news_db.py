"""Tests for M3 news_db SQLite layer."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_db import NewsDB, ArticleRecord  # noqa: E402


@pytest.fixture
def tmp_db(tmp_path):
    db = NewsDB(str(tmp_path / "test.db"))
    db.init_schema()
    yield db
    db.close()


def _rec(url="https://example.com/news/1", url_hash="abc123", **kw):
    base = dict(
        url=url, url_hash=url_hash,
        title="Test", source="S",
        published_at="2026-05-17T10:00:00+00:00",
        detected_at="2026-05-17T11:00:00+00:00",
    )
    base.update(kw)
    return ArticleRecord(**base)


def test_init_creates_tables(tmp_db):
    tables = tmp_db.list_tables()
    for t in ("articles", "fetch_logs", "url_hashes"):
        assert t in tables, f"missing table {t}"


def test_insert_article_and_retrieve(tmp_db):
    tmp_db.insert_article(_rec(title="Hola"))
    found = tmp_db.get_article_by_hash("abc123")
    assert found is not None
    assert found.title == "Hola"


def test_insert_duplicate_is_ignored(tmp_db):
    tmp_db.insert_article(_rec())
    tmp_db.insert_article(_rec())
    assert tmp_db.count_articles() == 1


def test_is_known_url_after_insert(tmp_db):
    tmp_db.insert_article(_rec(url="https://x.com/1", url_hash="def456"))
    assert tmp_db.is_known_url("def456") is True
    assert tmp_db.is_known_url("notexist") is False


def test_mark_url_seen_without_insert(tmp_db):
    tmp_db.mark_url_seen("https://x.com/2", "ghi789")
    assert tmp_db.is_known_url("ghi789") is True


def test_log_fetch(tmp_db):
    tmp_db.log_fetch("S", "https://x.com/feed", "ok", 10, 3)
    logs = tmp_db.recent_fetch_logs(limit=5)
    assert len(logs) == 1
    assert logs[0]["source_name"] == "S"
    assert logs[0]["items_new"] == 3


def test_get_recent_articles(tmp_db):
    for i in range(5):
        tmp_db.insert_article(_rec(
            url=f"https://x.com/{i}", url_hash=f"h{i}", title=f"S{i}",
        ))
    assert len(tmp_db.get_recent_articles(limit=3)) == 3
    assert tmp_db.count_articles() == 5


def test_stats(tmp_db):
    for i in range(3):
        tmp_db.insert_article(_rec(
            url=f"https://x.com/{i}", url_hash=f"h{i}", source=f"S{i}",
        ))
    s = tmp_db.stats()
    assert s["articles_total"] == 3
    assert s["distinct_sources"] == 3


def test_insert_articles_bulk(tmp_db):
    recs = [_rec(url=f"https://x.com/{i}", url_hash=f"h{i}") for i in range(4)]
    landed = tmp_db.insert_articles(recs)
    assert landed == 4
    # Re-insert same → 0 new
    assert tmp_db.insert_articles(recs) == 0
