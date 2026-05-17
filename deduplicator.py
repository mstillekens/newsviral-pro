"""In-memory deduplication for the news pipeline.

Two complementary strategies:

  1. **URL hash** — MD5 of normalized URL. Catches the same article URL
     showing up in multiple feeds (very common: a story appears in
     Google News, the publisher's own RSS, and an aggregator simultaneously).

  2. **Title hash** — MD5 of normalized (title + source). Catches the same
     story REPUBLISHED at a different URL — common with Google News
     wrapping publisher articles in their own URLs, or sites that move
     articles between sections.

For CROSS-RUN persistence, check `NewsDB.is_known_url(item.url_hash)` BEFORE
instantiating `Deduplicator` — that catches articles seen in previous
pipeline runs that aren't in this session's in-memory sets.

Pipeline integration:

    db = NewsDB()
    dedup = Deduplicator()
    fresh: list[NewsItem] = []
    for item in fetched_items:
        if db.is_known_url(item.url_hash):
            continue   # cross-run duplicate
        if not dedup.is_new(item):
            continue   # in-session duplicate
        fresh.append(item)
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from typing import List, Set

from news_sources import NewsItem


def _normalize(text: str) -> str:
    """Lowercase, strip diacritics, collapse whitespace."""
    n = unicodedata.normalize("NFKD", text or "")
    n = "".join(c for c in n if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", n.strip().lower())


def url_hash(url: str) -> str:
    return hashlib.md5(_normalize(url).encode("utf-8", "ignore")).hexdigest()


def title_hash(title: str, source: str) -> str:
    key = f"{_normalize(title)}|{_normalize(source)}"
    return hashlib.md5(key.encode("utf-8", "ignore")).hexdigest()


class Deduplicator:
    """Session-scoped dedup. Instantiate once per pipeline run.

    Stateful: each call to is_new() also marks the item as seen, so the
    next identical item returns False. Idempotent within a session.
    """

    def __init__(self):
        self._seen_urls: Set[str] = set()
        self._seen_titles: Set[str] = set()

    def is_new(self, item: NewsItem) -> bool:
        """Return True if this item has not been seen this session.

        Side effect: marks the item as seen on first True return.
        """
        uh = url_hash(item.url)
        if uh in self._seen_urls:
            return False

        th = title_hash(item.title, item.source)
        if th in self._seen_titles:
            return False

        self._seen_urls.add(uh)
        self._seen_titles.add(th)
        return True

    def filter_new(self, items: List[NewsItem]) -> List[NewsItem]:
        """Return items in original order with duplicates removed."""
        return [item for item in items if self.is_new(item)]

    def stats(self) -> dict:
        return {
            "unique_urls_seen": len(self._seen_urls),
            "unique_title_keys_seen": len(self._seen_titles),
        }
