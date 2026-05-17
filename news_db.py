"""SQLite persistence layer for the news engine.

Stores articles, URL hashes for deduplication, and fetch logs.
All DB access goes through `NewsDB` — no raw sqlite3 outside this module.

SQLite was chosen over Postgres because:
  - Zero infra: no service to manage, no auth, no migrations system
  - Handles tens of thousands of articles per day without issue
  - File-local — easy to back up (just copy news.db), easy to inspect with
    `sqlite3 news.db` from the terminal
  - Single-writer is fine for our use case (one webapp + one cron job)

If we ever need multi-writer or concurrent reads at scale, the schema
translates trivially to Postgres via SQLAlchemy.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ArticleRecord:
    url: str
    url_hash: str
    title: str
    source: str
    published_at: str
    detected_at: str
    snippet: str = ""
    body: str = ""
    category: str = ""
    source_region: str = ""
    author: str = ""
    image_url: str = ""
    score: float = 0.0


class NewsDB:
    """Thin wrapper around sqlite3 with the schema this project needs.

    `check_same_thread=False` so the FastAPI worker can share one connection
    across threads — SQLite serializes its own writes so this is safe.
    """

    def __init__(self, db_path: str = "news.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript(
            """
            CREATE TABLE IF NOT EXISTS articles (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                url          TEXT NOT NULL UNIQUE,
                url_hash     TEXT NOT NULL,
                title        TEXT NOT NULL,
                source       TEXT NOT NULL,
                published_at TEXT NOT NULL,
                detected_at  TEXT NOT NULL,
                snippet      TEXT DEFAULT '',
                body         TEXT DEFAULT '',
                category     TEXT DEFAULT '',
                source_region TEXT DEFAULT '',
                author       TEXT DEFAULT '',
                image_url    TEXT DEFAULT '',
                score        REAL DEFAULT 0.0
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_articles_url_hash
                ON articles(url_hash);
            CREATE INDEX IF NOT EXISTS idx_articles_detected_at
                ON articles(detected_at);
            CREATE INDEX IF NOT EXISTS idx_articles_source
                ON articles(source);

            CREATE TABLE IF NOT EXISTS url_hashes (
                url_hash TEXT PRIMARY KEY,
                url      TEXT NOT NULL,
                seen_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS fetch_logs (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                source_name  TEXT NOT NULL,
                feed_url     TEXT NOT NULL,
                fetched_at   TEXT NOT NULL DEFAULT (datetime('now')),
                status       TEXT NOT NULL,
                items_found  INTEGER DEFAULT 0,
                items_new    INTEGER DEFAULT 0,
                error_msg    TEXT DEFAULT ''
            );
            CREATE INDEX IF NOT EXISTS idx_fetch_logs_fetched_at
                ON fetch_logs(fetched_at);
            """
        )
        self._conn.commit()

    def list_tables(self) -> List[str]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cur.fetchall()]

    def insert_article(self, rec: ArticleRecord) -> None:
        """INSERT OR IGNORE — duplicates (by url UNIQUE) silently skipped."""
        try:
            self._conn.execute(
                """INSERT OR IGNORE INTO articles
                   (url, url_hash, title, source, published_at, detected_at,
                    snippet, body, category, source_region, author, image_url, score)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (rec.url, rec.url_hash, rec.title, rec.source,
                 rec.published_at, rec.detected_at, rec.snippet, rec.body,
                 rec.category, rec.source_region, rec.author, rec.image_url, rec.score),
            )
            self.mark_url_seen(rec.url, rec.url_hash)
            self._conn.commit()
        except sqlite3.Error:
            pass

    def insert_articles(self, recs: List[ArticleRecord]) -> int:
        """Bulk insert. Returns count of rows that actually landed."""
        before = self.count_articles()
        for rec in recs:
            self.insert_article(rec)
        return self.count_articles() - before

    def get_article_by_hash(self, url_hash: str) -> Optional[ArticleRecord]:
        cur = self._conn.execute(
            "SELECT * FROM articles WHERE url_hash = ?", (url_hash,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        keys = row.keys()
        return ArticleRecord(**{k: row[k] for k in keys if k != "id"})

    def is_known_url(self, url_hash: str) -> bool:
        cur = self._conn.execute(
            "SELECT 1 FROM url_hashes WHERE url_hash = ?", (url_hash,)
        )
        return cur.fetchone() is not None

    def mark_url_seen(self, url: str, url_hash: str) -> None:
        self._conn.execute(
            """INSERT OR IGNORE INTO url_hashes (url_hash, url, seen_at)
               VALUES (?, ?, datetime('now'))""",
            (url_hash, url),
        )
        self._conn.commit()

    def count_articles(self) -> int:
        return self._conn.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]

    def get_recent_articles(self, limit: int = 50) -> List[ArticleRecord]:
        cur = self._conn.execute(
            "SELECT * FROM articles ORDER BY detected_at DESC LIMIT ?",
            (limit,),
        )
        out = []
        for row in cur.fetchall():
            keys = row.keys()
            out.append(ArticleRecord(**{k: row[k] for k in keys if k != "id"}))
        return out

    def log_fetch(
        self,
        source_name: str,
        feed_url: str,
        status: str,
        items_found: int,
        items_new: int,
        error_msg: str = "",
    ) -> None:
        self._conn.execute(
            """INSERT INTO fetch_logs
               (source_name, feed_url, status, items_found, items_new, error_msg)
               VALUES (?,?,?,?,?,?)""",
            (source_name, feed_url, status, items_found, items_new, error_msg),
        )
        self._conn.commit()

    def recent_fetch_logs(self, limit: int = 20) -> List[dict]:
        cur = self._conn.execute(
            "SELECT * FROM fetch_logs ORDER BY fetched_at DESC LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cur.fetchall()]

    def stats(self) -> dict:
        """Quick summary for /admin or /health."""
        articles = self.count_articles()
        recent_24h = self._conn.execute(
            "SELECT COUNT(*) FROM articles WHERE detected_at >= datetime('now', '-1 day')"
        ).fetchone()[0]
        sources_count = self._conn.execute(
            "SELECT COUNT(DISTINCT source) FROM articles"
        ).fetchone()[0]
        recent_errors = self._conn.execute(
            "SELECT COUNT(*) FROM fetch_logs WHERE status != 'ok' AND fetched_at >= datetime('now', '-1 day')"
        ).fetchone()[0]
        return {
            "articles_total": articles,
            "articles_last_24h": recent_24h,
            "distinct_sources": sources_count,
            "fetch_errors_last_24h": recent_errors,
        }

    def close(self) -> None:
        self._conn.close()
