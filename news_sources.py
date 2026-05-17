"""News ingestion for Cancún / Riviera Maya / Quintana Roo.

Two sources:
1. Google News RSS — broad, fast, headline + snippet only.
2. Sipse scraper (sipse.com) — for items where we want the full body.

The RSS path is the primary feed. Sipse is fetched on demand once the user
selects a story (saves bandwidth and avoids hammering the diario).
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Google News RSS — geo MX, lang es-419. Original QR-only constant kept for
# back-compat; the M1 multi-query fan-out below covers politics/security/
# corruption/clima/turismo/etc that the old constant filtered out.
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q=Cancun+OR+%22Quintana+Roo%22+OR+%22Riviera+Maya%22+OR+Tulum+OR+%22Playa+del+Carmen%22"
    "&hl=es-419&gl=MX&ceid=MX:es-419"
)

# M1 quick-win: multi-query fan-out. Each query is a topic-focused RSS feed
# URL. Fetched in parallel; results merged + deduped by URL canonical.
# Covers what the single QR-region query was dropping (political, security,
# corruption, weather, tourism, denuncias).
GOOGLE_NEWS_RSS_QUERIES = [
    ("qr_region", "Cancun OR \"Quintana Roo\" OR \"Riviera Maya\" OR Tulum OR \"Playa del Carmen\""),
    ("politica_mx", "política México OR \"gobierno federal\" OR Sheinbaum OR \"Mara Lezama\""),
    ("seguridad", "seguridad Quintana Roo OR \"violencia Cancún\" OR \"FGE Quintana Roo\""),
    ("corrupcion", "corrupción Quintana Roo OR \"denuncia política\" OR \"funcionario Cancún\""),
    ("clima_caribe", "clima Cancún OR \"frente frío Yucatán\" OR \"huracán Caribe\""),
    ("turismo_qr", "turismo Cancún OR \"hoteles Riviera Maya\" OR \"vacaciones Tulum\""),
    ("chetumal_sur", "Chetumal OR \"Othón P Blanco\" OR Bacalar OR \"Felipe Carrillo Puerto\""),
    ("denuncias", "denuncia ciudadana Cancún OR \"queja vecinal Quintana Roo\""),
]


def _build_google_news_url(query: str) -> str:
    from urllib.parse import quote_plus
    return (
        "https://news.google.com/rss/search?q="
        + quote_plus(query)
        + "&hl=es-419&gl=MX&ceid=MX:es-419"
    )


# Publisher RSS feeds — go direct to source instead of through Google News
# aggregator. feedparser handles these identically; just add the URL.
# Active=True ones are tried; failures are logged but don't break the run.
PUBLISHER_RSS_FEEDS = [
    ("Sipse", "https://sipse.com/feed/", "local"),
    ("Por Esto", "https://www.poresto.net/feed/", "local"),
    ("Quadratin QR", "https://quintanaroo.quadratin.com.mx/feed/", "local"),
    ("Noticaribe", "https://noticaribe.com.mx/feed/", "local"),
    ("Animal Político", "https://www.animalpolitico.com/feed", "national"),
    ("Aristegui Noticias", "https://aristeguinoticias.com/feed/", "national"),
    ("Proceso", "https://www.proceso.com.mx/feed/", "national"),
    ("La Jornada", "https://www.jornada.com.mx/rss/edicion.xml", "national"),
]

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


@dataclass
class NewsItem:
    """A single news item, source-agnostic."""
    title: str
    url: str
    source: str               # name of the publishing outlet
    published_at: str         # ISO 8601
    snippet: str = ""         # short summary from RSS
    body: str = ""            # full article body (populated by scraper)
    region_hits: List[str] = field(default_factory=list)
    # Populated by NewsEnrichmentSystem when the item passes through it.
    verified_facts: List[str] = field(default_factory=list)
    source_refs: List[str] = field(default_factory=list)
    enriched_quality_score: Optional[int] = None
    selected_image_urls: List[str] = field(default_factory=list)
    # M2: extended metadata for multi-source pipeline + cross-run dedup.
    detected_at: str = ""             # ISO 8601 — when we first fetched it
    url_hash: str = ""                # md5 of normalized URL — primary dedup key
    category: str = ""                # from brand_style.classify_vertical
    source_region: str = ""           # 'qr' | 'national' | 'mx_southeast' | 'international'
    author: str = ""
    image_url: str = ""               # og:image or feed enclosure

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()
        if not self.url_hash:
            self.url_hash = hashlib.md5(
                (self.url or "").lower().strip().encode("utf-8", "ignore")
            ).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)


# === M2: Source registry + multi-RSS fan-out ===

@dataclass
class Source:
    """A single news source entry loaded from source_registry.json."""
    name: str
    url: str
    type: str                          # local | national | government | independent | aggregator
    region: str                        # qr | national | mx_southeast | international
    method: str                        # rss | sitemap | homepage | api
    priority: str                      # high | medium | low
    active: bool
    feed_url: Optional[str] = None
    requires_region_filter: bool = False
    crawl_delay_seconds: float = 2.0
    notes: str = ""


class SourceRegistry:
    """Loads and manages the list of news sources from JSON.

    Designed to scale from 10 to 100+ sources without code changes — just
    add entries to source_registry.json.
    """

    def __init__(self, sources: List[Source]):
        self.sources = sources

    @classmethod
    def from_file(cls, path: str = "source_registry.json") -> "SourceRegistry":
        p = Path(path)
        if not p.is_absolute():
            # Resolve against the directory containing this module so
            # callers don't have to cwd into the worktree.
            p = Path(__file__).resolve().parent / path
        data = json.loads(p.read_text(encoding="utf-8"))
        sources = [
            Source(
                name=s["name"],
                url=s["url"],
                type=s["type"],
                region=s["region"],
                method=s["method"],
                priority=s["priority"],
                active=s.get("active", False),
                feed_url=s.get("feed_url"),
                requires_region_filter=s.get("requires_region_filter", False),
                crawl_delay_seconds=float(s.get("crawl_delay_seconds", 2.0)),
                notes=s.get("notes", ""),
            )
            for s in data["sources"]
        ]
        return cls(sources)

    def active_rss_sources(self) -> List[Source]:
        return [s for s in self.sources
                if s.active and s.method == "rss" and s.feed_url]

    def by_priority(self, priority: str) -> List[Source]:
        return [s for s in self.sources if s.priority == priority]

    def by_region(self, region: str) -> List[Source]:
        return [s for s in self.sources if s.region == region]


REGION_KEYWORDS = [
    "Cancún", "Cancun",
    "Quintana Roo",
    "Riviera Maya",
    "Tulum",
    "Playa del Carmen",
    "Cozumel",
    "Bacalar",
    "Isla Mujeres",
    "Holbox",
    "Chetumal",
    "Puerto Morelos",
]


def _region_hits(text: str) -> List[str]:
    """Return the list of region keywords mentioned in the given text."""
    if not text:
        return []
    low = text.lower()
    hits = []
    for kw in REGION_KEYWORDS:
        if kw.lower() in low:
            hits.append(kw)
    return hits


def fetch_google_news(
    *,
    since_days: int = 3,
    max_items: int = 200,
    require_region_hit: bool = False,
    extra_feeds: Optional[List[str]] = None,
) -> List[NewsItem]:
    """Fetch and parse Google News RSS for the QR region + topic queries.

    M1 defaults raised vs legacy:
      - since_days 3 (was 2) — broader recency window
      - max_items 200 (was 30) — no artificial ceiling
      - require_region_hit False (was True) — keep national/political stories
        that DON'T mention QR keywords; the post-pipeline scorer demotes them
        appropriately. The old default dropped >70% of political coverage.

    Iterates over GOOGLE_NEWS_RSS_QUERIES (8 topic feeds) and the publisher
    feeds list. Failures per-feed are logged but don't break the run.
    """
    logger.info("📰 Fetching Google News RSS (multi-query)...")

    # Build the list of feed URLs to fetch: legacy QR URL + topic queries +
    # publisher feeds + any caller-provided extras.
    feed_urls: List[str] = [GOOGLE_NEWS_RSS]
    feed_urls.extend(_build_google_news_url(q) for _, q in GOOGLE_NEWS_RSS_QUERIES)
    feed_urls.extend(u for _, u, _ in PUBLISHER_RSS_FEEDS)
    if extra_feeds:
        feed_urls.extend(extra_feeds)
    # Dedupe identical URLs.
    feed_urls = list(dict.fromkeys(feed_urls))

    all_entries = []
    for url in feed_urls:
        try:
            parsed = feedparser.parse(url)
            all_entries.extend(parsed.entries)
        except Exception as e:
            logger.warning(f"feed failed {url[:80]}: {e}")
    # Stash for the loop body below to iterate.
    parsed = type("FauxParsed", (), {"entries": all_entries})()

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    items: List[NewsItem] = []
    seen_urls: set = set()  # M1: dedupe within a single fetch

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        snippet_html = entry.get("summary", "") or entry.get("description", "")
        snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)

        published_struct = entry.get("published_parsed")
        if published_struct:
            published_dt = datetime(*published_struct[:6], tzinfo=timezone.utc)
        else:
            published_dt = datetime.now(timezone.utc)

        if published_dt < cutoff:
            continue

        # Google News titles look like "Foo bar baz - Source Name"
        source_name = ""
        if " - " in title:
            head, _, tail = title.rpartition(" - ")
            if tail and len(tail) < 60:
                title = head.strip()
                source_name = tail.strip()
        if not source_name:
            source_name = urlparse(link).netloc

        text_for_region = f"{title}\n{snippet}"
        hits = _region_hits(text_for_region)
        if require_region_hit and not hits:
            continue

        # M1 dedup: skip URLs we've already added this run (multi-feed
        # overlap is common — same story shows up in qr_region + sipse RSS).
        canon = _canonical_link(link)
        if canon and canon in seen_urls:
            continue
        if canon:
            seen_urls.add(canon)

        items.append(NewsItem(
            title=title,
            url=link,
            source=source_name,
            published_at=published_dt.isoformat(),
            snippet=snippet,
            region_hits=hits,
        ))

        if len(items) >= max_items:
            break

    logger.info(f"📰 {len(items)} news items after filtering (from {len(feed_urls)} feeds)")
    return items


def _canonical_link(url: str) -> str:
    """Strip protocol, www, trailing slash to detect duplicates across feeds."""
    if not url:
        return ""
    try:
        p = urlparse(url)
        host = (p.netloc or "").lower().removeprefix("www.")
        path = p.path.rstrip("/")
        return f"{host}{path}"
    except Exception:
        return url


def filter_by_date(items: List[NewsItem], date_iso: str) -> List[NewsItem]:
    """Keep only items published on the given UTC calendar day (YYYY-MM-DD)."""
    return [i for i in items if i.published_at.startswith(date_iso)]


def scrape_sipse(url: str, timeout: float = 15.0) -> Optional[str]:
    """Fetch the full article body from a sipse.com URL.

    Returns the cleaned plain-text body, or None on any failure. Failures are
    expected (paywalls, layout changes, non-Sipse URLs) and the caller falls
    back to using the RSS snippet as the article body.
    """
    if "sipse.com" not in url:
        return None

    try:
        resp = httpx.get(
            url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.warning(f"sipse fetch failed for {url}: {e}")
        return None

    soup = BeautifulSoup(resp.text, "html.parser")

    # Sipse uses an article tag wrapping the post body. Fall back to the
    # first long <div class*="content"> if the article tag isn't there.
    article = soup.find("article")
    if article is None:
        article = soup.find("div", class_=lambda c: c and "content" in c.lower())
    if article is None:
        return None

    # Drop scripts, ads, social embeds.
    for tag in article(["script", "style", "iframe", "aside", "figure", "ins"]):
        tag.decompose()

    paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
    body = "\n\n".join(p for p in paragraphs if len(p) > 20)
    return body if body else None


def enrich_with_body(item: NewsItem) -> NewsItem:
    """If item.url points at sipse.com, scrape full body. Otherwise leave it.

    Legacy. M4's `extractor.extract_article` should be used for new code
    (works on any domain via trafilatura).
    """
    if item.body:
        return item
    body = scrape_sipse(item.url)
    if body:
        item.body = body
        logger.info(f"📰 Sipse body fetched ({len(body)} chars) for: {item.title[:60]}")
    return item


# === M2: multi-source RSS fetcher ===

def _parse_rss_source(
    source: Source, since_days: int, seen_urls: set,
) -> List[NewsItem]:
    """Fetch and parse a single RSS source. Returns new items not in seen_urls.

    Failures are logged and return [] — never break the run.
    """
    logger.info(f"📰 Fetching RSS: {source.name}")
    try:
        parsed = feedparser.parse(source.feed_url)
    except Exception as e:
        logger.warning(f"RSS fetch failed for {source.name}: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    items: List[NewsItem] = []

    for entry in parsed.entries:
        title = (entry.get("title") or "").strip()
        link = (entry.get("link") or "").strip()
        if not title or not link:
            continue

        canon = _canonical_link(link)
        if canon in seen_urls:
            continue
        seen_urls.add(canon)

        snippet_html = entry.get("summary", "") or entry.get("description", "")
        snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)

        published_struct = entry.get("published_parsed")
        if published_struct:
            published_dt = datetime(*published_struct[:6], tzinfo=timezone.utc)
        else:
            published_dt = datetime.now(timezone.utc)

        if published_dt < cutoff:
            continue

        # Google News titles look like "Foo - Source"; publisher feeds usually don't
        source_name = source.name
        if " - " in title:
            head, _, tail = title.rpartition(" - ")
            if tail and len(tail) < 60:
                title = head.strip()
                source_name = tail.strip()

        # og:image / enclosure from feed when available
        image_url = ""
        for enc in entry.get("enclosures", []) or []:
            if enc.get("type", "").startswith("image"):
                image_url = enc.get("href") or enc.get("url", "")
                break
        if not image_url:
            media = entry.get("media_thumbnail") or entry.get("media_content")
            if media and isinstance(media, list) and media:
                image_url = media[0].get("url", "")

        text_for_region = f"{title}\n{snippet}"
        hits = _region_hits(text_for_region)
        if source.requires_region_filter and not hits:
            continue

        items.append(NewsItem(
            title=title,
            url=link,
            source=source_name,
            published_at=published_dt.isoformat(),
            snippet=snippet,
            region_hits=hits,
            source_region=source.region,
            image_url=image_url,
            author=(entry.get("author") or "").strip()[:80],
        ))

    logger.info(f"📰 {len(items)} new items from {source.name}")
    return items


def fetch_all_rss_sources(
    sources: List[Source],
    *,
    since_days: int = 3,
    max_per_source: int = 50,
    max_total: int = 500,
) -> List[NewsItem]:
    """Fetch every active RSS source in the list and return deduplicated items.

    Synchronous: feedparser is sync-only and the parallelism gain from a
    handful of feeds isn't worth a thread pool — the typical bottleneck is
    the slowest publisher RSS endpoint, not aggregate throughput. If you
    have 50+ sources and notice it's slow, wrap this with `asyncio.to_thread`.
    """
    seen_urls: set = set()
    all_items: List[NewsItem] = []

    for source in sources:
        if len(all_items) >= max_total:
            break
        items = _parse_rss_source(source, since_days, seen_urls)
        all_items.extend(items[:max_per_source])

    logger.info(
        f"📰 Total after multi-source fetch: {len(all_items)} items "
        f"from {len(sources)} sources"
    )
    return all_items
