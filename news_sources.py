"""News ingestion for Cancún / Riviera Maya / Quintana Roo.

Two sources:
1. Google News RSS — broad, fast, headline + snippet only.
2. Sipse scraper (sipse.com) — for items where we want the full body.

The RSS path is the primary feed. Sipse is fetched on demand once the user
selects a story (saves bandwidth and avoids hammering the diario).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from typing import List, Optional
from urllib.parse import urlparse

import feedparser
import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# Google News RSS — geo MX, lang es-419, filtered for Quintana Roo terms.
GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q=Cancun+OR+%22Quintana+Roo%22+OR+%22Riviera+Maya%22+OR+Tulum+OR+%22Playa+del+Carmen%22"
    "&hl=es-419&gl=MX&ceid=MX:es-419"
)

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

    def to_dict(self) -> dict:
        return asdict(self)


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
    since_days: int = 2,
    max_items: int = 30,
    require_region_hit: bool = True,
) -> List[NewsItem]:
    """Fetch and parse Google News RSS for the QR region.

    since_days: drop items older than this many days.
    max_items: cap on returned items (post-filter).
    require_region_hit: drop items whose title+snippet doesn't mention any
        region keyword. Useful because Google News pads results with national
        news that matched on a different term.
    """
    logger.info("📰 Fetching Google News RSS...")
    parsed = feedparser.parse(GOOGLE_NEWS_RSS)

    cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
    items: List[NewsItem] = []

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

    logger.info(f"📰 {len(items)} news items after filtering")
    return items


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
    """If item.url points at sipse.com, scrape full body. Otherwise leave it."""
    if item.body:
        return item
    body = scrape_sipse(item.url)
    if body:
        item.body = body
        logger.info(f"📰 Sipse body fetched ({len(body)} chars) for: {item.title[:60]}")
    return item
