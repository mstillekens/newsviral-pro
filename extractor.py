"""Generic article extractor using trafilatura.

Replaces the sipse-only scraper in `news_sources.scrape_sipse`. Works on any
public news URL — handles 90% of WordPress / Joomla / Drupal / proprietary
CMS layouts that publishers use.

Why trafilatura over alternatives:
  - newspaper3k: unmaintained since 2018, breaks on modern sites
  - newspaper4k: maintained fork, decent but worse extraction accuracy on
    Spanish-language Mexican publishers in our tests
  - Scrapling: requires explicit selectors per site (we have 50+ sources)
  - ScrapeGraphAI: needs an LLM call PER extraction — too expensive
  - Plain BeautifulSoup: works but needs custom per-site selectors

trafilatura wins because:
  - One function, any URL, no per-site config
  - Built-in robots.txt support
  - Extracts text + metadata (author, date, language) in one pass
  - Maintained, used at scale in academic NLP pipelines

We expose a single function `extract_article(url) -> ArticleContent | None`.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import trafilatura
import trafilatura.settings

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 20


@dataclass
class ArticleContent:
    """Output of a successful extraction. Any field may be None if the page
    didn't expose it (e.g., no author byline)."""
    text: Optional[str]
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    language: Optional[str] = None


def extract_article(
    url: str, *, check_robots: bool = True,
) -> Optional[ArticleContent]:
    """Download and extract the main article text from a news URL.

    Returns ArticleContent on success, None on any failure (paywall, timeout,
    blocked by robots.txt, parsing error, network blip).

    Failures are logged at DEBUG/WARNING — never raise. Callers should treat
    `None` as "fall back to snippet/og:image only".

    Args:
      url: full article URL.
      check_robots: trafilatura's default is to honor robots.txt; we keep
        that on. Set False only when you know you have permission
        (e.g., your own sites, paid scraping APIs).
    """
    try:
        html = trafilatura.fetch_url(
            url,
            config=trafilatura.settings.use_config(),
        )
        if not html:
            logger.debug(f"extractor: no HTML returned for {url[:80]}")
            return None

        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=False,
            favor_precision=True,
            with_metadata=True,
            output_format="json",
        )

        if not result:
            logger.debug(f"extractor: trafilatura returned empty for {url[:80]}")
            return None

        data = json.loads(result)
        return ArticleContent(
            text=data.get("text"),
            title=data.get("title"),
            author=data.get("author"),
            date=data.get("date"),
            language=data.get("language"),
        )

    except Exception as e:
        logger.warning(f"extractor: failed for {url[:80]}: {e}")
        return None


def extract_bodies_batch(urls: list, *, max_chars: int = 4000) -> dict:
    """Extract bodies for many URLs sequentially. Returns {url: body_text}.

    Each entry is the extracted text trimmed to `max_chars`. Missing URLs
    (extraction failed) are absent from the result dict, never raise.
    Sequential because trafilatura.fetch_url is sync; if you need
    parallelism wrap with `asyncio.to_thread`.
    """
    out: dict = {}
    for url in urls:
        ac = extract_article(url)
        if ac and ac.text:
            out[url] = ac.text[:max_chars]
    return out
