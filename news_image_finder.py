"""Find a reference image for a news item.

Strategy (in priority order):

1. Open Graph: scrape the article URL, look for <meta property="og:image">.
   That's the image the publisher curated as the article's hero.
2. Twitter Card: fall back to <meta name="twitter:image">.
3. Article scrape: first <img> tag inside an <article> element bigger than
   a tiny thumbnail (heuristic: width or height attribute > 400 or src
   contains keywords like "hero", "lead", "main").

If none found, return None and the caller falls back to text-only FLUX.

This module is used to provide a `reference_image_url` to the orchestrator,
which then routes that scene through flux-canny-pro instead of flux-pro so
the caricature preserves the facial structure / composition of the source.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse, urljoin

import httpx
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 "
    "(KHTML, like Gecko) Version/17.0 Safari/605.1.15"
)


@dataclass
class ReferenceImage:
    """A reference image we can pass to flux-canny-pro as control_image."""
    url: str
    source: str       # "og:image" | "twitter:image" | "article-img"
    article_url: str  # where we found it


def _absolutize(base: str, url: str) -> str:
    """Resolve a possibly-relative image URL against the article URL."""
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    if url.startswith("//"):
        parsed = urlparse(base)
        return f"{parsed.scheme}:{url}"
    return urljoin(base, url)


def find_reference_image(article_url: str, *, timeout: float = 12.0) -> Optional[ReferenceImage]:
    """Scrape article_url and return the best reference image we can find.

    Returns None on any failure (404, paywall, no image, blocked, etc.).
    Failures are quiet — the caller is expected to fall back to text-only
    generation, never raise.
    """
    if not article_url:
        return None

    # Google News RSS gives us aggregator URLs that redirect. httpx follows.
    try:
        resp = httpx.get(
            article_url,
            timeout=timeout,
            headers={"User-Agent": USER_AGENT, "Accept-Language": "es-MX,es;q=0.9,en;q=0.5"},
            follow_redirects=True,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.info(f"ref-image fetch failed for {article_url[:80]}: {e}")
        return None

    final_url = str(resp.url)  # after redirects
    soup = BeautifulSoup(resp.text, "html.parser")

    # 1. og:image
    og = soup.find("meta", attrs={"property": "og:image"})
    if og and og.get("content"):
        url = _absolutize(final_url, og["content"])
        if url:
            return ReferenceImage(url=url, source="og:image", article_url=final_url)

    # 2. twitter:image (some sites use 'name', some use 'property')
    for selector in [{"name": "twitter:image"}, {"property": "twitter:image"}]:
        tw = soup.find("meta", attrs=selector)
        if tw and tw.get("content"):
            url = _absolutize(final_url, tw["content"])
            if url:
                return ReferenceImage(url=url, source="twitter:image", article_url=final_url)

    # 3. First sizeable <img> in <article>
    article = soup.find("article") or soup
    for img in article.find_all("img"):
        src = img.get("src") or img.get("data-src") or ""
        if not src:
            continue
        # Skip obvious avatars, ads, and trackers.
        bad_hints = ("avatar", "tracker", "pixel", "logo-", "/logo.", "icon-", "favicon", "1x1")
        if any(h in src.lower() for h in bad_hints):
            continue
        # Prefer images with a known size attribute that's reasonably large.
        w = _safe_int(img.get("width"))
        h = _safe_int(img.get("height"))
        if w >= 400 or h >= 400 or any(k in src.lower() for k in ("hero", "lead", "main", "full")):
            url = _absolutize(final_url, src)
            if url:
                return ReferenceImage(url=url, source="article-img", article_url=final_url)

    return None


def _safe_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0
