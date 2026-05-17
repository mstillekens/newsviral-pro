"""DuckDuckGo image search as fallback for news_image_finder.

When og:image scraping fails (paywall, missing meta tags, layout change)
we fall back to a simple image search using the news headline as the query.
DuckDuckGo doesn't require an API key — perfect for this throwaway use.

Returns the FIRST acceptable image URL or None. We skip obvious non-photo
extensions and any URL that doesn't resolve to a typical image file. The
caller (flux-canny-pro) will validate the URL by actually loading it; if
the URL is broken we silently fall back to plain text-to-image FLUX.
"""
from __future__ import annotations

import logging
from typing import Optional

from duckduckgo_search import DDGS

logger = logging.getLogger(__name__)

# Skip SVGs (vector, doesn't work as a canny control image), favicons, gifs,
# and obvious stock-photo icons.
_BAD_EXTENSIONS = (".svg", ".ico", ".gif")
_BAD_PATH_HINTS = ("icon", "favicon", "logo-", "/logo.")


def _is_usable(url: str) -> bool:
    lower = (url or "").lower()
    if not lower.startswith(("http://", "https://")):
        return False
    if any(lower.endswith(ext) for ext in _BAD_EXTENSIONS):
        return False
    if any(hint in lower for hint in _BAD_PATH_HINTS):
        return False
    return True


def search_news_image(query: str, max_results: int = 5) -> Optional[str]:
    """Return the first usable image URL DuckDuckGo finds for this query.

    Returns None on any failure (network down, no results, all results
    filtered). Caller falls back to text-only FLUX in that case."""
    try:
        with DDGS() as ddgs:
            results = list(ddgs.images(query, max_results=max_results))
    except Exception as e:
        logger.info(f"DuckDuckGo image search failed for {query!r}: {e}")
        return None

    for r in results:
        url = r.get("image") if isinstance(r, dict) else None
        if url and _is_usable(url):
            return url
    return None
