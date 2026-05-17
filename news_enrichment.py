"""News enrichment system.

Inserted between user selection and ScriptWriter to elevate a raw NewsItem
into a fact-checked, multi-source "premium" research package. Runs in
30-60s end-to-end; degrades gracefully when individual sources fail.

Five phases:
  1. Multi-source search (≤20s): Google News RSS extended + Reddit public JSON
     + international RSS feeds. Dedupes by URL canonical + Jaccard title.
  2. Verified facts (≤15s): scrape bodies, ask Claude to extract atomic claims
     with supporting source indices. Keep facts with ≥2 supports.
  3. Brief + 3 scenes (≤25s): single Claude call returns JSON with a
     1200-1500 word brief and 3 derived TTS scenes (18-24 words each).
  4. Real images (≤30s): scrape og:image, twitter:image, and <article> <img>
     from every source. Score by resolution+credibility+diversity, ask Claude
     to pick 3 (one per scene).
  5. Quality validation (≤1s): 0-100 score over source/fact/brief/image counts.
     passed = score >= threshold (default 70).

No third-party API keys required. Reddit is anonymous via the public .json
endpoint. Twitter/X is skipped in MVP (no scraping; users can add a v2 client
later by extending _phase1_multi_source_search).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, urljoin, quote_plus

import feedparser
import httpx
from bs4 import BeautifulSoup

from news_sources import NewsItem, USER_AGENT

logger = logging.getLogger(__name__)


# ---------- Tunable constants ----------

INTL_RSS_FEEDS = [
    ("BBC", "https://feeds.bbci.co.uk/news/world/rss.xml"),
    ("Reuters", "https://feeds.reuters.com/Reuters/worldNews"),
    ("AP", "https://feeds.apnews.com/rss/apf-topnews"),
    ("Al Jazeera", "https://www.aljazeera.com/xml/rss/all.xml"),
]

REDDIT_SUBS = ["news", "worldnews", "nottheonion"]

AI_IMAGE_URL_HINTS = (
    "midjourney", "dalle", "dall-e", "stable-diffusion",
    "openai", "generated", "synthesia", "runwayml",
)

CREDIBLE_OUTLETS = {
    "bbc.com": 1.0, "reuters.com": 1.0, "apnews.com": 1.0,
    "nytimes.com": 0.95, "washingtonpost.com": 0.9, "theguardian.com": 0.9,
    "aljazeera.com": 0.85, "cnn.com": 0.8, "ft.com": 0.95,
    "bloomberg.com": 0.95, "wsj.com": 0.95, "elpais.com": 0.9,
    "elmundo.es": 0.85, "reforma.com": 0.8, "milenio.com": 0.75,
    "eluniversal.com.mx": 0.75, "jornada.com.mx": 0.75, "sipse.com": 0.7,
}


# ---------- Dataclasses ----------

@dataclass
class SourceRef:
    url: str
    outlet: str            # "google_news:nytimes", "reddit:r/worldnews", "rss:bbc"
    title: str
    published_at: Optional[str] = None
    snippet: str = ""
    body: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VerifiedFact:
    text: str
    supporting_source_indices: List[int] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsImage:
    url: str
    source_article: str
    width: int = 0
    height: int = 0
    score: float = 0.0
    flagged_ai: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NewsCluster:
    """A group of SourceRefs that all cover the same event.

    Used by the /refresh endpoint to present "1 story, N sources" in the UI
    without spending Claude tokens. The deep NewsEnrichmentSystem.enrich()
    only runs when the user clicks Generar.
    """
    primary: SourceRef
    members: List[SourceRef] = field(default_factory=list)  # includes primary

    @property
    def source_count(self) -> int:
        return len(self.members)

    @property
    def credibility(self) -> float:
        if not self.members:
            return 0.0
        return sum(_credibility(m.url) for m in self.members) / len(self.members)

    def to_dict(self) -> dict:
        return {
            "primary": self.primary.to_dict(),
            "members": [m.to_dict() for m in self.members],
            "source_count": self.source_count,
            "credibility": round(self.credibility, 3),
        }


@dataclass
class EnrichedNews:
    item: NewsItem
    sources: List[SourceRef] = field(default_factory=list)
    facts: List[VerifiedFact] = field(default_factory=list)
    brief: str = ""
    scenes: Dict[str, str] = field(default_factory=dict)
    images: List[NewsImage] = field(default_factory=list)
    selected_image_urls: List[str] = field(default_factory=list)
    quality_score: int = 0
    quality_breakdown: Dict[str, int] = field(default_factory=dict)
    passed: bool = False
    errors: List[str] = field(default_factory=list)
    timings_ms: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "item": self.item.to_dict() if hasattr(self.item, "to_dict") else asdict(self.item),
            "sources": [s.to_dict() for s in self.sources],
            "facts": [f.to_dict() for f in self.facts],
            "brief": self.brief,
            "scenes": self.scenes,
            "images": [i.to_dict() for i in self.images],
            "selected_image_urls": self.selected_image_urls,
            "quality_score": self.quality_score,
            "quality_breakdown": self.quality_breakdown,
            "passed": self.passed,
            "errors": self.errors,
            "timings_ms": self.timings_ms,
        }


class EnrichmentError(Exception):
    """Raised when enrichment cannot proceed at all (e.g., Anthropic down)."""


# ---------- Utilities ----------

_STOPWORDS = {
    "el", "la", "los", "las", "un", "una", "y", "o", "de", "del", "en",
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "at", "for",
    "with", "by", "from",
}


def _tokenize_title(s: str) -> set:
    s = re.sub(r"[^\w\sáéíóúñü]", " ", (s or "").lower())
    return {t for t in s.split() if t and t not in _STOPWORDS and len(t) > 2}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def _canonical_url(url: str) -> str:
    try:
        p = urlparse(url)
    except Exception:
        return url
    host = (p.netloc or "").lower().removeprefix("www.")
    path = p.path.rstrip("/")
    return f"{host}{path}"


def _domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().removeprefix("www.")
    except Exception:
        return ""


def _credibility(url: str) -> float:
    d = _domain(url)
    if not d:
        return 0.3
    if d in CREDIBLE_OUTLETS:
        return CREDIBLE_OUTLETS[d]
    for known, score in CREDIBLE_OUTLETS.items():
        if d.endswith("." + known):
            return score * 0.95
    return 0.4


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _absolutize(base: str, url: str) -> str:
    if not url:
        return ""
    if url.startswith(("http://", "https://")):
        return url
    if url.startswith("//"):
        p = urlparse(base)
        return f"{p.scheme}:{url}"
    return urljoin(base, url)


def _strip_code_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return text.strip()


def _extract_json(text: str) -> dict:
    """Tolerant JSON extraction. Strips fences, finds first balanced object."""
    t = _strip_code_fences(text)
    try:
        return json.loads(t)
    except json.JSONDecodeError:
        pass
    start = t.find("{")
    if start < 0:
        raise json.JSONDecodeError("no json object found", t, 0)
    depth = 0
    for i in range(start, len(t)):
        if t[i] == "{":
            depth += 1
        elif t[i] == "}":
            depth -= 1
            if depth == 0:
                return json.loads(t[start:i + 1])
    raise json.JSONDecodeError("unbalanced json", t, start)


# ---------- HTTP helpers ----------

DEFAULT_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept-Language": "es-MX,es;q=0.9,en;q=0.5",
}


async def _aget(client: httpx.AsyncClient, url: str, *, timeout: float = 8.0,
                headers: Optional[Dict[str, str]] = None) -> Optional[httpx.Response]:
    try:
        r = await client.get(
            url, timeout=timeout,
            headers={**DEFAULT_HEADERS, **(headers or {})},
            follow_redirects=True,
        )
        r.raise_for_status()
        return r
    except Exception as e:
        logger.debug(f"aget failed {url[:80]}: {e}")
        return None


# ---------- NewsEnrichmentSystem ----------

@dataclass
class EnrichmentConfig:
    min_sources: int = 7
    min_facts: int = 5
    min_images: int = 8
    quality_threshold: int = 70
    brief_min_words: int = 1200
    brief_max_words: int = 1500
    model: str = "claude-haiku-4-5"
    phase1_timeout: float = 20.0
    phase2_timeout: float = 15.0
    phase3_timeout: float = 25.0
    phase4_timeout: float = 30.0
    per_request_timeout: float = 8.0
    user_agent: str = USER_AGENT


class NewsEnrichmentSystem:
    """Five-phase pipeline. Pass `anthropic_client` (an `anthropic.Anthropic`
    instance). `logger_obj` is optional; falls back to module logger.

    The Anthropic client is invoked via `asyncio.to_thread` so the rest of the
    pipeline stays async without needing the official async client.
    """

    def __init__(
        self,
        anthropic_client,
        logger_obj=None,
        *,
        config: Optional[EnrichmentConfig] = None,
        http_client: Optional[httpx.AsyncClient] = None,
    ):
        self.anthropic = anthropic_client
        self.log = logger_obj or logger
        self.config = config or EnrichmentConfig()
        self._http_owned = http_client is None
        self._http_client = http_client

    # ----- public -----

    async def enrich(self, item: NewsItem) -> EnrichedNews:
        result = EnrichedNews(item=item)

        client = self._http_client or httpx.AsyncClient(headers=DEFAULT_HEADERS)
        try:
            t0 = time.monotonic()
            try:
                result.sources = await asyncio.wait_for(
                    self._phase1_multi_source_search(client, item),
                    timeout=self.config.phase1_timeout,
                )
            except asyncio.TimeoutError:
                result.errors.append("phase1_timeout")
            result.timings_ms["phase1"] = int((time.monotonic() - t0) * 1000)
            self._log_info(
                f"📚 phase1: {len(result.sources)} sources "
                f"({result.timings_ms['phase1']}ms)"
            )

            t1 = time.monotonic()
            try:
                result.facts = await asyncio.wait_for(
                    self._phase2_extract_verified_facts(client, item, result.sources),
                    timeout=self.config.phase2_timeout,
                )
            except asyncio.TimeoutError:
                result.errors.append("phase2_timeout")
            except EnrichmentError as e:
                result.errors.append(f"phase2:{e}")
            result.timings_ms["phase2"] = int((time.monotonic() - t1) * 1000)
            self._log_info(
                f"🧪 phase2: {len(result.facts)} verified facts "
                f"({result.timings_ms['phase2']}ms)"
            )

            t2 = time.monotonic()
            try:
                brief, scenes = await asyncio.wait_for(
                    self._phase3_rewrite_with_llm(item, result.facts, result.sources),
                    timeout=self.config.phase3_timeout,
                )
                result.brief = brief
                result.scenes = scenes
            except asyncio.TimeoutError:
                result.errors.append("phase3_timeout")
            except EnrichmentError as e:
                result.errors.append(f"phase3:{e}")
            result.timings_ms["phase3"] = int((time.monotonic() - t2) * 1000)
            self._log_info(
                f"✍️  phase3: brief {len(result.brief.split())} words "
                f"({result.timings_ms['phase3']}ms)"
            )

            t3 = time.monotonic()
            try:
                images, selected = await asyncio.wait_for(
                    self._phase4_extract_real_images(client, result.sources, item),
                    timeout=self.config.phase4_timeout,
                )
                result.images = images
                result.selected_image_urls = selected
            except asyncio.TimeoutError:
                result.errors.append("phase4_timeout")
            result.timings_ms["phase4"] = int((time.monotonic() - t3) * 1000)
            self._log_info(
                f"📸 phase4: {len(result.images)} images, "
                f"{len(result.selected_image_urls)} selected "
                f"({result.timings_ms['phase4']}ms)"
            )

            t4 = time.monotonic()
            self._phase5_validate_quality(result)
            result.timings_ms["phase5"] = int((time.monotonic() - t4) * 1000)
            self._log_info(
                f"✅ phase5: score {result.quality_score}/100 "
                f"passed={result.passed}"
            )

            return result
        finally:
            if self._http_owned:
                await client.aclose()

    # ----- phase 1 -----

    async def _phase1_multi_source_search(
        self, client: httpx.AsyncClient, item: NewsItem
    ) -> List[SourceRef]:
        query = item.title
        results = await asyncio.gather(
            self._search_google_news(client, query),
            self._search_reddit(client, query),
            self._search_intl_rss(client, query),
            return_exceptions=True,
        )

        all_sources: List[SourceRef] = []
        seed = SourceRef(
            url=item.url, outlet=f"original:{item.source}",
            title=item.title, published_at=item.published_at,
            snippet=item.snippet, body=item.body or None,
        )
        all_sources.append(seed)

        for r in results:
            if isinstance(r, Exception):
                logger.debug(f"phase1 fan-out: {r}")
                continue
            all_sources.extend(r)

        return self._dedupe_sources(all_sources)

    async def _search_google_news(
        self, client: httpx.AsyncClient, query: str
    ) -> List[SourceRef]:
        url = (
            "https://news.google.com/rss/search"
            f"?q={quote_plus(query)}+when:7d&hl=es-419&gl=MX&ceid=MX:es-419"
        )
        r = await _aget(client, url, timeout=self.config.per_request_timeout)
        if r is None:
            return []
        parsed = feedparser.parse(r.text)
        out: List[SourceRef] = []
        for e in parsed.entries[:15]:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            outlet = "google_news"
            if " - " in title:
                head, _, tail = title.rpartition(" - ")
                if tail and len(tail) < 60:
                    title = head.strip()
                    outlet = f"google_news:{tail.strip()}"
            summary = e.get("summary", "") or ""
            snippet = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)
            out.append(SourceRef(
                url=link, outlet=outlet, title=title,
                published_at=_iso_from_struct(e.get("published_parsed")),
                snippet=snippet,
            ))
        return out

    async def _search_reddit(
        self, client: httpx.AsyncClient, query: str
    ) -> List[SourceRef]:
        out: List[SourceRef] = []
        for sub in REDDIT_SUBS:
            url = (
                f"https://www.reddit.com/r/{sub}/search.json"
                f"?q={quote_plus(query)}&restrict_sr=1&sort=relevance&t=week&limit=10"
            )
            r = await _aget(
                client, url,
                timeout=self.config.per_request_timeout,
                headers={"User-Agent": f"NewsViralBot/0.1 by /u/anon"},
            )
            if r is None:
                continue
            try:
                data = r.json()
            except Exception:
                continue
            children = (data.get("data") or {}).get("children") or []
            for c in children[:8]:
                d = c.get("data") or {}
                link = d.get("url_overridden_by_dest") or d.get("url") or ""
                if not link or "reddit.com" in link:
                    perm = d.get("permalink") or ""
                    link = f"https://reddit.com{perm}" if perm else ""
                if not link:
                    continue
                created = d.get("created_utc")
                published = None
                if created:
                    from datetime import datetime, timezone
                    published = datetime.fromtimestamp(
                        float(created), tz=timezone.utc
                    ).isoformat()
                out.append(SourceRef(
                    url=link,
                    outlet=f"reddit:r/{sub}",
                    title=(d.get("title") or "").strip(),
                    published_at=published,
                    snippet=(d.get("selftext") or "")[:500],
                ))
        return out

    async def _search_intl_rss(
        self, client: httpx.AsyncClient, query: str
    ) -> List[SourceRef]:
        tokens = _tokenize_title(query)
        if not tokens:
            return []

        async def _one(name: str, feed_url: str) -> List[SourceRef]:
            r = await _aget(client, feed_url, timeout=self.config.per_request_timeout)
            if r is None:
                return []
            parsed = feedparser.parse(r.text)
            picks: List[SourceRef] = []
            for e in parsed.entries[:30]:
                title = (e.get("title") or "").strip()
                if _jaccard(_tokenize_title(title), tokens) < 0.15:
                    continue
                link = (e.get("link") or "").strip()
                summary = e.get("summary", "") or ""
                snippet = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)
                picks.append(SourceRef(
                    url=link, outlet=f"rss:{name.lower()}", title=title,
                    published_at=_iso_from_struct(e.get("published_parsed")),
                    snippet=snippet,
                ))
                if len(picks) >= 4:
                    break
            return picks

        results = await asyncio.gather(
            *[_one(name, url) for name, url in INTL_RSS_FEEDS],
            return_exceptions=True,
        )
        out: List[SourceRef] = []
        for r in results:
            if isinstance(r, list):
                out.extend(r)
        return out

    def _dedupe_sources(self, sources: List[SourceRef]) -> List[SourceRef]:
        seen_urls: set = set()
        kept: List[SourceRef] = []
        kept_token_sets: List[set] = []
        for s in sources:
            if not s.url:
                continue
            canon = _canonical_url(s.url)
            if canon in seen_urls:
                continue
            toks = _tokenize_title(s.title)
            duplicate = False
            for prev in kept_token_sets:
                if _jaccard(toks, prev) > 0.75:
                    duplicate = True
                    break
            if duplicate:
                continue
            seen_urls.add(canon)
            kept.append(s)
            kept_token_sets.append(toks)
        return kept

    # ----- phase 2 -----

    async def _phase2_extract_verified_facts(
        self, client: httpx.AsyncClient, item: NewsItem, sources: List[SourceRef]
    ) -> List[VerifiedFact]:
        if not sources:
            return []

        async def _fetch_body(s: SourceRef) -> None:
            if s.body:
                return
            r = await _aget(client, s.url, timeout=self.config.per_request_timeout)
            if r is None:
                return
            soup = BeautifulSoup(r.text, "html.parser")
            article = soup.find("article") or soup
            for tag in article(["script", "style", "iframe", "aside", "figure", "ins", "nav"]):
                tag.decompose()
            paragraphs = [p.get_text(" ", strip=True) for p in article.find_all("p")]
            body = "\n\n".join(p for p in paragraphs if len(p) > 20)
            if body:
                s.body = body[:3000]

        await asyncio.gather(*[_fetch_body(s) for s in sources], return_exceptions=True)

        numbered = []
        for i, s in enumerate(sources):
            text = (s.body or s.snippet or s.title)[:1500]
            numbered.append(f"[S{i}] ({s.outlet}) {s.title}\n{text}")
        compiled = "\n\n---\n\n".join(numbered)

        system = (
            "Eres un fact-checker periodístico. Te entregan N fuentes "
            "etiquetadas [S0]..[SN]. Extrae afirmaciones fácticas ATÓMICAS "
            "(una idea concreta cada una) que aparezcan en las fuentes. "
            "Para cada afirmación lista los índices de fuentes que la "
            "sustentan literal o casi literalmente. NO incluyas opiniones, "
            "especulación, ni inferencias. NO inventes. Si una afirmación "
            "sólo aparece en una fuente, márcala igual; el caller filtra."
        )
        user = (
            f"TITULAR DE TRABAJO: {item.title}\n\n"
            f"FUENTES:\n{compiled}\n\n"
            "Responde EXCLUSIVAMENTE este JSON, sin markdown ni texto extra:\n"
            "{\n"
            '  "facts": [\n'
            '    {"text": "<afirmación atómica>", "supporting": [0,2], "confidence": 0.0-1.0}\n'
            "  ]\n"
            "}"
        )

        text = await self._call_claude(system, user, max_tokens=1500)
        try:
            data = _extract_json(text)
        except json.JSONDecodeError as e:
            raise EnrichmentError(f"facts_json_parse:{e}")

        facts: List[VerifiedFact] = []
        for f in data.get("facts", []) or []:
            txt = (f.get("text") or "").strip()
            sup = f.get("supporting") or []
            conf = float(f.get("confidence") or 0.0)
            if not txt:
                continue
            sup_ints = [int(i) for i in sup if isinstance(i, (int, float))
                        and 0 <= int(i) < len(sources)]
            if len(sup_ints) >= 2 or conf >= 0.85:
                facts.append(VerifiedFact(
                    text=txt, supporting_source_indices=sup_ints, confidence=conf,
                ))
        return facts

    # ----- phase 3 -----

    async def _phase3_rewrite_with_llm(
        self, item: NewsItem, facts: List[VerifiedFact], sources: List[SourceRef]
    ) -> Tuple[str, Dict[str, str]]:
        fmin = self.config.brief_min_words
        fmax = self.config.brief_max_words

        facts_block = "\n".join(
            f"[F{i+1}] (sources {f.supporting_source_indices}) {f.text}"
            for i, f in enumerate(facts)
        ) or "(sin hechos verificados — usa cuerpo original con cautela)"
        sources_block = "\n".join(
            f"[S{i}] {s.outlet} — {s.title}" for i, s in enumerate(sources[:12])
        ) or "(sin fuentes adicionales)"

        async def _one_call(extra_instruction: str = "") -> Tuple[str, Dict[str, str]]:
            system = (
                "Eres redactor jefe del noticiero 'VOZ DEL PUEBLO'. "
                "Generas un BRIEF de investigación periodística profundo "
                f"({fmin}-{fmax} palabras EN ESPAÑOL) basado SÓLO en "
                "hechos verificados con citas [F1],[F2],..., y derivas "
                "tres escenas TTS cortas (18-24 palabras CADA UNA, en "
                "español, primera persona del ancla, tono conversacional, "
                "presente). Prohibido: especulación, hechos no citados, "
                "exclamaciones huecas, 'última hora'. La escena 3 termina "
                "con un hook que deja a la audiencia pensando.\n\n"
                + extra_instruction
            )
            user = (
                f"TITULAR: {item.title}\n"
                f"FUENTE ORIGINAL: {item.source}\n\n"
                f"HECHOS VERIFICADOS:\n{facts_block}\n\n"
                f"ÍNDICE DE FUENTES:\n{sources_block}\n\n"
                "Responde JSON EXACTO sin markdown:\n"
                "{\n"
                f'  "brief": "<texto de {fmin}-{fmax} palabras con citas [F1],[F2]>",\n'
                '  "scenes": {\n'
                '    "escena_1": "<español, 18-24 palabras, ancla planteando hook>",\n'
                '    "escena_2": "<español, 18-24 palabras, narración del evento>",\n'
                '    "escena_3": "<español, 18-24 palabras, ancla cerrando con intriga>"\n'
                "  }\n"
                "}"
            )
            text = await self._call_claude(system, user, max_tokens=2800)
            data = _extract_json(text)
            brief = (data.get("brief") or "").strip()
            scenes_raw = data.get("scenes") or {}
            scenes = {
                k: (v or "").strip()
                for k, v in scenes_raw.items()
                if k in ("escena_1", "escena_2", "escena_3")
            }
            return brief, scenes

        try:
            brief, scenes = await _one_call()
        except json.JSONDecodeError as e:
            raise EnrichmentError(f"brief_json_parse:{e}")

        wc = len(brief.split())
        if not (fmin <= wc <= fmax):
            logger.warning(f"brief words={wc} out of [{fmin},{fmax}], retrying once")
            try:
                brief2, scenes2 = await _one_call(
                    f"AVISO: tu intento anterior tuvo {wc} palabras. "
                    f"DEBE estar entre {fmin} y {fmax}. Ajusta longitud."
                )
                if fmin <= len(brief2.split()) <= fmax:
                    brief, scenes = brief2, scenes2
            except Exception as e:
                logger.warning(f"brief retry failed: {e}")

        for k in ("escena_1", "escena_2", "escena_3"):
            scenes.setdefault(k, "")

        return brief, scenes

    # ----- phase 4 -----

    async def _phase4_extract_real_images(
        self, client: httpx.AsyncClient, sources: List[SourceRef], item: NewsItem
    ) -> Tuple[List[NewsImage], List[str]]:
        if not sources:
            return [], []

        async def _scrape(s: SourceRef) -> List[NewsImage]:
            r = await _aget(client, s.url, timeout=self.config.per_request_timeout)
            if r is None:
                return []
            soup = BeautifulSoup(r.text, "html.parser")
            base = str(r.url)
            found: List[NewsImage] = []

            og = soup.find("meta", attrs={"property": "og:image"})
            if og and og.get("content"):
                u = _absolutize(base, og["content"])
                if u:
                    found.append(NewsImage(url=u, source_article=s.url))

            for sel in ({"name": "twitter:image"}, {"property": "twitter:image"}):
                tw = soup.find("meta", attrs=sel)
                if tw and tw.get("content"):
                    u = _absolutize(base, tw["content"])
                    if u:
                        found.append(NewsImage(url=u, source_article=s.url))

            article = soup.find("article") or soup
            for img in article.find_all("img"):
                src = img.get("src") or img.get("data-src") or ""
                if not src:
                    continue
                low = src.lower()
                if any(h in low for h in
                       ("avatar", "tracker", "pixel", "logo-", "/logo.",
                        "icon-", "favicon", "1x1", "sprite")):
                    continue
                w = _safe_int(img.get("width"))
                h = _safe_int(img.get("height"))
                if not (w * h >= 400 * 300 or
                        any(k in low for k in ("hero", "lead", "main", "full"))):
                    continue
                u = _absolutize(base, src)
                if u:
                    found.append(NewsImage(
                        url=u, source_article=s.url, width=w, height=h,
                    ))
            return found

        per_source = await asyncio.gather(
            *[_scrape(s) for s in sources], return_exceptions=True,
        )
        raw: List[NewsImage] = []
        for r in per_source:
            if isinstance(r, list):
                raw.extend(r)

        deduped = self._dedupe_images(raw)
        scored = self._score_images(deduped)

        selected = await self._llm_pick_images(scored, item)
        return scored, selected

    def _dedupe_images(self, imgs: List[NewsImage]) -> List[NewsImage]:
        seen_urls: set = set()
        kept: List[NewsImage] = []
        for img in imgs:
            if not img.url:
                continue
            key = hashlib.sha1(img.url.encode("utf-8", "ignore")).hexdigest()
            if key in seen_urls:
                continue
            seen_urls.add(key)
            low = img.url.lower()
            img.flagged_ai = any(h in low for h in AI_IMAGE_URL_HINTS)
            kept.append(img)
        return kept

    def _score_images(self, imgs: List[NewsImage]) -> List[NewsImage]:
        if not imgs:
            return imgs
        max_pixels = max((i.width * i.height) for i in imgs) or 1
        domain_counts: Dict[str, int] = {}
        for img in imgs:
            d = _domain(img.source_article)
            domain_counts[d] = domain_counts.get(d, 0) + 1

        seen_domains: set = set()
        for img in imgs:
            pixels = img.width * img.height
            res_norm = (pixels / max_pixels) if max_pixels else 0.0
            cred = _credibility(img.source_article)
            d = _domain(img.source_article)
            diversity_bonus = 0.0 if d in seen_domains else 1.0
            seen_domains.add(d)
            ai_penalty = -0.4 if img.flagged_ai else 0.0
            img.score = (
                0.55 * res_norm + 0.30 * cred + 0.15 * diversity_bonus + ai_penalty
            )
        # AI-flagged images always demoted below clean ones, regardless of
        # resolution or outlet score — we don't want generated art in a news
        # report even if it's pretty.
        return sorted(imgs, key=lambda x: (not x.flagged_ai, x.score), reverse=True)

    async def _llm_pick_images(
        self, imgs: List[NewsImage], item: NewsItem
    ) -> List[str]:
        if not imgs:
            return []
        usable = [i for i in imgs if not i.flagged_ai] or imgs
        if len(usable) <= 3:
            return [i.url for i in usable[:3]]

        candidates = usable[:12]
        listing = "\n".join(
            f"[I{idx}] outlet={_domain(c.source_article)} "
            f"size={c.width}x{c.height} score={c.score:.2f} url={c.url}"
            for idx, c in enumerate(candidates)
        )
        system = (
            "Eres editor visual. Te dan candidatos de imagen para 3 escenas "
            "de un video noticioso. Escoge UNA imagen por escena (3 total) "
            "que mejor ilustre el evento. Prefiere imágenes de alta "
            "resolución y outlets creíbles. Devuelve sólo índices."
        )
        user = (
            f"TITULAR: {item.title}\n\n"
            f"CANDIDATOS:\n{listing}\n\n"
            'Responde JSON: {"selected": [<idx_escena_1>, <idx_escena_2>, <idx_escena_3>]}'
        )
        try:
            text = await self._call_claude(system, user, max_tokens=200)
            data = _extract_json(text)
            picks = data.get("selected") or []
            urls: List[str] = []
            seen: set = set()
            for p in picks[:3]:
                try:
                    i = int(p)
                except (TypeError, ValueError):
                    continue
                if 0 <= i < len(candidates) and i not in seen:
                    urls.append(candidates[i].url)
                    seen.add(i)
            if len(urls) < 3:
                for c in candidates:
                    if c.url not in urls:
                        urls.append(c.url)
                    if len(urls) >= 3:
                        break
            return urls[:3]
        except Exception as e:
            logger.warning(f"llm_pick_images fell back to top-3: {e}")
            return [c.url for c in candidates[:3]]

    # ----- phase 5 -----

    def _phase5_validate_quality(self, result: EnrichedNews) -> None:
        bd: Dict[str, int] = {}
        bd["sources"] = 25 if len(result.sources) >= self.config.min_sources else int(
            25 * len(result.sources) / self.config.min_sources
        )
        bd["facts"] = 25 if len(result.facts) >= self.config.min_facts else int(
            25 * len(result.facts) / self.config.min_facts
        )
        wc = len(result.brief.split())
        if self.config.brief_min_words <= wc <= self.config.brief_max_words:
            bd["brief"] = 20
        elif wc >= self.config.brief_min_words * 0.7:
            bd["brief"] = 10
        elif wc > 0:
            bd["brief"] = 5
        else:
            bd["brief"] = 0
        bd["images"] = 20 if len(result.images) >= self.config.min_images else int(
            20 * len(result.images) / self.config.min_images
        )
        fatal = [e for e in result.errors if "timeout" in e or "parse" in e]
        bd["errors"] = 10 if not fatal else max(0, 10 - 3 * len(fatal))

        total = sum(bd.values())
        result.quality_breakdown = bd
        result.quality_score = min(100, total)
        result.passed = result.quality_score >= self.config.quality_threshold

    # ----- internals -----

    async def _call_claude(self, system: str, user: str, *, max_tokens: int) -> str:
        def _sync_call() -> str:
            msg = self.anthropic.messages.create(
                model=self.config.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in msg.content if hasattr(b, "text")).strip()

        try:
            return await asyncio.to_thread(_sync_call)
        except Exception as e:
            raise EnrichmentError(f"anthropic_call:{e}")

    def _log_info(self, msg: str) -> None:
        if hasattr(self.log, "info"):
            try:
                self.log.info(msg)
                return
            except Exception:
                pass
        logger.info(msg)


def _iso_from_struct(ps) -> Optional[str]:
    if not ps:
        return None
    try:
        from datetime import datetime, timezone
        return datetime(*ps[:6], tzinfo=timezone.utc).isoformat()
    except Exception:
        return None


# ===== Public no-LLM aggregator (used by /refresh) =====
#
# These functions implement Phase 1 of the enrichment pipeline as a
# standalone unit: fetch news from N sources, dedupe, cluster by similar
# title. No Anthropic calls, no body scraping, no image work. Cheap
# (~2-4s end-to-end) and safe to call on every UI refresh.


async def _search_google_news_q(
    client: httpx.AsyncClient, query: str, timeout: float = 8.0
) -> List[SourceRef]:
    url = (
        "https://news.google.com/rss/search"
        f"?q={quote_plus(query)}+when:7d&hl=es-419&gl=MX&ceid=MX:es-419"
    )
    r = await _aget(client, url, timeout=timeout)
    if r is None:
        return []
    parsed = feedparser.parse(r.text)
    out: List[SourceRef] = []
    for e in parsed.entries[:25]:
        title = (e.get("title") or "").strip()
        link = (e.get("link") or "").strip()
        outlet = "google_news"
        if " - " in title:
            head, _, tail = title.rpartition(" - ")
            if tail and len(tail) < 60:
                title = head.strip()
                outlet = f"google_news:{tail.strip()}"
        summary = e.get("summary", "") or ""
        snippet = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)
        out.append(SourceRef(
            url=link, outlet=outlet, title=title,
            published_at=_iso_from_struct(e.get("published_parsed")),
            snippet=snippet,
        ))
    return out


async def _search_reddit_q(
    client: httpx.AsyncClient, query: str, timeout: float = 8.0,
    subs: Optional[List[str]] = None,
) -> List[SourceRef]:
    async def _one_sub(sub: str) -> List[SourceRef]:
        url = (
            f"https://www.reddit.com/r/{sub}/search.json"
            f"?q={quote_plus(query)}&restrict_sr=1&sort=relevance&t=week&limit=8"
        )
        r = await _aget(
            client, url, timeout=timeout,
            headers={"User-Agent": "NewsViralBot/0.1 by /u/anon"},
        )
        if r is None:
            return []
        try:
            data = r.json()
        except Exception:
            return []
        items: List[SourceRef] = []
        for c in (data.get("data") or {}).get("children") or []:
            d = c.get("data") or {}
            link = d.get("url_overridden_by_dest") or d.get("url") or ""
            if not link or "reddit.com" in link:
                perm = d.get("permalink") or ""
                link = f"https://reddit.com{perm}" if perm else ""
            if not link:
                continue
            created = d.get("created_utc")
            published = None
            if created:
                from datetime import datetime, timezone
                published = datetime.fromtimestamp(
                    float(created), tz=timezone.utc
                ).isoformat()
            items.append(SourceRef(
                url=link, outlet=f"reddit:r/{sub}",
                title=(d.get("title") or "").strip(),
                published_at=published,
                snippet=(d.get("selftext") or "")[:500],
            ))
        return items

    target_subs = subs or REDDIT_SUBS
    results = await asyncio.gather(
        *[_one_sub(s) for s in target_subs], return_exceptions=True,
    )
    out: List[SourceRef] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out


def cluster_sources_by_title(
    sources: List[SourceRef], jaccard_threshold: float = 0.45
) -> List[NewsCluster]:
    """Group sources that look like they cover the same event.

    Greedy clustering: each source either joins the first cluster whose
    primary title has Jaccard token similarity ≥ threshold, or starts a new
    cluster. Within each cluster the most credible outlet (or longest body
    if tied) becomes the primary.
    """
    clusters: List[NewsCluster] = []
    cluster_tokens: List[set] = []
    for s in sources:
        tokens = _tokenize_title(s.title)
        joined = False
        for i, prev_tokens in enumerate(cluster_tokens):
            if _jaccard(tokens, prev_tokens) >= jaccard_threshold:
                clusters[i].members.append(s)
                # union tokens — clusters drift toward stable shared vocabulary
                cluster_tokens[i] = prev_tokens | tokens
                joined = True
                break
        if not joined:
            clusters.append(NewsCluster(primary=s, members=[s]))
            cluster_tokens.append(tokens)

    # Re-pick primary per cluster: highest credibility, then longest body.
    for c in clusters:
        c.members.sort(
            key=lambda m: (_credibility(m.url), len(m.body or m.snippet or "")),
            reverse=True,
        )
        c.primary = c.members[0]
    return clusters


async def aggregate_news_clusters(
    query: str,
    *,
    extra_queries: Optional[List[str]] = None,
    intl_rss: bool = True,
    reddit: bool = True,
    timeout_total: float = 12.0,
    jaccard_threshold: float = 0.35,
    http_client: Optional[httpx.AsyncClient] = None,
) -> List[NewsCluster]:
    """One-shot multi-source aggregation + clustering. No LLM.

    Use this from a web UI on every refresh — it's fast and free.

    Partial-result safe: if `timeout_total` elapses, returns whatever sub-tasks
    have already finished (instead of dropping everything).
    """
    owned = http_client is None
    client = http_client or httpx.AsyncClient(headers=DEFAULT_HEADERS)

    queries = [query] + list(extra_queries or [])

    # Build labelled tasks so we can log which sources actually returned.
    labelled: List[Tuple[str, "asyncio.Task[List[SourceRef]]"]] = []
    for i, q in enumerate(queries):
        labelled.append((f"google_news[{i}]",
                         asyncio.create_task(_search_google_news_q(client, q))))
        if reddit:
            labelled.append((f"reddit[{i}]",
                             asyncio.create_task(_search_reddit_q(client, q))))
    if intl_rss:
        labelled.append(("intl_rss",
                         asyncio.create_task(_search_intl_rss_module(client, query))))

    try:
        # asyncio.wait does NOT cancel pending tasks on timeout, so we can
        # collect partial results from whatever finished.
        done, pending = await asyncio.wait(
            [t for _, t in labelled],
            timeout=timeout_total,
            return_when=asyncio.ALL_COMPLETED,
        )
        if pending:
            logger.warning(
                f"aggregator: {len(pending)} task(s) still running at {timeout_total}s — cancelling"
            )
            for t in pending:
                t.cancel()

        merged: List[SourceRef] = []
        per_source_counts: Dict[str, int] = {}
        for label, t in labelled:
            if t in done:
                try:
                    items = t.result()
                except Exception as e:
                    logger.debug(f"aggregator task {label} raised: {e}")
                    items = []
                if items:
                    merged.extend(items)
                    per_source_counts[label] = len(items)
            else:
                per_source_counts[label] = -1  # cancelled / not finished
    finally:
        if owned:
            await client.aclose()

    # Dedupe by canonical URL only (clustering handles near-duplicates by title).
    seen: set = set()
    deduped: List[SourceRef] = []
    for s in merged:
        if not s.url:
            continue
        canon = _canonical_url(s.url)
        if canon in seen:
            continue
        seen.add(canon)
        deduped.append(s)

    clusters = cluster_sources_by_title(deduped, jaccard_threshold=jaccard_threshold)
    multi = sum(1 for c in clusters if c.source_count >= 2)
    logger.info(
        f"aggregator: per-task={per_source_counts} "
        f"raw={len(merged)} deduped={len(deduped)} "
        f"clusters={len(clusters)} multi_source={multi}"
    )
    return clusters


async def _search_intl_rss_module(
    client: httpx.AsyncClient, query: str, timeout: float = 8.0
) -> List[SourceRef]:
    """Module-level twin of NewsEnrichmentSystem._search_intl_rss for reuse."""
    tokens = _tokenize_title(query)
    if not tokens:
        return []

    async def _one(name: str, feed_url: str) -> List[SourceRef]:
        r = await _aget(client, feed_url, timeout=timeout)
        if r is None:
            return []
        parsed = feedparser.parse(r.text)
        picks: List[SourceRef] = []
        for e in parsed.entries[:30]:
            title = (e.get("title") or "").strip()
            if _jaccard(_tokenize_title(title), tokens) < 0.15:
                continue
            link = (e.get("link") or "").strip()
            summary = e.get("summary", "") or ""
            snippet = BeautifulSoup(summary, "html.parser").get_text(" ", strip=True)
            picks.append(SourceRef(
                url=link, outlet=f"rss:{name.lower()}", title=title,
                published_at=_iso_from_struct(e.get("published_parsed")),
                snippet=snippet,
            ))
            if len(picks) >= 4:
                break
        return picks

    results = await asyncio.gather(
        *[_one(name, url) for name, url in INTL_RSS_FEEDS],
        return_exceptions=True,
    )
    out: List[SourceRef] = []
    for r in results:
        if isinstance(r, list):
            out.extend(r)
    return out
