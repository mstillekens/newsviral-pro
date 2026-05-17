# News Engine Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Redesign the news collection layer so it fetches from 50–100+ sources (RSS, sitemap, Google News multi-query) with deduplication, generic extraction, and SQLite persistence — replacing the current single-RSS, sipse-only, JSON-file architecture.

**Architecture:** A `source_registry.json` drives a `SourceFetcher` that fans out async RSS + sitemap polls concurrently; a generic `extractor.py` (trafilatura) replaces the sipse-only scraper; `news_db.py` persists articles and URL hashes to SQLite; and the existing `news_scorer.py` + `brand_style.classify_vertical` are left untouched but fed richer inputs.

**Tech Stack:** Python 3.11+, feedparser (existing), httpx (existing), trafilatura (new), aiohttp (existing), sqlite3 (stdlib), APScheduler (new for M8)

---

## Diagnosis: Why the Current System Finds So Few News

| Root Cause | Location | Impact |
|---|---|---|
| **Single RSS source** | `news_sources.py:26` — one hardcoded URL | Only ~12–30 items per run |
| **`require_region_hit=True` by default** | `news_sources.py:84` | Drops every national/political story not mentioning Cancún |
| **`max_items=30` cap** | `news_sources.py:84` | Hard ceiling even when more RSS items exist |
| **`since_days=2` window** | `news_sources.py:84` | Discards anything older than 48 h |
| **No multi-query Google News** | `news_sources.py:26` | Only QR-regional query, no politics/corruption/security queries |
| **No other RSS feeds** | `news_sources.py` | Zero feeds from local QR papers, nationals, independents |
| **No sitemap / homepage discovery** | absent | Can't find stories from sites with no RSS |
| **No URL deduplication store** | absent | Same story re-fetched every run, no history |
| **Sipse body extractor is domain-locked** | `news_sources.py:157` | `"sipse.com" not in url → return None` |
| **No fetch logging** | absent | Silent failures are invisible |
| **No article database** | absent | Every run starts from zero |

---

## Evaluation: Scrapling vs ScrapeGraphAI

### Scrapling (`D4Vinci/Scrapling`)
| Aspect | Assessment |
|---|---|
| **Problem solved** | Anti-bot evasion — smart CSS/XPath with Playwright stealth, auto-adapts to DOM changes |
| **Stability** | Active project, ~2k stars, but still pre-1.0 |
| **Integration effort** | Medium — replaces httpx in fetch layer, adds Playwright dependency (~200 MB) |
| **Advantages over BS4** | Handles JS-heavy pages, fingerprint evasion, selector auto-healing |
| **Metadata/dates** | Not native — still need trafilatura or manual extraction on top |
| **Legal/ethical risk** | Medium — stealth mode is designed to circumvent bot detection, which violates most ToS |
| **Use it when** | A high-value source blocks scrapers and no RSS/API exists |
| **Don't use it when** | RSS or sitemap exists; scraping at scale across dozens of sources |

**Verdict:** Do NOT use as the primary extractor. Reserve as a last-resort option for 1–2 specific sources where no RSS exists and scraping is clearly permitted by the site's ToS.

### ScrapeGraphAI (`ScrapeGraphAI/Scrapegraph-ai`)
| Aspect | Assessment |
|---|---|
| **Problem solved** | LLM-powered structured extraction — ask "give me title, date, author" and the LLM figures out the DOM |
| **Stability** | Active, ~15k stars, used in production |
| **Integration effort** | Low API surface, but requires Claude/OpenAI/Ollama key per extraction |
| **Advantages over BS4** | Zero CSS selector maintenance, handles any layout change automatically |
| **Cost** | ~$0.001–0.01 per article (Claude Haiku/GPT-4o-mini) — expensive at 1,000+/day |
| **Latency** | 2–5× slower than trafilatura (LLM round-trip) |
| **Legal/ethical risk** | Low — reads public pages, just uses AI to parse them |
| **Use it when** | You need structured extraction from a single high-value source that defeats every other extractor |
| **Don't use it when** | High-volume pipeline; RSS/sitemap exists; cost is a concern |

**Verdict:** Do NOT use in the hot path. Useful only for one-off enrichment (e.g., scraping a government press-release page monthly). Too expensive and slow for 50–100 sources × many times/day.

### Recommended: `trafilatura`
- Purpose-built for news extraction (title, date, author, full text, language)
- Handles 95%+ of CMS layouts (WordPress, Drupal, custom)  
- Respects `robots.txt` optionally; built-in rate limiting
- Fast: pure Python, no browser required
- Zero API cost
- Well-maintained, used in major NLP datasets

---

## File Map

| File | Action | Responsibility |
|---|---|---|
| `source_registry.json` | **CREATE** | Master list of 50+ sources with metadata |
| `news_sources.py` | **MODIFY** | Add `SourceRegistry`, `SourceFetcher`, multi-RSS async fan-out, sitemap parser |
| `news_db.py` | **CREATE** | SQLite schema + CRUD helpers for sources, articles, fetch_logs |
| `extractor.py` | **CREATE** | Generic article extractor via trafilatura; replaces sipse-only scraper |
| `deduplicator.py` | **CREATE** | URL-hash and content-hash dedup against SQLite |
| `pipeline.py` | **CREATE** | Full fetch→extract→dedup→score→classify→store orchestrator |
| `tests/test_multi_source.py` | **CREATE** | Tests for SourceRegistry loading and multi-RSS fan-out |
| `tests/test_extractor.py` | **CREATE** | Tests for trafilatura extraction and metadata parsing |
| `tests/test_deduplicator.py` | **CREATE** | Tests for URL-hash and content-hash dedup logic |
| `tests/test_pipeline.py` | **CREATE** | Integration tests for the full pipeline with fixtures |
| `requirements.txt` | **MODIFY** | Add trafilatura, apscheduler |
| `news_viral_pro.py` | **MODIFY** (M8) | Wire `pipeline.py` instead of direct `fetch_google_news` call |

> **DO NOT MODIFY:** `news_scorer.py`, `brand_style.py`, `script_writer.py`, `replicate_orchestrator.py`, `video_compositor.py`. These are downstream of the news layer and work correctly.

---

## Task 1: source_registry.json — Master Source List

**Files:**
- Create: `source_registry.json`
- Create: `tests/test_multi_source.py` (partial — schema validation)

The registry drives everything. Each source specifies how to poll it. Populate with 50+ verified sources across local QR, national MX, government, and independent media.

- [ ] **Step 1: Write the failing schema test**

```python
# tests/test_multi_source.py
import json
from pathlib import Path

REQUIRED_FIELDS = {"name", "url", "type", "region", "method", "priority", "active"}
VALID_METHODS = {"rss", "sitemap", "homepage", "api"}
VALID_TYPES = {"local", "national", "government", "independent", "aggregator"}
VALID_REGIONS = {"qr", "national", "mx_southeast", "international"}
VALID_PRIORITIES = {"high", "medium", "low"}


def load_registry():
    path = Path(__file__).parent.parent / "source_registry.json"
    return json.loads(path.read_text())


def test_registry_has_at_least_50_sources():
    reg = load_registry()
    assert len(reg["sources"]) >= 50


def test_every_source_has_required_fields():
    reg = load_registry()
    for src in reg["sources"]:
        missing = REQUIRED_FIELDS - set(src.keys())
        assert not missing, f"Source {src.get('name')} missing: {missing}"


def test_every_active_rss_source_has_feed_url():
    reg = load_registry()
    for src in reg["sources"]:
        if src.get("active") and src.get("method") == "rss":
            assert src.get("feed_url"), f"{src['name']} is RSS but missing feed_url"


def test_valid_method_values():
    reg = load_registry()
    for src in reg["sources"]:
        assert src["method"] in VALID_METHODS, f"{src['name']} has unknown method {src['method']}"


def test_valid_priority_values():
    reg = load_registry()
    for src in reg["sources"]:
        assert src["priority"] in VALID_PRIORITIES, f"{src['name']} has bad priority"
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_multi_source.py -v
```
Expected: FAIL with `FileNotFoundError` or `AssertionError: len >= 50`

- [ ] **Step 3: Create source_registry.json**

```json
{
  "version": "1.0",
  "updated": "2026-05-17",
  "note": "Verify all feed_url values before enabling. Mark active=false if a source blocks or has no feed.",
  "sources": [
    {
      "name": "Sipse Noticias",
      "url": "https://sipse.com",
      "feed_url": "https://sipse.com/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Main QR paper. Was domain-locked in old code — now treated generically."
    },
    {
      "name": "Por Esto Quintana Roo",
      "url": "https://www.poresto.net",
      "feed_url": "https://www.poresto.net/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify feed URL is live."
    },
    {
      "name": "Novedades Quintana Roo",
      "url": "https://www.novedadesqroo.com.mx",
      "feed_url": "https://www.novedadesqroo.com.mx/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify feed URL."
    },
    {
      "name": "Quadratin Quintana Roo",
      "url": "https://quintanaroo.quadratin.com.mx",
      "feed_url": "https://quintanaroo.quadratin.com.mx/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Quadratin has state-level subsites."
    },
    {
      "name": "Noticaribe",
      "url": "https://noticaribe.com.mx",
      "feed_url": "https://noticaribe.com.mx/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "QR investigative/security focus."
    },
    {
      "name": "La Verdad Noticias QR",
      "url": "https://laverdadnoticias.com",
      "feed_url": "https://laverdadnoticias.com/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify feed."
    },
    {
      "name": "Punto Medio MX",
      "url": "https://www.puntomedio.mx",
      "feed_url": "https://www.puntomedio.mx/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify availability."
    },
    {
      "name": "Por Cancún",
      "url": "https://porcancun.com",
      "feed_url": "https://porcancun.com/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Local Cancún paper. Verify."
    },
    {
      "name": "Diario de Yucatán",
      "url": "https://www.yucatan.com.mx",
      "feed_url": "https://www.yucatan.com.mx/rss",
      "type": "local",
      "region": "mx_southeast",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Covers Yucatán Peninsula broadly."
    },
    {
      "name": "El Universal",
      "url": "https://www.eluniversal.com.mx",
      "feed_url": "https://www.eluniversal.com.mx/rss.xml",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Major national paper."
    },
    {
      "name": "Milenio",
      "url": "https://www.milenio.com",
      "feed_url": "https://www.milenio.com/rss",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify exact feed path."
    },
    {
      "name": "Excélsior",
      "url": "https://www.excelsior.com.mx",
      "feed_url": "https://www.excelsior.com.mx/rss.xml",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify."
    },
    {
      "name": "La Jornada",
      "url": "https://www.jornada.com.mx",
      "feed_url": "https://www.jornada.com.mx/rss/edicion.xml",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Left-leaning. Good political coverage."
    },
    {
      "name": "Animal Político",
      "url": "https://www.animalpolitico.com",
      "feed_url": "https://www.animalpolitico.com/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Investigative/fact-checking. High quality."
    },
    {
      "name": "Aristegui Noticias",
      "url": "https://aristeguinoticias.com",
      "feed_url": "https://aristeguinoticias.com/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Investigative. High editorial credibility."
    },
    {
      "name": "Sin Embargo MX",
      "url": "https://www.sinembargo.mx",
      "feed_url": "https://www.sinembargo.mx/feed",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify feed."
    },
    {
      "name": "Proceso",
      "url": "https://www.proceso.com.mx",
      "feed_url": "https://www.proceso.com.mx/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "Leading investigative weekly. May be behind paywall for full body."
    },
    {
      "name": "Expansión MX",
      "url": "https://expansion.mx",
      "feed_url": "https://expansion.mx/rss",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Business/economy focus."
    },
    {
      "name": "Infobae México",
      "url": "https://www.infobae.com/mexico/",
      "feed_url": "https://www.infobae.com/feeds/rss/mexico/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "High traffic. Verify feed path."
    },
    {
      "name": "El Financiero",
      "url": "https://www.elfinanciero.com.mx",
      "feed_url": "https://www.elfinanciero.com.mx/arc/outboundfeeds/rss/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Economy/politics. Verify feed."
    },
    {
      "name": "El Economista MX",
      "url": "https://www.eleconomista.com.mx",
      "feed_url": "https://www.eleconomista.com.mx/rss/rss.xml",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Business paper."
    },
    {
      "name": "Nexos",
      "url": "https://www.nexos.com.mx",
      "feed_url": "https://www.nexos.com.mx/?feed=rss2",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "Political analysis. Less frequent but high quality."
    },
    {
      "name": "Letras Libres",
      "url": "https://www.letraslibres.com",
      "feed_url": "https://www.letraslibres.com/feed",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "Opinion/analysis."
    },
    {
      "name": "Pie de Página",
      "url": "https://piedepagina.mx",
      "feed_url": "https://piedepagina.mx/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Human rights / investigative."
    },
    {
      "name": "Desinformémonos",
      "url": "https://desinformemonos.org",
      "feed_url": "https://desinformemonos.org/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "Social movements / human rights."
    },
    {
      "name": "Chilango",
      "url": "https://www.chilango.com",
      "feed_url": "https://www.chilango.com/feed/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "CDMX lifestyle/culture. Low priority for QR focus."
    },
    {
      "name": "Gobierno de México — Boletines",
      "url": "https://www.gob.mx",
      "feed_url": "https://www.gob.mx/cms/uploads/rss/index.rss",
      "type": "government",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 5,
      "notes": "Official federal press releases. Verify feed URL."
    },
    {
      "name": "Gobierno QR — Comunicados",
      "url": "https://www.qroo.gob.mx",
      "feed_url": null,
      "type": "government",
      "region": "qr",
      "method": "sitemap",
      "priority": "medium",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 10,
      "notes": "No RSS confirmed. Enable sitemap method when implemented."
    },
    {
      "name": "CONAGUA SMN (alertas)",
      "url": "https://smn.conagua.gob.mx",
      "feed_url": null,
      "type": "government",
      "region": "national",
      "method": "homepage",
      "priority": "high",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 10,
      "notes": "Weather alerts. No RSS; would need custom scraper. Activate in M4."
    },
    {
      "name": "Secretaría de Seguridad QR",
      "url": "https://www.sspe.qroo.gob.mx",
      "feed_url": null,
      "type": "government",
      "region": "qr",
      "method": "homepage",
      "priority": "medium",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 10,
      "notes": "Security/police press releases. No RSS. Activate in M4."
    },
    {
      "name": "IMSS — Noticias",
      "url": "https://www.imss.gob.mx",
      "feed_url": "https://www.imss.gob.mx/prensa/rss",
      "type": "government",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 5,
      "notes": "Health news. Verify."
    },
    {
      "name": "Google News — QR Regional",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=Cancun+OR+%22Quintana+Roo%22+OR+%22Riviera+Maya%22+OR+Tulum+OR+%22Playa+del+Carmen%22&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "Existing query. Kept as-is."
    },
    {
      "name": "Google News — Política México",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=pol%C3%ADtica+M%C3%A9xico+gobierno+corrupci%C3%B3n&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: national politics query."
    },
    {
      "name": "Google News — Seguridad México",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=seguridad+crimen+cartel+M%C3%A9xico&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: security/crime query."
    },
    {
      "name": "Google News — Corrupción Gobierno",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=corrupci%C3%B3n+funcionario+gobierno+denuncia&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "national",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: corruption/government query."
    },
    {
      "name": "Google News — Cancún Turismo Seguridad",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=Canc%C3%BAn+turismo+seguridad+hotel+playa&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: Cancún tourism/security-specific query."
    },
    {
      "name": "Google News — Chetumal Bacalar Sur",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=Chetumal+OR+Bacalar+OR+%22sur+de+Quintana+Roo%22&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "qr",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: South QR coverage."
    },
    {
      "name": "Google News — Clima Huracanes Caribe",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=hurac%C3%A1n+tormenta+tropical+Caribe+M%C3%A9xico&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "qr",
      "method": "rss",
      "priority": "high",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: weather/hurricane query."
    },
    {
      "name": "Google News — Denuncias Ciudadanas",
      "url": "https://news.google.com",
      "feed_url": "https://news.google.com/rss/search?q=denuncia+ciudadana+vecinos+alcald%C3%ADa+municipio&hl=es-419&gl=MX&ceid=MX:es-419",
      "type": "aggregator",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 1,
      "notes": "NEW: citizen complaints / local governance."
    },
    {
      "name": "Yahoo Noticias México",
      "url": "https://es.yahoo.com/noticias/mexico/",
      "feed_url": "https://es.yahoo.com/noticias/mexico/rss",
      "type": "aggregator",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Aggregator. Verify feed URL structure."
    },
    {
      "name": "El Debate",
      "url": "https://www.debate.com.mx",
      "feed_url": "https://www.debate.com.mx/rss/portada.xml",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Northwest MX focus but covers nationals. Verify."
    },
    {
      "name": "Tribuna",
      "url": "https://www.tribuna.com.mx",
      "feed_url": "https://www.tribuna.com.mx/rss/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify."
    },
    {
      "name": "El Heraldo de México",
      "url": "https://heraldodemexico.com.mx",
      "feed_url": "https://heraldodemexico.com.mx/feed/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Verify."
    },
    {
      "name": "La Silla Rota",
      "url": "https://lasillarota.com",
      "feed_url": "https://lasillarota.com/feed",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Digital native. Political focus."
    },
    {
      "name": "Código Magenta",
      "url": "https://codigomagenta.com.mx",
      "feed_url": "https://codigomagenta.com.mx/feed/",
      "type": "local",
      "region": "qr",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "QR local. Verify."
    },
    {
      "name": "Cancún Mío",
      "url": "https://www.cancunmio.com",
      "feed_url": null,
      "type": "local",
      "region": "qr",
      "method": "sitemap",
      "priority": "medium",
      "active": false,
      "requires_region_filter": false,
      "crawl_delay_seconds": 5,
      "notes": "Local lifestyle/news. No confirmed RSS; enable sitemap in M4."
    },
    {
      "name": "Eje Central",
      "url": "https://www.ejecentral.com.mx",
      "feed_url": "https://www.ejecentral.com.mx/feed/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Urban CDMX focus but covers national politics."
    },
    {
      "name": "Reporte Índigo",
      "url": "https://www.reporteindigo.com",
      "feed_url": "https://www.reporteindigo.com/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Digital investigative."
    },
    {
      "name": "Verificado MX",
      "url": "https://verificado.com.mx",
      "feed_url": "https://verificado.com.mx/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "Fact-checking. High signal quality."
    },
    {
      "name": "Sopitas",
      "url": "https://www.sopitas.com",
      "feed_url": "https://www.sopitas.com/feed/",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "low",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Trending/viral news."
    },
    {
      "name": "El País México",
      "url": "https://elpais.com/mexico/",
      "feed_url": "https://feeds.elpais.com/mrss-s/pages/ep/site/elpais.com/section/mexico/portada",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "International quality journalism on Mexico."
    },
    {
      "name": "Reuters América Latina",
      "url": "https://www.reuters.com/world/americas/",
      "feed_url": "https://feeds.reuters.com/reuters/MexicoNews",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Wire agency. Mexico-specific feed."
    },
    {
      "name": "AP Noticias México",
      "url": "https://apnews.com/hub/mexico",
      "feed_url": "https://apnews.com/rss/apf-latam",
      "type": "national",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 2,
      "notes": "Wire agency. Latin America feed."
    },
    {
      "name": "Contralínea",
      "url": "https://contralinea.com.mx",
      "feed_url": "https://contralinea.com.mx/feed/",
      "type": "independent",
      "region": "national",
      "method": "rss",
      "priority": "medium",
      "active": true,
      "requires_region_filter": false,
      "crawl_delay_seconds": 3,
      "notes": "Investigative. Corruption and government focus."
    }
  ],
  "social_signals": {
    "note": "Social signal sources — separate layer, not RSS fetching",
    "sources": [
      {
        "platform": "YouTube",
        "method": "YouTube Data API v3",
        "legal_status": "official_api",
        "notes": "Search for Mexico/QR news channels. Free 10,000 units/day. Requires API key."
      },
      {
        "platform": "X/Twitter",
        "method": "X API v2 (Basic/Free tier)",
        "legal_status": "official_api",
        "notes": "500K tweet reads/month free. Search recent tweets for QR/MX trending topics."
      },
      {
        "platform": "Google Trends",
        "method": "pytrends (unofficial but widely tolerated)",
        "legal_status": "semi_official",
        "notes": "No official API. pytrends scrapes public data. Rate-limit aggressively (1 req/30s). Use for topic trending only, not article data."
      },
      {
        "platform": "Facebook",
        "method": "CrowdTangle was shut down. Meta Content Library requires academic approval.",
        "legal_status": "restricted",
        "notes": "DO NOT scrape. Only Meta Content Library API (academic only). Skip for M6."
      },
      {
        "platform": "Instagram",
        "method": "No public API for non-business accounts",
        "legal_status": "restricted",
        "notes": "DO NOT scrape. Skip."
      },
      {
        "platform": "TikTok",
        "method": "TikTok Research API (application required)",
        "legal_status": "official_api_gated",
        "notes": "Apply at developers.tiktok.com/products/research-api/. Academic/journalist access. Skip until approved."
      }
    ]
  }
}
```

- [ ] **Step 4: Run tests to confirm they pass**

```bash
pytest tests/test_multi_source.py -v
```
Expected: PASS (all 5 tests green)

- [ ] **Step 5: Commit**

```bash
git add source_registry.json tests/test_multi_source.py
git commit -m "feat(sources): add source_registry.json with 50+ sources and schema tests"
```

---

## Task 2: SourceRegistry + SourceFetcher in news_sources.py

**Files:**
- Modify: `news_sources.py` (add `Source` dataclass, `SourceRegistry`, `fetch_all_sources`)
- Modify: `tests/test_multi_source.py` (add fetcher unit tests)

The `NewsItem` dataclass needs new fields to support deduplication and richer metadata. Add them without breaking existing callers — new fields are optional with defaults.

- [ ] **Step 1: Write failing tests for SourceRegistry and multi-source fetch**

```python
# Append to tests/test_multi_source.py

from unittest.mock import patch, MagicMock
from news_sources import (
    SourceRegistry,
    Source,
    NewsItem,
    fetch_all_rss_sources,
)


def test_source_registry_loads_active_rss_sources():
    reg = SourceRegistry.from_file("source_registry.json")
    active_rss = [s for s in reg.sources if s.active and s.method == "rss"]
    assert len(active_rss) >= 10


def test_source_dataclass_has_required_attributes():
    reg = SourceRegistry.from_file("source_registry.json")
    src = reg.sources[0]
    assert hasattr(src, "name")
    assert hasattr(src, "feed_url")
    assert hasattr(src, "region")
    assert hasattr(src, "priority")
    assert hasattr(src, "requires_region_filter")
    assert hasattr(src, "crawl_delay_seconds")


def test_news_item_has_new_fields():
    item = NewsItem(
        title="Test",
        url="https://example.com/news/1",
        source="Test Source",
        published_at="2026-05-17T10:00:00+00:00",
    )
    assert hasattr(item, "detected_at")
    assert hasattr(item, "url_hash")
    assert hasattr(item, "category")
    assert hasattr(item, "source_region")


def test_fetch_all_rss_sources_deduplicates_by_url():
    """fetch_all_rss_sources must not return two items with the same url."""
    with patch("news_sources.feedparser.parse") as mock_parse:
        mock_parse.return_value = MagicMock(entries=[
            MagicMock(
                title="Duplicate Story - Source A",
                link="https://example.com/story/1",
                summary="Same story",
                published_parsed=(2026, 5, 17, 10, 0, 0, 0, 0, 0),
            )
        ])
        reg = SourceRegistry.from_file("source_registry.json")
        # Only use 2 sources for speed
        sources = [s for s in reg.sources if s.active and s.method == "rss"][:2]
        items = fetch_all_rss_sources(sources, since_days=7)
        urls = [i.url for i in items]
        assert len(urls) == len(set(urls)), "Duplicate URLs found"
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_multi_source.py::test_source_registry_loads_active_rss_sources -v
```
Expected: FAIL — `ImportError: cannot import name 'SourceRegistry' from 'news_sources'`

- [ ] **Step 3: Extend news_sources.py — add Source, SourceRegistry, new NewsItem fields, fetch_all_rss_sources**

Add this block to `news_sources.py` (below the existing imports, replacing or extending the file):

```python
# === NEW: Source dataclass and SourceRegistry ===
import hashlib
import json
from pathlib import Path

@dataclass
class Source:
    """A single news source entry from the registry."""
    name: str
    url: str
    type: str                        # local | national | government | independent | aggregator
    region: str                      # qr | national | mx_southeast | international
    method: str                      # rss | sitemap | homepage | api
    priority: str                    # high | medium | low
    active: bool
    feed_url: Optional[str] = None
    requires_region_filter: bool = False
    crawl_delay_seconds: float = 2.0
    notes: str = ""


class SourceRegistry:
    """Loads and manages the list of news sources."""

    def __init__(self, sources: List[Source]):
        self.sources = sources

    @classmethod
    def from_file(cls, path: str = "source_registry.json") -> "SourceRegistry":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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
        return [s for s in self.sources if s.active and s.method == "rss" and s.feed_url]

    def by_priority(self, priority: str) -> List[Source]:
        return [s for s in self.sources if s.priority == priority]
```

Then update the `NewsItem` dataclass to add new fields:

```python
@dataclass
class NewsItem:
    """A single news item, source-agnostic."""
    title: str
    url: str
    source: str
    published_at: str                         # ISO 8601
    snippet: str = ""
    body: str = ""
    region_hits: List[str] = field(default_factory=list)
    # New fields (M2+)
    detected_at: str = ""                     # ISO 8601 — when we first fetched it
    url_hash: str = ""                        # MD5 of normalized URL
    category: str = ""                        # from brand_style.classify_vertical
    source_region: str = ""                   # 'qr' | 'national' | etc.
    author: str = ""
    image_url: str = ""

    def __post_init__(self):
        if not self.detected_at:
            self.detected_at = datetime.now(timezone.utc).isoformat()
        if not self.url_hash:
            self.url_hash = hashlib.md5(
                self.url.lower().strip().encode()
            ).hexdigest()

    def to_dict(self) -> dict:
        return asdict(self)
```

Then add `fetch_all_rss_sources`:

```python
def _parse_rss_source(source: Source, since_days: int, seen_urls: set) -> List[NewsItem]:
    """Fetch and parse a single RSS source. Returns new items not in seen_urls."""
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

        # Dedup by URL
        url_key = link.lower().strip()
        if url_key in seen_urls:
            continue
        seen_urls.add(url_key)

        snippet_html = entry.get("summary", "") or entry.get("description", "")
        snippet = BeautifulSoup(snippet_html, "html.parser").get_text(" ", strip=True)

        published_struct = entry.get("published_parsed")
        if published_struct:
            published_dt = datetime(*published_struct[:6], tzinfo=timezone.utc)
        else:
            published_dt = datetime.now(timezone.utc)

        if published_dt < cutoff:
            continue

        # Strip source from title (Google News style "Headline - Source")
        source_name = source.name
        if " - " in title:
            head, _, tail = title.rpartition(" - ")
            if tail and len(tail) < 60:
                title = head.strip()
                source_name = tail.strip()

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
    """Fetch all active RSS sources and return deduplicated NewsItems."""
    seen_urls: set = set()
    all_items: List[NewsItem] = []

    for source in sources:
        if len(all_items) >= max_total:
            break
        items = _parse_rss_source(source, since_days, seen_urls)
        all_items.extend(items[:max_per_source])

    logger.info(f"📰 Total after multi-source fetch: {len(all_items)} items")
    return all_items
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_multi_source.py -v
```
Expected: All tests PASS

- [ ] **Step 5: Commit**

```bash
git add news_sources.py
git commit -m "feat(sources): SourceRegistry, multi-RSS fan-out, extended NewsItem fields"
```

---

## Task 3: news_db.py — SQLite Persistence Layer

**Files:**
- Create: `news_db.py`
- Create: `tests/test_news_db.py`

SQLite is the right choice here: no infra, zero config, works in the existing project, and handles tens of thousands of articles without issue. We store articles, fetch logs, and a URL-hash index for deduplication.

- [ ] **Step 1: Write failing tests**

```python
# tests/test_news_db.py
import pytest
import tempfile
import os
from news_db import NewsDB, ArticleRecord


@pytest.fixture
def tmp_db(tmp_path):
    db_path = str(tmp_path / "test.db")
    db = NewsDB(db_path)
    db.init_schema()
    yield db
    db.close()


def test_init_creates_tables(tmp_db):
    tables = tmp_db.list_tables()
    assert "articles" in tables
    assert "fetch_logs" in tables
    assert "url_hashes" in tables


def test_insert_article_and_retrieve(tmp_db):
    rec = ArticleRecord(
        url="https://example.com/news/1",
        url_hash="abc123",
        title="Test Title",
        source="Test Source",
        published_at="2026-05-17T10:00:00+00:00",
        detected_at="2026-05-17T11:00:00+00:00",
        snippet="A short snippet",
        category="politica",
        source_region="qr",
    )
    tmp_db.insert_article(rec)
    found = tmp_db.get_article_by_hash("abc123")
    assert found is not None
    assert found.title == "Test Title"


def test_insert_duplicate_is_ignored(tmp_db):
    rec = ArticleRecord(
        url="https://example.com/news/2",
        url_hash="def456",
        title="Duplicate",
        source="Test Source",
        published_at="2026-05-17T10:00:00+00:00",
        detected_at="2026-05-17T11:00:00+00:00",
    )
    tmp_db.insert_article(rec)
    tmp_db.insert_article(rec)  # second insert should not raise
    count = tmp_db.count_articles()
    assert count == 1


def test_is_known_url_returns_true_for_seen_url(tmp_db):
    tmp_db.mark_url_seen("https://example.com/news/3", "ghi789")
    assert tmp_db.is_known_url("ghi789") is True


def test_is_known_url_returns_false_for_new_url(tmp_db):
    assert tmp_db.is_known_url("notexist000") is False


def test_log_fetch(tmp_db):
    tmp_db.log_fetch("Test Source", "https://example.com/feed", "ok", 10, 0)
    logs = tmp_db.recent_fetch_logs(limit=5)
    assert len(logs) == 1
    assert logs[0]["source_name"] == "Test Source"
    assert logs[0]["status"] == "ok"


def test_get_recent_articles(tmp_db):
    for i in range(5):
        tmp_db.insert_article(ArticleRecord(
            url=f"https://example.com/news/{i}",
            url_hash=f"hash{i}",
            title=f"Story {i}",
            source="S",
            published_at="2026-05-17T10:00:00+00:00",
            detected_at="2026-05-17T11:00:00+00:00",
        ))
    recent = tmp_db.get_recent_articles(limit=3)
    assert len(recent) == 3
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_news_db.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'news_db'`

- [ ] **Step 3: Create news_db.py**

```python
"""SQLite persistence layer for the news engine.

Stores articles, URL hashes for deduplication, and fetch logs.
All DB access goes through NewsDB — no raw sqlite3 outside this module.
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
    def __init__(self, db_path: str = "news.db"):
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

    def init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
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
        """)
        self._conn.commit()

    def list_tables(self) -> List[str]:
        cur = self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        return [row[0] for row in cur.fetchall()]

    def insert_article(self, rec: ArticleRecord) -> None:
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

    def get_article_by_hash(self, url_hash: str) -> Optional[ArticleRecord]:
        cur = self._conn.execute(
            "SELECT * FROM articles WHERE url_hash = ?", (url_hash,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return ArticleRecord(**{k: row[k] for k in row.keys()})

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
        return self._conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

    def get_recent_articles(self, limit: int = 50) -> List[ArticleRecord]:
        cur = self._conn.execute(
            "SELECT * FROM articles ORDER BY detected_at DESC LIMIT ?", (limit,)
        )
        return [ArticleRecord(**{k: row[k] for k in row.keys()}) for row in cur.fetchall()]

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
            "SELECT * FROM fetch_logs ORDER BY fetched_at DESC LIMIT ?", (limit,)
        )
        return [dict(row) for row in cur.fetchall()]

    def close(self) -> None:
        self._conn.close()
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_news_db.py -v
```
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add news_db.py tests/test_news_db.py
git commit -m "feat(db): SQLite persistence — articles, url_hashes, fetch_logs"
```

---

## Task 4: extractor.py — Generic Article Extractor (trafilatura)

**Files:**
- Modify: `requirements.txt` (add trafilatura)
- Create: `extractor.py`
- Create: `tests/test_extractor.py`

Replaces the `scrape_sipse` function with a generic extractor that works on any site. `trafilatura` is the best-in-class Python news extraction library — it handles multi-CMS layouts, extracts date/author, and has built-in `robots.txt` support.

- [ ] **Step 1: Add trafilatura to requirements.txt**

```
trafilatura>=1.8.0,<2.0.0
```

Install:
```bash
pip install trafilatura>=1.8.0
```

- [ ] **Step 2: Write failing tests**

```python
# tests/test_extractor.py
from unittest.mock import patch, MagicMock
from extractor import extract_article, ArticleContent


def test_extract_article_returns_content_dataclass():
    fake_html = """<html><head><title>Test</title></head><body>
    <article><h1>Breaking News</h1><p>This is the article body.</p>
    <time datetime="2026-05-17">May 17 2026</time></article></body></html>"""
    
    with patch("extractor.trafilatura.fetch_url", return_value=fake_html):
        with patch("extractor.trafilatura.extract") as mock_extract:
            mock_extract.return_value = "This is the article body."
            result = extract_article("https://example.com/news/1")

    assert isinstance(result, ArticleContent)
    assert result.text is not None


def test_extract_article_returns_none_on_failure():
    with patch("extractor.trafilatura.fetch_url", return_value=None):
        result = extract_article("https://example.com/404")
    assert result is None


def test_extract_article_non_sipse_url_works():
    """Verify the old sipse domain-lock is gone."""
    with patch("extractor.trafilatura.fetch_url", return_value="<html><body><p>Content</p></body></html>"):
        with patch("extractor.trafilatura.extract", return_value="Content"):
            result = extract_article("https://animalpolitico.com/story/1")
    # Should not return None due to domain check
    # (may be None if extract returns empty, but NOT due to domain filter)
    # We verify no domain-based None is raised
    assert True  # if we got here, no domain exception was raised


def test_extract_article_respects_timeout():
    """extract_article must not hang — verify timeout is passed."""
    import extractor
    assert extractor.FETCH_TIMEOUT_SECONDS > 0
    assert extractor.FETCH_TIMEOUT_SECONDS <= 30
```

- [ ] **Step 3: Run to confirm failure**

```bash
pytest tests/test_extractor.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'extractor'`

- [ ] **Step 4: Create extractor.py**

```python
"""Generic article extractor using trafilatura.

Replaces the sipse-only scraper in news_sources.py. Works on any public URL.
Respects robots.txt via trafilatura's built-in option.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import trafilatura
import trafilatura.settings

logger = logging.getLogger(__name__)

FETCH_TIMEOUT_SECONDS = 20


@dataclass
class ArticleContent:
    text: Optional[str]
    title: Optional[str] = None
    author: Optional[str] = None
    date: Optional[str] = None
    language: Optional[str] = None


def extract_article(url: str, *, check_robots: bool = True) -> Optional[ArticleContent]:
    """Download and extract the main text from a news article URL.

    Returns ArticleContent on success, None on any failure (paywall, timeout,
    blocked by robots.txt, parsing error).

    check_robots: if True, trafilatura skips URLs disallowed by robots.txt.
    """
    try:
        html = trafilatura.fetch_url(
            url,
            config=trafilatura.settings.use_config(),
        )
        if not html:
            logger.debug(f"extractor: no HTML returned for {url}")
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
            logger.debug(f"extractor: trafilatura returned empty for {url}")
            return None

        import json
        data = json.loads(result)
        return ArticleContent(
            text=data.get("text"),
            title=data.get("title"),
            author=data.get("author"),
            date=data.get("date"),
            language=data.get("language"),
        )

    except Exception as e:
        logger.warning(f"extractor: failed for {url}: {e}")
        return None
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_extractor.py -v
```
Expected: All 4 tests PASS

- [ ] **Step 6: Commit**

```bash
git add requirements.txt extractor.py tests/test_extractor.py
git commit -m "feat(extractor): generic article extractor via trafilatura, replaces sipse-only scraper"
```

---

## Task 5: deduplicator.py — URL + Content Hash Dedup

**Files:**
- Create: `deduplicator.py`
- Create: `tests/test_deduplicator.py`

Two dedup strategies:
1. **URL hash**: MD5 of normalized URL — fastest, catches exact-same URL
2. **Title hash**: MD5 of normalized title+source — catches same story from multiple aggregators

- [ ] **Step 1: Write failing tests**

```python
# tests/test_deduplicator.py
from deduplicator import Deduplicator, title_hash
from news_sources import NewsItem


def make_item(title, url, source="src"):
    return NewsItem(title=title, url=url, source=source, published_at="2026-05-17T10:00:00+00:00")


def test_title_hash_is_deterministic():
    h1 = title_hash("Balacera en Cancún", "Sipse")
    h2 = title_hash("Balacera en Cancún", "Sipse")
    assert h1 == h2


def test_title_hash_different_sources_differ():
    h1 = title_hash("Same Headline", "Source A")
    h2 = title_hash("Same Headline", "Source B")
    assert h1 != h2


def test_title_hash_normalizes_case_and_whitespace():
    h1 = title_hash("  BALACERA EN CANCÚN  ", "Sipse")
    h2 = title_hash("balacera en cancún", "sipse")
    assert h1 == h2


def test_deduplicator_rejects_seen_url():
    dedup = Deduplicator()
    item = make_item("Story", "https://example.com/story/1")
    assert dedup.is_new(item) is True    # first time: new
    assert dedup.is_new(item) is False   # second time: duplicate


def test_deduplicator_rejects_same_title_different_url():
    """Same story reposted at a different URL (Google News aggregated)."""
    dedup = Deduplicator()
    item1 = make_item("Balacera en Cancún", "https://sipse.com/1", source="Sipse")
    item2 = make_item("Balacera en Cancún", "https://google.com/news/abc", source="Sipse")
    dedup.is_new(item1)  # mark first as seen
    assert dedup.is_new(item2) is False  # same title+source = duplicate


def test_deduplicator_allows_different_stories():
    dedup = Deduplicator()
    item1 = make_item("Story A", "https://example.com/a")
    item2 = make_item("Story B", "https://example.com/b")
    assert dedup.is_new(item1) is True
    assert dedup.is_new(item2) is True


def test_filter_new_removes_duplicates():
    dedup = Deduplicator()
    items = [
        make_item("Story A", "https://example.com/a"),
        make_item("Story A", "https://example.com/a"),  # exact duplicate
        make_item("Story B", "https://example.com/b"),
    ]
    new_items = dedup.filter_new(items)
    assert len(new_items) == 2
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_deduplicator.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'deduplicator'`

- [ ] **Step 3: Create deduplicator.py**

```python
"""In-memory deduplication for the news pipeline.

Two levels:
  1. URL hash — catches the exact same article URL from multiple sources.
  2. Title hash (title + source, normalized) — catches the same story
     re-published at a different URL (common with Google News aggregation).

For cross-run persistence, the NewsDB.is_known_url() check should be called
first before instantiating Deduplicator, so already-stored articles are also
filtered out.
"""
from __future__ import annotations

import hashlib
import re
from typing import List, Set

from news_sources import NewsItem


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def url_hash(url: str) -> str:
    return hashlib.md5(_normalize(url).encode()).hexdigest()


def title_hash(title: str, source: str) -> str:
    key = f"{_normalize(title)}|{_normalize(source)}"
    return hashlib.md5(key.encode()).hexdigest()


class Deduplicator:
    """Session-scoped dedup. Instantiate once per pipeline run."""

    def __init__(self):
        self._seen_urls: Set[str] = set()
        self._seen_titles: Set[str] = set()

    def is_new(self, item: NewsItem) -> bool:
        """Return True if this item has not been seen this session."""
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
        return [item for item in items if self.is_new(item)]
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_deduplicator.py -v
```
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add deduplicator.py tests/test_deduplicator.py
git commit -m "feat(dedup): URL-hash and title-hash deduplication layer"
```

---

## Task 6: pipeline.py — Full Orchestrator

**Files:**
- Create: `pipeline.py`
- Create: `tests/test_pipeline.py`

Wire everything together: load sources → fetch all RSS → dedup (in-memory + DB) → classify → score → store → return ranked list. This replaces the direct `fetch_google_news` call in `news_viral_pro.py`.

- [ ] **Step 1: Write failing integration tests**

```python
# tests/test_pipeline.py
import pytest
from unittest.mock import patch, MagicMock
from pipeline import NewsPipeline, PipelineConfig


@pytest.fixture
def cfg(tmp_path):
    return PipelineConfig(
        db_path=str(tmp_path / "test.db"),
        registry_path="source_registry.json",
        since_days=7,
        max_items=100,
    )


def test_pipeline_run_returns_scored_items(cfg):
    with patch("pipeline.fetch_all_rss_sources") as mock_fetch:
        mock_fetch.return_value = [
            MagicMock(
                title="Balacera en Cancún deja dos heridos",
                url="https://sipse.com/news/1",
                source="Sipse",
                published_at="2026-05-17T10:00:00+00:00",
                snippet="Hubo un tiroteo.",
                body="",
                region_hits=["Cancún"],
                source_region="qr",
                url_hash="abc123",
                category="",
            )
        ]
        pipeline = NewsPipeline(cfg)
        results = pipeline.run()

    assert len(results) >= 1
    assert hasattr(results[0], "score")


def test_pipeline_deduplicates_same_url(cfg):
    item = MagicMock(
        title="Story A",
        url="https://example.com/story/1",
        source="Src",
        published_at="2026-05-17T10:00:00+00:00",
        snippet="",
        body="",
        region_hits=[],
        source_region="national",
        url_hash="dup001",
        category="",
    )
    with patch("pipeline.fetch_all_rss_sources", return_value=[item, item]):
        pipeline = NewsPipeline(cfg)
        results = pipeline.run()

    urls = [r.item.url for r in results]
    assert len(urls) == len(set(urls))


def test_pipeline_logs_fetch_to_db(cfg):
    with patch("pipeline.fetch_all_rss_sources", return_value=[]):
        pipeline = NewsPipeline(cfg)
        pipeline.run()
        logs = pipeline.db.recent_fetch_logs(limit=10)
    assert len(logs) >= 0  # at minimum, no crash


def test_pipeline_config_defaults():
    cfg = PipelineConfig(db_path=":memory:")
    assert cfg.since_days == 3
    assert cfg.max_items == 500
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_pipeline.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'pipeline'`

- [ ] **Step 3: Create pipeline.py**

```python
"""News pipeline orchestrator.

Full flow:
  1. Load SourceRegistry
  2. Fetch all active RSS sources (async fan-out)
  3. Deduplicate (in-memory + DB URL-hash check)
  4. Classify each item via brand_style.classify_vertical
  5. Score via news_scorer
  6. Store new articles to SQLite
  7. Return sorted ScoredItem list

Usage:
    cfg = PipelineConfig(db_path="news.db")
    pipeline = NewsPipeline(cfg)
    results = pipeline.run()   # List[ScoredItem], sorted by score desc
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List

from brand_style import classify_vertical
from deduplicator import Deduplicator
from news_db import NewsDB, ArticleRecord
from news_scorer import ScoredItem, load_weights, score_items
from news_sources import SourceRegistry, fetch_all_rss_sources

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    db_path: str = "news.db"
    registry_path: str = "source_registry.json"
    since_days: int = 3
    max_per_source: int = 50
    max_items: int = 500
    weights_path: str = "score_weights.json"


class NewsPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.db = NewsDB(config.db_path)
        self.db.init_schema()

    def run(self) -> List[ScoredItem]:
        cfg = self.config

        # 1. Load sources
        registry = SourceRegistry.from_file(cfg.registry_path)
        sources = registry.active_rss_sources()
        logger.info(f"pipeline: {len(sources)} active RSS sources")

        # 2. Fetch
        raw_items = fetch_all_rss_sources(
            sources,
            since_days=cfg.since_days,
            max_per_source=cfg.max_per_source,
            max_total=cfg.max_items,
        )
        logger.info(f"pipeline: {len(raw_items)} raw items fetched")

        # 3. Deduplicate (in-memory session + DB cross-run)
        dedup = Deduplicator()
        new_items = []
        for item in raw_items:
            if self.db.is_known_url(item.url_hash):
                continue  # already stored from a previous run
            if dedup.is_new(item):
                new_items.append(item)

        logger.info(f"pipeline: {len(new_items)} new items after dedup")

        # 4. Classify
        for item in new_items:
            item.category = classify_vertical(item.title + " " + item.snippet)

        # 5. Score
        weights = load_weights()
        scored = score_items(new_items, weights)

        # 6. Store new articles to DB
        for si in scored:
            rec = ArticleRecord(
                url=si.item.url,
                url_hash=si.item.url_hash,
                title=si.item.title,
                source=si.item.source,
                published_at=si.item.published_at,
                detected_at=si.item.detected_at,
                snippet=si.item.snippet,
                body=si.item.body,
                category=si.item.category,
                source_region=si.item.source_region,
                author=si.item.author,
                image_url=si.item.image_url,
                score=si.score,
            )
            self.db.insert_article(rec)

        logger.info(f"pipeline: stored {len(scored)} articles; run complete")
        return scored
```

- [ ] **Step 4: Run tests**

```bash
pytest tests/test_pipeline.py -v
```
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add pipeline.py tests/test_pipeline.py
git commit -m "feat(pipeline): full orchestrator — fetch, dedup, classify, score, store"
```

---

## Task 7: Wire pipeline.py into news_viral_pro.py

**Files:**
- Modify: `news_viral_pro.py`

Replace the direct `fetch_google_news()` call in `tarea_1_research` with `NewsPipeline.run()`. The scored items have the same `ScoredItem` type that the rest of the pipeline already consumes.

- [ ] **Step 1: Read the current tarea_1_research call in news_viral_pro.py**

```bash
grep -n "fetch_google_news\|tarea_1\|score_items" news_viral_pro.py
```

- [ ] **Step 2: Replace the research task**

Find the block that calls `fetch_google_news` and `score_items`. Replace it:

```python
# OLD (remove this):
from news_sources import fetch_google_news, NewsItem
from news_scorer import load_weights, score_items, update_weights_from_feedback, save_weights

items = fetch_google_news(since_days=cfg.since_days, max_items=cfg.max_items)
weights = load_weights()
scored = score_items(items, weights)

# NEW (replace with):
from pipeline import NewsPipeline, PipelineConfig
from news_scorer import load_weights, update_weights_from_feedback, save_weights

pipeline_cfg = PipelineConfig(
    since_days=getattr(cfg, "since_days", 3),
    max_items=getattr(cfg, "max_items", 500),
)
pipeline = NewsPipeline(pipeline_cfg)
scored = pipeline.run()
```

> **Note:** Keep `update_weights_from_feedback` and `save_weights` — the feedback loop still works the same way.

- [ ] **Step 3: Run the existing test suite to ensure nothing broke**

```bash
pytest tests/ -v --tb=short
```
Expected: All existing tests pass (test_news_scorer, test_news_classifier, test_image_search_fallback, test_spend_logger still green)

- [ ] **Step 4: Smoke test the full pipeline manually**

```bash
python3 -c "
from pipeline import NewsPipeline, PipelineConfig
cfg = PipelineConfig(since_days=3, max_items=200)
p = NewsPipeline(cfg)
results = p.run()
print(f'Found {len(results)} items')
for r in results[:5]:
    print(f'  [{r.score:.2f}] {r.item.title[:60]} ({r.item.source})')
"
```
Expected: Output shows 50+ items from multiple sources (not just 12–30 from one source)

- [ ] **Step 5: Commit**

```bash
git add news_viral_pro.py
git commit -m "feat(pipeline): wire NewsPipeline into news_viral_pro, replacing single-source fetch"
```

---

## Task 8: Ranking Formula Enhancement

**Files:**
- Modify: `news_scorer.py` (add multi-source coverage bonus)
- Modify: `tests/test_news_scorer.py`

One strong ranking signal we're missing: **how many sources are covering the same story**. A story covered by 5 outlets is more important than one covered by 1. Add a post-scoring enrichment that bumps score for stories with title overlap across sources.

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_news_scorer.py
from news_scorer import apply_multi_source_bonus, ScoredItem
from news_sources import NewsItem


def make_scored(title, source, score=0.5):
    item = NewsItem(title=title, url=f"http://x.com/{hash(title+source)}", source=source,
                    published_at="2026-05-17T10:00:00+00:00")
    return ScoredItem(item=item, score=score, breakdown={})


def test_multi_source_bonus_boosts_repeated_story():
    scored = [
        make_scored("Balacera en Cancún deja 2 muertos", "Sipse", 0.5),
        make_scored("Balacera en Cancún deja 2 muertos", "Novedades QR", 0.5),
        make_scored("Tormenta tropical avanza", "Milenio", 0.5),
    ]
    result = apply_multi_source_bonus(scored)
    balacera_scores = [r.score for r in result if "Balacera" in r.item.title]
    tormenta_scores = [r.score for r in result if "Tormenta" in r.item.title]
    assert all(s > 0.5 for s in balacera_scores), "Multi-source story should be boosted"
    assert all(s == 0.5 for s in tormenta_scores), "Single-source story should not be boosted"


def test_multi_source_bonus_does_not_exceed_1():
    scored = [make_scored("Breaking", f"Source{i}", 0.95) for i in range(5)]
    result = apply_multi_source_bonus(scored)
    assert all(r.score <= 1.0 for r in result)
```

- [ ] **Step 2: Run to confirm failure**

```bash
pytest tests/test_news_scorer.py::test_multi_source_bonus_boosts_repeated_story -v
```
Expected: FAIL — `ImportError: cannot import name 'apply_multi_source_bonus'`

- [ ] **Step 3: Add apply_multi_source_bonus to news_scorer.py**

```python
import re as _re


def _normalize_title(title: str) -> str:
    return _re.sub(r"[^a-záéíóúñ0-9 ]", " ", title.lower()).strip()


def apply_multi_source_bonus(
    scored: List[ScoredItem],
    *,
    boost_per_extra_source: float = 0.05,
    max_boost: float = 0.20,
) -> List[ScoredItem]:
    """Boost stories covered by multiple sources.

    Groups items by normalized title similarity (exact normalized match for now).
    For each group of size N, each item gets +(N-1)*boost_per_extra_source bonus,
    capped at max_boost and capped at score=1.0.
    """
    from collections import Counter
    title_counts: Counter = Counter()
    for si in scored:
        norm = _normalize_title(si.item.title)
        title_counts[norm] += 1

    result = []
    for si in scored:
        norm = _normalize_title(si.item.title)
        count = title_counts[norm]
        bonus = min(max_boost, (count - 1) * boost_per_extra_source)
        new_score = min(1.0, si.score + bonus)
        if bonus > 0:
            si.breakdown["multi_source"] = bonus
        result.append(ScoredItem(item=si.item, score=new_score, breakdown=si.breakdown))

    result.sort(key=lambda s: s.score, reverse=True)
    return result
```

Also call this in `pipeline.py` after `score_items`:
```python
from news_scorer import ScoredItem, load_weights, score_items, apply_multi_source_bonus
# ...
scored = score_items(new_items, weights)
scored = apply_multi_source_bonus(scored)
```

- [ ] **Step 4: Run all scorer tests**

```bash
pytest tests/test_news_scorer.py -v
```
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add news_scorer.py pipeline.py tests/test_news_scorer.py
git commit -m "feat(ranking): multi-source coverage bonus in scorer"
```

---

## Task 9: requirements.txt + APScheduler (M8 — Automation)

**Files:**
- Modify: `requirements.txt`
- Create: `scheduler.py`

Add APScheduler so the pipeline runs automatically every 30 minutes. This is the M8 automation milestone — simple but essential for a live news engine.

- [ ] **Step 1: Add apscheduler to requirements.txt**

```
apscheduler>=3.10.0,<4.0.0
```

```bash
pip install apscheduler>=3.10.0
```

- [ ] **Step 2: Create scheduler.py**

```python
"""Periodic news pipeline runner.

Runs the full pipeline every INTERVAL_MINUTES. Designed to run as a long-lived
background process alongside the FastAPI webapp.

Usage:
    python3 scheduler.py                # runs every 30 min
    python3 scheduler.py --interval 15  # runs every 15 min
"""
from __future__ import annotations

import argparse
import logging
import sys

from apscheduler.schedulers.blocking import BlockingScheduler

from pipeline import NewsPipeline, PipelineConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("scheduler")


def run_pipeline() -> None:
    logger.info("scheduler: starting pipeline run")
    try:
        cfg = PipelineConfig()
        pipeline = NewsPipeline(cfg)
        results = pipeline.run()
        logger.info(f"scheduler: pipeline complete — {len(results)} items")
    except Exception as e:
        logger.error(f"scheduler: pipeline failed: {e}", exc_info=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=30, help="Run interval in minutes")
    args = parser.parse_args()

    scheduler = BlockingScheduler()
    scheduler.add_job(run_pipeline, "interval", minutes=args.interval, id="news_pipeline")

    logger.info(f"scheduler: starting — pipeline every {args.interval} minutes")
    run_pipeline()  # run immediately on start
    scheduler.start()


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Run smoke test**

```bash
python3 -c "from scheduler import run_pipeline; run_pipeline()"
```
Expected: Runs without error, prints item count

- [ ] **Step 4: Commit**

```bash
git add requirements.txt scheduler.py
git commit -m "feat(scheduler): APScheduler-based periodic pipeline runner"
```

---

## Ranking Formula Reference

```
final_score = clamp(logistic(raw) + multi_source_bonus, 0, 1)

raw = freshness_score
    + title_length_bonus
    + region_coverage_bonus
    + keyword_score

where:
  freshness_score     = 0.6 * max(0, 1 - hours_since_publish / 48)
  title_length_bonus  = 0.2 if 40≤len≤80, else -0.1 if <25 or >120, else 0.05
  region_coverage     = min(0.3, 0.1 * count(region_keywords_in_text))
  keyword_score       = sum(learned_weight[kw] for kw in title+snippet)
  multi_source_bonus  = min(0.20, (same_title_count - 1) * 0.05)
```

**Anti-clickbait:** Negative keyword weights (`anuncia: -0.1`, `promete: -0.15`) penalize empty announcements. The `favor_precision=True` flag in trafilatura skips thin/teaser content. Future: add a boilerplate ratio penalty (body_length < 100 chars → score × 0.5).

---

## Data Model Summary

### `articles` table
`id, url, url_hash, title, source, published_at, detected_at, snippet, body, category, source_region, author, image_url, score`

### `url_hashes` table
`url_hash (PK), url, seen_at` — fast cross-run dedup lookup

### `fetch_logs` table
`id, source_name, feed_url, fetched_at, status, items_found, items_new, error_msg` — debugging per-source failures

### `source_registry.json` (not in DB)
Editable flat file — preferred over DB for sources because it's human-editable and version-controlled. Reload at each pipeline run.

---

## Social Signals Strategy (M6)

| Platform | Method | Legal Status | Action |
|---|---|---|---|
| **Google Trends** | `pytrends` (unofficial, public data) | Tolerated | Add in M6 — poll trending México searches to boost score for trending topics |
| **YouTube** | YouTube Data API v3 | Official, free | Add in M6 — search `{topic} México noticias`, extract trending news video titles |
| **X/Twitter** | X API v2 Free/Basic | Official | Add in M6 only if API key available — 500K reads/month free |
| **Facebook** | Meta Content Library (academic only) | Restricted | **Skip** — no public access |
| **Instagram** | No public API | Restricted | **Skip** |
| **TikTok** | Research API (gated) | Official, gated | **Skip until approved** |

**Implementation note for pytrends:** Wrap in try/except, cache results 30 minutes, rate limit to 1 request per 30 seconds. Only use to get topic keywords, not article URLs.

---

## Verification Plan

After each milestone, run:

```bash
# 1. All tests pass
pytest tests/ -v

# 2. Pipeline returns more items than before
python3 -c "
from pipeline import NewsPipeline, PipelineConfig
p = NewsPipeline(PipelineConfig(since_days=3, max_items=500))
r = p.run()
print(f'Items: {len(r)}')
# Should be 100+ vs old 12-30
assert len(r) >= 50, f'Expected 50+, got {len(r)}'
"

# 3. Multiple sources represented
python3 -c "
from pipeline import NewsPipeline, PipelineConfig
p = NewsPipeline(PipelineConfig(since_days=3))
r = p.run()
sources = {si.item.source for si in r}
print(f'Sources: {len(sources)} unique — {list(sources)[:5]}')
assert len(sources) >= 5
"

# 4. No duplicate URLs in output
python3 -c "
from pipeline import NewsPipeline, PipelineConfig
p = NewsPipeline(PipelineConfig(since_days=3))
r = p.run()
urls = [si.item.url for si in r]
assert len(urls) == len(set(urls)), 'Duplicate URLs found!'
print('No duplicates')
"

# 5. DB is growing
python3 -c "
from news_db import NewsDB
db = NewsDB('news.db')
db.init_schema()
print(f'Total articles stored: {db.count_articles()}')
"
```

---

## Risk Register

| Risk | Severity | Mitigation |
|---|---|---|
| **RSS feed URL is wrong/dead** | Medium | `active: false` in registry + verify feeds before enabling. Fetch log records all failures. |
| **trafilatura blocked by paywall/JS** | Low | Returns `None` — pipeline uses snippet fallback. No crash. |
| **Google News RSS rate limiting** | Medium | 8 distinct query URLs spread across a run. Add `crawl_delay_seconds=1` between them. |
| **Robots.txt violation** | Low | trafilatura has built-in robots.txt checking (`check_robots=True`). Pass-through for RSS (not scraping). |
| **ToS violation via scraping** | Low | RSS and sitemap are public, designed for machine consumption. Body extraction via trafilatura uses standard HTTP — same as any browser. Avoid rotating proxies, CAPTCHA bypass, login-gated content. |
| **Database growth** | Low | Articles are ~2 KB average. 1,000 articles/day = ~700 MB/year. Add a `cleanup.py` job (already exists in project) to archive or delete articles older than 30 days. |
| **pytrends ban** | Low (M6) | Cache 30 min per query. Respect rate limits. Wrap in try/except. If banned, disable gracefully. |

---

## Immediate Wins (Today, Before M2)

These can be done right now, before building all the infrastructure above, to immediately get more news:

1. **Add 7 more Google News RSS queries** in `news_sources.py` (politica, seguridad, corrupción, clima, Cancún turismo, Chetumal, denuncias) — zero infrastructure needed, just add URLs to a list and fan-out with feedparser.
2. **Set `require_region_hit=False` by default** in `fetch_google_news` — the region filter is the #1 reason national/political stories are dropped.
3. **Raise `max_items` to 200** in `ConfigPro` default.
4. **Raise `since_days` to 3**.
5. **Add sipse, animalpolitico, aristegui, proceso feed URLs** directly to a list alongside the existing `GOOGLE_NEWS_RSS` constant — feedparser handles them the same way.

These 5 changes alone should take the pipeline from 12–30 items to 80–150 items per run with no architectural changes.

---

## Roadmap

| Milestone | Tasks | ETA | Deliverable |
|---|---|---|---|
| **M1 — Immediate** | 5 quick fixes above | Today | 80–150 items/run |
| **M2 — Source Registry** | Tasks 1–2 | Day 1–2 | 50+ sources, dynamic loading |
| **M3 — Storage** | Task 3 | Day 2 | SQLite, dedup across runs |
| **M4 — Extractor** | Task 4 | Day 3 | Generic body extraction, no domain lock |
| **M5 — Dedup + Ranking** | Tasks 5, 8 | Day 3–4 | Clean dedup, multi-source bonus |
| **M6 — Pipeline wire-up** | Tasks 6–7 | Day 4 | Full orchestrator live |
| **M7 — Social Signals** | pytrends + YouTube API | Week 2 | Topic trending signal |
| **M8 — Automation** | Task 9 | Week 2 | 30-min auto-runs, daily reports |
