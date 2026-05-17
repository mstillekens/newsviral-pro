"""Unit tests for NewsEnrichmentSystem.

All HTTP and Anthropic calls are mocked. No network. No API keys.
"""
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Make the repo root importable when running `pytest` from any CWD.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from news_sources import NewsItem  # noqa: E402
import news_enrichment as ne  # noqa: E402


# ---------- Fixtures ----------

def make_item(
    title: str = "Tourists stranded after Cancún storm",
    url: str = "https://example.com/article-1",
) -> NewsItem:
    return NewsItem(
        title=title,
        url=url,
        source="Example",
        published_at="2026-05-17T10:00:00+00:00",
        snippet="A storm hit the coast leaving travelers stuck.",
        body="The storm hit on Monday. Hotels reported full occupancy.",
        region_hits=["Cancún"],
    )


def make_anthropic_returning(text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [SimpleNamespace(text=text)]
    client.messages.create.return_value = msg
    return client


class FakeResp:
    def __init__(self, text: str = "", json_data=None, status: int = 200, url: str = "x"):
        self.text = text
        self._json = json_data
        self.status_code = status
        self.url = url

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._json


# ---------- Pure helpers ----------

def test_jaccard():
    a = {"foo", "bar"}
    b = {"foo", "baz"}
    assert 0.3 < ne._jaccard(a, b) < 0.34


def test_tokenize_strips_stopwords():
    toks = ne._tokenize_title("El gato y la luna sobre Cancún")
    assert "cancún" in toks
    assert "luna" in toks
    assert "el" not in toks
    assert "la" not in toks


def test_canonical_url_strips_www_and_trailing_slash():
    assert ne._canonical_url("https://www.example.com/path/") == "example.com/path"
    assert ne._canonical_url("https://example.com/path") == "example.com/path"


def test_credibility_known_and_unknown():
    assert ne._credibility("https://bbc.com/news/x") == 1.0
    assert 0.0 <= ne._credibility("https://random-blog.xyz/x") <= 0.5


def test_extract_json_handles_fences_and_prefixes():
    raw = "```json\n{\"k\": 1}\n```"
    assert ne._extract_json(raw) == {"k": 1}
    raw2 = "Sure, here:\n{\"a\": [1,2], \"b\": {\"x\": 3}}\nbye"
    assert ne._extract_json(raw2) == {"a": [1, 2], "b": {"x": 3}}


# ---------- Dedupe ----------

def test_dedupe_sources_removes_url_duplicates_and_similar_titles():
    sys_ = ne.NewsEnrichmentSystem(MagicMock())
    srcs = [
        ne.SourceRef(url="https://a.com/x", outlet="a", title="Storm hits Cancún today"),
        ne.SourceRef(url="https://www.a.com/x/", outlet="a", title="Storm hits Cancún today"),
        ne.SourceRef(url="https://b.com/y", outlet="b", title="Cancún storm hits today"),
        ne.SourceRef(url="https://c.com/z", outlet="c", title="Volcano erupts in Iceland"),
    ]
    kept = sys_._dedupe_sources(srcs)
    urls = [s.url for s in kept]
    assert "https://a.com/x" in urls
    assert "https://www.a.com/x/" not in urls  # canonical dup
    assert "https://b.com/y" not in urls  # jaccard dup
    assert "https://c.com/z" in urls
    assert len(kept) == 2


def test_dedupe_images_dedupes_by_url_and_flags_ai():
    sys_ = ne.NewsEnrichmentSystem(MagicMock())
    imgs = [
        ne.NewsImage(url="https://x.com/a.jpg", source_article="https://x.com"),
        ne.NewsImage(url="https://x.com/a.jpg", source_article="https://x.com"),
        ne.NewsImage(url="https://midjourney.example/y.png", source_article="https://x.com"),
    ]
    kept = sys_._dedupe_images(imgs)
    assert len(kept) == 2
    assert kept[1].flagged_ai is True


def test_score_images_sorts_desc_and_penalizes_ai():
    sys_ = ne.NewsEnrichmentSystem(MagicMock())
    imgs = [
        ne.NewsImage(url="https://x.com/1.jpg", source_article="https://bbc.com/", width=1920, height=1080),
        ne.NewsImage(url="https://midjourney.example/2.png", source_article="https://bbc.com/", width=1920, height=1080, flagged_ai=True),
        ne.NewsImage(url="https://x.com/3.jpg", source_article="https://random.xyz/", width=800, height=600),
    ]
    scored = sys_._score_images(imgs)
    assert scored[0].url == "https://x.com/1.jpg"
    assert scored[-1].flagged_ai is True


# ---------- Phase 5 quality scoring ----------

def test_phase5_perfect_score_passes():
    sys_ = ne.NewsEnrichmentSystem(MagicMock())
    result = ne.EnrichedNews(item=make_item())
    result.sources = [ne.SourceRef(url=f"https://s{i}.com", outlet="x", title="t") for i in range(7)]
    result.facts = [ne.VerifiedFact(text=f"f{i}", supporting_source_indices=[0, 1]) for i in range(5)]
    result.brief = " ".join(["palabra"] * 1350)
    result.images = [ne.NewsImage(url=f"https://i{i}.com/x.jpg", source_article="x") for i in range(8)]
    sys_._phase5_validate_quality(result)
    assert result.quality_score == 100
    assert result.passed is True


def test_phase5_partial_score_fails():
    sys_ = ne.NewsEnrichmentSystem(MagicMock())
    result = ne.EnrichedNews(item=make_item())
    result.sources = [ne.SourceRef(url=f"https://s{i}.com", outlet="x", title="t") for i in range(3)]
    result.facts = [ne.VerifiedFact(text="f1", supporting_source_indices=[0])]
    result.brief = " ".join(["palabra"] * 300)
    result.images = [ne.NewsImage(url="https://i.com/x.jpg", source_article="x")]
    result.errors = ["phase2_timeout"]
    sys_._phase5_validate_quality(result)
    assert result.quality_score < 70
    assert result.passed is False
    assert result.quality_breakdown["sources"] < 25
    assert result.quality_breakdown["errors"] < 10


# ---------- Phase 2 with mocked Anthropic ----------

def test_phase2_filters_facts_with_less_than_two_sources():
    client = make_anthropic_returning(json.dumps({
        "facts": [
            {"text": "Storm hit Cancún", "supporting": [0, 1, 2], "confidence": 0.9},
            {"text": "5 people stranded", "supporting": [0], "confidence": 0.6},
            {"text": "Hotels at capacity", "supporting": [1, 2], "confidence": 0.8},
        ]
    }))
    sys_ = ne.NewsEnrichmentSystem(client)

    item = make_item()
    sources = [
        ne.SourceRef(url=f"https://s{i}.com", outlet="x", title=f"t{i}", body="body text " * 30)
        for i in range(3)
    ]

    fake_http = AsyncMock()
    facts = asyncio.run(sys_._phase2_extract_verified_facts(fake_http, item, sources))
    assert len(facts) == 2
    assert all(len(f.supporting_source_indices) >= 2 for f in facts)


def test_phase2_keeps_single_source_fact_if_confidence_high():
    client = make_anthropic_returning(json.dumps({
        "facts": [
            {"text": "Direct quote from official", "supporting": [0], "confidence": 0.95}
        ]
    }))
    sys_ = ne.NewsEnrichmentSystem(client)
    sources = [ne.SourceRef(url="https://s.com", outlet="x", title="t", body="text " * 30)]
    facts = asyncio.run(sys_._phase2_extract_verified_facts(AsyncMock(), make_item(), sources))
    assert len(facts) == 1


# ---------- Phase 3 brief generation ----------

def test_phase3_returns_brief_and_scenes():
    brief_text = " ".join(["palabra"] * 1350)
    client = make_anthropic_returning(json.dumps({
        "brief": brief_text,
        "scenes": {
            "escena_1": "Te cuento que un evento tremendo paso hoy en la costa caribe",
            "escena_2": "Las imagenes muestran turistas atrapados en hoteles repletos durante la tormenta",
            "escena_3": "Y aqui entre nos queda una pregunta sobre la temporada",
        }
    }))
    sys_ = ne.NewsEnrichmentSystem(client)
    brief, scenes = asyncio.run(sys_._phase3_rewrite_with_llm(make_item(), [], []))
    assert 1200 <= len(brief.split()) <= 1500
    assert set(scenes.keys()) == {"escena_1", "escena_2", "escena_3"}


def test_phase3_retries_when_brief_out_of_range():
    short_brief = " ".join(["palabra"] * 500)
    long_brief = " ".join(["palabra"] * 1300)
    responses = [
        json.dumps({"brief": short_brief, "scenes": {}}),
        json.dumps({"brief": long_brief, "scenes": {
            "escena_1": "a", "escena_2": "b", "escena_3": "c"
        }}),
    ]
    client = MagicMock()
    msgs = []
    for r in responses:
        m = MagicMock()
        m.content = [SimpleNamespace(text=r)]
        msgs.append(m)
    client.messages.create.side_effect = msgs

    sys_ = ne.NewsEnrichmentSystem(client)
    brief, scenes = asyncio.run(sys_._phase3_rewrite_with_llm(make_item(), [], []))
    assert 1200 <= len(brief.split()) <= 1500
    assert client.messages.create.call_count == 2


# ---------- Phase 4 LLM image picking ----------

def test_llm_pick_images_falls_back_when_llm_fails():
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("rate limited")
    sys_ = ne.NewsEnrichmentSystem(client)
    imgs = [
        ne.NewsImage(url=f"https://x.com/{i}.jpg", source_article="https://bbc.com/",
                     width=1920, height=1080, score=1.0 - i * 0.1)
        for i in range(6)
    ]
    urls = asyncio.run(sys_._llm_pick_images(imgs, make_item()))
    assert len(urls) == 3
    assert urls[0] == "https://x.com/0.jpg"


def test_llm_pick_images_uses_llm_selection():
    client = make_anthropic_returning(json.dumps({"selected": [4, 2, 0]}))
    sys_ = ne.NewsEnrichmentSystem(client)
    imgs = [
        ne.NewsImage(url=f"https://x.com/{i}.jpg", source_article="https://bbc.com/", width=1920, height=1080)
        for i in range(6)
    ]
    urls = asyncio.run(sys_._llm_pick_images(imgs, make_item()))
    assert urls == ["https://x.com/4.jpg", "https://x.com/2.jpg", "https://x.com/0.jpg"]


# ---------- Integration: full enrich() with all networks/LLM mocked ----------

def test_enrich_end_to_end_happy_path(monkeypatch):
    rss_xml = """<?xml version="1.0"?>
<rss><channel>
<item><title>Storm hits Cancún - BBC</title>
<link>https://bbc.com/storm</link>
<description>A storm has hit the coast</description>
<pubDate>Mon, 17 May 2026 10:00:00 GMT</pubDate>
</item>
<item><title>Cancún storm leaves tourists stranded - Reuters</title>
<link>https://reuters.com/storm</link>
<description>Travelers stranded</description>
<pubDate>Mon, 17 May 2026 11:00:00 GMT</pubDate>
</item>
</channel></rss>"""

    article_html = """
<html><head>
<meta property="og:image" content="https://bbc.com/img/hero.jpg">
<meta name="twitter:image" content="https://bbc.com/img/twitter.jpg">
</head><body><article>
<p>This is the body of the article. It has enough text to count as content.</p>
<p>Second paragraph with more details about the storm hitting Cancún.</p>
<img src="https://bbc.com/img/article-1.jpg" width="800" height="600">
<img src="https://bbc.com/img/article-2.jpg" width="1024" height="768">
</article></body></html>
"""

    reddit_json = {
        "data": {
            "children": [
                {"data": {
                    "url_overridden_by_dest": "https://news-blog.com/storm",
                    "title": "Storm in Cancún - unique angle",
                    "selftext": "Some discussion",
                    "created_utc": 1715900000,
                    "permalink": "/r/news/1",
                }},
                {"data": {
                    "url_overridden_by_dest": "https://other-news.com/storm",
                    "title": "Hurricane Cancún tourists trapped",
                    "selftext": "",
                    "created_utc": 1715900100,
                    "permalink": "/r/news/2",
                }},
            ]
        }
    }

    async def fake_aget(client, url, *, timeout=8.0, headers=None):
        if "news.google.com" in url:
            return FakeResp(text=rss_xml, url=url)
        if "reddit.com" in url:
            return FakeResp(json_data=reddit_json, url=url)
        if "feeds." in url or "rss" in url:
            return FakeResp(text=rss_xml, url=url)
        if any(host in url for host in ("bbc.com", "reuters.com", "news-blog.com", "other-news.com")):
            return FakeResp(text=article_html, url=url)
        return None

    monkeypatch.setattr(ne, "_aget", fake_aget)

    facts_resp = json.dumps({
        "facts": [
            {"text": "Storm hit Cancún", "supporting": [0, 1], "confidence": 0.9},
            {"text": "Tourists stranded in hotels", "supporting": [0, 2], "confidence": 0.85},
            {"text": "Hotels at full occupancy", "supporting": [1, 2], "confidence": 0.8},
            {"text": "Authorities issued warning", "supporting": [0, 1], "confidence": 0.9},
            {"text": "Storm dissipated by Tuesday", "supporting": [0, 2], "confidence": 0.85},
        ]
    })
    brief_resp = json.dumps({
        "brief": " ".join(["palabra"] * 1300),
        "scenes": {
            "escena_1": "Te cuento lo que pasa en la costa caribe ahora mismo amigos miren bien",
            "escena_2": "Las imagenes del temporal muestran turistas atrapados en hoteles llenos sin salida",
            "escena_3": "Y aqui entre nos queda la pregunta sobre la siguiente alerta climatica de la temporada",
        }
    })
    image_resp = json.dumps({"selected": [0, 1, 2]})

    client = MagicMock()
    responses = [facts_resp, brief_resp, image_resp]
    msgs = []
    for r in responses:
        m = MagicMock()
        m.content = [SimpleNamespace(text=r)]
        msgs.append(m)
    client.messages.create.side_effect = msgs

    sys_ = ne.NewsEnrichmentSystem(client)
    result = asyncio.run(sys_.enrich(make_item()))

    # Sources from RSS + Reddit + seed (original) — should easily exceed minimum
    assert len(result.sources) >= 3  # may be deduped below 7 with limited fixture
    # Facts from mocked Claude (all have ≥2 supports)
    assert len(result.facts) == 5
    # Brief in range
    assert 1200 <= len(result.brief.split()) <= 1500
    # Scenes present
    assert set(result.scenes.keys()) == {"escena_1", "escena_2", "escena_3"}
    # Images scraped from articles
    assert len(result.images) >= 3
    # Quality breakdown has all keys
    assert set(result.quality_breakdown.keys()) == {"sources", "facts", "brief", "images", "errors"}
    # 3 calls to Claude (facts, brief, image-pick)
    assert client.messages.create.call_count == 3


# ---------- Public aggregator (no LLM) ----------

def test_cluster_sources_by_title_groups_similar_and_keeps_distinct():
    srcs = [
        ne.SourceRef(url="https://bbc.com/a", outlet="rss:bbc",
                     title="Storm hits Cancún tourists stranded"),
        ne.SourceRef(url="https://reuters.com/b", outlet="rss:reuters",
                     title="Cancún storm strands many tourists"),
        ne.SourceRef(url="https://apnews.com/c", outlet="rss:ap",
                     title="Tourists stranded Cancún storm"),
        ne.SourceRef(url="https://other.com/d", outlet="rss:other",
                     title="Volcano erupts in Iceland today"),
    ]
    clusters = ne.cluster_sources_by_title(srcs, jaccard_threshold=0.35)
    assert len(clusters) == 2
    big = next(c for c in clusters if c.source_count > 1)
    assert big.source_count == 3
    # Primary should be the most credible (bbc / reuters / ap are all top)
    assert big.primary.url in {
        "https://bbc.com/a", "https://reuters.com/b", "https://apnews.com/c"
    }


def test_cluster_credibility_is_average():
    srcs = [
        ne.SourceRef(url="https://bbc.com/a", outlet="rss:bbc", title="Same event same vocab"),
        ne.SourceRef(url="https://random-blog.xyz/b", outlet="rss:blog", title="Same event same vocab"),
    ]
    clusters = ne.cluster_sources_by_title(srcs, jaccard_threshold=0.3)
    assert len(clusters) == 1
    c = clusters[0]
    assert c.source_count == 2
    assert 0.3 < c.credibility < 1.0


def test_aggregate_news_clusters_orchestrates_and_dedupes(monkeypatch):
    rss_xml = """<?xml version="1.0"?>
<rss><channel>
<item><title>Storm hits Cancún - BBC</title>
<link>https://bbc.com/storm</link>
<description>x</description><pubDate>Mon, 17 May 2026 10:00:00 GMT</pubDate></item>
<item><title>Cancún storm — Reuters</title>
<link>https://www.bbc.com/storm/</link>
<description>y</description><pubDate>Mon, 17 May 2026 11:00:00 GMT</pubDate></item>
</channel></rss>"""

    reddit_json = {"data": {"children": [
        {"data": {
            "url_overridden_by_dest": "https://other.com/storm",
            "title": "Storm Cancún tourists stuck",
            "permalink": "/r/news/1",
            "created_utc": 1715900000,
        }},
    ]}}

    async def fake_aget(client, url, *, timeout=8.0, headers=None):
        if "news.google.com" in url:
            return FakeResp(text=rss_xml, url=url)
        if "reddit.com" in url:
            return FakeResp(json_data=reddit_json, url=url)
        return None

    monkeypatch.setattr(ne, "_aget", fake_aget)

    clusters = asyncio.run(ne.aggregate_news_clusters(
        "Cancún storm", intl_rss=False, reddit=True, timeout_total=5.0,
    ))
    # Same canonical URL "bbc.com/storm" appears twice in RSS — should dedupe.
    all_urls = {m.url for c in clusters for m in c.members}
    assert "https://bbc.com/storm" in all_urls
    # And the cluster grouping pulls in the Reddit-mirrored other.com link.
    flat_count = sum(c.source_count for c in clusters)
    assert flat_count >= 2
