"""Unit tests for political_filter — keyword scan, cache, batch LLM, modes.

All Anthropic calls are mocked. No network.
"""
from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import political_filter as pf  # noqa: E402


# ---------- helpers ----------

def make_anthropic(text: str) -> MagicMock:
    client = MagicMock()
    msg = MagicMock()
    msg.content = [SimpleNamespace(text=text)]
    client.messages.create.return_value = msg
    return client


@pytest.fixture
def rules_path() -> Path:
    return ROOT / "config" / "political_keywords.yaml"


@pytest.fixture
def tmp_cache(tmp_path) -> Path:
    return tmp_path / "cache.json"


# ---------- _norm + keyword scan ----------

def test_norm_strips_diacritics_and_case():
    assert pf._norm("Cárteles del Pacífico") == "carteles del pacifico"


def test_keyword_scan_blocks_narco():
    rules = pf.PoliticalRules.from_yaml(ROOT / "config" / "political_keywords.yaml")
    decision, review = rules.keyword_scan(
        "Cae cártel del CJNG en Cancún tras enfrentamiento"
    )
    assert decision is not None
    assert decision.verdict == "block"
    assert decision.decided_by == "keyword"
    assert any("cjng" in k or "cartel" in k for k in decision.matched_keywords)


def test_keyword_scan_blocks_violence():
    rules = pf.PoliticalRules.from_yaml(ROOT / "config" / "political_keywords.yaml")
    decision, _ = rules.keyword_scan("Hallan cadáver tras balacera en Tulum")
    assert decision is not None
    assert decision.verdict == "block"


def test_keyword_scan_returns_review_hits_only():
    rules = pf.PoliticalRules.from_yaml(ROOT / "config" / "political_keywords.yaml")
    decision, review = rules.keyword_scan(
        "Migrantes denuncian condiciones en frontera de Quintana Roo"
    )
    # No block keyword → block decision is None
    assert decision is None
    # 'migrantes' is in review keywords
    assert any("migrant" in r for r in review)


def test_keyword_scan_allows_service_news():
    rules = pf.PoliticalRules.from_yaml(ROOT / "config" / "political_keywords.yaml")
    decision, review = rules.keyword_scan(
        "Mara Lezama inaugura nuevo hospital en Cancún"
    )
    assert decision is None
    assert review == []  # nothing flagged


def test_keyword_word_boundary_prevents_false_positives():
    rules = pf.PoliticalRules.from_yaml(ROOT / "config" / "political_keywords.yaml")
    # "Suiza" contains "suiz" but not the standalone "suicidio"
    decision, _ = rules.keyword_scan("Turistas suizos visitan Bacalar este verano")
    assert decision is None


# ---------- FilterCache ----------

def test_cache_round_trip(tmp_cache):
    cache = pf.FilterCache(tmp_cache, ttl_seconds=60)
    d = pf.FilterDecision(verdict="block", category="sensitive",
                          confidence=95, reason="narco", matched_keywords=["cartel"])
    cache.put("https://example.com/a", d)
    cache.flush()

    cache2 = pf.FilterCache(tmp_cache, ttl_seconds=60)
    got = cache2.get("https://example.com/a")
    assert got is not None
    assert got.verdict == "block"
    assert got.decided_by == "cache"


def test_cache_expires(tmp_cache):
    cache = pf.FilterCache(tmp_cache, ttl_seconds=0)
    cache.put("https://x.com/y", pf.FilterDecision(
        verdict="allow", category="service", confidence=100, reason="ok",
    ))
    cache.flush()
    time.sleep(0.05)
    cache2 = pf.FilterCache(tmp_cache, ttl_seconds=0)
    assert cache2.get("https://x.com/y") is None


def test_cache_canonicalizes_url(tmp_cache):
    cache = pf.FilterCache(tmp_cache, ttl_seconds=60)
    cache.put("https://www.example.com/a/", pf.FilterDecision(
        verdict="allow", category="service", confidence=100, reason="ok",
    ))
    assert cache.get("https://example.com/a") is not None
    assert cache.get("https://www.example.com/a") is not None


# ---------- PoliticalFilter.batch_filter ----------

def _build_filter(client, tmp_cache, mode="strict") -> pf.PoliticalFilter:
    cfg = pf.PoliticalFilterConfig(enabled=True, mode=mode,
                                   model="claude-haiku-4-5",
                                   confidence_threshold=70)
    return pf.PoliticalFilter(
        client,
        rules_path=ROOT / "config" / "political_keywords.yaml",
        cache_path=tmp_cache,
        config=cfg,
    )


def test_batch_keyword_block_skips_llm(tmp_cache):
    client = make_anthropic("{}")
    f = _build_filter(client, tmp_cache)
    items = [
        pf.ItemSummary(url="https://x.com/1",
                       title="Cae cártel CJNG en Cancún", snippet=""),
        pf.ItemSummary(url="https://x.com/2",
                       title="Hallan cadáver tras balacera", snippet=""),
    ]
    decisions = asyncio.run(f.batch_filter(items))
    assert all(d.verdict == "block" for d in decisions.values())
    # Both blocked via keyword — no LLM call needed
    assert client.messages.create.call_count == 0


def test_batch_llm_decides_service_passes(tmp_cache):
    llm_response = json.dumps({"results": [
        {"idx": 0, "category": "service", "confidence": 92, "reason": "weather"},
        {"idx": 1, "category": "service", "confidence": 88, "reason": "hospital opening"},
    ]})
    client = make_anthropic(llm_response)
    f = _build_filter(client, tmp_cache)
    items = [
        pf.ItemSummary(url="https://x.com/1",
                       title="Clima en Cancún hoy", snippet="Sol y 32 grados"),
        pf.ItemSummary(url="https://x.com/2",
                       title="Gobernadora abre nuevo hospital regional",
                       snippet="Inversión de 200 millones"),
    ]
    decisions = asyncio.run(f.batch_filter(items))
    assert all(d.verdict == "allow" for d in decisions.values())
    assert client.messages.create.call_count == 1


def test_batch_llm_categories_political_to_review(tmp_cache):
    llm_response = json.dumps({"results": [
        {"idx": 0, "category": "political", "confidence": 85, "reason": "campaign"},
    ]})
    client = make_anthropic(llm_response)
    f = _build_filter(client, tmp_cache, mode="strict")
    items = [pf.ItemSummary(url="https://x.com/1",
                            title="MORENA presenta candidato a presidencia municipal",
                            snippet="")]
    decisions = asyncio.run(f.batch_filter(items))
    # strict mode: political → REVIEW (not block)
    assert decisions["https://x.com/1"].verdict == "review"


def test_batch_llm_sensitive_high_conf_blocks(tmp_cache):
    llm_response = json.dumps({"results": [
        {"idx": 0, "category": "sensitive", "confidence": 92, "reason": "violence"},
    ]})
    client = make_anthropic(llm_response)
    f = _build_filter(client, tmp_cache)
    items = [pf.ItemSummary(url="https://x.com/1",
                            title="Some borderline news", snippet="")]
    decisions = asyncio.run(f.batch_filter(items))
    assert decisions["https://x.com/1"].verdict == "block"


def test_batch_uses_cache_on_second_call(tmp_cache):
    llm_response = json.dumps({"results": [
        {"idx": 0, "category": "service", "confidence": 90, "reason": "ok"},
    ]})
    client = make_anthropic(llm_response)
    f = _build_filter(client, tmp_cache)
    items = [pf.ItemSummary(url="https://x.com/1", title="Clima soleado", snippet="")]
    asyncio.run(f.batch_filter(items))
    asyncio.run(f.batch_filter(items))
    # Second call should hit cache, no extra LLM invocation
    assert client.messages.create.call_count == 1


def test_disabled_mode_allows_everything(tmp_cache):
    cfg = pf.PoliticalFilterConfig(enabled=False, mode="off")
    client = make_anthropic("{}")
    f = pf.PoliticalFilter(
        client,
        rules_path=ROOT / "config" / "political_keywords.yaml",
        cache_path=tmp_cache,
        config=cfg,
    )
    # Even with a hard block keyword, disabled filter allows
    items = [pf.ItemSummary(url="https://x.com/1",
                            title="Cae cártel CJNG", snippet="")]
    decisions = asyncio.run(f.batch_filter(items))
    assert decisions["https://x.com/1"].verdict == "allow"
    assert client.messages.create.call_count == 0


def test_llm_failure_falls_back_to_allow(tmp_cache):
    client = MagicMock()
    client.messages.create.side_effect = RuntimeError("anthropic down")
    f = _build_filter(client, tmp_cache)
    items = [pf.ItemSummary(url="https://x.com/1",
                            title="Some news without block keywords", snippet="")]
    decisions = asyncio.run(f.batch_filter(items))
    # LLM fallback → still allow, but decided_by=fallback
    assert decisions["https://x.com/1"].verdict == "allow"
    assert decisions["https://x.com/1"].decided_by == "fallback"


# ---------- apply_filter_to_clusters integration shim ----------

def _make_cluster(url: str, title: str):
    primary = SimpleNamespace(url=url, title=title, outlet="x", snippet="")
    return SimpleNamespace(primary=primary, members=[primary], source_count=1)


def test_apply_filter_drops_blocked_caps_review_keeps_allow():
    clusters = [
        _make_cluster("https://a.com/1", "Clima soleado"),
        _make_cluster("https://b.com/2", "Cae cártel"),
        _make_cluster("https://c.com/3", "Migrantes denuncian"),
    ]
    decisions = {
        "https://a.com/1": pf.FilterDecision(verdict="allow", category="service",
                                              confidence=90, reason="ok"),
        "https://b.com/2": pf.FilterDecision(verdict="block", category="sensitive",
                                              confidence=100, reason="narco"),
        "https://c.com/3": pf.FilterDecision(verdict="review", category="political",
                                              confidence=80, reason="sensitive"),
    }
    stats = pf.FilterStats()
    kept = pf.apply_filter_to_clusters(clusters, decisions, score_cap_review=59, stats=stats)
    urls = [c.primary.url for c in kept]
    assert "https://a.com/1" in urls
    assert "https://b.com/2" not in urls       # blocked
    assert "https://c.com/3" in urls           # review → kept with cap
    review_cluster = [c for c in kept if c.primary.url == "https://c.com/3"][0]
    assert getattr(review_cluster, "_score_cap", None) == 59
    assert stats.blocked == 1
    assert stats.review == 1
    assert stats.allowed == 1


# ---------- _extract_json robustness ----------

def test_extract_json_handles_fenced_array():
    text = "```json\n{\"results\": [{\"idx\": 0, \"category\": \"service\"}]}\n```"
    assert pf._extract_json(text) == {"results": [{"idx": 0, "category": "service"}]}


def test_extract_json_handles_prefix_garbage():
    text = "Here's my answer:\n{\"results\": []}\nThanks!"
    assert pf._extract_json(text) == {"results": []}
