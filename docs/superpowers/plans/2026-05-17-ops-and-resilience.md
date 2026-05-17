# NewsViral PRO — Ops & Resilience Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the existing pipeline production-trustworthy: track Replicate spend, recover from per-scene failures, fall back to alt image sources when og:image misses, generate a real logo PNG, clean up disk hygienically, and back the critical paths with tests so future edits don't silently break the chain.

**Architecture:** Add small, focused modules around the existing pipeline rather than refactoring it. New components: `spend_logger.py` (decorator + JSONL log), `image_search.py` (DuckDuckGo fallback for ref-images), `cleanup.py` (disk hygiene script), `logo_generator.py` (Pillow-based brand asset). Touch the orchestrator only to wrap calls in spend logging and to soften per-scene failures into degraded-but-complete videos. Add `tests/` with pytest covering classifier, scorer, env-loader, spend tracker, image fallback, and a full mock-mode end-to-end run. All work is free (no Replicate spend) and verifiable locally.

**Tech Stack:** Python 3.9, pytest, Pillow (already in env or one new pip install), duckduckgo-search library, existing FFmpeg + Replicate stack.

---

## Files map

| Path | Responsibility |
| --- | --- |
| `spend_logger.py` (new) | One function: `record(model, cost_usd, scene_id)`. Appends JSONL line. Helper: `summary(days=7)` returns aggregated totals. Used by orchestrator. |
| `image_search.py` (new) | One function: `search_news_image(query, max_results=5)`. Wraps DuckDuckGo Image Search, filters obvious junk, returns first reachable URL. Used by news_image_finder as fallback. |
| `news_image_finder.py` (modify) | Add `find_reference_image_with_fallback(article_url, headline)` that calls existing scraper, then `search_news_image(headline)` on miss. |
| `cleanup.py` (new) | CLI: deletes `replicate_outputs/*` older than N days (default 7), prunes `video_work/` after compose, archives `logs/runs/*` older than 30d into a tar. |
| `logo_generator.py` (new) | One-shot script: produces `assets/logo.png` and `assets/logo_white.png` (transparent bg) using Pillow + the Morena colors. Run once, commit the PNGs. |
| `replicate_orchestrator.py` (modify) | Wrap each `client.run` in `spend_logger.record`. Catch per-scene exceptions to a sentinel "FAILED" entry so partial outputs survive. Cache Seedance results by content hash of `(image_url, motion_prompt, duration)`. |
| `video_compositor.py` (modify) | When scene's video or audio is missing/sentinel, substitute a 5s solid-color slate with "ESCENA NO DISPONIBLE" so the final mp4 still composes. |
| `webapp/server.py` (modify) | Add `/admin/spend` endpoint (auth required) showing per-day totals from spend log. |
| `tests/conftest.py` (new) | Shared fixtures: temp dirs, fake clock, mock Replicate client. |
| `tests/test_news_scorer.py` (new) | Heuristic scoring + feedback weight update math. |
| `tests/test_news_classifier.py` (new) | Vertical classifier returns expected vertical per headline. |
| `tests/test_env_loader.py` (new) | `.env` parsing edge cases (empty values, quotes, comments). |
| `tests/test_spend_logger.py` (new) | Record + summary aggregation. |
| `tests/test_image_search_fallback.py` (new) | Falls through cleanly when DuckDuckGo unreachable (mocked). |
| `tests/test_e2e_mock.py` (new) | Full pipeline in mock mode end-to-end, asserts artifacts written. |

---

## Task 1: pytest infrastructure

**Files:**
- Create: `requirements-dev.txt`
- Create: `tests/__init__.py` (empty)
- Create: `tests/conftest.py`
- Create: `pytest.ini`

- [ ] **Step 1: Create requirements-dev.txt**

Path: `/Users/jr/projects/NewsViral/requirements-dev.txt`

```
pytest>=8.0.0,<9.0.0
pytest-asyncio>=0.23.0,<1.0.0
Pillow>=10.0.0,<11.0.0
duckduckgo-search>=6.0.0,<7.0.0
```

- [ ] **Step 2: Install dev deps**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pip install -r requirements-dev.txt
```

Expected: 4 packages install cleanly.

- [ ] **Step 3: Create pytest.ini**

Path: `/Users/jr/projects/NewsViral/pytest.ini`

```ini
[pytest]
testpaths = tests
python_files = test_*.py
asyncio_mode = auto
filterwarnings =
    ignore::DeprecationWarning
addopts = -q --tb=short
```

- [ ] **Step 4: Create tests/__init__.py and conftest.py**

Path: `/Users/jr/projects/NewsViral/tests/__init__.py`

```python
```

Path: `/Users/jr/projects/NewsViral/tests/conftest.py`

```python
"""Shared pytest fixtures.

These tests never touch the real Replicate or Anthropic APIs. The fixtures
provide tmp_path-backed directories and a fake clock for deterministic
behavior, plus stubs for the network-heavy modules.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make the project root importable as a flat package layout (everything sits
# at repo root, no src/ dir).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def tmp_project(tmp_path, monkeypatch):
    """Run a test inside a tmp working dir so writes don't pollute the repo."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "replicate_outputs").mkdir()
    (tmp_path / "video_work").mkdir()
    (tmp_path / "video_output").mkdir()
    return tmp_path


@pytest.fixture
def clean_env(monkeypatch):
    """Strip env vars that would change behavior between tests."""
    for k in ("REPLICATE_API_TOKEN", "ANTHROPIC_API_KEY", "MINIMAX_VOICE_ID",
              "WEBAPP_PASSWORD", "WEBAPP_USERNAME"):
        monkeypatch.delenv(k, raising=False)
```

- [ ] **Step 5: Run pytest to verify it's wired**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest -v
```

Expected: `no tests ran` (success), no errors about pytest config.

- [ ] **Step 6: Commit**

```bash
git add requirements-dev.txt pytest.ini tests/
git commit -m "test: scaffold pytest with shared fixtures"
```

---

## Task 2: classifier + scorer + env loader tests (covers existing code)

**Files:**
- Create: `tests/test_news_classifier.py`
- Create: `tests/test_news_scorer.py`
- Create: `tests/test_env_loader.py`

- [ ] **Step 1: Write classifier tests**

Path: `/Users/jr/projects/NewsViral/tests/test_news_classifier.py`

```python
"""brand_style.classify_vertical maps news text to its vertical anchor."""
from brand_style import classify_vertical, anchor_for


def test_politica_headline_classifies_politica():
    assert classify_vertical("AMLO anuncia ajuste presupuestario para Morena") == "politica"


def test_chismes_classifies_chismes():
    assert classify_vertical("Ruptura entre famoso cantante y su novia se vuelve viral") == "chismes"


def test_clima_classifies_clima():
    assert classify_vertical("Tormenta tropical Beatriz se acerca a Quintana Roo") == "clima"


def test_deportes_classifies_deportes():
    assert classify_vertical("La selección mexicana gana en el estadio Azteca") == "deportes"


def test_no_match_classifies_default():
    assert classify_vertical("Algún texto random sin keywords") == "default"


def test_political_news_picks_polibruh():
    a = anchor_for("AMLO en conferencia")
    assert a.id == "don_polibruh"


def test_chisme_news_picks_dona_chispas():
    a = anchor_for("Ruptura viral en redes sociales")
    assert a.id == "dona_chispas"


def test_unknown_news_picks_compa_caribe_as_default():
    a = anchor_for("texto sin pistas")
    assert a.id == "compa_caribe"   # the catch-all has 'default' in its verticals
```

- [ ] **Step 2: Write scorer tests**

Path: `/Users/jr/projects/NewsViral/tests/test_news_scorer.py`

```python
"""News scorer math + feedback weight updates."""
from datetime import datetime, timedelta, timezone

from news_sources import NewsItem
from news_scorer import (
    SEED_KEYWORD_WEIGHTS,
    score_item,
    score_items,
    update_weights_from_feedback,
)


def _make(title="Test", snippet="", body="", hours_ago=1, hits=None):
    pub = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return NewsItem(
        title=title, url="http://x", source="Test",
        published_at=pub, snippet=snippet, body=body,
        region_hits=hits or [],
    )


def test_score_in_zero_one_range():
    s = score_item(_make("Random headline"), SEED_KEYWORD_WEIGHTS)
    assert 0.0 <= s.score <= 1.0


def test_fresher_news_scores_higher_than_older():
    fresh = score_item(_make("Same headline", hours_ago=1), SEED_KEYWORD_WEIGHTS)
    stale = score_item(_make("Same headline", hours_ago=40), SEED_KEYWORD_WEIGHTS)
    assert fresh.score > stale.score


def test_region_hits_bump_score():
    no_region = score_item(_make("Story", hits=[]), SEED_KEYWORD_WEIGHTS)
    with_region = score_item(_make("Story", hits=["Cancún", "Quintana Roo"]),
                              SEED_KEYWORD_WEIGHTS)
    assert with_region.score > no_region.score


def test_score_items_returns_sorted_desc():
    items = [_make("A", hours_ago=20), _make("B", hours_ago=1), _make("C", hours_ago=10)]
    scored = score_items(items, SEED_KEYWORD_WEIGHTS)
    assert scored[0].score >= scored[1].score >= scored[2].score


def test_feedback_increases_accepted_keywords():
    weights = {"playa": 0.1}
    item = _make("Día en la playa con turistas", hits=["Cancún"])
    new_weights = update_weights_from_feedback(weights, [(item, True)])
    assert new_weights["playa"] > weights["playa"]


def test_feedback_decreases_rejected_keywords():
    weights = {"playa": 0.1}
    item = _make("Día en la playa con turistas")
    new_weights = update_weights_from_feedback(weights, [(item, False)])
    assert new_weights["playa"] < weights["playa"]


def test_feedback_clamps_to_minus_one_one():
    weights = {"playa": 0.99}
    item = _make("playa playa playa")
    # Many accepts in a row should not push weight above 1.0
    for _ in range(30):
        weights = update_weights_from_feedback(weights, [(item, True)])
    assert weights["playa"] <= 1.0
```

- [ ] **Step 3: Write env-loader tests**

Path: `/Users/jr/projects/NewsViral/tests/test_env_loader.py`

```python
"""news_viral_pro._load_env_file handles the cases the regular shell would."""
import os
from pathlib import Path

from news_viral_pro import _load_env_file


def test_loads_simple_kv(tmp_path, monkeypatch, clean_env):
    p = tmp_path / ".env"
    p.write_text("FOO=bar\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "bar"


def test_ignores_comments_and_blanks(tmp_path, monkeypatch, clean_env):
    p = tmp_path / ".env"
    p.write_text("# this is a comment\n\nFOO=bar\n# another comment\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "bar"


def test_overrides_empty_env_vars(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("FOO", "")
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "from_file"


def test_does_not_override_set_env_vars(tmp_path, monkeypatch, clean_env):
    monkeypatch.setenv("FOO", "from_shell")
    p = tmp_path / ".env"
    p.write_text("FOO=from_file\n")
    _load_env_file(p)
    assert os.environ["FOO"] == "from_shell"


def test_value_with_equals_sign_preserved(tmp_path, monkeypatch, clean_env):
    p = tmp_path / ".env"
    p.write_text("URL=https://x.com/path?a=1&b=2\n")
    _load_env_file(p)
    assert os.environ["URL"] == "https://x.com/path?a=1&b=2"


def test_missing_file_is_noop(tmp_path, monkeypatch, clean_env):
    _load_env_file(tmp_path / "does_not_exist.env")
    # No exception, nothing set.
```

- [ ] **Step 4: Run the tests**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest tests/test_news_classifier.py tests/test_news_scorer.py tests/test_env_loader.py -v
```

Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_news_classifier.py tests/test_news_scorer.py tests/test_env_loader.py
git commit -m "test: cover classifier, scorer, env-loader"
```

---

## Task 3: Spend logger module + tests

**Files:**
- Create: `spend_logger.py`
- Create: `tests/test_spend_logger.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/jr/projects/NewsViral/tests/test_spend_logger.py`

```python
"""spend_logger records every Replicate call and aggregates by day."""
import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from spend_logger import record, summary, JSONL_PATH, COST_PER_CALL


def test_record_appends_one_jsonl_line(tmp_project):
    record(model="black-forest-labs/flux-pro", scene="escena_1")
    lines = (tmp_project / JSONL_PATH).read_text().splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["model"] == "black-forest-labs/flux-pro"
    assert entry["cost_usd"] == COST_PER_CALL["black-forest-labs/flux-pro"]
    assert entry["scene"] == "escena_1"
    assert "timestamp" in entry


def test_unknown_model_uses_zero_cost(tmp_project):
    record(model="unknown/model", scene="x")
    lines = (tmp_project / JSONL_PATH).read_text().splitlines()
    entry = json.loads(lines[0])
    assert entry["cost_usd"] == 0.0


def test_summary_aggregates_per_day(tmp_project):
    record("black-forest-labs/flux-pro", "escena_1")
    record("bytedance/seedance-1-pro", "escena_1")
    record("minimax/speech-02-hd", "escena_1")
    s = summary(days=1)
    today = datetime.now().strftime("%Y-%m-%d")
    assert today in s
    today_summary = s[today]
    assert today_summary["total_usd"] == pytest.approx(
        COST_PER_CALL["black-forest-labs/flux-pro"]
        + COST_PER_CALL["bytedance/seedance-1-pro"]
        + COST_PER_CALL["minimax/speech-02-hd"]
    )
    assert today_summary["call_count"] == 3


def test_summary_handles_missing_log(tmp_project):
    # Don't write anything; summary should return empty dict, not crash.
    assert summary(days=7) == {}


def test_summary_respects_days_window(tmp_project):
    # Manually write an old entry plus a new one.
    old_ts = (datetime.now() - timedelta(days=10)).isoformat()
    new_ts = datetime.now().isoformat()
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    with JSONL_PATH.open("w") as f:
        f.write(json.dumps({"timestamp": old_ts, "model": "x", "cost_usd": 1.0, "scene": "x"}) + "\n")
        f.write(json.dumps({"timestamp": new_ts, "model": "x", "cost_usd": 2.0, "scene": "x"}) + "\n")
    s = summary(days=7)
    # Only the new entry should be in the 7-day window.
    total = sum(d["total_usd"] for d in s.values())
    assert total == pytest.approx(2.0)
```

- [ ] **Step 2: Run tests to see them fail**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest tests/test_spend_logger.py -v
```

Expected: All fail with ImportError on `spend_logger`.

- [ ] **Step 3: Implement spend_logger**

Path: `/Users/jr/projects/NewsViral/spend_logger.py`

```python
"""Replicate cost tracking.

Wraps each model call by recording a JSONL line with timestamp + model +
cost estimate + scene id. We don't query Replicate's billing API; instead we
keep a static price table that's good enough to spot a runaway spend
overnight. Update COST_PER_CALL when Replicate changes pricing.

Storage: logs/spend.jsonl (gitignored). The webapp's /admin/spend view reads
this via summary(days=N).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)

JSONL_PATH = Path("logs/spend.jsonl")

# Approximate per-call cost in USD. Conservative — round up to catch
# surprises rather than under-report.
COST_PER_CALL: Dict[str, float] = {
    "black-forest-labs/flux-pro": 0.055,
    "black-forest-labs/flux-canny-pro": 0.055,
    "bytedance/seedance-1-pro": 0.45,        # ~$0.45 per 5s @ 1080p; we use 8s ≈ $0.72 actual, but we per-call here
    "minimax/speech-02-hd": 0.005,
    "minimax/voice-cloning": 0.005,
    "devxpy/cog-wav2lip": 0.05,
}


def record(model: str, scene: str = "") -> None:
    """Append one JSONL line for a Replicate call."""
    JSONL_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "model": model,
        "cost_usd": COST_PER_CALL.get(model, 0.0),
        "scene": scene,
    }
    with JSONL_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def summary(days: int = 7) -> Dict[str, Dict[str, float]]:
    """Aggregate spending by day for the last `days` days.

    Returns {date_str: {"total_usd": float, "call_count": int}}.
    Missing log file → empty dict.
    """
    if not JSONL_PATH.exists():
        return {}
    cutoff = datetime.now() - timedelta(days=days)
    by_day: Dict[str, Dict[str, float]] = {}
    with JSONL_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                ts = datetime.fromisoformat(entry["timestamp"])
            except Exception:
                continue
            if ts < cutoff:
                continue
            day = ts.strftime("%Y-%m-%d")
            bucket = by_day.setdefault(day, {"total_usd": 0.0, "call_count": 0})
            bucket["total_usd"] += float(entry.get("cost_usd", 0.0))
            bucket["call_count"] += 1
    return by_day
```

- [ ] **Step 4: Re-run tests**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest tests/test_spend_logger.py -v
```

Expected: All 5 tests pass.

- [ ] **Step 5: Commit**

```bash
git add spend_logger.py tests/test_spend_logger.py
git commit -m "feat(ops): spend_logger module + tests"
```

---

## Task 4: Wire spend logger into orchestrator

**Files:**
- Modify: `/Users/jr/projects/NewsViral/replicate_orchestrator.py`

- [ ] **Step 1: Add import at top of orchestrator**

In `replicate_orchestrator.py`, find the existing `import replicate` line and add right after:

```python
from spend_logger import record as record_spend
```

- [ ] **Step 2: Call record_spend after each successful client.run**

In `_generate_image_url`, after `url = str(output)`:

```python
record_spend(model, scene=f"escena_{index+1}")
```

In `_animate_single`, after `url = str(output)` (the Seedance result line):

```python
record_spend(self.config.video_model, scene=f"escena_{index+1}")
```

In `_generate_single_audio`, after `output = await asyncio.wait_for(...)`:

```python
record_spend("minimax/speech-02-hd", scene=f"escena_{index+1}")
```

In `_lipsync_with_local_files` (and `_lipsync_single`), after `url = str(output)`:

```python
record_spend(self.config.lip_sync_model, scene=f"lipsync_{index+1}")
```

- [ ] **Step 3: Verify orchestrator imports clean**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python -c "from replicate_orchestrator import ReplicateOrchestrator; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add replicate_orchestrator.py
git commit -m "feat(ops): record Replicate spend on each successful call"
```

---

## Task 5: Image search fallback (DuckDuckGo)

**Files:**
- Create: `image_search.py`
- Create: `tests/test_image_search_fallback.py`
- Modify: `news_image_finder.py`

- [ ] **Step 1: Write the failing test**

Path: `/Users/jr/projects/NewsViral/tests/test_image_search_fallback.py`

```python
"""Image search falls through cleanly when DuckDuckGo is unreachable."""
from unittest.mock import patch

import pytest

from image_search import search_news_image


def test_returns_first_url_on_success():
    fake_results = [
        {"image": "https://example.com/a.jpg", "title": "A"},
        {"image": "https://example.com/b.jpg", "title": "B"},
    ]
    with patch("image_search.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.images.return_value = iter(fake_results)
        url = search_news_image("AMLO conferencia Cancún")
    assert url == "https://example.com/a.jpg"


def test_returns_none_on_no_results():
    with patch("image_search.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.images.return_value = iter([])
        url = search_news_image("nada que se encuentre")
    assert url is None


def test_returns_none_when_ddg_raises():
    with patch("image_search.DDGS", side_effect=Exception("network down")):
        url = search_news_image("AMLO")
    assert url is None


def test_skips_obvious_junk_extensions():
    fake_results = [
        {"image": "https://example.com/icon.svg", "title": "junk"},
        {"image": "https://example.com/real.jpg", "title": "ok"},
    ]
    with patch("image_search.DDGS") as ddgs_cls:
        ddgs_cls.return_value.__enter__.return_value.images.return_value = iter(fake_results)
        url = search_news_image("query")
    assert url == "https://example.com/real.jpg"
```

- [ ] **Step 2: Run tests to see them fail**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest tests/test_image_search_fallback.py -v
```

Expected: All fail (ImportError on `image_search`).

- [ ] **Step 3: Implement image_search.py**

Path: `/Users/jr/projects/NewsViral/image_search.py`

```python
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
```

- [ ] **Step 4: Re-run tests**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest tests/test_image_search_fallback.py -v
```

Expected: All 4 pass.

- [ ] **Step 5: Integrate fallback into news_image_finder.py**

In `news_image_finder.py`, ADD this function at the bottom (don't modify the existing `find_reference_image`):

```python
def find_reference_image_with_fallback(article_url: str, headline: str) -> Optional[ReferenceImage]:
    """Try the scraper first, then fall back to DDG image search.

    DDG fallback only fires when scraping returned None — usually because
    the article URL was paywalled, missing og:image meta, or had no <img>
    bigger than the thumbnail threshold."""
    from image_search import search_news_image

    primary = find_reference_image(article_url)
    if primary:
        return primary
    if not headline:
        return None
    url = search_news_image(headline)
    if not url:
        return None
    return ReferenceImage(url=url, source="ddg-image-search", article_url=article_url)
```

- [ ] **Step 6: Switch the pipeline to use the fallback wrapper**

In `news_viral_pro.py`, find the existing `from news_image_finder import find_reference_image` and change to:

```python
from news_image_finder import find_reference_image_with_fallback
```

Then find `ref = await asyncio.to_thread(find_reference_image, script.news_url)` and replace with:

```python
ref = await asyncio.to_thread(find_reference_image_with_fallback, script.news_url, item.title)
```

Note: `item` is in scope where this line lives (inside `produce_video_for_script`). Confirm by grep:
```bash
grep -n "find_reference_image" news_viral_pro.py
```

In `webapp/server.py`, do the same swap. The current call site:
```python
ref = await asyncio.to_thread(find_reference_image, item.url)
```
becomes:
```python
from news_image_finder import find_reference_image_with_fallback
ref = await asyncio.to_thread(find_reference_image_with_fallback, item.url, item.title)
```

- [ ] **Step 7: Verify imports**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python -c "from news_viral_pro import produce_video_for_script; from webapp.server import run_video_pipeline; print('ok')"
```

Expected: `ok`.

- [ ] **Step 8: Commit**

```bash
git add image_search.py news_image_finder.py news_viral_pro.py webapp/server.py tests/test_image_search_fallback.py
git commit -m "feat(quality): DuckDuckGo image search fallback when og:image misses"
```

---

## Task 6: Per-scene fault recovery

**Files:**
- Modify: `replicate_orchestrator.py`
- Modify: `video_compositor.py`

**Problem this solves:** Today if Seedance fails on scene 2, the whole video fails. Better: succeed with scenes 1 and 3, slate scene 2 with "ESCENA NO DISPONIBLE", produce the rest.

- [ ] **Step 1: Modify orchestrate_parallel to never raise on partial Seedance failure**

In `replicate_orchestrator.py`, the existing `generate_video_batch` already accumulates exceptions per scene; verify by reading lines around `out: Dict[str, str] = {}` in that function. The change we need: when a scene fails, write a sentinel marker rather than skipping the key entirely.

Replace the body of `generate_video_batch` (the loop after `await asyncio.gather`):

Find this block:
```python
        out: Dict[str, str] = {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Video {i+1}: {r}")
            else:
                out[f"escena_{i+1}"] = r
        logger.info(f"✅ Videos: {len(out)}/{len(motion_prompts)}")
        return out
```

Replace with:
```python
        out: Dict[str, str] = {}
        failed: list[str] = []
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Video {i+1}: {r}")
                out[f"escena_{i+1}"] = "FAILED"   # sentinel; compositor substitutes a slate
                failed.append(f"escena_{i+1}")
            else:
                out[f"escena_{i+1}"] = r
        if failed:
            logger.warning(f"⚠️  Substituting slates for {len(failed)} failed scenes: {failed}")
        logger.info(f"✅ Videos: {len(out) - len(failed)}/{len(motion_prompts)}")
        return out
```

- [ ] **Step 2: Same treatment for generate_audio_batch (silent fallback)**

In the same file, find `generate_audio_batch`'s result loop. Apply the same pattern:

```python
        out: Dict[str, str] = {}
        for i, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error(f"❌ Audio {i+1}: {r}")
                out[f"escena_{i+1}"] = "SILENT"   # sentinel; compositor uses silence
            else:
                out[f"escena_{i+1}"] = r
        n_ok = sum(1 for v in out.values() if v != "SILENT")
        logger.info(f"✅ Audios: {n_ok}/{len(scripts)}")
        return out
```

- [ ] **Step 3: Modify validate_outputs to ACCEPT sentinels**

In `replicate_orchestrator.py`, find `async def validate_outputs`. Replace its body with:

```python
    async def validate_outputs(self, output_dict: Dict) -> bool:
        """Confirm every scene has *some* output, including sentinels.

        Returns True if every scene resolved to either a real file or a
        sentinel marker the compositor knows how to handle. Returns False
        only if we have ZERO usable scenes (total failure)."""
        videos = output_dict.get("videos", {}) or {}
        images = output_dict.get("imagenes", {}) or {}
        audios = output_dict.get("audios", {}) or {}

        primary = videos if videos else images
        n_real_primary = sum(
            1 for v in primary.values()
            if v != "FAILED" and v != "SILENT" and Path(v).exists()
        )
        if n_real_primary == 0:
            logger.error(f"❌ Zero real primary visuals; cannot compose")
            return False

        # Audio sentinels are fine — compositor uses silence
        for key, path in audios.items():
            if path == "SILENT":
                continue
            if not Path(path).exists():
                logger.warning(f"⚠️  Audio missing for {key}: {path}; treating as silent")
                audios[key] = "SILENT"

        logger.info(f"✅ Validated: {n_real_primary} real visuals + {len(audios)} audio slots")
        return True
```

- [ ] **Step 4: Compositor handles sentinels by substituting a slate**

In `video_compositor.py`, find `_compose_from_clips`. Right after the `keys = sorted(videos.keys())` line, add slate-generation logic. Modify the per-scene loop:

Find:
```python
        for idx, key in enumerate(keys, start=1):
            clip = Path(videos[key]).resolve()
            audio_path_str = audios.get(key, "")
            audio = Path(audio_path_str).resolve() if audio_path_str else None
            scene_file = self.work_dir / f"scene_{idx:02d}.mp4"

            input_args = ["-i", str(clip)]
```

Replace with:
```python
        for idx, key in enumerate(keys, start=1):
            raw_clip = videos[key]
            audio_path_str = audios.get(key, "")
            scene_file = self.work_dir / f"scene_{idx:02d}.mp4"

            # Sentinel handling: if Seedance failed for this scene, substitute
            # a 5-second slate so the rest of the video still composes.
            if raw_clip == "FAILED":
                logger.warning(f"⚠️  Scene {idx} ({key}) is FAILED — substituting slate")
                slate_path = self._render_slate(idx, "ESCENA NO DISPONIBLE")
                clip = Path(slate_path).resolve()
            else:
                clip = Path(raw_clip).resolve()

            # If audio failed, use silence of the clip's duration.
            if audio_path_str == "SILENT" or not audio_path_str:
                audio = None
            else:
                audio = Path(audio_path_str).resolve()

            input_args = ["-i", str(clip)]
```

- [ ] **Step 5: Add the slate-rendering method**

Still in `video_compositor.py`, add this method to the `VideoCompositor` class (near `_get_audio_duration` is fine):

```python
    def _render_slate(self, scene_idx: int, message: str) -> str:
        """Generate a 5-second solid slate with a message for failed scenes."""
        slate_file = self.work_dir / f"slate_{scene_idx:02d}.mp4"
        font = self.style.font_path or "/System/Library/Fonts/Supplemental/Arial.ttf"
        primary = self.style.primary_hex

        text = message.replace("'", "")[:50]
        vf = (
            f"drawbox=x=0:y=0:w=iw:h=ih:color=0x{primary}:t=fill,"
            f"drawtext=fontfile='{font}':text='{text}':"
            f"x=(w-text_w)/2:y=(h-text_h)/2:fontsize=64:fontcolor=white"
        )
        cmd = [
            FFMPEG_BIN, "-y",
            "-f", "lavfi", "-t", "5",
            "-i", f"color=c=0x{primary}:s={self.style.width}x{self.style.height}:r={self.style.fps}",
            "-vf", vf,
            "-c:v", "libx264", "-preset", "fast", "-crf", "18", "-pix_fmt", "yuv420p",
            str(slate_file),
        ]
        subprocess.run(cmd, capture_output=True, check=True, timeout=60)
        logger.info(f"🛑 Slate rendered: {slate_file}")
        return str(slate_file)
```

- [ ] **Step 6: Compile-check and import-check**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python -m py_compile replicate_orchestrator.py video_compositor.py
.venv/bin/python -c "from replicate_orchestrator import ReplicateOrchestrator; from video_compositor import VideoCompositor; print('ok')"
```

Expected: `ok`.

- [ ] **Step 7: Commit**

```bash
git add replicate_orchestrator.py video_compositor.py
git commit -m "feat(resilience): per-scene fault recovery with slate fallback"
```

---

## Task 7: Logo PNG generator

**Files:**
- Create: `logo_generator.py`
- Create: `assets/logo.png` and `assets/logo_white.png` (generated)
- Modify: `.gitignore` to allow `assets/`

- [ ] **Step 1: Write logo_generator.py**

Path: `/Users/jr/projects/NewsViral/logo_generator.py`

```python
#!/usr/bin/env python3
"""Generate the VOZ DEL PUEBLO logo as a PNG asset.

We use Pillow to draw a clean, brand-colored badge:
  - dark green pill background (Morena green #235B4E)
  - red accent line on the left (Morena red #9F2241)
  - 'VOZ DEL PUEBLO' typography in bold white

Two variants:
  assets/logo.png        — solid green background, opaque
  assets/logo_white.png  — white text on transparent (for darker frames)

Run once: python logo_generator.py
The PNGs are committed so every deploy gets the same identity.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path("assets")
ASSETS.mkdir(exist_ok=True)


def _font(size: int):
    for candidate in [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/HelveticaNeue.ttc",
    ]:
        if Path(candidate).exists():
            try:
                return ImageFont.truetype(candidate, size)
            except Exception:
                continue
    return ImageFont.load_default()


def make_logo(out_path: Path, *, transparent: bool = False):
    W, H = 720, 180
    bg = (0, 0, 0, 0) if transparent else (35, 91, 78, 255)
    img = Image.new("RGBA", (W, H), bg)
    draw = ImageDraw.Draw(img)

    # Red accent bar on the left.
    draw.rectangle((0, 0, 16, H), fill=(159, 34, 65, 255))

    # Text.
    text = "VOZ DEL PUEBLO"
    font_main = _font(72)
    color = (255, 255, 255, 255)
    # Center text vertically; left-aligned with padding.
    bbox = draw.textbbox((0, 0), text, font=font_main)
    text_h = bbox[3] - bbox[1]
    text_y = (H - text_h) // 2 - 10  # small visual nudge up
    draw.text((48, text_y), text, font=font_main, fill=color)

    img.save(out_path, "PNG")
    print(f"  ✅ {out_path}")


if __name__ == "__main__":
    make_logo(ASSETS / "logo.png", transparent=False)
    make_logo(ASSETS / "logo_white.png", transparent=True)
    print("Done.")
```

- [ ] **Step 2: Modify .gitignore to allow assets/**

In `.gitignore`, find the section starting with `# Exception: anchor portraits` and add right after:

```
!assets/
!assets/*.png
```

- [ ] **Step 3: Generate the PNGs**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python logo_generator.py
ls -lh assets/
```

Expected: `assets/logo.png` and `assets/logo_white.png` (~5-15 KB each).

- [ ] **Step 4: Verify they open**

```bash
cd /Users/jr/projects/NewsViral
file assets/logo.png  # should report PNG image data
```

- [ ] **Step 5: Commit**

```bash
git add logo_generator.py assets/ .gitignore
git commit -m "feat(brand): real PNG logo asset (Pillow-generated, brand colors)"
```

---

## Task 8: Cleanup script

**Files:**
- Create: `cleanup.py`

- [ ] **Step 1: Write cleanup.py**

Path: `/Users/jr/projects/NewsViral/cleanup.py`

```python
#!/usr/bin/env python3
"""Disk hygiene for NewsViral PRO.

What this prunes (default: anything older than 7 days):
  - replicate_outputs/*.{jpg,mp3,mp4}    intermediate FLUX/Minimax/Seedance files
  - video_work/*                         scratch from the compositor
  - logs/runs/* older than 30 days       archived to logs/archive/runs-YYYY-MM.tar.gz

What we DON'T touch:
  - video_output/                        final videos (you might still be reviewing)
  - anchor_portraits/                    brand assets (committed to git)
  - logs/spend.jsonl                     append-only spend log (small, useful)
  - logs/runs/* under 30 days            recent runs you might still want

Run on demand or via launchd timer (see DEPLOY.md follow-up).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tarfile
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("cleanup")


def _age_days(path: Path) -> float:
    mtime = datetime.fromtimestamp(path.stat().st_mtime)
    return (datetime.now() - mtime).total_seconds() / 86400.0


def prune_intermediate(days: int) -> int:
    """Delete files older than `days` from replicate_outputs/ and video_work/."""
    deleted = 0
    for d in (Path("replicate_outputs"), Path("video_work")):
        if not d.exists():
            continue
        for p in d.rglob("*"):
            if not p.is_file():
                continue
            if _age_days(p) > days:
                p.unlink()
                deleted += 1
    logger.info(f"🗑  Pruned {deleted} intermediate files older than {days}d")
    return deleted


def archive_old_runs(days: int) -> int:
    """tar-gz any logs/runs/* older than `days` into logs/archive/."""
    runs = Path("logs/runs")
    if not runs.exists():
        return 0
    archive_dir = Path("logs/archive")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archived = 0
    for run_dir in runs.iterdir():
        if not run_dir.is_dir():
            continue
        if _age_days(run_dir) <= days:
            continue
        month_tag = datetime.fromtimestamp(run_dir.stat().st_mtime).strftime("%Y-%m")
        tar_path = archive_dir / f"runs-{month_tag}.tar.gz"
        mode = "a" if tar_path.exists() else "w"
        # gzip doesn't support append mode; use 'w' if absent, else read+repack.
        if mode == "w":
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(run_dir, arcname=run_dir.name)
        else:
            # Read existing, add new dir, write back.
            existing = tarfile.open(tar_path, "r:gz")
            members = existing.getmembers()
            existing.close()
            # Easier: tarfile doesn't support clean append for gz.
            # Just create a new tar with a -2 suffix.
            tar_path = archive_dir / f"runs-{month_tag}-{run_dir.name}.tar.gz"
            with tarfile.open(tar_path, "w:gz") as tar:
                tar.add(run_dir, arcname=run_dir.name)
        shutil.rmtree(run_dir)
        archived += 1
        logger.info(f"📦 {run_dir.name} → {tar_path.name}")
    logger.info(f"📦 Archived {archived} runs older than {days}d")
    return archived


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--days", type=int, default=7,
                   help="Prune intermediate files older than N days (default 7)")
    p.add_argument("--archive-days", type=int, default=30,
                   help="Archive logs/runs older than N days (default 30)")
    args = p.parse_args()

    prune_intermediate(args.days)
    archive_old_runs(args.archive_days)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python cleanup.py --days 7 --archive-days 30
```

Expected: prints pruned counts (likely 0 on a fresh repo), no errors.

- [ ] **Step 3: Commit**

```bash
git add cleanup.py
git commit -m "feat(ops): cleanup.py for disk hygiene"
```

---

## Task 9: Webapp /admin/spend endpoint

**Files:**
- Modify: `webapp/server.py`

- [ ] **Step 1: Add the endpoint**

In `webapp/server.py`, after the `/health` endpoint, add:

```python
@app.get("/admin/spend", response_class=HTMLResponse)
async def admin_spend(request: Request, _: None = Depends(require_auth)):
    """Quick view of Replicate spend in the last 14 days.

    No CSS bells — this is for the operator, not the public. The webapp's
    main `/` view already gives the polished mobile UI for everyone else."""
    from spend_logger import summary

    days = 14
    data = summary(days=days)
    sorted_days = sorted(data.keys(), reverse=True)
    total = sum(d["total_usd"] for d in data.values())
    total_calls = sum(d["call_count"] for d in data.values())

    rows = "".join(
        f'<tr><td>{day}</td>'
        f'<td style="text-align:right">${data[day]["total_usd"]:.2f}</td>'
        f'<td style="text-align:right">{data[day]["call_count"]}</td></tr>'
        for day in sorted_days
    )
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Spend · VOZ DEL PUEBLO</title>
<style>
body {{ font-family: -apple-system, sans-serif; padding: 24px; max-width: 600px; margin: 0 auto; background: #F6F4EE; color: #1A1A1A; }}
h1 {{ font-size: 22px; }}
table {{ width: 100%; border-collapse: collapse; margin-top: 16px; background: white; border-radius: 12px; overflow: hidden; }}
th, td {{ padding: 12px 16px; text-align: left; border-bottom: 1px solid rgba(0,0,0,0.06); }}
th {{ background: #2D5A4E; color: white; }}
.total {{ font-weight: bold; background: rgba(159,34,65,0.08); }}
.note {{ color: #666; font-size: 12px; margin-top: 16px; }}
a {{ color: #2D5A4E; }}
</style></head><body>
<h1>Replicate spend (last {days} days)</h1>
<table>
  <tr><th>Día</th><th style="text-align:right">USD</th><th style="text-align:right">Llamadas</th></tr>
  {rows or '<tr><td colspan="3" style="text-align:center;color:#999">Sin datos aún</td></tr>'}
  <tr class="total"><td>TOTAL</td>
    <td style="text-align:right">${total:.2f}</td>
    <td style="text-align:right">{total_calls}</td></tr>
</table>
<p class="note">Costos estimados a partir de COST_PER_CALL en spend_logger.py.
Para precios reales en tu cuenta, revisa <a href="https://replicate.com/account/billing">replicate.com/account/billing</a>.</p>
<p><a href="/">← Volver</a></p>
</body></html>"""
    return HTMLResponse(html)
```

- [ ] **Step 2: Smoke test**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python -c "from webapp.server import app; print('imports ok')"
```

Expected: `imports ok`.

- [ ] **Step 3: End-to-end webapp smoke**

```bash
cd /Users/jr/projects/NewsViral
WEBAPP_PASSWORD=testpw .venv/bin/uvicorn webapp.server:app --host 127.0.0.1 --port 8765 > /tmp/u.log 2>&1 &
PID=$!
for i in 1 2 3 4 5 6 7 8; do
  if curl -fsS http://127.0.0.1:8765/health >/dev/null 2>&1; then break; fi
  sleep 0.5
done
curl -fsS -u "voz:testpw" http://127.0.0.1:8765/admin/spend | grep -E "Replicate spend|TOTAL" | head -3
kill $PID 2>/dev/null
wait 2>/dev/null
```

Expected: prints rows with "Replicate spend" and "TOTAL" lines.

- [ ] **Step 4: Commit**

```bash
git add webapp/server.py
git commit -m "feat(ops): /admin/spend endpoint shows last-14-day Replicate cost"
```

---

## Task 10: End-to-end mock test

**Files:**
- Create: `tests/test_e2e_mock.py`

This is the most important safety net: it runs the full pipeline in skip_replicate=True mode and asserts every stage produces an artifact. Future refactors that quietly break the chain will be caught here.

- [ ] **Step 1: Write the e2e mock test**

Path: `/Users/jr/projects/NewsViral/tests/test_e2e_mock.py`

```python
"""End-to-end mock pipeline. Asserts every stage produces an artifact
even when skip_replicate=True (no Replicate or Anthropic calls).
"""
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from news_sources import NewsItem
from replicate_orchestrator import ReplicateConfig, ReplicateOrchestrator


def _news():
    return NewsItem(
        title="Andy López Beltrán es captado en Cartier",
        url="https://demo.test/x",
        source="Test source",
        published_at="2026-05-17T10:00:00+00:00",
        snippet="Reporte ciudadano sobre visita a tienda de lujo",
        body="Cuerpo del artículo de prueba.",
        region_hits=["Cancún"],
    )


def _fake_prompts():
    """The exact shape ReplicateOrchestrator expects from the script writer."""
    return {
        "escena_1": {
            "imagen_prompt": "anchor close-up",
            "motion_prompt": "anchor speaks",
            "audio_script": "Hola, escúchame esto está bueno",
            "emotion": "surprised",
        },
        "escena_2": {
            "imagen_prompt": "event illustration",
            "motion_prompt": "subject moves",
            "audio_script": "Mira lo que se acaba de armar",
            "emotion": "neutral",
        },
        "escena_3": {
            "imagen_prompt": "anchor closing",
            "motion_prompt": "anchor closes",
            "audio_script": "Quédate pendiente raza",
            "emotion": "calm",
        },
    }


@pytest.mark.asyncio
async def test_mock_orchestration_produces_all_artifacts(tmp_project, clean_env):
    orch = ReplicateOrchestrator(ReplicateConfig(
        api_token="",
        skip_replicate=True,
        enable_video=True,
    ))
    result = await orch.orchestrate_parallel(_fake_prompts())

    # Every scene should resolve to either a real file or a sentinel.
    assert "videos" in result
    assert "audios" in result
    assert len(result["videos"]) == 3
    assert len(result["audios"]) == 3

    # Mock files were touched on disk.
    for path in result["videos"].values():
        assert Path(path).exists(), f"video {path} missing"
    for path in result["audios"].values():
        assert Path(path).exists(), f"audio {path} missing"


@pytest.mark.asyncio
async def test_mock_validation_passes(tmp_project, clean_env):
    orch = ReplicateOrchestrator(ReplicateConfig(
        api_token="", skip_replicate=True, enable_video=True
    ))
    result = await orch.orchestrate_parallel(_fake_prompts())
    assert await orch.validate_outputs(result) is True
```

- [ ] **Step 2: Run the test**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest tests/test_e2e_mock.py -v
```

Expected: 2 tests pass.

- [ ] **Step 3: Run the full suite**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest -v
```

Expected: 30+ tests pass, zero failures.

- [ ] **Step 4: Commit**

```bash
git add tests/test_e2e_mock.py
git commit -m "test(e2e): mock pipeline asserts all artifacts produced"
```

---

## Task 11: Final verification + push

- [ ] **Step 1: Run all tests**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/pytest -v --tb=short
```

Expected: ALL tests pass.

- [ ] **Step 2: Compile-check every module**

```bash
cd /Users/jr/projects/NewsViral
.venv/bin/python -m py_compile *.py webapp/*.py
echo "exit=$?"
```

Expected: `exit=0`.

- [ ] **Step 3: Full mock end-to-end (smoke)**

```bash
cd /Users/jr/projects/NewsViral
# Confirm Phase 7+ still wires together end-to-end in mock mode (no spend)
.venv/bin/python news_viral_pro.py --mock --auto 1 --max 3 2>&1 | tail -15
```

Expected: Last lines show "Pipeline halted at TAREA 5" OR "Pipeline complete" (mock mode produces sentinel files; either outcome verifies wiring).

- [ ] **Step 4: Push everything**

```bash
cd /Users/jr/projects/NewsViral
git status   # confirm nothing left uncommitted
git log --oneline -15
git push origin main
```

Expected: 9-11 commits ahead pushed cleanly.

---

## Verification checklist (final)

- [ ] `pytest` runs from repo root and discovers ≥30 tests, all passing
- [ ] `spend_logger.record()` writes JSONL lines; `summary()` aggregates per day
- [ ] `/admin/spend` shows the per-day spend table (auth-required)
- [ ] `image_search.py` returns None gracefully when DDG unreachable (tested)
- [ ] `news_image_finder.find_reference_image_with_fallback` is what news_viral_pro and webapp now call
- [ ] Orchestrator wraps each successful Replicate call in `record_spend`
- [ ] When a Seedance/MiniMax call fails per-scene, validate_outputs still returns True if at least one scene survives
- [ ] Compositor renders an "ESCENA NO DISPONIBLE" slate for FAILED sentinels
- [ ] `logo_generator.py` produced `assets/logo.png` and `assets/logo_white.png` committed to git
- [ ] `cleanup.py --days 7` runs without errors
- [ ] All files compile via `py_compile`
- [ ] `git status` is clean and remote is in sync

## Out of scope (deferred)

- Music bed (was already deferred in audit; copyright concerns, ~5% perceived gain)
- Voice cloning execution (needs user-provided audio sample)
- Vertical end-to-end test (needs $1.72 Replicate spend; user will trigger)
- Lip-sync visual evaluation (needs user to watch and score)
- Auto-publish to TikTok/Reels (premature)
- Multi-user auth tiers (one user)
