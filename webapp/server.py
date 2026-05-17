"""FastAPI webapp for the VOZ DEL PUEBLO pipeline.

Three views, all mobile-first:

- `/`            → Tinder-style cards of today's QR news (tap Sí / No).
- `/queue`       → Items the user accepted, with generation status. Polls.
- `/videos`      → Library of already-produced videos, playable inline.

Background workers keep a single video generating at a time so we don't
blow the Replicate rate limit. State is held in `webapp/state.json` (one
file, no DB needed for a one-user app).

Run locally:
    uvicorn webapp.server:app --host 0.0.0.0 --port 8000 --reload

Access from your phone:
    1) Same Wi-Fi: open http://<your-mac-ip>:8000 on phone.
    2) Anywhere: run `cloudflared tunnel --url http://localhost:8000` and
       open the .trycloudflare.com URL it prints.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Allow running as `uvicorn webapp.server:app` from project root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import secrets
from base64 import b64decode

from fastapi import BackgroundTasks, FastAPI, Form, Request, HTTPException, Depends
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from news_sources import (
    NewsItem, fetch_google_news, enrich_with_body,
    SourceRegistry, fetch_all_rss_sources,
)
from news_db import NewsDB, ArticleRecord
from deduplicator import Deduplicator
from news_scorer import ScoredItem, load_weights, save_weights, score_items, update_weights_from_feedback
from news_image_finder import find_reference_image_with_fallback
from script_writer import (
    ScriptWriter, NARRATIVE_MODES, DEFAULT_MODE,
    anchor_scene_keys, event_scene_keys,
)
from replicate_orchestrator import ReplicateConfig, ReplicateOrchestrator
from video_compositor import BrandingConfig, VideoCompositor
from run_logger import RunLogger
from brand_style import STYLE_VARIANTS, ANCHORS, anchor_for, pick_voice_id_for
from news_enrichment import (
    aggregate_news_clusters,
    NewsCluster,
    NewsEnrichmentSystem,
    EnrichmentConfig,
    EnrichmentError,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("webapp")


# ---------- Env ----------

def _find_env_file(start: Path) -> Optional[Path]:
    """Walk upward from `start` looking for a `.env`.

    Git worktrees don't inherit `.env` (it's gitignored), but the main repo
    has it. This finds it whether we run from the main repo or any worktree.
    """
    for d in [start, *start.parents]:
        candidate = d / ".env"
        if candidate.exists():
            return candidate
    return None


def _load_env(path: Optional[Path] = None) -> None:
    env_path = path or _find_env_file(ROOT)
    if env_path is None:
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip()
    logger.info(f"📒 Loaded env from {env_path}")


_load_env()


# Health check: warn loudly if the keys video generation needs are missing.
# We don't refuse to start (the dashboard, refresh, and queue views still work
# without them) — but every Generar click will fail, so the user should know.
_MISSING_KEYS = [k for k in ("ANTHROPIC_API_KEY", "REPLICATE_API_TOKEN")
                 if not os.environ.get(k)]
if _MISSING_KEYS:
    logger.warning(
        f"⚠️  Missing env vars: {', '.join(_MISSING_KEYS)} — "
        f"'Generar' will fail until these are set. "
        f"Add them to {ROOT}/.env or the parent repo .env"
    )


# ---------- State (one user, no DB needed) ----------

STATE_FILE = ROOT / "webapp" / "state.json"
STATE_LOCK = threading.Lock()


def _empty_state() -> Dict[str, Any]:
    return {
        "current_run": None,        # id of the most recent /news refresh
        "news_by_run": {},          # run_id → [scored_item dicts]
        "decisions": {},            # url → bool (True = accepted)
        "jobs": {},                 # job_id → {status, news_url, video_path, error, started, finished}
        "job_order": [],            # job_id list for FIFO display
    }


def load_state() -> Dict[str, Any]:
    with STATE_LOCK:
        if not STATE_FILE.exists():
            return _empty_state()
        try:
            return json.loads(STATE_FILE.read_text())
        except Exception:
            return _empty_state()


def save_state(state: Dict[str, Any]) -> None:
    with STATE_LOCK:
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def update_state(mutator) -> Dict[str, Any]:
    """Atomically load → mutate → save."""
    with STATE_LOCK:
        state = _empty_state()
        if STATE_FILE.exists():
            try:
                state = json.loads(STATE_FILE.read_text())
            except Exception:
                pass
        mutator(state)
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))
        return state


# ---------- Job worker ----------

GEN_LOCK = asyncio.Lock()  # serialize Replicate-heavy work


def _env_flag(key: str) -> bool:
    """Read a boolean-ish env var. Accepts 1/true/yes/on (case-insensitive)."""
    return os.environ.get(key, "").strip().lower() in ("1", "true", "yes", "on")


def _env_default_style() -> str:
    return os.environ.get("WEBAPP_STYLE", "caricature")


def _inject_anchor_portrait_for(
    prompts: Dict[str, Any], anchor_id: str,
    scene_keys: Optional[List[str]] = None,
) -> None:
    """Attach the cached portrait URL to the scene keys that feature the anchor.

    `scene_keys` defaults to ("escena_1", "escena_3") for the legacy 3-scene
    anchor_camera pipeline. M6 callers pass an explicit list from
    `script_writer.anchor_scene_keys(scenes, mode)`.
    """
    if scene_keys is None:
        scene_keys = ["escena_1", "escena_3"]
    if not scene_keys:
        return
    path = ROOT / "anchor_portraits" / "manifest.json"
    if not path.exists():
        return
    try:
        manifest = json.loads(path.read_text())
    except Exception:
        return
    entry = manifest.get(anchor_id)
    url = (entry or {}).get("url")
    if not url:
        return
    for key in scene_keys:
        if key in prompts and isinstance(prompts[key], dict):
            prompts[key]["anchor_portrait_url"] = url


async def run_video_pipeline(job_id: str, item_dict: Dict[str, Any]) -> None:
    """Run enrichment → script → Replicate → compose for one news item.

    Stage updates write to s["jobs"][job_id]["stage"] for live UI display:
      "enriching" → "scripting" → "rendering" → "composing" → "done"|"failed"
    """
    item = NewsItem(**{k: v for k, v in item_dict.items() if k in NewsItem.__dataclass_fields__})

    def _set_stage(stage: str, **extra) -> None:
        def _mut(s):
            j = s["jobs"].get(job_id) or {}
            j["stage"] = stage
            for k, v in extra.items():
                j[k] = v
            s["jobs"][job_id] = j
        update_state(_mut)

    def mark_started(s):
        s["jobs"][job_id]["status"] = "running"
        s["jobs"][job_id]["started"] = datetime.now().isoformat()
        s["jobs"][job_id]["stage"] = "enriching"
    update_state(mark_started)

    # Per-job options — read at job dispatch time so the user can change them
    # via the webapp settings form (or .env) and the very next job picks up
    # the new values without restarting the service.
    opts = item_dict.get("_options", {}) or {}
    style_name = (opts.get("style") or _env_default_style()).strip() or "caricature"
    if style_name not in STYLE_VARIANTS:
        style_name = "caricature"
    use_lipsync = bool(opts.get("lipsync", _env_flag("WEBAPP_LIPSYNC")))
    use_vertical = bool(opts.get("vertical", _env_flag("WEBAPP_VERTICAL")))
    # M6 narrative mode — read from settings with safe defaults.
    narrative_mode = (opts.get("narrative_mode")
                      or os.environ.get("WEBAPP_NARRATIVE_MODE", DEFAULT_MODE)).strip()
    if narrative_mode not in NARRATIVE_MODES:
        narrative_mode = DEFAULT_MODE
    try:
        target_scenes = int(opts.get("num_scenes") or os.environ.get("WEBAPP_NUM_SCENES", 3))
    except (TypeError, ValueError):
        target_scenes = 3
    target_scenes = max(2, min(target_scenes, 8))
    try:
        target_duration_s = int(opts.get("target_duration") or os.environ.get("WEBAPP_TARGET_DURATION", 30))
    except (TypeError, ValueError):
        target_duration_s = 30
    target_duration_s = max(10, min(target_duration_s, 90))
    # voiceover_only forcibly disables lip-sync (no anchor on camera).
    if narrative_mode == "voiceover_only" and use_lipsync:
        logger.info(f"Job {job_id}: voiceover_only → lipsync auto-disabled")
        use_lipsync = False

    async with GEN_LOCK:
        try:
            writer = ScriptWriter(model="claude-haiku-4-5", style=style_name)

            # Deep enrichment: 7+ sources, fact-check, brief, real images.
            # Runs BEFORE script_writer so the brief + verified facts feed the
            # prompt. Reuses the writer's Anthropic client.
            enricher = NewsEnrichmentSystem(
                writer.client,
                logger,
                config=EnrichmentConfig(quality_threshold=70),
            )
            try:
                enriched = await enricher.enrich(item)
                _set_stage("scripting",
                           enrichment_score=enriched.quality_score,
                           sources_count=len(enriched.sources),
                           facts_count=len(enriched.facts),
                           images_count=len(enriched.images),
                           passed=enriched.passed)
                if enriched.brief:
                    item.body = enriched.brief
                item.verified_facts = [f.text for f in enriched.facts]
                item.source_refs = [s.url for s in enriched.sources]
                item.enriched_quality_score = enriched.quality_score
                item.selected_image_urls = list(enriched.selected_image_urls)
            except EnrichmentError as e:
                logger.warning(f"Enrichment failed for job {job_id}: {e} (continuing)")
                _set_stage("scripting", enrichment_error=str(e))

            anchor = anchor_for(f"{item.title}\n{item.snippet}\n{item.body}")
            script = await asyncio.to_thread(
                writer.write, item, anchor,
                mode=narrative_mode,
                num_scenes=target_scenes,
                target_duration_s=target_duration_s,
            )
            _set_stage("rendering",
                       narrative_mode=narrative_mode,
                       scene_count=len(script.scenes))

            prompts = dict(script.scenes)
            # Anchor portrait → only on the scenes that actually feature
            # the anchor on camera (anchor_camera: first+last, hybrid: first,
            # voiceover_only: none).
            anchor_keys = anchor_scene_keys(prompts, narrative_mode)
            if anchor_keys:
                _inject_anchor_portrait_for(prompts, anchor.id, anchor_keys)

            # Reference image enrichment for event scenes (those WITHOUT
            # anchor). Distribute the 3 LLM-selected images across event
            # scenes (cycled). Fallback to DuckDuckGo-augmented og:image.
            event_keys = event_scene_keys(prompts, narrative_mode)
            selected = list(getattr(item, "selected_image_urls", None) or [])
            scraped_ref = None
            if event_keys and not selected and item.url:
                scraped_ref = await asyncio.to_thread(
                    find_reference_image_with_fallback, item.url, item.title,
                )
            for i, key in enumerate(event_keys):
                if not isinstance(prompts.get(key), dict):
                    continue
                if selected:
                    chosen = selected[i % len(selected)]
                    prompts[key]["reference_image_url"] = chosen
                    logger.info(f"📸 ref-image {key} (enriched): {chosen[:80]}")
                elif scraped_ref:
                    prompts[key]["reference_image_url"] = scraped_ref.url

            # Gender-matched voice (with optional cloned override).
            voice_id = pick_voice_id_for(anchor, dict(os.environ))
            prompts["voice_params"] = {"voice_id": voice_id, "language_boost": "Spanish"}

            orch = ReplicateOrchestrator(ReplicateConfig(
                api_token=os.environ["REPLICATE_API_TOKEN"],
                enable_video=True,
                enable_lip_sync=use_lipsync,
                video_aspect_ratio="9:16" if use_vertical else "16:9",
            ))
            elementos = await orch.orchestrate_parallel(prompts)
            if not await orch.validate_outputs(elementos):
                raise RuntimeError("validate_outputs failed")
            _set_stage("composing")

            compositor = VideoCompositor(
                BrandingConfig(
                    colors={"primary": "#235B4E", "accent": "#9F2241", "bg": "#000000"},
                    vertical=use_vertical,
                    narrative_mode=narrative_mode,
                    cover_kicker=(
                        item.region_hits[0].upper() if item.region_hits else ""
                    ),
                ),
                news_title=item.title,
                news_source=item.source,
                anchor=anchor,
            )
            composed = await asyncio.to_thread(compositor.compose_with_audio, elementos)
            result = await asyncio.to_thread(compositor.export_mp4, composed)

            run_log = RunLogger()
            run_log.fetched([item])
            run_log.scripts([{
                "news_url": script.news_url,
                "anchor_id": script.anchor_id,
                "anchor_name": script.anchor_name,
                "style": script.style,
                "scenes": script.scenes,
            }])
            archived = run_log.video(1, item.title, result, Path(result["video_path"]))

            def mark_done(s):
                s["jobs"][job_id]["status"] = "done"
                s["jobs"][job_id]["stage"] = "done"
                s["jobs"][job_id]["video_path"] = str(archived)
                s["jobs"][job_id]["anchor"] = anchor.name
                s["jobs"][job_id]["finished"] = datetime.now().isoformat()
            update_state(mark_done)

        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            def mark_fail(s):
                s["jobs"][job_id]["status"] = "failed"
                s["jobs"][job_id]["stage"] = "failed"
                s["jobs"][job_id]["error"] = str(e)
                s["jobs"][job_id]["finished"] = datetime.now().isoformat()
            update_state(mark_fail)


# ---------- Auth ----------
#
# Public hosting requires gating expensive endpoints — Replicate jobs cost
# real money. We use HTTP Basic auth with a single shared password from the
# WEBAPP_PASSWORD env var. Set it in .env on the Mac mini and share with
# friends. If unset, the app falls back to OPEN MODE (logs a warning) so
# local development still works without auth headaches.
#
# Auth scope:
#   - /         GET  → REQUIRES auth (the swipe UI)
#   - /refresh  POST → REQUIRES auth
#   - /decide   POST → REQUIRES auth (this is what costs money)
#   - /queue           READ-ONLY but auth still required (it shows pending jobs)
#   - /videos          PUBLIC (anyone with the link can view produced videos)
#   - /video/<id>      PUBLIC (the actual mp4 — share with friends)
#   - /health          PUBLIC

WEBAPP_PASSWORD = os.environ.get("WEBAPP_PASSWORD", "").strip()
WEBAPP_USERNAME = os.environ.get("WEBAPP_USERNAME", "voz").strip()

if not WEBAPP_PASSWORD:
    logger.warning(
        "⚠️  WEBAPP_PASSWORD not set — running in OPEN MODE. "
        "Set it in .env before exposing the app to the public internet."
    )


def require_auth(request: Request) -> None:
    """Dependency: require HTTP Basic credentials. No-op if no password set."""
    if not WEBAPP_PASSWORD:
        return

    header = request.headers.get("authorization", "")
    if not header.lower().startswith("basic "):
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": 'Basic realm="Voz del Pueblo"'},
        )
    try:
        decoded = b64decode(header.split(" ", 1)[1]).decode("utf-8")
        user, _, password = decoded.partition(":")
    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Malformed credentials",
            headers={"WWW-Authenticate": 'Basic realm="Voz del Pueblo"'},
        )

    # Constant-time compare to keep timing leaks off the table.
    user_ok = secrets.compare_digest(user, WEBAPP_USERNAME)
    pass_ok = secrets.compare_digest(password, WEBAPP_PASSWORD)
    if not (user_ok and pass_ok):
        raise HTTPException(
            status_code=401,
            detail="Wrong username or password",
            headers={"WWW-Authenticate": 'Basic realm="Voz del Pueblo"'},
        )


# ---------- App ----------

# Lazy singletons for the news engine (M2-M5). Created on first /refresh
# so import-time isn't blocked by I/O and tests can swap them.
_SOURCE_REGISTRY: Optional[SourceRegistry] = None
_NEWS_DB: Optional[NewsDB] = None


def _get_source_registry() -> SourceRegistry:
    global _SOURCE_REGISTRY
    if _SOURCE_REGISTRY is None:
        _SOURCE_REGISTRY = SourceRegistry.from_file(
            str(ROOT / "source_registry.json")
        )
    return _SOURCE_REGISTRY


def _get_news_db() -> NewsDB:
    global _NEWS_DB
    if _NEWS_DB is None:
        _NEWS_DB = NewsDB(str(ROOT / "news.db"))
        _NEWS_DB.init_schema()
    return _NEWS_DB


app = FastAPI(title="VOZ DEL PUEBLO")
templates = Jinja2Templates(directory=str(ROOT / "webapp" / "templates"))
app.mount("/static", StaticFiles(directory=str(ROOT / "webapp" / "static")), name="static")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, _: None = Depends(require_auth)):
    state = load_state()
    run_id = state.get("current_run")
    items = state.get("news_by_run", {}).get(run_id, []) if run_id else []
    decisions = state.get("decisions", {})
    # Show only items the user hasn't decided on yet.
    pending = [i for i in items if i["url"] not in decisions]
    # Current generation settings — read from session state with env fallback.
    settings = state.get("settings", {}) or {}
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pending": pending,
        "total": len(items),
        "decided": len(decisions),
        "queue_size": sum(1 for j in state.get("jobs", {}).values() if j["status"] in ("queued", "running")),
        "videos_ready": sum(1 for j in state.get("jobs", {}).values() if j["status"] == "done"),
        "style_options": list(STYLE_VARIANTS.keys()),
        "current_style": settings.get("style", _env_default_style()),
        "current_lipsync": settings.get("lipsync", _env_flag("WEBAPP_LIPSYNC")),
        "current_vertical": settings.get("vertical", _env_flag("WEBAPP_VERTICAL")),
        "narrative_modes": list(NARRATIVE_MODES),
        "current_mode": settings.get("narrative_mode", DEFAULT_MODE),
        "current_num_scenes": settings.get("num_scenes", 3),
        "current_target_duration": settings.get("target_duration", 30),
    })


@app.post("/settings", response_class=RedirectResponse)
async def update_settings(
    request: Request,
    _: None = Depends(require_auth),
    style: str = Form("caricature"),
    lipsync: str = Form(""),    # "on" or ""
    vertical: str = Form(""),
    narrative_mode: str = Form(DEFAULT_MODE),
    num_scenes: str = Form("3"),
    target_duration: str = Form("30"),
):
    """Persist generation settings so subsequent /decide calls pick them up."""
    new_style = style if style in STYLE_VARIANTS else "caricature"
    nm = narrative_mode if narrative_mode in NARRATIVE_MODES else DEFAULT_MODE
    try:
        ns = max(2, min(int(num_scenes), 8))
    except (TypeError, ValueError):
        ns = 3
    try:
        td = max(10, min(int(target_duration), 90))
    except (TypeError, ValueError):
        td = 30
    # voiceover_only forcibly disables lipsync regardless of toggle.
    lip = lipsync == "on" and nm != "voiceover_only"

    def mut(state):
        state["settings"] = {
            "style": new_style,
            "lipsync": lip,
            "vertical": vertical == "on",
            "narrative_mode": nm,
            "num_scenes": ns,
            "target_duration": td,
        }
    update_state(mut)
    return RedirectResponse(url="/", status_code=303)


# Simple OR query — Google News RSS handles this fine; extra_queries below
# add a second angle without exploding into N HTTP calls.
QR_QUERY = "Cancun OR \"Quintana Roo\" OR \"Riviera Maya\" OR Tulum"


@app.post("/refresh", response_class=RedirectResponse)
async def refresh(_: None = Depends(require_auth)):
    """Multi-source aggregator: Google News + Reddit + intl RSS, clustered
    by similar title so one card = one story across many outlets.

    Falls back to legacy single-source RSS if the aggregator returns nothing
    (network blip, all sources down) — the UI never goes empty as long as
    Google News works.
    """
    # M2-M5 NEW SOURCE: poll the SourceRegistry's active RSS feeds. This is
    # the primary, broader path — 50+ publishers + cross-run dedup via
    # NewsDB. Falls back to the legacy aggregator on failure.
    registry_items: List[NewsItem] = []
    try:
        registry = _get_source_registry()
        active = registry.active_rss_sources()
        registry_items = await asyncio.to_thread(
            fetch_all_rss_sources, active, since_days=3, max_per_source=30,
            max_total=400,
        )
        # Cross-run dedup against DB: skip URLs we've already shown.
        db = _get_news_db()
        before = len(registry_items)
        registry_items = [i for i in registry_items if not db.is_known_url(i.url_hash)]
        logger.info(
            f"registry fetch: {len(registry_items)}/{before} fresh items "
            f"from {len(active)} sources (cross-run dedup removed {before - len(registry_items)})"
        )
    except Exception as e:
        logger.warning(f"registry fetch failed: {e}")

    # ALSO run the existing aggregator (Google News + clusters) — it
    # gives us multi-source clustering which the registry feeds don't.
    clusters: List[NewsCluster] = []
    try:
        clusters = await aggregate_news_clusters(
            QR_QUERY,
            extra_queries=["Cancún noticias"],
            intl_rss=False,
            reddit=False,
            timeout_total=10.0,
            jaccard_threshold=0.30,
        )
    except Exception as e:
        logger.warning(f"aggregator failed, falling back to RSS: {e}")

    weights = load_weights()
    items_for_scoring: List[NewsItem] = []
    cluster_by_url: Dict[str, NewsCluster] = {}

    # Merge: cluster items + registry items, deduped in-session.
    dedup = Deduplicator()
    if clusters:
        for c in clusters:
            primary = c.primary
            item = NewsItem(
                title=primary.title,
                url=primary.url,
                source=primary.outlet.split(":", 1)[-1] if ":" in primary.outlet else primary.outlet,
                published_at=primary.published_at or datetime.now().isoformat(),
                snippet=primary.snippet,
                body="",
                region_hits=[],
            )
            if dedup.is_new(item):
                items_for_scoring.append(item)
                cluster_by_url[item.url] = c

    for item in registry_items:
        if dedup.is_new(item):
            items_for_scoring.append(item)

    if not items_for_scoring:
        # Last-resort fallback: legacy single-source pipeline.
        legacy = await asyncio.to_thread(
            fetch_google_news, since_days=3, max_items=200, require_region_hit=False,
        )
        items_for_scoring = legacy

    logger.info(
        f"/refresh source mix: clusters={len(cluster_by_url)} "
        f"registry={len(registry_items)} total_pre_score={len(items_for_scoring)}"
    )

    scored = score_items(items_for_scoring, weights)

    # Source-count boost: stories covered by ≥2 outlets get a confidence bump.
    # +0.05 per extra source (capped at +0.20). Consensus = reliability.
    for s in scored:
        c = cluster_by_url.get(s.item.url)
        if c and c.source_count >= 2:
            boost = min(0.20, 0.05 * (c.source_count - 1))
            s.score = min(1.0, s.score + boost)
            s.breakdown["multi_source"] = round(boost, 3)

    # M1 hard filter: drop anything below 60/100 — these don't make the cut.
    MIN_SCORE = 60
    scored = [s for s in scored if int(round(s.score * 100)) >= MIN_SCORE]

    # Sort: score desc, then date desc (newest of the tied scores first).
    scored.sort(
        key=lambda x: (x.score, x.item.published_at or ""),
        reverse=True,
    )
    scored = scored[:15]

    def _tier(score_int: int) -> str:
        if score_int >= 85:
            return "featured"   # gold/red, top editorial
        if score_int >= 70:
            return "good"       # amber
        return "regular"        # grey

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def mut(state):
        state["current_run"] = run_id
        rows = []
        for s in scored:
            c = cluster_by_url.get(s.item.url)
            source_count = c.source_count if c else 1
            score_int = int(round(s.score * 100))
            rows.append({
                "url": s.item.url,
                "title": s.item.title,
                "source": s.item.source,
                "published_at": s.item.published_at,
                "snippet": s.item.snippet,
                "region_hits": s.item.region_hits,
                "score": s.score,                 # float, kept for back-compat
                "score_int": score_int,           # 0-100 integer for UI
                "tier": _tier(score_int),         # regular | good | featured
                "unconfirmed": source_count < 2,  # single-source exclusive
                "breakdown": s.breakdown,
                "source_count": source_count,
                "alternate_sources": [
                    {"url": m.url, "outlet": m.outlet, "title": m.title}
                    for m in (c.members[1:6] if c else [])
                ],
                "credibility": (round(c.credibility, 3) if c else None),
            })
        state["news_by_run"][run_id] = rows
    update_state(mut)

    # Persist the top N to NewsDB so future /refresh cross-run dedups
    # them out. We persist after MIN_SCORE filter — we don't want the DB
    # bloated with low-quality items, but we DO want shown items to be
    # remembered so refreshes don't repeat them.
    try:
        db = _get_news_db()
        records = [
            ArticleRecord(
                url=s.item.url, url_hash=s.item.url_hash,
                title=s.item.title, source=s.item.source,
                published_at=s.item.published_at,
                detected_at=s.item.detected_at or datetime.now().isoformat(),
                snippet=s.item.snippet or "",
                body=s.item.body or "",
                category="",
                source_region=getattr(s.item, "source_region", "") or "",
                author=getattr(s.item, "author", "") or "",
                image_url=getattr(s.item, "image_url", "") or "",
                score=float(s.score),
            )
            for s in scored
        ]
        landed = db.insert_articles(records)
        logger.info(f"/refresh persisted {landed} new articles to DB")
    except Exception as e:
        logger.warning(f"DB persist failed (non-fatal): {e}")

    crossed = sum(1 for s in scored
                  if cluster_by_url.get(s.item.url) and cluster_by_url[s.item.url].source_count >= 2)
    logger.info(
        f"/refresh: {len(scored)} stories after MIN_SCORE={MIN_SCORE} filter, "
        f"{crossed} cross-verified (≥2 sources)"
    )
    return RedirectResponse(url="/", status_code=303)


@app.get("/api/news-engine-stats")
async def news_engine_stats(_: None = Depends(require_auth)):
    """Quick stats from NewsDB — articles seen, sources, recent errors."""
    try:
        db = _get_news_db()
        return JSONResponse(db.stats())
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/decide")
async def decide(
    request: Request,
    background_tasks: BackgroundTasks,
    url: str = Form(...),
    accept: str = Form(...),  # "yes" | "no"
    _: None = Depends(require_auth),
):
    state = load_state()
    accepted = accept == "yes"
    item_data = None
    for ri in state.get("news_by_run", {}).get(state.get("current_run") or "", []):
        if ri["url"] == url:
            item_data = ri
            break
    if not item_data:
        raise HTTPException(404, "Item not found in current run")

    # Pre-flight: if the user is asking us to spend money/tokens, make sure
    # we have the keys to do so. Fail with a useful message instead of
    # queueing a job that's guaranteed to crash.
    if accepted:
        missing = [k for k in ("ANTHROPIC_API_KEY", "REPLICATE_API_TOKEN")
                   if not os.environ.get(k)]
        if missing:
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Falta configurar: {', '.join(missing)}. "
                    f"Agrega las keys en {ROOT}/.env (o en el .env del repo padre) "
                    f"y reinicia el server."
                ),
            )

    def mut(s):
        s["decisions"][url] = accepted

    if accepted:
        job_id = uuid.uuid4().hex[:10]
        # Snapshot the user's current settings so this specific job uses them
        # even if the user toggles them between dispatching and execution.
        current_settings = state.get("settings", {}) or {}
        item_with_opts = {**item_data, "_options": {
            "style": current_settings.get("style", _env_default_style()),
            "lipsync": current_settings.get("lipsync", _env_flag("WEBAPP_LIPSYNC")),
            "vertical": current_settings.get("vertical", _env_flag("WEBAPP_VERTICAL")),
            "narrative_mode": current_settings.get(
                "narrative_mode",
                os.environ.get("WEBAPP_NARRATIVE_MODE", DEFAULT_MODE),
            ),
            "num_scenes": current_settings.get(
                "num_scenes", int(os.environ.get("WEBAPP_NUM_SCENES", 3)),
            ),
            "target_duration": current_settings.get(
                "target_duration", int(os.environ.get("WEBAPP_TARGET_DURATION", 30)),
            ),
        }}

        def mut_with_job(s):
            mut(s)
            s["jobs"][job_id] = {
                "status": "queued",
                "news_url": item_data["url"],
                "title": item_data["title"],
                "source": item_data["source"],
                "video_path": None,
                "error": None,
                "started": None,
                "finished": None,
                "created": datetime.now().isoformat(),
                "options": item_with_opts["_options"],
            }
            s["job_order"].insert(0, job_id)
        update_state(mut_with_job)

        # Fire-and-forget background job. The asyncio lock inside ensures only
        # one heavy job runs at a time.
        background_tasks.add_task(run_video_pipeline, job_id, item_with_opts)
    else:
        update_state(mut)

    # Learn from this decision.
    nw = update_weights_from_feedback(
        load_weights(),
        [(NewsItem(
            title=item_data["title"], url=item_data["url"], source=item_data["source"],
            published_at=item_data["published_at"], snippet=item_data["snippet"], body="",
            region_hits=item_data["region_hits"],
        ), accepted)],
    )
    save_weights(nw)

    return RedirectResponse(url="/", status_code=303)


@app.get("/queue", response_class=HTMLResponse)
async def queue_view(request: Request, _: None = Depends(require_auth)):
    state = load_state()
    jobs = []
    for jid in state.get("job_order", []):
        job = state["jobs"].get(jid)
        if job:
            jobs.append({"id": jid, **job})
    return templates.TemplateResponse("queue.html", {"request": request, "jobs": jobs})


@app.get("/api/queue")
async def queue_json(_: None = Depends(require_auth)):
    """Polled by the queue view for live updates."""
    state = load_state()
    out = []
    for jid in state.get("job_order", []):
        job = state["jobs"].get(jid)
        if job:
            out.append({"id": jid, **job})
    return JSONResponse(out)


@app.get("/videos", response_class=HTMLResponse)
async def videos_view(request: Request):
    state = load_state()
    items = []
    for jid in state.get("job_order", []):
        job = state["jobs"].get(jid)
        if job and job["status"] == "done" and job.get("video_path"):
            items.append({"id": jid, **job})
    return templates.TemplateResponse("videos.html", {"request": request, "items": items})


@app.get("/video/{job_id}")
async def video_file(job_id: str):
    state = load_state()
    job = state.get("jobs", {}).get(job_id)
    if not job or job["status"] != "done" or not job.get("video_path"):
        raise HTTPException(404)
    path = Path(job["video_path"])
    if not path.exists():
        raise HTTPException(410, "Video file missing")
    return FileResponse(str(path), media_type="video/mp4")


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/admin/spend", response_class=HTMLResponse)
async def admin_spend(request: Request, _: None = Depends(require_auth)):
    """Last-14-day Replicate spend, aggregated per day.

    No CSS bells — this is for the operator, not the public. The webapp's
    main / view already gives the polished mobile UI."""
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
