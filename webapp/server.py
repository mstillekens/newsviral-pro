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

from news_sources import NewsItem, fetch_google_news, enrich_with_body
from news_scorer import ScoredItem, load_weights, save_weights, score_items, update_weights_from_feedback
from news_image_finder import find_reference_image
from script_writer import ScriptWriter
from replicate_orchestrator import ReplicateConfig, ReplicateOrchestrator
from video_compositor import BrandingConfig, VideoCompositor
from run_logger import RunLogger
from brand_style import STYLE_VARIANTS, ANCHORS, anchor_for


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("webapp")


# ---------- Env ----------

def _load_env(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip()

_load_env()


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


async def run_video_pipeline(job_id: str, item_dict: Dict[str, Any]) -> None:
    """Run script → Replicate → compose for one news item."""
    item = NewsItem(**{k: v for k, v in item_dict.items() if k in NewsItem.__dataclass_fields__})

    def mark_started(s):
        s["jobs"][job_id]["status"] = "running"
        s["jobs"][job_id]["started"] = datetime.now().isoformat()
    update_state(mark_started)

    async with GEN_LOCK:
        try:
            anchor = anchor_for(f"{item.title}\n{item.snippet}\n{item.body}")
            writer = ScriptWriter(model="claude-haiku-4-5", style="caricature")
            script = await asyncio.to_thread(writer.write, item, anchor)

            prompts = dict(script.scenes)
            # ref-image enrichment for scene 2
            if "escena_2" in prompts and item.url:
                ref = await asyncio.to_thread(find_reference_image, item.url)
                if ref:
                    prompts["escena_2"]["reference_image_url"] = ref.url

            vid = os.environ.get("MINIMAX_VOICE_ID", "")
            if vid:
                prompts["voice_params"] = {"voice_id": vid, "language_boost": "Spanish"}

            orch = ReplicateOrchestrator(ReplicateConfig(
                api_token=os.environ["REPLICATE_API_TOKEN"],
                enable_video=True,
            ))
            elementos = await orch.orchestrate_parallel(prompts)
            if not await orch.validate_outputs(elementos):
                raise RuntimeError("validate_outputs failed")

            compositor = VideoCompositor(
                BrandingConfig(colors={"primary": "#235B4E", "accent": "#9F2241", "bg": "#000000"}),
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
                s["jobs"][job_id]["video_path"] = str(archived)
                s["jobs"][job_id]["anchor"] = anchor.name
                s["jobs"][job_id]["finished"] = datetime.now().isoformat()
            update_state(mark_done)

        except Exception as e:
            logger.exception(f"Job {job_id} failed")
            def mark_fail(s):
                s["jobs"][job_id]["status"] = "failed"
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
    return templates.TemplateResponse("index.html", {
        "request": request,
        "pending": pending,
        "total": len(items),
        "decided": len(decisions),
        "queue_size": sum(1 for j in state.get("jobs", {}).values() if j["status"] in ("queued", "running")),
        "videos_ready": sum(1 for j in state.get("jobs", {}).values() if j["status"] == "done"),
    })


@app.post("/refresh", response_class=RedirectResponse)
async def refresh(_: None = Depends(require_auth)):
    """Refetch RSS + score, replace current_run."""
    items = await asyncio.to_thread(
        fetch_google_news, since_days=2, max_items=30, require_region_hit=True
    )
    weights = load_weights()
    scored = score_items(items, weights)[:15]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")

    def mut(state):
        state["current_run"] = run_id
        state["news_by_run"][run_id] = [
            {
                "url": s.item.url,
                "title": s.item.title,
                "source": s.item.source,
                "published_at": s.item.published_at,
                "snippet": s.item.snippet,
                "region_hits": s.item.region_hits,
                "score": s.score,
                "breakdown": s.breakdown,
            }
            for s in scored
        ]
    update_state(mut)
    return RedirectResponse(url="/", status_code=303)


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

    def mut(s):
        s["decisions"][url] = accepted

    if accepted:
        job_id = uuid.uuid4().hex[:10]

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
            }
            s["job_order"].insert(0, job_id)
        update_state(mut_with_job)

        # Fire-and-forget background job. The asyncio lock inside ensures only
        # one heavy job runs at a time.
        background_tasks.add_task(run_video_pipeline, job_id, item_data)
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
