"""NewsViral PRO — Voz del Pueblo. Async pipeline orchestrator.

Demo flow:
  1. tarea_1_research → fetch QR/Cancún/RM news from Google News RSS,
     score by virality heuristic.
  2. CLI selection prompt → user accepts/rejects each ranked item.
  3. tarea_2_script → for each accepted item, generate a 3-scene first-person
     Spanish script via Claude (with persona variation).
  4. tarea_3_prompts → (already done by script_writer; this just adapts the
     shape).
  5. tarea_4_replicate_pro → parallel FLUX + Minimax.
  6. tarea_5_componer_video_pro → FFmpeg compose.
  7. Update score weights from Y/N decisions for the next run.

Everything is logged under logs/runs/<timestamp>/.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from web_dashboard import ProgressTracker
from replicate_orchestrator import ReplicateOrchestrator, ReplicateConfig
from video_compositor import VideoCompositor, BrandingConfig
from news_sources import NewsItem, fetch_google_news, filter_by_date, enrich_with_body
from news_image_finder import find_reference_image
from news_scorer import ScoredItem, load_weights, save_weights, score_items, update_weights_from_feedback
from script_writer import Script, ScriptWriter
from run_logger import RunLogger
from brand_style import STYLE_VARIANTS, anchor_for, pick_voice_id_for


# Lazy-load the anchor portrait manifest. The orchestrator can skip FLUX
# for scenes that show the anchor when we have a cached portrait — gives
# us brand-consistent characters across every video.
_ANCHOR_MANIFEST: Optional[Dict[str, Dict[str, str]]] = None


def _load_anchor_manifest() -> Dict[str, Dict[str, str]]:
    global _ANCHOR_MANIFEST
    if _ANCHOR_MANIFEST is None:
        path = Path("anchor_portraits/manifest.json")
        if path.exists():
            try:
                _ANCHOR_MANIFEST = json.loads(path.read_text())
            except Exception:
                _ANCHOR_MANIFEST = {}
        else:
            _ANCHOR_MANIFEST = {}
    return _ANCHOR_MANIFEST


def _inject_anchor_portraits(prompts: Dict, anchor_id: str) -> None:
    """Attach the cached portrait URL to scenes 1 and 3 (the anchor scenes).

    Convention from script_writer: escena_1 = anchor hook, escena_2 = event,
    escena_3 = anchor close. We tag 1 and 3 with the cached portrait so the
    orchestrator skips FLUX for them entirely. Scene 2 stays untouched so it
    can use FLUX text-to-image (or canny+ref-image when scraped).
    """
    manifest = _load_anchor_manifest()
    entry = manifest.get(anchor_id)
    if not entry:
        logger.warning(f"⚠️  No cached portrait for anchor {anchor_id} — run setup_anchors.py")
        return
    url = entry.get("url")
    if not url:
        return
    for key in ("escena_1", "escena_3"):
        if key in prompts and isinstance(prompts[key], dict):
            prompts[key]["anchor_portrait_url"] = url


# ---------- Logger ----------

class _ViralLogger:
    """Thin wrapper over stdlib logging that adds .success() and a tiempo= kwarg."""

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self._log = logging.getLogger("news_viral_pro")

    def _fmt(self, msg: str, tiempo: bool) -> str:
        return f"[{time.strftime('%H:%M:%S')}] {msg}" if tiempo else msg

    def info(self, msg: str, tiempo: bool = False) -> None:
        self._log.info(self._fmt(msg, tiempo))

    def success(self, msg: str) -> None:
        self._log.info(f"✅ {msg}")

    def warning(self, msg: str) -> None:
        self._log.warning(msg)

    def error(self, msg: str) -> None:
        self._log.error(msg)


logger = _ViralLogger()


# ---------- Config ----------

@dataclass
class ConfigPro:
    """Runtime configuration for the NewsViral PRO pipeline."""
    replicate_api_token: str = ""
    anthropic_api_key: str = ""
    colores_morena: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#235B4E",
        "accent": "#9F2241",
        "bg": "#000000",
    })
    scene_count: int = 3
    voice: str = "adam"
    # Demo controls
    since_days: int = 2
    max_items: int = 12
    date_filter: Optional[str] = None       # "YYYY-MM-DD" UTC day filter
    auto_accept_top: int = 0                # >0: skip CLI, accept top-N automatically
    script_model: str = "claude-haiku-4-5"
    enable_video: bool = True               # True = Seedance image→video; False = stills only
    style: str = "documentary"              # one of STYLE_VARIANTS keys


# ---------- Tareas 1-3: news → script ----------

async def tarea_1_research(config: ConfigPro, run_log: RunLogger) -> List[ScoredItem]:
    """Fetch + score news. Synchronous under the hood; wrapped async to keep
    the orchestrator shape consistent."""
    logger.info("🔎 TAREA 1: Research (noticias QR)", tiempo=True)

    items = await asyncio.to_thread(
        fetch_google_news,
        since_days=config.since_days,
        max_items=max(config.max_items * 3, 30),
        require_region_hit=True,
    )

    if config.date_filter:
        items = filter_by_date(items, config.date_filter)
        logger.info(f"🗓  Filtered to date {config.date_filter}: {len(items)} items")

    run_log.fetched(items)

    weights = load_weights()
    scored = score_items(items, weights)
    scored = scored[: config.max_items]
    run_log.scored(scored)

    logger.success(f"{len(scored)} noticias ranqueadas")
    return scored


def _format_card(idx: int, total: int, scored: ScoredItem) -> str:
    item = scored.item
    bd = scored.breakdown
    when = item.published_at[:16].replace("T", " ")
    title = item.title[:140]
    snippet = (item.snippet or "")[:200]
    region = ", ".join(item.region_hits) or "—"
    breakdown_str = " ".join(f"{k}={v:+.2f}" for k, v in bd.items())
    return (
        f"\n┌─ [{idx}/{total}] score {scored.score:.2f} ─────────────────────────\n"
        f"│ {title}\n"
        f"│ {item.source} · {when} · {region}\n"
        f"│ {snippet}\n"
        f"│ {breakdown_str}\n"
        f"│ {item.url}\n"
        f"└─ [S]í   [N]o   [Q]uit"
    )


def select_news_cli(scored_items: List[ScoredItem], config: ConfigPro) -> List[Tuple[NewsItem, bool]]:
    """Present each scored item, collect Y/N decisions. Returns the FULL list
    of (item, accepted) decisions — accepted=False included — so the scorer
    can learn from rejections too."""
    if config.auto_accept_top > 0:
        decisions: List[Tuple[NewsItem, bool]] = []
        for i, s in enumerate(scored_items):
            decisions.append((s.item, i < config.auto_accept_top))
        logger.info(f"🤖 auto_accept_top={config.auto_accept_top}: skipping CLI prompt")
        return decisions

    total = len(scored_items)
    decisions = []
    for idx, s in enumerate(scored_items, start=1):
        print(_format_card(idx, total, s))
        while True:
            try:
                ans = input("→ ").strip().lower()
            except EOFError:
                ans = "q"
            if ans in ("s", "y", "yes", "si", "sí"):
                decisions.append((s.item, True))
                break
            if ans in ("n", "no"):
                decisions.append((s.item, False))
                break
            if ans in ("q", "quit"):
                for remaining in scored_items[idx-1:]:
                    decisions.append((remaining.item, False))
                return decisions
            print("  (responde s/n/q)")
    return decisions


async def tarea_2_3_scripts_and_prompts(
    accepted: List[NewsItem],
    config: ConfigPro,
    run_log: RunLogger,
) -> List[Script]:
    """TAREAS 2+3 merged: enrich each accepted item with its full body (Sipse
    scraper) when possible, then call Claude to write the 3-scene script.
    The script already contains imagen_prompt + audio_script per scene, so
    there's no separate tarea_3."""
    if not accepted:
        return []

    logger.info("✍️  TAREA 2-3: Scripts + Prompts (Claude)", tiempo=True)

    # Scrape full bodies in parallel.
    enriched = await asyncio.gather(*[
        asyncio.to_thread(enrich_with_body, item) for item in accepted
    ])

    writer = ScriptWriter(
        api_key=config.anthropic_api_key,
        model=config.script_model,
        style=config.style,
    )

    scripts: List[Script] = []
    for item in enriched:
        try:
            script = await asyncio.to_thread(writer.write, item)
            scripts.append(script)
        except Exception as e:
            logger.error(f"Script failed for «{item.title[:60]}»: {e}")

    run_log.scripts(scripts)
    logger.success(f"{len(scripts)} guiones generados")
    return scripts


# ---------- Tareas 4-5 (unchanged from Phase 2) ----------

async def tarea_4_replicate_pro(
    prompts: Dict[str, Any],
    config: ConfigPro,
    skip_replicate: bool = False,
) -> Optional[Dict[str, Any]]:
    """TAREA 4: Orchestrate Replicate (parallel execution)"""
    logger.info("🎥 TAREA 4: Replicate Orchestration (PARALELO)", tiempo=True)

    try:
        replicate_config = ReplicateConfig(
            api_token=config.replicate_api_token or os.getenv("REPLICATE_API_TOKEN", ""),
            skip_replicate=skip_replicate,
            enable_video=config.enable_video,
        )
        orchestrator = ReplicateOrchestrator(replicate_config)
        elementos = await orchestrator.orchestrate_parallel(prompts, config)
        is_valid = await orchestrator.validate_outputs(elementos)
        if is_valid:
            logger.success("Replicate orchestration complete")
            return elementos
        logger.error("Validation failed")
        return None
    except Exception as e:
        logger.error(f"Error Tarea 4: {str(e)}")
        return None


async def tarea_5_componer_video_pro(
    elementos: Dict[str, Any],
    config: ConfigPro,
    news_title: str = "",
    news_source: str = "",
    anchor=None,
) -> Optional[Dict[str, Any]]:
    """TAREA 5: Compose final video with FFmpeg, with newsroom branding."""
    logger.info("🎬 TAREA 5: Composición Video Final", tiempo=True)

    try:
        branding = BrandingConfig(colors=config.colores_morena)
        compositor = VideoCompositor(
            branding, news_title=news_title, news_source=news_source, anchor=anchor
        )
        composed_video = compositor.compose_with_audio(elementos)
        resultado = compositor.export_mp4(composed_video)
        logger.success(f"Video final: {resultado['video_path']}")
        return resultado
    except Exception as e:
        logger.error(f"Error Tarea 5: {str(e)}")
        return None


# ---------- Per-news video production ----------

async def produce_video_for_script(
    idx: int,
    script: Script,
    news_title: str,
    config: ConfigPro,
    run_log: RunLogger,
    mock: bool,
    news_source: str = "",
) -> Optional[Dict[str, Any]]:
    """Run TAREA 4 + 5 for one news item's script, copy the resulting mp4 into
    the per-run log dir, and return the result metadata."""
    logger.info(f"━━━ Video #{idx}: «{news_title[:70]}» (persona={script.persona_id}) ━━━")

    prompts = script.to_prompts_dict()

    # Anchor portrait injection. Scenes 1 and 3 always feature the anchor;
    # using the cached portrait keeps the character visually identical
    # across every video and skips a FLUX call per scene.
    _inject_anchor_portraits(prompts, script.anchor_id)

    # Reference image enrichment for scene 2 (the "event" scene per our
    # template). If we can scrape an og:image from the news URL we pass it
    # as control_image to flux-canny-pro, which preserves the source's
    # composition while applying the style prompt — so caricatures look
    # like the actual subjects.
    # We deliberately attach the ref image only to scene 2; scenes 1 and 3
    # show the anchor and don't benefit from outside imagery.
    if "escena_2" in prompts and script.news_url:
        ref = await asyncio.to_thread(find_reference_image, script.news_url)
        if ref:
            prompts["escena_2"]["reference_image_url"] = ref.url
            logger.info(f"📸 ref-image escena_2: {ref.source} → {ref.url[:80]}")

    # Pick a gender-matched MiniMax voice for this anchor. Priority chain:
    # per-anchor env var → global env var → anchor's hard-coded default.
    # See pick_voice_id_for() in brand_style.py for details.
    anchor_obj = anchor_for(f"{news_title}\n{news_source}")
    voice_id = pick_voice_id_for(anchor_obj, os.environ)
    prompts = {**prompts, "voice_params": {"voice_id": voice_id, "language_boost": "Spanish"}}
    logger.info(f"🎙  Voz para {anchor_obj.id} ({anchor_obj.gender}): {voice_id}")

    elementos = await tarea_4_replicate_pro(prompts, config, skip_replicate=mock)
    if elementos is None:
        return None

    # The anchor is derived from the news item itself, so the intro/outro
    # iris cards pick up the right signature lines automatically.
    derived_anchor = anchor_for(f"{news_title}\n{news_source}")
    resultado = await tarea_5_componer_video_pro(
        elementos, config, news_title=news_title, news_source=news_source,
        anchor=derived_anchor,
    )
    if resultado is None:
        return None

    src_video = Path(resultado["video_path"])
    archived = run_log.video(idx, news_title, resultado, src_video)
    resultado["archived_path"] = str(archived)
    return resultado


# ---------- Entrypoint ----------

async def run(config: ConfigPro, mock: bool) -> int:
    run_log = RunLogger()
    tracker = ProgressTracker()

    # 1. Research + score
    tracker.start("research")
    scored = await tarea_1_research(config, run_log)
    tracker.done("research", message=f"{len(scored)} items")

    if not scored:
        logger.error("No hay noticias después del filtro. Revisa --date o --since-days.")
        return 1

    # 2. CLI selection
    tracker.start("select")
    decisions = select_news_cli(scored, config)
    accepted_items = [item for item, ok in decisions if ok]
    run_log.selected([
        {"url": it.url, "title": it.title, "accepted": ok}
        for it, ok in decisions
    ])
    tracker.done("select", message=f"{len(accepted_items)}/{len(decisions)} aceptadas")
    logger.info(f"📥 {len(accepted_items)} noticias seleccionadas para video")

    # 3. Update weights from feedback (do this BEFORE generating videos so
    #    even if the rest fails, learning persists).
    new_weights = update_weights_from_feedback(load_weights(), decisions)
    save_weights(new_weights)
    logger.info("🧠 Pesos del scorer actualizados (score_weights.json)")

    if not accepted_items:
        run_log.summary({
            "timestamp": run_log.timestamp,
            "fetched": len(scored),
            "accepted": 0,
            "videos_produced": 0,
        })
        logger.info("Ninguna noticia aceptada. Nada que generar.")
        return 0

    # 4. Scripts + prompts via Claude
    tracker.start("scripts")
    scripts = await tarea_2_3_scripts_and_prompts(accepted_items, config, run_log)
    tracker.done("scripts", message=f"{len(scripts)} scripts")

    # 5. Per-script: Replicate + FFmpeg
    tracker.start("videos")
    produced: List[Dict[str, Any]] = []
    for idx, (item, script) in enumerate(zip(accepted_items[: len(scripts)], scripts), start=1):
        result = await produce_video_for_script(
            idx, script, item.title, config, run_log, mock, news_source=item.source
        )
        if result:
            produced.append(result)
    tracker.done("videos", message=f"{len(produced)} videos")

    # 6. Summary
    run_log.summary({
        "timestamp": run_log.timestamp,
        "fetched": len(scored),
        "accepted": len(accepted_items),
        "scripts_generated": len(scripts),
        "videos_produced": len(produced),
        "videos": [r.get("archived_path") or r.get("video_path") for r in produced],
        "mock": mock,
        "script_model": config.script_model,
    })

    logger.success(
        f"Demo terminó. {len(produced)} videos en {run_log.dir}/"
    )
    return 0 if produced or not accepted_items else 1


def _load_env_file(path: Path = Path(".env")) -> None:
    """Load KEY=VALUE pairs from .env. Overrides empty environment values
    (Claude Code and some shells export ANTHROPIC_API_KEY="") but does NOT
    override a meaningfully-set env var, so callers can still pass a key
    inline."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if not os.environ.get(k):
            os.environ[k] = v


def main() -> None:
    _load_env_file()

    p = argparse.ArgumentParser(description="NewsViral PRO demo runner")
    p.add_argument("--mock", action="store_true", help="Skip real Replicate calls (no media generated)")
    p.add_argument("--date", default=None, help="Only consider news from this UTC date (YYYY-MM-DD)")
    p.add_argument("--since-days", type=int, default=2, help="How many days back to fetch")
    p.add_argument("--max", type=int, default=12, help="Max items to score & show")
    p.add_argument("--auto", type=int, default=0, help="Skip CLI prompt, auto-accept the top N scored items")
    p.add_argument("--model", default="claude-haiku-4-5", help="Anthropic model for script writing")
    p.add_argument("--no-video", action="store_true",
                   help="Skip Seedance step (use still images only — ~9x cheaper, lower quality)")
    p.add_argument("--style", default="documentary",
                   choices=list(STYLE_VARIANTS.keys()),
                   help="Visual style variant for FLUX + Seedance prompts")
    args = p.parse_args()

    config = ConfigPro(
        replicate_api_token=os.environ.get("REPLICATE_API_TOKEN", ""),
        anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
        since_days=args.since_days,
        max_items=args.max,
        date_filter=args.date,
        auto_accept_top=args.auto,
        script_model=args.model,
        enable_video=not args.no_video,
        style=args.style,
    )

    if not config.anthropic_api_key and not args.mock:
        print("ERROR: ANTHROPIC_API_KEY no configurado. Pon uno en .env o usa --mock.", file=sys.stderr)
        raise SystemExit(2)

    raise SystemExit(asyncio.run(run(config, mock=args.mock)))


if __name__ == "__main__":
    main()
