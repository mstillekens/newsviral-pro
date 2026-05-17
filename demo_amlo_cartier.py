#!/usr/bin/env python3
"""One-off demo: nota Andy López Beltrán en Cartier (Plaza La Isla, Cancún).

Usa la nueva arquitectura de Phase 7 (anchor por vertical + estilo
configurable + intriga > drama + audio sin cortes + intro/outro Looney
Tunes iris). Style override desde CLI para iterar sobre estética:

    python demo_amlo_cartier.py                       # documentary
    python demo_amlo_cartier.py --style caricature    # cartón mexicano
    python demo_amlo_cartier.py --style comic_book    # cómic 60s
    python demo_amlo_cartier.py --style retro_noir    # noir 1940s
    python demo_amlo_cartier.py --style loonytunes    # Looney Tunes

El ancla se elige automáticamente por el clasificador: esta nota cae en
"política", así que aparece Don Polibruh.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from anthropic import Anthropic   # noqa: F401   (only here so a missing dep fails fast)

from news_sources import NewsItem
from script_writer import ScriptWriter
from replicate_orchestrator import ReplicateConfig, ReplicateOrchestrator
from video_compositor import BrandingConfig, VideoCompositor
from run_logger import RunLogger
from brand_style import STYLE_VARIANTS, anchor_for, pick_voice_id_for, ANCHORS

import json as _json


def _inject_portrait_from_manifest(prompts: dict, anchor_id: str) -> None:
    """Attach the cached anchor portrait URL to scenes 1 and 3. Same
    convention as news_viral_pro._inject_anchor_portraits — duplicated here
    so the standalone demo doesn't import the main orchestrator."""
    manifest_path = Path("anchor_portraits/manifest.json")
    if not manifest_path.exists():
        logger.warning("⚠️  No hay manifest.json — corre 'python setup_anchors.py' primero")
        return
    try:
        manifest = _json.loads(manifest_path.read_text())
    except Exception:
        logger.warning("⚠️  manifest.json corrupto, ignorando")
        return
    entry = manifest.get(anchor_id)
    if not entry or not entry.get("url"):
        logger.warning(f"⚠️  No hay portrait cacheado para {anchor_id}")
        return
    url = entry["url"]
    for key in ("escena_1", "escena_3"):
        if key in prompts and isinstance(prompts[key], dict):
            prompts[key]["anchor_portrait_url"] = url
    logger.info(f"🎭 Portrait inyectado: {anchor_id} (escenas 1 y 3)")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("demo_amlo_cartier")


NEWS = NewsItem(
    title="Andy López Beltrán es captado en la tienda Cartier de Plaza La Isla, Cancún",
    url="https://demo.newsviral.local/andy-cartier-cancun",
    source="Reporte ciudadano · redes sociales",
    published_at="2026-05-15T12:00:00+00:00",
    snippet=(
        "Andrés Manuel López Beltrán, hijo del expresidente y dirigente nacional de Morena, "
        "fue grabado en video y fotografiado por usuarios de redes mientras visitaba la tienda "
        "Cartier de Plaza La Isla en Cancún. Las imágenes generaron debate por el contraste "
        "entre el discurso oficialista de austeridad y el consumo de lujo del funcionario."
    ),
    body=(
        "Reportes en redes sociales registraron la presencia de Andy López Beltrán, hijo del "
        "expresidente Andrés Manuel López Obrador y actual coordinador de Organización del "
        "partido Morena, en la tienda de la marca francesa de joyería Cartier en Plaza La Isla, "
        "Cancún. La situación detonó comentarios en plataformas como X y TikTok por el contraste "
        "entre el discurso público de austeridad republicana defendido por su partido y la "
        "naturaleza del lugar visitado. No se ha confirmado oficialmente si realizó alguna "
        "compra, ni el monto si la hubo. Cartier es una de las casas de lujo más caras del "
        "mercado mundial, con piezas que van desde decenas de miles a millones de pesos."
    ),
    region_hits=["Cancún", "Quintana Roo"],
)


def _load_env(path: Path = Path(".env")) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        if not os.environ.get(k.strip()):
            os.environ[k.strip()] = v.strip()


async def main() -> int:
    _load_env()

    parser = argparse.ArgumentParser()
    parser.add_argument("--style", default="caricature",
                        choices=list(STYLE_VARIANTS.keys()),
                        help="Visual style for FLUX + Seedance")
    parser.add_argument("--anchor",
                        default=None,
                        choices=[a.id for a in ANCHORS],
                        help="Force a specific anchor (default = classifier picks)")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ERROR: ANTHROPIC_API_KEY missing", file=sys.stderr)
        return 1
    if not os.environ.get("REPLICATE_API_TOKEN") and not args.mock:
        print("ERROR: REPLICATE_API_TOKEN missing", file=sys.stderr)
        return 1

    run_log = RunLogger()
    logger.info(f"📒 Run dir: {run_log.dir}")
    run_log.fetched([NEWS])
    run_log.selected([{"url": NEWS.url, "title": NEWS.title, "accepted": True}])

    # 1. Pick anchor (auto by vertical, unless --anchor forces one).
    if args.anchor:
        anchor = next(a for a in ANCHORS if a.id == args.anchor)
    else:
        anchor = anchor_for(f"{NEWS.title}\n{NEWS.snippet}\n{NEWS.body}")
    logger.info(f"🎤 Anchor: {anchor.name} (id={anchor.id}) · style={args.style}")

    # 2. Generate script with the anchor + style.
    writer = ScriptWriter(model="claude-haiku-4-5", style=args.style)
    script = writer.write(NEWS, anchor=anchor)
    logger.info("📜 Guion:")
    for k, v in script.scenes.items():
        em = v.get("emotion", "?")
        line = v.get("audio_script", "")
        logger.info(f"  {k} ({em}):  {line[:100]}")

    run_log.scripts([{
        "news_url": NEWS.url,
        "persona_id": script.persona_id,
        "anchor_id": script.anchor_id,
        "anchor_name": script.anchor_name,
        "style": script.style,
        "scenes": script.scenes,
        "model": script.model,
    }])

    # 3. Replicate (with optional cloned voice from .env).
    prompts = dict(script.scenes)

    # Inject cached anchor portrait into scenes 1 and 3 so the character
    # stays visually consistent across runs (and we skip 2 FLUX calls).
    _inject_portrait_from_manifest(prompts, anchor.id)

    voice_id = pick_voice_id_for(anchor, dict(os.environ))
    prompts["voice_params"] = {"voice_id": voice_id, "language_boost": "Spanish"}
    logger.info(f"🎙  Voz para {anchor.id} ({anchor.gender}): {voice_id}")

    orch = ReplicateOrchestrator(ReplicateConfig(
        api_token=os.environ["REPLICATE_API_TOKEN"],
        skip_replicate=args.mock,
        enable_video=True,
    ))
    elementos = await orch.orchestrate_parallel(prompts)
    if not await orch.validate_outputs(elementos):
        logger.error("validation failed")
        return 1

    # 4. Compose with branding (anchor passed → custom intro/outro lines).
    compositor = VideoCompositor(
        BrandingConfig(colors={"primary": "#235B4E", "accent": "#9F2241", "bg": "#000000"}),
        news_title=NEWS.title,
        news_source=NEWS.source,
        anchor=anchor,
    )
    composed = compositor.compose_with_audio(elementos)
    result = compositor.export_mp4(composed)
    src_video = Path(result["video_path"])
    archived = run_log.video(1, NEWS.title, result, src_video)
    result["archived_path"] = str(archived)

    run_log.summary({
        "timestamp": run_log.timestamp,
        "demo": "amlo_cartier",
        "anchor_id": anchor.id,
        "anchor_name": anchor.name,
        "style": args.style,
        "video": str(archived),
        "duration": result.get("duration"),
        "file_size_mb": result.get("file_size_mb"),
    })

    logger.info("=" * 70)
    logger.info(f"✅ Demo lista: {archived}")
    logger.info(f"   open {archived}")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
